from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

from rag import chatbot, commands


def quietly(func: Any, *args: Any, **kwargs: Any) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


@dataclass
class FakeCollection:
    metadata: dict[str, Any] | None = None

    def modify(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


class TestTextHelpers(unittest.TestCase):
    def test_tokenize_lowercases_and_drops_punctuation(self):
        self.assertEqual(chatbot.tokenize("Hello, World!"), ["hello", "world"])

    def test_tokenize_keeps_vietnamese_diacritics(self):
        self.assertEqual(
            chatbot.tokenize("Chủ nghĩa xã hội"), ["chủ", "nghĩa", "xã", "hội"]
        )

    def test_sanitize_preserves_vietnamese(self):
        self.assertEqual(chatbot.sanitize("biện chứng"), "biện chứng")

    def test_sanitize_strips_lone_surrogates(self):
        self.assertEqual(chatbot.sanitize("a\ud800b"), "ab")

    def test_format_pages(self):
        self.assertEqual(chatbot.format_pages([3, 4]), "3, 4")
        self.assertEqual(chatbot.format_pages([]), "")


class TestFindPages(unittest.TestCase):
    full_text = "alpha" + chatbot.PAGE_SEPARATOR + "beta" + chatbot.PAGE_SEPARATOR
    offsets = [(0, 5, [1]), (7, 11, [2])]

    def test_chunk_within_one_block(self):
        self.assertEqual(chatbot.find_pages("alpha", self.full_text, self.offsets), [1])
        self.assertEqual(chatbot.find_pages("beta", self.full_text, self.offsets), [2])

    def test_chunk_spanning_two_blocks_cites_both(self):
        chunk = "alpha" + chatbot.PAGE_SEPARATOR + "beta"
        self.assertEqual(chatbot.find_pages(chunk, self.full_text, self.offsets), [1, 2])

    def test_absent_chunk_yields_no_pages(self):
        self.assertEqual(chatbot.find_pages("gamma", self.full_text, self.offsets), [])

    def test_block_covering_several_pages(self):
        offsets = [(0, 5, [3, 4])]
        self.assertEqual(chatbot.find_pages("alpha", "alpha", offsets), [3, 4])

    def test_separator_between_blocks_belongs_to_neither(self):
        self.assertEqual(
            chatbot.find_pages(chatbot.PAGE_SEPARATOR, self.full_text, self.offsets), []
        )


class TestReciprocalRankFusion(unittest.TestCase):
    def test_single_ranking_preserves_order(self):
        scores = chatbot.reciprocal_rank_fusion([["a", "b", "c"]])
        ranked = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
        self.assertEqual(ranked, ["a", "b", "c"])

    def test_agreement_between_rankings_wins(self):
        scores = chatbot.reciprocal_rank_fusion([["a", "b"], ["c", "b"]])
        self.assertEqual(max(scores, key=lambda doc_id: scores[doc_id]), "b")

    def test_scores_accumulate_across_rankings(self):
        scores = chatbot.reciprocal_rank_fusion([["a"], ["a"]], k=60)
        self.assertAlmostEqual(scores["a"], 2 / 61)

    def test_empty_input(self):
        self.assertEqual(chatbot.reciprocal_rank_fusion([[], []]), {})


class TestFileHash(unittest.TestCase):
    def test_hash_is_namespaced_by_extractor_version(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"pdf bytes")
            path = handle.name
        self.addCleanup(os.unlink, path)
        self.assertTrue(chatbot.file_hash(path).startswith(f"{chatbot.EXTRACTOR_VERSION}:"))

    def test_hash_changes_with_content(self):
        paths = []
        for payload in (b"one", b"two"):
            with tempfile.NamedTemporaryFile(delete=False) as handle:
                handle.write(payload)
                paths.append(handle.name)
            self.addCleanup(os.unlink, paths[-1])
        self.assertNotEqual(chatbot.file_hash(paths[0]), chatbot.file_hash(paths[1]))


class TestIndexMarkers(unittest.TestCase):
    def test_absent_metadata_reads_as_empty(self):
        self.assertEqual(chatbot.indexed_hashes(FakeCollection()), {})

    def test_mark_then_read_round_trip(self):
        collection = FakeCollection()
        chatbot.mark_indexed(collection, "a.pdf", "v1:abc")
        self.assertEqual(chatbot.indexed_hashes(collection), {"a.pdf": "v1:abc"})

    def test_unrelated_collection_metadata_is_ignored(self):
        collection = FakeCollection(metadata={"hnsw:space": "cosine"})
        chatbot.mark_indexed(collection, "a.pdf", "v1:abc")
        self.assertEqual(chatbot.indexed_hashes(collection), {"a.pdf": "v1:abc"})
        self.assertEqual((collection.metadata or {})["hnsw:space"], "cosine")

    def test_unmark_removes_only_its_own_entry(self):
        collection = FakeCollection()
        chatbot.mark_indexed(collection, "a.pdf", "v1:abc")
        chatbot.mark_indexed(collection, "b.pdf", "v1:def")
        chatbot.unmark_indexed(collection, "a.pdf")
        self.assertEqual(chatbot.indexed_hashes(collection), {"b.pdf": "v1:def"})

    def test_unmark_missing_entry_is_a_no_op(self):
        collection = FakeCollection(metadata={"other": "kept"})
        chatbot.unmark_indexed(collection, "absent.pdf")
        self.assertEqual(collection.metadata, {"other": "kept"})

    def test_removing_the_last_marker_leaves_metadata_non_empty(self):
        collection = FakeCollection()
        chatbot.mark_indexed(collection, "a.pdf", "v1:abc")
        chatbot.unmark_indexed(collection, "a.pdf")
        self.assertEqual(chatbot.indexed_hashes(collection), {})
        self.assertTrue(collection.metadata)


class TestSessionSetters(unittest.TestCase):
    session: commands.Session

    def setUp(self):
        self.session = commands.Session()

    def test_top_n_accepts_positive_integer(self):
        quietly(commands.set_top_n, self.session, "7")
        self.assertEqual(self.session.top_n, 7)

    def test_top_n_rejects_zero_and_non_numeric(self):
        for bad in ("0", "-1", "many", ""):
            with self.subTest(value=bad):
                quietly(commands.set_top_n, self.session, bad)
                self.assertEqual(self.session.top_n, 3)

    def test_rerank_k_rejects_non_positive(self):
        quietly(commands.set_rerank_k, self.session, "0")
        self.assertEqual(self.session.rerank_k, 30)

    def test_temperature_rejects_non_numeric(self):
        quietly(commands.set_temperature, self.session, "hot")
        self.assertEqual(self.session.temperature, 0.0)

    def test_mode_rejects_unknown_and_is_case_insensitive(self):
        quietly(commands.set_mode, self.session, "DENSE")
        self.assertEqual(self.session.mode, "dense")
        quietly(commands.set_mode, self.session, "semantic")
        self.assertEqual(self.session.mode, "dense")

    def test_rerank_flag_parsing(self):
        for value in commands.TRUTHY:
            quietly(commands.set_rerank, self.session, value)
            self.assertTrue(self.session.rerank, value)
        for value in commands.FALSY:
            quietly(commands.set_rerank, self.session, value)
            self.assertFalse(self.session.rerank, value)

    def test_rerank_flag_rejects_garbage(self):
        quietly(commands.set_rerank, self.session, "maybe")
        self.assertFalse(self.session.rerank)


class TestSetSources(unittest.TestCase):
    session: commands.Session

    def setUp(self):
        self.session = commands.Session(known_sources={"a.pdf", "b.pdf"})

    def test_all_resets_the_filter(self):
        self.session.sources = ["a.pdf"]
        quietly(commands.set_sources, self.session, "all")
        self.assertIsNone(self.session.sources)

    def test_known_sources_are_accepted_and_trimmed(self):
        quietly(commands.set_sources, self.session, " a.pdf , b.pdf ")
        self.assertEqual(self.session.sources, ["a.pdf", "b.pdf"])

    def test_typo_is_rejected_and_lists_what_is_indexed(self):
        output = quietly(commands.set_sources, self.session, "a.pfd")
        self.assertIsNone(self.session.sources)
        self.assertIn("a.pfd", output)
        self.assertIn("a.pdf", output)

    def test_one_bad_name_rejects_the_whole_command(self):
        quietly(commands.set_sources, self.session, "a.pdf,nope.pdf")
        self.assertIsNone(self.session.sources)

    def test_validation_is_skipped_before_anything_is_indexed(self):
        session = commands.Session()
        quietly(commands.set_sources, session, "a.pdf")
        self.assertEqual(session.sources, ["a.pdf"])

    def test_empty_value_leaves_the_filter_alone(self):
        quietly(commands.set_sources, self.session, "  ,  ")
        self.assertIsNone(self.session.sources)


class TestHandleCommand(unittest.TestCase):
    session: commands.Session

    def setUp(self):
        self.session = commands.Session()

    def test_plain_question_is_not_a_command(self):
        self.assertFalse(commands.handle_command("what is dialectics?", self.session))

    def test_command_is_consumed(self):
        consumed: list[bool] = []
        quietly(
            lambda: consumed.append(commands.handle_command("/status", self.session))
        )
        self.assertEqual(consumed, [True])

    def test_name_is_case_insensitive_and_whitespace_tolerant(self):
        quietly(commands.handle_command, "/ TOP_N =5", self.session)
        self.assertEqual(self.session.top_n, 5)

    def test_unknown_command_is_reported_not_raised(self):
        output = quietly(commands.handle_command, "/nonsense=1", self.session)
        self.assertIn("Unknown command", output)

    def test_reindex_requires_a_filename(self):
        quietly(commands.handle_command, "/reindex=", self.session)
        self.assertIsNone(self.session.pending_reindex)
        quietly(commands.handle_command, "/reindex=a.pdf", self.session)
        self.assertEqual(self.session.pending_reindex, "a.pdf")


if __name__ == "__main__":
    unittest.main()
