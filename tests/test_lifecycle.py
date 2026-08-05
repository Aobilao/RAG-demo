from __future__ import annotations

import io
import itertools
import os
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from typing import Any
from unittest import mock

import chromadb

from rag import chatbot

VECTOR = [0.1, 0.2, 0.3]


class LifecycleTestCase(unittest.TestCase):
    CHUNKS_PER_DOC = 35
    BATCH_SIZE = 10

    docs: str
    collection: chromadb.Collection
    extractions: Counter[str]
    empty_sources: set[str]

    def setUp(self) -> None:
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)

        self.docs = os.path.join(workspace.name, "docs")
        os.makedirs(self.docs)

        client = chromadb.PersistentClient(path=os.path.join(workspace.name, "chroma"))
        self.collection = client.get_or_create_collection(
            name="documents", configuration={"hnsw": {"space": "cosine"}}
        )

        self.extractions = Counter()
        self.empty_sources = set()

        for target, replacement in (
            ("embed", self.fake_embed),
            ("chunk_pdf", self.fake_chunk_pdf),
            ("BATCH_SIZE", self.BATCH_SIZE),
        ):
            patch = mock.patch.object(chatbot, target, replacement)
            patch.start()
            self.addCleanup(patch.stop)

    def fake_embed(self, texts: list[str] | str) -> list[Any]:
        return [VECTOR] * (1 if isinstance(texts, str) else len(texts))

    def fake_chunk_pdf(self, path: str) -> list[tuple[str, list[int]]]:
        source = os.path.basename(path)
        self.extractions[source] += 1
        if source in self.empty_sources:
            return []
        return [
            (f"chunk {i} of {source}", [i // 5 + 1]) for i in range(self.CHUNKS_PER_DOC)
        ]

    def write_pdf(self, name: str, content: bytes = b"fake pdf bytes") -> str:
        path = os.path.join(self.docs, name)
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def sync(self) -> str:
        index = chatbot.build_corpus_index(self.collection)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            chatbot.ingest_directory(self.collection, self.docs, index.file_hashes)
        return buffer.getvalue()

    def interrupt_after(self, batches: int) -> Any:
        counter = itertools.count(1)

        def flaky(texts: list[str] | str) -> list[Any]:
            if next(counter) > batches:
                raise KeyboardInterrupt("user interrupted")
            return self.fake_embed(texts)

        return mock.patch.object(chatbot, "embed", flaky)

    def markers(self) -> dict[str, str]:
        return chatbot.indexed_hashes(self.collection)


class TestFirstIngest(LifecycleTestCase):
    def test_chunks_are_stored_and_the_document_is_marked_complete(self):
        self.write_pdf("a.pdf")
        self.sync()
        self.assertEqual(self.collection.count(), self.CHUNKS_PER_DOC)
        self.assertEqual(set(self.markers()), {"a.pdf"})

    def test_marker_records_the_hash_the_file_was_ingested_at(self):
        path = self.write_pdf("a.pdf")
        self.sync()
        self.assertEqual(self.markers()["a.pdf"], chatbot.file_hash(path))


class TestSkippingUnchangedWork(LifecycleTestCase):
    def test_unchanged_pdf_is_not_extracted_again(self):
        self.write_pdf("a.pdf")
        self.sync()
        self.sync()
        self.assertEqual(self.extractions["a.pdf"], 1)
        self.assertEqual(self.collection.count(), self.CHUNKS_PER_DOC)

    def test_edited_pdf_is_re_extracted_without_duplicating_chunks(self):
        self.write_pdf("a.pdf")
        self.sync()
        self.write_pdf("a.pdf", b"different bytes entirely")
        self.sync()
        self.assertEqual(self.extractions["a.pdf"], 2)
        self.assertEqual(self.collection.count(), self.CHUNKS_PER_DOC)

    def test_legacy_per_chunk_hash_still_counts_as_indexed(self):
        path = self.write_pdf("a.pdf")
        self.collection.add(
            ids=["a.pdf::0"],
            documents=["a chunk written by an older version"],
            embeddings=[VECTOR],
            metadatas=[
                {"source": "a.pdf", "pages": "1", "file_hash": chatbot.file_hash(path)}
            ],
        )
        self.sync()
        self.assertEqual(self.extractions["a.pdf"], 0)


class TestInterruptedIngest(LifecycleTestCase):
    def test_interruption_leaves_chunks_but_no_marker(self):
        self.write_pdf("a.pdf")
        with self.interrupt_after(2), self.assertRaises(KeyboardInterrupt):
            self.sync()
        self.assertEqual(self.collection.count(), 2 * self.BATCH_SIZE)
        self.assertEqual(self.markers(), {})

    def test_next_run_discards_the_orphans_and_re_indexes(self):
        self.write_pdf("a.pdf")
        with self.interrupt_after(2), self.assertRaises(KeyboardInterrupt):
            self.sync()

        output = self.sync()
        self.assertIn("interrupted", output)
        self.assertEqual(self.collection.count(), self.CHUNKS_PER_DOC)
        self.assertEqual(set(self.markers()), {"a.pdf"})

    def test_a_half_written_document_is_never_reported_as_up_to_date(self):
        self.write_pdf("a.pdf")
        with self.interrupt_after(2), self.assertRaises(KeyboardInterrupt):
            self.sync()
        self.assertNotIn("already indexed", self.sync())


class TestUnreadablePdf(LifecycleTestCase):
    def test_pdf_yielding_no_text_is_marked_rather_than_retried_forever(self):
        self.empty_sources.add("scan.pdf")
        self.write_pdf("scan.pdf")

        self.sync()
        self.assertEqual(self.extractions["scan.pdf"], 1)
        self.assertEqual(self.collection.count(), 0)
        self.assertEqual(set(self.markers()), {"scan.pdf"})

        self.sync()
        self.assertEqual(self.extractions["scan.pdf"], 1)


class TestDeletion(LifecycleTestCase):
    def test_deleting_a_pdf_removes_its_chunks_and_marker(self):
        self.write_pdf("a.pdf")
        self.write_pdf("b.pdf")
        self.sync()

        os.unlink(os.path.join(self.docs, "a.pdf"))
        self.sync()

        self.assertEqual(set(self.markers()), {"b.pdf"})
        self.assertEqual(self.collection.count(), self.CHUNKS_PER_DOC)

    def test_removing_the_only_pdf_does_not_raise(self):
        self.write_pdf("a.pdf")
        self.sync()

        os.unlink(os.path.join(self.docs, "a.pdf"))
        self.sync()

        self.assertEqual(self.collection.count(), 0)
        self.assertEqual(self.markers(), {})

    def test_a_readded_pdf_is_ingested_afresh(self):
        self.write_pdf("a.pdf")
        self.sync()
        os.unlink(os.path.join(self.docs, "a.pdf"))
        self.sync()

        self.write_pdf("a.pdf")
        self.sync()
        self.assertEqual(self.extractions["a.pdf"], 2)
        self.assertEqual(self.collection.count(), self.CHUNKS_PER_DOC)


if __name__ == "__main__":
    unittest.main()
