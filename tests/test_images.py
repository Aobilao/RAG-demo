from __future__ import annotations

import io
import os
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from unittest import mock

import chromadb

from rag import chatbot, images


class ImagesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)

        self.images_dir = os.path.join(workspace.name, "images")
        os.makedirs(self.images_dir)

        client = chromadb.PersistentClient(path=os.path.join(workspace.name, "chroma"))
        self.collection = client.get_or_create_collection(
            name="images", configuration={"hnsw": {"space": "cosine"}}
        )

        self.descriptions: dict[str, str] = {}
        self.describe_calls: Counter[str] = Counter()

        def fake_describe(path: str) -> str:
            source = os.path.basename(path)
            self.describe_calls[source] += 1
            return self.descriptions.get(source, f"a photo of {source}")

        def fake_embed(text: str) -> list[list[float]]:
            codes = [float(ord(c)) for c in text[:8]]
            return [codes + [0.0] * (8 - len(codes))]

        targets = (
            (images, "describe", fake_describe),
            (chatbot, "embed", fake_embed),
        )
        for module, target, replacement in targets:
            patch = mock.patch.object(module, target, replacement)
            patch.start()
            self.addCleanup(patch.stop)

    def write_image(self, name: str, content: bytes = b"fake image bytes") -> str:
        path = os.path.join(self.images_dir, name)
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def sync(self) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            images.ingest_directory(self.collection, self.images_dir)
        return buffer.getvalue()


class TestIngestion(ImagesTestCase):
    def test_new_image_is_described_and_indexed(self) -> None:
        self.write_image("a.png")
        self.sync()
        self.assertEqual(self.collection.count(), 1)
        self.assertEqual(self.describe_calls["a.png"], 1)

    def test_unchanged_image_is_not_redescribed(self) -> None:
        self.write_image("a.png")
        self.sync()
        self.sync()
        self.assertEqual(self.describe_calls["a.png"], 1)

    def test_description_is_stored_as_the_document(self) -> None:
        self.descriptions["a.png"] = "Karl Marx. hai nguoi dan ong trong anh den trang"
        self.write_image("a.png")
        self.sync()

        stored = self.collection.get(ids=["a.png"])["documents"][0]
        self.assertEqual(stored, "Karl Marx. hai nguoi dan ong trong anh den trang")

    def test_edited_image_is_redescribed_without_duplicating(self) -> None:
        self.write_image("a.png", b"one")
        self.sync()
        self.write_image("a.png", b"two")
        self.sync()
        self.assertEqual(self.describe_calls["a.png"], 2)
        self.assertEqual(self.collection.count(), 1)

    def test_deleted_image_is_removed_from_the_index(self) -> None:
        self.write_image("a.png")
        self.write_image("b.png")
        self.sync()
        os.remove(os.path.join(self.images_dir, "a.png"))
        self.sync()
        self.assertEqual(set(self.collection.get(include=[])["ids"]), {"b.png"})

    def test_unsupported_file_is_ignored(self) -> None:
        self.write_image("notes.txt")
        self.sync()
        self.assertEqual(self.collection.count(), 0)

    def test_case_insensitive_extension_is_picked_up(self) -> None:
        self.write_image("A.PNG")
        self.sync()
        self.assertEqual(self.collection.count(), 1)


class TestSearch(ImagesTestCase):
    def test_closest_description_ranks_first(self) -> None:
        self.descriptions["cat.png"] = "meo ngoi tren ghe"
        self.descriptions["dog.png"] = "cho chay trong cong vien"
        self.write_image("cat.png")
        self.write_image("dog.png")
        self.sync()

        results = images.search(self.collection, "meo ngoi tren ghe")
        self.assertEqual(results[0].source, "cat.png")

    def test_empty_collection_returns_nothing(self) -> None:
        self.assertEqual(images.search(self.collection, "anything"), [])

    def test_blank_query_returns_nothing(self) -> None:
        self.write_image("a.png")
        self.sync()
        self.assertEqual(images.search(self.collection, "   "), [])


if __name__ == "__main__":
    unittest.main()
