import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")

from fastapi.testclient import TestClient  # noqa: E402

from rag import images  # noqa: E402
from rag.web.server import app, event  # noqa: E402


class EventTest(unittest.TestCase):
    def test_frames_match_the_sse_format(self) -> None:
        self.assertEqual(event("token", "hi"), 'event: token\ndata: "hi"\n\n')

    def test_vietnamese_is_not_escaped(self) -> None:
        frame = event("token", "biện chứng")
        self.assertIn("biện chứng", frame)
        self.assertNotIn("\\u", frame)

    def test_payload_survives_a_round_trip(self) -> None:
        payload = [{"rank": 1, "source": "GT.pdf", "pages": "10, 11"}]
        data = event("sources", payload).split("data: ", 1)[1].strip()
        self.assertEqual(json.loads(data), payload)


class GalleryFileRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        Path(self.temp.name, "real.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        patcher = mock.patch.object(images, "IMAGES_PATH", self.temp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(app)

    def test_serves_an_indexed_image(self) -> None:
        self.assertEqual(self.client.get("/api/gallery/file/real.png").status_code, 200)

    def test_missing_image_is_a_404(self) -> None:
        self.assertEqual(self.client.get("/api/gallery/file/nope.png").status_code, 404)

    def test_traversal_cannot_escape_the_images_directory(self) -> None:
        secret = Path(self.temp.name).parent / "secret.txt"
        secret.write_text("private")
        self.addCleanup(secret.unlink)

        for attempt in ("../secret.txt", "..%2fsecret.txt", "....//secret.txt"):
            with self.subTest(attempt=attempt):
                response = self.client.get(f"/api/gallery/file/{attempt}")
                self.assertNotEqual(response.status_code, 200)
                self.assertNotIn(b"private", response.content)


if __name__ == "__main__":
    unittest.main()
