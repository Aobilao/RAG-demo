from __future__ import annotations

import json
import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import images
from ..service import RAGService, to_evidence

STATIC_DIR = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return RAGService()


@lru_cache(maxsize=1)
def get_gallery() -> chromadb.Collection:
    os.makedirs(images.IMAGES_PATH, exist_ok=True)
    collection = images.open_collection()
    images.ingest_directory(collection, images.IMAGES_PATH)
    return collection


class AskRequest(BaseModel):
    question: str
    top_n: int = 4
    mode: str = "hybrid"
    rerank: bool = False
    temperature: float = 0.0
    sources: list[str] | None = None


class GallerySearchRequest(BaseModel):
    query: str
    top_n: int = 6


def event(name: str, payload: Any) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


app = FastAPI(title="RAG")


@app.get("/api/status")
def status() -> dict[str, Any]:
    service = get_service()
    return {
        "chunks": service.collection.count(),
        "sources": sorted(service.corpus_index.sources),
        "model": service.language_model,
    }


@app.post("/api/refresh")
def refresh() -> dict[str, Any]:
    service = get_service()
    try:
        chunks = service.refresh()
    except Exception as exc:
        raise HTTPException(500, f"Không thể làm mới chỉ mục: {exc}") from exc
    return {"chunks": chunks, "sources": sorted(service.corpus_index.sources)}


@app.post("/api/ask")
def ask(request: AskRequest) -> StreamingResponse:
    service = get_service()

    def events() -> Iterator[str]:
        try:
            rows = service.retrieve_rows(
                request.question,
                top_n=request.top_n,
                mode=request.mode,
                rerank=request.rerank,
                sources=request.sources,
            )
            if not rows:
                yield event("empty", {})
                return

            evidence = to_evidence(rows)
            yield event("sources", [item.__dict__ for item in evidence])

            for fragment in service.stream_answer(
                request.question, rows, temperature=request.temperature
            ):
                if fragment:
                    yield event("token", fragment)
            yield event("done", {})
        except Exception as exc:
            yield event("error", str(exc))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/gallery/status")
def gallery_status() -> dict[str, Any]:
    return {"count": get_gallery().count()}


@app.post("/api/gallery/refresh")
def gallery_refresh() -> dict[str, Any]:
    collection = get_gallery()
    try:
        images.ingest_directory(collection, images.IMAGES_PATH)
    except Exception as exc:
        raise HTTPException(500, f"Không thể làm mới thư viện ảnh: {exc}") from exc
    return {"count": collection.count()}


@app.post("/api/gallery/search")
def gallery_search(request: GallerySearchRequest) -> list[dict[str, Any]]:
    matches = images.search(get_gallery(), request.query, request.top_n)
    return [
        {
            "source": m.source,
            "caption": m.caption,
            "score": m.score,
            "url": f"/api/gallery/file/{m.source}",
        }
        for m in matches
    ]


@app.get("/api/gallery/file/{name}")
def gallery_file(name: str) -> FileResponse:
    images_path = Path(images.IMAGES_PATH).resolve()
    path = images_path / Path(name).name
    if not path.is_file():
        raise HTTPException(404, "Không tìm thấy ảnh")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
