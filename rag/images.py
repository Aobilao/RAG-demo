from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import chromadb
import ollama

from . import chatbot

IMAGES_PATH = "images"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "images"
VISION_MODEL = "qwen2.5vl:3b"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

DESCRIBE_PROMPT = (
    "If you recognize any specific, famous, real people in this photo, begin with "
    "their names, comma-separated, followed by a period. Skip the names entirely if "
    "you do not recognize anyone. Then describe the image in Vietnamese: the subject, "
    "the setting, the colors, the composition, and any visible text. Reply with only "
    "the names and the Vietnamese description, no preamble."
)


@dataclass(frozen=True)
class Match:
    source: str
    caption: str
    score: float


def open_collection(chroma_path: str = CHROMA_PATH) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}
    )


def file_hash(path: str) -> str:
    with open(path, "rb") as file:
        digest = hashlib.blake2b(file.read()).hexdigest()
    return f"{VISION_MODEL}:{digest}"


def list_images(images_dir: str) -> list[str]:
    if not os.path.isdir(images_dir):
        return []
    return sorted(
        os.path.join(images_dir, name)
        for name in os.listdir(images_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def describe(path: str) -> str:
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": DESCRIBE_PROMPT, "images": [path]}],
        options={"temperature": 0},
    )
    return (response.message.content or "").strip()


def ingest_image(
    collection: chromadb.Collection, path: str, file_hashes: dict[str, str]
) -> bool:
    source = os.path.basename(path)
    current_hash = file_hash(path)

    if file_hashes.get(source) == current_hash:
        print(f"{source}: already indexed, skipping")
        return False

    text = describe(path)
    collection.upsert(
        ids=[source],
        documents=[text],
        embeddings=chatbot.embed(text),
        metadatas=[{"path": path, "file_hash": current_hash}],
    )
    print(f"{source}: indexed — {text}")
    return True


def ingest_directory(collection: chromadb.Collection, images_dir: str) -> bool:
    paths = list_images(images_dir)
    current_sources = {os.path.basename(p) for p in paths}

    existing = collection.get(include=["metadatas"])
    file_hashes = {
        source: str(meta.get("file_hash") or "")
        for source, meta in zip(existing["ids"], existing["metadatas"] or [])
    }

    stale = set(file_hashes) - current_sources
    if stale:
        collection.delete(ids=list(stale))
        for source in stale:
            print(f"{source}: removed from index (file deleted)")

    if not paths:
        print(f"No images found in {images_dir}/")

    changed = bool(stale)
    for path in paths:
        changed |= ingest_image(collection, path, file_hashes)
    return changed


def search(collection: chromadb.Collection, query: str, top_n: int = 6) -> list[Match]:
    count = collection.count()
    if count == 0 or not query.strip():
        return []
    result = collection.query(
        query_embeddings=chatbot.embed(query), n_results=min(top_n, count)
    )
    documents = result["documents"]
    distances = result["distances"]
    if not documents or not distances:
        return []
    return [
        Match(source=source, caption=doc, score=1.0 - dist)
        for source, doc, dist in zip(result["ids"][0], documents[0], distances[0])
    ]
