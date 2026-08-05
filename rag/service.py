from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import chromadb
import ollama
from chromadb.api.types import Metadata

from .chatbot import (
    CHROMA_PATH,
    COLLECTION_NAME,
    LANGUAGE_MODEL,
    PDF_PATH,
    CorpusIndex,
    build_corpus_index,
    build_prompt,
    ingest_directory,
    retrieve,
    sanitize,
)

NO_CONTEXT_ANSWER = "Không tìm thấy ngữ cảnh phù hợp trong tài liệu."
NO_EVIDENCE_MARKDOWN = "_Không có đoạn tài liệu nào được truy xuất._"

MAX_EVIDENCE_CHARS = 700

Row = tuple[str, float, Metadata]


@dataclass(frozen=True)
class Evidence:
    rank: int
    text: str
    score: float
    source: str
    pages: str

    @property
    def citation(self) -> str:
        return f"[{self.rank}] {self.source}, p. {self.pages}"


def to_evidence(rows: list[Row]) -> list[Evidence]:
    return [
        Evidence(
            rank=rank,
            text=text,
            score=float(score),
            source=str(metadata.get("source", "unknown")),
            pages=str(metadata.get("pages", "?")),
        )
        for rank, (text, score, metadata) in enumerate(rows, start=1)
    ]


class RAGService:
    def __init__(
        self,
        docs_path: str = PDF_PATH,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        language_model: str = LANGUAGE_MODEL,
    ) -> None:
        self.docs_path = docs_path
        self.language_model = language_model
        os.makedirs(self.docs_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )
        self.corpus_index: CorpusIndex = build_corpus_index(self.collection)
        self.refresh()

    def refresh(self) -> int:
        if ingest_directory(
            self.collection, self.docs_path, self.corpus_index.file_hashes
        ):
            self.corpus_index = build_corpus_index(self.collection)
        return self.collection.count()

    def retrieve_rows(
        self,
        query: str,
        top_n: int = 4,
        mode: str = "hybrid",
        rerank: bool = True,
        rerank_k: int = 30,
        sources: list[str] | None = None,
    ) -> list[Row]:
        query = sanitize(query.strip())
        if not query or self.collection.count() == 0:
            return []
        return retrieve(
            collection=self.collection,
            corpus_index=self.corpus_index,
            query=query,
            top_n=top_n,
            sources=sources,
            mode=mode,
            rerank=rerank,
            rerank_k=rerank_k,
        )

    def search(self, query: str, **options: Any) -> list[Evidence]:
        return to_evidence(self.retrieve_rows(query, **options))

    def stream_answer(
        self, question: str, rows: list[Row], temperature: float = 0.0
    ) -> Iterator[str]:
        stream = ollama.chat(
            model=self.language_model,
            messages=[
                {"role": "system", "content": build_prompt(rows)},
                {"role": "user", "content": sanitize(question)},
            ],
            stream=True,
            think=False,
            options={"temperature": temperature},
        )
        for chunk in stream:
            yield chunk.message.content or ""

    def answer(
        self, question: str, temperature: float = 0.0, **options: Any
    ) -> tuple[str, list[Evidence]]:
        rows = self.retrieve_rows(question, **options)
        if not rows:
            return NO_CONTEXT_ANSWER, []

        response: Any = ollama.chat(
            model=self.language_model,
            messages=[
                {"role": "system", "content": build_prompt(rows)},
                {"role": "user", "content": sanitize(question)},
            ],
            think=False,
            options={"temperature": temperature},
        )
        return response.message.content.strip(), to_evidence(rows)


def evidence_to_markdown(
    evidence: list[Evidence], max_chars: int = MAX_EVIDENCE_CHARS
) -> str:
    if not evidence:
        return NO_EVIDENCE_MARKDOWN

    blocks = []
    for item in evidence:
        text = " ".join(item.text.split())
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        blocks.append(f"**{item.citation}** — score `{item.score:.3f}`\n\n{text}")
    return "\n\n---\n\n".join(blocks)
