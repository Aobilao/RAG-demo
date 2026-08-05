from __future__ import annotations

import glob
import hashlib
import os
import re
import readline  # noqa: F401  # pyright: ignore[reportUnusedImport]
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import bm25s
import chromadb
import ollama
from chromadb.api.types import Metadata, PyEmbedding, Where

from . import images
from .commands import Session, handle_command
from .extractor import EXTRACTOR_VERSION, extract_blocks
from .reranker import load_reranker, rerank_scores

if TYPE_CHECKING:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"
MAGENTA = "\033[35m"

EMBEDDING_MODEL = "bge-m3"
LANGUAGE_MODEL = "qwen3.5:4b"

PDF_PATH = "docs"
PAGE_SEPARATOR = "\n\n"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
INDEXED_PREFIX = "indexed:"
SCHEMA_KEY = "rag:schema"
SCHEMA_VERSION = "1"
BATCH_SIZE = 32
RRF_K = 60
RERANK_K = 30
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

SCORE_LABELS = {"dense": "Similarity", "bm25": "BM25", "hybrid": "RRF"}
RERANK_LABEL = "Rerank"

_splitter: RecursiveCharacterTextSplitter | None = None


def get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    return _splitter


def sanitize(s: str) -> str:
    return s.encode("utf-8", errors="ignore").decode("utf-8")


def file_hash(path: str) -> str:
    with open(path, "rb") as file:
        digest = hashlib.blake2b(file.read()).hexdigest()
    return f"{EXTRACTOR_VERSION}:{digest}"


def embed(texts: list[str] | str) -> list[PyEmbedding]:
    return list(ollama.embed(model=EMBEDDING_MODEL, input=texts).embeddings)


def format_pages(pages: list[int]) -> str:
    return ", ".join(str(p) for p in pages)


def find_pages(
    chunk: str, full_text: str, page_offsets: list[tuple[int, int, list[int]]]
) -> list[int]:
    idx = full_text.find(chunk)
    if idx == -1:
        return []
    chunk_start, chunk_end = idx, idx + len(chunk)
    return sorted(
        {
            page_num
            for start, end, pages in page_offsets
            if start < chunk_end and end > chunk_start
            for page_num in pages
        }
    )


def chunk_pdf(path: str) -> list[tuple[str, list[int]]]:
    blocks = extract_blocks(path)
    full_text = ""
    page_offsets: list[tuple[int, int, list[int]]] = []

    for pages, block_text in blocks:
        start = len(full_text)
        full_text += block_text
        page_offsets.append((start, len(full_text), pages))
        full_text += PAGE_SEPARATOR

    chunks = get_splitter().split_text(full_text)
    return [(chunk, find_pages(chunk, full_text, page_offsets)) for chunk in chunks]


class MetadataStore(Protocol):
    @property
    def metadata(self) -> dict[str, Any] | None: ...

    def modify(self, *, metadata: dict[str, Any]) -> None: ...


def indexed_hashes(collection: MetadataStore) -> dict[str, str]:
    return {
        key.removeprefix(INDEXED_PREFIX): str(value)
        for key, value in (collection.metadata or {}).items()
        if key.startswith(INDEXED_PREFIX)
    }


def write_metadata(collection: MetadataStore, metadata: dict[str, Any]) -> None:
    collection.modify(metadata={SCHEMA_KEY: SCHEMA_VERSION, **metadata})


def mark_indexed(collection: MetadataStore, source: str, digest: str) -> None:
    metadata = dict(collection.metadata or {})
    metadata[INDEXED_PREFIX + source] = digest
    write_metadata(collection, metadata)


def unmark_indexed(collection: MetadataStore, source: str) -> None:
    metadata = dict(collection.metadata or {})
    if metadata.pop(INDEXED_PREFIX + source, None) is not None:
        write_metadata(collection, metadata)


def ingest_pdf(
    collection: chromadb.Collection, path: str, file_hashes: dict[str, str]
) -> bool:
    source = os.path.basename(path)
    current_hash = file_hash(path)
    stored_hash = file_hashes.get(source)

    if stored_hash == current_hash:
        print(f"{source}: already indexed, skipping")
        return False
    if stored_hash is not None:
        collection.delete(where={"source": source})
        print(f"{source}: file changed, re-indexing")
    elif collection.get(where={"source": source}, limit=1)["ids"]:
        collection.delete(where={"source": source})
        print(f"{source}: previous indexing was interrupted, re-indexing")

    pieces = chunk_pdf(path)
    if not pieces:
        print(f"{source}: no text extracted, skipping")
        mark_indexed(collection, source, current_hash)
        return True

    ids = [f"{source}::{i}" for i in range(len(pieces))]
    docs = [sanitize(chunk) for chunk, _ in pieces]
    metas: list[Metadata] = [
        {"source": source, "pages": format_pages(pages)} for _, pages in pieces
    ]

    for start in range(0, len(docs), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            embeddings=embed(docs[start:end]),
            metadatas=metas[start:end],
        )
        print(f"{source}: indexed {min(end, len(docs))}/{len(docs)} chunks")

    mark_indexed(collection, source, current_hash)
    return True


def ingest_directory(
    collection: chromadb.Collection, pdf_dir: str, file_hashes: dict[str, str]
) -> bool:
    paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    current_sources = {os.path.basename(p) for p in paths}

    changed = False
    for source in file_hashes.keys() - current_sources:
        collection.delete(where={"source": source})
        unmark_indexed(collection, source)
        print(f"{source}: removed from database (file deleted)")
        changed = True

    if not paths:
        print(f"No PDFs found in {pdf_dir}/")
    for path in paths:
        changed |= ingest_pdf(collection, path, file_hashes)
    return changed


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class CorpusIndex:
    bm25: bm25s.BM25 | None
    ids: list[str]
    by_id: dict[str, tuple[str, Metadata]]
    file_hashes: dict[str, str]
    sources: set[str]


def build_corpus_index(collection: chromadb.Collection) -> CorpusIndex:
    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    docs = data["documents"] or []
    metas = data["metadatas"] or []
    bm25 = None
    if docs:
        bm25 = bm25s.BM25()
        bm25.index([tokenize(doc) for doc in docs], show_progress=False)
    by_id = {doc_id: (doc, meta) for doc_id, doc, meta in zip(ids, docs, metas)}

    file_hashes = {
        str(meta["source"]): str(meta["file_hash"])
        for meta in metas
        if meta.get("file_hash")
    }
    file_hashes.update(indexed_hashes(collection))
    return CorpusIndex(
        bm25=bm25,
        ids=ids,
        by_id=by_id,
        file_hashes=file_hashes,
        sources={str(meta["source"]) for meta in metas},
    )


def dense_search(
    collection: chromadb.Collection,
    query: str,
    top_k: int,
    sources: list[str] | None = None,
) -> list[tuple[str, float]]:
    count = collection.count()
    if count == 0:
        return []
    where: Where | None = None
    if sources:
        where = {"source": {"$in": list(sources)}}
    result = collection.query(
        query_embeddings=embed(query),
        n_results=min(top_k, count),
        where=where,
    )
    distances = result["distances"]
    if not distances:
        return []
    return [
        (doc_id, 1.0 - dist) for doc_id, dist in zip(result["ids"][0], distances[0])
    ]


def bm25_search(
    index: CorpusIndex,
    query: str,
    top_k: int,
    sources: list[str] | None = None,
) -> list[tuple[str, float]]:
    if index.bm25 is None:
        return []
    scores = index.bm25.get_scores(tokenize(query))
    candidates = [
        (doc_id, score)
        for doc_id, score in zip(index.ids, scores)
        if sources is None or index.by_id[doc_id][1]["source"] in sources
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return candidates[:top_k]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_retrieve(
    collection: chromadb.Collection,
    corpus_index: CorpusIndex,
    query: str,
    top_n: int,
    sources: list[str] | None,
) -> list[tuple[str, float, Metadata]]:
    candidate_k = max(top_n * 4, 20)
    dense_ids = [
        doc_id for doc_id, _ in dense_search(collection, query, candidate_k, sources)
    ]
    bm25_ids = [
        doc_id for doc_id, _ in bm25_search(corpus_index, query, candidate_k, sources)
    ]

    fused_scores = reciprocal_rank_fusion([dense_ids, bm25_ids])
    if not fused_scores:
        return []

    ranked_ids = sorted(
        fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True
    )[:top_n]
    return [
        (
            corpus_index.by_id[doc_id][0],
            fused_scores[doc_id],
            corpus_index.by_id[doc_id][1],
        )
        for doc_id in ranked_ids
    ]


def to_triples(
    corpus_index: CorpusIndex, results: list[tuple[str, float]]
) -> list[tuple[str, float, Metadata]]:
    return [
        (corpus_index.by_id[doc_id][0], score, corpus_index.by_id[doc_id][1])
        for doc_id, score in results
    ]


def apply_rerank(
    query: str,
    candidates: list[tuple[str, float, Metadata]],
    top_n: int,
) -> list[tuple[str, float, Metadata]]:
    reranker = load_reranker()
    if reranker is None:
        print(f"{YELLOW}Falling back to first-stage ranking.{RESET}")
        return candidates[:top_n]

    scores = rerank_scores(reranker, query, [chunk for chunk, _, _ in candidates])
    reranked = [
        (chunk, score, meta) for (chunk, _, meta), score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda triple: triple[1], reverse=True)
    return reranked[:top_n]


def retrieve(
    collection: chromadb.Collection,
    corpus_index: CorpusIndex,
    query: str,
    top_n: int = 3,
    sources: list[str] | None = None,
    mode: str = "hybrid",
    rerank: bool = False,
    rerank_k: int = RERANK_K,
) -> list[tuple[str, float, Metadata]]:
    fetch_n = max(rerank_k, top_n) if rerank else top_n

    if mode == "dense":
        candidates = to_triples(
            corpus_index, dense_search(collection, query, fetch_n, sources)
        )
    elif mode == "bm25":
        candidates = to_triples(
            corpus_index, bm25_search(corpus_index, query, fetch_n, sources)
        )
    else:
        candidates = hybrid_retrieve(collection, corpus_index, query, fetch_n, sources)

    if rerank and candidates:
        return apply_rerank(query, candidates, top_n)
    return candidates[:top_n]


def build_prompt(retrieved: list[tuple[str, float, Metadata]]) -> str:
    context_parts = []
    for i, (chunk, _, meta) in enumerate(retrieved, start=1):
        context_parts.append(
            f"[{i}] (Source: {meta['source']}, pages {meta['pages']})\n{chunk}"
        )
    context = "\n\n".join(context_parts)
    return (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. "
        "If there is no information about the question, say that there is no information. "
        "Cite your sources inline right after the relevant statement using the matching "
        "bracket number, e.g. write [1] or [2][3] — using the actual number of the source, "
        'never the literal text "[n]". '
        "Don't make up any new information and don't use markdown formatting:\n\n"
        f"{context}"
    )


def find_images(images_collection: chromadb.Collection, query: str) -> None:
    if images_collection.count() == 0:
        print("No images indexed. Add pictures to images/ and run /refresh=images.")
        return
    matches = images.search(images_collection, query)
    if not matches:
        print("No matching images found.")
        return
    print(f"\n{CYAN}Images matching '{query}':{RESET}")
    for i, match in enumerate(matches, start=1):
        print(
            f"{YELLOW} - [{i}] ({match.score:.3f}) {match.source}{RESET}\n{match.caption}\n"
        )


def reindex(collection: chromadb.Collection, source: str) -> CorpusIndex:
    path = os.path.join(PDF_PATH, source)
    collection.delete(where={"source": source})
    unmark_indexed(collection, source)
    print(f"Deleted {source}, re-ingesting...")
    if os.path.exists(path):
        ingest_pdf(collection, path, {})
    else:
        print(f"{source} not found in {PDF_PATH}/")
    return build_corpus_index(collection)


def refresh_docs(
    collection: chromadb.Collection, corpus_index: CorpusIndex
) -> CorpusIndex:
    if ingest_directory(collection, PDF_PATH, corpus_index.file_hashes):
        return build_corpus_index(collection)
    return corpus_index


def apply_pending(
    session: Session,
    collection: chromadb.Collection,
    images_collection: chromadb.Collection,
    corpus_index: CorpusIndex,
) -> CorpusIndex:
    if session.pending_reindex:
        corpus_index = reindex(collection, session.pending_reindex)
        session.known_sources = corpus_index.sources
        session.pending_reindex = None
    if session.pending_image_query:
        find_images(images_collection, session.pending_image_query)
        session.pending_image_query = None
    if session.pending_refresh == "docs":
        corpus_index = refresh_docs(collection, corpus_index)
        session.known_sources = corpus_index.sources
        session.pending_refresh = None
    elif session.pending_refresh == "images":
        images.ingest_directory(images_collection, images.IMAGES_PATH)
        session.pending_refresh = None
    return corpus_index


def answer_query(
    collection: chromadb.Collection,
    corpus_index: CorpusIndex,
    session: Session,
    query: str,
) -> None:
    if collection.count() == 0:
        print("Database is empty. Add PDFs to docs/ and run /refresh=docs.")
        return

    retrieved_knowledge = retrieve(
        collection,
        corpus_index,
        query,
        session.top_n,
        session.sources,
        session.mode,
        session.rerank,
        session.rerank_k,
    )
    if not retrieved_knowledge:
        print("No matching results found.")
        return

    label = RERANK_LABEL if session.rerank else SCORE_LABELS[session.mode]
    print(f"\n{CYAN}Retrieved knowledge:{RESET}")
    for i, (chunk, score, meta) in enumerate(retrieved_knowledge, start=1):
        location = f"{meta['source']}, p. {meta['pages']}"
        print(f"{YELLOW} - [{i}] ({label}: {score:.3f}) [{location}]{RESET}\n{chunk}\n")

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": build_prompt(retrieved_knowledge)},
            {"role": "user", "content": sanitize(query)},
        ],
        stream=True,
        think=False,
        options={"temperature": session.temperature},
    )
    print(f"{BOLD}{GREEN}Chatbot response:{RESET}")
    for chunk in stream:
        print(chunk.message.content, end="", flush=True)
    print()


def main() -> None:
    os.makedirs(PDF_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    corpus_index = refresh_docs(collection, build_corpus_index(collection))

    os.makedirs(images.IMAGES_PATH, exist_ok=True)
    images_collection = images.open_collection()
    images.ingest_directory(images_collection, images.IMAGES_PATH)

    session = Session(known_sources=corpus_index.sources)
    while True:
        stage = f"{session.mode} + rerank" if session.rerank else session.mode
        try:
            input_query = input(
                f"\n{MAGENTA}({stage}) Ask a question (or 'q' to quit):{RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not input_query:
            continue
        if input_query.lower() in ("q", "quit", "exit"):
            break

        if handle_command(input_query, session):
            corpus_index = apply_pending(
                session, collection, images_collection, corpus_index
            )
            continue

        answer_query(collection, corpus_index, session, input_query)


if __name__ == "__main__":
    main()
