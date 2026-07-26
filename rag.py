import glob
import hashlib
import os
import re
import readline  # noqa: F401  # pyright: ignore[reportUnusedImport]
from dataclasses import dataclass

import bm25s
import chromadb
import ollama
import pymupdf
from chromadb.api.types import Metadata, PyEmbedding, Where
from langchain_text_splitters import RecursiveCharacterTextSplitter

from commands import Session, handle_command

BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"
MAGENTA = "\033[35m"

EMBEDDING_MODEL = "bge-m3"
LANGUAGE_MODEL = "qwen3.5:4b"

PDF_PATH = "docs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
BATCH_SIZE = 32
RRF_K = 60

SCORE_LABELS = {"dense": "Similarity", "bm25": "BM25", "hybrid": "RRF"}


def sanitize(s: str) -> str:
    return s.encode("utf-8", errors="ignore").decode("utf-8")


def file_hash(path: str) -> str:
    with open(path, "rb") as file:
        return hashlib.blake2b(file.read()).hexdigest()


def embed(texts: list[str] | str) -> list[PyEmbedding]:
    return list(ollama.embed(model=EMBEDDING_MODEL, input=texts).embeddings)


def format_pages(pages: list[int]) -> str:
    return ", ".join(str(p) for p in pages)


def extract_pages(path: str) -> list[tuple[int, str]]:
    doc = pymupdf.open(path)
    return [
        (page_num, page.get_text())
        for page_num, page in enumerate(doc.pages(), start=1)
    ]


def find_pages(
    chunk: str, full_text: str, page_offsets: list[tuple[int, int, int]]
) -> list[int]:
    idx = full_text.find(chunk)
    if idx == -1:
        return []
    chunk_start, chunk_end = idx, idx + len(chunk)
    return [
        page_num
        for start, end, page_num in page_offsets
        if start < chunk_end and end > chunk_start
    ]


def chunk_pdf(
    path: str, splitter: RecursiveCharacterTextSplitter
) -> list[tuple[str, list[int]]]:
    pages = extract_pages(path)
    full_text = ""
    page_offsets: list[tuple[int, int, int]] = []

    for page_num, page_text in pages:
        start = len(full_text)
        full_text += page_text
        page_offsets.append((start, len(full_text), page_num))

    chunks = splitter.split_text(full_text)
    return [(chunk, find_pages(chunk, full_text, page_offsets)) for chunk in chunks]


def already_indexed(
    collection: chromadb.Collection, source: str, current_hash: str
) -> bool:
    result = collection.get(where={"source": source}, limit=1)
    metadatas = result["metadatas"]
    if not result["ids"] or not metadatas:
        return False
    stored_hash = metadatas[0].get("file_hash", "")
    if stored_hash != current_hash:
        collection.delete(where={"source": source})
        print(f"{source}: file changed, re-indexing")
        return False
    return True


def ingest_pdf(
    collection: chromadb.Collection,
    path: str,
    splitter: RecursiveCharacterTextSplitter,
) -> None:
    source = os.path.basename(path)
    current_hash = file_hash(path)
    if already_indexed(collection, source, current_hash):
        print(f"{source}: already indexed, skipping")
        return

    pieces = chunk_pdf(path, splitter)
    if not pieces:
        print(f"{source}: no text extracted, skipping")
        return

    ids = [f"{source}::{i}" for i in range(len(pieces))]
    docs = [sanitize(chunk) for chunk, _ in pieces]
    metas: list[Metadata] = [
        {"source": source, "pages": format_pages(pages), "file_hash": current_hash}
        for _, pages in pieces
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


def ingest_directory(
    collection: chromadb.Collection,
    pdf_dir: str,
    splitter: RecursiveCharacterTextSplitter,
) -> None:
    paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    current_sources = {os.path.basename(p) for p in paths}

    stored_metas = collection.get(include=["metadatas"])["metadatas"] or []
    indexed_sources = {str(meta["source"]) for meta in stored_metas}
    for source in indexed_sources - current_sources:
        collection.delete(where={"source": source})
        print(f"{source}: removed from database (file deleted)")

    if not paths:
        print(f"No PDFs found in {pdf_dir}/")
    for path in paths:
        ingest_pdf(collection, path, splitter)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class CorpusIndex:
    bm25: bm25s.BM25 | None
    ids: list[str]
    by_id: dict[str, tuple[str, Metadata]]


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
    return CorpusIndex(bm25=bm25, ids=ids, by_id=by_id)


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


def retrieve(
    collection: chromadb.Collection,
    corpus_index: CorpusIndex,
    query: str,
    top_n: int = 3,
    sources: list[str] | None = None,
    mode: str = "hybrid",
) -> list[tuple[str, float, Metadata]]:
    if mode == "dense":
        results = dense_search(collection, query, top_n, sources)
    elif mode == "bm25":
        results = bm25_search(corpus_index, query, top_n, sources)
    else:
        return hybrid_retrieve(collection, corpus_index, query, top_n, sources)

    return [
        (corpus_index.by_id[doc_id][0], score, corpus_index.by_id[doc_id][1])
        for doc_id, score in results
    ]


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


def reindex(
    collection: chromadb.Collection,
    source: str,
    splitter: RecursiveCharacterTextSplitter,
) -> CorpusIndex:
    path = os.path.join(PDF_PATH, source)
    collection.delete(where={"source": source})
    print(f"Deleted {source}, re-ingesting...")
    if os.path.exists(path):
        ingest_pdf(collection, path, splitter)
    else:
        print(f"{source} not found in {PDF_PATH}/")
    return build_corpus_index(collection)


def main() -> None:
    os.makedirs(PDF_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    ingest_directory(collection, PDF_PATH, splitter)
    corpus_index = build_corpus_index(collection)

    session = Session()
    while True:
        try:
            input_query = input(
                f"\n{MAGENTA}Ask a question (or 'q' to quit):{RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not input_query:
            continue
        if input_query.lower() in ("q", "quit", "exit"):
            break

        if handle_command(input_query, session):
            if session.pending_reindex:
                corpus_index = reindex(collection, session.pending_reindex, splitter)
                session.pending_reindex = None
            continue

        if collection.count() == 0:
            print("Database is empty. Add PDFs to the docs/ directory.")
            continue

        retrieved_knowledge = retrieve(
            collection,
            corpus_index,
            input_query,
            session.top_n,
            session.sources,
            session.mode,
        )
        if not retrieved_knowledge:
            print("No matching results found.")
            continue

        print(f"\n{CYAN}Retrieved knowledge ({session.mode}):{RESET}")
        for i, (chunk, score, meta) in enumerate(retrieved_knowledge, start=1):
            location = f"{meta['source']}, p. {meta['pages']}"
            print(
                f"{YELLOW} - [{i}] ({SCORE_LABELS[session.mode]}: {score:.3f}) [{location}]{RESET}\n{chunk}\n"
            )

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": build_prompt(retrieved_knowledge),
                },
                {"role": "user", "content": sanitize(input_query)},
            ],
            stream=True,
            think=False,
            options={"temperature": session.temperature},
        )
        print(f"{BOLD}{GREEN}Chatbot response:{RESET}")
        for chunk in stream:
            print(chunk.message.content, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
