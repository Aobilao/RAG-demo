# RAG

A RAG chatbot that answers questions from your own PDFs, and searches your own
pictures by describing what's in them. Command line or [web UI](#web-ui).

## Requirements

Python 3.11+ and [Ollama](https://ollama.com) running locally:

```
ollama pull bge-m3
ollama pull qwen3.5:4b
ollama pull qwen2.5vl:3b    # image search only
```

Optional, for better OCR of scanned Vietnamese documents:

```
sudo apt install tesseract-ocr tesseract-ocr-vie
```

## Setup

```
pip install -r requirements.txt
```

Put PDFs in `docs/` and pictures in `images/` (both created automatically), then run
from the repository root, since all paths resolve relative to the working directory:

```
python -m rag
```

Both directories are scanned at startup and new or changed files are indexed into
`chroma_db/`; unchanged ones are skipped, deleted ones are dropped, and an ingest
interrupted partway is detected and redone. Use `/refresh=docs` or `/refresh=images`
to pick up files added while the chatbot is running. The first run downloads Docling's
models (a few hundred MB) and is slow.

## Usage

Ask a question at the prompt. The answer carries inline citations `[1]`, `[2]` that
point at the retrieved passages printed above it. Type `q`, `quit`, `exit`, or press
Ctrl+D to leave.

### Commands

| Command | Description |
|---|---|
| `/status` | Show current session settings |
| `/mode=<dense\|bm25\|hybrid>` | Retrieval mode (default: `hybrid`) |
| `/top_n=<N>` | Passages to retrieve and use as context (default: `3`) |
| `/temperature=<N>` | LLM sampling temperature (default: `0.0`) |
| `/sources=<file1,file2,...>` or `/sources=all` | Restrict retrieval to specific PDFs, or reset to all |
| `/rerank=<on\|off>` | Rerank retrieved passages (default: `off`) |
| `/rerank_k=<N>` | First-stage candidates passed to the reranker (default: `30`) |
| `/reindex=<filename.pdf>` | Re-ingest one PDF without restarting |
| `/refresh=<docs\|images>` | Rescan a directory for new and changed files |
| `/find=<description>` | Search `images/` by description, see [Image search](#image-search) |

Booleans accept `on`/`true`/`1`/`yes` or `off`/`false`/`0`/`no`. `/sources` only takes
filenames that are actually indexed, and lists them if you get one wrong.

`/rerank=on` downloads `BAAI/bge-reranker-base` (~1.1GB) the first time, then reads it
from `~/.cache/huggingface/hub`. If it fails to load, retrieval falls back to the
first-stage ranking.

## Web UI

```
pip install -r requirements-web.txt
python -m rag.web
```

Serves http://127.0.0.1:8000. Sources appear first and the answer streams in
underneath; inline citations are clickable and scroll to the passage they cite. The
**Tìm ảnh** tab is image search, with each match's description and score next to a
thumbnail you can open full size.

## Image search

Put pictures in `images/` and describe what you want with `/find=<description>`, or use
the **Tìm ảnh** tab. It returns your own pictures, not generated ones.

A vision model describes each image in Vietnamese once at index time, leading with the
names of any famous people it recognizes, and queries are ordinary text search over
those descriptions. Describing is the slow part, roughly ten seconds per image on CPU,
so the first startup with a full `images/` directory takes a while; afterwards each
image is skipped unless it changes. Changing `VISION_MODEL` re-describes everything.

`qwen2.5vl:3b` is the default because it reliably follows a Vietnamese prompt.
`moondream` returned an empty caption for every image, worth knowing if you swap the
model and indexing suddenly produces nothing.

## PDF extraction

PDFs are read with [Docling](https://github.com/docling-project/docling), which keeps
headings, tables and reading order, and tags each chunk with its page numbers for
citations. Pages are checked for a usable text layer:

| Pages without a text layer | Behaviour |
|---|---|
| under 20% | read directly, no OCR |
| 20% to 80% | OCR fills in the image-only regions |
| over 80% | treated as scanned, every page is OCR'd |

Text typeset in pre-Unicode Vietnamese fonts (TCVN3/VNI) decodes to garbage, so it is
detected and OCR'd as well. OCR prefers Tesseract (`vie`+`eng`) and falls back to
EasyOCR (`vi`+`en`) when the system packages are missing. It costs several seconds per
page but runs only when a document is first indexed.

Bump `EXTRACTOR_VERSION` in `rag/extractor.py` after changing extraction to make
existing documents re-index instead of being skipped as unchanged.

## Tests

```
python -m unittest -v
```

68 tests in under a second, needing no Ollama, no Docling, no GPU and no network.

`tests/test_rag.py` covers pure logic: page attribution, rank fusion, marker
bookkeeping and command parsing. `tests/test_lifecycle.py` and `tests/test_images.py`
run against a real Chroma collection in a temporary directory, replacing only
extraction, captioning and embedding. `tests/test_web.py` covers event framing and the
traversal guard on the file routes.

Whether a vision model actually describes an image well is not something a fast test
suite can judge, so that part and the endpoints needing Ollama are exercised by hand.

## Layout

```
rag/           the package
  __main__.py  python -m rag       the chatbot, image search included
  images.py                        image indexing and search, used by both front ends
  web/         python -m rag.web   the web interface
tests/         python -m unittest
docs/          PDFs to index (gitignored)
images/        pictures to search (gitignored)
chroma_db/     the vector database (gitignored)
```
