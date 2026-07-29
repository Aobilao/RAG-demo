# RAG

A command-line RAG chatbot that answers questions from your own PDF documents.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally, with these models pulled:
  ```
  ollama pull bge-m3
  ollama pull qwen3.5:4b
  ```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Optionally install Tesseract for better OCR of scanned Vietnamese documents (see [PDF extraction](#pdf-extraction)):
   ```
   sudo apt install tesseract-ocr tesseract-ocr-vie
   ```
3. Put the PDFs you want to query into the `docs/` directory (created automatically if missing).
4. Run the chatbot:
   ```
   python rag.py
   ```

On startup, any new or changed PDFs in `docs/` are automatically indexed into a local Chroma database (`chroma_db/`). Already-indexed, unchanged files are skipped. PDFs removed from `docs/` are automatically dropped from the index.

The first run downloads Docling's layout and table models (a few hundred MB) and is noticeably slower than later ones.

## PDF extraction

PDFs are read with [Docling](https://github.com/docling-project/docling), which recovers headings, tables and reading order and drops running headers and footers. Extracted pages are kept as markdown, and each chunk is tagged with the page(s) it came from for citations.

Docling reads pages through its pypdfium backend rather than the default parser, which splits accented characters into separate tokens on some Vietnamese fonts (`m ộ t` instead of `một`). The trade-off is slightly coarser table structure.

Documents are checked page by page for a usable text layer:

| Pages without a text layer | Behaviour |
|---|---|
| under 20% | text layer is read directly, no OCR |
| 20%–80% | OCR fills in the image-only regions |
| over 80% | treated as scanned; every page is OCR'd |

A text layer can also be present but unreadable. Documents typeset in pre-Unicode Vietnamese fonts (TCVN3/VNI, e.g. `.VnTime`) carry no usable character mapping, so `biện chứng` extracts as `biÖn chøng` — text that matches neither Vietnamese nor English queries and is therefore dead weight in the index. Such a text layer is detected by how much of it decodes to the byte range those fonts use for diacritics (`LEGACY_FONT_CHARS` in `extractor.py`; affected documents sit near 4%, clean ones below 0.01%) and is discarded in favour of OCR'ing every page.

OCR is tuned for Vietnamese and picks the first engine available:

1. **Tesseract** (`vie`+`eng`) — most accurate, needs the system packages above.
2. **EasyOCR** (`vi`+`en`) — installed from `requirements.txt`, used automatically when Tesseract is missing.

RapidOCR ships with Docling but is deliberately unused: its Latin recognition dictionary contains no Vietnamese diacritics, so it mangles accented text.

OCR is slow — expect several seconds per page — but it only runs when a document is first indexed. Extraction runs on the GPU when torch reports one, otherwise on the CPU.

Chunks record which version of the extraction pipeline produced them (`EXTRACTOR_VERSION` in `extractor.py`). Changing extraction and bumping that constant makes existing documents re-index automatically, rather than being skipped as unchanged.

## Usage

Type a question at the prompt and press Enter to get an answer grounded in your documents, with inline citations `[1]`, `[2]`, ... pointing to the retrieved passages shown above the answer.

Type `q`, `quit`, or `exit` (or press Ctrl+D / Ctrl+C) to leave.

### Session commands

Commands start with `/` and are entered at the same prompt as questions.

| Command | Description |
|---|---|
| `/status` | Show current session settings |
| `/mode=<dense\|bm25\|hybrid>` | Set retrieval mode (default: `hybrid`) |
| `/top_n=<N>` | Number of passages to retrieve and use as context (default: `3`) |
| `/temperature=<N>` | LLM sampling temperature (default: `0.0`) |
| `/sources=<file1,file2,...>` or `/sources=all` | Restrict retrieval to specific PDFs by filename, or reset to all |
| `/rerank=<on\|off>` | Enable/disable reranking of retrieved passages (default: `off`) |
| `/rerank_k=<N>` | Number of first-stage candidates to pass to the reranker (default: `30`) |
| `/reindex=<filename.pdf>` | Force a specific PDF in `docs/` to be deleted and re-ingested |

Boolean values for `/rerank` accept `on`/`true`/`1`/`yes` or `off`/`false`/`0`/`no`.

Notes:
- Reranking downloads and runs `BAAI/bge-reranker-base` (~1.1GB) locally the first time `/rerank=on` is used; if it fails to load, retrieval falls back to the first-stage ranking. Later runs read the model from the local Hugging Face cache (`~/.cache/huggingface/hub`) instead of re-downloading it.
- Add or update a PDF while the chatbot is running by editing `docs/` and restarting, or use `/reindex=<filename.pdf>` to pick it up without restarting.
