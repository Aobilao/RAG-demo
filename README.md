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
2. Put the PDFs you want to query into the `docs/` directory (created automatically if missing).
3. Run the chatbot:
   ```
   python rag.py
   ```

On startup, any new or changed PDFs in `docs/` are automatically indexed into a local Chroma database (`chroma_db/`). Already-indexed, unchanged files are skipped. PDFs removed from `docs/` are automatically dropped from the index.

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
- Reranking downloads and runs `BAAI/bge-reranker-base` locally on first use; if it fails to load, retrieval falls back to the first-stage ranking.
- Add or update a PDF while the chatbot is running by editing `docs/` and restarting, or use `/reindex=<filename.pdf>` to pick it up without restarting.
