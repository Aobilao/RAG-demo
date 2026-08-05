import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def main() -> None:
    print(f"RAG web interface: http://{HOST}:{PORT}")
    uvicorn.run("rag.web.server:app", host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
