from dataclasses import dataclass

MODES = ("dense", "bm25", "hybrid")
TRUTHY = ("on", "true", "1", "yes")
FALSY = ("off", "false", "0", "no")


@dataclass
class Session:
    top_n: int = 3
    temperature: float = 0.0
    sources: list[str] | None = None
    mode: str = "hybrid"
    rerank: bool = False
    rerank_k: int = 30
    pending_reindex: str | None = None


def set_top_n(session: Session, value: str) -> None:
    try:
        top_n = int(value)
        if top_n < 1:
            raise ValueError
    except ValueError:
        print("top_n must be a positive integer")
        return
    session.top_n = top_n
    print(f"top_n set to {session.top_n}")


def set_temperature(session: Session, value: str) -> None:
    try:
        session.temperature = float(value)
    except ValueError:
        print("Temperature must be a number")
        return
    print(f"Temperature set to {session.temperature}")


def set_mode(session: Session, value: str) -> None:
    mode = value.lower()
    if mode not in MODES:
        print(f"Mode must be one of: {', '.join(MODES)}")
        return
    session.mode = mode
    print(f"Retrieval mode set to {session.mode}")


def set_rerank(session: Session, value: str) -> None:
    flag = value.lower()
    if flag in TRUTHY:
        session.rerank = True
    elif flag in FALSY:
        session.rerank = False
    else:
        print(f"rerank must be one of: {', '.join(TRUTHY + FALSY)}")
        return
    print(f"Reranking {'enabled' if session.rerank else 'disabled'}")


def set_rerank_k(session: Session, value: str) -> None:
    try:
        rerank_k = int(value)
        if rerank_k < 1:
            raise ValueError
    except ValueError:
        print("rerank_k must be a positive integer")
        return
    session.rerank_k = rerank_k
    print(f"rerank_k set to {session.rerank_k}")


def set_sources(session: Session, value: str) -> None:
    if value.lower() == "all":
        session.sources = None
        print("Retrieving from all sources")
        return
    session.sources = [s.strip() for s in value.split(",")]
    print(f"Retrieving from: {session.sources}")


def set_pending_reindex(session: Session, value: str) -> None:
    if not value:
        print("Usage: /reindex=<filename.pdf>")
        return
    session.pending_reindex = value


def print_status(session: Session) -> None:
    sources = "all" if session.sources is None else ", ".join(session.sources)
    print(
        "Current session settings:\n"
        f"  mode        = {session.mode}\n"
        f"  rerank      = {'on' if session.rerank else 'off'}\n"
        f"  rerank_k    = {session.rerank_k}\n"
        f"  top_n       = {session.top_n}\n"
        f"  temperature = {session.temperature}\n"
        f"  sources     = {sources}"
    )


def handle_command(text: str, session: Session) -> bool:
    if not text.startswith("/"):
        return False

    name, _, value = text[1:].partition("=")
    name, value = name.strip().lower(), value.strip()

    if name == "top_n":
        set_top_n(session, value)
    elif name == "temperature":
        set_temperature(session, value)
    elif name == "mode":
        set_mode(session, value)
    elif name == "rerank":
        set_rerank(session, value)
    elif name == "rerank_k":
        set_rerank_k(session, value)
    elif name == "sources":
        set_sources(session, value)
    elif name == "reindex":
        set_pending_reindex(session, value)
    elif name == "status":
        print_status(session)
    else:
        print(
            "Unknown command. Available: /top_n=, /temperature=, /mode=, "
            "/rerank=, /rerank_k=, /sources=, /reindex=, /status"
        )
    return True
