import os
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer

RERANKER_REPO = "BAAI/bge-reranker-base"
RERANKER_ONNX = "onnx/model.onnx"
RERANKER_MAX_LENGTH = 512
RERANKER_BATCH_SIZE = 8
PAD_TOKEN_ID = 1
PAD_TOKEN = "<pad>"


@dataclass
class Reranker:
    session: ort.InferenceSession
    tokenizer: Tokenizer
    input_names: set[str]


_reranker: Reranker | None = None
_load_failed = False


def _download(filename: str) -> str:
    try:
        return hf_hub_download(RERANKER_REPO, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(RERANKER_REPO, filename)


def load_reranker() -> Reranker | None:
    global _reranker, _load_failed
    if _reranker is not None or _load_failed:
        return _reranker

    try:
        print(f"Loading reranker {RERANKER_REPO}...")
        tokenizer_path = _download("tokenizer.json")
        model_path = _download(RERANKER_ONNX)

        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_truncation(
            max_length=RERANKER_MAX_LENGTH, strategy="longest_first"
        )
        tokenizer.enable_padding(pad_id=PAD_TOKEN_ID, pad_token=PAD_TOKEN)

        inf_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        _reranker = Reranker(
            session=inf_session,
            tokenizer=tokenizer,
            input_names={i.name for i in inf_session.get_inputs()},
        )
    except Exception as exc:
        print(f"Reranker unavailable: {exc}")
        _load_failed = True
        return None

    return _reranker


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def rerank_scores(reranker: Reranker, query: str, docs: list[str]) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(docs), RERANKER_BATCH_SIZE):
        batch = docs[start : start + RERANKER_BATCH_SIZE]
        encodings = reranker.tokenizer.encode_batch([(query, doc) for doc in batch])

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in reranker.input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        logits = reranker.session.run(None, inputs)[0]
        scores.extend(sigmoid(np.asarray(logits).reshape(-1)).tolist())

    return scores
