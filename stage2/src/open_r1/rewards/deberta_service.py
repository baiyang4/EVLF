"""
DeBERTa NLI FastAPI Service for Thinking Reward.

Deploy on a remote server with GPU:
    pip install fastapi uvicorn transformers torch
    CUDA_VISIBLE_DEVICES=0 python deberta_service.py --port 8000

Endpoints:
    GET  /health            → health check
    POST /entailment_batch  → batch entailment scoring
"""

import argparse
from typing import List

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="DeBERTa NLI Service")

model = None
tokenizer = None
device = None
CONTRADICTION_LABEL_ID = None


class TextPair(BaseModel):
    text1: str
    text2: str


class EntailmentRequest(BaseModel):
    pairs: List[TextPair]


class EntailmentResponse(BaseModel):
    scores: List[float]


@app.on_event("startup")
def load_model():
    global model, tokenizer, device, CONTRADICTION_LABEL_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "microsoft/deberta-v2-xlarge-mnli"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    # Resolve rather than hardcode: this checkpoint happens to use
    # {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}, but swapping in a MedNLI
    # checkpoint later must not silently start returning the wrong head.
    id2label = getattr(model.config, "id2label", {}) or {}
    for key, value in id2label.items():
        if "contrad" in str(value).lower():
            CONTRADICTION_LABEL_ID = int(key)
            break
    if CONTRADICTION_LABEL_ID is None:
        raise RuntimeError(f"no contradiction label in id2label={id2label}")
    print(f"Model loaded successfully. contradiction label id = {CONTRADICTION_LABEL_ID} "
          f"({id2label[CONTRADICTION_LABEL_ID]})")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "contradiction_label_id": CONTRADICTION_LABEL_ID,
    }


def _score(pairs, label_id: int, max_length: int) -> List[float]:
    texts1 = [p.text1 for p in pairs]
    texts2 = [p.text2 for p in pairs]

    # Process in chunks to avoid OOM
    max_batch = 32
    all_scores = []
    for i in range(0, len(texts1), max_batch):
        batch_t1 = texts1[i : i + max_batch]
        batch_t2 = texts2[i : i + max_batch]

        inputs = tokenizer(
            batch_t1, batch_t2,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        all_scores.extend(probs[:, label_id].cpu().tolist())

    return all_scores


@app.post("/entailment_batch", response_model=EntailmentResponse)
def entailment_batch(request: EntailmentRequest):
    if not request.pairs:
        return EntailmentResponse(scores=[])
    # index 2 = entailment
    return EntailmentResponse(scores=_score(request.pairs, 2, 512))


@app.post("/contradiction_batch", response_model=EntailmentResponse)
def contradiction_batch(request: EntailmentRequest):
    """P(contradiction) for the ThinkAlign logic gate.

    ThinkAlign only ever asks this model whether two clinical statements are
    incompatible -- it never uses the entailment head, because measured on real
    rollouts MNLI entailment sits near the floor on medical text (P(entail)=0.033
    for a textbook-valid inference). Contradiction is the one direction it is
    reliable in: on laterality/polarity flips it fires at p=0.82.

    Sentences here are single clinical statements, so 256 tokens is ample and
    halves the attention cost relative to /entailment_batch.
    """
    if not request.pairs:
        return EntailmentResponse(scores=[])
    label_id = CONTRADICTION_LABEL_ID
    if label_id is None:
        raise RuntimeError("contradiction label id was not resolved at startup")
    return EntailmentResponse(scores=_score(request.pairs, label_id, 256))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
