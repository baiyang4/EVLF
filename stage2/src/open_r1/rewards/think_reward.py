"""ThinkAlign clinical reasoning-alignment reward for GRPO.

Sentence pairs use directional semi-relaxed token OT and a bidirectional
contradiction gate. Completeness and faithfulness are combined with a harmonic
mean, with same-dataset distractors providing a contrastive baseline.

Environment:
    DEBERTA_SERVICE_URLS  comma-separated NLI service replicas, sharded by rank
    DEBERTA_SERVICE_URL   fallback NLI endpoint, default http://127.0.0.1:8001
    THINK_REWARD_WEIGHT   output scaling, default 0.5
    THINK_ENCODER         frozen biomedical token encoder
    THINK_LAMBDA          OT marginal relaxation, default 0.05
    THINK_GAMMA           contradiction-gate exponent, default 2.0
    THINK_REF_POOL        training JSON directory for distractor references;
                          unset disables the contrastive baseline
    THINK_NUM_DISTRACTORS maximum distractors per reference, default 4
"""

import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .deberta_client import DebertaClient
import re as _re_think

def extract_think_content(completion) -> str:
    """Extract text inside <think>...</think> from a completion."""
    if isinstance(completion, list) and len(completion) > 0:
        content = completion[0].get("content", "")
    elif isinstance(completion, dict):
        content = completion.get("content", "")
    else:
        content = str(completion)
    match = _re_think.search(r"<think>(.*?)</think>", content, _re_think.DOTALL)
    return match.group(1).strip() if match else ""

# The `think_align` package is vendored under src/ alongside `open_r1`; make src/ importable.
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.isdir(os.path.join(_SRC_ROOT, "think_align")) and _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from think_align import (  # noqa: E402
    ThinkAlignConfig,
    ThinkAlignSentenceAligner,
    FrozenHFTokenEncoder,
    IDFTokenWeighter,
    OTConfig,
)
from think_align.nli import BaseContradictionScorer  # noqa: E402
from think_align.text import clean_reasoning, pretokenize, split_sentences  # noqa: E402

DEFAULT_ENCODER = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
_WORDLIKE = re.compile(r"[A-Za-z0-9]")

_aligner: Optional[ThinkAlignSentenceAligner] = None
_scorer: Optional["PrefetchedContradictionScorer"] = None
_client: Optional[DebertaClient] = None
_ref_pool: Optional[Dict[str, List[str]]] = None
_distractor_cache: Dict[Tuple[str, str], List[str]] = {}


class PrefetchedContradictionScorer(BaseContradictionScorer):
    """Serves contradiction probabilities from a table filled by one batched call.

    ThinkAlignSentenceAligner.score() asks for its sentence pairs one sample at a time.
    Left alone that is one HTTP round trip per rollout; instead every pair in the
    batch is fetched up front and looked up here, so the whole reward costs a
    single request per rank per step. A miss is a bug in the pre-pass (the pair
    set must be built with the same splitter), never something to paper over with
    a default, so it raises.
    """

    def __init__(self) -> None:
        self.table: Dict[Tuple[str, str], float] = {}

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> torch.Tensor:
        try:
            values = [self.table[pair] for pair in pairs]
        except KeyError as exc:
            raise RuntimeError(
                f"contradiction score missing for pair {exc.args[0]!r}; the prefetch "
                "pass and ThinkAlign disagree about sentence splitting"
            ) from exc
        return torch.tensor(values, dtype=torch.float32).clamp(0.0, 1.0)


def get_client() -> DebertaClient:
    """One client per rank, sharded across service replicas by LOCAL_RANK.

    Rank r uses entry r % len(DEBERTA_SERVICE_URLS). If the replica list is
    unset, use DEBERTA_SERVICE_URL. Service health is checked on initialization.
    """
    global _client
    urls = [u.strip() for u in os.getenv("DEBERTA_SERVICE_URLS", "").split(",") if u.strip()]
    if not urls:
        urls = [os.getenv("DEBERTA_SERVICE_URL", "http://127.0.0.1:8001")]
    rank = int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))
    url = urls[rank % len(urls)]

    if _client is None:
        _client = DebertaClient(
            service_url=url,
            timeout=int(os.getenv("DEBERTA_TIMEOUT", "60")),
            max_batch_size=int(os.getenv("DEBERTA_BATCH", "64")),
        )
        if not _client.check_health():
            # a warn-and-continue here would silently swallow every later
            # failure into 0.0, which silently turns the run into an
            # accuracy-only experiment that looks exactly like a successful one.
            raise RuntimeError(
                f"DeBERTa NLI service at {url} (rank {rank} of {len(urls)} replicas) is "
                "not healthy. Start it on the l40 box (CUDA_VISIBLE_DEVICES=<gpu> python "
                "deberta_service.py --port <port> --host 127.0.0.1) and open the tunnel "
                "(ssh -N -L <port>:127.0.0.1:<port> l40)."
            )
    return _client


def get_aligner() -> Tuple[ThinkAlignSentenceAligner, PrefetchedContradictionScorer]:
    global _aligner, _scorer
    if _aligner is None:
        _scorer = PrefetchedContradictionScorer()
        device = (
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        encoder = FrozenHFTokenEncoder(
            os.getenv("THINK_ENCODER", DEFAULT_ENCODER),
            device=device,
            weighter=IDFTokenWeighter(),
            # References repeat across all G rollouts of a group, so the cache
            # absorbs most of the encoder cost.
            cache_size=int(os.getenv("THINK_CACHE", "32768")),
        )
        config = ThinkAlignConfig(
            ot=OTConfig(marginal_relaxation=float(os.getenv("THINK_LAMBDA", "0.05"))),
            contradiction_gamma=float(os.getenv("THINK_GAMMA", "2.0")),
        )
        _aligner = ThinkAlignSentenceAligner(encoder, _scorer, config)
        print(
            f"[think_align] encoder on {device}, lambda={config.ot.marginal_relaxation}, "
            f"gamma={config.contradiction_gamma}, nli={os.getenv('DEBERTA_SERVICE_URL')}",
            flush=True,
        )
    return _aligner, _scorer


def _split(text: str) -> List[str]:
    cfg = ThinkAlignConfig().text
    return split_sentences(
        text,
        split_on_semicolon=cfg.split_on_semicolon,
        max_sentences=cfg.max_generated_sentences,
    )


def _encodable(text: str) -> bool:
    """Require ASCII alphabetic content in every sentence used by ThinkAlign.

    Validation uses the aligner's sentence splitter. Samples with unsupported
    or punctuation-only sentences receive zero reward without calling the encoder.
    """
    sentences = _split(text)
    return bool(sentences) and all(
        # Punctuation may tokenize but does not provide usable word content.
        any(_WORDLIKE.search(word) for word in pretokenize(sentence))
        for sentence in sentences
    )


def get_ref_pool() -> Dict[str, List[str]]:
    """Reference rationales grouped by source dataset, for the contrastive baseline.

    Load training JSON files from THINK_REF_POOL so distractors can be sampled
    from the same dataset as each reference.
    """
    global _ref_pool
    if _ref_pool is None:
        root = os.getenv("THINK_REF_POOL", "")
        pool: Dict[str, List[str]] = {}
        if root and os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                if not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(root, name), encoding="utf-8") as f:
                        rows = json.load(f)
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(rows, list):
                    continue
                for r in rows:
                    # The dir also holds _manifest.json, whose rows are lists. Skip
                    # anything that is not a sample rather than assuming the layout.
                    if not isinstance(r, dict):
                        continue
                    ds = r.get("dataset")
                    raw = r.get("reasoning_content_gpt") or r.get("cot") or ""
                    if ds and raw.strip():
                        # clean_reasoning here too: refs[i] is cleaned, and a distractor
                        # that went through a different pipeline is not comparable to it.
                        pool.setdefault(ds, []).append(clean_reasoning(raw))
        _ref_pool = {k: sorted(set(v)) for k, v in pool.items()}
        print(
            f"[think_align] distractor pool: {len(_ref_pool)} datasets, "
            f"{sum(len(v) for v in _ref_pool.values())} unique references"
            + ("" if _ref_pool else f"  -- EMPTY, THINK_REF_POOL={root!r}; the reward "
                                    "falls back to plain (non-contrastive) ThinkAlign"),
            flush=True,
        )
    return _ref_pool


def _distractors(dataset: Optional[str], own: str, k: int) -> List[str]:
    """k references of *other* images from the same dataset. Stable per (dataset, ref).

    Stability matters: all G rollouts of a group share one reference, so they must also
    share one distractor set. If each rollout drew its own, the baseline would vary for
    reasons unrelated to the rollout's quality, and that noise lands directly in the
    z-scored advantage. Seeded from the reference text rather than a global RNG so it is
    also identical across ranks and across a resume.
    """
    if not dataset:
        return []
    key = (dataset, own)
    if key in _distractor_cache:
        return _distractor_cache[key]
    cands = get_ref_pool().get(dataset) or []
    out: List[str] = []
    if len(cands) > 1:
        seed = int.from_bytes(
            hashlib.blake2b(f"{dataset}\x00{own}".encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        rng = random.Random(seed)
        for c in rng.sample(cands, min(len(cands), 4 * k)):
            if c == own or c in out or not _encodable(c):
                continue
            out.append(c)
            if len(out) == k:
                break
    _distractor_cache[key] = out
    return out


def think_reward(completions, cot_gpt=None, cot_gemini=None, **kwargs) -> List[float]:
    """Contrastive sentence-alignment reward over the <think> block. One value per completion.

        baseline = mean_k ThinkAlign(think, other_ref_k)
        reward = weight * (ThinkAlign(think, own_ref) - baseline)

    References use cot_gpt with cot_gemini as a fallback. Same-dataset
    distractors reduce the reward for generic reasoning shared across images.
    The baseline is zero when no distractors are available. Missing or
    unsupported reasoning receives zero reward; valid rewards may be negative.
    """
    weight = float(os.getenv("THINK_REWARD_WEIGHT", "0.5"))
    num_distractors = int(os.getenv("THINK_NUM_DISTRACTORS", "4"))
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    n = len(completions)

    if cot_gpt is None and cot_gemini is None:
        return [0.0] * n

    raw_gpt = cot_gpt or [""] * n
    raw_gem = cot_gemini or [""] * n
    refs = [clean_reasoning(g) if g else (clean_reasoning(m) if m else "")
            for g, m in zip(raw_gpt, raw_gem)]
    preds = [extract_think_content(c) for c in completions]

    # A completion with no <think> scores 0 rather than being judged on its
    # <answer>: clean_reasoning falls back to the whole string when the tag is
    # absent, which would quietly score the answer text as if it were reasoning.
    usable = [
        i for i in range(n)
        if preds[i] and refs[i] and _encodable(preds[i]) and _encodable(refs[i])
    ]
    if not usable:
        return [0.0] * n

    aligner, scorer = get_aligner()

    datasets = kwargs.get("dataset") or [None] * n
    distractors = {
        i: _distractors(datasets[i], refs[i], num_distractors) for i in usable
    }

    # Prefetch every contradiction probability the aligner will ask for, in both
    # directions, deduplicated -- within a group the same reference sentence is
    # paired against G rollouts, so dedup typically removes most of the payload.
    # The distractor pairs go in the same request: a miss here is a hard error in
    # PrefetchedContradictionScorer, not a silent default.
    wanted = set()
    for i in usable:
        p_sents = _split(preds[i])
        for target in [refs[i]] + distractors[i]:
            for r_sent in _split(target):
                for p_sent in p_sents:
                    wanted.add((p_sent, r_sent))
                    wanted.add((r_sent, p_sent))
    pairs = sorted(wanted)
    scores = get_client().get_contradiction_scores(
        [a for a, _ in pairs], [b for _, b in pairs]
    )
    scorer.table = dict(zip(pairs, scores))

    rewards = [0.0] * n
    diags = {}
    baselines = {}
    for i in usable:
        d = aligner.score(preds[i], refs[i])
        # No pool entry for this dataset (or a dataset with a single row) -> baseline 0,
        # i.e. plain ThinkAlign. Degrading to the old behaviour is safe; guessing is not.
        baselines[i] = (
            sum(aligner.score(preds[i], x).reward for x in distractors[i]) / len(distractors[i])
            if distractors[i] else 0.0
        )
        # Deliberately not clamped to >= 0. A think block that matches other images of this
        # dataset better than its own earns a negative reward, and that is the correct
        # gradient -- clamping would restore exactly the free lunch this term exists to
        # remove. GRPO z-scores within the group, so the sign costs nothing downstream.
        rewards[i] = (d.reward - baselines[i]) * weight
        diags[i] = d

    if debug_mode and (log_path := os.getenv("LOG_PATH")):
        i = usable[0]
        d = diags[i]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"------------- {datetime.now():%d-%H-%M-%S-%f} ThinkAlign Reward -------------\n")
            with_distr = sum(1 for j in usable if distractors[j])
            f.write(f"Usable: {len(usable)}/{n}   NLI pairs (deduped): {len(pairs)}   "
                    f"with distractors: {with_distr}/{len(usable)}\n")
            f.write(f"think ({len(d.generated_sentences)} sents): {preds[i][:200]}\n")
            f.write(f"ref   ({len(d.reference_sentences)} sents): {refs[i][:150]}\n")
            f.write(f"comp={d.completeness:.3f}  faith={d.faithfulness:.3f}  "
                    f"own={d.reward:.3f}  base={baselines[i]:.3f}  "
                    f"contrast={d.reward - baselines[i]:+.3f}  "
                    f"maxContra={float(d.contradiction_matrix.max()):.3f}\n")
            f.write(f"Rewards (scaled): {[f'{r:.3f}' for r in rewards[:8]]}\n")

    return rewards
