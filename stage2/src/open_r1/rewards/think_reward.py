"""ThinkAlign reasoning reward (clinical reasoning alignment).

The outer framework
is unchanged -- split both sides into sentences, build completeness/faithfulness
matrices, take the harmonic mean -- only the sentence-pair scorer differs:

    directional semi-relaxed token OT  x  bidirectional contradiction gate

Why, measured on 315 reconstructed 8-rollout groups (2520 real cls rollouts):

                        median   within-group std   correct-vs-wrong gap
    NLI sentence-F1      0.071        0.099               +0.042
    ThinkAlign                0.740        0.206               +0.168

Within-group std is the only thing GRPO can use (advantages are z-scored inside
each group), and it doubles. The gap quadrupling means the term pulls the same
way as accuracy, so it is a dense supplement rather than a competing objective --
39.7% of groups are accuracy-degenerate (all G rollouts equally right or wrong)
and contribute zero gradient without it.

Config deltas from the ThinkAlign repo defaults, both measured rather than guessed:
  * marginal_relaxation 1.0 -> 0.05. At 1.0, rho = lam/(lam+eps) = 0.952, i.e. the
    "relaxed" marginal is 95% hard and the two directions solve nearly the same
    problem. 0.05 gives identical quality 44% faster.
  * contradiction_gamma 1.0 -> 2.0. Ceiling is untouched (0.896 -> 0.895) while the
    floor drops (other-reference 0.316 -> 0.238); gap and within-group std each
    gain 14%. gamma=3.0 gains a further ~3% but extrapolates further on an NLI
    model already known to be out of domain, so 2.0 is the stopping point.
  * No corpus IDF. Built from all 184,258 reference rationales it changed every
    measured number in the third decimal (gap +0.1679 -> +0.1677): OT masses are
    normalised per sentence, and a ~20-word clinical sentence is already almost
    all content words.

Deliberately NOT done: masking the answer string out of both sides. That was
tested as a diagnostic, not adopted as a fix -- the gap survives masking
(+0.168 -> +0.163), which is evidence the signal does not come from answer
leakage, so there is nothing to correct. Naming the organ you just described is
correct clinical prose, and masking corrupts the encoder context of the whole
sentence. The leak is monitored instead (see the answer-coverage diag in
by_trainer) so reward hacking would be visible if it ever started.

Environment:
    DEBERTA_SERVICE_URL   NLI service (contradiction head only). Default
                          http://127.0.0.1:8001, normally an ssh -L tunnel to the
                          l40 box so the endpoint is never network-exposed.
    THINK_REWARD_WEIGHT   output scaling, default 0.5 (~13% of within-group variance)
    THINK_ENCODER         frozen biomedical token encoder
    THINK_REF_POOL        dir of training .json files, read for the contrastive baseline's
                          distractor references. Unset -> plain ThinkAlign, which is known to be
                          hackable by a per-modality template; see think_reward's docstring.
    THINK_NUM_DISTRACTORS K, default 4. Costs K+1 OT solves per rollout instead of 1.
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

    A single replica is the dominant cost of the contrastive reward, not the OT solve.
    Measured 2026-07-22: one L40S serves 240 pairs/s and fully serializes concurrent
    callers (8 simultaneous requests take 8.5x one request), while the GPU sits at
    95-98% utilisation -- so it is compute-bound and neither a larger DEBERTA_BATCH nor
    a bigger service-side batch helps. K=4 distractors multiply the payload 5.13x
    (deduped pairs: median 60 for plain ThinkAlign, 308 for contrastive), and all of it
    queues behind the other seven ranks. That is where v2's 38.38 s/it came from
    against accuracy-only's 20.40.

    One step (8 ranks x 2 calls x 308 pairs), measured end to end:
        1 replica    22.2 s  -> 39.3 s/it   (matches v2's observed 38.38)
        3 replicas    7.9 s  -> 25.0 s/it

    DEBERTA_SERVICE_URLS is a comma-separated list; rank r uses entry r % len. Replicas
    must be roughly equal in speed -- static sharding is limited by the SLOWEST one, so
    a 3x fast set beat a 4x set containing one card that ran at 60% speed (7.9 s vs
    8.8 s). Falls back to the single DEBERTA_SERVICE_URL when the list is unset.
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
    """True if every sentence ThinkAlign will split out has at least one encodable word.

    ThinkAlign's word regex is `[A-Za-z]+ | digits | [^\\w\\s]`, and CJK matches none of the
    three (Chinese characters are `\\w`, so they are not punctuation either), so a Chinese
    sentence pretokenizes to zero words and FrozenHFTokenEncoder raises "Cannot encode an
    empty sentence" -- which killed a probe at step 7. SLAKE is ~49% Chinese questions and
    the policy answers in kind, so this is routine, not an edge case.

    Checking here rather than filtering sentences is deliberate: ThinkAlignSentenceAligner
    re-splits internally, so a filtered list would not survive the round trip. A sample
    that fails simply scores 0.0, the same as one with no <think> block -- the reward is
    English-only by construction and abstains instead of guessing.
    """
    sentences = _split(text)
    return bool(sentences) and all(
        # Not merely "pretokenize returned something": its `[^\w\s]` branch matches CJK
        # punctuation, so 图像中肝脏位于右上腹。 tokenizes to ['。'] -- non-empty, so it
        # does not crash, but IDF gives punctuation zero mass, the encoder falls back to
        # uniform weights over punctuation embeddings, and the pair scores a meaningless
        # 0.325. That is worse than abstaining: it injects noise straight into the
        # within-group variance that GRPO turns into gradient. Require real word content.
        any(_WORDLIKE.search(word) for word in pretokenize(sentence))
        for sentence in sentences
    )


def get_ref_pool() -> Dict[str, List[str]]:
    """Reference rationales grouped by source dataset, for the contrastive baseline.

    Loaded from THINK_REF_POOL (the training_data dir), not from the batch: a batch holds
    only 4 distinct references and they are drawn across the whole task mix, so in-batch
    distractors would be mostly cross-modality -- measured to leave 58% of the template's
    advantage standing, where same-dataset distractors flip it negative.
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

        reward = ThinkAlign(think, own_ref) - mean_k ThinkAlign(think, other_ref_k)

    The subtraction is the whole point. Plain ThinkAlign is a bag-of-content-words alignment, so
    against a diverse-but-lexically-homogeneous reference distribution its expected-reward
    maximiser is the modality's *modal* sentence -- one generic line naming optic disc /
    margins / vessels / macula scores well against any fundus reference, while a genuinely
    image-specific line only wins when the reference happens to mention the same finding.
    Measured on the first run's checkpoint: emitting the modal template beat emitting a real
    image-specific description by +0.128 (fundus) / +0.138 (chest), which is 76% of the
    +0.168 correct-vs-wrong gap ThinkAlign was adopted for, and available unconditionally. The
    policy duly collapsed -- 56% of seed_DR_referable and 52% of MESSIDOR2 <think> blocks
    were one byte-identical sentence, and the chest-X-ray template was emitted verbatim on
    CT images -- costing -15.9 pt on AMi-Br and -9.6 pt on 5-class DR grading, while binary
    tasks on the same images were untouched (-0.4 pt on DR-referable).

    A generic sentence scores the same against its own reference as against any other, so
    the difference nets to ~0; a specific one keeps its margin. Measured with same-dataset
    distractors, K=4: template +0.653 -> -0.032, oracle 0.881 -> 0.373, and the spread that
    GRPO actually consumes is unchanged (0.371 -> 0.366), i.e. the hack is removed without
    taking the signal with it.
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
