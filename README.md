# EVLF-FM

Training and evaluation code for **EVLF-FM**, an InternVL3-8B–based medical multimodal
model with two auxiliary medical encoders (MedImageInsight disease encoder + MedSAM
pixel-level encoder). The pipeline has three steps:

```
  Stage 1 (SFT)                 Stage 2 (GRPO)                  Testing
  ┌──────────────┐   merged     ┌──────────────┐   merged      ┌──────────────┐
  │   SFT/       │──────────────▶│   stage2/    │──────────────▶│  SFT/eval    │
  │ supervised   │  SFT model   │ RL w/ verifi- │  final model  │  inference   │
  │ fine-tuning  │  = init      │ able rewards  │               │  + scoring   │
  └──────────────┘              └──────────────┘               └──────────────┘
```

1. **Stage 1 — SFT** (`SFT/`): supervised fine-tuning of the LLM (LoRA) and the two
   auxiliary encoders on medical image–text data.
2. **Stage 2 — GRPO** (`stage2/`): reinforcement learning from the SFT model with
   verifiable task rewards (accuracy) plus a clinical reasoning-alignment reward (`r_think`).
3. **Testing** (`SFT/eval/`): multi-GPU inference + scoring, used for both the SFT and the
   GRPO model.

Each subdirectory has its own README with full details:
[SFT/README.md](SFT/README.md) · [stage2/README.md](stage2/README.md).

---

## Layout

```
EVLF_FM/
├── SFT/        # Stage 1: supervised fine-tuning (also hosts the shared model code + eval)
├── stage2/     # Stage 2: GRPO reinforcement learning
└── README.md   # (this file)
```

`stage2/` reuses the model code from `SFT/`: the `internvl` (InternVLSAMModel) and
`segment_anything` packages, the base `InternVL3-8B` weights, the LoRA-merge tool, and the
evaluation script all live under `SFT/`. Keep the two directories side by side.

## Environments

The two stages use **different** conda environments (different pinned torch/transformers):

| | env | key pins |
|---|---|---|
| Stage 1 (SFT) + testing | `internvl` | py3.9, torch 2.1, transformers 4.37 |
| Stage 2 (GRPO) | see `stage2/environment.yml` | py3.10, torch 2.6, transformers 4.49 |

```bash
# Stage 1 / testing
conda env create -f SFT/environment.yaml && conda activate internvl
pip install flash-attn==2.4.0.post1 --no-build-isolation
# Stage 2
conda env create -f stage2/environment.yml
```

---

## Step 1 — SFT

Train and then **merge** the LoRA adapters into a standalone checkpoint (details in
[SFT/README.md](SFT/README.md)):

```bash
conda activate internvl
cd SFT
bash scripts/train_evlf_sft.sh
python tools/merge_lora_sam.py  work_dirs/<sft_out>  work_dirs/<sft_out>_merged
```

Output: a merged SFT model, e.g. `SFT/work_dirs/<sft_out>_merged`. This is the
initialisation for Stage 2.

Demo (100 samples, 1 GPU):

```bash
bash demo/run_demo.sh
```

## Step 2 — GRPO

Start from the merged SFT model and run GRPO with the `r_think` reward (details in
[stage2/README.md](stage2/README.md)). Stage 2 needs the DeBERTa NLI service running on
`127.0.0.1` first (see stage2/README).

```bash
conda activate <stage2-env>
cd stage2
MODEL_PATH=../SFT/work_dirs/<sft_out>_merged \
THINK_REF_POOL=/path/to/training_data \
bash run_scripts/run_full_grpo.sh
```

The GRPO checkpoint carries LoRA adapters — **merge it** before testing, with the same tool:

```bash
python ../SFT/tools/merge_lora_sam.py  output/internvl3_8b_grpo  output/internvl3_8b_grpo_merged
```

Demo (100 classification samples, verifies the pipeline end-to-end):

```bash
MODEL_PATH=OpenGVLab/InternVL3-8B bash run_scripts/run_demo_grpo.sh
```

## Step 3 — Testing

Evaluation uses the same script for both stages (run in the `internvl` env). Point
`--checkpoint` at any **merged** model — the SFT model or the merged GRPO model:

```bash
conda activate internvl
cd SFT
python eval/eval_batch_sam.py \
    --checkpoint /path/to/merged_model \
    --question-file <test.json> \
    --image-folder <images> \
    --answers-file work_dirs/eval/preds.json \
    --model-name sam --batch-size 8
```

Quick demo test (30-sample SLAKE VQA, lenient accuracy):

```bash
CKPT=/path/to/merged_model bash demo/run_test_demo.sh
```

---

## Notes

- Checkpoints must be **merged** (LoRA folded in) before evaluation — a raw training output
  still carries adapters. Use `SFT/tools/merge_lora_sam.py`.
- **Not in this repo** (see `.gitignore`): the training manifests under `SFT/data/` (~1.4 GB of
  `.jsonl`) and the base/auxiliary weights under `SFT/pretrained/` (`InternVL3-8B`,
  `medimageinsigt.pth`, `medmnist_tsm.bin`, `sam-med2d_b.pth`). Obtain them separately and
  place them at those paths — the demo data under `SFT/demo/` and `stage2/demo_data/` *is*
  included, so the demos run out of the box once the weights are in place.
- No API keys are shipped; `WANDB_API_KEY` comes from `~/.netrc` or `wandb login`.
- The DeBERTa NLI service for Stage 2 must be bound to `127.0.0.1` and reached over an ssh
  tunnel — never exposed on the network.
