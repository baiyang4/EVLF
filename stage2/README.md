# EVLF-FM — Stage 2: Reinforcement Learning with GRPO

This is the Stage-2 (GRPO) training code for EVLF-FM. Starting from the Stage-1 SFT model,
it optimises the policy with Group Relative Policy Optimization (GRPO) using verifiable
task rewards plus **ThinkAlign**, a clinical reasoning-alignment reward. This package reproduces
the final model `internvl3_8b_grpo` (accuracy + ThinkAlign).

For Stage-1 (SFT) see the sibling `../SFT` directory.

## Layout

```
stage2/
├── src/
│   ├── open_r1/
│   │   ├── grpo_train.py              # training entry point
│   │   ├── trainer/                   # GRPO trainer (BYVLMGRPOTrainer) + config
│   │   ├── vlm_modules/               # InternVL module + base
│   │   └── rewards/                   # think_reward, deberta client/service
│   └── think_align/                         # vendored ThinkAlign package (OT + NLI gate)
├── run_scripts/
│   ├── run_full_grpo.sh               # full 8-GPU training (ThinkAlign, GRPO config)
│   └── run_demo_grpo.sh              # end-to-end smoke test on 100 demo samples
├── data_config/
│   ├── grpo_train.yaml               # the full training mix (reference config)
│   └── demo.yaml                # demo config
├── demo_data/                         # 100 classification demo samples + images
├── deepspeed_configs/zero_stage2_config.json
├── environment.yml / requirements.txt
└── output/
```

## Dependencies on the SFT tree

The InternVL+SAM model code (`internvl.model.internvl_chat.InternVLSAMModel`) and
`segment_anything` are shared with Stage-1 and live under `../SFT`. The run scripts add
`../SFT` to `PYTHONPATH` and point `EVLF_INTERNVL_BASE` at `../SFT/pretrained/InternVL3-8B`
for the tokenizer/processor. Keep `stage2/` next to `SFT/` (both under `EVLF_FM/`).

## Setup

```bash
conda env create -f environment.yml    # or: pip install -r requirements.txt
```

## The DeBERTa NLI service (contradiction gate)

ThinkAlign's contradiction gate needs a DeBERTa-v2 MNLI service. Start it on a GPU host,
**bound to 127.0.0.1 only**, and reach it over an ssh tunnel — never expose it:

```bash
python src/open_r1/rewards/deberta_service.py --port 8001 --host 127.0.0.1
# from the training host:
ssh -N -L 8001:127.0.0.1:8001 <gpu-host> &
curl -s 127.0.0.1:8001/health
```

For throughput, run several replicas and list them in `DEBERTA_SERVICE_URLS`
(comma-separated); rank `r` uses entry `r % len`.

## Run

Demo (100 classification samples across modalities; verifies the pipeline end-to-end):

```bash
MODEL_PATH=OpenGVLab/InternVL3-8B bash run_scripts/run_demo_grpo.sh
```

Full training (reproduces `internvl3_8b_grpo`):

```bash
MODEL_PATH=/path/to/sft/merged \
THINK_REF_POOL=/path/to/training_data_grpo_cot \
DATA_CONFIG=data_config/grpo_train.yaml \
bash run_scripts/run_full_grpo.sh
```

## Reward

`--reward_funcs accuracy think`:

- **accuracy** — IoU for grounding; answer-overlap F1 for classification / VQA.
- **think_align** — clinical reasoning-alignment reward (token-level OT + contradiction gate +
  contrastive baseline), weight `alpha = 0.25`.
