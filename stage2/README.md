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

## Testing after GRPO

Evaluate the final Stage-2 model with **merge → infer → score**. The merge and
inference utilities live under `../SFT` and use the `internvl` environment.
Run the following commands from `stage2/`.

### 1. Merge the GRPO checkpoint

Fold the trained LoRA adapters into the model weights:

```bash
conda activate internvl
export PYTHONPATH="$(pwd)/../SFT:${PYTHONPATH}"
python ../SFT/tools/merge_lora_sam.py \
    output/internvl3_8b_grpo output/internvl3_8b_grpo_merged
```

### 2. Run inference

`../SFT/eval/eval_batch_sam.py` performs multi-GPU (DDP) batched inference.
The `--question-file` is a JSON array of objects:

```json
[ {"question_id": 0, "image": "xmlab1/source.jpg",
   "text": "<image>\nWhat modality is used to take this image?"} ]
```

It loads `InternVLSAMModel`, builds the image inputs, and writes predictions to
`--answers-file` as JSONL (`question_id`, `prompt`, `text` = the model's answer).

```bash
CUDA_VISIBLE_DEVICES=0 python ../SFT/eval/eval_batch_sam.py \
    --checkpoint output/internvl3_8b_grpo_merged \
    --question-file <your_test>.json \
    --image-folder <image_root> \
    --answers-file output/eval/preds.json \
    --model-name sam --batch-size 8 --num-gpus 1
```

### 3. Score predictions

Compare the predictions with your ground-truth answers using the task's metric
(for example, exact / VQA accuracy for open QA or F1 for multi-label classification).

### Demo test (30 samples)

```bash
# defaults to output/internvl3_8b_grpo_merged
bash ../SFT/demo/run_test_demo.sh

# use another merged GRPO checkpoint (absolute path)
CKPT=/path/to/grpo_merged bash ../SFT/demo/run_test_demo.sh
```

The demo uses `../SFT/demo/demo_test.json` (30 SLAKE VQA items with reference answers)
and prints a lenient VQA accuracy plus a few example predictions. Predictions are
written to `output/demo_test/preds.json`.
