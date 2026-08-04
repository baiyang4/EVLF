#!/bin/bash
# EVLF-FM Stage-2 GRPO — FULL training with the ThinkAlign reasoning reward (GRPO config).
# This reproduces the final model internvl3_8b_grpo: accuracy + ThinkAlign on the
# kw-tagged, cls-rebalanced mix, 1 epoch, 8 GPUs.
#
# Set MODEL_PATH to the Stage-1 (SFT) merged InternVL3-8B checkpoint, and THINK_REF_POOL
# to the directory of training .json files (used for the contrastive distractor pool;
# it MUST match the training data's CoT format).
#
# The DeBERTa NLI service (contradiction gate) must be running first. Bind it to
# 127.0.0.1 only and reach it over an ssh tunnel — never expose it on the network:
#   python src/open_r1/rewards/deberta_service.py --port 8001 --host 127.0.0.1
# For throughput you may run several replicas and list them in DEBERTA_SERVICE_URLS.
#
# WANDB_API_KEY is intentionally not set here — put it in ~/.netrc or `wandb login`.

set -e
cd "$(dirname "$0")/.."
STAGE2_ROOT="$(pwd)"

# open_r1 + the vendored `think_align` package live under ./src; the `internvl`
# (InternVLSAMModel) and `segment_anything` packages are shared with the SFT tree.
export PYTHONPATH="${STAGE2_ROOT}/src:${STAGE2_ROOT}/../SFT:${PYTHONPATH}"
# Base InternVL3-8B (for the processor/tokenizer); defaults to the SFT copy.
export EVLF_INTERNVL_BASE="${EVLF_INTERNVL_BASE:-${STAGE2_ROOT}/../SFT/pretrained/InternVL3-8B}"

RUN_NAME="internvl3_8b_grpo"
export DEBUG_MODE="true"
export LOG_PATH="./output/${RUN_NAME}/debug_log.txt"
export WANDB_DIR="./output/${RUN_NAME}"
mkdir -p "./output/${RUN_NAME}"

# --- ThinkAlign reasoning reward ---
export DEBERTA_SERVICE_URLS="${DEBERTA_SERVICE_URLS:-http://127.0.0.1:8001}"   # comma-list for replicas
export THINK_REWARD_WEIGHT="0.25"     # alpha, reasoning-term weight
export THINK_LAMBDA="0.05"            # OT marginal relaxation (semi-relaxed)
export THINK_GAMMA="2.0"              # contradiction-gate exponent
export THINK_NUM_DISTRACTORS="4"      # K, same-dataset contrastive distractors
export THINK_REF_POOL="${THINK_REF_POOL:-/path/to/training_data}"

MODEL_PATH="${MODEL_PATH:-/path/to/sft/internvl3_8b_evlf_merged}"
DATA_CONFIG="${DATA_CONFIG:-data_config/grpo_train.yaml}"

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12346 \
    src/open_r1/grpo_train.py \
    --deepspeed deepspeed_configs/zero_stage2_config.json \
    --output_dir output/${RUN_NAME} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_CONFIG} \
    --reward_funcs accuracy think \
    --freeze_vision_modules true \
    --freeze_connector true \
    --use_peft true \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.0 \
    --learning_rate 1e-5 \
    --num_generations 8 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --kl_anneal_ratio 0.4 \
    --beta_min 0.001 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_total_limit 8 \
    --save_only_model true
