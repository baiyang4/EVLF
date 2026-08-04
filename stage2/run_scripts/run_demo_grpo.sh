#!/bin/bash
# EVLF-FM Stage-2 GRPO — DEMO run on 100 classification samples with the ThinkAlign reward.
# Meant only to verify the pipeline runs end-to-end on the bundled demo data.
#
# Prerequisites:
#   1. Start the DeBERTa NLI service (contradiction gate) on a GPU host, 127.0.0.1 only:
#        python src/open_r1/rewards/deberta_service.py --port 8001 --host 127.0.0.1
#   2. Point DEBERTA_SERVICE_URLS at it (default below assumes localhost:8001).
#   3. Set MODEL_PATH to a local InternVL3-8B (or the SFT-merged) checkpoint.
#
# THINK_REF_POOL is left unset here, so the demo uses plain (non-contrastive) ThinkAlign —
# enough to smoke-test; the full run in run_full_grpo.sh enables the distractor pool.

set -e
cd "$(dirname "$0")/.."
STAGE2_ROOT="$(pwd)"
export PYTHONPATH="${STAGE2_ROOT}/src:${STAGE2_ROOT}/../SFT:${PYTHONPATH}"
export EVLF_INTERNVL_BASE="${EVLF_INTERNVL_BASE:-${STAGE2_ROOT}/../SFT/pretrained/InternVL3-8B}"

RUN_NAME="demo_grpo"
export DEBUG_MODE="true"
export LOG_PATH="./output/${RUN_NAME}/debug_log.txt"
mkdir -p "./output/${RUN_NAME}"

export DEBERTA_SERVICE_URLS="${DEBERTA_SERVICE_URLS:-http://127.0.0.1:8001}"
export THINK_REWARD_WEIGHT="0.25"
export THINK_LAMBDA="0.05"
export THINK_GAMMA="2.0"

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3-8B}"

torchrun --nproc_per_node=2 \
    --master_port=12346 \
    src/open_r1/grpo_train.py \
    --deepspeed deepspeed_configs/zero_stage2_config.json \
    --output_dir output/${RUN_NAME} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name data_config/demo.yaml \
    --image_root demo_data \
    --reward_funcs accuracy think \
    --freeze_vision_modules true \
    --use_peft true \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.0 \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --kl_anneal_ratio 0.4 \
    --beta_min 0.001 \
    --run_name ${RUN_NAME} \
    --save_steps 50 \
    --save_only_model true
