#!/usr/bin/env bash
# =============================================================================
# EVLF — Stage-1 supervised fine-tuning.
#
# InternVL3-8B backbone (frozen) + LLM LoRA (r=128), augmented with two
# auxiliary medical encoders, both LoRA-trained (aux-LoRA r=64, a=128) and
# then merged:
#   - MedImageInsight (DaViT-Huge)  disease encoder    -> pretrained/medimageinsigt.pth
#   - MedSAM          (SAM ViT-B)   pixel-level encoder -> pretrained/sam-med2d_b.pth
# The second MedMNIST ViT head is dropped (--use_tsm2 False).
#
# Usage (from anywhere):
#   bash scripts/train_evlf_sft.sh
#   GPUS=8 PER_DEVICE_BATCH_SIZE=32 BATCH_SIZE=256 bash scripts/train_evlf_sft.sh
#
# Config: per-device 32 x 8 GPUs x grad-accum 1 = global batch 256, 1 epoch,
# lr 1e-4 cosine, warmup 0.03, wd 0.01, bf16, DeepSpeed ZeRO-1, 448x448, 6 tiles.
# =============================================================================
set -x
eval "$(conda shell.bash hook)" && conda activate internvl

# Run from the repo root regardless of where the script is invoked.
cd "$(dirname "$0")/.." || exit 1

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-256}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
AUX_LORA=${AUX_LORA:-64}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=${MASTER_PORT:-34231}
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR=${OUTPUT_DIR:-'work_dirs/evlf_sft'}
mkdir -p "$OUTPUT_DIR"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune_sam.py \
  --model_name_or_path "pretrained/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --use_tsm2 False \
  --use_aux_lora ${AUX_LORA} \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "configs/evlf_sft.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --use_llm_lora 128 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size False \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "configs/deepspeed_zero1.json" \
  --report_to "none" \
  --max_steps ${MAX_STEPS:--1} \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
