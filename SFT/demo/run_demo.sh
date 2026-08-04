#!/usr/bin/env bash
# Demo training: fine-tunes EVLF on 100 demo samples for a few steps, using the
# SAME model code / trainer / data pipeline as the full run — only the meta path,
# batch size and max_steps are reduced so it finishes in ~15 s on 2x H200.
#
# Usage:  bash demo/run_demo.sh
#   multi-GPU:  CUDA_VISIBLE_DEVICES=5,6 GPUS=2 bash demo/run_demo.sh
#
# Requirements:
#   - conda env `internvl` activated (or activated below automatically)
#   - pretrained/InternVL3-8B          (base InternVL3)
#   - pretrained/sam-med2d_b.pth       (MedSAM ViT-B pixel encoder)
#   - pretrained/medimageinsigt.pth    (MedImageInsight DaViT-Huge disease encoder)
#   - >= 1 GPU (>= 40 GB recommended; lower PER_DEVICE_BATCH_SIZE for smaller GPUs)
#
# The demo mirrors the released Stage-1 config: --use_tsm2 False (no MedMNIST head)
# and --use_aux_lora 64 (both aux encoders LoRA-trained). A checkpoint is written
# to work_dirs/demo_evlf/.

set -x
eval "$(conda shell.bash hook)" && conda activate internvl
cd "$(dirname "$0")/.." || exit 1

GPUS=${GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-1}
AUX_LORA=${AUX_LORA:-64}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=${MASTER_PORT:-34230}
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/demo_evlf'
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
  --meta_path "demo/demo_meta.json" \
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
  --dataloader_num_workers 2 \
  --bf16 True \
  --max_steps 10 \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size False \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "configs/deepspeed_zero1.json" \
  --report_to "none" \
  2>&1 | tee "${OUTPUT_DIR}/demo_log.txt"
