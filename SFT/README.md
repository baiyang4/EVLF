# EVLF — SFT Training Code

Stage-1 supervised fine-tuning code for **EVLF**, an InternVL3-based medical
multimodal model augmented with two auxiliary medical encoders:

- a **disease encoder** — MedImageInsight (DaViT-Huge)
- a **pixel-level encoder** — MedSAM (SAM ViT-B)

Both auxiliary encoders are **LoRA-trained** (auxiliary LoRA rank 64, α=128) on
top of frozen pretrained weights and then merged; the InternViT visual backbone
is frozen and the LLM is adapted with LoRA (rank 128). This is the exact
configuration used for the released model.

## Directory layout

```
SFT/
├── internvl/                       # core training package
│   ├── conversation.py
│   ├── dist_utils.py
│   ├── model/
│   │   ├── internvl_chat/          # InternViT + InternVL backbone + EVLF (SAM) model
│   │   ├── internlm2/              # required by InternVL backbone
│   │   └── phi3/                   # required by InternVL backbone
│   ├── patch/                      # FlashAttn / sampler monkey-patches
│   └── train/
│       ├── internvl_chat_finetune_sam.py   # main training entry
│       ├── dataset.py / dataset_packed.py
│       ├── constants.py
│       ├── model_davit.py          # MedImageInsight DaViT-Huge backbone
│       └── tsm_transform.py        # disease-encoder image transforms
├── segment_anything/               # MedSAM (SAM ViT-B) encoder code
├── eval/
│   └── eval_batch_sam.py           # multi-GPU inference for the EVLF (SAM) model
├── tools/
│   └── merge_lora_sam.py           # fold LLM + aux-encoder LoRA into the base weights
├── scripts/
│   └── train_evlf_sft.sh           # Stage-1 SFT launch script
├── configs/
│   ├── evlf_sft.json               # training data meta (dataset list)
│   └── deepspeed_zero1.json        # DeepSpeed ZeRO-1 config
├── data/
│   └── train/                      # training annotations (jsonl) referenced by the meta
├── demo/
│   ├── demo_meta.json              # 100-sample demo meta (training)
│   ├── demo.jsonl                  # 100 SLAKE samples (training)
│   ├── demo_test.json              # 30-sample demo test set (evaluation)
│   ├── images/                     # 100 demo images
│   ├── run_demo.sh                 # one-GPU demo training script
│   └── run_test_demo.sh            # demo inference + scoring script
├── pretrained/                     # the pretrained checkpoints (or symlinks)
├── environment.yaml
└── README.md
```

## Environment setup

Verified against Python 3.9 + CUDA 12.1:

| Package        | Version     |
| -------------- | ----------- |
| python         | 3.9.21      |
| torch          | 2.1.0+cu121 |
| transformers   | 4.37.2      |
| deepspeed      | 0.15.4      |
| peft           | 0.10.0      |
| timm           | 0.9.12      |
| flash-attn     | 2.4.0.post1 |
| albumentations | 2.0.7       |

### Option A — from `environment.yaml` (recommended)

```bash
conda env create -f environment.yaml
conda activate internvl

# flash-attn must be built AFTER torch is installed, so it is kept out of the
# YAML. This step takes ~10 min the first time (compiles against CUDA).
pip install flash-attn==2.4.0.post1 --no-build-isolation
```

### Option B — manual install

```bash
conda create -n internvl python=3.9 -y
conda activate internvl

# PyTorch (CUDA 12.1 build used in our experiments)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Core training stack
pip install transformers==4.37.2 deepspeed==0.15.4 peft==0.10.0 timm==0.9.12

# Flash-Attention 2 (requires a matching CUDA/torch build — ~10 min to compile)
pip install flash-attn==2.4.0.post1 --no-build-isolation

# Data / image utilities
pip install albumentations==2.0.7 opencv-python imageio decord orjson \
            sentencepiece einops protobuf tokenizers accelerate \
            pillow numpy scipy pyyaml tqdm

# Required by MedImageInsight (DaViT-Huge) backbone
pip install mup
```

`segment_anything/` is shipped inside this repo and is picked up automatically
via `PYTHONPATH=$(pwd)` (set by the launch scripts) — no separate install.

## Pretrained checkpoints

Three checkpoints go under `pretrained/`:

| File                            | Purpose                                      | Source                                   |
| ------------------------------- | -------------------------------------------- | ---------------------------------------- |
| `pretrained/InternVL3-8B/`      | base InternVL3 (LLM = Qwen3-8B)              | HF `OpenGVLab/InternVL3-8B`              |
| `pretrained/sam-med2d_b.pth`    | MedSAM (SAM-Med2D ViT-B) pixel-level encoder | `OpenGVLab/SAM-Med2D` (Google Drive)     |
| `pretrained/medimageinsigt.pth` | MedImageInsight DaViT-Huge disease encoder   | HF `lion-ai/MedImageInsights` (vision)   |

### Download

```bash
mkdir -p pretrained

# 1) InternVL3-8B  (base VLM)
huggingface-cli download --resume-download OpenGVLab/InternVL3-8B \
    --local-dir pretrained/InternVL3-8B --local-dir-use-symlinks False

# 2) SAM-Med2D ViT-B pixel encoder  (sam-med2d_b.pth, ~2.6 GB)
#    Hosted on Google Drive by the official OpenGVLab/SAM-Med2D repo.
pip install gdown
gdown 1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl -O pretrained/sam-med2d_b.pth

# 3) MedImageInsight DaViT-Huge disease encoder  (medimageinsigt.pth, ~1.4 GB)
#    Community mirror lion-ai/MedImageInsights; we use only the VISION weight
#    (2024.09.27/vision_model/medimageinsigt-v1.0.0.pt), renamed to medimageinsigt.pth.
huggingface-cli download --resume-download lion-ai/MedImageInsights \
    2024.09.27/vision_model/medimageinsigt-v1.0.0.pt \
    --local-dir pretrained/_mii
mv pretrained/_mii/2024.09.27/vision_model/medimageinsigt-v1.0.0.pt \
   pretrained/medimageinsigt.pth
rm -rf pretrained/_mii
```

If the weights already live elsewhere on the machine, symlink instead of
re-downloading:

```bash
ln -sfn /path/to/InternVL3-8B       pretrained/InternVL3-8B
ln -sfn /path/to/sam-med2d_b.pth    pretrained/sam-med2d_b.pth
ln -sfn /path/to/medimageinsigt.pth pretrained/medimageinsigt.pth
```

> `medmnist_tsm.bin` (the auxiliary MedMNIST ViT head) is **not** used by this
> configuration — the released model runs with `--use_tsm2 False`, so it does not
> need to be downloaded.

Sources:
[InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B) ·
[SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) ·
[MedImageInsight (lion-ai mirror)](https://huggingface.co/lion-ai/MedImageInsights)

## Full training

```bash
bash scripts/train_evlf_sft.sh
```

Training configuration: **8 × H200**, per-device batch **32**, global batch
**256** (= 32 × 8 × grad-accum 1), **1 epoch**, LLM LoRA rank 128, auxiliary
encoder LoRA rank 64, lr **1e-4** cosine with **3 %** warm-up, weight decay
0.01, bf16, DeepSpeed **ZeRO-1**, 448×448 input, max 6 tiles. The two auxiliary
encoders are trained via `--use_aux_lora 64`; the MedMNIST head is removed via
`--use_tsm2 False`.

All training **annotations** are bundled under `data/train/` and referenced by
the meta (`configs/evlf_sft.json`) with relative paths — no external annotation
files are needed. The **images** are not bundled (tens of GB); each dataset's
`root` in the meta points to an absolute image directory that you must provide.
The image roots expected by the meta are:

```
/mnt/data/by/data/cls                         # MedMNIST / classification
/mnt/data/by/data/grounding                   # grounding
/mnt/data/by/data/vqa/{SLAKE,PMC-VQA,Path-VQA,VQA-RAD}/...   # VQA
/home/baiyang/project/Vitask/data/{nih-chest-xrays,vindr}    # CXR
/mnt/data2/EVLF_rebuttal                       # EVLF CoT data
```

Point these at your local copies (edit the `root` fields in `configs/evlf_sft.json`),
or use the demo below to verify the pipeline end-to-end without the full corpus.

## Demo (verified on 2× GPUs, 100 samples, 10 steps)

```bash
# single GPU
bash demo/run_demo.sh

# multi-GPU example (GPUs 5 and 6)
CUDA_VISIBLE_DEVICES=5,6 GPUS=2 PER_DEVICE_BATCH_SIZE=1 GRADIENT_ACC=1 \
    bash demo/run_demo.sh
```

The demo reuses the same trainer, model code and data pipeline as the full run
(including `--use_tsm2 False --use_aux_lora 64`) — only the meta path, batch
size and `max_steps` are reduced so it finishes in ~15 s on 2× H200. A
checkpoint is written to `work_dirs/demo_evlf/`.

## Testing / Evaluation

Testing is a three-step pipeline: **merge → infer → score**.

### 1. Merge the trained LoRA

Training saves a base + LoRA checkpoint (LLM LoRA r=128 and the two auxiliary
encoder LoRAs). Fold them into the weights before inference:

```bash
python tools/merge_lora_sam.py  work_dirs/<train_out>  work_dirs/<train_out>_merged
```

The merged directory is a plain HuggingFace model (`use_llm_lora=0`,
`use_aux_lora=0` in its `config.json` — the LoRA is already folded in).

### 2. Run inference

`eval/eval_batch_sam.py` does multi-GPU (DDP) batched inference. The
`--question-file` is a JSON array of objects:

```json
[ {"question_id": 0, "image": "xmlab1/source.jpg",
   "text": "<image>\nWhat modality is used to take this image?"} ]
```

It loads the model with `InternVLSAMModel` (pixel + disease-encoder + MedSAM
inputs are built automatically) and writes predictions to `--answers-file` as
JSONL (`question_id`, `prompt`, `text` = the model's answer).

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_batch_sam.py \
    --checkpoint work_dirs/<train_out>_merged \
    --question-file  <your_test>.json \
    --answers-file   work_dirs/eval/preds.json \
    --image-folder   <image_root> \
    --model-name sam --batch-size 8 --num-gpus 1
```

Then score the predictions against your ground-truth answers (task-appropriate
metric: exact / VQA accuracy for open QA, F1 for multi-label classification).

### Demo test (30 samples, one command)

```bash
# uses the released Stage-1 merged base by default
bash demo/run_test_demo.sh

# test your own merged checkpoint
CKPT=work_dirs/<train_out>_merged bash demo/run_test_demo.sh
```

`run_test_demo.sh` runs `eval/eval_batch_sam.py` on `demo/demo_test.json`
(30 SLAKE VQA items with reference answers) and prints a lenient VQA accuracy
plus a few example predictions. Predictions are written to
`work_dirs/demo_test/preds.json`.
