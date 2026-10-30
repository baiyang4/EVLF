# EVLF-FM

[Paper](https://www.nature.com/articles/s41591-026-04655-5) ·
[Model weights](https://huggingface.co/yangbai123/EVLF-FM)

Official training and evaluation code for **EVLF-FM (Explainable Vision Language
Foundation Model for Medicine)**, a medical multimodal model for disease diagnosis,
visual question answering, and medical visual grounding.

**[A clinical multimodal vision language foundation model with fine-grain explainability](https://www.nature.com/articles/s41591-026-04655-5)**

Yang Bai, Haoran Cheng, Yang Zhou, et al. · *Nature Medicine*, 2026.

## Released model

The model weights are publicly available at
[yangbai123/EVLF-FM](https://huggingface.co/yangbai123/EVLF-FM).
This released version is trained on publicly available datasets. The download includes
BF16 weights in Safetensors format, the tokenizer, model code, and an `inference.py`
entry point.

After setting up the [SFT/testing environment](#environments), run the following from
the repository root to download the checkpoint and its inference dependencies:

```bash
conda activate internvl
pip install huggingface-hub==0.34.4
hf download yangbai123/EVLF-FM --local-dir SFT/work_dirs/EVLF-FM
pip install -r SFT/work_dirs/EVLF-FM/requirements.txt
```

Run inference on an image using a CUDA GPU with BF16 support:

```bash
CUDA_VISIBLE_DEVICES=0 python SFT/work_dirs/EVLF-FM/inference.py \
    --image /path/to/image.jpg \
    --question "What modality is used to take this image?" \
    --task vqa
```

Supported tasks are `vqa`, `cls`, `grounding`, and `raw`. Add `--thinking` for VQA
reasoning output, or `--no-flash-attn` to run without FlashAttention. See the
[model card](https://huggingface.co/yangbai123/EVLF-FM#usage) for Python and batch
inference examples, and the
[evaluation results](https://huggingface.co/yangbai123/EVLF-FM/blob/main/evaluation/results.txt)
for the released checkpoint's benchmark scores.

## Training and evaluation pipeline

The pipeline has three steps:

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

## Citation

If you use EVLF-FM in your research, please cite our paper:

```bibtex
@article{bai2026evlffm,
  title   = {A clinical multimodal vision language foundation model with fine-grain explainability},
  author  = {Bai, Yang and Cheng, Haoran and Zhou, Yang and Zhou, Jun and
             Thirunavukarasu, Arun and Ke, Yuhe and Yao, Jie and Fukutsu, Kanae and
             Quek, Chrystie Wan Ning and Hong, Ashley Shuen Ying and
             Gutierrez, Laura and Berni, Alessandro and Teo, Zhen Ling and
             Ting, Darren Shu Jeng and Soetikno, Brian T. and Nielsen, Christopher S. and
             Elze, Tobias and Li, Zengxiang and Sharma, Sumit and Ruthmann, Nicholas and
             Linh, Le Dinh and Chan, Hiok Hong and Koh, Victor and Tan, Marcus and
             Li, Kelvin Z. and Yip, Leonard and Cheng, Ching Yu and Tham, Yih Chung and
             Tan, Gavin Siew Wei and Schmetterer, Leopold and Kim, Rae-Young and
             Wong, Damon and Ang, Marcus and Hussain, Rahat and Mehta, Jodhbir S. and
             Aung, Tin and Cheng, Lionel Tim-Ee and Tran, Nguyen Tuan Anh and
             Cheng, Chee Leong and Wong, Tien Yin and Moshfeghi, Darius M. and Liu, Nan and
             Tan, Iain Beehuat and Lim, Soon Thye and Klang, Eyal and Lim, Tony Kiat Hon and
             Goh, Rick Siow Mong and Liu, Yong and Ting, Daniel Shu Wei},
  journal = {Nature Medicine},
  year    = {2026},
  doi     = {10.1038/s41591-026-04655-5},
  url     = {https://www.nature.com/articles/s41591-026-04655-5}
}
```
