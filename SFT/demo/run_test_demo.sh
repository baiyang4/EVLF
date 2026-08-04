#!/usr/bin/env bash
# =============================================================================
# Demo TESTING / inference for EVLF.
#
# Runs a trained+merged EVLF checkpoint on the 30-sample demo test set
# (demo/demo_test.json, SLAKE VQA) and prints a lenient VQA accuracy.
#
# Usage:
#   CKPT=/path/to/merged_model bash demo/run_test_demo.sh
#   # or just `bash demo/run_test_demo.sh` to use the default merged checkpoint.
#
# The checkpoint MUST be a *merged* model (LoRA folded in). A raw training
# output still carries LoRA adapters — merge it first:
#   python tools/merge_lora_sam.py  work_dirs/<train_out>  work_dirs/<train_out>_merged
# =============================================================================
set -x
eval "$(conda shell.bash hook)" && conda activate internvl
cd "$(dirname "$0")/.." || exit 1

# Default = the released Stage-1 SFT base (already merged). Override with CKPT=...
CKPT=${CKPT:-/mnt/data2/EVLF_rebuttal/work_dirs/internvl_chat_v2_5/internvl3_8b_sam_no_tsm_auxlora64_biglora_128_evlf_papilafix_merged}

GPUS=${GPUS:-1}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=3

QFILE=demo/demo_test.json
IMG=demo/images
OUT=work_dirs/demo_test/preds.json
mkdir -p "$(dirname "$OUT")"

python eval/eval_batch_sam.py \
  --checkpoint "$CKPT" \
  --question-file "$QFILE" \
  --answers-file "$OUT" \
  --image-folder "$IMG" \
  --model-name sam \
  --task-type vqa \
  --batch-size 8 \
  --num-gpus "$GPUS"

echo "########## score (lenient VQA accuracy) ##########"
python3 - "$QFILE" "$OUT" <<'PY'
import json, sys, re
q=json.load(open(sys.argv[1]))
gt={str(r["question_id"]): r["answer"] for r in q}
def norm(s): return re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).strip()
pred={}
for line in open(sys.argv[2]):
    line=line.strip()
    if not line: continue
    r=json.loads(line); pred[str(r["question_id"])]=r["text"]
n=cor=0; shown=0
for qid,g in gt.items():
    p=pred.get(qid,"")
    gn, pn = norm(g), norm(p)
    ok = gn and (gn==pn or gn in pn)          # exact or GT contained in prediction
    n+=1; cor+=ok
    if shown<8:
        print(f"[{'OK ' if ok else 'x  '}] gt={g!r:20s} pred={p[:60]!r}")
        shown+=1
print(f"\nDemo VQA accuracy: {cor}/{n} = {cor/n:.3f}")
PY
