#!/bin/sh
# Download Qwen3-0.6B (~1.5 GB), import it to TGIR, and -- if torch is present
# -- verify the logits match Hugging Face. Needs curl, python3 + numpy, ~3 GB
# disk. The import writes WORKDIR/qwen3.tg (a plain, editable TGIR program)
# and WORKDIR/w/*.bin (f32 weights).
#   run.sh TGC WORKDIR [--seqlen N]
set -e
tgc=$1; w=${2:?usage: run.sh TGC WORKDIR [--seqlen N]}; shift 2
sl=16; [ "$1" = --seqlen ] && sl=$2
here=$(cd "$(dirname "$0")" && pwd)
base=https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main
mkdir -p "$w"
[ -s "$w/config.json" ]      || curl -fsSL "$base/config.json" -o "$w/config.json"
[ -s "$w/model.safetensors" ] || curl -fL  "$base/model.safetensors" -o "$w/model.safetensors"
python3 "$here/import.py" "$w/model.safetensors" "$w/config.json" "$w" --seqlen "$sl"
if python3 -c "import torch, transformers" 2>/dev/null; then
	ln -sf model.safetensors "$w/model.safetensors" 2>/dev/null || true
	python3 "$here/verify.py" "$tgc" "$w"
else
	echo "torch/transformers not installed; skipping the Hugging Face logit check."
	echo "run: $tgc run $w/qwen3.tg <16 comma-separated token ids>"
fi
