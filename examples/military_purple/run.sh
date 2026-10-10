#!/bin/sh
# Teach the sequence model military science and purple teaming, then measure
# what that knowledge is worth on a purple-team task: naming the ATT&CK tactic
# of a real procedure example (15 classes).
# Usage: run.sh TGC WORKDIR [PRETRAIN_EPOCHS]
set -e
tgc=$1; w=${2:?usage: run.sh TGC WORKDIR [EPOCHS]}; ep=${3:-4}
here=$(cd "$(dirname "$0")" && pwd)
sh "$here/build.sh" "$w"
"$tgc" pretrain "$w/corpus.jsonl" --column text --max-rows 1000000 --epochs "$ep" -o "$w/lm" -j auto
for m in bow ssm; do
	"$tgc" train "$w/procedures.jsonl" --target tactic --model $m --seed 1 -o "$w/$m" -j auto 2> "$w/$m.log"
done
"$tgc" train "$w/procedures.jsonl" --target tactic --model ssm --seed 1 --init "$w/lm" -o "$w/ssm_pre" -j auto 2> "$w/ssm_pre.log"
for m in bow ssm ssm_pre; do printf '%-8s %s\n' "$m" "$(grep 'best epoch' "$w/$m.log")"; done
