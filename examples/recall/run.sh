#!/bin/sh
# Recall of P key-value pairs written in context, by delta-rule memories with
# a dense address (DeltaNet, dk = 16 and 256) and a sparse top-8 address over
# 256 slots. Usage: run.sh TGC WORKDIR [ROWS] [P...]. Prints the recall on the
# last 10% of batches; every batch is new sequences, so this is held out.
set -e
TGC=$1; W=$2; N=${3:-3000}; shift 3 2>/dev/null || shift $#
PS=${*:-"8 16 32 64"}
DIR=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"; W=$(cd "$W" && pwd)
TGC=$(cd "$(dirname "$TGC")" && pwd)/$(basename "$TGC")
for P in $PS; do
	perl "$DIR/data.pl" "$P" 16 "$N" 1 > "$W/d$P.csv"
	for M in dense:16 dense:256 sparse:256; do
		K=${M%%:*}; S=${M#*:}; D=$W/${K}${S}_$P; mkdir -p "$D"
		perl "$DIR/models.pl" "$K" $((P + 16)) "$D" "$S"
		t0=$(date +%s)
		(cd "$D" && "$TGC" batch train.tg "$W/d$P.csv" -o out > /dev/null)
		t1=$(date +%s)
		awk -v n="$N" -v m="$K $S" -v p="$P" -v s=$((t1 - t0)) 'NR > n * 0.9 { a += $1; c++ } END { printf "P=%d %-10s recall %.3f  (%d s)\n", p, m, a / c, s }' "$D/out"
	done
done
