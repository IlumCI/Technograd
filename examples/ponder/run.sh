#!/bin/sh
# Train with the expected loss over 8 steps, then run with learned halting.
# Usage: run.sh TGC WORKDIR [TRAIN_ROWS]. Prints accuracy and the mean
# number of steps taken for each hop count.
set -e
TGC=$1; W=$2; N=${3:-20000}
DIR=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"; W=$(cd "$W" && pwd)
gen() { perl -e 'my ($n, $seed, $lab) = @ARGV; srand($seed);
	for (1 .. $n) { my @succ = map { int rand 8 } 0 .. 7; my $s = int rand 8; my $k = 1 + int rand 6; my $t = $s; $t = $succ[$t] for 1 .. $k;
		my @a = (0) x 64; $a[$_ * 8 + $succ[$_]] = 1 for 0 .. 7;
		my @x = (@a, (map { $_ == $s ? 1 : 0 } 0 .. 7), map { $_ == $k ? 1 : 0 } 1 .. 6);
		print join(",", @x, $lab ? (map { $_ == $t ? 1 : 0 } 0 .. 7) : ()), $lab ? "" : "|$k|$t", "\n" }' "$@"; }
gen "$N" 1 1 > "$W/train.csv"
"$TGC" batch "$DIR/train.tg" "$W/train.csv" -o "$W/train.out" --save-state "$W/w"
cp "$DIR/infer.tg" "$W/infer.tg"
gen 1200 2 0 > "$W/test.meta"
cut -d'|' -f1 "$W/test.meta" > "$W/test.csv"
"$TGC" batch "$W/infer.tg" "$W/test.csv" -o "$W/test.out"
paste -d'|' "$W/test.meta" "$W/test.out" | awk -F'|' '{
	k = $2; t = $3; n = split($4, o, ","); best = 1; for (i = 2; i <= 8; i++) if (o[i] > o[best]) best = i;
	steps = o[9]; ok[k] += (best - 1 == t); cnt[k]++; st[k] += steps; all += (best - 1 == t) }
	END { for (k = 1; k <= 6; k++) printf "hops %d: accuracy %5.1f%%, mean steps %.2f\n", k, 100 * ok[k] / cnt[k], st[k] / cnt[k]; printf "all: accuracy %.1f%%\n", 100 * all / NR }'
