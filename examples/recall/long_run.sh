#!/bin/sh
# Train the long-context recall models at T = 256 (64 writes, 16 queries,
# filler elsewhere), then evaluate the same weights on longer sequences:
# one pass per sequence, and for the memory also streamed in chunks of 4096
# steps with the memory carried in a state between chunks.
# Usage: long_run.sh TGC WORKDIR [ROWS] [T...]  (default 2000 rows; T 4096 16384 65536)
set -e
TGC=$1; W=$2; N=${3:-2000}; shift 3 2>/dev/null || shift $#
TS=${*:-"4096 16384 65536"}
DIR=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"; W=$(cd "$W" && pwd)
TGC=$(cd "$(dirname "$TGC")" && pwd)/$(basename "$TGC")
L="perl $DIR/long.pl"
score() { awk -F, '{ h += $1; q += $2 } END { printf "%.4f (%d queries)", h / q, q }' "$1"; }
[ -f "$W/train.csv" ] || $L data 64 16 256 16 "$N" 1 > "$W/train.csv"
for K in pkm dense; do
	mkdir -p "$W/$K"
	$L model $K 256 16 "$W/$K"
	if [ ! -f "$W/$K/w.Ek.bin" ]; then
		t0=$(date +%s)
		(cd "$W/$K" && "$TGC" batch train.tg "$W/train.csv" -o train.out --save-state w > /dev/null)
		echo "$K: trained at T=256 in $(( $(date +%s) - t0 )) s, recall on the last 10%: $(awk -v n="$N" 'NR > n * 0.9 { a += $1; c++ } END { printf "%.4f", a / c }' "$W/$K/train.out")"
	fi
done
for T in $TS; do
	[ -f "$W/eval$T.csv" ] || $L data 64 64 "$T" 1 16 7 > "$W/eval$T.csv"
	for K in pkm dense; do
		E=$W/$K/e$T; mkdir -p "$E"; cp "$W/$K"/w.*.bin "$E/"
		$L model $K "$T" 1 "$E" "" eval
		t0=$(date +%s)
		(cd "$E" && "$TGC" batch eval.tg "$W/eval$T.csv" -o out > /dev/null)
		echo "T=$T $K: recall $(score "$E/out") in $(( $(date +%s) - t0 )) s"
	done
	if [ "$T" -gt 4096 ]; then # the memory streamed in chunks of 4096 steps
		E=$W/pkm/s$T; mkdir -p "$E"; cp "$W/pkm"/w.*.bin "$E/"
		$L model pkm 4096 1 "$E" "" stream
		perl -e 'my $T = shift; while (<>) { chomp; my @a = split /,/; for (my $c = 0; $c < $T; $c += 4096) {
			print join(",", map { my $o = $_ * $T + $c; @a[$o .. $o + 4095] } 0 .. 3), ",", $c ? 0 : 1, "\n" } }' "$T" "$W/eval$T.csv" > "$E/chunks.csv"
		t0=$(date +%s)
		(cd "$E" && "$TGC" batch stream.tg chunks.csv -o out > /dev/null)
		echo "T=$T pkm streamed in chunks of 4096: recall $(score "$E/out") in $(( $(date +%s) - t0 )) s"
	fi
done
