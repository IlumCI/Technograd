#!/bin/sh
# Knowledge beyond active compute: recall of F arbitrary facts (fixed random
# 64-d key -> one of 256 classes) by a dense MLP and by a product-key memory
# (Lample et al., arXiv:1907.05242) with about the same multiply-adds per
# token. Usage: run.sh TGC WORKDIR [EPOCHS] [F...]. Prints the fraction of
# the F facts each model recalls after training on them.
set -e
TGC=$1; W=$2; E=${3:-40}; shift 3 2>/dev/null || shift $#
FS=${*:-"1024 4096 16384 65536"}
DIR=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"; W=$(cd "$W" && pwd)
TGC=$(cd "$(dirname "$TGC")" && pwd)/$(basename "$TGC")
for F in $FS; do
	perl -e 'my ($F, $E, $W) = @ARGV; srand(100); my @lab = map { int rand 256 } 1 .. $F; srand(7);
		for (1 .. $E) { my @p = 0 .. $F - 1; for (my $i = $#p; $i > 0; $i--) { my $j = int rand($i + 1); @p[$i, $j] = @p[$j, $i] }
			for (my $b = 0; $b + 32 <= $F; $b += 32) { my @i = @p[$b .. $b + 31]; print join(",", @i, map { $lab[$_] } @i), "\n" } }
		open my $t, ">", "$W/test_$F.csv"; open my $l, ">", "$W/test_$F.lab";
		for (my $b = 0; $b + 32 <= $F; $b += 32) { print $t join(",", $b .. $b + 31), "\n"; print $l join(",", @lab[$b .. $b + 31]), "\n" }' "$F" "$E" "$W" > "$W/train_$F.csv"
	for K in dense pkm; do
		D=$W/${K}_$F; mkdir -p "$D"
		perl "$DIR/models.pl" "$K" "$F" "$D"
		t0=$(date +%s)
		(cd "$D" && "$TGC" batch train.tg "$W/train_$F.csv" -o train.out --save-state w)
		t1=$(date +%s)
		(cd "$D" && "$TGC" batch infer.tg "$W/test_$F.csv" -o test.out)
		acc=$(paste -d'|' "$W/test_$F.lab" "$D/test.out" | awk -F'|' '{ n = split($1, a, ","); split($2, b, ","); for (i = 1; i <= n; i++) { c += (a[i] == b[i]); t++ } } END { printf "%.2f", 100 * c / t }')
		echo "F=$F $K recall ${acc}% train $((t1 - t0)) s"
	done
done
