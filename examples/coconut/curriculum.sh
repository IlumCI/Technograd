#!/bin/sh
# Coconut curriculum: stage k = k latent steps on k-hop answers, warm-started
# from stage k-1. Usage: curriculum.sh TGC WORKDIR [ROWS_PER_STAGE] [STAGES]
# Prints the accuracy over the last 500 rows of each stage, then a direct
# run (STAGES latent steps from scratch, same total rows) for comparison.
set -e
TGC=$1; W=$2; N=${3:-4000}; K=${4:-4}
DIR=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W"
W=$(cd "$W" && pwd)
gen() { # rows of: A (64), start one-hot (8), target one-hot after $1 hops (8)
	perl -e 'my ($hops, $n, $seed) = @ARGV; srand($seed);
	for (1 .. $n) { my @succ = map { int rand 8 } 0 .. 7; my $s = int rand 8; my $t = $s; $t = $succ[$t] for 1 .. $hops;
		my @a = (0) x 64; $a[$_ * 8 + $succ[$_]] = 1 for 0 .. 7;
		print join(",", @a, map({ $_ == $s ? 1 : 0 } 0 .. 7), map({ $_ == $t ? 1 : 0 } 0 .. 7)), "\n" }' "$1" "$2" "$3"; }
acc() { paste -d, "$1" "$2" | tail -500 | awk -F, '{ bt = 0; bo = 0; for (i = 0; i < 8; i++) { if ($(73 + i) > $(73 + bt)) bt = i; if ($(81 + i) > $(81 + bo)) bo = i } c += bt == bo } END { printf "%.1f%%", c / 5 }'; }
model() { # model K INIT_FROM_PREFIX (or "rand")
	if [ "$2" = rand ]; then
		sed -e "s/HOPS/$1/" -e 's/INIT_W1/rand(81, 0.3)/' -e 's/INIT_W2/rand(82, 0.3)/' -e 's/INIT_E/rand(83, 0.5)/' \
		    -e 's/INIT_R/rand(84, 0.5)/' -e 's/INIT_B/zeros/' "$DIR/coconut.tg"
	else
		sed -e "s/HOPS/$1/" -e "s#INIT_W1#file(\"$2.W1.bin\")#" -e "s#INIT_W2#file(\"$2.W2.bin\")#" -e "s#INIT_E#file(\"$2.E.bin\")#" \
		    -e "s#INIT_R#file(\"$2.R.bin\")#" -e "s#INIT_B#file(\"$2.b.bin\")#" "$DIR/coconut.tg"
	fi
}
prev=rand
for k in $(seq 1 "$K"); do
	model "$k" "$prev" > "$W/stage$k.tg"
	gen "$k" "$N" "$k" > "$W/stage$k.csv"
	"$TGC" batch "$W/stage$k.tg" "$W/stage$k.csv" -o "$W/stage$k.out" --save-state "$W/s$k"
	echo "stage $k ($k latent steps, $k-hop answers): $(acc "$W/stage$k.csv" "$W/stage$k.out")"
	prev="$W/s$k"
done
model "$K" rand > "$W/direct.tg"
gen "$K" $((N * K)) 99 > "$W/direct.csv"
"$TGC" batch "$W/direct.tg" "$W/direct.csv" -o "$W/direct.out"
echo "direct ($K latent steps from scratch, $((N * K)) rows): $(acc "$W/direct.csv" "$W/direct.out")"
