#!/bin/sh
# Technograd test suite.
#   1. VM results against known answers.
#   2. TGIR round trip: surface -> IR -> IR must be a fixed point and run identically.
#   3. C backend: generated unit compiled with -DTG_MAIN must match the VM exactly.
#   4. Diagnostics: malformed programs must be rejected with the expected message.
set -u
cd "$(dirname "$0")/.."
TGC=build/tgc
CC=${CC:-cc}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
pass=0
fail=0

ok()  { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); echo "FAIL: $*"; }

expect_run() { # file expected-output inputs...
	f=$1; want=$2; shift 2
	got=$($TGC run "$f" "$@" 2>&1)
	[ "$got" = "$want" ] && ok || bad "run $f $*: got [$got] want [$want]"
}

# 1. known answers
expect_run examples/xor.tg 0 0,0
expect_run examples/xor.tg 1 1,0
expect_run examples/xor.tg 1 0,1
expect_run examples/xor.tg 0 1,1
expect_run examples/newton.tg "1.41421354 3 4
steps 0 5" 2,9,16
expect_run tests/cases/ops.tg "$(cat tests/cases/ops.expected)" 1,-2,3,0.5

# 2-3. every example and positive case
CASES="examples/xor.tg:1,0 examples/newton.tg:2,9,16 examples/latent_reasoner.tg:1,0,0,1,0,1,1,0
examples/latent_reasoner.tg:0.1,0.2,-0.3,0.5,0.9,-1,0.3,0 tests/cases/ops.tg:1,-2,3,0.5
tests/cases/nested.tg:0.3,-0.7 tests/cases/multi_input.tg:1,2,3:0.5 tests/cases/file_param.tg:1,2
examples/use_import.tg:1,-1,2,-2 examples/delta_memory.tg:1,0,0,0:1,2,3,4
examples/drift_calibration.tg:10,20,30"
for c in $CASES; do
	f=${c%%:*}
	args=$(echo "${c#*:}" | tr ':' ' ')
	base=$TMP/$(basename "$f" .tg)
	vm=$($TGC run "$f" $args 2>&1) || { bad "vm $f: $vm"; continue; }

	$TGC ir "$f" > "$base.tgir" || { bad "ir $f"; continue; }
	$TGC ir "$base.tgir" > "$base.2.tgir" || { bad "ir reread $f"; continue; }
	cmp -s "$base.tgir" "$base.2.tgir" && ok || bad "ir not a fixed point: $f"
	ir=$($TGC run "$base.tgir" $args 2>&1)
	[ "$ir" = "$vm" ] && ok || bad "ir run differs $f: [$ir] vs [$vm]"

	$TGC c "$f" -o "$base.c" || { bad "cgen $f"; continue; }
	$CC -std=c99 -O2 -Wall -Wextra -Werror -pedantic -DTG_MAIN -o "$base.bin" "$base.c" -lm 2>"$base.err" \
		|| { bad "cc $f: $(cat "$base.err")"; continue; }
	ok
	cb=$("$base.bin" $args 2>&1)
	[ "$cb" = "$vm" ] && ok || bad "C differs $f: [$cb] vs [$vm]"
done

# Generated units must also build without the test driver and without libc beyond libm.
$TGC c examples/latent_reasoner.tg -o "$TMP/lr.c" && $CC -std=c99 -Wall -Werror -ffreestanding -c -o "$TMP/lr.o" "$TMP/lr.c" \
	&& ok || bad "freestanding compile"
if nm "$TMP/lr.o" 2>/dev/null | grep ' U ' | grep -qv -e expf -e tanhf -e sqrtf -e fabsf -e log1pf -e logf; then
	bad "unit has unexpected external symbols: $(nm "$TMP/lr.o" | grep ' U ')"
else
	ok
fi

# 4b. file I/O: @file inputs, -o outputs, batch (text and raw f32), atomic failure
expect_run examples/newton.tg "1.41421354 3 4
steps 0 5" @tests/io/newton_a.txt
got=$($TGC batch examples/xor.tg tests/io/xor.csv)
[ "$got" = "$(cat tests/io/xor.expected)" ] && ok || bad "batch xor: [$got]"
# each batch row must equal the corresponding single run
rows=""
for x in 0,0 1,0 0,1 1,1; do rows="$rows$($TGC run examples/xor.tg $x)
"; done
[ "$(printf "$rows")" = "$got" ] && ok || bad "batch differs from run"
# raw f32: input stream -> output stream, decoded and compared with the text path
perl -e 'print pack("f<*", 0,0, 1,0, 0,1, 1,1)' > "$TMP/xor.bin"
$TGC batch examples/xor.tg "$TMP/xor.bin" -o "$TMP/xor_out.bin"
dec=$(perl -e 'local $/; $_ = <STDIN>; print join("\n", unpack("f<*", $_)), "\n"' < "$TMP/xor_out.bin")
[ "$dec" = "$got" ] && ok || bad "binary batch: [$dec]"
$TGC run examples/newton.tg @"$TMP/xor.bin" >/dev/null 2>&1 && bad "size mismatch must fail" || ok
$TGC run examples/newton.tg 2,9,16 -o "$TMP/nw.csv" && [ "$(cat "$TMP/nw.csv")" = "1.41421354,3,4,5" ] && ok || bad "run -o csv"
$TGC batch examples/xor.tg tests/io/bad_row.csv -o "$TMP/partial.csv" >/dev/null 2>&1 && bad "bad row must fail" || ok
[ ! -e "$TMP/partial.csv" ] && ok || bad "failed batch left a partial output file"

# 4c. tracing and logs: stdout is never affected; -v reports stages; --log is valid JSONL
plain=$($TGC run examples/latent_reasoner.tg 1,0,0,1,0,1,1,0)
[ "$($TGC run examples/latent_reasoner.tg 1,0,0,1,0,1,1,0 -vv 2>/dev/null)" = "$plain" ] && ok || bad "-vv changed stdout"
err=$($TGC run examples/latent_reasoner.tg 1,0,0,1,0,1,1,0 -v 2>&1 >/dev/null)
case "$err" in *"[plan] arena 320 bytes"*"think[0]: 9/32 iteration(s)"*converged*) ok ;; *) bad "-v trace: [$err]" ;; esac
log=$TMP/t.jsonl
$TGC run examples/newton.tg 2,9,16 --log "$log" >/dev/null
$TGC check tests/errors/matmul_dims.tg --log "$log" 2>/dev/null
$TGC fix tests/fix/colon.tg --log "$log" >/dev/null 2>&1
perl -MJSON::PP -ne 'decode_json($_)' "$log" 2>/dev/null && ok || bad "log is not valid JSON Lines"
grep -q '"level":"error","stage":"error".*"file":"tests/errors/matmul_dims.tg","line":5' "$log" && ok || bad "error event missing file/line"
grep -q '"stage":"autofix".*"decision":"applied"' "$log" && ok || bad "autofix decision not logged"
grep -q '"level":"debug"' "$log" && bad "debug events leaked into log without -vv" || ok
$TGC run examples/newton.tg 2,9,16 -vv --log "$TMP/d.jsonl" >/dev/null 2>&1
grep -q '"stage":"vm".*"op":"div"' "$TMP/d.jsonl" && ok || bad "-vv log lacks vm events"

# 4d. threads: results are bit-identical at any thread count (batch rows and
#     intra-op matmul, including the nested case), and -j is validated
perl -e 'srand(7); for (1..3000) { print join(",", map { sprintf "%.4f", rand()*4-2 } 1..8), "\n" }' > "$TMP/p.csv"
$TGC batch examples/latent_reasoner.tg "$TMP/p.csv" -j 1 -o "$TMP/p1.csv"
$TGC batch examples/latent_reasoner.tg "$TMP/p.csv" -j 4 -o "$TMP/p4.csv"
cmp -s "$TMP/p1.csv" "$TMP/p4.csv" && [ "$(wc -l < "$TMP/p4.csv")" -eq 3000 ] && ok || bad "batch -j4 differs from -j1"
perl -e 'srand(7); print pack("f<*", map { rand()*4-2 } 1..24000)' > "$TMP/p.bin"
$TGC batch examples/latent_reasoner.tg "$TMP/p.bin" -j 1 -o "$TMP/q1.bin"
$TGC batch examples/latent_reasoner.tg "$TMP/p.bin" -j auto -o "$TMP/q4.bin"
cmp -s "$TMP/q1.bin" "$TMP/q4.bin" && ok || bad "binary batch -j auto differs from -j1"
W=1,0,-1,0.5,0,0,1,0,0,2,0,0,-1,0,0,1
[ "$($TGC run tests/par/wide.tg $W -j 4)" = "$($TGC run tests/par/wide.tg $W -j 1)" ] && ok || bad "parallel matmul changed the result"
printf "%s\n%s\n%s\n" $W $W $W > "$TMP/w.csv"
[ "$($TGC batch tests/par/wide.tg "$TMP/w.csv" -j 4)" = "$($TGC batch tests/par/wide.tg "$TMP/w.csv" -j 1)" ] && ok || bad "nested parallelism changed the result"
$TGC run examples/xor.tg 1,0 -j abc >/dev/null 2>&1 && bad "-j abc must be rejected" || ok

# 4e. self-updating state: exact delta-rule recall, drift convergence, C unit
#     == VM across calls, order kept under -j, and the firmware reset/save API
[ "$($TGC batch examples/delta_memory.tg tests/state/mem.csv)" = "$(cat tests/state/mem.expected)" ] && ok || bad "delta memory recall"
for i in $(seq 40); do echo 10,20,30; done > "$TMP/drift.csv"
[ "$($TGC batch examples/drift_calibration.tg "$TMP/drift.csv" | tail -1)" = "0.164232254,0.328464508,0.492694855" ] && ok || bad "drift calibration"
$TGC c examples/delta_memory.tg -o "$TMP/unit.c" && $CC -std=c99 -O2 -Wall -Wextra -Werror -pedantic -DTG_MAIN -o "$TMP/dm" "$TMP/unit.c" -lm && ok || bad "stateful unit compile"
c=$("$TMP/dm" 1,0,0,0 1,2,3,4 0,1,0,0 5,6,7,8 1,0,0,0 0,0,0,0 0,1,0,0 0,0,0,0 1,0,0,0 9,9,9,9)
[ "$c" = "$(tr , " " < tests/state/mem.expected)" ] && ok || bad "C unit adaptation differs from VM: [$c]"
[ "$($TGC batch examples/delta_memory.tg tests/state/mem.csv -j 4)" = "$(cat tests/state/mem.expected)" ] && ok || bad "stateful batch reordered under -j"
$CC -std=c99 -Wall -Wextra -Werror -I"$TMP" -o "$TMP/harness" tests/state/harness.c -lm && "$TMP/harness" >/dev/null && ok || bad "reset/save/restore harness"
$TGC ir examples/delta_memory.tg | grep -q "(update mem %" && ok || bad "TGIR lost the update"

# 4f. autodiff: finite-difference check of every VJP rule (incl. implicit
#     differentiation through think), online training, deploy via file(), and
#     on-device training in generated C matching the VM bit for bit
mkdir -p "$TMP/gc"
if perl tests/gradcheck.pl "$TGC" "$TMP/gc" > "$TMP/gc.out"; then ok; else bad "gradcheck: $(grep FAIL "$TMP/gc.out")"; fi
[ "$(grep -c "^ok" "$TMP/gc.out")" -ge 22 ] && ok || bad "gradcheck ran too few cases"
perl -e 'srand(3); my @d=([0,0,0],[0,1,1],[1,0,1],[1,1,0]); for (1..1500) { for my $r (sort { rand() <=> 0.5 } @d) { print join(",", @$r), "\n" } }' > "$TMP/xt.csv"
$TGC batch examples/train_xor.tg "$TMP/xt.csv" --save-state "$TMP/xor" -o "$TMP/xp.csv"
paste -d, "$TMP/xt.csv" "$TMP/xp.csv" | tail -8 | awk -F, '{ if ($3 == 1 && $4 < 0.9 || $3 == 0 && $4 > 0.1) bad = 1 } END { exit bad }' && ok || bad "XOR not learned"
printf 'model xd\nparam w1 : f32[4, 2] = file("xor.w1.bin")\nparam b1 : f32[4] = file("xor.b1.bin")\nparam w2 : f32[4] = file("xor.w2.bin")\nparam b2 : f32 = file("xor.b2.bin")\ndef forward(x: f32[2]) -> f32:\n    return sigmoid(dot(w2, tanh(w1 @ x + b1)) + b2)\n' > "$TMP/xd.tg"
a=$($TGC run "$TMP/xd.tg" 0,1); b=$($TGC run "$TMP/xd.tg" 1,1)
awk -v a="$a" -v b="$b" 'BEGIN { exit !(a > 0.9 && b < 0.1) }' && ok || bad "deployed weights do not reproduce training: $a $b"
$TGC c examples/train_xor.tg -o "$TMP/tx.c" && $CC -std=c99 -O2 -Wall -Wextra -Werror -pedantic -DTG_MAIN -o "$TMP/tx" "$TMP/tx.c" -lm && ok || bad "training unit compile"
head -200 "$TMP/xt.csv" > "$TMP/t200.csv"
[ "$("$TMP/tx" $(awk -F, '{ printf "%s,%s %s ", $1, $2, $3 }' "$TMP/t200.csv"))" = "$($TGC batch examples/train_xor.tg "$TMP/t200.csv")" ] && ok || bad "on-device training differs from VM"

# 4g. datasets, shared gradients, optimizers, one-command training (offline;
#     set TG_TEST_NETWORK=1 to also train on a Hugging Face dataset)
ins=$($TGC data inspect tests/data/messy.csv)
case "$ins" in *"Id"*"dropped: identifier"*) ok ;; *) bad "identifier column not dropped" ;; esac
case "$ins" in *"label_name"*"would leak the answer"*) ok ;; *) bad "leaking column not dropped" ;; esac
case "$ins" in *"weight, kg"*"(has missing)"*) ok ;; *) bad "quoted header with comma / missing values" ;; esac
case "$ins" in *"city"*"categorical"*"New York, NY"*) ok ;; *) bad "quoted field with comma" ;; esac
case "$ins" in *"classification on 'label' (2 classes)"*) ok ;; *) bad "target not picked: [$ins]" ;; esac
ins=$($TGC data inspect tests/data/rows.jsonl)
case "$ins" in *"emb"*"vector"*"3 numbers per row"*"meta"*"nested objects"*"regression on 'y'"*) ok ;; *) bad "jsonl inspect: [$ins]" ;; esac
case "$($TGC data inspect tests/data/hfpage.json)" in *"rows     2"*'x "quoted"'*) ok ;; *) bad "HF page / JSON escapes" ;; esac
case "$($TGC data inspect tests/data/arr.npy)" in *"rows     3"*"col1"*"numeric"*) ok ;; *) bad "npy" ;; esac
# separable 3-class blobs: train, then predict on unseen rows (deterministic)
perl -e 'srand(9); print "a,b,kind\n"; for (1..900) { my $k = $_ % 3; my @c = ([0,0],[3,0],[0,3]); printf "%.3f,%.3f,k%d\n", $c[$k][0] + rand() - 0.5, $c[$k][1] + rand() - 0.5, $k }' > "$TMP/blobs.csv"
$TGC train "$TMP/blobs.csv" -o "$TMP/blobs_model" > "$TMP/train.log" 2>&1 || bad "tgc train failed: $(tail -3 "$TMP/train.log")"
grep -q "accuracy 100.00%" "$TMP/train.log" && ok || bad "blobs not separated: $(grep 'best epoch' "$TMP/train.log")"
for f in model.tg infer.tg infer.c features.tgf report.txt weights.w1.bin; do [ -s "$TMP/blobs_model/$f" ] || bad "missing artifact $f"; done; ok
printf 'b,a\n0.1,0.2\n3.1,0.1\n0.2,2.9\n' > "$TMP/new.csv"   # columns reordered, no target
[ "$($TGC predict "$TMP/blobs_model" "$TMP/new.csv" 2>/dev/null | cut -d, -f1 | tr '\n' ' ')" = 'prediction "k0" "k2" "k1" ' ] && ok || bad "predict on reordered columns"
$CC -std=c99 -Wall -Werror -c -o "$TMP/infer.o" "$TMP/blobs_model/infer.c" && ok || bad "generated infer.c does not compile"
# shared backward pass: four grad() of one loss -> one pass; DCE keeps IR small
[ "$($TGC check examples/train_xor.tg -vv 2>&1 | grep -c '\[grad\] backward pass')" -eq 1 ] && ok || bad "grad passes not shared"
printf 'model d\ndef forward(x: f32[2]) -> f32[2]:\n    unused = exp(x) * 3\n    return x + 1\n' > "$TMP/dce.tg"
$TGC ir "$TMP/dce.tg" | grep -q exp && bad "dead code survived" || ok
# every optimizer learns XOR online
for o in "sgd(lr=0.5)" "momentum(lr=0.1)" "adam(lr=0.02)" "adamw(lr=0.02)" "muon(lr=0.02)"; do
	sed "s/OPT/$o/" > "$TMP/opt.tg" <<'TG'
model xo
state w1 : f32[4, 2] = rand(1, 1)
state b1 : f32[4] = zeros
state w2 : f32[4] = rand(2, 1)
state b2 : f32 = 0
def forward(x: f32[2], target: f32) -> f32:
    p = sigmoid(dot(w2, tanh(w1 @ x + b1)) + b2)
    train bce(p, target) with OPT
    return p
TG
	$TGC batch "$TMP/opt.tg" "$TMP/xt.csv" -o "$TMP/op.csv"
	paste -d, "$TMP/xt.csv" "$TMP/op.csv" | tail -400 | awk -F, '{ e = $3 - $4; if (e < 0) e = -e; if (e > 0.05) bad = 1 } END { exit bad }' && ok || bad "optimizer $o did not learn XOR"
done
if [ "${TG_TEST_NETWORK:-0}" = 1 ]; then
	$TGC train hf:scikit-learn/iris -o "$TMP/iris" > "$TMP/iris.log" 2>&1 && grep -q "classification of 'Species'" "$TMP/iris.log" && ok || bad "hf iris"
fi

# 4a. cross-file imports: diamond dedup and cycle termination
expect_run tests/imports/diamond.tg "23 43" 10,20
expect_run tests/imports/cycle.tg "11 21" 10,20

# 4. diagnostics
for f in tests/errors/*.tg tests/errors/*.tgir; do
	want=$(sed -n '1s/^[#;] expect: //p' "$f")
	got=$($TGC check "$f" 2>&1)
	case "$got" in
	*"$want"*) ok ;;
	*) bad "diagnostic $f: got [$got] want [*$want*]" ;;
	esac
done

# 5. auto-fixer: each broken program must be repaired to its .expected text;
#    a program without .expected must be left unrepaired (the fixer abstains).
for f in tests/fix/*.tg; do
	exp=${f%.tg}.expected
	out=$($TGC fix "$f" 2>/dev/null); st=$?
	if [ -f "$exp" ]; then
		[ $st -eq 0 ] && [ "$out" = "$(cat "$exp")" ] && ok || bad "fix $f (status $st)"
	else
		[ $st -ne 0 ] && ok || bad "fix $f should abstain"
	fi
done
# --autofix repairs in memory and continues with the requested command.
got=$($TGC run tests/fix/colon.tg 2,9,16 --autofix 2>/dev/null)
[ "$got" = "$($TGC run examples/newton.tg 2,9,16)" ] && ok || bad "--autofix run: [$got]"
$TGC run tests/fix/colon.tg 2,9,16 >/dev/null 2>&1 && bad "without --autofix the error must stand" || ok
# Held-out repair quality (deterministic): programs never seen in training.
ev=$($TGC fixer-eval tests/cases/*.tg | awk '/^total/ { gsub("%", ""); print $3, $4 }')
set -- $ev
awk -v e="$1" -v w="$2" 'BEGIN { exit !(e >= 85 && w <= 1) }' && ok || bad "fixer-eval exact $1% wrong $2%"

echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
