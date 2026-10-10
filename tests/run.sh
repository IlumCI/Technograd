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
examples/use_import.tg:1,-1,2,-2 examples/delta_memory.tg:1,0,0,0:1,2,3,4 tests/bcast/broadcast.tg:0.1,-0.2,0.3,1,2,-1,0,0,0,-3,0.5,0.25
tests/sparse/spmm.tg:0,1,2,0.5,5,-1,0,0,1,1,1,1,9,3,-1,2:0.5,-0.5
tests/sparse/rows.tg:0,1,2,0.5,5,-1,1,1,1,1,9,0.3:0.5,-0.5
tests/attn/window.tg:0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,0.3,0.1,0.4,0.1,0.5,0.9,0.2,0.6,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0.5,0.5,0.5,0.5,-0.5,-0.5,-0.5,-0.5 tests/scan/scan.tg:1,2,3,4,5,6,7,8:1,0.5,-1,2 tests/think/halt.tg:1,1 tests/think/halt.tg:0,0 tests/scan/nested_grad.tg:0.1,0.2,0.3,-0.4,0.5,0.1,-0.2,0.3 examples/selective_ssm.tg:0.1,0,0.2,0,-0.3,1,0.4,0,0.1,0,0.2,1,-0.1,0,0.3,0,0.2,0,-0.4,1,0.1,0,0.2,0:0.1,0.3,-0.3,0.1,0.2,0.2,0.1,0.4,0.6,-0.4,-0.3,-0.1
examples/drift_calibration.tg:10,20,30 tests/native/broadcast.tg:1,2,3:0.5,-1,2,0:$(seq -s, -1 0.1 1.3)"
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
[ "$(grep -c "^ok" "$TMP/gc.out")" -ge 46 ] && ok || bad "gradcheck ran too few cases"
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
# scan + backpropagation through time: a selective SSM learns a resettable sum
perl -e 'srand(7); for (1..20000) { my (@x, @t); my $s = 0; for (1..12) { my $r = rand() < 0.2 ? 1 : 0; my $v = sprintf("%.2f", rand() - 0.5); $s = 0 if $r; $s += $v; push @x, $v, $r; push @t, sprintf("%.4f", $s) } print join(",", @x, @t), "\n" }' > "$TMP/ssm.csv"
$TGC batch examples/selective_ssm.tg "$TMP/ssm.csv" -o "$TMP/ssm.out"
paste -d, "$TMP/ssm.csv" "$TMP/ssm.out" | awk -F, '{ se = 0; z = 0; for (i = 1; i <= 12; i++) { e = $(24 + i) - $(36 + i); se += e * e; z += $(24 + i) ^ 2 } if (NR > 19000) { b += se; zz += z } } END { exit !(b < 0.3 * zz) }' && ok || bad "selective SSM did not learn"
$TGC c examples/selective_ssm.tg -o "$TMP/ssm.c" && $CC -std=c99 -O2 -Wall -Wextra -Werror -pedantic -DTG_MAIN -o "$TMP/ssmb" "$TMP/ssm.c" -lm && ok || bad "BPTT training unit compile"
head -100 "$TMP/ssm.csv" > "$TMP/ssm100.csv"
[ "$("$TMP/ssmb" $(awk -F, '{ x = $1; for (i = 2; i <= 24; i++) x = x "," $i; t = $25; for (i = 26; i <= 36; i++) t = t "," $i; printf "%s %s ", x, t }' "$TMP/ssm100.csv") | tr ' ' ',')" = "$($TGC batch examples/selective_ssm.tg "$TMP/ssm100.csv")" ] && ok || bad "on-device BPTT training differs from VM"
[ "$($TGC check examples/selective_ssm.tg -vv 2>&1 | grep -c 'backpropagation through a 12-step scan')" = 1 ] && ok || bad "BPTT trace"
# word order: a label set by the order of two words; bag of words cannot see it, the SSM layer can
perl -e 'srand(11); my @f=qw(the a film plot actor story it was and very quite); print "text,label\n"; for (1..1500) { my $k = int rand 2; my @w = map { $f[int rand @f] } 1..(2 + int rand 6); my @pair = $k ? ("not", "bad") : ("bad", "not"); my $at = int rand(@w + 1); splice @w, $at, 0, $pair[0]; my $at2 = $at + 1 + int rand(@w - $at); splice @w, $at2, 0, $pair[1]; printf "\"%s\",%s\n", join(" ", @w), $k ? "pos" : "neg" }' > "$TMP/order.csv"
$TGC train "$TMP/order.csv" --epochs 10 -o "$TMP/ord_bow" > "$TMP/ob.log" 2>&1
awk '/best epoch/ { gsub("%)", ""); exit !($NF < 65) }' "$TMP/ob.log" && ok || bad "bag of words should not see word order: $(grep 'best epoch' "$TMP/ob.log")"
$TGC train "$TMP/order.csv" --model ssm --epochs 10 -o "$TMP/ord_ssm" > "$TMP/os.log" 2>&1
grep -q "accuracy 100.00%" "$TMP/os.log" && ok || bad "SSM did not learn word order: $(grep 'best epoch' "$TMP/os.log")"
grep -q "selective SSM over 'text'" "$TMP/os.log" && grep -q "scan h, pv, p, n over" "$TMP/ord_ssm/infer.tg" && ok || bad "ssm model text"
printf 'text\nthe film was not bad\nthe film was bad not\n' > "$TMP/onew.csv"
[ "$($TGC predict "$TMP/ord_ssm" "$TMP/onew.csv" 2>/dev/null | cut -d, -f1 | tr '\n' ' ')" = 'prediction "pos" "neg" ' ] && ok || bad "ssm predict"
$CC -std=c99 -Wall -Werror -c -o "$TMP/oinfer.o" "$TMP/ord_ssm/infer.c" && ok || bad "ssm infer.c does not compile"
# attention: streaming over a KV ring (one token per run) equals the windowed parallel form
perl -e 'srand(3); for (1..7) { print join(",", map { sprintf("%.3f", 2*rand()-1) } 1..8), "\n" }' > "$TMP/seq.csv"
$TGC batch examples/kv_attention.tg "$TMP/seq.csv" | tr ',' '\n' > "$TMP/stream.out"
$TGC run tests/attn/window.tg "$(paste -sd, "$TMP/seq.csv")" | tr ' ' '\n' > "$TMP/par.out"
paste "$TMP/stream.out" "$TMP/par.out" | awk '{ d = $1 - $2; if (d < 0) d = -d; if (d > m) m = d } END { exit !(NR == 56 && m < 1e-5) }' && ok || bad "KV-ring attention differs from windowed attention"
$TGC c examples/kv_attention.tg -o "$TMP/kv.c" && $CC -std=c99 -O2 -Wall -Wextra -Werror -pedantic -DTG_MAIN -o "$TMP/kvb" "$TMP/kv.c" -lm && ok || bad "kv_attention unit compile"
[ "$("$TMP/kvb" $(cat "$TMP/seq.csv") | tr ' ' ',')" = "$($TGC batch examples/kv_attention.tg "$TMP/seq.csv")" ] && ok || bad "C KV ring differs from VM"
# Coconut curriculum: latent think steps trained by BPTT, stage k warm-started from stage k-1
sh examples/coconut/curriculum.sh "$TGC" "$TMP/coconut" 1000 4 > "$TMP/coconut.log" 2>&1
grep -q "stage 4 (4 latent steps, 4-hop answers): 100.0%" "$TMP/coconut.log" && ok || bad "Coconut curriculum: $(tail -3 "$TMP/coconut.log")"
# OS and network host layer: env() initializers, stdin, stream, serve, signals, URL datasets
printf 'model ev\nparam g : f32[3] = env("TG_TEST_GAIN", 1)\ndef forward(x: f32[3]) -> f32[3]:\n    return x * g\n' > "$TMP/ev.tg"
[ "$($TGC run "$TMP/ev.tg" 1,2,3)" = "1 2 3" ] && [ "$(TG_TEST_GAIN=1,0,-1 $TGC run "$TMP/ev.tg" 1,2,3)" = "1 0 -3" ] && [ "$(TG_TEST_GAIN=2 $TGC run "$TMP/ev.tg" 1,2,3)" = "2 4 6" ] && ok || bad "env() initializer"
printf '1,0\n0,0\n1,1\n' > "$TMP/x3.csv"
[ "$($TGC batch examples/xor.tg - < "$TMP/x3.csv")" = "$($TGC batch examples/xor.tg "$TMP/x3.csv")" ] && ok || bad "batch from stdin"
[ "$(printf '1,0\nbad\n\n0,0\n' | $TGC stream examples/xor.tg | tr '\n' '|')" = "1|error: bad number near 'bad'|0|" ] && ok || bad "stream rows"
mkfifo "$TMP/fifo"
$TGC stream examples/delta_memory.tg "$TMP/fifo" -o "$TMP/fifo.out" --save-state "$TMP/fs" & SPID=$!
exec 7>"$TMP/fifo"; printf '1,0,0,0,1,2,3,4\n' >&7; sleep 0.3; kill -TERM $SPID; wait $SPID; st=$?; exec 7>&-
[ $st = 143 ] && [ "$(wc -l < "$TMP/fifo.out")" = 1 ] && [ -s "$TMP/fs.mem.bin" ] && ok || bad "stream SIGTERM: status $st"
if command -v curl > /dev/null; then
	PORT=$((30000 + $$ % 20000))
	$TGC serve examples/delta_memory.tg --listen 127.0.0.1:$PORT --save-state "$TMP/sv" 2> "$TMP/serve.log" & SPID=$!
	for i in $(seq 50); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 0.1; done
	[ "$(curl -s "http://127.0.0.1:$PORT/health")" = ok ] && ok || bad "serve /health"
	curl -s "http://127.0.0.1:$PORT/" | grep -q '"values_per_row":8,"think_loops":0,"stateful":true' && ok || bad "serve signature"
	head -4 tests/state/mem.csv | curl -s --data-binary @- "http://127.0.0.1:$PORT/run" > "$TMP/srv.out"
	sed -n 5,6p tests/state/mem.csv | perl -MIO::Socket::INET -e '$c = IO::Socket::INET->new("127.0.0.1:$ARGV[0]") or die; $c->autoflush(1); while (<STDIN>) { print $c $_; print scalar <$c> }' $PORT >> "$TMP/srv.out"
	kill -TERM $SPID; wait $SPID; st=$?
	head -6 tests/state/mem.csv > "$TMP/mem6.csv"
	$TGC batch examples/delta_memory.tg "$TMP/mem6.csv" -o "$TMP/bat.out" --save-state "$TMP/bt"
	cmp -s "$TMP/srv.out" "$TMP/bat.out" && ok || bad "serve (HTTP + row protocol) differs from batch"
	[ $st = 143 ] && cmp -s "$TMP/sv.mem.bin" "$TMP/bt.mem.bin" && ok || bad "serve SIGTERM: status $st, state not saved"
	perl -MIO::Socket::INET -e '$s = IO::Socket::INET->new(LocalAddr => "127.0.0.1", LocalPort => $ARGV[0], Listen => 1, ReuseAddr => 1) or die; $c = $s->accept; print $c "1,0\n0,0\n1,1\n"; shutdown($c, 1); print while <$c>' $((PORT + 1)) > "$TMP/client.out" & PP=$!
	sleep 0.3; $TGC stream examples/xor.tg "tcp://127.0.0.1:$((PORT + 1))"; wait $PP
	[ "$(tr '\n' ' ' < "$TMP/client.out")" = "1 0 0 " ] && ok || bad "stream as a TCP client"
	perl -MIO::Socket::INET -e '$s = IO::Socket::INET->new(LocalAddr => "127.0.0.1", LocalPort => $ARGV[0], Listen => 1, ReuseAddr => 1) or die; open F, "<", $ARGV[1]; { local $/; $b = <F>; } $c = $s->accept; while (<$c> =~ /\S/) {} print $c "HTTP/1.1 200 OK\r\nContent-Length: " . length($b) . "\r\nConnection: close\r\n\r\n$b"; close $c' $((PORT + 2)) tests/data/messy.csv & PP=$!
	sleep 0.3; $TGC data inspect "http://127.0.0.1:$((PORT + 2))/messy.csv?v=1" | grep -q "rows     5" && ok || bad "dataset over http"; wait $PP
fi
# quantization: int8/int4 weights with per-channel scales (rtn, gptq); VM, TGIR and C agree
perl -e 'srand(5); for (1..300) { my @x = map { sprintf("%.3f", 2*rand()-1) } 1..16; my @s = map { (int(rand 64), sprintf("%.2f", rand())) } 1..3; print join(",", @x, @s), "\n" }' > "$TMP/qcal.csv"
qerr() { $TGC quantize tests/quant/mlp.tg --bits "$1" --method "$2" --calib "$TMP/qcal.csv" -o "$TMP/q$1$2.tgir" 2>&1 | awk '/output error/ { print $(NF - 4) }' | tr -d ,; }
e8=$(qerr 8 rtn); e4=$(qerr 4 rtn); e4g=$(qerr 4 gptq)
awk -v a="$e8" -v b="$e4" -v c="$e4g" 'BEGIN { exit !(a < 0.01 && c < b && b < 0.1) }' && ok || bad "quantization errors int8 $e8, int4 rtn $e4, int4 gptq $e4g"
$TGC ir "$TMP/q4gptq.tgir" > "$TMP/q4b.tgir" && cmp -s "$TMP/q4gptq.tgir" "$TMP/q4b.tgir" && ok || bad "quantized TGIR not a fixed point"
grep -q "(quant 4 1" "$TMP/q4gptq.tgir" && grep -q "(quant 4 0" "$TMP/q4gptq.tgir" && ok || bad "quant axes"
$TGC c "$TMP/q4gptq.tgir" -o "$TMP/q4.c" && $CC -std=c99 -O2 -Wall -Wextra -Werror -pedantic -DTG_MAIN -o "$TMP/q4b" "$TMP/q4.c" -lm && ok || bad "quantized C unit"
grep -q "static const signed char tgq_T" "$TMP/q4.c" && ! grep -q "tgp_W1\[" "$TMP/q4.c" && ok || bad "quantized C unit keeps f32 weights"
head -20 "$TMP/qcal.csv" > "$TMP/qc20.csv"
[ "$(while read r; do "$TMP/q4b" "$(echo "$r" | cut -d, -f1-16)" "$(echo "$r" | cut -d, -f17-)"; done < "$TMP/qc20.csv" | tr ' ' ',')" = "$($TGC batch "$TMP/q4gptq.tgir" "$TMP/qc20.csv")" ] && ok || bad "quantized C differs from VM"
$TGC quantize tests/quant/mlp.tg --method gptq 2>&1 | grep -q "needs --calib" && ok || bad "gptq without calibration"
# fixed point: Q16.16 integer units agree with the float VM; on an FPU-less Cortex-M3 too
fxcmp() { # MODEL INPUT...: max |fixed - float| over the outputs, from the host build
	f=$1; shift
	$TGC c "$f" --fixed -o "$TMP/fx.c" && $CC -std=c99 -O2 -Wall -Wextra -Werror -pedantic -DTG_MAIN -o "$TMP/fx" "$TMP/fx.c" || { echo 1; return; }
	"$TMP/fx" "$@" | head -1 | tr ' ' '\n' > "$TMP/fxa"; $TGC run "$f" "$@" | head -1 | tr ' ' '\n' > "$TMP/fxb"
	paste "$TMP/fxa" "$TMP/fxb" | awk '{ d = $1 - $2; if (d < 0) d = -d; if (d > m) m = d } END { print m + 0 }'
}
for c in "examples/newton.tg 2,9,16" "examples/latent_reasoner.tg 1,0,0,1,0,1,1,0" "tests/cases/ops.tg 1,-2,3,0.5" \
	"tests/sparse/spmm.tg 0,1,2,0.5,5,-1,0,0,1,1,1,1,9,3,-1,2 0.5,-0.5" "tests/scan/scan.tg 1,2,3,4,5,6,7,8 1,0.5,-1,2" \
	"$TMP/q4gptq.tgir 0.1,0.2,-0.3,0.4,0.5,-0.6,0.7,0.8,0.9,-0.1,0.2,0.3,0.4,0.5,0.6,0.7 3,0.5,10,1,63,0.25" \
	"tests/native/broadcast.tg 1,2,3 0.5,-1,2,0 $(seq -s, -1 0.1 1.3)"; do
	set -- $c
	e=$(fxcmp "$@")
	awk -v e="$e" 'BEGIN { exit !(e < 5e-4) }' && ok || bad "fixed point $1: max error $e"
done
sed 's#/\*.*\*/##' "$TMP/fx.c" | grep -v '^ \*' | grep -qwE 'float|double' && bad "fixed-point unit mentions float" || ok
if command -v arm-none-eabi-gcc > /dev/null && command -v qemu-system-arm > /dev/null; then
	$TGC c examples/latent_reasoner.tg --fixed -o "$TMP/lr_fx.c"
	arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -mfloat-abi=soft -O2 -std=c99 -Wall -Wextra -Werror -c "$TMP/lr_fx.c" -o "$TMP/lr_m3.o"
	[ -z "$(arm-none-eabi-nm -u "$TMP/lr_m3.o" | grep -v -E '__aeabi_u?ldivmod$')" ] && ok || bad "fixed-point unit needs float support: $(arm-none-eabi-nm -u "$TMP/lr_m3.o" | tr '\n' ' ')"
	tests/mcu/run.sh cortex-m3 "$TMP/lr_fx.c" "$TMP/lr_m3.elf" 1,0,0,1,0,1,1,0 > "$TMP/m3.out" 2>&1
	head -1 "$TMP/m3.out" | tr ' ' '\n' > "$TMP/m3a"; $TGC run examples/latent_reasoner.tg 1,0,0,1,0,1,1,0 | head -1 | tr ' ' '\n' > "$TMP/m3b"
	paste "$TMP/m3a" "$TMP/m3b" | awk '{ d = $1 - $2; if (d < 0) d = -d; if (d > m) m = d } END { exit !(NR == 4 && m < 1e-4) }' && grep -q "steps 0 9" "$TMP/m3.out" && ok || bad "Cortex-M3 run: $(cat "$TMP/m3.out")"
fi
# native backends (tgc asm): Cortex-M3 Q16.16 with SMLAL, Cortex-M4F with VFMA, RV64GCV with RVV; against the VM
NATIVE="examples/latent_reasoner.tg:1,0,0,1,0,1,1,0 tests/cases/ops.tg:1,-2,3,0.5 tests/scan/scan.tg:1,2,3,4,5,6,7,8:1,0.5,-1,2
tests/think/halt.tg:0,0 tests/sparse/rows.tg:0,1,2,0.5,5,-1,1,1,1,1,9,0.3:0.5,-0.5
tests/native/matmul.tg:1,2,3,4,5,6,7:$(seq -s, 0.1 0.1 3.5):1,-1,2,-2,0.5:0.5:1,2,3
examples/selective_ssm.tg:0.1,0,0.2,0,-0.3,1,0.4,0,0.1,0,0.2,1,-0.1,0,0.3,0,0.2,0,-0.4,1,0.1,0,0.2,0:0.1,0.3,-0.3,0.1,0.2,0.2,0.1,0.4,0.6,-0.4,-0.3,-0.1
$TMP/q4gptq.tgir:0.1,0.2,-0.3,0.4,0.5,-0.6,0.7,0.8,0.9,-0.1,0.2,0.3,0.4,0.5,0.6,0.7:3,0.5,10,1,63,0.25
tests/native/broadcast.tg:1,2,3:0.5,-1,2,0:$(seq -s, -1 0.1 1.3)"
perl -e 'srand(3); for (1..6) { print join(",", map { sprintf("%.1f", 2*rand()-1) } 1..8), "\n" }' > "$TMP/seq6.csv"
native() { # TARGET TOLERANCE
	for c in $NATIVE; do
		f=${c%%:*}
		r=$(tests/mcu/native.sh "$TGC" "$TMP" "$1" "$f" $(echo "${c#*:}" | tr ':' ' ') 2>&1 | tail -1)
		echo "$r" | awk -v t="$2" '{ exit !(NF == 2 && $1 < t && $2 == "same") }' && ok || bad "native $1 $f: $r"
	done
	tests/mcu/native.sh "$TGC" "$TMP" "$1" examples/delta_memory.tg 1,0,0,0 1,2,3,4 0,1,0,0 5,6,7,8 1,0,0,0 0,0,0,0 0,1,0,0 0,0,0,0 1,0,0,0 9,9,9,9 > /dev/null 2>&1
	tr ' ' '\n' < "$TMP/native_$1.out" | awk '{ printf "%s%g", NR % 4 == 1 ? (NR > 1 ? "\n" : "") : ",", $1 } END { print "" }' > "$TMP/nat_dm"
	cmp -s "$TMP/nat_dm" tests/state/mem.expected && ok || bad "native $1 self-updating state: $(tr '\n' ' ' < "$TMP/nat_dm")"
	tests/mcu/native.sh "$TGC" "$TMP" "$1" examples/kv_attention.tg $(cat "$TMP/seq6.csv") > /dev/null 2>&1
	$TGC batch examples/kv_attention.tg "$TMP/seq6.csv" | tr ',' '\n' > "$TMP/nat_kvb"
	tr ' ' '\n' < "$TMP/native_$1.out" | paste - "$TMP/nat_kvb" | awk -v t="$2" '{ d = $1 - $2; if (d < 0) d = -d; if (d > m) m = d } END { exit !(NR == 48 && m < t) }' && ok || bad "native $1 KV ring (row updates)"
}
if command -v arm-none-eabi-gcc > /dev/null && command -v qemu-system-arm > /dev/null; then
	native cortex-m3 5e-4
	$TGC c examples/latent_reasoner.tg --fixed -o "$TMP/lrh.c" && $CC -std=c99 -O2 -DTG_MAIN -o "$TMP/lrh" "$TMP/lrh.c"
	tests/mcu/native.sh "$TGC" "$TMP" cortex-m3 examples/latent_reasoner.tg 1,0,0,1,0,1,1,0 > /dev/null 2>&1
	"$TMP/lrh" 1,0,0,1,0,1,1,0 | cmp -s - "$TMP/native_cortex-m3.out" && ok || bad "Cortex-M3 native not bit-identical to the fixed-point C unit"
	arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -mfloat-abi=soft -O2 -std=c99 -Wall -Wextra -Werror -c "$TMP/native_cortex-m3.c" -o "$TMP/nat3.o" \
		&& arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -c "$TMP/native_cortex-m3.s" -o "$TMP/nat3s.o" \
		&& [ -z "$(arm-none-eabi-nm -u "$TMP/nat3.o" "$TMP/nat3s.o" | grep -v -E '^$|:$|__aeabi_u?ldivmod$|memset$|tg_matmul$|tg_latent_reasoner_(run|steps)$|tg_(copy|k_[a-z]+|rmsnorm|softmax|tanh)$')" ] \
		&& ok || bad "Cortex-M3 native unit needs float support: $(arm-none-eabi-nm -u "$TMP/nat3.o" "$TMP/nat3s.o" | tr '\n' ' ')"
	native cortex-m4f 1e-5
fi
if command -v riscv64-linux-gnu-gcc > /dev/null && command -v qemu-riscv64-static > /dev/null; then
	native rv64gcv 1e-5
	r=$(TG_VLEN=256 tests/mcu/native.sh "$TGC" "$TMP" rv64gcv tests/native/matmul.tg 1,2,3,4,5,6,7 "$(seq -s, 0.1 0.1 3.5)" 1,-1,2,-2,0.5 0.5 1,2,3 2>&1 | tail -1)
	echo "$r" | awk '{ exit !(NF == 2 && $1 < 1e-5 && $2 == "same") }' && ok || bad "rv64gcv at VLEN 256: $r"
fi
# learned halting (PonderNet): steps follow the difficulty; a fixed budget overshoots
sh examples/ponder/run.sh "$TGC" "$TMP/ponder" 20000 > "$TMP/ponder.log" 2>&1
grep -q "all: accuracy 100.0%" "$TMP/ponder.log" && grep -q "hops 1: accuracy 100.0%, mean steps 1.00" "$TMP/ponder.log" \
	&& grep -q "hops 4: accuracy 100.0%, mean steps 4.00" "$TMP/ponder.log" && ok || bad "learned halting: $(tr '\n' ' ' < "$TMP/ponder.log")"
[ "$($TGC run tests/think/halt.tg 1,1 | tail -1)" = "steps 0 5" ] && [ "$($TGC run tests/think/halt.tg 0,0 | tail -1)" = "steps 0 19" ] && ok || bad "halt step counts"
# row-sparse (lazy) optimizer steps on spmm tables
lz() { sed "s/OPT/$1/" > "$TMP/lz.tg" <<'TG'
model lz
state table : f32[DIM, 3] = rand(3, 0.5)
param head : f32[3] = [1, -0.5, 0.25]
def forward(s: f32[2, 3, 2], t: f32[2]) -> f32[2]:
    y = tanh(spmm(s, table)) @ head
    train mse(y, t) with OPT
    return y
TG
sed -i "s/DIM/$2/" "$TMP/lz.tg"; $TGC batch "$TMP/lz.tg" "$3"; }
perl -e 'srand(4); for (1..60) { print join(",", map { (int(rand 12), sprintf("%.3f", rand() - 0.5)) } 1..6), sprintf(",%.3f,%.3f\n", rand() - 0.5, rand() - 0.5) }' > "$TMP/lz.csv"
[ "$(lz 'sgd(lr=0.5)' 12 "$TMP/lz.csv")" = "$(lz 'sgd(lr=0.5, lazy=0)' 12 "$TMP/lz.csv")" ] && ok || bad "lazy sgd differs from dense sgd"
perl -e 'srand(5); for (1..60) { print "0,0.3,1,-0.2,2,0.4,2,0.1,1,0.5,0,-0.3,", sprintf("%.3f,%.3f\n", rand() - 0.5, rand() - 0.5) }' > "$TMP/lzall.csv"   # every row, every step
[ "$(lz 'adamw(lr=0.05)' 3 "$TMP/lzall.csv")" = "$(lz 'adamw(lr=0.05, lazy=0)' 3 "$TMP/lzall.csv")" ] && ok || bad "lazy adamw differs from dense when every row is touched"
lz 'adamw(lr=0.05)' 12 "$TMP/lz.csv" > /dev/null
$TGC ir "$TMP/lz.tg" | grep -q "(update table %[0-9]* %[0-9]*)" && ok || bad "no row update in TGIR"
[ "$($TGC run "$TMP/lz.tg" 1,0.5,2,0.5,3,0.5,4,0.5,5,0.5,6,0.5 0.1,0.2 -v 2>&1 | grep -c 'state table: 6 of 12 rows updated')" = 1 ] && ok || bad "row update trace"
# text + numeric: hashed words enter as sparse ELLPACK rows through spmm
perl -e 'srand(5); my @p=qw(great superb lovely fun moving); my @n=qw(awful dull boring weak bland); my @f=qw(the a film plot actor story it was and); print "review,score,mood\n"; for (1..600) { my $k=$_%2; my @w=map { $f[int rand @f] } 1..(3+int rand 6); push @w, ($k ? $p[int rand @p] : $n[int rand @n]) for 1..2; @w = sort { rand() <=> 0.5 } @w; printf "\"%s\",%.2f,%s\n", join(" ",@w), rand(), $k ? "pos" : "neg" }' > "$TMP/rev.csv"
$TGC train "$TMP/rev.csv" --target mood --epochs 40 -o "$TMP/rev_model" > "$TMP/rev.log" 2>&1 || bad "text train failed: $(tail -3 "$TMP/rev.log")"
grep -q "1 dense, 32768 sparse text buckets, <= 10 active per row" "$TMP/rev.log" && ok || bad "text not sparse: $(grep features "$TMP/rev.log")"
grep -q "accuracy 100.00%" "$TMP/rev.log" && ok || bad "text not learned: $(grep 'best epoch' "$TMP/rev.log")"
grep -q "spmm(s, wt)" "$TMP/rev_model/infer.tg" && ok || bad "infer.tg without spmm"
printf 'review\nwhat a lovely and moving story\nthe plot was dull\n' > "$TMP/rnew.csv"   # one string column: header still detected
[ "$($TGC predict "$TMP/rev_model" "$TMP/rnew.csv" 2>/dev/null | cut -d, -f1 | tr '\n' ' ')" = 'prediction "pos" "neg" ' ] && ok || bad "text predict"
$CC -std=c99 -Wall -Werror -c -o "$TMP/rinfer.o" "$TMP/rev_model/infer.c" && ok || bad "sparse infer.c does not compile"
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
