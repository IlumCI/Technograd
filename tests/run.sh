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
examples/use_import.tg:1,-1,2,-2"
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
