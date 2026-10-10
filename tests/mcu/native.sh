#!/bin/sh
# Run a model through a native backend and compare with the VM.
# Usage: native.sh TGC DIR TARGET MODEL INPUT...
#   TARGET: cortex-m3 | cortex-m4f (QEMU mps2 boards) | rv64gcv (qemu-riscv64,
#   VLEN from $TG_VLEN, default 128). Prints "ERR STEPS": the largest output
#   difference relative to max(1, |vm|), and "same" or "differ" for the
#   think/scan step counts. Exits 1 if the unit fails to build or run.
set -e
tgc=$1; dir=$2; target=$3; model=$4; shift 4
here=$(cd "$(dirname "$0")" && pwd)
b=$dir/native_$target
"$tgc" asm "$model" --target "$target" -o "$b.s"
"$tgc" asm "$model" --target "$target" --support -o "$b.c"
case $target in
	cortex-m3) TG_EXTRA="$b.s" "$here/run.sh" cortex-m3 "$b.c" "$b.elf" "$@" > "$b.out" 2>&1 ;;
	cortex-m4f) TG_EXTRA="$b.s" "$here/run.sh" cortex-m4 "$b.c" "$b.elf" "$@" > "$b.out" 2>&1 ;;
	rv64gcv)
		riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -O2 -std=c99 -static -DTG_MAIN -o "$b.elf" "$b.c" "$b.s" -lm
		qemu-riscv64-static -cpu "rv64,v=true,vext_spec=v1.0,vlen=${TG_VLEN:-128}" "$b.elf" "$@" > "$b.out" ;;
	*) echo "unknown target $target" >&2; exit 2 ;;
esac
"$tgc" run "$model" "$@" > "$b.vm"
grep -v '^steps' "$b.out" | tr ' ' '\n' > "$b.a"; grep -v '^steps' "$b.vm" | tr ' ,' '\n\n' > "$b.b"
[ "$(wc -l < "$b.a")" = "$(wc -l < "$b.b")" ] || { echo "output count differs: $(cat "$b.out")" >&2; exit 1; }
e=$(paste "$b.a" "$b.b" | awk '{ d = $1 - $2; if (d < 0) d = -d; s = $2 < 0 ? -$2 : $2; if (s < 1) s = 1; if (d / s > m) m = d / s } END { print m + 0 }')
[ "$(grep '^steps' "$b.out")" = "$(grep '^steps' "$b.vm")" ] && st=same || st=differ
echo "$e $st"
