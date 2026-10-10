#!/bin/sh
# Build a C unit for an emulated Cortex-M board and run it under QEMU with
# semihosting. Usage: run.sh CPU UNIT.c OUT.elf ARG...
#   CPU: cortex-m3 (mps2-an385, no FPU) or cortex-m4 (mps2-an386, FPU + DSP)
# Each ARG is one model input (comma-separated values). TG_EXTRA: more sources
# to link (the assembly of a native unit).
set -e
cpu=$1; src=$2; elf=$3; shift 3
dir=$(cd "$(dirname "$0")" && pwd)
case $cpu in
	cortex-m3) board=mps2-an385; fl="-mfloat-abi=soft" ;;
	cortex-m4) board=mps2-an386; fl="-mfloat-abi=hard -mfpu=fpv4-sp-d16" ;;
	*) echo "unknown cpu $cpu" >&2; exit 2 ;;
esac
arm-none-eabi-gcc -mcpu=$cpu -mthumb $fl -O2 -std=c99 -DTG_MAIN --specs=rdimon.specs -T "$dir/mps2.ld" \
	-o "$elf" "$src" $TG_EXTRA "$dir/startup.c" -lm
args="arg=unit"
for a in "$@"; do args="$args,arg=$(printf %s "$a" | sed 's/,/,,/g')"; done
timeout 60 qemu-system-arm -M $board -nographic -monitor none -semihosting-config "enable=on,target=native,$args" -kernel "$elf"
