#!/bin/sh
# Teach the byte-level sequence model a fixed-key classical cipher from
# (plaintext, ciphertext) example pairs, then measure exact-match on held-out
# inputs it never saw -- the cryptography-CTF shape: an exact-string transform
# learned from examples. Classical/educational ciphers only.
#   run.sh TGC WORKDIR [TRANSFORM...] [--epochs N] [--layers N]
# Position-wise length-preserving ciphers train in --aligned mode (output byte
# i from input byte i, one pass); length-changing or permuting ones use the
# general autoregressive mode.
set -e
tgc=$1; w=${2:?usage: run.sh TGC WORKDIR [TRANSFORM...]}; shift 2
here=$(cd "$(dirname "$0")" && pwd)
ep=14; layers=3; tfs=""
while [ $# -gt 0 ]; do
	case $1 in
		--epochs) ep=$2; shift 2 ;;
		--layers) layers=$2; shift 2 ;;
		*) tfs="$tfs $1"; shift ;;
	esac
done
[ -n "$tfs" ] || tfs="rot13 caesar5 atbash vigenere sub reverse xor base64 morse"
mkdir -p "$w"
printf '%-9s %-12s %10s   %s\n' transform mode exact-match "val byte-loss"
for tf in $tfs; do
	case $tf in
		rot13|caesar5|atbash|vigenere|sub) mode=--aligned; label=aligned ;;
		*) mode=; label=autoregressive ;;
	esac
	perl "$here/gen.pl" "$tf" 5000 1   > "$w/$tf.jsonl"
	perl "$here/gen.pl" "$tf" 500  999 > "$w/${tf}_test.jsonl"   # disjoint seed: unseen inputs
	"$tgc" gen-train "$w/$tf.jsonl" -o "$w/$tf" $mode --epochs "$ep" --layers "$layers" --maxlen 80 -j auto > "$w/$tf.log" 2>&1
	em=$("$tgc" generate "$w/$tf" --inputs "$w/${tf}_test.jsonl" -o /dev/null 2>&1 | sed -n 's/.*exact-match \([0-9.]*\)%.*/\1/p')
	vl=$(sed -n 's/^best epoch.*loss \([0-9.]*\).*/\1/p' "$w/$tf/report.txt")
	printf '%-9s %-12s %9s%%   %s\n' "$tf" "$label" "$em" "$vl"
done
