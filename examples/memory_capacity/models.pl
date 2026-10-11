#!/usr/bin/perl
# Writes the train and inference models of the capacity experiment.
# Usage: models.pl KIND F DIR [n] [H]
#   KIND dense: x -> gelu(x W1 + b1) W2, width H
#   KIND pkm:   x -> product-key memory (two sub-key sets of n, n^2 slots,
#               top 8 of 8x8 candidates, values via spmm so AdamW steps only
#               the touched rows) -> W2
# Both read a fixed random 64-d key for fact id, and are trained to emit the
# fact's class (256 classes).
use strict;
my ($kind, $F, $dir, $n, $H) = @ARGV;
$n ||= 128; $H ||= 96;
my ($B, $D, $C, $dq, $k, $dv) = (32, 64, 256, 32, 8, 64);
my $lr = $ENV{LR} || ($kind eq 'pkm' ? 0.003 : 0.003);
my (@st, $body);
if ($kind eq 'dense') {
	@st = ([W1 => "$D, $H", "rand(1, 0.125)"], [b1 => "$H", "zeros"], [W2 => "$H, $C", "rand(2, 0.1)"]);
	$body = "    logits = gelu(x \@ W1 + b1) \@ W2\n";
} else {
	my ($M, $kk) = ($n * $n, $k * $k);
	my $vs = sprintf "%.4f", 1 / sqrt($dv);
	@st = ([Wq1 => "$D, $dq", "rand(1, 0.125)"], [Wq2 => "$D, $dq", "rand(2, 0.125)"],
		[C1 => "$n, $dq", "rand(3, 1)"], [C2 => "$n, $dq", "rand(4, 1)"],
		[V => "$M, $dv", "rand(5, $vs)"], [W2 => "$dv, $C", "rand(6, 0.125)"]);
	$body = <<"T";
    s1 = rmsnorm(x \@ Wq1) \@ transpose(C1)
    s2 = rmsnorm(x \@ Wq2) \@ transpose(C2)
    i1 = topk(s1, $k)
    i2 = topk(s2, $k)
    cs = reshape(reshape(take_along(s1, i1), $B, $k, 1) + reshape(take_along(s2, i2), $B, 1, $k), $B, $kk)
    ci = reshape(reshape(i1, $B, $k, 1) * $n + reshape(i2, $B, 1, $k), $B, $kk)
    j = topk(cs, $k)
    w = softmax(take_along(cs, j))
    pairs = reshape(take_along(ci, j), $B, $k, 1) * e0 + reshape(w, $B, $k, 1) * e1
    logits = spmm(pairs, V) \@ W2
T
}
my $consts = "param Kf : f32[$F, $D] = rand(9, 1)\n";
$consts .= "param e0 : f32[1, 2] = [[1, 0]]\nparam e1 : f32[1, 2] = [[0, 1]]\n" if $kind eq 'pkm';
open my $t, '>', "$dir/train.tg" or die;
print $t "model cap_${kind}_train\n\n", (map { "state $_->[0] : f32[$_->[1]] = $_->[2]\n" } @st), $consts,
	"param I : f32[$C, $C] = file(\"eye.bin\")\n\n",
	"def forward(id: f32[$B], t: f32[$B]) -> f32[$B, 1]:\n    x = take(Kf, id)\n$body",
	"    train xent(logits, take(I, t)) with adamw(lr=$lr, wd=0)\n    return topk(logits, 1)\n";
open my $i, '>', "$dir/infer.tg" or die;
print $i "model cap_$kind\n\n", (map { "param $_->[0] : f32[$_->[1]] = file(\"w.$_->[0].bin\")\n" } @st), $consts,
	"\ndef forward(id: f32[$B]) -> f32[$B, 1]:\n    x = take(Kf, id)\n$body    return topk(logits, 1)\n";
open my $e, '>:raw', "$dir/eye.bin" or die;
print $e pack('f<*', map { my $r = $_; map { $_ == $r ? 1 : 0 } 0 .. $C - 1 } 0 .. $C - 1);
