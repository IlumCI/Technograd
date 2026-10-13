#!/usr/bin/perl
# In-context associative recall with a delta-rule memory written while the
# sequence is read. Both models share one scan; they differ only in the
# address a_t at which a token writes and reads:
#   dense   a = k / |k|, k in R^dk          (DeltaNet; arXiv:2406.06484)
#   sparse  a = softmax over the top 8 of S slot scores, zero elsewhere
#           (sparse delta memory; arXiv:2607.07386)
# Step t reads r = a M and, on a write step, applies the delta rule
#   M += beta w a^T (v - r)
# so a write at an address that already holds a value replaces it.
# Usage: models.pl KIND STEPS DIR [SLOTS]   (KIND dense|sparse; SLOTS = dk or S)
use strict;
my ($kind, $T, $dir, $S) = @ARGV;
my ($B, $NK, $NV, $d, $dv) = (16, 64, 64, 32, 32);
$S ||= $kind eq 'dense' ? 16 : 256;
my $lr = $ENV{LR} || 0.003;
my $TB = $T * $B;
my $addr = $kind eq 'dense'
	? "    k = rmsnorm(take(Ek, kid) \@ Wk) * " . sprintf("%.6g", 1 / sqrt($S)) . "\n    A = reshape(k, $T, $B, $S)\n"
	: "    sc = (rmsnorm(take(Ek, kid) \@ Wk) \@ transpose(rmsnorm(Keys))) * temp\n"
	. "    j = topk(sc, 8)\n"
	. "    a8 = softmax(take_along(sc, j))\n"
	. "    A = reshape(reshape(reshape(a8, $TB, 1, 8) \@ reshape(take(I_S, reshape(j, " . ($TB * 8) . ")), $TB, 8, $S), $TB, $S), $T, $B, $S)\n";
my $st = "state Ek : f32[$NK, $d] = rand(1, 1)\nstate Ev : f32[$NV, $dv] = rand(2, 1)\nstate Wk : f32[$d, " . ($kind eq 'dense' ? $S : 16) . "] = rand(3, 0.3)\n"
	. "state Wo : f32[$dv, $NV] = rand(4, 0.2)\nstate bw : f32 = 2\n";
$st .= "state Keys : f32[$S, 16] = rand(5, 1)\nstate temp : f32 = 4\nparam I_S : f32[$S, $S] = file(\"eye_s.bin\")\n" if $kind eq 'sparse';
open my $f, '>', "$dir/train.tg" or die;
print $f <<"M";
model recall_$kind
$st
param I_V : f32[$NV, $NV] = file("eye_v.bin")
param onesV : f32[$NV] = ones
param M0 : f32[$B, $S, $dv] = zeros

# kid, vid: key and value token of each step (time-major); w: 1 on a write
# step, 0 on a query step, whose target is vid.
def forward(kid: f32[$TB], vid: f32[$TB], w: f32[$TB]) -> f32:
$addr    V = reshape(take(Ev, vid) * reshape(w, $TB, 1), $T, $B, $dv)
    beta = sigmoid(bw)
    W = reshape(w, $T, $B, 1, 1) * beta
    M = M0
    scan M over at in A, vt in V, wt in W:
        r = reshape(reshape(at, $B, 1, $S) \@ M, $B, $dv)
        emit rs = r
        M = M + reshape(at, $B, $S, 1) * wt * reshape(vt - r, $B, 1, $dv)
    q = 1 - w
    logits = reshape(rs, $TB, $dv) \@ Wo
    tv = take(I_V, vid)
    loss = -sum(reshape(q, $TB, 1) * tv * log(softmax(logits) + 0.000000001)) / sum(q)
    train loss with adamw(lr=$lr, wd=0, clip=1)
    hit = (tv * take(I_V, reshape(topk(logits, 1), $TB))) @ onesV
    return sum(q * hit) / sum(q)
M
for my $p (["eye_s.bin", $S], ["eye_v.bin", $NV]) {
	next if $p->[0] eq 'eye_s.bin' && $kind ne 'sparse';
	open my $e, '>:raw', "$dir/$p->[0]" or die;
	my $n = $p->[1];
	print $e pack('f<*', map { my $r = $_; map { $_ == $r ? 1 : 0 } 0 .. $n - 1 } 0 .. $n - 1);
}
