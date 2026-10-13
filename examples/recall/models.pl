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
# KIND rows is the sparse model with the same arithmetic, but each step reads
# its 8 rows with take and writes them with take_t (scatter-add) instead of
# multiplying by the full address. KIND mem is it again as one memory_scan,
# whose steps touch only the addressed rows and whose backward pass undoes
# the writes instead of storing the memory at every step.
# KIND pkm is mem with product-key addressing (arXiv:1907.05242): two sets of
# sqrt(S) sub-keys, the top 8 of each, and the best 8 of their 64 sums, so
# scoring costs O(sqrt S) instead of O(S).
# Usage: models.pl KIND STEPS DIR [SLOTS]   (KIND dense|sparse|rows|mem|pkm; SLOTS = dk or S)
use strict;
my ($kind, $T, $dir, $S) = @ARGV;
my ($B, $NV, $d, $dv) = (16, 64, 32, 32);
my $NK = $ENV{NK} || 64; # distinct keys
$S ||= $kind eq 'dense' ? 16 : 256;
my $lr = $ENV{LR} || 0.003;
my $TB = $T * $B;
my $sp = $kind ne 'dense';
my $pk = $kind eq 'pkm';
my $n = int(sqrt($S) + 0.5);
die "pkm needs a square slot count\n" if $pk && $n * $n != $S;
$kind = 'mem' if $pk; # the same memory, another address
my $addr = $pk
	? "    x = take(Ek, kid)\n"
	. "    s1 = (rmsnorm(x \@ Wk1) \@ transpose(rmsnorm(K1))) * temp\n"
	. "    s2 = (rmsnorm(x \@ Wk2) \@ transpose(rmsnorm(K2))) * temp\n"
	. "    i1 = topk(s1, 8)\n    i2 = topk(s2, 8)\n"
	. "    cs = reshape(reshape(take_along(s1, i1), $TB, 8, 1) + reshape(take_along(s2, i2), $TB, 1, 8), $TB, 64)\n"
	. "    ci = reshape(reshape(i1, $TB, 8, 1) * $n + reshape(i2, $TB, 1, 8), $TB, 64)\n"
	. "    j0 = topk(cs, 8)\n    j = take_along(ci, j0)\n    a8 = softmax(take_along(cs, j0))\n"
	. "    J = reshape(reshape(j, $T, $B, 8) + boff, $T, $B, 8, 1)\n    a4 = reshape(a8, $T, $B, 8, 1)\n"
	. "    P = J * pidx + a4 * prd + a4 * reshape(w, $T, $B, 1, 1) * sigmoid(bw) * pwr\n"
	: $kind eq 'dense'
	? "    k = rmsnorm(take(Ek, kid) \@ Wk) * " . sprintf("%.6g", 1 / sqrt($S)) . "\n    A = reshape(k, $T, $B, $S)\n"
	: "    sc = (rmsnorm(take(Ek, kid) \@ Wk) \@ transpose(rmsnorm(Keys))) * temp\n"
	. "    j = topk(sc, 8)\n"
	. "    a8 = softmax(take_along(sc, j))\n"
	. ($kind eq 'mem' ? "    J = reshape(reshape(j, $T, $B, 8) + boff, $T, $B, 8, 1)\n    a4 = reshape(a8, $T, $B, 8, 1)\n"
	. "    P = J * pidx + a4 * prd + a4 * reshape(w, $T, $B, 1, 1) * sigmoid(bw) * pwr\n"
	: $kind eq 'rows' ? "    J = reshape(reshape(j, $T, $B, 8) + boff, $T, " . ($B * 8) . ")\n    A = reshape(a8, $T, $B, 8)\n"
	: "    A = reshape(reshape(reshape(a8, $TB, 1, 8) \@ reshape(take(I_S, reshape(j, " . ($TB * 8) . ")), $TB, 8, $S), $TB, $S), $T, $B, $S)\n");
my $st = "state Ek : f32[$NK, $d] = rand(1, 1)\nstate Ev : f32[$NV, $dv] = rand(2, 1)\nstate Wk : f32[$d, " . ($kind eq 'dense' ? $S : 16) . "] = rand(3, 0.3)\n"
	. "state Wo : f32[$dv, $NV] = rand(4, 0.2)\nstate bw : f32 = 2\n";
$st .= $pk ? "state Wk1 : f32[$d, 8] = rand(6, 0.3)\nstate Wk2 : f32[$d, 8] = rand(7, 0.3)\nstate K1 : f32[$n, 8] = rand(8, 1)\nstate K2 : f32[$n, 8] = rand(9, 1)\nstate temp : f32 = 4\n"
	: "state Keys : f32[$S, 16] = rand(5, 1)\nstate temp : f32 = 4\n" if $sp;
$st .= "param I_S : f32[$S, $S] = file(\"eye_s.bin\")\n" if $kind eq 'sparse';
$st .= "param boff : f32[$B, 1] = [" . join(", ", map { "[" . $_ * $S . "]" } 0 .. $B - 1) . "]\n" if $kind eq 'rows' || $kind eq 'mem';
$st .= "param pidx : f32[1, 3] = [[1, 0, 0]]\nparam prd : f32[1, 3] = [[0, 1, 0]]\nparam pwr : f32[1, 3] = [[0, 0, 1]]\n" if $kind eq 'mem';
my $scan = $kind eq 'mem' ? "    rs = memory_scan(M0f, P, reshape(take(Ev, vid), $T, $B, $dv))\n" : $kind eq 'rows' ? <<"R" : <<"D";
    M = M0f
    scan M over jt in J, at in A, vt in V, wt in W:
        g = reshape(take(M, jt), $B, 8, $dv)
        r = reshape(reshape(at, $B, 1, 8) \@ g, $B, $dv)
        emit rs = r
        M = M + take_t(jt, reshape(reshape(at, $B, 8, 1) * wt * reshape(vt - r, $B, 1, $dv), ${\($B * 8)}, $dv), M)
R
    M = M0
    scan M over at in A, vt in V, wt in W:
        r = reshape(reshape(at, $B, 1, $S) \@ M, $B, $dv)
        emit rs = r
        M = M + reshape(at, $B, $S, 1) * wt * reshape(vt - r, $B, 1, $dv)
D
open my $f, '>', "$dir/train.tg" or die;
print $f <<"M";
model recall_$kind
$st
param I_V : f32[$NV, $NV] = file("eye_v.bin")
param onesV : f32[$NV] = ones
param M0 : f32[$B, $S, $dv] = zeros
param M0f : f32[${\($B * $S)}, $dv] = zeros

# kid, vid: key and value token of each step (time-major); w: 1 on a write
# step, 0 on a query step, whose target is vid.
def forward(kid: f32[$TB], vid: f32[$TB], w: f32[$TB]) -> f32:
$addr    V = reshape(take(Ev, vid) * reshape(w, $TB, 1), $T, $B, $dv)
    beta = sigmoid(bw)
    W = reshape(w, $T, $B, 1, 1) * beta
$scan    q = 1 - w
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
