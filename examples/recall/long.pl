#!/usr/bin/perl
# Long-context recall with a learned write gate. A sequence of T steps holds
# P writes (key, value) at random positions in its first three quarters,
# Q queries (key, QUERY) in its last quarter, and filler (random key, FILL)
# everywhere else. Nothing marks which steps write: the gate
# sigmoid(value embedding . wg + bg) must learn to write values and to skip
# QUERY and FILL. The loss and the score are taken at the queries.
#
#   long.pl data P Q T B ROWS SEED        rows of kid, vid, tid, q (T*B each, time-major)
#   long.pl model KIND T B DIR [S] [MODE]  KIND pkm|dense; MODE train (default), eval, stream
#
# pkm: memory_scan over S = n^2 slots per sequence, product-key address (two
# sets of n sub-keys, top 8 each, best 8 of the 64 sums). dense: a delta-rule
# state of S = dk dimensions (DeltaNet), the same gate. eval loads the
# weights a training run saved (w.*.bin) and returns (hits, queries); stream
# carries the memory between runs in a state, so a long sequence is fed in
# chunks of T steps (input reset = 1 on a sequence's first chunk).
use strict;
my $cmd = shift;
my ($NK, $NV, $d, $dv) = (512, 64, 32, 32);
my ($QRY, $FILL) = ($NV, $NV + 1);
if ($cmd eq 'data') {
	my ($P, $Q, $T, $B, $n, $seed) = @_ = @ARGV;
	my $W = int(0.75 * $T);
	die "P must be <= 3T/4 and <= $NK, Q <= T/4\n" if $P > $W || $P > $NK || $Q > $T - $W;
	srand($seed);
	for (1 .. $n) {
		my (@k, @v, @t, @q);
		for my $b (0 .. $B - 1) {
			my @keys = 0 .. $NK - 1;
			for (my $i = $#keys; $i > $#keys - $P; $i--) { my $j = int rand($i + 1); @keys[$i, $j] = @keys[$j, $i] }
			my @ks = @keys[$NK - $P .. $NK - 1];
			my %val = map { $_ => int rand $NV } @ks;
			my %wp = pick($P, 0, $W);
			my @wpos = sort { $a <=> $b } keys %wp;
			my %qp = pick($Q, $W, $T);
			my %at;
			@at{@wpos} = @ks;
			for my $t (0 .. $T - 1) {
				my $i = $t * $B + $b;
				if (exists $at{$t}) { ($k[$i], $v[$i], $t[$i], $q[$i]) = ($at{$t}, $val{$at{$t}}, 0, 0) }
				elsif (exists $qp{$t}) { my $key = $ks[int rand $P]; ($k[$i], $v[$i], $t[$i], $q[$i]) = ($key, $QRY, $val{$key}, 1) }
				else { ($k[$i], $v[$i], $t[$i], $q[$i]) = (int rand $NK, $FILL, 0, 0) }
			}
		}
		print join(",", @k, @v, @t, @q), "\n";
	}
	exit;
}
sub pick { my ($n, $lo, $hi) = @_; my %s; while (keys %s < $n) { $s{$lo + int rand($hi - $lo)} = 1 } %s }
die "usage: long.pl data|model ...\n" unless $cmd eq 'model';
my ($kind, $T, $B, $dir, $S, $mode) = @ARGV;
$mode ||= 'train';
$S ||= $kind eq 'dense' ? 16 : 4096;
my $TB = $T * $B;
my $n = int(sqrt($S) + 0.5);
die "pkm needs a square slot count\n" if $kind eq 'pkm' && $n * $n != $S;
my $lr = $ENV{LR} || 0.003;
my @st = (["Ek", "$NK, $d", "rand(1, 1)"], ["Ev", ($NV + 2) . ", $dv", "rand(2, 1)"], ["Wo", "$dv, $NV", "rand(4, 0.2)"],
	["wg", "$dv", "rand(10, 0.2)"], ["bg", "", "0"]);
push @st, $kind eq 'dense' ? (["Wk", "$d, $S", "rand(3, 0.3)"])
	: (["Wk1", "$d, 8", "rand(6, 0.3)"], ["Wk2", "$d, 8", "rand(7, 0.3)"], ["K1", "$n, 8", "rand(8, 1)"], ["K2", "$n, 8", "rand(9, 1)"], ["temp", "", "4"]);
my $decl = join "", map {
	my $sh = $_->[1] eq '' ? "f32" : "f32[$_->[1]]";
	$mode eq 'train' ? "state $_->[0] : $sh = $_->[2]\n" : "param $_->[0] : $sh = file(\"w.$_->[0].bin\")\n"
} @st;
$decl .= "param I_V : f32[$NV, $NV] = file(\"eye_v.bin\")\nparam onesV : f32[$NV] = ones\n";
my $body = "    gate = sigmoid(take(Ev, vid) \@ wg + bg)\n";
if ($kind eq 'dense') {
	$decl .= "param M0 : f32[$B, $S, $dv] = zeros\n";
	$body .= "    A = reshape(rmsnorm(take(Ek, kid) \@ Wk) * " . sprintf("%.6g", 1 / sqrt($S)) . ", $T, $B, $S)\n"
		. "    V = reshape(take(Ev, vid), $T, $B, $dv)\n    W = reshape(gate, $T, $B, 1, 1)\n    M = M0\n"
		. "    scan M over at in A, vt in V, wt in W:\n"
		. "        r = reshape(reshape(at, $B, 1, $S) \@ M, $B, $dv)\n        emit rs = r\n"
		. "        M = M + reshape(at, $B, $S, 1) * wt * reshape(vt - r, $B, 1, $dv)\n";
	die "dense has no stream mode\n" if $mode eq 'stream';
} else {
	$decl .= "param boff : f32[$B, 1] = [" . join(", ", map { "[" . $_ * $S . "]" } 0 .. $B - 1) . "]\n"
		. "param pidx : f32[1, 3] = [[1, 0, 0]]\nparam prd : f32[1, 3] = [[0, 1, 0]]\nparam pwr : f32[1, 3] = [[0, 0, 1]]\n";
	$decl .= $mode eq 'stream' ? "state Mem : f32[" . ($B * $S) . ", $dv] = zeros\n" : "param M0 : f32[" . ($B * $S) . ", $dv] = zeros\n";
	$body .= "    x = take(Ek, kid)\n"
		. "    s1 = (rmsnorm(x \@ Wk1) \@ transpose(rmsnorm(K1))) * temp\n"
		. "    s2 = (rmsnorm(x \@ Wk2) \@ transpose(rmsnorm(K2))) * temp\n"
		. "    i1 = topk(s1, 8)\n    i2 = topk(s2, 8)\n"
		. "    cs = reshape(reshape(take_along(s1, i1), $TB, 8, 1) + reshape(take_along(s2, i2), $TB, 1, 8), $TB, 64)\n"
		. "    ci = reshape(reshape(i1, $TB, 8, 1) * $n + reshape(i2, $TB, 1, 8), $TB, 64)\n"
		. "    j0 = topk(cs, 8)\n"
		. "    a4 = reshape(softmax(take_along(cs, j0)), $T, $B, 8, 1)\n"
		. "    P = reshape(reshape(take_along(ci, j0), $T, $B, 8) + boff, $T, $B, 8, 1) * pidx + a4 * prd + a4 * reshape(gate, $T, $B, 1, 1) * pwr\n"
		. "    V = reshape(take(Ev, vid), $T, $B, $dv)\n";
	if ($mode eq 'stream') {
		$body .= "    m0 = Mem * (1 - reset)\n    rs = memory_scan(m0, P, V)\n    update Mem = memory_final(m0, P, V)\n";
	} else {
		$body .= "    rs = memory_scan(M0, P, V)\n";
	}
}
$body .= "    logits = reshape(rs, $TB, $dv) \@ Wo\n    tv = take(I_V, tid)\n"
	. "    hit = (tv * take(I_V, reshape(topk(logits, 1), $TB))) \@ onesV\n";
my $args = "kid: f32[$TB], vid: f32[$TB], tid: f32[$TB], q: f32[$TB]" . ($mode eq 'stream' ? ", reset: f32" : "");
open my $f, '>', "$dir/$mode.tg" or die;
if ($mode eq 'train') {
	print $f "model long_${kind}\n$decl\ndef forward($args) -> f32:\n$body"
		. "    loss = -sum(reshape(q, $TB, 1) * tv * log(softmax(logits) + 0.000000001)) / (sum(q) + 0.000001)\n"
		. "    train loss with adamw(lr=$lr, wd=0, clip=1)\n    return sum(q * hit) / (sum(q) + 0.000001)\n";
} else {
	print $f "model long_${kind}_$mode\n${decl}param h2 : f32[2] = [1, 0]\nparam c2 : f32[2] = [0, 1]\n\n"
		. "def forward($args) -> f32[2]:\n$body    return sum(q * hit) * h2 + sum(q) * c2\n";
}
close $f;
open my $e, '>:raw', "$dir/eye_v.bin" or die;
print $e pack('f<*', map { my $r = $_; map { $_ == $r ? 1 : 0 } 0 .. $NV - 1 } 0 .. $NV - 1);
