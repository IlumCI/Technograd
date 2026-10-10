#!/usr/bin/perl
# Finite-difference check of `grad`.
#
# For each case, two models are generated from one scalar objective f(x):
#   F: forward(x) -> f(x)          G: forward(x) -> grad(f(x), x)
# The analytic gradient from G is compared with central differences of F,
# computed in a single `tgc batch` call over all x +/- h*e_i rows.
# Usage: gradcheck.pl TGC TMPDIR      exit status = number of failures
use strict;
use warnings;

my ($tgc, $tmp) = @ARGV;
my @cases = (
	# name, input type, extra declarations, objective in x
	[ 'arith+broadcast', 'f32[4]', '', 'sum((x + 2) * x / (x * x + 1) - 3 * x)' ],
	[ 'max/min',         'f32[4]', '', 'sum(max(x, 0.3) + min(x, -0.2) * x)' ],
	[ 'activations',     'f32[4]', '', 'sum(tanh(x) + relu(x) * x + sigmoid(-x) + exp(0.5 * x))' ],
	[ 'sqrt/log/softplus','f32[4]', '', 'sum(sqrt(x * x + 1) + log(x * x + 2) + softplus(x))' ],
	[ 'silu/gelu',       'f32[4]', '', 'sum(silu(x) + gelu(x))' ],
	[ 'matvec x right',  'f32[4]', 'param W : f32[3, 4] = rand(3, 0.7)', 'sum(tanh(W @ x))' ],
	[ 'matvec x left',   'f32[3, 4]', 'param v : f32[4] = rand(4, 0.7)', 'sum(tanh(x @ v))' ],
	[ 'vecmat',          'f32[4]', 'param B : f32[4, 3] = rand(5, 0.7)', 'sum((x @ B) * (x @ B))' ],
	[ 'matmat left',     'f32[2, 3]', 'param C : f32[3, 2] = rand(6, 0.7)', 'sum(tanh(x @ C))' ],
	[ 'matmat right',    'f32[3, 2]', 'param P : f32[2, 3] = rand(7, 0.7)', 'sum(tanh(P @ x))' ],
	[ 'dot',             'f32[4]', 'param c : f32[4] = [1, -2, 3, 0.5]', 'dot(x, x) + dot(x, c)' ],
	[ 'transpose',       'f32[2, 3]', 'param p : f32[2] = [0.5, -1]', 'sum(tanh(transpose(x) @ p))' ],
	[ 'outer',           'f32[4]', 'param Q : f32[4, 4] = rand(8, 1)', 'sum(outer(x, x) * Q)' ],
	[ 'softmax vec',     'f32[4]', 'param c : f32[4] = [1, 2, 3, 4]', 'sum(softmax(x) * c)' ],
	[ 'softmax rows',    'f32[2, 3]', 'param D : f32[2, 3] = [[1, 2, 3], [-1, 0, 2]]', 'sum(softmax(x) * D)' ],
	[ 'rmsnorm vec',     'f32[4]', 'param c : f32[4] = [1, 2, 3, 4]', 'sum(rmsnorm(x) * c)' ],
	[ 'rmsnorm rows',    'f32[2, 3]', 'param D : f32[2, 3] = [[1, 2, 3], [-1, 0, 2]]', 'sum(rmsnorm(x) * D)' ],
	[ 'softmax rank 3',  'f32[2, 2, 3]', 'param D : f32[2, 2, 3] = rand(39, 1)', 'sum(softmax(x) * D)' ],
	[ 'rmsnorm rank 3',  'f32[2, 2, 3]', 'param D : f32[2, 2, 3] = rand(40, 1)', 'sum(rmsnorm(x) * D)' ],
	[ 'sum/mean',        'f32[4]', '', 'mean(x * x) * sum(x)' ],
	[ 'scalar input',    'f32', '', 'tanh(x * x) + x' ],
	[ 'reuse via def',   'f32[4]', "def sq(z: f32[4]) -> f32[4]:\n    return z * z", 'sum(sq(x) * x + sq(tanh(x)))' ],
	[ 'row broadcast: into vector', 'f32[4]', 'param M : f32[3, 4] = rand(10, 1)', 'sum(tanh(M * x + x) * M)' ],
	[ 'row broadcast: into matrix', 'f32[3, 4]', 'param v : f32[4] = [0.5, -1, 2, 0.3]', 'sum(tanh(x * v - v))' ],
	[ 'batched mlp + xent', 'f32[4, 3]', "param w : f32[3, 5] = rand(11, 0.8)\nparam b : f32[5] = rand(12, 0.5)\nparam t : f32[4, 5] = [[1,0,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,0,0,1]]", 'xent(gelu(x @ w + b), t)' ],
	[ 'spmm: weights', 'f32[6, 3]', 'param S : f32[2, 4, 2] = [[[0, 1], [2, 0.5], [5, -1], [0, 0]], [[1, 1], [1, 0.7], [9, 3], [4, 2]]]', 'sum(tanh(spmm(S, x)) * spmm(S, x))' ],
	[ 'spmm: values', 'f32[2, 4, 2]', "param I : f32[2, 4, 2] = [[[0, 0], [2, 0], [5, 0], [3, 0]], [[1, 0], [1, 0], [4, 0], [0, 0]]]\nparam M : f32[2, 4, 2] = [[[0, 1], [0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1], [0, 1]]]\nparam W : f32[6, 3] = rand(13, 0.8)", 'sum(tanh(spmm(x * M + I, W)))' ],
	[ 'spmm: second order (scatter)', 'f32[6, 3]', "param S : f32[2, 4, 2] = [[[0, 1], [2, 0.5], [5, -1], [0, 0]], [[1, 1], [1, 0.7], [9, 3], [4, 2]]]\nparam C : f32[6, 3] = rand(14, 1)", 'sum(grad(sum(tanh(spmm(S, x))), x) * C)' ],
	[ 'spmm: second order (values)', 'f32[6, 3]', "param S : f32[2, 4, 2] = [[[0, 1], [2, 0.5], [5, -1], [0, 0]], [[1, 1], [1, 0.7], [9, 3], [4, 2]]]\nparam P : f32[2, 4, 2] = rand(15, 1)", 'sum(grad(sum(tanh(spmm(S, x))), S) * P)' ],
	[ 'take (rows, repeated)', 'f32[5, 3]', "param r : f32[4] = [3, 0, -1, 3]\nparam C : f32[4, 3] = rand(16, 1)", 'sum(tanh(take(x, r)) * C)' ],
	[ 'take: second order', 'f32[5, 3]', "param r : f32[4] = [3, 0, -1, 3]\nparam C : f32[5, 3] = rand(17, 1)", 'sum(grad(sum(tanh(take(x, r))), x) * C)' ],
	[ 'scan: rnn, wrt sequence', 'f32[4, 2]', "param W : f32[3, 3] = rand(21, 0.6)\nparam U : f32[3, 2] = rand(22, 0.8)\nparam h0 : f32[3] = [0.1, -0.2, 0.3]\nparam c : f32[3] = [1, -2, 0.5]",
	  "SETUP h = h0\n    scan h over xt in x:\n        h = tanh(W @ h + U @ xt)\n        emit hs = h\n    OBJ sum(h * c) + sum(hs * hs)" ],
	[ 'scan: rnn, wrt weights', 'f32[3, 3]', "param S : f32[5, 2] = [[1, 0], [0.5, -1], [-0.3, 0.8], [0.2, 0.2], [-1, 0.4]]\nparam U : f32[3, 2] = rand(22, 0.8)\nparam h0 : f32[3] = [0.1, -0.2, 0.3]",
	  "SETUP h = h0\n    scan h over st in S:\n        h = tanh(x @ h + U @ st)\n        emit hs = h\n    OBJ sum(hs * hs) + sum(h)" ],
	[ 'scan: wrt initial carry', 'f32[3]', "param S : f32[5, 2] = [[1, 0], [0.5, -1], [-0.3, 0.8], [0.2, 0.2], [-1, 0.4]]\nparam W : f32[3, 3] = rand(21, 0.6)\nparam U : f32[3, 2] = rand(22, 0.8)",
	  "SETUP h = x\n    scan h over st in S:\n        h = tanh(W @ h + U @ st)\n    OBJ sum(h * h)" ],
	[ 'scan: two carries, swap, start-of-step emit', 'f32[4, 3]', "param W : f32[3, 3] = rand(23, 0.6)\nparam a0 : f32[3] = [0.3, 0.1, -0.4]\nparam b0 : f32[3] = [-0.2, 0.5, 0.2]",
	  "SETUP p = a0\n    q = b0\n    scan p, q over xt in x:\n        emit ps = p\n        t = p\n        p = tanh(W @ q + xt)\n        q = t * 0.5\n        emit qs = q\n    OBJ sum(ps * qs) + sum(p * q)" ],
	[ 'scan: selective SSM (Mamba-style)', 'f32[6, 2]', "param Wd : f32[2, 3] = rand(24, 0.7)\nparam bd : f32[3] = [0.1, -0.3, 0.2]\nparam A : f32[3] = [0.5, 1, 2]\nparam B : f32[3, 2] = rand(25, 0.8)\nparam C : f32[3, 2] = rand(26, 0.8)\nparam h0 : f32[3] = zeros",
	  "SETUP h = h0\n    scan h over xt in x:\n        dt = softplus(xt @ Wd + bd)\n        h = exp(-(dt * A)) * h + dt * (B @ xt)\n        emit ys = dot(h, C @ xt)\n    OBJ sum(ys * ys)" ],
	[ 'scan: second order', 'f32[3, 3]', "param S : f32[4, 2] = [[1, 0], [0.5, -1], [-0.3, 0.8], [0.2, 0.2]]\nparam U : f32[3, 2] = rand(22, 0.8)\nparam h0 : f32[3] = [0.1, -0.2, 0.3]\nparam P : f32[3, 3] = rand(27, 1)",
	  "SETUP h = h0\n    scan h over st in S:\n        h = tanh(x @ h + U @ st)\n    OBJ sum(grad(sum(h * h), x) * P)" ],
	[ 'broadcast: column and outer', 'f32[3, 1]', "param y : f32[1, 4] = [0.5, -1, 2, 0.3]\nparam M : f32[3, 4] = rand(29, 1)", 'sum(tanh(x * y + x) * M)' ],
	[ 'broadcast: rank 3, both sides', 'f32[2, 1, 3]', "param y : f32[4, 1] = rand(30, 1)\nparam M : f32[2, 4, 3] = rand(31, 1)", 'sum(tanh(x * y - y / (x * x + 2)) * M)' ],
	[ 'sum_to', 'f32[3, 4]', "param C : f32[1, 4] = rand(32, 1)", 'sum(tanh(sum_to(x * x, C)) * C)' ],
	[ 'batched matmul, wrt left', 'f32[2, 3, 4]', "param B : f32[2, 4, 2] = rand(33, 0.7)", 'sum(tanh(x @ B))' ],
	[ 'batched matmul, wrt right', 'f32[2, 4, 2]', "param A : f32[2, 3, 4] = rand(34, 0.7)", 'sum(tanh(A @ x) * (A @ x))' ],
	[ 'batched transpose', 'f32[2, 3, 4]', "param B : f32[2, 3, 2] = rand(35, 0.7)", 'sum(tanh(transpose(x) @ B))' ],
	[ 'rank 3 @ shared matrix', 'f32[2, 3, 4]', "param W : f32[4, 2] = rand(36, 0.7)", 'sum(tanh(x @ W))' ],
	[ 'shared matrix, wrt matrix', 'f32[4, 2]', "param A : f32[2, 3, 4] = rand(37, 0.7)", 'sum(tanh(A @ x))' ],
	[ 'rank 3 @ shared vector', 'f32[4]', "param A : f32[2, 3, 4] = rand(38, 0.7)", 'sum(tanh(A @ x) * (A @ x))' ],
	[ 'sin/cos/reshape', 'f32[2, 6]', "param M : f32[3, 4] = rand(28, 1)", 'sum(reshape(sin(x) * cos(2 * x), 3, 4) * M)' ],
	[ 'think: fixed budget (BPTT)', 'f32[3]', "param W : f32[3, 3] = rand(41, 0.6)\nparam c : f32[3] = [1, -0.5, 2]",
	  "SETUP h = x * 0.5\n    think h for 6:\n        h = tanh(W @ h + x)\n    OBJ sum(h * c)" ],
	[ 'nested: scan in scan', 'f32[3, 2]', "param W : f32[2, 2] = rand(42, 0.7)\nparam S : f32[4, 2] = [[1, 0], [0.5, -1], [-0.3, 0.8], [0.2, 0.2]]\nparam h0 : f32[2] = [0.1, 0.3]",
	  "SETUP h = h0\n    scan h over xt in x:\n        r = h\n        scan r over st in S:\n            r = tanh(W @ r + st + xt)\n        h = r\n        emit hs = h\n    OBJ sum(hs * hs)" ],
	[ 'nested: think until in scan', 'f32[4, 2]', "param W : f32[2, 2] = rand(43, 0.4)\nparam h0 : f32[2] = [0.2, -0.1]",
	  "SETUP h = h0\n    scan h over xt in x:\n        q = h\n        think q for 200 until 0.0000001:\n            q = tanh(W @ q + xt)\n        h = 0.5 * h + q\n    OBJ sum(h * h)" ],
	[ 'nested: fixed think in scan', 'f32[4, 2]', "param W : f32[2, 2] = rand(44, 0.6)\nparam h0 : f32[2] = [0.2, -0.1]",
	  "SETUP h = h0\n    scan h over xt in x:\n        q = h\n        think q for 3:\n            q = tanh(W @ q + xt)\n        h = q\n        emit hs = h\n    OBJ sum(hs * hs)" ],
	[ 'nested: scan in think until', 'f32[3]', "param W : f32[3, 3] = rand(45, 0.3)\nparam S : f32[4, 3] = rand(46, 1)",
	  "SETUP h = x * 0\n    think h for 300 until 0.0000001:\n        r = h\n        scan r over st in S:\n            r = 0.5 * tanh(W @ r + st * 0.3) + 0.2 * x\n        h = r * 0.5 + 0.1 * h\n    OBJ sum(h * h) + sum(h)" ],
	[ 'think: fixed budget, second order', 'f32[3, 3]', "param h0 : f32[3] = [0.3, -0.2, 0.5]\nparam P : f32[3, 3] = rand(47, 1)",
	  "SETUP h = h0\n    think h for 4:\n        h = tanh(x @ h + 0.1)\n    OBJ sum(grad(sum(h * h), x) * P)" ],
	[ 'attention + rope (window 3)', 'f32[5, 4]', "param Wq : f32[4, 4] = rand(51, 0.7)\nparam Wk : f32[4, 4] = rand(52, 0.7)\nparam P : f32[5] = [0, 1, 2, 3, 4]\nparam C : f32[5, 4] = rand(53, 1)",
	  'sum(attention(rope(x @ Wq, P), rope(x @ Wk, P), x, 3) * C)' ],
	[ 'rope at a scalar position', 'f32[4]', "param c : f32[4] = [1, -1, 0.5, 2]", 'sum(rope(x * x, 3) * c)' ],
	[ 'think: sqrt fixed point', 'f32[3]', '',
	  "SETUP h = x * 0.5 + 0.5\n    think h for 200 until 0.0000001:\n        h = 0.5 * (h + x / h)\n    OBJ sum(h)" ],
	[ 'think: contraction with input', 'f32[4]', "param W : f32[4, 4] = rand(9, 0.3)\nparam c : f32[4] = [1, -1, 2, 0.5]",
	  "SETUP e = x\n    h = tanh(e)\n    think h for 400 until 0.00000001:\n        h = tanh(W @ h + e)\n    OBJ sum(h * c)" ],
);

my $fails = 0;
srand(1234);
for my $c (@cases) {
	my ($name, $ty, $decl, $obj) = @$c;
	my @dims = $ty =~ /\[(.*)\]/ ? split(/\s*,\s*/, $1) : ();
	my $n = 1;
	$n *= $_ for @dims;
	my ($setup, $f) = ('', $obj);
	if ($obj =~ /^SETUP (.*)\n    OBJ (.*)$/s) { ($setup, $f) = ("    $1\n", $2); }
	my $rt = $ty;
	write_file("$tmp/F.tg", "model F\n$decl\ndef forward(x: $ty) -> f32:\n${setup}    return $f\n");
	write_file("$tmp/G.tg", "model G\n$decl\ndef forward(x: $ty) -> $rt:\n${setup}    return grad($f, x)\n");

	my @x = map { sprintf('%.4f', $name =~ /sqrt fixed/ ? 0.5 + 3 * rand() : 2 * rand() - 1) } 1 .. $n;
	my $h = 1e-2;
	my @rows;
	for my $i (0 .. $n - 1) {
		for my $s (1, -1) {
			my @y = @x;
			$y[$i] += $s * $h;
			push @rows, join(',', @y);
		}
	}
	write_file("$tmp/fd.csv", join("\n", @rows) . "\n");
	my @fv = map { (split /,/)[0] } split /\n/, `$tgc batch $tmp/F.tg $tmp/fd.csv 2>&1`;
	my $gout = `$tgc run $tmp/G.tg @{[join(',', @x)]} 2>&1`;
	my @an = split ' ', (split /\n/, $gout)[0];
	if (@an != $n || @fv != 2 * $n) {
		print "FAIL gradcheck $name: tool output\n$gout\n";
		$fails++;
		next;
	}
	my $worst = 0;
	for my $i (0 .. $n - 1) {
		my $fd = ($fv[2 * $i] - $fv[2 * $i + 1]) / (2 * $h);
		my $err = abs($fd - $an[$i]) / (1 + abs($an[$i]));
		$worst = $err if $err > $worst;
	}
	my $ok = $worst < 5e-3;
	printf "%s gradcheck %-30s max rel err %.2e\n", $ok ? 'ok  ' : 'FAIL', $name, $worst;
	$fails++ unless $ok;
}
exit $fails;

sub write_file {
	my ($p, $s) = @_;
	open my $fh, '>', $p or die "$p: $!";
	print $fh $s;
	close $fh;
}
