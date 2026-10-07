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
	[ 'sum/mean',        'f32[4]', '', 'mean(x * x) * sum(x)' ],
	[ 'scalar input',    'f32', '', 'tanh(x * x) + x' ],
	[ 'reuse via def',   'f32[4]', "def sq(z: f32[4]) -> f32[4]:\n    return z * z", 'sum(sq(x) * x + sq(tanh(x)))' ],
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
