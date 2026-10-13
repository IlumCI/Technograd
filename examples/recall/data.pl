#!/usr/bin/perl
# Rows for models.pl: each row is a batch of 16 sequences of P writes (distinct
# keys of NK, default 64, each with a random value of 64) followed by Q queries of written
# keys, time-major: kid (T*16), vid (T*16), w (T*16), T = P + Q.
# Usage: data.pl P Q ROWS SEED [NK]
use strict;
my ($P, $Q, $n, $seed, $NK) = @ARGV;
$NK ||= 64;
my $B = 16;
srand($seed);
for (1 .. $n) {
	my (@k, @v, @w);
	for my $b (0 .. $B - 1) {
		my @keys = 0 .. $NK - 1;
		for (my $i = $#keys; $i > 0; $i--) { my $j = int rand($i + 1); @keys[$i, $j] = @keys[$j, $i] }
		my @ks = @keys[0 .. $P - 1];
		my %val = map { $_ => int rand 64 } @ks;
		my @qs = map { $ks[int rand $P] } 1 .. $Q;
		my @sk = (@ks, @qs);
		for my $t (0 .. $P + $Q - 1) {
			my $i = $t * $B + $b;
			$k[$i] = $sk[$t]; $v[$i] = $val{$sk[$t]}; $w[$i] = $t < $P ? 1 : 0;
		}
	}
	print join(",", @k, @v, @w), "\n";
}
