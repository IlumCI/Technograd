#!/usr/bin/perl
# Multi-query associative recall (MQAR; Arora et al., arXiv:2312.04927) as
# string pairs: P distinct keys (letters) each bound to a value (digit), then
# Q of those keys; the output is their values in order.
#   in:  "q3a7m1x0:mxq"   out: "103"
# Usage: gen.pl PAIRS QUERIES ROWS SEED   (jsonl on stdout)
use strict;
my ($P, $Q, $n, $seed) = @ARGV;
die "at most 26 pairs\n" if $P > 26;
srand($seed);
my @L = ('a' .. 'z');
for (1 .. $n) {
	my @k = @L;
	for (my $i = $#k; $i > 0; $i--) { my $j = int rand($i + 1); @k[$i, $j] = @k[$j, $i] }
	@k = @k[0 .. $P - 1];
	my %v = map { $_ => int rand 10 } @k;
	my @q = map { $k[int rand $P] } 1 .. $Q;
	printf "{\"in\": \"%s:%s\", \"out\": \"%s\"}\n", join("", map { "$_$v{$_}" } @k), join("", @q), join("", map { $v{$_} } @q);
}
