#!/usr/bin/perl
# Generate (plaintext, ciphertext) pairs for a fixed-key classical transform,
# as JSONL {"in","out"} on stdout. The key is fixed per transform, so the
# model learns the specific deterministic map -- the exact-string task a
# cryptography CTF challenge is. These are classical/educational ciphers and
# reversible encodings, not attacks on modern cryptography.
#   gen.pl TRANSFORM N [SEED]
# TRANSFORM: rot13 atbash reverse caesar5 vigenere xor sub base64 morse
use strict; use warnings;
my ($tf, $n, $seed) = @ARGV; $n //= 4000; $seed //= 1; srand($seed);
my @MORSE = split / /, ".- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. -- -. --- .--. --.- .-. ... - ..- ...- .-- -..- -.-- --..";
my @b64 = ('A'..'Z','a'..'z','0'..'9','+','/');
# a fixed substitution permutation, seeded deterministically (same every run)
my @perm = do { my @a = ('a'..'z'); srand(12345); for (my $i=$#a;$i>0;$i--){my $j=int rand($i+1);@a[$i,$j]=@a[$j,$i]} srand($seed); @a };
my %sub = map { ('a'..'z')[$_] => $perm[$_] } 0..25;
sub enc {
	my $s = shift;
	if ($tf eq 'rot13')   { (my $o=$s)=~tr/A-Za-z/N-ZA-Mn-za-m/; return $o }
	if ($tf eq 'atbash')  { (my $o=$s)=~tr/a-z/zyxwvutsrqponmlkjihgfedcba/; return $o }
	if ($tf eq 'reverse') { return scalar reverse $s }
	if ($tf eq 'caesar5') { (my $o=$s)=~tr/a-z/f-za-e/; return $o }
	if ($tf eq 'vigenere'){ my @k=split//, "key"; my $o=""; my $i=0;
		for my $c (split//,$s){ if($c=~/[a-z]/){ my $sh=ord($k[$i%@k])-97; $o.=chr(97+((ord($c)-97+$sh)%26)); $i++ } else {$o.=$c} } return $o }
	if ($tf eq 'xor')     { my $o=""; for my $c (split//,$s){ $o.=sprintf("%02x", ord($c)^0x2a) } return $o }
	if ($tf eq 'sub')     { (my $o=$s)=~s/([a-z])/$sub{$1}/g; return $o }
	if ($tf eq 'base64')  { my $o=""; my @b=map{ord}split//,$s;
		for(my $i=0;$i<@b;$i+=3){ my @g=@b[$i..$i+2]; my $pad=0;
			for(1..2){ if(!defined $g[$_]){$g[$_]=0;$pad++} }
			my $x=($g[0]<<16)|($g[1]<<8)|$g[2];
			$o.=$b64[($x>>18)&63].$b64[($x>>12)&63];
			$o.=$pad>=2?"=":$b64[($x>>6)&63]; $o.=$pad>=1?"=":$b64[$x&63]; } return $o }
	if ($tf eq 'morse')   { my @o; for my $c (split//,$s){ if($c=~/[a-z]/){push @o,$MORSE[ord($c)-97]} elsif($c eq ' '){push @o,'/'} } return join(" ",@o) }
	die "unknown transform '$tf'\n";
}
my @c = ('a'..'z', (' ') x 3);
for (1..$n) {
	my $len = 3 + int(rand 18);
	my $s = join("", map { $c[int rand @c] } 1..$len);
	$s =~ s/^ +//; $s =~ s/ +$//; $s =~ s/  +/ /g;
	next if $s eq "";
	my $o = enc($s);
	for ($s, $o) { s/([\\"])/\\$1/g }
	print "{\"in\":\"$s\",\"out\":\"$o\"}\n";
}
