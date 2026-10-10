#!/bin/sh
# Build the military-science + purple-teaming corpus and a labeled task.
# Usage: build.sh OUTDIR
#
#   OUTDIR/corpus.jsonl      unlabeled text for `tgc pretrain` ({"source", "text"} per line)
#   OUTDIR/procedures.jsonl  labeled task for `tgc train` ({"text", "tactic"}): real
#                            ATT&CK procedure examples, kept out of the corpus
#
# Sources (all public):
#   Project Gutenberg military classics (public domain): Sun Tzu, Clausewitz,
#     Jomini, du Picq, Halleck, Mahan, Napoleon's maxims, Lectures on Land
#     Warfare, Lippitt, Sound Military Decision (Naval War College, 1942), ...
#   MITRE ATT&CK Enterprise (STIX 2.1): one document per technique joining the
#     red side (what the adversary does) with the blue side (detection
#     strategies and their analytics, mitigations with technique-specific
#     guidance): the pairing a purple team works from.
#   MITRE D3FEND: defensive technique definitions.
set -e
out=${1:?usage: build.sh OUTDIR}
mkdir -p "$out/src"
get() { [ -s "$out/src/$1" ] || curl -sSf -m 600 --retry 6 --retry-all-errors --retry-delay 5 -o "$out/src/$1" "$2"; }
BOOKS="132 1946 44200 13549 13529 25911 25912 7294 16170 23473 28178 50750 20866"
for b in $BOOKS; do get "pg$b.txt" "https://www.gutenberg.org/cache/epub/$b/pg$b.txt"; done
get enterprise-attack.json https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json
get d3fend.json https://d3fend.mitre.org/ontologies/d3fend.json
perl -MJSON::PP -e '
use strict; use warnings;
my ($dir, @books) = @ARGV;
my $js = JSON::PP->new->canonical->utf8;
sub clean { my $t = shift // ""; $t =~ s/\(Citation:[^)]*\)//g; $t =~ s/\[([^\]]*)\]\([^)]*\)/$1/g;
	$t =~ s/<[^>]+>//g; $t =~ s/\s+/ /g; $t =~ s/^ | $//g; return $t; }
sub slurp { local $/; open my $f, "<", shift or die "$!"; my $s = <$f>; return $s; }
open my $c, ">", "$dir/corpus.jsonl" or die; open my $p, ">", "$dir/procedures.jsonl" or die;
my %n = (gutenberg => 0, attack => 0, d3fend => 0, procedures => 0);
for my $b (@books) { # strip the Gutenberg header and license, one row per paragraph
	my $t = slurp("$dir/src/pg$b.txt"); $t =~ s/\r//g;
	$t =~ s/\A.*?\*\*\* ?START OF[^\n]*\n//s; $t =~ s/\*\*\* ?END OF.*\z//s;
	for my $para (split /\n\s*\n/, $t) { my $x = clean($para); next if ($x =~ tr/ //) < 8;
		print $c $js->encode({ source => "gutenberg:$b", text => $x }), "\n"; $n{gutenberg}++; }
}
my $objs = $js->decode(slurp("$dir/src/enterprise-attack.json"))->{objects};
my %o; $o{$_->{id}} = $_ for @$objs;
my $live = sub { my $x = shift; !$x->{revoked} && !$x->{x_mitre_deprecated} };
my (%det, %mit);
for my $r (grep { $_->{type} eq "relationship" && $live->($_) } @$objs) {
	my $t = $o{$r->{target_ref}} or next; my $s = $o{$r->{source_ref}} or next;
	next unless $t->{type} eq "attack-pattern" && $live->($t) && $live->($s);
	if ($r->{relationship_type} eq "detects") { push @{ $det{$t->{id}} }, $s; }
	elsif ($r->{relationship_type} eq "mitigates") { push @{ $mit{$t->{id}} }, [ $s, clean($r->{description}) ]; }
	elsif ($r->{relationship_type} eq "uses" && $r->{description}) { # procedure examples: the labeled task
		my @k = @{ $t->{kill_chain_phases} || [] }; next unless @k == 1; # one tactic: an unambiguous label
		my $x = clean($r->{description}); next if ($x =~ tr/ //) < 4;
		print $p $js->encode({ text => $x, tactic => $k[0]{phase_name} }), "\n"; $n{procedures}++;
	}
}
for my $t (sort { $a->{name} cmp $b->{name} } grep { $_->{type} eq "attack-pattern" && $live->($_) } @$objs) {
	my @d = ("Technique: $t->{name}. " . clean($t->{description}));
	for my $s (@{ $det{$t->{id}} || [] }) {
		push @d, "Detection strategy: $s->{name}.";
		for my $ar (@{ $s->{x_mitre_analytic_refs} || [] }) { my $an = $o{$ar} or next; push @d, "Analytic: " . clean($an->{description}) if $an->{description}; }
	}
	for my $m (@{ $mit{$t->{id}} || [] }) { push @d, "Mitigation: $m->[0]{name}. " . ($m->[1] || clean($m->[0]{description})); }
	print $c $js->encode({ source => "attack", text => join(" ", @d) }), "\n"; $n{attack}++;
}
for my $m (grep { $_->{type} eq "course-of-action" && $live->($_) } @$objs) {
	print $c $js->encode({ source => "attack", text => "Mitigation: $m->{name}. " . clean($m->{description}) }), "\n"; $n{attack}++;
}
for my $g (@{ $js->decode(slurp("$dir/src/d3fend.json"))->{"\@graph"} }) {
	my $d = $g->{"d3f:definition"} or next; next if ref $d;
	my $l = $g->{"rdfs:label"} // ""; $l = ref $l eq "HASH" ? $l->{"\@value"} // "" : ref $l ? "" : $l;
	print $c $js->encode({ source => "d3fend", text => clean("$l. $d") }), "\n"; $n{d3fend}++;
}
close $c; close $p;
printf STDERR "corpus: %d Gutenberg paragraphs, %d ATT&CK documents, %d D3FEND definitions; %d labeled procedures\n", @n{qw(gutenberg attack d3fend procedures)};
' "$out" $BOOKS
