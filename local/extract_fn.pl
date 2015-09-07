#!/usr/bin/perl

$srcfn = shift;

$ext = ".SPH";
$sep = "/";
open L, "<$srcfn" || die;
while($ln=<L>)
{
	chomp($ln);
	@ln_arr = split($sep,$ln);
	$filename = $ln_arr[$#ln_arr];
	$filename =~ s/.SPH//;
	$filename = lc "$filename";
	print "$filename\n";	
}
close(L);
