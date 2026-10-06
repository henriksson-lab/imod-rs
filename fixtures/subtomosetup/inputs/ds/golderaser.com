# Command file to erase gold
$ccderaser -StandardInput
InputFile	b_ali.mrc
OutputFile	b_erase.ali
ModelFile	b_erase.fid
CircleObjects	/
BetterRadius	4.5
MaxPixelsInDiffPatch	0
PolynomialOrder	0
ExcludeAdjacent
