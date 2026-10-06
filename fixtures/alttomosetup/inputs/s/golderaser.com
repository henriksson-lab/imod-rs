# Command file to erase gold
$ccderaser -StandardInput
InputFile	g.ali
OutputFile	g_erase.ali
ModelFile	b_erase.fid
CircleObjects	/
BetterRadius	4.5
MaxPixelsInDiffPatch	0
PolynomialOrder	0
ExcludeAdjacent
