# Command file to run mtffilter
$mtffilter -StandardInput
InputFile	g.ali
OutputFile	g_filt.ali
PixelSize	1.0
DoseWeightingFile	b.dose
