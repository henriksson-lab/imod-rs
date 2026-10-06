# Command file to run mtffilter
$mtffilter -StandardInput
InputFile	b_ali.mrc
OutputFile	b_filt.mrc
PixelSize	1.0
DoseWeightingFile	b.dose
