# THIS IS A COMMAND FILE TO PRODUCE A PRE-ALIGNED STACK
# 
# The stack will be floated and converted to bytes under the assumption that
# you will go back to the raw stack to make the final aligned stack
#
$setenv IMOD_OUTPUT_FORMAT MRC
$xftoxg
0
b.prexf
b.prexg
$newstack -StandardInput
InputFile	b.st
OutputFile	b_preali.mrc
TransformFile	b.prexg
ModeToOutput	0
FloatDensities 2
BinByFactor	1
#DistortionField	.idf
ImagesAreBinned	1
#GradientFile	b.maggrad
$if (-e ./savework) ./savework
