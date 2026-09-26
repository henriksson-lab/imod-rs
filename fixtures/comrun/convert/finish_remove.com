# THIS COMMAND FILE REASSEMBLES THE PIECES
#
$setenv IMOD_OUTPUT_FORMAT tif
$assemblevol -StandardInput
OutputFile vol_x.tif
StartEndToExtractInX 14,525
StartEndToExtractInY 14,525
StartEndToExtractInZ 9,43
InputFile filt-001.out
$b3dremove -g filt-[0-9][0-9][0-9]*.com* filt-[0-9][0-9][0-9]*.log* filt-[0-9][0-9][0-9]*.out*
