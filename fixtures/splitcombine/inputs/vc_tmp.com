# THIS FILE DOES EVERYTHING FOR COMBINING VOLUMES
#
####CreatedVersion####5.2.17
#
# CHANGE THIS VALUE TO SET ReductionFactor NONZERO IN COMBINEFFT
#
$set combinefft_reduce = 0
#
# CHANGE THIS VALUE TO SET LowFromBothRadius NONZERO IN COMBINEFFT
#
$set combinefft_lowboth = 0
#
$setenv IMOD_BRIEF_HEADER 1
$setenv PIP_PRINT_ENTRIES 0
#
# TO RESTART AT A PARTICULAR PIECE, CHANGE 0 IN THE FOLLOWING "goto dopiece0"
# TO THE DESIRED PIECE NUMBER
#
$goto dopiece0
#
$dopiece0:
#
$if (-e ./savework) ./savework
#
$echo "STATUS: RUNNING DENSMATCH TO MATCH DENSITIES"
$echo
#
# Scale the densities in the match file to match the first tomogram.  Inputs:
#     File being matched (first tomogram)
#     File to be scaled (the second tomogram)
#     BLANK LINE TO HAVE SCALED VALUES PUT BACK IN THE SAME FILE
#
$densmatch
ga.rec
/tmp/combine.3505246/gb.mat

#
#
# purge some previous versions if necessary: these are the huge files
#
$set nonomatch
$\rm -f /tmp/combine.3505246/*.mat~ /tmp/combine.3505246/mat.fft* /tmp/combine.3505246/rec.fft* /tmp/combine.3505246/sum.fft* sum.rec* /tmp/combine.3505246/sum[1-9]*.rec* /tmp/combine.3505246/sum[1-9]*_rec.*
#
$echo STATUS: RUNNING FILLTOMO TO FILL IN GRAY AREAS IN THE .MAT FILE
$echo 
#
$filltomo -StandardInput
FillTomogram	/tmp/combine.3505246/gb.mat
MatchedToTomogram	ga.rec
SourceTomogram	gb.rec
InverseTransformFile	inverse.xf
#
$dopiece1:
$echo STATUS: EXTRACTING AND COMBINING PIECE  1  of 1
$echo
#
$combinefft -StandardInput
AInputFFT	ga.rec
BInputFFT	/tmp/combine.3505246/gb.mat
OutputFFT	/tmp/combine.3505246/sum1.rec
XMinAndMax	0,63
YMinAndMax	0,19
ZMinAndMax	0,63
TaperPadsInXYZ	8,4,8
InverseTransformFile	inverse.xf
ATiltFile	ga.tlt
BTiltFile	gb.tlt
ReductionFraction	$combinefft_reduce
LowFromBothRadius	$combinefft_lowboth
#
$echo STATUS: REASSEMBLING PIECES
$echo
#
$assemblevol -StandardInput
OutputFile sum.rec
StartEndToExtractInX 8,71
StartEndToExtractInY 5,24
StartEndToExtractInZ 8,71
InputFile /tmp/combine.3505246/sum1.rec
#
$echo 
$echo STATUS: RUNNING FILLTOMO ON FINAL VOLUME
$echo 
#
$\rm -f /tmp/combine.3505246/sum[1-9]*.rec /tmp/combine.3505246/sum[1-9]*_rec.*
$filltomo -StandardInput
FillTomogram	sum.rec
MatchedToTomogram	ga.rec
SourceTomogram	gb.rec
InverseTransformFile	inverse.xf
$\rm -r /tmp/combine.3505246
$if (-e ./savework) ./savework
