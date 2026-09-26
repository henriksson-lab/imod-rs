$set tmpext = `hostname`.$$
$set tmpdir = /usr/tmp
$if ($?IMOD_DIR) then
$if (-e "$IMOD_DIR/bin/settmpdir") source "$IMOD_DIR/bin/settmpdir"
$endif
$set combinefft_reduce = 0
$set combinefft_lowboth = 0
$setenv IMOD_BRIEF_HEADER 1
$echo
#
$combinefft -StandardInput
AInputFFT	ga.rec
BInputFFT	gb.mat
OutputFFT	sum5.rec
XMinAndMax	0,516
YMinAndMax	0,199
ZMinAndMax	676,1023
TaperPadsInXYZ	8,4,8
InverseTransformFile	inverse.xf
ATiltFile	ga.tlt
BTiltFile	gb.tlt
ReductionFraction	$combinefft_reduce
LowFromBothRadius	$combinefft_lowboth
#
$dopiece6:
