# THIS FILE RUNS CORRSEARCH3D
#
####CreatedVersion#### 4.9.0
#
# REMEMBER THAT Y IS THICKNESS HERE
#
$corrsearch3d -StandardInput
ReferenceFile   ga.rec
FileToAlign     gb.mat
OutputFile      patch.out
RegionModel     patch_region.mod
PatchSizeXYZ    56,20,56
NumberOfPatchesXYZ      3,2,5
XMinAndMax      36,988
YMinAndMax      10,72
ZMinAndMax      36,988
BSourceOrSizeXYZ        gb.rec
BSourceTransform        solve.xf
BSourceBorderXLoHi      36,36
BSourceBorderYZLoHi     36,36
KernelSigma     1.45
FlipYZMessages
InvertYLimits
#
# Make a patch vector model
$patch2imod -n "Values are correlation coefficients" patch.out \
patch_vector_ccc.mod
$if (-e savework-file) savework-file
