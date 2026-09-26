# THIS IS A COMMAND FILE TO RUN TILTALIGN
#
####CreatedVersion####5.2.17
#
# To exclude views, add a line "ExcludeList view_list" with the list of views
#
# To specify sets of views to be grouped separately in automapping, add a line
# "SeparateGroup view_list" with the list of views, one line per group
#
$setenv IMOD_OUTPUT_FORMAT MRC
$tiltalign -StandardInput
ModelFile	b.fid
ImageFile	b_preali.mrc
#ImageSizeXandY	128,96
ImagesAreBinned	1
OutputModelFile	b.3dmod
OutputResidualFile	b.resid
OutputFidXYZFile	bfid.xyz
OutputTiltFile	b.tlt
OutputXAxisTiltFile	b.xtilt
OutputTransformFile	b.tltxf
OutputFilledInModel     b_nogaps.fid
RotationAngle	-85.3
UnbinnedPixelSize	1.0
CreatedDayStamp	2460
TiltFile	b.rawtlt
#
# ADD a recommended tilt angle change to the existing AngleOffset value
#
AngleOffset	0.
RotOption	1
RotDefaultGrouping	5
#
# TiltOption 0 fixes tilts, 2 solves for all tilt angles; change to 5 to solve
# for fewer tilts by grouping views by the amount in TiltDefaultGrouping
#
TiltOption	5
TiltDefaultGrouping	5
MagReferenceView	1
MagOption	1
MagDefaultGrouping	4
#
# To solve for distortion, change both XStretchOption and SkewOption to 3;
# to solve for skew only leave XStretchOption at 0
#
XStretchOption	0
SkewOption	0
XStretchDefaultGrouping	7
SkewDefaultGrouping	11
BeamTiltOption	0
#
# To solve for X axis tilt between two halves of a dataset, set XTiltOption to 4
#
XTiltOption	0
XTiltDefaultGrouping	2000
# 
# Criterion # of S.D's above mean residual to report (- for local mean)
#
ResidualReportCriterion	3.0
SurfacesToAnalyze	2
MetroFactor	.25
MaximumCycles	1000
KFactorScaling	1.
NoSeparateTiltGroups	1
CrossValidate   1
#
# ADD a recommended amount to shift up to the existing AxisZShift value
#
AxisZShift	0.
ShiftZFromOriginal      1
#
# Set to 1 to do local alignments
#
LocalAlignments	0
OutputLocalFile	blocal.xf
#
# Target size of local patches to solve for in X and Y
#
TargetPatchSizeXandY	700,700
MinSizeOrOverlapXandY	0.5,0.5
#
# Minimum fiducials total and on one surface if two surfaces
#
MinFidsTotalAndEachSurface	10,4
FixXYZCoordinates	0
LocalOutputOptions	1,0,1
LocalRotOption	3
LocalRotDefaultGrouping	6
LocalTiltOption	5
LocalTiltDefaultGrouping	6
LocalMagReferenceView	1
LocalMagOption	3
LocalMagDefaultGrouping	7
LocalXStretchOption	0
LocalXStretchDefaultGrouping	7
LocalSkewOption	0
LocalSkewDefaultGrouping	11
#
# COMBINE TILT TRANSFORMS WITH PREALIGNMENT TRANSFORMS
#
$xfproduct -StandardInput
InputFile1 b.prexg
InputFile2 b.tltxf
OutputFile b_fid.xf
$b3dcopy -p "b_fid.xf" "b.xf"
$b3dcopy -p "b.tlt" "b_fid.tlt"
#
# CONVERT RESIDUAL FILE TO MODEL
#
$if (-e "b.resid") patch2imod -s 10 "b.resid" "b.resmod"
$if (-e ./savework) ./savework
