# Command file to run ctfphaseflip
#
####CreatedVersion####5.2.17
# 
$setenv IMOD_OUTPUT_FORMAT MRC
$ctfphaseflip -StandardInput
InputStack  b_ali.mrc
AngleFile   b.tlt
OutputFileName b_ctfcorr_ali.mrc
TransformFile   b.xf
#
# Defocus file from ctfplotter (see man page for format)
DefocusFile b.defocus
#
# Microscope voltage in kV
Voltage      200
#
# Microscope spherical aberration in millimeters
SphericalAberration 2
#
# Defocus tolerance in nanometers limiting the strip width
DefocusTol   50
#
# Image pixel size in nanometers of input images and unbinned stack
PixelSize	1.0
UnbinnedPixelSize	1.0
#
# Fraction of amplitude contrast
AmplitudeContrast 0.07
#
# The distance in pixels between two consecutive strips
InterpolationWidth 15
$if (-e ./savework) ./savework
XAxisTilt 5
ExpandedByFactor 2
InvertTiltAngles
