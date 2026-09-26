# command file to run ctfplotter
#
####CreatedVersion####5.2.17
# 
#
$setenv IMOD_OUTPUT_FORMAT MRC
$ctfplotter -StandardInput
#
# Your entry should be something like:
#
InputStack  b.st
#
# The tilt angle file - .rawtlt could be used instead if .tlt not available yet
AngleFile   b.tlt
DefocusFile b.defocus
#
# How many degrees the tilt axis deviates from vertical (Y axis) (CCW positive)
AxisAngle	-85.3
#
# Image pixel size in nanometers
PixelSize	1.0
#
# Expected defocus at the tilt axis in nanometers (underfocus is positive)
ExpectedDefocus 5000.0 
#
# Starting and ending tilt angles for initial analysis
AngleRange  -3.6 3.6
#
# Microscope voltage in kV
Voltage      200
#
# Microscope spherical aberration in millimeters
SphericalAberration 2
#
# Fraction of amplitude contrast
AmplitudeContrast 0.07
#
TileSize     256
LeftDefTol  2000.0
RightDefTol 2000.0 
$if (-e ./savework) ./savework
