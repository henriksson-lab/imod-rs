# THIS IS A COMMAND FILE TO RUN TILTXCORR AND DETERMINE CROSS-CORRELATION
# ALIGNMENT OF A TILT SERIES
#
#
# TO RUN TILTXCORR
#
####CreatedVersion####5.2.17
#
# Add BordersInXandY to use a centered region smaller than the default
# or XMinAndMax and YMinAndMax  to specify a non-centered region
#
$setenv IMOD_OUTPUT_FORMAT MRC
$tiltxcorr -StandardInput
InputFile	b.st
OutputFile	b.prexf
PixelSize	1.0
TiltFile	b.rawtlt
RotationAngle	-85.3
FilterSigma1	0.03
FilterRadius2	0.25
FilterSigma2	0.05
$if (-e ./savework) ./savework
