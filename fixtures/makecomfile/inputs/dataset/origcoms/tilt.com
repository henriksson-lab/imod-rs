# Command file to run Tilt
#
####CreatedVersion####5.2.17
# 
# RADIAL specifies the frequency at which the Gaussian low pass filter begins
#   followed by the standard deviation of the Gaussian roll-off
#
# LOG takes the logarithm of tilt data after adding the given value
#
$setenv IMOD_OUTPUT_FORMAT MRC
$tilt -StandardInput
InputProjections b_ali.mrc
OutputFile b_rec.mrc
IMAGEBINNED 1
TILTFILE b.tlt
XTILTFILE b.xtilt
THICKNESS 100
RADIAL .35 .035
FalloffIsTrueSigma 1
XAXISTILT 0.
LOG 0
SCALE 0 1000
PERPENDICULAR
MODE 1
FULLIMAGE 96 128
SUBSETSTART 0 0
AdjustOrigin 1
$if (-e ./savework) ./savework
