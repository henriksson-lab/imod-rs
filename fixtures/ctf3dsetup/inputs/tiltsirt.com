# Command file to run Tilt
#
####CreatedVersion#### 4.5.59
# 
# RADIAL specifies the frequency at which the Gaussian low pass filter begins
#   followed by the standard deviation of the Gaussian roll-off
#
# LOG takes the logarithm of tilt data after adding the given value
#
$tilt -StandardInput
InputProjections g.ali
OutputFile g.rec
IMAGEBINNED 1
TILTFILE g.tlt
XTILTFILE g.xtilt
THICKNESS 30
RADIAL .35 .035
FalloffIsTrueSigma 1
XAXISTILT 2.5
LOG 0
SCALE 0 500
PERPENDICULAR
MODE 1
FULLIMAGE 48 40
SUBSETSTART 0 0
AdjustOrigin 1
SIRTIterations 4
