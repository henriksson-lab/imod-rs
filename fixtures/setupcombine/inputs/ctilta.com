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
InputProjections ga.ali
OutputFile ga.rec
IMAGEBINNED 2
TILTFILE ga.tlt
XTILTFILE ga.xtilt
THICKNESS 100
RADIAL .35 .035
FalloffIsTrueSigma 1
XAXISTILT 1.5
LOG 0
SCALE 0 500
PERPENDICULAR
MODE 1
EXCLUDELIST
FULLIMAGE
SUBSETSTART 0 0
AdjustOrigin 1
SHIFT 0 1
SLICE 25 35
