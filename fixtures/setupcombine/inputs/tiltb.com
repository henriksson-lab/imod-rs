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
InputProjections gb.ali
OutputFile gb.rec
IMAGEBINNED 1
TILTFILE gb.tlt
XTILTFILE gb.xtilt
THICKNESS 100
RADIAL .35 .035
FalloffIsTrueSigma 1
XAXISTILT -2.25
LOG 0
SCALE 0 500
PERPENDICULAR
MODE 1
EXCLUDELIST
FULLIMAGE
SUBSETSTART 0 0
AdjustOrigin 1
OFFSET 0.7
SHIFT 0. 3.
