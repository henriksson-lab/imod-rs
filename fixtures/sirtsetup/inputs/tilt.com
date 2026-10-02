# Command file to run Tilt
#
$tilt -StandardInput
InputProjections ts.ali
OutputFile ts.rec
IMAGEBINNED 1
TILTFILE ts.tlt
XTILTFILE ts.xtilt
THICKNESS 30
RADIAL .35 .035
FalloffIsTrueSigma 1
XAXISTILT 0.
LOG 0
SCALE 0 1000
PERPENDICULAR
MODE 2
FULLIMAGE 64 48
SUBSETSTART 0 0
AdjustOrigin 1
$if (-e ./savework) ./savework
