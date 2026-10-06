# Command file to run Tilt
$tilt -StandardInput
InputProjections g.ali
OutputFile g.rec
IMAGEBINNED 1
TILTFILE g.tlt
XTILTFILE g.xtilt
THICKNESS 30
FULLIMAGE 48 40
SUBSETSTART 0 0
$if (-e ./savework) ./savework
