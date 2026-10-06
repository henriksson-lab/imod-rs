# Command file to run Tilt
$tilt -StandardInput
InputProjections gb.ali
OutputFile gb.rec
IMAGEBINNED 1
TILTFILE gb.tlt
THICKNESS 30
FULLIMAGE 48 40
SUBSETSTART 0 0
$if (-e ./savework) ./savework
