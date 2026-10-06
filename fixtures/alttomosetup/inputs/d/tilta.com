# Command file to run Tilt
$tilt -StandardInput
InputProjections ga.ali
OutputFile ga.rec
IMAGEBINNED 1
TILTFILE ga.tlt
THICKNESS 30
FULLIMAGE 48 40
SUBSETSTART 0 0
$if (-e ./savework) ./savework
