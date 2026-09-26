# old style
$tilt
TS01.ali
# comment
TS01.rec
IMAGEBINNED 1
THICKNESS 120
FULLIMAGE 512 512
LOG 0
SCALE 0 1000
$if (-e ./savework) ./savework
