# THIS IS A COMMAND FILE
$newstack -mode 0 -image 2 -xform ts.prexg ts.st ts.preali
$if (-e ./savework) ./savework
