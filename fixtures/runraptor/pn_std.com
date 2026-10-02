$newstack -StandardInput
InputFile	ts.st
OutputFile	ts.preali
TransformFile	ts.prexg
ModeToOutput	
ImagesAreBinned	2
$if (-e ./savework) ./savework
