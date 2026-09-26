# A command file exercising the constructs the runner supports
#
$echo "Starting the run"
$set name = out
$set scale = 1.5
$setenv COMRUN_TEST_VALUE fromsetenv
$newstack -StandardInput
InputFile	in.mrc
# a comment between entries
OutputFile	%name.mrc
ModeToOutput	2
$b3dcopy -p in.mrc copy.mrc
$if (-e copy.mrc) header -size copy.mrc
$if (! -e copy.mrc) newstack in.mrc notmade.mrc
$if (-e "copy.mrc") newstack -in in.mrc \
  -ou continued.mrc
$goto skip
$newstack in.mrc skipped.mrc
$skip:
$echo value ${COMRUN_TEST_VALUE} scale $scale
$b3dremove -g cop?.mrc
$echo
