# THIS IS A COM FILE FOR RUNNING MTFFILTER
#
####CreatedVersion####5.2.17
#
# THE LOW PASS RADIUS AND SIGMA WILL APPLY A TWO-DIMENSIONAL FILTER TO HIGH
# FREQUENCIES AND CAN BE USED TO REPLACE THE ONE-DIMENSIONAL RADIAL
# FILTER IN TILT
#
# TO TEST ON A SUBSET OF VIEWS, INSERT A LINE WITH
#   StartingAndEndingZ      View1,View2
#
# TO APPLY AN INVERSE MTF FILTER, INSERT A LINE WITH
#   MtfFile       filename.mtf
#
# TO USE THE OUTPUT FOR GENERATING A TOMOGRAM, 
#  RENAME b_filt_ali.mrc TO b_ali.mrc
#
$setenv IMOD_OUTPUT_FORMAT MRC
$mtffilter -StandardInput
InputFile       b_ali.mrc
OutputFile      b_filt_ali.mrc
LowPassRadiusSigma        0.35,0.05
InverseRolloffRadiusSigma       0.12,0.05
MaximumInverse  4.0
Voltage         300
PixelSize	1.0
DoseWeightingFile	b.st.mdoc
#
# INSERT NEW LINES ABOVE HERE
$if (-e ./savework) ./savework
