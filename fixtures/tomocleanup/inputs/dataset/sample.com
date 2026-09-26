# THIS IS A COMMAND FILE TO MAKE 3 TOMOGRAM SAMPLES
#
####CreatedVersion#### 5.2.17
# 
# The sample aligned stacks will each be this number of pixels in Y
$setenv IMOD_OUTPUT_FORMAT MRC
>numLines = 62
# The sample tomograms will have this number of slices
>numSlices = 20
#
$b3dcopy tilt.com tilt_sample.com
>montYlims = [27, 69, 60, 102, -6, 36]
>noSFS = False
>try:
>  sliceOut = runcmd('slicesforsample 42 33 128 96 tilt.com', inStderr = 'stdout')
>  lsplit = sliceOut[0].split()
>  newLines = int(lsplit[0])
>  if newLines:
>    numLines = newLines
>    for ind in range(6):
>      montYlims[ind] = int(lsplit[ind + 1])
>except ImodpyError:
>  scriptErr = False
>  for l in getErrStrings():
>    if 'ERROR:' in l:
>      scriptErr = True
>      prnstr(l, end='', file=log)
>    else:
>      prnstr('ERROR: ' + l, end='', file=log)
>  if scriptErr:
>    closeExit(1)
>  noSFS = True
>except Exception:
>  prnstr('ERROR: sample com file - converting output from slicesforsample', file = log)
>  closeExit(1)
>ymin = montYlims[0]
>ymax = montYlims[1]
# THESE ARE COMMANDS TO MAKE AN ALIGNED STACK FROM THE ORIGINAL STACK
#
####CreatedVersion####5.2.17
#
# It assumes that the views are in order in the image stack
#  
# The size argument should be ,, for the full area or specify the desired 
# size (e.g.: ,10)
#
# The offset argument should be 0,0 for no offset, 0,300 to take an area
# 300 pixels above the center, etc.
#
$setenv IMOD_OUTPUT_FORMAT MRC
$newstack -StandardInput
InputFile	b.st
OutputFile 	b_pos_ali.mrc
ModeToOutput 1
TransformFile	b.xf
SizeToOutputInXandY	96,%numLines
OffsetsInXandY	 0,0
#DistortionField	.idf
ImagesAreBinned	1
#GradientFile	b.maggrad
>if noSFS:
$  sampletilt 11 30 43 b mid_rec.mrc tilt_sample.com b_pos_ali.mrc
>else:
$  sampletilt 11 30 43 b mid_rec.mrc tilt_sample.com 42 %numLines %numSlices b_pos_ali.mrc
>ymin = montYlims[2]
>ymax = montYlims[3]
# THESE ARE COMMANDS TO MAKE AN ALIGNED STACK FROM THE ORIGINAL STACK
#
####CreatedVersion####5.2.17
#
# It assumes that the views are in order in the image stack
#  
# The size argument should be ,, for the full area or specify the desired 
# size (e.g.: ,10)
#
# The offset argument should be 0,0 for no offset, 0,300 to take an area
# 300 pixels above the center, etc.
#
$setenv IMOD_OUTPUT_FORMAT MRC
$newstack -StandardInput
InputFile	b.st
OutputFile 	b_pos_ali.mrc
ModeToOutput 1
TransformFile	b.xf
SizeToOutputInXandY	96,%numLines
OffsetsInXandY	 0,33
#DistortionField	.idf
ImagesAreBinned	1
#GradientFile	b.maggrad
>if noSFS:
$  sampletilt 11 30 76 b top_rec.mrc tilt_sample.com b_pos_ali.mrc
>else:
$  sampletilt 11 30 76 b top_rec.mrc tilt_sample.com 42 %numLines %numSlices b_pos_ali.mrc
>ymin = montYlims[4]
>ymax = montYlims[5]
# THESE ARE COMMANDS TO MAKE AN ALIGNED STACK FROM THE ORIGINAL STACK
#
####CreatedVersion####5.2.17
#
# It assumes that the views are in order in the image stack
#  
# The size argument should be ,, for the full area or specify the desired 
# size (e.g.: ,10)
#
# The offset argument should be 0,0 for no offset, 0,300 to take an area
# 300 pixels above the center, etc.
#
$setenv IMOD_OUTPUT_FORMAT MRC
$newstack -StandardInput
InputFile	b.st
OutputFile 	b_pos_ali.mrc
ModeToOutput 1
TransformFile	b.xf
SizeToOutputInXandY	96,%numLines
OffsetsInXandY	 0,-33
#DistortionField	.idf
ImagesAreBinned	1
#GradientFile	b.maggrad
>if noSFS:
$  sampletilt 11 30 10 b bot_rec.mrc tilt_sample.com b_pos_ali.mrc
>else:
$  sampletilt 11 30 10 b bot_rec.mrc tilt_sample.com 42 %numLines %numSlices b_pos_ali.mrc
$if (-e ./savework) ./savework
