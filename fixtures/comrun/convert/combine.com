# THIS IS A MASTER COMMAND FILE TO COMBINE TOMOGRAMS FROM A TWO-AXIS TILT
# SERIES
#
####CreatedVersion#### 4.8.14
# 
# It runs a command file for each step in turn.  To change parameters for
# a particular procedure, edit the command file for that step.
#
$echo "This is the master log file for combine.com"
$echo "It will end with COMBINE SUCCESSFULLY COMPLETED upon success"
$echo "Look in individual log files for the results of particular operations"
$echo " "
#
# To restart at a particular place, just change the following goto
# to go to the destination needed (patchcorr, matchorwarp, warpvol, etc.)
#
$goto solvematch
#
# Run solvematch to find the initial correspondence between volumes
#
$solvematch:
$set process = solvematch.com
$echo "Running solvematch.com"
$vmstocsh solvematch.log < solvematch.com | csh -ef
$if ($status) goto error
$goto matchvol1
#
# Or run dualvolmatch
#
$dualvolmatch:
$set process = dualvolmatch.com
$echo "Running dualvolmatch.com"
$vmstocsh dualvolmatch.log < dualvolmatch.com | csh -ef
$if ($status) goto error
#
# Next run matchvol to make matching volume
#
$matchvol1:
$set process = matchvol1.com
$echo "Initial registration found, next running matchvol1.com"
$vmstocsh matchvol1.log < matchvol1.com | csh -ef
$if ($status) goto error
$goto patchcorr
#
# Run autopatchfit and go straight to volcombine
#
$autopatch:
$echo "Matchvol finished, next running autopatchfit"
$set process = autopatchfit
$autopatchfit -final E
$if ($status) goto error
$goto volcombine
#
#
# Next run patchcrawl3d to find patches
# If you change your patch sizes or limits, restart here
#
$patchcorr:
$set process = patchcorr.com
$echo "Matchvol finished, next running patchcorr.com"
$vmstocsh patchcorr.log < patchcorr.com | csh -ef
$if ($status) goto error
#
# Matchorwarp runs Refinematch and Matchvol, or Findwarp and Warpvol.
# If you edit your patches, restart here
#
$matchorwarp:
$set process = matchorwarp.com
$echo "Patchcrawl3d found displacements, next running matchorwarp.com"
$vmstocsh matchorwarp.log < matchorwarp.com | csh -ef
$if ($status) goto error
$goto volcombine
#
# If you run Findwarp by hand and get a warp.xf, restart here
#
$warpvol:
$set process = warpvol.com
$vmstocsh warpvol.log < warpvol.com | csh -ef
$if ($status) goto error
$goto volcombine
#
# This Matchvol is run only if you have to do Findwarp by hand, get a good
# refine.xf by omitting rows or columns, and have to restart the process
#
$matchvol2:
$set process = matchvol2.com
$vmstocsh matchvol2.log < matchvol2.com | csh -ef
$if ($status) goto error
$goto volcombine
#
# This runs everything else to combine volumes
#
$volcombine:
$set process = volcombine.com
$echo "Matchvol or Warpvol finished, next running volcombine.com"
$(vmstocsh volcombine.log < volcombine.com >! volcombine.csh)
$if (-e /bin/dos2unix.exe) dos2unix volcombine.csh
$csh -ef volcombine.csh
$if ($status) goto error
$echo "COMBINE SUCCESSFULLY COMPLETED"
$\rm -f volcombine.csh
$exit 0
$error:
$echo "ERROR: $process failed"
$\rm -f volcombine.csh
$exit 1
