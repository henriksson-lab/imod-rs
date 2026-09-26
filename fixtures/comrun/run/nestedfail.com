$vmstocsh subfail.log < subfail.com | csh -ef
$if ($status) goto error
$newstack in.mrc afternested.mrc
$exit 0
$error:
$echo "ERROR: nested failed"
$exit 1
