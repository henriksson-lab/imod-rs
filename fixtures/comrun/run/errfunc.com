$set process = step1.com
$header -size in.mrc
$if ($status) goto error
$set process = step2.com
$header -size missing.mrc
$if ($status) goto error
$newstack in.mrc never.mrc
$echo "ALL DONE"
$exit 0
$error:
$echo "ERROR: $process failed"
$exit 1
