$echo "before"
$header -size missing.mrc
$newstack in.mrc after.mrc
