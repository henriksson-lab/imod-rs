#!/bin/bash
# Regenerate fixtures/genhstplt/golden/ from the native reference genhstplt.
# Every row of fixtures/genhstplt/cases.tsv (name, input, arguments) runs the
# reference program under Xvfb in a fresh directory holding copies of the
# data files, with standard input from inputs/<input>.in, PLAX_REDRAWS=2 (one
# redraw leaves the window a pixel wider than requested when it saves, which
# changes the scaling), unbuffered output (so a run that is ended keeps it)
# and its plax calls recorded into genhstplt.calls by
# the LD_PRELOAD shim fixtures/plaxrec.c.  A run that reaches
# plax_wait_for_close (a "wait" record) has its window closed: the program is
# ended with status 0, as closing it does.  golden/<name>.rc is the exit
# status, .stdout the standard output, and golden/<name>.out.<file> each file
# the run created (the call log, gmeta.ps, saved images, typed-out values).
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/genhstplt" && pwd)
S=$(mktemp -d)
gcc -shared -fPIC -O2 -o "$S/plaxrec.so" "$F/../plaxrec.c" -ldl
mkdir -p "$F/golden"
rm -f "$F"/golden/*
sed "/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name input args; do
  [ -z "$name" ] && continue
  [ "$args" = "-" ] && args=""
  d="$S/$name"; mkdir "$d"
  cp "$F"/inputs/*.txt "$d"/
  cat > "$S/inner" <<INNER
#!/bin/bash
echo \$\$ > "$S/pid"
cd "$d"
export GFORTRAN_UNBUFFERED_PRECONNECTED=y PLAX_REDRAWS=2 PLAX_CALL_LOG=genhstplt.calls LD_PRELOAD=$S/plaxrec.so LD_LIBRARY_PATH=$R/buildlib IMOD_DIR=$R
exec $R/flib/graphics/genhstplt $args < "$F/inputs/$input.in" > "$S/stdout" 2> /dev/null
INNER
  chmod +x "$S/inner"
  set +e
  xvfb-run -a -s "-screen 0 1600x1200x24" "$S/inner" &
  bg=$!
  rc=
  while kill -0 $bg 2>/dev/null; do
    if [ "$(tail -n 1 "$d/genhstplt.calls" 2>/dev/null)" = wait ]; then
      sleep 1; kill "$(cat "$S/pid")"; wait $bg 2>/dev/null; rc=0; break
    fi
    sleep 0.2
  done
  [ -z "$rc" ] && { wait $bg; rc=$?; }
  set -e
  echo $rc > "$F/golden/$name.rc"
  cp "$S/stdout" "$F/golden/$name.stdout"
  for f in "$d"/*; do
    b=$(basename "$f")
    if [ ! -f "$F/inputs/$b" ] || ! cmp -s "$f" "$F/inputs/$b"; then cp "$f" "$F/golden/$name.out.$b"; fi
  done
done
rm -rf "$S"
