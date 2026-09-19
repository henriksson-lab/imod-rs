#!/bin/bash
export LC_ALL=C   # /usr/bin/time and printf must use '.' decimals
# Parity + speed + peak-RSS comparison, native IMOD vs imod-rs (release).
# Each side runs in its own directory; stdout is captured through a pipe so the
# C/Fortran stream ordering is preserved.
REF=/tmp/imod-reference-build
RSBIN=/data/henriksson/github/claude/imod-rs/target/release/imod
B=/big/henriksson/bench
LINKS=$B/links
REPS=${REPS:-3}

mkdir -p $LINKS
for c in header clip binvol newstack mrc2tif tif2mrc trimvol imodinfo mrcbyte mrctaper; do
  ln -sf $RSBIN $LINKS/$c
done

# name | native binary | argv
CASES=$(cat <<'EOF'
header-meta|flib/image/header|vol_f.mrc
clip-stat|clip/clip|stat vol_f.mrc
clip-median3|clip/clip|median -3 -n 3 vol_f.mrc OUT.mrc
clip-smooth|clip/clip|smooth vol_f.mrc OUT.mrc
clip-fft|clip/clip|fft vol_f.mrc OUT.mrc
clip-bright|clip/clip|brightness -n 1.2 vol_f.mrc OUT.mrc
binvol-bin2|flib/image/binvol|-bin 2 vol_f.mrc OUT.mrc
newstack-bin2|flib/image/newstack|-bin 2 vol_f.mrc OUT.mrc
newstack-mode0|flib/image/newstack|-mode 0 -float 2 vol_f.mrc OUT.mrc
mrc2tif-stack|qttools/mrc2tif/mrc2tif|-s vol_f.mrc OUT.tif
newstack-shrink|flib/image/newstack|-shrink 1.5 vol_f.mrc OUT.mrc
mrctaper-edge|mrc/mrctaper|vol_f.mrc OUT.mrc
mrcbyte-conv|mrc/mrcbyte|vol_f.mrc OUT.mrc
imodinfo-model|imodutil/imodinfo|model.mod
EOF
)

timeit() { # dir cmd... -> "wall_s max_rss_kb rc"
  local d=$1; shift
  /usr/bin/time -f "%e %M %U %S" -o $d/.time \
    env AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib IMOD_DIR=$REF \
    "$@" 2>$d/err.txt | cat > $d/out.txt
  local rc=${PIPESTATUS[0]}
  echo "$(cat $d/.time) $rc"
}

printf "%-16s %-7s %7s %7s %7s %8s %8s %8s %7s %7s %6s %s\n" \
  CASE PARITY "nat_s" "rs_s" "WALL" "natCPU" "rsCPU" "CPU" "natMB" "rsMB" "RSS" "NOTE"

while IFS='|' read -r name natbin args; do
  [ -z "$name" ] && continue
  nd=$B/run/$name/nat; rd=$B/run/$name/rs
  rm -rf $B/run/$name; mkdir -p $nd $rd
  cp $B/data/vol_f.mrc $B/data/model.mod $nd/ 2>/dev/null
  cp $B/data/vol_f.mrc $B/data/model.mod $rd/ 2>/dev/null
  cmdname=$(basename $natbin)

  nbest=99999; nrss=0; rbest=99999; rrss=0; nrc=0; rrc=0; ncpu=999; rcpu=999
  for i in $(seq $REPS); do
    (cd $nd && rm -f OUT.*; read t m uu ss rc < <(timeit $nd $REF/$natbin $args); echo "$t $m $uu $ss $rc" > .res)
    read t m nu ns rc < $nd/.res
    awk -v a=$t -v b=$nbest 'BEGIN{exit !(a<b)}' && nbest=$t
    c=$(awk -v u=$nu -v s=$ns 'BEGIN{printf "%.2f", u+s}')
    awk -v a=$c -v b=$ncpu 'BEGIN{exit !(a<b)}' && ncpu=$c
    [ "$m" -gt "$nrss" ] && nrss=$m; nrc=$rc
    (cd $rd && rm -f OUT.*; read t m uu ss rc < <(timeit $rd $LINKS/$cmdname $args); echo "$t $m $uu $ss $rc" > .res)
    read t m ru rs2 rc < $rd/.res
    awk -v a=$t -v b=$rbest 'BEGIN{exit !(a<b)}' && rbest=$t
    c=$(awk -v u=$ru -v s=$rs2 'BEGIN{printf "%.2f", u+s}')
    awk -v a=$c -v b=$rcpu 'BEGIN{exit !(a<b)}' && rcpu=$c
    [ "$m" -gt "$rrss" ] && rrss=$m; rrc=$rc
  done

  # parity: rc + stdout + stderr + output bytes, with the MRC label timestamp masked
  par="OK"; note=""
  [ "$nrc" != "$rrc" ] && { par="RC"; note="rc $nrc vs $rrc"; }
  cmp -s $nd/out.txt $rd/out.txt || { par="DIFF"; note="$note stdout"; }
  cmp -s $nd/err.txt $rd/err.txt || { par="DIFF"; note="$note stderr"; }
  for f in $(cd $nd && ls OUT.* 2>/dev/null); do
    if [ -f "$rd/$f" ]; then
      d=$(python3 - "$nd/$f" "$rd/$f" <<'PY'
import sys
a=open(sys.argv[1],'rb').read(); b=open(sys.argv[2],'rb').read()
if len(a)!=len(b): print("size"); raise SystemExit
def mask(x):
    x=bytearray(x)
    for s in range(224,224+80*10,80):      # MRC label slots: blank the dd-Mmm-yy HH:MM:SS stamp
        if s+80<=len(x): x[s+56:s+76]=b' '*20
    return bytes(x)
n=sum(1 for i,j in zip(mask(a),mask(b)) if i!=j)
print(n if n else "")
PY
)
      [ -n "$d" ] && { par="DIFF"; note="$note $f:$d"; }
    else
      par="DIFF"; note="$note missing:$f"
    fi
  done

  sp=$(awk -v n=$nbest -v r=$rbest 'BEGIN{printf "%.2fx", (r>0? n/r : 0)}')
  rr=$(awk -v n=$nrss -v r=$rrss 'BEGIN{printf "%.2fx", (n>0? r/n : 0)}')
  nmb=$(awk -v v=$nrss 'BEGIN{printf "%.1f", v/1024}')
  rmb=$(awk -v v=$rrss 'BEGIN{printf "%.1f", v/1024}')
  cpr=$(awk -v n=$ncpu -v r=$rcpu 'BEGIN{printf "%.2fx", (r>0? n/r : 0)}')
  printf "%-16s %-7s %7s %7s %7s %8s %8s %8s %7s %7s %6s %s\n" \
    "$name" "$par" "$nbest" "$rbest" "$sp" "$ncpu" "$rcpu" "$cpr" "$nmb" "$rmb" "$rr" "$note"
done <<< "$CASES"
