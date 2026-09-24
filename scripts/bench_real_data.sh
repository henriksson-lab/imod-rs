#!/bin/bash
# Parity + speed + peak-RSS comparison on REAL cryo-EM movie data, native IMOD
# vs imod-rs (release).  Counterpart to scripts/bench_vs_native.sh, which uses a
# synthetic 120 MB volume; this one runs the commands a microscope operator
# actually runs, on files straight off the detector.
#
#   SRC     where the two raw files live (read-only; they are copied out)
#   B       scratch root; needs ~4 GB
#   REPS    repetitions, best-of (default 3)
#   ONLY    run only cases whose name matches this substring
#
# Each side runs in its own directory, stdout is captured through a pipe so the
# C/Fortran stream ordering is preserved, and every output file is compared
# byte for byte with the two documented upstream non-achievables masked:
# the MRC label timestamp + uninitialised label slots (BUGS.md §2) and the
# TIFF DateTime tag (BUGS.md §7).
export LC_ALL=C   # /usr/bin/time and printf must use '.' decimals

SRC=${SRC:-/husky/otherdataset/teresa/rawEM}
REF=${REF:-/tmp/imod-reference-build}
RSBIN=${RSBIN:-/data/henriksson/github/claude/imod-rs/target/release/imod}
B=${B:-/big/henriksson/realbench}
REPS=${REPS:-3}
ONLY=${ONLY:-}

LINKS=$B/links
DATA=$B/data
mkdir -p $LINKS $DATA
for c in header clip newstack binvol tif2mrc mrc2tif mrcbyte mrctaper; do
  ln -sf $RSBIN $LINKS/$c
done

nat() { env AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib IMOD_DIR=$REF "$@"; }

# ---------------------------------------------------------------- input data
# The two raw files, renamed to keep the case table readable.  Derived MRCs are
# made by NATIVE IMOD so the container is written by IMOD itself (CLAUDE.md
# § "Authoring real inputs") and neither side is measured against its own output.
[ -f $DATA/movie.tif ] || { cp $SRC/*Fractions.tiff $DATA/movie.tif && chmod u+w $DATA/movie.tif; }
[ -f $DATA/movie.eer ] || { cp $SRC/*_EER.eer       $DATA/movie.eer && chmod u+w $DATA/movie.eer; }
if [ ! -f $DATA/mov.mrc ]; then
  echo "setup: tif2mrc movie.tif -> mov.mrc (4096 x 4096 x 45 byte, 755 MB)" >&2
  ( cd $DATA && nat $REF/mrc/tif2mrc movie.tif mov.mrc >/dev/null 2>&1 )
fi
if [ ! -f $DATA/movf.mrc ]; then
  echo "setup: newstack -mode 2 -secs 0-9 -> movf.mrc (4096 x 4096 x 10 float, 671 MB)" >&2
  ( cd $DATA && nat $REF/flib/image/newstack -mode 2 -secs 0-9 mov.mrc movf.mrc >/dev/null 2>&1 )
fi
if [ ! -f $DATA/frame.mrc ]; then
  echo "setup: newstack -secs 0 -> frame.mrc (one 4096 x 4096 byte frame)" >&2
  ( cd $DATA && nat $REF/flib/image/newstack -secs 0 mov.mrc frame.mrc >/dev/null 2>&1 )
fi

# ------------------------------------------------------------------- the cases
# name | native binary | argv        (OUT.* is the output both sides write)
CASES=$(cat <<'EOF'
tif-header|flib/image/header|movie.tif
tif-stat|clip/clip|stat movie.tif
tif-to-mrc|mrc/tif2mrc|movie.tif OUT.mrc
tif-avg2d|clip/clip|average -2d movie.tif OUT.mrc
tif-newstack-bin4|flib/image/newstack|-bin 4 movie.tif OUT.mrc
tif-newstack-mode2|flib/image/newstack|-mode 2 -secs 0-9 movie.tif OUT.mrc
mrc-newstack-bin4|flib/image/newstack|-bin 4 mov.mrc OUT.mrc
mrc-binvol-bin4|flib/image/binvol|-binning 4 mov.mrc OUT.mrc
mrc-mrcbyte|mrc/mrcbyte|movf.mrc OUT.mrc
mrc-taper|mrc/mrctaper|frame.mrc OUT.mrc
mrc-median3|clip/clip|median -3 -n 3 movf.mrc OUT.mrc
mrc-smooth|clip/clip|smooth movf.mrc OUT.mrc
mrc-fft4096|clip/clip|fft frame.mrc OUT.mrc
mrc-to-tif|qttools/mrc2tif/mrc2tif|-s mov.mrc OUT.tif
eer-header|flib/image/header|movie.eer
eer-sec0|flib/image/newstack|-secs 0 movie.eer OUT.mrc
eer-bin8|flib/image/newstack|-bin 8 -secs 0-3 movie.eer OUT.mrc
EOF
)

# which input files each case needs staged into its run directory
inputs_for() {
  case "$1" in
    tif-*)  echo movie.tif ;;
    eer-*)  echo movie.eer ;;
    mrc-newstack-bin4|mrc-binvol-bin4|mrc-to-tif) echo mov.mrc ;;
    mrc-mrcbyte|mrc-median3|mrc-smooth) echo movf.mrc ;;
    mrc-taper|mrc-fft4096) echo frame.mrc ;;
  esac
}

timeit() { # dir cmd... -> "wall maxrss_kb user sys rc"
  local d=$1; shift
  # The command must run with the run directory as cwd -- it writes OUT.* there
  # and reads the staged inputs by bare name, the way a user would.
  ( cd $d && /usr/bin/time -f "%e %M %U %S" -o .time \
      env AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib IMOD_DIR=$REF \
      timeout 1800 "$@" 2>err.txt | cat > out.txt
    exit ${PIPESTATUS[0]} )
  local rc=$?
  # `/usr/bin/time` prepends "Command exited with non-zero status N" on failure,
  # so the format line is the last one, not the only one.
  echo "$(tail -1 $d/.time) $rc"
}

printf "%-19s %-6s %8s %8s %7s %9s %9s %7s %9s %9s %6s %s\n" \
  CASE PARITY "nat_s" "rs_s" "WALL" "natCPU" "rsCPU" "CPU" "natMB" "rsMB" "RSS" "NOTE"

while IFS='|' read -r name natbin args; do
  [ -z "$name" ] && continue
  [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]] && continue
  nd=$B/run/$name/nat; rd=$B/run/$name/rs
  rm -rf $B/run/$name; mkdir -p $nd $rd
  for f in $(inputs_for "$name"); do ln -sf $DATA/$f $nd/$f; ln -sf $DATA/$f $rd/$f; done
  cmdname=$(basename $natbin)

  nbest=999999; nrss=0; rbest=999999; rrss=0; nrc=0; rrc=0; ncpu=999999; rcpu=999999
  for i in $(seq $REPS); do
    ( cd $nd && rm -f OUT.* )
    read t m nu ns rc < <(timeit $nd $REF/$natbin $args)
    awk -v a=$t -v b=$nbest 'BEGIN{exit !(a<b)}' && nbest=$t
    c=$(awk -v u=$nu -v s=$ns 'BEGIN{printf "%.2f", u+s}')
    awk -v a=$c -v b=$ncpu 'BEGIN{exit !(a<b)}' && ncpu=$c
    [ "$m" -gt "$nrss" ] && nrss=$m; nrc=$rc

    ( cd $rd && rm -f OUT.* )
    read t m ru rs2 rc < <(timeit $rd $LINKS/$cmdname $args)
    awk -v a=$t -v b=$rbest 'BEGIN{exit !(a<b)}' && rbest=$t
    c=$(awk -v u=$ru -v s=$rs2 'BEGIN{printf "%.2f", u+s}')
    awk -v a=$c -v b=$rcpu 'BEGIN{exit !(a<b)}' && rcpu=$c
    [ "$m" -gt "$rrss" ] && rrss=$m; rrc=$rc
  done

  par="OK"; note=""
  [ "$nrc" != "$rrc" ] && { par="RC"; note="rc $nrc vs $rrc"; }
  cmp -s $nd/out.txt $rd/out.txt || { par="DIFF"; note="$note stdout"; }
  cmp -s $nd/err.txt $rd/err.txt || { par="DIFF"; note="$note stderr"; }
  for f in $(cd $nd && ls OUT.* 2>/dev/null); do
    if [ -f "$rd/$f" ]; then
      d=$(python3 - "$nd/$f" "$rd/$f" <<'PY'
import re, struct, sys
A, Bf = open(sys.argv[1],'rb'), open(sys.argv[2],'rb')
a, b = A.read(1024), Bf.read(1024)
if len(a) != len(b):
    print("shorthdr"); raise SystemExit
def mask_mrc(x):
    # `mrc_head_new` never clears `hdata->labels` and the header sits on the
    # stack, so native's slots past `nlabl` carry stack/heap addresses that vary
    # with ASLR (BUGS.md §2).  Blank those, and the dd-Mmm-yy HH:MM:SS stamp in
    # the slots that are used.
    x = bytearray(x)
    try: nlabl = struct.unpack('<i', bytes(x[220:224]))[0]
    except Exception: return bytes(x)
    if not 0 <= nlabl <= 10: return bytes(x)
    for i in range(10):
        s = 224 + 80*i
        if s + 80 > len(x): break
        if i < nlabl: x[s+56:s+76] = b' '*20
        else:         x[s:s+80]    = b' '*80
    return bytes(x)
# `iitif.c:2643` writes tm_mon without the +1, so native's TIFF DateTime is
# permanently one month low (BUGS.md §7).  Blank the whole 19-byte tag.
DT = re.compile(rb'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}')
istiff = a[:2] in (b'II', b'MM')
if istiff:
    ta, tb = a + A.read(), b + Bf.read()
    ta, tb = DT.sub(b'?'*19, ta), DT.sub(b'?'*19, tb)
    if len(ta) != len(tb): print("size"); raise SystemExit
    n = sum(1 for i, j in zip(ta, tb) if i != j)
    print(n if n else ""); raise SystemExit
n = sum(1 for i, j in zip(mask_mrc(a), mask_mrc(b)) if i != j)
while True:                                   # pixel data, streamed
    x, y = A.read(1 << 22), Bf.read(1 << 22)
    if not x and not y: break
    if len(x) != len(y): print("size"); raise SystemExit
    n += sum(1 for i, j in zip(x, y) if i != j)
    if n > 1 << 20: break
print(n if n else "")
PY
)
      [ -n "$d" ] && { par="DIFF"; note="$note $f:$d"; }
    else
      par="DIFF"; note="$note missing:$f"
    fi
  done

  # `/usr/bin/time` resolves to 10 ms, so a ratio between two sub-0.05 s numbers
  # is quantisation noise; print "-" rather than a figure that invites reading.
  sp=$(awk  -v n=$nbest -v r=$rbest 'BEGIN{print (n<0.05||r<0.05) ? "-" : sprintf("%.2fx", n/r)}')
  cpr=$(awk -v n=$ncpu  -v r=$rcpu  'BEGIN{print (n<0.05||r<0.05) ? "-" : sprintf("%.2fx", n/r)}')
  rr=$(awk -v n=$nrss -v r=$rrss 'BEGIN{printf "%.2fx", (n>0? r/n : 0)}')
  nmb=$(awk -v v=$nrss 'BEGIN{printf "%.1f", v/1024}')
  rmb=$(awk -v v=$rrss 'BEGIN{printf "%.1f", v/1024}')
  printf "%-19s %-6s %8s %8s %7s %9s %9s %7s %9s %9s %6s %s\n" \
    "$name" "$par" "$nbest" "$rbest" "$sp" "$ncpu" "$rcpu" "$cpr" "$nmb" "$rmb" "$rr" "$note"
done <<< "$CASES"
