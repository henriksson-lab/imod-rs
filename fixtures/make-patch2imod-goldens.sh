#!/bin/bash
# Regenerate fixtures/patch2imod/ (inputs and golden/) from the native reference.
# Inputs are authored here, seeded: corrsearch3d-style patch files (a first
# line with the patch count, optionally words and value-column IDs, then
# "x z y dx dz dy [values]" lines), the patchcrawl3d comma form, a findwarp
# residual listing, files for -l, broken files for the error paths, and
# version-3 warping files (a two-section grid and a two-section control-point
# file).  For each row of cases.tsv the native program runs in a fresh
# directory with stdout captured through a pipe; golden/<name>.rc is the exit
# status, .stdout the standard output, and .mod the output model, when native
# leaves one.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/patch2imod" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
import numpy as np
r = np.random.default_rng(3)
def patchfile(name, n, ids, nvals, extra='', comma=False, zeros=False, head=True):
    L = [f"{n}{extra}" + ''.join(f" {i}" for i in ids)] if head else []
    k = 0
    for z in (20, 60, 100):
        for y in range(16, 120, 24):
            for x in range(16, 150, 26):
                if k >= n:
                    break
                d = r.normal(0, 1.5, 3)
                if comma:
                    L.append(f"{x:6d}{z:6d}{y:6d}{d[0]:9.2f},{d[2]:9.2f},{d[1]:9.2f}")
                else:
                    v = [0.0 if zeros and r.uniform() < 0.3 else r.uniform(0, 1) for j in range(nvals)]
                    L.append(f"{x:6d}{z:6d}{y:6d}{d[0]:9.2f}{d[2]:9.2f}{d[1]:9.2f}"
                             + ''.join(f"{e:12.4f}" for e in v))
                k += 1
    open(name, 'w').write('\n'.join(L) + '\n')
patchfile('plain.out', 24, [], 0)
patchfile('vals3.out', 24, [3, 7, 12], 3, extra=' positions')
patchfile('vals6.out', 30, [1, 2, 3, 4, 5, 6], 6, zeros=True)
patchfile('comma.out', 12, [], 0, comma=True)
patchfile('lvals.out', 12, [], 2, head=False)
open('short.out', 'w').write('30\n' + ''.join(open('plain.out').read().splitlines(True)[1:]))
open('nolastnl.out', 'w').write('2\n1 2 3 1 1 1\n4 5 6 1 1 1')
# Native equivalents of the cases whose behaviour imod-rs defines (BUGS.md,
# patch2imod): -l on lvals.out is compared with the native run on the same
# lines under a count line with zero value IDs, and nolastnl.out with the
# native run on the same file ending in a newline.
open('lvalshdr.out', 'w').write('12 0 0\n' + open('lvals.out').read())
open('nolastnlnl.out', 'w').write(open('nolastnl.out').read() + '\n')
open('badhead.out', 'w').write('abc 5\n1 2 3 1 1 1\n')
L = ['12 residuals']
for k in range(12):
    L.append(f"{r.uniform(0,500):9.2f}{r.uniform(0,500):9.2f}{k%5:5d}{r.normal():9.3f}{r.normal():9.3f}")
open('resid.out', 'w').write('\n'.join(L) + '\n')
L = ['3', '512 480 2 1 2.5 1']
for s in range(2):
    L += [f"{20.0+s} 60.0 3 30.0 55.0 2", "1 0 0 1 0 0"]
    L += [f"{r.normal():.4f} {r.normal():.4f}" for k in range(6)]
open('grid.warp', 'w').write('\n'.join(L) + '\n')
L = ['3', '512 480 2 1 2.5 3']
for s in range(2):
    L += [f"{4+s}", "1 0 0 1 0 0"]
    L += [f"{r.uniform(0,512):.2f} {r.uniform(0,480):.2f} {r.normal():.4f} {r.normal():.4f}" for k in range(4+s)]
open('ctrl.warp', 'w').write('\n'.join(L) + '\n')
PY
rm -f "$F"/*.out "$F"/*.warp
cp *.out *.warp "$F"/
rm -rf "$F/golden"; mkdir "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ "$args" = "-" ] && args=""
  d="$S/run/$name"; mkdir -p "$d"
  cp "$F"/*.out "$F"/*.warp "$d"/
  set +e
  (cd "$d" && timeout 120 $R/imodutil/patch2imod $args 2>/dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[0]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  [ -f "$d/o.mod" ] && cp "$d/o.mod" "$F/golden/$name.mod"
done
rm -rf "$S"
