#!/usr/bin/env python3
"""Regenerate the native goldens for one of the eTomo-launched Python scripts
translated on 2026-10-05 (b3dtouch, slicesforsample, sampletilt, squeezevol,
splitcorrection, finishjoin, makejoincom, ... -- see the make-<script>-goldens.sh
wrappers).

Usage: make-pyscript-goldens.py <script>     (called by make-<script>-goldens.sh)

The case format and the recorded files are those of make-pysetup-goldens.py, so
the suites run through tests/pysetup_common: each fixtures/<script>/cases.tsv
row is  name<TAB>inputs<TAB>env<TAB>arguments  (inputs `src:dst` copies from
fixtures/<script>/inputs, `-` for none; env `NAME=value` pairs; arguments split
on blanks).  The native script (IMOD/pysrc/<script>) runs under python3 with
PYTHONUNBUFFERED=1 -- so its output is in program order, as ours is (CLAUDE.md:
the pipe-buffering order is not a criterion) -- in a fresh directory, with
IMOD_DIR at REF (default /tmp/imod-reference-build) and EVERY native program
and native Python script on PATH.  Recorded: golden/<name>.rc, .out (stdout),
.err, .files (every file left) and golden/<name>/ (every file that is not an
unchanged input).  The inputs themselves are committed fixtures (their
provenance is in each script's make-<script>-goldens.sh or inputs/README).
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.environ.get('REF', '/tmp/imod-reference-build')
SCRIPT = sys.argv[1]
HERE = os.path.join(ROOT, 'fixtures', SCRIPT)


def native_bindir():
    """A directory linking every native program of the reference build and
    every IMOD/pysrc script."""
    bindir = tempfile.mkdtemp()
    for top in os.listdir(REF):
        sub = os.path.join(REF, top)
        if not os.path.isdir(sub) or top in ('buildlib', 'pysrc', 'scripts', 'sysdep', 'dist',
                                             'Etomo', 'html', 'manpages'):
            continue
        for rootdir, dirs, files in os.walk(sub):
            if rootdir.count(os.sep) - sub.count(os.sep) > 1:
                continue
            for fn in files:
                p = os.path.join(rootdir, fn)
                if (os.access(p, os.X_OK) and '.' not in fn and not fn.endswith('.sh')
                        and not os.path.exists(os.path.join(bindir, fn))):
                    os.symlink(p, os.path.join(bindir, fn))
    pysrc = os.path.join(ROOT, 'IMOD', 'pysrc')
    for fn in os.listdir(pysrc):
        p = os.path.join(pysrc, fn)
        if os.access(p, os.X_OK) and not os.path.isdir(p) and '.' not in fn:
            t = os.path.join(bindir, fn)
            if os.path.lexists(t):
                os.remove(t)
            open(t, 'w').write('#!/bin/sh\nexec python3 %s "$@"\n' % p)
            os.chmod(t, 0o755)
    # The window program genhstplt (onegenplot, tomodataplots): the reference
    # binary under Xvfb, with its plax calls recorded into $PLAX_CALL_LOG by
    # the LD_PRELOAD shim fixtures/plaxrec.c.  PLAX_REDRAWS=2 keeps its
    # window at the requested width when it saves (one redraw leaves it a
    # pixel wider, which changes the scaling).  When it reaches
    # plax_wait_for_close (a "wait" record) the window is closed: the
    # program is ended with status 0, as closing it does.
    native = os.path.join(REF, 'flib', 'graphics', 'genhstplt')
    if os.path.exists(native):
        so = os.path.join(bindir, 'plaxrec.so')
        subprocess.run(['gcc', '-shared', '-fPIC', '-O2', '-o', so,
                        os.path.join(ROOT, 'fixtures', 'plaxrec.c'), '-ldl'], check=True)
        inner = os.path.join(bindir, 'genhstplt-inner')
        open(inner, 'w').write(r"""#!/bin/bash
echo $$ > "$PLAX_PID_FILE"
export PLAX_REDRAWS=2 LD_PRELOAD=%s
exec %s "$@" < "$PLAX_INPUT"
""" % (so, native))
        os.chmod(inner, 0o755)
        t = os.path.join(bindir, 'genhstplt')
        if os.path.lexists(t):
            os.remove(t)
        open(t, 'w').write(r"""#!/bin/bash
export PLAX_INPUT=$(mktemp) PLAX_PID_FILE=$(mktemp)
cat > "$PLAX_INPUT"
xvfb-run -a -s "-screen 0 1600x1200x24" %s "$@" &
bg=$!
while kill -0 $bg 2>/dev/null; do
  if [ -n "$PLAX_CALL_LOG" ] && [ "$(tail -n 1 "$PLAX_CALL_LOG" 2>/dev/null)" = wait ]; then
    sleep 1; kill "$(cat "$PLAX_PID_FILE")" 2>/dev/null; wait $bg
    rm -f "$PLAX_INPUT" "$PLAX_PID_FILE"; exit 0
  fi
  sleep 0.2
done
wait $bg; rc=$?; rm -f "$PLAX_INPUT" "$PLAX_PID_FILE"; exit $rc
""" % inner)
        os.chmod(t, 0o755)
    return bindir


def native_env(bindir):
    env = dict(os.environ)
    env.update({'AUTODOC_DIR': REF + '/autodoc', 'LD_LIBRARY_PATH': REF + '/buildlib',
                'IMOD_DIR': REF, 'PYTHONPATH': ROOT + '/IMOD/pysrc', 'PYTHONUNBUFFERED': '1',
                'PATH': bindir + ':' + os.environ['PATH'], 'OMP_NUM_THREADS': '1',
                'PLAX_CALL_LOG': 'genhstplt.calls'})
    for k in ('IMOD_OUTPUT_FORMAT', 'TEST_NAMING_STYLE', 'TEST_USE_PCM_FOR_COM',
              'PARALLEL_BOUNDARY_SIZE', 'RUNCMD_VERBOSE'):
        env.pop(k, None)
    return env


def populate(work, inputs):
    placed = {}
    if inputs == '-':
        return placed
    for item in inputs.split():
        src, dst = item.split(':')
        s = os.path.join(HERE, 'inputs', src)
        if src.endswith('/'):
            for rootdir, dirs, files in os.walk(s):
                for fn in files:
                    rel = os.path.relpath(os.path.join(rootdir, fn), s)
                    t = os.path.join(work, dst, rel)
                    os.makedirs(os.path.dirname(t), exist_ok=True)
                    shutil.copy(os.path.join(rootdir, fn), t)
                    placed[os.path.relpath(t, work)] = open(t, 'rb').read()
        else:
            t = os.path.join(work, dst)
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copy(s, t)
            placed[os.path.normpath(dst)] = open(t, 'rb').read()
    return placed


def main():
    bindir = native_bindir()
    env = native_env(bindir)
    golden = os.path.join(HERE, 'golden')
    shutil.rmtree(golden, ignore_errors=True)
    os.makedirs(golden)
    for row in open(os.path.join(HERE, 'cases.tsv')):
        if os.environ.get('FULL') and row.startswith('#full\t'):
            row = row[len('#full\t'):]
        if row.startswith('#') or not row.strip():
            continue
        name, inputs, envs, args = row.rstrip('\n').split('\t')
        work = tempfile.mkdtemp()
        placed = populate(work, inputs)
        cenv = dict(env)
        if envs != '-':
            for kv in envs.split():
                k, v = kv.split('=', 1)
                cenv[k] = v
        argv = [] if args == '-' else args.split(' ')
        p = subprocess.run(['python3', os.path.join(ROOT, 'IMOD/pysrc', SCRIPT)] + argv, cwd=work,
                           env=cenv, capture_output=True, stdin=subprocess.DEVNULL)
        open(os.path.join(golden, name + '.rc'), 'w').write('%d\n' % p.returncode)
        open(os.path.join(golden, name + '.out'), 'wb').write(p.stdout)
        open(os.path.join(golden, name + '.err'), 'wb').write(p.stderr)
        remaining = []
        for rootdir, dirs, files in os.walk(work):
            for fn in files:
                remaining.append(os.path.relpath(os.path.join(rootdir, fn), work))
        open(os.path.join(golden, name + '.files'), 'w').write(
            ''.join(f + '\n' for f in sorted(remaining)))
        for rootdir, dirs, files in os.walk(work):
            for fn in files:
                f = os.path.join(rootdir, fn)
                rel = os.path.relpath(f, work)
                data = open(f, 'rb').read()
                if placed.get(rel) == data:
                    continue
                t = os.path.join(golden, name, rel)
                os.makedirs(os.path.dirname(t), exist_ok=True)
                open(t, 'wb').write(data)
                shutil.copymode(f, t)
        shutil.rmtree(work)
        print(name, p.returncode)
    shutil.rmtree(bindir)


main()
