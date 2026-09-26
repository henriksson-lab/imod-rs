# `fixtures/` — test inputs and native golden manifests

Everything the integration tests read lives here: the **inputs** each suite
runs on, and the **native goldens** — what the reference IMOD build
(`/tmp/imod-reference-build`, see `CLAUDE.md`) produced from them.
Provenance of individual inputs is in `FIXTURE-MANIFEST.md`; the vendored
header-only stubs are in `IMOD-FIXTURES.md`.

## Goldens are digests, not files

Native outputs are **not stored byte for byte**.  Each suite has one file,
`fixtures/<suite>/golden.manifest` (`fixtures/golden.manifest` for the loose
files at the top of `fixtures/`), holding one entry per native output, keyed by
the path the output used to have (`golden/<case>.rc`,
`golden/<case>/o.mrc`, `defined/<case>.out`, …):

```
text <len> golden/<case>.rc          <- followed by the <len> bytes, whole
0
sha256 <hex> <len> golden/<case>/o.mrc
```

- **`text`** entries are kept whole: exit statuses, standard output of at
  most 512 bytes, and anything a test *reads* rather than only compares
  (file listings, stand-in program output), so a failure shows both sides.
- **`sha256`** entries are the SHA-256 and length of the **expected** bytes:
  native's output after the same masking the suite applies to both sides
  (`common::mask_stamps` and friends — wall-clock stamps, version banners,
  sorted line multisets, …) and, where the suite reconciles the regions native
  writes from uninitialised memory (`common::reconcile_uninitialised`,
  `BUGS.md` §2), with the defined values substituted.  Our masked output must
  hash to exactly that.

Tests address goldens through `tests/common/golden.rs` by their old paths:
`golden::load(path)` / `expect(path)` return a `Golden`, compared with
`matches`, `matches_masked(ours, mask)`, `matches_reconciled(ours, mask)` or
`compare(ours, mask, reconcile) -> Result<(), String>`; `read`/`read_to_string`
return a whole (`text`) entry; `list(dir)` lists what a golden directory held.

### Regenerating

```
fixtures/regen-golden.sh <suite>          # make native outputs, record, delete them
KEEP=1 fixtures/regen-golden.sh <suite>   # ... and keep the native files
RECORD_ONLY=1 fixtures/regen-golden.sh <suite>   # native files already in place
```

The script runs `fixtures/make-<suite>-goldens.sh` (unchanged: it still writes
native's full outputs to `fixtures/<suite>/golden/`, which is git-ignored),
then the suite's tests with `IMOD_RS_GOLDEN_RECORD=1`, then deletes the native
files.  In record mode every golden is read from the real file, every
comparison runs exactly as it always did — so a record run is also a full
native differential — and each entry is (re)written into the manifest.  The
manifest is merged, not replaced: delete it first to drop stale entries.

`defined/` expectations (upstream bugs fixed in translation; `BUGS.md`) come
from our own build.  A record run keeps them from the old manifest unless the
native files for them exist again (`make-<suite>-goldens.sh defined` where the
script has that mode).

### Debugging a mismatch

A failing comparison prints the key, the manifest, both sides (text) or both
digests and lengths, and the command that brings the native file back:

```
golden golden/dist.out (fixtures/xfmodel/golden.manifest) differs:
    native: sha256 3f2a… (44860 bytes)
    ours:   sha256 91c0… (44860 bytes)
    native files: KEEP=1 fixtures/regen-golden.sh xfmodel
```

Run that, and diff `fixtures/<suite>/golden/<file>` against the failing case's
output.  (Or, without rewriting the manifest, run the make script alone.)

**Tolerance suites.**  `tiltalign` accepts LAPACK/BLAS differences
(`CLAUDE.md`).  Replaying a digest is necessarily exact; the tolerance
comparison runs when the native bytes are present (record mode).  Today every
tiltalign golden is byte-identical, so the exact replay is a margin, not a
restriction; if it fails, `KEEP=1 fixtures/regen-golden.sh tiltalign` applies
the tolerance against the real native files and reports what differs.

### Pruned cases and the exhaustive differential

Not every row of a `cases.tsv` runs in the gate.  On 2026-09-26 the tables
were pruned to a representative set — each option / code path once, each
distinct error path once, every case that pins a fixed upstream bug or is
named by another test — and each suite's test header says in one line what
was dropped.  A pruned row is not deleted: it stays in the table as
`#full<TAB><row>`.  The ordinary run skips it (`common::golden::case_rows`);
its golden is not in the manifest.  To run everything against fresh native
outputs:

```
DIFF=1 fixtures/regen-golden.sh <suite>
```

which runs the make script with `FULL=1` (it then includes the `#full` rows)
and the suite with `IMOD_RS_GOLDEN_NATIVE=1 IMOD_RS_FULL_CASES=1`: every
comparison reads the native files, nothing is recorded, and the native files
are left in `fixtures/<suite>/golden/` (git-ignored) for inspection.  Three
inputs only `#full` rows needed were removed because a script remakes them or
the row was rewritten: `ctfphaseflip/s160.mrc` (`strips`; rerun
`make-ctfphaseflip-mtffilter-inputs.py` first), and findsection's `fy.mrc`
and `m1.mrc` (the flipped and multi-tomogram rows now use `m0.mrc`; the old
slabs are in git history).

After pruning rows, drop their now-unused manifest entries: run every suite
that reads the manifest with `IMOD_RS_GOLDEN_ACCESS_LOG=<log>`, then
`fixtures/prune-manifest.py <log> <manifest>...`.

### Goldens that are not plain native output

Some goldens carry the *defined* behaviour of an upstream bug the translation
fixes (`BUGS.md`).  The make scripts apply those after the native run — sed
fixups (`setupcombine`, `matchorwarp`, `splitcombine`, `findsection`,
`corrsearch3d`, `fakevolume`, `combinefft`, …), cases run with our own build
(`xfsimplex` `s_sobel`, `tiltxcorr` `nonopt`/`scan`, `xyzproj`
`make-fixed-cases.txt`), `defined/` from `make-<suite>-goldens.sh defined`
(`ccderaser`, `findbeads3d`), or native-equivalent argument columns.  When a
regeneration changes an entry, check that first: a make script that does not
reapply its fixup reverts the golden to native's bug.

## Inputs

Inputs are real files, kept small (34 MB of fixtures became under 5 MB on
2026-09-26, most of it by storing goldens as digests; the rest by reducing
inputs).  How each suite's inputs were made is in its make script or
`make-inputs.*`; since 2026-09-26:

- **Header-only stubs** where the program never reads pixels: the pysetup
  suites' `.st`/`.rec`/`.ali` volumes (`make-pysetup-goldens.py`,
  `header_only`), `chunksetup/inputs/vol.mrc`, `splittilt/inputs/ali.mrc` —
  the same idea as IMOD's own `IMOD/Etomo/unitTestData` stubs.  Regenerating
  from them gave byte-identical native outputs.
- **Bytes instead of shorts/floats** (native `newstack -mode 0 -scale 0,255`,
  or authored as bytes): `findsection`, `findbeads3d`, `tiltxcorr`,
  `corrsearch3d`, `beadtrack` (`t1.mrc`, 646 -> 324 KB).  `beadtrack`'s
  Sobel-centring goldens are *defined* (`defined.list`,
  `make-beadtrack-goldens.sh defined`): the Sobel peak-scaling fix
  (`BUGS.md`) moves every Sobel position, so on any input those cases leave
  native's path at the first Sobel round.  An earlier note here blamed a
  byte-converted or 13-view input for diverging from native there; a build
  with only that fix reverted matches native on the original, the byte copy
  and the 13-view series through the end (2026-09-26), so the divergence was
  the fix and the old fixture had simply been recorded with it.
- **Smaller or fewer images**: `xfsimplex` (64x56 pairs), `matchvol`'s `mb.mrc`
  (30 sections; the same multi-cube layouts), `alignlog`'s local-area log
  (2 x 2 areas).
- **Shared rather than stored twice** (read from the other suite's directory;
  named in the test and make script): `warpvol` ← `matchvol` (`vf`, `vs`,
  `mb`), `findcontrast` (and `trimvol`) ← `densmatch` (`f`, `s`, `b`),
  `refinematch` ← `findwarp` (patches, transforms, models), `mtffilter` ←
  `ctfphaseflip` (`f48.mrc`), `alignlog` ← `IMOD/Etomo/unitTestData/aligna.log`
  (read in place), `mrcinfo` ← `mrcx`.
- **Not reduced**: `matchvol/mb.mrc` needs its size for the
  `-memory` multi-cube layouts.
