# Phase 0 real-world fixture manifest

These files are versioned in the pinned IMOD source tree, not copied into a
second fixture directory.  Their hashes make the reference inputs reproducible
while avoiding an unnecessary duplicate of upstream data.  They are authentic
IMOD eTomo unit-test stacks and are the first differential inputs for `header`,
`alterheader`, `newstack`, and `binvol` once their shared MRC I/O layer exists.

| Reference path | Bytes | SHA-256 | First command use |
|---|---:|---|---|
| `IMOD/Etomo/unitTestData/headerTest.st` | 2048 | `d35b1de6fe680a76fd0b084cbc4209d2de9268435b998b6fc354f16657096f0e` | `header` read/report and `alterheader` output |
| `IMOD/Etomo/unitTestData/binning0.5Header.st` | 1152 | `56fb6d6d0afeb29baccad3f40c209e328d3ef3cf4e1c71d5be446065a1f39467` | `header` and `binvol` |
| `IMOD/Etomo/unitTestData/feiHeader.st` | 1280 | `03fcc3b88f09b43bdb6a7f26bce61c27b4dc9f7b4df95f206186f743de02e09c` | extended-header handling |
| `IMOD/Etomo/unitTestData/newerHeaderBinned.st` | 3072 | `3fa2704e2ddef22ea7961671d6b5231c8e212cabc9dc30d153d8f5b7e17d70e9` | binned-stack handling |
| `IMOD/Etomo/unitTestData/bidirSeries.st` | 6144 | `e643ef16f8d37d0da0244a34b9101a96d4f707d2a712d0a36100f4a7ef149a6a` | multi-section / bidirectional stack paths |

The comparison harness must copy inputs into separate reference/Rust working
directories, record the precise command line and environment, and compare
stdout, stderr, status, filenames, and output bytes.  Do not mutate these
reference fixtures in place.

Verification on 2026-09-09 confirmed all five hashes above against the clean
vendored Mercurial revision `1da960f68556`.

## Orchestration command differential evidence

On 2026-09-09 the source Python launchers were run with an isolated temporary
`IMOD_DIR` whose `pylib` points at the vendored `IMOD/pysrc`; no files in the
reference tree were modified.

| Command/input | Reference Python | Rust | Result |
|---|---|---|---|
| `submfg` with no command arguments | exit 0; usage on stdout | exit 0; same usage text | matched, including upstream's trailing apostrophe in the `-k` line |
| `submfg no-such-command` | exit 1; absent `.com`/`.pcm` error on stderr | same | matched |
| `trimvol -s` | exit 1; eliminated-option error on stderr | same | matched |
| `batchruntomo -validation 1 -directive IMOD/Etomo/tests/batch.adoc` | exit 1 because `/tmp/imod-ref/com/directives.csv` is absent | Rust validates the bundled `.adoc`, exits 0, and prints `Directives all seem OK in that file` | source validation dependency unavailable in source checkout |

`vmstopy`, `vmstocsh`, `tcsh`, `newstack`, `clip`, `densmatch`, `findcontrast`,
and `etomo` are unavailable on this workspace. They remain explicit process
boundaries; successful conversion/reconstruction differential runs require an
installed IMOD runtime at the pinned revision.

## Reference-runtime availability

The vendored `IMOD/` tree is an upstream source checkout, not an installed
IMOD runtime: it does not contain executable `trimvol`, `submfg`, or
`batchruntomo` reference commands, and none were available on this workspace's
`PATH` on 2026-09-09.  Rust-side tests therefore use these authentic inputs and
explicit command boundaries, but byte-for-byte reference-command differential
tests remain pending an installed runtime at the pinned revision.  This is a
verification limitation, not evidence of parity.
