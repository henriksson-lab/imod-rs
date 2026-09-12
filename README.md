# imod-rs

An in-progress, source-auditable Rust translation of the IMOD commands used by
[SNARTomo](https://github.com/rubenlab/snartomo).  The crate includes working,
partially translated command paths with focused native-reference regressions;
it is not yet a complete or production-ready replacement for IMOD.

## Reference source

The unmodified reference tree is [`IMOD/`](IMOD/), pinned to Mercurial changeset
`1da960f68556bf3d737eb166c2623b69e9f66baf` (short hash `1da960f68556`), dated
2026-08-29.  Each Rust module mirrors its reference path beneath `src/imod/`;
paired C/C++ headers are merged into their implementation module.  The exact
translation and verification rules are in [TODO.md](TODO.md).

## Scope

The command set is `newstack`, `header`, `binvol`, `alterheader`, `clip`,
`tif2mrc`, `mrc2tif`, `imodinfo`, `imodjoin`, `wmod2imod`, `convertmod`,
`trimvol`, `submfg`, and `batchruntomo`.

All of them are built into a **single `imod` binary** rather than one
executable per command (a deliberate deviation from upstream; see
[Invocation](#invocation) below). The optional `etomo-gui` launcher renders
the translated Slint presentation of eTomo's `MainFrame`; it is not yet the
source-equivalent `EtomoDirector`/manager/process GUI path. Its source-mapped
wiring order and the separate 3dmod, midas, and processchunks boundaries are
tracked in [GUI_WIRING.md](GUI_WIRING.md). Build that experimental launcher
with `cargo run --features gui --bin imod -- etomo-gui`. The default `mrc2tif`
path uses
the source-shaped Qt/libtiff boundaries. Default-off experimental pure-Rust
TIFF, JPEG/PNG, and FFT paths are documented in
[RUST_NATIVE_BACKENDS_PLAN.md](RUST_NATIVE_BACKENDS_PLAN.md); they are selected
only with their corresponding `IMOD_RS_*_BACKEND` environment variables.

## Invocation

Upstream IMOD installs one executable per command.  This crate deliberately
builds **one** binary, `imod`, which dispatches busybox-style:

1. `basename(argv[0])` is checked first.  A link named after a command runs
   that command with `argv` untouched, so an IMOD-style install can keep its
   usual `$IMOD_DIR/bin/<command>` layout by linking every command name to
   `imod`, and each program sees exactly the `argv` it always saw:

   ```bash
   ln -s /path/to/imod /usr/local/IMOD/bin/newstack
   newstack -bin 2 in.mrc out.mrc
   ```

2. Otherwise `argv[1]` is the subcommand:

   ```bash
   imod header -size file.mrc
   cargo run --bin imod -- newstack -bin 2 in.mrc out.mrc
   ```

   This form re-execs the binary with `argv[0]` rewritten to the command name,
   so the translated program still observes `["<bindir>/header", "-size",
   "file.mrc"]`.  That matters: `clip`, `mrc2tif` and `imodinfo` build their
   exit prefixes from `imodProgName(argv[0])`, and PIP reports the program
   name in its errors.  No translated unit knows the launcher exists.

`imod` with no arguments, with `-h`/`--help`, or with an unrecognised
subcommand prints the command listing on stderr and exits 1.

## Audit workflow

`ccc_mapping.toml` and the reports under `audit/` are retained as historical
Phase 0 scaffold evidence only.  They are not used to select work, establish
coverage, or accept a translation.  The active source-file order is
[ORDER.md](ORDER.md): each selected upstream implementation/header unit is
closed as a whole and compared directly against its original functions, types,
and behavior.

The reference build/link evidence for dependency closure is
[audit/dependency-closure.md](audit/dependency-closure.md).  Acceptance uses
focused tests and side-by-side reference-versus-Rust runs on real fixtures,
including exit status, stdout, stderr, output file bytes, and metadata.

For difficult C units, `scripts/c2rust_baseline.sh IMOD/path/unit.c /tmp/output`
can generate a disposable C2Rust parity reference using the IMOD configuration
types.  It is not a second implementation and must not be committed as a
standalone crate: each generated source function is integrated into the
source-mirrored module, then tested against the original implementation.

## Attribution and citation

This project translates IMOD; it is not an original implementation of IMOD.
Please cite the applicable IMOD work when publishing results produced with it:

- Kremer JR, Mastronarde DN, McIntosh JR. *Computer Visualization of
  Three-Dimensional Image Data Using IMOD.* Journal of Structural Biology.
  1996;116(1):71–76. doi:10.1006/jsbi.1996.0013.
- Mastronarde DN, Held SR. *Automated tilt series alignment and tomographic
  reconstruction in IMOD.* Journal of Structural Biology. 2017;197(2):102–113.
  doi:10.1016/j.jsb.2016.07.011.

## Licensing

The vendored IMOD copyright notice and license texts are reproduced unchanged
in [`LICENSES/`](LICENSES/).  IMOD states that most software is GPL-2.0 and
its entirely C/C++ libraries are LGPL; translated modules must retain the
applicable upstream license and notices.  No independent distribution license
has been selected for this work beyond that obligation.
