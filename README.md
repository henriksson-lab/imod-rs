# imod-rs

An in-progress, source-auditable Rust translation of the IMOD commands used by
[SNARTomo](https://github.com/rubenlab/snartomo).  It is currently a Phase 0
scaffold: command binaries intentionally stop at untranslated stubs and must
not be used for data processing.

**Not yet ready**

## Reference source

The unmodified reference tree is [`IMOD/`](IMOD/), pinned to Mercurial changeset
`1da960f68556bf3d737eb166c2623b69e9f66baf` (short hash `1da960f68556`), dated
2026-08-29.  Each Rust module mirrors its reference path beneath `src/imod/`;
paired C/C++ headers are merged into their implementation module.  The exact
translation and verification rules are in [TODO.md](TODO.md).

## Scope

The initial command set is `newstack`, `header`, `binvol`, `alterheader`,
`clip`, `tif2mrc`, `mrc2tif`, `imodinfo`, `imodjoin`, `wmod2imod`, and
`convertmod`.  Rust command names intentionally match IMOD's installed binary
names.  Python/Java orchestration (`trimvol`, `submfg`, `etomo`, and
`batchruntomo`) is tracked but not yet translated.

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
