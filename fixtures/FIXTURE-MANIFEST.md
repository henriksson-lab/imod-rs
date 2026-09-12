# Fixture provenance manifest

Reference revision: Mercurial `1da960f68556bf3d737eb166c2623b69e9f66baf`
(`IMOD/.version` = 5.2.17, `IMOD/setup2:10` copyright 1994-2025).

Reference binaries: built from the vendored tree under `/tmp/imod-reference-build`,
invoked with
`AUTODOC_DIR=/tmp/imod-reference-build/autodoc LD_LIBRARY_PATH=/tmp/imod-reference-build/buildlib`.
That build is configured `NO_HDF_LIB` (`include/imodconfig.h:11`), so it has no
HDF support; the Rust crate links real hdf5.  A **second, HDF-enabled**
reference build now exists at `<scratchpad>/imod-abi-src`, run the same way
(`AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib`), and HDF
differentials are taken against that one — they are no longer a verification
limitation.

## Versioned fixtures used directly

These live in the pinned tree and are never mutated in place; every harness
copies them into a scratch directory first.

| SHA-256 | bytes | path | used for |
|---|---:|---|---|
| `d35b1de6fe680a76fd0b084cbc4209d2…` | 2048 | `IMOD/Etomo/unitTestData/headerTest.st` | `header`/`alterheader` report and machine-readable sweeps |
| `56fb6d6d0afeb29baccad3f40c209e32…` | 1152 | `IMOD/Etomo/unitTestData/binning0.5Header.st` | `header` binned-stack path |
| `03fcc3b88f09b43bdb6a7f26bce61c27…` | 1280 | `IMOD/Etomo/unitTestData/feiHeader.st` | Agard/old-FEI extended header, `-modefix` |
| `3fa2704e2ddef22ea7961671d6b5231c…` | 3072 | `IMOD/Etomo/unitTestData/newerHeaderBinned.st` | binned-stack header handling |
| `e643ef16f8d37d0da0244a34b9101a96…` | 6144 | `IMOD/Etomo/unitTestData/bidirSeries.st` | SerialEM extended header inventory, multi-section |
| `ae4172a601c9be4d9102bde7fede729f…` | 44860 | `IMOD/Etomo/uitestData/BB/BBa_erase.fid` | real model with an `OBST` store chunk and 114 bytes of name residue past the NUL; drives `imodinfo` (25 option modes), `imodjoin`, `istore` |
| `c5bfb5739a50a03206f68f281dfea153…` | 528 | `IMOD/com/empty.seed` | real model with no store chunk; `csum` 396 |
| `c5bfb5739a50a03206f68f281dfea1…` | 528 | `fixtures/model-empty-seed.mod` |
| `f040b72f31336923fd0cb7336ebabf…` | 2126 | `fixtures/model-view-clip-label.joined.mod` |

## Autodoc corpus

`IMOD/autodoc/*.adoc` — 140 files, combined SHA-256
`c576ce68bf91efa77afeba41c9d64db7…`.  Round-tripped through `AdocRead`/`AdocWrite`
against the reference-linked C driver (156 inputs, 0 mismatches).  Also the
live PIP option tables for `binvol`, `alterheader`, `header` and `newstack`,
which now require a reachable autodoc directory exactly as the reference does.

## Authored fixtures, and why they are necessary

The vendored `.st` files are **header-only stubs with no pixel data**, so they
cannot exercise any command that reads pixels.  Real volumes are therefore
authored with IMOD's own converter so that the container is written by IMOD
rather than by this project:

```
python3 -c "
import numpy as np
r=np.random.default_rng(20260910); ny,nx,nz=48,64,5
y,x=np.mgrid[0:ny,0:nx]; base=np.sin(x/5.)*np.cos(y/7.)*60+128
v=np.stack([base+r.normal(0,12,(ny,nx))+k*7 for k in range(nz)])
v.astype(np.float32).tofile('f.raw'); np.clip(v,0,255).astype(np.uint8).tofile('b.raw')
np.clip(v*100,-32768,32767).astype(np.int16).tofile('s.raw')"

/tmp/imod-reference-build/mrc/raw2mrc -x 64 -y 48 -z 5 -t float f.raw in_float.mrc
/tmp/imod-reference-build/mrc/raw2mrc -x 64 -y 48 -z 5 -t byte  b.raw in_byte.mrc
/tmp/imod-reference-build/mrc/raw2mrc -x 64 -y 48 -z 5 -t short s.raw in_short.mrc
```

The seed is fixed, so these are reproducible.  Binary models are authored with
the reference `wmod2imod` from a WIMP text file; WIMP text itself is defined by
`IMOD/libimod/imodel_to.c:36-107`.  TIFF variants are authored with Pillow,
`tiffcp` and `tiffset`, with the exact invocation recorded beside each fixture
constant in `tests/tif2mrc_cli.rs`.

### `fixtures/model-view-clip-label.mod`

No vendored model carries an object label (`OLBL`), a contour label (`LABL`), an
object clip-plane chunk (`CLIP`), or a `VIEW` chunk with object views and a
model clip chunk (`MCLP`), or a `MESH` chunk on an object that also has
contours; only `3dmod` writes those, and it is a GUI.  The fixture is
therefore authored byte by byte by
`fixtures/make-model-view-clip-label.py`, written from `IMOD/libimod/imodel_files.c`
and `IMOD/libimod/iview.c` rather than from either implementation's writer, so
it is an independent check on both.  Regenerate with

```
python3 fixtures/make-model-view-clip-label.py fixtures/model-view-clip-label.mod
```

The `.txt` files beside it are the **reference** `imodinfo`'s output for it,
captured through a pipe (`imodinfo -a m.mod | cat > golden.txt`), never a
redirect: the `# MODEL` banner goes out through `printf` and the body through
`fout`, so a redirect moves the banner to the end of the file.  `.ascii.txt` is
both a golden and an input — it is itself a readable ASCII model, because
`imodFgetline` skips the `#` banner.  `.t1.txt`, `.x05.txt` and `.x05s.txt`
cover `-t 1` (clip planes) and `-x 0,5` (subarea), which are the only inputs
that reach `trim_scan_contour` and `scan_contour_area`.

Passing `--closed` to the generator writes `model-closed-contours.mod`: the
same model with object flags 0 and two polygonal contours.  `imodinfo_ellipse`
and `imodinfo_ratios` return immediately for an open or scattered object, so
the `imodContourEquivEllipse` fit and the length/area ratio are only reachable
with a closed one.

`model-empty-seed.mod` is a second, contourless model, and
`model-view-clip-label.joined.mod` is the reference
`imodjoin model-view-clip-label.mod model-empty-seed.mod out.mod` output with
the `MINX` chunk's `oscale` and `orot` zeroed -- `imodjoin.c:181` `malloc`s the
`IrefImage` and fills only `otrans`, `ctrans`, `crot` and `cscale`, so those
twenty-four bytes differ between two reference runs.  Every other byte must
match.

| SHA-256 | bytes | path |
|---|---:|---|
| `a991429c3791ac4582341d3bee07a5…` | 1631 | `fixtures/model-view-clip-label.mod` |
| `784365771788d70fd4e6cf61ceacff…` | 1283 | `fixtures/model-view-clip-label.ascii.txt` |
| `a55f05942294b77024c996e7ab3deb…` | 683 | `fixtures/model-view-clip-label.verbose.txt` |
| `b398b937a193fa4c3ee682ae776af6…` | 427 | `fixtures/model-view-clip-label.c.txt` |
| `cad3d09dd94b1d7d243b8b6b4a3680…` | 397 | `fixtures/model-view-clip-label.i.txt` |
| `c5ac68e2ac76eedda86061e13795c7…` | 785 | `fixtures/model-view-clip-label.F.txt` |
| `525304f208f62238409aa5f03c9646…` | 549 | `fixtures/model-view-clip-label.s.txt` |
| `65c631a4c1927a07ff2accee830cff…` | 397 | `fixtures/model-view-clip-label.t1.txt` |
| `b57011fef36f667b1c01c48a88bd1f…` | 398 | `fixtures/model-view-clip-label.x05.txt` |
| `22cd996ca25a0daf58388e74584171…` | 549 | `fixtures/model-view-clip-label.x05s.txt` |
| `631de2121b9d6a6b5f7aed5b592837…` | 434 | `fixtures/model-view-clip-label.ascii.c.txt` |
| `c558b0b8a8314e4935e77efce76eef…` | 1835 | `fixtures/model-closed-contours.mod` |
| `d09ea55fd1b2338c1c0156b3fae057…` | 725 | `fixtures/model-closed-contours.e.txt` |
| `edcab14154af4efc4771bcbdb8e319…` | 206 | `fixtures/model-closed-contours.r.txt` |
| `78b054f808883a5b176fc58d2b4bf2…` | 427 | `fixtures/model-closed-contours.c.txt` |
| `c5bfb5739a50a03206f68f281dfea1…` | 528 | `fixtures/model-empty-seed.mod` |
| `f040b72f31336923fd0cb7336ebabf…` | 2126 | `fixtures/model-view-clip-label.joined.mod` |
| `04dcd6934eb1e1713095dbb44e57a3…` | 8440 | `fixtures/make-model-view-clip-label.py` |

### `fixtures/newstack-reduce-input.mrc`

The vendored `.st` files carry no pixel data and the authored `in_*.mrc`
volumes are 64x48, a multiple of the binning factors most likely to be tried.
`getBinnedSize`'s X and Y offsets are only nonzero when the factor does *not*
divide the size, and those offsets are what `readBinnedOrReduced` starts from,
so a 21x13x3 volume is authored instead -- directly from the MRC layout in
`IMOD/libiimod/mrcfiles.h`, from an analytic pattern with no RNG, by
`fixtures/make-newstack-reduce-input.py`.  The `.mrc` files beside it are the
reference `newstack`'s own output for `-bin 3`, `-bin 3 -origin`, `-shrink 3`,
`-shrink 2.5` and `-shrink 3 -bin 2`, with the label block zeroed.

| SHA-256 | bytes | path |
|---|---:|---|
| `0247c4dc79dc8a82b193ca551f82c4…` | 2100 | `fixtures/make-newstack-reduce-input.py` |
| `7e5a1037a3d8ea166fbb17bedd54e3…` | 4300 | `fixtures/newstack-reduce-input.mrc` |
| `eb643f29869a3fbe2e383753a504a5…` | 1444 | `fixtures/newstack-reduce-bin3.mrc` |
| `eb643f29869a3fbe2e383753a504a5…` | 1444 | `fixtures/newstack-reduce-bin3origin.mrc` |
| `9dacf24920e20428ad246cc5281a76…` | 1444 | `fixtures/newstack-reduce-shrink3.mrc` |
| `9252c6278bd9f98e00a30979fad692…` | 1504 | `fixtures/newstack-reduce-shrink25.mrc` |
| `0c6322e1a2d245219880a07a8ebd42…` | 1120 | `fixtures/newstack-reduce-shrink3bin2.mrc` |

### `fixtures/newstack-mixed-*.mrc`

`newstack.f90` re-reads the header, recomputes `getReducedSize` and re-decides
rescaling once per *input file*, so a stack built from files that disagree is
what separates a per-file translation from one that takes those from the first
file only.  Three inputs are authored directly from the MRC layout by
`fixtures/make-newstack-mixed-inputs.py`: a 21x13x3 mode 0, a 21x13x3 mode 1
and an 11x7x3 mode 0.  The `.mrc` goldens beside them are the reference
`newstack`'s own output with the label block zeroed, and the `.txt` goldens its
stdout captured through a pipe.

| SHA-256 | bytes | path |
|---|---:|---|
| `5de9a5d90d7df57b7cbd256e77f5b3…` | 2466 | `fixtures/make-newstack-mixed-inputs.py` |
| `a6c7368ceb796a3b9455f2cdc4fe39…` | 1843 | `fixtures/newstack-mixed-byte.mrc` |
| `00eb9e9a8d5a48ae5cc0938928158b…` | 2662 | `fixtures/newstack-mixed-short.mrc` |
| `1e667bd842793d37397378bf0ac2c9…` | 1255 | `fixtures/newstack-mixed-small.mrc` |
| `b8623472e04f8ba9d99e6a6f421668…` | 2662 | `fixtures/newstack-mixed-byteshort.mrc` |
| `1ecdc37860130c4958876b77181e70…` | 2507 | `fixtures/newstack-mixed-byteshort.txt` |
| `cdf24afddb309439ed4851079b3f85…` | 4300 | `fixtures/newstack-mixed-shortbyte.mrc` |
| `b3a849cbcaa2e7437695532f08d544…` | 2507 | `fixtures/newstack-mixed-shortbyte.txt` |
| `dfb18199138147745f69f63df0948a…` | 2662 | `fixtures/newstack-mixed-bytesmall.mrc` |
| `ee4f5af0c8cf4c134ba743f096878a…` | 2579 | `fixtures/newstack-mixed-bytesmall.txt` |
| `8461fa369188cfbe364f45ed037d78…` | 1486 | `fixtures/newstack-mixed-smallbyte.mrc` |
| `6660e26327a653a31021c994603ecc…` | 2579 | `fixtures/newstack-mixed-smallbyte.txt` |
| `2f223c4eeb55a70fb83a5f9a126a51…` | 1486 | `fixtures/newstack-mixed-smallshortfloat2.mrc` |
| `fc027bc404341c4f2982dfd51a49d9…` | 2593 | `fixtures/newstack-mixed-smallshortfloat2.txt` |

### `fixtures/newstack-replace-*.mrc`

`-replace` writes into an existing output file and finishes with
`iiuWriteHeader(2, title, -1, ...)`, whose `labFlag = -1` changes no label at
all -- so with a target whose label block is already zeroed, every golden here
is byte-deterministic and needs no masking.  `newstack-replace-target.mrc` is
the reference `newstack`'s plain copy of `newstack-mixed-byte.mrc` with that
block zeroed; the rest are its own output for four `-replace` invocations, with
their stdout beside them.

| SHA-256 | bytes | path |
|---|---:|---|
| `438de7b0e0deaa2e828369414f5d31…` | 1843 | `fixtures/newstack-replace-target.mrc` |
| `3b954e92874a5109147c8ab1316222…` | 1843 | `fixtures/newstack-replace-sec1to0.mrc` |
| `21cf6a3b289bb8c4b7a5adb732e9f0…` | 2284 | `fixtures/newstack-replace-sec1to0.txt` |
| `5fe2fe02f9b9e887cb54bc663279c0…` | 1843 | `fixtures/newstack-replace-sec02to21.mrc` |
| `381366105db425cb2721fedec485a6…` | 2343 | `fixtures/newstack-replace-sec02to21.txt` |
| `532a5daea56218f1166aacbb783606…` | 1843 | `fixtures/newstack-replace-float2.mrc` |
| `3ceaa8b83cbb82fd7a567149bf040b…` | 2284 | `fixtures/newstack-replace-float2.txt` |
| `683ab42e46428df39787c3f8a7b021…` | 1843 | `fixtures/newstack-replace-scale.mrc` |
| `12e8f5105ddf6f40f94a663546ea88…` | 2343 | `fixtures/newstack-replace-scale.txt` |

### `fixtures/newstack-mdoc-*.mdoc`

Two authored `.mdoc` metadata files for `newstack -mdoc`, both describing the
`newstack-mixed-byte.mrc` image (the test copies it under each name).  The
first carries a global section, a `T` collection, a `MontSection` collection
and three `ZValue` sections -- `T` and `ZValue` are the two `transferCollections`
skips, so `MontSection` is what proves the collection loop runs and is indexed
correctly.  The goldens are the reference `newstack`'s own output `.mdoc` and
stdout; `.mdoc` files carry no timestamp, so they compare byte for byte.

| SHA-256 | bytes | path |
|---|---:|---|
| `f8bf53b285b254a6e4eabae23cf509…` | 447 | `fixtures/newstack-mdoc-byte.mrc.mdoc` |
| `ed23d6a9fa6c876fff59dc3548916f…` | 228 | `fixtures/newstack-mdoc-second.mrc.mdoc` |
| `0dd92659a9ecbea7d69f466f0c6e19…` | 415 | `fixtures/newstack-mdoc-plain.mdoc` |
| `6fa2eb5ad701ea75e24fa39bce7c27…` | 1330 | `fixtures/newstack-mdoc-plain.txt` |
| `4900e18b7f9ca3ed72e5342ce19438…` | 346 | `fixtures/newstack-mdoc-reorder.mdoc` |
| `0d081d9198b2196a9086316d4d8e1f…` | 1271 | `fixtures/newstack-mdoc-reorder.txt` |
| `34c0f5ebd0f218095f8536b6df055b…` | 556 | `fixtures/newstack-mdoc-twofiles.mdoc` |
| `381c05a828e98b107ea9259ae299c6…` | 2570 | `fixtures/newstack-mdoc-twofiles.txt` |
| `286c35490ff42dbb0ed3f2c2f642b3…` | 341 | `fixtures/newstack-mdoc-tilts.mdoc` |
| `aaef5a801cd3eb897a8fd6ed542278…` | 1311 | `fixtures/newstack-mdoc-tilts.txt` |

### `fixtures/newstack-warp.*`

`-distort` and `-gradient` replace `cubinterp` with `warpInterp`, so their
goldens have to be mode 2: rounding into a byte hides most of what the
interpolation does.  `fixtures/make-newstack-warp-inputs.py` authors a 64x48x3
float volume, a version-1 distortion field whose 5x4 grid does *not* divide
that size (so `getSizeAdjustedGrid` must expand and extrapolate it), and a
five-view mag-gradient table.  The goldens are the reference `newstack`'s own
output with the label block zeroed, and its stdout.

| SHA-256 | bytes | path |
|---|---:|---|
| `ee087ca285521a5ddeb8a930667294…` | 2648 | `fixtures/make-newstack-warp-inputs.py` |
| `5c51333022c2a7d1b7b6ece9180b49…` | 37888 | `fixtures/newstack-warp-input.mrc` |
| `8378e544d1cb91d4b7fed12c47b6c6…` | 327 | `fixtures/newstack-warp.idf` |
| `4f251cce174930164f3d316aa581b8…` | 104 | `fixtures/newstack-warp.mgt` |
| `83ea3c5ade6af2cc6ef401b50f0ffe…` | 37888 | `fixtures/newstack-warp-distort.mrc` |
| `c1e26ecb997fd5b4c0d1c6a6005fd7…` | 1242 | `fixtures/newstack-warp-distort.txt` |
| `eb3eaf76241faadf02780a1b610d0f…` | 37888 | `fixtures/newstack-warp-gradient.mrc` |
| `2e1c99388abe11cb7cf0b721602e3e…` | 1290 | `fixtures/newstack-warp-gradient.txt` |
| `5c84861e8c6202b64177cfb8363ea9…` | 37888 | `fixtures/newstack-warp-both.mrc` |
| `3ca8e61052f7fa0dff37bfe38de9bd…` | 1290 | `fixtures/newstack-warp-both.txt` |
| `83d53074eb759739972a559484a1a2…` | 37888 | `fixtures/newstack-warp-subarea.mrc` |
| `c93208f055271ff4cacda5a94c404b…` | 1242 | `fixtures/newstack-warp-subarea.txt` |
| `e3657483e73f4aafe6c0755623a2b1…` | 37888 | `fixtures/newstack-warp-linear.mrc` |
| `fdb9072a5b59c8638023c1d6d034e2…` | 1290 | `fixtures/newstack-warp-linear.txt` |
| `2433d933ff34d86050f0a98d6ff20c…` | 37888 | `fixtures/newstack-warp-nearest.mrc` |
| `a414054b5aeaf008249f8ae6798ba3…` | 1242 | `fixtures/newstack-warp-nearest.txt` |

`fixtures/newstack-warp.xf` is a version-3 **warping** file in the text format
`writeWarpFile` (`warpfiles.c:353`) emits: a header line, then per section a
grid-geometry line, the linear transform, and the displacement vectors four
pairs to a line.  Given to `-xform` it makes `newstack` warp *after* the linear
transform rather than undistort before it.

| SHA-256 | bytes | path |
|---|---:|---|
| `d49f9b4ee5342dc06730c89391e9cb…` | 1228 | `fixtures/newstack-warp.xf` |
| `2127852ee60c4db85d8ac1333533fe…` | 37888 | `fixtures/newstack-warp-xfwarp.mrc` |
| `5ecd43776671aebde29e7747791c6d…` | 1280 | `fixtures/newstack-warp-xfwarp.txt` |
| `a071b9160e149c7426dc91da2187e7…` | 37888 | `fixtures/newstack-warp-xfwarpoff.mrc` |
| `0a0e52cbf77b5b07a035fe5122df1d…` | 1280 | `fixtures/newstack-warp-xfwarpoff.txt` |
| `7f29e26b42bb314123422f9408940a…` | 10240 | `fixtures/newstack-warp-xfwarpshrink.mrc` |
| `6d98c2422a69551c670d0a7d3a5017…` | 1280 | `fixtures/newstack-warp-xfwarpshrink.txt` |

### `fixtures/newstack-warp-big.*`

The 64x48x3 volume above never chunks under `-memory`, whose smallest accepted
limit is 39 MB (`newstack.f90:1030-1036` rejects a `lenTemp` above half the
allocation, and `lenTemp` defaults to five million elements), so it cannot
exercise the chunked warping route at all.
`fixtures/make-newstack-warp-big-inputs.py` authors a **1200x900x3** float
volume -- 1,080,000 pixels a section -- together with an 8x7 distortion field,
the same five-view mag-gradient table, and a three-section version-3 warping
file, all scaled to that size and all with grids that do not divide it evenly.
`-test <total>,1` then splits the output two, three, four, eight, nine, ten,
sixteen and twenty-three ways.  Every pixel is a multiple of 0.25 in 60..240,
so it is exact in `float` and `tests/newstack_warp_chunked.rs` authors the
identical volume from the same expression without depending on either
runtime's libm -- which is why the 12.9 MB `.mrc` is **not** committed.

The goldens are the reference `newstack`'s own `-verbose 1` stdout, captured
through a pipe with its `loadtime`/`savetime`/`rottime` line replaced by
`loadtime MASKED` (that line is wall clock and differs between two *native*
runs), and, in `newstack-warp-big-outputs.txt`, the SHA-256 of its output file
with the label block (bytes 224..1024) zeroed -- the documented non-achievable,
since `mrc_head_new` never clears `labels` and the date stamp moves by a
second.  The digest stands in for a golden too large to commit.  The script
also takes an optional `nx ny nz`, used only to author the much larger
scratch-only volumes (2600x2000x2 and 3400x2600x2) that `-memory 39` needs
before it will chunk; those are not committed either.

| SHA-256 | bytes | path |
|---|---:|---|
| `b34eb443e7c98550871cfd3627404e…` | 4568 | `fixtures/make-newstack-warp-big-inputs.py` |
| `a7337e9a2aeff0090292c25be98bb3…` | 878 | `fixtures/newstack-warp-big.idf` |
| `4f251cce174930164f3d316aa581b8…` | 104 | `fixtures/newstack-warp-big.mgt` |
| `ede78a5c9956e0a90c72a32a708ef8…` | 2923 | `fixtures/newstack-warp-big.xf` |
| `91f791106e421d63573eec46e1b291…` | 212 | `fixtures/newstack-warp-big-sparse.idf` |
| `9888316cbe6ba7801e55f38673e618…` | 746 | `fixtures/newstack-warp-big-outputs.txt` |
| `8a3ea79d345c5c85bee5d640df6c51…` | 3198 | `fixtures/newstack-warp-big-distort2.txt` |
| `fe268f884e6ee143ec855046108385…` | 3915 | `fixtures/newstack-warp-big-distort3.txt` |
| `f7cb83686b48741068e881076f2809…` | 7434 | `fixtures/newstack-warp-big-distort8.txt` |
| `96a29fc3831a43a476adbb5cfb6f61…` | 7110 | `fixtures/newstack-warp-big-distort10.txt` |
| `00ce9fcb6090ef30cb6676c0c53b20…` | 4125 | `fixtures/newstack-warp-big-gradient3.txt` |
| `ab8a92ee88915cbbd710b0adebc906…` | 12822 | `fixtures/newstack-warp-big-gradient16.txt` |
| `7a49cb3299d9e0f2f6de939d89b2fd…` | 3224 | `fixtures/newstack-warp-big-xfwarp2.txt` |
| `af735218aba8fa06fa30895609053d…` | 4115 | `fixtures/newstack-warp-big-xfwarp3.txt` |
| `40982bcc81d5acadf8444eb2b11ce4…` | 8129 | `fixtures/newstack-warp-big-xfwarp9.txt` |
| `207cac5a4dd0ecadf8a3c3d4cd0731…` | 7434 | `fixtures/newstack-warp-big-distortsparse.txt` |

`fixtures/newstack-warp-big-sparse.idf` is a deliberately *under-covering* 4x3
grid: it spans 0..510 of the 1200 columns and 0..280 of the 900 rows, so
`getSizeAdjustedGrid` has to run `expandAndExtrapGrid` (`warputils.c:707`) over
most of the image instead of using the grid as it stands.  It is the committed
regression guard for the `float`-versus-`double` widths inside
`extrapolateGrid` (`warputils.c:436, 489-495, 511-516, 583`); before they were
fixed this case differed from the reference by 22113 bytes.

### `fixtures/usage-*.txt`

Each command's usage text, captured from the **reference** binary through a
pipe with its first line (the one carrying the version and build stamp)
removed.  `tests/usage_text.rs` compares the crate's own output against them.
`mrc2tif`'s text had been paraphrased and printed with `println!`, which also
moved `imodCopyright`'s banner to the end of the output under a pipe.

`mrc2tif-float-scaled.png` and `.jpg` are the reference `mrc2tif -p`/`-j`
output for section 0 of the float `mrcsec-mode2.mrc`.  `mrc2tif.cpp:356-358`
sets `convert` for a non-colour PNG/JPEG *before* `mrcContrastScaling` runs at
line 376; doing it afterwards leaves a float image with scale 1 and offset 0
and it reaches the writer unscaled.

| SHA-256 | bytes | path |
|---|---:|---|
| `c1af0448aa5c78bb6a0528891825df…` | 1419 | `fixtures/usage-mrc2tif.txt` |
| `de2aa3e14ba132f6eebed411c6655b…` | 1114 | `fixtures/usage-tif2mrc.txt` |
| `6ca65cb0f71ef1cd9cb4f5f32d08ba…` | 4413 | `fixtures/usage-clip.txt` |
| `aae5b453b759f10de99f85187d8d74…` | 306 | `fixtures/mrc2tif-float-scaled.png` |
| `fd2ac0d281c29fe1283e0bcfb709ef…` | 440 | `fixtures/mrc2tif-float-scaled.jpg` |

### `fixtures/mrcsec-mode*.mrc`

One 21x13x4 volume per storage mode `mrcsec.c` converts between: 0 (byte), 1
(signed short), 2 (float), 4 (complex float) and 6 (unsigned short), authored
from the MRC layout by `fixtures/make-mrcsec-inputs.py` out of an analytic
pattern with no RNG.  The size is odd on purpose so the sub-rectangle reads in
`tests/mrcsec_sections.rs` are unaligned.

`fixtures/mrcsec-driver.c` is a C program linked against the reference
`libiimod` that calls the same fifteen `mrcsec` entry points in the same order
and prints a rolling hash of each result; `mrcsec-mode*.out.txt` and
`mrcsec-mode*.err.txt` are its output.  Rebuild it with

```
gcc -fopenmp -o cdrv_mrcsec fixtures/mrcsec-driver.c \
    -I/tmp/imod-reference-build/include -L/tmp/imod-reference-build/buildlib \
    -liimod -lcfshr -limxml -ltiff -lm -lgomp
```

| SHA-256 | bytes | path |
|---|---:|---|
| `50b3873be4ed9f084bba2158576cb2…` | 3117 | `fixtures/mrcsec-driver.c` |
| `8b6b73cd794e370fcc92a6a66a42ef…` | 261 | `fixtures/mrcsec-mode0.err.txt` |
| `2c55d5e69f55282948489746f45086…` | 2116 | `fixtures/mrcsec-mode0.mrc` |
| `4146c397149852b379267b39f87d8e…` | 915 | `fixtures/mrcsec-mode0.out.txt` |
| `8b6b73cd794e370fcc92a6a66a42ef…` | 261 | `fixtures/mrcsec-mode1.err.txt` |
| `6ee94d3a4150c72b173b5e76dd34bc…` | 3208 | `fixtures/mrcsec-mode1.mrc` |
| `7a3df2b6a15aa48b67232673dca8b3…` | 905 | `fixtures/mrcsec-mode1.out.txt` |
| `8b6b73cd794e370fcc92a6a66a42ef…` | 261 | `fixtures/mrcsec-mode2.err.txt` |
| `5f9c7cd9c7eff4912c0535deb5bd7d…` | 5392 | `fixtures/mrcsec-mode2.mrc` |
| `dd91c720c73b243ab530291f368583…` | 914 | `fixtures/mrcsec-mode2.out.txt` |
| `f1a655a15eaea04f1b6dcab5a03f5a…` | 586 | `fixtures/mrcsec-mode4.err.txt` |
| `a3a2b8c484df9e83bc6c3f4c1af152…` | 9760 | `fixtures/mrcsec-mode4.mrc` |
| `bb3de32975f90e0510dc77d4868908…` | 818 | `fixtures/mrcsec-mode4.out.txt` |
| `8b6b73cd794e370fcc92a6a66a42ef…` | 261 | `fixtures/mrcsec-mode6.err.txt` |
| `6585c0b65b6c97ffc23e7191e8e852…` | 3208 | `fixtures/mrcsec-mode6.mrc` |
| `3bf8a1460678f97cc90e9ca8d4e592…` | 914 | `fixtures/mrcsec-mode6.out.txt` |

### HDF fixtures authored at run time

`tests/newstack_hdf_volumes.rs` needs a 3-D-volume HDF file and a two-volume
HDF file, neither of which exists in the pinned tree and neither of which is
worth committing: the crate builds both in the process temp directory with its
own `ii_open_new`/`ii_hdf_open_new`/`ii_write_section_float`, the same calls
`iiFOpenNewVolume` makes, and deletes them afterwards.  Nothing lands in
`fixtures/`.  The expectations in that suite — the `Actual chunk size:` line,
the `-chunk`/`-3d -1` exit message, `iiFOpenVolume`'s out-of-range text — were
each read off the HDF-enabled native reference first, with the equivalent
inputs authored by the reference's own `raw2mrc` and `newstack`.

### Multi-page TIFF fixtures authored at run time

`tests/newstack_stream.rs::newstack_tiff_stack_repeats_description_and_min_max_on_every_directory`
and
`tests/mrc2tif_cli.rs::mrc2tif_stack_writes_running_min_max_on_every_directory_and_no_description`
each need a small multi-section MRC whose header range is deliberately wider
than its data, so that the per-directory `SMinSampleValue`/`SMaxSampleValue` a
TIFF stack carries can be told apart page by page.  Both build it in the
process temp directory with the crate's own `mrc_head_new`/`mrc_head_write`
plus a plain `fwrite` of the pixels and delete it afterwards; nothing lands in
`fixtures/`.  The expectations in both suites were read off the HDF-enabled
native reference first, running it on the very file the test builds: for
`newstack` the first two directories carry the input header's `-5`/`250` and
the last carries the written range `0`/`35`; for `mrc2tif -s` each directory
carries its own section's range and the last carries the whole-stack range.

### Chunked-scaling fixtures authored at run time

`tests/newstack_stream.rs::newstack_memory_limit_keeps_one_chunk_where_the_source_does`
and
`tests/newstack_stream.rs::newstack_float_two_and_three_scale_under_a_memory_limit`
each need an image big enough that a `-memory`/`-test` limit forces the source
to split the section into several chunks; no fixture in the pinned tree is
anywhere near that size, and a committed one would be tens of megabytes.  Both
build a 1620x1620 image (the second one two sections of mode 2, with a
deterministic sine/cosine pattern) in the process temp directory by writing the
1024-byte MRC header and the pixels directly, and delete it afterwards; nothing
lands in `fixtures/`.  The second test's expectations were taken against the
HDF-enabled native reference on the very file it builds, driven by
`IMOD_NATIVE_NEWSTACK`: with that set it compares whole output files with only
the 800-byte label area masked, for `-float 2` and `-float 3` at no limit and
at `-test 3000000,1000`, `-test 700000,1000` and `-test 300000,1000`.  The
first two of those keep the whole output in memory (`ifOutChunk` 0) and the
last two send every chunk but the last to the scratch file on unit 3, which is
what the ` SCRATCH image file on unit   3 : ` assertion pins.

### Typed extended-header fixtures authored at run time

`tests/newstack_stream.rs`'s six extended-header tests
(`newstack_copies_serialem_typed_extended_header_for_selected_sections`,
`newstack_strip_keeps_serialem_type_fields_with_no_extended_data`,
`newstack_tilt_replaces_serialem_tilt_short_and_keeps_the_rest`,
`newstack_reorder_uses_extended_header_tilt_angles`,
`newstack_refuses_saving_tilt_angles_into_an_fei1_extended_header` and
`newstack_refuses_tilt_angles_for_a_serialem_header_without_the_tilt_flag`)
each build their own small MRC in the process temp directory with the crate's
`mrc_head_new` / `mrc_write_extra_header` / `mrc_head_write` and delete it
afterwards; nothing lands in `fixtures/`.  The three header shapes are:

- SerialEM: `nint` = bytes per section, `nreal` = flag bits (1 tilt angle
  short, 2 three piece-coordinate shorts, 4 two stage-position shorts), so
  `nint = 12, nreal = 7` is the full set and `nint = 10, nreal = 6` is a valid
  pair with no tilt angle;
- Agard/old FEI: `nint = 0`, `nreal` = number of reals per section, the first
  of them the tilt angle;
- FEI1: `extType = "FEI1"` with a non-zero `nversion`, each section's record
  starting with its own 4-byte length (`extraheader.c:937-950`).

The same three shapes, plus an `extType = "ZZ99"` unknown type, were authored
as scratch MRC files with `raw2mrc` plus a Python header patch and compared
byte for byte against the reference `newstack` for `-secs`, `-strip`, `-tilt`,
`-reorder` with and without `-angle`, `-mdoc`, `-numout` and multi-input runs.

## Fields that must be masked before byte comparison

Recorded here because comparing without masking produces false regressions, and
did so three times in one session:

- the MRC label date stamp `dd-Mmm-yy  HH:MM:SS` — two runs straddling a second
  boundary differ by one byte;
- in an HDF file, the 4-byte little-endian Unix time in each dataset's object
  header (HDF5 object-header message type `0x0012`), one per dataset, for the
  same reason; two runs inside the same second are byte-identical without it,
  so prefer that to masking.  `<scratchpad>/diff/hcmp.py` compares two HDF
  files semantically (object set, shapes, dtypes, chunking, compression,
  attributes, raw data) with the MRC label stamp masked;
- TIFF `DateTime` tags, one per IFD, same reason;
- MRC label slots past `nlabl` — `mrc_head_new` never clears them and callers
  put the header on the stack, so the reference writes stack garbage there;
  `clip blankfile` produces three different SHA-256s across three identical
  runs;
- for `newstack` copying an **FEI1** extended header, the bytes of the output
  extended header past what `copyExtraHeaderSection` wrote — `extraOut` is a
  bare Fortran `allocate` (`newstack.f90:1926`) and the copy writes only each
  section's own record size, so when the sections differ in size the reference
  leaves uninitialised memory in the `getExtraHeaderMaxSecSize` padding; three
  identical runs gave three different values there.  The translation writes
  zeros;
- `__DATE__`/`__TIME__` in version banners — compilation metadata;
- for `imodjoin`, the `MINX` chunk's `oscale` and `orot` fields, from the
  `IrefImage` that `imodjoin.c:181` `malloc`s and only partly fills; these vary
  run to run in the reference itself;
- for any model the reference wrote after reading an ASCII model, the
  `name[128]` bytes past the 13 `imodDefault` writes — `imodNew` `malloc`s the
  model and `imodel_write` emits the whole array, so those bytes are heap
  residue.

## Comparison procedure

Each side runs in its own directory, with stdout through a **pipe** — never a
file redirect, because gfortran block-buffers unit 6 under a redirect and the
*reference's own* output reorders.  Exit status, stdout bytes, stderr bytes and
output-file bytes are all compared.
