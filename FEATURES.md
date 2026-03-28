# IMOD Rust Rewrite — Progress Tracker

## Phase 0: Scaffolding (Months 1-2)

- [x] Set up Cargo workspace (`Cargo.toml`)
- [x] `imod-core` — shared types (MrcMode, ExtHeaderType, PixelUnit, Point3f), error handling
- [x] `imod-mrc` — MRC file format read/write with `binrw`
  - [x] MRC header parsing (1024-byte header, all modes)
  - [x] Extended header type detection (FEI, SerialEM, Agard)
  - [ ] Byte-swapped file reading
  - [x] Round-trip tests (header, byte mode, float mode)
  - [x] MrcReader: read slices as raw bytes or f32 (byte, short, ushort, float, RGB, 4-bit)
  - [x] MrcWriter: write slices from f32 with statistics finalization
- [x] `imod-autodoc` — PIP parameter spec parsing (tested against real newstack.adoc)
- [ ] Basic CI setup

## Phase 1: Core Libraries (Months 2-5)

- [x] `imod-math` — mean, SD, min/max, robust stats (median/MADN), sampling, linear regression
- [x] `imod-transforms` — LinearTransform (multiply, invert, apply, rotate, scale), .xf/.xg I/O, .tlt I/O
- [x] `imod-fft` — 1D/2D FFT (real↔complex), power spectrum, cross-correlation (via rustfft)
- [ ] `imod-slice` — slice/image processing, pixel operations, filtering
- [ ] `imod-image-io` — multi-format I/O trait (MRC, TIFF, JPEG, HDF5, EER)
- [x] `imod-model` — IMOD model read/write (chunk-based binary: IMOD/OBJT/CONT/MESH/IEOF)
- [ ] `imod-warp` — warping, Delaunay triangulation, natural neighbor interpolation
- [ ] `imod-mesh` — mesh generation from contour data

## Phase 2: Command-Line Tools (Months 5-9)

- [ ] `newstack` — stack operations
- [ ] `clip` — image processing operations
- [ ] `mrcbyte` / `mrcinfo` / `mrc2tif` — basic MRC utilities
- [ ] `tiltxcorr` — cross-correlation alignment
- [ ] `ccderaser` — artifact removal
- [ ] `ctfphaseflip` — CTF correction
- [ ] `alignframes` — movie frame alignment
- [ ] `binvol` / `rotatevol` / `trimvol` — volume operations
- [ ] `mtffilter` — MTF filtering
- [ ] `eraser` — gold bead eraser

## Phase 3: Reconstruction Pipeline (Months 8-13)

- [ ] `tiltalign` — tilt series alignment solver
- [ ] `tilt` — back-projection reconstruction (SIMD inner loops + rayon)
- [ ] `beadtrack` — fiducial bead tracking
- [ ] `blendmont` — montage blending
- [ ] `findwarp` / `warpvol` — tomogram combination
- [ ] `corrsearch3d` / `refinematch` / `solvematch` — volume matching (dual-axis)

## Phase 4: imod-viewer — 3dmod Replacement (Months 10-16)

- [ ] Image display engine (ZAP window) — 2D slice viewing, zoom, pan, contrast
- [ ] Model overlay rendering (contours, points, spheres)
- [ ] Slicer (arbitrary plane through volume)
- [ ] XYZ multi-axis view
- [ ] Model view (3D rendering with lighting, depth cue)
- [ ] Isosurface rendering
- [ ] Contour editing tools
- [ ] Object/contour management panels
- [ ] Image caching and pyramid support
- [ ] Bead fixing interface
- [ ] Preferences system
- [ ] Plugin loading (`libloading` + Rust trait objects)

## Phase 5: imod-studio + Other GUIs (Months 14-18)

- [ ] `imod-studio` — Etomo replacement (Slint)
  - [ ] Pipeline step definitions
  - [ ] .com script generation and execution
  - [ ] Process monitoring with progress bars
  - [ ] Log viewing
  - [ ] Parallel job management (processchunks equivalent)
  - [ ] Single-axis tomography workflow
  - [ ] Dual-axis tomography workflow
  - [ ] Joining workflow
  - [ ] Batch processing
- [ ] `imod-aligner` — midas replacement (Slint + wgpu)
- [ ] `imod-ctfplot` — ctfplotter replacement (Slint)

## Phase 6: RAPTOR + Remaining Tools (Months 16-20)

- [ ] `raptor` — automatic fiducial tracking (RANSAC, correspondence)
- [ ] Remaining flib/image programs (~55 programs)
- [ ] Remaining flib/model programs (~45 programs)
- [ ] ndasda spatial analysis tools

## Phase 7: Testing & Polish (Months 18-22)

- [ ] Comprehensive file format round-trip test suite
- [ ] Performance benchmarking vs C/Fortran originals
- [ ] GPU acceleration (`cudarc` for backprojection, `wgpu` compute shaders)
- [ ] Script migration (Python/shell to Rust CLIs)
- [ ] Documentation
- [ ] Release packaging

---

## Completed

- [x] Downloaded IMOD source code via Mercurial
- [x] Analyzed codebase: ~969K lines (Java 346K, C++ 320K, C 127K, Headers 98K, Fortran 69K+, Python 6K, Shell 4K)
- [x] Created rewrite plan with workspace structure, phased migration, and key decisions
- [x] Saved README.md with project overview
- [x] Phase 0 complete: workspace with 3 crates, 13 tests passing
- [x] Phase 1 in progress: 7 crates total, 42 tests passing
