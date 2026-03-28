# IMOD Rust Rewrite — Progress Tracker

## Phase 0: Scaffolding (Months 1-2)

- [x] Set up Cargo workspace (`Cargo.toml`)
- [x] `imod-core` — shared types (MrcMode, ExtHeaderType, PixelUnit, Point3f), error handling
- [x] `imod-mrc` — MRC file format read/write with `binrw`
  - [x] MRC header parsing (1024-byte header, all modes)
  - [x] Extended header type detection (FEI, SerialEM, Agard)
  - [x] Byte-swapped file reading (auto-detect big-endian)
  - [x] Round-trip tests (header, byte mode, float mode)
  - [x] MrcReader: read slices as raw bytes or f32 (byte, short, ushort, float, RGB, 4-bit)
  - [x] MrcWriter: write slices from f32 with statistics finalization
- [x] `imod-autodoc` — PIP parameter spec parsing (tested against real newstack.adoc)
- [ ] Basic CI setup

## Phase 1: Core Libraries (Months 2-5)

- [x] `imod-math` — mean, SD, min/max, robust stats (median/MADN), sampling, linear regression
- [x] `imod-transforms` — LinearTransform (multiply, invert, apply, rotate, scale), .xf/.xg I/O, .tlt I/O
- [x] `imod-fft` — 1D/2D FFT (real↔complex), power spectrum, cross-correlation (via rustfft)
- [x] `imod-slice` — Slice type, bilinear interp, scale/clamp/threshold/invert, add/subtract/multiply, convolve, sobel, blur, median, bin, subregion
- [x] `imod-image-io` — ImageFile trait, MRC backend, format detection (.mrc/.st/.ali/.rec/.tif/.hdf)
- [x] `imod-model` — IMOD model read/write (chunk-based binary: IMOD/OBJT/CONT/MESH/IEOF)
- [x] `imod-warp` — Bowyer-Watson Delaunay triangulation, warp file read/write, point-in-triangle search
- [x] `imod-mesh` — contour skinning (triangle strip between Z sections), Douglas-Peucker simplification

## Phase 2: Command-Line Tools (Months 5-9)

- [x] `newstack` — section selection, binning, transform application, mode conversion, scaling
- [x] `clip` — stats, flip, multiply, add, resize, median filter, gradient (Sobel)
- [x] `mrcinfo` — header display (dimensions, mode, pixel size, labels, IMOD stamp)
- [x] `mrcbyte` — convert to byte mode with auto/manual/contrast scaling
- [x] `tiltxcorr` — cross-correlation alignment of tilt series (FFT-based, reference section, exclusion)
- [x] `ccderaser` — automatic X-ray/hot pixel detection and replacement
- [x] `ctfphaseflip` — CTF phase-flip correction (strip-based, tilt-dependent defocus, relativistic wavelength)
- [x] `alignframes` — iterative frame alignment by cross-correlation, aligned sum output, optional .xf output
- [x] `binvol` — 3D volume binning (independent XY and Z factors)
- [x] `trimvol` — subvolume extraction, Y/Z rotation, float-to-byte
- [x] `rotatevol` — Z-axis rotation with bilinear interpolation
- [x] `mtffilter` — Fourier-space low-pass/high-pass filtering with Gaussian falloff
- [x] `eraser` — gold bead eraser using IMOD model positions, ring interpolation

## Phase 3: Reconstruction Pipeline (Months 8-13)

- [x] `tiltalign` — fiducial-based alignment solver (iterative, per-view shifts, rotation, magnification)
- [x] `tilt` — weighted back-projection reconstruction (bilinear interpolation, rayon-ready)
- [x] `beadtrack` — template-matching bead tracker (cross-correlation, bidirectional tracking)
- [x] `blendmont` — montage blending (piece list, linear edge weighting, multi-Z)
- [x] `findwarp` — 3D affine transform from matched fiducials (least-squares, residuals)
- [x] `warpvol` — apply 3D affine transform with trilinear interpolation
- [x] `corrsearch3d` — 3D patch correlation between two volumes (FFT-based)
- [x] `solvematch` — solve 3D matching transform from correspondence points

## Phase 4: imod-viewer — 3dmod Replacement (Months 10-16)

- [x] Image display engine (ZAP window) — Slint UI, slice viewing, Z navigation, contrast (black/white), scroll-zoom
- [x] Model overlay rendering — contour lines (Bresenham), point crosses, per-object colors, toggle on/off
- [x] Slicer — arbitrary oblique plane through volume, X/Y angle sliders, trilinear interpolation
- [x] XYZ multi-axis view — simultaneous XY, XZ, YZ planes in one image
- [x] Object/contour management panel — sidebar listing objects with colors and contour counts
- [x] Pixel value readout — mouse position tracking with coordinate and value display
- [x] Volume caching — full volume loaded on demand for slicer/XYZ modes
- [x] View mode switching — toolbar buttons for ZAP/Slicer/XYZ
- [x] Command-line model loading — pass .mod/.fid files alongside image
- [ ] Model view (3D rendering with lighting, depth cue) — requires wgpu
- [ ] Isosurface rendering — requires wgpu
- [ ] Contour editing tools
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
- [x] Phase 0 complete: 3 crates, 13 tests
- [x] Phase 1 complete: 11 crates, 68 tests passing
- [x] Phase 2 complete: 13 CLI tools
- [x] Phase 3 complete: 8 reconstruction pipeline tools
- [x] Phase 4 mostly complete: imod-viewer with ZAP/Slicer/XYZ views, model overlay, pixel readout (3D rendering deferred to wgpu)
