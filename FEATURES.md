# IMOD Rust Rewrite — Status

**Current totals: 11 crates, 126 apps, ~54K lines Rust+Slint+WGSL, 148 test suites, zero failures.**

## Library Crates (11)

| Crate | Features |
|-------|----------|
| `imod-core` | MrcMode (all 9 modes), ExtHeaderType, PixelUnit, Point3f, ImodError |
| `imod-mrc` | MRC read/write, byte-swap detection, sub-area reading, complex/half-float modes, extended header parsing (SERI+FEI), Y-axis reading, old-style MRC support |
| `imod-autodoc` | .adoc PIP parameter spec parser |
| `imod-math` | Stats, robust stats, regression (multiple/robust/polynomial), gaussj, amoeba (Nelder-Mead), circlefit, parselist, cubinterp |
| `imod-transforms` | LinearTransform, .xf/.xg/.tlt I/O, anglesToMatrix, matrixToAngles, findTransform, 3D rotation |
| `imod-fft` | 1D/2D FFT (real↔complex), power spectrum, cross-correlation |
| `imod-model` | Full chunk-based read/write: OBJT, CONT, MESH, VIEW, IMAT, CLIP, MCLP, SLAN, SIZE, MINX, MOST/OBST/COST/MEST, unknown chunk preservation |
| `imod-slice` | Slice type, bilinear/cubic interpolation, filtering (sobel, blur, median, aniso diffusion), morphological ops, binning |
| `imod-image-io` | ImageFile trait, MRC + TIFF backends, format detection (MRC/TIFF/HDF5/EER) |
| `imod-warp` | Delaunay triangulation, warp file I/O, natural neighbor interpolation |
| `imod-mesh` | Contour skinning, marching cubes isosurface, Douglas-Peucker simplification |

## GUI Applications (4)

| App | Features |
|-----|----------|
| `imod-viewer` | ZAP/Slicer/XYZ/3D/Isosurface views, model overlay, contour editing (add/delete), undo/redo (50 states), keyboard shortcuts, file dialogs, software 3D renderer with Z-buffer + lighting |
| `imod-studio` | 9-step reconstruction workflow using library calls, bead tracking, fine alignment, reconstruction, post-processing, log output |
| `imod-aligner` | Reference/current section overlay, shift/rotation/mag sliders, auto-align (FFT CC), save .xf |
| `imod-ctfplot` | Power spectrum display, radial average + CTF overlay, defocus fitting, per-section defocus, GPU support |

## CLI Tools — Reconstruction Pipeline (15)

| Tool | Key Features |
|------|-------------|
| `tiltxcorr` | FFT CC alignment, bandpass filtering, cumulative mode, patch-based correlation |
| `beadtrack` | Tilt-compensated search, sub-pixel refinement, gap filling, residual rejection, trajectory fitting, adaptive template, elongation filtering |
| `tiltalign` | Full projection model, robust fitting (Tukey bisquare), surface fitting (1/2 surfaces), beam tilt, group variables, local alignment output, leave-one-out, fixed view constraints |
| `tilt` | R-weighting filter, SIRT, cosine stretch, local alignment, X-axis tilt, Z-factor, GPU (wgpu), log scaling, density weighting, Hamming filter, edge fill modes |
| `raptor` | Automatic fiducial detection (NCC template matching), RANSAC correspondence, trajectory building, 3D estimation, bundle adjustment, contour refinement |
| `newstack` | Antialias (Lanczos-2), float modes, rotation (bicubic), expand/shrink, subarea, piece lists, distortion, taper, warp transforms, phase shift, Fourier reduce, transpose, gradient, multi-file |
| `ctfphaseflip` | Defocus file, 2D strip processing, astigmatism, zero-crossing detection, auto strip width, plate phase, amplitude contrast, cuton frequency, GPU (wgpu) |
| `blendmont` | Edge cross-correlation refinement, linear weight blending, multi-Z |
| `alignframes` | Gain reference correction, dose weighting (Fourier), aligned stack output, iterative alignment |
| `ccderaser` | Grown pixel flood-fill, diffraction spike detection, model-based erasure, edge exclusion |
| `findwarp` | 3D affine from matched fiducials |
| `warpvol` | 3D affine transform with trilinear interpolation |
| `corrsearch3d` | 3D patch cross-correlation |
| `solvematch` | 3D matching from correspondences |
| `trimvol` | Subvolume extraction, Y/Z rotation, float-to-byte |

## CLI Tools — Image Utilities (20+)

mrcinfo, mrcbyte, mrc2tif, binvol, rotatevol, mtffilter, eraser, clip, fftrans, avgstack, header, alterheader, extracttilts, extractpieces, xftoxg, xcorrstack, checkxforms, xfsimplex, xfinverse, xfproduct, densmatch, assemblevol, taperoutvol, squeezevol, xyzproj, extstack, tomopieces, montagesize, findcontrast, densnorm, fixmont, goodframe, edgemtf, edpiecepoint, extposition, extractmagrad, fixboundaries, maxjoinsize, numericdiff, rotmont, calc, reducestack, matchvol, combinefft, addtostack, subimage, excise, taperprep, tapervoledge, enhancecontrast, preNAD, nad_eed_3d, stitchalign, subimanova, subimstat, avganova, rotmatwarp, mtdetect, stitchvars

## CLI Tools — Model Utilities (25+)

imodinfo, imodtrans, imodmesh, imodjoin, point2model, imodextract, imodfillin, imodsortsurf, imodsetvalues, clonemodel, clipmodel, smoothsurf, findbeads3d, model2point, remapmodel, patch2imod, imod2patch, imodchopconts, imodcurvature, imodexplode, imodauto, pickbestseed, rec2imod, findsection, flattenwarp, joinwarp2model, clonevolume, imodmop

## CLI Tools — Format Converters (15+)

imod2obj, imod2vrml, imod2vrml2, imod2meta, imod2nff, imod2rib, imod2synu, imod2ccdbxml, wmod2imod, imod2wmod, adocxmlconv, holefinder, imodholefinder, slashfindspheres, slashmasksort, nogputxc

## CLI Tools — Spatial Analysis (3)

nda, sda, mtk

---

## Not Yet Ported

### High-Value Missing (~10 programs)

| Program | Category | Purpose |
|---------|----------|---------|
| `refinematch` | flib/model | Refine 3D volume matching for dual-axis combination |
| `tomopitch` | flib/model | Analyze section boundaries for positioning |
| `filltomo` | flib/model | Fill in missing wedge regions |
| `repackseed` | flib/model | Repackage seed model for Transferfid |
| `sortbeadsurfs` | flib/model | Sort beads onto surfaces |
| `framealign` (C++) | mrc | Movie frame alignment (more features than our alignframes) |
| `tif2mrc` | mrc | TIFF to MRC conversion |
| `raw2mrc` | mrc | Raw data to MRC conversion |
| `mrctaper` | mrc | Taper MRC file edges |
| `processchunks` | standalone | Parallel job distribution (critical for large datasets) |

### Medium-Value Missing (~40 programs)

**flib/model (~25):** boxavg, boxstartend, checkmtmod, contourmod, convertmod, edgeeraser, edmont, endmodel, fenestra, fiberpitch, findhotpixels, get_region_contours, howflared, imavgstat, joinmodel, mtlengths, mtmodel, mtrotlong, mtsmooth, planefit, realscalemod, reducecont, reducemtmod, resamplemod, scalemodel, solve_wo_outliers, sumdensity, xfinterstack, xfjointomo, xfmodel

**mrc (~15):** mrcx, mrclog, mrctilt, fakevolume, frameutil, measuredrift, modifymdoc, preNID, recline, tiff, tifinfo, dm3props

### Low-Value / Obsolete (~20)

- `vmstocsh` — VMS to csh converter (obsolete)
- `echo2`, `imodwincpu`, `imod-dist`, `recfile` — platform/packaging utilities
- `mtsubs`, `rotmatwarpsubs` — subroutine libraries (not standalone programs)
- `ShrMemClient`, `manageshrmem`, `shrmemframe` — shared memory utilities
- `nogpuctf`, `nogpuframe`, `nrutil` — GPU stubs / numerical recipes

### Scripts Not Ported (~12)

- `prochunks.csh`, `slurmCleanup.sh`, `slurmInit.sh` — job management
- `autodoc.py`, `comchanger.py`, `imodpy.py`, `pip.py`, `prochunks.py`, `pysed.py`, `supermont.py`, `tiltmatch.py`, `tomocoords.py` — Python workflow scripts

### Not Ported By Design

| Component | Original LOC | Replacement |
|-----------|-------------|-------------|
| 3dmod (Qt/OpenGL) | 105K | imod-viewer (Slint) |
| midas (Qt) | 8.5K | imod-aligner (Slint) |
| ctfplotter (Qt) | 15K | imod-ctfplot (Slint) |
| Etomo (Java Swing) | 346K | imod-studio (Slint) |
| Plugins (C++) | 63K | Not ported (plugin system not yet implemented) |

---

## Build Configuration

- `RUSTFLAGS="-C target-cpu=native"` via `.cargo/config.toml`
- GPU acceleration via wgpu compute shaders (tilt + ctfphaseflip)
- Parallel processing via rayon (tilt backprojection, bead detection)
- All crates documented with `cargo doc`
