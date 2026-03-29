# IMOD Rust Rewrite — Remaining Work

## Current State
- 11 crates, 126 apps, ~56K lines Rust+Slint+WGSL
- 148 test suites, zero failures
- Native menu bar matching 3dmod, wgpu 3D rendering, split-panel layout

## Priority 1: Viewer Menu Stubs (44 items)

See FEATURES.md for the full checklist. Key items:

### Quick wins (< 30 lines each)
- `menu-file-new-model` — `model = Some(ImodModel::default())`
- `menu-file-quit` — `std::process::exit(0)`
- `menu-file-save-as` — file dialog + write_model
- `menu-edit-object-new` — push new ImodObject to model
- `menu-edit-object-delete` — remove current object
- `menu-edit-contour-new` — push new ImodContour
- `menu-edit-contour-delete` — remove current contour
- `menu-edit-point-delete` — remove current point
- `menu-edit-model-clean` — remove empty objects/contours
- `menu-edit-image-fillcache` — call ensure_volume_loaded()
- `menu-file-snapshot` — render current view to PNG file

### Medium effort (50-200 lines each)
- `menu-edit-object-color` — color picker (Slint dialog or simple RGB input)
- `menu-edit-contour-break` — split contour at current point
- `menu-edit-contour-join` — merge two contours
- `menu-edit-point-distance` — compute distance between current and previous point
- `menu-edit-image-flip` — flip/rotate loaded volume
- `menu-file-write-obj` — call imod2obj logic as library function
- `menu-file-write-vrml` — call imod2vrml logic
- `menu-image-graph` — render 1D intensity profile to Image

### Larger effort (200+ lines)
- `menu-special-beadfixer` — interactive bead editing panel
- `menu-special-interpolator` — contour interpolation between Z sections
- `menu-special-drawing` — drawing tools (shapes, measurements)
- `menu-edit-options` — preferences dialog with persistent settings
- `menu-image-pixel-view` — pixel value table view
- `menu-image-tumbler` — animated Z-slice playback

## Priority 2: High-Value Missing Programs (~10)

| Program | Source | Purpose |
|---------|--------|---------|
| refinematch | flib/model | Refine 3D volume matching (dual-axis) |
| tomopitch | flib/model | Analyze section boundaries |
| filltomo | flib/model | Fill missing wedge regions |
| repackseed | flib/model | Repackage seed model for Transferfid |
| sortbeadsurfs | flib/model | Sort beads onto surfaces |
| framealign (C++) | mrc | Full-featured movie frame alignment |
| tif2mrc | mrc | TIFF to MRC conversion |
| raw2mrc | mrc | Raw data to MRC |
| mrctaper | mrc | Taper MRC edges |
| processchunks | standalone | Parallel job distribution |

## Priority 3: Medium-Value Missing Programs (~40)

See FEATURES.md "Not Yet Ported" section for full list.

## Priority 4: Scripts Not Ported (~12)

Python workflow scripts: autodoc.py, comchanger.py, imodpy.py, pip.py, prochunks.py, pysed.py, supermont.py, tiltmatch.py, tomocoords.py

## Priority 5: Plugins

Original 3dmod has ~63K lines of C++ plugins (interpolator, beadfix, drawingtools, analysistools, stereology, etc.). Not yet ported. Would need a plugin trait + libloading system.

## Architecture Notes

- All apps use workspace crates as libraries (no process spawning)
- imod-studio calls library functions directly for each workflow step
- GPU via wgpu compute shaders (tilt, ctfphaseflip) and wgpu render pipeline (viewer 3D)
- Software fallback renderer for systems without GPU
- Native file dialogs via rfd crate
- Native menu bar via Slint MenuBar
- Parallel processing via rayon
- target-cpu=native via .cargo/config.toml

## Key Files

### Library crates
- `crates/imod-mrc/` — MRC file I/O (all modes, sub-area, complex, extended headers)
- `crates/imod-model/` — IMOD model I/O (all chunk types, round-trip preservation)
- `crates/imod-math/` — Statistics, regression, optimization, linear algebra
- `crates/imod-fft/` — 1D/2D FFT
- `crates/imod-transforms/` — 2D/3D transforms, .xf/.tlt I/O
- `crates/imod-slice/` — Image processing (filtering, morphology, interpolation)
- `crates/imod-mesh/` — Marching cubes, contour skinning
- `crates/imod-warp/` — Delaunay triangulation, warp files

### Critical apps
- `apps/imod-viewer/` — Main viewer (Slint + wgpu)
- `apps/imod-studio/` — Workflow manager (Slint)
- `apps/tilt/` — Back-projection reconstruction (rayon + wgpu GPU)
- `apps/tiltalign/` — Fiducial alignment solver
- `apps/beadtrack/` — Bead tracking
- `apps/raptor/` — Automatic fiducial detection
- `apps/newstack/` — Stack operations
- `apps/ctfphaseflip/` — CTF correction (wgpu GPU)
