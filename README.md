# IMOD Rust Rewrite

Rewriting the [IMOD](https://bio3d.colorado.edu/imod/) electron microscopy suite in Rust with [Slint](https://slint.dev/) for the user interface.

IMOD is a set of programs for 3D reconstruction, modeling, and visualization of electron microscopy (EM) tomographic data. The original codebase (~969K lines) is a mix of C, C++, Fortran, and Java.

## Original Architecture

| Component | Language | Lines | Purpose |
|-----------|----------|------:|---------|
| Etomo | Java/Swing | 346K | Workflow manager GUI |
| 3dmod | C++/Qt/OpenGL | 94K | Interactive 3D viewer/modeler |
| Core libs (libcfshr, libimod, libiimod, libwarp, ...) | C | ~130K | Math, file I/O, model data |
| flib/ | Fortran 77/90 | ~126K | Reconstruction, alignment, ~100 programs |
| raptor | C++ | 106K | Automatic fiducial tracking |
| CLI tools (clip, mrc, imodutil) | C++ | ~50K | Image/model utilities |
| CUDA kernels | CUDA | 8K | GPU backprojection, frame alignment, CTF |

## Rust Workspace Structure

```
imod-rs/
  crates/
    imod-core/          # Shared types, errors, constants
    imod-mrc/           # MRC file format read/write (binrw)
    imod-image-io/      # Multi-format I/O (TIFF, JPEG, HDF5, EER)
    imod-model/         # IMOD model structures & I/O
    imod-math/          # Statistics, regression, interpolation
    imod-fft/           # FFT (rustfft / fftw wrapper)
    imod-slice/         # Slice/image processing operations
    imod-warp/          # Warping, Delaunay, distortion correction
    imod-mesh/          # Mesh generation from contours
    imod-ctf/           # CTF estimation & correction
    imod-align/         # Tilt series alignment
    imod-reconstruct/   # Back-projection (SIMD + rayon)
    imod-track/         # Bead tracking
    imod-blend/         # Montage blending
    imod-transforms/    # Affine transforms, .xf/.xg file I/O
    imod-autodoc/       # Autodoc/PIP parameter parsing
    imod-gpu/           # GPU acceleration (wgpu compute / cudarc)
    imod-plugin/        # Plugin trait + dynamic loading
  apps/
    imod-viewer/        # 3dmod replacement (Slint + wgpu)
    imod-aligner/       # midas replacement (Slint + wgpu)
    imod-ctfplot/       # ctfplotter replacement (Slint)
    imod-studio/        # Etomo replacement (Slint) - workflow manager
    clip/               # Image processing CLI
    newstack/           # Stack operations
    tilt/               # Back-projection reconstruction
    tiltalign/          # Tilt series alignment
    beadtrack/          # Fiducial bead tracking
    blendmont/          # Montage blending
    ...                 # One binary per command-line tool
```

## Key Dependencies

| Domain | Crate | Purpose |
|--------|-------|---------|
| Binary formats | `binrw` | MRC header parsing |
| Linear algebra | `nalgebra`, `ndarray` | Matrix operations, transforms |
| FFT | `rustfft` (+ optional `fftw`) | Fourier transforms |
| Image I/O | `tiff`, `image`, `hdf5` | Format backends |
| GPU rendering | `wgpu` | Replaces legacy OpenGL |
| UI framework | `slint` | Replaces Qt + Java Swing |
| Parallelism | `rayon` | Data-parallel processing |
| CLI parsing | `clap` | Command-line arguments |
| Geometry | `delaunator` | Delaunay triangulation |
| GPU compute | `cudarc` | CUDA backprojection |
| Process mgmt | `tokio` | Parallel job orchestration |

## Migration Phases

### Phase 0: Scaffolding (Months 1-2)
- Workspace setup, `imod-core` types
- `imod-mrc`: MRC file format with round-trip tests
- `imod-autodoc`: PIP parameter spec parsing

### Phase 1: Core Libraries (Months 2-5)
Build in dependency order: math -> transforms -> fft -> slice -> image-io -> model -> warp -> mesh

### Phase 2: Command-Line Tools (Months 5-9)
Priority: newstack, clip, mrcbyte/mrcinfo, tiltxcorr, ccderaser, ctfphaseflip, alignframes

### Phase 3: Reconstruction Pipeline (Months 8-13)
tiltalign, tilt (with SIMD inner loops), beadtrack, blendmont, findwarp/warpvol

### Phase 4: imod-viewer (Months 10-16)
3dmod replacement with Slint + wgpu: image display, model overlay, slicer, XYZ view, 3D model view, isosurface, contour editing, plugin system

### Phase 5: Workflow Manager + Other GUIs (Months 14-18)
- `imod-studio`: Etomo replacement with wizard-style workflow, process monitoring, log viewing
- `imod-aligner`: midas replacement
- `imod-ctfplot`: ctfplotter replacement

### Phase 6: RAPTOR + Remaining Tools (Months 16-20)
Automatic fiducial tracking rewrite, remaining ~100 Fortran programs

### Phase 7: Testing & Polish (Months 18-22)
Format compatibility testing, performance benchmarking, script migration

## Key Design Decisions

- **No auto-translation**: All code is manually rewritten in idiomatic Rust
- **File format compatibility**: MRC, IMOD model, .xf/.xg, tilt angles, and warp files remain byte-compatible with original IMOD
- **Fortran inner loops**: Backprojection rewritten with SIMD intrinsics (`std::arch`) + `rayon` for competitive performance
- **GPU strategy**: Start CPU-only, add `cudarc` for backprojection and `wgpu` compute shaders later
- **OpenGL -> wgpu**: Complete rendering rewrite from legacy fixed-function to modern shader pipelines

## Building

```sh
cargo build --release
```

## Usage Examples

### GUI Applications

```sh
# Image viewer — open an MRC stack, optionally with a model overlay
cargo run --release --bin imod-viewer -- dataset.mrc
cargo run --release --bin imod-viewer -- dataset.st model.mod

# Workflow manager — tomographic reconstruction pipeline
cargo run --release --bin imod-studio
```

### Image Utilities

```sh
# Print MRC header info
cargo run --release --bin mrcinfo -- dataset.st

# Convert to byte mode (auto-scaled from header min/max)
cargo run --release --bin mrcbyte -- input.mrc output_byte.mrc

# Convert with contrast scaling (mean +/- 3 sigma)
cargo run --release --bin mrcbyte -- input.mrc output.mrc -c 3

# Bin a stack by factor 2
cargo run --release --bin newstack -- -i input.st -o binned.st -b 2

# Extract sections 0-10
cargo run --release --bin newstack -- -i input.st -o subset.st -S 0-10

# Apply transforms from an .xf file
cargo run --release --bin newstack -- -i input.st -o aligned.st -x transforms.xf

# Print per-section statistics
cargo run --release --bin clip -- stats input.mrc

# Fourier low-pass filter
cargo run --release --bin mtffilter -- -i input.mrc -o filtered.mrc -l 0.4
```

### Volume Operations

```sh
# Bin a volume (2x in XY, 2x in Z)
cargo run --release --bin binvol -- input.rec output.rec -b 2

# Extract a subvolume
cargo run --release --bin trimvol -- input.rec output.rec -x 100 -y 100 --nx 256 --ny 256

# Rotate volume for XZ viewing (swap Y and Z)
cargo run --release --bin trimvol -- input.rec output.rec -r

# Rotate volume around Z axis
cargo run --release --bin rotatevol -- input.rec rotated.rec -a 45
```

### Reconstruction Pipeline

```sh
# 1. Cross-correlation alignment
cargo run --release --bin tiltxcorr -- -i dataset.st -o dataset.prexf

# 2. Apply coarse alignment
cargo run --release --bin newstack -- -i dataset.st -o dataset.preali -x dataset.prexf

# 3. Track fiducial beads
cargo run --release --bin beadtrack -- -i dataset.preali -s dataset.seed -o dataset.fid -t dataset.rawtlt -d 10

# 4. Solve alignment from fiducials
cargo run --release --bin tiltalign -- -m dataset.fid -t dataset.rawtlt -o dataset.xf

# 5. Create final aligned stack
cargo run --release --bin newstack -- -i dataset.st -o dataset.ali -x dataset.xf

# 6. Reconstruct by back-projection
cargo run --release --bin tilt -- -i dataset.ali -o dataset.rec -t dataset.tlt -z 200

# 7. Erase gold beads from reconstruction
cargo run --release --bin eraser -- -i dataset.rec -o dataset_erased.rec -m beads.mod -r 5
```

### Model Utilities

```sh
# Print model information
cargo run --release --bin imodinfo -- model.mod

# Transform model points (translate, rotate, scale)
cargo run --release --bin imodtrans -- model.mod output.mod --tx 10 --ty 20 --rz 45

# Create model from point list
cargo run --release --bin point2model -- points.txt -o model.mod

# Merge multiple models
cargo run --release --bin imodjoin -- model1.mod model2.mod -o combined.mod

# Mesh contours
cargo run --release --bin imodmesh -- model.mod -o meshed.mod
```

### Transform Utilities

```sh
# Invert transforms
cargo run --release --bin xfinverse -- -i input.xf -o inverted.xf

# Compose two transform files
cargo run --release --bin xfproduct -- -1 first.xf -2 second.xf -o product.xf
```

## License

Derived from IMOD; thus, see original IMOD licenses

## References

- [IMOD home page](https://bio3d.colorado.edu/imod/)
- [Slint UI framework](https://slint.dev/)
- [MRC file format](https://www.ccpem.ac.uk/mrc_format/mrc2014.php)
