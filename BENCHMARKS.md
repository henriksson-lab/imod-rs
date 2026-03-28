# IMOD-RS Performance Benchmarks

Measured on macOS, Apple Silicon, `cargo build --release`.
Test data: synthetic float32 MRC stacks with sinusoidal patterns.

## Results

### I/O Operations

| Operation | Data Size | Time | Throughput |
|-----------|-----------|------|------------|
| mrcinfo (header read) | 10 MB | 0.086s | instant |
| mrcinfo (header read) | 61 MB | 0.075s | instant |
| mrcinfo (header read) | 164 MB | 0.103s | instant |
| mrcbyte (float→byte) | 10 MB (256x256x41) | 0.104s | 96 MB/s |
| mrcbyte (float→byte) | 61 MB (512x512x61) | 0.246s | 248 MB/s |
| mrcbyte (float→byte) | 164 MB (1024x1024x41) | 0.700s | 234 MB/s |

### Stack Operations

| Operation | Data Size | Time | Throughput |
|-----------|-----------|------|------------|
| newstack bin-by-2 | 256x256x41 (10 MB) | 0.107s | 93 MB/s |
| newstack bin-by-2 | 512x512x61 (61 MB) | 0.236s | 258 MB/s |
| newstack bin-by-2 | 1024x1024x41 (164 MB) | 0.670s | 245 MB/s |
| clip stats | 256x256x41 | 0.096s | 104 MB/s |
| clip stats | 512x512x61 | 0.200s | 305 MB/s |
| avgstack | 256x256x41 | 0.859s | 12 MB/s |
| binvol 3D | 256x256x41 | 0.126s | 79 MB/s |

### Alignment

| Operation | Data Size | Time |
|-----------|-----------|------|
| tiltxcorr (CC alignment) | 256x256x41 | 0.284s |
| tiltxcorr (CC alignment) | 512x512x61 | 0.942s |

### Reconstruction

| Operation | Data Size | 1 thread | 8 threads | Speedup |
|-----------|-----------|----------|-----------|---------|
| tilt (back-projection) | 256x256x41, z=128 | 2.8s | 0.9s | **3.1x** |
| tilt (back-projection) | 512x512x61, z=256 | 33.3s | 9.1s | **3.7x** |

### Filtering

| Operation | Data Size | Time | Throughput |
|-----------|-----------|------|------------|
| mtffilter (FFT lowpass) | 256x256x41 | 0.233s | 43 MB/s |
| mtffilter (FFT lowpass) | 512x512x61 | 1.145s | 53 MB/s |
| ccderaser (artifact removal) | 256x256x41 | 0.186s | 54 MB/s |

## Analysis

### Fast operations (>100 MB/s throughput)
- **mrcinfo**: Effectively instant (header only)
- **mrcbyte**: ~240 MB/s — I/O bound
- **newstack binning**: ~250 MB/s — I/O bound
- **clip stats**: ~200 MB/s — simple arithmetic

### Medium operations (10-100 MB/s)
- **FFT filtering**: ~50 MB/s — CPU-bound on FFT
- **ccderaser**: ~54 MB/s — neighborhood scanning
- **binvol**: ~80 MB/s — 3D averaging

### Reconstruction (after rayon parallelization)
- **tilt back-projection**: 3.1-3.7x speedup with rayon on 8 cores
- 512x512x61 tilt series → 256-thick reconstruction in **9.1 seconds**
- Further optimization possible with SIMD intrinsics and GPU compute

### Comparison to original IMOD
Without building the original IMOD from source, direct comparison isn't possible.
Expected relative performance:
- **I/O operations**: Similar (both I/O bound)
- **FFT**: Similar (both use optimized FFT libraries — rustfft vs FFTW)
- **Back-projection**: Now competitive. Original IMOD with OpenMP is likely similar speed; CUDA version would still be faster for large datasets.
- **Statistics/math**: Similar (both straightforward floating-point arithmetic)
