//! `IMOD/flib/tilt`: the Fortran back-projection/projection kernels that
//! `tilt.cpp` calls, and the no-GPU stubs (`nogpu.cpp` + `gpubp.h`) the CUDA-free
//! reference build links, and the program itself, `tilt.cpp` + `tilt.h` ->
//! `tilt.rs` (`ORDER.md` §9 wave 3), which is their caller.
//!
//! # Argument conventions (for `tilt.rs`)
//!
//! `tilt.cpp:60-83` calls these through `extern "C"` prototypes with every
//! argument by pointer, via the thin `Tilt::bpSumNoX` … `Tilt::loadedProjectingPoint`
//! wrappers (`tilt.cpp:7265-7308`).  The Rust signatures follow those wrappers:
//!
//! - **Arrays are the whole C arrays**, passed as slices from element 0
//!   (`outArr`, `inputArr`, `mInputArray`, `mXprojFs`, …, `indDelRay`,
//!   `numRaysHit`, `rayAreas`), and **every index argument keeps its Fortran
//!   1-based value unchanged** (`ind1`/`index`, `ipoint`, `ipBase`, `jStart`,
//!   `jEnd`, …).  Element `arr(i)` of the source is `arr[i - 1]` here.  So a call
//!   site in `tilt.rs` passes exactly the integers `tilt.cpp` passes — no offset
//!   arithmetic and no pre-offset subslices.  (`tilt.cpp` never passes an offset
//!   pointer to these five routines; the offset pointers in `project`,
//!   `&inputArr[indStart - 1]`, go only to the GPU stubs.)
//! - A scalar the source **assigns and the C caller reads back** is `&mut`
//!   (`ind1`/`index` in `bpSumNoX`/`bpSumXtilt`/`bpSumLocal`; `xx`, `yy`, `zz`,
//!   `sum` in `projSumLocal`; `xx`, `yy` in `loadedProjectingPoint`).
//! - `xProj1` is assigned by the three `bpSum*NoX`/`Xtilt` routines but the C
//!   wrappers pass a copy of a by-value `double`, so it is `&mut f64` here and
//!   `tilt.rs` passes `&mut` a local copy, as the wrapper does.
//! - `bpSumAreaNoX` never assigns `ind1`, so it is by value; the caller advances
//!   its own index (`tilt.cpp:1898`).
//! - `bpSumLocal`'s `projSuperFac` is `Tilt::mProjSuperFac` (added by the
//!   wrapper); `projSumLocal`'s `array`, `xProjFs`…, `numWarpDelz`,
//!   `dxWarpDelz`, `warpDelz`, `ithickReproj`, `inPlaneSize` are the wrapper's
//!   `m*` members.
//!
//! Out-of-range indices panic.  Because the slices are the whole C allocations,
//! a panic corresponds to a read or write outside the C array, i.e. undefined
//! behaviour in the source.
//!
//! # Precision
//!
//! Compiled by the reference with `TILTFFLAGS` = `-m64 -O3 …` (no `-march`, no
//! `-ffast-math`, no `-fdefault-real-8`): SSE2 scalar arithmetic, no FMA
//! contraction, left-to-right evaluation by declared type.  Each module
//! documents its `real*4`/`real*8` mixes.  None of these files has an OpenMP
//! directive.

/// Fortran conversion of a real to `integer*4` (`iProj = xProj`, `int(...)`), as
/// the reference gfortran compiles it on x86-64: a bare `cvttsd2si` / `cvttss2si`,
/// truncating toward zero.  Rust's `as i32` saturates instead, which costs a
/// clamp and a NaN test per conversion in these loops and differs from native
/// only where the value is out of `i32` range (NaN or overflow; the Fortran is
/// undefined there and the instruction returns `i32::MIN`).  Off x86-64 the scalar
/// `gfortran_rt::cvttss2si`/`cvttsd2si` give the same `i32::MIN`.
macro_rules! fortran_int {
    (f64: $x:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE2 is part of the x86-64 baseline, so the target feature
            // these intrinsics require is always enabled.
            unsafe { core::arch::x86_64::_mm_cvttsd_si32(core::arch::x86_64::_mm_set_sd($x)) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            crate::imod::flib::subrs::compat::gfortran_rt::cvttsd2si($x)
        }
    }};
    (f32: $x:expr) => {{
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: as above; SSE is part of the x86-64 baseline.
            unsafe { core::arch::x86_64::_mm_cvttss_si32(core::arch::x86_64::_mm_set_ss($x)) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            crate::imod::flib::subrs::compat::gfortran_rt::cvttss2si($x)
        }
    }};
}

pub mod bpsumlocal;
pub mod bpsumnox;
pub mod bpsumxtilt;
pub mod nogpu;
pub mod projsumlocal;
pub mod tilt;
