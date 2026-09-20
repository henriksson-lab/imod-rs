//! Translation of `IMOD/mrc/frameutil.cpp` and `IMOD/mrc/frameutil.h`
//! (utilities needed from framealign and gpuframe).
//!
//! The C unit's four file statics (`sDumpInd`, `sPrintFunc`, `sDot`,
//! `sDumpDir`, `sSetDumpDir`) live here as module statics.  `utilPrint` is
//! variadic in the C; as everywhere else in this tree the argument list
//! becomes a `&[CArg]` handed to `b3dutil::c_format`, which is the translation
//! of the `vsprintf` the C performs.

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format, data_size_for_mode, imod_backup_file,
};
use crate::imod::libcfshr::filtxcorr::wrap_fft_slice;
use crate::imod::libcfshr::islice::{Islice, MrcData, slice_init};
use crate::imod::libfft::todfft::todfft_c;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_write,
    mrc_write_slice,
};
use crate::imod::libiimod::mrcslice::slice_mmm;
use std::cell::{Cell, RefCell};
use std::io::Write;

thread_local! {
    /// C `static int sDumpInd = 0;`
    static S_DUMP_IND: Cell<i32> = const { Cell::new(0) };
    /// C `static CharArgType sPrintFunc = NULL;`
    static S_PRINT_FUNC: Cell<Option<fn(&str)>> = const { Cell::new(None) };
    /// C `static const char *sDumpDir = sDot;` plus `static bool sSetDumpDir`.
    static S_DUMP_DIR: RefCell<String> = const { RefCell::new(String::new()) };
    static S_SET_DUMP_DIR: Cell<bool> = const { Cell::new(false) };
}

/// C `sDot`.
const S_DOT: &str = ".";

/// C `typedef void (*CharArgType)(const char *message);` (`frameutil.h:10`).
/// The message is already formatted by the time it reaches the callback, so
/// the `const char *` becomes `&str`.
pub type CharArgType = Option<fn(&str)>;

/// C `utilCoordsForWrap` (`frameutil.cpp:31`).
///
/// Compute coordinates for wrapping an image or subset of it to move corner to
/// center.  The C fills eight caller-supplied `int[4]` arrays; here they are
/// `&mut [i32; 4]` in the same argument order.
#[allow(clippy::too_many_arguments)]
pub fn util_coords_for_wrap(
    nx_from: i32,
    ny_from: i32,
    nx_to: i32,
    ny_to: i32,
    x_offset: i32,
    y_offset: i32,
    ix_from0: &mut [i32; 4],
    ix_to0: &mut [i32; 4],
    iy_from0: &mut [i32; 4],
    iy_to0: &mut [i32; 4],
    ix_from1: &mut [i32; 4],
    ix_to1: &mut [i32; 4],
    iy_from1: &mut [i32; 4],
    iy_to1: &mut [i32; 4],
) {
    let mut num: i32;
    ix_from0[0] = if 0 > x_offset - nx_to / 2 {
        0
    } else {
        x_offset - nx_to / 2
    };
    ix_from0[3] = ix_from0[0];
    ix_from1[0] = nx_to / 2 + x_offset - 1;
    ix_from1[3] = ix_from1[0];
    num = ix_from1[0] + 1 - ix_from0[0];
    ix_to1[0] = nx_to - 1;
    ix_to1[3] = ix_to1[0];
    ix_to0[0] = nx_to - num;
    ix_to0[3] = ix_to0[0];

    iy_from0[0] = if 0 > y_offset - ny_to / 2 {
        0
    } else {
        y_offset - ny_to / 2
    };
    iy_from0[1] = iy_from0[0];
    iy_from1[0] = ny_to / 2 + y_offset - 1;
    iy_from1[1] = iy_from1[0];
    num = iy_from1[0] + 1 - iy_from0[0];
    iy_to1[0] = ny_to - 1;
    iy_to1[1] = iy_to1[0];
    iy_to0[0] = ny_to - num;
    iy_to0[1] = iy_to0[0];

    ix_from0[1] = x_offset + nx_from - nx_to / 2;
    ix_from0[2] = ix_from0[1];
    ix_from1[1] = if nx_from - 1 < ix_from0[1] + nx_to - 1 {
        nx_from - 1
    } else {
        ix_from0[1] + nx_to - 1
    };
    ix_from1[2] = ix_from1[1];
    num = ix_from1[1] + 1 - ix_from0[1];
    ix_to0[1] = 0;
    ix_to0[2] = 0;
    ix_to1[1] = num - 1;
    ix_to1[2] = num - 1;

    iy_from0[2] = y_offset + ny_from - ny_to / 2;
    iy_from0[3] = iy_from0[2];
    iy_from1[2] = if ny_from - 1 < iy_from0[2] + ny_to - 1 {
        ny_from - 1
    } else {
        iy_from0[2] + ny_to - 1
    };
    iy_from1[3] = iy_from1[2];
    num = iy_from1[2] + 1 - iy_from0[2];
    iy_to0[2] = 0;
    iy_to0[3] = 0;
    iy_to1[2] = num - 1;
    iy_to1[3] = num - 1;
}

/// C `utilRollSavedFrames` (`frameutil.cpp:65`).
///
/// The C rolls a `std::vector<float *>`; the Rust stacks own their buffers, so
/// the element type is generic and the rotation moves the owned buffers in the
/// same order the pointers move.
pub fn util_roll_saved_frames<T>(saved_vec: &mut [T], num_frames: i32) {
    if num_frames < 1 {
        return;
    }
    saved_vec[..num_frames as usize].rotate_left(1);
}

/// C `utilDumpFFT` (`frameutil.cpp:76`).
///
/// Output an FFT, or convert to image and dump that and convert back.
pub fn util_dump_fft(
    fft: &mut [f32],
    nx_pad: i32,
    ny_pad: i32,
    descrip: &str,
    real: i32,
    frame: i32,
    scale: i32,
) {
    let mut hdr = MrcHeader::default();
    let scale_fac = (1. / ((nx_pad as f64) * (ny_pad as f64)).sqrt()) as f32;
    check_dump_dir();
    if real != 0 {
        todfft_c(fft, nx_pad, ny_pad, 1);
        util_dump_image(fft, nx_pad + 2, nx_pad, ny_pad, 0, descrip, 0);
        todfft_c(fft, nx_pad, ny_pad, 0);
        return;
    }
    let mut shift_temp = vec![0.0f32; (2 * nx_pad + 4) as usize];
    wrap_fft_slice(fft, &mut shift_temp, (nx_pad + 2) / 2, ny_pad, 0);
    if scale != 0 {
        for ind in 0..((nx_pad + 2) * ny_pad) as usize {
            fft[ind] *= scale_fac;
        }
    }
    mrc_head_new(
        &mut hdr,
        (nx_pad + 2) / 2,
        ny_pad,
        1,
        MRC_MODE_COMPLEX_FLOAT,
    );
    let mut slice = new_islice();
    let count = ((nx_pad + 2) * ny_pad) as usize;
    slice_init(
        &mut slice,
        (nx_pad + 2) / 2,
        ny_pad,
        MRC_MODE_COMPLEX_FLOAT,
        MrcData::F(fft[..count].to_vec()),
    );
    slice_mmm(&mut slice);
    hdr.amin = slice.min;
    hdr.amax = slice.max;
    hdr.amean = slice.mean;
    let fname = c_format(
        "%s/fafft-%d.mrc",
        &[
            CArg::Str(&s_dump_dir()),
            CArg::Int(S_DUMP_IND.with(|c| c.get()) as i64),
        ],
    );
    let fp = ImodFile::open(&fname, "wb");
    if let Some(mut fp) = fp {
        mrc_head_write(&mut fp, &mut hdr);
        let bytes: Vec<u8> = fft[..count].iter().flat_map(|v| v.to_ne_bytes()).collect();
        mrc_write_slice(&bytes, &mut fp, &mut hdr, 0, b'Z');
        drop(fp);
        util_print(
            "Saved %s fft frame %d in %s\n",
            &[
                CArg::Str(descrip),
                CArg::Int(frame as i64),
                CArg::Str(&fname),
            ],
        );
    }
    S_DUMP_IND.with(|c| c.set(c.get() + 1));
    if scale != 0 {
        for ind in 0..count {
            fft[ind] /= scale_fac;
        }
    }
    wrap_fft_slice(fft, &mut shift_temp, (nx_pad + 2) / 2, ny_pad, 1);
}

/// C `utilDumpImage` (`frameutil.cpp:127`).
///
/// Output an image, wrapping it properly if it is a correlation with
/// `ifCorr > 0`.  To output a non-float image, pass `ifCorr` as `-1 - mode`;
/// the C's `float *buf` then carries the raw bytes of that mode, which is why
/// the copy below is over bytes.
pub fn util_dump_image(
    buf: &[f32],
    nx_dim: i32,
    nx_pad: i32,
    ny_pad: i32,
    if_corr: i32,
    descrip: &str,
    frame: i32,
) {
    let mut hdr = MrcHeader::default();
    let mut data_size = 0;
    let mut csize = 0;
    let mut mode = MRC_MODE_FLOAT;
    check_dump_dir();
    if if_corr < 0 {
        mode = -if_corr - 1;
    }
    if data_size_for_mode(mode, &mut data_size, &mut csize) < 0 {
        return;
    }
    // C: `malloc(nxPad * nyPad * dataSize)`, with no `csize` factor.
    let mut temp = vec![0u8; (nx_pad * ny_pad * data_size) as usize];
    if if_corr > 0 {
        let mut ix_in = nx_pad / 2;
        let mut iy_in = ny_pad / 2;
        let mut iout = 0usize;
        for _iy in 0..ny_pad {
            for _ix in 0..nx_pad {
                let value = buf[(ix_in + iy_in * nx_dim) as usize];
                temp[iout * 4..iout * 4 + 4].copy_from_slice(&value.to_ne_bytes());
                iout += 1;
                ix_in = (ix_in + 1) % nx_pad;
            }
            iy_in = (iy_in + 1) % ny_pad;
        }
    } else {
        let src: Vec<u8> = buf.iter().flat_map(|v| v.to_ne_bytes()).collect();
        for iy in 0..ny_pad {
            let to = (iy * nx_pad * data_size) as usize;
            let from = (iy * nx_dim * data_size) as usize;
            let n = (data_size * nx_pad) as usize;
            temp[to..to + n].copy_from_slice(&src[from..from + n]);
        }
    }
    mrc_head_new(&mut hdr, nx_pad, ny_pad, 1, mode);
    let fname = c_format(
        "%s/faimg-%d.mrc",
        &[
            CArg::Str(&s_dump_dir()),
            CArg::Int(S_DUMP_IND.with(|c| c.get()) as i64),
        ],
    );
    imod_backup_file(&fname);
    let fp = ImodFile::open(&fname, "wb");
    let mut slice = new_islice();
    slice_init(
        &mut slice,
        nx_pad,
        ny_pad,
        mode,
        bytes_as_mrc_data(mode, &temp),
    );
    slice_mmm(&mut slice);
    hdr.amin = slice.min;
    hdr.amax = slice.max;
    hdr.amean = slice.mean;
    if let Some(mut fp) = fp {
        mrc_head_write(&mut fp, &mut hdr);
        mrc_write_slice(&temp, &mut fp, &mut hdr, 0, b'Z');
        drop(fp);
        util_print(
            "Saved %s image frame %d in %s    mean %.2f\n",
            &[
                CArg::Str(descrip),
                CArg::Int(frame as i64),
                CArg::Str(&fname),
                CArg::Dbl(hdr.amean as f64),
            ],
        );
    }
    S_DUMP_IND.with(|c| c.set(c.get() + 1));
}

/// The `Islice` the C declares on the stack and fills with `sliceInit`.  Rust
/// has no uninitialised stack struct, and `sliceInit` overwrites every field
/// the dumps read.
fn new_islice() -> Islice {
    Islice {
        data: MrcData::default(),
        xsize: 0,
        ysize: 0,
        mode: 0,
        csize: 0,
        dsize: 0,
        min: 0.,
        max: 0.,
        mean: 0.,
        index: -1,
        cval: [0.; 4],
    }
}

/// The C hands `sliceInit` the same `temp` buffer whatever the mode; the Rust
/// `Islice` owns a typed `MrcData`, so the bytes are reinterpreted here.
fn bytes_as_mrc_data(mode: i32, bytes: &[u8]) -> MrcData {
    match mode {
        1 => MrcData::S(
            bytes
                .chunks_exact(2)
                .map(|c| i16::from_ne_bytes([c[0], c[1]]))
                .collect(),
        ),
        6 => MrcData::Us(
            bytes
                .chunks_exact(2)
                .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                .collect(),
        ),
        2 | 4 => MrcData::F(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        _ => MrcData::B(bytes.to_vec()),
    }
}

/// C `checkDumpDir` (`frameutil.cpp:184`).
fn check_dump_dir() {
    if S_SET_DUMP_DIR.with(|c| c.get()) {
        return;
    }
    if let Some(dir) = std::env::var_os("FRAMEALIGN_DUMPDIR") {
        S_DUMP_DIR.with_borrow_mut(|d| *d = dir.to_string_lossy().into_owned());
    }
    S_SET_DUMP_DIR.with(|c| c.set(true));
    if S_DUMP_DIR.with_borrow(|d| d.is_empty()) {
        S_DUMP_DIR.with_borrow_mut(|d| *d = S_DOT.to_string());
    }
}

fn s_dump_dir() -> String {
    S_DUMP_DIR.with_borrow(|d| {
        if d.is_empty() {
            S_DOT.to_string()
        } else {
            d.clone()
        }
    })
}

/// C `utilSetPrintFunc` (`frameutil.cpp:195`).
pub fn util_set_print_func(func: Option<fn(&str)>) {
    S_PRINT_FUNC.with(|c| c.set(func));
}

/// C `utilPrint` (`frameutil.cpp:201`).
///
/// Print a message with flushes that were needed for fortran.  The C's
/// `vsprintf` into `char errorMess[512]` is `c_format`.
pub fn util_print(format: &str, args: &[CArg]) {
    let error_mess = c_format(format, args);
    if let Some(func) = S_PRINT_FUNC.with(|c| c.get()) {
        func(&error_mess);
    } else {
        print!("{error_mess}");
        let _ = std::io::stdout().flush();
        let _ = std::io::stdout().flush();
    }
}
