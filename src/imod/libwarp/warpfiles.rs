//! Translation of `IMOD/libwarp/warpfiles.c`.
//!
//! Converted to idiomatic Rust per `NATIVE.md`.  Three shapes changed:
//!
//! * the file's twenty-odd `static` variables are thread-local `Cell`s and
//!   `RefCell`s, and `sCurWarpFile` — a pointer into `sWarpFiles` — is just
//!   `S_CUR_FILE_IND`, since the array it pointed into is now a `Vec`;
//! * `sDelau` and `sNninterp` were two separate statics that
//!   `freeStaticArrays` destroyed independently.  `nnai_build` now *moves* the
//!   triangulation into the `Nnai` (Rust cannot alias it), so only
//!   `S_NNINTERP` is kept and the triangulation is reached as `nn.d`;
//! * every `fprintf` goes through [`c_format`] with the source's literal
//!   format string.  `%f` is six decimals and `{}` is not `%f`, so the warp
//!   file this writes is byte-identical to the C's only through that path.
#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, fgetline, imod_backup_file};
use crate::imod::libcfshr::linearxforms::{xf_copy, xf_mult, xf_unit};
use crate::imod::libcfshr::robuststat::rs_sort_floats;
use crate::imod::libwarp::delaunay::delaunay_build;
use crate::imod::libwarp::nn::{NN_VERBOSE, Point};
use crate::imod::libwarp::nnai::{Nnai, nnai_build, nnai_interpolate, nnai_setwmin};
use crate::imod::libwarp::warputils::{extract_linear_xform, extrapolate_done, extrapolate_grid};

/// C `WARP_INVERSE` (`warpfiles.h`).
pub const WARP_INVERSE: i32 = 1;
/// C `WARP_CONTROL_PTS` (`warpfiles.h`).
pub const WARP_CONTROL_PTS: i32 = 2;

/// C `DEFAULT_PERCENTILE` (`warpfiles.c:23`).
const DEFAULT_PERCENTILE: f32 = 0.5;
/// C `DEFAULT_FACTOR` (`warpfiles.c:24`).
const DEFAULT_FACTOR: f32 = 0.15;
/// C `MAX_LINE` (`warpfiles.c:176`).
const MAX_LINE: usize = 1024;
/// C `MAX_VALS` (`warpfiles.c:177`).
const MAX_VALS: i32 = 50;

/// C `WarpFile` (`warpfiles.c:28`).
///
/// `Ilist *warpings` becomes a `Vec<Warping>`: the source only ever
/// `ilistItem`s it by index, `ilistAppend`s to it and `ilistTruncate`s it to 0.
pub struct WarpFile {
    pub nx: i32,
    pub ny: i32,
    pub num_frames: i32,
    pub binning: i32,
    pub pixel_size: f32,
    pub flags: i32,
    pub warpings: Vec<Warping>,
    pub in_use: i32,
}

/// C `Warping` (`warpfiles.c:37`).
#[derive(Clone)]
pub struct Warping {
    pub xform: [f32; 6],
    pub nx_grid: i32,
    pub ny_grid: i32,
    pub x_start: f32,
    pub y_start: f32,
    pub x_interval: f32,
    pub y_interval: f32,
    pub x_vector: Vec<f32>,
    pub y_vector: Vec<f32>,
    pub n_control: i32,
    pub x_control: Vec<f32>,
    pub y_control: Vec<f32>,
    pub max_vectors: i32,
}

/// The `void *values` / `int floats` pair `readLineOfValues` takes
/// (`warpfiles.c:1330`), which selects between an `int` and a `float` array.
pub enum LineValues<'a> {
    Ints(&'a mut [i32]),
    Floats(&'a mut [f32]),
}

thread_local! {
    /// C `sWarpFiles` and `sNumWarpFiles` (`warpfiles.c:49-50`).
    static S_WARP_FILES: RefCell<Vec<WarpFile>> = const { RefCell::new(Vec::new()) };
    /// C `sCurFileInd` (`warpfiles.c:51`); also stands in for `sCurWarpFile`.
    static S_CUR_FILE_IND: Cell<i32> = const { Cell::new(-1) };
    /// C `sGridx` (`warpfiles.c:53`).
    static S_GRIDX: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// C `sGridy` (`warpfiles.c:54`).
    static S_GRIDY: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// C `sSolved` (`warpfiles.c:55`).
    static S_SOLVED: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// C `sDpoints` (`warpfiles.c:56`).
    static S_DPOINTS: RefCell<Vec<Point>> = const { RefCell::new(Vec::new()) };
    /// C `sDelau` (`warpfiles.c:57`) and `sNninterp` (`warpfiles.c:58`) in one:
    /// the triangulation is owned by the interpolator now.
    static S_NNINTERP: RefCell<Option<Nnai>> = const { RefCell::new(None) };
    /// C `sLastNumCont` (`warpfiles.c:59`).
    static S_LAST_NUM_CONT: Cell<i32> = const { Cell::new(0) };
    /// C `sLastNxGrid` (`warpfiles.c:60`).
    static S_LAST_NX_GRID: Cell<i32> = const { Cell::new(0) };
    /// C `sLastNyGrid` (`warpfiles.c:61`).
    static S_LAST_NY_GRID: Cell<i32> = const { Cell::new(0) };
    /// C `sLastXStart` (`warpfiles.c:62`).
    static S_LAST_X_START: Cell<f32> = const { Cell::new(-1.) };
    /// C `sLastYStart` (`warpfiles.c:63`).
    static S_LAST_Y_START: Cell<f32> = const { Cell::new(-1.) };
    /// C `sLastXinterval` (`warpfiles.c:64`).
    static S_LAST_X_INTERVAL: Cell<f32> = const { Cell::new(0.) };
    /// C `sLastYinterval` (`warpfiles.c:65`).
    static S_LAST_Y_INTERVAL: Cell<f32> = const { Cell::new(0.) };
    /// C `sLastXdim` (`warpfiles.c:66`).
    static S_LAST_XDIM: Cell<i32> = const { Cell::new(0) };
    /// C `sLastZforSpacing` (`warpfiles.c:68`).
    static S_LAST_Z_FOR_SPACING: Cell<i32> = const { Cell::new(-1) };
    /// C `sLastPercentile` (`warpfiles.c:69`), uninitialised in the source.
    static S_LAST_PERCENTILE: Cell<f32> = const { Cell::new(0.) };
    /// C `sLastSpacing` (`warpfiles.c:70`), uninitialised in the source.
    static S_LAST_SPACING: Cell<f32> = const { Cell::new(0.) };
    /// C `sTimeOutput` (`warpfiles.c:71`); nothing in the source sets it.
    static S_TIME_OUTPUT: Cell<i32> = const { Cell::new(0) };
    /// C `sHeightBaseCrit` (`warpfiles.c:72`).
    static S_HEIGHT_BASE_CRIT: Cell<f64> = const { Cell::new(0.2) };
    /// C `sAreaFractionCrit` (`warpfiles.c:73`).
    static S_AREA_FRACTION_CRIT: Cell<f64> = const { Cell::new(0.1) };
    /// C `sMinNumForPruning` (`warpfiles.c:74`).
    static S_MIN_NUM_FOR_PRUNING: Cell<i32> = const { Cell::new(2) };
}

/// Original static `initWarping` (`warpfiles.c:92`).
///
/// The source fills a caller-supplied struct; here it returns the value, which
/// is the same thing for the two call sites (`addWarpingsIfNeeded` appends it).
pub fn init_warping() -> Warping {
    let mut xform = [0.0_f32; 6];
    xf_unit(&mut xform, 1.0, 2);
    Warping {
        xform,
        nx_grid: 0,
        ny_grid: 0,
        x_start: 0.,
        y_start: 0.,
        x_interval: 0.,
        y_interval: 0.,
        x_vector: Vec::new(),
        y_vector: Vec::new(),
        n_control: 0,
        x_control: Vec::new(),
        y_control: Vec::new(),
        max_vectors: 0,
    }
}

/// Original static `addWarpFile` (`warpfiles.c:101`).
pub fn add_warp_file() -> i32 {
    S_WARP_FILES.with_borrow_mut(|files| {
        let mut index: i32 = -1;

        /* Search for a free warp file in array */
        for i in 0..files.len() {
            if files[i].in_use == 0 {
                index = i as i32;
                break;
            }
        }

        if index < 0 {
            /* Allocate just one at a time when needed */
            files.push(WarpFile {
                nx: 0,
                ny: 0,
                num_frames: 0,
                binning: 0,
                pixel_size: 0.,
                flags: 0,
                warpings: Vec::new(),
                in_use: 0,
            });
            index = files.len() as i32 - 1;
        }

        let wfile = &mut files[index as usize];
        wfile.num_frames = 0;
        wfile.warpings = Vec::new();
        wfile.in_use = 1;
        index
    })
}

/// Original `newWarpFile` (`warpfiles.c:144`).
pub fn new_warp_file(nx: i32, ny: i32, binning: i32, pixel_size: f32, flags: i32) -> i32 {
    let err = add_warp_file();
    if err < 0 {
        return err;
    }
    set_current_warp_file(err);
    S_WARP_FILES.with_borrow_mut(|files| {
        let file = &mut files[err as usize];
        file.nx = nx;
        file.ny = ny;
        file.binning = binning;
        file.pixel_size = pixel_size;
        file.flags = flags;
    });
    err
}

/// Original `readWarpFile` (`warpfiles.c:172`).
#[allow(clippy::too_many_arguments)]
pub fn read_warp_file(
    filename: &str,
    nx: &mut i32,
    ny: &mut i32,
    nz: &mut i32,
    binning: &mut i32,
    pixel_size: &mut f32,
    version: &mut i32,
    flags: &mut i32,
) -> i32 {
    let mut line = [0_u8; MAX_LINE];
    let mut values = [0.0_f32; MAX_VALS as usize];
    let mut err;

    *version = -1;
    let Some(mut fp) = ImodFile::open(filename, "r") else {
        return -1;
    };
    let mut num_vals = 0;
    err = read_line_of_values(
        &mut fp,
        &mut line,
        MAX_LINE as i32,
        &mut LineValues::Floats(&mut values),
        1,
        &mut num_vals,
        MAX_VALS,
    );
    if err < 0 {
        if err != -2 {
            return -2;
        }

        /* An empty file gives -2 and this is a simple file */
        *version = 0;
        return -3;
    }
    if num_vals == 6 {
        *version = 0;
    }
    if num_vals != 1 || err > 0 {
        return -3;
    }
    *version = (values[0] as f64 + 0.5).floor() as i32;
    if *version < 1 || *version > 3 {
        return -4;
    }

    num_vals = 0;
    let _ = read_line_of_values(
        &mut fp,
        &mut line,
        MAX_LINE as i32,
        &mut LineValues::Floats(&mut values),
        1,
        &mut num_vals,
        MAX_VALS,
    );
    if num_vals != *version + 3 {
        return -4;
    }
    let mut i = 0;
    *nx = (values[i] as f64 + 0.5).floor() as i32;
    i += 1;
    *ny = (values[i] as f64 + 0.5).floor() as i32;
    i += 1;
    *nz = 1;
    if *version > 1 {
        *nz = (values[i] as f64 + 0.5).floor() as i32;
        i += 1;
    }
    *binning = (values[i] as f64 + 0.5).floor() as i32;
    i += 1;
    *pixel_size = values[i];
    i += 1;
    *flags = WARP_INVERSE;
    if *version == 3 {
        *flags = (values[i] as f64 + 0.5).floor() as i32;
    }
    if *nx <= 0 || *ny <= 0 || *nz <= 0 || *binning <= 0 {
        return -5;
    }

    /* Ready to create a structure */
    err = new_warp_file(*nx, *ny, *binning, *pixel_size, *flags);
    if err < 0 {
        return -6;
    }

    /* Loop on the frames */
    for iz in 0..*nz {
        if add_warpings_if_needed(iz) != 0 {
            err = -6;
            break;
        }
        num_vals = 0;
        err = read_line_of_values(
            &mut fp,
            &mut line,
            MAX_LINE as i32,
            &mut LineValues::Floats(&mut values),
            1,
            &mut num_vals,
            MAX_VALS,
        );
        if err != 0 {
            break;
        }

        /* Read the header line and allocate arrays.  `PipMemoryError` on each
        of the four allocations cannot fire with a `Vec`. */
        let cur = S_CUR_FILE_IND.get();
        S_WARP_FILES.with_borrow_mut(|files| {
            let warp = &mut files[cur as usize].warpings[iz as usize];
            if *flags & WARP_CONTROL_PTS == 0 {
                warp.nx_grid = (values[2] as f64 + 0.5).floor() as i32;
                warp.ny_grid = (values[5] as f64 + 0.5).floor() as i32;
                warp.x_start = values[0];
                warp.y_start = values[3];
                warp.x_interval = values[1];
                warp.y_interval = values[4];
                warp.x_vector = vec![0.0_f32; (warp.nx_grid * warp.ny_grid) as usize];
                warp.y_vector = vec![0.0_f32; (warp.nx_grid * warp.ny_grid) as usize];
                warp.max_vectors = warp.nx_grid * warp.ny_grid;
            } else {
                warp.n_control = (values[0] as f64 + 0.5).floor() as i32;
                if warp.n_control > 0 {
                    warp.x_vector = vec![0.0_f32; warp.n_control as usize];
                    warp.y_vector = vec![0.0_f32; warp.n_control as usize];
                    warp.x_control = vec![0.0_f32; warp.n_control as usize];
                    warp.y_control = vec![0.0_f32; warp.n_control as usize];
                    warp.max_vectors = warp.n_control;
                }
            }
        });

        /* All version 3 files have a transform so read it */
        if *version > 2 {
            num_vals = 0;
            err = read_line_of_values(
                &mut fp,
                &mut line,
                MAX_LINE as i32,
                &mut LineValues::Floats(&mut values),
                1,
                &mut num_vals,
                MAX_VALS,
            );
            let n_control = S_WARP_FILES
                .with_borrow(|files| files[cur as usize].warpings[iz as usize].n_control);
            if err < 0 || n_control < 0 || num_vals != 6 {
                if err >= 0 {
                    err = -7;
                }
                break;
            }
            S_WARP_FILES.with_borrow_mut(|files| {
                let warp = &mut files[cur as usize].warpings[iz as usize];
                warp.xform[0] = values[0];
                warp.xform[2] = values[1];
                warp.xform[1] = values[2];
                warp.xform[3] = values[3];
                warp.xform[4] = values[4];
                warp.xform[5] = values[5];
            });
        }

        /* Read the grid */
        err = 0;
        if *flags & WARP_CONTROL_PTS == 0 {
            let total = S_WARP_FILES.with_borrow(|files| {
                let warp = &files[cur as usize].warpings[iz as usize];
                warp.nx_grid * warp.ny_grid
            });
            let mut num_read = 0;
            while num_read < total && err == 0 {
                num_vals = 0;
                err = read_line_of_values(
                    &mut fp,
                    &mut line,
                    MAX_LINE as i32,
                    &mut LineValues::Floats(&mut values),
                    1,
                    &mut num_vals,
                    MAX_VALS,
                );
                if err < 0 {
                    break;
                }
                if num_vals % 2 != 0 {
                    err = -7;
                    break;
                }
                S_WARP_FILES.with_borrow_mut(|files| {
                    let warp = &mut files[cur as usize].warpings[iz as usize];
                    let mut i = 0;
                    while i < num_vals as usize {
                        warp.x_vector[num_read as usize] = values[i];
                        warp.y_vector[num_read as usize] = values[i + 1];
                        num_read += 1;
                        i += 2;
                    }
                });
            }
            if num_read < total {
                break;
            }
        } else {
            /* OR read the control points */
            let n_control = S_WARP_FILES
                .with_borrow(|files| files[cur as usize].warpings[iz as usize].n_control);
            let mut i = 0;
            while i < n_control {
                if err != 0 {
                    break;
                }
                num_vals = 4;
                err = read_line_of_values(
                    &mut fp,
                    &mut line,
                    MAX_LINE as i32,
                    &mut LineValues::Floats(&mut values),
                    1,
                    &mut num_vals,
                    MAX_VALS,
                );
                if err < 0 {
                    break;
                }
                S_WARP_FILES.with_borrow_mut(|files| {
                    let warp = &mut files[cur as usize].warpings[iz as usize];
                    warp.x_control[i as usize] = values[0];
                    warp.y_control[i as usize] = values[1];
                    warp.x_vector[i as usize] = values[2];
                    warp.y_vector[i as usize] = values[3];
                });
                i += 1;
            }
            if i < n_control {
                break;
            }
        }
        err = 0;
    }
    if err != 0 {
        let cur = S_CUR_FILE_IND.get();
        if cur >= 0 {
            S_WARP_FILES.with_borrow_mut(|files| delete_warp_file(&mut files[cur as usize]));
        }
        set_current_warp_file(-1);
    }
    S_CUR_FILE_IND.get()
}

/// Original `writeWarpFile` (`warpfiles.c:353`).
pub fn write_warp_file(filename: &str, skip_backup: i32) -> i32 {
    let mut backerr = 0;
    if S_CUR_FILE_IND.get() < 0 {
        return -1;
    }
    if skip_backup == 0 {
        backerr = imod_backup_file(filename);
    }
    let Some(mut fp) = ImodFile::open(filename, "w") else {
        return -1;
    };
    let cur = S_CUR_FILE_IND.get();
    S_WARP_FILES.with_borrow(|files| {
        let cur_file = &files[cur as usize];
        let _ = fp.write_all(b"3\n");
        let _ = fp.write_all(
            c_format(
                "%d %d %d %d %f %d\n",
                &[
                    CArg::Int(cur_file.nx as i64),
                    CArg::Int(cur_file.ny as i64),
                    CArg::Int(cur_file.num_frames as i64),
                    CArg::Int(cur_file.binning as i64),
                    CArg::Dbl(cur_file.pixel_size as f64),
                    CArg::Int(cur_file.flags as i64),
                ],
            )
            .as_bytes(),
        );
        for iz in 0..cur_file.num_frames {
            let warp = &cur_file.warpings[iz as usize];
            if cur_file.flags & WARP_CONTROL_PTS != 0 {
                let _ =
                    fp.write_all(c_format("%d\n", &[CArg::Int(warp.n_control as i64)]).as_bytes());
            } else {
                let _ = fp.write_all(
                    c_format(
                        "%f  %f  %d  %f  %f  %d\n",
                        &[
                            CArg::Dbl(warp.x_start as f64),
                            CArg::Dbl(warp.x_interval as f64),
                            CArg::Int(warp.nx_grid as i64),
                            CArg::Dbl(warp.y_start as f64),
                            CArg::Dbl(warp.y_interval as f64),
                            CArg::Int(warp.ny_grid as i64),
                        ],
                    )
                    .as_bytes(),
                );
            }
            let _ = fp.write_all(
                c_format(
                    "%.6f  %.6f  %.6f  %.6f  %.3f %.3f\n",
                    &[
                        CArg::Dbl(warp.xform[0] as f64),
                        CArg::Dbl(warp.xform[2] as f64),
                        CArg::Dbl(warp.xform[1] as f64),
                        CArg::Dbl(warp.xform[3] as f64),
                        CArg::Dbl(warp.xform[4] as f64),
                        CArg::Dbl(warp.xform[5] as f64),
                    ],
                )
                .as_bytes(),
            );

            if cur_file.flags & WARP_CONTROL_PTS != 0 {
                for i in 0..warp.n_control as usize {
                    let _ = fp.write_all(
                        c_format(
                            "%.3f %.3f %.3f %.3f\n",
                            &[
                                CArg::Dbl(warp.x_control[i] as f64),
                                CArg::Dbl(warp.y_control[i] as f64),
                                CArg::Dbl(warp.x_vector[i] as f64),
                                CArg::Dbl(warp.y_vector[i] as f64),
                            ],
                        )
                        .as_bytes(),
                    );
                }
            } else {
                for j in 0..warp.ny_grid {
                    for i in 0..warp.nx_grid {
                        let _ = fp.write_all(
                            c_format(
                                "  %.3f  %.3f",
                                &[
                                    CArg::Dbl(
                                        warp.x_vector[(i + j * warp.nx_grid) as usize] as f64,
                                    ),
                                    CArg::Dbl(
                                        warp.y_vector[(i + j * warp.nx_grid) as usize] as f64,
                                    ),
                                ],
                            )
                            .as_bytes(),
                        );
                        if i % 4 == 3 || i == warp.nx_grid - 1 {
                            let _ = fp.write_all(b"\n");
                        }
                    }
                }
            }
        }
    });
    let _ = fp.flush();
    drop(fp);
    backerr
}

/// Original static `addWarpingsIfNeeded` (`warpfiles.c:400`).
pub fn add_warpings_if_needed(iz: i32) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 || iz < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow_mut(|files| {
        let file = &mut files[cur as usize];
        if iz < file.num_frames {
            return 0;
        }
        // `for (i = sCurWarpFile->numFrames; i <= iz + 1; i++)` appends one
        // more warping than `numFrames` records, and the source keeps it.
        for _i in file.num_frames..=iz + 1 {
            file.warpings.push(init_warping());
        }
        file.num_frames = iz + 1;
        0
    })
}

/// Original `setCurrentWarpFile` (`warpfiles.c:419`).
pub fn set_current_warp_file(index: i32) -> i32 {
    S_CUR_FILE_IND.set(-1);
    let num = S_WARP_FILES.with_borrow(|files| files.len() as i32);
    if index < 0 || index >= num {
        return -1;
    }
    S_CUR_FILE_IND.set(index);
    0
}

/// Original `clearWarpFile` (`warpfiles.c:434`).
pub fn clear_warp_file(index: i32) -> i32 {
    let num = S_WARP_FILES.with_borrow(|files| files.len() as i32);
    if index < 0 || index >= num {
        return 1;
    }
    if S_WARP_FILES.with_borrow(|files| files[index as usize].in_use) == 0 {
        return 2;
    }
    S_WARP_FILES.with_borrow_mut(|files| delete_warp_file(&mut files[index as usize]));
    if S_CUR_FILE_IND.get() == index {
        S_CUR_FILE_IND.set(-1);
    }
    0
}

/// Original `warpFilesDone` (`warpfiles.c:452`).
pub fn warp_files_done() {
    S_WARP_FILES.with_borrow_mut(|files| {
        for file in files.iter_mut() {
            delete_warp_file(file);
        }
        files.clear();
    });
    free_static_arrays();
    S_CUR_FILE_IND.set(-1);
    extrapolate_done();
}

/// Original static `deleteWarpFile` (`warpfiles.c:468`).
pub fn delete_warp_file(warp_file: &mut WarpFile) {
    warp_file.num_frames = 0;
    warp_file.in_use = 0;
    warp_file.warpings.clear();
}

/// Original `getWarpFileSize` (`warpfiles.c:489`).
pub fn get_warp_file_size(nx: &mut i32, ny: &mut i32, nz: &mut i32, if_control: &mut i32) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow(|files| {
        let file = &files[cur as usize];
        *nx = file.nx;
        *ny = file.ny;
        *nz = file.num_frames;
        *if_control = if file.flags & WARP_CONTROL_PTS != 0 {
            1
        } else {
            0
        };
        0
    })
}

/// Original `setLinearTransform` (`warpfiles.c:505`).
pub fn set_linear_transform(iz: i32, xform: &[f32], rows: i32) -> i32 {
    if add_warpings_if_needed(iz) != 0 {
        return 1;
    }
    let cur = S_CUR_FILE_IND.get();
    S_WARP_FILES.with_borrow_mut(|files| {
        let warp = &mut files[cur as usize].warpings[iz as usize];
        xf_copy(xform, rows as usize, &mut warp.xform, 2);
    });
    0
}

/// Original `setWarpGrid` (`warpfiles.c:522`).
#[allow(clippy::too_many_arguments)]
pub fn set_warp_grid(
    iz: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_start: f32,
    y_start: f32,
    x_interval: f32,
    y_interval: f32,
    dx_grid: &[f32],
    dy_grid: &[f32],
    xdim: i32,
) -> i32 {
    if nx_grid <= 0 || ny_grid <= 0 {
        return 1;
    }
    if add_warpings_if_needed(iz) != 0 {
        return 1;
    }
    let cur = S_CUR_FILE_IND.get();
    S_WARP_FILES.with_borrow_mut(|files| {
        let warp = &mut files[cur as usize].warpings[iz as usize];
        warp.nx_grid = nx_grid;
        warp.ny_grid = ny_grid;
        warp.x_start = x_start;
        warp.y_start = y_start;
        warp.x_interval = x_interval;
        warp.y_interval = y_interval;
        if nx_grid * ny_grid > warp.max_vectors {
            warp.x_vector = vec![0.0_f32; (nx_grid * ny_grid) as usize];
            warp.y_vector = vec![0.0_f32; (nx_grid * ny_grid) as usize];
            warp.max_vectors = nx_grid * ny_grid;
        }
        for iy in 0..ny_grid {
            for ix in 0..nx_grid {
                warp.x_vector[(ix + iy * nx_grid) as usize] = dx_grid[(ix + iy * xdim) as usize];
                warp.y_vector[(ix + iy * nx_grid) as usize] = dy_grid[(ix + iy * xdim) as usize];
            }
        }
        0
    })
}

/// Original `setWarpPoints` (`warpfiles.c:566`).
pub fn set_warp_points(
    iz: i32,
    n_control: i32,
    x_control: &[f32],
    y_control: &[f32],
    x_vector: &[f32],
    y_vector: &[f32],
) -> i32 {
    if n_control < 0 {
        return 1;
    }
    if add_warpings_if_needed(iz) != 0 {
        return 1;
    }
    let cur = S_CUR_FILE_IND.get();
    S_WARP_FILES.with_borrow_mut(|files| {
        let warp = &mut files[cur as usize].warpings[iz as usize];
        warp.n_control = n_control;
        if n_control > warp.max_vectors {
            warp.x_vector = Vec::new();
            warp.y_vector = Vec::new();
            warp.x_control = Vec::new();
            warp.y_control = Vec::new();
            warp.max_vectors = 0;
            if n_control == 0 {
                return 0;
            }
            warp.x_vector = vec![0.0_f32; n_control as usize];
            warp.y_vector = vec![0.0_f32; n_control as usize];
            warp.x_control = vec![0.0_f32; n_control as usize];
            warp.y_control = vec![0.0_f32; n_control as usize];
            warp.max_vectors = n_control;
        }
        for i in 0..n_control as usize {
            warp.x_vector[i] = x_vector[i];
            warp.y_vector[i] = y_vector[i];
            warp.x_control[i] = x_control[i];
            warp.y_control[i] = y_control[i];
        }
        0
    })
}

/// Original `addWarpPoint` (`warpfiles.c:612`).
pub fn add_warp_point(
    iz: i32,
    x_control: f32,
    y_control: f32,
    x_vector: f32,
    y_vector: f32,
) -> i32 {
    let chunk = 32;
    if add_warpings_if_needed(iz) != 0 {
        return -1;
    }
    let cur = S_CUR_FILE_IND.get();
    S_WARP_FILES.with_borrow_mut(|files| {
        let warp = &mut files[cur as usize].warpings[iz as usize];
        if warp.n_control + 1 > warp.max_vectors {
            let newsize = warp.n_control + chunk;
            warp.x_vector.resize(newsize as usize, 0.);
            warp.y_vector.resize(newsize as usize, 0.);
            warp.x_control.resize(newsize as usize, 0.);
            warp.y_control.resize(newsize as usize, 0.);
            warp.max_vectors = newsize;
        }
        warp.x_vector[warp.n_control as usize] = x_vector;
        warp.x_control[warp.n_control as usize] = x_control;
        warp.y_vector[warp.n_control as usize] = y_vector;
        warp.y_control[warp.n_control as usize] = y_control;
        warp.n_control += 1;
        warp.n_control
    })
}

/// Original `removeWarpPoint` (`warpfiles.c:655`).
pub fn remove_warp_point(iz: i32, index: i32) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow_mut(|files| {
        let file = &mut files[cur as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &mut file.warpings[iz as usize];
        if index < 0 || index >= warp.n_control {
            return 1;
        }
        for i in index + 1..warp.n_control {
            warp.x_control[(i - 1) as usize] = warp.x_control[i as usize];
            warp.y_control[(i - 1) as usize] = warp.y_control[i as usize];
            warp.x_vector[(i - 1) as usize] = warp.x_vector[i as usize];
            warp.y_vector[(i - 1) as usize] = warp.y_vector[i as usize];
        }
        warp.n_control -= 1;
        0
    })
}

/// Original `getLinearTransform` (`warpfiles.c:680`).
pub fn get_linear_transform(iz: i32, xform: &mut [f32], rows: i32) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow(|files| {
        let file = &files[cur as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        xf_copy(&file.warpings[iz as usize].xform, 2, xform, rows as usize);
        0
    })
}

/// Original `getNumWarpPoints` (`warpfiles.c:695`).
pub fn get_num_warp_points(iz: i32, n_control: &mut i32) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow(|files| {
        let file = &files[cur as usize];
        if iz >= file.num_frames {
            return 1;
        }
        *n_control = 0;
        let (izst, iznd) = if iz < 0 {
            (0, file.num_frames - 1)
        } else {
            (iz, iz)
        };
        for i in izst..=iznd {
            let warp = &file.warpings[i as usize];
            *n_control = if *n_control > warp.n_control {
                *n_control
            } else {
                warp.n_control
            };
        }
        0
    })
}

/// Original `getWarpPoints` (`warpfiles.c:719`).
pub fn get_warp_points(
    iz: i32,
    x_control: &mut [f32],
    y_control: &mut [f32],
    x_vector: &mut [f32],
    y_vector: &mut [f32],
) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow(|files| {
        let file = &files[cur as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &file.warpings[iz as usize];
        for i in 0..warp.n_control as usize {
            x_vector[i] = warp.x_vector[i];
            y_vector[i] = warp.y_vector[i];
            x_control[i] = warp.x_control[i];
            y_control[i] = warp.y_control[i];
        }
        0
    })
}

/// Original `getWarpPointArrays` (`warpfiles.c:742`).
///
/// The source hands out four interior pointers into the `Warping`.  Rust
/// cannot lend them out of the thread-local, so the arrays are copied into
/// caller-supplied `Vec`s; every value is the same and nothing in the tree
/// wrote through the returned pointers.
pub fn get_warp_point_arrays(
    iz: i32,
    x_control: &mut Vec<f32>,
    y_control: &mut Vec<f32>,
    x_vector: &mut Vec<f32>,
    y_vector: &mut Vec<f32>,
) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow(|files| {
        let file = &files[cur as usize];
        if iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &file.warpings[iz as usize];
        *x_control = warp.x_control.clone();
        *y_control = warp.y_control.clone();
        *x_vector = warp.x_vector.clone();
        *y_vector = warp.y_vector.clone();
        0
    })
}

/// Original `getWarpGridSize` (`warpfiles.c:763`).
pub fn get_warp_grid_size(iz: i32, nx_max: &mut i32, ny_max: &mut i32, prod_max: &mut i32) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow(|files| {
        let file = &files[cur as usize];
        if iz >= file.num_frames {
            return 1;
        }
        *nx_max = 0;
        *ny_max = 0;
        *prod_max = 0;
        let (izst, iznd) = if iz < 0 {
            (0, file.num_frames - 1)
        } else {
            (iz, iz)
        };
        for i in izst..=iznd {
            let warp = &file.warpings[i as usize];
            if warp.nx_grid > *nx_max {
                *nx_max = warp.nx_grid;
            }
            if warp.ny_grid > *ny_max {
                *ny_max = warp.ny_grid;
            }
            if warp.nx_grid * warp.ny_grid > *prod_max {
                *prod_max = warp.nx_grid * warp.ny_grid;
            }
        }
        0
    })
}

/// Original `getGridParameters` (`warpfiles.c:790`).
#[allow(clippy::too_many_arguments)]
pub fn get_grid_parameters(
    iz: i32,
    nx_grid: &mut i32,
    ny_grid: &mut i32,
    x_start: &mut f32,
    y_start: &mut f32,
    x_interval: &mut f32,
    y_interval: &mut f32,
) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow(|files| {
        let file = &files[cur as usize];
        if iz >= file.num_frames || iz < 0 {
            return 1;
        }
        let warp = &file.warpings[iz as usize];
        *nx_grid = warp.nx_grid;
        *ny_grid = warp.ny_grid;
        *x_start = warp.x_start;
        *y_start = warp.y_start;
        *x_interval = warp.x_interval;
        *y_interval = warp.y_interval;
        0
    })
}

/// Original `setGridSizeToMake` (`warpfiles.c:813`).
#[allow(clippy::too_many_arguments)]
pub fn set_grid_size_to_make(
    iz: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_start: f32,
    y_start: f32,
    x_interval: f32,
    y_interval: f32,
) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow_mut(|files| {
        let file = &mut files[cur as usize];
        if file.flags & WARP_CONTROL_PTS == 0 || iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &mut file.warpings[iz as usize];
        warp.nx_grid = nx_grid;
        warp.ny_grid = ny_grid;
        warp.x_start = x_start;
        warp.y_start = y_start;
        warp.x_interval = x_interval;
        warp.y_interval = y_interval;
        0
    })
}

/// Original `controlPointRange` (`warpfiles.c:836`).
pub fn control_point_range(
    iz: i32,
    xmin: &mut f32,
    xmax: &mut f32,
    ymin: &mut f32,
    ymax: &mut f32,
) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    S_WARP_FILES.with_borrow(|files| {
        let file = &files[cur as usize];
        if file.flags & WARP_CONTROL_PTS == 0 || iz < 0 || iz >= file.num_frames {
            return 1;
        }
        let warp = &file.warpings[iz as usize];
        if warp.n_control == 0 {
            return 1;
        }
        *xmin = warp.x_control[0];
        *xmax = *xmin;
        *ymin = warp.y_control[0];
        *ymax = *ymin;
        for i in 1..warp.n_control as usize {
            if warp.x_control[i] < *xmin {
                *xmin = warp.x_control[i];
            }
            if warp.y_control[i] < *ymin {
                *ymin = warp.y_control[i];
            }
            if warp.x_control[i] > *xmax {
                *xmax = warp.x_control[i];
            }
            if warp.y_control[i] > *ymax {
                *ymax = warp.y_control[i];
            }
        }
        0
    })
}

/// Original `controlPointSpacing` (`warpfiles.c:866`).
pub fn control_point_spacing(iz: i32, percentile: f32, spacing: &mut f32) -> i32 {
    let mut percentile = percentile;
    if percentile < 0. {
        percentile = DEFAULT_PERCENTILE;
    }
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    let n_control = S_WARP_FILES.with_borrow(|files| {
        let file = &files[cur as usize];
        if file.flags & WARP_CONTROL_PTS == 0 || iz < 0 || iz >= file.num_frames || percentile > 1.0
        {
            return -1;
        }
        file.warpings[iz as usize].n_control
    });
    if n_control < 0 {
        return 1;
    }
    if n_control < 2 {
        return 1;
    }
    let match_stat = S_WARP_FILES
        .with_borrow(|files| points_match_stat_array(&files[cur as usize].warpings[iz as usize]));
    if match_stat != 0
        && iz == S_LAST_Z_FOR_SPACING.get()
        && (percentile as f64 - S_LAST_PERCENTILE.get() as f64).abs() < 1.0e-6
    {
        *spacing = S_LAST_SPACING.get();
        return 0;
    }

    let mut nearest = vec![0.0_f32; n_control as usize];
    S_WARP_FILES.with_borrow(|files| {
        let warp = &files[cur as usize].warpings[iz as usize];
        for i in 0..warp.n_control as usize {
            let mut minsq = 1.0e30_f32;
            for j in 0..warp.n_control as usize {
                if i == j {
                    continue;
                }
                let dx = warp.x_control[i] - warp.x_control[j];
                let dxsq = dx * dx;
                if dxsq < minsq {
                    let dy = warp.y_control[i] - warp.y_control[j];
                    let dysq = dy * dy;
                    if dysq + dxsq < minsq {
                        minsq = dxsq + dysq;
                    }
                }
            }
            nearest[i] = minsq;
        }
    });
    let mut sel = (percentile as f64 * n_control as f64 + 0.5).floor() as i32;
    sel = if n_control - 1 < sel {
        n_control - 1
    } else {
        sel
    };
    rs_sort_floats(&mut nearest, n_control);
    *spacing = (nearest[sel as usize] as f64).sqrt() as f32;
    if match_stat != 0 {
        S_LAST_Z_FOR_SPACING.set(iz);
        S_LAST_PERCENTILE.set(percentile);
        S_LAST_SPACING.set(*spacing);
    } else {
        S_LAST_Z_FOR_SPACING.set(-1);
    }
    0
}

/// Original `gridSizeFromSpacing` (`warpfiles.c:932`).
///
/// `controlPointRange`'s return value is not tested by the source, and the
/// range fetch is hoisted above the `warp` lookup here because the lookup is
/// only a pointer in the C and the two do not interact.
pub fn grid_size_from_spacing(iz: i32, percentile: f32, factor: f32, full_extent: i32) -> i32 {
    let mut factor = factor;
    if factor < 0. {
        factor = DEFAULT_FACTOR;
    }
    let mut spacing = 0.0_f32;
    if control_point_spacing(iz, percentile, &mut spacing) != 0 || factor > 2. || factor < 0.05 {
        return 1;
    }
    spacing *= factor;
    let cur = S_CUR_FILE_IND.get();
    let (xmin, xmax, ymin, ymax);
    if full_extent != 0 {
        let (nx, ny) =
            S_WARP_FILES.with_borrow(|files| (files[cur as usize].nx, files[cur as usize].ny));
        xmin = (spacing as f64 / 10.) as f32;
        xmax = nx as f32 - xmin;
        ymin = (spacing as f64 / 10.) as f32;
        ymax = ny as f32 - ymin;
    } else {
        let (mut a, mut b, mut c, mut d) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let _ = control_point_range(iz, &mut a, &mut b, &mut c, &mut d);
        xmin = a;
        xmax = b;
        ymin = c;
        ymax = d;
    }

    S_WARP_FILES.with_borrow_mut(|files| {
        let warp = &mut files[cur as usize].warpings[iz as usize];
        // `ceil(1 + (xmax - xmin) / spacing)`: the quotient and the `1 +` are
        // both `float` in the source and only `ceil` promotes them.
        warp.nx_grid = (((1. + (xmax - xmin) / spacing) as f64).ceil()) as i32;
        warp.nx_grid = if 2 > warp.nx_grid { 2 } else { warp.nx_grid };
        warp.x_interval = (xmax - xmin) / (warp.nx_grid - 1) as f32;
        warp.ny_grid = (((1. + (ymax - ymin) / spacing) as f64).ceil()) as i32;
        warp.ny_grid = if 2 > warp.ny_grid { 2 } else { warp.ny_grid };
        warp.y_interval = (ymax - ymin) / (warp.ny_grid - 1) as f32;
        warp.x_start = xmin;
        warp.y_start = ymin;
    });
    0
}

/// Original static `freeStaticArrays` (`warpfiles.c:960`).
pub fn free_static_arrays() {
    S_DPOINTS.with_borrow_mut(|v| *v = Vec::new());
    S_GRIDX.with_borrow_mut(|v| *v = Vec::new());
    S_GRIDY.with_borrow_mut(|v| *v = Vec::new());
    S_SOLVED.with_borrow_mut(|v| *v = Vec::new());
    // `delaunay_destroy(sDelau)` and `nnai_destroy(sNninterp)`: the
    // triangulation lives inside the interpolator now, so dropping the one
    // option frees both.
    S_NNINTERP.with_borrow_mut(|v| *v = None);
    S_LAST_Z_FOR_SPACING.set(-1);
    crate::imod::libwarp::hull::hull_cleanup();
    crate::imod::libwarp::hull::STORAGE
        .with_borrow_mut(crate::imod::libwarp::hull_ch::hull_ch_cleanup);
}

/// Original static `pointsMatchStatArray` (`warpfiles.c:977`).
pub fn points_match_stat_array(warp: &Warping) -> i32 {
    S_DPOINTS.with_borrow(|dpoints| {
        if dpoints.is_empty() || S_LAST_NUM_CONT.get() != warp.n_control {
            return 0;
        }
        for ix in 0..warp.n_control as usize {
            if (dpoints[ix].x - warp.x_control[ix] as f64).abs() > 1.0e-3
                || (dpoints[ix].y - warp.y_control[ix] as f64).abs() > 1.0e-3
            {
                return 0;
            }
        }
        1
    })
}

/// Original `getWarpGrid` (`warpfiles.c:1002`).
#[allow(clippy::too_many_arguments)]
pub fn get_warp_grid(
    iz: i32,
    nx_grid: &mut i32,
    ny_grid: &mut i32,
    x_start: &mut f32,
    y_start: &mut f32,
    x_interval: &mut f32,
    y_interval: &mut f32,
    dx_grid: &mut [f32],
    dy_grid: &mut [f32],
    xdim: i32,
) -> i32 {
    let mut xdim = xdim;
    let mut need_new = 1;
    let wall_top = crate::imod::libcfshr::b3dutil::wall_time();
    let mut wall_start;
    let prune_crit = [S_HEIGHT_BASE_CRIT.get(), S_AREA_FRACTION_CRIT.get()];

    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    // Every field the source reads through `warp` is copied out here so that no
    // borrow of the thread-local is held across `nnai_build`, which can reach
    // back into this module through nothing but is safer read once.
    let (flags, num_frames) = S_WARP_FILES
        .with_borrow(|files| (files[cur as usize].flags, files[cur as usize].num_frames));
    if iz < 0 || iz >= num_frames {
        return 1;
    }
    let (w_nx_grid, w_ny_grid, w_x_start, w_y_start, w_x_interval, w_y_interval, w_n_control) =
        S_WARP_FILES.with_borrow(|files| {
            let warp = &files[cur as usize].warpings[iz as usize];
            (
                warp.nx_grid,
                warp.ny_grid,
                warp.x_start,
                warp.y_start,
                warp.x_interval,
                warp.y_interval,
                warp.n_control,
            )
        });
    if w_nx_grid == 0 || w_ny_grid == 0 {
        return 1;
    }
    if xdim <= 0 {
        xdim = w_nx_grid;
    }
    if flags & WARP_CONTROL_PTS == 0 {
        /* Return existing grid */
        S_WARP_FILES.with_borrow(|files| {
            let warp = &files[cur as usize].warpings[iz as usize];
            for iy in 0..warp.ny_grid {
                for ix in 0..warp.nx_grid {
                    dx_grid[(ix + iy * xdim) as usize] =
                        warp.x_vector[(ix + iy * warp.nx_grid) as usize];
                    dy_grid[(ix + iy * xdim) as usize] =
                        warp.y_vector[(ix + iy * warp.nx_grid) as usize];
                }
            }
        });
    } else {
        /* Return grid based on control points */
        if w_n_control < 3 {
            return 1;
        }

        /* See if the points and grid match previous analysis */
        let have_all = S_DPOINTS.with_borrow(|v| !v.is_empty())
            && S_GRIDX.with_borrow(|v| !v.is_empty())
            && S_GRIDY.with_borrow(|v| !v.is_empty())
            && S_SOLVED.with_borrow(|v| !v.is_empty())
            && S_NNINTERP.with_borrow(|v| v.is_some());
        if have_all
            && S_LAST_NUM_CONT.get() == w_n_control
            && w_nx_grid == S_LAST_NX_GRID.get()
            && w_ny_grid == S_LAST_NY_GRID.get()
            && w_x_start == S_LAST_X_START.get()
            && w_y_start == S_LAST_Y_START.get()
            && w_x_interval == S_LAST_X_INTERVAL.get()
            && w_y_interval == S_LAST_Y_INTERVAL.get()
            && xdim == S_LAST_XDIM.get()
        {
            need_new = 0;
            if S_WARP_FILES.with_borrow(|files| {
                points_match_stat_array(&files[cur as usize].warpings[iz as usize])
            }) == 0
            {
                need_new = 1;
            }
        }

        /* Determine size of grid within the range */
        let ngrid = xdim * w_ny_grid;
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let _ = control_point_range(iz, &mut xmin, &mut xmax, &mut ymin, &mut ymax);
        // `0.1` is a double literal, so the whole quotient is evaluated in
        // double before `ceil`/`floor`.
        let mut off_x_solve =
            ((xmin as f64 - 0.1 - w_x_start as f64) / w_x_interval as f64).ceil() as i32;
        off_x_solve = if 0 > off_x_solve { 0 } else { off_x_solve };
        let mut last =
            ((xmax as f64 + 0.1 - w_x_start as f64) / w_x_interval as f64).floor() as i32;
        last = if w_nx_grid - 1 < last {
            w_nx_grid - 1
        } else {
            last
        };
        let nx_solve = last + 1 - off_x_solve;
        let mut off_y_solve =
            ((ymin as f64 - 0.1 - w_y_start as f64) / w_y_interval as f64).ceil() as i32;
        off_y_solve = if 0 > off_y_solve { 0 } else { off_y_solve };
        last = ((ymax as f64 + 0.1 - w_y_start as f64) / w_y_interval as f64).floor() as i32;
        last = if w_ny_grid - 1 < last {
            w_ny_grid - 1
        } else {
            last
        };
        let ny_solve = last + 1 - off_y_solve;
        let nsolve = nx_solve * ny_solve;
        if nsolve == 0 {
            return 1;
        }

        NN_VERBOSE.set(0);

        if need_new != 0 {
            free_static_arrays();
            S_DPOINTS.with_borrow_mut(|v| *v = vec![Point::default(); w_n_control as usize]);
            S_GRIDX.with_borrow_mut(|v| *v = vec![0.0_f64; nsolve as usize]);
            S_GRIDY.with_borrow_mut(|v| *v = vec![0.0_f64; nsolve as usize]);
            S_SOLVED.with_borrow_mut(|v| *v = vec![0_u8; (6 * ngrid) as usize]);
        }
        let mut dxyout = vec![0.0_f64; nsolve as usize];
        let mut dxyin = vec![0.0_f64; w_n_control as usize];

        S_LAST_NUM_CONT.set(w_n_control);
        S_LAST_NX_GRID.set(w_nx_grid);
        S_LAST_NY_GRID.set(w_ny_grid);
        S_LAST_X_START.set(w_x_start);
        S_LAST_Y_START.set(w_y_start);
        S_LAST_X_INTERVAL.set(w_x_interval);
        S_LAST_Y_INTERVAL.set(w_y_interval);
        S_LAST_XDIM.set(xdim);

        /* Zero out the whole grid in case extrapolation fails and to keep valgrind happy */
        for ix in 0..ngrid as usize {
            dx_grid[ix] = 0.;
            dy_grid[ix] = 0.;
        }

        /* Load the control points and the grid of points to solve at */
        if need_new != 0 {
            S_WARP_FILES.with_borrow(|files| {
                let warp = &files[cur as usize].warpings[iz as usize];
                S_DPOINTS.with_borrow_mut(|dpoints| {
                    for ix in 0..warp.n_control as usize {
                        dpoints[ix].x = warp.x_control[ix] as f64;
                        dpoints[ix].y = warp.y_control[ix] as f64;
                        dpoints[ix].z = 0.;
                    }
                });
            });
            S_GRIDX.with_borrow_mut(|gx| {
                S_GRIDY.with_borrow_mut(|gy| {
                    for iy in 0..ny_solve {
                        for ix in 0..nx_solve {
                            gx[(ix + iy * nx_solve) as usize] =
                                (w_x_start + (ix + off_x_solve) as f32 * w_x_interval) as f64;
                            gy[(ix + iy * nx_solve) as usize] =
                                (w_y_start + (iy + off_y_solve) as f32 * w_y_interval) as f64;
                        }
                    }
                });
            });
        }

        if need_new != 0 {
            wall_start = crate::imod::libcfshr::b3dutil::wall_time();
            let delau = S_DPOINTS.with_borrow(|dpoints| {
                delaunay_build(
                    w_n_control,
                    dpoints,
                    0,
                    None,
                    S_MIN_NUM_FOR_PRUNING.get(),
                    Some(&prune_crit),
                )
            });
            if S_TIME_OUTPUT.get() != 0 {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "Delaunay time %.3f\n",
                        &[CArg::Dbl(
                            1000. * (crate::imod::libcfshr::b3dutil::wall_time() - wall_start),
                        )],
                    )
                    .as_bytes(),
                );
            }
            wall_start = crate::imod::libcfshr::b3dutil::wall_time();
            if let Some(delau) = delau {
                let interp = S_GRIDX
                    .with_borrow(|gx| S_GRIDY.with_borrow(|gy| nnai_build(delau, nsolve, gx, gy)));
                S_NNINTERP.with_borrow_mut(|v| *v = interp);
            }
            if S_NNINTERP.with_borrow(|v| v.is_none()) {
                free_static_arrays();
                return 1;
            }
            if S_TIME_OUTPUT.get() != 0 {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "nnai_build time %.3f\n",
                        &[CArg::Dbl(
                            1000. * (crate::imod::libcfshr::b3dutil::wall_time() - wall_start),
                        )],
                    )
                    .as_bytes(),
                );
            }
        }
        wall_start = crate::imod::libcfshr::b3dutil::wall_time();

        /* This is needed to ensure Nan's outside the convex hull */
        S_NNINTERP.with_borrow_mut(|v| nnai_setwmin(v.as_mut().unwrap(), 0.));

        /* Interpolate DX values and store them in dxGrid, set solved array */
        S_WARP_FILES.with_borrow(|files| {
            let warp = &files[cur as usize].warpings[iz as usize];
            for ix in 0..warp.n_control as usize {
                dxyin[ix] = warp.x_vector[ix] as f64;
            }
        });
        S_NNINTERP.with_borrow_mut(|v| nnai_interpolate(v.as_mut().unwrap(), &dxyin, &mut dxyout));
        S_SOLVED.with_borrow_mut(|solved| {
            for ix in 0..ngrid as usize {
                solved[ix] = 0;
            }
            for iy in 0..ny_solve {
                for ix in 0..nx_solve {
                    let ind = (ix + iy * nx_solve) as usize;

                    /* config.h defines isnan as _isnan for WIN32 */
                    if !dxyout[ind].is_nan() {
                        let ind2 = (ix + off_x_solve + (iy + off_y_solve) * xdim) as usize;
                        dx_grid[ind2] = dxyout[ind] as f32;
                        solved[ind2] = 1;
                    }
                }
            }
        });

        /* Load and interpolate DY values and store them in dyGrid */
        S_WARP_FILES.with_borrow(|files| {
            let warp = &files[cur as usize].warpings[iz as usize];
            for ix in 0..warp.n_control as usize {
                dxyin[ix] = warp.y_vector[ix] as f64;
            }
        });
        S_NNINTERP.with_borrow_mut(|v| nnai_interpolate(v.as_mut().unwrap(), &dxyin, &mut dxyout));
        if S_TIME_OUTPUT.get() != 0 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "nnai_interpolate time %.3f\n",
                    &[CArg::Dbl(
                        1000. * (crate::imod::libcfshr::b3dutil::wall_time() - wall_start),
                    )],
                )
                .as_bytes(),
            );
        }
        wall_start = crate::imod::libcfshr::b3dutil::wall_time();
        S_SOLVED.with_borrow(|solved| {
            for iy in 0..ny_solve {
                for ix in 0..nx_solve {
                    let ind2 = (ix + off_x_solve + (iy + off_y_solve) * xdim) as usize;
                    if solved[ind2] != 0 {
                        dy_grid[ind2] = dxyout[(ix + iy * nx_solve) as usize] as f32;
                    }
                }
            }
        });

        /* Fill in outside the hull. If it fails, mark as non-resumable */
        let extrap = S_SOLVED.with_borrow_mut(|solved| {
            extrapolate_grid(
                dx_grid,
                dy_grid,
                solved,
                xdim,
                w_nx_grid,
                w_ny_grid,
                w_x_interval,
                w_y_interval,
                1 - need_new,
            )
        });
        if extrap != 0 {
            S_LAST_NUM_CONT.set(0);
        }
        if S_TIME_OUTPUT.get() != 0 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "extrapolate time %.3f\n",
                    &[CArg::Dbl(
                        1000. * (crate::imod::libcfshr::b3dutil::wall_time() - wall_start),
                    )],
                )
                .as_bytes(),
            );
        }
        dxyin.clear();
        dxyout.clear();
    }

    /* Return grid parameters in either case */
    *nx_grid = w_nx_grid;
    *ny_grid = w_ny_grid;
    *x_start = w_x_start;
    *y_start = w_y_start;
    *x_interval = w_x_interval;
    *y_interval = w_y_interval;
    if S_TIME_OUTPUT.get() != 0 {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Total time in getWarpGrid %.3f\n",
                &[CArg::Dbl(
                    1000. * (crate::imod::libcfshr::b3dutil::wall_time() - wall_top),
                )],
            )
            .as_bytes(),
        );
    }
    0
}

/// Original `separateLinearTransform` (`warpfiles.c:1261`).
///
/// The source passes `warp->xVector`/`warp->yVector` to `extractLinearXform` as
/// both the input and the output vectors; the input copy is taken here
/// instead, which is what that aliasing amounted to.
pub fn separate_linear_transform(iz: i32) -> i32 {
    let cur = S_CUR_FILE_IND.get();
    if cur < 0 {
        return 1;
    }
    let (flags, num_frames, file_nx, file_ny) = S_WARP_FILES.with_borrow(|files| {
        (
            files[cur as usize].flags,
            files[cur as usize].num_frames,
            files[cur as usize].nx,
            files[cur as usize].ny,
        )
    });
    if iz < 0 || iz >= num_frames || flags & WARP_INVERSE == 0 {
        return 1;
    }

    let x_point: Vec<f32>;
    let y_point: Vec<f32>;
    let n_points;

    /* For control points, set the pointers */
    if flags & WARP_CONTROL_PTS != 0 {
        let n =
            S_WARP_FILES.with_borrow(|files| files[cur as usize].warpings[iz as usize].n_control);
        if n < 3 {
            return 1;
        }
        n_points = n;
        (x_point, y_point) = S_WARP_FILES.with_borrow(|files| {
            let warp = &files[cur as usize].warpings[iz as usize];
            (warp.x_control.clone(), warp.y_control.clone())
        });
    } else {
        /* For a grid, create arrays and load them with the points */
        let (nxg, nyg, xs, ys, xi, yi) = S_WARP_FILES.with_borrow(|files| {
            let warp = &files[cur as usize].warpings[iz as usize];
            (
                warp.nx_grid,
                warp.ny_grid,
                warp.x_start,
                warp.y_start,
                warp.x_interval,
                warp.y_interval,
            )
        });
        n_points = nxg * nyg;
        if n_points < 3 {
            return 1;
        }
        let mut xp = vec![0.0_f32; n_points as usize];
        let mut yp = vec![0.0_f32; n_points as usize];
        for iy in 0..nyg {
            for ix in 0..nxg {
                xp[(ix + iy * nxg) as usize] = xs + ix as f32 * xi;
                yp[(ix + iy * nxg) as usize] = ys + iy as f32 * yi;
            }
        }
        x_point = xp;
        y_point = yp;
    }

    let mut xfinv = [0.0_f32; 6];
    let ix = S_WARP_FILES.with_borrow_mut(|files| {
        let warp = &mut files[cur as usize].warpings[iz as usize];
        let xv = warp.x_vector[..n_points as usize].to_vec();
        let yv = warp.y_vector[..n_points as usize].to_vec();
        extract_linear_xform(
            &x_point,
            &y_point,
            &xv,
            &yv,
            n_points,
            file_nx as f32 / 2.,
            file_ny as f32 / 2.,
            &mut warp.x_vector,
            &mut warp.y_vector,
            &mut xfinv,
            2,
        )
    });
    if ix == 0 {
        S_WARP_FILES.with_borrow_mut(|files| {
            let warp = &mut files[cur as usize].warpings[iz as usize];
            let mut product = [0.0_f32; 6];
            xf_mult(&warp.xform, &xfinv, &mut product, 2);
            warp.xform = product;
        });
    }
    ix
}

/// Original static `readLineOfValues` (`warpfiles.c:1330`).
///
/// `strtok(str, ", \t")` splits on comma, space and tab only, and `strtod`
/// must consume the whole token (`if (*endPtr != 0x00) return -4;`), so the
/// scan is written out rather than replaced with `str::parse` on whitespace
/// (`NATIVE.md` §2).
pub fn read_line_of_values(
    fp: &mut ImodFile,
    line: &mut [u8],
    limit: i32,
    values: &mut LineValues<'_>,
    floats: i32,
    num_to_get: &mut i32,
    max_vals: i32,
) -> i32 {
    loop {
        let length = fgetline(fp, line, limit);
        if length == 0 {
            continue;
        }
        if length == -1 || length == -2 {
            return length;
        }
        let mut num_got = 0;
        let text_len = if length < 0 {
            (-length - 2) as usize
        } else {
            length as usize
        };
        let text = &line[..text_len];
        let mut pos = 0usize;
        loop {
            while pos < text.len() && (text[pos] == b',' || text[pos] == b' ' || text[pos] == b'\t')
            {
                pos += 1;
            }
            if pos >= text.len() {
                break;
            }
            let start = pos;
            while pos < text.len()
                && !(text[pos] == b',' || text[pos] == b' ' || text[pos] == b'\t')
            {
                pos += 1;
            }
            let token = &text[start..pos];
            if num_got >= max_vals {
                return -3;
            }
            let s = String::from_utf8_lossy(token);
            let s = s.as_ref();
            // `strtod`/`strtol` set `endPtr` past what they consumed and the
            // source rejects the token unless that is the whole of it, so a
            // parse that consumes everything is the same test.  Two accepts
            // `strtod` has that `f64::from_str` does not — a hex-float literal
            // and an out-of-range integer that `strtol` clamps — cannot occur
            // in a warp file, and `floats` is 1 at every call site in the
            // source.
            if floats != 0 {
                let LineValues::Floats(array) = values else {
                    return -4;
                };
                let Ok(v) = s.parse::<f64>() else {
                    return -4;
                };
                array[num_got as usize] = v as f32;
            } else {
                let LineValues::Ints(array) = values else {
                    return -4;
                };
                let Ok(v) = s.parse::<i64>() else {
                    return -4;
                };
                array[num_got as usize] = v as i32;
            }
            num_got += 1;
        }
        if num_got == 0 {
            continue;
        }

        if *num_to_get == 0 {
            *num_to_get = num_got;
        }
        if num_got < *num_to_get {
            return -5;
        }
        return if length < 0 { 1 } else { 0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_file_lifecycle_preserves_packed_source_vectors() {
        warp_files_done();
        let index = new_warp_file(8, 6, 2, 1.5, WARP_INVERSE);
        assert_eq!(index, 0);
        let dx = [1., 2., 3., 4.];
        let dy = [-1., -2., -3., -4.];
        assert_eq!(set_warp_grid(0, 2, 2, 0.5, 1.5, 4., 3., &dx, &dy, 2), 0);
        let (mut nx, mut ny, mut nz, mut control) = (0, 0, 0, 0);
        assert_eq!(
            get_warp_file_size(&mut nx, &mut ny, &mut nz, &mut control),
            0
        );
        assert_eq!((nx, ny, nz, control), (8, 6, 1, 0));
        let (mut xs, mut ys, mut xi, mut yi) = (0., 0., 0., 0.);
        let mut outx = [0.; 4];
        let mut outy = [0.; 4];
        assert_eq!(
            get_warp_grid(
                0, &mut nx, &mut ny, &mut xs, &mut ys, &mut xi, &mut yi, &mut outx, &mut outy, 2
            ),
            0
        );
        assert_eq!((nx, ny, xs, ys, xi, yi), (2, 2, 0.5, 1.5, 4., 3.));
        assert_eq!((outx, outy), (dx, dy));
        warp_files_done();
    }

    #[test]
    fn version_three_warp_file_round_trips_source_text_format() {
        warp_files_done();
        new_warp_file(8, 6, 2, 1.5, WARP_INVERSE);
        let dx = [1., 2., 3., 4.];
        let dy = [-1., -2., -3., -4.];
        set_warp_grid(0, 2, 2, 0.5, 1.5, 4., 3., &dx, &dy, 2);
        let path = std::env::temp_dir().join(format!(
            "imod-rs-warpfiles-roundtrip-{}.xf",
            std::process::id()
        ));
        let name = path.to_string_lossy().into_owned();
        assert_eq!(write_warp_file(&name, 1), 0);
        // `%f` is six decimals; `{}` would have written `1.5`.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("3\n8 6 1 2 1.500000 1\n"));
        warp_files_done();
        let (mut nx, mut ny, mut nz, mut bin, mut pixel, mut version, mut flags) =
            (0, 0, 0, 0, 0., 0, 0);
        assert_eq!(
            read_warp_file(
                &name,
                &mut nx,
                &mut ny,
                &mut nz,
                &mut bin,
                &mut pixel,
                &mut version,
                &mut flags
            ),
            0
        );
        assert_eq!(
            (nx, ny, nz, bin, pixel, version, flags),
            (8, 6, 1, 2, 1.5, 3, WARP_INVERSE)
        );
        let (mut xs, mut ys, mut xi, mut yi) = (0., 0., 0., 0.);
        let (mut ox, mut oy) = ([0.; 4], [0.; 4]);
        assert_eq!(
            get_warp_grid(
                0, &mut nx, &mut ny, &mut xs, &mut ys, &mut xi, &mut yi, &mut ox, &mut oy, 2
            ),
            0
        );
        assert_eq!((ox, oy), (dx, dy));
        let _ = std::fs::remove_file(path);
        warp_files_done();
    }

    #[test]
    fn control_points_produce_a_natural_neighbour_warp_grid() {
        warp_files_done();
        new_warp_file(2, 2, 1, 1.0, WARP_CONTROL_PTS);
        let xc = [0.0_f32, 1.0, 0.0];
        let yc = [0.0_f32, 0.0, 1.0];
        let dx = [0.0_f32, 1.0, 2.0];
        let dy = [3.0_f32, 3.0, 3.0];
        assert_eq!(set_warp_points(0, 3, &xc, &yc, &dx, &dy), 0);
        assert_eq!(set_grid_size_to_make(0, 2, 2, 0.0, 0.0, 1.0, 1.0), 0);
        let (mut nx, mut ny) = (0, 0);
        let (mut xs, mut ys, mut xi, mut yi) = (0.0, 0.0, 0.0, 0.0);
        let mut outx = [f32::NAN; 4];
        let mut outy = [f32::NAN; 4];
        assert_eq!(
            get_warp_grid(
                0, &mut nx, &mut ny, &mut xs, &mut ys, &mut xi, &mut yi, &mut outx, &mut outy, 2
            ),
            0
        );
        assert_eq!((nx, ny, xs, ys, xi, yi), (2, 2, 0.0, 0.0, 1.0, 1.0));
        assert_eq!(&outx[..3], &[0.0, 1.0, 2.0]);
        assert_eq!(&outy[..3], &[3.0, 3.0, 3.0]);
        warp_files_done();
    }
}
