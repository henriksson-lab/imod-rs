//! Translation of `IMOD/imodutil/patch2imod.c`: converts a patch
//! displacement file (or a warping transformation file) to an IMOD model of
//! two-point vectors.

use std::io::{Seek, SeekFrom, Write};

use crate::imod::clip::clip::{ScanArg, atoi, sscanf};
use crate::imod::libcfshr::b3dutil::{
    ImodFile, fgetline, imod_backup_file, imod_copyright, imod_version,
};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix, strtol};
use crate::imod::libimod::icont::imod_contours_new;
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OPEN, IMODF_FLIPYZ, Imod, Ipoint, imod_delete, imod_new, imod_new_object,
};
use crate::imod::libimod::imodel_files::imod_write;
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_MCOLOR, IMOD_OBJFLAG_THICK_CONT, IMOD_OBJFLAG_TIME, IMOD_OBJFLAG_USE_VALUE,
    IOBJ_EXSIZE, IOBJ_SYMF_ARROW, imod_object_set_name,
};
use crate::imod::libimod::istore::{
    GEN_STORE_FLOAT, GEN_STORE_MINMAX1, GEN_STORE_VALUE1, Istore, istore_add_min_max, istore_insert,
};
use crate::imod::libimod::iview::imod_view_model_new;
use crate::imod::libwarp::warpfiles::{
    get_num_warp_points, get_warp_file_size, get_warp_grid, get_warp_grid_size,
    get_warp_point_arrays, read_warp_file,
};

/// `#define P2I_NO_FLIP 1` (`patch2imod.c:17`).
const P2I_NO_FLIP: i32 = 1;
/// `#define P2I_IGNORE_ZERO 2`.
const P2I_IGNORE_ZERO: i32 = 2;
/// `#define P2I_COUNT_LINES 4`.
const P2I_COUNT_LINES: i32 = 4;
/// `#define P2I_DISPLAY_VALUES 8`.
const P2I_DISPLAY_VALUES: i32 = 8;
/// `#define P2I_READ_WARP 16`.
const P2I_READ_WARP: i32 = 16;
/// `#define P2I_TIMES_FOR_Z 32`.
const P2I_TIMES_FOR_Z: i32 = 32;

/// `#define MAX_VALUE_COLS 6` (`patch2imod.c:24`).
const MAX_VALUE_COLS: usize = 6;
/// `#define DEFAULT_SCALE 10.0`.
const DEFAULT_SCALE: f64 = 10.0;

/// `iobj.h:76-77`: `IOBJ_SYM_CIRCLE 0`, `IOBJ_SYM_NONE 1`.
const IOBJ_SYM_CIRCLE: u8 = 0;
const IOBJ_SYM_NONE: u8 = 1;
/// `imodel.h:218`: `WORLD_MOVE_ALL_CLIP (1l << 12)`.
const WORLD_MOVE_ALL_CLIP: u32 = 1 << 12;

/// Original: `usage` (`patch2imod.c:30`, static).
fn usage(prog: &str) -> ! {
    imod_version(Some(prog));
    imod_copyright();
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(format!("Usage: {prog} [options] patch_file output_model\n").as_bytes());
    let _ = out.write_all(b"Options:\n");
    let _ = out.write_all(
        format!("\t-s #\tScale vectors by given value (default {DEFAULT_SCALE:.1})\n").as_bytes(),
    );
    let _ = out.write_all(b"\t-f\tDo NOT flip the Y and Z coordinates\n");
    let _ = out.write_all(b"\t-n name\tAdd given name to model object\n");
    let _ = out.write_all(b"\t-c #\tSet up clipping planes enclosing area of given size\n");
    let _ = out.write_all(b"\t-d #\tSet flag to display values in false color in 3dmod\n");
    let _ = out.write_all(
        format!(
            "\t-v #[,#]\tValue column numbers (1-{MAX_VALUE_COLS}) to store as value and value2\n"
        )
        .as_bytes(),
    );
    let _ =
        out.write_all(b"\t-z\tIgnore zero values when using SD to limit stored maximum value\n");
    let _ = out.write_all(b"\t-l\tUse all lines in file; do not get line count from first line\n");
    let _ = out.write_all(b"\t-w\tRead input file as a warping transformation file\n");
    let _ = out.write_all(b"\t-t\tGive each contour a time equal to its Z value plus 1\n");

    crate::imod::libcfshr::b3dutil::exit(1)
}

/// Original: `main` (`patch2imod.c:51`).
///
/// Deviation note: where the source reads `argv[++i]` past the last argument
/// (`-s`, `-n`, `-c` or `-v` given last) it dereferences the terminating
/// NULL and crashes.  Here a missing value reads as the empty string, which
/// assigns what `sscanf`/`atoi`/`strdup` of an empty string would, and the
/// run then ends in the source's own "Wrong # of arguments" exit.
pub fn patch2imod() {
    let argv: Vec<String> = crate::imod::libcfshr::b3dutil::program_args();
    let argc = argv.len();
    let mut i: usize;
    let mut fin: Option<ImodFile> = None;
    let mut scale: f32 = 10.0;
    let mut clip_size = 0_i32;
    let mut flags = 0_i32;
    let (mut nx_warp, mut ny_warp, mut nz_warp, mut warp_flags, mut version, ind_warp, mut ibin) =
        (0_i32, 0_i32, 0_i32, 0_i32, 0_i32, 0_i32, 0_i32);
    let mut val_col_out = -1_i32;
    let mut warp_pixel = 0.0_f32;
    let mut name: Option<Vec<u8>> = None;
    let _ = ind_warp;

    /* This name is hard-coded because of the script wrapper needed in Vista */
    let progname = "patch2imod";

    if argc < 3 {
        usage(progname);
    }

    setExitPrefix(b"ERROR: patch2imod - ");

    let arg_at = |k: usize| -> &str { argv.get(k).map(String::as_str).unwrap_or("") };
    i = 1;
    while i < argc {
        let arg = argv[i].as_bytes();
        if arg.first() == Some(&b'-') {
            match arg.get(1).copied().unwrap_or(0) {
                b's' => {
                    i += 1;
                    sscanf(arg_at(i), "%f", &mut [ScanArg::Flt(&mut scale)]);
                }

                b'n' => {
                    i += 1;
                    name = Some(arg_at(i).as_bytes().to_vec());
                }

                b'c' => {
                    i += 1;
                    clip_size = atoi(arg_at(i));
                }

                b'f' => {
                    flags |= P2I_NO_FLIP;
                }

                b'z' => {
                    flags |= P2I_IGNORE_ZERO;
                }

                b'l' => {
                    flags |= P2I_COUNT_LINES;
                }

                b'd' => {
                    flags |= P2I_DISPLAY_VALUES;
                }

                b'w' => {
                    flags |= P2I_READ_WARP;
                }

                b't' => {
                    flags |= P2I_TIMES_FOR_Z;
                }

                b'v' => {
                    i += 1;
                    val_col_out = atoi(arg_at(i));
                    if val_col_out == 0 || val_col_out < -(MAX_VALUE_COLS as i32) {
                        exit_error(
                            format!(
                                "Value column # or ID must be positive and between -1 and -{}",
                                MAX_VALUE_COLS
                            )
                            .as_bytes(),
                        );
                    }
                }

                _ => {
                    exit_error(format!("Illegal argument {}", argv[i]).as_bytes());
                }
            }
        } else {
            break;
        }
        i += 1;
    }
    if i as isize > argc as isize - 2 {
        let _ = ImodFile::Stdout.write_all(b"ERROR: patch2imod - Wrong # of arguments\n");
        usage(progname);
    }

    if (flags & P2I_COUNT_LINES) != 0 && val_col_out > 0 {
        exit_error(b"You cannot enter a column ID if there is no first line with data count");
    }

    if (flags & P2I_READ_WARP) != 0 {
        let ind_warp = read_warp_file(
            &argv[i],
            &mut nx_warp,
            &mut ny_warp,
            &mut nz_warp,
            &mut ibin,
            &mut warp_pixel,
            &mut version,
            &mut warp_flags,
        );
        i += 1;
        if ind_warp < 0 {
            i -= 1;
            exit_error(format!("Reading {} as a warping file", argv[i]).as_bytes());
        }
    } else {
        fin = ImodFile::open(&argv[i], "r");
        i += 1;
        if fin.is_none() {
            i -= 1;
            exit_error(format!("Couldn't open {}", argv[i]).as_bytes());
        }
    }

    if imod_backup_file(&argv[i]) != 0 {
        exit_error(format!("Renaming existing output file to {}~", argv[i]).as_bytes());
    }
    let Some(mut fout) = ImodFile::open(&argv[i], "wb") else {
        exit_error(format!("Could not open {}", argv[i]).as_bytes())
    };
    let mut model = imod_from_patches(
        fin.as_mut(),
        scale,
        clip_size,
        name.as_deref(),
        flags,
        val_col_out,
    );

    let _ = imod_write(&model, &mut fout);

    imod_delete(&mut model);
    crate::imod::libcfshr::b3dutil::exit(0);
}

/// `#define MAXLINE 128` (`patch2imod.c:163`).
const MAXLINE: i32 = 128;

/// Original: `imod_from_patches` (`patch2imod.c:165`, static).
///
/// The source leaves `ix`, `iy`, `iz`, `dx`, `dy`, `xx`, `yy`, `values`,
/// `valueIDs`, `valTypeMap` and `orderedIDs` uninitialised; a line that
/// `sscanf` cannot fully convert keeps the previous line's values, and the
/// first such line reads stack residue.  They start at zero here.  With
/// `-l` the first-line parse never fills `valTypeMap`, so a file whose lines
/// carry value columns indexes the per-type arrays with stack residue in the
/// source (`BUGS.md`); zero sends every column to type 0.
fn imod_from_patches(
    mut fin: Option<&mut ImodFile>,
    scale: f32,
    clip_size: i32,
    name: Option<&[u8]>,
    flags: i32,
    mut val_col_out: i32,
) -> Imod {
    let noflip = flags & P2I_NO_FLIP;
    let ignore_zero = flags & P2I_IGNORE_ZERO;
    let count_lines = flags & P2I_COUNT_LINES;
    let display_values = flags & P2I_DISPLAY_VALUES;
    let read_warp = flags & P2I_READ_WARP;
    let times_for_z = flags & P2I_TIMES_FOR_Z;
    let mut len: i32;
    let mut value_ids = [0_i32; MAX_VALUE_COLS];
    let mut val_type_map = [0_i32; MAX_VALUE_COLS];
    let mut ordered_ids = [0_i32; MAX_VALUE_COLS];
    let mut values = [0.0_f32; MAX_VALUE_COLS];

    let mut line = [0_u8; MAXLINE as usize];
    let mut store = Istore::default();
    // Native leaves these, `dx`/`dy` and the ID arrays uninitialised, so a
    // first line `sscanf` cannot fully convert keeps stack residue (BUGS.md).
    // Defined behaviour: they start at 0; later partial lines keep the
    // previous line's values, as the source's `sscanf` does.
    let (mut ix, mut iy, mut iz, mut itmp, mut ind) = (0_i32, 0_i32, 0_i32, 0_i32, 0_usize);
    let mut num_val_ids = 0_usize;
    let (mut nx_warp, mut ny_warp, mut if_control) = (0_i32, 0_i32, 0_i32);
    let (mut nx_grid, mut ny_grid, mut num_control, mut max_prod) = (0_i32, 0_i32, 0_i32, 0_i32);
    let (mut x_start, mut y_start, mut x_interval, mut y_interval) =
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let (mut xmin, mut ymin, mut zmin, mut xmax, mut ymax, mut zmax): (
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
    );
    let (mut dx, mut dy, mut dz, mut xx, mut yy, mut value, mut tmp) = (
        0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32,
    );
    let mut val_min = [0.0_f32; MAX_VALUE_COLS];
    let mut val_max = [0.0_f32; MAX_VALUE_COLS];
    let mut val_sum = [0.0_f64; MAX_VALUE_COLS];
    let mut val_sq_sum = [0.0_f64; MAX_VALUE_COLS];
    let mut num_vals = [0_i32; MAX_VALUE_COLS];
    let (mut valsd, mut sdmax): (f64, f64);
    let (mut x_vector, mut y_vector, mut x_control, mut y_control): (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut dx_grid: Vec<f32> = Vec::new();
    let mut dy_grid: Vec<f32> = Vec::new();
    let mut residuals = 0;
    let mut dzvary = 0;
    let mut npatch = 0_i32;
    let mut nz_warp = 1_i32;
    let mut nread = 0_i32;
    let mut cont_base = 0_i32;
    let mut max_val_cols = 0_i32;

    if read_warp != 0 {
        /* Read in warping file */
        get_warp_file_size(&mut nx_warp, &mut ny_warp, &mut nz_warp, &mut if_control);
        max_prod = 0;
        iz = 0;
        while iz < nz_warp {
            if if_control != 0 {
                ix = get_num_warp_points(iz, &mut num_control);
            } else {
                ix = get_warp_grid_size(iz, &mut nx_grid, &mut ny_grid, &mut num_control);
            }
            if ix != 0 {
                exit_error(
                    format!("Getting number of points in warping file at section {iz}").as_bytes(),
                );
            }
            npatch += num_control;
            max_prod = if max_prod > num_control {
                max_prod
            } else {
                num_control
            };
            iz += 1;
        }
        if if_control == 0 {
            dx_grid = vec![0.0; max_prod.max(0) as usize];
            dy_grid = vec![0.0; max_prod.max(0) as usize];
        }
    } else if count_lines != 0 {
        let fin = fin.as_deref_mut().unwrap();
        /* Count the lines in the file to get # of patches */
        loop {
            ix = fgetline(fin, &mut line, MAXLINE);
            if ix > 2 {
                npatch += 1;
            } else {
                // `fgetline` returns `-(length + 2)` for a line ended by EOF
                // rather than a newline; native (`patch2imod.c:228-234`)
                // drops such a last line (BUGS.md).  Defined behaviour: a
                // complete last line counts like any other.
                if ix < -4 {
                    npatch += 1;
                }
                break;
            }
        }
        if npatch < 1 {
            exit_error(b"No usable lines in the file");
        }
        let _ = fin.seek(SeekFrom::Start(0));

        // Native builds `valTypeMap` and `orderedIDs` only in the first-line
        // branch (`patch2imod.c:282-296`), so with `-l` both are read
        // uninitialised (`:417`, `:483`; BUGS.md).  Defined behaviour: the
        // same conversion of `-v -col` and the same maps, with no value IDs
        // (0) since the file has no first line to supply them.
        if val_col_out < 0 {
            val_col_out = -val_col_out - 1;
        }
        ordered_ids[0] = value_ids[val_col_out as usize];
        val_type_map[val_col_out as usize] = 0;
        ind = 0;
        for i in 1..MAX_VALUE_COLS {
            if ind as i32 == val_col_out {
                ind += 1;
            }
            ordered_ids[i] = value_ids[ind];
            ind += 1;
        }
        ind = 1;
        for i in 0..MAX_VALUE_COLS {
            if i as i32 == val_col_out {
                continue;
            }
            val_type_map[i] = ind as i32;
            ind += 1;
        }
    } else {
        let fin = fin.as_deref_mut().unwrap();
        /* Parse the first line for the # of patches and up to 6 value IDs */
        /* Any number of non-integer entries is skipped here */
        fgetline(fin, &mut line, MAXLINE);
        let line_len = line.iter().position(|&c| c == 0).unwrap_or(line.len());
        let text = &line[..line_len];
        if text.windows(9).any(|w| w == b"residuals") {
            residuals = 1;
        }
        // `strtok(str, " ")`: tokens are the runs of non-space bytes.
        num_val_ids = 0;
        ind = 0;
        for token in text.split(|&c| c == b' ').filter(|t| !t.is_empty()) {
            let mut end = 0usize;
            let long_val = strtol(token, &mut end, 10);
            if !token.is_empty() && end == token.len() {
                if ind == 0 {
                    npatch = long_val as i32;
                } else if num_val_ids < MAX_VALUE_COLS {
                    value_ids[num_val_ids] = long_val as i32;
                    num_val_ids += 1;
                }
            } else if ind == 0 {
                exit_error(b"Converting the first item on the first line to get number of patches");
            }
            ind += 1;
        }

        if npatch < 1 {
            exit_error(format!("Implausible number of patches = {npatch}.").as_bytes());
        }

        /* Convert the valColOut ID or -# to a column index */
        if val_col_out < 0 {
            val_col_out = -val_col_out - 1;
        } else {
            ind = 0;
            while ind < num_val_ids {
                if value_ids[ind] == val_col_out {
                    let _ = ImodFile::Stdout.write_all(
                        format!(
                            "The value with ID {} is in extra column {}\n",
                            val_col_out,
                            ind + 1
                        )
                        .as_bytes(),
                    );
                    val_col_out = ind as i32;
                    break;
                }
                ind += 1;
            }
            if ind == num_val_ids {
                exit_error(b"There is no value column ID # corresponding to the entered ID #");
            }
        }
        /*printf("# vla ID %d %d %d %d\n", numValIDs, valueIDs[0],valueIDs[1],valueIDs[2]);*/

        /* Make the lists of ordered types IDs and map from column to type */
        ordered_ids[0] = value_ids[val_col_out as usize];
        val_type_map[val_col_out as usize] = 0;
        ind = 0;
        for i in 1..MAX_VALUE_COLS {
            if ind as i32 == val_col_out {
                ind += 1;
            }
            ordered_ids[i] = value_ids[ind];
            ind += 1;
        }
        ind = 1;
        for i in 0..MAX_VALUE_COLS {
            if i as i32 == val_col_out {
                continue;
            }
            val_type_map[i] = ind as i32;
            ind += 1;
        }
    }
    /* printf("type map %d %d %d %d %d %d\n", valTypeMap[0], valTypeMap[1], valTypeMap[2],
    valTypeMap[3], valTypeMap[4], valTypeMap[5]); */

    let Some(mut model) = imod_new() else {
        exit_error(b"Could not get new model")
    };

    if imod_new_object(&mut model) != 0 {
        exit_error(b"Could not get new object");
    }
    {
        let obj = &mut model.obj[0];
        let Some(cont) = imod_contours_new(npatch) else {
            exit_error(b"Could not get contour array")
        };
        obj.cont = cont;
        obj.flags |= IMOD_OBJFLAG_OPEN;
    }
    if residuals == 0 && noflip == 0 && read_warp == 0 {
        model.flags |= IMODF_FLIPYZ;
    }
    if times_for_z != 0 {
        model.obj[0].flags |= IMOD_OBJFLAG_TIME;
    }
    model.pixsize = scale;
    xmin = 1000000;
    ymin = 1000000;
    zmin = 1000000;
    xmax = -1000000;
    ymax = -1000000;
    zmax = -1000000;
    for i in 0..MAX_VALUE_COLS {
        val_min[i] = 1.0e30;
        val_max[i] = -1.0e30;
        num_vals[i] = 0;
        val_sum[i] = 0.;
        val_sq_sum[i] = 0.;
    }
    dz = 0.;
    store.flags = GEN_STORE_FLOAT << 2;

    let obj = &mut model.obj[0];
    for iz_warp in 0..nz_warp {
        if read_warp != 0 {
            iz = iz_warp;
            if if_control != 0 {
                if get_num_warp_points(iz, &mut npatch) != 0
                    || get_warp_point_arrays(
                        iz,
                        &mut x_control,
                        &mut y_control,
                        &mut x_vector,
                        &mut y_vector,
                    ) != 0
                {
                    exit_error(
                        format!("Getting number of control points or arrays for section {iz}")
                            .as_bytes(),
                    );
                }
            } else {
                if get_warp_grid(
                    iz,
                    &mut nx_grid,
                    &mut ny_grid,
                    &mut x_start,
                    &mut y_start,
                    &mut x_interval,
                    &mut y_interval,
                    &mut dx_grid,
                    &mut dy_grid,
                    0,
                ) != 0
                {
                    exit_error(format!("Getting warp grid for section {iz}").as_bytes());
                }
                npatch = nx_grid * ny_grid;
            }
        }

        for pat in 0..npatch {
            let cont = &mut obj.cont[(pat + cont_base) as usize];
            cont.pts = vec![Ipoint::default(); 2];
            if times_for_z != 0 {
                cont.time = 1.max(iz + 1);
            }
            if read_warp != 0 {
                if if_control != 0 {
                    xx = x_control[pat as usize];
                    yy = y_control[pat as usize];
                    dx = -x_vector[pat as usize];
                    dy = -y_vector[pat as usize];
                } else {
                    ix = pat % nx_grid;
                    iy = pat / nx_grid;
                    xx = x_start + ix as f32 * x_interval;
                    yy = y_start + iy as f32 * y_interval;
                    dx = -dx_grid[pat as usize];
                    dy = -dy_grid[pat as usize];
                }
            } else {
                len = fgetline(fin.as_deref_mut().unwrap(), &mut line, MAXLINE);
                // A last line ended by EOF comes back as `-(length + 2)`, and
                // native (`patch2imod.c:370-372`) rejects it as a read error
                // (BUGS.md).  Defined behaviour: it is a complete line.
                if len < -4 {
                    len = -len - 2;
                }
                if len < 3 {
                    exit_error(format!("Error reading file at line {}.", pat + 1).as_bytes());
                }
                let line_len = line.iter().position(|&c| c == 0).unwrap_or(line.len());
                let text = String::from_utf8_lossy(&line[..line_len]).into_owned();

                /* DNM 7/26/02: read in residuals as real coordinates, without a
                flip */
                if residuals != 0 {
                    nread = sscanf(
                        &text,
                        "%f %f %d %f %f",
                        &mut [
                            ScanArg::Flt(&mut xx),
                            ScanArg::Flt(&mut yy),
                            ScanArg::Int(&mut iz),
                            ScanArg::Flt(&mut dx),
                            ScanArg::Flt(&mut dy),
                        ],
                    );
                } else {
                    /* DNM 11/15/01: have to handle either with commas or without,
                    depending on whether it was produced by patchcorr3d or
                    patchcrawl3d */
                    if text.contains(',') {
                        nread = sscanf(
                            &text,
                            "%d %d %d %f, %f, %f",
                            &mut [
                                ScanArg::Int(&mut ix),
                                ScanArg::Int(&mut iz),
                                ScanArg::Int(&mut iy),
                                ScanArg::Flt(&mut dx),
                                ScanArg::Flt(&mut dz),
                                ScanArg::Flt(&mut dy),
                            ],
                        );
                    } else {
                        let [v0, v1, v2, v3, v4, v5] = &mut values;
                        nread = sscanf(
                            &text,
                            "%d %d %d %f %f %f %f %f %f %f %f %f",
                            &mut [
                                ScanArg::Int(&mut ix),
                                ScanArg::Int(&mut iz),
                                ScanArg::Int(&mut iy),
                                ScanArg::Flt(&mut dx),
                                ScanArg::Flt(&mut dz),
                                ScanArg::Flt(&mut dy),
                                ScanArg::Flt(v0),
                                ScanArg::Flt(v1),
                                ScanArg::Flt(v2),
                                ScanArg::Flt(v3),
                                ScanArg::Flt(v4),
                                ScanArg::Flt(v5),
                            ],
                        );
                    }
                    if noflip != 0 {
                        itmp = iy;
                        iy = iz;
                        iz = itmp;
                        tmp = dy;
                        dy = dz;
                        dz = tmp;
                    }
                    xx = ix as f32;
                    yy = iy as f32;
                }
            }
            let pts = &mut cont.pts;
            pts[0].x = xx;
            pts[0].y = yy;
            pts[0].z = iz as f32;
            pts[1].x = xx + scale * dx;
            pts[1].y = yy + scale * dy;
            pts[1].z = iz as f32 + scale * dz;
            // `B3DMIN(xmin, xx)`: an int against a float compares in float,
            // and the float result is truncated back into the int.
            xmin = (if (xmin as f32) < xx { xmin as f32 } else { xx }) as i32;
            ymin = (if (ymin as f32) < yy { ymin as f32 } else { yy }) as i32;
            zmin = if zmin < iz { zmin } else { iz };
            // `B3DMAX(xmax, xx + 1.)`: `xx + 1.` is a double.
            xmax = (if xmax as f64 > xx as f64 + 1. {
                xmax as f64
            } else {
                xx as f64 + 1.
            }) as i32;
            ymax = (if ymax as f64 > yy as f64 + 1. {
                ymax as f64
            } else {
                yy as f64 + 1.
            }) as i32;
            zmax = (if zmax as f64 > iz as f64 + 1. {
                zmax as f64
            } else {
                iz as f64 + 1.
            }) as i32;
            if dz != 0. {
                dzvary = 1;
            }

            max_val_cols = if max_val_cols > nread - 6 {
                max_val_cols
            } else {
                nread - 6
            };
            for i in 0..(nread - 6).max(0) as usize {
                let ind = val_type_map[i] as usize;
                value = values[i];
                val_min[ind] = if val_min[ind] < value {
                    val_min[ind]
                } else {
                    value
                };
                val_max[ind] = if val_max[ind] > value {
                    val_max[ind]
                } else {
                    value
                };
                store.type_ = GEN_STORE_VALUE1 + 2 * ind as i16;
                if value != 0. || ignore_zero == 0 {
                    num_vals[ind] += 1;
                    val_sum[ind] += value as f64;
                    val_sq_sum[ind] += (value * value) as f64;
                }
                // Native stores `pat` (`patch2imod.c:427`) while the contour
                // is `pat + contBase`; unreachable there (only warp input has
                // several sections, and it has no values), defined here as
                // the contour's own index (BUGS.md).
                store.index.set_i(pat + cont_base);
                store.value.set_f(value);
                if istore_insert(&mut obj.store, store) != 0 {
                    exit_error(b"Could not add general storage item");
                }
            }
        }
        cont_base += npatch;
    }
    /* printf("mvc %d %d %d %d %d %d %d\n", maxValCols, numVals[0], numVals[1], numVals[2],
    numVals[3], numVals[4], numVals[5]); */

    if residuals != 0 {
        let obj = &mut model.obj[0];
        obj.symflags = IOBJ_SYMF_ARROW as u8;
        obj.symbol = IOBJ_SYM_NONE;
        obj.symsize = 7;
    } else if clip_size != 0 {
        imod_view_model_new(&mut model);
        for view in model.view.iter_mut() {
            view.world |= WORLD_MOVE_ALL_CLIP;
        }
        let obj = &mut model.obj[0];
        obj.clips.count = 4;
        obj.clips.flags = 0;
        obj.clips.normal[0].x = 1.;
        obj.clips.normal[1].x = -1.;
        obj.clips.normal[2].x = 0.;
        obj.clips.normal[3].x = 0.;
        obj.clips.normal[0].y = 0.;
        obj.clips.normal[1].y = 0.;
        obj.clips.normal[2].y = 1.;
        obj.clips.normal[3].y = -1.;
        for i in 0..4 {
            obj.clips.normal[i].z = 0.;
            obj.clips.point[i].x = (-0.5 * (xmax + xmin) as f64) as f32;
            obj.clips.point[i].y = (-0.5 * (ymax + ymin) as f64) as f32;
            obj.clips.point[i].z = (-0.5 * (zmax + zmin) as f64) as f32;
            if obj.clips.normal[i].x != 0. {
                obj.clips.point[i].x += obj.clips.normal[i].x * clip_size as f32 / 2.;
            }
            if obj.clips.normal[i].y != 0. {
                obj.clips.point[i].y += obj.clips.normal[i].y * clip_size as f32 / 2.;
            }
        }
    }

    model.xmax = xmax + xmin;
    model.ymax = ymax + ymin;
    model.zmax = zmax + zmin;
    let obj = &mut model.obj[0];
    if dzvary != 0 {
        obj.symbol = IOBJ_SYM_CIRCLE;
    }

    /* Set current thicken contour flag to aid deleting patches */
    obj.flags |= IMOD_OBJFLAG_THICK_CONT | IMOD_OBJFLAG_MCOLOR;
    if display_values != 0 {
        obj.flags |= IMOD_OBJFLAG_USE_VALUE;
    }
    if max_val_cols != 0 {
        max_val_cols = if max_val_cols > val_col_out {
            max_val_cols
        } else {
            val_col_out
        };
        obj.extra[IOBJ_EXSIZE - 1] = max_val_cols as u32;
    }
    for i in 0..max_val_cols.max(0) as usize {
        obj.extra[IOBJ_EXSIZE - 2 - i] = ordered_ids[i] as u32;
        if num_vals[i] != 0 {
            if num_vals[i] > 10 {
                valsd = (val_sq_sum[i] - val_sum[i] * val_sum[i] / num_vals[i] as f64)
                    / (num_vals[i] - 1) as f64;
                if valsd > 0. {
                    sdmax = val_sum[i] / num_vals[i] as f64 + 10. * valsd.sqrt();
                    val_max[i] = (if (val_max[i] as f64) < sdmax {
                        val_max[i] as f64
                    } else {
                        sdmax
                    }) as f32;
                }
            }

            if istore_add_min_max(
                &mut obj.store,
                GEN_STORE_MINMAX1 + 2 * i as i16,
                val_min[i],
                val_max[i],
            ) != 0
            {
                exit_error(b"Could not add general storage item");
            }
        }
    }

    if let Some(name) = name {
        imod_object_set_name(obj, name);
    }

    let _ = (nx_warp, ny_warp);
    model
}
