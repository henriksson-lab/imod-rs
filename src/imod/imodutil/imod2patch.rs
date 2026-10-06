//! Translation of `IMOD/imodutil/imod2patch.c`: converts a model of two-point
//! vector contours (as made by `patch2imod`, possibly edited) back to a patch
//! file, with any general values stored in the model as extra columns.
//!
//! The program maps to [`imod2patch`] and the static `istoreFindValue` to
//! [`istore_find_value`].  Output goes through C stdio formatting
//! (`c_format`) into the output file.

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, exit, imod_backup_file, program_args};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
use crate::imod::libimod::imodel::IMODF_FLIPYZ;
use crate::imod::libimod::imodel_files::imod_read;
use crate::imod::libimod::iobj::IOBJ_EXSIZE;
use crate::imod::libimod::istore::{GEN_STORE_NOINDEX, GEN_STORE_VALUE1, Istore};
use std::io::Write;

/// `#define MAX_VALUE_COLS 6` (`imod2patch.c:18`).
const MAX_VALUE_COLS: usize = 6;

/// Original program `main` (`imod2patch.c:20`).
pub fn imod2patch() {
    let argv = program_args();
    let argc = argv.len();
    let mut ind: usize;
    let mut col: usize;
    let mut max_val_type: i32;
    let mut npatch = 0_i32;
    let mut num_values = [0_i32; MAX_VALUE_COLS];
    let mut value_ids = [0_i32; MAX_VALUE_COLS];
    let mut col_for_val1: i32 = 1;
    let mut value = 0.0_f32;
    let mut max_val = [0.0_f32; MAX_VALUE_COLS];
    let mut list_ind: usize;
    let mut list_start: usize;
    let mut num_cols: usize;
    let mut format: [&str; MAX_VALUE_COLS] = ["%10.4f"; MAX_VALUE_COLS];
    let mut col_to_type_map = [-1_i32; MAX_VALUE_COLS];

    setExitPrefix(b"ERROR: imod2patch - ");

    let usage = |wrong: bool| -> ! {
        let mut out = ImodFile::Stdout;
        if wrong {
            let _ = out.write_all(b"ERROR: imod2patch - wrong # of arguments\n");
        }
        let _ = out.write_all(b"imod2patch usage:\n");
        let _ = out.write_all(b"imod2patch [-v col] imod_model patch_file\n");
        let _ = out.flush();
        exit(1);
    };
    if argc < 3 {
        usage(argc != 1);
    }
    for ind in 0..MAX_VALUE_COLS {
        num_values[ind] = 0;
        value_ids[ind] = 0;
        col_to_type_map[ind] = -1;
        max_val[ind] = -1.0e37;
    }

    let mut i = 1;
    if argv[i] == "-v" {
        i += 1;
        col_for_val1 = crate::imod::clip::clip::atoi(&argv[i]);
        i += 1;
    }
    // Fixed in translation (BUGS.md, `imod2patch`): with `-v col` and fewer
    // than two names the source reads `argv[argc]` (NULL) as a file name.
    // Defined: the usage message for a wrong number of arguments.
    if i + 1 >= argc {
        usage(true);
    }

    let mut model = match imod_read(&argv[i]) {
        Ok(model) => model,
        Err(_) => exit_error(format!("Reading model {}\n", argv[i]).as_bytes()),
    };

    i += 1;
    if imod_backup_file(&argv[i]) != 0 {
        exit_error(format!("Renaming existing output file to {}\n", argv[i]).as_bytes());
    }

    let Some(mut fout) = ImodFile::open(&argv[i], "w") else {
        exit_error(format!("Could not open {}\n", argv[i]).as_bytes())
    };

    /* Count up patches and values and find max of values */
    for ob in 0..model.obj.len() {
        list_ind = 0;
        for co in 0..model.obj[ob].cont.len() {
            list_start = list_ind;
            if model.obj[ob].cont[co].pts.len() >= 2 {
                npatch += 1;
                for ind in 0..MAX_VALUE_COLS {
                    list_ind = list_start;
                    if istore_find_value(
                        &model.obj[ob].store,
                        co as i32,
                        GEN_STORE_VALUE1 as i32 + 2 * ind as i32,
                        &mut value,
                        &mut list_ind,
                    ) != 0
                    {
                        // B3DMAX(a, b): `a > b ? a : b`.
                        max_val[ind] = if max_val[ind] > value { max_val[ind] } else { value };
                        num_values[ind] += 1;
                    }
                }
            }
        }
    }

    /* Get the value ID's if any and convert an entered ID to a type #.  Also set format
    for output based on maximum value */
    max_val_type = -1;
    num_cols = 0;
    for ind in 0..MAX_VALUE_COLS {
        format[ind] = "%10.4f";
        if num_values[ind] != 0 {
            num_cols += 1;
            max_val_type = ind as i32;
            if max_val[ind] > 10.1 {
                format[ind] = "%10.2f";
            } else if max_val[ind] > 1.01 {
                format[ind] = "%10.3f";
            }
        }
    }
    let extra = |index: usize| model.obj[0].extra[index] as i32;
    ind = 0;
    while (ind as i32) < (max_val_type + 1).min(extra(IOBJ_EXSIZE - 1)) {
        value_ids[ind] = extra(IOBJ_EXSIZE - 2 - ind);
        ind += 1;
    }

    /* Error checks on the column for value 1 */
    if col_for_val1 < 1 || col_for_val1 > num_cols as i32 {
        if num_cols != 0 {
            exit_error(
                format!(
                    "The column for general value type must be between 1 and {}",
                    num_cols
                )
                .as_bytes(),
            );
        }
        exit_error(b"There are no general values stored in the model");
    }
    col_for_val1 -= 1;

    /* Set up map from column to value type index */
    if num_values[0] != 0 {
        col_to_type_map[col_for_val1 as usize] = 0;
    }
    col = 0;
    ind = 1;
    while ind as i32 <= max_val_type {
        if num_values[ind] != 0 {
            if col_to_type_map[col] == 0 {
                col += 1;
            }
            col_to_type_map[col] = ind as i32;
            col += 1;
        }
        ind += 1;
    }

    /* Output the header line */
    let mut text = c_format("%d   edited positions", &[CArg::Int(npatch as i64)]);
    if extra(IOBJ_EXSIZE - 1) > 0 {
        for col in 0..num_cols {
            text.push_str(&c_format(
                "  %d",
                &[CArg::Int(value_ids[col_to_type_map[col] as usize] as i64)],
            ));
        }
    }
    text.push('\n');
    let _ = fout.write_all(text.as_bytes());

    for ob in 0..model.obj.len() {
        list_ind = 0;
        for co in 0..model.obj[ob].cont.len() {
            list_start = list_ind;
            if model.obj[ob].cont[co].pts.len() >= 2 {
                let pts = &model.obj[ob].cont[co].pts;
                let ix = (pts[0].x as f64 + 0.5) as i32;
                let iy = (pts[0].y as f64 + 0.5) as i32;
                let iz = (pts[0].z as f64 + 0.5) as i32;
                let dx = (pts[1].x - pts[0].x) / model.pixsize;
                let dy = (pts[1].y - pts[0].y) / model.pixsize;
                let dz = (pts[1].z - pts[0].z) / model.pixsize;
                let mut line = if model.flags & IMODF_FLIPYZ != 0 {
                    c_format(
                        "%6d %5d %5d %8.2f %8.2f %8.2f",
                        &[
                            CArg::Int(ix as i64),
                            CArg::Int(iz as i64),
                            CArg::Int(iy as i64),
                            CArg::Dbl(dx as f64),
                            CArg::Dbl(dz as f64),
                            CArg::Dbl(dy as f64),
                        ],
                    )
                } else {
                    c_format(
                        "%6d %5d %5d %8.2f %8.2f %8.2f",
                        &[
                            CArg::Int(ix as i64),
                            CArg::Int(iy as i64),
                            CArg::Int(iz as i64),
                            CArg::Dbl(dx as f64),
                            CArg::Dbl(dy as f64),
                            CArg::Dbl(dz as f64),
                        ],
                    )
                };
                for col in 0..num_cols {
                    let ind = col_to_type_map[col];
                    if num_values[ind as usize] != 0 {
                        list_ind = list_start;
                        value = 0.;
                        istore_find_value(
                            &model.obj[ob].store,
                            co as i32,
                            GEN_STORE_VALUE1 as i32 + 2 * ind,
                            &mut value,
                            &mut list_ind,
                        );
                        line.push_str(&c_format(format[ind as usize], &[CArg::Dbl(value as f64)]));
                    }
                }
                line.push('\n');
                let _ = fout.write_all(line.as_bytes());
            }
        }
    }
    let _ = fout.flush();
    drop(fout);
    model.obj.clear();
    exit(0);
}

/// Original `istoreFindValue` (`imod2patch.c:183`, static).
///
/// This is meant to be called sequentially for all indexes in the entity,
/// not for random access.
fn istore_find_value(
    list: &[Istore],
    index: i32,
    type_: i32,
    value: &mut f32,
    list_ind: &mut usize,
) -> i32 {
    while *list_ind < list.len() {
        let store = &list[*list_ind];
        if (store.flags & GEN_STORE_NOINDEX) != 0 || store.index.i() > index {
            break;
        }
        *list_ind += 1;

        if store.index.i() == index && store.type_ as i32 == type_ {
            *value = store.value.f();
            return 1;
        }
    }
    0
}
