//! Translation of `IMOD/flib/model/refinematch.f90`.
//!
//! The main program is [`refinematch`].  Fortran unit 1 is a `BufReader`
//! over the patch file while it is read and a `BufWriter` over each output
//! file while it is written; `rewind(1)` reopens the reader at the start.
//! Two-dimensional arrays keep the source's column-major linear addressing:
//! `fitMat(j, i)` is `fit_mat[(j - 1) + (i - 1) * matCols]`, `cenXYZ(i, j)`
//! is `cen_xyz[(i - 1) + (j - 1) * limPatch]`.
//!
//! Output is written through Rust's stdout, as `dopen` and `irdhdr` write
//! theirs; the gfortran editing of each descriptor is reproduced by
//! [`format_f`] and the two list-directed/`Iw` editors at the end of this
//! file.  The fitting goes through `solve_wo_outliers` and `multRegress`
//! (Gauss-Jordan elimination, no LAPACK), so every value is expected to be
//! bit-identical to native.

use crate::imod::flib::model::get_region_contours::{
    check_boundary_conts, get_contour_array_sizes, get_extra_selections, get_region_contours,
    summarize_drops,
};
use crate::imod::flib::model::solve_wo_outliers::solve_wo_outliers;
use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, frefor2, list_read};
use crate::imod::flib::subrs::hvem::get_nxyz::get_nxyz;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::readnumpatches::read_num_patches;
use crate::imod::flib::subrs::hvem::xfmult3d::xfmult3d;
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libcfshr::parse_params::{pip_get_float, pip_get_two_floats};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// `parameter (LIMEXTRA = 20, LIMCRIT = 20)` (`refinematch.f90:23`).
const LIMEXTRA: i32 = 20;
const LIMCRIT: i32 = 20;
/// `parameter (numOptions = 17)` (`refinematch.f90:51`).
const NUM_OPTIONS: i32 = 17;
/// Fallback PIP table `options(1)` (`refinematch.f90:53-61`).
const OPTIONS: &str = "patch:PatchFile:FN:@output:OutputFile:FN:@region:RegionModel:FN:@\
volume:VolumeOrSizeXYZ:FN:@residual:ResidualPatchOutput:FN:@\
reduced:ReducedVectorOutput:FN:@limit:MeanResidualLimit:F:@\
extra:ExtraValueSelection:IPM:@select:SelectionCriteria:FAM:@\
maxfrac:MaxFractionToDrop:F:@minresid:MinResidualToDrop:F:@\
prob:CriterionProbabilities:FP:@initial:InitialTransformFile:FN:@\
product:ProductTransformFile:FN:@scale:ScaleShiftByFactor:F:@\
param:ParameterFile:PF:@help:usage:B:";

/// Rust-only: the values `refinematch` reports for a caller that used to
/// parse its output lines (`dualvolmatch`, `matchorwarp`): the mean and
/// maximum residual of every FORMAT 101 report in order, the implied center
/// shift (`refinematch.f90:350`) when a product file was written, and whether
/// the final mean residual was above the limit (the status-2 exit).  The
/// program's own output is unchanged; it records these where it prints them
/// (`CLAUDE.md`, "Wherever we control both sides, use a direct function call
/// now").
#[derive(Clone, Debug, Default)]
pub struct RefinematchResult {
    pub residuals: Vec<(f32, f32)>,
    pub center_shift: Option<f32>,
    pub above_limit: bool,
}

thread_local! {
    /// Where [`refinematch`] records its [`RefinematchResult`] on this
    /// thread, when a direct caller set it through [`refinematch_recording`].
    static RESULT_SINK: std::cell::RefCell<Option<std::sync::Arc<std::sync::Mutex<RefinematchResult>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Rust-only: runs program [`refinematch`] on this thread with its reported
/// values recorded into `sink`.  The program ends through `exit`, so the
/// values reach the caller through `sink`; run it under
/// `commands::call_in_process`.
pub fn refinematch_recording(sink: std::sync::Arc<std::sync::Mutex<RefinematchResult>>) {
    RESULT_SINK.with_borrow_mut(|slot| *slot = Some(sink));
    refinematch();
}

/// Rust-only: records into the direct caller's [`RefinematchResult`], if any.
fn record_result(update: impl FnOnce(&mut RefinematchResult)) {
    RESULT_SINK.with_borrow(|slot| {
        if let Some(sink) = slot {
            update(&mut sink.lock().expect("refinematch result sink"));
        }
    });
}

/// Rust-only: the text line of FORMAT 101 (`refinematch.f90:370`) that
/// carries the values, without the blank records around it.
pub fn refinematch_residual_line(dev_mean: f32, dev_max: f32) -> String {
    format!(
        " Mean residual{},  maximum{}",
        format_f(dev_mean as f64, 8, 3),
        format_f(dev_max as f64, 8, 3)
    )
}

/// Original program `refinematch` (`refinematch.f90:21`).
///
/// REFINEMATCH will solve for a general 3-dimensional linear transformation
/// to align two volumes to each other.  It performs multiple linear
/// regression on the displacements between the volumes determined at a
/// matrix of positions.
pub fn refinematch() {
    let mut a_mat = [0.0_f32; 9];
    let mut del_xyz = [0.0_f32; 3];
    let mut a_mat_init = [0.0_f32; 9];
    let mut del_init = [0.0_f32; 3];
    let mut prod_mat = [0.0_f32; 9];
    let mut dev_xyz_max = [0.0_f32; 3];
    let mut fit_center = [0.0_f32; 3];
    let mut cxlast = [0.0_f32; 3];
    let mut freinp = [0.0_f32; LIMEXTRA as usize];
    let mut prod_del = [0.0_f32; 3];
    let mut nxyz = [0_i32; 3];
    let mut numeric = [0_i32; LIMEXTRA as usize];
    let mut i_dextra = [0_i32; LIMEXTRA as usize];
    let mut select_crit = [0.0_f32; (LIMCRIT * LIMEXTRA) as usize];
    let mut num_select_crit: i32;
    let mut icol_select = [0_i32; LIMEXTRA as usize];
    let mut isign_select = [0_i32; LIMEXTRA as usize];
    let mut idrop: Vec<i32>;
    let mut ind_vert: Vec<i32> = Vec::new();
    let mut num_verts: Vec<i32> = Vec::new();
    let mut ind_to_patch: Vec<i32>;
    let mut fit_mat: Vec<f32>;
    let mut x_verts: Vec<f32> = Vec::new();
    let mut y_verts: Vec<f32> = Vec::new();
    let mut contour_z: Vec<f32> = Vec::new();
    let mut extra_vals: Vec<f32> = Vec::new();
    let mut cen_xyz: Vec<f32>;
    let mut vec_xyz: Vec<f32>;
    let mut drop_res: Vec<f32>;
    let mut one_layer = [false; 3];
    // `character*320 filename, line, initialFile, productFile`
    let mut filename = [b' '; 320];
    let mut line: Vec<u8>;
    let mut initial_file = [b' '; 320];
    let mut product_file = [b' '; 320];
    //
    let mut num_conts: i32;
    let mut ierr: i32;
    let mut if_flip = 0_i32;
    let ind_y: i32;
    let ind_z: i32;
    let mut num_data = 0_i32;
    let mut num_fit: i32 = 0;
    let mut frac_drop: f32;
    let mut prod_scale_fac: f32 = 0.;
    let mut cen_dist: f32;
    let mut cen_dist_min: f32;
    let mut if_use = 0_i32;
    let mut ind: i32;
    let mut max_drop: i32;
    let mut ndrop = 0_i32;
    let mut ipnt_max = 0_i32;
    let mut num_fields = 0_i32;
    let mut crit_prob: f32;
    let mut elim_min: f32;
    let mut abs_prob_crit: f32;
    let mut dev_mean = 0.0_f32;
    let mut dev_sd = 0.0_f32;
    let mut dev_max = 0.0_f32;
    let mut stop_lim = 0.0_f32;
    let mut icol_fixed: i32;
    let mut lim_cont = 0_i32;
    let mut lim_vert = 0_i32;
    let mut max_extra: i32;
    let mut num_extra_id = 0_i32;
    let lim_patch: i32;
    let mat_cols: i32;
    let mut num_col_select: i32;
    let mut ipat: i32;
    let mut ind_closest = 0_i32;
    let id_resid_col = 2_i32;

    let pip_input: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    // `use fortmodel` (through `get_region_contours`)
    let mut fm = FortModel::default();

    //
    frac_drop = 0.1;
    crit_prob = 0.01;
    elim_min = 0.5;
    abs_prob_crit = 0.002;
    mat_cols = 20;
    num_col_select = 0;
    num_select_crit = 0;
    initial_file.fill(b' ');
    product_file.fill(b' ');
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "refinematch",
        "ERROR: REFINEMATCH - ",
        true,
        3,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pip_input = num_opt_arg + num_non_opt_arg > 0;
    //
    // Open patch file
    //
    let mut patch_name = String::new();
    if pip_get_in_out_file(
        "PatchFile",
        1,
        "Name of file with correlation positions and results",
        &mut patch_name,
        320,
    ) != 0
    {
        exit_error("No input patch file specified");
    }
    set_record(&mut filename, patch_name.as_bytes());
    let mut unit1 = BufReader::new(dopen(1, &fortran_string(&filename), "old", "f"));

    if !pip_input {
        println!(" Enter file name or NX, NY, NZ of tomogram being matched to");
    }
    get_nxyz(pip_input, "VolumeOrSizeXYZ", "REFINEMATCH", 1, &mut nxyz);

    if pip_input {
        filename.fill(b' ');
        let _ = pipgetstring_(b"RegionModel", &mut filename);
        let _ = pip_get_float(b"MaxFractionToDrop", &mut frac_drop);
        let _ = pip_get_float(b"MinResidualToDrop", &mut elim_min);
        let _ = pip_get_two_floats(
            b"CriterionProbabilities",
            &mut crit_prob,
            &mut abs_prob_crit,
        );
        if pip_get_float(b"MeanResidualLimit", &mut stop_lim) > 0 {
            exit_error("You must enter a mean residual limit");
        }
        ierr = pipgetstring_(b"InitialTransformFile", &mut initial_file);
        ind = pipgetstring_(b"ProductTransformFile", &mut product_file);
        if ierr + ind == 1 {
            exit_error("You must enter both -initial and -product files, not just one");
        }
        prod_scale_fac = 1.;
        let _ = pip_get_float(b"ScaleShiftByFactor", &mut prod_scale_fac);
        if !is_blank(&initial_file) {
            let mut unit3 = BufReader::new(dopen(3, &fortran_string(&initial_file), "ro", "f"));
            // `read(3,*, err = 99) ((aMatInit(i, j), j = 1, 3), delInit(i), i = 1, 3)`
            let mut vals = [0.0_f32; 12];
            let result = {
                let mut items: Vec<ListItem> = vals.iter_mut().map(ListItem::Real).collect();
                list_read(&mut unit3, &mut items)
            };
            for i in 0..3 {
                for j in 0..3 {
                    a_mat_init[i + j * 3] = vals[i * 4 + j];
                }
                del_init[i] = vals[i * 4 + 3];
            }
            match result {
                Ok(()) => {}
                Err(ListReadError::Error) => exit_error("Reading initial transform file"),
                Err(ListReadError::End) => read_runtime_error(ListReadError::End),
            }
            drop(unit3);
        }
    } else {
        print!(
            " Enter name of model file with contour enclosing area to use,\n or Return to use all patches: "
        );
        let _ = std::io::stdout().flush();
        match read_a(&mut std::io::stdin().lock(), 320) {
            Ok(record) => filename.copy_from_slice(&record),
            Err(err) => read_runtime_error(err),
        }
    }

    num_conts = 0;
    if !is_blank(&filename) {
        get_contour_array_sizes(
            &mut fm,
            &fortran_string(&filename),
            0,
            &mut lim_cont,
            &mut lim_vert,
        );
        x_verts = vec![0.0; lim_vert as usize];
        y_verts = vec![0.0; lim_vert as usize];
        contour_z = vec![0.0; lim_cont as usize];
        ind_vert = vec![0; lim_cont as usize];
        num_verts = vec![0; lim_cont as usize];
        memory_error(0, "arrays for boundary contours");
        get_region_contours(
            &mut fm,
            &fortran_string(&filename),
            "REFINEMATCH",
            &mut x_verts,
            &mut y_verts,
            &mut num_verts,
            &mut ind_vert,
            &mut contour_z,
            &mut num_conts,
            &mut if_flip,
            &mut lim_cont,
            &mut lim_vert,
            0,
        );
    } else {
        if_flip = 0;
        if nxyz[1] < nxyz[2] {
            if_flip = 1;
        }
    }
    let mut iy = 2;
    if if_flip != 0 {
        iy = 3;
    }
    ind_y = iy;
    ind_z = 5 - ind_y;
    //
    i_dextra.fill(0);
    if read_num_patches(
        &mut unit1,
        &mut num_data,
        &mut num_extra_id,
        &mut i_dextra,
        LIMEXTRA,
    ) != 0
    {
        exit_error("Reading first line of patch file");
    }
    max_extra = 0;
    for _i in 1..=num_data {
        line = match read_a(&mut unit1, 320) {
            Ok(record) => record,
            Err(err) => read_runtime_error(err),
        };
        frefor2(
            &String::from_utf8_lossy(&line),
            &mut freinp,
            &mut numeric,
            &mut num_fields,
            LIMEXTRA,
        );
        max_extra = max_extra.max(num_fields - 6);
    }
    // `rewind(1)`
    drop(unit1);
    let mut unit1 = BufReader::new(
        File::open(patch_name.trim_end_matches(' '))
            .unwrap_or_else(|_| read_runtime_error(ListReadError::Error)),
    );
    if let Err(err) = read_a(&mut unit1, 320) {
        read_runtime_error(err);
    }

    lim_patch = num_data + 10;
    let lp = lim_patch as usize;
    let mc = mat_cols as usize;
    fit_mat = vec![0.0; mc * lp];
    cen_xyz = vec![0.0; lp * 3];
    vec_xyz = vec![0.0; lp * 3];
    ind_to_patch = vec![0; lp];
    idrop = vec![0; lp];
    drop_res = vec![0.0; ((frac_drop * lim_patch as f32).round() as i32 + 10).max(0) as usize];
    memory_error(0, "arrays for data matrix and patches");
    if max_extra > 0 {
        extra_vals = vec![0.0; max_extra as usize * lp];
        memory_error(0, "array for extra values");
    }
    let me = max_extra.max(0) as usize;
    // `fitMat(j, i)`, `cenXYZ(i, j)`, `vecXYZ(i, j)`, `extraVals(j, i)`.
    let fi = |j: i32, i: i32| (j - 1) as usize + (i - 1) as usize * mc;
    let ci = |i: i32, j: i32| (i - 1) as usize + (j - 1) as usize * lp;
    let ei = |j: i32, i: i32| (j - 1) as usize + (i - 1) as usize * me;

    if pip_input {
        get_extra_selections(
            &mut icol_select,
            &mut isign_select,
            &mut num_col_select,
            &mut num_select_crit,
            &i_dextra,
            max_extra,
            &mut select_crit,
            LIMEXTRA,
            LIMCRIT,
        );
    }
    // `selectCrit(indSelect, ind)`
    let sci = |i: i32, j: i32| (i - 1) as usize + (j - 1) as usize * LIMCRIT as usize;

    // Read in the patch data and extra data
    for j in 1..=3usize {
        one_layer[j - 1] = true;
        cxlast[j - 1] = 0.;
    }
    for i in 1..=num_data {
        //
        // these are center coordinates and location of the second volume
        // relative to the first volume
        //
        line = match read_a(&mut unit1, 320) {
            Ok(record) => record,
            Err(err) => read_runtime_error(err),
        };
        frefor2(
            &String::from_utf8_lossy(&line),
            &mut freinp,
            &mut numeric,
            &mut num_fields,
            LIMEXTRA,
        );
        for j in 1..=3 {
            cen_xyz[ci(i, j)] = freinp[(j - 1) as usize];
            vec_xyz[ci(i, j)] = freinp[(j + 3 - 1) as usize];
            if i > 1 && cen_xyz[ci(i, j)] != cxlast[(j - 1) as usize] {
                one_layer[(j - 1) as usize] = false;
            }
            cxlast[(j - 1) as usize] = cen_xyz[ci(i, j)];
        }
        for j in 7..=num_fields {
            extra_vals[ei(j - 6, i)] = freinp[(j - 1) as usize];
        }
    }
    drop(unit1);
    //
    if !pip_input {
        print!(" Mean residual above which to STOP and exit with an error: ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Real(&mut stop_lim)],
        ) {
            read_runtime_error(err);
        }
    }
    //
    icol_fixed = 0;
    for i in 1..=3 {
        if icol_fixed != 0 && one_layer[(i - 1) as usize] {
            exit_error("Cannot fit to patches that extend in only one dimension");
        }
        if one_layer[(i - 1) as usize] {
            icol_fixed = i;
        }
    }
    if icol_fixed > 0 {
        println!(
            " There is only one layer of patches in the {} dimension",
            (b'W' + icol_fixed as u8) as char
        );
    }
    //
    // Loop on the selection criteria
    if num_col_select > 0 {
        println!("{} total patches in patch file\n", fmt_i(num_data, 8));
    }
    for ind_select in 1..=1.max(num_select_crit) {
        if num_col_select > 0 {
            println!(
                "Applying extra column selection criteria{}",
                format_f(select_crit[sci(ind_select, 1)] as f64, 9, 3)
            );
        }
        num_fit = 0;
        for i in 1..=num_data {
            if_use = 1;
            if num_conts > 0 {
                check_boundary_conts(
                    cen_xyz[ci(i, 1)],
                    cen_xyz[ci(i, ind_y)],
                    cen_xyz[ci(i, ind_z)],
                    &mut if_use,
                    num_conts,
                    &num_verts,
                    &x_verts,
                    &y_verts,
                    &contour_z,
                    &ind_vert,
                );
            }
            //
            // Eliminate points that do not pass selection criteria
            if num_col_select > 0 && if_use > 0 {
                for ind in 1..=num_col_select {
                    if (extra_vals[ei(icol_select[(ind - 1) as usize], i)]
                        - select_crit[sci(ind_select, ind)])
                        * (isign_select[(ind - 1) as usize] as f32)
                        < 0.
                    {
                        if_use = 0;
                    }
                }
            }
            //
            if if_use > 0 {
                num_fit += 1;
                ind_to_patch[(num_fit - 1) as usize] = i;
                for j in 1..=3 {
                    //
                    // the regression requires coordinates of second volume as
                    // independent variables (columns 1-3), those in first volume
                    // as dependent variables (stored in 5-7), to obtain
                    // transformation to get from second to first volume
                    // cx+dx in second volume matches cx in first volume
                    //
                    fit_mat[fi(j + 4, num_fit)] =
                        cen_xyz[ci(i, j)] - 0.5_f32 * nxyz[(j - 1) as usize] as f32;
                    fit_mat[fi(j, num_fit)] = fit_mat[fi(j + 4, num_fit)] + vec_xyz[ci(i, j)];
                }
            }
        }

        if num_fit < 4 {
            exit_error("Too few data points for fitting");
        }
        println!("{}  data points will be used for fit", ld_int(num_fit));
        //
        // Get the solution
        max_drop = (frac_drop * num_fit as f32).round() as i32;
        solve_wo_outliers(
            &mut fit_mat,
            mat_cols,
            num_fit,
            3,
            icol_fixed,
            max_drop,
            crit_prob,
            abs_prob_crit,
            elim_min,
            &mut idrop,
            &mut ndrop,
            &mut a_mat,
            &mut del_xyz,
            &mut fit_center,
            &mut dev_mean,
            &mut dev_sd,
            &mut dev_max,
            &mut ipnt_max,
            &mut dev_xyz_max,
        );
        //
        // Leave the loop if pass the limit
        if dev_mean <= stop_lim {
            break;
        }
        if ind_select < num_select_crit {
            print_101(dev_mean, dev_max);
        }
    }

    if ndrop != 0 {
        println!(
            "\n{} patches dropped by outlier elimination:",
            fmt_i(ndrop, 3)
        );
        if ndrop <= 10 {
            println!("     patch position     residual");
            for i in 1..=ndrop {
                println!(
                    "{}{}{}{}",
                    format_f(fit_mat[fi(1, num_fit + i - ndrop)] as f64, 7, 0),
                    format_f(fit_mat[fi(2, num_fit + i - ndrop)] as f64, 7, 0),
                    format_f(fit_mat[fi(3, num_fit + i - ndrop)] as f64, 7, 0),
                    format_f(fit_mat[fi(4, num_fit + i - ndrop)] as f64, 9, 2)
                );
            }
        } else {
            for i in 1..=ndrop {
                drop_res[(i - 1) as usize] = fit_mat[fi(4, num_fit + i - ndrop)];
            }
            summarize_drops(&drop_res, ndrop, " ");
        }
    }
    //
    print_101(dev_mean, dev_max);
    //
    println!(" Refining transformation:");
    for i in 0..3 {
        println!(
            "{}{}{}{}",
            format_f(a_mat[i] as f64, 10, 6),
            format_f(a_mat[i + 3] as f64, 10, 6),
            format_f(a_mat[i + 6] as f64, 10, 6),
            format_f(del_xyz[i] as f64, 10, 3)
        );
    }
    //
    if pip_input {
        //
        // For patch residual, first make a proper index from original to ordered rows
        for i in 1..=num_fit {
            let row = fit_mat[fi(5, i)].round() as i32;
            fit_mat[fi(6, row)] = i as f32;
        }
        if pipgetstring_(b"ResidualPatchOutput", &mut filename) == 0 {
            let mut unit1 = BufWriter::new(dopen(1, &fortran_string(&filename), "new", "f"));
            let mut text = format!("{} positions{}", fmt_i(num_fit, 7), fmt_i(id_resid_col, 8));
            if max_extra > 0 {
                for i in 1..=max_extra {
                    text += &fmt_i(i_dextra[(i - 1) as usize], 8);
                }
            }
            write_record(&mut unit1, &text);
            for i in 1..=num_fit {
                ipat = ind_to_patch[(i - 1) as usize];
                let mut text = String::new();
                for j in 1..=3 {
                    text += &fmt_i(cen_xyz[ci(ipat, j)].round() as i32, 6);
                }
                for j in 1..=3 {
                    text += &format_f(vec_xyz[ci(ipat, j)] as f64, 9, 2);
                }
                text += &format_f(
                    fit_mat[fi(4, fit_mat[fi(6, i)].round() as i32)] as f64,
                    10,
                    2,
                );
                if max_extra > 0 {
                    for j in 1..=max_extra {
                        text += &format_f(extra_vals[ei(j, ipat)] as f64, 12, 4);
                    }
                }
                write_record(&mut unit1, &text);
            }
            let _ = unit1.flush();
        }
        //
        // For reduced vectors, put out the residual vector stored in cols 15-17
        // plus either the residual or original extra values
        if pipgetstring_(b"ReducedVectorOutput", &mut filename) == 0 {
            let mut unit1 = BufWriter::new(dopen(1, &fortran_string(&filename), "new", "f"));
            let mut text = format!("{} positions", fmt_i(num_fit, 7));
            if max_extra > 0 {
                for i in 1..=max_extra {
                    text += &fmt_i(i_dextra[(i - 1) as usize], 8);
                }
            } else {
                text += &fmt_i(id_resid_col, 8);
            }
            write_record(&mut unit1, &text);
            for i in 1..=num_fit {
                ind = fit_mat[fi(6, i)].round() as i32;
                ipat = ind_to_patch[(i - 1) as usize];
                let mut text = String::new();
                for j in 1..=3 {
                    text += &fmt_i(cen_xyz[ci(ipat, j)].round() as i32, 6);
                }
                for j in 15..=17 {
                    text += &format_f(fit_mat[fi(j, ind)] as f64, 9, 2);
                }
                if max_extra > 0 {
                    for j in 1..=max_extra {
                        text += &format_f(extra_vals[ei(j, ipat)] as f64, 12, 4);
                    }
                } else {
                    text += &format_f(fit_mat[fi(4, ind)] as f64, 10, 2);
                }
                write_record(&mut unit1, &text);
            }
            let _ = unit1.flush();
        }
        filename.fill(b' ');
        let _ = pipgetstring_(b"OutputFile", &mut filename);
    } else {
        println!(" Enter name of file to place transformation in, or Return for none");
        match read_a(&mut std::io::stdin().lock(), 320) {
            Ok(record) => filename.copy_from_slice(&record),
            Err(err) => read_runtime_error(err),
        }
    }
    if !is_blank(&filename) {
        let mut unit1 = BufWriter::new(dopen(1, &fortran_string(&filename), "new", "f"));
        for i in 0..3 {
            let text = format!(
                "{}{}{}{}",
                format_f(a_mat[i] as f64, 10, 6),
                format_f(a_mat[i + 3] as f64, 10, 6),
                format_f(a_mat[i + 6] as f64, 10, 6),
                format_f(del_xyz[i] as f64, 10, 3)
            );
            write_record(&mut unit1, &text);
        }
        let _ = unit1.flush();
    }

    // Take product with initial file and output that if requested
    if !is_blank(&initial_file) {
        xfmult3d(
            &a_mat_init,
            &del_init,
            &a_mat,
            &del_xyz,
            &mut prod_mat,
            &mut prod_del,
        );
        let mut unit1 = BufWriter::new(dopen(1, &fortran_string(&product_file), "new", "f"));
        for i in 0..3 {
            let text = format!(
                "{}{}{}{}",
                format_f(prod_mat[i] as f64, 10, 6),
                format_f(prod_mat[i + 3] as f64, 10, 6),
                format_f(prod_mat[i + 6] as f64, 10, 6),
                format_f((prod_scale_fac * prod_del[i]) as f64, 10, 3)
            );
            write_record(&mut unit1, &text);
        }
        let _ = unit1.flush();
        drop(unit1);
        //
        // This triggers reporting of center shift as Y residual of used patch  nearest
        // center
        cen_dist_min = 1.0e37;
        for i in 1..=num_fit - ndrop {
            // `refinematch.f90:345` takes `ipat = nint(fitMat(5, i))`, which
            // is the row of the fitted data (solve_wo_outliers' cross index),
            // and indexes `cenXYZ` (by patch) with it; when a region model or
            // a selection has excluded patches native measures the distance
            // to a different patch.  Fixed in translation (BUGS.md): the row
            // goes through `indToPatch` to the patch it came from.
            ipat = ind_to_patch[(fit_mat[fi(5, i)].round() as i32 - 1) as usize];
            let dx = cen_xyz[ci(ipat, 1)] - nxyz[0] as f32 / 2.0_f32;
            let dz = cen_xyz[ci(ipat, 3)] - nxyz[2] as f32 / 2.0_f32;
            cen_dist = (dx * dx + dz * dz).sqrt();
            if cen_dist < cen_dist_min {
                cen_dist_min = cen_dist;
                ind_closest = i;
            }
        }
        println!(
            "Implied center shift is {}",
            format_f((fit_mat[fi(16, ind_closest)] * prod_scale_fac) as f64, 8, 1)
        );
        let shift = fit_mat[fi(16, ind_closest)] * prod_scale_fac;
        record_result(|result| result.center_shift = Some(shift));
    }

    if dev_mean > stop_lim {
        println!("\nREFINEMATCH - Mean residual too high; either raise the limit or use warping");
        record_result(|result| result.above_limit = true);
        exit(2);
    }
    exit(0);
}

/// `write(*,101) devMean, devMax` with
/// `101 format(/,' Mean residual',f8.3,',  maximum',f8.3,/)`.
fn print_101(dev_mean: f32, dev_max: f32) {
    println!("\n{}\n", refinematch_residual_line(dev_mean, dev_max));
    record_result(|result| result.residuals.push((dev_mean, dev_max)));
}

/// Whether a `character` variable equals `' '` (is all blanks).
fn is_blank(record: &[u8]) -> bool {
    record.iter().all(|&b| b == b' ')
}

/// Character assignment into a fixed-length variable: cut or blank padded.
fn set_record(record: &mut [u8], text: &[u8]) {
    record.fill(b' ');
    let count = text.len().min(record.len());
    record[..count].copy_from_slice(&text[..count]);
}

/// A formatted write of one record to a file unit.
fn write_record(unit: &mut BufWriter<File>, text: &str) {
    let _ = unit.write_all(text.as_bytes());
    let _ = unit.write_all(b"\n");
}

/// gfortran `read(unit, '(a)') var` into a `character*len` variable: the
/// next record, cut at `len` or blank padded to it.  End of file is `END`.
fn read_a<R: BufRead>(unit: &mut R, len: usize) -> Result<Vec<u8>, ListReadError> {
    let mut record: Vec<u8> = Vec::new();
    match unit.read_until(b'\n', &mut record) {
        Ok(0) => return Err(ListReadError::End),
        Err(_) => return Err(ListReadError::Error),
        Ok(_) => {}
    }
    if record.last() == Some(&b'\n') {
        record.pop();
    }
    record.resize(len, b' ');
    Ok(record)
}

/// A formatted or list-directed `read` with no `END=`/`ERR=` that fails:
/// libgfortran reports it and stops with status 2.
fn read_runtime_error(err: ListReadError) -> ! {
    let _ = std::io::stdout().flush();
    match err {
        ListReadError::End => eprintln!("Fortran runtime error: End of file"),
        ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
    }
    exit(2);
}

/// gfortran `Iw` output editing.
fn fmt_i(value: i32, w: usize) -> String {
    let text = format!("{value}");
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// A list-directed (`print *`) `integer*4` item: a blank separator (the
/// record's leading blank when it is the first item) and `I11`.
fn ld_int(value: i32) -> String {
    format!("{value:>12}")
}
