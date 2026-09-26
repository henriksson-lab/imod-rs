//! Translation of `IMOD/flib/model/solvematch.f90`.
//!
//! The main program maps to [`solvematch`] and the external procedures to
//! [`get_delta`], [`get_fiducials`], [`read_fid_model_file`],
//! [`rotate_fids`], [`fill_local_data`] and [`determ3`].  The `fortmodel`
//! module arrays are the [`FortModel`] that `readw_or_imod` fills, passed by
//! reference to the units that `use fortmodel`.  The library routines
//! `solve_wo_outliers`/`do3multr` are `flib/model/solve_wo_outliers.rs`.
//!
//! Arrays keep their Fortran shapes as flat column-major storage:
//! `xMat(MAT_SIZE, IDIM)` element `(j, i)` is `x_mat[(j - 1) + (i - 1) * 20]`;
//! `pointsA(3, IDIM)` is `[f32; 3]` per point; the `(IDIM, 2)` arrays
//! (`icontAB`, `icontToPointAB`, `listCorrAB`, `modObj`, `modCont`,
//! `transferX/Y`, `fidModX/Y`, `modObjFid`, `modContFid`) hold column `k` at
//! `[(k - 1) * IDIM ..]`, which is what the source's `EQUIVALENCE`s of
//! `icontA`/`icontB` etc. to the two columns name.
//!
//! Degree trigonometry follows the reference build (`solvematch.o` calls
//! `_gfortran_sind_r4`/`_gfortran_cosd_r4`), `NINT` is `lroundf`, and
//! formatted output uses gfortran `Fw.d`/`Iw` editing; list-directed
//! `print *` writes an integer as a blank and `I11`, with a blank between a
//! number and a following string and none between two strings.

use crate::imod::flib::model::solve_wo_outliers::{
    D3MULTVARS, D3MultVars, do3multr, solve_wo_outliers,
};
use crate::imod::flib::subrs::compat::gfortran_rt::{
    format_f, gfortran_cosd_r4, gfortran_sind_r4, maxss, minss,
};
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, frefor, frefor2, list_read};
use crate::imod::flib::subrs::hvem::get_nxyz::{get_nxyz, line_is_filename};
use crate::imod::flib::subrs::hvem::objtocont::objtocont;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::rdlist::{parselist2, rdlist2};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen};
use crate::imod::flib::subrs::model::fortmodel::{FortModel, allocate_fort_model};
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::scale_model;
use crate::imod::libcfshr::b3dutil::{exit, fortran_string, imodgetenv};
use crate::imod::libcfshr::parse_params::{pip_get_float, pip_get_integer, pip_get_two_floats};
use crate::imod::libcfshr::writelist::wrlist;
use crate::imod::libiimod::unit_fileio::iiu_close;
use crate::imod::libiimod::unit_header::iiu_ret_delta;
use crate::imod::libimod::imodel_fwrap::{getimodhead, getimodscales};
use std::io::{BufRead, BufReader, BufWriter, Write};

/// `parameter (IDIM = 10000, ...)` (`solvematch.f90:14`).
const IDIM: i32 = 10000;
/// `parameter (..., MAT_SIZE = 20)` (`solvematch.f90:14`).
const MAT_SIZE: i32 = 20;
/// `parameter (numOptions = 26)` (`solvematch.f90:74`).
const NUM_OPTIONS: i32 = 26;
/// Fallback PIP table `options(1)` (`solvematch.f90:76-87`).
const OPTIONS: &str = "output:OutputFile:FN:@afiducials:AFiducialFile:FN:@\
bfiducials:BFiducialFile:FN:@alist:ACorrespondenceList:LI:@\
blist:BCorrespondenceList:LI:@transfer:TransferCoordinateFile:FN:@\
amodel:AFiducialModel:FN:@bmodel:BFiducialModel:FN:@use:UsePoints:LI:@\
atob:MatchingAtoB:B:@xtilts:XAxisTilts:FP:@angles:AngleOffsetsToTilt:FP:@\
zshifts:ZShiftsToTilt:FP:@surfaces:SurfacesOrUseModels:I:@\
inverted:InvertedInDepth:I:@maxresid:MaximumResidual:F:@local:LocalFitting:I:@\
center:CenterShiftLimit:F:@amatch:AMatchingModel:FN:@\
bmatch:BMatchingModel:FN:@atomogram:ATomogramOrSizeXYZ:CH:@\
btomogram:BTomogramOrSizeXYZ:CH:@scales:ScaleFactors:FP:@\
aniso:AnisotropicLimit:F:@param:ParameterFile:PF:@help:usage:B:";

/// gfortran `Fw.d` output editing of a `real*4` (`format_f` on the exact
/// widening).
fn fmt_f(value: f32, w: usize, d: usize) -> String {
    format_f(value as f64, w, d)
}

/// gfortran `Iw` output editing.
fn fmt_i(value: i32, w: usize) -> String {
    let text = format!("{value}");
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// Fortran `NINT` of a `real*4`, which gfortran emits as `lroundf`: round
/// half away from zero, then the `long` narrowed to `integer*4`.
fn nint(value: f32) -> i32 {
    value.round() as i64 as i32
}

/// gfortran runtime failure of a `READ` with no `END=`/`ERR=` branch: the
/// runtime reports it and stops with status 2.
fn read_abort(err: ListReadError) -> ! {
    let _ = std::io::stdout().flush();
    match err {
        ListReadError::End => eprintln!("Fortran runtime error: End of file"),
        ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
    }
    exit(2);
}

/// The Fortran wrapper `pipgetstring` (`pip_fwrap.c:206`) into a
/// `character*length` variable: the variable is left untouched unless the
/// option is found, and `c2fString` fills at most `length` characters,
/// returning -1 with the truncated text left in it when the entry is longer.
fn get_string(option: &[u8], string: &mut String, length: usize) -> i32 {
    let mut record = vec![b' '; length];
    let current = string.as_bytes();
    let count = current.len().min(length);
    record[..count].copy_from_slice(&current[..count]);
    let err = crate::imod::libcfshr::pip_fwrap::pipgetstring_(option, &mut record);
    *string = fortran_string(&record);
    err
}

/// `read(*,'(a)') string` with no `END=`: the record without its newline.
fn read_a() -> String {
    let mut line = String::new();
    if matches!(std::io::stdin().lock().read_line(&mut line), Ok(0) | Err(_)) {
        let _ = std::io::stdout().flush();
        eprintln!("Fortran runtime error: End of file");
        exit(2);
    }
    line.trim_end_matches(['\r', '\n']).to_owned()
}

/// Original program `solvematch` (`solvematch.f90:10`).
///
/// SOLVEMATCH will solve for a 3-dimensional linear transformation relating
/// the tomogram volumes resulting from tilt series around two different
/// axes.  It uses information from the 3-D coordinates of fiducials found by
/// TILTALIGN.
pub fn solvematch() {
    let idim = IDIM as usize;
    let ms = MAT_SIZE as usize;
    // A fresh process image starts the `d3multvars` module at its DATA
    // values; the command may run in-process after another use of it.
    *D3MULTVARS.lock().unwrap() = D3MultVars {
        sum_sq: [0.0; 24],
        aa: [0.0; 9],
        err_min: 0.0,
        null_axis: 0,
        mag_sign: 0,
        if_trace: 0,
        num_trials: 0,
        first_time: true,
        var: [0.0; 6],
    };
    let mut x_mat = vec![0.0_f32; ms * idim];
    let xm = |j: i32, i: i32| (j - 1) as usize + (i - 1) as usize * ms;
    let mut points_a = vec![[0.0_f32; 3]; idim];
    let mut points_b = vec![[0.0_f32; 3]; idim];
    // `a(3,4)`: element `(i, j)` at `(i - 1) + 3 * (j - 1)`.
    let mut a = [0.0_f32; 12];
    let ai = |i: i32, j: i32| (i - 1) as usize + 3 * (j - 1) as usize;
    let mut del_xyz = [0.0_f32; 3];
    let mut dev_xyz_max = [0.0_f32; 3];
    let mut point_rot = [0.0_f32; 3];
    let mut cen_mean_loc = [0.0_f32; 3];
    let mut amat_local = [0.0_f32; 12];
    let mut dxyz_local = [0.0_f32; 3];
    let mut idrop = vec![0_i32; idim];
    let mut ind_orig = vec![0_i32; idim];
    let mut mapped = vec![0_i32; idim];
    let mut map_a_to_b = vec![0_i32; idim];
    // `nxyz(3,2)`; stack storage the source may read before any assignment.
    let mut nxyz = [[0_i32; 3]; 2];
    let jxyz: [i32; 3] = [1, 3, 2];
    // `icontAB(IDIM,2)` with `icontA`, `icontB` its columns, and likewise.
    let mut icont_ab = vec![0_i32; 2 * idim];
    let mut icont_to_point_ab = vec![0_i32; 2 * idim];
    let mut list_corr_ab = vec![0_i32; 2 * idim];
    let c2 = |i: i32, k: i32| (i - 1) as usize + (k - 1) as usize * idim;
    let mut num_ab_points = [0_i32; 2];
    let mut list_use = vec![0_i32; idim];
    let mut transfer_x = vec![0.0_f32; 2 * idim];
    let mut transfer_y = vec![0.0_f32; 2 * idim];
    let mut fid_mod_x = vec![0.0_f32; 2 * idim];
    let mut fid_mod_y = vec![0.0_f32; 2 * idim];
    let mut mod_obj = vec![0_i32; 2 * idim];
    let mut mod_cont = vec![0_i32; 2 * idim];
    let mut iz_best = [0_i32; 2];
    let mut num_fid = [0_i32; 2];
    let mut mod_obj_fid = vec![0_i32; 2 * idim];
    let mut mod_cont_fid = vec![0_i32; 2 * idim];
    // `character*320 filename`
    let mut filename: String;
    let ab_text: [&str; 2] = ["A", "B"];
    let mut bad_axis1: char;
    let mut bad_axis2: char;
    let min_num_to_start: i32;
    let mut num_list: i32;
    let mut num_list_a: i32;
    let mut num_list_b: i32;
    let mut num_data: i32;
    // Set only on PIP input (`solvematch.f90:427-429`) but read for every
    // run at `:843`; natively stack residue in interactive entry.  Fixed in
    // translation (BUGS.md): they start at 0 (not entered), so interactive
    // runs never give the "points on one surface" advice.
    let mut if_zshifts = 0_i32;
    let mut if_angle_ofs = 0_i32;
    let mut ia: i32;
    let mut num_surf: i32;
    let mut num_mod_pts: i32;
    let mut ipt: i32;
    let mut ip: i32;
    let mut ndata = 0_i32;
    let mut if_added: i32;
    let mut ipnt_max = 0_i32;
    let mut ipt_a: i32;
    let mut ipt_b: i32;
    let mut ipt_a_at_min = 0_i32;
    let mut ipt_b_at_min = 0_i32;
    let mut max_drop: i32;
    let mut num_drop = 0_i32;
    let mut iofs: i32;
    let ierr_factor: i32;
    let add_ratio: f32;
    let add_crit: f32;
    let mut dist_min: f32;
    let mut dev_avg = 0.0_f32;
    let mut dev_sd = 0.0_f32;
    let mut dev_max = 0.0_f32;
    let mut dx: f32;
    let mut dist: f32;
    let crit_prob: f32;
    let elim_min: f32;
    let abs_prob_crit: f32;
    let mut stop_limit: f32;
    let mut xtilt_a: f32;
    let mut xtilt_b: f32;
    let mut dy: f32;
    let mut dz: f32;
    let mut a_scale: f32;
    let mut b_scale: f32;
    let mut ierr: i32;
    let mut num_col_fit: i32;
    let mut max_cont_a: i32;
    let mut max_cont_b: i32;
    let mut icol_fixed: i32;
    let (mut xy_scale, mut z_scale) = (0.0_f32, 0.0_f32);
    let (mut x_offset, mut y_offset, mut z_offset) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut if_flip = 0_i32;
    let (mut x_im_scale, mut y_im_scale, mut z_im_scale) = (0.0_f32, 0.0_f32, 0.0_f32);
    let lo_max_avg_ratio: f32;
    let hi_max_avg_ratio: f32;
    let lo_max_lim_ratio: f32;
    let hi_max_lim_ratio: f32;
    let mut a_delta: f32;
    let mut b_delta: f32;
    let mut a_pixel_size: f32;
    let mut b_pixel_size: f32;
    let mut xcen: f32;
    let mut ycen: f32;
    let trans_tol: f32;
    let mut angle_offset_a: f32;
    let mut angle_offset_b: f32;
    let mut z_shift_a: f32;
    let mut z_shift_b: f32;
    let (mut nx_fid_a, mut ny_fid_a, mut nx_fid_b, mut ny_fid_b) = (0_i32, 0_i32, 0_i32, 0_i32);
    let i_trans_a: i32;
    let i_trans_b: i32;
    let if_trans_b_to_a: i32;
    let mut num_trans_coord: i32;
    let mut iz_ind: i32;
    let mut ind_a: i32;
    let ind_b: i32;
    let mut n_list_use: i32;
    let mut local_num: i32;
    let mut inverted_in_depth = 0_i32;
    let num_local_x: i32;
    let num_local_y: i32;
    let mut num_big: i32;
    let ind_orig_at_max: i32;
    let mut lim_raised: i32;
    let (mut xmin, mut xmax, mut ymin, mut ymax): (f32, f32, f32, f32);
    let mut size: f32;
    let target_size: f32;
    let mut dx_local: f32;
    let mut dy_local: f32;
    let mut sum_mean: f32;
    let mut sum_max: f32;
    let mut shift_limit: f32;
    let cen_shift_x: f32;
    let cen_shift_y: f32;
    let cen_shift_z: f32;
    // Read before assignment only when `devAllMax > 0`, which the local
    // fits that set it also establish.
    let mut dev_avg_loc = 0.0_f32;
    let mut dev_max_loc = 0.0_f32;
    let mut dev_all_max: f32;
    let glob_loc_avg_ratio: f32;
    let mut axis_crit: f32;
    let mut report_diff = 0.0_f32;
    let yz_scale_diff: f32;
    let mut sum_sq: f32;
    let xy_scale_diff: f32;
    let xz_scale_diff: f32;
    let mut axis_scale = [0.0_f32; 3];
    let mut trans_pixel: f32;
    let mut free_input = [0.0_f32; 20];
    let determ_positive: f32;
    let determ_inverted: f32;
    let mut relative_fids: bool;
    let mut match_a_to_b: bool;
    let mut invert_entered: bool;
    let pip_input: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    // `character*6 modelOption(2)`, `character*9 tomoOption(2)`
    let model_option: [&str; 2] = ["AMatch", "BMatch"];
    let tomo_option: [&str; 2] = ["ATomogram", "BTomogram"];
    // `character*10240 listString`
    let mut list_string = String::new();
    // `use fortmodel`
    let mut fm = FortModel::default();
    let blank = |string: &str| string.bytes().all(|b| b == b' ');
    let mut out = std::io::stdout();
    //
    // don't add a point if it's this much higher than the limit for
    // quitting
    //
    add_ratio = 1.;
    //
    // require at least this many points to start
    //
    min_num_to_start = 4;
    //
    // initialize these items here in case no fiducials are entered
    //
    num_ab_points[0] = 0;
    num_ab_points[1] = 0;
    num_list_a = 0;
    num_list_b = 0;
    num_data = 0;
    num_surf = 0;
    num_col_fit = 3;
    icol_fixed = 0;
    filename = String::new();
    xtilt_a = 0.;
    xtilt_b = 0.;
    angle_offset_a = 0.;
    angle_offset_b = 0.;
    local_num = 0;
    shift_limit = 10.;
    z_shift_a = 0.;
    z_shift_b = 0.;
    stop_limit = 8.;
    a_scale = 1.;
    b_scale = 1.;
    a_delta = 0.;
    b_delta = 0.;
    a_pixel_size = 0.;
    b_pixel_size = 0.;
    relative_fids = true;
    ind_a = 1;
    let mut ind_b_init = 2;
    trans_tol = 3.;
    num_trans_coord = 0;
    match_a_to_b = false;
    axis_crit = 10.;
    trans_pixel = 0.;
    invert_entered = false;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "solvematch",
        "ERROR: SOLVEMATCH - ",
        true,
        3,
        0,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pip_input = num_opt_arg + num_non_opt_arg > 0;
    allocate_fort_model(&mut fm);
    //
    // Get first fid filename if any
    // Also get the surfaces option; if it's -2, then force models only
    // even if there are fids
    //
    if pip_input {
        ierr = pip_get_logical("MatchingAtoB", &mut match_a_to_b);
        if match_a_to_b {
            ind_a = 2;
        }
        ind_b_init = 3 - ind_a;
        ierr = get_string(b"AFiducialFile", &mut filename, 320);
        num_surf = 2;
        ierr = pip_get_integer(b"SurfacesOrUseModels", &mut num_surf);
        if num_surf == -2 {
            num_surf = 0;
            filename = String::new();
        }
        invert_entered = pip_get_integer(b"InvertedInDepth", &mut inverted_in_depth) == 0;
    } else {
        print!(
            " Name of file with 3-D fiducial coordinates for first tilt series,\n or Return to align with matching model files only: "
        );
        let _ = out.flush();
        filename = read_a();
    }
    ind_b = ind_b_init;
    let _ = ierr;
    //
    // If no fiducials are to be used, skip down to maximum residual entry
    // after initializing some variables
    max_cont_a = 0;
    max_cont_b = 0;
    if !blank(&filename) {
        //
        num_col_fit = 4;
        {
            let (col_a, _) = icont_ab.split_at_mut(idim);
            get_fiducials(
                &filename,
                col_a,
                &mut points_a,
                &mut num_ab_points[0],
                &mut mod_obj[..idim],
                &mut mod_cont[..idim],
                IDIM,
                "first tomogram",
                &mut a_pixel_size,
                &mut nx_fid_a,
                &mut ny_fid_a,
            );
        }
        //
        if pip_input {
            if get_string(b"BFiducialFile", &mut filename, 320) > 0 {
                exit_error("No file specified for fiducials in second tilt series");
            }
        } else {
            print!(" Name of file with 3-D fiducial coordinates for second tilt series: ");
            let _ = out.flush();
            filename = read_a();
        }
        {
            let (_, col_b) = icont_ab.split_at_mut(idim);
            get_fiducials(
                &filename,
                col_b,
                &mut points_b,
                &mut num_ab_points[1],
                &mut mod_obj[idim..],
                &mut mod_cont[idim..],
                IDIM,
                "second tomogram",
                &mut b_pixel_size,
                &mut nx_fid_b,
                &mut ny_fid_b,
            );
        }
        //
        // fill listcorr with actual contour numbers, and find maximum contours
        //
        for i in 1..=num_ab_points[0] {
            list_corr_ab[c2(i, 1)] = icont_ab[c2(i, 1)];
            map_a_to_b[i as usize - 1] = 0;
            max_cont_a = max_cont_a.max(icont_ab[c2(i, 1)]);
        }
        for i in 1..=num_ab_points[1] {
            list_corr_ab[c2(i, 2)] = icont_ab[c2(i, 2)];
            mapped[i as usize - 1] = 0;
            max_cont_b = max_cont_b.max(icont_ab[c2(i, 2)]);
        }
        //
        // build index from contour numbers to points in array
        //
        if max_cont_a > IDIM || max_cont_b > IDIM {
            exit_error("Contour numbers too high for arrays");
        }
        for j in 1..=2 {
            for i in 1..=max_cont_a.max(max_cont_b) {
                icont_to_point_ab[c2(i, j)] = 0;
            }
            for i in 1..=num_ab_points[j as usize - 1] {
                // `solvematch.f90:207` stores at a zero or negative point
                // number too, writing outside the column natively.  Fixed in
                // translation (BUGS.md): such a point is not entered in the
                // lookup table (no point can be found by that number).
                let icont = icont_ab[c2(i, j)];
                if icont >= 1 {
                    icont_to_point_ab[c2(icont, j)] = i;
                }
            }
        }
        //
        if pip_input && get_string(b"TransferCoordinateFile", &mut filename, 320) == 0 {
            //
            // Read transfer file, get best section and coordinates
            //
            let mut unit1 = BufReader::new(dopen(1, filename.trim_end_matches(' '), "ro", "f"));
            {
                let mut record = String::new();
                // `read(1, '(a)') listString` with no `END=`.
                if matches!(unit1.read_line(&mut record), Ok(0) | Err(_)) {
                    let _ = out.flush();
                    eprintln!("Fortran runtime error: End of file");
                    exit(2);
                }
                list_string = record.trim_end_matches(['\r', '\n']).to_owned();
                list_string.truncate(10240);
            }
            let mut j = 0_i32;
            frefor(&list_string, &mut free_input, &mut j);
            iz_best[0] = nint(free_input[0]);
            iz_best[1] = nint(free_input[1]);
            if_trans_b_to_a = nint(free_input[2]);
            //
            // If there is a fourth value
            // then get the scaling to apply to model coords to match transfer coord
            if j > 3 {
                trans_pixel = free_input[3];
            }
            //
            let mut i = 0_i32;
            loop {
                let (xa, xb) = transfer_x.split_at_mut(idim);
                let (ya, yb) = transfer_y.split_at_mut(idim);
                let k = i as usize;
                let res = list_read(
                    &mut unit1,
                    &mut [
                        ListItem::Real(&mut xa[k]),
                        ListItem::Real(&mut ya[k]),
                        ListItem::Real(&mut xb[k]),
                        ListItem::Real(&mut yb[k]),
                    ],
                );
                if res.is_err() {
                    break;
                }
                i += 1;
                if i >= IDIM {
                    exit_error("Too many transfer coordinates for arrays");
                }
            }
            num_trans_coord = i;
            drop(unit1);
            //
            // Set up indexes to best z for the first and second model, and for
            // accessing the absolute A or B axis transfer coords and fid coords
            //
            iz_ind = 1;
            if match_a_to_b != (if_trans_b_to_a > 0) {
                iz_ind = 2;
            }
            let mut ita = 1;
            if if_trans_b_to_a > 0 {
                ita = 2;
            }
            i_trans_a = ita;
            i_trans_b = 3 - i_trans_a;

            //
            // Read the two fiducial models and get coordinates at best Z and
            // their objects and contours
            //
            {
                let ka = (ind_a - 1) as usize * idim;
                read_fid_model_file(
                    "AFiducialModel",
                    iz_best[iz_ind as usize - 1],
                    ab_text[ind_a as usize - 1],
                    &mut fid_mod_x[ka..ka + idim],
                    &mut fid_mod_y[ka..ka + idim],
                    &mut mod_obj_fid[ka..ka + idim],
                    &mut mod_cont_fid[ka..ka + idim],
                    &mut num_fid[ind_a as usize - 1],
                    IDIM,
                    trans_pixel,
                    &mut fm,
                );
                let kb = (ind_b - 1) as usize * idim;
                read_fid_model_file(
                    "BFiducialModel",
                    iz_best[(3 - iz_ind) as usize - 1],
                    ab_text[ind_b as usize - 1],
                    &mut fid_mod_x[kb..kb + idim],
                    &mut fid_mod_y[kb..kb + idim],
                    &mut mod_obj_fid[kb..kb + idim],
                    &mut mod_cont_fid[kb..kb + idim],
                    &mut num_fid[ind_b as usize - 1],
                    IDIM,
                    trans_pixel,
                    &mut fm,
                );
            }
            //
            // Get the list of points to use from the true A series
            //
            n_list_use = num_ab_points[ind_a as usize - 1];
            for i in 1..=num_ab_points[ind_a as usize - 1] {
                list_use[i as usize - 1] = icont_ab[c2(i, ind_a)];
            }
            if get_string(b"UsePoints", &mut list_string, 10240) == 0 {
                let _ = parselist2(
                    &list_string,
                    &mut list_use,
                    &mut n_list_use,
                    &mut IDIM.clone(),
                );
            }
            //
            // Build list of corresponding points
            //
            num_list = 0;
            for i in 1..=n_list_use {
                let lu = list_use[i as usize - 1];
                if lu > 0
                    && lu <= max_cont_a.max(max_cont_b)
                    && icont_to_point_ab[c2(lu, ind_a)] > 0
                {
                    //
                    // IF the point in A is a legal point from fid file, find the
                    // same object/contour in the fiducial model
                    //
                    ipt_a = icont_to_point_ab[c2(lu, ind_a)];
                    ip = 0;
                    for j in 1..=num_fid[0] {
                        if mod_obj[c2(ipt_a, ind_a)] == mod_obj_fid[c2(j, 1)]
                            && mod_cont[c2(ipt_a, ind_a)] == mod_cont_fid[c2(j, 1)]
                        {
                            ip = j;
                        }
                    }
                    if ip > 0 {
                        //
                        // Now match the model coords to the transfer coords
                        //
                        ia = 0;
                        for j in 1..=num_trans_coord {
                            let ddx = fid_mod_x[c2(ip, 1)] - transfer_x[c2(j, i_trans_a)];
                            let ddy = fid_mod_y[c2(ip, 1)] - transfer_y[c2(j, i_trans_a)];
                            if (ddx * ddx + ddy * ddy).sqrt() < trans_tol {
                                ia = j;
                            }
                        }
                        if ia > 0 {
                            //
                            // Found one, then look for a match to the B transfer coords
                            // in the B fiducial model
                            //
                            ipt = 0;
                            for j in 1..=num_trans_coord {
                                let ddx = fid_mod_x[c2(j, 2)] - transfer_x[c2(ia, i_trans_b)];
                                let ddy = fid_mod_y[c2(j, 2)] - transfer_y[c2(ia, i_trans_b)];
                                if (ddx * ddx + ddy * ddy).sqrt() < trans_tol {
                                    ipt = j;
                                }
                            }
                            if ipt > 0 {
                                //
                                // Find the obj/cont in the points of the fid file
                                //
                                ipt_b = 0;
                                for j in 1..=num_fid[1] {
                                    if mod_obj[c2(j, ind_b)] == mod_obj_fid[c2(ipt, 2)]
                                        && mod_cont[c2(j, ind_b)] == mod_cont_fid[c2(ipt, 2)]
                                    {
                                        ipt_b = j;
                                    }
                                }
                                if ipt_b > 0 {
                                    //
                                    // check for uniqueness
                                    //
                                    num_list_b = 0;
                                    for j in 1..=num_list {
                                        if list_corr_ab[c2(j, ind_b)] == icont_ab[c2(ipt_b, ind_b)]
                                        {
                                            num_list_b = 1;
                                        }
                                    }
                                    if num_list_b == 0 {
                                        //
                                        // Bingo: we have corresponding points
                                        //
                                        num_list += 1;
                                        list_corr_ab[c2(num_list, ind_a)] = lu;
                                        list_corr_ab[c2(num_list, ind_b)] =
                                            icont_ab[c2(ipt_b, ind_b)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if num_list == 0 {
                exit_error("No corresponding points found using transfer coords");
            }
            num_list_a = num_list;
            num_list_b = num_list;
        } else {
            //
            // No transfer coordinates, get corresponding lists the old way
            //
            num_list = num_ab_points[0].min(num_ab_points[1]);
            num_list_a = num_list;
            if pip_input {
                if get_string(b"ACorrespondenceList", &mut list_string, 10240) == 0 {
                    let _ = parselist2(
                        &list_string,
                        &mut list_corr_ab[..idim],
                        &mut num_list_a,
                        &mut IDIM.clone(),
                    );
                }
            } else {
                println!(
                    "Enter a list of points in the first series for which you are sure of the\n corresponding point in the second series (Ranges are OK;\n enter / if the first{} points are in one-to-one correspondence between the series",
                    fmt_i(num_list, 3)
                );
                let _ = rdlist2(
                    &mut std::io::stdin().lock(),
                    &mut list_corr_ab[..idim],
                    &mut num_list_a,
                    &mut IDIM.clone(),
                );
            }
            if num_list_a > IDIM {
                exit_error("Too many points for arrays");
            }
            if num_list_a > num_list {
                exit_error(
                    "You have entered more numbers than the minimum number of points in A and B",
                );
            }
            //
            num_list_b = num_list_a;
            if pip_input {
                if get_string(b"BCorrespondenceList", &mut list_string, 10240) == 0 {
                    let _ = parselist2(
                        &list_string,
                        &mut list_corr_ab[idim..],
                        &mut num_list_b,
                        &mut IDIM.clone(),
                    );
                }
            } else {
                print!(
                    "Enter a list of the corresponding points in the second series \n - enter / for "
                );
                let _ = out.flush();
                wrlist(&list_corr_ab[idim..], &num_list);
                let _ = rdlist2(
                    &mut std::io::stdin().lock(),
                    &mut list_corr_ab[idim..],
                    &mut num_list_b,
                    &mut IDIM.clone(),
                );
            }
        }
        //
        if num_list_a < min_num_to_start {
            println!();
            println!(
                " ERROR: SOLVEMATCH - Need at least {:>11}  points to get started",
                min_num_to_start
            );
            exit(1);
        }
        if num_list_b != num_list_a {
            println!();
            println!(
                " ERROR: SOLVEMATCH - You must have the same number  of entries in each listyou made {:>11}  and {:>11}  entries for lists {} and {}",
                num_list_a,
                num_list_b,
                ab_text[ind_a as usize - 1],
                ab_text[ind_b as usize - 1]
            );
            exit(1);
        }
        //
        // check legality and build map lists
        //
        for i in 1..=num_list_a {
            let lca = list_corr_ab[c2(i, 1)];
            let lcb = list_corr_ab[c2(i, 2)];
            if lca <= 0 || lcb <= 0 {
                exit_error("You entered a point number less than or equal to zero");
            }
            if lca > max_cont_a {
                exit_error(&format!(
                    " You entered a point number higher than the number of points in {}",
                    ab_text[ind_a as usize - 1]
                ));
            }
            ipt_a = icont_to_point_ab[c2(lca, 1)];
            if ipt_a == 0 {
                exit_error(&format!(
                    " You entered a point number that is not included in the points from {}",
                    ab_text[ind_a as usize - 1]
                ));
            }
            if lcb > max_cont_b {
                exit_error(&format!(
                    " You entered a point number higher than the number of points in {}",
                    ab_text[ind_b as usize - 1]
                ));
            }
            ipt_b = icont_to_point_ab[c2(lcb, 2)];
            if ipt_b == 0 {
                exit_error(&format!(
                    " You entered a point number that is not included in the points from {}",
                    ab_text[ind_b as usize - 1]
                ));
            }
            if map_a_to_b[ipt_a as usize - 1] != 0 {
                println!();
                println!(
                    " ERROR: SOLVEMATCH - Point # {:>11}  IN {} referred to twice",
                    lca,
                    ab_text[ind_a as usize - 1]
                );
                exit(1);
            } else if mapped[ipt_b as usize - 1] != 0 {
                println!();
                println!(
                    " ERROR: SOLVEMATCH - Point # {:>11}  IN {} referred to twice",
                    lcb,
                    ab_text[ind_b as usize - 1]
                );
                exit(1);
            }
            map_a_to_b[ipt_a as usize - 1] = ipt_b;
            mapped[ipt_b as usize - 1] = ipt_a;
        }
        //
        // Get tilt axis angle, plus try to get tomogram pixel size (delta)
        // and compute scaling factors, overriden by an entry
        //
        if pip_input {
            let _ = pip_get_two_floats(b"XAxisTilts", &mut xtilt_a, &mut xtilt_b);
            if_zshifts = 1 - pip_get_two_floats(b"ZShiftsToTilt", &mut z_shift_a, &mut z_shift_b);
            if_angle_ofs = 1 - pip_get_two_floats(
                b"AngleOffsetsToTilt",
                &mut angle_offset_a,
                &mut angle_offset_b,
            );
            let _ = pip_get_integer(b"LocalFitting", &mut local_num);
            let _ = pip_get_float(b"CenterShiftLimit", &mut shift_limit);
            let _ = pip_get_float(b"AnisotropicLimit", &mut axis_crit);
            get_delta("ATomogramOrSizeXYZ", &mut a_delta, &mut nxyz[0]);
            get_delta("BTomogramOrSizeXYZ", &mut b_delta, &mut nxyz[1]);
            if a_delta * a_pixel_size > 0. {
                a_scale = a_pixel_size / a_delta;
            }
            if b_delta * b_pixel_size > 0. {
                b_scale = b_pixel_size / b_delta;
            }
            ierr_factor = pip_get_two_floats(b"ScaleFactors", &mut a_scale, &mut b_scale);
            //
            // Conditions for absolute fiducials are that the pixel sizes be
            // available (meaning new absolute coordinates) and that either
            // the tomogram pixel sizes were available too or scale factors
            // were entered and tomogram sizes were provided by getDelta
            //
            if num_surf != 0
                && a_pixel_size > 0.
                && b_pixel_size > 0.
                && (a_delta > 0. || (ierr_factor == 0 && a_delta == 0.))
                && (b_delta > 0. || (ierr_factor == 0 && b_delta == 0.))
            {
                //
                // Shift the original X and Y to the center of the volume
                // have to divide size by scale because points aren't scaled up yet
                // Use the actual fiducial size if it exists
                //
                xcen = 0.5_f32 * nxyz[0][0] as f32 / a_scale;
                ycen = 0.5_f32 * nxyz[0][jxyz[1] as usize - 1] as f32 / a_scale;
                if nx_fid_a > 0 && ny_fid_a > 0 {
                    xcen = 0.5_f32 * nx_fid_a as f32;
                    ycen = 0.5_f32 * ny_fid_a as f32;
                }
                for i in 1..=num_ab_points[0] {
                    points_a[i as usize - 1][0] -= xcen;
                    points_a[i as usize - 1][1] -= ycen;
                }
                //
                xcen = 0.5_f32 * nxyz[1][0] as f32 / b_scale;
                ycen = 0.5_f32 * nxyz[1][jxyz[1] as usize - 1] as f32 / b_scale;
                if nx_fid_b > 0 && ny_fid_b > 0 {
                    xcen = 0.5_f32 * nx_fid_b as f32;
                    ycen = 0.5_f32 * ny_fid_b as f32;
                }
                for i in 1..=num_ab_points[1] {
                    points_b[i as usize - 1][0] -= xcen;
                    points_b[i as usize - 1][1] -= ycen;
                }
                relative_fids = false;
            }
        } else {
            print!(" Tilts around the X-axis applied in generating tomograms A and B: ");
            let _ = out.flush();
            if let Err(err) = list_read(
                &mut std::io::stdin().lock(),
                &mut [ListItem::Real(&mut xtilt_a), ListItem::Real(&mut xtilt_b)],
            ) {
                read_abort(err);
            }
        }
        //
        // Adjust the positions of the points for x axis tilt, scaling, angle
        // offset and z shift when building the tomogram
        //
        rotate_fids(
            &mut points_a,
            num_ab_points[0],
            xtilt_a,
            a_scale,
            angle_offset_a,
            z_shift_a,
        );
        rotate_fids(
            &mut points_b,
            num_ab_points[1],
            xtilt_b,
            b_scale,
            angle_offset_b,
            z_shift_b,
        );
    }
    //
    // 40
    if pip_input {
        let _ = pip_get_float(b"MaximumResidual", &mut stop_limit);
    } else {
        print!(" Maximum residual value above which this program should\n exit with an error: ");
        let _ = out.flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Real(&mut stop_limit)],
        ) {
            read_abort(err);
        }
    }
    let num_a_points = num_ab_points[0];
    let num_b_points = num_ab_points[1];
    //
    // if no fiducials, now skip to model entry section
    //
    if num_a_points != 0 {
        //
        // fill array for regression
        //
        for ia in 1..=num_a_points {
            let map = map_a_to_b[ia as usize - 1];
            if map != 0 {
                num_data += 1;
                for j in 1..=3 {
                    x_mat[xm(j, num_data)] =
                        points_b[map as usize - 1][jxyz[j as usize - 1] as usize - 1];
                    x_mat[xm(j + 5, num_data)] =
                        points_a[ia as usize - 1][jxyz[j as usize - 1] as usize - 1];
                }
                x_mat[xm(4, num_data)] = 1.;
                ind_orig[num_data as usize - 1] = icont_ab[c2(ia, 1)];
            }
        }
        //
        println!(
            " {:>11}  pairs of fiducial points originally specified",
            num_data
        );
        if !pip_input {
            num_surf = 2;
            print!(
                " Enter 0 to solve for displacements using matching model files, or -1, 1 or 2\n  to solve only for 3x3 matrix (2 if fiducials are on 2 surfaces, 1 if they are\n  on one surface and tomograms are NOT inverted, or -1 if fiducials are on one\n  surface and tomograms ARE inverted relative to each other): "
            );
            let _ = out.flush();
            if let Err(err) = list_read(
                &mut std::io::stdin().lock(),
                &mut [ListItem::Integer(&mut num_surf)],
            ) {
                read_abort(err);
            }
            if num_surf.abs() == 1 {
                invert_entered = true;
                inverted_in_depth = (1 - num_surf) / 2;
            }
        }
    }
    //
    // Enter matching models if nsurf is 0
    //
    // 50
    if num_surf == 0 {
        iofs = num_col_fit + 1;
        num_mod_pts = 0;
        for model in 1..=2 {
            //
            // DNM 7/20/02: Changed to just get the nx, ny, nz and not the
            // origin from the image file; to use model header information
            // to scale back to image index coordinates; and to not use or mess
            // up the y-z transposition variable
            //
            if !pip_input {
                println!(
                    " Enter NX, NY, NZ of tomogram{}, or name of tomogram file",
                    fmt_i(model, 2)
                );
            }
            get_nxyz(
                pip_input,
                tomo_option[model as usize - 1],
                "SOLVEMATCH",
                1,
                &mut nxyz[model as usize - 1],
            );
            if pip_input {
                if get_string(
                    model_option[model as usize - 1].as_bytes(),
                    &mut filename,
                    320,
                ) > 0
                {
                    println!();
                    println!(
                        " ERROR: SOLVEMATCH - No matching model for tomogram {:>11}",
                        model
                    );
                    exit(1);
                }
            } else {
                print!(" Name of model file from tomogram{}: ", fmt_i(model, 2));
                let _ = out.flush();
                filename = read_a();
            }
            if !readw_or_imod(filename.trim_end_matches(' '), &mut fm) {
                exit_error("Reading model file");
            }

            let _ = getimodhead(
                &mut xy_scale,
                &mut z_scale,
                &mut x_offset,
                &mut y_offset,
                &mut z_offset,
                &mut if_flip,
            );
            let _ = getimodscales(&mut x_im_scale, &mut y_im_scale, &mut z_im_scale);

            num_mod_pts = 0;
            let nx = &nxyz[model as usize - 1];
            for iobj in 1..=fm.max_mod_obj {
                for ip in 1..=fm.npt_in_obj[iobj as usize - 1] {
                    let ipt = fm.object[(fm.ibase_obj[iobj as usize - 1] + ip) as usize - 1].abs();
                    num_mod_pts += 1;
                    if num_data + num_mod_pts > IDIM {
                        exit_error("Too many points for arrays");
                    }
                    let p = fm.p_coord[ipt as usize - 1];
                    let col = num_data + num_mod_pts;
                    x_mat[xm(1 + iofs, col)] =
                        (p[0] - x_offset) / x_im_scale - 0.5_f32 * nx[0] as f32;
                    x_mat[xm(2 + iofs, col)] =
                        (p[1] - y_offset) / y_im_scale - 0.5_f32 * nx[1] as f32;
                    x_mat[xm(3 + iofs, col)] =
                        (p[2] - z_offset) / z_im_scale - 0.5_f32 * nx[2] as f32;
                    x_mat[xm(4, col)] = 0.;
                    ind_orig[col as usize - 1] = -num_mod_pts;
                    if num_a_points == 0 {
                        ind_orig[col as usize - 1] = num_mod_pts;
                    }
                }
            }
            if model == 1 {
                ndata = num_mod_pts;
            }
            iofs = 0;
        }
        if num_mod_pts != ndata {
            exit_error("# of points does not match between matching models");
        }
        println!(" {:>11}  point pairs from models", num_mod_pts);
        iofs = num_col_fit + 1;
    } else {
        //
        // If no model points
        // get rid of dummy column: fit 3 columns, pack dependent vars to
        // the left, and set the offset for adding more dependent var data
        //
        num_mod_pts = 0;
        num_col_fit = 3;
        for i in 1..=num_data {
            for j in 5..=7 {
                x_mat[xm(j, i)] = x_mat[xm(j + 1, i)];
            }
        }
        iofs = 4;
        //
        // if only one surface, "fix" column 2, encode sign in icolfix
        //
        if num_surf.abs() == 1 {
            icol_fixed = 2;
            if invert_entered && inverted_in_depth > 0 {
                icol_fixed = -2;
            }
        }
    }
    num_data += num_mod_pts;
    //
    // loop to add points that weren't initially indicated - as long as
    // there are any left to add and the last minimum distance was still
    // low enough
    //
    if_added = 0;
    add_crit = add_ratio * stop_limit;
    dist_min = 0.;
    //
    while num_data - num_mod_pts < num_a_points.min(num_b_points) && dist_min < add_crit {
        do3multr(
            &mut x_mat,
            MAT_SIZE,
            num_data,
            num_col_fit,
            num_data,
            icol_fixed,
            &mut a,
            &mut del_xyz,
            &mut cen_mean_loc,
            &mut dev_avg,
            &mut dev_sd,
            &mut dev_max,
            &mut ipnt_max,
            &mut dev_xyz_max,
        );
        dist_min = 1.0e10;
        //
        // apply to each point in B that is not mapped to
        //
        for ipt_b in 1..=num_b_points {
            if mapped[ipt_b as usize - 1] == 0 {
                for ixyz in 1..=3 {
                    point_rot[ixyz as usize - 1] = del_xyz[ixyz as usize - 1];
                    if num_col_fit == 4 {
                        point_rot[ixyz as usize - 1] = del_xyz[ixyz as usize - 1] + a[ai(ixyz, 4)];
                    }
                    for j in 1..=3 {
                        point_rot[ixyz as usize - 1] = point_rot[ixyz as usize - 1]
                            + a[ai(ixyz, j)]
                                * points_b[ipt_b as usize - 1][jxyz[j as usize - 1] as usize - 1];
                    }
                }
                //
                // search through points in A that don't have map
                //
                for ipt_a in 1..=num_a_points {
                    if map_a_to_b[ipt_a as usize - 1] == 0 {
                        let pa = &points_a[ipt_a as usize - 1];
                        dx = pa[jxyz[0] as usize - 1] - point_rot[0];
                        if dx.abs() < dist_min {
                            dy = pa[jxyz[1] as usize - 1] - point_rot[1];
                            if dy.abs() < dist_min {
                                dz = pa[jxyz[2] as usize - 1] - point_rot[2];
                                if dz.abs() < dist_min {
                                    dist = (dx * dx + dy * dy + dz * dz).sqrt();
                                    if dist < dist_min {
                                        ipt_a_at_min = ipt_a;
                                        ipt_b_at_min = ipt_b;
                                        dist_min = dist;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        //
        // add the closest fitting point-pair
        //
        if dist_min < add_crit {
            if_added = 1;
            num_data += 1;
            map_a_to_b[ipt_a_at_min as usize - 1] = ipt_b_at_min;
            mapped[ipt_b_at_min as usize - 1] = ipt_a_at_min;
            for j in 1..=3 {
                x_mat[xm(j, num_data)] =
                    points_b[ipt_b_at_min as usize - 1][jxyz[j as usize - 1] as usize - 1];
                x_mat[xm(j + iofs, num_data)] =
                    points_a[ipt_a_at_min as usize - 1][jxyz[j as usize - 1] as usize - 1];
            }
            x_mat[xm(4, num_data)] = 1.;
            ind_orig[num_data as usize - 1] = icont_ab[c2(ipt_a_at_min, 1)];
        }
    }
    //
    println!(
        " {:>11}  pairs of points are available for fitting",
        num_data
    );
    if if_added != 0 || num_trans_coord > 0 {
        //
        // rebuild lists of actual contour numbers
        //
        num_list_a = 0;
        for i in 1..=num_a_points {
            let map = map_a_to_b[i as usize - 1];
            if map != 0 {
                num_list_a += 1;
                list_corr_ab[c2(num_list_a, 1)] = icont_ab[c2(i, 1)];
                list_corr_ab[c2(num_list_a, 2)] = icont_ab[c2(map, 2)];
            }
        }
        println!(
            " In the final list of correspondences used for fits, points from {} are:",
            ab_text[ind_a as usize - 1]
        );
        let _ = out.flush();
        wrlist(&list_corr_ab[..idim], &num_list_a);
        println!(" Points from {} are:", ab_text[ind_b as usize - 1]);
        let _ = out.flush();
        wrlist(&list_corr_ab[idim..], &num_list_a);
    }

    //
    // Now figure out whether Z inversion is needed or not for numSurf = +/-1
    if num_surf.abs() == 1 && !invert_entered {
        do3multr(
            &mut x_mat,
            MAT_SIZE,
            num_data,
            num_col_fit,
            num_data,
            icol_fixed,
            &mut a,
            &mut del_xyz,
            &mut cen_mean_loc,
            &mut dev_avg,
            &mut dev_sd,
            &mut dev_max,
            &mut ipnt_max,
            &mut dev_xyz_max,
        );
        determ_positive = determ3(&a);
        do3multr(
            &mut x_mat,
            MAT_SIZE,
            num_data,
            num_col_fit,
            num_data,
            -icol_fixed,
            &mut a,
            &mut del_xyz,
            &mut cen_mean_loc,
            &mut dev_avg,
            &mut dev_sd,
            &mut dev_max,
            &mut ipnt_max,
            &mut dev_xyz_max,
        );
        determ_inverted = determ3(&a);
        if (determ_positive > 0.) == (determ_inverted > 0.) {
            exit_error(
                "Cannot determine if there is an inversion in the depth dimension; add InvertedInDepth option",
            );
        }
        if determ_inverted > 0. {
            icol_fixed = -icol_fixed;
        }
    }
    //
    max_drop = nint(0.1_f32 * (num_data - 1) as f32);
    crit_prob = 0.01;
    elim_min = 3.;
    abs_prob_crit = 0.002;
    solve_wo_outliers(
        &mut x_mat,
        MAT_SIZE,
        num_data,
        num_col_fit,
        icol_fixed,
        max_drop,
        crit_prob,
        abs_prob_crit,
        elim_min,
        &mut idrop,
        &mut num_drop,
        &mut a,
        &mut del_xyz,
        &mut cen_mean_loc,
        &mut dev_avg,
        &mut dev_sd,
        &mut dev_max,
        &mut ipnt_max,
        &mut dev_xyz_max,
    );
    //
    if num_drop != 0 {
        // format 104, with the `(9i7)` group reverting to a new record
        let mut text = format!(
            "\n{} points dropped by outlier elimination; residual mean ={}, SD ={}\n point # in {}:",
            fmt_i(num_drop, 3),
            fmt_f(dev_avg, 7, 2),
            fmt_f(dev_sd, 7, 2),
            ab_text[ind_a as usize - 1]
        );
        for i in 1..=num_drop {
            if i > 1 && (i - 1) % 9 == 0 {
                text.push('\n');
            }
            text.push_str(&fmt_i(ind_orig[idrop[i as usize - 1] as usize - 1], 7));
        }
        println!("{text}");
        // format 115
        let mut text = String::from(" deviations  :");
        for (n, i) in (num_data + 1 - num_drop..=num_data).enumerate() {
            if n > 0 && n % 9 == 0 {
                text.push('\n');
            }
            text.push_str(&fmt_f(x_mat[xm(num_col_fit + 1, i)], 7, 1));
        }
        println!("{text}");
    }
    //
    ind_orig_at_max = ind_orig[ipnt_max as usize - 1];
    if ind_orig_at_max <= max_cont_a && mod_obj[c2(1, 1)] > 0 {
        // A matching-model point carries a negative `indOrig` (`:576`),
        // and native evaluates `icontToPointA(indOrigAtMax)` and then
        // `modObj(iptA, 1)` outside both arrays (`:733-738`), landing on
        // zeros in the reference build.  Fixed in translation (BUGS.md): a
        // point with no fiducial-file entry prints object and contour 0,
        // which is what native prints there.
        ipt_a = if ind_orig_at_max >= 1 {
            icont_to_point_ab[c2(ind_orig_at_max, 1)]
        } else {
            0
        };
        let (obj, cont) = if ipt_a >= 1 {
            (mod_obj[c2(ipt_a, 1)], mod_cont[c2(ipt_a, 1)])
        } else {
            (0, 0)
        };
        //
        // BRT is using 'Mean residual' as tag
        println!(
            "\n\n Mean residual{},  maximum{} at point #{} (Obj{} cont{} in {})",
            fmt_f(dev_avg, 8, 3),
            fmt_f(dev_max, 9, 3),
            fmt_i(ind_orig_at_max, 4),
            fmt_i(obj, 3),
            fmt_i(cont, 4),
            ab_text[ind_a as usize - 1]
        );
    } else {
        println!(
            "\n\n Mean residual{},  maximum{} at point #{} (in {})",
            fmt_f(dev_avg, 8, 3),
            fmt_f(dev_max, 9, 3),
            fmt_i(ind_orig_at_max, 4),
            ab_text[ind_a as usize - 1]
        );
    }
    println!(
        "  Deviations:{}{}{}",
        fmt_f(dev_xyz_max[0], 9, 3),
        fmt_f(dev_xyz_max[1], 9, 3),
        fmt_f(dev_xyz_max[2], 9, 3)
    );
    //
    if num_col_fit > 3 {
        //
        // fit to both: report the dummy variable offset, make sure that
        // the dxyz are non-zero for matchshifts scanning
        //
        println!(
            "\n X, Y, Z offsets for fiducial dummy variable:{}{}{}",
            fmt_f(a[ai(1, 4)], 10, 3),
            fmt_f(a[ai(2, 4)], 10, 3),
            fmt_f(a[ai(3, 4)], 10, 3)
        );
        if del_xyz[0].abs() < 0.001 && del_xyz[1].abs() < 0.001 && del_xyz[2].abs() < 0.001 {
            del_xyz[0] = 0.0013;
        }
        //
    } else if num_mod_pts == 0 && relative_fids {
        //
        // No matching models and relative fiducials: set dxyz to zero
        // and report the offset from the fit
        //
        println!(
            "\n X, Y, Z offsets from fiducial fit:{}{}{}",
            fmt_f(del_xyz[0], 10, 3),
            fmt_f(del_xyz[1], 10, 3),
            fmt_f(del_xyz[2], 10, 3)
        );
        del_xyz[0] = 0.;
        del_xyz[1] = 0.;
        del_xyz[2] = 0.;
        let mut value = vec![b' '; 320];
        if imodgetenv(b"SOLVEMATCH_TEST", &mut value) != 0 {
            exit_error(
                "Relative fiducial coordinates are no longer allowed; rerun Tiltalign for both axes to get absolute coordinates",
            );
        }
    } else {
        //
        // Absolute fiducials: just make sure they are not exactly zero
        //
        if del_xyz[0].abs() < 0.001 && del_xyz[1].abs() < 0.001 && del_xyz[2].abs() < 0.001 {
            del_xyz[0] = 0.0013;
        }
    }
    //
    println!();
    println!(" Transformation matrix for matchvol:");
    let matrix_text = |a: &[f32; 12], del_xyz: &[f32; 3]| -> String {
        let mut text = String::new();
        for i in 1..=3 {
            text.push_str(&format!(
                "{}{}{}{}\n",
                fmt_f(a[ai(i, 1)], 10, 6),
                fmt_f(a[ai(i, 2)], 10, 6),
                fmt_f(a[ai(i, 3)], 10, 6),
                fmt_f(del_xyz[i as usize - 1], 10, 3)
            ));
        }
        text
    };
    print!("{}", matrix_text(&a, &del_xyz));
    //
    // Compute scaling of vectors and report it
    // Then give warnings of unequal scalings
    for j in 1..=3 {
        sum_sq = 0.;
        for i in 1..=3 {
            sum_sq += a[ai(i, j)] * a[ai(i, j)];
        }
        axis_scale[j as usize - 1] = sum_sq.sqrt();
    }
    //
    // BRT is using 'Scaling along' as tag
    println!(
        "\nScaling along the three axes - X:{}  Y:{}  Z:{}",
        fmt_f(axis_scale[0], 7, 3),
        fmt_f(axis_scale[1], 7, 3),
        fmt_f(axis_scale[2], 7, 3)
    );
    xy_scale_diff = 100.0_f32 * ((axis_scale[0] - axis_scale[1]) / axis_scale[0]).abs();
    xz_scale_diff = 100.0_f32 * ((axis_scale[0] - axis_scale[2]) / axis_scale[0]).abs();
    yz_scale_diff = 100.0_f32 * ((axis_scale[1] - axis_scale[2]) / axis_scale[2]).abs();
    bad_axis1 = ' ';
    bad_axis2 = ' ';
    if axis_crit > 0. {
        if xy_scale_diff > axis_crit && xz_scale_diff > axis_crit && yz_scale_diff > axis_crit {
            report_diff = xy_scale_diff;
            if xz_scale_diff < report_diff {
                report_diff = xz_scale_diff;
            }
            if yz_scale_diff < report_diff {
                report_diff = yz_scale_diff;
            }
            println!(
                "\nWARNING: The scalings along all three axes differ from each other by more than{}%",
                fmt_f(report_diff, 3, 0)
            );
            bad_axis1 = 'A';
        } else if xy_scale_diff > axis_crit && xz_scale_diff > axis_crit {
            bad_axis1 = 'X';
            report_diff = 0.5_f32 * (xy_scale_diff + xz_scale_diff);
        } else if xy_scale_diff > axis_crit && yz_scale_diff > axis_crit {
            bad_axis1 = 'Y';
            report_diff = 0.5_f32 * (xy_scale_diff + yz_scale_diff);
        } else if xz_scale_diff > axis_crit && yz_scale_diff > axis_crit {
            bad_axis1 = 'Z';
            report_diff = 0.5_f32 * (xz_scale_diff + yz_scale_diff);
        } else if xy_scale_diff > axis_crit {
            bad_axis1 = 'X';
            bad_axis2 = 'Y';
            report_diff = xy_scale_diff;
        } else if xz_scale_diff > axis_crit {
            bad_axis1 = 'X';
            bad_axis2 = 'Z';
            report_diff = xz_scale_diff;
        } else if yz_scale_diff > axis_crit {
            bad_axis1 = 'Y';
            bad_axis2 = 'Z';
            report_diff = yz_scale_diff;
        }
    }
    if bad_axis2 != ' ' {
        println!(
            "\nWARNING: The scaling along the {} and {} axes differ by{}%",
            bad_axis1,
            bad_axis2,
            fmt_f(report_diff, 4, 0)
        );
    } else if bad_axis1 != 'A' && bad_axis1 != ' ' {
        println!(
            "\nWARNING: The scaling along the {} axis differs from the other two axes by{}%",
            bad_axis1,
            fmt_f(report_diff, 4, 0)
        );
    }
    //
    // Issue specific dual-axis warning with advice
    // BRT is looking for 'Try specifying' and 'on one surface'
    if bad_axis1 == 'Y' && num_surf == 2 && (if_angle_ofs != 0 || if_zshifts != 0) {
        println!(
            "WARNING: Y scaling is probably wrong because you specified that points are on\nWARNING:    two surfaces but there are too few points on one surface.\nWARNING:    Try specifying that points are on one surface"
        );
    }

    filename = String::new();
    if pip_input {
        if pip_get_in_out_file("OutputFile", 1, " ", &mut filename, 320) != 0 {
            exit_error("No output file specified");
        }
    } else {
        println!(" Enter name of file to place transformation in, or Return for none");
        filename = read_a();
    }
    if !blank(&filename) {
        let unit = dopen(1, filename.trim_end_matches(' '), "new", "f");
        let mut writer = BufWriter::new(unit);
        let _ = writer.write_all(matrix_text(&a, &del_xyz).as_bytes());
        let _ = writer.flush();
    }
    //
    ierr = 0;
    lim_raised = (dev_max + 1.2_f32) as i32;
    dev_all_max = 0.;
    if dev_max > stop_limit && num_mod_pts == 0 && local_num > 0 {
        if local_num < 6 || num_data <= local_num {
            if local_num < 6 {
                println!("\nERROR: SOLVEMATCH - Local fits must have a minimum of 6 points");
            }
            if num_data <= local_num {
                println!(
                    "\nLocal fitting is not available because the number of matched points\n is no bigger than the minimum for local fitting"
                );
            }
        } else {
            //
            // For local fits, get extent of the data
            //
            xmin = 1.0e10;
            xmax = -xmin;
            ymin = 1.0e10;
            ymax = -ymin;
            for ia in 1..=num_a_points {
                if map_a_to_b[ia as usize - 1] > 0 {
                    let pa = &points_a[ia as usize - 1];
                    // `solvematch.f90:884-887`: the running extremes are
                    // the `minss`/`maxss` destinations in the reference object.
                    xmin = minss(xmin, pa[0]);
                    ymin = minss(ymin, pa[1]);
                    xmax = maxss(xmax, pa[0]);
                    ymax = maxss(ymax, pa[1]);
                }
            }
            //
            // Set up number and interval between local areas
            //
            target_size =
                (local_num as f32 * (xmax - xmin) * (ymax - ymin) / num_data as f32).sqrt();
            // `max(1., expr)` is a gfortran `MAX` of two `real*4` values,
            // then truncated toward zero by the integer assignment.
            num_local_x = maxss(1.0_f32, 2.0_f32 * (xmax - xmin) / target_size + 0.1_f32) as i32;
            num_local_y = maxss(1.0_f32, 2.0_f32 * (ymax - ymin) / target_size + 0.1_f32) as i32;
            dx_local = 0.;
            dy_local = 0.;
            if num_local_x > 1 {
                dx_local = (xmax - xmin - target_size) / (num_local_x - 1) as f32;
            }
            if num_local_y > 1 {
                dy_local = (ymax - ymin - target_size) / (num_local_y - 1) as f32;
            }
            //
            // Loop on local areas, getting mean residual and number with max
            // above limit
            //
            sum_mean = 0.;
            sum_max = 0.;
            num_big = 0;
            for ixl in 1..=num_local_x {
                xcen = xmin + target_size / 2.0_f32 + (ixl - 1) as f32 * dx_local;
                for iyl in 1..=num_local_y {
                    ycen = ymin + target_size / 2.0_f32 + (iyl - 1) as f32 * dy_local;
                    size = target_size;
                    fill_local_data(
                        xcen,
                        ycen,
                        &mut size,
                        local_num,
                        &points_a,
                        &points_b,
                        num_a_points,
                        &map_a_to_b,
                        &jxyz,
                        &mut x_mat,
                        MAT_SIZE,
                        &mut num_data,
                    );
                    max_drop = nint(0.1_f32 * num_data as f32);
                    if num_data <= 6 {
                        max_drop = 0;
                    }
                    solve_wo_outliers(
                        &mut x_mat,
                        MAT_SIZE,
                        num_data,
                        num_col_fit,
                        icol_fixed,
                        max_drop,
                        crit_prob,
                        abs_prob_crit,
                        elim_min,
                        &mut idrop,
                        &mut num_drop,
                        &mut amat_local,
                        &mut dxyz_local,
                        &mut cen_mean_loc,
                        &mut dev_avg_loc,
                        &mut dev_sd,
                        &mut dev_max_loc,
                        &mut ipnt_max,
                        &mut dev_xyz_max,
                    );
                    sum_mean += dev_avg_loc;
                    sum_max += dev_max_loc;
                    // `solvematch.f90:928`: `maxss devAllMax, devMaxLoc`.
                    dev_all_max = maxss(dev_all_max, dev_max_loc);
                    if dev_max_loc > stop_limit {
                        num_big += 1;
                    }
                }
            }
            //
            // BRT is using 'Local fits' and 'Average mean' as tags
            let num_areas = num_local_x * num_local_y;
            println!(
                "\nLocal fits to a minimum of{} points give:\n   Average mean residual{}\n   Average max residual{}\n   Biggest max residual{}\n{} of{} local fits with max residual above{}",
                fmt_i(local_num, 4),
                fmt_f(sum_mean / num_areas as f32, 8, 2),
                fmt_f(sum_max / num_areas as f32, 8, 2),
                fmt_f(dev_all_max, 9, 2),
                fmt_i(num_big, 5),
                fmt_i(num_areas, 5),
                fmt_f(stop_limit, 8, 1)
            );

            if dev_all_max < 1.5_f32 * stop_limit
                && (num_big as f32) <= 0.05_f32 * num_local_x as f32 * num_local_y as f32
            {
                println!("\nThe local fits indicate that THIS SOLUTION IS GOOD ENOUGH\n");
                dev_max = stop_limit - 1.;
            }
            lim_raised = (dev_all_max + 1.2_f32) as i32;

            if shift_limit > 0. {
                //
                // Get the data for the central area and transform the points by
                // the global transformation and measure shift in the points.
                // 3/19/07: switched to this from doing a fit because the fit can
                // screw up the shifts if there are not points on both sides
                xcen = 0.;
                ycen = 0.;
                size = target_size;
                fill_local_data(
                    xcen,
                    ycen,
                    &mut size,
                    local_num,
                    &points_a,
                    &points_b,
                    num_a_points,
                    &map_a_to_b,
                    &jxyz,
                    &mut x_mat,
                    MAT_SIZE,
                    &mut num_data,
                );
                dxyz_local[0] = 0.;
                dxyz_local[1] = 0.;
                dxyz_local[2] = 0.;
                for ip in 1..=num_data {
                    for j in 1..=3 {
                        x_mat[xm(10 + j, ip)] = del_xyz[j as usize - 1];
                        for i in 1..=3 {
                            x_mat[xm(10 + j, ip)] =
                                x_mat[xm(10 + j, ip)] + a[ai(j, i)] * x_mat[xm(i, ip)];
                        }
                        dxyz_local[j as usize - 1] = dxyz_local[j as usize - 1]
                            + (x_mat[xm(10 + j, ip)] - x_mat[xm(num_col_fit + 1 + j, ip)])
                                / num_data as f32;
                    }
                }
                cen_shift_x = dxyz_local[0];
                cen_shift_y = dxyz_local[1];
                cen_shift_z = dxyz_local[2];
                dist = (cen_shift_x * cen_shift_x
                    + cen_shift_y * cen_shift_y
                    + cen_shift_z * cen_shift_z)
                    .sqrt();
                if dist >= shift_limit {
                    //
                    // BRT is looking for 'InitialShiftXYZ' and 'needs'
                    println!(
                        "\nCenter shift indicated by local fit is{}, bigger than the specified limit\n   The InitialShiftXYZ for corrsearch3d needs to be{}{}{}\n   In Etomo, set Patchcorr Initial shifts in X, Y, Z to{}{}{}",
                        fmt_f(dist, 6, 0),
                        fmt_i(nint(cen_shift_x), 5),
                        fmt_i(nint(cen_shift_y), 5),
                        fmt_i(nint(cen_shift_z), 5),
                        fmt_i(nint(cen_shift_x), 5),
                        fmt_i(nint(cen_shift_z), 5),
                        fmt_i(nint(cen_shift_y), 5)
                    );
                    if nxyz[1][jxyz[2] as usize - 1] > nxyz[0][jxyz[2] as usize - 1] {
                        println!(
                            "   You should also set thickness of initial matching file to at least{}\n     (In Etomo, Initial match size for Matchvol1)",
                            fmt_i(nxyz[jxyz[2] as usize - 1][1], 5)
                        );
                    }
                    //
                    // BRT is looking for 'CenterShiftLimit' and 'avoid stopping'
                    println!(
                        "   To avoid stopping with this error, set CenterShiftLimit to{}\n     (In Etomo, Limit on center shift for Solvematch)",
                        fmt_i(nint(dist) + 1, 4)
                    );
                    if dev_max < stop_limit {
                        //
                        // BRT is looking for INITIAL SHIFT' and 'SOLUTION IS OK'
                        println!(
                            "\nERROR: SOLVEMATCH - Initial shift needs to be set for patch correlation (but solution is OK)"
                        );
                    } else {
                        println!(
                            "\nERROR: SOLVEMATCH - Initial shift needs to be set for patch correlation"
                        );
                    }
                    ierr = 1;
                }
            }
        }
    }

    if dev_max > stop_limit {
        //
        // Give some guidance based upon the ratios between max and mean
        // deviation and max deviation and stopping limit
        //
        println!(
            "\nThe maximum residual is{}, too high to proceed",
            fmt_f(dev_max, 8, 2)
        );
        lo_max_avg_ratio = 4.;
        hi_max_avg_ratio = 12.;
        lo_max_lim_ratio = 2.;
        hi_max_lim_ratio = 3.;
        glob_loc_avg_ratio = 3.;
        if (dev_max < lo_max_avg_ratio * dev_avg && dev_max < lo_max_lim_ratio * stop_limit)
            || (dev_all_max > 0.
                && dev_avg_loc * glob_loc_avg_ratio < dev_avg
                && dev_all_max < lo_max_lim_ratio * stop_limit)
        {
            if num_trans_coord > 0 {
                println!(
                    "Since corresponding points were picked using coordinates from transferfid,\nthis is almost certainly due to distortion between the volumes,\n and you should just raise the residual limit to{}",
                    fmt_i(lim_raised, 4)
                );
            } else if dev_all_max <= 0. {
                println!(
                    "Since the maximum residual is less than{} times the mean residual ({})\n and less than{} times the specified residual limit,\n this is probably due to distortion between the volumes,\nand you should probably just raise the residual limit to{}",
                    fmt_f(lo_max_avg_ratio, 6, 1),
                    fmt_f(dev_avg, 8, 2),
                    fmt_f(lo_max_lim_ratio, 6, 1),
                    fmt_i(lim_raised, 4)
                );
            } else {
                println!(
                    "Since the local fits improved the mean residual by more than a factor of{}\n and the local maximum residual ({}) is less than{} times the\n specified residual limit, this is probably due to distortion between the\n volumes, and you should probably just raise the residual limit to{}",
                    fmt_f(glob_loc_avg_ratio, 6, 1),
                    fmt_f(dev_all_max, 8, 2),
                    fmt_f(lo_max_lim_ratio, 6, 1),
                    fmt_i(lim_raised, 4)
                );
            }
        } else if num_trans_coord > 0 {
            println!(
                "Since corresponding points were picked using coordinates from transferfid,\nthis is probably due to distortion between the volumes.\nYou could raise the residual limit to{} or start with a subset of points.\nBad correspondence is unlikely but you could check points (especially{}).",
                fmt_i(lim_raised, 4),
                fmt_i(ind_orig_at_max, 4)
            );
        } else if dev_max > hi_max_avg_ratio * dev_avg || dev_max > hi_max_lim_ratio * stop_limit {
            if dev_max > hi_max_avg_ratio * dev_avg {
                println!(
                    "The maximum residual is more than{} times the mean residual ({})",
                    fmt_f(hi_max_avg_ratio, 6, 1),
                    fmt_f(dev_avg, 8, 2)
                );
            }
            if dev_max > hi_max_lim_ratio * stop_limit {
                println!(
                    "The maximum residual is more than{} times the specified residual limit",
                    fmt_f(hi_max_lim_ratio, 6, 1)
                );
            }
            println!(" This is probably due to a bad correspondence list.");
            println!(
                "Check the points (especially{}) or start with a subset of the list",
                fmt_i(ind_orig_at_max, 4)
            );
        } else {
            println!(" The situation is ambiguous but could be due to a bad correspondence list.");
            println!(
                "Check the points (especially{}) or start with a subset of the list",
                fmt_i(ind_orig_at_max, 4)
            );
        }
        exit_error("Maximum residual is too high to proceed");
    }
    exit(ierr);
}

/// Original `getDelta` (`solvematch.f90:1092`).
///
/// GETDELTA returns the "delta" or pixel size from a tomogram as well as the
/// nxyz.  OPTION is the option that specifies the tomogram or nx, ny, nz.
/// DELTA is returned with -1 if the option was not entered at all, 0 if
/// sizes were entered, or the actual delta value from the file.
pub fn get_delta(option: &str, delta: &mut f32, nxyz: &mut [i32; 3]) {
    let mut mxyz = [0_i32; 3];
    let mut mode = 0_i32;
    // `character*80 line` (`solvematch.f90:1096`): a longer `-atomogram`/
    // `-btomogram` value makes `PipGetString` return -1, which `> 0` takes
    // as entered, and native opens the name truncated to 80 characters.
    // Fixed in translation (BUGS.md): the buffer holds the whole entry
    // (4096 bytes, the path limit), so the name is used as entered.
    let mut line = String::new();
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let del_tmp: [f32; 3];
    //
    *delta = -1.;
    if get_string(option.as_bytes(), &mut line, 4096) > 0 {
        return;
    }
    *delta = 0.;
    let mut padded = line.clone().into_bytes();
    padded.resize(line.len().max(80), b' ');
    if !line_is_filename(&padded) {
        let mut cursor = std::io::Cursor::new(padded.clone());
        let (n0, rest) = nxyz.split_at_mut(1);
        let (n1, n2) = rest.split_at_mut(1);
        if list_read(
            &mut cursor,
            &mut [
                ListItem::Integer(&mut n0[0]),
                ListItem::Integer(&mut n1[0]),
                ListItem::Integer(&mut n2[0]),
            ],
        )
        .is_ok()
        {
            return;
        }
    }
    //
    // 10
    ialprt(false);
    imopen(5, line.trim_end_matches(' '), "ro");
    // SAFETY: `irdhdr` writes three integers into each of `nxyz` and `mxyz`
    // and one value through each scalar pointer, all live locals.
    unsafe {
        irdhdr(
            5,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &raw mut mode,
            &raw mut dmin,
            &raw mut dmax,
            &raw mut dmean,
        );
    }
    del_tmp = iiu_ret_delta(5);
    // SAFETY: unit 5 was opened above and nothing else holds it.
    unsafe {
        iiu_close(5);
    }
    ialprt(true);
    *delta = del_tmp[0];
}

/// Original `getFiducials` (`solvematch.f90:1126`).
///
/// getFiducials reads the fiducials from FILENAME, storing the sequential
/// point number in ICONTA, the X, Y, Z coordinates in PNTA, and the number
/// of points in NPNTA.  IDIM specifies the dimensions of the arrays.  AXIS
/// specifies the axis for a message.  If a pixel size is found on the first
/// line, it is returned in PIXELSIZE, otherwise this is set to 0.
#[allow(clippy::too_many_arguments)]
pub fn get_fiducials(
    filename: &str,
    icont_a: &mut [i32],
    points_a: &mut [[f32; 3]],
    num_a_points: &mut i32,
    mod_obj: &mut [i32],
    mod_cont: &mut [i32],
    idim: i32,
    axis: &str,
    pixel_size: &mut f32,
    nx_fid_dim: &mut i32,
    ny_fid: &mut i32,
) {
    // `character*100 line`
    const LINE_LEN: usize = 100;
    let mut unm_fields = 0_i32;
    let mut numeric = [0_i32; 10];
    let mut xnum_in = [0.0_f32; 10];
    // `read(1, '(a)', ...) line`: the record truncated or blank-padded to
    // the variable's length.  `None` is the `END=` branch; an I/O error is
    // the `ERR=` branch.
    let read_line = |unit: &mut BufReader<std::fs::File>| -> Option<Vec<u8>> {
        let mut record: Vec<u8> = Vec::new();
        match unit.read_until(b'\n', &mut record) {
            Ok(0) => None,
            Ok(_) => {
                if record.last() == Some(&b'\n') {
                    record.pop();
                }
                record.truncate(LINE_LEN);
                record.resize(LINE_LEN, b' ');
                Some(record)
            }
            Err(_) => exit_error("Reading fiducial point file"),
        }
    };
    //
    *num_a_points = 0;
    *pixel_size = 0.;
    *nx_fid_dim = 0;
    *ny_fid = 0;
    let mut unit1 = BufReader::new(dopen(1, filename.trim_end_matches(' '), "old", "f"));
    //
    // get the first line and search for PixelSize: (first version)
    // or Pix: and Dim: (second version)
    //
    if let Some(mut line) = read_line(&mut unit1) {
        let len = line
            .iter()
            .rposition(|&c| c != b' ')
            .map_or(0, |p| p as i32 + 1);
        // `read(line(first:len), *, err = 8)` with no `END=`.
        let internal_read = |text: &[u8], items: &mut [ListItem]| {
            let mut cursor = std::io::Cursor::new(text.to_vec());
            match list_read(&mut cursor, items) {
                Ok(()) => {}
                Err(ListReadError::Error) => {
                    exit_error("Reading pixel size or dimensions from point file")
                }
                Err(err) => read_abort(err),
            }
        };
        let mut i = 1_i32;
        while i < len - 11 {
            let at = |k: i32, n: usize| &line[(k - 1) as usize..(k - 1) as usize + n];
            if at(i, 11) == b"Pixel size:" {
                internal_read(
                    &line[(i + 11 - 1) as usize..len as usize],
                    &mut [ListItem::Real(pixel_size)],
                );
            }
            if at(i, 4) == b"Pix:" {
                internal_read(
                    &line[(i + 4 - 1) as usize..len as usize],
                    &mut [ListItem::Real(pixel_size)],
                );
            }
            if at(i, 4) == b"Dim:" {
                internal_read(
                    &line[(i + 4 - 1) as usize..len as usize],
                    &mut [ListItem::Integer(nx_fid_dim), ListItem::Integer(ny_fid)],
                );
            }
            i += 1;
        }
        //
        // process first line then loop until end
        //
        loop {
            *num_a_points += 1;
            if *num_a_points > idim {
                exit_error("Too many points for arrays");
            }
            let n = *num_a_points as usize - 1;
            let card = String::from_utf8_lossy(&line).into_owned();
            frefor2(&card, &mut xnum_in, &mut numeric, &mut unm_fields, 6);
            let mut cursor = std::io::Cursor::new(line.clone());
            let [p0, p1, p2] = &mut points_a[n];
            let result = if unm_fields == 6 && numeric[4] == 1 {
                list_read(
                    &mut cursor,
                    &mut [
                        ListItem::Integer(&mut icont_a[n]),
                        ListItem::Real(p0),
                        ListItem::Real(p1),
                        ListItem::Real(p2),
                        ListItem::Integer(&mut mod_obj[n]),
                        ListItem::Integer(&mut mod_cont[n]),
                    ],
                )
            } else {
                let result = list_read(
                    &mut cursor,
                    &mut [
                        ListItem::Integer(&mut icont_a[n]),
                        ListItem::Real(p0),
                        ListItem::Real(p1),
                        ListItem::Real(p2),
                    ],
                );
                if result.is_ok() {
                    mod_obj[n] = 0;
                    mod_cont[n] = 0;
                }
                result
            };
            if let Err(err) = result {
                read_abort(err);
            }
            //
            // invert the Z coordinates because the tomogram built by TILT is
            // inverted relative to the solution produced by TILTALIGN
            // 12/1/09: THIS IS NOT TRUE AFTER ACCOUNTING FOR FLIPPING IN 3dMOD
            points_a[n][2] = -points_a[n][2];
            match read_line(&mut unit1) {
                Some(next) => line = next,
                None => break,
            }
        }
    }
    //
    // 15
    println!(" {:>11}  points from {}", *num_a_points, axis);
    // `close(1)` is the drop of the reader.
}

/// Original `readFidModelFile` (`solvematch.f90:1195`).
///
/// readFidModel gets the filename with the given OPTION, reads fiducial
/// model, finds points with Z = IZBEST, and returns their X/Y coords in
/// FIDMODX, FIDMODY and their object and contour numbers in MODOBJFID and
/// MODCONTFID.  NUMFID is the number of fiducials returned; IDIM specifies
/// the dimensions of the arrays; ABTEXT is A or B.  transPixel is a pixel
/// size for transfer coords if non-zero.
#[allow(clippy::too_many_arguments)]
pub fn read_fid_model_file(
    option: &str,
    iz_best: i32,
    ab_text: &str,
    fid_mod_x: &mut [f32],
    fid_mod_y: &mut [f32],
    mod_obj_fid: &mut [i32],
    mod_cont_fid: &mut [i32],
    num_fid: &mut i32,
    idim: i32,
    trans_pixel: f32,
    fm: &mut FortModel,
) {
    let mut fid_scale: f32;
    let (mut x_im_scale, mut y_im_scale, mut z_im_scale) = (0.0_f32, 0.0_f32, 0.0_f32);
    // `character*160 filename`
    let mut filename = String::new();
    let mut ip: i32;
    let mut ipt: i32;
    let mut looking: bool;

    if get_string(option.as_bytes(), &mut filename, 160) != 0 {
        exit_error("Fiducial models for both axes must be entered to use transfer coords");
    }
    if !readw_or_imod(filename.trim_end_matches(' '), fm) {
        exit_error(&format!("Reading fiducial model for axis {ab_text}"));
    }
    //
    // If a pixel size was defined for transfer, scale coords to match that
    fid_scale = 1.;
    ip = getimodscales(&mut x_im_scale, &mut y_im_scale, &mut z_im_scale);
    if ip == 0 && trans_pixel > 0. {
        fid_scale = x_im_scale / trans_pixel;
    }
    scale_model(0, fm);
    *num_fid = 0;
    for iobj in 1..=fm.max_mod_obj {
        looking = true;
        ip = 1;
        //
        // Find point at best z value, record it and its obj/cont
        //
        while looking && ip <= fm.npt_in_obj[iobj as usize - 1] {
            ipt = fm.object[(fm.ibase_obj[iobj as usize - 1] + ip) as usize - 1].abs();
            if nint(fm.p_coord[ipt as usize - 1][2]) == iz_best {
                *num_fid += 1;
                if *num_fid > idim {
                    exit_error(&format!(
                        "Too many fiducial contours for arrays in model for axis {ab_text}"
                    ));
                }
                let k = *num_fid as usize - 1;
                fid_mod_x[k] = fm.p_coord[ipt as usize - 1][0] * fid_scale;
                fid_mod_y[k] = fm.p_coord[ipt as usize - 1][1] * fid_scale;
                objtocont(
                    iobj,
                    &fm.obj_color,
                    &mut mod_obj_fid[k],
                    &mut mod_cont_fid[k],
                );
                looking = false;
            }
            ip += 1;
        }
    }
}

/// Original `rotateFids` (`solvematch.f90:1251`).
///
/// ROTATEFIDS rotates, scales and shifts the NPNTA fiducials in PNTA.  XTILT
/// is the x axis tilt, aScale is the scaling, angleOffset the angle offset
/// and zShift the Z shift.
pub fn rotate_fids(
    points: &mut [[f32; 3]],
    num_points: i32,
    xtilt: f32,
    scale: f32,
    angle_offset: f32,
    z_shift: f32,
) {
    let cosa: f32;
    let cosb: f32;
    let sin_alpha: f32;
    let sin_beta: f32;
    let mut ytmp: f32;
    //
    // use the negative of the angle to account for the inversion of the
    // tomogram; rotate the 3-d points about the Y axis first, then the Z
    // axis; add Z shift
    // 12/1/09: THE NEGATIVE COMPENSATES FOR THE FLIPPING OF Z ABOVE, BUT
    // ARE THERE OTHER CONSEQUENCES?  IS ALL THIS RIGHT?
    cosa = gfortran_cosd_r4(-xtilt) * scale;
    sin_alpha = gfortran_sind_r4(-xtilt) * scale;
    sin_beta = gfortran_sind_r4(-angle_offset);
    cosb = gfortran_cosd_r4(-angle_offset);
    for i in 1..=num_points {
        let p = &mut points[i as usize - 1];
        ytmp = cosb * p[0] - sin_beta * p[2];
        p[2] = sin_beta * p[0] + cosb * p[2];
        p[0] = ytmp;
        ytmp = cosa * p[1] - sin_alpha * p[2];
        p[2] = sin_alpha * p[1] + cosa * p[2] + z_shift;
        p[1] = ytmp;
        p[0] *= scale;
    }
}

/// Original `fillLocalData` (`solvematch.f90:1287`).
///
/// fillLocalData fills the data array XMAT with at least MINDAT points from
/// an area centered at XCEN, YCEN.  SIZE is called with an initial trial
/// size, and returned with the final size needed to include the required
/// points.  PNTA and PNTB have the points, NPNTA is the total number in
/// PNTA, MAPAB is the mapping to indices in PNTB, JXYZ is the dimension
/// mapping.  NDAT is returned with number of points.
#[allow(clippy::too_many_arguments)]
pub fn fill_local_data(
    xcen: f32,
    ycen: f32,
    size: &mut f32,
    min_data: i32,
    points_a: &[[f32; 3]],
    points_b: &[[f32; 3]],
    num_a_points: i32,
    map_a_to_b: &[i32],
    jxyz: &[i32; 3],
    x_mat: &mut [f32],
    mat_size: i32,
    num_data: &mut i32,
) {
    let ms = mat_size as usize;
    let xm = |j: i32, i: i32| (j - 1) as usize + (i - 1) as usize * ms;
    let (mut xmin, mut ymin, mut xmax, mut ymax): (f32, f32, f32, f32);
    let size_in: f32;
    //
    size_in = *size;
    *num_data = 0;
    while *num_data < min_data {
        *num_data = 0;
        xmin = xcen - *size / 2.0_f32;
        xmax = xcen + *size / 2.0_f32;
        ymin = ycen - *size / 2.0_f32;
        ymax = ycen + *size / 2.0_f32;
        for ia in 1..=num_a_points {
            let pa = &points_a[ia as usize - 1];
            let map = map_a_to_b[ia as usize - 1];
            if map > 0 && pa[0] >= xmin && pa[0] <= xmax && pa[1] >= ymin && pa[1] <= ymax {
                *num_data += 1;
                for j in 1..=3 {
                    x_mat[xm(j, *num_data)] =
                        points_b[map as usize - 1][jxyz[j as usize - 1] as usize - 1];
                    x_mat[xm(j + 4, *num_data)] = pa[jxyz[j as usize - 1] as usize - 1];
                }
            }
        }
        if *num_data < min_data {
            *size += 0.02_f32 * size_in;
        }
    }
}

/// Original `determ3` (`solvematch.f90:1319`).
///
/// Determinant of a 3x3 matrix (the leading `3x3` of the argument, which is
/// `a(3,4)` at the call sites).
pub fn determ3(a: &[f32]) -> f32 {
    let at = |i: usize, j: usize| a[(i - 1) + 3 * (j - 1)];
    at(1, 1) * at(2, 2) * at(3, 3) - at(1, 1) * at(2, 3) * at(3, 2) + at(1, 2) * at(2, 3) * at(3, 1)
        - at(1, 2) * at(2, 1) * at(3, 3)
        + at(1, 3) * at(2, 1) * at(3, 2)
        - at(1, 3) * at(2, 2) * at(3, 1)
}
