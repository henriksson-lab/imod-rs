//! Translation of `IMOD/flib/model/get_region_contours.f90`.
//!
//! This file has some routines shared by corrsearch3d, findwarp, refinematch
//! and filltomo.  The `use fortmodel` module arrays are the [`FortModel`]
//! passed in, as in `readw_or_imod.rs`.

use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::inside::inside;
use crate::imod::flib::subrs::hvem::parse_input_params::exit_error;
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::{scale_model, scale_model_to_image};
use crate::imod::libcfshr::parse_params::{
    pip_get_float_array, pip_get_two_integers, pip_number_of_entries,
};
use crate::imod::libcfshr::simplestat::avg_sd;
use crate::imod::libimod::imodel_fwrap::getimodhead;

/// Original `get_region_contours` (`get_region_contours.f90:22`).
///
/// GET_REGION_CONTOURS will read the MODELFILE as a small model, and
/// extract contours.  It first determines whether the contours lie in
/// X/Y or X/Z planes, setting IFFLIP to 1 if they are in Y/Z planes.
/// It extracts the planar points into XVERT, YVERT, where INDVERT is an
/// index to the start of each contour, NVERT contains the number of
/// points in each contour, ZCONT is the contour Y or Z value, NCONT is
/// the number of contours.  LIMCONT and LIMVERT specify the limiting
/// dimensions for contours and vertices.  Pass LIMCONT as <= 0 on an initial
/// call to have them returned with the needed sizes.  If IMUNIT is greater
/// than 0 the model is scaled to the index coordinates of the image file open
/// on that unit.
///
/// The 1-based Fortran arrays are 0-based slices: `xVerts(k)` is
/// `x_verts[k - 1]`; `indVert` values stay 1-based, as the source stores them.
pub fn get_region_contours(
    fm: &mut FortModel,
    model_file: &str,
    progname: &str,
    x_verts: &mut [f32],
    y_verts: &mut [f32],
    num_verts: &mut [i32],
    ind_vert: &mut [i32],
    contour_z: &mut [f32],
    num_conts: &mut i32,
    if_flip: &mut i32,
    lim_cont: &mut i32,
    lim_vert: &mut i32,
    im_unit: i32,
) {
    let (mut xy_scale, mut z_scale, mut x_offset, mut y_offset, mut z_offset) =
        (0f32, 0f32, 0f32, 0f32, 0f32);
    // `object(k)`, `p_coord(k, i)`, `npt_in_obj(i)`, `ibase_obj(i)`.
    let object = |fm: &FortModel, k: i32| fm.object[(k - 1) as usize];
    let pc = |fm: &FortModel, k: i32, i: i32| fm.p_coord[(i - 1) as usize][(k - 1) as usize];
    let npt_in_obj = |fm: &FortModel, i: i32| fm.npt_in_obj[(i - 1) as usize];
    let ibase_obj = |fm: &FortModel, i: i32| fm.ibase_obj[(i - 1) as usize];
    // gfortran `nint` of a real*4: round half away from zero.
    let nint = |v: f32| v.round() as i32;
    //
    fm.fm_mod_size_type = 2;
    let exist = readw_or_imod(model_file.trim_end_matches(' '), fm);
    if !exist {
        exit_error("Error reading model");
    }
    if im_unit > 0 {
        scale_model_to_image(im_unit, 0, fm);
    } else {
        scale_model(0, fm);
    }
    *num_conts = 0;
    //
    // figure out if all contours are coplanar Z or Y
    // Loop through contours until both planar flags go false or all done
    //
    let mut iobj = 1;
    let mut planar_in_z = true;
    let mut planar_in_y = true;
    while iobj <= fm.max_mod_obj && (planar_in_z || planar_in_y) {
        if npt_in_obj(fm, iobj) >= 3 {
            let mut ip = 2;
            let mut ipt = object(fm, ibase_obj(fm, iobj) + 1).abs();
            let iy = nint(pc(fm, 2, ipt));
            let iz = nint(pc(fm, 3, ipt));
            while ip <= npt_in_obj(fm, iobj) && (planar_in_z || planar_in_y) {
                ipt = object(fm, ibase_obj(fm, iobj) + ip).abs();
                if nint(pc(fm, 2, ipt)) != iy {
                    planar_in_y = false;
                }
                if nint(pc(fm, 3, ipt)) != iz {
                    planar_in_z = false;
                }
                ip += 1;
            }
        }
        iobj += 1;
    }
    //
    // Set flip flag or fallback to what is in model header
    //
    if planar_in_z && !planar_in_y {
        *if_flip = 0;
    } else if planar_in_y && !planar_in_z {
        *if_flip = 1;
    } else {
        let _ierr = getimodhead(
            &mut xy_scale,
            &mut z_scale,
            &mut x_offset,
            &mut y_offset,
            &mut z_offset,
            if_flip,
        );
        if *lim_cont > 0 {
            // write(*,'(/,a,a,a)')
            println!(
                "\nWARNING: {} - CONTOURS NOT ALL COPLANAR, USING HEADER FLIP FLAG",
                progname
            );
        }
    }
    let mut ind_y = 2;
    let mut ind_z = 3;
    if *if_flip != 0 {
        ind_y = 3;
        ind_z = 2;
    }
    //
    // If limCont <= 0, add up number of contours and points and return
    if *lim_cont <= 0 {
        *lim_cont = 0;
        *lim_vert = 0;
        for iobj in 1..=fm.max_mod_obj {
            if npt_in_obj(fm, iobj) >= 3 {
                *lim_cont += 1;
                *lim_vert += npt_in_obj(fm, iobj);
            }
        }
        return;
    }
    //
    // Load planar data into x/y arrays
    //
    let mut ind_cur = 0;
    for iobj in 1..=fm.max_mod_obj {
        if npt_in_obj(fm, iobj) >= 3 {
            *num_conts += 1;
            if *num_conts > *lim_cont {
                exit_error("Too many contours in model");
            }
            let nc = (*num_conts - 1) as usize;
            num_verts[nc] = npt_in_obj(fm, iobj);
            if ind_cur + num_verts[nc] > *lim_vert {
                exit_error("Too many points in contours");
            }
            let mut ipt = 0;
            for ip in 1..=num_verts[nc] {
                ipt = object(fm, ibase_obj(fm, iobj) + ip).abs();
                x_verts[(ip + ind_cur - 1) as usize] = pc(fm, 1, ipt);
                y_verts[(ip + ind_cur - 1) as usize] = pc(fm, ind_y, ipt);
            }
            contour_z[nc] = pc(fm, ind_z, ipt);
            ind_vert[nc] = ind_cur + 1;
            ind_cur += num_verts[nc];
        }
    }
    if progname.trim_end_matches(' ') != "FILLTOMO" {
        // print *, numConts, ' contours available ...' (list-directed I12)
        println!(
            " {:>11}  contours available for deciding which patches to analyze",
            *num_conts
        );
    }
}

/// Original `getContourArraySizes` (`get_region_contours.f90:139`).
///
/// getContourArraySizes is a convenience function for getting the array
/// sizes.
pub fn get_contour_array_sizes(
    fm: &mut FortModel,
    model_file: &str,
    im_unit: i32,
    lim_cont: &mut i32,
    lim_vert: &mut i32,
) {
    let (mut num_conts, mut if_flip) = (0, 0);
    let mut num_verts = [0_i32; 1];
    let mut ind_vert = [0_i32; 1];
    let mut dummy = [0f32; 1];
    let mut dummy2 = [0f32; 1];
    let mut dummy3 = [0f32; 1];
    *lim_cont = -1;
    *lim_vert = -1;
    get_region_contours(
        fm,
        model_file,
        "DUMMY",
        &mut dummy,
        &mut dummy2,
        &mut num_verts,
        &mut ind_vert,
        &mut dummy3,
        &mut num_conts,
        &mut if_flip,
        lim_cont,
        lim_vert,
        im_unit,
    );
    *lim_vert += 10;
    *lim_cont += 10;
}

/// Original `checkBoundaryConts` (`get_region_contours.f90:156`).
///
/// checkBoundaryConts checks a patch center against the boundary contours by
/// finding the contour at the nearest Z level and testing whether the center
/// is inside the contour.  `indVertStart` values are 1-based.
pub fn check_boundary_conts(
    cen_x: f32,
    cen_y: f32,
    cen_z: f32,
    if_use: &mut i32,
    num_conts: i32,
    num_verts: &[i32],
    x_verts: &[f32],
    y_verts: &[f32],
    contour_z: &[f32],
    ind_vert_start: &[i32],
) -> i32 {
    // `icontMin` is uninitialised in the source when no contour is closer
    // than 100000 in Z (`get_region_contours.f90:154-162`; also a NaN
    // `cenZ`), and native then indexes with stack residue.  Fixed in
    // translation (BUGS.md): with no nearest contour the patch is outside
    // every contour, `ifUse` stays 0 and 0 is returned.  The contour number
    // is returned so that `corrsearch3d`'s `-debug 2` message can print the
    // contour that eliminated the patch (the source prints an unset host
    // variable there, `corrsearch3d.f90:576`).
    let mut icont_min = 0;
    *if_use = 0;
    //
    // find nearest contour in Z and see if patch is inside it
    //
    let mut dz_min = 100000.0_f32;
    for icont in 1..=num_conts {
        let dz = (cen_z - contour_z[(icont - 1) as usize]).abs();
        if dz < dz_min {
            dz_min = dz;
            icont_min = icont;
        }
    }
    if icont_min == 0 {
        return 0;
    }
    let indv = ind_vert_start[(icont_min - 1) as usize];
    if inside(
        &x_verts[(indv - 1) as usize..],
        &y_verts[(indv - 1) as usize..],
        num_verts[(icont_min - 1) as usize],
        cen_x,
        cen_y,
    ) {
        *if_use = 1;
    }
    icont_min
}

/// Original `summarizeDrops` (`get_region_contours.f90:189`).
///
/// summarizeDrops outputs a summary of residuals dropped as outliers,
/// breaking them into 10 bins from the minimum to either the maximum or
/// 5 SDs above the mean, whichever is less.
pub fn summarize_drops(drop_sum: &[f32], num_list_drop: i32, mean_text: &str) {
    let (mut drop_avg, mut drop_sd, mut drop_sem) = (0f32, 0f32, 0f32);
    //
    // gfortran MIN/MAX of real*4 (MIN_EXPR/MAX_EXPR): the NaN operand order
    // was not checked against the reference object code.
    let mut drop_min = 1.0e10_f32;
    let mut drop_max = 0.0_f32;
    for i in 0..num_list_drop as usize {
        drop_min = if drop_sum[i] < drop_min {
            drop_sum[i]
        } else {
            drop_min
        };
        drop_max = if drop_sum[i] > drop_max {
            drop_sum[i]
        } else {
            drop_max
        };
    }
    avg_sd(
        drop_sum,
        num_list_drop,
        &mut drop_avg,
        &mut drop_sd,
        &mut drop_sem,
    );
    let upper = drop_avg + 5.0 * drop_sd;
    let drop_bin = ((if upper < drop_max { upper } else { drop_max }) - drop_min) / 10.0;
    //
    for j in 1..=10 {
        let bin_low = drop_min + (j - 1) as f32 * drop_bin;
        let mut bin_high = bin_low + drop_bin;
        if j == 10 {
            bin_high = drop_max + 0.001;
        }
        let mut in_bin = 0;
        for i in 0..num_list_drop as usize {
            if drop_sum[i] >= bin_low && drop_sum[i] < bin_high {
                in_bin += 1;
            }
        }
        if in_bin > 0 {
            // format(i8,' with ',a,'residuals in',f10.2,' -',f10.2)
            let i8 = format!("{in_bin:>8}");
            println!(
                "{} with {}residuals in{} -{}",
                if i8.len() > 8 { "*".repeat(8) } else { i8 },
                mean_text,
                format_f(f64::from(bin_low), 10, 2),
                format_f(f64::from(bin_high), 10, 2)
            );
        }
    }
}

/// Original `getExtraSelections` (`get_region_contours.f90:224`).
///
/// getExtraSelections gets specifications for selecting patches based on the
/// values in extra columns and checks them for validity.  `selectCrit` is
/// the column-major `selectCrit(limCrit, limSelect)`.
pub fn get_extra_selections(
    icol_select: &mut [i32],
    isign_select: &mut [i32],
    num_col_select: &mut i32,
    num_select_crit: &mut i32,
    i_dextra: &[i32],
    max_extra: i32,
    select_crit: &mut [f32],
    lim_select: i32,
    lim_crit: i32,
) {
    let mut ix = 0;
    let _ = pip_number_of_entries(b"ExtraValueSelection", num_col_select);
    let _ = pip_number_of_entries(b"SelectionCriteria", &mut ix);
    if ix != *num_col_select {
        exit_error("There must be one -select entry for each -extra entry");
    }
    if *num_col_select > 0 && max_extra == 0 {
        exit_error("There are no extra value columns in the patch file");
    }
    if *num_col_select > lim_select {
        exit_error("Too many selections for arrays");
    }
    for i in 1..=*num_col_select {
        let iu = (i - 1) as usize;
        let _ = pip_get_two_integers(
            b"ExtraValueSelection",
            &mut icol_select[iu],
            &mut isign_select[iu],
        );
        ix = 0;
        let start = iu * lim_crit as usize;
        let _ = pip_get_float_array(
            b"SelectionCriteria",
            &mut select_crit[start..start + lim_crit as usize],
            &mut ix,
            lim_crit,
        );
        if i > 1 && ix != *num_select_crit {
            exit_error("Every entry of -select must have the same number of criteria");
        }
        *num_select_crit = ix;
        if isign_select[iu].abs() != 1 {
            exit_error("The second value on the -select entry must be 1 or -1");
        }
        if icol_select[iu] < 0 {
            // `get_region_contours.f90:249` tests `icolSelect(i) > maxExtra`
            // inside `icolSelect(i) < 0`, which is never true, so native
            // accepts a column past the extra columns and indexes beyond
            // them.  Fixed in translation (BUGS.md): the column number
            // (the negated entry) is checked against `maxExtra`.
            if -icol_select[iu] > max_extra {
                exit_error(
                    "There is no extra value column corresponding to the column number entered with -extra",
                );
            }
            icol_select[iu] = -icol_select[iu];
        } else {
            if icol_select[iu] == 0 {
                exit_error("0 is not a value column number or id #");
            }
            let mut ierr = 1;
            for ix in 1..=max_extra {
                if icol_select[iu] == i_dextra[(ix - 1) as usize] {
                    ierr = 0;
                    icol_select[iu] = ix;
                    break;
                }
            }
            if ierr > 0 {
                exit_error(
                    "There is no extra value column with the column id # entered with -extra",
                );
            }
        }
    }
}
