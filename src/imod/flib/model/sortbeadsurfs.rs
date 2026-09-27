//! Translation of `IMOD/flib/model/sortbeadsurfs.f90`.
//!
//! SORTBEADSURFS sorts beads into two surfaces for a bead model from
//! findbeads3d or beadtrack, or applies X axis tilt, shifts coordinates and
//! combines objects for a model from tiltalign.  The main program maps to
//! [`sortbeadsurfs`] and the external subroutine to [`subarea_limits`].  The
//! `fortmodel` module arrays are the [`FortModel`] that `readw_or_imod`
//! fills; `xyz(3, max_pt)` and `xyzFit(3, max_pt)` are column major, so
//! `xyz(k, i)` is `xyz[3 * (i - 1) + k - 1]`.
//!
//! `sind`/`cosd` are the libgfortran `_gfortran_sind_r4`/`_gfortran_cosd_r4`
//! the reference binary imports.

use std::io::Write;

use crate::imod::flib::subrs::compat::gfortran_rt::{
    gfortran_cosd_r4, gfortran_sind_r4, maxss, minss,
};
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::objtocont::objtocont;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::scale_model;
use crate::imod::flib::subrs::model::write_wmod::write_wmod;
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::parse_params::{
    pip_get_float, pip_get_integer, pip_get_string, pip_get_two_floats, pip_get_two_integers,
    pip_number_of_entries,
};
use crate::imod::libcfshr::robuststat::rs_mad_median_outliers;
use crate::imod::libcfshr::surfacesort::{set_surf_sort_param, surface_sort};
use crate::imod::libimod::imodel_fwrap::{
    deleteiobj, getcontvalue, getimodmaxes, getobjcolor, getpointvalue, getscatsize, putimodflag,
    putimodmaxes, putobjcolor, putscatsize,
};

/// `parameter (numOptions = 22)` (`sortbeadsurfs.f90:39`).
const NUM_OPTIONS: i32 = 22;
/// Fallback PIP table `options(1)` (`sortbeadsurfs.f90:41-50`).
const OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@text:TextFileWithSurfaces:FN:@\
flipyz:FlipYandZ:I:@subarea:SubareaSize:I:@pick:PickAreasMinNumAndSize:IP:@\
majority:MajorityObjectOnly:B:@xaxis:XAxisTilt:F:@invert:InvertZAxis:B:@\
already:AlreadySorted:B:@one:OneSurface:B:@aligned:AlignedSizeXandY:IP:@\
xtrim:XTrimStartAndEnd:IP:@ytrim:YTrimStartAndEnd:IP:@\
prebin:PrealignedBinning:I:@recbin:ReconstructionBinning:I:@\
rescale:RescaleByBinnings:B:@check:CheckExistingGroups:B:@\
values:ValuesToRestrainSorting:I:@outlier:OutlierCriterionDeviation:F:@\
set:SetSurfaceSortParam:FPM:@help:usage:B:";

/// gfortran `Iw` editing of an integer.
fn fmt_i(value: i32, w: usize) -> String {
    let text = value.to_string();
    if text.len() > w {
        "*".repeat(w)
    } else {
        format!("{text:>w$}")
    }
}

/// Original program: `sortbeadsurfs` (`sortbeadsurfs.f90:10`).
pub fn sortbeadsurfs() {
    let mut in_file = String::new();
    let mut out_file = String::new();
    let mut text_output: String;
    let mut num_in_group = [0i32; 2];
    let mut j_red = [0i32; 2];
    let mut j_green = [0i32; 2];
    let mut j_blue = [0i32; 2];
    let mut already: bool;
    let mut majority: bool;
    let mut invert_z: bool;
    let mut rescale: bool;
    let mut one_surface: bool;
    let mut check_groups: bool;
    let mut if_flip: i32;
    let mut local: i32;
    let mut ierr: i32;
    let mut i: i32;
    let mut j: i32;
    let mut ix: i32;
    let mut iy: i32;
    let mut ind_y: usize;
    let mut ind_z: usize;
    let mut maxx = 0;
    let mut maxy = 0;
    let mut maxz = 0;
    let mut xmax: f32;
    let mut xmin: f32;
    let mut ymin: f32;
    let mut ymax: f32;
    let mut xlo: f32 = 0.;
    let mut xhi: f32 = 0.;
    let mut ylo: f32 = 0.;
    let mut yhi: f32 = 0.;
    let mut eps_x: f32 = 0.;
    let mut eps_y: f32 = 0.;
    let mut dx: f32;
    let mut dy: f32;
    let cosa: f32;
    let sina: f32;
    let mut xtilt: f32;
    let mut scale_fac: f32;
    let mut outlier_crit: f32;
    let mut cont_value: f32 = 0.;
    let mut num_area_x: i32;
    let mut num_area_y: i32;
    let mut num_points: i32;
    let mut isize: i32;
    // Fixed in translation (BUGS.md, sortbeadsurfs): `minNum` is set only
    // when sorting, but read at the end whenever `local > 0` -- uninitialised
    // with `-already` or `-one` plus `-subarea`.  Unset, it prints nothing.
    let mut min_num: i32 = i32::MAX;
    let mut i_red = 0;
    let mut i_green = 0;
    let mut i_blue = 0;
    let mut imod_obj = 0;
    let mut new_nx: i32;
    let mut new_ny: i32;
    let mut icolor_obj: i32;
    let mut num_colors: i32;
    let mut iy_start: i32;
    let mut iy_end: i32;
    let mut nx_ub: i32;
    let mut ny_ub: i32;
    let mut ix_trim0: i32;
    let mut ix_trim1: i32;
    let mut iy_trim0: i32;
    let mut iy_trim1: i32;
    let mut ibinning_rec: i32;
    let if_pick_local: i32;
    let mut ibinning_preali: i32;
    let mut num_params = 0;
    let mut if_use_values: i32;
    let mut imod_cont = 0;
    let mut min_local_size = 0;
    let mut min_local_num = 0;
    let mut last_num_x: i32;
    let mut last_num_y: i32;
    let mut num_try_x: i32;
    let mut num_try_y: i32;
    let mut num_opt_arg = 0;
    let mut num_non_opt_arg = 0;
    // `use fortmodel`
    let mut fm = FortModel::default();

    if_flip = -1;
    local = 0;
    new_nx = 0;
    new_ny = 0;
    xtilt = 0.;
    ibinning_rec = 1;
    ibinning_preali = 1;
    ix_trim0 = 0;
    ix_trim1 = 0;
    iy_trim0 = 0;
    iy_trim1 = 0;
    invert_z = false;
    already = false;
    majority = false;
    rescale = false;
    check_groups = false;
    one_surface = false;
    text_output = String::new();
    if_use_values = 0;
    outlier_crit = 2.24;
    fm.fm_mod_size_type = 2;

    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "sortbeadsurfs",
        "ERROR: SORTBEADSURFS - ",
        false,
        1,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );

    if pip_get_in_out_file("InputFile", 1, " ", &mut in_file, 320) != 0 {
        exit_error("No input file specified");
    }
    if pip_get_in_out_file("OutputFile", 2, " ", &mut out_file, 320) != 0 {
        exit_error("No output file specified");
    }
    pip_get_integer(b"FlipYandZ", &mut if_flip);
    pip_get_integer(b"SubareaSize", &mut local);
    if_pick_local = 1 - pip_get_two_integers(
        b"PickAreasMinNumAndSize",
        &mut min_local_num,
        &mut min_local_size,
    );
    if local > 0 && if_pick_local > 0 {
        exit_error("You cannot enter both -subarea and -pick");
    }
    pip_get_float(b"XAxisTilt", &mut xtilt);
    pip_get_float(b"OutlierCriterionDeviation", &mut outlier_crit);
    pip_get_logical("AlreadySorted", &mut already);
    pip_get_logical("OneSurface", &mut one_surface);
    pip_get_logical("MajorityObjectOnly", &mut majority);
    pip_get_integer(b"ValuesToRestrainSorting", &mut if_use_values);
    {
        // `PipGetString` into the `character*320` `textOutput`
        let mut text: Vec<u8> = Vec::new();
        if pip_get_string(b"TextFileWithSurfaces", &mut text) == 0 {
            text.truncate(320);
            text_output = String::from_utf8_lossy(&text)
                .trim_end_matches(' ')
                .to_string();
        }
    }
    pip_get_two_integers(b"AlignedSizeXandY", &mut new_nx, &mut new_ny);
    pip_get_integer(b"PrealignedBinning", &mut ibinning_preali);
    pip_get_integer(b"ReconstructionBinning", &mut ibinning_rec);
    pip_get_logical("RescaleByBinnings", &mut rescale);
    pip_get_logical("InvertZAxis", &mut invert_z);
    pip_get_logical("CheckExistingGroups", &mut check_groups);
    pip_get_two_integers(b"XTrimStartAndEnd", &mut ix_trim0, &mut ix_trim1);
    pip_get_two_integers(b"YTrimStartAndEnd", &mut iy_trim0, &mut iy_trim1);
    pip_number_of_entries(b"SetSurfaceSortParam", &mut num_params);
    for _ in 1..=num_params {
        pip_get_two_floats(b"SetSurfaceSortParam", &mut eps_x, &mut eps_y);
        if set_surf_sort_param(eps_x.round() as i32, eps_y) != 0 {
            exit_error("Improper parameter setting for surface sort routine");
        }
    }
    //
    // Get the model and its size
    let exist = readw_or_imod(&in_file, &mut fm);
    if !exist {
        exit_error("Reading model file");
    }
    scale_model(0, &mut fm);
    getimodmaxes(&mut maxx, &mut maxy, &mut maxz);
    let max_pt = fm.max_pt as usize;
    let mut xyz = vec![0f32; 3 * max_pt];
    let mut xyz_fit = vec![0f32; 3 * max_pt];
    let mut values = vec![0f32; max_pt];
    let mut outlie = vec![0f32; max_pt];
    let mut igroup = vec![0i32; max_pt];
    let mut igrp_sort = vec![0i32; max_pt];
    let mut imod_obj_orig = vec![0i32; max_pt];
    memory_error(0, "arrays for model data");
    //
    // Flip coordinates if appropriate given ymax and zmax, or if user says to
    ind_y = 2;
    ind_z = 3;
    if (if_flip < 0 && maxz > maxy) || if_flip > 0 {
        ind_y = 3;
        ind_z = 2;
        ierr = maxy;
        maxy = maxz;
        maxz = ierr;
        if xtilt != 0. {
            exit_error("X axis tilt cannot be applied to a flipped model");
        }
        if new_nx != 0
            || new_ny != 0
            || ibinning_preali > 1
            || ibinning_rec > 1
            || rescale
            || invert_z
            || ix_trim0 > 0
            || ix_trim1 > 0
            || iy_trim0 > 0
            || iy_trim1 > 0
        {
            exit_error("You cannot do size adjustments or Z inversion with a flipped model");
        }
    }
    //
    // Find the X/Y range of the data and copy into a separate array
    // Rotate by the tilt angle not the negative of it.
    xmin = 1.0e20;
    xmax = -xmin;
    ymin = xmin;
    ymax = xmax;
    cosa = gfortran_cosd_r4(xtilt);
    sina = gfortran_sind_r4(xtilt);
    i = 0;
    for iobj in 1..=fm.max_mod_obj {
        let io = iobj as usize - 1;
        objtocont(iobj, &fm.obj_color, &mut imod_obj, &mut imod_cont);
        if if_use_values != 0 {
            if getcontvalue(imod_obj, imod_cont, &mut cont_value) != 0 {
                cont_value = 0.;
            }
        }
        for jj in 1..=fm.npt_in_obj[io] {
            i += 1;
            let k = 3 * (i as usize - 1);
            let iyp = fm.object[(fm.ibase_obj[io] + jj) as usize - 1] as usize - 1;
            xyz[k] = fm.p_coord[iyp][0];
            if xtilt != 0. {
                let dyy = fm.p_coord[iyp][ind_y - 1] - maxy as f32 / 2.;
                xyz[k + 1] = cosa * dyy - sina * fm.p_coord[iyp][ind_z - 1] + maxy as f32 / 2.;
                xyz[k + 2] = sina * dyy + cosa * fm.p_coord[iyp][ind_z - 1];
            } else {
                xyz[k + 1] = fm.p_coord[iyp][ind_y - 1];
                xyz[k + 2] = fm.p_coord[iyp][ind_z - 1];
            }
            if invert_z {
                xyz[k + 2] = (maxz - 1) as f32 - xyz[k + 2];
            }
            xmin = minss(xyz[k], xmin);
            ymin = minss(xyz[k + 1], ymin);
            xmax = maxss(xyz[k], xmax);
            ymax = maxss(xyz[k + 1], ymax);
            igroup[i as usize - 1] = -1;
            imod_obj_orig[i as usize - 1] = imod_obj;
            if if_use_values != 0 {
                if getpointvalue(imod_obj, imod_cont, jj, &mut values[i as usize - 1]) != 0 {
                    values[i as usize - 1] = cont_value;
                }
            }
        }
    }
    fm.n_point = i;

    // Do outlier analysis of values if they were obtained
    if if_use_values != 0 {
        rs_mad_median_outliers(&values, fm.n_point, outlier_crit, &mut outlie);
    }
    //
    // If points are already sorted, look at the object colors to deduce the sorting,
    // unless the one surface option is given
    if one_surface {
        for i in 1..=fm.n_point as usize {
            igroup[i - 1] = 1;
        }
    } else if already {
        num_colors = 0;
        for iobj in 1..=fm.max_mod_obj as usize {
            if fm.npt_in_obj[iobj - 1] > 0 {
                imod_obj = 256 - fm.obj_color[iobj - 1][1];
                getobjcolor(imod_obj, &mut i_red, &mut i_green, &mut i_blue);
                icolor_obj = 0;
                for icolor in 1..=num_colors as usize {
                    if i_red == j_red[icolor - 1]
                        && i_green == j_green[icolor - 1]
                        && i_blue == j_blue[icolor - 1]
                    {
                        icolor_obj = icolor as i32;
                    }
                }
                if icolor_obj == 0 {
                    if num_colors > 1 {
                        exit_error("There are more than two colors among the various objects");
                    }
                    num_colors += 1;
                    icolor_obj = num_colors;
                    j_red[icolor_obj as usize - 1] = i_red;
                    j_green[icolor_obj as usize - 1] = i_green;
                    j_blue[icolor_obj as usize - 1] = i_blue;
                }
                for ixx in 1..=fm.npt_in_obj[iobj - 1] {
                    let ii = fm.object[(fm.ibase_obj[iobj - 1] + ixx) as usize - 1];
                    igroup[ii as usize - 1] = icolor_obj;
                }
            }
        }
    } else {
        //
        // To do sorting, first set up subareas
        if if_pick_local > 0 {
            last_num_x = 1;
            last_num_y = 1;

            // For picking area size, loop from biggest possible size down to minimum
            let start = maxss(xmax - xmin, ymax - ymin).round() as i32;
            let mut local_try = start;
            'local_loop: while local_try >= min_local_size {
                num_try_x = 1.max(((xmax - xmin) / local_try as f32).round() as i32);
                num_try_y = 1.max(((ymax - ymin) / local_try as f32).round() as i32);

                // Skip to next size down if it gives numbers that have already been assessed
                if (num_try_x == last_num_x && num_try_y == last_num_y) || num_try_x * num_try_y < 2
                {
                    local_try -= 1;
                    continue;
                }

                // Finish with last good size if the area is now too small
                if (xmax - xmin) / (num_try_x as f32) < min_local_size as f32
                    || (ymax - ymin) / (num_try_y as f32) < min_local_size as f32
                {
                    break 'local_loop;
                }

                // Otherwise count up the number in each area and finish if it falls below min
                for ixx in 1..=num_try_x {
                    subarea_limits(xmin, xmax, num_try_x, ixx, &mut xlo, &mut xhi);
                    for iyy in 1..=num_try_y {
                        subarea_limits(ymin, ymax, num_try_y, iyy, &mut ylo, &mut yhi);
                        num_points = 0;
                        for i in 1..=fm.n_point as usize {
                            let k = 3 * (i - 1);
                            if xyz[k] >= xlo
                                && xyz[k] <= xhi
                                && xyz[k + 1] >= ylo
                                && xyz[k + 1] <= yhi
                            {
                                num_points += 1;
                            }
                        }
                        if num_points < min_local_num {
                            break 'local_loop;
                        }
                    }
                }

                // This is a good area size so set the local value from it
                local = local_try;
                last_num_x = num_try_x;
                last_num_y = num_try_y;
                local_try -= 1;
            }

            if local > 0 {
                println!(
                    "Picked a subarea size of{} to divide area into{} by{} subareas",
                    fmt_i(local, 7),
                    fmt_i(last_num_x, 4),
                    fmt_i(last_num_y, 4)
                );
            } else {
                println!(" Analyzing entire area; no subarea size fits the constraints");
            }
        }

        // Set up subareas with picked or entered local size
        num_area_x = 1;
        num_area_y = 1;
        if local > 0 {
            num_area_x = 1.max(((xmax - xmin) / local as f32).round() as i32);
            num_area_y = 1.max(((ymax - ymin) / local as f32).round() as i32);
        }
        min_num = fm.n_point + 1;
        //
        // Loop on the areas
        for ixx in 1..=num_area_x {
            subarea_limits(xmin, xmax, num_area_x, ixx, &mut xlo, &mut xhi);
            for iyy in 1..=num_area_y {
                subarea_limits(ymin, ymax, num_area_y, iyy, &mut ylo, &mut yhi);
                num_points = 0;
                for i in 1..=fm.n_point as usize {
                    //
                    // If a point is in area and hasn't been done yet, copy it over
                    // and mark as in the fitting group
                    let k = 3 * (i - 1);
                    if igroup[i - 1] < 0
                        && xyz[k] >= xlo
                        && xyz[k] <= xhi
                        && xyz[k + 1] >= ylo
                        && xyz[k + 1] <= yhi
                    {
                        num_points += 1;
                        let kf = 3 * (num_points as usize - 1);
                        xyz_fit[kf..kf + 3].copy_from_slice(&xyz[k..k + 3]);
                        igrp_sort[num_points as usize - 1] = 0;
                        if if_use_values as f32 * outlie[i - 1] > 0. {
                            igrp_sort[num_points as usize - 1] = -1;
                        }
                        igroup[i - 1] = 0;
                    }
                }
                if local > 0 && !check_groups {
                    println!(
                        "Subarea{}{} has{} points",
                        fmt_i(ixx, 3),
                        fmt_i(iyy, 3),
                        fmt_i(num_points, 6)
                    );
                }
                min_num = min_num.min(num_points);
                //
                // Do the fits to find surfaces
                if num_points > 1 {
                    ierr = surface_sort(&xyz_fit, num_points, if_use_values.abs(), &mut igrp_sort);
                    if ierr != 0 {
                        exit_error("Allocating memory in surfaceSort");
                    }
                }
                //
                // Copy the group numbers back
                j = 0;
                for i in 1..=fm.n_point as usize {
                    if igroup[i - 1] == 0 {
                        j += 1;
                        igroup[i - 1] = igrp_sort[j as usize - 1];
                    }
                }
            }
        }
    }
    //
    // Count numbers in groups
    for iyy in 1..=2usize {
        num_in_group[iyy - 1] = 0;
        for i in 1..=fm.n_point as usize {
            if igroup[i - 1] == iyy as i32 {
                num_in_group[iyy - 1] += 1;
            }
        }
        if !check_groups {
            println!(
                "Group{} has{} points",
                fmt_i(iyy as i32, 2),
                fmt_i(num_in_group[iyy - 1], 6)
            );
        }
    }
    //
    // Set up for majority group if desired, or to avoid empty object
    iy_start = 1;
    iy_end = 2;
    if majority || num_in_group[1] == 0 {
        if num_in_group[0] >= num_in_group[1] {
            iy_end = 1;
        }
        if num_in_group[0] < num_in_group[1] {
            iy_start = 2;
        }
    }
    //
    // Set up to shift the data if size changing and set up scaling
    dx = 0.;
    dy = 0.;
    nx_ub = maxx * ibinning_preali;
    ny_ub = maxy * ibinning_preali;
    scale_fac = 1.;
    if new_nx > 0 && new_ny > 0 {
        dx = (new_nx - nx_ub) as f32 / 2.;
        dy = (new_ny - ny_ub) as f32 / 2.;
        nx_ub = new_nx;
        ny_ub = new_ny;
    }
    if ix_trim0 > 0 && ix_trim1 > 0 {
        dx -= ((ix_trim0 - 1) * ibinning_rec) as f32;
        nx_ub = (ix_trim1 + 1 - ix_trim0) * ibinning_rec;
    }
    if iy_trim0 > 0 && iy_trim1 > 0 {
        dy -= ((iy_trim0 - 1) * ibinning_rec) as f32;
        ny_ub = (iy_trim1 + 1 - iy_trim0) * ibinning_rec;
    }
    dx /= ibinning_preali as f32;
    dy /= ibinning_preali as f32;
    if rescale {
        scale_fac = ibinning_preali as f32 / ibinning_rec as f32;
        nx_ub = (nx_ub as f32 / ibinning_rec as f32).round() as i32;
        ny_ub = (ny_ub as f32 / ibinning_rec as f32).round() as i32;
    } else {
        nx_ub = (nx_ub as f32 / ibinning_preali as f32).round() as i32;
        ny_ub = (ny_ub as f32 / ibinning_preali as f32).round() as i32;
    }
    putimodmaxes(nx_ub, ny_ub, maxz);
    //
    // Set object properties
    isize = 3;
    getscatsize(1, &mut isize);
    isize = 3.max(isize);
    deleteiobj();
    putscatsize(1, isize);
    putimodflag(1, 2);
    putobjcolor(1, 0, 255, 0);
    if iy_start != iy_end {
        putscatsize(2, isize);
        putimodflag(2, 2);
        putobjcolor(2, 255, 0, 255);
    }

    // Output text file if requested
    if !text_output.is_empty() {
        let mut unit2 = std::io::BufWriter::new(dopen(2, &text_output, "new", "f"));
        i = 0;
        for iobj in 1..=fm.max_mod_obj {
            objtocont(iobj, &fm.obj_color, &mut imod_obj, &mut imod_cont);
            for jj in 1..=fm.npt_in_obj[iobj as usize - 1] {
                i += 1;
                let _ = writeln!(
                    unit2,
                    "{}{}{}{}",
                    fmt_i(imod_obj, 8),
                    fmt_i(imod_cont, 8),
                    fmt_i(jj, 8),
                    fmt_i(igroup[i as usize - 1], 3)
                );
            }
        }
        let _ = unit2.flush();
    }
    //
    // then rebuild the model, make it one point per contour
    ix = 0;
    for iyy in iy_start..=iy_end {
        for i in 1..=fm.n_point as usize {
            if igroup[i - 1] == iyy {
                ix += 1;
                let x = ix as usize - 1;
                let k = 3 * (i - 1);
                fm.ibase_obj[x] = ix - 1;
                fm.npt_in_obj[x] = 1;
                fm.obj_color[x][0] = 1;
                fm.obj_color[x][1] = 255 - (iyy - iy_start);
                fm.p_coord[x][0] = scale_fac * (xyz[k] + dx);
                fm.p_coord[x][ind_y - 1] = scale_fac * (xyz[k + 1] + dy);
                fm.p_coord[x][ind_z - 1] = scale_fac * xyz[k + 2];
            }
        }
    }
    fm.max_mod_obj = ix;
    fm.n_point = ix;
    scale_model(1, &mut fm);
    write_wmod(&out_file, &mut fm);
    if check_groups {
        ierr = fm.n_point;
        for iyy in 1..=2 {
            ix = 0;
            for i in 1..=fm.n_point as usize {
                if (iyy == 1 && igroup[i - 1] != imod_obj_orig[i - 1])
                    || (iyy == 2 && igroup[i - 1] == imod_obj_orig[i - 1])
                {
                    ix += 1;
                }
            }
            ierr = ierr.min(ix);
        }
        if ierr > 0 {
            println!(
                "\nERROR: SORTBEADSURFS - {} has {} of {} points in the wrong group",
                in_file.trim_end_matches(' '),
                fmt_i(ierr, 5),
                fmt_i(fm.n_point, 5)
            );
        }
        exit(ierr);
    }
    if local > 0 && min_num <= 4 {
        println!(" Some areas have very few points - you should rerun this with larger subareas");
    } else if local > 0 && min_num <= 10 {
        println!(" If points are sorted incorrectly, try rerunning with larger subareas");
    }
    //
    exit(0);
}

/// Original: `subareaLimits` (`sortbeadsurfs.f90:437`).
pub fn subarea_limits(
    xmin: f32,
    xmax: f32,
    num_area_x: i32,
    ix: i32,
    xlo: &mut f32,
    xhi: &mut f32,
) {
    let eps_x: f32 = 0.0001 * (xmax - xmin);
    let delta_x: f32 = (xmax - xmin) / num_area_x as f32;
    *xlo = xmin + (ix as f32 - 1.) * delta_x - eps_x;
    *xhi = xmin + ix as f32 * delta_x + eps_x;
}
