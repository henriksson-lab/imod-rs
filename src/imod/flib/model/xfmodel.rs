//! Translation of `IMOD/flib/model/xfmodel.f90`.
//!
//! The main program maps to [`xfmodel`], its contained procedure to
//! [`warp_model`], and the two external subroutines to [`transform_model`]
//! and [`rescale_write_model`].  `warpModel` reaches its host's variables by
//! host association; here the host variables it reads or writes are passed
//! explicitly.  The `fortmodel` module arrays are the [`FortModel`] that
//! `readw_or_imod` fills, passed by reference to every unit that `use`s it.
//!
//! Transforms are held as `[f32; 6]` in Fortran `(2,3)` storage order.  The
//! grid arrays `warpDx(limWarpX, limWarpY, limWarpZ)` are flat, section `i`
//! starting at `(i - 1) * limWarpX * limWarpY`; `xmat(MSIZE_XMAT, idim)` is
//! flat with the column index fastest, as `findTransform` takes it.  Library
//! calls go to the C entry points with the Fortran wrappers' `iz - 1` and
//! `rows = 2` inlined (`warpwrapfort.c`, `linearxforms.c`), and `findxf`'s
//! error exit (`findtransform.c:229-232`) is inlined at its one call.
//!
//! Formatted output is written with the gfortran editing the source asks for:
//! `Fw.d` (overflow is `w` asterisks; a leading zero is dropped when that is
//! what makes the value fit; `d = 0` keeps the decimal point) and `Iw`, and
//! list-directed `print *` puts a blank before the record, writes an integer
//! as a blank and `I11`, and a blank between a number and a following string.

use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::rdlist::{parselist, rdlist};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::scale_model;
use crate::imod::flib::subrs::model::write_wmod::write_wmod;
use crate::imod::flib::subrs::piecesubs::read_piece_list::read_piece_list;
use crate::imod::flib::subrs::xfsubs::readdistortions::read_mag_gradients;
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall2;
use crate::imod::flib::subrs::xfsubs::xfwrite::xfwrite;
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::findtransform::find_transform;
use crate::imod::libcfshr::linearxforms::{xf_apply, xf_copy, xf_invert, xf_unit};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_string, pip_get_two_floats,
};
use crate::imod::libcfshr::piecefuncs::{checklist, fill_listz};
use crate::imod::libiimod::unit_header::iiu_ret_delta;
use crate::imod::libimod::imodel_fwrap::{getimodmaxes, getimodscales};
use crate::imod::libwarp::maggradfield::mag_gradient_shift;
use crate::imod::libwarp::warpfiles::{
    clear_warp_file, get_grid_parameters, get_linear_transform, set_current_warp_file,
};
use crate::imod::libwarp::warputils::{
    find_inverse_point, find_max_grid_size, get_size_adjusted_grid, interpolate_grid,
    read_check_warp_file,
};
use std::io::{BufRead, BufReader, BufWriter, Write};

/// `parameter (LIMNUMXF = 100000, ...)` (`xfmodel.f90:15`).
const LIMNUMXF: i32 = 100000;
/// `parameter (..., LIMPCLIST = 100000)` (`xfmodel.f90:15`).
const LIMPCLIST: i32 = 100000;
/// `parameter (MSIZE_XMAT = 19)` (`xfmodel.f90:23`).
const MSIZE_XMAT: i32 = 19;
/// `parameter (numOptions = 25)` (`xfmodel.f90:70`).
const NUM_OPTIONS: i32 = 25;
/// Fallback PIP table `options(1)` (`xfmodel.f90:72-82`).
const OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@image:ImageFile:FN:@\
piece:PieceListFile:FN:@allz:AllZhaveTransforms:B:@center:CenterInXandY:FP:@\
transonly:TranslationOnly:B:@rottrans:RotationTranslation:B:@\
magrot:MagRotTrans:B:@sections:SectionsToAnalyze:LI:@single:SingleSection:I:@\
full:FullReportMeanAndMax:FP:@prealign:PrealignTransforms:FN:@\
edit:EditTransforms:FN:@xforms:XformsToApply:FN:@useline:UseTransformLine:I:@\
chunks:ChunkSizes:LI:@back:BackTransform:B:@scale:ScaleShifts:F:@\
adjust:AdjustForRotationBy90:I:@distort:DistortionField:FN:@\
binning:BinningOfImages:F:@gradient:GradientFile:FN:@param:ParameterFile:PF:@\
help:usage:B:";

/// Original program `xfmodel` (`xfmodel.f90:11`).
///
/// Will take a model from wimp or imod, and either
/// a) use corresponding points in two sections to obtain a transformation
/// between the sections, or
/// b) transform the points in the model to match a new alignment of images
pub fn xfmodel() {
    let limnumxf = LIMNUMXF as usize;
    let mut f_xf: Vec<[f32; 6]> = vec![[0.0; 6]; limnumxf];
    let mut g_xf: Vec<[f32; 6]> = vec![[0.0; 6]; limnumxf];
    let mut g_temp = [0.0_f32; 6];
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut mode = 0_i32;
    // `real*4 delta(3) /0., 0., 0./`
    let mut delta = [0.0_f32; 3];
    let mut num_sec = vec![0_i32; limnumxf];
    let mut list_z = vec![0_i32; limnumxf];
    let mut ind_zto_flist = vec![0_i32; limnumxf];
    let mut num_in_chunks = vec![0_i32; limnumxf];
    let mut num_chunks: i32;
    let mut ix_pc_list = vec![0_i32; LIMPCLIST as usize];
    let mut iy_pc_list = vec![0_i32; LIMPCLIST as usize];
    let mut iz_pc_list = vec![0_i32; LIMPCLIST as usize];
    let mut xmat: Vec<f32>;
    let mut modelfile = String::new();
    let mut new_model = String::new();
    let mut old_xfg_file: String;
    let mut old_xf_file: String;
    let mut new_xf_file = String::new();
    let mut idf_file: String;
    let mut mag_grad_file: String;
    let mut got_this: bool;
    let mut got_last: bool;
    let exist: bool;
    // `integer*4 limPoints/4/`
    let mut lim_points = 4_i32;
    //
    let mut i: i32 = 0;
    let mut num_list_z: i32;
    let mut num_fout: i32;
    let mut num_pc_list = 0_i32;
    let iz_range: i32;
    let mut if_fill_gap: i32;
    let mut ind_f_xf: i32;
    let mut ind_val: i32;
    let (mut min_xpiece, mut num_xpieces, mut nx_overlap) = (0_i32, 0_i32, 0_i32);
    let (mut min_ypiece, mut num_ypieces, mut ny_overlap) = (0_i32, 0_i32, 0_i32);
    let x_half_size: f32;
    let y_half_size: f32;
    let mut x_cen: f32;
    let mut y_cen: f32;
    let mut report_mean_crit = 0.0_f32;
    let mut report_max_crit = 0.0_f32;
    let (mut dmin, mut dmax) = (0.0_f32, 0.0_f32);
    let mut ifxfmod: i32;
    let mut if_trans: i32;
    let mut if_rot_trans: i32;
    let mut if_prealign: i32;
    let mut num_to_find: i32;
    let mut if_mag_rot: i32;
    let mut if_single: i32;
    let mut iz_single = 0_i32;
    let mut if_full_report: i32;
    let mut ierr: i32;
    let mut ind_g_xf: i32;
    // Not assigned by the interactive branch of the source, which reads it
    // at `xfmodel.f90:602` and `:672` all the same (uninitialised there); 0.
    let mut if_shift_scale = 0_i32;
    let mut zz = 0.0_f32;
    let mut z_index: f32;
    let mut z_this = 0.0_f32;
    let mut z_last = 0.0_f32;
    let mut shift_scale: f32;
    let mut dmean = 0.0_f32;
    let mut num_undefined = 0_i32;
    let mut num_fg_in = 0_i32;
    let mut iz_min: i32;
    let mut iz_max: i32;
    let mut ibase: i32;
    let mut last_sec: i32;
    let mut num_points: i32;
    let mut num_in_obj: i32;
    let mut ipnt: i32;
    // Uninitialised in the source until `findTransform` sets it; it is not
    // set when every deviation is NaN, and the source then indexes `xmat`
    // with whatever it held.  1 here, so that case reads the first point.
    let mut ipnt_max = 1_i32;
    let mut devmax = 0.0_f32;
    let mut dev_avg = 0.0_f32;
    let mut dev_sd = 0.0_f32;
    let mut x_last: f32;
    let mut y_last: f32;
    let mut x_new = 0.0_f32;
    let mut y_new = 0.0_f32;
    let mut num_old_g = 0_i32;
    let mut iz_sec: i32;
    let (mut max_x, mut max_y, mut max_z) = (0_i32, 0_i32, 0_i32);
    let mut if_back: i32;
    let mut iter: i32;
    let mut line_use: i32;
    let mut line_to_use: i32;
    let mut if_mag_grad: i32;
    let mut ind_control = 0_i32;
    let mut done: bool;
    let mut iz = 0_i32;
    //
    let mut if_distort: i32;
    let mut idf_binning = 0_i32;
    let (mut idf_nx, mut idf_ny) = (0_i32, 0_i32);
    let mut ind_idf: i32;
    let mut ind_pre_warp: i32;
    let (mut nx_grid, mut ny_grid, mut lim_grid) = (0_i32, 0_i32, 0_i32);
    let (mut i_pre_warp_nx, mut i_pre_warp_ny) = (0_i32, 0_i32);
    let (mut iwarp_nx, mut iwarp_ny) = (0_i32, 0_i32);
    let (mut x_grid_strt, mut y_grid_strt) = (0.0_f32, 0.0_f32);
    let (mut x_grid_intrv, mut y_grid_intrv) = (0.0_f32, 0.0_f32);
    let mut pixel_idf = 0.0_f32;
    let mut bin_ratio: f32;
    let mut f_binning: f32;
    let mut pixel_model: f32;
    let mut pixel_warp = 0.0_f32;
    let mut pixel_prewarp = 0.0_f32;
    let mut prewarp_scale = 0.0_f32;
    let mut warp_scale = 0.0_f32;
    let mut field_dx: Vec<f32> = Vec::new();
    let mut field_dy: Vec<f32> = Vec::new();
    let mut warp_dx: Vec<f32> = Vec::new();
    let mut warp_dy: Vec<f32> = Vec::new();
    let mut x_warp_start: Vec<f32> = Vec::new();
    let mut y_warp_start: Vec<f32> = Vec::new();
    let mut x_warp_intrv: Vec<f32> = Vec::new();
    let mut y_warp_intrv: Vec<f32> = Vec::new();
    let mut num_control: Vec<i32> = Vec::new();
    let mut nx_warp: Vec<i32> = Vec::new();
    let mut ny_warp: Vec<i32> = Vec::new();
    let mut string_list = String::new();
    let (mut pixel_mag_grad, mut axis_rot) = (0.0_f32, 0.0_f32);
    let (mut xmod_min, mut xmod_max, mut y_mod_min, mut y_mod_max): (f32, f32, f32, f32);
    let grid_extend_frac: f32;
    let mut tilt_angles = vec![0.0_f32; limnumxf];
    let mut dmag_per_um = vec![0.0_f32; limnumxf];
    let mut rot_per_um = vec![0.0_f32; limnumxf];
    let (mut dx, mut dy, mut dx1, mut dy1): (f32, f32, f32, f32);
    let mut num_mag_grad = 0_i32;
    let mut ind_warp_file: i32;
    let (mut lim_warp_x, mut lim_warp_y, mut lim_warp_z): (i32, i32, i32);
    let mut if_adjust_for90: i32;
    //
    let pip_input: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    // `use fortmodel`
    let mut fm = FortModel::default();

    // gfortran `Fw.d` and `Iw` output editing.
    let fmt_f = |value: f32, w: usize, d: usize| -> String {
        if value.is_nan() {
            return format!("{:>w$}", "NaN");
        }
        if value.is_infinite() {
            let mut text = if value < 0. { "-Infinity" } else { "Infinity" };
            if text.len() > w {
                text = if value < 0. { "-Inf" } else { "Inf" };
            }
            if text.len() > w {
                return "*".repeat(w);
            }
            return format!("{text:>w$}");
        }
        let mut text = format!("{value:.d$}");
        if d == 0 {
            text.push('.');
        }
        if text.len() > w {
            if let Some(rest) = text.strip_prefix("0.") {
                text = format!(".{rest}");
            } else if let Some(rest) = text.strip_prefix("-0.") {
                text = format!("-.{rest}");
            }
        }
        if text.len() > w {
            return "*".repeat(w);
        }
        format!("{text:>w$}")
    };
    let fmt_i = |value: i32, w: usize| -> String {
        let text = format!("{value}");
        if text.len() > w {
            return "*".repeat(w);
        }
        format!("{text:>w$}")
    };
    // `read(*,*)` with no `END=`/`ERR=`: the gfortran runtime reports a
    // failed read and stops with status 2.
    let read_abort = |err: ListReadError| -> ! {
        let _ = std::io::stdout().flush();
        match err {
            ListReadError::End => eprintln!("Fortran runtime error: End of file"),
            ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
        }
        exit(2);
    };
    // `read(*,'(a)') name`, likewise with no `END=`.
    let read_a = || -> String {
        let mut line = String::new();
        if matches!(std::io::stdin().lock().read_line(&mut line), Ok(0) | Err(_)) {
            eprintln!("Fortran runtime error: End of file");
            exit(2);
        }
        line.trim_end_matches(['\r', '\n']).to_owned()
    };
    // The Fortran wrapper `pipgetstring` (`pip_fwrap.c:206`): the variable is
    // left untouched unless the option is found.
    let get_string = |option: &[u8], string: &mut String| -> i32 {
        let mut value: Vec<u8> = Vec::new();
        let err = pip_get_string(option, &mut value);
        if err == 0 {
            *string = String::from_utf8_lossy(&value).into_owned();
        }
        err
    };
    let blank = |string: &str| string.bytes().all(|b| b == b' ');
    //
    // set defaults
    //
    x_cen = 0.;
    y_cen = 0.;
    if_back = 0;
    if_adjust_for90 = 0;
    f_binning = 1.;
    if_distort = 0;
    if_mag_grad = 0;
    num_chunks = 0;
    shift_scale = 1.;
    grid_extend_frac = 0.1;
    let mut idim: i32 = 100000;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "xfmodel",
        "ERROR: XFMODEL - ",
        true,
        2,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pip_input = num_opt_arg + num_non_opt_arg > 0;

    //
    // get parameters
    //
    if pip_input {
        get_string(b"ImageFile", &mut modelfile);
    } else {
        print!(" Image file (or Return to enter Xcen, Ycen directly): ");
        let _ = std::io::stdout().flush();
        modelfile = read_a();
    }
    //
    // if no image file, get crucial info directly
    // DNM 8/28/02: just get xcen, ycen since origin and delta aren't needed
    //
    if blank(&modelfile) {
        if pip_input {
            pip_get_two_floats(b"CenterInXandY", &mut x_cen, &mut y_cen);
        } else {
            println!(" \u{7}Be SURE to enter CENTER coordinates, NOT NX and NY");
            print!(" Xcen, Ycen: ");
            let _ = std::io::stdout().flush();
            if let Err(err) = list_read(
                &mut std::io::stdin().lock(),
                &mut [ListItem::Real(&mut x_cen), ListItem::Real(&mut y_cen)],
            ) {
                read_abort(err);
            }
        }
        for i in 1..=LIMNUMXF {
            list_z[i as usize - 1] = i - 1;
        }
        num_list_z = LIMNUMXF;
        num_fout = 0;
    } else {
        //
        // otherwise get header info from image file
        imopen(1, modelfile.trim_end_matches(' '), "ro");
        //
        // get header info for proper coordinate usage
        // SAFETY: `irdhdr` writes three integers into each of `nxyz` and
        // `mxyz` and one value through each scalar pointer, all live locals.
        unsafe {
            irdhdr(
                1,
                nxyz.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &raw mut mode,
                &raw mut dmin,
                &raw mut dmax,
                &raw mut dmean,
            );
        }
        delta = iiu_ret_delta(1);
        // call irtorg(1, xorig, yorig, zorig)
        // write(*,'(/,a,a,/)') ' This header info MUST be the' &
        // , ' same as when model was built'
        //
        modelfile = String::new();
        if pip_input {
            get_string(b"PieceListFile", &mut modelfile);
        } else {
            print!(" Piece list file if image is a montage, otherwise Return: ");
            let _ = std::io::stdout().flush();
            modelfile = read_a();
        }
        read_piece_list(
            modelfile.trim_end_matches(' '),
            &mut ix_pc_list,
            &mut iy_pc_list,
            &mut iz_pc_list,
            &mut num_pc_list,
        );
        if num_pc_list > LIMPCLIST {
            exit_error("too many piece coordinates for arrays");
        }
        let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);
        //
        // if no pieces, set up mocklist
        if num_pc_list == 0 {
            for i in 1..=nz {
                ix_pc_list[i as usize - 1] = 0;
                iy_pc_list[i as usize - 1] = 0;
                iz_pc_list[i as usize - 1] = i - 1;
            }
            num_pc_list = nz;
        }
        // get ordered list of z values
        let mut number_list_z = 0_usize;
        fill_listz(
            &iz_pc_list[..num_pc_list as usize],
            &mut list_z,
            &mut number_list_z,
        );
        num_list_z = number_list_z as i32;
        if num_list_z > LIMNUMXF {
            exit_error("Too many Z values for arrays");
        }

        checklist(
            &ix_pc_list[..num_pc_list as usize],
            1,
            nx,
            &mut min_xpiece,
            &mut num_xpieces,
            &mut nx_overlap,
        );
        checklist(
            &iy_pc_list[..num_pc_list as usize],
            1,
            ny,
            &mut min_ypiece,
            &mut num_ypieces,
            &mut ny_overlap,
        );
        x_half_size = (nx + (num_xpieces - 1) * (nx - nx_overlap)) as f32 / 2.;
        y_half_size = (ny + (num_ypieces - 1) * (ny - ny_overlap)) as f32 / 2.;

        // DNM 8/28/02: don't scale any more
        // but still add minxpiece - the index coordinates are montage
        // coordinates starting at minxpiece.
        //
        // xcen=(minxpiece+xhaf)*delt(1) -xorig
        // ycen=(minypiece+yhaf)*delt(2) -yorig
        x_cen = min_xpiece as f32 + x_half_size;
        y_cen = min_ypiece as f32 + y_half_size;
        num_fout = num_list_z;
    }
    //
    // find out if gaps in z and ask how to store f's if there are gaps
    //
    iz_range = list_z[num_list_z as usize - 1] + 1 - list_z[0];
    if_fill_gap = 0;
    if iz_range > num_list_z {
        println!(
            " There are{} gaps in section Z values",
            fmt_i(iz_range - num_list_z, 4)
        );
        if pip_input {
            pip_get_boolean(b"AllZhaveTransforms", &mut if_fill_gap);
            if if_fill_gap == 0 {
                println!(" Xform lists will be assumed to have xforms only for existing sections");
            } else {
                println!(" Xform lists will be assumed to have xforms for all sections");
            }
        } else {
            print!(
                " Enter 0 for xform lists that have xforms only for existing sections,\n    or 1 for xform lists that have xforms for all Z values in range: "
            );
            let _ = std::io::stdout().flush();
            if let Err(err) = list_read(
                &mut std::io::stdin().lock(),
                &mut [ListItem::Integer(&mut if_fill_gap)],
            ) {
                read_abort(err);
            }
        }
    }
    //
    // make index from z values to transform list: index=0 for non-existent
    //
    if pip_input && get_string(b"ChunkSizes", &mut string_list) == 0 {
        let _ = parselist(&string_list, &mut num_in_chunks, &mut num_chunks);
    }
    for i in 1..=LIMNUMXF {
        ind_zto_flist[i as usize - 1] = 0;
    }
    if num_chunks == 0 {
        for i in 1..=num_list_z {
            ind_f_xf = list_z[i as usize - 1] + 1 - list_z[0];
            ind_val = i;
            if if_fill_gap != 0 {
                ind_val = ind_f_xf;
            }
            ind_zto_flist[ind_f_xf as usize - 1] = ind_val;
        }
        if num_fout > 0 {
            num_fout =
                ind_zto_flist[(list_z[num_list_z as usize - 1] + 1 - list_z[0]) as usize - 1];
        }
    } else {
        //
        // or fill index list
        //
        // print *,numChunks, (numInChunks(j), j=1, numChunks)
        ind_f_xf = 1;
        for j in 1..=num_chunks {
            for _ in 1..=num_in_chunks[j as usize - 1] {
                ind_zto_flist[ind_f_xf as usize - 1] = j;
                ind_f_xf += 1;
                if ind_f_xf >= LIMNUMXF {
                    exit_error("Too many sections in chunks for arrays");
                }
            }
        }
    }
    // print *,(indfl(i), i=1, nlistz)
    //
    ifxfmod = 0;
    if_prealign = 0;
    if pip_get_in_out_file("InputFile", 1, "Input model file", &mut modelfile) != 0 {
        exit_error("No input file specified");
    }
    old_xfg_file = String::new();
    old_xf_file = String::new();
    idf_file = String::new();
    mag_grad_file = String::new();
    if_trans = 0;
    if_rot_trans = 0;
    if_mag_rot = 0;
    line_use = -1;
    if pip_input {
        pip_get_boolean(b"TranslationOnly", &mut if_trans);
        pip_get_boolean(b"RotationTranslation", &mut if_rot_trans);
        pip_get_boolean(b"MagRotTrans", &mut if_mag_rot);
        pip_get_integer(b"UseTransformLine", &mut line_use);
        pip_get_integer(b"AdjustForRotationBy90", &mut if_adjust_for90);
        if_shift_scale = 1 - pip_get_float(b"ScaleShifts", &mut shift_scale);

        if get_string(b"XformsToApply", &mut old_xf_file) == 0 {
            ifxfmod = 1;
        }
        if get_string(b"DistortionField", &mut idf_file) == 0 {
            ifxfmod = 1;
        }
        if get_string(b"GradientFile", &mut mag_grad_file) == 0 {
            ifxfmod = 1;
        }
        if get_string(b"PrealignTransforms", &mut old_xfg_file) == 0 {
            if_prealign = 1;
        }
        //
        // if back-transform, first check for legality
        //
        if pip_get_boolean(b"BackTransform", &mut if_back) == 0 {
            if !blank(&old_xf_file) && !blank(&old_xfg_file) {
                exit_error("You cannot enter both -xform and -prealign with -back");
            }
            if ifxfmod + if_prealign == 0 {
                exit_error("You must enter -xform, -prealign, -distort or -gradient with -back");
            }
            //
            // in any case, set filename for use in back-transform, clear out
            // transform filename, set flag for prealign back-transform if any
            // Set xfmodel -1 if xform is givem, or if preali with no
            // distortions; i.e. preali with distortion will retransform forward
            // to original (distorted) aligned stack
            //
            if !blank(&old_xf_file) || (blank(&idf_file) && blank(&mag_grad_file)) {
                ifxfmod = -1;
            }
            if !blank(&old_xf_file) {
                old_xfg_file = old_xf_file.clone();
            }
            old_xf_file = String::new();
            if !blank(&old_xfg_file) {
                if_prealign = 1;
            }
        }
        //
        if if_trans + if_rot_trans + if_mag_rot > 1 {
            exit_error("Only one of -trans, -rottrans, -magrot may be entered");
        }
        if if_trans + if_rot_trans + if_mag_rot > 0 && ifxfmod != 0 {
            exit_error("You cannot both find transforms and transform a model");
        }
        if ifxfmod == 0 && if_shift_scale != 0 {
            exit_error("You cannot enter -scale unless you are transforming a model");
        }
        if if_trans != 0 {
            ifxfmod = 2;
        }
        if if_rot_trans != 0 {
            ifxfmod = 3;
        }
        if if_mag_rot != 0 {
            ifxfmod = 4;
        }

        if !(ifxfmod == -1 || (ifxfmod == 1 && (!blank(&old_xf_file) || !blank(&old_xfg_file)))) {
            if line_use >= 0 {
                exit_error("You cannot enter -useline unless you are transforming a model");
            }
            if num_chunks > 0 {
                exit_error("You cannot enter -chunks unless you are transforming a model");
            }
        }
        if line_use >= 0 && num_chunks > 0 {
            exit_error("It is meaningless to enter both -useline and -chunks");
        }
    } else {
        //
        print!(
            " Enter 0 to find transformations, 2 to find X/Y translations only,\n    3 to find translations and rotations only,\n    4 to find translation, rotation and mag change only,\n    1 to transform model, or -1 to back-transform model: "
        );
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Integer(&mut ifxfmod)],
        ) {
            read_abort(err);
        }
    }
    //
    // set variables for restricted fits
    //
    if ifxfmod == 2 {
        ifxfmod = 0;
        if_trans = 1;
        lim_points = 1;
    } else if ifxfmod == 3 {
        if_rot_trans = 1;
        ifxfmod = 0;
        lim_points = 2;
    } else if ifxfmod == 4 {
        if_rot_trans = 2;
        ifxfmod = 0;
        lim_points = 3;
    }
    //
    // read in the model now since its range is needed
    exist = readw_or_imod(modelfile.trim_end_matches(' '), &mut fm);
    if !exist {
        // `goto 91`
        exit_error("Reading model file");
    }
    //
    // shift the data to image coordinates before using
    scale_model(0, &mut fm);
    //
    // Get model range
    xmod_min = 1.0e38;
    xmod_max = -1.0e38;
    y_mod_min = 1.0e38;
    y_mod_max = -1.0e38;
    for i in 1..=fm.n_point {
        let p = fm.p_coord[i as usize - 1];
        xmod_min = if xmod_min < p[0] { xmod_min } else { p[0] };
        xmod_max = if xmod_max > p[0] { xmod_max } else { p[0] };
        y_mod_min = if y_mod_min < p[1] { y_mod_min } else { p[1] };
        y_mod_max = if y_mod_max > p[1] { y_mod_max } else { p[1] };
    }
    //
    // Extend the range so that properly extrapolated grids can be used to find inverse
    // point for forward transforms
    // (`xmodMax + gridExtendFrac + xLast` is `+`, not `*`, in the source;
    // BUGS.md.)
    x_last = xmod_max + 1. - xmod_min;
    xmod_min -= grid_extend_frac * x_last;
    xmod_max = xmod_max + grid_extend_frac + x_last;
    y_last = y_mod_max + 1. - y_mod_min;
    y_mod_min -= grid_extend_frac * y_last;
    y_mod_max = y_mod_max + grid_extend_frac + y_last;
    //
    if_single = 0;
    if_full_report = 0;
    num_to_find = 0;
    if pip_input {
        if pip_get_in_out_file("OutputFile", 2, " ", &mut new_xf_file) != 0 {
            exit_error("No output file specified");
        }
        if ifxfmod == 0 {
            old_xf_file = String::new();
            get_string(b"EditTransforms", &mut old_xf_file);

            if pip_get_two_floats(
                b"FullReportMeanAndMax",
                &mut report_mean_crit,
                &mut report_max_crit,
            ) == 0
            {
                if_full_report = 1;
            }
            if pip_get_integer(b"SingleSection", &mut iz_single) == 0 {
                if_single = 1;
            }
            if get_string(b"SectionsToAnalyze", &mut string_list) == 0 {
                let _ = parselist(&string_list, &mut num_sec, &mut num_to_find);
            }
        } else {
            new_model = new_xf_file.clone();
            if !blank(&idf_file) {
                if_distort = 1;
                ind_idf = read_check_warp_file(
                    idf_file.trim_end_matches(' '),
                    1,
                    1,
                    &mut idf_nx,
                    &mut idf_ny,
                    &mut i,
                    &mut idf_binning,
                    &mut pixel_idf,
                    &mut iz,
                    &mut string_list,
                );
                if ind_idf < 0 {
                    exit_error(&string_list);
                }
                if get_grid_parameters(
                    0,
                    &mut nx_grid,
                    &mut ny_grid,
                    &mut x_grid_strt,
                    &mut y_grid_strt,
                    &mut x_grid_intrv,
                    &mut y_grid_intrv,
                ) != 0
                {
                    exit_error("Getting distortion field parameters");
                }
                lim_grid = nx_grid
                    .max(((xmod_max - xmod_min) / x_grid_intrv).ceil() as i32)
                    .max(ny_grid)
                    .max(((y_mod_max - y_mod_min) / y_grid_intrv).ceil() as i32)
                    + 2;
                field_dx = vec![0.0; (lim_grid * lim_grid) as usize];
                field_dy = vec![0.0; (lim_grid * lim_grid) as usize];
                memory_error(0, "arrays for distortion field");
                //
                // if the center is not yet defined, need to get it now
                //
                if x_cen == 0. && y_cen == 0. {
                    getimodmaxes(&mut max_x, &mut max_y, &mut max_z);
                    x_cen = max_x as f32 / 2.;
                    y_cen = max_y as f32 / 2.;
                    println!(
                        "Using model header to determine center coordinates:{}{}",
                        fmt_f(x_cen, 8, 1),
                        fmt_f(y_cen, 8, 1)
                    );
                }
                //
                // insist on binning unless situation is unambiguous, and convert
                // the distortion field by the difference in binning
                //
                if pip_get_float(b"BinningOfImages", &mut f_binning) != 0
                    && 2. * x_cen <= (idf_nx * idf_binning / 2) as f32
                    && 2. * y_cen <= (idf_ny * idf_binning / 2) as f32
                {
                    exit_error(
                        "You must specify binning of images because they are not larger than half the camera size",
                    );
                }
                if f_binning <= 0. {
                    exit_error("Image binning must be a positive number");
                }
                bin_ratio = 1.;
                if f_binning != idf_binning as f32 {
                    bin_ratio = idf_binning as f32 / f_binning;
                }
                if get_size_adjusted_grid(
                    0,
                    (xmod_max - xmod_min) / bin_ratio,
                    (y_mod_max - y_mod_min) / bin_ratio,
                    ((xmod_max + xmod_min) / bin_ratio - idf_nx as f32) / 2.,
                    ((y_mod_max + y_mod_min) / bin_ratio - idf_ny as f32) / 2.,
                    0,
                    bin_ratio,
                    1,
                    &mut nx_grid,
                    &mut ny_grid,
                    &mut x_grid_strt,
                    &mut y_grid_strt,
                    &mut x_grid_intrv,
                    &mut y_grid_intrv,
                    &mut field_dx,
                    &mut field_dy,
                    lim_grid,
                    lim_grid,
                    &mut string_list,
                ) != 0
                {
                    exit_error(&string_list);
                }
                clear_warp_file(ind_idf);
                //
                // Need to shift field by difference between image and camera
                // centers
                //
                x_grid_strt -= idf_nx as f32 * bin_ratio / 2. - x_cen;
                y_grid_strt -= idf_ny as f32 * bin_ratio / 2. - y_cen;
                // print *,ixGridStrt, ixGridStrt, xcen, ycen, idfnx, idfny, binratio
            }
        }
        //
        // mag gradients now
        //
        if !blank(&mag_grad_file) {
            if_mag_grad = 1;
            read_mag_gradients(
                mag_grad_file.trim_end_matches(' '),
                LIMNUMXF,
                &mut pixel_mag_grad,
                &mut axis_rot,
                &mut tilt_angles,
                &mut dmag_per_um,
                &mut rot_per_um,
                &mut num_mag_grad,
            );
        }
    } else {
        //
        // old input: get prealignment or back transforms
        //
        if ifxfmod < 0 {
            if_prealign = 1;
        } else {
            print!(" Model built on raw sections (0) or prealigned ones(1): ");
            let _ = std::io::stdout().flush();
            if let Err(err) = list_read(
                &mut std::io::stdin().lock(),
                &mut [ListItem::Integer(&mut if_prealign)],
            ) {
                read_abort(err);
            }
        }
        //
        if if_prealign != 0 {
            print!(" File of old g transforms used in prealignment: ");
            let _ = std::io::stdout().flush();
            old_xfg_file = read_a();
        }
        //
        if ifxfmod == 0 {
            print!(" Name of old file of f transforms to edit (Return if none): ");
            let _ = std::io::stdout().flush();
            old_xf_file = read_a();
            //
            print!(" Name of new file of f transforms: ");
            let _ = std::io::stdout().flush();
            new_xf_file = read_a();
            //
            // `write(*,118)`
            println!(" Enter / to find transforms for all section pairs,");
            println!("      or -999 to find transforms relative to a single section,");
            println!("      or a list of the section numbers to find transforms for");
            println!("         (enter number of second section of each pair, ranges are ok)");
            let _ = rdlist(&mut std::io::stdin().lock(), &mut num_sec, &mut num_to_find);
            if num_to_find == 1 && num_sec[0] == -999 {
                print!(" Number of single section to find transforms relative to: ");
                let _ = std::io::stdout().flush();
                if let Err(err) = list_read(
                    &mut std::io::stdin().lock(),
                    &mut [ListItem::Integer(&mut iz_single)],
                ) {
                    read_abort(err);
                }
                if_single = 1;
                println!(" Now enter list of sections to find transforms for (/ for all)");
                num_to_find = 0;
                let _ = rdlist(&mut std::io::stdin().lock(), &mut num_sec, &mut num_to_find);
            }
            //
            print!(" 1 for full reports of deviations for sections with bad fits, 0 for none: ");
            let _ = std::io::stdout().flush();
            if let Err(err) = list_read(
                &mut std::io::stdin().lock(),
                &mut [ListItem::Integer(&mut if_full_report)],
            ) {
                read_abort(err);
            }
            //
            if if_full_report != 0 {
                // `write(*,'(1x,a,$)')` of two items against one `a` edit
                // descriptor: format reversion re-applies `1x` but, with the
                // `$`, gfortran starts no new record, so both land on one line.
                print!(
                    " Enter criterion mean deviation and max deviation; full reports will be given      for sections with mean OR max greater than these criteria: "
                );
                let _ = std::io::stdout().flush();
                if let Err(err) = list_read(
                    &mut std::io::stdin().lock(),
                    &mut [
                        ListItem::Real(&mut report_mean_crit),
                        ListItem::Real(&mut report_max_crit),
                    ],
                ) {
                    read_abort(err);
                }
            }
            //
        } else {
            print!(" New model file name: ");
            let _ = std::io::stdout().flush();
            new_model = read_a();
            //
            if ifxfmod > 0 {
                print!(" File for list of g transforms to apply: ");
                let _ = std::io::stdout().flush();
                old_xf_file = read_a();
            }
            //
        }
    }
    pip_done();
    //
    // if the center is not yet defined, use the model header sizes
    //
    if x_cen == 0. && y_cen == 0. {
        getimodmaxes(&mut max_x, &mut max_y, &mut max_z);
        x_cen = max_x as f32 / 2.;
        y_cen = max_y as f32 / 2.;
        println!(
            "Using model header to determine center coordinates:{}{}",
            fmt_f(x_cen, 8, 1),
            fmt_f(y_cen, 8, 1)
        );
    }
    //
    // first fill array with unit transforms in case things get weird
    //
    for i in 1..=LIMNUMXF {
        xf_unit(&mut f_xf[i as usize - 1], 1., 2);
    }
    //
    // Assess whether either file is a warping file
    ind_pre_warp = -1;
    ind_warp_file = -1;
    lim_warp_x = 0;
    lim_warp_y = 0;
    lim_warp_z = 0;
    if if_prealign != 0 {
        ind_pre_warp = read_check_warp_file(
            old_xfg_file.trim_end_matches(' '),
            0,
            1,
            &mut i_pre_warp_nx,
            &mut i_pre_warp_ny,
            &mut num_old_g,
            &mut i,
            &mut pixel_prewarp,
            &mut iz,
            &mut string_list,
        );
        if ind_pre_warp < -1 {
            exit_error(&string_list);
        }
        if ind_pre_warp >= 0 {
            lim_warp_z = num_old_g;
        }
    }
    if !blank(&old_xf_file) {
        ind_warp_file = read_check_warp_file(
            old_xf_file.trim_end_matches(' '),
            0,
            1,
            &mut iwarp_nx,
            &mut iwarp_ny,
            &mut num_fg_in,
            &mut i,
            &mut pixel_warp,
            &mut iz,
            &mut string_list,
        );
        if ind_warp_file < -1 {
            exit_error(&string_list);
        }
        if ind_warp_file >= 0 {
            lim_warp_z = lim_warp_z.max(num_fg_in);
        }
    }
    //
    // One or other warping exists
    if lim_warp_z > 0 {
        num_control = vec![0; (lim_warp_z * 2) as usize];
        memory_error(0, "array for number of control points");
        //
        // Need to know the pixel size of the model.  Take image pixel first.
        pixel_model = delta[0];
        if delta[0] <= 0. {
            getimodscales(&mut pixel_model, &mut x_last, &mut y_last);
        }
        //
        // Find out how big grids need to be
        if ind_pre_warp >= 0 {
            println!(
                "Warping file opened: {}",
                old_xfg_file.trim_end_matches(' ')
            );
            prewarp_scale = pixel_prewarp / pixel_model;
            set_current_warp_file(ind_pre_warp);
            if find_max_grid_size(
                xmod_min / prewarp_scale,
                xmod_max / prewarp_scale,
                y_mod_min / prewarp_scale,
                y_mod_max / prewarp_scale,
                &mut num_control[..lim_warp_z as usize],
                &mut lim_warp_x,
                &mut lim_warp_y,
                &mut string_list,
            ) != 0
            {
                exit_error(&string_list);
            }
        }
        if ind_warp_file >= 0 {
            println!("Warping file opened: {}", old_xf_file.trim_end_matches(' '));
            warp_scale = pixel_warp / pixel_model;
            set_current_warp_file(ind_warp_file);
            if find_max_grid_size(
                xmod_min / warp_scale,
                xmod_max / warp_scale,
                y_mod_min / warp_scale,
                y_mod_max / warp_scale,
                &mut num_control[lim_warp_z as usize..],
                &mut i,
                &mut iz,
                &mut string_list,
            ) != 0
            {
                exit_error(&string_list);
            }
            lim_warp_x = lim_warp_x.max(i);
            lim_warp_y = lim_warp_y.max(iz);
        }
        //
        // Allocate arrays for grid and parameters for each section
        let grid_total = (lim_warp_x * lim_warp_y * lim_warp_z) as usize;
        warp_dx = vec![0.0; grid_total];
        warp_dy = vec![0.0; grid_total];
        nx_warp = vec![0; lim_warp_z as usize];
        x_warp_start = vec![0.0; lim_warp_z as usize];
        x_warp_intrv = vec![0.0; lim_warp_z as usize];
        ny_warp = vec![0; lim_warp_z as usize];
        y_warp_start = vec![0.0; lim_warp_z as usize];
        y_warp_intrv = vec![0.0; lim_warp_z as usize];
        memory_error(0, "arrays for warping grids");
    }
    let grid_size = (lim_warp_x * lim_warp_y) as usize;
    //
    // back-transform if necessary
    if if_prealign != 0 {
        if ind_pre_warp >= 0 {
            set_current_warp_file(ind_pre_warp);
            if num_old_g > LIMNUMXF {
                exit_error("too many sections in warp file for transform array");
            }
            if if_shift_scale == 0 {
                shift_scale = prewarp_scale;
            }
            ind_control = 1;
            for i in 1..=num_old_g {
                if get_linear_transform(i - 1, &mut g_xf[i as usize - 1], 2) != 0 {
                    exit_error("Getting linear transform from warp file");
                }
                if num_control[i as usize - 1] > 2 {
                    let k = i as usize - 1;
                    if get_size_adjusted_grid(
                        i - 1,
                        (xmod_max - xmod_min) / prewarp_scale,
                        (y_mod_max - y_mod_min) / prewarp_scale,
                        ((xmod_max + xmod_min) / prewarp_scale - i_pre_warp_nx as f32) / 2.,
                        ((y_mod_max + y_mod_min) / prewarp_scale - i_pre_warp_ny as f32) / 2.,
                        0,
                        prewarp_scale,
                        1,
                        &mut nx_warp[k],
                        &mut ny_warp[k],
                        &mut x_warp_start[k],
                        &mut y_warp_start[k],
                        &mut x_warp_intrv[k],
                        &mut y_warp_intrv[k],
                        &mut warp_dx[k * grid_size..],
                        &mut warp_dy[k * grid_size..],
                        lim_warp_x,
                        lim_warp_y,
                        &mut string_list,
                    ) != 0
                    {
                        exit_error(&string_list);
                    }
                }
            }
        } else {
            let unit3 = dopen(3, old_xfg_file.trim_end_matches(' '), "ro", "f");
            //
            // get g transforms into g list
            let mut list: Vec<[f32; 6]> = Vec::new();
            ierr = xfrdall2(&mut BufReader::new(unit3), &mut list, LIMNUMXF);
            num_old_g = list.len() as i32;
            g_xf[..list.len()].copy_from_slice(&list);
            if ierr != 0 {
                exit_error("Too many transforms for arrays");
            }
            if num_old_g == 0 {
                exit_error(&format!(
                    "The file of prealign transforms is empty: {}",
                    old_xfg_file.trim_end_matches(' ')
                ));
            }
            // `close(3)` is the drop of the reader.
        }
        //
        // invert the g's into the g list
        for ind_g_xf in 1..=num_old_g {
            let g = &mut g_xf[ind_g_xf as usize - 1];
            g[4] *= shift_scale;
            g[5] *= shift_scale;
            xf_invert(g, &mut g_temp, 2);
            xf_copy(&g_temp, 2, g, 2);
        }
        //
        // apply inverse g's to all points in model
        // Set up to use a single section if back transforming and user
        // specified it or there is only one transform
        //
        line_to_use = -1;
        if ifxfmod < 0 {
            line_to_use = line_use;
            if num_old_g == 1 && num_chunks == 0 {
                line_to_use = 0;
                println!(" There is only one transform and it is being applied at all Z values");
            }
        }
        println!(
            "Back-transforming model with inverse of transforms from {}",
            old_xfg_file.trim_end_matches(' ')
        );
        if ind_pre_warp >= 0 {
            warp_model(
                num_old_g,
                1,
                &mut num_undefined,
                &mut fm,
                &list_z,
                line_to_use,
                &ind_zto_flist,
                &num_control,
                ind_control,
                lim_warp_z,
                &warp_dx,
                &warp_dy,
                lim_warp_x,
                lim_warp_y,
                &nx_warp,
                &ny_warp,
                &x_warp_start,
                &y_warp_start,
                &x_warp_intrv,
                &y_warp_intrv,
            );
        }
        transform_model(
            &g_xf,
            num_old_g,
            LIMNUMXF,
            x_cen,
            y_cen,
            &ind_zto_flist,
            &list_z,
            line_to_use,
            &mut num_undefined,
            if_adjust_for90,
            &mut fm,
        );

        if ifxfmod < 0 && if_distort + if_mag_grad == 0 {
            //
            // write out back-transformed model
            rescale_write_model(&new_model, num_undefined, &mut fm);
            exit(0);
        }
    }
    //
    // read in the old f's or g's for whatever purpose
    //
    if ind_warp_file < 0 {
        num_fg_in = 0;
    }
    line_to_use = -1;
    if !blank(&old_xf_file) {
        if ind_warp_file >= 0 {
            set_current_warp_file(ind_warp_file);
            if num_fg_in > LIMNUMXF {
                exit_error("Too many sections in warp file for transform array");
            }
            if if_shift_scale == 0 {
                shift_scale = warp_scale;
            }
            ind_control = 2;
            for i in 1..=num_fg_in {
                if get_linear_transform(i - 1, &mut f_xf[i as usize - 1], 2) != 0 {
                    exit_error("Getting linear transform from warp file");
                }
                if num_control[(lim_warp_z + i) as usize - 1] > 2 {
                    let k = i as usize - 1;
                    if get_size_adjusted_grid(
                        i - 1,
                        (xmod_max - xmod_min) / warp_scale,
                        (y_mod_max - y_mod_min) / warp_scale,
                        ((xmod_max + xmod_min) / warp_scale - iwarp_nx as f32) / 2.,
                        ((y_mod_max + y_mod_min) / warp_scale - iwarp_ny as f32) / 2.,
                        0,
                        warp_scale,
                        1,
                        &mut nx_warp[k],
                        &mut ny_warp[k],
                        &mut x_warp_start[k],
                        &mut y_warp_start[k],
                        &mut x_warp_intrv[k],
                        &mut y_warp_intrv[k],
                        &mut warp_dx[k * grid_size..],
                        &mut warp_dy[k * grid_size..],
                        lim_warp_x,
                        lim_warp_y,
                        &mut string_list,
                    ) != 0
                    {
                        exit_error(&string_list);
                    }
                }
            }
        } else {
            let unit1 = dopen(1, old_xf_file.trim_end_matches(' '), "ro", "f");
            let mut list: Vec<[f32; 6]> = Vec::new();
            ierr = xfrdall2(&mut BufReader::new(unit1), &mut list, LIMNUMXF);
            num_fg_in = list.len() as i32;
            f_xf[..list.len()].copy_from_slice(&list);
            if ierr != 0 {
                exit_error("Too many transforms for arrays");
            }
            if num_fg_in == 0 {
                exit_error(&format!(
                    "The input file of transforms is empty: {}",
                    old_xf_file.trim_end_matches(' ')
                ));
            }
            // `close(1)` is the drop of the reader.
        }
        for ind_g_xf in 1..=num_fg_in {
            f_xf[ind_g_xf as usize - 1][4] *= shift_scale;
            f_xf[ind_g_xf as usize - 1][5] *= shift_scale;
        }
        line_to_use = line_use;
        if num_fg_in == 1 && num_chunks == 0 {
            line_to_use = 0;
            println!(" There is only one transform and it is being applied at all Z values");
        }
    }
    //
    // TRANSFORMING/UNDISTORTING MODEL
    //
    if ifxfmod != 0 {
        if if_distort + if_mag_grad != 0 {
            if if_back != 0 {
                println!(" Redistorting model");
            } else {
                println!(" Undistorting model");
            }

            for i in 1..=fm.n_point {
                let ip = i as usize - 1;
                if if_mag_grad != 0 {
                    iz = 1.max(((fm.p_coord[ip][2] + 1.).round() as i32).min(num_mag_grad));
                }

                if if_back == 0 {
                    //
                    // undistort the model - find point that distorts to the
                    // given model point
                    //
                    iter = 1;
                    x_last = fm.p_coord[ip][0];
                    y_last = fm.p_coord[ip][1];
                    done = false;
                    while iter < 10 && !done {
                        dx1 = 0.;
                        dy1 = 0.;
                        dx = 0.;
                        dy = 0.;
                        if if_mag_grad != 0 {
                            let izu = iz as usize - 1;
                            mag_gradient_shift(
                                x_last,
                                y_last,
                                (2. * x_cen).round() as i32,
                                (2. * y_cen).round() as i32,
                                x_cen,
                                y_cen,
                                pixel_mag_grad,
                                axis_rot,
                                tilt_angles[izu],
                                dmag_per_um[izu],
                                rot_per_um[izu],
                                &mut dx1,
                                &mut dy1,
                            );
                        }

                        if if_distort != 0 {
                            interpolate_grid(
                                x_last + dx1,
                                y_last + dy1,
                                &field_dx,
                                &field_dy,
                                lim_grid,
                                nx_grid,
                                ny_grid,
                                x_grid_strt,
                                y_grid_strt,
                                x_grid_intrv,
                                y_grid_intrv,
                                &mut dx,
                                &mut dy,
                            );
                        }
                        x_new = fm.p_coord[ip][0] - (dx + dx1);
                        y_new = fm.p_coord[ip][1] - (dy + dy1);
                        done = (x_new - x_last).abs() < 0.01 && (y_new - y_last).abs() < 0.01;
                        x_last = x_new;
                        y_last = y_new;
                        iter += 1;
                    }
                    fm.p_coord[ip][0] = x_new;
                    fm.p_coord[ip][1] = y_new;
                } else {
                    //
                    // or redistort the model
                    //
                    dx1 = 0.;
                    dy1 = 0.;
                    dx = 0.;
                    dy = 0.;
                    if if_mag_grad != 0 {
                        let izu = iz as usize - 1;
                        mag_gradient_shift(
                            fm.p_coord[ip][0],
                            fm.p_coord[ip][1],
                            (2. * x_cen).round() as i32,
                            (2. * y_cen).round() as i32,
                            x_cen,
                            y_cen,
                            pixel_mag_grad,
                            axis_rot,
                            tilt_angles[izu],
                            dmag_per_um[izu],
                            rot_per_um[izu],
                            &mut dx1,
                            &mut dy1,
                        );
                    }

                    if if_distort != 0 {
                        interpolate_grid(
                            fm.p_coord[ip][0] + dx1,
                            fm.p_coord[ip][1] + dy1,
                            &field_dx,
                            &field_dy,
                            lim_grid,
                            nx_grid,
                            ny_grid,
                            x_grid_strt,
                            y_grid_strt,
                            x_grid_intrv,
                            y_grid_intrv,
                            &mut dx,
                            &mut dy,
                        );
                    }
                    fm.p_coord[ip][0] = fm.p_coord[ip][0] + dx1 + dx;
                    fm.p_coord[ip][1] = fm.p_coord[ip][1] + dy1 + dy;
                }
            }
            //
            // if there was prealignment and no new transforms, get the
            // prealignment transforms back by inversion and set up to use them
            //
            if num_fg_in == 0 && if_prealign != 0 && ifxfmod > 0 {
                println!(
                    "Transforming model with reinverted transforms from {}",
                    old_xfg_file.trim_end_matches(' ')
                );
                for ind_g_xf in 1..=num_old_g {
                    xf_invert(
                        &g_xf[ind_g_xf as usize - 1],
                        &mut f_xf[ind_g_xf as usize - 1],
                        2,
                    );
                }
                num_fg_in = num_old_g;
                ind_warp_file = ind_pre_warp;
            }
        }
        //
        num_undefined = 0;
        //
        // transform the model
        //
        if !blank(&old_xf_file) {
            println!(
                "Transforming model with transforms from {}",
                old_xf_file.trim_end_matches(' ')
            );
        }
        if num_fg_in != 0 {
            transform_model(
                &f_xf,
                num_fg_in,
                LIMNUMXF,
                x_cen,
                y_cen,
                &ind_zto_flist,
                &list_z,
                line_to_use,
                &mut num_undefined,
                if_adjust_for90,
                &mut fm,
            );
        }
        if ind_warp_file >= 0 {
            warp_model(
                num_fg_in,
                0,
                &mut num_undefined,
                &mut fm,
                &list_z,
                line_to_use,
                &ind_zto_flist,
                &num_control,
                ind_control,
                lim_warp_z,
                &warp_dx,
                &warp_dy,
                lim_warp_x,
                lim_warp_y,
                &nx_warp,
                &ny_warp,
                &x_warp_start,
                &y_warp_start,
                &x_warp_intrv,
                &y_warp_intrv,
            );
        }

        rescale_write_model(&new_model, num_undefined, &mut fm);
    } else {
        //
        // SEARCH FOR POINTS TO DERIVE XFORMS FROM
        //
        // allocate xr, limiting it to much more than could be needed for smaller model
        idim = idim.min(fm.n_point);
        xmat = vec![0.0; (MSIZE_XMAT * idim.max(0)) as usize];
        memory_error(0, "array for data matrix");
        let ms = MSIZE_XMAT as usize;
        // `xmat(c, p)` with the column index fastest.
        let xm = |c: i32, p: i32| (p as usize - 1) * ms + c as usize - 1;

        // first find min and max z in model
        iz_min = 100000;
        iz_max = -iz_min;
        for iobj in 1..=fm.max_mod_obj {
            ibase = fm.ibase_obj[iobj as usize - 1];
            for ipt in 1..=fm.npt_in_obj[iobj as usize - 1] {
                i = fm.object[(ipt + ibase) as usize - 1].wrapping_abs();
                zz = fm.p_coord[i as usize - 1][2];
                // zdex=(zz+zorig) /delt(3)
                z_index = zz;
                iz =
                    (z_index - z_index.round() as i32 as f32 + 0.5) as i32 + z_index.round() as i32;
                // if (iz<listz(1) .or.iz>listz(nlistz)) goto 93
                iz_min = iz_min.min(iz);
                iz_max = iz_max.max(iz);
            }
        }
        //
        // if didn't specify list of section #'s, make such a list from range
        //
        if num_to_find <= 0 {
            num_to_find = if_single + iz_max - iz_min;
            for i in 1..=num_to_find {
                num_sec[i as usize - 1] = iz_min + i - if_single;
            }
        }
        num_fout = num_fout.max(num_fg_in);
        // `write(*,122)`
        println!("{:40}Deviations between transformed points on", "");
        println!("{:41}section and points on previous section", "");
        println!("{:32}Mean     Max  @object & point #   X-Y Position", "");
        for loop_ in 1..=num_to_find {
            iz_sec = num_sec[loop_ as usize - 1];
            //
            // make sure this is a section in list and find previous z
            //
            last_sec = -100000;
            for il in 2..=num_list_z {
                if iz_sec == list_z[il as usize - 1] {
                    last_sec = list_z[il as usize - 2];
                }
            }
            //
            // but if doing to single section, allow section to be first in list
            // as well and set lastsec to z of single section
            //
            if if_single != 0 && (last_sec != -100000 || iz_sec == list_z[0]) {
                last_sec = iz_single;
            }
            if last_sec != -100000 {
                //
                // get points out of objects with points in both this and previous
                // section
                //
                num_points = 0;
                for iobject in 1..=fm.max_mod_obj {
                    got_last = false;
                    got_this = false;
                    num_in_obj = fm.npt_in_obj[iobject as usize - 1];
                    for ind_in_obj in 1..=num_in_obj {
                        ipnt = fm.object
                            [(ind_in_obj + fm.ibase_obj[iobject as usize - 1]) as usize - 1];
                        if ipnt > 0 && ipnt <= fm.n_point {
                            let p = fm.p_coord[ipnt as usize - 1];
                            // zdex=(p_coord(3, ipnt) +zorig) /delt(3)
                            z_index = p[2];
                            iz = (z_index - z_index.round() as i32 as f32 + 0.5) as i32
                                + z_index.round() as i32;
                            if iz == iz_sec && (!got_this || p[2] < z_this) {
                                //
                                // if in second section, and either haven't gotten a point
                                // there before, or this point has a lower z than the
                                // previous point, put x and y into 1st and 2nd
                                // column: independent vars
                                //
                                xmat[xm(1, num_points + 1)] = p[0] - x_cen;
                                xmat[xm(2, num_points + 1)] = p[1] - y_cen;
                                got_this = true;
                                z_this = p[2];
                                xmat[xm(6, num_points + 1)] = iobject as f32;
                                xmat[xm(7, num_points + 1)] = ind_in_obj as f32;
                            } else if iz == last_sec && (!got_last || p[2] > z_last) {
                                //
                                // if in first section, and either haven't gotten a point
                                // there before, or this point is higher in z than the
                                // previous point, put x and y into 4th and 5th
                                // column: dependent vars
                                //
                                xmat[xm(4, num_points + 1)] = p[0] - x_cen;
                                xmat[xm(5, num_points + 1)] = p[1] - y_cen;
                                got_last = true;
                            }
                        }
                    }
                    if got_this && got_last {
                        num_points += 1;
                    }
                    if num_points >= idim {
                        println!();
                        println!(
                            " ERROR: XFMODEL - too many points for arrays on section {:11}",
                            iz
                        );
                        exit(1);
                    }
                } //done with looking at objects
                if num_points >= lim_points {
                    //
                    // now if there are at least limpnts points, do regressions:
                    // first last section x then l.s. y as function of this section
                    // x and y
                    // save results in xform for izsec
                    //
                    ind_f_xf = ind_zto_flist[(iz_sec + 1 - list_z[0]) as usize - 1];
                    if ind_f_xf <= 0 {
                        println!();
                        println!(
                            " ERROR: XFMODEL - z value out of range for transforms: {:>15.8}    ",
                            zz
                        );
                        exit(1);
                    }
                    //
                    // `findxf` (`findtransform.c:225`)
                    if find_transform(
                        &mut xmat,
                        MSIZE_XMAT,
                        4,
                        num_points,
                        x_cen,
                        y_cen,
                        if_trans,
                        if_rot_trans,
                        2,
                        &mut f_xf[ind_f_xf as usize - 1],
                        &mut dev_avg,
                        &mut dev_sd,
                        &mut devmax,
                        &mut ipnt_max,
                    ) != 0
                    {
                        println!("ERROR: Findxf function - Allocating array for matrices");
                        exit(1);
                    }
                    //
                    // keep track of the highest & lowest transforms obtained
                    //
                    num_fout = num_fout.max(ind_f_xf);
                    // `write(*,121)`
                    println!(
                        "{} points, section #{}  {}{}{}{}    {}{}",
                        fmt_i(num_points, 4),
                        fmt_i(iz_sec, 4),
                        fmt_f(dev_avg, 9, 2),
                        fmt_f(devmax, 9, 2),
                        fmt_i(xmat[xm(6, ipnt_max)].round() as i32, 6),
                        fmt_i(xmat[xm(7, ipnt_max)].round() as i32, 6),
                        fmt_f(xmat[xm(8, ipnt_max)], 9, 2),
                        fmt_f(xmat[xm(9, ipnt_max)], 9, 2)
                    );
                    if if_full_report != 0
                        && (devmax >= report_max_crit || dev_avg > report_mean_crit)
                    {
                        // `write(*,124) ((xmat(i, j), i = 6, 13), j = 1, numPoints)`
                        println!(
                            "    Object  Point       position        deviation vector   angle   magnitude"
                        );
                        for j in 1..=num_points {
                            println!(
                                "{}{}{}{}{}{}{}{}",
                                fmt_f(xmat[xm(6, j)], 10, 0),
                                fmt_f(xmat[xm(7, j)], 6, 0),
                                fmt_f(xmat[xm(8, j)], 10, 2),
                                fmt_f(xmat[xm(9, j)], 10, 2),
                                fmt_f(xmat[xm(10, j)], 10, 2),
                                fmt_f(xmat[xm(11, j)], 10, 2),
                                fmt_f(xmat[xm(12, j)], 9, 0),
                                fmt_f(xmat[xm(13, j)], 10, 2)
                            );
                        }
                    }
                } else {
                    println!(
                        " less than {:11}  points for section #  {:11}",
                        lim_points, iz_sec
                    );
                }
            }
        } //end of loop the loop
        //
        // write out enough for whole file, and at least as many as were in
        // an input file if any
        //
        let mut unit2 = BufWriter::new(dopen(2, new_xf_file.trim_end_matches(' '), "new", "f"));
        for i in 1..=num_fout {
            if xfwrite(&mut unit2, &f_xf[i as usize - 1]).is_err() {
                // `*94`
                exit_error("Writing out f file");
            }
        }
        // `close(2)`
        if unit2.flush().is_err() {
            exit_error("Writing out f file");
        }
    }
    exit(0);
    // 91 call exitError('Reading model file') -- reached by `goto 91` above.
    // 92 call exitError('Reading old f/g file') -- no `goto 92` in the source.
    // 94 call exitError('Writing out f file') -- the `*94` returns above.
}

/// Original contained subroutine `warpModel` (`xfmodel.f90:965`).
///
/// Host variables: `numUndefined` (written), the `fortmodel` arrays,
/// `listZ`, `lineToUse`, `indZtoFlist`, `numControl(limWarpZ, 2)` with
/// `indControl`, and the per-section warping grids.
#[allow(clippy::too_many_arguments)]
pub fn warp_model(
    num_sec_in: i32,
    if_back: i32,
    num_undefined: &mut i32,
    fm: &mut FortModel,
    list_z: &[i32],
    line_to_use: i32,
    ind_zto_flist: &[i32],
    num_control: &[i32],
    ind_control: i32,
    lim_warp_z: i32,
    warp_dx: &[f32],
    warp_dy: &[f32],
    lim_warp_x: i32,
    lim_warp_y: i32,
    nx_warp: &[i32],
    ny_warp: &[i32],
    x_warp_start: &[f32],
    y_warp_start: &[f32],
    x_warp_intrv: &[f32],
    y_warp_intrv: &[f32],
) {
    let mut i: i32;
    let mut iz: i32;
    let mut ind_to_ind: i32;
    let mut ind_g_xf: i32;
    let mut zz: f32;
    let (mut dx, mut dy) = (0.0_f32, 0.0_f32);
    let grid_size = (lim_warp_x * lim_warp_y) as usize;
    //
    *num_undefined = 0;
    for iobj in 1..=fm.max_mod_obj {
        for ipt in 1..=fm.npt_in_obj[iobj as usize - 1] {
            i = fm.object[(fm.ibase_obj[iobj as usize - 1] + ipt) as usize - 1].wrapping_abs();
            let ip = i as usize - 1;
            zz = fm.p_coord[ip][2];
            iz = (zz - zz.round() as i32 as f32 + 0.5) as i32 + zz.round() as i32;
            ind_to_ind = iz + 1 - list_z[0];
            if !((ind_to_ind < 1 || ind_to_ind > LIMNUMXF) && line_to_use < 0) {
                if line_to_use < 0 {
                    ind_g_xf = ind_zto_flist[ind_to_ind as usize - 1];
                } else {
                    ind_g_xf = line_to_use + 1;
                }
                if ind_g_xf >= 1 && ind_g_xf <= num_sec_in {
                    let k = ind_g_xf as usize - 1;
                    if num_control[((ind_control - 1) * lim_warp_z) as usize + k] > 2 {
                        if if_back != 0 {
                            interpolate_grid(
                                fm.p_coord[ip][0],
                                fm.p_coord[ip][1],
                                &warp_dx[k * grid_size..],
                                &warp_dy[k * grid_size..],
                                lim_warp_x,
                                nx_warp[k],
                                ny_warp[k],
                                x_warp_start[k],
                                y_warp_start[k],
                                x_warp_intrv[k],
                                y_warp_intrv[k],
                                &mut dx,
                                &mut dy,
                            );
                            fm.p_coord[ip][0] += dx;
                            fm.p_coord[ip][1] += dy;
                        } else {
                            let (mut xnew, mut ynew) = (0.0_f32, 0.0_f32);
                            find_inverse_point(
                                fm.p_coord[ip][0],
                                fm.p_coord[ip][1],
                                &warp_dx[k * grid_size..],
                                &warp_dy[k * grid_size..],
                                lim_warp_x,
                                nx_warp[k],
                                ny_warp[k],
                                x_warp_start[k],
                                y_warp_start[k],
                                x_warp_intrv[k],
                                y_warp_intrv[k],
                                &mut xnew,
                                &mut ynew,
                                &mut dx,
                                &mut dy,
                            );
                            fm.p_coord[ip][0] = xnew;
                            fm.p_coord[ip][1] = ynew;
                        }
                    }
                }
            }
        }
    }
}

/// Original subroutine `transformModel` (`xfmodel.f90:1010`).
#[allow(clippy::too_many_arguments)]
pub fn transform_model(
    f_xf: &[[f32; 6]],
    num_fg_in: i32,
    limnumxf: i32,
    x_cen: f32,
    y_cen: f32,
    ind_zto_flist: &[i32],
    list_z: &[i32],
    line_to_use: i32,
    num_undefined: &mut i32,
    if_adjust_for90: i32,
    fm: &mut FortModel,
) {
    let mut ftmp = [0.0_f32; 6];
    let mut i: i32;
    let mut iz: i32;
    let mut ind_to_ind: i32;
    let mut ind_g_xf: i32;
    let mut iscan: i32;
    let mut zz: f32;
    let mut transpose_xy: bool;
    //
    // Assess need to transpose sizes as in newstack
    transpose_xy = if_adjust_for90 > 1;
    if if_adjust_for90 == 1 {
        iscan = 0;
        for i in 1..=num_fg_in {
            xf_copy(&f_xf[i as usize - 1], 2, &mut ftmp, 2);
            ftmp[4] = 0.;
            ftmp[5] = 0.;
            let (tmp_min, tmp_max) = xf_apply(&ftmp, 0., 0., 2., -1., 2);
            let (tmp_min2, tmp_max2) = xf_apply(&ftmp, 0., 0., 2., 1., 2);
            let a = tmp_max.abs();
            let b = tmp_max2.abs();
            let c = tmp_min.abs();
            let d = tmp_min2.abs();
            if (if a > b { a } else { b }) > (if c > d { c } else { d }) {
                iscan += 1;
            }
        }
        transpose_xy = iscan == num_fg_in;
    }

    *num_undefined = 0;
    for iobj in 1..=fm.max_mod_obj {
        for ipt in 1..=fm.npt_in_obj[iobj as usize - 1] {
            i = fm.object[(fm.ibase_obj[iobj as usize - 1] + ipt) as usize - 1].wrapping_abs();
            let ip = i as usize - 1;
            zz = fm.p_coord[ip][2];
            iz = (zz - zz.round() as i32 as f32 + 0.5) as i32 + zz.round() as i32;
            ind_to_ind = iz + 1 - list_z[0];
            if (ind_to_ind < 1 || ind_to_ind > limnumxf) && line_to_use < 0 {
                *num_undefined += 1;
            } else {
                if line_to_use < 0 {
                    ind_g_xf = ind_zto_flist[ind_to_ind as usize - 1];
                } else {
                    ind_g_xf = line_to_use + 1;
                }
                if ind_g_xf < 1 || ind_g_xf > num_fg_in {
                    *num_undefined += 1;
                } else {
                    let (xp, yp) = xf_apply(
                        &f_xf[ind_g_xf as usize - 1],
                        x_cen,
                        y_cen,
                        fm.p_coord[ip][0],
                        fm.p_coord[ip][1],
                        2,
                    );
                    fm.p_coord[ip][0] = xp;
                    fm.p_coord[ip][1] = yp;

                    // Adjust by difference in centers
                    if transpose_xy {
                        fm.p_coord[ip][0] = fm.p_coord[ip][0] + y_cen - x_cen;
                        fm.p_coord[ip][1] = fm.p_coord[ip][1] + x_cen - y_cen;
                    }
                }
            }
        }
    }
}

/// Original subroutine `rescaleWriteModel` (`xfmodel.f90:1070`).
pub fn rescale_write_model(new_model: &str, num_undefined: i32, fm: &mut FortModel) {
    //
    if num_undefined > 0 {
        println!(
            " {:11}  points with Z values out of range of transforms",
            num_undefined
        );
    }
    //
    // write model out
    // shift the data back for saving
    //
    scale_model(1, fm);
    write_wmod(new_model.trim_end_matches(' '), fm);
}
