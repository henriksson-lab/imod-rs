//! Translation of `IMOD/flib/model/boxstartend.f90`.
//!
//! BOXSTARTEND will clip out volumes of image centered on the starting or
//! ending points of model contours, or on all points in a model contour.  It
//! can place each volume into a separate file, or stack all the extracted
//! volumes into a single output file.  In the latter case, the program can
//! generate two lists of piece coordinates to allow the volumes to be
//! examined in two different ways.  David Mastronarde 4/23/90.
//!
//! The main program maps to [`boxstartend`] and its internal subroutine
//! `fillIfNeeded` to [`fill_if_needed`], which takes the host variables it
//! uses as arguments.  `array`, `brray` and
//! `avgArray` are EQUIVALENCEd to one block (`common /bigarr/`), so here they
//! are one vector, `bigarr`, addressed with the source's 1-based indices
//! (`array(k)` is `bigarr[k - 1]`).  The `fortmodel` module arrays are the
//! [`FortModel`] that `readw_or_imod` fills.

use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::{format_f, maxss, minss};
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::rdlist::{parselist, rdlist};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdpas, irdsec};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::{get_model_object_list, readw_or_imod};
use crate::imod::flib::subrs::model::scale_model::scale_model_to_image;
use crate::imod::flib::subrs::piecesubs::read_piece_list::read_piece_list;
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall;
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libcfshr::linearxforms::{xfapply, xfcopy, xfinvert};
use crate::imod::libcfshr::parse_params::{
    pip_get_integer, pip_get_three_integers, pip_get_two_integers,
};
use crate::imod::libcfshr::piecefuncs::{checklist, fill_listz};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use crate::imod::libcfshr::taperatfill::taperatfill;
use crate::imod::libiimod::mrcfiles::MRC_LABEL_SIZE;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position, iiu_write_section};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_origin, iiu_alt_sample, iiu_alt_size, iiu_ret_delta, iiu_ret_origin,
    iiu_trans_header, iiu_write_header,
};
use crate::imod::libimod::imodel_fwrap::{getimodflags, getimodobjsize, imodpartialmode};
use std::io::{BufRead, BufReader, Write};

/// `parameter (IXDIM = 500, ...)` (`boxstartend.f90:23`).
const IXDIM: i64 = 500;
/// `IDIM = IXDIM * IXDIM * IXDIM`.
const IDIM: i64 = IXDIM * IXDIM * IXDIM;
/// `LIMPCL = 50000`.
const LIMPCL: usize = 50000;
/// `LIMOBJ = 10000`.
const LIMOBJ: usize = 10000;
/// `parameter (numOptions = 19)` (`boxstartend.f90:73`).
const BOXSTARTEND_NUM_OPTIONS: i32 = 19;
/// Fallback PIP table, the `options(1)` string (`boxstartend.f90:75-84`).
const BOXSTARTEND_OPTIONS: &str = "image:InputImageFile:FN:@model:ModelFile:FN:@\
output:OutputFile:FN:@series:SeriesRootName:CH:@\
piece:PieceListFile:FN:@array:ArrayPieceList:FN:@\
true:TruePieceList:FN:@objects:ObjectsToUse:LI:@box:BoxSizeXY:I:@\
slices:SlicesBelowAndAbove:IP:@which:WhichPointsToExtract:I:@\
xminmax:XMinAndMax:IP:@yminmax:YMinAndMax:IP:@\
zminmax:ZMinAndMax:IP:@xforms:XformsToApply:FN:@\
back:BackTransform:B:@blank:BlankBetweenImages:I:@\
param:ParameterFile:PF:@help:usage:B:";

/// Internal subroutine `fillIfNeeded(fillVal)` (`boxstartend.f90`, after
/// `contains`): fills the box (`brray(1:numPixSquare)`, here `bigarr` from
/// `ib_base + 1`) once.  Fixed in translation (BUGS.md, `boxstartend`): the
/// source fills with `dmean`, ignoring its argument; defined: `fill_val`.
fn fill_if_needed(
    bigarr: &mut [f32],
    need_fill: &mut bool,
    fill_val: f32,
    num_pix_sq: i32,
    ib_base: i64,
) {
    if *need_fill {
        for iy in 1..=num_pix_sq as i64 {
            bigarr[(iy + ib_base - 1) as usize] = fill_val;
        }
    }
    *need_fill = false;
}

/// Original program `boxstartend` (`boxstartend.f90:20`).
pub fn boxstartend() {
    let mut fm = FortModel::default();
    // `common /bigarr/ avgArray`, with `array` and `brray` EQUIVALENCEd to it.
    let mut bigarr = vec![0.0_f32; IDIM as usize];
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut nxyz2 = [0_i32; 3];
    // `data nxyzst/0, 0, 0/`
    let nxyzst = [0_i32; 3];
    let mut ind = [0_i32; 3];
    let mut model_file = String::new();
    let mut cell = [0.0_f32; 6];
    let mut offset = [0.0_f32; 3];
    let mut ix_pc_list = vec![0_i32; LIMPCL];
    let mut iy_pc_list = vec![0_i32; LIMPCL];
    let mut iz_pc_list = vec![0_i32; LIMPCL];
    let mut listz = vec![0_i32; LIMPCL];
    let mut iobj_clip = vec![0_i32; LIMOBJ];
    let mut iobj_flags = vec![0_i32; LIMOBJ];
    let mut load_objs = vec![0_i32; LIMOBJ];
    let mut in_file = String::new();
    let mut piece_file = [b' '; 320];
    let mut piece_file2 = [b' '; 320];
    let mut root_name = [b' '; 320];
    let mut list_string = [b' '; 10240];
    let mut date_strn = [b' '; 9];
    let mut time_strn = [b' '; 8];
    let mut g: Vec<[f32; 6]> = Vec::new();
    let mut gtmp = [0.0_f32; 6];
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut mode = 0_i32;
    let mut num_pc_list = 0_i32;
    let (mut min_xpiece, mut num_xpieces, mut nx_overlap) = (0_i32, 0_i32, 0_i32);
    let (mut min_ypiece, mut num_ypieces, mut ny_overlap) = (0_i32, 0_i32, 0_i32);
    let mut if_xform: i32;
    let mut num_obj_clip: i32;
    let mut if_start_end: i32;
    let mut num_pix_box = 0_i32;
    let mut ny_pix_box = 0_i32;
    let mut nz_pix_box = 0_i32;
    let mut num_gutter: i32;
    let mut back_xform: bool;
    let mut num_taper: i32;
    let mut inside_taper = 0_i32;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    let runtime_abort = |err: ListReadError| -> ! {
        let _ = std::io::stdout().flush();
        match err {
            ListReadError::End => eprintln!("Fortran runtime error: End of file"),
            ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
        }
        exit(2);
    };
    // `read(*,*) ...` from standard input
    let read_stdin_list = |items: &mut [ListItem]| {
        let mut stdin = std::io::stdin().lock();
        if let Err(err) = list_read(&mut stdin, items) {
            runtime_abort(err);
        }
    };
    // `read(5, 50) name`, `50 format(A)`, into a `character*320`
    let read_name = |field: &mut [u8; 320]| {
        let mut line = Vec::new();
        match std::io::stdin().lock().read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => runtime_abort(ListReadError::End),
            Ok(_) => {}
        }
        if line.last() == Some(&b'\n') {
            line.pop();
        }
        line.resize(320, b' ');
        field.copy_from_slice(&line[..320]);
    };
    let prompt = |text: &str| {
        print!(" {text}");
        let _ = std::io::stdout().flush();
    };
    let blank = |field: &[u8]| field.iter().all(|&c| c == b' ');
    // `NN.ixyz`, a Fortran `Iw.m`
    let fmt_im = |value: i32, w: usize| -> String { format!("{:0>w$}", value) };
    //
    // defaults
    if_start_end = -1;
    num_gutter = 5;
    back_xform = false;
    if_xform = 0;
    num_taper = 0;
    fm.fm_mod_size_type = 2;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    pip_read_or_parse_options(
        &[BOXSTARTEND_OPTIONS],
        BOXSTARTEND_NUM_OPTIONS,
        "boxstartend",
        "ERROR: BOXSTARTEND - ",
        true,
        4,
        2,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    let pip_input = num_opt_arg + num_non_opt_arg > 0;

    if pip_get_in_out_file(
        "InputImageFile",
        1,
        "Enter input image file name",
        &mut in_file,
        320,
    ) != 0
    {
        exit_error("No input image file specified");
    }

    imopen(1, &in_file, "RO");
    unsafe {
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
    }
    let delta = iiu_ret_delta(1);
    let origin = iiu_ret_origin(1);
    let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);
    //
    // Open other files
    //
    piece_file.fill(b' ');
    if pip_input {
        let _ = pipgetstring_(b"PieceListFile", &mut piece_file);
    } else {
        prompt("Piece list file if image is a montage, otherwise Return: ");
        read_name(&mut piece_file);
    }
    read_piece_list(
        &fortran_string(&piece_file),
        &mut ix_pc_list,
        &mut iy_pc_list,
        &mut iz_pc_list,
        &mut num_pc_list,
    );
    //
    // if no pieces, set up mocklist
    if num_pc_list == 0 {
        for i in 1..=nz {
            ix_pc_list[(i - 1) as usize] = 0;
            iy_pc_list[(i - 1) as usize] = 0;
            iz_pc_list[(i - 1) as usize] = i - 1;
        }
        num_pc_list = nz;
    }
    // make ordered list of z values
    let num_listz = {
        let mut count = 0usize;
        fill_listz(&iz_pc_list[..num_pc_list as usize], &mut listz, &mut count);
        count as i32
    };

    let _iz_range = listz[(num_listz - 1) as usize] + 1 - listz[0];
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
    let x_cen = min_xpiece as f32 + (nx + (num_xpieces - 1) * (nx - nx_overlap)) as f32 / 2.;
    let y_cen = min_ypiece as f32 + (ny + (num_ypieces - 1) * (ny - ny_overlap)) as f32 / 2.;
    let max_xpiece = min_xpiece + (nx + (num_xpieces - 1) * (nx - nx_overlap)) - 1;
    let max_ypiece = min_ypiece + (ny + (num_ypieces - 1) * (ny - ny_overlap)) - 1;
    //
    // A list of objects will be loaded, do not set fmMaxObjLoaded and it will make space
    // for all
    imodpartialmode(1);

    if pip_get_in_out_file(
        "ModelFile",
        2,
        "Name of input model file",
        &mut model_file,
        320,
    ) != 0
    {
        exit_error("No input model file specified");
    }
    let exist = readw_or_imod(model_file.trim_end_matches(' '), &mut fm);
    if !exist {
        exit_error("Opening model file");
    }

    let num_obj_tot = getimodobjsize();
    if num_obj_tot > LIMOBJ as i32 {
        exit_error("Too many model objects for arrays");
    }
    //
    if pip_input {
        if_xform = 1 - pipgetstring_(b"XformsToApply", &mut piece_file);
        let _ = pip_get_logical("BackTransform", &mut back_xform);
        if if_xform != 0 && back_xform {
            if_xform = -1;
        }
    } else {
        prompt("0 to take coordinates as they are, 1 to transform, -1 to back-transform: ");
        read_stdin_list(&mut [ListItem::Integer(&mut if_xform)]);
        if if_xform != 0 {
            prompt("Name of file with transforms: ");
            read_name(&mut piece_file);
        }
    }

    if if_xform != 0 {
        let mut unit4 = BufReader::new(dopen(4, &fortran_string(&piece_file), "old", "f"));
        if xfrdall(&mut unit4, &mut g).is_err() {
            // 99
            exit_error("Reading file");
        }
        if if_xform < 0 {
            for transform in g.iter_mut() {
                xfinvert(transform, &mut gtmp);
                xfcopy(&gtmp, transform);
            }
        }
    }
    //
    let mut ind_xmin = min_xpiece;
    let mut ind_xmax = max_xpiece;
    let mut ind_ymin = min_ypiece;
    let mut ind_ymax = max_ypiece;
    let mut ind_zmin = -99999_i32;
    let mut ind_zmax = -99999_i32;
    num_obj_clip = 0;
    if pip_input {
        let _ = pip_get_two_integers(b"XMinAndMax", &mut ind_xmin, &mut ind_xmax);
        let _ = pip_get_two_integers(b"YMinAndMax", &mut ind_ymin, &mut ind_ymax);
        let _ = pip_get_two_integers(b"ZMinAndMax", &mut ind_zmin, &mut ind_zmax);
        if pipgetstring_(b"ObjectsToUse", &mut list_string) == 0 {
            let _ = parselist(
                &fortran_string(&list_string),
                &mut iobj_clip,
                &mut num_obj_clip,
            );
        }
    } else {
        prompt(
            "Enter minimum and maximum X and Y index coordinates within which\n    ends should be contained, or / for no limits: ",
        );
        read_stdin_list(&mut [
            ListItem::Integer(&mut ind_xmin),
            ListItem::Integer(&mut ind_xmax),
            ListItem::Integer(&mut ind_ymin),
            ListItem::Integer(&mut ind_ymax),
        ]);
        prompt(
            "Enter minimum and maximum section numbers within which\n    boxes should be contained, or / for no limits: ",
        );
        read_stdin_list(&mut [
            ListItem::Integer(&mut ind_zmin),
            ListItem::Integer(&mut ind_zmax),
        ]);
        println!(" Enter list of numbers of objects whose ends should be clipped (Return for all)");
        let _ = std::io::stdout().flush();
        let mut stdin = std::io::stdin().lock();
        let _ = rdlist(&mut stdin, &mut iobj_clip, &mut num_obj_clip);
    }
    //
    ind_xmin = ind_xmin.max(min_xpiece);
    ind_ymin = ind_ymin.max(min_ypiece);
    ind_xmax = ind_xmax.min(max_xpiece);
    ind_ymax = ind_ymax.min(max_ypiece);
    //
    let mut num_load_obj = 0_usize;
    let _ = getimodflags(&mut iobj_flags);

    for imod_obj in 1..=num_obj_tot {
        let mut if_use = 0;
        if num_obj_clip > 0 {
            for imobj in 0..num_obj_clip as usize {
                if iobj_clip[imobj] == imod_obj {
                    if_use = 1;
                }
            }
        } else {
            if_use = iobj_flags[(imod_obj - 1) as usize] % 4;
        }
        if if_use != 0 {
            num_load_obj += 1;
            load_objs[num_load_obj - 1] = imod_obj;
        }
    }
    if num_load_obj == 0 && num_obj_clip > 0 {
        exit_error("The list of objects to use does not specify any valid object numbers");
    }
    if num_load_obj == 0 {
        exit_error(
            "All objects have closed contours and will not be used unless specified with -objects",
        );
    }
    if !get_model_object_list(&load_objs[..num_load_obj], num_load_obj as i32, &mut fm) {
        exit_error("Loading model data");
    }
    //
    // convert to index coords in the current volume
    //
    scale_model_to_image(1, 0, &mut fm);
    //
    if pip_input {
        let _ = pip_get_integer(b"WhichPointsToExtract", &mut if_start_end);
        if pip_get_integer(b"BoxSizeXY", &mut num_pix_box) != 0 {
            if pip_get_three_integers(
                b"VolumeSizeXYZ",
                &mut num_pix_box,
                &mut ny_pix_box,
                &mut nz_pix_box,
            ) != 0
            {
                exit_error("Box size must be entered with -box or -volume");
            }
        } else {
            ny_pix_box = num_pix_box;
            nz_pix_box = num_pix_box;
        }
        if pip_get_two_integers(b"TaperAtFill", &mut num_taper, &mut inside_taper) == 0 {
            if !(0..=1).contains(&inside_taper) {
                exit_error("Value for whether to taper inside must be 0 or 1");
            }
            if num_taper <= 0
                || (num_taper > num_pix_box.min(ny_pix_box).min(nz_pix_box) - 2 && nz_pix_box > 1)
            {
                exit_error(
                    "Number of pixels to taper must be positive and less than the smallest box dimension",
                );
            }
        }
    } else {
        prompt("Clip out starts (0) or ends (1) or all points (-1): ");
        read_stdin_list(&mut [ListItem::Integer(&mut if_start_end)]);
        //
        prompt("Box size in pixels: ");
        read_stdin_list(&mut [ListItem::Integer(&mut num_pix_box)]);
        ny_pix_box = num_pix_box;
        nz_pix_box = num_pix_box;
    }
    //
    let num_pix_sq = num_pix_box * ny_pix_box;
    let npix_left = (num_pix_box - 1) / 2;
    let npix_right = num_pix_box - npix_left - 1;
    let npix_bot = (ny_pix_box - 1) / 2;
    let npix_top = ny_pix_box - npix_bot - 1;
    offset[0] = -0.5 * (1 + npix_right - npix_left) as f32;
    offset[1] = -0.5 * (1 + npix_top - npix_bot) as f32;
    let ib_base = IDIM - num_pix_sq as i64 - 1;
    let ia_base = ib_base - num_pix_sq as i64;
    if ia_base < 0 {
        exit_error("Box size too large for arrays");
    }
    //
    let mut nz_before = nz_pix_box / 2;
    let mut nz_after = nz_pix_box - 1 - nz_before;
    if pip_input {
        let _ = pip_get_two_integers(b"SlicesBelowAndAbove", &mut nz_before, &mut nz_after);
    } else {
        prompt("# of sections before and # of sections after endpoint to clip out: ");
        read_stdin_list(&mut [
            ListItem::Integer(&mut nz_before),
            ListItem::Integer(&mut nz_after),
        ]);
    }
    let nz_clip = nz_before + nz_after + 1;
    if nz_clip < 1 || num_pix_box < 1 || ny_pix_box < 1 {
        exit_error("Box size and number of slices must be positive");
    }
    offset[2] = 0.;
    if nz_before % 2 == nz_after % 2 {
        offset[2] = -0.495;
    }
    let mut dmin_tmp = vec![0.0_f32; nz_clip as usize];
    let mut dmax_tmp = vec![0.0_f32; nz_clip as usize];
    let mut dmean_tmp = vec![0.0_f32; nz_clip as usize];
    memory_error(0, "arrays for min/max/mean");
    //
    // Manage the Z limits if none entered
    if ind_zmin == -99999 {
        ind_zmin = listz[0] - nz_before;
    }
    if ind_zmax == -99999 {
        ind_zmax = listz[(num_listz - 1) as usize] + nz_after;
    }
    //
    let if_series: i32;
    if pip_input {
        if_series = 1 - pipgetstring_(b"SeriesRootName", &mut root_name);
    } else {
        let mut value = 0_i32;
        prompt("1 to output numbered series of files, 0 for single output file: ");
        read_stdin_list(&mut [ListItem::Integer(&mut value)]);
        if_series = value;
        if if_series != 0 {
            prompt("Root name for output files: ");
            read_name(&mut root_name);
        }
    }
    let local_mean_fill = blank(&piece_file) && nz_before == 0 && nz_after == 0;

    let mut if_average: i32;
    let mut num_predict = 0_i32;
    let mut unit3: Option<std::fs::File> = None;
    let mut unit4: Option<std::fs::File> = None;
    if if_series != 0 {
        if_average = 0;
        piece_file.fill(b' ');
        piece_file2.fill(b' ');
        //
        // precount the points
        //
        num_predict = 0;
        for iobj in 0..fm.max_mod_obj as usize {
            if fm.npt_in_obj[iobj] > 0 {
                if if_start_end < 0 {
                    num_predict += fm.npt_in_obj[iobj];
                } else {
                    num_predict += 1;
                }
            }
        }
    } else {
        if_average = 1;
        if ia_base < num_pix_sq as i64 * nz_clip as i64 {
            if_average = 0;
            println!(" WARNING: BOXSTARTEND - VOLUMES TOO LARGE TO COMPUTE AVERAGE");
        } else {
            for value in bigarr[..(num_pix_sq * nz_clip) as usize].iter_mut() {
                *value = 0.;
            }
        }
        if pip_get_in_out_file("OutputFile", 3, "Output image file name", &mut in_file, 320) != 0 {
            exit_error("No output image file specified");
        }
        //
        imopen(2, &in_file, "new");
        nxyz2[0] = num_pix_box;
        nxyz2[1] = ny_pix_box;
        nxyz2[2] = 1;
        iiu_trans_header(2, 1);
        iiu_alt_size(2, &nxyz2, &nxyzst);
        //
        piece_file.fill(b' ');
        piece_file2.fill(b' ');
        if pip_input {
            let _ = pipgetstring_(b"ArrayPieceList", &mut piece_file);
            let _ = pipgetstring_(b"TruePieceList", &mut piece_file2);
            let _ = pip_get_integer(b"BlankBetweenImages", &mut num_gutter);
        } else {
            prompt("Output file for 2D array piece list, or Return for none: ");
            read_name(&mut piece_file);
            //
            if !blank(&piece_file) {
                prompt("Number of empty pixels between clips: ");
                read_stdin_list(&mut [ListItem::Integer(&mut num_gutter)]);
            }
            println!(" Output file for real coordinate piece list, or Return for none: ");
            let _ = std::io::stdout().flush();
            read_name(&mut piece_file2);
        }
        //
        if !blank(&piece_file) {
            unit3 = Some(dopen(3, &fortran_string(&piece_file), "new", "f"));
        }
        if !blank(&piece_file2) {
            unit4 = Some(dopen(4, &fortran_string(&piece_file2), "new", "f"));
        }
    }
    //
    // set up for loop on model objects
    //
    let mut nz_out = 0_i32;
    let mut nz_file_out = 0_i32;
    let mut num_clip = 0_i32;
    let mut num_file = 0_i32;
    let mut dsum = 0.0_f32;
    let mut dmin2 = 1.0e30_f32;
    let mut dmax2 = -1.0e30_f32;
    let (mut ind_left, mut ind_bot, mut ind_zlo) = (0_i32, 0_i32, 0_i32);
    // `brray(k)`/`array(k)` index into the shared block
    let bi = |k: i64| (k - 1) as usize;
    for iobj in 1..=fm.max_mod_obj {
        let ko = (iobj - 1) as usize;
        if fm.npt_in_obj[ko] > 0 {
            let mut loop_start = 1;
            let mut loop_end = fm.npt_in_obj[ko];
            if if_start_end == 0 {
                loop_end = 1;
            } else if if_start_end > 0 {
                loop_start = loop_end;
            }

            for lp in loop_start..=loop_end {
                let ipnt = fm.object[(lp + fm.ibase_obj[ko] - 1) as usize].abs();
                let p = fm.p_coord[(ipnt - 1) as usize];
                for i in 0..3 {
                    ind[i] = (p[i] + offset[i]).round() as i32;
                }
                // `write(*,'(3f12.5,3i7)')`
                println!(
                    "{}{}{}{:>7}{:>7}{:>7}",
                    format_f(p[0] as f64, 12, 5),
                    format_f(p[1] as f64, 12, 5),
                    format_f(p[2] as f64, 12, 5),
                    ind[0],
                    ind[1],
                    ind[2]
                );
                //
                // is it inside limits?
                //
                ind_left = ind[0] - npix_left;
                let mut ind_right = ind[0] + npix_right;
                ind_bot = ind[1] - npix_bot;
                let mut ind_top = ind[1] + npix_top;
                ind_zlo = ind[2] - nz_before;
                let ind_zhi = ind[2] + nz_after;
                if ind[0] >= ind_xmin
                    && ind[0] <= ind_xmax
                    && ind[1] >= ind_ymin
                    && ind[1] <= ind_ymax
                    && ind_zlo >= ind_zmin
                    && ind_zhi <= ind_zmax
                {
                    //
                    // set up the file if doing series, use this counter regardless
                    //
                    num_file += 1;
                    if if_series != 0 {
                        let conv_num = if num_predict < 100 {
                            fmt_im(num_file, 2)
                        } else if num_predict < 1000 {
                            fmt_im(num_file, 3)
                        } else if num_predict < 10000 {
                            fmt_im(num_file, 4)
                        } else {
                            fmt_im(num_file, 5)
                        };
                        in_file = format!("{}.{}", fortran_string(&root_name), conv_num.trim());
                        imopen(2, &in_file, "new");
                        nxyz2[0] = num_pix_box;
                        nxyz2[1] = ny_pix_box;
                        nxyz2[2] = nz_clip;
                        iiu_trans_header(2, 1);
                        iiu_alt_size(2, &nxyz2, &nxyzst);
                        dsum = 0.;
                        dmin2 = 1.0e30;
                        dmax2 = -1.0e30;
                        nz_file_out = 0;
                    }
                    //
                    // loop on sections
                    //
                    let iz_at_start = nz_file_out;
                    let mut ind_zfirst = -1_i32;
                    let mut ind_zlast = -1_i32;
                    for ind_z in ind_zlo..=ind_zhi {
                        if if_xform != 0 {
                            let mut indg = 0_usize;
                            for ilist in 1..=num_listz as usize {
                                if listz[ilist - 1] == ind_z {
                                    indg = ilist;
                                }
                            }
                            if indg != 0 {
                                let (x_new, y_new) =
                                    xfapply(&g[indg - 1], x_cen, y_cen, p[0], p[1]);
                                ind[0] = (x_new + offset[0]).round() as i32;
                                ind[1] = (y_new + offset[1]).round() as i32;
                                ind_left = ind[0] - npix_left;
                                ind_right = ind[0] + npix_right;
                                ind_bot = ind[1] - npix_bot;
                                ind_top = ind[1] + npix_top;
                            }
                        }
                        let indar = ind_z + 1 - ind_zlo;
                        //
                        // zero out the box
                        //
                        let mut need_xy_taper = num_taper > 0;
                        let mut need_fill = true;
                        let mut max_pixels = 0_i32;
                        if !local_mean_fill {
                            fill_if_needed(&mut bigarr, &mut need_fill, dmean, num_pix_sq, ib_base);
                        }
                        //
                        // loop on pieces, find intersection with each piece
                        //
                        for ipc in 1..=num_pc_list as usize {
                            if iz_pc_list[ipc - 1] == ind_z {
                                let ixpc = ix_pc_list[ipc - 1];
                                let iypc = iy_pc_list[ipc - 1];
                                let ipc_xstart = ind_left.max(ixpc);
                                let ipc_ystart = ind_bot.max(iypc);
                                let ipc_xend = ind_right.min(ixpc + nx - 1);
                                let ipc_yend = ind_top.min(iypc + ny - 1);
                                if ipc_xstart <= ipc_xend && ipc_ystart <= ipc_yend {
                                    //
                                    // if it intersects, read in intersecting part,
                                    //
                                    max_pixels = ((ipc_xend + 1 - ipc_xstart)
                                        * (ipc_yend + 1 - ipc_ystart))
                                        .max(max_pixels);
                                    unsafe {
                                        iiu_set_position(1, ipc as i32 - 1, 0);
                                        if irdpas(
                                            1,
                                            &mut bigarr[ia_base as usize..],
                                            num_pix_box,
                                            ny_pix_box,
                                            ipc_xstart - ixpc,
                                            ipc_xend - ixpc,
                                            ipc_ystart - iypc,
                                            ipc_yend - iypc,
                                        )
                                        .is_err()
                                        {
                                            exit_error("Reading file");
                                        }
                                    }
                                    //
                                    // cancel tapering if the whole area is filled, record last
                                    // image that has data and first one if not set yet
                                    if ipc_xend + 1 - ipc_xstart == num_pix_box
                                        && ipc_yend + 1 - ipc_ystart == ny_pix_box
                                    {
                                        need_xy_taper = false;
                                        need_fill = false;
                                    }
                                    ind_zlast = ind_z;
                                    if ind_zfirst < 0 {
                                        ind_zfirst = ind_z;
                                    }
                                    //
                                    // Get mean and fill array if needed
                                    if need_fill {
                                        let (mut tlf_min, mut tlf_max, mut tlf_mean) =
                                            (0.0_f32, 0.0_f32, 0.0_f32);
                                        array_min_max_mean_fortran(
                                            &bigarr[ia_base as usize..],
                                            &num_pix_box,
                                            &ny_pix_box,
                                            &1,
                                            &(ipc_xend + 1 - ipc_xstart),
                                            &1,
                                            &(ipc_yend + 1 - ipc_ystart),
                                            &mut tlf_min,
                                            &mut tlf_max,
                                            &mut tlf_mean,
                                        );
                                        fill_if_needed(
                                            &mut bigarr,
                                            &mut need_fill,
                                            tlf_mean,
                                            num_pix_sq,
                                            ib_base,
                                        );
                                    }
                                    //
                                    // move it into appropriate part of array
                                    for iy in 1..=(ipc_yend + 1 - ipc_ystart) as i64 {
                                        for ix in 1..=(ipc_xend + 1 - ipc_xstart) as i64 {
                                            let to = ib_base
                                                + ix
                                                + (ipc_xstart - ind_left) as i64
                                                + (iy + (ipc_ystart - ind_bot) as i64 - 1)
                                                    * num_pix_box as i64;
                                            let from = ia_base + ix + (iy - 1) * num_pix_box as i64;
                                            bigarr[bi(to)] = bigarr[bi(from)];
                                        }
                                    }
                                }
                            }
                        }
                        //
                        // Taper if called for and one box didn't fill it
                        if need_xy_taper
                            && max_pixels > num_pix_box * ny_pix_box / 5
                            && taperatfill(
                                &mut bigarr[ib_base as usize..],
                                &num_pix_box,
                                &ny_pix_box,
                                &num_taper,
                                &inside_taper,
                            ) != 0
                        {
                            exit_error("memory error tapering slice");
                        }
                        fill_if_needed(&mut bigarr, &mut need_fill, dmean, num_pix_sq, ib_base);
                        //
                        // write piece list and slice
                        //
                        if let Some(file) = unit4.as_mut() {
                            let _ = writeln!(file, "{:>6}{:>6}{:>6}", ind_left, ind_bot, ind_z);
                        }
                        let ix = (ind_z + 1 - ind_zlo) as usize;
                        array_min_max_mean_fortran(
                            &bigarr[ib_base as usize..],
                            &num_pix_box,
                            &ny_pix_box,
                            &1,
                            &num_pix_box,
                            &1,
                            &ny_pix_box,
                            &mut dmin_tmp[ix - 1],
                            &mut dmax_tmp[ix - 1],
                            &mut dmean_tmp[ix - 1],
                        );
                        unsafe {
                            iiu_write_section(2, bigarr[ib_base as usize..].as_mut_ptr().cast());
                        }
                        nz_out += 1;
                        nz_file_out += 1;
                        //
                        // Add to average only if flag set
                        if if_average != 0 {
                            for ix in 1..=num_pix_sq as i64 {
                                let jnd = ix + (indar - 1) as i64 * num_pix_sq as i64;
                                bigarr[bi(jnd)] += bigarr[bi(ib_base + ix)];
                            }
                        }
                    }
                    //
                    // Done with Z loop, see if anything needs Z tapering and if so, loop on
                    // all Z's
                    if num_taper > 0
                        && ind_zfirst > -1
                        && (ind_zfirst > ind_zlo || ind_zlast < ind_zhi)
                    {
                        let mut iz_load = 0_i32;
                        for ind_z in ind_zlo..=ind_zhi {
                            let mut atten = 1.0_f32;
                            //
                            // Evaluate smallest attenuation value based on nearest boundary for
                            // an inside taper and set up to load this Z
                            if inside_taper > 0 {
                                let mut atten2 = 1.0_f32;
                                if ind_zfirst > ind_zlo
                                    && ind_z >= ind_zfirst
                                    && ind_z < ind_zfirst + num_taper
                                {
                                    atten = (ind_z as f32 + 1. - ind_zfirst as f32)
                                        / (num_taper as f32 + 1.);
                                }
                                if ind_zlast < ind_zhi
                                    && ind_z <= ind_zlast
                                    && ind_z > ind_zlast - num_taper
                                {
                                    atten2 = (ind_zlast as f32 + 1. - ind_z as f32)
                                        / (num_taper as f32 + 1.);
                                }
                                atten = minss(atten, atten2);
                                iz_load = iz_at_start + ind_z - ind_zlo;
                            } else {
                                //
                                // Or find which boundary the Z is past for an outside taper and
                                // set up to load that boundary slice
                                if ind_z >= ind_zfirst - num_taper && ind_z < ind_zfirst {
                                    iz_load = iz_at_start + ind_zfirst - ind_zlo;
                                    atten =
                                        1. - (ind_zfirst - ind_z) as f32 / (num_taper as f32 + 1.);
                                } else if ind_z <= ind_zlast + num_taper && ind_z > ind_zlast {
                                    iz_load = iz_at_start + ind_zlast - ind_zlo;
                                    atten =
                                        1. - (ind_z - ind_zlast) as f32 / (num_taper as f32 + 1.);
                                }
                            }
                            if atten > 0. && atten < 1. {
                                //
                                // Load the slice, attenuate it, rewrite and fix mean
                                unsafe {
                                    iiu_set_position(2, iz_load, 0);
                                    if irdsec(2, &mut bigarr[ib_base as usize..]).is_err() {
                                        exit_error("Reading file");
                                    }
                                }
                                for ix in 1..=num_pix_sq as i64 {
                                    let v = bigarr[bi(ib_base + ix)];
                                    bigarr[bi(ib_base + ix)] = atten * v + (1. - atten) * dmean;
                                }
                                unsafe {
                                    iiu_set_position(2, iz_at_start + ind_z - ind_zlo, 0);
                                }
                                let ix = (ind_z + 1 - ind_zlo) as usize;
                                array_min_max_mean_fortran(
                                    &bigarr[ib_base as usize..],
                                    &num_pix_box,
                                    &ny_pix_box,
                                    &1,
                                    &num_pix_box,
                                    &1,
                                    &ny_pix_box,
                                    &mut dmin_tmp[ix - 1],
                                    &mut dmax_tmp[ix - 1],
                                    &mut dmean_tmp[ix - 1],
                                );
                                unsafe {
                                    iiu_write_section(
                                        2,
                                        bigarr[ib_base as usize..].as_mut_ptr().cast(),
                                    );
                                }
                            }
                        }
                        unsafe {
                            iiu_set_position(2, nz_file_out, 0);
                        }
                    }
                    //
                    // Maintain min/max/mean
                    for ix in 0..nz_clip as usize {
                        dmin2 = minss(dmin2, dmin_tmp[ix]);
                        dmax2 = maxss(dmax2, dmax_tmp[ix]);
                        dsum += dmean_tmp[ix];
                    }
                    //
                    // Close up particle file
                    if if_series != 0 {
                        let dmean2 = dsum / nz_clip as f32;
                        for i in 0..3 {
                            cell[i] = nxyz2[i] as f32 * delta[i];
                            cell[i + 3] = 90.;
                        }
                        iiu_alt_size(2, &nxyz2, &nxyzst);
                        iiu_alt_sample(2, &nxyz2);
                        iiu_alt_cell(2, &cell);
                        iiu_alt_origin(
                            2,
                            &[
                                origin[0] - delta[0] * ind_left as f32,
                                origin[1] - delta[1] * ind_bot as f32,
                                origin[2] - delta[2] * ind_zlo as f32,
                            ],
                        );
                        b3d_date(&mut date_strn);
                        time(&mut time_strn);
                        // `111 format('BOXSTARTEND: Individual box clipped out' ,t57,a9,2x,a8)`
                        let mut title = [b' '; MRC_LABEL_SIZE];
                        let head = b"BOXSTARTEND: Individual box clipped out";
                        title[..head.len()].copy_from_slice(head);
                        title[56..65].copy_from_slice(&date_strn);
                        title[67..75].copy_from_slice(&time_strn);
                        iiu_write_header(2, &title, 1, dmin2, dmax2, dmean2);
                        unsafe {
                            iiu_close(2);
                        }
                    }
                    num_clip += 1;
                }
            }
        }
    }
    //
    // take care of averages and finish stack file
    //
    if if_series == 0 {
        if if_average != 0 {
            for indar in 1..=nz_clip as i64 {
                for ix in 1..=num_pix_sq as i64 {
                    let jnd = ix + (indar - 1) * num_pix_sq as i64;
                    bigarr[bi(jnd)] /= num_clip as f32;
                }
                let jnd = 1 + (indar - 1) * num_pix_sq as i64;
                let (mut tmin, mut tmax, mut tmean) = (0.0_f32, 0.0_f32, 0.0_f32);
                array_min_max_mean_fortran(
                    &bigarr[bi(jnd)..],
                    &num_pix_box,
                    &ny_pix_box,
                    &1,
                    &num_pix_box,
                    &1,
                    &ny_pix_box,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
                dmin2 = minss(dmin2, tmin);
                dmax2 = maxss(dmax2, tmax);
                dsum += tmean;
                unsafe {
                    iiu_write_section(2, bigarr[bi(jnd)..].as_mut_ptr().cast());
                }
                nz_out += 1;
            }
        }
        let dmean2 = dsum / nz_out as f32;
        nxyz2[2] = nz_out;
        for i in 0..3 {
            cell[i] = nxyz2[i] as f32 * delta[i];
            cell[i + 3] = 90.;
        }
        iiu_alt_size(2, &nxyz2, &nxyzst);
        iiu_alt_sample(2, &nxyz2);
        iiu_alt_cell(2, &cell);

        b3d_date(&mut date_strn);
        time(&mut time_strn);
        let mut text_strt_end = "starts";
        if if_start_end > 0 {
            text_strt_end = "ends  ";
        }
        if if_start_end < 0 {
            text_strt_end = "points";
        }
        // `101 format('BOXSTARTEND: ',i4,1x,a6,' clipped out and averaged' ,t57,a9,2x,a8)`
        let mut title = [b' '; MRC_LABEL_SIZE];
        let num_text = format!("{:>4}", num_clip);
        let num_text = if num_text.len() > 4 {
            "****".to_owned()
        } else {
            num_text
        };
        let head = format!("BOXSTARTEND: {num_text} {text_strt_end} clipped out and averaged");
        let count = head.len().min(56);
        title[..count].copy_from_slice(&head.as_bytes()[..count]);
        title[56..65].copy_from_slice(&date_strn);
        title[67..75].copy_from_slice(&time_strn);
        iiu_write_header(2, &title, 1, dmin2, dmax2, dmean2);
        unsafe {
            iiu_close(2);
        }
        num_clip += 1;
    }
    drop(unit4);
    //
    // now put out piece list if desired
    //
    if let Some(mut file) = unit3.take() {
        let new_ypiece = (num_clip as f32).sqrt().round() as i32;
        let new_xpiece = (num_clip + new_ypiece - 1) / new_ypiece;
        let mut ind_x = 0;
        let mut ind_y = 0;
        let mut text = String::new();
        for _ in 1..=num_clip {
            let ixpc = ind_x * (num_pix_box + num_gutter);
            let iypc = ind_y * (ny_pix_box + num_gutter);
            for iz in 0..nz_clip {
                text.push_str(&format!("{:>6}{:>6}{:>6}\n", ixpc, iypc, iz));
            }
            ind_x += 1;
            if ind_x >= new_xpiece {
                ind_x = 0;
                ind_y += 1;
            }
        }
        let _ = file.write_all(text.as_bytes());
    }
    println!("{:>12}  points boxed out", num_file);
    let _ = std::io::stdout().flush();
    exit(0);
}
