//! Translation of `IMOD/flib/model/edmont.f90`.
//!
//! EDMONT is a general montage editor to move images into, out of, or
//! between montages.  It can float the images to a common range or mean of
//! density, scaling all of the pieces in a section by the same amount.  It
//! can output only the pieces that intersect a defined subset of the image
//! area.  David Mastronarde for VAX 5/9/89.
//!
//! The main program maps to [`edmont`]; the internal routines are
//! [`not_knock`], [`read_pl_or_header`] and [`open_and_analyze_files`].
//! Library calls go through the Fortran wrappers the source calls:
//! `iiuRetAdocIndex`, `AdocLookupByNameValue` return the C index plus one,
//! and `AdocSetCurrent`/`AdocClear`/`AdocTransferSection`/
//! `AdocSetThreeIntegers` subtract one (`adoc_fwrap.c`, `unit_fileio.c:402`);
//! `get_extra_header_pieces`/`get_metadata_pieces` are the 1-based
//! `extraheader.c` wrappers; `iclden`, `iclavgsd` and `sums_to_avgsd8` the
//! `simplestat.c` ones.  Allocation failure aborts in Rust, so the source's
//! `memoryError` checks after each `allocate` cannot fire and are not
//! repeated.  The NaN operand order of the Fortran `MIN`/`MAX` sites is taken
//! as accumulator first (`minss`/`maxss` destination), as in `subimage`.

use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::{format_f, maxss, minss};
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::int_iwrite::int_iwrite;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_logical, pip_read_or_parse_options, set_current_adoc_or_exit,
};
use crate::imod::flib::subrs::hvem::rdlist::parselist2;
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::scale_model;
use crate::imod::flib::subrs::piecesubs::read_piece_list::read_piece_list2;
use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, ADOC_ZVALUE_NAME, adoc_clear, adoc_get_image_meta_info,
    adoc_lookup_by_name_value, adoc_set_three_integers, adoc_transfer_section, adoc_write,
};
use crate::imod::libcfshr::b3dutil::{
    ImodFile, b3d_output_file_type, exit, fortran_string, set_float_output_for_entered_mode,
};
use crate::imod::libcfshr::extraheader::{
    get_extra_header_pieces_fortran, get_metadata_pieces_fortran,
};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_boolean, pip_get_integer, pip_get_integer_array, pip_get_two_integers,
    pip_number_of_entries,
};
use crate::imod::libcfshr::piecefuncs::{checklist, fill_listz};
use crate::imod::libcfshr::pip_fwrap::{pipgetnonoptionarg_, pipgetstring_};
use crate::imod::libcfshr::simplestat::{
    array_min_max_mean_fortran, array_min_max_mean_sd_fortran, sums_to_avg_sd_dbl,
};
use crate::imod::libiimod::mrcfiles::MRC_LABEL_SIZE;
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_file_type, iiu_ret_adoc_index, iiu_set_position, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_extended_data, iiu_alt_mode, iiu_alt_num_extended, iiu_alt_origin,
    iiu_alt_sample, iiu_alt_size, iiu_ret_cell, iiu_ret_delta, iiu_ret_extended_data,
    iiu_ret_extended_type, iiu_ret_num_extended, iiu_ret_origin, iiu_ret_size, iiu_trans_header,
    iiu_write_header,
};
use crate::imod::libiimod::unit_reduced::iiu_read_binned;
use std::io::Write;

/// `parameter (numOptions = 18)` (`edmont.f90:82`).
const EDMONT_NUM_OPTIONS: i32 = 18;
/// Fallback PIP table, the `options(1)` string (`edmont.f90:84-93`).
const EDMONT_OPTIONS: &str = "imin:ImageInputFile:FNM:@plin:PieceListInput:FNM:@\
imout:ImageOutputFile:FNM:@plout:PieceListOutput:FNM:@\
secs:SectionsToRead:LIM:@numout:NumberToOutput:IAM:@\
mode:ModeToOutput:I:@xminmax:XMinAndMax:IP:@\
Yminmax:YMinAndMax:IP:@xframes:XFrameMinAndMax:IP:@\
yframes:YFrameMinAndMax:IP:@float:FloatDensities:I:@\
bin:BinByFactor:I:@exclude:ExclusionModel:FN:@\
renumber:RenumberZFromZero:B:@shift:ShiftXYToZero:B:@\
param:ParameterFile:PF:@help:usage:B:";

/// `data optimalMax/255., 32767., 4*255., 65535., 2*255., 511., 1023., 2047.,
/// 4095., 8191., 16383., 32767./` (`edmont.f90:33-34`).
const OPTIMAL_MAX: [f32; 16] = [
    255., 32767., 255., 255., 255., 255., 65535., 255., 255., 511., 1023., 2047., 4095., 8191.,
    16383., 32767.,
];

/// `optimalMax(mode + 1)`.  Fixed in translation: the source indexes its
/// 16-element table with any mode, reading past it for a mode above 15;
/// such a mode takes the byte maximum here.
fn optimal_max(mode: i32) -> f32 {
    OPTIMAL_MAX.get(mode as usize).copied().unwrap_or(255.)
}

/// Original program `edmont` (`edmont.f90:10`).
pub fn edmont() {
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    // `data nxyzst/0, 0, 0/`
    let mut nxyzst = [0_i32; 3];
    let mut nxyz2 = [0_i32; 3];
    let mut mxyz2 = [0_i32; 3];
    let mut title = [b' '; MRC_LABEL_SIZE];
    let mut cell2 = [0.0_f32; 6];
    let mut cell: [f32; 6];
    let mut delta: [f32; 3];
    let (mut x_origin, mut y_origin, mut z_origin) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut in_file: Vec<String>;
    let mut out_file: Vec<String>;
    let mut piece_file_in: Vec<String>;
    let mut piece_file_out: Vec<String>;
    let mut model_file = [b' '; 320];
    // `character*20 floatText/' '/, truncText/' '/`
    let mut float_text = [b' '; 20];
    let mut trunc_text = [b' '; 20];
    let mut fm = FortModel::default();
    let mut dat = [b' '; 9];
    let mut tim = [b' '; 8];
    let mut list_string = vec![b' '; 100000];
    let mut num_opt_arg = 0_i32;
    let mut num_non_opt_arg = 0_i32;
    let mut ierr: i32;
    let mut mode = 0_i32;
    let mut dmin_in = 0.0_f32;
    let mut nbyte_extra_in = 0_i32;
    let mut nbyte_per_sec_in = 0_i32;
    let mut num_pc_list = 0_i32;
    let mut num_in_zlist = 0_i32;
    let (mut min_xpiece, mut num_xpieces, mut mx_overlap) = (0_i32, 0_i32, 0_i32);
    let (mut min_ypiece, mut num_ypieces, mut my_overlap) = (0_i32, 0_i32, 0_i32);
    let mut new_mode = 0_i32;
    let (mut min_all_xpiece, mut min_all_ypiece) = (0_i32, 0_i32);
    let (mut max_xpiece, mut max_ypiece) = (0_i32, 0_i32);
    let (mut min_frame, mut max_frame) = (0_i32, 0_i32);
    let mut nbyte_per_sec_out = 0_i32;
    let mut nbyte_extra_out = 0_i32;
    let mut ind_xout = 0_i32;
    let mut iflag_extra_in = 0_i32;
    let mut ind_adoc_out = -1_i32;
    let mut out_doc_changed = false;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut tmp_min, mut tmp_max, mut sum) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut avg, mut sd) = (0.0_f32, 0.0_f32);
    let (mut tsum8, mut tsumsq8) = (0.0_f64, 0.0_f64);
    let (mut dmin2, mut dmax2): (f32, f32);

    let lim_extra: i32 = 20000000;
    let lim_piece: i32 = 2500000;
    let lim_list: i32 = 10000000;
    let max_binning = 32;
    let frac_zero = 0.0_f32;
    let mut if_mean = 0_i32;
    let mut if_float = 0_i32;
    let mut if_renumber = 0_i32;
    let mut if_shift_xy = 0_i32;
    let mut ibinning = 1_i32;
    let mut min_xinc = 0_i32;
    let mut max_xinc = 0_i32;
    let mut min_yinc = 0_i32;
    let mut max_yinc = 0_i32;
    let mut number_offset = 0_i32;
    let mut all_have_header_coords = true;
    let mut use_mdoc = false;
    fm.fm_mod_size_type = 2;
    // `call AdocGetStandardNames(globalName, zvalueName)`
    let global_name = ADOC_GLOBAL_NAME;
    let zvalue_name = ADOC_ZVALUE_NAME;

    pip_read_or_parse_options(
        &[EDMONT_OPTIONS],
        EDMONT_NUM_OPTIONS,
        "edmont",
        "ERROR: EDMONT - ",
        false,
        2,
        2,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    //
    // Allocate oversized temporary arrays
    let mut inlist_tmp = vec![0_i32; lim_list as usize];
    let mut listz = vec![0_i32; lim_piece as usize];
    let mut ix_pc_list = vec![0_i32; lim_piece as usize];
    let mut iy_pc_list = vec![0_i32; lim_piece as usize];
    let mut iz_pc_list = vec![0_i32; lim_piece as usize];
    let mut extra_in = vec![0_u8; lim_extra as usize];

    //
    // Get the input files
    let mut num_image_in_files = 0_i32;
    pip_number_of_entries(b"ImageInputFile", &mut num_image_in_files);
    let num_files_in = num_image_in_files + 0.max(num_non_opt_arg - 1);
    if num_files_in == 0 {
        exit_error("No input image file name entered");
    }
    let mut num_piece_files = 0_i32;
    pip_number_of_entries(b"PieceListInput", &mut num_piece_files);
    if num_piece_files > 0 && num_piece_files != num_files_in {
        exit_error(
            "There must be one piece list entry for each image file if there are any piece \
             list files",
        );
    }
    let mut num_sec_lists = 0_i32;
    pip_number_of_entries(b"SectionsToRead", &mut num_sec_lists);
    if num_sec_lists > num_files_in {
        exit_error("There are more section lists than input files");
    }
    let _ = pip_get_boolean(b"NumberedFromOne", &mut number_offset);
    let _ = pip_get_logical("UseMdocFiles", &mut use_mdoc);

    let nfi = num_files_in as usize;
    in_file = vec![String::new(); nfi];
    piece_file_in = vec![String::new(); nfi];
    let mut nlist = vec![0_i32; nfi];
    let mut list_ind = vec![0_i32; nfi];

    let mut list_total = 0_i32;
    let mut max_extra = 1_i32;
    let mut max_pieces = 0_i32;
    let mut max_byte_extra_in = 0_i32;
    let mut max_num_xpieces = 0_i32;
    let mut max_num_ypieces = 0_i32;
    let mut nx_overlap = -10000_i32;
    let mut ny_overlap = -10000_i32;
    let mut name = [b' '; 320];
    for ifile in 1..=num_files_in {
        let fi = (ifile - 1) as usize;
        if ifile <= num_image_in_files {
            let _ = pipgetstring_(b"ImageInputFile", &mut name);
        } else {
            let _ = pipgetnonoptionarg_(ifile - num_image_in_files, &mut name);
        }
        in_file[fi] = fortran_string(&name);
        piece_file_in[fi].clear();
        if num_piece_files > 0 {
            name.fill(b' ');
            let _ = pipgetstring_(b"PieceListInput", &mut name);
            piece_file_in[fi] = fortran_string(&name);
            if piece_file_in[fi] == "none" {
                piece_file_in[fi].clear();
            }
        }
        name.fill(b' ');
        //
        // open the files to get properties and check them
        open_and_analyze_files(
            &in_file[fi],
            &piece_file_in[fi],
            &mut mxyz,
            &mut dmin_in,
            false,
            &mut mode,
            &mut extra_in,
            lim_extra,
            &mut nbyte_extra_in,
            &mut nbyte_per_sec_in,
            &mut ix_pc_list,
            &mut iy_pc_list,
            &mut iz_pc_list,
            &mut num_pc_list,
            lim_piece,
            &mut listz,
            &mut num_in_zlist,
            &mut min_xpiece,
            &mut num_xpieces,
            &mut mx_overlap,
            &mut min_ypiece,
            &mut num_ypieces,
            &mut my_overlap,
            use_mdoc,
        );
        unsafe { iiu_close(1) };
        max_extra = max_extra.max(nbyte_extra_in);
        max_pieces = max_pieces.max(num_pc_list);
        max_byte_extra_in = max_byte_extra_in.max(nbyte_per_sec_in);
        max_num_xpieces = max_num_xpieces.max(num_xpieces);
        max_num_ypieces = max_num_ypieces.max(num_ypieces);
        if nbyte_extra_in == 0 {
            all_have_header_coords = false;
        }
        //
        // The first file with multiple pieces defines the overlap on an axis
        if nx_overlap == -10000 && num_xpieces > 1 {
            nx_overlap = mx_overlap;
        }
        if ny_overlap == -10000 && num_ypieces > 1 {
            ny_overlap = my_overlap;
        }
        //
        // The first file defines output mode and size; initialize for mins/maxs
        if ifile == 1 {
            new_mode = mode;
            nxyz = mxyz;
            //
            // Keep track of overall minimum and maximum piece coordinate
            min_all_xpiece = min_xpiece;
            min_all_ypiece = min_ypiece;
            max_xpiece = min_xpiece + (num_xpieces - 1) * (nxyz[0] - nx_overlap);
            max_ypiece = min_ypiece + (num_ypieces - 1) * (nxyz[1] - ny_overlap);
        } else {
            let (nx, ny) = (nxyz[0], nxyz[1]);
            //
            // if overlap still not defined on an axis and there is now a
            // disparity, try to set up an overlap that makes sense
            if nx_overlap == -10000 && min_xpiece != min_all_xpiece {
                let mut i = 1;
                while (min_xpiece - min_all_xpiece).abs() / i > nx {
                    i += 1;
                }
                nx_overlap = nx - (min_xpiece - min_all_xpiece).abs() / i;
            }
            if ny_overlap == -10000 && min_ypiece != min_all_ypiece {
                let mut i = 1;
                while (min_ypiece - min_all_ypiece).abs() / i > ny {
                    i += 1;
                }
                ny_overlap = ny - (min_ypiece - min_all_ypiece).abs() / i;
            }
            if nx != mxyz[0] || ny != mxyz[1] {
                exit_error("All image files must have the same size pieces");
            }
            if (num_xpieces > 1 && nx_overlap != mx_overlap)
                || (num_ypieces > 1 && ny_overlap != my_overlap)
            {
                exit_error("All montages must have the same overlaps");
            }
            if (min_xpiece - min_all_xpiece).abs() % (nx - nx_overlap) > 0
                || (min_ypiece - min_all_ypiece).abs() % (ny - ny_overlap) > 0
            {
                exit_error("All montages must have piece coordinates on the same regular grid");
            }
            min_all_xpiece = min_all_xpiece.min(min_xpiece);
            min_all_ypiece = min_all_ypiece.min(min_ypiece);
            max_xpiece = max_xpiece.max(min_xpiece + (num_xpieces - 1) * (nx - nx_overlap));
            max_ypiece = max_ypiece.max(min_ypiece + (num_ypieces - 1) * (ny - ny_overlap));
        }
        //
        // Initialize the section list then get it if there is one
        if list_total + num_in_zlist > lim_list {
            exit_error("Too many sections for initial arrays");
        }
        let lt = list_total as usize;
        inlist_tmp[lt..lt + num_in_zlist as usize].copy_from_slice(&listz[..num_in_zlist as usize]);
        nlist[fi] = num_in_zlist;
        if ifile <= num_sec_lists {
            let _ = pipgetstring_(b"SectionsToRead", &mut list_string);
            let mut lim = lim_list - list_total;
            let _ = parselist2(
                &fortran_string(&list_string),
                &mut inlist_tmp[lt..],
                &mut nlist[fi],
                &mut lim,
            );
            //
            // Check for legality
            for i in 1..=nlist[fi] {
                let idx = lt + (i - 1) as usize;
                inlist_tmp[idx] -= number_offset;
                ierr = 1;
                for j in 1..=num_in_zlist {
                    if listz[(j - 1) as usize] == inlist_tmp[idx] {
                        ierr = 0;
                        break;
                    }
                }
                if ierr == 1 {
                    // write(*,'(/,a,i5,a,i6,a)')
                    print!(
                        "\nERROR: EDMONT - SECTION LIST #{:>5} CONTAINS {:>6}, WHICH IS NOT A \
                         SECTION IN THAT FILE\n",
                        ifile, inlist_tmp[idx]
                    );
                    let _ = std::io::stdout().flush();
                    exit(1);
                }
            }
        }
        list_ind[fi] = list_total + 1;
        list_total += nlist[fi];
    }
    let (nx, ny) = (nxyz[0], nxyz[1]);
    //
    // Get output files and numbers to output
    let mut num_image_out_files = 0_i32;
    pip_number_of_entries(b"ImageOutputFile", &mut num_image_out_files);
    let num_out_files = num_image_out_files + 1.min(num_non_opt_arg);
    if num_out_files == 0 {
        exit_error("No output image file name entered");
    }
    let nfo = num_out_files as usize;
    out_file = vec![String::new(); nfo];
    piece_file_out = vec![String::new(); nfo];
    let mut num_sec_out = vec![0_i32; nfo];
    let mut num_pl_file_out = 0_i32;
    pip_number_of_entries(b"PieceListOutput", &mut num_pl_file_out);
    if (num_pl_file_out == 0 && num_piece_files > 0)
        || (num_pl_file_out != 0 && num_pl_file_out != num_out_files)
    {
        exit_error(
            "There must be an output piece list file for each output image file if there are \
             input piece list files",
        );
    }
    //
    // Take care of output numbers first so they can be totaled
    if num_out_files == 1 {
        num_sec_out[0] = list_total;
    } else if num_out_files == list_total {
        for i in 1..=num_out_files {
            num_sec_out[(i - 1) as usize] = 1;
        }
    } else {
        let mut num_out_entries = 0_i32;
        pip_number_of_entries(b"NumberToOutput", &mut num_out_entries);
        if num_out_entries == 0 {
            exit_error("You must specify number of sections to write to each output file");
        }

        let mut num_out_values = 0_i32;
        for _ in 1..=num_out_entries {
            let mut num_to_get = 0_i32;
            let _ = pip_get_integer_array(
                b"NumberToOutput",
                &mut num_sec_out[num_out_values as usize..],
                &mut num_to_get,
                num_out_files - num_out_values,
            );
            num_out_values += num_to_get;
        }
        if num_out_values != num_out_files {
            exit_error(
                "The number of values for sections to output does not equal the number of \
                 output files",
            );
        }
    }

    let mut num_out_total = 0_i32;
    let mut max_sec_out = 0_i32;
    for ifile in 1..=num_out_files {
        let fo = (ifile - 1) as usize;
        name.fill(b' ');
        if ifile <= num_image_out_files {
            let _ = pipgetstring_(b"ImageOutputFile", &mut name);
        } else {
            let _ = pipgetnonoptionarg_(num_non_opt_arg, &mut name);
        }
        out_file[fo] = fortran_string(&name);
        piece_file_out[fo].clear();
        if num_pl_file_out > 0 {
            name.fill(b' ');
            let _ = pipgetstring_(b"PieceListOutput", &mut name);
            piece_file_out[fo] = fortran_string(&name);
        }
        num_out_total += num_sec_out[fo];
        max_sec_out = max_sec_out.max(num_sec_out[fo]);
    }
    if num_out_total != list_total {
        exit_error("Number of input and output sections does not match");
    }
    let nout_xpiece = 1 + (max_xpiece - min_all_xpiece) / (nx - nx_overlap);
    let nout_ypiece = 1 + (max_ypiece - min_all_ypiece) / (ny - ny_overlap);
    //
    // get limits on output size
    //
    ierr = pip_get_two_integers(b"XMinAndMax", &mut min_xinc, &mut max_xinc);
    if pip_get_two_integers(b"XFrameMinAndMax", &mut min_frame, &mut max_frame) == 0 {
        if ierr == 0 {
            exit_error("You cannot enter both -xminmax and -xframes");
        }
        if min_frame < 1 || max_frame > nout_xpiece || min_frame > max_frame {
            exit_error("Minimum and maximum frames in X out of range or out of order");
        }
        min_xinc = min_all_xpiece + (min_frame - 2) * (nx - nx_overlap) + nx;
        max_xinc = min_all_xpiece + max_frame * (nx - nx_overlap) - 1;
        max_num_xpieces = max_frame + 1 - min_frame;
    }
    ierr = pip_get_two_integers(b"YMinAndMax", &mut min_yinc, &mut max_yinc);
    if pip_get_two_integers(b"YFrameMinAndMax", &mut min_frame, &mut max_frame) == 0 {
        if ierr == 0 {
            exit_error("You cannot enter both -yminmax and -yframes");
        }
        if min_frame < 1 || max_frame > nout_ypiece || min_frame > max_frame {
            exit_error("Minimum and maximum frames in Y out of range or out of order");
        }
        min_yinc = min_all_ypiece + (min_frame - 2) * (ny - ny_overlap) + ny;
        max_yinc = min_all_ypiece + max_frame * (ny - ny_overlap) - 1;
        max_num_ypieces = max_frame + 1 - min_frame;
    }
    //
    // Get model to knock out pieces
    let mut knocks = 0_i32;
    let if_knock = 1 - pipgetstring_(b"ExclusionModel", &mut model_file);
    if if_knock != 0 {
        if !readw_or_imod(&fortran_string(&model_file), &mut fm) {
            exit_error("Reading piece exclusion model");
        }
        scale_model(0, &mut fm);
        for iobj in 1..=fm.max_mod_obj {
            knocks += fm.npt_in_obj[(iobj - 1) as usize];
        }
    }
    let mut ixko = vec![0_i32; (knocks + 1) as usize];
    let mut iyko = vec![0_i32; (knocks + 1) as usize];
    let mut izko = vec![0_i32; (knocks + 1) as usize];
    if if_knock != 0 {
        knocks = 0;
        for iobj in 1..=fm.max_mod_obj {
            for ipt in 1..=fm.npt_in_obj[(iobj - 1) as usize] {
                knocks += 1;
                let p = fm.p_coord[(fm.ibase_obj[(iobj - 1) as usize] + ipt - 1) as usize];
                let k = (knocks - 1) as usize;
                // `nint`
                ixko[k] = p[0].round() as i32;
                iyko[k] = p[1].round() as i32;
                izko[k] = p[2].round() as i32;
            }
        }
    }
    //
    ierr = pip_get_integer(b"ModeToOutput", &mut new_mode);
    if ierr == 0 {
        new_mode = set_float_output_for_entered_mode(new_mode);
    }
    //
    let _ = pip_get_integer(b"FloatDensities", &mut if_float);
    if if_float > 1 {
        if_mean = 1;
    }
    //
    // renumber sections starting at zero: and if not, check for duplicate Z
    let _ = pip_get_boolean(b"RenumberZFromZero", &mut if_renumber);
    let _ = pip_get_boolean(b"ShiftXYToZero", &mut if_shift_xy);
    let _ = pip_get_integer(b"BinByFactor", &mut ibinning);
    if ibinning < 1 || ibinning > max_binning {
        exit_error("Binning is outside of allowed range");
    }
    let mut min_sub_xpiece = min_all_xpiece / ibinning;
    let mut min_sub_ypiece = min_all_ypiece / ibinning;
    pip_done();
    //
    // Manage memory
    let in_zlist: Vec<i32> = inlist_tmp[..list_total as usize].to_vec();
    drop(inlist_tmp);
    let max_extra_out = max_sec_out * max_num_xpieces * max_num_ypieces * max_byte_extra_in + 1;
    let i = max_sec_out * max_num_xpieces * max_num_ypieces;
    let mp = max_pieces as usize;
    listz = vec![0_i32; mp];
    ix_pc_list = vec![0_i32; mp];
    iy_pc_list = vec![0_i32; mp];
    iz_pc_list = vec![0_i32; mp];
    extra_in = vec![0_u8; max_extra as usize];
    // Fixed in translation (BUGS.md, `edmont`): a section's extra-header
    // bytes are copied only while `indXout < nbyteExtraOut`, but the copy
    // itself is `nbytePerSecOut` bytes, which can run past the
    // `maxExtraOut` array when input files have different bytes per
    // section.  The array has room for one more section; only
    // `nbyteExtraOut` bytes are ever written to the header.
    let mut extra_out = vec![0_u8; (max_extra_out + max_byte_extra_in) as usize];
    let lt = list_total as usize;
    let mut dmin_sec = vec![0.0_f32; lt];
    let mut dmax_sec = vec![0.0_f32; lt];
    let mut avgsec = vec![0.0_f32; lt];
    let mut sdsec = vec![0.0_f32; lt];
    let mut ix_piece_out = vec![0_i32; i.max(0) as usize];
    let mut iy_piece_out = vec![0_i32; i.max(0) as usize];
    let mut iz_piece_out = vec![0_i32; i.max(0) as usize];
    let nx_bin = nx / ibinning;
    let nxb_overlap = (((nx_overlap - nx % ibinning) as f32) / ibinning as f32).round() as i32;
    let ix_start = (nx % ibinning) / 2;
    let ny_bin = ny / ibinning;
    let nyb_overlap = (((ny_overlap - ny % ibinning) as f32) / ibinning as f32).round() as i32;
    let iy_start = (ny % ibinning) / 2;
    let len_temp = 2 * ibinning * nx;
    let idim = nx_bin * ny_bin;
    let mut temp_line = vec![0.0_f32; len_temp as usize];
    let mut array = vec![0.0_f32; idim as usize];
    let mut zmin = 1.0e30_f32;
    let mut zmax = -1.0e30_f32;
    let mut dmin_out = 1.0e30_f32;
    let mut dmax_out = -1.0e30_f32;
    let mut isec_out: i32;
    let mut ifile_out: i32;
    let mut ipiece_out: i32;
    if if_renumber == 0
        || if_float != 0
        || knocks > 0
        || min_xinc != 0
        || max_xinc != 0
        || min_yinc != 0
        || max_yinc != 0
    {
        //
        // need to go through all the files and check for unique output pieces
        // if not renumbering in Z; get the actual range of pieces and check
        // for existence of pieces on each section if any pieces are being
        // removed, and get means if floating
        isec_out = 1;
        ifile_out = 1;
        ipiece_out = 0;
        if if_float != 0 {
            set_text(&mut float_text, ", floated to range");
            if if_float < 0 {
                set_text(&mut float_text, ", scaled to range");
            }
            if frac_zero != 0. {
                // write(truncText, '(a,f6.3)') ', truncated by', fracZero
                set_text(
                    &mut trunc_text,
                    &format!(", truncated by{}", format_f(frac_zero as f64, 6, 3)),
                );
            }
            if if_mean != 0 {
                set_text(&mut float_text, ", floated to means");
            }
        }
        zmin = 1.0e30;
        zmax = -1.0e30;
        dmin_out = 1.0e30;
        dmax_out = -1.0e30;
        min_sub_xpiece = max_xpiece + nx;
        min_sub_ypiece = max_ypiece + ny;
        for ifile in 1..=num_files_in {
            let fi = (ifile - 1) as usize;
            open_and_analyze_files(
                &in_file[fi],
                &piece_file_in[fi],
                &mut mxyz,
                &mut dmin_in,
                false,
                &mut mode,
                &mut extra_in,
                max_extra,
                &mut nbyte_extra_in,
                &mut nbyte_per_sec_in,
                &mut ix_pc_list,
                &mut iy_pc_list,
                &mut iz_pc_list,
                &mut num_pc_list,
                max_pieces,
                &mut listz,
                &mut num_in_zlist,
                &mut min_xpiece,
                &mut num_xpieces,
                &mut mx_overlap,
                &mut min_ypiece,
                &mut num_ypieces,
                &mut my_overlap,
                use_mdoc,
            );

            for ilis in 1..=nlist[fi] {
                let ind_sec = ilis + list_ind[fi] - 1;
                let num_sec_read = in_zlist[(ind_sec - 1) as usize];
                dmin2 = 1.0e30;
                dmax2 = -1.0e30;
                let mut sum8 = 0.0_f64;
                let mut sumsq8 = 0.0_f64;
                let mut if_any_out = 0_i32;
                for ipc in 1..=num_pc_list {
                    let pc = (ipc - 1) as usize;
                    if iz_pc_list[pc] == num_sec_read
                        && ((min_xinc == 0 && max_xinc == 0)
                            || (ix_pc_list[pc].max(min_xinc)
                                <= (ix_pc_list[pc] + nx - 1).min(max_xinc)))
                        && ((min_yinc == 0 && max_yinc == 0)
                            || (iy_pc_list[pc].max(min_yinc)
                                <= (iy_pc_list[pc] + ny - 1).min(max_yinc)))
                        && not_knock(
                            ix_pc_list[pc],
                            iy_pc_list[pc],
                            num_sec_read,
                            nx,
                            ny,
                            nx_overlap,
                            ny_overlap,
                            &ixko,
                            &iyko,
                            &izko,
                            knocks,
                        )
                    {
                        min_sub_xpiece = min_sub_xpiece.min(ix_pc_list[pc]);
                        min_sub_ypiece = min_sub_ypiece.min(iy_pc_list[pc]);
                        if if_float != 0 {
                            //
                            // if floating, need to read all sections to get stats
                            // find the minimum of the ratio (dmean-dmin) /(dmax-dmin)
                            ierr = 0;
                            iiu_read_binned(
                                1,
                                ipc - 1,
                                &mut array,
                                nx_bin,
                                ny_bin,
                                ix_start,
                                iy_start,
                                ibinning,
                                nx_bin,
                                ny_bin,
                                &mut temp_line,
                                len_temp,
                                &mut ierr,
                            );
                            if ierr != 0 {
                                exit_error("Reading image file");
                            }
                            if if_mean == 0 {
                                array_min_max_mean_fortran(
                                    &array,
                                    &nx_bin,
                                    &ny_bin,
                                    &1,
                                    &nx_bin,
                                    &1,
                                    &ny_bin,
                                    &mut tmp_min,
                                    &mut tmp_max,
                                    &mut sum,
                                );
                            } else {
                                array_min_max_mean_sd_fortran(
                                    &array,
                                    &nx_bin,
                                    &ny_bin,
                                    &1,
                                    &nx_bin,
                                    &1,
                                    &ny_bin,
                                    &mut tmp_min,
                                    &mut tmp_max,
                                    &mut tsum8,
                                    &mut tsumsq8,
                                    &mut avg,
                                    &mut sd,
                                );

                                sum8 += tsum8;
                                sumsq8 += tsumsq8;
                            }
                            dmin2 = minss(dmin2, tmp_min);
                            dmax2 = maxss(dmax2, tmp_max);
                        }
                        if_any_out += 1;
                        if if_renumber == 0 {
                            for i in 1..=ipiece_out {
                                let po = (i - 1) as usize;
                                if ix_piece_out[po] == ix_pc_list[pc]
                                    && iy_piece_out[po] == iy_pc_list[pc]
                                    && iz_piece_out[po] == iz_pc_list[pc]
                                {
                                    // write(*,'(/,a,i5,a,3i8)').  Fixed in
                                    // translation (BUGS.md, `edmont`): the
                                    // source prints the input file number
                                    // `ifile` as the output file number.
                                    print!(
                                        "\nERROR: EDMONT - YOU MUST RENUMBER Z; OUTPUT FILE \
                                         #{:>5} WOULD CONTAIN TWO SECTIONS WITH X,Y,Z OF{:>8}\
                                         {:>8}{:>8}\n",
                                        ifile_out,
                                        ix_piece_out[po],
                                        iy_piece_out[po],
                                        iz_piece_out[po]
                                    );
                                    let _ = std::io::stdout().flush();
                                    exit(1);
                                }
                            }
                        }
                        ipiece_out += 1;
                        let po = (ipiece_out - 1) as usize;
                        ix_piece_out[po] = ix_pc_list[pc];
                        iy_piece_out[po] = iy_pc_list[pc];
                        iz_piece_out[po] = iz_pc_list[pc];
                    }
                }
                //
                // Insist that some pieces  be found on this section!
                if if_any_out == 0 {
                    // write(*,'(/,a,i6,a,i5)')
                    print!(
                        "\nERROR: EDMONT - NO PIECES ARE INCLUDED FROM SECTION{:>6} WHICH IS IN \
                         THE LIST OF SECTIONS TO USE FROM FILE #{:>5}\n",
                        num_sec_read, ifile
                    );
                    let _ = std::io::stdout().flush();
                    exit(1);
                }
                if if_float != 0 {
                    let is = (ind_sec - 1) as usize;
                    dmin_sec[is] = dmin2;
                    dmax_sec[is] = dmax2;
                    dmin_out = minss(dmin_out, dmin2);
                    dmax_out = maxss(dmax_out, dmax2);
                    if if_mean != 0 {
                        sums_to_avg_sd_dbl(
                            sum8,
                            sumsq8,
                            nx_bin,
                            ny_bin * if_any_out,
                            &mut avg,
                            &mut sd,
                        );
                        avgsec[is] = avg;
                        sdsec[is] = sd;
                        zmin = minss(zmin, (dmin2 - avg) / sd);
                        zmax = maxss(zmax, (dmax2 - avg) / sd);
                    }
                }
                isec_out += 1;
                if isec_out > num_sec_out[(ifile_out - 1) as usize] {
                    isec_out = 1;
                    ifile_out += 1;
                    ipiece_out = 0;
                }
            }
            unsafe { iiu_close(1) };
        }
        //
        // Convert minimum coordinates to binned coordinates
        let i = (min_sub_xpiece - min_all_xpiece) / (nx - nx_overlap);
        min_sub_xpiece = i * (nx_bin - nxb_overlap) + min_all_xpiece / ibinning;
        let i = (min_sub_ypiece - min_all_ypiece) / (ny - ny_overlap);
        min_sub_ypiece = i * (ny_bin - nyb_overlap) + min_all_ypiece / ibinning;
    }
    let _ = &trunc_text;
    //
    // start looping over input images
    //
    time(&mut tim);
    b3d_date(&mut dat);
    let mut isec = 1_i32;
    isec_out = 1;
    ifile_out = 1;
    ipiece_out = 0;
    let mut ind_adoc_in: i32;
    for ifile in 1..=num_files_in {
        let fi = (ifile - 1) as usize;
        open_and_analyze_files(
            &in_file[fi],
            &piece_file_in[fi],
            &mut mxyz,
            &mut dmin_in,
            true,
            &mut mode,
            &mut extra_in,
            max_extra,
            &mut nbyte_extra_in,
            &mut nbyte_per_sec_in,
            &mut ix_pc_list,
            &mut iy_pc_list,
            &mut iz_pc_list,
            &mut num_pc_list,
            max_pieces,
            &mut listz,
            &mut num_in_zlist,
            &mut min_xpiece,
            &mut num_xpieces,
            &mut mx_overlap,
            &mut min_ypiece,
            &mut num_ypieces,
            &mut my_overlap,
            use_mdoc,
        );
        (nxyz, mxyz, nxyzst) = iiu_ret_size(1);
        let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);
        cell = iiu_ret_cell(1);
        delta = iiu_ret_delta(1);
        [x_origin, y_origin, z_origin] = iiu_ret_origin(1);
        //
        // get extra header information if any
        //
        if nbyte_extra_in > 0 {
            let mut data: Vec<u8> = Vec::new();
            let _ = iiu_ret_extended_data(1, &mut data);
            let count = (nbyte_extra_in as usize)
                .min(data.len())
                .min(extra_in.len());
            extra_in[..count].copy_from_slice(&data[..count]);
            [nbyte_per_sec_in, iflag_extra_in] = iiu_ret_extended_type(1);
        }
        ind_adoc_in = -1;
        if unsafe { iiu_file_type(1) } == 5 || use_mdoc {
            // The Fortran wrapper returns the index plus one.
            ind_adoc_in = unsafe { iiu_ret_adoc_index(1, 0, 1) };
            if ind_adoc_in >= 0 {
                ind_adoc_in += 1;
            }
            if ind_adoc_in < 0 {
                // write(*,'(/,a,a)'), after the library's own message.
                let _ = ImodFile::Stdout.flush();
                print!(
                    "\nERROR: EDMONT - COULD NOT OPEN AUTODOC INFORMATION FOR INPUT FILE {}\n",
                    in_file[fi]
                );
                let _ = std::io::stdout().flush();
                exit(1);
            }
        }
        //
        // get each section in input file
        for ilis in 1..=nlist[fi] {
            let ind_sec = ilis + list_ind[fi] - 1;
            let is = (ind_sec - 1) as usize;
            let num_sec_read = in_zlist[is];
            let mut if_any_out = 0_i32;
            for ipc in 1..=num_pc_list {
                let pc = (ipc - 1) as usize;
                if iz_pc_list[pc] == num_sec_read
                    && ((min_xinc == 0 && max_xinc == 0)
                        || (ix_pc_list[pc].max(min_xinc)
                            <= (ix_pc_list[pc] + nx - 1).min(max_xinc)))
                    && ((min_yinc == 0 && max_yinc == 0)
                        || (iy_pc_list[pc].max(min_yinc)
                            <= (iy_pc_list[pc] + ny - 1).min(max_yinc)))
                    && not_knock(
                        ix_pc_list[pc],
                        iy_pc_list[pc],
                        num_sec_read,
                        nx,
                        ny,
                        nx_overlap,
                        ny_overlap,
                        &ixko,
                        &iyko,
                        &izko,
                        knocks,
                    )
                {
                    ierr = 0;
                    iiu_read_binned(
                        1,
                        ipc - 1,
                        &mut array,
                        nx_bin,
                        ny_bin,
                        ix_start,
                        iy_start,
                        ibinning,
                        nx_bin,
                        ny_bin,
                        &mut temp_line,
                        len_temp,
                        &mut ierr,
                    );
                    if ierr != 0 {
                        exit_error("Reading image file");
                    }
                    let num_pixels: i64 = nx_bin as i64 * ny_bin as i64;
                    //
                    // calculate new min and max after rescaling under various
                    // possibilities
                    //
                    let mut optimal_in = optimal_max(mode);
                    let mut optimal_out = optimal_max(new_mode);
                    //
                    // set bottom of input range to 0 unless mode 1 or 2; set bottom
                    // of output range to 0 unless not changing modes
                    //
                    let mut bottom_in = 0.0_f32;
                    if dmin_in < 0. && (mode == 1 || mode == 2) {
                        bottom_in = -optimal_in;
                    }
                    let mut bottom_out = 0.0_f32;
                    if mode == new_mode {
                        bottom_out = bottom_in;
                    }
                    let mut rescale = false;
                    dmin2 = 0.;
                    dmax2 = 0.;
                    if if_float <= 0 {
                        //
                        // get min and max
                        array_min_max_mean_fortran(
                            &array,
                            &nx_bin,
                            &ny_bin,
                            &1,
                            &nx_bin,
                            &1,
                            &ny_bin,
                            &mut tmp_min,
                            &mut tmp_max,
                            &mut sum,
                        );
                        dmin2 = tmp_min;
                        dmax2 = tmp_max;
                    }
                    //
                    // for no float: if mode = 2, no rescale
                    //
                    if if_float <= 0 && new_mode != 2 && (mode != 2 || if_float < 0) {
                        //
                        // if within proper input range, rescale from input range to
                        // output range only if mode is changing
                        //
                        rescale = mode != new_mode && (mode != 2 || if_float < 0);
                        if if_float < 0 && rescale {
                            bottom_in = dmin_out;
                            optimal_in = dmax_out;
                        }
                        if tmp_min >= bottom_in && tmp_max <= optimal_in {
                            dmin2 = (tmp_min - bottom_in) * (optimal_out - bottom_out)
                                / (optimal_in - bottom_in)
                                + bottom_out;
                            dmax2 = (tmp_max - bottom_in) * (optimal_out - bottom_out)
                                / (optimal_in - bottom_in)
                                + bottom_out;
                        } else if rescale {
                            // :if outside proper range, tell user to start over
                            exit_error(
                                "Input data outside expected range. Start over, specifying \
                                 float to range",
                            );
                        }
                    } else if if_float > 0 {
                        // If floating: scale to a dmin2 that will knock out fraczero
                        // of the range after truncation to zero
                        dmin2 = -optimal_out * frac_zero / (1. - frac_zero);
                        rescale = true;
                        tmp_min = dmin_sec[is];
                        tmp_max = dmax_sec[is];
                        if if_mean == 0 {
                            // :float to range, new dmax2 is the max of the range
                            dmax2 = optimal_out;
                        } else {
                            // :float to mean, it's very hairy, first need mean again
                            let zmin_sec = (tmp_min - avgsec[is]) / sdsec[is];
                            let zmax_sec = (tmp_max - avgsec[is]) / sdsec[is];
                            dmin2 = (zmin_sec - zmin) * optimal_out / (zmax - zmin);
                            dmax2 = (zmax_sec - zmin) * optimal_out / (zmax - zmin);
                            dmin2 = maxss(0., dmin2);
                            // but in case of problems, just limit dmax2 to the range
                            dmax2 = minss(dmax2, optimal_out);
                        }
                    }
                    let mut dmean2 = 0.0_f64;
                    // set up minimum value to output based on mode
                    let den_out_min: f32;
                    if new_mode == 1 {
                        den_out_min = -32767.;
                    } else if new_mode == 2 {
                        den_out_min = -1.0e30;
                        optimal_out = 1.0e30;
                    } else {
                        den_out_min = 0.;
                    }
                    //
                    if rescale {
                        // if scaling, set up equation, scale and compute new mean
                        let scale_fac = (dmax2 - dmin2) / (tmp_max - tmp_min);
                        let constant = dmin2 - scale_fac * tmp_min;
                        dmin2 = 1.0e20;
                        dmax2 = -1.0e20;
                        for i8 in 0..num_pixels as usize {
                            let mut den = scale_fac * array[i8] + constant;
                            if den < den_out_min {
                                // ntrunclo=ntrunclo+1
                                den = den_out_min;
                            } else if den > optimal_out {
                                // ntrunchi=ntrunchi+1
                                den = optimal_out;
                            }
                            array[i8] = den;
                            dmean2 += den as f64;
                            dmin2 = minss(dmin2, den);
                            dmax2 = maxss(dmax2, den);
                        }
                    } else {
                        // if not scaling, just need new mean
                        for i8 in 0..num_pixels as usize {
                            dmean2 += array[i8] as f64;
                        }
                    }
                    dmean2 /= num_pixels as f64;
                    // print *,'frame', isec - 1, ': min&max before and after, mean:'
                    print!(" frame{:>12} : min&max before and after, mean:\n", isec - 1);
                    // write(*,'(5f10.2)')
                    print!(
                        "{}{}{}{}{}\n",
                        format_f(tmp_min as f64, 10, 2),
                        format_f(tmp_max as f64, 10, 2),
                        format_f(dmin2 as f64, 10, 2),
                        format_f(dmax2 as f64, 10, 2),
                        format_f(dmean2, 10, 2)
                    );
                    // see if need to open an output file
                    if ipiece_out == 0 {
                        //
                        // Create output file, transfer header from currently open
                        // file, fix it enough to get going
                        imopen(2, &out_file[(ifile_out - 1) as usize], "NEW");
                        iiu_trans_header(2, 1);
                        iiu_alt_mode(2, new_mode);
                        nxyz2[0] = nx_bin;
                        nxyz2[1] = ny_bin;
                        nxyz2[2] = 1;
                        iiu_alt_size(2, &nxyz2, &nxyzst);
                        //
                        // adjust extra header information if current file has it
                        //
                        nbyte_extra_out = 0;
                        if nbyte_extra_in > 0 && all_have_header_coords {
                            nbyte_per_sec_out = nbyte_per_sec_in;
                            nbyte_extra_out = num_sec_out[(ifile_out - 1) as usize]
                                * max_byte_extra_in
                                * max_num_xpieces
                                * max_num_ypieces;
                            iiu_alt_num_extended(2, nbyte_extra_out);
                            unsafe { iiu_set_position(2, 0, 0) };
                            ind_xout = 0;
                        }
                        //
                        // Open autodoc for output file if appropriate
                        // Transfer global data.
                        out_doc_changed = false;
                        ind_adoc_out = -1;
                        if (b3d_output_file_type() == 5 || use_mdoc) && ind_adoc_in >= 0 {
                            ind_adoc_out = unsafe { iiu_ret_adoc_index(2, 0, -1) };
                            if ind_adoc_out >= 0 {
                                ind_adoc_out += 1;
                            }
                            if ind_adoc_out <= 0 {
                                exit_error("Cannot get autodoc index for output");
                            }
                            set_current_adoc_or_exit(ind_adoc_in, "input");
                            if adoc_transfer_section(
                                global_name,
                                0,
                                ind_adoc_out - 1,
                                Some(global_name),
                                0,
                            ) != 0
                            {
                                exit_error("Transferring global data between autodocs");
                            }
                        }
                        // 301 format('EDMONT: Images transferred',a18,t57,a9,2x,a8)
                        title.fill(b' ');
                        let head = b"EDMONT: Images transferred";
                        title[..head.len()].copy_from_slice(head);
                        title[26..44].copy_from_slice(&float_text[..18]);
                        title[56..65].copy_from_slice(&dat);
                        title[67..75].copy_from_slice(&tim);
                        dmax = -100000.;
                        dmin = 100000.;
                        dmean = 0.;
                    }
                    //
                    dmin = minss(dmin, dmin2);
                    dmax = maxss(dmax, dmax2);
                    dmean = (dmean as f64 + dmean2) as f32;
                    //
                    unsafe { iiu_write_section(2, array.as_mut_ptr().cast()) };
                    isec += 1;
                    ipiece_out += 1;
                    let po = (ipiece_out - 1) as usize;
                    let i = (ix_pc_list[pc] - min_all_xpiece) / (nx - nx_overlap);
                    ix_piece_out[po] = i * (nx_bin - nxb_overlap) + min_all_xpiece / ibinning;
                    let i = (iy_pc_list[pc] - min_all_ypiece) / (ny - ny_overlap);
                    iy_piece_out[po] = i * (ny_bin - nyb_overlap) + min_all_ypiece / ibinning;
                    iz_piece_out[po] = iz_pc_list[pc];
                    if if_renumber != 0 {
                        iz_piece_out[po] = isec_out - 1;
                    }
                    if if_shift_xy != 0 {
                        ix_piece_out[po] -= min_sub_xpiece;
                        iy_piece_out[po] -= min_sub_ypiece;
                    }
                    if_any_out += 1;
                    //
                    // transfer extra header bytes if present
                    //
                    if nbyte_extra_out != 0 && ind_xout < nbyte_extra_out {
                        let num_byte_copy =
                            nbyte_per_sec_out.min(nbyte_per_sec_in).min(nbyte_extra_in);
                        let nbyte_clear = nbyte_per_sec_out - num_byte_copy;
                        for i in 1..=num_byte_copy {
                            ind_xout += 1;
                            // Bytes past the extra header read here as zero
                            // (the source reads its array past the data).
                            extra_out[(ind_xout - 1) as usize] = extra_in
                                .get(((ipc - 1) * nbyte_per_sec_in + i - 1) as usize)
                                .copied()
                                .unwrap_or(0);
                        }
                        for _ in 1..=nbyte_clear {
                            ind_xout += 1;
                            extra_out[(ind_xout - 1) as usize] = 0;
                        }
                        //
                        // Need to replace values if there is renumbering, shifting,
                        // binning, or the piece coordinates actually came from pl file
                        if (if_renumber != 0
                            || if_shift_xy != 0
                            || ibinning > 1
                            || !piece_file_in[fi].is_empty())
                            && num_byte_copy >= 6
                        {
                            // `integer*2 temp`; `call move(extraOut(ind), temp, 2)`
                            let mut temp = iz_piece_out[po] as i16;
                            let mut ind = ind_xout + 5 - nbyte_per_sec_out;
                            if iflag_extra_in % 2 != 0 {
                                ind += 2;
                            }
                            let mut put = |at: i32, value: i16| {
                                let at = (at - 1) as usize;
                                extra_out[at..at + 2].copy_from_slice(&value.to_ne_bytes());
                            };
                            if if_renumber != 0 || !piece_file_in[fi].is_empty() {
                                put(ind, temp);
                            }
                            if if_shift_xy != 0 || ibinning > 1 || !piece_file_in[fi].is_empty() {
                                temp = ix_piece_out[po] as i16;
                                put(ind - 4, temp);
                                temp = iy_piece_out[po] as i16;
                                put(ind - 2, temp);
                            }
                        }
                    }
                    //
                    // Transfer an adoc section and set the piece coordinates
                    if ind_adoc_in > 0 && ind_adoc_out > 0 {
                        set_current_adoc_or_exit(ind_adoc_in, "input");
                        let mut ind_sect_in = adoc_lookup_by_name_value(zvalue_name, ipc - 1);
                        if ind_sect_in >= 0 {
                            ind_sect_in += 1;
                        }
                        if ind_sect_in > 0 {
                            let mut nchar = 0_i32;
                            int_iwrite(&mut list_string, ipiece_out - 1, &mut nchar);
                            let new_name = fortran_string(&list_string);
                            if adoc_transfer_section(
                                zvalue_name,
                                ind_sect_in - 1,
                                ind_adoc_out - 1,
                                Some(new_name.as_bytes()),
                                1,
                            ) != 0
                            {
                                exit_error("Transferring section data between autodocs");
                            }
                            out_doc_changed = true;
                            set_current_adoc_or_exit(ind_adoc_out, "output");
                            let mut ind_sect_out =
                                adoc_lookup_by_name_value(zvalue_name, ipiece_out - 1);
                            if ind_sect_out >= 0 {
                                ind_sect_out += 1;
                            }
                            if ind_sect_out < 0
                                || adoc_set_three_integers(
                                    zvalue_name,
                                    ind_sect_out - 1,
                                    b"PieceCoordinates",
                                    ix_piece_out[po],
                                    iy_piece_out[po],
                                    iz_piece_out[po],
                                ) != 0
                            {
                                exit_error("Setting piece coordinates in output autodoc");
                            }
                        }
                    }
                }
            }
            if if_any_out > 0 {
                isec_out += 1;
            }
            // see if need to close stack file
            if isec_out > num_sec_out[(ifile_out - 1) as usize]
                || (ifile == num_files_in && ilis == nlist[fi])
            {
                let fo = (ifile_out - 1) as usize;
                // set new size, keep old nxyzst
                if !piece_file_out[fo].is_empty() {
                    let mut unit3 = dopen(3, &piece_file_out[fo], "new", "f");
                    // write(3, '(2i9,i7)')
                    let mut text = String::new();
                    for i in 0..ipiece_out as usize {
                        text.push_str(&format!(
                            "{:>9}{:>9}{:>7}\n",
                            ix_piece_out[i], iy_piece_out[i], iz_piece_out[i]
                        ));
                    }
                    let _ = unit3.write_all(text.as_bytes());
                    drop(unit3);
                }
                nxyz2[2] = ipiece_out;
                iiu_alt_size(2, &nxyz2, &nxyzst);
                // if mxyz=nxyz, keep this relationship
                if mxyz[0] == nx && mxyz[1] == ny && mxyz[2] == nz {
                    mxyz2[0] = nx_bin;
                    mxyz2[1] = ny_bin;
                    mxyz2[2] = ipiece_out;
                    iiu_alt_sample(2, &mxyz2);
                } else {
                    // Fixed in translation (BUGS.md, `edmont`): the source
                    // scales the cell below by `mxyz2`, which it sets only in
                    // the branch above, so for an input whose sampling
                    // differs from its size the cell comes from whatever
                    // `mxyz2` held.  The output keeps the input's sampling,
                    // which is what the scaling then has to use.
                    mxyz2 = mxyz;
                }
                // keep delta the same by scaling cell size from change in mxyz
                for i in 1..=3_usize {
                    cell2[i - 1] = mxyz2[i - 1] as f32 * (cell[i - 1] / mxyz[i - 1] as f32);
                    if i < 3 {
                        cell2[i - 1] *= ibinning as f32;
                    }
                    cell2[i + 2] = 90.;
                }
                iiu_alt_cell(2, &cell2);
                //
                // adjust origin if shifting piece coords to 0
                if if_shift_xy != 0 {
                    iiu_alt_origin(
                        2,
                        &[
                            x_origin - (ibinning * min_sub_xpiece) as f32 * delta[0],
                            y_origin - (ibinning * min_sub_ypiece) as f32 * delta[1],
                            z_origin,
                        ],
                    );
                }
                if nbyte_extra_out > 0 {
                    let _ = iiu_alt_extended_data(2, &extra_out[..nbyte_extra_out as usize]);
                }
                dmean /= ipiece_out as f32;

                if out_doc_changed && unsafe { iiu_file_type(2) } != 5 {
                    set_current_adoc_or_exit(ind_adoc_out, "output");
                    if adoc_write(format!("{}.mdoc", out_file[fo]).as_bytes()) != 0 {
                        exit_error("Writing mdoc file for output file");
                    }
                    adoc_clear(ind_adoc_out - 1);
                }

                //
                iiu_write_header(2, &title, 1, dmin, dmax, dmean);
                unsafe { iiu_close(2) };
                ipiece_out = 0;
                isec_out = 1;
                ifile_out += 1;
            }
        }
        unsafe { iiu_close(1) };
    }
    //
    let _ = std::io::stdout().flush();
    exit(0);
}

/// Stores `text` into a blank-padded Fortran `character` variable.
fn set_text(dest: &mut [u8], text: &str) {
    dest.fill(b' ');
    let count = text.len().min(dest.len());
    dest[..count].copy_from_slice(&text.as_bytes()[..count]);
}

/// Original `notKnock` (`edmont.f90:705`).
///
/// Tests for whether a piece should be included given its coordinates and
/// the set of points to knock out.
pub fn not_knock(
    ix_piece: i32,
    iy_piece: i32,
    iz_piece: i32,
    nx: i32,
    ny: i32,
    nx_overlap: i32,
    ny_overlap: i32,
    ixko: &[i32],
    iyko: &[i32],
    izko: &[i32],
    knocks: i32,
) -> bool {
    let nx_border = nx_overlap.min(nx / 4);
    let ny_border = ny_overlap.min(ny / 4);
    for i in 0..knocks as usize {
        if izko[i] == iz_piece
            && ix_piece + nx_border <= ixko[i]
            && ix_piece + nx - nx_border > ixko[i]
            && iy_piece + ny_border <= iyko[i]
            && iy_piece + ny - ny_border > iyko[i]
        {
            return false;
        }
    }
    true
}

/// Original `read_pl_or_header` (`edmont.f90:725`).
///
/// Given a piece list file, read the piece list; given no piece file,
/// attempt to read piece coordinates from image header.
pub fn read_pl_or_header(
    piece_file_in: &str,
    in_file: &str,
    extra_in: &mut [u8],
    max_extra: i32,
    nbyte_extra_in: &mut i32,
    nbyte_per_sec_in: &mut i32,
    ix_pc_list: &mut [i32],
    iy_pc_list: &mut [i32],
    iz_pc_list: &mut [i32],
    num_pc_list: &mut i32,
    lim_pc_list: i32,
    use_mdoc: bool,
) {
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut mode = 0_i32;
    let (mut dmin_in, mut dmaxin, mut dmean_in) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut montage, mut num_sect, mut i_type_adoc) = (0_i32, 0_i32, 0_i32);
    let iflag_extra_in: i32;
    //
    *nbyte_extra_in = 0;
    *nbyte_per_sec_in = 0;
    // Fixed in translation (BUGS.md, `edmont`): when there is no piece list
    // file and the autodoc has no image metadata, the source never sets the
    // piece count before testing it below and returning it, so the count is
    // whatever the caller's variable held (the previous file's count).  No
    // source of pieces means no pieces.
    *num_pc_list = 0;
    let use_file = !piece_file_in.is_empty() && piece_file_in != "none";
    //
    // Use piece list as primary source if defined
    if use_file {
        read_piece_list2(
            piece_file_in,
            ix_pc_list,
            iy_pc_list,
            iz_pc_list,
            num_pc_list,
            lim_pc_list,
        );
    }
    //
    // Open file and autodoc if indicated or for HDF file
    ialprt(false);
    imopen(4, in_file, "RO");
    unsafe {
        irdhdr(
            4,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin_in,
            &mut dmaxin,
            &mut dmean_in,
        );
    }
    let use_autodoc = use_mdoc || unsafe { iiu_file_type(4) } == 5;
    if use_autodoc {
        // The Fortran wrapper returns the index plus one.
        let mut ind_adoc = unsafe { iiu_ret_adoc_index(4, 0, 1) };
        if ind_adoc >= 0 {
            ind_adoc += 1;
        }
        if ind_adoc < 0 {
            // write(*,'(/,a,a)'), after the library's own message.
            let _ = ImodFile::Stdout.flush();
            print!(
                "\nERROR: EDMONT - Could not open autodoc information for input file {}\n",
                in_file
            );
            let _ = std::io::stdout().flush();
            exit(1);
        }
        set_current_adoc_or_exit(ind_adoc, "input");
        if adoc_get_image_meta_info(&mut montage, &mut num_sect, &mut i_type_adoc) == 0 && !use_file
        {
            get_metadata_pieces_fortran(
                ind_adoc,
                i_type_adoc,
                nxyz[2],
                ix_pc_list,
                iy_pc_list,
                iz_pc_list,
                lim_pc_list,
                num_pc_list,
            );
        }
        if unsafe { iiu_file_type(4) } != 5 {
            adoc_clear(ind_adoc - 1);
        }
    }
    //
    // get extra header information if any
    //
    *nbyte_extra_in = iiu_ret_num_extended(4);
    if *nbyte_extra_in > 0 {
        if *nbyte_extra_in > max_extra && !use_file {
            // write(*,'(/,a,a,a)')
            print!(
                "\nERROR: EDMONT - No piece list file was given for input file {} and the extra \
                 header data are too large for the array\n",
                in_file
            );
            let _ = std::io::stdout().flush();
            exit(1);
        } else {
            [*nbyte_per_sec_in, iflag_extra_in] = iiu_ret_extended_type(4);
            if use_file {
                if (iflag_extra_in / 2) % 2 == 0 {
                    *nbyte_extra_in = 0;
                }
            } else {
                let mut data: Vec<u8> = Vec::new();
                let _ = iiu_ret_extended_data(4, &mut data);
                let count = (*nbyte_extra_in as usize)
                    .min(data.len())
                    .min(extra_in.len());
                extra_in[..count].copy_from_slice(&data[..count]);
                get_extra_header_pieces_fortran(
                    &extra_in[..*nbyte_extra_in as usize],
                    *nbyte_extra_in,
                    *nbyte_per_sec_in,
                    iflag_extra_in,
                    nxyz[2],
                    ix_pc_list,
                    iy_pc_list,
                    iz_pc_list,
                    num_pc_list,
                    lim_pc_list,
                );
            }
        }
    }
    if !(use_file || use_autodoc) && (*nbyte_extra_in == 0 || *num_pc_list == 0) {
        // write(*,'(/,a,a,a)')
        print!(
            "\nERROR: EDMONT - No piece list file was given for input file {} and the header \
             does not contain piece coordinates\n",
            in_file
        );
        let _ = std::io::stdout().flush();
        exit(1);
    }
    unsafe { iiu_close(4) };
}

/// Original `openAndAnalyzeFiles` (`edmont.f90:798`).
///
/// Open an image file, get piece coordinates one way or another, analyze the
/// list of Z values in the file, and determine the montage characteristics
/// in each dimension.
pub fn open_and_analyze_files(
    image_file: &str,
    piece_file: &str,
    nxyz: &mut [i32; 3],
    dmin2: &mut f32,
    printing: bool,
    mode: &mut i32,
    extra_in: &mut [u8],
    max_extra: i32,
    nbyte_extra_in: &mut i32,
    nbyte_per_sec_in: &mut i32,
    ix_pc_list: &mut [i32],
    iy_pc_list: &mut [i32],
    iz_pc_list: &mut [i32],
    num_pc_list: &mut i32,
    lim_pc_list: i32,
    listz: &mut [i32],
    num_in_zlist: &mut i32,
    min_xpiece: &mut i32,
    num_xpieces: &mut i32,
    nx_overlap: &mut i32,
    min_ypiece: &mut i32,
    num_ypieces: &mut i32,
    ny_overlap: &mut i32,
    use_mdoc: bool,
) {
    let mut mxyz = [0_i32; 3];
    let (mut dmax2, mut dmean2) = (0.0_f32, 0.0_f32);
    //
    // Read the piece data first in case there is a problem opening the
    // same file on two channels
    read_pl_or_header(
        piece_file,
        image_file,
        extra_in,
        max_extra,
        nbyte_extra_in,
        nbyte_per_sec_in,
        ix_pc_list,
        iy_pc_list,
        iz_pc_list,
        num_pc_list,
        lim_pc_list,
        use_mdoc,
    );
    ialprt(printing);
    imopen(1, image_file, "RO");
    unsafe {
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            mode,
            dmin2,
            &mut dmax2,
            &mut dmean2,
        );
    }

    let npc = (*num_pc_list).max(0) as usize;
    let mut nlz = 0_usize;
    fill_listz(&iz_pc_list[..npc], listz, &mut nlz);
    *num_in_zlist = nlz as i32;
    checklist(
        &ix_pc_list[..npc],
        1,
        nxyz[0],
        min_xpiece,
        num_xpieces,
        nx_overlap,
    );
    checklist(
        &iy_pc_list[..npc],
        1,
        nxyz[1],
        min_ypiece,
        num_ypieces,
        ny_overlap,
    );
}
