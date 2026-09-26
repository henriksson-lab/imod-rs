//! Translation of `IMOD/flib/image/extracttilts.f90`.
//!
//! The whole unit is the main program, [`extracttilts`].  Library calls go to
//! the Fortran wrappers (`extraheader.c`'s `*_fortran`, which take 1-based
//! autodoc indexes and print and exit on error; `adoc_fwrap.c`'s 1-based
//! `AdocSetCurrent`/`AdocGetSectionName` and 1-based returned index from
//! `AdocOpenImageMetadata`/`iiuRetAdocIndex`).  Unit 6 is written with the
//! gfortran editing the source asks for: `Fw.d` and `Iw` (overflow is `w`
//! asterisks, a leading zero is dropped when that is what makes the value
//! fit), and a formatted `write` of an implied-do list whose format has one
//! record's worth of descriptors writes one record per group, or one empty
//! record when the list is empty.  List-directed `print *` puts a blank
//! before the record, writes an integer as `I11` after a blank separator, and
//! puts a blank between a number and a following string.

use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::libcfshr::autodoc::{
    ADOC_ZVALUE_NAME, adoc_get_image_meta_info, adoc_get_number_of_sections, adoc_get_section_name,
    adoc_open_image_metadata, adoc_order_write_by_value, adoc_set_current, adoc_write,
};
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::extraheader::{
    get_extra_header_items_fortran, get_extra_header_pieces_fortran, get_metadata_by_key_fortran,
    get_metadata_items_fortran, get_metadata_pieces_fortran, get_metadata_weighting_doses_fortran,
};
use crate::imod::libcfshr::parse_params::{pip_done, pip_get_boolean, pip_get_float};
use crate::imod::libcfshr::robuststat::rssortindexedfloats;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_NLABELS};
use crate::imod::libiimod::unit_fileio::{
    iiu_alt_print, iiu_close, iiu_file_type, iiu_ret_adoc_index,
};
use crate::imod::libiimod::unit_header::{
    iiu_ret_extended_data, iiu_ret_extended_type, iiu_ret_labels, iiu_ret_num_extended,
};
use std::io::{BufWriter, Write};

/// `parameter (numOptions = 21)` (`extracttilts.f90:46`).
const NUM_OPTIONS: i32 = 21;
/// Fallback PIP table `options(1)` (`extracttilts.f90:48-56`).
const OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@mdoc:MdocMetadataFile:B:@\
other:OtherMetadataFile:FN:@tilts:TiltAngles:B:@stage:StagePositions:B:@\
mag:Magnifications:B:@intensities:Intensities:B:@exp:ExposureDose:B:@\
mintilt:MinimumTiltAndStartAngle:B:@fixed:FixedDoseForInfoFile:F:@\
aretomo:DoseFileForAreTomo:B:@prior:PriorDose:F:@camera:CameraExposure:B:@\
pixel:PixelSpacing:B:@defocus:Defocus:B:@key:KeyName:CH:@\
attrib:AttributesInHDFfile:B:@warn:WarnIfTiltsSuspicious:B:@all:AllPieces:B:@\
help:usage:B:";

/// Original program `extracttilts` (`extracttilts.f90:13`).
///
/// EXTRACTTILTS will extract tilt angles or other per-section information
/// from the header of an image file, if they are present, and produce a file
/// with a list of the values.
pub fn extracttilts() {
    let max_extra: i32;
    let max_tilts: i32;
    let max_piece: i32;
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut tilt: Vec<f32>;
    let mut val2: Vec<f32>;
    let mut array: Vec<u8>;
    let mut sec_dose: Vec<f32> = Vec::new();
    let mut cumul_dose: Vec<f32> = Vec::new();
    let mut ix_piece: Vec<i32>;
    let mut iy_piece: Vec<i32>;
    let mut iz_piece: Vec<i32>;
    let mut ind_time: Vec<i32> = Vec::new();
    let mut val_string: Vec<String> = Vec::new();
    //
    let mut in_file = String::new();
    let mut out_file = String::new();
    let mut metafile = String::new();
    let mut key_name = String::new();
    let mut temp_label_str: Vec<u8>;
    let mut start_type: &str = "";
    let mut all_labels = [[0_u8; MRC_LABEL_SIZE]; MRC_NLABELS];
    let mut type_text: [String; 10] = [
        "tilt angle".to_owned(),
        " ".to_owned(),
        "stage position".to_owned(),
        "magnification".to_owned(),
        "intensity value".to_owned(),
        "exposure dose".to_owned(),
        "pixel spacing".to_owned(),
        "defocus".to_owned(),
        "exposure time".to_owned(),
        "image time stamp".to_owned(),
    ];
    //
    let mut ierr: i32;
    let mut if_tilt: i32;
    let mut if_mag: i32;
    let mut if_stage: i32;
    let mut num_pieces: i32;
    let mut nbytes = 0_i32;
    let mut iflags = 0_i32;
    let maxz: i32;
    let mut iunit_out: i32;
    let mut mode = 0_i32;
    // Uninitialised in the source until an option sets it (it always is).
    let mut itype = 1_i32;
    let mut if_c2: i32;
    let mut num_tilts: i32;
    let mut num_extra_bytes = 0_i32;
    let mut num_tilt_out: i32;
    let mut if_dose: i32;
    let mut if_warn: i32;
    let mut if_all: i32;
    let mut if_cam_exp: i32;
    let mut if_defocus: i32;
    let mut if_pixel: i32;
    let mut num_found = 0_i32;
    let if_add_mdoc: i32;
    // Uninitialised in the source on the paths that do not open or look up
    // an autodoc; every such path then sets it to -4 (`numSect .ne. nz`).
    let mut ind_adoc = 0_i32;
    let mut i_type_adoc = 0_i32;
    let mut num_sect: i32;
    let mut montage: i32;
    let mut if_mdoc: i32;
    let if_no_image: i32;
    let mut if_key: i32;
    let mut if_attrib: i32;
    let mut if_info_file = 0_i32;
    let mut if_are_tomo: i32;
    let mut if_info_from_fei: i32;
    let mut num_times: i32;
    let mut if_min_tilt: i32;
    let mut num_labels = 0_i32;
    // Uninitialised in the source when there are no tilts to scan.
    let mut min_view = 0_i32;
    let mut iv_start = 0_i32;
    let mut ind: i32;
    let mut jnd: i32;
    let mdoc_primary: bool;
    let mut hdf_file: bool;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut fixed_dose: f32;
    let mut prior_dose: f32;
    let mut tilt_min: f32;
    let mut start_angle: f32;
    let mut start_min: f32;
    //
    let pipinput: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);

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
    // The Fortran wrapper `pipgetstring`: the variable is left untouched
    // unless the option is found.
    // `c2fString` there copies at most the variable's declared length
    // (extracttilts.f90:21: `character*320 metafile, keyName`); a longer entry fails with `In PipGetString, string is too
    // long for character variable`, which exits under the exit prefix.
    let get_string = |option: &[u8], length: usize, string: &mut String| -> i32 {
        let mut record = vec![b' '; length];
        let err = crate::imod::libcfshr::pip_fwrap::pipgetstring_(option, &mut record);
        if err == 0 {
            *string = crate::imod::libcfshr::b3dutil::fortran_string(&record);
        }
        err
    };
    let blank = |string: &str| string.bytes().all(|b| b == b' ');
    //
    out_file.clear();
    if_mag = 0;
    if_stage = 0;
    if_c2 = 0;
    if_tilt = 0;
    if_dose = 0;
    if_cam_exp = 0;
    if_pixel = 0;
    if_defocus = 0;
    if_warn = 0;
    num_pieces = 0;
    if_all = 0;
    if_min_tilt = 0;
    if_mdoc = 0;
    let mut if_add_mdoc_v = 1_i32;
    metafile.clear();
    if_key = 0;
    if_info_from_fei = 0;
    if_are_tomo = 0;
    fixed_dose = 0.;
    prior_dose = 0.;
    num_sect = 0;
    montage = 0;
    start_angle = 1.0e10;
    hdf_file = false;
    if_attrib = 0;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "extracttilts",
        "ERROR: EXTRACTTILTS - ",
        true,
        1,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pipinput = num_opt_arg + num_non_opt_arg > 0;

    if_no_image = pip_get_in_out_file("InputFile", 1, "Image input file", &mut in_file, 320);
    ierr = pip_get_in_out_file(
        "OutputFile",
        2,
        "Name of output file, or return to print out values",
        &mut out_file,
        320,
    );
    //
    if pipinput {
        ierr = pip_get_boolean(b"AllPieces", &mut if_all);
        ierr = pip_get_boolean(b"TiltAngles", &mut if_tilt);
        ierr = pip_get_boolean(b"Magnifications", &mut if_mag);
        ierr = pip_get_boolean(b"StagePositions", &mut if_stage);
        ierr = pip_get_boolean(b"Intensities", &mut if_c2);
        ierr = pip_get_boolean(b"ExposureDose", &mut if_dose);
        ierr = pip_get_boolean(b"CameraExposure", &mut if_cam_exp);
        ierr = pip_get_boolean(b"Defocus", &mut if_defocus);
        ierr = pip_get_boolean(b"PixelSpacing", &mut if_pixel);
        ierr = pip_get_boolean(b"MinimumTiltAndStartAngle", &mut if_min_tilt);
        ierr = pip_get_boolean(b"MdocMetadataFile", &mut if_mdoc);
        ierr = pip_get_boolean(b"WarnIfTiltsSuspicious", &mut if_warn);
        ierr = pip_get_boolean(b"AttributesInHDFfile", &mut if_attrib);
        if_add_mdoc_v = get_string(b"OtherMetadataFile", 320, &mut metafile);
        if if_mdoc != 0 && !blank(&metafile) {
            exit_error("You cannot enter both -mdoc and -other");
        }
        if get_string(b"KeyName", 320, &mut key_name) == 0 {
            if_key = 1;
            itype = 2;
            // `typeText` is `character*40`
            let mut end = key_name.len().min(40);
            while !key_name.is_char_boundary(end) {
                end -= 1;
            }
            type_text[1] = key_name[..end].to_owned();
        }
        if_info_from_fei = 1 - pip_get_float(b"FixedDoseForInfoFile", &mut fixed_dose);
        ierr = pip_get_boolean(b"DoseFileForAreTomo", &mut if_are_tomo);
        if_info_file = if_info_from_fei + if_are_tomo;
        if if_info_file > 0 {
            ierr = pip_get_float(b"PriorDose", &mut prior_dose);
        }
        ierr = if_tilt
            + if_mag
            + if_stage
            + if_c2
            + if_dose
            + if_defocus
            + if_cam_exp
            + if_pixel
            + if_key
            + if_min_tilt;
        if if_info_file > 0 {
            ierr = ierr + 1;
        }

        if if_attrib > 0 && ierr > 0 {
            exit_error("You cannot specify anything to extract when writing attributes to a file");
        }
        if ierr == 0 {
            if_tilt = 1;
        }
        if ierr > 1 {
            exit_error("You must enter only one option for data to extract");
        }
        if if_tilt != 0 {
            itype = 1;
        }
        if if_stage != 0 {
            itype = 3;
        }
        if if_mag != 0 {
            itype = 4;
        }
        if if_c2 != 0 {
            itype = 5;
        }
        if if_dose != 0 {
            itype = 6;
        }
        if if_pixel != 0 {
            itype = 7;
        }
        if if_defocus != 0 {
            itype = 8;
        }
        if if_cam_exp != 0 {
            itype = 9;
        }
        if if_info_from_fei != 0 {
            itype = 10;
        }
        if if_are_tomo != 0 {
            itype = 1;
        }
        if if_min_tilt != 0 {
            itype = 1;
        }
        if if_min_tilt != 0 && blank(&out_file) {
            iiu_alt_print(0);
        }
    } else {
        if_tilt = 1;
        itype = 1;
    }
    if_add_mdoc = if_add_mdoc_v;

    if if_attrib > 0 && if_no_image != 0 {
        exit_error("You must enter an input image file with -attrib");
    }
    if if_attrib > 0 && (if_mdoc != 0 || !blank(&metafile)) {
        exit_error("You cannot enter -mdoc or -other with -attrib");
    }
    if if_attrib > 0 && blank(&out_file) {
        exit_error("You must enter an output file to extract attributes");
    }

    // Require an image unless -other is used with an mdoc file
    if blank(&metafile) && if_no_image != 0 {
        exit_error("You must enter either an input image file or a metadata file with -other");
    }
    if if_info_from_fei != 0 && (!blank(&metafile) || if_mdoc != 0) {
        exit_error(
            "You cannot produce a dose information file for an FEI file when using an mdoc file",
        );
    }
    pip_done();

    max_extra = if if_no_image == 0 {
        let mut max_extra_v = 40;
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
        //
        let file_type = unsafe { iiu_file_type(1) };
        hdf_file = file_type == 5 || file_type == 7;
        if if_attrib > 0 && file_type != 5 {
            exit_error("The input image file is not an HDF file; cannot extract attributes");
        }
        if hdf_file && if_add_mdoc != 0 {
            // `iiuretadocindex` returns the index plus one.
            ind_adoc = unsafe { iiu_ret_adoc_index(1, 0, 0) };
            if ind_adoc >= 0 {
                ind_adoc += 1;
            }
            if ind_adoc <= 0 {
                exit_error("Getting autodoc index for HDF or IDOC file");
            }
            if adoc_set_current(ind_adoc - 1).is_err() {
                exit_error("Setting autodoc structure of HDF or IDOC file as current autodoc");
            }
            num_sect = nxyz[2];
            //
            // Write out autodoc and exit if requested
            if if_attrib > 0 {
                // `AdocGetStandardNames(globalName, zvalueName)`
                // (`adoc_fwrap.c:664`): the two standard names, of which only
                // the Z-value one is used.
                let zvalue_name = ADOC_ZVALUE_NAME;
                if adoc_order_write_by_value(Some(zvalue_name)) != 0
                    || adoc_write(out_file.trim_end_matches(' ').as_bytes()) != 0
                {
                    exit_error("Writing autodoc structure to file");
                }
                println!(" Attributes written to {}", out_file.trim_end_matches(' '));
                exit(0);
            }
            if adoc_get_image_meta_info(&mut montage, &mut num_sect, &mut i_type_adoc) < 0
                || num_sect < nxyz[2]
            {
                exit_error("This file does not have metadata about each section");
            }
        } else {
            num_extra_bytes = iiu_ret_num_extended(1);
            [nbytes, iflags] = iiu_ret_extended_type(1);
            max_extra_v = num_extra_bytes + 1024;
            if if_info_from_fei != 0 && nbytes != -3 {
                exit_error(
                    "The input image file does not have an FEI extended header with time stamps",
                );
            }
        }
        max_extra_v
    } else {
        40
    };

    //
    // Always try to open a metadata file, and check for errors if it was required
    if blank(&metafile) {
        metafile = in_file.clone();
    }
    if (!hdf_file || if_add_mdoc == 0) && (if_info_from_fei == 0 || if_are_tomo != 0) {
        // `adocopenimagemetadata` trims the name and returns the index plus one.
        ind_adoc = adoc_open_image_metadata(
            metafile.trim_end_matches(' ').as_bytes(),
            if_add_mdoc,
            &mut montage,
            &mut num_sect,
            &mut i_type_adoc,
        );
        if ind_adoc >= 0 {
            ind_adoc += 1;
        }
    }
    mdoc_primary = if_mdoc != 0 || if_add_mdoc == 0 || hdf_file;
    if mdoc_primary {
        if ind_adoc == -1 {
            exit_error("Opening or reading the metadata file");
        }
        if ind_adoc == -2 {
            exit_error("The metadata file does not exist");
        }
        if ind_adoc == -3 {
            exit_error("The autodoc file is not a recognized form of image metadata file");
        }
        if if_no_image > 0 {
            nxyz[2] = num_sect;
        }
        if num_sect != nxyz[2] {
            exit_error("The image and metadata files do not have the same number of sections");
        }
    }
    let nz = nxyz[2];

    // If the adoc is just a fallback, just disallow it if it doesn't match nz
    if num_sect != nz {
        ind_adoc = -4;
    }
    if (if_key > 0 || ((itype > 6 || if_are_tomo != 0) && if_info_from_fei == 0)) && ind_adoc < 0 {
        exit_error("This kind of information can be obtained only from a metadata file");
    }

    max_piece = nz + 1024;
    max_tilts = max_piece;
    tilt = vec![0.0; max_tilts as usize];
    val2 = vec![0.0; max_tilts as usize];
    array = vec![0; (max_extra / 4 * 4) as usize];
    ix_piece = vec![0; max_piece as usize];
    iy_piece = vec![0; max_piece as usize];
    iz_piece = vec![0; max_piece as usize];
    memory_error(0, "arrays for extra header data");
    if if_key > 0 {
        val_string = vec![String::new(); max_tilts as usize];
        memory_error(0, "array for string metadata");
    }
    if if_info_from_fei != 0 {
        ind_time = vec![0; max_tilts as usize];
        memory_error(0, "array for time stamp indexes");
    }
    if if_info_file != 0 {
        cumul_dose = vec![0.0; max_tilts as usize];
        sec_dose = vec![0.0; max_tilts as usize];
        memory_error(0, "arrays for dose information");
    }

    if if_no_image == 0 {
        // `call iiuRetExtendedData(1, numExtraBytes, array)`
        let mut extra: Vec<u8> = Vec::new();
        let _ = iiu_ret_extended_data(1, &mut extra);
        num_extra_bytes = iiu_ret_num_extended(1);
        let count = extra.len().min(array.len());
        array[..count].copy_from_slice(&extra[..count]);
    }

    //
    // Get piece coordinates unless doing "all".  First get from image file unless mdoc is
    // specified; then get from the mdoc if it is primary or if image file gave nothing
    if if_all == 0 {
        if !mdoc_primary {
            get_extra_header_pieces_fortran(
                &array,
                num_extra_bytes,
                nbytes,
                iflags,
                nz,
                &mut ix_piece,
                &mut iy_piece,
                &mut iz_piece,
                &mut num_pieces,
                max_piece,
            );
        }
        if (montage != 0 || hdf_file) && (mdoc_primary || (ind_adoc >= 0 && num_pieces == 0)) {
            //
            // Get from any HDF file in case montage flag is missing; but if there are no
            // pieces then move on without error
            get_metadata_pieces_fortran(
                ind_adoc,
                i_type_adoc,
                nz,
                &mut ix_piece,
                &mut iy_piece,
                &mut iz_piece,
                max_piece,
                &mut num_pieces,
            );
            if num_pieces > 0 {
                montage = 1;
            }
            if montage > 0 && num_pieces < nz {
                if hdf_file && if_add_mdoc != 0 {
                    exit_error(
                        "The HDF file is marked as a montage but does not have piece \
                         coordinates for every section",
                    );
                }
                if mdoc_primary {
                    exit_error(
                        "The metadata file does not have piece coordinates for every section",
                    );
                }
                exit_error(
                    "There are no piece coordinates in the image file; the metadata file \
                     indicates a montage but does not have piece coordinates for every section",
                );
            }
        }
    }
    if num_pieces == 0 {
        for i in 1..=nz {
            iz_piece[(i - 1) as usize] = i - 1;
        }
        maxz = nz;
    } else {
        let mut maxz_v = 0;
        for i in 1..=num_pieces {
            maxz_v = maxz_v.max(iz_piece[(i - 1) as usize] + 1);
        }
        maxz = maxz_v;
    }
    //
    // set up a marker value for empty slots
    //
    for i in 1..=maxz {
        tilt[(i - 1) as usize] = -999.;
    }
    //
    // Get from extended header
    num_tilts = 0;
    if !mdoc_primary && (itype <= 6 || itype == 10) && if_key == 0 {
        get_extra_header_items_fortran(
            &array,
            num_extra_bytes,
            nbytes,
            iflags,
            nz,
            itype,
            &mut tilt,
            Some(&mut val2),
            &mut num_tilts,
            &iz_piece,
        );
        if num_tilts == 0 && ind_adoc < 0 {
            println!(
                "\nERROR: EXTRACTTILTS - No {} information in this image file",
                type_text[(itype - 1) as usize].trim_end_matches(' ')
            );
            exit(1);
        }
        if if_info_from_fei != 0 && if_are_tomo != 0 {
            num_times = 0;
            get_extra_header_items_fortran(
                &array,
                num_extra_bytes,
                nbytes,
                iflags,
                nz,
                10,
                &mut sec_dose,
                Some(&mut val2),
                &mut num_times,
                &iz_piece,
            );
            if num_times == 0 {
                exit_error("No image time stamps in the image file");
            }
            if num_tilts > 0 && num_times != num_tilts {
                exit_error("Not all images have image time stamps");
            }
        }
    }
    if mdoc_primary || (itype > 6 && itype < 10) || num_tilts == 0 || if_key > 0 {
        if !mdoc_primary {
            println!(" Taking information from associated metadata file");
        }
        if if_key == 0 {
            get_metadata_items_fortran(
                ind_adoc,
                i_type_adoc,
                nz,
                itype,
                &mut tilt,
                &mut val2,
                &mut num_tilts,
                &mut num_found,
                &iz_piece,
            );
        } else {
            // `val2` is passed as both the second and third value arrays; a
            // string request (type 0) fills neither.
            let mut val3 = [0.0_f32; 1];
            get_metadata_by_key_fortran(
                ind_adoc,
                i_type_adoc,
                nz,
                &key_name,
                0,
                &mut tilt,
                &mut val2,
                &mut val3,
                &mut val_string,
                1000,
                &mut num_tilts,
                &mut num_found,
                max_tilts,
                &iz_piece,
            );
        }
        if num_found != nz {
            let text = type_text[(itype - 1) as usize].trim_end_matches(' ');
            if !mdoc_primary && itype <= 6 {
                println!(
                    "\nERROR: EXTRACTTILTS - {text} information is not present in image file \
                     and is missing for all or some sections in metadata file"
                );
            } else if hdf_file && if_add_mdoc != 0 {
                println!(
                    "\nERROR: EXTRACTTILTS - {text} is missing for all or some sections in HDF \
                     file"
                );
            } else {
                println!(
                    "\nERROR: EXTRACTTILTS - {text} is missing for all or some sections in \
                     metadata file"
                );
            }
            exit(1);
        }
    }
    //
    // pack the tilts down
    //
    num_tilt_out = 0;
    for i in 1..=num_tilts {
        let iu = (i - 1) as usize;
        if tilt[iu] != -999. {
            num_tilt_out = num_tilt_out + 1;
            let ou = (num_tilt_out - 1) as usize;
            tilt[ou] = tilt[iu];
            val2[ou] = val2[iu];
            if if_key > 0 {
                val_string[ou] = val_string[iu].clone();
            }
        } else if if_info_file != 0 {
            if if_are_tomo == 0 {
                exit_error("Some images in the file do not have a time stamp");
            }
            exit_error("Some images in the file do not have a tilt angle");
        }
    }
    //
    iunit_out = 6;
    let mut unit1: Option<BufWriter<std::fs::File>> = None;
    if !blank(&out_file) {
        unit1 = Some(BufWriter::new(dopen(1, &out_file, "new", "f")));
        iunit_out = 1;
    }
    let _ = iunit_out;
    // A formatted `write(iunitOut, fmt)` of the records in `lines`: an empty
    // implied-do list still writes the one (empty) record.
    let mut write_out = |lines: Vec<String>| {
        let lines = if lines.is_empty() {
            vec![String::new()]
        } else {
            lines
        };
        match unit1.as_mut() {
            Some(unit) => {
                for line in lines {
                    let _ = writeln!(unit, "{line}");
                }
            }
            None => {
                let mut stdout = std::io::stdout().lock();
                for line in lines {
                    let _ = writeln!(stdout, "{line}");
                }
            }
        }
    };
    let n_out = num_tilt_out.max(0) as usize;

    if if_tilt != 0 {
        write_out((0..n_out).map(|i| fmt_f(tilt[i], 7, 2)).collect());
    }
    if if_mag != 0 {
        write_out(
            (0..n_out)
                .map(|i| fmt_i(tilt[i].round() as i32, 7))
                .collect(),
        );
    }
    if if_c2 != 0 {
        write_out((0..n_out).map(|i| fmt_f(tilt[i], 8, 5)).collect());
    }
    if if_stage != 0 {
        write_out(
            (0..n_out)
                .map(|i| format!("{}{}", fmt_f(tilt[i], 9, 2), fmt_f(val2[i], 9, 2)))
                .collect(),
        );
    }
    if if_dose != 0 {
        write_out((0..n_out).map(|i| fmt_f(tilt[i], 13, 5)).collect());
    }
    if if_defocus != 0 || if_pixel != 0 || if_cam_exp != 0 {
        write_out((0..n_out).map(|i| fmt_f(tilt[i], 11, 3)).collect());
    }
    if if_key > 0 {
        write_out(
            (0..n_out)
                .map(|i| val_string[i].trim_end_matches(' ').to_owned())
                .collect(),
        );
    }

    // Output the dose info file by sorting the time stamps
    if if_info_file != 0 {
        if if_info_from_fei != 0 {
            for i in 1..=num_tilt_out {
                let iu = (i - 1) as usize;
                ind_time[iu] = i;
                if if_are_tomo == 0 {
                    sec_dose[iu] = tilt[iu];
                }
            }
            rssortindexedfloats(&sec_dose, &mut ind_time, &num_tilt_out);
            for i in 1..=num_tilt_out {
                cumul_dose[(ind_time[(i - 1) as usize] - 1) as usize] = prior_dose;
                prior_dose = prior_dose + fixed_dose;
            }
        } else {
            ierr = get_metadata_weighting_doses_fortran(
                ind_adoc,
                i_type_adoc,
                nz,
                &iz_piece,
                0,
                &mut cumul_dose,
                &mut sec_dose,
            );
            if ierr > 0 {
                exit_error("Getting doses from metadata");
            }
            if ierr < 0 {
                println!(
                    "WARNING: EXTRACTTILTS - Assuming the images were acquired in order because \
                     the metadata has no PriorRecordDose or DateTime entries and -bidir was not \
                     entered"
                );
            }
        }
        if if_are_tomo != 0 {
            write_out(
                (0..n_out)
                    .map(|i| format!("{}{}", fmt_f(tilt[i], 8, 2), fmt_f(cumul_dose[i], 10, 3)))
                    .collect(),
            );
        } else {
            write_out(
                (0..n_out)
                    .map(|i| {
                        format!(
                            "{}{}",
                            fmt_f(cumul_dose[i], 10, 3),
                            fmt_f(fixed_dose, 10, 3)
                        )
                    })
                    .collect(),
            );
        }
    }
    //
    // Reporting of minimum and starting tilt and view
    if if_min_tilt != 0 {
        // Get lables, or number thereof
        if if_no_image == 0 {
            iiu_ret_labels(1, &mut all_labels, &mut num_labels);
        } else {
            num_labels = adoc_get_number_of_sections(b"T").unwrap_or(-1);
        }
        //
        // Loop on labels: copy into temp or get from mdoc.  `tempLabelStr` is
        // `character*81` sharing its first 80 bytes with `tempLabel(20)`; the
        // 81st is never written (a blank here).
        'labels: for i in 1..=num_labels {
            if if_no_image == 0 {
                temp_label_str = all_labels[(i - 1) as usize].to_vec();
                temp_label_str.push(b' ');
            } else {
                // `adocgetsectionname` (`adoc_fwrap.c:397`): 1-based index;
                // `c2fString` into the 81 characters, an error if longer.
                match adoc_get_section_name(b"T", i - 1) {
                    Ok(name) if name.len() <= 81 => {
                        temp_label_str = name;
                        temp_label_str.resize(81, b' ');
                    }
                    _ => {
                        break 'labels;
                    }
                }
            }
            let index = |haystack: &[u8], needle: &[u8]| -> i32 {
                haystack
                    .windows(needle.len())
                    .position(|window| window == needle)
                    .map_or(0, |pos| pos as i32 + 1)
            };
            //
            // Look for current axis angle tag
            if index(&temp_label_str, b"Tilt axis angle") > 0 {
                //
                // Look for bidir or dosym tag
                ind = index(&temp_label_str, b"bidir");
                if ind > 0 {
                    start_type = "bidirectional";
                } else {
                    ind = index(&temp_label_str, b"dosym");
                    if ind > 0 {
                        start_type = "dose-symmetric";
                    } else {
                        break 'labels;
                    }
                }
                //
                // Get - after that and convert #
                jnd = index(&temp_label_str[(ind - 1) as usize..80], b"=");
                if jnd < 1 {
                    break 'labels;
                }
                // `read(tempLabelStr(ind+jnd:80), *, err = 102, end = 102) startAngle`
                let start = (ind + jnd - 1) as usize;
                let mut internal: &[u8] = if start < 80 {
                    &temp_label_str[start..80]
                } else {
                    b""
                };
                let mut value = start_angle;
                if list_read(&mut internal, &mut [ListItem::Real(&mut value)]).is_ok() {
                    start_angle = value;
                }
                break 'labels;
            }
        }
        //
        // Now find minimum tilt, view of minimum tilt, and view of start angle
        tilt_min = 1.0e10;
        start_min = 1.0e10;
        for i in 1..=num_tilt_out {
            let iu = (i - 1) as usize;
            if tilt[iu].abs() < tilt_min {
                tilt_min = tilt[iu].abs();
                min_view = i;
            }
            if start_angle < 1.0e9 {
                if (tilt[iu] - start_angle).abs() < start_min {
                    start_min = (tilt[iu] - start_angle).abs();
                    iv_start = i;
                }
            }
        }
        //
        // Outputs
        write_out(vec![format!(
            "Minimum tilt angle {} at view {}",
            fmt_f(tilt_min, 8, 2),
            fmt_i(min_view, 4)
        )]);
        if start_angle < 1.0e9 {
            write_out(vec![format!(
                "Starting {} angle {} at view {}",
                start_type,
                fmt_f(start_angle, 8, 2),
                fmt_i(iv_start, 4)
            )]);
        }
    }
    drop(write_out);

    if let Some(mut unit) = unit1.take() {
        let _ = unit.flush();
        drop(unit);
        if if_info_file == 0 && if_min_tilt == 0 {
            println!(
                "{:>12}  {}s output to file",
                num_tilt_out,
                type_text[(itype - 1) as usize].trim_end_matches(' ')
            );
        }
        if if_min_tilt != 0 {
            println!(" Minimum angle and view information output to file");
        }
    }

    if if_tilt > 0 && if_warn > 0 {
        num_tilts = 0;
        for i in 1..=num_tilt_out {
            let iu = (i - 1) as usize;
            if tilt[iu].abs() < 0.1 {
                num_tilts = num_tilts + 1;
            }
            if tilt[iu].abs() > 95. {
                if_mag = if_mag + 1;
            }
        }
        if num_tilt_out > 2 && num_tilts > num_tilt_out / 2 {
            println!(
                "WARNING: extracttilts - {} of the extracted tilt angles are near zero",
                fmt_i(num_tilts, 4)
            );
        }
        if if_mag > 0 {
            println!(
                "WARNING: extracttilts - {} of the extracted tilt angles are greater than 95 \
                 degrees",
                fmt_i(if_mag, 4)
            );
        }
    }

    unsafe {
        iiu_close(1);
    }
    //
    exit(0);
}
