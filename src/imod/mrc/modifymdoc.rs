//! Translation of `IMOD/mrc/modifymdoc.cpp`.
//!
//! The unit is one function, `main`.  Every `exitError` goes through
//! [`exit_error`], whose prefix `PipReadOrParseOptions` set, and ends in
//! `b3dutil::exit`, so the command is safe to run in process.

use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, ADOC_ZVALUE_NAME, adoc_change_section_name, adoc_delete_key_value, adoc_done,
    adoc_get_float, adoc_get_two_integers, adoc_open_image_metadata, adoc_order_write_by_value,
    adoc_set_float, adoc_set_two_integers, adoc_write,
};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_get_error, c_format_bytes, exit, imod_prog_name, imod_usage_header,
};
use crate::imod::libcfshr::extraheader::{get_metadata_items, get_metadata_weighting_doses};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_float, pip_get_in_out_file, pip_get_integer,
    pip_read_or_parse_options,
};
use crate::imod::libcfshr::robuststat::rs_sort_indexed_floats;
use std::io::Write;

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    // `PipPrintHelp` writes through Rust's stdout; the banner is on the C
    // stream, so hand it over before the help body follows it.
    let _ = ImodFile::Stdout.flush();
}

/// C `main` in `modifymdoc.cpp:19`.
pub fn modifymdoc(arguments: &[String]) -> ! {
    let prog_name = imod_prog_name(arguments.first().map_or("", String::as_str));
    let mut in_file = Vec::new();
    let mut out_file = Vec::new();
    let mut reorder: i32 = 0;
    let mut prior: f32 = 0.;
    let mut dose: f32 = 0.;
    let mut new_binning: f32 = 0.;
    let mut old_binning: f32;
    // Uninitialised in the source; it is read only when `newBinning` is
    // nonzero, which is exactly when it has been assigned.
    let mut bin_scale: f32 = 0.;
    let mut old_pixel: f32;
    let mut new_pixel: f32 = 0.;
    let mut montage: i32 = 0;
    let mut num_sect: i32 = 0;
    let mut sect_type: i32 = 0;
    let mut ind: i32;
    let mut num_found: i32 = 0;
    let mut num_vals: i32 = 0;
    let mut nx: i32 = 0;
    let mut ny: i32 = 0;
    let mut iz_piece: Vec<i32> = Vec::new();
    let mut tilts: Vec<f32> = Vec::new();
    let mut sec_doses: Vec<f32> = Vec::new();
    let mut prior_doses: Vec<f32> = Vec::new();

    // Fallbacks from ../manpages/autodoc2man 2 1 modifymdoc
    let mut num_opt_args = 0;
    let mut num_non_opt_args = 0;
    let num_options = 8;
    let options: [&[u8]; 8] = [
        b"input:InputFile:FN:",
        b"output:OutputFile:FN:",
        b"order:OrderToProduce:I:",
        b"dose:ElectronDosePerImage:F:",
        b"binning:BinningToScaleTo:F:",
        b"pixel:PixelSpacingToSet:F:",
        b"param:ParameterFile:PF:",
        b"help:usage:B:",
    ];

    // Startup with fallback
    let argv = arguments
        .iter()
        .map(|arg| arg.as_bytes().to_vec())
        .collect::<Vec<_>>();
    pip_read_or_parse_options(
        argv.len() as i32,
        &argv,
        &options,
        num_options,
        prog_name.as_bytes(),
        3,
        1,
        1,
        &mut num_opt_args,
        &mut num_non_opt_args,
        Some(imod_usage_header_for_pip),
    );

    // Get options
    if pip_get_in_out_file(b"InputFile", 0, &mut in_file) != 0 {
        exit_error(b"No input file specified");
    }
    if pip_get_in_out_file(b"OutputFile", 1, &mut out_file) != 0 {
        exit_error(b"No output file specified");
    }
    if pip_get_float(b"ElectronDosePerImage", &mut dose) == 0 && dose <= 0. {
        exit_error(b"The electron dose must be positive");
    }
    pip_get_integer(b"OrderToProduce", &mut reorder);
    // `fabs(newBinning - 0.5)` is a double expression; `B3DNINT(newBinning) -
    // newBinning` is int - float, a float, widened for `fabs`.
    if pip_get_float(b"BinningToScaleTo", &mut new_binning) == 0
        && !((new_binning as f64 - 0.5).abs() < 0.001
            || (new_binning as f64 > 0.51
                && ((((new_binning as f64 + 0.5).floor() as i32) as f32 - new_binning) as f64)
                    .abs()
                    < 0.001))
    {
        exit_error(b"New binning must be 0.5 or an integer");
    }
    if pip_get_float(b"PixelSpacingToSet", &mut new_pixel) == 0 && new_pixel as f64 <= 0. {
        exit_error(b"New pixel size must be positive");
    }
    pip_done();

    // Open the mdoc, check errors
    let adoc_ind =
        adoc_open_image_metadata(&in_file, 0, &mut montage, &mut num_sect, &mut sect_type);
    if adoc_ind == -1 {
        exit_error(&c_format_bytes(
            "Opening or reading input file %s",
            &[CArg::Bytes(&in_file)],
        ));
    }
    if adoc_ind == -2 {
        exit_error(&c_format_bytes(
            // BUGS.md: the source's message reads "dose not exist".
            "Input file %s does not exist",
            &[CArg::Bytes(&in_file)],
        ));
    }
    if adoc_ind < -2 {
        exit_error(&c_format_bytes(
            "The input file %s is not a valid autodoc",
            &[CArg::Bytes(&in_file)],
        ));
    }
    if montage != 0 {
        exit_error(b"An mdoc file from a montage cannot be reordered");
    }
    if sect_type != 1 {
        exit_error(&c_format_bytes(
            "This program can be used only with a file having sections named %s",
            &[CArg::Bytes(ADOC_ZVALUE_NAME)],
        ));
    }
    if num_sect < 2 {
        exit_error(b"There must be at least two sections in the input file");
    }

    // Set up Z/index list and get the tilts
    for iz in 0..num_sect {
        iz_piece.push(iz);
    }
    tilts.resize(num_sect as usize, 0.);
    // The source passes `NULL` for `val2`, which `getMetadataByKey` never
    // touches for a single-float key; the translated routine takes a slice.
    let mut null_val2: Vec<f32> = vec![0.; num_sect as usize];
    if get_metadata_items(
        adoc_ind,
        sect_type,
        num_sect,
        1,
        &mut tilts,
        &mut null_val2,
        &mut num_vals,
        &mut num_found,
        &iz_piece,
    ) != 0
    {
        exit_error(&c_format_bytes(
            "Getting tilt angles from mdoc: %s",
            &[CArg::Str(&b3d_get_error())],
        ));
    }
    if num_found < num_sect {
        exit_error(&c_format_bytes(
            "There are tilt angles in only %d of %d sections",
            &[CArg::Int(num_found as i64), CArg::Int(num_sect as i64)],
        ));
    }

    // Handle dose: Set dose and remove prior information if present
    if dose != 0. {
        sec_doses.resize(num_sect as usize, 0.);
        prior_doses.resize(num_sect as usize, 0.);
        for iz in 0..num_sect {
            if adoc_set_float(ADOC_ZVALUE_NAME, iz, b"ExposureDose", dose) != 0 {
                exit_error(&c_format_bytes(
                    "Adding ExposureDose for section %d",
                    &[CArg::Int(iz as i64)],
                ));
            }
            if adoc_get_float(ADOC_ZVALUE_NAME, iz, b"PriorRecordDose", &mut prior) == 0
                && adoc_delete_key_value(ADOC_ZVALUE_NAME, iz, b"PriorRecordDose").is_err()
            {
                exit_error(&c_format_bytes(
                    "Removing PriorRecordDose for section %d",
                    &[CArg::Int(iz as i64)],
                ));
            }
        }

        // Get accumulated doses using time stamps
        // BUGS.md, fixed in translation: `modifymdoc.cpp:97` fails on any nonzero
        // return, but -1 is `getMetadataWeightingDoses`'s documented success
        // for an mdoc with no DateTime entries (doses summed in file order, as
        // the man page promises).  Only a positive return is an error here.
        if get_metadata_weighting_doses(
            adoc_ind,
            sect_type,
            num_sect,
            &iz_piece,
            0,
            &mut prior_doses,
            &mut sec_doses,
        ) > 0
        {
            exit_error(&c_format_bytes(
                "Getting accumulated dose information back: %s",
                &[CArg::Str(&b3d_get_error())],
            ));
        }

        // Set them (again)
        for iz in 0..num_sect {
            if adoc_set_float(
                ADOC_ZVALUE_NAME,
                iz,
                b"PriorRecordDose",
                prior_doses[iz as usize],
            ) != 0
            {
                exit_error(b"Putting new PriorRecordDose entries into autodoc");
            }
        }
    }

    // Handle binning change
    if new_binning != 0. || new_pixel != 0. {
        old_binning = 1.;
        old_pixel = 0.;
        if new_binning != 0. {
            if adoc_get_float(ADOC_ZVALUE_NAME, 0, b"Binning", &mut old_binning) < 0 {
                exit_error(b"Getting old binning from section 0 of autodoc");
            }
            bin_scale = new_binning / old_binning;
        }

        if new_pixel == 0. {
            // Get pixel spacing from global, fall back to first section
            ind = adoc_get_float(ADOC_GLOBAL_NAME, 0, b"PixelSpacing", &mut old_pixel);
            if ind < 0 {
                exit_error(b"Getting old PixelSpacing from global section of autodoc");
            }
            if ind > 0 {
                ind = adoc_get_float(ADOC_ZVALUE_NAME, 0, b"PixelSpacing", &mut old_pixel);
                if ind < 0 {
                    exit_error(b"Getting old PixelSpacing from section 0 of autodoc");
                }
            }
            if old_pixel != 0. {
                new_pixel = old_pixel * bin_scale;
            }
        }

        // Handle image size (poorly)
        if new_binning != 0. {
            ind = adoc_get_two_integers(ADOC_GLOBAL_NAME, 0, b"ImageSize", &mut nx, &mut ny);
            if ind < 0 {
                exit_error(b"Getting old ImageSize from global section of autodoc");
            }
            if ind == 0 {
                // `B3DNINT(nx / binScale)`: int / float is a float quotient,
                // widened for `+ 0.5` and `floor`.
                nx = ((nx as f32 / bin_scale) as f64 + 0.5).floor() as i32;
                ny = ((ny as f32 / bin_scale) as f64 + 0.5).floor() as i32;
                if adoc_set_two_integers(ADOC_GLOBAL_NAME, 0, b"ImageSize", nx, ny) != 0 {
                    exit_error(b"Setting new ImageSize in autodoc");
                }
            }
        }

        // Set the pixel size and binning in the sections
        if (old_pixel != 0. || new_pixel != 0.)
            && adoc_set_float(ADOC_GLOBAL_NAME, 0, b"PixelSpacing", new_pixel) != 0
        {
            exit_error(b"Setting new PixelSpacing in global section of autodoc");
        }
        for iz in 0..num_sect {
            if (new_binning != 0.
                && adoc_set_float(ADOC_ZVALUE_NAME, iz, b"Binning", new_binning) != 0)
                || ((old_pixel != 0. || new_pixel != 0.)
                    && adoc_set_float(ADOC_ZVALUE_NAME, iz, b"PixelSpacing", new_pixel) != 0)
            {
                exit_error(&c_format_bytes(
                    "Setting new PixelSpacing or Binning in section %d of autodoc",
                    &[CArg::Int(iz as i64)],
                ));
            }
        }
    }

    // For reordering
    if reorder != 0 {
        rs_sort_indexed_floats(&tilts, &mut iz_piece, num_sect);

        // reverse indexes
        if reorder < 0 {
            for iz in 0..num_sect / 2 {
                iz_piece.swap(iz as usize, ((num_sect - 1) - iz) as usize);
            }
        }

        // Set the names and set up to write in order by name
        for iz in 0..num_sect {
            let buffer = c_format_bytes("%d", &[CArg::Int(iz as i64)]);
            if adoc_change_section_name(ADOC_ZVALUE_NAME, iz_piece[iz as usize], &buffer).is_err() {
                exit_error(&c_format_bytes(
                    "Changing name for section %d to %s",
                    &[
                        CArg::Int(iz_piece[iz as usize] as i64),
                        CArg::Bytes(&buffer),
                    ],
                ));
            }
        }
        if adoc_order_write_by_value(Some(ADOC_ZVALUE_NAME)) != 0 {
            exit_error(b"Memory error setting up to write sections in new order");
        }
    }

    // Write file
    if adoc_write(&out_file) < 0 {
        exit_error(&c_format_bytes(
            "Writing the new mdoc file %s",
            &[CArg::Bytes(&out_file)],
        ));
    }
    adoc_done();
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "Wrote new mdoc file %s\n",
        &[CArg::Bytes(&out_file)],
    ));
    exit(0);
}
