//! Translation of `IMOD/flib/image/extractpieces.f90`.
//!
//! EXTRACTPIECES will extract piece coordinates from the header of an image
//! file, if they are present, and produce a file with those coordinates (a
//! piece list file).  David Mastronarde, 1/2/00.
//!
//! The main program maps to [`extractpieces`].  Library calls go to the
//! Fortran wrappers the source calls: `AdocOpenImageMetadata` and
//! `iiuRetAdocIndex` return the index plus one, `AdocSetCurrent` subtracts
//! one (`adoc_fwrap.c`), and `get_extra_header_pieces` /
//! `get_metadata_pieces` are the 1-based `extraheader.c` wrappers.

use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_exit_on_error, pip_get_in_out_file, pip_get_logical,
    pip_parse_input,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::libcfshr::autodoc::{
    adoc_get_image_meta_info, adoc_open_image_metadata, adoc_set_current,
};
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libcfshr::extraheader::{
    get_extra_header_pieces_fortran, get_metadata_pieces_fortran,
};
use crate::imod::libcfshr::parse_params::{
    pip_allow_comma_defaults, pip_done, pip_get_boolean, pip_print_help,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_file_type, iiu_ret_adoc_index};
use crate::imod::libiimod::unit_header::{
    iiu_ret_extended_data, iiu_ret_extended_type, iiu_ret_num_extended,
};
use std::io::Write;

/// `parameter (numOptions = 5)` (`extractpieces.f90:28`).
const EXTRACTPIECES_NUM_OPTIONS: i32 = 5;
/// The `options(1)` string (`extractpieces.f90:30-37`).
const EXTRACTPIECES_OPTIONS: &str = "input:InputFile:FN:Name of input image file@\
output:OutputFile:FN:Name of output piece list file@\
mdoc:MdocMetadataFile:B:Take coordinates from metadata file named inputfile.mdoc (omit name)@\
other:OtherMetadataFile:FN:Other metadata file to take coordinates from (follow with name)@\
help:usage:B:Print help output";

/// Original program `extractpieces` (`extractpieces.f90:9`).
pub fn extractpieces() {
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut max_piece: i32;
    let mut ix_piece: Vec<i32> = Vec::new();
    let mut iy_piece: Vec<i32> = Vec::new();
    let mut iz_piece: Vec<i32> = Vec::new();
    let mut mode = 0_i32;
    let mut num_pieces: i32;
    let mut ierr = 0_i32;
    let mut ind_adoc = 0_i32;
    let mut i_type_adoc = 0_i32;
    let mut num_sect = 0_i32;
    let mut montage = 0_i32;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut use_mdoc: bool;
    let mut hdf_file: bool;
    //
    let mut in_file = String::new();
    let mut out_file = String::new();
    let mut meta_file = [b' '; 320];
    // `character*10 metaOrHDF /'metadata'/`
    let mut meta_or_hdf = "metadata";
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);

    meta_file.fill(b' ');
    in_file.clear();
    use_mdoc = false;
    hdf_file = false;
    nxyz[2] = 0;
    max_piece = 0;
    //
    pip_exit_on_error(0, "ERROR: EXTRACTPIECES - ");
    pip_allow_comma_defaults(1);
    let _ = pip_parse_input(
        &[EXTRACTPIECES_OPTIONS],
        EXTRACTPIECES_NUM_OPTIONS,
        '@',
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    if pip_get_boolean(b"help", &mut ierr) == 0 {
        pip_print_help(b"extractpieces", 0, 1, 1);
        exit(0);
    }
    let if_add_mdoc = pipgetstring_(b"OtherMetadataFile", &mut meta_file);
    let _ = pip_get_logical("MdocMetadataFile", &mut use_mdoc);
    if use_mdoc && if_add_mdoc == 0 {
        exit_error("You cannot enter both -mdoc and -other");
    }
    if pip_get_in_out_file("InputFile", 1, "Name of input image file", &mut in_file, 320) != 0
        && if_add_mdoc != 0
    {
        exit_error("You must enter either an input image file or a metadata file with -other");
    }
    if pip_get_in_out_file(
        "OutputFile",
        2,
        "Name of output piece list file",
        &mut out_file,
        320,
    ) != 0
    {
        exit_error("No output file specified");
    }
    pip_done();
    //
    if !in_file.trim_end_matches(' ').is_empty() {
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
        hdf_file = unsafe { iiu_file_type(1) } == 5 || unsafe { iiu_file_type(1) } == 7;
        if hdf_file && if_add_mdoc != 0 {
            // The Fortran wrapper `iiuretadocindex` returns the index plus one.
            ind_adoc = unsafe { iiu_ret_adoc_index(1, 0, 0) };
            if ind_adoc >= 0 {
                ind_adoc += 1;
            }
            if ind_adoc <= 0 {
                exit_error("Getting autodoc index for HDF or IDOC file");
            }
            // `AdocSetCurrent` from Fortran subtracts one.
            if adoc_set_current(ind_adoc - 1).is_err() {
                exit_error("Setting autodoc structure of HDF or IDOC file as current autodoc");
            }
            if adoc_get_image_meta_info(&mut montage, &mut num_sect, &mut i_type_adoc) < 0 {
                println!(" This file does not have metadata about each section");
                let _ = std::io::stdout().flush();
                exit(0);
            }
            meta_or_hdf = "HDF";
        }
    }
    //
    // try to open a metadata file regardless, unless got one from HDF file
    if meta_file.iter().all(|&c| c == b' ') {
        meta_file.fill(b' ');
        let bytes = in_file.as_bytes();
        let count = bytes.len().min(320);
        meta_file[..count].copy_from_slice(&bytes[..count]);
    }
    if !hdf_file || if_add_mdoc == 0 {
        // The Fortran wrapper strips the blank padding and returns the index
        // plus one.
        ind_adoc = adoc_open_image_metadata(
            fortran_string(&meta_file).as_bytes(),
            if_add_mdoc,
            &mut montage,
            &mut num_sect,
            &mut i_type_adoc,
        );
        if ind_adoc >= 0 {
            ind_adoc += 1;
        }
    }
    if ind_adoc > 0 && in_file.trim_end_matches(' ').is_empty() {
        nxyz[2] = num_sect;
    }
    num_pieces = 0;
    let nz = nxyz[2];

    if !use_mdoc && if_add_mdoc != 0 && !hdf_file {
        //
        // Get data from the image header
        let num_extra_bytes = iiu_ret_num_extended(1);
        let [nbytes, iflags] = iiu_ret_extended_type(1);
        let max_extra = num_extra_bytes + 1024;
        max_piece = nz + 1024;
        // `real*4 array(maxExtra / 4)`, handled as its bytes.
        let mut array: Vec<u8> = Vec::new();
        let failed = array
            .try_reserve_exact((max_extra / 4).max(0) as usize * 4)
            .is_err()
            || ix_piece.try_reserve_exact(max_piece as usize).is_err()
            || iy_piece.try_reserve_exact(max_piece as usize).is_err()
            || iz_piece.try_reserve_exact(max_piece as usize).is_err();
        memory_error(i32::from(failed), "arrays for extra header or piece data");
        array.resize((max_extra / 4).max(0) as usize * 4, 0);
        ix_piece.resize(max_piece as usize, 0);
        iy_piece.resize(max_piece as usize, 0);
        iz_piece.resize(max_piece as usize, 0);
        let mut extra: Vec<u8> = Vec::new();
        let _ = iiu_ret_extended_data(1, &mut extra);
        let count = extra.len().min(array.len());
        array[..count].copy_from_slice(&extra[..count]);
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
        if num_pieces == 0 {
            println!(" There are no piece coordinates in this image file");
            if ind_adoc != -2 {
                println!(" Looking for piece coordinates in the associated metadata file");
            }
        }
    }

    if num_pieces == 0 {
        //
        // Or get data from the autodoc file
        //
        // Thanks to a bug in SerialEM, Montage flag may be missing.  So just plow
        // ahead regardless
        if ind_adoc > 0 && num_sect == nz {
            if max_piece == 0 {
                max_piece = nz + 1024;
                let failed = ix_piece.try_reserve_exact(max_piece as usize).is_err()
                    || iy_piece.try_reserve_exact(max_piece as usize).is_err()
                    || iz_piece.try_reserve_exact(max_piece as usize).is_err();
                memory_error(i32::from(failed), "arrays for piece data");
                ix_piece.resize(max_piece as usize, 0);
                iy_piece.resize(max_piece as usize, 0);
                iz_piece.resize(max_piece as usize, 0);
            }
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
        }
        //
        // Give lots of different error messages
        if ind_adoc < 0 || num_pieces < nz {
            if ind_adoc == -2 {
                if use_mdoc || if_add_mdoc == 0 {
                    println!(" The metadata file does not exist");
                }
            } else if ind_adoc == -3 {
                println!(" The autodoc file is not a recognized type of image metadata file");
            } else if ind_adoc == -1 {
                println!(" There was an error opening or reading the metadata file");
            } else if num_pieces > 0 || num_sect != nz {
                println!(
                    "The {} file does not have piece coordinates for every image in the file",
                    meta_or_hdf
                );
            } else {
                println!("There are no piece coordinates in the {} file", meta_or_hdf);
            }
        }
    }
    //
    // It used to output whatever is there, even if it is short, so restore that 6/24/16
    if num_pieces > 0 {
        let mut unit1 = dopen(1, &out_file, "new", "f");
        // write(1, '(2i9,i7)')
        let mut text = String::new();
        for i in 0..num_pieces as usize {
            text.push_str(&format!(
                "{:>9}{:>9}{:>7}\n",
                ix_piece[i], iy_piece[i], iz_piece[i]
            ));
        }
        let _ = unit1.write_all(text.as_bytes());
        drop(unit1);
        println!("{:>12}  piece coordinates output to file", num_pieces);
    }

    unsafe {
        iiu_close(1);
    }
    //
    let _ = std::io::stdout().flush();
    exit(0);
}
