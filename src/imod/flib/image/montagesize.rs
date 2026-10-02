//! Translation of `IMOD/flib/image/montagesize.f90`.
//!
//! MONTAGESIZE will determine the X, Y, and Z dimensions of a montaged image
//! file, from piece coordinates that are contained either in the file header
//! or in a separate piece list file.  The file names are specified
//! exclusively as command line arguments: first the image file name, then the
//! piece list file name, if any.  If there is one argument, the program
//! attempts to read the coordinates from the image file header.
//!
//! The main program maps to [`montagesize`].  Library calls go to the Fortran
//! wrappers the source calls: `iiuRetAdocIndex` returns the index plus one,
//! `AdocSetCurrent` subtracts one (`adoc_fwrap.c`), and `get_metadata_pieces`
//! is the 1-based `extraheader.c` wrapper.

use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, set_exit_prefix,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::flib::subrs::piecesubs::read_piece_list::read_piece_list;
use crate::imod::libcfshr::autodoc::{adoc_get_image_meta_info, adoc_set_current};
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{exit, program_args};
use crate::imod::libcfshr::extraheader::{
    get_extra_header_pieces_fortran, get_metadata_pieces_fortran,
};
use crate::imod::libcfshr::piecefuncs::checklist;
use crate::imod::libiimod::unit_fileio::{iiu_alt_print, iiu_file_type, iiu_ret_adoc_index};
use crate::imod::libiimod::unit_header::{
    iiu_ret_extended_data, iiu_ret_extended_type, iiu_ret_num_extended,
};
use std::io::Write;

/// Original program `montagesize` (`montagesize.f90:16`).
pub fn montagesize() {
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut mode_in = 0_i32;
    let (mut dmin, mut dmax, mut dmean) = (0f32, 0f32, 0f32);
    let mut num_pc_list = 0_i32;
    let (mut min_xpiece, mut nx_pieces, mut nx_overlap) = (0_i32, 0_i32, 0_i32);
    let (mut min_ypiece, mut ny_pieces, mut ny_overlap) = (0_i32, 0_i32, 0_i32);
    let (mut montage, mut num_sect, mut i_type_adoc) = (0_i32, 0_i32, 0_i32);
    let args = program_args();
    // iargc()
    let iargc = args.len() as i32 - 1;
    // `character*320`: getarg keeps the first 320 characters.
    let getarg = |index: usize| -> String {
        let arg = args[index].as_bytes();
        String::from_utf8_lossy(&arg[..arg.len().min(320)])
            .trim_end_matches(' ')
            .to_owned()
    };
    //
    let mut max_piece = 1000000_i32;
    if !(1..=2).contains(&iargc) {
        let mut out = ImodFile::Stdout;
        let _ = writeln!(out, " Usage: montagesize image_file piece_list_file");
        let _ = writeln!(
            out,
            "    (piece_list_file is optional if image_file contains piece coordinates)"
        );
        let _ = out.flush();
        exit(0);
    }
    set_exit_prefix("ERROR: MONTAGESIZE - ");
    let image_file = getarg(1);
    iiu_alt_print(0);
    imopen(1, &image_file, "ro");
    unsafe {
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode_in,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
    }
    max_piece = max_piece.max(2 * nxyz[2]);
    let mut ix_pc_list = vec![0_i32; max_piece as usize];
    let mut iy_pc_list = vec![0_i32; max_piece as usize];
    let mut iz_pc_list = vec![0_i32; max_piece as usize];
    memory_error(0, "arrays for piece coordinates");
    if iargc == 1 {
        let num_extra_bytes = iiu_ret_num_extended(1);
        let max_extra = num_extra_bytes + 1000;
        // `real*4 array(maxExtra / 4)`, handled as its bytes.
        let mut array = vec![0_u8; (max_extra / 4) as usize * 4];
        memory_error(0, "array for extra header data");
        let mut extra: Vec<u8> = Vec::new();
        let _ = iiu_ret_extended_data(1, &mut extra);
        let count = extra.len().min(array.len());
        array[..count].copy_from_slice(&extra[..count]);
        let [num_sec_bytes, iflags] = iiu_ret_extended_type(1);
        get_extra_header_pieces_fortran(
            &array,
            num_extra_bytes,
            num_sec_bytes,
            iflags,
            nxyz[2],
            &mut ix_pc_list,
            &mut iy_pc_list,
            &mut iz_pc_list,
            &mut num_pc_list,
            max_piece,
        );
        if num_pc_list == 0 {
            // The Fortran wrapper `iiuretadocindex` returns the index plus one.
            let mut ind_adoc = unsafe { iiu_ret_adoc_index(1, 0, 1) };
            if ind_adoc >= 0 {
                ind_adoc += 1;
            }
            if ind_adoc < 0 {
                exit_error("No piece list information in this image file");
            }
            // `AdocSetCurrent` from Fortran subtracts one.
            if adoc_set_current(ind_adoc - 1).is_err() {
                exit_error("Setting current autodoc");
            }
            if adoc_get_image_meta_info(&mut montage, &mut num_sect, &mut i_type_adoc) == 0 {
                get_metadata_pieces_fortran(
                    ind_adoc,
                    i_type_adoc,
                    nxyz[2],
                    &mut ix_pc_list,
                    &mut iy_pc_list,
                    &mut iz_pc_list,
                    max_piece,
                    &mut num_pc_list,
                );
            }
            if num_pc_list == 0 {
                if unsafe { iiu_file_type(1) } == 5 {
                    exit_error("No piece list information in this HDF file");
                }
                exit_error("No piece list information in this image file or associated .mdoc file");
            }
        }
    } else {
        let piece_file = getarg(2);
        read_piece_list(
            &piece_file,
            &mut ix_pc_list,
            &mut iy_pc_list,
            &mut iz_pc_list,
            &mut num_pc_list,
        );
        if num_pc_list == 0 {
            exit_error("No piece list information in the piece list file");
        }
    }
    //
    // find min and max Z
    //
    let mut min_zpc = 1000000_i32;
    let mut max_zpc = -min_zpc;
    for i in 1..=nxyz[2].min(num_pc_list) {
        min_zpc = min_zpc.min(iz_pc_list[(i - 1) as usize]);
        max_zpc = max_zpc.max(iz_pc_list[(i - 1) as usize]);
    }
    let num_sections = max_zpc + 1 - min_zpc;
    //
    // now check lists and get basic properties of overlap etc
    //
    checklist(
        &ix_pc_list[..num_pc_list as usize],
        1,
        nxyz[0],
        &mut min_xpiece,
        &mut nx_pieces,
        &mut nx_overlap,
    );
    checklist(
        &iy_pc_list[..num_pc_list as usize],
        1,
        nxyz[1],
        &mut min_ypiece,
        &mut ny_pieces,
        &mut ny_overlap,
    );
    if nx_pieces <= 0 || ny_pieces <= 0 {
        exit_error("Piece list information not good");
    }
    //
    let nx_tot_pix = nx_pieces * (nxyz[0] - nx_overlap) + nx_overlap;
    let ny_tot_pix = ny_pieces * (nxyz[1] - ny_overlap) + ny_overlap;
    let mut out = ImodFile::Stdout;
    let _ = writeln!(
        out,
        " Total NX, NY, NZ:{:10}{:10}{:10}",
        nx_tot_pix, ny_tot_pix, num_sections
    );
    let _ = out.flush();
    if iargc > 1 {
        if nxyz[2] < num_pc_list {
            let _ = writeln!(
                out,
                "\nERROR: MONTAGESIZE - The Z size of the image file is smaller than the size of the piece list"
            );
            let _ = out.flush();
            exit(2);
        }
        if nxyz[2] > num_pc_list {
            let _ = writeln!(
                out,
                "\nERROR: MONTAGESIZE - The Z size of the image file is larger than the size of the piece list"
            );
            let _ = out.flush();
            exit(3);
        }
    }
    exit(0);
}
