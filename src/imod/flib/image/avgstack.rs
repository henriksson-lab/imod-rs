//! Translation of `IMOD/flib/image/avgstack.f`.
//!
//! This program reads in an image file containing multiple sections and
//! averages the sections into a single output image.  Written 27-Jan-1989 by
//! sjm; 19-Dec-1989 dnm fixed cell setting in header.
//!
//! The main program maps to [`avgstack`] and its `errorexit` subroutine to
//! [`errorexit`].  The old-style unit calls go to the C functions their
//! Fortran wrappers call (`unit_header.c`, `unit_fileio.c`): `itrhdr` is
//! `iiuTransHeader`, `ialmod` `iiuAltMode`, `ialnbsym` `iiuAltNumExtended`,
//! `ialsymtyp` `iiuAltExtendedType`, `ialsam` `iiuAltSample`, `ialsiz`
//! `iiuAltSize`, `irtcel` `iiuRetCell`, `ialcel` `iiuAltCell`, `imposn`
//! `iiuSetPosition`, `iwrhdr` `iiuWriteHeader`, `iwrsec` `iiuWriteSection`
//! and `imclose` `iiuClose`.  A list-directed or `(a)` `read` with no
//! `END=`/`ERR=` that fails is the gfortran runtime error, status 2.

use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::memory_error;
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdsecl};
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use crate::imod::libiimod::mrcfiles::MRC_LABEL_SIZE;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position, iiu_write_section};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_extended_type, iiu_alt_mode, iiu_alt_num_extended, iiu_alt_sample,
    iiu_alt_size, iiu_ret_cell, iiu_trans_header, iiu_write_header,
};
use std::io::{BufRead, Write};

/// Original program `avgstack` (`avgstack.f:28`).
pub fn avgstack() {
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    // `data nxyzst / 0.,0.,0. /`
    let nxyzst = [0_i32; 3];
    let mut mode = 0_i32;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut dat = [b' '; 9];
    let mut tim = [b' '; 8];
    let mut ifst: i32;
    let mut ilst: i32;

    // `read(*, 2000) var`, `2000 format(a)`, into a `character*320`.
    let read_name = || -> String {
        let mut line = Vec::new();
        match std::io::stdin().lock().read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => {
                let _ = std::io::stdout().flush();
                eprintln!("Fortran runtime error: End of file");
                exit(2);
            }
            Ok(_) => {}
        }
        if line.last() == Some(&b'\n') {
            line.pop();
        }
        line.truncate(320);
        String::from_utf8_lossy(&line).trim_end_matches(' ').to_owned()
    };
    //
    //	---------------------------------
    //	--- initialize and input info ---

    // `1000 format(1x, a6, ' file : ', $)`
    print!("  Input file : ");
    let _ = std::io::stdout().flush();
    let input = read_name();
    print!(" Output file : ");
    let _ = std::io::stdout().flush();
    let output = read_name();

    imopen(1, &input, "ro");
    imopen(2, &output, "new");

    //--- Get Image Header ---!
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
    if mode == 4 {
        errorexit("USE THE CLIP PROGRAM TO AVERAGE FFTS");
    }
    let maxarr = nxyz[0] * nxyz[1];
    let limarr = nxyz[0].max((5000 * 5000).min(maxarr));
    let mut array: Vec<f32> = Vec::new();
    let mut brray: Vec<f32> = Vec::new();
    let ierr = i32::from(
        array.try_reserve_exact(maxarr.max(0) as usize).is_err()
            || brray.try_reserve_exact(limarr.max(0) as usize).is_err(),
    );
    memory_error(ierr, "ARRAYS FOR IMAGE DATA");
    array.resize(maxarr.max(0) as usize, 0.0);
    brray.resize(limarr.max(0) as usize, 0.0);

    //--- first section is 0, so nz = total # sections - 1
    nxyz[2] -= 1;
    ifst = 0;
    ilst = nxyz[2];
    println!("  Sections to average [First,Last] (/ for all):");
    {
        let mut stdin = std::io::stdin().lock();
        if let Err(err) = list_read(
            &mut stdin,
            &mut [ListItem::Integer(&mut ifst), ListItem::Integer(&mut ilst)],
        ) {
            let _ = std::io::stdout().flush();
            match err {
                ListReadError::End => eprintln!("Fortran runtime error: End of file"),
                ListReadError::Error => {
                    eprintln!("Fortran runtime error: Bad integer for item 1 in list input")
                }
            }
            exit(2);
        }
    }

    //--- Check end limit ---!
    ilst = ilst.min(nxyz[2]);
    //--- Check beginning limit ---!
    ifst = ifst.min(nxyz[2]);
    //--- Check beginning/end order ---!
    ifst = ifst.min(ilst);

    //	-------------------------------
    //	--- Clear the storage array ---

    for value in array[..(nxyz[1] * nxyz[0]).max(0) as usize].iter_mut() {
        *value = 0.0;
    }

    iiu_trans_header(2, 1);
    iiu_alt_mode(2, 2);
    iiu_alt_num_extended(2, 0);
    iiu_alt_extended_type(2, &[0, 0]);

    nxyz[2] = 1;
    mxyz[2] = 1;
    iiu_alt_sample(2, &mxyz);
    iiu_alt_size(2, &nxyz, &nxyzst);
    let mut cell = iiu_ret_cell(2);
    cell[2] = cell[0] / mxyz[0] as f32;
    iiu_alt_cell(2, &cell);

    b3d_date(&mut dat);
    time(&mut tim);
    let numsec = ilst - ifst + 1;

    //       7/7/00 CER: remove the encodes
    // `3000 format('AVGSTACK: ', i4, ' sections averaged.', t57, a9, 2x, a8)`
    let mut title = [b' '; MRC_LABEL_SIZE];
    let head = format!("AVGSTACK: {:>4} sections averaged.", numsec);
    let head = if head.len() == 33 {
        head.into_bytes()
    } else {
        // An `i4` too narrow for the value writes four asterisks.
        b"AVGSTACK: **** sections averaged.".to_vec()
    };
    title[..head.len()].copy_from_slice(&head);
    title[56..65].copy_from_slice(&dat);
    title[67..75].copy_from_slice(&tim);

    let (nx, ny) = (nxyz[0], nxyz[1]);
    let max_lines = limarr / nx;
    let num_chunks = (ny + max_lines - 1) / max_lines;

    //	------------------------------------------------------------
    //	--- Add all requested sections from the input image file ---

    for isec in ifst..=ilst {
        println!("  Adding section {:>12}", isec);
        let mut iy: usize = 0;
        unsafe {
            iiu_set_position(1, isec, 0);
        }
        for i_chunk in 1..=num_chunks {
            let num_lines = max_lines.min(ny - (i_chunk - 1) * max_lines);
            if unsafe { irdsecl(1, &mut brray, num_lines) }.is_err() {
                // 100 call errorexit('READING FILE')
                errorexit("READING FILE");
            }
            for ix in 0..(nx * num_lines) as usize {
                array[iy] += brray[ix];
                iy += 1;
            }
        }
    }

    unsafe {
        iiu_close(1);
    }

    //	--------------------------------
    //	--- Average the output image ---
    println!("  Averaging stack...");
    for value in array[..(nx * ny) as usize].iter_mut() {
        *value /= numsec as f32;
    }

    //	-----------------------------
    //	--- Write out the average ---

    array_min_max_mean_fortran(
        &array, &nx, &ny, &1, &nx, &1, &ny, &mut dmin, &mut dmax, &mut dmean,
    );

    iiu_write_header(2, &title, 1, dmin, dmax, dmean);

    unsafe {
        iiu_write_section(2, array.as_mut_ptr().cast());
        iiu_close(2);
    }

    exit(0);
}

/// Original subroutine `errorexit` (`avgstack.f:163`).
pub fn errorexit(message: &str) -> ! {
    println!();
    println!(" ERROR: AVGSTACK - {}", message);
    let _ = std::io::stdout().flush();
    exit(1);
}
