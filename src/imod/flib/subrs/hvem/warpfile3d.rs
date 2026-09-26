//! Translation of `IMOD/flib/subrs/hvem/warpfile3d.f90`.
//!
//! Functions to read in a 3D warping file, used by both Warpvol and Findwarp.
//!
//! Fortran unit 1, which `readWarpFileHeader` opens and `readWarpTransforms`
//! reads and closes, is the `BufReader` the first returns and the second
//! takes; the close is its drop.

use crate::imod::flib::subrs::compat::gfortran_rt::{maxss, minss};
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, frefor, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::exit_error;
use crate::imod::libcfshr::b3dutil::fortran_string;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

/// Original `readWarpFileHeader` (`warpfile3d.f90:8`).
///
/// Open the file and read the header line, returning number of positions and
/// the starts and intervals if they are defined.  It returns the number of
/// positions swapped between X and Y but doesn't swap any oter values.
///
/// `file_inv` is the caller's `character*(*)` variable: the source reads the
/// header line back into it (`read(1, '(a)') fileInv`), so the caller sees
/// the line afterwards, cut or blank padded to the variable's length.  The
/// line has no `END=`: an empty file is the gfortran runtime's end-of-file
/// error, status 2.
#[allow(clippy::too_many_arguments)]
pub fn read_warp_file_header(
    file_inv: &mut [u8],
    num_input: &mut i32,
    num_loc_x: &mut i32,
    num_loc_y: &mut i32,
    num_loc_z: &mut i32,
    x_loc_start: &mut f32,
    y_loc_start: &mut f32,
    z_loc_start: &mut f32,
    dx_loc: &mut f32,
    dy_loc: &mut f32,
    dz_loc: &mut f32,
    xf_scale: f32,
) -> BufReader<File> {
    let mut free_input = [0.0_f32; 10];
    let mut unit1 = BufReader::new(dopen(1, &fortran_string(file_inv), "ro", "f"));
    let mut record: Vec<u8> = Vec::new();
    match unit1.read_until(b'\n', &mut record) {
        Ok(0) | Err(_) => {
            let _ = std::io::stdout().flush();
            eprintln!("Fortran runtime error: End of file");
            crate::imod::libcfshr::b3dutil::exit(2);
        }
        Ok(_) => {}
    }
    if record.last() == Some(&b'\n') {
        record.pop();
    }
    record.resize(file_inv.len(), b' ');
    file_inv.copy_from_slice(&record[..file_inv.len()]);
    frefor(
        &String::from_utf8_lossy(file_inv),
        &mut free_input,
        num_input,
    );
    if *num_input > 3 && *num_input < 9 {
        *num_loc_x = 0;
        *num_loc_y = 0;
        *num_loc_z = 0;
        return unit1;
    }
    *num_loc_y = free_input[0].round() as i32;
    if *num_input == 2 {
        *num_loc_x = 1;
        *num_loc_z = free_input[1].round() as i32;
    } else {
        *num_loc_x = free_input[1].round() as i32;
        *num_loc_z = free_input[2].round() as i32;
        if *num_input == 9 {
            *x_loc_start = free_input[3] * xf_scale;
            *y_loc_start = free_input[4] * xf_scale;
            *z_loc_start = free_input[5] * xf_scale;
            *dx_loc = free_input[6] * xf_scale;
            *dy_loc = free_input[7] * xf_scale;
            *dz_loc = free_input[8] * xf_scale;
        }
    }
    unit1
}

/// Original `readWarpTransforms` (`warpfile3d.f90:52`).
///
/// Read the transforms, deducing the start and interval if necessary when
/// there is a full set of positions.  It fsets solveTemp for positions that
/// are present.
///
/// `aloc_temp(3, 3, numLocTot)` and `dloc_temp(3, numLocTot)` are column
/// major; `unit1` is the file `readWarpFileHeader` opened, closed (dropped)
/// on return.  `min`/`max` of the positions are gfortran `MIN`/`MAX` of reals,
/// written as `f32::min`/`max`, which differ from them only for a NaN
/// position.
#[allow(clippy::too_many_arguments)]
pub fn read_warp_transforms(
    mut unit1: BufReader<File>,
    num_input: i32,
    num_loc_tot: i32,
    num_loc_x: i32,
    num_loc_y: i32,
    num_loc_z: i32,
    x_loc_start: &mut f32,
    y_loc_start: &mut f32,
    z_loc_start: &mut f32,
    dx_loc: &mut f32,
    dy_loc: &mut f32,
    dz_loc: &mut f32,
    xf_scale: f32,
    x_loc_max: &mut f32,
    y_loc_max: &mut f32,
    z_loc_max: &mut f32,
    aloc_temp: &mut [f32],
    dloc_temp: &mut [f32],
    solve_temp: &mut [bool],
) {
    let mut cen_loc_x = 0.0_f32;
    let mut cen_loc_y = 0.0_f32;
    let mut cen_loc_z = 0.0_f32;
    let mut l_for_err: i32 = 0;
    let mut l: i32;
    let (mut ix, mut iy, mut iz): (i32, i32, i32);

    // `98 write(errLine, ...) ...; call exitError(errLine)`
    let error98 = |l_for_err: i32| -> ! {
        let err_line = format!(
            "Reading warp file at position{:>7} - ~ line{:>8}",
            l_for_err,
            2 + 4 * (l_for_err - 1)
        );
        exit_error(&err_line);
    };
    // `((alocTemp(i, j, l), j = 1, 3), dlocTemp(i, l), i = 1, 3)`
    let read_xform = |unit1: &mut BufReader<File>,
                      aloc: &mut [f32],
                      dloc: &mut [f32],
                      l: i32|
     -> Result<(), ListReadError> {
        let base = ((l - 1) * 9) as usize;
        let dbase = ((l - 1) * 3) as usize;
        let mut vals = [0.0_f32; 12];
        for i in 0..3 {
            for j in 0..3 {
                vals[i * 4 + j] = aloc[base + i + j * 3];
            }
            vals[i * 4 + 3] = dloc[dbase + i];
        }
        let result = {
            let mut items: Vec<ListItem> = vals.iter_mut().map(ListItem::Real).collect();
            list_read(unit1, &mut items)
        };
        // Items read before a failure are stored, as the runtime stores them.
        for i in 0..3 {
            for j in 0..3 {
                aloc[base + i + j * 3] = vals[i * 4 + j];
            }
            dloc[dbase + i] = vals[i * 4 + 3];
        }
        result
    };

    if num_input != 9 {
        //
        // Every position must be present, find min and max to deduce interval
        //
        *x_loc_start = 1.0e10;
        *x_loc_max = -1.0e10;
        *y_loc_start = 1.0e10;
        *y_loc_max = -1.0e10;
        *z_loc_start = 1.0e10;
        *z_loc_max = -1.0e10;

        l = 1;
        while l <= num_loc_tot {
            l_for_err = l;
            if list_read(
                &mut unit1,
                &mut [
                    ListItem::Real(&mut cen_loc_x),
                    ListItem::Real(&mut cen_loc_y),
                    ListItem::Real(&mut cen_loc_z),
                ],
            )
            .is_err()
            {
                error98(l_for_err);
            }
            if read_xform(&mut unit1, aloc_temp, dloc_temp, l).is_err() {
                error98(l_for_err);
            }
            cen_loc_x *= xf_scale;
            cen_loc_y *= xf_scale;
            cen_loc_z *= xf_scale;
            let dbase = ((l - 1) * 3) as usize;
            for k in 0..3 {
                dloc_temp[dbase + k] *= xf_scale;
            }
            solve_temp[(l - 1) as usize] = true;
            // `warpfile3d.f90:78-83`: the reference object has the running
            // start as `minss` destination and the center as `maxss`
            // destination.
            *x_loc_start = minss(*x_loc_start, cen_loc_x);
            *y_loc_start = minss(*y_loc_start, cen_loc_y);
            *z_loc_start = minss(*z_loc_start, cen_loc_z);
            *x_loc_max = maxss(cen_loc_x, *x_loc_max);
            *y_loc_max = maxss(cen_loc_y, *y_loc_max);
            *z_loc_max = maxss(cen_loc_z, *z_loc_max);
            l += 1;
        }
        *dx_loc = (*x_loc_max - *x_loc_start) / 1.max(num_loc_y - 1) as f32;
        *dy_loc = (*y_loc_max - *y_loc_start) / 1.max(num_loc_x - 1) as f32;
        *dz_loc = (*z_loc_max - *z_loc_start) / 1.max(num_loc_z - 1) as f32;
    }

    //
    // Now fix the dlocs if only one position, then read other way
    if num_loc_y == 1 {
        *dx_loc = 1.;
    }
    if num_loc_x == 1 {
        *dy_loc = 1.;
    }
    if num_loc_z == 1 {
        *dz_loc = 1.;
    }
    if num_input == 9 {
        //
        // Know min and interval, so read each entry, find where it goes in
        // array and put it there
        for l in 1..=num_loc_tot {
            solve_temp[(l - 1) as usize] = false;
        }
        l_for_err = 1;
        loop {
            match list_read(
                &mut unit1,
                &mut [
                    ListItem::Real(&mut cen_loc_x),
                    ListItem::Real(&mut cen_loc_y),
                    ListItem::Real(&mut cen_loc_z),
                ],
            ) {
                Ok(()) => {}
                Err(ListReadError::End) => break,
                Err(ListReadError::Error) => error98(l_for_err),
            }
            cen_loc_x *= xf_scale;
            cen_loc_y *= xf_scale;
            cen_loc_z *= xf_scale;
            ix = num_loc_y.min(1.max(((cen_loc_x - *x_loc_start) / *dx_loc + 1.).round() as i32));
            iy = num_loc_x.min(1.max(((cen_loc_y - *y_loc_start) / *dy_loc + 1.).round() as i32));
            iz = num_loc_z.min(1.max(((cen_loc_z - *z_loc_start) / *dz_loc + 1.).round() as i32));
            l = ix + (iy - 1) * num_loc_y + (iz - 1) * num_loc_y * num_loc_x;
            if read_xform(&mut unit1, aloc_temp, dloc_temp, l).is_err() {
                error98(l_for_err);
            }
            let dbase = ((l - 1) * 3) as usize;
            for k in 0..3 {
                dloc_temp[dbase + k] *= xf_scale;
            }
            solve_temp[(l - 1) as usize] = true;
            l_for_err += 1;
        }
    }
    // `12 close(1)`: dropping the reader.
    drop(unit1);
}
