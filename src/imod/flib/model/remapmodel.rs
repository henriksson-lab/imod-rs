//! Translation of `IMOD/flib/model/remapmodel.f90`.
//!
//! This program allows one to remap coordinates in a model, in two ways:
//! 1) The set of Z values may be mapped, one-to-one, to any arbitrary new set
//! of Z values; and 2) The X, Y or Z coordinates may be shifted by a
//! constant.  Also, a mapping can be set up easily for the case where serial
//! section tomograms are being rejoined with different spacings.
//! David Mastronarde 5/8/89.
//!
//! The main program maps to [`remapmodel`]; the `fortmodel` module arrays are
//! the [`FortModel`] that `readw_or_imod` fills.  Fortran 1-based arrays are
//! 0-based vectors here: `object(k)` is `fm.object[k - 1]` and
//! `p_coord(3, ipnt)` is `fm.p_coord[ipnt - 1][2]`.

use crate::imod::flib::subrs::compat::gfortran_rt::cvttss2si;
use crate::imod::flib::subrs::hvem::b3dxor::b3dxor;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::rdlist::{parselist, rdlist};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::flib::subrs::model::scale_model::scale_model;
use crate::imod::flib::subrs::model::write_wmod::write_wmod;
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libcfshr::parse_params::{
    pip_get_boolean, pip_get_float, pip_get_integer, pip_get_integer_array, pip_get_three_floats,
};
use crate::imod::libcfshr::piecefuncs::fill_listz;
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::writelist::wrlist;
use crate::imod::libimod::imodel_fwrap::{getimodheado, getimodmaxes, putimodmaxes, putimodzscale};
use std::io::{BufReader, Write};

/// `parameter (LIMSEC = 100000, ...)` (`remapmodel.f90:21`).
const LIMSEC: i32 = 100000;
/// `LIMINDEX = 10 * LIMSEC`.
const LIMINDEX: i32 = 10 * LIMSEC;
/// `LIMCHUNK = 10000`.
const LIMCHUNK: i32 = 10000;
/// `parameter (numOptions = 13)` (`remapmodel.f90:52`).
const REMAPMODEL_NUM_OPTIONS: i32 = 13;
/// Fallback PIP table, the `options(1)` string (`remapmodel.f90:54-59`).
const REMAPMODEL_OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@old:OldZList:LI:@\
full:FullRangeInZ:B:@new:NewZList:LI:@add:AddToAllPoints:FT:@\
reorder:ReorderPointsInZ:I:@fromchunks:FromChunkLimits:IA:@\
tochunks:ToChunkLimits:IA:@thick:ThicknessFile:FN:@\
long:LongRangeThickness:FT:@pixel:PixelSizeOrScaling:F:@help:usage:B:";

/// Original program `remapmodel` (`remapmodel.f90:17`).
pub fn remapmodel() {
    let mut fm = FortModel::default();
    let mut list_z = vec![0_i32; LIMSEC as usize];
    let mut new_list = vec![0_i32; LIMSEC as usize];
    let mut index_zlist = vec![0_i32; LIMINDEX as usize];
    let mut iz_st_end_from = vec![0_i32; LIMCHUNK as usize];
    let mut iz_st_end_to = vec![0_i32; LIMCHUNK as usize];
    let mut z_new_list = vec![0.0_f32; LIMSEC as usize];
    // A large main-program array: static storage, so zero before any read.
    let mut thicknesses = vec![0.0_f32; LIMSEC as usize];
    //
    let mut model_file = String::new();
    let mut new_model = String::new();
    let mut thick_file = [b' '; 320];
    let mut list_string = [b' '; 1024];
    let mut do_thickness: bool;
    //
    let mut num_in_zlist: i32;
    let mut nlist_z: i32 = 0;
    let mut num_new: i32;
    let mut num_thick: i32;
    let mut if_reorder: i32;
    let mut if_add: i32;
    let mut if_entered: i32;
    let (mut max_x, mut max_y, mut max_z) = (0_i32, 0_i32, 0_i32);
    let mut min_z: i32;
    let mut max_index: i32;
    let mut ierr: i32;
    let (mut num_from, mut num_to): (i32, i32);
    let (mut xadd, mut y_add, mut z_add): (f32, f32, f32);
    let mut cum_thick: f32;
    let mut scale_thick: f32;
    let (mut big_thickness, mut big_sec_start, mut big_sec_end) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut pixel_size = 0.0_f32;
    let mut zscale = 0.0_f32;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);

    // `read(*,*) ...` with no END=/ERR=: a failure is the runtime's error.
    let read_stdin_list = |items: &mut [ListItem]| {
        let mut stdin = std::io::stdin().lock();
        if let Err(err) = list_read(&mut stdin, items) {
            let _ = std::io::stdout().flush();
            match err {
                ListReadError::End => eprintln!("Fortran runtime error: End of file"),
                ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
            }
            exit(2);
        }
    };
    //
    if_reorder = 0;
    xadd = 0.;
    y_add = 0.;
    z_add = 0.;
    if_entered = 0;
    do_thickness = false;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[REMAPMODEL_OPTIONS],
        REMAPMODEL_NUM_OPTIONS,
        "remapmodel",
        "ERROR: REMAPMODEL - ",
        true,
        3,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    let pip_input = num_opt_arg + num_non_opt_arg > 0;
    //
    if pip_get_in_out_file(
        "InputFile",
        1,
        "Name of input model file",
        &mut model_file,
        320,
    ) != 0
    {
        exit_error("No input file specified");
    }
    // Fixed in translation (BUGS.md, `remapmodel`): the source asks for
    // option `InputFile` here (`remapmodel.f90:75`), so `-output` is ignored
    // and the model is written over the input file named by `-input`.
    if pip_get_in_out_file(
        "OutputFile",
        2,
        "Name of output model file",
        &mut new_model,
        320,
    ) != 0
    {
        exit_error("No output file specified");
    }
    //
    // read in the model
    //
    let exist = readw_or_imod(model_file.trim_end_matches(' '), &mut fm);
    if !exist {
        exit_error("Reading model file");
    }
    let mut iz_list: Vec<i32> = Vec::new();
    ierr = i32::from(iz_list.try_reserve_exact(fm.max_pt as usize).is_err());
    memory_error(ierr, "array for point z values");
    iz_list.resize(fm.max_pt as usize, 0);
    //
    // scale to index coordinates
    //
    scale_model(0, &mut fm);
    //
    // make list of all z values in the model and set default output list
    //
    num_in_zlist = 0;
    for iobj in 1..=fm.max_mod_obj {
        for ind_obj in 1..=fm.npt_in_obj[(iobj - 1) as usize] {
            let ipnt = fm.object[(ind_obj + fm.ibase_obj[(iobj - 1) as usize] - 1) as usize].abs();
            let iz_val = fm.p_coord[(ipnt - 1) as usize][2].round() as i32;
            if_add = 1;
            let mut i = 1;
            while i <= num_in_zlist && if_add == 1 {
                if iz_list[(i - 1) as usize] == iz_val {
                    if_add = 0;
                }
                i += 1;
            }
            if if_add == 1 {
                num_in_zlist += 1;
                if num_in_zlist > LIMSEC {
                    exit_error("Too many Z values for arrays");
                }
                iz_list[(num_in_zlist - 1) as usize] = iz_val;
            }
        }
    }
    {
        let mut count = 0_usize;
        fill_listz(&iz_list[..num_in_zlist as usize], &mut list_z, &mut count);
        nlist_z = count as i32;
    }
    min_z = list_z[0];
    max_index = list_z[(nlist_z - 1).max(0) as usize];
    for i in 0..nlist_z as usize {
        new_list[i] = list_z[i];
    }
    num_new = nlist_z;
    let _ = getimodmaxes(&mut max_x, &mut max_y, &mut max_z);
    //
    // get options from pip input
    //
    if pip_input {
        let _ = pip_get_integer(b"ReorderPointsInZ", &mut if_reorder);
        if_add = 1 - pip_get_three_floats(b"AddToAllPoints", &mut xadd, &mut y_add, &mut z_add);
        num_from = 0;
        num_to = 0;
        let _ = pip_get_integer_array(b"FromChunkLimits", &mut iz_st_end_from, &mut num_from, LIMCHUNK);
        let _ = pip_get_integer_array(b"ToChunkLimits", &mut iz_st_end_to, &mut num_to, LIMCHUNK);
        if b3dxor(num_from > 0, num_to > 0) {
            exit_error("You must enter both -fromchunks and -tochunks or neither");
        }
        if pipgetstring_(b"ThicknessFile", &mut thick_file) == 0 {
            do_thickness = true;
            if if_reorder != 0 || num_from > 0 {
                exit_error("You cannot endter -reorder or -tochunks with thicknesses");
            }
            let mut unit3 = BufReader::new(dopen(3, &fortran_string(&thick_file), "ro", "f"));
            num_thick = 1;
            // 10 read(3, *, err = 20, end = 30) thicknesses(numThick)
            loop {
                match list_read(
                    &mut unit3,
                    &mut [ListItem::Real(&mut thicknesses[(num_thick - 1) as usize])],
                ) {
                    Ok(()) => {}
                    // 20 call exitError('Reading thickness file')
                    Err(ListReadError::Error) => exit_error("Reading thickness file"),
                    Err(ListReadError::End) => break,
                }
                num_thick += 1;
                if num_thick > LIMSEC {
                    exit_error("Too many thicknesses for arrays");
                }
            }
            // 30
            scale_thick = 1.;

            // Get possible parameters for scaling the values to long-range accuracy
            if pip_get_three_floats(
                b"LongRangeThickness",
                &mut big_thickness,
                &mut big_sec_start,
                &mut big_sec_end,
            ) == 0
            {
                if big_sec_start < 0. || (big_sec_end.round() as i32) >= num_thick {
                    exit_error("Starting or ending section for long range thickness out of range");
                }
                // Fixed in translation (BUGS.md, `remapmodel`): the source
                // tests `nint(bigSecStart) <= nint(bigSecEnd)`, the inverse of
                // what its message says, and sums `do i = 1, nint(bigSecStart)
                // + 1, nint(bigSecEnd)`.  Defined: a start that is not below
                // the end is the error, and the sum runs over the thicknesses
                // from the starting section to the one before the ending
                // section, `nint(start) + 1 .. nint(end)`.
                if big_sec_start.round() as i32 >= big_sec_end.round() as i32 {
                    exit_error(
                        "Starting section number for long range thickness is larger than ending section",
                    );
                }
                cum_thick = 0.;
                for i in (big_sec_start.round() as i32 + 1)..=(big_sec_end.round() as i32) {
                    cum_thick += thicknesses[(i - 1) as usize];
                }
                scale_thick = big_thickness / cum_thick;
            }

            // Get a pixel size one way or the other: the Z values need to be divided by that
            // so incorporate it into the scaling
            ierr = getimodheado(&mut pixel_size, &mut zscale);
            if ierr == 0 {
                pixel_size *= 1000.;
            }
            if pip_get_float(b"PixelSizeOrScaling", &mut pixel_size) != 0 && ierr != 0 {
                exit_error("A pixel size or scaling must be entered; the model has no pixel size");
            }
            scale_thick /= pixel_size;

            cum_thick = 0.;
            for i in 0..num_thick as usize {
                z_new_list[i] = scale_thick * (cum_thick + 0.5 * thicknesses[i]);
                cum_thick += thicknesses[i];
            }

            if max_index >= num_thick {
                exit_error("There are not thicknesses in the file to correct all model Z values");
            }
            max_z = max_index + 1;
        }

        if num_from == 0 {
            //
            // No chunks: get old list if entered, or full one
            //
            let _ = pip_get_boolean(b"FullRangeInZ", &mut if_entered);
            if pipgetstring_(b"OldZList", &mut list_string) == 0 {
                if do_thickness {
                    exit_error("You cannot enter an old list with thicknesses");
                }
                if if_entered != 0 {
                    exit_error("You cannot enter both an old list AND -full");
                }
                let _ = parselist(&fortran_string(&list_string), &mut list_z, &mut nlist_z);
                if nlist_z > LIMSEC {
                    exit_error("Too many Z values in new list for arrays");
                }
                if_entered = 1;
                for i in 0..(nlist_z - 1).max(0) as usize {
                    if list_z[i] >= list_z[i + 1] {
                        exit_error("Input list of Z values must be in increasing order");
                    }
                }
            } else if if_entered != 0 || do_thickness {
                if max_z > LIMSEC {
                    exit_error("Model range in Z too big for arrays");
                }
                for i in 1..=max_z {
                    list_z[(i - 1) as usize] = i - 1;
                }
                nlist_z = max_z;
            }
            //
            // Make sure input list is still default for output
            //
            for i in 0..nlist_z.max(0) as usize {
                new_list[i] = list_z[i];
            }
            num_new = nlist_z;
            //
            // No chunks: get new list; it's required unless adding to coordinates
            //
            if pipgetstring_(b"NewZList", &mut list_string) == 0 {
                if do_thickness {
                    exit_error("You cannot enter a new Z list with thicknesses");
                }
                let _ = parselist(&fortran_string(&list_string), &mut new_list, &mut num_new);
                if num_new > LIMSEC {
                    exit_error("Too many Z values for arrays");
                }
            } else if if_add == 0 && !do_thickness {
                exit_error("No list of new Z values specified");
            }
        } else {
            //
            // Chunks: build up in and out lists
            //
            if pip_get_boolean(b"FullRangeInZ", &mut if_entered)
                + pipgetstring_(b"OldZList", &mut list_string)
                + pipgetstring_(b"NewZList", &mut list_string)
                != 3
            {
                exit_error("You cannot enter -new, -old, or -full with chunk entries");
            }
            if num_from != num_to {
                exit_error("Same number of values required with -tochunk and -fromchunk");
            }
            nlist_z = 0;
            let mut iz_base = 0_i32;
            for ich in 1..=num_from / 2 {
                let iz_start_old = iz_st_end_from[(2 * ich - 2) as usize];
                let iz_end_old = iz_st_end_from[(2 * ich - 1) as usize];
                let num_in_sec = (iz_start_old - iz_end_old).abs() + 1;
                let mut idir_old = 1_i32;
                let mut idir_new = 1_i32;
                let iz_start_new = iz_st_end_to[(2 * ich - 2) as usize];
                let iz_end_new = iz_st_end_to[(2 * ich - 1) as usize];
                if iz_start_old > iz_end_old {
                    idir_old = -1;
                }
                if iz_start_new > iz_end_new {
                    idir_new = -1;
                }
                if nlist_z + num_in_sec > LIMSEC {
                    exit_error("Chunk ranges make too many values for arrays");
                }
                //
                // For each z value in a chunk, get z value inside original section
                // and figure out the fate of that z value in the new joined volume
                //
                for i in 0..num_in_sec {
                    let iz_chunk = iz_start_old + i * idir_old;
                    let iz = i + nlist_z;
                    list_z[iz as usize] = iz;
                    let mut iz_out = -999_i32;
                    if idir_new * (iz_chunk - iz_start_new) >= 0
                        && idir_new * (iz_chunk - iz_end_new) <= 0
                    {
                        iz_out = iz_base + (iz_chunk - iz_start_new) / idir_new;
                    }
                    new_list[iz as usize] = iz_out;
                }
                nlist_z += num_in_sec;
                iz_base += (iz_start_new - iz_end_new).abs() + 1;
            }
            num_new = nlist_z;
            min_z = min_z.min(0);
            if_entered = 1;
            println!(" Z values being mapped:");
            let _ = std::io::stdout().flush();
            wrlist(&list_z, &nlist_z);
            println!(" Values being mapped to:");
            let _ = std::io::stdout().flush();
            wrlist(&new_list, &num_new);
        }
    }
    //
    if if_entered == 0 && !do_thickness {
        println!(" The current list of Z values (nearest integers) in model is:");
        let _ = std::io::stdout().flush();
        wrlist(&list_z, &nlist_z);
    }
    //
    // get new list of z values for remapping
    //
    if !pip_input {
        // write(*,'(/,a,i4,a,/,a,/,a,/,a)')
        println!();
        println!(
            " Enter new list of{:>4} Z values to remap these values to.",
            nlist_z
        );
        println!(" Enter ranges just as above, or / to leave list alone,");
        println!("   or -1 to replace each Z value with its negative.");
        println!(" Use numbers from -999 to -990 to remove points with a particular Z value.");
        let _ = std::io::stdout().flush();
        {
            let mut stdin = std::io::stdin().lock();
            let _ = rdlist(&mut stdin, &mut new_list, &mut num_new);
        }
        if num_new > LIMSEC {
            exit_error("Too many Z values for arrays");
        }
        //
        //
        print!(" Amounts to ADD to ALL X, Y and Z coordinates: ");
        let _ = std::io::stdout().flush();
        read_stdin_list(&mut [
            ListItem::Real(&mut xadd),
            ListItem::Real(&mut y_add),
            ListItem::Real(&mut z_add),
        ]);
    }

    if num_new == 1 && new_list[0] == -1 {
        for i in 0..nlist_z.max(0) as usize {
            new_list[i] = -list_z[i];
        }
    } else if num_new != nlist_z {
        exit_error("Number of Z values does not correspond between old and new lists");
    }
    if !do_thickness {
        for i in 0..nlist_z.max(0) as usize {
            z_new_list[i] = new_list[i] as f32;
        }
    }
    //
    // Build index list from all Z's that are being mapped
    // Initialize list for range of anything in model or input z list
    //
    min_z = min_z.min(list_z[0]);
    max_index = max_index.max(list_z[(nlist_z - 1).max(0) as usize]);
    if max_index - min_z + 1 > LIMINDEX {
        exit_error("Too many Z values for index list array");
    }
    for i in min_z..=max_index {
        index_zlist[(i - min_z) as usize] = 0;
    }
    for i in 1..=nlist_z {
        index_zlist[(list_z[(i - 1) as usize] - min_z) as usize] = i;
    }
    //
    // see if there are any Z's out of order: find direction between
    // first retained Z's and make sure all other intervals match
    //
    let mut in_order = 1_i32;
    let mut idir_first = 0_i32;
    let mut last_keep = 0_i32;
    for i in 1..=nlist_z {
        let znew = z_new_list[(i - 1) as usize];
        if znew < -999. || znew > -990. {
            //
            // if this is a retained Z and there is a previous one
            // get the sign of the interval
            //
            if last_keep > 0 {
                // `idir = sign(1., zNewList(i) - zNewList(lastKeep))`
                let idir = 1.0_f32.copysign(znew - z_new_list[(last_keep - 1) as usize]) as i32;
                //
                // store the sign the first time, compare after that
                //
                if idir_first == 0 {
                    idir_first = idir;
                } else if idir != idir_first {
                    in_order = 0;
                }
            }
            last_keep = i;
        }
    }
    //
    if in_order == 0 && !pip_input {
        print!(
            " Your new Z's are not in monotonically increasing order.\n Enter 1 or -1 to reorder points in each object by increasing or decreasing\n   new Z value, or 0 not to: "
        );
        let _ = std::io::stdout().flush();
        read_stdin_list(&mut [ListItem::Integer(&mut if_reorder)]);
    }
    //
    // loop through objects, recompute # of objects
    //
    fm.n_object = 0;
    for iobj in 1..=fm.max_mod_obj {
        let mut num_in_obj = fm.npt_in_obj[(iobj - 1) as usize];
        let mut ind_obj = 1_i32;
        let ibase = fm.ibase_obj[(iobj - 1) as usize];
        while ind_obj <= num_in_obj {
            let ipnt = fm.object[(ind_obj + ibase - 1) as usize];
            if ipnt > 0 && ipnt <= fm.n_point {
                //
                // for valid point, get new z value
                //
                let iz_val = fm.p_coord[(ipnt - 1) as usize][2].round() as i32;
                let index = index_zlist[(iz_val - min_z) as usize];
                let mut z_new_val = iz_val;
                if index > 0 {
                    // integer = real: truncation, as `cvttss2si`.
                    z_new_val = cvttss2si(z_new_list[(index - 1) as usize]);
                }
                //
                // just shift values if not -999
                //
                if z_new_val < -999 || z_new_val > -990 {
                    let point = &mut fm.p_coord[(ipnt - 1) as usize];
                    point[0] += xadd;
                    point[1] += y_add;
                    point[2] = point[2] + z_add + z_new_val as f32 - iz_val as f32;
                    ind_obj += 1;
                    max_z = max_z.max(point[2].round() as i32);
                } else {
                    //
                    // or delete point by moving rest of pointers in object down
                    //
                    num_in_obj -= 1;
                    for ind_move in ind_obj..=num_in_obj {
                        fm.object[(ind_move + ibase - 1) as usize] =
                            fm.object[(ind_move + ibase) as usize];
                    }
                }
            } else {
                // Fixed in translation (BUGS.md, `remapmodel`): the source
                // advances only past a valid point, so an object entry that is
                // not a point number makes `do while` spin forever.  Defined:
                // such an entry is left as it is and skipped.
                ind_obj += 1;
            }
        }
        fm.npt_in_obj[(iobj - 1) as usize] = num_in_obj;
        if num_in_obj > 0 {
            fm.n_object += 1;
        }
        //
        // if reordering, sort object array by Z value
        //
        if if_reorder != 0 {
            for i in 1..num_in_obj {
                for j in i..=num_in_obj {
                    let pz_j = fm.p_coord[(fm.object[(j + ibase - 1) as usize].abs() - 1) as usize][2];
                    let pz_i = fm.p_coord[(fm.object[(i + ibase - 1) as usize].abs() - 1) as usize][2];
                    if if_reorder as f32 * (pz_j - pz_i) < 0. {
                        fm.object.swap((i + ibase - 1) as usize, (j + ibase - 1) as usize);
                    }
                }
            }
        }
        //
    }
    //
    // scale data back
    //
    scale_model(1, &mut fm);
    let _ = putimodmaxes(max_x, max_y, max_z);
    if do_thickness {
        putimodzscale(1.);
    }
    write_wmod(new_model.trim_end_matches(' '), &mut fm);
    let _ = std::io::stdout().flush();
    exit(0);
}
