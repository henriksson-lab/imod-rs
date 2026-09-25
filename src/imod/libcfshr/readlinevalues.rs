//! Translation of `IMOD/libcfshr/readlinevalues.c`: flexible reading of lines
//! from a file into arrays.

use super::b3dutil::{CArg, ImodFile, c_format_bytes, fgetline};
use super::parse_params::{
    PIP_DOUBLE, PIP_FLOAT, PIP_INTEGER, PipValueArray, exit_error, pip_get_error,
    pip_get_line_of_values,
};

/// `cfsemshare.h:567`: `#define RLFV_SEPARATE_LINES  1`.
pub const RLFV_SEPARATE_LINES: i32 = 1;

/// One of the pointers `readLinesForValues` takes through its `...`: an
/// `int *`, `float *` or `double *` according to the matching letter of
/// `types`.  A variant that does not match its letter, or a missing entry, is
/// the source's `-7` programming error (its NULL-pointer and bad-letter
/// returns), since a C caller has no other way to get the pairing wrong.
pub enum ReadValueArray<'a> {
    Integers(&'a mut [i32]),
    Floats(&'a mut [f32]),
    Doubles(&'a mut [f64]),
}

/// Original C `readLinesForValues` (`readlinevalues.c:35`).
///
/// Reads values from one or more lines of `fp` and places them repeatedly into
/// the successive arrays in `args`, whose types are given by the letters of
/// `types` ("i", "f", "d").  `val_size` is the size of the value arrays.  A
/// `num_to_get_p` greater than zero is the number of sets of values to get; 0
/// gets all lines in the file with an error if they do not fit, -1 gets
/// whatever fits without an error.  The number obtained is returned in
/// `num_to_get_p` in the latter two cases.  Lines are read into `line`, of size
/// `max_line`.  With `RLFV_SEPARATE_LINES` in `flags` one set is read per line.
/// Returns -1 for a read error, -2 for end of file before `num_to_get_p` sets,
/// -3 for the array being full, -4 for a parse error, -5 for a non-integer
/// value for an integer argument, -6 for a memory error, -7 for a programming
/// error.
///
/// The temporary array is a `Vec` of the read type rather than a `char *`
/// aliased three ways; its unused tail is zero where `malloc` leaves it
/// uninitialised, which is only observable through PIP's comma defaults (a
/// skipped entry is never written), off unless `PipAllowCommaDefaults` was
/// called.
///
/// When the loop is never entered (`val_size <= 0` with no positive count),
/// the source reads `ierr` uninitialised in every test after the loop
/// (`BUGS.md`); it is 0 here, which takes the success path.
#[allow(clippy::too_many_arguments)]
pub fn read_lines_for_values(
    fp: &mut ImodFile,
    num_to_get_p: &mut i32,
    mut val_size: i32,
    line: &mut [u8],
    max_line: i32,
    flags: i32,
    types: &str,
    args: &mut [ReadValueArray<'_>],
) -> i32 {
    let types = types.as_bytes();
    let mut read_type = PIP_INTEGER;
    let num_args = types.len() as i32;
    let any_ints;
    let any_floats;
    let any_dbls;
    let mut data_size = size_of::<i32>() as i32;
    let num_to_get_in = *num_to_get_p;
    let mut num_to_get = num_to_get_in;
    let mut ierr: i32 = 0;
    let mut perr: i32;
    let mut num_on_line: i32;
    let mut num_got: i32 = 0;
    let separate_lines = if flags & RLFV_SEPARATE_LINES != 0 {
        1
    } else {
        0
    };
    let mut diff: f64;
    let mut int_use: Vec<i32> = Vec::new();
    let mut float_use: Vec<f32> = Vec::new();
    let mut dbl_use: Vec<f64> = Vec::new();
    if num_args == 0 {
        return -7;
    }

    /* Assess what types are needed and the type that has to be read as */
    any_ints = if types.contains(&b'i') { 1 } else { 0 };
    any_floats = if types.contains(&b'f') { 1 } else { 0 };
    any_dbls = if types.contains(&b'd') { 1 } else { 0 };
    if any_dbls != 0 {
        read_type = PIP_DOUBLE;
        data_size = size_of::<f64>() as i32;
    } else if any_floats != 0 {
        read_type = PIP_FLOAT;
        data_size = size_of::<f32>() as i32;
    }

    /* Get the pointers */
    for ind in 0..num_args as usize {
        match (types[ind], args.get(ind)) {
            (b'i', Some(ReadValueArray::Integers(_))) => {}
            (b'f', Some(ReadValueArray::Floats(_))) => {}
            (b'd', Some(ReadValueArray::Doubles(_))) => {}
            _ => return -7,
        }
    }

    /* If one arg, assign the argument pointer directly */
    if num_args != 1 {
        // Otherwise allocate the temp array big enough for all data */
        // `B3DMALLOC(char, dataSize * numArgs * valSize)`: an `int` product,
        // converted to `size_t` by `malloc`.
        let bytes = data_size.wrapping_mul(num_args).wrapping_mul(val_size) as i64 as u64;
        let count = (bytes / data_size as u64) as usize;
        let ok = if read_type == PIP_DOUBLE {
            dbl_use.try_reserve_exact(count).is_ok() && {
                dbl_use.resize(count, 0.);
                true
            }
        } else if read_type == PIP_FLOAT {
            float_use.try_reserve_exact(count).is_ok() && {
                float_use.resize(count, 0.);
                true
            }
        } else {
            int_use.try_reserve_exact(count).is_ok() && {
                int_use.resize(count, 0);
                true
            }
        };
        if !ok {
            return -6;
        }
        num_to_get = num_to_get.wrapping_mul(num_args);
        val_size = val_size.wrapping_mul(num_args);
    }
    if num_to_get <= 0 {
        num_to_get = val_size;
    }

    /* Read lines to get the requested, or all, data */
    perr = 0;
    while num_got < num_to_get {
        ierr = fgetline(fp, line, max_line);
        if ierr == -2 || ierr == -1 {
            break;
        }
        if ierr == 0 {
            continue;
        }
        num_on_line = if separate_lines != 0 { num_args } else { -1 };
        //printf("%d  %s\n", numOnLine, line);
        let len = line.iter().position(|&c| c == 0).unwrap_or(line.len());
        let text = &line[..len];
        let off = num_got as usize;
        let array = if num_args == 1 {
            match &mut args[0] {
                ReadValueArray::Integers(a) => PipValueArray::Int(&mut a[off..]),
                ReadValueArray::Floats(a) => PipValueArray::Float(&mut a[off..]),
                ReadValueArray::Doubles(a) => PipValueArray::Double(&mut a[off..]),
            }
        } else if read_type == PIP_DOUBLE {
            PipValueArray::Double(&mut dbl_use[off..])
        } else if read_type == PIP_FLOAT {
            PipValueArray::Float(&mut float_use[off..])
        } else {
            PipValueArray::Int(&mut int_use[off..])
        };
        perr = match pip_get_line_of_values(
            text,
            text,
            array,
            read_type,
            &mut num_on_line,
            val_size - num_got,
        ) {
            Ok(()) => 0,
            Err(()) => -1,
        };
        if perr != 0 {
            break;
        }
        num_got = (num_got + num_on_line).min(num_to_get);
        //printf("%d %d\n", numOnLine, numGot);
        if ierr < 0 {
            break;
        }
    }

    /* If no error condition, return the data to multiple args */
    if num_args > 1
        && !(perr != 0
            || ierr == -1
            || (ierr < 0 && num_to_get_in > 0 && num_got < num_to_get)
            || (ierr == 0 && num_to_get_in == 0 && num_got == val_size))
    {
        num_got = num_args * (num_got / num_args);
        let mut ent = 0;
        while ent < num_got / num_args && perr == 0 {
            for ind in 0..num_args as usize {
                let ent_ind = (ent * num_args) as usize + ind;
                let e = ent as usize;
                match (types[ind], &mut args[ind]) {
                    (b'i', ReadValueArray::Integers(int_arg)) => {
                        if any_dbls != 0 {
                            // `B3DNINT` is `(int)floor((a) + 0.5)`; the x86
                            // conversion gives INT_MIN out of range or on NaN
                            // where a Rust cast saturates.
                            let nint = (dbl_use[ent_ind] + 0.5).floor();
                            int_arg[e] = if nint >= -2147483648.0 && nint < 2147483648.0 {
                                nint as i32
                            } else {
                                i32::MIN
                            };
                            diff = int_arg[e] as f64 - dbl_use[ent_ind];
                            if diff > 0.001 || diff < -0.001 {
                                perr = -5;
                            }
                        } else if any_floats != 0 {
                            let nint = (float_use[ent_ind] as f64 + 0.5).floor();
                            int_arg[e] = if nint >= -2147483648.0 && nint < 2147483648.0 {
                                nint as i32
                            } else {
                                i32::MIN
                            };
                            // `int - float` is a float subtraction.
                            diff = (int_arg[e] as f32 - float_use[ent_ind]) as f64;
                            if diff > 0.001 || diff < -0.001 {
                                perr = -5;
                            }
                        } else {
                            int_arg[e] = int_use[ent_ind];
                        }
                    }
                    (b'f', ReadValueArray::Floats(float_arg)) => {
                        if any_dbls != 0 {
                            float_arg[e] = dbl_use[ent_ind] as f32;
                        } else {
                            float_arg[e] = float_use[ent_ind];
                        }
                    }
                    (b'd', ReadValueArray::Doubles(dbl_arg)) => {
                        dbl_arg[e] = dbl_use[ent_ind];
                    }
                    _ => {}
                }
            }
            ent += 1;
        }
    }

    /* Clean up */
    if perr == -5 {
        return -5;
    }
    if perr != 0 {
        return -4; /* Parse error OR not enough on line if separate lines */
    }

    if ierr == -1 {
        return -1; /* Read error */
    }

    if ierr < 0 && num_to_get_in > 0 && num_got < num_to_get {
        return -2; /* End of file looking for specific number */
    }

    if ierr < 0 && num_to_get_in == 0 && num_got == val_size {
        return -3; /* Array was full */
    }

    if num_to_get_in <= 0 {
        *num_to_get_p = num_got / num_args;
    }
    0
}

/// Original C `exitFromValueReadError` (`readlinevalues.c:221`).
///
/// Given a non-zero error code from `readLinesForValues` in `ierr` and a
/// description of the file in `descrip`, calls `exitError` with an appropriate
/// message including `descrip` after the words "file of".  Returns for 0.
pub fn exit_from_value_read_error(ierr: i32, descrip: &str) {
    let mut temp: Vec<u8> = Vec::new();
    if ierr == 0 {
        return;
    }
    if ierr == -1 {
        exit_error(&c_format_bytes(
            "Reading a line from file of %s",
            &[CArg::Str(descrip)],
        ));
    }
    if ierr == -2 {
        exit_error(&c_format_bytes(
            "End of file before all values gotten from file of %s",
            &[CArg::Str(descrip)],
        ));
    }
    if ierr == -3 {
        exit_error(&c_format_bytes(
            "Array full before all values gotten from file of %s",
            &[CArg::Str(descrip)],
        ));
    }
    if ierr == -4 {
        pip_get_error(&mut temp);
        exit_error(&c_format_bytes(
            "Parsing values on line from file of %s: %s",
            &[CArg::Str(descrip), CArg::Bytes(&temp)],
        ));
    }
    if ierr == -5 {
        exit_error(&c_format_bytes(
            "Non-integer value when expecting an integer in file of %s",
            &[CArg::Str(descrip)],
        ));
    }
    if ierr == -6 {
        exit_error(&c_format_bytes(
            "Allocating temporary array to get values from file of %s",
            &[CArg::Str(descrip)],
        ));
    }
    if ierr == -7 {
        exit_error(&c_format_bytes(
            "Programming error trying to get values from file of %s",
            &[CArg::Str(descrip)],
        ));
    }
    exit_error(&c_format_bytes(
        "Unknown error code (%d) reading values from file of %s",
        &[CArg::Int(ierr as i64), CArg::Str(descrip)],
    ));
}
