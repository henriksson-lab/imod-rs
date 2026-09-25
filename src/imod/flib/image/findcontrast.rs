//! Translation of `IMOD/flib/image/findcontrast.f90`.
//!
//! FINDCONTRAST finds the black and white contrast settings that would
//! truncate a specified small number of pixels when converting a volume to
//! bytes.  The Fortran main program maps to [`findcontrast`]; the source has
//! no other program units.

use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdpas};
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::parse_params::pip_get_two_integers;
use crate::imod::libiimod::unit_fileio::iiu_set_position;
use std::io::Write as _;

/// `parameter (IDIM = 10000 * 500)` (`findcontrast.f90:18`).
const IDIM: i32 = 10000 * 500;
/// `parameter (LIMDEN = 1000000)` (`findcontrast.f90:19`).
const LIMDEN: i32 = 1000000;
/// `parameter (numOptions = 8)` (`findcontrast.f90:39`).
const FINDCONTRAST_NUM_OPTIONS: i32 = 8;
/// Fallback PIP table, the `options(1)` string (`findcontrast.f90:41-44`).
const FINDCONTRAST_OPTIONS: &str = "input:InputFile:FN:@slices:SlicesMinAndMax:IP:@xminmax:XMinAndMax:IP:@\
yminmax:YMinAndMax:IP:@flipyz:FlipYandZ:B:@oldflip:OldFlipping:B:@\
truncate:TruncateBlackAndWhite:IP:@help:usage:B:";

/// Original program `findcontrast` (`findcontrast.f90:15`).
///
/// `array(IDIM)` and `ihist(-LIMDEN:LIMDEN)` are static-size Fortran arrays;
/// they are heap vectors here, `ihist` indexed at `ival + LIMDEN`.
pub fn findcontrast() {
    unsafe {
        // `Gw.d` editing, as `header.rs` carries it (FORMAT 101's `g13.5`).
        let g_edit = |value: f32, w: usize, d: i32| -> String {
            if value.is_nan() {
                return format!("{:>w$}", "NaN");
            }
            if value.is_infinite() {
                let text = match (value < 0.0, w) {
                    (false, 8..) => "Infinity",
                    (false, _) => "Inf",
                    (true, 9..) => "-Infinity",
                    (true, _) => "-Inf",
                };
                return format!("{text:>w$}");
            }
            let magnitude = value.abs();
            let mut digits = String::new();
            let mut exponent = 1_i32;
            if magnitude != 0.0 {
                let scientific = format!("{:.*e}", (d - 1) as usize, magnitude);
                let (mantissa, power) = scientific.split_once('e').unwrap();
                digits = mantissa.replace('.', "");
                exponent = power.parse::<i32>().unwrap() + 1;
            }
            if (0..=d).contains(&exponent) {
                let mut text = format!("{:.*}", (d - exponent) as usize, value);
                if exponent == d {
                    text.push('.');
                }
                format!("{:>1$}    ", text, w - 4)
            } else {
                format!(
                    "{:>1$}",
                    format!(
                        "{}0.{}E{}{:02}",
                        if value < 0.0 { "-" } else { "" },
                        digits,
                        if exponent < 0 { '-' } else { '+' },
                        exponent.abs()
                    ),
                    w
                )
            }
        };
        // `Iw`: a value too wide for the field is written as `w` asterisks.
        let i_edit = |value: i32, w: usize| -> String {
            let text = format!("{value:>w$}");
            if text.len() > w { "*".repeat(w) } else { text }
        };
        // A list-directed `read(5,*)` of integers into `values`: values are
        // separated by blanks or commas, records are read until every item
        // has one, and a `/` ends the read leaving the rest unchanged.  An
        // end of file is the gfortran runtime error, exit status 2.
        let read_list = |values: &mut [&mut i32]| {
            let mut index = 0;
            while index < values.len() {
                let mut line = String::new();
                if matches!(std::io::stdin().read_line(&mut line), Ok(0) | Err(_)) {
                    eprintln!("Fortran runtime error: End of file");
                    exit(2);
                }
                for token in line
                    .trim_end_matches(['\r', '\n'])
                    .split(|c: char| c == ',' || c == ' ' || c == '\t')
                    .filter(|t| !t.is_empty())
                {
                    if token.starts_with('/') {
                        return;
                    }
                    if index >= values.len() {
                        break;
                    }
                    match token.parse::<i32>() {
                        Ok(value) => *values[index] = value,
                        Err(_) => {
                            eprintln!(
                                "Fortran runtime error: Bad integer for item {} in list input",
                                index + 1
                            );
                            exit(2);
                        }
                    }
                    index += 1;
                }
            }
        };

        let mut nxyz = [0_i32; 3];
        let mut mxyz = [0_i32; 3];
        let mut array = vec![0.0_f32; IDIM as usize];
        let mut ihist = vec![0_i32; (2 * LIMDEN + 1) as usize];
        let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
        let mut mode = 0_i32;
        let mut filename = String::new();
        let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
        //
        // Pip startup: set error, parse options, check help, set flag if used
        //
        pip_read_or_parse_options(
            &[FINDCONTRAST_OPTIONS],
            FINDCONTRAST_NUM_OPTIONS,
            "findcontrast",
            "ERROR: FINDCONTRAST - ",
            true,
            1,
            1,
            0,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
        );
        let pipinput = num_opt_arg + num_non_opt_arg > 0;

        if pip_get_in_out_file("InputFile", 1, "Name of image file", &mut filename) != 0 {
            exit_error("No input file specified");
        }
        //
        // Open image file
        //
        imopen(1, &filename, "RO");
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &raw mut mode,
            &raw mut dmin,
            &raw mut dmax,
            &raw mut dmean,
        );
        let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);
        //
        let mut hist_scale = 1.0_f32;
        //
        // For non-integer mode, set up a scaling that should fill 1/5 of the histogram
        // at most, but allow the rest of the range in case the min and max are in error
        if mode != 1 && mode != 6 && mode != 0 {
            // `max(abs(dmin), abs(dmax), 1.e-10)`: gfortran's MAX keeps the
            // first argument unless a later one is greater.
            let mut largest = dmin.abs();
            if dmax.abs() > largest {
                largest = dmax.abs();
            }
            if 1.0e-10 > largest {
                largest = 1.0e-10;
            }
            hist_scale = (LIMDEN / 5) as f32 / largest;
        }

        if nx > IDIM {
            exit_error("Images too large in X for arrays");
        }

        let mut flipped = !pipinput;
        let mut old_flip = false;
        if pipinput {
            let _ = pip_get_logical("FlipYandZ", &mut flipped);
            let _ = pip_get_logical("OldFlipping", &mut old_flip);
        }
        //
        // Set up default limits
        //
        let mut ix_low = nx / 10;
        let mut ix_high = nx - 1 - ix_low;
        //
        let (iz_lim, iylim) = if flipped { (ny, nz) } else { (nz, ny) };
        let mut iz_low = 1_i32;
        let mut iz_high = iz_lim;
        let mut iy_low = iylim / 10;
        let mut iy_high = iylim - 1 - iy_low;
        //
        if pipinput {
            let _ = pip_get_two_integers(b"SlicesMinAndMax", &mut iz_low, &mut iz_high);
            let _ = pip_get_two_integers(b"XMinAndMax", &mut ix_low, &mut ix_high);
            let _ = pip_get_two_integers(b"YMinAndMax", &mut iy_low, &mut iy_high);
        } else {
            // `write(*,'(1x,a,/,a,$)')`
            print!(
                " First and last slice (Imod section # in flipped volume)\n  to include in analysis: "
            );
            let _ = std::io::stdout().flush();
            read_list(&mut [&mut iz_low, &mut iz_high]);
            // `write(*,'(1x,a,/,a,4i7,a,$)')`
            print!(
                " Lower & upper X, lower & upper Y (in flipped volume) to include in analysis\n  (/ for {:>7}{:>7}{:>7}{:>7}): ",
                ix_low, ix_high, iy_low, iy_high
            );
            let _ = std::io::stdout().flush();
            read_list(&mut [&mut ix_low, &mut ix_high, &mut iy_low, &mut iy_high]);
        }
        if iz_low <= 0 || iz_high > iz_lim || iz_low > iz_high {
            exit_error("Slice numbers outside range of image file");
        }
        iz_low -= 1;
        iz_high -= 1;
        //
        if ix_low < 0
            || ix_high >= nx
            || ix_low >= ix_high
            || iy_low < 0
            || iy_high >= iylim
            || iy_low > iy_high
        {
            exit_error("X or Y values outside range of volume");
        }
        //
        let area_fac = 1.0_f32.max(nx.wrapping_mul(iylim) as f32 * 1.0e-6);
        let mut num_trunc_lo = (area_fac * (iz_high + 1 - iz_low) as f32) as i32;
        let mut num_trunc_hi = num_trunc_lo;
        if pipinput {
            let _ = pip_get_two_integers(
                b"TruncateBlackAndWhite",
                &mut num_trunc_lo,
                &mut num_trunc_hi,
            );
        } else {
            // `write(*,'(1x,a,/,a,2i8,a,$)')`
            print!(
                " Maximum numbers of pixels to truncate at black and white in analyzed volume\n  (/ for {:>8}{:>8}): ",
                num_trunc_lo, num_trunc_hi
            );
            let _ = std::io::stdout().flush();
            read_list(&mut [&mut num_trunc_lo, &mut num_trunc_hi]);
        }
        //
        // Flip coordinates
        //
        if flipped {
            let ierr = iy_low;
            let iy_end = iy_high;
            if old_flip {
                iy_low = iz_low;
                iy_high = iz_high;
            } else {
                iy_low = ny - 1 - iz_high;
                iy_high = ny - 1 - iz_low;
            }
            iz_low = ierr;
            iz_high = iy_end;
        }
        // `write(*,'(3(a,2i6))')`
        println!(
            "Analyzing X:{}{}  Y:{}{}  Z:{}{}",
            i_edit(ix_low, 6),
            i_edit(ix_high, 6),
            i_edit(iy_low, 6),
            i_edit(iy_high, 6),
            i_edit(iz_low, 6),
            i_edit(iz_high, 6)
        );
        //
        for count in ihist.iter_mut() {
            *count = 0;
        }
        let nx_tot = ix_high + 1 - ix_low;
        let ny_tot = iy_high + 1 - iy_low;
        let max_lines = IDIM / nx_tot;
        let num_chunks = (ny_tot + max_lines - 1) / max_lines;

        let mut ival_min = LIMDEN;
        let mut ival_max = -LIMDEN;
        for iz in iz_low..=iz_high {
            let mut iy_chunk = iy_low;
            for _ichunk in 1..=num_chunks {
                let iy_end = iy_high.min(iy_chunk + max_lines - 1);
                let num_ylines = iy_end + 1 - iy_chunk;
                // `imposn(1, iz, 0)` is `iiuSetPosition` (`unit_fileio.c:492`).
                iiu_set_position(1, iz, 0);
                // `irdpas` is called with no alternate return, so a read
                // failure is ignored and the array keeps what it held.
                let _ = irdpas(
                    1, &mut array, nx_tot, num_ylines, ix_low, ix_high, iy_chunk, iy_end,
                );
                for i in 0..(nx_tot * num_ylines) as usize {
                    // `nint` of a `real*4` to default integer: gcc lowers it to
                    // `lroundf` and keeps the low 32 bits of the `long`.
                    let rounded = (hist_scale * array[i]).round();
                    let mut ival = if rounded >= -9.223372e18 && rounded < 9.223372e18 {
                        rounded as i64 as i32
                    } else {
                        (i64::MIN) as i32
                    };
                    ival = (-LIMDEN).max(LIMDEN.min(ival));
                    ihist[(ival + LIMDEN) as usize] += 1;
                    ival_min = ival_min.min(ival);
                    ival_max = ival_max.max(ival);
                }
                iy_chunk = iy_end + 1;
            }
        }
        //
        let mut num_trunc = 0_i32;
        let mut ind_low = ival_min;
        while num_trunc <= num_trunc_lo && ind_low < ival_max {
            num_trunc += ihist[(ind_low + LIMDEN) as usize];
            ind_low += 1;
        }
        //
        num_trunc = 0;
        let mut ind_hi = ival_max;
        while num_trunc <= num_trunc_hi && ind_hi > ival_min {
            num_trunc += ihist[(ind_hi + LIMDEN) as usize];
            ind_hi -= 1;
        }
        let real_low = ind_low as f32 / hist_scale;
        let real_high = ind_hi as f32 / hist_scale;
        // Real-to-integer assignment is `cvttss2si`: an out-of-range or NaN
        // value becomes -2147483648 rather than saturating.
        let to_int = |value: f32| -> i32 {
            if value >= -2147483648.0 && value < 2147483648.0 {
                value as i32
            } else {
                i32::MIN
            }
        };
        let icon_low = to_int(255.0 * (real_low - dmin) / (dmax - dmin));
        let icon_high = to_int(255.0 * (real_high - dmin) / (dmax - dmin) + 0.99);
        if icon_low < 0 || icon_high > 255 {
            exit_error(
                "The file minimum or maximum is too far off to allow contrast scaling; use Alterheader with mmm option to fix min/max",
            );
        }
        // FORMAT 101
        println!(
            "Min and max densities in the analyzed volume are{} and{}",
            g_edit(ival_min as f32 / hist_scale, 13, 5),
            g_edit(ival_max as f32 / hist_scale, 13, 5)
        );
        println!(
            "Min and max densities with truncation are{} and{}",
            g_edit(real_low, 13, 5),
            g_edit(real_high, 13, 5)
        );
        println!(
            "Implied black and white contrast levels are{} and{}",
            i_edit(icon_low, 4),
            i_edit(icon_high, 4)
        );
        exit(0);
    }
}
