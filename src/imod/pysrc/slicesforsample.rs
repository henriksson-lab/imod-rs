//! Translation of `IMOD/pysrc/slicesforsample`: assesses whether the number of
//! lines for a sample aligned stack is enough.
//!
//! A Python command script with no functions; its top level is
//! [`slicesforsample`].  Python semantics carried explicitly: `//` floors,
//! `round()` rounds half to even and `int()` truncates.

use super::imodpy::{
    FLOAT_VALUE, INT_VALUE, OptionValue, add_imod_bin_ignore_sighup, fmtstr, option_value,
    prnstr, py_int, py_int_floordiv, py_int_of_float, py_round, read_text_file,
};
use std::ffi::OsString;
use std::io::Write as _;

/// The script's top level (`slicesforsample:1-86`).  Returns the status of
/// its `sys.exit`.
pub fn slicesforsample(arguments: &[OsString]) -> i32 {
    let progname = "slicesforsample";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    if argv.len() != 6 {
        prnstr(
            &format!(
                "{prefix}Need 5 params: current lines, Y offset, ali Y size, raw Y size, and tilt com file name"
            ),
            "\n",
            false,
        );
        return done(1);
    }

    let (cur_lines, y_offset, full_y, ysize) = match (
        py_int(&argv[1]),
        py_int(&argv[2]),
        py_int(&argv[3]),
        py_int(&argv[4]),
    ) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => {
            prnstr(
                &format!("{prefix}Error converting one of the parameters to integer"),
                "\n",
                false,
            );
            return done(1);
        }
    };
    let tilt_com = &argv[5];

    let tilt_lines = match read_text_file(tilt_com, None, true, None) {
        Ok(lines) => lines,
        Err(_) => {
            prnstr("0", "\n", false);
            return done(0);
        }
    };

    let xtilt = match option_value(&tilt_lines, "XAXISTILT", FLOAT_VALUE, false, 1, None, None) {
        Some(OptionValue::Floats(values)) => Some(values[0]),
        _ => None,
    };
    let thickness = match option_value(&tilt_lines, "THICKNESS", INT_VALUE, false, 1, None, None)
    {
        Some(OptionValue::Integers(values)) => Some(values[0] as i64),
        _ => None,
    };
    let (Some(xtilt), Some(thickness)) = (xtilt, thickness) else {
        prnstr("0", "\n", false);
        return done(0);
    };

    // The slices needed have two components: first the number of slices required for the
    // reconstruction, based on sin * thickness, then the change in the center offset in the
    // aligned stack needed to reconstruct tilted slices at that offset, based on offset/cos
    // minus the aligned stack offset.  This is doubled to make the sample extend that far
    // out but still be centered on the original offset
    let radt = xtilt.to_radians().abs();
    let mut new_lines = 2 * py_int_floordiv(
        py_int_of_float(py_round(
            radt.sin() * thickness as f64 + 2. * y_offset as f64 * (1. / radt.cos() - 1.),
        )) + 4,
        2,
    );
    let max_lines = 2 * (py_int_floordiv(full_y, 2) - y_offset - 8);
    new_lines = new_lines.min(max_lines);
    if new_lines > cur_lines {
        let mut out_line = new_lines.to_string();

        // Copy the computation of blend Y limits from copytomocoms.  Yes the resulting
        // possibly out of range values do work with a montage being rotated by 90 degrees
        let offarr = [0, y_offset, -y_offset];
        let halfsampali = py_int_floordiv(new_lines, 2);
        for offind in offarr {
            let ymin = py_int_floordiv(ysize, 2) + offind - halfsampali;
            let ymax = py_int_floordiv(ysize, 2) + offind + halfsampali;
            out_line += &fmtstr(" {} {}", &[ymin.to_string(), ymax.to_string()]);
        }

        prnstr(&out_line, "\n", false);
    } else {
        prnstr("0", "\n", false);
    }

    done(0)
}
