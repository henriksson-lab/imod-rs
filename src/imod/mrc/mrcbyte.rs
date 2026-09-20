//! Translation of `IMOD/mrc/mrcbyte.c`.

use crate::imod::clip::clip::{ScanArg, sscanf};
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::fgetline;
use crate::imod::libcfshr::b3dutil::{
    CArg, b3d_set_store_error, c_format_bytes, imod_backup_file, imod_copyright, imod_prog_name,
};
use crate::imod::libcfshr::islice::{Islice, MrcData, slice_init};
use crate::imod::libcfshr::parse_params::{exit_error, set_standard_exit_prefix};
use crate::imod::libiimod::iimage::ii_fopen;
use crate::imod::libiimod::mrcfiles::get_loadinfo;
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_SHORT, MRC_MODE_USHORT, MRC_RAMP_EXP,
    MRC_RAMP_LIN, MRC_RAMP_LOG, MrcHeader, mrc_contrast_scaling, mrc_get_scale, mrc_head_label,
    mrc_head_read, mrc_head_write, mrc_init_li, mrc_init_output_header, mrc_set_scale,
    mrc_write_slice,
};
use crate::imod::libiimod::mrcsec::mrc_read_z_byte;
use crate::imod::libiimod::mrcslice::slice_mmm;
use std::io::Write as _;

pub fn mrcbyte_help(name: &str) {
    // `mrcbyte.c:21-34`: thirteen separate `printf` calls through libc stdout.
    // Two details are load-bearing.  The `-s` line really does read
    // "min,min" in the source -- that is a typo upstream, and "correcting" it
    // to "min,max" made the usage text differ from native.  And these must go
    // through the same stdout as every other message this program writes; a
    // `println!` here interleaves wrongly under a redirect, which put the
    // illegal-option error *after* the usage text instead of before it.
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "Usage: %s -[rRblecsxyz] <infile> <outfile>\n",
        &[CArg::Bytes(name.as_bytes())],
    ));
    let _ = ImodFile::Stdout.write_all(b"Options:\n");
    let _ = ImodFile::Stdout.write_all(b"\t-r\tRequest user input for size and contrast info.\n");
    let _ =
        ImodFile::Stdout.write_all(b"\t-c blk,wht\tContrast scaling with black and white levels\n");
    let _ = ImodFile::Stdout
        .write_all(b"\t-s min,min\tInitial intensity scaling (min to 0, max to 255)\n");
    let _ = ImodFile::Stdout.write_all(b"\t-R\tReverse contrast in output.\n");
    let _ = ImodFile::Stdout.write_all(b"\t-l\tUse logarithmic ramp.\n");
    let _ = ImodFile::Stdout.write_all(b"\t-e\tUse exponential ramp.\n");
    let _ = ImodFile::Stdout.write_all(b"\t-x min,max\tWrite subset of image in X\n");
    let _ = ImodFile::Stdout.write_all(b"\t-y min,max\tWrite subset of image in Y\n");
    let _ = ImodFile::Stdout.write_all(b"\t-z min,max\tWrite subset of image in Z\n");
    let _ = ImodFile::Stdout.write_all(b"\t-b\tWrite raw byte data, no header\n");
}

/// C `main` in `mrcbyte.c`.
pub fn mrcbyte(arguments: &[String]) -> i32 {
    // `mrcbyte.c:55-56`: `imodProgName(argv[0])` strips the directory, and
    // `setStandardExitPrefix` installs "\nERROR: <name> - " as the *stdout*
    // exit prefix.  Using `argv[0]` raw put the whole binary path in every
    // message.
    let progname_owned = arguments
        .first()
        .map_or_else(|| "mrcbyte".to_string(), |argv0| imod_prog_name(argv0));
    let program = progname_owned.as_str();
    set_standard_exit_prefix(program.as_bytes());
    if arguments.len() < 3 {
        // `mrcbyte.c:60-64`: the version line and the copyright come *before*
        // the usage text.  Note the source's own `printf("%s version %s\n")`
        // spells "version" lowercase here, unlike `imodVersion`'s "Version".
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "%s version %s\n",
            &[CArg::Bytes(program.as_bytes()), CArg::Bytes(b"5.2.17")],
        ));
        imod_copyright();
        mrcbyte_help(program);
        return 3;
    }
    /* Make library error output to stderr go to stdout */
    // `mrcbyte.c:65-67`.  Without this the library's own `ERROR: iiOpen - ...`
    // diagnostic lands on stderr while every message this program emits itself
    // goes to stdout, so a redirect captures only half the error.
    b3d_set_store_error(-1);

    let mut li = LoadInfo::default();
    mrc_init_li(Some(&mut li), None);
    let mut data_only = false;
    let mut reverse = false;
    let mut ramp = MRC_RAMP_LIN;
    let mut resize = false;
    let mut index = 1;
    while index < arguments.len() && arguments[index].starts_with('-') && arguments[index] != "-" {
        let option = &arguments[index];
        let key = option.as_bytes().get(1).copied();
        let value = |index: &mut usize| -> Option<&str> {
            if option.len() > 2 {
                Some(&option[2..])
            } else {
                *index += 1;
                arguments.get(*index).map(String::as_str)
            }
        };
        match key {
            Some(b'b') if option.len() == 2 => data_only = true,
            Some(b'r') if option.len() == 2 => resize = true,
            Some(b'R') if option.len() == 2 => reverse = true,
            Some(b'l') if option.len() == 2 => ramp = MRC_RAMP_LOG,
            Some(b'e') if option.len() == 2 => ramp = MRC_RAMP_EXP,
            Some(b'c') => {
                let Some(text) = value(&mut index) else {
                    return 1;
                };
                let Some((black, white)) = text
                    .split_once(',')
                    .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
                else {
                    exit_error(b"invalid contrast range");
                };
                li.black = black;
                li.white = white;
            }
            Some(b's') => {
                let Some(text) = value(&mut index) else {
                    return 1;
                };
                let Some((minimum, maximum)) = text
                    .split_once(',')
                    .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
                else {
                    exit_error(b"invalid scaling range");
                };
                li.smin = minimum;
                li.smax = maximum;
            }
            Some(b'x') | Some(b'y') | Some(b'z') => {
                let Some(text) = value(&mut index) else {
                    return 1;
                };
                let Some((minimum, maximum)) = text
                    .split_once(',')
                    .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
                else {
                    exit_error(b"invalid subset range");
                };
                match key {
                    Some(b'x') => {
                        li.xmin = minimum;
                        li.xmax = maximum;
                    }
                    Some(b'y') => {
                        li.ymin = minimum;
                        li.ymax = maximum;
                    }
                    _ => {
                        li.zmin = minimum;
                        li.zmax = maximum;
                    }
                }
            }
            _ => {
                // `mrcbyte.c:132` is a plain `printf` -- stdout, one space,
                // no leading blank line -- not `exitError`.
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "ERROR: %s - Illegal option %s\n",
                    &[
                        CArg::Bytes(program.as_bytes()),
                        CArg::Bytes(option.as_bytes()),
                    ],
                ));
                mrcbyte_help(program);
                std::process::exit(1);
            }
        }
        index += 1;
    }
    if arguments.len() - index != 2 {
        mrcbyte_help(program);
        return 3;
    }

    let Some(mut input) = ii_fopen(arguments[index].as_bytes(), "rb") else {
        exit_error(&c_format_bytes(
            "Opening %s.",
            &[CArg::Bytes(arguments[index].as_bytes())],
        ));
    };
    // `mrcbyte.c:147-150`: between the two opens the source backs up an
    // existing output file unless `IMOD_NO_IMAGE_BACKUP` is set, and warns on
    // stdout if that fails.
    if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none()
        && imod_backup_file(&arguments[index + 1]) != 0
    {
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "WARNING: %s - Error making backup file from existing %s\n",
            &[
                CArg::Bytes(program.as_bytes()),
                CArg::Bytes(arguments[index + 1].as_bytes()),
            ],
        ));
    }
    let Some(mut output) = ii_fopen(arguments[index + 1].as_bytes(), "wb") else {
        exit_error(&c_format_bytes(
            "Opening %s.",
            &[CArg::Bytes(arguments[index + 1].as_bytes())],
        ));
    };
    let mut header = MrcHeader::default();
    if mrc_head_read(&mut input, &mut header) != 0 {
        exit_error(b"Reading input file header.");
    }
    if resize {
        // `mrcbyte.c:162-174`.  This is an interactive prompt on stdin, not an
        // error: it prints the defaults, lets `get_loadinfo` collect a subset,
        // then reads the black/white levels from one line of stdin.
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "Defaults for %s size = ( %d x %d x %d):\n",
            &[
                CArg::Bytes(arguments[index].as_bytes()),
                CArg::Int(header.nx as i64),
                CArg::Int(header.ny as i64),
                CArg::Int(header.nz as i64),
            ],
        ));
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "x = ( 0, %d), y = ( 0, %d), z = ( 0, %d), c = ( 0, 255)\n",
            &[
                CArg::Int(header.nx as i64 - 1),
                CArg::Int(header.ny as i64 - 1),
                CArg::Int(header.nz as i64 - 1),
            ],
        ));
        get_loadinfo(&header, &mut li);
        let _ = ImodFile::Stdout.write_all(b" Enter (black level, white level) >");
        let mut line = [0u8; 128];
        fgetline(&mut ImodFile::Stdin, &mut line, 127);
        let text = String::from_utf8_lossy(&line[..line.iter().position(|c| *c == 0).unwrap_or(0)])
            .into_owned();
        sscanf(
            &text,
            "%d%*c%d\n",
            &mut [ScanArg::Int(&mut li.black), ScanArg::Int(&mut li.white)],
        );
        let _ = ImodFile::Stdout.write_all(b"\n");
        // `mrcbyte.c:172` is a bitwise `&` of two `!` results, not `&&`.
        if ((li.black == 0) as i32) & ((li.white == 0) as i32) != 0 {
            li.white = 255;
        }
    }
    if ramp != MRC_RAMP_LIN
        && !matches!(
            header.mode,
            MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_FLOAT
        )
    {
        exit_error(b"Nonlinear scaling can be used only with integer and float data");
    }
    li.ramp = ramp;
    mrc_init_li(Some(&mut li), Some(&header));
    let (slope, offset) =
        mrc_contrast_scaling(&header, li.smin, li.smax, li.black, li.white, li.ramp);
    li.slope = slope;
    li.offset = offset;
    let mut output_header = header.clone();
    output_header.amin = 0.;
    output_header.amax = 255.;
    output_header.mode = MRC_MODE_BYTE;
    mrc_init_output_header(&mut output_header);
    output_header.nx = li.xmax + 1 - li.xmin;
    output_header.ny = li.ymax + 1 - li.ymin;
    output_header.nz = li.zmax + 1 - li.zmin;
    let (xscale, yscale, zscale) = mrc_get_scale(&header);
    mrc_set_scale(
        &mut output_header,
        xscale as f64,
        yscale as f64,
        zscale as f64,
    );
    mrc_head_label(
        &mut output_header,
        b"mrcbyte: Converted and scaled to byte mode.",
    );
    let area = output_header.nx as usize * output_header.ny as usize;
    let mut sum = 0.;
    if !data_only && mrc_head_write(&mut output, &mut output_header) != 0 {
        exit_error(b"Writing header to output file");
    }
    // `mrcbyte.c:221`: one buffer for the whole run; `sliceInit` aliases it
    // and `mrc_write_slice` writes it, so it is lent to the slice and taken
    // back after each section.
    let mut buf = MrcData::B(vec![0; area]);
    for section in 0..output_header.nz {
        let input_section = section + li.zmin;
        // `mrcbyte.c:229-230`: a carriage return, not a newline, and an
        // explicit flush -- so the 30 reports overwrite one line on a terminal
        // and land as one long line under a pipe.
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "Converting Image # %3d\r",
            &[CArg::Int(input_section as i64)],
        ));
        let _ = ImodFile::Stdout.flush();
        if mrc_read_z_byte(&mut header, &mut li, buf.b_mut(), input_section) != 0 {
            exit_error(&c_format_bytes(
                "Reading section %d from file",
                &[CArg::Int(input_section as i64)],
            ));
        }
        if reverse {
            for byte in buf.b_mut() {
                *byte = 255 - *byte;
            }
        }
        // `mrcbyte.c:238-247`: `sliceInit` + `sliceMMM`, not an inline scan.
        // `sliceMMM` accumulates one row at a time into a double and then
        // stores the result in `Islice::mean`, which is a **float** -- so the
        // per-section mean is rounded to f32 before it reaches `meansum`.  A
        // flat f64 sum over the whole section instead put `amean` one float
        // away from native (24.324752807617188 vs 24.32476806640625).
        let mut slice = Islice {
            data: MrcData::default(),
            xsize: 0,
            ysize: 0,
            mode: 0,
            csize: 0,
            dsize: 0,
            min: 0.,
            max: 0.,
            mean: 0.,
            index: 0,
            cval: [0.; 4],
        };
        slice_init(
            &mut slice,
            output_header.nx,
            output_header.ny,
            MRC_MODE_BYTE,
            std::mem::take(&mut buf),
        );
        slice_mmm(&mut slice);
        if section == 0 {
            output_header.amin = slice.min;
            output_header.amax = slice.max;
        } else {
            // `B3DMIN`/`B3DMAX` are `a < b ? a : b` -- second operand on NaN.
            output_header.amin = if output_header.amin < slice.min {
                output_header.amin
            } else {
                slice.min
            };
            output_header.amax = if output_header.amax > slice.max {
                output_header.amax
            } else {
                slice.max
            };
        }
        sum += slice.mean as f64;
        if mrc_write_slice(
            slice.data.bytes(),
            &mut output,
            &mut output_header,
            section,
            b'Z',
        ) != 0
        {
            exit_error(&c_format_bytes(
                "Writing section %d to file",
                &[CArg::Int(section as i64)],
            ));
        }
        buf = std::mem::take(&mut slice.data);
    }
    output_header.amean = (sum / output_header.nz as f64) as f32;
    if !data_only && mrc_head_write(&mut output, &mut output_header) != 0 {
        exit_error(b"Writing header to output file");
    };
    // `mrcbyte.c:257` is `printf("\nDone\n")` -- the newline is *leading*, and
    // it goes through the same stdout as the progress reports above.
    let _ = ImodFile::Stdout.write_all(b"\nDone\n");
    let _ = ImodFile::Stdout.flush();
    0
}

#[cfg(test)]
mod tests {
    use super::mrcbyte;
    use crate::imod::libcfshr::b3dutil::ImodFile;
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write, mrc_write_slice,
    };
    #[test]
    fn converts_a_real_float_mrc_to_byte() {
        let root = std::env::temp_dir().join(format!("mrcbyte-{}", std::process::id()));
        let input = root.with_extension("in.mrc");
        let output = root.with_extension("out.mrc");
        let mut file = ImodFile::open(&input, "wb").unwrap();
        let mut header = MrcHeader::default();
        mrc_head_new(&mut header, 2, 1, 1, MRC_MODE_FLOAT);
        header.amin = 0.;
        header.amax = 10.;
        mrc_head_write(&mut file, &mut header);
        let data = [0_f32.to_ne_bytes(), 10_f32.to_ne_bytes()].concat();
        mrc_write_slice(&data, &mut file, &mut header, 0, b'Z');
        drop(file);
        assert_eq!(
            mrcbyte(&[
                "mrcbyte".into(),
                input.display().to_string(),
                output.display().to_string()
            ]),
            0
        );
        let mut file = ImodFile::open(&output, "rb").unwrap();
        let mut result = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut result), 0);
        assert_eq!(result.mode, 0);
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }
}
