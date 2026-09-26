//! Translation of `IMOD/mrc/mrctaper.c`.
//!
//! Retranslated statement by statement (2026-09-26) after the body-fidelity
//! audit found an invented parser (`str::parse`, a `parse_range` helper and
//! four error messages the source does not have) and `eprintln!` in place of
//! `exitError`.

use std::io::Write;

use crate::imod::clip::clip::{ScanArg, sscanf};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_output_file_type, c_format_bytes, imod_copyright, imod_prog_name,
};
use crate::imod::libcfshr::islice::{Islice, MrcData, slice_init, slice_mode_if_real};
use crate::imod::libcfshr::parse_params::{exit_error, set_standard_exit_prefix};
use crate::imod::libcfshr::taperatfill::slice_taper_at_fill;
use crate::imod::libiimod::iimage::{IIFILE_TIFF, ii_fclose, ii_fopen};
use crate::imod::libiimod::mrcfiles::{
    MrcHeader, mrc_getdcsize, mrc_head_label, mrc_head_read, mrc_head_write,
    mrc_init_output_header, mrc_read_slice, mrc_write_slice,
};

const DEFAULT_TAPER: i32 = 16;

/// `mrctaper_help` (`mrctaper.c:23`).  The usage line goes to stderr and the
/// rest to stdout, as in the source.
pub fn mrctaper_help(name: &str) {
    let _ = ImodFile::Stderr.write_all(&c_format_bytes(
        "Usage: %s [-i] [-t #] [-z min,max] input_file [output_file]\n",
        &[CArg::Bytes(name.as_bytes())],
    ));
    let _ = ImodFile::Stdout.write_all(b"Options:\n");
    let _ = ImodFile::Stdout.write_all(b"\t-i\tTaper inside (default is outside).\n");
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "\t-t #\tTaper over the given # of pixels (default %d or 1%% of size).\n",
        &[CArg::Int(DEFAULT_TAPER as i64)],
    ));
    let _ = ImodFile::Stdout.write_all(b"\t-z min,max\tDo only sections between min and max.\n");
    let _ = ImodFile::Stdout
        .write_all(b"\tWith no output file, images are written back to input file.\n");
}

/// C `main` in `mrctaper.c:34`, returning its process status for the dispatcher.
pub fn mrctaper(argv: &[String]) -> i32 {
    let argc = argv.len();
    let mut inside = false;
    let mut ntaper = DEFAULT_TAPER;
    let mut taper_entered = false;
    let mut zmin = -1_i32;
    let mut zmax = -1_i32;
    let progname = imod_prog_name(argv.first().map_or("mrctaper", |a| a.as_str()));
    set_standard_exit_prefix(progname.as_bytes());

    // `mrctaper.c:52`: `argv[1][0] == '-' && argv[1][0] == 'h'` can never be
    // true, so `-h` falls through to the illegal-option branch below.
    #[allow(clippy::nonminimal_bool)]
    if argc < 2
        || (argv[1].as_bytes().first() == Some(&b'-') && argv[1].as_bytes().first() == Some(&b'h'))
    {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "%s version %s\n",
            &[CArg::Bytes(progname.as_bytes()), CArg::Bytes(b"5.2.17")],
        ));
        imod_copyright();
        mrctaper_help(&progname);
        return 3;
    }

    let mut i = 1;
    while i < argc {
        let arg = argv[i].as_bytes();
        if arg.first() == Some(&b'-') {
            match arg.get(1).copied().unwrap_or(0) {
                b'i' => inside = true,
                b't' => {
                    taper_entered = true;
                    if arg.get(2).copied().unwrap_or(0) != 0 {
                        sscanf(&argv[i], "-t%d", &mut [ScanArg::Int(&mut ntaper)]);
                    } else {
                        i += 1;
                        // `sscanf(argv[++i], ...)` past the last argument
                        // dereferences `argv[argc]`, a NULL: the source
                        // crashes.  Not reproducible; refuse with the usage.
                        let Some(value) = argv.get(i) else {
                            mrctaper_help(&progname);
                            return 3;
                        };
                        sscanf(value, "%d", &mut [ScanArg::Int(&mut ntaper)]);
                    }
                }
                b'z' => {
                    if arg.get(2).copied().unwrap_or(0) != 0 {
                        sscanf(
                            &argv[i],
                            "-z%d%*c%d",
                            &mut [ScanArg::Int(&mut zmin), ScanArg::Int(&mut zmax)],
                        );
                    } else {
                        i += 1;
                        let Some(value) = argv.get(i) else {
                            mrctaper_help(&progname);
                            return 3;
                        };
                        sscanf(
                            value,
                            "%d%*c%d",
                            &mut [ScanArg::Int(&mut zmin), ScanArg::Int(&mut zmax)],
                        );
                    }
                }
                _ => {
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "ERROR: %s - illegal option\n",
                        &[CArg::Bytes(progname.as_bytes())],
                    ));
                    mrctaper_help(&progname);
                    return 1;
                }
            }
        } else {
            break;
        }
        i += 1;
    }

    if (i as isize) < argc as isize - 2 || i == argc {
        mrctaper_help(&progname);
        return 3;
    }

    if !(1..=127).contains(&ntaper) {
        exit_error(b"Taper must be between 1 and 127.");
    }

    let fin = if i < argc - 1 {
        i += 1;
        ii_fopen(argv[i - 1].as_bytes(), "rb")
    } else {
        i += 1;
        ii_fopen(argv[i - 1].as_bytes(), "rb+")
    };
    let Some(mut fin) = fin else {
        exit_error(&c_format_bytes(
            "Opening %s.",
            &[CArg::Bytes(argv[i - 1].as_bytes())],
        ));
    };
    let mut hdata = MrcHeader::default();
    if mrc_head_read(&mut fin, &mut hdata) != 0 {
        exit_error(b"Can't Read Input File Header.");
    }

    if slice_mode_if_real(hdata.mode) < 0 {
        exit_error(b"Can operate only on byte, integer and real data.");
    }

    if !taper_entered {
        ntaper = (hdata.nx + hdata.ny) / 200;
        // B3DMIN(127, B3DMAX(DEFAULT_TAPER, ntaper))
        let m = if DEFAULT_TAPER > ntaper {
            DEFAULT_TAPER
        } else {
            ntaper
        };
        ntaper = if 127 < m { 127 } else { m };
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "Tapering over %d pixels\n",
            &[CArg::Int(ntaper as i64)],
        ));
    }

    if zmin == -1 && zmax == -1 {
        zmin = 0;
        zmax = hdata.nz - 1;
    } else {
        if zmin < 0 {
            zmin = 0;
        }
        if zmax >= hdata.nz {
            zmax = hdata.nz - 1;
        }
    }

    // `hptr` is `&hout` for a separate output and `&hdata` in place; `None`
    // here stands for the latter so that the one header is used for both the
    // reads and the writes, as the source does.
    let mut hout: Option<MrcHeader> = None;
    let mut fout;
    let secofs;
    if i < argc {
        let Some(opened) = ii_fopen(argv[i].as_bytes(), "wb") else {
            exit_error(&c_format_bytes(
                "Opening %s.",
                &[CArg::Bytes(argv[i].as_bytes())],
            ));
        };
        fout = opened;
        let mut header = hdata.clone();
        header.fp = Some(fout.clone());

        /* DNM: eliminate extra header info in the output, and mark it as not swapped  */
        mrc_init_output_header(&mut header);
        header.nz = zmax + 1 - zmin;
        header.mz = header.nz;
        header.zlen = header.nz as f32;
        hout = Some(header);
        secofs = zmin;
    } else {
        if b3d_output_file_type() == IIFILE_TIFF {
            exit_error(b"Cannot write to an existing TIFF file.");
        }
        fout = fin.clone();
        secofs = 0;
    }

    let mut dsize = 0;
    let mut csize = 0;
    mrc_getdcsize(hdata.mode, &mut dsize, &mut csize);

    let bsize = hdata.nx * hdata.ny;
    let Some(buf) = MrcData::try_zeroed(hdata.mode, (dsize * csize * bsize) as usize) else {
        exit_error(b"Couldn't get memory for slice.");
    };
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
    let _ = slice_init(&mut slice, hdata.nx, hdata.ny, hdata.mode, buf);

    let mut i = zmin;
    while i <= zmax {
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "\rDoing section #%4d",
            &[CArg::Int(i as i64)],
        ));
        let _ = ImodFile::Stdout.flush();
        if mrc_read_slice(slice.data.bytes_mut(), &mut fin, &mut hdata, i, b'Z') != 0 {
            exit_error(&c_format_bytes(
                "Reading section %d.",
                &[CArg::Int(i as i64)],
            ));
        }

        if slice_taper_at_fill(&mut slice, ntaper, inside) != 0 {
            exit_error(b"Can't get memory for taper operation.");
        }

        let hptr = match hout.as_mut() {
            Some(h) => h,
            None => &mut hdata,
        };
        if mrc_write_slice(slice.data.bytes(), &mut fout, hptr, i - secofs, b'Z') != 0 {
            exit_error(&c_format_bytes(
                "Writing section %d.",
                &[CArg::Int(i as i64)],
            ));
        }
        i += 1;
    }
    let _ = ImodFile::Stdout.write_all(b"\nDone!\n");

    let hptr = match hout.as_mut() {
        Some(h) => h,
        None => &mut hdata,
    };
    mrc_head_label(hptr, b"mrctaper: Image tapered down to fill value at edges");

    mrc_head_write(&mut fout, hptr);
    ii_fclose(&mut fout);

    0
}

#[cfg(test)]
mod tests {
    use super::mrctaper;
    use crate::imod::libcfshr::b3dutil::ImodFile;
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_BYTE, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write, mrc_read_slice,
        mrc_write_slice,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FILE_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn tapers_a_real_mrc_into_a_new_output_stack() {
        let number = FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("imod-rs-mrctaper-{}-{number}", std::process::id()));
        let input_path = root.with_extension("input.mrc");
        let output_path = root.with_extension("output.mrc");
        let mut input = ImodFile::open(&input_path, "wb").expect("create input");
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 14, 14, 2, MRC_MODE_BYTE), 0);
        let plane = (0..14)
            .flat_map(|y| {
                (0..14).map(move |x| {
                    if x == 0 || x == 13 || y == 0 || y == 13 {
                        0
                    } else {
                        100
                    }
                })
            })
            .collect::<Vec<u8>>();
        for section in 0..2 {
            assert_eq!(
                mrc_write_slice(&plane, &mut input, &mut header, section, b'Z'),
                0
            );
        }
        assert_eq!(mrc_head_write(&mut input, &mut header), 0);
        drop(input);
        assert_eq!(
            mrctaper(&[
                "mrctaper".into(),
                "-t2".into(),
                input_path.display().to_string(),
                output_path.display().to_string()
            ]),
            0
        );
        let mut output = ImodFile::open(&output_path, "rb").expect("open output");
        let mut output_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut output, &mut output_header), 0);
        assert_eq!(output_header.nz, 2);
        assert!(
            output_header.labels[..output_header.nlabl as usize]
                .iter()
                .any(|label| label.starts_with(b"mrctaper:"))
        );
        let mut tapered = vec![0_u8; 196];
        assert_eq!(
            mrc_read_slice(&mut tapered, &mut output, &mut output_header, 0, b'Z'),
            0
        );
        assert_ne!(tapered, plane);
        let _ = std::fs::remove_file(input_path);
        let _ = std::fs::remove_file(output_path);
    }
}
