//! Translation of `IMOD/mrc/mrcinfo.c` -- output header info to standard out.
//!
//! The source is one `main`.  Its report goes through `printf` with `%g` for
//! every floating value, so each line is formatted by the C library's rules
//! through [`c_format_bytes`]; its banner, usage and error text go to
//! `stderr` with `fprintf`.  `argv[0]` is printed as given (the source does
//! not pass it through `imodProgName`).
//!
//! Upstream IMOD does not build this program (`IMOD/mrc/Makefile` omits it),
//! so its native reference is compiled from the vendored `mrcinfo.c` against
//! the reference `libiimod` (see `fixtures/mrcinfo/make-goldens.sh`).

use crate::imod::libcfshr::b3dutil::OsStrExt as _;
use std::ffi::OsString;
use std::io::Write as _;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes, exit};
use crate::imod::libiimod::mrcfiles::{
    MRC_LABEL_SIZE, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_NLABELS, MrcHeader, mrc_head_read,
};

/// C `main` in `mrcinfo.c` (`mrcinfo.c:32`).
pub fn mrcinfo(argv: &[OsString]) -> ! {
    let argc = argv.len();
    let argv0 = argv.first().map_or(&b""[..], |a| a.as_bytes());
    let mut hdata = MrcHeader::default();
    let mut vd1: f32;
    let mut vd2: f32;
    let mut nd1: f32;
    let mut nd2: f32;

    let _ = ImodFile::Stderr.write_all(&c_format_bytes(
        "%s version 1.0 Copyright (C)1994 Boulder Laboratory for\n",
        &[CArg::Bytes(argv0)],
    ));
    let _ = ImodFile::Stderr.write_all(b"3-Dimensional Fine Structure, University of Colorado.\n");

    if argc < 2 {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "Usage: %s <image file>\n",
            &[CArg::Bytes(argv0)],
        ));
        exit(3);
    }

    let Some(mut fin) = ImodFile::open(&argv[1], "rb") else {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "Error opening %s.\n",
            &[CArg::Bytes(argv[1].as_bytes())],
        ));
        exit(3);
    };

    let error = mrc_head_read(&mut fin, &mut hdata);
    if error < 0 {
        let _ = ImodFile::Stderr.write_all(b"Can't Read Input File Header.\n");
        exit(3);
    }
    if error > 0 {
        let _ = ImodFile::Stderr.write_all(b"Warning: File is not in MRC format.\n\n");
    }
    let data = &hdata;

    // Output info
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(&c_format_bytes(
        "MRCinfo: Info on image file %s\n",
        &[CArg::Bytes(argv[1].as_bytes())],
    ));
    match data.mode {
        MRC_MODE_BYTE => {
            let _ = out.write_all(b"mode = Byte\n");
        }
        MRC_MODE_SHORT => {
            let _ = out.write_all(b"mode = Short\n");
        }
        MRC_MODE_FLOAT => {
            let _ = out.write_all(b"mode = Float\n");
        }
        MRC_MODE_COMPLEX_SHORT => {
            let _ = out.write_all(b"mode = Complex Short\n");
        }
        MRC_MODE_COMPLEX_FLOAT => {
            let _ = out.write_all(b"mode = Complex Float\n");
        }
        MRC_MODE_RGB => {
            let _ = out.write_all(b"mode = rgb byte\n");
        }
        _ => {
            let _ = out.write_all(b"mode is unknown.\n");
        }
    }

    let _ = out.write_all(&c_format_bytes(
        "Image size    =  ( %d, %d, %d)\n",
        &[
            CArg::Int(data.nx as i64),
            CArg::Int(data.ny as i64),
            CArg::Int(data.nz as i64),
        ],
    ));

    let _ = out.write_all(&c_format_bytes(
        "minimum value = %g\n",
        &[CArg::Dbl(data.amin as f64)],
    ));
    let _ = out.write_all(&c_format_bytes(
        "maximum value = %g\n",
        &[CArg::Dbl(data.amax as f64)],
    ));
    let _ = out.write_all(&c_format_bytes(
        "mean value    = %g\n",
        &[CArg::Dbl(data.amean as f64)],
    ));

    let _ = out.write_all(&c_format_bytes(
        "Start reading image at ( %d, %d, %d).\n",
        &[
            CArg::Int(data.nxstart as i64),
            CArg::Int(data.nystart as i64),
            CArg::Int(data.nzstart as i64),
        ],
    ));

    let _ = out.write_all(&c_format_bytes(
        "Read length   =  ( %d, %d, %d).\n",
        &[
            CArg::Int(data.mx as i64),
            CArg::Int(data.my as i64),
            CArg::Int(data.mz as i64),
        ],
    ));

    let _ = out.write_all(&c_format_bytes(
        "Size of voxel is ( %g x %g x %g ) um.\n",
        &[
            CArg::Dbl(data.xlen as f64),
            CArg::Dbl(data.ylen as f64),
            CArg::Dbl(data.zlen as f64),
        ],
    ));

    let _ = out.write_all(&c_format_bytes(
        "Rotation      =  ( %g, %g, %g).\n",
        &[
            CArg::Dbl(data.alpha as f64),
            CArg::Dbl(data.beta as f64),
            CArg::Dbl(data.gamma as f64),
        ],
    ));

    let _ = out.write_all(&c_format_bytes(
        "Col, rows, sections  = axis (%d , %d, %d)\n",
        &[
            CArg::Int(data.mapc as i64),
            CArg::Int(data.mapr as i64),
            CArg::Int(data.maps as i64),
        ],
    ));

    let _ = out.write_all(&c_format_bytes(
        "Angles (%g, %g, %g) --> (%g, %g, %g)\n",
        &[
            CArg::Dbl(data.tiltangles[0] as f64),
            CArg::Dbl(data.tiltangles[1] as f64),
            CArg::Dbl(data.tiltangles[2] as f64),
            CArg::Dbl(data.tiltangles[3] as f64),
            CArg::Dbl(data.tiltangles[4] as f64),
            CArg::Dbl(data.tiltangles[5] as f64),
        ],
    ));

    if data.ispg != 0 {
        let _ = out.write_all(&c_format_bytes(
            "ispg =\t\t%d\n",
            &[CArg::Int(data.ispg as i64)],
        ));
    }
    if data.idtype != 0 {
        let _ = out.write_all(&c_format_bytes(
            "idtype =\t%d\n",
            &[CArg::Int(data.idtype as i64)],
        ));

        nd1 = data.nd1 as f32;
        nd2 = data.nd2 as f32;
        vd1 = data.vd1 as f32;
        vd2 = data.vd2 as f32;

        let _ = out.write_all(&c_format_bytes("nd1 =\t\t%g\n", &[CArg::Dbl(nd1 as f64)]));
        let _ = out.write_all(&c_format_bytes("nd2 =\t\t%g\n", &[CArg::Dbl(nd2 as f64)]));
        // `vd1 / 100.0`: a float over a double literal, so the float
        // promotes and the division is done in double.
        let _ = out.write_all(&c_format_bytes(
            "vd1 =\t\t%g\n",
            &[CArg::Dbl(vd1 as f64 / 100.0)],
        ));
        let _ = out.write_all(&c_format_bytes(
            "vd2 =\t\t%g\n",
            &[CArg::Dbl(vd2 as f64 / 100.0)],
        ));
    }

    let _ = out.write_all(&c_format_bytes(
        "orgin = ( %g, %g, %g)\n",
        &[
            CArg::Dbl(data.xorg as f64),
            CArg::Dbl(data.yorg as f64),
            CArg::Dbl(data.zorg as f64),
        ],
    ));

    if data.nlabl > MRC_NLABELS as i32 {
        let _ = out.write_all(b"There are to many labels.\n\n");
        exit(0);
    }

    let _ = out.write_all(&c_format_bytes(
        "Thare are %d labels.\n\n",
        &[CArg::Int(data.nlabl as i64)],
    ));

    let mut i = 0;
    while i < data.nlabl {
        // `%s` of `char labels[i][MRC_LABEL_SIZE + 1]`, which
        // `mrc_head_read` terminates at `MRC_LABEL_SIZE` (`mrcfiles.c:155`):
        // the text up to the first NUL within the 80 bytes.
        let label = &data.labels[i as usize];
        let end = label
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(MRC_LABEL_SIZE);
        let _ = out.write_all(&c_format_bytes("%s", &[CArg::Bytes(&label[..end])]));
        i += 1;
    }

    exit(0);
}
