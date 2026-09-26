//! Translation of `IMOD/mrc/mrclog.c`.
//!
//! Retranslated statement by statement (2026-09-26).  The earlier body added a
//! seek to the data before the minimum scan that the source does not make,
//! returned 3 on every short read or write where the source ignores them,
//! used `f32::min`/`max`/`max(1.0)` where the source compares with `<` (NaN
//! behaviour), took the logarithm in `float` where the source calls the
//! double `log`, and invented two dimension errors.
//!
//! `mrclog` is not in `IMOD/mrc/Makefile`'s `PROGS`, and it cannot link:
//! `flog` is defined nowhere in the vendored tree.  So there is no native
//! binary, and three details of the source have no defined behaviour to
//! reproduce: `flog` (taken here as the natural logarithm in `float`, the
//! earlier translation's reading), the uninitialised `pixsize` of the first
//! `fread` in the float branch (taken as `sizeof(float)`, the value it is
//! given two statements later), and the uninitialised `total` (taken as 0).

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, SEEK_SET, b3d_fread, b3d_fseek, b3d_fwrite, c_format_bytes,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MRC_MODE_SHORT,
    MrcHeader, mrc_head_label, mrc_head_read, mrc_head_write,
};

/// C `main` in `mrclog.c:34`, returning its process status for the launcher.
pub fn mrclog(argv: &[String]) -> i32 {
    let argc = argv.len();
    let argv0 = argv.first().map_or("mrclog", String::as_str).as_bytes();
    let mut bdata = [0_u8; 1];
    let mut sdata = [0_u8; 2];
    let mut fbytes = [0_u8; 4];
    let mut fpixel: f32;
    let fadd: f32;
    let mut min: f32;
    let mut max: f32;
    let mut total: f32 = 0.0;
    let mut dpixel: f64;
    let mut pixsize: usize;

    let mut datasize: i32;

    if argc < 3 {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "Usage: %s [infile] [outfile]\n",
            &[CArg::Bytes(argv0)],
        ));
        return 3;
    }

    let Some(mut fin) = ImodFile::open(&argv[1], "rb") else {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "Error opening %s.\n",
            &[CArg::Bytes(argv[1].as_bytes())],
        ));
        return 3;
    };

    // `mrclog.c:61` tests `fin` again, so a failed output open goes
    // unreported and the source writes through a NULL stream.  That would
    // crash; exit 3 with the message the source has for it instead.
    let Some(mut fout) = ImodFile::open(&argv[2], "wb") else {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "Error opening %s.\n",
            &[CArg::Bytes(argv[2].as_bytes())],
        ));
        return 3;
    };

    let mut hdata = MrcHeader::default();
    if mrc_head_read(&mut fin, &mut hdata) != 0 {
        let _ = ImodFile::Stderr.write_all(b"Can't Read Input File Header.\n");
        return 3;
    }

    datasize = hdata.nx.wrapping_mul(hdata.ny).wrapping_mul(hdata.nz);

    if hdata.mode == MRC_MODE_COMPLEX_SHORT {
        datasize = datasize.wrapping_mul(2);
    }

    if hdata.mode == MRC_MODE_COMPLEX_FLOAT {
        datasize = datasize.wrapping_mul(2);
    }

    mrc_head_label(&mut hdata, b"mrclog: Took log base 10 of data");
    mrc_head_write(&mut fout, &mut hdata);

    match hdata.mode {
        MRC_MODE_BYTE => {
            pixsize = 1;
            for _ in 0..datasize {
                b3d_fread(&mut bdata, pixsize, 1, &mut fin);
                fpixel = bdata[0] as f32;
                fpixel = fpixel.ln();
                // `bdata = fpixel`: gcc converts through `int` (`cvttss2si`) and keeps
                // the low byte, which is what `log(0) = -inf` becomes natively (0).
                bdata[0] = fpixel as i32 as u8;
                b3d_fwrite(&bdata, pixsize, 1, &mut fout);
            }
        }

        MRC_MODE_SHORT | MRC_MODE_COMPLEX_SHORT => {
            pixsize = 2;
            for _ in 0..datasize {
                b3d_fread(&mut sdata, pixsize, 1, &mut fin);
                fpixel = i16::from_ne_bytes(sdata) as f32;
                fpixel = fpixel.ln();
                // Through `int` and the low 16 bits, as gcc converts (see above).
                sdata = (fpixel as i32 as i16).to_ne_bytes();
                b3d_fwrite(&sdata, pixsize, 1, &mut fout);
            }
        }

        MRC_MODE_FLOAT | MRC_MODE_COMPLEX_FLOAT => {
            pixsize = 4;
            b3d_fread(&mut fbytes, pixsize, 1, &mut fin);
            min = f32::from_ne_bytes(fbytes);
            let mut i = 1;
            while i < datasize {
                b3d_fread(&mut fbytes, pixsize, 1, &mut fin);
                fpixel = f32::from_ne_bytes(fbytes);
                if fpixel < min {
                    min = fpixel;
                }
                i += 1;
            }
            b3d_fseek(&mut fin, 1024, SEEK_SET);

            pixsize = 4;
            if min < 1.0 {
                fadd = 1.0 - min;
            } else {
                fadd = 0.0;
            }
            min = 999999.0;
            max = 0.0;
            for _ in 0..datasize {
                b3d_fread(&mut fbytes, pixsize, 1, &mut fin);
                fpixel = f32::from_ne_bytes(fbytes);
                fpixel += fadd;
                dpixel = fpixel as f64;
                if dpixel < 1.0 {
                    dpixel = 1.0;
                }
                fpixel = dpixel.ln() as f32;
                if fpixel < min {
                    min = fpixel;
                }
                if max < fpixel {
                    max = fpixel;
                }
                total += fpixel;
                b3d_fwrite(&fpixel.to_ne_bytes(), pixsize, 1, &mut fout);
            }
            total = total / datasize as f32;
            hdata.amin = min;
            hdata.amax = max;
            hdata.amean = total;
            mrc_head_write(&mut fout, &mut hdata);
        }

        _ => {
            let _ = ImodFile::Stderr.write_all(&c_format_bytes(
                "%s: data type %d unsupported.\n",
                &[CArg::Bytes(argv0), CArg::Int(hdata.mode as i64)],
            ));
        }
    }

    drop(fin);
    drop(fout);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::mrcfiles::{mrc_head_new, mrc_read_slice};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn transforms_float_stack_and_updates_the_header_statistics() {
        let number = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("imod-rs-mrclog-{}-{number}", std::process::id()));
        let input_path = root.with_extension("input.mrc");
        let output_path = root.with_extension("output.mrc");
        let mut input = ImodFile::open(&input_path, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        assert_eq!(mrc_head_write(&mut input, &mut header), 0);
        let mut data = Vec::new();
        data.extend_from_slice(&(-1.0_f32).to_ne_bytes());
        data.extend_from_slice(&(3.0_f32).to_ne_bytes());
        assert_eq!(b3d_fwrite(&data, 1, data.len(), &mut input), data.len());
        drop(input);
        assert_eq!(
            mrclog(&[
                "mrclog".into(),
                input_path.display().to_string(),
                output_path.display().to_string()
            ]),
            0
        );
        let mut output = ImodFile::open(&output_path, "rb").unwrap();
        let mut result = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut output, &mut result), 0);
        let mut got = vec![0_u8; 8];
        assert_eq!(
            mrc_read_slice(&mut got, &mut output, &mut result, 0, b'Z'),
            0
        );
        let values = got
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<_>>();
        assert_eq!(values[0], 0.0);
        // The source's minimum scan starts wherever `mrc_head_read` left the
        // stream, not at the data (there is no seek before it), so the
        // minimum comes out 0 and `fadd` 1: ln(3 + 1).  Verified against the
        // reference `mrclog.c` built with `flog`, `pixsize` and `total`
        // defined (2026-09-26); the previous expectation, ln(5), encoded a
        // seek the source does not make.
        assert_eq!(values[1], (4.0_f64).ln() as f32, "{values:?}");
        assert_eq!(result.amin, values[0]);
        assert_eq!(result.amax, values[1]);
        drop(output);
        let _ = std::fs::remove_file(input_path);
        let _ = std::fs::remove_file(output_path);
    }
}
