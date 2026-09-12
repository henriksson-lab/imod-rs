//! Parity guards on real cryo-EM movie data.
//!
//! Two real acquisition movies exercise code paths no synthetic fixture in this
//! repository reaches: a 45-frame 4096x4096 8-bit LZW TIFF written in strips of
//! 512 rows (`iitif.c`'s multi-page strip reader), and an EER movie -- a
//! BigTIFF whose 7-bit electron-event directories `iitif.c` decodes onto the
//! 4x super-resolution grid (`IICOMPRESSION_EER_7BIT`, `sEERflags`).
//!
//! The files are 429 MB together and are **not vendored**; see
//! `fixtures/FIXTURE-MANIFEST.md`.  Every test here is skipped unless
//! `IMOD_RS_RAWEM_DIR` names the directory holding them, so the default gate is
//! unaffected.
//!
//! Every expectation below was captured from the native IMOD binaries built
//! from the vendored `IMOD/` tree, each side run in its own directory with
//! stdout taken through a pipe.  MRC outputs are compared with the two
//! documented non-achievables masked: the label slots past `nlabl`, which
//! `mrc_head_new` never clears (`mrcfiles.c:689-766`), and the
//! `dd-Mmm-yy  HH:MM:SS` stamp inside the labels that are used.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Every PIP-driven invocation needs an autodoc directory, exactly as a real
/// IMOD install provides one through `AUTODOC_DIR` or `IMOD_DIR`.
const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

const TIFF_MOVIE: &str = "FoilHole_7751092_Data_7746149_27_20260608_205700_Fractions.tiff";
const EER_MOVIE: &str = "Position_1_9_001_0.00_20250909_165844_EER.eer";

/// The dataset directory, or `None` when `IMOD_RS_RAWEM_DIR` is unset or does
/// not hold both movies -- in which case every test in this file returns
/// without asserting anything.
fn dataset() -> Option<PathBuf> {
    let directory = PathBuf::from(std::env::var_os("IMOD_RS_RAWEM_DIR")?);
    if directory.join(TIFF_MOVIE).is_file() && directory.join(EER_MOVIE).is_file() {
        Some(directory)
    } else {
        None
    }
}

/// A scratch directory of this test's own, so each command's outputs are the
/// only files in it.
fn work(name: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("imod-rs-rawem-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// Runs a translated command in `directory` and returns its exit status,
/// stdout and stderr.
fn run(command: &str, directory: &Path, args: &[&str]) -> (i32, String, String) {
    let output = common::imod_cmd(command)
        .env("AUTODOC_DIR", AUTODOC)
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap();
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// SHA-256 of an MRC file with the two documented non-achievables masked.
///
/// `mrc_head_new` (`mrcfiles.c:689-766`) never clears `labels`, and the native
/// callers put the header on the stack, so the reference's label slots past
/// `nlabl` carry stack residue that varies run to run; and the label the
/// program does write carries a `dd-Mmm-yy  HH:MM:SS` stamp.
fn masked_mrc_digest(path: &Path) -> String {
    let mut bytes = std::fs::read(path).unwrap();
    assert!(bytes.len() >= 1024, "{} is not an MRC file", path.display());
    let nlabl = i32::from_le_bytes(bytes[220..224].try_into().unwrap());
    assert!((0..=10).contains(&nlabl), "nlabl {nlabl} out of range");
    let used = 224 + 80 * nlabl as usize;
    bytes[used..1024].fill(0);
    // Blank `dd-Mmm-yy  HH:MM:SS` wherever it falls inside the used labels.
    let stamp: Vec<usize> = (224..used.saturating_sub(18))
        .filter(|&i| {
            bytes[i].is_ascii_digit()
                && bytes[i + 1].is_ascii_digit()
                && bytes[i + 2] == b'-'
                && bytes[i + 6] == b'-'
                && bytes[i + 7].is_ascii_digit()
                && bytes[i + 8].is_ascii_digit()
        })
        .collect();
    for start in stamp {
        for byte in &mut bytes[start..(start + 19).min(1024)] {
            *byte = b'#';
        }
    }
    sha256_hex(&bytes)
}

/// SHA-256, written out here because the crate has no hashing dependency.
fn sha256_hex(message: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut padded = message.to_vec();
    let bits = (message.len() as u64) * 8;
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bits.to_be_bytes());
    for block in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in block.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

/// `header` on the 45-frame LZW TIFF movie.  Native prints the strip geometry
/// line from `iiuPrintHeader`; the whole stream is byte-identical.
#[test]
fn header_reports_the_tiff_movie() {
    let Some(directory) = dataset() else {
        return;
    };
    let movie = directory.join(TIFF_MOVIE);
    let work = work("header-tiff");
    let (status, stdout, stderr) = run("header", &work, [movie.to_str().unwrap()].as_slice());
    assert_eq!(status, 0);
    assert_eq!(stderr, "");
    assert_eq!(
        stdout,
        "\n RO image file on unit   1 : {TIFF}     Size=     122628 K\n\n                    This is a TIFF file (in strips of   4096 x    512).\n\n Number of columns, rows, sections .....    4096    4096      45\n Map mode ..............................    0   (byte)                     \n Start cols, rows, sects, grid x,y,z ...    0     0     0    4096   4096     45\n Pixel spacing (Angstroms)..............   1.000      1.000      1.000    \n Cell angles ...........................   90.000   90.000   90.000\n Fast, medium, slow axes ...............    X    Y    Z\n Origin on x,y,z .......................    0.000       0.000       0.000    \n Minimum density .......................   0.0000    \n Maximum density .......................   255.00    \n Mean density ..........................   127.50    \n tilt angles (original,current) ........   0.0   0.0   0.0   0.0   0.0   0.0\n Space group,# extra bytes,idtype,lens .        0        0        0        0\n\n     0 Titles :\n\n".replace("{TIFF}", movie.to_str().unwrap())
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// `header -size` on both movies.  The EER file reads as the 4x
/// super-resolution grid, 16384 x 16384 x 522.
#[test]
fn header_size_reports_both_movies() {
    let Some(directory) = dataset() else {
        return;
    };
    let work = work("header-size");
    let tiff = directory.join(TIFF_MOVIE);
    let (status, stdout, stderr) = run("header", &work, &["-size", tiff.to_str().unwrap()]);
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(stdout, "    4096    4096      45\n");
    let eer = directory.join(EER_MOVIE);
    let (status, stdout, stderr) = run("header", &work, &["-size", eer.to_str().unwrap()]);
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(stdout, "   16384   16384     522\n");
    let _ = std::fs::remove_dir_all(&work);
}

/// `header` on the EER movie.
#[test]
fn header_reports_the_eer_movie() {
    let Some(directory) = dataset() else {
        return;
    };
    let movie = directory.join(EER_MOVIE);
    let work = work("header-eer");
    let (status, stdout, stderr) = run("header", &work, [movie.to_str().unwrap()].as_slice());
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(
        stdout,
        "\n RO image file on unit   1 : {EER}     Size=     316033 K\n\n                    This is a TIFF file.\n\n Number of columns, rows, sections .....   16384   16384     522\n Map mode ..............................    0   (byte)                     \n Start cols, rows, sects, grid x,y,z ...    0     0     0   16384  16384    522\n Pixel spacing (Angstroms)..............   1.000      1.000      1.000    \n Cell angles ...........................   90.000   90.000   90.000\n Fast, medium, slow axes ...............    X    Y    Z\n Origin on x,y,z .......................    0.000       0.000       0.000    \n Minimum density .......................   0.0000    \n Maximum density .......................   1.0000    \n Mean density ..........................  0.16812E-02\n tilt angles (original,current) ........   0.0   0.0   0.0   0.0   0.0   0.0\n Space group,# extra bytes,idtype,lens .        0        0        0        0\n\n     0 Titles :\n\n".replace("{EER}", movie.to_str().unwrap())
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// `clip info` on both movies.
#[test]
fn clip_info_reports_both_movies() {
    let Some(directory) = dataset() else {
        return;
    };
    let work = work("clip-info");
    let tiff = directory.join(TIFF_MOVIE);
    let (status, stdout, stderr) = run("clip", &work, &["info", tiff.to_str().unwrap()]);
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(
        stdout,
        "MRC header info:\nmode = Byte\nImage size    =  ( 4096, 4096, 45)\nminimum value = 0\nmaximum value = 255\nmean value    = 127.5\nStart reading image at ( 0, 0, 0).\nRead length   =  ( 4096, 4096, 45).\nScale         =  ( 1 x 1 x 1 ) Angstrom.\nCell Rotation =  ( 90, 90, 90).\nColumns are   = axis 1\nRows are      = axis 2\nSections are  = axis 3\nangles = ( 0, 0, 0, 0, 0, 0)\norgin  = ( 0, 0, 0)\ncreator id = 0\nThare are 0 labels.\n\n"
    );
    let eer = directory.join(EER_MOVIE);
    let (status, stdout, stderr) = run("clip", &work, &["info", eer.to_str().unwrap()]);
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(
        stdout,
        "MRC header info:\nmode = Byte\nImage size    =  ( 16384, 16384, 522)\nminimum value = 0\nmaximum value = 1\nmean value    = 0.0016812\nStart reading image at ( 0, 0, 0).\nRead length   =  ( 16384, 16384, 522).\nScale         =  ( 1 x 1 x 1 ) Angstrom.\nCell Rotation =  ( 90, 90, 90).\nColumns are   = axis 1\nRows are      = axis 2\nSections are  = axis 3\nangles = ( 0, 0, 0, 0, 0, 0)\norgin  = ( 0, 0, 0)\ncreator id = 0\nThare are 0 labels.\n\n"
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// `clip stats` decodes every LZW strip of all 45 frames and prints a row per
/// frame.  The sub-pixel maximum coordinates and the very low means are what
/// raw counting-mode data looks like, so this is a real workout for the byte to
/// float conversion and the per-slice statistics.
#[test]
fn clip_stats_walks_every_tiff_frame() {
    let Some(directory) = dataset() else {
        return;
    };
    let work = work("clip-stats");
    let tiff = directory.join(TIFF_MOVIE);
    let (status, stdout, stderr) = run("clip", &work, &["stats", tiff.to_str().unwrap()]);
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(
        stdout,
        "slice|   min   |(   x,   y)|    max  |(      x,      y)|   mean    |  std dev.\n-----|---------|-----------|---------|-----------------|-----------|----------\n   0     0.0000 (   0,   0)   12.0000 ( 159.00, 709.95)    0.3414     0.5821\n   1     0.0000 (   0,   0)   16.0000 ( 159.02, 709.98)    0.3413     0.5827\n   2     0.0000 (   0,   0)   11.0000 (1764.00,3789.02)    0.3415     0.5827\n   3     0.0000 (   0,   0)   13.0000 ( 159.00, 710.00)    0.3414     0.5828\n   4     0.0000 (   0,   0)   12.0000 ( 158.95, 709.98)    0.3415     0.5828\n   5     0.0000 (   0,   0)   14.0000 ( 159.07, 709.93)    0.3416     0.5829\n   6     0.0000 (   0,   0)    8.0000 ( 159.00, 710.03)    0.3416     0.5827\n   7     0.0000 (   0,   0)   14.0000 ( 159.02, 709.94)    0.3415     0.5824\n   8     0.0000 (   1,   0)   16.0000 ( 159.00, 710.02)    0.3416     0.5824\n   9     0.0000 (   0,   0)   13.0000 ( 158.98, 709.98)    0.3415     0.5823\n  10     0.0000 (   0,   0)   14.0000 ( 159.00, 710.02)    0.3417     0.5829\n  11     0.0000 (   0,   0)   15.0000 ( 158.98, 709.98)    0.3417     0.5830\n  12     0.0000 (   0,   0)   13.0000 ( 159.02, 710.02)    0.3416     0.5829\n  13     0.0000 (   0,   0)    9.0000 ( 159.07, 710.00)    0.3421     0.5832\n  14     0.0000 (   0,   0)   13.0000 ( 159.00, 709.93)    0.3421     0.5830\n  15     0.0000 (   0,   0)   12.0000 ( 159.00, 710.00)    0.3418     0.5826\n  16     0.0000 (   0,   0)   14.0000 ( 158.98, 710.00)    0.3420     0.5830\n  17     0.0000 (   0,   0)   12.0000 ( 159.06, 709.98)    0.3418     0.5829\n  18     0.0000 (   0,   0)   13.0000 ( 159.02, 710.02)    0.3419     0.5831\n  19     0.0000 (   0,   0)   12.0000 ( 159.04, 709.98)    0.3419     0.5831\n  20     0.0000 (   0,   0)   11.0000 ( 159.05, 710.03)    0.3421     0.5832\n  21     0.0000 (   0,   0)   11.0000 ( 159.00, 709.95)    0.3422     0.5834\n  22     0.0000 (   0,   0)    7.0000 ( 159.00, 709.97)    0.3422     0.5832\n  23     0.0000 (   0,   0)   10.0000 ( 159.05, 710.05)    0.3418     0.5827\n  24     0.0000 (   0,   0)   12.0000 ( 159.02, 709.98)    0.3421     0.5830\n  25     0.0000 (   0,   0)   13.0000 ( 158.96, 710.00)    0.3422     0.5833\n  26     0.0000 (   0,   0)   11.0000 ( 159.02, 710.02)    0.3420     0.5832\n  27     0.0000 (   0,   0)   10.0000 ( 253.97,3967.05)    0.3420     0.5831\n  28     0.0000 (   0,   0)   14.0000 ( 158.96, 710.00)    0.3423     0.5833\n  29     0.0000 (   0,   0)   10.0000 ( 158.95, 709.97)    0.3420     0.5832\n  30     0.0000 (   0,   0)   11.0000 ( 158.94, 710.03)    0.3425     0.5832\n  31     0.0000 (   0,   0)   15.0000 ( 158.98, 709.97)    0.3421     0.5828\n  32     0.0000 (   0,   0)   15.0000 ( 159.00, 710.05)    0.3423     0.5831\n  33     0.0000 (   0,   0)   11.0000 ( 159.03, 709.92)    0.3423     0.5832\n  34     0.0000 (   0,   0)   11.0000 ( 158.97, 709.94)    0.3423     0.5834\n  35     0.0000 (   0,   0)   11.0000 ( 158.98, 710.05)    0.3423     0.5834\n  36     0.0000 (   0,   0)   10.0000 ( 159.06, 710.03)    0.3425     0.5834\n  37     0.0000 (   0,   0)   11.0000 ( 159.04, 709.93)    0.3425     0.5834\n  38     0.0000 (   0,   0)   14.0000 ( 158.98, 710.00)    0.3425     0.5833\n  39     0.0000 (   0,   0)   12.0000 ( 159.00, 710.02)    0.3424     0.5833\n  40     0.0000 (   0,   0)   13.0000 ( 159.02, 709.93)    0.3426     0.5835\n  41     0.0000 (   0,   0)   13.0000 ( 159.04, 710.00)    0.3424     0.5835\n  42     0.0000 (   0,   0)   13.0000 ( 158.98, 710.02)    0.3423     0.5834\n  43     0.0000 (   0,   0)   11.0000 ( 159.03, 709.92)    0.3423     0.5833\n  44     0.0000 (   0,   0)   12.0000 ( 158.98, 710.02)    0.3425     0.5834\n all     0.0000 (@ z=    0)   16.0000 (@ z=    1      )    0.3420     0.5830\n"
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// `newstack -secs 0,1` off the TIFF movie.  Before the fix at
/// `newstack.rs`'s preliminary pass this aborted: the translation read
/// `ImodImageFile.header` as an `MrcHeader`, which for a TIFF is the libtiff
/// `TIFF *` (`iitif.c:204, 687`), so the sizes were garbage.
#[test]
fn newstack_extracts_tiff_movie_sections() {
    let Some(directory) = dataset() else {
        return;
    };
    let work = work("newstack-secs");
    let tiff = directory.join(TIFF_MOVIE);
    let (status, stdout, stderr) = run(
        "newstack",
        &work,
        &["-secs", "0,1", tiff.to_str().unwrap(), "f.mrc"],
    );
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(
        stdout,
        "\n RO image file on unit   1 : {TIFF}     Size=     122628 K\n\n                    This is a TIFF file (in strips of   4096 x    512).\n\n Number of columns, rows, sections .....    4096    4096      45\n Map mode ..............................    0   (byte)                     \n Start cols, rows, sects, grid x,y,z ...    0     0     0    4096   4096     45\n Pixel spacing (Angstroms)..............   1.000      1.000      1.000    \n Cell angles ...........................   90.000   90.000   90.000\n Fast, medium, slow axes ...............    X    Y    Z\n Origin on x,y,z .......................    0.000       0.000       0.000    \n Minimum density .......................   0.0000    \n Maximum density .......................   255.00    \n Mean density ..........................   127.50    \n tilt angles (original,current) ........   0.0   0.0   0.0   0.0   0.0   0.0\n Space group,# extra bytes,idtype,lens .        0        0        0        0\n\n     0 Titles :\n\n\n NEW image file on unit   2 : f.mrc\n section   input min&max       output min&max  &  mean\n       0      0.00     12.00      0.00     12.00      0.34\n       1      0.00     16.00      0.00     16.00      0.34\n".replace("{TIFF}", tiff.to_str().unwrap())
    );
    let output = work.join("f.mrc");
    assert_eq!(std::fs::metadata(&output).unwrap().len(), 33555456);
    assert_eq!(
        masked_mrc_digest(&output),
        "c51587299849617efa0b128fd907607cf79a70aced255ebdcfd4e560e5af28f1"
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// The reductions and mode conversions off the TIFF movie, each verified
/// against native on stdout as well as on the output bytes.
#[test]
fn newstack_reduces_and_converts_tiff_movie_sections() {
    let Some(directory) = dataset() else {
        return;
    };
    let tiff = directory.join(TIFF_MOVIE);
    let tiff = tiff.to_str().unwrap();
    for (name, args, size, digest, golden) in [
        (
            "bin4",
            vec!["-secs", "0", "-bin", "4", tiff, "bin4.mrc"],
            1049600u64,
            "4872d5ae13a9700ffcc3476f1e8560d437bf452282d33f3974a0ad1a1ac75469",
            "\n RO image file on unit   1 : {TIFF}     Size=     122628 K\n\n                    This is a TIFF file (in strips of   4096 x    512).\n\n Number of columns, rows, sections .....    4096    4096      45\n Map mode ..............................    0   (byte)                     \n Start cols, rows, sects, grid x,y,z ...    0     0     0    4096   4096     45\n Pixel spacing (Angstroms)..............   1.000      1.000      1.000    \n Cell angles ...........................   90.000   90.000   90.000\n Fast, medium, slow axes ...............    X    Y    Z\n Origin on x,y,z .......................    0.000       0.000       0.000    \n Minimum density .......................   0.0000    \n Maximum density .......................   255.00    \n Mean density ..........................   127.50    \n tilt angles (original,current) ........   0.0   0.0   0.0   0.0   0.0   0.0\n Space group,# extra bytes,idtype,lens .        0        0        0        0\n\n     0 Titles :\n\n\n NEW image file on unit   2 : bin4.mrc\n section   input min&max       output min&max  &  mean\n       0      0.00      1.19      0.00      1.19      0.34\n",
        ),
        (
            "bin2",
            vec!["-secs", "0-4", "-bin", "2", tiff, "bin2.mrc"],
            20972544,
            "b3f8363c777866e2098b134f9fc76d1c7a55953b7508cded6ea7448722467711",
            "\n RO image file on unit   1 : {TIFF}     Size=     122628 K\n\n                    This is a TIFF file (in strips of   4096 x    512).\n\n Number of columns, rows, sections .....    4096    4096      45\n Map mode ..............................    0   (byte)                     \n Start cols, rows, sects, grid x,y,z ...    0     0     0    4096   4096     45\n Pixel spacing (Angstroms)..............   1.000      1.000      1.000    \n Cell angles ...........................   90.000   90.000   90.000\n Fast, medium, slow axes ...............    X    Y    Z\n Origin on x,y,z .......................    0.000       0.000       0.000    \n Minimum density .......................   0.0000    \n Maximum density .......................   255.00    \n Mean density ..........................   127.50    \n tilt angles (original,current) ........   0.0   0.0   0.0   0.0   0.0   0.0\n Space group,# extra bytes,idtype,lens .        0        0        0        0\n\n     0 Titles :\n\n\n NEW image file on unit   2 : bin2.mrc\n section   input min&max       output min&max  &  mean\n       0      0.00      3.00      0.00      3.00      0.34\n       1      0.00      4.00      0.00      4.00      0.34\n       2      0.00      3.00      0.00      3.00      0.34\n       3      0.00      3.25      0.00      3.25      0.34\n       4      0.00      3.25      0.00      3.25      0.34\n",
        ),
        (
            "shrink3",
            vec!["-secs", "0,1", "-shrink", "3", tiff, "sh.mrc"],
            3732936,
            "6a2350bd39d0d28aa434495f8640a53994c1d730694869076202fdb93fb56cc8",
            "\n RO image file on unit   1 : {TIFF}     Size=     122628 K\n\n                    This is a TIFF file (in strips of   4096 x    512).\n\n Number of columns, rows, sections .....    4096    4096      45\n Map mode ..............................    0   (byte)                     \n Start cols, rows, sects, grid x,y,z ...    0     0     0    4096   4096     45\n Pixel spacing (Angstroms)..............   1.000      1.000      1.000    \n Cell angles ...........................   90.000   90.000   90.000\n Fast, medium, slow axes ...............    X    Y    Z\n Origin on x,y,z .......................    0.000       0.000       0.000    \n Minimum density .......................   0.0000    \n Maximum density .......................   255.00    \n Mean density ..........................   127.50    \n tilt angles (original,current) ........   0.0   0.0   0.0   0.0   0.0   0.0\n Space group,# extra bytes,idtype,lens .        0        0        0        0\n\n     0 Titles :\n\n\n NEW image file on unit   2 : sh.mrc\n section   input min&max       output min&max  &  mean\n       0      0.00      1.50      0.00      1.50      0.34\n       1      0.00      1.65      0.00      1.65      0.34\n TRUNCATIONS OCCURRED:      33092 at low end,          0 at high end of range\n",
        ),
        (
            "float2",
            vec!["-secs", "0,1", "-float", "2", tiff, "fl2.mrc"],
            33555456,
            "bb9e6a0cbf006399689310fd72b8c39a362ba79ae5230c816409310aac66d23b",
            "\n RO image file on unit   1 : {TIFF}     Size=     122628 K\n\n                    This is a TIFF file (in strips of   4096 x    512).\n\n Number of columns, rows, sections .....    4096    4096      45\n Map mode ..............................    0   (byte)                     \n Start cols, rows, sects, grid x,y,z ...    0     0     0    4096   4096     45\n Pixel spacing (Angstroms)..............   1.000      1.000      1.000    \n Cell angles ...........................   90.000   90.000   90.000\n Fast, medium, slow axes ...............    X    Y    Z\n Origin on x,y,z .......................    0.000       0.000       0.000    \n Minimum density .......................   0.0000    \n Maximum density .......................   255.00    \n Mean density ..........................   127.50    \n tilt angles (original,current) ........   0.0   0.0   0.0   0.0   0.0   0.0\n Space group,# extra bytes,idtype,lens .        0        0        0        0\n\n     0 Titles :\n\n\n NEW image file on unit   2 : fl2.mrc\n section   input min&max       output min&max  &  mean\n       0      0.00     12.00      0.00    191.43      5.45\n       1      0.00     16.00      0.01    255.00      5.45\n",
        ),
    ] {
        let work = work(name);
        let output = work.join(args[args.len() - 1]);
        let (status, stdout, stderr) = run("newstack", &work, &args);
        assert_eq!((status, stderr.as_str()), (0, ""), "{name}");
        assert_eq!(stdout, golden.replace("{TIFF}", tiff), "{name}");
        assert_eq!(std::fs::metadata(&output).unwrap().len(), size, "{name}");
        assert_eq!(masked_mrc_digest(&output), digest, "{name}");
        let _ = std::fs::remove_dir_all(&work);
    }
}

/// Frame summing -- the operation the dataset's own readme calls the useful one
/// ("you could also do the equivalent by simply summing the frames of the
/// movie").  `clip avg` is IMOD's way: `clipAverage` (`processing.cpp`)
/// accumulates every section of the input into one output section.
#[test]
fn clip_avg_sums_the_tiff_movie_frames() {
    let Some(directory) = dataset() else {
        return;
    };
    let work = work("clip-avg");
    let tiff = directory.join(TIFF_MOVIE);
    let (status, stdout, stderr) = run("clip", &work, &["avg", tiff.to_str().unwrap(), "sum.mrc"]);
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(stdout, "2D Averaging...\n");
    let output = work.join("sum.mrc");
    assert_eq!(std::fs::metadata(&output).unwrap().len(), 16778240);
    assert_eq!(
        masked_mrc_digest(&output),
        "c4e2b1d613d90503ed372bdfac36196ea5924553c5902144d93592aac228743f"
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// `tif2mrc -s` converting the whole multi-page TIFF movie -- the command's
/// core job, and the first real one this translation has seen.  The output is
/// 755 MB, so the test removes it immediately.
#[test]
fn tif2mrc_converts_the_whole_tiff_movie() {
    let Some(directory) = dataset() else {
        return;
    };
    let work = work("tif2mrc");
    let tiff = directory.join(TIFF_MOVIE);
    let (status, stdout, stderr) = run("tif2mrc", &work, &["-s", tiff.to_str().unwrap(), "s.mrc"]);
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(
        stdout,
        "Reading multi-paged TIFF file.\nConverting 45 images size 4096 x 4096\nMin = 0, Max = 16, Mean = 0.342012\n"
    );
    let output = work.join("s.mrc");
    assert_eq!(std::fs::metadata(&output).unwrap().len(), 754975744);
    let digest = masked_mrc_digest(&output);
    let _ = std::fs::remove_dir_all(&work);
    assert_eq!(
        digest,
        "1f1cff95081c64c250e0433306acd06dcac6c344a07f97967be8cf10ee7f7ae4"
    );
}

/// The EER movie through `newstack`: two frames of the 16384 x 16384
/// super-resolution grid, binned by 8.  This is the 7-bit electron-event
/// decode (`iitif.c`'s `IICOMPRESSION_EER_7BIT` path) feeding the binned
/// reader.
#[test]
fn newstack_bins_eer_movie_frames() {
    let Some(directory) = dataset() else {
        return;
    };
    let work = work("newstack-eer");
    let eer = directory.join(EER_MOVIE);
    let (status, stdout, stderr) = run(
        "newstack",
        &work,
        &["-secs", "0,1", "-bin", "8", eer.to_str().unwrap(), "e8.mrc"],
    );
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(
        stdout,
        "\n RO image file on unit   1 : {EER}     Size=     316033 K\n\n                    This is a TIFF file.\n\n Number of columns, rows, sections .....   16384   16384     522\n Map mode ..............................    0   (byte)                     \n Start cols, rows, sects, grid x,y,z ...    0     0     0   16384  16384    522\n Pixel spacing (Angstroms)..............   1.000      1.000      1.000    \n Cell angles ...........................   90.000   90.000   90.000\n Fast, medium, slow axes ...............    X    Y    Z\n Origin on x,y,z .......................    0.000       0.000       0.000    \n Minimum density .......................   0.0000    \n Maximum density .......................   1.0000    \n Mean density ..........................  0.16812E-02\n tilt angles (original,current) ........   0.0   0.0   0.0   0.0   0.0   0.0\n Space group,# extra bytes,idtype,lens .        0        0        0        0\n\n     0 Titles :\n\n\n NEW image file on unit   2 : e8.mrc\n section   input min&max       output min&max  &  mean\n       0      0.00      0.06      0.00      0.06      0.00\n       1      0.00      0.05      0.00      0.05      0.00\n".replace("{EER}", eer.to_str().unwrap())
    );
    let output = work.join("e8.mrc");
    assert_eq!(std::fs::metadata(&output).unwrap().len(), 8389632);
    assert_eq!(
        masked_mrc_digest(&output),
        "37bbc53eee99fefcf715a7b2a0842a27f584b15b4ed9a80f856ce68f3973dac2"
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// Asking for a section the TIFF movie does not have.  The message and the
/// exit status both come from native.
#[test]
fn newstack_rejects_a_section_past_the_tiff_movie() {
    let Some(directory) = dataset() else {
        return;
    };
    let work = work("newstack-oor");
    let tiff = directory.join(TIFF_MOVIE);
    let (status, stdout, stderr) = run(
        "newstack",
        &work,
        &["-secs", "50", tiff.to_str().unwrap(), "x.mrc"],
    );
    assert_eq!(status, 1);
    assert_eq!(stderr, "");
    assert_eq!(
        stdout,
        "\nERROR: NEWSTACK -       50 is an illegal section number for {TIFF}\n"
            .replace("{TIFF}", tiff.to_str().unwrap())
    );
    let _ = std::fs::remove_dir_all(&work);
}
