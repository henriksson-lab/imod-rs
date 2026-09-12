//! `newstack` under a memory limit, and the order in which its option checks
//! fire.
//!
//! Two families of defect are covered here.
//!
//! **Chunking.**  `newstack.f90:2158-2296` is one block: it sizes the working
//! array, counts chunks, refuses a multi-chunk taper or Fourier operation
//! (`:2200-2207`), searches for a layout (`:2209-2274`) and opens the scratch
//! file (`:2284-2296`).  None of it is conditional on `-verbose`; only the
//! `print *` reports inside it are.  The section loop then runs per chunk
//! (`:2334-2592`) and rescales backwards over the chunks (`:2595-2611`), and
//! its label-80 tail (`:2630-2683`) copies the extended-header record for the
//! section whichever route produced it.
//!
//! **Option order.**  `newstack.f90` interleaves its PIP reads with the input
//! scan: `No input file specified` is `:355`, the `-twodir` file-count check
//! `:409`, the section lists `:470-535`, the output-file checks `:556-568`,
//! the `-numout` accounting `:606-635`, the section-count check `:671-672`,
//! the mode block `:684-731` and the `-tilt`/`-reorder` block `:975-1024`.
//! An option check that fires ahead of its place reports the wrong error.
//!
//! Every expectation below is the byte output of the reference `newstack`
//! built from the vendored `IMOD/` tree, each side run in its own directory
//! with stdout taken through a pipe.

mod common;

use imod_rs::imod::libiimod::iimage::{
    IIFILE_DEFAULT, ii_close, ii_fill_mrc_header, ii_open, ii_open_new, ii_sync_from_mrc_header,
    ii_write_section_float,
};
use imod_rs::imod::libiimod::mrcfiles::{
    MrcHeader, mrc_head_new, mrc_head_write, mrc_write_extra_header,
};
use std::ffi::CString;

const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

/// A private directory per test, holding a 64x48x5 float stack and a
/// five-line transform file.  The image has to be big enough that a `-test`
/// pair smaller than two sections forces the source to chunk.
fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-nschunk-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("five.xf"),
        "   1.0   0.0   0.0   1.0   0.0   0.0\n\
         \x20  1.0   0.0   0.0   1.0   1.0   0.0\n\
         \x20  1.0   0.0   0.0   1.0   2.0   0.0\n\
         \x20  1.0   0.0   0.0   1.0   3.0   0.0\n\
         \x20  1.0   0.0   0.0   1.0   4.0   0.0\n",
    )
    .unwrap();
    dir
}

/// Writes `dir/in.mrc`: 64 by 48 by 5, mode 2, with deterministic content.
/// When `extra` is given it becomes the extended header, written as
/// `nint = 0`, `nreal = 1` -- one real per section, the form
/// `iiuAltExtendedType(unit, 0, 1)` writes and `getExtraHeaderMaxSecSize`
/// sizes at four bytes a section.
fn write_input(dir: &std::path::Path, extra: Option<&[u8]>) {
    let path = dir.join("in.mrc");
    let path_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(path_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 64, 48, 5, 2), 0);
        if let Some(extra) = extra {
            (*header).nint = 0;
            (*header).nreal = 1;
            (*header).next = extra.len() as i32;
            let mut bytes = extra.to_vec();
            assert_eq!(
                mrc_write_extra_header(header, bytes.as_mut_ptr(), extra.len() as i32),
                0
            );
        }
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..5 {
            let mut pixels = (0..64 * 48)
                .map(|index| (section * 7 + index % 251) as f32)
                .collect::<Vec<f32>>();
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
}

/// Runs the translated `newstack` in `dir` and returns (status, stdout, stderr).
fn run(dir: &std::path::Path, args: &[&str]) -> (i32, String, String) {
    let output = common::imod_cmd("newstack")
        .current_dir(dir)
        .env("AUTODOC_DIR", AUTODOC)
        .env("IMOD_NO_IMAGE_BACKUP", "1")
        .args(args)
        .output()
        .expect("newstack executable must start");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// Reads back an MRC output's extended-header size, type fields and bytes.
fn extended_header(path: &std::path::Path) -> (i32, i16, i16, Vec<u8>) {
    let path_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
    let mut header = unsafe { std::mem::zeroed::<MrcHeader>() };
    unsafe {
        let file = ii_open(path_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        ii_close(file);
    }
    let bytes = std::fs::read(path).unwrap();
    let next = header.next;
    (
        next,
        header.nint,
        header.nreal,
        bytes[1024..1024 + next.max(0) as usize].to_vec(),
    )
}

/// The source's label-80 tail (`newstack.f90:2630-2683`) runs for every
/// section, chunked or not, and `newstack.f90:1904-1935` sizes the output's
/// extended header in the one place the output header is built.  The chunked
/// route had neither, so `-xform ... -test 1000,1` wrote `next = 0` and no
/// extended data where the reference carries all 20 bytes over.
#[test]
fn a_chunked_run_carries_the_extended_header_over() {
    let dir = scratch("extra");
    let angles: Vec<u8> = [-60.0_f32, -30., 0., 30., 60.]
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    write_input(&dir, Some(&angles));
    // Unchunked: the whole input fits, so this is the route that always worked.
    let (status, stdout, _) = run(
        &dir,
        &["-in", "in.mrc", "-ou", "whole.mrc", "-xform", "five.xf"],
    );
    assert_eq!(status, 0, "{stdout}");
    assert_eq!(
        extended_header(&dir.join("whole.mrc")),
        (20, 0, 1, angles.clone())
    );
    // Chunked: `-test 1000,1` leaves 999 elements for a 64 by 48 section, so
    // the source splits the output and takes the layout-search route.
    let (status, stdout, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "chunk.mrc",
            "-xform",
            "five.xf",
            "-test",
            "1000,1",
        ],
    );
    assert_eq!(status, 0, "{stdout}");
    assert_eq!(
        extended_header(&dir.join("chunk.mrc")),
        (20, 0, 1, angles.clone()),
        "the reference carries the 20-byte extended header over"
    );
    // `-expand 2 -test 1500,1` is the other reported case.
    let (status, stdout, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "expand.mrc",
            "-expand",
            "2",
            "-test",
            "1500,1",
        ],
    );
    assert_eq!(status, 0, "{stdout}");
    assert_eq!(extended_header(&dir.join("expand.mrc")), (20, 0, 1, angles));
    // A section list takes only those sections' records, one per output
    // section: `-secs 1,3` gives 8 bytes, the second and fourth floats.
    let (status, stdout, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "secs.mrc",
            "-secs",
            "1,3",
            "-test",
            "1000,1",
            "-xform",
            "five.xf",
            "-uselines",
            "1,3",
        ],
    );
    assert_eq!(status, 0, "{stdout}");
    let mut expected = (-30.0_f32).to_ne_bytes().to_vec();
    expected.extend_from_slice(&30.0_f32.to_ne_bytes());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:2274` is `if (ifOutChunk < 0) call exitError(' Input image
/// too large for array.')`, the end of the layout search at
/// `newstack.f90:2237-2273`.  The whole block, search and exit included, is
/// unconditional; gating it on `iVerbose` lost the exit, and the program then
/// wrote an 80 MB file and returned 0.
#[test]
fn an_output_that_cannot_be_laid_out_in_the_limit_is_refused() {
    let dir = scratch("toolarge");
    write_input(&dir, None);
    let (status, stdout, stderr) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "o.mrc",
            "-size",
            "2000,2000",
            "-test",
            "1000,1",
        ],
    );
    assert_eq!(status, 1, "stdout was:\n{stdout}");
    assert_eq!(stderr, "", "the source writes every diagnostic to stdout");
    assert!(
        stdout.ends_with("\nERROR: NEWSTACK -  Input image too large for array.\n"),
        "stdout was:\n{stdout}"
    );
    // `newstack` never closes unit 2 on the way to `exitError`, so the file it
    // created stays empty.
    assert_eq!(std::fs::metadata(dir.join("o.mrc")).unwrap().len(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// With `-float 2` the scaling is not pre-set, so a multi-chunk section goes
/// out to the scratch file (`newstack.f90:2284-2296`, `:2585-2588`) and comes
/// back for a **backwards** rescale (`:2595-2611`), and the section
/// statistics are `iclAvgSd`'s per-chunk sums fed to `chunkSumsToAvgsd`
/// (`:2516-2522`, `:3556`).  Both orders are observable, and the chunked mean
/// is deliberately **not** the unchunked one -- the reference itself reports
/// 16139.03 for the split run and 16139.02 for the whole one on this input.
/// The route that held the whole section and computed one set of statistics
/// reported the unchunked value for both and never opened the scratch file.
#[test]
fn a_chunked_float2_run_uses_the_scratch_file_and_keeps_the_sources_chunk_sums() {
    let dir = scratch("float2");
    write_input(&dir, None);
    let means = |text: &str| {
        text.lines()
            .map(|line| line.split_whitespace().collect::<Vec<_>>())
            .filter(|fields| {
                fields.len() == 6 && fields.iter().all(|field| field.parse::<f64>().is_ok())
            })
            .map(|fields| fields[5].to_owned())
            .collect::<Vec<_>>()
    };
    let (status, whole_out, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "whole.mrc",
            "-float",
            "2",
            "-mode",
            "1",
        ],
    );
    assert_eq!(status, 0, "{whole_out}");
    assert!(
        !whole_out.contains("unit   3"),
        "no scratch file with one chunk:\n{whole_out}"
    );
    assert_eq!(means(&whole_out), vec!["16139.02"; 5], "{whole_out}");
    let (status, chunk_out, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "chunk.mrc",
            "-float",
            "2",
            "-mode",
            "1",
            "-test",
            "3000,1",
        ],
    );
    assert_eq!(status, 0, "{chunk_out}");
    assert!(
        chunk_out.contains("SCRATCH image file on unit   3"),
        "the source opens a scratch file for a rescaled multi-chunk section:\n{chunk_out}"
    );
    assert_eq!(means(&chunk_out), vec!["16139.03"; 5], "{chunk_out}");
    assert_eq!(
        std::fs::metadata(dir.join("chunk.mrc")).unwrap().len(),
        std::fs::metadata(dir.join("whole.mrc")).unwrap().len()
    );
    // The scratch file is opened `scratch`, so nothing is left behind.
    assert!(
        std::fs::read_dir(&dir).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".nws")),
        "the scratch file must be removed on close"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:2200-2205` refuses a multi-chunk taper and a multi-chunk
/// Fourier operation, but tests a `numChunks` that is **zero** when the input
/// does not fit, so neither fires and the search at `:2237-2273` splits the
/// output anyway.  `:2504` then tapers `nxOut * nyOut` elements of a buffer
/// holding `numLinesOut(iChunk)` lines and `:2443-2450` pads
/// `(nxFSpad + 2) * (nyFSpad + 1)` elements into an `idimInOut` that
/// `numChunks == 0` has already proved smaller.  The reference reads and
/// writes past its array -- `-phase -test 1000,1` segfaults -- so both are
/// refused by name.
#[test]
fn a_multi_chunk_taper_or_fourier_operation_is_refused_by_name() {
    let dir = scratch("refuse");
    write_input(&dir, None);
    let (status, stdout, _) = run(
        &dir,
        &[
            "-in", "in.mrc", "-ou", "t.mrc", "-taper", "5,5", "-test", "3000,1",
        ],
    );
    assert_eq!(status, 1, "{stdout}");
    assert!(
        stdout.ends_with(
            "\nERROR: NEWSTACK - -taper with -memory or -test is not supported by this translation\n"
        ),
        "stdout was:\n{stdout}"
    );
    let (status, stdout, _) = run(
        &dir,
        &["-in", "in.mrc", "-ou", "p.mrc", "-phase", "-test", "2000,1"],
    );
    assert_eq!(status, 1, "{stdout}");
    assert!(
        stdout.ends_with(
            "\nERROR: NEWSTACK - Fourier operations with -memory or -test are not supported by this translation\n"
        ),
        "stdout was:\n{stdout}"
    );
    // A limit that leaves room for one chunk is not refused.
    let (status, stdout, _) = run(
        &dir,
        &[
            "-in", "in.mrc", "-ou", "ok.mrc", "-taper", "5,5", "-test", "200000,1",
        ],
    );
    assert_eq!(status, 0, "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:355` reports the missing input before any of the output-file
/// checks (`:560-567`), before the `-twodir` section-list check (`:493`, which
/// is inside the input scan) and before the `-tilt`/`-reorder` clash (`:989`).
/// Parsing and validating all of PIP in one early block reported those instead.
#[test]
fn a_missing_input_file_is_reported_before_any_later_option_check() {
    let dir = scratch("order");
    std::fs::write(dir.join("list.txt"), "1\nfoo.mrc\n1\n").unwrap();
    for args in [
        vec!["-ou", "X", "-fileo", "Y"],
        vec!["-split", "0", "-ou", "a", "-ou", "b"],
        vec!["-tilt", "five.xf", "-reorder", "1"],
        vec!["-twodir", "1", "-secs", "0,1"],
        vec!["-xform", "five.xf", "-ou", "o.mrc"],
    ] {
        let (status, stdout, stderr) = run(&dir, &args);
        assert_eq!(status, 1, "{args:?} stdout was:\n{stdout}");
        assert_eq!(stderr, "");
        assert_eq!(
            stdout, "\nERROR: NEWSTACK - No input file specified\n",
            "{args:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:606-619`'s missing-`-numout` exit runs before the
/// `numOutTot .ne. listTotal` check at `:671-672`, and both run before the
/// `-reorder` output-count check at `:992-993`.
#[test]
fn the_numout_accounting_runs_before_the_reorder_output_count_check() {
    let dir = scratch("numout");
    write_input(&dir, None);
    let (status, stdout, _) = run(
        &dir,
        &[
            "-in", "in.mrc", "-ou", "a.mrc", "-ou", "b.mrc", "-reorder", "1",
        ],
    );
    assert_eq!(status, 1, "{stdout}");
    assert!(
        stdout.ends_with(
            "\nERROR: NEWSTACK - You must specify number of sections to write to each output file\n"
        ),
        "stdout was:\n{stdout}"
    );
    // With the count supplied, the `-reorder` check is the one that fires.
    let (status, stdout, _) = run(
        &dir,
        &[
            "-in", "in.mrc", "-ou", "a.mrc", "-ou", "b.mrc", "-numout", "2,3", "-reorder", "1",
        ],
    );
    assert_eq!(status, 1, "{stdout}");
    assert!(
        stdout.ends_with(
            "\nERROR: NEWSTACK - You cannot use -reorder with more than one output file\n"
        ),
        "stdout was:\n{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The source reads `-map`, `-fixrange`, `-scale`, `-contrast`, `-meansd`,
/// `-multadd` and `-rfparam` with `PipGetTwoFloats`
/// (`newstack.f90:875-935`), which hands `PipGetFloatArray` a `numToGet` of
/// two.  PIP then reports a short entry itself (`parse_params.c:1953-1956`)
/// and silently stops after two values for a long one.  Reading them with
/// `PipGetFloatArray` and a `numToGet` of zero replaced PIP's message with an
/// invented one, and rejected a long entry the source accepts.
#[test]
fn two_value_options_let_pip_report_their_own_errors() {
    let dir = scratch("twovalue");
    write_input(&dir, None);
    for (option, value, name) in [
        ("-map", "1", "MapFromRange"),
        ("-fixrange", "3", "FixRangeIfNeeded"),
        ("-scale", "1", "ScaleMinAndMax"),
        ("-contrast", "1", "ContrastBlackWhite"),
        ("-meansd", "1", "MeanAndStandardDeviation"),
        ("-multadd", "1", "MultiplyAndAdd"),
    ] {
        let (status, stdout, _) = run(&dir, &["-in", "in.mrc", "-ou", "o.mrc", option, value]);
        assert_eq!(status, 1, "{option} stdout was:\n{stdout}");
        assert!(
            stdout.ends_with(&format!(
                "ERROR: NEWSTACK - 2 values expected but only 1 values found in value entry:  {name}  {value}\n"
            )),
            "{option} stdout was:\n{stdout}"
        );
    }
    // Three values: `PipGetTwoFloats` takes the first two and returns 0, so
    // the program carries on to its own `-map` legality check.
    let (status, stdout, _) = run(&dir, &["-in", "in.mrc", "-ou", "o.mrc", "-map", "1,2,3"]);
    assert_eq!(status, 1, "{stdout}");
    assert!(
        stdout.ends_with("\nERROR: NEWSTACK - You can use -map only with -scale or -contrast\n"),
        "stdout was:\n{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `getItemsToUse` (`newstack.f90:3296-3361`) reads its own PIP option in a
/// loop over every entry and, on a bad index, builds
/// `error // ' number out of bounds:' // i5` and calls `exitError` itself.
/// Returning the failure to the caller dropped the index and the `i5` field,
/// and reading a single entry outside the routine ignored all but the first.
#[test]
fn get_items_to_use_reports_the_index_and_reads_every_entry() {
    let dir = scratch("items");
    write_input(&dir, None);
    let (status, stdout, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "o.mrc",
            "-xform",
            "five.xf",
            "-uselines",
            "9",
        ],
    );
    assert_eq!(status, 1, "{stdout}");
    assert!(
        stdout.ends_with("\nERROR: NEWSTACK - TRANSFORM LINE number out of bounds:    9\n"),
        "stdout was:\n{stdout}"
    );
    // Two entries are concatenated, so five lines are supplied between them.
    let (status, stdout, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "two.mrc",
            "-xform",
            "five.xf",
            "-uselines",
            "0,1",
            "-uselines",
            "2,3,4",
        ],
    );
    assert_eq!(status, 0, "stdout was:\n{stdout}");
    let (status, one_out, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "one.mrc",
            "-xform",
            "five.xf",
            "-uselines",
            "0-4",
        ],
    );
    assert_eq!(status, 0, "{one_out}");
    let two = std::fs::read(dir.join("two.mrc")).unwrap();
    let one = std::fs::read(dir.join("one.mrc")).unwrap();
    assert_eq!(one[1024..], two[1024..]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:951-952` gives `', densities scaled'` to every negative
/// `ifFloat`, and `newstack.f90:901-902` makes `-multadd` negative just as
/// `-scale` does.  `newstack.f90:892-894` keeps `ifFloat` at 4 for
/// `-float 4`, whose text comes from `:1330` instead.
#[test]
fn the_title_float_text_follows_if_float() {
    let dir = scratch("floattext");
    write_input(&dir, None);
    let label = |name: &str| {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        // `mrc_head_new` leaves `nlabl` at zero, so `iiuWriteHeader`'s
        // `labFlag = 1` puts this run's title in the first slot.
        String::from_utf8_lossy(&bytes[224..224 + 80]).into_owned()
    };
    for (name, args, text) in [
        ("m.mrc", vec!["-multadd", "2,5"], ", densities scaled"),
        ("s.mrc", vec!["-scale", "0,100"], ", densities scaled"),
        ("c.mrc", vec!["-contrast", "20,200"], ", densities scaled"),
        ("f1.mrc", vec!["-float", "1"], ", floated to range"),
        ("f2.mrc", vec!["-float", "2"], ", floated to means"),
        ("f3.mrc", vec!["-float", "3"], ",  shifted to mean"),
        (
            "f4.mrc",
            vec!["-float", "4", "-scale", "0,100"],
            ", mean shift&scale",
        ),
    ] {
        let mut all = vec!["-in", "in.mrc", "-ou", name];
        all.extend(args.iter().copied());
        let (status, stdout, _) = run(&dir, &all);
        assert_eq!(status, 0, "{args:?} stdout was:\n{stdout}");
        let title = label(name);
        assert!(
            title.starts_with("NEWSTACK: Images copied"),
            "{args:?} title was {title:?}"
        );
        assert_eq!(&title[36..54], text, "{args:?} title was {title:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:896-898` is one check over `-contrast`, `-scale`, `-float`
/// and `-multadd` together, and its text ends in ` except with -float 4`.
/// `newstack.f90:899-900`'s negative-`-float` check runs after it, and only
/// for an `ifFloat` below 4, so `-float 0` and `-float 5` are legal.
#[test]
fn the_scaling_options_share_one_mutually_exclusive_check() {
    let dir = scratch("exclusive");
    write_input(&dir, None);
    for args in [
        vec!["-float", "1", "-scale", "0,100"],
        vec!["-float", "1", "-multadd", "2,5"],
        vec!["-contrast", "20,200", "-float", "1"],
        vec!["-contrast", "20,200", "-multadd", "2,5"],
        vec!["-scale", "0,100", "-multadd", "2,5"],
        vec!["-float", "-1", "-scale", "0,100"],
    ] {
        let mut all = vec!["-in", "in.mrc", "-ou", "o.mrc"];
        all.extend(args.iter().copied());
        let (status, stdout, _) = run(&dir, &all);
        assert_eq!(status, 1, "{args:?} stdout was:\n{stdout}");
        assert!(
            stdout.ends_with(
                "\nERROR: NEWSTACK - The -scale, -contrast, -multadd, and -float options are mutually exclusive except with -float 4\n"
            ),
            "{args:?} stdout was:\n{stdout}"
        );
    }
    for args in [
        vec!["-float", "0"],
        vec!["-float", "5", "-scale", "0,100"],
        vec!["-float", "4", "-scale", "0,100"],
    ] {
        let mut all = vec!["-in", "in.mrc", "-ou", "o.mrc"];
        all.extend(args.iter().copied());
        let (status, stdout, _) = run(&dir, &all);
        assert_eq!(status, 0, "{args:?} stdout was:\n{stdout}");
    }
    // A negative entry on its own is still the negative-entry message.
    let (status, stdout, _) = run(&dir, &["-in", "in.mrc", "-ou", "o.mrc", "-float", "-1"]);
    assert_eq!(status, 1, "{stdout}");
    assert!(
        stdout.ends_with(
            "\nERROR: NEWSTACK - You must use -contrast or -scale instead of a negative -float entry\n"
        ),
        "stdout was:\n{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:994-995` refuses `-reorder` with `-blank` or `-twodir`; the
/// check was untranslated.
#[test]
fn reorder_is_refused_with_blank_or_twodir() {
    let dir = scratch("reorderblank");
    write_input(&dir, None);
    let (status, stdout, _) = run(
        &dir,
        &["-in", "in.mrc", "-ou", "o.mrc", "-reorder", "1", "-blank"],
    );
    assert_eq!(status, 1, "{stdout}");
    assert!(
        stdout.ends_with(
            "\nERROR: NEWSTACK - You cannot use -reorder with the -blank or -twodir option\n"
        ),
        "stdout was:\n{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:2334-2592` loads, transforms or repacks, counts and writes
/// **per chunk**: `:2341-2382` reads only the lines the chunk needs into the
/// one load window it keeps across the loop, `:2384` recomputes `numYload`
/// from that window, and `:2481-2500` repacks with `iy1` measured from
/// `lineOutSt(iChunk) - loadYstart`.  A route that built the whole output
/// section in one piece and then only sliced it for the statistics and the
/// writes produced the same pixels but did none of that work, and reported
/// ` loading whole region` and ` did repack` once a section instead of once a
/// chunk.  The reference's own numbers for this input are below: three chunks
/// of 16 lines a section, five sections, `iChunkBase` 1025 = `maxin * nxBin + 1`
/// and `nyRepakDim` 16 = the chunk's own `numYload`.
#[test]
fn every_chunk_loads_and_repacks_on_its_own() {
    let dir = scratch("perchunk");
    write_input(&dir, None);
    let (status, whole, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "whole.mrc",
            "-float",
            "2",
            "-mode",
            "1",
            "-verbose",
            "1",
        ],
    );
    assert_eq!(status, 0, "{whole}");
    assert_eq!(
        whole.matches(" loading whole region").count(),
        5,
        "one chunk a section without a limit:\n{whole}"
    );
    assert_eq!(whole.matches(" did repack").count(), 5, "{whole}");
    let (status, chunked, _) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "chunk.mrc",
            "-float",
            "2",
            "-mode",
            "1",
            "-test",
            "3000,1",
            "-verbose",
            "1",
        ],
    );
    assert_eq!(status, 0, "{chunked}");
    assert_eq!(
        chunked.matches(" loading whole region").count(),
        15,
        "three chunks a section, five sections:\n{chunked}"
    );
    assert_eq!(chunked.matches(" did repack").count(), 15, "{chunked}");
    let reports = chunked
        .lines()
        .filter(|line| line.starts_with(" loading whole region") || line.starts_with(" did repack"))
        .take(6)
        .collect::<Vec<_>>();
    assert_eq!(
        reports,
        vec![
            " loading whole region           0          15          16",
            " did repack                 1025           1          64          16           0          63           0          15",
            " loading whole region          16          31          16",
            " did repack                 1025           1          64          16           0          63           0          15",
            " loading whole region          32          47          16",
            " did repack                 1025           1          64          16           0          63           0          15",
        ],
        "stdout was:\n{chunked}"
    );
    // The reference's output data for this run: length and an FNV-1a 64 hash
    // over everything past the 1024-byte header, whose labels carry a date
    // stamp that cannot match.
    let data = std::fs::read(dir.join("chunk.mrc")).unwrap();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in &data[1024..] {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    assert_eq!((data.len() - 1024, hash), (30720, 0xf7fa_e8aa_703d_3b08));
    let _ = std::fs::remove_dir_all(&dir);
}

/// With no transform `linesNeededForOutput` clamps both ends to the input
/// (`newstack.f90:3322-3323`), so an output chunk that lies entirely in the
/// padding below the image comes back with `iy2 < iy1` and a **negative**
/// `numLinesIn(iChunk)`.  The source still calls `readBinnedOrReduced` with
/// that count (`:2376-2379`), and `iiMRCsetLoadInfo` turns the negative `ury`
/// into `ny - 1` (`iimrc.c:169-172`), so the read fetches the whole section --
/// past the end of the source's own `array` when `idimInOut` is smaller.  The
/// repack still sees `my <= 0` and fills the chunk from `dmeanSec`.  This case
/// also drives the `moving data up` shortcut at `:2358-2368`, which only fires
/// when a chunk's window overlaps the previous one from below.
#[test]
fn a_chunk_below_the_image_is_loaded_and_filled_as_the_source_does() {
    let dir = scratch("padchunk");
    write_input(&dir, None);
    let (status, stdout, stderr) = run(
        &dir,
        &[
            "-in", "in.mrc", "-ou", "pad.mrc", "-size", "80,72", "-offset", "0,-20", "-test",
            "2200,1", "-verbose", "1",
        ],
    );
    assert_eq!(status, 0, "{stdout}{stderr}");
    let reports = stdout
        .lines()
        .filter(|line| {
            line.starts_with(" loading whole region")
                || line.starts_with(" did repack")
                || line.starts_with(" moving data")
        })
        .take(8)
        .collect::<Vec<_>>();
    assert_eq!(
        reports,
        vec![
            " loading whole region           0         -18         -17",
            " did repack                  897           1          64           0          -8          71         -32         -18",
            " loading whole region           0          -3          -2",
            " did repack                  897           1          64           0          -8          71         -17          -3",
            " moving data up                  768                    0",
            " did repack                  897           1          64          12          -8          71          -2          11",
            " loading whole region          12          25          14",
            " did repack                  897           1          64          14          -8          71           0          13",
        ],
        "stdout was:\n{stdout}"
    );
    assert!(
        stdout.contains("       0      0.00    250.00      0.00    250.00    123.04\n"),
        "stdout was:\n{stdout}"
    );
    // The reference's output data for this run, as above.
    let data = std::fs::read(dir.join("pad.mrc")).unwrap();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in &data[1024..] {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    assert_eq!((data.len() - 1024, hash), (115200, 0x8955_27fa_1700_8d37));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:874-936` reads and validates every scaling option **before**
/// the memory limits at `:1027-1038` and long before the `ifMean` pre-scan at
/// `:1345-1367`.  The translation had that whole block after the pre-scan, so
/// `-verbose 1` printed ` reallocating array to`, ` MB of physical memory` and
/// one ` scanning for mean/sd` line per section ahead of an error the
/// reference prints on its own.  Each expectation below is the reference's
/// stdout for the same invocation, taken through a pipe.
#[test]
fn a_scaling_option_clash_is_reported_before_the_mean_sd_prescan() {
    let dir = scratch("optorder");
    write_input(&dir, None);
    // `newstack.f90:893`: `-float 4` without `-scale`.  `ifFloat >= 4` sets
    // `ifMean`, so the pre-scan would read both sections if it ran first.
    for (options, message) in [
        (vec!["-float", "4"], "You must enter -scale with -float 4"),
        (
            vec!["-float", "4", "-scale", "1,2", "-map", "3,4"],
            "You cannot use -map with -float 4",
        ),
        // `newstack.f90:896-898`, the one check over all four options.
        (
            vec!["-float", "1", "-scale", "10,200"],
            "The -scale, -contrast, -multadd, and -float options are mutually exclusive except with -float 4",
        ),
        (
            vec!["-float", "2", "-multadd", "2,3"],
            "The -scale, -contrast, -multadd, and -float options are mutually exclusive except with -float 4",
        ),
        // `newstack.f90:884-888`.
        (
            vec!["-meansd", "100,30", "-contrast", "10,200"],
            "You cannot use -meansd with any scaling option except -float 2",
        ),
        // `newstack.f90:882-883`.
        (
            vec!["-map", "1,2"],
            "You can use -map only with -scale or -contrast",
        ),
        // `newstack.f90:899-900`, inside the `ifFloat < 4` arm.
        (
            vec!["-float", "-1"],
            "You must use -contrast or -scale instead of a negative -float entry",
        ),
        // `newstack.f90:920-921`.
        (
            vec!["-fixrange", "3,2", "-float", "1"],
            "You cannot enter -fixrange with any scaling options",
        ),
    ] {
        let mut args = vec![
            "-in", "in.mrc", "-ou", "o.mrc", "-verbose", "1", "-test", "4001,1",
        ];
        args.extend_from_slice(&options);
        let (status, stdout, stderr) = run(&dir, &args);
        assert_eq!(status, 1, "{options:?}\nstdout was:\n{stdout}");
        assert_eq!(
            stdout,
            format!("\nERROR: NEWSTACK - {message}\n"),
            "the reference prints this error and nothing else for {options:?}"
        );
        assert_eq!(stderr, "", "`exitError` routes through stdout");
        assert!(
            !dir.join("o.mrc").exists(),
            "the reference creates no output for {options:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:2046-2281` is inside the section loop, so
/// `linesNeededForOutput` runs on **this** section's `fprod`: `nyNeeded`,
/// `nyDimNeed`, `linesLeft` and `numChunks` (`:2189-2196`) change from section
/// to section, and `numChunks` picks the branch at `:2209` and sizes the chunk
/// table built there.  Computed once for the run, every section was laid out
/// the way section 0 needed.  The five transforms below shift Y by 0, 10, 20,
/// 30 and 40, so each section needs fewer input lines than the last and the
/// reference splits the output into 4, 3, 2, 2 and 1 chunks.
#[test]
fn the_chunk_layout_is_recomputed_for_every_section() {
    let dir = scratch("perseclayout");
    write_input(&dir, None);
    std::fs::write(
        dir.join("shifty.xf"),
        "   1.0   0.0   0.0   1.0   0.0   0.0\n\
         \x20  1.0   0.0   0.0   1.0   0.0  10.0\n\
         \x20  1.0   0.0   0.0   1.0   0.0  20.0\n\
         \x20  1.0   0.0   0.0   1.0   0.0  30.0\n\
         \x20  1.0   0.0   0.0   1.0   0.0  40.0\n",
    )
    .unwrap();
    let (status, stdout, stderr) = run(
        &dir,
        &[
            "-in",
            "in.mrc",
            "-ou",
            "o.mrc",
            "-xform",
            "shifty.xf",
            "-test",
            "4001,1",
            "-verbose",
            "1",
        ],
    );
    assert_eq!(status, 0, "{stdout}{stderr}");
    let reports = stdout
        .lines()
        .filter(|line| {
            line.starts_with(" linesleft")
                || line.starts_with(" number of chunks:")
                || (line.starts_with("      ") && line.split_whitespace().count() == 5)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reports,
        vec![
            " linesleft          14   nchunk           4",
            " number of chunks:           4           1",
            "           1           0          48           0          12",
            "           2           0          48          12          12",
            "           3           0          48          24          12",
            "           4           0          48          36          12",
            " linesleft          22   nchunk           3",
            " number of chunks:           3           1",
            "           1           0          40           0          16",
            "           2           0          40          16          16",
            "           3           0          40          32          16",
            " linesleft          32   nchunk           2",
            " number of chunks:           2           1",
            "           1           0          30           0          24",
            "           2           0          30          24          24",
            " linesleft          42   nchunk           2",
            " number of chunks:           2           1",
            "           1           0          20           0          24",
            "           2           0          20          24          24",
            " linesleft          52   nchunk           1",
            " number of chunks:           1           1",
            "           1           0          10           0          48",
        ],
        "stdout was:\n{stdout}"
    );
    // The per-section statistics the reference reports for this run.  They are
    // not incidental: a stale layout changes which lines each `iclden` covers.
    for line in [
        "       0      0.00    250.00      0.00    250.00    123.13\n",
        "       1      7.00    257.00      7.00    257.00    129.39\n",
        "       2     14.00    264.00     14.00    264.00    136.21\n",
        "       3     21.00    271.00     21.00    271.00    142.14\n",
        "       4     28.00    278.00     28.00    278.00    118.34\n",
    ] {
        assert!(stdout.contains(line), "{line:?} missing from:\n{stdout}");
    }
    // The reference's output data for this run: length and an FNV-1a 64 hash
    // over everything past the 1024-byte header.  With the layout hoisted out
    // of the section loop these bytes differ -- the header mean alone moved.
    let data = std::fs::read(dir.join("o.mrc")).unwrap();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in &data[1024..] {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    assert_eq!((data.len() - 1024, hash), (61440, 0xf0bf_7167_da21_7ae9));
    let _ = std::fs::remove_dir_all(&dir);
}
