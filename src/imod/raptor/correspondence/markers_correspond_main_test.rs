//! Translation of `IMOD/raptor/correspondence/markersCorrespondMainTest.cpp`,
//! the `MarkersCorrespond` program: reads the markers of one projection,
//! the candidates in the next and the singleton potentials RAPTOR wrote
//! under a common test name, runs the MRF inference of
//! [`SvlMarkerCorrespondenceLbModel`], and writes the marginal log-beliefs
//! to `<test name>_final_beliefs_scr.m`.
//!
//! Everything it prints goes through `cxx_stream::cout`/`cerr`, so that
//! RAPTOR can send both to its `_temp.txt` file when it runs this in
//! process, as its `popen("... > x_temp.txt 2>&1")` did.

use crate::imod::cxx_stream::{IStream, cerr, cout, ostream_double};
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::raptor::correspondence::svl_marker_correspondence_lb_model::{
    GglMatrix, SvlMarkerCorrespondenceLbModel,
};

/// C `atoi(s)`, glibc's `(int) strtol(s, NULL, 10)`: optional white space
/// and sign, then decimal digits; `strtol` saturates at the `long` range and
/// the cast keeps the low 32 bits.
fn atoi(s: &str) -> i32 {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r') {
        i += 1;
    }
    let mut negative = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        negative = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v.saturating_mul(10).saturating_add((b[i] - b'0') as i64);
        i += 1;
    }
    if negative {
        v = v.saturating_neg();
    }
    v as i32
}

/// `main(argc, argv)` (`markersCorrespondMainTest.cpp:28`).  `argv[0]` is the
/// program name.  Returns the exit status; the source's `exit(0)` error
/// paths end through `b3dutil::exit`.
///
/// Two upstream defects are fixed (`BUGS.md`): `-debug_pair_pots` without
/// its two numbers passed a NULL `argv` entry to `atoi` (a crash); a missing
/// number reads as 0 here.  The constructor arguments are passed in the
/// source's (swapped) order, which only matters with that option.
pub fn main(argv: &[String]) -> i32 {
    // INPUTS:
    //  - M = Image1 marker coordinates
    //  - K = Image2 marker candidate coordinates
    //  - SP = Initial Singleton Beliefs for marker i to candidate j
    // OUTPUTS:
    //  - B = M X K matrix of marginal beliefs
    let debug_init_beliefs = 0;
    let mut debug_pair_pots = 0;
    let mut pair_pots_m1 = 1;
    let mut pair_pots_m2 = 1;
    let dist_diff_intcpt = 25.0;
    let verbose = 0;
    let mtime = 1;
    let mut lock_pots_file_exists = 0;
    let usage = "Usage: markersCorrespond <test name> [-debug_init_beliefs pair_pots_m1  pair_pots_m2] \n NOTE: files <test name>.cfg, <test name>.lock_pots, <test name>.mrkr, <test name>.mrkr_trans, <test name>.cand, and <test name>.sp are required to exist\n";

    let argc = argv.len();
    let min_args = 1;
    let max_args = 5;
    if (argc <= min_args) || (argc > max_args) {
        cerr(usage);
        exit(0);
    } else if verbose != 0 {
        cout(&format!("Num args = {argc}\n"));
    }

    let test_name = argv[min_args].clone();
    if argc > (min_args + 1) && argv[min_args + 1] == "-debug_pair_pots" {
        debug_pair_pots = 1;
        pair_pots_m1 = argv.get(min_args + 2).map_or(0, |s| atoi(s));
        pair_pots_m2 = argv.get(min_args + 3).map_or(0, |s| atoi(s));
    }

    let config_filename = format!("{test_name}.cfg");
    let lock_pots_filename = format!("{test_name}.lock_pots");
    let marker_filename = format!("{test_name}.mrkr");
    let marker_filename2 = format!("{test_name}.mrkr_trans");
    let candidate_filename = format!("{test_name}.cand");
    let sp_matrix_filename = format!("{test_name}.sp");
    let c_matrix_filename = format!("{test_name}_final_beliefs_scr.m");

    let mut rows = 2;
    // this is ignored by ReadMatrix
    let cols = 5;

    let Some(mut infile1) = IStream::open(&marker_filename) else {
        cerr(&format!(
            "Marker File: {marker_filename} could NOT be opened..exiting now...\n"
        ));
        exit(0);
    };
    let m = ggl_farshid_read_matrix(rows, cols, &mut infile1, 0);
    if verbose != 0 {
        cout("Read in M\n");
    }

    let Some(mut infile5) = IStream::open(&marker_filename2) else {
        cerr(&format!(
            "Marker File: {marker_filename2} could NOT be opened..exiting now...\n"
        ));
        exit(0);
    };
    let m_trans = ggl_farshid_read_matrix(rows, cols, &mut infile5, 0);
    if verbose != 0 {
        cout("Read in M (translated)\n");
    }

    let Some(mut infile2) = IStream::open(&candidate_filename) else {
        cerr(&format!(
            "Candidate File: {candidate_filename} could NOT be opened..exiting now...\n"
        ));
        exit(0);
    };
    let k = ggl_farshid_read_matrix(rows, cols, &mut infile2, 0);
    if verbose != 0 {
        cout("Read in K\n");
    }

    rows = m.dim2();
    if verbose != 0 {
        cout(&format!("Read in num rows {rows}\n"));
    }
    let Some(mut infile3) = IStream::open(&sp_matrix_filename) else {
        cerr(&format!(
            "SP Matrix File: {sp_matrix_filename} could NOT be opened..exiting now...\n"
        ));
        exit(0);
    };
    let sp = ggl_farshid_read_matrix(rows, cols, &mut infile3, 0);
    if verbose != 0 {
        cout(&format!("Read in SP, num rows is {rows}\n"));
    }

    let Some(mut infile4) = IStream::open(&config_filename) else {
        cerr(&format!(
            "Config File: {config_filename} could NOT be opened..exiting now...\n"
        ));
        exit(0);
    };

    let mut infile6 = IStream::open(&lock_pots_filename);
    if infile6.is_some() {
        if verbose != 0 {
            cerr(&format!(
                "LockPots File: {lock_pots_filename} was found...\n"
            ));
        }
        lock_pots_file_exists = 1;
    }

    // Create Inference Object (arguments in the source's order: its
    // `debug_pair_pots, pair_pots_m1, pair_pots_m2, debug_init_beliefs`
    // land in the constructor's `_debug_init_beliefs, _debug_pair_pots,
    // _pair_pots_m1, _pair_pots_m2`)
    let mut model = SvlMarkerCorrespondenceLbModel::new(
        &m,
        &m_trans,
        &k,
        &sp,
        &test_name,
        lock_pots_file_exists,
        &mut infile4,
        infile6.as_mut(),
        debug_pair_pots,
        pair_pots_m1,
        pair_pots_m2,
        debug_init_beliefs,
        dist_diff_intcpt,
        verbose,
        mtime,
    );

    // Run Inference
    let c = model.get_final_marginal_beliefs();

    if verbose != 0 {
        cout("MATRIX M\n");
        cout("\n");
        cout("MATRIX K\n");
        cout("\n");
    }

    let c_matrix_name = format!("{test_name}_final_beliefs");
    let mut c_outfile = String::new();
    c_outfile.push_str(&format!("{c_matrix_name}=[\n"));
    print_matrix(&c, &mut c_outfile, true);
    c_outfile.push_str("];\n");
    // `ofstream c_outfile(...)`: nothing is written when it cannot be opened.
    let _ = std::fs::write(&c_matrix_filename, c_outfile.as_bytes());

    0
}

/// `gglFarshidReadMatrix(dim1, dim2, in, verbose)`
/// (`markersCorrespondMainTest.cpp:193`): a count `n`, then `dim1` rows of
/// `n` values; `dim2` is ignored.
pub fn ggl_farshid_read_matrix(
    dim1: i32,
    _dim2: i32,
    input: &mut IStream,
    verbose: i32,
) -> GglMatrix {
    let mut nmarkers = 0i32;
    input.read_i32(&mut nmarkers);
    if verbose != 0 {
        cerr(&format!("You have just read in nmarkers = {nmarkers}\n"));
    }
    let mut m = GglMatrix::new(dim1, nmarkers);
    for i in 0..dim1.max(0) as usize {
        for j in 0..nmarkers.max(0) as usize {
            input.read_f64(&mut m[i][j]);
        }
    }

    if verbose != 0 {
        cerr("You have just read in the matrix:\n");
        let mut s = String::new();
        print_matrix(&m, &mut s, true);
        cerr(&s);
    }

    m
}

/// `PrintMatrix(mx, out, delimiters)` (`markersCorrespondMainTest.cpp:213`).
pub fn print_matrix(mx: &GglMatrix, out: &mut String, delimiters: bool) {
    for i in 0..mx.dim1().max(0) as usize {
        for j in 0..mx.dim2().max(0) as usize {
            if j > 0 {
                if delimiters {
                    out.push(',');
                }
                out.push(' ');
            }
            out.push_str(&ostream_double(mx[i][j]));
        }

        if delimiters {
            out.push(';');
        }
        out.push('\n');
    }
}
