//! Native-golden coverage for `blendmont` (`IMOD/flib/blend/blendmont.f90`).
//!
//! Every case in `fixtures/blendmont/cases.tsv` was run through the native
//! reference `blendmont` at `OMP_NUM_THREADS=1` by
//! `fixtures/make-blendmont-goldens.sh`, stdout captured through a pipe.  The
//! exit status is `golden/<case>.rc`, stdout is `golden/<case>.out`, and every
//! file native left behind is in `golden/<case>/`.  The input
//! (`make-blendmont-inputs.py`) is a seeded synthetic 3 x 3 montage of two
//! sections written by the native `raw2mrc`; the `old.*` edge files the reuse
//! cases read were made by the native program too.  The cases cover plain
//! blending, sloppy and piece-shifting modes (edges, correlations, robust
//! fitting, alternative peaks), output modes, floating, binning, windows and
//! multiple output frames, section lists, aligned coordinates, test mode, edge
//! functions only, intensity scaling with and without gradients (including
//! `-sum`, whose `clip plane` runs in process), mag gradients, g transforms,
//! an exclusion model, multiple negatives, old edge functions, read-in and
//! expected correlations, parallel setup and header writing, and error exits.
//!
//! Where a fix of an upstream defect applies (`BUGS.md`, blendmont), the
//! golden holds the defined behaviour instead, taken from this translation
//! (2026-09-26); every other file is native's.  The edge grids of this
//! montage are one position across the overlap (20 pixels at the default
//! grid spacing), so the source's interpolation reads a second row or
//! column `readEdgeFunc` never wrote for that edge (stale values from the
//! buffer's previous edge): `o.st` of basic, sloppy, shift, xcorr, edge,
//! robust, numpeaks, mode0f, mode2, bin2, window, nofft, frames, sections,
//! aligned, int1, int2, sum, gradient, xform, skip, oldedge, readxcorr,
//! oldint and expected is the fixed interpolation's.  `negfile` and `perneg`
//! also carry the `findMultinegTransforms` fixes, and `perneg` the
//! `-MissingFromFirstNegativeXandY` fix (so its stdout numbers the
//! negatives 1-4 where native, taking 2 missing, numbers them 5-9).  Every
//! other case, and every other file of these cases, is native's.

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/blendmont")
}

/// Blank the wall-clock stamps (`common::mask_stamps`).  The regions native
/// writes from uninitialised memory (`BUGS.md` §2) are reconciled first by
/// `common::reconcile_uninitialised`, which checks ours holds the defined value.
fn mask(bytes: &[u8]) -> Vec<u8> {
    common::mask_stamps(bytes)
}

fn scratch(name: &str) -> (PathBuf, Vec<String>) {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-blendmont-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut inputs = Vec::new();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file == "cases.tsv" || file == "excl.txt" {
                continue;
            }
            std::fs::copy(&path, dir.join(&file)).unwrap();
            inputs.push(file);
        }
    }
    (dir, inputs)
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let (name, args) = line.split_once('\t').unwrap();
        let rc: i32 = std::fs::read_to_string(golden.join(format!("{name}.rc")))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let expected_out = std::fs::read(golden.join(format!("{name}.out"))).unwrap();
        let (dir, inputs) = scratch(name);
        let output = common::imod_cmd("blendmont")
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .env("OMP_NUM_THREADS", "1")
            .args(args.split_whitespace())
            .output()
            .unwrap();
        count += 1;
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        // Compared as a multiset of lines: the lines `dopen` and the image
        // open print go to a stream of their own here, so their order against
        // the program's C-stdio lines can differ from native's (messages need
        // not interleave the same way; the lines themselves must all be there).
        let lines = |b: &[u8]| -> Vec<String> {
            let mut v: Vec<String> = String::from_utf8_lossy(b)
                .lines()
                .map(str::to_owned)
                .collect();
            v.sort();
            v
        };
        let (ours, theirs) = (lines(&output.stdout), lines(&expected_out));
        if ours != theirs {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&expected_out),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let mut expected_files: Vec<String> = std::fs::read_dir(golden.join(name))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .collect();
        expected_files.sort();
        let mut produced: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .filter(|f| !inputs.contains(f))
            .collect();
        produced.sort();
        if produced != expected_files {
            failures.push(format!(
                "{name}: files {produced:?}, native {expected_files:?}"
            ));
        }
        for file in &expected_files {
            let Ok(ours) = std::fs::read(dir.join(file)) else {
                continue;
            };
            let theirs = std::fs::read(golden.join(name).join(file)).unwrap();
            if mask(&ours) != mask(&common::reconcile_uninitialised(&ours, &theirs)) {
                failures.push(format!("{name}: {file} differs from native"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 38, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Run `blendmont` in `dir` with `args` (and `stdin`, if any).
fn run_in(dir: &Path, args: &[&str], stdin: Option<&[u8]>) -> std::process::Output {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = common::imod_cmd("blendmont")
        .current_dir(dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .env("OMP_NUM_THREADS", "1")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut pipe = child.stdin.take().unwrap();
        if let Some(bytes) = stdin {
            pipe.write_all(bytes).unwrap();
        }
    }
    child.wait_with_output().unwrap()
}

/// The output image with the header's time stamps blanked.
fn image(dir: &Path, name: &str) -> Vec<u8> {
    mask(&std::fs::read(dir.join(name)).unwrap())
}

const BASE: [&str; 6] = ["-imin", "bm.st", "-plin", "bm.pl", "-imout", "o.st"];

/// `-MissingFromFirstNegativeXandY` is read (`blendmont.f90:638` reads
/// `FramesPerNegativeXandY` twice, so natively the missing counts equal the
/// frames per negative; fixed in translation, `BUGS.md`).  Unentered, the
/// counts are 0, and an entered count changes the negative assignment.
#[test]
fn missing_from_first_negative_is_read() {
    let (dir, _) = scratch("missing");
    let mut outs = Vec::new();
    for (k, extra) in [&[][..], &["-missing", "0,0"], &["-missing", "1,1"]]
        .iter()
        .enumerate()
    {
        let root = format!("r{k}");
        let mut args: Vec<&str> = BASE.to_vec();
        args.extend(["-rootname", &root, "-perneg", "2,2"]);
        args.extend(extra.iter());
        let out = run_in(&dir, &args, None);
        assert_eq!(out.status.code(), Some(0), "{extra:?}");
        outs.push(image(&dir, "o.st"));
    }
    assert!(outs[0] == outs[1], "-missing 0,0 is not the default");
    assert!(outs[0] != outs[2], "-missing 1,1 had no effect");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `-ExpectedShiftsFromEcd` on a montage that is not 3 pieces wide reads
/// the Y edges from the `.ecd` file too.  Natively the Y unit is the stale
/// `nxPieces + 1` (unit 3 here: `fort.3`, End of file, status 2); fixed in
/// translation (`BUGS.md`).  The 2 x 2 montage is pieces 0, 1, 3, 4 of the
/// fixture.
#[test]
fn expected_shifts_on_two_wide_montage() {
    let (dir, _) = scratch("expected22");
    let out = common::imod_cmd("newstack")
        .current_dir(&dir)
        .args(["-secs", "0,1,3,4", "bm.st", "m22.st"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    std::fs::write(dir.join("m22.pl"), "0 0 0\n44 0 0\n0 36 0\n44 36 0\n").unwrap();
    let m22 = ["-imin", "m22.st", "-plin", "m22.pl", "-imout", "o.st"];
    let mut args: Vec<&str> = m22.to_vec();
    args.extend(["-rootname", "e", "-sloppy"]);
    assert_eq!(run_in(&dir, &args, None).status.code(), Some(0));
    let mut args: Vec<&str> = m22.to_vec();
    args.extend(["-rootname", "f", "-sloppy", "-expected", "e.ecd"]);
    let out = run_in(&dir, &args, None);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!dir.join("fort.3").exists());
    assert!(dir.join("o.st").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Interactive input with read-in correlations: standard input stays open
/// for the blending-width prompt after the `.ecd` file is read.  Natively
/// `close(5)` closes it and the read ends in End of file on `fort.5`
/// (status 2); fixed in translation (`BUGS.md`).
#[test]
fn interactive_read_in_correlations_keeps_stdin() {
    let (dir, _) = scratch("inter_readx");
    std::fs::copy(dir.join("old.ecd"), dir.join("e.ecd")).unwrap();
    let out = run_in(
        &dir,
        &[],
        Some(b"bm.st\no.st\n/\n0\n\nbm.pl\n4\n\n/\n/\n/\n0\n0\ne\n/\n"),
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!dir.join("fort.5").exists());
    assert!(dir.join("o.st").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// An edge excluded with `-skip` blends with a zero edge function when the
/// functions come from an old edge file, as it does when they are computed
/// in the same run: `readEdgeFunc` zeroes the buffer it read (natively it
/// zeroes the scratch grids and blends with the old function; fixed in
/// translation, `BUGS.md`).
#[test]
fn skipped_edge_from_old_file_blends_with_zero_function() {
    let (dir, _) = scratch("skip_old");
    let mut args: Vec<&str> = BASE.to_vec();
    args.extend(["-rootname", "e"]);
    assert_eq!(run_in(&dir, &args, None).status.code(), Some(0));
    let mut args: Vec<&str> = BASE.to_vec();
    args.extend(["-rootname", "f", "-skip", "excl.mod"]);
    assert_eq!(run_in(&dir, &args, None).status.code(), Some(0));
    let fresh = image(&dir, "o.st");
    let mut args: Vec<&str> = BASE.to_vec();
    args.extend(["-rootname", "e", "-oldedge", "-skip", "excl.mod"]);
    assert_eq!(run_in(&dir, &args, None).status.code(), Some(0));
    let old = image(&dir, "o.st");
    assert!(old == fresh);
    // And the exclusion does change the blend (natively, with old edge
    // files, it does not).
    let mut args: Vec<&str> = BASE.to_vec();
    args.extend(["-rootname", "e", "-oldedge"]);
    assert_eq!(run_in(&dir, &args, None).status.code(), Some(0));
    assert!(image(&dir, "o.st") != old);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Old edge function and edge density files of the other byte order give
/// the same result as the native-order files: the density files are
/// swapped along with the edge functions (natively only the edge-function
/// counts are; fixed in translation, `BUGS.md`).
#[test]
fn byte_swapped_old_edge_and_density_files() {
    let (dir, _) = scratch("swapped");
    let mut args: Vec<&str> = BASE.to_vec();
    args.extend(["-rootname", "old", "-intensity", "2", "-oldedge", "-sloppy"]);
    assert_eq!(run_in(&dir, &args, None).status.code(), Some(0));
    let native_order = image(&dir, "o.st");
    for ext in ["xef", "yef", "xaed", "yaed"] {
        let path = dir.join(format!("old.{ext}"));
        let mut bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() % 4, 0);
        for word in bytes.chunks_mut(4) {
            word.reverse();
        }
        std::fs::write(&path, bytes).unwrap();
    }
    let out = run_in(&dir, &args, None);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(image(&dir, "o.st") == native_order);
    let _ = std::fs::remove_dir_all(&dir);
}
