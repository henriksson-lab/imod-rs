//! Native-golden coverage for `findsection` (`IMOD/imodutil/findsection.cpp`).
//!
//! Every case in `fixtures/findsection/cases.tsv` was run through the native
//! reference `findsection` by `fixtures/make-findsection-goldens.sh`, stdout
//! captured through a pipe.  The exit status is `golden/<case>.rc`, stdout is
//! `golden/<case>.out`, and every file native left behind is in
//! `golden/<case>/`.  The inputs are seeded synthetic slabs written by the
//! native `raw2mrc`, a bead model made by the native `findsection` and
//! `imodtrans -i`, and `fixtures/imodtrans/multi.mod` as a model with no
//! `IrefImage`.
//!
//! Pruned 2026-09-26: 17 of 18 rows kept (dropped `basic`, the same run as `nonopt` with -tomo instead of a positional argument); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).  `flipped` runs on the flipped slab `m0.mrc` rather than a separate 139 KB `fy.mrc`, and the two multi-tomogram rows give `m0.mrc` twice instead of a second 93 KB slab `m1.mrc`.

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/findsection")
}

/// Blank the wall-clock stamps (`common::mask_stamps`).  The regions native
/// writes from uninitialised memory (`BUGS.md` §2) are reconciled first by
/// `common::reconcile_uninitialised`, which checks ours holds the defined value.
fn mask(bytes: &[u8]) -> Vec<u8> {
    common::mask_stamps(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-findsection-{}-{}",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(fixture_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let file = path.file_name().unwrap().to_str().unwrap().to_string();
            if file.ends_with(".mrc") || file.ends_with(".mod") {
                std::fs::copy(&path, dir.join(&file)).unwrap();
            }
        }
    }
    dir
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let (name, args) = line.split_once('\t').unwrap();
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        let expected_out = common::golden::expect(&golden.join(format!("{name}.out")));
        let dir = scratch(name);
        let inputs: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .collect();
        let output = common::imod_cmd("findsection")
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
        // The usage listing starts with a version banner carrying the build
        // date (`imodVersion`); only the lines after it are compared.
        let skip = |b: &[u8]| -> Vec<u8> {
            if b.starts_with(b"findsection Version") {
                b.iter()
                    .position(|&c| c == b'\n')
                    .map_or(Vec::new(), |p| b[p + 1..].to_vec())
            } else {
                b.to_vec()
            }
        };
        if !expected_out.matches_masked(&output.stdout, skip) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                expected_out.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected_files: Vec<String> = common::golden::list(&golden.join(name));
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
            let theirs = common::golden::expect(&golden.join(name).join(file));
            if let Err(why) = theirs.compare(&ours, mask, true) {
                failures.push(format!("{name}: {file} differs from native: {why}"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 17, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// BUGS.md, fixed in translation: `findsection.cpp:411` passes the NULL model
/// pointer to `%s` (native prints "(null)"); the message names the file.
/// (The binning-column and low-SD-message fixes are in the `binning` and
/// `lowest` goldens.)
#[test]
fn unreadable_bead_model_message_names_the_file() {
    let dir = scratch("nobead");
    let output = common::imod_cmd("findsection")
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(["-tomo", "fz.mrc", "-size", "8,8,2", "-high", "3"])
        .args(["-bead", "nosuch.mod", "-diameter", "6"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("Reading in bead model file nosuch.mod"),
        "{text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
