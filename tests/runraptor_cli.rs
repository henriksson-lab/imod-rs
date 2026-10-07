//! Native-golden coverage for `runraptor` (`IMOD/pysrc/runraptor`,
//! translated in `src/imod/pysrc/runraptor.rs`).
//!
//! Every row of `fixtures/runraptor/cases.tsv` was run through the native
//! Python script by `fixtures/make-runraptor-goldens.sh`, with the native
//! `header` and `xfmodel` on `PATH` and `RAPTOR_BIN` pointing at the RAPTOR
//! stand-in `fixtures/runraptor/RAPTOR`, which writes what runraptor reads
//! from a RAPTOR run (a real RAPTOR fiducial text model and a log of its
//! arguments).  `golden/<case>.rc` is the exit status, `.out` the standard
//! output, and `golden/<case>/` every file new or changed afterwards, with
//! `/` in its path written as `%`.  Here our `imod` runs runraptor with the
//! same stand-in; `header` and `xfmodel` are ours, run in process.  Cases
//! whose output an upstream-bug fix changes take their expectation from
//! `defined/` (`defined.list`, `BUGS.md`).
//!
//! The runraptor-with-real-RAPTOR differential on the TS_01 stack is
//! recorded in `TODO.md` ("runraptor"); it needs the reference build's
//! RAPTOR and takes half a minute per run, so it is not a gate test.

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/runraptor")
}

/// An `IMOD_DIR` whose `bin` links runraptor and the programs it runs to our
/// binary.
fn install(root: &Path) -> PathBuf {
    let bin = root.join("imod/bin");
    std::fs::create_dir_all(&bin).unwrap();
    for command in ["runraptor", "xfmodel", "header"] {
        #[cfg(unix)]
        let _ = std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_imod"), bin.join(command));
        #[cfg(not(unix))]
        let _ = std::fs::hard_link(
            env!("CARGO_BIN_EXE_imod"),
            bin.join(format!("{command}{}", std::env::consts::EXE_SUFFIX)),
        );
    }
    root.join("imod")
}

/// Every regular file under `dir`, as a path relative to it.
fn walk(dir: &Path, prefix: &str, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let name = format!("{prefix}{}", entry.file_name().to_str().unwrap());
        if entry.file_type().unwrap().is_dir() {
            walk(&entry.path(), &format!("{name}/"), out);
        } else {
            out.push(name);
        }
    }
}

/// `<root>.<digits>` -> `<root>.PID`, as the make script masks it.
fn mask_pid(text: &str) -> String {
    let mut result = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("ts.") {
        result.push_str(&rest[..at + 3]);
        rest = &rest[at + 3..];
        let digits = rest.chars().take_while(char::is_ascii_digit).count();
        if digits > 0 {
            result.push_str("PID");
            rest = &rest[digits..];
        }
    }
    result.push_str(rest);
    result
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let root = std::env::temp_dir().join(format!("imod-rs-runraptor-{}", std::process::id()));
    let imod_dir = install(&root);
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, setup, args) = (fields[0], fields[1], fields[2]);
        let defined = fixture_dir().join("defined");
        let golden = if common::golden::exists(&defined.join(format!("{name}.rc"))) {
            defined
        } else {
            fixture_dir().join("golden")
        };
        let work = root.join(name);
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).unwrap();
        let mut rbin = fixture_dir();
        if setup != "-" {
            for token in setup.split(' ') {
                if let Some(dir) = token.strip_prefix("dir:") {
                    std::fs::create_dir_all(work.join(dir)).unwrap();
                } else if let Some(file) = token.strip_prefix("old:") {
                    std::fs::write(work.join(file), "old\n").unwrap();
                } else if token == "rbin:none" {
                    rbin = work.join("nothere");
                } else {
                    let (src, dst) = token.split_once(':').unwrap();
                    std::fs::copy(fixture_dir().join(src), work.join(dst)).unwrap();
                }
            }
        }
        let mut input_names = Vec::new();
        walk(&work, "", &mut input_names);
        let inputs: Vec<(String, Vec<u8>)> = input_names
            .into_iter()
            .map(|file| {
                let bytes = std::fs::read(work.join(&file)).unwrap();
                (file, bytes)
            })
            .collect();

        let output = common::imod_cmd("runraptor")
            .current_dir(&work)
            .env("IMOD_DIR", &imod_dir)
            .env("RAPTOR_BIN", &rbin)
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    imod_dir.join("bin").display(),
                    std::env::var("PATH").unwrap_or_default()
                ),
            )
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args(args.split_whitespace())
            .output()
            .unwrap();
        count += 1;
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let stdout = mask_pid(
            &String::from_utf8_lossy(&output.stdout).replace(rbin.to_str().unwrap(), "RAPTOR_BIN"),
        );
        let expected_out = common::golden::expect(&golden.join(format!("{name}.out")));
        if !expected_out.matches(stdout.as_bytes()) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{stdout}",
                expected_out.display(),
            ));
        }
        let mut all = Vec::new();
        walk(&work, "", &mut all);
        let mut produced: Vec<(String, String)> = all
            .into_iter()
            .filter(|file| {
                !inputs.iter().any(|(name, bytes)| {
                    name == file && std::fs::read(work.join(file)).ok().as_ref() == Some(bytes)
                })
            })
            .map(|file| (mask_pid(&file).replace('/', "%"), file))
            .collect();
        produced.sort();
        let flat: Vec<String> = produced.iter().map(|(flat, _)| flat.clone()).collect();
        let expected_files = common::golden::list(&golden.join(name));
        if flat != expected_files {
            failures.push(format!("{name}: files {flat:?}, native {expected_files:?}"));
        }
        for (flat, file) in &produced {
            if let Some(expected) = common::golden::load(&golden.join(name).join(flat)) {
                let actual = std::fs::read(work.join(file)).unwrap();
                if let Err(why) = expected.compare(&actual, common::golden::identity, true) {
                    failures.push(format!("{name}: {file} differs from native: {why}"));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&work);
    }
    let _ = std::fs::remove_dir_all(&root);
    assert!(count >= 18, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Runs `batchruntomo -start 4 -end 4` with `trackingMethod = 2` on the stub
/// dataset (`stub.mrc` as `ts.st` and `ts_preali.mrc`, which is the name
/// `datasetFilename('.preali')` gives with no `.st`-style com files, `stub.prexg` as
/// `ts.prexg`, restrictalign's `align.com` and `ts.rawtlt`), our commands on `PATH` and `RAPTOR_BIN` pointing at the
/// RAPTOR stand-in.  Gold 10 nm at 0.4 nm/pixel is 25 pixels, and 12.5 at the
/// coarse binning of 2: `int(round(12.5))` is 12 in Python 3.  Returns the
/// exit status, standard output, `runraptor.com` and `ts.fid`.
fn batchruntomo_raptor(case: &str, aligned: bool) -> (Option<i32>, String, String, Vec<u8>) {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("IMOD");
    let dir = std::env::temp_dir().join(format!("imod-rs-brtraptor-{case}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (from, to) in [
        ("stub.mrc", "ts.st"),
        ("stub.mrc", "ts_preali.mrc"),
        ("stub.prexg", "ts.prexg"),
    ] {
        std::fs::copy(fixture_dir().join(from), dir.join(to)).unwrap();
    }
    // Read by the axis setup before any step runs
    let template = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/restrictalign/align.tmpl"),
    )
    .unwrap();
    std::fs::write(dir.join("align.com"), template.replace("MODEL", "ts.fid")).unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/restrictalign/ts.rawtlt"),
        dir.join("ts.rawtlt"),
    )
    .unwrap();
    std::fs::write(
        dir.join("track.com"),
        "$beadtrack -StandardInput\nImageFile\tts.preali\nOutputModel\tts.fid\n",
    )
    .unwrap();
    std::fs::write(dir.join("ts.edf"), "Setup.DatasetName=ts\n").unwrap();
    std::fs::write(
        dir.join("brt.adoc"),
        format!(
            "setupset.copyarg.dual = 0\nsetupset.copyarg.pixel = 0.4\n\
             setupset.copyarg.gold = 10\nsetupset.copyarg.rotation = -90\n\
             setupset.copyarg.userawtlt = 1\n\
             comparam.prenewst.newstack.BinByFactor = 2\n\
             runtime.Fiducials.any.trackingMethod = 2\n\
             runtime.RAPTOR.any.numberOfMarkers = 20\n\
             runtime.RAPTOR.any.useAlignedStack = {}\n",
            i32::from(aligned)
        ),
    )
    .unwrap();
    for command in imod_rs::imod::commands::COMMANDS {
        common::imod_link(command.name);
    }
    let path = format!(
        "{}:/usr/bin:/bin",
        common::command_link_directory().display()
    );
    let result = common::imod_cmd("batchruntomo")
        .current_dir(&dir)
        .env("PATH", path)
        .env("IMOD_DIR", &source)
        .env("AUTODOC_DIR", source.join("autodoc"))
        .env("RAPTOR_BIN", fixture_dir())
        .args(["-RootName", "ts", "-CurrentLocation"])
        .arg(&dir)
        .arg("-DirectiveFile")
        .arg(dir.join("brt.adoc"))
        .args(["-StartingStep", "4", "-EndingStep", "4"])
        .output()
        .unwrap();
    common::remove_command_links();
    let stdout = String::from_utf8_lossy(&result.stdout).into_owned();
    let com = std::fs::read_to_string(dir.join("runraptor.com")).unwrap_or_default();
    let fid = std::fs::read(dir.join("ts.fid")).unwrap_or_default();
    std::fs::remove_dir_all(&dir).unwrap();
    (result.status.code(), stdout, com, fid)
}

/// batchruntomo's RAPTOR step (`batchruntomo:2568-2591`) on the aligned
/// stack: it writes `runraptor.com`, runs it through processchunks (our
/// `runraptor`, as a child of our binary), and makes `ts_raptor.fid` the
/// fiducial model.  The model is runraptor's `preali_mrc` case (native golden).
/// Native batchruntomo on this dataset wrote the same `runraptor.com` and
/// `ts.fid` (TODO.md, "runraptor").
#[test]
fn batchruntomo_tracks_with_raptor_on_the_aligned_stack() {
    let (status, stdout, com, fid) = batchruntomo_raptor("aligned", true);
    assert_eq!(status, Some(0), "{stdout}");
    assert!(
        stdout.contains("Tracking fiducials with RAPTOR (running runraptor.com)"),
        "{stdout}"
    );
    assert!(!stdout.contains("ABORT"), "{stdout}");
    assert_eq!(
        com,
        "# Command file to run raptor\n$runraptor -mark 20 -diam 12 ts_preali.mrc\n"
    );
    let expected = common::golden::expect(&fixture_dir().join("golden/preali_mrc/ts_raptor.fid"));
    expected
        .compare(&fid, common::golden::identity, true)
        .unwrap();
}

/// As above on the raw stack: the diameter is not divided by the binning,
/// and runraptor maps the model to the aligned stack with `xfmodel`
/// (runraptor's `raw_st` case).
#[test]
fn batchruntomo_tracks_with_raptor_on_the_raw_stack() {
    let (status, stdout, com, fid) = batchruntomo_raptor("raw", false);
    assert_eq!(status, Some(0), "{stdout}");
    assert!(!stdout.contains("ABORT"), "{stdout}");
    assert_eq!(
        com,
        "# Command file to run raptor\n$runraptor -mark 20 -diam 25 ts.st\n"
    );
    let expected = common::golden::expect(&fixture_dir().join("golden/raw_st/ts_raptor.fid"));
    expected
        .compare(&fid, common::golden::identity, true)
        .unwrap();
}
