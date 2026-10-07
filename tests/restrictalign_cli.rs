//! Native-golden coverage for `restrictalign` (`IMOD/pysrc/restrictalign`,
//! translated in `src/imod/pysrc/restrictalign.rs`).
//!
//! Every row of `fixtures/restrictalign/cases.tsv` was run through the native
//! Python script by `fixtures/make-restrictalign-goldens.sh`, with the native
//! tiltalign, imodinfo, submfg, xfproduct, ... on `PATH`.  `golden/<case>.rc`
//! is the exit status, `.out` the standard output, and `golden/<case>/` every
//! file new or changed afterwards: the rewritten `align.com` and, for the
//! cross-validation cases, the final `submfg align.com` run's log and
//! outputs.  Here our `imod` runs with every command it calls linked to it.
//! Cases whose output an upstream-bug fix changes take their expectation from
//! `defined/` (`defined.list`, `BUGS.md`).  Tiltalign's values pass through
//! LAPACK (`CLAUDE.md`); every recorded output is byte-identical to native.
//!
//! Pruned: 23 of 40 rows kept (each option and path once: ratio restriction
//! with and without `-order`, 1/3/8/25 beads, cross-validation global,
//! robust, local areas and variables, `-onestep 2`, `-permute`, distortion
//! variables, the patch-tracking count, the defined cases and the error
//! exits); the rest stay in cases.tsv as `#full` rows (FULL=1,
//! fixtures/README.md).

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/restrictalign")
}

/// An `IMOD_DIR` whose `bin` links every command restrictalign runs to our
/// binary.
fn install() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-restrictalign-{}", std::process::id()));
    let bin = dir.join("imod/bin");
    std::fs::create_dir_all(&bin).unwrap();
    for command in [
        "restrictalign",
        "tiltalign",
        "imodinfo",
        "submfg",
        "xfproduct",
        "b3dcopy",
        "patch2imod",
        "header",
    ] {
        #[cfg(unix)]
        let _ = std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_imod"), bin.join(command));
        #[cfg(not(unix))]
        let _ = std::fs::hard_link(
            env!("CARGO_BIN_EXE_imod"),
            bin.join(format!("{command}{}", std::env::consts::EXE_SUFFIX)),
        );
    }
    dir.join("imod")
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let template = std::fs::read_to_string(fixture_dir().join("align.tmpl")).unwrap();
    let imod_dir = install();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (name, model, script, args) = (fields[0], fields[1], fields[2], fields[3]);
        let defined = fixture_dir().join("defined");
        let golden = if common::golden::exists(&defined.join(format!("{name}.rc"))) {
            defined
        } else {
            fixture_dir().join("golden")
        };
        let work = imod_dir.parent().unwrap().join(name);
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).unwrap();
        let mut inputs = Vec::new();
        for entry in std::fs::read_dir(fixture_dir()).unwrap() {
            let path = entry.unwrap().path();
            let file = path.file_name().unwrap().to_str().unwrap().to_owned();
            if file.ends_with(".fid")
                || ["ts.rawtlt", "ts.prexg", "stub.mrc"].contains(&file.as_str())
            {
                std::fs::copy(&path, work.join(&file)).unwrap();
                inputs.push((file.clone(), std::fs::read(&path).unwrap()));
            }
        }
        // The case's sed script, applied as the make script does
        let mut com = template.replace("MODEL", model);
        if script != "-" {
            let output = std::process::Command::new("sed")
                .arg(script)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .and_then(|mut child| {
                    use std::io::Write as _;
                    child.stdin.take().unwrap().write_all(com.as_bytes())?;
                    child.wait_with_output()
                })
                .unwrap();
            com = String::from_utf8(output.stdout).unwrap();
        }
        std::fs::write(work.join("align.com"), &com).unwrap();
        inputs.push(("align.com".to_owned(), com.into_bytes()));

        let output = common::imod_cmd("restrictalign")
            .current_dir(&work)
            .env("IMOD_DIR", &imod_dir)
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
            .env("OMP_NUM_THREADS", "1")
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
        let expected_out = common::golden::expect(&golden.join(format!("{name}.out")));
        if !expected_out.matches(&output.stdout) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                expected_out.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let mut produced: Vec<String> = std::fs::read_dir(&work)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_owned())
            .filter(|file| {
                !inputs.iter().any(|(name, bytes)| {
                    name == file && std::fs::read(work.join(file)).ok().as_ref() == Some(bytes)
                })
            })
            .collect();
        produced.sort();
        let expected_files = common::golden::list(&golden.join(name));
        if produced != expected_files {
            failures.push(format!(
                "{name}: files {produced:?}, native {expected_files:?}"
            ));
        }
        for file in &produced {
            if let Some(expected) = common::golden::load(&golden.join(name).join(file)) {
                let actual = std::fs::read(work.join(file)).unwrap();
                if let Err(why) = expected.compare(&actual, common::mask_stamps, true) {
                    failures.push(format!("{name}: {file} differs from native: {why}"));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&work);
    }
    let _ = std::fs::remove_dir_all(imod_dir.parent().unwrap());
    assert!(count >= 23, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
