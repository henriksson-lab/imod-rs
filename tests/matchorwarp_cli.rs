//! Native-golden coverage for `matchorwarp` (`IMOD/pysrc/matchorwarp`, translated in
//! `src/imod/pysrc/matchorwarp.rs`).
//!
//! Every row of `fixtures/matchorwarp/cases.tsv` was run through the native Python
//! script by `fixtures/make-matchorwarp-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens (with the
//! native combine programs on `PATH`) is recorded in `TODO.md` (dual-axis
//! combine scripts).

mod common;
mod pysetup_common;

#[test]
fn matchorwarp_matches_native_goldens() {
    let failures = pysetup_common::run_cases("matchorwarp");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// Fixed in translation (BUGS.md): the source's fallback option table names
// `CriteriaToStopIterating` while the program reads `StopIteratingCriteria`,
// so without the autodoc native stops with "Illegal option" for
// -iterations 2 or more.  With the table fixed the run gets as far as the
// patchcorr command file's check, as it does with the autodoc.
#[test]
fn fallback_option_table_has_stop_iterating_criteria() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/matchorwarp/inputs");
    let work = std::env::temp_dir().join(format!(
        "imod-rs-matchorwarp-fallback-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(work.join("imod")).unwrap();
    for (source, target) in [
        ("ga.rec", "ga.rec"),
        ("gb.rec", "gb.rec"),
        ("patch.out", "patch.out"),
        ("solve.xf", "solve.xf"),
        ("pc_nomat.com", "pc.com"),
    ] {
        std::fs::copy(inputs.join(source), work.join(target)).unwrap();
    }
    let output = common::imod_cmd("matchorwarp")
        .args([
            "-size",
            "ga.rec",
            "-iterations",
            "2",
            "-patchcorr",
            "pc.com",
            "gb.rec",
            "gb.mat",
        ])
        .current_dir(&work)
        .env("IMOD_DIR", work.join("imod"))
        .env_remove("AUTODOC_DIR")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(!stdout.contains("Illegal option"), "{stdout}");
    assert!(
        stdout.contains("Cannot find name of file to align in pc.com"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&work);
}
