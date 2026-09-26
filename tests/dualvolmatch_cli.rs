//! Native-golden coverage for `dualvolmatch` (`IMOD/pysrc/dualvolmatch`, translated in
//! `src/imod/pysrc/dualvolmatch.rs`).
//!
//! Every row of `fixtures/dualvolmatch/cases.tsv` was run through the native Python
//! script by `fixtures/make-dualvolmatch-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens (with the
//! native combine programs on `PATH`) is recorded in `TODO.md` (dual-axis
//! combine scripts).

mod common;
mod pysetup_common;

#[test]
fn dualvolmatch_matches_native_goldens() {
    let failures = pysetup_common::run_cases("dualvolmatch");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// Fixed in translation (BUGS.md): native reads the A tomogram's size twice
// (`dualvolmatch:89`), so an unreadable B tomogram goes unnoticed there.  The
// translation reads B, and an unreadable one stops the run, exit 1.
#[test]
fn b_tomogram_size_is_read_from_b() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!("imod-rs-dualvolmatch-b-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    std::fs::copy(
        root.join("fixtures/dualvolmatch/inputs/ga.rec"),
        work.join("ga.rec"),
    )
    .unwrap();
    std::fs::write(work.join("gb.rec"), b"this is not an image file\n").unwrap();
    let output = common::imod_cmd("dualvolmatch")
        .args(["-name", "g"])
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("gb.rec")
            || String::from_utf8_lossy(&output.stdout).contains("gb.rec"),
        "{output:?}"
    );
    let _ = std::fs::remove_dir_all(&work);
}
