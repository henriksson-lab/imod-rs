//! Native-golden coverage for `tomocleanup` (`IMOD/pysrc/tomocleanup`, translated in
//! `src/imod/pysrc/tomocleanup.rs`).
//!
//! Every row of `fixtures/tomocleanup/cases.tsv` was run through the native Python
//! script by `fixtures/make-tomocleanup-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens is recorded
//! in `TODO.md` (single-axis setup scripts).

mod common;
mod pysetup_common;

#[test]
fn tomocleanup_matches_native_goldens() {
    let failures = pysetup_common::run_cases("tomocleanup");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn tomocleanup_requires_imod_dir() {
    let output = common::imod_cmd("tomocleanup")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run tomocleanup without IMOD_DIR");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: tomocleanup -  IMOD_DIR is not defined!\n"
    );
}
