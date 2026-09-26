//! Native-golden coverage for `splittilt` (`IMOD/pysrc/splittilt`, translated in
//! `src/imod/pysrc/splittilt.rs`).
//!
//! Every row of `fixtures/splittilt/cases.tsv` was run through the native Python
//! script by `fixtures/make-splittilt-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens is recorded
//! in `TODO.md` (single-axis setup scripts).

mod common;
mod pysetup_common;

#[test]
fn splittilt_matches_native_goldens() {
    let failures = pysetup_common::run_cases("splittilt");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn splittilt_requires_imod_dir() {
    let output = common::imod_cmd("splittilt")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run splittilt without IMOD_DIR");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: splittilt -  IMOD_DIR is not defined!\n"
    );
}
