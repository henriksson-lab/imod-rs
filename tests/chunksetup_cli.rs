//! Native-golden coverage for `chunksetup` (`IMOD/pysrc/chunksetup`, translated in
//! `src/imod/pysrc/chunksetup.rs`).
//!
//! Every row of `fixtures/chunksetup/cases.tsv` was run through the native Python
//! script by `fixtures/make-chunksetup-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens is recorded
//! in `TODO.md` (single-axis setup scripts).

mod common;
mod pysetup_common;

#[test]
fn chunksetup_matches_native_goldens() {
    let failures = pysetup_common::run_cases("chunksetup");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn chunksetup_requires_imod_dir() {
    let output = common::imod_cmd("chunksetup")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run chunksetup without IMOD_DIR");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: chunksetup -  IMOD_DIR is not defined!\n"
    );
}
