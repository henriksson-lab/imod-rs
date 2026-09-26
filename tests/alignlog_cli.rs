//! Native-golden coverage for `alignlog` (`IMOD/pysrc/alignlog`, translated in
//! `src/imod/pysrc/alignlog.rs`).
//!
//! Every row of `fixtures/alignlog/cases.tsv` was run through the native Python
//! script by `fixtures/make-alignlog-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens is recorded
//! in `TODO.md` (single-axis setup scripts).

mod common;
mod pysetup_common;

#[test]
fn alignlog_matches_native_goldens() {
    let failures = pysetup_common::run_cases("alignlog");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn alignlog_requires_imod_dir() {
    let output = common::imod_cmd("alignlog")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run alignlog without IMOD_DIR");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: alignlog -  IMOD_DIR is not defined!\n"
    );
}
