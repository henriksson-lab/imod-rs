//! Native-golden coverage for `makecomfile` (`IMOD/pysrc/makecomfile`, translated in
//! `src/imod/pysrc/makecomfile.rs`).
//!
//! Every row of `fixtures/makecomfile/cases.tsv` was run through the native Python
//! script by `fixtures/make-makecomfile-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens is recorded
//! in `TODO.md` (single-axis setup scripts).
//!
//! Pruned 2026-09-26: 23 of 25 rows kept (dropped `autofidseed`, the no-axis-letter form other types cover, and `xcorr_pt_change`, a -change run other types cover); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;
mod pysetup_common;

#[test]
fn makecomfile_matches_native_goldens() {
    let failures = pysetup_common::run_cases("makecomfile");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn makecomfile_requires_imod_dir() {
    let output = common::imod_cmd("makecomfile")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run makecomfile without IMOD_DIR");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: makecomfile -  IMOD_DIR is not defined!\n"
    );
}
