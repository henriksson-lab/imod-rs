//! Native-golden coverage for `startprocess` (`IMOD/pysrc/startprocess`, translated in
//! `src/imod/pysrc/startprocess.rs`).
//!
//! Every row of `fixtures/startprocess/cases.tsv` was run through the native Python
//! script by `fixtures/make-startprocess-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn startprocess_matches_native_goldens() {
    let failures = pysetup_common::run_cases("startprocess");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
