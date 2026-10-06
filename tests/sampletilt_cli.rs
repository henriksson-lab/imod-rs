//! Native-golden coverage for `sampletilt` (`IMOD/pysrc/sampletilt`, translated in
//! `src/imod/pysrc/sampletilt.rs`).
//!
//! Every row of `fixtures/sampletilt/cases.tsv` was run through the native Python
//! script by `fixtures/make-sampletilt-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn sampletilt_matches_native_goldens() {
    let failures = pysetup_common::run_cases("sampletilt");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
