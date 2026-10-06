//! Native-golden coverage for `multifiltsetup` (`IMOD/pysrc/multifiltsetup`, translated in
//! `src/imod/pysrc/multifiltsetup.rs`).
//!
//! Every row of `fixtures/multifiltsetup/cases.tsv` was run through the native Python
//! script by `fixtures/make-multifiltsetup-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn multifiltsetup_matches_native_goldens() {
    let failures = pysetup_common::run_cases("multifiltsetup");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
