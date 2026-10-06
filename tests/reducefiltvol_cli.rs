//! Native-golden coverage for `reducefiltvol` (`IMOD/pysrc/reducefiltvol`, translated in
//! `src/imod/pysrc/reducefiltvol.rs`).
//!
//! Every row of `fixtures/reducefiltvol/cases.tsv` was run through the native Python
//! script by `fixtures/make-reducefiltvol-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn reducefiltvol_matches_native_goldens() {
    let failures = pysetup_common::run_cases("reducefiltvol");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
