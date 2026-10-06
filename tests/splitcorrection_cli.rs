//! Native-golden coverage for `splitcorrection` (`IMOD/pysrc/splitcorrection`, translated in
//! `src/imod/pysrc/splitcorrection.rs`).
//!
//! Every row of `fixtures/splitcorrection/cases.tsv` was run through the native Python
//! script by `fixtures/make-splitcorrection-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn splitcorrection_matches_native_goldens() {
    let failures = pysetup_common::run_cases("splitcorrection");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
