//! Native-golden coverage for `copyheader` (`IMOD/pysrc/copyheader`, translated in
//! `src/imod/pysrc/copyheader.rs`).
//!
//! Every row of `fixtures/copyheader/cases.tsv` was run through the native Python
//! script by `fixtures/make-copyheader-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn copyheader_matches_native_goldens() {
    let failures = pysetup_common::run_cases("copyheader");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
