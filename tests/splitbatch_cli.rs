//! Native-golden coverage for `splitbatch` (`IMOD/pysrc/splitbatch`, translated in
//! `src/imod/pysrc/splitbatch.rs`).
//!
//! Every row of `fixtures/splitbatch/cases.tsv` was run through the native Python
//! script by `fixtures/make-splitbatch-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn splitbatch_matches_native_goldens() {
    let failures = pysetup_common::run_cases("splitbatch");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
