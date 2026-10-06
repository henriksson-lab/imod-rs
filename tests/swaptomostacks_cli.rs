//! Native-golden coverage for `swaptomostacks` (`IMOD/pysrc/swaptomostacks`, translated in
//! `src/imod/pysrc/swaptomostacks.rs`).
//!
//! Every row of `fixtures/swaptomostacks/cases.tsv` was run through the native Python
//! script by `fixtures/make-swaptomostacks-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn swaptomostacks_matches_native_goldens() {
    let failures = pysetup_common::run_cases("swaptomostacks");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
