//! Native-golden coverage for `slicesforsample` (`IMOD/pysrc/slicesforsample`, translated in
//! `src/imod/pysrc/slicesforsample.rs`).
//!
//! Every row of `fixtures/slicesforsample/cases.tsv` was run through the native Python
//! script by `fixtures/make-slicesforsample-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn slicesforsample_matches_native_goldens() {
    let failures = pysetup_common::run_cases("slicesforsample");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
