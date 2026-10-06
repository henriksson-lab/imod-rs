//! Native-golden coverage for `excludeviews` (`IMOD/pysrc/excludeviews`, translated in
//! `src/imod/pysrc/excludeviews.rs`).
//!
//! Every row of `fixtures/excludeviews/cases.tsv` was run through the native Python
//! script by `fixtures/make-excludeviews-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn excludeviews_matches_native_goldens() {
    let failures = pysetup_common::run_cases("excludeviews");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
