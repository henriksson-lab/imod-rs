//! Native-golden coverage for `onegenplot` (`IMOD/pysrc/onegenplot`, translated in
//! `src/imod/pysrc/onegenplot.rs`).
//!
//! Every row of `fixtures/onegenplot/cases.tsv` was run through the native Python
//! script by `fixtures/make-onegenplot-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn onegenplot_matches_native_goldens() {
    let failures = pysetup_common::run_cases("onegenplot");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
