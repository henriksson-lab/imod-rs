//! Native-golden coverage for `autopatchfit` (`IMOD/pysrc/autopatchfit`, translated in
//! `src/imod/pysrc/autopatchfit.rs`).
//!
//! Every row of `fixtures/autopatchfit/cases.tsv` was run through the native Python
//! script by `fixtures/make-autopatchfit-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens (with the
//! native combine programs on `PATH`) is recorded in `TODO.md` (dual-axis
//! combine scripts).
//!
//! Pruned 2026-09-26: 6 of 7 rows kept (dropped `err_nopc`, the same open-failure error as `err_nomow`); the rest stay in cases.tsv as `#full` rows (FULL=1, fixtures/README.md).

mod common;
mod pysetup_common;

#[test]
fn autopatchfit_matches_native_goldens() {
    let failures = pysetup_common::run_cases("autopatchfit");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
