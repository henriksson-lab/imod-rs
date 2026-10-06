//! Native-golden coverage for `squeezevol` (`IMOD/pysrc/squeezevol`, translated in
//! `src/imod/pysrc/squeezevol.rs`).
//!
//! Every row of `fixtures/squeezevol/cases.tsv` was run through the native Python
//! script by `fixtures/make-squeezevol-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn squeezevol_matches_native_goldens() {
    let failures = pysetup_common::run_cases("squeezevol");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
