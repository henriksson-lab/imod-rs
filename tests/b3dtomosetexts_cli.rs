//! Native-golden coverage for `b3dtomosetexts` (`IMOD/pysrc/b3dtomosetexts`, translated in
//! `src/imod/pysrc/b3dtomosetexts.rs`).
//!
//! Every row of `fixtures/b3dtomosetexts/cases.tsv` was run through the native Python
//! script by `fixtures/make-b3dtomosetexts-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn b3dtomosetexts_matches_native_goldens() {
    let failures = pysetup_common::run_cases("b3dtomosetexts");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
