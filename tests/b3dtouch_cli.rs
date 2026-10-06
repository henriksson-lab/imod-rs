//! Native-golden coverage for `b3dtouch` (`IMOD/pysrc/b3dtouch`, translated in
//! `src/imod/pysrc/b3dtouch.rs`).
//!
//! Every row of `fixtures/b3dtouch/cases.tsv` was run through the native Python
//! script by `fixtures/make-b3dtouch-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn b3dtouch_matches_native_goldens() {
    let failures = pysetup_common::run_cases("b3dtouch");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
