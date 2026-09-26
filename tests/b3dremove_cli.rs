//! Native-golden coverage for `b3dremove` (`IMOD/pysrc/b3dremove`, translated in
//! `src/imod/pysrc/b3dremove.rs`).
//!
//! Every row of `fixtures/b3dremove/cases.tsv` was run through the native Python
//! script by `fixtures/make-b3dremove-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens (with the
//! native combine programs on `PATH`) is recorded in `TODO.md` (dual-axis
//! combine scripts).

mod common;
mod pysetup_common;

#[test]
fn b3dremove_matches_native_goldens() {
    let failures = pysetup_common::run_cases("b3dremove");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
