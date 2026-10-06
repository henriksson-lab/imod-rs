//! Native-golden coverage for `tomodataplots` (`IMOD/pysrc/tomodataplots`, translated in
//! `src/imod/pysrc/tomodataplots.rs`).
//!
//! Every row of `fixtures/tomodataplots/cases.tsv` was run through the native Python
//! script by `fixtures/make-tomodataplots-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn tomodataplots_matches_native_goldens() {
    let failures = pysetup_common::run_cases("tomodataplots");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
