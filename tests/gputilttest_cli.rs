//! Native-golden coverage for `gputilttest` (`IMOD/pysrc/gputilttest`, translated in
//! `src/imod/pysrc/gputilttest.rs`).
//!
//! Every row of `fixtures/gputilttest/cases.tsv` was run through the native Python
//! script by `fixtures/make-gputilttest-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn gputilttest_matches_native_goldens() {
    let failures = pysetup_common::run_cases("gputilttest");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
