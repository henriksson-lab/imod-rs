//! Native-golden coverage for `xfjointomo` (`IMOD/flib/model/xfjointomo.f`).
//!
//! Every row of `fixtures/xfjointomo/cases.tsv` was run through the native reference
//! program by `fixtures/make-xfjointomo-goldens.sh` (inputs from
//! `fixtures/make-etomo-prog-inputs.sh`), in a fresh directory holding copies
//! of the fixture inputs, with standard output captured through a pipe.  Exit
//! status, standard output and every output file must match byte for byte,
//! apart from MRC label time stamps (`common::small_prog`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "xfjointomo",
        fixtures: "xfjointomo",
        min_cases: 10,
        reconcile: false,
        env: &[],
        stdout_mask: common::mask_stamps,
    });
}
