//! Native-golden coverage for `avgstack` (`IMOD/flib/image/avgstack.f`).
//!
//! Every row of `fixtures/avgstack/cases.tsv` was run through the native reference
//! program by `fixtures/make-avgstack-goldens.sh` (inputs from
//! `fixtures/make-etomo-prog-inputs.sh`), in a fresh directory holding copies
//! of the fixture inputs, with standard output captured through a pipe.  Exit
//! status, standard output and every output file must match byte for byte,
//! apart from MRC label time stamps (`common::small_prog`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "avgstack",
        fixtures: "avgstack",
        min_cases: 4,
        reconcile: true,
        env: &[],
        stdout_mask: common::mask_stamps,
    });
}
