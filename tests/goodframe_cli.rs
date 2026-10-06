//! Native-golden coverage for `goodframe` (`IMOD/flib/image/goodframe.f90`).
//!
//! Every row of `fixtures/goodframe/cases.tsv` was run through the native reference
//! program by `fixtures/make-goodframe-goldens.sh` (inputs from
//! `fixtures/make-etomo-prog-inputs.sh`), in a fresh directory holding copies
//! of the fixture inputs, with standard output captured through a pipe.  Exit
//! status, standard output and every output file must match byte for byte,
//! apart from MRC label time stamps (`common::small_prog`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "goodframe",
        fixtures: "goodframe",
        min_cases: 6,
        reconcile: false,
        env: &[],
        stdout_mask: common::mask_stamps,
    });
}
