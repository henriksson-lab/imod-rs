//! Native-golden coverage for `joinwarp2model` (`IMOD/imodutil/joinwarp2model.c`).
//!
//! Every row of `fixtures/joinwarp2model/cases.tsv` was run through the native reference
//! program by `fixtures/make-joinwarp2model-goldens.sh` (inputs from
//! `fixtures/make-etomo-prog-inputs.sh`), in a fresh directory holding copies
//! of the fixture inputs, with standard output captured through a pipe.  Exit
//! status, standard output and every output file must match byte for byte;
//! model bytes native writes from uninitialised memory (the name tail, the
//! `MINX` fields) are reconciled (`common::reconcile_uninitialised`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "joinwarp2model",
        fixtures: "joinwarp2model",
        min_cases: 9,
        reconcile: true,
        env: &[],
        stdout_mask: common::mask_stamps,
    });
}
