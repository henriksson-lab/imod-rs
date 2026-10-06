//! Native-golden coverage for `imod2patch` (`IMOD/imodutil/imod2patch.c`).
//!
//! Every row of `fixtures/imod2patch/cases.tsv` was run through the native reference
//! program by `fixtures/make-imod2patch-goldens.sh` (inputs from
//! `fixtures/make-etomo-prog-inputs.sh`), in a fresh directory holding copies
//! of the fixture inputs, with standard output captured through a pipe.  Exit
//! status, standard output and every output file must match byte for byte;
//! model bytes native writes from uninitialised memory (the name tail, the
//! `MINX` fields) are reconciled (`common::reconcile_uninitialised`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "imod2patch",
        fixtures: "imod2patch",
        min_cases: 10,
        reconcile: true,
        env: &[],
        stdout_mask: common::mask_stamps,
    });
}
