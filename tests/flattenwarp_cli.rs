//! Native-golden coverage for `flattenwarp` (`IMOD/imodutil/flattenwarp.c`).
//!
//! Every row of `fixtures/flattenwarp/cases.tsv` was run through the native
//! reference program by `fixtures/make-flattenwarp-goldens.sh` (inputs from
//! `fixtures/make-etomo-prog-inputs.sh`), in a fresh directory holding copies
//! of the fixture inputs, with standard output captured through a pipe.  Exit
//! status, standard output and every output file must match byte for byte;
//! the name tail of the middle-contour model, which native writes from
//! uninitialised heap, is reconciled (`common::reconcile_uninitialised`).
//! The thin plate spline cases go through LAPACK `dsysv` (`faer` here), and
//! are byte-identical on these well-conditioned systems; the committed
//! tolerance is in `src/imod/flib/subrs/lapack/dsysv.rs`.

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "flattenwarp",
        fixtures: "flattenwarp",
        min_cases: 14,
        reconcile: true,
        env: &[],
        stdout_mask: common::small_prog::mask_banner,
    });
}
