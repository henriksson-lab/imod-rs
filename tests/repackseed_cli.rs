//! Native-golden coverage for `repackseed` (`IMOD/flib/model/repackseed.f90`).
//!
//! Every row of `fixtures/repackseed/cases.tsv` was run through the native reference
//! program by `fixtures/make-repackseed-goldens.sh`, in a fresh directory holding
//! copies of the fixture inputs, with standard output captured through a
//! pipe.  Exit status, standard output and every output file must match byte
//! for byte, apart from MRC label time stamps, usage-banner build dates and
//! model bytes native writes from uninitialised memory (`common::small_prog`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "repackseed",
        fixtures: "repackseed",
        min_cases: 4,
        reconcile: true,
        env: &[("OMP_NUM_THREADS", "1")],
        stdout_mask: common::small_prog::mask_banner,
    });
}
