//! Native-golden coverage for `nad_eed_3d` (`IMOD/mrc/nad_eed_3d.c`), the
//! edge-enhancing nonlinear anisotropic diffusion program eTomo's NAD
//! interface runs.
//!
//! Every row of `fixtures/nad_eed_3d/cases.tsv` was run through the native
//! reference program by `fixtures/make-nad_eed_3d-goldens.sh` (inputs from
//! `fixtures/make-etomo-prog-inputs.sh`), in a fresh directory holding copies
//! of the fixture inputs.  Exit status, standard output and every output file
//! must match byte for byte, apart from MRC label time stamps and the
//! `asctime` lines ("started at:", "finished at:").

mod common;

/// Masks the two `asctime` lines and the MRC-style stamps.
fn mask_times(bytes: &[u8]) -> Vec<u8> {
    let masked = common::mask_stamps(bytes);
    let text = String::from_utf8_lossy(&masked);
    text.lines()
        .map(|line| {
            if line.contains("started at:") || line.contains("finished at:") {
                "<time>"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .into_bytes()
}

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "nad_eed_3d",
        fixtures: "nad_eed_3d",
        min_cases: 14,
        reconcile: true,
        env: &[("OMP_NUM_THREADS", "1")],
        stdout_mask: mask_times,
    });
}

/// Defined behaviour (BUGS.md, `nad_eed_3d`): an option that takes a value,
/// given last, is an error naming it (native reads `argv[argc]` and crashes).
#[test]
fn an_option_without_its_value_is_an_error() {
    let output = common::imod_cmd("nad_eed_3d")
        .args(["-n", "5", "-k"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Option -k must be followed by an entry"),
        "{stdout}"
    );
}
