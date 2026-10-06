//! Native-golden coverage for `imodextract` (`IMOD/imodutil/imodextract.c`).
//!
//! Every row of `fixtures/imodextract/cases.tsv` was run through the native
//! reference program by `fixtures/make-imodextract-goldens.sh` (inputs from
//! `fixtures/make-imodextract-inputs.sh`: a model with views and four object
//! groups), in a fresh directory holding copies of the fixture inputs.  Exit
//! status, standard output and every output file must match byte for byte;
//! model bytes native writes from uninitialised memory are reconciled
//! (`common::reconcile_uninitialised`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "imodextract",
        fixtures: "imodextract",
        min_cases: 25,
        reconcile: true,
        env: &[],
        stdout_mask: common::mask_stamps,
    });
}

/// The empty argument: `parselist` returns NULL for an empty string
/// (`parselist.c:51-52`), so the program reports a parse error (native
/// identical; the table cannot hold an empty argument).
#[test]
fn an_empty_list_is_a_parse_error() {
    let work =
        std::env::temp_dir().join(format!("imod-rs-imodextract-empty-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    std::fs::copy(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/imodextract/six.mod"),
        work.join("six.mod"),
    )
    .unwrap();
    let output = common::imod_cmd("imodextract")
        .current_dir(&work)
        .args(["", "six.mod", "o.mod"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stdout).contains("Parsing object list"));
    assert!(!work.join("o.mod").exists());
    let _ = std::fs::remove_dir_all(work);
}
