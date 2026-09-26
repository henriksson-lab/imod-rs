//! Native-golden coverage for `collectmmm` (`IMOD/pysrc/collectmmm`, translated in
//! `src/imod/pysrc/collectmmm.rs`).
//!
//! Every row of `fixtures/collectmmm/cases.tsv` was run through the native Python
//! script by `fixtures/make-collectmmm-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens (with the
//! native combine programs on `PATH`) is recorded in `TODO.md` (dual-axis
//! combine scripts).

mod common;
mod pysetup_common;

#[test]
fn collectmmm_matches_native_goldens() {
    let failures = pysetup_common::run_cases("collectmmm");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// Fixed in translation (BUGS.md): native divides by the zero pixel sum of no
// logs (`collectmmm:98`, ZeroDivisionError traceback, exit 1); the
// translation reports an ordinary error, exit 1, and leaves the image alone.
#[test]
fn zero_logs_is_an_error_not_a_division_by_zero() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!("imod-rs-collectmmm-zero-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let image = root.join("fixtures/collectmmm/inputs/img.mrc");
    std::fs::copy(&image, work.join("img.mrc")).unwrap();
    let output = common::imod_cmd("collectmmm")
        .args(["pixels=", "r", "0", "img.mrc"])
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout).starts_with("ERROR: collectmmm - "),
        "{output:?}"
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("Traceback"));
    assert_eq!(
        std::fs::read(work.join("img.mrc")).unwrap(),
        std::fs::read(&image).unwrap()
    );
    let _ = std::fs::remove_dir_all(&work);
}
