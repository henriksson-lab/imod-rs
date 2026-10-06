//! Native-golden coverage for `cryoposition` (`IMOD/pysrc/cryoposition`, translated in
//! `src/imod/pysrc/cryoposition.rs`).
//!
//! Every row of `fixtures/cryoposition/cases.tsv` was run through the native Python
//! script by `fixtures/make-cryoposition-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.  The four cases that ran the
//! whole pipeline on a 1.3 MB float stack (`plain`, `bin2`, `light_erase`,
//! `scales_box`) were deleted with that stack; the argument and error cases
//! remain.

mod common;
mod pysetup_common;

#[test]
fn cryoposition_matches_native_goldens() {
    let failures = pysetup_common::run_cases("cryoposition");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Fixed in translation (BUGS.md, `cryoposition`): with `-find` and no
/// `BeadDiameter` in `track.com` the script gives its intended error (native
/// raises a TypeError comparing `None`, and would then name an undefined
/// `trackcom`).
#[test]
fn find_beads_without_a_bead_diameter() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/cryoposition/inputs");
    let work = std::env::temp_dir().join(format!("imod-rs-cryoposition-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for entry in std::fs::read_dir(&inputs).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, work.join(path.file_name().unwrap())).unwrap();
    }
    // Any MRC serves as the raw stack: the error comes right after its size
    // is read.  (The suite's 128x128x21 `g.st` was removed to keep fixtures
    // small.)
    std::fs::copy(root.join("fixtures/newstack-mixed-small.mrc"), work.join("g.st")).unwrap();
    std::fs::write(
        work.join("track.com"),
        "$beadtrack -StandardInput\nImageFile\tg.preali\n",
    )
    .unwrap();
    let output = common::imod_cmd("cryoposition")
        .args(["-root", "g", "-thick", "40", "-find", "1"])
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: cryoposition - There is no positive BeadDiameter entry in track.com; fix this or enter a size with -size\n"
    );
    let _ = std::fs::remove_dir_all(&work);
}
