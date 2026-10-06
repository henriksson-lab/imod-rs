//! Native-golden coverage for `subimage` (`IMOD/flib/image/subimage.f90`).
//!
//! Every row of `fixtures/subimage/cases.tsv` was run through the native reference
//! program by `fixtures/make-subimage-goldens.sh`, in a fresh directory holding
//! copies of the fixture inputs, with standard output captured through a
//! pipe.  Exit status, standard output and every output file must match byte
//! for byte, apart from MRC label time stamps, usage-banner build dates and
//! model bytes native writes from uninitialised memory (`common::small_prog`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "subimage",
        fixtures: "subimage",
        min_cases: 11,
        reconcile: true,
        env: &[("OMP_NUM_THREADS", "1")],
        stdout_mask: common::small_prog::mask_banner,
    });
}

/// Defined behaviour (BUGS.md, `subimage`): with a subset of sections the Z
/// cell is scaled so the pixel size is kept (native leaves the cell, so the Z
/// pixel size grows by nz / numAsec).
#[test]
fn a_section_subset_keeps_the_z_pixel_size() {
    let dir = std::env::temp_dir().join(format!("imod-rs-subimage-cell-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/subimage");
    for f in ["v.mrc", "v2.mrc"] {
        std::fs::copy(fixtures.join(f), dir.join(f)).unwrap();
    }
    let status = common::imod_cmd("subimage")
        .current_dir(&dir)
        .env("AUTODOC_DIR", concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"))
        .args(["-afile", "v.mrc", "-bfile", "v2.mrc", "-asec", "0-1", "-bsec", "1-2", "-o", "d.mrc"])
        .output()
        .unwrap();
    assert_eq!(status.status.code(), Some(0));
    let header = |name: &str| -> (i32, f32) {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        (
            i32::from_le_bytes(bytes[36..40].try_into().unwrap()),
            f32::from_le_bytes(bytes[48..52].try_into().unwrap()),
        )
    };
    let (mz_in, zlen_in) = header("v.mrc");
    let (mz_out, zlen_out) = header("d.mrc");
    assert_eq!(mz_out, 2);
    assert_eq!(zlen_out / mz_out as f32, zlen_in / mz_in as f32);
    let _ = std::fs::remove_dir_all(&dir);
}
