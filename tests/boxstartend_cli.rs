//! Native-golden coverage for `boxstartend` (`IMOD/flib/model/boxstartend.f90`).
//!
//! Every row of `fixtures/boxstartend/cases.tsv` was run through the native reference
//! program by `fixtures/make-boxstartend-goldens.sh`, in a fresh directory holding
//! copies of the fixture inputs, with standard output captured through a
//! pipe.  Exit status, standard output and every output file must match byte
//! for byte, apart from MRC label time stamps, usage-banner build dates and
//! model bytes native writes from uninitialised memory (`common::small_prog`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "boxstartend",
        fixtures: "boxstartend",
        min_cases: 7,
        reconcile: true,
        env: &[("OMP_NUM_THREADS", "1")],
        stdout_mask: common::small_prog::mask_banner,
    });
}

/// Defined behaviour (BUGS.md, `boxstartend`): with no slices before or after
/// (local mean fill), the part of a box outside the image is filled with the
/// mean of the part read, not with the whole file's mean (native ignores the
/// fill value it is given).  Image: left half 10, right half 30 (mean 20); a
/// point near the left edge whose box reads only the left half.
#[test]
fn local_fill_uses_the_mean_of_the_part_read() {
    let dir = std::env::temp_dir().join(format!("imod-rs-boxstartend-fill-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut raw = Vec::new();
    for _y in 0..20 {
        for x in 0..20 {
            raw.extend_from_slice(&(if x < 10 { 10.0f32 } else { 30.0 }).to_le_bytes());
        }
    }
    std::fs::write(dir.join("i.raw"), raw).unwrap();
    let made = common::imod_cmd("raw2mrc")
        .current_dir(&dir)
        .args(["-x", "20", "-y", "20", "-z", "1", "-t", "float", "i.raw", "i.mrc"])
        .output()
        .unwrap();
    assert_eq!(made.status.code(), Some(0));
    std::fs::write(dir.join("p.txt"), "1 1 2 10 0\n1 1 3 10 0\n").unwrap();
    let made = common::imod_cmd("point2model")
        .current_dir(&dir)
        .args(["-input", "p.txt", "-output", "p.mod", "-image", "i.mrc", "-open"])
        .output()
        .unwrap();
    assert_eq!(made.status.code(), Some(0));
    let output = common::imod_cmd("boxstartend")
        .current_dir(&dir)
        .env("AUTODOC_DIR", concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"))
        .args(["-image", "i.mrc", "-model", "p.mod", "-output", "o.mrc", "-box", "7", "-slices", "0,0"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0), "{}", String::from_utf8_lossy(&output.stdout));
    let bytes = std::fs::read(dir.join("o.mrc")).unwrap();
    let next = i32::from_le_bytes(bytes[92..96].try_into().unwrap()) as usize;
    let first: Vec<f32> = bytes[1024 + next..1024 + next + 49 * 4]
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(first.iter().all(|&v| v == 10.0), "{first:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
