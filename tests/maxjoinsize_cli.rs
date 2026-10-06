//! Native-golden coverage for `maxjoinsize` (`IMOD/flib/image/maxjoinsize.f`).
//!
//! Every row of `fixtures/maxjoinsize/cases.tsv` was run through the native reference
//! program by `fixtures/make-maxjoinsize-goldens.sh`, in a fresh directory holding
//! copies of the fixture inputs, with standard output captured through a
//! pipe.  Exit status, standard output and every output file must match byte
//! for byte, apart from MRC label time stamps, usage-banner build dates and
//! model bytes native writes from uninitialised memory (`common::small_prog`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "maxjoinsize",
        fixtures: "maxjoinsize",
        min_cases: 7,
        reconcile: true,
        env: &[("OMP_NUM_THREADS", "1")],
        stdout_mask: common::small_prog::mask_banner,
    });
}

/// Defined behaviour (BUGS.md, `maxjoinsize`): the Y range starts at the Y
/// center.  For tomograms much wider than tall, native starts it at the X
/// center, which is above every corner, and reports a Y size of nx/2.
#[test]
fn y_range_starts_at_the_y_center() {
    let dir = std::env::temp_dir().join(format!("imod-rs-maxjoinsize-y-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("w.raw"), vec![0u8; 100 * 20 * 4 * 2]).unwrap();
    let made = common::imod_cmd("raw2mrc")
        .current_dir(&dir)
        .args(["-x", "100", "-y", "20", "-z", "2", "-t", "float", "w.raw", "w.mrc"])
        .output()
        .unwrap();
    assert_eq!(made.status.code(), Some(0));
    std::fs::write(dir.join("j.info"), "w.mrc\nw.mrc\n").unwrap();
    std::fs::write(
        dir.join("j.tomoxg"),
        "   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n",
    )
    .unwrap();
    let output = common::imod_cmd("maxjoinsize")
        .current_dir(&dir)
        .args(["2", "0", "j"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Maximum size required:      100       20\nOffset needed to center:       0       0\n"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
