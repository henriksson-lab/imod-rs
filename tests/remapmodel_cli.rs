//! Native-golden coverage for `remapmodel` (`IMOD/flib/model/remapmodel.f90`).
//!
//! Every row of `fixtures/remapmodel/cases.tsv` was run through the native reference
//! program by `fixtures/make-remapmodel-goldens.sh` (inputs from
//! `fixtures/make-etomo-prog-inputs.sh`), in a fresh directory holding copies
//! of the fixture inputs, with standard output captured through a pipe.  Exit
//! status, standard output and every output file must match byte for byte,
//! apart from MRC label time stamps (`common::small_prog`).

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "remapmodel",
        fixtures: "remapmodel",
        min_cases: 12,
        reconcile: false,
        env: &[],
        stdout_mask: common::mask_stamps,
    });
}

/// Runs `imod remapmodel` with `args` in a fresh directory holding the fixture
/// model and `thick.txt` with `thicknesses`, and returns the exit status,
/// standard output and the output model.
fn run_thickness(tag: &str, thicknesses: &str, args: &[&str]) -> (i32, String, Vec<u8>) {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-remapmodel-long-{}-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/remapmodel");
    std::fs::copy(fixtures.join("r1.mod"), dir.join("r1.mod")).unwrap();
    std::fs::write(dir.join("thick.txt"), thicknesses).unwrap();
    let output = common::imod_cmd("remapmodel")
        .current_dir(&dir)
        .args(["-input", "r1.mod", "-output", "o.mod", "-thick", "thick.txt"])
        .args(args)
        .output()
        .unwrap();
    let model = std::fs::read(dir.join("o.mod")).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        model,
    )
}

/// Defined behaviour of `-long` (BUGS.md, `remapmodel`, fixed in
/// translation): the long-range thickness measured between sections 2 and 6
/// scales the Z values by its ratio to the sum of the individual thicknesses
/// of sections 2 to 5.  With thicknesses exact in binary and a long-range value
/// twice that sum, the result must equal the run without `-long` whose pixel
/// size is halved.
#[test]
fn long_range_thickness_scales_by_the_ratio_to_the_summed_sections() {
    let thicknesses = "0.5\n0.75\n0.5\n0.25\n0.5\n0.75\n0.5\n0.25\n".repeat(3);
    let thicknesses = thicknesses.as_str();
    // Sections 2..5 (thicknesses 3..6, 1-based): 0.5 + 0.25 + 0.5 + 0.75 = 2.
    let (rc_long, _, long) = run_thickness("long", thicknesses, &["-long", "4,2,6", "-pixel", "0.5"]);
    let (rc_plain, _, plain) = run_thickness("plain", thicknesses, &["-pixel", "0.25"]);
    assert_eq!((rc_long, rc_plain), (0, 0));
    assert!(!long.is_empty());
    assert_eq!(long, plain);
}

/// Defined behaviour of `-long` with a starting section not below the ending
/// one: the source's own message, exit 1.  (Native rejects every start below
/// the end instead, the inverse of its message.)
#[test]
fn long_range_thickness_rejects_a_start_not_below_the_end() {
    let thicknesses = "0.5\n0.75\n0.5\n0.25\n0.5\n0.75\n0.5\n0.25\n".repeat(3);
    let thicknesses = thicknesses.as_str();
    let (rc, stdout, _) = run_thickness("bad", thicknesses, &["-long", "4,6,2", "-pixel", "0.5"]);
    assert_eq!(rc, 1);
    assert!(
        stdout.contains(
            "ERROR: REMAPMODEL - Starting section number for long range thickness is larger than ending section"
        ),
        "{stdout}"
    );
}
