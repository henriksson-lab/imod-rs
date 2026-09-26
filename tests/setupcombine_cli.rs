//! Native-golden coverage for `setupcombine` (`IMOD/pysrc/setupcombine`, translated in
//! `src/imod/pysrc/setupcombine.rs`).
//!
//! Every row of `fixtures/setupcombine/cases.tsv` was run through the native Python
//! script by `fixtures/make-setupcombine-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens (with the
//! native combine programs on `PATH`) is recorded in `TODO.md` (dual-axis
//! combine scripts).

mod common;
mod pysetup_common;

#[test]
fn setupcombine_matches_native_goldens() {
    let failures = pysetup_common::run_cases("setupcombine");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn setupcombine_requires_imod_dir() {
    let output = common::imod_cmd("setupcombine")
        .arg("-info")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run setupcombine without IMOD_DIR");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: setupcombine -  IMOD_DIR is not defined!\n"
    );
}

fn setupcombine_in(
    name: &str,
    tilt_coms: Option<(&str, &str)>,
) -> (std::process::Output, std::path::PathBuf) {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/setupcombine/inputs");
    let work = std::env::temp_dir().join(format!(
        "imod-rs-setupcombine-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for file in [
        "ga.rec", "gb.rec", "ga.st", "gb.st", "ga.xf", "gb.xf", "ga.tlt", "gb.tlt",
    ] {
        std::fs::copy(inputs.join(file), work.join(file)).unwrap();
    }
    if let Some((tilta, tiltb)) = tilt_coms {
        std::fs::write(work.join("tilta.com"), tilta).unwrap();
        std::fs::write(work.join("tiltb.com"), tiltb).unwrap();
    }
    let output = common::imod_cmd("setupcombine")
        .args([
            "-name",
            "g",
            "-surfaces",
            "1",
            "-zlimits",
            "3,17",
            "-stackext",
            "st",
        ])
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .env_remove("IMOD_OUTPUT_FORMAT")
        .env_remove("TEST_NAMING_STYLE")
        .env_remove("TEST_USE_PCM_FOR_COM")
        .output()
        .unwrap();
    (output, work)
}

// Fixed in translation (BUGS.md): without both tilt command files native
// reads the unset `xshifta` (`setupcombine:444`) and dies with NameError, no
// files written.  The translation defaults the shifts (and slice limits) to 0
// like the other values of that branch, warns, and sets up the combine.
#[test]
fn missing_tilt_command_files_warn_and_carry_on() {
    let (output, work) = setupcombine_in("notilt", None);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let both = format!("{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert!(
        both.contains("CANNOT FIND tilta.com or tiltb.com"),
        "{both}"
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("Traceback"));
    assert!(work.join("combine.com").exists());
    assert!(work.join("solvematch.com").exists());
    let _ = std::fs::remove_dir_all(&work);
}

// Fixed in translation (BUGS.md): `IMAGEBINNED 0` makes native divide by zero
// (`setupcombine:43`); like `tilt` (`tilt.cpp:3119`) the translation takes a
// binning below 1 as 1, so the result equals that of `IMAGEBINNED 1`.
#[test]
fn image_binned_zero_is_taken_as_one() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/setupcombine/inputs");
    let tilta = std::fs::read_to_string(inputs.join("tilta.com")).unwrap();
    let tiltb = std::fs::read_to_string(inputs.join("tiltb.com")).unwrap();
    let with = |binning: &str| {
        (
            tilta
                .replace("IMAGEBINNED 2", &format!("IMAGEBINNED {binning}"))
                .replace("IMAGEBINNED 1", &format!("IMAGEBINNED {binning}")),
            tiltb
                .replace("IMAGEBINNED 2", &format!("IMAGEBINNED {binning}"))
                .replace("IMAGEBINNED 1", &format!("IMAGEBINNED {binning}")),
        )
    };
    let (zero_a, zero_b) = with("0");
    assert_ne!(zero_a, tilta);
    let (one_a, one_b) = with("1");
    let (zero, zero_work) = setupcombine_in("bin0", Some((&zero_a, &zero_b)));
    let (one, one_work) = setupcombine_in("bin1", Some((&one_a, &one_b)));
    assert_eq!(zero.status.code(), Some(0), "{zero:?}");
    assert_eq!(zero.stdout, one.stdout);
    for file in ["solvematch.com", "patchcorr.com", "combine.com"] {
        assert_eq!(
            std::fs::read(zero_work.join(file)).unwrap(),
            std::fs::read(one_work.join(file)).unwrap(),
            "{file}"
        );
    }
    let _ = std::fs::remove_dir_all(&zero_work);
    let _ = std::fs::remove_dir_all(&one_work);
}
