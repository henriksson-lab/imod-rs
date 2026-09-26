//! Native-golden coverage for `copytomocoms` (`IMOD/pysrc/copytomocoms`, translated in
//! `src/imod/pysrc/copytomocoms.rs`).
//!
//! Every row of `fixtures/copytomocoms/cases.tsv` was run through the native Python
//! script by `fixtures/make-copytomocoms-goldens.sh`; see `tests/pysetup_common` for
//! what is compared.  The wider differential behind these goldens is recorded
//! in `TODO.md` (single-axis setup scripts).

mod common;
mod pysetup_common;

#[test]
fn copytomocoms_matches_native_goldens() {
    let failures = pysetup_common::run_cases("copytomocoms");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn copytomocoms_requires_imod_dir() {
    let output = common::imod_cmd("copytomocoms")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run copytomocoms without IMOD_DIR");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: copytomocoms -  IMOD_DIR is not defined!\n"
    );
}

/// Runs `copytomocoms` in a fresh directory holding the named fixture inputs
/// (`src:dst`), returning the output and the directory.
fn run_in(
    name: &str,
    inputs: &[(&str, &str)],
    args: &[&str],
) -> (std::process::Output, std::path::PathBuf) {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!(
        "imod-rs-copytomocoms-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for (source, target) in inputs {
        std::fs::copy(
            root.join("fixtures/copytomocoms/inputs").join(source),
            work.join(target),
        )
        .unwrap();
    }
    let output = common::imod_cmd("copytomocoms")
        .args(args)
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

// Fixed in translation (BUGS.md): native reads `DoseSymmetricAngle` for the B
// axis too (`copytomocoms:316`) and assigns `bangles = angles`
// (`copytomocoms:552`), so -bdosesym and -bangles are ignored.  The B axis
// now gets its own angles and dose-symmetric offset.
#[test]
fn b_axis_uses_bangles_and_bdosesym() {
    let (output, work) = run_in(
        "bopts",
        &[("b.st", "ba.st"), ("b.st", "bb.st")],
        &[
            "-name",
            "b",
            "-dual",
            "-pixel",
            "1",
            "-gold",
            "5",
            "-rotation",
            "-85.3",
            "-angles",
            "-50,-30,-10,10,30,50",
            "-dosesym",
            "2",
            "-bangles",
            "-45,-25,-5,15,35,55",
            "-bdosesym",
            "4",
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let read = |file: &str| std::fs::read_to_string(work.join(file)).unwrap();
    assert!(
        read("xcorra.com").contains("AngleOffset\t-2.0")
            || read("xcorra.com").contains("AngleOffset -2.0")
    );
    let xcorrb = read("xcorrb.com");
    assert!(xcorrb.contains("-4.0"), "{xcorrb}");
    assert!(xcorrb.contains("-45,-25,-5,15,35,55"), "{xcorrb}");
    assert!(!xcorrb.contains("-50,-30,-10,10,30,50"), "{xcorrb}");
    let _ = std::fs::remove_dir_all(&work);
}

// Fixed in translation (BUGS.md): with `-gradient` and no stack native reads
// the never-assigned `pixelx` (`copytomocoms:879`, NameError).  An unknown
// header pixel size is treated like an unset one, so the entered -pixel goes
// to the deferred `extractmagrad` input.
#[test]
fn gradient_without_stack_defers_extractmagrad() {
    let (output, work) = run_in(
        "gradnostack",
        &[("b.rawtlt", "b.rawtlt")],
        &[
            "-name",
            "b",
            "-pixel",
            "1.5",
            "-gold",
            "5",
            "-rotation",
            "-85.3",
            "-userawtlt",
            "-gradient",
            "nosuch.dat",
            "-xsize",
            "200",
            "-ysize",
            "300",
            "-stackext",
            "st",
        ],
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let later = std::fs::read_to_string(work.join("laterbsetup.com")).unwrap();
    assert!(later.contains("$extractmagrad -StandardInput"), "{later}");
    assert!(later.contains("PixelSize 1.5"), "{later}");
    assert!(!String::from_utf8_lossy(&output.stderr).contains("NameError"));
    let _ = std::fs::remove_dir_all(&work);
}

// Fixed in translation (BUGS.md): removing `LOG` with no `clip stat` sample
// (a stack under 1 MB) makes native divide by a zero SD
// (`copytomocoms:1350`, ZeroDivisionError); the translation leaves the linear
// scale unadjusted (1000 / 5000).  And native's `atof` (undefined in Python 3,
// `copytomocoms:1336`) means a directive SCALE already below 3 never stops the
// division by 5000; the translation converts it, so the default file keeps
// the undivided scale.
#[test]
fn log_removal_scale_without_sd_and_with_linear_scale() {
    let inputs = [("b.st", "b.st"), ("b.rawtlt", "b.rawtlt")];
    let base = [
        "-name",
        "b",
        "-pixel",
        "1",
        "-gold",
        "5",
        "-rotation",
        "-85.3",
        "-userawtlt",
        "-one",
        "comparam.tilt.tilt.LOG=",
    ];
    let (output, work) = run_in("lognosd", &inputs, &base);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let tilt = std::fs::read_to_string(work.join("tilt.com")).unwrap();
    assert!(tilt.contains("SCALE 0 0.200"), "{tilt}");
    let _ = std::fs::remove_dir_all(&work);

    let mut args = base.to_vec();
    args.extend(["-one", "comparam.tilt.tilt.SCALE=0 0.5"]);
    let (output, work) = run_in("loglinear", &inputs, &args);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let default = std::fs::read_to_string(work.join("dfltcoms/tilt.com")).unwrap();
    assert!(default.contains("SCALE 0 1000"), "{default}");
    let tilt = std::fs::read_to_string(work.join("tilt.com")).unwrap();
    assert!(tilt.contains("SCALE\t0 0.5"), "{tilt}");
    let _ = std::fs::remove_dir_all(&work);
}
