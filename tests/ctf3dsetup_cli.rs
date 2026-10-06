//! Native-golden coverage for `ctf3dsetup` (`IMOD/pysrc/ctf3dsetup`, translated in
//! `src/imod/pysrc/ctf3dsetup.rs`).
//!
//! Every row of `fixtures/ctf3dsetup/cases.tsv` was run through the native Python
//! script by `fixtures/make-ctf3dsetup-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.
//!
//! Pruned 2026-10-06 (manifest size): `par3`, `raw`, `adjust`, `rawerase` and
//! `filt_erase2` stay in cases.tsv as `#full` rows (their options are covered
//! by `procs1`/`boundary`, `raw1`, `adjust_shift`, `rawerase_warn`/`_par` and
//! `filt_erase1`; `-perproc` is reached only under FULL=1 now).

mod common;
mod pysetup_common;

#[test]
fn ctf3dsetup_matches_native_goldens() {
    let failures = pysetup_common::run_cases("ctf3dsetup");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Lays out `fixtures/ctf3dsetup/inputs/ds` (with `extra` `(source, target)`
/// overrides from `inputs/`) in a fresh directory and runs `imod ctf3dsetup`
/// there; returns the exit status, stdout and the directory.
fn run_defined(
    name: &str,
    extra: &[(&str, &str)],
    args: &[&str],
) -> (i32, String, std::path::PathBuf) {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/ctf3dsetup/inputs");
    let work = std::env::temp_dir().join(format!(
        "imod-rs-ctf3dsetup-defined-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for entry in std::fs::read_dir(inputs.join("ds")).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, work.join(path.file_name().unwrap())).unwrap();
    }
    for (source, target) in extra {
        if source.is_empty() {
            std::fs::remove_file(work.join(target)).unwrap();
        } else {
            std::fs::copy(inputs.join(source), work.join(target)).unwrap();
        }
    }
    let output = common::imod_cmd("ctf3dsetup")
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .env_remove("PARALLEL_BOUNDARY_SIZE")
        .args(args)
        .output()
        .expect("run imod");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        work,
    )
}

/// `BUGS.md` (ctf3dsetup/subtomosetup `AxisZShift`), defined behaviour: with
/// `-adjust` and no `AxisZShift` in align.com, native dies with a TypeError
/// (`None + 0`); here the missing shift is tiltalign's default 0, so the
/// command files are those made without `-adjust`.
#[test]
fn adjust_without_axis_z_shift_uses_zero() {
    let args = ["-slabs", "3", "-procs", "1", "tilt.com"];
    let (rc_plain, _, plain) = run_defined("plain", &[], &args);
    let (rc_adjust, _, adjust) = run_defined(
        "noz",
        &[("alignnoz.com", "align.com")],
        &["-slabs", "3", "-adjust", "-procs", "1", "tilt.com"],
    );
    assert_eq!((rc_plain, rc_adjust), (0, 0));
    for number in 1..=6 {
        let file = format!("ctf3d-{number:03}-sync.com");
        assert_eq!(
            std::fs::read(plain.join(&file)).unwrap(),
            std::fs::read(adjust.join(&file)).unwrap(),
            "{file}"
        );
    }
    let _ = std::fs::remove_dir_all(plain);
    let _ = std::fs::remove_dir_all(adjust);
}

/// `BUGS.md` (tomocoords `getCTFoptionsCheckIfCorrected`), defined
/// behaviour: with no `InputStack` in the CTF command file native dies in
/// `os.path.exists(None)`; here the source's own "does not exist" error.
#[test]
fn missing_ctf_input_stack_is_an_error() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let text =
        std::fs::read_to_string(root.join("fixtures/ctf3dsetup/inputs/ds/ctfcorrection.com"))
            .unwrap()
            .replace("InputStack  b_ali.mrc\n", "");
    let (_, _, work) = run_defined("noinput", &[], &["-help"]);
    std::fs::write(work.join("ctfcorrection.com"), text).unwrap();
    let output = common::imod_cmd("ctf3dsetup")
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .args(["-slabs", "3", "tilt.com"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("Input file in ctf correction command file, None, does not exist"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let _ = std::fs::remove_dir_all(work);
}

/// `BUGS.md` (ctf3dsetup raw `PixelSize`), defined behaviour: from raw
/// images with no `PixelSize` in the CTF command file native divides `None`
/// (TypeError); here the source's "Cannot find needed information" error.
#[test]
fn raw_without_ctf_pixel_size_is_an_error() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let text =
        std::fs::read_to_string(root.join("fixtures/ctf3dsetup/inputs/ds/ctfcorrection.com"))
            .unwrap()
            .replace("\nPixelSize\t1.0\n", "\n");
    let (_, _, work) = run_defined("nopixel", &[], &["-help"]);
    std::fs::write(work.join("ctfcorrection.com"), text).unwrap();
    let output = common::imod_cmd("ctf3dsetup")
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .args(["-slabs", "3", "-unaligned", "-procs", "1", "tilt.com"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains(
            "Cannot find needed information in ctfcorrection.com (PixelSize, input or output file)"
        ),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let _ = std::fs::remove_dir_all(work);
}
