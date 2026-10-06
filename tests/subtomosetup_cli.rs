//! Native-golden coverage for `subtomosetup` (`IMOD/pysrc/subtomosetup`, translated in
//! `src/imod/pysrc/subtomosetup.rs`).
//!
//! Every row of `fixtures/subtomosetup/cases.tsv` was run through the native Python
//! script by `fixtures/make-subtomosetup-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn subtomosetup_matches_native_goldens() {
    let failures = pysetup_common::run_cases("subtomosetup");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Lays out `fixtures/subtomosetup/inputs/ds` (with `extra` `(source,
/// target)` copies from `inputs/`) in a fresh directory and runs `imod
/// subtomosetup` there; returns the exit status, stdout and the directory.
fn run_defined(
    name: &str,
    extra: &[(&str, &str)],
    args: &[&str],
) -> (i32, String, std::path::PathBuf) {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let inputs = root.join("fixtures/subtomosetup/inputs");
    let work = std::env::temp_dir().join(format!(
        "imod-rs-subtomosetup-defined-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for entry in std::fs::read_dir(inputs.join("ds")).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, work.join(path.file_name().unwrap())).unwrap();
    }
    for (source, target) in extra {
        std::fs::copy(inputs.join(source), work.join(target)).unwrap();
    }
    // The script tells a model from a point file by running `imodinfo`
    // through the shell (`runcmd` with `inStderr`), so ours goes on PATH as
    // in the table cases (`own_commands`).
    common::imod_link("imodinfo");
    let path = format!(
        "{}:{}",
        common::command_link_directory().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let output = common::imod_cmd("subtomosetup")
        .current_dir(&work)
        .env("PATH", path)
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

const BASE: [&str; 10] = [
    "-root",
    "g",
    "-volume",
    "v.mrc",
    "-size",
    "16,10,12",
    "-center",
    "pts.txt",
    "-reorient",
    "0",
];

/// `BUGS.md` (ctf3dsetup/subtomosetup `AxisZShift`), defined behaviour: with
/// `-adjust` and no two-value SHIFT entry in tilt.com, native dies with a
/// NameError (`startingZshift`); here the starting Z shift is tilt's default
/// 0, so the command files equal those made from a tilt.com with `SHIFT 0 0`
/// (which the script deletes and re-adds anyway).
#[test]
fn adjust_without_shift_entry_uses_zero() {
    let mut args = BASE.to_vec();
    args.extend(["-zlevels", "2", "-adjust", "-proc", "1"]);
    let (rc_plain, out_plain, plain) = run_defined("noshift", &[], &args);
    let (rc_zero, out_zero, zero) = run_defined("shift0", &[("tiltshift0.com", "tilt.com")], &args);
    assert_eq!((rc_plain, rc_zero), (0, 0), "{out_plain}\n{out_zero}");
    assert_eq!(out_plain, out_zero);
    let mut names = std::fs::read_dir(&zero)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.starts_with("tilt-sub"))
        .collect::<Vec<_>>();
    names.sort();
    assert!(names.len() > 3, "{names:?}");
    for name in names {
        assert_eq!(
            std::fs::read(plain.join(&name)).unwrap(),
            std::fs::read(zero.join(&name)).unwrap(),
            "{name}"
        );
    }
    let _ = std::fs::remove_dir_all(plain);
    let _ = std::fs::remove_dir_all(zero);
}

/// `BUGS.md` (subtomosetup `-proc 0`), defined behaviour: `-proc 0` with no
/// GPU for Tilt gives zero chunks and native dies with a ZeroDivisionError;
/// here it is an error with exit status 1.
#[test]
fn proc_zero_without_gpu_is_an_error() {
    let mut args = BASE.to_vec();
    args.extend(["-proc", "0"]);
    let (rc, out, work) = run_defined("proc0", &[], &args);
    assert_eq!(rc, 1);
    assert!(
        out.contains(
            "ERROR: subtomosetup - The runs cannot be divided into chunks for 0 processors"
        ),
        "{out}"
    );
    let _ = std::fs::remove_dir_all(work);
}

/// `-objects` with an object the model does not have: `imodextract` (run in
/// process) fails and the script reports it, exit 1, with no command files
/// and no temporary model left (native identical but for the PID in the
/// temporary file name, which is why this case is not in the table).
#[test]
fn objects_out_of_range_reports_imodextract_error() {
    let (rc, out, work) = run_defined(
        "objbad",
        &[("pts3.mod", "pts3.mod")],
        &[
            "-root",
            "g",
            "-volume",
            "v.mrc",
            "-size",
            "16,10,12",
            "-center",
            "pts3.mod",
            "-objects",
            "5",
            "-reorient",
            "0",
        ],
    );
    assert_eq!(rc, 1, "{out}");
    assert!(
        out.starts_with("ERROR: imodextract -  Invalid object number 5\n"),
        "{out}"
    );
    assert!(
        out.contains("ERROR: subtomosetup - imodextract \"5\" \"pts3.mod\" \"pts3.mod.obj."),
        "{out}"
    );
    let left = std::fs::read_dir(&work)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| {
            name.starts_with("tilt-sub") || name.contains(".obj.") || name.contains(".pt.")
        })
        .collect::<Vec<_>>();
    assert!(left.is_empty(), "{left:?}");
    let _ = std::fs::remove_dir_all(work);
}
