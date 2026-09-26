//! End-to-end command fixture for `IMOD/pysrc/batchruntomo`.

mod common;

use imod_rs::imod::pysrc::comchanger::modify_for_change_list;
use std::path::PathBuf;

#[test]
fn batchruntomo_requires_imod_dir_before_parsing_arguments() {
    let result = common::imod_cmd("batchruntomo")
        .env_remove("IMOD_DIR")
        .arg("-help")
        .output()
        .expect("run batchruntomo without runtime environment");
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "ERROR: batchruntomo -  IMOD_DIR is not defined!\n"
    );
    assert!(result.stderr.is_empty());
}

#[test]
fn root_name_value_is_not_treated_as_an_unnamed_directive_file() {
    // Native `batchruntomo -root sample` reaches source lines 5221-5224:
    // RootName is consumed by PIP, leaving no directive file, and exitError
    // writes this diagnostic to stdout.
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("IMOD");
    let result = common::imod_cmd("batchruntomo")
        .env("IMOD_DIR", source)
        .args(["-root", "sample"])
        .output()
        .expect("run batchruntomo with a root name but no directive");
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "ERROR: batchruntomo - You must enter at least one directive file\n"
    );
    assert!(result.stderr.is_empty());
}

#[test]
fn validates_the_bundled_batch_directive_file() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("IMOD");
    let directive = source.join("Etomo/tests/batch.adoc");
    let result = common::imod_cmd("batchruntomo")
        .env("IMOD_DIR", &source)
        .args(["-validation", "1", "-directive"])
        .arg(&directive)
        .output()
        .expect("run batchruntomo");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    // Verified against the authority, `python3 IMOD/pysrc/batchruntomo` with
    // `PYTHONPATH=IMOD/pysrc`, on this same directive file: it exits 0 and
    // ends with these two lines.  It does NOT print "Directives all seem OK"
    // — that string came from the scaffold this module replaced on
    // 2026-09-20, and the earlier assertion pinned the scaffold.
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("ABORT SET: Bad directives"), "{stdout}");
    assert!(
        stdout.contains("Batch run finished; failures occurred for 1 datasets"),
        "{stdout}"
    );
}

#[test]
fn batchruntomo_pid_option_reports_source_pid_before_validating_real_directive() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("IMOD");
    let directive = source.join("Etomo/tests/batch.adoc");
    let result = common::imod_cmd("batchruntomo")
        .env("IMOD_DIR", &source)
        .args(["-PID", "-validation", "1", "-directive"])
        .arg(&directive)
        .output()
        .expect("run batchruntomo PID launcher fixture");
    assert!(result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    let pid = stderr
        .strip_prefix("Python PID: ")
        .and_then(|line| line.trim().parse::<u32>().ok());
    assert!(pid.is_some(), "{stderr}");
    // As above: the Python prints the abort/summary pair, not "Directives all
    // seem OK".
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("ABORT SET: Bad directives"), "{stdout}");
}

#[cfg(unix)]
#[test]
fn batchruntomo_validation_zero_uses_nonvalidation_launcher_branch() {
    use std::os::unix::fs::PermissionsExt;

    let root = std::env::temp_dir().join(format!("imod-rs-batchruntomo-v0-{}", std::process::id()));
    let bin = root.join("bin");
    let com = root.join("com");
    let directive = root.join("fixture.adoc");
    let marker = root.join("etomo-arguments");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::create_dir_all(&com).unwrap();
    std::fs::write(com.join("directives.csv"), "").unwrap();
    std::fs::write(&directive, "setupset.copyarg.name = fixture\n").unwrap();
    let etomo = bin.join("etomo");
    std::fs::write(
        &etomo,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$BRT_MARKER\"\n",
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&etomo).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&etomo, permissions).unwrap();
    let result = common::imod_cmd("batchruntomo")
        .env("IMOD_DIR", &root)
        .env("BRT_MARKER", &marker)
        .args(["-validation", "0", "-directive"])
        .arg(&directive)
        .output()
        .expect("run batchruntomo validation zero launcher fixture");
    assert!(result.status.success(), "{:?}", result);
    // Checked against `python3 IMOD/pysrc/batchruntomo` on this exact
    // fixture: it also exits 0, also never runs the stub `etomo` (the marker
    // file is not created), and also ends with "Batch run finished; failures
    // occurred for 1 datasets".  The previous expectation — that `etomo` was
    // invoked with `--fromBRT --directive <file>` — described the scaffold
    // this module replaced on 2026-09-20.
    assert!(
        !marker.exists(),
        "the Python does not reach etomo for this fixture"
    );
    assert!(
        String::from_utf8_lossy(&result.stdout)
            .contains("Batch run finished; failures occurred for 1 datasets"),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    for path in [etomo, directive] {
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_file(com.join("directives.csv")).unwrap();
    std::fs::remove_dir(bin).unwrap();
    std::fs::remove_dir(com).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[cfg(unix)]
#[test]
fn batchruntomo_validation_zero_requires_source_directives_csv_before_etomo() {
    use std::os::unix::fs::PermissionsExt;

    let root = std::env::temp_dir().join(format!(
        "imod-rs-batchruntomo-validation-table-{}",
        std::process::id()
    ));
    let bin = root.join("bin");
    let directive = root.join("fixture.adoc");
    let marker = root.join("etomo-ran");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(&directive, "setupset.copyarg.name = fixture\n").unwrap();
    let etomo = bin.join("etomo");
    std::fs::write(&etomo, "#!/bin/sh\ntouch \"$BRT_MARKER\"\n").unwrap();
    let mut permissions = std::fs::metadata(&etomo).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&etomo, permissions).unwrap();
    let result = common::imod_cmd("batchruntomo")
        .env("IMOD_DIR", &root)
        .env("BRT_MARKER", &marker)
        .args(["-validation", "0", "-directive"])
        .arg(&directive)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    // The Python puts this on **stdout**, not stderr — `exitError` in
    // `IMOD/pysrc/imodpy.py` writes there — and exits 1.  Verified by running
    // `python3 IMOD/pysrc/batchruntomo` on this fixture.
    //
    // Its stderr is empty, while ours carries one extra line:
    //   ERROR: AdocRead - Error opening autodoc file <IMOD_DIR>/com/progDefaults.adoc
    // That is not a defect in this module.  `IMOD/pysrc/pip.py` wraps the
    // defaults-file open in a bare `try:` (`pip.py:1079-1084`) and silently
    // swallows a missing file, whereas the C's `PipReadProgDefaults`
    // (`parse_params.c`) calls `AdocRead`, which prints through
    // `b3dError(stderr, …)` — native `newstack` and `fakevolume` both emit
    // exactly this line under the same conditions, so `parse_params.rs` is
    // faithful.  The gap is that `src/imod/pysrc/pip.rs` forwards to the C's
    // PIP instead of translating `pip.py`; it is recorded in TOFIX.md.
    // The PIP banner and the "To quit all processing" line precede it on both
    // sides, so the error is checked as the tail of stdout.
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.ends_with(&format!(
            "ERROR: batchruntomo - Cannot find file for validating directives, {}\n",
            root.join("com/directives.csv").display()
        )),
        "{stdout}"
    );
    assert!(!marker.exists());
    for path in [etomo, directive] {
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_dir(bin).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn changes_a_real_imod_command_file_block() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("IMOD/com/tilt.com");
    let lines = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let changes = vec![vec![
        "tilt".to_owned(),
        "tilt".to_owned(),
        "THICKNESS".to_owned(),
        "250".to_owned(),
    ]];
    let changed = modify_for_change_list(&lines, "tilt", "", &changes, false).unwrap();
    assert!(changed.iter().any(|line| line == "THICKNESS\t250"));
}

/// Writes an 8x6x2 float MRC through our `raw2mrc` into `dir/<name>.mrc`
/// and replaces its titles with `labels` (each blank-padded to 80 bytes, as
/// SerialEM writes them).
fn titled_stack(dir: &std::path::Path, name: &str, labels: &[&str]) -> PathBuf {
    let raw = dir.join("src.raw");
    let pixels: Vec<u8> = (0..8 * 6 * 2)
        .flat_map(|index| (index as f32 * 0.25).to_le_bytes())
        .collect();
    std::fs::write(&raw, pixels).unwrap();
    let mrc = dir.join(format!("{name}.mrc"));
    let status = common::imod_cmd("raw2mrc")
        .args(["-x", "8", "-y", "6", "-z", "2", "-t", "float"])
        .arg(&raw)
        .arg(&mrc)
        .output()
        .unwrap()
        .status;
    assert!(status.success());
    let mut bytes = std::fs::read(&mrc).unwrap();
    bytes[220..224].copy_from_slice(&(labels.len() as i32).to_le_bytes());
    for (index, label) in labels.iter().enumerate() {
        let mut padded = label.as_bytes().to_vec();
        padded.resize(80, b' ');
        bytes[224 + 80 * index..224 + 80 * (index + 1)].copy_from_slice(&padded[..80]);
    }
    std::fs::write(&mrc, bytes).unwrap();
    mrc
}

/// `imodpy.getmrc(angleLineValues=True)` crashes on Python 3 (`len()` of a
/// `filter` object; BUGS.md, "fixed in translation").  The defined behaviour
/// is the loop the code intends, `list(multiCharSplit(line, ' ,='))`.  Every
/// expectation below was produced by the reference Python with exactly that
/// one change, running the native `header` on the same titles.
#[test]
fn getmrc_angle_line_values_defined_behaviour() {
    use imod_rs::imod::pysrc::imodpy::{MrcInfo, get_mrc};

    let dir = std::env::temp_dir().join(format!("imod-rs-brt-getmrc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // A 79-byte title (all `header` prints) whose last token is a value, and
    // one whose last token is a key: the line's newline is then a token of
    // its own only when the title is shorter than 79 bytes.
    let full = format!(
        "Tilt axis angle = 12.25 binning = 3 {} camera = 1",
        "z".repeat(79 - 36 - 11)
    );
    let key_last = format!("{:<71} binning", "Tilt axis angle = 5 junk");
    type Values = [Option<f64>; 5];
    let cases: Vec<(&str, Vec<&str>, Option<Values>)> = vec![
        (
            "serialem",
            vec!["SerialEM: Tilt axis angle = 85.3, binning = 1  spot = 8  camera = 0"],
            Some([Some(85.3), Some(1.0), Some(8.0), Some(0.0), None]),
        ),
        (
            "neg",
            vec!["Tilt axis angle = -11.5, binning = 2  spot = 2  camera = 2"],
            Some([Some(-11.5), Some(2.0), Some(2.0), Some(2.0), None]),
        ),
        (
            "full",
            vec![full.as_str()],
            Some([Some(12.25), Some(3.0), None, Some(1.0), None]),
        ),
        (
            "keylastfull",
            vec![key_last.as_str()],
            Some([Some(5.0), None, None, None, None]),
        ),
        // Python: `float('\n')` raises ValueError -> ImodpyError.
        ("keylast", vec!["Tilt axis angle = 5 binning"], None),
        ("bad", vec!["Tilt axis angle = abc"], None),
        ("badint", vec!["Tilt axis angle = 4 spot = 2.5"], None),
        ("none", vec!["Just some title"], Some([None; 5])),
        ("nolabels", vec![], Some([None; 5])),
        (
            "second",
            vec!["Generic title", "Tilt axis angle = 3.5,binning=2,bidir=-20"],
            Some([Some(3.5), Some(2.0), None, None, Some(-20.0)]),
        ),
        (
            "twolines",
            vec!["Tilt axis angle = 1.5", "Tilt axis angle = 7.5 spot = 9"],
            Some([Some(7.5), None, Some(9.0), None, None]),
        ),
        (
            "nospace",
            vec!["axis angle=44"],
            Some([Some(44.0), None, None, None, None]),
        ),
    ];
    for (name, labels, expected) in cases {
        let mrc = titled_stack(&dir, name, &labels);
        match (get_mrc(mrc.to_str().unwrap(), false, true), expected) {
            (Ok(MrcInfo::AngleLines(values)), Some(expected)) => {
                assert_eq!(values, expected, "{name}")
            }
            (Err(_), None) => {}
            (other, expected) => panic!("{name}: got {other:?}, expected {expected:?}"),
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `batchruntomo -BypassEtomo` with no `rotation` directive on a SerialEM
/// stack: the reference Python dies there with the `getmrc` TypeError and
/// exits 1.  With the defined behaviour it takes the axis angle from the
/// title and sets the data set up.  The com files and `batchruntomo.log`
/// match, byte for byte (time stamps aside), those of the reference Python
/// with the one-line `list(...)` fix, run with the native binaries.
#[test]
fn bypass_setup_reads_rotation_from_serialem_title() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("IMOD");
    let dir = std::env::temp_dir().join(format!("imod-rs-brt-rotation-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    titled_stack(
        &dir,
        "ts",
        &["SerialEM: Tilt axis angle = 85.3, binning = 1  spot = 8  camera = 0"],
    );
    std::fs::write(dir.join("ts.rawtlt"), "-3.0\n3.0\n").unwrap();
    std::fs::write(
        dir.join("dir.adoc"),
        format!(
            "setupset.copyarg.name = ts\nsetupset.copyarg.stackext = mrc\n\
             setupset.copyarg.dual = 0\nsetupset.copyarg.pixel = 1.0\n\
             setupset.copyarg.gold = 10\nsetupset.copyarg.userawtlt = 1\n\
             setupset.copyarg.buserawtlt = 1\nsetupset.datasetDirectory = {}\n",
            dir.display()
        ),
    )
    .unwrap();
    // `copytomocoms` runs as a child process, found through `PATH`.
    for command in imod_rs::imod::commands::COMMANDS {
        common::imod_link(command.name);
    }
    let path = format!(
        "{}:/usr/bin:/bin",
        common::command_link_directory().display()
    );
    let result = common::imod_cmd("batchruntomo")
        .current_dir(&dir)
        .env("PATH", path)
        .env("IMOD_DIR", &source)
        .env("AUTODOC_DIR", source.join("autodoc"))
        .args(["-directive", "dir.adoc", "-end", "0", "-bypass"])
        .output()
        .unwrap();
    common::remove_command_links();
    assert!(result.status.success(), "{result:?}");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!stderr.contains("TypeError"), "{stderr}");
    let align = std::fs::read_to_string(dir.join("align.com")).unwrap();
    assert!(align.contains("RotationAngle\t85.3\n"), "{align}");
    let ctfplotter = std::fs::read_to_string(dir.join("ctfplotter.com")).unwrap();
    assert!(ctfplotter.contains("AxisAngle\t85.3\n"), "{ctfplotter}");
    std::fs::remove_dir_all(&dir).unwrap();
}
