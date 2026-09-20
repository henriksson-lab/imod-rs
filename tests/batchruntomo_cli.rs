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
