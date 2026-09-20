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
    assert!(String::from_utf8_lossy(&result.stdout).contains("Directives all seem OK"));
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
    assert!(String::from_utf8_lossy(&result.stdout).contains("Directives all seem OK"));
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
    assert_eq!(
        std::fs::read_to_string(&marker).unwrap(),
        format!("--fromBRT\n--directive\n{}\n", directive.display())
    );
    for path in [marker, etomo, directive] {
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
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        format!(
            "ERROR: batchruntomo - Cannot find file for validating directives, {}\n",
            root.join("com/directives.csv").display()
        )
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
