mod common;

#[test]
fn usage_and_missing_command_file_follow_source_contract() {
    let usage = common::imod_cmd("submfg")
        .env("IMOD_DIR", "/fixture/imod")
        .output()
        .expect("run submfg");
    assert!(usage.status.success());
    assert!(String::from_utf8_lossy(&usage.stdout).contains("Usage:  submfg"));
    assert!(
        String::from_utf8_lossy(&usage.stdout)
            .contains("Keep backslashes instead of converting to forward slashes'")
    );

    let missing = common::imod_cmd("submfg")
        .env("IMOD_DIR", "/fixture/imod")
        .arg("does-not-exist")
        .output()
        .expect("run submfg missing input");
    assert_eq!(missing.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&missing.stdout)
            .contains("Neither does-not-exist.com nor does-not-exist.pcm exists")
    );
    assert!(missing.stderr.is_empty());
}

#[test]
fn option_value_error_has_submfg_prefix() {
    let output = common::imod_cmd("submfg")
        .env("IMOD_DIR", "/fixture/imod")
        .args(["-n", "not-an-integer", "unused"])
        .output()
        .expect("run submfg");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: submfg - Converting \"nice\" value (not-an-integer) to integer\n"
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn submfg_missing_option_value_reports_source_no_command_error_on_stdout() {
    let output = common::imod_cmd("submfg")
        .env("IMOD_DIR", "/fixture/imod")
        .arg("-n")
        .output()
        .expect("run submfg with missing nice value");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: submfg - No command file was entered\n"
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn submfg_unrecognized_option_uses_source_pip_stdout_route() {
    let output = common::imod_cmd("submfg")
        .env("IMOD_DIR", "/fixture/imod")
        .arg("-unknown")
        .output()
        .expect("run submfg with unrecognized option");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: submfg - Unrecognized argument -unknown\n"
    );
    assert!(output.stderr.is_empty());
}

#[cfg(unix)]
#[test]
fn submfg_resolves_vmstopy_from_imod_bin_and_runs_generated_command() {
    use std::os::unix::fs::PermissionsExt;

    let root = std::env::temp_dir().join(format!("imod-rs-submfg-launch-{}", std::process::id()));
    let bin = root.join("bin");
    let com = root.join("fixture.com");
    let vmstopy_args = root.join("vmstopy-arguments");
    let generated = root.join("generated-command");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(&com, "$ fake command file\n").unwrap();
    let python = bin.join("python");
    std::fs::write(&python, "#!/bin/sh\nexec /usr/bin/python3 \"$@\"\n").unwrap();
    let mut permissions = std::fs::metadata(&python).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&python, permissions).unwrap();
    let vmstopy = bin.join("vmstopy");
    std::fs::write(
        &vmstopy,
        "#!/bin/sh\nprintf '%s\\n%s\\n%s\\n' \"$1\" \"$2\" \"$3\" > \"$VMSTOPY_MARKER\"\nprintf 'import os\\nopen(os.environ[\"SUBMFG_MARKER\"], \"w\").write(\"generated\")\\n' > \"$3\"\n",
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&vmstopy).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&vmstopy, permissions).unwrap();
    let result = common::imod_cmd("submfg")
        .current_dir(&root)
        .env("IMOD_DIR", &root)
        .env("PATH", "/usr/bin:/bin")
        .env("VMSTOPY_MARKER", &vmstopy_args)
        .env("SUBMFG_MARKER", &generated)
        .arg(&com)
        .output()
        .expect("run submfg with isolated installed vmstopy");
    assert!(result.status.success(), "{:?}", result);
    let arguments = std::fs::read_to_string(&vmstopy_args).unwrap();
    let values = arguments.lines().collect::<Vec<_>>();
    assert_eq!(values[0], com.to_string_lossy());
    assert_eq!(values[1], root.join("fixture.log").to_string_lossy());
    assert!(values[2].starts_with("submtemp."));
    assert_eq!(std::fs::read_to_string(&generated).unwrap(), "generated");
    for path in [generated, vmstopy_args, vmstopy, python, com] {
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_dir(bin).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn subm_requires_imod_dir_before_launching_background_process() {
    let output = common::imod_cmd("subm")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run subm");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: subm -  IMOD_DIR is not defined!\n"
    );
}

#[cfg(unix)]
#[test]
fn subm_launches_imod_bin_submfg_and_routes_child_stderr_to_stdout() {
    use std::os::unix::fs::PermissionsExt;

    let root = std::env::temp_dir().join(format!("imod-rs-subm-launch-{}", std::process::id()));
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let program = bin.join("submfg");
    std::fs::write(
        &program,
        "#!/bin/sh\nprintf 'subm fixture stdout: %s\\n' \"$1\"\nprintf 'subm fixture stderr: %s\\n' \"$1\" >&2\n",
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&program).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&program, permissions).unwrap();
    let result = common::imod_cmd("subm")
        .env("IMOD_DIR", &root)
        .env("PATH", "/usr/bin:/bin")
        .arg("fixture.com")
        .output()
        .expect("run subm with isolated installed submfg");
    assert!(result.status.success());
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "subm fixture stdout: fixture.com\nsubm fixture stderr: fixture.com\n"
    );
    assert!(result.stderr.is_empty());
    std::fs::remove_file(program).unwrap();
    std::fs::remove_dir(bin).unwrap();
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn real_imod_com_fixture_reaches_vmstopy_boundary() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/com/tilt.com");
    assert!(
        fixture.is_file(),
        "bundled IMOD command fixture must be present"
    );
    let output = common::imod_cmd("submfg")
        .env("IMOD_DIR", "/fixture/imod")
        .arg(&fixture)
        .output()
        .expect("run submfg against real command fixture");
    // The fixture is deliberately allowed to reach the external `vmstopy`
    // boundary; the translator is an installed IMOD program, not Rust code.
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains(&format!("Error executing {}", fixture.display()))
    );
}

#[test]
fn missing_com_and_pcm_root_uses_source_exiterror_stdout() {
    let root =
        std::env::temp_dir().join(format!("imod-rs-submfg-no-command-{}", std::process::id()));
    let source = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("IMOD");
    let result = common::imod_cmd("submfg")
        .env("IMOD_DIR", source)
        .arg(&root)
        .output()
        .expect("run submfg with a missing command-file root");
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        format!(
            "ERROR: submfg - Neither {}.com nor {}.pcm exists\n",
            root.display(),
            root.display()
        )
    );
    assert!(result.stderr.is_empty());
}
