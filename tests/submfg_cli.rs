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

// The command file runs in process (comrun, 2026-09-26): no `vmstopy`,
// `python` or temporary script is needed, so a PATH holding none of them
// still runs it, and nothing is left behind but the log.
#[cfg(unix)]
#[test]
fn submfg_runs_command_file_without_python_or_vmstopy() {
    let root = std::env::temp_dir().join(format!("imod-rs-submfg-launch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(root.join("fixture.com"), "$echo hello from the runner\n").unwrap();
    let result = common::imod_cmd("submfg")
        .current_dir(&root)
        .env("IMOD_DIR", &root)
        .env("PATH", &bin)
        .env_remove("SUBM_MESSAGE")
        .env_remove("SUBM_LOG_TYPE")
        .arg("fixture")
        .output()
        .expect("run submfg with no python or vmstopy on PATH");
    assert!(result.status.success(), "{:?}", result);
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "Running fixture.com ... fixture.com  finished successfully\x07\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("fixture.log")).unwrap(),
        "hello from the runner\nSUCCESSFULLY COMPLETED\n"
    );
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["bin", "fixture.com", "fixture.log"]);
    std::fs::remove_dir_all(root).unwrap();
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
fn real_imod_com_fixture_fails_in_tilt_without_its_inputs() {
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/com/tilt.com");
    assert!(
        source.is_file(),
        "bundled IMOD command fixture must be present"
    );
    // A copy: the log is written next to the command file, never under IMOD/
    let root = std::env::temp_dir().join(format!("imod-rs-submfg-tilt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let fixture = root.join("tilt.com");
    std::fs::copy(&source, &fixture).unwrap();
    let output = common::imod_cmd("submfg")
        .current_dir(&root)
        .env("IMOD_DIR", "/fixture/imod")
        .arg(&fixture)
        .output()
        .expect("run submfg against real command fixture");
    // The runner runs `tilt`, which cannot open the `g5a` inputs
    assert_eq!(output.status.code(), Some(1));
    // `prnstr('Error executing ' + comname + bell)` writes to stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&format!("Error executing {}", fixture.display())));
    let log = std::fs::read_to_string(root.join("tilt.log")).unwrap();
    assert!(log.contains("ERROR:"), "{log}");
    std::fs::remove_dir_all(root).unwrap();
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

// ---------------------------------------------------------------------------
// Native differentials (2026-09-26).  The expected output below was captured
// from the Python source (`python3 IMOD/pysrc/submfg`) running the same
// command files through the vendored `IMOD/pysrc/vmstopy`, and compared byte
// for byte (stdout, every file left behind, exit status) against
// `imod submfg` over 32 cases, including `-s` through the native `vmstocsh`
// and a real `tcsh`.

struct Fixture {
    root: std::path::PathBuf,
    run: std::path::PathBuf,
}

impl Fixture {
    fn new(name: &str, files: &[(&str, &str)]) -> Fixture {
        let root = std::env::temp_dir().join(format!(
            "imod-rs-submfg-native-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let bin = root.join("bin");
        let run = root.join("run");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(root.join("imod/bin")).unwrap();
        std::fs::create_dir_all(&run).unwrap();
        // A native Python stand-in: these comparisons run on Unix only.
        #[cfg(unix)]
        std::os::unix::fs::symlink("/usr/bin/python3", bin.join("python")).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/pysrc/vmstopy"),
            bin.join("vmstopy"),
        )
        .unwrap();
        for (name, text) in files {
            let path = run.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        Fixture { root, run }
    }

    fn run(&self, args: &[&str]) -> std::process::Output {
        common::imod_cmd("submfg")
            .args(args)
            .current_dir(&self.run)
            .env("IMOD_DIR", self.root.join("imod"))
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            )
            .env(
                "PYTHONPATH",
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/pysrc"),
            )
            .env_remove("SUBM_MESSAGE")
            .env_remove("SUBM_LOG_TYPE")
            .env_remove("RUNCMD_VERBOSE")
            .output()
            .unwrap()
    }

    fn files(&self) -> Vec<String> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.run).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            if name.starts_with("submtemp.") {
                names.push("submtemp.PID".to_owned());
            } else {
                names.push(name);
            }
        }
        names.sort();
        names
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.run.join(name)).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

const OK_COM: (&str, &str) = ("ok.com", "# a comment\n$echo running ok\n$echo second\n");
const OK2_COM: (&str, &str) = ("ok2.com", "$echo ok two\n");
const FAIL_COM: (&str, &str) = (
    "fail.com",
    "$sh -c 'echo before; echo ERROR: bad thing; exit 2'\n",
);
const FAILNE_COM: (&str, &str) = (
    "failne.com",
    "$sh -c 'echo l1; echo l2; echo l3; echo l4; echo l5; exit 3'\n",
);

#[test]
fn native_runs_each_command_file_and_logs_it() {
    let fixture = Fixture::new("ok", &[OK_COM, OK2_COM]);
    let output = fixture.run(&["ok.com", "ok2"]);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Running ok.com ... ok.com  finished successfully\x07\nRunning ok2.com ... ok2.com  finished successfully\x07\n"
    );
    assert_eq!(
        fixture.read("ok.log"),
        "running ok\nsecond\nSUCCESSFULLY COMPLETED\n"
    );
    assert_eq!(fixture.files(), ["ok.com", "ok.log", "ok2.com", "ok2.log"]);
}

// Fixed in translation (BUGS.md): native leaves the previous file's
// `submtemp.<pid>` behind on these exits; the translation removes it.
#[test]
fn native_both_and_neither_exit_at_once_even_under_continue() {
    // `exitError` is a SystemExit, which the per-file `try` does not catch:
    // the program stops, and the previous file's temporary is left behind
    let fixture = Fixture::new(
        "both",
        &[
            OK_COM,
            OK2_COM,
            ("x.com", "$echo a\n"),
            ("x.pcm", "$echo b\n"),
        ],
    );
    let output = fixture.run(&["-c", "ok", "x", "ok2"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Running ok.com ... ok.com  finished successfully\x07\nERROR: submfg - Both x.com and x.pcm exist; specify which\n"
    );
    assert_eq!(
        fixture.files(),
        ["ok.com", "ok.log", "ok2.com", "x.com", "x.pcm"]
    );

    let fixture = Fixture::new("neither", &[OK_COM, OK2_COM]);
    let output = fixture.run(&["-c", "ok", "nothere", "ok2"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Running ok.com ... ok.com  finished successfully\x07\nERROR: submfg - Neither nothere.com nor nothere.pcm exists\n"
    );
    assert_eq!(fixture.files(), ["ok.com", "ok.log", "ok2.com"]);
}

#[test]
fn native_failures_report_log_errors_and_continue_under_c() {
    let fixture = Fixture::new("fail", &[OK_COM, FAIL_COM, FAILNE_COM]);
    let output = fixture.run(&["-c", "fail", "failne", "ok"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Running fail.com ... Error executing fail.com\x07\n\
ERROR: bad thing\n\
ERROR: sh -c 'echo before; echo ERROR: bad thing; exit 2': exited with status 2\n\
Running failne.com ... Error executing failne.com\x07\n\
ERROR: sh -c 'echo l1; echo l2; echo l3; echo l4; echo l5; exit 3': exited with status 3\n\
Running ok.com ... ok.com  finished successfully\x07\n"
    );
    assert_eq!(
        fixture.files(),
        [
            "fail.com",
            "fail.log",
            "failne.com",
            "failne.log",
            "ok.com",
            "ok.log"
        ]
    );

    // Without -c the loop stops at the first failure
    let fixture = Fixture::new("fail-stop", &[OK_COM, FAIL_COM]);
    let output = fixture.run(&["fail", "ok"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("Running ok.com"));
    assert_eq!(fixture.files(), ["fail.com", "fail.log", "ok.com"]);
}

#[test]
fn native_failure_without_a_log_prints_the_error_strings() {
    let fixture = Fixture::new("nolog", &[OK_COM]);
    let output = fixture.run(&["-c", "missing.com", "ok"]);
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let pid = regex::Regex::new(r"submtemp\.\d+").unwrap();
    assert_eq!(
        pid.replace_all(&stdout, "submtemp.PID"),
        "ERROR: vmstopy - Opening command file missing.com\n\
Error executing missing.com\x07\n\
vmstopy missing.com missing.log: exited with status 1\n\
\n\
Running ok.com ... ok.com  finished successfully\x07\n"
    );
}

#[test]
fn native_numbered_logs_take_the_integer_after_the_last_dash() {
    // `int()` accepts `1_5` as 15; `7-2` gives 2; `x` and `9a` are skipped
    let fixture = Fixture::new(
        "lognum",
        &[
            OK_COM,
            OK2_COM,
            ("ok.log-3", "x"),
            ("ok.log-12", "x"),
            ("ok.log-x", "x"),
            ("ok.log-1_5", "x"),
            ("ok.log-7-2", "x"),
            ("ok.log-9a", "x"),
        ],
    );
    let output = fixture.run(&["-l", "2", "ok", "ok2"]);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Running ok.com with log in ok.log-16 ... ok.com  finished successfully\x07\nRunning ok2.com with log in ok2.log-01 ... ok2.com  finished successfully\x07\n"
    );

    // The digit count is capped at 4 but a longer number is kept whole
    let fixture = Fixture::new("lognum9", &[OK_COM, ("ok.log-12345", "x")]);
    let output = fixture.run(&["-l", "9", "ok"]);
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Running ok.com with log in ok.log-12346 ... ok.com  finished successfully\x07\n"
    );
}

// `-s` ran the file through `vmstocsh` and `tcsh -ef`, which leave
// programs' standard error out of the log; the runner keeps that, and like
// the default path writes no temporary file.  (Native's tcsh file had a blank
// line after every line -- BUGS.md -- and no longer exists.)
#[test]
fn dash_s_leaves_program_stderr_out_of_the_log() {
    let com = ("err.com", "$sh -c 'echo to stdout; echo to stderr >&2'\n");
    let fixture = Fixture::new("dash-s", &[com]);
    let output = fixture.run(&["-s", "err"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        fixture.read("err.log"),
        "to stdout\nSUCCESSFULLY COMPLETED\n"
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("to stderr"));
    assert_eq!(fixture.files(), ["err.com", "err.log"]);

    // Without -s both streams are logged
    let fixture = Fixture::new("dash-s-off", &[com]);
    let output = fixture.run(&["err"]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        fixture.read("err.log"),
        "to stdout\nto stderr\nSUCCESSFULLY COMPLETED\n"
    );
}
