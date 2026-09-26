//! The command file converters and the in-process command file runner.
//!
//! * `vmstocsh` (`IMOD/flib/image/vmstocsh.f`) and `vmstopy`
//!   (`IMOD/pysrc/vmstopy`) against goldens the native converters wrote for
//!   the command files in `fixtures/comrun/convert` (IMOD templates, files
//!   written by splitcombine/splittilt/chunksetup, a continuation-line case
//!   and an error case).  The goldens are digests in
//!   `fixtures/comrun/golden.manifest`, keyed `convert/<case>.{csh,py,pyout}`
//!   and `convert/combine.nolog.csh` (`fixtures/README.md`): to re-record,
//!   put the native outputs back under those names and run
//!   `RECORD_ONLY=1 fixtures/regen-golden.sh comrun`, then delete them.
//!   The wider differential -- 1131 distinct command
//!   files, byte-identical except the `0o766` fix -- is recorded in
//!   `TODO.md`.
//! * `runcom` (`src/imod/comrun.rs`) on the command files in
//!   `fixtures/comrun/run`, with this crate's own programs: exit status,
//!   which later steps ran, and the log's `ERROR:`/`SUCCESSFULLY COMPLETED`
//!   lines.  These run through the binary, not the library: libtest's
//!   output capture would swallow what the in-process programs print.

mod common;

use std::path::{Path, PathBuf};

fn fixture_dir(sub: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/comrun")
        .join(sub)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("imod-rs-comrun-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const CONVERT_CASES: &[&str] = &[
    "align",
    "combine",
    "continuation",
    "finish_remove",
    "localpart",
    "tilt_sync",
    "volcombine",
    "volcombine_tmpdir",
];

#[test]
fn vmstocsh_matches_native_goldens() {
    let dir = fixture_dir("convert");
    for case in CONVERT_CASES {
        let input = std::fs::read(dir.join(format!("{case}.com"))).unwrap();
        let golden = common::golden::expect(&dir.join(format!("{case}.csh")));
        let mut child = common::imod_cmd("vmstocsh")
            .arg("x.log")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        use std::io::Write as _;
        child.stdin.take().unwrap().write_all(&input).unwrap();
        let output = child.wait_with_output().unwrap();
        assert_eq!(output.status.code(), Some(0), "{case}");
        if let Err(why) = golden.compare(&output.stdout, common::golden::identity, false) {
            panic!("vmstocsh output differs for {case}: {why}");
        }
    }
    // No log argument: every command line ends in a blank instead
    let input = std::fs::read(dir.join("combine.com")).unwrap();
    let golden = common::golden::expect(&dir.join("combine.nolog.csh"));
    let output = common::imod_cmd("vmstocsh")
        .stdin(std::fs::File::open(dir.join("combine.com")).unwrap())
        .output()
        .unwrap();
    let _ = input;
    if let Err(why) = golden.compare(&output.stdout, common::golden::identity, false) {
        panic!("vmstocsh output differs without a log: {why}");
    }
}

#[test]
fn vmstopy_matches_native_goldens() {
    let dir = fixture_dir("convert");
    let work = scratch("vmstopy");
    for case in CONVERT_CASES {
        let out = work.join(format!("{case}.py"));
        let output = common::imod_cmd("vmstopy")
            .env("IMOD_DIR", "/fixture/imod")
            .arg(dir.join(format!("{case}.com")))
            .arg("x.log")
            .arg(&out)
            .output()
            .unwrap();
        let golden_out = common::golden::expect(&dir.join(format!("{case}.pyout")));
        let expected_status = if golden_out.len() == 0 { 0 } else { 1 };
        assert_eq!(output.status.code(), Some(expected_status), "{case}");
        if let Err(why) = golden_out.compare(&output.stdout, common::golden::identity, false) {
            panic!("vmstopy messages differ for {case}: {why}");
        }
        let golden = common::golden::expect(&dir.join(format!("{case}.py")));
        if let Err(why) = golden.compare(
            &std::fs::read(&out).unwrap(),
            common::golden::identity,
            false,
        ) {
            panic!("vmstopy script differs for {case}: {why}");
        }
    }
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn vmstopy_defined_behaviour_for_upstream_defects() {
    // `$if (-e f) \mv f f~` crashes native vmstopy (`backupmatch`), and
    // `$mkdir` gives a Python 2 literal; see BUGS.md, "vmstopy".
    let work = scratch("vmstopy-fixes");
    let com = work.join("fixes.com");
    std::fs::write(&com, "$if (-e f.log) \\mv f.log f.log~\n$mkdir newdir\n$\n").unwrap();
    let output = common::imod_cmd("vmstopy")
        .env("IMOD_DIR", "/fixture/imod")
        .arg(&com)
        .arg("x.log")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("\n  makeBackupFile(\"f.log\")\n"), "{text}");
    assert!(text.contains("\n  os.mkdir(\"newdir\", 0o766)\n"), "{text}");
    assert!(text.contains("\n  command = \"\"\"\"\"\"\n"), "{text}");
    let _ = std::fs::remove_dir_all(&work);
}

/// A scratch directory holding the run fixtures and a 16x16x2 float `in.mrc`
/// written by this crate's `raw2mrc`.
fn run_dir(name: &str) -> PathBuf {
    let work = scratch(name);
    for entry in std::fs::read_dir(fixture_dir("run")).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, work.join(path.file_name().unwrap())).unwrap();
    }
    let mut raw = Vec::new();
    for i in 0..(16 * 16 * 2) {
        raw.extend_from_slice(&((i % 37) as f32 * 0.25).to_le_bytes());
    }
    std::fs::write(work.join("in.raw"), raw).unwrap();
    let made = common::imod_cmd("raw2mrc")
        .current_dir(&work)
        .args([
            "-x", "16", "-y", "16", "-z", "2", "-t", "float", "in.raw", "in.mrc",
        ])
        .output()
        .unwrap();
    assert_eq!(made.status.code(), Some(0));
    work
}

fn runcom(work: &Path, com: &str) -> (i32, String, String) {
    let output = common::imod_cmd("runcom")
        .current_dir(work)
        .env_remove("IMOD_DIR")
        .env_remove("COMRUN_NOT_DEFINED_XYZ")
        .arg(com)
        .output()
        .unwrap();
    let log = work.join(Path::new(com).with_extension("log"));
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        std::fs::read_to_string(log).unwrap_or_default(),
    )
}

#[test]
fn runcom_runs_every_supported_construct() {
    let work = run_dir("ok");
    let (status, stdout, log) = runcom(&work, "ok.com");
    assert_eq!(status, 0, "{stdout}\n{log}");
    // standard input entries, with a `$set` variable substituted
    assert!(work.join("out.mrc").exists());
    // `$if (-e ...)` with a continued command; the negated test is false
    assert!(work.join("continued.mrc").exists());
    assert!(!work.join("notmade.mrc").exists());
    // `$goto` skips forward to the label
    assert!(!work.join("skipped.mrc").exists());
    // b3dcopy made it, `b3dremove -g cop?.mrc` removed it
    assert!(!work.join("copy.mrc").exists());
    assert!(log.starts_with("Starting the run\n"), "{log}");
    // `header -size copy.mrc` ran in process with its output in the log
    assert!(log.contains("      16      16       2\n"), "{log}");
    assert!(log.contains("\nvalue fromsetenv scale 1.5\n \n"), "{log}");
    assert!(log.ends_with("SUCCESSFULLY COMPLETED\n"), "{log}");
    // the log of a previous run is backed up
    let (status, _, _) = runcom(&work, "ok.com");
    assert_eq!(status, 0);
    assert!(work.join("ok.log~").exists());
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn runcom_stops_at_the_first_failing_program() {
    let work = run_dir("fail");
    let (status, _, log) = runcom(&work, "fail.com");
    assert_eq!(status, 1);
    assert!(!work.join("after.mrc").exists());
    assert!(log.starts_with("before\n"), "{log}");
    assert!(
        log.ends_with("ERROR: header -size missing.mrc: exited with status 1\n"),
        "{log}"
    );
    assert!(!log.contains("SUCCESSFULLY COMPLETED"));
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn runcom_status_goto_runs_the_error_block() {
    let work = run_dir("errfunc");
    let (status, _, log) = runcom(&work, "errfunc.com");
    assert_eq!(status, 1);
    assert!(!work.join("never.mrc").exists());
    assert!(
        log.ends_with(
            "ERROR: header -size missing.mrc: exited with status 1\nERROR: step2.com failed\n"
        ),
        "{log}"
    );
    assert!(!log.contains("ALL DONE"));
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn runcom_runs_nested_command_files() {
    let work = run_dir("nested");
    let (status, _, log) = runcom(&work, "nested.com");
    assert_eq!(status, 0, "{log}");
    assert!(work.join("fromsub.mrc").exists());
    assert!(work.join("afternested.mrc").exists());
    assert!(!work.join("sub.com.py").exists());
    let sub = std::fs::read_to_string(work.join("sub.log")).unwrap();
    assert!(sub.ends_with("SUCCESSFULLY COMPLETED\n"));
    assert!(log.starts_with("master log\n") && log.ends_with("SUCCESSFULLY COMPLETED\n"));

    let (status, _, log) = runcom(&work, "nestedfail.com");
    assert_eq!(status, 1);
    assert!(
        log.ends_with(
            "ERROR: python -u subfail.com.py: exited with status 1\nERROR: nested failed\n"
        ),
        "{log}"
    );
    let sub = std::fs::read_to_string(work.join("subfail.log")).unwrap();
    assert!(sub.ends_with("ERROR: header -size missing.mrc: exited with status 1\n"));
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn runcom_error_paths() {
    let work = run_dir("errors");
    // an undefined ${ENV} stops the run after the step before it
    let (status, _, log) = runcom(&work, "envmissing.com");
    assert_eq!(status, 1);
    assert!(work.join("first.mrc").exists());
    assert!(!work.join("second.mrc").exists());
    assert!(
        log.ends_with("ERROR: Environment variable not defined: 'COMRUN_NOT_DEFINED_XYZ'\n"),
        "{log}"
    );
    // `$exit 3` ends the run with that status and no success line
    let (status, _, log) = runcom(&work, "exit3.com");
    assert_eq!(status, 3);
    assert!(!work.join("second.mrc").exists());
    assert!(!log.contains("SUCCESSFULLY COMPLETED"));
    // vmstopy's rejection: nothing runs, no log
    let (status, stdout, _) = runcom(&work, "bad.com");
    assert_eq!(status, 1);
    assert!(stdout.contains("ERROR: vmstopy - Expected command or Python line: stray entry line"));
    assert!(!work.join("bad.mrc").exists());
    assert!(!work.join("bad.log").exists());
    // a line needing a shell goes to /bin/sh; `$sync` is done directly
    let (status, _, log) = runcom(&work, "shell.com");
    assert_eq!(status, 0, "{log}");
    assert_eq!(
        std::fs::read_to_string(work.join("shellout.txt")).unwrap(),
        "viashell"
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// `vmstopy -x` executes the script it wrote with the runner, not `python -u`
/// (2026-09-26): with no `python` on `PATH` it still runs, and its messages,
/// status and log match native `vmstopy -x` (measured in
/// `/big/henriksson/realbench/comwire/vx`; native also prints `Python PID:` on
/// standard error, which the runner does not).
#[test]
fn vmstopy_x_executes_through_the_runner() {
    let work = scratch("vmstopy-x");
    let bin = work.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::os::unix::fs::symlink("/bin/sh", bin.join("sh")).unwrap();
    std::fs::write(work.join("ok.com"), "$echo hi\n").unwrap();
    std::fs::write(
        work.join("bad.com"),
        "$echo before\n$sh -c \"echo ERROR: bad; exit 2\"\n$echo after\n",
    )
    .unwrap();
    let run = |args: &[&str]| {
        common::imod_cmd("vmstopy")
            .args(args)
            .current_dir(&work)
            .env("PATH", &bin)
            .env("IMOD_DIR", work.join("imod"))
            .output()
            .unwrap()
    };
    let output = run(&["-x", "ok.com", "ok.log"]);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Executing Python script...  DONE!\n"
    );
    assert_eq!(
        std::fs::read_to_string(work.join("ok.log")).unwrap(),
        "hi\nSUCCESSFULLY COMPLETED\n"
    );
    let output = run(&["-x", "bad.com", "bad.log"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Executing Python script...  ERROR: vmstopy - Executing the script; see log for error\n"
    );
    assert_eq!(
        std::fs::read_to_string(work.join("bad.log")).unwrap(),
        "before\nERROR: bad\nERROR: sh -c \"echo ERROR: bad; exit 2\": exited with status 2\n"
    );
    let output = run(&["-x", "-q", "bad.com", "bad.log"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    // the temporary script is removed
    let mut names: Vec<String> = std::fs::read_dir(&work)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        ["bad.com", "bad.log", "bad.log~", "bin", "ok.com", "ok.log"]
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// `runcom -P` prints the PID line processchunks' kill path reads (`PID:`
/// followed by the number, on standard error), as the `vmstopy` script's
/// `printPID(True)` did.
#[test]
fn runcom_dash_p_prints_its_pid() {
    let work = scratch("runcom-pid");
    std::fs::write(work.join("ok-001.com"), "$echo hi\n").unwrap();
    let output = common::imod_cmd("runcom")
        .args(["-P", "-c", "-n", "0", "-e", "X=1", "ok-001.com"])
        .current_dir(&work)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&output.stderr);
    let pid = stderr
        .strip_prefix("Runcom PID: ")
        .and_then(|rest| rest.strip_suffix('\n'))
        .expect("PID line");
    assert!(pid.parse::<u32>().is_ok(), "{stderr}");
    assert_eq!(
        std::fs::read_to_string(work.join("ok-001.log")).unwrap(),
        "hi\nSUCCESSFULLY COMPLETED\nCHUNK DONE\n"
    );
    let _ = std::fs::remove_dir_all(&work);
}
