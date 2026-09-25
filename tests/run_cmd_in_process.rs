//! `imodpy::run_cmd` runs our own commands in this process (the command
//! table in `src/imod/commands.rs`), which is how every `pysrc` translation
//! -- `batchruntomo`'s `xftoxg`, `xfproduct`, `imodtrans`, `xfmodel` calls
//! among them -- reaches them.  Each case here runs a command through
//! `run_cmd` and through the `imod` binary and requires the same exit status,
//! the same collected standard output, and byte-identical output files.
//! `PATH` is emptied for the in-process call, so a fall-back to `sh -c` could
//! not find the program and would fail instead of passing silently.

mod common;

use imod_rs::imod::pysrc::imodpy::{get_last_exit_status, run_cmd};
use std::path::{Path, PathBuf};

/// libtest captures `print!` per test thread and hands the capture to every
/// thread the test spawns, so a command run in process under the default
/// harness writes its Rust-side output into the harness rather than to the
/// descriptor `run_cmd` redirects -- which a real process such as
/// `batchruntomo` never does.  Each test therefore re-runs itself in a child
/// test process with `--nocapture` and does its work there; returns `true`
/// in the parent, which then has nothing left to do.
fn rerun_uncaptured(name: &str) -> bool {
    if std::env::var_os("IMOD_RS_RUN_CMD_CHILD").is_some() {
        return false;
    }
    // The child's own harness summary is kept out of this suite's output
    // (it would read as an extra suite) and shown only on failure.
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env("IMOD_RS_RUN_CMD_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{name} failed in its uncaptured child:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "imod-rs-run-cmd-in-process-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn copy_in(dir: &Path, source: &Path) {
    std::fs::copy(source, dir.join(source.file_name().unwrap())).unwrap();
}

/// Runs `args` (relative file names, resolved in `dir`) once through the
/// binary in `dir/direct` and once through `run_cmd` in `dir/inproc`, and
/// compares status, collected stdout and the named output files.
fn compare(
    dir: &Path,
    inputs: &[PathBuf],
    program: &str,
    args: &[&str],
    outputs: &[&str],
    expect: i32,
) {
    // Both sides see the same environment: the in-process run reads this
    // process's, and the child inherits it.
    unsafe {
        std::env::set_var(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        );
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("IMOD_NO_IMAGE_BACKUP", "1");
    }
    let direct = dir.join("direct");
    let inproc = dir.join("inproc");
    for side in [&direct, &inproc] {
        std::fs::create_dir_all(side).unwrap();
        for input in inputs {
            copy_in(side, input);
        }
    }

    let native = common::imod_cmd(program)
        .current_dir(&direct)
        .args(args)
        .output()
        .unwrap();
    let direct_lines: Vec<String> = String::from_utf8_lossy(&native.stdout)
        .lines()
        .map(str::to_owned)
        .collect();

    // Absolute paths, since `run_cmd` has no working-directory argument.
    let mut command = program.to_owned();
    for arg in args {
        let candidate = inproc.join(arg);
        if arg.starts_with('-') || arg.parse::<f64>().is_ok() || arg.contains(',') {
            command.push_str(&format!(" {arg}"));
        } else {
            command.push_str(&format!(" \"{}\"", candidate.display()));
        }
    }
    let saved_path = std::env::var_os("PATH");
    unsafe { std::env::set_var("PATH", "") };
    let result = run_cmd(&command, None, None, None, &[]);
    unsafe {
        match saved_path {
            Some(path) => std::env::set_var("PATH", path),
            None => std::env::remove_var("PATH"),
        }
    }
    assert_eq!(result.is_ok(), expect == 0, "{command}: {result:?}");
    let status = if result.is_ok() {
        0
    } else {
        get_last_exit_status()
    };
    assert_eq!(
        status,
        native.status.code().unwrap(),
        "{command}: exit status"
    );
    assert_eq!(status, expect, "{command}: expected exit status");
    let Ok(in_lines) = result else {
        return;
    };
    let in_lines = in_lines.unwrap_or_default();
    // The collected output names the files it opens; compare with the
    // directory prefix removed.
    let strip = |lines: &[String], prefix: &Path| -> Vec<String> {
        let prefix = format!("{}/", prefix.display());
        lines.iter().map(|l| l.replace(&prefix, "")).collect()
    };
    assert_eq!(
        strip(&in_lines, &inproc),
        strip(&direct_lines, &direct),
        "{command}: standard output"
    );
    for output in outputs {
        let a = std::fs::read(direct.join(output)).unwrap();
        let b = std::fs::read(inproc.join(output)).unwrap();
        assert!(a == b, "{command}: {output} differs");
    }
}

#[test]
fn xftoxg_through_run_cmd_matches_direct() {
    if rerun_uncaptured("xftoxg_through_run_cmd_matches_direct") {
        return;
    }
    let dir = scratch("xftoxg");
    let bba = Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/Etomo/uitestData/BB/BBa.xf");
    compare(
        &dir,
        &[bba],
        "xftoxg",
        &["-nfit", "0", "BBa.xf", "o.xg"],
        &["o.xg"],
        0,
    );
}

#[test]
fn tilt_through_run_cmd_matches_direct() {
    if rerun_uncaptured("tilt_through_run_cmd_matches_direct") {
        return;
    }
    let dir = scratch("tilt");
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tilt");
    compare(
        &dir,
        &[fixtures.join("t.ali"), fixtures.join("t.tlt")],
        "tilt",
        &[
            "-input",
            "t.ali",
            "-output",
            "o.rec",
            "-TILTFILE",
            "t.tlt",
            "-THICKNESS",
            "16",
            "-MODE",
            "1",
        ],
        &[],
        0,
    );
    // The MRC label carries a time stamp; compare the rest.
    let mut a = std::fs::read(dir.join("direct/o.rec")).unwrap();
    let mut b = std::fs::read(dir.join("inproc/o.rec")).unwrap();
    assert_eq!(a.len(), b.len());
    for bytes in [&mut a, &mut b] {
        bytes[224..1024].fill(0);
    }
    assert!(a == b, "tilt: o.rec differs outside the labels");
}

#[test]
fn failing_command_reports_status_in_process() {
    if rerun_uncaptured("failing_command_reports_status_in_process") {
        return;
    }
    let dir = scratch("fail");
    compare(
        &dir,
        &[],
        "xftoxg",
        &["-nfit", "0", "missing.xf", "o.xg"],
        &[],
        1,
    );
}
