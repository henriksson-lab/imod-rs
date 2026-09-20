//! End-to-end scheduling coverage for `qttools/processchunks`: the local,
//! remote (`ssh`) and cluster-queue (`-q`/`-Q`) paths of `ProcessHandler` and
//! `MachineHandler`.
//!
//! The remote and queue paths are driven by stub `ssh` and queue commands on
//! `PATH` that record the exact argument vector they were handed, which is
//! what `runProcess` and `killSignal` build.  Every expectation here was taken
//! from the native binary
//! (`/tmp/imod-reference-build/qttools/processchunks/processchunks`) run
//! against the same stubs.

mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// A fresh working directory plus a `bin` directory holding the stub
/// `vmstopy`, `python`, `ssh` and queue commands processchunks will run.
fn work_dir(tag: &str) -> (PathBuf, PathBuf) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!(
        "imod-rs-processchunks-{tag}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let bin = work.join("bin");
    fs::create_dir_all(&bin).unwrap();
    write_script(
        &bin.join("vmstopy"),
        &format!(
            "#!/bin/sh\nexec python3 '{}' \"$@\"\n",
            root.join("IMOD/pysrc/vmstopy").display()
        ),
    );
    write_script(&bin.join("python"), "#!/bin/sh\nexec python3 \"$@\"\n");
    // `ssh -V` has to answer, or `setupSshOpts` cannot decide the version and
    // leaves `-o ConnectTimeout=5 ` off the option list.
    write_script(
        &bin.join("ssh"),
        r#"#!/bin/sh
{ n=0; for a in "$@"; do n=$((n+1)); printf 'argv[%d]=<%s>\n' "$n" "$a"; done; } >> ssh.argv
if [ "$1" = "-V" ]; then
  echo "OpenSSH_8.9p1 Ubuntu-3ubuntu0.13, OpenSSL 3.0.2 15 Mar 2022" >&2
  exit 0
fi
if [ -n "$SSHSTUB_REFUSE" ]; then
  echo "ssh: connect to host remotebox port 22: Connection refused" >&2
  exit 255
fi
last=""
for a in "$@"; do last="$a"; done
cmd=$(printf '%s' "$last" | sed -e 's/^"//' -e 's/"$//')
exec /bin/sh -c "$cmd"
"#,
    );
    write_script(
        &bin.join("qstub"),
        r#"#!/bin/sh
printf '%s\n' "$*" >> queue.args
action=""
root=""
prev=""
for a in "$@"; do
  case "$prev" in -a) action="$a" ;; esac
  prev="$a"
  root="$a"
done
if [ "$action" = "R" ]; then
  echo "PID: $$" > "$root.qid"
  : > "$root.job"
  if [ -f "$root.py" ]; then
    ( python -u "$root.py" >> "$root.job" 2>&1; rm -f "$root.py" ) &
  fi
fi
exit 0
"#,
    );
    (work, bin)
}

fn write_script(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

fn processchunks(work: &Path, bin: &Path, args: &[&str]) -> std::process::Output {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap());
    let mut command: Command = common::imod_cmd("processchunks");
    command
        .current_dir(work)
        .args(args)
        .env("PATH", path)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("PYTHONPATH", root.join("IMOD/pysrc"));
    command.output().unwrap()
}

#[test]
fn processchunks_runs_a_real_tiny_comfile_through_imod_vmstopy() {
    let (work, bin) = work_dir("local");
    // A real IMOD command-file command starts with `$`; vmstopy owns the
    // conversion and emits `CHUNK DONE` because processchunks passes `-c`.
    fs::write(work.join("tiny-001.com"), "$echo processchunks-tiny\n").unwrap();
    let result = processchunks(&work, &bin, &["-g", "1", "tiny"]);
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        stdout.contains("tiny-001.com finished on localhost"),
        "{stdout}"
    );
    assert!(stdout.contains("Finished reassembling"), "{stdout}");
    assert!(
        fs::read_to_string(work.join("tiny-001.log"))
            .unwrap()
            .contains("CHUNK DONE")
    );
    fs::remove_dir_all(work).unwrap();
}

/// `ProcessHandler::runProcess`'s remote branch: `ssh -x <sshOpts> <machine>
/// bash --login -c "cd <dir> && python -u < <root>.py"`.  The option list is
/// `Processchunks::getSshOpts`, which an abridged translation left out.
#[test]
fn remote_job_ssh_command_line_carries_the_ssh_options() {
    let (work, bin) = work_dir("sshargv");
    fs::write(work.join("tiny-001.com"), "$echo remote-chunk\n").unwrap();
    let result = processchunks(&work, &bin, &["-g", "remotebox", "tiny"]);
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("tiny-001.com finished on remotebox"),
        "{stdout}"
    );
    let argv = fs::read_to_string(work.join("ssh.argv")).unwrap();
    // The first recorded call is `setupSshOpts`'s version probe.
    assert!(argv.starts_with("argv[1]=<-V>\n"), "{argv}");
    let run = argv.split_once("argv[1]=<-x>\n").expect("no -x call").1;
    assert_eq!(
        run,
        format!(
            "argv[2]=<-o ConnectTimeout=5 >\n\
             argv[3]=<-o PreferredAuthentications=publickey>\n\
             argv[4]=<-o StrictHostKeyChecking=no>\n\
             argv[5]=<remotebox>\n\
             argv[6]=<bash>\n\
             argv[7]=<--login>\n\
             argv[8]=<-c>\n\
             argv[9]=<\"cd {} && python -u < tiny-001.py\">\n",
            work.canonicalize().unwrap().display()
        ),
        "{argv}"
    );
    fs::remove_dir_all(work).unwrap();
}

/// `ProcessHandler::getSshError` finds `ssh: connect to host` in the process's
/// standard error and turns it into the machine's drop message.
#[test]
fn a_refused_ssh_connection_drops_the_machine_with_its_own_message() {
    let (work, bin) = work_dir("sshrefuse");
    fs::write(work.join("tiny-001.com"), "$echo remote-chunk\n").unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap());
    let result = common::imod_cmd("processchunks")
        .current_dir(&work)
        .args(["-g", "remotebox", "tiny"])
        .env("PATH", path)
        .env("SSHSTUB_REFUSE", "1")
        .env("IMOD_DIR", root.join("IMOD"))
        .env("PYTHONPATH", root.join("IMOD/pysrc"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains(
            "Dropping remotebox - cannot connect (ssh: connect to host remotebox port 22: \
             Connection refused)"
        ),
        "{stdout}"
    );
    assert_eq!(result.status.code(), Some(1), "{stdout}");
    fs::remove_dir_all(work).unwrap();
}

/// `ProcessHandler::setJob` builds the queue parameter list as
/// `<queue params> -w <dir> -a R <root>`, and `runProcess` runs the queue
/// command with it.
#[test]
fn a_queue_job_is_submitted_with_the_sources_parameter_list() {
    let (work, bin) = work_dir("queue");
    fs::write(work.join("tiny-001.com"), "$echo queue-chunk\n").unwrap();
    let result = processchunks(&work, &bin, &["-g", "-q", "1", "qstub -x", "tiny"]);
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("tiny-001.com finished on queue"),
        "{stdout}"
    );
    assert_eq!(
        fs::read_to_string(work.join("queue.args")).unwrap(),
        format!(
            "-x -w {} -a R tiny-001\n",
            work.canonicalize().unwrap().display()
        )
    );
    assert!(
        fs::read_to_string(work.join("tiny-001.log"))
            .unwrap()
            .contains("CHUNK DONE")
    );
    fs::remove_dir_all(work).unwrap();
}

/// `ProcessHandler::killSignal` re-runs the queue command with the action
/// letter in place of `R`, then restores `R`.  An abridged translation only
/// set the kill flags and never ran the command.
#[test]
fn a_queue_kill_reruns_the_queue_command_with_the_action_letter() {
    let (work, bin) = work_dir("queuekill");
    fs::write(
        work.join("tiny-001.com"),
        "$sh -c \"echo Q > processchunks.input\"\n$sleep 8\n",
    )
    .unwrap();
    let result = processchunks(
        &work,
        &bin,
        &["-q", "1", "-c", "processchunks.input", "qstub -x", "tiny"],
    );
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(stdout.contains("Killing jobs on queue"), "{stdout}");
    let dir = work.canonicalize().unwrap();
    assert_eq!(
        fs::read_to_string(work.join("queue.args")).unwrap(),
        format!(
            "-x -w {dir} -a R tiny-001\n-x -w {dir} -a K tiny-001\n",
            dir = dir.display()
        )
    );
    assert_eq!(result.status.code(), Some(2), "{stdout}");
    fs::remove_dir_all(work).unwrap();
}

/// `ProcessHandler::printWarnings` reads and discards one line before its
/// loop, so the first line of a chunk log is never scanned; and it collapses
/// warnings differing only in their last word.
#[test]
fn print_warnings_discards_the_first_log_line_and_collapses_the_rest() {
    let (work, bin) = work_dir("warn");
    fs::write(
        work.join("tiny-001.com"),
        "$echo WARNING: a problem with foo\n\
         $echo WARNING: a problem with bar\n\
         $echo MESSAGE: hello\n\
         $echo LOGFILE: other.log\n",
    )
    .unwrap();
    let result = processchunks(&work, &bin, &["-g", "1", "tiny"]);
    let stdout = String::from_utf8_lossy(&result.stdout);
    // The log's first line is `WARNING: a problem with foo`, which the source
    // reads before the loop starts; only `bar` reaches the warning list, so
    // nothing is collapsed to `...` and no count is printed.
    assert!(stdout.contains("MESSAGE: hello - on localhost"), "{stdout}");
    assert!(stdout.contains("LOGFILE: other.log"), "{stdout}");
    assert!(stdout.contains("WARNING: a problem with bar\n"), "{stdout}");
    assert!(!stdout.contains("a problem with ..."), "{stdout}");
    assert!(!stdout.contains("times)"), "{stdout}");
    fs::remove_dir_all(work).unwrap();
}
