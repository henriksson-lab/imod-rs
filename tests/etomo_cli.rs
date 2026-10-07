// The fixture is a shell-script stand-in for `java`: Unix only.
#![cfg(unix)]
mod common;

#[test]
fn requires_imod_runtime_before_java_boundary() {
    let output = common::imod_cmd("etomo")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run etomo");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "The IMOD_DIR environment variable has not been set\nSet it to point to the directory where IMOD is installed\n"
    );
}

/// The crate's `etomo` binary translates `IMOD/pysrc/etomo`, the Python
/// launcher -- not `EtomoDirector.java`, which is reachable only through the
/// JVM.  This is the launcher's own no-`IMOD_DIR` exit, byte-compared against
/// the Python source's two `sys.stdout.write` lines (`pysrc/etomo:54-56`).
#[test]
fn launcher_reports_the_python_sources_own_imod_dir_message() {
    let output = common::imod_cmd("etomo")
        .env_remove("IMOD_DIR")
        .output()
        .expect("run etomo");
    assert_eq!(output.status.code(), Some(1));
    assert!(
        output.stderr.is_empty(),
        "the launcher writes to stdout, not stderr"
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "The IMOD_DIR environment variable has not been set\nSet it to point to the directory where IMOD is installed\n"
    );
}

// ---------------------------------------------------------------------------
// Native differentials (2026-09-26).  Every expected value below was captured
// by running the Python source (`python3 IMOD/pysrc/etomo`, PYTHONPATH
// `IMOD/pysrc`) with the same stub `java` on PATH, and compared byte for byte
// (stdout, files, exit status) against `imod etomo` over 29 cases; the
// timestamps and the IMOD directory are the only substitutions.  The stub
// records every invocation's argv and the three environment variables the
// launcher sets or clears, then behaves like `java -version`,
// `java -XshowSettings` (a headless build when asked) or the eTomo run.

const STUB_JAVA: &str = r#"#!/bin/sh
{ printf 'ARGV:'; for a in "$@"; do printf ' [%s]' "$a"; done; printf ' LC_NUMERIC=%s PIP=%s RV=%s\n' "$LC_NUMERIC" "$PIP_PRINT_ENTRIES" "${RUNCMD_VERBOSE-unset}"; } >> java-calls.txt
mode=${STUB_JAVA_MODE:-ok}
case "$1" in
 -version)
   [ "$mode" = fail ] && exit 1
   printf '%s\n' "${STUB_JAVA_VERSION:-openjdk version \"17.0.8\" 2023-07-18}" >&2
   printf 'OpenJDK Runtime Environment (build 17.0.8+7)\n' >&2
   exit 0;;
 -XshowSettings)
   echo "Property settings:" >&2
   if [ "$mode" = headless ]; then echo "    java.awt.headless = true" >&2; fi
   echo "Usage: java [options]" >&2
   exit 1;;
esac
[ "$mode" = helpempty ] && exit 0
echo "stub etomo stdout"
echo "stub etomo stderr" >&2
exit ${STUB_JAVA_EXIT:-0}
"#;

struct Fixture {
    root: std::path::PathBuf,
    imod: std::path::PathBuf,
    run: std::path::PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Fixture {
        use std::os::unix::fs::PermissionsExt;
        let root =
            std::env::temp_dir().join(format!("imod-rs-etomo-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bin = root.join("stub");
        let imod = root.join("imod");
        let run = root.join("run");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(imod.join("Plugins")).unwrap();
        std::fs::create_dir_all(&run).unwrap();
        let java = bin.join("java");
        std::fs::write(&java, STUB_JAVA).unwrap();
        std::fs::set_permissions(&java, std::fs::Permissions::from_mode(0o755)).unwrap();
        Fixture { root, imod, run }
    }

    fn command(&self, args: &[&str]) -> std::process::Command {
        let mut command = common::imod_cmd("etomo");
        command
            .args(args)
            .current_dir(&self.run)
            .env("IMOD_DIR", &self.imod)
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("stub").display()),
            )
            .env("LC_NUMERIC", "en_US.UTF-8")
            .env("ETOMO_THREAD_LIM", "16")
            .env("ETOMO_LOG_DIR", "logs")
            .env_remove("PIP_PRINT_ENTRIES")
            .env_remove("RUNCMD_VERBOSE")
            .env_remove("IMOD_JAVADIR")
            .env_remove("IMOD_QTLIBDIR")
            .env_remove("ETOMO_MEM_LIM")
            .env_remove("ETOMO_LOGS_TO_RETAIN");
        command
    }

    fn write(&self, name: &str, text: &str) {
        let path = self.run.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.run.join(name)).unwrap_or_else(|_| "<missing>".to_owned())
    }

    /// The background eTomo run writes its argv line after the launcher exits.
    fn wait_for_calls(&self, lines: usize) -> String {
        for _ in 0..200 {
            let calls = self.read("java-calls.txt");
            if calls.lines().count() >= lines && self.read("etomo_out.log") != "<missing>" {
                std::thread::sleep(std::time::Duration::from_millis(100));
                return self.read("java-calls.txt");
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        self.read("java-calls.txt")
    }

    /// The expected launch line, with the IMOD directory of this fixture.
    fn launch(&self, jar: &str, args: &str) -> String {
        let jar = jar.replace("{IMOD}", &self.imod.display().to_string());
        format!(
            "ARGV: [-Xmx512m] [-XX:ConcGCThreads=16] [-XX:ParallelGCThreads=16] [-XX:ActiveProcessorCount=16] [-cp] [{jar}:{}/Plugins/*] [etomo.EtomoDirector]{args} LC_NUMERIC=C PIP=1 RV=unset\n",
            self.imod.display()
        )
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

const VERSION_CALL: &str = "ARGV: [-version] LC_NUMERIC=en_US.UTF-8 PIP= RV=unset\n";
const SETTINGS_CALL: &str = "ARGV: [-XshowSettings] LC_NUMERIC=C PIP=1 RV=unset\n";

/// Replaces the `%b-%d-%H%M%S` stamp and the `%a %b %d %H:%M:%S %Y` date.
fn mask(text: &str) -> String {
    let stamp = regex::Regex::new(r"[A-Z][a-z]{2}-\d\d-\d{6}").unwrap();
    let date = regex::Regex::new(r"[A-Z][a-z]{2} [A-Z][a-z]{2} \d\d \d\d:\d\d:\d\d \d{4}").unwrap();
    let text = date.replace_all(text, "DATE");
    stamp.replace_all(&text, "STAMP").into_owned()
}

#[test]
fn native_background_launch_creates_error_log_pointer_without_rolling() {
    let fixture = Fixture::new("bg-new");
    fixture.write("logs/keep", "x");
    let output = fixture.command(&["a.edf"]).output().unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        mask(&String::from_utf8_lossy(&output.stdout)),
        "Starting Etomo with log in logs/etomo_err_STAMP.log\nThis log may contain personal information, such as your username\n"
    );
    let calls = fixture.wait_for_calls(3);
    assert_eq!(
        calls,
        format!(
            "{VERSION_CALL}{SETTINGS_CALL}{}",
            fixture.launch("{IMOD}/bin//etomo.jar", " [a.edf]")
        )
    );
    // The Python checks only an *existing* etomo_err.log; a new one is
    // created with the pointer line and nothing is rolled
    assert_eq!(
        mask(&fixture.read("etomo_err.log")),
        "Error log for DATE is in logs/etomo_err_STAMP.log\n"
    );
    assert_eq!(fixture.read("etomo_err1.log"), "<missing>");
    assert_eq!(fixture.read("etomo_out.log"), "stub etomo stdout\n");
}

#[test]
fn native_existing_real_error_log_is_rolled_and_pointer_log_appended() {
    let fixture = Fixture::new("bg-roll");
    fixture.write("logs/keep", "x");
    fixture.write("etomo_err.log", "real log\n");
    fixture.write("etomo_err1.log", "one\n");
    fixture.write("etomo_err11.log", "eleven\n");
    fixture.write("etomo_err12.log", "twelve\n");
    let output = fixture.command(&[]).output().unwrap();
    assert_eq!(output.status.code(), Some(0));
    fixture.wait_for_calls(3);
    assert_eq!(fixture.read("etomo_err1.log"), "real log\n");
    assert_eq!(fixture.read("etomo_err2.log"), "one\n");
    assert_eq!(fixture.read("etomo_err12.log"), "eleven\n");
    assert_eq!(fixture.read("etomo_err11.log"), "<missing>");
    assert_eq!(
        mask(&fixture.read("etomo_err.log")),
        "Error log for DATE is in logs/etomo_err_STAMP.log\n"
    );

    let fixture = Fixture::new("bg-append");
    fixture.write("logs/keep", "x");
    fixture.write("etomo_err.log", "Error log for x is in y\n");
    fixture.write("etomo_out.log", "old out\n");
    let output = fixture.command(&[]).output().unwrap();
    assert_eq!(output.status.code(), Some(0));
    fixture.wait_for_calls(3);
    assert_eq!(
        mask(&fixture.read("etomo_err.log")),
        "Error log for x is in y\nError log for DATE is in logs/etomo_err_STAMP.log\n"
    );
    assert_eq!(fixture.read("etomo_out.log~"), "old out\n");
    assert_eq!(fixture.read("etomo_err1.log"), "<missing>");
}

#[test]
fn native_log_purge_keeps_newest_matches_by_time_then_name() {
    let fixture = Fixture::new("purge");
    for (name, text, time) in [
        ("logs/etomo_a.log", "a", 1000),
        ("logs/etomo_b.log", "b", 3000),
        ("logs/etomo_c.log", "c", 2000),
        ("logs/other.txt", "o", 500),
        ("logs/etomo_d.log", "d", 2000),
    ] {
        fixture.write(name, text);
        let file = std::fs::File::options()
            .write(true)
            .open(fixture.run.join(name))
            .unwrap();
        file.set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(time))
            .unwrap();
    }
    let output = fixture
        .command(&[])
        .env("ETOMO_LOGS_TO_RETAIN", "2")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    fixture.wait_for_calls(3);
    let mut names = std::fs::read_dir(fixture.run.join("logs"))
        .unwrap()
        .map(|entry| mask(&entry.unwrap().file_name().to_string_lossy()))
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(
        names,
        [
            "etomo_b.log",
            "etomo_d.log",
            "etomo_err_STAMP.log",
            "other.txt"
        ]
    );

    let bad = Fixture::new("purge-bad");
    bad.write("logs/k", "k");
    let output = bad
        .command(&[])
        .env("ETOMO_LOGS_TO_RETAIN", "abc")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: etomo - Converting environment variable ETOMO_LOGS_TO_RETAIN (abc) to integer\n"
    );
}

#[test]
fn native_java_checks_exit_with_the_sources_messages() {
    let cases = [
        (
            "nojava",
            [("STUB_JAVA_MODE", "fail")],
            "ERROR: There is no java runtime in the current search path.  A Java\nruntime environment needs to be installed and the command search path may need\nto be defined or IMOD_JAVADIR set to locate the java command.\n",
        ),
        (
            "gnu",
            [(
                "STUB_JAVA_VERSION",
                "java version \"1.5.0\" gij (GNU libgcj)",
            )],
            "ERROR: Etomo will not work with GNU java.  You should install an\nOpenJDK version of the Java runtime environment and put it on\nyour command search path\n",
        ),
        (
            "headless",
            [("STUB_JAVA_MODE", "headless")],
            "ERROR: The installed java is \"headless\"; to open the Etomo interface\n  you need to use a full installation of java that does not have\n  \"headless\" in its package name\n",
        ),
    ];
    for (name, env, expected) in cases {
        let fixture = Fixture::new(name);
        let output = fixture.command(&[]).envs(env).output().unwrap();
        assert_eq!(output.status.code(), Some(1), "{name}");
        assert_eq!(String::from_utf8_lossy(&output.stdout), expected, "{name}");
    }

    let fixture = Fixture::new("java15");
    let output = fixture
        .command(&[])
        .env("STUB_JAVA_VERSION", "java version \"1.5.0_22\"")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        format!(
            "ERROR: You are trying to run a version of Java before 1.6, located at {}\nEtomo will no longer work with java 1.4-1.5.  You should install an\nOracle or OpenJDK version of the Java runtime environment, version 1.6 or higher,\nand put it on your command search path, or point IMOD_JAVADIR to it\n",
            fixture.root.join("stub/java").display()
        )
    );
}

#[test]
fn native_help_runs_java_in_the_foreground_and_prints_its_output() {
    let fixture = Fixture::new("help");
    let output = fixture.command(&["-h", "-x", "--foo"]).output().unwrap();
    assert_eq!(output.status.code(), Some(0));
    // `-h` becomes `--help` before the single-dash check, so only `-x` warns
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "WARNING: YOU ENTERED AN ARGUMENT WITH A SINGLE DASH: -x\nstub etomo stdout\n"
    );
    assert_eq!(
        fixture.read("java-calls.txt"),
        format!(
            "{VERSION_CALL}{SETTINGS_CALL}{}",
            fixture.launch("{IMOD}/bin//etomo.jar", " [--help] [-x] [--foo]")
        )
    );

    let failing = Fixture::new("help-fail");
    let output = failing
        .command(&["--help"])
        .env("STUB_JAVA_EXIT", "3")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "An error occurred running etomo for help output\n"
    );

    // No help output: native iterates `None` and dies with a TypeError
    // (exit 1); fixed in translation (BUGS.md): nothing is printed, exit 0
    let empty = Fixture::new("help-empty");
    let output = empty
        .command(&["--help"])
        .env("STUB_JAVA_MODE", "helpempty")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stdout.is_empty());
}

#[test]
fn native_foreground_failure_keeps_both_logs_and_reports() {
    let fixture = Fixture::new("fg-fail");
    let output = fixture
        .command(&["--fg"])
        .env("ETOMO_LOG_DIR", "")
        .env("STUB_JAVA_EXIT", "4")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Starting Etomo with log in etomo_err.log\nThis log may contain personal information, such as your username\nERROR: etomo exited with an error status, check: etomo_err.log\n"
    );
    assert_eq!(fixture.read("etomo_out.log"), "stub etomo stdout\n");
    assert_eq!(fixture.read("etomo_err.log"), "stub etomo stderr\n");
}

#[test]
fn native_jardir_and_argument_passing() {
    let fixture = Fixture::new("jardir");
    let output = fixture
        .command(&["--jardir", "/my/jars", "-single", "--newstuff", "x y"])
        .env("ETOMO_LOG_DIR", "")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "WARNING: YOU ENTERED AN ARGUMENT WITH A SINGLE DASH: -single\nStarting Etomo with log in etomo_err.log\nThis log may contain personal information, such as your username\n"
    );
    assert_eq!(
        fixture.wait_for_calls(3),
        format!(
            "{VERSION_CALL}{SETTINGS_CALL}{}",
            fixture.launch("/my/jars/etomo.jar", " [-single] [--newstuff] [x y]")
        )
    );
}

#[test]
fn native_java_version_decides_active_processor_count() {
    for (version, active) in [
        ("openjdk version \"1.8.0_191\"", true),
        ("openjdk version \"1.8.0_181\"", false),
        ("openjdk version \"21\" 2023-09-19", false),
    ] {
        let fixture = Fixture::new("version");
        let output = fixture
            .command(&["--fg"])
            .env("STUB_JAVA_VERSION", version)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(0));
        let calls = fixture.read("java-calls.txt");
        assert_eq!(
            calls.contains("-XX:ActiveProcessorCount=16"),
            active,
            "{version}: {calls}"
        );
    }
}
