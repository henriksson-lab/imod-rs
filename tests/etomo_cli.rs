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
