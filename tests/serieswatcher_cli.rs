//! Native-golden coverage for `serieswatcher` (`IMOD/pysrc/serieswatcher`, translated in
//! `src/imod/pysrc/serieswatcher.rs`).
//!
//! Every row of `fixtures/serieswatcher/cases.tsv` was run through the native Python
//! script by `fixtures/make-serieswatcher-goldens.sh` (`make-pyscript-goldens.py`); see
//! `tests/pysetup_common` for what is compared.

mod common;
mod pysetup_common;

#[test]
fn serieswatcher_matches_native_goldens() {
    let failures = pysetup_common::run_cases("serieswatcher");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Defined behaviour (BUGS.md): without a command file native dies on its
/// first pass (`NameError` on `topCheckFile`); here the stack is delivered
/// with its `.mdoc`, and an interrupt ends the watcher as the source's
/// `except KeyboardInterrupt` does.
#[test]
fn serieswatcher_deliver_only() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!(
        "imod-rs-serieswatcher-deliver-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(work.join("out")).unwrap();
    std::fs::copy(
        root.join("fixtures/extracttilts/agard.st"),
        work.join("tilt.mrc"),
    )
    .unwrap();
    std::fs::write(work.join("tilt.mrc.mdoc"), "mdoc\n").unwrap();
    let child = common::imod_cmd("serieswatcher")
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .args(["-deliver", "out", "-age", "1", "-views", "5"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let delivered = work.join("out/tilt.mrc");
    let mut waited = 0;
    while !delivered.exists() && waited < 600 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        waited += 1;
    }
    std::thread::sleep(std::time::Duration::from_millis(500));
    // SAFETY: signals our own child
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) };
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Moved stack tilt.mrc\nSent signal to quit to all processes\n"
    );
    assert!(delivered.exists());
    assert!(work.join("out/tilt.mrc.mdoc").exists());
    assert!(!work.join("tilt.mrc").exists());
    let _ = std::fs::remove_dir_all(&work);
}
