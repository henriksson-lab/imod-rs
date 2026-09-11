//! End-to-end local scheduling coverage for `qttools/processchunks`.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn processchunks_runs_a_real_tiny_comfile_through_imod_vmstopy() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!(
        "imod-rs-processchunks-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let bin = work.join("bin");
    fs::create_dir_all(&bin).unwrap();
    let vmstopy = bin.join("vmstopy");
    fs::write(
        &vmstopy,
        format!(
            "#!/bin/sh\nexec python3 '{}' \"$@\"\n",
            root.join("IMOD/pysrc/vmstopy").display()
        ),
    )
    .unwrap();
    fs::set_permissions(&vmstopy, fs::Permissions::from_mode(0o755)).unwrap();
    let python = bin.join("python");
    fs::write(&python, "#!/bin/sh\nexec python3 \"$@\"\n").unwrap();
    fs::set_permissions(&python, fs::Permissions::from_mode(0o755)).unwrap();
    // A real IMOD command-file command starts with `$`; vmstopy owns the
    // conversion and emits `CHUNK DONE` because processchunks passes `-c`.
    fs::write(work.join("tiny-001.com"), "$echo processchunks-tiny\n").unwrap();
    let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap());
    let result = Command::new(env!("CARGO_BIN_EXE_processchunks"))
        .current_dir(&work)
        .args(["-g", "1", "tiny"])
        .env("PATH", path)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("PYTHONPATH", root.join("IMOD/pysrc"))
        .output()
        .unwrap();
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
