//! Native-golden coverage for `matchrotpairs` (`IMOD/pysrc/matchrotpairs` with
//! its library `IMOD/pysrc/tiltmatch.py`, translated in
//! `src/imod/pysrc/matchrotpairs.rs` and `src/imod/pysrc/tiltmatch.rs`).
//!
//! Every row of `fixtures/matchrotpairs/cases.tsv` was run through the native
//! Python script by `fixtures/make-matchrotpairs-goldens.sh`, with the native
//! `newstack`, `tiltxcorr`, `xfsimplex` and `xfproduct`; see
//! `tests/pysetup_common` for what is compared.  Here `newstack`, `tiltxcorr`
//! and `xfproduct` run in process (they are in the command table).
//! `xfsimplex` runs through `PATH`: when the launcher does not provide it, the
//! reference build's binary is put on `PATH` through a wrapper, and when that
//! is absent too the test reports why and does nothing.  The wider
//! differential behind these goldens is recorded in `TODO.md`.

mod common;
mod pysetup_common;

use std::os::unix::fs::PermissionsExt as _;

#[test]
fn matchrotpairs_matches_native_goldens() {
    let listing = std::process::Command::new(env!("CARGO_BIN_EXE_imod"))
        .output()
        .expect("run imod");
    let has_xfsimplex = String::from_utf8_lossy(&listing.stderr).contains("\n  xfsimplex\n");
    let reference = std::path::Path::new("/tmp/imod-reference-build");
    let bin =
        std::env::temp_dir().join(format!("imod-rs-matchrotpairs-bin-{}", std::process::id()));
    if !has_xfsimplex {
        let native = reference.join("flib/image/xfsimplex");
        if !native.exists() {
            eprintln!(
                "matchrotpairs_cli: skipped -- xfsimplex is neither a command of this crate nor at {}",
                native.display()
            );
            return;
        }
        std::fs::create_dir_all(&bin).unwrap();
        let wrapper = bin.join("xfsimplex");
        std::fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\nLD_LIBRARY_PATH={}/buildlib exec {} \"$@\"\n",
                reference.display(),
                native.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = std::env::var("PATH").unwrap_or_default();
        // SAFETY: this test binary runs this one test; nothing else reads the
        // environment concurrently.
        unsafe { std::env::set_var("PATH", format!("{}:{path}", bin.display())) };
    }
    let failures = pysetup_common::run_cases("matchrotpairs");
    let _ = std::fs::remove_dir_all(&bin);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
