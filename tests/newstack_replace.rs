//! `newstack -replace`: writing sections into an existing output file in place.
//!
//! The source opens the output `OLD` (`newstack.f90:1173`), never creates a
//! header for it, writes each section at `listReplace(i)`
//! (`newstack.f90:1759, 2738`), leaves `dmean` alone so the file keeps the mean
//! it already had, and finishes with `iiuWriteHeader(2, title, -1, ...)`
//! (`newstack.f90:2765`) -- `labFlag = -1` changes no label at all.
//!
//! `fixtures/newstack-replace-target.mrc` is the reference `newstack`'s own
//! copy of `fixtures/newstack-mixed-byte.mrc` with its label block zeroed, so
//! that -- because `-replace` does not touch labels -- every golden beside it
//! is byte-deterministic with no masking.

use std::process::Command;

const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

#[test]
fn replaced_sections_match_the_reference_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-nsreplace-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(
        root.join("fixtures/newstack-mixed-byte.mrc"),
        dir.join("in.mrc"),
    )
    .unwrap();

    for (name, args) in [
        ("sec1to0", vec!["-secs", "1", "-replace", "0"]),
        ("sec02to21", vec!["-secs", "0,2", "-replace", "2,1"]),
        ("float2", vec!["-secs", "2", "-replace", "0", "-float", "2"]),
        (
            "scale",
            vec!["-secs", "0,1", "-replace", "1,2", "-scale", "0,200"],
        ),
    ] {
        // Each case starts from the same untouched existing output file.
        std::fs::copy(
            root.join("fixtures/newstack-replace-target.mrc"),
            dir.join("o.mrc"),
        )
        .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_newstack"))
            .current_dir(&dir)
            .env("AUTODOC_DIR", AUTODOC)
            .args(["-input", "in.mrc"])
            .args(&args)
            .args(["-output", "o.mrc"])
            .output()
            .expect("newstack executable must start");
        assert!(
            output.status.success(),
            "newstack {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let want_text =
            std::fs::read_to_string(root.join(format!("fixtures/newstack-replace-{name}.txt")))
                .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            want_text,
            "{name}: stdout must match the reference run"
        );
        let got = std::fs::read(dir.join("o.mrc")).unwrap();
        let want =
            std::fs::read(root.join(format!("fixtures/newstack-replace-{name}.mrc"))).unwrap();
        assert_eq!(
            got.len(),
            want.len(),
            "{name}: output size must match the reference"
        );
        let differing = (0..want.len()).filter(|i| want[*i] != got[*i]).count();
        assert_eq!(
            differing, 0,
            "{name}: {differing} bytes differ from the reference output"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
