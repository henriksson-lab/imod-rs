//! `newstack` parity for several input files that differ from each other.
//!
//! `newstack.f90` loops over input files, and inside that loop it re-reads the
//! header (`newstack.f90:1526-1527`), recomputes the binned size to read
//! (`newstack.f90:1570-1571`) and decides rescaling from *that* file's mode and
//! minimum (`newstack.f90:1985-2007`).  Taking any of those once from the first
//! input file is wrong as soon as the files disagree: sections of a wider mode
//! are truncated into the output range instead of being scaled into it, and a
//! file of a different X/Y size is read with the wrong extent.
//!
//! `fixtures/newstack-mixed-{byte,short,small}.mrc` are authored by
//! `fixtures/make-newstack-mixed-inputs.py` straight to the MRC layout.  The
//! `.mrc` and `.txt` goldens beside them are the reference `newstack`'s own
//! output and stdout, the former with the label block (which carries a date
//! stamp) zeroed.

use std::process::Command;

const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

#[test]
fn inputs_of_differing_mode_and_size_match_the_reference_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-nsmixed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for input in ["byte", "short", "small"] {
        let name = format!("newstack-mixed-{input}.mrc");
        std::fs::copy(root.join("fixtures").join(&name), dir.join(&name)).unwrap();
    }

    for (name, args) in [
        (
            "byteshort",
            vec![
                "-input",
                "newstack-mixed-byte.mrc",
                "-input",
                "newstack-mixed-short.mrc",
            ],
        ),
        (
            "shortbyte",
            vec![
                "-input",
                "newstack-mixed-short.mrc",
                "-input",
                "newstack-mixed-byte.mrc",
            ],
        ),
        (
            "bytesmall",
            vec![
                "-input",
                "newstack-mixed-byte.mrc",
                "-input",
                "newstack-mixed-small.mrc",
            ],
        ),
        (
            "smallbyte",
            vec![
                "-input",
                "newstack-mixed-small.mrc",
                "-input",
                "newstack-mixed-byte.mrc",
            ],
        ),
        (
            "smallshortfloat2",
            vec![
                "-float",
                "2",
                "-input",
                "newstack-mixed-small.mrc",
                "-input",
                "newstack-mixed-short.mrc",
            ],
        ),
    ] {
        let out_name = format!("{name}.mrc");
        let output = Command::new(env!("CARGO_BIN_EXE_newstack"))
            .current_dir(&dir)
            .env("AUTODOC_DIR", AUTODOC)
            .args(&args)
            .args(["-output", &out_name])
            .output()
            .expect("newstack executable must start");
        assert!(
            output.status.success(),
            "newstack {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let want_text =
            std::fs::read_to_string(root.join(format!("fixtures/newstack-mixed-{name}.txt")))
                .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            want_text,
            "{name}: stdout must match the reference run"
        );
        let mut got = std::fs::read(dir.join(&out_name)).unwrap();
        let want = std::fs::read(root.join(format!("fixtures/newstack-mixed-{name}.mrc"))).unwrap();
        for byte in &mut got[224..1024] {
            *byte = 0;
        }
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
