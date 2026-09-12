//! `newstack -mdoc`: the autodoc (`.mdoc`) metadata transfer.
//!
//! The source attaches or opens an autodoc for each input file
//! (`newstack.f90:1535-1565, 1663-1665`), copies the input's global section and
//! every collection other than `ZValue` and `T` into the output's autodoc
//! (`newstack.f90:1941-1947`), transfers one `ZValue` section per written
//! section under its new number (`newstack.f90:2709-2713`), overwrites
//! `TiltAngle` when `-tilt` was given (`newstack.f90:2721-2735`), and writes
//! `<output>.mdoc` before the final header (`newstack.f90:2745-2750`).
//!
//! The goldens are the reference `newstack`'s own `.mdoc` output and stdout.
//! `.mdoc` files carry no timestamp, so they compare byte for byte.

mod common;

const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

#[test]
fn mdoc_metadata_transfer_matches_the_reference_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-nsmdoc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for image in ["newstack-mdoc-byte.mrc", "newstack-mdoc-second.mrc"] {
        std::fs::copy(
            root.join("fixtures/newstack-mixed-byte.mrc"),
            dir.join(image),
        )
        .unwrap();
        let mdoc = format!("{image}.mdoc");
        std::fs::copy(root.join("fixtures").join(&mdoc), dir.join(&mdoc)).unwrap();
    }
    std::fs::write(dir.join("tilts.tlt"), "-12.0\n0.0\n12.0\n").unwrap();

    for (name, args) in [
        ("plain", vec!["-mdoc", "-input", "newstack-mdoc-byte.mrc"]),
        (
            "reorder",
            vec!["-mdoc", "-input", "newstack-mdoc-byte.mrc", "-secs", "2,0"],
        ),
        (
            "twofiles",
            vec![
                "-mdoc",
                "-input",
                "newstack-mdoc-byte.mrc",
                "-input",
                "newstack-mdoc-second.mrc",
            ],
        ),
        (
            "tilts",
            vec![
                "-mdoc",
                "-tilt",
                "tilts.tlt",
                "-input",
                "newstack-mdoc-byte.mrc",
                "-secs",
                "2,0",
            ],
        ),
    ] {
        let _ = std::fs::remove_file(dir.join("o.mrc"));
        let _ = std::fs::remove_file(dir.join("o.mrc.mdoc"));
        let output = common::imod_cmd("newstack")
            .current_dir(&dir)
            .env("AUTODOC_DIR", AUTODOC)
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
            std::fs::read_to_string(root.join(format!("fixtures/newstack-mdoc-{name}.txt")))
                .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            want_text,
            "{name}: stdout must match the reference run"
        );
        let got = std::fs::read_to_string(dir.join("o.mrc.mdoc"))
            .unwrap_or_else(|_| panic!("{name}: no output mdoc was written"));
        let want =
            std::fs::read_to_string(root.join(format!("fixtures/newstack-mdoc-{name}.mdoc")))
                .unwrap();
        assert_eq!(got, want, "{name}: output mdoc must match the reference");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
