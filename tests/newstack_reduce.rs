//! `newstack` image-reduction parity: `-bin` with a factor that does not divide
//! the input size, `-shrink` on the read, and `-shrink` with `-bin` (the
//! post-read `zoomFiltInterp` route).
//!
//! `fixtures/newstack-reduce-input.mrc` is authored by
//! `fixtures/make-newstack-reduce-input.py` from the MRC layout itself.  Its
//! 21x13 size is deliberately not a multiple of 3, so `getBinnedSize`'s X and Y
//! offsets are nonzero and `readBinnedOrReduced` must start off the corner.
//! The `.mrc` goldens beside it are the reference `newstack`'s own output with
//! the label block (which carries a date stamp) zeroed.

mod common;

const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

fn mask_labels(bytes: &mut [u8]) {
    for byte in &mut bytes[224..1024] {
        *byte = 0;
    }
}

#[test]
fn reduction_options_match_the_reference_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-nsreduce-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(
        root.join("fixtures/newstack-reduce-input.mrc"),
        dir.join("in.mrc"),
    )
    .unwrap();

    for (name, args) in [
        ("bin3", vec!["-bin", "3"]),
        ("shrink3", vec!["-shrink", "3"]),
        ("shrink25", vec!["-shrink", "2.5"]),
        ("shrink3bin2", vec!["-shrink", "3", "-bin", "2"]),
        ("bin3origin", vec!["-bin", "3", "-origin"]),
    ] {
        let out_name = format!("{name}.mrc");
        let output = common::imod_cmd("newstack")
            .current_dir(&dir)
            .env("AUTODOC_DIR", AUTODOC)
            .args(&args)
            .args(["-input", "in.mrc", "-output", &out_name])
            .output()
            .expect("newstack executable must start");
        assert!(
            output.status.success(),
            "newstack {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut got = std::fs::read(dir.join(&out_name)).unwrap();
        let want =
            std::fs::read(root.join(format!("fixtures/newstack-reduce-{name}.mrc"))).unwrap();
        mask_labels(&mut got);
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
