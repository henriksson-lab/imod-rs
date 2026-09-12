//! Every command's usage/help text must be the source's text, byte for byte.
//!
//! `mrc2tif`'s was a paraphrase and `tif2mrc`'s version line dropped the
//! version and build stamp; both printed through `println!` in places where
//! the source uses `printf`, which also reorders the copyright banner under a
//! pipe.  These goldens are the reference binaries' own output with the one
//! line that carries compile metadata removed.

mod common;

fn golden(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/usage-{name}.txt")),
    )
    .unwrap()
}

#[test]
fn mrc2tif_usage_matches_the_reference_text() {
    // `mrc2tif.cpp:30` prints the version on the first line; everything after
    // it is input-independent.
    let out = common::imod_cmd("mrc2tif")
        .output()
        .expect("mrc2tif executable must start");
    let text = String::from_utf8_lossy(&out.stdout);
    let body: String = text.lines().skip(1).map(|l| format!("{l}\n")).collect();
    assert_eq!(body, golden("mrc2tif"));
}

#[test]
fn tif2mrc_usage_matches_the_reference_text() {
    let out = common::imod_cmd("tif2mrc")
        .output()
        .expect("tif2mrc executable must start");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.starts_with("Tif2mrc Version 5.2.17 "),
        "tif2mrc must print the version and build stamp, got {:?}",
        text.lines().next()
    );
    let body: String = text.lines().skip(1).map(|l| format!("{l}\n")).collect();
    assert_eq!(body, golden("tif2mrc"));
}

#[test]
fn clip_usage_matches_the_reference_text() {
    let out = common::imod_cmd("clip")
        .output()
        .expect("clip executable must start");
    let text = String::from_utf8_lossy(&out.stdout);
    let body: String = text.lines().skip(1).map(|l| format!("{l}\n")).collect();
    assert_eq!(body, golden("clip"));
}

/// `mrc2tif.cpp:356-358` sets `convert` for a non-colour PNG/JPEG output
/// *before* `mrcContrastScaling` runs at line 376.  Doing it afterwards leaves
/// a float image with scale 1 and offset 0, so it reaches the writer unscaled.
/// The goldens are the reference `mrc2tif -p`/`-j` output for
/// `fixtures/mrcsec-mode2.mrc` (a float volume), section 0.
#[test]
fn mrc2tif_scales_a_float_image_for_png_and_jpeg_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-m2tpng-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(root.join("fixtures/mrcsec-mode2.mrc"), dir.join("in.mrc")).unwrap();

    for (flag, root_name, golden, ext) in [
        ("-p", "o", "fixtures/mrc2tif-float-scaled.png", "png"),
        ("-j", "j", "fixtures/mrc2tif-float-scaled.jpg", "jpg"),
    ] {
        let out = common::imod_cmd("mrc2tif")
            .current_dir(&dir)
            .args([flag, "-z", "0,0", "in.mrc", root_name])
            .output()
            .expect("mrc2tif executable must start");
        assert!(
            out.status.success(),
            "mrc2tif {flag} failed: {}",
            String::from_utf8_lossy(&out.stdout)
        );
        let got = std::fs::read(dir.join(format!("{root_name}.000.{ext}"))).unwrap();
        let want = std::fs::read(root.join(golden)).unwrap();
        assert_eq!(got, want, "mrc2tif {flag} must match the reference output");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
