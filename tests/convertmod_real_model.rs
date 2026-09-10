//! Real bundled IMOD-model conformance for `convertmod`.

use std::process::Command;

use imod_rs::imod::libimod::imodel_files::imod_read;

#[test]
fn converts_bundled_fiducial_model_to_source_wimp_text() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let source = imod_read(&fixture).expect("bundled model must decode");
    let contours: usize = source.obj.iter().map(|object| object.cont.len()).sum();
    let points: usize = source
        .obj
        .iter()
        .flat_map(|object| &object.cont)
        .map(|contour| contour.pts.len())
        .sum();
    let output =
        std::env::temp_dir().join(format!("imod-rs-convertmod-{}.wimp", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .arg(&fixture)
        .arg(&output)
        .status()
        .expect("convertmod executable must start");
    assert!(status.success());
    let text = std::fs::read_to_string(&output).expect("WIMP output must be text");
    assert!(text.starts_with(&format!(
        " Model file name........................{}\n",
        output.display()
    )));
    assert_eq!(text.matches("  Object #:").count(), contours);
    assert!(text.contains(&format!(
        " # of object........................... {:4}\n",
        contours
    )));
    assert!(text.contains(&format!(
        " # of node............................. {:4}\n",
        2 * points
    )));
    assert!(text.ends_with("\n  END\n"));
    // `convertmod` passes binary models through the legacy Fortran bridge:
    // `openImodData` applies its reference-image normalization and
    // `imodWriteAsWimp` reverses it.  These real points lie on hundredth
    // boundaries where that source f32 round trip is observable.
    assert!(text.contains("    970  328.36  224.00   39.00   0\n"));
    assert!(text.contains("   1200  203.69  503.72   21.00   0\n"));
    let reread = std::env::temp_dir().join(format!(
        "imod-rs-convertmod-reread-{}.wimp",
        std::process::id()
    ));
    let status = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .arg(&output)
        .arg(&reread)
        .status()
        .expect("convertmod must read its WIMP output");
    assert!(status.success());
    let reread_text = std::fs::read_to_string(&reread).expect("roundtrip WIMP output must be text");
    assert_eq!(reread_text.matches("  Object #:").count(), contours);
    assert_eq!(
        text.split_once('\n').expect("source WIMP header").1,
        reread_text
            .split_once('\n')
            .expect("roundtrip WIMP header")
            .1
    );
    assert!(reread_text.ends_with("\n  END\n"));
    let _ = std::fs::remove_file(output);
    let _ = std::fs::remove_file(reread);
}

#[test]
fn rejects_malformed_legacy_wimp_with_source_stdout_diagnostic_and_no_output() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-convertmod-invalid-{}.wimp",
        std::process::id()
    ));
    let output = std::env::temp_dir().join(format!(
        "imod-rs-convertmod-invalid-{}.out.wimp",
        std::process::id()
    ));
    // This starts with the WIMP model-file record but lacks its required
    // count/object records, so `readw_or_imod` follows the source failure path.
    std::fs::write(
        &input,
        " Model file name........................truncated.wimp\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .arg(&input)
        .arg(&output)
        .output()
        .expect("convertmod executable must start");
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        " Error reading mode file\n"
    );
    assert!(result.stderr.is_empty());
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn overwrites_existing_wimp_output_like_source_writeimod() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-convertmod-existing-{}.wimp",
        std::process::id()
    ));
    std::fs::write(&output, "preserve this existing WIMP output\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .arg(&fixture)
        .arg(&output)
        .output()
        .unwrap();
    assert!(result.status.success());
    let text = std::fs::read_to_string(&output).unwrap();
    assert!(text.starts_with(&format!(
        " Model file name........................{}\n",
        output.display()
    )));
    assert!(text.ends_with("\n  END\n"));
    let _ = std::fs::remove_file(output);
}
