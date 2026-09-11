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
    let _ = std::fs::remove_file(&output);
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
    let _ = std::fs::remove_file(&reread);
    let status = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .arg(&output)
        .arg(&reread)
        .status()
        .expect("convertmod must read its WIMP output");
    assert!(status.success());
    let reread_text = std::fs::read_to_string(&reread).expect("roundtrip WIMP output must be text");
    assert_eq!(reread_text.matches("  Object #:").count(), contours);
    // A WIMP input never reaches `imodWriteAsWimp`'s model, so `writeimod`
    // returns `FWRAP_ERROR_NO_MODEL` and `convertmod.f:23-24` writes the file
    // with `store_mod` instead of `imod_to_wmod`.  Its records carry the
    // Fortran field widths, not the C ones, so only the point lines survive
    // the round trip unchanged.
    let source_points: Vec<&str> = text
        .lines()
        .filter(|line| line.starts_with("      ") || line.starts_with("     1"))
        .collect();
    let reread_points: Vec<&str> = reread_text
        .lines()
        .filter(|line| line.starts_with("      ") || line.starts_with("     1"))
        .collect();
    assert_eq!(source_points, reread_points);
    assert!(reread_text.ends_with("\n \n  END\n"));
    let _ = std::fs::remove_file(output);
    let _ = std::fs::remove_file(reread);
}

/// `store_mod.f:34-87` writes every record with a Fortran fixed-width edit
/// descriptor.  Each expected record below was taken from the native reference
/// build run on the same WIMP input:
///
/// ```text
/// AUTODOC_DIR=/tmp/imod-reference-build/autodoc \
/// LD_LIBRARY_PATH=/tmp/imod-reference-build/buildlib \
/// /tmp/imod-reference-build/flib/model/convertmod in.wimp out.wimp
/// ```
#[test]
fn wimp_input_is_written_by_store_mod_with_fortran_field_widths() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let dir = std::env::temp_dir().join(format!("imod-rs-convertmod-store-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Author the WIMP input with the binary path, which already matches the
    // reference build byte for byte.
    let status = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .current_dir(&dir)
        .arg(&fixture)
        .arg("in.wimp")
        .status()
        .unwrap();
    assert!(status.success());
    let result = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .current_dir(&dir)
        .arg("in.wimp")
        .arg("out.wimp")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(0));
    assert!(result.stdout.is_empty());
    assert!(result.stderr.is_empty());
    let text = std::fs::read_to_string(dir.join("out.wimp")).unwrap();
    let lines: Vec<&str> = text.split('\n').collect();
    // `write(20,'(1x,a39,a30)')dummy,model_file(1:30)` blank pads the name to
    // the full 30-character field; `imod_to_wmod` never pads.
    assert_eq!(
        lines[0],
        " Model file name........................out.wimp                      "
    );
    assert_eq!(lines[0].len(), 70);
    // `needmax` is `ifix(sqrt(len_object*float(n_object)/maxlen))` over the
    // Fortran array sizes, not `2 * cnum`; 24000000 * 58 / 61 gives 4776.
    assert_eq!(lines[1], " max # of object....................... 4776");
    assert_eq!(lines[2], " # of node............................. 7076");
    assert_eq!(lines[3], " # of object...........................   58");
    assert_eq!(lines[4], "  Object sequence : ");
    // `'(a11,i8)'`, `'(a12,i10)'` and `'(a16,i1,2x,i3)'`.
    assert_eq!(lines[5], "  Object #:       1");
    assert_eq!(lines[6], " # of point:        61");
    assert_eq!(lines[7], " Display switch:1  247");
    assert_eq!(lines[8], "     #    X       Y       Z      Mark    Label ");
    assert_eq!(text.matches("  Object #:").count(), 58);
    // `write(20,'(a)')' '` then `write(20,'(a5)')'  END'`.
    assert!(text.ends_with("   0\n \n  END\n"));
    assert_eq!(text.len(), 134215);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `store_mod.f:60-82` only emits the label field once `n_clabel` is non-zero,
/// and then only for points `read_mod.f:84-88` recorded in `label_list`.
/// Verified against the reference build on the same input.
#[test]
fn wimp_point_labels_survive_the_store_mod_round_trip() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let dir = std::env::temp_dir().join(format!("imod-rs-convertmod-label-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .current_dir(&dir)
        .arg(&fixture)
        .arg("plain.wimp")
        .status()
        .unwrap();
    assert!(status.success());
    // Give the first four points a `character*10` label in the columns
    // `read_mod.f:72` reads them from.
    let plain = std::fs::read_to_string(dir.join("plain.wimp")).unwrap();
    let mut labelled = String::new();
    let mut added = 0;
    for line in plain.split('\n') {
        if added < 4 && line.len() == 35 && line.ends_with("   0") {
            labelled.push_str(&format!("{:37}LBL{added:07}\n", line));
            added += 1;
        } else {
            labelled.push_str(line);
            labelled.push('\n');
        }
    }
    labelled.pop();
    std::fs::write(dir.join("in.wimp"), &labelled).unwrap();
    assert_eq!(added, 4);
    let result = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .current_dir(&dir)
        .arg("in.wimp")
        .arg("out.wimp")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(0));
    let text = std::fs::read_to_string(dir.join("out.wimp")).unwrap();
    assert!(text.contains("      1    0.00    0.00    0.00   0  LBL0000000\n"));
    assert!(text.contains("      2  249.98  108.74    1.00   0  LBL0000001\n"));
    assert_eq!(text.matches("  LBL").count(), 4);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `read_mod.f:99-102` closes unit 20, prints its two diagnostics and returns
/// `.false.`, and `convertmod.f:16-17` then exits 1.  `ii` and `irec` are
/// never assigned on this path and `i` is left at `max_obj_num + 1` by the
/// `do i=1,max_obj_num` loop at `read_mod.f:49-51`; the reference build prints
/// exactly these three values for this input.
#[test]
fn rejects_format_incompatible_wimp_with_source_stdout_diagnostic_and_no_output() {
    let dir = std::env::temp_dir().join(format!("imod-rs-convertmod-bad-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // The five leading records of a valid WIMP model followed by a record that
    // is neither ` Obj` nor blank, which is the `151` branch of `read_mod`.
    // A *truncated* file is a different case: its `read(20,101,err=151)` hits
    // end of file, which `err=` does not catch, so the reference build dies
    // with a gfortran runtime error, status 2 and an address-bearing backtrace
    // that cannot be reproduced here.
    std::fs::write(
        dir.join("in.wimp"),
        " Model file name........................in.wimp\n\
         \x20max # of object....................... 4776\n\
         \x20# of node............................. 7076\n\
         \x20# of object...........................   58\n\
         \x20 Object sequence : \n\
         GARBAGE\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .current_dir(&dir)
        .arg("in.wimp")
        .arg("out.wimp")
        .output()
        .expect("convertmod executable must start");
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        " Model file format incompatible,input a new one\n\
         \x20 ipt=           0     1000001           0\n\
         \x20Error reading mode file\n"
    );
    assert!(result.stderr.is_empty());
    assert!(!dir.join("out.wimp").exists());
    let _ = std::fs::remove_dir_all(&dir);
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

/// The WIMP branch reaches `open(20,file=newfile,status='new')`
/// (`convertmod.f:23`), which refuses an existing file.  The reference build
/// dies there with status 2 and leaves the file untouched.
#[test]
fn wimp_input_refuses_an_existing_output_file() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let dir = std::env::temp_dir().join(format!("imod-rs-convertmod-new-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .current_dir(&dir)
        .arg(&fixture)
        .arg("in.wimp")
        .status()
        .unwrap();
    assert!(status.success());
    std::fs::write(dir.join("out.wimp"), "keep me\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_convertmod"))
        .current_dir(&dir)
        .arg("in.wimp")
        .arg("out.wimp")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert!(result.stdout.is_empty());
    assert_eq!(
        std::fs::read_to_string(dir.join("out.wimp")).unwrap(),
        "keep me\n"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
