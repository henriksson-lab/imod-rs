//! Real bundled IMOD model conformance for `imodjoin`.

use std::ffi::CString;
use std::process::Command;

use imod_rs::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_new, mrc_head_write};
use imod_rs::imod::libimod::imodel::{
    IMODF_FLIPYZ, Icont, Imod, Iobj, Iobjview, Ipoint, Iref_image, Iview, imod_contour_get,
    imod_contour_get_first, imod_contour_get_next, imod_new_object, imod_next_contour,
    imod_next_object, imod_prev_contour, imod_prev_object, imodel_dist,
};
use imod_rs::imod::libimod::imodel_files::{imod_file_write, imod_read};

#[test]
fn joins_selected_objects_from_bundled_fid_model() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let input = imod_read(&fixture).expect("bundled model must decode");
    assert!(!input.obj.is_empty());
    let output = std::env::temp_dir().join(format!("imod-rs-imodjoin-{}.mod", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .args(["-o", "1"])
        .arg(&fixture)
        .args(["-objects", "1"])
        .arg(&fixture)
        .arg(&output)
        .status()
        .expect("imodjoin executable must start");
    assert!(status.success());
    let joined = imod_read(&output).expect("imodjoin output must be an IMOD model");
    assert_eq!(joined.obj.len(), 2);
    assert_eq!(joined.obj[0].name, input.obj[0].name);
    assert_eq!(joined.obj[1].cont.len(), input.obj[0].cont.len());
    let _ = std::fs::remove_file(&output);
}

#[test]
fn source_rejects_both_first_model_replace_and_object_list_before_real_models() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-imodjoin-replace-list-conflict-{}.mod",
        std::process::id()
    ));
    let result = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .args(["-r", "1", "-o", "1"])
        .arg(&fixture)
        .arg(&fixture)
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(3));
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("ERROR: imodjoin - You cannot use both -o and -r with model 1"));
    assert!(stdout.contains("Usage: imodjoin [options] model_1"));
    assert!(result.stderr.is_empty());
    assert!(!output.exists());
}

#[test]
fn changed_colors_restore_the_stock_cycle_color_of_a_newly_created_object() {
    // `imodjoin.c:427` creates the destination with `imodNewObject`, so the
    // colour saved at `:429-431` and restored under -c is the stock colour
    // cycle entry for that index.  Native
    // /tmp/imod-reference-build/imodutil/imodjoin -c BBa_erase.fid
    // BBa_erase.fid out.mod leaves object 1 green (0, 1, 0) and object 2 cyan
    // (0, 1, 1); its output is byte-identical to this crate's.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-imodjoin-changecolor-{}.mod",
        std::process::id()
    ));
    let status = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .arg("-c")
        .arg(&fixture)
        .arg(&fixture)
        .arg(&output)
        .status()
        .expect("imodjoin executable must start");
    assert!(status.success());
    let joined = imod_read(&output).expect("imodjoin output must be an IMOD model");
    assert_eq!(joined.obj.len(), 2);
    assert_eq!(
        (joined.obj[0].red, joined.obj[0].green, joined.obj[0].blue),
        (0., 1., 0.)
    );
    assert_eq!(
        (joined.obj[1].red, joined.obj[1].green, joined.obj[1].blue),
        (0., 1., 1.)
    );
    let _ = std::fs::remove_file(&output);
    let _ = std::fs::remove_file(output.with_file_name(format!(
        "{}~",
        output.file_name().unwrap().to_string_lossy()
    )));
}

#[test]
fn one_input_model_is_rejected_before_the_requested_output_is_touched() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-imodjoin-missing-second-{}.mod",
        std::process::id()
    ));
    let original = b"must not be replaced";
    std::fs::write(&output, original).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .arg(&fixture)
        .arg(&output)
        .output()
        .expect("imodjoin executable must start");
    assert_eq!(result.status.code(), Some(3));
    assert!(
        String::from_utf8_lossy(&result.stdout).contains(
            "Usage: imodjoin [options] model_1 [-o list] model_2 [more models] out_model"
        )
    );
    assert_eq!(std::fs::read(&output).unwrap(), original);
    assert!(
        !output
            .with_file_name(format!(
                "{}~",
                output.file_name().unwrap().to_string_lossy()
            ))
            .exists()
    );
    let _ = std::fs::remove_file(output);
}

#[test]
fn replacement_keeps_colors_and_transfers_expanded_object_views() {
    let stem = format!("imod-rs-imodjoin-views-{}", std::process::id());
    let first_path = std::env::temp_dir().join(format!("{stem}-first.mod"));
    let second_path = std::env::temp_dir().join(format!("{stem}-second.mod"));
    let output_path = std::env::temp_dir().join(format!("{stem}-out.mod"));

    let mut first = Imod {
        obj: vec![Iobj {
            name: name_array("first"),
            red: 0.1,
            green: 0.2,
            blue: 0.3,
            ..Iobj::default()
        }],
        cview: 1,
        view: vec![Iview::default(), Iview::default()],
        ..Imod::default()
    };
    first.view[1].objview = vec![Iobjview {
        red: 0.1,
        green: 0.2,
        blue: 0.3,
        ..Iobjview::default()
    }];
    imod_file_write(&first, &first_path).expect("first real model writes");

    let mut second = Imod {
        obj: vec![Iobj {
            name: name_array("second"),
            red: 0.9,
            green: 0.8,
            blue: 0.7,
            ..Iobj::default()
        }],
        view: vec![Iview::default(), Iview::default(), Iview::default()],
        ..Imod::default()
    };
    second.view[1].objview = vec![Iobjview {
        red: 0.9,
        green: 0.8,
        blue: 0.7,
        ..Iobjview::default()
    }];
    second.view[2].objview = vec![Iobjview {
        red: 0.9,
        green: 0.8,
        blue: 0.7,
        ..Iobjview::default()
    }];
    imod_file_write(&second, &second_path).expect("second real model writes");

    let status = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .args(["-r", "1", "-c"])
        .arg(&first_path)
        .arg(&second_path)
        .arg(&output_path)
        .status()
        .expect("imodjoin executable must start");
    assert!(status.success());
    let output = imod_read(&output_path).expect("joined model must decode");
    assert_eq!(output.obj.len(), 1);
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(output.obj[0].name.as_ptr()) },
        c"second"
    );
    assert_eq!(
        (output.obj[0].red, output.obj[0].green, output.obj[0].blue),
        (0.1, 0.2, 0.3)
    );
    assert_eq!(output.view.len(), 3);
    assert_eq!(
        (
            output.view[2].objview[0].red,
            output.view[2].objview[0].green,
            output.view[2].objview[0].blue
        ),
        (0.1, 0.2, 0.3)
    );
    let _ = std::fs::remove_file(first_path);
    let _ = std::fs::remove_file(second_path);
    let _ = std::fs::remove_file(output_path);
}

#[test]
fn same_volume_no_transform_and_different_volume_origin_paths_are_distinct() {
    let stem = format!("imod-rs-imodjoin-transform-{}", std::process::id());
    let first_path = std::env::temp_dir().join(format!("{stem}-first.mod"));
    let second_path = std::env::temp_dir().join(format!("{stem}-second.mod"));
    let same_path = std::env::temp_dir().join(format!("{stem}-same.mod"));
    let suppress_path = std::env::temp_dir().join(format!("{stem}-suppress.mod"));
    let different_path = std::env::temp_dir().join(format!("{stem}-different.mod"));
    let first = Imod {
        obj: vec![Iobj::default()],
        ref_image: Some(Iref_image::default()),
        ..Imod::default()
    };
    let second = Imod {
        obj: vec![Iobj {
            cont: vec![Icont {
                pts: vec![Ipoint {
                    x: 5.,
                    y: 0.,
                    z: 0.,
                }],
                ..Icont::default()
            }],
            ..Iobj::default()
        }],
        ref_image: Some(Iref_image {
            ctrans: Ipoint {
                x: 2.,
                y: 0.,
                z: 0.,
            },
            otrans: Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            ..Iref_image::default()
        }),
        ..Imod::default()
    };
    imod_file_write(&first, &first_path).expect("first real model writes");
    imod_file_write(&second, &second_path).expect("second real model writes");
    for (options, output) in [
        (Vec::<&str>::new(), &same_path),
        (vec!["-n"], &suppress_path),
        (vec!["-d"], &different_path),
    ] {
        let status = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
            .args(options)
            .arg(&first_path)
            .arg(&second_path)
            .arg(output)
            .status()
            .expect("imodjoin executable must start");
        assert!(status.success());
    }
    assert_eq!(imod_read(&same_path).unwrap().obj[1].cont[0].pts[0].x, 3.);
    assert_eq!(
        imod_read(&suppress_path).unwrap().obj[1].cont[0].pts[0].x,
        5.
    );
    assert_eq!(
        imod_read(&different_path).unwrap().obj[1].cont[0].pts[0].x,
        4.
    );
    for path in [
        first_path,
        second_path,
        same_path,
        suppress_path,
        different_path,
    ] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn absent_first_reference_uses_source_unit_scale_for_different_volumes() {
    let stem = format!("imod-rs-imodjoin-missing-ref-{}", std::process::id());
    let first_path = std::env::temp_dir().join(format!("{stem}-first.mod"));
    let second_path = std::env::temp_dir().join(format!("{stem}-second.mod"));
    let output_path = std::env::temp_dir().join(format!("{stem}-out.mod"));
    imod_file_write(
        &Imod {
            obj: vec![Iobj::default()],
            ..Imod::default()
        },
        &first_path,
    )
    .unwrap();
    imod_file_write(
        &Imod {
            obj: vec![Iobj::default()],
            ..Imod::default()
        },
        &second_path,
    )
    .unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_imodjoin"))
            .arg("-d")
            .arg(&first_path)
            .arg(&second_path)
            .arg(&output_path)
            .status()
            .unwrap()
            .success()
    );
    let output = imod_read(&output_path).unwrap();
    assert_eq!(
        output.ref_image.unwrap().cscale,
        Ipoint {
            x: 1.,
            y: 1.,
            z: 1.
        }
    );
    for path in [first_path, second_path, output_path] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn image_reference_header_scale_controls_different_volume_transform() {
    let stem = format!("imod-rs-imodjoin-image-reference-{}", std::process::id());
    let first_path = std::env::temp_dir().join(format!("{stem}-first.mod"));
    let second_path = std::env::temp_dir().join(format!("{stem}-second.mod"));
    let image_path = std::env::temp_dir().join(format!("{stem}-reference.mrc"));
    let output_path = std::env::temp_dir().join(format!("{stem}-out.mod"));
    let first = Imod {
        obj: vec![Iobj::default()],
        ref_image: Some(Iref_image::default()),
        ..Imod::default()
    };
    let second = Imod {
        obj: vec![Iobj {
            cont: vec![Icont {
                pts: vec![Ipoint {
                    x: 5.,
                    y: 0.,
                    z: 0.,
                }],
                ..Icont::default()
            }],
            ..Iobj::default()
        }],
        ref_image: Some(Iref_image {
            ctrans: Ipoint {
                x: 2.,
                y: 0.,
                z: 0.,
            },
            otrans: Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            ..Iref_image::default()
        }),
        ..Imod::default()
    };
    imod_file_write(&first, &first_path).unwrap();
    imod_file_write(&second, &second_path).unwrap();
    let image_name = CString::new(image_path.to_string_lossy().as_bytes()).unwrap();
    let image_file = unsafe { libc::fopen(image_name.as_ptr(), c"wb".as_ptr()) };
    assert!(!image_file.is_null());
    let mut header = unsafe { std::mem::zeroed::<MrcHeader>() };
    assert_eq!(mrc_head_new(&mut header, 10, 1, 1, 2), 0);
    header.mx = 10;
    header.xlen = 20.;
    assert_eq!(unsafe { mrc_head_write(image_file, &mut header) }, 0);
    unsafe { libc::fclose(image_file) };
    let status = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .arg("-i")
        .arg(&image_path)
        .arg(&first_path)
        .arg(&second_path)
        .arg(&output_path)
        .status()
        .expect("imodjoin executable must start");
    assert!(status.success());
    assert_eq!(imod_read(&output_path).unwrap().obj[1].cont[0].pts[0].x, 2.);
    for path in [first_path, second_path, image_path, output_path] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn keep_scale_and_keep_flip_preserve_their_source_transform_states() {
    let stem = format!("imod-rs-imodjoin-scale-flip-{}", std::process::id());
    let first_path = std::env::temp_dir().join(format!("{stem}-first.mod"));
    let second_path = std::env::temp_dir().join(format!("{stem}-second.mod"));
    let scale_path = std::env::temp_dir().join(format!("{stem}-scale.mod"));
    let keep_scale_path = std::env::temp_dir().join(format!("{stem}-keep-scale.mod"));
    let flip_path = std::env::temp_dir().join(format!("{stem}-flip.mod"));
    let keep_flip_path = std::env::temp_dir().join(format!("{stem}-keep-flip.mod"));
    let first = Imod {
        obj: vec![Iobj::default()],
        ref_image: Some(Iref_image {
            cscale: Ipoint {
                x: 2.,
                y: 1.,
                z: 1.,
            },
            ..Iref_image::default()
        }),
        ..Imod::default()
    };
    let second = Imod {
        obj: vec![Iobj {
            cont: vec![Icont {
                pts: vec![Ipoint {
                    x: 5.,
                    y: 2.,
                    z: 3.,
                }],
                ..Icont::default()
            }],
            ..Iobj::default()
        }],
        flags: IMODF_FLIPYZ,
        ref_image: Some(Iref_image {
            ctrans: Ipoint {
                x: 2.,
                y: 0.,
                z: 0.,
            },
            otrans: Ipoint {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            ..Iref_image::default()
        }),
        ..Imod::default()
    };
    imod_file_write(&first, &first_path).unwrap();
    imod_file_write(&second, &second_path).unwrap();
    for (options, output) in [
        (vec!["-d"], &scale_path),
        (vec!["-d", "-s"], &keep_scale_path),
        (Vec::<&str>::new(), &flip_path),
        (vec!["-f"], &keep_flip_path),
    ] {
        let status = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
            .args(options)
            .arg(&first_path)
            .arg(&second_path)
            .arg(output)
            .status()
            .unwrap();
        assert!(status.success());
    }
    assert_eq!(imod_read(&scale_path).unwrap().obj[1].cont[0].pts[0].x, 2.);
    assert_eq!(
        imod_read(&keep_scale_path).unwrap().obj[1].cont[0].pts[0].x,
        4.
    );
    let flipped = imod_read(&flip_path).unwrap();
    assert_eq!(
        (
            flipped.obj[1].cont[0].pts[0].y,
            flipped.obj[1].cont[0].pts[0].z
        ),
        (3., 2.)
    );
    assert_eq!(flipped.flags & IMODF_FLIPYZ, 0);
    let retained = imod_read(&keep_flip_path).unwrap();
    assert_eq!(
        (
            retained.obj[1].cont[0].pts[0].y,
            retained.obj[1].cont[0].pts[0].z
        ),
        (2., 3.)
    );
    for path in [
        first_path,
        second_path,
        scale_path,
        keep_scale_path,
        flip_path,
        keep_flip_path,
    ] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn suppressing_transforms_still_reconciles_the_joined_model_flip_state() {
    let stem = format!("imod-rs-imodjoin-suppress-flip-{}", std::process::id());
    let first_path = std::env::temp_dir().join(format!("{stem}-first.mod"));
    let second_path = std::env::temp_dir().join(format!("{stem}-second.mod"));
    let output_path = std::env::temp_dir().join(format!("{stem}-out.mod"));
    imod_file_write(
        &Imod {
            obj: vec![Iobj::default()],
            ..Imod::default()
        },
        &first_path,
    )
    .unwrap();
    imod_file_write(
        &Imod {
            flags: IMODF_FLIPYZ,
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![Ipoint {
                        x: 1.,
                        y: 2.,
                        z: 3.,
                    }],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            // No ref image: C still executes its flip reconciliation.
            ..Imod::default()
        },
        &second_path,
    )
    .unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_imodjoin"))
            .arg("-n")
            .arg(&first_path)
            .arg(&second_path)
            .arg(&output_path)
            .status()
            .unwrap()
            .success()
    );
    let output = imod_read(&output_path).unwrap();
    assert_eq!(
        output.obj[1].cont[0].pts[0],
        Ipoint {
            x: 1.,
            y: 3.,
            z: 2.
        }
    );
    for path in [first_path, second_path, output_path] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn existing_output_is_backed_up_before_the_joined_model_is_written() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let output = std::env::temp_dir().join(format!(
        "imod-rs-imodjoin-backup-{}.mod",
        std::process::id()
    ));
    let backup = std::path::PathBuf::from(format!("{}~", output.display()));
    let old = Imod {
        name: name_array("model replaced by imodjoin"),
        ..Imod::default()
    };
    imod_file_write(&old, &output).expect("old output writes");
    imod_file_write(
        &Imod {
            name: name_array("stale backup replaced by source imodBackupFile"),
            ..Imod::default()
        },
        &backup,
    )
    .expect("stale backup writes");
    let status = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .arg(&fixture)
        .arg(&fixture)
        .arg(&output)
        .status()
        .expect("imodjoin executable must start");
    assert!(status.success());
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(imod_read(&backup).unwrap().name.as_ptr()) },
        c"model replaced by imodjoin"
    );
    assert!(!imod_read(&output).unwrap().obj.is_empty());
    let _ = std::fs::remove_file(output);
    let _ = std::fs::remove_file(backup);
}

#[test]
fn source_option_parse_read_object_and_image_errors_use_the_right_exit_and_stream() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let missing =
        std::env::temp_dir().join(format!("imod-rs-imodjoin-missing-{}", std::process::id()));
    let malformed = std::env::temp_dir().join(format!(
        "imod-rs-imodjoin-malformed-header-{}.mrc",
        std::process::id()
    ));
    std::fs::write(&malformed, [0_u8; 100]).unwrap();
    let invalid_option = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .args(["-q", "model.mod", "out.mod"])
        .output()
        .unwrap();
    assert_eq!(invalid_option.status.code(), Some(3));
    assert!(
        String::from_utf8_lossy(&invalid_option.stdout).contains("Invalid option before model 1")
    );
    assert!(invalid_option.stderr.is_empty());
    let bad_list = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .args(["-o", "1x"])
        .output()
        .unwrap();
    assert_eq!(bad_list.status.code(), Some(3));
    assert!(
        String::from_utf8_lossy(&bad_list.stdout)
            .contains("Error parsing object list before model 1")
    );
    assert!(bad_list.stderr.is_empty());
    let read_error = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .arg(&missing)
        .arg("out.mod")
        .output()
        .unwrap();
    assert_eq!(read_error.status.code(), Some(1));
    // `exitError` writes the exit prefix and message to stdout, as the native
    // imodjoin does for each of the diagnostics below.
    assert!(
        String::from_utf8_lossy(&read_error.stdout)
            .contains("ERROR: imodjoin -  Error reading file for model 1")
    );
    let image_error = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .arg("-i")
        .arg(&missing)
        .output()
        .unwrap();
    assert_eq!(image_error.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&image_error.stdout).contains("ERROR: imodjoin -  Couldn't open")
    );
    let header_error = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .arg("-i")
        .arg(&malformed)
        .output()
        .unwrap();
    assert_eq!(header_error.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&header_error.stdout)
            .contains("ERROR: imodjoin -  Reading header from")
    );
    let object_error = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .args(["-o", "999"])
        .arg(&fixture)
        .arg(&fixture)
        .arg(&missing)
        .output()
        .unwrap();
    assert_eq!(object_error.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&object_error.stdout)
            .contains("ERROR: imodjoin -  Invalid object number 999 for model 1")
    );
    let _ = std::fs::remove_file(malformed);
}

#[test]
fn imodel_dist_uses_xy_distance_from_a_decoded_binary_model() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let mut model = imod_read(&fixture).expect("bundled binary model must decode");
    let (object, contour) = model
        .obj
        .iter()
        .enumerate()
        .find_map(|(object, object_data)| {
            object_data
                .cont
                .iter()
                .enumerate()
                .find(|(_, contour_data)| contour_data.pts.len() >= 2)
                .map(|(contour, _)| (object, contour))
        })
        .expect("bundled binary model must contain a two-point contour");
    model.cindex.object = object as i32;
    model.cindex.contour = contour as i32;
    model.cindex.point = 1;
    let points = &model.obj[object].cont[contour].pts;
    let expected = (((points[1].x - points[0].x) as f64).powi(2)
        + ((points[1].y - points[0].y) as f64).powi(2))
    .sqrt();
    assert_eq!(imodel_dist(&model), expected);
}

#[test]
fn imod_new_object_assigns_source_defaults_and_stock_colors_in_a_binary_model() {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-new-object-{}.mod",
        std::process::id()
    ));
    let mut model = Imod::default();
    for _ in 0..36 {
        assert_eq!(imod_new_object(&mut model), 0);
    }
    assert_eq!(
        (
            model.cindex.object,
            model.cindex.contour,
            model.cindex.point
        ),
        (35, -1, -1)
    );
    assert_eq!(
        (model.obj[0].red, model.obj[0].green, model.obj[0].blue),
        (0., 1., 0.)
    );
    assert_eq!(
        (model.obj[5].red, model.obj[5].green, model.obj[5].blue),
        (1., 0., 0.)
    );
    assert_eq!(
        (model.obj[35].red, model.obj[35].green, model.obj[35].blue),
        (0., 1., 0.)
    );
    assert_eq!(model.obj[0].flags, (1 << 27) | (1 << 28));
    assert_eq!(model.obj[0].drawmode, 1);
    assert_eq!(model.obj[0].symbol, 1);
    assert_eq!(model.obj[0].symsize, 3);
    assert_eq!(model.obj[0].clips.normal[0].z, -1.);
    assert_eq!(
        (
            model.obj[0].ambient,
            model.obj[0].diffuse,
            model.obj[0].specular,
            model.obj[0].shininess,
            model.obj[0].valwhite,
        ),
        (102, 255, 127, 4, 255)
    );
    imod_file_write(&model, &path).expect("source-default binary model writes");
    let decoded = imod_read(&path).expect("source-default binary model decodes");
    assert_eq!(decoded.obj.len(), 36);
    assert_eq!(
        (
            decoded.obj[35].red,
            decoded.obj[35].green,
            decoded.obj[35].blue
        ),
        (0., 1., 0.)
    );
    assert_eq!(decoded.obj[0].flags, (1 << 27) | (1 << 28));
    assert_eq!(decoded.obj[0].drawmode, 1);
    assert_eq!(decoded.obj[0].symbol, 1);
    assert_eq!(decoded.obj[0].symsize, 3);
    let _ = std::fs::remove_file(path);
}

#[test]
fn imod_next_and_prev_object_clamp_indices_from_a_decoded_binary_model() {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-object-traverse-{}.mod",
        std::process::id()
    ));
    let model = Imod {
        obj: vec![
            Iobj {
                cont: vec![
                    Icont {
                        pts: vec![Ipoint::default(), Ipoint::default()],
                        ..Icont::default()
                    },
                    Icont {
                        pts: vec![Ipoint::default()],
                        ..Icont::default()
                    },
                ],
                ..Iobj::default()
            },
            Iobj::default(),
            Iobj {
                cont: vec![Icont {
                    pts: vec![Ipoint::default()],
                    ..Icont::default()
                }],
                ..Iobj::default()
            },
        ],
        ..Imod::default()
    };
    imod_file_write(&model, &path).expect("object traversal binary model writes");
    let mut decoded = imod_read(&path).expect("object traversal binary model decodes");

    decoded.cindex.object = -3;
    decoded.cindex.contour = 9;
    decoded.cindex.point = 9;
    assert_eq!(imod_next_object(Some(&mut decoded)), 0);
    assert_eq!(
        (
            decoded.cindex.object,
            decoded.cindex.contour,
            decoded.cindex.point
        ),
        (0, 1, 0)
    );
    assert_eq!(imod_next_object(Some(&mut decoded)), 1);
    assert_eq!(
        (
            decoded.cindex.object,
            decoded.cindex.contour,
            decoded.cindex.point
        ),
        (1, -1, -1)
    );
    decoded.cindex.contour = 4;
    decoded.cindex.point = 4;
    assert_eq!(imod_next_object(Some(&mut decoded)), 2);
    assert_eq!(
        (
            decoded.cindex.object,
            decoded.cindex.contour,
            decoded.cindex.point
        ),
        (2, 0, 0)
    );
    decoded.cindex.contour = 7;
    decoded.cindex.point = 7;
    assert_eq!(imod_next_object(Some(&mut decoded)), 2);
    assert_eq!(
        (
            decoded.cindex.object,
            decoded.cindex.contour,
            decoded.cindex.point
        ),
        (2, 7, 7)
    );
    assert_eq!(imod_prev_object(Some(&mut decoded)), 1);
    assert_eq!(
        (
            decoded.cindex.object,
            decoded.cindex.contour,
            decoded.cindex.point
        ),
        (1, -1, -1)
    );
    decoded.cindex.object = 8;
    decoded.cindex.contour = 9;
    decoded.cindex.point = 9;
    assert_eq!(imod_prev_object(Some(&mut decoded)), 2);
    assert_eq!(
        (
            decoded.cindex.object,
            decoded.cindex.contour,
            decoded.cindex.point
        ),
        (2, 0, 0)
    );
    assert_eq!(imod_next_object(None), -1);
    assert_eq!(imod_prev_object(None), -1);
    let _ = std::fs::remove_file(path);
}

#[test]
fn imod_next_and_prev_contour_clamp_indices_from_a_decoded_binary_model() {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-contour-traverse-{}.mod",
        std::process::id()
    ));
    let model = Imod {
        obj: vec![
            Iobj {
                cont: vec![
                    Icont {
                        pts: vec![Ipoint::default(), Ipoint::default()],
                        ..Icont::default()
                    },
                    Icont::default(),
                    Icont {
                        pts: vec![Ipoint::default()],
                        ..Icont::default()
                    },
                ],
                ..Iobj::default()
            },
            Iobj::default(),
        ],
        ..Imod::default()
    };
    imod_file_write(&model, &path).expect("contour traversal binary model writes");
    let mut decoded = imod_read(&path).expect("contour traversal binary model decodes");

    decoded.cindex.object = 0;
    decoded.cindex.contour = -1;
    decoded.cindex.point = 7;
    assert_eq!(imod_prev_contour(Some(&mut decoded)), 2);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (2, 0));
    assert_eq!(imod_prev_contour(Some(&mut decoded)), 1);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (1, -1));
    decoded.cindex.point = 7;
    assert_eq!(imod_prev_contour(Some(&mut decoded)), 0);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (0, 1));
    decoded.cindex.point = 8;
    assert_eq!(imod_prev_contour(Some(&mut decoded)), 0);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (0, 8));

    decoded.cindex.contour = -1;
    decoded.cindex.point = 7;
    assert_eq!(imod_next_contour(Some(&mut decoded)), 0);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (0, 1));
    decoded.cindex.point = 7;
    assert_eq!(imod_next_contour(Some(&mut decoded)), 1);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (1, -1));
    assert_eq!(imod_next_contour(Some(&mut decoded)), 2);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (2, -1));
    decoded.cindex.point = 4;
    assert_eq!(imod_next_contour(Some(&mut decoded)), 2);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (2, 4));

    decoded.cindex.object = 1;
    decoded.cindex.contour = 5;
    decoded.cindex.point = 6;
    assert_eq!(imod_prev_contour(Some(&mut decoded)), -1);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (5, 6));
    assert_eq!(imod_next_contour(Some(&mut decoded)), -1);
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (5, 6));
    decoded.cindex.object = -1;
    assert_eq!(imod_prev_contour(Some(&mut decoded)), -1);
    assert_eq!(decoded.cindex.contour, -1);
    decoded.cindex.contour = 4;
    assert_eq!(imod_next_contour(Some(&mut decoded)), -1);
    assert_eq!(decoded.cindex.contour, 4);
    assert_eq!(imod_prev_contour(None), -1);
    assert_eq!(imod_next_contour(None), -1);
    let _ = std::fs::remove_file(path);
}

#[test]
fn imod_contour_get_accessors_select_decoded_binary_model_contours() {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-contour-access-{}.mod",
        std::process::id()
    ));
    let model = Imod {
        obj: vec![
            Iobj {
                cont: vec![
                    Icont {
                        time: 10,
                        pts: vec![Ipoint::default(), Ipoint::default()],
                        ..Icont::default()
                    },
                    Icont {
                        time: 20,
                        pts: vec![Ipoint::default()],
                        ..Icont::default()
                    },
                ],
                ..Iobj::default()
            },
            Iobj::default(),
        ],
        ..Imod::default()
    };
    imod_file_write(&model, &path).expect("contour accessor binary model writes");
    let mut decoded = imod_read(&path).expect("contour accessor binary model decodes");

    decoded.cindex.object = 0;
    decoded.cindex.contour = 1;
    decoded.cindex.point = 2;
    assert_eq!(
        imod_contour_get(Some(&decoded)).map(|contour| contour.time),
        Some(20)
    );
    decoded.cindex.contour = 2;
    assert!(imod_contour_get(Some(&decoded)).is_none());
    decoded.cindex.object = -1;
    assert!(imod_contour_get(Some(&decoded)).is_none());

    decoded.cindex.object = 0;
    decoded.cindex.contour = 1;
    decoded.cindex.point = 9;
    assert_eq!(
        imod_contour_get_first(Some(&mut decoded)).map(|contour| contour.time),
        Some(10)
    );
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (0, 1));
    assert_eq!(
        imod_contour_get_next(Some(&mut decoded)).map(|contour| contour.time),
        Some(20)
    );
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (1, 0));
    assert!(imod_contour_get_next(Some(&mut decoded)).is_none());
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (1, 0));
    decoded.cindex.contour = -1;
    assert_eq!(
        imod_contour_get_next(Some(&mut decoded)).map(|contour| contour.time),
        Some(10)
    );

    decoded.cindex.object = 1;
    decoded.cindex.contour = 5;
    decoded.cindex.point = 6;
    assert!(imod_contour_get_first(Some(&mut decoded)).is_none());
    assert_eq!((decoded.cindex.contour, decoded.cindex.point), (-1, -1));
    assert!(imod_contour_get_next(Some(&mut decoded)).is_none());
    assert!(imod_contour_get(None).is_none());
    assert!(imod_contour_get_first(None).is_none());
    assert!(imod_contour_get_next(None).is_none());
    let _ = std::fs::remove_file(path);
}

#[test]
fn read_error_uses_the_source_exit_prefix_on_stdout() {
    // `readerr` calls `exitError`, which routes through `PipSetError` and
    // writes "<prefix> <message>" to stdout, leaving stderr empty.  The native
    // imodjoin prints exactly "ERROR: imodjoin -  Error reading file for
    // model 1" there (note the two spaces from the prefix's own trailing one).
    let missing = std::env::temp_dir().join(format!(
        "imod-rs-imodjoin-prefix-missing-{}.mod",
        std::process::id()
    ));
    let output = std::env::temp_dir().join(format!(
        "imod-rs-imodjoin-prefix-out-{}.mod",
        std::process::id()
    ));
    let result = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .arg(&missing)
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(result.stderr.is_empty());
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "ERROR: imodjoin -  Error reading file for model 1\n"
    );
    let _ = std::fs::remove_file(output);
}

/// Test-only: builds a fixed-size NUL-padded model/object name array.
fn name_array<const N: usize>(text: &str) -> [std::ffi::c_char; N] {
    let mut name = [0; N];
    for (slot, byte) in name.iter_mut().zip(text.as_bytes()) {
        *slot = *byte as std::ffi::c_char;
    }
    name
}
