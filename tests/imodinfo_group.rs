//! Real binary OGRP selection through the source-shaped imodinfo CLI.

use std::process::Command;

use imod_rs::imod::imodutil::imodinfo::imodinfo_special;
use imod_rs::imod::libimod::imodel::{
    IMOD_OBJFLAG_OFF, IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_OUT, Icont, Imesh, Imod, Iobj, Iobj_group,
    Ipoint,
};
use imod_rs::imod::libimod::imodel_files::{imod_file_write, imod_read};
use imod_rs::imod::libimod::istore::{Istore, StoreUnion};

#[test]
fn group_option_falls_through_to_source_chart_mode_for_binary_ogrp_chunk() {
    let path =
        std::env::temp_dir().join(format!("imod-rs-imodinfo-group-{}.mod", std::process::id()));
    let model = Imod {
        obj: vec![
            Iobj {
                name: "not selected".into(),
                ..Iobj::default()
            },
            Iobj {
                name: "selected".into(),
                ..Iobj::default()
            },
        ],
        group_list: vec![Iobj_group {
            name: *b"second only\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            obj_list: vec![1],
        }],
        ..Imod::default()
    };
    imod_file_write(&model, &path).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-g", "1"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("#Obj       Cyl. Vol"));
    assert!(stdout.contains("   2              0"));
    assert!(!stdout.contains("   1              0"));
    let normal = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .arg(&path)
        .output()
        .unwrap();
    assert!(normal.status.success());
    let normal = String::from_utf8(normal.stdout).unwrap();
    assert!(normal.contains("NAME:  selected"));
    assert!(normal.contains("NAME:  not selected"));
    let _ = std::fs::remove_file(path);
}

#[test]
fn standard_report_emits_source_object_drawing_and_color_preamble() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-object-preamble-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                name: "preamble".into(),
                flags: IMOD_OBJFLAG_OFF | IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_OUT,
                red: 0.1,
                green: 0.2,
                blue: 0.3,
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("object drawing is turned off"));
    assert!(stdout.contains("object uses open contours."));
    assert!(stdout.contains("contours in object are inside out."));
    assert!(stdout.contains("color (red, green, blue) = (0.1, 0.2, 0.3)"));
    let _ = std::fs::remove_file(input);
}

#[test]
fn standard_report_scales_closed_contour_area_by_source_pixel_size_squared() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-scaled-area-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            pixsize: 2.,
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![
                        Ipoint {
                            x: 0.,
                            y: 0.,
                            z: 0.,
                        },
                        Ipoint {
                            x: 4.,
                            y: 0.,
                            z: 0.,
                        },
                        Ipoint {
                            x: 0.,
                            y: 3.,
                            z: 0.,
                        },
                    ],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .contains(", length = 24,  area = 24\n")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn source_special_edit_writes_its_modified_binary_model() {
    let output = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-special-{}.mod",
        std::process::id()
    ));
    let mut model = Imod {
        obj: (0..138).map(|_| Iobj::default()).collect(),
        ..Imod::default()
    };
    model.obj[1].cont = (0..12).map(|_| Icont::default()).collect();
    imodinfo_special(&mut model, output.to_str().unwrap());
    let output_model = imod_read(&output).unwrap();
    assert_eq!(output_model.obj.len(), 136);
    assert_eq!(output_model.obj[1].cont.len(), 11);
    let _ = std::fs::remove_file(output);
}

#[test]
fn file_option_writes_cli_report_and_backs_up_existing_output() {
    let input =
        std::env::temp_dir().join(format!("imod-rs-imodinfo-file-{}.mod", std::process::id()));
    let output =
        std::env::temp_dir().join(format!("imod-rs-imodinfo-file-{}.txt", std::process::id()));
    let backup = std::path::PathBuf::from(format!("{}~", output.display()));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                name: "written to file".into(),
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    std::fs::write(&output, "old report").unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-f", output.to_str().unwrap()])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(std::fs::read_to_string(&backup).unwrap(), "old report");
    assert!(
        std::fs::read_to_string(&output)
            .unwrap()
            .contains("NAME:  written to file")
    );
    for path in [input, output, backup] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn list_and_group_conflict_backs_up_source_output_before_error_for_binary_model() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-list-group-conflict-{}.mod",
        std::process::id()
    ));
    let output = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-list-group-conflict-{}.txt",
        std::process::id()
    ));
    let backup = std::path::PathBuf::from(format!("{}~", output.display()));
    imod_file_write(
        &Imod {
            obj: vec![Iobj::default()],
            group_list: vec![Iobj_group {
                name: [0; 32],
                obj_list: vec![0],
            }],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    std::fs::write(&output, "existing source report").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args([
            "-f",
            output.to_str().unwrap(),
            "-o",
            "1",
            "-g",
            "1",
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        "ERROR: imodinfo - You cannot enter both an object list and an object group\n"
    );
    assert_eq!(
        std::fs::read_to_string(&backup).unwrap(),
        "existing source report"
    );
    assert_eq!(std::fs::read_to_string(&output).unwrap(), "");
    for path in [input, output, backup] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn malformed_h_option_uses_source_help_diagnostic_before_binary_model_read() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-malformed-help-{}.mod",
        std::process::id()
    ));
    imod_file_write(&Imod::default(), &input).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-huh", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        "ERROR: imodinfo - Unknown option -huh; enter -help for help\n"
    );
    assert!(result.stdout.is_empty());
    let _ = std::fs::remove_file(input);
}

#[test]
fn malformed_model_reports_and_continues_to_next_binary_model() {
    let bad = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-malformed-continue-{}.mod",
        std::process::id()
    ));
    let good = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-good-after-malformed-{}.mod",
        std::process::id()
    ));
    std::fs::write(&bad, b"not an IMOD binary model").unwrap();
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                name: "valid second model".into(),
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &good,
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args([bad.to_str().unwrap(), good.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("imodinfo: Error ("));
    assert!(stdout.contains(&format!("reading imod model. ({})", bad.display())));
    assert!(stdout.contains(&format!("# MODEL {}", good.display())));
    assert!(stdout.contains("NAME:  valid second model"));
    assert!(result.stderr.is_empty());
    for path in [bad, good] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn standard_report_emits_source_ref_image_coordinates_for_binary_model() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-ref-image-header-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            ref_image: Some(imod_rs::imod::libimod::imodel::Iref_image {
                cscale: Ipoint {
                    x: 2.,
                    y: 3.,
                    z: 4.,
                },
                ctrans: Ipoint {
                    x: 5.,
                    y: 6.,
                    z: 7.,
                },
                crot: Ipoint {
                    x: 8.,
                    y: 9.,
                    z: 10.,
                },
                ..Default::default()
            }),
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("# Model to Image index coords:"));
    assert!(stdout.contains("#      SCALE  = ( 2, 3, 4)"));
    assert!(stdout.contains("#      OFFSET = ( 5, 6, 7)"));
    assert!(stdout.contains("#      ANGLES = ( 8, 9, 10)"));
    let _ = std::fs::remove_file(input);
}

#[test]
fn full_report_uses_source_cylinder_volume_and_surface_scaling() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-cylinder-scaling-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            pixsize: 2.,
            zscale: 3.,
            obj: vec![Iobj {
                name: "scaled triangle".into(),
                cont: vec![Icont {
                    pts: vec![
                        Ipoint {
                            x: 0.,
                            y: 0.,
                            z: 0.,
                        },
                        Ipoint {
                            x: 4.,
                            y: 0.,
                            z: 0.,
                        },
                        Ipoint {
                            x: 0.,
                            y: 3.,
                            z: 0.,
                        },
                    ],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let rust = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-F", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(rust.status.success(), "{rust:?}");
    let rust = String::from_utf8(rust.stdout).unwrap();
    assert!(
        rust.contains("Cylinder Volume         = 144 pixels^3"),
        "{rust}"
    );
    assert!(
        rust.contains("Cylinder Surface Area   = 144 pixels^2"),
        "{rust}"
    );
    for line in [
        "Color  = (Red, Green, Blue, Alpha) = (0, 0, 0, 0)",
        "Ambient Light  = 0",
        "Diffuse Light  = 0",
        "Specular Light = 0",
        "Shininess      = 0",
        "Bounding Box   = { (0, 0, 0), (4, 3, 0)}",
    ] {
        assert!(rust.contains(line), "missing {line} in {rust}");
    }

    let native = std::path::Path::new("/tmp/imod-reference-build/imodutil/imodinfo");
    if native.exists() {
        let reference = Command::new(native)
            .env("LD_LIBRARY_PATH", "/tmp/imod-reference-build/buildlib")
            .args(["-F", input.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(reference.status.success(), "{reference:?}");
        let reference = String::from_utf8(reference.stdout).unwrap();
        assert!(reference.contains("Cylinder Volume         = 144 pixels^3"));
        assert!(reference.contains("Cylinder Surface Area   = 144 pixels^2"));
        for line in [
            "Color  = (Red, Green, Blue, Alpha) = (0, 0, 0, 0)",
            "Ambient Light  = 0",
            "Diffuse Light  = 0",
            "Specular Light = 0",
            "Shininess      = 0",
            "Bounding Box   = { (0, 0, 0), (4, 3, 0)}",
        ] {
            assert!(reference.contains(line), "missing {line} in {reference}");
        }
    }
    let _ = std::fs::remove_file(input);
}

#[test]
fn full_report_emits_source_clip_and_secondary_values_block() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-full-clip-{}.mod",
        std::process::id()
    ));
    let mut object = Iobj {
        name: "clipped open line".into(),
        flags: IMOD_OBJFLAG_OPEN,
        cont: vec![Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 4.,
                    y: 0.,
                    z: 2.,
                },
            ],
            ..Icont::default()
        }],
        ..Iobj::default()
    };
    object.clips.count = 1;
    object.clips.flags = 1;
    object.clips.normal[0] = Ipoint {
        x: 1.,
        y: 2.,
        z: 4.,
    };
    object.clips.point[0] = Ipoint {
        x: 3.,
        y: 4.,
        z: 5.,
    };
    imod_file_write(
        &Imod {
            zscale: 2.,
            obj: vec![object],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let rust = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-F", "-t", "1", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(rust.status.success(), "{rust:?}");
    let rust = String::from_utf8(rust.stdout).unwrap();
    for line in [
        "Clip 0 Normal    = (1, 2, 2)",
        "Clip 0 Point     = (3, 4, 5)",
        "Clipped and/or subsetted values:",
        "Cylinder Volume         = 0 pixels^3",
    ] {
        assert!(rust.contains(line), "missing {line} in {rust}");
    }
    let native = std::path::Path::new("/tmp/imod-reference-build/imodutil/imodinfo");
    if native.exists() {
        let native = Command::new(native)
            .env("LD_LIBRARY_PATH", "/tmp/imod-reference-build/buildlib")
            .args(["-F", "-t", "1", input.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(native.status.success(), "{native:?}");
        let native = String::from_utf8(native.stdout).unwrap();
        for line in [
            "Clip 0 Normal    = (1, 2, 2)",
            "Clip 0 Point     = (3, 4, 5)",
            "Clipped and/or subsetted values:",
            "Cylinder Volume         = 0 pixels^3",
        ] {
            assert!(native.contains(line), "missing {line} in {native}");
        }
    }
    let _ = std::fs::remove_file(input);
}

#[test]
fn standard_report_emits_source_empty_model_line_for_binary_model() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-empty-model-{}.mod",
        std::process::id()
    ));
    imod_file_write(&Imod::default(), &input).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .contains("Model has no objects!!!")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn standard_report_ends_with_source_model_separator_for_binary_model() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-model-separator-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            obj: vec![Iobj::default()],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert!(String::from_utf8(result.stdout).unwrap().ends_with("\n\n"));
    let _ = std::fs::remove_file(input);
}

#[test]
fn length_mode_emits_source_units_header_for_binary_model() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-length-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            units: -9,
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![Ipoint::default()],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-l"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("# Obj Cont Pnts Length (in nm)\n#------------------------")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn point_mode_emits_source_object_header_for_sized_binary_contour() {
    let input =
        std::env::temp_dir().join(format!("imod-rs-imodinfo-point-{}.mod", std::process::id()));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                name: "sized".into(),
                cont: vec![Icont {
                    pts: vec![Ipoint::default()],
                    sizes: vec![2.],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-p"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("#Object 1 data, sized")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn ratio_mode_keeps_source_row_for_zero_length_binary_contour() {
    let input =
        std::env::temp_dir().join(format!("imod-rs-imodinfo-ratio-{}.mod", std::process::id()));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![Ipoint::default(), Ipoint::default(), Ipoint::default()],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-r"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8(output.stdout).unwrap().contains("1 NaN"));
    let _ = std::fs::remove_file(input);
}

#[test]
fn ellipse_mode_emits_source_heading_units_and_contour_row() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-imodinfo-ellipse-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            units: -9,
            pixsize: 1.,
            obj: vec![Iobj {
                name: "ellipse".into(),
                cont: vec![Icont {
                    pts: vec![
                        Ipoint {
                            x: 0.,
                            y: 0.,
                            z: 0.,
                        },
                        Ipoint {
                            x: 2.,
                            y: 0.,
                            z: 0.,
                        },
                        Ipoint {
                            x: 0.,
                            y: 1.,
                            z: 0.,
                        },
                    ],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &input,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-e"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(output.status.success());
    let output = String::from_utf8(output.stdout).unwrap();
    assert!(output.contains("axes (nm)"));
    assert!(output.contains("semi-major   semi-minor"));
    assert!(output.contains("   1"));
    let _ = std::fs::remove_file(input);
}

#[test]
fn ascii_file_mode_writes_implemented_binary_model_categories_and_backup() {
    let input =
        std::env::temp_dir().join(format!("imod-rs-imodinfo-ascii-{}.mod", std::process::id()));
    let output =
        std::env::temp_dir().join(format!("imod-rs-imodinfo-ascii-{}.txt", std::process::id()));
    let backup = std::path::PathBuf::from(format!("{}~", output.display()));
    let mut view = imod_rs::imod::libimod::imodel::Iview::default();
    view.fovy = 30.;
    view.clips.count = 1;
    view.label[..17].copy_from_slice(b"source view label");
    let model = Imod {
        units: -9,
        alpha: 1.,
        beta: 2.,
        gamma: 3.,
        flags: imod_rs::imod::libimod::imodel::IMODF_TILTOK
            | imod_rs::imod::libimod::imodel::IMODF_OTRANS_ORIGIN,
        ref_image: Some(imod_rs::imod::libimod::imodel::Iref_image {
            crot: Ipoint {
                x: 4.,
                y: 5.,
                z: 6.,
            },
            otrans: Ipoint {
                x: 7.,
                y: 8.,
                z: 9.,
            },
            ..Default::default()
        }),
        view: vec![Default::default(), view],
        obj: vec![Iobj {
            name: "ascii object".into(),
            fillred: 1,
            flags: 1 << 8,
            ambient: 2,
            store: vec![Istore {
                type_: 10,
                flags: 4,
                index: StoreUnion { i: 0 },
                value: StoreUnion { f: 3.5 },
            }],
            cont: vec![Icont {
                pts: vec![Ipoint::default()],
                sizes: vec![2.],
                ..Icont::default()
            }],
            mesh: vec![
                Imesh {
                    flag: 1,
                    vert: vec![Ipoint {
                        x: 1.,
                        y: 2.,
                        z: 3.,
                    }],
                    list: vec![0],
                    ..Imesh::default()
                },
                Imesh {
                    flag: 1 << 24,
                    vert: vec![Ipoint {
                        x: 4.,
                        y: 5.,
                        z: 6.,
                    }],
                    list: vec![0],
                    ..Imesh::default()
                },
            ],
            ..Iobj::default()
        }],
        ..Imod::default()
    };
    imod_file_write(&model, &input).unwrap();
    std::fs::write(&output, "old ascii").unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-a", "-f", output.to_str().unwrap()])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(std::fs::read_to_string(&backup).unwrap(), "old ascii");
    let text = std::fs::read_to_string(&output).unwrap();
    // `imodWriteAscii` rewinds the `-f` stream in the source, so the command
    // replaces its preliminary report with ASCII rather than appending it.
    assert!(text.starts_with("# imod ascii file version 2.0\n\n"));
    assert!(!text.contains("# MODEL "));
    for line in [
        "# imod ascii file version 2.0",
        "angles 1 2 3",
        "units      nm",
        "refcurscale",
        "refcurrot 4 5 6",
        "refoldtrans 7 8 9",
        "view 1",
        "viewlabel source view label",
        "globalclips",
        "object 0",
        "Fillcolor 1 0 0",
        "fill",
        "ambient   2",
        "contour 0 0 1 3.5",
        "object 0 1 1",
        "mesh 0 1 1",
        "# end of IMOD model",
    ] {
        assert!(text.contains(line), "missing {line}");
    }
    assert!(text.find("Fillcolor 1 0 0").unwrap() < text.find("fill\n").unwrap());
    let stdout = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .args(["-a"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(stdout.status.success());
    assert!(
        String::from_utf8(stdout.stdout)
            .unwrap()
            .contains("# imod ascii file version 2.0")
    );
    for path in [input, output, backup] {
        let _ = std::fs::remove_file(path);
    }
}
