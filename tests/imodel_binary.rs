//! Binary-model coverage for source-shaped `imodel.c` operations.

use imod_rs::imod::libimod::ilabel::{
    Ilabel, imod_label_dup, imod_label_item_add, imod_label_item_delete, imod_label_item_get,
    imod_label_item_match, imod_label_item_move, imod_label_match, imod_label_name,
    imod_label_name_get, imod_label_new, imod_label_print, imod_label_read, imod_label_write,
};
use imod_rs::imod::libimod::imesh::{
    IMESH_FLAG_RES_SHIFT, IMOD_MESH_BGNPOLY, IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2,
    IMOD_MESH_END, IMOD_MESH_ENDPOLY, imesh_copy_skip_list, imesh_params_delete, imesh_params_dup,
    imesh_params_new, imesh_sort_surfaces, imesh_surface_area, imesh_volume, imod_mesh_add_index,
    imod_mesh_add_normal, imod_mesh_add_vert, imod_mesh_copy, imod_mesh_delete_index,
    imod_mesh_dup, imod_mesh_get_bbox, imod_mesh_get_index, imod_mesh_get_max_index,
    imod_mesh_get_max_vert, imod_mesh_get_vert, imod_mesh_get_verts, imod_mesh_insert_index,
    imod_mesh_interp_cont, imod_mesh_nearest_res, imod_mesh_new, imod_mesh_poly_norm_factors,
    imodel_mesh_add,
};
use imod_rs::imod::libimod::imodel::imod_checksum;
use imod_rs::imod::libimod::imodel::{
    ICONT_OPEN, IMODF_FLIPYZ, Icont, Imesh, Imod, Iobj, Ipoint, imod_clean_surf, imod_delete_point,
    imod_get_bounding_box, imod_get_cur_mesh_surf, imod_get_filename, imod_get_flipped,
    imod_get_max_object, imod_get_max_time, imod_get_pixel_size, imod_get_z_scale,
    imod_insert_point, imod_new_contour, imod_new_point, imod_set_cur_mesh_surf, imodel_maxpt,
    imodel_minpt,
};
use imod_rs::imod::libimod::imodel::{IMOD_CLIPSIZE, Iobjview, Iview, imod_new, imod_new_object};
use imod_rs::imod::libimod::imodel_files::{
    byteswap, imod_close_file, imod_fgetline, imod_file_write, imod_from_vms_floats,
    imod_get_bytes, imod_get_floats, imod_get_ints, imod_open_file, imod_put_bytes,
    imod_put_floats, imod_put_ints, imod_put_scaled_points, imod_read, imod_read_file,
    imod_test_if_model_file, imod_write_file, imod_write_skip_mesh, swap_longs, tovmsfloat,
};
use imod_rs::imod::libimod::iobj::{
    imod_object_checksum, imod_object_copy, imod_object_copy_clear, imod_object_default,
    imod_object_delete, imod_object_dup, imod_object_new, imod_objects_delete, imod_objects_new,
};
use imod_rs::imod::libimod::iview::{
    BYTES_PER_OBJVIEW, VIEW_STRSIZE, imod_imnx_new, imod_objview_complete, imod_objview_delete,
    imod_objview_from_object, imod_objview_to_object, imod_objviews_free, imod_view_default,
    imod_view_default_scale, imod_view_model_default, imod_view_model_new, imod_view_new,
    imod_view_store, imod_view_use, imod_view_write,
};

#[test]
fn imod_get_bounding_box_uses_contours_or_meshes_per_binary_object() {
    let path =
        std::env::temp_dir().join(format!("imod-rs-imodel-bounds-{}.mod", std::process::id()));
    let model = Imod {
        obj: vec![
            Iobj {
                cont: vec![Icont {
                    pts: vec![
                        Ipoint {
                            x: -2.,
                            y: 5.,
                            z: 3.,
                        },
                        Ipoint {
                            x: 4.,
                            y: -1.,
                            z: 8.,
                        },
                    ],
                    ..Icont::default()
                }],
                // Source imodObjectGetBBox ignores mesh vertices whenever
                // an object has contours.
                mesh: vec![Imesh {
                    vert: vec![Ipoint {
                        x: -100.,
                        y: -100.,
                        z: -100.,
                    }],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            },
            Iobj {
                cont: Vec::new(),
                mesh: vec![Imesh {
                    vert: vec![
                        Ipoint {
                            x: -4.,
                            y: 2.,
                            z: -7.,
                        },
                        Ipoint {
                            x: 9.,
                            y: 8.,
                            z: 6.,
                        },
                    ],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            },
        ],
        ..Imod::default()
    };
    imod_file_write(&model, &path).unwrap();
    let model = imod_read(&path).unwrap();
    let mut min = Ipoint::default();
    let mut max = Ipoint::default();
    imod_get_bounding_box(&model, &mut min, &mut max);
    assert_eq!(
        min,
        Ipoint {
            x: -4.,
            y: -1.,
            z: -7.,
        }
    );
    assert_eq!(
        max,
        Ipoint {
            x: 9.,
            y: 8.,
            z: 8.,
        }
    );
    let mut contour_max = Ipoint::default();
    imodel_maxpt(&model, &mut contour_max);
    assert_eq!(
        contour_max,
        Ipoint {
            x: 4.,
            y: 5.,
            z: 8.,
        }
    );
    let mut contour_min = Ipoint::default();
    imodel_minpt(&model, &mut contour_min);
    assert_eq!(
        contour_min,
        Ipoint {
            x: -2.,
            y: -1.,
            z: 3.,
        }
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn imodel_scalar_accessors_preserve_binary_model_header_fields() {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-accessors-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            obj: vec![Iobj::default(), Iobj::default()],
            name: name_array("accessor-model"),
            zscale: 2.5,
            pixsize: 1.25,
            flags: IMODF_FLIPYZ,
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let model = imod_read(&path).unwrap();
    assert_eq!(imod_get_max_object(&model), 2);
    assert_eq!(imod_get_z_scale(&model), 2.5);
    assert_eq!(imod_get_pixel_size(&model), 1.25);
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(imod_get_filename(&model)) },
        c"accessor-model"
    );
    assert_eq!(imod_get_flipped(&model), IMODF_FLIPYZ);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn imod_get_max_time_scans_positive_contour_times_in_binary_model() {
    let path = std::env::temp_dir().join(format!("imod-rs-imodel-time-{}.mod", std::process::id()));
    imod_file_write(
        &Imod {
            obj: vec![
                Iobj {
                    cont: vec![
                        Icont {
                            time: -2,
                            ..Icont::default()
                        },
                        Icont {
                            time: 3,
                            ..Icont::default()
                        },
                    ],
                    ..Iobj::default()
                },
                Iobj {
                    cont: vec![Icont {
                        time: 7,
                        ..Icont::default()
                    }],
                    ..Iobj::default()
                },
            ],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let model = imod_read(&path).unwrap();
    assert_eq!(imod_get_max_time(None), 0);
    assert_eq!(imod_get_max_time(Some(&model)), 7);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn imod_clean_surf_uses_maximum_contour_or_mesh_surface_after_binary_read() {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-clean-surf-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            obj: vec![
                Iobj {
                    cont: vec![Icont {
                        surf: 4,
                        ..Icont::default()
                    }],
                    mesh: vec![Imesh {
                        surf: 7,
                        ..Imesh::default()
                    }],
                    ..Iobj::default()
                },
                Iobj {
                    cont: vec![Icont {
                        surf: -2,
                        ..Icont::default()
                    }],
                    ..Iobj::default()
                },
            ],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let mut model = imod_read(&path).unwrap();
    model.obj[0].surfsize = -1;
    model.obj[1].surfsize = -1;
    imod_clean_surf(&mut model);
    assert_eq!(model.obj[0].surfsize, 7);
    assert_eq!(model.obj[1].surfsize, 0);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn imod_new_contour_inherits_current_surface_and_open_state_after_binary_read() {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-new-contour-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                cont: vec![Icont {
                    surf: 5,
                    flags: ICONT_OPEN,
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let mut model = imod_read(&path).unwrap();
    model.cindex.object = 0;
    model.cindex.contour = 0;
    assert_eq!(imod_new_contour(&mut model), 0);
    assert_eq!((model.cindex.contour, model.cindex.point), (1, -1));
    assert_eq!(model.obj[0].cont[1].surf, 5);
    assert_eq!(model.obj[0].cont[1].flags, ICONT_OPEN);
    model.cindex.object = -1;
    assert_eq!(imod_new_contour(&mut model), -1);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn imod_point_addition_and_insertion_follow_source_indices_after_binary_read() {
    let path =
        std::env::temp_dir().join(format!("imod-rs-imodel-points-{}.mod", std::process::id()));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![
                        Ipoint {
                            x: 1.,
                            y: 0.,
                            z: 0.,
                        },
                        Ipoint {
                            x: 3.,
                            y: 0.,
                            z: 0.,
                        },
                    ],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let mut model = imod_read(&path).unwrap();
    model.cindex.object = 0;
    model.cindex.contour = 0;
    model.cindex.point = 0;
    assert_eq!(
        imod_new_point(
            &mut model,
            Some(Ipoint {
                x: 5.,
                y: 0.,
                z: 0.
            })
        ),
        3
    );
    assert_eq!(model.cindex.point, 1);
    assert_eq!(
        imod_insert_point(
            Some(&mut model),
            Some(Ipoint {
                x: 2.,
                y: 0.,
                z: 0.
            }),
            -5
        ),
        4
    );
    assert_eq!(model.cindex.point, 0);
    assert_eq!(model.obj[0].cont[0].pts[0].x, 2.);
    assert_eq!(
        imod_insert_point(
            Some(&mut model),
            Some(Ipoint {
                x: 7.,
                y: 0.,
                z: 0.
            }),
            99
        ),
        5
    );
    assert_eq!(model.cindex.point, 4);
    model.cindex.contour = -1;
    assert_eq!(imod_new_point(&mut model, Some(Ipoint::default())), 0);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn imod_delete_point_retains_source_empty_contour_removal_rules_after_binary_read() {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-delete-point-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![
                        Ipoint::default(),
                        Ipoint {
                            x: 1.,
                            y: 0.,
                            z: 0.,
                        },
                    ],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let mut model = imod_read(&path).unwrap();
    model.cindex.object = 0;
    model.cindex.contour = 0;
    model.cindex.point = 1;
    assert_eq!(imod_delete_point(&mut model), 1);
    assert_eq!(model.cindex.point, 0);
    assert_eq!(imod_delete_point(&mut model), 0);
    assert_eq!(model.cindex.point, -1);
    assert!(model.obj[0].cont[0].pts.is_empty());
    assert_eq!(imod_delete_point(&mut model), 0);
    assert!(model.obj[0].cont.is_empty());
    assert_eq!(imod_delete_point(&mut model), -1);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn imodel_maxpt_returns_source_no_contour_sentinel_after_binary_read() {
    let path =
        std::env::temp_dir().join(format!("imod-rs-imodel-maxpt-{}.mod", std::process::id()));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                mesh: vec![Imesh {
                    vert: vec![Ipoint {
                        x: 7.,
                        y: 8.,
                        z: 9.,
                    }],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let model = imod_read(&path).unwrap();
    let mut maximum = Ipoint::default();
    imodel_maxpt(&model, &mut maximum);
    assert_eq!(
        maximum,
        Ipoint {
            x: -1.,
            y: -1.,
            z: -1.,
        }
    );
    let mut minimum = Ipoint::default();
    imodel_minpt(&model, &mut minimum);
    assert_eq!(
        minimum,
        Ipoint {
            x: -1.,
            y: -1.,
            z: -1.,
        }
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn current_mesh_surface_uses_source_contourless_binary_model_conditions() {
    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-current-mesh-surface-{}.mod",
        std::process::id()
    ));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                surfsize: 3,
                mesh: vec![Imesh {
                    surf: 3,
                    vert: vec![Ipoint::default()],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let mut model = imod_read(&path).unwrap();
    model.cindex.object = 0;
    model.cur_mesh_surf = 2;
    assert_eq!(imod_get_cur_mesh_surf(&model), 2);
    imod_set_cur_mesh_surf(&mut model, 3);
    assert_eq!(imod_get_cur_mesh_surf(&model), 3);
    imod_set_cur_mesh_surf(&mut model, 4);
    assert_eq!(imod_get_cur_mesh_surf(&model), 3);
    model.obj[0].cont.push(Icont::default());
    assert_eq!(imod_get_cur_mesh_surf(&model), -1);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn objt_extra_words_survive_real_binary_model_roundtrip() {
    let path = std::env::temp_dir().join(format!("imod-rs-iobj-extra-{}.mod", std::process::id()));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                extra: [17, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let model = imod_read(&path).unwrap();
    assert_eq!(
        model.obj[0].extra,
        [17, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41]
    );
    assert_eq!(imod_object_checksum(&model.obj[0], 0), 50.);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn iobj_source_defaults_persist_through_real_binary_objt() {
    let mut object = Iobj::default();
    imod_object_default(&mut object);
    assert_eq!(imod_objects_new(2).unwrap().len(), 2);
    assert_eq!(imod_object_new().unwrap().symbol, 1);
    let path =
        std::env::temp_dir().join(format!("imod-rs-iobj-default-{}.mod", std::process::id()));
    imod_file_write(
        &Imod {
            obj: vec![object],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let model = imod_read(&path).unwrap();
    assert_eq!(
        (model.obj[0].red, model.obj[0].green, model.obj[0].blue),
        (0.5, 0.5, 0.5)
    );
    assert_eq!(
        (
            model.obj[0].ambient,
            model.obj[0].diffuse,
            model.obj[0].specular,
            model.obj[0].shininess
        ),
        (102, 255, 127, 4)
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn iobj_copy_delete_and_dup_own_decoded_binary_object_data() {
    let path = std::env::temp_dir().join(format!("imod-rs-iobj-copy-{}.mod", std::process::id()));
    imod_file_write(
        &Imod {
            obj: vec![Iobj {
                name: name_array("source"),
                cont: vec![Icont {
                    pts: vec![Ipoint {
                        x: 1.,
                        y: 2.,
                        z: 3.,
                    }],
                    ..Icont::default()
                }],
                mesh: vec![Imesh {
                    vert: vec![Ipoint {
                        x: 4.,
                        y: 5.,
                        z: 6.,
                    }],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        },
        &path,
    )
    .unwrap();
    let mut source = imod_read(&path).unwrap().obj.remove(0);
    let mut copied = Iobj::default();
    assert_eq!(imod_object_copy(&source, &mut copied), 0);
    source.cont[0].pts[0].x = 99.;
    assert_eq!(copied.cont[0].pts[0].x, 1.);
    let duplicate = imod_object_dup(&copied).unwrap();
    assert_eq!(duplicate.mesh[0].vert[0].z, 6.);
    assert_eq!(imod_object_copy_clear(&copied, &mut source), 0);
    assert!(source.cont.is_empty() && source.mesh.is_empty());
    assert_eq!(imod_object_delete(&mut copied), 0);
    assert!(copied.cont.is_empty() && copied.mesh.is_empty());
    let mut objects = vec![source, copied];
    assert_eq!(imod_objects_delete(&mut objects), 0);
    assert!(objects.is_empty());
    assert_eq!(imod_objects_delete(&mut objects), -1);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn model_and_object_name_bytes_past_the_nul_and_csum_survive_a_binary_round_trip() {
    // `imodel_files.c:837`/`:862` read IMOD_STRSIZE/IOBJ_STRSIZE raw bytes into
    // the fixed `char name[]` arrays and `:300`/`:352` write every one of them
    // back; `:845`/`:307` do the same for the checksum word.  `BBa_erase.fid`
    // carries 114 bytes of heap residue after "IMOD-NewModel\0" and
    // `IMOD/com/empty.seed` carries csum 396.  Native
    // /tmp/imod-reference-build/imodutil/imodjoin reproduces both verbatim:
    // `imodjoin BBa_erase.fid BBa_erase.fid out.mod` and
    // `imodjoin empty.seed BBa_erase.fid out.mod` are byte-identical to this
    // crate's imodjoin output.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
    let raw = std::fs::read(&fixture).unwrap();
    let model = imod_read(&fixture).unwrap();
    let name: Vec<u8> = model.name.iter().map(|byte| *byte as u8).collect();
    assert_eq!(&name[..], &raw[8..8 + 128]);
    assert_eq!(name[13], 0);
    assert_eq!(name[14..].iter().filter(|byte| **byte != 0).count(), 35);
    assert_eq!(model.csum, 0);

    let path = std::env::temp_dir().join(format!(
        "imod-rs-imodel-namebytes-{}.mod",
        std::process::id()
    ));
    imod_file_write(&model, &path).unwrap();
    let written = std::fs::read(&path).unwrap();
    assert_eq!(&written[8..8 + 128], &raw[8..8 + 128]);
    let reread = imod_read(&path).unwrap();
    assert_eq!(reread.name, model.name);
    for (rewritten, original) in reread.obj.iter().zip(&model.obj) {
        assert_eq!(rewritten.name, original.name);
    }
    std::fs::remove_file(&path).unwrap();

    let seed = imod_read(&root.join("IMOD/com/empty.seed")).unwrap();
    assert_eq!(seed.csum, 396);
    imod_file_write(&seed, &path).unwrap();
    let written = std::fs::read(&path).unwrap();
    assert_eq!(&written[224..228], &[0, 0, 1, 0x8c]);
    std::fs::remove_file(&path).unwrap();
}

/// Test-only: builds a fixed-size NUL-padded model/object name array.
fn name_array<const N: usize>(text: &str) -> [std::ffi::c_char; N] {
    let mut name = [0; N];
    for (slot, byte) in name.iter_mut().zip(text.as_bytes()) {
        *slot = *byte as std::ffi::c_char;
    }
    name
}

// ---------------------------------------------------------------------------
// Differential against the pinned C `IMOD/libimod/imodel_files.c`
//
// `scratchpad/mfwork/read_driver.c` is compiled against the pinned headers and
// linked to the reference `libimod` in `/tmp/imod-reference-build/buildlib`.
// It calls `imodTestIfModelFile` and `imodRead` on a model file and dumps every
// field of the resulting `Imod`; floats are printed as raw IEEE bit patterns so
// the comparison is exact.  The two fixtures below are hand-built binary models
// (`scratchpad/mfwork/gen.py`): a legacy `V0.1` file that exercises
// `imodel_read_v01`, `imodel_read_object_v01` and `imodel_read_contour_v01`
// including their "skip unknown chunk" loops, and a `V1.2` file carrying
// `SLAN`, `MEPA`, `SKLI`, `OLBL` and `LABL` chunks plus an object whose header
// declares more contours than the file contains.
//
// `OLBL`/`LABL` label data is not represented by this translation
// (`ilabel.c` has no module and `Iobj`/`Icont` carry no `label` member), so the
// driver does not print label contents; that the fields *after* those chunks
// still read correctly is what shows the chunks are skipped in step with the
// source.

const V01_MODEL_HEX: &str = concat!(
    "494d4f4456302e31763031206d6f64656c00000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000020000000180",
    "0000003c0000000200000000000000010000000100000000000000ff3fc0000040200000406000003f8000003f800000",
    "3f40000000000000000000000000000000000003000000804010000000000001000030393f000000be8000003e000000",
    "4a554e4b0000000801020304050607084f424a54666972737420763031206f626a656374000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "00000000000000020000000300000001000000023e8000003f0000003f40000000000000000000000000004d00000000",
    "434f4e5400000003000000000000000000000005504e54530000000000000000404000003f8000004000000040400000",
    "400000004080000040400000585452410000000461626364434f4e5400000002000000080000000100000002504e5453",
    "bf800000c0000000c04000004120000041a0000041f000004f424a547365636f6e640000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000900000000000000013f80000000000000000000000000000700000003",
    "000000ff00000000",
);

const V12_MODEL_HEX: &str = concat!(
    "494d4f4456312e32763132206368756e6b73000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000020000000180",
    "0000003c0000000200003200000000010000000100000000000000ff3fc0000040200000406000003f8000003f800000",
    "3f40000000000009000000090000000900000003000000804010000000000001000030393f000000be8000003e000000",
    "4f424a546f626a207a65726f000000000000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000300000006000000090000000c0000000f00000012",
    "00000015000000180000001b0000001e0000002100000024000000270000002a0000002d000000030000000800000001",
    "000000013dcccccd3e4ccccd3e99999a00000005020000040103003c00000000000000044f4c424c0000002200000002",
    "00000004746f702100000000000000036161610000000100000003626262434f4e540000000200000000000000000000",
    "00013f80000040000000404000004080000040a0000040c000004c41424c000000140000000100000002686900000000",
    "000000027a7a53495a45000000083fc0000040200000434f4e5400000001000000100000000200000003c0e00000c100",
    "0000c1100000494d41540000001064c832070b16212c000003e7010203044d4550410000004c00000011000000020000",
    "0003000000020000000100000001000000000000000a000000003f0000003fc000000000000042c80000000000004348",
    "00003e8000003f0000004060000000000000534b4c490000000800000004000000094f424a546f626a206f6e65000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "00000000000000000000000000000000000000000000000000000000020000000000000000013f6666663f4ccccd3f33",
    "33330000000201030101000000000000000000000000534c414e0000003c000000010000000000000000000000000000",
    "00000000000000000000736c696365723000000000000000000000000000000000000000000000000000534c414e0000",
    "003c000000024120000041a0000041f000003f8000004000000040400000736c69636572310000000000000000000000",
    "000000000000000000000000000049454f46",
);

const NATIVE_V01_DUMP: &str = r#"test 0
name [v01 model]
max 512 384 60 objsize 2 flags 0 draw 1 mouse 1 bl 0 wl 255
off 3fc00000 40200000 40600000 scale 3f800000 3f800000 3f400000
cindex 0 0 0 res 3 thresh 128 units 1 csum 12345
pix 40100000 abg 3f000000 be800000 3e000000
views 1 cview 0 slicer 0 groups 0
obj 0 [first v01 object] cont 2 mesh 0 flags 3 axis 1 draw 2 pdraw 1 surf 5
rgb 3e800000 3f000000 3f400000
obj 0 bytes 0 3 1 1 0 0 0 77
obj 0 mat 102 255 127 4 / 0 0 0 0 / 0 / 0 255 0 0
obj 0 clips 0 0 0 0
obj 0 extra 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
obj 0 store 0
cont 0 0 psize 3 flags 0 time 0 surf 5 sizes 0 store 0
pt 0 0 0 00000000 00000000 40400000
pt 0 0 1 3f800000 40000000 40400000
pt 0 0 2 40000000 40800000 40400000
cont 0 1 psize 2 flags 8 time 1 surf 2 sizes 0 store 0
pt 0 1 0 bf800000 c0000000 c0400000
pt 0 1 1 41200000 41a00000 41f00000
obj 1 [second] cont 0 mesh 0 flags 9 axis 0 draw 1 pdraw 7 surf 0
rgb 3f800000 00000000 00000000
obj 1 bytes 0 3 1 3 0 0 0 255
obj 1 mat 102 255 127 4 / 0 0 0 0 / 0 / 0 255 0 0
obj 1 clips 0 0 0 0
obj 1 extra 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
obj 1 store 0
checksum 2399750
"#;

const NATIVE_V12_DUMP: &str = r#"test 0
name [v12 chunks]
max 512 384 60 objsize 2 flags 12800 draw 1 mouse 1 bl 0 wl 255
off 3fc00000 40200000 40600000 scale 3f800000 3f800000 3f400000
cindex 1 -1 -1 res 3 thresh 128 units 1 csum 12345
pix 40100000 abg 3f000000 be800000 3e000000
views 1 cview 0 slicer 2 groups 0
slan 0 1 00000000 00000000 00000000 00000000 00000000 00000000 [slicer0]
slan 1 2 41200000 41a00000 41f00000 3f800000 40000000 40400000 [slicer1]
obj 0 [obj zero] cont 3 mesh 0 flags 8 axis 1 draw 1 pdraw 5 surf 4
rgb 3dcccccd 3e4ccccd 3e99999a
obj 0 bytes 2 3 1 4 1 3 0 60
obj 0 mat 100 200 50 7 / 11 22 33 44 / 999 / 1 2 3 4
obj 0 clips 0 0 0 0
obj 0 extra 0 3 6 9 12 15 18 21 24 27 30 33 36 39 42 45
obj 0 store 0
cont 0 0 psize 2 flags 0 time 0 surf 1 sizes 1 store 0
pt 0 0 0 3f800000 40000000 40400000 3fc00000
pt 0 0 1 40800000 40a00000 40c00000 40200000
cont 0 1 psize 1 flags 16 time 2 surf 3 sizes 0 store 0
pt 0 1 0 c0e00000 c1000000 c1100000
cont 0 2 psize 0 flags 0 time 0 surf 0 sizes 0 store 0
obj 1 [obj one] cont 0 mesh 0 flags 512 axis 0 draw 1 pdraw 2 surf 0
rgb 3f666666 3f4ccccd 3f333333
obj 1 bytes 1 3 1 1 0 0 0 0
obj 1 mat 102 255 127 4 / 0 0 0 0 / 0 / 0 255 0 0
obj 1 clips 0 0 0 0
obj 1 extra 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
obj 1 store 0
checksum 16109250
"#;

/// Test-only: decodes the hex fixtures above.
fn hex_bytes(text: &str) -> Vec<u8> {
    (0..text.len() / 2)
        .map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// Test-only: renders a `char[]` field the way `printf("%s")` does.
fn c_string(bytes: &[std::ffi::c_char]) -> String {
    bytes
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8 as char)
        .collect()
}

/// Test-only: reproduces the C driver's `fb()` float rendering.
fn fb(tag: &str, value: f32) -> String {
    format!("{} {:08x}", tag, value.to_bits())
}

/// Test-only: reproduces `scratchpad/mfwork/read_driver.c` line for line.
fn dump_model(path: &std::path::Path) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(out, "test {}", imod_test_if_model_file(path)).unwrap();
    let Ok(m) = imod_read(path) else {
        writeln!(out, "read NULL").unwrap();
        return out;
    };
    writeln!(out, "name [{}]", c_string(&m.name)).unwrap();
    writeln!(
        out,
        "max {} {} {} objsize {} flags {} draw {} mouse {} bl {} wl {}",
        m.xmax,
        m.ymax,
        m.zmax,
        m.obj.len(),
        m.flags,
        m.drawmode,
        m.mousemode,
        m.blacklevel,
        m.whitelevel
    )
    .unwrap();
    writeln!(
        out,
        "{}{}{}{}{}{}",
        fb("off", m.xoffset),
        fb("", m.yoffset),
        fb("", m.zoffset),
        fb(" scale", m.xscale),
        fb("", m.yscale),
        fb("", m.zscale)
    )
    .unwrap();
    writeln!(
        out,
        "cindex {} {} {} res {} thresh {} units {} csum {}",
        m.cindex.object, m.cindex.contour, m.cindex.point, m.res, m.thresh, m.units, m.csum
    )
    .unwrap();
    writeln!(
        out,
        "{}{}{}{}",
        fb("pix", m.pixsize),
        fb(" abg", m.alpha),
        fb("", m.beta),
        fb("", m.gamma)
    )
    .unwrap();
    writeln!(
        out,
        "views {} cview {} slicer {} groups {}",
        m.view.len(),
        m.cview,
        m.slicer_ang.len(),
        m.group_list.len()
    )
    .unwrap();
    for (i, sl) in m.slicer_ang.iter().enumerate() {
        write!(out, "slan {} {}", i, sl.time).unwrap();
        write!(
            out,
            "{}{}{}{}{}{}",
            fb("", sl.angles[0]),
            fb("", sl.angles[1]),
            fb("", sl.angles[2]),
            fb("", sl.center.x),
            fb("", sl.center.y),
            fb("", sl.center.z)
        )
        .unwrap();
        let label: String = sl
            .label
            .iter()
            .take_while(|b| **b != 0)
            .map(|b| *b as char)
            .collect();
        writeln!(out, " [{}]", label).unwrap();
    }
    for (ob, o) in m.obj.iter().enumerate() {
        writeln!(
            out,
            "obj {} [{}] cont {} mesh {} flags {} axis {} draw {} pdraw {} surf {}",
            ob,
            c_string(&o.name),
            o.cont.len(),
            o.mesh.len(),
            o.flags,
            o.axis,
            o.drawmode,
            o.pdrawsize,
            o.surfsize
        )
        .unwrap();
        writeln!(
            out,
            "{}{}{}",
            fb("rgb", o.red),
            fb("", o.green),
            fb("", o.blue)
        )
        .unwrap();
        writeln!(
            out,
            "obj {} bytes {} {} {} {} {} {} {} {}",
            ob,
            o.symbol,
            o.symsize,
            o.linewidth2,
            o.linewidth,
            o.linesty,
            o.symflags,
            o.sympad,
            o.trans
        )
        .unwrap();
        writeln!(
            out,
            "obj {} mat {} {} {} {} / {} {} {} {} / {} / {} {} {} {}",
            ob,
            o.ambient,
            o.diffuse,
            o.specular,
            o.shininess,
            o.fillred,
            o.fillgreen,
            o.fillblue,
            o.quality,
            o.mat2,
            o.valblack,
            o.valwhite,
            o.matflags2,
            o.mesh_thickness
        )
        .unwrap();
        writeln!(
            out,
            "obj {} clips {} {} {} {}",
            ob, o.clips.count, o.clips.flags, o.clips.trans, o.clips.plane
        )
        .unwrap();
        write!(out, "obj {} extra", ob).unwrap();
        for value in o.extra {
            write!(out, " {}", value).unwrap();
        }
        writeln!(out).unwrap();
        writeln!(out, "obj {} store {}", ob, o.store.len()).unwrap();
        for (co, c) in o.cont.iter().enumerate() {
            writeln!(
                out,
                "cont {} {} psize {} flags {} time {} surf {} sizes {} store {}",
                ob,
                co,
                c.pts.len(),
                c.flags,
                c.time,
                c.surf,
                !c.sizes.is_empty() as i32,
                c.store.len()
            )
            .unwrap();
            for (pt, p) in c.pts.iter().enumerate() {
                write!(out, "pt {} {} {}", ob, co, pt).unwrap();
                write!(out, "{}{}{}", fb("", p.x), fb("", p.y), fb("", p.z)).unwrap();
                if !c.sizes.is_empty() {
                    write!(out, "{}", fb("", c.sizes[pt])).unwrap();
                }
                writeln!(out).unwrap();
            }
        }
        for (j, ms) in o.mesh.iter().enumerate() {
            writeln!(
                out,
                "mesh {} {} v {} l {} flag {} time {} surf {}",
                ob,
                j,
                ms.vert.len(),
                ms.list.len(),
                ms.flag,
                ms.time,
                ms.surf
            )
            .unwrap();
        }
    }
    writeln!(out, "checksum {}", imod_checksum(&m)).unwrap();
    out
}

#[test]
fn imodel_read_v01_matches_native_driver() {
    let path = std::env::temp_dir().join(format!("imod-rs-v01-{}.mod", std::process::id()));
    std::fs::write(&path, hex_bytes(V01_MODEL_HEX)).unwrap();
    let mine = dump_model(&path);
    let _ = std::fs::remove_file(&path);
    for (line, (a, b)) in mine.lines().zip(NATIVE_V01_DUMP.lines()).enumerate() {
        assert_eq!(a, b, "line {}", line + 1);
    }
    assert_eq!(mine.lines().count(), NATIVE_V01_DUMP.lines().count());
}

#[test]
fn imodel_read_chunk_model_matches_native_driver() {
    let path = std::env::temp_dir().join(format!("imod-rs-v12-{}.mod", std::process::id()));
    std::fs::write(&path, hex_bytes(V12_MODEL_HEX)).unwrap();
    let mine = dump_model(&path);
    let _ = std::fs::remove_file(&path);
    for (line, (a, b)) in mine.lines().zip(NATIVE_V12_DUMP.lines()).enumerate() {
        assert_eq!(a, b, "line {}", line + 1);
    }
    assert_eq!(mine.lines().count(), NATIVE_V12_DUMP.lines().count());
}

#[test]
fn imod_test_if_model_file_classifies_binary_ascii_junk_and_missing() {
    let dir = std::env::temp_dir().join(format!("imod-rs-testif-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let binary = dir.join("v12.mod");
    std::fs::write(&binary, hex_bytes(V12_MODEL_HEX)).unwrap();
    let ascii = dir.join("ascii.mod");
    std::fs::write(&ascii, b"imod 1\nmax 10 10 10\n").unwrap();
    let junk = dir.join("junk.txt");
    std::fs::write(&junk, b"not a model at all\n").unwrap();
    let empty = dir.join("empty.bin");
    std::fs::write(&empty, b"").unwrap();
    // Values from the reference driver: 0 binary, -1 ascii, 2 not a model,
    // 1 cannot open.
    assert_eq!(imod_test_if_model_file(&binary), 0);
    assert_eq!(imod_test_if_model_file(&ascii), -1);
    assert_eq!(imod_test_if_model_file(&junk), 2);
    assert_eq!(imod_test_if_model_file(&empty), 2);
    assert_eq!(imod_test_if_model_file(dir.join("missing.mod")), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn imod_open_file_and_close_file_round_trip_a_written_model() {
    let path = std::env::temp_dir().join(format!("imod-rs-openfile-{}.mod", std::process::id()));
    let mut model = Imod::default();
    model.obj = imod_objects_new(1).unwrap();
    model.obj[0].cont = vec![Icont {
        pts: vec![Ipoint {
            x: 1.,
            y: 2.,
            z: 3.,
        }],
        ..Icont::default()
    }];
    let mut written = Imod::default();
    let mut out = imod_open_file(path.to_str().unwrap(), "wb", &mut written).unwrap();
    imod_write_file(&model, &mut out).unwrap();
    assert_eq!(imod_close_file(Some(out)), 0);

    let mut read_back = Imod::default();
    let mut input = imod_open_file(path.to_str().unwrap(), "rb", &mut read_back).unwrap();
    imod_read_file(&mut read_back, &mut input).unwrap();
    assert_eq!(imod_close_file(Some(input)), 0);
    assert_eq!(imod_close_file(None), -1);
    assert_eq!(read_back.obj.len(), 1);
    assert_eq!(read_back.obj[0].cont.len(), 1);
    assert_eq!(read_back.obj[0].cont[0].pts.len(), 1);
    // `imodObjectDefault` values survive a round trip through the header.
    assert_eq!(read_back.obj[0].ambient, 102);
    assert_eq!(read_back.obj[0].valwhite, 255);
    assert!(imod_open_file(path.to_str().unwrap(), "zz", &mut read_back).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn imod_write_skip_mesh_drops_meshes_by_object_kind() {
    let mesh = Imesh {
        vert: vec![Ipoint {
            x: 1.,
            y: 2.,
            z: 3.,
        }],
        list: vec![-1],
        ..Imesh::default()
    };
    let mut model = Imod::default();
    model.obj = imod_objects_new(2).unwrap();
    model.obj[0].cont = vec![Icont {
        pts: vec![Ipoint::default()],
        ..Icont::default()
    }];
    model.obj[0].mesh = vec![mesh.clone()];
    model.obj[1].mesh = vec![mesh];
    let dir = std::env::temp_dir().join(format!("imod-rs-skipmesh-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // `imodel_write_object` (`imodel_files.c:359`) always writes the mesh count,
    // so a skipped mesh comes back as an allocated but empty `Imesh`.
    for (skip, contour_verts, isosurface_verts) in [(0, 1, 1), (1, 0, 1), (2, 1, 0), (3, 0, 0)] {
        let path = dir.join(format!("skip{}.mod", skip));
        let mut file = std::fs::File::create(&path).unwrap();
        imod_write_skip_mesh(&model, &mut file, skip).unwrap();
        drop(file);
        let back = imod_read(&path).unwrap();
        assert_eq!(back.obj[0].mesh.len(), 1, "skip {}", skip);
        assert_eq!(back.obj[1].mesh.len(), 1, "skip {}", skip);
        assert_eq!(
            back.obj[0].mesh[0].vert.len(),
            contour_verts,
            "skip {}",
            skip
        );
        assert_eq!(
            back.obj[1].mesh[0].vert.len(),
            isosurface_verts,
            "skip {}",
            skip
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn imodel_write_mesh_omits_paired_thickness_meshes() {
    // imeshThickness (`imesh.h:54`) is bits 24-29 of the flag word;
    // `imodel_write_object` (`imodel_files.c:349`) counts only meshes without
    // thickness and `imodel_write_mesh` (`imodel_files.c:484`) writes only those.
    let mut model = Imod::default();
    model.obj = imod_objects_new(1).unwrap();
    model.obj[0].mesh = vec![
        Imesh {
            vert: vec![Ipoint::default()],
            list: vec![-1],
            ..Imesh::default()
        },
        Imesh {
            vert: vec![Ipoint::default(), Ipoint::default()],
            list: vec![-1],
            flag: 3 << 24,
            ..Imesh::default()
        },
    ];
    let path = std::env::temp_dir().join(format!("imod-rs-thickmesh-{}.mod", std::process::id()));
    imod_file_write(&model, &path).unwrap();
    let back = imod_read(&path).unwrap();
    assert_eq!(back.obj[0].mesh.len(), 1);
    assert_eq!(back.obj[0].mesh[0].vert.len(), 1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn imod_fgetline_skips_comments_and_blank_lines_and_signals_eof() {
    let path = std::env::temp_dir().join(format!("imod-rs-fgetline-{}.txt", std::process::id()));
    std::fs::write(&path, b"# a comment\r\n\nreal line\nlast").unwrap();
    let mut file = std::fs::File::open(&path).unwrap();
    let mut line = [0u8; 81];
    assert_eq!(imod_fgetline(&mut file, &mut line, 81), 10);
    assert_eq!(&line[..10], b"real line\n");
    assert_eq!(imod_fgetline(&mut file, &mut line, 81), -5);
    assert_eq!(&line[..4], b"last");
    assert_eq!(imod_fgetline(&mut file, &mut line, 81), 0);
    assert_eq!(imod_fgetline(&mut file, &mut line, 2), -1);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn low_level_get_and_put_helpers_round_trip_big_endian_arrays() {
    let path = std::env::temp_dir().join(format!("imod-rs-lowlevel-{}.bin", std::process::id()));
    let ints = [1_i32, -2, 0x0102_0304, i32::MIN];
    let floats = [1.5_f32, -0.25, 1e30, 0.];
    let bytes = [1_u8, 2, 250, 255];
    let points = [
        Ipoint {
            x: 1.,
            y: 2.,
            z: 3.,
        },
        Ipoint {
            x: -4.,
            y: 5.,
            z: -6.,
        },
    ];
    let scale = Ipoint {
        x: 2.,
        y: 0.5,
        z: -1.,
    };
    {
        let mut file = std::fs::File::create(&path).unwrap();
        imod_put_ints(&mut file, &ints, 4).unwrap();
        imod_put_floats(&mut file, &floats, 4).unwrap();
        imod_put_bytes(&mut file, &bytes, 4).unwrap();
        imod_put_scaled_points(&mut file, &points, 2, &scale).unwrap();
    }
    let raw = std::fs::read(&path).unwrap();
    // Every word is written most significant byte first.
    assert_eq!(&raw[..4], &[0, 0, 0, 1]);
    assert_eq!(&raw[8..12], &[1, 2, 3, 4]);
    let mut file = std::fs::File::open(&path).unwrap();
    let mut back_ints = [0_i32; 4];
    imod_get_ints(&mut file, &mut back_ints, 4).unwrap();
    assert_eq!(back_ints, ints);
    let mut back_floats = [0_f32; 4];
    imod_get_floats(&mut file, &mut back_floats, 4).unwrap();
    assert_eq!(back_floats, floats);
    let mut back_bytes = [0_u8; 4];
    imod_get_bytes(&mut file, &mut back_bytes, 4).unwrap();
    assert_eq!(back_bytes, bytes);
    let mut back_points = [0_f32; 6];
    imod_get_floats(&mut file, &mut back_points, 6).unwrap();
    assert_eq!(back_points, [2., 1., -3., -8., 2.5, 6.]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn byteswap_and_vms_float_conversions_match_the_source_byte_order() {
    let mut word = [0x11_u8, 0x22, 0x33, 0x44];
    byteswap(&mut word, 4);
    assert_eq!(word, [0x44, 0x33, 0x22, 0x11]);
    // An odd size leaves the last byte alone.
    let mut odd = [1_u8, 2, 3];
    byteswap(&mut odd, 3);
    assert_eq!(odd, [2, 1, 3]);
    // A zero size touches nothing.
    let mut none = [7_u8, 8];
    byteswap(&mut none, 0);
    assert_eq!(none, [7, 8]);
    let mut longs = [1_u8, 2, 3, 4, 5, 6, 7, 8];
    swap_longs(&mut longs, 2);
    assert_eq!(longs, [4, 3, 2, 1, 8, 7, 6, 5]);
    // `tovmsfloat` and `imodFromVmsFloats` are inverses on a normal value.
    let mut vms = 1.5_f32.to_be_bytes();
    let original = vms;
    tovmsfloat(&mut vms, 1);
    assert_ne!(vms, original);
    imod_from_vms_floats(&mut vms, 1);
    assert_eq!(vms, original);
}

/// Native output of `scratchpad/ilabel_diff.c`, compiled against the pinned
/// `IMOD/include/imodel.h` and linked to the reference `libimod`.
const NATIVE_ILABEL: &str = r#"
new len=0 nl=0 name=(null)
name() ret=0
afterName len=1 nl=0 name=
name(a) ret=0
afterName len=2 nl=0 name=a
name(ab) ret=0
afterName len=3 nl=0 name=ab
name(abc) ret=0
afterName len=4 nl=0 name=abc
name(abcd) ret=0
afterName len=5 nl=0 name=abcd
name(abcde) ret=0
afterName len=6 nl=0 name=abcde
name(surface one) ret=0
afterName len=12 nl=0 name=surface one
name(0123456789012345) ret=0
afterName len=17 nl=0 name=0123456789012345
name(xy) ret=0
afterShrink len=17 nl=0 name=xy
afterAdd len=17 nl=3 name=xy
afterAdd  item 0: index=0 len=5 name=zero
afterAdd  item 1: index=1 len=19 name=longer replacement
afterAdd  item 2: index=2 len=4 name=x
itemGet(0)=zero
itemGet(1)=longer replacement
itemGet(9)=(null)
nameGet=xy
afterMove len=17 nl=3 name=xy
afterMove  item 0: index=0 len=5 name=zero
afterMove  item 1: index=7 len=19 name=longer replacement
afterMove  item 2: index=2 len=4 name=x
afterMoveMiss len=17 nl=3 name=xy
afterMoveMiss  item 0: index=0 len=5 name=zero
afterMoveMiss  item 1: index=7 len=19 name=longer replacement
afterMoveMiss  item 2: index=2 len=4 name=x
dup len=2 nl=3 name=xy
dup  item 0: index=0 len=5 name=zero
dup  item 1: index=7 len=19 name=longer replacement
dup  item 2: index=2 len=2 name=x
print:
contour label : "xy"
	  0 : "zero"
	  7 : "longer replacement"
	  2 : "x"
match(xy)=1 itemMatch(xy,0)=0
match(x)=0 itemMatch(x,0)=0
match(x?)=1 itemMatch(x?,0)=0
match(*y)=1 itemMatch(*y,0)=0
match(x*)=1 itemMatch(x*,0)=0
match(*)=1 itemMatch(*,0)=1
match(?y)=1 itemMatch(?y,0)=0
match(xz)=0 itemMatch(xz,0)=0
match(\xy)=1 itemMatch(\xy,0)=0
match(x*y)=1 itemMatch(x*y,0)=0
match(*z)=0 itemMatch(*z,0)=0
write=0
write bytes 76: 4c 41 42 4c 00 00 00 44 00 00 00 03 00 00 00 04 78 79 00 00 00 00 00 00 00 00 00 08 7a 65 72 6f 00 00 00 00 00 00 00 07 00 00 00 14 6c 6f 6e 67 65 72 20 72 65 70 6c 61 63 65 6d 65 6e 74 00 00 00 00 00 02 00 00 00 04 78 00 00 00
readErr=0
read len=4 nl=3 name=xy
read  item 0: index=0 len=8 name=zero
read  item 1: index=7 len=20 name=longer replacement
read  item 2: index=2 len=4 name=x
write2=0
write2 bytes 76: 4c 41 42 4c 00 00 00 44 00 00 00 03 00 00 00 04 78 79 00 00 00 00 00 00 00 00 00 08 7a 65 72 6f 00 00 00 00 00 00 00 07 00 00 00 14 6c 6f 6e 67 65 72 20 72 65 70 6c 61 63 65 6d 65 6e 74 00 00 00 00 00 02 00 00 00 04 78 00 00 00
write3=0
write3 bytes 20: 4c 41 42 4c 00 00 00 0c 00 00 00 00 00 00 00 04 00 00 00 00
read3 len=4 nl=0 name=
readErr3=0
dupEmpty:
dupEmpty len=0 nl=0 name=(null)
dupNull=(nil)
afterDel0 len=17 nl=2 name=xy
afterDel0  item 0: index=7 len=19 name=longer replacement
afterDel0  item 1: index=2 len=4 name=x
afterDelMiss len=17 nl=2 name=xy
afterDelMiss  item 0: index=7 len=19 name=longer replacement
afterDelMiss  item 1: index=2 len=4 name=x
beforeDelMid len=17 nl=4 name=xy
beforeDelMid  item 0: index=7 len=19 name=longer replacement
beforeDelMid  item 1: index=2 len=4 name=x
beforeDelMid  item 2: index=3 len=6 name=third
beforeDelMid  item 3: index=4 len=7 name=fourth
afterDelMid len=17 nl=3 name=xy
afterDelMid  item 0: index=7 len=19 name=longer replacement
afterDelMid  item 1: index=3 len=6 name=third
afterDelMid  item 2: index=4 len=7 name=fourth
write4=0
write4 bytes 80: 4c 41 42 4c 00 00 00 48 00 00 00 03 00 00 00 04 78 79 00 00 00 00 00 07 00 00 00 14 6c 6f 6e 67 65 72 20 72 65 70 6c 61 63 65 6d 65 6e 74 00 00 00 00 00 03 00 00 00 08 74 68 69 72 64 00 00 00 00 00 00 04 00 00 00 08 66 6f 75 72 74 68 00 00
writeNull=-1
nameNullLabel=1
nameNullVal=1
"#;

/// Original: `ID_LABL` (`imodel.h:86`).
const ID_LABL: u32 = u32::from_be_bytes(*b"LABL");

/// Test-only: renders a `b3dByte *` label buffer the way `printf("%s")` does.
fn label_string(bytes: Option<&[u8]>) -> String {
    match bytes {
        None => "(null)".to_string(),
        Some(bytes) => bytes
            .iter()
            .take_while(|b| **b != 0)
            .map(|b| *b as char)
            .collect(),
    }
}

/// Test-only: reproduces the C driver's `showlab()`.
fn show_label(tag: &str, lab: Option<&Ilabel>) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let Some(lab) = lab else {
        writeln!(out, "{} NULL", tag).unwrap();
        return out;
    };
    writeln!(
        out,
        "{} len={} nl={} name={}",
        tag,
        lab.len,
        lab.label.len(),
        label_string(lab.name.as_deref())
    )
    .unwrap();
    for i in 0..lab.label.len() {
        writeln!(
            out,
            "{}  item {}: index={} len={} name={}",
            tag,
            i,
            lab.label[i].index,
            lab.label[i].len,
            label_string(lab.label[i].name.as_deref())
        )
        .unwrap();
    }
    out
}

/// Test-only: reproduces the C driver's `dumpfile()`.
fn dump_label_file(path: &std::path::Path, tag: &str) -> String {
    use std::fmt::Write as _;
    let bytes = std::fs::read(path).unwrap();
    let mut out = String::new();
    write!(out, "{} bytes {}:", tag, bytes.len()).unwrap();
    for byte in &bytes {
        write!(out, " {:02x}", byte).unwrap();
    }
    out.push('\n');
    out
}

/// Reproduces the `ilabel.c` differential driver line for line.
#[test]
fn ilabel_matches_native_libimod_driver() {
    use std::fmt::Write as _;
    use std::io::{Seek, SeekFrom};

    let dir = std::env::temp_dir().join(format!("imod-rs-ilabel-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut out = String::new();
    let tests: [&[u8]; 8] = [
        b"",
        b"a",
        b"ab",
        b"abc",
        b"abcd",
        b"abcde",
        b"surface one",
        b"0123456789012345",
    ];

    /* new */
    let mut lab = imod_label_new();
    out.push_str(&show_label("new", Some(&lab)));

    /* name setting sequence exercising the realloc test */
    for test in tests {
        writeln!(
            out,
            "name({}) ret={}",
            label_string(Some(test)),
            imod_label_name(Some(&mut lab), Some(test))
        )
        .unwrap();
        out.push_str(&show_label("afterName", Some(&lab)));
    }
    /* shrink back */
    writeln!(
        out,
        "name(xy) ret={}",
        imod_label_name(Some(&mut lab), Some(b"xy"))
    )
    .unwrap();
    out.push_str(&show_label("afterShrink", Some(&lab)));

    /* item add */
    imod_label_item_add(&mut lab, Some(b"zero"), 0);
    imod_label_item_add(&mut lab, Some(b"one"), 1);
    imod_label_item_add(&mut lab, Some(b"two"), 2);
    imod_label_item_add(&mut lab, Some(b""), 3);
    imod_label_item_add(&mut lab, Some(b"longer replacement"), 1);
    imod_label_item_add(&mut lab, Some(b"x"), 2);
    imod_label_item_add(&mut lab, None, 4);
    out.push_str(&show_label("afterAdd", Some(&lab)));

    writeln!(
        out,
        "itemGet(0)={}",
        label_string(imod_label_item_get(Some(&lab), 0))
    )
    .unwrap();
    writeln!(
        out,
        "itemGet(1)={}",
        label_string(imod_label_item_get(Some(&lab), 1))
    )
    .unwrap();
    writeln!(
        out,
        "itemGet(9)={}",
        label_string(imod_label_item_get(Some(&lab), 9))
    )
    .unwrap();
    writeln!(
        out,
        "nameGet={}",
        label_string(imod_label_name_get(Some(&lab)))
    )
    .unwrap();

    imod_label_item_move(Some(&mut lab), 7, 1);
    out.push_str(&show_label("afterMove", Some(&lab)));
    imod_label_item_move(Some(&mut lab), 8, 99);
    out.push_str(&show_label("afterMoveMiss", Some(&lab)));

    let dup = imod_label_dup(Some(&lab));
    out.push_str(&show_label("dup", dup.as_ref()));

    /* imodLabelPrint writes through libc; capture it from a tmpfile */
    out.push_str("print:\n");
    unsafe {
        let tmp = libc::tmpfile();
        imod_label_print(Some(&lab), tmp);
        libc::fflush(tmp);
        libc::rewind(tmp);
        let mut buf = [0_u8; 4096];
        let n = libc::fread(buf.as_mut_ptr() as *mut libc::c_void, 1, buf.len(), tmp);
        out.push_str(std::str::from_utf8(&buf[..n]).unwrap());
        libc::fclose(tmp);
    }

    /* matching */
    let exps: [&[u8]; 11] = [
        b"xy", b"x", b"x?", b"*y", b"x*", b"*", b"?y", b"xz", b"\\xy", b"x*y", b"*z",
    ];
    for exp in exps {
        let expz = [exp, b"\0"].concat();
        writeln!(
            out,
            "match({})={} itemMatch({},0)={}",
            label_string(Some(exp)),
            imod_label_match(Some(&lab), Some(&expz)),
            label_string(Some(exp)),
            imod_label_item_match(Some(&lab), Some(&expz), 0)
        )
        .unwrap();
    }

    /* write */
    let p1 = dir.join("ilab.bin");
    let mut f = std::fs::File::create(&p1).unwrap();
    writeln!(
        out,
        "write={}",
        imod_label_write(Some(&lab), ID_LABL, &mut f)
    )
    .unwrap();
    drop(f);
    out.push_str(&dump_label_file(&p1, "write"));

    /* read back, skipping the 4 byte tag */
    let mut f = std::fs::File::open(&p1).unwrap();
    f.seek(SeekFrom::Start(4)).unwrap();
    let mut err = 0;
    let rd = imod_label_read(&mut f, &mut err);
    drop(f);
    writeln!(out, "readErr={}", err).unwrap();
    out.push_str(&show_label("read", rd.as_ref()));

    /* rewrite what was read */
    let p2 = dir.join("ilab2.bin");
    let mut f = std::fs::File::create(&p2).unwrap();
    writeln!(
        out,
        "write2={}",
        imod_label_write(rd.as_ref(), ID_LABL, &mut f)
    )
    .unwrap();
    drop(f);
    out.push_str(&dump_label_file(&p2, "write2"));

    /* empty label round trip */
    let e = imod_label_new();
    let p3 = dir.join("ilab3.bin");
    let mut f = std::fs::File::create(&p3).unwrap();
    writeln!(
        out,
        "write3={}",
        imod_label_write(Some(&e), ID_LABL, &mut f)
    )
    .unwrap();
    drop(f);
    out.push_str(&dump_label_file(&p3, "write3"));
    let mut f = std::fs::File::open(&p3).unwrap();
    f.seek(SeekFrom::Start(4)).unwrap();
    let mut err3 = 0;
    let rd3 = imod_label_read(&mut f, &mut err3);
    drop(f);
    out.push_str(&show_label("read3", rd3.as_ref()));
    writeln!(out, "readErr3={}", err3).unwrap();
    out.push_str("dupEmpty:\n");
    out.push_str(&show_label("dupEmpty", imod_label_dup(Some(&e)).as_ref()));
    writeln!(
        out,
        "dupNull={}",
        match imod_label_dup(None) {
            None => "(nil)",
            Some(_) => "(some)",
        }
    )
    .unwrap();

    /* item delete */
    imod_label_item_delete(Some(&mut lab), 0);
    out.push_str(&show_label("afterDel0", Some(&lab)));
    imod_label_item_delete(Some(&mut lab), 42);
    out.push_str(&show_label("afterDelMiss", Some(&lab)));
    /* delete an item that is not first in the array */
    imod_label_item_add(&mut lab, Some(b"third"), 3);
    imod_label_item_add(&mut lab, Some(b"fourth"), 4);
    out.push_str(&show_label("beforeDelMid", Some(&lab)));
    imod_label_item_delete(Some(&mut lab), 2);
    out.push_str(&show_label("afterDelMid", Some(&lab)));
    imod_label_item_delete(None, 0);

    /* write after delete */
    let p4 = dir.join("ilab4.bin");
    let mut f = std::fs::File::create(&p4).unwrap();
    writeln!(
        out,
        "write4={}",
        imod_label_write(Some(&lab), ID_LABL, &mut f)
    )
    .unwrap();
    drop(f);
    out.push_str(&dump_label_file(&p4, "write4"));

    let mut sink = std::fs::File::create(dir.join("sink.bin")).unwrap();
    writeln!(
        out,
        "writeNull={}",
        imod_label_write(None, ID_LABL, &mut sink)
    )
    .unwrap();
    imod_label_print(None, std::ptr::null_mut());
    writeln!(out, "nameNullLabel={}", imod_label_name(None, Some(b"a\0"))).unwrap();
    writeln!(out, "nameNullVal={}", imod_label_name(Some(&mut lab), None)).unwrap();

    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(out, NATIVE_ILABEL.trim_start_matches('\n'));
}

/// Native output of `scratchpad/imesh_diff.c`, compiled against the pinned
/// `IMOD/include/imesh.h` and linked to the reference `libimod`.
const NATIVE_IMESH: &str = r#"
factors(-26)=0 -99 -99 -99
factors(-25)=1 1 0 1
factors(-24)=0 -99 -99 -99
factors(-23)=1 2 1 0
factors(-22)=0 -99 -99 -99
factors(-21)=0 -99 -99 -99
boxNew vsize=16 lsize=39
boxNew vol(NULL,NULL)   = 175
boxNew area(NULL)       = 242
boxNew vol(z1)          = 175
boxNew area(z1)         = 242
boxNew vol(z2.7)        = 472.500031
boxNew area(z2.7)       = 415.400024
boxNew vol(z0.13)       = 22.75
boxNew area(z0.13)      = 153.259995
boxNew vol(z1.7,ctr)    = 242.403015
boxNew vol(NULL,ctr)    = 142.589996
boxNew bbox=0 ll=0 0 0 ur=10 7 3
boxOld vsize=16 lsize=75
boxOld vol(NULL,NULL)   = 175
boxOld area(NULL)       = 242
boxOld vol(z1)          = 175
boxOld area(z1)         = 242
boxOld vol(z2.7)        = 472.500031
boxOld area(z2.7)       = 415.400024
boxOld vol(z0.13)       = 22.75
boxOld area(z0.13)      = 153.259995
boxOld vol(z1.7,ctr)    = 242.403015
boxOld vol(NULL,ctr)    = 142.589996
boxOld bbox=0 ll=0 0 0 ur=10 7 3
boxJit vsize=16 lsize=39
boxJit vol(NULL,NULL)   = 469.107819
boxJit area(NULL)       = 509.670776
boxJit vol(z1)          = 469.107819
boxJit area(z1)         = 509.670776
boxJit vol(z2.7)        = 1266.59119
boxJit area(z2.7)       = 1180.11487
boxJit vol(z0.13)       = 60.9840164
boxJit area(z0.13)      = 179.993729
boxJit vol(z1.7,ctr)    = 545.764038
boxJit vol(NULL,ctr)    = 321.037659
boxJit bbox=0 ll=-0.55367738 -0.361631989 -1.21702778 ur=14.4425869 5.31063414 10.4446344
boxBig vsize=16 lsize=39
boxBig vol(NULL,NULL)   = 461666304
boxBig area(NULL)       = 4148520.75
boxBig vol(z1)          = 461666304
boxBig area(z1)         = 4148520.75
boxBig vol(z2.7)        = 1.24649907e+09
boxBig area(z2.7)       = 8240246
boxBig vol(z0.13)       = 60016616
boxBig area(z0.13)      = 2075550.12
boxBig vol(z1.7,ctr)    = 633350592
boxBig vol(NULL,ctr)    = 372559168
boxBig bbox=0 ll=-15.2894344 -37.2326202 -33.3706169 ur=1060.24573 838.219116 654.194946
boxBigOld vsize=16 lsize=75
boxBigOld vol(NULL,NULL)   = 6.54437427e+09
boxBigOld area(NULL)       = 23754120
boxBigOld vol(z1)          = 6.54437427e+09
boxBigOld area(z1)         = 23754120
boxBigOld vol(z2.7)        = 1.76698122e+10
boxBigOld area(z2.7)       = 48438312
boxBigOld vol(z0.13)       = 850768640
boxBigOld area(z0.13)      = 11199996
boxBigOld vol(z1.7,ctr)    = 8.95276339e+09
boxBigOld vol(NULL,ctr)    = 5.26633114e+09
boxBigOld bbox=0 ll=-40.3075142 -90.4260483 -54.3867378 ur=2420.11304 1958.00525 1704.51978
boxOpen185 vsize=16 lsize=15
boxOpen185 vol(NULL,NULL)   = 162482192
boxOpen185 area(NULL)       = 1549140
boxOpen185 vol(z1)          = 162482192
boxOpen185 area(z1)         = 1549140
boxOpen185 vol(z2.7)        = 438701920
boxOpen185 area(z2.7)       = 1736406
boxOpen185 vol(z0.13)       = 21122684
boxOpen185 area(z0.13)      = 1517289.75
boxOpen185 vol(z1.7,ctr)    = 186771664
boxOpen185 vol(NULL,ctr)    = 109865680
boxOpen185 bbox=0 ll=9.56074238 -142.175201 -124.423553 ur=1086.76318 950.508057 738.559998
boxOpen6 vsize=16 lsize=15
boxOpen6 vol(NULL,NULL)   = 214277600
boxOpen6 area(NULL)       = 1778070
boxOpen6 vol(z1)          = 214277600
boxOpen6 area(z1)         = 1778070
boxOpen6 vol(z2.7)        = 578549504
boxOpen6 area(z2.7)       = 1924986.75
boxOpen6 vol(z0.13)       = 27856086
boxOpen6 area(z0.13)      = 1753834.75
boxOpen6 vol(z1.7,ctr)    = 402572512
boxOpen6 vol(NULL,ctr)    = 236807360
boxOpen6 bbox=0 ll=-35.441803 -32.600811 -129.790924 ur=1126.19263 972.550903 763.034363
boxOpen11 vsize=16 lsize=15
boxOpen11 vol(NULL,NULL)   = 180594864
boxOpen11 area(NULL)       = 1675274.5
boxOpen11 vol(z1)          = 180594864
boxOpen11 area(z1)         = 1675274.5
boxOpen11 vol(z2.7)        = 487606176
boxOpen11 area(z2.7)       = 1844801
boxOpen11 vol(z0.13)       = 23477332
boxOpen11 area(z0.13)      = 1645255.5
boxOpen11 vol(z1.7,ctr)    = 417342656
boxOpen11 vol(NULL,ctr)    = 245495680
boxOpen11 bbox=0 ll=-144.24556 -91.0882797 -96.1834564 ur=1135.44666 833.94873 687.191833
boxRough1460 vsize=16 lsize=39
boxRough1460 vol(NULL,NULL)   = 427292224
boxRough1460 area(NULL)       = 3831899
boxRough1460 vol(z1)          = 427292224
boxRough1460 area(z1)         = 3831899
boxRough1460 vol(z2.7)        = 1.15368896e+09
boxRough1460 area(z2.7)       = 7565322.5
boxRough1460 vol(z0.13)       = 55547984
boxRough1460 area(z0.13)      = 2116571.5
boxRough1460 vol(z1.7,ctr)    = 622868096
boxRough1460 vol(NULL,ctr)    = 366392992
boxRough1460 bbox=0 ll=-111.664185 -136.122513 -13.6892157 ur=1112.42041 808.949219 768.288208
boxRough2992 vsize=16 lsize=39
boxRough2992 vol(NULL,NULL)   = 424571808
boxRough2992 area(NULL)       = 4125799.75
boxRough2992 vol(z1)          = 424571808
boxRough2992 area(z1)         = 4125799.75
boxRough2992 vol(z2.7)        = 1.14634394e+09
boxRough2992 area(z2.7)       = 8465985
boxRough2992 vol(z0.13)       = 55194336
boxRough2992 area(z0.13)      = 2086805.5
boxRough2992 vol(z1.7,ctr)    = 531662752
boxRough2992 vol(NULL,ctr)    = 312742784
boxRough2992 bbox=0 ll=-119.166122 -130.347855 -85.2632446 ur=1114.59802 858.398804 762.580505
volNull=0 areaNull=0
volEmpty=0 areaEmpty=0 bboxEmpty=-1
getIndex(0)=-25 getIndex(-1)=-1 getIndex(9999)=-1
maxIndex=39 maxVert=16
getVert(3)=0.374182105 -0.174415112 0.172719836
getVert(-1)=(nil) getVert(9999)=(nil) verts=1
nullAccessors -1 0 0 (nil) (nil)
dup vsize=16 lsize=39 vol=175
copyNull=-1 -1
afterInsert lsize=40 list[0..5]=-25 0 4242 4 2 2
afterDelIdx lsize=39 list[0..5]=-25 0 4 2 2 4
afterDelMiss lsize=39
afterAddNorm vsize=17 lsize=41 tail=-20 16
addNormNull=-1 -1
meshAddSize=5
nearestRes(-2)=1 out=0
nearestRes(-1)=1 out=0
nearestRes(0)=1 out=0
nearestRes(1)=1 out=2
nearestRes(2)=1 out=2
nearestRes(3)=1 out=2
nearestRes(4)=1 out=5
nearestRes(5)=1 out=5
nearestRes(6)=1 out=5
nearestRes(7)=1 out=9
nearestRes(8)=1 out=9
nearestRes(9)=1 out=9
nearestRes(10)=1 out=9
nearestRes(11)=1 out=9
nearestRes1=0 out=0
nearestResNull=-1 out=7
nearestRes0=-1 out=7
sortInput vsize=48 lsize=111 vol=525 area=642
sort nsurf=3 err=0
surf 0: surfnum=0 vsize=16 lsize=39 vol=175 area=214
surf 0 list: -25 0 2 4 4 2 6 8 10 12 10 14 12 0 4 8 4 10 8 2 12 6 6 12 14 0 8 2 2 8 12 4 6 10 6 10 14 -22 -1
surf 0 vert: 0/0/0 0.812491655/0.986288786/0.58218658 0/6/0 -0.810331941/-0.565639615/-0.149167776 5/0/0 0.984699607/0.0325194597/-0.286839485 5/6/0 -0.461397052/0.991991043/0.577188492 0/0/7 -0.513336062/-0.664072394/-0.47296524 5/0/7 -0.768256903/-0.285033941/-0.881943822 0/6/7 -0.711518288/0.871757507/0.244215012 5/6/7 0.281051159/-0.916882992/0.134318233
surf 1: surfnum=1 vsize=16 lsize=39 vol=175 area=214
surf 1 list: -25 0 2 4 4 2 6 8 10 12 10 14 12 0 4 8 4 10 8 2 12 6 6 12 14 0 8 2 2 8 12 4 6 10 6 10 14 -22 -1
surf 1 vert: 20/0/0 0.420392036/-0.599619865/-0.363824964 20/6/0 -0.0865166187/0.294776797/0.0420577526 25/0/0 -0.634847283/0.975884199/0.923664927 25/6/0 -0.396335125/0.129152894/-0.560747981 20/0/7 0.144253135/-0.984755039/0.907565117 25/0/7 -0.680698991/0.397952318/7.76052475e-05 20/6/7 -0.431593537/-0.908466101/-0.37513268 25/6/7 0.382829309/0.00349915028/-0.314467311
surf 2: surfnum=2 vsize=16 lsize=39 vol=175 area=214
surf 2 list: -25 0 2 4 4 2 6 8 10 12 10 14 12 0 4 8 4 10 8 2 12 6 6 12 14 0 8 2 2 8 12 4 6 10 6 10 14 -22 -1
surf 2 vert: 40/0/0 -0.392298937/0.802581429/0.114179492 40/6/0 0.277972341/0.535542727/-0.473841667 45/0/0 0.0405460596/-0.797670126/-0.996764064 45/6/0 0.42665422/-0.186687231/-0.793625355 40/0/7 -0.0586047173/0.452281833/-0.0889350176 45/0/7 -0.156200528/0.0520021915/-0.872993112 40/6/7 -0.998412609/0.56611228/0.433063149 45/6/7 0.729002118/0.0260950327/0.960137129
sortNull=(nil) nsurf=0 err=1
sortBigPoly=(nil) nsurf=0 err=3
interp psize=8
interp 0: 6.66666651 0 1
interp 1: 0 0 1
interp 2: 10 0 1
interp 3: 0 7 1
interp 4: 6.66666698 7 1
interp 5: 10 7 1
interp 6: 0 4.66666698 1
interp 7: 10 4.66666651 1
interpJit z=0 psize=8
interpJit 0 0: 12.6879444 1.21608913 0
interpJit 0 1: 0.406139135 -0.194064081 0
interpJit 0 2: 14.3547449 1.09626353 0
interpJit 0 3: -0.541510463 5.14979553 0
interpJit 0 4: 11.815093 3.91855097 0
interpJit 0 5: 12.0175705 3.91423988 0
interpJit 0 6: -0.499322027 4.74238873 0
interpJit 0 7: 12.0399618 3.8432579 0
interpJit z=4 psize=8
interpJit 4 0: 7.24332094 1.20949101 4
interpJit 4 1: 0.229509607 0.404190153 4
interpJit 4 2: 14.3883858 0.695835471 4
interpJit 4 3: -0.494112909 5.2177701 4
interpJit 4 4: 6.64027786 4.50688028 4
interpJit 4 5: 12.3308258 4.38572454 4
interpJit 4 6: -0.287575096 3.2232666 4
interpJit 4 7: 12.9600992 2.39080834 4
interpJit z=8 psize=8
interpJit 8 0: 1.79869676 1.20289278 8
interpJit 8 1: 0.0528800525 1.00244439 8
interpJit 8 2: 14.4220276 0.295407325 8
interpJit 8 3: -0.446715385 5.28574467 8
interpJit 8 4: 1.46546304 5.09521008 8
interpJit 8 5: 12.6440811 4.85720873 8
interpJit 8 6: -0.0758281574 1.70414436 8
interpJit 8 7: 13.8802366 0.938359022 8
interpBig z=200 psize=8
interpBig 200 0: 662.967651 6.43155241 200
interpBig 200 1: -6.27571487 -34.5847282 200
interpBig 200 2: 1018.64563 16.0089417 200
interpBig 200 3: 1.41169429 827.556091 200
interpBig 200 4: 712.390991 834.970581 200
interpBig 200 5: 1019.54962 837.911499 200
interpBig 200 6: -3.17472148 520.861816 200
interpBig 200 7: 1024.23242 567.722839 200
interpBig z=400 psize=8
interpBig 400 0: 343.85907 -14.1117077 400
interpBig 400 1: -10.5165186 -35.8305168 400
interpBig 400 2: 1040.16675 4.63786507 400
interpBig 400 3: -0.357489675 827.820801 400
interpBig 400 4: 388.319183 831.874084 400
interpBig 400 5: 1033.0343 838.046936 400
interpBig 400 6: -8.87449074 258.287781 400
interpBig 400 7: 1042.8634 270.930542 400
params flags=272 cap=0 passes=1 capSkipNz=0 incz=4/1 minz=2147483647 maxz=2147483647 spareInt=0
params f 0 10 -1.00000002e+30 1.00000002e+30 -1.00000002e+30 1.00000002e+30 2 0.25 1.5 0
params zlist=(nil)
copySkip=0 nto=4
copySkip 0=3
copySkip 1=9
copySkip 2=27
copySkip 3=81
copySkipAgain=0 nto=2
copySkipAgain 0=3
copySkipAgain 1=9
copySkipNull=0 nto=0
dupParams nz=2 z0=11 z1=22 flags=272
dupParamsNull=(nil)
"#;

/// Test-only: `printf("%.9g")` through libc, so the reference digits match.
fn g9(value: f64) -> String {
    unsafe {
        let mut buf = [0_u8; 64];
        libc::snprintf(
            buf.as_mut_ptr() as *mut std::ffi::c_char,
            buf.len(),
            c"%.9g".as_ptr(),
            value,
        );
        let end = buf.iter().position(|b| *b == 0).unwrap();
        String::from_utf8(buf[..end].to_vec()).unwrap()
    }
}

/// Test-only: the C driver's deterministic `nextf()`.
struct MeshSeed(u32);
impl MeshSeed {
    fn nextf(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        lo + (hi - lo) * ((self.0 >> 8) & 0xffffff) as f32 / 0x1000000 as f32
    }
}

const MESH_TRI: [[usize; 3]; 12] = [
    [0, 2, 1],
    [1, 2, 3],
    [4, 5, 6],
    [5, 7, 6],
    [0, 1, 4],
    [1, 5, 4],
    [2, 6, 3],
    [3, 6, 7],
    [0, 4, 2],
    [2, 4, 6],
    [1, 3, 5],
    [3, 5, 7],
];

/// Test-only: the C driver's `buildBoxN()`.
fn build_box_n(
    seed: &mut MeshSeed,
    old_style: bool,
    sx: f32,
    sy: f32,
    sz: f32,
    jitter: f32,
    ntri: usize,
) -> Imesh {
    let mut m = imod_mesh_new().unwrap().remove(0);
    for c in 0..8 {
        let p = Ipoint {
            x: (if c & 1 != 0 { sx } else { 0. }) + jitter * seed.nextf(-1., 1.),
            y: (if c & 2 != 0 { sy } else { 0. }) + jitter * seed.nextf(-1., 1.),
            z: (if c & 4 != 0 { sz } else { 0. }) + jitter * seed.nextf(-1., 1.),
        };
        imod_mesh_add_vert(&mut m, &p);
        let n = Ipoint {
            x: seed.nextf(-1., 1.),
            y: seed.nextf(-1., 1.),
            z: seed.nextf(-1., 1.),
        };
        imod_mesh_add_vert(&mut m, &n);
    }
    imod_mesh_add_index(
        &mut m,
        if old_style {
            IMOD_MESH_BGNPOLYNORM
        } else {
            IMOD_MESH_BGNPOLYNORM2
        },
    );
    for t in 0..ntri {
        for i in 0..3 {
            if old_style {
                imod_mesh_add_index(&mut m, 2 * MESH_TRI[t][i] as i32 + 1);
            }
            imod_mesh_add_index(&mut m, 2 * MESH_TRI[t][i] as i32);
        }
    }
    imod_mesh_add_index(&mut m, IMOD_MESH_ENDPOLY);
    imod_mesh_add_index(&mut m, IMOD_MESH_END);
    m
}

/// Test-only: the C driver's `buildBox()`.
fn build_box(
    seed: &mut MeshSeed,
    old_style: bool,
    sx: f32,
    sy: f32,
    sz: f32,
    jitter: f32,
) -> Imesh {
    build_box_n(seed, old_style, sx, sy, sz, jitter, 12)
}

/// Test-only: the C driver's `report()`.
fn report_mesh(tag: &str, m: &Imesh) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let mut scale = Ipoint {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let mut ll = Ipoint::default();
    let mut ur = Ipoint::default();
    writeln!(out, "{} vsize={} lsize={}", tag, m.vert.len(), m.list.len()).unwrap();
    writeln!(
        out,
        "{} vol(NULL,NULL)   = {}",
        tag,
        g9(imesh_volume(Some(m), None, None) as f64)
    )
    .unwrap();
    writeln!(
        out,
        "{} area(NULL)       = {}",
        tag,
        g9(imesh_surface_area(Some(m), None) as f64)
    )
    .unwrap();
    scale.z = 1.;
    writeln!(
        out,
        "{} vol(z1)          = {}",
        tag,
        g9(imesh_volume(Some(m), Some(&scale), None) as f64)
    )
    .unwrap();
    writeln!(
        out,
        "{} area(z1)         = {}",
        tag,
        g9(imesh_surface_area(Some(m), Some(&scale)) as f64)
    )
    .unwrap();
    scale.z = 2.7;
    writeln!(
        out,
        "{} vol(z2.7)        = {}",
        tag,
        g9(imesh_volume(Some(m), Some(&scale), None) as f64)
    )
    .unwrap();
    writeln!(
        out,
        "{} area(z2.7)       = {}",
        tag,
        g9(imesh_surface_area(Some(m), Some(&scale)) as f64)
    )
    .unwrap();
    scale.z = 0.13;
    writeln!(
        out,
        "{} vol(z0.13)       = {}",
        tag,
        g9(imesh_volume(Some(m), Some(&scale), None) as f64)
    )
    .unwrap();
    writeln!(
        out,
        "{} area(z0.13)      = {}",
        tag,
        g9(imesh_surface_area(Some(m), Some(&scale)) as f64)
    )
    .unwrap();
    let center = Ipoint {
        x: 0.37,
        y: -2.5,
        z: 11.125,
    };
    scale.z = 1.7;
    writeln!(
        out,
        "{} vol(z1.7,ctr)    = {}",
        tag,
        g9(imesh_volume(Some(m), Some(&scale), Some(&center)) as f64)
    )
    .unwrap();
    writeln!(
        out,
        "{} vol(NULL,ctr)    = {}",
        tag,
        g9(imesh_volume(Some(m), None, Some(&center)) as f64)
    )
    .unwrap();
    let bb = imod_mesh_get_bbox(Some(m), &mut ll, &mut ur);
    writeln!(
        out,
        "{} bbox={} ll={} {} {} ur={} {} {}",
        tag,
        bb,
        g9(ll.x as f64),
        g9(ll.y as f64),
        g9(ll.z as f64),
        g9(ur.x as f64),
        g9(ur.y as f64),
        g9(ur.z as f64)
    )
    .unwrap();
    out
}

/// Reproduces the `imesh.c` differential driver line for line.
#[test]
fn imesh_matches_native_libimod_driver() {
    use std::fmt::Write as _;
    let mut out = String::new();

    /* factors */
    for i in -26..=-21 {
        let mut list_inc = -99;
        let mut vert_base = -99;
        let mut norm_add = -99;
        let err = imod_mesh_poly_norm_factors(i, &mut list_inc, &mut vert_base, &mut norm_add);
        writeln!(
            out,
            "factors({})={} {} {} {}",
            i, err, list_inc, vert_base, norm_add
        )
        .unwrap();
    }

    let mut seed = MeshSeed(12345);
    let m = build_box(&mut seed, false, 10., 7., 3., 0.);
    out.push_str(&report_mesh("boxNew", &m));
    let m2 = build_box(&mut seed, true, 10., 7., 3., 0.);
    out.push_str(&report_mesh("boxOld", &m2));

    seed = MeshSeed(999);
    let j = build_box(&mut seed, false, 13.25, 4.5, 9.75, 1.3);
    out.push_str(&report_mesh("boxJit", &j));

    /* Large-magnitude jittered boxes: float vs double accumulation */
    let mut seed = MeshSeed(31337);
    let g = build_box(&mut seed, false, 1013.5, 827.25, 631.75, 47.3);
    out.push_str(&report_mesh("boxBig", &g));
    let mut seed = MeshSeed(6001);
    let g = build_box(&mut seed, true, 2411.75, 1907.5, 1633.25, 91.7);
    out.push_str(&report_mesh("boxBigOld", &g));

    /* Open and closed jittered meshes at seeds where the float-width choices in
    the imeshVolume centroid sums and the imeshSurfaceArea normal magnitudes are
    visible in the returned float */
    for open_seed in [185u32, 6, 11] {
        let mut seed = MeshSeed(open_seed);
        let o = build_box_n(&mut seed, false, 1013.5, 827.25, 631.75, 147.3, 4);
        out.push_str(&report_mesh(&format!("boxOpen{}", open_seed), &o));
    }
    for closed_seed in [1460u32, 2992] {
        let mut seed = MeshSeed(closed_seed);
        let o = build_box_n(&mut seed, false, 1013.5, 827.25, 631.75, 147.3, 12);
        out.push_str(&report_mesh(&format!("boxRough{}", closed_seed), &o));
    }

    /* empty and null cases */
    writeln!(
        out,
        "volNull={} areaNull={}",
        g9(imesh_volume(None, None, None) as f64),
        g9(imesh_surface_area(None, None) as f64)
    )
    .unwrap();
    let e = imod_mesh_new().unwrap().remove(0);
    let mut ll = Ipoint::default();
    let mut ur = Ipoint::default();
    writeln!(
        out,
        "volEmpty={} areaEmpty={} bboxEmpty={}",
        g9(imesh_volume(Some(&e), None, None) as f64),
        g9(imesh_surface_area(Some(&e), None) as f64),
        imod_mesh_get_bbox(Some(&e), &mut ll, &mut ur)
    )
    .unwrap();

    /* accessors */
    writeln!(
        out,
        "getIndex(0)={} getIndex(-1)={} getIndex(9999)={}",
        imod_mesh_get_index(Some(&m), 0),
        imod_mesh_get_index(Some(&m), -1),
        imod_mesh_get_index(Some(&m), 9999)
    )
    .unwrap();
    writeln!(
        out,
        "maxIndex={} maxVert={}",
        imod_mesh_get_max_index(Some(&m)),
        imod_mesh_get_max_vert(Some(&m))
    )
    .unwrap();
    let v3 = imod_mesh_get_vert(Some(&m), 3).unwrap();
    writeln!(
        out,
        "getVert(3)={} {} {}",
        g9(v3.x as f64),
        g9(v3.y as f64),
        g9(v3.z as f64)
    )
    .unwrap();
    writeln!(
        out,
        "getVert(-1)={} getVert(9999)={} verts={}",
        if imod_mesh_get_vert(Some(&m), -1).is_none() {
            "(nil)"
        } else {
            "(ptr)"
        },
        if imod_mesh_get_vert(Some(&m), 9999).is_none() {
            "(nil)"
        } else {
            "(ptr)"
        },
        imod_mesh_get_verts(Some(&m)).is_some() as i32
    )
    .unwrap();
    writeln!(
        out,
        "nullAccessors {} {} {} {} {}",
        imod_mesh_get_index(None, 0),
        imod_mesh_get_max_index(None),
        imod_mesh_get_max_vert(None),
        if imod_mesh_get_vert(None, 0).is_none() {
            "(nil)"
        } else {
            "(ptr)"
        },
        if imod_mesh_get_verts(None).is_none() {
            "(nil)"
        } else {
            "(ptr)"
        }
    )
    .unwrap();

    /* dup */
    let mut dup = imod_mesh_dup(Some(&m)).unwrap();
    writeln!(
        out,
        "dup vsize={} lsize={} vol={}",
        dup.vert.len(),
        dup.list.len(),
        g9(imesh_volume(Some(&dup), None, None) as f64)
    )
    .unwrap();
    writeln!(
        out,
        "copyNull={} {}",
        imod_mesh_copy(None, Some(&mut dup)),
        imod_mesh_copy(Some(&m), None)
    )
    .unwrap();

    /* insert / delete index */
    imod_mesh_insert_index(&mut dup, 4242, 2);
    writeln!(
        out,
        "afterInsert lsize={} list[0..5]={} {} {} {} {} {}",
        dup.list.len(),
        dup.list[0],
        dup.list[1],
        dup.list[2],
        dup.list[3],
        dup.list[4],
        dup.list[5]
    )
    .unwrap();
    imod_mesh_delete_index(Some(&mut dup), 2);
    writeln!(
        out,
        "afterDelIdx lsize={} list[0..5]={} {} {} {} {} {}",
        dup.list.len(),
        dup.list[0],
        dup.list[1],
        dup.list[2],
        dup.list[3],
        dup.list[4],
        dup.list[5]
    )
    .unwrap();
    imod_mesh_delete_index(Some(&mut dup), -1);
    imod_mesh_delete_index(Some(&mut dup), 99999);
    imod_mesh_delete_index(None, 0);
    writeln!(out, "afterDelMiss lsize={}", dup.list.len()).unwrap();
    let p = Ipoint {
        x: 1.5,
        y: 2.5,
        z: 3.5,
    };
    imod_mesh_add_normal(Some(&mut dup), Some(&p));
    let lsize = dup.list.len();
    writeln!(
        out,
        "afterAddNorm vsize={} lsize={} tail={} {}",
        dup.vert.len(),
        lsize,
        dup.list[lsize - 2],
        dup.list[lsize - 1]
    )
    .unwrap();
    writeln!(
        out,
        "addNormNull={} {}",
        imod_mesh_add_normal(None, Some(&p)),
        imod_mesh_add_normal(Some(&mut dup), None)
    )
    .unwrap();

    /* nearest res */
    let mut arr: Vec<Imesh> = Vec::new();
    let flags = [0_u32, 2, 5, 5, 9];
    for flag in flags {
        let tmp = Imesh {
            flag: flag << IMESH_FLAG_RES_SHIFT,
            ..Imesh::default()
        };
        imodel_mesh_add(Some(&tmp), &mut arr);
    }
    writeln!(out, "meshAddSize={}", arr.len()).unwrap();
    let mut res = 0;
    for i in -2..=11 {
        let ndiff = imod_mesh_nearest_res(&arr, arr.len() as i32, i, &mut res);
        writeln!(out, "nearestRes({})={} out={}", i, ndiff, res).unwrap();
    }
    let ndiff = imod_mesh_nearest_res(&arr, 1, 7, &mut res);
    writeln!(out, "nearestRes1={} out={}", ndiff, res).unwrap();
    let ndiff = imod_mesh_nearest_res(&[], 3, 7, &mut res);
    writeln!(out, "nearestResNull={} out={}", ndiff, res).unwrap();
    let ndiff = imod_mesh_nearest_res(&arr, 0, 7, &mut res);
    writeln!(out, "nearestRes0={} out={}", ndiff, res).unwrap();

    /* sort surfaces: three disjoint boxes in one mesh */
    let mut s = imod_mesh_new().unwrap().remove(0);
    let mut seed = MeshSeed(4242);
    for b in 0..3 {
        for k in 0..8 {
            let p = Ipoint {
                x: (if k & 1 != 0 { 5. } else { 0. }) + 20. * b as f32,
                y: if k & 2 != 0 { 6. } else { 0. },
                z: if k & 4 != 0 { 7. } else { 0. },
            };
            imod_mesh_add_vert(&mut s, &p);
            let n = Ipoint {
                x: seed.nextf(-1., 1.),
                y: seed.nextf(-1., 1.),
                z: seed.nextf(-1., 1.),
            };
            imod_mesh_add_vert(&mut s, &n);
        }
    }
    imod_mesh_add_index(&mut s, IMOD_MESH_BGNPOLYNORM2);
    for b in 0..3 {
        let base = 16 * b;
        for t in 0..12 {
            for k in 0..3 {
                imod_mesh_add_index(&mut s, base + 2 * MESH_TRI[t][k] as i32);
            }
        }
    }
    imod_mesh_add_index(&mut s, IMOD_MESH_ENDPOLY);
    imod_mesh_add_index(&mut s, IMOD_MESH_END);
    writeln!(
        out,
        "sortInput vsize={} lsize={} vol={} area={}",
        s.vert.len(),
        s.list.len(),
        g9(imesh_volume(Some(&s), None, None) as f64),
        g9(imesh_surface_area(Some(&s), None) as f64)
    )
    .unwrap();
    let mut nsurf = 0;
    let mut err = 0;
    let sorted = imesh_sort_surfaces(Some(&s), &mut nsurf, &mut err).unwrap();
    writeln!(out, "sort nsurf={} err={}", nsurf, err).unwrap();
    for i in 0..nsurf as usize {
        writeln!(
            out,
            "surf {}: surfnum={} vsize={} lsize={} vol={} area={}",
            i,
            sorted[i].surf,
            sorted[i].vert.len(),
            sorted[i].list.len(),
            g9(imesh_volume(Some(&sorted[i]), None, None) as f64),
            g9(imesh_surface_area(Some(&sorted[i]), None) as f64)
        )
        .unwrap();
        write!(out, "surf {} list:", i).unwrap();
        for k in 0..sorted[i].list.len() {
            write!(out, " {}", sorted[i].list[k]).unwrap();
        }
        out.push('\n');
        write!(out, "surf {} vert:", i).unwrap();
        for k in 0..sorted[i].vert.len() {
            write!(
                out,
                " {}/{}/{}",
                g9(sorted[i].vert[k].x as f64),
                g9(sorted[i].vert[k].y as f64),
                g9(sorted[i].vert[k].z as f64)
            )
            .unwrap();
        }
        out.push('\n');
    }
    let mut nsurf2 = 0;
    let mut err2 = 0;
    let sorted_null = imesh_sort_surfaces(None, &mut nsurf2, &mut err2);
    writeln!(
        out,
        "sortNull={} nsurf={} err={}",
        if sorted_null.is_none() {
            "(nil)"
        } else {
            "(ptr)"
        },
        nsurf2,
        err2
    )
    .unwrap();
    imod_mesh_add_index(&mut s, IMOD_MESH_BGNPOLY);
    let sorted_bad = imesh_sort_surfaces(Some(&s), &mut nsurf2, &mut err2);
    writeln!(
        out,
        "sortBigPoly={} nsurf={} err={}",
        if sorted_bad.is_none() {
            "(nil)"
        } else {
            "(ptr)"
        },
        nsurf2,
        err2
    )
    .unwrap();

    /* interp cont */
    let mut cont = Icont::default();
    let mut seedb = MeshSeed(12345);
    let b = build_box(&mut seedb, false, 10., 7., 3., 0.);
    let blist = b.list[1..].to_vec();
    imod_mesh_interp_cont(&blist, &b.vert, 12, 0, 1, 1, &mut cont);
    writeln!(out, "interp psize={}", cont.pts.len()).unwrap();
    for i in 0..cont.pts.len() {
        writeln!(
            out,
            "interp {}: {} {} {}",
            i,
            g9(cont.pts[i].x as f64),
            g9(cont.pts[i].y as f64),
            g9(cont.pts[i].z as f64)
        )
        .unwrap();
    }

    /* interp cont on a jittered box: non-integral interpolation fractions */
    let mut zv = 0;
    while zv <= 8 {
        let mut cont = Icont::default();
        let mut seed = MeshSeed(999);
        let b = build_box(&mut seed, false, 13.25, 4.5, 9.75, 1.3);
        let blist = b.list[1..].to_vec();
        imod_mesh_interp_cont(&blist, &b.vert, 12, 0, 1, zv, &mut cont);
        writeln!(out, "interpJit z={} psize={}", zv, cont.pts.len()).unwrap();
        for i in 0..cont.pts.len() {
            writeln!(
                out,
                "interpJit {} {}: {} {} {}",
                zv,
                i,
                g9(cont.pts[i].x as f64),
                g9(cont.pts[i].y as f64),
                g9(cont.pts[i].z as f64)
            )
            .unwrap();
        }
        zv += 4;
    }
    let mut zv = 200;
    while zv <= 400 {
        let mut cont = Icont::default();
        let mut seed = MeshSeed(31337);
        let b = build_box(&mut seed, false, 1013.5, 827.25, 631.75, 47.3);
        let blist = b.list[1..].to_vec();
        imod_mesh_interp_cont(&blist, &b.vert, 12, 0, 1, zv, &mut cont);
        writeln!(out, "interpBig z={} psize={}", zv, cont.pts.len()).unwrap();
        for i in 0..cont.pts.len() {
            writeln!(
                out,
                "interpBig {} {}: {} {} {}",
                zv,
                i,
                g9(cont.pts[i].x as f64),
                g9(cont.pts[i].y as f64),
                g9(cont.pts[i].z as f64)
            )
            .unwrap();
        }
        zv += 200;
    }

    /* mesh params */
    let mut par = imesh_params_new().unwrap();
    writeln!(
        out,
        "params flags={} cap={} passes={} capSkipNz={} incz={}/{} minz={} maxz={} spareInt={}",
        par.flags,
        par.cap,
        par.passes,
        par.cap_skip_nz,
        par.incz_low_res,
        par.incz_high_res,
        par.minz,
        par.maxz,
        par.spare_int
    )
    .unwrap();
    writeln!(
        out,
        "params f {} {} {} {} {} {} {} {} {} {}",
        g9(par.overlap as f64),
        g9(par.tube_diameter as f64),
        g9(par.xmin as f64),
        g9(par.xmax as f64),
        g9(par.ymin as f64),
        g9(par.ymax as f64),
        g9(par.tol_low_res as f64),
        g9(par.tol_high_res as f64),
        g9(par.flat_crit as f64),
        g9(par.spare_float as f64)
    )
    .unwrap();
    writeln!(
        out,
        "params zlist={}",
        if par.cap_skip_zlist.is_none() {
            "(nil)"
        } else {
            "(ptr)"
        }
    )
    .unwrap();
    let from = [3, 9, 27, 81];
    let mut to: Option<Vec<i32>> = None;
    let mut nto = 0;
    let rc = imesh_copy_skip_list(Some(&from), 4, &mut to, &mut nto);
    writeln!(out, "copySkip={} nto={}", rc, nto).unwrap();
    for i in 0..nto as usize {
        writeln!(out, "copySkip {}={}", i, to.as_ref().unwrap()[i]).unwrap();
    }
    let rc = imesh_copy_skip_list(Some(&from), 2, &mut to, &mut nto);
    writeln!(out, "copySkipAgain={} nto={}", rc, nto).unwrap();
    for i in 0..nto as usize {
        writeln!(out, "copySkipAgain {}={}", i, to.as_ref().unwrap()[i]).unwrap();
    }
    let rc = imesh_copy_skip_list(None, 4, &mut to, &mut nto);
    writeln!(out, "copySkipNull={} nto={}", rc, nto).unwrap();
    par.cap_skip_zlist = to;
    par.cap_skip_nz = 0;

    par.cap_skip_nz = 2;
    par.cap_skip_zlist = Some(vec![11, 22]);
    let pdup = imesh_params_dup(Some(&par)).unwrap();
    writeln!(
        out,
        "dupParams nz={} z0={} z1={} flags={}",
        pdup.cap_skip_nz,
        pdup.cap_skip_zlist.as_ref().unwrap()[0],
        pdup.cap_skip_zlist.as_ref().unwrap()[1],
        pdup.flags
    )
    .unwrap();
    par.cap_skip_zlist = None;
    par.cap_skip_nz = 0;
    writeln!(
        out,
        "dupParamsNull={}",
        if imesh_params_dup(None).is_none() {
            "(nil)"
        } else {
            "(ptr)"
        }
    )
    .unwrap();
    imesh_params_delete(None);

    assert_eq!(out, NATIVE_IMESH.trim_start_matches('\n'));
}

/// Native output of `scratchpad/iview_diff.c`, compiled against the pinned
/// `IMOD/include/iview.h` and linked to the reference `libimod`.
const NATIVE_IVIEW: &str = r#"
BYTES_PER_OBJVIEW 187 IMOD_CLIPSIZE 6
default fovy=0 rad=1 aspect=1 near=0 far=1
default rot=0 0 0 trans=0 0 0 scale=1 1 1
default mat 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
default world=2 lightx=0 lighty=0 plax=5 dcstart=0 dcend=1 objvsize=0
default label 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
default clips 0 0 0 0
default clip 0 n=0 0 -1 p=0 0 0
default clip 1 n=0 0 -1 p=0 0 0
default clip 2 n=0 0 -1 p=0 0 0
default clip 3 n=0 0 -1 p=0 0 0
default clip 4 n=0 0 -1 p=0 0 0
default clip 5 n=0 0 -1 p=0 0 0
imnx o=1 1 1 / 0 0 0 / 0 0 0
imnx c=1 1 1 / 0 0 0 / 0 0 0
objsize=3 viewsize=1
viewModelNew=0
viewModelNew=0
viewsize=3
objviewComplete=0
view1 fovy=0 rad=1 aspect=1 near=0 far=1
view1 rot=0 0 0 trans=0 0 0 scale=1 1 1
view1 mat 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
view1 world=2 lightx=0 lighty=0 plax=5 dcstart=0 dcend=1 objvsize=3
view1 label 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
view1 clips 0 0 0 0
view1 clip 0 n=0 0 -1 p=0 0 0
view1 clip 1 n=0 0 -1 p=0 0 0
view1 clip 2 n=0 0 -1 p=0 0 0
view1 clip 3 n=0 0 -1 p=0 0 0
view1 clip 4 n=0 0 -1 p=0 0 0
view1 clip 5 n=0 0 -1 p=0 0 0
view1 ov 0 flags=4660 rgb=0.1614483 0.631117225 0.0733317137 pdraw=17 lw=3 ls=1 tr=40
view1 ov 0 mat=11 22 33 44 / 55 66 77 88 / 3735928559 / 5 250 3 7
view1 ov 0 clips 3 5 9 1
view1 ov 0 clip 0 n=1.72844648 -1.58806372 2.13094664 p=-35.4297371 11.4840622 -82.3592758
view1 ov 0 clip 1 n=-2.67935133 -0.479504585 -1.3894254 p=-7.6544342 63.0921631 -81.3040619
view1 ov 0 clip 2 n=-1.69955027 -1.53721476 1.14788198 p=-2.21979523 58.4151001 23.3835068
view1 ov 0 clip 3 n=-1.07919824 -0.302576542 1.44804192 p=10.9008865 -33.6138954 -63.4404221
view1 ov 0 clip 4 n=-2.26266384 1.01762676 1.64821148 p=29.546875 4.94488525 48.8490601
view1 ov 0 clip 5 n=-2.67342424 -1.40390182 1.02473259 p=-14.8396683 -72.2933502 -73.9479828
view1 ov 1 flags=4661 rgb=0.0235437751 0.816229582 0.789678633 pdraw=18 lw=4 ls=1 tr=41
view1 ov 1 mat=12 22 33 44 / 55 66 77 88 / 3735928558 / 5 250 3 8
view1 ov 1 clips 3 5 9 1
view1 ov 1 clip 0 n=-0.802461147 2.91710567 -2.03845954 p=-19.6986542 -68.8363495 -67.4694824
view1 ov 1 clip 1 n=-2.03033447 2.66448975 2.86802006 p=36.9245529 -29.0390434 -45.8509789
view1 ov 1 clip 2 n=0.488464355 0.0080280304 -0.880050182 p=43.3789978 31.4328308 22.8115845
view1 ov 1 clip 3 n=-0.33157897 -0.192867994 2.29654408 p=64.4556274 46.2038574 69.7678375
view1 ov 1 clip 4 n=1.78647947 1.17300272 -2.36799645 p=-48.7205124 56.1702881 -64.8587494
view1 ov 1 clip 5 n=-0.0943787098 1.18770027 2.21604538 p=75.8422241 -26.3197861 28.4714966
view1 ov 2 flags=4662 rgb=0.0539777875 0.667248189 0.511128783 pdraw=19 lw=5 ls=1 tr=42
view1 ov 2 mat=13 22 33 44 / 55 66 77 88 / 3735928557 / 5 250 3 9
view1 ov 2 clips 3 5 9 1
view1 ov 2 clip 0 n=0.921127081 1.148139 -1.57825077 p=-74.0151062 66.5318146 -21.4609909
view1 ov 2 clip 1 n=-1.70985353 2.86526203 -0.592483044 p=-66.0928497 -37.3064041 39.7884521
view1 ov 2 clip 2 n=-0.40527606 0.198831081 -2.85628843 p=32.2652283 -55.962307 18.7983704
view1 ov 2 clip 3 n=-1.86458516 -2.95784211 -1.36015785 p=51.9441376 45.5198517 44.1321564
view1 ov 2 clip 4 n=1.43442678 -1.67569363 -2.11779976 p=-77.285759 -86.6771622 -50.8935623
view1 ov 2 clip 5 n=-0.796382189 0.0160703659 0.346750259 p=26.9240265 -69.4426498 20.1814041
view2 fovy=0 rad=1 aspect=1 near=0 far=1
view2 rot=0 0 0 trans=0 0 0 scale=1 1 1
view2 mat 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
view2 world=2 lightx=0 lighty=0 plax=5 dcstart=0 dcend=1 objvsize=3
view2 label 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
view2 clips 0 0 0 0
view2 clip 0 n=0 0 -1 p=0 0 0
view2 clip 1 n=0 0 -1 p=0 0 0
view2 clip 2 n=0 0 -1 p=0 0 0
view2 clip 3 n=0 0 -1 p=0 0 0
view2 clip 4 n=0 0 -1 p=0 0 0
view2 clip 5 n=0 0 -1 p=0 0 0
view2 ov 0 flags=4660 rgb=0.1614483 0.631117225 0.0733317137 pdraw=17 lw=3 ls=1 tr=40
view2 ov 0 mat=11 22 33 44 / 55 66 77 88 / 3735928559 / 5 250 3 7
view2 ov 0 clips 3 5 9 1
view2 ov 0 clip 0 n=1.72844648 -1.58806372 2.13094664 p=-35.4297371 11.4840622 -82.3592758
view2 ov 0 clip 1 n=-2.67935133 -0.479504585 -1.3894254 p=-7.6544342 63.0921631 -81.3040619
view2 ov 0 clip 2 n=-1.69955027 -1.53721476 1.14788198 p=-2.21979523 58.4151001 23.3835068
view2 ov 0 clip 3 n=-1.07919824 -0.302576542 1.44804192 p=10.9008865 -33.6138954 -63.4404221
view2 ov 0 clip 4 n=-2.26266384 1.01762676 1.64821148 p=29.546875 4.94488525 48.8490601
view2 ov 0 clip 5 n=-2.67342424 -1.40390182 1.02473259 p=-14.8396683 -72.2933502 -73.9479828
view2 ov 1 flags=4661 rgb=0.0235437751 0.816229582 0.789678633 pdraw=18 lw=4 ls=1 tr=41
view2 ov 1 mat=12 22 33 44 / 55 66 77 88 / 3735928558 / 5 250 3 8
view2 ov 1 clips 3 5 9 1
view2 ov 1 clip 0 n=-0.802461147 2.91710567 -2.03845954 p=-19.6986542 -68.8363495 -67.4694824
view2 ov 1 clip 1 n=-2.03033447 2.66448975 2.86802006 p=36.9245529 -29.0390434 -45.8509789
view2 ov 1 clip 2 n=0.488464355 0.0080280304 -0.880050182 p=43.3789978 31.4328308 22.8115845
view2 ov 1 clip 3 n=-0.33157897 -0.192867994 2.29654408 p=64.4556274 46.2038574 69.7678375
view2 ov 1 clip 4 n=1.78647947 1.17300272 -2.36799645 p=-48.7205124 56.1702881 -64.8587494
view2 ov 1 clip 5 n=-0.0943787098 1.18770027 2.21604538 p=75.8422241 -26.3197861 28.4714966
view2 ov 2 flags=4662 rgb=0.0539777875 0.667248189 0.511128783 pdraw=19 lw=5 ls=1 tr=42
view2 ov 2 mat=13 22 33 44 / 55 66 77 88 / 3735928557 / 5 250 3 9
view2 ov 2 clips 3 5 9 1
view2 ov 2 clip 0 n=0.921127081 1.148139 -1.57825077 p=-74.0151062 66.5318146 -21.4609909
view2 ov 2 clip 1 n=-1.70985353 2.86526203 -0.592483044 p=-66.0928497 -37.3064041 39.7884521
view2 ov 2 clip 2 n=-0.40527606 0.198831081 -2.85628843 p=32.2652283 -55.962307 18.7983704
view2 ov 2 clip 3 n=-1.86458516 -2.95784211 -1.36015785 p=51.9441376 45.5198517 44.1321564
view2 ov 2 clip 4 n=1.43442678 -1.67569363 -2.11779976 p=-77.285759 -86.6771622 -50.8935623
view2 ov 2 clip 5 n=-0.796382189 0.0160703659 0.346750259 p=26.9240265 -69.4426498 20.1814041
viewWrite=0
write1 bytes 813
 56 49 45 57 00 00 02 e9 bf cb e1 f8 40 a6 11 5d
 3f 80 00 00 00 00 00 00 3f 80 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 3f 80 00 00 3f 80 00 00 3f 80 00 00
 bf 67 52 c0 3f 6d da fa be aa 2d 9c be 34 c9 a0
 3e 0b 71 60 3e e9 50 ac bf 35 64 b0 bf 5f 5b 1a
 bf 75 be b0 bf 00 e4 4a 3f 42 0d ea 3f 02 ae a4
 be b3 ce 78 3f 38 51 d0 bc 5d 29 80 be 06 76 d8
 00 00 00 02 61 20 76 69 65 77 20 6c 61 62 65 6c
 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 3f 80 00 00 00 00 00 00
 00 00 00 00 3e 5a e3 08 00 00 00 03 00 00 02 31
 00 00 12 34 3e 25 52 b4 3f 21 90 e6 3d 96 2e f0
 00 00 00 11 03 01 28 03 05 09 01 3f b0 fe 30 c0
 07 83 c8 3f 1b dd 10 c2 31 26 10 41 09 cf 0a c3
 90 20 f5 0b 16 21 2c 37 42 4d 58 de ad be ef 05
 fa 03 07 c0 09 2e cb bf 23 ab c0 be cb 40 cb bf
 ae 08 b1 c0 03 2c f8 3e a7 eb 33 bf 5d 05 12 be
 ce 8f 16 3e d3 d3 ee bf e7 b2 60 3f ad ac cb 3e
 f1 1c 1c c0 08 e1 1b bf ef 99 68 3e 95 e7 5c c1
 19 16 b4 42 3d 46 c8 c3 8e 48 38 c0 31 95 68 42
 2f 3e cc 42 a3 af 3e 41 5a 04 8a c1 c9 ae f2 c3
 5e 0a 9e 42 13 bc 00 40 6d 5a c0 43 2a f8 c2 c1
 94 65 8d c2 58 e1 4b c3 81 68 b2 00 00 12 35 3c
 c0 de e0 3f 50 f4 6c 3f 4a 28 61 00 00 00 12 04
 01 29 03 05 09 01 bf 24 58 13 40 78 ed 26 bf 15
 19 48 c1 c4 fc 8e c2 4e 82 51 c3 6c 24 a8 0c 16
 21 2c 37 42 4d 58 de ad be ee 05 fa 03 08 bf cf
 e8 00 40 63 5e ab 3f 51 c6 73 3e c8 13 33 3c 2f
 60 00 be 80 bd 20 be 87 d0 93 be 83 aa 20 3f 27
 f9 ce 3f b6 ef 7d 3f c8 31 46 bf 2d 33 b9 bd 9a
 a1 4d 3f ca b3 6b 3f 22 16 80 42 38 9f 6e c1 ae
 3b f8 c3 20 7a 7a 42 58 e5 1e 41 bc 98 d4 42 9f
 ae 5c 42 a1 23 9a 42 0a 9c 90 43 74 2f fc c2 73
 9a 42 42 28 82 c8 c3 63 01 70 42 bd 9b 06 c1 9d
 eb 31 42 c7 4c ec 00 00 12 36 3d 5d 17 d0 3f 2a
 d0 c7 3f 02 d9 56 00 00 00 13 05 01 2a 03 05 09
 01 3f 3c a5 97 3f c3 f2 f6 be e6 e0 24 c2 b9 09
 ab 42 47 98 6f c2 96 3a 18 0d 16 21 2c 37 42 4d
 58 de ad be ed 05 fa 03 09 bf af 16 c9 40 74 80
 9b be 2d 57 f7 be a6 00 47 3e 87 bc 40 bf 50 ea
 c7 bf be ee fb c0 7c 67 0c be c6 f8 bf 3f 92 e2
 a3 c0 0e fe 16 bf 1a e6 e5 bf 23 19 5d 3c af 88
 00 3d ca e6 01 c2 a5 3b 6c c1 df d6 a3 43 0b 42
 74 42 21 53 7e c2 27 e3 0d 42 83 96 ae 42 81 dc
 40 42 08 8f 3f 43 1a 76 6a c2 c1 36 e3 c2 82 04
 08 c3 32 20 a2 42 06 9e c1 c2 50 53 f4 42 8d 45
 14 4d 43 4c 50 00 00 00 34 02 03 0c 01 3f 5f c0
 d7 3f 2b 2a 60 3f 08 21 10 bf 1a a3 da bf 85 6a
 b3 be 3d 3c ea 40 a9 78 9c 41 9b 02 3e 42 92 74
 ac c2 5b d1 e0 40 8b 98 90 43 42 f0 c7
viewWrite2=0
write2 bytes 753
 56 49 45 57 00 00 02 e9 00 00 00 00 3f 80 00 00
 3f 80 00 00 00 00 00 00 3f 80 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 3f 80 00 00 3f 80 00 00 3f 80 00 00
 3f 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 3f 80 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 3f 80 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 00 00 00 00 3f 80 00 00
 00 00 00 02 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 3f 80 00 00 00 00 00 00
 00 00 00 00 40 a0 00 00 00 00 00 03 00 00 02 31
 00 00 12 34 3e 25 52 b4 3f 21 90 e6 3d 96 2e f0
 00 00 00 11 03 01 28 00 00 09 01 3f b0 fe 30 c0
 07 83 c8 3f 1b dd 10 c2 31 26 10 41 09 cf 0a c3
 90 20 f5 0b 16 21 2c 37 42 4d 58 de ad be ef 05
 fa 03 07 c0 09 2e cb bf 23 ab c0 be cb 40 cb bf
 ae 08 b1 c0 03 2c f8 3e a7 eb 33 bf 5d 05 12 be
 ce 8f 16 3e d3 d3 ee bf e7 b2 60 3f ad ac cb 3e
 f1 1c 1c c0 08 e1 1b bf ef 99 68 3e 95 e7 5c c1
 19 16 b4 42 3d 46 c8 c3 8e 48 38 c0 31 95 68 42
 2f 3e cc 42 a3 af 3e 41 5a 04 8a c1 c9 ae f2 c3
 5e 0a 9e 42 13 bc 00 40 6d 5a c0 43 2a f8 c2 c1
 94 65 8d c2 58 e1 4b c3 81 68 b2 00 00 12 35 3c
 c0 de e0 3f 50 f4 6c 3f 4a 28 61 00 00 00 12 04
 01 29 01 01 09 01 bf 24 58 13 40 78 ed 26 bf 15
 19 48 c1 c4 fc 8e c2 4e 82 51 c3 6c 24 a8 0c 16
 21 2c 37 42 4d 58 de ad be ee 05 fa 03 08 bf cf
 e8 00 40 63 5e ab 3f 51 c6 73 3e c8 13 33 3c 2f
 60 00 be 80 bd 20 be 87 d0 93 be 83 aa 20 3f 27
 f9 ce 3f b6 ef 7d 3f c8 31 46 bf 2d 33 b9 bd 9a
 a1 4d 3f ca b3 6b 3f 22 16 80 42 38 9f 6e c1 ae
 3b f8 c3 20 7a 7a 42 58 e5 1e 41 bc 98 d4 42 9f
 ae 5c 42 a1 23 9a 42 0a 9c 90 43 74 2f fc c2 73
 9a 42 42 28 82 c8 c3 63 01 70 42 bd 9b 06 c1 9d
 eb 31 42 c7 4c ec 00 00 12 36 3d 5d 17 d0 3f 2a
 d0 c7 3f 02 d9 56 00 00 00 13 05 01 2a 03 05 09
 01 3f 3c a5 97 3f c3 f2 f6 be e6 e0 24 c2 b9 09
 ab 42 47 98 6f c2 96 3a 18 0d 16 21 2c 37 42 4d
 58 de ad be ed 05 fa 03 09 bf af 16 c9 40 74 80
 9b be 2d 57 f7 be a6 00 47 3e 87 bc 40 bf 50 ea
 c7 bf be ee fb c0 7c 67 0c be c6 f8 bf 3f 92 e2
 a3 c0 0e fe 16 bf 1a e6 e5 bf 23 19 5d 3c af 88
 00 3d ca e6 01 c2 a5 3b 6c c1 df d6 a3 43 0b 42
 74 42 21 53 7e c2 27 e3 0d 42 83 96 ae 42 81 dc
 40 42 08 8f 3f 43 1a 76 6a c2 c1 36 e3 c2 82 04
 08 c3 32 20 a2 42 06 9e c1 c2 50 53 f4 42 8d 45
 14
viewWrite3=0
write3 bytes 184
 56 49 45 57 00 00 00 b0 00 00 00 00 3f 80 00 00
 3f 80 00 00 00 00 00 00 3f 80 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 3f 80 00 00 3f 80 00 00 3f 80 00 00
 3f 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 3f 80 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 3f 80 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 00 00 00 00 3f 80 00 00
 00 00 00 02 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
 00 00 00 00 00 00 00 00 3f 80 00 00 00 00 00 00
 00 00 00 00 40 a0 00 00
fromObj flags=4661 rgb=0.0235437751 0.816229582 0.789678633 pd=18 lw=4 ls=1 tr=41
fromObj mat=12 22 33 44 / 55 66 77 88 / 3735928558 / 5 250 3 8
fromObj clips 3 5 9 1
toObj flags=4661 rgb=0.0235437751 0.816229582 0.789678633 pd=18 lw=4 ls=1 tr=41
toObj mat=12 22 33 44 / 55 66 77 88 / 3735928558 / 5 250 3 8
viewStore=0
stored fovy=0 rad=42.5 aspect=1 near=0 far=1
stored rot=0 0 0 trans=0 0 0 scale=1 1 1
stored mat 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
stored world=2 lightx=0 lighty=0 plax=5 dcstart=0 dcend=1 objvsize=3
stored label 97 32 118 105 101 119 32 108 97 98 101 108 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
stored clips 0 0 0 0
stored clip 0 n=0 0 -1 p=0 0 0
stored clip 1 n=0 0 -1 p=0 0 0
stored clip 2 n=0 0 -1 p=0 0 0
stored clip 3 n=0 0 -1 p=0 0 0
stored clip 4 n=0 0 -1 p=0 0 0
stored clip 5 n=0 0 -1 p=0 0 0
stored ov 0 flags=4660 rgb=0.1614483 0.631117225 0.0733317137 pdraw=17 lw=3 ls=1 tr=40
stored ov 0 mat=11 22 33 44 / 55 66 77 88 / 3735928559 / 5 250 3 7
stored ov 0 clips 3 5 9 1
stored ov 0 clip 0 n=1.72844648 -1.58806372 2.13094664 p=-35.4297371 11.4840622 -82.3592758
stored ov 0 clip 1 n=-2.67935133 -0.479504585 -1.3894254 p=-7.6544342 63.0921631 -81.3040619
stored ov 0 clip 2 n=-1.69955027 -1.53721476 1.14788198 p=-2.21979523 58.4151001 23.3835068
stored ov 0 clip 3 n=-1.07919824 -0.302576542 1.44804192 p=10.9008865 -33.6138954 -63.4404221
stored ov 0 clip 4 n=-2.26266384 1.01762676 1.64821148 p=29.546875 4.94488525 48.8490601
stored ov 0 clip 5 n=-2.67342424 -1.40390182 1.02473259 p=-14.8396683 -72.2933502 -73.9479828
stored ov 1 flags=4661 rgb=0.0235437751 0.816229582 0.789678633 pdraw=18 lw=4 ls=1 tr=41
stored ov 1 mat=12 22 33 44 / 55 66 77 88 / 3735928558 / 5 250 3 8
stored ov 1 clips 3 5 9 1
stored ov 1 clip 0 n=-0.802461147 2.91710567 -2.03845954 p=-19.6986542 -68.8363495 -67.4694824
stored ov 1 clip 1 n=-2.03033447 2.66448975 2.86802006 p=36.9245529 -29.0390434 -45.8509789
stored ov 1 clip 2 n=0.488464355 0.0080280304 -0.880050182 p=43.3789978 31.4328308 22.8115845
stored ov 1 clip 3 n=-0.33157897 -0.192867994 2.29654408 p=64.4556274 46.2038574 69.7678375
stored ov 1 clip 4 n=1.78647947 1.17300272 -2.36799645 p=-48.7205124 56.1702881 -64.8587494
stored ov 1 clip 5 n=-0.0943787098 1.18770027 2.21604538 p=75.8422241 -26.3197861 28.4714966
stored ov 2 flags=4662 rgb=0.0539777875 0.667248189 0.511128783 pdraw=19 lw=5 ls=1 tr=42
stored ov 2 mat=13 22 33 44 / 55 66 77 88 / 3735928557 / 5 250 3 9
stored ov 2 clips 3 5 9 1
stored ov 2 clip 0 n=0.921127081 1.148139 -1.57825077 p=-74.0151062 66.5318146 -21.4609909
stored ov 2 clip 1 n=-1.70985353 2.86526203 -0.592483044 p=-66.0928497 -37.3064041 39.7884521
stored ov 2 clip 2 n=-0.40527606 0.198831081 -2.85628843 p=32.2652283 -55.962307 18.7983704
stored ov 2 clip 3 n=-1.86458516 -2.95784211 -1.36015785 p=51.9441376 45.5198517 44.1321564
stored ov 2 clip 4 n=1.43442678 -1.67569363 -2.11779976 p=-77.285759 -86.6771622 -50.8935623
stored ov 2 clip 5 n=-0.796382189 0.0160703659 0.346750259 p=26.9240265 -69.4426498 20.1814041
used fovy=0 rad=42.5 aspect=1 near=0 far=1
used rot=0 0 0 trans=0 0 0 scale=1 1 1
used mat 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
used world=2 lightx=0 lighty=0 plax=5 dcstart=0 dcend=1 objvsize=0
used label 100 101 102 97 117 108 116 32 108 97 98 101 108 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
used clips 0 0 0 0
used clip 0 n=0 0 -1 p=0 0 0
used clip 1 n=0 0 -1 p=0 0 0
used clip 2 n=0 0 -1 p=0 0 0
used clip 3 n=0 0 -1 p=0 0 0
used clip 4 n=0 0 -1 p=0 0 0
used clip 5 n=0 0 -1 p=0 0 0
useObj 0 flags=4660 rgb=0.1614483 0.631117225 0.0733317137 pd=17 clips=3
useObj 1 flags=4661 rgb=0.0235437751 0.816229582 0.789678633 pd=18 clips=3
useObj 2 flags=4662 rgb=0.0539777875 0.667248189 0.511128783 pd=19 clips=3
afterDel view 1 objvsize=2
afterDel view 2 objvsize=2
afterDelMiss view 1 objvsize=2
afterDelMiss view 2 objvsize=2
afterDelEdge view 1 objvsize=2
afterDelEdge view 2 objvsize=2
beforeDel0 fovy=0 rad=42.5 aspect=1 near=0 far=1
beforeDel0 rot=0 0 0 trans=0 0 0 scale=1 1 1
beforeDel0 mat 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
beforeDel0 world=2 lightx=0 lighty=0 plax=5 dcstart=0 dcend=1 objvsize=2
beforeDel0 label 97 32 118 105 101 119 32 108 97 98 101 108 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
beforeDel0 clips 0 0 0 0
beforeDel0 clip 0 n=0 0 -1 p=0 0 0
beforeDel0 clip 1 n=0 0 -1 p=0 0 0
beforeDel0 clip 2 n=0 0 -1 p=0 0 0
beforeDel0 clip 3 n=0 0 -1 p=0 0 0
beforeDel0 clip 4 n=0 0 -1 p=0 0 0
beforeDel0 clip 5 n=0 0 -1 p=0 0 0
beforeDel0 ov 0 flags=4660 rgb=0.1614483 0.631117225 0.0733317137 pdraw=17 lw=3 ls=1 tr=40
beforeDel0 ov 0 mat=11 22 33 44 / 55 66 77 88 / 3735928559 / 5 250 3 7
beforeDel0 ov 0 clips 3 5 9 1
beforeDel0 ov 0 clip 0 n=1.72844648 -1.58806372 2.13094664 p=-35.4297371 11.4840622 -82.3592758
beforeDel0 ov 0 clip 1 n=-2.67935133 -0.479504585 -1.3894254 p=-7.6544342 63.0921631 -81.3040619
beforeDel0 ov 0 clip 2 n=-1.69955027 -1.53721476 1.14788198 p=-2.21979523 58.4151001 23.3835068
beforeDel0 ov 0 clip 3 n=-1.07919824 -0.302576542 1.44804192 p=10.9008865 -33.6138954 -63.4404221
beforeDel0 ov 0 clip 4 n=-2.26266384 1.01762676 1.64821148 p=29.546875 4.94488525 48.8490601
beforeDel0 ov 0 clip 5 n=-2.67342424 -1.40390182 1.02473259 p=-14.8396683 -72.2933502 -73.9479828
beforeDel0 ov 1 flags=4662 rgb=0.0539777875 0.667248189 0.511128783 pdraw=19 lw=5 ls=1 tr=42
beforeDel0 ov 1 mat=13 22 33 44 / 55 66 77 88 / 3735928557 / 5 250 3 9
beforeDel0 ov 1 clips 3 5 9 1
beforeDel0 ov 1 clip 0 n=0.921127081 1.148139 -1.57825077 p=-74.0151062 66.5318146 -21.4609909
beforeDel0 ov 1 clip 1 n=-1.70985353 2.86526203 -0.592483044 p=-66.0928497 -37.3064041 39.7884521
beforeDel0 ov 1 clip 2 n=-0.40527606 0.198831081 -2.85628843 p=32.2652283 -55.962307 18.7983704
beforeDel0 ov 1 clip 3 n=-1.86458516 -2.95784211 -1.36015785 p=51.9441376 45.5198517 44.1321564
beforeDel0 ov 1 clip 4 n=1.43442678 -1.67569363 -2.11779976 p=-77.285759 -86.6771622 -50.8935623
beforeDel0 ov 1 clip 5 n=-0.796382189 0.0160703659 0.346750259 p=26.9240265 -69.4426498 20.1814041
afterDel0 fovy=0 rad=42.5 aspect=1 near=0 far=1
afterDel0 rot=0 0 0 trans=0 0 0 scale=1 1 1
afterDel0 mat 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
afterDel0 world=2 lightx=0 lighty=0 plax=5 dcstart=0 dcend=1 objvsize=1
afterDel0 label 97 32 118 105 101 119 32 108 97 98 101 108 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
afterDel0 clips 0 0 0 0
afterDel0 clip 0 n=0 0 -1 p=0 0 0
afterDel0 clip 1 n=0 0 -1 p=0 0 0
afterDel0 clip 2 n=0 0 -1 p=0 0 0
afterDel0 clip 3 n=0 0 -1 p=0 0 0
afterDel0 clip 4 n=0 0 -1 p=0 0 0
afterDel0 clip 5 n=0 0 -1 p=0 0 0
afterDel0 ov 0 flags=4662 rgb=0.0539777875 0.667248189 0.511128783 pdraw=19 lw=5 ls=1 tr=42
afterDel0 ov 0 mat=13 22 33 44 / 55 66 77 88 / 3735928557 / 5 250 3 9
afterDel0 ov 0 clips 3 5 9 1
afterDel0 ov 0 clip 0 n=0.921127081 1.148139 -1.57825077 p=-74.0151062 66.5318146 -21.4609909
afterDel0 ov 0 clip 1 n=-1.70985353 2.86526203 -0.592483044 p=-66.0928497 -37.3064041 39.7884521
afterDel0 ov 0 clip 2 n=-0.40527606 0.198831081 -2.85628843 p=32.2652283 -55.962307 18.7983704
afterDel0 ov 0 clip 3 n=-1.86458516 -2.95784211 -1.36015785 p=51.9441376 45.5198517 44.1321564
afterDel0 ov 0 clip 4 n=1.43442678 -1.67569363 -2.11779976 p=-77.285759 -86.6771622 -50.8935623
afterDel0 ov 0 clip 5 n=-0.796382189 0.0160703659 0.346750259 p=26.9240265 -69.4426498 20.1814041
afterFree view 1 objvsize=0
afterFree view 2 objvsize=0
defScale trans=-320 -240 -30 rad=349.654938
modelDef trans=-320 -240 -30 rad=342.43927 world=2
"#;

/// Test-only: reproduces the C driver's `dumpfile()` for the view chunks.
fn dump_view_file(path: &std::path::Path, tag: &str) -> String {
    use std::fmt::Write as _;
    let bytes = std::fs::read(path).unwrap();
    let mut out = String::new();
    writeln!(out, "{} bytes {}", tag, bytes.len()).unwrap();
    for (i, byte) in bytes.iter().enumerate() {
        write!(out, " {:02x}", byte).unwrap();
        if i % 16 == 15 || i == bytes.len() - 1 {
            out.push('\n');
        }
    }
    out
}

/// Test-only: reproduces the C driver's `showView()`.
fn show_view(tag: &str, vw: &Iview) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(
        out,
        "{} fovy={} rad={} aspect={} near={} far={}",
        tag,
        g9(vw.fovy as f64),
        g9(vw.rad as f64),
        g9(vw.aspect as f64),
        g9(vw.cnear as f64),
        g9(vw.cfar as f64)
    )
    .unwrap();
    writeln!(
        out,
        "{} rot={} {} {} trans={} {} {} scale={} {} {}",
        tag,
        g9(vw.rot.x as f64),
        g9(vw.rot.y as f64),
        g9(vw.rot.z as f64),
        g9(vw.trans.x as f64),
        g9(vw.trans.y as f64),
        g9(vw.trans.z as f64),
        g9(vw.scale.x as f64),
        g9(vw.scale.y as f64),
        g9(vw.scale.z as f64)
    )
    .unwrap();
    write!(out, "{} mat", tag).unwrap();
    for i in 0..16 {
        write!(out, " {}", g9(vw.mat[i] as f64)).unwrap();
    }
    out.push('\n');
    writeln!(
        out,
        "{} world={} lightx={} lighty={} plax={} dcstart={} dcend={} objvsize={}",
        tag,
        vw.world,
        g9(vw.lightx as f64),
        g9(vw.lighty as f64),
        g9(vw.plax as f64),
        g9(vw.dcstart as f64),
        g9(vw.dcend as f64),
        vw.objview.len()
    )
    .unwrap();
    write!(out, "{} label", tag).unwrap();
    for i in 0..VIEW_STRSIZE {
        write!(out, " {}", vw.label[i] as i8 as i32).unwrap();
    }
    out.push('\n');
    writeln!(
        out,
        "{} clips {} {} {} {}",
        tag, vw.clips.count, vw.clips.flags, vw.clips.trans, vw.clips.plane
    )
    .unwrap();
    for i in 0..IMOD_CLIPSIZE {
        writeln!(
            out,
            "{} clip {} n={} {} {} p={} {} {}",
            tag,
            i,
            g9(vw.clips.normal[i].x as f64),
            g9(vw.clips.normal[i].y as f64),
            g9(vw.clips.normal[i].z as f64),
            g9(vw.clips.point[i].x as f64),
            g9(vw.clips.point[i].y as f64),
            g9(vw.clips.point[i].z as f64)
        )
        .unwrap();
    }
    for j in 0..vw.objview.len() {
        let ov = &vw.objview[j];
        writeln!(
            out,
            "{} ov {} flags={} rgb={} {} {} pdraw={} lw={} ls={} tr={}",
            tag,
            j,
            ov.flags,
            g9(ov.red as f64),
            g9(ov.green as f64),
            g9(ov.blue as f64),
            ov.pdrawsize,
            ov.linewidth,
            ov.linesty,
            ov.trans
        )
        .unwrap();
        writeln!(
            out,
            "{} ov {} mat={} {} {} {} / {} {} {} {} / {} / {} {} {} {}",
            tag,
            j,
            ov.ambient,
            ov.diffuse,
            ov.specular,
            ov.shininess,
            ov.fillred,
            ov.fillgreen,
            ov.fillblue,
            ov.quality,
            ov.mat2,
            ov.valblack,
            ov.valwhite,
            ov.matflags2,
            ov.mesh_thickness
        )
        .unwrap();
        writeln!(
            out,
            "{} ov {} clips {} {} {} {}",
            tag, j, ov.clips.count, ov.clips.flags, ov.clips.trans, ov.clips.plane
        )
        .unwrap();
        for i in 0..IMOD_CLIPSIZE {
            writeln!(
                out,
                "{} ov {} clip {} n={} {} {} p={} {} {}",
                tag,
                j,
                i,
                g9(ov.clips.normal[i].x as f64),
                g9(ov.clips.normal[i].y as f64),
                g9(ov.clips.normal[i].z as f64),
                g9(ov.clips.point[i].x as f64),
                g9(ov.clips.point[i].y as f64),
                g9(ov.clips.point[i].z as f64)
            )
            .unwrap();
        }
    }
    out
}

/// Test-only: the C driver's `fillObj()`.
fn fill_obj(obj: &mut Iobj, k: i32, seed: &mut MeshSeed) {
    obj.flags = (0x1234 + k) as u32;
    obj.red = seed.nextf(0., 1.);
    obj.green = seed.nextf(0., 1.);
    obj.blue = seed.nextf(0., 1.);
    obj.pdrawsize = 17 + k;
    obj.linewidth = (3 + k) as u8;
    obj.linesty = 1;
    obj.trans = (40 + k) as u8;
    obj.ambient = (11 + k) as u8;
    obj.diffuse = 22;
    obj.specular = 33;
    obj.shininess = 44;
    obj.fillred = 55;
    obj.fillgreen = 66;
    obj.fillblue = 77;
    obj.quality = 88;
    obj.mat2 = 0xdeadbeefu32.wrapping_sub(k as u32);
    obj.valblack = 5;
    obj.valwhite = 250;
    obj.matflags2 = 3;
    obj.mesh_thickness = (7 + k) as u8;
    obj.clips.count = 3;
    obj.clips.flags = 5;
    obj.clips.trans = 9;
    obj.clips.plane = 1;
    for i in 0..IMOD_CLIPSIZE {
        obj.clips.normal[i].x = seed.nextf(-3., 3.);
        obj.clips.normal[i].y = seed.nextf(-3., 3.);
        obj.clips.normal[i].z = seed.nextf(-3., 3.);
        obj.clips.point[i].x = seed.nextf(-90., 90.);
        obj.clips.point[i].y = seed.nextf(-90., 90.);
        obj.clips.point[i].z = seed.nextf(-90., 90.);
    }
}

/// Reproduces the `iview.c` differential driver line for line.
#[test]
fn iview_matches_native_libimod_driver() {
    use std::fmt::Write as _;
    let dir = std::env::temp_dir().join(format!("imod-rs-iview-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut out = String::new();
    let mut seed = MeshSeed(7777);

    writeln!(
        out,
        "BYTES_PER_OBJVIEW {} IMOD_CLIPSIZE {}",
        BYTES_PER_OBJVIEW, IMOD_CLIPSIZE
    )
    .unwrap();

    /* imodViewNew/Default */
    let mut vw = imod_view_new(1).unwrap().remove(0);
    imod_view_default(&mut vw);
    out.push_str(&show_view("default", &vw));

    /* imodIMNXNew */
    let r = imod_imnx_new().unwrap();
    writeln!(
        out,
        "imnx o={} {} {} / {} {} {} / {} {} {}",
        g9(r.oscale.x as f64),
        g9(r.oscale.y as f64),
        g9(r.oscale.z as f64),
        g9(r.otrans.x as f64),
        g9(r.otrans.y as f64),
        g9(r.otrans.z as f64),
        g9(r.orot.x as f64),
        g9(r.orot.y as f64),
        g9(r.orot.z as f64)
    )
    .unwrap();
    writeln!(
        out,
        "imnx c={} {} {} / {} {} {} / {} {} {}",
        g9(r.cscale.x as f64),
        g9(r.cscale.y as f64),
        g9(r.cscale.z as f64),
        g9(r.ctrans.x as f64),
        g9(r.ctrans.y as f64),
        g9(r.ctrans.z as f64),
        g9(r.crot.x as f64),
        g9(r.crot.y as f64),
        g9(r.crot.z as f64)
    )
    .unwrap();

    /* Build a model with objects and views */
    let mut model = imod_new().unwrap();
    for k in 0..3 {
        imod_new_object(&mut model);
        fill_obj(&mut model.obj[k as usize], k, &mut seed);
    }
    writeln!(
        out,
        "objsize={} viewsize={}",
        model.obj.len(),
        model.view.len()
    )
    .unwrap();

    writeln!(out, "viewModelNew={}", imod_view_model_new(&mut model)).unwrap();
    writeln!(out, "viewModelNew={}", imod_view_model_new(&mut model)).unwrap();
    writeln!(out, "viewsize={}", model.view.len()).unwrap();

    writeln!(out, "objviewComplete={}", imod_objview_complete(&mut model)).unwrap();
    for i in 1..model.view.len() {
        out.push_str(&show_view(&format!("view{}", i), &model.view[i]));
    }

    /* Give view 1 model clip planes and a label */
    model.view[1].clips.count = 2;
    model.view[1].clips.flags = 3;
    model.view[1].clips.trans = 12;
    model.view[1].clips.plane = 1;
    for i in 0..IMOD_CLIPSIZE {
        model.view[1].clips.normal[i].x = seed.nextf(-2., 2.);
        model.view[1].clips.normal[i].y = seed.nextf(-2., 2.);
        model.view[1].clips.normal[i].z = seed.nextf(-2., 2.);
        model.view[1].clips.point[i].x = seed.nextf(-70., 70.);
        model.view[1].clips.point[i].y = seed.nextf(-70., 70.);
        model.view[1].clips.point[i].z = seed.nextf(-70., 70.);
    }
    let label = b"a view label";
    model.view[1].label[..label.len()].copy_from_slice(label);
    model.view[1].fovy = seed.nextf(-5., 5.);
    model.view[1].rad = seed.nextf(1., 9.);
    model.view[1].plax = seed.nextf(-1., 1.);
    for i in 0..16 {
        model.view[1].mat[i] = seed.nextf(-1., 1.);
    }

    /* imodViewWrite with a non-unit scale */
    let scale = Ipoint {
        x: 1.25,
        y: 0.75,
        z: 3.5,
    };
    let p1 = dir.join("iview1.bin");
    let mut f = std::fs::File::create(&p1).unwrap();
    writeln!(
        out,
        "viewWrite={}",
        imod_view_write(&model.view[1], &mut f, &scale)
    )
    .unwrap();
    drop(f);
    out.push_str(&dump_view_file(&p1, "write1"));

    /* View 2 has objviews but one clip plane that is off -> clipOut 0 */
    model.view[2].objview[0].clips.count = 1;
    model.view[2].objview[0].clips.flags = 0;
    model.view[2].objview[1].clips.count = 1;
    model.view[2].objview[1].clips.flags = 1;
    model.view[2].clips.count = 0;
    let p2 = dir.join("iview2.bin");
    let mut f = std::fs::File::create(&p2).unwrap();
    writeln!(
        out,
        "viewWrite2={}",
        imod_view_write(&model.view[2], &mut f, &scale)
    )
    .unwrap();
    drop(f);
    out.push_str(&dump_view_file(&p2, "write2"));

    /* A view with no object views at all */
    let mut one = Iview::default();
    imod_view_default(&mut one);
    one.clips.count = 0;
    let p3 = dir.join("iview3.bin");
    let mut f = std::fs::File::create(&p3).unwrap();
    writeln!(out, "viewWrite3={}", imod_view_write(&one, &mut f, &scale)).unwrap();
    drop(f);
    out.push_str(&dump_view_file(&p3, "write3"));

    /* imodObjviewFromObject / ToObject round trip */
    let mut ov = Iobjview::default();
    let mut obj2 = Iobj::default();
    imod_objview_from_object(&model.obj[1], &mut ov);
    writeln!(
        out,
        "fromObj flags={} rgb={} {} {} pd={} lw={} ls={} tr={}",
        ov.flags,
        g9(ov.red as f64),
        g9(ov.green as f64),
        g9(ov.blue as f64),
        ov.pdrawsize,
        ov.linewidth,
        ov.linesty,
        ov.trans
    )
    .unwrap();
    writeln!(
        out,
        "fromObj mat={} {} {} {} / {} {} {} {} / {} / {} {} {} {}",
        ov.ambient,
        ov.diffuse,
        ov.specular,
        ov.shininess,
        ov.fillred,
        ov.fillgreen,
        ov.fillblue,
        ov.quality,
        ov.mat2,
        ov.valblack,
        ov.valwhite,
        ov.matflags2,
        ov.mesh_thickness
    )
    .unwrap();
    writeln!(
        out,
        "fromObj clips {} {} {} {}",
        ov.clips.count, ov.clips.flags, ov.clips.trans, ov.clips.plane
    )
    .unwrap();
    imod_objview_to_object(&ov, &mut obj2);
    writeln!(
        out,
        "toObj flags={} rgb={} {} {} pd={} lw={} ls={} tr={}",
        obj2.flags,
        g9(obj2.red as f64),
        g9(obj2.green as f64),
        g9(obj2.blue as f64),
        obj2.pdrawsize,
        obj2.linewidth,
        obj2.linesty,
        obj2.trans
    )
    .unwrap();
    writeln!(
        out,
        "toObj mat={} {} {} {} / {} {} {} {} / {} / {} {} {} {}",
        obj2.ambient,
        obj2.diffuse,
        obj2.specular,
        obj2.shininess,
        obj2.fillred,
        obj2.fillgreen,
        obj2.fillblue,
        obj2.quality,
        obj2.mat2,
        obj2.valblack,
        obj2.valwhite,
        obj2.matflags2,
        obj2.mesh_thickness
    )
    .unwrap();

    /* imodViewStore / imodViewUse */
    imod_view_default(&mut model.view[0]);
    model.view[0].rad = 42.5;
    let label = b"default label";
    model.view[0].label[..label.len()].copy_from_slice(label);
    writeln!(out, "viewStore={}", imod_view_store(&mut model, 1)).unwrap();
    out.push_str(&show_view("stored", &model.view[1]));
    model.cview = 1;
    imod_view_use(&mut model);
    out.push_str(&show_view("used", &model.view[0]));
    for k in 0..model.obj.len() {
        writeln!(
            out,
            "useObj {} flags={} rgb={} {} {} pd={} clips={}",
            k,
            model.obj[k].flags,
            g9(model.obj[k].red as f64),
            g9(model.obj[k].green as f64),
            g9(model.obj[k].blue as f64),
            model.obj[k].pdrawsize,
            model.obj[k].clips.count
        )
        .unwrap();
    }

    /* imodObjviewDelete then imodObjviewsFree */
    imod_objview_delete(&mut model, 1);
    for i in 1..model.view.len() {
        writeln!(
            out,
            "afterDel view {} objvsize={}",
            i,
            model.view[i].objview.len()
        )
        .unwrap();
    }
    imod_objview_delete(&mut model, 99);
    for i in 1..model.view.len() {
        writeln!(
            out,
            "afterDelMiss view {} objvsize={}",
            i,
            model.view[i].objview.len()
        )
        .unwrap();
    }
    /* index exactly equal to objvsize must be a no-op */
    let edge = model.view[1].objview.len() as i32;
    imod_objview_delete(&mut model, edge);
    for i in 1..model.view.len() {
        writeln!(
            out,
            "afterDelEdge view {} objvsize={}",
            i,
            model.view[i].objview.len()
        )
        .unwrap();
    }
    /* index 0 shifts the rest down */
    out.push_str(&show_view("beforeDel0", &model.view[1]));
    imod_objview_delete(&mut model, 0);
    out.push_str(&show_view("afterDel0", &model.view[1]));
    imod_objviews_free(&mut model);
    for i in 1..model.view.len() {
        writeln!(
            out,
            "afterFree view {} objvsize={}",
            i,
            model.view[i].objview.len()
        )
        .unwrap();
    }

    /* imodViewDefaultScale on the model */
    let image_max = Ipoint {
        x: 640.,
        y: 480.,
        z: 60.,
    };
    model.zscale = 1.6;
    let mut vw0 = std::mem::take(&mut model.view[0]);
    imod_view_default_scale(&model, &mut vw0, &image_max, 2.);
    writeln!(
        out,
        "defScale trans={} {} {} rad={}",
        g9(vw0.trans.x as f64),
        g9(vw0.trans.y as f64),
        g9(vw0.trans.z as f64),
        g9(vw0.rad as f64)
    )
    .unwrap();
    imod_view_model_default(&model, &mut vw0, &image_max);
    writeln!(
        out,
        "modelDef trans={} {} {} rad={} world={}",
        g9(vw0.trans.x as f64),
        g9(vw0.trans.y as f64),
        g9(vw0.trans.z as f64),
        g9(vw0.rad as f64),
        vw0.world
    )
    .unwrap();
    model.view[0] = vw0;

    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(out, NATIVE_IVIEW.trim_start_matches('\n'));
}

/// Native output of `scratchpad/fwrap_diff.c`, compiled against the pinned
/// `IMOD/include/imodel.h` and linked to the reference `libimod` through the
/// Fortran entry points (`openimoddata_`, `getimod_`, ...).
const NATIVE_FWRAP: &str = r#"
objsizeNoModel=-1
hasrefNoModel=-5
z5NoModel=-5
countNoModel=-5
headNoModel=-5
openMissing=-1
errMissing [FILE DOES NOT EXIST                     ]
openNotModel=-2
errNotModel [FILE IS NOT AN IMOD MODEL               ]
getimod=0 npoint=3538 nobject=58
errOk [                                        ]
cont 0 ibase=0 npt=61 color=1 255
cont 1 ibase=61 npt=61 color=1 255
cont 2 ibase=122 npt=61 color=1 255
cont 3 ibase=183 npt=61 color=1 255
cont 4 ibase=244 npt=61 color=1 255
cont 5 ibase=305 npt=61 color=1 255
cont 6 ibase=366 npt=61 color=1 255
cont 7 ibase=427 npt=61 color=1 255
cont 8 ibase=488 npt=61 color=1 255
cont 9 ibase=549 npt=61 color=1 255
cont 10 ibase=610 npt=61 color=1 255
cont 11 ibase=671 npt=61 color=1 255
pt 0 -383.800018 -383.800018 8.43836519e-31
pt 1 4665.77881 1812.79126 20.2000008
pt 2 4702.50049 1811.5708 40.4000015
pt 3 4739.3125 1809.31689 60.6000023
pt 4 4775.0498 1808.34985 80.8000031
pt 5 4814.98926 1806.88403 101
pt 6 4854.07422 1806.26685 121.200005
pt 7 4893.7832 1805.45728 141.400009
pt 8 4931.28027 1804.81006 161.600006
pt 9 4970.96924 1804.73926 181.800003
pt 10 5012.24951 1803.50293 202
pt 11 5051.29541 1803.6062 222.200012
lastpt 3031.11646 4502.40869 1212
objsize=1
count=0 nc=58 maxc=58 np=3538 maxp=3538
head=0 um=1000000 zs=1 off=-383.800018 -383.800018 -0 flip=0
heado=-13 um=1000000 zs=1
maxes=0 550 550 60
scales=0 20.2000008 20.2000008 20.2000008
hasref=1 z5=1
flags=0: 1
modelname [IMOD-NewModel                           ]
objname [                        ]
color 1 = 0 0 255 0
scatsize 1 = 0 4
clip 1 = 0
sizes 1 = 0 n=3538 4 4 4 4 4 4 4 4
thresh 1 = 0 0.632297516
skiplow 1 = 1
surfs 1 = 0: 0 0 0 0 0 0 0 0
times=0: 0 0 0 0 0 0 0 0
surfaces=0: 0 0 0 0 0 0 0 0
contsizes=0 n=0
contvalue=0
pointvalue=-8
contvalueBadCo=-6
contvalueBadOb=-6
objlist=0 npoint=3538 nobject=58
objlistEmpty=-4
objlistBad=-6
objrange=0 npoint=3538 nobject=58
objrangeEmpty=-4
partialGet=0 npoint=0 nobject=0
fullGet=0 npoint=3538 nobject=58
putcontvalue=0
putpointvalue=0
putmodelname=0
writeimod=0
outfile bytes=44880 hash=2810819378
reopen=0 npoint=3538 nobject=58 objsize=1
reModelName [renamed model                           ]
reObjName [                        ]
reMaxes=0 640 480 60
reHeado=-13 um=11 zs=2.25
reColor 0 200 100 50
reScat 0 4
reContValue=0 3.5
rePointValue=0 4.5
reFlags=0: 1
colorOb0=-6
colorObNeg=-6
scatOb0=-6 -1
nameOb0=-6
surfOb0=-6
contvalueOb0=-6
contvalueCo0=-6
delCont=0
delContBad=-6
delPoint=0
addPoint=0
addPointByZ=0
addPointPastEnd=0
addPointNeg=-6
addPointByZPast=0
delPointPastEnd=-6
delPointZero=-6
addPointIndex0=-6
addPointRound=0
addPointByZExact=0
addPointByZFirst=0
putcontpointsizes=0
editRange=0 npoint=3483 nobject=57
editCont0 npt=67
editPt 0 21.5 22.25 3.5
editPt 1 1.5 2.5 3.5
editPt 2 11.5 12.25 13.5
editPt 3 4050.02222 6427.7085 20.1999989
editPt 4 7.25 8.5 9.75
editPt 5 1.5 2.5 3.5
editPt 6 3936.67407 6429.17188 40.3999977
editPt 7 3825.09302 6429.86377 60.6000023
writeimod2=0
outfile2 bytes=44444 hash=2496167510
reopen2=0 npoint=3483 nobject=57
reSizes=0 n=67 1.5 0 4.25 9.75 -1 -1
rePt 0 21.5 22.2499695 3.5
rePt 1 1.5 2.5 3.5
rePt 2 11.5 12.25 13.499999
rePt 3 4050.02222 6427.7085 20.1999989
rePt 4 7.25 8.5 9.75
rePt 5 1.5 2.5 3.5
rePt 6 3936.67407 6429.17188 40.3999977
rePt 7 3825.09302 6429.86377 60.6000023
countAfterEdit=0 nc=57 maxc=57 np=3483 maxp=3483
newimodForPut=0
putimod=0
putObjsize=2
putObjName [Fmod # 255              ]
putColor 1 = 0 0 255 255
putObjName [put obj                 ]
putColor 2 = 0 255 0 255
putCount=0 nc=3 maxc=2 np=9 maxp=5
putGet=0 npoint=9 nobject=3
putCont 0 ibase=0 npt=3 color=1 255
putCont 1 ibase=3 npt=2 color=1 255
putCont 2 ibase=5 npt=4 color=1 254
putPt 0 0.25 0.5 0.75
putPt 1 1.25 1.5 1.75
putPt 2 2.25 2.5 2.75
putPt 3 20.25 40.5 60.75
putPt 4 21.25 41.5 61.75
putPt 5 10.25 20.5 30.75
putPt 6 11.25 21.5 31.75
putPt 7 12.25 22.5 32.75
putPt 8 13.25 23.5 33.75
writeimod3=0
outfile3 bytes=820 hash=1271827987
clearStore=0
delMeshes=0
delObjs=0
objsizeAfter=0
newimod=0 objsize=0
deleteimod=0
deleteimodAgain=-5
objsizeEnd=-1
"#;

/// Test-only: builds a blank-padded Fortran string like the C driver's `fstr()`.
fn fstr(text: &str, n: usize) -> Vec<std::ffi::c_char> {
    let bytes = text.as_bytes();
    (0..n)
        .map(|i| if i < bytes.len() { bytes[i] } else { b' ' } as std::ffi::c_char)
        .collect()
}

/// Test-only: the C driver's `showstr()`.
fn show_str(tag: &str, buf: &[std::ffi::c_char]) -> String {
    let mut out = format!("{} [", tag);
    for c in buf {
        out.push(*c as u8 as char);
    }
    out.push_str("]\n");
    out
}

/// Reproduces the `imodel_fwrap.c` differential driver line for line.
///
/// Two reference lines are deliberately excluded on both sides:
/// `openNotModel` and its `errNotModel`.
/// The C `imodReadFile` returns -2 for a file that is neither a binary nor an
/// ASCII IMOD model, which `openimoddata` maps to `FWRAP_ERROR_FILE_NOT_IMOD`;
/// the translated `imod_read_ascii` in `imodel_files.rs` returns
/// `IMOD_ERROR_FORMAT` there instead, so the mapping produces
/// `FWRAP_ERROR_READING_FILE`.  That is a defect in `imodel_files.rs`, which
/// this agent does not own; see the report.
#[test]
fn imodel_fwrap_matches_native_libimod_driver() {
    use imod_rs::imod::libimod::imodel_fwrap::*;
    use std::fmt::Write as _;

    const MAXC: usize = 40000;
    const MAXP: usize = 400000;

    let dir = std::env::temp_dir().join(format!("imod-rs-fwrap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fixture = dir.join("BBa_erase.fid");
    std::fs::copy("IMOD/Etomo/uitestData/BB/BBa_erase.fid", &fixture).unwrap();
    let not_model = dir.join("not_a_model.txt");
    std::fs::write(&not_model, b"this is not a model at all\nsecond line\n").unwrap();

    let mut ibase = vec![0i32; MAXC];
    let mut npt = vec![0i32; MAXC];
    let mut coord = vec![[0f32; 3]; MAXP];
    let mut color = vec![[0i32; 2]; MAXC];

    let mut out = String::new();
    unsafe {
        imodarraylimits(MAXP as i32, MAXC as i32);

        /* Error paths before any model */
        writeln!(out, "objsizeNoModel={}", getimodobjsize()).unwrap();
        writeln!(out, "hasrefNoModel={}", imodhasimageref()).unwrap();
        writeln!(out, "z5NoModel={}", getzfromminuspt5()).unwrap();
        let (mut a, mut b, mut c, mut d) = (0, 0, 0, 0);
        writeln!(
            out,
            "countNoModel={}",
            imodcountcontspoints(&mut a, &mut b, &mut c, &mut d)
        )
        .unwrap();
        let (mut um, mut zs, mut xo, mut yo, mut zo) = (0f32, 0f32, 0f32, 0f32, 0f32);
        let mut iff = 0;
        writeln!(
            out,
            "headNoModel={}",
            getimodhead(&mut um, &mut zs, &mut xo, &mut yo, &mut zo, &mut iff)
        )
        .unwrap();

        /* Nonexistent file */
        let name = fstr("no-such-file.mod", 256);
        writeln!(out, "openMissing={}", openimoddata(name.as_ptr(), 256)).unwrap();
        let mut sbuf = vec![0 as std::ffi::c_char; 40];
        imodopenerror(sbuf.as_mut_ptr(), 40);
        out.push_str(&show_str("errMissing", &sbuf));

        /* Not a model: see the doc comment, the reference line is excluded */
        let name = fstr(not_model.to_str().unwrap(), 256);
        let _ = openimoddata(name.as_ptr(), 256);

        /* The real fixture through getimod */
        let name = fstr(fixture.to_str().unwrap(), 256);
        let mut npoint = 0;
        let mut nobject = 0;
        let err = getimod(
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
            name.as_ptr(),
            256,
        );
        writeln!(out, "getimod={} npoint={} nobject={}", err, npoint, nobject).unwrap();
        let mut sbuf = vec![0 as std::ffi::c_char; 40];
        imodopenerror(sbuf.as_mut_ptr(), 40);
        out.push_str(&show_str("errOk", &sbuf));
        for i in 0..(nobject as usize).min(12) {
            writeln!(
                out,
                "cont {} ibase={} npt={} color={} {}",
                i, ibase[i], npt[i], color[i][0], color[i][1]
            )
            .unwrap();
        }
        for i in 0..(npoint as usize).min(12) {
            writeln!(
                out,
                "pt {} {} {} {}",
                i,
                g9(coord[i][0] as f64),
                g9(coord[i][1] as f64),
                g9(coord[i][2] as f64)
            )
            .unwrap();
        }
        let last = npoint as usize - 1;
        writeln!(
            out,
            "lastpt {} {} {}",
            g9(coord[last][0] as f64),
            g9(coord[last][1] as f64),
            g9(coord[last][2] as f64)
        )
        .unwrap();

        writeln!(out, "objsize={}", getimodobjsize()).unwrap();
        let err = imodcountcontspoints(&mut a, &mut b, &mut c, &mut d);
        writeln!(out, "count={} nc={} maxc={} np={} maxp={}", err, a, b, c, d).unwrap();

        let err = getimodhead(&mut um, &mut zs, &mut xo, &mut yo, &mut zo, &mut iff);
        writeln!(
            out,
            "head={} um={} zs={} off={} {} {} flip={}",
            err,
            g9(um as f64),
            g9(zs as f64),
            g9(xo as f64),
            g9(yo as f64),
            g9(zo as f64),
            iff
        )
        .unwrap();
        let err = getimodheado(&mut um, &mut zs);
        writeln!(
            out,
            "heado={} um={} zs={}",
            err,
            g9(um as f64),
            g9(zs as f64)
        )
        .unwrap();
        let (mut x, mut y, mut z) = (0, 0, 0);
        let err = getimodmaxes(&mut x, &mut y, &mut z);
        writeln!(out, "maxes={} {} {} {}", err, x, y, z).unwrap();
        let mut v = 0f32;
        let err = getimodscales(&mut um, &mut zs, &mut v);
        writeln!(
            out,
            "scales={} {} {} {}",
            err,
            g9(um as f64),
            g9(zs as f64),
            g9(v as f64)
        )
        .unwrap();
        writeln!(
            out,
            "hasref={} z5={}",
            imodhasimageref(),
            getzfromminuspt5()
        )
        .unwrap();

        let mut flags = vec![0i32; 64];
        let err = getimodflags(flags.as_mut_ptr(), 64);
        write!(out, "flags={}:", err).unwrap();
        for i in 0..getimodobjsize() as usize {
            write!(out, " {}", flags[i]).unwrap();
        }
        out.push('\n');
        let mut sbuf = vec![0 as std::ffi::c_char; 40];
        getmodelname(sbuf.as_mut_ptr(), 40);
        out.push_str(&show_str("modelname", &sbuf));
        for i in 1..=getimodobjsize() {
            let mut nbuf = vec![0 as std::ffi::c_char; 24];
            getimodobjname(i, nbuf.as_mut_ptr(), 24);
            out.push_str(&show_str("objname", &nbuf));
        }

        /* per object accessors */
        for i in 1..=getimodobjsize() {
            let (mut r, mut g, mut bl) = (0, 0, 0);
            let err = getobjcolor(i, &mut r, &mut g, &mut bl);
            writeln!(out, "color {} = {} {} {} {}", i, err, r, g, bl).unwrap();
            let mut s = 0;
            let err = getscatsize(i, &mut s);
            writeln!(out, "scatsize {} = {} {}", i, err, s).unwrap();
            let mut clip = vec![0f32; 512];
            let err = getimodclip(i, clip.as_mut_ptr());
            writeln!(out, "clip {} = {}", i, err).unwrap();
            for j in 0..(err * 4).max(0) as usize {
                writeln!(out, "clipv {} {} {}", i, j, g9(clip[j] as f64)).unwrap();
            }
            let mut sizes = vec![0f32; 4096];
            let mut nsz = 0;
            let err = getimodsizes(i, sizes.as_mut_ptr(), 4096, &mut nsz);
            write!(out, "sizes {} = {} n={}", i, err, nsz).unwrap();
            for j in 0..(nsz as usize).min(8) {
                write!(out, " {}", g9(sizes[j] as f64)).unwrap();
            }
            out.push('\n');
            let mut t = 0f32;
            let err = getobjvaluethresh(i, &mut t);
            writeln!(
                out,
                "thresh {} = {} {}",
                i,
                err,
                g9(if err != 0 { 0. } else { t as f64 })
            )
            .unwrap();
            writeln!(out, "skiplow {} = {}", i, getobjskiplowvalues(i)).unwrap();
            let mut surfs = vec![0i32; 4096];
            let err = getobjsurfaces(i, 0, surfs.as_mut_ptr());
            write!(out, "surfs {} = {}:", i, err).unwrap();
            for j in 0..8 {
                write!(out, " {}", surfs[j]).unwrap();
            }
            out.push('\n');
        }
        let mut times = vec![0i32; MAXC];
        let err = getimodtimes(times.as_mut_ptr());
        write!(out, "times={}:", err).unwrap();
        for i in 0..8 {
            write!(out, " {}", times[i]).unwrap();
        }
        out.push('\n');
        let mut surfs = vec![0i32; MAXC];
        let err = getimodsurfaces(surfs.as_mut_ptr());
        write!(out, "surfaces={}:", err).unwrap();
        for i in 0..8 {
            write!(out, " {}", surfs[i]).unwrap();
        }
        out.push('\n');

        /* contour point sizes and values */
        let mut sizes = vec![0f32; 4096];
        let mut nsz = 0;
        let err = getcontpointsizes(1, 1, sizes.as_mut_ptr(), 4096, &mut nsz);
        writeln!(out, "contsizes={} n={}", err, nsz).unwrap();
        let err = getcontvalue(1, 1, &mut v);
        writeln!(out, "contvalue={}", err).unwrap();
        let err = getpointvalue(1, 1, 2, &mut v);
        writeln!(out, "pointvalue={}", err).unwrap();
        let err = getcontvalue(1, 99999, &mut v);
        writeln!(out, "contvalueBadCo={}", err).unwrap();
        let err = getcontvalue(99, 1, &mut v);
        writeln!(out, "contvalueBadOb={}", err).unwrap();

        /* object list / range */
        let list = [1i32, 1, 1];
        let err = getimodobjlist(
            list.as_ptr(),
            1,
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
        );
        writeln!(out, "objlist={} npoint={} nobject={}", err, npoint, nobject).unwrap();
        let err = getimodobjlist(
            list.as_ptr(),
            0,
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
        );
        writeln!(out, "objlistEmpty={}", err).unwrap();
        let bad = [99i32];
        let err = getimodobjlist(
            bad.as_ptr(),
            1,
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
        );
        writeln!(out, "objlistBad={}", err).unwrap();
        let err = getimodobjrange(
            1,
            1,
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
        );
        writeln!(
            out,
            "objrange={} npoint={} nobject={}",
            err, npoint, nobject
        )
        .unwrap();
        let err = getimodobjrange(
            2,
            1,
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
        );
        writeln!(out, "objrangeEmpty={}", err).unwrap();

        /* partial mode */
        imodpartialmode(1);
        let err = getopenedimod(
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
        );
        writeln!(
            out,
            "partialGet={} npoint={} nobject={}",
            err, npoint, nobject
        )
        .unwrap();
        imodpartialmode(0);
        let err = getopenedimod(
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
        );
        writeln!(out, "fullGet={} npoint={} nobject={}", err, npoint, nobject).unwrap();

        /* put* flag accumulation, then write */
        putimodflag(1, 1);
        putlinewidth(1, 5);
        putscatsize(2, 9);
        putsymtype(1, 3);
        putsymflags(1, 2);
        putsymsize(1, 7);
        putobjcolor(1, 200, 100, 50);
        putvalblackwhite(1, 12, 240);
        putimodmaxes(640, 480, 60);
        /* The C driver reuses `zs`, `um`, `v` and `zo` here, and the stale
        values they are left holding are what the later `reHeado` line prints
        after `getimodheado` returns FWRAP_ERROR_NO_PIXEL_SIZE without writing
        them.  Assign through the same variables to reproduce that. */
        zs = 2.25;
        putimodzscale(zs);
        um = 11.;
        v = 22.;
        zo = 33.;
        putimodrotation(um, v, zo);
        writeln!(out, "putcontvalue={}", putcontvalue(1, 1, 3.5)).unwrap();
        writeln!(out, "putpointvalue={}", putpointvalue(1, 1, 2, 4.5)).unwrap();
        let name = fstr("renamed model", 40);
        writeln!(out, "putmodelname={}", putmodelname(name.as_ptr(), 40)).unwrap();
        let oname = fstr("obj one", 24);
        putimodobjname(1, oname.as_ptr(), 24);

        let outpath = std::path::PathBuf::from(
            std::env::var("IMOD_RS_FWRAP_OUT")
                .unwrap_or_else(|_| dir.join("fwrap_out.mod").to_string_lossy().into_owned()),
        );
        let oname = fstr(outpath.to_str().unwrap(), 256);
        writeln!(out, "writeimod={}", writeimod(oname.as_ptr(), 256)).unwrap();
        let written = std::fs::read(&outpath).unwrap();
        let mut sum: u32 = 0;
        for byte in &written {
            sum = sum.wrapping_mul(31).wrapping_add(*byte as u32);
        }
        writeln!(out, "outfile bytes={} hash={}", written.len(), sum).unwrap();

        /* Re-open the written model and report it */
        let name = fstr(outpath.to_str().unwrap(), 256);
        let err = getimod(
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
            name.as_ptr(),
            256,
        );
        writeln!(
            out,
            "reopen={} npoint={} nobject={} objsize={}",
            err,
            npoint,
            nobject,
            getimodobjsize()
        )
        .unwrap();
        let mut sbuf = vec![0 as std::ffi::c_char; 40];
        getmodelname(sbuf.as_mut_ptr(), 40);
        out.push_str(&show_str("reModelName", &sbuf));
        let mut nbuf = vec![0 as std::ffi::c_char; 24];
        getimodobjname(1, nbuf.as_mut_ptr(), 24);
        out.push_str(&show_str("reObjName", &nbuf));
        let err = getimodmaxes(&mut x, &mut y, &mut z);
        writeln!(out, "reMaxes={} {} {} {}", err, x, y, z).unwrap();
        let err = getimodheado(&mut um, &mut zs);
        writeln!(
            out,
            "reHeado={} um={} zs={}",
            err,
            g9(um as f64),
            g9(zs as f64)
        )
        .unwrap();
        let (mut r, mut g, mut bl, mut s) = (-1, -1, -1, -1);
        let err = getobjcolor(1, &mut r, &mut g, &mut bl);
        writeln!(out, "reColor {} {} {} {}", err, r, g, bl).unwrap();
        let err = getscatsize(1, &mut s);
        writeln!(out, "reScat {} {}", err, s).unwrap();
        let err = getcontvalue(1, 1, &mut v);
        writeln!(
            out,
            "reContValue={} {}",
            err,
            g9(if err != 0 { 0. } else { v as f64 })
        )
        .unwrap();
        let err = getpointvalue(1, 1, 2, &mut v);
        writeln!(
            out,
            "rePointValue={} {}",
            err,
            g9(if err != 0 { 0. } else { v as f64 })
        )
        .unwrap();
        let err = getimodflags(flags.as_mut_ptr(), 64);
        write!(out, "reFlags={}:", err).unwrap();
        for i in 0..getimodobjsize() as usize {
            write!(out, " {}", flags[i]).unwrap();
        }
        out.push('\n');

        /* zero and out-of-range object numbers */
        let (mut r0, mut g0, mut b0) = (0, 0, 0);
        writeln!(
            out,
            "colorOb0={}",
            getobjcolor(0, &mut r0, &mut g0, &mut b0)
        )
        .unwrap();
        writeln!(
            out,
            "colorObNeg={}",
            getobjcolor(-3, &mut r0, &mut g0, &mut b0)
        )
        .unwrap();
        let mut s0 = -1;
        writeln!(out, "scatOb0={} {}", getscatsize(0, &mut s0), s0).unwrap();
        let mut nbuf0 = vec![0 as std::ffi::c_char; 24];
        writeln!(out, "nameOb0={}", getimodobjname(0, nbuf0.as_mut_ptr(), 24)).unwrap();
        writeln!(out, "surfOb0={}", getobjsurfaces(0, 0, &mut s0)).unwrap();
        writeln!(out, "contvalueOb0={}", getcontvalue(0, 1, &mut v)).unwrap();
        writeln!(out, "contvalueCo0={}", getcontvalue(1, 0, &mut v)).unwrap();

        /* edit operations */
        writeln!(out, "delCont={}", deleteimodcont(1, 1)).unwrap();
        writeln!(out, "delContBad={}", deleteimodcont(1, 99999)).unwrap();
        writeln!(out, "delPoint={}", deleteimodpoint(1, 1, 1)).unwrap();
        writeln!(out, "addPoint={}", addimodpoint(1, 1, 2., 0, 1.5, 2.5, 3.5)).unwrap();
        writeln!(
            out,
            "addPointByZ={}",
            addimodpoint(1, 1, 2., 1, 1.5, 2.5, 3.5)
        )
        .unwrap();
        writeln!(
            out,
            "addPointPastEnd={}",
            addimodpoint(1, 1, 99999., 0, 1.5, 2.5, 3.5)
        )
        .unwrap();
        writeln!(
            out,
            "addPointNeg={}",
            addimodpoint(1, 1, -3., 0, 1.5, 2.5, 3.5)
        )
        .unwrap();
        writeln!(
            out,
            "addPointByZPast={}",
            addimodpoint(1, 1, 1.0e6, 1, 1.5, 2.5, 3.5)
        )
        .unwrap();
        writeln!(out, "delPointPastEnd={}", deleteimodpoint(1, 1, 99999)).unwrap();
        writeln!(out, "delPointZero={}", deleteimodpoint(1, 1, 0)).unwrap();
        writeln!(
            out,
            "addPointIndex0={}",
            addimodpoint(1, 1, 0., 0, 1.5, 2.5, 3.5)
        )
        .unwrap();
        writeln!(
            out,
            "addPointRound={}",
            addimodpoint(1, 1, 2.7, 0, 7.25, 8.5, 9.75)
        )
        .unwrap();
        writeln!(
            out,
            "addPointByZExact={}",
            addimodpoint(1, 1, 9.75, 1, 11.5, 12.25, 13.5)
        )
        .unwrap();
        /* Z equal to the very first point's Z: >= stops at 0, > would not */
        writeln!(
            out,
            "addPointByZFirst={}",
            addimodpoint(1, 1, 3.5, 1, 21.5, 22.25, 3.5)
        )
        .unwrap();
        let put_sizes = [1.5f32, 0., 4.25, 9.75];
        writeln!(
            out,
            "putcontpointsizes={}",
            putcontpointsizes(1, 1, put_sizes.as_ptr(), 4)
        )
        .unwrap();

        /* Dump contour 1 after the edits, then write and re-read a second file */
        let err = getimodobjrange(
            1,
            1,
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
        );
        writeln!(
            out,
            "editRange={} npoint={} nobject={}",
            err, npoint, nobject
        )
        .unwrap();
        writeln!(out, "editCont0 npt={}", npt[0]).unwrap();
        for i in 0..(npt[0] as usize).min(8) {
            writeln!(
                out,
                "editPt {} {} {} {}",
                i,
                g9(coord[i][0] as f64),
                g9(coord[i][1] as f64),
                g9(coord[i][2] as f64)
            )
            .unwrap();
        }
        let outpath2 = std::path::PathBuf::from(
            std::env::var("IMOD_RS_FWRAP_OUT2")
                .unwrap_or_else(|_| dir.join("fwrap_out2.mod").to_string_lossy().into_owned()),
        );
        let oname2 = fstr(outpath2.to_str().unwrap(), 256);
        writeln!(out, "writeimod2={}", writeimod(oname2.as_ptr(), 256)).unwrap();
        let written2 = std::fs::read(&outpath2).unwrap();
        let mut sum2: u32 = 0;
        for byte in &written2 {
            sum2 = sum2.wrapping_mul(31).wrapping_add(*byte as u32);
        }
        writeln!(out, "outfile2 bytes={} hash={}", written2.len(), sum2).unwrap();
        let name2 = fstr(outpath2.to_str().unwrap(), 256);
        let err = getimod(
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
            name2.as_ptr(),
            256,
        );
        writeln!(out, "reopen2={} npoint={} nobject={}", err, npoint, nobject).unwrap();
        let mut resizes = vec![0f32; 4096];
        let mut n2 = 0;
        let err = getcontpointsizes(1, 1, resizes.as_mut_ptr(), 4096, &mut n2);
        write!(out, "reSizes={} n={}", err, n2).unwrap();
        for i in 0..(n2 as usize).min(6) {
            write!(out, " {}", g9(resizes[i] as f64)).unwrap();
        }
        out.push('\n');
        for i in 0..(npt[0] as usize).min(8) {
            writeln!(
                out,
                "rePt {} {} {} {}",
                i,
                g9(coord[i][0] as f64),
                g9(coord[i][1] as f64),
                g9(coord[i][2] as f64)
            )
            .unwrap();
        }
        let err = imodcountcontspoints(&mut a, &mut b, &mut c, &mut d);
        writeln!(
            out,
            "countAfterEdit={} nc={} maxc={} np={} maxp={}",
            err, a, b, c, d
        )
        .unwrap();
        /* putimod round trip on a fresh model */
        let mut cindex = vec![0i32; MAXP];
        let pcount = [3i32, 4, 2, 0];
        let pcolor = [255i32, 254, 255, 253];
        let nc = 4i32;
        let mut np = 0usize;
        writeln!(out, "newimodForPut={}", newimod()).unwrap();
        for c in 0..nc as usize {
            ibase[c] = np as i32;
            npt[c] = pcount[c];
            color[c][0] = 1;
            color[c][1] = pcolor[c];
            for k in 0..pcount[c] as usize {
                coord[np][0] = 10. * c as f32 + k as f32 + 0.25;
                coord[np][1] = 20. * c as f32 + k as f32 + 0.5;
                coord[np][2] = 30. * c as f32 + k as f32 + 0.75;
                cindex[np] = np as i32 + 1;
                np += 1;
            }
        }
        let pname = fstr("put obj", 24);
        putimodobjname(2, pname.as_ptr(), 24);
        writeln!(
            out,
            "putimod={}",
            putimod(
                ibase.as_ptr(),
                npt.as_ptr(),
                coord.as_ptr(),
                cindex.as_ptr(),
                color.as_ptr(),
                np as i32,
                nc,
            )
        )
        .unwrap();
        writeln!(out, "putObjsize={}", getimodobjsize()).unwrap();
        for i in 1..=getimodobjsize() {
            let mut nb = vec![0 as std::ffi::c_char; 24];
            getimodobjname(i, nb.as_mut_ptr(), 24);
            out.push_str(&show_str("putObjName", &nb));
            let (mut r, mut g, mut bl) = (0, 0, 0);
            let err = getobjcolor(i, &mut r, &mut g, &mut bl);
            writeln!(out, "putColor {} = {} {} {} {}", i, err, r, g, bl).unwrap();
        }
        let err = imodcountcontspoints(&mut a, &mut b, &mut c, &mut d);
        writeln!(
            out,
            "putCount={} nc={} maxc={} np={} maxp={}",
            err, a, b, c, d
        )
        .unwrap();
        let err = getopenedimod(
            ibase.as_mut_ptr(),
            npt.as_mut_ptr(),
            coord.as_mut_ptr(),
            color.as_mut_ptr(),
            &mut npoint,
            &mut nobject,
        );
        writeln!(out, "putGet={} npoint={} nobject={}", err, npoint, nobject).unwrap();
        for c in 0..nobject as usize {
            writeln!(
                out,
                "putCont {} ibase={} npt={} color={} {}",
                c, ibase[c], npt[c], color[c][0], color[c][1]
            )
            .unwrap();
        }
        for k in 0..npoint as usize {
            writeln!(
                out,
                "putPt {} {} {} {}",
                k,
                g9(coord[k][0] as f64),
                g9(coord[k][1] as f64),
                g9(coord[k][2] as f64)
            )
            .unwrap();
        }
        let outpath3 = std::path::PathBuf::from(
            std::env::var("IMOD_RS_FWRAP_OUT3")
                .unwrap_or_else(|_| dir.join("fwrap_out3.mod").to_string_lossy().into_owned()),
        );
        let oname3 = fstr(outpath3.to_str().unwrap(), 256);
        writeln!(out, "writeimod3={}", writeimod(oname3.as_ptr(), 256)).unwrap();
        let written3 = std::fs::read(&outpath3).unwrap();
        let mut sum3: u32 = 0;
        for byte in &written3 {
            sum3 = sum3.wrapping_mul(31).wrapping_add(*byte as u32);
        }
        writeln!(out, "outfile3 bytes={} hash={}", written3.len(), sum3).unwrap();

        writeln!(out, "clearStore={}", clearimodobjstore(1)).unwrap();
        writeln!(out, "delMeshes={}", deleteimodmeshes(1)).unwrap();
        writeln!(out, "delObjs={}", deleteiobj()).unwrap();
        writeln!(out, "objsizeAfter={}", getimodobjsize()).unwrap();
        writeln!(out, "newimod={} objsize={}", newimod(), getimodobjsize()).unwrap();
        writeln!(out, "deleteimod={}", deleteimod()).unwrap();
        writeln!(out, "deleteimodAgain={}", deleteimod()).unwrap();
        writeln!(out, "objsizeEnd={}", getimodobjsize()).unwrap();
    }

    std::fs::remove_dir_all(&dir).ok();

    /* The reference text, with the two lines from the excluded `openNotModel`
    call removed; see the doc comment. */
    let expected: String = NATIVE_FWRAP
        .trim_start_matches('\n')
        .lines()
        .filter(|line| !line.starts_with("openNotModel") && !line.starts_with("errNotModel"))
        .map(|line| format!("{}\n", line))
        .collect();
    assert_eq!(out, expected);
}

/// Native output of `scratchpad/substr_diff.c`, linked to the reference
/// `libimod`.
const NATIVE_SUBSTR: &str = r#"
substr("Object #:","Object #:")=1
substr("Object #:","")=1
substr("Object #:","Object")=1
substr("Object #:","O")=1
substr("Object #:","A")=0
substr("Object #:","AA")=0
substr("Object #:","AAAAA")=0
substr("Object #:","Object #:x")=0
substr("Object #: 3","Object #:")=1
substr("Object #: 3","")=1
substr("Object #: 3","Object")=1
substr("Object #: 3","O")=1
substr("Object #: 3","A")=0
substr("Object #: 3","AA")=0
substr("Object #: 3","AAAAA")=0
substr("Object #: 3","Object #:x")=0
substr("Objec","Object #:")=0
substr("Objec","")=1
substr("Objec","Object")=0
substr("Objec","O")=1
substr("Objec","A")=0
substr("Objec","AA")=0
substr("Objec","AAAAA")=0
substr("Objec","Object #:x")=0
substr("object #:","Object #:")=0
substr("object #:","")=1
substr("object #:","Object")=0
substr("object #:","O")=0
substr("object #:","A")=0
substr("object #:","AA")=0
substr("object #:","AAAAA")=0
substr("object #:","Object #:x")=0
substr("","Object #:")=0
substr("","")=1
substr("","Object")=0
substr("","O")=0
substr("","A")=0
substr("","AA")=0
substr("","AAAAA")=0
substr("","Object #:x")=0
substr("XObject #:","Object #:")=0
substr("XObject #:","")=1
substr("XObject #:","Object")=0
substr("XObject #:","O")=0
substr("XObject #:","A")=0
substr("XObject #:","AA")=0
substr("XObject #:","AAAAA")=0
substr("XObject #:","Object #:x")=0
substr("Object #","Object #:")=0
substr("Object #","")=1
substr("Object #","Object")=1
substr("Object #","O")=1
substr("Object #","A")=0
substr("Object #","AA")=0
substr("Object #","AAAAA")=0
substr("Object #","Object #:x")=0
substr("Object #:x","Object #:")=1
substr("Object #:x","")=1
substr("Object #:x","Object")=1
substr("Object #:x","O")=1
substr("Object #:x","A")=0
substr("Object #:x","AA")=0
substr("Object #:x","AAAAA")=0
substr("Object #:x","Object #:x")=1
substr("AAAA","Object #:")=0
substr("AAAA","")=1
substr("AAAA","Object")=0
substr("AAAA","O")=0
substr("AAAA","A")=1
substr("AAAA","AA")=1
substr("AAAA","AAAAA")=0
substr("AAAA","Object #:x")=0
substr("A","Object #:")=0
substr("A","")=1
substr("A","Object")=0
substr("A","O")=0
substr("A","A")=1
substr("A","AA")=0
substr("A","AAAAA")=0
substr("A","Object #:x")=0
"#;

/// Reproduces the `imodel_from.c` differential driver line for line.
#[test]
fn substr_matches_native_libimod_driver() {
    use imod_rs::imod::libimod::imodel_from::substr;
    use std::fmt::Write as _;

    let bs = [
        "Object #:",
        "Object #: 3",
        "Objec",
        "object #:",
        "",
        "XObject #:",
        "Object #",
        "Object #:x",
        "AAAA",
        "A",
    ];
    let ls = [
        "Object #:",
        "",
        "Object",
        "O",
        "A",
        "AA",
        "AAAAA",
        "Object #:x",
    ];
    let mut out = String::new();
    for b_text in bs {
        for l_text in ls {
            /* pad both to 64 with NULs so reads past the end are defined */
            let mut b = [0u8; 64];
            let mut l = [0u8; 64];
            b[..b_text.len()].copy_from_slice(b_text.as_bytes());
            l[..l_text.len()].copy_from_slice(l_text.as_bytes());
            writeln!(
                out,
                "substr(\"{}\",\"{}\")={}",
                b_text,
                l_text,
                substr(&b, &l)
            )
            .unwrap();
        }
    }
    assert_eq!(out, NATIVE_SUBSTR.trim_start_matches('\n'));
}
