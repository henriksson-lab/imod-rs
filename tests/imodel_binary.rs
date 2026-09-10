//! Binary-model coverage for source-shaped `imodel.c` operations.

use imod_rs::imod::libimod::imodel::{
    ICONT_OPEN, IMODF_FLIPYZ, Icont, Imesh, Imod, Iobj, Ipoint, imod_clean_surf, imod_delete_point,
    imod_get_bounding_box, imod_get_cur_mesh_surf, imod_get_filename, imod_get_flipped,
    imod_get_max_object, imod_get_max_time, imod_get_pixel_size, imod_get_z_scale,
    imod_insert_point, imod_new_contour, imod_new_point, imod_set_cur_mesh_surf, imodel_maxpt,
    imodel_minpt,
};
use imod_rs::imod::libimod::imodel_files::{imod_file_write, imod_read};
use imod_rs::imod::libimod::iobj::{
    imod_object_checksum, imod_object_copy, imod_object_copy_clear, imod_object_default,
    imod_object_delete, imod_object_dup, imod_object_new, imod_objects_delete, imod_objects_new,
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
            name: "accessor-model".into(),
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
    assert_eq!(imod_get_filename(&model), "accessor-model");
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
                name: "source".into(),
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
