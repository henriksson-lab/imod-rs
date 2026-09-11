//! Conformance for the model chunks no vendored fixture carries: object and
//! contour labels (`OLBL`/`LABL`), the object clip-plane chunk (`CLIP`), and a
//! `VIEW` chunk with object views and a model clip chunk (`MCLP`).
//!
//! `fixtures/model-view-clip-label.mod` is authored by
//! `fixtures/make-model-view-clip-label.py` straight from `imodel_files.c` and
//! `iview.c`, so it is independent of both implementations' writers.  The
//! golden text files are the reference `imodinfo`'s own output for it.

use std::process::Command;

use imod_rs::imod::libimod::imodel_files::{imod_file_write, imod_read};

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/model-view-clip-label.mod")
}

#[test]
fn reads_every_chunk_the_authored_model_carries() {
    let model = imod_read(fixture()).expect("authored model must decode");

    assert_eq!(model.obj.len(), 1);
    let obj = &model.obj[0];

    // OLBL: the object label, two items on surfaces 1 and 2.
    let label = obj.label.as_ref().expect("object label chunk must be read");
    // `imodLabelRead` (`ilabel.c:449`) reads the padded length the writer put
    // out, so the name keeps its NUL padding.
    assert_eq!(label.name.as_deref(), Some(&b"surfaces\0\0\0\0"[..]));
    assert_eq!(label.label.len(), 2);
    assert_eq!(label.label[0].index, 1);
    assert_eq!(label.label[0].name.as_deref(), Some(&b"top\0"[..]));
    assert_eq!(label.label[1].index, 2);
    assert_eq!(label.label[1].name.as_deref(), Some(&b"bottom side\0"[..]));

    // LABL: one label on each contour.
    assert_eq!(obj.cont.len(), 2);
    for (co, want) in [(0usize, &b"c0\0\0"[..]), (1, &b"c1\0\0"[..])] {
        let label = obj.cont[co]
            .label
            .as_ref()
            .expect("contour label chunk must be read");
        assert_eq!(label.name.as_deref(), Some(want));
        assert_eq!(label.label.len(), 1);
        assert_eq!(
            label.label[0].name.as_deref(),
            Some(&b"first point label\0\0\0"[..])
        );
    }

    // CLIP: two object clip planes.  All the normals precede all the points in
    // the chunk, so an interleaved reader mixes plane 1's normal into plane 0's
    // point.
    assert_eq!(obj.clips.count, 2);
    assert_eq!(
        (obj.clips.flags, obj.clips.trans, obj.clips.plane),
        (3, 0, 1)
    );
    assert_eq!(
        (obj.clips.normal[1].x, obj.clips.normal[1].y),
        (1., 0.),
        "clip normal 1 must come from the normals block"
    );
    assert_eq!(
        (
            obj.clips.point[0].x,
            obj.clips.point[0].y,
            obj.clips.point[0].z
        ),
        (-1., -2., -3.)
    );
    assert_eq!(
        (
            obj.clips.point[1].x,
            obj.clips.point[1].y,
            obj.clips.point[1].z
        ),
        (-4., -5., -6.)
    );

    // MESH: a closed tetrahedron on surface 1, written as normal/vertex index
    // pairs behind IMOD_MESH_BGNPOLYNORM.
    assert_eq!(obj.mesh.len(), 1);
    assert_eq!(obj.mesh[0].surf, 1);
    assert_eq!(obj.mesh[0].vert.len(), 24);
    assert_eq!(obj.mesh[0].list.len(), 27);
    assert_eq!(obj.mesh[0].list[0], -23);
    assert_eq!(obj.mesh[0].list[25], -22);
    assert_eq!(obj.mesh[0].list[26], -1);

    // VIEW: one real view past the default, carrying one object view whose
    // clip set has three planes.  Planes 1..5 live past the material bytes and
    // are only read when the record is a full BYTES_PER_OBJVIEW wide.
    assert_eq!(model.view.len(), 2);
    let view = &model.view[1];
    assert_eq!(view.objview.len(), 1);
    let ov = &view.objview[0];
    assert_eq!(ov.clips.count, 3);
    assert_eq!(
        (
            ov.clips.normal[1].x,
            ov.clips.normal[1].y,
            ov.clips.normal[1].z
        ),
        (0., 1., 0.)
    );
    assert_eq!(
        (
            ov.clips.normal[2].x,
            ov.clips.normal[2].y,
            ov.clips.normal[2].z
        ),
        (0., 0., 1.)
    );
    assert_eq!(
        (
            ov.clips.point[1].x,
            ov.clips.point[1].y,
            ov.clips.point[1].z
        ),
        (-10., -11., -12.)
    );
    assert_eq!(
        (
            ov.clips.point[2].x,
            ov.clips.point[2].y,
            ov.clips.point[2].z
        ),
        (-13., -14., -15.)
    );

    // MCLP: the view's own clip planes, same normals-then-points layout.
    assert_eq!(view.clips.count, 2);
    assert_eq!(
        (
            view.clips.normal[1].x,
            view.clips.normal[1].y,
            view.clips.normal[1].z
        ),
        (0., -1., 0.)
    );
    assert_eq!(
        (
            view.clips.point[0].x,
            view.clips.point[0].y,
            view.clips.point[0].z
        ),
        (-16., -17., -18.)
    );
}

#[test]
fn rewrites_the_authored_model_byte_for_byte() {
    let model = imod_read(fixture()).expect("authored model must decode");
    let out = std::env::temp_dir().join(format!("imod-rs-chunks-{}.mod", std::process::id()));
    imod_file_write(&model, &out).expect("model must write");
    let want = std::fs::read(fixture()).unwrap();
    let got = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&out);
    assert_eq!(
        got.len(),
        want.len(),
        "rewritten model must be the same size as the authored one"
    );
    let differing = (0..want.len()).filter(|i| want[*i] != got[*i]).count();
    assert_eq!(differing, 0, "rewritten model must be byte-identical");
}

#[test]
fn imodinfo_ascii_and_verbose_match_the_reference_output() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // The goldens are captured through a pipe, never a file redirect: under a
    // redirect glibc block-buffers the `printf` header and it lands after the
    // `fout` body.
    for (args, golden) in [
        (
            vec!["-a"],
            root.join("fixtures/model-view-clip-label.ascii.txt"),
        ),
        (
            vec!["-vv"],
            root.join("fixtures/model-view-clip-label.verbose.txt"),
        ),
        (
            vec!["-c"],
            root.join("fixtures/model-view-clip-label.c.txt"),
        ),
        (
            vec!["-i"],
            root.join("fixtures/model-view-clip-label.i.txt"),
        ),
        (
            vec!["-F"],
            root.join("fixtures/model-view-clip-label.F.txt"),
        ),
        (
            vec!["-s"],
            root.join("fixtures/model-view-clip-label.s.txt"),
        ),
        // `-t` applies the object's and the view's clip planes; `-x` sets a
        // subarea.  Both drive `trim_scan_contour` and `scan_contour_area`.
        (
            vec!["-t", "1"],
            root.join("fixtures/model-view-clip-label.t1.txt"),
        ),
        (
            vec!["-x", "0,5"],
            root.join("fixtures/model-view-clip-label.x05.txt"),
        ),
        (
            vec!["-x", "0,5", "-s"],
            root.join("fixtures/model-view-clip-label.x05s.txt"),
        ),
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
            .current_dir(root.join("fixtures"))
            .args(&args)
            .arg("model-view-clip-label.mod")
            .output()
            .expect("imodinfo executable must start");
        assert!(out.status.success());
        let got = String::from_utf8_lossy(&out.stdout).to_string();
        let want = std::fs::read_to_string(&golden).unwrap();
        assert_eq!(got, want, "imodinfo {args:?} must match the reference text");
    }
}

#[test]
fn reads_the_ascii_form_the_reference_writes() {
    // `fixtures/model-view-clip-label.ascii.txt` is the reference `imodinfo
    // -a` output, which is itself a readable ASCII model: `imodFgetline`
    // (`imodel_files.c:2020`) skips the `#` banner and the blank lines ahead of
    // the `imod 1` record.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let ascii = root.join("fixtures/model-view-clip-label.ascii.txt");
    let from_ascii = imod_read(&ascii).expect("the ascii model must decode");
    let from_binary = imod_read(fixture()).expect("authored model must decode");

    assert_eq!(from_ascii.obj.len(), from_binary.obj.len());
    assert_eq!(from_ascii.xmax, from_binary.xmax);
    assert_eq!(from_ascii.pixsize, from_binary.pixsize);
    assert_eq!(from_ascii.obj[0].name, from_binary.obj[0].name);
    // The ASCII reader only ORs object flags in, so the bits `imodObjectDefault`
    // (`iobj.c:81`) sets -- IMOD_OBJFLAG_DRAW_LABEL and IMOD_OBJFLAG_SCALE_WDTH
    // -- survive, where the binary OBJT record assigns the word outright.
    let defaults = (1u32 << 27) | (1 << 28);
    assert_eq!(
        from_ascii.obj[0].flags & !defaults,
        from_binary.obj[0].flags
    );
    assert_eq!(from_ascii.obj[0].flags & defaults, defaults);
    assert_eq!(from_ascii.obj[0].clips, from_binary.obj[0].clips);
    assert_eq!(from_ascii.view[1].clips, from_binary.view[1].clips);
    assert_eq!(
        from_ascii.obj[0].cont[0].pts,
        from_binary.obj[0].cont[0].pts
    );

    // The reference `imodinfo -c` reading the same ascii file is the golden.
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_imodinfo"))
        .current_dir(root.join("fixtures"))
        .args(["-c", "model-view-clip-label.ascii.txt"])
        .output()
        .expect("imodinfo executable must start");
    assert!(out.status.success());
    let got = String::from_utf8_lossy(&out.stdout).to_string();
    let want =
        std::fs::read_to_string(root.join("fixtures/model-view-clip-label.ascii.c.txt")).unwrap();
    assert_eq!(got, want);
}

#[test]
fn imodjoin_reproduces_the_reference_output_byte_for_byte() {
    // `fixtures/model-view-clip-label.joined.mod` is the reference `imodjoin
    // model-view-clip-label.mod empty.seed out.mod` output with the two
    // regions the source never initialises zeroed: `imodjoin.c:181` `malloc`s
    // an `IrefImage` and fills only `otrans`, `ctrans`, `crot` and `cscale`,
    // so the `MINX` chunk's `oscale` and `orot` are heap residue and differ
    // between two reference runs.  Everything else must match exactly.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-join-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(fixture(), dir.join("model-view-clip-label.mod")).unwrap();
    std::fs::copy(
        root.join("fixtures/model-empty-seed.mod"),
        dir.join("model-empty-seed.mod"),
    )
    .unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_imodjoin"))
        .current_dir(&dir)
        .args([
            "model-view-clip-label.mod",
            "model-empty-seed.mod",
            "out.mod",
        ])
        .status()
        .expect("imodjoin executable must start");
    assert!(status.success());

    let mut got = std::fs::read(dir.join("out.mod")).unwrap();
    let want = std::fs::read(root.join("fixtures/model-view-clip-label.joined.mod")).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(got.len(), want.len(), "joined model must be the same size");
    let minx = want
        .windows(4)
        .position(|w| w == b"MINX")
        .expect("the joined model must carry a MINX chunk");
    for (from, to) in [(minx + 8, minx + 20), (minx + 32, minx + 44)] {
        for byte in &mut got[from..to] {
            *byte = 0;
        }
    }
    let differing = (0..want.len()).filter(|i| want[*i] != got[*i]).count();
    assert_eq!(differing, 0, "joined model must match the reference");
}

#[test]
fn imodinfo_closed_contour_modes_match_the_reference_output() {
    // `imodinfo_ellipse` and `imodinfo_ratios` return immediately for an open
    // or scattered object, so the ellipse fit (`imodContourEquivEllipse`) and
    // the length/area ratio need a closed-contour object.
    // `fixtures/model-closed-contours.mod` is the same authored model with
    // `--closed`: object flags 0 and two polygonal contours.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for (args, golden) in [
        (vec!["-e"], "fixtures/model-closed-contours.e.txt"),
        (vec!["-r"], "fixtures/model-closed-contours.r.txt"),
        (vec!["-c"], "fixtures/model-closed-contours.c.txt"),
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_imodinfo"))
            .current_dir(root.join("fixtures"))
            .args(&args)
            .arg("model-closed-contours.mod")
            .output()
            .expect("imodinfo executable must start");
        assert!(out.status.success());
        let got = String::from_utf8_lossy(&out.stdout).to_string();
        let want = std::fs::read_to_string(root.join(golden)).unwrap();
        assert_eq!(got, want, "imodinfo {args:?} must match the reference text");
    }
}
