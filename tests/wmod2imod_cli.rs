//! Real legacy-WIMP command-line coverage for `wmod2imod`.

use imod_rs::imod::libimod::imodel_files::imod_read;
use std::process::Command;

#[test]
fn wmod2imod_preserves_imodnew_scale_defaults_and_source_x_option_typo() {
    let root = std::env::temp_dir().join(format!("imod-rs-wmod2imod-cli-{}", std::process::id()));
    let input = root.with_extension("wimp");
    let output = root.with_extension("mod");
    std::fs::write(
        &input,
        " Model file name........................fixture.wimp\n\
          max # of object.......................    2\n\
          # of node.............................    4\n\
          # of object...........................    1\n\
           Object sequence : \n\
           Object #:           1\n\
          # of point:           2\n\
          Display switch:1  247\n\
              #    X       Y       Z      Mark    Label \n\
                1    1.00    2.00    3.00   0\n\
                2    4.00    5.00    6.00   0\n\
          \n  END\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_wmod2imod"))
        .args([
            "-x",
            "2",
            "-y",
            "3",
            "-z",
            "4",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .expect("run wmod2imod");
    assert!(result.status.success(), "{:?}", result);
    let model = imod_read(&output).expect("converted WIMP must be a readable IMOD model");
    // C assigns zscale = Xscale, yscale = Yscale, then zscale = Zscale.
    assert_eq!((model.xscale, model.yscale, model.zscale), (1., 3., 4.));
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(model.name.as_ptr()) },
        c"IMOD-NewModel"
    );
    // `imodWrite` adds the byte-material, multiple-clip and mesh-thickness
    // format bits before it writes the source-created default model.
    assert_eq!(
        model.flags,
        (1 << 13) | (1 << 12) | (1 << 11) | (1 << 10) | (1 << 9)
    );
    assert_eq!(
        (
            model.drawmode,
            model.mousemode,
            model.whitelevel,
            model.res,
            model.thresh,
            model.pixsize,
            model.xmax,
            model.ymax,
            model.zmax,
        ),
        (1, 2, 255, 3, 128, 1., 1, 1, 1)
    );
    assert_eq!(model.obj[0].cont[0].pts.len(), 2);
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn truncated_wimp_record_is_tolerated_the_way_the_source_fgetline_loop_is() {
    // `imod_from_wmod` reads point lines with `fgetline`, which returns a
    // negative length and an empty buffer at EOF; the following `sscanf` then
    // fails and leaves the function-scope `struct Mod_Point point` alone, so
    // the last scanned point is added again for every missing line.  The
    // native wmod2imod exits 0 here (verified against
    // /tmp/imod-reference-build/imodutil/wmod2imod); the record declares three
    // points and supplies one.
    let root = std::env::temp_dir().join(format!(
        "imod-rs-wmod2imod-truncated-{}",
        std::process::id()
    ));
    let input = root.with_extension("wimp");
    let output = root.with_extension("mod");
    std::fs::write(
        &input,
        "  Object #: 1\n # of point: 3\n Display switch:1  247\n\
              #    X       Y       Z\n      1 1.0 2.0 3.0\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_wmod2imod"))
        .arg(&input)
        .arg(&output)
        .output()
        .expect("run wmod2imod");
    assert_eq!(result.status.code(), Some(0), "{result:?}");
    assert!(result.stderr.is_empty());
    let model = imod_read(&output).expect("truncated WIMP still yields a model");
    assert_eq!(model.obj.len(), 1);
    assert_eq!(model.obj[0].cont[0].pts.len(), 3);
    for point in &model.obj[0].cont[0].pts {
        assert_eq!((point.x, point.y, point.z), (1., 2., 3.));
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn non_wimp_text_produces_the_source_empty_model_rather_than_an_error() {
    // No line contains the "Object #:" marker, so the source builds an empty
    // model and writes it; the native wmod2imod exits 0 with no diagnostic.
    let root =
        std::env::temp_dir().join(format!("imod-rs-wmod2imod-garbage-{}", std::process::id()));
    let input = root.with_extension("wimp");
    let output = root.with_extension("mod");
    std::fs::write(&input, "garbage\nnot a wimp file\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_wmod2imod"))
        .arg(&input)
        .arg(&output)
        .output()
        .expect("run wmod2imod");
    assert_eq!(result.status.code(), Some(0), "{result:?}");
    assert!(result.stderr.is_empty());
    let model = imod_read(&output).expect("empty WIMP still yields a model");
    assert!(model.obj.is_empty());
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}
