//! Differential checks against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/util/MRCHeader.java`'s comment-section parser.
//!
//! `parseCommentData` is private, so the expectations below were read out of a reference
//! runtime with a reflection harness compiled into `etomo.util`, built the way the header
//! of `tests/etomo_utilities_probe.rs` describes.  It is the half of `MRCHeader`'s
//! parsing that does not need a `SystemProgram`: `read` calls it once per line of
//! `header`'s output, and the `binning`, `imageRotation`, `bidir`, `dosym` and
//! `feiPixelSize` fields come from it alone.
use imod_rs::imod::etomo::r#type::axis_id::AxisID;
use imod_rs::imod::etomo::util::mrc_header::{
    self, FLOATING_POINT_MODE, MRCHeader, N_SECTIONS_INDEX, N_SECTIONS_INDEX_BRIEF, SIZE_HEADER,
    SIZE_HEADER_BRIEF,
};

#[test]
fn jvm_verified_parse_comment_data() {
    // (line, binning, imageRotation, bidir, doseSym) as the JVM printed them.
    let cases: Vec<(&str, &str, &str, &str, &str)> = vec![
        (
            " Tilt axis angle = -11.5, binning = 2  spot = 2  camera = 2",
            "2.0",
            "-11.5",
            "",
            "",
        ),
        (
            " Tilt axis rotation angle = -24.9 (Corrected sign)",
            "1.0",
            "-24.9",
            "",
            "",
        ),
        (" Pixel size in nanometers = 1.016", "1.0", "", "", ""),
        (" b3dbidir = 1", "1.0", "", "1.0", ""),
        (" dosym = 1", "1.0", "", "", "1.0"),
        (
            "     Tilt axis angle = 85.3, binning = 1  spot = 8  camera = 0",
            "1.0",
            "85.3",
            "",
            "",
        ),
        (" no divider here", "1.0", "", "", ""),
        ("", "1.0", "", "", ""),
        ("   =   ", "1.0", "", "", ""),
        (
            " RO image file on unit   1 : a.st     Size=      1024 x 1024, 61 sections",
            "1.0",
            "",
            "",
            "",
        ),
        (" binning=4", "4.0", "", "", ""),
        (" Tilt axis angle = 12,binning = 3", "3.0", "12.0", "", ""),
        (
            " SerialEM: Tilt axis angle = 0.0, binning = 1  spot = 6  camera = 0",
            "1.0",
            "0.0",
            "",
            "",
        ),
        (" Something bidir = 7", "1.0", "", "7.0", ""),
        (" Tilt axis angle = abc", "1.0", "", "", ""),
        (
            " Pixel size in nanometers = 1.016 extra stuff",
            "1.0",
            "",
            "",
            "",
        ),
    ];
    for (index, (line, binning, rotation, twodir, dose_sym)) in cases.iter().enumerate() {
        let header = MRCHeader::get_instance(
            Some(&format!("/tmp/imod-rs-mhprobe{}.mrc", index)),
            Some(AxisID::Only),
        )
        .unwrap();
        header.borrow_mut().parse_comment_data(line);
        let header = header.borrow();
        assert_eq!(header.get_binning(), *binning, "binning of [{}]", line);
        assert_eq!(
            header.get_image_rotation().to_string(),
            *rotation,
            "imageRotation of [{}]",
            line
        );
        assert_eq!(
            header.get_twodir().to_string(),
            *twodir,
            "bidir of [{}]",
            line
        );
        assert_eq!(
            header.get_dose_sym().to_string(),
            *dose_sym,
            "doseSym of [{}]",
            line
        );
    }
}

#[test]
fn jvm_verified_param_string_and_constants() {
    let header =
        MRCHeader::get_instance(Some("/tmp/imod-rs-mhprobe-all.mrc"), Some(AxisID::First)).unwrap();
    header
        .borrow_mut()
        .parse_comment_data(" Tilt axis angle = -11.5, binning = 2  spot = 2  camera = 2");
    header
        .borrow_mut()
        .parse_comment_data(" Pixel size in nanometers = 1.016");
    header.borrow_mut().parse_comment_data(" dosym = 1");
    assert_eq!(
        header.borrow().param_string(),
        ",\nfilename=/tmp/imod-rs-mhprobe-all.mrc,nColumns=-1,nRows=-1,\
         \nnSections=-1,mode=-1,\nxPixelSize=,yPixelSize=,\
         \nzPixelSize=,xPixelSpacing=NaN,\nyPixelSpacing=NaN,zPixelSpacing=NaN,\
         \nimageRotation=-11.5,binning=2.0,\naxisID=First,dosesym =1.0,feiPixelSize=1.016"
    );
    {
        let header = header.borrow();
        assert_eq!(header.get_n_columns(), -1);
        assert_eq!(header.get_n_rows(), -1);
        assert_eq!(header.get_n_sections(), -1);
        assert_eq!(header.get_mode(), -1);
        assert_eq!(header.get_x_pixel_size().to_string(), "");
        assert!(header.get_x_pixel_spacing().is_nan());
    }

    assert_eq!(SIZE_HEADER, "Number of columns, rows, sections");
    assert_eq!(N_SECTIONS_INDEX, 8);
    assert_eq!(SIZE_HEADER_BRIEF, "Dimensions:");
    assert_eq!(N_SECTIONS_INDEX_BRIEF, 3);
    assert_eq!(FLOATING_POINT_MODE, 2);

    // The n'ton table hands back the same object for the same absolute path.
    let again =
        MRCHeader::get_instance(Some("/tmp/imod-rs-mhprobe-all.mrc"), Some(AxisID::Only)).unwrap();
    assert!(std::rc::Rc::ptr_eq(&header, &again));
    assert!(MRCHeader::get_instance(None, Some(AxisID::Only)).is_none());

    // Silence the unused-import warning for the module path itself.
    let _ = std::mem::size_of::<mrc_header::CommentData>();
}
