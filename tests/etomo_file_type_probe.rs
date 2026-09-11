//! Differential checks against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/type/FileType.java`'s instance half.
//!
//! The reference values were read out of a running JVM with a reflection harness (the
//! `Variable` singletons and `getLeftSide` are package-private or private), built the
//! way the header of `tests/etomo_utilities_probe.rs` describes.  Set
//! `IMOD_RS_ETOMO_PROBE` to dump the table again for a fresh comparison.
use imod_rs::imod::etomo::r#type::axis_id::AxisID;
use imod_rs::imod::etomo::r#type::axis_type::AxisType;
use imod_rs::imod::etomo::r#type::file_type::{self, FileType, Variable};

fn p(label: &str, value: Option<String>) {
    if std::env::var("IMOD_RS_ETOMO_PROBE").is_ok() {
        println!(
            "{}\t{}",
            label,
            value.unwrap_or_else(|| "<null>".to_string())
        );
    }
}

#[test]
fn jvm_verified_file_type_expectations() {
    // Variable regexes, read out of the JVM's private fields.  DATASET's ":" branch is
    // taken only on macOS and Windows; the reference runs on Linux, like this test.
    let dataset = Variable::dataset();
    let axis = Variable::axis();
    let dataset_and_axis = Variable::dataset_and_axis();
    let orig_raw = Variable::orig_raw_image_extension();
    p("Variable.DATASET.regex", Some(dataset.to_string()));
    p("Variable.AXIS.regex", Some(axis.to_string()));
    p(
        "Variable.DATASET_AND_AXIS.regex",
        Some(dataset_and_axis.to_string()),
    );
    p(
        "Variable.ORIG_RAW_IMAGE_EXTENSION.regex",
        Some(orig_raw.to_string()),
    );
    assert_eq!(dataset.to_string(), "[^\\s`'\"!#$%&*(){};/?\\|]+");
    assert_eq!(axis.to_string(), "[ab]?");
    assert_eq!(
        dataset_and_axis.to_string(),
        "[^\\s`'\"!#$%&*(){};/?\\|]+[ab]?"
    );
    assert_eq!(
        orig_raw.to_string(),
        "\\.\\Qhdf\\E|\\Qmrc\\E|\\Qst\\E|\\Qtif\\E|\\Qtiff\\E"
    );
    assert_eq!(
        Variable::select_integer_instance(1).map(|v| v.to_string()),
        Some("\\d{1,}".to_string())
    );
    assert_eq!(
        Variable::select_integer_instance(4).map(|v| v.to_string()),
        Some("\\d{4,}".to_string())
    );
    assert_eq!(Variable::select_integer_instance(5), None);

    // containsValidDatasetName.
    for (path, expected_ok, expected_msg) in [
        ("BBa.st", true, ""),
        ("/a/b/BBa.st", true, ""),
        (
            "BB a.st",
            false,
            "file name BB a.st contains illegal characters[s].  See Tomography Guide 1.5. File Format and Naming Conventions.  ",
        ),
        (
            "BB`a.st",
            false,
            "file name BB`a.st contains illegal characters[s].  See Tomography Guide 1.5. File Format and Naming Conventions.  ",
        ),
        ("BB:a.st", true, ""),
        ("", false, "No input image file path found.  "),
        (
            "bad|name.mrc",
            false,
            "file name bad|name.mrc contains illegal characters[s].  See Tomography Guide 1.5. File Format and Naming Conventions.  ",
        ),
        ("dataset", true, ""),
        ("datasetb.mrc", true, ""),
        (
            "data(set).mrc",
            false,
            "file name data(set).mrc contains illegal characters[s].  See Tomography Guide 1.5. File Format and Naming Conventions.  ",
        ),
    ] {
        let mut err_msg = String::new();
        let ok = file_type::contains_valid_dataset_name(Some(path), Some(&mut err_msg));
        p(
            &format!("containsValidDatasetName|{}", path),
            Some(format!("{}|{}", ok, err_msg)),
        );
        assert_eq!(
            (ok, err_msg.as_str()),
            (expected_ok, expected_msg),
            "{}",
            path
        );
    }

    // The old-style instance path, through the protected factory the JVM harness called
    // by reflection.
    let file_type = FileType::construct_instance(true, true, Some("_fixed"), Some(".st"));
    assert_eq!(
        file_type.get_left_side(
            Some("BB"),
            Some(AxisType::DualAxis),
            Some(AxisID::Only),
            None
        ),
        Some("BBa_fixed".to_string())
    );
    assert_eq!(
        file_type.get_left_side(
            Some("BB"),
            Some(AxisType::SingleAxis),
            Some(AxisID::Only),
            None
        ),
        Some("BB_fixed".to_string())
    );
    assert_eq!(
        file_type.get_left_side(
            Some("BB"),
            Some(AxisType::DualAxis),
            Some(AxisID::Second),
            Some("v1")
        ),
        Some("BBb_fixedv1".to_string())
    );
    assert_eq!(file_type.to_string(), "_fixed.st");
    assert!(file_type.has_fixed_name(Some(AxisType::SingleAxis)));
    assert_eq!(
        file_type.get_extension_for_axis_type(Some(AxisType::SingleAxis)),
        Some(".st".to_string())
    );
    assert_eq!(
        file_type.derive_file_name(
            Some("root"),
            Some(AxisType::DualAxis),
            Some(AxisID::First),
            None,
            None
        ),
        Some("roota_fixed.st".to_string())
    );
    assert!(file_type.equals_name_description(
        Some(AxisType::SingleAxis),
        true,
        true,
        Some("_fixed"),
        Some(".st")
    ));
    assert!(file_type.equals_file_name(
        Some(AxisType::SingleAxis),
        "BBa_fixed.st",
        true,
        true,
        "\\QBBa\\E",
        ""
    ));

    let file_type2 = FileType::construct_instance(false, false, Some("flatten"), Some(".com"));
    assert_eq!(
        file_type2.get_left_side(
            Some("BB"),
            Some(AxisType::DualAxis),
            Some(AxisID::Only),
            None
        ),
        Some("flatten".to_string())
    );
    assert_eq!(file_type2.to_string(), "flatten.com");
}
