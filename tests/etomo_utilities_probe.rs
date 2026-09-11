//! Differential checks against the reference eTomo JVM for
//! `IMOD/Etomo/src/etomo/util/Utilities.java` and
//! `IMOD/Etomo/src/etomo/type/Extension.java`.
//!
//! `jvm_verified_expectations` asserts values captured from a real JVM run; it always
//! runs.  `etomo_utilities_probe` dumps the full 1130-line comparison table used to
//! capture them and is skipped unless `IMOD_RS_ETOMO_PROBE` is set.  To regenerate the
//! table, build the reference classes outside `IMOD/` with
//! `javac -nowarn -d <out> -encoding ISO-8859-1` over every `.java` under
//! `IMOD/Etomo/src` except `*Test.java`, `*Tests.java`, `JUnit*`, `etomo/uitest/` and
//! `util/TestUtilites.java`, compile a Java harness that prints the same
//! `label<TAB>value` lines, and diff the two streams (stdout *and* stderr).
use imod_rs::imod::etomo::r#type::const_etomo_number::{Type, java_lang_double_to_string};
use imod_rs::imod::etomo::r#type::etomo_number::EtomoNumber;
use imod_rs::imod::etomo::r#type::extension::{self, Extension};
use imod_rs::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use imod_rs::imod::etomo::util::utilities as u;
use std::path::Path;

fn p(label: &str, value: Option<String>) {
    println!(
        "{}\t{}",
        label,
        value.unwrap_or_else(|| "<null>".to_string())
    );
}

const STRINGS: &[&str] = &[
    "",
    " ",
    "  \t ",
    "abc",
    " abc ",
    "a.b.c",
    ".abc",
    "abc.",
    "/a/b/c.mrc",
    "/a//b/",
    "BBa.st",
    "BBa_rec.mrc",
    "dataset_srec86.mrc",
    "root-020.com",
    "root-1000-sync.log",
    "x~~",
    "a,b,,c",
    ",,a,b",
    "a-b_0000-c",
    "New project root name: batchNov10-140224 [SRW6]",
    "With log in: swbrt_Batch1BBa.1032348.log (ebt1) [SRW2]",
    "Finished processing stack Batch1BBa.st with successful completion (ebt2) [SRW3]",
    "label: value ",
    "a b c",
    "..  .. Artie.Chuck .&. Bob.",
    "srec86",
    "tif",
    "TIF",
];

#[test]
fn etomo_utilities_probe() {
    if std::env::var("IMOD_RS_ETOMO_PROBE").is_err() {
        return;
    }
    for s in STRINGS {
        let s = *s;
        p(
            &format!("isEmpty|{s}"),
            Some(u::is_empty(Some(s)).to_string()),
        );
        p(
            &format!("containsWildcard|{s}"),
            Some(u::contains_wildcard(Some(s)).to_string()),
        );
        p(&format!("getExtension|{s}"), u::get_extension(Some(s)));
        p(
            &format!("removeExtension|{s}"),
            u::remove_extension(Some(s)),
        );
        p(
            &format!("getSuffix.|{s}"),
            u::get_suffix(Some(s), Some(".")),
        );
        p(
            &format!("removeLeftSide_|{s}"),
            u::remove_left_side(Some(s), Some("_")),
        );
        p(
            &format!("removeRightSide-|{s}"),
            u::remove_right_side(Some(s), Some("-")),
        );
        p(&format!("stripIDs|{s}"), u::strip_ids(Some(s)));
        p(&format!("stripLabel|{s}"), u::strip_label(Some(s)));
        p(
            &format!("getMessageIDTag|{s}"),
            u::get_message_id_tag(Some(s), Some("SRW")),
        );
        p(
            &format!("getStackID|{s}"),
            u::get_stack_id(Some(s), Some("ebt")),
        );
        p(
            &format!("getNumberElements|{s}"),
            Some(u::get_number_elements(Some(s)).to_string()),
        );
        p(
            &format!("getStrippedFileName|{s}"),
            u::get_stripped_file_name(Some(s)),
        );
        p(&format!("cleanUpLabel|{s}"), Some(u::clean_up_label(s)));
        p(&format!("quoteLabel|{s}"), u::quote_label(Some(s)));
        p(&format!("escapeSpaces|{s}"), Some(u::escape_spaces(s)));
        p(
            &format!("escapeSpaces2|{s}"),
            Some(u::escape_spaces_double(s, true)),
        );
        p(
            &format!("stripRepeatingString.|{s}"),
            u::strip_repeating_string(Some(s), Some(".")),
        );
        p(
            &format!("stripRepeatingStringa|{s}"),
            u::strip_repeating_string(Some(s), Some("a")),
        );
        p(
            &format!("getElementFromList0|{s}"),
            u::get_element_from_list(Some(s), 0),
        );
        p(
            &format!("getElementFromList1|{s}"),
            u::get_element_from_list(Some(s), 1),
        );
        p(
            &format!("setAlignFramesRootname|{s}"),
            Some(u::set_align_frames_rootname(Some(s))),
        );
        p(
            &format!("extractDataset|{s}"),
            u::extract_dataset(true, Some(Path::new(s))),
        );
        p(
            &format!("extractAxisID|{s}"),
            u::extract_axis_id(true, Some(s)).map(|a| a.to_string()),
        );
        p(
            &format!("createPropertyKey|{s}"),
            u::create_property_key(Some(s), Some("key")),
        );
        p(
            &format!("getRegularExpressionClass|{s}"),
            u::get_regular_expression_class(Some(s)),
        );
        p(
            &format!("toStringIfSet|{s}"),
            Some(u::to_string_if_set(Some("L:"), Some(s))),
        );
        p(
            &format!("convertPathToUniversalWindows|{s}"),
            Some(u::convert_path_to_universal_windows(s)),
        );
        p(
            &format!("fileGetName|{s}"),
            Some(u::java_io_file_get_name(s)),
        );
        p(&format!("fileGetParent|{s}"), u::java_io_file_get_parent(s));
        let ext = Extension::get_instance(s);
        p(
            &format!("Extension.getInstance|{s}"),
            ext.map(|e| e.to_string()),
        );
        p(
            &format!("Extension.isInputImageFile|{s}"),
            Some(Extension::is_input_image_file_path(s).to_string()),
        );
        p(
            &format!("Extension.isComscript|{s}"),
            Some(Extension::is_comscript(s).to_string()),
        );
        p(
            &format!("Extension.getChunkNumber|{s}"),
            Extension::get_chunk_number(Some(s)).map(|n| n.to_string()),
        );
        p(
            &format!("Extension.substituteExtension|{s}"),
            Extension::substitute_extension(Some(s), Some(&extension::CLASS.mrc)),
        );
        p(
            &format!("Extension.substituteStdExtension|{s}"),
            Extension::substitute_standardized_extension(Some(s), Some(&extension::CLASS.rec)),
        );
        p(
            &format!("Extension.getLiteralInstance|{s}"),
            Extension::get_literal_instance(Some(s), None, true).map(|e| e.to_string()),
        );
    }
    p(
        "Extension.getImageInputRegex",
        Some(Extension::get_image_input_regex()),
    );
    p(
        "Extension.getInputImageFileDescr",
        Some(Extension::get_input_image_file_descr()),
    );
    p(
        "Extension.SINT.getFileNumber(dataset.sint86,OLD)",
        extension::CLASS
            .sint
            .get_file_number(Some("dataset.sint86"), Some(ImageFilenameStyle::Old))
            .map(|n| n.to_string()),
    );
    p(
        "Extension.SINT.getFileNumber(dataset_sint86.mrc,MRC)",
        extension::CLASS
            .sint
            .get_file_number(Some("dataset_sint86.mrc"), Some(ImageFilenameStyle::Mrc))
            .map(|n| n.to_string()),
    );
    p(
        "Extension.SINT.getSuffix(OLD)",
        Some(
            extension::CLASS
                .sint
                .get_suffix(Some(ImageFilenameStyle::Old)),
        ),
    );
    p(
        "Extension.SINT.getSuffix(MRC)",
        Some(
            extension::CLASS
                .sint
                .get_suffix(Some(ImageFilenameStyle::Mrc)),
        ),
    );
    p(
        "Extension.REC.getSuffix(HDF)",
        Some(
            extension::CLASS
                .rec
                .get_suffix(Some(ImageFilenameStyle::Hdf)),
        ),
    );
    p(
        "Extension.MRC.getSuffix(MRC)",
        Some(
            extension::CLASS
                .mrc
                .get_suffix(Some(ImageFilenameStyle::Mrc)),
        ),
    );
    p(
        "Extension.SINT.fileNameEndsWith(dataset.sint86,OLD)",
        Some(
            extension::CLASS
                .sint
                .file_name_ends_with(Some(ImageFilenameStyle::Old), Some("dataset.sint86"))
                .to_string(),
        ),
    );
    p(
        "Extension.REC.fileNameEndsWith(dataset_rec.mrc,MRC)",
        Some(
            extension::CLASS
                .rec
                .file_name_ends_with(Some(ImageFilenameStyle::Mrc), Some("dataset_rec.mrc"))
                .to_string(),
        ),
    );
    p(
        "Extension.COM.fileNameEndsWith(com,MRC)",
        Some(
            extension::CLASS
                .com
                .file_name_ends_with(Some(ImageFilenameStyle::Mrc), Some("com"))
                .to_string(),
        ),
    );
    p(
        "Extension.equals(files)",
        Some(
            Extension::equals_files(Some(Path::new("a.mrc")), Some(Path::new("b.mrc"))).to_string(),
        ),
    );
    p(
        "Extension.equals(files2)",
        Some(
            Extension::equals_files(Some(Path::new("a.mrc")), Some(Path::new("b.st"))).to_string(),
        ),
    );
    p(
        "Extension.equals(files3)",
        Some(
            Extension::equals_files(Some(Path::new("a.zzz")), Some(Path::new("b.zzz"))).to_string(),
        ),
    );
    p(
        "Extension.equals(files4)",
        Some(Extension::equals_files(Some(Path::new("a")), Some(Path::new("b"))).to_string()),
    );

    let wrap_me = "The quick brown fox, jumps over, the lazy dog, again and again and again";
    p("wrap(10)", u::wrap(Some(wrap_me), Some(", "), 0, 10, 0));
    p("wrap(20)", u::wrap(Some(wrap_me), Some(", "), 0, 20, 0));
    p(
        "wrap(20,max25)",
        u::wrap(Some(wrap_me), Some(", "), 0, 20, 25),
    );
    p("wrap(0,max8)", u::wrap(Some(wrap_me), Some(", "), 0, 0, 8));
    p(
        "wrap(5,min30)",
        u::wrap(Some(wrap_me), Some(", "), 30, 5, 0),
    );
    p(
        "canWrap",
        Some(u::can_wrap(Some(wrap_me), Some(", "), 0, 10, 0).to_string()),
    );

    let millis: &[f64] = &[
        0.0, 999.0, 1000.0, 59999.0, 60000.0, 61000.0, 3600000.0, 123456.7,
    ];
    for m in millis {
        p(
            &format!("millisToMinAndSecs|{}", java_lang_double_to_string(*m)),
            Some(u::millis_to_min_and_secs(*m)),
        );
    }
    let ds: &[f64] = &[
        0.0,
        0.5,
        1.2345,
        1.2355,
        -1.2345,
        0.0005,
        0.0015,
        12345.6789,
        1.0005,
        2.0005,
        0.9999,
        1e-9,
        123456789.12345,
    ];
    for d in ds {
        p(
            &format!("df000|{}", java_lang_double_to_string(*d)),
            Some(u::java_text_decimal_format_three_fraction_digits(*d)),
        );
        p(
            &format!("doubleToString|{}", java_lang_double_to_string(*d)),
            Some(java_lang_double_to_string(*d)),
        );
    }
    p(
        "buildString",
        u::build_string(
            Some(&[
                Some("Artie".to_string()),
                None,
                Some("".to_string()),
                Some("Bob".to_string()),
            ]),
            Some("^"),
        ),
    );
    p(
        "concatenate",
        u::concatenate(Some("a"), None, Some("c"), Some("-")),
    );
    p(
        "concatenate2",
        u::concatenate(None, Some("b"), Some("c"), Some("-")),
    );
    p(
        "setRootnameSelectedFiles",
        Some(u::set_rootname_selected_files(&[
            "abc_0001.mrc".to_string(),
            "abc_0002.mrc".to_string(),
        ])),
    );
    p(
        "setRootnameSelectedFiles2",
        Some(u::set_rootname_selected_files(&[
            "abc-000-x".to_string(),
            "abc-000-y".to_string(),
        ])),
    );
    p(
        "setRootnameSelectedFiles3",
        Some(u::set_rootname_selected_files(&[
            "0000abc".to_string(),
            "0000abd".to_string(),
        ])),
    );
    p(
        "getCommandAction",
        u::get_command_action_array(
            Some(&[
                "/usr/bin/tcsh".to_string(),
                "-f".to_string(),
                "/a/b/tilta.com".to_string(),
            ]),
            None,
        ),
    );
    p(
        "getCommandAction2",
        u::get_command_action_array(
            Some(&[
                "python".to_string(),
                "-u".to_string(),
                "/a/startprocess".to_string(),
                "x.log".to_string(),
                "1,2,3".to_string(),
                "-opt".to_string(),
            ]),
            Some(&[
                "# comment".to_string(),
                "makeBackupFile('foo.log')".to_string(),
            ]),
        ),
    );
    p(
        "getCommandActionStr",
        u::get_command_action(Some("  clip flipyz a.mrc b.mrc ")),
    );
    p(
        "getCommandActionStr2",
        u::get_command_action(Some("ssh host")),
    );
    p(
        "getFile",
        Some(
            u::get_file("/tmp/dir", Some(" sub/file.txt "))
                .to_string_lossy()
                .to_string(),
        ),
    );
    p(
        "getFileAbs",
        Some(
            u::get_file("/tmp/dir", Some("/abs/file.txt"))
                .to_string_lossy()
                .to_string(),
        ),
    );
    p(
        "getFileNull",
        Some(
            u::get_file("/tmp/dir", Some("   "))
                .to_string_lossy()
                .to_string(),
        ),
    );
    p("isWindowsOS", Some(u::is_windows_os().to_string()));
    p("isMacOS", Some(u::is_mac_os().to_string()));
    p("isJava7", Some(u::is_java7().to_string()));
    p(
        "isEmptyArray",
        Some(u::is_empty_array(Some(&[Some("a".to_string()), Some("b".to_string())])).to_string()),
    );
    p(
        "isEmptyArray2",
        Some(u::is_empty_array(Some(&[Some("".to_string()), Some("b".to_string())])).to_string()),
    );
    p(
        "getDateTimeStampRootNameLen",
        Some(u::get_date_time_stamp_root_name().len().to_string()),
    );
    let _ = EtomoNumber::new_with_type(Some(Type::Double));
}

/// Values captured from the reference eTomo JVM (see the module header).  Every
/// expectation here is a byte-for-byte copy of what
/// `java -cp <out> Probe` printed for the same input.
#[test]
fn jvm_verified_expectations() {
    // Utilities string handling.
    assert_eq!(
        u::strip_ids(Some("New project root name: batchNov10-140224 [SRW6]")),
        Some("New project root name: batchNov10-140224 ".to_string())
    );
    assert_eq!(
        u::get_message_id_tag(
            Some("Finished processing stack Batch1BBa.st with successful completion (ebt2) [SRW3]"),
            Some("SRW")
        ),
        Some("[SRW3]".to_string())
    );
    assert_eq!(
        u::get_stack_id(
            Some("Finished processing stack Batch1BBa.st with successful completion (ebt2) [SRW3]"),
            Some("ebt")
        ),
        Some("ebt2".to_string())
    );
    assert_eq!(
        u::strip_repeating_string(Some("..  .. Artie.Chuck .&. Bob."), Some(".")),
        Some("Artie.Chuck .&. Bob".to_string())
    );
    assert_eq!(
        u::strip_repeating_string(Some("..  .. Artie.Chuck .&. Bob."), Some("a")),
        Some("..  .. Artie.Chuck .&. Bob.".to_string())
    );
    assert_eq!(
        u::get_extension(Some("/a/b/c.mrc")),
        Some("mrc".to_string())
    );
    assert_eq!(u::remove_extension(Some("a.b.c")), Some("a.b".to_string()));
    assert_eq!(u::java_io_file_get_name("/a//b/"), "b".to_string());
    assert_eq!(
        u::java_io_file_get_parent("/a/b/c.mrc"),
        Some("/a/b".to_string())
    );
    assert_eq!(u::get_number_elements(Some("a,b,,c")), 3);
    assert_eq!(u::get_number_elements(Some(",,a,b")), 2);

    // Utilities wrapping and formatting.
    let wrap_me = "The quick brown fox, jumps over, the lazy dog, again and again and again";
    assert_eq!(
        u::wrap(Some(wrap_me), Some(", "), 0, 10, 0),
        Some(
            "The quick brown fox,\njumps over,\nthe lazy dog,\nagain and again and again"
                .to_string()
        )
    );
    assert_eq!(
        u::wrap(Some(wrap_me), Some(", "), 0, 0, 8),
        Some(
            "The quic\nk brown\nfox, jum\nps over,\nthe laz\ny dog, a\ngain and\nagain a\nnd again"
                .to_string()
        )
    );
    assert_eq!(u::millis_to_min_and_secs(999.0), "0:00");
    assert_eq!(u::millis_to_min_and_secs(61000.0), "1:01");
    assert_eq!(u::millis_to_min_and_secs(3600000.0), "60:00");
    assert_eq!(u::millis_to_min_and_secs(123456.7), "2:03");

    // java.text.DecimalFormat(".000"), whose HALF_EVEN ties are decided by whether
    // Double.toString already rounded up.
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(0.0),
        ".000"
    );
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(0.5),
        ".500"
    );
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(1.2345),
        "1.234"
    );
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(1.2355),
        "1.236"
    );
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(1.0005),
        "1.000"
    );
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(2.0005),
        "2.001"
    );
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(0.0005),
        ".000"
    );
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(0.0015),
        ".002"
    );
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(0.9999),
        "1.000"
    );
    assert_eq!(
        u::java_text_decimal_format_three_fraction_digits(123456789.12345),
        "123456789.123"
    );

    // Utilities command-line and path handling.
    assert_eq!(
        u::get_command_action_array(
            Some(&[
                "/usr/bin/tcsh".to_string(),
                "-f".to_string(),
                "/a/b/tilta.com".to_string()
            ]),
            None
        ),
        Some("tcsh tilta.com".to_string())
    );
    assert_eq!(
        u::get_command_action_array(
            Some(&[
                "python".to_string(),
                "-u".to_string(),
                "/a/startprocess".to_string(),
                "x.log".to_string(),
                "1,2,3".to_string(),
                "-opt".to_string()
            ]),
            Some(&[
                "# comment".to_string(),
                "makeBackupFile('foo.log')".to_string()
            ])
        ),
        Some("python startprocess x.log 1,2,3".to_string())
    );
    assert_eq!(
        u::get_command_action(Some("  clip flipyz a.mrc b.mrc ")),
        Some("clip flipyz a.mrc b.mrc".to_string())
    );
    assert_eq!(u::get_command_action(Some("ssh host")), None);
    assert_eq!(
        u::get_file("/tmp/dir", Some(" sub/file.txt "))
            .to_string_lossy()
            .to_string(),
        "/tmp/dir/sub/file.txt"
    );
    assert_eq!(
        u::get_file("/tmp/dir", Some("   "))
            .to_string_lossy()
            .to_string(),
        "/tmp/dir"
    );
    assert_eq!(
        u::build_string(
            Some(&[
                Some("Artie".to_string()),
                None,
                Some("".to_string()),
                Some("Bob".to_string())
            ]),
            Some("^")
        ),
        Some("Artie^Bob".to_string())
    );
    assert_eq!(
        u::set_rootname_selected_files(&["0000abc".to_string(), "0000abd".to_string()]),
        "0000ab"
    );

    // Extension.
    assert_eq!(
        Extension::get_image_input_regex(),
        "\\Qhdf\\E|\\Qmrc\\E|\\Qst\\E|\\Qtif\\E|\\Qtiff\\E"
    );
    assert_eq!(
        Extension::get_input_image_file_descr(),
        "hdf, mrc, st, tif, tiff"
    );
    assert_eq!(
        Extension::get_instance("dataset_srec86.mrc").map(|e| e.to_string()),
        Some("mrc".to_string())
    );
    assert_eq!(
        Extension::get_chunk_number(Some("root-1000-sync.log")).map(|n| n.to_string()),
        Some("1000".to_string())
    );
    assert_eq!(
        extension::CLASS
            .sint
            .get_file_number(Some("dataset_sint86.mrc"), Some(ImageFilenameStyle::Mrc))
            .map(|n| n.to_string()),
        Some("86".to_string())
    );
    assert_eq!(
        extension::CLASS
            .sint
            .get_suffix(Some(ImageFilenameStyle::Old)),
        ".sint"
    );
    assert_eq!(
        extension::CLASS
            .sint
            .get_suffix(Some(ImageFilenameStyle::Mrc)),
        "_sint.mrc"
    );
    assert_eq!(
        extension::CLASS
            .rec
            .get_suffix(Some(ImageFilenameStyle::Hdf)),
        "_rec.hdf"
    );
    assert!(
        extension::CLASS
            .rec
            .file_name_ends_with(Some(ImageFilenameStyle::Mrc), Some("dataset_rec.mrc"))
    );
    assert!(Extension::is_comscript("root-020.com"));
    assert!(!Extension::is_input_image_file_path("root-020.com"));
    let _ = EtomoNumber::new_with_type(Some(Type::Double));
    let _ = java_lang_double_to_string(1.0);
}
