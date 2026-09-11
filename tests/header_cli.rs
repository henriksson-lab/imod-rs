use imod_rs::imod::libiimod::iihdf::ii_hdf_open_new;
use imod_rs::imod::libiimod::iimage::{
    IIFILE_HDF, ii_delete, ii_open_new, ii_sync_from_mrc_header,
};
use imod_rs::imod::libiimod::mrcfiles::{MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_write};
use std::ffi::CString;
use std::process::Command;

/// Every PIP-driven invocation needs an autodoc directory, exactly as a real
/// IMOD install provides one through `AUTODOC_DIR` or `IMOD_DIR`.
const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

#[test]
fn header_reports_source_ordered_machine_readable_mrc_fields() {
    unsafe {
        let input =
            std::env::temp_dir().join(format!("imod-rs-header-cli-{}.mrc", std::process::id()));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut mrc: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut mrc, 3, 4, 2, MRC_MODE_FLOAT), 0);
        mrc.xlen = 6.0;
        mrc.ylen = 12.0;
        mrc.zlen = 10.0;
        mrc.xorg = 1.5;
        mrc.yorg = -2.0;
        mrc.zorg = 3.25;
        mrc.amin = -1.25;
        mrc.amax = 9.5;
        mrc.amean = 4.125;
        mrc.labels[0][..80].fill(b'A');
        mrc.nlabl = 1;
        mrc.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut mrc), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_header"))
            .env("AUTODOC_DIR", AUTODOC)
            .args(["-size", "-mode", "-minimum", "-maximum", "-mean"])
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "status={:?}, stderr={}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&result.stdout),
            format!(
                "{:8}{:8}{:8}\n{:4}\n{:>9}    \n{:>9}    \n{:>9}    \n",
                3, 4, 2, 2, "-1.2500", "9.5000", "4.1250"
            )
        );
        let ordinary = Command::new(env!("CARGO_BIN_EXE_header"))
            .env("AUTODOC_DIR", AUTODOC)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            ordinary.status.success(),
            "{}",
            String::from_utf8_lossy(&ordinary.stderr)
        );
        let text = String::from_utf8_lossy(&ordinary.stdout);
        let open_report = format!("RO image file on unit   1 : {}", input.display());
        assert!(text.contains(&open_report));
        assert!(
            text.find(&open_report).unwrap()
                < text.find("Number of columns, rows, sections").unwrap()
        );
        assert!(text.contains("Number of columns, rows, sections .....       3       4       2"));
        assert!(text.contains("Map mode ..............................    2   (32-bit float)"));
        assert!(text.contains("Space group,# extra bytes,idtype,lens"));
        // `irdhdr.f90` FORMAT 1020 uses `1x,i5` and `19a4,a3`, so it
        // prints a five-wide count after a leading blank and exactly 79 label
        // characters, despite an MRC label occupying 80 bytes.
        assert!(text.contains(&format!("\n     1 Titles :\n{}\n", "A".repeat(79))));
        assert!(!text.contains(&"A".repeat(80)));
        let brief = Command::new(env!("CARGO_BIN_EXE_header"))
            .env("AUTODOC_DIR", AUTODOC)
            .args(["-brief"])
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            brief.status.success(),
            "{}",
            String::from_utf8_lossy(&brief.stderr)
        );
        let brief_text = String::from_utf8_lossy(&brief.stdout);
        assert!(brief_text.contains("Dimensions:      3      4      2   Pixel size:"));
        assert!(brief_text.contains("Mode:  2               Min, max, mean:"));
        assert!(!brief_text.contains("Number of columns, rows, sections"));
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn header_pip_option_without_an_input_file_exits_as_source_does() {
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg("-brief")
        .output()
        .unwrap();
    // `call exitError` reaches `parse_input_params.f90:236`, which writes a
    // blank record and the prefixed message on the standard output unit and
    // then exits with status 1.
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: HEADER - No input file specified\n"
    );
    assert!(result.stderr.is_empty());
}

#[test]
fn header_reports_old_fei_and_mdoc_rotation_angle_source_branches() {
    unsafe {
        let old_fei =
            std::env::temp_dir().join(format!("imod-rs-header-old-fei-{}.mrc", std::process::id()));
        let old_fei_c = CString::new(old_fei.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(old_fei_c.as_ptr(), c"wb".as_ptr());
        let mut mrc: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut mrc, 2, 2, 1, MRC_MODE_FLOAT), 0);
        mrc.next = 48;
        mrc.nint = 0;
        mrc.nreal = 12;
        mrc.labels[0][..4].copy_from_slice(b"Fei ");
        mrc.nlabl = 1;
        mrc.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut mrc), 0);
        let mut extended = [0_f32; 12];
        extended[10] = 30.0;
        extended[11] = 20.0;
        assert_eq!(libc::fwrite(extended.as_ptr().cast(), 4, 12, file), 12);
        libc::fclose(file);
        let old_fei_result = Command::new(env!("CARGO_BIN_EXE_header"))
            .env("AUTODOC_DIR", AUTODOC)
            .arg(&old_fei)
            .output()
            .unwrap();
        assert!(
            old_fei_result.status.success(),
            "{}",
            String::from_utf8_lossy(&old_fei_result.stderr)
        );
        let old_fei_text = String::from_utf8_lossy(&old_fei_result.stdout);
        assert!(old_fei_text.contains("Tilt axis rotation angle =   -30.0 (Corrected sign)"));
        // FORMAT 102 writes the nanometer pixel size with `g11.4`.
        assert!(
            old_fei_text.contains("Pixel size in nanometers =  2.000    \n"),
            "{old_fei_text}"
        );
        assert!(
            old_fei_text
                .ends_with("Extended header has tilt angles - extract with \"extracttilts\"\n"),
            "{old_fei_text}"
        );

        let new_fei =
            std::env::temp_dir().join(format!("imod-rs-header-new-fei-{}.mrc", std::process::id()));
        let new_fei_c = CString::new(new_fei.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(new_fei_c.as_ptr(), c"wb".as_ptr());
        let mut mrc: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut mrc, 2, 2, 1, MRC_MODE_FLOAT), 0);
        mrc.next = 160;
        mrc.ext_type = *b"FEI1";
        mrc.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut mrc), 0);
        let mut extended = [0_u8; 160];
        extended[8..12].copy_from_slice(&(1_i32 << 12).to_ne_bytes());
        extended[140..148].copy_from_slice(&30.0_f64.to_ne_bytes());
        assert_eq!(libc::fwrite(extended.as_ptr().cast(), 1, 160, file), 160);
        libc::fclose(file);
        let new_fei_result = Command::new(env!("CARGO_BIN_EXE_header"))
            .env("AUTODOC_DIR", AUTODOC)
            .arg(&new_fei)
            .output()
            .unwrap();
        assert!(
            new_fei_result.status.success(),
            "{}",
            String::from_utf8_lossy(&new_fei_result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&new_fei_result.stdout)
                .contains("Tilt axis rotation angle =   -30.0 (Corrected sign)")
        );

        let mdoc_input =
            std::env::temp_dir().join(format!("imod-rs-header-mdoc-{}.mrc", std::process::id()));
        let mdoc_input_c = CString::new(mdoc_input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(mdoc_input_c.as_ptr(), c"wb".as_ptr());
        let mut mrc: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut mrc, 2, 2, 1, MRC_MODE_FLOAT), 0);
        mrc.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut mrc), 0);
        libc::fclose(file);
        let mdoc = mdoc_input.with_extension("mrc.mdoc");
        std::fs::write(
            &mdoc,
            "PixelSpacing = 25.0\n[ZValue = 0]\nRotationAngle = 30.0\n[T = TiltAxisAngle = -120.0]\n",
        )
        .unwrap();
        let mdoc_result = Command::new(env!("CARGO_BIN_EXE_header"))
            .env("AUTODOC_DIR", AUTODOC)
            .arg(&mdoc_input)
            .output()
            .unwrap();
        assert!(
            mdoc_result.status.success(),
            "{}",
            String::from_utf8_lossy(&mdoc_result.stderr)
        );
        let mdoc_text = String::from_utf8_lossy(&mdoc_result.stdout);
        assert!(
            mdoc_text.contains("Pixel size in nanometers =  2.500      , from mdoc"),
            "{mdoc_text}"
        );
        assert!(
            mdoc_text.contains("Tilt axis rotation angle =    30.0  (from RotationAngle in mdoc)")
        );
        std::fs::remove_file(old_fei).unwrap();
        std::fs::remove_file(new_fei).unwrap();
        std::fs::remove_file(mdoc_input).unwrap();
        std::fs::remove_file(mdoc).unwrap();
    }
}

#[test]
fn header_opens_source_hdf_multivolume_and_selects_requested_volume() {
    unsafe {
        let input =
            std::env::temp_dir().join(format!("imod-rs-header-volumes-{}.h5", std::process::id()));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let image = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_HDF);
        assert!(!image.is_null());
        let header = (*image).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 2, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(image, header);
        (*image).z_chunk_size = 1;
        assert_eq!((*image).write_header.unwrap()(image), 0);
        assert_eq!(ii_hdf_open_new(image, c"wb".as_ptr()), 0);
        let second_volume = *(*image).ii_volumes.add(1);
        let second_header = (*second_volume).header.cast::<MrcHeader>();
        assert_eq!(
            mrc_head_new(&mut *second_header, 2, 2, 2, MRC_MODE_FLOAT),
            0
        );
        ii_sync_from_mrc_header(second_volume, second_header);
        assert_eq!((*second_volume).write_header.unwrap()(second_volume), 0);
        ii_delete(second_volume);
        ii_delete(image);
        let result = Command::new(env!("CARGO_BIN_EXE_header"))
            .env("AUTODOC_DIR", AUTODOC)
            .args(["-volume", "2", "-size"])
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "status={:?}, stderr={}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&result.stdout),
            "       2       2       2\n"
        );
        std::fs::remove_file(input).unwrap();
    }
}

/// Copies one of the versioned `IMOD/Etomo/unitTestData` stacks listed in
/// `fixtures/IMOD-FIXTURES.md` into the process temp directory so the pinned
/// upstream file is never touched by a test run.
fn copy_real_fixture(name: &str, tag: &str) -> std::path::PathBuf {
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("IMOD/Etomo/unitTestData")
        .join(name);
    let target = std::env::temp_dir().join(format!(
        "imod-rs-header-{tag}-{}-{name}",
        std::process::id()
    ));
    std::fs::copy(&source, &target).unwrap();
    target
}

#[test]
fn header_machine_readable_fields_use_fortran_g_editing() {
    // `header.f90:153-168` writes these with `g15.5` and `g13.5`, so a
    // magnitude that rounds to five significant digits inside [0.1, 1.e5) is
    // `F(w-4).(5-k)` plus four blanks, never a fixed number of decimals.
    let fei = copy_real_fixture("feiHeader.st", "gedit");
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-minimum", "-maximum", "-mean", "-rms"])
        .arg(&fei)
        .output()
        .unwrap();
    assert!(result.status.success());
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "  -32767.    \n   725.00    \n  -16307.    \n   0.0000    (not computed)\n"
    );

    let binned = copy_real_fixture("newerHeaderBinned.st", "gedit");
    let maximum = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-maximum"])
        .arg(&binned)
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&maximum.stdout), "   16521.    \n");
    let spacing = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-pixel", "-origin"])
        .arg(&binned)
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&spacing.stdout),
        "     15.140         15.140         15.140    \n    -0.0000        -0.0000        -0.0000    \n"
    );
    std::fs::remove_file(fei).unwrap();
    std::fs::remove_file(binned).unwrap();
}

#[test]
fn header_machine_readable_fields_fall_back_to_e_editing() {
    // Outside [0.1, 1.e5) the `g13.5`/`g15.5` fields switch to `Ew.d` with the
    // default scale factor, so the mantissa is written as `0.ddddd`.
    let input = copy_real_fixture("headerTest.st", "eedit");
    let mut bytes = std::fs::read(&input).unwrap();
    // MRC `amin` is at byte 76 and `amax` at byte 80.
    bytes[76..80].copy_from_slice(&1.0e-6_f32.to_le_bytes());
    bytes[80..84].copy_from_slice(&123456.0_f32.to_le_bytes());
    std::fs::write(&input, &bytes).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-minimum", "-maximum"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success());
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "  0.10000E-05\n  0.12346E+06\n"
    );
    std::fs::remove_file(input).unwrap();
}

#[test]
fn header_serialem_extended_header_pads_type_names_to_the_format_field() {
    // FORMAT 103 (`header.f90:252`) writes `typeName` from its declared
    // `character*17` field, so the names are blank padded to 17 characters.
    let input = copy_real_fixture("bidirSeries.st", "serialem");
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success());
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(
        text.ends_with(concat!(
            "\nExtended header from SerialEM contains:\n",
            "  Tilt angles       - Extract with \"extracttilts\"\n",
            "  Stage positions   - Extract with \"extracttilts -stage\"\n",
            "  Magnifications    - Extract with \"extracttilts -mag\"\n",
            "  Intensities       - Extract with \"extracttilts -int\"\n",
            "  Exposure doses    - Extract with \"extracttilts -exp\"\n"
        )),
        "{text}"
    );
    std::fs::remove_file(input).unwrap();
}

#[test]
fn header_agard_extended_header_reports_pixel_size_and_tilt_extraction() {
    // `header.f90:216` writes the nanometer pixel size with FORMAT 102's
    // `g11.4`, and line 267 asks `get_extra_header_items` for the number of
    // tilt angles it recovered rather than for its error status.
    let input = copy_real_fixture("feiHeader.st", "agard");
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success());
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(
        text.ends_with(concat!(
            "          Tilt axis rotation angle =   -24.9 (Corrected sign)\n",
            "          Pixel size in nanometers =  1.016    \n",
            "Extended header has tilt angles - extract with \"extracttilts\"\n"
        )),
        "{text}"
    );
    std::fs::remove_file(input).unwrap();
}

#[test]
fn header_brief_report_appends_the_program_blank_line() {
    // `header.f90:353` writes one more blank record per input file whenever
    // `-brief` was entered.
    let input = copy_real_fixture("headerTest.st", "briefblank");
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg("-brief")
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success());
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(text.ends_with("  Contains: Tilts\n\n"), "{text}");
    std::fs::remove_file(input).unwrap();
}

#[test]
fn header_exit_error_writes_to_stdout_and_exits_with_status_one() {
    // `exitError` (`parse_input_params.f90:231`) writes a blank record, then
    // the exit prefix and message, on the standard output unit, and exits 1.
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg("-brief")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: HEADER - No input file specified\n"
    );
    assert!(result.stderr.is_empty());

    let input = copy_real_fixture("headerTest.st", "volerr");
    let volume = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-volume", "3", "-size"])
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(volume.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&volume.stdout),
        "\nERROR: HEADER - The volume number entered is higher than the number of volumes in the file\n"
    );
    assert!(volume.stderr.is_empty());
    std::fs::remove_file(input).unwrap();
}

#[test]
fn header_option_parse_errors_match_the_pip_messages() {
    // PIP reports these on the standard output through `PipSetError`, with the
    // program exit prefix and no leading blank record, and exits 1.
    let input = copy_real_fixture("headerTest.st", "opterr");
    for (arguments, message) in [
        (
            vec!["-nosuchopt"],
            "ERROR: HEADER - Illegal option: nosuchopt\n",
        ),
        (
            vec!["-volume", "abc"],
            "ERROR: HEADER - Illegal character in value entry:  VolumeNumber  abc\n",
        ),
        (
            vec!["-tag", "abc"],
            "ERROR: HEADER - Illegal character in value entry:  TiffStringTagToPrint  abc\n",
        ),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_header"))
            .env("AUTODOC_DIR", AUTODOC)
            .args(&arguments)
            .arg(&input)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1), "{arguments:?}");
        assert_eq!(String::from_utf8_lossy(&result.stdout), message);
        assert!(result.stderr.is_empty());
    }
    let missing = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg("-input")
        .output()
        .unwrap();
    assert_eq!(missing.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&missing.stdout),
        "ERROR: HEADER - A value was expected but not found for the last option on the command line\n"
    );
    std::fs::remove_file(input).unwrap();
}

#[test]
fn header_machine_readable_switches_still_report_mdoc_values() {
    // `header.f90:277-351` runs after the silent/ordinary branch, so the mdoc
    // fallback prints even when `iiuAltPrint(0)` silenced the header report.
    let input = copy_real_fixture("headerTest.st", "silentmdoc");
    let mut bytes = std::fs::read(&input).unwrap();
    // Make the sampling intervals match the cell size so the pixel spacing is
    // 1 in all three axes and `foundPixel` stays false.
    bytes[28..32].copy_from_slice(&512_i32.to_le_bytes());
    bytes[32..36].copy_from_slice(&512_i32.to_le_bytes());
    bytes[36..40].copy_from_slice(&1_i32.to_le_bytes());
    bytes[40..44].copy_from_slice(&512.0_f32.to_le_bytes());
    bytes[44..48].copy_from_slice(&512.0_f32.to_le_bytes());
    bytes[48..52].copy_from_slice(&1.0_f32.to_le_bytes());
    std::fs::write(&input, &bytes).unwrap();
    let mdoc = std::path::PathBuf::from(format!("{}.mdoc", input.display()));
    std::fs::write(
        &mdoc,
        "PixelSpacing = 25.0\n[ZValue = 0]\nRotationAngle = 30.0\n[T = TiltAxisAngle = -120.0]\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg("-size")
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        concat!(
            "     512     512       1\n",
            "          Pixel size in nanometers =  2.500      , from mdoc\n",
            "          Tilt axis rotation angle =    30.0  (from RotationAngle in mdoc)\n"
        )
    );

    // `briefSep`, `foundPixel` and `foundAxisRot` are program-level variables
    // in `header.f90`, so their state carries into the next input file.
    let repeated = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg("-brief")
        .arg(&input)
        .arg(&input)
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&repeated.stdout);
    assert_eq!(text.matches("  Contains: Tilts\n").count(), 1, "{text}");
    assert_eq!(text.matches("\n - Tilts\n").count(), 1, "{text}");
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(mdoc).unwrap();
}

#[test]
fn header_help_prints_the_pip_usage_block_and_exits_zero() {
    // `PipReadOrParseOptions(..., .true., 1, 2, 0, ...)`
    // (`parse_input_params.f90:171`) calls `PipPrintHelp('header', 0, 2, 0)`
    // and `exit(0)`.  Verified against the native binary.
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg("-help")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(0));
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(
        text.starts_with(concat!(
            "Usage: header [Options] input_files...\n",
            "Options can be abbreviated, current short name abbreviations are in parentheses\n",
            "Options:\n",
            " -input (-i)  OR  -InputFile   File\n"
        )),
        "{text}"
    );
    assert!(text.contains(" -brief (-b)  OR  -Brief\n"), "{text}");
    assert!(
        text.ends_with(" -help (-h)  OR  -usage\n    Print help output\n"),
        "{text}"
    );
    assert!(result.stderr.is_empty());
}

#[test]
fn header_accepts_pip_option_abbreviations() {
    // PIP matches an option by any unambiguous leading substring
    // (`LookupOption`), so `-si -mo` is `-size -mode`.  Verified against the
    // native binary.
    let input = copy_real_fixture("headerTest.st", "abbrev");
    let abbreviated = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-si", "-mo"])
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(abbreviated.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&abbreviated.stdout),
        "     512     512       1\n   1\n"
    );
    let spelled = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-size", "-mode"])
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(abbreviated.stdout, spelled.stdout);
    std::fs::remove_file(input).unwrap();
}

#[test]
fn header_reports_an_ambiguous_option_abbreviation() {
    // `-m` matches both `-minimum` and `-mode`, so `LookupOption` reports it
    // through `PipSetError` and the exit prefix ends the run.  Verified against
    // the native binary.
    let input = copy_real_fixture("headerTest.st", "ambig");
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .arg("-m")
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        concat!(
            "ERROR: HEADER - An option specified by \"m\" is ambiguous between option ",
            "minimum -  Minimum  and option mode -  Mode\n"
        )
    );
    assert!(result.stderr.is_empty());
    std::fs::remove_file(input).unwrap();
}

#[test]
fn header_brief_is_boolean_so_a_following_number_becomes_an_input_file() {
    // `Brief` is declared `type = B` in `header.adoc`, so `PipNextArg` does not
    // take a value for it and `2` becomes the first non-option argument, which
    // the program then fails to open.  Verified against the native binary,
    // which prints the same two `iiOpen`/`iiuOpen` records and exits 1.
    let input = copy_real_fixture("headerTest.st", "briefvalue");
    let result = Command::new(env!("CARGO_BIN_EXE_header"))
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-brief", "2"])
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        concat!(
            "ERROR: iiOpen - Opening file 2 (No such file or directory)\n",
            "\n",
            "ERROR: iiuOpen - Could not open '2'\n"
        )
    );
    assert!(result.stderr.is_empty());
    std::fs::remove_file(input).unwrap();
}
