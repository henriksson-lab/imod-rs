use imod_rs::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write,
};
use std::ffi::CString;
use std::process::Command;

/// Every PIP-driven invocation needs an autodoc directory, exactly as a real
/// IMOD install provides one through `AUTODOC_DIR` or `IMOD_DIR`.
const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

#[test]
fn alterheader_rejects_copy_combined_with_another_source_option_before_opening() {
    let result = Command::new(env!("CARGO_BIN_EXE_alterheader"))
        .env("AUTODOC_DIR", AUTODOC)
        .args([
            "-copy",
            "not-opened-reference.mrc",
            "-org",
            "1,2,3",
            "not-opened-target.mrc",
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    // `exitError` (`parse_input_params.f90:231`) writes `write(*,'(/,a,a,a)')`,
    // so the message is preceded by a blank record.  Verified against native.
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: ALTERHEADER - No other options can be entered with -copy\n"
    );
    assert!(result.stderr.is_empty());
}

#[test]
fn alterheader_persists_iiunit_origin_map_sample_mode_space_group_and_labels() {
    unsafe {
        let input = std::env::temp_dir().join(format!(
            "imod-rs-alterheader-cli-{}.mrc",
            std::process::id()
        ));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 4, 3, 2, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_alterheader"))
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-org",
                "1,2,3",
                "-map",
                "3,2,1",
                "-sam",
                "8,6,4",
                "-ispg",
                "401",
                "-setmmm",
                "-2,9,4",
                "-title",
                "iiunit mutation",
            ])
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(stdout.contains("OLD image file on unit   2"), "{stdout}");
        assert!(stdout.contains("RO image file on unit   3"), "{stdout}");
        assert!(
            stdout.contains("Number of columns, rows, sections"),
            "{stdout}"
        );
        let file = libc::fopen(name.as_ptr(), c"rb".as_ptr());
        let mut output: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut output), 0);
        libc::fclose(file);
        assert_eq!((output.xorg, output.yorg, output.zorg), (1.0, 2.0, 3.0));
        assert_eq!((output.mapc, output.mapr, output.maps), (3, 2, 1));
        assert_eq!((output.mx, output.my, output.mz), (8, 6, 4));
        assert_eq!(output.ispg, 401);
        assert_eq!((output.amin, output.amax, output.amean), (-2.0, 9.0, 4.0));
        assert_eq!(output.nlabl, 1);
        assert_eq!(&output.labels[0][..15], b"iiunit mutation");
        // `mrc_head_write` stamps the IMOD signature at offset 152; native
        // writes the same four bytes for this run.
        assert_eq!(output.imod_stamp, 0x444f_4d49);
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn alterheader_inserts_title_at_source_one_based_position() {
    unsafe {
        let input = std::env::temp_dir().join(format!(
            "imod-rs-alterheader-title-position-{}.mrc",
            std::process::id()
        ));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        header.nlabl = 2;
        header.labels[0][..5].copy_from_slice(b"first");
        header.labels[1][..6].copy_from_slice(b"second");
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_alterheader"))
            .env("AUTODOC_DIR", AUTODOC)
            .args(["-position", "2", "-title", "inserted"])
            .arg(&input)
            .output()
            .unwrap();
        assert!(result.status.success(), "{:?}", result);
        let file = libc::fopen(name.as_ptr(), c"rb".as_ptr());
        let mut output: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut output), 0);
        libc::fclose(file);
        assert_eq!(output.nlabl, 3);
        assert_eq!(&output.labels[0][..5], b"first");
        assert_eq!(&output.labels[1][..8], b"inserted");
        assert_eq!(&output.labels[2][..6], b"second");
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn alterheader_modefix_reports_source_mode_conversion_and_range_warning() {
    unsafe {
        let input = std::env::temp_dir().join(format!(
            "imod-rs-alterheader-modefix-{}.mrc",
            std::process::id()
        ));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, 1), 0);
        header.amin = -4.0;
        header.amax = 12.0;
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_alterheader"))
            .env("AUTODOC_DIR", AUTODOC)
            .args(["-modefix", input.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(result.status.success(), "{:?}", result);
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(stdout.contains("\nChanging mode to 6\n"), "{stdout}");
        assert!(
            stdout.contains(
                "\nThe file minimum is        -4.0 and negative numbers will not be\n represented correctly in this mode.\n"
            ),
            "{stdout}"
        );
        let file = libc::fopen(name.as_ptr(), c"rb".as_ptr());
        let mut changed = core::mem::zeroed::<MrcHeader>();
        assert_eq!(mrc_head_read(file, &mut changed), 0);
        libc::fclose(file);
        assert_eq!(changed.mode, 6);
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn alterheader_runs_the_interactive_option_loop_from_piped_input() {
    // `alterheader.f90:92-206`: with no PIP options the program prints the
    // option menu, prompts, and dispatches each typed keyword.  Every line
    // asserted here was taken from the native binary run on the same stack.
    unsafe {
        let input = std::env::temp_dir().join(format!(
            "imod-rs-alterheader-interactive-{}.mrc",
            std::process::id()
        ));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 4, 3, 2, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);

        let mut child = Command::new(env!("CARGO_BIN_EXE_alterheader"))
            .env("AUTODOC_DIR", AUTODOC)
            .arg(&input)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        use std::io::Write;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"org\n1,2,3\ndel\n2,3,4\ndone\n")
            .unwrap();
        let result = child.wait_with_output().unwrap();
        assert_eq!(result.status.code(), Some(0));
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            stdout.contains(
                " If you make a mistake, interrupt with Ctrl-C instead of exiting with DONE\n"
            ),
            "{stdout}"
        );
        assert!(
            stdout.contains(
                " Options: org, cel, dat, del, map, sam, tlt, tlt_orig, tlt_rot, lab, mmm,\n rms, fixpixel, feipixel, fixextra, fixmode, invertorg, setmmm, real, fft,\n ispg, volstack, 4bit, toggleorg, fixgrid, start, help, OR done\n Enter option: "
            ),
            "{stdout}"
        );
        assert!(
            stdout.contains(
                " Enter option:  Alter origin.  The origin is the offset FROM the first point in the image\n file TO the center of the coordinate system, expressed in true coordinates.\n Current x, y, z:    0.0000        0.0000        0.0000    \nNew x, y, z: "
            ),
            "{stdout}"
        );
        assert!(
            stdout.contains(
                " Enter option:  Alter delta - changes cell sizes to achieve desired pixel spacing\n Current delta x, y, z:    1.0000        1.0000        1.0000    \nNew delta x, y, z: "
            ),
            "{stdout}"
        );
        // Label 15 reopens the file on unit 3 and reprints the header.
        assert!(stdout.contains(" RO image file on unit   3 : "), "{stdout}");

        let file = libc::fopen(name.as_ptr(), c"rb".as_ptr());
        let mut output: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut output), 0);
        libc::fclose(file);
        assert_eq!((output.xorg, output.yorg, output.zorg), (1.0, 2.0, 3.0));
        // `del` rewrites the cell as mxyz * delta (`alterheader.f90:340`).
        assert_eq!((output.xlen, output.ylen, output.zlen), (8.0, 9.0, 8.0));
        assert_eq!((output.mx, output.my, output.mz), (4, 3, 2));
        // `mrc_head_write` stamps "IMOD" at offset 152, as native does.
        assert_eq!(output.imod_stamp, 0x444f_4d49);
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn alterheader_interactive_rejects_an_unknown_keyword_and_reprompts() {
    unsafe {
        let input = std::env::temp_dir().join(format!(
            "imod-rs-alterheader-bogus-{}.mrc",
            std::process::id()
        ));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let mut child = Command::new(env!("CARGO_BIN_EXE_alterheader"))
            .env("AUTODOC_DIR", AUTODOC)
            .arg(&input)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        use std::io::Write;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"bogus\ndone\n")
            .unwrap();
        let result = child.wait_with_output().unwrap();
        assert_eq!(result.status.code(), Some(0));
        let stdout = String::from_utf8_lossy(&result.stdout);
        // `print *,'Not a legal entry, try again'` (`alterheader.f90:217`).
        assert!(
            stdout.contains(" Enter option:  Not a legal entry, try again\n"),
            "{stdout}"
        );
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn alterheader_falls_back_to_the_program_option_table_without_an_autodoc() {
    // Native behaves identically with no autodoc reachable: it prints the
    // padded `character*240` PIP warning, announces the fallback table, and
    // then dies on `VolumeStack`, which that table does not contain.
    let input = std::env::temp_dir().join(format!(
        "imod-rs-alterheader-fallback-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
    }
    let result = Command::new(env!("CARGO_BIN_EXE_alterheader"))
        .env_remove("AUTODOC_DIR")
        .env_remove("IMOD_DIR")
        .args(["-org", "1,2,3"])
        .arg(&input)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&result.stderr), "");
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.starts_with(
            "PIP WARNING: Autodoc file alterheader.adoc was not found or not readable.\nCheck environment variable settings of AUTODOC_DIR and IMOD_DIR\nor place autodoc file in current directory"
        ),
        "{stdout}"
    );
    assert!(
        stdout.contains("\n Using fallback options in main program\n"),
        "{stdout}"
    );
    assert!(
        stdout.ends_with("\nERROR: ALTERHEADER - Illegal option: VolumeStack\n"),
        "{stdout}"
    );
    std::fs::remove_file(input).unwrap();
}

#[test]
fn alterheader_rejects_the_interactive_only_data_type_keyword_as_an_option() {
    // `DAT` has no PIP field in `alterheader.adoc`, so native reports it as an
    // illegal option before opening anything.
    let result = Command::new(env!("CARGO_BIN_EXE_alterheader"))
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-dat", "1,2,3,4,5,6", "no-such-file.mrc"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "ERROR: ALTERHEADER - Illegal option: dat\n"
    );
}
