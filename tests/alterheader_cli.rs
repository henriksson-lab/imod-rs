use imod_rs::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write,
};
use std::ffi::CString;
use std::process::Command;

#[test]
fn alterheader_rejects_copy_combined_with_another_source_option_before_opening() {
    let result = Command::new(env!("CARGO_BIN_EXE_alterheader"))
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
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "ERROR: ALTERHEADER - No other options can be entered with -copy\n"
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
            .args([
                "-org",
                "1,2,3",
                "-map",
                "3,2,1",
                "-sam",
                "8,6,4",
                "-ispg",
                "401",
                "-dat",
                "1,2,3,4,5,6",
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
            String::from_utf8_lossy(&result.stderr)
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
        assert_eq!(
            (output.idtype, output.lens, output.nd1, output.nd2),
            (1, 2, 3, 4)
        );
        assert_eq!((output.vd1, output.vd2), (500, 600));
        assert_eq!((output.amin, output.amax, output.amean), (-2.0, 9.0, 4.0));
        assert_eq!(output.nlabl, 1);
        assert_eq!(&output.labels[0][..15], b"iiunit mutation");
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
