use imod_rs::imod::libiimod::iihdf::ii_hdf_open_new;
use imod_rs::imod::libiimod::iimage::{
    IIFILE_HDF, ii_delete, ii_open_new, ii_sync_from_mrc_header,
};
use imod_rs::imod::libiimod::mrcfiles::{MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_write};
use std::ffi::CString;
use std::process::Command;

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
        .arg("-brief")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(3));
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        "ERROR: HEADER - No input file specified\n"
    );
    assert!(result.stdout.is_empty());
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
        assert!(old_fei_text.contains("Pixel size in nanometers =     2.0000"));

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
            .arg(&mdoc_input)
            .output()
            .unwrap();
        assert!(
            mdoc_result.status.success(),
            "{}",
            String::from_utf8_lossy(&mdoc_result.stderr)
        );
        let mdoc_text = String::from_utf8_lossy(&mdoc_result.stdout);
        assert!(mdoc_text.contains("Pixel size in nanometers =     2.5000  , from mdoc"));
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
