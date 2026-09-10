use imod_rs::imod::libiimod::iimage::{ii_delete, ii_new};
use imod_rs::imod::libiimod::iitif::{IICOMPRESSION_ZIP, ii_tiff_check};
use imod_rs::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_SHORT, MrcHeader, mrc_get_scale, mrc_head_new, mrc_head_read,
    mrc_head_write,
};
use imod_rs::imod::mrc::tiff::tiff_ifd_number;
use std::ffi::CString;
use std::process::Command;

#[test]
fn mrc2tif_rejects_source_illegal_compression_before_opening_input() {
    let result = Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
        .arg("-c")
        .arg("42")
        .arg("not-opened.mrc")
        .arg("not-written.tif")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("Compression value 42 not allowed"));
}

#[test]
fn mrc2tif_uses_source_validation_diagnostic_before_opening_input() {
    let result = Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
        .args(["-o", "-P", "not-opened.mrc", "not-written.tif"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        "ERROR: mrc2tif - Resolution setting is not available with old writing code\n"
    );
}

#[test]
fn mrc2tif_malformed_option_and_non_qt_qimage_request_fail() {
    let malformed = Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
        .args(["-r", "not-a-number", "not-opened.mrc", "not-written.tif"])
        .output()
        .unwrap();
    assert!(!malformed.status.success());
    let qimage = Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
        .args(["-j", "not-opened.mrc", "not-written.jpg"])
        .output()
        .unwrap();
    assert!(!qimage.status.success());
    assert!(String::from_utf8_lossy(&qimage.stderr).contains("QImage Qt boundary"));
}

#[test]
fn mrc2tif_rejects_source_out_of_range_z_after_reading_header() {
    unsafe {
        let input =
            std::env::temp_dir().join(format!("imod-rs-mrc2tif-z-{}.mrc", std::process::id()));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 1, 1, 1, MRC_MODE_BYTE), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
            .args(["-z", "1:1"])
            .arg(&input)
            .arg(input.with_extension("tif"))
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr)
                .contains("zmin,zmax values are reversed or out of the range 0 to 0")
        );
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn mrc2tif_chunked_tiff_roundtrips_each_source_y_range() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-chunks-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}.tif"));
        let roundtrip = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let roundtrip_c = CString::new(roundtrip.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 4, 1, MRC_MODE_BYTE), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 1, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
            .arg("-T")
            .arg("2:1")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout).contains("Actual tile size = 16 x 16\n"),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        let result = Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
            .arg(&output)
            .arg(&roundtrip)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let file = libc::fopen(roundtrip_c.as_ptr(), c"rb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut header), 0);
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (2, 4, 1, MRC_MODE_BYTE)
        );
        assert_eq!(
            libc::fseek(file, header.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut written = [0_u8; 8];
        assert_eq!(
            libc::fread(written.as_mut_ptr().cast(), 1, written.len(), file),
            written.len()
        );
        libc::fclose(file);
        assert_eq!(written, pixels);
        for path in [input, output, roundtrip] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_contrast_scales_a_real_short_mrc_before_tiff_writing() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-contrast-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_SHORT), 0);
        header.amin = 0.0;
        header.amax = 300.0;
        header.amean = 150.0;
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = [0_i16, 100, 200, 300];
        assert_eq!(
            libc::fwrite(
                pixels.as_ptr().cast(),
                core::mem::size_of::<i16>(),
                pixels.len(),
                file
            ),
            pixels.len()
        );
        libc::fclose(file);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .args(["-C", "0:255"])
                .arg(&input)
                .arg(&tiff)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg(&tiff)
                .arg(&output)
                .status()
                .unwrap()
                .success()
        );
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        let mut written_header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written_header), 0);
        assert_eq!(written_header.mode, MRC_MODE_BYTE);
        assert_eq!(
            libc::fseek(file, written_header.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut written = [0_u8; 4];
        assert_eq!(
            libc::fread(written.as_mut_ptr().cast(), 1, written.len(), file),
            written.len()
        );
        libc::fclose(file);
        // `tif2mrc` stores byte-mode MRC data with the source signed-byte
        // offset, so these are the on-disk forms of TIFF values 0,85,170,255.
        assert_eq!(written, [128, 213, 42, 127]);
        for path in [input, tiff, output] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_old_writer_converts_real_mrc_pixels_to_classic_tiff() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}.tif"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
        header.amin = 1.;
        header.amax = 4.;
        header.amean = 2.5;
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = [1u8, 2, 3, 4];
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 1, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
            .arg("-o")
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let bytes = std::fs::read(&output).unwrap();
        assert_eq!(&bytes[..4], b"II*\0");
        // `mrcHeadNew` marks byte data as signed; mrcReadZ applies its source
        // byte map before the legacy writer flips the two scan lines.
        assert_eq!(&bytes[8..12], &[131, 132, 129, 130]);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn mrc2tif_writes_explicit_and_header_pixel_spacing_as_real_tiff_resolution() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-resolution-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let from_header = std::env::temp_dir().join(format!("{stamp}-header.tif"));
        let from_inches = std::env::temp_dir().join(format!("{stamp}-inches.tif"));
        let from_zero = std::env::temp_dir().join(format!("{stamp}-zero.tif"));
        let header_mrc = std::env::temp_dir().join(format!("{stamp}-header.mrc"));
        let inches_mrc = std::env::temp_dir().join(format!("{stamp}-inches.mrc"));
        let zero_mrc = std::env::temp_dir().join(format!("{stamp}-zero.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 1, 1, MRC_MODE_BYTE), 0);
        header.xlen = 10.0;
        header.ylen = 5.0;
        header.zlen = 5.0;
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        assert_eq!(libc::fwrite([11_u8, 22].as_ptr().cast(), 1, 2, file), 2);
        libc::fclose(file);

        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .arg("-P")
                .arg(&input)
                .arg(&from_header)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg("-P")
                .arg(&from_header)
                .arg(&header_mrc)
                .status()
                .unwrap()
                .success()
        );
        let header_file = CString::new(header_mrc.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(header_file.as_ptr(), c"rb".as_ptr());
        let mut read_header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut read_header), 0);
        libc::fclose(file);
        assert!((mrc_get_scale(&read_header).0 - 5.0).abs() < 0.001);

        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .args(["-r", "300"])
                .arg(&input)
                .arg(&from_inches)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg("-P")
                .arg(&from_inches)
                .arg(&inches_mrc)
                .status()
                .unwrap()
                .success()
        );
        let inches_file = CString::new(inches_mrc.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(inches_file.as_ptr(), c"rb".as_ptr());
        assert_eq!(mrc_head_read(file, &mut read_header), 0);
        libc::fclose(file);
        assert!((mrc_get_scale(&read_header).0 - 2.54e8 / 300.0).abs() < 1.0);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .args(["-r", "not-a-number"])
                .arg(&input)
                .arg(&from_zero)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg("-P")
                .arg(&from_zero)
                .arg(&zero_mrc)
                .status()
                .unwrap()
                .success()
        );
        let zero_file = CString::new(zero_mrc.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(zero_file.as_ptr(), c"rb".as_ptr());
        assert_eq!(mrc_head_read(file, &mut read_header), 0);
        libc::fclose(file);
        assert!((mrc_get_scale(&read_header).0 - 1.0).abs() < 0.001);
        for path in [
            input,
            from_header,
            from_inches,
            from_zero,
            header_mrc,
            inches_mrc,
            zero_mrc,
        ] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_uses_each_mdoc_pixel_spacing_for_numbered_tiff_output() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-mdoc-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let mdoc = std::path::PathBuf::from(format!("{}.mdoc", input.display()));
        let root = std::env::temp_dir().join(format!("{stamp}-out"));
        let first_mrc = std::env::temp_dir().join(format!("{stamp}-first.mrc"));
        let second_mrc = std::env::temp_dir().join(format!("{stamp}-second.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 1, 1, 2, MRC_MODE_BYTE), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        assert_eq!(libc::fwrite([17_u8, 34].as_ptr().cast(), 1, 2, file), 2);
        libc::fclose(file);
        std::fs::write(
            &mdoc,
            "[ZValue = 0]\nPixelSpacing = 3.0\n\n[ZValue = 1]\nPixelSpacing = 7.0\n",
        )
        .unwrap();
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .args(["-m", "-i", "42"])
                .arg(&input)
                .arg(&root)
                .status()
                .unwrap()
                .success()
        );
        let first_tif = std::path::PathBuf::from(format!("{}.042.tif", root.display()));
        let second_tif = std::path::PathBuf::from(format!("{}.043.tif", root.display()));
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg("-P")
                .arg(&first_tif)
                .arg(&first_mrc)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg("-P")
                .arg(&second_tif)
                .arg(&second_mrc)
                .status()
                .unwrap()
                .success()
        );
        let first_file = CString::new(first_mrc.as_os_str().as_encoded_bytes()).unwrap();
        let second_file = CString::new(second_mrc.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(first_file.as_ptr(), c"rb".as_ptr());
        let mut first_header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut first_header), 0);
        libc::fclose(file);
        let file = libc::fopen(second_file.as_ptr(), c"rb".as_ptr());
        let mut second_header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut second_header), 0);
        libc::fclose(file);
        assert!((mrc_get_scale(&first_header).0 - 3.0).abs() < 0.001);
        assert!((mrc_get_scale(&second_header).0 - 7.0).abs() < 0.001);
        for path in [input, mdoc, first_tif, second_tif, first_mrc, second_mrc] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_zip_stack_quality_and_slice_scaling_use_source_libtiff_paths() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-stack-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let stack = std::env::temp_dir().join(format!("{stamp}.tif"));
        let stack_mrc = std::env::temp_dir().join(format!("{stamp}-stack.mrc"));
        let scaled = std::env::temp_dir().join(format!("{stamp}-scaled"));
        let scaled_tif = std::path::PathBuf::from(format!("{}.000.tif", scaled.display()));
        let scaled_mrc = std::env::temp_dir().join(format!("{stamp}-scaled.mrc"));
        let auto = std::env::temp_dir().join(format!("{stamp}-auto"));
        let auto_tif = auto.clone();
        let auto_mrc = std::env::temp_dir().join(format!("{stamp}-auto.mrc"));
        let auto_input = std::env::temp_dir().join(format!("{stamp}-auto-input.mrc"));
        let numeric = std::env::temp_dir().join(format!("{stamp}-numeric.tif"));
        let numeric_mrc = std::env::temp_dir().join(format!("{stamp}-numeric.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 2, MRC_MODE_SHORT), 0);
        header.amin = 0.0;
        header.amax = 300.0;
        header.amean = 150.0;
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = [0_i16, 100, 200, 300, 300, 200, 100, 0];
        assert_eq!(
            libc::fwrite(
                pixels.as_ptr().cast(),
                core::mem::size_of::<i16>(),
                pixels.len(),
                file,
            ),
            pixels.len()
        );
        libc::fclose(file);

        // `zip` must select source IICOMPRESSION_ZIP (8), which in turn
        // accepts the 1..9 quality setting in `tiffWriteSetup`.
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .args(["-s", "-c", "zip", "-q", "9"])
                .arg(&input)
                .arg(&stack)
                .status()
                .unwrap()
                .success()
        );
        // The direct libtiff writer records both source sections as IFDs.
        let stack_c = CString::new(stack.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(stack_c.as_ptr(), c"rb".as_ptr());
        assert_eq!(tiff_ifd_number(file), 2);
        libc::fclose(file);
        let reader = ii_new();
        assert!(!reader.is_null());
        (*reader).filename = libc::strdup(stack_c.as_ptr());
        (*reader).fmode = [b'r' as i8, b'b' as i8, 0, 0];
        (*reader).fp = libc::fopen(stack_c.as_ptr(), c"rb".as_ptr());
        assert_eq!(ii_tiff_check(reader), 0);
        assert_eq!((*reader).tiff_compression, IICOMPRESSION_ZIP);
        assert_eq!((*reader).nz, 2);
        ii_delete(reader);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg(&stack)
                .arg(&stack_mrc)
                .status()
                .unwrap()
                .success()
        );
        let stack_mrc_c = CString::new(stack_mrc.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(stack_mrc_c.as_ptr(), c"rb".as_ptr());
        let mut stack_header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut stack_header), 0);
        assert_eq!((stack_header.nz, stack_header.mode), (2, MRC_MODE_SHORT));
        assert_eq!(
            libc::fseek(file, stack_header.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut stack_pixels = [0_i16; 8];
        assert_eq!(
            libc::fread(
                stack_pixels.as_mut_ptr().cast(),
                core::mem::size_of::<i16>(),
                stack_pixels.len(),
                file,
            ),
            stack_pixels.len()
        );
        libc::fclose(file);
        assert_eq!(stack_pixels, pixels);

        // `-S`, unlike `-C`, gives explicit data min/max limits before the
        // source linear byte conversion.  Values below/above 100..200 clamp.
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .args(["-z", "0:0", "-S", "100:200"])
                .arg(&input)
                .arg(&scaled)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg(&scaled_tif)
                .arg(&scaled_mrc)
                .status()
                .unwrap()
                .success()
        );
        let scaled_c = CString::new(scaled_mrc.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(scaled_c.as_ptr(), c"rb".as_ptr());
        let mut scaled_header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut scaled_header), 0);
        assert_eq!(scaled_header.mode, MRC_MODE_BYTE);
        assert_eq!(
            libc::fseek(file, scaled_header.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut scaled_pixels = [0_u8; 4];
        assert_eq!(
            libc::fread(
                scaled_pixels.as_mut_ptr().cast(),
                1,
                scaled_pixels.len(),
                file
            ),
            scaled_pixels.len()
        );
        libc::fclose(file);
        // MRC signed-byte storage is the TIFF [0, 0, 255, 255] result shifted by 128.
        assert_eq!(scaled_pixels, [128, 128, 127, 127]);

        // Auto-contrast requires the source sampling minimum of five pixels.
        let auto_input_c = CString::new(auto_input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(auto_input_c.as_ptr(), c"wb".as_ptr());
        let mut auto_input_header: MrcHeader = core::mem::zeroed();
        assert_eq!(
            mrc_head_new(&mut auto_input_header, 3, 2, 1, MRC_MODE_SHORT),
            0
        );
        auto_input_header.amin = 0.0;
        auto_input_header.amax = 500.0;
        auto_input_header.amean = 250.0;
        auto_input_header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut auto_input_header), 0);
        let auto_input_pixels = [0_i16, 100, 200, 300, 400, 500];
        assert_eq!(
            libc::fwrite(
                auto_input_pixels.as_ptr().cast(),
                core::mem::size_of::<i16>(),
                auto_input_pixels.len(),
                file,
            ),
            auto_input_pixels.len()
        );
        libc::fclose(file);
        // Auto-contrast computes the source sample mean/SD per section and
        // converts the real slice to byte values before libtiff writing.
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .args(["-z", "0:0", "-a", "200:10"])
                .arg(&auto_input)
                .arg(&auto)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg(&auto_tif)
                .arg(&auto_mrc)
                .status()
                .unwrap()
                .success()
        );
        let auto_c = CString::new(auto_mrc.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(auto_c.as_ptr(), c"rb".as_ptr());
        let mut auto_header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut auto_header), 0);
        assert_eq!(auto_header.mode, MRC_MODE_BYTE);
        assert_eq!(
            libc::fseek(file, auto_header.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut auto_pixels = [0_u8; 6];
        assert_eq!(
            libc::fread(auto_pixels.as_mut_ptr().cast(), 1, auto_pixels.len(), file),
            auto_pixels.len()
        );
        libc::fclose(file);
        assert_eq!(auto_pixels, [58, 63, 69, 74, 80, 85]);

        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .args(["-O", "0", "-c", "32946"])
                .arg(&auto_input)
                .arg(&numeric)
                .status()
                .unwrap()
                .success()
        );
        let numeric_c = CString::new(numeric.as_os_str().as_encoded_bytes()).unwrap();
        let reader = ii_new();
        assert!(!reader.is_null());
        (*reader).filename = libc::strdup(numeric_c.as_ptr());
        (*reader).fmode = [b'r' as i8, b'b' as i8, 0, 0];
        (*reader).fp = libc::fopen(numeric_c.as_ptr(), c"rb".as_ptr());
        assert_eq!(ii_tiff_check(reader), 0);
        assert_eq!((*reader).tiff_compression, 32946);
        ii_delete(reader);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg(&numeric)
                .arg(&numeric_mrc)
                .status()
                .unwrap()
                .success()
        );
        let numeric_mrc_c = CString::new(numeric_mrc.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(numeric_mrc_c.as_ptr(), c"rb".as_ptr());
        let mut numeric_header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut numeric_header), 0);
        assert_eq!(
            libc::fseek(file, numeric_header.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut numeric_pixels = [0_i16; 6];
        assert_eq!(
            libc::fread(
                numeric_pixels.as_mut_ptr().cast(),
                core::mem::size_of::<i16>(),
                numeric_pixels.len(),
                file,
            ),
            numeric_pixels.len()
        );
        libc::fclose(file);
        assert_eq!(numeric_pixels, auto_input_pixels);
        for path in [
            input,
            stack,
            stack_mrc,
            scaled_tif,
            scaled_mrc,
            auto_input,
            auto_tif,
            auto_mrc,
            numeric,
            numeric_mrc,
        ] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn mrc2tif_jpeg_compression_uses_the_installed_libtiff_codec() {
    unsafe {
        let stamp = format!("imod-rs-mrc2tif-jpeg-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let jpeg = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 4, 4, 1, MRC_MODE_BYTE), 0);
        header.amin = 0.0;
        header.amax = 255.0;
        header.amean = 127.5;
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = [
            128_u8, 144, 160, 176, 192, 208, 224, 240, 129, 145, 161, 177, 193, 209, 225, 241,
        ];
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 1, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .args(["-O", "0", "-c", "jpeg", "-q", "100"])
                .arg(&input)
                .arg(&jpeg)
                .status()
                .unwrap()
                .success()
        );
        let jpeg_c = CString::new(jpeg.as_os_str().as_encoded_bytes()).unwrap();
        let reader = ii_new();
        assert!(!reader.is_null());
        (*reader).filename = libc::strdup(jpeg_c.as_ptr());
        (*reader).fmode = [b'r' as i8, b'b' as i8, 0, 0];
        (*reader).fp = libc::fopen(jpeg_c.as_ptr(), c"rb".as_ptr());
        assert_eq!(ii_tiff_check(reader), 0);
        assert_eq!((*reader).tiff_compression, 7);
        ii_delete(reader);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg(&jpeg)
                .arg(&output)
                .status()
                .unwrap()
                .success()
        );
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        let mut output_header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut output_header), 0);
        libc::fclose(file);
        assert_eq!(
            (output_header.nx, output_header.ny, output_header.mode),
            (4, 4, MRC_MODE_BYTE)
        );
        for path in [input, jpeg, output] {
            std::fs::remove_file(path).unwrap();
        }
    }
}
