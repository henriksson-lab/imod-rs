use imod_rs::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_RGB, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write,
};
use std::ffi::CString;
use std::process::Command;

#[test]
fn tif2mrc_roundtrips_the_native_legacy_tiff_path() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
        header.amin = 1.;
        header.amax = 4.;
        header.amean = 2.5;
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = [1_u8, 2, 3, 4];
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 1, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .arg("-o")
                .arg(&input)
                .arg(&tiff)
                .status()
                .unwrap()
                .success()
        );
        let result = Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .contains(&format!("Opening {} for input\n", tiff.display()))
        );
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut written: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written), 0);
        assert_eq!(
            (written.nx, written.ny, written.nz, written.mode),
            (2, 2, 1, MRC_MODE_BYTE)
        );
        let mut bytes = [0_u8; 4];
        assert_eq!(
            libc::fseek(file, written.header_size as i64, libc::SEEK_SET),
            0
        );
        assert_eq!(
            libc::fread(bytes.as_mut_ptr().cast(), 1, bytes.len(), file),
            bytes.len()
        );
        libc::fclose(file);
        assert_eq!(bytes, pixels);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(tiff).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn tif2mrc_converts_real_rgb_tiff_to_source_average_grayscale() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-rgb-{}", std::process::id());
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let tiff_c = CString::new(tiff.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(tiff_c.as_ptr(), c"wb".as_ptr());
        let mut ifd = 0;
        let mut data_offset = 0;
        let rgb = [30_u8, 60, 90, 9, 12, 15];
        assert_eq!(
            imod_rs::imod::mrc::tiff::tiff_write_image(
                file,
                2,
                1,
                MRC_MODE_RGB,
                rgb.as_ptr().cast_mut(),
                &mut ifd,
                &mut data_offset,
                0.,
                255.,
            ),
            0
        );
        let rgb_second = [0_u8, 3, 6, 30, 33, 36];
        assert_eq!(
            imod_rs::imod::mrc::tiff::tiff_write_image(
                file,
                2,
                1,
                MRC_MODE_RGB,
                rgb_second.as_ptr().cast_mut(),
                &mut ifd,
                &mut data_offset,
                0.,
                255.,
            ),
            0
        );
        libc::fclose(file);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg("-g")
                .arg(&tiff)
                .arg(&output)
                .status()
                .unwrap()
                .success()
        );
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        let mut written: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written), 0);
        assert_eq!(
            (written.nx, written.ny, written.nz, written.mode),
            (2, 1, 2, MRC_MODE_BYTE)
        );
        assert_eq!(
            libc::fseek(file, written.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut gray = [0_u8; 4];
        assert_eq!(
            libc::fread(gray.as_mut_ptr().cast(), 1, gray.len(), file),
            gray.len()
        );
        libc::fclose(file);
        assert_eq!(gray, [188, 140, 131, 161]);
        std::fs::remove_file(tiff).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn tif2mrc_converts_signed_32_bit_tiff_through_the_source_float_path() {
    let stamp = format!("imod-rs-tif2mrc-signed32-{}", std::process::id());
    let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
    let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
    let mut image = vec![0_u8; 162];
    image[..8].copy_from_slice(b"II*\0\x08\0\0\0");
    image[8..10].copy_from_slice(&11_u16.to_le_bytes());
    for (entry, (tag, kind, count, value)) in [
        (256_u16, 4_u16, 1_u32, 2_u32),
        (257, 4, 1, 2),
        (258, 3, 1, 32),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (273, 4, 1, 146),
        (277, 3, 1, 1),
        (278, 4, 1, 2),
        (279, 4, 1, 16),
        (284, 3, 1, 1),
        (339, 3, 1, 2),
    ]
    .into_iter()
    .enumerate()
    {
        let start = 10 + entry * 12;
        image[start..start + 2].copy_from_slice(&tag.to_le_bytes());
        image[start + 2..start + 4].copy_from_slice(&kind.to_le_bytes());
        image[start + 4..start + 8].copy_from_slice(&count.to_le_bytes());
        image[start + 8..start + 12].copy_from_slice(&value.to_le_bytes());
    }
    image[146..150].copy_from_slice(&(-1_i32).to_le_bytes());
    image[150..154].copy_from_slice(&2_i32.to_le_bytes());
    image[154..158].copy_from_slice(&(-300_000_000_i32).to_le_bytes());
    image[158..162].copy_from_slice(&(400_000_000_i32).to_le_bytes());
    std::fs::write(&tiff, image).unwrap();
    let conversion = Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
        .arg(&tiff)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        conversion.status.success(),
        "{}",
        String::from_utf8_lossy(&conversion.stderr)
    );
    unsafe {
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (2, 2, 1, 2));
        assert_eq!(
            libc::fseek(file, header.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut values = [0_f32; 4];
        assert_eq!(
            libc::fread(values.as_mut_ptr().cast(), 4, values.len(), file),
            values.len()
        );
        libc::fclose(file);
        assert_eq!(values, [-300_000_000., 400_000_000., -1., 2.]);
    }
    std::fs::remove_file(tiff).unwrap();
    std::fs::remove_file(output).unwrap();
}

#[test]
fn tif2mrc_reads_every_directory_in_a_native_legacy_tiff_stack() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-stack-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 2, MRC_MODE_BYTE), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 1, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .arg("-o")
                .arg("-s")
                .arg(&input)
                .arg(&tiff)
                .status()
                .unwrap()
                .success()
        );
        std::fs::write(&output, b"pre-existing multi-page output").unwrap();
        let conversion = Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
            .args(["-o", "1,1"])
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            conversion.status.success(),
            "{}",
            String::from_utf8_lossy(&conversion.stderr)
        );
        let stdout = String::from_utf8_lossy(&conversion.stdout);
        assert!(
            stdout.contains("Reading multi-paged TIFF file."),
            "{stdout}"
        );
        assert!(
            stdout.contains("Warning: output file size option ignored for multi-paged file"),
            "{stdout}"
        );
        assert!(
            stdout.contains("Converting 2 images size 2 x 2"),
            "{stdout}"
        );
        assert!(
            stdout.contains("Min = 129, Max = 136, Mean = 132.5"),
            "{stdout}"
        );
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        let mut written: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written), 0);
        assert_eq!((written.nx, written.ny, written.nz), (2, 2, 2));
        assert_eq!(
            libc::fseek(file, written.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut bytes = [0_u8; 8];
        assert_eq!(
            libc::fread(bytes.as_mut_ptr().cast(), 1, bytes.len(), file),
            bytes.len()
        );
        libc::fclose(file);
        assert_eq!(bytes, pixels);
        let backup = std::path::PathBuf::from(format!("{}~", output.display()));
        assert_eq!(
            std::fs::read(&backup).unwrap(),
            b"pre-existing multi-page output"
        );
        for path in [input, tiff, output, backup] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn tif2mrc_applies_source_background_inversion_and_subtraction() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-bg-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let background = std::env::temp_dir().join(format!("{stamp}-bg.mrc"));
        let input_tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let background_tiff = std::env::temp_dir().join(format!("{stamp}-bg.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        for (path, pixels) in [
            (&input, [10_u8, 11, 12, 13]),
            (&background, [1_u8, 2, 3, 4]),
        ] {
            let name = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
            let mut header: MrcHeader = core::mem::zeroed();
            assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
            header.fp = file.cast();
            assert_eq!(mrc_head_write(file, &mut header), 0);
            assert_eq!(
                libc::fwrite(pixels.as_ptr().cast(), 1, pixels.len(), file),
                pixels.len()
            );
            libc::fclose(file);
        }
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .arg("-o")
                .arg(&input)
                .arg(&input_tiff)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_mrc2tif"))
                .arg("-o")
                .arg(&background)
                .arg(&background_tiff)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
                .arg("-b")
                .arg(&background_tiff)
                .arg(&input_tiff)
                .arg(&output)
                .status()
                .unwrap()
                .success()
        );
        let output_name = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(output_name.as_ptr(), c"rb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut header), 0);
        assert_eq!(
            libc::fseek(file, header.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut pixels = [0_u8; 4];
        assert_eq!(
            libc::fread(pixels.as_mut_ptr().cast(), 1, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        assert_eq!(pixels, [13, 13, 13, 13]);
        for path in [input, background, input_tiff, background_tiff, output] {
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn tif2mrc_reports_source_specific_pixel_spacing_conflict() {
    let stamp = format!("imod-rs-tif2mrc-pixel-conflict-{}", std::process::id());
    let missing_input = std::env::temp_dir().join(format!("{stamp}.tif"));
    let output = std::env::temp_dir().join(format!("{stamp}.mrc"));
    let result = Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
        .args([
            "-p",
            "1.5",
            "-P",
            missing_input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stdout).contains("You cannot enter both -p and -P"),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(!output.exists());
}

#[test]
fn tif2mrc_rejects_background_for_real_multi_directory_tiff() {
    unsafe {
        let stamp = format!(
            "imod-rs-tif2mrc-multipage-background-{}",
            std::process::id()
        );
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let name = CString::new(tiff.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut ifd = 0;
        let mut data_offset = 0;
        for pixels in [[1_u8, 2], [3, 4]] {
            assert_eq!(
                imod_rs::imod::mrc::tiff::tiff_write_image(
                    file,
                    2,
                    1,
                    MRC_MODE_BYTE,
                    pixels.as_ptr().cast_mut(),
                    &mut ifd,
                    &mut data_offset,
                    0.,
                    255.,
                ),
                0
            );
        }
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
            .args(["-b", tiff.to_str().unwrap()])
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(!result.status.success());
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            stdout.contains("Reading multi-paged TIFF file."),
            "{stdout}"
        );
        assert!(
            stdout.contains("Background subtraction not supported for multi-paged images."),
            "{stdout}"
        );
        assert!(!output.exists());
        std::fs::remove_file(tiff).unwrap();
    }
}

#[test]
fn tif2mrc_accepts_negative_chunk_criterion_for_real_legacy_tiff() {
    unsafe {
        let stamp = format!("imod-rs-tif2mrc-negative-chunks-{}", std::process::id());
        let tiff = std::env::temp_dir().join(format!("{stamp}.tif"));
        let output = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let tiff_c = CString::new(tiff.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(tiff_c.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut ifd = 0;
        let mut data_offset = 0;
        let pixels = [5_u8, 8];
        assert_eq!(
            imod_rs::imod::mrc::tiff::tiff_write_image(
                file,
                2,
                1,
                MRC_MODE_BYTE,
                pixels.as_ptr().cast_mut(),
                &mut ifd,
                &mut data_offset,
                0.,
                255.,
            ),
            0
        );
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_tif2mrc"))
            .args(["-t", "-1"])
            .arg(&tiff)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut header), 0);
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (2, 1, 1, MRC_MODE_BYTE)
        );
        assert_ne!(header.bytes_signed, 0);
        assert_eq!(
            libc::fseek(file, header.header_size as i64, libc::SEEK_SET),
            0
        );
        let mut written = [0_u8; 2];
        assert_eq!(
            libc::fread(written.as_mut_ptr().cast(), 1, written.len(), file),
            written.len()
        );
        libc::fclose(file);
        // Source byte-mode output is signed and therefore shifts the raw
        // storage by 128 while preserving logical image values 5 and 8.
        assert_eq!(written, [133, 136]);
        std::fs::remove_file(tiff).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}
