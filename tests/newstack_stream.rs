use imod_rs::imod::libcfshr::cubinterp::cubinterp;
use imod_rs::imod::libcfshr::linearxforms::xfmult;
use imod_rs::imod::libiimod::iimage::{
    IIFILE_DEFAULT, IIFILE_TIFF, ii_close, ii_fill_mrc_header, ii_open, ii_open_new,
    ii_read_section_float, ii_sync_from_mrc_header, ii_write_section_float,
};
use imod_rs::imod::libiimod::mrcfiles::{
    MrcHeader, mrc_head_new, mrc_head_write, mrc_write_extra_header,
};
use std::ffi::CString;
use std::process::Command;

#[test]
fn newstack_streams_real_mrc_sections() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 2, 2), 0);
        (*header).nlabl = 1;
        (&mut (*header).labels[0])[..24].copy_from_slice(b"acquisition source label");
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut first = [1.0_f32, 2.0, 3.0, 4.0];
        let mut second = [5.0_f32, 6.0, 7.0, 8.0];
        assert_eq!(
            ii_write_section_float(file, first.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(
            ii_write_section_float(file, second.as_mut_ptr().cast(), 1),
            0
        );
        ii_close(file);
    }
    unsafe {
        let file = ii_open(input_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 1),
            0
        );
        assert_eq!(pixels, [5., 6., 7., 8.]);
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-secs",
                "1",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz), (2, 2, 1));
        assert_eq!(header.nlabl, 2);
        assert_eq!(&header.labels[0][..24], b"acquisition source label");
        assert_eq!(&header.labels[1][..23], b"NEWSTACK: Images copied");
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [5., 6., 7., 8.]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn newstack_reorders_real_tilt_stack_from_explicit_angle_file() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-reorder-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let angles = base.with_extension("angles.txt");
    let reordered = base.with_extension("reordered.txt");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 1, 1, 4, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for (section, value) in [10.0_f32, 20.0, 30.0, 40.0].into_iter().enumerate() {
            let mut pixel = [value];
            assert_eq!(
                ii_write_section_float(file, pixel.as_mut_ptr().cast(), section as i32),
                0
            );
        }
        ii_close(file);
    }
    std::fs::write(&angles, "10\n-20\n0\n20\n").unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-reorder",
                "1",
                "-angle",
                angles.to_str().unwrap(),
                "-newangle",
                reordered.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        for (section, expected) in [20.0_f32, 30.0, 10.0, 40.0].into_iter().enumerate() {
            let mut pixel = [f32::NAN];
            assert_eq!(
                ii_read_section_float(file, pixel.as_mut_ptr().cast(), section as i32),
                0
            );
            assert_eq!(pixel, [expected]);
        }
        ii_close(file);
    }
    assert_eq!(
        std::fs::read_to_string(&reordered).unwrap(),
        "   -20.00\n     0.00\n    10.00\n    20.00\n"
    );
    for path in [input, output, angles, reordered] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_inserts_tilts_as_generic_mrc_extended_header_reals() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-tilt-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let angles = base.with_extension("angles.txt");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 1, 1, 2, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for (section, value) in [10.0_f32, 20.0].into_iter().enumerate() {
            let mut pixel = [value];
            assert_eq!(
                ii_write_section_float(file, pixel.as_mut_ptr().cast(), section as i32),
                0
            );
        }
        ii_close(file);
    }
    std::fs::write(&angles, "-30.5\n12.25\n").unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-tilt",
                angles.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.next, header.nint, header.nreal), (8, 0, 1));
        ii_close(file);
    }
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(bytes.len(), 1024 + 8 + 8);
    assert_eq!(
        [
            f32::from_le_bytes(bytes[1024..1028].try_into().unwrap()),
            f32::from_le_bytes(bytes[1028..1032].try_into().unwrap()),
        ],
        [-30.5, 12.25]
    );
    for path in [input, output, angles] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_replaces_generic_extended_header_tilts_for_selected_sections() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-newstack-existing-tilts-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let angles = base.with_extension("angles.txt");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 1, 1, 3, 2), 0);
        (*header).nint = 0;
        (*header).nreal = 1;
        (*header).next = 12;
        (*header).header_size = 1036;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut initial_angles = [-60.0_f32, 0.0, 60.0];
        assert_eq!(
            mrc_write_extra_header(header, initial_angles.as_mut_ptr().cast(), 12),
            0
        );
        for (section, value) in [10.0_f32, 20.0, 30.0].into_iter().enumerate() {
            let mut pixel = [value];
            assert_eq!(
                ii_write_section_float(file, pixel.as_mut_ptr().cast(), section as i32),
                0
            );
        }
        ii_close(file);
    }
    std::fs::write(&angles, "-4.5\n18.25\n").unwrap();
    let native_output = base.with_extension("native-output.mrc");
    if let Ok(native_newstack) = std::env::var("IMOD_NATIVE_NEWSTACK") {
        assert!(
            Command::new(native_newstack)
                .env(
                    "AUTODOC_DIR",
                    std::env::var("AUTODOC_DIR")
                        .unwrap_or_else(|_| "/tmp/imod-reference-build/autodoc".into()),
                )
                .args([
                    "-input",
                    input.to_str().unwrap(),
                    "-output",
                    native_output.to_str().unwrap(),
                    "-secs",
                    "2,0",
                    "-tilt",
                    angles.to_str().unwrap(),
                ])
                .status()
                .unwrap()
                .success()
        );
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-secs",
                "2,0",
                "-tilt",
                angles.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success()
    );
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(bytes.len(), 1024 + 8 + 8);
    assert_eq!(
        [
            f32::from_le_bytes(bytes[1024..1028].try_into().unwrap()),
            f32::from_le_bytes(bytes[1028..1032].try_into().unwrap()),
        ],
        [-4.5, 18.25]
    );
    assert_eq!(
        [
            f32::from_le_bytes(bytes[1032..1036].try_into().unwrap()),
            f32::from_le_bytes(bytes[1036..1040].try_into().unwrap()),
        ],
        [30.0, 10.0]
    );
    if native_output.exists() {
        let native = std::fs::read(&native_output).unwrap();
        assert_eq!(&native[92..100], &bytes[92..100]);
        assert_eq!(&native[128..132], &bytes[128..132]);
        assert_eq!(&native[1024..], &bytes[1024..]);
    }
    for path in [input, output, native_output, angles] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_replaces_selected_serialem_extended_header_tilts() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-newstack-serialem-tilts-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let angles = base.with_extension("angles.txt");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 1, 1, 3, 2), 0);
        // SerialEM represents these fields as bytes per section and flags.
        // Bit 0 says that the leading short is a tilt angle in centidegrees.
        (*header).nint = 8;
        (*header).nreal = 3;
        (*header).next = 24;
        (*header).header_size = 1048;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut serialem = [0_u8; 24];
        for (record, angle) in [-6000_i16, 0, 6000].into_iter().enumerate() {
            serialem[8 * record..8 * record + 2].copy_from_slice(&angle.to_le_bytes());
            serialem[8 * record + 2..8 * record + 8].copy_from_slice(&[record as u8 + 1; 6]);
        }
        assert_eq!(mrc_write_extra_header(header, serialem.as_mut_ptr(), 24), 0);
        for (section, value) in [10.0_f32, 20.0, 30.0].into_iter().enumerate() {
            let mut pixel = [value];
            assert_eq!(
                ii_write_section_float(file, pixel.as_mut_ptr().cast(), section as i32),
                0
            );
        }
        ii_close(file);
    }
    std::fs::write(&angles, "-4.505\n18.255\n").unwrap();
    let native_output = base.with_extension("native-output.mrc");
    if let Ok(native_newstack) = std::env::var("IMOD_NATIVE_NEWSTACK") {
        assert!(
            Command::new(native_newstack)
                .env(
                    "AUTODOC_DIR",
                    std::env::var("AUTODOC_DIR")
                        .unwrap_or_else(|_| "/tmp/imod-reference-build/autodoc".into()),
                )
                .args([
                    "-input",
                    input.to_str().unwrap(),
                    "-output",
                    native_output.to_str().unwrap(),
                    "-secs",
                    "2,0",
                    "-tilt",
                    angles.to_str().unwrap(),
                ])
                .status()
                .unwrap()
                .success()
        );
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-secs",
                "2,0",
                "-tilt",
                angles.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success()
    );
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(bytes.len(), 1024 + 16 + 8);
    assert_eq!(
        i16::from_le_bytes(bytes[1024..1026].try_into().unwrap()),
        -451
    );
    assert_eq!(
        i16::from_le_bytes(bytes[1032..1034].try_into().unwrap()),
        1825
    );
    assert_eq!(&bytes[1026..1032], &[3; 6]);
    assert_eq!(&bytes[1034..1040], &[1; 6]);
    if native_output.exists() {
        let native = std::fs::read(&native_output).unwrap();
        assert_eq!(&native[92..100], &bytes[92..100]);
        assert_eq!(&native[128..132], &bytes[128..132]);
        assert_eq!(&native[1024..], &bytes[1024..]);
    }
    for path in [input, output, native_output, angles] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_format_of_output_file_writes_native_tiff() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-format-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.tif");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 1, 1, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixel = [7_f32];
        assert_eq!(
            ii_write_section_float(file, pixel.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-format",
                "TIFF"
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert_eq!((*file).file, IIFILE_TIFF);
        let mut pixel = [f32::NAN];
        assert_eq!(ii_read_section_float(file, pixel.as_mut_ptr().cast(), 0), 0);
        assert_eq!(pixel, [7.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_bytes_signed_output_option_controls_real_byte_mrc_header() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-bytes-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 1, 1, 0), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32, 255.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-bytes",
                "0",
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!(header.bytes_signed, 0);
        let mut pixels = [0.0_f32; 2];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [1.0, 255.0]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_size_preserves_real_mrc_sampling_and_cell_geometry() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-geometry-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 4, 1, 2), 0);
        (*header).xlen = 10.0;
        (*header).ylen = 12.0;
        (*header).zlen = 7.0;
        (*header).nxstart = 4;
        (*header).nystart = 5;
        (*header).nzstart = 6;
        (*header).xorg = 1.5;
        (*header).yorg = 2.5;
        (*header).zorg = 3.5;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32; 16];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-size",
                "2,3",
                "-origin",
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.mx, header.my, header.mz), (2, 3, 1));
        assert_eq!((header.xlen, header.ylen, header.zlen), (5.0, 9.0, 7.0));
        assert_eq!((header.nxstart, header.nystart, header.nzstart), (4, 5, 6));
        assert_eq!((header.xorg, header.yorg, header.zorg), (-1.0, -0.5, 3.5));
        assert_eq!(header.nlabl, 1);
        assert_eq!(&header.labels[0][..23], b"NEWSTACK: Images copied");
        assert_eq!((header.labels[0][58], header.labels[0][62]), (b'-', b'-'));
        assert_eq!((header.labels[0][69], header.labels[0][72]), (b':', b':'));
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_pixel_from_mdoc_uses_first_selected_zvalue_spacing() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-newstack-pixel-mdoc-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let mdoc = std::path::PathBuf::from(format!("{}.mdoc", input.display()));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 1, 1, 2, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for (section, value) in [10.0_f32, 20.0].into_iter().enumerate() {
            let mut pixel = [value];
            assert_eq!(
                ii_write_section_float(file, pixel.as_mut_ptr().cast(), section as i32),
                0
            );
        }
        ii_close(file);
    }
    std::fs::write(
        &mdoc,
        "[ZValue = 0]\nPixelSpacing = 2.5\n\n[ZValue = 1]\nPixelSpacing = 4.0\n",
    )
    .unwrap();
    let native_output = base.with_extension("native-output.mrc");
    if let Ok(native_newstack) = std::env::var("IMOD_NATIVE_NEWSTACK") {
        assert!(
            Command::new(native_newstack)
                .env(
                    "AUTODOC_DIR",
                    std::env::var("AUTODOC_DIR")
                        .unwrap_or_else(|_| "/tmp/imod-reference-build/autodoc".into()),
                )
                .args([
                    "-input",
                    input.to_str().unwrap(),
                    "-output",
                    native_output.to_str().unwrap(),
                    "-secs",
                    "1",
                    "-pixel",
                ])
                .status()
                .unwrap()
                .success()
        );
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-secs",
                "1",
                "-pixel",
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.mx, header.my, header.mz), (1, 1, 1));
        assert_eq!((header.xlen, header.ylen, header.zlen), (4.0, 4.0, 4.0));
        ii_close(file);
    }
    if native_output.exists() {
        unsafe {
            let name = CString::new(native_output.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open(name.as_ptr(), c"rb".as_ptr());
            assert!(!file.is_null());
            let mut header = std::mem::zeroed::<MrcHeader>();
            assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
            assert_eq!((header.mx, header.my, header.mz), (1, 1, 1));
            assert_eq!((header.xlen, header.ylen, header.zlen), (4.0, 4.0, 4.0));
            ii_close(file);
        }
    }
    for path in [input, output, native_output, mdoc] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_print_size_exits_after_real_mrc_header() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-print-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 3, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        ii_close(file);
    }
    let result = Command::new(env!("CARGO_BIN_EXE_newstack"))
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-linear",
            "-nearest",
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        "ERROR: NEWSTACK - You cannot enter both -linear and -nearest\n"
    );
    let result = Command::new(env!("CARGO_BIN_EXE_newstack"))
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-size",
            "2,5",
            "-print",
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        " Output size:            2           5\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn newstack_expand_derives_output_dimensions_from_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-expand-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-expand",
                "2",
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (4, 4));
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_rotate_transposes_default_real_mrc_dimensions() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-rotate-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 3, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-rotate",
                "90",
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (3, 2));
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_bin_averages_real_mrc_blocks() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-bin-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 4, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels: [f32; 16] = core::array::from_fn(|index| index as f32 + 1.0);
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-bin",
                "2"
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (2, 2));
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [3.5, 5.5, 11.5, 13.5]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_allow_odd_even_changes_binned_real_mrc_size() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-oddeven-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 9, 9, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32; 81];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    for (suffix, odd, expected) in [("normal", false, (3, 3)), ("odd", true, (2, 2))] {
        let output = base.with_extension(format!("{suffix}.mrc"));
        let mut command = Command::new(env!("CARGO_BIN_EXE_newstack"));
        command.args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-bin",
            "4",
        ]);
        if odd {
            command.arg("-oddeven");
        }
        assert!(command.status().unwrap().success());
        unsafe {
            let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open(name.as_ptr(), c"rb".as_ptr());
            let mut header = std::mem::zeroed::<MrcHeader>();
            assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
            assert_eq!((header.nx, header.ny), expected);
            ii_close(file);
        }
        let _ = std::fs::remove_file(output);
    }
    let _ = std::fs::remove_file(input);
}

#[test]
fn newstack_affine_bin_interpolates_binned_real_mrc() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-newstack-affine-bin-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let xform = base.with_extension("xf");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 4, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels: [f32; 16] = core::array::from_fn(|index| index as f32 + 1.0);
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    std::fs::write(&xform, "1 0 0 1 0 0\n").unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-bin",
                "2",
                "-xform",
                xform.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [3.5, 5.5, 11.5, 13.5]);
        ii_close(file);
    }
    for path in [input, output, xform] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_routes_repeated_input_files_to_repeated_outputs() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-multifile-{}", std::process::id()));
    let input_a = base.with_extension("a.mrc");
    let input_b = base.with_extension("b.mrc");
    let output_a = base.with_extension("out-a.mrc");
    let output_b = base.with_extension("out-b.mrc");
    for (path, sections) in [
        (&input_a, [[1.0_f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        (
            &input_b,
            [[11.0_f32, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
        ),
    ] {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        unsafe {
            let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
            assert!(!file.is_null());
            let header = (*file).header.cast::<MrcHeader>();
            assert_eq!(mrc_head_new(&mut *header, 2, 2, 2, 2), 0);
            ii_sync_from_mrc_header(file, header);
            assert_eq!(mrc_head_write((*file).fp, header), 0);
            for (z, section) in sections.into_iter().enumerate() {
                let mut section = section;
                assert_eq!(
                    ii_write_section_float(file, section.as_mut_ptr().cast(), z as i32),
                    0
                );
            }
            ii_close(file);
        }
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input_a.to_str().unwrap(),
                "-input",
                input_b.to_str().unwrap(),
                "-output",
                output_a.to_str().unwrap(),
                "-output",
                output_b.to_str().unwrap(),
                "-numout",
                "1,3",
            ])
            .status()
            .unwrap()
            .success()
    );
    for (path, expected) in [
        (&output_a, vec![[1.0_f32, 2.0, 3.0, 4.0]]),
        (
            &output_b,
            vec![
                [5.0_f32, 6.0, 7.0, 8.0],
                [11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0],
            ],
        ),
    ] {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        unsafe {
            let file = ii_open(name.as_ptr(), c"rb".as_ptr());
            assert!(!file.is_null());
            let mut header = std::mem::zeroed::<MrcHeader>();
            assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
            assert_eq!(header.nz, expected.len() as i32);
            for (z, expected) in expected.into_iter().enumerate() {
                let mut actual = [0.0_f32; 4];
                assert_eq!(
                    ii_read_section_float(file, actual.as_mut_ptr().cast(), z as i32),
                    0
                );
                assert_eq!(actual, expected);
            }
            ii_close(file);
        }
    }
    for path in [input_a, input_b, output_a, output_b] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_routes_source_shaped_input_and_output_list_files() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-lists-{}", std::process::id()));
    let input_a = base.with_extension("list-a.mrc");
    let input_b = base.with_extension("list-b.mrc");
    let output_a = base.with_extension("list-out-a.mrc");
    let output_b = base.with_extension("list-out-b.mrc");
    let input_list = base.with_extension("inputs.txt");
    let output_list = base.with_extension("outputs.txt");
    for (path, pixels) in [
        (&input_a, [3.0_f32, 4.0, 5.0, 6.0]),
        (&input_b, [13.0_f32, 14.0, 15.0, 16.0]),
    ] {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        unsafe {
            let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
            assert!(!file.is_null());
            let header = (*file).header.cast::<MrcHeader>();
            assert_eq!(mrc_head_new(&mut *header, 2, 2, 1, 2), 0);
            ii_sync_from_mrc_header(file, header);
            assert_eq!(mrc_head_write((*file).fp, header), 0);
            let mut pixels = pixels;
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
                0
            );
            ii_close(file);
        }
    }
    std::fs::write(
        &input_list,
        format!("2\n{}\n/\n{}\n/\n", input_a.display(), input_b.display()),
    )
    .unwrap();
    std::fs::write(
        &output_list,
        format!("2\n{}\n1\n{}\n1\n", output_a.display(), output_b.display()),
    )
    .unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-FileOfInputs",
                input_list.to_str().unwrap(),
                "-FileOfOutputs",
                output_list.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );
    for (path, expected) in [
        (&output_a, [3.0_f32, 4.0, 5.0, 6.0]),
        (&output_b, [13.0_f32, 14.0, 15.0, 16.0]),
    ] {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        unsafe {
            let file = ii_open(name.as_ptr(), c"rb".as_ptr());
            assert!(!file.is_null());
            let mut actual = [0.0_f32; 4];
            assert_eq!(
                ii_read_section_float(file, actual.as_mut_ptr().cast(), 0),
                0
            );
            assert_eq!(actual, expected);
            ii_close(file);
        }
    }
    for path in [
        input_a,
        input_b,
        output_a,
        output_b,
        input_list,
        output_list,
    ] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_applies_identity_transform_file_to_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-xform-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let xform = base.with_extension("xf");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 5, 5, 2, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = (0..25).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        let mut second = (100..125).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(
            ii_write_section_float(file, second.as_mut_ptr().cast(), 1),
            0
        );
        ii_close(file);
    }
    std::fs::write(&xform, "1 0 0 1 0 0\n1 0 0 1 1 0\n").unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-xform",
                xform.to_str().unwrap(),
                "-uselines",
                "0"
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        let mut pixels = [0.; 25];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, core::array::from_fn(|value| value as f32));
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 1),
            0
        );
        assert_eq!(pixels, core::array::from_fn(|value| value as f32 + 100.));
        ii_close(file);
    }
    for path in [input, output, xform] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_applies_repeated_offsets_in_source_composition_order() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-offsets-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output_last = base.with_extension("last.mrc");
    let output_first = base.with_extension("first.mrc");
    let xform = base.with_extension("xf");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    let first = (0..49).map(|value| value as f32).collect::<Vec<_>>();
    let second = (100..149).map(|value| value as f32).collect::<Vec<_>>();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 7, 7, 2, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut section = first.clone();
        assert_eq!(
            ii_write_section_float(file, section.as_mut_ptr().cast(), 0),
            0
        );
        let mut section = second.clone();
        assert_eq!(
            ii_write_section_float(file, section.as_mut_ptr().cast(), 1),
            0
        );
        ii_close(file);
    }
    std::fs::write(&xform, "2 0 0 2 0 0\n2 0 0 2 0 0\n").unwrap();
    for (output, apply_first) in [(&output_last, false), (&output_first, true)] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_newstack"));
        command.args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-xform",
            xform.to_str().unwrap(),
            "-uselines",
            "0",
            "-offset",
            "1,0",
            "-offset",
            "0,0",
        ]);
        if apply_first {
            command.arg("-applyfirst");
        }
        assert!(command.status().unwrap().success());
    }
    let read_sections = |path: &std::path::Path| unsafe {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut output = [[0.0_f32; 49]; 2];
        for (index, section) in output.iter_mut().enumerate() {
            assert_eq!(
                ii_read_section_float(file, section.as_mut_ptr().cast(), index as i32),
                0
            );
        }
        ii_close(file);
        output
    };
    let actual_last = read_sections(&output_last);
    let actual_first = read_sections(&output_first);
    let expected = |input: &[f32], x_offset: f32, offset_first: bool| unsafe {
        let mut fprod = [2.0_f32, 0.0, 0.0, 2.0, -x_offset, 0.0];
        if offset_first {
            let frot = [1.0_f32, 0.0, 0.0, 1.0, -x_offset, 0.0];
            xfmult(&frot, &[2.0, 0.0, 0.0, 2.0, 0.0, 0.0], &mut fprod);
        }
        let matrix = [[fprod[0], fprod[1]], [fprod[2], fprod[3]]];
        let mut source = input.to_vec();
        let mut result = [0.0_f32; 49];
        cubinterp(
            source.as_mut_ptr(),
            result.as_mut_ptr(),
            7,
            7,
            7,
            7,
            &matrix,
            3.5,
            3.5,
            fprod[4],
            fprod[5],
            1.0,
            0.0,
            0,
        );
        result
    };
    assert_eq!(actual_last[0], expected(&first, 1.0, false));
    assert_eq!(actual_first[0], expected(&first, 1.0, true));
    assert_eq!(actual_last[1], expected(&second, 0.0, false));
    assert_eq!(actual_first[1], expected(&second, 0.0, false));
    assert_ne!(actual_last[0], actual_first[0]);
    for path in [input, output_last, output_first, xform] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_memory_limit_chunks_ordinary_affine_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-chunks-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output_full = base.with_extension("full.mrc");
    let output_chunks = base.with_extension("chunks.mrc");
    let xform = base.with_extension("xf");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    let mut pixels = vec![0.0_f32; 512 * 512];
    for (index, pixel) in pixels.iter_mut().enumerate() {
        *pixel = (index / 512) as f32 * 1000.0 + (index % 512) as f32;
    }
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 512, 512, 1, 2), 0);
        (*header).xlen = 1024.0;
        (*header).ylen = 1536.0;
        (*header).zlen = 7.0;
        (*header).nxstart = 4;
        (*header).nystart = 5;
        (*header).nzstart = 6;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    std::fs::write(&xform, "1 0 0 1 0 0\n").unwrap();
    for (output, memory_limit) in [(&output_full, None), (&output_chunks, Some("1"))] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_newstack"));
        command.args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-xform",
            xform.to_str().unwrap(),
            "-size",
            "200,150",
            "-bin",
            "2",
        ]);
        if let Some(memory_limit) = memory_limit {
            command.args(["-memory", memory_limit]);
        }
        assert!(command.status().unwrap().success());
    }
    let read_output = |path: &std::path::Path| unsafe {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (200, 150));
        assert_eq!(header.nlabl, 1);
        assert_eq!(&header.labels[0][..23], b"NEWSTACK: Images copied");
        assert_eq!(&header.labels[0][23..36], b", transformed");
        let mut output = vec![0.0_f32; 200 * 150];
        assert_eq!(
            ii_read_section_float(file, output.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
        output
    };
    let chunked = read_output(&output_chunks);
    let full = read_output(&output_full);
    assert_eq!(chunked.len(), full.len());
    assert_eq!(
        chunked
            .iter()
            .zip(&full)
            .position(|(left, right)| left != right),
        None,
        "first differing output pixel: chunk {:?}, full {:?}",
        &chunked[..8],
        &full[..8]
    );
    for path in [input, output_full, output_chunks, xform] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_blank_accepts_out_of_range_sections_and_writes_zero_metadata() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-blank-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 3, 2, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [3.0_f32; 6];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-secs",
                "7,9",
                "-blank",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!((*header).nz, 2);
        assert_eq!((*header).amin, 0.0);
        assert_eq!((*header).amax, 0.0);
        assert_eq!((*header).amean, 0.0);
        let mut pixels = [1.0_f32; 6];
        for section in 0..2 {
            assert_eq!(
                ii_read_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
            assert_eq!(pixels, [0.0; 6]);
        }
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_multadd_uses_pip_factor_and_constant_on_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-multadd-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-multadd",
                "2,3",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [5.0, 7.0, 9.0, 11.0]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_reports_source_truncations_for_scaled_byte_output() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-trunc-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    let result = Command::new(env!("CARGO_BIN_EXE_newstack"))
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-mode",
            "0",
            "-multadd",
            "100,0",
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        " TRUNCATIONS OCCURRED:          0 at low end,          2 at high end\n"
    );
    let quiet_output = base.with_extension("quiet.mrc");
    let result = Command::new(env!("CARGO_BIN_EXE_newstack"))
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            quiet_output.to_str().unwrap(),
            "-mode",
            "0",
            "-multadd",
            "100,0",
            "-quiet",
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert!(result.stdout.is_empty());
    unsafe {
        let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [100.0, 200.0, 255.0, 255.0]);
        ii_close(file);
    }
    for path in [input, output, quiet_output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_scale_maps_source_header_range_on_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-scale-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 1, 2), 0);
        (*header).amin = 1.0;
        (*header).amax = 4.0;
        (*header).amean = 2.5;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-scale",
                "10,20",
                "-map",
                "2,5",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!(&header.labels[0][36..54], b", densities scaled");
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [6.666667, 10.0, 13.333334, 16.666666]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_contrast_converts_black_white_to_source_scale_range() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-contrast-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 1, 1, 2), 0);
        (*header).amin = 1.0;
        (*header).amax = 4.0;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32, 4.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-contrast",
                "0,255",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!(&header.labels[0][36..54], b", densities scaled");
        let mut pixels = [0.0_f32; 2];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [0.0, 255.0]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_float_three_prescans_sections_then_shifts_to_shared_mean() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-float3-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 1, 2, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut first = [1.0_f32, 3.0];
        let mut second = [10.0_f32, 14.0];
        assert_eq!(
            ii_write_section_float(file, first.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(
            ii_write_section_float(file, second.as_mut_ptr().cast(), 1),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-float",
                "3",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut pixels = [0.0_f32; 2];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [6.0, 8.0]);
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 1),
            0
        );
        assert_eq!(pixels, [5.0, 9.0]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_float_four_prescans_and_scales_shifted_global_range() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-float4-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 1, 2, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut first = [1.0_f32, 3.0];
        let mut second = [10.0_f32, 14.0];
        assert_eq!(
            ii_write_section_float(file, first.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(
            ii_write_section_float(file, second.as_mut_ptr().cast(), 1),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-float",
                "4",
                "-scale",
                "10,20",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        let mut pixels = [0.0_f32; 2];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [12.5, 17.5]);
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 1),
            0
        );
        assert_eq!(pixels, [10.0, 20.0]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_float_two_uses_mad_filtered_global_z_range() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-float2-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 10, 10, 9, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..8 {
            let mut pixels = (0..100).map(|value| value as f32).collect::<Vec<_>>();
            pixels[99] += section as f32;
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        let mut outlier = vec![0.0_f32; 100];
        outlier[99] = 100.0;
        assert_eq!(
            ii_write_section_float(file, outlier.as_mut_ptr().cast(), 8),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-float",
                "2",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        let mut normal = vec![0.0_f32; 100];
        assert_eq!(
            ii_read_section_float(file, normal.as_mut_ptr().cast(), 7),
            0
        );
        // The highest retained normal Z maps to the float output maximum.
        assert!(normal[99] > 0.99e30);
        let mut outlier = vec![0.0_f32; 100];
        assert_eq!(
            ii_read_section_float(file, outlier.as_mut_ptr().cast(), 8),
            0
        );
        // The discarded high-Z section is still transformed with the retained
        // global range and consequently saturates at the source float maximum.
        assert_eq!(outlier[99], 1.0e30);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_meansd_uses_float_two_section_statistics() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-meansd-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 1, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32, 3.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-meansd",
                "10,2",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        let mut pixels = [0.0_f32; 2];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [8.585786, 11.414213]);
        let mean = (pixels[0] + pixels[1]) / 2.0;
        let sample_sd = ((pixels[0] - mean).powi(2) + (pixels[1] - mean).powi(2)).sqrt();
        assert!((mean - 10.0).abs() < 1.0e-5);
        assert!((sample_sd - 2.0).abs() < 1.0e-5);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_float_one_maps_header_range_to_float_range() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-float1-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 1, 1, 2), 0);
        (*header).amin = 1.0;
        (*header).amax = 3.0;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32, 3.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        Command::new(env!("CARGO_BIN_EXE_newstack"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-float",
                "1",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        let mut pixels = [0.0_f32; 2];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [0.0, 1.0e30]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_fixrange_retains_source_legality_errors_on_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-fixrange-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut values = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            ii_write_section_float(file, values.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    let result = Command::new(env!("CARGO_BIN_EXE_newstack"))
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-fixrange",
            "1,2",
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        "ERROR: NEWSTACK - The entry for -fixrange must be at least 2\n"
    );
    let result = Command::new(env!("CARGO_BIN_EXE_newstack"))
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-scale",
            "1,2",
            "-fixrange",
            "2,2",
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        "ERROR: NEWSTACK - You cannot enter -fixrange with any scaling options\n"
    );
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn newstack_fixrange_scans_interpolated_real_mrc_and_scales_low_sd_values() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-newstack-fixrange-run-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let xform = base.with_extension("xf");
    let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 1, 2), 0);
        (*header).amin = 1.0;
        (*header).amax = 4.0;
        (*header).amean = 2.5;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut values = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            ii_write_section_float(file, values.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    std::fs::write(&xform, "1 0 0 1 0 0\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_newstack"))
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-xform",
            xform.to_str().unwrap(),
            "-fixrange",
            "2,2",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "INFO: Newstack scaling values by    2.0 to preserve intensity resolution\n  because SD of values is below   10.0\n"
    );
    let output_name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_name.as_ptr(), c"rb".as_ptr());
        let mut values = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, values.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(values, [2.0, 4.0, 6.0, 8.0]);
        ii_close(file);
    }
    for path in [input, output, xform] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_fixrange_uses_source_signed_mode_shift_after_low_sd_scan() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-newstack-fixrange-shift-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let xform = base.with_extension("xf");
    let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 1, 1), 0);
        (*header).amin = -30000.0;
        (*header).amax = -30000.0;
        (*header).amean = -30000.0;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut values = [-30000.0_f32; 4];
        assert_eq!(
            ii_write_section_float(file, values.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    std::fs::write(&xform, "1 0 0 1 0 0\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_newstack"))
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-xform",
            xform.to_str().unwrap(),
            "-fixrange",
            "2,2",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "INFO: Newstack scaling values by    2.0 to preserve intensity resolution\n  because SD of values is below   10.0\n"
    );
    let output_name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_name.as_ptr(), c"rb".as_ptr());
        let mut values = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, values.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(values, [-27232.0; 4]);
        ii_close(file);
    }
    for path in [input, output, xform] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn newstack_size_to_output_centres_real_mrc_crop_and_mean_padding() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-size-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let crop = base.with_extension("crop.mrc");
    let pad = base.with_extension("pad.mrc");
    let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 4, 1, 2), 0);
        (*header).amean = 7.5;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(
            ii_write_section_float(file, values.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    for (output, size, expected) in [
        (&crop, "2,2", vec![5.0_f32, 6.0, 9.0, 10.0]),
        (
            &pad,
            "6,6",
            vec![
                7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 0.0, 1.0, 2.0, 3.0, 7.5, 7.5, 4.0, 5.0, 6.0,
                7.0, 7.5, 7.5, 8.0, 9.0, 10.0, 11.0, 7.5, 7.5, 12.0, 13.0, 14.0, 15.0, 7.5, 7.5,
                7.5, 7.5, 7.5, 7.5, 7.5,
            ],
        ),
    ] {
        assert!(
            Command::new(env!("CARGO_BIN_EXE_newstack"))
                .args([
                    "-input",
                    input.to_str().unwrap(),
                    "-output",
                    output.to_str().unwrap(),
                    "-size",
                    size
                ])
                .status()
                .unwrap()
                .success()
        );
        let output_name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        unsafe {
            let file = ii_open(output_name.as_ptr(), c"rb".as_ptr());
            let mut header = std::mem::zeroed::<MrcHeader>();
            assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
            let mut values = vec![0.0_f32; expected.len()];
            assert_eq!(
                ii_read_section_float(file, values.as_mut_ptr().cast(), 0),
                0
            );
            assert_eq!(
                (header.nx, header.ny),
                if size == "2,2" { (2, 2) } else { (6, 6) }
            );
            assert_eq!(values, expected);
            ii_close(file);
        }
    }
    for path in [input, crop, pad] {
        let _ = std::fs::remove_file(path);
    }
}
