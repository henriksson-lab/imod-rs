mod common;

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

/// Every PIP-driven invocation needs an autodoc directory, exactly as a real
/// IMOD install provides one through `AUTODOC_DIR` or `IMOD_DIR`.
const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
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
    // Verified against the reference binary: `exitError` (`parse_input_params.f90:231`)
    // writes to stdout after a blank record and `pipexit(1)` exits 1.
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&result.stderr), "");
    assert!(
        String::from_utf8_lossy(&result.stdout)
            .ends_with("\nERROR: NEWSTACK - You cannot enter both -linear and -nearest\n"),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
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
    // The PIP fallback banner that precedes this when no autodoc is on
    // AUTODOC_DIR/IMOD_DIR is native behaviour too (`parse_input_params.f90:157`).
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .ends_with(" Output size:            2           5\n")
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        let mut command = common::imod_cmd("newstack");
        command.env("AUTODOC_DIR", AUTODOC);
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        let mut command = common::imod_cmd("newstack");
        command.env("AUTODOC_DIR", AUTODOC);
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
    // Note what this checks: that the chunked and unchunked routes agree.  It
    // catches a chunked route that computes the wrong thing (a deliberate
    // one-line offset error in the chunk read makes it fail), but not one that
    // never runs, because then both routes are the same code.
    // `-memory 1` is rejected by the source: `newstack.f90:1035-1036` requires
    // `limToAlloc >= 1000` and `lenTemp <= limToAlloc / 2`, and `lenTemp`
    // defaults to `MAXTEMP` (5000000), so one megabyte never passes.  `-test`
    // sets both halves directly, which is how the source's own test path
    // forces a small working array.
    for (output, memory_limit) in [(&output_full, None), (&output_chunks, Some("20000,100"))] {
        let mut command = common::imod_cmd("newstack");
        command.env("AUTODOC_DIR", AUTODOC);
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
            command.args(["-test", memory_limit]);
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
        // `iiuTransHeader` keeps the input's titles and the final
        // `iiuWriteHeader(..., 1, ...)` appends this run's.  This input has no
        // label of its own, so both routes end with exactly the one title --
        // native `newstack` gives the same `nlabl` for both.
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
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
    // Verified against `/tmp/imod-reference-build/flib/image/newstack` on this
    // same fixture: the run prints the unit report, the `irdhdr` header report
    // and the per-section table, and format 103 (`newstack.f90:2777`) ends with
    // " at high end of range".  The leading report is asserted by landmark
    // rather than in full because `irdhdr`'s "(undetermined)" density lines
    // still carry one dot too many (see the module note in the report).
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(
        stdout.contains(" Number of columns, rows, sections .....       2       2       1\n"),
        "{stdout}"
    );
    assert!(
        stdout.ends_with(
            " section   input min&max       output min&max  &  mean\n\
             \u{20}      0      1.00      4.00    100.00    255.00    202.50\n\
             \u{20}TRUNCATIONS OCCURRED:          0 at low end,          2 at high end of range\n"
        ),
        "{stdout}"
    );
    let quiet_output = base.with_extension("quiet.mrc");
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
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
    // Native keeps both of these under -quiet: `iiuOpen` prints the NEW unit
    // line from C without consulting the print flag, and format 103 is not
    // guarded by `quiet` either.
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        format!(
            "\n NEW image file on unit   2 : {}\n\
             \u{20}TRUNCATIONS OCCURRED:          0 at low end,          2 at high end of range\n",
            quiet_output.display()
        )
    );
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        // Reference-binary values for this fixture (`-scale 10,20 -map 2,5`).
        assert_eq!(pixels, [6.666667, 10.0, 13.333333, 16.666666]);
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        // The highest retained normal Z maps to the float output maximum,
        // which is `optimalMax(newMode + 1)` = 255 for mode 2.  Verified
        // against the reference binary on this fixture.
        assert_eq!(normal[99], 255.0);
        let mut outlier = vec![0.0_f32; 100];
        assert_eq!(
            ii_read_section_float(file, outlier.as_mut_ptr().cast(), 8),
            0
        );
        // The discarded high-Z section is still transformed with the retained
        // global range; mode 2 output is not clipped by `scaleAndWriteChunk`
        // (`newstack.f90:3229-3231`), so it runs past the 255 range.
        assert_eq!(outlier[99], 812.4673);
        assert_eq!(outlier[0], 112.439835);
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        // `optimalMax(newMode + 1)` is 255 for mode 2 (`newstack.f90:48`), so
        // -float 1 maps the header range onto 0..255, as the reference binary
        // does for this fixture.
        assert_eq!(pixels, [0.0, 255.0]);
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
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
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
    // Reference-binary behaviour: `exitError` writes to stdout and exits 1.
    assert_eq!(result.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&result.stdout)
            .ends_with("\nERROR: NEWSTACK - The entry for -fixrange must be at least 2\n"),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
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
    assert!(
        String::from_utf8_lossy(&result.stdout)
            .ends_with("\nERROR: NEWSTACK - You cannot enter -fixrange with any scaling options\n"),
        "{}",
        String::from_utf8_lossy(&result.stdout)
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
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
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
    // `newstack.f90:1486-1488`: the second literal ends at "below" with no
    // separating blank and the format's trailing "/" writes an empty record.
    // Confirmed byte for byte against the reference binary on this fixture.
    // The header report that follows is not asserted here: native also emits
    // `dopen`'s " FORMATTED  ro  file opened:" line for the transform file,
    // which this stream path does not, and native's -fixrange scaling produces
    // different output densities (both recorded as gaps in the report).
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(
        stdout.contains(
            "INFO: Newstack scaling values by    2.0 to preserve intensity resolution\n\
             \u{20} because SD of values is below  10.0\n\n"
        ),
        "{stdout}"
    );
    assert!(
        stdout.contains(" section   input min&max       output min&max  &  mean\n"),
        "{stdout}"
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
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
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
    // Native-verified on this fixture: the INFO block, then the header report,
    // then the section line below.  ("FORMATTED  ro  file opened" for the
    // transform file is native-only and is recorded as a `dopen` gap.)
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(
        stdout.contains(
            "INFO: Newstack scaling values by    2.0 to preserve intensity resolution\n\
             \u{20} because SD of values is below  10.0\n\n"
        ),
        "{stdout}"
    );
    assert!(
        stdout.ends_with(
            " section   input min&max       output min&max  &  mean\n\
             \u{20}      0 -30000.00 -30000.00 -27232.00 -27232.00 -27232.00\n"
        ),
        "{stdout}"
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
fn newstack_size_to_output_centres_real_mrc_crop_and_edge_median_padding() {
    // 5 by 5, so every edge that `sliceEdgeMedian` samples has an odd number of
    // pixels: top and bottom 5, left and right 3 (`taperpad.c:1009-1028`).  Its
    // border is a small ramp and its interior is around 1000, and the header
    // carries a deliberately wrong mean of 999, so padding with the source's
    // `sliceEdgeMedian` (12) is far from padding with `dmeanIn` (999) or with
    // the true mean.  Both expected images are the byte-identical output of the
    // native `newstack` on this same fixture.
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-size-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let crop = base.with_extension("crop.mrc");
    let pad = base.with_extension("pad.mrc");
    let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 5, 5, 1, 2), 0);
        (*header).amean = 999.0;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut values: [f32; 25] = core::array::from_fn(|index| {
            let (x, y) = (index % 5, index / 5);
            if x == 0 || x == 4 || y == 0 || y == 4 {
                index as f32
            } else {
                1000.0 + index as f32
            }
        });
        assert_eq!(
            ii_write_section_float(file, values.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    for (output, size, side, expected) in [
        (
            &crop,
            "3,3",
            3,
            vec![
                1006.0_f32, 1007.0, 1008.0, 1011.0, 1012.0, 1013.0, 1016.0, 1017.0, 1018.0,
            ],
        ),
        (
            &pad,
            "7,7",
            7,
            vec![
                12.0_f32, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 0.0, 1.0, 2.0, 3.0, 4.0, 12.0,
                12.0, 5.0, 1006.0, 1007.0, 1008.0, 9.0, 12.0, 12.0, 10.0, 1011.0, 1012.0, 1013.0,
                14.0, 12.0, 12.0, 15.0, 1016.0, 1017.0, 1018.0, 19.0, 12.0, 12.0, 20.0, 21.0, 22.0,
                23.0, 24.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
            ],
        ),
    ] {
        assert!(
            common::imod_cmd("newstack")
                .env("AUTODOC_DIR", AUTODOC)
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
            assert_eq!((header.nx, header.ny), (side, side));
            assert_eq!(values, expected);
            assert!(values.iter().all(|value| *value != 999.0));
            ii_close(file);
        }
    }
    for path in [input, crop, pad] {
        let _ = std::fs::remove_file(path);
    }
}
/// Regression: `newstack` printed nothing at all, because the processing loop
/// never turned unit printing back on after the preliminary pass and never went
/// through `imopen`/`irdhdr`/`iiuOpen` for the files it actually used
/// (`newstack.f90:1517-1527, 1790`).  The expected text below was taken from
/// `/tmp/imod-reference-build/flib/image/newstack` on this same fixture.
#[test]
fn newstack_prints_source_unit_and_header_reports_for_stream_copy() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-report-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 2, 2, 2), 0);
        (*header).amin = 1.0;
        (*header).amax = 16.0;
        (*header).amean = 8.5;
        (*header).nlabl = 1;
        // The whole 80-byte slot is written: `mrc_head_new` leaves the label
        // area uninitialised, and both binaries copy it to disk verbatim.
        (*header).labels[0] = [b' '; 81];
        (&mut (*header).labels[0])[..20].copy_from_slice(b"source fixture label");
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut first = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut second = [9.0_f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
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
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8(result.stdout).unwrap();
    // `iiuOpen` writes the NEW line from C and flushes it, while the Fortran
    // report stays in the unit 6 buffer, so native emits the NEW line before
    // the report.  This crate flushes every line as it is written, so the NEW
    // line lands where the source writes it; every other byte is identical.
    let expected_report = format!(
        "\n RO image file on unit   1 : {}     Size=          1 K\n\
         \n\
         \u{20}Number of columns, rows, sections .....       4       2       2\n\
         \u{20}Map mode ..............................    2   (32-bit float)             \n\
         \u{20}Start cols, rows, sects, grid x,y,z ...    0     0     0       4      2      2\n\
         \u{20}Pixel spacing (Angstroms)..............   1.000      1.000      1.000    \n\
         \u{20}Cell angles ...........................   90.000   90.000   90.000\n\
         \u{20}Fast, medium, slow axes ...............    X    Y    Z\n\
         \u{20}Origin on x,y,z .......................    0.000       0.000       0.000    \n\
         \u{20}Minimum density .......................   1.0000    \n\
         \u{20}Maximum density .......................   16.000    \n\
         \u{20}Mean density ..........................   8.5000    \n\
         \u{20}tilt angles (original,current) ........   0.0   0.0   0.0   0.0   0.0   0.0\n\
         \u{20}Space group,# extra bytes,idtype,lens .        0        0        0        0\n\
         \n\
         \u{20}    1 Titles :\n\
         source fixture label                                                           \n\
         \n",
        input.display()
    );
    assert!(stdout.contains(&expected_report), "{stdout}");
    assert!(
        stdout.ends_with(
            " section   input min&max       output min&max  &  mean\n\
             \u{20}      0      1.00      8.00      1.00      8.00      4.50\n\
             \u{20}      1      9.00     16.00      9.00     16.00     12.50\n"
        ),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!(
            "\n NEW image file on unit   2 : {}\n",
            output.display()
        )),
        "{stdout}"
    );
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// Regression: `-bin` replaced the single `array` allocation that the source
/// sizes once in `reallocateIfNeeded` (`newstack.f90:2798`) with the smaller
/// binned result, so reading the next section overran the heap and aborted the
/// process.  Also pins the X/Y cell scaling by `readReduction`
/// (`newstack.f90:1839`), which this path used to drop.
#[test]
fn newstack_bin_streams_every_section_and_scales_cell_by_read_reduction() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-binsecs-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 4, 3, 2), 0);
        (*header).amin = 0.0;
        (*header).amax = 47.0;
        (*header).amean = 23.5;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..3_i32 {
            let mut pixels = [0.0_f32; 16];
            for (index, pixel) in pixels.iter_mut().enumerate() {
                *pixel = (section * 16 + index as i32) as f32;
            }
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-bin",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz), (2, 2, 3));
        assert_eq!((header.mx, header.my, header.mz), (2, 2, 3));
        assert_eq!((header.xlen, header.ylen, header.zlen), (4.0, 4.0, 3.0));
        for section in 0..3_i32 {
            let mut pixels = [0.0_f32; 4];
            assert_eq!(
                ii_read_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
            let base = (section * 16) as f32;
            assert_eq!(
                pixels,
                [base + 2.5, base + 4.5, base + 10.5, base + 12.5],
                "section {section}"
            );
        }
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// Regression: an extended header whose `nint` and `nreal` are both zero is a
/// type `extraHeaderSizes` (`extraheader.c:915-935`) does not recognise, so
/// `getExtraHeaderMaxSecSize` reports no bytes per section and the source
/// writes no extended header at all; this path used to reserve
/// `next / nz * numSecOut` bytes for it.  It also pins `imodFlags`, which
/// `mrcInitOutputHeader` (`mrcfiles.c:790`) resets rather than carrying over.
#[test]
fn newstack_drops_unsupported_extended_header_and_resets_imod_flags() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-extdrop-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 2, 2), 0);
        (*header).amin = 1.0;
        (*header).amax = 8.0;
        (*header).amean = 4.5;
        // MRC_FLAGS_SBYTES | MRC_FLAGS_INV_ORIGIN, without MRC_FLAGS_BAD_RMS_NEG.
        (*header).imod_flags = 1 | 4;
        (*header).nint = 0;
        (*header).nreal = 0;
        let mut extra = [7_u8; 64];
        assert_eq!(mrc_write_extra_header(header, extra.as_mut_ptr(), 64), 0);
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
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
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
        assert_eq!(header.next, 0);
        assert_eq!((header.nint, header.nreal), (0, 0));
        assert_eq!(header.ext_type, *b"    ");
        assert_eq!(header.imod_flags, 1 | 4 | 8);
        assert_eq!(
            (header.alpha, header.beta, header.gamma),
            (90.0, 90.0, 90.0)
        );
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 1),
            0
        );
        assert_eq!(pixels, [5., 6., 7., 8.]);
        ii_close(file);
    }
    // `iiuTransHeader` ends in `iiuTransExtendedData` (`unit_header.c:395`),
    // which writes the input's 64 extended bytes to the output before
    // `newstack.f90:1907-1935` decides the output needs none and resets `next`
    // to 0.  The pixel data is then written from offset 1024 over the first 32
    // of those bytes and the file keeps the 32-byte tail, so the reference
    // binary leaves a 1088-byte file here, not a 1056-byte one -- verified
    // against the native `newstack` on this exact input.
    assert_eq!(std::fs::metadata(&output).unwrap().len(), 1024 + 64);
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// Regression for the self-handled error sites: the source's `exitError`
/// (`parse_input_params.f90:231`) writes to stdout after a blank record and
/// exits 1, and PIP's own option errors go through `PipSetError` with the
/// `'ERROR: NEWSTACK - '` prefix installed by `PipReadOrParseOptions`
/// (`newstack.f90:183`).  Both strings and both exit statuses were taken from
/// the reference binary.
#[test]
fn newstack_pip_and_section_errors_exit_one_on_stdout() {
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args(["-nosuchoption"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&result.stderr), "");
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "ERROR: NEWSTACK - Illegal option: nosuchoption\n"
    );

    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-exiterr-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 3, 2, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [1.0_f32; 12];
        for section in 0..2 {
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    // `write(*,'(/,a,i9,a,a)')` at `newstack.f90:525-526`.
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args([
            "-secs",
            "0,9",
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&result.stdout).ends_with(&format!(
            "\nERROR: NEWSTACK -        9 is an illegal section number for {}\n",
            input.display()
        )),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

/// `-help` must reach `PipPrintHelp` through `PipReadOrParseOptions`
/// (`parse_input_params.f90:165-168`) and exit 0, not print a hand-written
/// synopsis.  Fewer than `minArgs` entries takes the same branch.
#[test]
fn newstack_help_prints_pip_usage_and_exits_zero() {
    for args in [vec!["-help"], vec!["-quiet"]] {
        let result = common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args(&args)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(0), "{args:?}");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            stdout.starts_with("Usage: newstack [Options] input_files... output_file\n"),
            "{stdout}"
        );
        assert!(
            stdout.contains(" -float (-fl)  OR  -FloatDensities"),
            "{stdout}"
        );
    }
}

/// `newstack.f90:2024-2030`: a section outside the input file is written with
/// `dmeanIn`, or with the `-fill` value, not with zero.  Reference binary on a
/// 3x2x1 mode 2 file whose header mean is 3.0 writes 3.0 everywhere and
/// reports 3/3/3 as the output min/max/mean.
#[test]
fn newstack_blank_section_fills_with_input_mean_or_fill_value() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-blankfill-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let filled = base.with_extension("filled.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 3, 2, 1, 2), 0);
        (*header).amin = 3.0;
        (*header).amax = 3.0;
        (*header).amean = 3.0;
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
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!((*header).nz, 2);
        assert_eq!((*header).amin, 3.0);
        assert_eq!((*header).amax, 3.0);
        assert_eq!((*header).amean, 3.0);
        let mut pixels = [0.0_f32; 6];
        for section in 0..2 {
            assert_eq!(
                ii_read_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
            assert_eq!(pixels, [3.0; 6]);
        }
        ii_close(file);
    }
    // `if (ifUseFill .ne. 0) tmpMin = fillVal` (`newstack.f90:2027`).
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                filled.to_str().unwrap(),
                "-secs",
                "7",
                "-blank",
                "-fill",
                "42.5",
            ])
            .status()
            .unwrap()
            .success()
    );
    let filled_c = CString::new(filled.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(filled_c.as_ptr(), c"rb".as_ptr());
        let mut pixels = [0.0_f32; 6];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels, [42.5; 6]);
        ii_close(file);
    }
    for path in [input, output, filled] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:1988-2010` and `findScaleFactors`' `ifFloat == 0 .and.
/// rescale` branch: changing the output mode without `-float` maps the input
/// mode's range onto the output mode's range.  Values are the reference
/// binary's on this fixture (mode 1 in, mode 0 out: 4204 of 32767 -> 32.72).
#[test]
fn newstack_mode_change_rescales_input_mode_range_without_float() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-mode-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 1, 1), 0);
        (*header).amin = 4204.0;
        (*header).amax = 21899.0;
        (*header).amean = 12812.0;
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels = [4204.0_f32, 10000.0, 16000.0, 21899.0];
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-mode",
                "0",
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        // Reference-binary values for this fixture, to the printed precision.
        for (value, expected) in pixels.iter().zip([32.7_f32, 77.8, 124.5, 170.4]) {
            assert!((value - expected).abs() < 1.0, "{pixels:?}");
        }
        // The 255-step output range is fully used, which the unscaled copy
        // this used to produce never was.
        assert!(pixels[3] < 255.0 && pixels[3] > 169.0, "{pixels:?}");
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:2779-2787`: sections whose Z range was flagged as an outlier
/// are counted in `numSecTrunc` and reported after the section table.  The
/// reference binary prints exactly this line for a stack with one outlier
/// section.
#[test]
fn newstack_float_two_reports_extreme_range_note() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-note-{}", std::process::id()));
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
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-float",
            "2",
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(
        stdout.ends_with(
            "\nNOTE:    1 sections had extreme ranges and were truncated to \
             preserve dynamic range \n"
        ),
        "{stdout}"
    );
    // The reference binary's per-section line for the last retained section.
    assert!(
        stdout.contains("       7      0.00    106.00      0.36    255.00    119.44\n"),
        "{stdout}"
    );
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:799` and `dopen.f:20`: the transform file is opened through
/// `DOPEN`, which announces every file it connects.  Reference binary prints
/// the same record before the input header report.
#[test]
fn newstack_dopen_announces_opened_transform_file() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-dopen-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let xform = base.with_extension("xf");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
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
    std::fs::write(&xform, "1 0 0 1 0 0\n").unwrap();
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-xform",
            xform.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success());
    let stdout = String::from_utf8(result.stdout).unwrap();
    // FORMAT(/,1x,A,2X,A,'  file opened: ',A) with ITYPE echoed as entered.
    assert!(
        stdout.starts_with(&format!(
            "\n FORMATTED  ro  file opened: {}\n",
            xform.display()
        )),
        "{stdout}"
    );
    // A missing transform file is rejected by `readCheckWarpFile` before
    // `dopen` ever sees it (`newstack.f90:781-783`, `warputils.c:820`).
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-xform",
            "/nonexistent/no-such.xf",
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: NEWSTACK - OPENING OR READING TRANSFORM FILE\n"
    );
    for path in [input, output, xform] {
        let _ = std::fs::remove_file(path);
    }
}

/// `-bin` takes a PIP integer, so `2,2,1` is accepted as 2 exactly as native
/// does; `-onexform` and `-numout` are ordinary table entries.  All three used
/// to be rejected by a hand-rolled argument loop.
#[test]
fn newstack_accepts_bin_triplet_onexform_and_numout() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-opts-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 4, 2, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..2 {
            let mut pixels = (0..16).map(|value| value as f32).collect::<Vec<_>>();
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    for (tag, extra) in [
        ("bin", vec!["-bin", "2,2,1"]),
        ("onexform", vec!["-onexform"]),
        ("numout", vec!["-numout", "2"]),
    ] {
        let output = base.with_extension(format!("{tag}.mrc"));
        let mut args = vec![
            "-input".to_owned(),
            input.to_str().unwrap().to_owned(),
            "-output".to_owned(),
            output.to_str().unwrap().to_owned(),
        ];
        args.extend(extra.iter().map(|value| (*value).to_owned()));
        let result = common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args(&args)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{tag}: {}",
            String::from_utf8_lossy(&result.stdout)
        );
        let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        unsafe {
            let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
            assert!(!file.is_null(), "{tag}");
            let header = (*file).header.cast::<MrcHeader>();
            assert_eq!((*header).nz, 2, "{tag}");
            assert_eq!((*header).nx, if tag == "bin" { 2 } else { 4 }, "{tag}");
            ii_close(file);
        }
        let _ = std::fs::remove_file(output);
    }
    let _ = std::fs::remove_file(input);
}

/// `-rotate 90` must rotate the pixels, not merely transpose the output
/// dimensions.  Source `newstack.f90:1260-1279` builds `frot` with
/// `frot(1,2) = -sind(rotateAngle)` and `frot(2,1) = -frot(1,2)`, multiplies it
/// by the expansion transform, and `cubInterp` (`newstack.f90:2408`) resamples
/// through it.  The expected image is the byte-identical output of the native
/// `newstack` for this same input.
#[test]
fn newstack_rotate_90_resamples_pixels_as_native_does() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-rot90-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 8, 6, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels: [f32; 48] =
            core::array::from_fn(|i| (10 * (i % 8) + 3 * (i / 8)) as f32 + 0.5);
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
    let expected: [f32; 48] = [
        15.5, 12.5, 9.5, 6.5, 3.5, 0.5, 25.5, 22.5, 19.5, 16.5, 13.5, 10.5, 35.5, 32.5, 29.5, 26.5,
        23.5, 20.5, 45.5, 42.5, 39.5, 36.5, 33.5, 30.5, 55.5, 52.5, 49.5, 46.5, 43.5, 40.5, 65.5,
        62.5, 59.5, 56.5, 53.5, 50.5, 75.5, 72.5, 69.5, 66.5, 63.5, 60.5, 85.5, 82.5, 79.5, 76.5,
        73.5, 70.5,
    ];
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (6, 8));
        let mut pixels = [0.0_f32; 48];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
        assert_eq!(pixels, expected);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `-rotate 180` keeps the output dimensions but must reverse both axes, so it
/// separates a real rotation from the transpose that `newstack.f90:1693` does
/// for the size alone.  The expected image is the byte-identical native output.
#[test]
fn newstack_rotate_180_reverses_both_axes_as_native_does() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-rot180-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 8, 6, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels: [f32; 48] =
            core::array::from_fn(|i| (10 * (i % 8) + 3 * (i / 8)) as f32 + 0.5);
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-rotate",
                "180",
            ])
            .status()
            .unwrap()
            .success()
    );
    let expected: [f32; 48] = [
        85.5, 75.5, 65.5, 55.5, 45.5, 35.5, 25.5, 15.5, 82.5, 72.5, 62.5, 52.5, 42.5, 32.5, 22.5,
        12.5, 79.5, 69.5, 59.5, 49.5, 39.5, 29.5, 19.5, 9.5, 76.5, 66.5, 56.5, 46.5, 36.5, 26.5,
        16.5, 6.5, 73.5, 63.5, 53.5, 43.5, 33.5, 23.5, 13.5, 3.5, 70.5, 60.5, 50.5, 40.5, 30.5,
        20.5, 10.5, 0.5,
    ];
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (8, 6));
        let mut pixels = [0.0_f32; 48];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
        assert_eq!(pixels, expected);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `-ftreduce` must run the source's Fourier crop: `taperOutPad`, forward
/// `todfft`, `fourierReduceImage`, inverse `todfft`, and `irepak2` out of the
/// cropped padded array (`newstack.f90:2439-2500`).  A plain copy of the
/// central 8 by 6 pixels -- what an unimplemented `-ftreduce` produces -- is
/// nowhere near these values.  The expected image is the byte-identical native
/// output.
#[test]
fn newstack_ftreduce_fourier_crops_as_native_does() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-ftr-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 16, 12, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels: [f32; 192] = core::array::from_fn(|i| {
            let (x, y) = ((i % 16) as f32, (i / 16) as f32);
            100.0 + 30.0 * (x / 3.0).sin() * (y / 2.0).cos() + x - y
        });
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-ftreduce",
                "2",
            ])
            .status()
            .unwrap()
            .success()
    );
    let expected: [f32; 48] = [
        104.84710693359375,
        123.7936782836914,
        132.85739135742188,
        130.37692260742188,
        116.94195556640625,
        100.52591705322266,
        87.46034240722656,
        86.12016296386719,
        99.30813598632812,
        106.85552215576172,
        111.17491149902344,
        111.6349868774414,
        108.42706298828125,
        104.40016174316406,
        101.28439331054688,
        102.39579772949219,
        93.23049926757812,
        83.98344421386719,
        81.513671875,
        86.54619598388672,
        98.61357116699219,
        112.9201431274414,
        124.40254211425781,
        128.9793701171875,
        89.2294692993164,
        73.53305053710938,
        68.34504699707031,
        75.00499725341797,
        92.83067321777344,
        114.14210510253906,
        131.2735595703125,
        137.26773071289062,
        89.9690170288086,
        83.95033264160156,
        82.79048156738281,
        86.9637222290039,
        96.11254119873047,
        106.90081787109375,
        115.57984924316406,
        119.46945190429688,
        92.41239929199219,
        103.3392105102539,
        109.02668762207031,
        108.60026550292969,
        102.35617065429688,
        94.66632080078125,
        88.6390380859375,
        89.03550720214844,
    ];
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (8, 6));
        let mut pixels = [0.0_f32; 48];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
        assert_eq!(pixels, expected);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `-ftexpand` takes the `fourierExpandImage` arm of the same block
/// (`newstack.f90:2465-2467`) and `expandFactor = 1. / actualFac`
/// (`newstack.f90:1281`) doubles the output size.  The expected image is the
/// byte-identical native output.
#[test]
fn newstack_ftexpand_fourier_expands_as_native_does() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-fte-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 4, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels: [f32; 16] = core::array::from_fn(|i| (i * i % 13) as f32 + 1.0);
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-ftexpand",
                "2",
            ])
            .status()
            .unwrap()
            .success()
    );
    let expected: [f32; 64] = [
        2.0083959102630615,
        1.5609595775604248,
        0.6232619285583496,
        0.6782815456390381,
        2.8376052379608154,
        6.273039817810059,
        8.934592247009277,
        9.695206642150879,
        -0.5829885005950928,
        1.6800986528396606,
        4.789138317108154,
        5.891191482543945,
        5.60064697265625,
        6.889267444610596,
        9.938937187194824,
        11.725993156433105,
        -0.15856695175170898,
        3.5963070392608643,
        9.589454650878906,
        12.049140930175781,
        10.173576354980469,
        8.793445587158203,
        10.4606351852417,
        12.225069046020508,
        6.754340648651123,
        8.148056030273438,
        10.848499298095703,
        12.88012981414795,
        12.782299041748047,
        11.237120628356934,
        9.879487037658691,
        9.331459999084473,
        13.907453536987305,
        11.35610580444336,
        7.727535724639893,
        7.868696212768555,
        10.986170768737793,
        11.60791015625,
        8.190755844116211,
        5.140545845031738,
        13.006786346435547,
        9.097296714782715,
        3.2177789211273193,
        2.270204544067383,
        6.303021430969238,
        8.56157398223877,
        6.043978691101074,
        3.1491146087646484,
        5.485851287841797,
        3.5634751319885254,
        0.8117215037345886,
        0.26668769121170044,
        2.2936606407165527,
        4.280740737915039,
        4.532779216766357,
        4.062180519104004,
        0.08207952976226807,
        0.5252748727798462,
        1.1031630039215088,
        0.9904545545578003,
        0.8480013012886047,
        2.0358214378356934,
        4.237650394439697,
        5.566634654998779,
    ];
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (8, 8));
        let mut pixels = [0.0_f32; 64];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
        assert_eq!(pixels, expected);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// When an output pixel falls outside the input and no `-fill` was entered,
/// `newstack.f90:2307` takes the `needEdgeMean` branch and `newstack.f90:2392`
/// sets `dmeanSec` from `sliceEdgeMedian` of the loaded lines -- not from the
/// input file's mean.  This 4 by 4 image has a constant border of 10 and an
/// interior of 100, so its mean is 32.5 while every edge median is 10, and the
/// two answers are far apart.  The expected image is the byte-identical native
/// output.
#[test]
fn newstack_fill_outside_image_uses_edge_median_not_file_mean() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-fill-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 4, 4, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels: [f32; 16] = core::array::from_fn(|i| {
            let (x, y) = (i % 4, i / 4);
            if x == 0 || x == 3 || y == 0 || y == 3 {
                10.0
            } else {
                100.0
            }
        });
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-size",
                "6,6",
            ])
            .status()
            .unwrap()
            .success()
    );
    let expected: [f32; 36] = [
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0,
        100.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    ];
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (6, 6));
        let mut pixels = [0.0_f32; 36];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
        // The file mean is 32.5; every filled pixel must be the edge median 10.
        assert_eq!(pixels, expected);
        assert!(pixels.iter().all(|value| *value != 32.5));
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `-phase` shares the source's Fourier block with `-ftreduce`
/// (`newstack.f90:2439-2461`): `taperOutPad` into a `niceFrame` padded array,
/// forward `todfft`, `fourierShiftImage` by the fractional part of the offset,
/// inverse `todfft`, then `irepak2` out of the padded array with the origin
/// moved by `(nxFSpad - nxBin) / 2` (`newstack.f90:2489-2490`).  A plain copy,
/// or the same pipeline without the shift, is nowhere near these values.  The
/// expected image is the byte-identical output of the native `newstack`.
#[test]
fn newstack_phase_shifts_in_fourier_space_as_native_does() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-phase-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 8, 6, 1, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        let mut pixels: [f32; 48] =
            core::array::from_fn(|i| (10 * (i % 8) + 3 * (i / 8)) as f32 + 0.5);
        assert_eq!(
            ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-phase",
                "-offset",
                "0.75,-0.5",
            ])
            .status()
            .unwrap()
            .success()
    );
    let expected: [f32; 48] = [
        6.657503128051758,
        16.32770538330078,
        25.410783767700195,
        34.26654052734375,
        44.073856353759766,
        51.96650314331055,
        63.778465270996094,
        55.542640686035156,
        8.97956657409668,
        19.765546798706055,
        29.95425796508789,
        39.836402893066406,
        50.82899856567383,
        59.63490295410156,
        72.91069030761719,
        63.54486083984375,
        12.076053619384766,
        22.407787322998047,
        32.26414108276367,
        41.75355529785156,
        52.38506317138672,
        60.81877517700195,
        73.68193054199219,
        64.03959655761719,
        15.276275634765625,
        25.838642120361328,
        36.014259338378906,
        45.72804641723633,
        56.69281005859375,
        65.31757354736328,
        78.63493347167969,
        68.3592529296875,
        17.89120101928711,
        28.113037109375,
        38.04607391357422,
        47.46809005737305,
        58.16986083984375,
        66.51513671875,
        79.53158569335938,
        68.97845458984375,
        21.86872100830078,
        32.4111328125,
        42.76980209350586,
        52.50257873535156,
        63.650909423828125,
        72.26080322265625,
        85.8763656616211,
        74.49171447753906,
    ];
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny), (8, 6));
        let mut pixels = [0.0_f32; 48];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        ii_close(file);
        assert_eq!(pixels, expected);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:606-635` is a three-way branch that runs before the
/// `numOutTot .ne. listTotal` check at `newstack.f90:671-672`: one output file
/// takes every section, one output file per section takes one each, and only
/// otherwise is `-numout` consulted — with its own two error messages.  This
/// used to read `NumberToOutput` unconditionally and fall through to the
/// `:672` message for all of them.  Every string and status below is the
/// reference binary's on this fixture.
#[test]
fn newstack_output_section_counts_follow_source_three_way_branch() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-numout-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 5, 2), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..5 {
            let mut pixels = [section as f32, 1.0, 2.0, 3.0];
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    let out = |tag: &str| base.with_extension(format!("{tag}.mrc"));
    let run = |extra: &[&str], outputs: &[std::path::PathBuf]| {
        let mut args = vec!["-input".to_owned(), input.to_str().unwrap().to_owned()];
        for output in outputs {
            args.push("-output".to_owned());
            args.push(output.to_str().unwrap().to_owned());
        }
        args.extend(extra.iter().map(|value| (*value).to_owned()));
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args(&args)
            .output()
            .unwrap()
    };

    // Two outputs and no -numout: the source demands the counts.
    let result = run(&[], &[out("a"), out("b")]);
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: NEWSTACK - You must specify number of sections to write to each \
         output file\n"
    );
    // Too few values for the number of output files.
    let result = run(&["-numout", "2"], &[out("a"), out("b")]);
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: NEWSTACK - The number of values for sections to output does not equal \
         the number of output files\n"
    );
    // Right count of values, wrong total: the `:672` message, and only then.
    let result = run(&["-numout", "2,2"], &[out("a"), out("b")]);
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: NEWSTACK - Number of input and output sections does not match\n"
    );
    // Accumulated over separate entries, as `newstack.f90:620-627` does.
    for extra in [vec!["-numout", "2,3"], vec!["-numout", "2", "-numout", "3"]] {
        let result = run(&extra, &[out("a"), out("b")]);
        assert!(
            result.status.success(),
            "{:?}: {}",
            extra,
            String::from_utf8_lossy(&result.stdout)
        );
        for (tag, nz) in [("a", 2), ("b", 3)] {
            let name = CString::new(out(tag).to_string_lossy().as_bytes()).unwrap();
            unsafe {
                let file = ii_open(name.as_ptr(), c"rb".as_ptr());
                assert!(!file.is_null(), "{tag}");
                assert_eq!((*(*file).header.cast::<MrcHeader>()).nz, nz, "{tag}");
                ii_close(file);
            }
        }
    }
    // One output file takes every section and ignores -numout entirely;
    // one output file per section takes one each and also ignores it.
    let result = run(&["-numout", "3"], &[out("a")]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    let name = CString::new(out("a").to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(name.as_ptr(), c"rb".as_ptr());
        assert_eq!((*(*file).header.cast::<MrcHeader>()).nz, 5);
        ii_close(file);
    }
    let five: Vec<std::path::PathBuf> = ["a", "b", "c", "d", "e"].iter().map(|t| out(t)).collect();
    let result = run(&["-numout", "2,3"], &five);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    for path in &five {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        unsafe {
            let file = ii_open(name.as_ptr(), c"rb".as_ptr());
            assert_eq!((*(*file).header.cast::<MrcHeader>()).nz, 1);
            ii_close(file);
        }
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_file(input);
}

/// `reallocateIfNeeded` (`newstack.f90:2818-2837`) recomputes `lenTemp` from
/// what the reader actually needs -- one element when nothing is being binned
/// or shrunk -- for every limit except an entered `-test` pair.  Leaving the
/// `MAXTEMP` default (five million) in place instead reserves that much of the
/// working array for nothing, which moves `idimInOut` and therefore the chunk
/// boundary: `-memory 39` on an image just over 2.6 million pixels then writes
/// the output in pieces where the source keeps one chunk, and the two disagree.
///
/// The image has to be that big for the regime to exist at all -- the
/// difference is a fixed five million elements -- so it is generated here in
/// mode 0, which keeps it to a few megabytes on disk, rather than checked in.
#[test]
fn newstack_memory_limit_keeps_one_chunk_where_the_source_does() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-lentemp-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let unlimited = base.with_extension("unlimited.mrc");
    let limited = base.with_extension("limited.mrc");
    let xform = base.with_extension("xf");
    let (nx, ny) = (1620_i32, 1620_i32);
    let mut header = vec![0_u8; 1024];
    let put_i32 = |header: &mut Vec<u8>, offset: usize, value: i32| {
        header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    let put_f32 = |header: &mut Vec<u8>, offset: usize, value: f32| {
        header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    for (index, value) in [
        (0, nx),
        (4, ny),
        (8, 1),
        (12, 0),
        (28, nx),
        (32, ny),
        (36, 1),
    ] {
        put_i32(&mut header, index, value);
    }
    for (index, value) in [
        (40, nx as f32),
        (44, ny as f32),
        (48, 1.0),
        (52, 90.0),
        (56, 90.0),
        (60, 90.0),
    ] {
        put_f32(&mut header, index, value);
    }
    for (index, value) in [(64, 1), (68, 2), (72, 3)] {
        put_i32(&mut header, index, value);
    }
    for (index, value) in [(76, 0.0), (80, 255.0), (84, 128.0)] {
        put_f32(&mut header, index, value);
    }
    header[208..212].copy_from_slice(b"MAP ");
    header[212..216].copy_from_slice(&[68, 65, 0, 0]);
    let mut bytes = header;
    bytes.reserve(nx as usize * ny as usize);
    for iy in 0..ny {
        for ix in 0..nx {
            let value = 100.0 + 40.0 * (ix as f32 / 37.0).sin() + 25.0 * (iy as f32 / 23.0).cos();
            bytes.push(value as i8 as u8);
        }
    }
    std::fs::write(&input, &bytes).unwrap();
    std::fs::write(&xform, "1 0 0 1 7.5 -3.25\n").unwrap();

    // `-memory 39` is the smallest entry the source accepts: it requires
    // `lenTemp <= limToAlloc / 2` and `lenTemp` is still `MAXTEMP` when that
    // check runs (`newstack.f90:1035-1036`).
    for (output, memory) in [(&unlimited, None), (&limited, Some("39"))] {
        let mut command = common::imod_cmd("newstack");
        command.env("AUTODOC_DIR", AUTODOC);
        command.args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-xform",
            xform.to_str().unwrap(),
            "-mode",
            "2",
        ]);
        if let Some(memory) = memory {
            command.args(["-memory", memory]);
        }
        assert!(command.status().unwrap().success());
    }
    let mask = |path: &std::path::Path| {
        let mut bytes = std::fs::read(path).unwrap();
        bytes[224..1024].fill(0);
        bytes
    };
    assert_eq!(
        mask(&unlimited),
        mask(&limited),
        "-memory must not change the output when the source keeps one chunk"
    );
    for path in [&input, &unlimited, &limited, &xform] {
        let _ = std::fs::remove_file(path);
    }
}

/// Reads every directory of the TIFF at [path] and returns what each one
/// carries in `ImageDescription` (tag 270), `SMinSampleValue` (tag 340) and
/// `SMaxSampleValue` (tag 341).  A multi-page write sets those per directory,
/// so a stack has to be inspected page by page rather than through the first
/// IFD alone.
fn tiff_page_description_and_min_max(
    path: &std::path::Path,
) -> Vec<(Option<String>, Option<f64>, Option<f64>)> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        &bytes[0..2],
        b"II",
        "the libtiff writer makes little-endian files"
    );
    let short_at =
        |offset: usize| u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
    let long_at =
        |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    let mut pages = Vec::new();
    let mut directory = long_at(4);
    while directory != 0 {
        let entries = short_at(directory) as usize;
        let mut description = None;
        let mut minimum = None;
        let mut maximum = None;
        for entry in 0..entries {
            let field = directory + 2 + entry * 12;
            let count = long_at(field + 4);
            match short_at(field) {
                270 => {
                    let start = if count > 4 {
                        long_at(field + 8)
                    } else {
                        field + 8
                    };
                    description = Some(
                        String::from_utf8_lossy(&bytes[start..start + count - 1]).into_owned(),
                    );
                }
                tag @ (340 | 341) => {
                    // libtiff gives `SMinSampleValue`/`SMaxSampleValue` the
                    // field type of the sample data, so a float image stores a
                    // 4-byte TIFF_FLOAT inline while a wider type is written
                    // out of line.
                    let field_type = short_at(field + 2);
                    let size = match field_type {
                        1 | 6 => 1,
                        3 | 8 => 2,
                        11 => 4,
                        12 => 8,
                        other => panic!("unexpected sample value field type {other}"),
                    };
                    let start = if size * count > 4 {
                        long_at(field + 8)
                    } else {
                        field + 8
                    };
                    let value = match field_type {
                        1 => f64::from(bytes[start]),
                        6 => f64::from(bytes[start] as i8),
                        3 => f64::from(short_at(start)),
                        8 => f64::from(short_at(start) as i16),
                        11 => f64::from(f32::from_le_bytes(
                            bytes[start..start + 4].try_into().unwrap(),
                        )),
                        _ => f64::from_le_bytes(bytes[start..start + 8].try_into().unwrap()),
                    };
                    if tag == 340 {
                        minimum = Some(value);
                    } else {
                        maximum = Some(value);
                    }
                }
                _ => {}
            }
        }
        pages.push((description, minimum, maximum));
        directory = long_at(directory + 2 + entries * 12);
    }
    pages
}

/// Writing a TIFF stack, every directory carries the input's labels in
/// `ImageDescription` and a min/max pair, not just the first one.
/// `iiuWriteLines` re-syncs the ImodImageFile from the unit's MRC header
/// before each write "because the write is the time when it matters"
/// (`unit_fileio.c:661`, `unit_fileio.c:715-716`), and that sync re-makes the
/// static `sDescription` (`iitif.c:809`) that `tiffWriteSetup` frees again
/// after writing it into a directory (`iitif.c:2600-2603`).  The min/max come
/// from the same synced `amin`/`amax` through `constrainAndStoreMinMax`
/// (`iitif.c:2591-2593`), which is why the first directories carry the input
/// header's range while `tiffClose` (`iitif.c:696-700`) overwrites the last
/// one with the range `iiuWriteHeader` left behind.
#[test]
fn newstack_tiff_stack_repeats_description_and_min_max_on_every_directory() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-pages-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.tif");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 4, 3, 3, 2), 0);
        // A range deliberately wider than the data, so the directories that
        // take the input header's range are told apart from the last one.
        header.amin = -5.0;
        header.amax = 250.0;
        header.amean = 3.0;
        header.nlabl = 2;
        header.labels[0][..80].copy_from_slice(
            b"imod-rs multi-page TIFF fixture                                                 ",
        );
        header.labels[1][..80].copy_from_slice(
            b"second label line                                                               ",
        );
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels: Vec<f32> = (0..36).map(|value| value as f32).collect();
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 4, pixels.len(), file),
            36
        );
        libc::fclose(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-format",
                "tif",
                "-secs",
                "0,1,2",
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success()
    );
    let pages = tiff_page_description_and_min_max(&output);
    assert_eq!(pages.len(), 3, "one directory per written section");
    let description = "imod-rs multi-page TIFF fixture\nsecond label line";
    for (index, page) in pages.iter().enumerate() {
        assert_eq!(
            page.0.as_deref(),
            Some(description),
            "directory {index} must carry the MRC labels"
        );
    }
    assert_eq!((pages[0].1, pages[0].2), (Some(-5.0), Some(250.0)));
    assert_eq!((pages[1].1, pages[1].2), (Some(-5.0), Some(250.0)));
    assert_eq!((pages[2].1, pages[2].2), (Some(0.0), Some(35.0)));
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:1584-1618` reads a typed extended header with
/// `iiuRetExtendedData`/`iiuRetExtendedType`, `newstack.f90:1907-1931` sizes
/// the output with `getExtraHeaderMaxSecSize`, and `newstack.f90:2638-2683`
/// copies one `getExtraHeaderSecOffset` record per selected section.  A
/// SerialEM header (`nint` = bytes per section, `nreal` = flags with bit 0 for
/// the tilt angle, bit 1 for the piece coordinates and bit 2 for the stage
/// position) must come through with its 12 bytes per selected section and its
/// `nint`/`nreal` intact.  Compared byte for byte against the reference
/// `newstack` on the same input.
#[test]
fn newstack_copies_serialem_typed_extended_header_for_selected_sections() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-seri-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    let mut extra = [0_u8; 72];
    for section in 0..6_usize {
        let record = &mut extra[section * 12..section * 12 + 12];
        record[0..2].copy_from_slice(&((100 * (-60 + 20 * section as i32)) as i16).to_ne_bytes());
        record[2..4].copy_from_slice(&(section as i16).to_ne_bytes());
        record[4..6].copy_from_slice(&0_i16.to_ne_bytes());
        record[6..8].copy_from_slice(&(3 * section as i16).to_ne_bytes());
        record[8..10].copy_from_slice(&(100 + section as i16).to_ne_bytes());
        record[10..12].copy_from_slice(&(200 + section as i16).to_ne_bytes());
    }
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 6, 2), 0);
        (*header).nint = 12;
        (*header).nreal = 7;
        let mut bytes = extra;
        assert_eq!(mrc_write_extra_header(header, bytes.as_mut_ptr(), 72), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..6 {
            let mut pixels = [section as f32, 1.0, 2.0, 3.0];
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-secs",
                "1,3,5",
            ])
            .status()
            .unwrap()
            .success()
    );
    let written = std::fs::read(&output).unwrap();
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut header = std::mem::zeroed::<MrcHeader>();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!(header.next, 36);
        assert_eq!((header.nint, header.nreal), (12, 7));
        ii_close(file);
    }
    for (index, section) in [1_usize, 3, 5].iter().enumerate() {
        assert_eq!(
            &written[1024 + index * 12..1024 + index * 12 + 12],
            &extra[section * 12..section * 12 + 12],
        );
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:1907` excludes `-strip` from the extended-header branch, so
/// the `else` arm's `iiuAltNumExtended(2, 0)` leaves the output with no
/// extended data while `iiuTransExtendedData` (`unit_header.c:1198-1200`) has
/// already copied `nint` and `nreal` from the input.  Verified against the
/// reference binary, which produces exactly this header.
#[test]
fn newstack_strip_keeps_serialem_type_fields_with_no_extended_data() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-strip-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 2, 2), 0);
        (*header).nint = 12;
        (*header).nreal = 7;
        let mut extra = [3_u8; 24];
        assert_eq!(mrc_write_extra_header(header, extra.as_mut_ptr(), 24), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..2 {
            let mut pixels = [section as f32, 1.0, 2.0, 3.0];
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-strip",
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
        assert_eq!(header.next, 0);
        assert_eq!((header.nint, header.nreal), (12, 7));
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:2655-2678`: with `-tilt` into a SerialEM header the angle
/// goes in as `nint(100. * extraTilts(isec))` in the first short, and the copy
/// then starts two bytes into the input record so the rest of the section's
/// data follows unchanged.  Byte-compared against the reference binary.
#[test]
fn newstack_tilt_replaces_serialem_tilt_short_and_keeps_the_rest() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-seritilt-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let angles = base.with_extension("tlt");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    let mut extra = [0_u8; 24];
    for section in 0..2_usize {
        let record = &mut extra[section * 12..section * 12 + 12];
        record[0..2].copy_from_slice(&(-1234_i16).to_ne_bytes());
        for byte in 2..12 {
            record[byte] = (section * 12 + byte) as u8;
        }
    }
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 2, 2), 0);
        (*header).nint = 12;
        (*header).nreal = 7;
        let mut bytes = extra;
        assert_eq!(mrc_write_extra_header(header, bytes.as_mut_ptr(), 24), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..2 {
            let mut pixels = [section as f32, 1.0, 2.0, 3.0];
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    std::fs::write(&angles, "-60.25\n12.5\n").unwrap();
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
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
    let written = std::fs::read(&output).unwrap();
    for (section, angle) in [-60.25_f32, 12.5].iter().enumerate() {
        let record = &written[1024 + section * 12..1024 + section * 12 + 12];
        assert_eq!(
            i16::from_ne_bytes([record[0], record[1]]),
            (100. * angle).round() as i16
        );
        assert_eq!(&record[2..12], &extra[section * 12 + 2..section * 12 + 12]);
    }
    for path in [input, output, angles] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:1624-1650,1667-1683`: `-reorder` with no `-angle` file pulls
/// the angles out of the extended header with `get_extra_header_tilts` and
/// swaps both the section list and the angle array.  `-reorder -1` orders
/// descending, so the sections come out reversed and each copied SerialEM
/// record follows its section.  Verified against the reference binary.
#[test]
fn newstack_reorder_uses_extended_header_tilt_angles() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-reorder-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    let mut extra = [0_u8; 36];
    for section in 0..3_usize {
        let record = &mut extra[section * 12..section * 12 + 12];
        record[0..2].copy_from_slice(&((100 * (-20 + 20 * section as i32)) as i16).to_ne_bytes());
        for byte in 2..12 {
            record[byte] = (section * 12 + byte) as u8;
        }
    }
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 3, 2), 0);
        (*header).nint = 12;
        (*header).nreal = 7;
        let mut bytes = extra;
        assert_eq!(mrc_write_extra_header(header, bytes.as_mut_ptr(), 36), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..3 {
            let mut pixels = [section as f32, 1.0, 2.0, 3.0];
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    assert!(
        common::imod_cmd("newstack")
            .env("AUTODOC_DIR", AUTODOC)
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-reorder",
                "-1",
            ])
            .status()
            .unwrap()
            .success()
    );
    let written = std::fs::read(&output).unwrap();
    for (index, section) in [2_usize, 1, 0].iter().enumerate() {
        assert_eq!(
            &written[1024 + index * 12..1024 + index * 12 + 12],
            &extra[section * 12..section * 12 + 12],
        );
    }
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut pixels = [0.0_f32; 4];
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
            0
        );
        assert_eq!(pixels[0], 2.0);
        assert_eq!(
            ii_read_section_float(file, pixels.as_mut_ptr().cast(), 2),
            0
        );
        assert_eq!(pixels[0], 0.0);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:1610-1613,1661-1662`: an `FEI1` extended type comes back from
/// `iiuRetExtendedType` as `numIntOrBytesIn == -3`, which marks `itype = 2`,
/// and storing tilt angles into one is refused by name.  The message and the
/// exit status are the reference binary's.
#[test]
fn newstack_refuses_saving_tilt_angles_into_an_fei1_extended_header() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-fei1-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let angles = base.with_extension("tlt");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    let mut extra = [0_u8; 48];
    for section in 0..2_usize {
        extra[section * 24..section * 24 + 4].copy_from_slice(&24_i32.to_ne_bytes());
    }
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 2, 2), 0);
        (*header).ext_type = *b"FEI1";
        (*header).nversion = 20140;
        assert_eq!(mrc_write_extra_header(header, extra.as_mut_ptr(), 48), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..2 {
            let mut pixels = [section as f32, 1.0, 2.0, 3.0];
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    std::fs::write(&angles, "-60.0\n0.0\n").unwrap();
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-tilt",
            angles.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&result.stderr), "");
    assert!(
        String::from_utf8_lossy(&result.stdout).ends_with(
            "\nERROR: NEWSTACK - You cannot store tilt angles back into a new FEI1-style \
             extended header\n"
        ),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    for path in [input, output, angles] {
        let _ = std::fs::remove_file(path);
    }
}

/// `newstack.f90:1619-1621`: a SerialEM header whose flags have bit 0 clear
/// carries no tilt angle, so there is nowhere to put one.  `nint = 10` with
/// `nreal = 6` is a valid `extraIsNbytesAndFlags` pair (three piece shorts and
/// two stage shorts), which is what makes this reachable.  Message and status
/// from the reference binary.
#[test]
fn newstack_refuses_tilt_angles_for_a_serialem_header_without_the_tilt_flag() {
    let base = std::env::temp_dir().join(format!("imod-rs-newstack-noflag-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let angles = base.with_extension("tlt");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        assert!(!file.is_null());
        let header = (*file).header.cast::<MrcHeader>();
        assert_eq!(mrc_head_new(&mut *header, 2, 2, 2, 2), 0);
        (*header).nint = 10;
        (*header).nreal = 6;
        let mut extra = [5_u8; 20];
        assert_eq!(mrc_write_extra_header(header, extra.as_mut_ptr(), 20), 0);
        ii_sync_from_mrc_header(file, header);
        assert_eq!(mrc_head_write((*file).fp, header), 0);
        for section in 0..2 {
            let mut pixels = [section as f32, 1.0, 2.0, 3.0];
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), section),
                0
            );
        }
        ii_close(file);
    }
    std::fs::write(&angles, "-60.0\n0.0\n").unwrap();
    let result = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-tilt",
            angles.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&result.stderr), "");
    assert!(
        String::from_utf8_lossy(&result.stdout).ends_with(
            "\nERROR: NEWSTACK - You cannot save tilt angles into a SerialEM extended header \
             that was not saved with tilt angles\n"
        ),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    for path in [input, output, angles] {
        let _ = std::fs::remove_file(path);
    }
}

/// `-float 2` and `-float 3` under a `-memory` or `-test` limit: the source
/// writes each output chunk to temporary storage, works out the scale factors
/// from the section as a whole, and then loops **backwards** over the chunks
/// rescaling and writing (`newstack.f90:2580-2611`).  When `ifOutChunk` is
/// positive the temporary storage is a scratch file on unit 3
/// (`newstack.f90:2284-2296`), whose open prints two lines of its own.  The
/// image has to be big enough for the limit to force several chunks, so it is
/// built here rather than taken from a fixture.
#[test]
fn newstack_float_two_and_three_scale_under_a_memory_limit() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-newstack-floatchunk-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let xform = base.with_extension("xf");
    let (nx, ny, nz) = (1620_i32, 1620_i32, 2_i32);
    let mut header = vec![0_u8; 1024];
    let put_i32 = |header: &mut Vec<u8>, offset: usize, value: i32| {
        header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    let put_f32 = |header: &mut Vec<u8>, offset: usize, value: f32| {
        header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    for (index, value) in [
        (0, nx),
        (4, ny),
        (8, nz),
        (12, 2),
        (28, nx),
        (32, ny),
        (36, nz),
    ] {
        put_i32(&mut header, index, value);
    }
    for (index, value) in [
        (40, nx as f32),
        (44, ny as f32),
        (48, nz as f32),
        (52, 90.0),
        (56, 90.0),
        (60, 90.0),
    ] {
        put_f32(&mut header, index, value);
    }
    for (index, value) in [(64, 1), (68, 2), (72, 3)] {
        put_i32(&mut header, index, value);
    }
    let mut minimum = f32::MAX;
    let mut maximum = f32::MIN;
    let mut total = 0.0_f64;
    let mut pixels = Vec::<u8>::with_capacity((nx * ny * nz) as usize * 4);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let value = 1000.0
                    + 500.0 * (ix as f32 / 37.0 + iz as f32).sin()
                    + 300.0 * (iy as f32 / 53.0).cos()
                    + 40.0 * ((ix * 7 + iy * 13 + iz * 29) % 97) as f32 / 97.0;
                minimum = minimum.min(value);
                maximum = maximum.max(value);
                total += f64::from(value);
                pixels.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    for (index, value) in [
        (76, minimum),
        (80, maximum),
        (84, (total / f64::from(nx * ny * nz)) as f32),
    ] {
        put_f32(&mut header, index, value);
    }
    header[208..212].copy_from_slice(b"MAP ");
    header[212..216].copy_from_slice(&[68, 65, 0, 0]);
    let mut bytes = header;
    bytes.extend_from_slice(&pixels);
    std::fs::write(&input, &bytes).unwrap();
    std::fs::write(
        &xform,
        "   0.9980000   0.0300000  -0.0300000   0.9980000    2.100    -1.700\n   \
         0.9980000   0.0300000  -0.0300000   0.9980000    4.200    -3.400\n",
    )
    .unwrap();

    // `-test lim,temp` is the only way to push the limit below what `-memory`
    // accepts: `newstack.f90:1035-1036` refuses a memory entry whose
    // `limToAlloc / 2` is under the still-unrecomputed `MAXTEMP`.  2_000_000
    // elements hold the whole output (`ifOutChunk` 0), 700_000 and 300_000 do
    // not, so those take the scratch file.
    for float in ["2", "3"] {
        let mut unlimited_output = 0_usize;
        for (limits, wants_scratch) in [
            (None, false),
            (Some("3000000,1000"), false),
            (Some("700000,1000"), true),
            (Some("300000,1000"), true),
        ] {
            let output = base.with_extension(format!(
                "out-{float}-{}.mrc",
                limits.unwrap_or("none").replace(',', "-")
            ));
            let mut command = common::imod_cmd("newstack");
            command.env("AUTODOC_DIR", AUTODOC);
            command.args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-xform",
                xform.to_str().unwrap(),
                "-mode",
                "1",
                "-float",
                float,
            ]);
            if let Some(limits) = limits {
                command.args(["-test", limits]);
            }
            let result = command.output().unwrap();
            assert!(
                result.status.success(),
                "-float {float} with -test {limits:?} must not be refused: {}",
                String::from_utf8_lossy(&result.stdout)
            );
            let printed = String::from_utf8_lossy(&result.stdout).into_owned();
            assert_eq!(
                printed.contains(" SCRATCH image file on unit   3 : "),
                wants_scratch,
                "the scratch file is opened exactly when the source's `ifOutChunk` \
                 is positive and there is more than one chunk: -float {float} \
                 -test {limits:?}\n{printed}"
            );
            let written = std::fs::read(&output).unwrap();
            assert_eq!(
                written.len(),
                1024 + 2 * (nx * ny * nz) as usize,
                "every chunk of every section has to reach the output"
            );
            let values: Vec<i16> = written[1024..]
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes(pair.try_into().unwrap()))
                .collect();
            // The per-chunk statistics are what `findScaleFactors` reads, so
            // a different chunk count is a different scale factor and the
            // limited runs are *not* expected to match the unlimited one
            // pixel for pixel.  What has to hold is that every run covers the
            // same range of the input.
            let (low, high) = values
                .iter()
                .fold((i32::MAX, i32::MIN), |(low, high), value| {
                    (low.min(i32::from(*value)), high.max(i32::from(*value)))
                });
            assert!(
                high - low > 20000,
                "-float {float} with -test {limits:?} must still spread the section \
                 over the output range: {low}..{high}"
            );
            if limits.is_none() {
                unlimited_output = values.len();
            }
            assert_eq!(values.len(), unlimited_output);
            // Against a real IMOD build the comparison is the whole file, with
            // only the label area masked: `mrc_head_new` never clears
            // `hdata->labels`, and the date stamp differs by a second between
            // two runs.
            if let Ok(native_newstack) = std::env::var("IMOD_NATIVE_NEWSTACK") {
                let native_output = base.with_extension(format!(
                    "native-{float}-{}.mrc",
                    limits.unwrap_or("none").replace(',', "-")
                ));
                let mut native = Command::new(native_newstack);
                native.env(
                    "AUTODOC_DIR",
                    std::env::var("AUTODOC_DIR")
                        .unwrap_or_else(|_| "/tmp/imod-reference-build/autodoc".into()),
                );
                native.args([
                    "-input",
                    input.to_str().unwrap(),
                    "-output",
                    native_output.to_str().unwrap(),
                    "-xform",
                    xform.to_str().unwrap(),
                    "-mode",
                    "1",
                    "-float",
                    float,
                ]);
                if let Some(limits) = limits {
                    native.args(["-test", limits]);
                }
                assert!(native.status().unwrap().success());
                let mut reference = std::fs::read(&native_output).unwrap();
                let mut mine = written.clone();
                reference[224..1024].fill(0);
                mine[224..1024].fill(0);
                assert_eq!(
                    reference, mine,
                    "-float {float} with -test {limits:?} must match the native output"
                );
                let _ = std::fs::remove_file(&native_output);
            }
            let _ = std::fs::remove_file(&output);
        }
    }
    for path in [&input, &xform] {
        let _ = std::fs::remove_file(path);
    }
}

/// `-verbose 1` drives the 25 `if (iVerbose > 0)` reports in
/// `newstack.f90`.  All but `newstack.f90:2788` are Fortran list-directed
/// `print *`, whose field widths are part of the bytes: `integer*4` right
/// justified in 11, `integer(kind = 8)` in 20, a logical in 1, a `real*4` as
/// `G16.9E2` with a scale factor of 1, and every item after the first
/// preceded by one blank inside a record that itself starts with a blank.
/// `newstack.f90:2788` is a formatted `write` with no leading blank and
/// `f8.4` fields.
#[test]
fn newstack_verbose_reports_use_fortran_list_directed_fields() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-verbose-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let native_output = base.with_extension("native-output.mrc");
    let xform = base.with_extension("xf");
    let (nx, ny, nz) = (96_i32, 64_i32, 3_i32);
    let mut bytes = vec![0_u8; 1024];
    let put_i32 = |bytes: &mut Vec<u8>, offset: usize, value: i32| {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    let put_f32 = |bytes: &mut Vec<u8>, offset: usize, value: f32| {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    for (offset, value) in [
        (0, nx),
        (4, ny),
        (8, nz),
        (12, 2),
        (28, nx),
        (32, ny),
        (36, nz),
        (64, 1),
        (68, 2),
        (72, 3),
    ] {
        put_i32(&mut bytes, offset, value);
    }
    for (offset, value) in [
        (40, nx as f32),
        (44, ny as f32),
        (48, nz as f32),
        (52, 90.0),
        (56, 90.0),
        (60, 90.0),
        (76, 20.0),
        (80, 200.0),
        (84, 110.0),
    ] {
        put_f32(&mut bytes, offset, value);
    }
    bytes[208..212].copy_from_slice(b"MAP ");
    bytes[212..216].copy_from_slice(&[68, 65, 0, 0]);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let value = 110.0
                    + 60.0 * ((ix + 3 * iz) as f32 / 11.0).sin()
                    + 25.0 * (iy as f32 / 7.0).cos();
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    std::fs::write(&input, &bytes).unwrap();
    std::fs::write(
        &xform,
        "1 0 0 1 2.5 -1.5\n1 0 0 1 2.5 -1.5\n1 0 0 1 2.5 -1.5\n",
    )
    .unwrap();

    let arguments = |out: &std::path::Path| {
        vec![
            "-input".to_owned(),
            input.to_string_lossy().into_owned(),
            "-output".to_owned(),
            out.to_string_lossy().into_owned(),
            "-xform".to_owned(),
            xform.to_string_lossy().into_owned(),
            "-verbose".to_owned(),
            "1".to_owned(),
        ]
    };
    let run = common::imod_cmd("newstack")
        .env("AUTODOC_DIR", AUTODOC)
        .args(arguments(&output))
        .output()
        .unwrap();
    assert!(run.status.success());
    let stdout = String::from_utf8(run.stdout).unwrap();

    // `print *,'Size and offsets X:', nxBin, rxOffset, ', Y:', nyBin, ryOffset`
    // (`newstack.f90:1572-1573`): two `integer*4` in 11 and two `real*4` in
    // `G16.9E2`, whose F form pads with four trailing blanks.
    assert!(
        stdout.contains(
            " Size and offsets X:          96   0.00000000     , Y:          64   0.00000000    \n"
        ),
        "{stdout}"
    );
    // `print *,'rescale', rescale` (`newstack.f90:2046`): a logical is one
    // character behind one separating blank.
    assert_eq!(
        stdout.matches(" rescale F\n").count(),
        nz as usize,
        "{stdout}"
    );
    // `print *,'loading whole region', needYstart, needYend, numLinesLoad`
    // (`newstack.f90:2373-2374`).
    assert_eq!(
        stdout
            .matches(" loading whole region           0          63          64\n")
            .count(),
        nz as usize,
        "{stdout}"
    );
    // `print *,'writing', iChunk` (`newstack.f90:3255`).
    assert_eq!(
        stdout.matches(" writing           1\n").count(),
        nz as usize,
        "{stdout}"
    );
    // `print *,'preSetScaling ', preSetScaling, '   processInPlace ',
    // processInPlace` (`newstack.f90:2184-2185`).
    assert_eq!(
        stdout
            .matches(" preSetScaling  T    processInPlace  F\n")
            .count(),
        nz as usize,
        "{stdout}"
    );
    // `print *,'linesleft', linesLeft, '  nchunk', numChunks`
    // (`newstack.f90:2198`).
    assert_eq!(
        stdout
            .matches(" linesleft          64   nchunk           1\n")
            .count(),
        nz as usize,
        "{stdout}"
    );
    // `print *,'number of chunks:', numChunks, ifOutChunk` and the
    // `i, lineInSt(i), numLinesIn(i), lineOutSt(i), numLinesOut(i)` row under
    // it (`newstack.f90:2276-2281`).
    assert_eq!(
        stdout
            .matches(" number of chunks:           1           1\n           1           0          64           0          64\n")
            .count(),
        nz as usize,
        "{stdout}"
    );
    // `print *,'reallocate sizes:', nxOut, nyOut, nxBin, nyBin, needDim,
    // needTemp` (`newstack.f90:2865-2866`), whose `needDim` is
    // `integer(kind = 8)` and so 20 columns wide.
    assert!(
        stdout.contains(
            " reallocate sizes:          96          64          96          64                12288           1\n"
        ),
        "{stdout}"
    );
    // `write(*,'(a,f8.4,a,f8.4,a,f8.4,a,f8.4,a,f8.4)')`
    // (`newstack.f90:2788-2790`): no leading record blank, and five `f8.4`
    // fields.  The times themselves are wall clock.
    let timings = stdout
        .lines()
        .find(|line| line.starts_with("loadtime"))
        .unwrap_or_else(|| panic!("{stdout}"));
    assert_eq!(timings.len(), 8 + 8 + 10 + 8 + 5 + 8 + 9 + 8 + 7 + 8);
    for (offset, label) in [
        (8, "loadtime"),
        (26, "  savetime"),
        (39, "  sum"),
        (56, "  rottime"),
        (71, "  taper"),
    ] {
        let value = &timings[offset..offset + 8];
        assert!(value.trim().parse::<f64>().is_ok(), "{label} {value:?}");
    }
    // `print *,'did iclden ', tmin2, tmax2, tmpMin, tmpMax`
    // (`newstack.f90:2531`): four `G16.9E2` fields, each one separating blank
    // plus twelve columns of F editing plus four blanks.
    let iclden: Vec<&str> = stdout
        .lines()
        .filter(|line| line.starts_with(" did iclden "))
        .collect();
    assert_eq!(iclden.len(), nz as usize, "{stdout}");
    for line in &iclden {
        assert_eq!(line.len(), " did iclden ".len() + 4 * 17, "{line:?}");
        for field in 0..4 {
            let start = " did iclden ".len() + field * 17;
            assert_eq!(&line[start..start + 1], " ", "{line:?}");
            let value = &line[start + 1..start + 17];
            assert_eq!(value.len(), 16);
            assert!(value.ends_with("    "), "{value:?}");
            assert!(value[..12].trim().parse::<f64>().is_ok(), "{value:?}");
        }
    }

    // `newstack.f90:2788` reports wall-clock times, which differ between two
    // runs of the same command whichever binary makes them.  Everything else
    // has to match.
    let comparable = |text: &str, own_output: &std::path::Path| -> String {
        text.replace(&own_output.to_string_lossy().into_owned(), "OUTPUT")
            .lines()
            .filter(|line| !line.starts_with("loadtime"))
            // The MRC label date stamp differs by a second between two runs.
            .filter(|line| !line.contains("Titles") && !line.contains("-20"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    if let Ok(native_newstack) = std::env::var("IMOD_NATIVE_NEWSTACK") {
        let native = Command::new(native_newstack)
            .env(
                "AUTODOC_DIR",
                std::env::var("AUTODOC_DIR")
                    .unwrap_or_else(|_| "/tmp/imod-reference-build/autodoc".into()),
            )
            .args(arguments(&native_output))
            .output()
            .unwrap();
        assert!(native.status.success());
        assert_eq!(
            comparable(&String::from_utf8(native.stdout).unwrap(), &native_output),
            comparable(&stdout, &output),
            "verbose stdout must match the native command outside the wall-clock \
             timing report"
        );
        let _ = std::fs::remove_file(&native_output);
    }
    for path in [&input, &output, &xform] {
        let _ = std::fs::remove_file(path);
    }
}

/// The source's chunk loop reads each chunk's own input window
/// (`newstack.f90:2341-2382`), and the read start is
/// `ryOffset + readReduction * loadYoffset` (`:2377`) -- the *binned* offset
/// scaled by the reduction, not the raw line number.  A route that loaded the
/// whole section once and then sliced it made the same pixels while doing none
/// of that, and any error in the per-chunk offset would move every line of
/// every chunk but the first.  `-bin 2 -test 16000,1024` splits this 512 by 400
/// input into seven chunks of 29, 29, 29, 29, 28, 28 and 28 output lines whose
/// input windows are 0-28, 29-57, 58-86, 87-115, 116-143, 144-171 and 172-199;
/// the reference reports exactly that and its output is byte-identical to the
/// unchunked run.
#[test]
fn newstack_binned_chunks_load_their_own_window() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-chunkload-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let whole = base.with_extension("whole.mrc");
    let chunked = base.with_extension("chunked.mrc");
    let (nx, ny, nz) = (512_i32, 400_i32, 2_i32);
    let mut bytes = vec![0_u8; 1024];
    let put_i32 = |bytes: &mut Vec<u8>, offset: usize, value: i32| {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    let put_f32 = |bytes: &mut Vec<u8>, offset: usize, value: f32| {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    for (offset, value) in [
        (0, nx),
        (4, ny),
        (8, nz),
        (12, 0),
        (28, nx),
        (32, ny),
        (36, nz),
        (64, 1),
        (68, 2),
        (72, 3),
    ] {
        put_i32(&mut bytes, offset, value);
    }
    for (offset, value) in [
        (40, nx as f32),
        (44, ny as f32),
        (48, nz as f32),
        (52, 90.0),
        (56, 90.0),
        (60, 90.0),
        (76, 0.0),
        (80, 250.0),
        (84, 125.0),
    ] {
        put_f32(&mut bytes, offset, value);
    }
    bytes[208..212].copy_from_slice(b"MAP ");
    bytes[212..216].copy_from_slice(&[68, 65, 0, 0]);
    for section in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                bytes.push(((ix * 7 + iy * 13 + section * 29) % 251) as u8);
            }
        }
    }
    std::fs::write(&input, &bytes).unwrap();
    let run = |output: &std::path::Path, limit: Option<&str>| -> String {
        let mut command = common::imod_cmd("newstack");
        command.env("AUTODOC_DIR", AUTODOC);
        command.args([
            "-input",
            input.to_str().unwrap(),
            "-output",
            output.to_str().unwrap(),
            "-bin",
            "2",
            "-mode",
            "2",
            "-verbose",
            "1",
        ]);
        if let Some(limit) = limit {
            command.args(["-test", limit]);
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        String::from_utf8_lossy(&result.stdout).into_owned()
    };
    let whole_stdout = run(&whole, None);
    assert_eq!(
        whole_stdout.matches(" loading whole region").count(),
        nz as usize,
        "one chunk a section without a limit:\n{whole_stdout}"
    );
    let chunked_stdout = run(&chunked, Some("16000,1024"));
    let windows = chunked_stdout
        .lines()
        .filter(|line| line.starts_with(" loading whole region"))
        .map(|line| {
            line.split_whitespace()
                .skip(3)
                .map(|field| field.parse::<i32>().unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        windows,
        [
            [0, 28, 29],
            [29, 57, 29],
            [58, 86, 29],
            [87, 115, 29],
            [116, 143, 28],
            [144, 171, 28],
            [172, 199, 28],
        ]
        .repeat(nz as usize)
        .iter()
        .map(|window| window.to_vec())
        .collect::<Vec<_>>(),
        "stdout was:\n{chunked_stdout}"
    );
    assert_eq!(
        chunked_stdout.matches(" did repack").count(),
        7 * nz as usize,
        "stdout was:\n{chunked_stdout}"
    );
    // The MRC labels carry a date stamp and uninitialised slots past `nlabl`,
    // so only the data can be compared byte for byte.
    assert_eq!(
        std::fs::read(&whole).unwrap()[1024..],
        std::fs::read(&chunked).unwrap()[1024..],
        "the per-chunk binned reads must land on the same pixels"
    );
    for path in [&input, &whole, &chunked] {
        let _ = std::fs::remove_file(path);
    }
}

/// Cross-backend differential for every `newstack` option that reaches an FFT:
/// `-ftreduce` and `-ftexpand` (`fourierReduceImage`/`fourierExpandImage`
/// around a forward and inverse `todfft`, `newstack.f90:2455-2470`) and
/// `-phase` (`fourierShiftImage` between the same pair,
/// `newstack.f90:2456`).  Intended backends: the same command runs once on
/// `parity` and once on `rustfft`, in separate processes because the selector
/// is a process-local `OnceLock`.
///
/// `-phase` is the case a sign-convention error cannot survive: the shift is a
/// phase ramp applied to the forward spectrum, so a conjugated transform moves
/// the image the other way instead of differing in the last bits.  The input
/// sizes are deliberately not powers of two -- `newstack` pads each to a
/// `niceFrame` size, whose largest prime factor is `niceFFTlimit`'s 5
/// (`odfft.c:212`), so these exercise the radix-3 and radix-5 kernels rather
/// than radix 2 alone.  Lengths with larger prime factors are not reachable
/// from this command and are covered directly in `tests/fft_backend_matrix.rs`.
///
/// Tolerances: measured, then rounded up, and expressed relative to the
/// largest magnitude in the parity output so they do not depend on the
/// fixture's contrast.  Over the five cases the measured
/// parity-versus-RustFFT difference was at most 3.4e-7 of that magnitude with
/// an RMS of 1.1e-7 -- `f32` round-off from a differently ordered transform,
/// nothing structural -- and the bounds here are 4.0e-6 and 1.0e-6, about
/// twelve times the worst case.
#[cfg(feature = "rustfft-backend")]
#[test]
fn newstack_rustfft_fourier_options_match_the_parity_output() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-newstack-rustfft-{}", std::process::id()));
    for (case, nx, ny, options) in [
        ("ftreduce", 100, 30, vec!["-ftreduce", "2"]),
        ("ftexpand", 30, 18, vec!["-ftexpand", "2"]),
        ("ftreduceodd", 54, 45, vec!["-ftreduce", "1.5"]),
        ("phase", 38, 19, vec!["-phase", "-offset", "0.6,-0.35"]),
        ("phaseprime", 100, 7, vec!["-phase", "-offset", "-1.25,0.4"]),
    ] {
        let input = base.with_extension(format!("{case}.input.mrc"));
        unsafe {
            let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
            assert!(!file.is_null());
            let header = (*file).header.cast::<MrcHeader>();
            assert_eq!(mrc_head_new(&mut *header, nx, ny, 1, 2), 0);
            ii_sync_from_mrc_header(file, header);
            assert_eq!(mrc_head_write((*file).fp, header), 0);
            let mut pixels = (0..nx * ny)
                .map(|index| {
                    let (x, y) = ((index % nx) as f32, (index / nx) as f32);
                    100.0 + 30.0 * (x / 3.0).sin() * (y / 2.0).cos() + x - y
                })
                .collect::<Vec<f32>>();
            assert_eq!(
                ii_write_section_float(file, pixels.as_mut_ptr().cast(), 0),
                0
            );
            ii_close(file);
        }
        let mut decoded = Vec::new();
        let mut sizes = Vec::new();
        for backend in ["parity", "rustfft"] {
            let output = base.with_extension(format!("{case}.{backend}.mrc"));
            let result = common::imod_cmd("newstack")
                .env("AUTODOC_DIR", AUTODOC)
                .env("IMOD_RS_FFT_BACKEND", backend)
                .args(["-input", input.to_str().unwrap()])
                .args(["-output", output.to_str().unwrap()])
                .args(&options)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{case} {backend}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            unsafe {
                let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
                let file = ii_open(name.as_ptr(), c"rb".as_ptr());
                assert!(!file.is_null());
                let mut header = std::mem::zeroed::<MrcHeader>();
                assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
                let mut pixels = vec![0.0_f32; (header.nx * header.ny) as usize];
                assert_eq!(
                    ii_read_section_float(file, pixels.as_mut_ptr().cast(), 0),
                    0
                );
                ii_close(file);
                sizes.push((header.nx, header.ny, header.mode));
                decoded.push(pixels);
            }
            let _ = std::fs::remove_file(output);
        }
        let _ = std::fs::remove_file(input);
        assert_eq!(sizes[0], sizes[1], "{case} output geometry");
        let mut maximum = 0.0_f64;
        let mut sum_squares = 0.0_f64;
        let mut scale = 0.0_f64;
        for (parity, rustfft) in decoded[0].iter().zip(decoded[1].iter()) {
            let difference = (*parity as f64 - *rustfft as f64).abs();
            maximum = maximum.max(difference);
            sum_squares += difference * difference;
            scale = scale.max((*parity as f64).abs());
        }
        let rms = (sum_squares / decoded[0].len() as f64).sqrt();
        eprintln!("newstack {case}: max {maximum:.3e} rms {rms:.3e} scale {scale:.3e}");
        assert!(
            maximum < 4.0e-6 * scale,
            "newstack {case} maximum difference {maximum} against scale {scale}"
        );
        assert!(
            rms < 1.0e-6 * scale,
            "newstack {case} RMS difference {rms} against scale {scale}"
        );
    }
}
