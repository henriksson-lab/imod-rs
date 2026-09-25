mod common;

use imod_rs::imod::libiimod::iimage::{
    IIFILE_DEFAULT, MRSA_FLOAT, ii_delete, ii_fill_mrc_header, ii_open, ii_open_new,
    ii_read_section_any, ii_sync_from_mrc_header, ii_write_section_float,
};
use imod_rs::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_new, mrc_head_write};
use std::ffi::CString;

#[test]
fn trimvol_requires_imod_dir_before_parsing_options() {
    let output = common::imod_cmd("trimvol")
        .env_remove("IMOD_DIR")
        .arg("-s")
        .output()
        .expect("run trimvol without IMOD_DIR");
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: trimvol -  IMOD_DIR is not defined!\n"
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn trimvol_removed_s_option_exits_as_the_python_command_does() {
    let output = common::imod_cmd("trimvol")
        .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .arg("-s")
        .output()
        .expect("run trimvol removed option");
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .contains("The -s option has been eliminated; use -sz instead and add -f")
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn trimvol_rejects_mode_with_contrast_on_real_mrc_before_creating_output() {
    let base = std::env::temp_dir().join(format!("imod-rs-trimvol-mode-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let image = &mut *file;
        let mut header = image.mrc_header.take().expect("new MRC image header");
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, 2), 0);
        ii_sync_from_mrc_header(image, &mut header);
        assert_eq!(mrc_head_write(image.fp.as_mut().unwrap(), &mut header), 0);
        image.mrc_header = Some(header);
        let mut section = [1.0_f32, 2., 3., 4.];
        assert_eq!(ii_write_section_float(image, &mut section, 0), 0);
        ii_delete(file);
    }
    let result = common::imod_cmd("trimvol")
        .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args([
            "-mode",
            "1",
            "-c",
            "0,10",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .expect("run trimvol mode/contrast conflict");
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "ERROR: trimvol - You cannot enter -mode with -c, or when running findcontrast\n"
    );
    assert!(result.stderr.is_empty());
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn trimvol_reports_source_specific_coordinate_size_conflicts_before_file_access() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-trimvol-coordinate-size-conflict-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let image = &mut *file;
        let mut header = image.mrc_header.take().expect("new MRC image header");
        assert_eq!(mrc_head_new(&mut header, 1, 1, 1, 2), 0);
        ii_sync_from_mrc_header(image, &mut header);
        assert_eq!(mrc_head_write(image.fp.as_mut().unwrap(), &mut header), 0);
        image.mrc_header = Some(header);
        ii_delete(file);
    }
    for (limits, size, expected) in [
        ("-x", "-nx", "You cannot enter both -x and -nx options"),
        ("-y", "-ny", "You cannot enter both -y and -ny options"),
        ("-z", "-nz", "You cannot enter both -z and -nz options"),
    ] {
        let result = common::imod_cmd("trimvol")
            .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args([
                limits,
                "1,2",
                size,
                "2",
                input.to_str().unwrap(),
                output.to_str().unwrap(),
            ])
            .output()
            .expect("run trimvol coordinate/size conflict");
        assert_eq!(result.status.code(), Some(1));
        assert_eq!(
            String::from_utf8_lossy(&result.stdout),
            format!("ERROR: trimvol - {expected}\n")
        );
        assert!(result.stderr.is_empty());
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn trimvol_checks_missing_input_before_option_conflicts_as_python_does() {
    let result = common::imod_cmd("trimvol")
        .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(["-x", "1,2", "-nx", "2", "missing-input.mrc", "output.mrc"])
        .output()
        .expect("run trimvol missing input with conflict");
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "ERROR: trimvol - Input file missing-input.mrc does not exist\n"
    );
    assert!(result.stderr.is_empty());
}

#[test]
fn trimvol_old_flipped_coordinates_use_source_yz_limit_exchange() {
    let base = std::env::temp_dir().join(format!("imod-rs-trimvol-oldflip-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let image = &mut *file;
        let mut header = image.mrc_header.take().expect("new MRC image header");
        assert_eq!(mrc_head_new(&mut header, 2, 3, 4, 2), 0);
        ii_sync_from_mrc_header(image, &mut header);
        assert_eq!(mrc_head_write(image.fp.as_mut().unwrap(), &mut header), 0);
        image.mrc_header = Some(header);
        for z in 0..4 {
            let mut section = [
                (z * 10) as f32,
                (z * 10 + 1) as f32,
                (z * 10 + 2) as f32,
                (z * 10 + 3) as f32,
                (z * 10 + 4) as f32,
                (z * 10 + 5) as f32,
            ];
            assert_eq!(ii_write_section_float(image, &mut section, z), 0);
        }
        ii_delete(file);
    }
    let result = common::imod_cmd("trimvol")
        .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args([
            "-f",
            "-old",
            "1",
            "-y",
            "2,3",
            "-z",
            "1,2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .expect("run trimvol old flipped coordinates");
    assert!(result.status.success(), "{:?}", result);
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut header = MrcHeader::default();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz), (2, 2, 2));
        let mut section_bytes = [0_u8; 16];
        assert_eq!(
            ii_read_section_any(&mut *file, &mut section_bytes, 0, MRSA_FLOAT),
            0
        );
        let section: [f32; 4] = core::array::from_fn(|index| {
            f32::from_ne_bytes(section_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(section, [10., 11., 12., 13.]);
        assert_eq!(
            ii_read_section_any(&mut *file, &mut section_bytes, 1, MRSA_FLOAT),
            0
        );
        let section: [f32; 4] = core::array::from_fn(|index| {
            f32::from_ne_bytes(section_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(section, [20., 21., 22., 23.]);
        ii_delete(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn trimvol_even_old_flipped_coordinates_reverse_swapped_y_limits() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-trimvol-oldflip-even-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let image = &mut *file;
        let mut header = image.mrc_header.take().expect("new MRC image header");
        assert_eq!(mrc_head_new(&mut header, 2, 3, 4, 2), 0);
        ii_sync_from_mrc_header(image, &mut header);
        assert_eq!(mrc_head_write(image.fp.as_mut().unwrap(), &mut header), 0);
        image.mrc_header = Some(header);
        for z in 0..4 {
            let mut section = [
                (z * 10) as f32,
                (z * 10 + 1) as f32,
                (z * 10 + 2) as f32,
                (z * 10 + 3) as f32,
                (z * 10 + 4) as f32,
                (z * 10 + 5) as f32,
            ];
            assert_eq!(ii_write_section_float(image, &mut section, z), 0);
        }
        ii_delete(file);
    }
    let result = common::imod_cmd("trimvol")
        .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args([
            "-f",
            "-old",
            "2",
            "-y",
            "2,3",
            "-z",
            "1,2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .expect("run trimvol even old flipped coordinates");
    assert!(result.status.success(), "{:?}", result);
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut header = MrcHeader::default();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz), (2, 2, 2));
        let mut section_bytes = [0_u8; 16];
        assert_eq!(
            ii_read_section_any(&mut *file, &mut section_bytes, 0, MRSA_FLOAT),
            0
        );
        let section: [f32; 4] = core::array::from_fn(|index| {
            f32::from_ne_bytes(section_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(section, [12., 13., 14., 15.]);
        assert_eq!(
            ii_read_section_any(&mut *file, &mut section_bytes, 1, MRSA_FLOAT),
            0
        );
        let section: [f32; 4] = core::array::from_fn(|index| {
            f32::from_ne_bytes(section_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(section, [22., 23., 24., 25.]);
        ii_delete(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn trimvol_integer_min_max_maps_observed_real_mrc_range_to_requested_range() {
    let base = std::env::temp_dir().join(format!("imod-rs-trimvol-minmax-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let image = &mut *file;
        let mut header = image.mrc_header.take().expect("new MRC image header");
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, 2), 0);
        // `newstack -sca` scales from the *header* min/max; left at
        // `mrc_head_new`'s sentinels the scale is NaN and native writes
        // -32768 everywhere, so state the true range.
        header.amin = 10.;
        header.amax = 40.;
        header.amean = 25.;
        ii_sync_from_mrc_header(image, &mut header);
        assert_eq!(mrc_head_write(image.fp.as_mut().unwrap(), &mut header), 0);
        image.mrc_header = Some(header);
        let mut section = [10.0_f32, 20., 30., 40.];
        assert_eq!(ii_write_section_float(image, &mut section, 0), 0);
        ii_delete(file);
    }
    let result = common::imod_cmd("trimvol")
        .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args([
            "-mm",
            "100,200",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .expect("run trimvol integer min/max scaling");
    assert!(result.status.success(), "{:?}", result);
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut header = MrcHeader::default();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!(header.mode, 1);
        let mut section_bytes = [0_u8; 16];
        assert_eq!(
            ii_read_section_any(&mut *file, &mut section_bytes, 0, MRSA_FLOAT),
            0
        );
        let section: [f32; 4] = core::array::from_fn(|index| {
            f32::from_ne_bytes(section_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        ii_delete(file);
        for (actual, expected) in section.into_iter().zip([100., 133., 167., 200.]) {
            assert!((actual - expected).abs() <= 1., "{actual} != {expected}");
        }
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn trimvol_crops_real_mrc_volume_with_one_based_coordinates() {
    let base = std::env::temp_dir().join(format!("imod-rs-trimvol-crop-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let image = &mut *file;
        let mut header = image.mrc_header.take().expect("new MRC image header");
        assert_eq!(mrc_head_new(&mut header, 3, 3, 2, 2), 0);
        header.nlabl = 1;
        header.labels[0][..22].copy_from_slice(b"acquisition title line");
        ii_sync_from_mrc_header(image, &mut header);
        assert_eq!(mrc_head_write(image.fp.as_mut().unwrap(), &mut header), 0);
        image.mrc_header = Some(header);
        let mut first = [1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9.];
        let mut second = [11.0_f32, 12., 13., 14., 15., 16., 17., 18., 19.];
        assert_eq!(ii_write_section_float(image, &mut first, 0), 0);
        assert_eq!(ii_write_section_float(image, &mut second, 1), 0);
        ii_delete(file);
    }
    let result = common::imod_cmd("trimvol")
        .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args([
            "-x",
            "2,3",
            "-y",
            "1,2",
            "-z",
            "1,2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{:?}", result);
    assert!(String::from_utf8_lossy(&result.stdout).contains("newstack -siz 2,2 -off 1,0"));
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut header = MrcHeader::default();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz), (2, 2, 2));
        assert_eq!(header.nlabl, 2);
        assert_eq!(&header.labels[0][..22], b"acquisition title line");
        assert_eq!(&header.labels[1][..23], b"NEWSTACK: Images copied");
        let mut pixel_bytes = [0_u8; 16];
        assert_eq!(
            ii_read_section_any(&mut *file, &mut pixel_bytes, 0, MRSA_FLOAT),
            0
        );
        let pixels: [f32; 4] = core::array::from_fn(|index| {
            f32::from_ne_bytes(pixel_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(pixels, [2., 3., 5., 6.]);
        assert_eq!(
            ii_read_section_any(&mut *file, &mut pixel_bytes, 1, MRSA_FLOAT),
            0
        );
        let pixels: [f32; 4] = core::array::from_fn(|index| {
            f32::from_ne_bytes(pixel_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(pixels, [12., 13., 15., 16.]);
        ii_delete(file);
    }
    let bytes = std::fs::read(&output).unwrap();
    let pixels: Vec<f32> = bytes[1024..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(pixels, [2., 3., 5., 6., 12., 13., 15., 16.]);
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn trimvol_flip_yz_preserves_clip_plane_order() {
    let base = std::env::temp_dir().join(format!("imod-rs-trimvol-flip-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let image = &mut *file;
        let mut header = image.mrc_header.take().expect("new MRC image header");
        assert_eq!(mrc_head_new(&mut header, 2, 2, 3, 2), 0);
        ii_sync_from_mrc_header(image, &mut header);
        assert_eq!(mrc_head_write(image.fp.as_mut().unwrap(), &mut header), 0);
        image.mrc_header = Some(header);
        for (z, mut section) in [
            [1.0_f32, 2., 3., 4.],
            [11.0_f32, 12., 13., 14.],
            [21.0_f32, 22., 23., 24.],
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(ii_write_section_float(image, &mut section, z as i32), 0);
        }
        ii_delete(file);
    }
    assert!(
        common::imod_cmd("trimvol")
            .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc")
            )
            .args(["-yz", input.to_str().unwrap(), output.to_str().unwrap()])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut header = MrcHeader::default();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz), (2, 3, 2));
        let mut pixel_bytes = [0_u8; 24];
        assert_eq!(
            ii_read_section_any(&mut *file, &mut pixel_bytes, 0, MRSA_FLOAT),
            0
        );
        let pixels: [f32; 6] = core::array::from_fn(|index| {
            f32::from_ne_bytes(pixel_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(pixels, [1., 2., 11., 12., 21., 22.]);
        assert_eq!(
            ii_read_section_any(&mut *file, &mut pixel_bytes, 1, MRSA_FLOAT),
            0
        );
        let pixels: [f32; 6] = core::array::from_fn(|index| {
            f32::from_ne_bytes(pixel_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(pixels, [3., 4., 13., 14., 23., 24.]);
        ii_delete(file);
    }
    let bytes = std::fs::read(&output).unwrap();
    let pixels: Vec<f32> = bytes[1024..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        pixels,
        [1., 2., 11., 12., 21., 22., 3., 4., 13., 14., 23., 24.]
    );
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn trimvol_rotate_x_uses_source_clip_rotx_minus_ninety_plane_order() {
    let base = std::env::temp_dir().join(format!("imod-rs-trimvol-rotx-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let image = &mut *file;
        let mut header = image.mrc_header.take().expect("new MRC image header");
        assert_eq!(mrc_head_new(&mut header, 4, 4, 2, 1), 0);
        header.xlen = 8.0;
        header.ylen = 12.0;
        header.zlen = 20.0;
        header.xorg = 1.0;
        header.yorg = 3.0;
        header.zorg = -5.0;
        header.tiltangles = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        ii_sync_from_mrc_header(image, &mut header);
        assert_eq!(mrc_head_write(image.fp.as_mut().unwrap(), &mut header), 0);
        image.mrc_header = Some(header);
        let mut first = [
            1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ];
        let mut second = [
            17.0_f32, 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
        ];
        assert_eq!(ii_write_section_float(image, &mut first, 0), 0);
        assert_eq!(ii_write_section_float(image, &mut second, 1), 0);
        ii_delete(file);
    }
    assert!(
        common::imod_cmd("trimvol")
            .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc")
            )
            .args(["-rx", input.to_str().unwrap(), output.to_str().unwrap()])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut header = MrcHeader::default();
        assert_eq!(ii_fill_mrc_header(file, &mut header), 0);
        assert_eq!((header.nx, header.ny, header.nz), (4, 2, 4));
        assert_eq!((header.xorg, header.yorg, header.zorg), (1.0, -5.0, 9.0));
        assert_eq!(header.tiltangles, [4.0, 5.0, 6.0, -86.0, 5.0, 6.0]);
        let mut pixel_bytes = [0_u8; 32];
        assert_eq!(
            ii_read_section_any(&mut *file, &mut pixel_bytes, 0, MRSA_FLOAT),
            0
        );
        let pixels: [f32; 8] = core::array::from_fn(|index| {
            f32::from_ne_bytes(pixel_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(pixels, [13., 14., 15., 16., 29., 30., 31., 32.]);
        assert_eq!(
            ii_read_section_any(&mut *file, &mut pixel_bytes, 3, MRSA_FLOAT),
            0
        );
        let pixels: [f32; 8] = core::array::from_fn(|index| {
            f32::from_ne_bytes(pixel_bytes[index * 4..index * 4 + 4].try_into().unwrap())
        });
        assert_eq!(pixels, [1., 2., 3., 4., 17., 18., 19., 20.]);
        ii_delete(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

/// Native-golden coverage: every case in `fixtures/trimvol/cases.tsv` was run
/// through the native Python `IMOD/pysrc/trimvol` driving the native
/// `densmatch`, `findcontrast`, `newstack`, `clip` and `header` by
/// `fixtures/make-trimvol-goldens.sh`, stdout captured through a pipe.  Output
/// files are compared with their MRC labels (date/time stamps) blanked.
/// Stdout is compared as a multiset of lines, with the `.tmp.<pid>` name of
/// the flip/rotate intermediate normalised: native Python block-buffers its
/// own `print`s into the pipe, so the called programs' output lands ahead of
/// the script's lines there, and the interleaving is not an acceptance
/// criterion.
#[test]
fn trimvol_cases_match_native_golden() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("fixtures/trimvol");
    let inputs_dir = root.join("fixtures/densmatch");
    let golden = fixtures.join("golden");
    let table = std::fs::read_to_string(fixtures.join("cases.tsv")).unwrap();
    let lines_of = |bytes: &[u8]| -> Vec<String> {
        let text = String::from_utf8_lossy(bytes);
        let mut lines: Vec<String> = text
            .lines()
            .map(|line| {
                let line = line.trim();
                match line.find(".tmp.") {
                    Some(at) => {
                        let rest = &line[at + 5..];
                        let digits = rest.chars().take_while(char::is_ascii_digit).count();
                        format!("{}.tmp.PID{}", &line[..at], &rest[digits..])
                    }
                    None => line.to_string(),
                }
            })
            .filter(|line| !line.is_empty())
            .collect();
        lines.sort();
        lines
    };
    let mask = |bytes: &[u8]| -> Vec<u8> {
        let mut masked = bytes.to_vec();
        if masked.len() > 1024 {
            masked[224..1024].fill(0);
        }
        masked
    };
    let mut failures = Vec::new();
    let mut count = 0;
    for line in table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let (name, args) = line.split_once('\t').unwrap();
        let rc: i32 = std::fs::read_to_string(golden.join(format!("{name}.rc")))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let expected_out = std::fs::read(golden.join(format!("{name}.out"))).unwrap();
        let dir = std::env::temp_dir().join(format!(
            "imod-rs-trimvol-golden-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut inputs = Vec::new();
        for entry in std::fs::read_dir(&inputs_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "mrc") {
                let file = path.file_name().unwrap().to_owned();
                std::fs::copy(&path, dir.join(&file)).unwrap();
                inputs.push(file.to_string_lossy().into_owned());
            }
        }
        let output = common::imod_cmd("trimvol")
            .current_dir(&dir)
            .env("IMOD_DIR", env!("CARGO_MANIFEST_DIR"))
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .env("OMP_NUM_THREADS", "1")
            .args(args.split_whitespace())
            .output()
            .unwrap();
        count += 1;
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}",
                output.status.code()
            ));
        }
        if lines_of(&output.stdout) != lines_of(&expected_out) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                String::from_utf8_lossy(&expected_out),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let mut expected_files: Vec<String> = std::fs::read_dir(golden.join(name))
            .map(|entries| {
                entries
                    .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
                    .collect()
            })
            .unwrap_or_default();
        expected_files.sort();
        let mut produced: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .filter(|f| !inputs.contains(f))
            .collect();
        produced.sort();
        if produced != expected_files {
            failures.push(format!(
                "{name}: files {produced:?}, native {expected_files:?}"
            ));
        }
        for file in &expected_files {
            let Ok(ours) = std::fs::read(dir.join(file)) else {
                continue;
            };
            let theirs = std::fs::read(golden.join(name).join(file)).unwrap();
            if mask(&ours) != mask(&theirs) {
                failures.push(format!("{name}: {file} differs from native"));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count > 30, "only {count} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
