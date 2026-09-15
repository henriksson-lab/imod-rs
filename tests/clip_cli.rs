mod common;

use imod_rs::imod::clip::clip::{
    CameraDefects, ClipOptions, IP_APPEND_ADD, IP_APPEND_FALSE, IP_APPEND_OVERWRITE,
    IP_APPEND_TRUNCATE, IP_DEFAULT,
};
use imod_rs::imod::clip::file_io::{grap_volume_free, grap_volume_read};
use imod_rs::imod::libcfshr::islice::slice_get_pixel_magnitude;
use imod_rs::imod::libiimod::iimage::{
    IIFILE_DEFAULT, IIFILE_TIFF, ii_close, ii_open, ii_open_new, ii_read_section_float,
    ii_sync_from_mrc_header, ii_write_section_float,
};
use imod_rs::imod::libiimod::iitif::tiff_get_field;
use imod_rs::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MRC_MODE_RGB,
    MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader, mrc_head_label, mrc_head_new, mrc_head_read,
    mrc_head_write,
};
use std::ffi::CString;
use std::process::Command;

#[test]
fn planefit_sums_each_real_mrc_input_before_fitting() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-planefit-{}", std::process::id()));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("slopes.txt");
    unsafe {
        for path in [&first, &second] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 11, 11, 1, 2), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            let mut pixels = [0_f32; 121];
            for y in 0..11 {
                for x in 0..11 {
                    pixels[x + 11 * y] =
                        100. * (1. + 0.01 * (x as f32 - 5.) + 0.02 * (y as f32 - 5.));
                }
            }
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let command = common::imod_cmd("clip")
        .args([
            "pl",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(stdout.starts_with("clip: summing slices...\n"), "{stdout}");
    assert!(
        stdout.contains(
            "Plane slopes imply a gradient over full extent in X and Y of 11.000 and 22.000\n"
        ),
        "{stdout}"
    );
    assert!(
        stdout.contains("Root-mean-squared residual = 0.000\n"),
        "{stdout}"
    );
    let slopes: Vec<f32> = std::fs::read_to_string(&output)
        .unwrap()
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(slopes.len(), 2);
    assert!((slopes[0] - 0.01).abs() < 1.0e-6, "{slopes:?}");
    assert!((slopes[1] - 0.02).abs() < 1.0e-6, "{slopes:?}");
    for path in [first, second, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn flatfield_order_one_writes_polynomial_inverse_for_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-flatfield-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 15, 15, 1, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 225];
        for y in 0..15 {
            for x in 0..15 {
                pixels[x + 15 * y] = 100. * (1. + 0.01 * (x as f32 - 7.));
            }
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let command = common::imod_cmd("clip")
        .args([
            "fla",
            "-n",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(stdout.starts_with("clip: summing slices...\n"), "{stdout}");
    assert!(stdout.contains("Constant term -0.000000\n"), "{stdout}");
    assert!(
        stdout.contains(
            "X & Y order and coefficients (times half-size in X/Y to respective powers):\n"
        ),
        "{stdout}"
    );
    assert!(
        stdout.contains("Root-mean-squared residual = 0.000\n"),
        "{stdout}"
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert!(
            std::str::from_utf8(&header.labels[0])
                .unwrap()
                .starts_with("clip: Flatfield based on order 1 fit to image sum")
        );
        let mut pixels = [0_f32; 225];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert!(
            (pixels[7 + 15 * 7] - 1.).abs() < 1.0e-5,
            "{:?}",
            &pixels[105..120]
        );
        assert!((pixels[0] - 1. / 0.93).abs() < 2.0e-4, "{}", pixels[0]);
        assert!((pixels[14] - 1. / 1.07).abs() < 2.0e-4, "{}", pixels[14]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn planefit_reports_source_dimension_mismatch_for_real_mrc_inputs() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-planefit-error-{}",
        std::process::id()
    ));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("slopes.txt");
    unsafe {
        for (path, nx) in [(&first, 11), (&second, 10)] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, nx, 11, 1, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            let mut pixels = vec![1_f32; (nx * 11) as usize];
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let command = common::imod_cmd("clip")
        .args([
            "planefit",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!command.status.success());
    assert!(
        String::from_utf8(command.stdout)
            .unwrap()
            // Confirmed against native `clip planefit`: `processing.cpp:2593`
            // uses exitError, so the setExitPrefix banner precedes the newline
            // the message itself starts with.
            .contains("ERROR: clip -  \nDoing plane fit: files must be same size in X and Y;")
    );
    for path in [first, second, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn stat_reports_source_table_for_real_multisection_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-stat-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 3, 2, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut first = [1_f32, 2., 3., 4., 9., 6., 7., 8., 5.];
        let mut second = [10_f32, 11., 12., 13., 20., 15., 16., 17., 14.];
        assert_eq!(ii_write_section_float(&mut *file, &mut first, 0), 0);
        assert_eq!(ii_write_section_float(&mut *file, &mut second, 1), 0);
        ii_close(file);
    }
    let command = common::imod_cmd("clip")
        .args(["stats", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(
        stdout.starts_with(
            "slice|   min   |(   x,   y)|    max  |(      x,      y)|   mean    |  std dev.\n"
        ),
        "{stdout}"
    );
    assert!(
        stdout.contains("   0     1.0000 (   0,   0)    9.0000 (  -0.14,  -0.79)    5.0000"),
        "{stdout}"
    );
    assert!(
        stdout.contains("   1    10.0000 (   0,   0)   20.0000 (  -1.00,  -1.00)   14.2222"),
        "{stdout}"
    );
    assert!(
        stdout.contains(" all     1.0000 (@ z=    0)   20.0000 (@ z=    1"),
        "{stdout}"
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn stat_uses_real_piece_list_coordinates_and_overlap_path() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-stat-piece-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let pieces = base.with_extension("pl");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 3, 2, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut one = [1_f32, 2., 3., 4., 9., 6., 7., 8., 5.];
        let mut two = [10_f32, 11., 12., 13., 20., 15., 16., 17., 14.];
        assert_eq!(ii_write_section_float(&mut *file, &mut one, 0), 0);
        assert_eq!(ii_write_section_float(&mut *file, &mut two, 1), 0);
        ii_close(file);
    }
    std::fs::write(&pieces, "0 0 4\n3 0 5\n").unwrap();
    let command = common::imod_cmd("clip")
        .args([
            "stats",
            "-Pieces",
            pieces.to_str().unwrap(),
            "-Overlap",
            "1;1",
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(
        stdout.starts_with("piece|   min   |(   x,   y,   z)|"),
        "{stdout}"
    );
    assert!(
        stdout.contains("   0     1.0000 (   0,   0,   0)    9.0000 (   0,  -1,   0)    5.0000"),
        "{stdout}"
    );
    assert!(
        stdout.contains("   1    10.0000 (   2,   0,   1)   20.0000 (   1,  -1,   1)   14.2222"),
        "{stdout}"
    );
    assert!(
        stdout.contains(" all     1.0000 (@ piece =    1)   20.0000 (@ piece =    2)"),
        "{stdout}"
    );
    for path in [input, pieces] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn stat_marks_real_mrc_extreme_sections_with_mad_windows() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-stat-outlier-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 5, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for (section, value) in [1_f32, 2., 3., 100., 5.].into_iter().enumerate() {
            let mut pixels = [value; 4];
            assert_eq!(
                ii_write_section_float(&mut *file, &mut pixels, section as i32),
                0
            );
        }
        ii_close(file);
    }
    let command = common::imod_cmd("clip")
        .args(["stats", "-n", "2.24", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(stdout.contains(" 100.0000*"), "{stdout}");
    assert!(
        stdout.contains("Slices with extreme values:   3"),
        "{stdout}"
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn stat_marks_piece_coordinates_in_real_mrc_mad_outlier_rows() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-stat-piece-outlier-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let pieces = base.with_extension("pl");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 5, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for (section, value) in [1_f32, 2., 3., 100., 5.].into_iter().enumerate() {
            let mut pixels = [value; 4];
            assert_eq!(
                ii_write_section_float(&mut *file, &mut pixels, section as i32),
                0
            );
        }
        ii_close(file);
    }
    std::fs::write(&pieces, "0 0 0\n0 0 1\n0 0 2\n0 0 3\n0 0 4\n").unwrap();
    let command = common::imod_cmd("clip")
        .args([
            "stats",
            "-n",
            "2.24",
            "-P",
            pieces.to_str().unwrap(),
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(
        stdout.starts_with("piece|   min   |(   x,   y,   z)|"),
        "{stdout}"
    );
    assert!(
        stdout.contains(" 100.0000 (   0,   0,   3)  100.0000*(   1,   1,   3)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Pieces with extreme values:   3"),
        "{stdout}"
    );
    for path in [input, pieces] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn clip_accepts_source_two_letter_histogram_prefix_for_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-hi-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, MRC_MODE_BYTE), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 2., 3., 4.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["hi", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(result.status.success(), "{:?}", result);
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .starts_with(" Value   counts")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn resize_accepts_source_semicolon_coordinate_pairs_for_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-resize-pairs-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 2, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 2., 3., 4., 5., 6.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    assert!(
        common::imod_cmd("clip")
            .args([
                "res",
                "-xrange",
                "1;2",
                "-yrange",
                "0;1",
                input.to_str().unwrap(),
                output.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!((header.nx, header.ny), (2, 2));
        let mut pixels = [0_f32; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        // C `-x 1;2 -y 0;1` selects inclusive coordinates: x=1..2 in
        // each of rows y=0 and y=1.
        assert_eq!(pixels, [2., 3., 5., 6.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_real_mrc_normalizes_reversed_source_x_range() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-reversed-x-range-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 2., 3.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-x",
            "2;1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny), (1, 1));
        let mut pixels = [f32::NAN];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [3.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn resize_keeps_a_nondefault_negative_fractional_center_for_real_mrc_coordinates() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-negative-fractional-center-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 2, 1, MRC_MODE_FLOAT), 0);
        header.xorg = 10.;
        header.xlen = 6.;
        header.mx = 3;
        header.nxstart = 7;
        header.zorg = 30.;
        header.zlen = 4.;
        header.mz = 1;
        header.nzstart = 9;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 2., 3., 4., 5., 6.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "res",
            "-cx",
            "-1.5",
            "-cz",
            "-1.5",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        // C compares centers to the floating IP_DEFAULT sentinel.  Thus -1.5
        // is retained, then `set_mrc_coords` truncates it to -1 for X and
        // floors it after the half-Z adjustment: 10 - ((-1 - 3 / 2) * 2) =
        // 14 and 30 - floor(-1.5 - 1 / 2) * 4 = 38.
        assert_eq!(
            (header.xorg, header.nxstart, header.zorg, header.nzstart),
            (14., 7, 38., 9)
        );
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn clip_reports_source_error_for_unreadable_mrc_header() {
    let input =
        std::env::temp_dir().join(format!("imod-rs-clip-bad-header-{}", std::process::id()));
    std::fs::write(&input, [0_u8, 1, 2, 3]).unwrap();
    let result = common::imod_cmd("clip")
        .args(["inf", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stdout).contains("ERROR: clip -  Error"));
    assert!(String::from_utf8_lossy(&result.stderr).contains("mrc_head_read"));
    let _ = std::fs::remove_file(input);
}

#[test]
fn two_input_process_reports_source_error_when_second_real_mrc_is_missing() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-second-missing-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let missing = base.with_extension("missing.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "add",
            input.to_str().unwrap(),
            missing.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        format!("ERROR: clip -  Error opening {0}\n", missing.display())
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn info_reports_source_open_error_for_corrupt_second_mrc() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-info-second-corrupt-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let corrupt = base.with_extension("corrupt.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    std::fs::write(&corrupt, [0_u8, 1, 2, 3]).unwrap();
    let result = common::imod_cmd("clip")
        .args(["info", input.to_str().unwrap(), corrupt.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert_eq!(
        stdout,
        format!("ERROR: clip -  Error opening {0}\n", corrupt.display())
    );
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(corrupt);
}

#[test]
fn info_real_mrc_prints_source_idtype_metadata_fields() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-file-io-info-idtype-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        header.idtype = 42;
        header.ispg = 7;
        header.next = 8;
        header.nd1 = 3;
        header.nd2 = 4;
        header.vd1 = 150;
        header.vd2 = -50;
        header.tiltangles = [1., 2., 3., 4., 5., 6.];
        header.alpha = 91.;
        header.beta = 92.;
        header.gamma = 93.;
        header.mapc = 3;
        header.mapr = 1;
        header.maps = 2;
        header.mx = 2;
        header.my = 4;
        header.mz = 5;
        header.xlen = 3.;
        header.ylen = 8.;
        header.zlen = 15.;
        header.nxstart = 7;
        header.nystart = 8;
        header.nzstart = 9;
        header.xorg = 1.25;
        header.yorg = -2.5;
        header.zorg = 3.75;
        header.nlabl = 1;
        core::ptr::copy_nonoverlapping(
            c"source label".as_ptr(),
            header.labels[0].as_mut_ptr().cast(),
            c"source label".to_bytes_with_nul().len(),
        );
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["info", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("idtype =\t42\n"), "{stdout}");
    assert!(stdout.contains("ispg =\t\t7\n"), "{stdout}");
    assert!(stdout.contains("extra header = \t8\n"), "{stdout}");
    assert!(stdout.contains("nd1 =\t\t3\n"), "{stdout}");
    assert!(stdout.contains("nd2 =\t\t4\n"), "{stdout}");
    assert!(stdout.contains("vd1 =\t\t1.5\n"), "{stdout}");
    assert!(stdout.contains("vd2 =\t\t-0.5\n"), "{stdout}");
    assert!(
        stdout.contains("angles = ( 1, 2, 3, 4, 5, 6)\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Cell Rotation =  ( 91, 92, 93).\n"),
        "{stdout}"
    );
    assert!(stdout.contains("Columns are   = axis 3\n"), "{stdout}");
    assert!(stdout.contains("Rows are      = axis 1\n"), "{stdout}");
    assert!(stdout.contains("Sections are  = axis 2\n"), "{stdout}");
    assert!(
        stdout.contains("Read length   =  ( 2, 4, 5).\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Scale         =  ( 1.5 x 2 x 3 ) Angstrom.\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Start reading image at ( 7, 8, 9).\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("orgin  = ( 1.25, -2.5, 3.75)\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Thare are 1 labels.\n\nsource label"),
        "{stdout}"
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn info_real_byte_mrc_uses_source_byte_mode_report() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-file-io-info-byte-mode-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_BYTE), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [17_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["info", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .starts_with("MRC header info:\nmode = Byte\n")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn info_real_rgb_mrc_uses_source_rgb_mode_report() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-file-io-info-rgb-mode-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_RGB), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["info", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .starts_with("MRC header info:\nmode = rgb byte\n")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn info_real_complex_float_mrc_uses_source_complex_mode_report() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-file-io-info-complex-mode-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_COMPLEX_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["info", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .starts_with("MRC header info:\nmode = Complex Float\n")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn info_real_unsigned_short_mrc_uses_source_ushort_mode_report() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-file-io-info-ushort-mode-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_USHORT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["info", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .starts_with("MRC header info:\nmode = Unsigned Short\n")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn info_real_complex_short_mrc_uses_source_complex_short_mode_report() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-file-io-info-complex-short-mode-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_COMPLEX_SHORT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["info", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .starts_with("MRC header info:\nmode = Complex Short\n")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn info_real_short_mrc_uses_source_short_mode_report() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-file-io-info-short-mode-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_SHORT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["info", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .starts_with("MRC header info:\nmode = Short\n")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn clip_append_reports_source_missing_output_error_for_real_mrc_input() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-append-missing-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("missing.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "bri",
            "-a",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .contains("ERROR: clip -  Error finding")
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn clip_append_reports_source_error_for_unreadable_existing_output_header() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-append-bad-header-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    std::fs::write(&output, [0_u8, 1, 2, 3]).unwrap();
    let result = common::imod_cmd("clip")
        .args([
            "bri",
            "-a",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stdout).contains("ERROR: clip -  Error"),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stderr).contains("mrc_head_read"));
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn threshold_process_error_exits_before_clip_main_finalization_for_real_mrc() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-threshold-finalization-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let backup = std::path::PathBuf::from(format!("{}~", output.to_string_lossy()));
    unsafe {
        for (path, value) in [(&input, 3_f32), (&output, 97_f32)] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            let mut pixel = [value];
            assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "threshold",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(255));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "Threshold...\nERROR: clip threshold: You must enter a threshold value\n"
    );
    assert!(output.exists());
    unsafe {
        let name = CString::new(backup.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let mut pixel = [f32::NAN];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixel, 0), 0);
        assert_eq!(pixel, [97.]);
        ii_close(file);
    }
    for path in [input, output, backup] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn fft_with_tiff_output_environment_warns_and_writes_source_forced_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-fft-output-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 4, 4, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 16];
        for (index, value) in pixels.iter_mut().enumerate() {
            *value = index as f32;
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .env("IMOD_OUTPUT_FORMAT", "TIFF")
        .args(["fft", input.to_str().unwrap(), output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8_lossy(&result.stdout).contains(
            "WARNING: clip - Writing an MRC file; TIFF or JPEG files cannot contain FFTs\n"
        )
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(header.mode, MRC_MODE_COMPLEX_FLOAT);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(unix)]
#[test]
fn view_option_runs_source_3dmod_command_after_real_mrc_output_close() {
    use std::os::unix::fs::PermissionsExt;

    let base = std::env::temp_dir().join(format!("imod-rs-clip-view-{}", std::process::id()));
    let bin = base.with_extension("bin");
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let marker = base.with_extension("view-argument");
    std::fs::create_dir(&bin).unwrap();
    let viewer = bin.join("3dmod");
    std::fs::write(
        &viewer,
        "#!/bin/sh\nprintf '%s\\n' \"$1\" > \"$CLIP_VIEW_MARKER\"\n",
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&viewer).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&viewer, permissions).unwrap();
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [3_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .env("PATH", &bin)
        .env("CLIP_VIEW_MARKER", &marker)
        .args([
            "resize",
            "-viewer",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    assert_eq!(
        std::fs::read_to_string(&marker).unwrap(),
        format!("{}\n", output.display())
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        ii_close(file);
    }
    for path in [input, output, marker] {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_dir_all(bin);
}

#[test]
fn nonappend_real_mrc_output_is_renamed_to_source_backup_before_replacement() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-output-backup-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let backup = std::path::PathBuf::from(format!("{}~", output.to_string_lossy()));
    unsafe {
        for (path, value) in [(&input, 3_f32), (&output, 97_f32)] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            let mut pixel = [value];
            assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args(["resize", input.to_str().unwrap(), output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        for (path, expected) in [(&output, 3_f32), (&backup, 97_f32)] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open(name.to_bytes(), "rb");
            assert!(!file.is_null(), "{}", path.display());
            let mut pixel = [f32::NAN];
            assert_eq!(ii_read_section_float(&mut *file, &mut pixel, 0), 0);
            assert_eq!(pixel, [expected], "{}", path.display());
            ii_close(file);
        }
    }
    for path in [input, output, backup] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn average_2d_means_all_real_input_sections() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-average-2d-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 2, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut one = [1_f32, 3., 5., 7.];
        let mut two = [11_f32, 13., 15., 17.];
        assert_eq!(ii_write_section_float(&mut *file, &mut one, 0), 0);
        assert_eq!(ii_write_section_float(&mut *file, &mut two, 1), 0);
        ii_close(file);
    }
    assert!(
        common::imod_cmd("clip")
            .args([
                "average",
                "-2d",
                input.to_str().unwrap(),
                output.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let mut pixels = [0_f32; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [6., 8., 10., 12.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn standev_2d_thresholds_real_mrc_pixels_before_source_variance() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-standev2d-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 2, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for (section, mut pixels) in [[1_f32, 10.], [7., 30.]].into_iter().enumerate() {
            assert_eq!(
                ii_write_section_float(&mut *file, &mut pixels, section as i32),
                0
            );
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "standev",
            "-2d",
            "-n",
            "5",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "2D Averaging...\n"
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let mut pixels = [0_f32; 2];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels[0], 0.);
        assert!((pixels[1] - 200_f32.sqrt()).abs() < 1.0e-5, "{:?}", pixels);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn append_real_mrc_keeps_existing_mode_with_source_file_io_warning() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-append-mode-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, mode, mut pixels) in [
            (&input, MRC_MODE_BYTE, [7_f32, 9.]),
            (&output, MRC_MODE_SHORT, [2_f32, 4.]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 1, 1, mode), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "flatfield",
            "-a",
            "-m",
            "float",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "WARNING: clip - Appended file can't change mode.\nclip: summing slices...\nAveraged image min = 7, max = 9, mean = 8\n"
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!((header.nz, header.mode), (2, MRC_MODE_SHORT));
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_real_mrc_append_updates_source_weighted_header_mean() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-append-mean-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, mean, mut pixels) in [
            (&input, 8_f32, vec![7_f32, 9.]),
            (&output, 4_f32, vec![2_f32, 6.]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
            header.amean = mean;
            // `mrcFillLabelString` must replace all 80 bytes of a reused
            // output label, including its source `strftime` NUL terminator.
            // This makes the append path detect a stale final byte.
            header.labels[0][79] = b'/';
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-a",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), "Brightness...\n");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        // `set_output_options` first scales 4 by 1/2, then clipWriteSlice
        // adds the appended slice mean 8 divided by the new nz=2.
        assert_eq!((header.nz, header.amean), (2, 6.));
        ii_close(file);
    }
    // The in-memory header reader normalizes embedded NUL title padding to
    // spaces.  Check the serialized MRC label, where C `strftime` supplies
    // the terminator at byte 80.
    assert_eq!(std::fs::read(&output).unwrap()[224 + 79], 0);
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_real_mrc_preserves_source_input_header_label() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-header-label-copy-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        header.nlabl = 1;
        core::ptr::copy_nonoverlapping(
            c"input acquisition label".as_ptr(),
            header.labels[0].as_mut_ptr().cast(),
            c"input acquisition label".to_bytes_with_nul().len(),
        );
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [7_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(header.nlabl, 2);
        let input_label = String::from_utf8_lossy(&header.labels[0]);
        let operation_label = String::from_utf8_lossy(&header.labels[1]);
        assert!(
            input_label.starts_with("input acquisition label"),
            "{input_label:?}"
        );
        assert!(
            operation_label.starts_with("clip: brightness"),
            "{operation_label:?}"
        );
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn append_real_mrc_keeps_existing_size_and_resizes_the_written_slice() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-append-size-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, nx, mut pixels) in [
            (&input, 3, vec![1_f32, 2., 3.]),
            (&output, 2, vec![10_f32, 20.]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, nx, 1, 1, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-a",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "WARNING: clip - Appended file can't change size.\nBrightness...\n"
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz), (2, 1, 2));
        let mut pixels = [0_f32; 2];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 1), 0);
        assert_eq!(pixels, [1., 2.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_real_mrc_truncate_replaces_at_source_section_and_updates_header() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-truncate-output-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, nz, sections) in [
            (&input, 1, vec![vec![7_f32, 9.]]),
            (
                &output,
                3,
                vec![vec![1_f32, 2.], vec![3., 4.], vec![5., 6.]],
            ),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 1, nz, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            for (section, mut pixels) in sections.into_iter().enumerate() {
                assert_eq!(
                    ii_write_section_float(&mut *file, &mut pixels, section as i32),
                    0
                );
            }
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-o",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), "Brightness...\n");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz), (2, 1, 2));
        for (section, expected) in [(0, [1_f32, 2.]), (1, [7., 9.])] {
            let mut pixels = [f32::NAN; 2];
            assert_eq!(ii_read_section_float(&mut *file, &mut pixels, section), 0);
            assert_eq!(pixels, expected);
        }
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_real_mrc_overwrite_retains_tail_at_source_section() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-overwrite-output-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, nz, sections) in [
            (&input, 1, vec![vec![7_f32, 9.]]),
            (
                &output,
                3,
                vec![vec![1_f32, 2.], vec![3., 4.], vec![5., 6.]],
            ),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 1, nz, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            for (section, mut pixels) in sections.into_iter().enumerate() {
                assert_eq!(
                    ii_write_section_float(&mut *file, &mut pixels, section as i32),
                    0
                );
            }
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-or",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), "Brightness...\n");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz), (2, 1, 3));
        for (section, expected) in [(0, [1_f32, 2.]), (1, [7., 9.]), (2, [5., 6.])] {
            let mut pixels = [f32::NAN; 2];
            assert_eq!(ii_read_section_float(&mut *file, &mut pixels, section), 0);
            assert_eq!(pixels, expected);
        }
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_expands_a_real_mrc_with_source_blank_slices_before_and_after() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-blank-output-lifecycle-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        header.amean = 9.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [3_f32, 5.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-oz",
            "3",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), "Brightness...\n");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz), (2, 1, 3));
        // clipWriteSlice accumulates min/max from the real slice only, and
        // includes the two source pad slices in the running output mean.
        assert_eq!((header.amin, header.amax), (3., 5.));
        assert_eq!(header.amean, 22. / 3.);
        for (section, expected) in [(0, [9_f32, 9.]), (1, [3., 5.]), (2, [9., 9.])] {
            let mut pixels = [f32::NAN; 2];
            assert_eq!(ii_read_section_float(&mut *file, &mut pixels, section), 0);
            assert_eq!(pixels, expected);
        }
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_2d_real_mrc_keeps_only_source_centered_output_window() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-centered-output-window-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 3, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for (section, mut pixels) in [[1_f32], [4.], [9.]].into_iter().enumerate() {
            assert_eq!(
                ii_write_section_float(&mut *file, &mut pixels, section as i32),
                0
            );
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-2d",
            "-oz",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(header.nz, 1);
        let mut pixels = [f32::NAN];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [4.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_real_mrc_explicit_pad_overrides_source_default_for_blank_sections() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-explicit-blank-pad-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        header.amean = 9.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [3_f32, 5.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-oz",
            "3",
            "-p",
            "2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(header.amean, 8. / 3.);
        for (section, expected) in [(0, [2_f32, 2.]), (2, [2., 2.])] {
            let mut pixels = [f32::NAN; 2];
            assert_eq!(ii_read_section_float(&mut *file, &mut pixels, section), 0);
            assert_eq!(pixels, expected);
        }
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_3d_real_mrc_places_source_boundary_pad_slices() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-boundary-pad-placement-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 2, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for (section, mut pixels) in [[1_f32], [2.]].into_iter().enumerate() {
            assert_eq!(
                ii_write_section_float(&mut *file, &mut pixels, section as i32),
                0
            );
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-oz",
            "4",
            "-cz",
            "1",
            "-p",
            "9",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(header.nz, 4);
        for (section, expected) in [(0, [9_f32]), (1, [1.]), (2, [2.]), (3, [9.])] {
            let mut pixels = [f32::NAN];
            assert_eq!(ii_read_section_float(&mut *file, &mut pixels, section), 0);
            assert_eq!(pixels, expected);
        }
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_3d_real_mrc_trims_input_to_source_center_section() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-3d-centered-input-trim-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 3, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for (section, mut pixels) in [[1_f32], [4.], [9.]].into_iter().enumerate() {
            assert_eq!(
                ii_write_section_float(&mut *file, &mut pixels, section as i32),
                0
            );
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-oz",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(header.nz, 1);
        let mut pixels = [f32::NAN];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [4.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_3d_iz_sets_source_input_z_range_before_section_list_rebuild() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-iz-input-range-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 4, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for (section, mut pixels) in [[1_f32], [2.], [3.], [4.]].into_iter().enumerate() {
            assert_eq!(
                ii_write_section_float(&mut *file, &mut pixels, section as i32),
                0
            );
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-1",
            "-iz",
            "2,3",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(header.nz, 2);
        let mut pixels = [f32::NAN];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [2.]);
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 1), 0);
        assert_eq!(pixels, [3.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_real_mrc_allows_requested_output_resize() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-fixed-output-size-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [3_f32, 5.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-ox",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), "Brightness...\n");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny), (1, 1));
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_real_mrc_converts_new_output_to_requested_source_mode() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-new-output-mode-conversion-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1.6_f32, -2.4];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-m",
            "short",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(header.mode, MRC_MODE_SHORT);
        let mut pixels = [f32::NAN; 2];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [1., -2.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn flatfield_real_mrc_keeps_source_fixed_output_size_with_warning() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-flatfield-fixed-output-size-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 3, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 2., 3., 4., 5., 6., 7., 8., 9.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "flatfield",
            "-ox",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .contains("WARNING: clip - Process can't change output size.\n")
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny), (3, 3));
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn grap_volume_read_pads_and_crops_real_mrc_with_source_coordinates() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-file-io-volume-read-{}.mrc",
        std::process::id()
    ));
    let output = input.with_extension("output.mrc");
    let mode_mismatch = input.with_extension("mode-mismatch.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 2., 3., 4.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        let mut options = ClipOptions {
            pname: String::new(),
            command: String::new(),
            process: 0,
            x: IP_DEFAULT,
            y: IP_DEFAULT,
            z: IP_DEFAULT,
            x2: IP_DEFAULT,
            y2: IP_DEFAULT,
            z2: IP_DEFAULT,
            ix: 3,
            iy: 3,
            iz: 3,
            iz2: IP_DEFAULT,
            ox: IP_DEFAULT,
            oy: IP_DEFAULT,
            oz: IP_DEFAULT,
            chunk_x: IP_DEFAULT,
            chunk_y: IP_DEFAULT,
            chunk_z: IP_DEFAULT,
            cx: 1.,
            cy: 1.,
            cz: 1.,
            high: IP_DEFAULT as f32,
            low: IP_DEFAULT as f32,
            red: IP_DEFAULT as f32,
            green: IP_DEFAULT as f32,
            blue: IP_DEFAULT as f32,
            thresh: IP_DEFAULT as f32,
            weight: IP_DEFAULT as f32,
            pctl_frac: IP_DEFAULT as f32,
            falloff_frac: IP_DEFAULT as f32,
            min_size: IP_DEFAULT,
            pad: 9.,
            mode: IP_DEFAULT,
            dim: 3,
            infiles: 0,
            fnames: Vec::new(),
            sano: 0,
            add2file: IP_APPEND_FALSE,
            isec: 0,
            val: IP_DEFAULT as f32,
            nofsecs: IP_DEFAULT,
            secs: Vec::new(),
            out_before: IP_DEFAULT,
            out_after: IP_DEFAULT,
            ocanresize: 1,
            ocanchmode: 1,
            from_one: 0,
            ofname: None,
            plname: None,
            super_gain_name: None,
            point_out_name: None,
            new_xoverlap: IP_DEFAULT,
            new_yoverlap: IP_DEFAULT,
            rotation_flip: 0,
            read_defects: 0,
            defects: CameraDefects {
                was_scaled: 0,
                rotation_flip: 0,
                k2_type: 0,
                falcon_type: 0,
                usable_top: 0,
                usable_left: 0,
                usable_bottom: 0,
                usable_right: 0,
                num_avg_super_res: 0,
                bad_column_start: vec![],
                bad_column_width: vec![],
                partial_bad_col: vec![],
                partial_bad_width: vec![],
                partial_bad_start_y: vec![],
                partial_bad_end_y: vec![],
                bad_row_start: vec![],
                bad_row_height: vec![],
                partial_bad_row: vec![],
                partial_bad_height: vec![],
                partial_bad_start_x: vec![],
                partial_bad_end_x: vec![],
                bad_pixel_x: vec![],
                bad_pixel_y: vec![],
                pix_use_mean: vec![],
            },
            cam_size_x: 0,
            cam_size_y: 0,
            binning: IP_DEFAULT as f32,
            scale_defects: 0,
        };
        let mut volume = grap_volume_read(header, &mut options).expect("volume allocation");
        assert_eq!(volume.slices.len(), 3);
        let middle = volume.slices[1].as_ref();
        let values: Vec<f32> = (0..3)
            .flat_map(|y| (0..3).map(move |x| slice_get_pixel_magnitude(middle, x, y)))
            .collect();
        assert_eq!(values, [9., 9., 9., 9., 1., 2., 9., 3., 4.]);
        let output_name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let output_file = ii_open_new(output_name.to_bytes(), "wb", IIFILE_DEFAULT);
        assert!(!output_file.is_null());
        let output_header = (*output_file)
            .mrc_header
            .as_deref_mut()
            .expect("MRC header");
        assert_eq!(
            imod_rs::imod::clip::file_io::grap_volume_write(
                &mut volume,
                output_header,
                &mut options
            ),
            0
        );
        ii_close(output_file);
        let append_file = ii_open(output_name.to_bytes(), "rb+");
        assert!(!append_file.is_null());
        let append_header = (*append_file)
            .mrc_header
            .as_deref_mut()
            .expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*append_file).fp).as_mut().unwrap(), append_header),
            0
        );
        options.add2file = IP_APPEND_ADD;
        assert_eq!(
            imod_rs::imod::clip::file_io::grap_volume_write(
                &mut volume,
                append_header,
                &mut options
            ),
            0
        );
        ii_close(append_file);
        let truncate_file = ii_open(output_name.to_bytes(), "rb+");
        assert!(!truncate_file.is_null());
        let truncate_header = (*truncate_file)
            .mrc_header
            .as_deref_mut()
            .expect("MRC header");
        assert_eq!(
            mrc_head_read(
                (&mut (*truncate_file).fp).as_mut().unwrap(),
                truncate_header
            ),
            0
        );
        options.add2file = IP_APPEND_TRUNCATE;
        options.isec = 1;
        assert_eq!(
            imod_rs::imod::clip::file_io::grap_volume_write(
                &mut volume,
                truncate_header,
                &mut options
            ),
            0
        );
        ii_close(truncate_file);
        let overwrite_file = ii_open(output_name.to_bytes(), "rb+");
        assert!(!overwrite_file.is_null());
        let overwrite_header = (*overwrite_file)
            .mrc_header
            .as_deref_mut()
            .expect("MRC header");
        assert_eq!(
            mrc_head_read(
                (&mut (*overwrite_file).fp).as_mut().unwrap(),
                overwrite_header
            ),
            0
        );
        options.add2file = IP_APPEND_OVERWRITE;
        options.isec = 3;
        assert_eq!(
            imod_rs::imod::clip::file_io::grap_volume_write(
                &mut volume,
                overwrite_header,
                &mut options
            ),
            0
        );
        ii_close(overwrite_file);
        let mismatch_name = CString::new(mode_mismatch.to_string_lossy().as_bytes()).unwrap();
        let mismatch_file = ii_open_new(mismatch_name.to_bytes(), "wb", IIFILE_DEFAULT);
        assert!(!mismatch_file.is_null());
        let mismatch_header = (*mismatch_file)
            .mrc_header
            .as_deref_mut()
            .expect("MRC header");
        assert_eq!(mrc_head_new(mismatch_header, 3, 3, 6, MRC_MODE_SHORT), 0);
        ii_sync_from_mrc_header(&mut *mismatch_file, mismatch_header);
        assert_eq!(
            mrc_head_write(
                (&mut (*mismatch_file).fp).as_mut().unwrap(),
                mismatch_header
            ),
            0
        );
        options.add2file = IP_APPEND_OVERWRITE;
        options.isec = 0;
        assert_eq!(
            imod_rs::imod::clip::file_io::grap_volume_write(
                &mut volume,
                mismatch_header,
                &mut options
            ),
            -1
        );
        ii_close(mismatch_file);
        assert_eq!(grap_volume_free(volume), 0);
        ii_close(file);
    }
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!((header.nx, header.ny, header.nz), (3, 3, 6));
        let mut pixels = [0_f32; 9];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 3), 0);
        assert_eq!(pixels, [9.; 9]);
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 4), 0);
        assert_eq!(pixels, [9., 9., 9., 9., 1., 2., 9., 3., 4.]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
    let _ = std::fs::remove_file(mode_mismatch);
}

#[test]
fn multifile_real_mrc_mismatch_reports_source_file_io_diagnostic() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-file-io-multifile-{}", std::process::id()));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, nx, mut pixels) in [(&first, 2, vec![1_f32; 4]), (&second, 3, vec![2_f32; 6])] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, nx, 2, 1, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        format!(
            "ERROR: clip -  Files must be same mode and same size in X, Y, and Z; {} differs\n",
            second.to_string_lossy()
        )
    );
    assert!(result.stderr.is_empty());
    for path in [first, second, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn multifile_real_mrc_combines_source_sections_into_output_depth() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-multifile-success-{}",
        std::process::id()
    ));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, mut pixels) in [(&first, vec![1_f32, 2.]), (&second, vec![3_f32, 4.])] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), "Brightness...\n");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz), (2, 1, 2));
        for (section, expected) in [(0, [1_f32, 2.]), (1, [3., 4.])] {
            let mut pixels = [f32::NAN; 2];
            assert_eq!(ii_read_section_float(&mut *file, &mut pixels, section), 0);
            assert_eq!(pixels, expected);
        }
        ii_close(file);
    }
    for path in [first, second, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn multifile_real_mrc_append_is_rejected_before_output_lifecycle_setup() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-multifile-append-{}",
        std::process::id()
    ));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, mut pixels) in [
            (&first, vec![1_f32, 2.]),
            (&second, vec![3_f32, 4.]),
            (&output, vec![5_f32, 6.]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-a",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  Multiple input files can not be entered with appending or overwriting\n"
    );
    assert!(result.stderr.is_empty());
    for path in [first, second, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn multifile_real_mrc_rejects_source_blank_output_sections() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-file-io-multifile-blank-sections-{}",
        std::process::id()
    ));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, mut pixels) in [(&first, vec![1_f32, 2.]), (&second, vec![3_f32, 4.])] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-oz",
            "3",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  Blank slices cannot be output with multiple input files\n"
    );
    assert!(result.stderr.is_empty());
    for path in [first, second, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn divide_reuses_single_second_mrc_slice_and_rounds_integer_output() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-divide-{}", std::process::id()));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, sections) in [
            (&first, vec![[5_f32, 6., 7., 8.], [9., 10., 11., 12.]]),
            (&second, vec![[2_f32, 0., 3., 4.]]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(
                mrc_head_new(header, 2, 2, sections.len() as i32, MRC_MODE_SHORT),
                0
            );
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            for (section, mut pixels) in sections.into_iter().enumerate() {
                assert_eq!(
                    ii_write_section_float(&mut *file, &mut pixels, section as i32),
                    0
                );
            }
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "divide",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "\rclip: Dividing slice 1 of 2\rclip: Dividing slice 2 of 2\nWARNING: Division by zero occurred 2 times\n"
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (2, 2, 2, MRC_MODE_SHORT)
        );
        let title = String::from_utf8_lossy(&header.labels[0]);
        assert!(title.starts_with("clip: Divide"), "{title:?}");
        for (section, expected) in [[3_f32, 0., 2., 2.], [5., 0., 4., 3.]]
            .into_iter()
            .enumerate()
        {
            let mut pixels = [0_f32; 4];
            assert_eq!(
                ii_read_section_float(&mut *file, &mut pixels, section as i32),
                0
            );
            assert_eq!(pixels, expected);
        }
        ii_close(file);
    }
    for path in [first, second, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn normalize_applies_real_float_gain_reference_to_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-normalize-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let reference = base.with_extension("reference.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, mode, mut pixels) in [
            (&input, MRC_MODE_SHORT, [1_f32, 2., 3., 4.]),
            (&reference, MRC_MODE_FLOAT, [1_f32, 0.5, 2., 1.25]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 2, 1, mode), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "normalize",
            "-m",
            "float",
            input.to_str().unwrap(),
            reference.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "\rclip: processing slice 1 of 1\n"
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (2, 2, 1, MRC_MODE_FLOAT)
        );
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(
            label.starts_with("clip: Normalize, scaled by 16.00"),
            "{label:?}"
        );
        let mut pixels = [0_f32; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [16., 16., 96., 80.]);
        ii_close(file);
    }
    for path in [input, reference, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn average_reads_all_rows_of_real_multifile_mrc_inputs() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-average-{}", std::process::id()));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, mut pixels) in [
            (&first, [1.0_f32, 2., 3., 4., 5., 6.]),
            (&second, [11.0_f32, 12., 13., 14., 15., 16.]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 3, 1, 2), 0);
            if path == &first {
                mrc_head_label(header, b"first input lifecycle label");
            }
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    unsafe {
        for (path, expected) in [
            (&first, [1.0_f32, 2., 3., 4., 5., 6.]),
            (&second, [11.0_f32, 12., 13., 14., 15., 16.]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open(name.to_bytes(), "rb");
            let mut pixels = [0.0_f32; 6];
            assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
            assert_eq!(pixels, expected);
            ii_close(file);
        }
    }
    assert!(
        common::imod_cmd("clip")
            .args([
                "average",
                "-3d",
                first.to_str().unwrap(),
                second.to_str().unwrap(),
                output.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(&(&header.labels[0])[..27], b"first input lifecycle label");
        let mut pixels = [0.0_f32; 6];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [6.0_f32, 7., 8., 9., 10., 11.]);
        ii_close(file);
    }
    for path in [first, second, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_reads_and_writes_real_mrc_pixels() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, 2), 0);
        header.amin = 1.;
        header.amean = 2.5;
        header.amax = 4.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    assert!(
        common::imod_cmd("clip")
            .args([
                "brightness",
                "-n",
                "2",
                input.to_str().unwrap(),
                output.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );
    assert!(
        common::imod_cmd("clip")
            .args([
                "brightness",
                "-a",
                "-n",
                "2",
                input.to_str().unwrap(),
                output.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut pixels = [0.0_f32; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [1., 3., 5., 7.]);
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 1), 0);
        assert_eq!(pixels, [1., 3., 5., 7.]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn brightness_reads_each_real_multifile_mrc_header_and_slice() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-multifile-{}", std::process::id()));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, pixels) in [
            (&first, [1.0_f32, 2.0, 3.0, 4.0]),
            (&second, [11.0_f32, 12.0, 13.0, 14.0]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 2, 1, 2), 0);
            header.amin = pixels[0];
            header.amean = (pixels[0] + pixels[1] + pixels[2] + pixels[3]) / 4.0;
            header.amax = pixels[3];
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            let mut pixels = pixels;
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-n",
            "2",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            output.to_str().unwrap(),
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
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz), (2, 2, 2));
        let mut pixels = [0.0_f32; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [1., 3., 5., 7.]);
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 1), 0);
        assert_eq!(pixels, [21., 23., 25., 27.]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
    let _ = std::fs::remove_file(output);
}

#[test]
fn unwrap_copies_a_real_default_selection_extended_header() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-unwrap-extra-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&input, "wb")
                .unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_SHORT), 0);
        header.fp = Some(file.clone());
        header.next = 8;
        header.nint = 2;
        header.nreal = 0;
        header.ext_type = *b"AGAR";
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let extra = [7_u8, 2, 8, 1, 5, 4, 3, 6];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(extra.as_ptr().cast::<u8>(), 1 * (extra.len()))
                },
                1,
                extra.len(),
                &mut file,
            ),
            extra.len()
        );
        let pixels = [1_i16, 2, 3, 4];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(
                        pixels.as_ptr().cast::<u8>(),
                        core::mem::size_of::<i16>() * pixels.len(),
                    )
                },
                core::mem::size_of::<i16>(),
                pixels.len(),
                &mut file,
            ),
            pixels.len()
        );
        drop(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "unwrap",
            "-n",
            "32768",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
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
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&output, "rb")
                .unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!(
            (header.next, header.nint, header.nreal, header.ext_type),
            (8, 2, 0, *b"AGAR")
        );
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                1024 as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut extra = [0_u8; 8];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        extra.as_mut_ptr().cast::<u8>(),
                        1 * (extra.len()),
                    )
                },
                1,
                extra.len(),
                &mut file,
            ),
            extra.len()
        );
        assert_eq!(extra, [7, 2, 8, 1, 5, 4, 3, 6]);
        drop(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn boxsd_keeps_source_output_scale_after_real_map_setup() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-boxsd-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 16, 16, 1, 2), 0);
        header.xlen = 12.;
        header.ylen = 20.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 256];
        for (index, value) in pixels.iter_mut().enumerate() {
            *value = index as f32;
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "boxsd",
            "-n",
            "2",
            "-l",
            "8",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
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
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (8, 8, 1, 2));
        // `clip_scaling` reads the scaled header just after `mrc_set_scale`,
        // then its source tail calls `set_mrc_coords`, whose `mrc_coord_cp`
        // restores the input pixel spacing on the smaller sample dimensions.
        assert_eq!((header.xlen, header.ylen), (6., 10.));
        let mut pixels = [f32::NAN; 64];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert!(pixels.iter().all(|value| value.is_finite()));
        assert!((pixels[0] - 36.850407).abs() < 1.0e-5);
        assert!((pixels[7] - 36.851738).abs() < 1.0e-5);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn boxsd_rejects_subunit_reduction_before_real_mrc_output_open() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-boxsd-bad-reduction-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "boxsd",
            "-n",
            "0.5",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  Reduction factor (-n) must be at least 1 for boxsd process\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn defect_list_without_both_camera_sizes_fatals_before_real_mrc_output_open() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-defect-camera-size-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let defects = base.with_extension("defects.txt");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    std::fs::write(&defects, "CameraSizeX 1\n").unwrap();
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-Defects",
            defects.to_str().unwrap(),
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  Problem with defect correction - Defect list file must have CameraSizeX and CameraSizeY entries\n"
    );
    assert!(!output.exists());
    for path in [input, defects] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn supergain_rejects_eer_option_before_real_mrc_lifecycle() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-supergain-eer-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.txt");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "supergain",
            "-es",
            "2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  The -es, -ez, and other EER options cannot be entered with the supergain operation\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn defect_binning_below_source_minimum_fatals_before_real_mrc_output_open() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-defect-binning-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-Binning",
            "0.49",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  Binning must be at least 0.5\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn missing_defect_list_reports_source_open_error_before_real_mrc_lifecycle() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-defect-missing-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let missing = base.with_extension("missing.defects");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-D",
            missing.to_str().unwrap(),
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        // Confirmed against native: `clip.cpp:562` hands three arguments to a
        // format string with a single %s, so the name is never printed.
        "ERROR: clip -  Error opening\n".to_string()
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn invalid_mode_reports_source_error_before_real_mrc_output_open() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-invalid-mode-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-mode",
            "invalid-mode",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  Invalid mode entry invalid-mode.\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn flatfield_forces_source_float_mode_for_real_mrc_byte_request() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-flatfield-force-float-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 15, 15, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 225];
        for (index, pixel) in pixels.iter_mut().enumerate() {
            *pixel = 100. + index as f32;
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "flatfield",
            "-m",
            "byte",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8_lossy(&result.stdout).starts_with(
            // Confirmed against native: `clip.cpp:421` passes the bare sentence.
            "WARNING: Output mode for a flatfield image must be floating point\n"
        ),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(header.mode, MRC_MODE_FLOAT);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn invalid_option_reports_source_fatal_before_real_mrc_lifecycle() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-invalid-option-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-q",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  Invalid option -q.\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn invalid_output_format_reports_source_fatal_before_real_mrc_lifecycle() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-invalid-format-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-format",
            "not-a-format",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  Output file format entry not-a-format is not recognized.\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn average_accepts_source_short_2d_option_for_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-short-2d-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 2, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for (section, mut pixel) in [[2_f32], [4_f32]].into_iter().enumerate() {
            assert_eq!(
                ii_write_section_float(&mut *file, &mut pixel, section as i32),
                0
            );
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "average",
            "-2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(header.nz, 1);
        let mut pixel = [f32::NAN];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixel, 0), 0);
        assert_eq!(pixel, [3.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn append_accepts_source_second_character_option_spelling_for_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-append-prefix-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, value) in [(&input, 3_f32), (&output, 2_f32)] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            let mut pixel = [value];
            assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-append",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(header.nz, 2);
        let mut old = [f32::NAN];
        let mut appended = [f32::NAN];
        assert_eq!(ii_read_section_float(&mut *file, &mut old, 0), 0);
        assert_eq!(ii_read_section_float(&mut *file, &mut appended, 1), 0);
        assert_eq!(old, [2.]);
        assert_eq!(appended, [3.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_accepts_source_second_character_number_option_for_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-number-prefix-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        header.amin = 1.;
        header.amax = 3.;
        header.amean = 2.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 3.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-number",
            "2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let mut pixels = [f32::NAN; 2];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [1., 5.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn threshold_accepts_source_second_character_sano_option_for_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-sano-prefix-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        header.amin = 1.;
        header.amax = 3.;
        header.amean = 2.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 3.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "threshold",
            "-source-range",
            "-threshold",
            "2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let mut pixels = [f32::NAN; 2];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [1., 3.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn blankfile_accepts_source_second_character_pad_option_for_real_mrc() {
    let output = std::env::temp_dir().join(format!(
        "imod-rs-clip-padding-prefix-{}.mrc",
        std::process::id()
    ));
    let result = common::imod_cmd("clip")
        .args([
            "blankfile",
            "-ox",
            "1",
            "-oy",
            "1",
            "-oz",
            "1",
            "-m",
            "float",
            "-padding",
            "7.5",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let mut pixel = [f32::NAN];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixel, 0), 0);
        assert_eq!(pixel, [7.5]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(output);
}

#[test]
fn missing_output_for_real_mrc_exits_with_source_usage_status() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-clip-missing-output-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["brightness", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(3));
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .starts_with("clip: Command Line Image Processing."),
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn unknown_command_exits_with_source_usage_status_before_real_mrc_open() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-unknown-command-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "no-such-process",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(
        String::from_utf8(result.stdout)
            .unwrap()
            .starts_with("clip: Command Line Image Processing."),
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn standalone_z_option_reports_source_invalid_option_for_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-standalone-z-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-z",
            "0",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  Invalid option -z.\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn long_one_based_option_overwrites_source_first_section_in_real_mrc() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-one-based-prefix-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, values) in [(&input, vec![3_f32]), (&output, vec![2_f32, 4_f32])] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(
                mrc_head_new(header, 1, 1, values.len() as i32, MRC_MODE_FLOAT),
                0
            );
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            for (section, mut pixel) in values.into_iter().map(|value| [value]).enumerate() {
                assert_eq!(
                    ii_write_section_float(&mut *file, &mut pixel, section as i32),
                    0
                );
            }
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-2",
            "-1based",
            "-or",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(header.nz, 2);
        let mut first = [f32::NAN];
        let mut second = [f32::NAN];
        assert_eq!(ii_read_section_float(&mut *file, &mut first, 0), 0);
        assert_eq!(ii_read_section_float(&mut *file, &mut second, 1), 0);
        assert_eq!(first, [3.]);
        assert_eq!(second, [4.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn defectmap_writes_source_byte_map_for_real_mrc_defect_list() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-defectmap-mrc-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let defects = base.with_extension("defects.txt");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32; 4];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    std::fs::write(&defects, "CameraSizeX 2\nCameraSizeY 2\n").unwrap();
    let result = common::imod_cmd("clip")
        .args([
            "defectmap",
            "-Defects",
            defects.to_str().unwrap(),
            "-Scale",
            "-Rotation",
            "0",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!((header.mode, header.nz), (MRC_MODE_BYTE, 1));
        let mut pixels = [f32::NAN; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [0.; 4]);
        ii_close(file);
    }
    for path in [input, defects, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn defectmap_writes_real_tiff_through_source_iimage_dispatch() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-defectmap-tiff-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let defects = base.with_extension("defects.txt");
    let output = base.with_extension("output.tif");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32; 4];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    std::fs::write(&defects, "CameraSizeX 2\nCameraSizeY 2\n").unwrap();
    let result = common::imod_cmd("clip")
        .args([
            "defectmap",
            "-f",
            "TIFF",
            "-D",
            defects.to_str().unwrap(),
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        assert_eq!((*file).file, IIFILE_TIFF);
        assert_eq!(
            ((*file).nx, (*file).ny, (*file).nz, (*file).mode),
            (2, 2, 1, MRC_MODE_BYTE)
        );
        let mut compression = 0_i32;
        assert_ne!(
            tiff_get_field(file, 259, (&mut compression as *mut i32).cast()),
            0
        );
        assert_eq!(compression, 8);
        let mut pixels = [f32::NAN; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [0.; 4]);
        ii_close(file);
    }
    for path in [input, defects, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_forces_source_bigtiff_output_for_real_mrc() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-bigtiff-output-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.tif");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 2., 3., 4.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .env("IMOD_ALL_BIG_TIFF", "1")
        .args([
            "brightness",
            "-f",
            "TIFF",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(&std::fs::read(&output).unwrap()[..4], b"II+\0");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        assert_eq!((*file).file, IIFILE_TIFF);
        let mut pixels = [f32::NAN; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [1., 2., 3., 4.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_writes_source_serialemccd_tiff_description() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-tiff-description-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.tif");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        mrc_head_label(header, b"SerialEMCCD 4 bits packed");
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-f",
            "TIFF",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let mut description = core::ptr::null_mut();
        assert_ne!(
            tiff_get_field(file, 270, (&mut description as *mut *mut i8).cast()),
            0
        );
        assert!(
            std::ffi::CStr::from_ptr(description)
                .to_bytes()
                .starts_with(b"SerialEMCCD 4-bits packed"),
            "{:?}",
            std::ffi::CStr::from_ptr(description).to_bytes()
        );
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn brightness_tiff_readback_strips_source_cr_before_description_newline() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-tiff-cr-label-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.tif");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        (&mut header.labels[0])[..6].copy_from_slice(b"first\r");
        (&mut header.labels[1])[..6].copy_from_slice(b"second");
        header.nlabl = 2;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-f",
            "TIFF",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        assert_eq!((*file).last_written_z, 0);
        let mut header = MrcHeader::default();
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), &mut header),
            0
        );
        assert_eq!(&header.labels[0][..5], b"first");
        assert_eq!(header.labels[0][5], b' ');
        assert_eq!(&header.labels[1][..6], b"second");
        assert_eq!(header.labels[1][6], b' ');
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn threshold_accepts_source_second_character_lower_option_for_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-lower-prefix-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        header.amin = 1.;
        header.amax = 3.;
        header.amean = 2.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32, 3.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "threshold",
            "-lower",
            "5",
            "-t",
            "2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let mut pixel = [f32::NAN; 2];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixel, 0), 0);
        assert_eq!(pixel, [5., 255.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn threshold_accepts_source_second_character_higher_option_for_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-higher-prefix-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        header.amin = 1.;
        header.amax = 3.;
        header.amean = 2.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32, 3.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "threshold",
            "-higher",
            "7",
            "-t",
            "2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let mut pixel = [f32::NAN; 2];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixel, 0), 0);
        assert_eq!(pixel, [0., 7.]);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn integral_uses_a_real_immutable_float_reference_slice() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-integral-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 10, 10, 1, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 100];
        for y in 0..10 {
            for x in 0..10 {
                pixels[x + y * 10] = (x * x + 3 * y) as f32;
            }
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "integral",
            "-n",
            "1",
            "-h",
            "20",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
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
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut values = [f32::NAN; 100];
        assert_eq!(ii_read_section_float(&mut *file, &mut values, 0), 0);
        assert_eq!(values[0], 0.);
        assert_eq!(values[9], 0.);
        assert_eq!(values[90], 0.);
        assert_eq!(values[99], 0.);
        assert!(values[4 + 4 * 10].is_finite());
        assert!(values.iter().all(|value| value.is_finite()));
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn integral_uses_source_x_extent_for_its_lower_y_border_on_non_square_mrc() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-integral-nonsquare-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 10, 12, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 120];
        // A peak precisely in the source's ix-based lower-Y border makes the
        // distinction observable: a conventional iy-based boundary would process it.
        pixels[4 + 7 * 10] = 1000.;
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "integral",
            "-n",
            "1",
            "-h",
            "1000000",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    unsafe {
        let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut values = [f32::NAN; 120];
        assert_eq!(ii_read_section_float(&mut *file, &mut values, 0), 0);
        // C++ processing.cpp tests j against ix here, rather than iy.  Thus its
        // lower Y border starts at 10 - 3, even though this image is 12 rows tall.
        assert_eq!(values[4 + 7 * 10], 0.);
        ii_close(file);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn logarithm_and_square_root_follow_single_precision_real_mrc_transforms() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-log-sqrt-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let log_output = base.with_extension("log.mrc");
    let sqrt_output = base.with_extension("sqrt.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, 2), 0);
        header.amin = -5.;
        header.amax = 100.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [-5_f32, -1., 0., 100.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    for (command, output) in [("logarithm", &log_output), ("sqroot", &sqrt_output)] {
        let result = common::imod_cmd("clip")
            .args([
                command,
                "-n",
                "1.234567",
                input.to_str().unwrap(),
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    unsafe {
        for (path, expected) in [
            (
                &log_output,
                [
                    0.00105_f32.log10(),
                    0.234567_f32.log10(),
                    1.234567_f32.log10(),
                    101.234566_f32.log10(),
                ],
            ),
            (
                &sqrt_output,
                [
                    0_f32,
                    0.234567_f32.sqrt(),
                    1.234567_f32.sqrt(),
                    101.234566_f32.sqrt(),
                ],
            ),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open(name.to_bytes(), "rb");
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(
                mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            if path == &log_output {
                let label = String::from_utf8_lossy(&header.labels[0]);
                assert!(
                    label.starts_with("clip: logarithm after adding 1.23457"),
                    "{label:?}"
                );
            }
            let mut values = [f32::NAN; 4];
            assert_eq!(ii_read_section_float(&mut *file, &mut values, 0), 0);
            for (got, expected) in values.into_iter().zip(expected) {
                assert!((got - expected).abs() < 1.0e-6, "{got} != {expected}");
            }
            ii_close(file);
        }
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(log_output);
    let _ = std::fs::remove_file(sqrt_output);
}

#[test]
fn threshold_writes_source_formatted_points_for_real_mrc_pixels() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-threshold-points-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let points = base.with_extension("points.txt");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 3., 2., 4.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "threshold",
            "-t",
            "2",
            "-op",
            points.to_str().unwrap(),
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(&points).unwrap(),
        "     1      0    0  3\n     1      1    0  4\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut values = [f32::NAN; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut values, 0), 0);
        assert_eq!(values, [0., 255., 0., 255.]);
        ii_close(file);
    }
    let failed_output = base.with_extension("failed-output.mrc");
    let missing_points = base.join("missing-parent").join("points.txt");
    let failed = common::imod_cmd("clip")
        .args([
            "threshold",
            "-t",
            "2",
            "-op",
            missing_points.to_str().unwrap(),
            input.to_str().unwrap(),
            failed_output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(failed.status.code(), Some(255));
    assert_eq!(
        String::from_utf8(failed.stdout).unwrap(),
        format!(
            "Threshold...\nERROR: clip threshold: Error opening output file for points {}\n",
            missing_points.display()
        )
    );
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
    let _ = std::fs::remove_file(points);
    let _ = std::fs::remove_file(failed_output);
}

#[test]
fn brightness_keeps_a_nondefault_float_near_the_source_sentinel() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-sentinel-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, 2), 0);
        header.amin = 1.;
        header.amax = 2.;
        header.amean = 1.5;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 2.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "brightness",
            "-n",
            "-99999.5",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
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
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut values = [f32::NAN; 2];
        assert_eq!(ii_read_section_float(&mut *file, &mut values, 0), 0);
        assert_eq!(values, [1., -99998.5]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn contrast_2d_uses_each_real_slice_mean_from_source_slice_mmm() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-contrast-2d-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 2, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut first = [1_f32, 3.];
        let mut second = [10_f32, 14.];
        assert_eq!(ii_write_section_float(&mut *file, &mut first, 0), 0);
        assert_eq!(ii_write_section_float(&mut *file, &mut second, 1), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "contrast",
            "-2d",
            "-n",
            "2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
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
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.amin, header.amax, header.amean), (0., 16., 7.));
        let mut values = [f32::NAN; 2];
        assert_eq!(ii_read_section_float(&mut *file, &mut values, 0), 0);
        assert_eq!(values, [0., 4.]);
        assert_eq!(ii_read_section_float(&mut *file, &mut values, 1), 0);
        assert_eq!(values, [8., 16.]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn sobel_uses_clip_edge_float_to_byte_and_real_mrc_header_path() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-sobel-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 8, 8, 1, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 64];
        for y in 2..6 {
            for x in 2..6 {
                pixels[x + y * 8] = 20.;
            }
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["sobel", input.to_str().unwrap(), output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "clip: Applying Sobel filter to 1 slices...\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (8, 8, 1, 0));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(label.starts_with("clip: Sobel filter"), "{label:?}");
        let mut values = [0_f32; 64];
        assert_eq!(ii_read_section_float(&mut *file, &mut values, 0), 0);
        assert!(values.iter().any(|value| *value > 0.));
        assert!(values.iter().all(|value| value.is_finite()));
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn color_3d_writes_source_scaled_rgb_bytes_to_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-color-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&input, "wb")
                .unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 1, 1, 0), 0);
        header.amin = 0.;
        header.amax = 255.;
        header.amean = 127.5;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let pixels = [10_u8, 100];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(pixels.as_ptr().cast::<u8>(), 1 * (pixels.len()))
                },
                1,
                pixels.len(),
                &mut file,
            ),
            pixels.len()
        );
        drop(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "color",
            "-red",
            "1.5",
            "-green",
            "0.5",
            "-blue",
            "3",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "clip: color (red, green, blue) = ( 1.5, 0.5, 3).\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&output, "rb")
                .unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (2, 1, 1, 16)
        );
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut rgb = [0_u8; 6];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(rgb.as_mut_ptr().cast::<u8>(), 1 * (rgb.len()))
                },
                1,
                rgb.len(),
                &mut file,
            ),
            rgb.len()
        );
        assert!(rgb.iter().any(|value| *value != 0));
        assert_ne!(&rgb[..3], &rgb[3..]);
        drop(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn color_2d_writes_source_rounded_rgb_bytes_to_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-color-2d-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 1, 1, 2), 0);
        header.amin = 10.;
        header.amean = 55.;
        header.amax = 100.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [10_f32, 100.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "color",
            "-2d",
            "-r",
            "1.5",
            "-g",
            "0.5",
            "-b",
            "3",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "False Color...\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&output, "rb")
                .unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (2, 1, 1, 16)
        );
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut rgb = [0_u8; 6];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(rgb.as_mut_ptr().cast::<u8>(), 1 * (rgb.len()))
                },
                1,
                rgb.len(),
                &mut file,
            ),
            rgb.len()
        );
        assert_eq!(rgb, [15, 5, 30, 150, 50, 255]);
        drop(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn laplacian_convolve_filters_a_real_mrc_slice() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-laplacian-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 3, 1, 2), 0);
        header.amin = 1.;
        header.amean = 5.;
        header.amax = 9.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 2., 3., 4., 5., 6., 7., 8., 9.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "laplacian",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "clip: Applying Laplacian to 1 slices...\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (3, 3, 1, 2));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(label.starts_with("clip: Laplacian"), "{label:?}");
        let mut pixels = [0_f32; 9];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels[4], 20.);
        assert!(pixels.iter().all(|value| value.is_finite()));
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn smooth_default_uses_the_source_3d_gaussian_route_on_real_mrc_data() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-smooth-3d-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 3, 3, 2), 0);
        header.amin = 0.;
        header.amean = 1. / 27.;
        header.amax = 1.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for z in 0..3 {
            let mut pixels = [0_f32; 9];
            if z == 1 {
                pixels[4] = 1.;
            }
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["smooth", input.to_str().unwrap(), output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "clip: Gaussian kernel 3D smoothing 3 slices...\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (3, 3, 3, 2));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(
            label.starts_with("clip: Gaussian 3D smoothing, sigma 0.85"),
            "{label:?}"
        );
        let mut middle = [0_f32; 9];
        assert_eq!(ii_read_section_float(&mut *file, &mut middle, 1), 0);
        assert!(middle.iter().all(|value| value.is_finite()));
        assert!(middle[4] > 0. && middle[4] < 1., "{middle:?}");
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn median_2d_removes_a_real_mrc_salt_pixel() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-median-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 3, 1, 2), 0);
        header.amin = 1.;
        header.amean = 12.;
        header.amax = 100.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 1., 1., 1., 100., 1., 1., 1., 1.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "median",
            "-2d",
            "-n",
            "3",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "clip: median filtering 1 slices...\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (3, 3, 1, 2));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(
            label.starts_with("clip: 2D median filter, size 3"),
            "{label:?}"
        );
        let mut pixels = [0_f32; 9];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels[4], 1.);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn median_3d_uses_the_source_slice_window_on_real_mrc_data() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-median-3d-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 3, 3, 3, 2), 0);
        header.amin = 1.;
        header.amean = 14. / 27.;
        header.amax = 100.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for z in 0..3 {
            let mut pixels = [1_f32; 9];
            if z == 1 {
                pixels[4] = 100.;
            }
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "median",
            "-3d",
            "-n",
            "3",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "clip: median filtering 3 slices...\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut middle = [0_f32; 9];
        assert_eq!(ii_read_section_float(&mut *file, &mut middle, 1), 0);
        assert_eq!(middle[4], 1.);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn blankfile_writes_a_real_constant_mrc_volume() {
    let output =
        std::env::temp_dir().join(format!("imod-rs-clip-blankfile-{}.mrc", std::process::id()));
    let result = common::imod_cmd("clip")
        .args([
            "blankfile",
            "-ox",
            "2",
            "-oy",
            "3",
            "-oz",
            "2",
            "-m",
            "2",
            "-p",
            "7.5",
            output.to_str().unwrap(),
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
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (2, 3, 2, 2));
        assert_eq!((header.amin, header.amean, header.amax), (7.5, 7.5, 7.5));
        assert_eq!((header.xorg, header.yorg, header.zorg), (0., 0., 0.));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(
            label.starts_with("clip blankfile: constant value 7.5"),
            "{label:?}"
        );
        let mut pixels = [0_f32; 6];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [7.5; 6]);
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 1), 0);
        assert_eq!(pixels, [7.5; 6]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(output);
}

#[test]
fn blankfile_rejects_nonpositive_source_output_size_before_output_open() {
    let output = std::env::temp_dir().join(format!(
        "imod-rs-clip-blankfile-nonpositive-{}.mrc",
        std::process::id()
    ));
    let result = common::imod_cmd("clip")
        .args([
            "blankfile",
            "-ox",
            "0",
            "-oy",
            "1",
            "-oz",
            "1",
            "-m",
            "2",
            "-p",
            "7.5",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  You must enter a positive output size for all dimensions\n"
    );
    assert!(!output.exists());
}

#[test]
fn chunk_sizes_reject_explicit_nonhdf_output_format_before_real_mrc_open() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-chunk-nonhdf-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "resize",
            "-CX",
            "16",
            "-f",
            "mrc",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  You cannot specify chunk sizes and an output format other than HDF\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn resize_rejects_source_x_and_center_conflict_before_real_mrc_output_open() {
    let base = std::env::temp_dir().join(format!(
        "imod-rs-clip-resize-x-center-conflict-{}",
        std::process::id()
    ));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 1, 1, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixel = [1_f32];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixel, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "resize",
            "-x",
            "0,0",
            "-cx",
            "0",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip -  You cannot use -x together with -cx or -ix\n"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_file(input);
}

#[test]
fn diffusion_filters_a_real_mrc_slice_with_source_status_and_title() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-diffusion-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 5, 5, 1, 2), 0);
        header.amin = 0.;
        header.amean = 4.;
        header.amax = 100.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 25];
        pixels[12] = 100.;
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "diffusion",
            "-n",
            "1",
            "-cc",
            "2",
            "-weight",
            "2",
            "-l",
            "0.2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "clip: anistropic diffusion 1 slices...\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (5, 5, 1, 2));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(label.starts_with("clip: diffusion"), "{label:?}");
        let mut pixels = [0_f32; 25];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert!(pixels.iter().all(|value| value.is_finite()));
        assert!(pixels[12] < 100. && pixels[12] > 0., "{pixels:?}");
        assert!(pixels.iter().any(|value| *value > 0. && *value < 100.));
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn flipxy_transposes_each_real_mrc_volume_slice() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-flipxy-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 3, 2, 2), 0);
        header.amin = 1.;
        header.amean = 6.5;
        header.amax = 12.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for z in 0..2 {
            let mut pixels = if z == 0 {
                [1_f32, 2., 3., 4., 5., 6.]
            } else {
                [7_f32, 8., 9., 10., 11., 12.]
            };
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["flipxy", input.to_str().unwrap(), output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), " Done!\n");
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (3, 2, 2, 2));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(label.starts_with("clip: flipxy"), "{label:?}");
        let mut pixels = [0_f32; 6];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [1., 3., 5., 2., 4., 6.]);
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 1), 0);
        assert_eq!(pixels, [7., 9., 11., 8., 10., 12.]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn rotx_rotates_real_mrc_yz_planes_in_source_order() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-rotx-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 3, 2, 2), 0);
        header.amin = 1.;
        header.amean = 6.5;
        header.amax = 12.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for z in 0..2 {
            let mut pixels = if z == 0 {
                [1_f32, 2., 3., 4., 5., 6.]
            } else {
                [7_f32, 8., 9., 10., 11., 12.]
            };
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
        }
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["rotx", input.to_str().unwrap(), output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), " Done!\n");
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (2, 2, 3, 2));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(
            label.starts_with("clip: rotx - rotation by -90 around X"),
            "{label:?}"
        );
        let mut pixels = [0_f32; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [5., 6., 11., 12.]);
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 1), 0);
        assert_eq!(pixels, [3., 4., 9., 10.]);
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 2), 0);
        assert_eq!(pixels, [1., 2., 7., 8.]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn quadrant_corrects_real_mrc_quadrant_intensities() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-quadrant-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 100, 100, 1, 2), 0);
        header.amin = 40.;
        header.amean = 70.;
        header.amax = 100.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 10_000];
        for y in 0..100 {
            for x in 0..100 {
                pixels[x + y * 100] = match (x < 50, y < 50) {
                    (true, true) => 60.,
                    (false, true) => 40.,
                    (true, false) => 80.,
                    (false, false) => 100.,
                };
            }
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "quadrant",
            "-h",
            "5",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(
        stdout.starts_with("Section 0: scale factors "),
        "{stdout:?}"
    );
    assert!(stdout.contains("Boundary diffs before:"), "{stdout:?}");
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (100, 100, 1, 2)
        );
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(label.starts_with("clip: quadrant correction"), "{label:?}");
        let mut pixels = [0_f32; 10_000];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        let mut means = [0_f32; 4];
        for y in 0..100 {
            for x in 0..100 {
                means[(x >= 50) as usize + 2 * (y >= 50) as usize] += pixels[x + y * 100] / 2500.;
            }
        }
        assert!(means.iter().all(|mean| mean.is_finite()), "{means:?}");
        assert_ne!(means, [60., 40., 80., 100.]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn edgefill_processes_real_short_mrc_drift_edges() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-edgefill-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 64, 64, 1, 1), 0);
        header.amin = 0.;
        header.amean = 90.;
        header.amax = 106.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 4096];
        for y in 0..64 {
            for x in 0..64 {
                pixels[x + y * 64] = if x < 3 || y < 3 || x >= 61 || y >= 61 {
                    0.
                } else {
                    100. + ((x + 3 * y) % 7) as f32
                };
            }
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "edgefill",
            "-l",
            "32",
            "-n",
            "4",
            "-h",
            "2",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), "\nSLICE 0\n");
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (64, 64, 1, 1)
        );
        let mut pixels = [0_f32; 4096];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert!(pixels.iter().all(|value| value.is_finite()));
        assert!(pixels.iter().any(|value| *value > 0.));
        assert!(
            pixels[0] > 0.,
            "corner was not drift-edge corrected: {}",
            pixels[0]
        );
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn spectrum_writes_a_real_mrc_slice_with_source_header_scale() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-spectrum-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 8, 8, 1, 2), 0);
        header.amin = 0.;
        header.amean = 1. / 64.;
        header.amax = 1.;
        header.xlen = 16.;
        header.ylen = 24.;
        header.zlen = 4.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 64];
        pixels[4 + 4 * 8] = 1.;
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "spectrum",
            "-l",
            "0",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "clip: Taking power spectrum of 1 slices...\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (8, 8, 1, 1));
        assert_eq!((header.xlen, header.ylen, header.zlen), (8., 8., 1.));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(
            label.starts_with("clip: scaled power spectrum"),
            "{label:?}"
        );
        let mut pixels = [0_f32; 64];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert!(pixels.iter().any(|value| *value > 0.));
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn joinrgb_combines_three_real_byte_mrc_files() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-joinrgb-{}", std::process::id()));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let third = base.with_extension("third.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for (path, mut pixels) in [
            (&first, [10_f32, 100.]),
            (&second, [20_f32, 110.]),
            (&third, [30_f32, 90.]),
        ] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 2, 1, 1, 0), 0);
            header.amin = pixels[0];
            header.amean = (pixels[0] + pixels[1]) / 2.;
            header.amax = pixels[1];
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "joinrgb",
            "-r",
            "1.5",
            "-g",
            "0.5",
            "-b",
            "3",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            third.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "\rJoining section 1 of 1\n"
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&output, "rb")
                .unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut header), 0);
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (2, 1, 1, 16)
        );
        assert_eq!((header.amin, header.amean, header.amax), (0., 128., 255.));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(label.starts_with("CLIP Join 3 files into RGB"), "{label:?}");
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                &mut file,
                header.header_size as i32,
                imod_rs::imod::libcfshr::b3dutil::SEEK_SET
            ),
            0
        );
        let mut rgb = [0_u8; 6];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(rgb.as_mut_ptr().cast::<u8>(), 1 * (rgb.len()))
                },
                1,
                rgb.len(),
                &mut file,
            ),
            rgb.len()
        );
        // Verified against the reference: `clip joinrgb -r 1.5 -g 0.5 -b 3`
        // on these three byte files gives [15, 10, 90, 150, 55, 14].  The blue
        // channel is 90 * 3 = 270, and `islice.c:270` narrows it with
        // `(unsigned char)`, which truncates and keeps the low bits — 14, not a
        // saturated 255.  The old expectation encoded Rust's saturating
        // `as u8`, which this test was written against before that was fixed.
        assert_eq!(rgb, [15, 10, 90, 150, 55, 14]);
        drop(file);
    }
    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
    let _ = std::fs::remove_file(third);
    let _ = std::fs::remove_file(output);
}

#[test]
fn splitrgb_writes_three_real_byte_mrc_files() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-splitrgb-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let prefix = base.with_extension("channels");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let mut file =
            imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&input, "wb")
                .unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 1, 1, 16), 0);
        header.amin = 0.;
        header.amean = 128.;
        header.amax = 255.;
        header.fp = Some(file.clone());
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let rgb = [15_u8, 10, 90, 150, 55, 255];
        assert_eq!(
            imod_rs::imod::libcfshr::b3dutil::b3d_fwrite(
                unsafe { core::slice::from_raw_parts(rgb.as_ptr().cast::<u8>(), 1 * (rgb.len()),) },
                1,
                rgb.len(),
                &mut file,
            ),
            rgb.len()
        );
        drop(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "splitrgb",
            input.to_str().unwrap(),
            prefix.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "\rSplitting section 1 of 1\n"
    );
    for (extension, expected) in [
        ("r", [143_u8, 22]),
        ("g", [138_u8, 183]),
        ("b", [218_u8, 127]),
    ] {
        let path = prefix.with_extension(format!("channels.{extension}"));
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        unsafe {
            let mut file =
                imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&path, "rb")
                    .unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut file, &mut header), 0);
            assert_eq!((header.nx, header.ny, header.nz, header.mode), (2, 1, 1, 0));
            assert_ne!(header.bytes_signed, 0);
            let label = String::from_utf8_lossy(&header.labels[0]);
            assert!(
                label.starts_with("CLIP Split RGB into 3 files"),
                "{label:?}"
            );
            assert_eq!(
                imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    header.header_size as i32,
                    imod_rs::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut pixels = [0_u8; 2];
            assert_eq!(
                imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(
                            pixels.as_mut_ptr().cast::<u8>(),
                            1 * (pixels.len()),
                        )
                    },
                    1,
                    pixels.len(),
                    &mut file,
                ),
                pixels.len()
            );
            assert_eq!(pixels, expected);
            drop(file);
        }
        let _ = std::fs::remove_file(path);
    }
    let _ = std::fs::remove_file(input);
}

#[test]
fn fft_writes_real_complex_mrc_with_source_dimensions() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-fft-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 4, 4, 1, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [
            1.0_f32, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    assert!(
        common::imod_cmd("clip")
            .args(["fft", input.to_str().unwrap(), output.to_str().unwrap()])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (3, 4, 1, MRC_MODE_COMPLEX_FLOAT)
        );
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn filter_reads_real_mrc_through_slice_read_subm() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-filter-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 4, 4, 1, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [
            1.0_f32, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    assert!(
        common::imod_cmd("clip")
            .args([
                "filter",
                "-l",
                "1",
                "-h",
                "0",
                input.to_str().unwrap(),
                output.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (4, 4, 1, 2));
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn info_prints_complete_real_mrc_header_path() {
    let path = std::env::temp_dir().join(format!("imod-rs-clip-info-{}.mrc", std::process::id()));
    let path_c = CString::new(path.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(path_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 3, 1, 2), 0);
        header.xorg = 1.5;
        header.yorg = 2.5;
        header.zorg = 3.5;
        header.amin = 0.000001;
        header.amax = 1000000.;
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0.0_f32; 6];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["info", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(result.status.success());
    let text = String::from_utf8(result.stdout).unwrap();
    assert!(text.contains("MRC header info:\nmode = Float"));
    assert!(text.contains("Image size    =  ( 2, 3, 1)"));
    assert!(text.contains("minimum value = 1e-06"), "{text}");
    assert!(text.contains("maximum value = 1e+06"), "{text}");
    assert!(text.contains("orgin  = ( 1.5, 2.5, 3.5)"));
    let _ = std::fs::remove_file(path);
}

#[test]
fn correlate_auto_reads_and_writes_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-corr-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 4, 4, 1, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [
            1.0_f32, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    assert!(
        common::imod_cmd("clip")
            .args([
                "correlation",
                "-2d",
                input.to_str().unwrap(),
                output.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (4, 4, 1, 2));
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn defectmap_uses_correct_defects_header_layout_on_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-defect-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let defects = base.with_extension("defects.txt");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 4, 4, 1, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0.0_f32; 16];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    std::fs::write(&defects, "CameraSizeX 4\nCameraSizeY 4\nBadColumns 1\n").unwrap();
    let result = common::imod_cmd("clip")
        .args([
            "defectmap",
            "-D",
            defects.to_str().unwrap(),
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!((header.nx, header.ny, header.nz, header.mode), (4, 4, 1, 0));
        assert_eq!(header.bytes_signed, 0);
        assert_eq!((header.amin, header.amax, header.amean), (0., 1., 0.25));
        let label = String::from_utf8_lossy(&header.labels[0]);
        assert!(
            label.starts_with("clip defectmap: Map of defective pixels in image"),
            "{label:?}"
        );
        let mut pixels = [0.0_f32; 16];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels[1], 1.);
        assert_eq!(pixels[5], 1.);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
    let _ = std::fs::remove_file(defects);
}

#[test]
fn supergain_rejects_a_real_non_eer_byte_mrc_with_source_diagnostic() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-supergain-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.txt");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 64, 64, 1, MRC_MODE_BYTE), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [4_f32; 4096];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args([
            "supergain",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "ERROR: clip supergain: all input files must be EER files.\n"
    );
    assert!(result.stderr.is_empty());
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

#[test]
fn histogram_prints_source_integer_bins_for_real_byte_mrc() {
    let input =
        std::env::temp_dir().join(format!("imod-rs-clip-histogram-{}.mrc", std::process::id()));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, MRC_MODE_BYTE), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 1., 2., 3.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let result = common::imod_cmd("clip")
        .args(["histogram", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        " Value   counts    (bin interval is 1)\n     1        2\n     2        1\n     3        1\n"
    );
    assert!(result.stderr.is_empty());
    let _ = std::fs::remove_file(input);
}

#[test]
fn histogram_post_bin_options_report_source_failures_for_real_mrc() {
    let input = std::env::temp_dir().join(format!(
        "imod-rs-clip-histogram-post-{}.mrc",
        std::process::id()
    ));
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, MRC_MODE_BYTE), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1_f32, 1., 2., 3.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let falloff = common::imod_cmd("clip")
        .args(["histogram", "-Falloff", "0.1;-1", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!falloff.status.success());
    assert!(String::from_utf8_lossy(&falloff.stdout)
        .contains("ERROR: CLIP - No point of maximum falloff could be found past 0.100 of total cumulative counts"));
    let extra = common::imod_cmd("clip")
        .args(["histogram", "-Extra", "0.5;-1", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!extra.status.success());
    assert!(
        String::from_utf8_lossy(&extra.stdout).contains("ERROR: CLIP - Peak is at"),
        "{}",
        String::from_utf8_lossy(&extra.stdout)
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn correlation_3d_real_mrc_reports_source_parabolic_peak_location() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-corr3d-{}", std::process::id()));
    let first = base.with_extension("first.mrc");
    let second = base.with_extension("second.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        for path in [&first, &second] {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 4, 4, 4, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            for section in 0..4 {
                let mut pixels = [0_f32; 16];
                if section == 1 {
                    pixels[1 + 2 * 4] = 1.;
                }
                assert_eq!(ii_write_section_float(&mut *file, &mut pixels, section), 0);
            }
            ii_close(file);
        }
    }
    let result = common::imod_cmd("clip")
        .args([
            "correlation",
            "-3d",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("location of max pixel ( "), "{stdout}");
    assert!(stdout.contains("( -0.00, -0.00, -0.00)"), "{stdout}");
    for path in [first, second, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn threshold_minimum_size_semicolon_sign_uses_source_fill_behavior() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-threshold-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    let input_c = CString::new(input.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open_new(input_c.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 2, 2, 1, 2), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [1.0_f32, 1., 0., 0.];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    assert!(
        common::imod_cmd("clip")
            .args([
                "threshold",
                "-t",
                "0.5",
                "-Minimum",
                "3;-1",
                input.to_str().unwrap(),
                output.to_str().unwrap(),
            ])
            .status()
            .unwrap()
            .success()
    );
    let output_c = CString::new(output.to_string_lossy().as_bytes()).unwrap();
    unsafe {
        let file = ii_open(output_c.to_bytes(), "rb");
        let mut pixels = [f32::NAN; 4];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        assert_eq!(pixels, [255., 255., 255., 255.]);
        ii_close(file);
    }
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
}

/// Builds a real multi-section MRC volume with structured content, so filters
/// that reach across Z produce different results at the volume edges than in
/// the middle.  `z_first` selects which section of the underlying pattern the
/// file starts at, which lets a short file hold the tail of a longer one.
fn write_edge_sensitive_volume(path: &std::path::Path, z_first: i32, nz: i32) {
    unsafe {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 8, 6, nz, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for z in 0..nz {
            let source_z = z_first + z;
            let mut pixels = [0_f32; 48];
            for y in 0..6 {
                for x in 0..8 {
                    // A Z ramp plus a per-pixel spike pattern: the median of a
                    // window that reaches past the last section differs from
                    // the median of one that stops there.
                    pixels[x + 8 * y] = (10 * source_z) as f32
                        + (x as f32) * 3.
                        + (y as f32)
                        + if (x + y + source_z as usize) % 5 == 0 {
                            40.
                        } else {
                            0.
                        };
                }
            }
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
        }
        ii_close(file);
    }
}

#[test]
fn median_three_dimensional_window_narrows_at_the_ends_of_the_volume() {
    // `processing.cpp:640-650` re-derives firstNeed from lastNeed only in the
    // kernel/smoothing branch.  The median branch leaves firstNeed at
    // secs[k] - size / 2, so at the last section of a five-section volume the
    // window covers sections 3..=4 rather than sliding back to a full 2..=4.
    // Filtering a two-section volume holding exactly those sections must
    // therefore produce the same last output section.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-median3d-{}", std::process::id()));
    let five = base.with_extension("five.mrc");
    let two = base.with_extension("two.mrc");
    let five_out = base.with_extension("five-out.mrc");
    let two_out = base.with_extension("two-out.mrc");
    for path in [&five_out, &two_out] {
        let _ = std::fs::remove_file(path);
    }
    write_edge_sensitive_volume(&five, 0, 5);
    write_edge_sensitive_volume(&two, 3, 2);

    for (input, output) in [(&five, &five_out), (&two, &two_out)] {
        let result = common::imod_cmd("clip")
            .args(["median", "-n", "3"])
            .arg(input)
            .arg(output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "status={:?} stdout={} stderr={}",
            result.status,
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }

    let read_section = |path: &std::path::Path, z: i32| -> Vec<f32> {
        let mut pixels = [0_f32; 48];
        unsafe {
            let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open(name.to_bytes(), "rb");
            assert_eq!(ii_read_section_float(&mut *file, &mut pixels, z), 0);
            ii_close(file);
        }
        pixels.to_vec()
    };

    // A window that wrongly reached back to section 2 would pull the Z ramp
    // down and differ from the two-section result here.
    assert_eq!(
        read_section(&five_out, 4),
        read_section(&two_out, 1),
        "the window at the last section must cover sections 3..=4 only"
    );
    for path in [&five, &two, &five_out, &two_out] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn validation_errors_use_the_source_exit_prefix_on_standard_output() {
    // `clip.cpp:211-212` installs "ERROR: clip - " as the PIP exit prefix and
    // the validation failures call exitError, so `PipSetError`
    // (`parse_params.c:1099-1102`) writes "<prefix> <message>" — two spaces
    // after the dash — to standard output, then exits 1.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-prefix-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let output = base.with_extension("out.mrc");
    write_edge_sensitive_volume(&input, 0, 2);
    for (args, message) in [
        (
            vec!["integral", "-t", "150"],
            "You must enter -n and either -l OR -h for integral process",
        ),
        (
            vec!["boxsd", "-n", "0.5"],
            "Reduction factor (-n) must be at least 1 for boxsd process",
        ),
        (vec!["brightness", "-zz", "1"], "Invalid option -zz."),
    ] {
        let result = common::imod_cmd("clip")
            .args(&args)
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert_eq!(
            String::from_utf8_lossy(&result.stdout),
            format!("ERROR: clip -  {message}\n")
        );
        assert!(result.stderr.is_empty());
    }
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn missing_arguments_print_the_source_banner_and_exit_three() {
    // `clip.cpp:215-218` calls usage() then exit(3), and usage()
    // (`clip.cpp:22-27`) prints the version banner with `printf` immediately
    // before `imodCopyright`, which also uses `printf`.
    let result = common::imod_cmd("clip").output().unwrap();
    assert_eq!(result.status.code(), Some(3));
    let text = String::from_utf8_lossy(&result.stdout);
    let mut lines = text.lines();
    let banner = lines.next().unwrap();
    assert!(
        banner.starts_with("clip: Command Line Image Processing. 5.2.17, "),
        "banner was {banner:?}"
    );
    // The trailing fields are C `__DATE__`/`__TIME__`; only their shape is
    // deterministic.  "Mmm dd yyyy HH:MM:SS", with a space-padded day.
    let stamp = banner.rsplit_once("5.2.17, ").unwrap().1;
    let (date, time) = stamp.rsplit_once(' ').unwrap();
    assert_eq!(date.len(), 11, "date field was {date:?}");
    assert_eq!(time.len(), 8, "time field was {time:?}");
    assert!(time.as_bytes()[2] == b':' && time.as_bytes()[5] == b':');
    // Copyright must be the very next line, not flushed out at exit behind the
    // rest of the usage text.
    assert_eq!(
        lines.next().unwrap(),
        "Copyright (C) 1994-2025 by the Regents of the University of Colorado"
    );
    assert_eq!(
        lines.next().unwrap(),
        "----------------------------------------------------"
    );
}

#[test]
fn opening_a_missing_input_reports_the_source_ii_open_diagnostic() {
    // `iimage.c:273-278` writes the iiOpen failure to stderr with the errno
    // text before clip's own exitError message goes to stdout.
    let missing =
        std::env::temp_dir().join(format!("imod-rs-clip-absent-{}.mrc", std::process::id()));
    let _ = std::fs::remove_file(&missing);
    let output =
        std::env::temp_dir().join(format!("imod-rs-clip-absent-{}.out", std::process::id()));
    let result = common::imod_cmd("clip")
        .arg("brightness")
        .arg(&missing)
        .arg(&output)
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        format!(
            "ERROR: iiOpen - Opening file {} (No such file or directory)\n",
            missing.display()
        )
    );
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        format!("ERROR: clip -  Error opening {}\n", missing.display())
    );
}

#[test]
fn writing_16_bit_floats_does_not_take_the_half_float_path_for_integer_modes() {
    // `mrcInitOutputHeader` sets hdata->halfFloats from write16BitModeForFloats()
    // for every mode, so `mrcsec.c:1147` gates the half-float conversion on the
    // header mode actually being MRC_MODE_FLOAT.  Using the raw flag sent byte,
    // short and ushort writes down the half-float path, where mode 0 emitted two
    // bytes per pixel into an nx-byte line buffer and aborted the process.
    // Only the integer modes are asserted here: those are the ones the raw
    // flag wrongly diverted.  The float mode legitimately writes mode 12 and
    // is covered by the native differential instead.
    for (mode, fill) in [
        (MRC_MODE_BYTE, 40.0_f32),
        (MRC_MODE_SHORT, 4000.0),
        (MRC_MODE_USHORT, 4000.0),
    ] {
        let base = std::env::temp_dir().join(format!(
            "imod-rs-clip-halffloat-{}-{}",
            std::process::id(),
            mode
        ));
        let input = base.with_extension("in.mrc");
        let output = base.with_extension("out.mrc");
        let _ = std::fs::remove_file(&output);
        unsafe {
            let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, 9, 4, 2, mode), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            for z in 0..2 {
                // An odd nx makes a doubled line length overrun the buffer.
                let mut pixels = [0_f32; 36];
                for (index, pixel) in pixels.iter_mut().enumerate() {
                    *pixel = fill + index as f32;
                }
                assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
            }
            ii_close(file);
        }

        let result = common::imod_cmd("clip")
            .env("IMOD_WRITE_FLOATS_16BIT", "1")
            .args(["brightness", "-n", "1"])
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "mode {mode} exited {:?}: {}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );

        // The written file must keep the input's own mode, not a half-float one.
        unsafe {
            let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
            let mut fp =
                imod_rs::imod::libcfshr::b3dutil::ImodFile::open(&output, "rb")
                    .unwrap();
            let mut written = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut fp, &mut written), 0);
            assert_eq!(written.mode, mode, "output mode for input mode {mode}");
            drop(fp);
        }
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}

// The three tests below pin last-ULP results of paths where the C computes in
// double and an earlier Rust translation computed in float.  They compare exact
// f32 bit patterns against values captured from native `clip`, so they fail on a
// one-ulp regression rather than tolerating it.

#[test]
fn bandpass_filter_matches_native_rounding_on_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-filter-ulp-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 12, 8, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 96];
        for (index, value) in pixels.iter_mut().enumerate() {
            *value = ((index * 37) % 251) as f32 / 8.0;
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let command = common::imod_cmd("clip")
        .args([
            "filter",
            "-l",
            "0.05",
            "-h",
            "0.25",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [f32::NAN; 96];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
        // Native `clip filter -l 0.05 -h 0.25`; `mrc_bandpass_filter` rounds the
        // Y frequency to float and applies `mval` as a double multiply.
        for (index, expected) in [
            (0_usize, -0.039264656603336334_f32),
            (11, -0.018901227042078972),
            (31, 0.01600455678999424),
            (41, -0.02449677512049675),
            (55, -0.02126469276845455),
            (74, 0.02821079082787037),
        ] {
            assert_eq!(
                pixels[index].to_bits(),
                expected.to_bits(),
                "pixel {index}: {} vs {expected}",
                pixels[index]
            );
        }
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn boxsd_matches_native_rounding_on_real_mrc() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-boxsd-ulp-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 12, 8, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 96];
        for (index, value) in pixels.iter_mut().enumerate() {
            *value = ((index * 37) % 251) as f32 / 8.0;
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let command = common::imod_cmd("clip")
        .args([
            "boxsd",
            "-n",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [f32::NAN; 96];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
        // Native `clip boxsd -n 1`; `makeStandardDevMap` finishes through
        // `sumsToAvgSD`, whose variance divide and `sqrt` are in double.
        for (index, expected) in [
            (4_usize, 8.989545822143555_f32),
            (16, 8.989545822143555),
            (28, 8.989545822143555),
            (40, 8.989545822143555),
            (54, 9.012887954711914),
            (78, 9.012887954711914),
        ] {
            assert_eq!(
                pixels[index].to_bits(),
                expected.to_bits(),
                "pixel {index}: {} vs {expected}",
                pixels[index]
            );
        }
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn flatfield_sum_matches_native_rounding_on_real_mrc() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-flatfield-ulp-{}", std::process::id()));
    let input = base.with_extension("input.mrc");
    let output = base.with_extension("output.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 12, 8, 3, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for section in 0..3 {
            let mut pixels = [0_f32; 96];
            for (index, value) in pixels.iter_mut().enumerate() {
                *value = (((section * 96 + index) * 83) % 251) as f32 / 3.0;
            }
            assert_eq!(
                ii_write_section_float(&mut *file, &mut pixels, section as i32),
                0
            );
        }
        ii_close(file);
    }
    let command = common::imod_cmd("clip")
        .args([
            "flatfield",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert_eq!(
        stdout,
        "clip: summing slices...\nAveraged image min = 21.333, max = 61.667, mean = 42.772\n"
    );
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [f32::NAN; 96];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
        // Native `clip flatfield`; `fullArrayMinMaxMean` rounds the mean to float
        // before `dmean / B3DMAX(0.05 * dmean, sumBuf[ix])` divides in double.
        for (index, expected) in [
            (0_usize, 1.2417676448822021_f32),
            (19, 1.4152498245239258),
            (35, 1.603949785232544),
            (46, 0.8207845687866211),
            (58, 0.8650515079498291),
            (73, 0.9275854229927063),
        ] {
            assert_eq!(
                pixels[index].to_bits(),
                expected.to_bits(),
                "pixel {index}: {} vs {expected}",
                pixels[index]
            );
        }
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// Writes a deterministic float MRC used by the fidelity regressions below.
fn write_audit_float_volume(path: &std::path::Path, nx: i32, ny: i32, nz: i32) {
    unsafe {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, nx, ny, nz, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for z in 0..nz {
            let mut pixels = vec![0_f32; (nx * ny) as usize];
            for y in 0..ny {
                for x in 0..nx {
                    pixels[(x + nx * y) as usize] =
                        100. + ((x * 37 + y * 11 + z * 53) % 97) as f32 + z as f32 * 7.;
                }
            }
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
        }
        ii_close(file);
    }
}

#[test]
fn forward_and_inverse_fft_warn_about_ignored_entries() {
    // `fft.cpp:35-36` and `fft.cpp:44-45` each emit a show_warning when the
    // ignored size/center/mode entries are present; both were absent.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-fftwarn-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let forward = base.with_extension("fwd.mrc");
    let inverse = base.with_extension("inv.mrc");
    write_audit_float_volume(&input, 16, 8, 2);
    let command = common::imod_cmd("clip")
        .args([
            "fft",
            "-2d",
            "-ox",
            "10",
            input.to_str().unwrap(),
            forward.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(
        stdout.starts_with("WARNING: clip forward fft - output sizes or mode are ignored\n"),
        "{stdout}"
    );
    let command = common::imod_cmd("clip")
        .args([
            "fft",
            "-2d",
            "-ix",
            "8",
            forward.to_str().unwrap(),
            inverse.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(
        stdout.starts_with("WARNING: clip inverse fft - input sizes or centers are ignored\n"),
        "{stdout}"
    );
    for path in [input, forward, inverse] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn odd_input_size_fft_prints_the_source_factor_diagnostic() {
    // `fft.cpp:69-71` (2-D) and `fft.cpp:203-204` (3-D) both print an error
    // that the translation had dropped, returning -1 silently.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-fftodd-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let output = base.with_extension("out.mrc");
    write_audit_float_volume(&input, 23, 8, 2);
    for (args, expected) in [
        (
            vec!["fft", "-2d"],
            "ERROR: clip - fft input size (23, 8) is odd and/or has factors greater than 19.\n",
        ),
        (
            vec!["fft"],
            "ERROR: clip - fft input size 23x8x2 is odd and/or has factors greater than 19.\n",
        ),
    ] {
        let mut all = args.clone();
        all.push(input.to_str().unwrap());
        all.push(output.to_str().unwrap());
        let command = common::imod_cmd("clip").args(&all).output().unwrap();
        assert!(!command.status.success(), "{command:?}");
        let stdout = String::from_utf8(command.stdout).unwrap();
        assert!(stdout.contains(expected), "{args:?}: {stdout}");
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn a_single_input_to_add_reports_the_process_error_not_the_usage_banner() {
    // `clip.cpp:794` only skips opening the output when needtwo is unmet; the
    // process itself then emits the diagnostic.  The translation had invented
    // a usage banner plus exit 3.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-needtwo-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let output = base.with_extension("out.mrc");
    write_audit_float_volume(&input, 8, 6, 2);
    for (process, expected) in [
        ("add", "ERROR: clip add: needs at least two input files.\n"),
        (
            "multiply",
            "ERROR: clip multiply/divide: Need exactly two input files.\n",
        ),
    ] {
        let command = common::imod_cmd("clip")
            .args([process, input.to_str().unwrap(), output.to_str().unwrap()])
            .output()
            .unwrap();
        assert_eq!(command.status.code(), Some(255), "{command:?}");
        assert_eq!(String::from_utf8(command.stdout).unwrap(), expected);
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn flatfield_mode_warning_has_no_program_prefix() {
    // `clip.cpp:421` passes the bare sentence to show_warning; the translation
    // had prepended "clip - ".
    let base = std::env::temp_dir().join(format!("imod-rs-clip-ffmode-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let output = base.with_extension("out.mrc");
    write_audit_float_volume(&input, 12, 8, 2);
    let command = common::imod_cmd("clip")
        .args([
            "flatfield",
            "-m",
            "byte",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(
        stdout.starts_with("WARNING: Output mode for a flatfield image must be floating point\n"),
        "{stdout}"
    );
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn option_values_follow_the_source_sscanf_atof_and_switch_shapes() {
    // `clip.cpp:388` uses sscanf("%f"), `:439` uses atof, and the switch keys
    // on argv[iarg][1] and [2] only.  The translation had used Rust's parse,
    // which rejects a trailing suffix, and exact-string option matching, which
    // rejected `-2foo`, `-oxy` and `-Ix`.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-optparse-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    write_audit_float_volume(&input, 16, 8, 2);
    for args in [
        vec!["brightness", "-n", "2x"],
        vec!["brightness", "-2foo", "-n", "2"],
        vec!["brightness", "-oxy", "12", "-n", "2"],
        vec!["resize", "-Ix", "8"],
        vec!["brightness", "-p", "abc", "-n", "2"],
        vec!["brightness", "-B", "2x", "-n", "2"],
        vec!["brightness", "-R", "3x", "-n", "2"],
    ] {
        let output = base.with_extension("out.mrc");
        let mut all = args.clone();
        all.push(input.to_str().unwrap());
        all.push(output.to_str().unwrap());
        let command = common::imod_cmd("clip").args(&all).output().unwrap();
        assert!(
            command.status.success(),
            "{args:?} rejected: {}",
            String::from_utf8_lossy(&command.stdout)
        );
        let _ = std::fs::remove_file(output);
    }
    // `-n 2x` must take the 2 that sscanf converts, so it is indistinguishable
    // from `-n 2` and distinguishable from `-n 3`.
    let suffixed = base.with_extension("suffixed.mrc");
    let plain = base.with_extension("plain.mrc");
    let other = base.with_extension("other.mrc");
    for (value, path) in [("2x", &suffixed), ("2", &plain), ("3", &other)] {
        let command = common::imod_cmd("clip")
            .args([
                "brightness",
                "-n",
                value,
                input.to_str().unwrap(),
                path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(command.status.success(), "{command:?}");
    }
    let suffixed_bytes = std::fs::read(&suffixed).unwrap();
    let plain_bytes = std::fs::read(&plain).unwrap();
    let other_bytes = std::fs::read(&other).unwrap();
    assert_eq!(suffixed_bytes[1024..], plain_bytes[1024..]);
    assert_ne!(suffixed_bytes[1024..], other_bytes[1024..]);
    for path in [input, suffixed, plain, other] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn a_negative_input_size_list_does_not_overflow_the_section_allocation() {
    // `file_io.cpp:113` multiplies sizeof(int) by a possibly negative nofsecs
    // as size_t, so malloc simply fails and the fill loop does not run.  The
    // translation had panicked on the multiply.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-negiz-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let output = base.with_extension("out.mrc");
    write_audit_float_volume(&input, 8, 6, 5);
    let command = common::imod_cmd("clip")
        .args([
            "resize",
            "-iz",
            "-1,3",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(command.status.code(), Some(0), "{command:?}");
    assert!(
        String::from_utf8(command.stderr).unwrap().is_empty(),
        "unexpected stderr"
    );
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn histogram_rejects_a_sub_unit_bin_size_with_the_source_message() {
    // `processing.cpp:3942-3947`: B3DNINT(opt->val) below one is an error the
    // translation had silently clamped to a bin size of one.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-histbin-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 8, 6, 1, MRC_MODE_BYTE), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [0_f32; 48];
        for (index, value) in pixels.iter_mut().enumerate() {
            *value = (index % 40) as f32 + 30.;
        }
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let command = common::imod_cmd("clip")
        .args(["histogram", "-n", "0.2", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(
        stdout.contains("ERROR: clip histogram - Entered bin size (0.200000) must be > 0.5\n"),
        "{stdout}"
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn stats_prints_the_overall_line_before_the_extreme_value_list() {
    // `processing.cpp:3752-3758` emits the " all " summary before the
    // "Slices with extreme values" block at `:3762`; the translation had them
    // the other way round.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-statorder-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    write_audit_float_volume(&input, 12, 8, 6);
    let command = common::imod_cmd("clip")
        .args(["stats", "-n", "2", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    let all = stdout.find(" all  ").expect("no overall line");
    let extremes = stdout
        .find("with extreme values:")
        .expect("no extreme list");
    assert!(all < extremes, "{stdout}");
    let _ = std::fs::remove_file(input);
}

#[test]
fn stats_outlier_length_never_exceeds_the_section_count() {
    // `processing.cpp:3548` is B3DMIN(nofsecs, B3DMAX(5, length)), which is
    // simply nofsecs when it is below five.  The translation had used
    // `i32::clamp(5, nofsecs)`, which panics when min > max.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-statclamp-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    write_audit_float_volume(&input, 8, 6, 3);
    let command = common::imod_cmd("clip")
        .args(["stats", "-n", "2", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(command.status.code(), Some(0), "{command:?}");
    assert!(
        !String::from_utf8_lossy(&command.stderr).contains("panicked"),
        "{command:?}"
    );
    let _ = std::fs::remove_file(input);
}

#[test]
fn logarithm_uses_the_single_precision_log_routine() {
    // `processing.cpp:291` calls log10() on a float and stores a float; the
    // reference build narrows that to log10f.  Rust's `f32::log10` evaluates
    // the double routine and rounds, which differs by one ulp on ~3% of
    // pixels.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-log10f-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let output = base.with_extension("out.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 4, 2, 1, MRC_MODE_FLOAT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [
            174.77246_f32,
            202.49753,
            192.21725,
            174.70349,
            95.711266,
            91.59483,
            132.5,
            240.125,
        ];
        assert_eq!(ii_write_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
    }
    let command = common::imod_cmd("clip")
        .args([
            "logarithm",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = [f32::NAN; 8];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
        // Native `clip logarithm` on the same eight values.
        for (index, expected) in [
            (0_usize, 2.2424731_f32),
            (1, 2.3064198),
            (2, 2.2837925),
            (3, 2.2423015),
            (4, 1.980963),
            (5, 1.9618709),
        ] {
            assert_eq!(
                pixels[index].to_bits(),
                expected.to_bits(),
                "pixel {index}: {} vs {expected}",
                pixels[index]
            );
        }
    }
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn a_full_bad_column_run_leaves_the_rows_the_source_never_reaches() {
    // `CorrectDefects.cpp:329-352`: CORRECT_THREE_FOUR_COL's leading row is a
    // single `if`, not a loop, so with ystart 0 the running indexes advance
    // only once before the fullStart..fullEnd loop and the top of the column
    // is left uncorrected.  The translation had rewritten it as one loop over
    // every row.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-badcol-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let output = base.with_extension("out.mrc");
    let defects = base.with_extension("defects.txt");
    write_audit_float_volume(&input, 32, 32, 1);
    std::fs::write(
        &defects,
        "CameraSizeX 32\nCameraSizeY 32\nBadColumns 12 13 14\n",
    )
    .unwrap();
    let command = common::imod_cmd("clip")
        .args([
            "brightness",
            "-n",
            "1",
            "-D",
            defects.to_str().unwrap(),
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(command.status.success(), "{command:?}");
    unsafe {
        let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open(name.to_bytes(), "rb");
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(
            mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        let mut pixels = vec![f32::NAN; 32 * 32];
        assert_eq!(ii_read_section_float(&mut *file, &mut pixels, 0), 0);
        ii_close(file);
        // The source stops writing after 1 + (fullEnd - fullStart + 1) + 1
        // rows, so the last rows of each bad column keep their input value.
        for y in 27..32 {
            for x in 12..15 {
                let expected = 100. + ((x * 37 + y * 11) % 97) as f32;
                assert_eq!(
                    pixels[(x + 32 * y) as usize],
                    expected,
                    "row {y} column {x} was rewritten"
                );
            }
        }
        // The rows the source does correct are replaced.
        assert_ne!(
            pixels[(13 + 32 * 10) as usize],
            100. + ((13 * 37 + 10 * 11) % 97) as f32
        );
    }
    for path in [input, output, defects] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn defect_file_parse_failures_report_the_source_single_argument_message() {
    // `clip.cpp:562` passes three arguments to a format holding a single %s,
    // so the file name never reaches the output.  The translation had
    // appended it.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-defmsg-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let output = base.with_extension("out.mrc");
    write_audit_float_volume(&input, 8, 6, 1);
    let missing = base.with_extension("nosuch.txt");
    let command = common::imod_cmd("clip")
        .args([
            "brightness",
            "-n",
            "1",
            "-D",
            missing.to_str().unwrap(),
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!command.status.success(), "{command:?}");
    let stdout = String::from_utf8(command.stdout).unwrap();
    assert!(stdout.contains("Error opening\n"), "{stdout}");
    assert!(!stdout.contains("nosuch.txt"), "{stdout}");
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn edge_fill_accepts_the_sizes_the_source_accepts() {
    // `CorrectDefects.cpp:2306` tests only the data mode; the translation had
    // added nx/ny/maxWidth range guards that rejected `clip edgefill -l 1`.
    let base = std::env::temp_dir().join(format!("imod-rs-clip-edgefill-{}", std::process::id()));
    let input = base.with_extension("in.mrc");
    let output = base.with_extension("out.mrc");
    unsafe {
        let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
        let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
        let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
        assert_eq!(mrc_head_new(header, 32, 24, 2, MRC_MODE_SHORT), 0);
        ii_sync_from_mrc_header(&mut *file, header);
        assert_eq!(
            mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
            0
        );
        for z in 0..2 {
            let mut pixels = vec![0_f32; 32 * 24];
            for y in 0..24 {
                for x in 0..32 {
                    pixels[(x + 32 * y) as usize] = (1000 + (x * 17 + y * 29 + z * 7) % 401) as f32;
                }
            }
            assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
        }
        ii_close(file);
    }
    let command = common::imod_cmd("clip")
        .args([
            "edgefill",
            "-l",
            "1",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(command.status.code(), Some(0), "{command:?}");
    for path in [input, output] {
        let _ = std::fs::remove_file(path);
    }
}

/// Cross-backend differential for `clip fft`, the caller that reaches the
/// complex-to-complex `odfft` directions and the negative `todfft` direction.
///
/// `clip_fftvol` (`clip/fft.cpp:258`) transforms each Z section with
/// `todfft(..., 0)` and then transforms along Z with `odfft(..., -1)`;
/// the inverse runs `odfft(..., -2)` and `todfft(..., -1)` — the direction
/// documented at `todfft.c:55` as the inverse and the one no other fixture
/// exercises.  `-2d` takes the `slice_fft` path (`clip/fft.cpp:111`) instead.
/// Intended backends: each case runs once on `parity` and once on `rustfft`,
/// in separate processes because the selector is a process-local `OnceLock`.
///
/// Sizes are chosen from what `clip_nicesize` admits -- even in X with no
/// prime factor above 19 -- and deliberately avoid powers of two, so that
/// across the cases the radix-3, radix-5 and general prime-factor kernels each
/// run: 38 is 2 * 19 in X, 7 is prime in Z, and no axis is the same length as
/// another.
///
/// Tolerances: measured, then rounded up, and expressed relative to the
/// largest magnitude in the buffer being compared, because a forward spectrum
/// of this fixture peaks near 8.5e3 while the volume it came from peaks near
/// 1.5e2.  Measured over the four cases, the parity-versus-RustFFT difference
/// was at most 1.3e-7 of that magnitude in a forward transform and 7.5e-7 in
/// the volume the inverse recovers, with RMS differences of 6.4e-9 and 2.4e-7;
/// the bounds are 8.0e-6 and 2.5e-6 of the magnitude, about ten times the
/// worst case.  A sign, normalization or axis error fails them by orders of
/// magnitude.
#[cfg(feature = "rustfft-backend")]
#[test]
fn fft_rustfft_transforms_match_the_parity_volume() {
    let base =
        std::env::temp_dir().join(format!("imod-rs-clip-fft-rustfft-{}", std::process::id()));
    for (case, dimension, nx, ny, nz) in [
        ("three", "-3d", 30, 18, 12),
        ("threeprime", "-3d", 38, 5, 7),
        ("two", "-2d", 100, 7, 2),
        ("twoodd", "-2d", 54, 45, 1),
    ] {
        let input = base.with_extension(format!("{case}.input.mrc"));
        unsafe {
            let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, nx, ny, nz, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            for z in 0..nz {
                let mut pixels = (0..nx * ny)
                    .map(|index| {
                        let (x, y) = ((index % nx) as f32, (index / nx) as f32);
                        100.0 + 30.0 * (x / 3.0).sin() * (y / 2.0 + z as f32).cos() + x - y
                    })
                    .collect::<Vec<f32>>();
                assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
            }
            ii_close(file);
        }
        let mut transformed = Vec::new();
        let mut recovered = Vec::new();
        let mut geometry = Vec::new();
        for backend in ["parity", "rustfft"] {
            let forward = base.with_extension(format!("{case}.{backend}.fft.mrc"));
            let inverse = base.with_extension(format!("{case}.{backend}.back.mrc"));
            for (source, destination) in [(&input, &forward), (&forward, &inverse)] {
                let result = common::imod_cmd("clip")
                    .env("IMOD_RS_FFT_BACKEND", backend)
                    .args([
                        "fft",
                        dimension,
                        source.to_str().unwrap(),
                        destination.to_str().unwrap(),
                    ])
                    .output()
                    .unwrap();
                assert!(
                    result.status.success(),
                    "{case} {backend} {}: {}",
                    destination.display(),
                    String::from_utf8_lossy(&result.stderr)
                );
            }
            for (path, into) in [(&forward, &mut transformed), (&inverse, &mut recovered)] {
                unsafe {
                    let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
                    let mut file = imod_rs::imod::libcfshr::b3dutil::ImodFile::open(
                        &path,
                        "rb",
                    )
                    .unwrap();
                    let mut header = MrcHeader::default();
                    assert_eq!(mrc_head_read(&mut file, &mut header), 0);
                    let count = (header.nx * header.ny * header.nz) as usize
                        * if header.mode == MRC_MODE_COMPLEX_FLOAT {
                            2
                        } else {
                            1
                        };
                    let mut values = vec![0.0_f32; count];
                    imod_rs::imod::libcfshr::b3dutil::b3d_fseek(
                        &mut file,
                        header.header_size as i32,
                        imod_rs::imod::libcfshr::b3dutil::SEEK_SET,
                    );
                    assert_eq!(
                        imod_rs::imod::libcfshr::b3dutil::b3d_fread(
                            unsafe {
                                core::slice::from_raw_parts_mut(
                                    values.as_mut_ptr().cast::<u8>(),
                                    4 * (count),
                                )
                            },
                            4,
                            count,
                            &mut file,
                        ),
                        count
                    );
                    drop(file);
                    geometry.push((header.nx, header.ny, header.nz, header.mode));
                    into.push(values);
                }
                let _ = std::fs::remove_file(path);
            }
        }
        let _ = std::fs::remove_file(input);
        assert_eq!(geometry[0], geometry[2], "{case} forward geometry");
        assert_eq!(geometry[1], geometry[3], "{case} inverse geometry");
        for (stage, values) in [("forward", &transformed), ("inverse", &recovered)] {
            let mut maximum = 0.0_f64;
            let mut sum_squares = 0.0_f64;
            let mut scale = 0.0_f64;
            assert_eq!(values[0].len(), values[1].len(), "{case} {stage} length");
            for (parity, rustfft) in values[0].iter().zip(values[1].iter()) {
                let difference = (*parity as f64 - *rustfft as f64).abs();
                maximum = maximum.max(difference);
                sum_squares += difference * difference;
                scale = scale.max((*parity as f64).abs());
            }
            let rms = (sum_squares / values[0].len() as f64).sqrt();
            eprintln!("clip fft {case} {stage}: max {maximum:.3e} rms {rms:.3e} scale {scale:.3e}");
            assert!(
                maximum < 8.0e-6 * scale,
                "clip fft {case} {stage} maximum difference {maximum} against scale {scale}"
            );
            assert!(
                rms < 2.5e-6 * scale,
                "clip fft {case} {stage} RMS difference {rms} against scale {scale}"
            );
        }
    }
}

/// Cross-backend differential for the remaining `clip` processes that reach an
/// FFT: `filter` (`slice_fft` per section, `clip/filter.cpp:103`),
/// `correlation` in both its 2-D form (`mrc_to_dfft` forward, `corr_conj`,
/// inverse, `clip/correlation.cpp:104-133`) and its 3-D form (`clip_fftvol`,
/// so the complex-to-complex `odfft` directions again), and `spectrum`
/// (`spectrumScaled`, which is handed `todfft` itself).  With `clip fft`
/// covered separately this is every Fourier operation the command has.
///
/// Intended backends: one `parity` process and one `rustfft` process per case.
/// The 30 by 18 by 12 fixture pads to 60 by 36 by 24 in correlation, so no
/// dimension anywhere in this test is a power of two.
///
/// Tolerances: measured, then rounded up.  Relative to the largest magnitude
/// in each output volume, the measured parity-versus-RustFFT difference was
/// 3.0e-7 maximum for `filter`, 1.5e-7 for the 2-D correlation and 8.5e-7 for
/// the 3-D one, with RMS differences of 8.4e-8, 6.8e-8 and 2.9e-7; the bounds
/// here are 1.0e-5 and 2.5e-6 of that magnitude, roughly ten times the worst
/// case.
///
/// `spectrum` needs a looser, separately derived bound and gets an absolute
/// one, because it is not a transform comparison: `spectrumScaled` maps
/// `scale * ln(logScale * value + 1)` onto 0 to 32000
/// (`spectrumscaled.c:209,229`), and that logarithm's slope is steepest where
/// the power is smallest, so the same `f32` round-off that is 1e-7 relative in
/// a coefficient becomes several counts at the dark end of the image.  The
/// measured difference was 13 counts of 32000, and every difference above two
/// counts sat at the bottom of the value range; the bound is 40 counts with an
/// RMS of 2.  Note that a power spectrum discards the sign of the transform,
/// so this case cannot detect a conjugation error -- `newstack -phase` and the
/// round trips in `fft_backend_matrix` are what cover that.
#[cfg(feature = "rustfft-backend")]
#[test]
fn fourier_processes_rustfft_outputs_match_the_parity_outputs() {
    let base = std::env::temp_dir().join(format!("imod-rs-clip-fourier-{}", std::process::id()));
    for (case, nz, options, relative, relative_rms, counts, counts_rms) in [
        (
            "filter",
            2,
            vec!["filter", "-l", "1", "-h", "0"],
            1.0e-5,
            2.5e-6,
            0.0,
            0.0,
        ),
        (
            "correlation2d",
            1,
            vec!["correlation", "-2d"],
            1.0e-5,
            2.5e-6,
            0.0,
            0.0,
        ),
        (
            "correlation3d",
            12,
            vec!["correlation"],
            1.0e-5,
            2.5e-6,
            0.0,
            0.0,
        ),
        (
            "spectrum",
            2,
            vec!["spectrum", "-l", "0"],
            0.0,
            0.0,
            40.0,
            2.0,
        ),
    ] {
        let (nx, ny) = (30, 18);
        let input = base.with_extension(format!("{case}.input.mrc"));
        unsafe {
            let name = CString::new(input.to_string_lossy().as_bytes()).unwrap();
            let file = ii_open_new(name.to_bytes(), "wb", IIFILE_DEFAULT);
            let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
            assert_eq!(mrc_head_new(header, nx, ny, nz, MRC_MODE_FLOAT), 0);
            ii_sync_from_mrc_header(&mut *file, header);
            assert_eq!(
                mrc_head_write((&mut (*file).fp).as_mut().unwrap(), header),
                0
            );
            for z in 0..nz {
                let mut pixels = (0..nx * ny)
                    .map(|index| {
                        let (x, y) = ((index % nx) as f32, (index / nx) as f32);
                        100.0 + 30.0 * (x / 3.0).sin() * (y / 2.0 + z as f32).cos() + x - y
                    })
                    .collect::<Vec<f32>>();
                assert_eq!(ii_write_section_float(&mut *file, &mut pixels, z), 0);
            }
            ii_close(file);
        }
        let mut decoded = Vec::new();
        let mut geometry = Vec::new();
        for backend in ["parity", "rustfft"] {
            let output = base.with_extension(format!("{case}.{backend}.mrc"));
            let result = common::imod_cmd("clip")
                .env("IMOD_RS_FFT_BACKEND", backend)
                .args(&options)
                .args([input.to_str().unwrap(), output.to_str().unwrap()])
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{case} {backend}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            unsafe {
                let name = CString::new(output.to_string_lossy().as_bytes()).unwrap();
                let file = ii_open(name.to_bytes(), "rb");
                let (nx, ny, nz, mode) = {
                    let header = (*file).mrc_header.as_deref_mut().expect("MRC header");
                    assert_eq!(
                        mrc_head_read((&mut (*file).fp).as_mut().unwrap(), header),
                        0
                    );
                    (header.nx, header.ny, header.nz, header.mode)
                };
                let mut values = vec![0.0_f32; (nx * ny * nz) as usize];
                let section = (nx * ny) as usize;
                for z in 0..nz {
                    assert_eq!(
                        ii_read_section_float(&mut *file, &mut values[z as usize * section..], z),
                        0
                    );
                }
                geometry.push((nx, ny, nz, mode));
                decoded.push(values);
                ii_close(file);
            }
            let _ = std::fs::remove_file(output);
        }
        let _ = std::fs::remove_file(input);
        assert_eq!(geometry[0], geometry[1], "{case} output geometry");
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
        eprintln!("clip {case}: max {maximum:.3e} rms {rms:.3e} scale {scale:.3e}");
        assert!(
            maximum <= relative * scale + counts,
            "clip {case} maximum difference {maximum} against scale {scale}"
        );
        assert!(
            rms <= relative_rms * scale + counts_rms,
            "clip {case} RMS difference {rms} against scale {scale}"
        );
    }
}
