use imod_rs::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write,
};
use std::ffi::CString;
use std::process::Command;

#[test]
fn binvol_missing_input_uses_the_source_error_exit() {
    let result = Command::new(env!("CARGO_BIN_EXE_binvol")).output().unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&result.stderr),
        "ERROR: BINVOL - No input file specified\n"
    );
}

#[test]
fn binvol_bins_an_mrc_stack_and_preserves_transferred_metadata() {
    unsafe {
        let stamp = format!("imod-rs-binvol-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 4, 4, 2, MRC_MODE_FLOAT), 0);
        header.xlen = 8.;
        header.ylen = 12.;
        header.zlen = 20.;
        header.xorg = 3.;
        header.yorg = -2.;
        header.zorg = 7.;
        header.amin = 1.;
        header.amax = 32.;
        header.amean = 16.5;
        header.nlabl = 1;
        header.labels[0][..24].copy_from_slice(b"acquisition source label");
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels: Vec<f32> = (1..=32).map(|value| value as f32).collect();
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 4, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);

        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .arg("-input")
            .arg(&input)
            .arg("-output")
            .arg(&output)
            .arg("-binning")
            .arg("2")
            .arg("-antialias")
            .arg("0")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );

        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut written: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written), 0);
        assert_eq!(
            (written.nx, written.ny, written.nz, written.mode),
            (2, 2, 1, MRC_MODE_FLOAT)
        );
        assert_eq!((written.xlen, written.ylen, written.zlen), (8., 12., 20.));
        assert_eq!((written.xorg, written.yorg, written.zorg), (3., -2., 7.));
        assert_eq!(written.nlabl, 2);
        assert_eq!(&written.labels[0][..24], b"acquisition source label");
        let label = String::from_utf8_lossy(&written.labels[1][..56]);
        assert!(
            label.starts_with("BINVOL: Volume binned down by factors   2   2   2"),
            "{label:?}"
        );
        assert_eq!(written.labels[1][56 + 2], b'-');
        assert_eq!(written.labels[1][56 + 6], b'-');
        assert_eq!(written.labels[1][67 + 2], b':');
        assert_eq!(written.labels[1][67 + 5], b':');
        let mut binned = [0_f32; 4];
        assert_eq!(
            libc::fseek(file, written.header_size as i64, libc::SEEK_SET),
            0
        );
        assert_eq!(
            libc::fread(binned.as_mut_ptr().cast(), 4, binned.len(), file),
            binned.len()
        );
        libc::fclose(file);
        assert_eq!(binned, [11.5, 13.5, 19.5, 21.5]);
        assert_eq!(
            (written.amin, written.amax, written.amean),
            (11.5, 21.5, 16.5)
        );
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn binvol_spread_keeps_the_source_sampled_extent_and_origin_shift() {
    unsafe {
        let stamp = format!("imod-rs-binvol-spread-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut header: MrcHeader = core::mem::zeroed();
        // The native `iiuReadReduced` source path requires enough rows for
        // its ten-line filter chunk; use a real, source-valid small stack.
        assert_eq!(mrc_head_new(&mut header, 2, 64, 3, MRC_MODE_FLOAT), 0);
        header.xlen = 4.;
        header.ylen = 128.;
        header.zlen = 6.;
        header.zorg = 10.;
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels: Vec<f32> = (1..=384).map(|value| value as f32).collect();
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 4, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);

        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .arg("-input")
            .arg(&input)
            .arg("-output")
            .arg(&output)
            .arg("-binning")
            .arg("2")
            .arg("-spread")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );

        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut written: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written), 0);
        libc::fclose(file);
        assert_eq!((written.nx, written.ny, written.nz), (1, 32, 2));
        assert_eq!((written.xlen, written.ylen, written.zlen), (4., 128., 8.));
        assert_eq!(written.zorg, 11.);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn binvol_applies_source_xy_antialias_through_unit_reduced() {
    unsafe {
        let stamp = format!("imod-rs-binvol-xy-filter-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 32, 32, 1, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let mut pixels = [0_f32; 1024];
        for iy in 0..32 {
            for ix in 0..32 {
                pixels[ix + iy * 32] = if (ix + iy) % 2 == 0 { 100. } else { 0. };
            }
        }
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 4, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-binning",
                "2",
                "-antialias",
                "2",
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .contains(" Antialiasing is being applied in X and Y as well as Z\n")
        );
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        let mut written: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written), 0);
        assert_eq!((written.nx, written.ny, written.nz), (16, 16, 1));
        let mut values = [f32::NAN; 256];
        libc::fseek(file, written.header_size as i64, libc::SEEK_SET);
        assert_eq!(
            libc::fread(values.as_mut_ptr().cast(), 4, values.len(), file),
            values.len()
        );
        libc::fclose(file);
        assert!(values.iter().all(|value| value.is_finite()));
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn binvol_uses_source_strip_fallback_at_one_megabyte_limit() {
    unsafe {
        let stamp = format!("imod-rs-binvol-strip-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 1024, 1024, 2, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = vec![4.0_f32; 1024 * 1024 * 2];
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 4, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-binning",
                "2",
                "-antialias",
                "0",
                "-memory",
                "2",
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        let mut written: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written), 0);
        assert_eq!((written.nx, written.ny, written.nz), (512, 512, 1));
        let mut value = 0_f32;
        libc::fseek(file, written.header_size as i64, libc::SEEK_SET);
        assert_eq!(libc::fread((&mut value as *mut f32).cast(), 4, 1, file), 1);
        libc::fclose(file);
        assert_eq!(value, 4.0);
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn binvol_runs_source_fourier_reduce_and_expand_on_real_mrc_volumes() {
    unsafe {
        let stamp = format!("imod-rs-binvol-fourier-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let reduced = std::env::temp_dir().join(format!("{stamp}-reduced.mrc"));
        let expanded = std::env::temp_dir().join(format!("{stamp}-expanded.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 16, 16, 16, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        header.amean = 3.0;
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = vec![3.0_f32; 16 * 16 * 16];
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 4, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        for (flag, output, dimensions) in [
            ("-ftreduce", &reduced, (8, 8, 8)),
            ("-ftexpand", &expanded, (32, 32, 32)),
        ] {
            let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
                .args([
                    "-input",
                    input.to_str().unwrap(),
                    "-output",
                    output.to_str().unwrap(),
                    "-binning",
                    "2",
                    flag,
                ])
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
            let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
            let mut written: MrcHeader = core::mem::zeroed();
            assert_eq!(mrc_head_read(file, &mut written), 0);
            assert_eq!((written.nx, written.ny, written.nz), dimensions);
            assert!(
                written.amin.is_finite() && written.amax.is_finite() && written.amean.is_finite()
            );
            libc::fclose(file);
        }
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(reduced).unwrap();
        std::fs::remove_file(expanded).unwrap();
    }
}

#[test]
fn binvol_accepts_source_permitted_noninteger_fourier_binning() {
    unsafe {
        let stamp = format!("imod-rs-binvol-fourier-ratio-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let input_c = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let output_c = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(input_c.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 16, 16, 16, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels = vec![2.0_f32; 16 * 16 * 16];
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 4, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-binning",
                "1.5",
                "-ftreduce",
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let file = libc::fopen(output_c.as_ptr(), c"rb".as_ptr());
        let mut written: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written), 0);
        libc::fclose(file);
        assert_eq!((written.nx, written.ny, written.nz), (10, 10, 10));
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn binvol_rejects_simultaneous_fourier_directions_on_real_mrc() {
    unsafe {
        let stamp = format!("imod-rs-binvol-fourier-conflict-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-ftreduce",
                "-ftexpand",
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert_eq!(
            String::from_utf8_lossy(&result.stderr),
            "ERROR: BINVOL - You cannot enter both -ftReduce and -ftExpand\n"
        );
        assert!(!output.exists());
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn binvol_rejects_spread_with_fourier_on_real_mrc() {
    unsafe {
        let stamp = format!("imod-rs-binvol-spread-fourier-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-ftreduce",
                "-spread",
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert_eq!(
            String::from_utf8_lossy(&result.stderr),
            "ERROR: BINVOL - You cannot enter -antialias or -spread with Fourier operations\n"
        );
        assert!(!output.exists());
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn binvol_rejects_noninteger_real_mrc_reduction_without_filter() {
    unsafe {
        let stamp = format!("imod-rs-binvol-noninteger-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 4, 4, 4, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-binning",
                "1.5",
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert_eq!(
            String::from_utf8_lossy(&result.stderr),
            "ERROR: BINVOL - Reduction factors must all be integers unless antialias filtering or a Fourier operation is specified\n"
        );
        assert!(!output.exists());
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn binvol_rejects_unequal_noninteger_xy_real_mrc_reduction() {
    unsafe {
        let stamp = format!("imod-rs-binvol-xy-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 6, 6, 2, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-xbinning",
                "1.5",
                "-ybinning",
                "2.5",
                "-antialias",
                "2",
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert_eq!(
            String::from_utf8_lossy(&result.stderr),
            "ERROR: BINVOL - Reduction in X and Y must be equal unless they are both integers\n"
        );
        assert!(!output.exists());
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn binvol_rejects_invalid_output_mode_on_real_mrc() {
    unsafe {
        let stamp = format!("imod-rs-binvol-mode-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-mode",
                "3",
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert_eq!(
            String::from_utf8_lossy(&result.stderr),
            "ERROR: BINVOL - The mode of the input or output must be 0, 1, 2, 6 or 12\n"
        );
        assert!(!output.exists());
        std::fs::remove_file(input).unwrap();
    }
}

#[test]
fn binvol_rejects_source_invalid_antialias_filter_on_real_mrc() {
    unsafe {
        let stamp = format!("imod-rs-binvol-antialias-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        let name = CString::new(input.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-antialias",
                "1",
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert_eq!(
            String::from_utf8_lossy(&result.stderr),
            "ERROR: BINVOL - Antialias filter type must be between 2 and 6, or < 0 for default\n"
        );
        assert!(!output.exists());
        std::fs::remove_file(input).unwrap();
    }
}
