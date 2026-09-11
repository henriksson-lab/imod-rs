use imod_rs::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write,
};
use std::ffi::CString;
use std::process::Command;

#[test]
fn binvol_without_two_arguments_prints_source_help_and_exits_zero() {
    // `PipReadOrParseOptions(..., .false., 2, 1, 1, ...)` (`binvol.f90:67`)
    // prints the autodoc help and calls `exit(0)` when fewer than `minArgs`
    // entries were given; verified against the native binary, which prints the
    // same block for no arguments and for `-input file` alone.
    let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&result.stderr), "");
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.starts_with("Usage: binvol [Options] input_file output_file\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains(" -binning (-b)  OR  -BinningFactor   Float\n"),
        "{stdout}"
    );
    assert!(
        stdout.ends_with(" -help (-h)  OR  -usage\n    Print help output\n"),
        "{stdout}"
    );
}

#[test]
fn binvol_reports_no_input_file_when_only_options_name_the_output() {
    // With two entries present PIP reaches `PipGetInOutFile('InputFile', ...)`
    // (`binvol.f90:69`), whose failure path is `exitError`, which writes to
    // standard output after a blank record.
    let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(["-output", "out.mrc", "-binning", "2"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&result.stderr), "");
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "\nERROR: BINVOL - No input file specified\n"
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .arg("-input")
            .arg(&input)
            .arg("-output")
            .arg(&output)
            .arg("-binning")
            .arg("2")
            .arg("-antialias")
            .arg("0")
            .arg("-memory")
            .arg("1000")
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .arg("-input")
            .arg(&input)
            .arg("-output")
            .arg(&output)
            .arg("-binning")
            .arg("2")
            .arg("-spread")
            // `maxLines` is `integer*4` (`binvol.f90:16`) while its expression
            // is `real*4`; without a pinned limit that product overflows on a
            // large-memory machine and native takes its "too large" exit.
            .arg("-memory")
            .arg("1000")
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
        // `binvol.f90:202` adds `delta(3) * extraPix / 2` after the loop at
        // `binvol.f90:172` has already scaled `delta` by the reduction factor,
        // so the shift is 4 * 1 / 2.  Native writes 12 for this stack.
        assert_eq!(written.zorg, 12.);
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
        // Two sections: with `nz == 1` the source forces `binZ = 1` but leaves
        // `ibinZ` at 2 (`binvol.f90:138` does not touch `iredFac`), so
        // `inputEnds(0)` is 1, no section ever completes, and native writes a
        // header-only file.  Two sections exercise the real filtering path.
        assert_eq!(mrc_head_new(&mut header, 32, 32, 2, MRC_MODE_FLOAT), 0);
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let mut pixels = [0_f32; 2048];
        for iz in 0..2 {
            for iy in 0..32 {
                for ix in 0..32 {
                    pixels[ix + iy * 32 + iz * 1024] =
                        if (ix + iy + iz) % 2 == 0 { 100. } else { 0. };
                }
            }
        }
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 4, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-binning",
                "2",
                "-antialias",
                "2",
                "-memory",
                "1000",
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
        // Native (`-memory 1000 -binning 2 -antialias 2` on this stack) writes
        // this min/max/mean triple; the filter mixes the checkerboard down to
        // a near-uniform 123.
        assert_eq!(
            (written.amin, written.amax, written.amean),
            (122.983383, 123.06146, 123.02244)
        );
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
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
                .env(
                    "AUTODOC_DIR",
                    concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
                )
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
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
        // `exitError` (`parse_input_params.f90:231`) writes
        // `write(*,'(/,a,a,a)')` to standard output, not standard error.
        assert_eq!(String::from_utf8_lossy(&result.stderr), "");
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .ends_with("\nERROR: BINVOL - You cannot enter both -ftReduce and -ftExpand\n"),
            "{}",
            String::from_utf8_lossy(&result.stdout)
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
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
        // A Fourier operation sets `ifiltType = 0` (`binvol.f90:89`), so the
        // spread check at `binvol.f90:111` fires before the combined check at
        // `binvol.f90:113`.  Native emits this message, not the combined one.
        assert_eq!(String::from_utf8_lossy(&result.stderr), "");
        assert!(
            String::from_utf8_lossy(&result.stdout).ends_with(
                "\nERROR: BINVOL - Spreading in Z can be used only with antialias filtering\n"
            ),
            "{}",
            String::from_utf8_lossy(&result.stdout)
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args([
                "-input",
                input.to_str().unwrap(),
                "-output",
                output.to_str().unwrap(),
                "-binning",
                "1.5",
                // `ifiltType` falls back to `ifiltDefault` = 6 at
                // `binvol.f90:108`, so the integer requirement is only
                // reachable with the filter explicitly turned off.
                "-antialias",
                "0",
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        // `exitError` (`parse_input_params.f90:231`) writes
        // `write(*,'(/,a,a,a)')` to standard output, not standard error.
        assert_eq!(String::from_utf8_lossy(&result.stderr), "");
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .ends_with("\nERROR: BINVOL - Reduction factors must all be integers unless antialias filtering or a Fourier operation is specified\n"),
            "{}",
            String::from_utf8_lossy(&result.stdout)
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
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
        // `exitError` (`parse_input_params.f90:231`) writes
        // `write(*,'(/,a,a,a)')` to standard output, not standard error.
        assert_eq!(String::from_utf8_lossy(&result.stderr), "");
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .ends_with("\nERROR: BINVOL - Reduction in X and Y must be equal unless they are both integers\n"),
            "{}",
            String::from_utf8_lossy(&result.stdout)
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
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
        // `exitError` (`parse_input_params.f90:231`) writes
        // `write(*,'(/,a,a,a)')` to standard output, not standard error.
        assert_eq!(String::from_utf8_lossy(&result.stderr), "");
        assert!(
            String::from_utf8_lossy(&result.stdout).ends_with(
                "\nERROR: BINVOL - The mode of the input or output must be 0, 1, 2, 6 or 12\n"
            ),
            "{}",
            String::from_utf8_lossy(&result.stdout)
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
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
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
        // `exitError` (`parse_input_params.f90:231`) writes
        // `write(*,'(/,a,a,a)')` to standard output, not standard error.
        assert_eq!(String::from_utf8_lossy(&result.stderr), "");
        assert!(
            String::from_utf8_lossy(&result.stdout)
                .ends_with("\nERROR: BINVOL - Antialias filter type must be between 2 and 6, or < 0 for default\n"),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        assert!(!output.exists());
        std::fs::remove_file(input).unwrap();
    }
}

/// Builds the 4x4x2 mode-2 stack used by the option-parsing regressions below.
/// Native runs on this same stack supplied every expectation asserted here.
fn write_tiny_float_stack(path: &std::path::Path) {
    unsafe {
        let name = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"wb".as_ptr());
        assert!(!file.is_null());
        let mut header: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_new(&mut header, 4, 4, 2, MRC_MODE_FLOAT), 0);
        header.amin = 1.;
        header.amax = 32.;
        header.amean = 16.5;
        header.fp = file.cast();
        assert_eq!(mrc_head_write(file, &mut header), 0);
        let pixels: Vec<f32> = (1..=32).map(|value| value as f32).collect();
        assert_eq!(
            libc::fwrite(pixels.as_ptr().cast(), 4, pixels.len(), file),
            pixels.len()
        );
        libc::fclose(file);
    }
}

#[test]
fn binvol_takes_the_first_value_of_a_comma_separated_binning_entry() {
    // `BinningFactor` is a single-value `F` option (`binvol.adoc`), and PIP
    // parses `2,2,1` as one float.  Native accepts `-bin 2,2,1` and produces
    // exactly the `-binning 2` result; the previous hand-rolled parser
    // rejected the abbreviated option outright.
    unsafe {
        let stamp = format!("imod-rs-binvol-comma-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        write_tiny_float_stack(&input);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args([
                "-memory",
                "1000",
                "-bin",
                "2,2,1",
                "-antialias",
                "0",
                input.to_str().unwrap(),
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(0));
        assert_eq!(String::from_utf8_lossy(&result.stderr), "");
        let name = CString::new(output.as_os_str().as_encoded_bytes()).unwrap();
        let file = libc::fopen(name.as_ptr(), c"rb".as_ptr());
        assert!(!file.is_null());
        let mut written: MrcHeader = core::mem::zeroed();
        assert_eq!(mrc_head_read(file, &mut written), 0);
        assert_eq!((written.nx, written.ny, written.nz), (2, 2, 1));
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
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn binvol_opens_the_output_unit_before_the_antialias_notice() {
    // `imopen(3, outFile, 'NEW')` is `binvol.f90:145` and the antialias notice
    // is `binvol.f90:179`.  The previous module printed the notice first.
    unsafe {
        let stamp = format!("imod-rs-binvol-order-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        write_tiny_float_stack(&input);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args([
                "-memory",
                "1000",
                "-binning",
                "2",
                "-antialias",
                "2",
                input.to_str().unwrap(),
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        // Native on this 4x4 stack cannot feed `irdReduced`'s filter chunk and
        // exits at label 99; the two lines of interest are printed first and
        // in this order.
        assert_eq!(result.status.code(), Some(1));
        let stdout = String::from_utf8_lossy(&result.stdout);
        let new_unit = stdout.find(" NEW image file on unit   3 : ").unwrap();
        let notice = stdout
            .find(" Antialiasing is being applied in X and Y as well as Z\n")
            .unwrap();
        assert!(new_unit < notice, "{stdout}");
        assert!(
            stdout.ends_with("\nERROR: BINVOL - Reading image\n"),
            "{stdout}"
        );
        std::fs::remove_file(input).unwrap();
        std::fs::remove_file(output).unwrap();
    }
}

#[test]
fn binvol_reports_a_zero_reduction_factor_after_reading_the_header() {
    // `binvol.f90:152` is inside the per-axis loop that runs after `irdhdr`
    // and after `imopen(3, ..., 'NEW')`, so native has already printed the
    // input report and the new-unit line when it exits.
    unsafe {
        let stamp = format!("imod-rs-binvol-zero-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stamp}.mrc"));
        let output = std::env::temp_dir().join(format!("{stamp}-out.mrc"));
        write_tiny_float_stack(&input);
        let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args([
                "-binning",
                "0",
                input.to_str().unwrap(),
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert_eq!(String::from_utf8_lossy(&result.stderr), "");
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            stdout.contains(" Number of columns, rows, sections ....."),
            "{stdout}"
        );
        assert!(
            stdout.contains(" NEW image file on unit   3 : "),
            "{stdout}"
        );
        assert!(
            stdout
                .ends_with("\nERROR: BINVOL - Reduction or expansion factor must be at least 1\n"),
            "{stdout}"
        );
        std::fs::remove_file(input).unwrap();
        let _ = std::fs::remove_file(output);
    }
}

#[test]
fn binvol_help_option_prints_the_autodoc_block_and_exits_zero() {
    // `PipReadOrParseOptions` calls `PipPrintHelp(progName, 0, 1, 1)` then
    // `exit(0)` (`parse_input_params.f90:163`); the block comes from
    // `IMOD/autodoc/binvol.adoc`, which includes `-shifts`, an option missing
    // from the in-program fallback table.
    let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .arg("-help")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&result.stderr), "");
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.starts_with("Usage: binvol [Options] input_file output_file\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains(" -shifts (-sh)  OR  -ShiftsInXYZ   3 floats\n"),
        "{stdout}"
    );
}

#[test]
fn binvol_reports_the_backend_read_diagnostic_before_its_own() {
    // A read past the end of the file prints `mrcsec.c:367`'s diagnostic once
    // per attempt and only then `binvol`'s `exitError`.  `b3dError` writes
    // through libc stdout, which is block buffered under a pipe, while
    // `exitError` (`parse_input_params.f90:236`) writes to Fortran unit 6; the
    // two interleave in program order in the reference, so the backend lines
    // come first.  Expectation captured from the native `binvol` on this input
    // with stdout taken through a pipe.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-binvol-short-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let whole = std::fs::read(root.join("fixtures/newstack-mixed-byte.mrc")).unwrap();
    std::fs::write(dir.join("tr.mrc"), &whole[..1200]).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_binvol"))
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .env("IMOD_NO_IMAGE_BACKUP", "1")
        .args(["-binning", "2", "-antialias", "0", "tr.mrc", "ob.mrc"])
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&result.stderr), "");
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.ends_with(
            "\n NEW image file on unit   3 : ob.mrc\n\
             ERROR: mrcReadSectionAny - reading data from file.\n\
             \n\
             ERROR: mrcReadSectionAny - reading data from file.\n\
             \n\
             \n\
             ERROR: BINVOL - Reading image\n"
        ),
        "{stdout}"
    );
    assert_eq!(std::fs::metadata(dir.join("ob.mrc")).unwrap().len(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}
