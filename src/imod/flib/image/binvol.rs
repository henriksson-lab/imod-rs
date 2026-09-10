//! Translation of `IMOD/flib/image/binvol.f90`.
//!
//! The Fortran main program maps to [`binvol`] and its one contained procedure
//! maps to [`slice_weighting`]; no non-source algorithm helpers are introduced.
#![allow(dead_code, unused_variables)]

use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::iiu_open_print;
use crate::imod::libcfshr::b3dutil::{b3d_physical_memory, standard_memory_limit_mb};
use crate::imod::libcfshr::filtxcorr::{
    fourier_crop_sizes, fourier_expand_volume, fourier_reduce_volume,
};
use crate::imod::libcfshr::reduce_by_binning::{bin_into_slice, irepak};
use crate::imod::libcfshr::taperpad::slice_taper_out_pad;
use crate::imod::libcfshr::zoomdown::{select_zoom_filter, zoom_filt_value};
use crate::imod::libfft::{nice_fft_limit, thrdfft};
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MrcHeader};
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_mrc_header, iiu_read_lines, iiu_read_section, iiu_set_position,
    iiu_sync_with_mrc_header, iiu_write_lines, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_delta, iiu_alt_mode, iiu_alt_origin, iiu_alt_sample, iiu_alt_size, iiu_ret_delta,
    iiu_ret_origin, iiu_trans_header, iiu_write_header, iiu_write_header_str,
};
use crate::imod::libiimod::unit_reduced::iiu_read_reduced;

/// Original program `binvol` (`binvol.f90:5`).
///
/// This source program maps its real-space and Fourier volume paths directly
/// to the corresponding translated IMOD library units.
pub fn binvol() {
    let mut big_file = String::new();
    let mut out_file = String::new();
    let mut red_all = 2.0_f32;
    let mut bin_x = None;
    let mut bin_y = None;
    let mut bin_z = None;
    let mut output_mode = None;
    let mut ifilt_type = -1_i32;
    let mut entered_antialias = false;
    let mut spread_z = false;
    let mut ft_crop = false;
    let mut ft_expand = false;
    let mut x_shift = 0.0_f32;
    let mut y_shift = 0.0_f32;
    let mut z_shift = 0.0_f32;
    let mut memory_limit = None;
    let mut positional = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument
            .trim_start_matches('-')
            .to_ascii_lowercase()
            .as_str()
        {
            "input" | "inputfile" => big_file = args.next().unwrap_or_default(),
            "output" | "outputfile" => out_file = args.next().unwrap_or_default(),
            "binning" | "binningfactor" => {
                red_all = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0)
            }
            "xbinning" | "xbinningfactor" => {
                bin_x = Some(args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0))
            }
            "ybinning" | "ybinningfactor" => {
                bin_y = Some(args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0))
            }
            "zbinning" | "zbinningfactor" => {
                bin_z = Some(args.next().and_then(|v| v.parse().ok()).unwrap_or(0.0))
            }
            "mode" | "modetooutput" => {
                output_mode = Some(args.next().and_then(|v| v.parse().ok()).unwrap_or(-1))
            }
            "antialias" | "antialiaszfilter" => {
                entered_antialias = true;
                ifilt_type = args.next().and_then(|v| v.parse().ok()).unwrap_or(-2)
            }
            "spread" | "spreadslicesinz" => spread_z = true,
            "ftreduce" | "fourierreducebybinning" => ft_crop = true,
            "ftexpand" | "fourierexpandbybinning" => ft_expand = true,
            "shifts" | "shiftsinxyz" => {
                let values = args.next().unwrap_or_default();
                let mut values = values.split(',');
                let parsed = (
                    values
                        .next()
                        .and_then(|value| value.trim().parse::<f32>().ok()),
                    values
                        .next()
                        .and_then(|value| value.trim().parse::<f32>().ok()),
                    values
                        .next()
                        .and_then(|value| value.trim().parse::<f32>().ok()),
                );
                match parsed {
                    (Some(x), Some(y), Some(z)) if values.next().is_none() => {
                        x_shift = x;
                        y_shift = y;
                        z_shift = z;
                    }
                    _ => {
                        eprintln!("ERROR: BINVOL - Invalid value for ShiftsInXYZ");
                        std::process::exit(1);
                    }
                }
            }
            "memory" | "memorylimit" => {
                memory_limit = Some(args.next().and_then(|value| value.parse::<i32>().ok()));
            }
            "verbose" | "verboseoutput" => {
                let _ = args.next();
            }
            "help" | "usage" => {
                println!(
                    "Usage: binvol -input INPUT -output OUTPUT [-binning FACTOR] [-xbinning X] [-ybinning Y] [-zbinning Z] [-mode MODE]"
                );
                return;
            }
            _ if argument.starts_with('-') => {
                eprintln!("ERROR: BINVOL - Unknown option: {argument}");
                std::process::exit(1);
            }
            _ => positional.push(argument),
        }
    }
    if big_file.is_empty() && !positional.is_empty() {
        big_file = positional.remove(0);
    }
    if out_file.is_empty() && !positional.is_empty() {
        out_file = positional.remove(0);
    }
    if big_file.is_empty() {
        eprintln!("ERROR: BINVOL - No input file specified");
        std::process::exit(1);
    }
    if out_file.is_empty() {
        eprintln!("ERROR: BINVOL - No output file specified");
        std::process::exit(1);
    }
    if ft_expand && ft_crop {
        eprintln!("ERROR: BINVOL - You cannot enter both -ftReduce and -ftExpand");
        std::process::exit(1);
    }
    let physical_memory = b3d_physical_memory() / (1024_i64 * 1024_i64) as f64;
    let mut maximum_memory = 8000_i32;
    if physical_memory > 0.0 {
        let standard_limit = standard_memory_limit_mb(30000);
        maximum_memory = standard_limit.max(0.8 * physical_memory) as i32;
    }
    if let Some(limit) = memory_limit {
        match limit {
            Some(value) if (1..=maximum_memory).contains(&value) => {}
            _ => {
                eprintln!("ERROR: BINVOL - Memory limit must be between 1 and {maximum_memory}");
                std::process::exit(1);
            }
        }
    }
    let limit = memory_limit.flatten().unwrap_or_else(|| {
        if physical_memory > 0.0 {
            standard_memory_limit_mb(30000) as i32
        } else {
            1000
        }
    });
    // This is deliberately measured in single-precision pixels, exactly as
    // `limDim = limit * 1024 * 256` in the source (MB / four bytes).
    let lim_dim = i64::from(limit) * 1024 * 256;
    // The Fortran source assigns zero before its post-parse `PipGetInteger`:
    // absent `-antialias` therefore stays zero in a Fourier operation, while
    // an entered option is retained and rejected below.
    if (ft_expand || ft_crop) && !entered_antialias {
        ifilt_type = 0;
    }
    if (ft_expand || ft_crop) && (spread_z || entered_antialias) {
        eprintln!("ERROR: BINVOL - You cannot enter -antialias or -spread with Fourier operations");
        std::process::exit(1);
    }
    if ifilt_type == 1 || ifilt_type > 6 {
        eprintln!(
            "ERROR: BINVOL - Antialias filter type must be between 2 and 6, or < 0 for default"
        );
        std::process::exit(1);
    }
    let bin_x = bin_x.unwrap_or(red_all);
    let mut bin_y = bin_y.unwrap_or(red_all);
    let mut bin_z = bin_z.unwrap_or(red_all);
    let ibin_x = bin_x.round() as i32;
    let mut ibin_y = bin_y.round() as i32;
    let mut ibin_z = bin_z.round() as i32;
    if bin_x < 1.0 || bin_y < 1.0 || bin_z < 1.0 {
        eprintln!("ERROR: BINVOL - Reduction or expansion factor must be at least 1");
        std::process::exit(1);
    }
    if ((bin_x - ibin_x as f32).abs() > 0.001
        || (bin_y - ibin_y as f32).abs() > 0.001
        || (bin_z - ibin_z as f32).abs() > 0.001)
        && ifilt_type <= 0
        && !(ft_crop || ft_expand)
    {
        eprintln!(
            "ERROR: BINVOL - Reduction factors must all be integers unless antialias filtering or a Fourier operation is specified"
        );
        std::process::exit(1);
    }
    let unequal_xy = (bin_x - bin_y).abs() > 0.001;
    if unequal_xy
        && ((bin_x - ibin_x as f32).abs() > 0.001 || (bin_y - ibin_y as f32).abs() > 0.001)
    {
        if ft_crop || ft_expand {
            eprintln!("ERROR: BINVOL - Scaling in X and Y must be equal for Fourier operations");
        } else {
            eprintln!(
                "ERROR: BINVOL - Reduction in X and Y must be equal unless they are both integers"
            );
        }
        std::process::exit(1);
    }
    if !unequal_xy {
        bin_y = bin_x;
        ibin_y = ibin_x;
    }
    if ifilt_type < 0 {
        ifilt_type = 6;
    }
    if spread_z && ifilt_type <= 0 {
        eprintln!("ERROR: BINVOL - Spreading in Z can be used only with antialias filtering");
        std::process::exit(1);
    }
    let mut lines_filt = 0_i32;
    if ifilt_type > 1
        && bin_z > 1.0
        && unsafe {
            select_zoom_filter(ifilt_type - 1, 1.0 / f64::from(bin_z), &raw mut lines_filt)
        } != 0
    {
        eprintln!("ERROR: BINVOL - Setting up the antialias filter");
        std::process::exit(1);
    }
    if unsafe { iiu_open_print(1, &big_file, "RO") } != 0 {
        eprintln!("ERROR: BINVOL - Opening input file");
        std::process::exit(1);
    }
    let mut input_header = unsafe { std::mem::zeroed::<MrcHeader>() };
    unsafe {
        irdhdr(
            1,
            &mut input_header.nx,
            &mut input_header.mx,
            &mut input_header.mode,
            &mut input_header.amin,
            &mut input_header.amax,
            &mut input_header.amean,
        );
    }
    let input_mrc = unsafe { iiu_mrc_header(1, c"binvol".as_ptr(), 0, 0) };
    if input_mrc.is_null() {
        eprintln!("ERROR: BINVOL - Reading image header");
        unsafe { iiu_close(1) };
        std::process::exit(1);
    }
    unsafe {
        core::ptr::copy_nonoverlapping(input_mrc, &mut input_header, 1);
    }
    if input_header.nz == 1 {
        bin_z = 1.0;
        ibin_z = 1;
    }
    let mode = output_mode.unwrap_or(input_header.mode);
    if !matches!(mode, 0 | 1 | 2 | 6 | 12) {
        eprintln!("ERROR: BINVOL - The mode of the input or output must be 0, 1, 2, 6 or 12");
        unsafe { iiu_close(1) };
        std::process::exit(1);
    }
    // The source divides by the real reduction factor here.  `iredFac` is
    // used only for ordinary integral binning; using it for all output sizes
    // loses the permitted non-integral antialias-filter geometry.
    let non_integer_z = (bin_z - ibin_z as f32).abs() > 0.001;
    let mut nx_bin = (input_header.nx as f32 / bin_x) as i32;
    let mut ny_bin = (input_header.ny as f32 / bin_y) as i32;
    let mut nz_bin = (input_header.nz as f32 / bin_z) as i32;
    let mut nf_spad = [0_i32; 3];
    let mut ncrop_pad = [0_i32; 3];
    let mut actual_factor = [bin_x, bin_y, bin_z];
    if ft_crop || ft_expand {
        for (index, size) in [input_header.nx, input_header.ny, input_header.nz]
            .into_iter()
            .enumerate()
        {
            let base = if ft_expand {
                1.0 / [bin_x, bin_y, bin_z][index]
            } else {
                [bin_x, bin_y, bin_z][index]
            };
            let mut actual = 0.0;
            if unsafe {
                fourier_crop_sizes(
                    size,
                    base,
                    0.01,
                    16,
                    nice_fft_limit(),
                    &mut nf_spad[index],
                    &mut ncrop_pad[index],
                    &mut actual,
                )
            } != 0
            {
                eprintln!(
                    "ERROR: BINVOL - Reduction or expansion factor must be an integer or integer divided by 2, 3, 4, 5, 6, 8, or 10 to 3 decimal places"
                );
                unsafe {
                    iiu_close(1);
                }
                std::process::exit(1);
            }
            let output = (size as f32 / actual) as i32;
            actual_factor[index] = actual;
            match index {
                0 => nx_bin = output,
                1 => ny_bin = output,
                _ => nz_bin = output,
            }
        }
    }
    if nx_bin < 1 || ny_bin < 1 || nz_bin < 1 {
        eprintln!("ERROR: BINVOL - Reduction factor too large");
        unsafe { iiu_close(1) };
        std::process::exit(1);
    }
    let antialias_z = ifilt_type > 1 && bin_z > 1.0;
    let antialias_xy = ifilt_type > 1 && !unequal_xy && bin_x > 1.0;
    if antialias_xy {
        // `binvol.f90:177-181`: this notice is emitted whenever the source
        // applies its XY filtering, independently of the Z filtering state.
        println!(" Antialiasing is being applied in X and Y as well as Z");
    }
    let mut z_center_offset = bin_z / 2.0;
    let mut output_z_origin = input_header.zorg;
    if antialias_z && spread_z && bin_z * input_header.nz as f32 - nz_bin as f32 > 0.49 {
        nz_bin += 1;
        let extra_pixels = nz_bin as f32 * bin_z - input_header.nz as f32;
        z_center_offset -= extra_pixels / 2.0;
        // `iiuRetDelta` followed by `iiuAltOrigin` in the source.  This MRC
        // path has the same delta representation in the copied header.
        output_z_origin += input_header.zlen / input_header.mz as f32 * extra_pixels / 2.0;
    }
    if unsafe { iiu_open_print(3, &out_file, "NEW") } != 0 {
        eprintln!("ERROR: BINVOL - Opening output file");
        unsafe { iiu_close(1) };
        std::process::exit(1);
    }
    // `iiuTransHeader` plus its `iiuAlt*` calls: copy then synchronize the MRC header.
    if unsafe { iiu_trans_header(3, 1) } != 0 {
        eprintln!("ERROR: BINVOL - Transferring input header");
        unsafe {
            iiu_close(3);
            iiu_close(1);
        }
        std::process::exit(1);
    }
    let output_header = unsafe { iiu_mrc_header(3, c"binvol".as_ptr(), 0, 2) };
    unsafe {
        let mut nxyz_bin = [nx_bin, ny_bin, nz_bin];
        let mut nxyz_start = [0_i32; 3];
        let mut delta = [0_f32; 3];
        let (mut x_origin, mut y_origin, mut z_origin) = (0_f32, 0_f32, 0_f32);
        iiu_alt_mode(3, mode);
        iiu_alt_size(3, nxyz_bin.as_mut_ptr(), nxyz_start.as_mut_ptr());
        iiu_alt_sample(3, nxyz_bin.as_mut_ptr());
        iiu_ret_delta(1, delta.as_mut_ptr());
        delta[0] *= actual_factor[0];
        delta[1] *= actual_factor[1];
        delta[2] *= actual_factor[2];
        iiu_alt_delta(3, delta.as_mut_ptr());
        if antialias_z && spread_z {
            iiu_ret_origin(1, &mut x_origin, &mut y_origin, &mut z_origin);
            iiu_alt_origin(3, x_origin, y_origin, output_z_origin);
        }
        iiu_sync_with_mrc_header(3);
    }
    let mut initial_label = [0_i32; 20];
    if unsafe {
        iiu_write_header(
            3,
            initial_label.as_mut_ptr(),
            -1,
            input_header.amin,
            input_header.amax,
            input_header.amean,
        )
    } != 0
    {
        eprintln!("ERROR: BINVOL - Writing output header");
        unsafe {
            iiu_close(3);
            iiu_close(1);
        }
        std::process::exit(1);
    }
    let nx = input_header.nx as usize;
    let ny = input_header.ny as usize;

    if ft_crop || ft_expand {
        let idim_in =
            i64::from(nf_spad[0] + 2) * i64::from(nf_spad[1]) * i64::from(nf_spad[2]) + 16;
        let idim_out =
            i64::from(ncrop_pad[0] + 2) * i64::from(ncrop_pad[1]) * i64::from(ncrop_pad[2]) + 16;
        let max_xdim = (nf_spad[0] + 2).max(ncrop_pad[0] + 2);
        let idim_work = i64::from(max_xdim) * i64::from(nf_spad[2].max(ncrop_pad[2])) + 16;
        if idim_in + idim_out + idim_work > lim_dim {
            eprintln!(
                "ERROR: BINVOL - Full volume input plus output array sizes are bigger than the memory limit"
            );
            unsafe {
                iiu_close(3);
                iiu_close(1);
            }
            std::process::exit(1);
        }
        let mut fft_in = vec![0.0_f32; idim_in as usize];
        let mut fft_out = vec![0.0_f32; idim_out as usize];
        let mut fft_work = vec![0.0_f32; idim_work as usize];
        let nx_dim = nf_spad[0] + 2;
        let iz_high = input_header.nz - 1;
        let iz_start = -((nf_spad[2] - input_header.nz) / 2);
        let iz_end = iz_start + nf_spad[2] - 1;
        for iz in iz_start..=iz_end {
            let iz_read = iz.clamp(0, iz_high);
            let base =
                (i64::from(nx_dim) * i64::from(nf_spad[1]) * i64::from(iz - iz_start)) as usize;
            unsafe {
                iiu_set_position(1, iz_read, 0);
            }
            if unsafe { iiu_read_section(1, fft_in[base..].as_mut_ptr().cast()) } != 0 {
                eprintln!("ERROR: BINVOL - Reading image file");
                unsafe {
                    iiu_close(3);
                    iiu_close(1);
                }
                std::process::exit(1);
            }
            unsafe {
                slice_taper_out_pad(
                    fft_in[base..].as_mut_ptr().cast(),
                    2,
                    input_header.nx,
                    input_header.ny,
                    fft_in[base..].as_mut_ptr(),
                    nx_dim,
                    nf_spad[0],
                    nf_spad[1],
                    1,
                    input_header.amean,
                );
            }
            if iz < 0 || iz > iz_high {
                let atten = if iz < 0 {
                    (iz - iz_start) as f32 / (0 - iz_start) as f32
                } else {
                    (iz_end - iz) as f32 / (iz_end - iz_high) as f32
                };
                let pad_mean = (1.0 - atten) * input_header.amean;
                for iy in 0..nf_spad[1] {
                    let row = base + (iy * nx_dim) as usize;
                    for value in &mut fft_in[row..row + nf_spad[0] as usize] {
                        *value = pad_mean + atten * *value;
                    }
                }
            }
        }
        unsafe {
            let (mut xin, mut yin, mut zin, mut forward) = (nf_spad[0], nf_spad[1], nf_spad[2], 0);
            thrdfft(
                fft_in.as_mut_ptr(),
                fft_work.as_mut_ptr(),
                &mut xin,
                &mut yin,
                &mut zin,
                &mut forward,
            );
        }
        let mut xyz_shift = [0.0_f32; 3];
        for index in 0..3 {
            let input = [input_header.nx, input_header.ny, input_header.nz][index];
            let output = [nx_bin, ny_bin, nz_bin][index];
            let factor = [bin_x, bin_y, bin_z][index];
            if ft_expand {
                if input % 2 != 0 {
                    xyz_shift[index] = 0.5;
                }
            } else {
                if output % 2 != 0 {
                    xyz_shift[index] = -factor * 0.5;
                }
                xyz_shift[index] += (input % factor.round() as i32 + 1) as f32 / 2.0;
            }
        }
        unsafe {
            if ft_crop {
                fourier_reduce_volume(
                    fft_in.as_mut_ptr(),
                    nf_spad[0],
                    nf_spad[1],
                    nf_spad[2],
                    fft_out.as_mut_ptr(),
                    ncrop_pad[0],
                    ncrop_pad[1],
                    ncrop_pad[2],
                    xyz_shift[0] + x_shift,
                    xyz_shift[1] + y_shift,
                    xyz_shift[2] + z_shift,
                    fft_work.as_mut_ptr(),
                );
            } else {
                fourier_expand_volume(
                    fft_in.as_mut_ptr(),
                    nf_spad[0],
                    nf_spad[1],
                    nf_spad[2],
                    fft_out.as_mut_ptr(),
                    ncrop_pad[0],
                    ncrop_pad[1],
                    ncrop_pad[2],
                    xyz_shift[0] + x_shift,
                    xyz_shift[1] + y_shift,
                    xyz_shift[2] + z_shift,
                    fft_work.as_mut_ptr(),
                );
            }
            let (mut xout, mut yout, mut zout, mut inverse) =
                (ncrop_pad[0], ncrop_pad[1], ncrop_pad[2], -1);
            thrdfft(
                fft_out.as_mut_ptr(),
                fft_work.as_mut_ptr(),
                &mut xout,
                &mut yout,
                &mut zout,
                &mut inverse,
            );
        }
        let ix_start = (ncrop_pad[0] - nx_bin) / 2;
        let iy_start = (ncrop_pad[1] - ny_bin) / 2;
        let iz_start = -((ncrop_pad[2] - nz_bin) / 2);
        let nx_dim = ncrop_pad[0] + 2;
        let mut dmin = 1.0e30_f32;
        let mut dmax = -1.0e30_f32;
        let mut mean_sum = 0.0_f64;
        for iz in 0..nz_bin {
            let base =
                (i64::from(nx_dim) * i64::from(ncrop_pad[1]) * i64::from(iz - iz_start)) as usize;
            unsafe {
                irepak(
                    fft_out[base..].as_mut_ptr().cast(),
                    fft_out[base..].as_mut_ptr().cast(),
                    &nx_dim,
                    &ncrop_pad[1],
                    &ix_start,
                    &(ix_start + nx_bin - 1),
                    &iy_start,
                    &(iy_start + ny_bin - 1),
                );
            }
            let section = &fft_out[base..base + (nx_bin * ny_bin) as usize];
            let section_min = section.iter().copied().fold(f32::INFINITY, f32::min);
            let section_max = section.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let section_mean =
                section.iter().map(|value| f64::from(*value)).sum::<f64>() / section.len() as f64;
            dmin = dmin.min(section_min);
            dmax = dmax.max(section_max);
            mean_sum += section_mean;
            if unsafe { iiu_write_section(3, fft_out[base..].as_mut_ptr().cast()) } != 0 {
                eprintln!("ERROR: BINVOL - Writing image");
                unsafe {
                    iiu_close(3);
                    iiu_close(1);
                };
                std::process::exit(1);
            }
        }
        let dmean = (mean_sum / nz_bin as f64) as f32;
        unsafe {
            let mut title = [b' '; MRC_LABEL_SIZE + 1];
            let action = if ft_crop { "reduced" } else { "expanded" };
            let prefix = format!(
                "BINVOL: Volume Fourier {action} by{:7.2}{:7.2}{:7.2}",
                bin_x, bin_y, bin_z
            );
            title[..prefix.len().min(MRC_LABEL_SIZE)]
                .copy_from_slice(&prefix.as_bytes()[..prefix.len().min(MRC_LABEL_SIZE)]);
            let mut date = [b' '; 9];
            b3d_date(&mut date);
            title[56..65].copy_from_slice(&date);
            let mut now = 0_i64;
            let mut local = core::mem::zeroed::<libc::tm>();
            libc::time(&raw mut now);
            libc::localtime_r(&raw const now, &raw mut local);
            let time = format!(
                "{:02}:{:02}:{:02}",
                local.tm_hour, local.tm_min, local.tm_sec
            );
            title[67..75].copy_from_slice(time.as_bytes());
            let mut title_c = [0_i8; MRC_LABEL_SIZE + 1];
            core::ptr::copy_nonoverlapping(
                title.as_ptr().cast(),
                title_c.as_mut_ptr(),
                MRC_LABEL_SIZE,
            );
            if iiu_write_header_str(3, title_c.as_ptr(), 1, dmin, dmax, dmean) != 0 {
                eprintln!("ERROR: BINVOL - Writing output header (final)");
                iiu_close(3);
                iiu_close(1);
                std::process::exit(1);
            }
            iiu_close(3);
            iiu_close(1);
        }
        println!(" PROGRAM EXECUTED TO END.");
        return;
    }

    // REAL-SPACE REDUCTION.  Keep the source's two storage plans distinct:
    // a ring of complete output sections when one input section fans out to
    // few enough outputs, and a sequential strip fallback otherwise.
    let mut input_starts = vec![0_i32; nz_bin as usize + 1];
    let mut input_ends = vec![0_i32; nz_bin as usize + 1];
    let mut iring_pos = vec![0_i32; nz_bin as usize + 1];
    let mut num_slice_used_in = 1_i32;
    if antialias_z {
        for iz_out in 0..nz_bin {
            let z_cen = iz_out as f32 * bin_z + z_center_offset;
            let iz_cen = (z_cen - 0.5).round() as i32;
            let mut num_z_before = -1_i32;
            let mut num_z_after = 0_i32;
            for i in -lines_filt..=lines_filt {
                let z_weight =
                    unsafe { zoom_filt_value(i as f32 + iz_cen as f32 + 0.5 - z_cen) as f32 };
                if num_z_before < 0 && z_weight != 0.0 {
                    num_z_before = -i;
                }
                if z_weight != 0.0 {
                    num_z_after = i;
                }
            }
            input_starts[iz_out as usize] = (iz_cen - num_z_before).max(0);
            input_ends[iz_out as usize] = (iz_cen + num_z_after).min(input_header.nz - 1);
        }
        // `numSlice` is deliberately outside the input-Z loop, as in the
        // Fortran source; it is the source's conservative ring sizing.
        let mut num_slice = 0_i32;
        for inz in 0..=input_header.nz {
            for iz_out in 0..nz_bin {
                if input_starts[iz_out as usize] <= inz && input_ends[iz_out as usize] >= inz {
                    num_slice += 1;
                }
                if input_ends[iz_out as usize] < inz {
                    break;
                }
            }
            num_slice_used_in = num_slice_used_in.max(num_slice);
        }
    } else {
        z_center_offset = 0.0;
        for iz_out in 0..nz_bin {
            input_starts[iz_out as usize] = iz_out * ibin_z;
            input_ends[iz_out as usize] = iz_out * ibin_z + ibin_z - 1;
        }
    }
    let nx_bin_into_slice = if antialias_xy {
        nx_bin
    } else {
        input_header.nx
    };
    let (ired_x_into_slice, ired_y_into_slice) = if antialias_xy {
        (1, 1)
    } else {
        (ibin_x, ibin_y)
    };
    let nx_equiv = input_header.nx as i64
        + if antialias_xy {
            ((nx_bin as f32 + bin_y - 0.9) / bin_y) as i64
        } else {
            0
        };
    let ceiling_bin_y = bin_y.ceil() as i64;
    let mut full_out_slices = false;
    let mut in_base = 0_i64;
    let mut max_lines = 0_i64;
    if i64::from(nx_bin) * i64::from(ny_bin) * i64::from(num_slice_used_in) < lim_dim {
        in_base = i64::from(nx_bin) * i64::from(ny_bin) * i64::from(num_slice_used_in);
        max_lines = i64::from(ibin_y) * ((lim_dim - in_base) / nx_equiv / ceiling_bin_y);
        full_out_slices = max_lines > 1;
        max_lines = max_lines.min(input_header.ny as i64);
    }
    if !full_out_slices {
        max_lines = i64::from(ibin_y)
            * (lim_dim
                / (nx_equiv + ((nx_bin as f32 + bin_y - 0.9) / bin_y) as i64)
                / ceiling_bin_y
                - 1);
        if max_lines < 2 {
            eprintln!(
                "ERROR: BINVOL - Input images too large with given reduction factors and memory limit"
            );
            unsafe {
                iiu_close(3);
                iiu_close(1);
            }
            std::process::exit(1);
        }
        max_lines = max_lines.min(input_header.ny as i64);
        in_base = i64::from(nx_bin) * (max_lines / i64::from(ibin_y));
    }
    let itemp_base = (nx_equiv - input_header.nx as i64) * max_lines + in_base;
    let mut array = vec![0.0_f32; (nx_equiv * max_lines + in_base) as usize];
    let num_chunks = (input_header.ny as i64 + max_lines - 1) / max_lines;
    let mut dmin = 1.0e30_f32;
    let mut dmax = -1.0e30_f32;
    let mut sum = 0.0_f64;
    if full_out_slices {
        let mut iring_start = 0_i32;
        let mut iring_end = 0_i32;
        let mut last_z_finished = -1_i32;
        for inz in 0..input_header.nz {
            let mut iy = 0_i64;
            let mut iy_binned = 0_i64;
            let mut last_z_added = last_z_finished;
            for i_chunk in 0..num_chunks {
                let num_lines = max_lines.min(input_header.ny as i64 - i_chunk * max_lines);
                let num_bin_lines = num_lines / i64::from(ibin_y);
                let input_at = in_base as usize;
                let read_error = if antialias_xy {
                    let mut error = 0;
                    unsafe {
                        iiu_read_reduced(
                            1,
                            inz,
                            array[input_at..].as_mut_ptr(),
                            nx_bin,
                            0.0,
                            iy as f32,
                            bin_x,
                            nx_bin,
                            num_bin_lines as i32,
                            ifilt_type - 1,
                            array[itemp_base as usize..].as_mut_ptr(),
                            (nx as i64 * max_lines) as i32,
                            &mut error,
                        );
                    }
                    error
                } else {
                    unsafe {
                        iiu_set_position(1, inz, iy as i32);
                        iiu_read_lines(1, array[input_at..].as_mut_ptr().cast(), num_lines as i32)
                    }
                };
                if read_error != 0 {
                    eprintln!("ERROR: BINVOL - Reading image");
                    unsafe {
                        iiu_close(3);
                        iiu_close(1);
                    };
                    std::process::exit(1);
                }
                for iz_out in last_z_finished + 1..nz_bin {
                    if inz > input_ends[iz_out as usize] {
                        break;
                    }
                    if inz >= input_starts[iz_out as usize] {
                        if iring_pos[iz_out as usize] == 0 {
                            iring_end += 1;
                            if iring_end > num_slice_used_in {
                                iring_end = 1;
                            }
                            if iring_end == iring_start {
                                eprintln!("ERROR: BINVOL - Problem with ring buffer");
                                unsafe {
                                    iiu_close(3);
                                    iiu_close(1);
                                };
                                std::process::exit(1);
                            }
                            let output_at =
                                (iring_end - 1) as usize * nx_bin as usize * ny_bin as usize;
                            array[output_at..output_at + nx_bin as usize * ny_bin as usize]
                                .fill(0.0);
                            iring_pos[iz_out as usize] = iring_end;
                            if iring_start == 0 {
                                iring_start = 1;
                            }
                        }
                        let z_cen = iz_out as f32 * bin_z + z_center_offset;
                        let z_weight = slice_weighting(
                            inz,
                            ifilt_type,
                            ibin_z,
                            non_integer_z,
                            input_header.nz,
                            z_cen,
                            lines_filt,
                        );
                        let output_at = (iring_pos[iz_out as usize] - 1) as usize
                            * nx_bin as usize
                            * ny_bin as usize
                            + nx_bin as usize * iy_binned as usize;
                        unsafe {
                            bin_into_slice(
                                array[input_at..].as_mut_ptr(),
                                nx_bin_into_slice,
                                array[output_at..].as_mut_ptr(),
                                nx_bin,
                                num_bin_lines as i32,
                                ired_x_into_slice,
                                ired_y_into_slice,
                                z_weight,
                            );
                        }
                        last_z_added = iz_out;
                    }
                }
                iy += num_lines;
                iy_binned += num_bin_lines;
            }
            for iz_out in last_z_finished + 1..=last_z_added {
                if inz == input_ends[iz_out as usize] {
                    let output_at = (iring_pos[iz_out as usize] - 1) as usize
                        * nx_bin as usize
                        * ny_bin as usize;
                    for value in &array[output_at..output_at + nx_bin as usize * ny_bin as usize] {
                        dmin = dmin.min(*value);
                        dmax = dmax.max(*value);
                        sum += f64::from(*value);
                    }
                    if unsafe { iiu_write_section(3, array[output_at..].as_mut_ptr().cast()) } != 0
                    {
                        eprintln!("ERROR: BINVOL - Writing image");
                        unsafe {
                            iiu_close(3);
                            iiu_close(1);
                        };
                        std::process::exit(1);
                    }
                    last_z_finished = iz_out;
                    iring_start += 1;
                }
            }
        }
    } else {
        for iz_out in 0..nz_bin {
            let mut iy = 0_i64;
            for i_chunk in 0..num_chunks {
                let num_lines = max_lines.min(input_header.ny as i64 - i_chunk * max_lines);
                let num_bin_lines = num_lines / i64::from(ibin_y);
                array[..in_base as usize].fill(0.0);
                let z_cen = iz_out as f32 * bin_z + z_center_offset;
                for inz in input_starts[iz_out as usize]..=input_ends[iz_out as usize] {
                    let input_at = in_base as usize;
                    let read_error = if antialias_xy {
                        let mut error = 0;
                        unsafe {
                            iiu_read_reduced(
                                1,
                                inz,
                                array[input_at..].as_mut_ptr(),
                                nx_bin,
                                0.0,
                                iy as f32,
                                bin_x,
                                nx_bin,
                                num_bin_lines as i32,
                                ifilt_type - 1,
                                array[itemp_base as usize..].as_mut_ptr(),
                                (nx as i64 * max_lines) as i32,
                                &mut error,
                            );
                        }
                        error
                    } else {
                        unsafe {
                            iiu_set_position(1, inz, iy as i32);
                            iiu_read_lines(
                                1,
                                array[input_at..].as_mut_ptr().cast(),
                                num_lines as i32,
                            )
                        }
                    };
                    if read_error != 0 {
                        eprintln!("ERROR: BINVOL - Reading image");
                        unsafe {
                            iiu_close(3);
                            iiu_close(1);
                        };
                        std::process::exit(1);
                    }
                    let z_weight = slice_weighting(
                        inz,
                        ifilt_type,
                        ibin_z,
                        non_integer_z,
                        input_header.nz,
                        z_cen,
                        lines_filt,
                    );
                    unsafe {
                        bin_into_slice(
                            array[input_at..].as_mut_ptr(),
                            nx_bin_into_slice,
                            array.as_mut_ptr(),
                            nx_bin,
                            num_bin_lines as i32,
                            ired_x_into_slice,
                            ired_y_into_slice,
                            z_weight,
                        );
                    }
                }
                iy += num_lines;
                for value in &array[..in_base as usize] {
                    dmin = dmin.min(*value);
                    dmax = dmax.max(*value);
                    sum += f64::from(*value);
                }
                if unsafe { iiu_write_lines(3, array.as_mut_ptr().cast(), num_bin_lines as i32) }
                    != 0
                {
                    eprintln!("ERROR: BINVOL - Writing image");
                    unsafe {
                        iiu_close(3);
                        iiu_close(1);
                    };
                    std::process::exit(1);
                }
            }
        }
    }
    unsafe {
        (*output_header).amin = dmin;
        (*output_header).amax = dmax;
        (*output_header).amean = (sum / (nx_bin as f64 * ny_bin as f64 * nz_bin as f64)) as f32;
        let mut title = [b' '; MRC_LABEL_SIZE + 1];
        let prefix = if antialias_xy || antialias_z {
            format!(
                "BINVOL: Volume reduced by factors{:7.2}{:7.2}{:7.2}",
                bin_x, bin_y, bin_z
            )
        } else {
            format!(
                "BINVOL: Volume binned down by factors{:4}{:4}{:4}",
                ibin_x, ibin_y, ibin_z
            )
        };
        let prefix_bytes = prefix.as_bytes();
        title[..prefix_bytes.len().min(MRC_LABEL_SIZE)]
            .copy_from_slice(&prefix_bytes[..prefix_bytes.len().min(MRC_LABEL_SIZE)]);
        let mut date = [b' '; 9];
        b3d_date(&mut date);
        title[56..65].copy_from_slice(&date);
        let mut now = 0_i64;
        let mut local = core::mem::zeroed::<libc::tm>();
        libc::time(&raw mut now);
        libc::localtime_r(&raw const now, &raw mut local);
        let time = format!(
            "{:02}:{:02}:{:02}",
            local.tm_hour, local.tm_min, local.tm_sec
        );
        title[67..75].copy_from_slice(time.as_bytes());
        let mut title_c = [0_i8; MRC_LABEL_SIZE + 1];
        core::ptr::copy_nonoverlapping(title.as_ptr().cast(), title_c.as_mut_ptr(), MRC_LABEL_SIZE);
        if iiu_write_header_str(
            3,
            title_c.as_ptr(),
            1,
            (*output_header).amin,
            (*output_header).amax,
            (*output_header).amean,
        ) != 0
        {
            eprintln!("ERROR: BINVOL - Writing output header (final)");
            iiu_close(3);
            iiu_close(1);
            std::process::exit(1);
        }
        iiu_close(3);
        iiu_close(1);
    }
    println!(" PROGRAM EXECUTED TO END.");
}

/// Original contained `sliceWeighting` (`binvol.f90:522`).
pub fn slice_weighting(
    iz_in: i32,
    ifilt_type: i32,
    ibin_z: i32,
    non_integer_z: bool,
    nz: i32,
    z_cen: f32,
    lines_filt: i32,
) -> f32 {
    if ifilt_type <= 0 || (ibin_z == 1 && !non_integer_z) {
        return 1.0 / ibin_z as f32;
    }
    if iz_in > 0 && iz_in < nz - 1 {
        return unsafe { zoom_filt_value(iz_in as f32 + 0.5 - z_cen) as f32 };
    }
    let (jstart, jend) = if iz_in == 0 {
        (-lines_filt, 0)
    } else {
        (iz_in, iz_in + lines_filt)
    };
    let mut weight = 0.0;
    for jj in jstart..=jend {
        if jj as f32 - z_cen <= lines_filt as f32 && jj as f32 - z_cen >= -lines_filt as f32 {
            weight += unsafe { zoom_filt_value(jj as f32 + 0.5 - z_cen) as f32 };
        }
    }
    weight
}

#[cfg(test)]
mod tests {
    use super::slice_weighting;
    use crate::imod::libcfshr::zoomdown::select_zoom_filter;
    #[test]
    fn slice_weighting_follows_the_unfiltered_fortran_branch() {
        assert_eq!(slice_weighting(3, 0, 4, false, 10, 0.0, 2), 0.25);
    }
    #[test]
    fn slice_weighting_sums_the_edge_filter_tail() {
        let mut lines = 0;
        assert_eq!(unsafe { select_zoom_filter(1, 0.5, &mut lines) }, 0);
        assert!(slice_weighting(0, 2, 2, false, 5, 0.5, lines) > 0.0);
    }
}
