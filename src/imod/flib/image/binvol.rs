//! Translation of `IMOD/flib/image/binvol.f90`.
//!
//! The Fortran main program maps to [`binvol`] and its one contained procedure
//! maps to [`slice_weighting`]; no non-source algorithm helpers are introduced.
#![allow(unused_variables)]

use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::libcfshr::b3dutil::{
    b3d_physical_memory, set_float_output_for_entered_mode, standard_memory_limit_mb,
};
use crate::imod::libcfshr::filtxcorr::{
    fourier_crop_sizes, fourier_expand_volume, fourier_reduce_volume,
};
use crate::imod::libcfshr::parse_params::{pip_get_float, pip_get_integer, pip_get_three_floats};
use crate::imod::libcfshr::reduce_by_binning::{bin_into_slice, irepak};
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use crate::imod::libcfshr::taperpad::slice_taper_out_pad;
use crate::imod::libcfshr::zoomdown::{select_zoom_filter, zoom_filt_value};
use crate::imod::libfft::nice_fft_limit;
use crate::imod::libfft::thrdfft::thrdfft;
use crate::imod::libiimod::mrcfiles::MRC_LABEL_SIZE;
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_read_lines, iiu_read_section, iiu_set_position, iiu_write_lines,
    iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_delta, iiu_alt_mode, iiu_alt_origin, iiu_alt_sample, iiu_alt_size, iiu_ret_delta,
    iiu_ret_origin, iiu_trans_header, iiu_write_header_str,
};
use crate::imod::libiimod::unit_reduced::iiu_read_reduced;
use chrono::{Local, Timelike};

/// `parameter (numOptions = 14)` (`binvol.f90:46`).
const BINVOL_NUM_OPTIONS: i32 = 14;

/// Source fallback PIP table (`binvol.f90:48`), kept as the 14 `@`-separated
/// entries of the Fortran `options(1)` string rather than a Rust option list.
const BINVOL_OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@mode:ModeToOutput:I:@\
binning:BinningFactor:F:@xbinning:XBinningFactor:F:@\
ybinning:YBinningFactor:F:@zbinning:ZBinningFactor:F:@\
antialias:AntialiasZFilter:I:@spread:SpreadSlicesInZ:B:@\
ftreduce:FourierReduceByBinning:B:@ftexpand:FourierExpandByBinning:B:@\
memory:MemoryLimit:I:@verbose:VerboseOutput:I:@help:usage:B:";

/// Original program `binvol` (`binvol.f90:5`).
///
/// This source program maps its real-space and Fourier volume paths directly
/// to the corresponding translated IMOD library units.
pub fn binvol() {
    unsafe {
        // gfortran writes list-directed output as a leading blank per record
        // plus one blank separator ahead of every value, with INTEGER(4) in an
        // eleven-column field and REAL(4) in `G16.9E2` editing with `1P`
        // scaling.  `print *` statements in this program are reproduced with
        // these two closures rather than with Rust's own `{}` formatting.
        let list_int = |value: i32| -> String { format!(" {value:>11}") };
        let list_real = |value: f32| -> String {
            // `libgfortran/io/write.c` `write_infnan`: a NaN prints as `NaN`
            // and an infinity as `Infinity` (`Inf` when the field is narrower
            // than 8, with a leading `-` for negative), right-justified in the 16-column list-directed field.
            if value.is_nan() {
                return format!("{:>16}", "NaN");
            }
            if value.is_infinite() {
                let text = match (value < 0.0, 16) {
                    (false, _) => "Infinity",
                    (true, _) => "-Infinity",
                };
                return format!("{text:>16}");
            }
            let magnitude = value.abs();
            let mut exponent = 1_i32;
            if magnitude != 0.0 {
                let scientific = format!("{magnitude:.8e}");
                let (_, power) = scientific.split_once('e').unwrap();
                exponent = power.parse::<i32>().unwrap() + 1;
            }
            if (0..=9).contains(&exponent) {
                format!(
                    " {:>12}    ",
                    format!("{:.*}", (9 - exponent) as usize, value)
                )
            } else {
                let scientific = format!("{value:.8e}");
                let (mantissa, power) = scientific.split_once('e').unwrap();
                let power = power.parse::<i32>().unwrap();
                format!(
                    " {:>16}",
                    format!(
                        "{mantissa}E{}{:02}",
                        if power < 0 { '-' } else { '+' },
                        power.abs()
                    )
                )
            }
        };

        let mut red_all = 2.0_f32;
        let mut limit = 1000_i32;
        let mut ifilt_type = -1_i32;
        let ifilt_default = 6_i32;
        let mut spread_z = false;
        let mut if_xy_anti_alias = 0_i32;
        let mut if_verbose = 0_i32;
        let mut ft_crop = false;
        let mut ft_expand = false;

        // `PipReadOrParseOptions(options, numOptions, 'binvol', 'ERROR: BINVOL - ',
        // .false., 2, 1, 1, numOptArg, numNonOptArg)` (`binvol.f90:67`).
        let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
        let mut ierr: i32;
        pip_read_or_parse_options(
            &[BINVOL_OPTIONS],
            BINVOL_NUM_OPTIONS,
            "binvol",
            "ERROR: BINVOL - ",
            false,
            2,
            1,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
        );
        let mut big_file = String::new();
        if pip_get_in_out_file("InputFile", 1, " ", &mut big_file) != 0 {
            exit_error("No input file specified");
        }
        let mut out_file = String::new();
        if pip_get_in_out_file("OutputFile", 2, " ", &mut out_file) != 0 {
            exit_error("No output file specified");
        }

        pip_get_float(b"BinningFactor", &mut red_all);
        let mut bin_x = red_all;
        let mut bin_y = red_all;
        let mut bin_z = red_all;
        pip_get_float(b"XBinningFactor", &mut bin_x);
        pip_get_float(b"YBinningFactor", &mut bin_y);
        pip_get_float(b"ZBinningFactor", &mut bin_z);
        pip_get_logical("FourierReduceByBinning", &mut ft_crop);
        pip_get_logical("FourierExpandByBinning", &mut ft_expand);
        let mut x_shift = 0.0_f32;
        let mut y_shift = 0.0_f32;
        let mut z_shift = 0.0_f32;
        pip_get_three_floats(b"ShiftsInXYZ", &mut x_shift, &mut y_shift, &mut z_shift);
        if ft_expand && ft_crop {
            exit_error("You cannot enter both -ftReduce and -ftExpand");
        }
        if ft_expand || ft_crop {
            ifilt_type = 0;
        }

        // Get default memory limit and entry
        let physical_mem = (b3d_physical_memory() / f64::from(1024_i32.pow(2))) as f32;
        let mut max_limit = 8000_i32;
        if physical_mem > 0.0 {
            limit = standard_memory_limit_mb(30000) as i32;
            max_limit = limit.max((0.8 * physical_mem) as i32);
        }
        ierr = pip_get_integer(b"MemoryLimit", &mut limit);
        if ierr == 0 && (limit < 1 || limit > max_limit) {
            // `write(bigfile, '(a, i7)')` then `call exitError(bigFile)`
            // (`binvol.f90:100`).
            exit_error(&format!("Memory limit must be between 1 and {max_limit:7}"));
        }
        let lim_dim = i64::from(limit) * 1024 * 256;

        pip_get_integer(b"AntialiasZFilter", &mut ifilt_type);
        if ifilt_type == 1 || ifilt_type > 6 {
            exit_error("Antialias filter type must be between 2 and 6, or < 0 for default");
        }
        if ifilt_type < 0 {
            ifilt_type = ifilt_default;
        }
        pip_get_integer(b"VerboseOutput", &mut if_verbose);
        pip_get_logical("SpreadSlicesInZ", &mut spread_z);
        if spread_z && ifilt_type <= 0 {
            exit_error("Spreading in Z can be used only with antialias filtering");
        }
        if (spread_z || ifilt_type != 0) && (ft_expand || ft_crop) {
            exit_error("You cannot enter -antialias or -spread with Fourier operations");
        }

        let red_fac = [bin_x, bin_y, bin_z];
        let mut ired_fac = [
            bin_x.round() as i32,
            bin_y.round() as i32,
            bin_z.round() as i32,
        ];
        let non_integer = [
            (ired_fac[0] as f32 - red_fac[0]).abs() > 0.001,
            (ired_fac[1] as f32 - red_fac[1]).abs() > 0.001,
            (ired_fac[2] as f32 - red_fac[2]).abs() > 0.001,
        ];
        if (non_integer[0] || non_integer[1] || non_integer[2])
            && ifilt_type <= 0
            && !(ft_crop || ft_expand)
        {
            exit_error(
                "Reduction factors must all be integers unless antialias filtering or a Fourier operation is specified",
            );
        }
        let unequal_xy = (bin_x - bin_y).abs() > 0.001;
        if unequal_xy {
            if non_integer[0] || non_integer[1] {
                if ft_crop || ft_expand {
                    exit_error("Scaling in X and Y must be equal for Fourier operations");
                }
                exit_error("Reduction in X and Y must be equal unless they are both integers");
            }
        } else {
            bin_y = bin_x;
            ired_fac[1] = ired_fac[0];
        }
        //
        // Open image files.
        //
        imopen(1, &big_file, "RO");
        let mut nxyz = [0_i32; 3];
        let mut mxyz = [0_i32; 3];
        let mut mode = 0_i32;
        let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &raw mut mode,
            &raw mut dmin,
            &raw mut dmax,
            &raw mut dmean,
        );
        let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);
        if nz == 1 {
            bin_z = 1.0;
        }
        let red_fac = [bin_x, bin_y, bin_z];
        //
        if pip_get_integer(b"ModeToOutput", &mut mode) == 0 {
            set_float_output_for_entered_mode(mode);
        }
        if mode != 0 && mode != 1 && mode != 2 && mode != 6 && mode != 12 {
            exit_error("The mode of the input or output must be 0, 1, 2, 6 or 12");
        }

        imopen(3, &out_file, "NEW");
        //
        iiu_trans_header(3, 1);
        iiu_alt_mode(3, mode);
        //
        let mut delta = [0.0_f32; 3];
        iiu_ret_delta(1, &mut delta);
        let mut nxyz_bin = [0_i32; 3];
        let mut nfs_pad = [0_i32; 3];
        let mut ncrop_pad = [0_i32; 3];
        let mut xyz_shift = [0.0_f32; 3];
        for i in 0..3 {
            if ired_fac[i] < 1 {
                exit_error("Reduction or expansion factor must be at least 1");
            }
            if ft_crop || ft_expand {
                let mut base = red_fac[i];
                if ft_expand {
                    base = 1.0 / red_fac[i];
                }
                let mut actual_fac = 0.0_f32;
                ierr = fourier_crop_sizes(
                    nxyz[i],
                    base,
                    0.01,
                    16,
                    nice_fft_limit(),
                    &mut nfs_pad[i],
                    &mut ncrop_pad[i],
                    &mut actual_fac,
                );
                if ierr > 0 {
                    exit_error(
                        "Reduction or expansion factor must be an integer or integer divided by 2, 3, 4, 5, 6, 8, or 10 to 3 decimal places",
                    );
                }
                nxyz_bin[i] = (nxyz[i] as f32 / actual_fac) as i32;
                delta[i] *= actual_fac;
                xyz_shift[i] = 0.0;
                if ft_expand {
                    if nxyz[i] % 2 != 0 {
                        xyz_shift[i] = 0.5;
                    }
                } else {
                    if nxyz_bin[i] % 2 != 0 {
                        xyz_shift[i] = -red_fac[i] * 0.5;
                    }
                    // `(mod(nxyz(i), nint(redFac(i))) + 1) / 2` is integer division.
                    xyz_shift[i] += (((nxyz[i] % red_fac[i].round() as i32) + 1) / 2) as f32;
                }
                println!(
                    "{}{}{}",
                    list_int(nxyz[i]),
                    list_int(nxyz_bin[i]),
                    list_real(xyz_shift[i])
                );
            } else {
                nxyz_bin[i] = (nxyz[i] as f32 / red_fac[i]) as i32;
                delta[i] *= red_fac[i];
            }
            if nxyz_bin[i] < 1 {
                exit_error("Reduction factor too large");
            }
        }
        let (nx_bin, ny_bin) = (nxyz_bin[0], nxyz_bin[1]);
        let mut nz_bin = nxyz_bin[2];
        let mut nx_bin_into_slice = nx;
        let mut ired_bin_into_slice = ired_fac;
        if ifilt_type > 1 && !unequal_xy && bin_x > 1.0 {
            println!(" Antialiasing is being applied in X and Y as well as Z");
            if_xy_anti_alias = 1;
            ired_bin_into_slice[0] = 1;
            ired_bin_into_slice[1] = 1;
            nx_bin_into_slice = nx_bin;
        }
        let antialias_z = ifilt_type > 1 && bin_z > 1.0;
        //
        // Set up parameters of slices to add into each slice, etc
        let mut lines_filt = 0_i32;
        let mut z_cen_offset = 0.0_f32;
        if antialias_z {
            if select_zoom_filter(ifilt_type - 1, f64::from(1.0_f32 / bin_z), &mut lines_filt) != 0
            {
                exit_error("Setting up the antialias filter");
            }
            z_cen_offset = bin_z / 2.0;

            // If the input Z size is not an exact multiple of the binning we can always
            // spread to another output slice, dividing the extra pixels equally between
            // top and bottom
            if spread_z && bin_z * nz as f32 - nz_bin as f32 > 0.49 {
                nz_bin += 1;
                let extra_pix = nz_bin as f32 * bin_z - nz as f32;
                z_cen_offset = bin_z / 2.0 - extra_pix / 2.0;

                // Add to the origin half the extra pixels
                let mut origin = [0.; 3];
                iiu_ret_origin(1, &mut origin);
                origin[2] += delta[2] * extra_pix / 2.0;
                iiu_alt_origin(3, &origin);
            }
        }
        nxyz_bin[2] = nz_bin;

        // Common header operations
        let mut nxyzst = [0_i32; 3];
        iiu_alt_size(3, &nxyz_bin, &nxyzst);
        iiu_alt_sample(3, &nxyz_bin);
        iiu_alt_delta(3, &delta);
        let mut dmean_sum = 0.0_f64;
        dmax = -1.0e30;
        dmin = 1.0e30;
        let mut pix_sum = 0.0_f64;
        let (mut dmin2, mut dmax2, mut dmean2) = (0.0_f32, 0.0_f32, 0.0_f32);

        // FOURIER CROPPING OR EXPANDING
        if ft_expand || ft_crop {
            //
            // Get memory for input, output, and work
            let idim_in =
                i64::from(nfs_pad[0] + 2) * i64::from(nfs_pad[1]) * i64::from(nfs_pad[2]) + 16;
            let idim_out =
                i64::from(ncrop_pad[0] + 2) * i64::from(ncrop_pad[1]) * i64::from(ncrop_pad[2])
                    + 16;
            let mut nx_dim = nfs_pad[0] + 2;
            let max_xdim = nx_dim.max(ncrop_pad[0] + 2);
            let idim_work = i64::from((max_xdim * nfs_pad[2].max(ncrop_pad[2])).max(max_xdim)) + 16;
            if idim_in + idim_out + idim_work > lim_dim {
                exit_error(
                    "Full volume input plus output array sizes are bigger than the memory limit",
                );
            }
            let mut fft_in = vec![0.0_f32; idim_in as usize];
            let mut fft_out = vec![0.0_f32; idim_out as usize];
            let mut fft_work = vec![0.0_f32; idim_work as usize];
            //
            // Load the input array with the full volume and tapers in Z
            let iz_high = nz - 1;
            let mut iz_start = -((nfs_pad[2] - nz) / 2);
            let iz_end = iz_start + nfs_pad[2] - 1;
            for iz in iz_start..=iz_end {
                let iz_read = 0.max(iz_high.min(iz));
                let ibase =
                    (i64::from(nx_dim) * i64::from(nfs_pad[1]) * i64::from(iz - iz_start)) as usize;
                iiu_set_position(1, iz_read, 0);
                if iiu_read_section(1, fft_in[ibase..].as_mut_ptr().cast()) != 0 {
                    exit_error("Reading image file");
                }
                // `binvol.f90` passes the same array as input and output;
                // `PadIn::InPlace` is that case.
                slice_taper_out_pad(
                    crate::imod::libcfshr::taperpad::PadIn::InPlace,
                    2,
                    nx,
                    ny,
                    &mut fft_in[ibase..],
                    nx_dim,
                    nfs_pad[0],
                    nfs_pad[1],
                    1,
                    dmean,
                );
                if iz < 0 || iz > iz_high {
                    let atten = if iz < 0 {
                        (iz - iz_start) as f32 / (0 - iz_start) as f32
                    } else {
                        (iz_end - iz) as f32 / (iz_end - iz_high) as f32
                    };
                    let base = (1.0 - atten) * dmean;
                    for iy in 1..=nfs_pad[1] {
                        let ix_base = ibase + ((iy - 1) * nx_dim) as usize;
                        for i8 in ix_base..ix_base + nfs_pad[0] as usize {
                            fft_in[i8] = base + atten * fft_in[i8];
                        }
                    }
                }
            }
            //
            // Take FFT, reduce or expand with 0 shift, inverse FFT
            let (mut fx, mut fy, mut fz, mut idir) = (nfs_pad[0], nfs_pad[1], nfs_pad[2], 0);
            thrdfft(&mut fft_in, &mut fft_work, fx, fy, fz, idir);
            if ft_crop {
                fourier_reduce_volume(
                    &fft_in,
                    nfs_pad[0],
                    nfs_pad[1],
                    nfs_pad[2],
                    &mut fft_out,
                    ncrop_pad[0],
                    ncrop_pad[1],
                    ncrop_pad[2],
                    xyz_shift[0] + x_shift,
                    xyz_shift[1] + y_shift,
                    xyz_shift[2] + z_shift,
                    Some(&mut fft_work),
                );
            } else {
                fourier_expand_volume(
                    &mut fft_in,
                    nfs_pad[0],
                    nfs_pad[1],
                    nfs_pad[2],
                    &mut fft_out,
                    ncrop_pad[0],
                    ncrop_pad[1],
                    ncrop_pad[2],
                    xyz_shift[0] + x_shift,
                    xyz_shift[1] + y_shift,
                    xyz_shift[2] + z_shift,
                    Some(&mut fft_work),
                );
            }
            let (mut cx, mut cy, mut cz, mut idir) = (ncrop_pad[0], ncrop_pad[1], ncrop_pad[2], -1);
            thrdfft(&mut fft_out, &mut fft_work, cx, cy, cz, idir);
            //
            // Write the planes
            dmean_sum = 0.0;
            dmax = -1.0e30;
            dmin = 1.0e30;
            let ix_start = (ncrop_pad[0] - nx_bin) / 2;
            let iy_start = (ncrop_pad[1] - ny_bin) / 2;
            iz_start = -((ncrop_pad[2] - nz_bin) / 2);
            nx_dim = ncrop_pad[0] + 2;
            for iz in 0..nz_bin {
                iiu_set_position(3, iz, 0);
                let ibase = (i64::from(nx_dim) * i64::from(ncrop_pad[1]) * i64::from(iz - iz_start))
                    as usize;
                // `binvol.f90` repacks in place: the source passes the same
                // array as input and output, which `reduce_by_binning.c`
                // documents as allowed.  Rust cannot hold a shared and a
                // mutable borrow of one buffer, so the source region is copied
                // out first; the repack only ever reads at an index >= the
                // index it writes, so the bytes are the same either way.
                let source: Vec<u8> = fft_out[ibase..]
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect();
                let region = &mut fft_out[ibase..];
                let destination = core::slice::from_raw_parts_mut(
                    region.as_mut_ptr().cast::<u8>(),
                    region.len() * 4,
                );
                irepak(
                    destination,
                    &source,
                    &nx_dim,
                    &ncrop_pad[1],
                    &ix_start,
                    &(ix_start + nx_bin - 1),
                    &iy_start,
                    &(iy_start + ny_bin - 1),
                );
                array_min_max_mean_fortran(
                    &fft_out[ibase..],
                    &nx_bin,
                    &ny_bin,
                    &1,
                    &nx_bin,
                    &1,
                    &ny_bin,
                    &mut dmin2,
                    &mut dmax2,
                    &mut dmean2,
                );
                iiu_write_section(3, fft_out[ibase..].as_mut_ptr().cast());
                dmax = dmax.max(dmax2);
                dmin = dmin.min(dmin2);
                dmean_sum += f64::from(dmean2);
            }
            dmean = (dmean_sum / f64::from(nz_bin)) as f32;
        } else {
            //
            // REAL-SPACE REDUCTION

            // Allocate and set up arrays for output slices
            let mut input_starts = vec![0_i32; nz_bin as usize + 1];
            let mut input_ends = vec![0_i32; nz_bin as usize + 1];
            let mut iring_pos = vec![0_i32; nz_bin as usize + 1];

            let mut num_slice_used_in = 1_i32;
            if antialias_z {
                //
                // For each output Z, find out the range of input slices needed
                for iz_out in 0..nz_bin {
                    let z_cen = iz_out as f32 * bin_z + z_cen_offset;
                    let iz_cen = (z_cen - 0.5).round() as i32;
                    //
                    // Get the weights to figure out the number of slices before and after
                    // the center
                    let mut num_z_before = -1_i32;
                    let mut num_z_after = 0_i32;
                    for i in -lines_filt..=lines_filt {
                        let z_weight = zoom_filt_value((i + iz_cen) as f32 + 0.5 - z_cen) as f32;
                        if num_z_before < 0 && z_weight != 0.0 {
                            num_z_before = -i;
                        }
                        if z_weight != 0.0 {
                            num_z_after = i;
                        }
                    }
                    input_starts[iz_out as usize] = 0.max(iz_cen - num_z_before);
                    input_ends[iz_out as usize] = (iz_cen + num_z_after).min(nz - 1);
                    iring_pos[iz_out as usize] = 0;
                }

                //
                // Determine the maximum number of slices a particular input slice is used in
                let mut num_slice = 0_i32;
                for inz in 0..=nz {
                    for iz_out in 0..nz_bin {
                        if input_starts[iz_out as usize] <= inz
                            && input_ends[iz_out as usize] >= inz
                        {
                            num_slice += 1;
                        }
                        if input_ends[iz_out as usize] < inz {
                            break;
                        }
                    }
                    num_slice_used_in = num_slice_used_in.max(num_slice);
                }
            } else {
                // Ordinary binning in Z, set up arrays for this case
                z_cen_offset = 0.0;
                num_slice_used_in = 1;
                for iz_out in 0..nz_bin {
                    input_starts[iz_out as usize] = iz_out * ired_fac[2];
                    input_ends[iz_out as usize] = iz_out * ired_fac[2] + ired_fac[2] - 1;
                    iring_pos[iz_out as usize] = 0;
                }
            }

            // First try holding all output slices that take data from one input slice
            let mut full_out_slices = false;
            let nx_equiv = (nx as f32
                + if_xy_anti_alias as f32 * ((nx_bin as f32 + bin_y - 0.9) / bin_y))
                as i32;
            let mut max_lines = 0_i32;
            let mut in_base = 0_i64;
            if f64::from(nx_bin as f32 * ny_bin as f32 * num_slice_used_in as f32) < lim_dim as f64
            {
                in_base = i64::from(nx_bin) * i64::from(ny_bin) * i64::from(num_slice_used_in);
                // `maxLines` is `integer*4` while the expression is `real*4`;
                // an out-of-range conversion is `cvttss2si`, which yields
                // `-2147483648`, and the source relies on that value being
                // rejected by the `maxLines > 1` test below.
                let real_lines = bin_y
                    * (((lim_dim - in_base) / i64::from(nx_equiv)) / i64::from(bin_y.ceil() as i32))
                        as f32;
                max_lines = if real_lines >= -2147483648.0 && real_lines < 2147483648.0 {
                    real_lines as i32
                } else {
                    i32::MIN
                };
                full_out_slices = max_lines > 1;
                max_lines = max_lines.min(ny);
            }
            if !full_out_slices {
                // If that fails, process a strip at a time, reading all input for it
                // sequentially and producing output
                let real_lines = bin_y
                    * ((lim_dim
                        / (i64::from(nx_equiv) + ((nx_bin as f32 + bin_y - 0.9) / bin_y) as i64))
                        / i64::from(bin_y.ceil() as i32)
                        - 1) as f32;
                max_lines = if real_lines >= -2147483648.0 && real_lines < 2147483648.0 {
                    real_lines as i32
                } else {
                    i32::MIN
                };
                if max_lines < 2 {
                    exit_error(
                        "Input images too large with given reduction factors and memory limit",
                    );
                }
                max_lines = max_lines.min(ny);
                in_base = i64::from(nx_bin * (max_lines as f32 / bin_y) as i32);
            }
            let itemp_base = (nx_equiv - nx) * max_lines + in_base as i32;
            let mut array = vec![0.0_f32; (i64::from(nx_equiv * max_lines) + in_base) as usize];

            pix_sum = 0.0;
            if if_verbose > 0 {
                println!(
                    " fullOutSlices{}{}{}{}",
                    if full_out_slices { " T" } else { " F" },
                    "  num used in, cen off",
                    list_int(num_slice_used_in),
                    list_real(z_cen_offset)
                );
            }

            let num_chunks = (ny + max_lines - 1) / max_lines;
            if if_verbose > 0 {
                println!(
                    "{}{}{}",
                    list_int(num_chunks),
                    "  chunks, maximum lines",
                    list_int(max_lines)
                );
            }
            if full_out_slices {
                //
                // Ring buffer
                let mut iring_start = 0_i32;
                let mut iring_end = 0_i32;
                let mut last_z_finished = -1_i32;
                let mut last_z_added = 0_i32;
                //
                // Loop on input slices and on strips in slices
                for inz in 0..nz {
                    let mut iy = 0_i32;
                    let mut iy_binned = 0_i32;
                    for i_chunk in 1..=num_chunks {
                        let num_lines = max_lines.min(ny - (i_chunk - 1) * max_lines);
                        let num_bin_lines = (num_lines as f32 / bin_y) as i32;
                        if if_xy_anti_alias > 0 {
                            let mut error = 0;
                            let (before_temp, temp) = array.split_at_mut(itemp_base as usize);
                            iiu_read_reduced(
                                1,
                                inz,
                                &mut before_temp[in_base as usize..],
                                nx_bin,
                                0.0,
                                iy as f32,
                                bin_x,
                                nx_bin,
                                num_bin_lines,
                                ifilt_type - 1,
                                temp,
                                nx * max_lines,
                                &mut error,
                            );
                            if error != 0 {
                                exit_error("Reading image");
                            }
                        } else {
                            iiu_set_position(1, inz, iy);
                            if iiu_read_lines(
                                1,
                                array[in_base as usize..].as_mut_ptr().cast(),
                                num_lines,
                            ) != 0
                            {
                                exit_error("Reading image");
                            }
                        }
                        //
                        // Loop on output slices that need this input
                        for iz_out in last_z_finished + 1..nz_bin {
                            if inz > input_ends[iz_out as usize] {
                                break;
                            }
                            if inz >= input_starts[iz_out as usize] {
                                //
                                // If slice is not in ring yet, assign and zero it
                                if iring_pos[iz_out as usize] == 0 {
                                    iring_end += 1;
                                    if iring_end > num_slice_used_in {
                                        iring_end = 1;
                                    }
                                    if iring_end == iring_start {
                                        exit_error("Problem with ring buffer");
                                    }
                                    let iout_base = ((iring_end - 1) * nx_bin * ny_bin) as usize;
                                    array[iout_base..iout_base + (nx_bin * ny_bin) as usize]
                                        .fill(0.0);
                                    iring_pos[iz_out as usize] = iring_end;
                                    if iring_start == 0 {
                                        iring_start = 1;
                                    }
                                    if if_verbose > 1 {
                                        println!(
                                            " Assigning {}{}{}",
                                            list_int(iz_out),
                                            "  to slot ",
                                            list_int(iring_end)
                                        );
                                    }
                                }
                                //
                                // Get weighting and add the strip in
                                let z_cen = iz_out as f32 * bin_z + z_cen_offset;
                                let z_weight = slice_weighting(
                                    inz,
                                    ifilt_type,
                                    ired_fac[2],
                                    non_integer[2],
                                    nz,
                                    z_cen,
                                    lines_filt,
                                );
                                let iout_base = ((iring_pos[iz_out as usize] - 1) * nx_bin * ny_bin
                                    + nx_bin * iy_binned)
                                    as usize;
                                if if_verbose > 1 {
                                    println!(
                                        " Adding{}{}{}{}{}{}",
                                        list_int(inz),
                                        "  into",
                                        list_int(iz_out),
                                        " , weight",
                                        list_real(z_weight),
                                        list_int(iout_base as i32 + 1)
                                    );
                                }
                                // `array` holds the output ring below
                                // `in_base` and the input lines above it, which
                                // is the boundary the source itself uses.
                                let (ring, input) = array.split_at_mut(in_base as usize);
                                bin_into_slice(
                                    input,
                                    nx_bin_into_slice,
                                    &mut ring[iout_base..],
                                    nx_bin,
                                    num_bin_lines,
                                    ired_bin_into_slice[0],
                                    ired_bin_into_slice[1],
                                    z_weight,
                                );
                            }
                            last_z_added = iz_out;
                        }

                        // Advance input and output lines for next chunk
                        iy += num_lines;
                        iy_binned += num_bin_lines;
                    }
                    //
                    // Loop again on output slices and see if any are done
                    // If so, write the slice, advance ring start
                    for iz_out in last_z_finished + 1..=last_z_added {
                        if inz == input_ends[iz_out as usize] {
                            let iout_base =
                                ((iring_pos[iz_out as usize] - 1) * nx_bin * ny_bin) as usize;
                            array_min_max_mean_fortran(
                                &array[iout_base..],
                                &nx_bin,
                                &ny_bin,
                                &1,
                                &nx_bin,
                                &1,
                                &ny_bin,
                                &mut dmin2,
                                &mut dmax2,
                                &mut dmean2,
                            );
                            iiu_write_section(3, array[iout_base..].as_mut_ptr().cast());
                            if if_verbose > 1 {
                                println!(
                                    " Wrote {}{}",
                                    list_int(iz_out),
                                    list_int(iout_base as i32 + 1)
                                );
                            }
                            //
                            dmax = dmax.max(dmax2);
                            dmin = dmin.min(dmin2);
                            dmean_sum += f64::from(dmean2 * ny_bin as f32 * nx_bin as f32);
                            pix_sum += f64::from(ny_bin * nx_bin);
                            last_z_finished = iz_out;
                            iring_start += 1;
                        }
                    }
                }
            } else {
                //
                // When output slices can't all be held for an input slice
                // Loop on output slices then on chunks in slice
                for iz_out in 0..nz_bin {
                    let mut iy = 0_i32;
                    for i_chunk in 1..=num_chunks {
                        let num_lines = max_lines.min(ny - (i_chunk - 1) * max_lines);
                        let num_bin_lines = (num_lines as f32 / bin_y) as i32;
                        array[..in_base as usize].fill(0.0);
                        let z_cen = iz_out as f32 * bin_z + z_cen_offset;
                        for inz in input_starts[iz_out as usize]..=input_ends[iz_out as usize] {
                            if if_xy_anti_alias > 0 {
                                let mut error = 0;
                                let (before_temp, temp) = array.split_at_mut(itemp_base as usize);
                                iiu_read_reduced(
                                    1,
                                    inz,
                                    &mut before_temp[in_base as usize..],
                                    nx_bin,
                                    0.0,
                                    iy as f32,
                                    bin_x,
                                    nx_bin,
                                    num_bin_lines,
                                    ifilt_type - 1,
                                    temp,
                                    nx * max_lines,
                                    &mut error,
                                );
                                if error != 0 {
                                    exit_error("Reading image");
                                }
                            } else {
                                iiu_set_position(1, inz, iy);
                                if iiu_read_lines(
                                    1,
                                    array[in_base as usize..].as_mut_ptr().cast(),
                                    num_lines,
                                ) != 0
                                {
                                    exit_error("Reading image");
                                }
                            }
                            let z_weight = slice_weighting(
                                inz,
                                ifilt_type,
                                ired_fac[2],
                                non_integer[2],
                                nz,
                                z_cen,
                                lines_filt,
                            );
                            let (output, input) = array.split_at_mut(in_base as usize);
                            bin_into_slice(
                                input,
                                nx_bin_into_slice,
                                output,
                                nx_bin,
                                num_bin_lines,
                                ired_bin_into_slice[0],
                                ired_bin_into_slice[1],
                                z_weight,
                            );
                        }
                        iy += num_lines;

                        array_min_max_mean_fortran(
                            &array,
                            &nx_bin,
                            &num_bin_lines,
                            &1,
                            &nx_bin,
                            &1,
                            &num_bin_lines,
                            &mut dmin2,
                            &mut dmax2,
                            &mut dmean2,
                        );
                        iiu_write_lines(3, array.as_mut_ptr().cast(), num_bin_lines);
                        //
                        dmax = dmax.max(dmax2);
                        dmin = dmin.min(dmin2);
                        dmean_sum += f64::from(dmean2 * num_bin_lines as f32 * nx_bin as f32);
                        pix_sum += f64::from(num_bin_lines * nx_bin);
                    }
                }
            }
            //
            dmean = (dmean_sum / pix_sum) as f32;
        }

        let mut dat = [b' '; 9];
        b3d_date(&mut dat);
        // `call time(tim)`.
        let local = Local::now();
        let tim = format!(
            "{:02}:{:02}:{:02}",
            local.hour(),
            local.minute(),
            local.second()
        );
        //
        let mut titlech = [b' '; MRC_LABEL_SIZE + 1];
        let head = if ft_crop {
            format!("BINVOL: Volume Fourier reduced by{bin_x:7.2}{bin_y:7.2}{bin_z:7.2}")
        } else if ft_expand {
            format!("BINVOL: Volume Fourier expanded by{bin_x:7.2}{bin_y:7.2}{bin_z:7.2}")
        } else if if_xy_anti_alias > 0 || antialias_z {
            format!("BINVOL: Volume reduced by factors{bin_x:7.2}{bin_y:7.2}{bin_z:7.2}")
        } else {
            format!(
                "BINVOL: Volume binned down by factors{:4}{:4}{:4}",
                ired_fac[0], ired_fac[1], ired_fac[2]
            )
        };
        let head = head.as_bytes();
        let head_len = head.len().min(MRC_LABEL_SIZE);
        titlech[..head_len].copy_from_slice(&head[..head_len]);
        titlech[56..65].copy_from_slice(&dat);
        titlech[67..75].copy_from_slice(tim.as_bytes());
        iiu_write_header_str(
            3,
            std::str::from_utf8(&titlech[..MRC_LABEL_SIZE]).unwrap_or_default(),
            1,
            dmin,
            dmax,
            dmean,
        );
        iiu_close(3);
        iiu_close(1);
        //
        println!(" PROGRAM EXECUTED TO END.");
        std::process::exit(0);
    }
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
    //
    // Add 0.5 to get to middle of pixel because pixel indexes start at 0
    if iz_in > 0 && iz_in < nz - 1 {
        return unsafe { zoom_filt_value(iz_in as f32 + 0.5 - z_cen) as f32 };
    }
    let (jstart, jend) = if iz_in == 0 {
        (-lines_filt, 0)
    } else {
        (iz_in, iz_in + lines_filt)
    };
    let mut weight = 0.0_f32;
    for jj in jstart..=jend {
        if jj as f32 - z_cen <= lines_filt as f32 && jj as f32 - z_cen >= -lines_filt as f32 {
            // `weight` is `real*4` and `zoomFiltValue` is `real*8`: the sum is
            // formed in double precision and rounded back on each assignment.
            weight =
                (f64::from(weight) + unsafe { zoom_filt_value(jj as f32 + 0.5 - z_cen) }) as f32;
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
