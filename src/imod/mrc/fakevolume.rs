//! Translation of `IMOD/mrc/fakevolume.cpp`.
//!
//! One Rust function per source function: `main`, `point_to_line` and
//! `pointLineSegDist` (the commented-out `point_to_line_dist` is not compiled
//! in the source and has no counterpart).
//!
//! Precision follows the C++ expression types exactly: `pow(float, 2.)` is the
//! `double` overload, so every squared-distance sum and every `critSphere`/
//! `critCylndr` is computed in `double` and only the assignment to a `float`
//! narrows it; `point_to_line`'s denominator `1. + aa * aa + cc * cc` is a
//! `double` sum of `float` products; the sub-pixel positions
//! `iz - 1 + (iDz - 0.5) / numDiv` are `double` expressions narrowed to
//! `float`.
//!
//! Two deviations for undefined behaviour in the source, both recorded in
//! `BUGS.md`:
//! - The per-object arrays hold `LIMOBJ` (200) entries and nothing bounds the
//!   entry counts, so a 201st sphere or cylinder writes past a stack array.
//!   The translation refuses that by name instead of overrunning.
//! - `fakevolume.cpp:199` fills `ifTrunc` up to `numSphere`, not `numCylndr`,
//!   and nothing fills `ifTrunc[0]` or `cylndrRad*[0]`/`cylndrDens*[0]` when
//!   the option is not entered, so those read uninitialised stack.  The
//!   translation's arrays are zero-initialised, which is what the reference
//!   binary was measured to read there (see `BUGS.md`).

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, exit, imod_prog_name, imod_usage_header,
    set_float_output_for_entered_mode,
};
use crate::imod::libcfshr::islice::slice_mode_if_real;
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_float, pip_get_in_out_file, pip_get_integer,
    pip_get_three_floats, pip_get_three_integers, pip_get_two_floats, pip_number_of_entries,
    pip_read_or_parse_options,
};
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_NLABELS};
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_open, iiu_write_section};
use crate::imod::libiimod::unit_header::{iiu_alt_cell, iiu_create_header, iiu_write_header_str};
use std::io::Write;

/// The `imodUsageHeader` callback in the shape `PipReadOrParseOptions` takes.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    // `PipPrintHelp` writes through Rust's stdout; the banner is on the C
    // stream, so hand it over before the help body follows it.
    let _ = ImodFile::Stdout.flush();
}

/// C `main` in `fakevolume.cpp` (`fakevolume.cpp:25`).
pub fn fakevolume(arguments: &[String]) -> i32 {
    // `char *progname = imodProgName(argv[0]);`
    let progname = imod_prog_name(arguments.first().map_or("", String::as_str));
    const LIMOBJ: usize = 200;
    let mut nxyz = [0i32; 3];
    let mut mxyz = [0i32; 3];
    let (mut nx, mut ny, mut nz) = (0i32, 0i32, 0i32);
    let mut cell2 = [0f32; 6];
    let mut out_file: Vec<u8> = Vec::new();
    let mut array: Vec<f32> = Vec::new();
    let mut sphere_dens1 = [0f32; LIMOBJ];
    let mut sphere_dens2 = [0f32; LIMOBJ];
    let mut sphere_dens3 = [0f32; LIMOBJ];
    let mut sphere_rad1 = [0f32; LIMOBJ];
    let mut sphere_rad2 = [0f32; LIMOBJ];
    let mut cylndr_dens1 = [0f32; LIMOBJ];
    let mut cylndr_dens2 = [0f32; LIMOBJ];
    let mut cylndr_rad1 = [0f32; LIMOBJ];
    let mut cylndr_rad2 = [0f32; LIMOBJ];
    let mut crit_cylndr = [0f32; LIMOBJ];
    let mut crit_sphere = [0f32; LIMOBJ];
    let mut aa = [0f32; LIMOBJ];
    let mut bb = [0f32; LIMOBJ];
    let mut cc = [0f32; LIMOBJ];
    let mut dd = [0f32; LIMOBJ];
    let mut sphere_xcen = [0f32; LIMOBJ];
    let mut sphere_ycen = [0f32; LIMOBJ];
    let mut sphere_zcen = [0f32; LIMOBJ];
    let mut trunc_cyl_x1 = [0f32; LIMOBJ];
    let mut trunc_cyl_y1 = [0f32; LIMOBJ];
    let mut trunc_cyl_z1 = [0f32; LIMOBJ];
    let mut trunc_cyl_x2 = [0f32; LIMOBJ];
    let mut trunc_cyl_y2 = [0f32; LIMOBJ];
    let mut trunc_cyl_z2 = [0f32; LIMOBJ];
    let mut i_type = [0i32; LIMOBJ];
    let mut if_trunc = [0i32; LIMOBJ];
    let num_div: i32;
    let num_div_pix: i32;
    let mut num_sphere: i32 = 0;
    let mut num_cylndr: i32 = 0;
    let mut ind: i32 = 0;
    let (mut x1, mut y1, mut z1, mut x2, mut y2, mut z2) = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
    let mut den: f32;
    let (mut x_mid, mut y_mid, mut z_mid): (f32, f32, f32);
    let mut den_sum: f32;
    let (mut x_div, mut y_div, mut z_div): (f32, f32, f32);
    let mut rad: f32;
    let mut t_cyl: f32;
    let mut p2l_sqr: f32 = 0.;
    let mut dmin: f32 = 1.0e30;
    let mut dmax: f32 = -1.0e30;
    let dmean: f32;
    let (mut x_offset, mut y_offset, mut z_offset) = (0f32, 0f32, 0f32);
    let mut t: f32 = 0.;
    let mut bkgd: f32 = 0.;
    let (mut num_opt_args, mut num_non_opt_args) = (0i32, 0i32);
    let mut num_type: i32 = 0;
    let mut num_radii: i32 = 0;
    let mut num_dens: i32 = 0;
    let mut mode: i32;
    let mut dsum: f64;
    let mut tsum: f64;

    // Fallbacks from    ../manpages/autodoc2man 2 1 fakevolume
    let num_options = 16;
    const OPTIONS: [&[u8]; 16] = [
        b"output:OutputFile:FN:",
        b"size:VolumeSizeInXYZ:IT:",
        b"offsets:OffsetsInXYZ:FT:",
        b"back:BackgroundDensity:F:",
        b"stype:SphereType:IM:",
        b"scen:SphereCenterInXYZ:FTM:",
        b"sradii:SphereRadii:FPM:",
        b"sdens:SphereDensities:FPM:",
        b"trunc:CylinderIsTruncated:IM:",
        b"cstart:CylinderStartInXYZ:FTM:",
        b"cend:CylinderEndInXYZ:FTM:",
        b"cradii:CylinderRadii:FPM:",
        b"cdens:CylinderDensities:FPM:",
        b"mode:ModeToOutput:I:",
        b"param:ParameterFile:PF:",
        b"help:usage:B:",
    ];

    // Startup with fallback
    let argv = arguments
        .iter()
        .map(|value| value.as_bytes().to_vec())
        .collect::<Vec<_>>();
    pip_read_or_parse_options(
        argv.len() as i32,
        &argv,
        &OPTIONS,
        num_options,
        progname.as_bytes(),
        3,
        1,
        1,
        &mut num_opt_args,
        &mut num_non_opt_args,
        Some(imod_usage_header_for_pip),
    );

    //
    num_div = 5;
    num_div_pix = num_div * num_div * num_div;

    if pip_get_in_out_file(b"OutputFile", 0, &mut out_file) != 0 {
        exit_error(b"An output file must be entered");
    }
    //
    // Create output header.
    unsafe {
        iiu_open(2, &String::from_utf8_lossy(&out_file), "NEW");
    }
    if pip_get_three_integers(b"VolumeSizeInXYZ", &mut nx, &mut ny, &mut nz) != 0 {
        exit_error(b"Volume size must be entered");
    }
    if nx < 1 || ny < 1 || nz < 1 || nx as f32 * ny as f32 > 2000000000 as f32 {
        exit_error(b"X, Y, or Z size out of range");
    }

    // `B3DMALLOC(float, nx * ny)`: every element is written before the section
    // is, so the zero fill stands in for `malloc`'s indeterminate contents.
    let count = nx.wrapping_mul(ny) as usize;
    if array.try_reserve_exact(count).is_err() {
        exit_error(b"Allocating array for slice");
    }
    array.resize(count, 0.);

    mode = 2;
    pip_get_integer(b"ModeToOutput", &mut mode);
    if slice_mode_if_real(mode) < 0 {
        exit_error(b"Output mode must be 0, 1, 2, 6, or 12");
    }
    mode = set_float_output_for_entered_mode(mode);
    nxyz[0] = nx;
    nxyz[1] = ny;
    nxyz[2] = nz;
    mxyz[0] = nx;
    mxyz[1] = ny;
    mxyz[2] = nz;
    cell2[0] = nx as f32;
    cell2[1] = ny as f32;
    cell2[2] = nz as f32;
    cell2[3] = 90.;
    cell2[4] = 90.;
    cell2[5] = 90.;
    //
    // `iiuCreateHeader(2, nxyz, mxyz, mode, NULL, 0)`: no labels are passed.
    iiu_create_header(
        2,
        &nxyz,
        &mxyz,
        mode,
        &[[0u8; MRC_LABEL_SIZE]; MRC_NLABELS],
        0,
    );
    iiu_alt_cell(2, &cell2);
    //
    pip_get_three_floats(b"OffsetsInXYZ", &mut x_offset, &mut y_offset, &mut z_offset);
    if pip_get_float(b"BackgroundDensity", &mut bkgd) != 0 {
        exit_error(b"Background density must be entered");
    }

    pip_number_of_entries(b"SphereCenterInXYZ", &mut num_sphere);
    pip_number_of_entries(b"SphereType", &mut num_type);
    pip_number_of_entries(b"SphereRadii", &mut num_radii);
    pip_number_of_entries(b"SphereDensities", &mut num_dens);
    // Deviation: the source writes past its LIMOBJ-sized stack arrays here
    // (see the module comment); refuse instead.
    if num_sphere > LIMOBJ as i32
        || num_type > LIMOBJ as i32
        || num_radii > LIMOBJ as i32
        || num_dens > LIMOBJ as i32
    {
        exit_error(b"Too many sphere entries for the arrays (fakevolume.cpp LIMOBJ = 200)");
    }
    if num_sphere != 0 {
        if num_type != 1 && num_type != num_sphere {
            exit_error(&c_format_bytes(
                "You must enter -stype either once or %d times",
                &[CArg::Int(num_sphere as i64)],
            ));
        }
        if num_radii != 1 && num_radii != num_sphere {
            exit_error(&c_format_bytes(
                "You must enter -sradii either once or %d times",
                &[CArg::Int(num_sphere as i64)],
            ));
        }
        if num_dens != 1 && num_dens != num_sphere {
            exit_error(&c_format_bytes(
                "You must enter -sdens either once or %d times",
                &[CArg::Int(num_sphere as i64)],
            ));
        }

        ind = 0;
        while ind < num_sphere {
            let i = ind as usize;
            pip_get_three_floats(
                b"SphereCenterInXYZ",
                &mut sphere_xcen[i],
                &mut sphere_ycen[i],
                &mut sphere_zcen[i],
            );
            sphere_xcen[i] += x_offset;
            sphere_ycen[i] += y_offset;
            sphere_zcen[i] += z_offset;
            ind += 1;
        }
        ind = 0;
        while ind < num_type {
            let i = ind as usize;
            pip_get_integer(b"SphereType", &mut i_type[i]);
            if ind != 0 && i_type[i] != i_type[i - 1] {
                if num_type != num_sphere || num_radii != num_sphere || num_dens != num_sphere {
                    exit_error(
                        // BUGS.md: the source names the cylinder options `-ctype, -cdens,
                        // -cradii` here; the sphere options are meant.
                        b"You must enter -stype, -sdens, and -sradii for all spheres since they are not all the same type",
                    );
                }
            }
            ind += 1;
        }
        ind = num_type;
        while ind < num_sphere {
            i_type[ind as usize] = i_type[0];
            ind += 1;
        }
        ind = 0;
        while ind < num_radii {
            let i = ind as usize;
            if i_type[i] <= 1 {
                pip_get_float(b"SphereRadii", &mut sphere_rad2[i]);
                sphere_rad1[i] = 0.;
            } else {
                pip_get_two_floats(b"SphereRadii", &mut sphere_rad1[i], &mut sphere_rad2[i]);
            }
            ind += 1;
        }
        ind = num_radii;
        while ind < num_sphere {
            let i = ind as usize;
            sphere_rad1[i] = sphere_rad1[0];
            sphere_rad2[i] = sphere_rad2[0];
            ind += 1;
        }

        ind = 0;
        while ind < num_dens {
            let i = ind as usize;
            if i_type[i] <= 1 {
                pip_get_float(b"SphereDensities", &mut sphere_dens2[i]);
                sphere_dens1[i] = sphere_dens2[i];
                sphere_dens3[i] = sphere_dens2[i];
            } else {
                pip_get_two_floats(
                    b"SphereDensities",
                    &mut sphere_dens1[i],
                    &mut sphere_dens3[i],
                );
                if i_type[i] == 2 {
                    sphere_dens2[i] = sphere_dens1[i];
                } else {
                    sphere_dens2[i] = sphere_dens3[i];
                }
            }
            ind += 1;
        }
        ind = num_dens;
        while ind < num_sphere {
            let i = ind as usize;
            sphere_dens1[i] = sphere_dens1[0];
            sphere_dens2[i] = sphere_dens2[0];
            sphere_dens3[i] = sphere_dens3[0];
            ind += 1;
        }

        ind = 0;
        while ind < num_sphere {
            let i = ind as usize;
            crit_sphere[i] = (sphere_rad2[i] as f64 + 1.5).powf(2.) as f32;
            ind += 1;
        }
    } else if num_type != 0 || num_radii != 0 || num_dens != 0 {
        exit_error(b"You cannot enter -sdens, -sradii, or -stype without entering sphere centers");
    }

    pip_number_of_entries(b"CylinderStartInXYZ", &mut num_cylndr);
    pip_number_of_entries(b"CylinderEndInXYZ", &mut ind);
    if ind != num_cylndr {
        exit_error(b"You must enter the same number of cylinder starts as ends");
    }
    pip_number_of_entries(b"CylinderIsTruncated", &mut num_type);
    pip_number_of_entries(b"CylinderRadii", &mut num_radii);
    pip_number_of_entries(b"CylinderDensities", &mut num_dens);
    // Deviation: past LIMOBJ the source overruns its stack arrays.
    if num_cylndr > LIMOBJ as i32
        || num_type > LIMOBJ as i32
        || num_radii > LIMOBJ as i32
        || num_dens > LIMOBJ as i32
    {
        exit_error(b"Too many cylinder entries for the arrays (fakevolume.cpp LIMOBJ = 200)");
    }
    if num_cylndr != 0 {
        ind = 0;
        while ind < num_cylndr {
            let i = ind as usize;
            pip_get_three_floats(b"CylinderStartInXYZ", &mut x1, &mut y1, &mut z1);
            pip_get_three_floats(b"CylinderEndInXYZ", &mut x2, &mut y2, &mut z2);
            trunc_cyl_x1[i] = x1;
            trunc_cyl_y1[i] = y1;
            trunc_cyl_z1[i] = z1;
            trunc_cyl_x2[i] = x2;
            trunc_cyl_y2[i] = y2;
            trunc_cyl_z2[i] = z2;
            aa[i] = (x2 - x1) / (z2 - z1);
            bb[i] = (x1 + x_offset) - aa[i] * (z1 + z_offset);
            cc[i] = (y2 - y1) / (z2 - z1);
            dd[i] = (y1 + y_offset) - cc[i] * (z1 + z_offset);
            ind += 1;
        }
        ind = 0;
        while ind < num_type {
            pip_get_integer(b"CylinderIsTruncated", &mut if_trunc[ind as usize]);
            ind += 1;
        }
        // BUGS.md, fixed in translation: the source's bound is `numSphere`
        // (fakevolume.cpp:199), so native leaves the flags of cylinders past
        // the entered ones uninitialised.  Like the radii and densities below,
        // the first entry is copied to every remaining cylinder.
        ind = num_type;
        while ind < num_cylndr {
            if_trunc[ind as usize] = if_trunc[0];
            ind += 1;
        }

        ind = 0;
        while ind < num_radii {
            let i = ind as usize;
            pip_get_two_floats(b"CylinderRadii", &mut cylndr_rad1[i], &mut cylndr_rad2[i]);
            crit_cylndr[i] = (cylndr_rad2[i] as f64 + 1.5).powf(2.) as f32;
            ind += 1;
        }
        ind = num_radii;
        while ind < num_cylndr {
            let i = ind as usize;
            cylndr_rad1[i] = cylndr_rad1[0];
            cylndr_rad2[i] = cylndr_rad2[0];
            crit_cylndr[i] = crit_cylndr[0];
            ind += 1;
        }
        ind = 0;
        while ind < num_dens {
            let i = ind as usize;
            pip_get_two_floats(
                b"CylinderDensities",
                &mut cylndr_dens1[i],
                &mut cylndr_dens2[i],
            );
            ind += 1;
        }
        ind = num_dens;
        while ind < num_cylndr {
            let i = ind as usize;
            cylndr_dens1[i] = cylndr_dens1[0];
            cylndr_dens2[i] = cylndr_dens2[0];
            ind += 1;
        }
    } else if num_type != 0 || num_radii != 0 || num_dens != 0 {
        // BUGS.md: the source names a nonexistent `-ctype`; the option is `-trunc`.
        exit_error(b"You cannot enter -cdens, -cradii, or -trunc without entering cylinder points");
    }
    //
    pip_done();
    dsum = 0.;

    // Loop on pixels
    for iz in 1..=nz {
        z_mid = (iz as f64 - 0.5) as f32;
        tsum = 0.;
        for iy in 1..=ny {
            y_mid = (iy as f64 - 0.5) as f32;
            for ix in 1..=nx {
                den = bkgd;
                x_mid = (ix as f64 - 0.5) as f32;

                // Loop on spheres
                for isp in 0..num_sphere as usize {
                    if ((sphere_xcen[isp] - x_mid) as f64).powf(2.)
                        + ((sphere_ycen[isp] - y_mid) as f64).powf(2.)
                        + ((sphere_zcen[isp] - z_mid) as f64).powf(2.)
                        <= crit_sphere[isp] as f64
                    {
                        //
                        // inside criterion radius, now consider  many positions
                        // inside the pixel
                        //
                        den_sum = 0.;
                        for i_dz in 1..=num_div {
                            z_div = ((iz - 1) as f64 + (i_dz as f64 - 0.5) / num_div as f64) as f32;
                            for i_dy in 1..=num_div {
                                y_div =
                                    ((iy - 1) as f64 + (i_dy as f64 - 0.5) / num_div as f64) as f32;
                                for i_dx in 1..=num_div {
                                    x_div = ((ix - 1) as f64 + (i_dx as f64 - 0.5) / num_div as f64)
                                        as f32;
                                    rad = (((sphere_xcen[isp] - x_div) as f64).powf(2.)
                                        + ((sphere_ycen[isp] - y_div) as f64).powf(2.)
                                        + ((sphere_zcen[isp] - z_div) as f64).powf(2.))
                                    .sqrt() as f32;
                                    if rad <= sphere_rad1[isp] {
                                        den_sum += sphere_dens1[isp];
                                    } else if rad <= sphere_rad2[isp] {
                                        den_sum += sphere_dens2[isp]
                                            + (sphere_dens3[isp] - sphere_dens2[isp])
                                                * (rad - sphere_rad1[isp])
                                                / (sphere_rad2[isp] - sphere_rad1[isp]);
                                    }
                                }
                            }
                        }
                        den += den_sum / num_div_pix as f32;
                    }
                }

                // Loop  on cylinders: first untruncated, then truncated
                for i_cyl in 0..num_cylndr as usize {
                    if if_trunc[i_cyl] == 0 {
                        t_cyl = point_to_line(
                            aa[i_cyl], bb[i_cyl], cc[i_cyl], dd[i_cyl], x_mid, y_mid, z_mid,
                        );
                        if ((aa[i_cyl] * t_cyl + bb[i_cyl] - x_mid) as f64).powf(2.)
                            + ((cc[i_cyl] * t_cyl + dd[i_cyl] - y_mid) as f64).powf(2.)
                            + ((t_cyl - z_mid) as f64).powf(2.)
                            <= crit_cylndr[i_cyl] as f64
                        {
                            den_sum = 0.;
                            for i_dz in 1..=num_div {
                                z_div =
                                    ((iz - 1) as f64 + (i_dz as f64 - 0.5) / num_div as f64) as f32;
                                for i_dy in 1..=num_div {
                                    y_div = ((iy - 1) as f64 + (i_dy as f64 - 0.5) / num_div as f64)
                                        as f32;
                                    for i_dx in 1..=num_div {
                                        x_div = ((ix - 1) as f64
                                            + (i_dx as f64 - 0.5) / num_div as f64)
                                            as f32;
                                        t_cyl = point_to_line(
                                            aa[i_cyl], bb[i_cyl], cc[i_cyl], dd[i_cyl], x_div,
                                            y_div, z_div,
                                        );
                                        rad = (((aa[i_cyl] * t_cyl + bb[i_cyl] - x_div) as f64)
                                            .powf(2.)
                                            + ((cc[i_cyl] * t_cyl + dd[i_cyl] - y_div) as f64)
                                                .powf(2.)
                                            + ((t_cyl - z_div) as f64).powf(2.))
                                        .sqrt()
                                            as f32;
                                        if rad <= cylndr_rad1[i_cyl] {
                                            den_sum += cylndr_dens1[i_cyl];
                                        } else if rad <= cylndr_rad2[i_cyl] {
                                            den_sum += cylndr_dens2[i_cyl];
                                        }
                                    }
                                }
                            }
                            den += den_sum / num_div_pix as f32;
                        }
                    } else {
                        //
                        point_line_seg_dist(
                            trunc_cyl_x1[i_cyl],
                            trunc_cyl_y1[i_cyl],
                            trunc_cyl_z1[i_cyl],
                            trunc_cyl_x2[i_cyl],
                            trunc_cyl_y2[i_cyl],
                            trunc_cyl_z2[i_cyl],
                            x_mid,
                            y_mid,
                            z_mid,
                            &mut t,
                            &mut p2l_sqr,
                        );
                        if p2l_sqr <= crit_cylndr[i_cyl] {
                            den_sum = 0.;
                            for i_dz in 1..=num_div {
                                z_div =
                                    ((iz - 1) as f64 + (i_dz as f64 - 0.5) / num_div as f64) as f32;
                                for i_dy in 1..=num_div {
                                    y_div = ((iy - 1) as f64 + (i_dy as f64 - 0.5) / num_div as f64)
                                        as f32;
                                    for i_dx in 1..=num_div {
                                        x_div = ((ix - 1) as f64
                                            + (i_dx as f64 - 0.5) / num_div as f64)
                                            as f32;
                                        point_line_seg_dist(
                                            trunc_cyl_x1[i_cyl],
                                            trunc_cyl_y1[i_cyl],
                                            trunc_cyl_z1[i_cyl],
                                            trunc_cyl_x2[i_cyl],
                                            trunc_cyl_y2[i_cyl],
                                            trunc_cyl_z2[i_cyl],
                                            x_div,
                                            y_div,
                                            z_div,
                                            &mut t,
                                            &mut p2l_sqr,
                                        );
                                        rad = (p2l_sqr as f64).sqrt() as f32;
                                        if rad <= cylndr_rad1[i_cyl] {
                                            den_sum += cylndr_dens1[i_cyl];
                                        } else if rad <= cylndr_rad2[i_cyl] {
                                            den_sum += cylndr_dens2[i_cyl];
                                        }
                                    }
                                }
                            }
                            den += den_sum / num_div_pix as f32;
                        }
                    }
                }
                //
                array[(ix + nx * (iy - 1) - 1) as usize] = den;
                // B3DMIN / B3DMAX: `a < b ? a : b`, second operand on NaN.
                dmin = if dmin < den { dmin } else { den };
                dmax = if dmax > den { dmax } else { den };
                tsum += den as f64;
            }
        }
        dsum += tsum;
        unsafe {
            iiu_write_section(2, array.as_mut_ptr().cast());
        }
    }
    dmean = (dsum / nx.wrapping_mul(ny).wrapping_mul(nz) as f64) as f32;
    iiu_write_header_str(
        2,
        "fakevolume: Volume with synthesized features",
        0,
        dmin,
        dmax,
        dmean,
    );
    unsafe {
        iiu_close(2);
    }
    exit(0);
}

/// `point_to_line` in `fakevolume.cpp:326`.
///
/// The numerator is a `float` sum; `1. + aa * aa + cc * cc` is a `double` sum
/// of `float` products, so the division is in `double` and the return narrows.
pub fn point_to_line(aa: f32, bb: f32, cc: f32, dd: f32, x: f32, y: f32, z: f32) -> f32 {
    ((aa * x + cc * y + z - aa * bb - cc * dd) as f64 / (1. + (aa * aa) as f64 + (cc * cc) as f64))
        as f32
}

/// `pointLineSegDist` in `fakevolume.cpp:338`.
///
/// POINTLINESEGDIST measures the distance from the point X, Y, Z to the
/// line segment from XSTRT, YSTRT, ZSTRT to XEND, YEND, ZEND.  It returns the
/// square of the distance in DISTSQR and the  parameter T specifying the
/// position along the segment of the point of closest approach (between 0 at
/// XSTRT, YSTRT, ZSTRT and 1 at XEND, YEND, ZEND).
///
/// `B3DMAX(0., B3DMIN(1., expr))` compares in `double` and takes the second
/// operand on NaN, so a zero-length segment yields `t = NaN` as in the source.
#[allow(clippy::too_many_arguments)]
pub fn point_line_seg_dist(
    x_strt: f32,
    y_strt: f32,
    z_strt: f32,
    x_end: f32,
    y_end: f32,
    z_end: f32,
    mut x: f32,
    mut y: f32,
    mut z: f32,
    t: &mut f32,
    dist_sqr: &mut f32,
) {
    let aa: f32 = x_end - x_strt;
    let bb: f32 = y_end - y_strt;
    let cc: f32 = z_end - z_strt;
    let inner: f64 = ((aa * (x - x_strt) + bb * (y - y_strt) + cc * (z - z_strt))
        / (aa * aa + bb * bb + cc * cc)) as f64;
    let lower: f64 = if 1. < inner { 1. } else { inner };
    *t = (if 0. > lower { 0. } else { lower }) as f32;
    x -= aa * *t + x_strt;
    y -= bb * *t + y_strt;
    z -= cc * *t + z_strt;
    *dist_sqr = x * x + y * y + z * z;
}
