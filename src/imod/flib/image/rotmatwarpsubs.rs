//! Translation of `IMOD/flib/image/rotmatwarpsubs.f90`.
//!
//! `rotmatwarpsubs.f90` - contains subroutines for Rotatevol, Matchvol, and
//! Warpvol.  Every routine that `use`s the `rotmatwarp` module takes the
//! module as `rmw: &mut RotMatWarp` (see `rotmatwarp.rs`).
//!
//! Array indexing keeps the source's column-major layout and 1-based
//! arithmetic; each access subtracts one at the point of use.  `array` is
//! `array(inputDim(1), inputDim(2), *)` in [`transform_cubes`] and the
//! reshaped `(nxOut, nyOut, *)` view in [`recompose_cubes`].
//!
//! OpenMP: the source parallelises the interpolation over the middle output
//! axis (`!$OMP PARALLEL DO` in `transform_cubes`).  Each iteration computes
//! its own output elements from shared read-only input and stores them into
//! disjoint elements of `brray`, with no reduction, so the result does not
//! depend on the thread count; the translation runs it on a
//! rayon pool of `numOMPthreads(8)` threads (the calling thread alone when
//! that is 1, e.g. under `OMP_NUM_THREADS=1`).

use crate::imod::flib::image::rotmatwarp::{LMCUBE, RotMatWarp};
use crate::imod::flib::subrs::compat::gfortran_rt::{format_f, maxss, minss};
use crate::imod::flib::subrs::hvem::parse_input_params::exit_error;
use crate::imod::flib::subrs::hvem::temp_filename::temp_filename;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen, irdpas, irdsec};
use crate::imod::libcfshr::b3dutil::{
    b3d_output_file_type, b3d_physical_memory, exit, num_omp_threads, override_output_type,
    wall_time,
};
use crate::imod::libcfshr::parse_params::{pip_get_integer, pip_get_three_integers};
use crate::imod::libcfshr::reduce_by_binning::irepak;
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use crate::imod::libiimod::iimage::ii_limited_tile_size;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_NLABELS};
use crate::imod::libiimod::unit_fileio::{
    iiu_alt_chunk_sizes, iiu_close, iiu_set_position, iiu_write_lines, iiu_write_sec_part,
    iiu_write_section,
};
use crate::imod::libiimod::unit_header::{iiu_create_header, iiu_write_header};
use std::io::Write as _;

/// A list-directed (`print *`) `real*4` item, blank separator included, as
/// libgfortran writes it: `F` form with nine significant digits and four
/// trailing blanks for magnitudes in [0.1, 1e9), `E` form otherwise (the
/// runtime's behaviour, as `ccderaser.rs` carries it).
fn list_directed_real(value: f32) -> String {
    if value.is_nan() {
        return format!("{:>17}", "NaN");
    }
    if value.is_infinite() {
        return format!("{:>17}", if value < 0. { "-Infinity" } else { "Infinity" });
    }
    if value == 0. {
        return format!("{:>13}    ", format!("{value:.8}"));
    }
    let scientific = format!("{:.8e}", value.abs());
    let (mantissa, power) = scientific.split_once('e').unwrap();
    let k = power.parse::<i32>().unwrap() + 1;
    if (0..=9).contains(&k) {
        let mut text = format!("{:.*}", (9 - k) as usize, value);
        if k == 9 {
            text.push('.');
        }
        format!("{text:>13}    ")
    } else {
        let e = k - 1;
        format!(
            "{:>17}",
            format!(
                "{}{}E{}{:02}",
                if value < 0. { "-" } else { "" },
                mantissa,
                if e < 0 { '-' } else { '+' },
                e.abs()
            )
        )
    }
}

/// A list-directed `real*8` item: the same rules with seventeen significant
/// digits in a 26-column field.
fn list_directed_double(value: f64) -> String {
    if value.is_nan() {
        return format!("{:>26}", "NaN");
    }
    if value == 0. {
        return format!("{:>22}    ", format!("{value:.16}"));
    }
    let scientific = format!("{:.16e}", value.abs());
    let (mantissa, power) = scientific.split_once('e').unwrap();
    let k = power.parse::<i32>().unwrap() + 1;
    if (0..=17).contains(&k) {
        let mut text = format!("{:.*}", (17 - k) as usize, value);
        if k == 17 {
            text.push('.');
        }
        format!("{text:>22}    ")
    } else {
        let e = k - 1;
        format!(
            "{:>26}",
            format!(
                "{}{}E{}{:03}",
                if value < 0. { "-" } else { "" },
                mantissa,
                if e < 0 { '-' } else { '+' },
                e.abs()
            )
        )
    }
}

/// Original `setup_cubes_scratch` (`rotmatwarpsubs.f90:10`).
///
/// SETUP_CUBES_SCRATCH determines the needed size of the scratch files
/// opens the scratch files, and sets up tables for which cubes use
/// which file.
///
/// `aFwd(3,3)` and `amInv(3,3,numLocTot)` are column major.  `tempExt` is the
/// caller's `character*320` extension, written in place (`tempExt(4:9)` from
/// the time, `tempExt(10:10)` per scratch file); `tim` is `hh:mm:ss`.
pub fn setup_cubes_scratch(
    rmw: &mut RotMatWarp,
    a_fwd: &[f32],
    am_inv: &[f32],
    num_loc_tot: i32,
    n_extra: i32,
    file_in: &str,
    temp_dir: &str,
    temp_ext: &mut [u8],
    tim: &[u8],
    warping: bool,
) {
    let mut nxys_cube_base = [0_i32; 3];
    let mut nxyz_scr = [0_i32; 3];
    let mut nbig_cube = [0_i32; 3];
    let mut num_chunks = [0_i32; 3];
    let mut n_chunk_save = [0_i32; 3];
    let mut in_min = [0_i32; 3];
    let mut in_max = [0_i32; 3];
    let mut efficiency = 0.0_f32;
    let mut cube_efficiency = 0.0_f32;
    let a = |i: usize, j: usize| a_fwd[(i - 1) + 3 * (j - 1)];
    //
    // Parameters controlling aspect ratio limit and search
    let aspect_max = 4.0_f32;
    let rel_efficiency_crit = 0.9_f32;
    let min_yz_size = 200_i32;
    let del_aspect = 0.2_f32;

    let mut mem_size8 = rmw.memory_lim as f64;
    mem_size8 = mem_size8 * 1024. * 1024. / 4.;
    //
    // Find dominant axis for output from elongation of input along X
    // Direction turns out to be counterproductive, so it is not used
    rmw.idir_xaxis = 1;
    if a(2, 1).abs() >= a(1, 1).abs() && a(2, 1).abs() >= a(3, 1).abs() {
        rmw.iout_xaxis = 2;
        if a(2, 1) < 0. {
            rmw.idir_xaxis = -1;
        }
        if a(1, 2).abs() >= a(2, 2).abs() && a(1, 2).abs() >= a(3, 2).abs() {
            rmw.iout_yaxis = 1;
            rmw.iout_zaxis = 3;
        } else {
            rmw.iout_yaxis = 3;
            rmw.iout_zaxis = 1;
        }
    } else if a(3, 1).abs() >= a(1, 1).abs() && a(3, 1).abs() >= a(2, 1).abs() {
        rmw.iout_xaxis = 3;
        if a(3, 1) < 0. {
            rmw.idir_xaxis = -1;
        }
        if a(2, 2).abs() >= a(1, 2).abs() && a(2, 2).abs() >= a(3, 2).abs() {
            rmw.iout_yaxis = 2;
            rmw.iout_zaxis = 1;
        } else {
            rmw.iout_yaxis = 1;
            rmw.iout_zaxis = 2;
        }
    } else {
        rmw.iout_xaxis = 1;
        if a(1, 1) < 0. {
            rmw.idir_xaxis = -1;
        }
        if a(2, 2).abs() >= a(1, 2).abs() && a(2, 2).abs() >= a(3, 2).abs() {
            rmw.iout_yaxis = 2;
            rmw.iout_zaxis = 3;
        } else {
            rmw.iout_yaxis = 3;
            rmw.iout_zaxis = 2;
        }
    }
    let iy_axis = rmw.iout_xaxis % 3 + 1;
    let iz_axis = (rmw.iout_xaxis + 1) % 3 + 1;
    //
    // Do not use multiple slices if 1) Not warping and outer axis = Z or
    // 2) warping and inner axis is not Z
    if (!warping && rmw.iout_zaxis == 3) || (warping && rmw.iout_xaxis != 3) {
        rmw.max_zout = 1;
    }
    //
    // Loop on increasing aspect ratios along that axis
    let mut aspect = 1.0_f32;
    loop {
        find_sizes_for_aspect(
            rmw,
            am_inv,
            num_loc_tot,
            n_extra,
            aspect,
            mem_size8,
            &mut efficiency,
            &mut nxys_cube_base,
        );
        //
        // first time, record efficiency of cube
        if aspect == 1. {
            cube_efficiency = efficiency;
        }
        //
        // If there is only one cube on the output of the X axis, use this ratio
        if rmw.n_cubes[(rmw.iout_xaxis - 1) as usize] == 1 {
            break;
        }
        //
        // If the efficiency has fallen too far, or if the aspect ratio is above
        // a limit AND one of the other axes has become too small, back up to
        // last aspect ratio (if any) and use that one
        let iy = (iy_axis - 1) as usize;
        let iz = (iz_axis - 1) as usize;
        if efficiency / cube_efficiency < rel_efficiency_crit
            || (aspect > aspect_max
                && ((rmw.n_cubes[iy] > 1 && nxys_cube_base[iy] < min_yz_size)
                    || (rmw.n_cubes[iz] > 1 && nxys_cube_base[iz] < min_yz_size)))
        {
            if rmw.i_verbose > 0 {
                println!(
                    " Efficiency ratio{}",
                    list_directed_real(efficiency / cube_efficiency)
                );
            }
            if aspect > 1. {
                aspect -= del_aspect;
                find_sizes_for_aspect(
                    rmw,
                    am_inv,
                    num_loc_tot,
                    n_extra,
                    aspect,
                    mem_size8,
                    &mut efficiency,
                    &mut nxys_cube_base,
                );
            }
            break;
        }
        //
        // Or, loop on next value of aspect ratio
        aspect += del_aspect;
    }

    // Find the true output chunk sizes if chunking
    // Make that be the cube size if there were going to be scratch files, and set up
    // chunk output regardless if there actually are chunks specified by at least one
    // non-default entry; otherwise cancel chunking
    if rmw.chunked_hdf {
        let mut ix = 0;
        // `rotmatwarpsubs.f90:123-124` takes this save as the first statement
        // *inside* the loop, so the restore at `:147` brings back X and Y
        // already clipped (BUGS.md).  Defined behaviour: save the entered
        // sizes once, before the loop, as the restore intends.
        n_chunk_save = rmw.nxyz_chunk;
        for i in 0..3 {
            if rmw.nxyz_chunk[i] > 0 {
                rmw.nxyz_chunk[i] = rmw.nxyz_chunk[i].min(nxys_cube_base[i]);
            } else {
                rmw.nxyz_chunk[i] = nxys_cube_base[i];
                ix += 1;
            }
            ii_limited_tile_size(
                rmw.nxyz_out[i],
                &mut rmw.nxyz_chunk[i],
                &mut num_chunks[i],
                1,
                nxys_cube_base[i],
            );
            if rmw.n_cubes[0] * rmw.n_cubes[1] > 1 {
                nxys_cube_base[i] = rmw.nxyz_chunk[i];
                rmw.n_cubes[i] = num_chunks[i];
            }
        }
        if !file_in.trim_end_matches(' ').is_empty() {
            if rmw.n_cubes[0] * rmw.n_cubes[1] == 1 && ix == 3 {
                rmw.chunked_hdf = false;
            } else if unsafe {
                iiu_alt_chunk_sizes(6, rmw.nxyz_chunk[0], rmw.nxyz_chunk[1], rmw.nxyz_chunk[2])
            } != 0
            {
                exit_error("Setting up chunks in the HDF output file");
            }
        } else {
            rmw.nxyz_chunk = n_chunk_save;
        }
    }
    //
    // Compute true needed input dimensions now that cube sizes are known
    // REPLACE inputDim rather than adjusting it downward, it can be too small
    // But don't let it get any bigger than actual file size
    in_min = [1000000; 3];
    in_max = [-1000000; 3];
    for j in 1..=num_loc_tot {
        let base = 9 * (j - 1) as usize;
        for ix in 0..=1 {
            for iy in 0..=1 {
                for iz in 0..=1 {
                    let xcen = ix as f32 * (nxys_cube_base[0] as f32 + 1.);
                    let ycen = iy as f32 * (nxys_cube_base[1] as f32 + 1.);
                    let zcen = iz as f32 * (nxys_cube_base[2] as f32 + 1.);
                    for i in 0..3 {
                        let ival = (am_inv[base + i] * xcen
                            + am_inv[base + i + 3] * ycen
                            + am_inv[base + i + 6] * zcen)
                            .round() as i32;
                        // Adding 2 instead of 1 may be overkill but gives extra safety
                        in_min[i] = in_min[i].min(ival - 2);
                        in_max[i] = in_max[i].max(ival + 2);
                    }
                }
            }
        }
    }
    for i in 0..3 {
        rmw.input_dim[i] = rmw.nxyz_in[i].min(n_extra + in_max[i] + 1 - in_min[i]);
    }
    if rmw.i_verbose > 0 {
        println!(
            "Input size adjusted for actual cubes:{:6}{:6}{:6}",
            rmw.input_dim[0], rmw.input_dim[1], rmw.input_dim[2]
        );
    }
    rmw.array_size8 = rmw.input_dim[0] as f64;
    rmw.array_size8 = (rmw.array_size8 * rmw.input_dim[1] as f64) * rmw.input_dim[2] as f64;
    let mut nz_extra = 0_i32;
    //
    // now compute sizes of nearly equal sized near cubes to fill output
    // volume, store the starting index coordinates
    //
    for i in 0..3 {
        if rmw.n_cubes[i] > LMCUBE {
            if !file_in.trim_end_matches(' ').is_empty() || n_extra < 100 {
                exit_error("Too many cubes in longest direction to fit in arrays");
            }
            exit_error(
                "Too many cubes for arrays, check for wild vectors/big jumps between transforms",
            );
        }

        nbig_cube[i] = rmw.nxyz_out[i] % rmw.n_cubes[i];
        let mut ind = 0;
        for j in 1..=rmw.n_cubes[i] {
            let ju = (j - 1) as usize;
            rmw.ixyz_cube[ju][i] = ind;
            rmw.nxyz_cube[ju][i] = (rmw.nxyz_out[i] - ind).min(nxys_cube_base[i]);
            if j <= nbig_cube[i] && !rmw.chunked_hdf {
                rmw.nxyz_cube[ju][i] += 1;
            }
            ind += rmw.nxyz_cube[ju][i];
        }
        nxyz_scr[i] = nxys_cube_base[i] + 1;
    }
    //
    // This was originally a test that could fail - and it failed for nz = 1, so
    // increase the number of planes allocated as needed to hold output row
    while (rmw.nxyz_out[0] as f32 * (nxys_cube_base[1] as f32 + 1.)) as f64 >= rmw.array_size8 {
        nz_extra += 1;
        rmw.array_size8 = rmw.input_dim[0] as f64;
        rmw.array_size8 =
            (rmw.array_size8 * rmw.input_dim[1] as f64) * (rmw.input_dim[2] + nz_extra) as f64;
        if rmw.array_size8 > 2. * mem_size8 {
            println!(
                "{:>12}{:>12}{:>12}{:>12}{:>12}{}{}{:>12}{}",
                rmw.input_dim[0],
                rmw.input_dim[1],
                rmw.input_dim[2],
                rmw.nxyz_out[0],
                nxys_cube_base[1],
                list_directed_real(rmw.nxyz_out[0] as f32 * (nxys_cube_base[1] as f32 + 1.)),
                list_directed_double(rmw.array_size8),
                nz_extra,
                list_directed_double(mem_size8)
            );
            exit_error("Something is wrong trying to make array big enough for output");
        }
    }
    //
    if file_in.trim_end_matches(' ').is_empty() {
        return;
    }
    //
    // Allocate memory
    // A Fortran extent below zero allocates a zero-size array (as happens
    // native's `-memory 1` became 0 through warpvol's 5% reserve and every
    // size here went negative; warpvol now keeps it at 1, BUGS.md); `brray`'s single extent is the `integer*4`
    // product.
    let extent = |n: i64| n.max(0) as usize;
    let n_array = extent(rmw.input_dim[0] as i64)
        * extent(rmw.input_dim[1] as i64)
        * extent((rmw.input_dim[2] + nz_extra) as i64);
    let n_brray = extent(
        rmw.idim_out[0]
            .wrapping_mul(rmw.idim_out[1])
            .wrapping_mul(rmw.max_zout) as i64,
    );
    let n_ifile = extent(rmw.n_cubes[0] as i64) * extent(rmw.n_cubes[1] as i64);
    let n_iz = n_ifile * extent(nxyz_scr[2] as i64);
    let mut failed = false;
    let mut array: Vec<f32> = Vec::new();
    let mut brray: Vec<f32> = Vec::new();
    failed |= array.try_reserve_exact(n_array).is_err();
    failed |= brray.try_reserve_exact(n_brray).is_err();
    if failed {
        exit_error("Failed to allocate memory for arrays");
    }
    array.resize(n_array, 0.);
    brray.resize(n_brray, 0.);
    rmw.array = array;
    rmw.brray = brray;
    rmw.ifile = vec![0; n_ifile];
    rmw.iz_in_file = vec![0; n_iz];
    rmw.iz_in_file_dim = [rmw.n_cubes[0], rmw.n_cubes[1], nxyz_scr[2]];
    //
    // Set limits of inner and outer loops depending on axis
    if rmw.iout_xaxis == 1 {
        rmw.lim_inner = rmw.n_cubes[0];
        rmw.lim_outer = rmw.n_cubes[1];
    } else {
        rmw.lim_inner = rmw.n_cubes[1];
        rmw.lim_outer = rmw.n_cubes[0];
    }
    //
    // get an array of file numbers for each cube in X/Y plane
    // If only one cube in plane, set to final output file
    rmw.need_scratch = [false; 4];
    for ix in 1..=rmw.n_cubes[0] {
        for iy in 1..=rmw.n_cubes[1] {
            let k = ((ix - 1) + rmw.n_cubes[0] * (iy - 1)) as usize;
            rmw.ifile[k] = 1;
            if ix > nbig_cube[0] {
                rmw.ifile[k] += 1;
            }
            if iy > nbig_cube[1] {
                rmw.ifile[k] += 2;
            }
            if rmw.n_cubes[0] * rmw.n_cubes[1] == 1 || rmw.chunked_hdf {
                rmw.ifile[k] = 6;
                if ix == 1 && iy == 1 {
                    if !rmw.chunked_hdf {
                        println!(" Writing directly to output file");
                    }
                    if rmw.chunked_hdf {
                        println!(
                            "Writing directly to output file in chunks of{:6} x{:5} x{:5}",
                            rmw.nxyz_chunk[0], rmw.nxyz_chunk[1], rmw.nxyz_chunk[2]
                        );
                    }
                }
            } else {
                rmw.need_scratch[(rmw.ifile[k] - 1) as usize] = true;
            }
        }
    }

    println!(
        " Rotations done in{:3} layers, with{:3} by{:3} cubes in each layer",
        rmw.n_cubes[2], rmw.n_cubes[0], rmw.n_cubes[1]
    );
    //
    // If needed, open scratch files with 4 different sizes.
    // compose temporary filenames from the time
    //
    temp_ext[3..5].copy_from_slice(&tim[0..2]);
    temp_ext[5..7].copy_from_slice(&tim[3..5]);
    temp_ext[7..9].copy_from_slice(&tim[6..8]);
    let ext = |temp_ext: &[u8]| String::from_utf8_lossy(temp_ext).into_owned();
    let mut temp_name = temp_filename(file_in, temp_dir, &ext(temp_ext));
    //
    // `title` at this point: the module's label, all zero bytes until the
    // program writes it after this call.
    let mut labels = [[0_u8; MRC_LABEL_SIZE]; MRC_NLABELS];
    labels[0] = rmw.title;
    nxyz_scr[2] = nxyz_scr[2] * rmw.n_cubes[0] * rmw.n_cubes[1];
    ialprt(false);
    if rmw.need_scratch[0] {
        imopen(1, &temp_name, "scratch");
        iiu_create_header(1, &nxyz_scr, &nxyz_scr, rmw.mode, &labels, 0);
    }
    //
    temp_ext[9] = b'2';
    temp_name = temp_filename(file_in, temp_dir, &ext(temp_ext));
    nxyz_scr[0] -= 1;
    if rmw.need_scratch[1] {
        imopen(2, &temp_name, "scratch");
        iiu_create_header(2, &nxyz_scr, &nxyz_scr, rmw.mode, &labels, 0);
    }
    //
    temp_ext[9] = b'4';
    temp_name = temp_filename(file_in, temp_dir, &ext(temp_ext));
    nxyz_scr[1] -= 1;
    if rmw.need_scratch[3] {
        imopen(4, &temp_name, "scratch");
        iiu_create_header(4, &nxyz_scr, &nxyz_scr, rmw.mode, &labels, 0);
    }
    //
    temp_ext[9] = b'3';
    temp_name = temp_filename(file_in, temp_dir, &ext(temp_ext));
    nxyz_scr[0] += 1;
    if rmw.need_scratch[2] {
        imopen(3, &temp_name, "scratch");
        iiu_create_header(3, &nxyz_scr, &nxyz_scr, rmw.mode, &labels, 0);
    }
}

/// Original internal routine `findSizesForAspect` of `setup_cubes_scratch`
/// (`rotmatwarpsubs.f90:303`).
///
/// INTERNAL routine FINDSIZESFORASPECT to find sizes of input/output given an
/// aspect ratio, adjusting memory as needed for output slices.
///
/// The host variables it reads (`amInv`, `numLocTot`, `nExtra`, `aspect`,
/// `memSize8`) are parameters here; the ones it sets (`efficiency`,
/// `nxysCubeBase`) are `&mut`.  It also assigns the host's loop variables
/// `i`, `j`, `ix`, `iy`, which the host reassigns before any further use, so
/// they are locals.
pub fn find_sizes_for_aspect(
    rmw: &mut RotMatWarp,
    am_inv: &[f32],
    num_loc_tot: i32,
    n_extra: i32,
    aspect: f32,
    mem_size8: f64,
    efficiency: &mut f32,
    nxys_cube_base: &mut [i32; 3],
) {
    let mut dev_max = [0.0_f32; 3];
    let mut dev_fac = [1.0_f32; 3];
    let mut expand_fac: f32;
    //
    // Find the maximum deviation in each direction of input required to
    // output an elongated cube
    dev_fac[(rmw.iout_xaxis - 1) as usize] = aspect;
    for j in 1..=num_loc_tot {
        let base = 9 * (j - 1) as usize;
        for ix in [-1_i32, 1] {
            for iy in [-1_i32, 1] {
                for i in 0..3 {
                    let value = (am_inv[base + i] * ix as f32 * dev_fac[0]
                        + am_inv[base + i + 3] * iy as f32 * dev_fac[1]
                        + am_inv[base + i + 6] * dev_fac[2])
                        .abs();
                    // gfortran `MAX` of reals; the deviations are finite.
                    dev_max[i] = dev_max[i].max(value);
                }
            }
        }
    }

    // Get the fraction of input data used in output
    *efficiency = aspect / (dev_max[0] * dev_max[1] * dev_max[2]);
    if rmw.i_verbose > 0 {
        println!(
            "Aspect:{} efficiency{} deviations:{}{}{}",
            format_f(aspect as f64, 6, 2),
            format_f(*efficiency as f64, 7, 4),
            format_f(dev_max[0] as f64, 10, 5),
            format_f(dev_max[1] as f64, 10, 5),
            format_f(dev_max[2] as f64, 10, 5)
        );
    }
    //
    // Find factor by which input can be expanded to use up the memory
    // But first get an initial estimate of the output size to allow a
    // reduction for memory needed for the multiple slices
    // of output to be stored; set up to iterate with better output size
    let mut num_iter_reduce = 1;
    let mut mem_reduced = mem_size8;
    // `**0.333333`: a real*8 base with a default-real exponent, which gfortran
    // widens to real*8 for `pow`.
    let third = 0.333333_f32 as f64;
    if rmw.max_zout > 1 {
        expand_fac = (mem_size8 / (dev_max[0] * dev_max[1] * dev_max[2]) as f64).powf(third) as f32;
        rmw.idim_out[0] = rmw.nxyz_out[0].min((dev_fac[0] * expand_fac) as i32);
        rmw.idim_out[1] = rmw.nxyz_out[1].min((dev_fac[1] * expand_fac) as i32);
        num_iter_reduce = 3;
    }
    //
    // Iterate to stabilize memory division between input and output
    for iter_reduce in 1..=num_iter_reduce {
        if rmw.max_zout > 1 {
            //
            // Reduce by the amount needed for output, but do not decrease the
            // reduction from previous time, and keep it to half the total
            let out_mem = rmw.max_zout as f32 * rmw.idim_out[0] as f32 * rmw.idim_out[1] as f32;
            mem_reduced = (mem_size8 / 2.).max(mem_reduced.min(mem_size8 - out_mem as f64));
            if rmw.i_verbose > 0 {
                println!(
                    "Iteration{:3}: for multiple output slices, input memory reduced to{:7} MB",
                    iter_reduce,
                    (mem_reduced / (1024. * 256.)).round() as i32
                );
            }
        }
        //
        // get expansion factor that would use up the memory
        // and get trial size for equal expansion
        expand_fac =
            (mem_reduced / (dev_max[0] * dev_max[1] * dev_max[2]) as f64).powf(third) as f32;
        for i in 0..3 {
            rmw.input_dim[i] = (dev_max[i] * expand_fac + 1.) as i32;
        }
        //
        // Count axes where this is big enough for whole volume, and keep track
        // of at least one axis where it is and is not
        let mut num_over = 0;
        let mut ind_under = 1_i32;
        let mut ind_over = 1_i32;
        for i in 1..=3 {
            if rmw.input_dim[i - 1] >= rmw.nxyz_in[i - 1] {
                num_over += 1;
                ind_over = i as i32;
            } else {
                ind_under = i as i32;
            }
        }
        if rmw.i_verbose > 0 {
            println!(
                "Initial fac{}{:8}{:8}{:8} numover{:3}",
                format_f(expand_fac as f64, 9, 2),
                rmw.input_dim[0],
                rmw.input_dim[1],
                rmw.input_dim[2],
                num_over
            );
        }
        //
        // If this is oversized in one axis, expand the other two axes and
        // re-evaluate if they are over
        let io = (ind_over - 1) as usize;
        if num_over == 1 && rmw.input_dim[io] > rmw.nxyz_in[io] {
            expand_fac *= (rmw.input_dim[io] as f32 / rmw.nxyz_in[io] as f32).sqrt();
            rmw.input_dim[io] = rmw.nxyz_in[io];
            for i in 0..=1 {
                let ix = ((ind_over + i) % 3 + 1) as usize;
                rmw.input_dim[ix - 1] = (dev_max[ix - 1] * expand_fac + 1.) as i32;
                if rmw.input_dim[ix - 1] >= rmw.nxyz_in[ix - 1] {
                    num_over += 1;
                } else {
                    ind_under = ix as i32;
                }
            }
            if rmw.i_verbose > 0 {
                println!(
                    "Expand 2 axes{}{:8}{:8}{:8} numover{:3}",
                    format_f(expand_fac as f64, 9, 2),
                    rmw.input_dim[0],
                    rmw.input_dim[1],
                    rmw.input_dim[2],
                    num_over
                );
            }
        }
        //
        // Now if it is oversized in two axes, expand the remaining axis
        if num_over == 2 {
            let ix = (ind_under % 3 + 1) as usize;
            let iy = ((ind_under + 1) % 3 + 1) as usize;
            let iu = (ind_under - 1) as usize;
            rmw.input_dim[ix - 1] = rmw.nxyz_in[ix - 1];
            rmw.input_dim[iy - 1] = rmw.nxyz_in[iy - 1];
            expand_fac = (((mem_reduced / rmw.nxyz_in[ix - 1] as f64) / rmw.nxyz_in[iy - 1] as f64)
                / dev_max[iu] as f64) as f32;
            rmw.input_dim[iu] = (dev_max[iu] * expand_fac + 1.) as i32;
            if rmw.input_dim[iu] >= rmw.nxyz_in[iu] {
                num_over = 3;
            }
            if rmw.i_verbose > 0 {
                println!(
                    "Expand 1 axis{}{:8}{:8}{:8} numover{:3}",
                    format_f(expand_fac as f64, 9, 2),
                    rmw.input_dim[0],
                    rmw.input_dim[1],
                    rmw.input_dim[2],
                    num_over
                );
            }
        }
        //
        // If over in all axes, set up input and output sizes
        if num_over == 3 {
            rmw.input_dim = rmw.nxyz_in;
            rmw.idim_out = rmw.nxyz_out;
        } else {
            //
            // Otherwise, set up the output sizes by expanding the elongated cube
            for ix in 0..3 {
                rmw.idim_out[ix] =
                    rmw.nxyz_out[ix].min((dev_fac[ix] * expand_fac) as i32 - 2 * n_extra - 2);
            }
        }
        //
        // Get number of cubes and basic cube size on each axis, adjust out size
        for i in 0..3 {
            rmw.n_cubes[i] = (rmw.nxyz_out[i] - 1) / rmw.idim_out[i] + 1;
            nxys_cube_base[i] = rmw.nxyz_out[i] / rmw.n_cubes[i];
        }
        if rmw.i_verbose > 0 {
            println!(
                "Initial out{:6}{:6}{:6}  # cubes{:4}{:4}{:4}  size{:6}{:6}{:6}",
                rmw.idim_out[0],
                rmw.idim_out[1],
                rmw.idim_out[2],
                rmw.n_cubes[0],
                rmw.n_cubes[1],
                rmw.n_cubes[2],
                nxys_cube_base[0],
                nxys_cube_base[1],
                nxys_cube_base[2]
            );
        }
        for i in 0..3 {
            rmw.idim_out[i] = nxys_cube_base[i] + 1;
        }
    }
}

/// Original `setMemoryLimitAndHdfChunks` (`rotmatwarpsubs.f90:437`).
///
/// setMemoryLimitAndHdfChunks sets the memory limit according to the default
/// and entered option, then checks if the chunk option has been entered, and
/// if so sets up for an HDF output file and sets the memory limit to account
/// for the chunks size limit.
///
/// `physMem/0./` and `ierr/1/` are `SAVE`d, but the routine runs once per
/// program, so they start at their initial values here.
pub fn set_memory_limit_and_hdf_chunks(rmw: &mut RotMatWarp, pip_input: bool) {
    let mut ierr: i32 = 1;

    if pip_input {
        ierr = pip_get_integer(b"MemoryLimit", &mut rmw.memory_lim);
    }
    let mut phys_mem = b3d_physical_memory();
    if ierr > 0 && phys_mem > 0. {
        phys_mem /= 1024.0_f64.powi(2);
        // `0.6`, `12000.`, `0.4` are default reals widened to real*8.
        if phys_mem < 30000. {
            rmw.memory_lim = 768.0_f64.max(
                (0.6_f32 as f64 * phys_mem)
                    .min(12000.)
                    .min(phys_mem - 1000.),
            ) as i32;
        } else {
            rmw.memory_lim = (0.4_f32 as f64 * phys_mem) as i32;
        }
    }
    if !pip_input {
        return;
    }

    // If chunks, limit size to 4 GPixel as the easiest way to keep chunks legal
    {
        let [x, y, z] = &mut rmw.nxyz_chunk;
        let _ = pip_get_three_integers(b"ChunkSizesForHDF", x, y, z);
    }
    let [nx_chunk, ny_chunk, nz_chunk] = rmw.nxyz_chunk;
    if nx_chunk >= 0 || ny_chunk >= 0 || nz_chunk >= 0 {
        if nx_chunk < 0 || ny_chunk < 0 || nz_chunk < 0 {
            exit_error(
                "You cannot enter a negative chunk size; enter 0 for default size on an axis",
            );
        }
        rmw.memory_lim = rmw.memory_lim.min(15000);
        override_output_type(5);
        rmw.chunked_hdf = true;
        if b3d_output_file_type() != 5 {
            exit_error("Chunks cannot be used; this package was not built with HDF support");
        }
    }
}

/// Original `transform_cubes` (`rotmatwarpsubs.f90:474`).
///
/// TRANSFORM_CUBES does the complete work of looping on all of the cubes,
/// transforming them, writing them to files, and calling the routine to
/// reassemble the output file.  It is used by Rotatevol and Matchvol.  It
/// ends the program (`call exit(0)`).
pub fn transform_cubes(rmw: &mut RotMatWarp, interp_order: i32) -> ! {
    // The OpenMP team for the interpolation loop, made on first use.
    let mut pool: Option<rayon::ThreadPool> = None;
    let mut iz_sec = [0_i32; 6];
    let mut in_min = [0_i32; 3];
    let mut in_max = [0_i32; 3];
    let mut icube = [0_i32; 3];
    let mut istride = [0_i32; 3];
    let (mut ix_cube, mut iy_cube) = (0_i32, 0_i32);
    let mut iz_end = 0_i32;
    //
    let mut dmin = 1.0e20_f32;
    let mut dmax = -dmin;
    let mut tsum = 0.0_f32;
    let mut num_done = 0;
    let mut wall_cum = 0.0_f64;
    istride[0] = 1;
    istride[1] = rmw.idim_out[0];
    istride[2] = rmw.idim_out[0] * rmw.idim_out[1];
    let inner_stride = istride[(rmw.iout_xaxis - 1) as usize];
    let middle_stride = istride[(rmw.iout_yaxis - 1) as usize];
    let iouter_stride = istride[(rmw.iout_zaxis - 1) as usize];
    let xa = (rmw.iout_xaxis - 1) as usize;
    let ya = (rmw.iout_yaxis - 1) as usize;
    let za = (rmw.iout_zaxis - 1) as usize;
    let a_inv = rmw.a_inv;
    let ainv = |i: usize, j: usize| a_inv[(i - 1) + 3 * (j - 1)];
    //
    // loop on layers of cubes in Z, do all I/O to complete layer
    //
    iz_sec[5] = 0;
    for iz_cube in 1..=rmw.n_cubes[2] {
        icube[2] = iz_cube;
        //
        // initialize files and counters
        //
        for i in 1..=4 {
            if rmw.need_scratch[i - 1] {
                unsafe { iiu_set_position(i as i32, 0, 0) };
            }
            iz_sec[i - 1] = 0;
        }
        //
        // loop on the cubes in the layer
        //
        for ind_outer in 1..=rmw.lim_outer {
            for ind_inner in 1..=rmw.lim_inner {
                cube_indexes(
                    rmw.iout_xaxis,
                    ind_inner,
                    ind_outer,
                    &mut ix_cube,
                    &mut iy_cube,
                );
                icube[0] = ix_cube;
                icube[1] = iy_cube;
                let cx = rmw.ixyz_cube[(ix_cube - 1) as usize];
                let nx = rmw.nxyz_cube[(ix_cube - 1) as usize];
                let cy = rmw.ixyz_cube[(iy_cube - 1) as usize];
                let ny = rmw.nxyz_cube[(iy_cube - 1) as usize];
                let cz = rmw.ixyz_cube[(iz_cube - 1) as usize];
                let nz = rmw.nxyz_cube[(iz_cube - 1) as usize];
                //
                // back-transform the corner coordinates of the output cube to
                // find the limiting index coordinates of the input cube
                //
                for i in 0..3 {
                    in_min[i] = 1000000;
                    in_max[i] = -1000000;
                }
                for indx in 0..=1 {
                    for indy in 0..=1 {
                        for indz in 0..=1 {
                            let xcen = (cx[0] + indx * nx[0]) as f32 - rmw.cxyz_out[0];
                            let ycen = (cy[1] + indy * ny[1]) as f32 - rmw.cxyz_out[1];
                            let zcen = (cz[2] + indz * nz[2]) as f32 - rmw.cxyz_out[2];
                            for i in 1..=3 {
                                let ival = (ainv(i, 1) * xcen
                                    + ainv(i, 2) * ycen
                                    + ainv(i, 3) * zcen
                                    + rmw.cxyz_in[i - 1])
                                    .round() as i32;
                                in_min[i - 1] = 0.max(in_min[i - 1].min(ival - 1));
                                in_max[i - 1] =
                                    (rmw.nxyz_in[i - 1] - 1).min(in_max[i - 1].max(ival + 1));
                            }
                        }
                    }
                }
                let mut ifempty = 0;
                for i in 0..3 {
                    if in_min[i] > in_max[i] {
                        ifempty = 1;
                    }
                    if in_max[i] + 1 - in_min[i] > rmw.input_dim[i] {
                        println!(
                            "\nERROR: Transform_cubes - input data larger than calculated size:{:2}{:6}{:6}{:6}",
                            i + 1,
                            rmw.input_dim[i],
                            in_min[i],
                            in_max[i]
                        );
                        exit(1);
                    }
                }
                //
                // load the input cube
                //
                let plane = rmw.input_dim[0] as usize * rmw.input_dim[1] as usize;
                if ifempty == 0 {
                    for iz in in_min[2]..=in_max[2] {
                        unsafe { iiu_set_position(5, iz, 0) };
                        if rmw.i_verbose > 1 {
                            println!(
                                "{:5}{:5}{:5}{:5}{:5}{:5}{:5}{:5}{:5}",
                                ix_cube,
                                iy_cube,
                                iz_cube,
                                iz,
                                iz + 1 - in_min[2],
                                in_min[0],
                                in_max[0],
                                in_min[1],
                                in_max[1]
                            );
                        }
                        let offset = (iz - in_min[2]) as usize * plane;
                        if unsafe {
                            irdpas(
                                5,
                                &mut rmw.array[offset..],
                                rmw.input_dim[0],
                                rmw.input_dim[1],
                                in_min[0],
                                in_max[0],
                                in_min[1],
                                in_max[1],
                            )
                        }
                        .is_err()
                        {
                            exit_error("Reading file");
                        }
                    }
                }
                //
                // prepare offsets and limits
                //
                let x_offs_out =
                    (rmw.ixyz_cube[(icube[xa] - 1) as usize][xa] - 1) as f32 - rmw.cxyz_out[xa];
                let x_offs_in = rmw.cxyz_in[0] + 1. - in_min[0] as f32;
                let y_offs_in = rmw.cxyz_in[1] + 1. - in_min[1] as f32;
                let z_offs_in = rmw.cxyz_in[2] + 1. - in_min[2] as f32;
                let ix_limit = in_max[0] + 1 - in_min[0];
                let iy_limit = in_max[1] + 1 - in_min[1];
                let iz_limit = in_max[2] + 1 - in_min[2];
                //
                // loop over Z segments of the output cube
                let mut iz_start = 1;
                let mut num_zleft = nz[2];
                while num_zleft > 0 {
                    let num_zto_do = num_zleft.min(rmw.max_zout);
                    iz_end = iz_start + num_zto_do - 1;
                    let wall_start;
                    if ifempty == 0 {
                        // Set up index limits for inner loop
                        let (inner_start, inner_end) = if rmw.iout_xaxis == 3 {
                            (iz_start, iz_end)
                        } else {
                            (1, rmw.nxyz_cube[(icube[xa] - 1) as usize][xa])
                        };
                        //
                        // Set up limits for middle loop
                        let (middle_start, middle_end) = if rmw.iout_yaxis == 3 {
                            (iz_start, iz_end)
                        } else {
                            (1, rmw.nxyz_cube[(icube[ya] - 1) as usize][ya])
                        };
                        //
                        // Set up limits for outer loop
                        let (iouter_start, iouter_end) = if rmw.iout_zaxis == 3 {
                            (iz_start, iz_end)
                        } else {
                            (1, rmw.nxyz_cube[(icube[za] - 1) as usize][za])
                        };
                        let ind_base =
                            -rmw.idim_out[0] - iz_start * rmw.idim_out[0] * rmw.idim_out[1];
                        wall_start = wall_time();
                        //
                        // There is no good way to base this on loop size, 8 threads is good for a
                        // a large volume, less for small one
                        let num_threads = num_omp_threads(8);
                        let dim1 = rmw.input_dim[0] as usize;
                        let dim12 = plane;
                        let dmean_in = rmw.dmean_in;
                        let array = &rmw.array;
                        // `!$OMP PARALLEL DO` over `middle`: iteration `middle` stores
                        // only `brray(indBase + inner*innerStride + middle*middleStride +
                        // iouter*iouterStride)`, and the three strides are those of
                        // three different axes of the `idimOut` box with every index
                        // inside its extent, so distinct `middle` store distinct
                        // elements; everything else it uses is read-only.  The rows can
                        // therefore run on any number of threads with the same result.
                        struct Out(*mut f32, usize);
                        unsafe impl Sync for Out {}
                        unsafe impl Send for Out {}
                        let out = Out(rmw.brray.as_mut_ptr(), rmw.brray.len());
                        // Every read below is at `(x, y, z)` with `1 <= x <= ixLimit`,
                        // `1 <= y <= iyLimit`, `1 <= z <= izLimit` (the loads are guarded by
                        // the range test and the `+1`/`-1` neighbours are clamped into it), so
                        // one check of the three limits against the allocated shape bounds
                        // every index; after it the reads need no per-element check.  The
                        // source reads out of its array if the limits ever exceeded the
                        // shape; here that stops the program instead.
                        assert!(
                            ix_limit as usize <= dim1
                                && (iy_limit as usize) * dim1 <= dim12
                                && (iz_limit.max(1) as usize) * dim12 <= array.len(),
                            "input region larger than the allocated array"
                        );
                        let arr = move |x: i32, y: i32, z: i32| -> f32 {
                            // SAFETY: bounded by the assertion above (see its comment).
                            unsafe {
                                *array.get_unchecked(
                                    (x - 1) as usize
                                        + dim1 * (y - 1) as usize
                                        + dim12 * (z - 1) as usize,
                                )
                            }
                        };
                        //
                        // Start looping over outer axis of this Z segment; parallelize
                        // loop over middle and inner axes
                        for iouter in iouter_start..=iouter_end {
                            let cen_outer = (rmw.ixyz_cube[(icube[za] - 1) as usize][za] + iouter
                                - 1) as f32
                                - rmw.cxyz_out[za];
                            let row = |middle: i32| {
                                let out = &out;
                                // Copy the loop invariants into locals: read through
                                // the closure's captured references they would be
                                // reloaded after every store through `out`, which
                                // may alias them as far as the compiler knows.
                                let (arr, ix_limit, iy_limit, iz_limit, dmean_in) =
                                    (arr, ix_limit, iy_limit, iz_limit, dmean_in);
                                let (ind_base, inner_stride, middle_stride, iouter_stride) =
                                    (ind_base, inner_stride, middle_stride, iouter_stride);
                                let (inner_start, inner_end, x_offs_out) =
                                    (inner_start, inner_end, x_offs_out);
                                let cen_mid = (rmw.ixyz_cube[(icube[ya] - 1) as usize][ya] + middle
                                    - 1) as f32
                                    - rmw.cxyz_out[ya];
                                let xp_offset = ainv(1, rmw.iout_yaxis as usize) * cen_mid
                                    + ainv(1, rmw.iout_zaxis as usize) * cen_outer
                                    + x_offs_in;
                                let yp_offset = ainv(2, rmw.iout_yaxis as usize) * cen_mid
                                    + ainv(2, rmw.iout_zaxis as usize) * cen_outer
                                    + y_offs_in;
                                let zp_offset = ainv(3, rmw.iout_yaxis as usize) * cen_mid
                                    + ainv(3, rmw.iout_zaxis as usize) * cen_outer
                                    + z_offs_in;
                                let a1x = ainv(1, rmw.iout_xaxis as usize);
                                let a2x = ainv(2, rmw.iout_xaxis as usize);
                                let a3x = ainv(3, rmw.iout_xaxis as usize);
                                // One check per row bounds every store below: the
                                // output index is increasing in `inner` (`innerStride`
                                // is positive), so checking the two ends of the row
                                // covers every element between them.
                                if inner_start <= inner_end {
                                    let ind = |inner: i32| {
                                        ind_base as i64
                                            + inner as i64 * inner_stride as i64
                                            + middle as i64 * middle_stride as i64
                                            + iouter as i64 * iouter_stride as i64
                                            - 1
                                    };
                                    assert!(
                                        inner_stride > 0
                                            && ind(inner_start) >= 0
                                            && ind(inner_end) < out.1 as i64,
                                        "transform_cubes: row outside the output array"
                                    );
                                }
                                if interp_order >= 2 {
                                    // The triquadratic loop runs out of line, with every
                                    // invariant passed by value, so they stay in
                                    // registers across the store through `out`; the
                                    // body is unchanged.
                                    //
                                    // # Safety
                                    // Every output index of the row must lie in
                                    // the `out` allocation (the row assertion
                                    // above), and `arr` must accept every in-limit
                                    // position (the array-shape assertion).
                                    #[inline(never)]
                                    unsafe fn quad_row<A: Fn(i32, i32, i32) -> f32>(
                                        arr: A,
                                        out: *mut f32,
                                        (inner_start, inner_end, x_offs_out): (i32, i32, f32),
                                        (a1x, a2x, a3x): (f32, f32, f32),
                                        (xp_offset, yp_offset, zp_offset): (f32, f32, f32),
                                        (ix_limit, iy_limit, iz_limit): (i32, i32, i32),
                                        dmean_in: f32,
                                        (ind_base, inner_stride, middle, middle_stride): (
                                            i32,
                                            i32,
                                            i32,
                                            i32,
                                        ),
                                        (iouter, iouter_stride): (i32, i32),
                                    ) {
                                        for inner in inner_start..=inner_end {
                                            let cen_inner = inner as f32 + x_offs_out;
                                            //
                                            // get indices in array of input data
                                            //
                                            let xp = a1x * cen_inner + xp_offset;
                                            let yp = a2x * cen_inner + yp_offset;
                                            let zp = a3x * cen_inner + zp_offset;
                                            let mut bval = dmean_in;
                                            //
                                            // do triquadratic interpolation with higher-order
                                            // terms omitted
                                            //
                                            let ixp = xp.round() as i32;
                                            let iyp = yp.round() as i32;
                                            let izp = zp.round() as i32;
                                            if ixp >= 1
                                                && ixp <= ix_limit
                                                && iyp >= 1
                                                && iyp <= iy_limit
                                                && izp >= 1
                                                && izp <= iz_limit
                                            {
                                                let dx = xp - ixp as f32;
                                                let dy = yp - iyp as f32;
                                                let dz = zp - izp as f32;
                                                let ixp_p1 = ix_limit.min(ixp + 1);
                                                let iyp_p1 = iy_limit.min(iyp + 1);
                                                let izp_p1 = iz_limit.min(izp + 1);
                                                let ixp_m1 = 1.max(ixp - 1);
                                                let iyp_m1 = 1.max(iyp - 1);
                                                let izp_m1 = 1.max(izp - 1);
                                                //
                                                // Set up terms for quadratic interpolation
                                                //
                                                let dx_sq = dx * dx;
                                                let dy_sq = dy * dy;
                                                let dz_sq = dz * dz;
                                                let fx = 1. - dx_sq;
                                                let fx_m1 = 0.5 * (dx_sq - dx);
                                                let fx_p1 = fx_m1 + dx;
                                                let fy = 1. - dy_sq;
                                                let fy_m1 = 0.5 * (dy_sq - dy);
                                                let fy_p1 = fy_m1 + dy;
                                                let fz = 1. - dz_sq;
                                                let fz_m1 = 0.5 * (dz_sq - dz);
                                                let fz_p1 = fz_m1 + dz;

                                                bval = fz_m1
                                                    * (fy_m1
                                                        * (fx_m1 * arr(ixp_m1, iyp_m1, izp_m1)
                                                            + fx * arr(ixp, iyp_m1, izp_m1)
                                                            + fx_p1 * arr(ixp_p1, iyp_m1, izp_m1))
                                                        + fy * (fx_m1 * arr(ixp_m1, iyp, izp_m1)
                                                            + fx * arr(ixp, iyp, izp_m1)
                                                            + fx_p1 * arr(ixp_p1, iyp, izp_m1))
                                                        + fy_p1
                                                            * (fx_m1
                                                                * arr(ixp_m1, iyp_p1, izp_m1)
                                                                + fx * arr(ixp, iyp_p1, izp_m1)
                                                                + fx_p1
                                                                    * arr(ixp_p1, iyp_p1, izp_m1)))
                                                    + fz * (fy_m1
                                                        * (fx_m1 * arr(ixp_m1, iyp_m1, izp)
                                                            + fx * arr(ixp, iyp_m1, izp)
                                                            + fx_p1 * arr(ixp_p1, iyp_m1, izp))
                                                        + fy * (fx_m1 * arr(ixp_m1, iyp, izp)
                                                            + fx * arr(ixp, iyp, izp)
                                                            + fx_p1 * arr(ixp_p1, iyp, izp))
                                                        + fy_p1
                                                            * (fx_m1 * arr(ixp_m1, iyp_p1, izp)
                                                                + fx * arr(ixp, iyp_p1, izp)
                                                                + fx_p1
                                                                    * arr(ixp_p1, iyp_p1, izp)))
                                                    + fz_p1
                                                        * (fy_m1
                                                            * (fx_m1
                                                                * arr(ixp_m1, iyp_m1, izp_p1)
                                                                + fx * arr(ixp, iyp_m1, izp_p1)
                                                                + fx_p1
                                                                    * arr(ixp_p1, iyp_m1, izp_p1))
                                                            + fy * (fx_m1
                                                                * arr(ixp_m1, iyp, izp_p1)
                                                                + fx * arr(ixp, iyp, izp_p1)
                                                                + fx_p1
                                                                    * arr(ixp_p1, iyp, izp_p1))
                                                            + fy_p1
                                                                * (fx_m1
                                                                    * arr(ixp_m1, iyp_p1, izp_p1)
                                                                    + fx * arr(
                                                                        ixp, iyp_p1, izp_p1,
                                                                    )
                                                                    + fx_p1
                                                                        * arr(
                                                                            ixp_p1, iyp_p1, izp_p1,
                                                                        )));
                                            }
                                            let index = (ind_base
                                                + inner * inner_stride
                                                + middle * middle_stride
                                                + iouter * iouter_stride
                                                - 1)
                                                as usize;
                                            // SAFETY: in bounds by the row assertion; distinct
                                            // `middle` store to distinct elements (see `out`).
                                            unsafe { *out.add(index) = bval };
                                        }
                                    }
                                    // SAFETY: see `quad_row`; both assertions hold here.
                                    unsafe {
                                        quad_row(
                                            arr,
                                            out.0,
                                            (inner_start, inner_end, x_offs_out),
                                            (a1x, a2x, a3x),
                                            (xp_offset, yp_offset, zp_offset),
                                            (ix_limit, iy_limit, iz_limit),
                                            dmean_in,
                                            (ind_base, inner_stride, middle, middle_stride),
                                            (iouter, iouter_stride),
                                        )
                                    };
                                } else {
                                    //
                                    // linear interpolation
                                    //
                                    for inner in inner_start..=inner_end {
                                        let cen_inner = inner as f32 + x_offs_out;
                                        //
                                        // get indices in array of input data
                                        //
                                        let xp = a1x * cen_inner + xp_offset;
                                        let yp = a2x * cen_inner + yp_offset;
                                        let zp = a3x * cen_inner + zp_offset;
                                        let mut bval = dmean_in;
                                        let ixp = xp as i32;
                                        let iyp = yp as i32;
                                        let izp = zp as i32;
                                        if ixp >= 1
                                            && ixp <= ix_limit
                                            && iyp >= 1
                                            && iyp <= iy_limit
                                            && izp >= 1
                                            && izp <= iz_limit
                                        {
                                            let dx = xp - ixp as f32;
                                            let dy = yp - iyp as f32;
                                            let dz = zp - izp as f32;
                                            let ixp_p1 = ix_limit.min(ixp + 1);
                                            let iyp_p1 = iy_limit.min(iyp + 1);
                                            let izp_p1 = iz_limit.min(izp + 1);
                                            //
                                            // Set up terms for linear interpolation
                                            //
                                            let d11 = (1. - dx) * (1. - dy);
                                            let d12 = (1. - dx) * dy;
                                            let d21 = dx * (1. - dy);
                                            let d22 = dx * dy;
                                            bval = (1. - dz)
                                                * (d11 * arr(ixp, iyp, izp)
                                                    + d12 * arr(ixp, iyp_p1, izp)
                                                    + d21 * arr(ixp_p1, iyp, izp)
                                                    + d22 * arr(ixp_p1, iyp_p1, izp))
                                                + dz * (d11 * arr(ixp, iyp, izp_p1)
                                                    + d12 * arr(ixp, iyp_p1, izp_p1)
                                                    + d21 * arr(ixp_p1, iyp, izp_p1)
                                                    + d22 * arr(ixp_p1, iyp_p1, izp_p1));
                                        }
                                        let index = (ind_base
                                            + inner * inner_stride
                                            + middle * middle_stride
                                            + iouter * iouter_stride
                                            - 1)
                                            as usize;
                                        // SAFETY: in bounds by the row assertion; distinct
                                        // `middle` store to distinct elements (see `out`).
                                        unsafe { *out.0.add(index) = bval };
                                    }
                                }
                            };
                            if num_threads > 1 {
                                let pool = pool.get_or_insert_with(|| {
                                    rayon::ThreadPoolBuilder::new()
                                        .num_threads(num_threads as usize)
                                        .build()
                                        .unwrap()
                                });
                                pool.install(|| {
                                    use rayon::prelude::*;
                                    (middle_start..=middle_end).into_par_iter().for_each(&row)
                                });
                            } else {
                                for middle in middle_start..=middle_end {
                                    row(middle);
                                }
                            }
                        }
                    } else {
                        wall_start = wall_time();
                        let count = (istride[2] * num_zto_do) as usize;
                        rmw.brray[..count].fill(rmw.dmean_in);
                    }
                    wall_cum += wall_time() - wall_start;
                    write_one_cube(
                        rmw,
                        ix_cube,
                        iy_cube,
                        num_zto_do,
                        iz_start,
                        &mut iz_sec,
                        &mut dmin,
                        &mut dmax,
                        &mut tsum,
                    );
                    //
                    // End of do while: update start and number left to do
                    iz_start = iz_end + 1;
                    num_zleft -= num_zto_do;
                }
                num_done += 1;
                println!(
                    "Finished{:6} of{:6}",
                    num_done,
                    rmw.n_cubes[0] * rmw.n_cubes[1] * rmw.n_cubes[2]
                );
                let _ = std::io::stdout().flush();
            }
        }
        //
        // whole layer of cubes in z is done.  now reread and compose one
        // row of the output section at a time in array
        //
        recompose_cubes(rmw, iz_cube, &mut dmin, &mut dmax, &mut tsum);
        if rmw.chunked_hdf {
            iz_sec[5] += iz_end;
        }
    }
    if rmw.i_verbose > 0 {
        println!("Wall time {}", format_f(wall_cum, 10, 4));
    }
    //
    let dmean = tsum / rmw.nxyz_out[2] as f32;
    iiu_write_header(6, &rmw.title, 1, dmin, dmax, dmean);
    for i in 1..=4 {
        if rmw.need_scratch[i - 1] {
            unsafe { iiu_close(i as i32) };
        }
    }
    unsafe {
        iiu_close(5);
        iiu_close(6);
    }
    exit(0);
}

/// Original `cubeIndexes` (`rotmatwarpsubs.f90:804`).
///
/// CUBEINDEXES Gets X and Y cube indexes from inner and outer loop indexes
/// If X becomes X on output, then make X the inner loop
/// If X becomes Y, then make X outer and Y inner, of course
/// If X becomes Z, it helps a little to make X outer and Y inner
pub fn cube_indexes(
    iout_xaxis: i32,
    ind_inner: i32,
    ind_outer: i32,
    ix_cube: &mut i32,
    iy_cube: &mut i32,
) {
    if iout_xaxis == 1 {
        *ix_cube = ind_inner;
        *iy_cube = ind_outer;
    } else {
        *iy_cube = ind_inner;
        *ix_cube = ind_outer;
    }
}

/// Original `writeOneCube` (`rotmatwarpsubs.f90:821`).
///
/// writeOneCube writes a set of sections to one cube at the given index,
/// maintaining the min/max/mean.
///
/// `dmin = min(dmin, tmin)` and `dmax = max(dmax, tmax)` are `minss`/`maxss`
/// with the running value as destination in the reference object.
/// `irepak(brray(ix), brray(ix), ...)` repacks in place; the region is copied
/// out first (as `binvol.rs` does) because the repack reads every element at
/// an index >= the one it writes.
pub fn write_one_cube(
    rmw: &mut RotMatWarp,
    ix_cube: i32,
    iy_cube: i32,
    num_zto_do: i32,
    iz_start: i32,
    iz_sec: &mut [i32; 6],
    dmin: &mut f32,
    dmax: &mut f32,
    tsum: &mut f32,
) {
    let (mut tmin, mut tmax, mut tmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let k = ((ix_cube - 1) + rmw.n_cubes[0] * (iy_cube - 1)) as usize;
    let iunit = rmw.ifile[k] as i32;
    let ncx = rmw.nxyz_cube[(ix_cube - 1) as usize][0];
    let ncy = rmw.nxyz_cube[(iy_cube - 1) as usize][1];
    let plane = rmw.idim_out[0] as usize * rmw.idim_out[1] as usize;
    let mut source = vec![0_u8; plane * 4];
    for iz in 0..num_zto_do {
        let ix = iz as usize * plane;
        if iunit == 6 {
            array_min_max_mean_fortran(
                &rmw.brray[ix..],
                &rmw.idim_out[0],
                &rmw.idim_out[1],
                &1,
                &ncx,
                &1,
                &ncy,
                &mut tmin,
                &mut tmax,
                &mut tmean,
            );
            *dmin = minss(*dmin, tmin);
            *dmax = maxss(*dmax, tmax);
            *tsum += tmean;
        }
        for (bytes, value) in source
            .chunks_exact_mut(4)
            .zip(rmw.brray[ix..ix + plane].iter())
        {
            bytes.copy_from_slice(&value.to_ne_bytes());
        }
        {
            let region = &mut rmw.brray[ix..];
            let destination = unsafe {
                core::slice::from_raw_parts_mut(region.as_mut_ptr().cast::<u8>(), region.len() * 4)
            };
            irepak(
                destination,
                &source,
                &rmw.idim_out[0],
                &rmw.idim_out[1],
                &0,
                &(ncx - 1),
                &0,
                &(ncy - 1),
            );
        }
        if rmw.chunked_hdf && rmw.n_cubes[0] * rmw.n_cubes[1] > 1 {
            unsafe {
                iiu_set_position(
                    iunit,
                    iz_sec[(iunit - 1) as usize] + iz + iz_start - 1,
                    rmw.ixyz_cube[(iy_cube - 1) as usize][1],
                );
                let x0 = rmw.ixyz_cube[(ix_cube - 1) as usize][0];
                iiu_write_sec_part(
                    iunit,
                    rmw.brray[ix..].as_mut_ptr().cast(),
                    ncx,
                    0,
                    x0,
                    x0 + ncx - 1,
                    0,
                    ncy - 1,
                );
            }
        } else {
            unsafe { iiu_write_section(iunit, rmw.brray[ix..].as_mut_ptr().cast()) };
            let d = rmw.iz_in_file_dim;
            let kz = (ix_cube - 1) + d[0] * ((iy_cube - 1) + d[1] * (iz + iz_start - 1));
            rmw.iz_in_file[kz as usize] = iz_sec[(iunit - 1) as usize];
            iz_sec[(iunit - 1) as usize] += 1;
        }
    }
}

/// Original `recompose_cubes` (`rotmatwarpsubs.f90:856`).
///
/// RECOMPOSE_CUBES takes a layer of cubes in Z out of the scratch files and
/// writes it to the output file.
///
/// The running min/max are `minss`/`maxss` with the running value as
/// destination in the reference object, as in [`write_one_cube`].
pub fn recompose_cubes(
    rmw: &mut RotMatWarp,
    iz_cube: i32,
    dmin: &mut f32,
    dmax: &mut f32,
    tsum: &mut f32,
) {
    let (mut tmin, mut tmax, mut tmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut ix_cube, mut iy_cube) = (0_i32, 0_i32);
    let nx_out = rmw.nxyz_out[0];
    let ny_out = rmw.nxyz_out[1];
    //
    // whole layer of cubes in z is done.  Find our how many planes can be
    // recomposed at once
    if rmw.n_cubes[0] * rmw.n_cubes[1] == 1 || rmw.chunked_hdf {
        return;
    }
    let max_planes = ((rmw.array_size8 / nx_out as f64) / ny_out as f64) as i32;
    let d = rmw.iz_in_file_dim;
    let nz_in_cube = rmw.nxyz_cube[(iz_cube - 1) as usize][2];
    if max_planes < 2 {
        //
        // If only one plane fits, there is no advantage to doing it all at
        // once, so reread and compose one row of the output section at a time
        // in array
        //
        for iz in 1..=nz_in_cube {
            let mut temp_sum = 0.0_f32;
            for iy_cube in 1..=rmw.n_cubes[1] {
                let n_lines_out = rmw.nxyz_cube[(iy_cube - 1) as usize][1];
                for ix_cube in 1..=rmw.n_cubes[0] {
                    let k = ((ix_cube - 1) + rmw.n_cubes[0] * (iy_cube - 1)) as usize;
                    let iunit = rmw.ifile[k] as i32;
                    let iz_sec = rmw.iz_in_file
                        [((ix_cube - 1) + d[0] * ((iy_cube - 1) + d[1] * (iz - 1))) as usize];
                    unsafe { iiu_set_position(iunit, iz_sec, 0) };
                    if unsafe { irdsec(iunit, &mut rmw.brray) }.is_err() {
                        exit_error("Reading file");
                    }
                    pack_piece(
                        &mut rmw.array,
                        nx_out,
                        ny_out,
                        rmw.ixyz_cube[(ix_cube - 1) as usize][0],
                        0,
                        &rmw.brray,
                        rmw.nxyz_cube[(ix_cube - 1) as usize][0],
                        n_lines_out,
                    );
                }
                array_min_max_mean_fortran(
                    &rmw.array,
                    &nx_out,
                    &n_lines_out,
                    &1,
                    &nx_out,
                    &1,
                    &n_lines_out,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
                *dmin = minss(*dmin, tmin);
                *dmax = maxss(*dmax, tmax);
                temp_sum += tmean * n_lines_out as f32;
                unsafe { iiu_write_lines(6, rmw.array.as_mut_ptr().cast(), n_lines_out) };
            }
            *tsum += temp_sum / ny_out as f32;
        }
    } else {
        //
        // Otherwise, do entire planes so that multiple successive slices can
        // be read from the cubes
        let mut num_left = nz_in_cube;
        let mut iz_start = 1;
        while num_left > 0 {
            let num_to_read = max_planes.min(num_left);
            num_left -= num_to_read;
            //
            // read this set of planes from each cube in turn
            for ind_outer in 1..=rmw.lim_outer {
                for ind_inner in 1..=rmw.lim_inner {
                    cube_indexes(
                        rmw.iout_xaxis,
                        ind_inner,
                        ind_outer,
                        &mut ix_cube,
                        &mut iy_cube,
                    );
                    let k = ((ix_cube - 1) + rmw.n_cubes[0] * (iy_cube - 1)) as usize;
                    let iunit = rmw.ifile[k] as i32;
                    for iz in 1..=num_to_read {
                        let iz_sec = rmw.iz_in_file[((ix_cube - 1)
                            + d[0] * ((iy_cube - 1) + d[1] * (iz + iz_start - 2)))
                            as usize];
                        unsafe { iiu_set_position(iunit, iz_sec, 0) };
                        if unsafe { irdsec(iunit, &mut rmw.brray) }.is_err() {
                            exit_error("Reading file");
                        }
                        pack_piece_3d(
                            &mut rmw.array,
                            nx_out,
                            ny_out,
                            max_planes,
                            rmw.ixyz_cube[(ix_cube - 1) as usize][0],
                            rmw.ixyz_cube[(iy_cube - 1) as usize][1],
                            iz,
                            &rmw.brray,
                            rmw.nxyz_cube[(ix_cube - 1) as usize][0],
                            rmw.nxyz_cube[(iy_cube - 1) as usize][1],
                        );
                    }
                }
            }
            //
            // Write the planes and maintain MMM
            for iz in 1..=num_to_read {
                write_full_plane(
                    &mut rmw.array,
                    nx_out,
                    ny_out,
                    max_planes,
                    iz,
                    dmin,
                    dmax,
                    tsum,
                );
            }
            iz_start += num_to_read;
        }
    }
}

/// Original `pack_piece` (`rotmatwarpsubs.f90:938`).
///
/// PACK_PIECE packs the contents of brray into appropriate region of array.
pub fn pack_piece(
    array: &mut [f32],
    ix_dim_out: i32,
    _iy_dim_out: i32,
    ix_offset: i32,
    iy_offset: i32,
    brray: &[f32],
    nx_in: i32,
    ny_in: i32,
) {
    for iy in 1..=ny_in {
        for ix in 1..=nx_in {
            array[(ix + ix_offset - 1) as usize
                + ix_dim_out as usize * (iy + iy_offset - 1) as usize] =
                brray[(ix - 1) as usize + nx_in as usize * (iy - 1) as usize];
        }
    }
}

/// Original `pack_piece3D` (`rotmatwarpsubs.f90:955`).
///
/// PACK_PIECE3D packs the contents of brray into appropriate region and
/// slice of the 3D array.
pub fn pack_piece_3d(
    array: &mut [f32],
    ix_dim_out: i32,
    iy_dim_out: i32,
    _iz_dim_out: i32,
    ix_offset: i32,
    iy_offset: i32,
    iz: i32,
    brray: &[f32],
    nx_in: i32,
    ny_in: i32,
) {
    let plane = ix_dim_out as usize * iy_dim_out as usize;
    for iy in 1..=ny_in {
        for ix in 1..=nx_in {
            array[(ix + ix_offset - 1) as usize
                + ix_dim_out as usize * (iy + iy_offset - 1) as usize
                + plane * (iz - 1) as usize] =
                brray[(ix - 1) as usize + nx_in as usize * (iy - 1) as usize];
        }
    }
}

/// Original `writeFullPlane` (`rotmatwarpsubs.f90:973`).
///
/// WRITEFULLPLANE calculates and maintains density MMM and writes a plane.
pub fn write_full_plane(
    array: &mut [f32],
    nx_out: i32,
    ny_out: i32,
    _max_planes: i32,
    iz: i32,
    dmin: &mut f32,
    dmax: &mut f32,
    tsum: &mut f32,
) {
    let (mut tmin, mut tmax, mut tmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let offset = nx_out as usize * ny_out as usize * (iz - 1) as usize;
    array_min_max_mean_fortran(
        &array[offset..],
        &nx_out,
        &ny_out,
        &1,
        &nx_out,
        &1,
        &ny_out,
        &mut tmin,
        &mut tmax,
        &mut tmean,
    );
    *dmin = minss(*dmin, tmin);
    *dmax = maxss(*dmax, tmax);
    *tsum += tmean;
    unsafe { iiu_write_section(6, array[offset..].as_mut_ptr().cast()) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BUGS.md: `rotmatwarpsubs.f90:123-124` saved the chunk sizes inside the
    /// clipping loop, so a provisional call (blank file name) restored X and Y
    /// already clipped to the cube size.  The defined behaviour restores the
    /// sizes as entered.
    #[test]
    fn provisional_call_restores_entered_chunk_sizes() {
        let mut rmw = RotMatWarp::default();
        rmw.nxyz_in = [600, 600, 600];
        rmw.nxyz_out = [600, 600, 600];
        rmw.memory_lim = 1;
        rmw.chunked_hdf = true;
        rmw.nxyz_chunk = [500, 450, -1];
        let ident = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let mut temp_ext = [b' '; 320];
        setup_cubes_scratch(
            &mut rmw,
            &ident,
            &ident,
            1,
            0,
            "",
            "",
            &mut temp_ext,
            b"12:00:00",
            false,
        );
        assert!(rmw.n_cubes[0] * rmw.n_cubes[1] > 1, "{:?}", rmw.n_cubes);
        assert_eq!(rmw.nxyz_chunk, [500, 450, -1]);
    }
}
