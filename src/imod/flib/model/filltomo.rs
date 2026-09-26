//! Translation of `IMOD/flib/model/filltomo.f90`.
//!
//! FILLTOMO improves a combined tomogram from a two-axis tilt series by
//! replacing pixels in locations where the "matching" tomogram had no data
//! with the values from the tomogram that was matched to.  It determines a
//! linear scaling between the latter tomogram and the combined tomogram so
//! that the intensities will match as well as possible.
//!
//! The main program maps to [`filltomo`] and the subroutine
//! `getBoundaryLimits` to [`get_boundary_limits`].
//!
//! The source's `!$OMP PARALLEL` region over Y lines is translated as its
//! one-thread execution: each thread keeps its own fill count and min/max,
//! combined under `!$OMP CRITICAL`.  Every pixel is written independently and
//! the count is an integer sum, so only the min/max could depend on the
//! partition, and only when a filled value is NaN.

use crate::imod::flib::model::get_region_contours::get_region_contours;
use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::{format_f, maxss, minss};
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, list_read};
use crate::imod::flib::subrs::hvem::get_nxyz::get_nxyz;
use crate::imod::flib::subrs::hvem::inside::inside;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen, irdlin, irdsec};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall2;
use crate::imod::libcfshr::b3dutil::{exit, num_omp_threads};
use crate::imod::libcfshr::linearxforms::xfapply;
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_integer, pip_get_three_integers, pip_get_two_integers,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::simplestat::avg_sd;
use crate::imod::libiimod::mrcfiles::MRC_LABEL_SIZE;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position, iiu_write_section};
use crate::imod::libiimod::unit_header::iiu_write_header;
use std::io::{BufRead, BufReader, Write};

/// `parameter (maxSample = 100000, ...)` (`filltomo.f90:19`).
const MAX_SAMPLE: i32 = 100000;
/// `parameter (..., limThreads = 8)` (`filltomo.f90:19`).
const LIM_THREADS: i32 = 8;
/// `parameter (numOptions = 15)` (`filltomo.f90:48`).
const NUM_OPTIONS: i32 = 15;
/// Fallback PIP table, the `options(1)` string (`filltomo.f90:50-57`).
const OPTIONS: &str = "matched:MatchedToTomogram:FN:@fill:FillTomogram:FN:@\
source:SourceTomogram:CH:@inverse:InverseTransformFile:FN:@\
xfill:LeftRightFill:IP:@yfill:BottomTopFill:IP:@:ImagesAreBinned:I:@\
sraw:SourceRawStackSize:IT:@sxform:SourceStackTransforms:FN:@\
sboundary:SourceBoundaryModel:FN:@mraw:MatchedToRawStackSize:IT:@\
mxform:MatchedToStackTransforms:FN:@mboundary:MatchedToBoundaryModel:FN:@\
param:ParameterFile:PF:@help:usage:B:";

/// A `read(5,*)` / `read(1,*)` with no `END=`/`ERR=`: a failure is the
/// gfortran runtime error, status 2.
fn list_read_or_die<R: BufRead>(unit: &mut R, items: &mut [ListItem]) {
    if list_read(unit, items).is_err() {
        eprintln!("Fortran runtime error: End of file");
        exit(2);
    }
}

/// Original program `filltomo` (`filltomo.f90:16`).
pub fn filltomo() {
    unsafe {
        let mut nxyz = [0_i32; 3];
        let mut mxyz = [0_i32; 3];
        let mut nxyz_in = [0_i32; 3];
        let mut nxyz_out = [0_i32; 3];
        let mut cxyz_in = [0f32; 3];
        let mut cxyz_out = [0f32; 3];
        // minv(i, j) is minv[i - 1][j - 1].
        let mut minv = [[0f32; 3]; 3];
        let mut avg = [0f32; 2];
        let mut sd = [0f32; 2];
        let mut sem = 0f32;
        let mut mode = 0_i32;
        let (mut dmin_out, mut dmax_out, mut dmean_out) = (0f32, 0f32, 0f32);
        let (mut dmin, mut dmax, mut dmean) = (0f32, 0f32, 0f32);
        let mut in_file = String::new();
        let (mut num_opt_arg, mut num_non_opt_arg) = (0, 0);
        //
        // Pip startup: set error, parse options, check help, set flag if used
        pip_read_or_parse_options(
            &[OPTIONS],
            NUM_OPTIONS,
            "filltomo",
            "ERROR: FILLTOMO - ",
            true,
            3,
            1,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
        );
        let pipinput = num_opt_arg + num_non_opt_arg > 0;
        if pip_get_in_out_file(
            "FillTomogram",
            2,
            "Name of tomogram file to fill",
            &mut in_file,
            320,
        ) != 0
        {
            exit_error("No tomogram file to fill specified");
        }
        imopen(1, &in_file, "old");
        irdhdr(
            1,
            nxyz_out.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin_out,
            &mut dmax_out,
            &mut dmean_out,
        );

        // `filltomo.f90:69` asks for non-option argument 2 here, the same
        // slot as `FillTomogram`, so native `filltomo a.rec b.mat` fills
        // `b.mat` from itself.  Fixed in translation (BUGS.md): the autodoc
        // documents the matched-to tomogram as the *first* non-option
        // argument, so slot 1 is used.
        if pip_get_in_out_file(
            "MatchedToTomogram",
            1,
            "Name of tomogram that was matched TO",
            &mut in_file,
            320,
        ) != 0
        {
            exit_error("No matched-to tomogram file specified");
        }
        imopen(2, &in_file, "ro");
        irdhdr(
            2,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
        let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);
        if nx != nxyz_out[0] || ny != nxyz_out[1] || nz != nxyz_out[2] {
            exit_error("Dimensions do not match between matched-to file and file being filled");
        }
        let idim = nx.wrapping_mul(ny);
        let mut array: Vec<f32> = Vec::new();
        let mut brray: Vec<f32> = Vec::new();
        let mut i = 0;
        if array.try_reserve_exact(idim.max(0) as usize).is_err()
            || brray
                .try_reserve_exact(idim.max(2 * MAX_SAMPLE).max(0) as usize)
                .is_err()
        {
            i = 1;
        }
        memory_error(i, "arrays for image");
        array.resize(idim.max(0) as usize, 0.0);
        brray.resize(idim.max(2 * MAX_SAMPLE).max(0) as usize, 0.0);
        //
        if !pipinput {
            // print *, three character items: list-directed output writes
            // adjacent character items with no separator.
            println!(
                " Enter either the X, Y and Z dimensions of the tomogram that wastransformed to match, or the name of that file"
            );
        }
        if pipinput {
            let mut record = [b' '; 320];
            if pipgetstring_(b"SourceTomogram", &mut record) != 0 {
                exit_error("No source tomogram file entered");
            }
            in_file = crate::imod::libcfshr::b3dutil::fortran_string(&record);
            ialprt(false);
            imopen(3, &in_file, "ro");
            irdhdr(
                3,
                nxyz_in.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &mut mode,
                &mut dmin,
                &mut dmax,
                &mut dmean,
            );
        } else {
            get_nxyz(false, " ", "FILLTOMO", 5, &mut nxyz_in);
        }
        //
        if pipinput {
            let mut record = [b' '; 320];
            if pipgetstring_(b"InverseTransformFile", &mut record) != 0 {
                exit_error("No inverse transform file entered");
            }
            in_file = crate::imod::libcfshr::b3dutil::fortran_string(&record);
        } else {
            println!(" Enter name of file containing inverse transformation used by MATCHVOL");
            let _ = std::io::stdout().flush();
            let mut line = String::new();
            if matches!(std::io::stdin().read_line(&mut line), Ok(0) | Err(_)) {
                eprintln!("Fortran runtime error: End of file");
                exit(2);
            }
            let line = line.trim_end_matches(['\r', '\n']);
            in_file = line.chars().take(320).collect::<String>();
        }
        {
            let file = dopen(1, in_file.trim_end_matches(' '), "ro", "f");
            let mut unit1 = BufReader::new(file);
            let [m1, m2, m3] = &mut minv;
            let [c1, c2, c3] = &mut cxyz_in;
            let [m11, m12, m13] = m1;
            let [m21, m22, m23] = m2;
            let [m31, m32, m33] = m3;
            list_read_or_die(
                &mut unit1,
                &mut [
                    ListItem::Real(m11),
                    ListItem::Real(m12),
                    ListItem::Real(m13),
                    ListItem::Real(c1),
                    ListItem::Real(m21),
                    ListItem::Real(m22),
                    ListItem::Real(m23),
                    ListItem::Real(c2),
                    ListItem::Real(m31),
                    ListItem::Real(m32),
                    ListItem::Real(m33),
                    ListItem::Real(c3),
                ],
            );
        }
        for i in 0..3 {
            cxyz_in[i] += nxyz_in[i] as f32 / 2.0;
            cxyz_out[i] = nxyz_out[i] as f32 / 2.0;
        }
        let (nx_in, ny_in, nz_in) = (nxyz_in[0], nxyz_in[1], nxyz_in[2]);
        let (cen_x_in, cen_y_in, cen_z_in) = (cxyz_in[0], cxyz_in[1], cxyz_in[2]);
        let (cen_x_out, cen_y_out, cen_z_out) = (cxyz_out[0], cxyz_out[1], cxyz_out[2]);
        //
        let mut n_left_fill = 0;
        let mut n_right_fill = 0;
        let mut n_bot_fill = 0;
        let mut n_top_fill = 0;
        let mut ibinning = 1;
        if pipinput {
            let _ = pip_get_two_integers(b"LeftRightFill", &mut n_left_fill, &mut n_right_fill);
            let _ = pip_get_two_integers(b"BottomTopFill", &mut n_bot_fill, &mut n_top_fill);
            let _ = pip_get_integer(b"ImagesAreBinned", &mut ibinning);
        } else {
            print!(
                " #s of pixels on left, right, bottom, and top \n(Y in flipped tomogram) to fill regardless: "
            );
            let _ = std::io::stdout().flush();
            let stdin = std::io::stdin();
            let mut lock = stdin.lock();
            list_read_or_die(
                &mut lock,
                &mut [
                    ListItem::Integer(&mut n_left_fill),
                    ListItem::Integer(&mut n_right_fill),
                    ListItem::Integer(&mut n_bot_fill),
                    ListItem::Integer(&mut n_top_fill),
                ],
            );
        }
        //
        // Make arrays for the limits in source and matched volumes
        let mut ierr = 0;
        let mut ix_src_start: Vec<i32> = Vec::new();
        let mut ix_src_end: Vec<i32> = Vec::new();
        let mut ix_mat_start: Vec<i32> = Vec::new();
        let mut ix_mat_end: Vec<i32> = Vec::new();
        if ix_src_start
            .try_reserve_exact(nz_in.max(0) as usize)
            .is_err()
            || ix_src_end.try_reserve_exact(nz_in.max(0) as usize).is_err()
            || ix_mat_start.try_reserve_exact(nz.max(0) as usize).is_err()
            || ix_mat_end.try_reserve_exact(nz.max(0) as usize).is_err()
        {
            ierr = 1;
        }
        memory_error(ierr, "arrays for boundaries in X");
        ix_src_start.resize(nz_in.max(0) as usize, 0);
        ix_src_end.resize(nz_in.max(0) as usize, 0);
        ix_mat_start.resize(nz.max(0) as usize, 0);
        ix_mat_end.resize(nz.max(0) as usize, 0);
        let mut if_src_limits = 0;
        let mut if_mat_limits = 0;
        get_boundary_limits(
            pipinput,
            ibinning,
            "Source",
            &nxyz_in,
            3,
            &mut ix_src_start,
            &mut ix_src_end,
            &mut if_src_limits,
        );
        get_boundary_limits(
            pipinput,
            ibinning,
            "MatchedTo",
            &nxyz_out,
            2,
            &mut ix_mat_start,
            &mut ix_mat_end,
            &mut if_mat_limits,
        );
        pip_done();
        //
        // sample each volume to find mean and SD
        for iunit in 1..=2 {
            let id_samp_f = ((nx.wrapping_mul(ny) as f32 * nz as f32) / (8.0 * MAX_SAMPLE as f32))
                .powf(0.3333_f32)
                + 1.0;
            let idel_samp = id_samp_f as i32;
            let num_samp_x = (nx / 2) / idel_samp + 1;
            let mut ix_start =
                ((nx / 2) as f32 - 0.5 * num_samp_x as f32 * idel_samp as f32) as i32;
            ix_start = ix_start.max(0);
            let num_samp_y = (ny / 2) / idel_samp + 1;
            let mut iy_start =
                ((ny / 2) as f32 - 0.5 * num_samp_y as f32 * idel_samp as f32) as i32;
            iy_start = iy_start.max(0);
            let num_samp_z = (nz / 2) / idel_samp + 1;
            let mut iz_start =
                ((nz / 2) as f32 - 0.5 * num_samp_z as f32 * idel_samp as f32) as i32;
            iz_start = iz_start.max(0);
            let mut ndat = 0_i32;
            for jz in 1..=num_samp_z {
                let iz = iz_start + (jz - 1) * idel_samp;
                for jy in 1..=num_samp_y {
                    let iy = iy_start + (jy - 1) * idel_samp;
                    iiu_set_position(iunit, iz, iy);
                    if irdlin(iunit, &mut array).is_err() {
                        exit_error("Reading file");
                    }
                    for jx in 1..=num_samp_x {
                        let ix = 1 + ix_start + (jx - 1) * idel_samp;
                        ndat += 1;
                        brray[(ndat - 1) as usize] = array[(ix - 1) as usize];
                    }
                }
            }
            let iu = (iunit - 1) as usize;
            avg_sd(&brray, ndat, &mut avg[iu], &mut sd[iu], &mut sem);
            // format(' Volume',i2,': mean =',f12.4,',  SD =',f12.4)
            println!(
                " Volume{:>2}: mean ={},  SD ={}",
                iunit,
                format_f(f64::from(avg[iu]), 12, 4),
                format_f(f64::from(sd[iu]), 12, 4)
            );
        }
        //
        // scale second volume to match first
        //
        let scale = sd[0] / sd[1];
        let fac_add = avg[0] - avg[1] * scale;
        //
        let do_mat_tests = if_mat_limits > 0
            || n_left_fill > 0
            || n_right_fill > 0
            || n_bot_fill > 0
            || n_top_fill > 0;
        let _num_threads = num_omp_threads(LIM_THREADS);
        let mut num_sec_read = 0_i32;
        let mut num_pt_filled: i64 = 0;
        for iz in 0..nz {
            let zcen = iz as f32 - cen_z_out;
            iiu_set_position(1, iz, 0);
            iiu_set_position(2, iz, 0);
            if irdsec(1, &mut array).is_err() {
                exit_error("Reading file");
            }
            if irdsec(2, &mut brray).is_err() {
                exit_error("Reading file");
            }
            let mut num_fill_on_sec = 0_i32;
            //
            // !$OMP PARALLEL ... one thread:
            let mut dmin_thread = 1.0e37_f32;
            let mut dmax_thread = -1.0e37_f32;
            let mut num_fill_thread = 0_i32;
            // !$OMP DO
            for iy in 0..ny {
                let ycen = iy as f32 - cen_y_out;
                let ind_base = 1 + iy * nx;
                for ix in 0..nx {
                    let xcen = ix as f32 - cen_x_out;

                    // Get the coordinate in the source volume and fill if it is out of bounds
                    let xp = minv[0][0] * xcen + minv[0][1] * ycen + minv[0][2] * zcen + cen_x_in;
                    let yp = minv[1][0] * xcen + minv[1][1] * ycen + minv[1][2] * zcen + cen_y_in;
                    let zp = minv[2][0] * xcen + minv[2][1] * ycen + minv[2][2] * zcen + cen_z_in;
                    let mut do_fill = xp < 0.0
                        || xp > (nx_in - 1) as f32
                        || yp < 0.0
                        || yp > (ny_in - 1) as f32
                        || zp < 0.0
                        || zp > (nz_in - 1) as f32;

                    // Then if NOT filling, check against the boundary limits in source volume
                    if if_src_limits > 0 && !do_fill {
                        let iz_src = (zp + 1.0) as i32;
                        do_fill = xp < ix_src_start[(iz_src - 1) as usize] as f32
                            || xp > ix_src_end[(iz_src - 1) as usize] as f32;
                    }

                    // Then if filling, only fill if inside the usable boundaries in matched
                    // volume.  But fill regardless outside the specified borders
                    if do_mat_tests {
                        do_fill = (do_fill
                            && ix >= ix_mat_start[iz as usize]
                            && ix <= ix_mat_end[iz as usize])
                            || ix < n_left_fill
                            || nx - ix <= n_right_fill
                            || iz < n_bot_fill
                            || nz - iz <= n_top_fill;
                    }
                    if do_fill {
                        num_fill_thread += 1;
                        let ind = (ind_base + ix - 1) as usize;
                        let val = scale * brray[ind] + fac_add;
                        array[ind] = val;
                        // `filltomo.f90:230-231`: `minss dminThread, val` and
                        // `maxss dmaxThread, val` in the reference object (both
                        // loop versions), so a NaN value replaces the running
                        // extreme.  (Per OpenMP thread natively; this is the
                        // one-thread result.)
                        dmin_thread = minss(dmin_thread, val);
                        dmax_thread = maxss(dmax_thread, val);
                    }
                }
            }
            // !$OMP CRITICAL
            num_fill_on_sec += num_fill_thread;
            // `filltomo.f90:238-239`: `minss dminThread, dminOut` /
            // `maxss dmaxThread, dmaxOut`.
            dmin_out = minss(dmin_thread, dmin_out);
            dmax_out = maxss(dmax_thread, dmax_out);
            // !$OMP END PARALLEL

            if num_fill_on_sec > 0 {
                iiu_set_position(1, iz, 0);
                iiu_write_section(1, array.as_mut_ptr().cast());
                num_sec_read += 1;
                num_pt_filled += i64::from(num_fill_on_sec);
            }
        }
        let mut dat = [b' '; 9];
        b3d_date(&mut dat);
        let mut tim = [b' '; 8];
        time(&mut tim);
        // format('FILLTOMO: Replacing parts of dual-axis tomogram',t57, a9,2x, a8)
        let mut title = [b' '; MRC_LABEL_SIZE];
        let head = b"FILLTOMO: Replacing parts of dual-axis tomogram";
        title[..head.len()].copy_from_slice(head);
        title[56..65].copy_from_slice(&dat);
        title[67..75].copy_from_slice(&tim);
        iiu_write_header(1, &title, 1, dmin_out, dmax_out, dmean_out);
        iiu_close(1);
        // format(i15,a,i7,a)
        println!(
            "{:>15} points replaced on{:>7} sections",
            num_pt_filled, num_sec_read
        );
        exit(0);
    }
}

/// Original subroutine `getBoundaryLimits` (`filltomo.f90:238`).
///
/// `ixStart`/`ixEnd` are indexed by Z (1-based in the source, 0-based here).
/// The `fortmodel` module the source's `get_region_contours` fills is a local
/// [`FortModel`]; the model is read afresh on each call.
pub fn get_boundary_limits(
    pipinput: bool,
    ibinning: i32,
    opt_prefix: &str,
    nxyz: &[i32; 3],
    im_unit: i32,
    ix_start: &mut [i32],
    ix_end: &mut [i32],
    icont_use: &mut i32,
) {
    // parameter (LIMVERT=100000, LIMCONT=1000)
    const LIMVERT: i32 = 100000;
    const LIMCONT: i32 = 1000;
    let mut xvert = vec![0f32; LIMVERT as usize];
    let mut yvert = vec![0f32; LIMVERT as usize];
    let mut zcont = vec![0f32; LIMCONT as usize];
    let mut ind_vert = vec![0_i32; LIMCONT as usize];
    let mut num_vert = vec![0_i32; LIMCONT as usize];
    // character*320 inFile
    let mut in_file = [b' '; 320];
    let (mut nx_raw, mut ny_raw, mut nz_raw) = (0_i32, 0_i32, 0_i32);
    let mut num_cont = 0;
    let mut if_flip = 0;

    *icont_use = 0;
    if pipinput {
        let prefix = opt_prefix.trim_end_matches(' ');
        let if_model = 1 - pipgetstring_(format!("{prefix}BoundaryModel").as_bytes(), &mut in_file);
        let if_raw = 1 - pip_get_three_integers(
            format!("{prefix}RawStackSize").as_bytes(),
            &mut nx_raw,
            &mut ny_raw,
            &mut nz_raw,
        );
        let if_xform =
            1 - pipgetstring_(format!("{prefix}StackTransforms").as_bytes(), &mut in_file);
        if if_xform > 0 && if_raw == 0 {
            exit_error("A raw stack size must be entered to use a stack transform file");
        }
        if if_model + if_xform > 1 {
            exit_error(
                "You cannot enter both a boundary model and raw stack transforms for the same tomogram",
            );
        }

        // Process model file if one given
        if if_model > 0 {
            let mut fm = FortModel::default();
            let (mut lim_cont, mut lim_vert) = (LIMCONT, LIMVERT);
            get_region_contours(
                &mut fm,
                &crate::imod::libcfshr::b3dutil::fortran_string(&in_file),
                "FILLTOMO",
                &mut xvert,
                &mut yvert,
                &mut num_vert,
                &mut ind_vert,
                &mut zcont,
                &mut num_cont,
                &mut if_flip,
                &mut lim_cont,
                &mut lim_vert,
                im_unit,
            );
            if num_cont == 0 {
                exit_error(&format!("The {prefix} boundary model has no contours"));
            }

            // Find contour closest to middle
            let mut mindiff = 100000000_i32;
            for icont in 1..=num_cont {
                // abs(real - integer) assigned to an integer: truncation.
                let idiff = (zcont[(icont - 1) as usize] - (nxyz[1] / 2) as f32).abs() as i32;
                if idiff < mindiff {
                    mindiff = idiff;
                    *icont_use = icont;
                }
            }
        }

        // Or get the transforms, and apply it to raw size to make a contour
        if if_xform > 0 {
            // allocate(flist(2, 3, nzRaw)); a negative or huge size is an
            // allocation failure.
            let mut flist: Vec<[f32; 6]> = Vec::new();
            let mut icont = 0;
            if flist.try_reserve_exact(nz_raw.max(0) as usize).is_err() {
                icont = 1;
            }
            memory_error(icont, "Array for stack transforms");
            let name = crate::imod::libcfshr::b3dutil::fortran_string(&in_file);
            let file = dopen(12, &name, "ro", "f");
            // The source never closes unit 12 (`filltomo.f90:280`), so when
            // `-sxform` and `-mxform` name the same file the second `dopen`
            // keeps gfortran's connection and position, and if the first read
            // reached the end of file native stops with "Reading MatchedTo
            // stack transform file".  Fixed in translation (BUGS.md): the unit
            // is closed after the read, so each call reads its file from the
            // start.
            let mut reader = BufReader::new(file);
            icont = xfrdall2(&mut reader, &mut flist, nz_raw);
            drop(reader);
            // xfrdall2 leaves nlist at limlist + 1 when there are too many
            // transforms (`xfrdall.f:25`), and native then clears
            // `flist(1:2, 3, 1:numFlist)` one element past its allocation and
            // averages over a window that can index past it.  Fixed in
            // translation (BUGS.md): numFlist is the number of transforms
            // actually stored.
            let num_flist = flist.len() as i32;
            if icont > 1 {
                exit_error(&format!("Reading {prefix} stack transform file"));
            }
            // An empty transform file leaves native averaging the
            // uninitialised `flist(:, :, 1)`; the translation stops instead.
            if num_flist == 0 {
                exit_error(&format!(
                    "The {prefix} stack transform file has no transforms"
                ));
            }

            // Adjust sizes for binning, zero out transform shifts
            nx_raw /= ibinning;
            ny_raw /= ibinning;
            nz_raw /= ibinning;
            for f in flist.iter_mut() {
                f[4] = 0.0;
                f[5] = 0.0;
            }

            // Average over the assumed middle of the tilt series
            let num_apply = 1.max(10.min(num_flist / 6));
            xvert[0..4].fill(0.0);
            yvert[0..4].fill(0.0);
            ind_vert[0] = 1;
            *icont_use = 1;
            num_vert[0] = 4;
            let xll = (nxyz[0] - nx_raw) as f32 / 2.0;
            let yll = (nxyz[2] - ny_raw) as f32 / 2.0;
            let xcen = nxyz[0] as f32 / 2.0;
            let ycen = nxyz[2] as f32 / 2.0;

            // Apply to each corner
            let istart = 1.max((num_flist - num_apply) / 2);
            for icont in istart..=istart + num_apply - 1 {
                let i = 1.max(num_flist.min(icont));
                let f = &flist[(i - 1) as usize];
                let napp = num_apply as f32;
                let (xtrans, ytrans) = xfapply(f, xcen, ycen, xll, yll);
                xvert[0] += xtrans / napp;
                yvert[0] += ytrans / napp;
                let (xtrans, ytrans) = xfapply(f, xcen, ycen, xll + nx_raw as f32, yll);
                xvert[1] += xtrans / napp;
                yvert[1] += ytrans / napp;
                let (xtrans, ytrans) =
                    xfapply(f, xcen, ycen, xll + nx_raw as f32, yll + ny_raw as f32);
                xvert[2] += xtrans / napp;
                yvert[2] += ytrans / napp;
                let (xtrans, ytrans) = xfapply(f, xcen, ycen, xll, yll + ny_raw as f32);
                xvert[3] += xtrans / napp;
                yvert[3] += ytrans / napp;
            }
            let _ = nz_raw;
        }
    }

    if *icont_use == 0 {
        for iz in 1..=nxyz[2] {
            ix_start[(iz - 1) as usize] = 0;
            ix_end[(iz - 1) as usize] = nxyz[0] - 1;
        }
    } else {
        let indv = ind_vert[(*icont_use - 1) as usize];
        let np = num_vert[(*icont_use - 1) as usize];
        let xv = &xvert[(indv - 1) as usize..];
        let yv = &yvert[(indv - 1) as usize..];
        for iz in 1..=nxyz[2] {
            let izu = (iz - 1) as usize;

            // Find first pixel inside the contour from the left
            for ix in 0..nxyz[0] {
                ix_start[izu] = nxyz[0] + 1;
                if inside(xv, yv, np, ix as f32 + 0.5, iz as f32 - 0.5) {
                    ix_start[izu] = ix;
                    break;
                }
            }

            // If none was found, set other limit, otherwise find first pixel inside from
            // right
            if ix_start[izu] > nxyz[0] {
                ix_end[izu] = 0;
            } else {
                let mut ix = nxyz[0] - 1;
                while ix >= ix_start[izu] {
                    if inside(xv, yv, np, ix as f32 + 0.5, iz as f32 - 0.5) {
                        ix_end[izu] = ix;
                        break;
                    }
                    ix -= 1;
                }
            }
        }
    }
}
