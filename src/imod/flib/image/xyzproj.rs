//! Translation of `IMOD/flib/image/xyzproj.f90`.
//!
//! This program will compute projections of a 3-dimensional block of an
//! image file at a series of tilts around either the X, the Y or the Z axis.
//! The block may be any arbitrary subset of the image file.
//!
//! The main program maps to [`xyzproj`], its internal subroutine
//! `useIntersectionAreas` to [`use_intersection_areas`] (the host variables
//! it reads and writes are passed in [`Host`]), and the external subroutines
//! `xtransp`, `ytransp`, `commonLineBox` (with its internal `rotatePoint`)
//! and `commonLineRays` to [`xtransp`], [`ytransp`], [`common_line_box`],
//! [`rotate_point`] and [`common_line_rays`].
//!
//! The source's arrays are 1-based; index variables here keep the source's
//! values and are lowered by one at each access.  The projection-box arrays
//! (`ixBoxLo(LIMPROJ)` ...) are indexed from 0 by the source (`iproj` runs
//! from 0), one element before their start, so natively `ixBoxLo(0)` is the
//! word in front of each array.  Fixed in translation (`BUGS.md`): they are
//! one element longer here and indexed from 0, so every view has storage of
//! its own.  The same holds for `cosAng`, `sinAng` and `nrayMax`.
//!
//! Fortran `MAX`/`MIN` of reals on data values follow the operand order of
//! the reference object's `maxss`/`minss` instructions
//! (`gfortran_rt::{maxss, minss}`); the geometric ones cannot see a NaN.

use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::{
    cvttss2si, gfortran_cosd_r4, gfortran_sind_r4, maxss, minss,
};
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, list_read};
use crate::imod::flib::subrs::hvem::get_tilt_angles::get_tilt_angles;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::set_projection_rays::set_projection_rays;
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdpas};
use crate::imod::libcfshr::b3dutil::{exit, set_float_output_for_entered_mode};
use crate::imod::libcfshr::parse_params::{
    pip_get_boolean, pip_get_float, pip_get_integer, pip_get_three_floats, pip_get_two_floats,
    pip_get_two_integers, pip_number_of_entries,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::projectpixel::make_ray_area_lookup_table;
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use crate::imod::libiimod::mrcfiles::MRC_LABEL_SIZE;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position, iiu_write_lines};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_mode, iiu_alt_sample, iiu_alt_size, iiu_ret_delta, iiu_trans_header,
    iiu_write_header,
};
use std::io::{BufRead, Write};

/// `parameter (LIMSTACK = 40000000, LIMPIX = 40000, LIMPROJ = 1440,
/// LIMRAY = 180 * 40000)` (`xyzproj.f90:16-17`).
const LIMSTACK: i32 = 40000000;
const LIMPIX: i32 = 40000;
const LIMPROJ: i32 = 1440;
const LIMRAY: i32 = 180 * 40000;
/// `parameter (numOptions = 22)` (`xyzproj.f90:61-62`).
const NUM_OPTIONS: i32 = 22;
/// Fallback PIP table, the `options(1)` string (`xyzproj.f90:64-72`).
const OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@axis:AxisToTiltAround:CH:@\
xminmax:XMinAndMax:IP:@yminmax:YMinAndMax:IP:@zminmax:ZMinAndMax:IP:@\
angles:StartEndIncAngle:FT:@tiltfile:TiltFile:FN:@adjust:AdjustFileAngles:FN:@\
tangles:TiltAngles:FAM:@mode:ModeToOutput:I:@width:WidthToOutput:I:@\
addmult:AddThenMultiply:FP:@fill:FillValue:F:@constant:ConstantScaling:B:@\
ray:UseRayIntersections:B:@series:InputIsTiltSeries:B:@\
first:FirstTiltAngle:F:@increment:TiltIncrement:F:@full:FullAreaAtTilt:B:@\
param:ParameterFile:PF:@help:usage:B:";

/// `read(*,*)` with no `END=`/`ERR=`: end of input is the gfortran runtime
/// error, status 2.
fn read_list(items: &mut [ListItem]) {
    let _ = std::io::stdout().flush();
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    if list_read(&mut lock, items).is_err() {
        eprintln!("Fortran runtime error: End of file");
        exit(2);
    }
}

/// Fortran `Iw` output of an `integer*4`: right-justified in `w` columns,
/// or `w` asterisks when it does not fit.
fn fortran_i(value: i32, w: usize) -> String {
    let text = value.to_string();
    if text.len() > w {
        "*".repeat(w)
    } else {
        format!("{text:>w$}")
    }
}

/// Fortran `NINT` of a `real*4` (`lroundf`, then the low 32 bits).
fn nint(x: f32) -> i32 {
    x.round() as i64 as i32
}

/// The host variables of `xyzproj` that its internal subroutine
/// `useIntersectionAreas` reads (`xyzproj.f90:573-645` uses them by host
/// association).
struct Host<'a> {
    invert_ang: i32,
    tilt_start: f32,
    tilt_inc: f32,
    iproj: i32,
    common_line: bool,
    have_angles: bool,
    tilt_angles: &'a [f32],
    sin_ang: &'a [f32],
    cos_ang: &'a [f32],
    nx_slice: i32,
    ny_slice: i32,
    nxout: i32,
    iout_base: i32,
    num_slices: i32,
    ist_del: i32,
    scale_add: f32,
    scale_fac: f32,
    ray_fac: f32,
    fill: f32,
    nray_max: &'a [i32],
    nray_inc: &'a [i32],
    iray_base: i32,
}

/// Original program `xyzproj` (`xyzproj.f90:11`).
pub fn xyzproj() {
    unsafe {
        // real*4 array(LIMSTACK) in common /bigarr/, with xrayStr, yrayStr and
        // nrayInc: static storage, zero before first use.
        let mut array: Vec<f32> = vec![0f32; LIMSTACK as usize];
        let mut xray_str: Vec<f32> = vec![0f32; LIMRAY as usize];
        let mut yray_str: Vec<f32> = vec![0f32; LIMRAY as usize];
        let mut nray_inc: Vec<i32> = vec![0_i32; LIMRAY as usize];
        let mut in_file = String::new();
        let mut out_file = String::new();
        // character*1 xyz: never set by the source on some paths (a tilt
        // series with no angle entries and no axis); blank here.
        let mut xyz: u8 = b' ';
        let mut nxyz_in = [0_i32; 3];
        let mut mxyz_in = [0_i32; 3];
        let nxyzst = [0_i32; 3];
        let map_scales: [[i32; 3]; 3] = [[2, 1, 2], [1, 2, 1], [1, 3, 1]];
        let mut cell: [f32; 6] = [0., 0., 0., 90., 90., 90.];
        // common /nxyz/ nxIn, nyin, nzIn, nxout, nyOut, nzOut
        let (mut nxout, mut ny_out, mut nz_out) = (0_i32, 0_i32, 0_i32);
        let mut nray_max = vec![0_i32; LIMPROJ as usize + 1];
        let mut pixtmp = vec![0f32; LIMPIX as usize];
        let mut cos_ang = vec![0f32; LIMPROJ as usize + 1];
        let mut sin_ang = vec![0f32; LIMPROJ as usize + 1];
        let mut tilt_angles = vec![0f32; LIMPROJ as usize + 1];
        let mut ix_box_lo = vec![0_i32; LIMPROJ as usize + 1];
        let mut iy_box_lo = vec![0_i32; LIMPROJ as usize + 1];
        let mut ix_box_hi = vec![0_i32; LIMPROJ as usize + 1];
        let mut iy_box_hi = vec![0_i32; LIMPROJ as usize + 1];
        let mut mode_in = 0_i32;
        let (mut nx_slice, mut ny_slice, mut len_load) = (0_i32, 0_i32, 0_i32);
        let (mut load0, mut load_dir, mut ind_axis) = (0_i32, 0_i32, 0_i32);
        let (mut load_xlo, mut load_xhi, mut load_ylo, mut load_yhi) = (0, 0, 0, 0);
        let (mut dmin, mut dmax, mut dmean) = (0f32, 0f32, 0f32);
        let (mut tilt_start, mut tilt_end, mut tilt_inc): (f32, f32, f32);
        let (mut scale_add, mut scale_fac): (f32, f32);
        let mut fill: f32;
        let mut if_ray_scale: i32;
        let mut common_line: bool;
        let mut have_angles: bool;
        let mut use_intersections: bool;
        let mut full_image = false;
        let mut adjust_angles: f32;
        let mut nview = 0_i32;
        let mut ixo_end = 0_i32;
        let mut tilt_max: f32;
        let mut proj_middle: f32 = 0.;
        //
        tilt_start = 0.;
        tilt_end = 0.;
        tilt_inc = 1.;
        scale_add = 0.;
        scale_fac = 1.;
        if_ray_scale = 1;
        common_line = false;
        have_angles = false;
        use_intersections = false;
        adjust_angles = 0.;
        //
        // Pip startup: set error, parse options, check help, set flag if used
        //
        let (mut num_opt_arg, mut num_non_opt_arg) = (0, 0);
        pip_read_or_parse_options(
            &[OPTIONS],
            NUM_OPTIONS,
            "xyzproj",
            "ERROR: XYZPROJ - ",
            true,
            3,
            1,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
        );
        let pip_input = num_opt_arg + num_non_opt_arg > 0;

        if pip_get_in_out_file("InputFile", 1, "Name of input file", &mut in_file, 320) != 0 {
            exit_error("No input file specified");
        }
        //
        imopen(1, &in_file, "ro");
        irdhdr(
            1,
            nxyz_in.as_mut_ptr(),
            mxyz_in.as_mut_ptr(),
            &mut mode_in,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
        let (nx_in, nyin, nz_in) = (nxyz_in[0], nxyz_in[1], nxyz_in[2]);
        //
        if pip_get_in_out_file("OutputFile", 2, "Name of output file", &mut out_file, 320) != 0 {
            exit_error("No output file specified");
        }
        //
        let mut ix0 = 0;
        let mut ix1 = nx_in - 1;
        let mut iy0 = 0;
        let mut iy1 = nyin - 1;
        let mut iz0 = 0;
        let mut iz1 = nz_in - 1;
        if pip_input {
            let _ = pip_get_two_integers(b"XMinAndMax", &mut ix0, &mut ix1);
            let _ = pip_get_two_integers(b"YMinAndMax", &mut iy0, &mut iy1);
            let _ = pip_get_two_integers(b"ZMinAndMax", &mut iz0, &mut iz1);
        } else {
            print!(
                " Enter index coordinates of block: ix0,ix1,iy0,iy1,iz0,iz1\n (or / for whole volume): "
            );
            read_list(&mut [
                ListItem::Integer(&mut ix0),
                ListItem::Integer(&mut ix1),
                ListItem::Integer(&mut iy0),
                ListItem::Integer(&mut iy1),
                ListItem::Integer(&mut iz0),
                ListItem::Integer(&mut iz1),
            ]);
        }
        ix0 = 0.max(ix0);
        ix1 = ix1.min(nx_in - 1);
        iy0 = 0.max(iy0);
        iy1 = iy1.min(nyin - 1);
        iz0 = 0.max(iz0.min(nz_in - 1));
        iz1 = 0.max(iz1.min(nz_in - 1));
        if ix0 > ix1 || iy0 > iy1 {
            exit_error("No volume specified");
        }
        //
        let mut ix_load0 = 0_i32;
        let mut ix_load1 = 0_i32;
        let mut iy_load0 = 0_i32;
        let mut iy_load1: i32;
        if pip_input {
            let _ = pip_get_logical("UseRayIntersections", &mut use_intersections);
            let _ = pip_get_logical("InputIsTiltSeries", &mut common_line);
            let _ = pip_number_of_entries(b"TiltAngles", &mut ix_load0);
            let _ = pip_number_of_entries(b"TiltFile", &mut ix_load1);
            let _ = pip_number_of_entries(b"FirstTiltAngle", &mut iy_load0);
            if common_line && use_intersections {
                exit_error("You cannot use intersection areas when input is a tilt series");
            }
            if iy_load0 > 0 && !common_line {
                exit_error(
                    "You cannot enter FirstTiltAngle unless the input file is a tilt series",
                );
            }
            let _ = pip_get_logical("FullAreaAtTilt", &mut full_image);
            if ix_load0 + ix_load1 + iy_load0 > 0 {
                nview = 0;
                if common_line {
                    nview = nz_in;
                }
                get_tilt_angles(
                    &mut nview,
                    3,
                    &mut tilt_angles[..LIMPROJ as usize],
                    LIMPROJ,
                    1,
                );
                if common_line && nview != nz_in {
                    exit_error("There must be a tilt angle for each view");
                }
                if common_line {
                    xyz = b'Z';
                }
                if !common_line {
                    nz_out = nview;
                }
                have_angles = true;
            }
            if ix_load1 > 0 && pip_get_float(b"AdjustFileAngles", &mut adjust_angles) == 0 {
                for i in 1..=nview {
                    tilt_angles[(i - 1) as usize] += adjust_angles;
                }
            }

            // PipGetString into `character*1 xyz` is the `pip_fwrap` wrapper:
            // a longer entry leaves its first character and returns -1.
            let mut record = [xyz];
            let ierr = pipgetstring_(b"AxisToTiltAround", &mut record);
            xyz = record[0];
            if ierr != 0 && !common_line {
                exit_error("You must enter an axis to tilt around");
            }
            if common_line && xyz != b'Z' && xyz != b'z' {
                exit_error(
                    "You can enter a tilt series as input only for projections around the Z axis",
                );
            }
            let ierr = pip_get_three_floats(
                b"StartEndIncAngle",
                &mut tilt_start,
                &mut tilt_end,
                &mut tilt_inc,
            );
            if have_angles && ierr == 0 && !common_line {
                exit_error("You can enter only one of StartEndIncAngle, TiltAngles, and TiltFile");
            }
        } else {
            print!(" Axis to tilt around (enter X, Y or Z): ");
            let _ = std::io::stdout().flush();
            // read(*,'(a)') xyz
            let mut line: Vec<u8> = Vec::new();
            match std::io::stdin().lock().read_until(b'\n', &mut line) {
                Ok(0) | Err(_) => {
                    eprintln!("Fortran runtime error: End of file");
                    exit(2);
                }
                Ok(_) => {}
            }
            xyz = match line.first() {
                Some(&b'\n') | None => b' ',
                Some(&c) => c,
            };
            //
            print!(" Starting, ending, increment tilt angles: ");
            read_list(&mut [
                ListItem::Real(&mut tilt_start),
                ListItem::Real(&mut tilt_end),
                ListItem::Real(&mut tilt_inc),
            ]);
        }
        if common_line || !have_angles {
            while tilt_start > 180. {
                tilt_start -= 360.;
            }
            while tilt_start <= -180. {
                tilt_start += 360.;
            }
            while tilt_end > 180. {
                tilt_end -= 360.;
            }
            while tilt_end <= -180. {
                tilt_end += 360.;
            }
            if tilt_inc >= 0. && tilt_end < tilt_start {
                tilt_end += 360.;
            }
            if tilt_inc < 0. && tilt_end > tilt_start {
                tilt_end -= 360.;
            }
            nz_out = 1;
            if tilt_inc != 0. {
                nz_out = cvttss2si((tilt_end - tilt_start) / tilt_inc + 1.);
            }
        }
        if nz_out > LIMPROJ {
            exit_error("Too many projections for arrays");
        }
        //
        let nx_block = ix1 + 1 - ix0;
        let ny_block = iy1 + 1 - iy0;
        let idirz;
        let nz_block;
        if iz1 >= iz0 {
            idirz = 1;
            nz_block = iz1 + 1 - iz0;
        } else {
            idirz = -1;
            nz_block = iz0 + 1 - iz1;
        }
        //
        // set up size of slices within which to project sets of lines,
        // total # of slices, and other parameters for the 3 cases
        //
        let mut invert_ang = 1;
        if xyz == b'x' || xyz == b'X' {
            //tilt around X
            nx_slice = nz_block;
            ny_slice = ny_block;
            nxout = ny_block;
            ny_out = nx_block;
            len_load = ny_block;
            load0 = ix0;
            load_dir = 1;
            ind_axis = 1;
        } else if xyz == b'y' || xyz == b'Y' {
            //tilt around Y
            nx_slice = nx_block;
            ny_slice = nz_block;
            nxout = nx_block;
            ny_out = ny_block;
            len_load = nx_block;
            load0 = iy0;
            load_dir = 1;
            invert_ang = -1;
            ind_axis = 2;
        } else if xyz == b'z' || xyz == b'Z' {
            //tilt around Z
            nx_slice = nx_block;
            ny_slice = ny_block;
            nxout = nx_block;
            ny_out = nz_block;
            len_load = 0;
            load0 = iz0;
            load_dir = idirz;
            ind_axis = 3;
        } else {
            exit_error("You must enter one of X, Y, Z, x, y, or z for axis");
        }
        //
        let mut mode_out = 1;
        if mode_in == 2 {
            mode_out = 2;
        }
        fill = dmean;
        ix_load0 = ix0;
        ix_load1 = ix1;
        iy_load0 = iy0;
        iy_load1 = iy1;
        if common_line {
            nxout = 0;
            ix_load0 = nx_in;
            ix_load1 = 0;
            iy_load0 = nyin;
            iy_load1 = 0;
            tilt_max = 0.;
            let mut iz_sec = iz0;
            while iz_sec <= iz1 {
                tilt_max = tilt_max.max(tilt_angles[iz_sec as usize].abs());
                iz_sec += 1;
            }
            proj_middle = invert_ang as f32 * (tilt_start + nz_out as f32 * tilt_inc / 2.);
            for iproj in 0..nz_out {
                let ip = iproj as usize;
                let angle = invert_ang as f32 * (tilt_start + iproj as f32 * tilt_inc);
                common_line_box(
                    ix0,
                    ix1,
                    iy0,
                    iy1,
                    nx_in,
                    nyin,
                    angle,
                    tilt_max,
                    proj_middle,
                    &mut ix_box_lo[ip],
                    &mut iy_box_lo[ip],
                    &mut ix_box_hi[ip],
                    &mut iy_box_hi[ip],
                    &mut load_xlo,
                    &mut load_xhi,
                    &mut load_ylo,
                    &mut load_yhi,
                    full_image,
                );
                nxout = nxout.max(ix_box_hi[ip] + 1 - ix_box_lo[ip]);
                ix_load0 = ix_load0.min(load_xlo);
                ix_load1 = ix_load1.max(load_xhi);
                iy_load0 = iy_load0.min(load_ylo);
                iy_load1 = iy_load1.max(load_yhi);
            }
            nx_slice = ix_load1 + 1 - ix_load0;
            ny_slice = iy_load1 + 1 - iy_load0;
        }
        let _ = proj_middle;
        //
        if pip_input {
            let _ = pip_get_integer(b"WidthToOutput", &mut nxout);
            let ierr = pip_get_integer(b"ModeToOutput", &mut mode_out);
            if ierr == 0 {
                set_float_output_for_entered_mode(mode_out);
            }
            let _ = pip_get_two_floats(b"AddThenMultiply", &mut scale_add, &mut scale_fac);
            let _ = pip_get_float(b"FillValue", &mut fill);
            let mut ixr = 0;
            let _ = pip_get_boolean(b"ConstantScaling", &mut ixr);
            if_ray_scale = 1 - ixr;
        } else {
            print!(" Width of output image [/ for{}]: ", fortran_i(nxout, 5));
            read_list(&mut [ListItem::Integer(&mut nxout)]);
            //
            print!(" Output data mode [/ {}]: ", fortran_i(mode_out, 2));
            read_list(&mut [ListItem::Integer(&mut mode_out)]);
            set_float_output_for_entered_mode(mode_out);
            //
            // write(*,'(1x,a,$)') '0 to scale by 1/(vertical thickness),'// &
            // ' or 1 to scale by 1/(ray length): '
            // read(*,*) ifRayScale
            //
            print!(" Additional scaling factors to add then multiply by [/ for 0,1]: ");
            read_list(&mut [
                ListItem::Real(&mut scale_add),
                ListItem::Real(&mut scale_fac),
            ]);
            //
            print!(
                " Value to fill parts of output not projected to\n   (before scaling, if any) [/ for mean={}]: ",
                crate::imod::flib::subrs::compat::gfortran_rt::format_f(dmean as f64, 10, 2)
            );
            read_list(&mut [ListItem::Real(&mut fill)]);
        }

        fill = (fill + scale_add) * scale_fac;
        //
        // set up output file and header
        //
        //
        let nxyz_out = [nxout, ny_out, nz_out];
        imopen(2, &out_file, "new");
        iiu_trans_header(2, 1);
        iiu_alt_mode(2, mode_out);
        iiu_alt_size(2, &nxyz_out, &nxyzst);
        iiu_alt_sample(2, &nxyz_out);
        let delta = iiu_ret_delta(1);
        for i in 1..=3_usize {
            cell[i - 1] = nxyz_out[i - 1] as f32
                * delta[(map_scales[(ind_axis - 1) as usize][i - 1] - 1) as usize];
        }
        iiu_alt_cell(2, &cell);
        let mut tim = [b' '; 8];
        time(&mut tim);
        let mut dat = [b' '; 9];
        b3d_date(&mut dat);
        //
        // write(titlech, 301) ix0, ix1, iy0, iy1, iz0, iz1, xyz, dat, tim
        // 301 format('XYZPROJ: x',2i5,', y',2i5,', z ',2i5,' about ',a1,t57,a9,2X,a8)
        // The T57 tab goes back over the last digit of iz1, ' about ' and the
        // axis letter, which the date then replaces.
        let mut titlech = [b' '; MRC_LABEL_SIZE];
        {
            let text = format!(
                "XYZPROJ: x{}{}, y{}{}, z {}{} about ",
                fortran_i(ix0, 5),
                fortran_i(ix1, 5),
                fortran_i(iy0, 5),
                fortran_i(iy1, 5),
                fortran_i(iz0, 5),
                fortran_i(iz1, 5)
            );
            let bytes = text.as_bytes();
            titlech[..bytes.len()].copy_from_slice(bytes);
            titlech[bytes.len()] = xyz;
            titlech[56..65].copy_from_slice(&dat);
            titlech[67..75].copy_from_slice(&tim);
        }
        let mut dmax2: f32 = -1.0e20;
        let mut dmin2: f32 = 1.0e20;
        let mut dsum: f32 = 0.;
        //
        // set up stack loading with slices
        //
        let ist_del = nx_slice * ny_slice;
        let mut lim_slices = LIMPIX.min(LIMSTACK / (ist_del + nxout.max(len_load)));
        if common_line {
            lim_slices = 1;
        }
        if lim_slices < 1 {
            exit_error("Images too large for stack");
        }
        if nxout * nz_out > LIMRAY {
            exit_error("Too many projections for output this wide");
        }
        let num_loads = (ny_out + lim_slices - 1) / lim_slices;
        let iout_base = 1 + lim_slices * ist_del;
        // A tilt series (one slice per load) can need more than LIMSTACK; the
        // source would run past its array there.
        let need = (iout_base as i64 - 1 + lim_slices as i64 * nxout.max(len_load) as i64) as usize;
        if need > array.len() {
            array.resize(need, 0.);
        }
        //
        // analyse the number of points on each ray in each projection
        //
        for iproj in 0..nz_out {
            let ip = iproj as usize;
            let mut angle = invert_ang as f32 * (tilt_start + iproj as f32 * tilt_inc);
            if !common_line && have_angles {
                angle = invert_ang as f32 * tilt_angles[ip];
            }
            sin_ang[ip] = gfortran_sind_r4(angle);
            cos_ang[ip] = gfortran_cosd_r4(angle);
            let iray_base = (iproj * nxout) as usize;
            if !common_line {
                let (mut s, mut c) = (sin_ang[ip], cos_ang[ip]);
                set_projection_rays(
                    &mut s,
                    &mut c,
                    nx_slice,
                    ny_slice,
                    nxout,
                    &mut xray_str[iray_base..],
                    &mut yray_str[iray_base..],
                    &mut nray_inc[iray_base..],
                    &mut nray_max[ip],
                );
                sin_ang[ip] = s;
                cos_ang[ip] = c;
            }
        }
        //
        // loop on loads of several to many slices at once
        //
        let mut iy_out_str = 0; //starting Y output line
        for _iload in 1..=num_loads {
            let num_slices = lim_slices.min(ny_out - iy_out_str);
            let load1 = load0 + load_dir * (num_slices - 1);
            //
            // load the slices one of 3 different ways: for X, get vertical
            // rectangles of sections in input file, transpose into slice array
            //
            if xyz == b'x' || xyz == b'X' {
                let mut iz_sec = iz0;
                while (idirz > 0 && iz_sec <= iz1) || (idirz < 0 && iz_sec >= iz1) {
                    iiu_set_position(1, iz_sec, 0);
                    let (stack, out) = array.split_at_mut((iout_base - 1) as usize);
                    if irdpas(1, out, num_slices, ny_block, load0, load1, iy0, iy1).is_err() {
                        exit_error("Reading file");
                    }
                    xtransp(
                        out,
                        idirz * (iz_sec - iz0) + 1,
                        stack,
                        nx_slice,
                        ny_slice,
                        num_slices,
                    );
                    iz_sec += idirz;
                }
                //
                // for Y, get horizontal rectangles of sections in input file,
                // transpose differently into slice array
                //
            } else if xyz == b'y' || xyz == b'Y' {
                let mut iz_sec = iz0;
                while (idirz > 0 && iz_sec <= iz1) || (idirz < 0 && iz_sec >= iz1) {
                    // readstart = walltime()
                    iiu_set_position(1, iz_sec, 0);
                    let (stack, out) = array.split_at_mut((iout_base - 1) as usize);
                    if irdpas(1, out, nx_block, num_slices, ix0, ix1, load0, load1).is_err() {
                        exit_error("Reading file");
                    }
                    // readdone = walltime()
                    // readtime = readtime + readdone - readstart
                    ytransp(
                        out,
                        idirz * (iz_sec - iz0) + 1,
                        stack,
                        nx_slice,
                        ny_slice,
                        num_slices,
                    );
                    // transtime =  transtime +  walltime() - readdone
                    iz_sec += idirz;
                }
                //
                // for Z, just get slices directly from sections in file
                //
            } else {
                let mut ind_stack = 1;
                let mut iz_sec = load0;
                while (load_dir > 0 && iz_sec <= load1) || (load_dir < 0 && iz_sec >= load1) {
                    iiu_set_position(1, iz_sec, 0);
                    if irdpas(
                        1,
                        &mut array[(ind_stack - 1) as usize..],
                        nx_slice,
                        ny_slice,
                        ix_load0,
                        ix_load1,
                        iy_load0,
                        iy_load1,
                    )
                    .is_err()
                    {
                        exit_error("Reading file");
                    }
                    ind_stack += ist_del;
                    iz_sec += load_dir;
                }
            }
            load0 += load_dir * num_slices;
            //
            // loop on different projection views
            //
            // readdone = walltime()
            for iproj in 0..nz_out {
                let ip = iproj as usize;
                let iray_base = iproj * nxout;
                let rb = iray_base as usize;
                if common_line {
                    let angle = invert_ang as f32 * (tilt_start + iproj as f32 * tilt_inc);
                    // Fixed in translation (`BUGS.md`): the source passes
                    // `tiltAngles(load0 + 1)` (`xyzproj.f90:403`) after `load0`
                    // has already been advanced past the section just loaded
                    // (`:393`), so native foreshortens each view with the next
                    // view's angle, reads `tiltAngles(nz + 1)` (0) for the last
                    // view, and `tiltAngles(0)` (before the array) for the
                    // last load of a reversed `-zminmax N,0`.  A tilt series
                    // loads one section per load (`LIMSLICES = 1`, `:328`), so
                    // the section just loaded is `load1`, and its own angle is
                    // used here.
                    common_line_rays(
                        ix0,
                        ix1,
                        iy0,
                        iy1,
                        nx_in,
                        nyin,
                        ix_load0,
                        iy_load0,
                        nxout,
                        angle,
                        tilt_angles[load1 as usize],
                        ix_box_lo[ip],
                        iy_box_lo[ip],
                        ix_box_hi[ip],
                        iy_box_hi[ip],
                        &mut xray_str[rb..],
                        &mut yray_str[rb..],
                        &mut nray_inc[rb..],
                        nx_slice,
                        ny_slice,
                        full_image,
                    );
                    nray_max[ip] = nray_inc[rb];
                }
                //
                // Set X-independent part of scaling
                let ray_fac: f32 = if if_ray_scale == 0 {
                    scale_fac / ny_slice as f32
                } else {
                    scale_fac / nray_max[ip] as f32
                };
                //
                // set the output lines for this view to the fill value
                let mut ray_add: f32 = fill;
                if if_ray_scale == 0 {
                    ray_add = (fill * nray_max[ip] as f32) / ny_slice as f32;
                }
                for ind in iout_base..=iout_base + num_slices * nxout - 1 {
                    array[(ind - 1) as usize] = ray_add;
                }

                if use_intersections {
                    use_intersection_areas(
                        &Host {
                            invert_ang,
                            tilt_start,
                            tilt_inc,
                            iproj,
                            common_line,
                            have_angles,
                            tilt_angles: &tilt_angles,
                            sin_ang: &sin_ang,
                            cos_ang: &cos_ang,
                            nx_slice,
                            ny_slice,
                            nxout,
                            iout_base,
                            num_slices,
                            ist_del,
                            scale_add,
                            scale_fac,
                            ray_fac,
                            fill,
                            nray_max: &nray_max,
                            nray_inc: &nray_inc,
                            iray_base,
                        },
                        &mut array,
                    );

                //
                // for projections at an angle, process multiple slices at once
                } else if sin_ang[ip] != 0. {
                    let sin_a = sin_ang[ip];
                    let cos_a = cos_ang[ip];
                    //
                    // loop on pixels along line
                    for ix_out in 1..=nxout {
                        let nraypts = nray_inc[(ix_out + iray_base - 1) as usize];
                        // print *,nraypts
                        if nraypts > 0 {
                            //
                            // if block along ray, clear temporary array for output pixels
                            for ipix in 1..=num_slices {
                                pixtmp[(ipix - 1) as usize] = 0.;
                            }
                            //
                            // move along ray, computing at each point indexes and factors
                            // for quadratic interpolation
                            let xs = xray_str[(ix_out + iray_base - 1) as usize];
                            let ys = yray_str[(ix_out + iray_base - 1) as usize];
                            for iray in 0..nraypts {
                                let xray = xs + iray as f32 * sin_a;
                                let yray = ys + iray as f32 * cos_a;
                                let ixr = nint(xray);
                                let iyr = nint(yray);
                                let dx = xray - ixr as f32;
                                let dy = yray - iyr as f32;
                                let mut ixy = ixr + (iyr - 1) * nx_slice;
                                let mut ixy4 = ixy - 1;
                                let mut ixy6 = ixy + 1;
                                let mut ixy8 = ixr + iyr.min(ny_slice - 1) * nx_slice;
                                let mut ixy2 = ixr + (iyr - 2).max(0) * nx_slice;
                                //
                                // loop through pixels in different slices, do quadratic
                                // interpolation limited by values of surrounding pixels
                                //
                                // The source indexes without checks.  One range test per
                                // ray point covers every slice of the column (the five
                                // subscripts only grow by `istackDel`), and then the loop
                                // reads through raw offsets; anything outside the array
                                // takes the checked loop, which panics where the source
                                // would read out of bounds.
                                let sd = ist_del as i64;
                                let lo = (ixy2.min(ixy4).min(ixy).min(ixy6).min(ixy8) - 1) as i64;
                                let hi = (ixy2.max(ixy4).max(ixy).max(ixy6).max(ixy8) - 1) as i64
                                    + (num_slices as i64 - 1) * sd;
                                if lo >= 0
                                    && num_slices >= 1
                                    && hi < array.len() as i64
                                    && num_slices as usize <= pixtmp.len()
                                {
                                    let ap = array.as_ptr();
                                    let pp = pixtmp.as_mut_ptr();
                                    let sdu = ist_del as usize;
                                    let (mut o2, mut o4, mut o5, mut o6, mut o8) = (
                                        (ixy2 - 1) as usize,
                                        (ixy4 - 1) as usize,
                                        (ixy - 1) as usize,
                                        (ixy6 - 1) as usize,
                                        (ixy8 - 1) as usize,
                                    );
                                    for ipix in 0..num_slices as usize {
                                        // SAFETY: every offset lies in [lo, hi], tested
                                        // above to be inside `array`; `ipix` is below
                                        // `num_slices <= pixtmp.len()`.
                                        unsafe {
                                            let v2 = *ap.add(o2);
                                            let v4 = *ap.add(o4);
                                            let v5 = *ap.add(o5);
                                            let v6 = *ap.add(o6);
                                            let v8 = *ap.add(o8);
                                            // Operand order from the reference object.
                                            let vmax =
                                                maxss(maxss(maxss(v4, v2), maxss(v6, v5)), v8);
                                            let vmin =
                                                minss(minss(minss(v4, v2), minss(v6, v5)), v8);
                                            //
                                            let a = (v6 + v4) * 0.5 - v5;
                                            let b = (v8 + v2) * 0.5 - v5;
                                            let c = (v6 - v4) * 0.5;
                                            let d = (v8 - v2) * 0.5;
                                            //
                                            *pp.add(ipix) += maxss(
                                                minss(
                                                    a * dx * dx
                                                        + b * dy * dy
                                                        + c * dx
                                                        + d * dy
                                                        + v5,
                                                    vmax,
                                                ),
                                                vmin,
                                            );
                                        }
                                        //
                                        o5 += sdu; //increment subscripts for
                                        o2 += sdu; //next slice
                                        o4 += sdu;
                                        o6 += sdu;
                                        o8 += sdu;
                                    }
                                    continue;
                                }
                                for ipix in 1..=num_slices {
                                    let v2 = array[(ixy2 - 1) as usize];
                                    let v4 = array[(ixy4 - 1) as usize];
                                    let v5 = array[(ixy - 1) as usize];
                                    let v6 = array[(ixy6 - 1) as usize];
                                    let v8 = array[(ixy8 - 1) as usize];
                                    // Operand order from the reference object.
                                    let vmax = maxss(maxss(maxss(v4, v2), maxss(v6, v5)), v8);
                                    let vmin = minss(minss(minss(v4, v2), minss(v6, v5)), v8);
                                    //
                                    let a = (v6 + v4) * 0.5 - v5;
                                    let b = (v8 + v2) * 0.5 - v5;
                                    let c = (v6 - v4) * 0.5;
                                    let d = (v8 - v2) * 0.5;
                                    //
                                    let p = &mut pixtmp[(ipix - 1) as usize];
                                    *p += maxss(
                                        minss(
                                            a * dx * dx + b * dy * dy + c * dx + d * dy + v5,
                                            vmax,
                                        ),
                                        vmin,
                                    );
                                    //
                                    ixy += ist_del; //increment subscripts for
                                    ixy2 += ist_del; //next slice
                                    ixy4 += ist_del;
                                    ixy6 += ist_del;
                                    ixy8 += ist_del;
                                }
                            }
                            //
                            // set up scaling and put pixels out in different lines
                            let ray_add = scale_add * scale_fac
                                + ray_fac
                                    * (nray_max[ip] - nraypts) as f32
                                    * (fill / scale_fac - scale_add);
                            //
                            let mut ind_out = ix_out + iout_base - 1;
                            for ipix in 1..=num_slices {
                                array[(ind_out - 1) as usize] =
                                    ray_fac * pixtmp[(ipix - 1) as usize] + ray_add;
                                ind_out += nxout;
                            }
                        }
                    }
                } else {
                    //
                    // simple case of straight projection
                    // find limits of X to loop on
                    let mut ixo_start = 0;
                    for ix_out in 1..=nxout {
                        if ixo_start == 0 && nray_inc[(ix_out + iray_base - 1) as usize] > 0 {
                            ixo_start = ix_out;
                        }
                        if nray_inc[(ix_out + iray_base - 1) as usize] > 0 {
                            ixo_end = ix_out;
                        }
                    }
                    //
                    // Get starting positions in X, Y, and directions
                    let mut ixdir = 1;
                    let ixr = nint(xray_str[(ixo_start + iray_base - 1) as usize]);
                    if xray_str[(ixo_end + iray_base - 1) as usize] < ixr as f32 {
                        ixdir = -1;
                    }
                    let iyr = nint(yray_str[(ixo_start + iray_base - 1) as usize]);
                    let ixy8 = 1f32.copysign(cos_ang[ip]) as i32;
                    let nraypts = nray_inc[(ixo_start + iray_base - 1) as usize];
                    let ray_add = scale_add * scale_fac
                        + ray_fac
                            * (nray_max[ip] - nraypts) as f32
                            * (fill / scale_fac - scale_add);
                    //
                    // loop on slices; clear the array segment for the slice
                    for ipix in 1..=num_slices {
                        let ind_out = nxout * (ipix - 1) + iout_base - 1;
                        for ix_out in ixo_start..=ixo_end {
                            array[(ind_out + ix_out - 1) as usize] = 0.;
                        }

                        // Loop on levels in Y (the ray points) and set starting index
                        for iray in 0..nraypts {
                            let mut ixy = ixr
                                + (iyr - 1) * nx_slice
                                + iray * nx_slice * ixy8
                                + (ipix - 1) * ist_del;
                            //
                            // Add line into array
                            //
                            // One range test for the whole line (both subscripts
                            // move monotonically), then raw offsets; outside the
                            // array the checked loop below panics where the source
                            // would run past it.
                            let count = ixo_end as i64 - ixo_start as i64;
                            let ixy_last = ixy as i64 + count * ixdir as i64;
                            let len = array.len() as i64;
                            if count >= 0
                                && ixy.min(ixy_last as i32) as i64 >= 1
                                && (ixy as i64).max(ixy_last) <= len
                                && ind_out as i64 + ixo_start as i64 >= 1
                                && ind_out as i64 + ixo_end as i64 <= len
                            {
                                let ap = array.as_mut_ptr();
                                let mut oi = ixy as i64 - 1;
                                let mut oo = (ind_out + ixo_start - 1) as usize;
                                for _ix_out in ixo_start..=ixo_end {
                                    // SAFETY: both offsets stay within the ranges
                                    // tested above.
                                    unsafe {
                                        *ap.add(oo) += *ap.add(oi as usize);
                                    }
                                    oi += ixdir as i64;
                                    oo += 1;
                                }
                                continue;
                            }
                            for ix_out in ixo_start..=ixo_end {
                                array[(ind_out + ix_out - 1) as usize] += array[(ixy - 1) as usize];
                                ixy += ixdir;
                            }
                        }
                        //
                        // Scale the data
                        for ix_out in ixo_start..=ixo_end {
                            let v = &mut array[(ind_out + ix_out - 1) as usize];
                            *v = *v * ray_fac + ray_add;
                        }
                    }
                }
                //
                // get min, max, mean; output the lines to proper section
                //
                let (mut tmin, mut tmax, mut tmean) = (0f32, 0f32, 0f32);
                array_min_max_mean_fortran(
                    &array[(iout_base - 1) as usize..],
                    &nxout,
                    &num_slices,
                    &1,
                    &nxout,
                    &1,
                    &num_slices,
                    &mut tmin,
                    &mut tmax,
                    &mut tmean,
                );
                dmin2 = minss(dmin2, tmin);
                dmax2 = maxss(dmax2, tmax);
                dsum += tmean * num_slices as f32 * nxout as f32;
                iiu_set_position(2, iproj, iy_out_str);
                iiu_write_lines(
                    2,
                    array[(iout_base - 1) as usize..].as_mut_ptr().cast(),
                    num_slices,
                );
            }
            iy_out_str += num_slices;
            // projtime = projtime + walltime() - readdone
        }
        //
        // finish up file header
        //
        let dmean2 = dsum / nxout.wrapping_mul(ny_out).wrapping_mul(nz_out) as f32;
        iiu_write_header(2, &titlech, 1, dmin2, dmax2, dmean2);
        iiu_close(2);
        iiu_close(1);
        // print *,'read time', readtime, '  transpose time', transtime, '
        // project time', projtime
        exit(0);
    }
}

/// Original internal subroutine `useIntersectionAreas` (`xyzproj.f90:573`).
///
/// useIntersectionAreas will project using areas of intersection between
/// rays and pixel
fn use_intersection_areas(h: &Host, array: &mut [f32]) {
    const MAX_DIST: i32 = 1000;
    let mut ind_del_ray = [0_i32; MAX_DIST as usize + 1];
    let mut num_rays_hit = [0_i32; MAX_DIST as usize + 1];
    let mut ray_areas = [0f32; 3 * (MAX_DIST as usize + 1)];
    let ip = h.iproj as usize;
    let nxout = h.nxout;
    let nx_slice = h.nx_slice;
    let ny_slice = h.ny_slice;

    // Get the angle with the usual inversion, then make table of intersection areas
    let mut angle = h.invert_ang as f32 * (h.tilt_start + h.iproj as f32 * h.tilt_inc);
    if !h.common_line && h.have_angles {
        angle = h.invert_ang as f32 * h.tilt_angles[ip];
    }
    make_ray_area_lookup_table(
        angle,
        1,
        MAX_DIST,
        0.01,
        &mut ind_del_ray,
        &mut num_rays_hit,
        &mut ray_areas,
    );
    //
    // Loop on slices in Y and set starting and ending points
    for iy_slice in 1..=ny_slice {
        let mut ix_start = 1;
        let mut ix_end = nx_slice;
        let ypart = h.sin_ang[ip] * (iy_slice as f32 - 0.5 - ny_slice as f32 / 2.);
        if h.cos_ang[ip] != 0. {
            let ix_left =
                cvttss2si((ypart - nxout as f32 / 2.) / h.cos_ang[ip] + nx_slice as f32 / 2. + 0.5);
            let ix_right =
                cvttss2si((ypart + nxout as f32 / 2.) / h.cos_ang[ip] + nx_slice as f32 / 2. + 0.5);
            ix_start = ix_left.min(ix_right);
            ix_end = ix_left.max(ix_right);
        }
        ix_start = 1.max(ix_start);
        // Fixed in translation (`BUGS.md`): the source clamps with
        // `min(nxOut, ixEnd)` (`xyzproj.f90:600`), the output width, where
        // the slice width is meant (`ixStart` is clamped to the slice's first
        // pixel on the line before): natively a slice narrower than the output
        // (`-ray -width` wider than the block) reads past the end of each
        // slice line into the next one, and one wider than the output skips
        // its pixels beyond `nxOut`.  Clamped to `nxSlice` here.
        ix_end = nx_slice.min(ix_end);
        //
        // Loop across each line, get the output coordinate (0 to nxout) of each point
        for ix_slice in ix_start..=ix_end {
            let mut rx_out = h.cos_ang[ip] * (ix_slice as f32 - 0.5 - nx_slice as f32 / 2.) - ypart
                + nxout as f32 / 2.;
            if h.cos_ang[ip] == 0. {
                rx_out = (iy_slice as f32 - (ny_slice as f32 + 1.) / 2.)
                    .abs()
                    .copysign(h.sin_ang[ip])
                    + nxout as f32 / 2.;
            }
            if rx_out >= 0. && rx_out <= nxout as f32 {
                //
                // If with limits, get fraction and index in table
                let ix_out = cvttss2si(rx_out + 1.);
                let ray_dist = ix_out as f32 - rx_out;
                let mut lut_ind = cvttss2si(ray_dist * MAX_DIST as f32 + 1.);
                lut_ind = MAX_DIST.min(lut_ind);
                //
                // Loop on rays in the table, test each for validity (yes this could be faster)
                // and then process corresponding pixels into all the slices
                for ray_ind in 1..=num_rays_hit[(lut_ind - 1) as usize] {
                    let iray = ix_out + ind_del_ray[(lut_ind - 1) as usize] + ray_ind - 1;
                    if iray > 0 && iray <= nxout {
                        let area = ray_areas[(3 * (lut_ind - 1) + ray_ind - 1) as usize];
                        let mut ixy = ix_slice + (iy_slice - 1) * nx_slice;
                        let mut ind_out = h.iout_base + iray - 1;
                        // One range test for the whole column of slices (both
                        // subscripts only grow), then raw offsets; outside the
                        // array the checked loop below panics where the source
                        // would run past it.
                        let last = h.num_slices as i64 - 1;
                        if h.num_slices >= 1
                            && ixy >= 1
                            && ind_out >= 1
                            && (ixy as i64 - 1 + last * h.ist_del as i64) < array.len() as i64
                            && (ind_out as i64 - 1 + last * nxout as i64) < array.len() as i64
                        {
                            let ap = array.as_mut_ptr();
                            let (mut oi, mut oo) = ((ixy - 1) as usize, (ind_out - 1) as usize);
                            let (sd, so) = (h.ist_del as usize, nxout as usize);
                            for _ipix in 1..=h.num_slices {
                                // SAFETY: both offsets stay within the ranges
                                // tested above.
                                unsafe {
                                    *ap.add(oo) += area * *ap.add(oi);
                                }
                                oi += sd;
                                oo += so;
                            }
                            continue;
                        }
                        for _ipix in 1..=h.num_slices {
                            array[(ind_out - 1) as usize] += area * array[(ixy - 1) as usize];
                            ixy += h.ist_del;
                            ind_out += nxout;
                        }
                    }
                }
            }
        }
    }
    //
    // Apply scale and offset to the sum when all lines are done
    for ix_out in 1..=nxout {
        let ray_add = h.scale_add * h.scale_fac
            + h.ray_fac
                * (h.nray_max[ip] - h.nray_inc[(ix_out + h.iray_base - 1) as usize]) as f32
                * (h.fill / h.scale_fac - h.scale_add);
        let mut ind_out = h.iout_base + ix_out - 1;
        for _ipix in 1..=h.num_slices {
            let v = &mut array[(ind_out - 1) as usize];
            *v = h.ray_fac * *v + ray_add;
            ind_out += nxout;
        }
    }
}

/// Original subroutine `xtransp` (`xyzproj.f90:653`).
///
/// Subroutines to transpose the pieces of sections read from input file
/// into the array of slices: `array(nsl,nysl)`, `brray(nxsl,nysl,nsl)`.
pub fn xtransp(array: &[f32], icolm: i32, brray: &mut [f32], nxsl: i32, nysl: i32, nsl: i32) {
    for iy in 1..=nysl {
        for ix in 1..=nsl {
            brray[((icolm - 1) + (iy - 1) * nxsl + (ix - 1) * nxsl * nysl) as usize] =
                array[((ix - 1) + (iy - 1) * nsl) as usize];
        }
    }
}

/// Original subroutine `ytransp` (`xyzproj.f90:663`): `array(nxsl,nsl)`,
/// `brray(nxsl,nysl,nsl)`.
pub fn ytransp(array: &[f32], irow: i32, brray: &mut [f32], nxsl: i32, nysl: i32, nsl: i32) {
    for iy in 1..=nsl {
        for ix in 1..=nxsl {
            brray[((ix - 1) + (irow - 1) * nxsl + (iy - 1) * nxsl * nysl) as usize] =
                array[((ix - 1) + (iy - 1) * nxsl) as usize];
        }
    }
}

/// Original internal subroutine `rotatePoint` of `commonLineBox`
/// (`xyzproj.f90:781`); `xcen`, `ycen` are the host's.
fn rotate_point(
    xcen: f32,
    ycen: f32,
    cosa: f32,
    sina: f32,
    xin: f32,
    yin: f32,
    xout: &mut f32,
    yout: &mut f32,
) {
    *xout = (xin - xcen) * cosa - (yin - ycen) * sina + xcen;
    *yout = (xin - xcen) * sina + (yin - ycen) * cosa + ycen;
}

/// Fortran `MIN` of several reals, left to right.
fn fmin(values: &[f32]) -> f32 {
    let mut m = values[0];
    for &v in &values[1..] {
        if v < m {
            m = v;
        }
    }
    m
}

/// Fortran `MAX` of several reals, left to right.
fn fmax(values: &[f32]) -> f32 {
    let mut m = values[0];
    for &v in &values[1..] {
        if v > m {
            m = v;
        }
    }
    m
}

/// Original subroutine `commonLineBox` (`xyzproj.f90:679`).
///
/// Compute the needed box for common line projections that fits within
/// the given area and within the image at the given projection angle
/// and when tilt-foreshortened at the maximum tilt angle.  return the
/// coordinates defining the limits of the projection box in rotated
/// space, and the coordinates that need loading
#[allow(clippy::too_many_arguments)]
pub fn common_line_box(
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    nx_in: i32,
    nyin: i32,
    proj_ang: f32,
    tilt_max: f32,
    proj_middle: f32,
    ixlo: &mut i32,
    iylo: &mut i32,
    ixhi: &mut i32,
    iyhi: &mut i32,
    load_xlo: &mut i32,
    load_xhi: &mut i32,
    load_ylo: &mut i32,
    load_yhi: &mut i32,
    full_image: bool,
) {
    let (mut xll, mut xul, mut xlr, mut xur) = (0f32, 0f32, 0f32, 0f32);
    let (mut yll, mut yul, mut ylr, mut yur) = (0f32, 0f32, 0f32, 0f32);
    let (mut xllr, mut xul_rot, mut xlr_rot, mut xur_rot) = (0f32, 0f32, 0f32, 0f32);
    let (mut yll_rot, mut yul_rot, mut ylr_rot, mut yur_rot) = (0f32, 0f32, 0f32, 0f32);
    let (mut xll_tfs, mut xul_tfs, mut xlr_tfs, mut xur_tfs) = (0f32, 0f32, 0f32, 0f32);
    let (mut yll_tfs, mut yul_tfs, mut ylr_tfs, mut yurt) = (0f32, 0f32, 0f32, 0f32);
    let (mut yl_tfs, mut yh_tfs): (f32, f32);
    //
    // Determine rotation of area so that it is within 45 deg, so that the
    // box that is projected is oriented as closely as possible to the
    // specified box at the middle angle
    let mut rot_ang = proj_ang;
    let mut cos_ang = proj_middle;
    while cos_ang.abs() > 45.01 {
        rot_ang -= 90f32.copysign(cos_ang);
        cos_ang -= 90f32.copysign(cos_ang);
    }
    // print *,'Proj angle', projAng, '  Reduced angle', rotAng
    //
    // Incoming and outgoing coordinates are all numbered from 0
    let cos_rot = gfortran_cosd_r4(rot_ang);
    let sin_rot = -gfortran_sind_r4(rot_ang);
    let cos_ang = gfortran_cosd_r4(proj_ang);
    let sin_ang = gfortran_sind_r4(proj_ang);
    let xcen = (ix1 + ix0) as f32 / 2.;
    let ycen = (iy1 + iy0) as f32 / 2.;
    //
    // Set up to loop on area, stepping size down symmetrically until rotated
    // box fits within the image
    let mut x0 = ix0 as f32;
    let mut y0 = iy0 as f32;
    let mut x1 = ix1 as f32;
    let mut y1 = iy1 as f32;
    let max_steps = (ix1 + 1 - ix0).max(iy1 + 1 - iy0) / 2;
    let xstep = ((ix1 + 1 - ix0) / max_steps) as f32;
    let ystep = ((iy1 + 1 - iy0) / max_steps) as f32;
    for istep in 1..=max_steps {
        //
        // Rotate the box by negative of angle and see if it fits
        rotate_point(xcen, ycen, cos_rot, sin_rot, x0, y0, &mut xll, &mut yll);
        rotate_point(xcen, ycen, cos_rot, sin_rot, x0, y1, &mut xul, &mut yul);
        rotate_point(xcen, ycen, cos_rot, sin_rot, x1, y0, &mut xlr, &mut ylr);
        rotate_point(xcen, ycen, cos_rot, sin_rot, x1, y1, &mut xur, &mut yur);
        //
        // Back-rotate these boundaries by the original, unconstrained angle
        // and get the coordinates defining limits of projection box
        rotate_point(
            xcen,
            ycen,
            cos_ang,
            sin_ang,
            xll,
            yll,
            &mut xllr,
            &mut yll_rot,
        );
        rotate_point(
            xcen,
            ycen,
            cos_ang,
            sin_ang,
            xul,
            yul,
            &mut xul_rot,
            &mut yul_rot,
        );
        rotate_point(
            xcen,
            ycen,
            cos_ang,
            sin_ang,
            xlr,
            ylr,
            &mut xlr_rot,
            &mut ylr_rot,
        );
        rotate_point(
            xcen,
            ycen,
            cos_ang,
            sin_ang,
            xur,
            yur,
            &mut xur_rot,
            &mut yur_rot,
        );
        *ixlo = fmin(&[xllr, xul_rot, xlr_rot, xur_rot]).ceil() as i32;
        *ixhi = fmax(&[xllr, xul_rot, xlr_rot, xur_rot]).floor() as i32;
        *iylo = fmin(&[yll_rot, yul_rot, ylr_rot, yur_rot]).ceil() as i32;
        *iyhi = fmax(&[yll_rot, yul_rot, ylr_rot, yur_rot]).floor() as i32;
        //
        // tilt-foreshorten the Y limits of projection box and recompute corners
        let axisy =
            -(nx_in as f32 / 2. - xcen) * sin_ang + (nyin as f32 / 2. - ycen) * cos_ang + ycen;
        if full_image {
            yl_tfs = *iylo as f32;
            yh_tfs = *iyhi as f32;
        } else {
            yl_tfs = nint((*iylo as f32 - axisy) * gfortran_cosd_r4(tilt_max) + axisy) as f32;
            yh_tfs = nint((*iyhi as f32 - axisy) * gfortran_cosd_r4(tilt_max) + axisy) as f32;
        }
        rotate_point(
            xcen,
            ycen,
            cos_ang,
            -sin_ang,
            *ixlo as f32,
            yl_tfs,
            &mut xll_tfs,
            &mut yll_tfs,
        );
        rotate_point(
            xcen,
            ycen,
            cos_ang,
            -sin_ang,
            *ixlo as f32,
            yh_tfs,
            &mut xul_tfs,
            &mut yul_tfs,
        );
        rotate_point(
            xcen,
            ycen,
            cos_ang,
            -sin_ang,
            *ixhi as f32,
            yl_tfs,
            &mut xlr_tfs,
            &mut ylr_tfs,
        );
        rotate_point(
            xcen,
            ycen,
            cos_ang,
            -sin_ang,
            *ixhi as f32,
            yh_tfs,
            &mut xur_tfs,
            &mut yurt,
        );
        //
        // Get limits of the box needing loading - allow for quadratic
        // interpolation with the margins here, no extra margin is needed
        *load_xlo =
            fmin(&[xll, xul, xlr, xur, xll_tfs, xul_tfs, xlr_tfs, xur_tfs]).floor() as i32 - 2;
        *load_xhi =
            fmax(&[xll, xul, xlr, xur, xll_tfs, xul_tfs, xlr_tfs, xur_tfs]).ceil() as i32 + 2;
        *load_ylo = fmin(&[yll, yul, ylr, yur, yll_tfs, yul_tfs, ylr_tfs, yurt]).floor() as i32 - 2;
        *load_yhi = fmax(&[yll, yul, ylr, yur, yll_tfs, yul_tfs, ylr_tfs, yurt]).ceil() as i32 + 2;
        //
        // Test and see if this fits; if not reduce box
        if *load_xlo >= 0 && *load_xhi < nx_in && *load_ylo >= 0 && *load_yhi < nyin {
            break;
        }
        x0 += xstep;
        x1 -= xstep;
        y0 += ystep;
        y1 -= ystep;
        if istep > max_steps - 10 {
            exit_error("Cannot find rotated box that fits within image");
        }
    }
    // print *,istep, ' steps, new box:', x0, x1, y0, y1
    // print *,'Projection box limits:', ixlo, ixhi, iylo, iyhi
    // print *,'load limits:', loadXlo, loadXhi, loadYlo, loadYhi
}

/// Original subroutine `commonLineRays` (`xyzproj.f90:795`).
///
/// Compute the ray parameters for a common line projection given the
/// original coordinate limits, the start of loaded data, and coordinates
/// defining the projection box computed before
#[allow(clippy::too_many_arguments)]
pub fn common_line_rays(
    ix0: i32,
    ix1: i32,
    iy0: i32,
    iy1: i32,
    nx_in: i32,
    nyin: i32,
    load_xst: i32,
    load_yst: i32,
    nxout: i32,
    proj_ang: f32,
    tilt: f32,
    in_ixlo: i32,
    iylo: i32,
    in_ixhi: i32,
    iyhi: i32,
    xray_str: &mut [f32],
    yray_str: &mut [f32],
    nray_inc: &mut [i32],
    _nx_slice: i32,
    _ny_slice: i32,
    full_image: bool,
) {
    let iylo_tfs: i32;
    let iyhi_tfs: i32;
    //
    let cos_rot = gfortran_cosd_r4(proj_ang);
    let sin_rot = -gfortran_sind_r4(proj_ang);
    let xcen = (ix1 + ix0) as f32 / 2.;
    let ycen = (iy1 + iy0) as f32 / 2.;
    //
    // Trim the X limits if they are bigger than nxout
    let mut ixlo = in_ixlo;
    let mut ixhi = in_ixhi;
    if ixhi + 1 - ixlo > nxout {
        let ix = ixhi + 1 - ixlo - nxout;
        ixlo += ix / 2;
        ixhi -= ix - ix / 2;
    }
    //
    // Rotate the center of the image to determine the Y value of tilt axis
    // then tilt-foreshorten the lower and upper Y limits
    let axisy = (nx_in as f32 / 2. - xcen) * sin_rot + (nyin as f32 / 2. - ycen) * cos_rot + ycen;
    if full_image {
        iylo_tfs = iylo;
        iyhi_tfs = iyhi;
    } else {
        iylo_tfs = nint((iylo as f32 - axisy) * gfortran_cosd_r4(tilt) + axisy);
        iyhi_tfs = nint((iyhi as f32 - axisy) * gfortran_cosd_r4(tilt) + axisy);
    }
    //
    // Rotate the bottom line of this box to get ray starts, adjust by load
    // Also, incoming coordinates are numbered from 0 but we need array index
    // coordinates numbered from 1, so add 1 at this stage
    for ix in ixlo..=ixhi {
        let i = (ix + 1 - ixlo) as usize;
        xray_str[i - 1] =
            (ix as f32 - xcen) * cos_rot - (iylo_tfs as f32 - ycen) * sin_rot + xcen + 1.
                - load_xst as f32;
        yray_str[i - 1] =
            (ix as f32 - xcen) * sin_rot + (iylo_tfs as f32 - ycen) * cos_rot + ycen + 1.
                - load_yst as f32;
        nray_inc[i - 1] = iyhi_tfs + 1 - iylo_tfs;
    }
    // Fixed in translation (`BUGS.md`): the source starts this clearing loop
    // at `ixhi + 1 - ixlo` (`xyzproj.f90:854`), the last ray the loop above
    // just set, so native drops the last ray of every projection row.  The
    // clearing starts after it here.
    for i in (ixhi + 2 - ixlo)..=nxout {
        nray_inc[(i - 1) as usize] = 0;
    }
}
