//! Translation of `IMOD/imodutil/joinwarp2model.c` -- convert warping control
//! points to the nucleus of a refining model.
//!
//! The program maps to [`joinwarp2model`].  Its `system("xfmodel ...")` runs
//! our own `xfmodel` in this process (`CLAUDE.md`, "Our own commands are
//! called in process"): the command line needs only word splitting, and the
//! value `system` returns is the wait status, so a non-zero exit is reported
//! as `status << 8`, as the source prints it.

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, exit, imod_prog_name, imod_usage_header, program_args,
};
use crate::imod::libcfshr::linearxforms::{xf_apply, xf_invert};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_float, pip_get_in_out_file, pip_get_integer, pip_get_integer_array,
    pip_get_string, pip_get_two_integers, pip_read_or_parse_options,
};
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_get_scale, mrc_head_read};
use crate::imod::libimod::icont::imod_contour_new;
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OPEN, Ipoint, Iref_image, imod_new, imod_new_object, imod_set_ref_image,
};
use crate::imod::libimod::imodel_files::{imod_read, imod_write};
use crate::imod::libimod::iobj::imod_object_add_contour;
use crate::imod::libimod::ipoint::imod_point_append_xyz;
use crate::imod::libwarp::warpfiles::{
    WARP_CONTROL_PTS, get_linear_transform, get_num_warp_points, get_warp_point_arrays,
    read_warp_file, warp_files_done,
};
use std::io::Write;

/// `#define MAX_CHUNKS 1000` (`joinwarp2model.c:20`).
const MAX_CHUNKS: usize = 1000;

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// The `PipReadOrParseOptions` header callback, `imodUsageHeader`.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// Original program `main` (`joinwarp2model.c:22`).
pub fn joinwarp2model() {
    let argv = program_args();
    let argv_bytes: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
    let mut warp_name: Vec<u8> = Vec::new();
    let mut out_name: Vec<u8> = Vec::new();
    let mut join_name: Option<Vec<u8>> = None;
    let mut xform_name: Vec<u8> = Vec::new();
    let progname = imod_prog_name(argv.first().map(String::as_str).unwrap_or(""));
    let mut hdata = MrcHeader::default();
    let mut binning = 1_i32;
    let (mut x_offset, mut y_offset) = (0_i32, 0_i32);
    let mut linear_xf = [0.0_f32; 6];
    let mut inverse_xf = [0.0_f32; 6];
    let mut join_pixel = 0.0_f32;
    let mut xform_pixel = 0.0_f32;
    let mut warp_pixel = 0.0_f32;
    let (mut x_join_size, mut y_join_size) = (0_i32, 0_i32);
    let mut chunk_sizes = [0_i32; MAX_CHUNKS];
    let (mut nx_warp, mut ny_warp, mut nz_warp) = (0_i32, 0_i32, 0_i32);
    let mut version = 0_i32;
    let mut warp_flags = 0_i32;
    let (mut nx_xform, mut ny_xform, mut nz_xform) = (0_i32, 0_i32, 0_i32);
    let (mut num_opt_args, mut num_non_opt_args) = (0_i32, 0_i32);
    let mut ierr: i32;
    let mut num_chunks: i32;
    let mut iz: i32 = 0;
    let mut max_control = 0_i32;
    let mut num_control = 0_i32;
    let mut x_control: Vec<f32> = Vec::new();
    let mut y_control: Vec<f32> = Vec::new();
    let mut x_vector: Vec<f32> = Vec::new();
    let mut y_vector: Vec<f32> = Vec::new();
    let no_warp_txt = "There are no warping transforms; no warp point model produced";

    /* Fallbacks from    ../manpages/autodoc2man 2 1 joinwarp2model */
    let num_options = 9;
    let options: [&[u8]; 9] = [
        b"input:InputWarpFile:FN:",
        b"output:OutputModelFile:FN:",
        b"joined:JoinedFile:FN:",
        b"xform:AppliedTransformFile:FN:",
        b"size:SizeOfJoinInXandY:IP:",
        b"pixel:PixelSpacing:F:",
        b"offset:OffsetInXandY:IP:",
        b"binning:BinningOfJoin:I:",
        b"chunks:ChunkSizes:IA:",
    ];

    pip_read_or_parse_options(
        argv_bytes.len() as i32,
        &argv_bytes,
        &options,
        num_options,
        progname.as_bytes(),
        5,
        1,
        1,
        &mut num_opt_args,
        &mut num_non_opt_args,
        Some(imod_usage_header_for_pip),
    );

    if pip_get_in_out_file(b"InputWarpFile", 0, &mut warp_name) != 0 {
        exit_error(b"No input warping file specified");
    }
    if pip_get_in_out_file(b"OutputModelFile", 1, &mut out_name) != 0 {
        exit_error(b"No output model file name specified");
    }
    if pip_get_string(b"AppliedTransformFile", &mut xform_name) != 0 {
        exit_error(b"No file of applied warping G transforms specified");
    }
    {
        let mut name: Vec<u8> = Vec::new();
        if pip_get_string(b"JoinedFile", &mut name) == 0 {
            join_name = Some(name);
        }
    }
    let if_pixel = 1 - pip_get_float(b"PixelSpacing", &mut join_pixel);
    let if_size = 1 - pip_get_two_integers(b"SizeOfJoinInXandY", &mut x_join_size, &mut y_join_size);
    if (join_name.is_some() && (if_pixel != 0 || if_size != 0))
        || (join_name.is_none() && (if_pixel == 0 || if_size == 0))
    {
        exit_error(
            b"You must enter either the joined image file name or the pixel size and X/Y size of that file",
        );
    }

    num_chunks = 0;
    if pip_get_integer_array(b"ChunkSizes", &mut chunk_sizes, &mut num_chunks, MAX_CHUNKS as i32) != 0
    {
        exit_error(b"You must enter the list of chunk sizes in the joined file");
    }
    let _ = pip_get_two_integers(b"OffsetInXandY", &mut x_offset, &mut y_offset);
    let _ = pip_get_integer(b"BinningOfJoin", &mut binning);

    /* Turn chunks sizes into cumulative Z's */
    for iz in 1..num_chunks.max(0) as usize {
        chunk_sizes[iz] += chunk_sizes[iz - 1];
    }

    /* Get the warp point file and check its properties */
    let warp_name = String::from_utf8_lossy(&warp_name).into_owned();
    ierr = read_warp_file(
        &warp_name,
        &mut nx_warp,
        &mut ny_warp,
        &mut nz_warp,
        &mut iz,
        &mut warp_pixel,
        &mut version,
        &mut warp_flags,
    );
    if ierr == -3 && version == 0 {
        let _ = ImodFile::Stdout.write_all(format!("{no_warp_txt}\n").as_bytes());
        let _ = ImodFile::Stdout.flush();
        exit(0); /* What to exit as ?? */
    }
    if ierr != 0 {
        exit_error_fmt!(
            "Error %d opening or reading the warp file %s",
            CArg::Int(ierr as i64),
            CArg::Str(&warp_name)
        );
    }
    if warp_flags & WARP_CONTROL_PTS == 0 {
        exit_error(b"The warp file does not contain control points to turn into a model");
    }
    if nz_warp != num_chunks {
        exit_error_fmt!(
            "The number of chunks entered (%d) does not match the number of transforms in the warp file (%d)",
            CArg::Int(num_chunks as i64),
            CArg::Int(nz_warp as i64)
        );
    }

    if get_num_warp_points(-1, &mut max_control) != 0 {
        exit_error(b"Getting maximum number of control points");
    }

    /* Open the joined file and get the header and sizes and pixel size */
    if let Some(name) = &join_name {
        let name = String::from_utf8_lossy(name).into_owned();
        let Some(mut fin) = ImodFile::open(&name, "rb") else {
            exit_error_fmt!("Opening joined image file %s", CArg::Str(&name))
        };
        if mrc_head_read(&mut fin, &mut hdata) != 0 {
            exit_error_fmt!("Reading MRC header from joined image file %s", CArg::Str(&name));
        }
        drop(fin);
        x_join_size = hdata.nx;
        y_join_size = hdata.ny;
        if hdata.nz != chunk_sizes[(num_chunks - 1).max(0) as usize] {
            // The source's format has "(d)" for its second value.
            exit_error_fmt!(
                "The sum of the chunk sizes (%d) does not match the Z size of the joined file (d)",
                CArg::Int(chunk_sizes[(num_chunks - 1).max(0) as usize] as i64),
                CArg::Int(hdata.nz as i64)
            );
        }
        join_pixel = mrc_get_scale(&hdata).0;
    }

    /* Get filenames for temp files */
    let out_name = String::from_utf8_lossy(&out_name).into_owned();
    let orig_name = format!("{out_name}.origpts");
    let xfpts_name = format!("{out_name}.xfpts");

    /* Set up the model */
    let Some(mut orig_mod) = imod_new() else {
        exit_error(b"Allocating new model or array for warp points")
    };
    if imod_new_object(&mut orig_mod) != 0 {
        exit_error(b"Adding an object to model");
    }
    {
        let obj = &mut orig_mod.obj[0];
        obj.flags |= IMOD_OBJFLAG_OPEN;
        obj.pdrawsize = 50;
        obj.red = 1.;
        obj.green = 0.;
        obj.blue = 0.;
    }

    /* Get the warp points for each section into new contours */
    for iz in 1..nz_warp {
        if get_num_warp_points(iz, &mut num_control) != 0
            || get_warp_point_arrays(
                iz,
                &mut x_control,
                &mut y_control,
                &mut x_vector,
                &mut y_vector,
            ) != 0
            || get_linear_transform(iz, &mut linear_xf, 2) != 0
        {
            exit_error_fmt!(
                "Getting warp control points or linear transform for iz = %d",
                CArg::Int(iz as i64)
            );
        }
        xf_invert(&linear_xf, &mut inverse_xf, 2);
        for pt in 0..num_control.max(0) as usize {
            /* The position on the unaligned image is the control point plus the vector
            times the inverse linear xform */
            let (xback, yback) = xf_apply(
                &inverse_xf,
                (nx_warp as f64 / 2.) as f32,
                (ny_warp as f64 / 2.) as f32,
                x_control[pt] + x_vector[pt],
                y_control[pt] + y_vector[pt],
                2,
            );
            let Some(mut cont) = imod_contour_new() else {
                exit_error(b"Adding point to model object")
            };
            if imod_point_append_xyz(&mut cont, xback, yback, iz as f32) == 0
                || imod_object_add_contour(&mut orig_mod.obj[0], cont) < 0
            {
                exit_error(b"Adding point to model object");
            }
        }
    }

    /* Make sure that the applied warp transform file matches the warp point file */
    let xform_name = String::from_utf8_lossy(&xform_name).into_owned();
    ierr = read_warp_file(
        &xform_name,
        &mut nx_xform,
        &mut ny_xform,
        &mut nz_xform,
        &mut iz,
        &mut xform_pixel,
        &mut version,
        &mut warp_flags,
    );
    if ierr < 0 {
        exit_error_fmt!(
            "Error # %d opening applied transform file %s",
            CArg::Int(ierr as i64),
            CArg::Str(&xform_name)
        );
    }
    // Kept native (BUGS.md, `joinwarp2model`): the third test repeats the
    // second; `nzXform` is never compared.
    if nx_xform != nx_warp
        || ny_xform != ny_warp
        || ny_xform != ny_warp
        || (xform_pixel - warp_pixel).abs() as f64 > 1.0e-4 * warp_pixel as f64
        || (warp_flags & WARP_CONTROL_PTS) != 0
    {
        exit_error(b"Applied transform file does appear to be derived from warp point file");
    }
    warp_files_done();

    /* Set maximum dimensions and pixel size data and write model */
    orig_mod.xmax = nx_warp;
    orig_mod.ymax = ny_warp;
    orig_mod.zmax = nz_warp;
    // `B3DMALLOC(IrefImage, 1)` with only `cscale` and `ctrans` set: the
    // source writes `oscale`, `otrans`, `orot` and `crot` from uninitialised
    // heap (BUGS.md §2).  Defined: `Iref_image::default()`, the identity.
    let mut ref_image = Iref_image::default();
    ref_image.cscale = Ipoint {
        x: warp_pixel,
        y: warp_pixel,
        z: warp_pixel,
    };
    ref_image.ctrans = Ipoint {
        x: 0.,
        y: 0.,
        z: 0.,
    };
    orig_mod.ref_image = Some(ref_image);

    let Some(mut fout) = ImodFile::open(&orig_name, "wb") else {
        exit_error_fmt!(
            "Opening new file for temporary model of original points %s",
            CArg::Str(&orig_name)
        )
    };
    if imod_write(&orig_mod, &mut fout).is_err() {
        exit_error(b"Writing temporary model of original points");
    }
    drop(fout);

    /* Run the xfmodel and get the transformed model */
    let command = format!("xfmodel -xform {xform_name} {orig_name} {xfpts_name}");
    let _ = ImodFile::Stdout.flush();
    ierr = match crate::imod::commands::find("xfmodel") {
        Some(entry) => {
            let mut words: Vec<std::ffi::OsString> = command
                .split_whitespace()
                .map(std::ffi::OsString::from)
                .collect();
            words[0] = match std::env::current_exe() {
                Ok(path) => path.with_file_name("xfmodel").into_os_string(),
                Err(_) => std::ffi::OsString::from("xfmodel"),
            };
            match crate::imod::commands::run_in_process(entry, words, None, false) {
                Ok((status, _)) => status << 8,
                Err(_) => 127 << 8,
            }
        }
        None => 127 << 8,
    };
    if ierr != 0 {
        let _ = ImodFile::Stdout.write_all(
            format!("This command was run and gave an error:\n{command}\n").as_bytes(),
        );
        exit_error_fmt!("Xfmodel failed with return value %d", CArg::Int(ierr as i64));
    }

    let mut xfpts_mod = match imod_read(&xfpts_name) {
        Ok(model) => model,
        Err(_) => exit_error_fmt!("Reading in transformed model %s", CArg::Str(&xfpts_name)),
    };

    /* Loop on points and scale them up and shift them and duplicate onto correct Z's */
    let obj = &mut xfpts_mod.obj[0];
    for co in 0..obj.cont.len() {
        let cont = &mut obj.cont[co];
        if cont.pts.len() != 1
            || (cont.pts[0].z as f64) < 0.9
            || cont.pts[0].z as f64 > num_chunks as f64 - 0.9
        {
            exit_error_fmt!(
                "Contour %d in transformed model %s has wrong number of points or Z out of range",
                CArg::Int(co as i64 + 1),
                CArg::Str(&xfpts_name)
            );
        }
        let xfpt = &mut cont.pts[0];
        // B3DNINT: `(int)floor(a + 0.5)`.
        let iz = (xfpt.z as f64 + 0.5).floor() as i32;
        xfpt.x = ((xfpt.x as f64 - nx_warp as f64 / 2.) * warp_pixel as f64 / join_pixel as f64
            + x_join_size as f64 / 2.
            - (x_offset / binning) as f64) as f32;
        xfpt.y = ((xfpt.y as f64 - ny_warp as f64 / 2.) * warp_pixel as f64 / join_pixel as f64
            + y_join_size as f64 / 2.
            - (y_offset / binning) as f64) as f32;
        xfpt.z = (chunk_sizes[(iz - 1) as usize] as f64 - 1.) as f32;
        let (x, y, z) = (xfpt.x, xfpt.y, xfpt.z);
        if imod_point_append_xyz(cont, x, y, z + 1.) == 0 {
            exit_error(b"Adding point to final model");
        }
    }

    /* Set the output size and the image reference information in the model */
    xfpts_mod.xmax = x_join_size;
    xfpts_mod.ymax = y_join_size;
    xfpts_mod.zmax = chunk_sizes[(num_chunks - 1).max(0) as usize];
    if join_name.is_some() {
        if imod_set_ref_image(&mut xfpts_mod, &hdata) != 0 {
            exit_error(b"Adding IrefImage structure to final model");
        }
    } else {
        xfpts_mod.ref_image = None;
    }

    let Some(mut fout) = ImodFile::open(&out_name, "wb") else {
        exit_error_fmt!("Opening new file for final model %s", CArg::Str(&out_name))
    };
    if imod_write(&xfpts_mod, &mut fout).is_err() {
        exit_error(b"Writing final model");
    }
    drop(fout);
    exit(0);
}
