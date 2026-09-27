//! `IMOD/imodutil/imodmop.c`: create an MRC image file from a model and an
//! MRC image file ("painting" the image inside model contours, tubes and
//! scattered points).
//!
//! The source's file-scope statics (`sScat2D`, `sBkgVal`, `sLIp`, ...) are
//! the fields of [`Mop`], which `main` owns and hands by reference to the
//! static functions that read them; `sLIp` pointed at `main`'s `li`, which is
//! [`Mop::li`] here.  Everything else is the source's own structure.

use std::io::Write as _;

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, exit, imod_backup_file, imod_copyright, imod_prog_name,
    imod_usage_header, imod_version, override_write_bytes, program_args,
    set_float_output_for_entered_mode, write_16_bit_mode_for_floats,
};
use crate::imod::libcfshr::islice::{
    Islice, slice_clear, slice_create, slice_get_val, slice_get_val_magnitude, slice_put_val,
};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_done, pip_get_boolean, pip_get_float, pip_get_float_array, pip_get_integer,
    pip_get_non_option_arg, pip_get_string, pip_get_three_floats, pip_get_two_floats,
    pip_get_two_integers, pip_print_help, pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::{ParseListError, parselist};
use crate::imod::libcfshr::simplestat::sums_to_avg_sd_dbl;
use crate::imod::libcfshr::statfuncs::gaussian_deviate;
use crate::imod::libiimod::iimage::{ii_fclose, ii_fopen};
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader, mrc_coord_cp, mrc_getdcsize,
    mrc_head_label, mrc_head_label_cp, mrc_head_new, mrc_head_read, mrc_head_write, mrc_init_li,
    mrc_read_slice, mrc_write_slice,
};
use crate::imod::libiimod::mrcsec::mrc_read_z;
use crate::imod::libiimod::mrcslice::slice_mmm;
use crate::imod::libimod::icont::{
    Nesting, imod_contour_check_nesting, imod_contour_delete, imod_contour_free_nests,
    imod_contour_free_z_tables, imod_contour_get_bbox, imod_contour_make_z_tables,
    imod_contour_nest_levels, imodel_contour_scan,
};
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OUT, Icont, Imod, Iobj, Ipoint, imod_trans_for_subset_load,
    imod_trans_to_match_image,
};
use crate::imod::libimod::imodel_files::imod_read;
use crate::imod::libimod::iobj::{imod_object_get_bbox, iobj_open, iobj_scat};
use crate::imod::libimod::ipoint::{
    imod_point_cont_distance, imod_point_get_size, imod_point_inside_cont,
    imod_point_line_seg_distance,
};

/// `#define FILE_STR_SIZE 4096` (`imodmop.c:44`); only a buffer size in C.
#[allow(dead_code)]
const FILE_STR_SIZE: usize = 4096;
/// `#define MAX_TUBE_DIAMS 1024` (`imodmop.c:45`).
const MAX_TUBE_DIAMS: i32 = 1024;
/// `#define TABLE_SIZE 1000` (`imodmop.c:46`).
const TABLE_SIZE: i32 = 1000;
/// `iimage.h:123`: `#define MAX_HALF_FLOAT 65504.`, a double.
const MAX_HALF_FLOAT: f64 = 65504.;

/// `b3dutil.h`: `#define B3DNINT(a) (int)floor((a) + 0.5)`; the `0.5` is a
/// double, so a float argument is widened before the add.
macro_rules! b3dnint {
    ($a:expr) => {
        (($a) as f64 + 0.5).floor() as i32
    };
}

/// `B3DMIN(a,b)`: `((a) < (b) ? (a) : (b))`.
macro_rules! b3dmin {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a < b { a } else { b }
    }};
}

/// `B3DMAX(a,b)`: `((a) > (b) ? (a) : (b))`.
macro_rules! b3dmax {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        if a > b { a } else { b }
    }};
}

/// `printf` with the source's format string.
macro_rules! printf {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {{
        let _ = ImodFile::Stdout.write_all(&c_format_bytes($fmt, &[$($arg),*]));
    }};
}

/// `exitError` with the source's variadic format.
macro_rules! exit_error_fmt {
    ($fmt:expr $(, $arg:expr)* $(,)?) => {
        exit_error(&c_format_bytes($fmt, &[$($arg),*]))
    };
}

/// The file-scope statics of `imodmop.c:48-63`.
struct Mop {
    /// `sScat2D`.
    scat2d: i32,
    /// `sScat3D`.
    scat3d: i32,
    /// `sTubes2D`.
    tubes2d: i32,
    /// `sUseScans`.
    use_scans: i32,
    /// `sBkgVal`.
    bkg_val: [f32; 4],
    /// `sMaskVal`.
    mask_val: [f32; 4],
    /// `*sLIp`, which is `main`'s `li`.
    li: LoadInfo,
    /// `sNumChan`.
    num_chan: i32,
    /// `sMasking`.
    masking: i32,
    /// `sTaper`.
    taper: i32,
    /// `sZtaper`.
    ztaper: i32,
    /// `sPadding`.
    padding: f32,
    /// `sWindowTable[TABLE_SIZE+1]`.
    window_table: [f32; TABLE_SIZE as usize + 1],
    /// `sRetainOutside`.
    retain_outside: i32,
    /// `sNoiseBordMin`.
    noise_bord_min: f32,
    /// `sNoiseBordMax`.
    noise_bord_max: f32,
    /// `sNoiseSeed`.
    noise_seed: i32,
}

/// `imodUsageHeader` as PIP's header callback.
fn imod_usage_header_for_pip(prog_name: &[u8]) {
    imod_usage_header(Some(&String::from_utf8_lossy(prog_name)));
    let _ = ImodFile::Stdout.flush();
}

/// C `parselist` on a PIP string: `Some(list)` or `None` with the C's
/// `*nlist` code (0 for an empty entry, -1 for a leading `/`, -3 for a bad
/// character).
fn parse_list_c(text: &str, count: &mut i32) -> Option<Vec<i32>> {
    match parselist(text) {
        Ok(list) if !text.is_empty() => {
            *count = list.len() as i32;
            Some(list)
        }
        Ok(_) => {
            *count = 0;
            None
        }
        Err(ParseListError::LeadingSlash) => {
            *count = -1;
            None
        }
        Err(ParseListError::InvalidCharacter) => {
            *count = -3;
            None
        }
    }
}

/// Original: `main` (`imodmop.c:65`).
pub fn imodmop() {
    let argv = program_args();
    let mut num_obj = 0;
    let mut retain = 0;
    let mut const_scale = 0;
    let mut num_diams: i32;
    let mut num_tubes = 0;
    let mut num_label = 0;
    let mut border = -1000;
    let mut num_allsec = 0;
    let mut black = 0;
    let mut white = 255;
    let mut bkg_fill: f32 = 0.;
    let mut bkg_red: f32 = 1.;
    let mut bkg_green: f32 = 1.;
    let mut bkg_blue: f32 = 1.;
    let mut obj_list: Option<Vec<i32>> = None;
    let mut tube_list: Option<Vec<i32>> = None;
    let mut label_list: Option<Vec<i32>> = None;
    let mut allsec_list: Option<Vec<i32>> = None;
    let mut tube_diams = vec![0f32; MAX_TUBE_DIAMS as usize];
    let mut minpt = Ipoint::default();
    let mut maxpt = Ipoint::default();
    let mut hdata = MrcHeader::default();
    let mut hdbyte = MrcHeader::default();
    let mut hdout: Vec<MrcHeader> = (0..3).map(|_| MrcHeader::default()).collect();
    let mut recnames: [String; 3] = Default::default();
    let mut xyznames: [String; 3] = Default::default();
    let mut ix: i32 = 0;
    let mut dsize = 0;
    let mut csize = 0;
    let mut out_mode: i32;
    let mut invert = 0;
    let mut reverse = 0;
    let if_thresh: i32;
    let if_proj: i32;
    let mut rgb_out = 0;
    let num_files: i32;
    let mut allsec = 0;
    let mut num_opt_args = 0;
    let mut num_non_opt_args = 0;
    let mut start_tilt: f32 = 0.;
    let mut end_tilt: f32 = 0.;
    let mut inc_tilt: f32 = 0.;
    let mut thresh: f32 = 0.;
    let mut smin: f32 = 0.;
    let mut smax: f32 = 0.;
    let rev_max: f32;
    let mut size: f32;
    let mut mask: f32 = 0.;
    let mut val = [0f32; 4];
    let mut pval = [0f32; 4];
    let mut axis = String::from("Y");
    let mut temp_dir = String::from(".");

    /* Fallbacks from    ../manpages/autodoc2man 2 1 imodmop  */
    let num_options = 34;
    let options: [&[u8]; 34] = [
        b"xminmax:XMinAndMax:IP:",
        b"yminmax:YMinAndMax:IP:",
        b"zminmax:ZMinAndMax:IP:",
        b"border:BorderAroundObjects:I:",
        b"invert:InvertPaintedArea:B:",
        b"noise:NoiseFillBorder:FP:",
        b"reverse:ReverseContrast:B:",
        b"thresh:Threshold:F:",
        b"fv:FillValue:F:",
        b"fc:FillColor:FT:",
        b"mask:MaskValue:F:",
        b"label:LabelMaskList:LI:",
        b"retain:RetainOutsideMask:B:",
        b"mode:ModeToOutput:I:",
        b"pad:PaddingSize:F:",
        b"taper:TaperOverPad:I:",
        b"ztaper:TaperInZOverPad:B:",
        b"objects:ObjectsToDo:LI:",
        b"2dscat:2DScatteredPoints:B:",
        b"3dscat:3DScatteredPoints:B:",
        b"tube:TubeObjects:LI:",
        b"diam:DiameterForTubes:F:",
        b"planar:PlanarTubes:B:",
        b"allsec:AllSectionObjects:LI:",
        b"color:ColorOutput:B:",
        b"scale:ScalingMinMax:FP:",
        b"project:ProjectTiltSeries:FT:",
        b"axis:AxisToTiltAround:CH:",
        b"constant:ConstantScaling:B:",
        b"bw:BlackAndWhite:IP:",
        b"tempdir:TemporaryDirectory:CH:",
        b"keep:KeepTempFiles:B:",
        b"fast:FastLegacyMethod:B:",
        b"help:usage:B:",
    ];

    let progname = imod_prog_name(argv.first().map(String::as_str).unwrap_or(""));
    let mut mop = Mop {
        scat2d: 0,
        scat3d: 0,
        tubes2d: 0,
        use_scans: 0,
        bkg_val: [0.; 4],
        mask_val: [0.; 4],
        li: LoadInfo::default(),
        num_chan: 0,
        masking: 0,
        taper: 0,
        ztaper: 0,
        padding: 0.,
        window_table: [0.; TABLE_SIZE as usize + 1],
        retain_outside: 0,
        noise_bord_min: 0.,
        noise_bord_max: 0.,
        noise_seed: -1,
    };
    mrc_init_li(Some(&mut mop.li), None);

    /* Startup with fallback */
    let argv_bytes: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
    pip_read_or_parse_options(
        argv_bytes.len() as i32,
        &argv_bytes,
        &options,
        num_options,
        progname.as_bytes(),
        0,
        2,
        1,
        &mut num_opt_args,
        &mut num_non_opt_args,
        Some(imod_usage_header_for_pip),
    );
    if pip_get_boolean(b"usage", &mut ix) == 0 {
        pip_print_help(progname.as_bytes(), 0, 2, 1);
        exit(0);
    }

    /* Need special usage statement because of 3 filename arguments */
    if num_non_opt_args < 3 {
        imod_version(Some(&progname));
        imod_copyright();
        printf!(
            "Usage: %s [options] model_file input_image output_image\n",
            CArg::Str(&progname)
        );
        if num_non_opt_args == 0 {
            pip_print_help(progname.as_bytes(), 0, 2, 1);
        }
        exit(3);
    }

    let modfile =
        String::from_utf8_lossy(&pip_get_non_option_arg(0).unwrap_or_default()).into_owned();
    let infile =
        String::from_utf8_lossy(&pip_get_non_option_arg(1).unwrap_or_default()).into_owned();
    let outfile =
        String::from_utf8_lossy(&pip_get_non_option_arg(2).unwrap_or_default()).into_owned();

    /* Get input model */
    let mut imod: Imod = match imod_read(&modfile) {
        Ok(model) => model,
        Err(_) => exit_error_fmt!("Reading model file %s", CArg::Str(&modfile)),
    };

    /* Open input image, read header */
    let Some(mut gfin) = ii_fopen(infile.as_bytes(), "rb") else {
        exit_error_fmt!("Could not open %s", CArg::Str(&infile))
    };
    if mrc_head_read(&mut gfin, &mut hdata) != 0 {
        exit_error_fmt!("Reading header of %s", CArg::Str(&infile));
    }

    /* Get more options after setting defaults */
    mop.li.xmin = 0;
    mop.li.ymin = 0;
    mop.li.zmin = 0;
    mop.li.xmax = hdata.nx - 1;
    mop.li.ymax = hdata.ny - 1;
    mop.li.zmax = hdata.nz - 1;
    let if_xin = 1 - pip_get_two_integers(b"XMinAndMax", &mut mop.li.xmin, &mut mop.li.xmax);
    let if_yin = 1 - pip_get_two_integers(b"YMinAndMax", &mut mop.li.ymin, &mut mop.li.ymax);
    let if_zin = 1 - pip_get_two_integers(b"ZMinAndMax", &mut mop.li.zmin, &mut mop.li.zmax);
    mop.li.xmin = b3dmax!(0, mop.li.xmin);
    mop.li.ymin = b3dmax!(0, mop.li.ymin);
    mop.li.zmin = b3dmax!(0, mop.li.zmin);
    mop.li.xmax = b3dmin!(hdata.nx - 1, mop.li.xmax);
    mop.li.ymax = b3dmin!(hdata.ny - 1, mop.li.ymax);
    mop.li.zmax = b3dmin!(hdata.nz - 1, mop.li.zmax);
    if mop.li.xmax <= mop.li.xmin || mop.li.ymax <= mop.li.ymin || mop.li.zmax < mop.li.zmin {
        exit_error(b"Coordinate limits are out of order");
    }

    pip_get_boolean(b"InvertPaintedArea", &mut invert);
    pip_get_boolean(b"ReverseContrast", &mut reverse);
    pip_get_boolean(b"ColorOutput", &mut rgb_out);
    if_thresh = 1 - pip_get_float(b"Threshold", &mut thresh);
    mop.masking = 1 - pip_get_float(b"MaskValue", &mut mask);
    pip_get_float(b"PaddingSize", &mut mop.padding);
    pip_get_integer(b"TaperOverPad", &mut mop.taper);
    if mop.taper > 1 {
        for i in 0..=TABLE_SIZE {
            // `(TABLE_SIZE - i) * 2.5 / TABLE_SIZE` is double, stored in float
            size = ((TABLE_SIZE - i) as f64 * 2.5 / TABLE_SIZE as f64) as f32;
            mop.window_table[i as usize] = ((-size * size) as f64 / 2.).exp() as f32;
        }
    }
    pip_get_boolean(b"TaperInZOverPad", &mut mop.ztaper);
    if mop.ztaper != 0 && mop.padding > 0. {
        exit_error(b"Tapering in Z cannot be done with padding outside, only inside");
    }

    /* Projection option */
    if_proj = 1 - pip_get_three_floats(
        b"ProjectTiltSeries",
        &mut start_tilt,
        &mut end_tilt,
        &mut inc_tilt,
    );
    if if_proj != 0 {
        let mut text: Vec<u8> = Vec::new();
        if pip_get_string(b"TemporaryDirectory", &mut text) == 0 {
            temp_dir = String::from_utf8_lossy(&text).into_owned();
        }
        pip_get_boolean(b"ConstantScaling", &mut const_scale);
        pip_get_two_integers(b"BlackAndWhite", &mut black, &mut white);
        let mut text: Vec<u8> = Vec::new();
        if pip_get_string(b"AxisToTiltAround", &mut text) == 0 {
            axis = String::from_utf8_lossy(&text).into_owned();
        }
        // Fixed in translation (BUGS.md, imodmop): the source tests
        // `!strcmp(axis, "X") && !strcmp(axis, "Y") && !strcmp(axis, "Z")`,
        // which can never be true, so a bad axis was only rejected later by
        // xyzproj.  The evident intent -- reject anything but X, Y or Z -- is
        // what is done here.
        if axis != "X" && axis != "Y" && axis != "Z" {
            exit_error(b"Axis extry must be X, Y, or Z");
        }

        if (start_tilt > end_tilt) && (inc_tilt > 0.) {
            exit_error(b"Ending tilt must be > starting tilt with a positive increment");
        }
        if (end_tilt > start_tilt) && (inc_tilt < 0.) {
            exit_error(b"Ending tilt must be < starting tilt with a negative increment");
        }
        if inc_tilt == 0. && end_tilt == start_tilt {
            exit_error(b"Tilt increment must be non-zero unless start equals end");
        }

        if black < 0 || black > 255 || white < 0 || white > 255 || black == white {
            exit_error(b"Black/white values must be in range 0-255 and nonequal");
        }
    }

    pip_get_boolean(b"KeepTempFiles", &mut retain);
    pip_get_boolean(b"FastLegacyMethod", &mut mop.use_scans);
    pip_get_boolean(b"2DScatteredPoints", &mut mop.scat2d);
    pip_get_boolean(b"3DScatteredPoints", &mut mop.scat3d);
    pip_get_boolean(b"PlanarTubes", &mut mop.tubes2d);
    if mop.scat2d != 0 && mop.scat3d != 0 {
        exit_error(b"You can not enter both -2 and -3");
    }
    if pip_get_integer(b"BorderAroundObjects", &mut border) == 0 {
        if if_xin + if_yin != 0 {
            exit_error(b"You can not use -border with -xminmax or -yminmax");
        }
        if border < 0 {
            exit_error(b"The border must be >= 0");
        }
    }

    // Get noise filling option and check validity
    if pip_get_two_floats(
        b"NoiseFillBorder",
        &mut mop.noise_bord_min,
        &mut mop.noise_bord_max,
    ) == 0
    {
        if mop.noise_bord_max < 0.5 {
            exit_error(b"The maximum border distance must be at least 0.5 pixels");
        }
        if mop.noise_bord_min < 0. || mop.noise_bord_min as f64 > mop.noise_bord_max as f64 - 0.5 {
            exit_error(
                b"The minimum border distance must not be negative or too close to the maximum",
            );
        }
        if mop.padding != 0. {
            exit_error(b"Noise filling cannot be done with padding");
        }
        if mop.use_scans != 0 {
            exit_error(b"Noise filling cannot be done with the legacy fast method");
        }
        if hdata.mode == MRC_MODE_RGB {
            exit_error(b"Noise filing cannot be done with RGB data");
        }
        invert = 1;
    }

    /* Get object and tube lists */
    let mut list_string: Vec<u8> = Vec::new();
    if pip_get_string(b"ObjectsToDo", &mut list_string) == 0 {
        obj_list = parse_list_c(&String::from_utf8_lossy(&list_string), &mut num_obj);
        if obj_list.is_none() {
            exit_error(b"Bad entry in list of objects to do");
        }
    }

    let mut list_string: Vec<u8> = Vec::new();
    if pip_get_string(b"TubeObjects", &mut list_string) == 0 {
        tube_list = parse_list_c(&String::from_utf8_lossy(&list_string), &mut num_tubes);
        if tube_list.is_none() {
            exit_error(b"Bad entry in list of tube objects");
        }

        num_diams = 0;
        if pip_get_float_array(
            b"DiameterForTubes",
            &mut tube_diams,
            &mut num_diams,
            MAX_TUBE_DIAMS,
        ) != 0
        {
            num_diams = 1;
            tube_diams[0] = 25.;
        }
        if num_diams != 1 && num_diams != num_tubes {
            exit_error(b"You must enter either 1 diameter or 1 per selected tube object");
        }
        for i in num_diams..num_tubes {
            tube_diams[i as usize] = tube_diams[0];
        }
    }

    /* Get label mask value list */
    let mut list_string: Vec<u8> = Vec::new();
    if pip_get_string(b"LabelMaskList", &mut list_string) == 0 {
        if mop.masking != 0 {
            exit_error(b"You cannot enter both -mask and -label options");
        }
        mop.masking = 1;
        label_list = parse_list_c(&String::from_utf8_lossy(&list_string), &mut num_label);
        let num_need = if num_obj > 0 {
            num_obj
        } else {
            imod.obj.len() as i32
        };
        if label_list.is_none() {
            /* Make sure it is a / entry and allocate and make default list */
            if num_label != -1 {
                exit_error(b"Bad entry in list of label mask values");
            }
            let mut list = vec![0i32; num_need.max(0) as usize];
            for i in 0..num_need as usize {
                list[i] = if num_obj > 0 {
                    obj_list.as_ref().unwrap()[i]
                } else {
                    i as i32 + 1
                };
            }
            label_list = Some(list);
            num_label = num_need;
        } else if num_need != num_label {
            exit_error_fmt!(
                "The label mask list has %d values and %d are needed",
                CArg::Int(num_label as i64),
                CArg::Int(num_need as i64)
            );
        }
    }
    pip_get_boolean(b"RetainOutsideMask", &mut mop.retain_outside);
    if mop.retain_outside != 0 && mop.masking == 0 {
        exit_error(b"You can use -retain only when outputting a mask");
    }
    if mop.retain_outside != 0 && (rgb_out != 0 || invert != 0) {
        exit_error(b"You cannot use -retain with -invert or -color");
    }

    /* Get all-section list and check it, count number of actual objects */
    let mut list_string: Vec<u8> = Vec::new();
    if pip_get_string(b"AllSectionObjects", &mut list_string) == 0 {
        allsec_list = parse_list_c(&String::from_utf8_lossy(&list_string), &mut num_allsec);
        let Some(list) = allsec_list.as_ref() else {
            exit_error(b"Bad entry in list of all-section objects")
        };
        for i in 0..num_allsec as usize {
            let objnum = list[i] - 1;
            if objnum < 0
                || objnum >= imod.obj.len() as i32
                || item_on_list(objnum + 1, obj_list.as_deref(), num_obj) < -1
            {
                continue;
            }
            allsec += 1;
            if iobj_open(imod.obj[objnum as usize].flags) != 0 {
                exit_error_fmt!(
                    "You cannot include an open contour object (%d) on an all-section list",
                    CArg::Int(objnum as i64 + 1)
                );
            }
            if iobj_scat(imod.obj[objnum as usize].flags) != 0
                && (mop.scat3d != 0 || mop.scat2d == 0)
            {
                exit_error_fmt!(
                    "You cannot include a scattered point object (%d) on an all-section list unless you specify -2dscat",
                    CArg::Int(objnum as i64 + 1)
                );
            }
        }
    }

    /* Get new mode if appropriate */
    out_mode = hdata.mode;
    if pip_get_integer(b"ModeToOutput", &mut out_mode) == 0 {
        if hdata.mode == MRC_MODE_COMPLEX_FLOAT || hdata.mode == MRC_MODE_COMPLEX_SHORT {
            exit_error(b"You cannot change the output mode for FFT data");
        }
        if if_proj != 0 || rgb_out != 0 {
            exit_error(b"You cannot change the output mode if using -color or -project");
        }
        if hdata.mode == MRC_MODE_RGB && mop.masking == 0 {
            exit_error(
                b"You cannot change the output mode with RGB input unless you are making a mask",
            );
        }
        out_mode = set_float_output_for_entered_mode(out_mode);
    }

    /* Take care of fill value and color, set up the background fill value
    for cases of gray or color output.  If a value is entered, it gets
    reversed for reverse contrast, before testing */
    if pip_get_float(b"FillValue", &mut bkg_fill) == 0 && reverse != 0 {
        bkg_fill = hdata.amax - bkg_fill;
    }
    pip_get_three_floats(b"FillColor", &mut bkg_red, &mut bkg_green, &mut bkg_blue);
    if bkg_red < 0.
        || bkg_red > 1.
        || bkg_green < 0.
        || bkg_green > 1.
        || bkg_blue < 0.
        || bkg_blue > 1.
    {
        exit_error(b"Red, green, blue fill color values must be between 0 and 1");
    }
    if ((out_mode == MRC_MODE_RGB || out_mode == MRC_MODE_BYTE)
        && (bkg_fill < 0. || bkg_fill > 255.))
        || (out_mode == MRC_MODE_SHORT && (bkg_fill < -32767. || bkg_fill > 32767.))
        || (out_mode == MRC_MODE_USHORT && (bkg_fill < 0. || bkg_fill > 65535.))
        || (out_mode == MRC_MODE_FLOAT
            && write_16_bit_mode_for_floats() != 0
            && (bkg_fill as f64).abs() > MAX_HALF_FLOAT)
    {
        exit_error(b"Fill value is outside allowed range for output data mode");
    }
    if mop.masking != 0 {
        for i in 0..b3dmax!(1, num_label) {
            if num_label != 0 {
                mask = label_list.as_ref().unwrap()[i as usize] as f32;
            }
            if ((out_mode == MRC_MODE_RGB || out_mode == MRC_MODE_BYTE)
                && (mask < 0. || mask > 255.))
                || (out_mode == MRC_MODE_SHORT && (mask < -32767. || mask > 32767.))
                || (out_mode == MRC_MODE_USHORT && (mask < 0. || mask > 65535.))
                || (out_mode == MRC_MODE_FLOAT
                    && write_16_bit_mode_for_floats() != 0
                    && (mask as f64).abs() > MAX_HALF_FLOAT)
            {
                exit_error_fmt!(
                    "Mask value %g is outside allowed range for output data mode",
                    CArg::Dbl(mask as f64)
                );
            }
            if mask == bkg_fill || (out_mode != MRC_MODE_FLOAT && mask as i32 == bkg_fill as i32) {
                exit_error_fmt!(
                    "Mask value %g is indistinguishable from fill value %g",
                    CArg::Dbl(mask as f64),
                    CArg::Dbl(bkg_fill as f64)
                );
            }
        }
    }

    if hdata.mode == MRC_MODE_RGB || rgb_out != 0 {
        mop.bkg_val[0] = bkg_red * bkg_fill;
        mop.bkg_val[1] = bkg_green * bkg_fill;
        mop.bkg_val[2] = bkg_blue * bkg_fill;
    } else {
        mop.bkg_val[0] = bkg_fill;
        mop.bkg_val[1] = bkg_fill;
        mop.bkg_val[2] = bkg_fill;
    }

    /* Check for incompatible options */
    if hdata.mode == MRC_MODE_COMPLEX_FLOAT || hdata.mode == MRC_MODE_COMPLEX_SHORT {
        if reverse != 0 || rgb_out != 0 || if_proj != 0 {
            exit_error(b"You can not use -reverse, -color, or -project with FFT data");
        }
        if if_xin != 0 || if_yin != 0 {
            exit_error(b"You can not limit the X or Y range with FFT data");
        }
        if if_zin != 0 && mop.scat3d != 0 {
            exit_error(b"You can not limit the Z range of 3D FFT data");
        }
    }

    if hdata.mode == MRC_MODE_RGB && (rgb_out != 0 || if_proj != 0) {
        exit_error(b"You can not use -color or -project with RGB input data");
    }

    if rgb_out != 0 && invert != 0 {
        exit_error(b"You can not use -invert when making colored data");
    }

    if mrc_getdcsize(hdata.mode, &mut dsize, &mut csize) != 0 {
        exit_error_fmt!(
            "Unsupported input data mode %d",
            CArg::Int(hdata.mode as i64)
        );
    }

    /* Determine limits with border option */
    if border >= 0 {
        mop.li.xmin = 10000000;
        mop.li.ymin = 10000000;
        mop.li.xmax = -10000000;
        mop.li.ymax = -10000000;
        let add_pad: f32 = if mop.padding > 0. { mop.padding } else { 0. };
        if if_zin == 0 && allsec == 0 {
            mop.li.zmin = 10000000;
            mop.li.zmax = -10000000;
        }
        for objnum in 0..imod.obj.len() {
            if item_on_list(objnum as i32 + 1, obj_list.as_deref(), num_obj) < -1 {
                continue;
            }
            let obj = &imod.obj[objnum];

            if iobj_open(obj.flags) != 0 {
                let i = item_on_list(objnum as i32 + 1, tube_list.as_deref(), num_tubes);
                if i >= 0 {
                    /* For a tube, add radius to bounding box in X/Y/Z */
                    imod_object_get_bbox(obj, &mut minpt, &mut maxpt);
                    let delta = tube_diams[i as usize] / 2. + add_pad;
                    minpt.x -= delta;
                    minpt.y -= delta;
                    maxpt.x += delta;
                    maxpt.y += delta;
                    if mop.tubes2d == 0 {
                        minpt.z -= delta;
                        maxpt.z += delta;
                    }
                } else {
                    // Fixed in translation (BUGS.md, imodmop): the source
                    // falls through with `minpt`/`maxpt` from the previous
                    // object, or uninitialised stack memory for the first
                    // one.  A repeat of an earlier object's box changes no
                    // min or max, so skipping the object is what native
                    // does whenever its behaviour is defined.
                    continue;
                }
            } else if iobj_scat(obj.flags) != 0 {
                if mop.scat2d != 0 || mop.scat3d != 0 {
                    /* For scattered point, allow for radius of each point */
                    minpt.x = 1.0e30;
                    minpt.y = 1.0e30;
                    minpt.z = 1.0e30;
                    maxpt.x = -1.0e30;
                    maxpt.y = -1.0e30;
                    maxpt.z = -1.0e30;
                    for co in 0..obj.cont.len() {
                        for pt in 0..obj.cont[co].pts.len() {
                            size = imod_point_get_size(obj, &obj.cont[co], pt as i32) + add_pad;
                            let pts = &obj.cont[co].pts[pt];
                            minpt.x = b3dmin!(minpt.x, pts.x - size);
                            minpt.y = b3dmin!(minpt.y, pts.y - size);
                            minpt.z =
                                b3dmin!(minpt.z, pts.z - if mop.scat3d != 0 { size } else { 0. });
                            maxpt.x = b3dmax!(maxpt.x, pts.x + size);
                            maxpt.y = b3dmax!(maxpt.y, pts.y + size);
                            maxpt.z =
                                b3dmax!(maxpt.z, pts.z + if mop.scat3d != 0 { size } else { 0. });
                        }
                    }
                } else {
                    // Same fix as for an open object not on the tube list.
                    continue;
                }
            } else {
                imod_object_get_bbox(obj, &mut minpt, &mut maxpt);
                minpt.x -= add_pad;
                minpt.y -= add_pad;
                maxpt.x += add_pad;
                maxpt.y += add_pad;
            }

            /* Form mins and maxes over all objects */
            mop.li.xmin = b3dmin!(mop.li.xmin, b3dnint!(minpt.x - border as f32));
            mop.li.ymin = b3dmin!(mop.li.ymin, b3dnint!(minpt.y - border as f32));
            mop.li.xmax = b3dmax!(mop.li.xmax, b3dnint!(maxpt.x + border as f32));
            mop.li.ymax = b3dmax!(mop.li.ymax, b3dnint!(maxpt.y + border as f32));
            if if_zin == 0 && allsec == 0 {
                mop.li.zmin = b3dmin!(mop.li.zmin, b3dnint!(minpt.z));
                mop.li.zmax = b3dmax!(mop.li.zmax, b3dnint!(maxpt.z));
            }
        }

        /* Limit mins and maxes and check */
        mop.li.xmin = b3dmax!(0, mop.li.xmin);
        mop.li.ymin = b3dmax!(0, mop.li.ymin);
        mop.li.zmin = b3dmax!(0, mop.li.zmin);
        mop.li.xmax = b3dmin!(hdata.nx - 1, mop.li.xmax);
        mop.li.ymax = b3dmin!(hdata.ny - 1, mop.li.ymax);
        mop.li.zmax = b3dmin!(hdata.nz - 1, mop.li.zmax);
        if mop.li.xmax <= mop.li.xmin || mop.li.ymax <= mop.li.ymin || mop.li.zmax < mop.li.zmin {
            exit_error(b"There are no objects to define the volume to output");
        }
    }

    /* Set up scaling for rgb volume output; default for byte input is no
    scaling */
    if rgb_out != 0 && if_proj == 0 {
        smin = hdata.amin;
        smax = hdata.amax;
        if hdata.mode == MRC_MODE_BYTE {
            smin = 0.;
            smax = 255.;
        }
        if pip_get_two_floats(b"ScalingMinMax", &mut smin, &mut smax) == 0
            || hdata.mode != MRC_MODE_BYTE
        {
            if reverse != 0 {
                val[0] = hdata.amax - smin;
                smin = hdata.amax - smax;
                smax = val[0];
            }
            if smin >= smax {
                exit_error(b"Minimum density for scaling should be less than maximum");
            }
        }
    }

    /* Reverse threshold so user can enter in terms of original units */
    if reverse != 0 && if_thresh != 0 {
        thresh = hdata.amax - thresh;
    }
    pip_done();

    /* Shift the model if it was loaded on a subset (takes care of
    mirrored/nonmirrored FFT problem) */
    if hdata.mode == MRC_MODE_COMPLEX_FLOAT || hdata.mode == MRC_MODE_COMPLEX_SHORT {
        imod_trans_for_subset_load(&mut imod, &hdata, None);
    } else if imod.ref_image.is_some() {
        /* Or scale the data as with imodtrans -i */
        if imod_trans_to_match_image(&mut imod, &hdata, 0).is_err() {
            exit_error(b"Memory error transforming model to match image");
        }
    }

    /* Open final output file now */
    if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none() {
        imod_backup_file(&outfile);
    }
    let Some(gfout) = ii_fopen(outfile.as_bytes(), "wb") else {
        exit_error_fmt!("Could not open %s", CArg::Str(&outfile))
    };
    let mut gfout = Some(gfout);

    /* Open temp files for projections */
    mop.num_chan = if rgb_out != 0 { 3 } else { 1 };
    let mut files: Vec<ImodFile> = Vec::new();
    if if_proj != 0 {
        num_files = mop.num_chan;
        let pid = std::process::id() as i32;
        for i in 0..mop.num_chan as usize {
            recnames[i] = format!("{}/{}.rec{}.{}", temp_dir, progname, i, pid);
            xyznames[i] = format!("{}/{}.xyz{}.{}", temp_dir, progname, i, pid);
            match ii_fopen(recnames[i].as_bytes(), "wb") {
                Some(file) => files.push(file),
                None => exit_error_fmt!("Could not open %s\n", CArg::Str(&recnames[i])),
            }
        }
    } else {
        // `files[0] = gfout`: the same stream, closed through `files[0]`.
        num_files = 1;
        files.push(gfout.take().unwrap());
    }

    /* Set up output file(s) in correct mode and size */
    let nxout = mop.li.xmax + 1 - mop.li.xmin;
    let nyout = mop.li.ymax + 1 - mop.li.ymin;
    let nzout = mop.li.zmax + 1 - mop.li.zmin;
    if rgb_out != 0 && if_proj == 0 {
        out_mode = MRC_MODE_RGB;
    }
    if mop.masking != 0 && out_mode == MRC_MODE_BYTE {
        override_write_bytes(0);
    }

    for i in 0..num_files as usize {
        mrc_head_new(&mut hdout[i], nxout, nyout, nzout, out_mode);
        mrc_coord_cp(&mut hdout[i], &hdata);
        mrc_head_label_cp(&hdata, &mut hdout[i]);
        let com_str = format!("{}: Painted image from model", progname);
        mrc_head_label(&mut hdout[i], com_str.as_bytes());
        hdout[i].amin = 1.0e30;
        hdout[i].amax = -1.0e30;
        hdout[i].amean = 0.;
    }

    /* Get slices for input and output data */
    let Some(mut islice) = slice_create(nxout, nyout, hdata.mode) else {
        exit_error(b"Memory error creating slice for input")
    };
    let mut pslice: Vec<Islice> = Vec::new();
    for _ in 0..mop.num_chan {
        match slice_create(nxout, nyout, out_mode) {
            Some(slice) => pslice.push(slice),
            None => exit_error(b"Memory error creating slice for painted data"),
        }
    }
    let mut rgbslice: Option<Islice> = None;
    if rgb_out != 0 && if_proj == 0 {
        rgbslice = slice_create(nxout, nyout, MRC_MODE_RGB);
        if rgbslice.is_none() {
            exit_error(b"Memory error creating slice for RGB painted data");
        }
    }

    /* Loop on slices */
    rev_max = if hdata.mode == MRC_MODE_RGB {
        255.
    } else {
        hdata.amax
    };
    for k in mop.li.zmin..=mop.li.zmax {
        printf!("Painting section %d ", CArg::Int(k as i64));
        let _ = ImodFile::Stdout.flush();
        if (mop.masking == 0 || mop.retain_outside != 0)
            && mrc_read_z(&mut hdata, &mut mop.li, islice.data.bytes_mut(), k) != 0
        {
            exit_error_fmt!("Reading data from file for slice %d", CArg::Int(k as i64));
        }
        if mop.retain_outside != 0 {
            val[1] = 0.;
            val[2] = 0.;
            for iy in 0..nyout {
                for ix in 0..nxout {
                    slice_get_val(&islice, ix, iy, &mut val);
                    slice_put_val(&mut pslice[0], ix, iy, val);
                }
            }
        }

        /* Apply contrast reversal and/or thresholding */
        if reverse != 0 || if_thresh != 0 {
            val[1] = 0.;
            val[2] = 0.;
            for iy in 0..nyout {
                for ix in 0..nxout {
                    slice_get_val(&islice, ix, iy, &mut val);
                    if reverse != 0 {
                        for i in 0..csize as usize {
                            val[i] = rev_max - val[i];
                        }
                    }

                    if if_thresh != 0 && slice_get_val_magnitude(val, hdata.mode) < thresh {
                        for i in 0..csize as usize {
                            val[i] = 0.;
                        }
                    }
                    slice_put_val(&mut islice, ix, iy, val);
                }
            }
        }

        /* Clear the slice with the background value */
        val[0] = 0.;
        val[1] = 0.;
        val[2] = 0.;
        if mop.retain_outside == 0 {
            if mop.num_chan == 1 {
                slice_clear(&mut pslice[0], mop.bkg_val);
            } else {
                for i in 0..mop.num_chan as usize {
                    val[0] = mop.bkg_val[i];
                    slice_clear(&mut pslice[i], val);
                }
            }
        }

        /* Do two passes through objects, doing inside-out ones on second pass */
        for inside in 0..2 {
            for objnum in 0..imod.obj.len() {
                /* Check if object on list */
                if item_on_list(objnum as i32 + 1, obj_list.as_deref(), num_obj) < -1 {
                    continue;
                }

                /* Get object specific mask if any and then set mask output val */
                if num_label != 0 {
                    let labels = label_list.as_ref().unwrap();
                    if num_obj != 0 {
                        let objs = obj_list.as_ref().unwrap();
                        for i in 0..num_obj as usize {
                            if objs[i] == objnum as i32 + 1 {
                                mask = labels[i] as f32;
                            }
                        }
                    } else {
                        mask = labels[objnum] as f32;
                    }
                }
                mop.mask_val[0] = mask;
                mop.mask_val[1] = mask;
                mop.mask_val[2] = mask;
                if hdata.mode == MRC_MODE_COMPLEX_FLOAT || hdata.mode == MRC_MODE_COMPLEX_SHORT {
                    mop.mask_val[1] = 0.;
                    mop.mask_val[2] = 0.;
                }

                allsec = if item_on_list(objnum as i32 + 1, allsec_list.as_deref(), num_allsec) >= 0
                {
                    1
                } else {
                    0
                };
                let obj = &mut imod.obj[objnum];

                /* Do open objects if on tube list */
                if iobj_open(obj.flags) != 0 {
                    let i = item_on_list(objnum as i32 + 1, tube_list.as_deref(), num_tubes);
                    if i >= 0 && inside == 0 {
                        paint_tubular_lines(
                            &mop,
                            obj,
                            &islice,
                            &mut pslice,
                            tube_diams[i as usize] / 2.,
                            k,
                        );
                    }

                /* Do scattered objects if specified and on first pass only */
                } else if iobj_scat(obj.flags) != 0 {
                    if inside == 0 && (mop.scat2d != 0 || mop.scat3d != 0) {
                        paint_scat_points(&mop, obj, &islice, &mut pslice, k, allsec);
                    }
                } else {
                    /* Do closed objects on appropriate pass */
                    if (inside != 0 && (obj.flags & IMOD_OBJFLAG_OUT) != 0)
                        || (inside == 0 && (obj.flags & IMOD_OBJFLAG_OUT) == 0)
                    {
                        if paint_contours(&mop, obj, &islice, &mut pslice, inside, k, allsec) != 0 {
                            exit_error_fmt!(
                                "Analyzing contours for object %d",
                                CArg::Int(objnum as i64 + 1)
                            );
                        }
                    }
                }
            }
        }

        /* To invert, subtract painted data from the original data or the mask */
        if invert != 0 {
            val[0] = mop.mask_val[0];
            val[1] = mop.mask_val[1];
            val[2] = mop.mask_val[2];
            for iy in 0..nyout {
                for ix in 0..nxout {
                    if mop.masking == 0 {
                        slice_get_val(&islice, ix, iy, &mut val);
                    }
                    slice_get_val(&pslice[0], ix, iy, &mut pval);
                    for i in 0..csize as usize {
                        pval[i] = val[i] - pval[i] + mop.bkg_val[i];
                    }
                    slice_put_val(&mut pslice[0], ix, iy, pval);
                }
            }
        }

        /* write data and maintain min/max/mean */
        for i in 0..num_files as usize {
            let oslice: &mut Islice = if rgb_out != 0 && if_proj == 0 {
                let rgb = rgbslice.as_mut().unwrap();
                scale_and_combine_slices(&pslice, rgb, smin, smax, 3);
                rgb
            } else {
                &mut pslice[i]
            };
            slice_mmm(oslice);
            hdout[i].amin = b3dmin!(hdout[i].amin, oslice.min);
            hdout[i].amax = b3dmax!(hdout[i].amax, oslice.max);
            hdout[i].amean += oslice.mean / nzout as f32;
            if mrc_write_slice(
                oslice.data.bytes(),
                &mut files[i],
                &mut hdout[i],
                k - mop.li.zmin,
                b'Z',
            ) != 0
            {
                exit_error_fmt!(
                    "Writing slice at Z = %d to file # %d",
                    CArg::Int((k - mop.li.zmin) as i64),
                    CArg::Int(i as i64 + 1)
                );
            }
        }
        printf!("\r");
    }
    printf!("\n");

    /* Write headers, close files, delete slices */
    for i in 0..num_files as usize {
        if mrc_head_write(&mut files[i], &mut hdout[i]) != 0 {
            exit_error_fmt!("Writing header to file # %d", CArg::Int(i as i64 + 1));
        }
        ii_fclose(&mut files[i]);
    }
    ii_fclose(&mut gfin);
    drop(islice);
    drop(pslice);
    drop(rgbslice);

    if if_proj == 0 {
        exit(0);
    }
    let mut gfout = gfout.unwrap();

    /* PROJECTING: Run xyzproj on the files */
    for i in 0..num_files as usize {
        printf!("Projecting file # %d...\n", CArg::Int(i as i64 + 1));
        let _ = ImodFile::Stdout.flush();
        let com_str = format!(
            "xyzproj -mode 2 -axis {} -angles {:.6},{:.6},{:.6} {} {} {} {}",
            axis,
            start_tilt,
            end_tilt,
            inc_tilt,
            if const_scale != 0 { "-const" } else { " " },
            if invert != 0 || bkg_fill != 0. {
                " "
            } else {
                "-fill 0"
            },
            recnames[i],
            xyznames[i]
        );
        // `system(comStr)`: xyzproj is one of our own commands, so it runs in
        // this process (CLAUDE.md, "Our own commands are called in
        // process"); the command line needs only word splitting.  `system`
        // returns the wait status, so a non-zero exit is reported as
        // `status << 8`, as the source prints it.
        let words: Vec<std::ffi::OsString> = com_str
            .split_whitespace()
            .map(std::ffi::OsString::from)
            .collect();
        ix = match crate::imod::commands::find("xyzproj") {
            Some(command) => {
                let mut argv = words;
                argv[0] = match std::env::current_exe() {
                    Ok(path) => path.with_file_name("xyzproj").into_os_string(),
                    Err(_) => std::ffi::OsString::from("xyzproj"),
                };
                match crate::imod::commands::run_in_process(command, argv, None, false) {
                    Ok((status, _)) => status << 8,
                    Err(_) => 127 << 8,
                }
            }
            None => 127 << 8,
        };
        if ix != 0 {
            exit_error_fmt!(
                "Running xyzproj on file %s (return value %d)",
                CArg::Str(&recnames[i]),
                CArg::Int(ix as i64)
            );
        }
        if retain == 0 {
            let _ = std::fs::remove_file(&recnames[i]);
        }
    }

    /* Open the files, determine size, set up output file */
    smin = 1.0e30;
    smax = -1.0e30;
    printf!("Scaling data into final output file...\n");
    files.clear();
    for i in 0..num_files as usize {
        match ii_fopen(xyznames[i].as_bytes(), "rb") {
            Some(file) => files.push(file),
            None => exit_error_fmt!("Could not open %s", CArg::Str(&xyznames[i])),
        }
        if mrc_head_read(&mut files[i], &mut hdout[i]) != 0 {
            exit_error_fmt!("Reading header of %s", CArg::Str(&xyznames[i]));
        }
        smin = b3dmin!(smin, hdout[i].amin);
        smax = b3dmax!(smax, hdout[i].amax);
    }

    out_mode = if rgb_out != 0 {
        MRC_MODE_RGB
    } else {
        MRC_MODE_BYTE
    };
    mrc_head_new(&mut hdbyte, hdout[0].nx, hdout[0].ny, hdout[0].nz, out_mode);
    mrc_head_label_cp(&hdata, &mut hdbyte);
    let com_str = format!("{}: Projected image painted from model", progname);
    mrc_head_label(&mut hdbyte, com_str.as_bytes());
    hdbyte.amin = 1.0e30;
    hdbyte.amax = -1.0e30;
    hdbyte.amean = 0.;

    /* Adjust scaling by finding the values that would map to black and white
    after mapping smin/smax to 0/255 */
    val[0] = (smin as f64 + (black as f32 * (smax - smin)) as f64 / 255.) as f32;
    smax = (smin as f64 + (white as f32 * (smax - smin)) as f64 / 255.) as f32;
    smin = val[0];

    /* Get slices for new size of data */
    let Some(mut oslice) = slice_create(hdout[0].nx, hdout[0].ny, out_mode) else {
        exit_error(b"Memory error creating slice for output")
    };
    let mut pslice: Vec<Islice> = Vec::new();
    for _ in 0..num_files {
        match slice_create(hdout[0].nx, hdout[0].ny, hdout[0].mode) {
            Some(slice) => pslice.push(slice),
            None => exit_error(b"Memory error creating slice for reading projections"),
        }
    }

    /* Read in the data and scale it into output, write it */
    for k in 0..hdout[0].nz {
        for i in 0..num_files as usize {
            if mrc_read_slice(
                pslice[i].data.bytes_mut(),
                &mut files[i],
                &mut hdout[i],
                k,
                b'Z',
            ) != 0
            {
                exit_error_fmt!(
                    "Reading slice %d from file # %d",
                    CArg::Int(k as i64),
                    CArg::Int(i as i64 + 1)
                );
            }
        }
        scale_and_combine_slices(&pslice, &mut oslice, smin, smax, num_files);
        slice_mmm(&mut oslice);
        hdbyte.amin = b3dmin!(hdbyte.amin, oslice.min);
        hdbyte.amax = b3dmax!(hdbyte.amax, oslice.max);
        hdbyte.amean += oslice.mean / hdbyte.nz as f32;
        if mrc_write_slice(oslice.data.bytes(), &mut gfout, &mut hdbyte, k, b'Z') != 0 {
            exit_error_fmt!(
                "Writing slice at Z = %d to final output file",
                CArg::Int(k as i64)
            );
        }
    }

    /* Finalize header and clean up */
    if mrc_head_write(&mut gfout, &mut hdbyte) != 0 {
        exit_error(b"Writing header to final output file");
    }
    ii_fclose(&mut gfout);
    for i in 0..num_files as usize {
        ii_fclose(&mut files[i]);
        if retain == 0 {
            let _ = std::fs::remove_file(&xyznames[i]);
        }
    }
    exit(0);
}

/// Original: `paintContours` (`imodmop.c:830`, static).
fn paint_contours(
    mop: &Mop,
    obj: &mut Iobj,
    islice: &Islice,
    pslice: &mut [Islice],
    inside: i32,
    iz: i32,
    allsec: i32,
) -> i32 {
    let mut zmin = 0;
    let mut zmax = 0;
    let mut nummax = 0;
    let mut contz: Vec<i32> = Vec::new();
    let mut numatz: Vec<i32> = Vec::new();
    let mut contatz: Vec<Vec<i32>> = Vec::new();
    let mut zlist: Vec<i32> = Vec::new();
    let mut zlsize = 0;
    let mut numwarn = -1;

    if obj.cont.is_empty() {
        return 0;
    }

    if imod_contour_make_z_tables(
        obj,
        1,
        0,
        &mut contz,
        &mut zlist,
        &mut numatz,
        &mut contatz,
        &mut zmin,
        &mut zmax,
        &mut zlsize,
        &mut nummax,
    ) != 0
    {
        return -1;
    }
    let obj: &Iobj = obj;

    if allsec == 0 && (iz < zmin || iz > zmax || numatz[(iz - zmin) as usize] == 0) {
        imod_contour_free_z_tables(
            &mut numatz,
            &mut contatz,
            &mut contz,
            &mut zlist,
            zmin,
            zmax,
        );
        return 0;
    }

    let mut indzst = iz - zmin;
    let mut indznd = iz - zmin;
    if allsec != 0 {
        indzst = 0;
        indznd = zmax - zmin;
    }
    for indz in indzst..=indznd {
        let nummax = numatz[indz as usize];
        if nummax == 0 {
            continue;
        }

        /* Allocate space for lists of min's max's, and scan contours */
        let mut pmin: Vec<Ipoint> = vec![Ipoint::default(); nummax as usize];
        let mut pmax: Vec<Ipoint> = vec![Ipoint::default(); nummax as usize];
        let mut scancont: Vec<Icont> = vec![Icont::default(); nummax as usize];
        let mut conum_in_obj: Vec<i32> = vec![0; nummax as usize];

        /* Get array for index to inside/outside information */
        let mut nestind: Vec<i32> = vec![0; nummax as usize];

        /* Make scan contours and get mins/maxes for ones with non-empty scans */
        let mut inbox: usize = 0;
        let mut numnests: i32 = 0;
        let mut nests: Vec<Nesting> = Vec::new();
        for kis in 0..numatz[indz as usize] as usize {
            let co = contatz[indz as usize][kis] as usize;
            let cont = &obj.cont[co];
            let mut scan = imodel_contour_scan(Some(cont)).unwrap_or_default();
            if !scan.pts.is_empty() {
                let (lower, upper) = imod_contour_get_bbox(Some(cont)).unwrap_or_default();
                pmin[inbox] = lower;
                pmax[inbox] = upper;
                scancont[inbox] = scan;
                conum_in_obj[inbox] = co as i32;
                nestind[inbox] = -1;
                inbox += 1;
            } else {
                imod_contour_delete(&mut scan);
            }
        }

        /* Look for overlapping contours as in imodmesh */
        for co in 0..inbox.saturating_sub(1) {
            for eco in (co + 1)..inbox {
                if imod_contour_check_nesting(
                    co as i32,
                    eco as i32,
                    &mut scancont[..inbox],
                    &pmin[..inbox],
                    &pmax[..inbox],
                    &mut nests,
                    &mut nestind[..inbox],
                    &mut numnests,
                    &mut numwarn,
                ) != 0
                {
                    return -1;
                }
            }
        }

        /* Analyze inside and outside contours to determine level, then go through
        contours from lowest level inward */
        imod_contour_nest_levels(&mut nests, &nestind[..inbox], numnests);
        let mut do_level = 1;
        loop {
            let mut found = 0;
            for co in 0..inbox {
                let mut level = 1;
                if nestind[co] >= 0 {
                    level = nests[nestind[co] as usize].level;
                }
                if level != do_level {
                    continue;
                }
                found = 1;
                if mop.use_scans != 0 {
                    paint_scan_contour(
                        mop,
                        obj,
                        &scancont[co],
                        islice,
                        pslice,
                        (level + inside) % 2,
                    );
                } else {
                    paint_exact_contour(
                        mop,
                        obj,
                        &obj.cont[conum_in_obj[co] as usize],
                        islice,
                        pslice,
                        (level + inside) % 2,
                        iz,
                        &numatz,
                        &contatz,
                        zmin,
                        zmax,
                    );
                }
            }
            do_level += 1;
            if found == 0 {
                break;
            }
        }

        /* clean up inside the nests */
        imod_contour_free_nests(&mut nests, numnests);

        /* clean up scan conversions */
        for co in 0..inbox {
            imod_contour_delete(&mut scancont[co]);
        }
    }
    imod_contour_free_z_tables(
        &mut numatz,
        &mut contatz,
        &mut contz,
        &mut zlist,
        zmin,
        zmax,
    );
    0
}

/// Original: `paintScanContour` (`imodmop.c:934`, static).
fn paint_scan_contour(
    mop: &Mop,
    obj: &Iobj,
    cont: &Icont,
    islice: &Islice,
    pslice: &mut [Islice],
    fill: i32,
) {
    let mut inval = [0f32; 4];
    let mut redval = [0f32; 4];
    let mut grnval = [0f32; 4];
    let mut bluval = [0f32; 4];
    let li = &mop.li;

    /* Initialize color values if not filling */
    if fill == 0 {
        redval[0] = mop.bkg_val[0];
        grnval[0] = mop.bkg_val[1];
        bluval[0] = mop.bkg_val[2];
    }

    let mut j = 0;
    while j < cont.pts.len() {
        /* Move Y down by 1 because otherwise bumps in X occur at the wrong Y
        height.  This is not perfect.  Without the -1, the boundary is up by 0.69 pixels
        and with it then are down 0.24 pixels on average. */
        let y = (cont.pts[j].y - 1.) as i32;

        /* Check if line is in the volume and get X limits */
        if y < li.ymin || y > li.ymax {
            j += 2;
            continue;
        }

        /* And here truncation matches the positions better than taking the nint,
        and running to < xnd instead of to <= xnd.  */
        let mut xst = cont.pts[j].x as i32;
        xst = b3dmax!(xst, li.xmin);
        let mut xnd = cont.pts[j + 1].x as i32;
        xnd = b3dmin!(xnd, li.xmax);
        for x in xst..xnd {
            /* Get value if filling */
            if fill != 0 {
                slice_get_val(islice, x - li.xmin, y - li.ymin, &mut inval);
                put_slice_value(mop, obj, pslice, x, y, inval, 1.);
            } else {
                /* Or output the background value */
                if mop.num_chan < 3 {
                    slice_put_val(&mut pslice[0], x - li.xmin, y - li.ymin, mop.bkg_val);
                } else {
                    slice_put_val(&mut pslice[0], x - li.xmin, y - li.ymin, redval);
                    slice_put_val(&mut pslice[1], x - li.xmin, y - li.ymin, grnval);
                    slice_put_val(&mut pslice[2], x - li.xmin, y - li.ymin, bluval);
                }
            }
        }
        j += 2;
    }
}

/// Original: `paintExactContour` (`imodmop.c:984`, static).
#[allow(clippy::too_many_arguments)]
fn paint_exact_contour(
    mop: &Mop,
    obj: &Iobj,
    cont: &Icont,
    islice: &Islice,
    pslice: &mut [Islice],
    fill: i32,
    iz: i32,
    numatz: &[i32],
    contatz: &[Vec<i32>],
    zmin: i32,
    zmax: i32,
) {
    let mut inval = [0f32; 4];
    let mut redval = [0f32; 4];
    let mut grnval = [0f32; 4];
    let mut bluval = [0f32; 4];
    let mut closest = 0;
    let mut num_bord = 0;
    let mut pnt = Ipoint::default();
    // `B3DMAX(0., fill ? sPadding : -sPadding)`: compared as double
    let pad_arg = if fill != 0 { mop.padding } else { -mop.padding };
    let more_pad: f32 = if 0. > pad_arg as f64 { 0. } else { pad_arg };
    let mut dist: f32;
    let mut taper_frac: f32;
    let mut min_zdist: f32;
    let mut bord_mean: f32 = 0.;
    let mut bord_sd: f32 = 0.;
    let mut bord_sum: f64 = 0.;
    let mut bord_sum_sq: f64 = 0.;
    let do_all_inside = (fill != 0 && mop.padding >= 0.) || (fill == 0 && mop.padding <= 0.);
    let look_outside = (fill != 0 && mop.padding > 0.) || (fill == 0 && mop.padding < 0.);
    let li = &mop.li;

    /* Initialize color values if not filling */
    if fill == 0 {
        redval[0] = mop.bkg_val[0];
        grnval[0] = mop.bkg_val[1];
        bluval[0] = mop.bkg_val[2];
    }
    let abs_pad = (mop.padding as f64).abs() as f32;
    let (pmin, pmax) = imod_contour_get_bbox(Some(cont)).unwrap_or_default();
    pnt.z = 0.;

    // If doing noise fill, find all the border points and get mean/sd
    if fill != 0 && mop.noise_bord_max > 0. {
        let xst = b3dmax!((pmin.x - 2. - mop.noise_bord_max) as i32, li.xmin);
        let xnd = b3dmin!((pmax.x + 3. + mop.noise_bord_max) as i32, li.xmax);
        let yst = b3dmax!((pmin.y - 2. - mop.noise_bord_max) as i32, li.ymin);
        let ynd = b3dmin!((pmax.y + 3. + mop.noise_bord_max) as i32, li.ymax);
        for y in yst..=ynd {
            for x in xst..=xnd {
                pnt.x = (x as f64 + 0.5) as f32;
                pnt.y = (y as f64 + 0.5) as f32;
                if imod_point_inside_cont(cont, &pnt) == 0 {
                    dist = imod_point_cont_distance(cont, &pnt, 0, 0, &mut closest);
                    if dist >= mop.noise_bord_min && dist <= mop.noise_bord_max {
                        slice_get_val(islice, x - li.xmin, y - li.ymin, &mut inval);
                        bord_sum += inval[0] as f64;
                        bord_sum_sq += (inval[0] * inval[0]) as f64;
                        num_bord += 1;
                    }
                }
            }
        }
        if num_bord > 2 {
            sums_to_avg_sd_dbl(
                bord_sum,
                bord_sum_sq,
                num_bord,
                1,
                &mut bord_mean,
                &mut bord_sd,
            );
        }
    }

    let xst = b3dmax!((pmin.x - 1. - more_pad) as i32, li.xmin);
    let xnd = b3dmin!((pmax.x + 2. + more_pad) as i32, li.xmax);
    let yst = b3dmax!((pmin.y - 1. - more_pad) as i32, li.ymin);
    let ynd = b3dmin!((pmax.y + 2. + more_pad) as i32, li.ymax);
    for y in yst..=ynd {
        for x in xst..=xnd {
            pnt.x = (x as f64 + 0.5) as f32;
            pnt.y = (y as f64 + 0.5) as f32;
            let mut do_fill = 0;
            let mut do_bkg = 0;
            taper_frac = 1.;
            if imod_point_inside_cont(cont, &pnt) != 0 {
                if do_all_inside {
                    do_fill = fill;
                    do_bkg = if fill != 0 { 0 } else { 1 };
                } else {
                    dist = imod_point_cont_distance(cont, &pnt, 0, 0, &mut closest);
                    if dist >= abs_pad {
                        do_fill = fill;
                        do_bkg = if fill != 0 { 0 } else { 1 };
                    } else if mop.taper != 0 {
                        taper_frac = dist / abs_pad;
                        do_fill = 1;
                        if fill == 0 {
                            taper_frac = (1. - taper_frac as f64) as f32;
                        }
                    }

                    if mop.ztaper != 0 && fill != 0 {
                        min_zdist = point_dist_from_zborder(
                            obj, &pnt, abs_pad, iz, numatz, contatz, zmin, zmax,
                        );
                        if min_zdist < abs_pad {
                            taper_frac *= min_zdist / abs_pad;
                        }
                    }
                }
            } else if look_outside {
                dist = imod_point_cont_distance(cont, &pnt, 0, 0, &mut closest);
                if dist <= abs_pad {
                    if mop.taper != 0 {
                        taper_frac = dist / abs_pad;
                        if fill != 0 {
                            taper_frac = (1. - taper_frac as f64) as f32;
                        }
                        if taper_frac > 0. {
                            do_fill = 1;
                        } else if fill == 0 {
                            do_bkg = 1;
                        }
                        if mop.ztaper != 0 && do_fill != 0 {
                            min_zdist = point_dist_from_zborder(
                                obj, &pnt, abs_pad, iz, numatz, contatz, zmin, zmax,
                            );
                            if min_zdist < abs_pad {
                                taper_frac *= min_zdist / abs_pad;
                            }
                        }
                    } else {
                        do_fill = fill;
                        do_bkg = if fill != 0 { 0 } else { 1 };
                    }
                }
            }

            /* Get value if filling */
            if do_fill != 0 {
                slice_get_val(islice, x - li.xmin, y - li.ymin, &mut inval);

                // Subtract the noise value because this will be subtracted from image
                // This is not thread-safe
                if mop.noise_bord_max > 0. && bord_sd != 0. {
                    inval[0] -= bord_mean + gaussian_deviate(mop.noise_seed) * bord_sd;
                }
                put_slice_value(mop, obj, pslice, x, y, inval, taper_frac);
            } else if do_bkg != 0 {
                /* Or output the background value */
                if mop.num_chan < 3 {
                    slice_put_val(&mut pslice[0], x - li.xmin, y - li.ymin, mop.bkg_val);
                } else {
                    slice_put_val(&mut pslice[0], x - li.xmin, y - li.ymin, redval);
                    slice_put_val(&mut pslice[1], x - li.xmin, y - li.ymin, grnval);
                    slice_put_val(&mut pslice[2], x - li.xmin, y - li.ymin, bluval);
                }
            }
        }
    }
}

/// Original: `pointDistFromZborder` (`imodmop.c:1110`, static).
///
/// For a point on slice at `iz`, tests whether it is inside contours on
/// adjacent slices out to the padding distance and gets the minimum distance
/// to an end in Z.
#[allow(clippy::too_many_arguments)]
fn point_dist_from_zborder(
    obj: &Iobj,
    pnt: &Ipoint,
    abs_pad: f32,
    iz: i32,
    numatz: &[i32],
    contatz: &[Vec<i32>],
    zmin: i32,
    zmax: i32,
) -> f32 {
    let max_del_z = b3dnint!(abs_pad as f64 + 0.01);
    let mut min_zdist: f32 = abs_pad + 10.;
    let mut dir = -1;
    while dir <= 1 {
        let mut dist_z: f32 = 0.5;
        for del_z in 1..=max_del_z {
            // Out of bounds is the same as being outside any contour
            if iz + dir * del_z < zmin || iz + dir * del_z > zmax {
                break;
            }

            // Count number of contours it is inside
            let mut num_inside = 0;
            let indz = (iz + dir * del_z - zmin) as usize;
            for kis in 0..numatz[indz] as usize {
                let co = contatz[indz][kis] as usize;
                if imod_point_inside_cont(&obj.cont[co], pnt) != 0 {
                    num_inside += 1;
                }
            }

            // Inside 0 contours, or even number of contours, means outside
            if num_inside % 2 == 0 {
                break;
            }

            // Distance at which it was last inside contours
            dist_z += 1.;
        }
        min_zdist = b3dmin!(min_zdist, dist_z);
        dir += 2;
    }
    min_zdist
}

/// Original: `paintScatPoints` (`imodmop.c:1147`, static).
fn paint_scat_points(
    mop: &Mop,
    obj: &Iobj,
    islice: &Islice,
    pslice: &mut [Islice],
    iz: i32,
    allsec: i32,
) {
    let mut inval = [0f32; 4];
    let mut rad: f32;
    let mut dist: f32;
    let mut size: f32;
    let mut taper_frac: f32;
    let mut delta_size: f32 = 0.;
    let abs_pad = (mop.padding as f64).abs() as f32;
    let mut dx: f64;
    let mut dy: f64;
    let mut dz: f64;
    let li = &mop.li;

    /* Change the size by the padding unless there is tapering inside */
    if mop.padding > 0. || mop.taper == 0 {
        delta_size = mop.padding;
    }

    /* Loop on all points in all contours */
    for co in 0..obj.cont.len() {
        let cont = &obj.cont[co];
        for pt in 0..cont.pts.len() {
            let pnt = &cont.pts[pt];

            /* Get size, check if point is within bounds in X */
            size = imod_point_get_size(obj, cont, pt as i32) + delta_size;
            if size <= 0.
                || pnt.x + size < li.xmin as f32
                || pnt.x - size > li.xmax as f32
                || pnt.y + size < li.ymin as f32
                || pnt.y - size > li.ymax as f32
            {
                continue;
            }

            /* Check if Z is within range (3D) or right on (2D), get radius on
            current Z  for 3D */
            if mop.scat3d != 0 {
                dz = ((pnt.z - iz as f32) as f64).abs();
                if dz >= size as f64 {
                    continue;
                }
                rad = ((size * size) as f64 - dz * dz).sqrt() as f32;
            } else {
                if allsec == 0 && b3dnint!(cont.pts[pt].z) != iz {
                    continue;
                }
                rad = size;
            }

            /* Should this limit be here ? */
            if rad < 0.25 {
                continue;
            }

            /* Loop on each line in circle, get X limits. Here, subtract 0.5 from
            Y coordinates since the middle of a pixel is at 0.5, 0.5 */
            let yst = b3dnint!(pnt.y as f64 - 0.5 - rad as f64);
            let ynd = b3dnint!(pnt.y as f64 - 0.5 + rad as f64);
            for y in yst..=ynd {
                dy = (pnt.y as f64 - 0.5 - y as f64).abs();
                if dy > rad as f64 {
                    continue;
                }
                dx = ((rad * rad) as f64 - dy * dy).sqrt();

                /* But subtracting 0.5 from both xst and xnd, then running from xst to
                xnd inclusive made it ~0.5 pixel too wide.  So don't subtract 0.5,
                and run to < xnd instead of <= xnd, and it's pretty good */
                let mut xst = b3dnint!(pnt.x as f64 - dx);
                xst = b3dmax!(xst, li.xmin);
                let mut xnd = b3dnint!(pnt.x as f64 + dx);
                xnd = b3dmin!(xnd, li.xmax);
                if !(mop.padding != 0. && mop.taper != 0) {
                    for x in xst..xnd {
                        slice_get_val(islice, x - li.xmin, y - li.ymin, &mut inval);
                        put_slice_value(mop, obj, pslice, x, y, inval, 1.);
                    }
                } else {
                    /* For tapering, loop on the points and measure their distances individually
                    then evaluate a taper fraction */
                    for x in xst..xnd {
                        dx = x as f64 + 0.5 - pnt.x as f64;
                        dy = y as f64 + 0.5 - pnt.y as f64;
                        dz = if mop.scat3d != 0 {
                            (iz as f32 - pnt.z) as f64
                        } else {
                            0.
                        };
                        dist = (dx * dx + dy * dy + dz * dz).sqrt() as f32;
                        taper_frac = 1.;
                        if dist > size - abs_pad {
                            taper_frac = (size - dist) / abs_pad;
                        }
                        if taper_frac > 0. {
                            slice_get_val(islice, x - li.xmin, y - li.ymin, &mut inval);
                            put_slice_value(mop, obj, pslice, x, y, inval, taper_frac);
                        }
                    }
                }
            }
        }
    }
}

/// Original: `paintTubularLines` (`imodmop.c:1236`, static).
fn paint_tubular_lines(
    mop: &Mop,
    obj: &Iobj,
    islice: &Islice,
    pslice: &mut [Islice],
    mut rad: f32,
    iz: i32,
) {
    let mut inval = [0f32; 4];
    let mut pix = Ipoint::default();
    let mut closest = 0;
    let mut tval: f32 = 0.;
    let mut taper_frac: f32;
    let mut dz: f64;
    let mut dzlas: f64;
    let abs_pad = (mop.padding as f64).abs() as f32;
    let li = &mop.li;

    if mop.padding > 0. || mop.taper == 0 {
        rad += mop.padding;
    }

    /* Loop on all line segments in all contours */
    let radsq = rad * rad;
    for co in 0..obj.cont.len() {
        let cont = &obj.cont[co];
        let psize = cont.pts.len() as i32;
        if psize < 1 || (mop.tubes2d == 0 && psize < 2) {
            continue;
        }
        let mut ptlas = &cont.pts[0];
        dzlas = (ptlas.z as f64 - iz as f64).abs();
        for pt in b3dmin!(1, psize - 1)..psize {
            let ptcur = &cont.pts[pt as usize];
            dz = (ptcur.z as f64 - iz as f64).abs();
            if (mop.tubes2d == 0
                && (dz <= rad as f64
                    || dzlas <= rad as f64
                    || (ptcur.z > iz as f32 && ptlas.z < iz as f32)
                    || (ptcur.z < iz as f32 && ptlas.z > iz as f32)))
                || (mop.tubes2d != 0 && dz < 0.5 && dzlas < 0.5)
            {
                /* If either endpoint is within radius in Z or the segment crosses the Z-plane,
                make a box around the segment */
                let radinc: f32 = (rad as f64 + 2.) as f32;
                let mut xst = b3dmin!(ptcur.x - radinc, (ptlas.x as i32) as f32 - radinc) as i32;
                let mut xnd = b3dmax!(ptcur.x + radinc, (ptlas.x as i32) as f32 + radinc) as i32;
                let mut yst = b3dmin!(ptcur.y - radinc, (ptlas.y as i32) as f32 - radinc) as i32;
                let mut ynd = b3dmax!(ptcur.y + radinc, (ptlas.y as i32) as f32 + radinc) as i32;
                xst = b3dmax!(xst, li.xmin);
                xnd = b3dmin!(xnd, li.xmax);
                yst = b3dmax!(yst, li.ymin);
                ynd = b3dmin!(ynd, li.ymax);
                for y in yst..=ynd {
                    for x in xst..=xnd {
                        let mut dist: f32;
                        pix.x = x as f32 + 0.5f32;
                        pix.y = y as f32 + 0.5f32;
                        pix.z = iz as f32;
                        dist = imod_point_line_seg_distance(ptlas, ptcur, &pix, &mut tval);
                        if dist <= radsq {
                            taper_frac = 1.;

                            /* Get a global value for distance from the whole contour so that the same
                            point in multiple segments will give the same tapered value */
                            if mop.padding != 0. && mop.taper != 0 {
                                dist = imod_point_cont_distance(cont, &pix, 1, 1, &mut closest);
                                if dist > rad - abs_pad {
                                    taper_frac = (rad - dist) / abs_pad;
                                }
                            }
                            if taper_frac > 0. {
                                slice_get_val(islice, x - li.xmin, y - li.ymin, &mut inval);
                                put_slice_value(mop, obj, pslice, x, y, inval, taper_frac);
                            }
                        }
                    }
                }
            }
            dzlas = dz;
            ptlas = ptcur;
        }
    }
}

/// Original: `putSliceValue` (`imodmop.c:1316`, static).
///
/// The source's `islice` parameter is unused; it is dropped here.
fn put_slice_value(
    mop: &Mop,
    obj: &Iobj,
    pslice: &mut [Islice],
    x: i32,
    y: i32,
    inval: [f32; 4],
    mut taper_frac: f32,
) {
    let mut outval = [0f32; 4];
    let mut useval = [0f32; 4];
    let li = &mop.li;

    /* In simple case, just put out the value( */
    if mop.num_chan < 3 && mop.masking == 0 && taper_frac as f64 > 0.99999 {
        slice_put_val(&mut pslice[0], x - li.xmin, y - li.ymin, inval);
        return;
    }

    /* Otherwise copy the value or turn it into mask */
    if mop.masking != 0 {
        useval[0] = mop.mask_val[0];
        useval[1] = mop.mask_val[1];
        useval[2] = mop.mask_val[2];
    } else {
        useval[0] = inval[0];
        useval[1] = inval[1];
        useval[2] = inval[2];
    }

    if taper_frac as f64 <= 0.9999 {
        if mop.taper > 1 {
            taper_frac = mop.window_table[(TABLE_SIZE as f32 * taper_frac) as i32 as usize];
        }
        // `taperFrac * useval[i]` is a float product; `(1. - taperFrac)` is
        // double, so the sum is done in double.
        for i in 0..3 {
            useval[i] = ((taper_frac * useval[i]) as f64
                + (1. - taper_frac as f64) * mop.bkg_val[i] as f64) as f32;
        }
    }

    if mop.num_chan < 3 {
        slice_put_val(&mut pslice[0], x - li.xmin, y - li.ymin, useval);
    } else {
        /* If doing 3 channels, only first element is needed */
        outval[0] = obj.red * useval[0];
        slice_put_val(&mut pslice[0], x - li.xmin, y - li.ymin, outval);
        outval[0] = obj.green * useval[0];
        slice_put_val(&mut pslice[1], x - li.xmin, y - li.ymin, outval);
        outval[0] = obj.blue * useval[0];
        slice_put_val(&mut pslice[2], x - li.xmin, y - li.ymin, outval);
    }
}

/// Original: `scaleAndCombineSlices` (`imodmop.c:1363`, static).
fn scale_and_combine_slices(
    pslice: &[Islice],
    oslice: &mut Islice,
    smin: f32,
    smax: f32,
    num_files: i32,
) {
    let mut inval = [0f32; 4];
    let mut outval = [0f32; 4];
    for y in 0..oslice.ysize {
        for x in 0..oslice.xsize {
            for i in 0..num_files as usize {
                slice_get_val(&pslice[i], x, y, &mut inval);
                outval[i] = (255. * (inval[0] - smin) as f64 / (smax - smin) as f64) as f32;
                // `B3DMAX(0., B3DMIN(255., outval[i]))`, compared in double
                let m: f64 = if 255. < outval[i] as f64 {
                    255.
                } else {
                    outval[i] as f64
                };
                outval[i] = (if 0. > m { 0. } else { m }) as f32;
            }
            slice_put_val(oslice, x, y, outval);
        }
    }
}

/// Original: `itemOnList` (`imodmop.c:1380`, static).
fn item_on_list(item: i32, list: Option<&[i32]>, num: i32) -> i32 {
    if num == 0 {
        return -1;
    }
    let list = list.unwrap_or(&[]);
    for i in 0..num as usize {
        if list[i] == item {
            return i as i32;
        }
    }
    -2
}
