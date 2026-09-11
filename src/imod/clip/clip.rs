//! Scaffold for `IMOD/clip/clip.cpp` and its paired `IMOD/clip/clip.h`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::imod_prog_name;
use crate::imod::libcfshr::parse_params::exit_error;
use crate::imod::libcfshr::parse_params::setExitPrefix;
use crate::imod::libiimod::mrcfiles::MrcHeader;

unsafe extern "C" {
    static mut stdout: *mut libc::FILE;
}

pub const IP_NONE: i32 = 0;
pub const IP_DEFAULT: i32 = -99_999;
/// `mrcslice.h:29-31`.
pub const SLICE_MODE_UNDEFINED: i32 = -1;
pub const SLICE_MODE_SBYTE: i32 = -2;
pub const SLICE_MODE_UBYTE: i32 = -3;
pub const IP_APPEND_FALSE: i32 = 0;
pub const IP_APPEND_OVERWRITE: i32 = 1;
pub const IP_APPEND_ADD: i32 = 2;
pub const IP_APPEND_TRUNCATE: i32 = 3;
pub const IP_BOXSD: i32 = 54;
// C `clip.h` process enum: explicit values keep Rust dispatch identical to C++.
pub const IP_ADD: i32 = 1;
pub const IP_AVERAGE: i32 = 2;
pub const IP_VARIANCE: i32 = 3;
pub const IP_STANDEV: i32 = 4;
pub const IP_BRIGHTNESS: i32 = 5;
pub const IP_COLOR: i32 = 6;
pub const IP_CONTRAST: i32 = 7;
pub const IP_CORRELATE: i32 = 8;
pub const IP_DIFFUSION: i32 = 9;
pub const IP_FFT: i32 = 10;
pub const IP_FILTER: i32 = 11;
pub const IP_FLIP: i32 = 12;
pub const IP_GRADIENT: i32 = 13;
pub const IP_SUBTRACT: i32 = 14;
pub const IP_MULTIPLY: i32 = 15;
pub const IP_DIVIDE: i32 = 16;
pub const IP_LOGARITHM: i32 = 17;
pub const IP_SQROOT: i32 = 18;
pub const IP_GRAHAM: i32 = 19;
pub const IP_INFO: i32 = 20;
pub const IP_JOINRGB: i32 = 21;
pub const IP_LAPLACIAN: i32 = 22;
pub const IP_MEDIAN: i32 = 23;
pub const IP_PEAK: i32 = 24;
pub const IP_PREWITT: i32 = 25;
pub const IP_UNWRAP: i32 = 26;
pub const IP_QUADRANT: i32 = 27;
pub const IP_UNPACK: i32 = 28;
pub const IP_HISTOGRAM: i32 = 29;
pub const IP_NORMALIZE: i32 = 30;
pub const IP_PROJECT: i32 = 31;
pub const IP_RESIZE: i32 = 32;
pub const IP_ROTATE: i32 = 33;
pub const IP_SHADOW: i32 = 34;
pub const IP_SHARPEN: i32 = 35;
pub const IP_SMOOTH: i32 = 36;
pub const IP_SPECTRUM: i32 = 37;
pub const IP_SOBEL: i32 = 38;
pub const IP_SPLITRGB: i32 = 39;
pub const IP_STAT: i32 = 40;
pub const IP_TRANSLATE: i32 = 41;
pub const IP_ZOOM: i32 = 42;
pub const IP_TRUNCATE: i32 = 43;
pub const IP_THRESHOLD: i32 = 44;
pub const IP_FILLEDGE: i32 = 45;
pub const IP_DEFECTMAP: i32 = 46;
pub const IP_SUPERGAIN: i32 = 47;
pub const IP_PLANARFIT: i32 = 48;
pub const IP_FLATFIELD: i32 = 49;
pub const IP_BLANKFILE: i32 = 50;
pub const IP_INTEGRAL: i32 = 51;

/// C `Grap_options` / `ClipOptions` (`clip.h`).  Pointer targets enter scope
/// with the corresponding IMOD headers; the field sequence is source order.
pub struct ClipOptions {
    pub pname: *mut core::ffi::c_char,
    pub command: *mut core::ffi::c_char,
    pub hin: *mut MrcHeader,
    pub hin2: *mut MrcHeader,
    pub hout: *mut MrcHeader,
    pub process: i32,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub x2: i32,
    pub y2: i32,
    pub z2: i32,
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
    pub iz2: i32,
    pub ox: i32,
    pub oy: i32,
    pub oz: i32,
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    pub cx: f32,
    pub cy: f32,
    pub cz: f32,
    pub high: f32,
    pub low: f32,
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub thresh: f32,
    pub weight: f32,
    pub pctl_frac: f32,
    pub falloff_frac: f32,
    pub min_size: i32,
    pub pad: f32,
    pub mode: i32,
    pub dim: i32,
    pub infiles: i32,
    pub fnames: *mut *mut core::ffi::c_char,
    pub sano: i32,
    pub add2file: i32,
    pub isec: i32,
    pub val: f32,
    pub nofsecs: i32,
    pub secs: *mut i32,
    pub out_before: i32,
    pub out_after: i32,
    pub ocanresize: i32,
    pub ocanchmode: i32,
    pub from_one: i32,
    pub ofname: *mut core::ffi::c_char,
    pub plname: *mut core::ffi::c_char,
    pub super_gain_name: *mut core::ffi::c_char,
    pub point_out_name: *mut core::ffi::c_char,
    pub new_xoverlap: i32,
    pub new_yoverlap: i32,
    pub rotation_flip: i32,
    pub read_defects: i32,
    pub defects: CameraDefects,
    pub cam_size_x: i32,
    pub cam_size_y: i32,
    pub binning: f32,
    pub scale_defects: i32,
}

/// `include/CorrectDefects.h` layout, re-exported from its merged source module.
pub use crate::imod::clip::correct_defects::CameraDefects;

/// Original: `PlaneConnectedPoints` (`threshminsize.cpp:21`).
pub struct PlaneConnectedPoints {
    pub points: std::collections::BTreeSet<u32>,
    pub xmin: i32,
    pub ymin: i32,
    pub xmax: i32,
    pub ymax: i32,
}
/// Original: `ZConnectedSets` (`threshminsize.cpp:27`).
pub struct ZConnectedSets {
    pub plane_z: Vec<i32>,
    pub index: Vec<i32>,
    pub xmin: i32,
    pub ymin: i32,
    pub xmax: i32,
    pub ymax: i32,
    pub zmin: i32,
    pub zmax: i32,
    pub num_points: i32,
}
/// Original: `zstats` (`processing.cpp:3453`).
pub struct Zstats {
    pub x: f32,
    pub y: f32,
    pub mean: f64,
    pub std: f64,
    pub xmin: i32,
    pub ymin: i32,
    pub outlier: i32,
}

/// Original: `usage` (`clip.cpp:19`).
///
/// The complete usage text is intentionally kept with the binary entry point,
/// as in the source this routine writes directly to standard output.
pub fn usage() {
    // The source writes every line of this routine with C `printf`.  Keep the
    // same stream so the interleaving with `imodCopyright`, which also uses
    // `printf`, matches the reference when standard output is redirected and
    // therefore fully buffered.
    unsafe {
        libc::printf(
            c"%s: Command Line Image Processing. %s, %s %s\n".as_ptr(),
            c"clip".as_ptr(),
            c"5.2.17".as_ptr(),
            crate::imod::libcfshr::b3dutil::IMOD_BUILD_DATE.as_ptr(),
            crate::imod::libcfshr::b3dutil::IMOD_BUILD_TIME.as_ptr(),
        );
    }
    crate::imod::libcfshr::b3dutil::imod_copyright();
    unsafe {
        libc::printf(
            c"----------------------------------------------------\n\
clip usage:\n\
clip [process] [options] [input files...] [output file]\n\
\n\
process:\n\
\tadd         - Add images together.\n\
\taverage     - Average files together.\n\
\tboxsd       - Compute SD or SD/mean in local boxes around each pixel.\n\
\tblankfile   - Make constant volume (enter -ox, -oy, -oz, -m, -p).\n\
\tbrightness  - Increase or decrease brightness.\n\
\tcolor       - Add false color.\n\
\tcontrast    - Increase or decrease contrast.\n\
\tcorrelation - Do a auto/cross correlation.\n\
\tdefectmap   - Output map of all pixels in defects entered with -B.\n\
\tdiffusion   - Do 2-D anisotropic diffusion on slices.\n\
\tdivide      - Divide one image volume by another.\n\
\tflatfield   - MAKE a flatfielding image from sum of all slices.\n\
\tgradient    - Compute gradient as in 3dmod.\n\
\tgraham      - Apply Graham filter as in 3dmod.\n\
\thistogram   - Print histogram of values.\n\
\tinfo        - Print information to stdout.\n\
\tintegral    - Compute local integral at points beyond threshold .\n\
\tfft         - Do a fft or inverse fft transform.\n\
\tfilter      - Do a bandpass filter.\n\
\tflip..      - Flip image about various axes.\n\
\tjoinrgb     - Join 3 byte files into an RGB file.\n\
\tlaplacian   - Apply Laplacian filter as in 3dmod.\n\
\tmedian      - Do median filtering.\n\
\tmultiply    - Multiple one image volume by another.\n\
\tnormalize   - Multiply by gain ref., scale, remove big values.\n\
\tplanefit    - Fit plane to sum of all slices to get gradient.\n\
\tprewitt     - Apply Prewitt filter as in 3dmod.\n\
\tquadrant    - Correct quadrant disparities from 4-port camera.\n\
\tresize      - Cut out and/or pad image data.\n\
\trotx        - Rotate volume by -90 about X axis.\n\
\tshadow      - Increase or decrease image shadows.\n\
\tsharpen     - Sharpen image as in 3dmod.\n\
\tsmooth      - Smooth image as in 3dmod.\n\
\tsobel       - Apply Sobel filter as in 3dmod.\n\
\tspectrum    - Compute scaled, reduced power spectrum.\n\
\tsplitrgb    - Split RGB image file into 3 byte files.\n\
\tstandev     - Compute standard deviation for averaged images.\n\
\tstats       - Print some stats on image file.\n\
\tsubtract    - Subtract one image volume from another.\n\
\tsupergain   - Analyze EER file(s) .\n\
\tthreshold   - Apply threshold to make binary image.\n\
\ttruncate    - Limit image values at low or high end.\n\
\tunwrap      - Undo a wraparound of integer intensity values.\n\
\tunpack      - Unpack 4-bit data into bytes - same as normalize.\n\
\tvariance    - Compute variance for averaged images.\n\
\n\
options:\n\
\t[-v]  view output data.\n\
\t[-3d] or [-2d] treat image as 3d (default) or 2d.\n\
\t[-s] Switch, [-n #] Amount; Depends on function.\n\
\t[-n #] [-l #] Iterations and Gaussian sigma for smoothing.\n\
\t[-h #] [-l #] [-t #] Values for filter, threshold, or truncate.\n\
\t[-op file] Name of file for output of points above threshold.\n\
\t[-cc #] [-l #] [-k #] values for anisotropic diffusion.\n\
\t[-r #] [-g #] [-b #] red, green, blue values.\n\
\t[-x #,#]  [-y #,#]  starting and ending input coords.\n\
\t[-cx #]  [-cy #]  [-cz #]  center coords.\n\
\t[-ix #]  [-iy #]  [-iz #]  input sizes.\n\
\t[-ox #]  [-oy #]  [-oz #]  output sizes.\n\
\t[-CX #]  [-CY #]  [-CZ #]  chunk sizes for tiled HDF output.\n\
\t[-a] Append output to file.\n\
\t[-ov #] Overwrite output starting at section # and truncate file\n\
\t[-or #] Overwrite output starting at section # and retain to end\n\
\t[-m (mode#)] Output data mode.\n\
\t[-f format] Output file format (MRC, TIFF, HDF, JPEG).\n\
\t[-p (pad#)] Pad empty data value.\n\
\t[-1] Number Z values from 1 instead of 0.\n\
\t[-P file] Name of piece list file for stats on a montage.\n\
\t[-O #,#]  Overlaps in X and Y in displayed montage.\n\
\t[-D file] Apply defect correction using defect list in given file.\n\
\t[-B #] Binning value to use in defect correction.\n\
\t[-S]   Scale defect list up by 2 if it is not already scaled.\n\
\t[-R #] Rotation/flip to apply to gain reference, or -1 for r/f.\n\
\t[-E #,#] Analyze histogram for extra counts on one side.\n\
\t[-F #,#] Analyze histogram for fastest falloff point.\n\
\t[-M #,#] Set minimum size of connected regions when thresholding.\n\
\t[-es #] Set super-resolution factor when reading from EER files.\n\
\t[-ez #] Set summing of frames when reading from EER files.\n\
\t[-et]   Read thumbnail, not frames, from EER file.\n\
\t[-ep #] Set padding of defects from gain reference for EER file.\n\
\t[-ed file] Write defects from EER gain reference to file.\n\
\t[-eg file] File for adjusting super-resolution gain reference.\n\n"
                .as_ptr(),
        );
    }
}
/// Original: `show_error` (`clip.cpp:124`).
pub fn show_error(message: &str) {
    // `clip.cpp:124-131` writes to standard output with C `printf`.  Keeping
    // the same stream matters when output is redirected, because the C-derived
    // modules on this path are block-buffered by libc.
    unsafe {
        let text = std::ffi::CString::new(message).unwrap();
        libc::printf(c"ERROR: %s\n".as_ptr(), text.as_ptr());
    }
}
/// Original: `show_warning` (`clip.cpp:133`).
pub fn show_warning(reason: &str) {
    unsafe {
        let text = std::ffi::CString::new(reason).unwrap();
        libc::printf(c"WARNING: %s\n".as_ptr(), text.as_ptr());
    }
}
/// Original: `show_status` (`clip.cpp:138`).
pub fn show_status(info: &str) {
    unsafe {
        let text = std::ffi::CString::new(info).unwrap();
        libc::printf(c"%s".as_ptr(), text.as_ptr());
        libc::fflush(stdout);
    }
}
/// C++ `default_options` (`clip.cpp:144`).
pub fn default_options(options: &mut ClipOptions) {
    options.hin = core::ptr::null_mut();
    options.hin2 = core::ptr::null_mut();
    options.hout = core::ptr::null_mut();
    options.x = IP_DEFAULT;
    options.x2 = IP_DEFAULT;
    options.y = IP_DEFAULT;
    options.y2 = IP_DEFAULT;
    options.z = IP_DEFAULT;
    options.z2 = IP_DEFAULT;
    options.ix = IP_DEFAULT;
    options.iy = IP_DEFAULT;
    options.iz = IP_DEFAULT;
    options.iz2 = IP_DEFAULT;
    options.ox = IP_DEFAULT;
    options.oy = IP_DEFAULT;
    options.oz = IP_DEFAULT;
    options.cx = IP_DEFAULT as f32;
    options.cy = IP_DEFAULT as f32;
    options.cz = IP_DEFAULT as f32;
    options.chunk_x = IP_DEFAULT;
    options.chunk_y = IP_DEFAULT;
    options.chunk_z = IP_DEFAULT;
    options.out_before = IP_DEFAULT;
    options.out_after = IP_DEFAULT;
    options.red = IP_DEFAULT as f32;
    options.green = IP_DEFAULT as f32;
    options.blue = IP_DEFAULT as f32;
    options.high = IP_DEFAULT as f32;
    options.low = IP_DEFAULT as f32;
    options.thresh = IP_DEFAULT as f32;
    options.pctl_frac = IP_DEFAULT as f32;
    options.falloff_frac = IP_DEFAULT as f32;
    options.weight = IP_DEFAULT as f32;
    options.pad = IP_DEFAULT as f32;
    options.min_size = IP_DEFAULT;
    options.process = IP_NONE;
    options.dim = 3;
    options.add2file = IP_APPEND_FALSE;
    options.sano = 0;
    options.val = IP_DEFAULT as f32;
    options.mode = IP_DEFAULT;
    options.nofsecs = IP_DEFAULT;
    options.secs = core::ptr::null_mut();
    options.ocanresize = 1;
    options.ocanchmode = 1;
    options.from_one = 0;
    options.ofname = core::ptr::null_mut();
    options.plname = core::ptr::null_mut();
    options.super_gain_name = core::ptr::null_mut();
    options.point_out_name = core::ptr::null_mut();
    options.new_xoverlap = IP_DEFAULT;
    options.new_yoverlap = IP_DEFAULT;
    options.read_defects = 0;
    options.rotation_flip = 0;
    options.binning = IP_DEFAULT as f32;
    options.scale_defects = 0;
}
/// Original: `main` (`clip.cpp:189`).
pub fn clip() {
    // `clip.cpp:main`: command-line lifetime owns the C strings and the option vectors
    // until all source-style C-ABI image calls have completed.
    unsafe {
        use crate::imod::clip::{correlation, fft, filter, processing};
        use crate::imod::libiimod::{iimage, mrcfiles};
        let raw: Vec<String> = std::env::args().collect();
        let program = std::ffi::CString::new(raw[0].as_bytes()).unwrap();
        let prefix = std::ffi::CString::new(format!(
            "ERROR: {} - ",
            core::ffi::CStr::from_ptr(imod_prog_name(program.as_ptr())).to_string_lossy()
        ))
        .unwrap();
        setExitPrefix(prefix.as_ptr());
        if raw.len() < 3 {
            usage();
            std::process::exit(3);
        }
        let command = &raw[1];
        let command_string = std::ffi::CString::new(command.as_bytes()).unwrap();
        let process = if command.starts_with("add") {
            IP_ADD
        // clip.cpp accepts either the `avg` abbreviation or the first three
        // letters of `average` ("ave").
        } else if command.starts_with("avg") || command.starts_with("ave") {
            IP_AVERAGE
        } else if command.starts_with("stan") {
            IP_STANDEV
        } else if command.starts_with("var") {
            IP_VARIANCE
        } else if command.starts_with("mul") {
            IP_MULTIPLY
        } else if command.starts_with("sub") {
            IP_SUBTRACT
        } else if command.starts_with("div") {
            IP_DIVIDE
        } else if command.starts_with("bri") {
            IP_BRIGHTNESS
        } else if command.starts_with("col") {
            IP_COLOR
        } else if command.starts_with("con") {
            IP_CONTRAST
        } else if command.starts_with("cor") {
            IP_CORRELATE
        } else if command.starts_with("dif") {
            IP_DIFFUSION
        } else if command.starts_with("fft") {
            IP_FFT
        } else if command.starts_with("fil") {
            IP_FILTER
        } else if command.starts_with("fla") {
            IP_FLATFIELD
        } else if command.starts_with("flip") || command.starts_with("rotx") {
            IP_FLIP
        } else if command.starts_with("grad") {
            IP_GRADIENT
        } else if command.starts_with("grah") {
            IP_GRAHAM
        } else if command.starts_with("hi") {
            IP_HISTOGRAM
        } else if command.starts_with("la") {
            IP_LAPLACIAN
        } else if command.starts_with("me") {
            IP_MEDIAN
        } else if command.starts_with("pl") {
            IP_PLANARFIT
        } else if command.starts_with("pr") {
            IP_PREWITT
        } else if command.starts_with("res") {
            IP_RESIZE
        } else if command.starts_with("shad") {
            IP_SHADOW
        } else if command.starts_with("shar") {
            IP_SHARPEN
        } else if command.starts_with("sm") {
            IP_SMOOTH
        } else if command.starts_with("so") {
            IP_SOBEL
        } else if command.starts_with("spl") {
            // In clip.cpp the later splitrgb check overrides spectrum's `sp` prefix.
            IP_SPLITRGB
        } else if command.starts_with("sp") {
            IP_SPECTRUM
        } else if command.starts_with("stat") {
            IP_STAT
        } else if command.starts_with("thr") {
            IP_THRESHOLD
        } else if command.starts_with("tru") {
            IP_TRUNCATE
        } else if command.starts_with("unw") {
            IP_UNWRAP
        } else if command.starts_with("sqr") {
            IP_SQROOT
        } else if command.starts_with("log") {
            IP_LOGARITHM
        } else if command.starts_with("qu") {
            IP_QUADRANT
        } else if command.starts_with("ed") {
            IP_FILLEDGE
        } else if command.starts_with("unp") {
            IP_UNPACK
        } else if command.starts_with("nor") {
            IP_NORMALIZE
        } else if command.starts_with("def") {
            IP_DEFECTMAP
        } else if command.starts_with("sup") {
            IP_SUPERGAIN
        } else if command.starts_with("int") {
            IP_INTEGRAL
        } else if command.starts_with("box") {
            IP_BOXSD
        } else if command.starts_with("bla") {
            IP_BLANKFILE
        } else if command.starts_with("joi") {
            IP_JOINRGB
        } else if command.starts_with("inf") {
            IP_INFO
        } else {
            usage();
            std::process::exit(1);
        };
        let defects = CameraDefects {
            was_scaled: 0,
            rotation_flip: 0,
            k2_type: 0,
            falcon_type: 0,
            usable_top: 0,
            usable_left: 0,
            usable_bottom: 0,
            usable_right: 0,
            num_avg_super_res: 0,
            bad_column_start: vec![],
            bad_column_width: vec![],
            partial_bad_col: vec![],
            partial_bad_width: vec![],
            partial_bad_start_y: vec![],
            partial_bad_end_y: vec![],
            bad_row_start: vec![],
            bad_row_height: vec![],
            partial_bad_row: vec![],
            partial_bad_height: vec![],
            partial_bad_start_x: vec![],
            partial_bad_end_x: vec![],
            bad_pixel_x: vec![],
            bad_pixel_y: vec![],
            pix_use_mean: vec![],
        };
        let mut options = ClipOptions {
            pname: core::ptr::null_mut(),
            command: core::ptr::null_mut(),
            hin: core::ptr::null_mut(),
            hin2: core::ptr::null_mut(),
            hout: core::ptr::null_mut(),
            process,
            x: IP_DEFAULT,
            y: IP_DEFAULT,
            z: IP_DEFAULT,
            x2: IP_DEFAULT,
            y2: IP_DEFAULT,
            z2: IP_DEFAULT,
            ix: IP_DEFAULT,
            iy: IP_DEFAULT,
            iz: IP_DEFAULT,
            iz2: IP_DEFAULT,
            ox: IP_DEFAULT,
            oy: IP_DEFAULT,
            oz: IP_DEFAULT,
            chunk_x: IP_DEFAULT,
            chunk_y: IP_DEFAULT,
            chunk_z: IP_DEFAULT,
            cx: IP_DEFAULT as f32,
            cy: IP_DEFAULT as f32,
            cz: IP_DEFAULT as f32,
            high: IP_DEFAULT as f32,
            low: IP_DEFAULT as f32,
            red: IP_DEFAULT as f32,
            green: IP_DEFAULT as f32,
            blue: IP_DEFAULT as f32,
            thresh: IP_DEFAULT as f32,
            weight: IP_DEFAULT as f32,
            pctl_frac: IP_DEFAULT as f32,
            falloff_frac: IP_DEFAULT as f32,
            min_size: IP_DEFAULT,
            pad: IP_DEFAULT as f32,
            mode: IP_DEFAULT,
            dim: if process == IP_QUADRANT { 2 } else { 3 },
            infiles: 0,
            fnames: core::ptr::null_mut(),
            sano: 0,
            add2file: IP_APPEND_FALSE,
            isec: 0,
            val: IP_DEFAULT as f32,
            nofsecs: IP_DEFAULT,
            secs: core::ptr::null_mut(),
            out_before: IP_DEFAULT,
            out_after: IP_DEFAULT,
            ocanresize: 1,
            ocanchmode: 1,
            from_one: 0,
            ofname: core::ptr::null_mut(),
            plname: core::ptr::null_mut(),
            super_gain_name: core::ptr::null_mut(),
            point_out_name: core::ptr::null_mut(),
            new_xoverlap: IP_DEFAULT,
            new_yoverlap: IP_DEFAULT,
            rotation_flip: 0,
            read_defects: 0,
            defects,
            cam_size_x: 0,
            cam_size_y: 0,
            binning: IP_DEFAULT as f32,
            scale_defects: 0,
        };
        // Keep the source initialization as an explicit mapped call.  The literal above
        // supplies safe Rust ownership values for fields that the C initializer leaves
        // untouched; this function then establishes every C-owned default.
        default_options(&mut options);
        options.process = process;
        if process == IP_QUADRANT {
            options.dim = 2;
        }
        if process == IP_FLATFIELD {
            options.ocanresize = 0;
            options.mode = crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT;
        }
        if process == IP_FILLEDGE {
            options.dim = 2;
        }
        let mut position = 2;
        let mut sections: Vec<i32> = Vec::new();
        let mut eer_super = 2_i32;
        let mut eer_group = 1_i32;
        let mut eer_flags = 0_i32;
        let mut fei_def_pad = 1_i32;
        let mut fei_pad_entered = 0_i32;
        let mut dump_defect_name: Option<std::ffi::CString> = None;
        let mut format_set = -2_i32;
        let mut view = false;
        // `clip.cpp:199` declares `int itemp;` without an initializer and reuses
        // it for -E, -F and -M; `sscanf` leaves it untouched when the second
        // conversion fails, so its value carries across option entries.  The
        // uninitialized first read is source-level indeterminate; zero is used.
        let mut itemp = 0_i32;
        iimage::get_dflt_eersumming_from_env(&mut eer_super, &mut eer_group);
        options.pname = imod_prog_name(program.as_ptr());
        if process == IP_SUPERGAIN {
            eer_group = 250;
            eer_super = 2;
            options.val = 4.;
        }
        // `clip.cpp:366-618`: the option loop switches on the second character of
        // the argument and, for several letters, on the third.
        while position < raw.len() {
            let flag = raw[position].clone();
            let bytes = flag.as_bytes();
            if bytes.first() != Some(&b'-') {
                break;
            }
            let second = *bytes.get(2).unwrap_or(&0);
            // `clip.cpp:391` inspects argv[iarg + 1] before advancing.
            let next_arg = raw.get(position + 1).cloned().unwrap_or_default();
            let mut need = || {
                position += 1;
                raw.get(position).cloned().unwrap_or_default()
            };
            match *bytes.get(1).unwrap_or(&0) {
                b'a' => options.add2file = IP_APPEND_ADD,
                b'3' => {
                    if process != IP_QUADRANT {
                        options.dim = 3;
                    }
                }
                b'2' => options.dim = 2,
                b'1' => options.from_one = 1,
                b's' => options.sano = 1,
                b'n' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &raw mut options.val);
                }
                b'm' => {
                    if next_arg == "4-bit" || next_arg == "101" {
                        crate::imod::libcfshr::b3dutil::set_4_bit_output_mode(1);
                        options.mode = mrcfiles::MRC_MODE_BYTE;
                        need();
                    } else if next_arg == "float16" || next_arg == "12" {
                        options.mode = mrcfiles::MRC_MODE_HALF_FLOAT;
                        need();
                    } else {
                        let text = std::ffi::CString::new(need()).unwrap();
                        options.mode = crate::imod::libcfshr::islice::slice_mode(text.as_ptr());
                    }
                    if options.mode == SLICE_MODE_UNDEFINED {
                        let message =
                            std::ffi::CString::new(format!("Invalid mode entry {next_arg}."))
                                .unwrap();
                        exit_error(message.as_ptr());
                    }
                    if options.mode == SLICE_MODE_SBYTE || options.mode == SLICE_MODE_UBYTE {
                        crate::imod::libcfshr::b3dutil::override_write_bytes(
                            if options.mode == SLICE_MODE_SBYTE {
                                1
                            } else {
                                0
                            },
                        );
                        options.mode = 0;
                    }
                    if options.process == IP_FLATFIELD && options.mode != mrcfiles::MRC_MODE_FLOAT {
                        show_warning("Output mode for a flatfield image must be floating point");
                        options.mode = 2;
                    }
                    if options.mode == mrcfiles::MRC_MODE_FLOAT {
                        crate::imod::libcfshr::b3dutil::set_float_16_output_mode(0, 0);
                    }
                }
                b'f' => {
                    let value = need();
                    format_set =
                        crate::imod::libcfshr::b3dutil::set_output_type_from_string(&value);
                    if format_set < 0 {
                        let message = std::ffi::CString::new(format!(
                            "Output file format entry {} is not {}.",
                            value,
                            if format_set == -1 {
                                "recognized"
                            } else {
                                "available in this copy of IMOD"
                            }
                        ))
                        .unwrap();
                        exit_error(message.as_ptr());
                    }
                }
                b'v' => view = true,
                b'p' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    options.pad = libc::atof(value.as_ptr()) as f32;
                }
                b't' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &raw mut options.thresh);
                }
                b'E' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(
                        value.as_ptr(),
                        c"%f%*c%d".as_ptr(),
                        &raw mut options.pctl_frac,
                        &raw mut itemp,
                    );
                    if itemp < 0 {
                        options.pctl_frac = -options.pctl_frac;
                    }
                }
                b'F' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(
                        value.as_ptr(),
                        c"%f%*c%d".as_ptr(),
                        &raw mut options.falloff_frac,
                        &raw mut itemp,
                    );
                    if itemp < 0 {
                        options.falloff_frac = -options.falloff_frac;
                    }
                }
                b'M' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(
                        value.as_ptr(),
                        c"%d%*c%d".as_ptr(),
                        &raw mut options.min_size,
                        &raw mut itemp,
                    );
                    if itemp < 0 {
                        options.min_size = -options.min_size;
                    }
                }
                b'k' | b'w' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &raw mut options.weight);
                }
                b'r' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &raw mut options.red);
                }
                b'g' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &raw mut options.green);
                }
                b'b' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &raw mut options.blue);
                }
                b'l' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &raw mut options.low);
                }
                b'h' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &raw mut options.high);
                }
                b'x' | b'X' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(
                        value.as_ptr(),
                        c"%d%*c%d".as_ptr(),
                        &raw mut options.x,
                        &raw mut options.x2,
                    );
                }
                b'y' | b'Y' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(
                        value.as_ptr(),
                        c"%d%*c%d".as_ptr(),
                        &raw mut options.y,
                        &raw mut options.y2,
                    );
                }
                b'o' => match second {
                    0x00 | b' ' | b'v' => {
                        options.add2file = IP_APPEND_TRUNCATE;
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.isec);
                    }
                    b'r' => {
                        options.add2file = IP_APPEND_OVERWRITE;
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.isec);
                    }
                    b'x' | b'X' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.ox);
                    }
                    b'y' | b'Y' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.oy);
                    }
                    b'z' | b'Z' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.oz);
                    }
                    b'p' => {
                        options.point_out_name =
                            libc::strdup(std::ffi::CString::new(need()).unwrap().as_ptr());
                    }
                    _ => {
                        let message =
                            std::ffi::CString::new(format!("Invalid option {flag}.")).unwrap();
                        exit_error(message.as_ptr());
                    }
                },
                b'i' | b'I' => match second {
                    b'x' | b'X' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.ix);
                    }
                    b'y' | b'Y' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.iy);
                    }
                    b'z' | b'Z' => {
                        let text = need();
                        let value = std::ffi::CString::new(text.clone()).unwrap();
                        libc::sscanf(
                            value.as_ptr(),
                            c"%d%*c%d".as_ptr(),
                            &raw mut options.iz,
                            &raw mut options.iz2,
                        );
                        sections = clip_make_sec_list(&text);
                    }
                    _ => {
                        let message =
                            std::ffi::CString::new(format!("Invalid option {flag}.")).unwrap();
                        exit_error(message.as_ptr());
                    }
                },
                b'c' => match second {
                    b'x' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%g".as_ptr(), &raw mut options.cx);
                    }
                    b'y' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%g".as_ptr(), &raw mut options.cy);
                    }
                    b'z' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%g".as_ptr(), &raw mut options.cz);
                    }
                    b'c' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &raw mut options.thresh);
                    }
                    _ => {
                        let message =
                            std::ffi::CString::new(format!("Invalid option {flag}.")).unwrap();
                        exit_error(message.as_ptr());
                    }
                },
                b'C' => match second {
                    b'X' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.chunk_x);
                    }
                    b'Y' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.chunk_y);
                    }
                    b'Z' => {
                        let value = std::ffi::CString::new(need()).unwrap();
                        libc::sscanf(value.as_ptr(), c"%d".as_ptr(), &raw mut options.chunk_z);
                    }
                    _ => {
                        let message =
                            std::ffi::CString::new(format!("Invalid option {flag}.")).unwrap();
                        exit_error(message.as_ptr());
                    }
                },
                b'P' => {
                    options.plname = libc::strdup(std::ffi::CString::new(need()).unwrap().as_ptr());
                }
                b'O' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    libc::sscanf(
                        value.as_ptr(),
                        c"%d%*c%d".as_ptr(),
                        &raw mut options.new_xoverlap,
                        &raw mut options.new_yoverlap,
                    );
                }
                b'D' => {
                    let name = need();
                    let defect_error = crate::imod::clip::correct_defects::cor_def_parse_defects(
                        &name,
                        false,
                        &mut options.defects,
                        &mut options.cam_size_x,
                        &mut options.cam_size_y,
                    );
                    if defect_error != 0 {
                        // `clip.cpp:562` passes argv[iarg] to a format holding a
                        // single %s, so the file name never reaches the output.
                        let message = std::ffi::CString::new(format!(
                            "Error {}",
                            if defect_error == 1 {
                                "opening"
                            } else {
                                "reading or parsing lines in"
                            }
                        ))
                        .unwrap();
                        exit_error(message.as_ptr());
                    }
                    options.read_defects = 1;
                }
                b'B' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    options.binning = libc::atof(value.as_ptr()) as f32;
                    if options.binning <= 0.49 {
                        exit_error(c"Binning must be at least 0.5".as_ptr());
                    }
                }
                b'S' => options.scale_defects = 1,
                b'R' => {
                    let value = std::ffi::CString::new(need()).unwrap();
                    options.rotation_flip = libc::atoi(value.as_ptr());
                }
                b'e' => {
                    if process == IP_SUPERGAIN {
                        exit_error(c"The -es, -ez, and other EER options cannot be entered with the supergain operation".as_ptr());
                    }
                    if second == b's' {
                        let value = std::ffi::CString::new(need()).unwrap();
                        eer_super = libc::atoi(value.as_ptr());
                        let low = crate::imod::libiimod::iitif::tiff_get_min_eer_super_res();
                        let high = crate::imod::libiimod::iitif::tiff_get_max_eer_super_res();
                        if eer_super < low {
                            eer_super = low;
                        }
                        if eer_super > high {
                            eer_super = high;
                        }
                    } else if second == b'z' {
                        let value = std::ffi::CString::new(need()).unwrap();
                        eer_group = libc::atoi(value.as_ptr());
                    } else if second == b't' {
                        eer_flags = crate::imod::libiimod::iitif::IIFLAG_SKIP_EER_DIRS;
                    } else if second == b'p' {
                        let value = std::ffi::CString::new(need()).unwrap();
                        fei_def_pad = libc::atoi(value.as_ptr());
                        fei_def_pad = fei_def_pad.max(0);
                        fei_pad_entered = 1;
                    } else if second == b'd' {
                        dump_defect_name = Some(std::ffi::CString::new(need()).unwrap());
                    } else if second == b'g' {
                        options.super_gain_name =
                            libc::strdup(std::ffi::CString::new(need()).unwrap().as_ptr());
                    } else if second == b'a' {
                        let value = std::ffi::CString::new(need()).unwrap();
                        let j = libc::atoi(value.as_ptr());
                        if j != 0 {
                            eer_flags |= crate::imod::libiimod::iitif::IIFLAG_ANTIALIAS_EER;
                        }
                        if j == 1 {
                            eer_flags |= crate::imod::libiimod::iitif::IIFLAG_EER_USE_LANCZOS;
                        }
                    } else if second == b'c' {
                        let value = std::ffi::CString::new(need()).unwrap();
                        let mut j = libc::atoi(value.as_ptr()) - 1;
                        if j < 0 {
                            j = 0;
                        }
                        if j > crate::imod::libiimod::iitif::EER_AA_SCALING_MASK {
                            j = crate::imod::libiimod::iitif::EER_AA_SCALING_MASK;
                        }
                        eer_flags |= j << crate::imod::libiimod::iitif::EER_AA_SCALING_BIT_SHIFT;
                    } else {
                        let message =
                            std::ffi::CString::new(format!("Invalid option {flag}.")).unwrap();
                        exit_error(message.as_ptr());
                    }
                }
                _ => {
                    let message =
                        std::ffi::CString::new(format!("Invalid option {flag}.")).unwrap();
                    exit_error(message.as_ptr());
                }
            }
            position += 1;
        }
        // `clip.cpp:621`: replaceFileArgVec expands wild cards only on Windows;
        // on this platform expandArgList returns the vector unchanged.
        if options.mode == mrcfiles::MRC_MODE_HALF_FLOAT {
            crate::imod::libcfshr::b3dutil::set_float_16_output_mode(1, 1);
            options.mode = mrcfiles::MRC_MODE_FLOAT;
        }
        if !sections.is_empty() {
            options.nofsecs = sections.len() as i32;
            options.secs = libc::malloc(core::mem::size_of_val(sections.as_slice())).cast();
            if options.secs.is_null() {
                exit_error(c"Memory allocation error.".as_ptr());
            }
            libc::memcpy(
                options.secs.cast(),
                sections.as_ptr().cast(),
                core::mem::size_of_val(sections.as_slice()),
            );
        }
        // `clip.cpp:625-633`: EER antialiasing defaults are settled after the
        // option loop and before tiffSetEERreadProperties.
        if eer_super < 0 && (eer_flags & crate::imod::libiimod::iitif::IIFLAG_ANTIALIAS_EER) == 0 {
            libc::printf(c"Using antialiasing for the EER reduction\n".as_ptr());
            eer_flags |= crate::imod::libiimod::iitif::IIFLAG_ANTIALIAS_EER
                | crate::imod::libiimod::iitif::IIFLAG_EER_USE_LANCZOS;
        }
        if fei_pad_entered == 0
            && (eer_flags & crate::imod::libiimod::iitif::IIFLAG_ANTIALIAS_EER) != 0
            && eer_super < 2
        {
            fei_def_pad = if eer_super < -2 { 40 } else { 20 };
        }
        if options.chunk_x != IP_DEFAULT
            || options.chunk_y != IP_DEFAULT
            || options.chunk_z != IP_DEFAULT
        {
            // `b3dutil.h:60` OUTPUT_TYPE_HDF is 5, the same value as IIFILE_HDF.
            if format_set >= 0 && format_set != iimage::IIFILE_HDF {
                exit_error(
                    c"You cannot specify chunk sizes and an output format other than HDF".as_ptr(),
                );
            }
            crate::imod::libcfshr::b3dutil::override_output_type(iimage::IIFILE_HDF);
        }
        crate::imod::libiimod::iitif::tiff_set_eer_read_properties(eer_super, eer_group, eer_flags);
        if options.read_defects != 0 {
            if options.cam_size_x == 0 || options.cam_size_y == 0 {
                exit_error(c"Problem with defect correction - Defect list file must have CameraSizeX and CameraSizeY entries".as_ptr());
            }
            crate::imod::clip::correct_defects::cor_def_flip_defects_in_y(
                &mut options.defects,
                options.cam_size_x,
                options.cam_size_y,
                0,
            );
            crate::imod::clip::correct_defects::cor_def_find_touching_pixels(
                &mut options.defects,
                options.cam_size_x,
                options.cam_size_y,
                0,
            );
        }
        if process == IP_DEFECTMAP
            && crate::imod::libcfshr::b3dutil::b3d_output_file_type() == iimage::IIFILE_TIFF
        {
            crate::imod::libcfshr::b3dutil::set_tiff_compression_type(3, 0);
        }
        if options.x != IP_DEFAULT && (options.cx != IP_DEFAULT as f32 || options.ix != IP_DEFAULT)
        {
            exit_error(c"You cannot use -x together with -cx or -ix".as_ptr());
        }
        if options.y != IP_DEFAULT && (options.cy != IP_DEFAULT as f32 || options.iy != IP_DEFAULT)
        {
            exit_error(c"You cannot use -y together with -cy or -iy".as_ptr());
        }
        if process == IP_FLATFIELD
            && (options.x != IP_DEFAULT
                || options.cx != IP_DEFAULT as f32
                || options.ix != IP_DEFAULT
                || options.y != IP_DEFAULT
                || options.cy != IP_DEFAULT as f32
                || options.iy != IP_DEFAULT)
        {
            exit_error(c"You cannot change the input size for flatfield process".as_ptr());
        }
        if process == IP_INTEGRAL
            && options.val == IP_DEFAULT as f32
            && ((options.low != IP_DEFAULT as f32) as i32
                + (options.high != IP_DEFAULT as f32) as i32
                != 1)
        {
            exit_error(c"You must enter -n and either -l OR -h for integral process".as_ptr());
        }
        if process == IP_BOXSD {
            if options.val == IP_DEFAULT as f32 {
                options.val = -2.;
            } else if options.val.abs() < 0.9 {
                exit_error(c"Reduction factor (-n) must be at least 1 for boxsd process".as_ptr());
            }
            if options.low == IP_DEFAULT as f32 {
                options.low = 6. * options.val.abs().round();
            } else if options.low / options.val.abs() < 4. {
                exit_error(c"Box size (-l) must be at least 4 times reduction factor (-n) for boxsd process".as_ptr());
            }
        }
        let data = &raw[position..];
        // `clip.cpp:186`: procout is cleared only by info, histogram and stats.
        let procout = !matches!(process, IP_INFO | IP_STAT | IP_HISTOGRAM);
        // `clip.cpp:187`: needtwo is set by add, multiply, subtract and divide.
        let need_two = matches!(process, IP_ADD | IP_MULTIPLY | IP_SUBTRACT | IP_DIVIDE);
        let input_count = if !procout || process == IP_BLANKFILE {
            if data.is_empty() {
                usage();
                std::process::exit(3);
            }
            data.len()
        } else {
            if data.len() < 2 {
                usage();
                std::process::exit(3);
            }
            data.len() - 1
        };
        options.infiles = input_count as i32;
        let mut strings: Vec<std::ffi::CString> = data
            .iter()
            .map(|s| std::ffi::CString::new(s.as_bytes()).unwrap())
            .collect();
        let mut names: Vec<*mut core::ffi::c_char> =
            strings.iter_mut().map(|s| s.as_ptr().cast_mut()).collect();
        options.fnames = names.as_mut_ptr();
        options.command = command_string.as_ptr().cast_mut();
        let mut input: MrcHeader = core::mem::zeroed();
        let mut second: MrcHeader = core::mem::zeroed();
        let mut output: MrcHeader = core::mem::zeroed();
        // `clip.cpp` advances iarg past each opened input file; a later
        // diagnostic reports argv[iarg].
        let mut file_index = position;
        if process == IP_BLANKFILE {
            if options.ox == IP_DEFAULT
                || options.oy == IP_DEFAULT
                || options.oz == IP_DEFAULT
                || options.pad == IP_DEFAULT as f32
                || options.mode == IP_DEFAULT
            {
                exit_error(
                    c"You must enter -ox, -oy, -oz, -m, and -p for blankfile process".as_ptr(),
                );
            }
            if options.ox < 1 || options.oy < 1 || options.oz < 1 {
                exit_error(c"You must enter a positive output size for all dimensions".as_ptr());
            }
            mrcfiles::mrc_head_new(&mut input, options.ox, options.oy, options.oz, options.mode);
            crate::imod::libcfshr::b3dutil::override_all_big_tiff(1);
        } else {
            input.fp = iimage::ii_fopen(strings[0].as_ptr(), c"rb".as_ptr()).cast();
            if input.fp.is_null() {
                let message = std::ffi::CString::new(format!("Error opening {}", data[0])).unwrap();
                exit_error(message.as_ptr());
            }
            if mrcfiles::mrc_head_read(input.fp.cast(), &mut input) != 0 {
                let message = std::ffi::CString::new(format!("Error reading {}", data[0])).unwrap();
                exit_error(message.as_ptr());
            }
            input.pathname = strings[0].as_ptr().cast_mut();
            file_index += 1;
        }
        if options.add2file == IP_APPEND_FALSE {
            iimage::ii_use_tiff_threads_for_fp(input.fp.cast(), 0);
        }
        output = core::ptr::read(&input);
        mrcfiles::mrc_init_output_header(&mut output);
        if input_count > 1 {
            second.fp = iimage::ii_fopen(strings[1].as_ptr(), c"rb".as_ptr()).cast();
            if second.fp.is_null() {
                let message = std::ffi::CString::new(format!("Error opening {}", data[1])).unwrap();
                exit_error(message.as_ptr());
            }
            if mrcfiles::mrc_head_read(second.fp.cast(), &mut second) != 0 {
                if process == IP_INFO {
                    crate::imod::clip::file_io::mrc_head_print(&input);
                    libc::printf(c"**********************************************\n".as_ptr());
                    libc::printf(c"WARNING: This file is not a readable MRC file.\n".as_ptr());
                    libc::printf(c"**********************************************\n".as_ptr());
                }
                let message = std::ffi::CString::new(format!("Error reading {}", data[1])).unwrap();
                exit_error(message.as_ptr());
            }
            second.pathname = strings[1].as_ptr().cast_mut();
            file_index += 1;
        }
        if matches!(process, IP_NORMALIZE | IP_UNPACK | IP_DEFECTMAP)
            && input_count == 2
            && options.read_defects == 0
        {
            let image_file = iimage::ii_lookup_file_from_fp(second.fp.cast());
            let mut count = 0_u32;
            let mut defect_text: *mut core::ffi::c_char = core::ptr::null_mut();
            if !image_file.is_null()
                && (*image_file).file == iimage::IIFILE_TIFF
                && crate::imod::libiimod::iitif::tiff_get_array(
                    image_file,
                    65_100,
                    &mut count,
                    (&mut defect_text).cast(),
                ) > 0
            {
                let super_fac = if input.nx >= second.nx {
                    let factor = input.nx / second.nx;
                    if input.ny / second.ny != factor
                        || factor * second.nx != input.nx
                        || factor * second.ny != input.ny
                        || !matches!(factor, 1 | 2 | 4)
                    {
                        let message = std::ffi::CString::new(format!(
                            "Image file size ({} x {}) must be exactly the same, twice, or 4 times the gain reference size ({} x {})",
                            input.nx, input.ny, second.nx, second.ny
                        ))
                        .unwrap();
                        exit_error(message.as_ptr());
                    }
                    factor
                } else {
                    let factor = -(second.nx / input.nx);
                    if second.ny / input.ny != -factor
                        || second.ny / input.ny * input.nx != second.nx
                        || second.ny / input.ny * input.ny != second.ny
                        || !matches!(-factor, 2 | 4 | 8)
                    {
                        let message = std::ffi::CString::new(format!(
                            "Image file size ({} x {}) must be exactly the 1/2, 1/4, or 1/8 times the gain reference size ({} x {})",
                            input.nx, input.ny, second.nx, second.ny
                        ))
                        .unwrap();
                        exit_error(message.as_ptr());
                    }
                    factor
                };
                let mut message = [0_i8; 1024];
                if crate::imod::clip::correct_defects::cor_def_process_fei_defects(
                    image_file,
                    &mut options.defects,
                    second.nx,
                    second.ny,
                    true,
                    super_fac,
                    fei_def_pad,
                    dump_defect_name
                        .as_ref()
                        .map_or(core::ptr::null(), |name| name.as_ptr()),
                    message.as_mut_ptr(),
                    1000,
                ) != 0
                {
                    exit_error(message.as_ptr());
                }
                options.read_defects = 1;
                options.cam_size_x = input.nx;
                options.cam_size_y = input.ny;
            }
        }
        if procout && (!need_two || input_count > 1) {
            options.ofname = strings.last().unwrap().as_ptr().cast_mut();
            output.fp = if matches!(process, IP_SUPERGAIN | IP_PLANARFIT) {
                crate::imod::libcfshr::b3dutil::imod_backup_file(strings.last().unwrap().as_ptr());
                libc::fopen(strings.last().unwrap().as_ptr(), c"w".as_ptr()).cast()
            } else if options.add2file != IP_APPEND_FALSE {
                iimage::ii_fopen(strings.last().unwrap().as_ptr(), c"rb+".as_ptr()).cast()
            } else if process != IP_SPLITRGB {
                if libc::getenv(c"IMOD_NO_IMAGE_BACKUP".as_ptr()).is_null() {
                    crate::imod::libcfshr::b3dutil::imod_backup_file(
                        strings.last().unwrap().as_ptr(),
                    );
                }
                if process == IP_FFT
                    && input.mode != mrcfiles::MRC_MODE_COMPLEX_FLOAT
                    && matches!(
                        crate::imod::libcfshr::b3dutil::b3d_output_file_type(),
                        iimage::IIFILE_TIFF | iimage::IIFILE_JPEG
                    )
                {
                    libc::printf(
                        c"WARNING: clip - Writing an MRC file; TIFF or JPEG files cannot contain FFTs\n"
                            .as_ptr(),
                    );
                    crate::imod::libcfshr::b3dutil::override_output_type(iimage::IIFILE_MRC);
                }
                iimage::ii_fopen(strings.last().unwrap().as_ptr(), c"wb+".as_ptr()).cast()
            } else {
                // `clip.cpp:817`: splitrgb leaves hout.fp as the copy of hin.fp.
                output.fp
            };
            if options.add2file != IP_APPEND_FALSE {
                if output.fp.is_null() {
                    let message =
                        std::ffi::CString::new(format!("Error finding {}", data.last().unwrap()))
                            .unwrap();
                    exit_error(message.as_ptr());
                }
                if mrcfiles::mrc_head_read(output.fp.cast(), &mut output) != 0 {
                    // `clip.cpp:812` reports argv[iarg], the next unconsumed
                    // argument, not the output file name.
                    let message = std::ffi::CString::new(format!(
                        "Error reading {}",
                        raw.get(file_index).cloned().unwrap_or_default()
                    ))
                    .unwrap();
                    exit_error(message.as_ptr());
                }
            }
            if output.fp.is_null() {
                let message = std::ffi::CString::new(format!(
                    "Error opening output file {}",
                    data.last().unwrap()
                ))
                .unwrap();
                exit_error(message.as_ptr());
            }
        }
        options.hin = &mut input;
        options.hin2 = &mut second;
        options.hout = &mut output;
        if options.from_one != 0 {
            if matches!(options.add2file, IP_APPEND_OVERWRITE | IP_APPEND_TRUNCATE) {
                options.isec -= 1;
            }
            if options.nofsecs != IP_DEFAULT {
                for section in 0..options.nofsecs {
                    *options.secs.add(section as usize) -= 1;
                }
            }
            if options.cz != IP_DEFAULT as f32 {
                options.cz -= 1.;
            }
            if options.dim == 2 {
                if options.iz != IP_DEFAULT {
                    options.iz -= 1;
                }
                if options.iz2 != IP_DEFAULT {
                    options.iz2 -= 1;
                }
            }
        }
        if matches!(process, IP_LOGARITHM | IP_FLATFIELD) && options.mode == IP_DEFAULT {
            options.mode = crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT;
        }
        let result = match process {
            IP_ADD | IP_AVERAGE | IP_VARIANCE | IP_STANDEV | IP_SUBTRACT => {
                processing::clip_average(&mut input, &mut second, &mut output, &mut options)
            }
            IP_MULTIPLY | IP_DIVIDE => {
                processing::clip_multdiv(&mut input, &mut second, &mut output, &mut options)
            }
            IP_UNPACK | IP_NORMALIZE => {
                processing::clip_unpack(&mut input, &mut second, &mut output, &mut options)
            }
            IP_BLANKFILE => processing::clip_blank_file(&mut output, &mut options),
            IP_BRIGHTNESS | IP_CONTRAST | IP_SHADOW | IP_RESIZE | IP_THRESHOLD | IP_TRUNCATE
            | IP_UNWRAP | IP_LOGARITHM | IP_SQROOT | IP_INTEGRAL | IP_BOXSD => {
                processing::clip_scaling(&mut input, &mut output, &mut options)
            }
            IP_COLOR => processing::clip_color(&mut input, &mut output, &mut options),
            IP_CORRELATE => {
                correlation::grap_corr(&mut input, &mut second, &mut output, &mut options)
            }
            IP_FFT => fft::clip_fft(&mut input, &mut output, &mut options),
            IP_FILTER => filter::clip_bandpass_filter(&mut input, &mut output, &mut options),
            IP_FLIP => processing::clip_flip(&mut input, &mut output, &mut options),
            IP_INFO => crate::imod::clip::file_io::mrc_head_print(&input),
            IP_STAT => processing::clip_stat(&mut input, &mut options),
            IP_HISTOGRAM => processing::clip_histogram(&mut input, &mut options),
            IP_MEDIAN => processing::clip_median(
                &mut input,
                &mut output,
                &mut options,
                core::ptr::null_mut(),
                0,
                0,
            ),
            IP_LAPLACIAN | IP_SMOOTH | IP_SHARPEN => {
                processing::clip_convolve(&mut input, &mut output, &mut options)
            }
            IP_GRADIENT | IP_GRAHAM | IP_PREWITT | IP_SOBEL => {
                processing::clip_edge(&mut input, &mut output, &mut options)
            }
            IP_DIFFUSION => processing::clip_diffusion(&mut input, &mut output, &mut options),
            IP_SPECTRUM => processing::clip_spectrum(&mut input, &mut output, &mut options),
            IP_QUADRANT => processing::clip_quadrant(&mut input, &mut output, &mut options),
            IP_SPLITRGB => processing::clip_splitrgb(&mut input, &mut options),
            IP_JOINRGB => {
                processing::clip_joinrgb(&mut input, &mut second, &mut output, &mut options)
            }
            IP_DEFECTMAP => processing::clip_defect_map(&mut input, &mut output, &mut options),
            IP_SUPERGAIN => processing::clip_super_gain(&mut input, output.fp.cast(), &mut options),
            IP_FILLEDGE => {
                processing::fill_drift_corrected_edges(&mut input, &mut output, &mut options)
            }
            IP_PLANARFIT | IP_FLATFIELD => {
                processing::clip_planar_fit(&mut input, &mut output, &mut options)
            }
            _ => -1,
        };
        if result != 0 {
            std::process::exit(result);
        }
        if process != IP_BLANKFILE && !input.fp.is_null() {
            iimage::ii_close_tiff_copies_for_fp(input.fp.cast());
            iimage::ii_fclose(input.fp.cast());
        }
        if procout && process != IP_SPLITRGB && process != IP_SUPERGAIN && !output.fp.is_null() {
            iimage::ii_fclose(output.fp.cast());
        }
        if view {
            let view_command =
                std::ffi::CString::new(format!("3dmod {}", data.last().unwrap())).unwrap();
            libc::system(view_command.as_ptr());
        }
    }
}
/// Original: `clipMakeSecList` (`clip.cpp:1002`).
///
/// Returns the same inclusive, ascending expansion which the C allocator
/// returned.  The source deliberately reverses a descending range before
/// expanding it, so `4-2` becomes `2, 3, 4`.
pub fn clip_make_sec_list(clst: &str) -> Vec<i32> {
    let bytes = clst.as_bytes();
    if bytes.is_empty() {
        return Vec::new();
    }
    let parse_at = |start: usize| -> i32 {
        let tail = &clst[start..];
        let end = tail
            .bytes()
            .position(|byte| byte == b',' || byte == b'-')
            .unwrap_or(tail.len());
        tail[..end].parse::<i32>().unwrap_or(0)
    };
    let mut secs = vec![parse_at(0)];
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b',' => {
                index += 1;
                secs.push(parse_at(index));
            }
            b'-' => {
                index += 1;
                let mut top = parse_at(index);
                let last = secs.len() - 1;
                if top < secs[last] {
                    core::mem::swap(&mut top, &mut secs[last]);
                }
                let range = top - secs[last];
                for value in 1..=range {
                    secs.push(secs[last] + value);
                }
            }
            _ => {}
        }
        index += 1;
    }
    secs
}

#[cfg(test)]
mod tests {
    use super::clip_make_sec_list;

    #[test]
    fn clip_make_sec_list_expands_source_syntax() {
        assert_eq!(clip_make_sec_list("0,4-10"), vec![0, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(clip_make_sec_list("4-2"), vec![2, 3, 4]);
    }
}
