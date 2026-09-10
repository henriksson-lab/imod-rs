//! Scaffold for `IMOD/clip/clip.cpp` and its paired `IMOD/clip/clip.h`.
#![allow(dead_code)]

use crate::imod::libiimod::mrcfiles::MrcHeader;

pub const IP_NONE: i32 = 0;
pub const IP_DEFAULT: i32 = -99_999;
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
    // `__DATE__` and `__TIME__` are C compilation properties; retain the source
    // IMOD release identifier while the Rust build has no corresponding pair.
    println!("clip: Command Line Image Processing. 4.8.16");
    crate::imod::libcfshr::b3dutil::imod_copyright();
    print!(
        "----------------------------------------------------\n\
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
    );
}
/// Original: `show_error` (`clip.cpp:124`).
pub fn show_error(message: &str) {
    println!("ERROR: {message}");
    println!("ERROR: {message}");
}
/// Original: `show_warning` (`clip.cpp:133`).
pub fn show_warning(reason: &str) {
    println!("WARNING: {reason}");
}
/// Original: `show_status` (`clip.cpp:138`).
pub fn show_status(info: &str) {
    print!("{info}");
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
        if raw.len() < 3 {
            usage();
            return;
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
        let mut dump_defect_name: Option<std::path::PathBuf> = None;
        let mut format_set: Option<String> = None;
        let mut view = false;
        iimage::get_dflt_eersumming_from_env(&mut eer_super, &mut eer_group);
        if process == IP_SUPERGAIN {
            eer_super = 2;
            eer_group = 250;
        }
        while position < raw.len() && raw[position].starts_with('-') {
            let flag = &raw[position];
            if process == IP_SUPERGAIN && flag.starts_with("-e") {
                show_error(
                    "CLIP - The -es, -ez, and other EER options cannot be entered with the supergain operation",
                );
                std::process::exit(1);
            }
            let mut need = || {
                position += 1;
                raw.get(position).cloned().unwrap_or_default()
            };
            match flag.as_str() {
                "-2" | "-2d" => options.dim = 2,
                "-3" | "-3d" => {
                    if process != IP_QUADRANT {
                        options.dim = 3;
                    }
                }
                flag if flag.starts_with("-a") => options.add2file = IP_APPEND_ADD,
                flag if flag.starts_with("-s") => options.sano = 1,
                flag if flag.starts_with("-1") => options.from_one = 1,
                flag if flag.starts_with("-n") => {
                    options.val = need().parse().unwrap_or(IP_DEFAULT as f32)
                }
                flag if flag.starts_with("-E") => {
                    let value = need();
                    let mut terms = value.splitn(2, |ch: char| {
                        !ch.is_ascii_digit() && !matches!(ch, '.' | '+' | '-' | 'e' | 'E')
                    });
                    options.pctl_frac = terms
                        .next()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(IP_DEFAULT as f32);
                    if terms
                        .next()
                        .and_then(|x| x.parse::<i32>().ok())
                        .unwrap_or(0)
                        < 0
                    {
                        options.pctl_frac = -options.pctl_frac;
                    }
                }
                flag if flag.starts_with("-F") => {
                    let value = need();
                    let mut terms = value.splitn(2, |ch: char| {
                        !ch.is_ascii_digit() && !matches!(ch, '.' | '+' | '-' | 'e' | 'E')
                    });
                    options.falloff_frac = terms
                        .next()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(IP_DEFAULT as f32);
                    if terms
                        .next()
                        .and_then(|x| x.parse::<i32>().ok())
                        .unwrap_or(0)
                        < 0
                    {
                        options.falloff_frac = -options.falloff_frac;
                    }
                }
                flag if flag.starts_with("-M") => {
                    let value = need();
                    let mut terms = value.splitn(2, |ch: char| {
                        !ch.is_ascii_digit() && !matches!(ch, '.' | '+' | '-' | 'e' | 'E')
                    });
                    options.min_size = terms
                        .next()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(IP_DEFAULT);
                    if terms
                        .next()
                        .and_then(|x| x.parse::<i32>().ok())
                        .unwrap_or(0)
                        < 0
                    {
                        options.min_size = -options.min_size;
                    }
                }
                flag if flag.starts_with("-k") || flag.starts_with("-w") => {
                    options.weight = need().parse().unwrap_or(IP_DEFAULT as f32)
                }
                flag if flag.starts_with("-l") => {
                    options.low = need().parse().unwrap_or(IP_DEFAULT as f32)
                }
                flag if flag.starts_with("-h") => {
                    options.high = need().parse().unwrap_or(IP_DEFAULT as f32)
                }
                flag if flag.starts_with("-t") => {
                    options.thresh = need().parse().unwrap_or(IP_DEFAULT as f32)
                }
                flag if flag.starts_with("-p") => {
                    options.pad = need().parse().unwrap_or(IP_DEFAULT as f32)
                }
                flag if flag.starts_with("-r") => {
                    options.red = need().parse().unwrap_or(IP_DEFAULT as f32)
                }
                flag if flag.starts_with("-g") => {
                    options.green = need().parse().unwrap_or(IP_DEFAULT as f32)
                }
                flag if flag.starts_with("-b") => {
                    options.blue = need().parse().unwrap_or(IP_DEFAULT as f32)
                }
                "-ix" | "-iX" => options.ix = need().parse().unwrap_or(IP_DEFAULT),
                "-iy" | "-iY" => options.iy = need().parse().unwrap_or(IP_DEFAULT),
                flag if flag.starts_with("-x") || flag.starts_with("-X") => {
                    let value = need();
                    let mut terms = value.splitn(2, |ch: char| {
                        !ch.is_ascii_digit() && !matches!(ch, '+' | '-')
                    });
                    options.x = terms
                        .next()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(IP_DEFAULT);
                    options.x2 = terms
                        .next()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(IP_DEFAULT);
                }
                flag if flag.starts_with("-y") || flag.starts_with("-Y") => {
                    let value = need();
                    let mut terms = value.splitn(2, |ch: char| {
                        !ch.is_ascii_digit() && !matches!(ch, '+' | '-')
                    });
                    options.y = terms
                        .next()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(IP_DEFAULT);
                    options.y2 = terms
                        .next()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(IP_DEFAULT);
                }
                "-ox" | "-oX" => options.ox = need().parse().unwrap_or(IP_DEFAULT),
                "-oy" | "-oY" => options.oy = need().parse().unwrap_or(IP_DEFAULT),
                "-oz" | "-oZ" => options.oz = need().parse().unwrap_or(IP_DEFAULT),
                "-op" => {
                    let value = need();
                    options.point_out_name = std::ffi::CString::new(value).unwrap().into_raw();
                }
                "-cx" => options.cx = need().parse().unwrap_or(IP_DEFAULT as f32),
                "-cy" => options.cy = need().parse().unwrap_or(IP_DEFAULT as f32),
                "-cz" => options.cz = need().parse().unwrap_or(IP_DEFAULT as f32),
                "-cc" => options.thresh = need().parse().unwrap_or(IP_DEFAULT as f32),
                "-CX" => options.chunk_x = need().parse().unwrap_or(IP_DEFAULT),
                "-CY" => options.chunk_y = need().parse().unwrap_or(IP_DEFAULT),
                "-CZ" => options.chunk_z = need().parse().unwrap_or(IP_DEFAULT),
                flag if flag.starts_with("-m") => {
                    let value = need();
                    if value == "4-bit" || value == "101" {
                        crate::imod::libcfshr::b3dutil::set_4_bit_output_mode(1);
                        options.mode = crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE;
                    } else if value == "float16" || value == "12" {
                        crate::imod::libcfshr::b3dutil::set_float_16_output_mode(1, 1);
                        options.mode = crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT;
                    } else {
                        let mode = std::ffi::CString::new(value.as_bytes()).unwrap();
                        options.mode = crate::imod::libcfshr::islice::slice_mode(mode.as_ptr());
                        if options.mode == -2 || options.mode == -3 {
                            crate::imod::libcfshr::b3dutil::override_write_bytes(
                                if options.mode == -2 { 1 } else { 0 },
                            );
                            options.mode = crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE;
                        }
                        if options.mode < 0 {
                            show_error(&format!("CLIP - Invalid mode entry {value}."));
                            std::process::exit(1);
                        }
                        if options.mode == crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT {
                            crate::imod::libcfshr::b3dutil::set_float_16_output_mode(0, 0);
                        }
                    }
                    if process == IP_FLATFIELD
                        && (options.mode != crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT
                            || value == "float16"
                            || value == "12")
                    {
                        show_warning(
                            "clip - Output mode for a flatfield image must be floating point",
                        );
                        options.mode = crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT;
                        crate::imod::libcfshr::b3dutil::set_float_16_output_mode(0, 0);
                    }
                }
                flag if flag.starts_with("-f") => {
                    let value = need();
                    let output_type =
                        crate::imod::libcfshr::b3dutil::set_output_type_from_string(&value);
                    if output_type < 0 {
                        show_error(&format!(
                            "CLIP - Output file format entry {value} is not {}.",
                            if output_type == -1 {
                                "recognized"
                            } else {
                                "available in this copy of IMOD"
                            }
                        ));
                        std::process::exit(1);
                    }
                    format_set = Some(value);
                }
                flag if flag.starts_with("-v") => view = true,
                "-or" => {
                    options.add2file = IP_APPEND_OVERWRITE;
                    options.isec = need().parse().unwrap_or(0);
                }
                "-o" | "-ov" => {
                    options.add2file = IP_APPEND_TRUNCATE;
                    options.isec = need().parse().unwrap_or(0);
                }
                flag if flag.starts_with("-P") => {
                    let value = need();
                    options.plname = std::ffi::CString::new(value).unwrap().into_raw();
                }
                flag if flag.starts_with("-O") => {
                    let value = need();
                    let mut parts = value.splitn(2, |ch: char| {
                        !ch.is_ascii_digit() && !matches!(ch, '+' | '-')
                    });
                    options.new_xoverlap = parts
                        .next()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(IP_DEFAULT);
                    options.new_yoverlap = parts
                        .next()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(IP_DEFAULT);
                }
                flag if flag.starts_with("-D") => {
                    let name = need();
                    let defect_error = crate::imod::clip::correct_defects::cor_def_parse_defects(
                        &name,
                        false,
                        &mut options.defects,
                        &mut options.cam_size_x,
                        &mut options.cam_size_y,
                    );
                    if defect_error != 0 {
                        show_error(&format!(
                            "CLIP - Error {} {}",
                            if defect_error == 1 {
                                "opening"
                            } else {
                                "reading or parsing lines in"
                            },
                            name
                        ));
                        std::process::exit(1);
                    }
                    options.read_defects = 1;
                }
                flag if flag.starts_with("-B") => {
                    options.binning = need().parse().unwrap_or(0.);
                    if options.binning <= 0.49 {
                        show_error("CLIP - Binning must be at least 0.5");
                        std::process::exit(1);
                    }
                }
                flag if flag.starts_with("-S") => options.scale_defects = 1,
                flag if flag.starts_with("-R") => {
                    options.rotation_flip = need().parse().unwrap_or(0)
                }
                "-es" => {
                    if process == IP_SUPERGAIN {
                        show_error(
                            "CLIP - The -es, -ez, and other EER options cannot be entered with the supergain operation",
                        );
                        std::process::exit(1);
                    }
                    eer_super = need().parse().unwrap_or(eer_super).clamp(
                        crate::imod::libiimod::iitif::tiff_get_min_eer_super_res(),
                        crate::imod::libiimod::iitif::tiff_get_max_eer_super_res(),
                    );
                }
                "-ez" => eer_group = need().parse().unwrap_or(eer_group),
                "-et" => eer_flags |= crate::imod::libiimod::iitif::IIFLAG_SKIP_EER_DIRS,
                "-ep" => {
                    fei_def_pad = need().parse().unwrap_or(0).max(0);
                }
                "-ed" => {
                    dump_defect_name = Some(std::path::PathBuf::from(need()));
                }
                "-eg" => {
                    let value = need();
                    options.super_gain_name = std::ffi::CString::new(value).unwrap().into_raw();
                }
                "-ea" => {
                    let value: i32 = need().parse().unwrap_or(0);
                    if value != 0 {
                        eer_flags |= crate::imod::libiimod::iitif::IIFLAG_ANTIALIAS_EER;
                    }
                    if value == 1 {
                        eer_flags |= crate::imod::libiimod::iitif::IIFLAG_EER_USE_LANCZOS;
                    }
                }
                "-ec" => {
                    let value = (need().parse::<i32>().unwrap_or(1) - 1)
                        .clamp(0, crate::imod::libiimod::iitif::EER_AA_SCALING_MASK);
                    eer_flags |= value << crate::imod::libiimod::iitif::EER_AA_SCALING_BIT_SHIFT;
                }
                "-iz" | "-iZ" => {
                    let value = need();
                    let value_c = std::ffi::CString::new(value.as_bytes()).unwrap();
                    libc::sscanf(
                        value_c.as_ptr(),
                        c"%d%*c%d".as_ptr(),
                        &mut options.iz,
                        &mut options.iz2,
                    );
                    sections = clip_make_sec_list(&value);
                }
                _ => {
                    show_error(&format!("CLIP - Invalid option {flag}."));
                    std::process::exit(1);
                }
            }
            position += 1;
        }
        if !sections.is_empty() {
            options.nofsecs = sections.len() as i32;
            options.secs = libc::malloc(core::mem::size_of_val(sections.as_slice())).cast();
            if options.secs.is_null() {
                show_error("CLIP - Memory allocation error.");
                std::process::exit(1);
            }
            libc::memcpy(
                options.secs.cast(),
                sections.as_ptr().cast(),
                core::mem::size_of_val(sections.as_slice()),
            );
        }
        if process == IP_SUPERGAIN {
            options.val = 4.;
        }
        if options.chunk_x != IP_DEFAULT
            || options.chunk_y != IP_DEFAULT
            || options.chunk_z != IP_DEFAULT
        {
            if format_set
                .as_deref()
                .is_some_and(|value| !value.eq_ignore_ascii_case("hdf"))
            {
                show_error(
                    "CLIP - You cannot specify chunk sizes and an output format other than HDF",
                );
                std::process::exit(1);
            }
            crate::imod::libcfshr::b3dutil::override_output_type(iimage::IIFILE_HDF);
        }
        crate::imod::libiimod::iitif::tiff_set_eer_read_properties(eer_super, eer_group, eer_flags);
        if options.read_defects != 0 {
            if options.cam_size_x == 0 || options.cam_size_y == 0 {
                show_error(
                    "CLIP - Problem with defect correction - Defect list file must have CameraSizeX and CameraSizeY entries",
                );
                std::process::exit(1);
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
        if options.x != IP_DEFAULT && (options.cx as i32 != IP_DEFAULT || options.ix != IP_DEFAULT)
        {
            show_error("CLIP - You cannot use -x together with -cx or -ix");
            std::process::exit(1);
        }
        if options.y != IP_DEFAULT && (options.cy as i32 != IP_DEFAULT || options.iy != IP_DEFAULT)
        {
            show_error("CLIP - You cannot use -y together with -cy or -iy");
            std::process::exit(1);
        }
        if process == IP_FLATFIELD
            && (options.x != IP_DEFAULT
                || options.cx as i32 != IP_DEFAULT
                || options.ix != IP_DEFAULT
                || options.y != IP_DEFAULT
                || options.cy as i32 != IP_DEFAULT
                || options.iy != IP_DEFAULT)
        {
            show_error("CLIP - You cannot change the input size for flatfield process");
            std::process::exit(1);
        }
        if process == IP_INTEGRAL
            && options.val as i32 == IP_DEFAULT
            && ((options.low as i32 != IP_DEFAULT) as i32
                + (options.high as i32 != IP_DEFAULT) as i32
                != 1)
        {
            show_error("CLIP - You must enter -n and either -l OR -h for integral process");
            std::process::exit(1);
        }
        if process == IP_BOXSD {
            if options.val as i32 == IP_DEFAULT {
                options.val = -2.;
            } else if options.val.abs() < 0.9 {
                show_error("CLIP - Reduction factor (-n) must be at least 1 for boxsd process");
                std::process::exit(1);
            }
            if options.low as i32 == IP_DEFAULT {
                options.low = 6. * options.val.abs().round();
            } else if options.low / options.val.abs() < 4. {
                show_error(
                    "CLIP - Box size (-l) must be at least 4 times reduction factor (-n) for boxsd process",
                );
                std::process::exit(1);
            }
        }
        let data = &raw[position..];
        let output_needed = !matches!(process, IP_INFO | IP_STAT | IP_HISTOGRAM | IP_SPLITRGB);
        if data.is_empty()
            || ((output_needed || process == IP_SPLITRGB)
                && process != IP_BLANKFILE
                && data.len() < 2)
        {
            usage();
            std::process::exit(3);
        }
        let input_count = if (output_needed || process == IP_SPLITRGB) && process != IP_BLANKFILE {
            data.len() - 1
        } else {
            data.len()
        };
        let need_two = matches!(process, IP_ADD | IP_MULTIPLY | IP_SUBTRACT | IP_DIVIDE);
        if need_two && input_count < 2 {
            usage();
            std::process::exit(3);
        }
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
        if process == IP_BLANKFILE {
            if options.ox == IP_DEFAULT
                || options.oy == IP_DEFAULT
                || options.oz == IP_DEFAULT
                || options.pad as i32 == IP_DEFAULT
                || options.mode == IP_DEFAULT
            {
                show_error("CLIP - You must enter -ox, -oy, -oz, -m, and -p for blankfile process");
                std::process::exit(1);
            }
            if options.ox < 1 || options.oy < 1 || options.oz < 1 {
                show_error("CLIP - You must enter a positive output size for all dimensions");
                std::process::exit(1);
            }
            mrcfiles::mrc_head_new(&mut input, options.ox, options.oy, options.oz, options.mode);
            crate::imod::libcfshr::b3dutil::override_all_big_tiff(1);
        } else {
            input.fp = iimage::ii_fopen(strings[0].as_ptr(), c"rb".as_ptr()).cast();
            if input.fp.is_null() {
                show_error(&format!("CLIP - Error opening {}", data[0]));
                std::process::exit(1);
            }
            if mrcfiles::mrc_head_read(input.fp.cast(), &mut input) != 0 {
                show_error(&format!("CLIP - Error reading {}", data[0]));
                std::process::exit(1);
            }
        }
        if options.add2file == IP_APPEND_FALSE {
            iimage::ii_use_tiff_threads_for_fp(input.fp.cast(), 0);
        }
        output = core::ptr::read(&input);
        mrcfiles::mrc_init_output_header(&mut output);
        if input_count > 1 {
            second.fp = iimage::ii_fopen(strings[1].as_ptr(), c"rb".as_ptr()).cast();
            if second.fp.is_null() {
                show_error(&format!("CLIP - Error opening {}", data[1]));
                std::process::exit(1);
            }
            if mrcfiles::mrc_head_read(second.fp.cast(), &mut second) != 0 {
                if process == IP_INFO {
                    crate::imod::clip::file_io::mrc_head_print(&input);
                    libc::printf(c"**********************************************\n".as_ptr());
                    libc::printf(c"WARNING: This file is not a readable MRC file.\n".as_ptr());
                    libc::printf(c"**********************************************\n".as_ptr());
                }
                show_error(&format!("CLIP - Error reading {}", data[1]));
                std::process::exit(1);
            }
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
                        return;
                    }
                    factor
                } else {
                    let factor = -(second.nx / input.nx);
                    if second.ny / input.ny != -factor
                        || second.ny / input.ny * input.nx != second.nx
                        || second.ny / input.ny * input.ny != second.ny
                        || !matches!(-factor, 2 | 4 | 8)
                    {
                        return;
                    }
                    factor
                };
                let dump_name = dump_defect_name.as_ref().and_then(|path| {
                    std::ffi::CString::new(path.to_string_lossy().as_bytes()).ok()
                });
                let mut message = [0_i8; 1024];
                if crate::imod::clip::correct_defects::cor_def_process_fei_defects(
                    image_file,
                    &mut options.defects,
                    second.nx,
                    second.ny,
                    true,
                    super_fac,
                    fei_def_pad,
                    dump_name
                        .as_ref()
                        .map_or(core::ptr::null(), |name| name.as_ptr()),
                    message.as_mut_ptr(),
                    message.len() as i32,
                ) != 0
                {
                    return;
                }
                options.read_defects = 1;
                options.cam_size_x = input.nx;
                options.cam_size_y = input.ny;
            }
        }
        if output_needed {
            options.ofname = strings.last().unwrap().as_ptr().cast_mut();
            output.fp = if matches!(process, IP_SUPERGAIN | IP_PLANARFIT) {
                crate::imod::libcfshr::b3dutil::imod_backup_file(strings.last().unwrap().as_ptr());
                libc::fopen(strings.last().unwrap().as_ptr(), c"w".as_ptr()).cast()
            } else if options.add2file != IP_APPEND_FALSE {
                iimage::ii_fopen(strings.last().unwrap().as_ptr(), c"rb+".as_ptr()).cast()
            } else {
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
            };
            if output.fp.is_null() {
                if options.add2file != IP_APPEND_FALSE {
                    show_error(&format!("CLIP - Error finding {}", data.last().unwrap()));
                } else {
                    show_error(&format!(
                        "CLIP - Error opening output file {}",
                        data.last().unwrap()
                    ));
                }
                std::process::exit(1);
            }
            if options.add2file != IP_APPEND_FALSE
                && mrcfiles::mrc_head_read(output.fp.cast(), &mut output) != 0
            {
                show_error(&format!("CLIP - Error reading {}", data.last().unwrap()));
                std::process::exit(1);
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
        if output_needed && process != IP_SUPERGAIN && !output.fp.is_null() {
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
