//! Translation of `IMOD/clip/clip.cpp` and its paired `IMOD/clip/clip.h`.
//!
//! **Output.**  Everything the source prints goes through
//! [`crate::imod::libcfshr::b3dutil::c_format`] with the source's own format
//! string, written to [`ImodFile::Stdout`] — which is the *C* stream.  The
//! sibling `clip` modules and `parse_params` still write there, and C stdio is
//! block-buffered under redirection while Rust's is not, so a line that went
//! through `std::io::stdout()` would move ahead of theirs in a captured file.
//!
//! **Scanning.**  `clip.cpp` parses its 30 option values with `sscanf`, not
//! with a parser: it accepts a partial match, leaves the remaining arguments
//! untouched when a conversion fails, and uses `%*c` to step over a separator
//! that may be a comma or an `x`.  [`sscanf`], [`strtod`] and [`strtol`] below
//! are translations of those C library routines, for the same reason
//! `c_format` is a translation of `printf`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, imod_prog_name};
use crate::imod::libcfshr::parse_params::exit_error;
use crate::imod::libcfshr::parse_params::setExitPrefix;
use std::io::Write;

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
#[derive(Clone)]
pub struct ClipOptions {
    pub pname: String,
    pub command: String,
    // C `MrcHeader *hin, *hin2, *hout`.  `main` sets these to the addresses of
    // its own three stack headers, which it also passes to every process
    // routine, so the fields are a second, aliasing path to objects the caller
    // already holds `&mut` to.  Rust cannot express that, and only one routine
    // reads them (`set_mrc_coords`, `file_io.cpp:37-38`), so that routine takes
    // the two headers as arguments instead and the fields are gone.
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
    pub fnames: Vec<String>,
    pub sano: i32,
    pub add2file: i32,
    pub isec: i32,
    pub val: f32,
    pub nofsecs: i32,
    pub secs: Vec<i32>,
    pub out_before: i32,
    pub out_after: i32,
    pub ocanresize: i32,
    pub ocanchmode: i32,
    pub from_one: i32,
    pub ofname: Option<String>,
    pub plname: Option<String>,
    pub super_gain_name: Option<String>,
    pub point_out_name: Option<String>,
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
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "%s: Command Line Image Processing. %s, %s %s\n",
            &[
                CArg::Str("clip"),
                CArg::Str("5.2.17"),
                CArg::Str(crate::imod::libcfshr::b3dutil::IMOD_BUILD_DATE),
                CArg::Str(crate::imod::libcfshr::b3dutil::IMOD_BUILD_TIME),
            ],
        )
        .as_bytes(),
    );
    crate::imod::libcfshr::b3dutil::imod_copyright();
    let _ = ImodFile::Stdout.write_all(
        b"----------------------------------------------------\n\
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
\t[-eg file] File for adjusting super-resolution gain reference.\n\n",
    );
}
/// Original: `show_error` (`clip.cpp:124`).

pub fn show_error(message: &str) {
    // `clip.cpp:124-131` writes to standard output with C `printf`.  Keeping
    // the same stream matters when output is redirected, because the C-derived
    // modules on this path are block-buffered by libc.
    let _ = ImodFile::Stdout.write_all(c_format("ERROR: %s\n", &[CArg::Str(message)]).as_bytes());
}
/// Original: `show_warning` (`clip.cpp:133`).
pub fn show_warning(reason: &str) {
    let _ = ImodFile::Stdout.write_all(c_format("WARNING: %s\n", &[CArg::Str(reason)]).as_bytes());
}
/// Original: `show_status` (`clip.cpp:138`).
pub fn show_status(info: &str) {
    let _ = ImodFile::Stdout.write_all(c_format("%s", &[CArg::Str(info)]).as_bytes());
    let _ = ImodFile::Stdout.flush();
}
/// C++ `default_options` (`clip.cpp:144`).
pub fn default_options(options: &mut ClipOptions) {
    // `clip.cpp:146` clears `hin`, `hin2` and `hout`; those fields are gone,
    // see the note on the struct.
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
    options.secs = Vec::new();
    options.ocanresize = 1;
    options.ocanchmode = 1;
    options.from_one = 0;
    options.ofname = None;
    options.plname = None;
    options.super_gain_name = None;
    options.point_out_name = None;
    options.new_xoverlap = IP_DEFAULT;
    options.new_yoverlap = IP_DEFAULT;
    options.read_defects = 0;
    options.rotation_flip = 0;
    options.binning = IP_DEFAULT as f32;
    options.scale_defects = 0;
}
/// Original: `main` (`clip.cpp:189`).
pub fn clip() {
    use crate::imod::clip::{correlation, fft, filter, processing};
    use crate::imod::libiimod::{iimage, mrcfiles};
    let raw: Vec<String> = std::env::args().collect();
    let progname = imod_prog_name(&raw[0]);
    // `clip.cpp:211` builds the prefix in `viewcmd`, the same 1024-byte buffer
    // it later reuses for the FEI defect message and the `3dmod` command line.
    setExitPrefix(c_format("ERROR: %s - ", &[CArg::Str(&progname)]).as_bytes());
    if raw.len() < 3 {
        usage();
        std::process::exit(3);
    }
    let command = raw[1].clone();
    // `clip.cpp:224-362` tests the command with a run of independent
    // `strncmp`s, so a later match overrides an earlier one.
    let mut process = IP_NONE;
    let mut procout = true;
    let mut need_two = false;
    let mut dim_from_command = 3;
    let mut ocanresize_from_command = 1;
    let mut mode_from_command = IP_DEFAULT;
    {
        let c = command.as_bytes();
        // `strncmp(argv[1], lit, n)`: compares at most `n` bytes and stops at
        // the NUL of either string, so a command shorter than `n` matches only
        // if it is a prefix of the literal *and* the literal ends there too.
        let strncmp_eq = |lit: &str, n: usize| -> bool {
            let l = lit.as_bytes();
            for i in 0..n {
                let a = if i < c.len() { c[i] } else { 0 };
                let b = if i < l.len() { l[i] } else { 0 };
                if a != b {
                    return false;
                }
                if a == 0 {
                    return true;
                }
            }
            true
        };
        if strncmp_eq("add", 3) {
            process = IP_ADD;
            need_two = true;
        }
        if strncmp_eq("avg", 3) || strncmp_eq("average", 3) {
            process = IP_AVERAGE;
        }
        if strncmp_eq("standev", 4) {
            process = IP_STANDEV;
        }
        if strncmp_eq("variance", 3) {
            process = IP_VARIANCE;
        }
        if strncmp_eq("multiply", 3) {
            process = IP_MULTIPLY;
            need_two = true;
        }
        if strncmp_eq("subtract", 3) {
            process = IP_SUBTRACT;
            need_two = true;
        }
        if strncmp_eq("divide", 3) {
            process = IP_DIVIDE;
            need_two = true;
        }
        if strncmp_eq("brightness", 3) {
            process = IP_BRIGHTNESS;
        }
        if strncmp_eq("color", 3) {
            process = IP_COLOR;
        }
        if strncmp_eq("contrast", 3) {
            process = IP_CONTRAST;
        }
        if strncmp_eq("correlation", 3) {
            process = IP_CORRELATE;
        }
        if strncmp_eq("diffusion", 3) {
            process = IP_DIFFUSION;
        }
        if strncmp_eq("info", 3) {
            process = IP_INFO;
            procout = false;
        }
        if strncmp_eq("fft", 3) {
            process = IP_FFT;
        }
        if strncmp_eq("filter", 3) {
            process = IP_FILTER;
        }
        if strncmp_eq("flatfield", 3) {
            process = IP_FLATFIELD;
            ocanresize_from_command = 0;
            mode_from_command = mrcfiles::MRC_MODE_FLOAT;
        }
        if strncmp_eq("flip", 4) {
            process = IP_FLIP;
        }
        if strncmp_eq("rotx", 4) {
            process = IP_FLIP;
        }
        if strncmp_eq("gradient", 4) {
            process = IP_GRADIENT;
        }
        if strncmp_eq("graham", 4) {
            process = IP_GRAHAM;
        }
        if strncmp_eq("histogram", 2) {
            process = IP_HISTOGRAM;
            procout = false;
        }
        if strncmp_eq("laplacian", 2) {
            process = IP_LAPLACIAN;
        }
        if strncmp_eq("median", 2) {
            process = IP_MEDIAN;
        }
        if strncmp_eq("planefit", 2) {
            process = IP_PLANARFIT;
        }
        if strncmp_eq("prewitt", 2) {
            process = IP_PREWITT;
        }
        if strncmp_eq("resize", 3) {
            process = IP_RESIZE;
        }
        if strncmp_eq("shadow", 4) {
            process = IP_SHADOW;
        }
        if strncmp_eq("sharpen", 4) {
            process = IP_SHARPEN;
        }
        if strncmp_eq("smooth", 2) {
            process = IP_SMOOTH;
        }
        if strncmp_eq("sobel", 2) {
            process = IP_SOBEL;
        }
        if strncmp_eq("spectrum", 2) {
            process = IP_SPECTRUM;
        }
        if strncmp_eq("stat", 4) {
            process = IP_STAT;
            procout = false;
        }
        if strncmp_eq("threshold", 3) {
            process = IP_THRESHOLD;
        }
        if strncmp_eq("truncate", 3) {
            process = IP_TRUNCATE;
        }
        if strncmp_eq("unwrap", 3) {
            process = IP_UNWRAP;
        }
        if strncmp_eq("sqroot", 3) {
            process = IP_SQROOT;
        }
        if strncmp_eq("logarithm", 3) {
            process = IP_LOGARITHM;
        }
        if strncmp_eq("quadrant", 2) {
            process = IP_QUADRANT;
            dim_from_command = 2;
        }
        if strncmp_eq("edgefill", 2) {
            process = IP_FILLEDGE;
            dim_from_command = 2;
        }
        if strncmp_eq("unpack", 3) {
            process = IP_UNPACK;
        }
        if strncmp_eq("normalize", 3) {
            process = IP_NORMALIZE;
        }
        if strncmp_eq("defectmap", 3) {
            process = IP_DEFECTMAP;
        }
        if strncmp_eq("supergain", 3) {
            process = IP_SUPERGAIN;
        }
        if strncmp_eq("integral", 3) {
            process = IP_INTEGRAL;
        }
        if strncmp_eq("boxsd", 3) {
            process = IP_BOXSD;
        }
        if strncmp_eq("blankfile", 3) {
            process = IP_BLANKFILE;
        }
        if strncmp_eq("splitrgb", 3) {
            process = IP_SPLITRGB;
        }
        if strncmp_eq("joinrgb", 3) {
            process = IP_JOINRGB;
        }
    }
    if process == IP_NONE {
        usage();
        std::process::exit(1);
    }
    let mut options = ClipOptions {
        pname: String::new(),
        command: command.clone(),
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
        dim: 3,
        infiles: 0,
        fnames: Vec::new(),
        sano: 0,
        add2file: IP_APPEND_FALSE,
        isec: 0,
        val: IP_DEFAULT as f32,
        nofsecs: IP_DEFAULT,
        secs: Vec::new(),
        out_before: IP_DEFAULT,
        out_after: IP_DEFAULT,
        ocanresize: 1,
        ocanchmode: 1,
        from_one: 0,
        ofname: None,
        plname: None,
        super_gain_name: None,
        point_out_name: None,
        new_xoverlap: IP_DEFAULT,
        new_yoverlap: IP_DEFAULT,
        rotation_flip: 0,
        read_defects: 0,
        defects: CameraDefects::new(),
        cam_size_x: 0,
        cam_size_y: 0,
        binning: IP_DEFAULT as f32,
        scale_defects: 0,
    };
    // `clip.cpp:220-221`: `opt.command` is set before `default_options`, which
    // does not touch it; the three command-driven fields at :270, :336 and
    // :340 are set after it, so they survive.
    default_options(&mut options);
    options.dim = dim_from_command;
    options.ocanresize = ocanresize_from_command;
    if mode_from_command != IP_DEFAULT {
        options.mode = mode_from_command;
    }
    let mut eer_super = 2_i32;
    let mut eer_group = 1_i32;
    let mut eer_flags = 0_i32;
    let mut fei_def_pad = 1_i32;
    let mut fei_pad_entered = 0_i32;
    let mut dump_defect_name: Option<String> = None;
    let mut format_set = -2_i32;
    let mut view = false;
    // `clip.cpp:198` declares `int itemp;` without an initializer and reuses
    // it for -E, -F and -M; `sscanf` leaves it untouched when the second
    // conversion fails, so its value carries across option entries.  The
    // uninitialized first read is source-level indeterminate; zero is used.
    let mut itemp = 0_i32;
    iimage::get_dflt_eersumming_from_env(&mut eer_super, &mut eer_group);
    options.process = process;
    options.pname = progname.clone();
    if process == IP_SUPERGAIN {
        eer_group = 250;
        eer_super = 2;
        options.val = 4.;
    }
    // `clip.cpp:378-620`: the option loop switches on the second character of
    // the argument and, for several letters, on the third.
    let mut iarg = 2usize;
    while iarg < raw.len() {
        let flag = raw[iarg].clone();
        let bytes = flag.as_bytes();
        if bytes.first() != Some(&b'-') {
            break;
        }
        let third = *bytes.get(2).unwrap_or(&0);
        // `clip.cpp:403` inspects argv[iarg + 1] before advancing.
        let next_arg = raw.get(iarg + 1).cloned().unwrap_or_default();
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
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f",
                    &mut [ScanArg::Flt(&mut options.val)],
                );
            }
            b'm' => {
                if next_arg == "4-bit" || next_arg == "101" {
                    crate::imod::libcfshr::b3dutil::set_4_bit_output_mode(1);
                    options.mode = mrcfiles::MRC_MODE_BYTE;
                    iarg += 1;
                } else if next_arg == "float16" || next_arg == "12" {
                    options.mode = mrcfiles::MRC_MODE_HALF_FLOAT;
                    iarg += 1;
                } else {
                    iarg += 1;
                    let mode_text = raw.get(iarg).cloned().unwrap_or_default().into_bytes();
                    options.mode = crate::imod::libcfshr::islice::slice_mode(&mode_text);
                }
                if options.mode == SLICE_MODE_UNDEFINED {
                    exit_error(
                        c_format(
                            "Invalid mode entry %s.",
                            &[CArg::Str(raw.get(iarg).map_or("", |v| v.as_str()))],
                        )
                        .as_bytes(),
                    );
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
                iarg += 1;
                let value = raw.get(iarg).cloned().unwrap_or_default();
                format_set = crate::imod::libcfshr::b3dutil::set_output_type_from_string(&value);
                if format_set < 0 {
                    exit_error(
                        c_format(
                            "Output file format entry %s is not %s.",
                            &[
                                CArg::Str(&value),
                                CArg::Str(if format_set == -1 {
                                    "recognized"
                                } else {
                                    "available in this copy of IMOD"
                                }),
                            ],
                        )
                        .as_bytes(),
                    );
                }
            }
            b'v' => view = true,
            b'p' => {
                iarg += 1;
                options.pad = atof(raw.get(iarg).map_or("", |v| v.as_str())) as f32;
            }
            b't' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f",
                    &mut [ScanArg::Flt(&mut options.thresh)],
                );
            }
            b'E' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f%*c%d",
                    &mut [
                        ScanArg::Flt(&mut options.pctl_frac),
                        ScanArg::Int(&mut itemp),
                    ],
                );
                if itemp < 0 {
                    options.pctl_frac = -options.pctl_frac;
                }
            }
            b'F' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f%*c%d",
                    &mut [
                        ScanArg::Flt(&mut options.falloff_frac),
                        ScanArg::Int(&mut itemp),
                    ],
                );
                if itemp < 0 {
                    options.falloff_frac = -options.falloff_frac;
                }
            }
            b'M' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%d%*c%d",
                    &mut [
                        ScanArg::Int(&mut options.min_size),
                        ScanArg::Int(&mut itemp),
                    ],
                );
                if itemp < 0 {
                    options.min_size = -options.min_size;
                }
            }
            b'k' | b'w' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f",
                    &mut [ScanArg::Flt(&mut options.weight)],
                );
            }
            b'r' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f",
                    &mut [ScanArg::Flt(&mut options.red)],
                );
            }
            b'g' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f",
                    &mut [ScanArg::Flt(&mut options.green)],
                );
            }
            b'b' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f",
                    &mut [ScanArg::Flt(&mut options.blue)],
                );
            }
            b'l' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f",
                    &mut [ScanArg::Flt(&mut options.low)],
                );
            }
            b'h' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%f",
                    &mut [ScanArg::Flt(&mut options.high)],
                );
            }
            b'x' | b'X' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%d%*c%d",
                    &mut [ScanArg::Int(&mut options.x), ScanArg::Int(&mut options.x2)],
                );
            }
            b'y' | b'Y' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%d%*c%d",
                    &mut [ScanArg::Int(&mut options.y), ScanArg::Int(&mut options.y2)],
                );
            }
            b'o' => match third {
                0x00 | b' ' | b'v' => {
                    options.add2file = IP_APPEND_TRUNCATE;
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.isec)],
                    );
                }
                b'r' => {
                    options.add2file = IP_APPEND_OVERWRITE;
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.isec)],
                    );
                }
                b'x' | b'X' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.ox)],
                    );
                }
                b'y' | b'Y' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.oy)],
                    );
                }
                b'z' | b'Z' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.oz)],
                    );
                }
                b'p' => {
                    iarg += 1;
                    options.point_out_name = Some(raw.get(iarg).cloned().unwrap_or_default());
                }
                _ => {
                    exit_error(c_format("Invalid option %s.", &[CArg::Str(&flag)]).as_bytes());
                }
            },
            b'i' | b'I' => match third {
                b'x' | b'X' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.ix)],
                    );
                }
                b'y' | b'Y' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.iy)],
                    );
                }
                b'z' | b'Z' => {
                    iarg += 1;
                    let text = raw.get(iarg).cloned().unwrap_or_default();
                    sscanf(
                        &text,
                        "%d%*c%d",
                        &mut [
                            ScanArg::Int(&mut options.iz),
                            ScanArg::Int(&mut options.iz2),
                        ],
                    );
                    options.secs = clip_make_sec_list(&text, &mut options.nofsecs);
                }
                _ => {
                    exit_error(c_format("Invalid option %s.", &[CArg::Str(&flag)]).as_bytes());
                }
            },
            b'c' => match third {
                b'x' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%g",
                        &mut [ScanArg::Flt(&mut options.cx)],
                    );
                }
                b'y' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%g",
                        &mut [ScanArg::Flt(&mut options.cy)],
                    );
                }
                b'z' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%g",
                        &mut [ScanArg::Flt(&mut options.cz)],
                    );
                }
                b'c' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%f",
                        &mut [ScanArg::Flt(&mut options.thresh)],
                    );
                }
                _ => {
                    exit_error(c_format("Invalid option %s.", &[CArg::Str(&flag)]).as_bytes());
                }
            },
            b'C' => match third {
                b'X' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.chunk_x)],
                    );
                }
                b'Y' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.chunk_y)],
                    );
                }
                b'Z' => {
                    iarg += 1;
                    sscanf(
                        raw.get(iarg).map_or("", |v| v.as_str()),
                        "%d",
                        &mut [ScanArg::Int(&mut options.chunk_z)],
                    );
                }
                _ => {
                    exit_error(c_format("Invalid option %s.", &[CArg::Str(&flag)]).as_bytes());
                }
            },
            b'P' => {
                iarg += 1;
                options.plname = Some(raw.get(iarg).cloned().unwrap_or_default());
            }
            b'O' => {
                iarg += 1;
                sscanf(
                    raw.get(iarg).map_or("", |v| v.as_str()),
                    "%d%*c%d",
                    &mut [
                        ScanArg::Int(&mut options.new_xoverlap),
                        ScanArg::Int(&mut options.new_yoverlap),
                    ],
                );
            }
            b'D' => {
                iarg += 1;
                let name = raw.get(iarg).cloned().unwrap_or_default();
                let j = crate::imod::clip::correct_defects::cor_def_parse_defects(
                    &name,
                    false,
                    &mut options.defects,
                    &mut options.cam_size_x,
                    &mut options.cam_size_y,
                );
                if j != 0 {
                    // `clip.cpp:562` passes argv[iarg] to a format holding a
                    // single %s, so the file name never reaches the output.
                    exit_error(
                        c_format(
                            "Error %s",
                            &[CArg::Str(if j == 1 {
                                "opening"
                            } else {
                                "reading or parsing lines in"
                            })],
                        )
                        .as_bytes(),
                    );
                }
                options.read_defects = 1;
            }
            b'B' => {
                iarg += 1;
                options.binning = atof(raw.get(iarg).map_or("", |v| v.as_str())) as f32;
                if options.binning <= 0.49 {
                    exit_error(b"Binning must be at least 0.5");
                }
            }
            b'S' => options.scale_defects = 1,
            b'R' => {
                iarg += 1;
                options.rotation_flip = atoi(raw.get(iarg).map_or("", |v| v.as_str()));
            }
            b'e' => {
                if process == IP_SUPERGAIN {
                    exit_error(
                        b"The -es, -ez, and other EER options cannot be entered with the supergain operation",
                    );
                }
                if third == b's' {
                    iarg += 1;
                    eer_super = atoi(raw.get(iarg).map_or("", |v| v.as_str()));
                    // B3DCLAMP(v, lo, hi) is MAX(lo, MIN(hi, v)).
                    let lo = crate::imod::libiimod::iitif::tiff_get_min_eer_super_res();
                    let hi = crate::imod::libiimod::iitif::tiff_get_max_eer_super_res();
                    eer_super = if lo > (if hi < eer_super { hi } else { eer_super }) {
                        lo
                    } else if hi < eer_super {
                        hi
                    } else {
                        eer_super
                    };
                } else if third == b'z' {
                    iarg += 1;
                    eer_group = atoi(raw.get(iarg).map_or("", |v| v.as_str()));
                } else if third == b't' {
                    eer_flags = crate::imod::libiimod::iitif::IIFLAG_SKIP_EER_DIRS;
                } else if third == b'p' {
                    iarg += 1;
                    fei_def_pad = atoi(raw.get(iarg).map_or("", |v| v.as_str()));
                    fei_def_pad = if fei_def_pad < 0 { 0 } else { fei_def_pad };
                    fei_pad_entered = 1;
                } else if third == b'd' {
                    iarg += 1;
                    dump_defect_name = Some(raw.get(iarg).cloned().unwrap_or_default());
                } else if third == b'g' {
                    iarg += 1;
                    options.super_gain_name = Some(raw.get(iarg).cloned().unwrap_or_default());
                } else if third == b'a' {
                    iarg += 1;
                    let j = atoi(raw.get(iarg).map_or("", |v| v.as_str()));
                    if j != 0 {
                        eer_flags |= crate::imod::libiimod::iitif::IIFLAG_ANTIALIAS_EER;
                    }
                    if j == 1 {
                        eer_flags |= crate::imod::libiimod::iitif::IIFLAG_EER_USE_LANCZOS;
                    }
                } else if third == b'c' {
                    iarg += 1;
                    let mut j = atoi(raw.get(iarg).map_or("", |v| v.as_str())) - 1;
                    let mask = crate::imod::libiimod::iitif::EER_AA_SCALING_MASK;
                    j = if 0 > (if mask < j { mask } else { j }) {
                        0
                    } else if mask < j {
                        mask
                    } else {
                        j
                    };
                    eer_flags |= j << crate::imod::libiimod::iitif::EER_AA_SCALING_BIT_SHIFT;
                } else {
                    exit_error(c_format("Invalid option %s.", &[CArg::Str(&flag)]).as_bytes());
                }
            }
            _ => {
                exit_error(c_format("Invalid option %s.", &[CArg::Str(&flag)]).as_bytes());
            }
        }
        iarg += 1;
    }
    // `clip.cpp:622`: replaceFileArgVec expands wild cards only on Windows;
    // on this platform expandArgList returns the vector unchanged.
    if options.mode == mrcfiles::MRC_MODE_HALF_FLOAT {
        crate::imod::libcfshr::b3dutil::set_float_16_output_mode(1, 1);
        options.mode = mrcfiles::MRC_MODE_FLOAT;
    }
    // `clip.cpp:630`: opt.fnames = &argv[iarg] -- the tail of the argument
    // vector, output file name included.
    options.fnames = raw[iarg.min(raw.len())..].to_vec();
    // `clip.cpp:633-639`: EER antialiasing defaults are settled after the
    // option loop and before tiffSetEERreadProperties.
    if eer_super < 0 && (eer_flags & crate::imod::libiimod::iitif::IIFLAG_ANTIALIAS_EER) == 0 {
        let _ = ImodFile::Stdout.write_all(b"Using antialiasing for the EER reduction\n");
        eer_flags |= crate::imod::libiimod::iitif::IIFLAG_ANTIALIAS_EER
            | crate::imod::libiimod::iitif::IIFLAG_EER_USE_LANCZOS;
    }
    if fei_pad_entered == 0
        && (eer_flags & crate::imod::libiimod::iitif::IIFLAG_ANTIALIAS_EER) != 0
        && eer_super < 2
    {
        fei_def_pad = if eer_super < -2 { 40 } else { 20 };
    }
    crate::imod::libiimod::iitif::tiff_set_eer_read_properties(eer_super, eer_group, eer_flags);
    if !procout || process == IP_BLANKFILE {
        if raw.len() - 1 < iarg {
            usage();
            std::process::exit(3);
        }
        options.infiles = (raw.len() - iarg) as i32;
    } else {
        if raw.len() < iarg + 2 {
            usage();
            std::process::exit(3);
        }
        options.infiles = (raw.len() - iarg - 1) as i32;
    }
    if options.chunk_x != IP_DEFAULT
        || options.chunk_y != IP_DEFAULT
        || options.chunk_z != IP_DEFAULT
    {
        // `b3dutil.h:60` OUTPUT_TYPE_HDF is 5, the same value as IIFILE_HDF.
        if format_set >= 0 && format_set != iimage::IIFILE_HDF {
            exit_error(b"You cannot specify chunk sizes and an output format other than HDF");
        }
        crate::imod::libcfshr::b3dutil::override_output_type(iimage::IIFILE_HDF);
    }
    if options.read_defects != 0 {
        if options.cam_size_x == 0 || options.cam_size_y == 0 {
            exit_error(
                b"Problem with defect correction - Defect list file must have CameraSizeX and CameraSizeY entries",
            );
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
    if options.x != IP_DEFAULT && (options.cx != IP_DEFAULT as f32 || options.ix != IP_DEFAULT) {
        exit_error(b"You cannot use -x together with -cx or -ix");
    }
    if options.y != IP_DEFAULT && (options.cy != IP_DEFAULT as f32 || options.iy != IP_DEFAULT) {
        exit_error(b"You cannot use -y together with -cy or -iy");
    }
    if process == IP_FLATFIELD
        && (options.x != IP_DEFAULT
            || options.cx != IP_DEFAULT as f32
            || options.ix != IP_DEFAULT
            || options.y != IP_DEFAULT
            || options.cy != IP_DEFAULT as f32
            || options.iy != IP_DEFAULT)
    {
        exit_error(b"You cannot change the input size for flatfield process");
    }
    if process == IP_INTEGRAL
        && options.val == IP_DEFAULT as f32
        && ((options.low != IP_DEFAULT as f32) as i32 + (options.high != IP_DEFAULT as f32) as i32
            != 1)
    {
        exit_error(b"You must enter -n and either -l OR -h for integral process");
    }
    if process == IP_BOXSD {
        if options.val == IP_DEFAULT as f32 {
            options.val = -2.;
        } else if options.val.abs() < 0.9 {
            exit_error(b"Reduction factor (-n) must be at least 1 for boxsd process");
        }
        if options.low == IP_DEFAULT as f32 {
            // B3DNINT is floor(x + 0.5).
            options.low = 6. * (options.val.abs() + 0.5).floor();
        } else if options.low / options.val.abs() < 4. {
            exit_error(
                b"Box size (-l) must be at least 4 times reduction factor (-n) for boxsd process",
            );
        }
    }
    let mut input = mrcfiles::MrcHeader::default();
    let mut second = mrcfiles::MrcHeader::default();
    let mut output;
    // `clip.cpp` advances iarg past each opened input file; a later
    // diagnostic reports argv[iarg].
    if process == IP_BLANKFILE {
        if options.ox == IP_DEFAULT
            || options.oy == IP_DEFAULT
            || options.oz == IP_DEFAULT
            || options.pad == IP_DEFAULT as f32
            || options.mode == IP_DEFAULT
        {
            exit_error(b"You must enter -ox, -oy, -oz, -m, and -p for blankfile process");
        }
        if options.ox < 1 || options.oy < 1 || options.oz < 1 {
            exit_error(b"You must enter a positive output size for all dimensions");
        }
        mrcfiles::mrc_head_new(&mut input, options.ox, options.oy, options.oz, options.mode);
        crate::imod::libcfshr::b3dutil::override_all_big_tiff(1);
    } else {
        input.fp = iimage::ii_fopen(raw[iarg].as_bytes(), "rb");
        input.pathname = Some(raw[iarg].clone());
        if input.fp.is_none() {
            exit_error(c_format("Error opening %s", &[CArg::Str(&raw[iarg])]).as_bytes());
        }
        let mut fp = input.fp.clone().unwrap();
        if mrcfiles::mrc_head_read(&mut fp, &mut input) != 0 {
            exit_error(c_format("Error reading %s", &[CArg::Str(&raw[iarg])]).as_bytes());
        }
        iarg += 1;
    }
    output = input.clone();
    if options.add2file == IP_APPEND_FALSE
        && let Some(fp) = input.fp.as_mut()
    {
        iimage::ii_use_tiff_threads_for_fp(fp, 0);
    }
    mrcfiles::mrc_init_output_header(&mut output);
    if options.infiles > 1 {
        second.fp = iimage::ii_fopen(raw[iarg].as_bytes(), "rb");
        second.pathname = Some(raw[iarg].clone());
        if second.fp.is_none() {
            exit_error(c_format("Error opening %s", &[CArg::Str(&raw[iarg])]).as_bytes());
        }
        let mut fp = second.fp.clone().unwrap();
        if mrcfiles::mrc_head_read(&mut fp, &mut second) != 0 {
            if process == IP_INFO {
                crate::imod::clip::file_io::mrc_head_print(&input);
                let _ = ImodFile::Stdout.write_all(
                    b"**********************************************\n\
WARNING: This file is not a readable MRC file.\n\
**********************************************\n",
                );
            }
            exit_error(c_format("Error reading %s", &[CArg::Str(&raw[iarg])]).as_bytes());
        }
        iarg += 1;
    }
    if matches!(process, IP_NORMALIZE | IP_UNPACK | IP_DEFECTMAP)
        && options.infiles == 2
        && options.read_defects == 0
    {
        let ii_file = second.fp.as_ref().and_then(iimage::ii_lookup_file_from_fp);
        if let Some(ii_file) = ii_file
            && unsafe { (*ii_file).file } == iimage::IIFILE_TIFF
            && unsafe { crate::imod::libiimod::iitif::tiff_get_array(&mut *ii_file, 65_100) }
                .is_ok()
        {
            let super_fac;
            if input.nx >= second.nx {
                super_fac = input.nx / second.nx;
                let j = input.ny / second.ny;
                if j != super_fac
                    || super_fac * second.nx != input.nx
                    || super_fac * second.ny != input.ny
                    || (super_fac != 1 && super_fac != 2 && super_fac != 4)
                {
                    exit_error(
                        c_format(
                            "Image file size (%d x %d) must be exactly the same, twice, or 4 times the gain reference size (%d x %d)",
                            &[
                                CArg::Int(input.nx as i64),
                                CArg::Int(input.ny as i64),
                                CArg::Int(second.nx as i64),
                                CArg::Int(second.ny as i64),
                            ],
                        )
                        .as_bytes(),
                    );
                }
            } else {
                super_fac = -second.nx / input.nx;
                let j = second.ny / input.ny;
                if j != -super_fac
                    || j * input.nx != second.nx
                    || j * input.ny != second.ny
                    || (j != 8 && j != 2 && j != 4)
                {
                    exit_error(
                        c_format(
                            "Image file size (%d x %d) must be exactly the 1/2, 1/4, or 1/8 times the gain reference size (%d x %d)",
                            &[
                                CArg::Int(input.nx as i64),
                                CArg::Int(input.ny as i64),
                                CArg::Int(second.nx as i64),
                                CArg::Int(second.ny as i64),
                            ],
                        )
                        .as_bytes(),
                    );
                }
            }
            // `clip.cpp:781` hands `viewcmd` to CorDefProcessFeiDefects as the
            // 1024-byte message buffer, with a 1000-byte usable length.
            let mut message = String::new();
            let ii_file = unsafe { ii_file.as_mut().unwrap() };
            if crate::imod::clip::correct_defects::cor_def_process_fei_defects(
                ii_file,
                &mut options.defects,
                second.nx,
                second.ny,
                true,
                super_fac,
                fei_def_pad,
                dump_defect_name.as_deref(),
                &mut message,
                1000,
            ) != 0
            {
                exit_error(message.as_bytes());
            }
            dump_defect_name = None;
            options.read_defects = 1;
            options.cam_size_x = input.nx;
            options.cam_size_y = input.ny;
        }
    }
    if procout && (!need_two || options.infiles > 1) {
        let last = raw[raw.len() - 1].clone();
        options.ofname = Some(last.clone());
        if matches!(process, IP_SUPERGAIN | IP_PLANARFIT) {
            crate::imod::libcfshr::b3dutil::imod_backup_file(&last);
            output.fp = ImodFile::open(&last, "w");
        } else if options.add2file != IP_APPEND_FALSE {
            output.fp = iimage::ii_fopen(last.as_bytes(), "rb+");
            if output.fp.is_none() {
                exit_error(c_format("Error finding %s", &[CArg::Str(&last)]).as_bytes());
            }
            let mut fp = output.fp.clone().unwrap();
            if mrcfiles::mrc_head_read(&mut fp, &mut output) != 0 {
                // `clip.cpp:812` reports argv[iarg], the next unconsumed
                // argument, not the output file name.
                exit_error(
                    c_format(
                        "Error reading %s",
                        &[CArg::Str(raw.get(iarg).map_or("", |v| v.as_str()))],
                    )
                    .as_bytes(),
                );
            }
        } else if process != IP_SPLITRGB {
            if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none() {
                crate::imod::libcfshr::b3dutil::imod_backup_file(&last);
            }
            if process == IP_FFT
                && input.mode != mrcfiles::MRC_MODE_COMPLEX_FLOAT
                && matches!(
                    crate::imod::libcfshr::b3dutil::b3d_output_file_type(),
                    iimage::IIFILE_TIFF | iimage::IIFILE_JPEG
                )
            {
                let _ = ImodFile::Stdout.write_all(
                    b"WARNING: clip - Writing an MRC file; TIFF or JPEG files cannot contain FFTs\n",
                );
                crate::imod::libcfshr::b3dutil::override_output_type(iimage::IIFILE_MRC);
            }
            output.fp = iimage::ii_fopen(last.as_bytes(), "wb+");
        }
        if output.fp.is_none() {
            exit_error(c_format("Error opening output file %s", &[CArg::Str(&last)]).as_bytes());
        }
    }
    if options.from_one != 0 {
        if matches!(options.add2file, IP_APPEND_OVERWRITE | IP_APPEND_TRUNCATE) {
            options.isec -= 1;
        }
        if options.nofsecs != IP_DEFAULT {
            for section in 0..options.nofsecs as usize {
                options.secs[section] -= 1;
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
        options.mode = mrcfiles::MRC_MODE_FLOAT;
    }
    // Every process routine is still an `unsafe fn`: they hand `Islice` and
    // `Istack` pointers to `libcfshr::islice`, which has not converted yet.
    let retval = unsafe {
        match process {
            IP_ADD | IP_AVERAGE | IP_VARIANCE | IP_STANDEV | IP_SUBTRACT => {
                processing::clip_average(&mut input, &mut second, &mut output, &mut options)
            }
            IP_MULTIPLY | IP_DIVIDE => {
                processing::clip_multdiv(&mut input, &mut second, &mut output, &mut options)
            }
            IP_UNPACK | IP_NORMALIZE => {
                processing::clip_unpack(&mut input, &mut second, &mut output, &mut options)
            }
            IP_DEFECTMAP => processing::clip_defect_map(&mut input, &mut output, &mut options),
            IP_SUPERGAIN => {
                // `clip.cpp:880` passes `hout.fp` by value; the `Rc` clone shares
                // the same open file, as the C's copied `FILE *` does.
                let mut fp = output.fp.clone().unwrap();
                processing::clip_super_gain(&mut input, &mut fp, &mut options)
            }
            IP_BLANKFILE => processing::clip_blank_file(&mut output, &mut options),
            IP_BRIGHTNESS | IP_CONTRAST | IP_SHADOW | IP_RESIZE | IP_THRESHOLD | IP_TRUNCATE
            | IP_UNWRAP | IP_LOGARITHM | IP_SQROOT | IP_INTEGRAL | IP_BOXSD => {
                processing::clip_scaling(&mut input, &mut output, &mut options)
            }
            IP_COLOR => processing::clip_color(&mut input, &mut output, &mut options),
            IP_QUADRANT => processing::clip_quadrant(&mut input, &mut output, &mut options),
            IP_PLANARFIT | IP_FLATFIELD => {
                processing::clip_planar_fit(&mut input, &mut output, &mut options)
            }
            IP_SPECTRUM => processing::clip_spectrum(&mut input, &mut output, &mut options),
            IP_FILLEDGE => {
                processing::fill_drift_corrected_edges(&mut input, &mut output, &mut options)
            }
            IP_CORRELATE => {
                correlation::grap_corr(&mut input, &mut second, &mut output, &mut options)
            }
            IP_DIFFUSION => processing::clip_diffusion(&mut input, &mut output, &mut options),
            IP_GRADIENT | IP_GRAHAM | IP_PREWITT | IP_SOBEL => {
                processing::clip_edge(&mut input, &mut output, &mut options)
            }
            IP_INFO => crate::imod::clip::file_io::mrc_head_print(&input),
            IP_FFT => fft::clip_fft(&mut input, &mut output, &mut options),
            IP_FILTER => filter::clip_bandpass_filter(&mut input, &mut output, &mut options),
            IP_FLIP => processing::clip_flip(&mut input, &mut output, &mut options),
            IP_HISTOGRAM => processing::clip_histogram(&mut input, &mut options),
            IP_JOINRGB => {
                processing::clip_joinrgb(&mut input, &mut second, &mut output, &mut options)
            }
            IP_LAPLACIAN | IP_SMOOTH | IP_SHARPEN => {
                processing::clip_convolve(&mut input, &mut output, &mut options)
            }
            IP_MEDIAN => processing::clip_median(&mut input, &mut output, &mut options, &[], 0, 0),
            IP_SPLITRGB => processing::clip_splitrgb(&mut input, &mut options),
            IP_STAT => processing::clip_stat(&mut input, &mut options),
            _ => {
                exit_error(b"No process selected.");
            }
        }
    };
    if retval != 0 {
        std::process::exit(retval);
    }
    if process != IP_BLANKFILE
        && let Some(fp) = input.fp.as_mut()
    {
        iimage::ii_close_tiff_copies_for_fp(fp);
        iimage::ii_fclose(fp);
    }
    if procout
        && process != IP_SPLITRGB
        && process != IP_SUPERGAIN
        && let Some(fp) = output.fp.as_mut()
    {
        iimage::ii_fclose(fp);
    }
    if view {
        // `clip.cpp:989-990`: sprintf into viewcmd, then system().
        let view_command = c_format("3dmod %s", &[CArg::Str(&raw[raw.len() - 1])]);
        // glibc's `system()` is `execl("/bin/sh", "sh", "-c", line, NULL)`:
        // the program is `/bin/sh` but `argv[0]` is `sh`, which is what the
        // shell puts in front of its own diagnostics.
        use std::os::unix::process::CommandExt as _;
        let mut shell = std::process::Command::new("/bin/sh");
        shell.arg0("sh").arg("-c").arg(&view_command);
        let _ = shell.status();
    }
    std::process::exit(0);
}
/// Original: `clipMakeSecList` (`clip.cpp:1002`).
///
/// Make a list of sections for use with the `-2d` and `-z` option; commands can
/// be like `-z 0,4-10` or `-z 2,3,4-8,12,19`.  The source deliberately reverses
/// a descending range before expanding it, so `4-2` becomes `2, 3, 4`.  The
/// `int *nofsecs` out-parameter stays, because the caller reads it as the list
/// length and later compares it against `IP_DEFAULT`.
pub fn clip_make_sec_list(clst: &str, nofsecs: &mut i32) -> Vec<i32> {
    let clst = clst.as_bytes();
    let len = clst.len();
    let mut secs: Vec<i32> = vec![atoi_bytes(clst)];
    *nofsecs = 1;
    let mut i = 0usize;
    while i < len {
        match clst[i] {
            b',' => {
                *nofsecs += 1;
                i += 1;
                secs.push(atoi_bytes(&clst[i.min(len)..]));
            }
            b'-' => {
                i += 1;
                let mut top = atoi_bytes(&clst[i.min(len)..]);
                let last = (*nofsecs - 1) as usize;
                if top < secs[last] {
                    let range = top;
                    top = secs[last];
                    secs[last] = range;
                }
                if top == secs[last] {
                    i += 1;
                    continue;
                }
                let range = top - secs[last];
                for r in 0..range {
                    secs.push(secs[last] + r + 1);
                }
                *nofsecs += range;
            }
            _ => {}
        }
        i += 1;
    }
    secs
}

/// The output targets the `sscanf` calls in `clip.cpp` pass.
///
/// C's `sscanf` is variadic and Rust is not, so the pointer arguments become a
/// slice.  Which arm a caller picks is decided by the source's conversion
/// specifier and the declared type of the object whose address it passes:
/// `%d` takes [`ScanArg::Int`], `%f`/`%g` take [`ScanArg::Flt`], and `%*c`
/// takes no argument at all.
pub enum ScanArg<'a> {
    Int(&'a mut i32),
    Flt(&'a mut f32),
}

/// The C library's `sscanf`, as a Rust function.
///
/// This is a boundary translation, like
/// [`crate::imod::libcfshr::b3dutil::c_format`] is of `printf`: `str::parse` is
/// **not** the same thing.  `sscanf` skips leading white space, accepts a
/// *partial* parse, stops at the first character that cannot continue the
/// current conversion, and leaves every later argument untouched when a
/// conversion fails — which is exactly what `clip.cpp` relies on, both for
/// `-x 100` (where `x2` keeps `IP_DEFAULT`) and for `-E`/`-F`/`-M`, whose
/// shared `itemp` carries its value from one option entry to the next.
///
/// Supported, because that is what `clip.cpp` uses: `%d`, `%f`, `%g` and the
/// assignment-suppressed `%*c`, plus literal and white-space directives.  The
/// return value is the number of fields assigned; the source discards it at all
/// 30 call sites, so the C `EOF` return for an input failure before the first
/// conversion is not distinguished from `0`.
pub fn sscanf(s: &str, fmt: &str, args: &mut [ScanArg]) -> i32 {
    /* glibc's `sscanf` is `vfscanf` over a string stream. */
    let mut pos = 0usize;
    fscanf(s.as_bytes(), &mut pos, fmt, args)
}

/// The C library's `fscanf`, as a Rust function.
///
/// The stream is the file's bytes plus a cursor, which is what a `FILE *`
/// reading a small text file is; `*pos` advances exactly as far as the scan
/// consumed, so consecutive calls continue where the previous one stopped —
/// `CorDefReadSuperGain` (`CorrectDefects.cpp:2168`) depends on that.
///
/// Returns the number of fields assigned, or `EOF` (-1) when the stream is
/// already exhausted before the first conversion, which is the distinction
/// that routine tests.
pub fn fscanf(sb: &[u8], pos: &mut usize, fmt: &str, args: &mut [ScanArg]) -> i32 {
    let fb = fmt.as_bytes();
    let mut si = *pos;
    let mut fi = 0usize;
    let mut ai = 0usize;
    let mut assigned = 0i32;
    /* C returns EOF only for an input failure -- an exhausted stream -- that
    happens before any conversion; a matching failure returns the count. */
    macro_rules! stop {
        ($eof:expr) => {{
            *pos = si;
            return if $eof && assigned == 0 { -1 } else { assigned };
        }};
    }
    while fi < fb.len() {
        let c = fb[fi];
        if c == b'%' {
            fi += 1;
            let suppress = fi < fb.len() && fb[fi] == b'*';
            if suppress {
                fi += 1;
            }
            if fi >= fb.len() {
                stop!(false);
            }
            let conv = fb[fi];
            fi += 1;
            match conv {
                /* %c reads exactly one character and does not skip white space. */
                b'c' => {
                    if si >= sb.len() {
                        stop!(true);
                    }
                    si += 1;
                }
                b'd' => {
                    while si < sb.len() && sb[si].is_ascii_whitespace() {
                        si += 1;
                    }
                    if si >= sb.len() {
                        stop!(true);
                    }
                    let mut end = 0usize;
                    let value = strtol(&sb[si..], &mut end, 10);
                    if end == 0 {
                        stop!(false);
                    }
                    si += end;
                    if !suppress {
                        match args.get_mut(ai) {
                            Some(ScanArg::Int(target)) => **target = value as i32,
                            Some(ScanArg::Flt(target)) => **target = value as f32,
                            None => stop!(false),
                        }
                        ai += 1;
                        assigned += 1;
                    }
                }
                b'f' | b'e' | b'E' | b'g' | b'G' | b'a' => {
                    while si < sb.len() && sb[si].is_ascii_whitespace() {
                        si += 1;
                    }
                    if si >= sb.len() {
                        stop!(true);
                    }
                    let mut end = 0usize;
                    let value = strtod(&sb[si..], &mut end);
                    if end == 0 {
                        stop!(false);
                    }
                    si += end;
                    if !suppress {
                        match args.get_mut(ai) {
                            Some(ScanArg::Flt(target)) => **target = value as f32,
                            Some(ScanArg::Int(target)) => **target = value as i32,
                            None => stop!(false),
                        }
                        ai += 1;
                        assigned += 1;
                    }
                }
                b'%' => {
                    if si >= sb.len() {
                        stop!(true);
                    }
                    if sb[si] != b'%' {
                        stop!(false);
                    }
                    si += 1;
                }
                _ => stop!(false),
            }
        } else if c.is_ascii_whitespace() {
            /* A white-space directive matches any amount of white space, none included. */
            while si < sb.len() && sb[si].is_ascii_whitespace() {
                si += 1;
            }
            fi += 1;
        } else {
            if si >= sb.len() {
                stop!(true);
            }
            if sb[si] != c {
                stop!(false);
            }
            si += 1;
            fi += 1;
        }
    }
    *pos = si;
    assigned
}

/// The C library's `atoi`, as a Rust function.
///
/// `atoi(s)` is `(int)strtol(s, NULL, 10)`: leading white space, an optional
/// sign, then digits, stopping at the first character that is not one.  It
/// yields 0 rather than failing, which is what `clip.cpp:1012`'s section list
/// depends on.
pub fn atoi(s: &str) -> i32 {
    atoi_bytes(s.as_bytes())
}

/// `atoi` over the bytes of a string, for the mid-string calls
/// `clipMakeSecList` makes with `&clst[++i]`.
pub fn atoi_bytes(s: &[u8]) -> i32 {
    let mut end = 0usize;
    strtol(s, &mut end, 10) as i32
}

/// The C library's `atof`, as a Rust function: `strtod(s, NULL)`.
pub fn atof(s: &str) -> f64 {
    let mut end = 0usize;
    strtod(s.as_bytes(), &mut end)
}

/// The C library's `strtol` with the given base, as a Rust function.
///
/// `*end` is returned as an index into `s`, and is 0 when no conversion was
/// performed, as C leaves `endptr` at `nptr`.  The same routine is translated
/// in `libcfshr/parse_params.rs`, which is private to that module; the two
/// merge when the tree grows one home for the C library boundary.
fn strtol(s: &[u8], end: &mut usize, base: i32) -> i64 {
    let mut i = 0usize;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    let mut negative = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        negative = s[i] == b'-';
        i += 1;
    }
    let mut base = base;
    if (base == 0 || base == 16)
        && i + 1 < s.len()
        && s[i] == b'0'
        && (s[i + 1] | 32) == b'x'
        && i + 2 < s.len()
        && (s[i + 2] as char).is_digit(16)
    {
        i += 2;
        base = 16;
    } else if base == 0 {
        base = if i < s.len() && s[i] == b'0' { 8 } else { 10 };
    }
    let digits_start = i;
    let mut value: i64 = 0;
    let mut overflow = false;
    while i < s.len() {
        let d = match (s[i] as char).to_digit(base as u32) {
            Some(d) => d as i64,
            None => break,
        };
        if !overflow {
            match value
                .checked_mul(base as i64)
                .and_then(|v| v.checked_add(d))
            {
                Some(v) => value = v,
                None => overflow = true,
            }
        }
        i += 1;
    }
    if i == digits_start {
        *end = 0;
        return 0;
    }
    *end = i;
    if overflow {
        return if negative { i64::MIN } else { i64::MAX };
    }
    if negative { -value } else { value }
}

/// The C library's `strtod`, as a Rust function.
///
/// Accepts what glibc accepts — leading white space, a sign, a decimal or C99
/// hexadecimal significand with an optional exponent, and `inf`/`infinity`/
/// `nan` — and reports in `*end` the index in `s` where the scan stopped, which
/// is 0 when no conversion was performed.
fn strtod(s: &[u8], end: &mut usize) -> f64 {
    let mut i = 0usize;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    let mut negative = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        negative = s[i] == b'-';
        i += 1;
    }
    let rest = &s[i..];
    let lower = |b: u8| b | 32;
    /* inf / infinity */
    if rest.len() >= 3 && lower(rest[0]) == b'i' && lower(rest[1]) == b'n' && lower(rest[2]) == b'f'
    {
        i += 3;
        if rest.len() >= 8
            && lower(rest[3]) == b'i'
            && lower(rest[4]) == b'n'
            && lower(rest[5]) == b'i'
            && lower(rest[6]) == b't'
            && lower(rest[7]) == b'y'
        {
            i += 5;
        }
        *end = i;
        return if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    /* nan, with an optional parenthesised character sequence */
    if rest.len() >= 3 && lower(rest[0]) == b'n' && lower(rest[1]) == b'a' && lower(rest[2]) == b'n'
    {
        i += 3;
        if i < s.len() && s[i] == b'(' {
            let mut j = i + 1;
            while j < s.len() && (s[j].is_ascii_alphanumeric() || s[j] == b'_') {
                j += 1;
            }
            if j < s.len() && s[j] == b')' {
                i = j + 1;
            }
        }
        *end = i;
        return if negative { -f64::NAN } else { f64::NAN };
    }
    /* Hexadecimal significand */
    if i + 1 < s.len() && s[i] == b'0' && lower(s[i + 1]) == b'x' {
        let mut j = i + 2;
        let mut digits = 0usize;
        while j < s.len() && (s[j] as char).is_ascii_hexdigit() {
            j += 1;
            digits += 1;
        }
        if j < s.len() && s[j] == b'.' {
            j += 1;
            while j < s.len() && (s[j] as char).is_ascii_hexdigit() {
                j += 1;
                digits += 1;
            }
        }
        if digits > 0 {
            let mut k = j;
            if k < s.len() && lower(s[k]) == b'p' {
                let mut m = k + 1;
                if m < s.len() && (s[m] == b'+' || s[m] == b'-') {
                    m += 1;
                }
                if m < s.len() && s[m].is_ascii_digit() {
                    while m < s.len() && s[m].is_ascii_digit() {
                        m += 1;
                    }
                    k = m;
                }
            }
            let text = core::str::from_utf8(&s[start..k]).unwrap_or("");
            /* Rust parses "0x1p3" only through a manual expansion. */
            let body = &s[i + 2..j];
            let mut mantissa = 0.0f64;
            let mut exponent = 0i32;
            let mut seen_point = false;
            for &b in body {
                if b == b'.' {
                    seen_point = true;
                    continue;
                }
                mantissa = mantissa * 16.0 + (b as char).to_digit(16).unwrap_or(0) as f64;
                if seen_point {
                    exponent -= 4;
                }
            }
            if k > j {
                let expo: i32 = core::str::from_utf8(&s[j + 1..k])
                    .unwrap_or("0")
                    .trim_start_matches('+')
                    .parse()
                    .unwrap_or(0);
                exponent += expo;
            }
            let _ = text;
            *end = k;
            let value = mantissa * (2.0f64).powi(exponent);
            return if negative { -value } else { value };
        }
    }
    /* Decimal */
    let mut j = i;
    let mut digits = 0usize;
    while j < s.len() && s[j].is_ascii_digit() {
        j += 1;
        digits += 1;
    }
    if j < s.len() && s[j] == b'.' {
        j += 1;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        *end = 0;
        return 0.0;
    }
    let mut k = j;
    if k < s.len() && lower(s[k]) == b'e' {
        let mut m = k + 1;
        if m < s.len() && (s[m] == b'+' || s[m] == b'-') {
            m += 1;
        }
        if m < s.len() && s[m].is_ascii_digit() {
            while m < s.len() && s[m].is_ascii_digit() {
                m += 1;
            }
            k = m;
        }
    }
    *end = k;
    core::str::from_utf8(&s[start..k])
        .ok()
        .and_then(|t| t.parse::<f64>().ok())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::{ScanArg, clip_make_sec_list, sscanf};

    #[test]
    fn clip_make_sec_list_expands_source_syntax() {
        let mut n = 0;
        assert_eq!(
            clip_make_sec_list("0,4-10", &mut n),
            vec![0, 4, 5, 6, 7, 8, 9, 10]
        );
        assert_eq!(n, 8);
        assert_eq!(clip_make_sec_list("4-2", &mut n), vec![2, 3, 4]);
        assert_eq!(n, 3);
    }

    #[test]
    fn sscanf_leaves_unmatched_fields_alone() {
        /* `-x 100` fills only the first field, as the source relies on. */
        let (mut x, mut x2) = (-99999, -99999);
        assert_eq!(
            sscanf(
                "100",
                "%d%*c%d",
                &mut [ScanArg::Int(&mut x), ScanArg::Int(&mut x2)]
            ),
            1
        );
        assert_eq!((x, x2), (100, -99999));
        /* The separator may be a comma or an `x`. */
        assert_eq!(
            sscanf(
                "10x20",
                "%d%*c%d",
                &mut [ScanArg::Int(&mut x), ScanArg::Int(&mut x2)]
            ),
            2
        );
        assert_eq!((x, x2), (10, 20));
        assert_eq!(
            sscanf(
                "3,4",
                "%d%*c%d",
                &mut [ScanArg::Int(&mut x), ScanArg::Int(&mut x2)]
            ),
            2
        );
        assert_eq!((x, x2), (3, 4));
        /* A partial float parse is accepted. */
        let mut f = 0.0f32;
        assert_eq!(sscanf("2.5abc", "%f", &mut [ScanArg::Flt(&mut f)]), 1);
        assert_eq!(f, 2.5);
        /* Nothing numeric assigns nothing. */
        let mut g = 7.0f32;
        assert_eq!(sscanf("abc", "%f", &mut [ScanArg::Flt(&mut g)]), 0);
        assert_eq!(g, 7.0);
    }
}
