//! Translation of `IMOD/mrc/tif2mrc.c`.
//!
//! The old `b3dtiff.h` reader remains the default C-ABI path.  A deliberately
//! opt-in Rust decoder may be selected for its currently supported TIFF cases;
//! palette and TVIPS behavior continue to require the parity reader.
#![allow(unused_variables, unused_assignments, unused_mut)]

use crate::imod::clip::clip::{ScanArg, atof, sscanf};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, SEEK_SET, b3d_fwrite, c_format_bytes, imod_backup_file, imod_prog_name,
    mrc_big_seek, override_write_bytes, replace_file_arg_vec,
};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
use crate::imod::libiimod::iimage::ImodImageFile;
use crate::imod::libiimod::iitif::{tiff_filter_warnings, tiff_set_mapping};
use crate::imod::libiimod::mrcfiles::{
    MRC_LABEL_SIZE, MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
    MRC_NLABELS, MrcHeader, mrc_head_label, mrc_head_new, mrc_head_write,
};
pub use crate::imod::mrc::tiff::{
    TfEntry, TfHeader, TfInfo, read_tiffentries, read_tiffheader, tiff_close_file, tiff_first_ifd,
    tiff_ifd_number, tiff_open_file, tiff_read_file, tiff_read_section,
};
use std::io::Write;

/// `b3dtiff.h:68`.
const WIDTHINDEX: usize = 1;
/// `b3dtiff.h:69`.
const LENGTHINDEX: usize = 2;

const IITYPE_INT: i32 = 4;
const IITYPE_UINT: i32 = 5;
const IITYPE_USHORT: i32 = 3;
const IIFLAG_TVIPS_DATA: u32 = 2;
const IIFLAG_BYTES_SWAPPED: u32 = 4;

/// Original `usage` (`tif2mrc.c:32`).
fn usage(progname: &[u8]) -> ! {
    // `tif2mrc.c:33`: `VERSION_NAME`, `__DATE__` and `__TIME__`.  Every line
    // goes through `printf`, so they share the C stdout with `imodCopyright`.
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(&c_format_bytes(
        "Tif2mrc Version %s %s %s\n",
        &[
            CArg::Str("5.2.17"),
            CArg::Str(crate::imod::libcfshr::b3dutil::IMOD_BUILD_DATE),
            CArg::Str(crate::imod::libcfshr::b3dutil::IMOD_BUILD_TIME),
        ],
    ));
    crate::imod::libcfshr::b3dutil::imod_copyright();
    let _ = out.write_all(&c_format_bytes(
        "Usage: %s [options] <tiff files...> <mrcfile>\n",
        &[CArg::Bytes(progname)],
    ));
    let _ = out.write_all(b"Options:\n");
    let _ = out.write_all(b"\t-g      Convert 24-bit RGB to 8-bit grayscale\n");
    let _ = out.write_all(b"\t-G      Convert 24-bit RGB to 8-bit grayscale with NTSC scaling\n");
    let _ = out.write_all(b"\t-u      Convert unsigned 16-bit values by subtracting 32768\n");
    let _ = out.write_all(b"\t-d      Convert unsigned 16-bit values by dividing by 2\n");
    let _ =
        out.write_all(b"\t-k      Keep unsigned 16-bit values; store in unsigned integer mode\n");
    let _ = out.write_all(b"\t-s      Store 16-bit as signed (mode 1) even if data are unsigned\n");
    let _ = out.write_all(b"\t-B #    Write bytes as unsigned (for # 0) or signed (for # 1)\n");
    let _ = out.write_all(b"\t-i      Invert order of sections in output stack\n");
    let _ = out.write_all(b"\t-p  #   Set pixel spacing in MRC header to given #\n");
    let _ =
        out.write_all(b"\t-P      Set pixel spacing in MRC header from resolution in TIFF file\n");
    let _ = out.write_all(b"\t-T file Output tilt angles from TVIPS input files to given file\n");
    let _ = out.write_all(b"\t-f      Read only first image of multi-page file\n");
    let _ = out.write_all(b"\t-o x,y  Set output file size in X and Y\n");
    let _ = out.write_all(b"\t-F  #   Set value to fill areas with no image data to given #\n");
    let _ = out.write_all(b"\t-b file Background subtract image in given file\n");
    let _ = out.write_all(b"\t-t #    Set criterion in megabytes for reading files in chunks\n");
    let _ = out.write_all(b"\t-m      Turn off file-to-memory mapping in libtiff\n");
    let _ = out.flush();
    std::process::exit(3)
}

/// Original `manageMode` (`tif2mrc.c:634`).
fn manage_mode(
    tiff: &TfInfo,
    keep_ushort: i32,
    force_signed: i32,
    makegray: i32,
    pix_size: &mut i32,
    mode: &mut i32,
) {
    *pix_size = 1;
    if tiff.bits_per_sample == 16 {
        *mode = MRC_MODE_SHORT;
        *pix_size = 2;
    }
    if tiff.bits_per_sample == 32 {
        *mode = MRC_MODE_FLOAT;
        *pix_size = 4;
    }
    if *mode == MRC_MODE_SHORT
        && (keep_ushort != 0
            || (force_signed == 0
                && tiff
                    .iifile
                    .as_ref()
                    .is_some_and(|file| file.type_ == IITYPE_USHORT)))
    {
        *mode = MRC_MODE_USHORT;
    }
    if tiff.photometric_interpretation / 2 == 1 && makegray == 0 {
        *mode = MRC_MODE_RGB;
        *pix_size = 3;
    }
}

/// Original `convertrgb` (`tif2mrc.c:661`).
fn convertrgb(tifdata: &mut [u8], xsize: i32, ysize: i32, ntsc: i32) {
    {
        let mut pixel: i32;
        let mut fpixel: f32;
        // The C walks one read and one write cursor over the same buffer, the
        // write cursor always three bytes behind; index arithmetic is the same
        // aliasing, with no pointer.
        let mut input = 0usize;
        let mut output = 0usize;
        let xysize = xsize as usize * ysize as usize;
        if ntsc != 0 {
            for _ in 0..xysize {
                // `tif2mrc.c:672-674` accumulates through a `float fpixel`
                // while each term is computed in double (the weights are
                // double constants), so the running sum is rounded back to
                // float after every term.  Keeping this as one f32 expression
                // shifts occasional pixels by one count.
                fpixel = (tifdata[input] as f64 * 0.3) as f32;
                input += 1;
                fpixel = (fpixel as f64 + tifdata[input] as f64 * 0.59) as f32;
                input += 1;
                fpixel = (fpixel as f64 + tifdata[input] as f64 * 0.11) as f32;
                input += 1;
                tifdata[output] = (fpixel + 0.5f32) as i32 as u8;
                output += 1;
            }
        } else {
            for _ in 0..xysize {
                pixel = tifdata[input] as i32;
                input += 1;
                pixel += tifdata[input] as i32;
                input += 1;
                pixel += tifdata[input] as i32;
                input += 1;
                tifdata[output] = (pixel / 3) as u8;
                output += 1;
            }
        }
    }
}

/// Original `expandIndexToRGB` (`tif2mrc.c:689`).
fn expand_index_to_rgb(datap: &mut Vec<u8>, iifile: Option<&ImodImageFile>, section: i32) {
    let Some(iifile) = iifile else {
        exit_error(b"Colormap data not read in properly.");
    };
    let Some(colormap) = iifile.colormap.as_deref() else {
        // `tif2mrc.c:696`: no trailing newline -- `exitError` supplies one.
        exit_error(b"Colormap data not read in properly.");
    };
    let mut size = iifile.nx as usize * iifile.ny as usize;
    if iifile.ury >= 0 {
        size = iifile.nx as usize * (iifile.ury + 1 - iifile.lly) as usize;
    }
    // `tif2mrc.c:702`: the C `malloc`s the expansion, swaps it into
    // `*datap` and *leaks* the index image it replaced.  Owning both as
    // `Vec`s releases the old one instead; nothing observable changes.
    let mut out = vec![0_u8; 3 * size];
    let input = core::mem::take(datap);
    let start = 768 * section as usize;
    let map = &colormap[start..start + 768];
    for i in 0..size {
        let ind = input[i] as usize;
        out[3 * i] = map[ind];
        out[3 * i + 1] = map[256 + ind];
        out[3 * i + 2] = map[512 + ind];
    }
    *datap = out;
}

/// Original `convertLongToFloat` (`tif2mrc.c:716`).
fn convert_long_to_float(tifdata: &mut [u8], iifile: Option<&ImodImageFile>) {
    let Some(iifile) = iifile else {
        return;
    };
    if iifile.type_ != IITYPE_UINT && iifile.type_ != IITYPE_INT {
        return;
    }
    let mut size = iifile.nx as usize * iifile.ny as usize;
    if iifile.ury >= 0 {
        size = iifile.nx as usize * (iifile.ury + 1 - iifile.lly) as usize;
    }
    for i in 0..size {
        let offset = 4 * i;
        let value = if iifile.type_ == IITYPE_UINT {
            u32::from_ne_bytes(tifdata[offset..offset + 4].try_into().unwrap()) as f32
        } else {
            i32::from_ne_bytes(tifdata[offset..offset + 4].try_into().unwrap()) as f32
        };
        tifdata[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
    }
}

/// Original `minmaxmean` (`tif2mrc.c:738`).
fn minmaxmean(
    tifdata: &mut [u8],
    mode: i32,
    unsign: i32,
    divide: i32,
    xsize: i32,
    ysize: i32,
    min: &mut f32,
    max: &mut f32,
) -> f32 {
    let size = xsize as usize * ysize as usize;
    let mut mean = 0.0_f64;
    if !matches!(
        mode,
        MRC_MODE_BYTE | MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_FLOAT
    ) {
        return 0.0;
    }
    if mode == MRC_MODE_SHORT && unsign != 0 {
        for i in 0..size {
            let offset = 2 * i;
            let value = u16::from_ne_bytes(tifdata[offset..offset + 2].try_into().unwrap());
            let converted = if divide != 0 {
                (value / 2) as i16
            } else {
                (value as i32 - 32768) as i16
            };
            tifdata[offset..offset + 2].copy_from_slice(&converted.to_ne_bytes());
        }
    }
    for i in 0..size {
        let value = match mode {
            MRC_MODE_BYTE => tifdata[i] as f32,
            MRC_MODE_SHORT => {
                let offset = 2 * i;
                i16::from_ne_bytes(tifdata[offset..offset + 2].try_into().unwrap()) as f32
            }
            MRC_MODE_USHORT => {
                let offset = 2 * i;
                u16::from_ne_bytes(tifdata[offset..offset + 2].try_into().unwrap()) as f32
            }
            // C assigns this float through its `int pixel` local before
            // updating statistics, so preserve its truncating conversion.
            MRC_MODE_FLOAT => {
                let offset = 4 * i;
                f32::from_ne_bytes(tifdata[offset..offset + 4].try_into().unwrap()) as i32 as f32
            }
            _ => unreachable!(),
        };
        if value < *min {
            *min = value;
        }
        if value > *max {
            *max = value;
        }
        mean += value as f64;
    }
    (mean / size as f64) as f32
}

/// Metadata that is compared across the TIFF inputs handled by one run.
///
/// The C program kept this as function-local static state.  It belongs to the
/// `tif2mrc` invocation instead: each input can refine the eventual shared
/// label, while separate conversions start with no carry-over.
struct TvipsMetadataState {
    last_axis: f32,
    last_spot: i32,
    last_binning: i32,
    same_spot: i32,
    same_bin: i32,
    same_axis: i32,
}

impl Default for TvipsMetadataState {
    fn default() -> Self {
        Self {
            last_axis: 0.0,
            last_spot: 0,
            last_binning: 0,
            same_spot: -1,
            same_bin: -1,
            same_axis: -1,
        }
    }
}

/// Original `manageTVIPSdata` (`tif2mrc.c:813`).
fn manage_tvipsdata(
    state: &mut TvipsMetadataState,
    iifile: Option<&ImodImageFile>,
    label: &mut [u8; MRC_LABEL_SIZE],
    tilt_angle: &mut f32,
) -> i32 {
    let Some(iifile) = iifile else {
        return 1;
    };
    if iifile.user_data.is_null()
        || iifile.user_count < 4260
        || iifile.user_flags & IIFLAG_TVIPS_DATA == 0
    {
        return 1;
    }
    // TIFF's callback-owned metadata cursor is the immediate ABI boundary;
    // all subsequent field extraction is bounds-checked native byte storage.
    let data = unsafe { core::slice::from_raw_parts(iifile.user_data, iifile.user_count as usize) };
    let mut axis = f32::from_ne_bytes(data[3704..3708].try_into().unwrap());
    *tilt_angle = f32::from_ne_bytes(data[3564..3568].try_into().unwrap());
    let mut spot_float = f32::from_ne_bytes(data[3624..3628].try_into().unwrap());
    let mut binning = i32::from_ne_bytes(data[3944..3948].try_into().unwrap());
    if iifile.user_flags & IIFLAG_BYTES_SWAPPED != 0 {
        axis = f32::from_bits(axis.to_bits().swap_bytes());
        *tilt_angle = f32::from_bits((*tilt_angle).to_bits().swap_bytes());
        spot_float = f32::from_bits(spot_float.to_bits().swap_bytes());
        binning = binning.swap_bytes();
    }
    let spot = spot_float.round() as i32;
    if spot < 1 {
        state.same_spot = 0;
    }
    if binning < 1 {
        state.same_bin = 0;
    }
    if state.same_spot < 0 {
        state.same_spot = 1;
    } else if state.same_spot > 0 && spot != state.last_spot {
        state.same_spot = 0;
    }
    if state.same_bin < 0 {
        state.same_bin = 1;
    } else if state.same_bin > 0 && binning != state.last_binning {
        state.same_bin = 0;
    }
    if state.same_axis < 0 {
        state.same_axis = 1;
    } else if state.same_axis > 0 && (axis - state.last_axis).abs() > 1.0e-5 {
        state.same_axis = 0;
    }
    state.last_axis = axis;
    state.last_spot = spot;
    state.last_binning = binning;
    axis -= 90.0;
    if axis < -180.0 {
        axis += 360.0;
    }
    if axis > 180.0 {
        axis -= 360.0;
    }
    // `tif2mrc.c:874-884`: three `snprintf`s into `label` with a size of
    // MRC_LABEL_SIZE, so at most MRC_LABEL_SIZE - 1 characters land and
    // the byte after them is the terminating NUL.
    let text = if state.same_axis > 0 && state.same_spot > 0 && state.same_bin > 0 {
        c_format_bytes(
            "    Tilt axis angle = %.1f, binning = %d  spot = %d",
            &[
                CArg::Dbl(axis as f64),
                CArg::Int(binning as i64),
                CArg::Int(spot as i64),
            ],
        )
    } else if state.same_axis > 0 && state.same_bin > 0 {
        c_format_bytes(
            "    Tilt axis angle = %.1f, binning = %d",
            &[CArg::Dbl(axis as f64), CArg::Int(binning as i64)],
        )
    } else if state.same_axis > 0 {
        c_format_bytes("    Tilt axis angle = %.1f", &[CArg::Dbl(axis as f64)])
    } else {
        Vec::new()
    };
    label.fill(0);
    let kept = text.len().min(MRC_LABEL_SIZE - 1);
    label[..kept].copy_from_slice(&text[..kept]);
    0
}

/// Original `main` (`tif2mrc.c:58`).
///
/// `argc`/`argv` become the argument vector itself.  `replaceFileArgVec`
/// (`b3dutil.c:2000`) can replace it, which is why it stays a mutable local
/// rather than the caller's slice.
pub fn tif2mrc(arguments: &[String]) -> i32 {
    let mut bgfp: crate::imod::libcfshr::b3dutil::ImodFile;
    let mut tiffp: crate::imod::libcfshr::b3dutil::ImodFile;
    let mut mrcfp: Option<crate::imod::libcfshr::b3dutil::ImodFile>;

    let mut tiff = TfInfo::default();
    let mut hdata = MrcHeader::default();

    let mut mode = 0_i32;
    let mut pix_size = 0_i32;
    let mut bgdata: Vec<u8> = Vec::new();
    let mut tifdata: Vec<u8>;
    let mut min: f32;
    let mut max: f32;
    let mut iarg: i32;
    let mut k: i32;
    let mut tmpdata: i32;
    let mut bg = 0_i32;
    let mut makegray = 0_i32;
    let mut fill_entered = 0_i32;
    let mut unsign = 0_i32;
    let mut divide = 0_i32;
    let mut keep_ushort = 0_i32;
    let mut force_signed = 0_i32;
    let mut read_first = 0_i32;
    let mut use_ntsc = 0_i32;
    let mut any_tif_pixel = 0_i32;
    let mut pixel_entered = 0_i32;
    let mut invert_stack = 0_i32;
    let mut xsize: i32;
    let mut ysize: i32;
    let mut iread: i32;
    let mut mrcxsize = 0_i32;
    let mut mrcysize = 0_i32;
    let mut user_fill = 0.0_f32;
    let mut fill_val = 0.0_f32;
    let mut mean: f32;
    let mut tmean: f32;
    let mut tilt_angle = 0.0_f32;
    let mut pixel_size = 1.0_f32;
    let mut y_pixel_size = 1.0_f32;
    let mut chunk_criterion = 100.0_f32;
    let mut bg_bits = 0_i32;
    let mut bgxsize = 0_i32;
    let mut bgysize = 0_i32;
    let mut xoffset: i32;
    let mut yoffset: i32;
    let mut xdo: usize;
    let mut ydo: i32;
    let first_file_ind: i32;
    let mut do_chunks: i32;
    let mut num_chunks: i32;
    let mut lines_per_chunk: i32;
    let mut nlines: i32;
    let mut lines_done: i32;
    // `tif2mrc.c:93-96`: the fill value is written through a `char *` that
    // points at whichever of these matches the output mode.
    let mut byte_fill = [0_u8; 3];
    let mut short_fill: i16 = 0;
    let mut ushort_fill: u16 = 0;
    let mut label = [0_u8; MRC_LABEL_SIZE];
    let mut tvips_metadata = TvipsMetadataState::default();
    let mut tilt_file: Option<String> = None;
    let mut tiltfp: Option<crate::imod::libcfshr::b3dutil::ImodFile> = None;
    let openmode = "rb";
    let mut bgfile: Option<String> = None;
    let mut argv: Vec<String> = arguments.to_vec();
    let mut argc = argv.len() as i32;
    let progname = imod_prog_name(argv.first().map(String::as_str).unwrap_or("tif2mrc"));
    let progname = progname.as_bytes();
    // `tif2mrc.c:105-106`.
    setExitPrefix(&c_format_bytes("\nERROR: %s - ", &[CArg::Bytes(progname)]));
    let mut out = ImodFile::Stdout;

    xsize = 0;
    ysize = 0;
    mean = 0.;
    min = 100000.;
    max = -100000.;
    label[0] = 0x00;

    if argc < 3 {
        usage(progname);
    }

    iarg = 1;
    while iarg < argc - 1 {
        let arg = argv[iarg as usize].as_bytes();
        if arg.first().copied() == Some(b'-') {
            match arg.get(1).copied().unwrap_or(0) {
                /* help */
                b'h' => usage(progname),

                /* convert rgb to gray scale */
                b'g' => makegray = 1,

                /* convert rgb to gray scale */
                b'G' => {
                    makegray = 1;
                    use_ntsc = 1;
                }

                /* treat ints as unsigned */
                b'u' => unsign = 1,

                /* treat ints as unsigned and divide by 2*/
                b'd' => divide = 1,

                /* save as unsigned */
                b'k' => keep_ushort = 1,

                /* save unsigned as signed */
                b's' => force_signed = 1,

                /* Control signed nature of byte output */
                b'B' => {
                    iarg += 1;
                    override_write_bytes(if atof(&argv[iarg as usize]) != 0. {
                        1
                    } else {
                        0
                    });
                }

                /* Invert output stack */
                b'i' => invert_stack = 1,

                /* Insert pixel size in header */
                b'p' => {
                    iarg += 1;
                    pixel_size = atof(&argv[iarg as usize]) as f32;
                    if pixel_size <= 0. {
                        pixel_size = 1.;
                    } else {
                        pixel_entered = 1;
                    }
                    y_pixel_size = pixel_size;
                }

                /* Use resolution from TIFF file regardless of value */
                b'P' => any_tif_pixel = 1,

                /* read only first image */
                b'f' => read_first = 1,

                /* Define fill value */
                b'F' => {
                    iarg += 1;
                    user_fill = atof(&argv[iarg as usize]) as f32;
                    fill_entered = 1;
                }

                b'b' => {
                    iarg += 1;
                    bgfile = Some(argv[iarg as usize].clone());
                    bg = 1;
                }

                /* Set output size */
                b'o' => {
                    iarg += 1;
                    sscanf(
                        &argv[iarg as usize],
                        "%d%*c%d",
                        &mut [ScanArg::Int(&mut mrcxsize), ScanArg::Int(&mut mrcysize)],
                    );
                }

                b'm' => tiff_set_mapping(0),

                b't' => {
                    iarg += 1;
                    chunk_criterion = atof(&argv[iarg as usize]) as f32;
                }

                b'T' => {
                    iarg += 1;
                    tilt_file = Some(argv[iarg as usize].clone());
                }

                _ => {}
            }
        } else {
            break;
        }
        iarg += 1;
    }

    if (argc - 1) < (iarg + 1) {
        exit_error(b"Argument error: no output file specified");
    }
    {
        let mut replaced = 0_i32;
        let mut vector: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
        let failed = replace_file_arg_vec(&mut vector, &mut argc, &mut iarg, &mut replaced);
        if failed != 0 {
            let _ = out.flush();
            std::process::exit(1);
        }
        argv = vector
            .iter()
            .take(argc as usize)
            .map(|a| String::from_utf8_lossy(a).into_owned())
            .collect();
    }

    if divide + unsign + keep_ushort + force_signed > 1 {
        exit_error(b"You must select only one of -u, -d, -k, or -s.");
    }
    if divide != 0 {
        unsign = 1;
    }
    if unsign != 0 {
        force_signed = 1;
    }
    tiff_filter_warnings();
    chunk_criterion *= 1024. * 1024.;
    if pixel_entered != 0 && any_tif_pixel != 0 {
        exit_error(b"You cannot enter both -p and -P");
    }
    if let Some(name) = tilt_file.as_deref() {
        imod_backup_file(name);
        tiltfp = crate::imod::libcfshr::b3dutil::ImodFile::open(name, "w");
        if tiltfp.is_none() {
            exit_error(&c_format_bytes(
                "Opening tilt angle file %s",
                &[CArg::Bytes(name.as_bytes())],
            ));
        }
    }

    if iarg == (argc - 2) && read_first == 0 {
        /* check for multi-paged tiff file. */
        /* Open the TIFF file. */
        let tiff_pages: i32;

        if tiff_open_file(
            argv[iarg as usize].as_bytes(),
            openmode,
            &mut tiff,
            any_tif_pixel,
        ) != 0
        {
            exit_error(&c_format_bytes(
                "Couldn't open %s.",
                &[CArg::Bytes(argv[iarg as usize].as_bytes())],
            ));
        }

        tiffp = tiff.fp.clone().unwrap();
        if let Some(iifile) = tiff.iifile.as_ref() {
            tiff_pages = iifile.nz;
        } else {
            tiff_pages = tiff_ifd_number(&mut tiffp);
        }
        if tiff_pages > 1 {
            let _ = out.write_all(b"Reading multi-paged TIFF file.\n");

            if bg != 0 {
                exit_error(b"Background subtraction not supported for multi-paged images.");
            }

            if mrcxsize != 0 {
                let _ = out
                    .write_all(b"Warning: output file size option ignored for multi-paged file\n");
            }

            if tiff.iifile.is_none() {
                read_tiffheader(&mut tiffp, &mut tiff.header);
                crate::imod::libcfshr::b3dutil::b3d_rewind(&mut tiffp);
                let mut byteorder = [0_u8; 2];
                crate::imod::libcfshr::b3dutil::b3d_fread(&mut byteorder, 2, 1, &mut tiffp);
                tiff.header.byteorder = i16::from_ne_bytes(byteorder);

                tiff.header.first_ifd_offset = tiff_first_ifd(&mut tiffp) as i32;
                crate::imod::libcfshr::b3dutil::b3d_rewind(&mut tiffp);
                read_tiffentries(&mut tiffp, &mut tiff);
            }

            if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none()
                && imod_backup_file(&argv[(argc - 1) as usize]) != 0
            {
                exit_error(b"Couldn't create backup file");
            }
            mrcfp =
                crate::imod::libcfshr::b3dutil::ImodFile::open(&argv[(argc - 1) as usize], "wb");
            if mrcfp.is_none() {
                // `perror("tif2mrc")` writes `"tif2mrc: <strerror>\n"`.
                // Rust's `io::Error` Display appends ` (os error N)`,
                // which the C library does not.
                let message = std::io::Error::last_os_error().to_string();
                let message = message
                    .split(" (os error ")
                    .next()
                    .unwrap_or(message.as_str());
                let _ = ImodFile::Stderr
                    .write_all(&c_format_bytes("tif2mrc: %s\n", &[CArg::Str(message)]));
                exit_error(&c_format_bytes(
                    "Opening %s\n",
                    &[CArg::Bytes(argv[(argc - 1) as usize].as_bytes())],
                ));
            }

            xsize = tiff.directory[WIDTHINDEX].value;
            ysize = tiff.directory[LENGTHINDEX].value;
            mrc_head_new(&mut hdata, xsize, ysize, tiff_pages, mode);
            mrc_head_write(mrcfp.as_mut().unwrap(), &mut hdata);
            manage_mode(
                &tiff,
                keep_ushort,
                force_signed,
                makegray,
                &mut pix_size,
                &mut mode,
            );

            let _ = out.write_all(&c_format_bytes(
                "Converting %d images size %d x %d\n",
                &[
                    CArg::Int(tiff_pages as i64),
                    CArg::Int(xsize as i64),
                    CArg::Int(ysize as i64),
                ],
            ));

            for section in 0..tiff_pages {
                let in_section = if invert_stack != 0 {
                    tiff_pages - 1 - section
                } else {
                    section
                };

                // `tif2mrc.c:293` and `:316`: the buffer is `malloc`ed by
                // `tiff_read_section` and freed at the end of the loop body,
                // so only one section buffer is alive at a time.
                let mut tifdata = match tiff_read_section(&mut tiffp, &mut tiff, in_section) {
                    Some(data) => data,
                    None => exit_error(&c_format_bytes(
                        "Failed to get image data for section %d",
                        &[CArg::Int(in_section as i64)],
                    )),
                };

                if tiff.photometric_interpretation == 3 {
                    expand_index_to_rgb(&mut tifdata, tiff.iifile.as_ref(), in_section);
                }

                /* convert RGB to gray scale */
                if tiff.photometric_interpretation / 2 == 1 && makegray != 0 {
                    convertrgb(&mut tifdata, xsize, ysize, use_ntsc);
                }

                /* Convert long ints to floats */
                convert_long_to_float(&mut tifdata, tiff.iifile.as_ref());

                mean += minmaxmean(
                    &mut tifdata,
                    mode,
                    unsign,
                    divide,
                    xsize,
                    ysize,
                    &mut min,
                    &mut max,
                );

                mrc_big_seek(
                    mrcfp.as_mut().unwrap(),
                    1024,
                    section * xsize,
                    ysize * pix_size,
                    SEEK_SET,
                );

                if mode == 0 && hdata.bytes_signed != 0 {
                    for byte in &mut tifdata[..(xsize * ysize) as usize] {
                        *byte = byte.wrapping_sub(128);
                    }
                }
                b3d_fwrite(
                    &tifdata[..(pix_size * xsize) as usize * ysize as usize],
                    (pix_size * xsize) as usize,
                    ysize as usize,
                    mrcfp.as_mut().unwrap(),
                );
            }
            /* write more info to mrc header. 1/17/04 eliminate unneeded rewind */
            if let Some(iifile) = tiff.iifile.as_ref().filter(|_| pixel_entered == 0) {
                pixel_size = iifile.xscale;
                y_pixel_size = iifile.yscale;
            }
            hdata.nx = xsize;
            hdata.ny = ysize;
            hdata.mx = hdata.nx;
            hdata.my = hdata.ny;
            hdata.mz = hdata.nz;
            hdata.xlen = hdata.nx as f32 * pixel_size;
            hdata.ylen = hdata.ny as f32 * y_pixel_size;
            hdata.zlen = hdata.nz as f32 * pixel_size;
            if mode == MRC_MODE_RGB {
                hdata.amax = 255.;
                hdata.amean = 128.0;
                hdata.amin = 0.;
            } else {
                hdata.amax = max;
                hdata.amean = mean / hdata.nz as f32;
                hdata.amin = min;
                let _ = out.write_all(&c_format_bytes(
                    "Min = %g, Max = %g, Mean = %g\n",
                    &[
                        CArg::Dbl(min as f64),
                        CArg::Dbl(max as f64),
                        CArg::Dbl(hdata.amean as f64),
                    ],
                ));
            }
            hdata.mode = mode;
            mrc_head_label(&mut hdata, b"tif2mrc: Converted to mrc format.");
            mrc_head_write(mrcfp.as_mut().unwrap(), &mut hdata);

            /* cleanup */
            drop(mrcfp.take());
            let _ = out.flush();
            std::process::exit(0);
        }
        tiff_close_file(&mut tiff);
    }

    /* read in bg file */
    if bg != 0 {
        let name = bgfile.clone().unwrap_or_default();
        if tiff_open_file(name.as_bytes(), openmode, &mut tiff, any_tif_pixel) != 0 {
            exit_error(&c_format_bytes(
                "Couldn't open %s.",
                &[CArg::Bytes(name.as_bytes())],
            ));
        }
        bgfp = tiff.fp.clone().unwrap();
        bgdata = match tiff_read_file(&mut bgfp, &mut tiff) {
            Some(data) => data,
            None => exit_error(&c_format_bytes(
                "Reading %s.",
                &[CArg::Bytes(name.as_bytes())],
            )),
        };
        bg_bits = tiff.bits_per_sample;
        if (bg_bits != 8 && bg_bits != 16) || tiff.photometric_interpretation >= 2 {
            exit_error(b"Background file must be 8 or 16-bit grayscale");
        }

        bgxsize = tiff.directory[WIDTHINDEX].value;
        bgysize = tiff.directory[LENGTHINDEX].value;

        if bg_bits == 8 {
            max = 0.;
            min = 255.;

            for y in 0..bgysize {
                for x in 0..bgxsize as usize {
                    tmpdata = bgdata[x + (y * bgxsize) as usize] as i32;
                    if tmpdata as f32 > max {
                        max = tmpdata as f32;
                    }
                    if (tmpdata as f32) < min {
                        min = tmpdata as f32;
                    }
                }
            }

            for y in 0..bgysize {
                for x in 0..bgxsize as usize {
                    let at = x + (y * bgxsize) as usize;
                    bgdata[at] = (max - bgdata[at] as f32) as u8;
                }
            }
        }

        tiff_close_file(&mut tiff);
    }

    /* Write out mrcheader */
    if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none()
        && imod_backup_file(&argv[(argc - 1) as usize]) != 0
    {
        exit_error(b"Couldn't create backup file");
    }
    mrcfp = crate::imod::libcfshr::b3dutil::ImodFile::open(&argv[(argc - 1) as usize], "wb");
    if mrcfp.is_none() {
        // `perror("tif2mrc")`; see the note at the other site.
        let message = std::io::Error::last_os_error().to_string();
        let message = message
            .split(" (os error ")
            .next()
            .unwrap_or(message.as_str());
        let _ = ImodFile::Stderr.write_all(&c_format_bytes("tif2mrc: %s\n", &[CArg::Str(message)]));
        exit_error(&c_format_bytes(
            "Opening %s",
            &[CArg::Bytes(argv[(argc - 1) as usize].as_bytes())],
        ));
    }
    mrc_head_new(&mut hdata, xsize, ysize, argc - iarg - 1, mode);
    mrc_head_write(mrcfp.as_mut().unwrap(), &mut hdata);

    /* Loop through all the tiff files adding them to the MRC stack. */
    first_file_ind = iarg;
    while iarg < argc - 1 {
        iread = if invert_stack != 0 {
            first_file_ind + argc - 2 - iarg
        } else {
            iarg
        };

        /* Open the TIFF file. */
        if tiff_open_file(
            argv[iread as usize].as_bytes(),
            openmode,
            &mut tiff,
            any_tif_pixel,
        ) != 0
        {
            exit_error(&c_format_bytes(
                "Couldn't open %s.",
                &[CArg::Bytes(argv[iread as usize].as_bytes())],
            ));
        }
        let _ = out.write_all(&c_format_bytes(
            "Opening %s for input\n",
            &[CArg::Bytes(argv[iread as usize].as_bytes())],
        ));
        let _ = out.flush();
        tiffp = tiff.fp.clone().unwrap();
        k = manage_tvipsdata(
            &mut tvips_metadata,
            tiff.iifile.as_ref(),
            &mut label,
            &mut tilt_angle,
        );
        if tilt_file.is_some() {
            if k != 0 {
                exit_error(b"There is no tilt angle value in this file");
            }
            let _ = tiltfp
                .as_mut()
                .unwrap()
                .write_all(&c_format_bytes("%7.2f\n", &[CArg::Dbl(tilt_angle as f64)]));
        }

        /* Decide whether to set up chunks */
        do_chunks = 0;
        num_chunks = 1;
        lines_per_chunk = 0;
        lines_done = 0;
        if let Some(iifile) = tiff.iifile.as_ref().filter(|_| bg == 0) {
            xsize = iifile.nx;
            ysize = iifile.ny;
            k = if tiff.photometric_interpretation == 2 {
                3
            } else {
                tiff.bits_per_sample / 8
            };
            if (mrcxsize == 0 || (xsize == mrcxsize && ysize == mrcysize))
                && xsize as f64 * ysize as f64 * k as f64 > chunk_criterion as f64
            {
                num_chunks =
                    1 + ((xsize as f64 * ysize as f64 * k as f64) / chunk_criterion as f64) as i32;
                do_chunks = 1;
                lines_per_chunk = (ysize + num_chunks - 1) / num_chunks;
                lines_done = 0;
                let _ = out.write_all(&c_format_bytes(
                    "Reading file in %d chunks of %d lines\n",
                    &[
                        CArg::Int(num_chunks as i64),
                        CArg::Int(lines_per_chunk as i64),
                    ],
                ));
            }
        }

        for _chunk in 0..num_chunks {
            nlines = 0;
            if do_chunks != 0 {
                nlines = if lines_per_chunk < ysize - lines_done {
                    lines_per_chunk
                } else {
                    ysize - lines_done
                };
                let iifile = tiff.iifile.as_mut().unwrap();
                iifile.lly = lines_done;
                iifile.ury = lines_done + nlines - 1;
                lines_done += nlines;
            }

            /* Read in tiff file */
            tifdata = match tiff_read_file(&mut tiffp, &mut tiff) {
                Some(data) => data,
                None => exit_error(&c_format_bytes(
                    "Reading %s.",
                    &[CArg::Bytes(argv[iread as usize].as_bytes())],
                )),
            };

            xsize = tiff.directory[WIDTHINDEX].value;
            ysize = tiff.directory[LENGTHINDEX].value;
            if nlines == 0 {
                nlines = ysize;
            }

            if _chunk == 0 && first_file_ind == iarg {
                if mrcxsize == 0 || mrcysize == 0 {
                    mrcxsize = xsize;
                    mrcysize = ysize;
                }
                manage_mode(
                    &tiff,
                    keep_ushort,
                    force_signed,
                    makegray,
                    &mut pix_size,
                    &mut mode,
                );

                /* Collect the pixel size the first time */
                if let Some(iifile) = tiff.iifile.as_ref().filter(|_| pixel_entered == 0) {
                    pixel_size = iifile.xscale;
                    y_pixel_size = iifile.yscale;
                }
            }

            if (tiff.bits_per_sample == 16 && mode != MRC_MODE_SHORT && mode != MRC_MODE_USHORT)
                || (tiff.bits_per_sample == 32 && mode != MRC_MODE_FLOAT)
                || (tiff.photometric_interpretation / 2 == 1
                    && makegray == 0
                    && mode != MRC_MODE_RGB)
                || (((tiff.photometric_interpretation / 2 == 1 && makegray != 0)
                    || (tiff.photometric_interpretation / 2 == 0 && tiff.bits_per_sample == 8))
                    && mode != MRC_MODE_BYTE)
            {
                exit_error(b"All files must have the same data type.");
            }

            if tiff.photometric_interpretation == 3 {
                expand_index_to_rgb(&mut tifdata, tiff.iifile.as_ref(), 0);
            }

            /* convert RGB to gray scale */
            if tiff.photometric_interpretation / 2 == 1 && makegray != 0 {
                convertrgb(&mut tifdata, xsize, nlines, use_ntsc);
            }

            /* Convert long ints to floats */
            convert_long_to_float(&mut tifdata, tiff.iifile.as_ref());

            /* Correct for bg */
            if bg != 0 {
                if (mode != MRC_MODE_SHORT && mode != MRC_MODE_USHORT && bg_bits == 16)
                    || (mode != MRC_MODE_BYTE && bg_bits == 8)
                {
                    exit_error(
                        b"Background data must have  the same data type as the image files.",
                    );
                }

                xdo = if bgxsize < xsize { bgxsize } else { xsize } as usize;
                ydo = if bgysize < ysize { bgysize } else { ysize };

                if mode == MRC_MODE_BYTE {
                    for y in 0..ydo {
                        for x in 0..xdo {
                            let at = x + (y as usize * xdo);
                            tmpdata = tifdata[at] as i32 + bgdata[at] as i32;
                            if tmpdata > 255 {
                                tmpdata = 255;
                            }
                            tifdata[at] = tmpdata as u8;
                        }
                    }
                } else {
                    for y in 0..ydo {
                        for x in 0..xdo {
                            let at = x + (y as usize * xdo);
                            let offset = 2 * at;
                            let value =
                                i16::from_ne_bytes(tifdata[offset..offset + 2].try_into().unwrap());
                            let background =
                                i16::from_ne_bytes(bgdata[offset..offset + 2].try_into().unwrap());
                            tifdata[offset..offset + 2]
                                .copy_from_slice(&value.wrapping_sub(background).to_ne_bytes());
                        }
                    }
                }
            }

            tmean = minmaxmean(
                &mut tifdata,
                mode,
                unsign,
                divide,
                xsize,
                nlines,
                &mut min,
                &mut max,
            );
            mean += (tmean * nlines as f32) / ysize as f32;

            if mode == 0 && hdata.bytes_signed != 0 {
                for byte in &mut tifdata[..(xsize * nlines) as usize] {
                    *byte = byte.wrapping_sub(128);
                }
            }

            if (xsize == mrcxsize) && (ysize == mrcysize) {
                /* Write out mrc file */
                b3d_fwrite(
                    &tifdata[..(pix_size * xsize) as usize * nlines as usize],
                    (pix_size * xsize) as usize,
                    nlines as usize,
                    mrcfp.as_mut().unwrap(),
                );
            } else {
                let _ = out.write_all(&c_format_bytes(
                    "WARNING: tif2mrc - File %s not same size.\n",
                    &[CArg::Bytes(argv[iread as usize].as_bytes())],
                ));

                /* Unequal sizes: set the fill value and pointer */
                fill_val = if fill_entered != 0 { user_fill } else { tmean };
                // `tif2mrc.c:1100-1130` points `fillPtr` at the fill value
                // that matches the mode; the bytes of that value are what
                // it writes, so the selection is a byte slice here.
                let mut wide_fill = [0_u8; 4];
                let mut fill_bytes: &[u8] = &[];
                match mode {
                    MRC_MODE_BYTE => {
                        byte_fill[0] = fill_val as u8;
                        if hdata.bytes_signed != 0 {
                            byte_fill[0] = ((fill_val as i32 - 128) & 255) as u8;
                        }
                        fill_bytes = &byte_fill[..];
                    }
                    MRC_MODE_RGB => {
                        let value = if fill_entered != 0 {
                            user_fill as i32 as u8
                        } else {
                            128
                        };
                        byte_fill[0] = value;
                        byte_fill[1] = value;
                        byte_fill[2] = value;
                        fill_bytes = &byte_fill[..];
                    }
                    MRC_MODE_SHORT => {
                        short_fill = fill_val as i16;
                        wide_fill[..2].copy_from_slice(&short_fill.to_ne_bytes());
                        fill_bytes = &wide_fill[..2];
                    }
                    MRC_MODE_USHORT => {
                        ushort_fill = fill_val as u16;
                        wide_fill[..2].copy_from_slice(&ushort_fill.to_ne_bytes());
                        fill_bytes = &wide_fill[..2];
                    }
                    MRC_MODE_FLOAT => {
                        wide_fill = fill_val.to_ne_bytes();
                        fill_bytes = &wide_fill[..];
                    }
                    _ => {}
                }

                /* Output centered data */
                yoffset = (ysize - mrcysize) / 2;
                xoffset = (xsize - mrcxsize) / 2;
                for y in 0..mrcysize {
                    if y + yoffset < 0 || y + yoffset >= ysize {
                        /* Do fill lines */
                        for _x in 0..mrcxsize {
                            b3d_fwrite(
                                &fill_bytes[..pix_size as usize],
                                pix_size as usize,
                                1,
                                mrcfp.as_mut().unwrap(),
                            );
                        }
                    } else {
                        /* Fill left edge if necessary, write data, fill right if needed */
                        k = xoffset;
                        while k < 0 {
                            b3d_fwrite(
                                &fill_bytes[..pix_size as usize],
                                pix_size as usize,
                                1,
                                mrcfp.as_mut().unwrap(),
                            );
                            k += 1;
                        }
                        xdo = (pix_size
                            * (if xoffset > 0 { xoffset } else { 0 } + (y + yoffset) * xsize))
                            as usize;
                        let count = (if xsize < mrcxsize { xsize } else { mrcxsize }) as usize;
                        b3d_fwrite(
                            &tifdata[xdo..xdo + pix_size as usize * count],
                            pix_size as usize,
                            count,
                            mrcfp.as_mut().unwrap(),
                        );
                        k = 0;
                        while k < mrcxsize - xsize + xoffset {
                            b3d_fwrite(
                                &fill_bytes[..pix_size as usize],
                                pix_size as usize,
                                1,
                                mrcfp.as_mut().unwrap(),
                            );
                            k += 1;
                        }
                    }
                }
            }
            tifdata = Vec::new();
        }

        tiff_close_file(&mut tiff);
        iarg += 1;
    }

    /* write more info to mrc header. 1/17/04 eliminate unneeded rewind */
    hdata.nx = mrcxsize;
    hdata.ny = mrcysize;
    hdata.mx = hdata.nx;
    hdata.my = hdata.ny;
    hdata.mz = hdata.nz;
    hdata.xlen = hdata.nx as f32 * pixel_size;
    hdata.ylen = hdata.ny as f32 * y_pixel_size;
    hdata.zlen = hdata.nz as f32 * pixel_size;
    if mode == MRC_MODE_RGB {
        hdata.amax = 255.;
        hdata.amean = 128.0;
        hdata.amin = 0.;
    } else {
        hdata.amax = max;
        hdata.amean = mean / hdata.nz as f32;
        hdata.amin = min;
        let _ = out.write_all(&c_format_bytes(
            "Min = %g, Max = %g, Mean = %g\n",
            &[
                CArg::Dbl(min as f64),
                CArg::Dbl(max as f64),
                CArg::Dbl(hdata.amean as f64),
            ],
        ));
    }
    hdata.mode = mode;
    mrc_head_label(&mut hdata, b"tif2mrc: Converted to MRC format.");
    if label[0] != 0x00 {
        // `tif2mrc.c:1236`: `strlen(label)`, then blank-fill to the full
        // fixed width -- MRC labels are blank-padded, not NUL-terminated.
        k = label.iter().position(|b| *b == 0).unwrap_or(MRC_LABEL_SIZE) as i32;
        while k < MRC_LABEL_SIZE as i32 {
            label[k as usize] = b' ';
            k += 1;
        }
        if hdata.nlabl < MRC_NLABELS as i32 {
            for index in 0..MRC_LABEL_SIZE {
                hdata.labels[hdata.nlabl as usize][index] = label[index];
            }
            hdata.nlabl += 1;
        }
    }
    mrc_head_write(mrcfp.as_mut().unwrap(), &mut hdata);

    /* cleanup */
    if mrcfp.is_some() {
        drop(mrcfp.take());
    }
    if tilt_file.is_some() {
        drop(tiltfp.take());
    }
    let _ = out.flush();
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn converts_rgb_in_place() {
        let mut values = [30_u8, 60, 90, 9, 12, 15];
        convertrgb(&mut values, 2, 1, 0);
        assert_eq!(&values[..2], &[60, 12]);
    }
    #[test]
    fn ntsc_weighting_rounds_between_terms_like_the_source() {
        // `tif2mrc.c:672-674` stores each partial sum back into a `float`, so
        // these triples come out one count above what a single f32 expression
        // gives (3, 10, 15, 12).  Verified against the native binary.
        let mut values = [0_u8, 5, 5, 0, 15, 15, 0, 19, 39, 0, 21, 1];
        convertrgb(&mut values, 4, 1, 1);
        assert_eq!(&values[..4], &[4, 11, 16, 13]);
    }
    #[test]
    fn unsigned_short_conversion_and_stats() {
        let mut values = [0_u8, 0, 255, 255];
        let mut min = 1.0e5;
        let mut max = -1.0e5;
        let mean = minmaxmean(&mut values, MRC_MODE_SHORT, 1, 0, 2, 1, &mut min, &mut max);
        assert_eq!(
            (
                i16::from_ne_bytes(values[..2].try_into().unwrap()),
                i16::from_ne_bytes(values[2..].try_into().unwrap())
            ),
            (-32768, 32767)
        );
        assert_eq!((min, max, mean), (-32768., 32767., -0.5));
    }

    #[test]
    fn unsigned_16_bit_tiff_selects_unsigned_short_mode() {
        let mut image = ImodImageFile::default();
        image.type_ = IITYPE_USHORT;
        let mut tiff = TfInfo::default();
        tiff.bits_per_sample = 16;
        tiff.iifile = Some(image);
        let mut pixel_size = 0;
        let mut mode = 0;
        manage_mode(&tiff, 0, 0, 0, &mut pixel_size, &mut mode);
        assert_eq!((pixel_size, mode), (2, MRC_MODE_USHORT));
    }

    #[test]
    fn tvips_metadata_carry_over_is_owned_by_one_conversion() {
        let mut bytes = vec![0_u8; 4260];
        bytes[3564..3568].copy_from_slice(&12.5_f32.to_ne_bytes());
        bytes[3624..3628].copy_from_slice(&4.0_f32.to_ne_bytes());
        bytes[3704..3708].copy_from_slice(&90.0_f32.to_ne_bytes());
        bytes[3944..3948].copy_from_slice(&2_i32.to_ne_bytes());
        let image = ImodImageFile {
            user_data: bytes.as_mut_ptr(),
            user_count: bytes.len() as i32,
            user_flags: IIFLAG_TVIPS_DATA,
            ..ImodImageFile::default()
        };
        let mut state = TvipsMetadataState::default();
        let mut label = [0_u8; MRC_LABEL_SIZE];
        let mut tilt_angle = 0.0;

        assert_eq!(
            manage_tvipsdata(&mut state, Some(&image), &mut label, &mut tilt_angle),
            0
        );
        assert_eq!(tilt_angle, 12.5);
        assert!(label.starts_with(b"    Tilt axis angle = 0.0, binning = 2  spot = 4"));

        bytes[3704..3708].copy_from_slice(&91.0_f32.to_ne_bytes());
        assert_eq!(
            manage_tvipsdata(&mut state, Some(&image), &mut label, &mut tilt_angle),
            0
        );
        assert_eq!(label[0], 0);
    }

    #[test]
    fn float_statistics_follow_c_integer_pixel_conversion() {
        let mut values = [1.9_f32.to_ne_bytes(), (-2.4_f32).to_ne_bytes()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let mut min = 100000.0;
        let mut max = -100000.0;
        let mean = minmaxmean(&mut values, MRC_MODE_FLOAT, 0, 0, 2, 1, &mut min, &mut max);
        assert_eq!((min, max, mean), (-2.0, 1.0, -0.5));
    }

    #[test]
    fn rgb_statistics_leave_source_minimum_and_maximum_unchanged() {
        let mut values = [30_u8, 60, 90];
        let mut min = 100000.0;
        let mut max = -100000.0;
        let mean = minmaxmean(&mut values, MRC_MODE_RGB, 0, 0, 1, 1, &mut min, &mut max);
        assert_eq!((min, max, mean), (100000.0, -100000.0, 0.0));
    }
}
