//! Translation of `IMOD/imodutil/imodauto.c` -- automatic model generation
//! program.

use std::io::Write as _;

use crate::imod::clip::clip::{ScanArg, atof, atoi, sscanf};
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_omp_thread_num, b3d_physical_memory, b3d_set_store_error, c_format_bytes,
    imod_backup_file, imod_copyright, imod_prog_name, imod_version, num_omp_threads,
};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
use crate::imod::libiimod::iimage::ii_fopen;
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_RAMP_LIN, MrcHeader, mrc_contrast_scaling, mrc_fix_li, mrc_head_read, mrc_init_li,
};
use crate::imod::libiimod::mrcsec::mrc_read_z_byte;
use crate::imod::libimod::autocont::{MAX_AUTO_SLICE_THREADS, imod_auto_contours_from_slice};
use crate::imod::libimod::imodel::{IOBJ_STRSIZE, Imod, Iobj};
use crate::imod::libimod::imodel::{imod_new, imod_set_ref_image, imod_trans_for_subset_load};
use crate::imod::libimod::imodel_files::{imod_read, imod_write};
use crate::imod::libimod::iobj::{imod_object_new, iobj_close};

/// C `usage` (`imodauto.c:30`).
fn usage() {
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(b"Usage: imodauto [options] <image file> <model file>\n");
    let _ = out.write_all(b"Options:\n");
    let _ = out.write_all(b"\t-l #\tLow threshold level.\n");
    let _ = out.write_all(b"\t-h #\tHigh threshold level.\n");
    let _ = out.write_all(b"\t-E #\tExact value to make contours around.\n");
    let _ = out.write_all(b"\t-d #\tThreshold flag 1=absolute 2=section-mean 3=stack-mean.\n");
    let _ = out.write_all(b"\t-u\tInterpret threshold levels as unscaled intensities.\n");
    let _ = out.write_all(b"\t-n\tFind inside contours at same threshold level.\n");
    let _ =
        out.write_all(b"\t-f #\tFollow diagonals: 0=never, 1=above high, 2=below low, 3=always.\n");

    let _ = out.write_all(b"\t-m #\tMinimum contour area mask.\n");
    let _ = out.write_all(b"\t-M #\tMaximum contour area mask.\n");
    let _ = out.write_all(b"\t-e #\tEdge mask: eliminate contours touching that # of edges.\n");
    let _ = out.write_all(b"\t-s min,max\tIntensity scaling: scale min to 0 and max to 255.\n");
    let _ = out.write_all(b"\t-X min,max\tLoad subset in X from min to max (numbered from 0).\n");
    let _ = out.write_all(b"\t-Y min,max\tLoad subset in Y from min to max (numbered from 0).\n");
    let _ = out.write_all(b"\t-Z min,max\tLoad subset in Z from min to max (numbered from 0).\n");
    let _ = out.write_all(b"\t-B file\tName of model file with boundary contours.\n");
    let _ = out.write_all(b"\t-O #\tNumber of object with boundary contours in boundary model.\n");
    let _ = out.write_all(b"\t-S \tApply boundary contours only on the same Z as the contour.\n");
    let _ = out.write_all(b"\t-k #\tSmooth with kernel filter of given sigma.\n");
    let _ = out.write_all(b"\t-z #\tSet model z scale.\n");
    let _ = out.write_all(b"\t-x \tExpand areas before enclosing in contours.\n");
    let _ = out.write_all(b"\t-i \tShrink areas before enclosing in contours.\n");
    let _ = out.write_all(b"\t-o \tSmooth areas (expand then shrink).\n");
    let _ =
        out.write_all(b"\t-a #\tNumber of times to apply expand, shrink, or smooth operation.\n");
    let _ = out.write_all(b"\t-r #\tResolution factor (pixels) for shaving points.\n");
    let _ = out.write_all(b"\t-R #\tTolerance (maximum error) for point reduction.\n");
    let _ = out
        .write_all(b"\t-c r,g,b   Color of model object (r,g,b between 0 and 1 or 0 and 255).\n");
    let _ = out
        .write_all(b"\t-N name\tName of model object (enclose in quotes if more than one word).\n");
    let _ = out
        .write_all(b"\t-I file\tName of image file to use for coordinate reference information.\n");
}

/// C `main` (`imodauto.c:65`).
pub fn imodauto(argv: &[String]) -> i32 {
    let argc = argv.len() as i32;

    let mut imod: Option<Imod>;
    let mut hdata = MrcHeader::default();
    let mut li = LoadInfo::default();

    let mut fin: ImodFile;
    let mut fout: ImodFile;
    let mut f_ref: Option<ImodFile> = None;
    let mut bound_file: Option<String> = None;
    let mut bound_mod: Option<Imod> = None;
    let mut bound_obj_index: Option<usize> = None;
    let mut im_ref_file: Option<String> = None;
    let mut obj_name: Option<String> = None;
    let mut nearest_bound: i32 = 1;
    let mut bound_obj_num: i32 = -1;
    let mut ht: f64 = 256.0;
    let mut lt: f64 = 0.0;
    let mut shave: f64 = 0.0;
    let mut tol: f64 = 0.0;
    let mut exact: i32 = -2;
    let mut exact_in: f64 = -2.;
    let mut dim: i32 = 1;
    let mut minsize: i32 = 10;
    let mut maxsize: i32 = -1;
    let mut delete_edge: i32 = 0;
    let mut apply_xio: i32 = 0;
    let mut i: i32;
    let mut zscale: f32 = 1.0;
    let mut ksigma: f32 = -1.;
    let mut smoothflags: i32 = 0;
    let mut followdiag: i32 = 0;
    let mut inside: i32 = 0;
    let mut red: f32 = 0.;
    let mut green: f32 = 1.;
    let mut blue: f32 = 0.;
    let mut unscaled: i32 = 0;
    let mut hentered: i32 = 0;
    let mut lentered: i32 = 0;
    let mut xentered: i32 = 0;
    let progname = imod_prog_name(&argv[0]);

    if argc < 3 {
        imod_version(Some(&progname));
        imod_copyright();
        usage();
        std::process::exit(3);
    }

    setExitPrefix(b"\nERROR: imodauto - ");

    /* Make library error output to stderr go to stdout */
    b3d_set_store_error(-1);

    mrc_init_li(Some(&mut li), None);
    /* `argv[++i]` reads one past the last argument when an option that takes
    a value is last; the C library's `atoi`/`atof` are handed `NULL` there.
    An absent argument is an empty string here, which is what `atoi("")` and
    `atof("")` see. */
    let arg = |k: i32| -> &str { argv.get(k as usize).map_or("", |s| s.as_str()) };
    i = 1;
    while i < argc {
        if arg(i).as_bytes().first() == Some(&b'-') {
            match arg(i).as_bytes().get(1).copied().unwrap_or(0) {
                b'm' => {
                    i += 1;
                    minsize = atoi(arg(i));
                }
                b'M' => {
                    i += 1;
                    maxsize = atoi(arg(i));
                }
                b'd' => {
                    i += 1;
                    dim = atoi(arg(i));
                }
                b'h' => {
                    i += 1;
                    ht = atof(arg(i));
                    hentered = 1;
                }
                b'k' => {
                    i += 1;
                    ksigma = atof(arg(i)) as f32;
                }
                b'l' => {
                    i += 1;
                    lt = atof(arg(i));
                    lentered = 1;
                }
                b'E' => {
                    xentered = 1;
                    i += 1;
                    exact_in = atof(arg(i));
                    exact = (exact_in + 0.5).floor() as i32;
                }
                b'r' => {
                    i += 1;
                    shave = atof(arg(i));
                }
                b'R' => {
                    i += 1;
                    tol = atof(arg(i));
                }
                b'e' => {
                    i += 1;
                    delete_edge = atoi(arg(i));
                }
                b'z' => {
                    i += 1;
                    zscale = atof(arg(i)) as f32;
                }
                b's' => {
                    i += 1;
                    sscanf(
                        arg(i),
                        "%f%*c%f",
                        &mut [ScanArg::Flt(&mut li.smin), ScanArg::Flt(&mut li.smax)],
                    );
                }
                b'x' => {
                    if smoothflags != 0 {
                        smoothflags = -1;
                    } else {
                        smoothflags = 2;
                    }
                }
                b'o' => {
                    if smoothflags != 0 {
                        smoothflags = -1;
                    } else {
                        smoothflags = 3;
                    }
                }
                b'i' => {
                    if smoothflags != 0 {
                        smoothflags = -1;
                    } else {
                        smoothflags = 1;
                    }
                }
                b'a' => {
                    i += 1;
                    apply_xio = atoi(arg(i));
                }
                b'n' => {
                    inside = 1;
                }
                b'u' => {
                    unscaled = 1;
                }
                b'f' => {
                    i += 1;
                    followdiag = atoi(arg(i));
                }
                b'c' => {
                    i += 1;
                    sscanf(
                        arg(i),
                        "%f%*c%f%*c%f",
                        &mut [
                            ScanArg::Flt(&mut red),
                            ScanArg::Flt(&mut green),
                            ScanArg::Flt(&mut blue),
                        ],
                    );
                    if red > 1. || green > 1. || blue > 1. {
                        red /= 255.;
                        green /= 255.;
                        blue /= 255.;
                    }
                }
                b'N' => {
                    i += 1;
                    obj_name = Some(arg(i).to_string());
                }
                b'X' => {
                    i += 1;
                    sscanf(
                        arg(i),
                        "%d%*c%d",
                        &mut [ScanArg::Int(&mut li.xmin), ScanArg::Int(&mut li.xmax)],
                    );
                }
                b'Y' => {
                    i += 1;
                    sscanf(
                        arg(i),
                        "%d%*c%d",
                        &mut [ScanArg::Int(&mut li.ymin), ScanArg::Int(&mut li.ymax)],
                    );
                }
                b'Z' => {
                    i += 1;
                    sscanf(
                        arg(i),
                        "%d%*c%d",
                        &mut [ScanArg::Int(&mut li.zmin), ScanArg::Int(&mut li.zmax)],
                    );
                }
                b'B' => {
                    i += 1;
                    bound_file = Some(arg(i).to_string());
                }
                b'O' => {
                    i += 1;
                    bound_obj_num = atoi(arg(i));
                }
                b'S' => {
                    nearest_bound = 0;
                }
                b'I' => {
                    i += 1;
                    im_ref_file = Some(arg(i).to_string());
                }
                _ => {
                    usage();
                    std::process::exit(3);
                }
            }
        } else {
            break;
        }
        i += 1;
    }

    if (dim < 0) || (dim > 3) {
        usage();
        std::process::exit(1);
    }

    if i >= argc - 1 {
        usage();
        std::process::exit(3);
    }
    if smoothflags < 0 {
        exit_error(b"Only one of -x, -i and -o may be entered.");
    }
    if apply_xio < 0 {
        exit_error(b"Entry for -a option must be positive");
    }
    if smoothflags != 0 && apply_xio != 0 {
        smoothflags |= apply_xio << 2;
    }

    if hentered == 0 && lentered == 0 && xentered == 0 {
        exit_error(
            b"You must enter at least one threshold with -l or -h or an exact value with -E",
        );
    }

    if xentered != 0 && dim > 1 {
        exit_error(b"You cannot enter -d with -E");
    }

    if xentered != 0 && (hentered != 0 || lentered != 0) {
        exit_error(b"You cannot enter -l or -h with -E");
    }

    if inside != 0 {
        if hentered != 0 && lentered != 0 {
            exit_error(
                b"Only a high or a low threshold, not both, may be entered when using -n to find inside contours.",
            );
        }

        /* Set thresholds equal, and set to follow diagonals only on the
        primary threshold */
        if xentered == 0 {
            if hentered != 0 {
                followdiag = 1;
                lt = ht;
                lentered = 1;
            } else {
                followdiag = 2;
                ht = lt;
                hentered = 1;
            }
        }
    }

    let image_arg = i;
    i += 1;
    match ii_fopen(arg(image_arg).as_bytes(), "rb") {
        Some(file) => fin = file,
        None => {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "ERROR: %s - Opening image file %s.\n",
                &[CArg::Str(&progname), CArg::Str(arg(image_arg))],
            ));
            perror_imodauto("imodauto open image");
            std::process::exit(3);
        }
    }

    if let Some(im_ref_file) = im_ref_file.as_ref() {
        f_ref = ii_fopen(im_ref_file.as_bytes(), "rb");
        if f_ref.is_none() {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "ERROR: %s - Opening image file %s for coordinate reference.\n",
                &[CArg::Str(&progname), CArg::Str(im_ref_file)],
            ));
            perror_imodauto("imodauto open image");
            std::process::exit(3);
        }
    }

    /* read in graphic header and image data */
    if mrc_head_read(&mut fin, &mut hdata) != 0 {
        exit_error(b"Reading input file header.");
    }

    if li.smin == li.smax {
        li.smin = hdata.amin;
        li.smax = hdata.amax;
    }
    mrc_fix_li(&mut li, hdata.nx, hdata.ny, hdata.nz);

    /* This sets the scaling */
    let (slope, offset) =
        mrc_contrast_scaling(&hdata, li.smin, li.smax, li.black, li.white, MRC_RAMP_LIN);
    li.slope = slope;
    li.offset = offset;

    /* DNM 6/19/02: scale entered values if unscaled option taken */
    if unscaled != 0 {
        if hentered != 0 {
            ht = 255. * (ht - li.smin as f64) / (li.smax - li.smin) as f64;
        }
        if lentered != 0 {
            lt = 255. * (lt - li.smin as f64) / (li.smax - li.smin) as f64;
        }
        if xentered != 0 {
            exact = (255. * (exact_in - li.smin as f64) / (li.smax - li.smin) as f64 + 0.5).floor()
                as i32;
        }
    }

    /* Now that the true exact value is known, set the other thresholds for doing inside */
    if xentered != 0 {
        if inside != 0 {
            lt = exact as f64 - 1.;
            ht = exact as f64 + 1.;
            followdiag = 1;
        } else {
            lt = -1.;
            if followdiag != 0 {
                followdiag = 3;
            }
        }
    }

    /* Open boundary model if any */
    if let Some(bound_file) = bound_file.as_ref() {
        let Ok(model) = imod_read(bound_file) else {
            exit_error(&c_format_bytes(
                "Reading boundary model file %s",
                &[CArg::Str(bound_file)],
            ));
        };
        bound_mod = Some(model);
        let bmod = bound_mod.as_mut().expect("boundary model just read");
        if (bmod.obj.len() as i32) < bound_obj_num {
            bound_obj_num = bmod.obj.len() as i32;
        }
        if bound_obj_num <= 0 {
            /* Find first closed contour object */
            for ob in 0..bmod.obj.len() {
                if iobj_close(bmod.obj[ob].flags) != 0 {
                    bound_obj_num = ob as i32 + 1;
                    break;
                }
            }
            if bound_obj_num <= 0 {
                exit_error(&c_format_bytes(
                    "No closed contour objects were found in boundary model %s",
                    &[CArg::Str(bound_file)],
                ));
            }
        } else if iobj_close(bmod.obj[(bound_obj_num - 1) as usize].flags) == 0 {
            exit_error(&c_format_bytes(
                "Object %d in boundary model %s is not a closed contour object",
                &[CArg::Int(bound_obj_num as i64), CArg::Str(bound_file)],
            ));
        }

        bound_obj_index = Some((bound_obj_num - 1) as usize);

        /* Shift the model if it was loaded on a subset and also adjust for the current
        restriction on loading */
        imod_trans_for_subset_load(bmod, &hdata, Some(&li));
    }

    /* Rename existing file if any */
    imod_backup_file(arg(i));

    match ImodFile::open(arg(i), "wb") {
        Some(file) => fout = file,
        None => {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "ERROR: %s - Opening %s\n",
                &[CArg::Str(&progname), CArg::Str(arg(i))],
            ));
            perror_imodauto("imodauto open model");
            std::process::exit(3);
        }
    }

    imod = imod_new();
    if imod.is_none() {
        exit_error(&c_format_bytes(
            "Creating model %s.\n",
            &[CArg::Str(arg(i))],
        ));
    }
    let imod = imod.as_mut().expect("model just created");

    let obj = imoda_object_create_threshold_data(
        &mut hdata,
        &mut li,
        ksigma,
        ht,
        lt,
        exact,
        dim,
        minsize,
        maxsize,
        followdiag,
        inside,
        shave,
        tol,
        delete_edge,
        smoothflags,
        bound_mod
            .as_ref()
            .and_then(|m| bound_obj_index.map(move |ind| (m, ind))),
        nearest_bound,
    );
    let Some(mut obj) = obj else {
        exit_error(b"Allocating memory while contouring the data");
    };

    obj.red = red;
    obj.green = green;
    obj.blue = blue;
    if let Some(obj_name) = obj_name.as_ref() {
        let bytes = obj_name.as_bytes();
        let mut j = 0usize;
        while j < IOBJ_STRSIZE - 1 && j < bytes.len() && bytes[j] != 0 {
            obj.name[j] = bytes[j];
            j += 1;
        }
        obj.name[j] = 0x00;
    }

    /* Shift all points in X/Y in the object by the load-in min values; correct Z was
    assigned already */
    for co in 0..obj.cont.len() {
        for pt in 0..obj.cont[co].pts.len() {
            obj.cont[co].pts[pt].x += li.xmin as f32;
            obj.cont[co].pts[pt].y += li.ymin as f32;
        }
    }

    imod.obj.push(obj);
    imod.zscale = zscale;
    imod.xmax = hdata.nx;
    imod.ymax = hdata.ny;
    imod.zmax = hdata.nz;
    imod.cindex.object = imod.obj.len() as i32 - 1;

    /* Set up image reference information. */
    if im_ref_file.is_some() {
        let f_ref = f_ref.as_mut().expect("reference file opened above");
        if mrc_head_read(f_ref, &mut hdata) != 0 {
            exit_error(b"Reading header of file with coordinate reference information.");
        }
    }
    imod_set_ref_image(imod, &hdata);

    let _ = imod_write(imod, &mut fout);
    drop(fout);
    std::process::exit(0);
}

/// `perror(prefix)`: the C library writes `"<prefix>: <strerror(errno)>\n"` to
/// stderr, and only when `errno` is set.  Rust's `io::Error` Display appends
/// ` (os error N)`, which the C library does not, so the suffix is trimmed.
fn perror_imodauto(prefix: &str) {
    let err = std::io::Error::last_os_error();
    if err.raw_os_error().unwrap_or(0) == 0 {
        return;
    }
    let message = err.to_string();
    let message = message
        .split(" (os error ")
        .next()
        .unwrap_or(message.as_str());
    let _ = ImodFile::Stderr.write_all(&c_format_bytes(
        "%s: %s\n",
        &[CArg::Str(prefix), CArg::Str(message)],
    ));
}

/// `MAX_THREADS` (`imodauto.c:433`).
const MAX_THREADS: i32 = 16;

/// C `imodaObjectCreateThresholdData` (`imodauto.c:437`).
///
/// Creates an object from a 3-D image array.  The C's `Iobj *boundObj` points
/// into the boundary model; here the model and the object's index travel
/// together so the borrow is explicit.
#[allow(clippy::too_many_arguments)]
fn imoda_object_create_threshold_data(
    hdata: &mut MrcHeader,
    li: &mut LoadInfo,
    ksigma: f32,
    highthresh: f64,
    lowthresh: f64,
    exact: i32,
    dim: i32,
    minsize: i32,
    maxsize: i32,
    followdiag: i32,
    inside: i32,
    shave: f64,
    tol: f64,
    delete_edge: i32,
    smoothflags: i32,
    bound: Option<(&Imod, usize)>,
    nearest_bound: i32,
) -> Option<Iobj> {
    let mut bound_conts: Vec<Vec<i32>> = Vec::new();
    let mut tdata: Vec<Vec<i32>> = Vec::new();
    let nx: i32 = li.xmax + 1 - li.xmin;
    let ny: i32 = li.ymax + 1 - li.ymin;
    let nz: i32 = li.zmax + 1 - li.zmin;
    let mut xlist: Vec<Vec<i32>> = Vec::new();
    let mut ylist: Vec<Vec<i32>> = Vec::new();
    let mut idata: Vec<Vec<u8>> = Vec::new();
    let mut fdata: Vec<Vec<u8>> = Vec::new();
    let mut ind: i32;
    let mut error: i32;
    let mut num_cont: i32;
    let listsize: i32;
    let mut num_started: i32 = 0;
    let mut slice_threads: i32 = 1;
    let mut num_threads: i32;
    let mut ksec: i32;
    let nxy: i32 = nx * ny;
    let mut mean: f32 = 0.;
    let mut bytes_per_slice: f32;
    let mut mem_limit_mb: f32 = 2000.;
    let mut physical: f64 = b3d_physical_memory();
    if physical != 0. {
        // If memory available, do it the same as newstack: the min of 15 GB, 3/4 of
        // physical, and physical  minus 1 GB, but at least 0.4 GB when memory is up to
        // 30 GB, and 1/2 of memory above that
        physical /= 1024. * 1024.;
        if physical < 30000. {
            mem_limit_mb = (if 0.75 * physical < physical - 1000. {
                0.75 * physical
            } else {
                physical - 1000.
            }) as f32;
            /* B3DCLAMP(a, b, c) is a = B3DMAX(b, B3DMIN(c, a)). */
            mem_limit_mb = {
                let inner = if 15000. < mem_limit_mb {
                    15000.
                } else {
                    mem_limit_mb
                };
                if 400. > inner { 400. } else { inner }
            };
        } else {
            mem_limit_mb = (physical / 2.) as f32;
        }
    }

    /* Roughly, 100x100 should be limited to 2 threads, 200x200 to 4 */
    listsize = 4 * (nx + ny);
    num_threads = ((nxy as f64).sqrt() / 50. + 0.5).floor() as i32;
    bytes_per_slice = (6. * nxy as f64 + 8. * listsize as f64) as f32;
    ksec = (mem_limit_mb as f64 * 1024. * 1024. / bytes_per_slice as f64) as i32;
    num_threads = if num_threads < ksec {
        num_threads
    } else {
        ksec
    };
    num_threads = if num_threads < hdata.nz {
        num_threads
    } else {
        hdata.nz
    };

    if num_threads <= 1 {
        // This is a bit silly, it should be done based on number of contour areas perhaps
        slice_threads = ((nxy as f64).sqrt() / 50. + 0.5).floor() as i32;
        bytes_per_slice = (nxy as f64 + 8. * listsize as f64) as f32;
        ksec = (mem_limit_mb as f64 * 1024. * 1024. / bytes_per_slice as f64) as i32;
        slice_threads = if slice_threads < ksec {
            slice_threads
        } else {
            ksec
        };
        slice_threads = {
            let inner = if MAX_AUTO_SLICE_THREADS < slice_threads {
                MAX_AUTO_SLICE_THREADS
            } else {
                slice_threads
            };
            if 1 > inner { 1 } else { inner }
        };
        slice_threads = num_omp_threads(slice_threads);
        num_threads = 1;
    }
    let _ = bytes_per_slice;
    num_threads = {
        let inner = if MAX_THREADS < num_threads {
            MAX_THREADS
        } else {
            num_threads
        };
        if 1 > inner { 1 } else { inner }
    };
    num_threads = num_omp_threads(num_threads);
    num_threads = if num_threads < MAX_THREADS {
        num_threads
    } else {
        MAX_THREADS
    };
    if num_threads > 1 {
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "Number of sections being done in parallel = %d\n",
            &[CArg::Int(num_threads as i64)],
        ));
    }
    let mut new_obj = imod_object_new()?;
    new_obj.red = 0.;
    new_obj.green = 1.;
    new_obj.blue = 0.;

    /* Allocate data for all the threads.  `linePtrs` is not built here: the
    translated `imodAutoContoursFromSlice` rebuilds the rows of `idata` after
    it has filtered them, which is what the C's line pointers see. */
    for _ in 0..num_threads {
        tdata.push(vec![0i32; nxy as usize]);
        idata.push(vec![0u8; nxy as usize]);
        fdata.push(vec![0u8; (nxy * slice_threads) as usize]);
        xlist.push(vec![0i32; (listsize * slice_threads) as usize]);
        ylist.push(vec![0i32; (listsize * slice_threads) as usize]);
        bound_conts.push(Vec::new());
    }

    /* And an object for each section */
    let mut sec_obj: Vec<Iobj> = Vec::with_capacity(nz as usize);
    for _ in 0..nz {
        let Some(one) = imod_object_new() else {
            exit_error(b"Allocating object for each section");
        };
        sec_obj.push(one);
    }

    /* Get whole-file mean for dim = 2.  5/30/08: Trust file mean! */
    if dim == 2 {
        mean = hdata.amean;
    }
    error = 0;

    ksec = li.zmin;
    while ksec <= li.zmax {
        ind = b3d_omp_thread_num();
        if error != 0 {
            ksec += 1;
            continue;
        }
        {
            if mrc_read_z_byte(hdata, li, &mut idata[ind as usize], ksec) != 0 {
                error = 10 + ksec;
            }
            num_started += 1;
            let mut out = ImodFile::Stdout;
            let _ = out.write_all(&c_format_bytes(
                "\rStarting section %d of %d",
                &[CArg::Int(num_started as i64), CArg::Int(nz as i64)],
            ));
            let _ = out.flush();
        }
        if error != 0 {
            ksec += 1;
            continue;
        }

        let bound_obj = bound.map(|(model, index)| &model.obj[index]);
        if imod_auto_contours_from_slice(
            ksigma,
            highthresh,
            lowthresh,
            exact,
            dim,
            minsize,
            maxsize,
            followdiag,
            inside,
            shave,
            tol,
            delete_edge,
            smoothflags,
            bound_obj,
            nearest_bound,
            &mut sec_obj[(ksec - li.zmin) as usize],
            &mut bound_conts[ind as usize],
            nx,
            ny,
            &mut tdata[ind as usize],
            &mut idata[ind as usize],
            &mut fdata[ind as usize],
            &mut xlist[ind as usize],
            &mut ylist[ind as usize],
            mean,
            ksec,
            listsize,
            slice_threads,
        ) != 0
        {
            error = 1;
            ksec += 1;
            continue;
        }
        ksec += 1;
    }
    if error != 0 {
        if error > 1 {
            exit_error(&c_format_bytes(
                "Reading section %d from file",
                &[CArg::Int((error - 10) as i64)],
            ));
        }
        return None;
    }

    /* Combine into one new object */
    num_cont = 0;
    for ind in 0..nz {
        num_cont += sec_obj[ind as usize].cont.len() as i32;
    }

    if num_cont != 0 {
        new_obj.cont = Vec::with_capacity(num_cont as usize);
        for ind in 0..nz {
            let taken = std::mem::take(&mut sec_obj[ind as usize].cont);
            new_obj.cont.extend(taken);
        }
    }

    let _ = ImodFile::Stdout.write_all(b"\ndone\n");
    Some(new_obj)
}
