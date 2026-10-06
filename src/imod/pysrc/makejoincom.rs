//! Translation of `IMOD/pysrc/makejoincom`: sets up the command file for
//! joining serial tomograms.
//!
//! The script's top level is [`makejoincom`] and its one function is
//! [`get_adjusted_z_range`].  `rotatevol` (size query) and `goodframe` are
//! our own programs and run through `imodpy::run_cmd`; `densmatch`, whose
//! printed "Scale factors" line the script parsed, is a direct call
//! (`densmatch_compute`, as `trimvol` does), with the two values formatted
//! exactly as that line prints them (`G14.6`) and split as the script split
//! them.

use super::imodpy::{
    MrcInfo, add_imod_bin_ignore_sighup, add_output_format_var_to_lines, call_own_program,
    com_extension_from_option, dataset_filename, default_naming_style, exit_from_imod_error,
    fmtstr, get_mrc, get_mrc_size, get_naming_style, make_backup_file, make_current_dir_writable,
    prnstr, py_int, py_str_float, run_cmd, run_goodframe, set_root_and_extension, write_text_file,
};
use super::pip::{
    exit_error, pip_get_boolean, pip_get_integer, pip_get_non_option_arg, pip_get_string,
    pip_get_three_floats, pip_get_three_integers, pip_get_two_integers, pip_linked_index,
    pip_number_of_entries, pip_read_or_parse_options, pip_set_linked_option, python_uncaught,
};
use crate::imod::flib::image::densmatch::{DensmatchParams, densmatch_compute, densmatch_g_edit};
use crate::imod::flib::subrs::hvem::parse_input_params;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// `version = 5` (`makejoincom:10`): the info file version written.
const VERSION: i32 = 5;
/// `DTOR = 0.0174533` (`makejoincom:11`).
const DTOR: f64 = 0.0174533;

/// The per-tomogram size option: `XYZsizeForRotation` values, or the
/// `FullSizeRotation` / `MaxXYsizeForRotation` flags (`'full'`, `'maxxy'`).
#[derive(Clone, Debug)]
enum SizeOption {
    Sizes([i64; 3]),
    Full,
    MaxXY,
}

/// `def getAdjustedZRange(zentry, itomo, nz, newnz, rotangle, ifflip,
/// bigrot, bottop)` (`makejoincom:13`).  Returns `(chunkadd, range,
/// ifinvert)`.
#[allow(clippy::too_many_arguments)]
fn get_adjusted_z_range(
    zentry: (i64, i64),
    itomo: usize,
    nz: i64,
    newnz: i64,
    rotangle: [f64; 3],
    ifflip: i32,
    bigrot: i32,
    bottop: &str,
) -> (i64, String, i32) {
    let mut zst = zentry.0 - 1;
    let mut znd = zentry.1 - 1;
    if zst < 0 || zst >= nz || znd < 0 || znd >= nz {
        exit_error(&format!(
            "{bottop} entry for tomogram # {} has coordinates out of range",
            itomo + 1
        ));
    }

    // If using rotatevol, need to adjust slice numbers
    // bigrot case rotates the Y axis of an unflipped volume
    // Other case is needed for rotation of Z in flipped vol
    if ifflip == 2 {
        if bigrot == 1 {
            let factor = (DTOR * rotangle[0]).cos()
                * (DTOR * rotangle[1]).sin()
                * (DTOR * rotangle[2]).sin()
                + (DTOR * rotangle[0]).sin() * (DTOR * rotangle[2]).cos();
            zst = (factor * (zst as f64 - 0.5 * (nz - 1) as f64) + 0.5 * newnz as f64) as i64;
            znd = (factor * (znd as f64 - 0.5 * (nz - 1) as f64) + 0.5 * newnz as f64) as i64;
        } else {
            zst = ((DTOR * rotangle[0]).cos()
                * (DTOR * rotangle[1]).cos()
                * (zst as f64 - 0.5 * (nz - 1) as f64)
                + 0.5 * newnz as f64) as i64;
            znd = ((DTOR * rotangle[0]).cos()
                * (DTOR * rotangle[1]).cos()
                * (znd as f64 - 0.5 * (nz - 1) as f64)
                + 0.5 * newnz as f64) as i64;
        }
    }

    // Figure out if inverting, and proper order for extraction
    let mut ifinvert = 0;
    if (bottop == "bottom" && zst > nz.div_euclid(2)) || (bottop == "top" && znd < nz.div_euclid(2))
    {
        ifinvert = 1;
    }
    if (zst < znd && ifinvert != 0) || (zst >= znd && ifinvert == 0) {
        (zst, znd) = (znd, zst);
    }
    let chunkadd = 1 + (zst - znd).abs();
    let mut range = zst.to_string();
    if zst != znd {
        range += &format!("-{znd}");
    }
    (chunkadd, range, ifinvert)
}

/// The script's top level (`makejoincom:48-459`).  Returns the status of its
/// `sys.exit`; error paths exit the process as `exitError` does.
pub fn makejoincom(arguments: &[OsString]) -> i32 {
    let progname = "makejoincom";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let done = |status: i32| {
        let _ = std::io::stdout().flush();
        status
    };

    //
    // Setup runtime environment
    if std::env::var_os("IMOD_DIR").is_some() {
        add_imod_bin_ignore_sighup();
    } else {
        print!("{prefix} IMOD_DIR is not defined!\n");
        return done(1);
    }

    // Fallbacks from ../manpages/autodoc2man 3 1 makejoincom
    let options: Vec<String> = [
        "root:RootName:CH:",
        "input:InputTomogram:FNM:",
        "top:TopSlices:IPL:",
        "bottom:BottomSlices:IPL:",
        "flip:FlipYandZ:BL:",
        "rotate:RotateByAngles:FTL:",
        "already:AlreadyRotated:BL:",
        "xyzsize:XYZsizeForRotation:ITL:",
        "fullsize:FullSizeRotation:BL:",
        "maxxysize:MaxXYsizeForRotation:BL:",
        "dir:DirectoryOfSource:FN:",
        "srcext:SourceExtension:CH:",
        "tmpext:TemporaryExtension:CH:",
        "reference:ReferenceForDensity:I:",
        "midaslim:MidasSizeLimit:I:",
        "style:NamingStyle:I:",
        "pcm:MakeComExtensionPcm:I:",
        "test:TestMode:B:",
        "help:usage:B:",
    ]
    .iter()
    .map(|option| (*option).to_owned())
    .collect();

    // On Windows, make sure the current directory is rxw or at least writeable
    if make_current_dir_writable(".").is_some() {
        exit_error("Cannot make the current directory writable or write files to it");
    }

    pip_set_linked_option("InputTomogram");
    let (_num_opts, num_non_opts) = pip_read_or_parse_options(&argv, &options, progname, 3, 0, 0);
    let mut num_non_opts = num_non_opts as i64;
    // SAFETY: the script sets its own environment before running anything.
    unsafe { std::env::set_var("PIP_PRINT_ENTRIES", "0") };

    // get defaults for adjusting filenames
    let mut defdir = pip_get_string("DirectoryOfSource", "").unwrap_or_default();
    let deftmpext = pip_get_string("TemporaryExtension", "tmp").unwrap_or_else(|_| "tmp".into());
    let defsrcext = pip_get_string("SourceExtension", "rec").unwrap_or_else(|_| "rec".into());
    let mut scaleref = pip_get_integer("ReferenceForDensity", 1).unwrap_or(1) as i64;
    let midas_limit = pip_get_integer("MidasSizeLimit", 1024).unwrap_or(1024);
    let test_mode = pip_get_boolean("TestMode", 0).unwrap_or(0);
    let (name_style_dflt, type_ext) = default_naming_style();
    let (mut name_style, type_ext) = get_naming_style(Some(&type_ext), false, true)
        .unwrap_or_else(|message| exit_error(&message));
    let type_ext = type_ext.unwrap_or_default();
    if name_style < 0 {
        name_style = name_style_dflt;
    }
    let com_ext = com_extension_from_option(0);
    let joincom = format!("startjoin{com_ext}");

    // Convert backslashes in default directory
    if !defdir.is_empty() {
        defdir = defdir.replace('\\', "/");
    }

    // Get the root name.  If it comes from last non-option argument, decrement number
    let mut joinroot = pip_get_string("RootName", "").unwrap_or_default();
    if joinroot.is_empty() {
        if num_non_opts == 0 {
            exit_error("You must enter the root name");
        }
        num_non_opts -= 1;
        joinroot = pip_get_non_option_arg(num_non_opts as i32).unwrap_or_default();
    }

    set_root_and_extension(&joinroot, &type_ext);

    // get the input names.  Insist they are either all option entries or all non-option args
    let num_input = pip_number_of_entries("InputTomogram").unwrap_or(0) as i64;
    if num_input != 0 && num_non_opts != 0 {
        exit_error(
            "You must enter input tomograms either with -input or as non-option arguments, but not both",
        );
    }

    let mut reclist: Vec<String> = Vec::new();
    let ntomo = (num_input + num_non_opts) as usize;
    if ntomo < 2 {
        exit_error("You must enter at least two tomograms");
    }
    for i in 0..ntomo {
        let namein = if num_input != 0 {
            // Fixed in translation (BUGS.md, `makejoincom`): the source calls
            // `PipGetString('InputTomogram')` without the default argument
            // that `pip.py` requires, a TypeError for any `-input` entry.
            pip_get_string("InputTomogram", "").unwrap_or_default()
        } else {
            pip_get_non_option_arg(i as i32).unwrap_or_default()
        };

        // Replace backslashes now
        reclist.push(namein.replace('\\', "/"));
    }

    // Build the lists for the other options
    let mut botlist: Vec<Option<(i64, i64)>> = vec![None; ntomo];
    let mut toplist: Vec<Option<(i64, i64)>> = vec![None; ntomo];
    let mut angleoplist: Vec<Option<[f64; 3]>> = vec![None; ntomo];
    let mut sizeoplist: Vec<Option<SizeOption>> = vec![None; ntomo];
    let mut ifrotlist: Vec<i32> = vec![0; ntomo];
    let mut didrotlist: Vec<i32> = vec![0; ntomo];
    let linkopts = [
        "BottomSlices",
        "TopSlices",
        "FlipYandZ",
        "RotateByAngles",
        "AlreadyRotated",
        "XYZsizeForRotation",
        "FullSizeRotation",
        "MaxXYsizeForRotation",
    ];
    let linktype = [2, 2, 0, 3, 0, 3, 0, 0];

    for (iopt, option) in linkopts.iter().enumerate() {
        let numin = pip_number_of_entries(option).unwrap_or(0);
        for _ in 0..numin {
            let index = pip_linked_index(option).unwrap_or(0) as usize;
            if index >= ntomo {
                exit_error(&format!(
                    "You cannot enter the option {option} after the last tomogram entry"
                ));
            }
            if linktype[iopt] == 3 {
                if *option == "RotateByAngles" {
                    if ifrotlist[index] != 0 {
                        exit_error(&format!(
                            "You entered both -flip and -rotate for tomogram # {}",
                            index + 1
                        ));
                    }
                    let sizes = pip_get_three_floats(option, (0., 0., 0.)).unwrap_or((0., 0., 0.));
                    ifrotlist[index] = 2;
                    angleoplist[index] = Some([sizes.0, sizes.1, sizes.2]);
                } else {
                    let sizes = pip_get_three_integers(option, (0, 0, 0)).unwrap_or((0, 0, 0));
                    sizeoplist[index] = Some(SizeOption::Sizes([
                        sizes.0 as i64,
                        sizes.1 as i64,
                        sizes.2 as i64,
                    ]));
                }
            } else if linktype[iopt] == 2 {
                let slices = pip_get_two_integers(option, (0, 0)).unwrap_or((0, 0));
                let slices = Some((slices.0 as i64, slices.1 as i64));
                if iopt == 0 {
                    botlist[index] = slices;
                } else {
                    toplist[index] = slices;
                }
            } else {
                let bval = pip_get_boolean(option, 0).unwrap_or(0);
                if bval != 0 {
                    if *option == "FullSizeRotation" {
                        sizeoplist[index] = Some(SizeOption::Full);
                    } else if *option == "MaxXYsizeForRotation" {
                        sizeoplist[index] = Some(SizeOption::MaxXY);
                    } else if *option == "FlipYandZ" {
                        ifrotlist[index] = bval;
                    } else {
                        didrotlist[index] = bval;
                    }
                }
            }
        }
    }

    // Check the bottom and top entries now that all is known
    if pip_number_of_entries("BottomSlices").unwrap_or(0) as usize != ntomo - 1
        || botlist[0].is_some()
    {
        exit_error("You must enter bottom slices for all but the first tomogram");
    }
    if pip_number_of_entries("TopSlices").unwrap_or(0) as usize != ntomo - 1
        || toplist[ntomo - 1].is_some()
    {
        exit_error("You must enter top slices for all but the last tomogram");
    }
    if scaleref < 1 || scaleref > ntomo as i64 {
        exit_error("The tomogram number as reference for scaling is out of range");
    }
    scaleref -= 1;

    // Loop on tomos
    let mut nxmax: i64 = 0;
    let mut nymax: i64 = 0;
    let mut fliplist: Vec<String> = Vec::new();
    let mut invertlist = String::new();
    let mut sizelist: Vec<[i64; 3]> = Vec::new();
    let mut anglelist: Vec<[f64; 3]> = Vec::new();
    let mut chunklist = String::new();
    let mut avglist: Vec<i64> = Vec::new();
    let mut botranges: Vec<Option<String>> = vec![None; ntomo];
    let mut topranges: Vec<Option<String>> = vec![None; ntomo];
    // `ifinvert` is a module-level name that each call reassigns
    let mut ifinvert = 0;
    for itomo in 0..ntomo {
        let mut rotsize: [i64; 3] = [0, 0, 0];
        let mut rotangle: [f64; 3] = [0., 0., 0.];
        let namein = reclist[itomo].clone();
        let ifflip = ifrotlist[itomo];

        // Decompose filename into header, root and tail
        let (dirname, tail) = match namein.rfind('/') {
            Some(index) => {
                let head = &namein[..index + 1];
                let head = if head.chars().all(|c| c == '/') {
                    head
                } else {
                    head.trim_end_matches('/')
                };
                (head.to_owned(), namein[index + 1..].to_owned())
            }
            None => (String::new(), namein.clone()),
        };
        let mut header = String::new();
        if dirname.is_empty() {
            if !defdir.is_empty() {
                header = format!("{defdir}/");
            }
        } else {
            header = format!("{dirname}/");
        }
        let (root, ext) = super::imodpy::os_path_splitext(&tail);

        // Compose filenames for source and flipped files, get size
        let recsource = if ext.is_empty() {
            format!("{header}{root}.{defsrcext}")
        } else {
            format!("{header}{tail}")
        };

        let (nx, mut ny, mut nz): (i64, i64, i64);
        if Path::new(&recsource).exists() {
            match get_mrc_size(&recsource) {
                Ok((x, y, z)) => (nx, ny, nz) = (x as i64, y as i64, z as i64),
                Err(_) => exit_from_imod_error(progname),
            }
        } else {
            (nx, ny, nz) = (1024, 1024, 100);
            prnstr(
                &format!(
                    "WARNING: {recsource} NOT FOUND; SETTING SIZE TO {nx} {ny} {nz} FOR TEST PURPOSES"
                ),
                "\n",
                false,
            );
        }

        let mut bigrot = 0;
        let flipsource;
        if ifflip != 0 {
            // If flipping, then need a rec source and flip source
            flipsource = dataset_filename(&format!(".{deftmpext}"), Some(&root), None);

            // If flipping, or if ny < nz and rotating, swap ny and nz
            if ny < nz {
                bigrot = 1;
            }
            if ifflip == 1 || bigrot != 0 {
                (ny, nz) = (nz, ny);
            }
        } else {
            // If already flipped, set flip source same (so it says..., is this right?)
            flipsource = recsource.clone();
        }

        let mut nxformax = nx;
        let mut nyformax = ny;
        let mut newnz = nz;

        // If doing rotatevol, get the size of output file and the angles
        if ifflip == 2 {
            rotangle = angleoplist[itomo].unwrap_or([0., 0., 0.]);
            match &sizeoplist[itomo] {
                None => rotsize = [nx, ny, nz],
                Some(option @ (SizeOption::MaxXY | SizeOption::Full)) => {
                    let rotcom = vec![
                        "QuerySizeNeeded".to_owned(),
                        fmtstr(
                            "RotationAnglesZYX {},{},{}",
                            &[
                                py_str_float(rotangle[2]),
                                py_str_float(rotangle[1]),
                                py_str_float(rotangle[0]),
                            ],
                        ),
                        format!("InputFile {recsource}"),
                    ];
                    let rotout =
                        match run_cmd("rotatevol -StandardInput", Some(&rotcom), None, None, &[]) {
                            Ok(lines) => lines.unwrap_or_default(),
                            Err(_) => exit_from_imod_error(progname),
                        };
                    if rotout.is_empty() || rotout[0].split_whitespace().count() < 3 {
                        exit_error(&format!("rotatevol size query failed on {recsource}"));
                    }
                    let rotsplit: Vec<&str> = rotout[0].split_whitespace().collect();
                    for (slot, text) in rotsize.iter_mut().zip(&rotsplit) {
                        *slot = py_int(text).unwrap_or_else(|| {
                            python_uncaught(&format!(
                                "ValueError: invalid literal for int() with base 10: '{text}'"
                            ))
                        });
                    }
                    if matches!(option, SizeOption::MaxXY) {
                        rotsize[2] = nz;
                    }
                    prnstr(
                        &format!(
                            "Dimensions of {flipsource} will be {} {} {}",
                            rotsize[0], rotsize[1], rotsize[2]
                        ),
                        "\n",
                        false,
                    );
                }
                Some(SizeOption::Sizes(sizes)) => rotsize = *sizes,
            }

            newnz = rotsize[2];
            nxformax = rotsize[0];
            nyformax = rotsize[1];
        }

        // Keep track of maximum nx, ny
        nxmax = nxmax.max(nxformax);
        nymax = nymax.max(nyformax);

        // A missing entry for a tomogram is a TypeError in the source
        let zentry = |entry: Option<(i64, i64)>| -> (i64, i64) {
            entry.unwrap_or_else(|| {
                python_uncaught("TypeError: 'NoneType' object is not subscriptable")
            })
        };

        // Get the slices to extract from bottom of section
        let mut chunksize = 0;
        let mut botrange = None;
        if itomo != 0 {
            let (size, range, invert) = get_adjusted_z_range(
                zentry(botlist[itomo]),
                itomo,
                nz,
                newnz,
                rotangle,
                ifflip,
                bigrot,
                "bottom",
            );
            (chunksize, ifinvert) = (size, invert);
            botrange = Some(range);
            avglist.push(chunksize);
            chunklist += &format!(",{chunksize}");
        }

        // Get slice numbers to extract at top
        let mut toprange = None;
        if itomo < ntomo - 1 {
            let (chunkadd, range, invert) = get_adjusted_z_range(
                zentry(toplist[itomo]),
                itomo,
                nz,
                newnz,
                rotangle,
                ifflip,
                bigrot,
                "top",
            );
            ifinvert = invert;
            toprange = Some(range);
            avglist.push(chunkadd);
            chunksize += chunkadd;
            if itomo != 0 {
                chunklist += ",";
            }
            chunklist += &chunkadd.to_string();
        }
        let _ = chunksize;

        // Add all parameters to lists
        reclist[itomo] = recsource;
        fliplist.push(flipsource);
        botranges[itomo] = botrange;
        topranges[itomo] = toprange;
        sizelist.push(rotsize);
        anglelist.push(rotangle);
        if itomo != 0 {
            invertlist += " ";
        }
        invertlist += &ifinvert.to_string();
    }

    // Work out whether squeezing is needed, set up transforms

    let xymax = nxmax.max(nymax) as f64;
    let mut ifsquoze = 0;
    let mut newsize = format!("{nxmax},{nymax}");
    let mut expand = 0.0_f64;
    if xymax > midas_limit as f64 {
        ifsquoze = 1;
        let squeeze = midas_limit as f64 / xymax;
        expand = xymax / midas_limit as f64;
        let (newx, newy) = run_goodframe(
            (squeeze * nxmax as f64) as i32,
            (squeeze * nymax as f64) as i32,
        );
        if newx < 0 {
            exit_error("Getting new size for squeezed slices from goodframe");
        }
        newsize = format!("{newx},{newy}");
        make_backup_file(&format!("{joinroot}.sqzxf"));
        make_backup_file(&format!("{joinroot}.xpndxf"));
        let _ = write_text_file(
            &format!("{joinroot}.sqzxf"),
            &[format!("{squeeze:.7} 0. 0. {squeeze:.7} 0. 0.")],
            false,
        );
        let _ = write_text_file(
            &format!("{joinroot}.xpndxf"),
            &[format!("{expand:.7} 0. 0. {expand:.7} 0. 0.")],
            false,
        );
    }

    // Get the scaling
    let mut newmode = 0;
    let mut scalelist: Vec<String> = Vec::new();
    for itomo in 0..ntomo {
        let mut scaling = ["1.".to_owned(), "0.".to_owned()];
        if itomo as i64 == scaleref {
            match get_mrc(&reclist[itomo], false, false) {
                Ok(MrcInfo::Basic(_, _, _, mode, _, _, _)) => newmode = mode,
                Ok(_) => {}
                Err(_) => exit_from_imod_error(progname),
            }
        } else if test_mode == 0 {
            prnstr(
                &format!("Determining scaling for tomogram # {} ...", itomo + 1),
                "\n",
                false,
            );
            // Direct call (owner rule, 2026-09-26) for `runcmd('densmatch
            // -StandardInput', ['ReportOnly', 'ReferenceFile ...', 'ScaledFile
            // ...'])` and the parse of its "Scale factor" line: the two fields
            // that line prints with `G14.6`, split on white space.
            let params = DensmatchParams {
                pip_input: true,
                target_mean_sd: None,
                reference_file: Some(reclist[scaleref as usize].clone()),
                scaled_file: reclist[itomo].clone(),
                output_file: String::from(" "),
                report_only: true,
                x_min_max: None,
                y_min_max: None,
                z_min_max: None,
                use_all_pixels: false,
                offset: None,
                mode: None,
                files_opened: false,
            };
            let result = match call_own_program(
                "densmatch -StandardInput",
                &["densmatch"],
                None,
                true,
                move || {
                    parse_input_params::pip_exit_on_error(0, "ERROR: DENSMATCH - ");
                    densmatch_compute(&params)
                },
            ) {
                Ok((Some(result), _lines)) => result,
                _ => exit_from_imod_error(progname),
            };
            let line = format!(
                "{}{}",
                densmatch_g_edit(result.scale_fac, 14, 6),
                densmatch_g_edit(result.add_fac, 14, 6)
            );
            let sctmp: Vec<&str> = line.split_whitespace().collect();
            if sctmp.len() == 2 {
                scaling = [sctmp[0].to_owned(), sctmp[1].to_owned()];
            } else {
                prnstr(
                    &format!(
                        "WARNING: Densmatch failed for tomogram # {} ; using 1.,0. for scaling",
                        itomo + 1
                    ),
                    "\n",
                    false,
                );
            }
        }

        scalelist.push(format!("{},{}", scaling[0], scaling[1]));
    }

    // Output info file with filenames
    let infofile = format!("{joinroot}.info");
    make_backup_file(&infofile);
    let mut allscale = scalelist[0].clone();
    for scale in &scalelist[1..] {
        allscale += &format!(" {scale}");
    }

    let mut infolines = vec![
        format!("{ntomo}  {ifsquoze}  {nxmax}  {nymax} {VERSION} {newmode} {name_style}"),
        invertlist,
        allscale,
    ];
    for flip in &fliplist {
        infolines.push(flip.clone());
    }
    let _ = write_text_file(&infofile, &infolines, false);

    // COMPOSE THE COMMAND FILE
    make_backup_file(&joincom);
    let mut joinlines: Vec<String> = Vec::new();
    let sample_file = dataset_filename(".sample", None, None);
    let samp_avg_file = dataset_filename(".sampavg", None, None);
    add_output_format_var_to_lines(&mut joinlines, name_style, None, false);

    // Loop through the files, flip or rotate if necessary
    for itomo in 0..ntomo {
        if ifrotlist[itomo] == 1 {
            joinlines.push(format!(
                "$clip flipyz \"{}\" {}",
                reclist[itomo], fliplist[itomo]
            ));
        } else if ifrotlist[itomo] == 2 && didrotlist[itomo] == 0 {
            joinlines.extend([
                "$rotatevol -StandardInput".to_owned(),
                format!("InputFile {}", reclist[itomo]),
                format!("OutputFile {}", fliplist[itomo]),
                format!(
                    "OutputSizeXYZ {},{},{}",
                    sizelist[itomo][0], sizelist[itomo][1], sizelist[itomo][2]
                ),
                format!(
                    "RotationAnglesZYX {},{},{}",
                    py_str_float(anglelist[itomo][2]),
                    py_str_float(anglelist[itomo][1]),
                    py_str_float(anglelist[itomo][0])
                ),
            ]);
        }
    }

    // Now create the newstack command
    joinlines.push("$newstack -StandardInput".to_owned());
    for itomo in 0..ntomo {
        joinlines.push(format!("InputFile {}", fliplist[itomo]));
        let top = topranges[itomo].clone().unwrap_or_default();
        let bot = botranges[itomo].clone().unwrap_or_default();
        if itomo == 0 {
            joinlines.push(format!("SectionsToRead {top}"));
        } else if itomo == ntomo - 1 {
            joinlines.push(format!("SectionsToRead {bot}"));
        } else {
            joinlines.push(format!("SectionsToRead {bot},{top}"));
        }
        joinlines.push(format!("MultiplyAndAdd {}", scalelist[itomo]));
    }

    joinlines.extend([
        format!("ModeToOutput {newmode}"),
        format!("OutputFile {sample_file}"),
        format!("SizeToOutputInXandY {newsize}"),
    ]);
    if ifsquoze != 0 {
        joinlines.push(format!("ShrinkByFactor {expand:.7}"));
    }

    // Now make the average commands
    let mut startsec: i64 = 0;
    let mut tmplist: Vec<String> = Vec::new();
    for (ind, &numavg) in avglist.iter().enumerate() {
        let endsec = startsec + numavg - 1;
        let tmpfile = format!("{joinroot}.tmpavg.{}", ind + 1);
        joinlines.extend([
            "$avgstack".to_owned(),
            sample_file.clone(),
            tmpfile.clone(),
            format!("{startsec},{endsec}"),
        ]);
        startsec += numavg;
        tmplist.push(tmpfile);
    }

    joinlines.extend([
        "$newstack -StandardInput".to_owned(),
        format!("OutputFile {samp_avg_file}"),
    ]);
    for tmpfile in &tmplist {
        joinlines.push(format!("InputFile {tmpfile}"));
    }

    joinlines.push(format!("$\\rm -f {joinroot}.tmpavg.*"));
    let _ = write_text_file(&joincom, &joinlines, false);

    prnstr(
        &format!(
            "\nThe command file {joincom} has been written and is ready to submit.
Run it to create the files {sample_file} and {samp_avg_file}
For optional automatic alignment of the top/bottom averages, try:
   xfalign -tomo -pre {samp_avg_file} {joinroot}.xf
Then run:
   midas -cs {chunklist} {sample_file} {joinroot}.xf
to run midas in chunk alignment mode and create or edit the transform file
For optional automatic refinement of transforms, then try:
   xfalign -tomo -ini {joinroot}.xf {samp_avg_file} {joinroot}.xf
(after which you should check results with the midas command again).
Finally run finishjoin to create the joined tomogram"
        ),
        "\n",
        false,
    );

    done(0)
}
