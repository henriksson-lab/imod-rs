//! Translation of `IMOD/pysrc/runraptor`: runs RAPTOR on a raw or
//! coarse-aligned stack and post-processes the fiducial model it writes.
//!
//! A Python command script with no functions; its top level is
//! [`runraptor`], translated statement by statement.
//!
//! How RAPTOR itself is run.  The source runs `"$raptorBin/RAPTOR"` through
//! `runcmd` (`sh -c`), where `raptorBin` is `$IMOD_DIR/bin` unless
//! `RAPTOR_BIN` names another directory.  With `RAPTOR_BIN` set this does
//! exactly that.  Otherwise, when this crate's command table has a `RAPTOR`
//! entry, our own program runs instead (owner decision 2026-09-24, "Our own
//! commands are called in process"): in this process when the entry allows
//! it (`runcmd` routes a command line whose first word is a table entry
//! there), else as a child of our own `imod` binary.  Without such an entry,
//! or when this process is not the `imod` launcher, the source's
//! `$IMOD_DIR/bin/RAPTOR` runs.  The `-exec` directory RAPTOR is given is
//! `raptorBin` in every case: RAPTOR runs `MarkersCorrespond` from it
//! (`correspondence.cpp:231`).
//!
//! `getmrc` runs `header` in process ([`get_mrc`]) and `xfmodel` goes
//! through [`run_cmd`], which runs our own `xfmodel` in process; neither
//! program's printed output is parsed here beyond what `getmrc` already
//! reproduces.
//!
//! Two upstream defects are fixed (BUGS.md, "runraptor"): the raw-stack test
//! compares the extension *with* its dot against `standardTypeExtensions()`,
//! whose entries have none, so a raw `.mrc`/`.hdf` stack was taken for an
//! aligned one; and a blank line after `newstack -StandardInput` in
//! `prenewst.com` indexed an empty `split()` and died with an IndexError.

use super::imodpy::py_str_float;
use super::imodpy::{
    MrcInfo, add_imod_bin_ignore_sighup, cleanup_files, exit_from_imod_error, fmtstr, get_mrc,
    make_backup_file, os_path_splitext, print_pid, prnstr, read_text_file, run_cmd,
    standard_type_extensions,
};
use super::pip::{exit_error, set_exit_prefix};
use super::pysed::{PysedSrc, pysed};
use std::ffi::OsString;
use std::io::Write as _;
use std::path::Path;

/// The script's top level (`runraptor:8-180`).  Returns the status of its
/// `sys.exit` calls; error paths exit the process as `exitError` and
/// `exitFromImodError` do.
pub fn runraptor(arguments: &[OsString]) -> i32 {
    let progname = "runraptor";
    let prefix = format!("ERROR: {progname} - ");
    let argv: Vec<String> = arguments
        .iter()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();

    //
    // Setup runtime environment
    let mut raptor_bin;
    match std::env::var("IMOD_DIR") {
        Ok(imod_dir) => {
            add_imod_bin_ignore_sighup();
            // `os.path.join(os.environ['IMOD_DIR'], 'bin')`
            raptor_bin = if imod_dir.is_empty() || imod_dir.ends_with('/') {
                format!("{imod_dir}bin")
            } else {
                format!("{imod_dir}/bin")
            };
        }
        Err(_) => {
            print!("{prefix} IMOD_DIR is not defined!\n");
            let _ = std::io::stdout().flush();
            return 1;
        }
    }

    //
    // load IMOD Libraries
    set_exit_prefix(prefix);

    // `if os.getenv('RAPTOR_BIN')`: set and not empty
    let raptor_bin_env = std::env::var("RAPTOR_BIN")
        .ok()
        .filter(|dir| !dir.is_empty());
    if let Some(dir) = &raptor_bin_env {
        raptor_bin = dir.clone();
    }
    if !Path::new(&raptor_bin).exists() {
        exit_error(&format!(
            "{raptor_bin}does not exist.   You need to leave RAPTOR_BIN undefined to use the RAPTOR in IMOD, or define it with the path to a RAPTOR executable"
        ));
    }

    if argv.len() < 5 {
        prnstr(
            "Usage: runraptor [options] image_stack
   Runs RAPTOR on the given image_stack, which may be a raw stack or preali file
   Options other than -P[ID] for a PID are RAPTOR options and MUST include:
     -diam #  Bead diameter in pixels
     -mark #  Number of markers to track

   This script takes care of adding -exec, -path, -input, -track, and -output
       options.
   It must be run inside the data directory where files are located.
   Coarse alignment must be run first",
            "\n",
            false,
        );
        let _ = std::io::stdout().flush();
        return 1;
    }

    // Set up a unique directory for the run
    let mut outdir = "raptor1".to_owned();
    for i in 1..100 {
        if !Path::new(&format!("raptor{i}")).exists() {
            outdir = format!("raptor{i}");
            break;
        }
    }

    // Get options and build option string for raptor
    let mut options: Vec<String> = Vec::new();
    let mut gotmark = 0;
    let mut gotdiam = 0;
    for i in 1..argv.len() {
        // `argv[i] in '-PID'` is a substring test
        if argv[i].starts_with("-P") && "-PID".contains(argv[i].as_str()) {
            print_pid(true);
        } else {
            options.push(argv[i].clone());
        }
        if argv[i].starts_with("-ma") {
            gotmark = 1;
        }
        if argv[i].starts_with("-di") {
            gotdiam = 1;
        }
    }

    let infile = argv[argv.len() - 1].clone();

    // The extension from splitext includes the . !
    let (mut rootname, ext) = os_path_splitext(&infile);
    // Defined behaviour (BUGS.md, "fixed in translation"): the source tests
    // `ext in standardTypeExtensions()`, but `ext` keeps its dot and the
    // list's entries ('', 'mrc', 'hdf') have none, so only a file with no
    // extension matched and a raw `.mrc`/`.hdf` stack was treated as an
    // aligned one.  The extension is compared without its dot, which keeps
    // the no-extension case and matches `.mrc` and `.hdf` as intended.
    let ext_no_dot = ext.strip_prefix('.').unwrap_or(&ext);
    let useraw = ext == ".st"
        || ((standard_type_extensions()
            .iter()
            .any(|standard| standard == ext_no_dot)
            || ext == ".tif"
            || ext == ".tiff")
            && !rootname.ends_with("_preali"));
    if rootname.ends_with("_preali") {
        rootname.truncate(rootname.len() - 7);
    }
    // `os.path.basename(rootname) != rootname`
    if rootname.contains('/') {
        exit_error("You must run this script in the data directory");
    }
    if !Path::new(&infile).exists() {
        exit_error(&format!(
            "The input file {infile} does not exist in the current directory"
        ));
    }
    if useraw && !Path::new(&format!("{rootname}.prexg")).exists() {
        exit_error("Should be run on a raw stack only after running coarse alignment");
    }

    if gotmark == 0 || gotdiam == 0 {
        exit_error("You must enter options with a diameter in pixels and a number of markers");
    }

    let outname = format!("{rootname}_raptor.fid");

    // For a raw stack, figure out distortion parameters to send to xfmodel
    let mut distort = String::new();
    let mut binning = String::new();
    let mut gradient = String::new();
    let mut got_newst = false;
    if useraw {
        for comf in ["prenewst.com", "prenewsta.com"] {
            if Path::new(comf).exists() {
                let newstlines = read_text_file(comf, None, false, None).unwrap_or_default();
                for l in &newstlines {
                    if got_newst {
                        let lsplit: Vec<&str> = l.split_whitespace().collect();
                        // Defined behaviour (BUGS.md, "fixed in
                        // translation"): `lsplit[0]` of a blank line raised
                        // an IndexError the source does not catch; a blank
                        // line carries no option and is passed over.
                        if lsplit.is_empty() {
                            continue;
                        }
                        if lsplit[0] == "DistortionField" && lsplit.len() > 1 {
                            distort = format!("-distort {}", lsplit[1]);
                        }
                        if lsplit[0] == "ImagesAreBinned" && lsplit.len() > 1 {
                            binning = format!("-binning {}", lsplit[1]);
                        }
                        if lsplit[0] == "GradientFile" && lsplit.len() > 1 {
                            gradient = format!("-gradient {}", lsplit[1]);
                        }
                    } else if l.contains("newstack") {
                        if l.contains(infile.as_str()) {
                            let lsplit: Vec<&str> = l.split_whitespace().collect();
                            for ind in 0..lsplit.len().saturating_sub(1) {
                                if lsplit[ind].starts_with("-dist") {
                                    distort = format!("-distort {}", lsplit[ind + 1]);
                                }
                                if lsplit[ind].starts_with("-image") {
                                    binning = format!("-binning {}", lsplit[ind + 1]);
                                }
                                if lsplit[ind].starts_with("-grad") {
                                    gradient = format!("-gradient {}", lsplit[ind + 1]);
                                }
                            }
                            break;
                        } else if l.contains("-Standard") {
                            got_newst = true;
                        }
                    }
                }
            }
        }
    }

    // Build a command string with quoted arguments.  The program word is the
    // source's `"{raptorBin}/RAPTOR"` unless our own RAPTOR stands in for it
    // (see the module documentation).
    let own_raptor = if raptor_bin_env.is_none() {
        crate::imod::commands::find("RAPTOR")
    } else {
        None
    };
    let program = match own_raptor {
        Some(entry) if entry.in_process => "RAPTOR".to_owned(),
        Some(_) => match std::env::current_exe() {
            Ok(exe) if exe.file_name().is_some_and(|base| base == "imod") => {
                format!("\"{}\" RAPTOR", exe.display())
            }
            _ => format!("\"{raptor_bin}/RAPTOR\""),
        },
        None => format!("\"{raptor_bin}/RAPTOR\""),
    };
    let mut comstr = format!(
        "{program} {}",
        fmtstr(
            "-exec \"{0}\" -path . -inp {1} -track -out {2}",
            &[raptor_bin.clone(), infile.clone(), outdir.clone()],
        )
    );
    for opt in &options {
        comstr += &format!(" \"{opt}\"");
    }

    prnstr(
        &format!("RAPTOR results for this run will be in {outdir}"),
        "\n",
        false,
    );
    let _ = std::io::stdout().flush();
    if run_cmd(&comstr, None, None, None, &[]).is_err() {
        let logfile = format!("{outdir}/align/{rootname}_RAPTOR.log");
        if Path::new(&logfile).exists() {
            let loglines = read_text_file(&logfile, None, false, None).unwrap_or_default();
            for l in &loglines {
                if l.contains("ERROR:") {
                    prnstr(l, "\n", false);
                }
            }
        }
        exit_from_imod_error(progname);
    }

    make_backup_file(&outname);

    // Fix the pixel size and other features in the model
    let mut sedout = outname.clone();
    if useraw {
        sedout = format!("{rootname}.{}", std::process::id());
    }
    let (px, py, pz) = match get_mrc(&infile, false, false) {
        Ok(MrcInfo::Basic(_nx, _ny, _nz, _mode, px, py, pz)) => (px, py, pz),
        Ok(_) => unreachable!(),
        Err(_) => exit_from_imod_error(progname),
    };

    let sedcom: Vec<String> = vec![
        "/symbol *circle/d".to_owned(),
        "/^size *7/d".to_owned(),
        fmtstr(
            "/^units/a/refcurscale {} {} {}/",
            &[py_str_float(px), py_str_float(py), py_str_float(pz)],
        ),
        "/^drawmode/a/symbol    0/".to_owned(),
        "/^drawmode/a/symsize   7/".to_owned(),
    ];

    let sed_input = format!("{outdir}/IMOD/{rootname}.fid.txt");
    let _ = pysed(
        &sedcom,
        PysedSrc::File(&sed_input),
        Some(&sedout),
        false,
        '/',
        false,
    );

    // If did raw stack, now align the model to the preali
    if useraw {
        let command = fmtstr(
            "xfmodel -xf {}.prexg {} {} {} {} {}",
            &[
                rootname.clone(),
                distort,
                binning,
                gradient,
                sedout.clone(),
                outname.clone(),
            ],
        );
        if run_cmd(&command, None, None, None, &[]).is_err() {
            exit_from_imod_error(progname);
        }

        cleanup_files(&[sedout]);
    }

    prnstr(
        "The model that will load on the coarse aligned stack is:",
        "\n",
        false,
    );
    prnstr(&outname, "\n", false);
    let _ = std::io::stdout().flush();
    0
}
