//! Single-binary busybox-style launcher for every translated IMOD command.
//!
//! Upstream IMOD installs one executable per command.  This crate deliberately
//! deviates: it builds exactly one binary, `imod`, which dispatches to the
//! translated program unit.  Dispatch is on `basename(argv[0])` first, so an
//! IMOD-style install that symlinks (or hardlinks) `header`, `newstack`, … to
//! `imod` gives each program the same `argv` it has always had — same PIP
//! parsing, same `imodProgName`-derived error prefixes.
//!
//! When `argv[0]` does not name a command, `argv[1]` is taken as the
//! subcommand (`imod header -size f.mrc`).  That form re-execs `/proc/self/exe`
//! with `argv[0]` rewritten to the command name, so the translated code — which
//! reads `std::env::args()` directly, exactly as the C/Fortran sources read
//! `argv` — observes `["…/header", "-size", "f.mrc"]` and never sees the
//! `imod` wrapper.  No translated unit is aware of the launcher.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Every command this binary can run, in the order the usage listing prints
/// them.  Kept sorted so the listing is stable.
const COMMANDS: &[&str] = &[
    "3dmod",
    "3dmodv",
    "alterheader",
    "batchruntomo",
    "binvol",
    "clip",
    "convertmod",
    "dm3props",
    "etomo",
    #[cfg(feature = "gui")]
    "etomo-gui",
    "header",
    "imodinfo",
    "imodjoin",
    "imodqtassist",
    "imodsendevent",
    "midas",
    "mrc2tif",
    "mrcinfo",
    "mrctaper",
    "mrctilt",
    "newstack",
    "processchunks",
    "sourcedoc",
    "subm",
    "submfg",
    "tif2mrc",
    "trimvol",
    "wmod2imod",
];

/// Runs the named command, or returns `false` when the name is not a command.
///
/// Each arm is the body the corresponding `src/bin/<command>.rs` shim used to
/// carry, unchanged.
fn dispatch(name: &str) -> bool {
    match name {
        // `3dmodv` is the same program under the name `imod.cpp::main` tests
        // with `imodv = program.ends_with('v')`, exactly as upstream links it.
        "3dmod" | "3dmodv" => {
            // `imod.cpp`'s `App`/`ImodHelp` globals and `imodv.cpp`'s Qt
            // objects are what these two hosts stand for.  They are installed
            // before `main` runs, as the C++ constructs them at file scope and
            // in `main`.
            imod_rs::imod::three_dmod::imod::IMOD_NATIVE_BOUNDARY.with(|slot| {
                *slot.borrow_mut() = Some(Box::new(
                    imod_rs::imod::three_dmod::imod::ImodNativeHost::default(),
                ));
            });
            #[cfg(feature = "three-dmod-gl")]
            imod_rs::imod::three_dmod::imodv::IMODV_NATIVE_BOUNDARY.with(|slot| {
                *slot.borrow_mut() = Some(Box::new(
                    imod_rs::imod::three_dmod::mv_window::ImodvNativeHost,
                ));
            });
            let arguments = std::env::args().collect::<Vec<_>>();
            match imod_rs::imod::three_dmod::imod::imod_main(&arguments) {
                Ok(status) => std::process::exit(status),
                Err(message) => {
                    eprintln!("{message}");
                    std::process::exit(1);
                }
            }
        }
        "alterheader" => imod_rs::imod::flib::image::alterheader::alterheader(),
        "batchruntomo" => std::process::exit(imod_rs::imod::pysrc::batchruntomo::batchruntomo(
            &std::env::args_os().collect::<Vec<_>>(),
        )),
        "binvol" => imod_rs::imod::flib::image::binvol::binvol(),
        "clip" => imod_rs::imod::clip::clip::clip(),
        "convertmod" => imod_rs::imod::flib::model::convertmod::convertmod(),
        "dm3props" => std::process::exit(imod_rs::imod::mrc::dm3props::dm3props(
            &std::env::args().collect::<Vec<_>>(),
        )),
        "etomo" => std::process::exit(imod_rs::imod::pysrc::etomo::etomo(
            &std::env::args_os().collect::<Vec<_>>(),
        )),
        #[cfg(feature = "gui")]
        "etomo-gui" => {
            let arguments = std::env::args().skip(1).collect::<Vec<_>>();
            let mut director = imod_rs::imod::etomo::etomo_director::EtomoDirector::new();
            if let Err(error) = director.main_gui(&arguments) {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        "header" => imod_rs::imod::flib::image::header::header(),
        "imodinfo" => imod_rs::imod::imodutil::imodinfo::imodinfo(),
        "imodjoin" => imod_rs::imod::imodutil::imodjoin::imodjoin(),
        "imodqtassist" => {
            let arguments = std::env::args().collect::<Vec<_>>();
            std::process::exit(
                imod_rs::imod::qttools::qtassist::imodqtassist::imodqtassist(&arguments),
            )
        }
        "imodsendevent" => {
            let arguments = std::env::args().collect::<Vec<_>>();
            std::process::exit(
                imod_rs::imod::qttools::sendevent::imodsendevent::imodsendevent(&arguments),
            )
        }
        "midas" => {
            let arguments = std::env::args().collect::<Vec<_>>();
            match imod_rs::imod::midas::midas::midas_main(&arguments) {
                Ok(status) => std::process::exit(status),
                Err(message) => {
                    eprintln!("{message}");
                    std::process::exit(1);
                }
            }
        }
        "mrc2tif" => imod_rs::imod::qttools::mrc2tif::mrc2tif::mrc2tif(),
        "mrcinfo" => {
            let arguments = std::env::args().collect::<Vec<_>>();
            std::process::exit(imod_rs::imod::mrc::mrcinfo::mrcinfo(&arguments))
        }
        "mrctaper" => {
            let arguments = std::env::args().collect::<Vec<_>>();
            std::process::exit(imod_rs::imod::mrc::mrctaper::mrctaper(&arguments))
        }
        "mrctilt" => {
            let arguments = std::env::args().collect::<Vec<_>>();
            std::process::exit(imod_rs::imod::mrc::mrctilt::mrctilt(&arguments))
        }
        "newstack" => imod_rs::imod::flib::image::newstack::newstack(),
        "processchunks" => {
            let arguments = std::env::args().collect::<Vec<_>>();
            std::process::exit(
                imod_rs::imod::qttools::processchunks::processchunks::processchunks(&arguments),
            )
        }
        "sourcedoc" => {
            let arguments = std::env::args().collect::<Vec<_>>();
            std::process::exit(imod_rs::imod::qttools::sourcedoc::sourcedoc::sourcedoc(
                &arguments,
            ))
        }
        "subm" => std::process::exit(imod_rs::imod::pysrc::subm::subm(
            &std::env::args_os().collect::<Vec<_>>(),
        )),
        "submfg" => std::process::exit(imod_rs::imod::pysrc::submfg::submfg(
            &std::env::args_os().collect::<Vec<_>>(),
        )),
        "tif2mrc" => {
            let arguments = std::env::args().collect::<Vec<_>>();
            std::process::exit(imod_rs::imod::mrc::tif2mrc::tif2mrc(&arguments))
        }
        "trimvol" => std::process::exit(imod_rs::imod::flib::image::trimvol::trimvol()),
        "wmod2imod" => imod_rs::imod::imodutil::wmod2imod::wmod2imod(),
        _ => return false,
    }
    true
}

/// Prints the command listing.  Used for no arguments, `-h`/`--help`, and an
/// unrecognised subcommand alike; the caller exits 1 in every case.
fn usage(launcher: &str) {
    eprintln!("Usage: {launcher} <command> [options ...]");
    eprintln!();
    eprintln!(
        "Every IMOD command translated by this crate is built into this single\n\
         binary.  Run one either as a subcommand, as above, or through a link\n\
         named after it (an IMOD-style install links each command name to this\n\
         binary, and the command then behaves exactly as its own executable)."
    );
    eprintln!();
    eprintln!("Commands:");
    for command in COMMANDS {
        eprintln!("  {command}");
    }
}

fn main() {
    let argv0 = std::env::args_os().next().unwrap_or_default();

    // 1. `basename(argv[0])` names a command: run it with `argv` untouched.
    if let Some(base) = Path::new(&argv0).file_name().and_then(|n| n.to_str())
        && dispatch(base)
    {
        return;
    }

    let launcher = Path::new(&argv0)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("imod")
        .to_string();

    // 2. `argv[1]` names a command: re-exec ourselves with `argv[0]` rewritten
    //    so the program sees the `argv` it would have had as its own binary.
    let arguments: Vec<OsString> = std::env::args_os().collect();
    let subcommand = match arguments.get(1).and_then(|a| a.to_str()) {
        Some(name) if COMMANDS.contains(&name) => name,
        _ => {
            usage(&launcher);
            std::process::exit(1);
        }
    };

    // The rewritten `argv[0]` keeps the directory of the running binary, so it
    // is the path a command link would have: `<bindir>/newstack`, not a bare
    // name.  That is what `imodProgName` and PIP's program name see.
    let new_argv0: PathBuf = match std::env::current_exe() {
        Ok(path) => path.with_file_name(subcommand),
        Err(_) => PathBuf::from(subcommand),
    };

    use std::os::unix::process::CommandExt;
    let error = std::process::Command::new("/proc/self/exe")
        .arg0(&new_argv0)
        .args(&arguments[2..])
        .exec();
    eprintln!("{launcher}: cannot run {subcommand}: {error}");
    std::process::exit(1);
}
