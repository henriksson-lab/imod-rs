//! The command table: every command this crate provides, and its entry point.
//!
//! **Not a translated unit.**  Upstream IMOD installs one executable per
//! command, so it has no table of commands at all; this module is Rust-only
//! launcher plumbing, like `backends.rs`.  It exists because two places need
//! the same list and used to keep their own copies:
//!
//! * the `imod` binary (`src/bin/imod.rs`), the busybox-style launcher that
//!   dispatches on `basename(argv[0])` or `argv[1]` and prints the list in its
//!   usage message; and
//! * [`crate::imod::pysrc::imodpy::run_cmd`], which runs one of our own
//!   commands **in this process** through [`run_in_process`] rather than
//!   through `sh -c` (owner decision, 2026-09-24; see `CLAUDE.md`, "Our own
//!   commands are called in process").
//!
//! Each [`Command`] maps a name to an entry point with one signature,
//! `fn()`: a translated `main` that reads its `argv` through
//! [`program_args`]/[`program_args_os`] and ends either by returning (a
//! Fortran `end program`, status 0) or through [`b3dutil::exit`].  The entry
//! points whose Rust function takes `argv` and returns a status are wrapped
//! here, so the normalisation lives in exactly one place.
//!
//! [`b3dutil::exit`]: crate::imod::libcfshr::b3dutil::exit

use crate::imod::libcfshr::b3dutil::{exit, program_args, program_args_os};
use crate::imod::libiimod::iimage;
use crate::imod::libiimod::unit_fileio;
use std::ffi::OsString;
use std::sync::atomic::Ordering;

/// One command of the table.
pub struct Command {
    /// The name the command is installed under: a link name, the launcher's
    /// subcommand, and the first word of a `runcmd` command line.
    pub name: &'static str,
    /// The command's `main`.  It never returns a status: it returns (status
    /// 0) or ends in [`exit`].
    pub entry: fn(),
    /// Whether `imodpy::run_cmd` may run it in this process.  False for the
    /// GUI programs, for the Python-script translations (whose PIP and
    /// `imodpy` state are process-global statics a nested call would clobber),
    /// and for the process managers that are meant to run detached
    /// (`processchunks`, `subm`, `submfg`).  Those still run as a child
    /// process when a script names them.
    pub in_process: bool,
}

/// Every command the `imod` binary can run, in the order its usage listing
/// prints them.
pub const COMMANDS: &[Command] = &[
    cmd("3dmod", three_dmod, false),
    cmd("3dmodv", three_dmod, false),
    cmd(
        "alterheader",
        crate::imod::flib::image::alterheader::alterheader,
        true,
    ),
    cmd("batchruntomo", batchruntomo, false),
    cmd("beadtrack", beadtrack, true),
    cmd("binvol", crate::imod::flib::image::binvol::binvol, true),
    cmd(
        "blendmont",
        crate::imod::flib::blend::blendmont::blendmont,
        true,
    ),
    cmd("clip", crate::imod::clip::clip::clip, true),
    cmd(
        "convertmod",
        crate::imod::flib::model::convertmod::convertmod,
        true,
    ),
    cmd(
        "densmatch",
        crate::imod::flib::image::densmatch::densmatch,
        true,
    ),
    cmd("dm3props", dm3props, true),
    cmd("fakevolume", fakevolume, true),
    cmd(
        "findcontrast",
        crate::imod::flib::image::findcontrast::findcontrast,
        true,
    ),
    cmd("findsection", findsection, true),
    cmd("echo2", echo2, true),
    cmd("etomo", etomo, false),
    #[cfg(feature = "gui")]
    cmd("etomo-gui", etomo_gui, false),
    cmd("header", crate::imod::flib::image::header::header, true),
    cmd("imodinfo", crate::imod::imodutil::imodinfo::imodinfo, true),
    cmd("imodjoin", crate::imod::imodutil::imodjoin::imodjoin, true),
    cmd("imodmesh", crate::imod::imodutil::imodmesh::imodmesh, true),
    cmd(
        "imodtrans",
        crate::imod::imodutil::imodtrans::imodtrans,
        true,
    ),
    cmd("imodqtassist", imodqtassist, false),
    cmd("imodsendevent", imodsendevent, false),
    cmd("midas", midas, false),
    cmd("manageshrmem", manageshrmem, false),
    cmd(
        "mrc2tif",
        crate::imod::qttools::mrc2tif::mrc2tif::mrc2tif,
        true,
    ),
    cmd("mrcbyte", mrcbyte, true),
    cmd("mrcinfo", mrcinfo, true),
    cmd("raw2mrc", raw2mrc, true),
    cmd("modifymdoc", modifymdoc, true),
    cmd("mrclog", mrclog, true),
    cmd("mrctaper", mrctaper, true),
    cmd("mrctilt", mrctilt, true),
    cmd(
        "newstack",
        crate::imod::flib::image::newstack::newstack,
        true,
    ),
    cmd("processchunks", processchunks, false),
    cmd("sourcedoc", sourcedoc, true),
    cmd("subm", subm, false),
    cmd("submfg", submfg, false),
    cmd("tif2mrc", tif2mrc, true),
    cmd("tifinfo", tifinfo, true),
    cmd("tilt", tilt, true),
    cmd("tiltalign", tiltalign, true),
    cmd("tiltxcorr", tiltxcorr, true),
    cmd("trimvol", trimvol, false),
    cmd(
        "wmod2imod",
        crate::imod::imodutil::wmod2imod::wmod2imod,
        true,
    ),
    cmd("xfmodel", crate::imod::flib::model::xfmodel::xfmodel, true),
    cmd(
        "xfproduct",
        crate::imod::flib::image::xfproduct::xfproduct,
        true,
    ),
    cmd("xftoxg", crate::imod::flib::image::xftoxg::xftoxg, true),
];

const fn cmd(name: &'static str, entry: fn(), in_process: bool) -> Command {
    Command {
        name,
        entry,
        in_process,
    }
}

/// The table entry named `name`, if any.
pub fn find(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|command| command.name == name)
}

/// Runs `command` in this process with `argv` (whose `argv[0]` is the path a
/// command link would have) through [`crate::imod::libcfshr::b3dutil::run_in_process`],
/// and returns its exit status and, when `capture` is set, its standard
/// output.
///
/// Around that runner it resets the process-global state of `libiimod` that
/// a fresh process starts with and that the generic runner does not know
/// about -- `iimage.c`'s `sAllowMultiVolume`, and the `IMOD_TIFF_COMPRESSION`
/// variable `setTiffCompressionType` `putenv`s -- and restores the caller's
/// values afterwards.  On the command's thread, after its `main` has ended
/// (by returning or by [`exit`]), it deletes every image file still held by
/// that thread's Fortran unit table: a process ending would have released
/// them, and without this they stayed allocated for the rest of the caller's
/// run.  The files are closed without the scratch-file removal `iiuClose`
/// adds, since an exiting process does not run it either.
pub fn run_in_process(
    command: &'static Command,
    argv: Vec<OsString>,
    input: Option<&[u8]>,
    capture: bool,
) -> std::io::Result<(i32, Vec<u8>)> {
    let saved_multi_volume = iimage::S_ALLOW_MULTI_VOLUME.swap(0, Ordering::SeqCst);
    let saved_compression = std::env::var_os("IMOD_TIFF_COMPRESSION");
    let entry = command.entry;
    let result = crate::imod::libcfshr::b3dutil::run_in_process(argv, input, capture, move || {
        /// Releases the unit table's images when the command's `main` ends,
        /// whether it returned or unwound from [`exit`].
        struct ReleaseUnits;
        impl Drop for ReleaseUnits {
            fn drop(&mut self) {
                unit_fileio::release_all_units();
            }
        }
        let _release = ReleaseUnits;
        entry();
    });
    iimage::S_ALLOW_MULTI_VOLUME.store(saved_multi_volume, Ordering::SeqCst);
    // The runner thread has been joined, so no other thread reads the
    // environment while it is changed here.
    unsafe {
        match saved_compression {
            Some(value) => std::env::set_var("IMOD_TIFF_COMPRESSION", value),
            None => std::env::remove_var("IMOD_TIFF_COMPRESSION"),
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Entry points whose Rust function does not already have the `fn()` shape.
// Each body is the arm the launcher used to carry, with `std::process::exit`
// replaced by `b3dutil::exit` (identical outside the in-process runner).
// ---------------------------------------------------------------------------

/// `3dmodv` is the same program under the name `imod.cpp::main` tests with
/// `imodv = program.ends_with('v')`, exactly as upstream links it.
fn three_dmod() {
    // `imod.cpp`'s `App`/`ImodHelp` globals and `imodv.cpp`'s Qt objects are
    // what these two hosts stand for.  They are installed before `main` runs,
    // as the C++ constructs them at file scope and in `main`.
    crate::imod::three_dmod::imod::IMOD_NATIVE_BOUNDARY.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(
            crate::imod::three_dmod::imod::ImodNativeHost::default(),
        ));
    });
    #[cfg(feature = "three-dmod-gl")]
    crate::imod::three_dmod::imodv::IMODV_NATIVE_BOUNDARY.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(
            crate::imod::three_dmod::mv_window::ImodvNativeHost,
        ));
    });
    let arguments = program_args();
    match crate::imod::three_dmod::imod::imod_main(&arguments) {
        Ok(status) => exit(status),
        Err(message) => {
            eprintln!("{message}");
            exit(1);
        }
    }
}

fn batchruntomo() {
    exit(crate::imod::pysrc::batchruntomo::batchruntomo(
        &program_args_os(),
    ))
}

fn beadtrack() {
    exit(crate::imod::flib::beadtrack::beadtrack::beadtrack(
        &program_args(),
    ))
}

fn dm3props() {
    exit(crate::imod::mrc::dm3props::dm3props(&program_args()))
}

fn fakevolume() {
    exit(crate::imod::mrc::fakevolume::fakevolume(&program_args()))
}

fn findsection() {
    exit(crate::imod::imodutil::findsection::findsection(
        &program_args(),
    ))
}

fn echo2() {
    exit(crate::imod::imodutil::echo2::echo2(&program_args()))
}

fn etomo() {
    exit(crate::imod::pysrc::etomo::etomo(&program_args_os()))
}

#[cfg(feature = "gui")]
fn etomo_gui() {
    let arguments = program_args().into_iter().skip(1).collect::<Vec<_>>();
    let mut director = crate::imod::etomo::etomo_director::EtomoDirector::new();
    if let Err(error) = director.main_gui(&arguments) {
        eprintln!("{error}");
        exit(1);
    }
}

fn imodqtassist() {
    exit(crate::imod::qttools::qtassist::imodqtassist::imodqtassist(
        &program_args(),
    ))
}

fn imodsendevent() {
    exit(crate::imod::qttools::sendevent::imodsendevent::imodsendevent(&program_args()))
}

fn midas() {
    match crate::imod::midas::midas::midas_main(&program_args()) {
        Ok(status) => exit(status),
        Err(message) => {
            eprintln!("{message}");
            exit(1);
        }
    }
}

fn manageshrmem() {
    exit(crate::imod::mrc::manageshrmem::manageshrmem(&program_args()))
}

fn mrcbyte() {
    exit(crate::imod::mrc::mrcbyte::mrcbyte(&program_args()))
}

fn raw2mrc() {
    exit(crate::imod::mrc::raw2mrc::raw2mrc(&program_args()))
}

fn mrcinfo() {
    exit(crate::imod::mrc::mrcinfo::mrcinfo(&program_args()))
}

fn modifymdoc() {
    exit(crate::imod::mrc::modifymdoc::modifymdoc(&program_args()))
}

fn mrclog() {
    exit(crate::imod::mrc::mrclog::mrclog(&program_args()))
}

fn mrctaper() {
    exit(crate::imod::mrc::mrctaper::mrctaper(&program_args()))
}

fn mrctilt() {
    exit(crate::imod::mrc::mrctilt::mrctilt(&program_args()))
}

fn processchunks() {
    // Unix-only: the scheduler drives chunks over ssh and batch queues.
    #[cfg(unix)]
    exit(crate::imod::qttools::processchunks::processchunks::processchunks(&program_args()));
    #[cfg(not(unix))]
    {
        eprintln!("ERROR: processchunks - not available on this platform");
        exit(1)
    }
}

fn sourcedoc() {
    exit(crate::imod::qttools::sourcedoc::sourcedoc::sourcedoc(
        &program_args(),
    ))
}

fn subm() {
    exit(crate::imod::pysrc::subm::subm(&program_args_os()))
}

fn submfg() {
    exit(crate::imod::pysrc::submfg::submfg(&program_args_os()))
}

fn tilt() {
    exit(crate::imod::flib::tilt::tilt::tilt(&program_args()))
}

fn tiltalign() {
    exit(crate::imod::flib::tiltalign::tiltalign::tiltalign(
        &program_args(),
    ))
}

fn tiltxcorr() {
    exit(crate::imod::imodutil::tiltxcorr::tiltxcorr(&program_args()))
}

fn tif2mrc() {
    exit(crate::imod::mrc::tif2mrc::tif2mrc(&program_args()))
}

fn tifinfo() {
    exit(crate::imod::mrc::tifinfo::tifinfo(&program_args()))
}

fn trimvol() {
    exit(crate::imod::pysrc::trimvol::trimvol(&program_args_os()))
}
