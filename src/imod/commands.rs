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
    cmd("alignframes", alignframes, true),
    cmd("alignlog", alignlog, false),
    cmd(
        "alterheader",
        crate::imod::flib::image::alterheader::alterheader,
        true,
    ),
    cmd(
        "assemblevol",
        crate::imod::flib::image::assemblevol::assemblevol,
        true,
    ),
    cmd("autofidseed", autofidseed, false),
    cmd("autopatchfit", autopatchfit, false),
    cmd("b3dcopy", b3dcopy, false),
    cmd("b3dremove", b3dremove, false),
    cmd("batchruntomo", batchruntomo, false),
    cmd("beadtrack", beadtrack, true),
    cmd("binvol", crate::imod::flib::image::binvol::binvol, true),
    cmd(
        "blendmont",
        crate::imod::flib::blend::blendmont::blendmont,
        true,
    ),
    cmd(
        "ccderaser",
        crate::imod::flib::model::ccderaser::ccderaser,
        true,
    ),
    cmd("chunksetup", chunksetup, false),
    cmd(
        "findwarp",
        crate::imod::flib::model::findwarp::findwarp,
        true,
    ),
    cmd(
        "refinematch",
        crate::imod::flib::model::refinematch::refinematch,
        true,
    ),
    cmd("clip", crate::imod::clip::clip::clip, true),
    cmd(
        "clipmodel",
        crate::imod::flib::model::clipmodel::clipmodel,
        true,
    ),
    cmd("collectmmm", collectmmm, false),
    cmd("runcom", crate::imod::comrun::runcom, false),
    cmd(
        "convertmod",
        crate::imod::flib::model::convertmod::convertmod,
        true,
    ),
    cmd(
        "combinefft",
        crate::imod::flib::image::combinefft::combinefft,
        true,
    ),
    cmd("copytomocoms", copytomocoms, false),
    cmd(
        "corrsearch3d",
        crate::imod::flib::model::corrsearch3d::corrsearch3d,
        true,
    ),
    cmd("ctfphaseflip", ctfphaseflip, true),
    cmd(
        "ctfplotter",
        crate::imod::ctfplotter::batch::ctfplotter,
        true,
    ),
    cmd(
        "densmatch",
        crate::imod::flib::image::densmatch::densmatch,
        true,
    ),
    cmd("dm3props", dm3props, true),
    cmd("dualvolmatch", dualvolmatch, false),
    cmd("fakevolume", fakevolume, true),
    cmd(
        "filltomo",
        crate::imod::flib::model::filltomo::filltomo,
        true,
    ),
    cmd(
        "findcontrast",
        crate::imod::flib::image::findcontrast::findcontrast,
        true,
    ),
    cmd("findbeads3d", findbeads3d, true),
    cmd(
        "flattenwarp",
        crate::imod::imodutil::flattenwarp::flattenwarp,
        true,
    ),
    cmd(
        "extractmagrad",
        crate::imod::flib::image::extractmagrad::extractmagrad,
        true,
    ),
    cmd(
        "extracttilts",
        crate::imod::flib::image::extracttilts::extracttilts,
        true,
    ),
    cmd("findsection", findsection, true),
    cmd("findsirtdiffs", findsirtdiffs, false),
    cmd(
        "fixboundaries",
        crate::imod::flib::image::fixboundaries::fixboundaries,
        true,
    ),
    cmd("echo2", echo2, true),
    cmd("etomo", etomo, false),
    cmd("etomodriver", etomodriver, false),
    #[cfg(feature = "gui")]
    cmd("etomo-gui", etomo_gui, false),
    #[cfg(feature = "gui")]
    cmd("etomo-popup", etomo_popup, false),
    #[cfg(feature = "gui")]
    cmd("etomo-filechooser", etomo_filechooser, false),
    // The plot window: runs its program on a second thread and the window
    // on the main one (`qtplax`), so never in another program's process.
    cmd(
        "genhstplt",
        crate::imod::flib::graphics::genhstplt::genhstplt,
        false,
    ),
    cmd("header", crate::imod::flib::image::header::header, true),
    cmd(
        "imodchopconts",
        crate::imod::imodutil::imodchopconts::imodchopconts,
        true,
    ),
    cmd(
        "imodfindbeads",
        crate::imod::imodutil::imodfindbeads::imodfindbeads,
        true,
    ),
    cmd(
        "imodextract",
        crate::imod::imodutil::imodextract::imodextract,
        true,
    ),
    cmd("imodinfo", crate::imod::imodutil::imodinfo::imodinfo, true),
    cmd("imodmop", crate::imod::imodutil::imodmop::imodmop, true),
    cmd("imodauto", imodauto, true),
    cmd(
        "imod2patch",
        crate::imod::imodutil::imod2patch::imod2patch,
        true,
    ),
    cmd(
        "joinwarp2model",
        crate::imod::imodutil::joinwarp2model::joinwarp2model,
        true,
    ),
    cmd("imodjoin", crate::imod::imodutil::imodjoin::imodjoin, true),
    cmd("imodmesh", crate::imod::imodutil::imodmesh::imodmesh, true),
    cmd(
        "imodtrans",
        crate::imod::imodutil::imodtrans::imodtrans,
        true,
    ),
    cmd("imodqtassist", imodqtassist, false),
    cmd("imodsendevent", imodsendevent, false),
    cmd("makecomfile", makecomfile, false),
    cmd("matchorwarp", matchorwarp, false),
    cmd("matchrotpairs", matchrotpairs, false),
    cmd(
        "matchvol",
        crate::imod::flib::image::matchvol::matchvol,
        true,
    ),
    cmd("midas", midas, false),
    cmd(
        "montagesize",
        crate::imod::flib::image::montagesize::montagesize,
        true,
    ),
    cmd("manageshrmem", manageshrmem, false),
    cmd(
        "mrc2tif",
        crate::imod::qttools::mrc2tif::mrc2tif::mrc2tif,
        true,
    ),
    cmd("mrcbyte", mrcbyte, true),
    cmd("mrcinfo", mrcinfo, true),
    cmd("mrcx", mrcx, true),
    cmd("raw2mrc", raw2mrc, true),
    cmd("modifymdoc", modifymdoc, true),
    cmd("mrclog", mrclog, true),
    cmd("mrctaper", mrctaper, true),
    cmd("mrctilt", mrctilt, true),
    cmd("mtffilter", mtffilter, true),
    cmd("nad_eed_3d", nad_eed_3d, true),
    cmd(
        "newstack",
        crate::imod::flib::image::newstack::newstack,
        true,
    ),
    cmd(
        "patch2imod",
        crate::imod::imodutil::patch2imod::patch2imod,
        true,
    ),
    cmd(
        "pickbestseed",
        crate::imod::imodutil::pickbestseed::pickbestseed,
        true,
    ),
    cmd(
        "point2model",
        crate::imod::imodutil::point2model::point2model,
        true,
    ),
    cmd("processchunks", processchunks, false),
    cmd("restrictalign", restrictalign, false),
    cmd("runraptor", runraptor, false),
    cmd("RAPTOR", raptor, true),
    cmd("MarkersCorrespond", markers_correspond, true),
    cmd("setupcombine", setupcombine, false),
    cmd(
        "solvematch",
        crate::imod::flib::model::solvematch::solvematch,
        true,
    ),
    cmd("sirtsetup", sirtsetup, false),
    cmd(
        "sortbeadsurfs",
        crate::imod::flib::model::sortbeadsurfs::sortbeadsurfs,
        true,
    ),
    cmd("sourcedoc", sourcedoc, true),
    cmd("splitcombine", splitcombine, false),
    cmd("splittilt", splittilt, false),
    cmd("subm", subm, false),
    cmd("submfg", submfg, false),
    cmd("tif2mrc", tif2mrc, true),
    cmd(
        "tomopitch",
        crate::imod::flib::model::tomopitch::tomopitch,
        true,
    ),
    cmd("tifinfo", tifinfo, true),
    cmd("tilt", tilt, true),
    cmd("tiltalign", tiltalign, true),
    cmd("tiltxcorr", tiltxcorr, true),
    cmd("tomocleanup", tomocleanup, false),
    cmd(
        "tomopieces",
        crate::imod::flib::image::tomopieces::tomopieces,
        true,
    ),
    cmd("trimvol", trimvol, false),
    cmd(
        "vmstocsh",
        crate::imod::flib::image::vmstocsh::vmstocsh,
        true,
    ),
    cmd("vmstopy", vmstopy, false),
    cmd("warpvol", crate::imod::flib::image::warpvol::warpvol, true),
    cmd(
        "wmod2imod",
        crate::imod::imodutil::wmod2imod::wmod2imod,
        true,
    ),
    cmd(
        "xcorrstack",
        crate::imod::flib::image::xcorrstack::xcorrstack,
        true,
    ),
    cmd(
        "xf2rotmagstr",
        crate::imod::flib::distort::xf2rotmagstr::xf2rotmagstr,
        true,
    ),
    cmd("xfmodel", crate::imod::flib::model::xfmodel::xfmodel, true),
    cmd(
        "xfsimplex",
        crate::imod::flib::image::xfsimplex::xfsimplex,
        true,
    ),
    cmd(
        "xfproduct",
        crate::imod::flib::image::xfproduct::xfproduct,
        true,
    ),
    cmd("xftoxg", crate::imod::flib::image::xftoxg::xftoxg, true),
    cmd(
        "subimage",
        crate::imod::flib::image::subimage::subimage,
        true,
    ),
    cmd(
        "maxjoinsize",
        crate::imod::flib::image::maxjoinsize::maxjoinsize,
        true,
    ),
    cmd(
        "boxstartend",
        crate::imod::flib::model::boxstartend::boxstartend,
        true,
    ),
    cmd(
        "model2point",
        crate::imod::flib::model::model2point::model2point,
        true,
    ),
    cmd(
        "repackseed",
        crate::imod::flib::model::repackseed::repackseed,
        true,
    ),
    cmd(
        "avgstack",
        crate::imod::flib::image::avgstack::avgstack,
        true,
    ),
    cmd(
        "extractpieces",
        crate::imod::flib::image::extractpieces::extractpieces,
        true,
    ),
    cmd(
        "goodframe",
        crate::imod::flib::image::goodframe::goodframe,
        true,
    ),
    cmd(
        "remapmodel",
        crate::imod::flib::model::remapmodel::remapmodel,
        true,
    ),
    cmd(
        "rotatevol",
        crate::imod::flib::image::rotatevol::rotatevol,
        true,
    ),
    cmd(
        "taperoutvol",
        crate::imod::flib::image::taperoutvol::taperoutvol,
        true,
    ),
    cmd(
        "xfjointomo",
        crate::imod::flib::model::xfjointomo::xfjointomo,
        true,
    ),
    cmd("xyzproj", crate::imod::flib::image::xyzproj::xyzproj, true),
    cmd("b3dhostname", b3dhostname, false),
    cmd("b3dwinps", b3dwinps, false),
    cmd("imodwincpu", imodwincpu, true),
    cmd("b3dtouch", b3dtouch, false),
    cmd("b3dtomosetexts", b3dtomosetexts, false),
    cmd("startprocess", startprocess, false),
    cmd("slicesforsample", slicesforsample, false),
    cmd("sampletilt", sampletilt, false),
    cmd("squeezevol", squeezevol, false),
    cmd("splitcorrection", splitcorrection, false),
    cmd("sorttiltframes", sorttiltframes, false),
    cmd("multifiltsetup", multifiltsetup, false),
    cmd("reducefiltvol", reducefiltvol, false),
    cmd("copyheader", copyheader, false),
    cmd("archiveorig", archiveorig, false),
    cmd("splitbatch", splitbatch, false),
    cmd("finishjoin", finishjoin, false),
    cmd("makejoincom", makejoincom, false),
    cmd("xfalign", xfalign, false),
    cmd("transferfid", transferfid, false),
    cmd("imodkillgroup", imodkillgroup, false),
    cmd("gputilttest", gputilttest, false),
    cmd("cryoposition", cryoposition, false),
    cmd("onegenplot", onegenplot, false),
    cmd("tomodataplots", tomodataplots, false),
    cmd("excludeviews", excludeviews, false),
    cmd("swaptomostacks", swaptomostacks, false),
    cmd("ctf3dsetup", ctf3dsetup, false),
    cmd("alttomosetup", alttomosetup, false),
    cmd("subtomosetup", subtomosetup, false),
    cmd("tomosnapshot", tomosnapshot, false),
    cmd("serieswatcher", serieswatcher, false),
    cmd("edmont", crate::imod::flib::model::edmont::edmont, true),
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
/// adds, since an exiting process does not run it either.  Image files the
/// command opened outside a unit (a direct `iiOpen`) and never deleted are
/// released the same way, from the thread's run record.
pub fn run_in_process(
    command: &'static Command,
    argv: Vec<OsString>,
    input: Option<&[u8]>,
    capture: bool,
) -> std::io::Result<(i32, Vec<u8>)> {
    run_body_in_process(argv, input, capture, command.entry)
}

/// Rust-only: runs `body` -- a translated program's computation called
/// directly with its parameters, in place of its `main` -- under exactly the
/// fresh-process state [`run_in_process`] gives a whole command, and returns
/// the command's exit status, the value `body` returned (`None` when it ended
/// through [`exit`] instead of returning), and, when `capture` is set, what
/// it printed on standard output.
///
/// This is the plumbing for the owner's rule that where we control both
/// sides a script calls our program directly and uses returned values rather
/// than parsing printed text (`CLAUDE.md`, "Wherever we control both sides,
/// use a direct function call now").  The program's own error paths still end
/// in [`exit`] -- `exitError` inside a library routine cannot return a value
/// -- and the runner turns that into the status, as a process would.
/// `words` are the program's name and arguments; its `argv[0]` becomes
/// `<bindir>/<name>`, as for [`run_in_process`].  A program whose options
/// are still read through PIP gets them here (and `input` on its standard
/// input); a compute function called with its parameters needs only the name.
pub fn call_in_process<R: Send + 'static>(
    words: &[&str],
    input: Option<&[u8]>,
    capture: bool,
    body: impl FnOnce() -> R + Send + 'static,
) -> std::io::Result<(i32, Option<R>, Vec<u8>)> {
    let name = words.first().copied().unwrap_or_default();
    let mut argv: Vec<OsString> = Vec::with_capacity(words.len().max(1));
    argv.push(match std::env::current_exe() {
        Ok(path) => path.with_file_name(name).into_os_string(),
        Err(_) => OsString::from(name),
    });
    argv.extend(words.iter().skip(1).map(OsString::from));
    let slot = std::sync::Arc::new(std::sync::Mutex::new(None));
    let inner = std::sync::Arc::clone(&slot);
    let (status, output) = run_body_in_process(argv, input, capture, move || {
        let value = body();
        *inner.lock().expect("in-process result slot") = Some(value);
    })?;
    let value = slot.lock().expect("in-process result slot").take();
    Ok((status, value, output))
}

/// Rust-only: the body of [`run_in_process`], generic over what runs on the
/// command's thread so [`call_in_process`] shares it.
fn run_body_in_process<F: FnOnce() + Send + 'static>(
    argv: Vec<OsString>,
    input: Option<&[u8]>,
    capture: bool,
    entry: F,
) -> std::io::Result<(i32, Vec<u8>)> {
    let saved_multi_volume = iimage::S_ALLOW_MULTI_VOLUME.swap(0, Ordering::SeqCst);
    let saved_compression = std::env::var_os("IMOD_TIFF_COMPRESSION");
    let result = crate::imod::libcfshr::b3dutil::run_in_process(argv, input, capture, move || {
        /// Releases the unit table's images, and then every other image file
        /// the command left allocated or open (its run record, see
        /// `iimage::ii_release_run_record`), when the command's `main` ends,
        /// whether it returned or unwound from [`exit`].
        struct ReleaseUnits;
        impl Drop for ReleaseUnits {
            fn drop(&mut self) {
                unit_fileio::release_all_units();
                iimage::ii_release_run_record();
            }
        }
        iimage::ii_begin_run_record();
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

fn ctfphaseflip() {
    exit(crate::imod::mrc::ctfphaseflip::ctfphaseflip(&program_args()))
}

fn dm3props() {
    exit(crate::imod::mrc::dm3props::dm3props(&program_args()))
}

fn fakevolume() {
    exit(crate::imod::mrc::fakevolume::fakevolume(&program_args()))
}

fn findbeads3d() {
    exit(crate::imod::imodutil::findbeads3d::findbeads3d(
        &program_args(),
    ))
}

fn findsection() {
    exit(crate::imod::imodutil::findsection::findsection(
        &program_args(),
    ))
}

fn echo2() {
    exit(crate::imod::imodutil::echo2::echo2(&program_args()))
}

fn etomodriver() {
    let arguments = program_args().into_iter().skip(1).collect::<Vec<_>>();
    exit(crate::imod::etomo::driver::etomodriver(&arguments))
}

fn etomo() {
    exit(crate::imod::pysrc::etomo::etomo(&program_args_os()))
}

/// Rust-only: the modal popup dialog `etomo-gui` shows in a child process
/// (`ui/swing/slint_bridge.rs`).
#[cfg(feature = "gui")]
fn etomo_popup() {
    let arguments = program_args().into_iter().skip(1).collect::<Vec<_>>();
    exit(crate::imod::etomo::ui::swing::slint_bridge::etomo_popup(
        &arguments,
    ))
}

/// Rust-only: the modal file chooser dialog `etomo-gui` shows in a child
/// process (`ui/swing/slint_bridge.rs`).
#[cfg(feature = "gui")]
fn etomo_filechooser() {
    let arguments = program_args().into_iter().skip(1).collect::<Vec<_>>();
    exit(crate::imod::etomo::ui::swing::slint_bridge::etomo_filechooser(&arguments))
}

#[cfg(feature = "gui")]
fn etomo_gui() {
    let arguments = program_args().into_iter().skip(1).collect::<Vec<_>>();
    if let Err(error) = crate::imod::etomo::etomo_director::EtomoDirector::main_gui(&arguments) {
        eprintln!("{error}");
        exit(1);
    }
}

fn nad_eed_3d() {
    exit(crate::imod::mrc::nad_eed_3d::nad_eed_3d(&program_args()))
}

fn imodauto() {
    exit(crate::imod::imodutil::imodauto::imodauto(&program_args()))
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
    crate::imod::mrc::mrcinfo::mrcinfo(&program_args_os())
}

fn mrcx() {
    crate::imod::mrc::mrcx::mrcx(&program_args_os())
}

fn modifymdoc() {
    exit(crate::imod::mrc::modifymdoc::modifymdoc(&program_args()))
}

fn mrclog() {
    exit(crate::imod::mrc::mrclog::mrclog(&program_args()))
}

fn mtffilter() {
    exit(crate::imod::mrc::mtffilter::mtffilter(&program_args()))
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
    crate::imod::mrc::tifinfo::tifinfo(&program_args_os())
}

fn trimvol() {
    exit(crate::imod::pysrc::trimvol::trimvol(&program_args_os()))
}

fn alignlog() {
    exit(crate::imod::pysrc::alignlog::alignlog(&program_args_os()))
}

fn chunksetup() {
    exit(crate::imod::pysrc::chunksetup::chunksetup(
        &program_args_os(),
    ))
}

fn makecomfile() {
    exit(crate::imod::pysrc::makecomfile::makecomfile(
        &program_args_os(),
    ))
}

fn autofidseed() {
    exit(crate::imod::pysrc::autofidseed::autofidseed(
        &program_args_os(),
    ))
}

fn sirtsetup() {
    exit(crate::imod::pysrc::sirtsetup::sirtsetup(&program_args_os()))
}

fn findsirtdiffs() {
    exit(crate::imod::pysrc::findsirtdiffs::findsirtdiffs(
        &program_args_os(),
    ))
}

fn splittilt() {
    exit(crate::imod::pysrc::splittilt::splittilt(&program_args_os()))
}

fn tomocleanup() {
    exit(crate::imod::pysrc::tomocleanup::tomocleanup(
        &program_args_os(),
    ))
}

fn copytomocoms() {
    exit(crate::imod::pysrc::copytomocoms::copytomocoms(
        &program_args_os(),
    ))
}

fn restrictalign() {
    exit(crate::imod::pysrc::restrictalign::restrictalign(
        &program_args_os(),
    ))
}

fn raptor() {
    exit(crate::imod::raptor::main::main())
}

fn markers_correspond() {
    exit(crate::imod::raptor::correspondence::markers_correspond_main_test::main(&program_args()))
}

fn runraptor() {
    exit(crate::imod::pysrc::runraptor::runraptor(&program_args_os()))
}

fn autopatchfit() {
    exit(crate::imod::pysrc::autopatchfit::autopatchfit(
        &program_args_os(),
    ))
}

fn b3dcopy() {
    exit(crate::imod::pysrc::b3dcopy::b3dcopy(&program_args_os()))
}

fn vmstopy() {
    exit(crate::imod::pysrc::vmstopy::vmstopy(&program_args_os()))
}

fn b3dhostname() {
    exit(crate::imod::pysrc::b3dhostname::b3dhostname())
}

fn b3dwinps() {
    exit(crate::imod::pysrc::b3dwinps::b3dwinps(&program_args_os()))
}

fn imodwincpu() {
    // Windows-only in the source: it is built from `<windows.h>` and reads
    // the Windows performance counters.
    #[cfg(windows)]
    exit(crate::imod::imodutil::imodwincpu::imodwincpu(
        &program_args(),
    ));
    #[cfg(not(windows))]
    {
        eprintln!("ERROR: imodwincpu - not available on this platform");
        exit(1)
    }
}

fn b3dtouch() {
    exit(crate::imod::pysrc::b3dtouch::b3dtouch(&program_args_os()))
}

fn b3dtomosetexts() {
    exit(crate::imod::pysrc::b3dtomosetexts::b3dtomosetexts(
        &program_args_os(),
    ))
}

fn startprocess() {
    exit(crate::imod::pysrc::startprocess::startprocess(
        &program_args_os(),
    ))
}

fn slicesforsample() {
    exit(crate::imod::pysrc::slicesforsample::slicesforsample(
        &program_args_os(),
    ))
}

fn sampletilt() {
    exit(crate::imod::pysrc::sampletilt::sampletilt(
        &program_args_os(),
    ))
}

fn squeezevol() {
    exit(crate::imod::pysrc::squeezevol::squeezevol(
        &program_args_os(),
    ))
}

fn splitcorrection() {
    exit(crate::imod::pysrc::splitcorrection::splitcorrection(
        &program_args_os(),
    ))
}

fn sorttiltframes() {
    exit(crate::imod::pysrc::sorttiltframes::sorttiltframes(
        &program_args_os(),
    ))
}

fn multifiltsetup() {
    exit(crate::imod::pysrc::multifiltsetup::multifiltsetup(
        &program_args_os(),
    ))
}

fn reducefiltvol() {
    exit(crate::imod::pysrc::reducefiltvol::reducefiltvol(
        &program_args_os(),
    ))
}

fn copyheader() {
    exit(crate::imod::pysrc::copyheader::copyheader(
        &program_args_os(),
    ))
}

fn alignframes() {
    exit(crate::imod::mrc::alignframes::alignframes(&program_args()))
}

fn archiveorig() {
    exit(crate::imod::pysrc::archiveorig::archiveorig(
        &program_args_os(),
    ))
}

fn splitbatch() {
    exit(crate::imod::pysrc::splitbatch::splitbatch(
        &program_args_os(),
    ))
}

fn finishjoin() {
    exit(crate::imod::pysrc::finishjoin::finishjoin(
        &program_args_os(),
    ))
}

fn makejoincom() {
    exit(crate::imod::pysrc::makejoincom::makejoincom(
        &program_args_os(),
    ))
}

fn xfalign() {
    exit(crate::imod::pysrc::xfalign::xfalign(&program_args_os()))
}

fn transferfid() {
    exit(crate::imod::pysrc::transferfid::transferfid(
        &program_args_os(),
    ))
}

fn imodkillgroup() {
    exit(crate::imod::pysrc::imodkillgroup::imodkillgroup(
        &program_args_os(),
    ))
}

fn gputilttest() {
    exit(crate::imod::pysrc::gputilttest::gputilttest(
        &program_args_os(),
    ))
}

fn cryoposition() {
    exit(crate::imod::pysrc::cryoposition::cryoposition(
        &program_args_os(),
    ))
}

fn onegenplot() {
    exit(crate::imod::pysrc::onegenplot::onegenplot(
        &program_args_os(),
    ))
}

fn tomodataplots() {
    exit(crate::imod::pysrc::tomodataplots::tomodataplots(
        &program_args_os(),
    ))
}

fn excludeviews() {
    exit(crate::imod::pysrc::excludeviews::excludeviews(
        &program_args_os(),
    ))
}

fn swaptomostacks() {
    exit(crate::imod::pysrc::swaptomostacks::swaptomostacks(
        &program_args_os(),
    ))
}

fn ctf3dsetup() {
    exit(crate::imod::pysrc::ctf3dsetup::ctf3dsetup(
        &program_args_os(),
    ))
}

fn alttomosetup() {
    exit(crate::imod::pysrc::alttomosetup::alttomosetup(
        &program_args_os(),
    ))
}

fn subtomosetup() {
    exit(crate::imod::pysrc::subtomosetup::subtomosetup(
        &program_args_os(),
    ))
}

fn tomosnapshot() {
    exit(crate::imod::pysrc::tomosnapshot::tomosnapshot(
        &program_args_os(),
    ))
}

fn serieswatcher() {
    exit(crate::imod::pysrc::serieswatcher::serieswatcher(
        &program_args_os(),
    ))
}

fn b3dremove() {
    exit(crate::imod::pysrc::b3dremove::b3dremove(&program_args_os()))
}

fn collectmmm() {
    exit(crate::imod::pysrc::collectmmm::collectmmm(
        &program_args_os(),
    ))
}

fn dualvolmatch() {
    exit(crate::imod::pysrc::dualvolmatch::dualvolmatch(
        &program_args_os(),
    ))
}

fn matchrotpairs() {
    exit(crate::imod::pysrc::matchrotpairs::matchrotpairs(
        &program_args_os(),
    ))
}

fn matchorwarp() {
    exit(crate::imod::pysrc::matchorwarp::matchorwarp(
        &program_args_os(),
    ))
}

fn setupcombine() {
    exit(crate::imod::pysrc::setupcombine::setupcombine(
        &program_args_os(),
    ))
}

fn splitcombine() {
    exit(crate::imod::pysrc::splitcombine::splitcombine(
        &program_args_os(),
    ))
}
