//! `IMOD/Etomo/src/etomo/storage/autodoc/AutodocFactory.java`.
//!
//! Description: Creates Autodoc classes.
//!
//! `BaseManager` has no module; every caller that reaches a factory passes a null one,
//! so the parameter is typed `Option<Infallible>`.
//!
//! Java's `private AutodocFactory() {}` only prevents instantiation of the utility
//! class; a Rust module needs no equivalent.  Java's statics are per-process; a
//! `*mut Autodoc` is neither `Send` nor `Sync`, so the registry is thread-local here,
//! as `etomo/util/mrc_header.rs` already does for its instance map.
#![allow(dead_code)]

use super::autodoc::Autodoc;
use super::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::autodoc_filter::AutodocFilter;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::util::utilities;
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;

/// Java `VERSION`.
pub const VERSION: &str = "1.2";

/// Java `TILTXCORR`.
pub const TILTXCORR: &str = "tiltxcorr";

/// Java `MTF_FILTER`.
pub const MTF_FILTER: &str = "mtffilter";

/// Java `COMBINE_FFT`.
pub const COMBINE_FFT: &str = "combinefft";

/// Java `TILTALIGN`.
pub const TILTALIGN: &str = "tiltalign";

/// Java `CCDERASER`.
pub const CCDERASER: &str = "ccderaser";

/// Java `SOLVEMATCH`.
pub const SOLVEMATCH: &str = "solvematch";

/// Java `BEADTRACK`.
pub const BEADTRACK: &str = "beadtrack";

/// Java `DENS_MATCH`.
pub const DENS_MATCH: &str = "densmatch";

/// Java `CORR_SEARCH_3D`.
pub const CORR_SEARCH_3D: &str = "corrsearch3d";

/// Java `XFJOINTOMO`.
pub const XFJOINTOMO: &str = "xfjointomo";

/// Java `CPU`.
pub const CPU: &str = "cpu";

/// Java `UITEST`.
pub const UITEST: &str = "uitest";

/// Java `PEET_PRM`.
pub const PEET_PRM: &str = "peetprm";

/// Java `NEWSTACK`.
pub const NEWSTACK: &str = "newstack";

/// Java `CTF_PLOTTER`.
pub const CTF_PLOTTER: &str = "ctfplotter";

/// Java `CTF_PHASE_FLIP`.
pub const CTF_PHASE_FLIP: &str = "ctfphaseflip";

/// Java `FLATTEN_WARP`.
pub const FLATTEN_WARP: &str = "flattenwarp";

/// Java `WARP_VOL`.
pub const WARP_VOL: &str = "warpvol";

/// Java `FIND_BEADS_3D`.
pub const FIND_BEADS_3D: &str = "findbeads3d";

/// Java `TILT`.
pub const TILT: &str = "tilt";

/// Java `SIRTSETUP`.
pub const SIRTSETUP: &str = "sirtsetup";

/// Java `BLENDMONT`.
pub const BLENDMONT: &str = "blendmont";

/// Java `XFTOXG`.
pub const XFTOXG: &str = "xftoxg";

/// Java `XFALIGN`.
pub const XFALIGN: &str = "xfalign";

/// Java `AUTOFIDSEED`.
pub const AUTOFIDSEED: &str = "autofidseed";

/// Java `ETOMO`.
pub const ETOMO: &str = "etomo";

/// Java `PROG_DEFAULTS`.
pub const PROG_DEFAULTS: &str = "progDefaults";

/// Java `IMODCHOPCONTS`.
pub const IMODCHOPCONTS: &str = "imodchopconts";

/// Java `DUALVOLMATCH`.
pub const DUALVOLMATCH: &str = "dualvolmatch";

/// Java `RESTRICT_ALIGN`.
pub const RESTRICT_ALIGN: &str = "restrictalign";

/// Java `BATCH_RUN_TOMO`.
pub const BATCH_RUN_TOMO: &str = "batchruntomo";

/// Java `MULTIFILT_SETUP`.
pub const MULTIFILT_SETUP: &str = "multifiltsetup";

/// Java `CTF_3D_SETUP`.
pub const CTF_3D_SETUP: &str = "ctf3dsetup";

/// Java `ALIGN_FRAMES`.
pub const ALIGN_FRAMES: &str = "alignframes";

/// Java `SUBTOMO_SETUP`.
pub const SUBTOMO_SETUP: &str = "subtomosetup";

/// Java `ALT_TOMO_SETUP`.
pub const ALT_TOMO_SETUP: &str = "alttomosetup";

/// Java `REDUCE_FILTER_VOLUME`.
pub const REDUCE_FILTER_VOLUME: &str = "reducefiltvol";

/// Java `SERIES_WATCHER`.
pub const SERIES_WATCHER: &str = "serieswatcher";

/// Java private `TEST`.
const TEST: &str = "test";

/// Java private `UITEST_AXIS`.
const UITEST_AXIS: &str = "uitest_axis";

thread_local! {
    /// Java private static `TILTXCORR_INSTANCE`, initialised to null.
    static TILTXCORR_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `TEST_INSTANCE`, initialised to null.
    static TEST_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `UITEST_INSTANCE`, initialised to null.
    static UITEST_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `MTF_FILTER_INSTANCE`, initialised to null.
    static MTF_FILTER_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `COMBINE_FFT_INSTANCE`, initialised to null.
    static COMBINE_FFT_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `TILTALIGN_INSTANCE`, initialised to null.
    static TILTALIGN_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `CCDERASER_INSTANCE`, initialised to null.
    static CCDERASER_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `SOLVEMATCH_INSTANCE`, initialised to null.
    static SOLVEMATCH_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `BEADTRACK_INSTANCE`, initialised to null.
    static BEADTRACK_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `CPU_INSTANCE`, initialised to null.
    static CPU_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `DENS_MATCH_INSTANCE`, initialised to null.
    static DENS_MATCH_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `CORR_SEARCH_3D_INSTANCE`, initialised to null.
    static CORR_SEARCH_3D_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `XFJOINTOMO_INSTANCE`, initialised to null.
    static XFJOINTOMO_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `PEET_PRM_INSTANCE`, initialised to null.
    static PEET_PRM_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `NEWSTACK_INSTANCE`, initialised to null.
    static NEWSTACK_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `CTF_PLOTTER_INSTANCE`, initialised to null.
    static CTF_PLOTTER_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `CTF_PHASE_FLIP_INSTANCE`, initialised to null.
    static CTF_PHASE_FLIP_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `FLATTEN_WARP_INSTANCE`, initialised to null.
    static FLATTEN_WARP_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `WARP_VOL_INSTANCE`, initialised to null.
    static WARP_VOL_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `FIND_BEADS_3D_INSTANCE`, initialised to null.
    static FIND_BEADS_3D_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `TILT_INSTANCE`, initialised to null.
    static TILT_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `SIRTSETUP_INSTANCE`, initialised to null.
    static SIRTSETUP_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `BLENDMONT_INSTANCE`, initialised to null.
    static BLENDMONT_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `XFTOXG_INSTANCE`, initialised to null.
    static XFTOXG_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `XFALIGN_INSTANCE`, initialised to null.
    static XFALIGN_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `AUTOFIDSEED_INSTANCE`, initialised to null.
    static AUTOFIDSEED_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `ETOMO_INSTANCE`, initialised to null.
    static ETOMO_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `PROG_DEFAULTS_INSTANCE`, initialised to null.
    static PROG_DEFAULTS_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `IMODCHOPCONTS_INSTANCE`, initialised to null.
    static IMODCHOPCONTS_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `DUALVOLMATCH_INSTANCE`, initialised to null.
    static DUALVOLMATCH_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `RESTRICT_ALIGN_INSTANCE`, initialised to null.
    static RESTRICT_ALIGN_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `BATCH_RUN_TOMO_INSTANCE`, initialised to null.
    static BATCH_RUN_TOMO_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `MULTIFILT_SETUP_INSTANCE`, initialised to null.
    static MULTIFILT_SETUP_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `ALIGN_FRAMES_INSTANCE`, initialised to null.
    static ALIGN_FRAMES_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `SUBTOMO_SETUP_INSTANCE`, initialised to null.
    static SUBTOMO_SETUP_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `ALT_TOMO_SETUP_INSTANCE`, initialised to null.
    static ALT_TOMO_SETUP_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `REDUCE_FILTER_VOLUME_INSTANCE`, initialised to null.
    static REDUCE_FILTER_VOLUME_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static `SERIES_WATCHER_INSTANCE`, initialised to null.
    static SERIES_WATCHER_INSTANCE: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private static final `UITEST_AXIS_MAP`.
    static UITEST_AXIS_MAP: RefCell<HashMap<std::path::PathBuf, *mut Autodoc>> =
        RefCell::new(HashMap::new());
    /// Java private static `replacementDir`, initialised to null.
    static REPLACEMENT_DIR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Java `getInstance(BaseManager, String)`.
///
/// # Safety
/// The returned pointer is the n'ton registry's, which owns it for the life of the
/// process.
pub unsafe fn get_instance_name(
    manager: Option<&'static dyn BaseManager>,
    name: Option<&str>,
) -> Result<*mut Autodoc, LogFileError> {
    unsafe { get_instance(manager, name, AxisID::Only, false) }
}

/// Java `getComInstance`.
///
/// # Safety
/// See `get_instance_name`.
pub unsafe fn get_com_instance(name: Option<&str>) -> Result<*mut Autodoc, LogFileError> {
    let name = match name {
        None => panic!("java.lang.IllegalStateException: name is null"),
        Some(name) => name,
    };
    let mut autodoc = get_existing_autodoc(name);
    if !autodoc.is_null() {
        return Ok(autodoc);
    }
    autodoc = unsafe { Autodoc::new(Some(name), std::ptr::null_mut()) };
    unsafe {
        Autodoc::initialize_generic_instance_env_var(
            autodoc,
            None,
            Some(etomo_director::IMOD_DIR_ENV_VAR),
            Some(file_type::COM_DIR),
            Some(name),
            AxisID::Only,
            false,
        )?
    };
    Ok(autodoc)
}

/// Java `getUnmanagedAutodocInstance`.
///
/// # Safety
/// The returned pointer owns a heap `Autodoc`, where the source's owner is the GC.
pub unsafe fn get_unmanaged_autodoc_instance(
    manager: Option<&'static dyn BaseManager>,
    name: Option<&str>,
    file_type: Option<&FileType>,
    axis_id: AxisID,
) -> Result<*mut Autodoc, LogFileError> {
    let name = match name {
        None => panic!("java.lang.IllegalStateException: name is null"),
        Some(name) => name,
    };
    let autodoc = unsafe { Autodoc::new(Some(name), std::ptr::null_mut()) };
    let mut autodoc_file: Option<std::path::PathBuf> = None;
    if let Some(file_type) = file_type {
        autodoc_file = file_type.get_file(manager, Some(axis_id));
    }
    unsafe {
        Autodoc::initialize_unmanaged_autodoc_instance(
            autodoc,
            manager,
            Some(name),
            autodoc_file.as_deref(),
            axis_id,
        )?
    };
    Ok(autodoc)
}

/// Java `getInstance(BaseManager, String, AxisID, boolean)`.
///
/// # Safety
/// See `get_instance_name`.
pub unsafe fn get_instance(
    manager: Option<&'static dyn BaseManager>,
    name: Option<&str>,
    axis_id: AxisID,
    debug: bool,
) -> Result<*mut Autodoc, LogFileError> {
    let name = match name {
        None => panic!("java.lang.IllegalStateException: name is null"),
        Some(name) => name,
    };
    let mut autodoc = get_existing_autodoc(name);
    if !autodoc.is_null() {
        return Ok(autodoc);
    }
    autodoc = unsafe { Autodoc::new(Some(name), std::ptr::null_mut()) };
    unsafe { set_instance(name, autodoc) };
    unsafe { (*autodoc).set_debug_to(debug) };
    if name == UITEST {
        unsafe { Autodoc::initialize_ui_test_instance(autodoc, manager, Some(name), axis_id)? };
    }
    if name == CPU {
        unsafe { Autodoc::initialize_cpu_instance(autodoc, manager, Some(name), axis_id)? };
    } else {
        unsafe { Autodoc::initialize_autodoc_instance(autodoc, manager, Some(name), axis_id)? };
    }
    Ok(autodoc)
}

/// Java `getMatlabInstance`.
///
/// # Safety
/// The returned pointer owns a heap `Autodoc`, where the source's owner is the GC.
pub unsafe fn get_matlab_instance(
    manager: Option<&'static dyn BaseManager>,
    file: Option<&std::path::Path>,
    writable: bool,
) -> Result<*mut Autodoc, LogFileError> {
    let file = match file {
        None => panic!("java.lang.IllegalStateException: file is null"),
        Some(file) => file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension_of_file(file)),
            std::ptr::null_mut(),
        )
    };
    match unsafe { Autodoc::initialize_matlab_instance(autodoc, manager, file, writable) } {
        Ok(()) => Ok(autodoc),
        Err(LogFileError::Io(e)) => {
            // Java's `catch (FileNotFoundException)`.
            // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
            eprintln!("{}", e);
            Ok(std::ptr::null_mut())
        }
        Err(e) => Err(e),
    }
}

/// Java `getWritableInstance`.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_writable_instance(
    manager: Option<&'static dyn BaseManager>,
    file: Option<&std::path::Path>,
) -> Result<*mut Autodoc, LogFileError> {
    let file = match file {
        None => panic!("java.lang.IllegalStateException: file is null"),
        Some(file) => file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension_of_file(file)),
            std::ptr::null_mut(),
        )
    };
    match unsafe { Autodoc::initialize_writable_instance(autodoc, manager, file) } {
        Ok(()) => Ok(autodoc),
        // Java's `catch (FileNotFoundException)`.
        Err(LogFileError::Io(_)) => Ok(std::ptr::null_mut()),
        Err(e) => Err(e),
    }
}

/// Java `getEmptyWritableInstance`.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_empty_writable_instance(
    manager: Option<&'static dyn BaseManager>,
    file: Option<&std::path::Path>,
) -> *mut Autodoc {
    let file = match file {
        None => panic!("java.lang.IllegalStateException: file is null"),
        Some(file) => file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension_of_file(file)),
            std::ptr::null_mut(),
        )
    };
    unsafe { Autodoc::initialize_empty_writable_instance(autodoc, manager, file) };
    autodoc
}

/// Java `getEmptyMatlabInstance`.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_empty_matlab_instance(
    manager: Option<&'static dyn BaseManager>,
    file: Option<&std::path::Path>,
) -> *mut Autodoc {
    let file = match file {
        None => panic!("java.lang.IllegalStateException: file is null"),
        Some(file) => file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension_of_file(file)),
            std::ptr::null_mut(),
        )
    };
    unsafe { Autodoc::initialize_empty_matlap_instance(autodoc, manager, file) };
    autodoc
}

/// Java `getInstance(BaseManager, File, boolean)`.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_instance_file(
    manager: Option<&'static dyn BaseManager>,
    file: Option<&std::path::Path>,
    writable: bool,
) -> Result<*mut Autodoc, LogFileError> {
    let file = match file {
        None => panic!("java.lang.IllegalStateException: file is null"),
        Some(file) => file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension_of_file(file)),
            std::ptr::null_mut(),
        )
    };
    match unsafe {
        Autodoc::initialize_generic_instance(
            autodoc,
            manager,
            file,
            AxisID::Only,
            writable,
            std::ptr::null_mut(),
        )
    } {
        Ok(()) => Ok(autodoc),
        // Java's `catch (FileNotFoundException)`.
        Err(LogFileError::Io(_)) => Ok(std::ptr::null_mut()),
        Err(e) => Err(e),
    }
}

/// Java `getUnmanagedInstance(BaseManager, AxisID, File)`.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_unmanaged_instance(
    manager: Option<&'static dyn BaseManager>,
    axis_id: AxisID,
    autodoc_file: Option<&std::path::Path>,
) -> Result<*mut Autodoc, LogFileError> {
    let autodoc_file = match autodoc_file {
        None => {
            eprintln!("Warning: autodocFile is null");
            // Deviation: `Thread.dumpStack()` prints the calling Java thread's stack
            // trace to stderr.  A Rust backtrace is neither the same frames nor the same
            // text, so nothing is printed in its place; see
            // `crate::imod::etomo::util::stack_trace`.
            return Ok(std::ptr::null_mut());
        }
        Some(autodoc_file) => autodoc_file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension_of_file(autodoc_file)),
            std::ptr::null_mut(),
        )
    };
    match unsafe {
        Autodoc::initialize_generic_instance(
            autodoc,
            manager,
            autodoc_file,
            axis_id,
            false,
            std::ptr::null_mut(),
        )
    } {
        Ok(()) => Ok(autodoc),
        // Java's `catch (FileNotFoundException)`.
        Err(LogFileError::Io(_)) => Ok(std::ptr::null_mut()),
        Err(e) => Err(e),
    }
}

/// Java `getInstance(BaseManager, File, String, boolean)`.
///
/// Initialize one of the known autodoc instances with a generic instance.  If the known
/// autodoc has an instance assigned to it, this overrides the old instance with the one
/// created here.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_instance_file_autodoc_name(
    manager: Option<&'static dyn BaseManager>,
    file: Option<&std::path::Path>,
    autodoc_name: Option<&str>,
    writable: bool,
) -> Result<*mut Autodoc, LogFileError> {
    let file = match file {
        None => panic!("java.lang.IllegalStateException: file is null"),
        Some(file) => file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension_of_file(file)),
            std::ptr::null_mut(),
        )
    };
    match unsafe {
        Autodoc::initialize_generic_instance(
            autodoc,
            manager,
            file,
            AxisID::Only,
            writable,
            std::ptr::null_mut(),
        )
    } {
        Ok(()) => {
            if let Some(autodoc_name) = autodoc_name {
                unsafe { set_instance(autodoc_name, autodoc) };
            }
            Ok(autodoc)
        }
        // Java's `catch (FileNotFoundException)`.
        Err(LogFileError::Io(_)) => Ok(std::ptr::null_mut()),
        Err(e) => Err(e),
    }
}

/// Java `getTestInstance`.  Open and preserve an autodoc without a type for testing.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_test_instance(
    manager: Option<&'static dyn BaseManager>,
    directory: Option<&std::path::Path>,
    autodoc_file_name: Option<&str>,
    axis_id: AxisID,
) -> Result<*mut Autodoc, LogFileError> {
    let autodoc_file_name = match autodoc_file_name {
        None => return Ok(std::ptr::null_mut()),
        Some(autodoc_file_name) => autodoc_file_name,
    };
    let autodoc_file = match directory {
        None => std::path::PathBuf::from(autodoc_file_name),
        Some(directory) => directory.join(autodoc_file_name),
    };
    let filter = AutodocFilter::new();
    if !filter.accept(&autodoc_file) {
        panic!(
            "java.lang.IllegalArgumentException: {} is not an autodoc.",
            autodoc_file.to_string_lossy()
        );
    }
    if etomo_director::ARGUMENTS.lock().unwrap().is_test()
        && etomo_director::ARGUMENTS.lock().unwrap().is_debug()
    {
        eprintln!(
            "autodoc file:{}",
            utilities::java_io_file_get_absolute_path(&autodoc_file.to_string_lossy())
        );
    }
    let mut autodoc = get_existing_ui_test_axis_autodoc(&autodoc_file);
    if !autodoc.is_null() {
        return Ok(autodoc);
    }
    autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension(autodoc_file_name)),
            std::ptr::null_mut(),
        )
    };
    UITEST_AXIS_MAP.with(|map| map.borrow_mut().insert(autodoc_file.clone(), autodoc));
    unsafe {
        Autodoc::initialize_generic_instance(
            autodoc,
            manager,
            &autodoc_file,
            axis_id,
            false,
            std::ptr::null_mut(),
        )?
    };
    Ok(autodoc)
}

/// Java `getInstance(BaseManager, File, AxisID, boolean)`.  Open and return an autodoc
/// without a type.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_instance_file_axis_id(
    manager: Option<&'static dyn BaseManager>,
    autodoc_file: Option<&std::path::Path>,
    axis_id: AxisID,
    writable: bool,
) -> Result<*mut Autodoc, LogFileError> {
    let autodoc_file = match autodoc_file {
        None => return Ok(std::ptr::null_mut()),
        Some(autodoc_file) if !autodoc_file.exists() => {
            let _ = autodoc_file;
            return Ok(std::ptr::null_mut());
        }
        Some(autodoc_file) => autodoc_file,
    };
    let filter = AutodocFilter::new();
    if !filter.accept(autodoc_file) {
        panic!(
            "java.lang.IllegalArgumentException: {} is not an autodoc.",
            autodoc_file.to_string_lossy()
        );
    }
    if etomo_director::ARGUMENTS.lock().unwrap().is_test()
        && etomo_director::ARGUMENTS.lock().unwrap().is_debug()
    {
        eprintln!(
            "autodoc file:{}",
            utilities::java_io_file_get_absolute_path(&autodoc_file.to_string_lossy())
        );
    }
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension(&utilities::java_io_file_get_name(
                &autodoc_file.to_string_lossy(),
            ))),
            std::ptr::null_mut(),
        )
    };
    unsafe {
        Autodoc::initialize_generic_instance(
            autodoc,
            manager,
            autodoc_file,
            axis_id,
            writable,
            std::ptr::null_mut(),
        )?
    };
    Ok(autodoc)
}

/// Java `getWritableAutodocInstance`.  Return writable autodoc.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_writable_autodoc_instance(
    manager: Option<&'static dyn BaseManager>,
    autodoc_file: Option<&std::path::Path>,
) -> Result<*mut Autodoc, LogFileError> {
    let autodoc_file = match autodoc_file {
        None => return Ok(std::ptr::null_mut()),
        Some(autodoc_file) => autodoc_file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension(&utilities::java_io_file_get_name(
                &autodoc_file.to_string_lossy(),
            ))),
            std::ptr::null_mut(),
        )
    };
    unsafe {
        Autodoc::initialize_generic_instance(
            autodoc,
            manager,
            autodoc_file,
            AxisID::Only,
            true,
            std::ptr::null_mut(),
        )?
    };
    Ok(autodoc)
}

/// Java `getAutodocInstance(BaseManager, File, StringBuilder)`.
///
/// # Safety
/// See `get_matlab_instance`; `err_msg` must be null or point to a live `String`.
pub unsafe fn get_autodoc_instance_err_msg(
    manager: Option<&'static dyn BaseManager>,
    autodoc_file: Option<&std::path::Path>,
    err_msg: *mut String,
) -> Result<*mut Autodoc, LogFileError> {
    let autodoc_file = match autodoc_file {
        None => return Ok(std::ptr::null_mut()),
        Some(autodoc_file) => autodoc_file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension(&utilities::java_io_file_get_name(
                &autodoc_file.to_string_lossy(),
            ))),
            err_msg,
        )
    };
    unsafe {
        Autodoc::initialize_generic_instance(
            autodoc,
            manager,
            autodoc_file,
            AxisID::Only,
            false,
            err_msg,
        )?
    };
    Ok(autodoc)
}

/// Java `getAutodocInstance(BaseManager, File)`.
///
/// # Safety
/// See `get_matlab_instance`.
pub unsafe fn get_autodoc_instance(
    manager: Option<&'static dyn BaseManager>,
    autodoc_file: Option<&std::path::Path>,
) -> Result<*mut Autodoc, LogFileError> {
    let autodoc_file = match autodoc_file {
        None => return Ok(std::ptr::null_mut()),
        Some(autodoc_file) => autodoc_file,
    };
    let autodoc = unsafe {
        Autodoc::new(
            Some(&strip_file_extension(&utilities::java_io_file_get_name(
                &autodoc_file.to_string_lossy(),
            ))),
            std::ptr::null_mut(),
        )
    };
    unsafe {
        Autodoc::initialize_generic_instance(
            autodoc,
            manager,
            autodoc_file,
            AxisID::Only,
            false,
            std::ptr::null_mut(),
        )?
    };
    Ok(autodoc)
}

/// Java private static `stripFileExtension(File)`.
pub fn strip_file_extension_of_file(file: &std::path::Path) -> String {
    strip_file_extension(&match file.file_name() {
        // `java.io.File.getName()` returns the last path segment, or "" for an empty
        // path.
        None => String::new(),
        Some(name) => name.to_string_lossy().to_string(),
    })
}

/// Java private static `stripFileExtension(String)`.
pub fn strip_file_extension(file_name: &str) -> String {
    let units: Vec<u16> = file_name.encode_utf16().collect();
    let extension_index = match units.iter().rposition(|unit| *unit == '.' as u16) {
        None => return file_name.to_string(),
        Some(index) => index,
    };
    String::from_utf16_lossy(&units[0..extension_index])
}

/// Java `setReplacementDir(String)`.
///
/// Causes all autodocs that don't already exist, and whose location comes from an
/// environment variable to be opened in the replacement directory instead of the
/// location specified by the environment variable.
pub fn set_replacement_dir(input: Option<&str>) {
    REPLACEMENT_DIR.with(|dir| *dir.borrow_mut() = input.map(|input| input.to_string()));
}

/// Java `getReplacementDir()`.
pub fn get_replacement_dir() -> Option<String> {
    REPLACEMENT_DIR.with(|dir| dir.borrow().clone())
}

/// Java private static `getExistingUITestAxisAutodoc(File)`.
pub fn get_existing_ui_test_axis_autodoc(autodoc_file: &std::path::Path) -> *mut Autodoc {
    // The source's `UITEST_AXIS_MAP == null` guard cannot fail; the field is final.
    UITEST_AXIS_MAP.with(|map| {
        *map.borrow()
            .get(autodoc_file)
            .unwrap_or(&std::ptr::null_mut())
    })
}

/// Java private static `getExistingAutodoc(String, String)`.
pub fn get_existing_autodoc_by_file_name(file_name: &str, name: &str) -> *mut Autodoc {
    if name == UITEST_AXIS {
        // The source's `UITEST_AXIS_MAP == null` guard cannot fail; the field is final.
        // The source keys this map with a `File` elsewhere and with the `fileName`
        // string here, so this lookup can only miss.
        return UITEST_AXIS_MAP.with(|map| {
            *map.borrow()
                .get(std::path::Path::new(file_name))
                .unwrap_or(&std::ptr::null_mut())
        });
    }
    panic!(
        "java.lang.IllegalArgumentException: Illegal autodoc name: {}.",
        name
    );
}

/// Java private static `getExistingAutodoc(String)`.
pub fn get_existing_autodoc(name: &str) -> *mut Autodoc {
    if name == TILTXCORR {
        return TILTXCORR_INSTANCE.with(Cell::get);
    }
    if name == TEST {
        return TEST_INSTANCE.with(Cell::get);
    }
    if name == UITEST {
        return UITEST_INSTANCE.with(Cell::get);
    }
    if name == MTF_FILTER {
        return MTF_FILTER_INSTANCE.with(Cell::get);
    }
    if name == NEWSTACK {
        return NEWSTACK_INSTANCE.with(Cell::get);
    }
    if name == CTF_PLOTTER {
        return CTF_PLOTTER_INSTANCE.with(Cell::get);
    }
    if name == CTF_PHASE_FLIP {
        return CTF_PHASE_FLIP_INSTANCE.with(Cell::get);
    }
    if name == FLATTEN_WARP {
        return FLATTEN_WARP_INSTANCE.with(Cell::get);
    }
    if name == WARP_VOL {
        return WARP_VOL_INSTANCE.with(Cell::get);
    }
    if name == FIND_BEADS_3D {
        return FIND_BEADS_3D_INSTANCE.with(Cell::get);
    }
    if name == COMBINE_FFT {
        return COMBINE_FFT_INSTANCE.with(Cell::get);
    }
    if name == TILTALIGN {
        return TILTALIGN_INSTANCE.with(Cell::get);
    }
    if name == CCDERASER {
        return CCDERASER_INSTANCE.with(Cell::get);
    }
    if name == SOLVEMATCH {
        return SOLVEMATCH_INSTANCE.with(Cell::get);
    }
    if name == BEADTRACK {
        return BEADTRACK_INSTANCE.with(Cell::get);
    }
    if name == CPU {
        return CPU_INSTANCE.with(Cell::get);
    }
    if name == DENS_MATCH {
        return DENS_MATCH_INSTANCE.with(Cell::get);
    }
    if name == CORR_SEARCH_3D {
        return CORR_SEARCH_3D_INSTANCE.with(Cell::get);
    }
    if name == XFJOINTOMO {
        return XFJOINTOMO_INSTANCE.with(Cell::get);
    }
    if name == PEET_PRM {
        return PEET_PRM_INSTANCE.with(Cell::get);
    }
    if name == TILT {
        return TILT_INSTANCE.with(Cell::get);
    }
    if name == SIRTSETUP {
        return SIRTSETUP_INSTANCE.with(Cell::get);
    }
    if name == BLENDMONT {
        return BLENDMONT_INSTANCE.with(Cell::get);
    }
    if name == XFTOXG {
        return XFTOXG_INSTANCE.with(Cell::get);
    }
    if name == XFALIGN {
        return XFALIGN_INSTANCE.with(Cell::get);
    }
    if name == AUTOFIDSEED {
        return AUTOFIDSEED_INSTANCE.with(Cell::get);
    }
    if name == ETOMO {
        return ETOMO_INSTANCE.with(Cell::get);
    }
    if name == PROG_DEFAULTS {
        return PROG_DEFAULTS_INSTANCE.with(Cell::get);
    }
    if name == IMODCHOPCONTS {
        return IMODCHOPCONTS_INSTANCE.with(Cell::get);
    }
    if name == DUALVOLMATCH {
        return DUALVOLMATCH_INSTANCE.with(Cell::get);
    }
    if name == RESTRICT_ALIGN {
        return RESTRICT_ALIGN_INSTANCE.with(Cell::get);
    }
    if name == BATCH_RUN_TOMO {
        return BATCH_RUN_TOMO_INSTANCE.with(Cell::get);
    }
    if name == MULTIFILT_SETUP {
        return MULTIFILT_SETUP_INSTANCE.with(Cell::get);
    }
    if name == ALIGN_FRAMES {
        return ALIGN_FRAMES_INSTANCE.with(Cell::get);
    }
    if name == SUBTOMO_SETUP {
        return SUBTOMO_SETUP_INSTANCE.with(Cell::get);
    }
    if name == ALT_TOMO_SETUP {
        return ALT_TOMO_SETUP_INSTANCE.with(Cell::get);
    }
    if name == REDUCE_FILTER_VOLUME {
        return REDUCE_FILTER_VOLUME_INSTANCE.with(Cell::get);
    }
    if name == SERIES_WATCHER {
        return SERIES_WATCHER_INSTANCE.with(Cell::get);
    }
    std::ptr::null_mut()
}

/// Java `isLoaded(String)`.
pub fn is_loaded(name: &str) -> bool {
    !get_existing_autodoc(name).is_null()
}

/// Java `resetInstance(String)`.  For testing.
pub fn reset_instance(name: &str) {
    if name == TILTXCORR {
        TILTXCORR_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == TEST {
        TEST_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == UITEST {
        UITEST_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == MTF_FILTER {
        MTF_FILTER_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == NEWSTACK {
        NEWSTACK_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == CTF_PLOTTER {
        CTF_PLOTTER_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == CTF_PHASE_FLIP {
        CTF_PHASE_FLIP_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == FLATTEN_WARP {
        FLATTEN_WARP_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == WARP_VOL {
        WARP_VOL_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == FIND_BEADS_3D {
        FIND_BEADS_3D_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == COMBINE_FFT {
        COMBINE_FFT_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == TILTALIGN {
        TILTALIGN_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == CCDERASER {
        CCDERASER_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == SOLVEMATCH {
        SOLVEMATCH_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == BEADTRACK {
        BEADTRACK_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == CPU {
        CPU_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == DENS_MATCH {
        DENS_MATCH_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == CORR_SEARCH_3D {
        CORR_SEARCH_3D_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == XFJOINTOMO {
        XFJOINTOMO_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == PEET_PRM {
        PEET_PRM_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == TILT {
        TILT_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == SIRTSETUP {
        SIRTSETUP_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == BLENDMONT {
        BLENDMONT_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == XFTOXG {
        XFTOXG_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == XFALIGN {
        XFALIGN_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == AUTOFIDSEED {
        AUTOFIDSEED_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == ETOMO {
        ETOMO_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == PROG_DEFAULTS {
        PROG_DEFAULTS_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == IMODCHOPCONTS {
        IMODCHOPCONTS_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == DUALVOLMATCH {
        DUALVOLMATCH_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == RESTRICT_ALIGN {
        RESTRICT_ALIGN_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == BATCH_RUN_TOMO {
        BATCH_RUN_TOMO_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == MULTIFILT_SETUP {
        MULTIFILT_SETUP_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == ALIGN_FRAMES {
        ALIGN_FRAMES_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == SUBTOMO_SETUP {
        SUBTOMO_SETUP_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == ALT_TOMO_SETUP {
        ALT_TOMO_SETUP_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == REDUCE_FILTER_VOLUME {
        REDUCE_FILTER_VOLUME_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else if name == SERIES_WATCHER {
        SERIES_WATCHER_INSTANCE.with(|instance| instance.set(std::ptr::null_mut()));
    } else {
        panic!(
            "java.lang.IllegalArgumentException: Illegal autodoc name: {}.",
            name
        );
    }
}

/// Java private static `setInstance(String, Autodoc)`.
///
/// Override an old autodoc instance with a new one.
///
/// # Safety
/// `autodoc` must be null or point to a live `Autodoc`.
pub unsafe fn set_instance(name: &str, autodoc: *mut Autodoc) -> bool {
    if name == TILTXCORR {
        TILTXCORR_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == TEST {
        TEST_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == UITEST {
        UITEST_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == MTF_FILTER {
        MTF_FILTER_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == NEWSTACK {
        NEWSTACK_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == CTF_PLOTTER {
        CTF_PLOTTER_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == CTF_PHASE_FLIP {
        CTF_PHASE_FLIP_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == FLATTEN_WARP {
        FLATTEN_WARP_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == WARP_VOL {
        WARP_VOL_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == FIND_BEADS_3D {
        FIND_BEADS_3D_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == COMBINE_FFT {
        COMBINE_FFT_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == TILTALIGN {
        TILTALIGN_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == CCDERASER {
        CCDERASER_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == SOLVEMATCH {
        SOLVEMATCH_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == BEADTRACK {
        BEADTRACK_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == CPU {
        CPU_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == DENS_MATCH {
        DENS_MATCH_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == CORR_SEARCH_3D {
        CORR_SEARCH_3D_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == XFJOINTOMO {
        XFJOINTOMO_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == PEET_PRM {
        PEET_PRM_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == TILT {
        TILT_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == SIRTSETUP {
        SIRTSETUP_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == BLENDMONT {
        BLENDMONT_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == XFTOXG {
        XFTOXG_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == XFALIGN {
        XFALIGN_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == AUTOFIDSEED {
        AUTOFIDSEED_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == ETOMO {
        ETOMO_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == PROG_DEFAULTS {
        PROG_DEFAULTS_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == IMODCHOPCONTS {
        IMODCHOPCONTS_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == DUALVOLMATCH {
        DUALVOLMATCH_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == RESTRICT_ALIGN {
        RESTRICT_ALIGN_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == BATCH_RUN_TOMO {
        BATCH_RUN_TOMO_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == MULTIFILT_SETUP {
        MULTIFILT_SETUP_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == ALIGN_FRAMES {
        ALIGN_FRAMES_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == SUBTOMO_SETUP {
        SUBTOMO_SETUP_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == ALT_TOMO_SETUP {
        ALT_TOMO_SETUP_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == REDUCE_FILTER_VOLUME {
        REDUCE_FILTER_VOLUME_INSTANCE.with(|instance| instance.set(autodoc));
    } else if name == SERIES_WATCHER {
        SERIES_WATCHER_INSTANCE.with(|instance| instance.set(autodoc));
    } else {
        return false;
    }
    true
}

/// Java `endsWithAutodocExtension(String)`.
pub fn ends_with_autodoc_extension(path: Option<&str>) -> bool {
    let path = match path {
        None => return false,
        Some(path) => path,
    };
    path.ends_with(&extension::DEFAULT.to_string())
        || path.ends_with(&extension::MATLAB.to_string())
}

/// Java's nested `public static final class Extension`.  Each constant is a distinct
/// object, so the typesafe enum is a Rust enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Extension {
    /// Java `Extension.DEFAULT`, built with "adoc".
    Default,
    /// Java private `Extension.MATLAB`, built with "prm".
    Matlab,
}

/// The two `Extension` constants, named as the source names them.
pub mod extension {
    /// Java `Extension.DEFAULT`.
    pub const DEFAULT: super::Extension = super::Extension::Default;
    /// Java private `Extension.MATLAB`.
    pub const MATLAB: super::Extension = super::Extension::Matlab;
}

impl Extension {
    /// Java private field `extensionString`, and `getExtensionString()`.
    pub fn get_extension_string(self) -> String {
        match self {
            Self::Default => "adoc".to_string(),
            Self::Matlab => "prm".to_string(),
        }
    }
}

/// Java `Extension.toString()`.
impl std::fmt::Display for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ".{}", self.get_extension_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{ends_with_autodoc_extension, extension, strip_file_extension_of_file};

    /// `stripFileExtension(File)` runs `getName()` first, so a directory prefix is
    /// dropped before the last dot is found.
    #[test]
    fn file_name_helpers_match_the_source() {
        assert_eq!(
            strip_file_extension_of_file(std::path::Path::new("/a/b/etomo.adoc")),
            "etomo"
        );
        assert_eq!(
            strip_file_extension_of_file(std::path::Path::new("/a.b/etomo")),
            "etomo"
        );
        assert_eq!(extension::DEFAULT.to_string(), ".adoc");
        assert_eq!(extension::MATLAB.to_string(), ".prm");
        assert!(ends_with_autodoc_extension(Some("x.adoc")));
        assert!(ends_with_autodoc_extension(Some("x.prm")));
        assert!(!ends_with_autodoc_extension(Some("x.txt")));
        assert!(!ends_with_autodoc_extension(None));
    }
}
