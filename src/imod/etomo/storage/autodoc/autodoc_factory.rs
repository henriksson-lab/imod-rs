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

/// The 38 autodoc names that have a `private static` instance field in Java.
///
/// `AutodocFactory` declares one static per name and three 38-arm dispatch
/// chains over them (`getExistingAutodoc`, `resetInstance`, `setInstance`).
/// The field *set* is the data, so it is one table here keyed by the same
/// constants; membership of this array is what reproduces Java's
/// "Illegal autodoc name" behaviour, which a bare map could not distinguish
/// from "known name, not yet loaded".
static INSTANCE_NAMES: [&str; 38] = [
    TILTXCORR,
    TEST,
    UITEST,
    MTF_FILTER,
    NEWSTACK,
    CTF_PLOTTER,
    CTF_PHASE_FLIP,
    FLATTEN_WARP,
    WARP_VOL,
    FIND_BEADS_3D,
    COMBINE_FFT,
    TILTALIGN,
    CCDERASER,
    SOLVEMATCH,
    BEADTRACK,
    CPU,
    DENS_MATCH,
    CORR_SEARCH_3D,
    XFJOINTOMO,
    PEET_PRM,
    TILT,
    SIRTSETUP,
    BLENDMONT,
    XFTOXG,
    XFALIGN,
    AUTOFIDSEED,
    ETOMO,
    PROG_DEFAULTS,
    IMODCHOPCONTS,
    DUALVOLMATCH,
    RESTRICT_ALIGN,
    BATCH_RUN_TOMO,
    MULTIFILT_SETUP,
    ALIGN_FRAMES,
    SUBTOMO_SETUP,
    ALT_TOMO_SETUP,
    REDUCE_FILTER_VOLUME,
    SERIES_WATCHER,
];

thread_local! {
    /// The Java `private static` instance fields, as one table.
    static INSTANCES: RefCell<HashMap<&'static str, *mut Autodoc>> =
        RefCell::new(HashMap::new());
    /// Java private static final `UITEST_AXIS_MAP`.
    static UITEST_AXIS_MAP: RefCell<HashMap<std::path::PathBuf, *mut Autodoc>> =
        RefCell::new(HashMap::new());
    /// Java private static `replacementDir`, initialised to null.
    static REPLACEMENT_DIR: RefCell<Option<String>> = const { RefCell::new(None) };
    /// Owns every `Autodoc` this factory builds.  Java's owner is the collector:
    /// a named instance is held by a `private static` field for the life of the
    /// process, and an unmanaged one lives as long as its caller keeps it.  This
    /// module is the sole producer, so one arena here gives every allocation an
    /// owner -- without it the raw pointers these functions return are never
    /// reclaimed by anyone.  A `Box` does not move its contents, so the pointers
    /// stay valid as the `Vec` grows.
    static OWNED_AUTODOCS: RefCell<Vec<Box<Autodoc>>> = const { RefCell::new(Vec::new()) };
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
    autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe { Autodoc::new(Some(name), std::ptr::null_mut()) });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe { Autodoc::new(Some(name), std::ptr::null_mut()) });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe { Autodoc::new(Some(name), std::ptr::null_mut()) });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
    set_instance(name, autodoc);
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension_of_file(file)),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension_of_file(file)),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension_of_file(file)),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension_of_file(file)),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension_of_file(file)),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension_of_file(autodoc_file)),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension_of_file(file)),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
                set_instance(autodoc_name, autodoc);
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
    autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension(autodoc_file_name)),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension(&utilities::java_io_file_get_name(
                    &autodoc_file.to_string_lossy(),
                ))),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension(&utilities::java_io_file_get_name(
                    &autodoc_file.to_string_lossy(),
                ))),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension(&utilities::java_io_file_get_name(
                    &autodoc_file.to_string_lossy(),
                ))),
                err_msg,
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    let autodoc: *mut Autodoc = OWNED_AUTODOCS.with_borrow_mut(|owned| {
        owned.push(unsafe {
            Autodoc::new(
                Some(&strip_file_extension(&utilities::java_io_file_get_name(
                    &autodoc_file.to_string_lossy(),
                ))),
                std::ptr::null_mut(),
            )
        });
        let autodoc: *mut Autodoc = &mut **owned.last_mut().unwrap();
        autodoc
    });
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
    INSTANCES.with_borrow(|instances| instances.get(name).copied().unwrap_or(std::ptr::null_mut()))
}

/// Java `isLoaded(String)`.
pub fn is_loaded(name: &str) -> bool {
    !get_existing_autodoc(name).is_null()
}

/// Java `resetInstance(String)`.  For testing.
pub fn reset_instance(name: &str) {
    let Some(name) = INSTANCE_NAMES.iter().copied().find(|known| *known == name) else {
        panic!(
            "java.lang.IllegalArgumentException: Illegal autodoc name: {}.",
            name
        );
    };
    INSTANCES.with_borrow_mut(|instances| instances.insert(name, std::ptr::null_mut()));
}

/// Java private static `setInstance(String, Autodoc)`.
///
/// Override an old autodoc instance with a new one.
///
pub fn set_instance(name: &str, autodoc: *mut Autodoc) -> bool {
    let Some(name) = INSTANCE_NAMES.iter().copied().find(|known| *known == name) else {
        return false;
    };
    INSTANCES.with_borrow_mut(|instances| instances.insert(name, autodoc));
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
