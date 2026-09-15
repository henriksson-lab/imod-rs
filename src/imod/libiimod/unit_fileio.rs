pub type fortStrLen_t = i32;
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{
    b3d_error, b3d_get_error as b3dGetError, b3d_milli_sleep as b3dMilliSleep,
    b3d_set_store_error as b3dSetStoreError, fortran_string, imod_backup_file as imodBackupFile,
};
use crate::imod::libiimod::iihdf::hdf_write_global_adoc as hdfWriteGlobalAdoc;
use crate::imod::libiimod::iimage::{
    IiFileCheckFunction as IIFileCheckFunction, IiSectionFunc as iiSectionFunc, ImodImageFile,
    ii_allow_multi_volume as iiAllowMultiVolume, ii_close as iiClose, ii_delete as iiDelete,
    ii_fill_mrc_header as iiFillMrcHeader, ii_fopen_new_volume as iiFOpenNewVolume,
    ii_fopen_volume as iiFOpenVolume, ii_get_adoc_index as iiGetAdocIndex,
    ii_insert_check_function as iiInsertCheckFunction, ii_open as iiOpen, ii_open_new as iiOpenNew,
    ii_read_section_callback as iiReadSection,
    ii_read_section_float_callback as iiReadSectionFloat, ii_set_chunk_sizes as iiSetChunkSizes,
    ii_sync_from_mrc_header as iiSyncFromMrcHeader,
    ii_transfer_adoc_sections as iiTransferAdocSections, ii_write_section as iiWriteSection,
    ii_write_section_float as iiWriteSectionFloat,
};
use crate::imod::libiimod::iimrc::ii_mrc_check as iiMRCCheck;
use crate::imod::libiimod::iishrmem::ii_shr_mem_check_size as iiShrMemCheckSize;
use crate::imod::libiimod::iitif::tiff_set_string_tag_to_print as tiffSetStringTagToPrint;
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_getdcsize};
use std::cell::{Cell, RefCell};
use std::io::Write as _;
use std::process::exit;

pub struct Unit {
    pub ii_file: *mut ImodImageFile,
    header: UnitHeader,
    pub current_sec: i32,
    pub current_line: i32,
    /// C `char *tailName` (`unit_fileio.c:119`), "pointer to filename only in
    /// fname".  Now the index of that byte within `iiFile->filename` rather
    /// than a pointer into it, because the name owns its storage; nothing in
    /// the C or in this translation ever reads the field.
    pub tail_name: usize,
    pub attribute: i32,
    pub being_used: bool,
    pub read_only: bool,
    pub no_convert: bool,
}

/// A unit owns its working MRC header.  Synchronization copies it into the
/// associated image record at the existing unit API boundary.
enum UnitHeader {
    None,
    Header(MrcHeader),
}

/// Per-thread Fortran unit state.  Boxes keep an individual unit's address
/// stable after callers receive its transient raw ABI cursor.  The legacy unit
/// API itself has no cross-thread ownership contract, and `MrcHeader` carries
/// non-Send file handles, so thread-local storage faithfully avoids inventing
/// one while making ownership explicit.
struct UnitTable {
    units: Vec<Box<Unit>>,
    map: Vec<Option<usize>>,
    no_convert_units: Vec<i32>,
}
pub const IIUNIT_SWAPPED: i32 = 1;
pub const IIUNIT_BYTES_SIGNED: i32 = 2;
pub const IIFILE_DEFAULT: i32 = -(1 as i32);
pub const IIFILE_TIFF: i32 = 1 as i32;
pub const IIFILE_MRC: i32 = 2 as i32;
pub const IIFILE_RAW: i32 = 4 as i32;
pub const IIFILE_HDF: i32 = 5 as i32;
pub const IIFILE_SHR_MEM: i32 = 8 as i32;
pub const IIFORMAT_COMPLEX: i32 = 3 as i32;
pub const MAX_UNIT: i32 = 1000 as i32;
pub const UNIT_ATBUT_RO: i32 = 1 as i32;
pub const UNIT_ATBUT_NEW: i32 = 2 as i32;
pub const UNIT_ATBUT_OLD: i32 = 3 as i32;
pub const UNIT_ATBUT_SCRATCH: i32 = 4 as i32;
thread_local! {
    static UNIT_TABLE: RefCell<UnitTable> = const { RefCell::new(UnitTable {
        units: Vec::new(),
        map: Vec::new(),
        no_convert_units: Vec::new(),
    }) };
    /// C's process-global output options are consumed only by this module's
    /// per-thread unit API.  Cells retain the mutation semantics without raw
    /// globals or synchronization that the original ABI never provided.
    static UNIT_OPTIONS: UnitOptions = const { UnitOptions::new() };
    /// `sWritePartMess` was only used as a non-NULL marker between the public
    /// whole/subarray write calls and their immediate `iiu_write_sec_part`
    /// call.  The text itself was never read.
    static WRITE_PART_MESSAGE: Cell<bool> = const { Cell::new(false) };
}

struct UnitOptions {
    brief_header: Cell<i32>,
    print_header: Cell<i32>,
    exit_on_error: Cell<i32>,
    store_error: Cell<i32>,
}

impl UnitOptions {
    const fn new() -> Self {
        Self {
            brief_header: Cell::new(-1),
            print_header: Cell::new(1),
            exit_on_error: Cell::new(1),
            store_error: Cell::new(-1),
        }
    }
}
/// Opens a unit through the crate-owned image I/O path.
///
/// The native API owns ordinary Rust text. [`iiu_open_ffi`] is the sole C
/// string adapter for callers which still use the legacy exported symbol.
pub unsafe fn iiu_open(iunit: i32, name: &str, attribute: &str) -> i32 {
    let mut u: *mut Unit = ::core::ptr::null_mut::<Unit>();
    let mut mode: i32 = 0;
    // `unit_fileio.c:207`: the `fopen` mode strings.  `iiFOpen`/`iiOpenNew`
    // now take the mode as a `&str`, so the table is plain Rust.
    let modes: [&str; 4] = ["rb", "rb+", "wb", "wb+"];
    u = find_new_unit(iunit);
    iiu_memory_error(u, "ERROR: iiuOpen - Allocating new unit");
    (*u).being_used = true;
    (*u).read_only = false;
    (*u).current_sec = 0 as i32;
    (*u).current_line = 0 as i32;
    if matches!(attribute.as_bytes().first(), Some(b'R' | b'r')) {
        mode = 0 as i32;
        (*u).attribute = UNIT_ATBUT_RO;
        (*u).read_only = true;
    }
    if name.is_empty() {
        iiInsertCheckFunction(Some(iiMRCCheck), 0 as i32);
    }
    if matches!(attribute.as_bytes().first(), Some(b'N' | b'n')) {
        if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none() {
            if imodBackupFile(name) != 0 {
                // `imod_backup_file` currently retains its historical integer
                // status, so take the OS diagnostic immediately after it
                // fails instead of reaching through libc's errno/strerror
                // globals.  This is the same operation that supplied errno
                // to the C caller.
                let system_error = std::io::Error::last_os_error();
                let _ = ImodFile::Stdout.write_all(
                    format!("\nWARNING: iiuOpen - Could not rename '{name}' to '{name}~'\n")
                        .as_bytes(),
                );
                if system_error.raw_os_error().is_some_and(|code| code != 0) {
                    let _ = ImodFile::Stdout
                        .write_all(format!("WARNING: from system - {system_error}\n").as_bytes());
                }
            }
        }
        mode = 3 as i32;
        (*u).attribute = UNIT_ATBUT_NEW;
    }
    if matches!(attribute.as_bytes().first(), Some(b'O' | b'o')) {
        mode = 1 as i32;
        (*u).attribute = UNIT_ATBUT_OLD;
    }
    if matches!(attribute.as_bytes().first(), Some(b'S' | b's')) {
        mode = 3 as i32;
        (*u).attribute = UNIT_ATBUT_SCRATCH;
    }
    if mode == 3 as i32 {
        (*u).ii_file = iiOpenNew(name.as_bytes(), modes[mode as usize], IIFILE_DEFAULT);
        if (*u).ii_file.is_null() {
            b3d_error(
                Some(&mut ImodFile::Stdout),
                format_args!("\nERROR: iiuOpen - Opening new output file\n"),
            );
            if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
                exit(1 as i32);
            } else {
                return 1 as i32;
            }
        }
        {
            use std::io::Write;
            let _ = ImodFile::Stdout
                .write_all(format!("\n NEW image file on unit {iunit:>3} : {name}\n").as_bytes());
        }
        let _ = ImodFile::Stdout.flush();
    } else {
        (*u).ii_file = iiOpen(name.as_bytes(), modes[mode as usize]);
        if (*u).ii_file.is_null() {
            b3d_error(
                Some(&mut ImodFile::Stdout),
                format_args!("\nERROR: iiuOpen - Could not open '{}'\n", name),
            );
            if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
                exit(1 as i32);
            } else {
                return 1 as i32;
            }
        }
        if !((*(*u).ii_file).write_section.is_some()
            && (*(*u).ii_file).write_section_float.is_some())
            && !(*u).read_only
        {
            b3d_error(
                Some(&mut ImodFile::Stdout),
                format_args!(
                    "\nERROR: iiuOpen - Non-MRC-type file '{}' with no write function must be opened read-only\n",
                    name
                ),
            );
            if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
                exit(1 as i32);
            } else {
                return 1 as i32;
            }
        }
        if (*(*u).ii_file).file == IIFILE_TIFF {
            if (*(*u).ii_file).mode < 0 as i32 {
                b3d_error(
                    Some(&mut ImodFile::Stdout),
                    format_args!(
                        "\nERROR: iiuOpen - TIFF file '{}' has a data type that is not supported\n",
                        name
                    ),
                );
                if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
                    exit(1 as i32);
                } else {
                    return 1 as i32;
                }
            }
        }
    }
    if (*(*u).ii_file).file != IIFILE_MRC
        && (*(*u).ii_file).file != IIFILE_RAW
        && (*(*u).ii_file).file != IIFILE_HDF
        && (*(*u).ii_file).file != IIFILE_SHR_MEM
    {
        let mut header = MrcHeader::default();
        if iiFillMrcHeader((*u).ii_file, &raw mut header) != 0 {
            b3d_error(
                Some(&mut ImodFile::Stdout),
                format_args!(
                    "\nERROR: iiuOpen - file '{}' is not a format that provides an MRC-like header and cannot be read\n",
                    name
                ),
            );
            if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
                exit(1 as i32);
            } else {
                return 1 as i32;
            }
        }
        (*u).header = UnitHeader::Header(header);
    } else {
        let header = (*(*u).ii_file)
            .mrc_header
            .as_deref()
            .cloned()
            .unwrap_or_default();
        (*u).header = UnitHeader::Header(header);
    }
    // `unit_fileio.c:273-280`: the last '/' or the last '\\', whichever is
    // later, plus one; index 0 when there is neither.
    let name = (*(*u).ii_file).filename.as_deref().unwrap_or_default();
    let slash = name.bytes().rposition(|b| b == b'/');
    let tailback = name.bytes().rposition(|b| b == b'\\');
    (*u).tail_name = match slash.max(tailback) {
        Some(pos) => pos + 1,
        None => 0,
    };
    0
}

/// C ABI entry point for legacy callers.  C-string decoding is deliberately
/// contained here; the implementation above has no C-string storage.
#[unsafe(export_name = "iiu_open")]
pub unsafe extern "C" fn iiu_open_ffi(
    iunit: i32,
    name: *const ::core::ffi::c_char,
    attribute: *const ::core::ffi::c_char,
) -> i32 {
    let name = if name.is_null() {
        String::new()
    } else {
        core::ffi::CStr::from_ptr(name)
            .to_string_lossy()
            .into_owned()
    };
    let attribute = if attribute.is_null() {
        String::new()
    } else {
        core::ffi::CStr::from_ptr(attribute)
            .to_string_lossy()
            .into_owned()
    };
    iiu_open(iunit, &name, &attribute)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuopen_(
    mut iunit: *mut i32,
    mut name: *mut ::core::ffi::c_char,
    mut attribute: *mut ::core::ffi::c_char,
    mut name_l: fortStrLen_t,
    mut attr_l: fortStrLen_t,
) -> i32 {
    let name = fortran_string(name, name_l as i32);
    let attribute = fortran_string(attribute, attr_l as i32);
    iiu_open(*iunit, &name, &attribute)
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_close(mut iunit: i32) {
    let mut u: *mut Unit = ::core::ptr::null_mut::<Unit>();
    let mut trial: i32 = 0;
    let mut delay: i32 = 500 as i32;
    let mut numTry: i32 = 10 as i32;
    let mut unit: i32 = iunit - 1 as i32;
    UNIT_TABLE.with(|unit_table| {
        let mut table = unit_table.borrow_mut();
        table
            .no_convert_units
            .retain(|&listed_unit| listed_unit as i32 != iunit);
        if let Some(index) = table.map.get(unit as usize).copied().flatten() {
            u = table.units[index].as_mut();
            if (*u).being_used {
                (*u).being_used = false;
                iiClose((*u).ii_file);
                if (*u).attribute == UNIT_ATBUT_SCRATCH {
                    trial = 0 as i32;
                    while trial < numTry {
                        let scratch = (*(*u).ii_file).filename.clone().unwrap_or_default();
                        if std::fs::remove_file(&scratch).is_ok() {
                            break;
                        }
                        if trial < numTry - 1 as i32 {
                            b3dMilliSleep(delay);
                        }
                        trial += 1;
                    }
                }
                (*u).header = UnitHeader::None;
                iiDelete((*u).ii_file);
            }
            table.map[unit as usize] = None;
        }
    });
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imclose_(mut iunit: *mut i32) {
    iiu_close(*iunit);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuclose_(mut iunit: *mut i32) {
    iiu_close(*iunit);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_get_ii_file(mut iunit: i32) -> *mut ImodImageFile {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_get_ii_file", 1 as i32, 0 as i32);
    return (*u).ii_file;
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_ret_num_volumes(mut iunit: i32) -> i32 {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_ret_num_volumes", 1 as i32, 0 as i32);
    return if (*(*u).ii_file).dataset_id != 0 {
        (*(*u).ii_file).num_volumes
    } else {
        0 as i32
    };
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuretnumvolumes_(mut iunit: *mut i32) -> i32 {
    return iiu_ret_num_volumes(*iunit);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_volume_open(mut newUnit: i32, mut mainUnit: i32, mut volIndex: i32) -> i32 {
    let fp: Option<crate::imod::libcfshr::b3dutil::ImodFile>;
    let mut unew: *mut Unit = find_new_unit(newUnit);
    iiu_memory_error(unew, "ERROR:  - Allocating new unit");
    // `find_new_unit` can grow the owned unit vector, so obtain the main
    // unit only after its storage is stable for this operation.
    let mut u: *mut Unit = lookup_unit(mainUnit, "iiuOpenVolume", 1 as i32, 0 as i32);
    (*unew).being_used = true;
    (*unew).read_only = (*u).read_only;
    (*unew).current_sec = 0 as i32;
    (*unew).current_line = 0 as i32;
    (*unew).attribute = if (*u).attribute == UNIT_ATBUT_SCRATCH {
        UNIT_ATBUT_NEW
    } else {
        (*u).attribute
    };
    if volIndex < 0 as i32 {
        fp = iiFOpenNewVolume(&mut *(*u).ii_file);
        volIndex = (*(*u).ii_file).num_volumes - 1 as i32;
    } else {
        fp = iiFOpenVolume(&mut *(*u).ii_file, volIndex);
    }
    if fp.is_none() {
        if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
            exit(1 as i32);
        } else {
            return 1 as i32;
        }
    }
    (*unew).ii_file = (&(*(*u).ii_file).ii_volumes)[volIndex as isize as usize]
        .expect("opened HDF volume has a cursor")
        .as_ptr();
    let header = (*(*unew).ii_file)
        .mrc_header
        .as_deref()
        .cloned()
        .unwrap_or_default();
    (*unew).header = UnitHeader::Header(header);
    return 0 as i32;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuvolumeopen_(
    mut newUnit: *mut i32,
    mut mainUnit: *mut i32,
    mut volIndex: *mut i32,
) -> i32 {
    return iiu_volume_open(*newUnit, *mainUnit, *volIndex);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_ret_adoc_index(mut iunit: i32, mut global: i32, mut openMdocOrNew: i32) -> i32 {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_ret_adoc_index", 1 as i32, 0 as i32);
    return iiGetAdocIndex(&mut *(*u).ii_file, global, openMdocOrNew);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuretadocindex_(
    mut iunit: *mut i32,
    mut global: *mut i32,
    mut openMdocOrNew: *mut i32,
) -> i32 {
    let mut err: i32 = iiu_ret_adoc_index(*iunit, *global, *openMdocOrNew);
    return if err < 0 as i32 { err } else { err + 1 as i32 };
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_trans_adoc_sections(mut toUnit: i32, mut fromUnit: i32) -> i32 {
    let mut uto: *mut Unit = lookup_unit(toUnit, "iiu_trans_adoc_sections", 1 as i32, 0 as i32);
    let mut ufrom: *mut Unit = lookup_unit(fromUnit, "iiu_trans_adoc_sections", 1 as i32, 0 as i32);
    if (*(*uto).ii_file).adoc_index >= 0 as i32 && (*(*ufrom).ii_file).adoc_index >= 0 as i32 {
        return iiTransferAdocSections(&*(*ufrom).ii_file, &*(*uto).ii_file);
    }
    return 0 as i32;
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_write_global_adoc(mut iunit: i32) -> i32 {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_write_global_adoc", 1 as i32, 0 as i32);
    if hdfWriteGlobalAdoc(&mut *(*u).ii_file) != 0 {
        if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
            exit(1 as i32);
        } else {
            return 1 as i32;
        }
    }
    return 0 as i32;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwriteglobaladoc_(mut iunit: *mut i32) -> i32 {
    return iiu_write_global_adoc(*iunit);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_ret_chunk_sizes(
    mut iunit: i32,
    mut xSize: *mut i32,
    mut ySize: *mut i32,
    mut zSize: *mut i32,
) {
    let mut u: *mut Unit = lookup_unit(iunit, "iiuRetChunkSize", 1 as i32, 0 as i32);
    *xSize = (*(*u).ii_file).tile_size_x;
    *ySize = (*(*u).ii_file).tile_size_y;
    *zSize = (*(*u).ii_file).z_chunk_size;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuretchunksizes_(
    mut iunit: *mut i32,
    mut xSize: *mut i32,
    mut ySize: *mut i32,
    mut zSize: *mut i32,
) {
    iiu_ret_chunk_sizes(*iunit, xSize, ySize, zSize);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_alt_chunk_sizes(
    mut iunit: i32,
    mut xSize: i32,
    mut ySize: i32,
    mut zSize: i32,
) -> i32 {
    let mut u: *mut Unit = lookup_unit(iunit, "iiuAltChunkSize", 1 as i32, 0 as i32);
    return iiSetChunkSizes(&mut *(*u).ii_file, xSize, ySize, zSize);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiualtchunksizes_(
    mut iunit: *mut i32,
    mut xSize: *mut i32,
    mut ySize: *mut i32,
    mut zSize: *mut i32,
) -> i32 {
    return iiu_alt_chunk_sizes(*iunit, *xSize, *ySize, *zSize);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_set_hdf_compression(mut iunit: i32, mut compression: i32) {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_set_hdf_compression", 1 as i32, 0 as i32);
    (*(*u).ii_file).hdf_compression = compression;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiusethdfcompression_(mut iunit: *mut i32, mut compression: *mut i32) {
    iiu_set_hdf_compression(*iunit, *compression);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_set_position(mut iunit: i32, mut section: i32, mut line: i32) {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_set_position", 1 as i32, 0 as i32);
    (*u).current_sec = section;
    (*u).current_line = line;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imposn_(mut iunit: *mut i32, mut section: *mut i32, mut line: *mut i32) {
    iiu_set_position(*iunit, *section, *line);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiusetposition_(
    mut iunit: *mut i32,
    mut section: *mut i32,
    mut line: *mut i32,
) {
    iiu_set_position(*iunit, *section, *line);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_read_section(mut iunit: i32, mut array: *mut ::core::ffi::c_void) -> i32 {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_read_section", 0 as i32, 1 as i32);
    if u.is_null() {
        return -(1 as i32);
    }
    return iiu_read_sec_part(
        iunit,
        array,
        (*(*u).ii_file).nx,
        0 as i32,
        (*(*u).ii_file).nx - 1 as i32,
        0 as i32,
        (*(*u).ii_file).ny - 1 as i32,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiureadsection_(
    mut iunit: *mut i32,
    mut array: *mut ::core::ffi::c_void,
) -> i32 {
    return iiu_read_section(*iunit, array);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_read_sec_part(
    mut iunit: i32,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: i32,
    mut indX0: i32,
    mut indX1: i32,
    mut indY0: i32,
    mut indY1: i32,
) -> i32 {
    let mut err: i32 = 0;
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_read_sec_part", 0 as i32, 1 as i32);
    if u.is_null() {
        return -(1 as i32);
    }
    (*(*u).ii_file).llx = indX0;
    (*(*u).ii_file).urx = indX1;
    (*(*u).ii_file).lly = indY0;
    (*(*u).ii_file).ury = indY1;
    (*(*u).ii_file).pad_left = 0 as i32;
    (*(*u).ii_file).pad_right = nxdim - (indX1 + 1 as i32 - indX0);
    if (*u).no_convert {
        err = iiReadSection((*u).ii_file, array as *mut u8, (*u).current_sec);
    } else {
        err = iiReadSectionFloat((*u).ii_file, array as *mut u8, (*u).current_sec);
    }
    (*u).current_sec += 1;
    (*u).current_line = 0 as i32;
    if err != 0 && UNIT_OPTIONS.with(|options| options.store_error.get()) < 0 {
        {
            use std::io::Write;
            let _ = ImodFile::Stdout.write_all(format!("\n{}\n", b3dGetError()).as_bytes());
        }
    }
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiureadsecpart_(
    mut iunit: *mut i32,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: *mut i32,
    mut indX0: *mut i32,
    mut indX1: *mut i32,
    mut indY0: *mut i32,
    mut indY1: *mut i32,
) -> i32 {
    return iiu_read_sec_part(*iunit, array, *nxdim, *indX0, *indX1, *indY0, *indY1);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_read_lines(
    mut iunit: i32,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: i32,
) -> i32 {
    let mut err: i32 = 0;
    let mut iz: i32 = 0;
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_read_lines", 0 as i32, 1 as i32);
    if u.is_null() {
        return -(1 as i32);
    }
    iz = (*u).current_sec;
    if setup_current_lines(u, numLines) != 0 {
        return -(2 as i32);
    }
    if (*u).no_convert {
        err = iiReadSection((*u).ii_file, array as *mut u8, iz);
    } else {
        err = iiReadSectionFloat((*u).ii_file, array as *mut u8, iz);
    }
    if err != 0 && UNIT_OPTIONS.with(|options| options.store_error.get()) < 0 {
        {
            use std::io::Write;
            let _ = ImodFile::Stdout.write_all(format!("\n{}\n", b3dGetError()).as_bytes());
        }
    }
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiureadlines_(
    mut iunit: *mut i32,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: *mut i32,
) -> i32 {
    return iiu_read_lines(*iunit, array, *numLines);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_write_section(mut iunit: i32, mut array: *mut ::core::ffi::c_void) -> i32 {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        "iiu_write_section",
        UNIT_OPTIONS.with(|options| options.exit_on_error.get()),
        2,
    );
    if u.is_null() {
        return -(1 as i32);
    }
    WRITE_PART_MESSAGE.with(|message| message.set(true));
    return iiu_write_sec_part(
        iunit,
        array,
        (*(*u).ii_file).nx,
        0 as i32,
        0 as i32,
        (*(*u).ii_file).nx - 1 as i32,
        0 as i32,
        (*(*u).ii_file).ny - 1 as i32,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwritesection_(
    mut iunit: *mut i32,
    mut array: *mut ::core::ffi::c_void,
) -> i32 {
    return iiu_write_section(*iunit, array);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iwrsec_(mut iunit: *mut i32, mut array: *mut ::core::ffi::c_void) {
    iiu_write_section(*iunit, array);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_write_subarray(
    mut iunit: i32,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: i32,
    mut ixStart: i32,
    mut iyStart: i32,
    mut iyEnd: i32,
) -> i32 {
    let mut err: i32 = 0;
    let mut u: *mut Unit = lookup_unit(
        iunit,
        "iiu_write_subarray",
        UNIT_OPTIONS.with(|options| options.exit_on_error.get()),
        2,
    );
    if u.is_null() {
        return -(1 as i32);
    }
    WRITE_PART_MESSAGE.with(|message| message.set(true));
    return iiu_write_sec_part(
        iunit,
        array,
        nxdim,
        ixStart,
        0 as i32,
        (*(*u).ii_file).nx - 1 as i32,
        iyStart,
        iyEnd,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwritesubarray_(
    mut iunit: *mut i32,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: *mut i32,
    mut ixStart: *mut i32,
    mut iyStart: *mut i32,
    mut iyEnd: *mut i32,
) -> i32 {
    return iiu_write_subarray(*iunit, array, *nxdim, *ixStart, *iyStart, *iyEnd);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_write_sec_part(
    mut iunit: i32,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: i32,
    mut ixStart: i32,
    mut indX0: i32,
    mut indX1: i32,
    mut iyStart: i32,
    mut iyEnd: i32,
) -> i32 {
    let mut err: i32 = 0;
    let use_message = WRITE_PART_MESSAGE.with(|message| message.replace(false));
    let mut u: *mut Unit = lookup_unit(
        iunit,
        "iiu_write_sec_part",
        UNIT_OPTIONS.with(|options| options.exit_on_error.get()),
        2,
    );
    let mut arrStart: *mut u8 = ::core::ptr::null_mut::<u8>();
    if u.is_null() {
        return -(1 as i32);
    }
    if (*(*u).ii_file).file != IIFILE_HDF && (indX0 != 0 || indX1 != (*(*u).ii_file).nx - 1 as i32)
    {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "\nERROR: iiuWriteSecPart - Attempting to write to a portion of a line for a non-HDF file, unit {}\n",
                iunit
            ),
        );
        if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
            exit(1 as i32);
        } else {
            return -(2 as i32);
        }
    }
    if (*(*u).ii_file).file == IIFILE_TIFF {
        let header = match &mut (*u).header {
            UnitHeader::Header(header) => &raw mut *header,
            UnitHeader::None => unreachable!("an open TIFF unit always has a header"),
        };
        iiSyncFromMrcHeader(&mut *(*u).ii_file, &mut *header);
    }
    (*(*u).ii_file).llx = indX0;
    (*(*u).ii_file).urx = indX1;
    (*(*u).ii_file).lly = (*u).current_line;
    (*(*u).ii_file).ury = (*u).current_line + iyEnd - iyStart;
    (*(*u).ii_file).pad_left = ixStart;
    (*(*u).ii_file).pad_right = nxdim - (indX1 + 1 as i32 - indX0) - ixStart;
    if (*(*u).ii_file).pad_right < 0 as i32 {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "\nERROR: iiuWriteSecPart - X dimension of data ({}) is not big enough for specified X indexes in writing to unit {} (xstart {} x0 {} x1 {})\n",
                nxdim, iunit, ixStart, indX0, indX1
            ),
        );
        if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
            exit(1 as i32);
        } else {
            return -(2 as i32);
        }
    }
    if (*(*u).ii_file).ury >= (*(*u).ii_file).ny {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "\nERROR: iiuWriteSecPart - Starting and ending lines ({} to {}) to write to unit {} go past end of end for section from line {}\n",
                iyStart,
                iyEnd,
                iunit,
                (*u).current_line
            ),
        );
        if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
            exit(1 as i32);
        } else {
            return -(3 as i32);
        }
    }
    arrStart =
        (array as *mut u8).offset((iyStart * nxdim * iiu_buf_bytes_per_pixel(iunit)) as isize);
    if (*u).no_convert {
        let row_bytes = nxdim * iiu_buf_bytes_per_pixel(iunit);
        let line_count = iyEnd - iyStart + 1;
        let Some(length) = row_bytes
            .checked_mul(line_count)
            .and_then(|length| usize::try_from(length).ok())
        else {
            return -(2 as i32);
        };
        err = iiWriteSection(
            &mut *(*u).ii_file,
            core::slice::from_raw_parts_mut(arrStart, length),
            (*u).current_sec,
        );
    } else {
        let line_count = iyEnd - iyStart + 1;
        let Some(length) = nxdim
            .checked_mul(line_count)
            .and_then(|length| usize::try_from(length).ok())
        else {
            return -(2 as i32);
        };
        err = iiWriteSectionFloat(
            &mut *(*u).ii_file,
            core::slice::from_raw_parts_mut(arrStart.cast(), length),
            (*u).current_sec,
        );
    }
    if err != 0 {
        if use_message {
            b3d_error(
                Some(&mut ImodFile::Stdout),
                format_args!(
                    "\nERROR: iiuWriteSecPart - writing X {} to {}, Y {} to {} to section in unit {}\n",
                    indX0, indX1, iyStart, iyEnd, iunit
                ),
            );
        }
        if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
            exit(1 as i32);
        } else {
            return err;
        }
    }
    (*u).current_sec += 1;
    (*u).current_line = 0 as i32;
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwritesecpart_(
    mut iunit: *mut i32,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: *mut i32,
    mut ixStart: *mut i32,
    mut indX0: *mut i32,
    mut indX1: *mut i32,
    mut iyStart: *mut i32,
    mut iyEnd: *mut i32,
) -> i32 {
    return iiu_write_sec_part(
        *iunit, array, *nxdim, *ixStart, *indX0, *indX1, *iyStart, *iyEnd,
    );
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_write_lines(
    mut iunit: i32,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: i32,
) -> i32 {
    let mut err: i32 = 0;
    let mut iz: i32 = 0;
    let mut u: *mut Unit = lookup_unit(
        iunit,
        "iiuWriteLine",
        UNIT_OPTIONS.with(|options| options.exit_on_error.get()),
        2,
    );
    if u.is_null() {
        return -(1 as i32);
    }
    if (*(*u).ii_file).file == IIFILE_TIFF {
        let header = match &mut (*u).header {
            UnitHeader::Header(header) => &raw mut *header,
            UnitHeader::None => unreachable!("an open TIFF unit always has a header"),
        };
        iiSyncFromMrcHeader(&mut *(*u).ii_file, &mut *header);
    }
    iz = (*u).current_sec;
    if setup_current_lines(u, numLines) != 0 {
        if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
            exit(1 as i32);
        } else {
            return -(3 as i32);
        }
    }
    if (*u).no_convert {
        let row_bytes = (*(*u).ii_file).nx * iiu_buf_bytes_per_pixel(iunit);
        let Some(length) = row_bytes
            .checked_mul(numLines)
            .and_then(|length| usize::try_from(length).ok())
        else {
            return -(3 as i32);
        };
        err = iiWriteSection(
            &mut *(*u).ii_file,
            core::slice::from_raw_parts_mut(array.cast(), length),
            iz,
        );
    } else {
        let Some(length) = (*(*u).ii_file)
            .nx
            .checked_mul(numLines)
            .and_then(|length| usize::try_from(length).ok())
        else {
            return -(3 as i32);
        };
        err = iiWriteSectionFloat(
            &mut *(*u).ii_file,
            core::slice::from_raw_parts_mut(array.cast(), length),
            iz,
        );
    }
    if err != 0 {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "\nERROR: iiuWriteLines - writing lines to unit {}.\n",
                iunit
            ),
        );
        if UNIT_OPTIONS.with(|options| options.exit_on_error.get()) != 0 {
            exit(1 as i32);
        } else {
            return 1 as i32;
        }
    }
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwritelines_(
    mut iunit: *mut i32,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: *mut i32,
) -> i32 {
    return iiu_write_lines(*iunit, array, *numLines);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iwrlin_(mut iunit: *mut i32, mut array: *mut ::core::ffi::c_void) {
    iiu_write_lines(*iunit, array, 1 as i32);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iwrsecl_(
    mut iunit: *mut i32,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: *mut i32,
) {
    iiu_write_lines(*iunit, array, *numLines);
}
unsafe fn setup_current_lines(mut u: *mut Unit, mut numLines: i32) -> i32 {
    (*(*u).ii_file).llx = 0 as i32;
    (*(*u).ii_file).urx = (*(*u).ii_file).nx - 1 as i32;
    (*(*u).ii_file).lly = (*u).current_line;
    (*(*u).ii_file).ury = (*u).current_line + numLines - 1 as i32;
    if (*(*u).ii_file).ury >= (*(*u).ii_file).ny {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "\nERROR: iiuRead/WriteLines - lines go past end of current section  (cur line {}  #l {}  to {}  ny {}).\n",
                (*u).current_line,
                numLines,
                (*(*u).ii_file).ury,
                (*(*u).ii_file).ny
            ),
        );
        return 1 as i32;
    }
    (*(*u).ii_file).pad_right = 0 as i32;
    (*(*u).ii_file).pad_left = (*(*u).ii_file).pad_right;
    (*u).current_line += numLines;
    if (*u).current_line == (*(*u).ii_file).ny {
        (*u).current_sec += 1;
        (*u).current_line = 0 as i32;
    }
    return 0 as i32;
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_file_info(
    mut iunit: i32,
    mut fileSize: *mut i32,
    mut fileType: *mut i32,
    mut flags: *mut i32,
) {
    let mut u: *mut Unit = lookup_unit(iunit, "iiuFileSize", 0 as i32, 0 as i32);
    *flags = 0 as i32;
    *fileType = IIFILE_MRC;
    *fileSize = -(1 as i32);
    if u.is_null() {
        return;
    }
    let name = (*(*u).ii_file).filename.clone().unwrap_or_default();
    if (*(*u).ii_file).file == IIFILE_SHR_MEM {
        *fileSize = (iiShrMemCheckSize(&name) as f64 / 1024.0) as i32;
    } else {
        *fileSize = std::fs::metadata(&name)
            .map(|metadata| (metadata.len() / 1024) as i32)
            .unwrap_or(-1);
    }
    *fileType = (*(*u).ii_file).file;
    let header = match &mut (*u).header {
        UnitHeader::Header(header) => header,
        UnitHeader::None => unreachable!("an open unit always has a header"),
    };
    *flags = header.iiu_flags
        | if header.swapped != 0 {
            IIUNIT_SWAPPED
        } else {
            0
        }
        | if header.bytes_signed != 0 {
            IIUNIT_BYTES_SIGNED
        } else {
            0
        };
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiufileinfo_(
    mut iunit: *mut i32,
    mut fileSize: *mut i32,
    mut fileType: *mut i32,
    mut flags: *mut i32,
) {
    iiu_file_info(*iunit, fileSize, fileType, flags);
}
#[unsafe(no_mangle)]
pub fn iiu_exit_on_error(doExit: i32, storeError: i32) {
    UNIT_OPTIONS.with(|options| {
        options.exit_on_error.set(doExit);
        options.store_error.set(storeError);
    });
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuexitonerror(mut doExit: *mut i32, mut storeError: *mut i32) {
    iiu_exit_on_error(*doExit, *storeError);
}
#[unsafe(no_mangle)]
pub fn iiu_get_exit_on_error() -> i32 {
    UNIT_OPTIONS.with(|options| options.exit_on_error.get())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ialbrief_(mut val: *mut i32) {
    iiu_alt_brief(*val);
}
/// Matches C `iiuAltBrief` (`unit_fileio.c`), used directly by translated
/// Fortran program units rather than through their underscore ABI wrapper.
pub fn iiu_alt_brief(val: i32) {
    UNIT_OPTIONS.with(|options| options.brief_header.set(val));
}
/// Matches C `iiuRetBrief` (`unit_fileio.c`).
pub fn iiu_ret_brief() -> i32 {
    let brief_header = UNIT_OPTIONS.with(|options| options.brief_header.get());
    if brief_header >= 0 {
        brief_header
    } else if std::env::var_os("IMOD_BRIEF_HEADER").is_some() {
        1
    } else {
        0
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn iiuretbrief_() -> i32 {
    iiu_ret_brief()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiualtprint_(mut val: *mut i32) {
    iiu_alt_print(*val);
}
/// Matches C `iiuAltPrint` (`unit_fileio.c`).
pub fn iiu_alt_print(val: i32) {
    UNIT_OPTIONS.with(|options| options.print_header.set(val));
}
/// Matches C `iiuRetPrint` (`unit_fileio.c`).
pub fn iiu_ret_print() -> i32 {
    UNIT_OPTIONS.with(|options| options.print_header.get())
}
#[unsafe(no_mangle)]
pub extern "C" fn iiuretprint_() -> i32 {
    iiu_ret_print()
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_alt_convert(mut iunit: i32, mut val: i32) {
    UNIT_TABLE.with(|unit_table| {
        let mut table = unit_table.borrow_mut();
        table
            .no_convert_units
            .retain(|&listed_unit| listed_unit as i32 != iunit);
        if val == 0 as i32 {
            table.no_convert_units.push(iunit);
        }
    });
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiualtconvert_(mut iunit: *mut i32, mut val: *mut i32) {
    iiu_alt_convert(*iunit, *val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiallowmultivolume_(mut allow: *mut i32) {
    iiAllowMultiVolume(*allow);
}
/// `function` is the caller's routine name for the error messages only; see
/// `lookup_unit`.  `#[no_mangle] extern "C"` was dropped with the `c_char`:
/// nothing outside this crate links the symbol.
pub unsafe fn iiu_mrc_header(
    mut iunit: i32,
    function: &str,
    mut doExit: i32,
    mut checkRW: i32,
) -> *mut MrcHeader {
    let mut u: *mut Unit = lookup_unit(iunit, function, doExit, checkRW);
    if u.is_null() {
        return ::core::ptr::null_mut::<MrcHeader>();
    }
    return match &mut (*u).header {
        UnitHeader::Header(header) => &raw mut *header,
        UnitHeader::None => ::core::ptr::null_mut(),
    };
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_sync_with_mrc_header(mut iunit: i32) {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_sync_with_mrc_header", 1 as i32, 0 as i32);
    let header = match &mut (*u).header {
        UnitHeader::Header(header) => &raw mut *header,
        UnitHeader::None => unreachable!("an open unit always has a header"),
    };
    let image = &mut *(*u).ii_file;
    iiSyncFromMrcHeader(image, &mut *header);
    if let Some(image_header) = image.mrc_header.as_deref_mut() {
        *image_header = (*header).clone();
    }
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_reassign_header_ptr(mut iunit: i32) {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_reassign_header_ptr", 1 as i32, 0 as i32);
    if (*(*u).ii_file).file != IIFILE_HDF {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "ERROR: iiuReassignHeaderPtr - File on unit {} is not HDF\n",
                iunit
            ),
        );
        exit(1 as i32);
    }
    let header = (*(*u).ii_file)
        .mrc_header
        .as_deref()
        .cloned()
        .unwrap_or_default();
    (*u).header = UnitHeader::Header(header);
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_file_type(mut iunit: i32) -> i32 {
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_file_type", 1 as i32, 0 as i32);
    return (*(*u).ii_file).file;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiufiletype_(mut iunit: *mut i32) -> i32 {
    return iiu_file_type(*iunit);
}
#[unsafe(no_mangle)]
/// Fortran wrapper `iisettifftagtoprint` (`unit_fileio.c:941`).
///
/// The source declares this `int` and then falls off the end without
/// returning, so its value is indeterminate in C; every caller ignores it
/// (`header.f90` calls it as a statement).  A c2rust-shaped `panic!` here
/// aborted `header -tag <n>` outright, which the reference does not do, so the
/// deterministic representation of that indeterminate value is returned
/// instead.
pub unsafe extern "C" fn iisettifftagtoprint_(mut tag: *mut i32) -> i32 {
    unsafe { tiffSetStringTagToPrint(*tag) };
    0
}
#[unsafe(no_mangle)]
pub unsafe fn iiu_buf_bytes_per_pixel(mut iunit: i32) -> i32 {
    let mut dsize: i32 = 0;
    let mut csize: i32 = 0;
    let mut u: *mut Unit = lookup_unit(iunit, "iiu_file_type", 1 as i32, 0 as i32);
    if (*u).no_convert {
        mrc_getdcsize((*(*u).ii_file).mode, &mut dsize, &mut csize);
        return dsize * csize;
    }
    return 4 as i32;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn move_(
    mut a: *mut ::core::ffi::c_char,
    mut b: *mut ::core::ffi::c_char,
    mut n: *mut i32,
) {
    let mut remaining = *n;
    while remaining != 0 {
        *a = *b;
        a = a.add(1);
        b = b.add(1);
        remaining -= 1;
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zero_(mut a: *mut ::core::ffi::c_char, mut n: *mut i32) {
    if let Ok(length) = usize::try_from(*n) {
        core::slice::from_raw_parts_mut(a.cast::<u8>(), length).fill(0);
    }
}
/// `unit_fileio.c:1465`.  `message` is a literal diagnostic, not a Fortran
/// string, so it is a `&str`; the write stays on the **C** stdout stream,
/// which is what `fprintf(stdout, …)` used.
pub fn iiu_memory_error(ptr: *const Unit, message: &str) {
    if !ptr.is_null() {
        return;
    }
    let _ = ImodFile::Stdout.write_all(format!("\n{message}\n").as_bytes());
    exit(1 as i32);
}
unsafe fn find_new_unit(iunit: i32) -> *mut Unit {
    let new_unit = Unit {
        ii_file: ::core::ptr::null_mut(),
        header: UnitHeader::None,
        current_sec: 0,
        current_line: 0,
        tail_name: 0,
        attribute: 0,
        being_used: false,
        read_only: false,
        no_convert: false,
    };
    if iunit <= 0 as i32 || iunit > MAX_UNIT {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "ERROR: iiuOpen - A unit number of {} is out of range\n",
                iunit
            ),
        );
        exit(1 as i32);
    }
    let already_open = UNIT_TABLE.with(|unit_table| {
        let mut table = unit_table.borrow_mut();
        if table.map.is_empty() {
            b3dSetStoreError(UNIT_OPTIONS.with(|options| options.store_error.get()));
        }
        if iunit as usize > table.map.len() {
            table.map.resize(iunit as usize + 9, None);
        }
        table.map[iunit as usize - 1].is_some()
    });
    if already_open {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "WARNING: iiuOpen - Unit number {} is already in use; closing it\n",
                iunit
            ),
        );
        iiu_close(iunit);
    }
    UNIT_TABLE.with(|unit_table| {
        let mut table = unit_table.borrow_mut();
        if let Some(index) = table.units.iter().position(|unit| !unit.being_used) {
            table.map[iunit as usize - 1] = Some(index);
            return table.units[index].as_mut() as *mut Unit;
        }
        table.units.push(Box::new(new_unit));
        let index = table.units.len() - 1;
        table.map[iunit as usize - 1] = Some(index);
        table.units[index].as_mut() as *mut Unit
    })
}
/// `function` is the *caller's routine name*, used only in the error messages
/// below (`unit_fileio.c:195` onward).  It carries no Fortran hidden string
/// length, so it is a plain `&str` rather than a `c_char` pointer.
unsafe fn lookup_unit(unit: i32, function: &str, mut doExit: i32, mut checkRW: i32) -> *mut Unit {
    let mut u: *mut Unit = ::core::ptr::null_mut::<Unit>();
    if unit <= 0 as i32 || unit > MAX_UNIT {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "\nERROR: {} - {} is not a legal unit number.\n",
                function, unit
            ),
        );
        return exit_or_null(doExit);
    }
    let found = UNIT_TABLE.with(|unit_table| {
        let mut table = unit_table.borrow_mut();
        if let Some(index) = table.map.get(unit as usize - 1).copied().flatten() {
            let no_convert = table.no_convert_units.contains(&unit);
            u = table.units[index].as_mut();
            (*u).no_convert = no_convert;
            true
        } else {
            false
        }
    });
    if found {
        if (*(*u).ii_file).format == IIFORMAT_COMPLEX {
            (*u).no_convert = true;
        }
        if (*u).being_used {
            if checkRW > 1 && (*u).read_only {
                b3d_error(
                    Some(&mut ImodFile::Stdout),
                    format_args!(
                        "\nERROR: {} - Trying to write to unit {}, which was opened read-only.\n",
                        function, unit
                    ),
                );
                return exit_or_null(doExit);
            }
            if checkRW == 1 as i32
                && ((*u).no_convert && (*(*u).ii_file).read_section.is_none()
                    || !(*u).no_convert && (*(*u).ii_file).read_section_float.is_none())
            {
                b3d_error(
                    Some(&mut ImodFile::Stdout),
                    format_args!(
                        "\nERROR: {} - There is no function for reading {} from the type of file on unit {}.\n",
                        function,
                        if (*u).no_convert {
                            "raw data"
                        } else {
                            "floats"
                        },
                        unit
                    ),
                );
                return exit_or_null(doExit);
            }
            if checkRW > 1 as i32
                && ((*u).no_convert && (*(*u).ii_file).write_section.is_none()
                    || !(*u).no_convert && (*(*u).ii_file).write_section_float.is_none())
            {
                b3d_error(
                    Some(&mut ImodFile::Stdout),
                    format_args!(
                        "\nERROR: {} - There is no function for writing {} to the type of file on unit {}.\n",
                        function,
                        if (*u).no_convert {
                            "raw data"
                        } else {
                            "floats"
                        },
                        unit
                    ),
                );
                return exit_or_null(doExit);
            }
            return u;
        }
    }
    b3d_error(
        Some(&mut ImodFile::Stdout),
        format_args!("\nERROR: {} - unit {} is not open.\n", function, unit),
    );
    exit_or_null(doExit)
}
unsafe fn exit_or_null(do_exit: i32) -> *mut Unit {
    if do_exit != 0 {
        exit(3 as i32);
    }
    core::ptr::null_mut()
}
#[cfg(test)]
mod tests {
    use super::{
        iiu_alt_brief, iiu_alt_print, iiu_exit_on_error, iiu_get_exit_on_error, iiu_ret_brief,
        iiu_ret_print, move_, zero_,
    };

    #[test]
    fn native_unit_output_options_are_typed_thread_local_state() {
        iiu_alt_brief(7);
        iiu_alt_print(0);
        iiu_exit_on_error(0, 1);

        assert_eq!(iiu_ret_brief(), 7);
        assert_eq!(iiu_ret_print(), 0);
        assert_eq!(iiu_get_exit_on_error(), 0);

        iiu_alt_brief(-1);
        iiu_alt_print(1);
        iiu_exit_on_error(1, -1);
    }

    #[test]
    fn move_and_zero_wrappers_preserve_requested_byte_count() {
        unsafe {
            let mut source = *b"abcdef";
            let mut destination = [0_i8; 6];
            let mut count = 4;
            move_(
                destination.as_mut_ptr(),
                source.as_mut_ptr().cast(),
                &mut count,
            );
            assert_eq!(
                &destination[..4],
                &[b'a' as i8, b'b' as i8, b'c' as i8, b'd' as i8]
            );
            zero_(destination.as_mut_ptr(), &mut count);
            assert_eq!(&destination[..4], &[0; 4]);
            assert_eq!(destination[4], 0);
        }
    }
}
