//! Translation of `IMOD/libiimod/iihdf.c`.
//!
//! This unit deliberately keeps the HDF5 C API at its original ABI boundary.
//! IMOD uses HDF5 identifiers, property lists, and datatype inspection directly;
//! substituting a Rust HDF abstraction would change both error and metadata
//! behaviour.  The functions below retain one Rust entry point per C function.
//! Linking the HDF5 ABI is the remaining platform integration step.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, ADOC_STRING, ADOC_ZVALUE_NAME, adoc_add_section, adoc_change_section_name,
    adoc_clear, adoc_delete_key_value, adoc_get_collection_name, adoc_get_float,
    adoc_get_float_array, adoc_get_integer, adoc_get_integer_array, adoc_get_key_by_index,
    adoc_get_num_collections, adoc_get_number_of_keys, adoc_get_number_of_sections,
    adoc_get_section_name, adoc_get_string, adoc_get_three_floats, adoc_get_three_integers,
    adoc_get_val_type_and_size, adoc_lookup_by_name_value, adoc_lookup_section, adoc_new,
    adoc_set_current, adoc_set_float, adoc_set_float_array, adoc_set_integer,
    adoc_set_integer_array, adoc_set_key_value, adoc_set_three_floats, adoc_set_three_integers,
    adoc_set_two_floats, adoc_set_two_integers,
};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, c_format_bytes};
use crate::imod::libiimod::hdf_imageio::{
    hdf_read_section_any, hdf_write_section_any, init_new_hdf_file,
};
use crate::imod::libiimod::iimage::{
    IIERR_IO_ERROR, IIERR_NOT_FORMAT, IIFILE_HDF, IIFORMAT_COMPLEX, IIFORMAT_LUMINANCE,
    IIFORMAT_RGB, IISTATE_NOTINIT, IISTATE_UNUSED, IITYPE_BYTE, IITYPE_FLOAT, IITYPE_SHORT,
    IITYPE_UBYTE, IITYPE_USHORT, ImodImageFile, MRSA_BYTE, MRSA_FLOAT, MRSA_USHORT, StackSetData,
    ii_close, ii_default_min_max_mean, ii_new_box, ii_sync_from_mrc_header,
};
use crate::imod::libiimod::iimrc::ii_mrc_fill_header;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT,
    MRC_MODE_USHORT, MrcHeader, fix_title_padding, mrc_head_new, mrc_set_scale,
};
use core::ffi::{c_char, c_void};
use std::sync::Mutex;

// C `sStringBuf`/`sStrBufSize` and `sAttribName`/`sAttrNameSize` (`iihdf.c:110`)
// were `malloc`ed scratch buffers grown by `manageMallocBuf`.  A `Vec<u8>`
// carries its own capacity, so the four are gone and each user owns the buffer
// it builds; `manageMallocBuf` still serves the four numeric buffers below.
static FLOAT_BUF: Mutex<Vec<f32>> = Mutex::new(Vec::new());
static INT_BUF: Mutex<Vec<i32>> = Mutex::new(Vec::new());
static SHORT_BUF: Mutex<Vec<i16>> = Mutex::new(Vec::new());
static USHORT_BUF: Mutex<Vec<u16>> = Mutex::new(Vec::new());
/// C `static char *sMrcPrefix` (`iihdf.c:117`): the `MRC.` key prefix found in
/// the file, or `NULL` when none has been found.  It is metadata text, not a
/// C buffer; C-compatible bytes are made only where an API needs them.
static MRC_PREFIX: Mutex<Option<String>> = Mutex::new(None);

/// HDF5 `hid_t`; IMOD obtains this from `hdf5.h`.
pub type HidT = i64;
/// HDF5 `hsize_t`; IMOD obtains this from `hdf5.h`.
pub type HsizeT = usize;
type HerrT = i32;

#[repr(C)]
#[derive(Copy, Clone)]
struct H5IhInfo {
    index_size: HsizeT,
    heap_size: HsizeT,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct H5OHdrSpace {
    total: HsizeT,
    meta: HsizeT,
    mesg: HsizeT,
    free: HsizeT,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct H5OHdrMesg {
    present: u64,
    shared: u64,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct H5OHdrInfo {
    version: u32,
    nmesgs: u32,
    nchunks: u32,
    flags: u32,
    space: H5OHdrSpace,
    mesg: H5OHdrMesg,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct H5OMetaSize {
    obj: H5IhInfo,
    attr: H5IhInfo,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct H5OInfo {
    fileno: libc::c_ulong,
    addr: libc::c_ulong,
    type_: i32,
    rc: u32,
    atime: libc::c_long,
    mtime: libc::c_long,
    ctime: libc::c_long,
    btime: libc::c_long,
    num_attrs: HsizeT,
    hdr: H5OHdrInfo,
    meta_size: H5OMetaSize,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct H5GInfo {
    storage_type: i32,
    nlinks: HsizeT,
    max_corder: i64,
    mounted: bool,
}

const H5_INDEX_NAME: i32 = 0;
const H5_ITER_INC: i32 = 0;
const H5O_TYPE_GROUP: i32 = 0;
const H5O_TYPE_DATASET: i32 = 1;
const H5D_CHUNKED: i32 = 2;
const H5T_INTEGER: i32 = 0;
const H5T_FLOAT: i32 = 1;
const H5T_STRING: i32 = 3;
const H5T_SGN_2: i32 = 1;
const H5S_SIMPLE: i32 = 1;

/// A discovered HDF group.  This is crate-owned scan state, not an HDF ABI
/// object.  It is kept in [`HdfScanState`] and its path owns its bytes.
struct GroupData {
    group_id: HidT,
    name: Option<Vec<u8>>,
    has_valid_datasets: u8,
    has_any_datasets: u8,
    has_groups: u8,
    non_global_attrib: u8,
    added_global: i16,
    adoc_collection: i16,
    num_attributes: i32,
}
/// A discovered HDF dataset.  This is crate-owned scan state, not an HDF ABI
/// object.  See [`GroupData`] for the owned path representation.
struct DatasetData {
    dset_id: HidT,
    name: Option<Vec<u8>>,
    data_type: i16,
    swapped: i16,
    nx: i32,
    ny: i32,
    nz: i32,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    num_attributes: i32,
    group_num: i32,
}

pub const IIHDF_OTHER_MRC: i32 = 2;
pub const IIHDF_EMAN: i32 = 3;
pub const IIHDF_CHIMERA: i32 = 4;
pub const IIHDF_UNKNOWN: i32 = 5;
pub const IIHDF_IMOD: i32 = 1;
pub const IIERR_NO_SUPPORT: i32 = 4;
const IIERR_MEMORY_ERR: i32 = 3;

/// Per-open-file discovery data.  In C this was spread across process-global
/// `Ilist` pointers and scalar globals; keeping it local makes ownership and
/// teardown explicit, and also prevents one open from clobbering another.
#[derive(Default)]
struct HdfScanState {
    groups: Vec<GroupData>,
    datasets: Vec<DatasetData>,
    numbered_groups: bool,
    max_group_num: i32,
    min_group_num: i32,
}

impl HdfScanState {
    fn new() -> Self {
        Self {
            groups: Vec::new(),
            datasets: Vec::new(),
            numbered_groups: true,
            max_group_num: -1_000_000_000,
            min_group_num: 1_000_000_000,
        }
    }
}

/* HDF5 calls stay here, rather than being replaced by a Rust HDF crate. */
#[link(name = "hdf5_serial")]
unsafe extern "C" {
    fn H5Fis_hdf5(filename: *const c_char) -> i32;
    fn H5Gopen2(file: HidT, name: *const c_char, access: HidT) -> HidT;
    fn H5Gcreate2(
        location: HidT,
        name: *const c_char,
        link_create: HidT,
        group_create: HidT,
        group_access: HidT,
    ) -> HidT;
    fn H5Gclose(group: HidT) -> HerrT;
    fn H5Fclose(file: HidT) -> HerrT;
    fn H5Fopen(filename: *const c_char, flags: u32, access: HidT) -> HidT;
    fn H5Fcreate(filename: *const c_char, flags: u32, create: HidT, access: HidT) -> HidT;
    fn H5Gget_info(group: HidT, info: *mut H5GInfo) -> HerrT;
    fn H5Oget_info(object: HidT, info: *mut H5OInfo) -> HerrT;
    fn H5Oget_info_by_idx(
        group: HidT,
        name: *const c_char,
        index: i32,
        order: i32,
        n: HsizeT,
        info: *mut H5OInfo,
        access: HidT,
    ) -> HerrT;
    fn H5Lget_name_by_idx(
        group: HidT,
        name: *const c_char,
        index: i32,
        order: i32,
        n: HsizeT,
        out: *mut c_char,
        size: HsizeT,
        access: HidT,
    ) -> isize;
    fn H5Lmove(
        source: HidT,
        source_name: *const c_char,
        destination: HidT,
        destination_name: *const c_char,
        link_create: HidT,
        link_access: HidT,
    ) -> HerrT;
    fn H5Dopen2(file: HidT, name: *const c_char, access: HidT) -> HidT;
    fn H5Dclose(dataset: HidT) -> HerrT;
    fn H5Dget_type(dataset: HidT) -> HidT;
    fn H5Dget_space(dataset: HidT) -> HidT;
    fn H5Dget_create_plist(dataset: HidT) -> HidT;
    fn H5Tget_class(typ: HidT) -> i32;
    fn H5Tget_size(typ: HidT) -> HsizeT;
    fn H5Tget_precision(typ: HidT) -> HsizeT;
    fn H5Tget_sign(typ: HidT) -> i32;
    fn H5Tget_order(typ: HidT) -> i32;
    fn H5Tclose(typ: HidT) -> HerrT;
    fn H5Tget_native_type(typ: HidT, direction: i32) -> HidT;
    fn H5Aopen_by_idx(
        object: HidT,
        object_name: *const c_char,
        index: i32,
        order: i32,
        n: HsizeT,
        attribute_access: HidT,
        link_access: HidT,
    ) -> HidT;
    fn H5Aread(attribute: HidT, typ: HidT, buffer: *mut c_void) -> HerrT;
    fn H5Aclose(attribute: HidT) -> HerrT;
    fn H5Aget_space(attribute: HidT) -> HidT;
    fn H5Aget_type(attribute: HidT) -> HidT;
    fn H5Aget_name(attribute: HidT, size: HsizeT, name: *mut c_char) -> isize;
    fn H5Sget_simple_extent_ndims(space: HidT) -> i32;
    fn H5Sget_simple_extent_npoints(space: HidT) -> isize;
    fn H5Sget_simple_extent_type(space: HidT) -> i32;
    fn H5Sget_simple_extent_dims(space: HidT, current: *mut HsizeT, maximum: *mut HsizeT) -> i32;
    fn H5Sclose(space: HidT) -> HerrT;
    fn H5Pget_layout(plist: HidT) -> i32;
    fn H5Pget_chunk(plist: HidT, max_dims: i32, dims: *mut HsizeT) -> i32;
    fn H5Pclose(plist: HidT) -> HerrT;
    fn H5Aexists_by_name(
        object: HidT,
        object_name: *const c_char,
        attribute_name: *const c_char,
        access: HidT,
    ) -> i32;
    fn H5Adelete_by_name(
        object: HidT,
        object_name: *const c_char,
        attribute_name: *const c_char,
        access: HidT,
    ) -> HerrT;
    fn H5Aget_num_attrs(object: HidT) -> i32;
    fn H5Adelete_by_idx(
        object: HidT,
        object_name: *const c_char,
        index: i32,
        order: i32,
        n: HsizeT,
        access: HidT,
    ) -> HerrT;
    fn H5Acreate2(
        object: HidT,
        name: *const c_char,
        typ: HidT,
        space: HidT,
        create: HidT,
        access: HidT,
    ) -> HidT;
    fn H5Awrite(attribute: HidT, typ: HidT, buffer: *const c_void) -> HerrT;
    fn H5Screate_simple(rank: i32, current: *const HsizeT, maximum: *const HsizeT) -> HidT;
    fn H5Screate(class: i32) -> HidT;
    fn H5Tcopy(typ: HidT) -> HidT;
    fn H5Tset_size(typ: HidT, size: HsizeT) -> HerrT;
    fn H5Tset_strpad(typ: HidT, padding: i32) -> HerrT;
    static H5T_NATIVE_INT_g: HidT;
    static H5T_NATIVE_SHORT_g: HidT;
    static H5T_NATIVE_USHORT_g: HidT;
    static H5T_NATIVE_FLOAT_g: HidT;
    static H5T_C_S1_g: HidT;
}

/// C `iiTestIfHDF` (`iihdf.c:125`).
pub fn ii_test_if_hdf(filename: &[u8]) -> i32 {
    // H5Fis_hdf5 is intentionally an HDF5 ABI call, and it takes the file name
    // as `char *`.
    let name = std::ffi::CString::new(filename).unwrap();
    unsafe { H5Fis_hdf5(name.as_ptr()) }
}
/// C `iiHDFCheck` (`iihdf.c:133`).
pub unsafe fn ii_hdf_check(in_file: *mut ImodImageFile) -> i32 {
    // The file name crosses into HDF5 as `char *`.
    let name = std::ffi::CString::new((*in_file).filename.as_deref().unwrap_or_default()).unwrap();
    let err = H5Fis_hdf5(name.as_ptr());
    if err < 0 {
        return IIERR_IO_ERROR;
    }
    if err == 0 {
        return IIERR_NOT_FORMAT;
    }
    (*in_file).fp = None;
    let mut state = HdfScanState::new();
    // C `slash = strdup("/")`, whose allocation-failure arm a `Vec` cannot
    // reach; it is the root group's name and `scanGroup` takes ownership.
    let slash = b"/".to_vec();
    let writable = (*in_file).fmode.contains('+');
    let file_id = H5Fopen(name.as_ptr(), if writable { 1 } else { 0 }, 0);
    let root = std::ffi::CString::new(&slash[..]).unwrap();
    let group_id = H5Gopen2(file_id, root.as_ptr(), 0);
    let mut section = 0;
    let scan_err = scan_group(&mut state, group_id, slash, &mut section);
    if scan_err != 0 || state.datasets.is_empty() {
        cleanup_from_open(&mut state, file_id, 1, 0, in_file);
        return if scan_err != 0 {
            scan_err
        } else {
            IIERR_NOT_FORMAT
        };
    }
    let mut single_image_stack = true;
    let first = state.datasets.as_mut_ptr();
    let nx_stack = (*first).nx;
    let ny_stack = (*first).ny;
    let stack_type = (*first).data_type;
    (*in_file).num_volumes = 1;
    for set in 0..state.datasets.len() as i32 {
        let dataset = state.datasets.as_mut_ptr().add(set as usize);
        if (set != 0
            && (nx_stack != (*dataset).nx
                || ny_stack != (*dataset).ny
                || stack_type != (*dataset).data_type))
            || (*dataset).nz > 1
        {
            single_image_stack = false;
            (*in_file).num_volumes = state.datasets.len() as i32;
            break;
        }
    }
    if single_image_stack
        && state.numbered_groups
        && (state.min_group_num < 0
            || state.max_group_num + 1 - state.min_group_num < state.datasets.len() as i32)
    {
        state.numbered_groups = false;
    }
    (*in_file).ii_volumes = vec![in_file];
    (*in_file).owned_hdf_volumes.clear();
    for ind in 1..(*in_file).num_volumes {
        let mut volume = ii_new_box();
        let volume_ptr = volume.as_mut() as *mut ImodImageFile;
        (*in_file).owned_hdf_volumes.push(volume);
        (*in_file).ii_volumes.push(volume_ptr);
    }
    let mut hdf_source = IIHDF_UNKNOWN;
    let mut eman_sect_type: Option<&[u8]> = None;
    for ind in 0..state.groups.len() {
        let group = state.groups.as_mut_ptr().add(ind);
        if (*group).name.as_deref() == Some(&b"/Chimera"[..]) {
            hdf_source = IIHDF_CHIMERA;
        }
        if (*group).num_attributes == 0 || (*group).adoc_collection > 0 {
            continue;
        }
        for set in 0..state.datasets.len() {
            let dataset = state.datasets.as_mut_ptr().add(set);
            if starts_with(
                (*dataset).name.as_deref().unwrap_or_default(),
                (*group).name.as_deref().unwrap_or_default(),
            ) == 0
            {
                (*group).non_global_attrib = 1;
                break;
            }
        }
    }
    if single_image_stack {
        (*in_file).nx = nx_stack;
        (*in_file).ny = ny_stack;
        (*in_file).nz = state.datasets.len() as i32;
        (*in_file).type_ = stack_type as i32;
        (*in_file).stack_set_list = Some(Vec::with_capacity((*in_file).nz as usize));
        for set in 0..(*in_file).nz {
            let dataset = state.datasets.as_mut_ptr().add(set as usize);
            (*in_file)
                .stack_set_list
                .as_mut()
                .expect("stack storage was initialized")
                .push(StackSetData {
                    name: (*dataset).name.take(),
                    dset_id: (*dataset).dset_id,
                    is_open: true,
                });
        }
    } else {
        for set in 0..state.datasets.len() {
            let dataset = state.datasets.as_mut_ptr().add(set);
            let volume = (&(*in_file).ii_volumes)[set as usize];
            (*volume).nx = (*dataset).nx;
            (*volume).ny = (*dataset).ny;
            (*volume).nz = (*dataset).nz;
            (*volume).type_ = (*dataset).data_type as i32;
            (*volume).dataset_name = (*dataset)
                .name
                .take()
                .map(|name| String::from_utf8_lossy(&name).into_owned());
            (*volume).dataset_id = (*dataset).dset_id;
            (*volume).dataset_is_open = 1;
            (*volume).num_volumes = (*in_file).num_volumes;
        }
    }
    let mut retval = 0;
    for ind in 0..(*in_file).num_volumes {
        let volume = (&(*in_file).ii_volumes)[ind as usize];
        if !single_image_stack && ind == 0 {
            (*in_file).global_adoc_index = adoc_new();
            if (*in_file).global_adoc_index < 0 {
                retval = IIERR_MEMORY_ERR;
            }
        }
        if retval == 0 {
            (*volume).adoc_index = adoc_new();
            if (*volume).adoc_index < 0 {
                retval = IIERR_MEMORY_ERR;
            }
        }
        if retval == 0 {
            (*volume).mrc_header = Some(Box::default());
            (*volume).header = ((*volume)
                .mrc_header
                .as_deref_mut()
                .expect("new HDF header is present")
                as *mut MrcHeader)
                .cast();
        }
        if retval != 0 || (*volume).header.is_null() {
            cleanup_from_open(&mut state, file_id, 1, (*in_file).num_volumes, in_file);
            return IIERR_MEMORY_ERR;
        }
    }
    if single_image_stack
        && state.numbered_groups
        && state.min_group_num >= 0
        && state.max_group_num + 1 - state.min_group_num >= (*in_file).nz
    {
        if H5Aexists_by_name(file_id, c"/MDF/images".as_ptr(), c"imageid_max".as_ptr(), 0) <= 0 {
            state.numbered_groups = false;
        }
        if state.numbered_groups {
            if setup_zto_set_map(file_id, in_file, state.max_group_num + 1, 0) != 0 {
                return IIERR_MEMORY_ERR;
            }
            for set in 0..(*in_file).nz {
                if !state.numbered_groups {
                    break;
                }
                let dataset = state.datasets.as_mut_ptr().add(set as usize);
                if (&(*in_file).z_to_data_set_map)[(*dataset).group_num as usize] >= 0 {
                    state.numbered_groups = false;
                } else {
                    (&mut (*in_file).z_to_data_set_map)[(*dataset).group_num as usize] = set;
                }
            }
        }
    }
    if single_image_stack && !state.numbered_groups {
        (*in_file).z_to_data_set_map.clear();
        if setup_zto_set_map(file_id, in_file, (*in_file).nz, 1) != 0 {
            return IIERR_MEMORY_ERR;
        }
    }
    for set in 0..state.datasets.len() as i32 {
        let vol_ind = if single_image_stack { 0 } else { set };
        let volume = (&(*in_file).ii_volumes)[vol_ind as usize];
        let adoc_index = (*volume).adoc_index;
        adoc_set_current(adoc_index);
        let dataset = state.datasets.as_mut_ptr().add(set as usize);
        // C `sprintf(sectText, "%d", ...)` into `char[32]`.
        let sect_text = c_format(
            "%d",
            &[CArg::Int(if state.numbered_groups {
                (*dataset).group_num as i64
            } else {
                set as i64
            })],
        )
        .into_bytes();
        let mut added_sect_ind = -1;
        let mut sect_ind = 0;
        let mut collection: &[u8] = ADOC_GLOBAL_NAME;
        let dataset_name = if single_image_stack {
            let stack_set = (*in_file)
                .stack_set_list
                .as_deref()
                .and_then(|stacks| stacks.get(set as usize))
                .expect("stack dataset index is valid");
            String::from_utf8_lossy(stack_set.name.as_deref().unwrap_or_default()).into_owned()
        } else {
            (*((&(*in_file).ii_volumes)[set as usize]))
                .dataset_name
                .clone()
                .unwrap_or_default()
        };
        if (*dataset).num_attributes != 0 {
            // The dataset path crosses into HDF5.
            let path = std::ffi::CString::new(dataset_name.clone()).unwrap();
            let dataset_id = H5Dopen2(file_id, path.as_ptr(), 0);
            if single_image_stack {
                sect_ind = adoc_add_section(ADOC_ZVALUE_NAME, &sect_text);
                if sect_ind < 0 {
                    retval = IIERR_MEMORY_ERR;
                }
                added_sect_ind = sect_ind;
                collection = ADOC_ZVALUE_NAME;
            }
            if retval == 0 {
                retval =
                    attributes_to_adoc(dataset_id, (*dataset).num_attributes, collection, sect_ind);
            }
            H5Dclose(dataset_id);
        }
        for ind in 0..state.groups.len() {
            if retval != 0 {
                break;
            }
            let group = state.groups.as_mut_ptr().add(ind);
            if (*group).num_attributes == 0
                || (*group).adoc_collection > 0
                || (*group).added_global != 0
            {
                continue;
            }
            let group_path =
                std::ffi::CString::new((*group).name.clone().unwrap_or_default()).unwrap();
            let group_id = H5Gopen2(file_id, group_path.as_ptr(), 0);
            if starts_with(
                dataset_name.as_bytes(),
                (*group).name.as_deref().unwrap_or_default(),
            ) != 0
            {
                let mut group_adoc_index =
                    (*((&(*in_file).ii_volumes)[vol_ind as usize])).adoc_index;
                let mut group_sect_ind = 0;
                let mut group_collection: &[u8] = ADOC_GLOBAL_NAME;
                if single_image_stack && ((*group).non_global_attrib != 0 || (*in_file).nz == 1) {
                    if (*group).non_global_attrib == 0 {
                        adoc_set_current(group_adoc_index);
                        retval = attributes_to_adoc(
                            group_id,
                            (*group).num_attributes,
                            group_collection,
                            group_sect_ind,
                        );
                    }
                    if added_sect_ind < 0 {
                        adoc_set_current(group_adoc_index);
                        added_sect_ind = adoc_add_section(ADOC_ZVALUE_NAME, &sect_text);
                        if added_sect_ind < 0 {
                            retval = IIERR_MEMORY_ERR;
                        }
                    }
                    group_sect_ind = added_sect_ind;
                    group_collection = ADOC_ZVALUE_NAME;
                } else if (*group).non_global_attrib == 0
                    && (single_image_stack || (*in_file).num_volumes > 1)
                {
                    (*group).added_global = 1;
                    if !single_image_stack {
                        group_adoc_index = (*in_file).global_adoc_index;
                    }
                }
                adoc_set_current(group_adoc_index);
                if retval == 0 {
                    retval = attributes_to_adoc(
                        group_id,
                        (*group).num_attributes,
                        group_collection,
                        group_sect_ind,
                    );
                }
            }
            H5Gclose(group_id);
        }
        if retval != 0 {
            cleanup_from_open(&mut state, file_id, 1, (*in_file).num_volumes, in_file);
            return retval;
        }
    }
    retval = 0;
    for grp in 0..state.groups.len() {
        let group = state.groups.as_mut_ptr().add(grp);
        if (*group).adoc_collection <= 0 {
            continue;
        }
        // C `strrchr(group->name, '/') + 1`: the last path element.
        let group_name = (*group).name.clone().unwrap_or_default();
        let collection = group_name[group_name
            .iter()
            .rposition(|byte| *byte == b'/')
            .unwrap_or(0)
            + 1..]
            .to_vec();
        for sub in 0..state.groups.len() {
            if retval != 0 {
                break;
            }
            let sub_group = state.groups.as_mut_ptr().add(sub);
            if sub == grp
                || (*sub_group).num_attributes == 0
                || starts_with(
                    (*sub_group).name.as_deref().unwrap_or_default(),
                    (*group).name.as_deref().unwrap_or_default(),
                ) == 0
            {
                continue;
            }
            let sub_name = (*sub_group).name.clone().unwrap_or_default();
            let section_name = sub_name
                [sub_name.iter().rposition(|byte| *byte == b'/').unwrap_or(0) + 1..]
                .to_vec();
            let sub_path = std::ffi::CString::new(&sub_name[..]).unwrap();
            let group_id = H5Gopen2(file_id, sub_path.as_ptr(), 0);
            adoc_set_current(if single_image_stack {
                (*in_file).adoc_index
            } else {
                (*in_file).global_adoc_index
            });
            let group_sect_ind = adoc_add_section(&collection, &section_name);
            if group_sect_ind < 0 {
                retval = IIERR_MEMORY_ERR;
            }
            if retval == 0 {
                retval = attributes_to_adoc(
                    group_id,
                    (*sub_group).num_attributes,
                    &collection,
                    group_sect_ind,
                );
            }
            H5Gclose(group_id);
        }
        if retval != 0 {
            cleanup_from_open(&mut state, file_id, 1, (*in_file).num_volumes, in_file);
            return retval;
        }
    }
    if hdf_source != IIHDF_CHIMERA {
        adoc_set_current((*in_file).adoc_index);
        let num_keys = adoc_get_number_of_keys(ADOC_GLOBAL_NAME, 0);
        let mrc_tags: [&[u8]; 6] = [
            b"MRC.mx",
            b"MRC.my",
            b"MRC.mz",
            b"MRC.xlen",
            b"MRC.ylen",
            b"MRC.zlen",
        ];
        let mut num_match = 0;
        retval = 0;
        for key_ind in 0..num_keys {
            if retval != 0 {
                break;
            }
            let mut key = None;
            retval = adoc_get_key_by_index(ADOC_GLOBAL_NAME, 0, key_ind, &mut key);
            let key = key.unwrap_or_default();
            if retval == 0 && starts_with(&key, b"EMAN.") == 0 {
                for tag in mrc_tags {
                    let sub = ends_with(&key, tag);
                    if sub >= 0 {
                        // C copies the first `sub` bytes of the key into
                        // `sStringBuf` and terminates it: that is the prefix.
                        let prefix = String::from_utf8_lossy(&key[..sub as usize]).into_owned();
                        if num_match == 0 {
                            *MRC_PREFIX.lock().unwrap() = Some(prefix);
                            num_match = 1;
                        } else if MRC_PREFIX.lock().unwrap().as_deref() == Some(prefix.as_str()) {
                            num_match += 1;
                        }
                        break;
                    }
                }
            }
        }
        if retval != 0 {
            cleanup_from_open(&mut state, file_id, 1, (*in_file).num_volumes, in_file);
            return IIERR_MEMORY_ERR;
        }
        if num_match == mrc_tags.len() as i32 {
            hdf_source = IIHDF_OTHER_MRC;
            if MRC_PREFIX.lock().unwrap().as_deref() == Some("IMOD.") {
                hdf_source = IIHDF_IMOD;
            }
        } else {
            let mut value = 0;
            if adoc_get_integer(ADOC_GLOBAL_NAME, 0, b"EMAN.nx", &mut value) == 0 {
                eman_sect_type = Some(ADOC_GLOBAL_NAME);
            } else if adoc_get_integer(ADOC_ZVALUE_NAME, 0, b"EMAN.nx", &mut value) == 0 {
                eman_sect_type = Some(ADOC_ZVALUE_NAME);
            }
            if eman_sect_type.is_some() {
                hdf_source = IIHDF_EMAN;
            }
        }
    }
    for vol_ind in 0..(*in_file).num_volumes {
        let volume = (&(*in_file).ii_volumes)[vol_ind as usize];
        (*volume).ii_volumes = (*in_file).ii_volumes.clone();
        (*volume).format = IIFORMAT_LUMINANCE;
        let mode = if (*volume).type_ == IITYPE_BYTE || (*volume).type_ == IITYPE_UBYTE {
            if (*volume).type_ == IITYPE_UBYTE {
                let mut rgb = 0;
                if hdf_source == IIHDF_IMOD
                    && get_prefixed_integer(b"is_rgb", &mut rgb, &mut retval) == 0
                    && rgb > 0
                {
                    (*volume).format = IIFORMAT_RGB;
                    (*volume).nx /= 3;
                    MRC_MODE_RGB
                } else {
                    MRC_MODE_BYTE
                }
            } else {
                MRC_MODE_BYTE
            }
        } else if (*volume).type_ == IITYPE_SHORT {
            MRC_MODE_SHORT
        } else if (*volume).type_ == IITYPE_USHORT {
            MRC_MODE_USHORT
        } else {
            let mut complex = 0;
            if hdf_source == IIHDF_IMOD
                && get_prefixed_integer(b"is_complex", &mut complex, &mut retval) == 0
                && complex > 0
            {
                (*volume).format = IIFORMAT_COMPLEX;
                (*volume).nx /= 2;
                MRC_MODE_COMPLEX_FLOAT
            } else {
                MRC_MODE_FLOAT
            }
        };
        (*volume).mode = mode;
        let hdata = (*volume).header.cast::<MrcHeader>();
        mrc_head_new(&mut *hdata, (*volume).nx, (*volume).ny, (*volume).nz, mode);
        (*hdata).bytes_signed = if (*volume).type_ == IITYPE_BYTE { 1 } else { 0 };
        let dataset = state.datasets.as_mut_ptr().add(vol_ind as usize);
        (*hdata).swapped = (*dataset).swapped as i32;
        (*hdata).packed4bits = 0;
        (*hdata).half_floats = 0;
        ii_default_min_max_mean(
            (*volume).type_,
            &mut (*hdata).amin,
            &mut (*hdata).amax,
            &mut (*hdata).amean,
        );
        if hdf_source == IIHDF_IMOD || hdf_source == IIHDF_OTHER_MRC {
            adoc_set_current((*volume).adoc_index);
            get_del_prefixed_integer(b"MRC.nxstart", &mut (*hdata).nxstart, &mut retval);
            get_del_prefixed_integer(b"MRC.nystart", &mut (*hdata).nystart, &mut retval);
            get_del_prefixed_integer(b"MRC.nzstart", &mut (*hdata).nzstart, &mut retval);
            get_del_prefixed_integer(b"MRC.mx", &mut (*hdata).mx, &mut retval);
            get_del_prefixed_integer(b"MRC.my", &mut (*hdata).my, &mut retval);
            get_del_prefixed_integer(b"MRC.mz", &mut (*hdata).mz, &mut retval);
            get_del_prefixed_float(b"MRC.xlen", &mut (*hdata).xlen, &mut retval);
            get_del_prefixed_float(b"MRC.ylen", &mut (*hdata).ylen, &mut retval);
            get_del_prefixed_float(b"MRC.zlen", &mut (*hdata).zlen, &mut retval);
            get_del_prefixed_float(b"MRC.alpha", &mut (*hdata).alpha, &mut retval);
            get_del_prefixed_float(b"MRC.beta", &mut (*hdata).beta, &mut retval);
            get_del_prefixed_float(b"MRC.gamma", &mut (*hdata).gamma, &mut retval);
            get_del_prefixed_integer(b"MRC.mapc", &mut (*hdata).mapc, &mut retval);
            get_del_prefixed_integer(b"MRC.mapr", &mut (*hdata).mapr, &mut retval);
            get_del_prefixed_integer(b"MRC.maps", &mut (*hdata).maps, &mut retval);
            get_del_prefixed_float(b"MRC.minimum", &mut (*hdata).amin, &mut retval);
            get_del_prefixed_float(b"MRC.maximum", &mut (*hdata).amax, &mut retval);
            get_del_prefixed_float(b"MRC.mean", &mut (*hdata).amean, &mut retval);
            get_del_prefixed_integer(b"MRC.ispg", &mut (*hdata).ispg, &mut retval);
            get_del_prefixed_float(b"MRC.xorigin", &mut (*hdata).xorg, &mut retval);
            get_del_prefixed_float(b"MRC.yorigin", &mut (*hdata).yorg, &mut retval);
            get_del_prefixed_float(b"MRC.zorigin", &mut (*hdata).zorg, &mut retval);
            get_del_prefixed_float(b"MRC.rms", &mut (*hdata).rms, &mut retval);
            get_del_prefixed_integer(b"MRC.nlabels", &mut (*hdata).nlabl, &mut retval);
            (*hdata).nlabl = (*hdata).nlabl.clamp(0, 10);
            let mut nsum = 6;
            if adoc_get_float_array(
                ADOC_GLOBAL_NAME,
                0,
                &prefixed_key(
                    MRC_PREFIX.lock().unwrap().as_deref().map(str::as_bytes),
                    b"MRC.tiltangles",
                ),
                &mut (*hdata).tiltangles,
                &mut nsum,
                6,
            ) < 0
            {
                retval += 1;
            }
            delete_prefixed_key_value(b"MRC.tiltangles");
            for sub in 0..(*hdata).nlabl {
                if retval != 0 {
                    break;
                }
                // C `sprintf(labelKey, "MRC.label%d", sub)` into `char[14]`.
                let label_key = c_format("MRC.label%d", &[CArg::Int(sub as i64)]).into_bytes();
                let mut label = Vec::new();
                if adoc_get_string(
                    ADOC_GLOBAL_NAME,
                    0,
                    &prefixed_key(
                        MRC_PREFIX.lock().unwrap().as_deref().map(str::as_bytes),
                        &label_key,
                    ),
                    &mut label,
                ) != 0
                {
                    retval += 1;
                } else {
                    // C `strncpy(hdata->labels[sub], label, 80)`, which pads
                    // the destination with NULs out to 80.
                    let copied = label.len().min(80);
                    let slot = &mut (*hdata).labels[sub as usize];
                    slot[..copied].copy_from_slice(&label[..copied]);
                    slot[copied..80].fill(0);
                    fix_title_padding(&mut (*hdata).labels[sub as usize]);
                    delete_prefixed_key_value(&label_key);
                }
            }
            delete_prefixed_key_value(b"MRC.nx");
            delete_prefixed_key_value(b"MRC.ny");
            delete_prefixed_key_value(b"MRC.nz");
            delete_prefixed_key_value(b"MRC.mode");
            if single_image_stack {
                (*in_file).has_piece_coords = 1;
                for sub in 0..(*in_file).nz {
                    let mut x = 0;
                    let mut y = 0;
                    let mut z = 0;
                    if adoc_get_three_integers(
                        ADOC_ZVALUE_NAME,
                        sub,
                        b"PieceCoordinates",
                        &mut x,
                        &mut y,
                        &mut z,
                    ) != 0
                    {
                        (*in_file).has_piece_coords = 0;
                        break;
                    }
                }
            }
        } else if hdf_source == IIHDF_CHIMERA {
            let mut xscale = 0.0;
            let mut yscale = 0.0;
            let mut zscale = 0.0;
            if adoc_get_three_floats(
                ADOC_GLOBAL_NAME,
                0,
                b"origin",
                &mut (*hdata).xorg,
                &mut (*hdata).yorg,
                &mut (*hdata).zorg,
            ) < 0
            {
                retval += 1;
            }
            if adoc_get_three_floats(
                ADOC_GLOBAL_NAME,
                0,
                b"step",
                &mut xscale,
                &mut yscale,
                &mut zscale,
            ) < 0
            {
                retval += 1;
            }
            mrc_set_scale(&mut *hdata, xscale as f64, yscale as f64, zscale as f64);
        } else if hdf_source == IIHDF_EMAN {
            let mut xscale = 0.0;
            let mut yscale = 0.0;
            let mut zscale = 0.0;
            if adoc_get_float(
                eman_sect_type.unwrap_or_default(),
                0,
                b"EMAN.apix_x",
                &mut xscale,
            ) < 0
                || adoc_get_float(
                    eman_sect_type.unwrap_or_default(),
                    0,
                    b"EMAN.apix_y",
                    &mut yscale,
                ) < 0
                || adoc_get_float(
                    eman_sect_type.unwrap_or_default(),
                    0,
                    b"EMAN.apix_z",
                    &mut zscale,
                ) < 0
            {
                retval += 1;
            }
            mrc_set_scale(&mut *hdata, xscale as f64, yscale as f64, zscale as f64);
            if eman_sect_type == Some(ADOC_GLOBAL_NAME) {
                if adoc_get_float(
                    eman_sect_type.unwrap_or_default(),
                    0,
                    b"EMAN.origin_x",
                    &mut (*hdata).xorg,
                ) < 0
                    || adoc_get_float(
                        eman_sect_type.unwrap_or_default(),
                        0,
                        b"EMAN.origin_y",
                        &mut (*hdata).yorg,
                    ) < 0
                    || adoc_get_float(
                        eman_sect_type.unwrap_or_default(),
                        0,
                        b"EMAN.origin_z",
                        &mut (*hdata).zorg,
                    ) < 0
                {
                    retval += 1;
                }
            }
            if single_image_stack {
                let mut origin_matches = true;
                let mut section_x_origin = 0.0;
                let mut section_y_origin = 0.0;
                let mut num_means = 0;
                for section in 0..(*in_file).nz {
                    if retval != 0 {
                        break;
                    }
                    let mut value = 0.0;
                    if adoc_get_float(ADOC_ZVALUE_NAME, section, b"EMAN.minimum", &mut value) == 0 {
                        (*hdata).amin = (*hdata).amin.min(value);
                    }
                    if adoc_get_float(ADOC_ZVALUE_NAME, section, b"EMAN.maximum", &mut value) == 0 {
                        (*hdata).amax = (*hdata).amax.max(value);
                    }
                    if adoc_get_float(ADOC_ZVALUE_NAME, section, b"EMAN.mean", &mut value) == 0 {
                        (*hdata).amean += value;
                        num_means += 1;
                    }
                    if origin_matches {
                        if adoc_get_float(ADOC_ZVALUE_NAME, section, b"EMAN.origin_x", &mut value)
                            != 0
                            || (section != 0 && value != section_x_origin)
                        {
                            origin_matches = false;
                        } else if section == 0 {
                            section_x_origin = value;
                        }
                        if adoc_get_float(ADOC_ZVALUE_NAME, section, b"EMAN.origin_y", &mut value)
                            != 0
                            || (section != 0 && value != section_y_origin)
                        {
                            origin_matches = false;
                        } else if section == 0 {
                            section_y_origin = value;
                        }
                    }
                }
                if num_means != 0 {
                    (*hdata).amean /= num_means as f32;
                }
                if origin_matches {
                    (*hdata).xorg = section_x_origin;
                    (*hdata).yorg = section_y_origin;
                }
            } else if adoc_get_float(
                eman_sect_type.unwrap_or_default(),
                0,
                b"EMAN.minimum",
                &mut (*hdata).amin,
            ) < 0
                || adoc_get_float(
                    eman_sect_type.unwrap_or_default(),
                    0,
                    b"EMAN.maximum",
                    &mut (*hdata).amax,
                ) < 0
                || adoc_get_float(
                    eman_sect_type.unwrap_or_default(),
                    0,
                    b"EMAN.mean",
                    &mut (*hdata).amean,
                ) < 0
            {
                retval += 1;
            }
        }
        (*hdata).xorg *= -1.0;
        (*hdata).yorg *= -1.0;
        (*hdata).zorg *= -1.0;
        if (*in_file).stack_set_list.is_none() {
            (*volume).tile_size_x = (*dataset).chunk_x;
            (*volume).tile_size_y = (*dataset).chunk_y;
            (*volume).z_chunk_size = (*dataset).chunk_z;
        }
        if (*hdata).bytes_signed != 0 {
            (*hdata).amin += 128.0;
            (*hdata).amax += 128.0;
            (*hdata).amean += 128.0;
        }
        (*volume).smin = (*hdata).amin;
        (*volume).smax = (*hdata).amax;
        (*volume).global_adoc_index = (*in_file).global_adoc_index;
        ii_sync_from_mrc_header(volume, hdata);
        set_io_funcs_plus(volume, hdf_source, writable as i32, file_id);
        if vol_ind != 0 && retval == 0 {
            // C `malloc(strlen(filename) + 10)` then
            // `sprintf(volume->filename, "%s-%d", inFile->filename, volInd + 1)`.
            (*volume).filename = Some(format!(
                "{}-{}",
                (*in_file).filename.as_deref().unwrap_or_default(),
                vol_ind + 1
            ));
        }
        if retval != 0 {
            cleanup_from_open(&mut state, file_id, 1, (*in_file).num_volumes, in_file);
            return IIERR_MEMORY_ERR;
        }
    }
    for vol_ind in 1..(*in_file).num_volumes {
        let volume = (&(*in_file).ii_volumes)[vol_ind as usize];
        H5Dclose((*volume).dataset_id);
        (*volume).dataset_is_open = 0;
        (*volume).state = 3;
        (*volume).fp = None;
        (*volume).fmode = (*in_file).fmode.clone();
    }
    cleanup_from_open(&mut state, file_id, 0, 0, in_file);
    0
}
/// C `scanGroup` (`iihdf.c:736`).
/// `group_name` is the owned group path, as in C where the `GroupData` entry
/// takes ownership of the `strdup`ed string.
unsafe fn scan_group(
    state: &mut HdfScanState,
    group_id: HidT,
    group_name: Vec<u8>,
    adoc_section: *mut i32,
) -> i32 {
    *adoc_section = 0;
    let group = GroupData {
        group_id,
        // The entry owns the name; the local copy below is what the C's
        // `groupName` argument keeps pointing at for the rest of the routine.
        name: Some(group_name.clone()),
        has_valid_datasets: 0,
        has_any_datasets: 0,
        has_groups: 0,
        non_global_attrib: 0,
        added_global: 0,
        adoc_collection: 0,
        num_attributes: 0,
    };
    // C `strrchr(groupName, '/')` then `strtol(objName + 1, &endptr, 10)`,
    // taking the number only when the whole tail converted (`!*endptr`).
    let mut group_number = 0;
    let mut numbered = false;
    if let Some(slash) = group_name.iter().rposition(|byte| *byte == b'/') {
        let tail = &group_name[slash + 1..];
        let mut end = 0;
        let value = crate::imod::libcfshr::parse_params::strtol(tail, &mut end, 10) as i32;
        if end == tail.len() {
            group_number = value;
            numbered = true;
        }
    }
    let list_index = state.groups.len();
    state.groups.push(group);
    let group_ptr = state.groups.as_mut_ptr().add(list_index);
    let mut object_info = core::mem::MaybeUninit::<H5OInfo>::uninit();
    if H5Oget_info(group_id, object_info.as_mut_ptr()) < 0 {
        return IIERR_IO_ERROR;
    }
    let mut object_info = object_info.assume_init();
    (*group_ptr).num_attributes = object_info.num_attrs as i32;
    let mut group_info = core::mem::MaybeUninit::<H5GInfo>::uninit();
    if H5Gget_info(group_id, group_info.as_mut_ptr()) < 0 {
        return IIERR_IO_ERROR;
    }
    let group_info = group_info.assume_init();
    let mut num_sets = 0;
    for ind in 0..group_info.nlinks {
        let size = H5Lget_name_by_idx(
            group_id,
            c".".as_ptr(),
            H5_INDEX_NAME,
            H5_ITER_INC,
            ind,
            core::ptr::null_mut(),
            0,
            0,
        ) + 1;
        if size <= 0 {
            return IIERR_IO_ERROR;
        }
        // HDF5 fills a `char *` of this size; the bytes are then the program's.
        let mut raw_name = vec![0u8; size as usize];
        if H5Lget_name_by_idx(
            group_id,
            c".".as_ptr(),
            H5_INDEX_NAME,
            H5_ITER_INC,
            ind,
            raw_name.as_mut_ptr().cast(),
            size as usize,
            0,
        ) < 0
        {
            return IIERR_IO_ERROR;
        }
        let terminator = raw_name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(raw_name.len());
        raw_name.truncate(terminator);
        let mut object_name = raw_name;
        if object_name.first() != Some(&b'/') {
            // C `sprintf(objName, "%s/%s", groupLength > 1 ? groupName : "",
            // relativeName)`.
            let group_length = group_name.len();
            object_name = c_format_bytes(
                "%s/%s",
                &[
                    CArg::Bytes(if group_length > 1 { &group_name } else { b"" }),
                    CArg::Bytes(&object_name),
                ],
            );
        }
        if H5Oget_info_by_idx(
            group_id,
            c".".as_ptr(),
            H5_INDEX_NAME,
            H5_ITER_INC,
            ind,
            &mut object_info,
            0,
        ) < 0
        {
            return IIERR_IO_ERROR;
        }
        if object_info.type_ == H5O_TYPE_GROUP {
            (*group_ptr).has_groups = 1;
            // The group path crosses into HDF5.
            let subgroup_name = std::ffi::CString::new(object_name.clone()).unwrap();
            let subgroup_id = H5Gopen2(group_id, subgroup_name.as_ptr(), 0);
            if subgroup_id < 0 {
                return IIERR_IO_ERROR;
            }
            let mut subsection = 0;
            let err = scan_group(state, subgroup_id, object_name, &mut subsection);
            if err != 0 {
                H5Gclose(subgroup_id);
                return err;
            }
            let parent = state.groups.as_mut_ptr().add(list_index);
            if subsection < 0 {
                (*parent).adoc_collection = -1;
            } else if subsection > 0 && (*parent).adoc_collection == 0 {
                (*parent).adoc_collection = 1;
            }
            continue;
        } else if object_info.type_ == H5O_TYPE_DATASET {
            (*group_ptr).has_any_datasets = 1;
            (*group_ptr).adoc_collection = -1;
            num_sets += 1;
            if num_sets > 1 {
                state.numbered_groups = false;
            }
            // The dataset path crosses into HDF5.
            let dataset_path = std::ffi::CString::new(object_name.clone()).unwrap();
            let dataset_id = H5Dopen2(group_id, dataset_path.as_ptr(), 0);
            if dataset_id < 0 {
                return IIERR_IO_ERROR;
            }
            let type_id = H5Dget_type(dataset_id);
            let bytes = (H5Tget_precision(type_id) / 8) as i32;
            let class = H5Tget_class(type_id);
            let signed = H5Tget_sign(type_id) == H5T_SGN_2;
            let data_type = if class == H5T_INTEGER && bytes == 1 {
                if signed { IITYPE_BYTE } else { IITYPE_UBYTE }
            } else if class == H5T_INTEGER && bytes == 2 {
                if signed { IITYPE_SHORT } else { IITYPE_USHORT }
            } else if class == H5T_FLOAT && bytes == 4 {
                IITYPE_FLOAT
            } else {
                -1
            };
            let space_id = H5Dget_space(dataset_id);
            let rank = H5Sget_simple_extent_ndims(space_id);
            if data_type >= 0 && (rank == 2 || rank == 3) {
                let mut current = [0; 3];
                let mut maximum = [0; 3];
                H5Sget_simple_extent_dims(space_id, current.as_mut_ptr(), maximum.as_mut_ptr());
                if H5Oget_info(dataset_id, &mut object_info) < 0 {
                    H5Sclose(space_id);
                    H5Tclose(type_id);
                    H5Dclose(dataset_id);
                    return IIERR_IO_ERROR;
                }
                let mut data = DatasetData {
                    dset_id: dataset_id,
                    name: Some(object_name),
                    data_type: data_type as i16,
                    swapped: if H5Tget_order(type_id) != H5Tget_order(H5T_NATIVE_INT_g) {
                        1
                    } else {
                        0
                    },
                    nx: current[(rank - 1) as usize] as i32,
                    ny: current[(rank - 2) as usize] as i32,
                    nz: if rank == 3 { current[0] as i32 } else { 1 },
                    chunk_x: 0,
                    chunk_y: 0,
                    chunk_z: 0,
                    num_attributes: object_info.num_attrs as i32,
                    group_num: group_number,
                };
                let plist_id = H5Dget_create_plist(dataset_id);
                if H5Pget_layout(plist_id) == H5D_CHUNKED {
                    let mut chunk = [0; 3];
                    if H5Pget_chunk(plist_id, 3, chunk.as_mut_ptr()) >= 0 {
                        if chunk[(rank - 1) as usize] < current[(rank - 1) as usize] {
                            data.chunk_x = chunk[(rank - 1) as usize] as i32;
                        }
                        if chunk[(rank - 2) as usize] < current[(rank - 2) as usize] {
                            data.chunk_y = chunk[(rank - 2) as usize] as i32;
                        }
                        if rank == 3 {
                            data.chunk_z = chunk[0] as i32;
                        }
                    }
                }
                H5Pclose(plist_id);
                if !numbered {
                    state.numbered_groups = false;
                }
                if state.numbered_groups {
                    state.min_group_num = state.min_group_num.min(group_number);
                    state.max_group_num = state.max_group_num.max(group_number);
                }
                state.datasets.push(data);
                H5Sclose(space_id);
                H5Tclose(type_id);
                continue;
            }
            H5Sclose(space_id);
            H5Tclose(type_id);
            H5Dclose(dataset_id);
        }
        // C `free(objName)`; the owned bytes go away with the loop iteration.
    }
    H5Gclose(group_id);
    let final_group = state.groups.as_mut_ptr().add(list_index);
    if (*final_group).has_any_datasets != 0 || (*final_group).has_groups != 0 {
        *adoc_section = -1;
    } else if (*final_group).num_attributes != 0 {
        *adoc_section = 1;
    }
    0
}
/// C `iiHDFopenNew` (`iihdf.c:979`).
pub unsafe fn ii_hdf_open_new(in_file: *mut ImodImageFile, mode: &str) -> i32 {
    if (*in_file).file != 0 {
        if (*in_file).file != IIFILE_HDF
            || (*in_file).hdf_source != IIHDF_IMOD
            || ((*in_file).stack_set_list.is_some() && (*in_file).nz > 1)
            || (*in_file).write_section.is_none()
        {
            return 1;
        }
        let mut volume = ii_new_box();
        let volume_ptr = volume.as_mut() as *mut ImodImageFile;
        let mut volumes = (*in_file).ii_volumes.clone();
        volumes.push(volume_ptr);
        (*in_file).owned_hdf_volumes.push(volume);
        (*in_file).num_volumes += 1;
        for &item in &volumes {
            (*item).ii_volumes = volumes.clone();
            (*item).num_volumes = (*in_file).num_volumes;
        }
        (*volume_ptr).adoc_index = adoc_new();
        (*volume_ptr).global_adoc_index = (*in_file).global_adoc_index;
        (*volume_ptr).mrc_header = Some(Box::default());
        (*volume_ptr).header = ((*volume_ptr)
            .mrc_header
            .as_deref_mut()
            .expect("new HDF volume header is present")
            as *mut MrcHeader)
            .cast();
        (*volume_ptr).fmode = (*in_file).fmode.clone();
        if (*volume_ptr).header.is_null() || (*volume_ptr).adoc_index < 0 {
            return 1;
        }
        set_io_funcs_plus(volume_ptr, IIHDF_IMOD, 1, (*in_file).hdf_file_id);
        mrc_head_new(&mut *(*in_file).header.cast::<MrcHeader>(), 1, 1, 1, 0);
        (*(*in_file).header.cast::<MrcHeader>()).packed4bits = 0;
        (*(*in_file).header.cast::<MrcHeader>()).half_floats = 0;
        (*volume_ptr).z_chunk_size = 1;
        return 0;
    }
    (*in_file).adoc_index = adoc_new();
    (*in_file).mrc_header = Some(Box::default());
    (*in_file).header = ((*in_file)
        .mrc_header
        .as_deref_mut()
        .expect("new HDF header is present") as *mut MrcHeader)
        .cast();
    (*in_file).ii_volumes = vec![in_file];
    let mut error = (*in_file).header.is_null() || (*in_file).adoc_index < 0;
    let mut file_id = -1;
    if !error {
        // The file name crosses into HDF5 as `char *`.
        let name =
            std::ffi::CString::new((*in_file).filename.as_deref().unwrap_or_default()).unwrap();
        file_id = H5Fcreate(name.as_ptr(), 2, 0, 0);
        if file_id < 0 {
            error = true;
        }
    }
    if !error {
        let mdf = H5Gcreate2(file_id, c"MDF".as_ptr(), 0, 0, 0);
        if mdf < 0 {
            error = true;
        } else {
            let images = H5Gcreate2(file_id, c"/MDF/images".as_ptr(), 0, 0, 0);
            if images < 0 {
                error = true;
            } else {
                H5Gclose(images);
            }
            H5Gclose(mdf);
        }
        if error {
            H5Fclose(file_id);
        }
    }
    if error {
        (*in_file).mrc_header = None;
        (*in_file).header = core::ptr::null_mut();
        if (*in_file).adoc_index >= 0 {
            adoc_clear((*in_file).adoc_index);
        }
        (*in_file).ii_volumes.clear();
        return 1;
    }
    (*in_file).num_volumes = 1;
    set_io_funcs_plus(in_file, IIHDF_IMOD, 1, file_id);
    mrc_head_new(&mut *(*in_file).header.cast::<MrcHeader>(), 1, 1, 1, 0);
    (*(*in_file).header.cast::<MrcHeader>()).packed4bits = 0;
    (*(*in_file).header.cast::<MrcHeader>()).half_floats = 0;
    0
}
/// C `iiReorderHDFstack` (`iihdf.c:1090`).
pub unsafe fn ii_reorder_hdf_stack(in_file: *mut ImodImageFile, sect_order: *mut i32) -> i32 {
    if in_file.is_null() || (*in_file).nz == 0 || (*in_file).z_map_size == 0 || sect_order.is_null()
    {
        return 1;
    }
    if (*in_file).stack_set_list.is_none() {
        return 1;
    }
    let mut new_size = 0;
    for iz in 0..(*in_file).nz {
        let new_z = *sect_order.add(iz as usize);
        if new_z < 0 {
            return 1;
        }
        new_size = new_size.max(new_z);
    }
    new_size += 8;
    let mut new_map = vec![-1; new_size as usize];
    let mut adoc_secs = vec![0; (*in_file).nz as usize];
    let mut ord_ind = 0;
    for map_ind in 0..(*in_file).z_map_size {
        let ds_ind = (&(*in_file).z_to_data_set_map)[map_ind as usize];
        if ds_ind < 0 {
            continue;
        }
        let new_z = *sect_order.add(ord_ind as usize);
        if new_map[new_z as usize] >= 0 {
            return 1;
        }
        new_map[new_z as usize] = ds_ind;
        ord_ind += 1;
    }
    // The source sized its temporary C strings from the longest name; owned
    // strings below grow naturally as needed.
    let file_id = (*in_file).hdf_file_id;
    ord_ind = 0;
    for map_ind in 0..(*in_file).z_map_size {
        let ds_ind = (&(*in_file).z_to_data_set_map)[map_ind as usize];
        if ds_ind < 0 {
            continue;
        }
        let new_z = *sect_order.add(ord_ind as usize);
        let Some(stack) = (*in_file)
            .stack_set_list
            .as_deref_mut()
            .and_then(|stacks| stacks.get_mut(ds_ind as usize))
        else {
            return 1;
        };
        // C `strcpy(tempName, stack->name)` then truncating the trailing
        // `/image`, and `sprintf(newName, "/MDF/images/Reordered%d", newZ)`.
        let mut temp_name = stack.name.clone().unwrap_or_default();
        temp_name.truncate(temp_name.len() - 6);
        let new_name = c_format("/MDF/images/Reordered%d", &[CArg::Int(new_z as i64)]);
        let temp_c = std::ffi::CString::new(temp_name).unwrap();
        let new_c = std::ffi::CString::new(new_name).unwrap();
        if H5Lmove(file_id, temp_c.as_ptr(), file_id, new_c.as_ptr(), 0, 0) < 0 {
            return 1;
        }
        adoc_secs[ord_ind as usize] = adoc_lookup_by_name_value(ADOC_ZVALUE_NAME, map_ind);
        ord_ind += 1;
    }
    ord_ind = 0;
    for map_ind in 0..(*in_file).z_map_size {
        let ds_ind = (&(*in_file).z_to_data_set_map)[map_ind as usize];
        if ds_ind < 0 {
            continue;
        }
        let new_z = *sect_order.add(ord_ind as usize);
        let Some(stack) = (*in_file)
            .stack_set_list
            .as_deref_mut()
            .and_then(|stacks| stacks.get_mut(ds_ind as usize))
        else {
            return 1;
        };
        let temp_c = std::ffi::CString::new(c_format(
            "/MDF/images/Reordered%d",
            &[CArg::Int(new_z as i64)],
        ))
        .unwrap();
        let mut new_name = c_format("/MDF/images/%d", &[CArg::Int(new_z as i64)]).into_bytes();
        let new_c = std::ffi::CString::new(new_name.clone()).unwrap();
        if H5Lmove(file_id, temp_c.as_ptr(), file_id, new_c.as_ptr(), 0, 0) < 0 {
            return 1;
        }
        // C `strcat(newName, "/image")`.
        new_name.extend_from_slice(b"/image");
        stack.name = Some(new_name);
        if adoc_secs[ord_ind as usize] >= 0 {
            let section_name = c_format("%d", &[CArg::Int(new_z as i64)]).into_bytes();
            if adoc_change_section_name(
                ADOC_ZVALUE_NAME,
                adoc_secs[ord_ind as usize],
                &section_name,
            ) != 0
            {
                return 1;
            }
        }
        ord_ind += 1;
    }
    (*in_file).z_to_data_set_map = new_map;
    (*in_file).z_map_size = new_size;
    0
}
/// C `removeAttributes` (`iihdf.c:1232`).
unsafe fn remove_attributes(group_id: HidT) -> i32 {
    let mut err = 0;
    let num = H5Aget_num_attrs(group_id);
    for ind in (0..num).rev() {
        if H5Adelete_by_idx(
            group_id,
            c".".as_ptr(),
            H5_INDEX_NAME,
            H5_ITER_INC,
            ind as HsizeT,
            0,
        ) < 0
        {
            err += 1;
        }
    }
    err
}
/// C `hdfWriteHeader` (`iihdf.c:1246`).
unsafe extern "C" fn hdf_write_header(in_file: *mut ImodImageFile) -> i32 {
    if (*in_file).stack_set_list.is_none()
        && (*in_file).dataset_name.is_none()
        && init_new_hdf_file(&mut *in_file) != 0
    {
        return 1;
    }
    let hdata = (*in_file).header.cast::<MrcHeader>();
    if hdata.is_null() {
        return 1;
    }
    if hdf_sync_from_mrc_header(in_file, hdata) != 0 {
        return 1;
    }
    if adoc_set_current((*in_file).adoc_index) != 0 {
        return 1;
    }
    // C `free(sMrcPrefix); sMrcPrefix = strdup("IMOD.MRC.");` with a failure
    // arm that an owned `String` cannot reach.
    *MRC_PREFIX.lock().unwrap() = Some("IMOD.MRC.".to_owned());
    let mut error = 0;
    let mut group_id = H5Gopen2((*in_file).hdf_file_id, c"/MDF/images".as_ptr(), 0);
    if group_id >= 0 {
        let mut image_id = (*in_file).num_volumes - 1;
        if (*in_file).stack_set_list.is_some() {
            image_id = (*in_file).nz - 1;
            for ind in (0..(*in_file).z_map_size).rev() {
                if (&(*in_file).z_to_data_set_map)[ind as usize] >= 0 {
                    image_id = image_id.max((&(*in_file).z_to_data_set_map)[ind as usize]);
                }
            }
        }
        if add_integer_attribute(group_id, b"imageid_max", &mut image_id, 1) != 0 {
            error += 1;
        }
        if add_string_attribute(group_id, b"DISPLAY_ORIGIN", b"LL") != 0 {
            error += 1;
        }
    }
    if error == 0 && group_id >= 0 && (*in_file).stack_set_list.is_none() {
        H5Gclose(group_id);
        group_id = open_dataset_group(
            in_file,
            (*in_file)
                .dataset_name
                .as_deref()
                .unwrap_or_default()
                .as_bytes(),
        );
    }
    if error != 0 || group_id < 0 {
        cleanup_malloc_bufs();
        if group_id >= 0 {
            H5Gclose(group_id);
        }
        return 1;
    }
    let mut value = if (*hdata).mode == MRC_MODE_RGB { 1 } else { 0 };
    if add_integer_attribute(group_id, b"IMOD.is_rgb", &mut value, 1) != 0 {
        error += 1;
    }
    value = if (*in_file).format == IIFORMAT_COMPLEX {
        1
    } else {
        0
    };
    if add_integer_attribute(group_id, b"IMOD.is_complex", &mut value, 1) != 0 {
        error += 1;
    }
    if error == 0 {
        add_one_prefixed_integer(group_id, b"nxstart", (*hdata).nxstart, &mut error);
        add_one_prefixed_integer(group_id, b"nystart", (*hdata).nystart, &mut error);
        add_one_prefixed_integer(group_id, b"nzstart", (*hdata).nzstart, &mut error);
        add_one_prefixed_integer(group_id, b"mx", (*hdata).mx, &mut error);
        add_one_prefixed_integer(group_id, b"my", (*hdata).my, &mut error);
        add_one_prefixed_integer(group_id, b"mz", (*hdata).mz, &mut error);
        add_one_prefixed_float(group_id, b"xlen", (*hdata).xlen, &mut error);
        add_one_prefixed_float(group_id, b"ylen", (*hdata).ylen, &mut error);
        add_one_prefixed_float(group_id, b"zlen", (*hdata).zlen, &mut error);
        add_one_prefixed_float(group_id, b"alpha", (*hdata).alpha, &mut error);
        add_one_prefixed_float(group_id, b"beta", (*hdata).beta, &mut error);
        add_one_prefixed_float(group_id, b"gamma", (*hdata).gamma, &mut error);
        add_one_prefixed_integer(group_id, b"mapc", (*hdata).mapc, &mut error);
        add_one_prefixed_integer(group_id, b"mapr", (*hdata).mapr, &mut error);
        add_one_prefixed_integer(group_id, b"maps", (*hdata).maps, &mut error);
        let mmm_offset = if (*hdata).mode == MRC_MODE_BYTE && (*hdata).bytes_signed != 0 {
            -128.0
        } else {
            0.0
        };
        add_one_prefixed_float(group_id, b"minimum", (*hdata).amin + mmm_offset, &mut error);
        add_one_prefixed_float(group_id, b"maximum", (*hdata).amax + mmm_offset, &mut error);
        add_one_prefixed_float(group_id, b"mean", (*hdata).amean + mmm_offset, &mut error);
        add_one_prefixed_integer(group_id, b"ispg", (*hdata).ispg, &mut error);
        add_one_prefixed_float(group_id, b"xorigin", -(*hdata).xorg, &mut error);
        add_one_prefixed_float(group_id, b"yorigin", -(*hdata).yorg, &mut error);
        add_one_prefixed_float(group_id, b"zorigin", -(*hdata).zorg, &mut error);
        if add_float_attribute(
            group_id,
            b"IMOD.MRC.tiltangles",
            (*hdata).tiltangles.as_mut_ptr(),
            6,
        ) != 0
        {
            error += 1;
        }
        add_one_prefixed_float(group_id, b"rms", (*hdata).rms, &mut error);
        add_one_prefixed_integer(group_id, b"nlabels", (*hdata).nlabl, &mut error);
        for ind in 0..(*hdata).nlabl.min(10) {
            // C `sprintf(labelKey, "IMOD.MRC.label%d", ind)` into `char[20]`.
            let label_key = c_format("IMOD.MRC.label%d", &[CArg::Int(ind as i64)]);
            let label = &(*hdata).labels[ind as usize];
            let label_end = label.iter().position(|byte| *byte == 0).unwrap_or(80);
            if add_string_attribute(group_id, label_key.as_bytes(), &label[..label_end]) != 0 {
                error += 1;
            }
        }
    }
    if error == 0 && adoc_to_attributes(group_id, ADOC_GLOBAL_NAME, 0, Some(b"IMOD.")) != 0 {
        error = 1;
    }
    H5Gclose(group_id);
    if error == 0 && (*in_file).num_volumes < 2 {
        error = hdf_write_global_adoc(in_file);
    }
    if (*in_file).stack_set_list.is_some() {
        for iz in 0..(*in_file).z_map_size {
            if error != 0 {
                break;
            }
            let ind = (&(*in_file).z_to_data_set_map)[iz as usize];
            if ind < 0 {
                continue;
            }
            // C `sprintf(sectionName, "%d", iz)` into `char[20]`.
            let section_name = c_format("%d", &[CArg::Int(iz as i64)]).into_bytes();
            let section = adoc_lookup_section(ADOC_ZVALUE_NAME, &section_name);
            if section < 0 {
                continue;
            }
            let Some(stack) = (*in_file)
                .stack_set_list
                .as_deref()
                .and_then(|stacks| stacks.get(ind as usize))
            else {
                error = 1;
                continue;
            };
            let section_group =
                open_dataset_group(in_file, stack.name.as_deref().unwrap_or_default());
            if section_group < 0 {
                error = 1;
            } else {
                if adoc_to_attributes(section_group, ADOC_ZVALUE_NAME, section, None) != 0 {
                    error = 1;
                }
                H5Gclose(section_group);
            }
        }
    }
    cleanup_malloc_bufs();
    error
}
/// C `hdfWriteGlobalAdoc` (`iihdf.c:1412`).
pub unsafe fn hdf_write_global_adoc(in_file: *mut ImodImageFile) -> i32 {
    if (*in_file).write_header.is_none() {
        return 1;
    }
    if adoc_set_current(if (*in_file).global_adoc_index >= 0 {
        (*in_file).global_adoc_index
    } else {
        (*in_file).adoc_index
    }) != 0
    {
        return 1;
    }
    let group = H5Gopen2((*in_file).hdf_file_id, c"/MDF/images".as_ptr(), 0);
    if group < 0 {
        return 1;
    }
    let mut err = 0;
    if (*in_file).global_adoc_index >= 0 {
        err = adoc_to_attributes(group, ADOC_GLOBAL_NAME, 0, None);
    }
    let num_collections = adoc_get_num_collections();
    for coll in 0..num_collections {
        if err != 0 {
            break;
        }
        let mut collection_name = Vec::new();
        if adoc_get_collection_name(coll, &mut collection_name) != 0 {
            err = 1;
            continue;
        }
        if collection_name != ADOC_ZVALUE_NAME && collection_name != b"T" {
            let num_sections = adoc_get_number_of_sections(&collection_name);
            // The collection is also an HDF5 group name.
            let collection_path = std::ffi::CString::new(&collection_name[..]).unwrap();
            let mut collection = H5Gopen2(group, collection_path.as_ptr(), 0);
            if collection < 0 {
                collection = H5Gcreate2(group, collection_path.as_ptr(), 0, 0, 0);
            }
            if collection < 0 {
                err = 1;
            } else {
                for section in 0..num_sections {
                    if err != 0 {
                        break;
                    }
                    let mut section_name = Vec::new();
                    if adoc_get_section_name(&collection_name, section, &mut section_name) != 0 {
                        err = 1;
                    } else {
                        let section_path = std::ffi::CString::new(&section_name[..]).unwrap();
                        let mut section_id = H5Gopen2(collection, section_path.as_ptr(), 0);
                        if section_id < 0 {
                            section_id = H5Gcreate2(collection, section_path.as_ptr(), 0, 0, 0);
                        }
                        if section_id < 0 {
                            err = 1;
                        } else {
                            if adoc_to_attributes(section_id, &collection_name, section, None) != 0
                            {
                                err = 1;
                            }
                            H5Gclose(section_id);
                        }
                    }
                }
                H5Gclose(collection);
            }
        }
    }
    H5Gclose(group);
    err
}
/// C `hdfSyncFromMrcHeader` (`iihdf.c:1484`).
unsafe extern "C" fn hdf_sync_from_mrc_header(
    in_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
) -> i32 {
    if !in_file.is_null() && !hdata.is_null() && (*in_file).header as *mut MrcHeader != hdata {
        // A `clone`, not a bitwise copy: see `mrcsec::mrc_write_z`.
        *((*in_file).header as *mut MrcHeader) = (*hdata).clone();
    }
    0
}
/// C `hdfClose` (`iihdf.c:1495`).
unsafe extern "C" fn hdf_close(in_file: *mut ImodImageFile) {
    if (*in_file).dataset_is_open != 0 {
        H5Dclose((*in_file).dataset_id);
    }
    (*in_file).dataset_is_open = 0;
    let mut num_left = 0;
    for ind in 0..(*in_file).num_volumes {
        let volume = (&(*in_file).ii_volumes)[ind as usize];
        if !volume.is_null() && volume != in_file && (*volume).fp.is_some() {
            num_left += 1;
        }
    }
    if num_left == 0 && (*in_file).fp.is_some() {
        for stack in (*in_file).stack_set_list.iter_mut().flatten() {
            if stack.is_open {
                H5Dclose(stack.dset_id);
            }
            stack.is_open = false;
        }
        H5Fclose((*in_file).hdf_file_id);
    }
    (*in_file).fp = None;
}
/// C `hdfReopen` (`iihdf.c:1523`).
unsafe extern "C" fn hdf_reopen(in_file: *mut ImodImageFile) -> i32 {
    for ind in 0..(*in_file).num_volumes {
        let volume = (&(*in_file).ii_volumes)[ind as usize];
        if !volume.is_null() && (*volume).fp.is_some() {
            // `iihdf.c:1529`: `(FILE *)inFile` — the file's own address as its
            // identity token, never used for I/O.
            (*in_file).fp = Some(ImodFile::Token(in_file as usize));
            return 0;
        }
    }
    if (*in_file).state == IISTATE_NOTINIT && (*in_file).filename.is_some() {
        hdf_delete(in_file);
        return ii_hdf_check(in_file);
    }
    // The file name is what crosses into HDF5.
    let name = std::ffi::CString::new(
        (*(&(*in_file).ii_volumes)[0])
            .filename
            .clone()
            .unwrap_or_default(),
    )
    .unwrap();
    let file_id = H5Fopen(
        name.as_ptr(),
        // C `strstr(inFile->fmode, "+")`: `fmode` is NUL-padded, so no byte
        // past the terminator can be a `+`.
        if (*in_file).fmode.contains('+') { 1 } else { 0 },
        0,
    );
    if file_id < 0 {
        return IIERR_IO_ERROR;
    }
    for ind in 0..(*in_file).num_volumes {
        let volume = (&(*in_file).ii_volumes)[ind as usize];
        (*volume).hdf_file_id = file_id;
    }
    (*in_file).fp = Some(ImodFile::Token(in_file as usize));
    0
}
/// C `hdfDelete` (`iihdf.c:1555`).
unsafe extern "C" fn hdf_delete(in_file: *mut ImodImageFile) {
    let primary = (*in_file).ii_volumes.first().copied();
    let mut num_left = 0;
    for ind in 0..(*in_file).num_volumes {
        let volume = (&(*in_file).ii_volumes)[ind as usize];
        if !volume.is_null() && volume != in_file && (*volume).state != IISTATE_UNUSED {
            num_left += 1;
        }
    }
    for ind in 0..(*in_file).num_volumes {
        if (&(*in_file).ii_volumes)[ind as usize] == in_file {
            for other in 0..(*in_file).num_volumes {
                let volume = (&(*in_file).ii_volumes)[other as usize];
                if other != ind && !volume.is_null() {
                    (&mut (*volume).ii_volumes)[ind as usize] = core::ptr::null_mut();
                }
            }
            (&mut (*in_file).ii_volumes)[ind as usize] = core::ptr::null_mut();
        }
    }
    if num_left == 0 && (*in_file).global_adoc_index >= 0 {
        adoc_clear((*in_file).global_adoc_index);
    }
    if num_left == 0 {
        (*in_file).ii_volumes.clear();
    }
    if (*in_file).adoc_index >= 0 {
        adoc_clear((*in_file).adoc_index);
    }
    (*in_file).dataset_name = None;
    // Dropping the owned vector releases each dataset path.
    (*in_file).stack_set_list = None;
    (*in_file).z_to_data_set_map.clear();
    (*in_file).z_map_size = 0;
    (*in_file).mrc_header = None;
    (*in_file).header = core::ptr::null_mut();

    let Some(primary) = primary else {
        return;
    };
    if in_file == primary {
        if num_left != 0 {
            // The legacy API permits closing the primary while secondary volume
            // cursors are still in use.  Transfer those boxes to the raw
            // `iiDelete` boundary; their later deletion reconstructs the box.
            for volume in (*primary).owned_hdf_volumes.drain(..) {
                let _ = Box::into_raw(volume);
            }
        } else {
            for mut volume in (*primary).owned_hdf_volumes.drain(..) {
                ii_close(volume.as_mut());
                if volume.adoc_index >= 0 {
                    adoc_clear(volume.adoc_index);
                }
                volume.mrc_header = None;
                volume.header = core::ptr::null_mut();
            }
        }
    } else if let Some(index) = (*primary)
        .owned_hdf_volumes
        .iter()
        .position(|volume| core::ptr::eq(volume.as_ref(), &*in_file))
    {
        // `iiDelete` immediately reclaims this legacy cursor after its cleanup
        // callback returns, so relinquish exactly this box to that boundary.
        let volume = (*primary).owned_hdf_volumes.swap_remove(index);
        debug_assert_eq!(Box::into_raw(volume), in_file);
    }
}
/// C `hdfReadSection` (`iihdf.c:1611`).
unsafe extern "C" fn hdf_read_section(in_file: *mut ImodImageFile, buf: *mut u8, cz: i32) -> i32 {
    hdf_read_section_any(in_file, buf, cz, 0)
}
/// C `hdfReadSectionByte` (`iihdf.c:1616`).
unsafe extern "C" fn hdf_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    cz: i32,
) -> i32 {
    hdf_read_section_any(in_file, buf, cz, MRSA_BYTE)
}
/// C `hdfReadSectionUShort` (`iihdf.c:1621`).
unsafe extern "C" fn hdf_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    cz: i32,
) -> i32 {
    hdf_read_section_any(in_file, buf, cz, MRSA_USHORT)
}
/// C `hdfReadSectionFloat` (`iihdf.c:1626`).
unsafe extern "C" fn hdf_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    cz: i32,
) -> i32 {
    hdf_read_section_any(in_file, buf, cz, MRSA_FLOAT)
}
/// C `hdfWriteSection` (`iihdf.c:1631`).
unsafe extern "C" fn hdf_write_section(in_file: *mut ImodImageFile, buf: *mut u8, cz: i32) -> i32 {
    hdf_write_section_any(in_file, buf, cz, 0)
}
/// C `hdfWriteSectionFloat` (`iihdf.c:1636`).
unsafe extern "C" fn hdf_write_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    cz: i32,
) -> i32 {
    hdf_write_section_any(in_file, buf, cz, 1)
}
/// C `hdfWriteDummySection` (`iihdf.c:1641`).
pub unsafe fn hdf_write_dummy_section(in_file: *mut ImodImageFile, buf: *mut u8, cz: i32) -> i32 {
    hdf_write_section_any(in_file, buf, cz, -1)
}
/// C `setIOFuncsPlus` (`iihdf.c:1649`).
unsafe fn set_io_funcs_plus(
    in_file: *mut ImodImageFile,
    hdf_source: i32,
    writable: i32,
    file_id: HidT,
) {
    if in_file.is_null() {
        return;
    }
    let f = &mut *in_file;
    f.hdf_source = hdf_source;
    f.hdf_file_id = file_id;
    f.file = IIFILE_HDF;
    f.fp = Some(ImodFile::Token(in_file as usize));
    if !f.header.is_null() {
        (*f.header.cast::<MrcHeader>()).fp = f.fp.clone();
    }
    f.read_section = Some(hdf_read_section);
    f.read_section_byte = Some(hdf_read_section_byte);
    f.read_section_ushort = Some(hdf_read_section_ushort);
    f.read_section_float = Some(hdf_read_section_float);
    f.close = Some(hdf_close);
    f.clean_up = Some(hdf_delete);
    f.reopen = Some(hdf_reopen);
    f.fill_mrc_header = Some(ii_mrc_fill_header);
    f.sync_from_mrc_header = Some(hdf_sync_from_mrc_header);
    if writable != 0 && hdf_source == IIHDF_IMOD {
        f.write_section = Some(hdf_write_section);
        f.write_section_float = Some(hdf_write_section_float);
        f.write_header = Some(hdf_write_header);
    }
}
/// C `setupZtoSetMap` (`iihdf.c:1677`).
unsafe fn setup_zto_set_map(
    file_id: HidT,
    in_file: *mut ImodImageFile,
    size: i32,
    sequence: i32,
) -> i32 {
    (*in_file).z_to_data_set_map = vec![if sequence != 0 { 0 } else { -1 }; size as usize];
    (*in_file).z_map_size = size;
    for set in 0..size {
        (&mut (*in_file).z_to_data_set_map)[set as usize] = if sequence != 0 { set } else { -1 };
    }
    0
}
/// C `openDatasetGroup` (`iihdf.c:1694`).
unsafe fn open_dataset_group(in_file: *mut ImodImageFile, ds_name: &[u8]) -> HidT {
    let Some(ind) = ds_name.iter().rposition(|byte| *byte == b'/') else {
        return -1;
    };
    // The group name is what crosses into HDF5; build it there.
    let group_name = std::ffi::CString::new(&ds_name[..ind]).unwrap();
    H5Gopen2((*in_file).hdf_file_id, group_name.as_ptr(), 0)
}
/// C `cleanupFromOpen` (`iihdf.c:1711`).
unsafe fn cleanup_from_open(
    state: &mut HdfScanState,
    file_id: HidT,
    close_file: i32,
    num_vol: i32,
    in_file: *mut ImodImageFile,
) {
    if close_file != 0 {
        H5Fclose(file_id);
        (*in_file).fp = ImodFile::open(
            (*in_file).filename.as_deref().unwrap_or_default(),
            &(*in_file).fmode,
        );
    }
    for group in &mut state.groups {
        group.name = None;
    }
    state.groups.clear();
    for dataset in &mut state.datasets {
        if close_file != 0 {
            H5Dclose(dataset.dset_id);
        }
        dataset.name = None;
    }
    state.datasets.clear();
    if num_vol != 0 && !(*in_file).ii_volumes.is_empty() {
        for mut volume in (*in_file).owned_hdf_volumes.drain(..) {
            volume.mrc_header = None;
            volume.header = core::ptr::null_mut();
        }
        (*in_file).ii_volumes.clear();
    }
    cleanup_malloc_bufs();
}
/// C `cleanupMallocBufs` (`iihdf.c:1764`).
unsafe fn cleanup_malloc_bufs() {
    // C also frees `sStringBuf` and `sAttribName` here; both are now owned by
    // the routines that build them and go away with their scopes.
    FLOAT_BUF.lock().unwrap().clear();
    INT_BUF.lock().unwrap().clear();
    SHORT_BUF.lock().unwrap().clear();
    USHORT_BUF.lock().unwrap().clear();
    *MRC_PREFIX.lock().unwrap() = None;
}
/// C `attributesToAdoc` (`iihdf.c:1789`).
unsafe fn attributes_to_adoc(
    obj_id: HidT,
    num_attrib: i32,
    type_name: &[u8],
    sect_ind: i32,
) -> i32 {
    let mut S_FLOAT_BUF = FLOAT_BUF.lock().unwrap();
    let mut S_INT_BUF = INT_BUF.lock().unwrap();
    let mut S_SHORT_BUF = SHORT_BUF.lock().unwrap();
    for ind in 0..num_attrib {
        let attrib_id = H5Aopen_by_idx(
            obj_id,
            c".".as_ptr(),
            H5_INDEX_NAME,
            H5_ITER_INC,
            ind as HsizeT,
            0,
            0,
        );
        let space_id = H5Aget_space(attrib_id);
        let type_id = H5Aget_type(attrib_id);
        let len = H5Aget_name(attrib_id, 0, core::ptr::null_mut()) as i32 + 1;
        // C `manageMallocBuf(&sAttribName, &sAttrNameSize, len, 1)`, then
        // `H5Aget_name` into it.  HDF5 fills a `char *`, so the buffer is a
        // `Vec<u8>` whose pointer is handed over at the call.
        let mut attrib_name = vec![0u8; len.max(1) as usize];
        H5Aget_name(
            attrib_id,
            attrib_name.len(),
            attrib_name.as_mut_ptr().cast(),
        );
        let terminator = attrib_name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(attrib_name.len());
        attrib_name.truncate(terminator);
        let s_attrib_name: &[u8] = &attrib_name;
        let class = H5Tget_class(type_id);
        let space_type = H5Sget_simple_extent_type(space_id);
        let mut retval = 0;
        if (class == H5T_INTEGER || class == H5T_FLOAT) && space_type == H5S_SIMPLE {
            let dsize = (H5Tget_precision(type_id) / 8) as i32;
            let signed_int = H5Tget_sign(type_id) == H5T_SGN_2;
            let num_vals = H5Sget_simple_extent_npoints(space_id) as i32;
            if class == H5T_FLOAT && dsize == 4 {
                S_FLOAT_BUF.resize(num_vals.max(0) as usize, 0.0);
                if H5Aread(
                    attrib_id,
                    H5T_NATIVE_FLOAT_g,
                    S_FLOAT_BUF.as_mut_ptr().cast(),
                ) < 0
                {
                    retval = IIERR_IO_ERROR;
                }
                if retval == 0 {
                    retval = match num_vals {
                        1 => adoc_set_float(type_name, sect_ind, s_attrib_name, S_FLOAT_BUF[0]),
                        2 => adoc_set_two_floats(
                            type_name,
                            sect_ind,
                            s_attrib_name,
                            S_FLOAT_BUF[0],
                            S_FLOAT_BUF[1],
                        ),
                        3 => adoc_set_three_floats(
                            type_name,
                            sect_ind,
                            s_attrib_name,
                            S_FLOAT_BUF[0],
                            S_FLOAT_BUF[1],
                            S_FLOAT_BUF[2],
                        ),
                        _ => adoc_set_float_array(
                            type_name,
                            sect_ind,
                            s_attrib_name,
                            &S_FLOAT_BUF,
                            num_vals,
                        ),
                    };
                    if retval != 0 {
                        retval = IIERR_MEMORY_ERR;
                    }
                }
            } else if dsize == 2 || (dsize == 4 && signed_int) {
                S_INT_BUF.resize(num_vals.max(0) as usize, 0);
                if dsize == 2 {
                    S_SHORT_BUF.resize(num_vals.max(0) as usize, 0);
                }
                if retval == 0 && dsize == 2 {
                    if H5Aread(
                        attrib_id,
                        if signed_int {
                            H5T_NATIVE_SHORT_g
                        } else {
                            H5T_NATIVE_USHORT_g
                        },
                        S_SHORT_BUF.as_mut_ptr().cast(),
                    ) < 0
                    {
                        retval = IIERR_IO_ERROR;
                    }
                    if retval == 0 {
                        for val in 0..num_vals as usize {
                            S_INT_BUF[val] = if signed_int {
                                S_SHORT_BUF[val] as i32
                            } else {
                                S_SHORT_BUF[val] as u16 as i32
                            };
                        }
                    }
                } else if retval == 0
                    && H5Aread(attrib_id, H5T_NATIVE_INT_g, S_INT_BUF.as_mut_ptr().cast()) < 0
                {
                    retval = IIERR_IO_ERROR;
                }
                if retval == 0 {
                    retval = match num_vals {
                        1 => adoc_set_integer(type_name, sect_ind, s_attrib_name, S_INT_BUF[0]),
                        2 => adoc_set_two_integers(
                            type_name,
                            sect_ind,
                            s_attrib_name,
                            S_INT_BUF[0],
                            S_INT_BUF[1],
                        ),
                        3 => adoc_set_three_integers(
                            type_name,
                            sect_ind,
                            s_attrib_name,
                            S_INT_BUF[0],
                            S_INT_BUF[1],
                            S_INT_BUF[2],
                        ),
                        _ => adoc_set_integer_array(
                            type_name,
                            sect_ind,
                            s_attrib_name,
                            &S_INT_BUF,
                            num_vals,
                        ),
                    };
                    if retval != 0 {
                        retval = IIERR_MEMORY_ERR;
                    }
                }
            }
        } else if class == H5T_STRING && space_type == 0 {
            let string_type = H5Tget_native_type(type_id, 0);
            // C `manageMallocBuf(&sStringBuf, &sStrBufSize, H5Tget_size + 1, 1)`.
            let mut string_buf = vec![0u8; H5Tget_size(type_id) + 1];
            if H5Aread(attrib_id, string_type, string_buf.as_mut_ptr().cast()) < 0 {
                retval = IIERR_IO_ERROR;
            }
            if retval == 0
                && adoc_set_key_value(
                    type_name,
                    sect_ind,
                    s_attrib_name,
                    // The value stops at the terminator HDF5 wrote.
                    Some(
                        &string_buf[..string_buf
                            .iter()
                            .position(|byte| *byte == 0)
                            .unwrap_or(string_buf.len())],
                    ),
                ) != 0
            {
                retval = IIERR_MEMORY_ERR;
            }
            H5Tclose(string_type);
        }
        H5Sclose(space_id);
        H5Tclose(type_id);
        H5Aclose(attrib_id);
        if retval != 0 {
            return retval;
        }
    }
    0
}
/// C `adocToAttributes` (`iihdf.c:1909`).
unsafe fn adoc_to_attributes(
    parent_id: HidT,
    type_name: &[u8],
    sect_ind: i32,
    prefix: Option<&[u8]>,
) -> i32 {
    let mut S_FLOAT_BUF = FLOAT_BUF.lock().unwrap();
    let mut S_INT_BUF = INT_BUF.lock().unwrap();
    let num_keys = adoc_get_number_of_keys(type_name, sect_ind);
    if num_keys < 0 {
        return 1;
    }
    let mut retval = 0;
    for key_ind in 0..num_keys {
        let mut key = None;
        if adoc_get_key_by_index(type_name, sect_ind, key_ind, &mut key) < 0 {
            return 1;
        }
        let Some(key) = key else {
            continue;
        };
        let pref_key = if prefix.is_some() && starts_with(&key, prefix.unwrap()) == 0 {
            prefixed_key(prefix, &key)
        } else {
            key.clone()
        };
        let mut value_type = 0;
        let mut num_vals = 0;
        if adoc_get_val_type_and_size(type_name, sect_ind, &key, &mut value_type, &mut num_vals)
            != 0
        {
            retval = 1;
        }
        if retval == 0 && value_type == ADOC_STRING {
            let mut value = Vec::new();
            if adoc_get_string(type_name, sect_ind, &key, &mut value) != 0 {
                retval = 1;
            } else {
                retval = add_string_attribute(parent_id, &pref_key, &value);
            }
        } else if retval == 0 && (1..=4).contains(&value_type) {
            S_INT_BUF.resize(num_vals.max(0) as usize, 0);
            let capacity = S_INT_BUF.len() as i32;
            num_vals = 0;
            if adoc_get_integer_array(
                type_name,
                sect_ind,
                &key,
                &mut S_INT_BUF,
                &mut num_vals,
                capacity,
            ) != 0
                || add_integer_attribute(parent_id, &pref_key, S_INT_BUF.as_mut_ptr(), num_vals)
                    != 0
            {
                retval = 1;
            }
        } else if retval == 0 && (5..=8).contains(&value_type) {
            S_FLOAT_BUF.resize(num_vals.max(0) as usize, 0.0);
            let capacity = S_FLOAT_BUF.len() as i32;
            num_vals = 0;
            if adoc_get_float_array(
                type_name,
                sect_ind,
                &key,
                &mut S_FLOAT_BUF,
                &mut num_vals,
                capacity,
            ) != 0
                || add_float_attribute(parent_id, &pref_key, S_FLOAT_BUF.as_mut_ptr(), num_vals)
                    != 0
            {
                retval = 1;
            }
        }
        if retval < 0 {
            return 1;
        }
    }
    0
}
/// C `addIntegerAttribute` (`iihdf.c:1968`).
unsafe fn add_integer_attribute(
    parent_id: HidT,
    key: &[u8],
    ivals: *mut i32,
    num_vals: i32,
) -> i32 {
    // The attribute name crosses into HDF5, which takes it as `char *`.
    let key = std::ffi::CString::new(key).unwrap();
    if H5Aexists_by_name(parent_id, c".".as_ptr(), key.as_ptr(), 0) > 0 {
        H5Adelete_by_name(parent_id, c".".as_ptr(), key.as_ptr(), 0);
    }
    let count = num_vals as HsizeT;
    let space = H5Screate_simple(1, &count, &count);
    if space < 0 {
        return 1;
    }
    let attribute = H5Acreate2(parent_id, key.as_ptr(), H5T_NATIVE_INT_g, space, 0, 0);
    let result = if attribute < 0 {
        -1
    } else {
        let value = H5Awrite(attribute, H5T_NATIVE_INT_g, ivals.cast());
        H5Aclose(attribute);
        value
    };
    H5Sclose(space);
    if result < 0 { 1 } else { 0 }
}
/// C `addFloatAttribute` (`iihdf.c:1993`).
unsafe fn add_float_attribute(parent_id: HidT, key: &[u8], vals: *mut f32, num_vals: i32) -> i32 {
    // The attribute name crosses into HDF5, which takes it as `char *`.
    let key = std::ffi::CString::new(key).unwrap();
    if H5Aexists_by_name(parent_id, c".".as_ptr(), key.as_ptr(), 0) > 0 {
        H5Adelete_by_name(parent_id, c".".as_ptr(), key.as_ptr(), 0);
    }
    let count = num_vals as HsizeT;
    let space = H5Screate_simple(1, &count, &count);
    if space < 0 {
        return 1;
    }
    let attribute = H5Acreate2(parent_id, key.as_ptr(), H5T_NATIVE_FLOAT_g, space, 0, 0);
    let result = if attribute < 0 {
        -1
    } else {
        let value = H5Awrite(attribute, H5T_NATIVE_FLOAT_g, vals.cast());
        H5Aclose(attribute);
        value
    };
    H5Sclose(space);
    if result < 0 { 1 } else { 0 }
}
/// C `addStringAttribute` (`iihdf.c:2019`).
unsafe fn add_string_attribute(parent_id: HidT, key: &[u8], val_str: &[u8]) -> i32 {
    // Both the attribute name and its value cross into HDF5 as `char *`.
    let key = std::ffi::CString::new(key).unwrap();
    let val_str = std::ffi::CString::new(val_str).unwrap();
    if H5Aexists_by_name(parent_id, c".".as_ptr(), key.as_ptr(), 0) > 0 {
        H5Adelete_by_name(parent_id, c".".as_ptr(), key.as_ptr(), 0);
    }
    let space = H5Screate(0);
    if space < 0 {
        return 1;
    }
    let typ = H5Tcopy(H5T_C_S1_g);
    if typ < 0 {
        H5Sclose(space);
        return 1;
    }
    H5Tset_size(typ, val_str.as_bytes().len() + 1);
    H5Tset_strpad(typ, 0);
    let attribute = H5Acreate2(parent_id, key.as_ptr(), typ, space, 0, 0);
    let result = if attribute < 0 {
        -1
    } else {
        let value = H5Awrite(attribute, typ, val_str.as_ptr().cast());
        H5Aclose(attribute);
        value
    };
    H5Sclose(space);
    H5Tclose(typ);
    if result < 0 { 1 } else { 0 }
}
/// C `addOnePrefixedInteger` (`iihdf.c:2052`).
unsafe fn add_one_prefixed_integer(
    parent_id: HidT,
    key: &[u8],
    mut ival: i32,
    err_sum: *mut i32,
) -> i32 {
    let err = add_integer_attribute(
        parent_id,
        &prefixed_key(
            MRC_PREFIX.lock().unwrap().as_deref().map(str::as_bytes),
            key,
        ),
        &raw mut ival,
        1,
    );
    if err != 0 {
        *err_sum += 1;
    }
    err
}
/// C `addOnePrefixedFloat` (`iihdf.c:2060`).
unsafe fn add_one_prefixed_float(
    parent_id: HidT,
    key: &[u8],
    mut val: f32,
    err_sum: *mut i32,
) -> i32 {
    let err = add_float_attribute(
        parent_id,
        &prefixed_key(
            MRC_PREFIX.lock().unwrap().as_deref().map(str::as_bytes),
            key,
        ),
        &raw mut val,
        1,
    );
    if err != 0 {
        *err_sum += 1;
    }
    err
}
/// C `getPrefixedInteger` (`iihdf.c:2071`).
unsafe fn get_prefixed_integer(key: &[u8], value: *mut i32, err_sum: *mut i32) -> i32 {
    let full = prefixed_key(
        MRC_PREFIX.lock().unwrap().as_deref().map(str::as_bytes),
        key,
    );
    let err = adoc_get_integer(ADOC_GLOBAL_NAME, 0, &full, &mut *value);
    if err < 0 {
        *err_sum += 1;
    }
    err
}
/// C `getDelPrefixedInteger` (`iihdf.c:2082`).
unsafe fn get_del_prefixed_integer(key: &[u8], value: *mut i32, err_sum: *mut i32) -> i32 {
    let full = prefixed_key(
        MRC_PREFIX.lock().unwrap().as_deref().map(str::as_bytes),
        key,
    );
    let err = adoc_get_integer(ADOC_GLOBAL_NAME, 0, &full, &mut *value);
    if err == 0 {
        delete_prefixed_key_value(key);
    }
    if err < 0 {
        *err_sum += 1;
    }
    err
}
/// C `getDelPrefixedFloat` (`iihdf.c:2092`).
unsafe fn get_del_prefixed_float(key: &[u8], value: *mut f32, err_sum: *mut i32) -> i32 {
    let full = prefixed_key(
        MRC_PREFIX.lock().unwrap().as_deref().map(str::as_bytes),
        key,
    );
    let err = adoc_get_float(ADOC_GLOBAL_NAME, 0, &full, &mut *value);
    if err == 0 {
        delete_prefixed_key_value(key);
    }
    if err < 0 {
        *err_sum += 1;
    }
    err
}
/// C `deletePrefixedKeyValue` (`iihdf.c:2103`).
unsafe fn delete_prefixed_key_value(key: &[u8]) -> i32 {
    let full = prefixed_key(
        MRC_PREFIX.lock().unwrap().as_deref().map(str::as_bytes),
        key,
    );
    adoc_delete_key_value(ADOC_GLOBAL_NAME, 0, &full)
}
/// C `startsWith` (`iihdf.c:2109`).
fn starts_with(full: &[u8], sub: &[u8]) -> i32 {
    // `strstr(full, sub) == full`: a match, and at offset zero.
    if full.starts_with(sub) { 1 } else { 0 }
}
/// C `endsWith` (`iihdf.c:2117`).
fn ends_with(full: &[u8], sub: &[u8]) -> i32 {
    // The source takes the *first* occurrence and rejects it unless it runs to
    // the end, so a later occurrence at the end still gives -1.
    if sub.is_empty() {
        // `strstr` returns `full` for an empty needle.
        return if full.is_empty() { 0 } else { -1 };
    }
    if sub.len() > full.len() {
        return -1;
    }
    let Some(sub_ind) = full.windows(sub.len()).position(|window| window == sub) else {
        return -1;
    };
    if full.len() - sub_ind != sub.len() {
        return -1;
    }
    sub_ind as i32
}
/// C `prefixedKey` (`iihdf.c:2127`).
///
/// The source writes into the shared `sStringBuf` and returns it; the caller
/// uses the result before the next call, so an owned `Vec<u8>` is the same
/// thing without the shared buffer.  Its `manageMallocBuf` failure arm, which
/// returned the unprefixed key, cannot arise for a `Vec`.
fn prefixed_key(prefix: Option<&[u8]>, key: &[u8]) -> Vec<u8> {
    let Some(prefix) = prefix else {
        return key.to_vec();
    };
    if prefix.is_empty() {
        return key.to_vec();
    }
    let mut out = Vec::with_capacity(prefix.len() + key.len());
    out.extend_from_slice(prefix);
    out.extend_from_slice(key);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_io_funcs_installs_hdf_header_sync_callback() {
        unsafe {
            let mut image = ImodImageFile::default();
            set_io_funcs_plus(&mut image, IIHDF_IMOD, 1, -1);
            assert!(image.sync_from_mrc_header.is_some());
        }
    }

    #[test]
    fn secondary_hdf_volume_cursors_alias_owned_records() {
        let mut primary = ii_new_box();
        let primary_ptr = primary.as_mut() as *mut ImodImageFile;
        primary.ii_volumes.push(primary_ptr);

        let mut secondary = ii_new_box();
        let secondary_ptr = secondary.as_mut() as *mut ImodImageFile;
        primary.owned_hdf_volumes.push(secondary);
        primary.ii_volumes.push(secondary_ptr);

        assert_eq!(primary.owned_hdf_volumes.len(), 1);
        assert_eq!(primary.ii_volumes.as_slice(), &[primary_ptr, secondary_ptr]);
        assert_eq!(
            primary.owned_hdf_volumes[0].as_ref() as *const ImodImageFile,
            secondary_ptr.cast_const()
        );
    }

    #[test]
    fn direct_hdf5_create_and_open_images_group() {
        unsafe {
            let mut path = std::env::temp_dir();
            path.push(format!("imod-rs-iihdf-{}-{}.h5", std::process::id(), 1));
            let text = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let file = H5Fcreate(text.as_ptr(), 2, 0, 0);
            assert!(file >= 0);
            let mdf = H5Gcreate2(file, c"MDF".as_ptr(), 0, 0, 0);
            assert!(mdf >= 0);
            let images = H5Gcreate2(file, c"/MDF/images".as_ptr(), 0, 0, 0);
            assert!(images >= 0);
            assert_eq!(H5Gclose(images), 0);
            assert_eq!(H5Gclose(mdf), 0);
            let images = H5Gopen2(file, c"/MDF/images".as_ptr(), 0);
            assert!(images >= 0);
            assert_eq!(H5Gclose(images), 0);
            assert_eq!(H5Fclose(file), 0);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn new_hdf_file_header_is_owned_by_the_image_record() {
        unsafe {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "imod-rs-iihdf-header-{}-{}.h5",
                std::process::id(),
                1
            ));
            let mut image = ImodImageFile::default();
            image.filename = Some(path.to_string_lossy().into_owned());
            image.fmode = "wb+".to_owned();

            assert_eq!(ii_hdf_open_new(&mut image, "w"), 0);
            assert!(image.mrc_header.is_some());
            assert_eq!(
                image.header.cast::<MrcHeader>(),
                image.mrc_header.as_deref_mut().unwrap() as *mut MrcHeader
            );

            hdf_close(&mut image);
            hdf_delete(&mut image);
            assert!(image.mrc_header.is_none());
            assert!(image.header.is_null());
            std::fs::remove_file(path).unwrap();
        }
    }
}
