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
use crate::imod::libcfshr::ilist::{
    Ilist, ilist_append, ilist_delete, ilist_item, ilist_new, ilist_size,
};
use crate::imod::libiimod::hdf_imageio::{
    hdf_read_section_any, hdf_write_section_any, init_new_hdf_file,
};
use crate::imod::libiimod::iimage::{
    IIERR_IO_ERROR, IIERR_NOT_FORMAT, IIFILE_HDF, IIFORMAT_COMPLEX, IIFORMAT_LUMINANCE,
    IIFORMAT_RGB, IISTATE_NOTINIT, IISTATE_UNUSED, IITYPE_BYTE, IITYPE_FLOAT, IITYPE_SHORT,
    IITYPE_UBYTE, IITYPE_USHORT, ImodImageFile, MRSA_BYTE, MRSA_FLOAT, MRSA_USHORT,
    ii_default_min_max_mean, ii_delete, ii_fill_mrc_header, ii_new, ii_sync_from_mrc_header,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT,
    MRC_MODE_USHORT, MrcHeader, fix_title_padding, mrc_head_new, mrc_set_scale,
};
use core::ffi::{c_char, c_void};

static mut S_STRING_BUF: *mut c_char = core::ptr::null_mut();
static mut S_STR_BUF_SIZE: i32 = 0;
static mut S_ATTRIB_NAME: *mut c_char = core::ptr::null_mut();
static mut S_ATTR_NAME_SIZE: i32 = 0;
static mut S_FLOAT_BUF: *mut f32 = core::ptr::null_mut();
static mut S_FLOAT_BUF_SIZE: i32 = 0;
static mut S_INT_BUF: *mut i32 = core::ptr::null_mut();
static mut S_INT_BUF_SIZE: i32 = 0;
static mut S_SHORT_BUF: *mut i16 = core::ptr::null_mut();
static mut S_SHORT_BUF_SIZE: i32 = 0;
static mut S_USHORT_BUF: *mut u16 = core::ptr::null_mut();
static mut S_USHORT_BUF_SIZE: i32 = 0;
static mut S_MRC_PREFIX: *mut c_char = core::ptr::null_mut();

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

#[repr(C)]
pub struct StackSetData {
    pub name: *mut c_char,
    pub dset_id: HidT,
    pub is_open: i32,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct GroupData {
    group_id: HidT,
    name: *mut c_char,
    has_valid_datasets: u8,
    has_any_datasets: u8,
    has_groups: u8,
    non_global_attrib: u8,
    added_global: i16,
    adoc_collection: i16,
    num_attributes: i32,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct DatasetData {
    dset_id: HidT,
    name: *mut c_char,
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

static mut S_GROUPS: *mut Ilist = core::ptr::null_mut();
static mut S_DATASETS: *mut Ilist = core::ptr::null_mut();
static mut S_NUMBERED_GROUPS: i32 = 1;
static mut S_MAX_GROUP_NUM: i32 = -1_000_000_000;
static mut S_MIN_GROUP_NUM: i32 = 1_000_000_000;

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
pub unsafe fn ii_test_if_hdf(filename: *const c_char) -> i32 {
    // H5Fis_hdf5 is intentionally an HDF5 ABI call.
    H5Fis_hdf5(filename)
}
/// C `iiHDFCheck` (`iihdf.c:133`).
pub unsafe fn ii_hdf_check(in_file: *mut ImodImageFile) -> i32 {
    let err = H5Fis_hdf5((*in_file).filename);
    if err < 0 {
        return IIERR_IO_ERROR;
    }
    if err == 0 {
        return IIERR_NOT_FORMAT;
    }
    if !(*in_file).fp.is_null() {
        libc::fclose((*in_file).fp);
    }
    S_GROUPS = ilist_new(core::mem::size_of::<GroupData>() as i32, 4);
    S_DATASETS = ilist_new(core::mem::size_of::<DatasetData>() as i32, 4);
    S_NUMBERED_GROUPS = 1;
    S_MAX_GROUP_NUM = -1_000_000_000;
    S_MIN_GROUP_NUM = 1_000_000_000;
    let slash = libc::strdup(c"/".as_ptr());
    if S_GROUPS.is_null() || S_DATASETS.is_null() || slash.is_null() {
        cleanup_from_open(0, 0, 0, in_file);
        return 3;
    }
    let writable = !libc::strstr((*in_file).fmode.as_ptr(), c"+".as_ptr()).is_null();
    let file_id = H5Fopen((*in_file).filename, if writable { 1 } else { 0 }, 0);
    let group_id = H5Gopen2(file_id, slash, 0);
    let mut section = 0;
    let scan_err = scan_group(group_id, slash, &mut section);
    if scan_err != 0 || ilist_size(S_DATASETS) == 0 {
        cleanup_from_open(file_id, 1, 0, in_file);
        return if scan_err != 0 {
            scan_err
        } else {
            IIERR_NOT_FORMAT
        };
    }
    let mut single_image_stack = true;
    let first = ilist_item(S_DATASETS, 0).cast::<DatasetData>();
    let nx_stack = (*first).nx;
    let ny_stack = (*first).ny;
    let stack_type = (*first).data_type;
    (*in_file).num_volumes = 1;
    for set in 0..ilist_size(S_DATASETS) {
        let dataset = ilist_item(S_DATASETS, set).cast::<DatasetData>();
        if (set != 0
            && (nx_stack != (*dataset).nx
                || ny_stack != (*dataset).ny
                || stack_type != (*dataset).data_type))
            || (*dataset).nz > 1
        {
            single_image_stack = false;
            (*in_file).num_volumes = ilist_size(S_DATASETS);
            break;
        }
    }
    if single_image_stack
        && S_NUMBERED_GROUPS != 0
        && (S_MIN_GROUP_NUM < 0 || S_MAX_GROUP_NUM + 1 - S_MIN_GROUP_NUM < ilist_size(S_DATASETS))
    {
        S_NUMBERED_GROUPS = 0;
    }
    (*in_file).ii_volumes = libc::malloc(
        ((*in_file).num_volumes as usize) * core::mem::size_of::<*mut ImodImageFile>(),
    )
    .cast();
    if (*in_file).ii_volumes.is_null() {
        cleanup_from_open(file_id, 1, 0, in_file);
        return 3;
    }
    *(*in_file).ii_volumes = in_file;
    for ind in 1..(*in_file).num_volumes {
        *(*in_file).ii_volumes.add(ind as usize) = ii_new();
        if (*(*in_file).ii_volumes.add(ind as usize)).is_null() {
            cleanup_from_open(file_id, 1, ind, in_file);
            return 3;
        }
    }
    let mut hdf_source = IIHDF_UNKNOWN;
    let mut eman_sect_type = core::ptr::null();
    for ind in 0..ilist_size(S_GROUPS) {
        let group = ilist_item(S_GROUPS, ind).cast::<GroupData>();
        if libc::strcmp((*group).name, c"/Chimera".as_ptr()) == 0 {
            hdf_source = IIHDF_CHIMERA;
        }
        if (*group).num_attributes == 0 || (*group).adoc_collection > 0 {
            continue;
        }
        for set in 0..ilist_size(S_DATASETS) {
            let dataset = ilist_item(S_DATASETS, set).cast::<DatasetData>();
            if starts_with((*dataset).name, (*group).name) == 0 {
                (*group).non_global_attrib = 1;
                break;
            }
        }
    }
    if single_image_stack {
        (*in_file).nx = nx_stack;
        (*in_file).ny = ny_stack;
        (*in_file).nz = ilist_size(S_DATASETS);
        (*in_file).type_ = stack_type as i32;
        (*in_file).stack_set_list =
            ilist_new(core::mem::size_of::<StackSetData>() as i32, (*in_file).nz).cast();
        if (*in_file).stack_set_list.is_null() {
            cleanup_from_open(file_id, 1, 1, in_file);
            return 3;
        }
        for set in 0..(*in_file).nz {
            let dataset = ilist_item(S_DATASETS, set).cast::<DatasetData>();
            let stack_set = StackSetData {
                name: (*dataset).name,
                dset_id: (*dataset).dset_id,
                is_open: 1,
            };
            if ilist_append(
                (*in_file).stack_set_list.cast(),
                (&raw const stack_set).cast_mut().cast(),
            ) != 0
            {
                cleanup_from_open(file_id, 1, 1, in_file);
                return 3;
            }
            (*dataset).name = core::ptr::null_mut();
        }
    } else {
        for set in 0..ilist_size(S_DATASETS) {
            let dataset = ilist_item(S_DATASETS, set).cast::<DatasetData>();
            let volume = *(*in_file).ii_volumes.add(set as usize);
            (*volume).nx = (*dataset).nx;
            (*volume).ny = (*dataset).ny;
            (*volume).nz = (*dataset).nz;
            (*volume).type_ = (*dataset).data_type as i32;
            (*volume).dataset_name = (*dataset).name;
            (*dataset).name = core::ptr::null_mut();
            (*volume).dataset_id = (*dataset).dset_id;
            (*volume).dataset_is_open = 1;
            (*volume).num_volumes = (*in_file).num_volumes;
        }
    }
    let mut retval = 0;
    for ind in 0..(*in_file).num_volumes {
        let volume = *(*in_file).ii_volumes.add(ind as usize);
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
            (*volume).header = libc::malloc(core::mem::size_of::<MrcHeader>()).cast();
        }
        if retval != 0 || (*volume).header.is_null() {
            cleanup_from_open(file_id, 1, (*in_file).num_volumes, in_file);
            return IIERR_MEMORY_ERR;
        }
    }
    if single_image_stack
        && S_NUMBERED_GROUPS != 0
        && S_MIN_GROUP_NUM >= 0
        && S_MAX_GROUP_NUM + 1 - S_MIN_GROUP_NUM >= (*in_file).nz
    {
        if H5Aexists_by_name(file_id, c"/MDF/images".as_ptr(), c"imageid_max".as_ptr(), 0) <= 0 {
            S_NUMBERED_GROUPS = 0;
        }
        if S_NUMBERED_GROUPS != 0 {
            if setup_zto_set_map(file_id, in_file, S_MAX_GROUP_NUM + 1, 0) != 0 {
                return IIERR_MEMORY_ERR;
            }
            for set in 0..(*in_file).nz {
                if S_NUMBERED_GROUPS == 0 {
                    break;
                }
                let dataset = ilist_item(S_DATASETS, set).cast::<DatasetData>();
                if *(*in_file)
                    .z_to_data_set_map
                    .add((*dataset).group_num as usize)
                    >= 0
                {
                    S_NUMBERED_GROUPS = 0;
                } else {
                    *(*in_file)
                        .z_to_data_set_map
                        .add((*dataset).group_num as usize) = set;
                }
            }
        }
    }
    if single_image_stack && S_NUMBERED_GROUPS == 0 {
        libc::free((*in_file).z_to_data_set_map.cast());
        (*in_file).z_to_data_set_map = core::ptr::null_mut();
        if setup_zto_set_map(file_id, in_file, (*in_file).nz, 1) != 0 {
            return IIERR_MEMORY_ERR;
        }
    }
    for set in 0..ilist_size(S_DATASETS) {
        let vol_ind = if single_image_stack { 0 } else { set };
        let volume = *(*in_file).ii_volumes.add(vol_ind as usize);
        let adoc_index = (*volume).adoc_index;
        adoc_set_current(adoc_index);
        let dataset = ilist_item(S_DATASETS, set).cast::<DatasetData>();
        let mut sect_text = [0 as c_char; 32];
        libc::snprintf(
            sect_text.as_mut_ptr(),
            sect_text.len(),
            c"%d".as_ptr(),
            if S_NUMBERED_GROUPS != 0 {
                (*dataset).group_num
            } else {
                set
            },
        );
        let mut added_sect_ind = -1;
        let mut sect_ind = 0;
        let mut collection = ADOC_GLOBAL_NAME.as_ptr();
        let dataset_name = if single_image_stack {
            let stack_set =
                ilist_item((*in_file).stack_set_list.cast(), set).cast::<StackSetData>();
            (*stack_set).name
        } else {
            (*(*(*in_file).ii_volumes.add(set as usize))).dataset_name
        };
        if (*dataset).num_attributes != 0 {
            let dataset_id = H5Dopen2(file_id, dataset_name, 0);
            if single_image_stack {
                sect_ind = adoc_add_section(ADOC_ZVALUE_NAME.as_ptr(), sect_text.as_ptr());
                if sect_ind < 0 {
                    retval = IIERR_MEMORY_ERR;
                }
                added_sect_ind = sect_ind;
                collection = ADOC_ZVALUE_NAME.as_ptr();
            }
            if retval == 0 {
                retval =
                    attributes_to_adoc(dataset_id, (*dataset).num_attributes, collection, sect_ind);
            }
            H5Dclose(dataset_id);
        }
        for ind in 0..ilist_size(S_GROUPS) {
            if retval != 0 {
                break;
            }
            let group = ilist_item(S_GROUPS, ind).cast::<GroupData>();
            if (*group).num_attributes == 0
                || (*group).adoc_collection > 0
                || (*group).added_global != 0
            {
                continue;
            }
            let group_id = H5Gopen2(file_id, (*group).name, 0);
            if starts_with(dataset_name, (*group).name) != 0 {
                let mut group_adoc_index =
                    (*(*(*in_file).ii_volumes.add(vol_ind as usize))).adoc_index;
                let mut group_sect_ind = 0;
                let mut group_collection = ADOC_GLOBAL_NAME.as_ptr();
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
                        added_sect_ind =
                            adoc_add_section(ADOC_ZVALUE_NAME.as_ptr(), sect_text.as_ptr());
                        if added_sect_ind < 0 {
                            retval = IIERR_MEMORY_ERR;
                        }
                    }
                    group_sect_ind = added_sect_ind;
                    group_collection = ADOC_ZVALUE_NAME.as_ptr();
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
            cleanup_from_open(file_id, 1, (*in_file).num_volumes, in_file);
            return retval;
        }
    }
    retval = 0;
    for grp in 0..ilist_size(S_GROUPS) {
        let group = ilist_item(S_GROUPS, grp).cast::<GroupData>();
        if (*group).adoc_collection <= 0 {
            continue;
        }
        let collection = libc::strrchr((*group).name, b'/' as i32).add(1);
        for sub in 0..ilist_size(S_GROUPS) {
            if retval != 0 {
                break;
            }
            let sub_group = ilist_item(S_GROUPS, sub).cast::<GroupData>();
            if sub == grp
                || (*sub_group).num_attributes == 0
                || starts_with((*sub_group).name, (*group).name) == 0
            {
                continue;
            }
            let section_name = libc::strrchr((*sub_group).name, b'/' as i32).add(1);
            let group_id = H5Gopen2(file_id, (*sub_group).name, 0);
            adoc_set_current(if single_image_stack {
                (*in_file).adoc_index
            } else {
                (*in_file).global_adoc_index
            });
            let group_sect_ind = adoc_add_section(collection, section_name);
            if group_sect_ind < 0 {
                retval = IIERR_MEMORY_ERR;
            }
            if retval == 0 {
                retval = attributes_to_adoc(
                    group_id,
                    (*sub_group).num_attributes,
                    collection,
                    group_sect_ind,
                );
            }
            H5Gclose(group_id);
        }
        if retval != 0 {
            cleanup_from_open(file_id, 1, (*in_file).num_volumes, in_file);
            return retval;
        }
    }
    if hdf_source != IIHDF_CHIMERA {
        adoc_set_current((*in_file).adoc_index);
        let num_keys = adoc_get_number_of_keys(ADOC_GLOBAL_NAME.as_ptr(), 0);
        let mrc_tags = [
            c"MRC.mx".as_ptr(),
            c"MRC.my".as_ptr(),
            c"MRC.mz".as_ptr(),
            c"MRC.xlen".as_ptr(),
            c"MRC.ylen".as_ptr(),
            c"MRC.zlen".as_ptr(),
        ];
        let mut num_match = 0;
        retval = 0;
        for key_ind in 0..num_keys {
            if retval != 0 {
                break;
            }
            let mut key = core::ptr::null_mut();
            retval = adoc_get_key_by_index(ADOC_GLOBAL_NAME.as_ptr(), 0, key_ind, &mut key);
            if retval == 0 && starts_with(key, c"EMAN.".as_ptr()) == 0 {
                for tag in mrc_tags {
                    let sub = ends_with(key, tag);
                    if sub >= 0 {
                        retval = manage_malloc_buf(
                            (&raw mut S_STRING_BUF).cast(),
                            &raw mut S_STR_BUF_SIZE,
                            sub + 1,
                            1,
                        );
                        if retval == 0 {
                            if sub != 0 {
                                libc::strncpy(S_STRING_BUF, key, sub as usize);
                            }
                            *S_STRING_BUF.add(sub as usize) = 0;
                            if num_match == 0 {
                                S_MRC_PREFIX = libc::strdup(S_STRING_BUF);
                                if S_MRC_PREFIX.is_null() {
                                    retval = 1;
                                }
                                num_match = 1;
                            } else if libc::strcmp(S_MRC_PREFIX, S_STRING_BUF) == 0 {
                                num_match += 1;
                            }
                        }
                        break;
                    }
                }
            }
            libc::free(key.cast());
        }
        if retval != 0 {
            cleanup_from_open(file_id, 1, (*in_file).num_volumes, in_file);
            return IIERR_MEMORY_ERR;
        }
        if num_match == mrc_tags.len() as i32 {
            hdf_source = IIHDF_OTHER_MRC;
            if libc::strcmp(S_MRC_PREFIX, c"IMOD.".as_ptr()) == 0 {
                hdf_source = IIHDF_IMOD;
            }
        } else {
            let mut value = 0;
            if adoc_get_integer(
                ADOC_GLOBAL_NAME.as_ptr(),
                0,
                c"EMAN.nx".as_ptr(),
                &mut value,
            ) == 0
            {
                eman_sect_type = ADOC_GLOBAL_NAME.as_ptr();
            } else if adoc_get_integer(
                ADOC_ZVALUE_NAME.as_ptr(),
                0,
                c"EMAN.nx".as_ptr(),
                &mut value,
            ) == 0
            {
                eman_sect_type = ADOC_ZVALUE_NAME.as_ptr();
            }
            if !eman_sect_type.is_null() {
                hdf_source = IIHDF_EMAN;
            }
        }
    }
    for vol_ind in 0..(*in_file).num_volumes {
        let volume = *(*in_file).ii_volumes.add(vol_ind as usize);
        (*volume).ii_volumes = (*in_file).ii_volumes;
        (*volume).format = IIFORMAT_LUMINANCE;
        let mode = if (*volume).type_ == IITYPE_BYTE || (*volume).type_ == IITYPE_UBYTE {
            if (*volume).type_ == IITYPE_UBYTE {
                let mut rgb = 0;
                if hdf_source == IIHDF_IMOD
                    && get_prefixed_integer(c"is_rgb".as_ptr(), &mut rgb, &mut retval) == 0
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
                && get_prefixed_integer(c"is_complex".as_ptr(), &mut complex, &mut retval) == 0
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
        let dataset = ilist_item(S_DATASETS, vol_ind).cast::<DatasetData>();
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
            get_del_prefixed_integer(c"MRC.nxstart".as_ptr(), &mut (*hdata).nxstart, &mut retval);
            get_del_prefixed_integer(c"MRC.nystart".as_ptr(), &mut (*hdata).nystart, &mut retval);
            get_del_prefixed_integer(c"MRC.nzstart".as_ptr(), &mut (*hdata).nzstart, &mut retval);
            get_del_prefixed_integer(c"MRC.mx".as_ptr(), &mut (*hdata).mx, &mut retval);
            get_del_prefixed_integer(c"MRC.my".as_ptr(), &mut (*hdata).my, &mut retval);
            get_del_prefixed_integer(c"MRC.mz".as_ptr(), &mut (*hdata).mz, &mut retval);
            get_del_prefixed_float(c"MRC.xlen".as_ptr(), &mut (*hdata).xlen, &mut retval);
            get_del_prefixed_float(c"MRC.ylen".as_ptr(), &mut (*hdata).ylen, &mut retval);
            get_del_prefixed_float(c"MRC.zlen".as_ptr(), &mut (*hdata).zlen, &mut retval);
            get_del_prefixed_float(c"MRC.alpha".as_ptr(), &mut (*hdata).alpha, &mut retval);
            get_del_prefixed_float(c"MRC.beta".as_ptr(), &mut (*hdata).beta, &mut retval);
            get_del_prefixed_float(c"MRC.gamma".as_ptr(), &mut (*hdata).gamma, &mut retval);
            get_del_prefixed_integer(c"MRC.mapc".as_ptr(), &mut (*hdata).mapc, &mut retval);
            get_del_prefixed_integer(c"MRC.mapr".as_ptr(), &mut (*hdata).mapr, &mut retval);
            get_del_prefixed_integer(c"MRC.maps".as_ptr(), &mut (*hdata).maps, &mut retval);
            get_del_prefixed_float(c"MRC.minimum".as_ptr(), &mut (*hdata).amin, &mut retval);
            get_del_prefixed_float(c"MRC.maximum".as_ptr(), &mut (*hdata).amax, &mut retval);
            get_del_prefixed_float(c"MRC.mean".as_ptr(), &mut (*hdata).amean, &mut retval);
            get_del_prefixed_integer(c"MRC.ispg".as_ptr(), &mut (*hdata).ispg, &mut retval);
            get_del_prefixed_float(c"MRC.xorigin".as_ptr(), &mut (*hdata).xorg, &mut retval);
            get_del_prefixed_float(c"MRC.yorigin".as_ptr(), &mut (*hdata).yorg, &mut retval);
            get_del_prefixed_float(c"MRC.zorigin".as_ptr(), &mut (*hdata).zorg, &mut retval);
            get_del_prefixed_float(c"MRC.rms".as_ptr(), &mut (*hdata).rms, &mut retval);
            get_del_prefixed_integer(c"MRC.nlabels".as_ptr(), &mut (*hdata).nlabl, &mut retval);
            (*hdata).nlabl = (*hdata).nlabl.clamp(0, 10);
            let mut nsum = 6;
            if adoc_get_float_array(
                ADOC_GLOBAL_NAME.as_ptr(),
                0,
                prefixed_key(S_MRC_PREFIX, c"MRC.tiltangles".as_ptr()),
                (*hdata).tiltangles.as_mut_ptr(),
                &mut nsum,
                6,
            ) < 0
            {
                retval += 1;
            }
            delete_prefixed_key_value(c"MRC.tiltangles".as_ptr());
            for sub in 0..(*hdata).nlabl {
                if retval != 0 {
                    break;
                }
                let mut label_key = [0 as c_char; 14];
                libc::snprintf(
                    label_key.as_mut_ptr(),
                    label_key.len(),
                    c"MRC.label%d".as_ptr(),
                    sub,
                );
                let mut label = core::ptr::null_mut();
                if adoc_get_string(
                    ADOC_GLOBAL_NAME.as_ptr(),
                    0,
                    prefixed_key(S_MRC_PREFIX, label_key.as_ptr()),
                    &mut label,
                ) != 0
                {
                    retval += 1;
                } else {
                    libc::strncpy((*hdata).labels[sub as usize].as_mut_ptr().cast(), label, 80);
                    (*hdata).labels[sub as usize][80] = 0;
                    fix_title_padding(&mut (*hdata).labels[sub as usize]);
                    libc::free(label.cast());
                    delete_prefixed_key_value(label_key.as_ptr());
                }
            }
            delete_prefixed_key_value(c"MRC.nx".as_ptr());
            delete_prefixed_key_value(c"MRC.ny".as_ptr());
            delete_prefixed_key_value(c"MRC.nz".as_ptr());
            delete_prefixed_key_value(c"MRC.mode".as_ptr());
            if single_image_stack {
                (*in_file).has_piece_coords = 1;
                for sub in 0..(*in_file).nz {
                    let mut x = 0;
                    let mut y = 0;
                    let mut z = 0;
                    if adoc_get_three_integers(
                        ADOC_ZVALUE_NAME.as_ptr(),
                        sub,
                        c"PieceCoordinates".as_ptr(),
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
                ADOC_GLOBAL_NAME.as_ptr(),
                0,
                c"origin".as_ptr(),
                &mut (*hdata).xorg,
                &mut (*hdata).yorg,
                &mut (*hdata).zorg,
            ) < 0
            {
                retval += 1;
            }
            if adoc_get_three_floats(
                ADOC_GLOBAL_NAME.as_ptr(),
                0,
                c"step".as_ptr(),
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
            if adoc_get_float(eman_sect_type, 0, c"EMAN.apix_x".as_ptr(), &mut xscale) < 0
                || adoc_get_float(eman_sect_type, 0, c"EMAN.apix_y".as_ptr(), &mut yscale) < 0
                || adoc_get_float(eman_sect_type, 0, c"EMAN.apix_z".as_ptr(), &mut zscale) < 0
            {
                retval += 1;
            }
            mrc_set_scale(&mut *hdata, xscale as f64, yscale as f64, zscale as f64);
            if eman_sect_type == ADOC_GLOBAL_NAME.as_ptr() {
                if adoc_get_float(
                    eman_sect_type,
                    0,
                    c"EMAN.origin_x".as_ptr(),
                    &mut (*hdata).xorg,
                ) < 0
                    || adoc_get_float(
                        eman_sect_type,
                        0,
                        c"EMAN.origin_y".as_ptr(),
                        &mut (*hdata).yorg,
                    ) < 0
                    || adoc_get_float(
                        eman_sect_type,
                        0,
                        c"EMAN.origin_z".as_ptr(),
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
                    if adoc_get_float(
                        ADOC_ZVALUE_NAME.as_ptr(),
                        section,
                        c"EMAN.minimum".as_ptr(),
                        &mut value,
                    ) == 0
                    {
                        (*hdata).amin = (*hdata).amin.min(value);
                    }
                    if adoc_get_float(
                        ADOC_ZVALUE_NAME.as_ptr(),
                        section,
                        c"EMAN.maximum".as_ptr(),
                        &mut value,
                    ) == 0
                    {
                        (*hdata).amax = (*hdata).amax.max(value);
                    }
                    if adoc_get_float(
                        ADOC_ZVALUE_NAME.as_ptr(),
                        section,
                        c"EMAN.mean".as_ptr(),
                        &mut value,
                    ) == 0
                    {
                        (*hdata).amean += value;
                        num_means += 1;
                    }
                    if origin_matches {
                        if adoc_get_float(
                            ADOC_ZVALUE_NAME.as_ptr(),
                            section,
                            c"EMAN.origin_x".as_ptr(),
                            &mut value,
                        ) != 0
                            || (section != 0 && value != section_x_origin)
                        {
                            origin_matches = false;
                        } else if section == 0 {
                            section_x_origin = value;
                        }
                        if adoc_get_float(
                            ADOC_ZVALUE_NAME.as_ptr(),
                            section,
                            c"EMAN.origin_y".as_ptr(),
                            &mut value,
                        ) != 0
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
                eman_sect_type,
                0,
                c"EMAN.minimum".as_ptr(),
                &mut (*hdata).amin,
            ) < 0
                || adoc_get_float(
                    eman_sect_type,
                    0,
                    c"EMAN.maximum".as_ptr(),
                    &mut (*hdata).amax,
                ) < 0
                || adoc_get_float(
                    eman_sect_type,
                    0,
                    c"EMAN.mean".as_ptr(),
                    &mut (*hdata).amean,
                ) < 0
            {
                retval += 1;
            }
        }
        (*hdata).xorg *= -1.0;
        (*hdata).yorg *= -1.0;
        (*hdata).zorg *= -1.0;
        if (*in_file).stack_set_list.is_null() {
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
            (*volume).filename = libc::malloc(libc::strlen((*in_file).filename) + 10).cast();
            if !(*volume).filename.is_null() {
                libc::sprintf(
                    (*volume).filename,
                    c"%s-%d".as_ptr(),
                    (*in_file).filename,
                    vol_ind + 1,
                );
            } else {
                retval = 1;
            }
        }
        if retval != 0 {
            cleanup_from_open(file_id, 1, (*in_file).num_volumes, in_file);
            return IIERR_MEMORY_ERR;
        }
    }
    for vol_ind in 1..(*in_file).num_volumes {
        let volume = *(*in_file).ii_volumes.add(vol_ind as usize);
        H5Dclose((*volume).dataset_id);
        (*volume).dataset_is_open = 0;
        (*volume).state = 3;
        (*volume).fp = core::ptr::null_mut();
        core::ptr::copy_nonoverlapping((*in_file).fmode.as_ptr(), (*volume).fmode.as_mut_ptr(), 3);
    }
    cleanup_from_open(file_id, 0, 0, in_file);
    0
}
/// C `scanGroup` (`iihdf.c:736`).
unsafe fn scan_group(group_id: HidT, group_name: *mut c_char, adoc_section: *mut i32) -> i32 {
    *adoc_section = 0;
    let group = GroupData {
        group_id,
        name: group_name,
        has_valid_datasets: 0,
        has_any_datasets: 0,
        has_groups: 0,
        non_global_attrib: 0,
        added_global: 0,
        adoc_collection: 0,
        num_attributes: 0,
    };
    if group.name.is_null() {
        return 3;
    }
    let slash = libc::strrchr(group_name, b'/' as i32);
    let mut group_number = 0;
    let mut numbered = false;
    if !slash.is_null() {
        let mut end = core::ptr::null_mut();
        group_number = libc::strtol(slash.add(1), &mut end, 10) as i32;
        numbered = !end.is_null() && *end == 0;
    }
    let list_index = ilist_size(S_GROUPS);
    if ilist_append(S_GROUPS, (&raw const group).cast_mut().cast()) != 0 {
        return 3;
    }
    let group_ptr = ilist_item(S_GROUPS, list_index).cast::<GroupData>();
    let mut object_info = core::mem::zeroed::<H5OInfo>();
    if H5Oget_info(group_id, &mut object_info) < 0 {
        return IIERR_IO_ERROR;
    }
    (*group_ptr).num_attributes = object_info.num_attrs as i32;
    let mut group_info = core::mem::zeroed::<H5GInfo>();
    if H5Gget_info(group_id, &mut group_info) < 0 {
        return IIERR_IO_ERROR;
    }
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
        let mut object_name = libc::malloc(size as usize).cast::<c_char>();
        if object_name.is_null() {
            return 3;
        }
        if H5Lget_name_by_idx(
            group_id,
            c".".as_ptr(),
            H5_INDEX_NAME,
            H5_ITER_INC,
            ind,
            object_name,
            size as usize,
            0,
        ) < 0
        {
            libc::free(object_name.cast());
            return IIERR_IO_ERROR;
        }
        if *object_name != b'/' as c_char {
            let relative_name = object_name;
            let group_length = libc::strlen(group_name);
            object_name = libc::malloc(size as usize + group_length + 1).cast();
            if object_name.is_null() {
                libc::free(relative_name.cast());
                return IIERR_MEMORY_ERR;
            }
            libc::sprintf(
                object_name,
                c"%s/%s".as_ptr(),
                if group_length > 1 {
                    group_name
                } else {
                    c"".as_ptr().cast_mut()
                },
                relative_name,
            );
            libc::free(relative_name.cast());
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
            libc::free(object_name.cast());
            return IIERR_IO_ERROR;
        }
        if object_info.type_ == H5O_TYPE_GROUP {
            (*group_ptr).has_groups = 1;
            let subgroup_id = H5Gopen2(group_id, object_name, 0);
            if subgroup_id < 0 {
                libc::free(object_name.cast());
                return IIERR_IO_ERROR;
            }
            let mut subsection = 0;
            let err = scan_group(subgroup_id, object_name, &mut subsection);
            if err != 0 {
                H5Gclose(subgroup_id);
                return err;
            }
            let parent = ilist_item(S_GROUPS, list_index).cast::<GroupData>();
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
                S_NUMBERED_GROUPS = 0;
            }
            let dataset_id = H5Dopen2(group_id, object_name, 0);
            if dataset_id < 0 {
                libc::free(object_name.cast());
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
                    libc::free(object_name.cast());
                    return IIERR_IO_ERROR;
                }
                let mut data = DatasetData {
                    dset_id: dataset_id,
                    name: object_name,
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
                    S_NUMBERED_GROUPS = 0;
                }
                if S_NUMBERED_GROUPS != 0 {
                    S_MIN_GROUP_NUM = S_MIN_GROUP_NUM.min(group_number);
                    S_MAX_GROUP_NUM = S_MAX_GROUP_NUM.max(group_number);
                }
                if ilist_append(S_DATASETS, (&raw const data).cast_mut().cast()) != 0 {
                    H5Sclose(space_id);
                    H5Tclose(type_id);
                    H5Dclose(dataset_id);
                    libc::free(object_name.cast());
                    return 3;
                }
                H5Sclose(space_id);
                H5Tclose(type_id);
                continue;
            }
            H5Sclose(space_id);
            H5Tclose(type_id);
            H5Dclose(dataset_id);
        }
        libc::free(object_name.cast());
    }
    H5Gclose(group_id);
    let final_group = ilist_item(S_GROUPS, list_index).cast::<GroupData>();
    if (*final_group).has_any_datasets != 0 || (*final_group).has_groups != 0 {
        *adoc_section = -1;
    } else if (*final_group).num_attributes != 0 {
        *adoc_section = 1;
    }
    0
}
/// C `iiHDFopenNew` (`iihdf.c:979`).
pub unsafe fn ii_hdf_open_new(in_file: *mut ImodImageFile, mode: *const c_char) -> i32 {
    if (*in_file).file != 0 {
        if (*in_file).file != IIFILE_HDF
            || (*in_file).hdf_source != IIHDF_IMOD
            || (!(*in_file).stack_set_list.is_null() && (*in_file).nz > 1)
            || (*in_file).write_section.is_none()
        {
            return 1;
        }
        let volumes = libc::malloc(
            ((*in_file).num_volumes + 1) as usize * core::mem::size_of::<*mut ImodImageFile>(),
        )
        .cast::<*mut ImodImageFile>();
        let volume = ii_new();
        if volumes.is_null() || (*in_file).ii_volumes.is_null() || volume.is_null() {
            libc::free(volumes.cast());
            return 1;
        }
        for ind in 0..(*in_file).num_volumes {
            *volumes.add(ind as usize) = *(*in_file).ii_volumes.add(ind as usize);
        }
        *volumes.add((*in_file).num_volumes as usize) = volume;
        (*in_file).num_volumes += 1;
        for ind in 0..(*in_file).num_volumes {
            let item = *volumes.add(ind as usize);
            (*item).ii_volumes = volumes;
            (*item).num_volumes = (*in_file).num_volumes;
        }
        (*volume).adoc_index = adoc_new();
        (*volume).global_adoc_index = (*in_file).global_adoc_index;
        (*volume).header = libc::malloc(core::mem::size_of::<MrcHeader>()).cast();
        core::ptr::copy_nonoverlapping((*in_file).fmode.as_ptr(), (*volume).fmode.as_mut_ptr(), 3);
        if (*volume).header.is_null() || (*volume).adoc_index < 0 {
            return 1;
        }
        set_io_funcs_plus(volume, IIHDF_IMOD, 1, (*in_file).hdf_file_id);
        mrc_head_new(&mut *(*in_file).header.cast::<MrcHeader>(), 1, 1, 1, 0);
        (*(*in_file).header.cast::<MrcHeader>()).packed4bits = 0;
        (*(*in_file).header.cast::<MrcHeader>()).half_floats = 0;
        (*volume).z_chunk_size = 1;
        return 0;
    }
    (*in_file).adoc_index = adoc_new();
    (*in_file).header = libc::malloc(core::mem::size_of::<MrcHeader>()).cast();
    (*in_file).ii_volumes = libc::malloc(core::mem::size_of::<*mut ImodImageFile>()).cast();
    let mut error =
        (*in_file).header.is_null() || (*in_file).adoc_index < 0 || (*in_file).ii_volumes.is_null();
    let mut file_id = -1;
    if !error {
        file_id = H5Fcreate((*in_file).filename, 2, 0, 0);
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
        libc::free((*in_file).header.cast());
        if (*in_file).adoc_index >= 0 {
            adoc_clear((*in_file).adoc_index);
        }
        libc::free((*in_file).ii_volumes.cast());
        return 1;
    }
    *(*in_file).ii_volumes = in_file;
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
    if (*in_file).stack_set_list.is_null() {
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
    let new_map = libc::malloc(new_size as usize * core::mem::size_of::<i32>()).cast::<i32>();
    let adoc_secs =
        libc::malloc((*in_file).nz as usize * core::mem::size_of::<i32>()).cast::<i32>();
    if new_map.is_null() || adoc_secs.is_null() {
        free_reorder_arrays(
            new_map,
            adoc_secs,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        return 1;
    }
    for iz in 0..new_size {
        *new_map.add(iz as usize) = -1;
    }
    let mut ord_ind = 0;
    let mut max_len = 0;
    for map_ind in 0..(*in_file).z_map_size {
        let ds_ind = *(*in_file).z_to_data_set_map.add(map_ind as usize);
        if ds_ind < 0 {
            continue;
        }
        let stack = ilist_item((*in_file).stack_set_list.cast(), ds_ind).cast::<StackSetData>();
        max_len = max_len.max(libc::strlen((*stack).name) as i32);
        let new_z = *sect_order.add(ord_ind as usize);
        if *new_map.add(new_z as usize) >= 0 {
            free_reorder_arrays(
                new_map,
                adoc_secs,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            );
            return 1;
        }
        *new_map.add(new_z as usize) = ds_ind;
        ord_ind += 1;
    }
    let temp_name = libc::malloc((max_len + 64) as usize).cast::<c_char>();
    let new_name = libc::malloc((max_len + 64) as usize).cast::<c_char>();
    if temp_name.is_null() || new_name.is_null() {
        free_reorder_arrays(new_map, adoc_secs, temp_name, new_name);
        return 1;
    }
    let file_id = (*in_file).hdf_file_id;
    ord_ind = 0;
    for map_ind in 0..(*in_file).z_map_size {
        let ds_ind = *(*in_file).z_to_data_set_map.add(map_ind as usize);
        if ds_ind < 0 {
            continue;
        }
        let new_z = *sect_order.add(ord_ind as usize);
        let stack = ilist_item((*in_file).stack_set_list.cast(), ds_ind).cast::<StackSetData>();
        libc::strcpy(temp_name, (*stack).name);
        *temp_name.add(libc::strlen(temp_name) as usize - 6) = 0;
        libc::sprintf(new_name, c"/MDF/images/Reordered%d".as_ptr(), new_z);
        if H5Lmove(file_id, temp_name, file_id, new_name, 0, 0) < 0 {
            free_reorder_arrays(new_map, adoc_secs, temp_name, new_name);
            return 1;
        }
        *adoc_secs.add(ord_ind as usize) =
            adoc_lookup_by_name_value(ADOC_ZVALUE_NAME.as_ptr(), map_ind);
        ord_ind += 1;
    }
    ord_ind = 0;
    for map_ind in 0..(*in_file).z_map_size {
        let ds_ind = *(*in_file).z_to_data_set_map.add(map_ind as usize);
        if ds_ind < 0 {
            continue;
        }
        let new_z = *sect_order.add(ord_ind as usize);
        let stack = ilist_item((*in_file).stack_set_list.cast(), ds_ind).cast::<StackSetData>();
        libc::sprintf(temp_name, c"/MDF/images/Reordered%d".as_ptr(), new_z);
        libc::sprintf(new_name, c"/MDF/images/%d".as_ptr(), new_z);
        if H5Lmove(file_id, temp_name, file_id, new_name, 0, 0) < 0 {
            free_reorder_arrays(new_map, adoc_secs, temp_name, new_name);
            return 1;
        }
        libc::strcat(new_name, c"/image".as_ptr());
        libc::free((*stack).name.cast());
        (*stack).name = libc::strdup(new_name);
        if *adoc_secs.add(ord_ind as usize) >= 0 {
            libc::sprintf(new_name, c"%d".as_ptr(), new_z);
            if adoc_change_section_name(
                ADOC_ZVALUE_NAME.as_ptr(),
                *adoc_secs.add(ord_ind as usize),
                new_name,
            ) != 0
            {
                free_reorder_arrays(new_map, adoc_secs, temp_name, new_name);
                return 1;
            }
        }
        ord_ind += 1;
    }
    libc::free((*in_file).z_to_data_set_map.cast());
    libc::free(adoc_secs.cast());
    libc::free(temp_name.cast());
    libc::free(new_name.cast());
    (*in_file).z_to_data_set_map = new_map;
    (*in_file).z_map_size = new_size;
    0
}
/// C `freeReorderArrays` (`iihdf.c:1221`).
unsafe fn free_reorder_arrays(
    new_map: *mut i32,
    adoc_secs: *mut i32,
    temp_name: *mut c_char,
    new_name: *mut c_char,
) {
    libc::free(adoc_secs.cast());
    libc::free(new_map.cast());
    libc::free(temp_name.cast());
    libc::free(new_name.cast());
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
    if (*in_file).stack_set_list.is_null()
        && (*in_file).dataset_name.is_null()
        && init_new_hdf_file(in_file) != 0
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
    libc::free(S_MRC_PREFIX.cast());
    S_MRC_PREFIX = libc::strdup(c"IMOD.MRC.".as_ptr());
    if S_MRC_PREFIX.is_null() {
        cleanup_malloc_bufs();
        return 1;
    }
    let mut error = 0;
    let mut group_id = H5Gopen2((*in_file).hdf_file_id, c"/MDF/images".as_ptr(), 0);
    if group_id >= 0 {
        let mut image_id = (*in_file).num_volumes - 1;
        if !(*in_file).stack_set_list.is_null() {
            image_id = (*in_file).nz - 1;
            for ind in (0..(*in_file).z_map_size).rev() {
                if *(*in_file).z_to_data_set_map.add(ind as usize) >= 0 {
                    image_id = image_id.max(*(*in_file).z_to_data_set_map.add(ind as usize));
                }
            }
        }
        if add_integer_attribute(group_id, c"imageid_max".as_ptr(), &mut image_id, 1) != 0 {
            error += 1;
        }
        if add_string_attribute(group_id, c"DISPLAY_ORIGIN".as_ptr(), c"LL".as_ptr()) != 0 {
            error += 1;
        }
    }
    if error == 0 && group_id >= 0 && (*in_file).stack_set_list.is_null() {
        H5Gclose(group_id);
        group_id = open_dataset_group(in_file, (*in_file).dataset_name);
    }
    if error != 0 || group_id < 0 {
        cleanup_malloc_bufs();
        if group_id >= 0 {
            H5Gclose(group_id);
        }
        return 1;
    }
    let mut value = if (*hdata).mode == MRC_MODE_RGB { 1 } else { 0 };
    if add_integer_attribute(group_id, c"IMOD.is_rgb".as_ptr(), &mut value, 1) != 0 {
        error += 1;
    }
    value = if (*in_file).format == IIFORMAT_COMPLEX {
        1
    } else {
        0
    };
    if add_integer_attribute(group_id, c"IMOD.is_complex".as_ptr(), &mut value, 1) != 0 {
        error += 1;
    }
    if error == 0 {
        add_one_prefixed_integer(group_id, c"nxstart".as_ptr(), (*hdata).nxstart, &mut error);
        add_one_prefixed_integer(group_id, c"nystart".as_ptr(), (*hdata).nystart, &mut error);
        add_one_prefixed_integer(group_id, c"nzstart".as_ptr(), (*hdata).nzstart, &mut error);
        add_one_prefixed_integer(group_id, c"mx".as_ptr(), (*hdata).mx, &mut error);
        add_one_prefixed_integer(group_id, c"my".as_ptr(), (*hdata).my, &mut error);
        add_one_prefixed_integer(group_id, c"mz".as_ptr(), (*hdata).mz, &mut error);
        add_one_prefixed_float(group_id, c"xlen".as_ptr(), (*hdata).xlen, &mut error);
        add_one_prefixed_float(group_id, c"ylen".as_ptr(), (*hdata).ylen, &mut error);
        add_one_prefixed_float(group_id, c"zlen".as_ptr(), (*hdata).zlen, &mut error);
        add_one_prefixed_float(group_id, c"alpha".as_ptr(), (*hdata).alpha, &mut error);
        add_one_prefixed_float(group_id, c"beta".as_ptr(), (*hdata).beta, &mut error);
        add_one_prefixed_float(group_id, c"gamma".as_ptr(), (*hdata).gamma, &mut error);
        add_one_prefixed_integer(group_id, c"mapc".as_ptr(), (*hdata).mapc, &mut error);
        add_one_prefixed_integer(group_id, c"mapr".as_ptr(), (*hdata).mapr, &mut error);
        add_one_prefixed_integer(group_id, c"maps".as_ptr(), (*hdata).maps, &mut error);
        let mmm_offset = if (*hdata).mode == MRC_MODE_BYTE && (*hdata).bytes_signed != 0 {
            -128.0
        } else {
            0.0
        };
        add_one_prefixed_float(
            group_id,
            c"minimum".as_ptr(),
            (*hdata).amin + mmm_offset,
            &mut error,
        );
        add_one_prefixed_float(
            group_id,
            c"maximum".as_ptr(),
            (*hdata).amax + mmm_offset,
            &mut error,
        );
        add_one_prefixed_float(
            group_id,
            c"mean".as_ptr(),
            (*hdata).amean + mmm_offset,
            &mut error,
        );
        add_one_prefixed_integer(group_id, c"ispg".as_ptr(), (*hdata).ispg, &mut error);
        add_one_prefixed_float(group_id, c"xorigin".as_ptr(), -(*hdata).xorg, &mut error);
        add_one_prefixed_float(group_id, c"yorigin".as_ptr(), -(*hdata).yorg, &mut error);
        add_one_prefixed_float(group_id, c"zorigin".as_ptr(), -(*hdata).zorg, &mut error);
        if add_float_attribute(
            group_id,
            c"IMOD.MRC.tiltangles".as_ptr(),
            (*hdata).tiltangles.as_mut_ptr(),
            6,
        ) != 0
        {
            error += 1;
        }
        add_one_prefixed_float(group_id, c"rms".as_ptr(), (*hdata).rms, &mut error);
        add_one_prefixed_integer(group_id, c"nlabels".as_ptr(), (*hdata).nlabl, &mut error);
        for ind in 0..(*hdata).nlabl.min(10) {
            let mut label_key = [0 as c_char; 20];
            libc::snprintf(
                label_key.as_mut_ptr(),
                label_key.len(),
                c"IMOD.MRC.label%d".as_ptr(),
                ind,
            );
            (*hdata).labels[ind as usize][80] = 0;
            if add_string_attribute(
                group_id,
                label_key.as_ptr(),
                (*hdata).labels[ind as usize].as_ptr().cast(),
            ) != 0
            {
                error += 1;
            }
        }
    }
    if error == 0
        && adoc_to_attributes(group_id, ADOC_GLOBAL_NAME.as_ptr(), 0, c"IMOD.".as_ptr()) != 0
    {
        error = 1;
    }
    H5Gclose(group_id);
    if error == 0 && (*in_file).num_volumes < 2 {
        error = hdf_write_global_adoc(in_file);
    }
    if !(*in_file).stack_set_list.is_null() {
        for iz in 0..(*in_file).z_map_size {
            if error != 0 {
                break;
            }
            let ind = *(*in_file).z_to_data_set_map.add(iz as usize);
            if ind < 0 {
                continue;
            }
            let mut section_name = [0 as c_char; 20];
            libc::snprintf(
                section_name.as_mut_ptr(),
                section_name.len(),
                c"%d".as_ptr(),
                iz,
            );
            let section = adoc_lookup_section(ADOC_ZVALUE_NAME.as_ptr(), section_name.as_ptr());
            if section < 0 {
                continue;
            }
            let stack = ilist_item((*in_file).stack_set_list.cast(), ind).cast::<StackSetData>();
            let section_group = open_dataset_group(in_file, (*stack).name);
            if section_group < 0 {
                error = 1;
            } else {
                if adoc_to_attributes(
                    section_group,
                    ADOC_ZVALUE_NAME.as_ptr(),
                    section,
                    core::ptr::null(),
                ) != 0
                {
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
        err = adoc_to_attributes(group, ADOC_GLOBAL_NAME.as_ptr(), 0, core::ptr::null());
    }
    let num_collections = adoc_get_num_collections();
    for coll in 0..num_collections {
        if err != 0 {
            break;
        }
        let mut collection_name = core::ptr::null_mut();
        if adoc_get_collection_name(coll, &mut collection_name) != 0 {
            err = 1;
            continue;
        }
        if libc::strcmp(collection_name, ADOC_ZVALUE_NAME.as_ptr()) != 0
            && libc::strcmp(collection_name, c"T".as_ptr()) != 0
        {
            let num_sections = adoc_get_number_of_sections(collection_name);
            let mut collection = H5Gopen2(group, collection_name, 0);
            if collection < 0 {
                collection = H5Gcreate2(group, collection_name, 0, 0, 0);
            }
            if collection < 0 {
                err = 1;
            } else {
                for section in 0..num_sections {
                    if err != 0 {
                        break;
                    }
                    let mut section_name = core::ptr::null_mut();
                    if adoc_get_section_name(collection_name, section, &mut section_name) != 0 {
                        err = 1;
                    } else {
                        let mut section_id = H5Gopen2(collection, section_name, 0);
                        if section_id < 0 {
                            section_id = H5Gcreate2(collection, section_name, 0, 0, 0);
                        }
                        if section_id < 0 {
                            err = 1;
                        } else {
                            if adoc_to_attributes(
                                section_id,
                                collection_name,
                                section,
                                core::ptr::null(),
                            ) != 0
                            {
                                err = 1;
                            }
                            H5Gclose(section_id);
                        }
                        libc::free(section_name.cast());
                    }
                }
                H5Gclose(collection);
            }
        }
        libc::free(collection_name.cast());
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
        core::ptr::copy_nonoverlapping(hdata, (*in_file).header as *mut MrcHeader, 1);
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
        let volume = *(*in_file).ii_volumes.add(ind as usize);
        if !volume.is_null() && volume != in_file && !(*volume).fp.is_null() {
            num_left += 1;
        }
    }
    if num_left == 0 && !(*in_file).fp.is_null() {
        for ind in 0..ilist_size((*in_file).stack_set_list.cast()) {
            let stack = ilist_item((*in_file).stack_set_list.cast(), ind).cast::<StackSetData>();
            if (*stack).is_open != 0 {
                H5Dclose((*stack).dset_id);
            }
            (*stack).is_open = 0;
        }
        H5Fclose((*in_file).hdf_file_id);
    }
    (*in_file).fp = core::ptr::null_mut();
}
/// C `hdfReopen` (`iihdf.c:1523`).
unsafe extern "C" fn hdf_reopen(in_file: *mut ImodImageFile) -> i32 {
    for ind in 0..(*in_file).num_volumes {
        let volume = *(*in_file).ii_volumes.add(ind as usize);
        if !volume.is_null() && !(*volume).fp.is_null() {
            (*in_file).fp = in_file.cast();
            return 0;
        }
    }
    if (*in_file).state == IISTATE_NOTINIT && !(*in_file).filename.is_null() {
        hdf_delete(in_file);
        return ii_hdf_check(in_file);
    }
    let file_id = H5Fopen(
        (*(*(*in_file).ii_volumes)).filename,
        if libc::strstr((*in_file).fmode.as_ptr(), c"+".as_ptr()).is_null() {
            0
        } else {
            1
        },
        0,
    );
    if file_id < 0 {
        return IIERR_IO_ERROR;
    }
    for ind in 0..(*in_file).num_volumes {
        let volume = *(*in_file).ii_volumes.add(ind as usize);
        (*volume).hdf_file_id = file_id;
    }
    (*in_file).fp = in_file.cast();
    0
}
/// C `hdfDelete` (`iihdf.c:1555`).
unsafe extern "C" fn hdf_delete(in_file: *mut ImodImageFile) {
    let mut num_left = 0;
    for ind in 0..(*in_file).num_volumes {
        let volume = *(*in_file).ii_volumes.add(ind as usize);
        if !volume.is_null() && volume != in_file && (*volume).state != IISTATE_UNUSED {
            num_left += 1;
        }
    }
    if num_left == 0 {
        for ind in 0..(*in_file).num_volumes {
            let volume = *(*in_file).ii_volumes.add(ind as usize);
            if !volume.is_null() && volume != in_file && (*volume).state == IISTATE_UNUSED {
                ii_delete(volume);
                *(*in_file).ii_volumes.add(ind as usize) = core::ptr::null_mut();
            }
        }
    }
    for ind in 0..(*in_file).num_volumes {
        if *(*in_file).ii_volumes.add(ind as usize) == in_file {
            for other in 0..(*in_file).num_volumes {
                let volume = *(*in_file).ii_volumes.add(other as usize);
                if other != ind && !volume.is_null() {
                    *(*volume).ii_volumes.add(ind as usize) = core::ptr::null_mut();
                }
            }
            *(*in_file).ii_volumes.add(ind as usize) = core::ptr::null_mut();
        }
    }
    if num_left == 0 && (*in_file).global_adoc_index >= 0 {
        adoc_clear((*in_file).global_adoc_index);
    }
    if num_left == 0 {
        libc::free((*in_file).ii_volumes.cast());
        (*in_file).ii_volumes = core::ptr::null_mut();
    }
    if (*in_file).adoc_index >= 0 {
        adoc_clear((*in_file).adoc_index);
    }
    libc::free((*in_file).dataset_name.cast());
    (*in_file).dataset_name = core::ptr::null_mut();
    for ind in 0..ilist_size((*in_file).stack_set_list.cast()) {
        let stack = ilist_item((*in_file).stack_set_list.cast(), ind).cast::<StackSetData>();
        libc::free((*stack).name.cast());
    }
    ilist_delete((*in_file).stack_set_list.cast());
    (*in_file).stack_set_list = core::ptr::null_mut();
    libc::free((*in_file).z_to_data_set_map.cast());
    (*in_file).z_to_data_set_map = core::ptr::null_mut();
    (*in_file).z_map_size = 0;
    libc::free((*in_file).header.cast());
    (*in_file).header = core::ptr::null_mut();
}
/// C `hdfReadSection` (`iihdf.c:1611`).
unsafe extern "C" fn hdf_read_section(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    cz: i32,
) -> i32 {
    hdf_read_section_any(in_file, buf.cast(), cz, 0)
}
/// C `hdfReadSectionByte` (`iihdf.c:1616`).
unsafe extern "C" fn hdf_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    cz: i32,
) -> i32 {
    hdf_read_section_any(in_file, buf.cast(), cz, MRSA_BYTE)
}
/// C `hdfReadSectionUShort` (`iihdf.c:1621`).
unsafe extern "C" fn hdf_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    cz: i32,
) -> i32 {
    hdf_read_section_any(in_file, buf.cast(), cz, MRSA_USHORT)
}
/// C `hdfReadSectionFloat` (`iihdf.c:1626`).
unsafe extern "C" fn hdf_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    cz: i32,
) -> i32 {
    hdf_read_section_any(in_file, buf.cast(), cz, MRSA_FLOAT)
}
/// C `hdfWriteSection` (`iihdf.c:1631`).
unsafe extern "C" fn hdf_write_section(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    cz: i32,
) -> i32 {
    hdf_write_section_any(in_file, buf.cast(), cz, 0)
}
/// C `hdfWriteSectionFloat` (`iihdf.c:1636`).
unsafe extern "C" fn hdf_write_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    cz: i32,
) -> i32 {
    hdf_write_section_any(in_file, buf.cast(), cz, 1)
}
/// C `hdfWriteDummySection` (`iihdf.c:1641`).
pub unsafe fn hdf_write_dummy_section(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    cz: i32,
) -> i32 {
    hdf_write_section_any(in_file, buf.cast(), cz, -1)
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
    f.fp = in_file.cast();
    if !f.header.is_null() {
        (*f.header.cast::<MrcHeader>()).fp = f.fp.cast();
    }
    f.read_section = Some(hdf_read_section);
    f.read_section_byte = Some(hdf_read_section_byte);
    f.read_section_ushort = Some(hdf_read_section_ushort);
    f.read_section_float = Some(hdf_read_section_float);
    f.close = Some(hdf_close);
    f.clean_up = Some(hdf_delete);
    f.reopen = Some(hdf_reopen);
    f.fill_mrc_header = Some(ii_fill_mrc_header);
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
    (*in_file).z_to_data_set_map =
        libc::malloc((size as usize).wrapping_mul(core::mem::size_of::<i32>())).cast();
    if (*in_file).z_to_data_set_map.is_null() {
        cleanup_from_open(file_id, 1, (*in_file).num_volumes, in_file);
        return 1;
    }
    (*in_file).z_map_size = size;
    for set in 0..size {
        *(*in_file).z_to_data_set_map.add(set as usize) = if sequence != 0 { set } else { -1 };
    }
    0
}
/// C `openDatasetGroup` (`iihdf.c:1694`).
unsafe fn open_dataset_group(in_file: *mut ImodImageFile, ds_name: *const c_char) -> HidT {
    let last = libc::strrchr(ds_name, b'/' as i32);
    if last.is_null() {
        return -1;
    }
    let ind = last.offset_from(ds_name) as i32;
    if manage_malloc_buf(
        (&raw mut S_STRING_BUF).cast(),
        &raw mut S_STR_BUF_SIZE,
        ind + 1,
        1,
    ) != 0
    {
        return -1;
    }
    libc::strncpy(S_STRING_BUF, ds_name, ind as usize);
    *S_STRING_BUF.add(ind as usize) = 0;
    H5Gopen2((*in_file).hdf_file_id, S_STRING_BUF, 0)
}
/// C `cleanupFromOpen` (`iihdf.c:1711`).
unsafe fn cleanup_from_open(
    file_id: HidT,
    close_file: i32,
    num_vol: i32,
    in_file: *mut ImodImageFile,
) {
    if close_file != 0 {
        H5Fclose(file_id);
        (*in_file).fp = libc::fopen((*in_file).filename, (*in_file).fmode.as_ptr());
    }
    if !S_GROUPS.is_null() {
        for ind in 0..ilist_size(S_GROUPS) {
            let group = ilist_item(S_GROUPS, ind).cast::<GroupData>();
            libc::free((*group).name.cast());
        }
        ilist_delete(S_GROUPS);
        S_GROUPS = core::ptr::null_mut();
    }
    if !S_DATASETS.is_null() {
        for ind in 0..ilist_size(S_DATASETS) {
            let dataset = ilist_item(S_DATASETS, ind).cast::<DatasetData>();
            if close_file != 0 {
                H5Dclose((*dataset).dset_id);
            }
            libc::free((*dataset).name.cast());
        }
        ilist_delete(S_DATASETS);
        S_DATASETS = core::ptr::null_mut();
    }
    if num_vol != 0 && !(*in_file).ii_volumes.is_null() {
        for ind in 1..num_vol {
            let volume = *(*in_file).ii_volumes.add(ind as usize);
            libc::free((*volume).header.cast());
            libc::free(volume.cast());
        }
        libc::free((*in_file).ii_volumes.cast());
        (*in_file).ii_volumes = core::ptr::null_mut();
    }
    cleanup_malloc_bufs();
}
/// C `manageMallocBuf` (`iihdf.c:1748`).
unsafe fn manage_malloc_buf(
    buffer: *mut *mut c_void,
    size: *mut i32,
    needed: i32,
    dsize: i32,
) -> i32 {
    if needed > *size {
        libc::free(*buffer);
        *buffer = libc::malloc((needed * dsize) as usize);
        *size = 0;
        if (*buffer).is_null() {
            return 3;
        }
        *size = needed;
    }
    0
}
/// C `cleanupMallocBufs` (`iihdf.c:1764`).
unsafe fn cleanup_malloc_bufs() {
    libc::free(S_STRING_BUF.cast());
    S_STRING_BUF = core::ptr::null_mut();
    S_STR_BUF_SIZE = 0;
    libc::free(S_ATTRIB_NAME.cast());
    S_ATTRIB_NAME = core::ptr::null_mut();
    S_ATTR_NAME_SIZE = 0;
    libc::free(S_FLOAT_BUF.cast());
    S_FLOAT_BUF = core::ptr::null_mut();
    S_FLOAT_BUF_SIZE = 0;
    libc::free(S_INT_BUF.cast());
    S_INT_BUF = core::ptr::null_mut();
    S_INT_BUF_SIZE = 0;
    libc::free(S_SHORT_BUF.cast());
    S_SHORT_BUF = core::ptr::null_mut();
    S_SHORT_BUF_SIZE = 0;
    libc::free(S_USHORT_BUF.cast());
    S_USHORT_BUF = core::ptr::null_mut();
    S_USHORT_BUF_SIZE = 0;
    libc::free(S_MRC_PREFIX.cast());
    S_MRC_PREFIX = core::ptr::null_mut();
}
/// C `attributesToAdoc` (`iihdf.c:1789`).
unsafe fn attributes_to_adoc(
    obj_id: HidT,
    num_attrib: i32,
    type_name: *const c_char,
    sect_ind: i32,
) -> i32 {
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
        if manage_malloc_buf(
            (&raw mut S_ATTRIB_NAME).cast(),
            &raw mut S_ATTR_NAME_SIZE,
            len,
            1,
        ) != 0
        {
            return IIERR_MEMORY_ERR;
        }
        H5Aget_name(attrib_id, S_ATTR_NAME_SIZE as usize, S_ATTRIB_NAME);
        let class = H5Tget_class(type_id);
        let space_type = H5Sget_simple_extent_type(space_id);
        let mut retval = 0;
        if (class == H5T_INTEGER || class == H5T_FLOAT) && space_type == H5S_SIMPLE {
            let dsize = (H5Tget_precision(type_id) / 8) as i32;
            let signed_int = H5Tget_sign(type_id) == H5T_SGN_2;
            let num_vals = H5Sget_simple_extent_npoints(space_id) as i32;
            if class == H5T_FLOAT && dsize == 4 {
                if manage_malloc_buf(
                    (&raw mut S_FLOAT_BUF).cast(),
                    &raw mut S_FLOAT_BUF_SIZE,
                    num_vals,
                    4,
                ) != 0
                {
                    retval = IIERR_MEMORY_ERR;
                }
                if retval == 0 && H5Aread(attrib_id, H5T_NATIVE_FLOAT_g, S_FLOAT_BUF.cast()) < 0 {
                    retval = IIERR_IO_ERROR;
                }
                if retval == 0 {
                    retval = match num_vals {
                        1 => adoc_set_float(type_name, sect_ind, S_ATTRIB_NAME, *S_FLOAT_BUF),
                        2 => adoc_set_two_floats(
                            type_name,
                            sect_ind,
                            S_ATTRIB_NAME,
                            *S_FLOAT_BUF,
                            *S_FLOAT_BUF.add(1),
                        ),
                        3 => adoc_set_three_floats(
                            type_name,
                            sect_ind,
                            S_ATTRIB_NAME,
                            *S_FLOAT_BUF,
                            *S_FLOAT_BUF.add(1),
                            *S_FLOAT_BUF.add(2),
                        ),
                        _ => adoc_set_float_array(
                            type_name,
                            sect_ind,
                            S_ATTRIB_NAME,
                            S_FLOAT_BUF,
                            num_vals,
                        ),
                    };
                    if retval != 0 {
                        retval = IIERR_MEMORY_ERR;
                    }
                }
            } else if dsize == 2 || (dsize == 4 && signed_int) {
                if manage_malloc_buf(
                    (&raw mut S_INT_BUF).cast(),
                    &raw mut S_INT_BUF_SIZE,
                    num_vals,
                    4,
                ) != 0
                {
                    retval = IIERR_MEMORY_ERR;
                }
                if retval == 0
                    && dsize == 2
                    && manage_malloc_buf(
                        (&raw mut S_SHORT_BUF).cast(),
                        &raw mut S_SHORT_BUF_SIZE,
                        num_vals,
                        4,
                    ) != 0
                {
                    retval = IIERR_MEMORY_ERR;
                }
                if retval == 0 && dsize == 2 {
                    if H5Aread(
                        attrib_id,
                        if signed_int {
                            H5T_NATIVE_SHORT_g
                        } else {
                            H5T_NATIVE_USHORT_g
                        },
                        S_SHORT_BUF.cast(),
                    ) < 0
                    {
                        retval = IIERR_IO_ERROR;
                    }
                    if retval == 0 {
                        for val in 0..num_vals as usize {
                            *S_INT_BUF.add(val) = if signed_int {
                                *S_SHORT_BUF.add(val) as i32
                            } else {
                                *(S_SHORT_BUF.add(val).cast::<u16>()) as i32
                            };
                        }
                    }
                } else if retval == 0 && H5Aread(attrib_id, H5T_NATIVE_INT_g, S_INT_BUF.cast()) < 0
                {
                    retval = IIERR_IO_ERROR;
                }
                if retval == 0 {
                    retval = match num_vals {
                        1 => adoc_set_integer(type_name, sect_ind, S_ATTRIB_NAME, *S_INT_BUF),
                        2 => adoc_set_two_integers(
                            type_name,
                            sect_ind,
                            S_ATTRIB_NAME,
                            *S_INT_BUF,
                            *S_INT_BUF.add(1),
                        ),
                        3 => adoc_set_three_integers(
                            type_name,
                            sect_ind,
                            S_ATTRIB_NAME,
                            *S_INT_BUF,
                            *S_INT_BUF.add(1),
                            *S_INT_BUF.add(2),
                        ),
                        _ => adoc_set_integer_array(
                            type_name,
                            sect_ind,
                            S_ATTRIB_NAME,
                            S_INT_BUF,
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
            if manage_malloc_buf(
                (&raw mut S_STRING_BUF).cast(),
                &raw mut S_STR_BUF_SIZE,
                H5Tget_size(type_id) as i32 + 1,
                1,
            ) != 0
            {
                retval = IIERR_MEMORY_ERR;
            }
            if retval == 0 && H5Aread(attrib_id, string_type, S_STRING_BUF.cast()) < 0 {
                retval = IIERR_IO_ERROR;
            }
            if retval == 0
                && adoc_set_key_value(type_name, sect_ind, S_ATTRIB_NAME, S_STRING_BUF) != 0
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
    type_name: *const c_char,
    sect_ind: i32,
    prefix: *const c_char,
) -> i32 {
    let num_keys = adoc_get_number_of_keys(type_name, sect_ind);
    if num_keys < 0 {
        return 1;
    }
    let mut retval = 0;
    for key_ind in 0..num_keys {
        let mut key = core::ptr::null_mut();
        if adoc_get_key_by_index(type_name, sect_ind, key_ind, &mut key) < 0 {
            return 1;
        }
        if key.is_null() {
            continue;
        }
        let pref_key = if !prefix.is_null() && starts_with(key, prefix) == 0 {
            prefixed_key(prefix, key)
        } else {
            key
        };
        let mut value_type = 0;
        let mut num_vals = 0;
        if adoc_get_val_type_and_size(type_name, sect_ind, key, &mut value_type, &mut num_vals) != 0
        {
            retval = 1;
        }
        if retval == 0 && value_type == ADOC_STRING {
            let mut value = core::ptr::null_mut();
            if adoc_get_string(type_name, sect_ind, key, &mut value) != 0 {
                retval = 1;
            } else {
                retval = add_string_attribute(parent_id, pref_key, value);
                libc::free(value.cast());
            }
        } else if retval == 0 && (1..=4).contains(&value_type) {
            if manage_malloc_buf(
                (&raw mut S_INT_BUF).cast(),
                &raw mut S_INT_BUF_SIZE,
                num_vals,
                4,
            ) != 0
            {
                retval = 1;
            } else {
                num_vals = 0;
                if adoc_get_integer_array(
                    type_name,
                    sect_ind,
                    key,
                    S_INT_BUF,
                    &mut num_vals,
                    S_INT_BUF_SIZE,
                ) != 0
                    || add_integer_attribute(parent_id, pref_key, S_INT_BUF, num_vals) != 0
                {
                    retval = 1;
                }
            }
        } else if retval == 0 && (5..=8).contains(&value_type) {
            if manage_malloc_buf(
                (&raw mut S_FLOAT_BUF).cast(),
                &raw mut S_FLOAT_BUF_SIZE,
                num_vals,
                4,
            ) != 0
            {
                retval = 1;
            } else {
                num_vals = 0;
                if adoc_get_float_array(
                    type_name,
                    sect_ind,
                    key,
                    S_FLOAT_BUF,
                    &mut num_vals,
                    S_FLOAT_BUF_SIZE,
                ) != 0
                    || add_float_attribute(parent_id, pref_key, S_FLOAT_BUF, num_vals) != 0
                {
                    retval = 1;
                }
            }
        }
        libc::free(key.cast());
        if retval < 0 {
            return 1;
        }
    }
    0
}
/// C `addIntegerAttribute` (`iihdf.c:1968`).
unsafe fn add_integer_attribute(
    parent_id: HidT,
    key: *const c_char,
    ivals: *mut i32,
    num_vals: i32,
) -> i32 {
    if H5Aexists_by_name(parent_id, c".".as_ptr(), key, 0) > 0 {
        H5Adelete_by_name(parent_id, c".".as_ptr(), key, 0);
    }
    let count = num_vals as HsizeT;
    let space = H5Screate_simple(1, &count, &count);
    if space < 0 {
        return 1;
    }
    let attribute = H5Acreate2(parent_id, key, H5T_NATIVE_INT_g, space, 0, 0);
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
unsafe fn add_float_attribute(
    parent_id: HidT,
    key: *const c_char,
    vals: *mut f32,
    num_vals: i32,
) -> i32 {
    if H5Aexists_by_name(parent_id, c".".as_ptr(), key, 0) > 0 {
        H5Adelete_by_name(parent_id, c".".as_ptr(), key, 0);
    }
    let count = num_vals as HsizeT;
    let space = H5Screate_simple(1, &count, &count);
    if space < 0 {
        return 1;
    }
    let attribute = H5Acreate2(parent_id, key, H5T_NATIVE_FLOAT_g, space, 0, 0);
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
unsafe fn add_string_attribute(parent_id: HidT, key: *const c_char, val_str: *const c_char) -> i32 {
    if H5Aexists_by_name(parent_id, c".".as_ptr(), key, 0) > 0 {
        H5Adelete_by_name(parent_id, c".".as_ptr(), key, 0);
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
    H5Tset_size(typ, libc::strlen(val_str) + 1);
    H5Tset_strpad(typ, 0);
    let attribute = H5Acreate2(parent_id, key, typ, space, 0, 0);
    let result = if attribute < 0 {
        -1
    } else {
        let value = H5Awrite(attribute, typ, val_str.cast());
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
    key: *const c_char,
    mut ival: i32,
    err_sum: *mut i32,
) -> i32 {
    let err = add_integer_attribute(parent_id, prefixed_key(S_MRC_PREFIX, key), &raw mut ival, 1);
    if err != 0 {
        *err_sum += 1;
    }
    err
}
/// C `addOnePrefixedFloat` (`iihdf.c:2060`).
unsafe fn add_one_prefixed_float(
    parent_id: HidT,
    key: *const c_char,
    mut val: f32,
    err_sum: *mut i32,
) -> i32 {
    let err = add_float_attribute(parent_id, prefixed_key(S_MRC_PREFIX, key), &raw mut val, 1);
    if err != 0 {
        *err_sum += 1;
    }
    err
}
/// C `getPrefixedInteger` (`iihdf.c:2071`).
unsafe fn get_prefixed_integer(key: *const c_char, value: *mut i32, err_sum: *mut i32) -> i32 {
    let err = adoc_get_integer(
        ADOC_GLOBAL_NAME.as_ptr(),
        0,
        prefixed_key(S_MRC_PREFIX, key),
        value,
    );
    if err < 0 {
        *err_sum += 1;
    }
    err
}
/// C `getDelPrefixedInteger` (`iihdf.c:2082`).
unsafe fn get_del_prefixed_integer(key: *const c_char, value: *mut i32, err_sum: *mut i32) -> i32 {
    let err = adoc_get_integer(
        ADOC_GLOBAL_NAME.as_ptr(),
        0,
        prefixed_key(S_MRC_PREFIX, key),
        value,
    );
    if err == 0 {
        delete_prefixed_key_value(key);
    }
    if err < 0 {
        *err_sum += 1;
    }
    err
}
/// C `getDelPrefixedFloat` (`iihdf.c:2092`).
unsafe fn get_del_prefixed_float(key: *const c_char, value: *mut f32, err_sum: *mut i32) -> i32 {
    let err = adoc_get_float(
        ADOC_GLOBAL_NAME.as_ptr(),
        0,
        prefixed_key(S_MRC_PREFIX, key),
        value,
    );
    if err == 0 {
        delete_prefixed_key_value(key);
    }
    if err < 0 {
        *err_sum += 1;
    }
    err
}
/// C `deletePrefixedKeyValue` (`iihdf.c:2103`).
unsafe fn delete_prefixed_key_value(key: *const c_char) -> i32 {
    adoc_delete_key_value(
        ADOC_GLOBAL_NAME.as_ptr(),
        0,
        prefixed_key(S_MRC_PREFIX, key),
    )
}
/// C `startsWith` (`iihdf.c:2109`).
unsafe fn starts_with(full: *const c_char, sub: *const c_char) -> i32 {
    if libc::strstr(full, sub) == full.cast_mut() {
        1
    } else {
        0
    }
}
/// C `endsWith` (`iihdf.c:2117`).
unsafe fn ends_with(full: *const c_char, sub: *const c_char) -> i32 {
    let sub_ptr = libc::strstr(full, sub);
    if sub_ptr.is_null() || libc::strlen(sub_ptr) != libc::strlen(sub) {
        return -1;
    }
    sub_ptr.offset_from(full) as i32
}
/// C `prefixedKey` (`iihdf.c:2127`).  Composition is owned by the HDF/autodoc ABI.
unsafe fn prefixed_key(prefix: *const c_char, key: *const c_char) -> *const c_char {
    if prefix.is_null() || *prefix == 0 {
        return key;
    }
    let len1 = libc::strlen(prefix) as i32;
    let len2 = libc::strlen(key) as i32;
    if manage_malloc_buf(
        (&raw mut S_STRING_BUF).cast(),
        &raw mut S_STR_BUF_SIZE,
        len1 + len2 + 1,
        1,
    ) != 0
    {
        return key;
    }
    libc::sprintf(S_STRING_BUF, c"%s%s".as_ptr(), prefix, key);
    S_STRING_BUF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_io_funcs_installs_hdf_header_sync_callback() {
        unsafe {
            let mut image: ImodImageFile = core::mem::zeroed();
            set_io_funcs_plus(&mut image, IIHDF_IMOD, 1, -1);
            assert!(image.sync_from_mrc_header.is_some());
        }
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
}
