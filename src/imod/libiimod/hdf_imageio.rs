//! Translation of `IMOD/libiimod/hdf_imageio.c`.
//!
//! This is intentionally an HDF5 C-ABI client.  IMOD's on-disk HDF layout is
//! defined by HDF5 dataspaces, hyperslabs, native datatypes, and property-list
//! semantics, so replacing these calls with a Rust HDF wrapper would not be a
//! faithful translation.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::autodoc::adoc_new;
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{CArg, b3d_error, b3d_shift_bytes, c_format};
use crate::imod::libcfshr::islice::slice_mode_if_real;
use crate::imod::libiimod::iimage::{
    IIFORMAT_COMPLEX, IIFORMAT_RGB, IITYPE_BYTE, IITYPE_FLOAT, IITYPE_SHORT, IITYPE_UBYTE,
    IITYPE_USHORT, ImodImageFile, LineProcData, MRSA_BYTE, MRSA_FLOAT, MRSA_USHORT, StackSetData,
    ii_convert_line_of_floats,
};
use crate::imod::libiimod::iimrc::ii_mrc_set_load_info;
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MrcHeader,
    mrc_getdcsize,
};
use crate::imod::libiimod::mrcsec::{ii_init_read_section_any, ii_process_read_line};
use core::ffi::{c_char, c_int, c_uint, c_void};
use core::ptr::NonNull;

pub type HidT = i64;
pub type HsizeT = usize;

const H5P_DEFAULT: HidT = 0;
const H5S_SELECT_SET: c_int = 0;
const H5T_ORDER_LE: c_int = 0;
const H5T_ORDER_BE: c_int = 1;
const H5S_UNLIMITED: HsizeT = !0;

unsafe extern "C" {
    fn H5Screate_simple(rank: c_int, dims: *const HsizeT, maxdims: *const HsizeT) -> HidT;
    fn H5Sclose(id: HidT) -> c_int;
    fn H5Sselect_hyperslab(
        id: HidT,
        op: c_int,
        start: *const HsizeT,
        stride: *const HsizeT,
        count: *const HsizeT,
        block: *const HsizeT,
    ) -> c_int;
    fn H5Sget_simple_extent_ndims(id: HidT) -> c_int;
    fn H5Dget_space(id: HidT) -> HidT;
    fn H5Dread(
        id: HidT,
        mem_type: HidT,
        mem_space: HidT,
        file_space: HidT,
        xfer: HidT,
        buf: *mut c_void,
    ) -> c_int;
    fn H5Dwrite(
        id: HidT,
        mem_type: HidT,
        mem_space: HidT,
        file_space: HidT,
        xfer: HidT,
        buf: *const c_void,
    ) -> c_int;
    fn H5Dopen2(file: HidT, name: *const c_char, access: HidT) -> HidT;
    fn H5Dcreate2(
        group: HidT,
        name: *const c_char,
        typ: HidT,
        space: HidT,
        lcpl: HidT,
        dcpl: HidT,
        dapl: HidT,
    ) -> HidT;
    fn H5Gcreate2(file: HidT, name: *const c_char, lcpl: HidT, gcpl: HidT, gapl: HidT) -> HidT;
    fn H5Gclose(id: HidT) -> c_int;
    fn H5Pcreate(class: HidT) -> HidT;
    fn H5Pclose(id: HidT) -> c_int;
    fn H5Pset_deflate(id: HidT, level: c_uint) -> c_int;
    fn H5Pset_chunk(id: HidT, rank: c_int, dims: *const HsizeT) -> c_int;
    fn H5Pget_chunk_cache(id: HidT, slots: *mut usize, bytes: *mut usize, w0: *mut f64) -> c_int;
    fn H5Pset_chunk_cache(id: HidT, slots: usize, bytes: usize, w0: f64) -> c_int;
    fn H5Dget_create_plist(dataset: HidT) -> HidT;
    fn H5Pget_chunk(id: HidT, max_ndims: c_int, dims: *mut HsizeT) -> c_int;
    fn H5Tcopy(id: HidT) -> HidT;
    fn H5Tclose(id: HidT) -> c_int;
    fn H5Tget_precision(id: HidT) -> usize;
    fn H5Tget_order(id: HidT) -> c_int;
    fn H5Tset_order(id: HidT, order: c_int) -> c_int;
    static H5T_NATIVE_SCHAR_g: HidT;
    static H5T_NATIVE_UCHAR_g: HidT;
    static H5T_NATIVE_SHORT_g: HidT;
    static H5T_NATIVE_USHORT_g: HidT;
    static H5T_NATIVE_FLOAT_g: HidT;
    static H5P_CLS_DATASET_CREATE_ID_g: HidT;
    static H5P_CLS_DATASET_ACCESS_ID_g: HidT;
}

/// C `hdfReadSectionAny` (`hdf_imageio.c:33`).
pub unsafe fn hdf_read_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    cz: i32,
    typ: i32,
) -> i32 {
    if in_file.is_null() {
        return 1;
    }
    let mut li = LoadInfo::default();
    ii_mrc_set_load_info(&*in_file, &mut li);
    let Some(hdata) = (*in_file).mrc_header.as_mut() else {
        return 1;
    };
    let mut d = LineProcData::default();
    // This selects read diagnostics in shared cleanup; map ownership is in `d`.
    let read_operation = 0;
    d.type_ = typ;
    d.read_y = if li.axis == 2 { 1 } else { 0 };
    d.cz = cz;
    let mut y_end = if d.read_y != 0 { li.zmax } else { li.ymax };
    if typ == MRSA_FLOAT || typ == 0 {
        li.outmin = (*in_file).smin as i32;
        li.outmax = (*in_file).smax as i32;
        li.mirror_fft = 0;
    } else {
        li.outmin = 0;
        li.outmax = if typ == MRSA_USHORT { 65535 } else { 255 };
        li.mirror_fft = (*in_file).mirror_fft;
    }
    let err = ii_init_read_section_any(hdata, &li, buf, &mut d, &mut y_end, "hdfReadSectionAny");
    if err != 0 {
        return err;
    }
    let pad_left = li.pad_left.max(0);
    let pad_right = li.pad_right.max(0);
    d.x_dimension = d.xsize + pad_left + pad_right;
    d.bufp_offset += match typ {
        MRSA_BYTE => 1,
        MRSA_FLOAT => 4,
        MRSA_USHORT => 2,
        _ => 0,
    } as isize
        * pad_left as isize;
    d.pix_index += pad_left as u32;
    let read_stack_y = d.read_y != 0 && (*in_file).stack_set_list.is_some();
    let mut chunk_lines = 1i32;
    let max_lines = (2_000_000.0 / (d.xsize * d.pix_size) as f32).round() as i32;
    if !read_stack_y
        && !(d.convert != 0 && li.mirror_fft != 0)
        && (max_lines > 1 || d.need_data == 0)
    {
        chunk_lines = y_end + 1 - d.y_start;
        if d.need_data != 0 {
            chunk_lines = chunk_lines.min(max_lines).max(1);
        }
    }
    let mut tmp_data = if d.need_data != 0 {
        vec![0; (d.pix_size * d.xsize * chunk_lines) as usize]
    } else {
        Vec::new()
    };
    let tmp = tmp_data.as_mut_ptr();
    let scale = get_file_xscale((*in_file).format);
    let mut mem_dim = [chunk_lines as HsizeT, (d.x_dimension * scale) as HsizeT, 0];
    let mut mem_count = [chunk_lines as HsizeT, (d.xsize * scale) as HsizeT, 0];
    let mut mem_offset = [
        0usize,
        if d.need_data != 0 {
            0
        } else {
            (pad_left * scale) as usize
        },
        0,
    ];
    if d.need_data != 0 {
        mem_dim[1] = (d.xsize * scale) as usize;
    }
    let mem_space = H5Screate_simple(2, mem_dim.as_ptr(), core::ptr::null());
    if H5Sselect_hyperslab(
        mem_space,
        H5S_SELECT_SET,
        mem_offset.as_ptr(),
        core::ptr::null(),
        mem_count.as_ptr(),
        core::ptr::null(),
    ) < 0
    {
        cleanup_tmp(
            tmp,
            read_operation,
            d.map.as_mut_ptr(),
            0,
            mem_space,
            Some("selecting memory area to use"),
        );
        return 3;
    }
    let mut dset = 0;
    let mut dspace = 0;
    let mut no_data = 0;
    let mut rank = 0;
    let mut fcount = [0usize; 3];
    let mut foffset = [0usize; 3];
    if !read_stack_y {
        dset = get_dataset_for_z(&mut *in_file, cz, &mut no_data);
        if no_data != 0 {
            for _ in d.y_start..=y_end {
                let pixel_size = match typ {
                    MRSA_BYTE => 1,
                    MRSA_FLOAT => 4,
                    MRSA_USHORT => 2,
                    _ => 0,
                };
                core::ptr::write_bytes(
                    d.buf.offset(d.bufp_offset),
                    0,
                    pixel_size * d.xsize as usize,
                );
                d.bufp_offset += pixel_size as isize * (pad_left + pad_right) as isize;
            }
            return 0;
        }
        dspace = H5Dget_space(dset);
        rank = H5Sget_simple_extent_ndims(dspace);
        fcount[(rank - 1) as usize] = (d.xsize * scale) as usize;
        foffset[(rank - 1) as usize] = (d.x_start * scale) as usize;
    }
    let native = lookup_native_datatype(&*in_file);
    d.line = d.y_start;
    while d.line <= y_end {
        let line_end = y_end.min(d.line + chunk_lines - 1);
        let num = line_end + 1 - d.line;
        if num as usize != mem_count[0] {
            mem_count[0] = num as usize;
            if H5Sselect_hyperslab(
                mem_space,
                H5S_SELECT_SET,
                mem_offset.as_ptr(),
                core::ptr::null(),
                mem_count.as_ptr(),
                core::ptr::null(),
            ) < 0
            {
                cleanup_tmp(
                    tmp,
                    read_operation,
                    d.map.as_mut_ptr(),
                    dspace,
                    mem_space,
                    Some("selecting memory area to use"),
                );
                return 3;
            }
        }
        if d.read_y == 0 && (*in_file).stack_set_list.is_some() {
            fcount[0] = 1;
            fcount[(rank - 2) as usize] = num as usize;
            foffset[0] = 0;
            foffset[(rank - 2) as usize] = d.line as usize;
        } else if read_stack_y {
            dset = get_dataset_for_z(&mut *in_file, d.line, &mut no_data);
            if no_data == 0 {
                dspace = H5Dget_space(dset);
                rank = H5Sget_simple_extent_ndims(dspace);
                fcount[0] = 1;
                fcount[(rank - 2) as usize] = 1;
                fcount[(rank - 1) as usize] = (d.xsize * scale) as usize;
                foffset[0] = 0;
                foffset[(rank - 2) as usize] = cz as usize;
                foffset[(rank - 1) as usize] = (d.x_start * scale) as usize;
            }
        } else if d.read_y == 0 {
            fcount[0] = 1;
            fcount[(rank - 2) as usize] = num as usize;
            foffset[0] = cz as usize;
            foffset[(rank - 2) as usize] = d.line as usize;
        } else {
            fcount[0] = num as usize;
            fcount[(rank - 2) as usize] = 1;
            foffset[0] = d.line as usize;
            foffset[(rank - 2) as usize] = cz as usize;
        }
        if no_data == 0
            && H5Sselect_hyperslab(
                dspace,
                H5S_SELECT_SET,
                foffset.as_ptr(),
                core::ptr::null(),
                fcount.as_ptr(),
                core::ptr::null(),
            ) < 0
        {
            cleanup_tmp(
                tmp,
                read_operation,
                d.map.as_mut_ptr(),
                dspace,
                mem_space,
                Some("selecting area to read from file"),
            );
            return 3;
        }
        d.bdata = if d.need_data != 0 {
            tmp
        } else {
            d.buf.offset(d.bufp_offset)
        };
        let read_ptr = d.bdata.sub((pad_left * d.pix_size) as usize);
        let mut needed = true;
        while d.line <= line_end {
            if no_data != 0 {
                core::ptr::write_bytes(d.bdata, 0, (d.pix_size * d.xsize) as usize);
            } else if needed {
                if H5Dread(
                    dset,
                    native,
                    mem_space,
                    dspace,
                    H5P_DEFAULT,
                    read_ptr.cast(),
                ) < 0
                {
                    cleanup_tmp(
                        tmp,
                        read_operation,
                        d.map.as_mut_ptr(),
                        dspace,
                        mem_space,
                        Some("reading data from file"),
                    );
                    return 3;
                }
                if read_stack_y || d.line == y_end {
                    H5Sclose(dspace);
                }
            }
            needed = false;
            if ii_process_read_line(
                hdata,
                &li,
                &mut d,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            ) != 0
            {
                return 1;
            }
            d.bdata = d.bdata.add((d.xsize * d.pix_size) as usize);
            d.line += 1;
        }
    }
    cleanup_tmp(tmp, read_operation, d.map.as_mut_ptr(), 0, mem_space, None);
    0
}

/// C `hdfWriteSectionAny` (`hdf_imageio.c:247`).
pub unsafe fn hdf_write_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    cz: i32,
    from_float: i32,
) -> i32 {
    if in_file.is_null() {
        return 1;
    }
    let Some(h) = (*in_file).mrc_header.as_mut() else {
        return 1;
    };
    let mut li = LoadInfo::default();
    ii_mrc_set_load_info(&*in_file, &mut li);
    let mut buf_mode = h.mode;
    let convert = from_float > 0 && !matches!(h.mode, MRC_MODE_COMPLEX_FLOAT | MRC_MODE_FLOAT);
    if convert {
        buf_mode = MRC_MODE_FLOAT;
    }
    let bytes_signed = (h.mode == 0 && h.bytes_signed != 0) as i32;
    let (xstart, xend, ystart, yend) = (li.xmin, li.xmax, li.ymin, li.ymax);
    let nxout = xend + 1 - xstart;
    let tiled = (*in_file).tile_size_x > 0 && (*in_file).tile_size_x < h.nx;
    if (!tiled && xstart != 0)
        || (tiled && xstart % (*in_file).tile_size_x != 0)
        || (xend != h.nx - 1 && (!tiled || (xend + 1) % (*in_file).tile_size_x != 0))
    {
        return 1;
    }
    let mut bpo = 0;
    let mut nco = 0;
    if mrc_getdcsize(h.mode, &mut bpo, &mut nco) != 0 || h.mode == MRC_MODE_COMPLEX_SHORT {
        return -1;
    }
    let mut bpb = 0;
    let mut ncb = 0;
    mrc_getdcsize(buf_mode, &mut bpb, &mut ncb);
    let pixout = bpo * nco;
    let pixbuf = bpb * ncb;
    if convert && slice_mode_if_real(h.mode) < 0 {
        return 1;
    }
    let pad_left = li.pad_left.max(0);
    let pad_right = li.pad_right.max(0);
    let xdim = nxout + pad_left + pad_right;
    let need_data = convert || bytes_signed != 0;
    let mut chunk = yend + 1 - ystart;
    if need_data && from_float >= 0 {
        chunk = ((65536.0 / (nxout * pixout) as f32).round() as i32).clamp(1, yend + 1 - ystart);
    }
    let mut tmp_data = if need_data && from_float >= 0 {
        vec![0; (pixout * nxout * chunk) as usize]
    } else {
        Vec::new()
    };
    let tmp = tmp_data.as_mut_ptr();
    if (*in_file).stack_set_list.is_none()
        && (*in_file).dataset_name.is_none()
        && init_new_hdf_file(&mut *in_file) != 0
    {
        cleanup_tmp(
            tmp,
            -1,
            core::ptr::null_mut(),
            0,
            0,
            Some("Initializing new HDF for writing"),
        );
        return 2;
    }
    let mut none = 0;
    let dset = if (*in_file).stack_set_list.is_some() {
        /* This is the stack branch from hdfWriteSectionAny itself, kept inline
        as in C rather than factored into a new Rust helper. */
        if cz >= (*in_file).z_map_size {
            let new_size = cz + 8;
            (*in_file).z_to_data_set_map.resize(new_size as usize, -1);
            (*in_file).z_map_size = new_size;
        }
        if (&(*in_file).z_to_data_set_map)[cz as usize] < 0 {
            let mut name = [0u8; 36];
            let new_dset = create_group_and_dataset(&mut *in_file, cz, &mut name);
            if new_dset < 0 {
                cleanup_tmp(
                    tmp,
                    -1,
                    core::ptr::null_mut(),
                    0,
                    0,
                    Some("Creating dataset for new section"),
                );
                return 1;
            }
            let stack_name = String::from_utf8_lossy(
                &name[..name.iter().position(|b| *b == 0).unwrap_or(name.len())],
            )
            .into_owned();
            (*in_file)
                .stack_set_list
                .as_mut()
                .expect("stack storage was initialized")
                .push(StackSetData {
                    name: Some(stack_name),
                    dset_id: new_dset,
                    is_open: true,
                });
            let stack_index = (*in_file).stack_set_list.as_deref().unwrap().len() as i32 - 1;
            (&mut (*in_file).z_to_data_set_map)[cz as usize] = stack_index;
        }
        get_dataset_for_z(&mut *in_file, cz, &mut none)
    } else {
        get_dataset_for_z(&mut *in_file, cz, &mut none)
    };
    if dset < 0 {
        cleanup_tmp(
            tmp,
            -1,
            core::ptr::null_mut(),
            0,
            0,
            Some("Opening existing dataset for writing a section"),
        );
        return 1;
    }
    if from_float < 0 {
        cleanup_tmp(tmp, -1, core::ptr::null_mut(), 0, 0, None);
        return 0;
    }
    let scale = get_file_xscale((*in_file).format);
    let dspace = H5Dget_space(dset);
    let rank = H5Sget_simple_extent_ndims(dspace) as usize;
    let mut md = [0usize; 3];
    let mut mc = [0usize; 3];
    let mut mo = [0usize; 3];
    md[0] = 1;
    mc[0] = 1;
    md[rank - 2] = chunk as usize;
    mc[rank - 2] = chunk as usize;
    md[rank - 1] = if need_data {
        (nxout * scale) as usize
    } else {
        (xdim * scale) as usize
    };
    mc[rank - 1] = (nxout * scale) as usize;
    let mspace = H5Screate_simple(rank as i32, md.as_ptr(), core::ptr::null());
    if md[rank - 1] != mc[rank - 1]
        && H5Sselect_hyperslab(
            mspace,
            H5S_SELECT_SET,
            mo.as_ptr(),
            core::ptr::null(),
            mc.as_ptr(),
            core::ptr::null(),
        ) < 0
    {
        cleanup_tmp(
            tmp,
            -1,
            core::ptr::null_mut(),
            dspace,
            mspace,
            Some("selecting memory area to use"),
        );
        return 3;
    }
    let mut fc = [0usize; 3];
    let mut fo = [0usize; 3];
    fc[0] = 1;
    fo[0] = cz as usize;
    fc[rank - 1] = (nxout * scale) as usize;
    fo[rank - 1] = (xstart * scale) as usize;
    let native = lookup_native_datatype(&*in_file);
    let mut line = ystart;
    let mut bufp = buf.add((pixbuf * pad_left) as usize);
    let mut fbufp = buf.cast::<f32>().add(pad_left as usize);
    while line <= yend {
        let end = yend.min(line + chunk - 1);
        let num = end + 1 - line;
        if num as usize != mc[rank - 2] {
            mc[rank - 2] = num as usize;
            if H5Sselect_hyperslab(
                mspace,
                H5S_SELECT_SET,
                mo.as_ptr(),
                core::ptr::null(),
                mc.as_ptr(),
                core::ptr::null(),
            ) < 0
            {
                cleanup_tmp(
                    tmp,
                    -1,
                    core::ptr::null_mut(),
                    dspace,
                    mspace,
                    Some("selecting memory area to use"),
                );
                return 3;
            }
        }
        fc[rank - 2] = num as usize;
        fo[rank - 2] = line as usize;
        if H5Sselect_hyperslab(
            dspace,
            H5S_SELECT_SET,
            fo.as_ptr(),
            core::ptr::null(),
            fc.as_ptr(),
            core::ptr::null(),
        ) < 0
        {
            cleanup_tmp(
                tmp,
                -1,
                core::ptr::null_mut(),
                dspace,
                mspace,
                Some("selecting area to write to file"),
            );
            return 3;
        }
        let mut bdata = if need_data { tmp } else { bufp };
        let write = bdata;
        while line <= end {
            if convert {
                ii_convert_line_of_floats(
                    core::slice::from_raw_parts(fbufp, nxout as usize),
                    core::slice::from_raw_parts_mut(bdata, (pixout * nxout) as usize),
                    (*h).mode,
                    bytes_signed != 0,
                    false,
                );
                fbufp = fbufp.add(xdim as usize);
                bufp = fbufp.cast();
            } else {
                if bytes_signed != 0 {
                    if core::ptr::eq(bufp, bdata) {
                        b3d_shift_bytes(
                            core::slice::from_raw_parts_mut(bdata, nxout as usize),
                            nxout,
                            1,
                            1,
                            1,
                        );
                    } else {
                        let source = core::slice::from_raw_parts(bufp, nxout as usize);
                        let destination = core::slice::from_raw_parts_mut(bdata, nxout as usize);
                        for (destination, source) in destination.iter_mut().zip(source) {
                            *destination = (*source as i32 - 128) as i8 as u8;
                        }
                    }
                }
                bufp = bufp.add((pixbuf * xdim) as usize);
            }
            bdata = bdata.add((pixout * nxout) as usize);
            line += 1;
        }
        if H5Dwrite(dset, native, mspace, dspace, H5P_DEFAULT, write.cast()) < 0 {
            cleanup_tmp(
                tmp,
                -1,
                core::ptr::null_mut(),
                dspace,
                mspace,
                Some("Writing data to file"),
            );
            return 3;
        }
    }
    cleanup_tmp(tmp, -1, core::ptr::null_mut(), dspace, mspace, None);
    0
}

/// C `initNewHDFfile` (`hdf_imageio.c:456`).
pub fn init_new_hdf_file(in_file: &mut ImodImageFile) -> i32 {
    let size = in_file.nz.max(4);
    if in_file.z_chunk_size != 0 {
        let mut name = [0u8; 36];
        let ds = unsafe { create_group_and_dataset(in_file, in_file.num_volumes - 1, &mut name) };
        if ds < 0 {
            return 1;
        }
        in_file.dataset_name = Some(
            String::from_utf8_lossy(
                &name[..name.iter().position(|b| *b == 0).unwrap_or(name.len())],
            )
            .into_owned(),
        );
        in_file.dataset_id = ds;
        in_file.dataset_is_open = 1;
        if in_file.global_adoc_index < 0 {
            in_file.global_adoc_index = adoc_new();
            if in_file.global_adoc_index < 0 {
                return 1;
            }
            for volume in in_file.ii_volumes.iter().flatten() {
                unsafe {
                    (*volume.as_ptr()).global_adoc_index = in_file.global_adoc_index;
                }
            }
        }
    } else {
        in_file.stack_set_list = Some(Vec::with_capacity(size as usize));
        in_file.z_to_data_set_map = vec![-1; size as usize];
        in_file.z_map_size = size;
    }
    0
}

/// C static `createGroupAndDataset` (`hdf_imageio.c:491`).
unsafe fn create_group_and_dataset(
    in_file: &mut ImodImageFile,
    z: i32,
    buf: &mut [u8; 36],
) -> HidT {
    let scale = get_file_xscale(in_file.format);
    let rank = if in_file.z_chunk_size > 0 { 3 } else { 2 };
    let xi = rank - 1;
    let yi = rank - 2;
    let mut fd = [0usize; 3];
    let mut md = [0usize; 3];
    fd[xi as usize] = (in_file.nx * scale) as usize;
    md[xi as usize] = fd[xi as usize];
    fd[yi as usize] = in_file.ny as usize;
    md[yi as usize] = fd[yi as usize];
    if rank == 3 {
        fd[0] = in_file.nz as usize;
        md[0] = H5S_UNLIMITED;
    }
    /* The property-list/chunk-cache logic is retained from the C source: HDF5
    itself, not a Rust image abstraction, selects the physical layout. */
    if in_file.hdf_compression < 0 {
        let comp_env = std::env::var_os("IMOD_HDF_COMPRESSION");
        in_file.hdf_compression = 0;
        if let Some(value) = comp_env {
            // C `atoi`, which is `strtol` base 10 truncated to `int`: "3x" is
            // 3 where `str::parse` would fail.
            let mut end = 0;
            in_file.hdf_compression =
                crate::imod::libcfshr::parse_params::strtol(value.as_encoded_bytes(), &mut end, 10)
                    as i32;
            in_file.hdf_compression = in_file.hdf_compression.clamp(0, 9);
        }
    }
    // `hdf_imageio.c:575`: `sprintf(buf, "/MDF/images/%d", zValue)`.
    let text = c_format("/MDF/images/%d", &[CArg::Int(z as i64)]);
    buf[..text.len()].copy_from_slice(text.as_bytes());
    buf[text.len()] = 0;
    let name = std::ffi::CString::new(text.as_str()).unwrap();
    let group = H5Gcreate2(
        in_file.hdf_file_id,
        name.as_ptr(),
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT,
    );
    let compressed = in_file.hdf_compression > 0;
    let chunked = rank == 3 || compressed;
    let mut cparms = H5P_DEFAULT;
    let mut aparms = H5P_DEFAULT;
    if chunked {
        cparms = H5Pcreate(H5P_CLS_DATASET_CREATE_ID_g);
        if H5Pset_deflate(cparms, in_file.hdf_compression as c_uint) < 0 {
            return -1;
        }
        let mut chunks = [0usize; 3];
        if rank == 3 {
            chunks[0] = in_file.z_chunk_size.max(1) as usize;
        }
        let tile_x = if rank == 3 {
            in_file.tile_size_x
        } else {
            in_file.nx
        };
        let tile_y = if rank == 3 {
            in_file.tile_size_y
        } else {
            (65_536 / in_file.nx.max(1)).max(4)
        };
        chunks[yi as usize] = if tile_y <= 0 || tile_y > in_file.ny {
            in_file.ny as usize
        } else {
            tile_y as usize
        };
        chunks[xi as usize] = if tile_x <= 0 || tile_x > in_file.nx {
            (in_file.nx * scale) as usize
        } else {
            (tile_x * scale) as usize
        };
        if H5Pset_chunk(cparms, rank, chunks.as_ptr()) < 0 {
            return -1;
        }
        aparms = H5Pcreate(H5P_CLS_DATASET_ACCESS_ID_g);
        let (mut slots, mut bytes, mut w0) = (0usize, 0usize, 0f64);
        if H5Pget_chunk_cache(aparms, &mut slots, &mut bytes, &mut w0) < 0 {
            return -1;
        }
        let (mut bpc, mut channels) = (0, 0);
        mrc_getdcsize(in_file.mode, &mut bpc, &mut channels);
        let nchunks = (in_file.nx + chunks[xi as usize] as i32 - 1) / chunks[xi as usize] as i32;
        let wanted = (bpc as usize
            * nchunks as usize
            * chunks[xi as usize]
            * scale as usize
            * chunks[yi as usize]
            * 11)
            / 10;
        if nchunks > 1 && wanted > bytes && H5Pset_chunk_cache(aparms, slots, wanted, w0) < 0 {
            return -1;
        }
    }
    let native = lookup_native_datatype(in_file);
    let typ = H5Tcopy(native);
    if H5Tget_precision(typ) > 8
        && in_file
            .mrc_header
            .as_ref()
            .is_some_and(|header| header.swapped != 0)
    {
        if H5Tget_order(native) == H5T_ORDER_LE {
            H5Tset_order(typ, H5T_ORDER_BE);
        } else {
            H5Tset_order(typ, H5T_ORDER_LE);
        }
    }
    let space = H5Screate_simple(rank, fd.as_ptr(), md.as_ptr());
    // The HDF5 dataset name, at the HDF5 boundary.
    let image = c"image";
    let ds = H5Dcreate2(
        group,
        image.as_ptr(),
        typ,
        space,
        H5P_DEFAULT,
        cparms,
        aparms,
    );
    if space < 0 || ds < 0 {
        return -1;
    }
    if chunked {
        H5Pclose(cparms);
        H5Pclose(aparms);
    }
    H5Tclose(typ);
    H5Sclose(space);
    H5Gclose(group);
    // `hdf_imageio.c:645`: `sprintf(buf, "/MDF/images/%d/image", zValue)`.
    let out = c_format("/MDF/images/%d/image", &[CArg::Int(z as i64)]);
    buf[..out.len()].copy_from_slice(out.as_bytes());
    buf[out.len()] = 0;
    ds
}

/// C static `getFileXscale` (`hdf_imageio.c:612`).
fn get_file_xscale(format: i32) -> i32 {
    if format == IIFORMAT_RGB {
        3
    } else if format == IIFORMAT_COMPLEX {
        2
    } else {
        1
    }
}
/// C static `lookupNativeDatatype` (`hdf_imageio.c:624`).
unsafe fn lookup_native_datatype(in_file: &ImodImageFile) -> HidT {
    match in_file.type_ {
        IITYPE_BYTE => H5T_NATIVE_SCHAR_g,
        IITYPE_UBYTE => H5T_NATIVE_UCHAR_g,
        IITYPE_SHORT => H5T_NATIVE_SHORT_g,
        IITYPE_USHORT => H5T_NATIVE_USHORT_g,
        IITYPE_FLOAT => H5T_NATIVE_FLOAT_g,
        _ => H5T_NATIVE_SCHAR_g,
    }
}
/// C static `getDatasetForZ` (`hdf_imageio.c:646`).
unsafe fn get_dataset_for_z(in_file: &mut ImodImageFile, cz: i32, no_data: &mut i32) -> HidT {
    *no_data = 0;
    if in_file.stack_set_list.is_some() {
        if cz >= in_file.z_map_size || in_file.z_to_data_set_map[cz as usize] < 0 {
            *no_data = 1;
            return 0;
        }
        let Some(stack) = in_file
            .stack_set_list
            .as_deref_mut()
            .and_then(|stacks| stacks.get_mut(in_file.z_to_data_set_map[cz as usize] as usize))
        else {
            *no_data = 1;
            return 0;
        };
        if stack.is_open {
            stack.dset_id
        } else {
            let name = std::ffi::CString::new(stack.name.as_deref().unwrap_or_default()).unwrap();
            let id = H5Dopen2(in_file.hdf_file_id, name.as_ptr(), H5P_DEFAULT);
            stack.dset_id = id;
            stack.is_open = true;
            id
        }
    } else if in_file.dataset_is_open != 0 {
        in_file.dataset_id
    } else {
        let name =
            std::ffi::CString::new(in_file.dataset_name.clone().unwrap_or_default()).unwrap();
        let id = H5Dopen2(in_file.hdf_file_id, name.as_ptr(), H5P_DEFAULT);
        in_file.dataset_id = id;
        in_file.dataset_is_open = 1;
        id
    }
}
/// C static `cleanupTmp` (`hdf_imageio.c:683`).
unsafe fn cleanup_tmp(
    tmp: *mut u8,
    read_operation: i32,
    _map: *mut u8,
    dspace: HidT,
    mspace: HidT,
    mess: Option<&str>,
) {
    {
        use std::io::Write;
        let _ = crate::imod::libcfshr::b3dutil::ImodFile::Stdout.flush();
    }
    if mspace != 0 {
        H5Sclose(mspace);
    }
    if dspace != 0 {
        H5Sclose(dspace);
    }
    if let Some(message) = mess {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: hdf{}SectionAny - {}.\n",
                if read_operation >= 0 { "Read" } else { "Write" },
                message
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[link(name = "hdf5_serial")]
    unsafe extern "C" {
        fn H5Fcreate(filename: *const c_char, flags: u32, create: HidT, access: HidT) -> HidT;
        fn H5Fclose(file: HidT) -> c_int;
        fn H5Dclose(dataset: HidT) -> c_int;
    }

    #[test]
    fn init_new_hdf_file_creates_volume_dataset_and_global_adoc() {
        unsafe {
            let mut path = std::env::temp_dir();
            path.push(format!(
                "imod-rs-hdf-imageio-{}-{}.h5",
                std::process::id(),
                1
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let file = H5Fcreate(name.as_ptr(), 2, H5P_DEFAULT, H5P_DEFAULT);
            assert!(file >= 0);
            let mdf = H5Gcreate2(file, c"MDF".as_ptr(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert!(mdf >= 0);
            let images = H5Gcreate2(
                file,
                c"/MDF/images".as_ptr(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT,
            );
            assert!(images >= 0);
            assert_eq!(H5Gclose(images), 0);
            assert_eq!(H5Gclose(mdf), 0);

            let mut header = MrcHeader::default();
            let mut image = ImodImageFile::default();
            let mut companion = ImodImageFile::default();
            image.nx = 2;
            image.ny = 2;
            image.nz = 1;
            header.nx = 2;
            header.ny = 2;
            header.nz = 1;
            header.mode = MRC_MODE_FLOAT;
            image.mrc_header = Some(header);
            image.num_volumes = 2;
            let volumes = vec![
                Some(NonNull::from(&mut image)),
                Some(NonNull::from(&mut companion)),
            ];
            image.ii_volumes = volumes;
            image.hdf_file_id = file;
            image.z_chunk_size = 1;
            image.hdf_compression = 0;
            image.type_ = IITYPE_FLOAT;
            image.global_adoc_index = -1;
            assert_eq!(init_new_hdf_file(&mut image), 0);
            assert!(image.dataset_id >= 0);
            assert_eq!(image.dataset_is_open, 1);
            assert_eq!(image.dataset_name.as_deref(), Some("/MDF/images/1/image"));
            assert!(image.global_adoc_index >= 0);
            assert_eq!(companion.global_adoc_index, image.global_adoc_index);
            let plist = H5Dget_create_plist(image.dataset_id);
            assert!(plist >= 0);
            let mut chunks = [0usize; 3];
            assert_eq!(H5Pget_chunk(plist, 3, chunks.as_mut_ptr()), 3);
            assert_eq!(chunks, [1, 2, 2]);
            assert_eq!(H5Pclose(plist), 0);
            assert_eq!(H5Dclose(image.dataset_id), 0);
            image.dataset_is_open = 0;
            let mut no_data = -1;
            assert!(get_dataset_for_z(&mut image, 0, &mut no_data) >= 0);
            assert_eq!(no_data, 0);
            assert_eq!(image.dataset_is_open, 1);
            let written = [1.0f32, 2.0, 3.0, 4.0];
            assert_eq!(
                H5Dwrite(
                    image.dataset_id,
                    H5T_NATIVE_FLOAT_g,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    written.as_ptr().cast(),
                ),
                0
            );
            let mut read = [0.0f32; 4];
            assert_eq!(
                H5Dread(
                    image.dataset_id,
                    H5T_NATIVE_FLOAT_g,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    read.as_mut_ptr().cast(),
                ),
                0
            );
            assert_eq!(read, written);
            image.llx = 1;
            image.urx = 1;
            image.lly = 0;
            image.ury = -1;
            image.llz = 0;
            image.urz = -1;
            image.pad_left = 1;
            image.pad_right = 1;
            image.slope = 1.0;
            image.tile_size_x = 1;
            let replacement = [-1.0f32, 20.0, -1.0, -1.0, 40.0, -1.0];
            assert_eq!(
                hdf_write_section_any(&mut image, replacement.as_ptr().cast_mut().cast(), 0, 0),
                0
            );
            let mut rewritten = [0.0f32; 4];
            assert_eq!(
                H5Dread(
                    image.dataset_id,
                    H5T_NATIVE_FLOAT_g,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    rewritten.as_mut_ptr().cast(),
                ),
                0
            );
            assert_eq!(rewritten, [1.0, 20.0, 3.0, 40.0]);
            let mut output = [-1.0f32; 6];
            assert_eq!(
                hdf_read_section_any(&mut image, output.as_mut_ptr().cast(), 0, MRSA_FLOAT),
                0
            );
            assert_eq!(output, [-1.0, 20.0, -1.0, -1.0, 40.0, -1.0]);
            assert_eq!(H5Dclose(image.dataset_id), 0);
            crate::imod::libcfshr::autodoc::adoc_clear(image.global_adoc_index);
            image.dataset_name = None;
            assert_eq!(H5Fclose(file), 0);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn stack_write_creates_sparse_section_dataset_and_map() {
        unsafe {
            let mut path = std::env::temp_dir();
            path.push(format!("imod-rs-hdf-stack-{}-{}.h5", std::process::id(), 1));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let file = H5Fcreate(name.as_ptr(), 2, H5P_DEFAULT, H5P_DEFAULT);
            assert!(file >= 0);
            let mdf = H5Gcreate2(file, c"MDF".as_ptr(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert!(mdf >= 0);
            let images = H5Gcreate2(
                file,
                c"/MDF/images".as_ptr(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT,
            );
            assert!(images >= 0);
            assert_eq!(H5Gclose(images), 0);
            assert_eq!(H5Gclose(mdf), 0);

            let mut header = MrcHeader::default();
            header.nx = 2;
            header.ny = 2;
            header.nz = 4;
            header.mode = MRC_MODE_FLOAT;
            let mut image = ImodImageFile::default();
            image.mrc_header = Some(header);
            image.nx = 2;
            image.ny = 2;
            image.nz = 4;
            image.urx = -1;
            image.ury = -1;
            image.urz = -1;
            image.hdf_file_id = file;
            image.type_ = IITYPE_FLOAT;
            image.global_adoc_index = -1;
            assert_eq!(init_new_hdf_file(&mut image), 0);
            assert!(image.stack_set_list.is_some());
            assert_eq!(image.z_map_size, 4);
            assert_eq!(image.z_to_data_set_map[3], -1);

            let section = [5.0f32, 6.0, 7.0, 8.0];
            assert_eq!(
                hdf_write_section_any(&mut image, section.as_ptr().cast_mut().cast(), 3, 0),
                0
            );
            assert_eq!(image.stack_set_list.as_deref().unwrap().len(), 1);
            assert_eq!(image.z_to_data_set_map[3], 0);
            let stack = image
                .stack_set_list
                .as_deref_mut()
                .unwrap()
                .first_mut()
                .unwrap();
            assert_eq!(stack.name.as_deref(), Some("/MDF/images/3/image"));
            let mut read = [0.0f32; 4];
            assert_eq!(
                H5Dread(
                    stack.dset_id,
                    H5T_NATIVE_FLOAT_g,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    read.as_mut_ptr().cast(),
                ),
                0
            );
            assert_eq!(read, section);
            assert_eq!(H5Dclose(stack.dset_id), 0);
            image.stack_set_list = None;
            assert_eq!(H5Fclose(file), 0);
            std::fs::remove_file(path).unwrap();
        }
    }
}
