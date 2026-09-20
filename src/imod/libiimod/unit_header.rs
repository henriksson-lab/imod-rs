//! Translation of `IMOD/libiimod/unit_header.c` and declarations from
//! `IMOD/include/iiunit.h`.
//!
//! The unit table itself belongs to `unit_fileio.c`, which is a later source
//! unit in the ordered port.  Calls to that source unit retain its C ABI here;
//! this file contains the complete header manipulation logic rather than a
//! second, incompatible header table.
#![allow(unused_variables)]

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, set_or_clear_flags, write_16_bit_mode_for_floats,
};
use crate::imod::libcfshr::linearxforms::{angles_to_matrix, icalc_angles};
use crate::imod::libiimod::iimage::{IIFILE_MRC, IIFILE_TIFF, ImodImageFile};
use crate::imod::libiimod::mrcfiles::{
    IIUNIT_4BIT_MODE, IIUNIT_HALF_FLOATS, IIUNIT_HALF_XSIZE, IMOD_MRC_STAMP, MRC_EXT_TYPE_AGAR,
    MRC_EXT_TYPE_NONE, MRC_EXT_TYPE_SERI, MRC_LABEL_SIZE, MRC_MODE_BYTE, MRC_MODE_FLOAT,
    MRC_MODE_HALF_FLOAT, MRC_NLABELS, MrcHeader, PACKED_4BIT_MODE, PACKED_HALF_XSIZE,
    fix_title_padding, mrc_copy_valid_extended_type, mrc_get_extended_type, mrc_get_scale,
    mrc_get_standard_version, mrc_head_label_cp, mrc_head_write, mrc_init_output_header,
    mrc_print_label_string, mrc_read_extra_header, mrc_write_extra_header,
};
use crate::imod::libiimod::unit_fileio::{
    iiu_file_type, iiu_get_exit_on_error, iiu_get_ii_file, iiu_mrc_header,
    iiu_sync_with_mrc_header, iiu_trans_adoc_sections,
};
use core::ffi::c_char;
use std::io::Write;

const IIFILE_SHR_MEM: i32 = 8;

/// Matches C `iiuRetBasicHead` (`unit_header.c`).
pub unsafe fn iiu_ret_basic_head(
    iunit: i32,
    nxyz: *mut i32,
    mxyz: *mut i32,
    mode: *mut i32,
    dmin: *mut f32,
    dmax: *mut f32,
    dmean: *mut f32,
) {
    let hdr = unsafe { iiu_mrc_header(iunit, "iiuRetBasicHead", 1, 0) };
    unsafe {
        *nxyz = (*hdr).nx;
        *nxyz.add(1) = (*hdr).ny;
        *nxyz.add(2) = (*hdr).nz;
        *mxyz = (*hdr).mx;
        *mxyz.add(1) = (*hdr).my;
        *mxyz.add(2) = (*hdr).mz;
        *mode = (*hdr).mode;
        *dmin = (*hdr).amin;
        *dmax = (*hdr).amax;
        *dmean = (*hdr).amean;
    }
}

/// Matches C `iiuCreateHeader` (`unit_header.c`).
pub fn iiu_create_header(
    iunit: i32,
    nxyz: &[i32; 3],
    mxyz: &[i32; 3],
    mode: i32,
    labels: &[[u8; MRC_LABEL_SIZE]; MRC_NLABELS],
    num_labels: i32,
) {
    let hdr = unsafe { iiu_mrc_header(iunit, "iiuCreateHeader", 1, 2) };
    unsafe {
        (*hdr).mode = mode;
    }
    if mode == MRC_MODE_FLOAT
        && write_16_bit_mode_for_floats() != 0
        && unsafe { iiu_file_type(iunit) } == IIFILE_MRC
    {
        unsafe {
            (*hdr).half_floats = 1;
            set_or_clear_flags(
                &mut *((&mut (*hdr).iiu_flags as *mut i32).cast::<u32>()),
                IIUNIT_HALF_FLOATS as u32,
                1,
            );
        }
    }
    let nxyzst = [0; 3];
    unsafe {
        iiu_alt_size(iunit, nxyz, &nxyzst);
        iiu_alt_sample(iunit, mxyz);
        (*hdr).xlen = mxyz[0] as f32;
        (*hdr).ylen = mxyz[1] as f32;
        (*hdr).zlen = mxyz[2] as f32;
    }
    iiu_alt_labels(iunit, labels, num_labels);
    unsafe { iiu_sync_with_mrc_header(iunit) };
}

pub fn iiu_write_header(
    iunit: i32,
    label: &[u8; MRC_LABEL_SIZE],
    lab_flag: i32,
    dmin: f32,
    dmax: f32,
    dmean: f32,
) -> i32 {
    let hdr = unsafe { iiu_mrc_header(iunit, "iiuWriteHeader", iiu_get_exit_on_error(), 2) };
    if hdr.is_null() {
        return -1;
    }
    unsafe {
        if lab_flag == 0 {
            iiu_alt_labels(iunit, core::slice::from_ref(label), 1)
        } else if lab_flag == 1 {
            (*hdr).nlabl = ((*hdr).nlabl + 1).min(MRC_NLABELS as i32);
            (*hdr).labels[((*hdr).nlabl - 1) as usize] = *label;
            fix_title_padding(&mut (*hdr).labels[((*hdr).nlabl - 1) as usize]);
        } else if lab_flag == 2 {
            (*hdr).nlabl = ((*hdr).nlabl + 1).min(MRC_NLABELS as i32);
            for i in (1..(*hdr).nlabl as usize).rev() {
                (*hdr).labels[i] = (*hdr).labels[i - 1];
            }
            (*hdr).labels[0] = *label;
            fix_title_padding(&mut (*hdr).labels[0]);
        }
        (*hdr).amin = dmin;
        (*hdr).amax = dmax;
        (*hdr).amean = dmean;
        if mrc_head_write(&mut (*hdr).fp.clone().unwrap(), &mut *hdr) != 0 {
            1
        } else {
            0
        }
    }
}
pub fn iiu_write_header_str(i: i32, label: &str, f: i32, a: f32, b: f32, c: f32) -> i32 {
    let mut out = [0u8; MRC_LABEL_SIZE];
    let bytes = label.as_bytes();
    let count = bytes.len().min(MRC_LABEL_SIZE);
    out[..count].copy_from_slice(&bytes[..count]);
    iiu_write_header(i, &out, f, a, b, c)
}

/// Matches C `iiuwriteheaderstr` (`unit_header.c`).
pub unsafe fn iiuwriteheaderstr(
    iunit: *mut i32,
    label_str: *const c_char,
    lab_flag: *mut i32,
    dmin: *mut f32,
    dmax: *mut f32,
    dmean: *mut f32,
    label_len: i32,
) -> i32 {
    if label_len < 0 || label_str.is_null() {
        return -1;
    }
    let source = unsafe { core::slice::from_raw_parts(label_str.cast::<u8>(), label_len as usize) };
    let trimmed = source
        .iter()
        .rposition(|&byte| byte != b' ')
        .map_or(&[][..], |last| &source[..=last]);
    let label = String::from_utf8_lossy(trimmed);
    unsafe { iiu_write_header_str(*iunit, &label, *lab_flag, *dmin, *dmax, *dmean) }
}

/// Matches C `iwrhdrc` (`unit_header.c`).
pub unsafe fn iwrhdrc(
    iunit: *mut i32,
    label_str: *const c_char,
    lab_flag: *mut i32,
    dmin: *mut f32,
    dmax: *mut f32,
    dmean: *mut f32,
    label_len: i32,
) {
    unsafe {
        iiuwriteheaderstr(iunit, label_str, lab_flag, dmin, dmax, dmean, label_len);
    }
}

/// Matches C `iiuPrintHeader` (`unit_header.c`).
pub unsafe fn iiu_print_header(iunit: i32, file_prefix: *const c_char) {
    let hdr = unsafe { iiu_mrc_header(iunit, "iiuPrintHeader", 1, 0) };
    unsafe {
        if !file_prefix.is_null() {
            let ii_file = iiu_get_ii_file(iunit).cast::<ImodImageFile>();
            // The prefix is an ABI C string, while the filename is owned Rust
            // text.  The centralized formatter preserves the source's `%s`
            // behavior without a varargs call or a temporary C string.
            let prefix = core::ffi::CStr::from_ptr(file_prefix).to_bytes();
            let name = (*ii_file).filename.as_deref().unwrap_or_default();
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "%s: %s\n",
                &[CArg::Bytes(prefix), CArg::Str(name)],
            ));
        }
        let (xscale, yscale, zscale) = mrc_get_scale(&*hdr);
        let mode = if (*hdr).half_floats != 0 && (*hdr).mode == MRC_MODE_FLOAT {
            MRC_MODE_HALF_FLOAT
        } else {
            (*hdr).mode
        };
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            " Dimensions: %6d %6d %6d   Pixel size:%10.4g %10.4g %10.4g\n\
             Mode: %2d               Min, max, mean: %12.5g %12.5g %12.5g\n",
            &[
                CArg::Int((*hdr).nx.into()),
                CArg::Int((*hdr).ny.into()),
                CArg::Int((*hdr).nz.into()),
                CArg::Dbl(xscale.into()),
                CArg::Dbl(yscale.into()),
                CArg::Dbl(zscale.into()),
                CArg::Int(mode.into()),
                CArg::Dbl((*hdr).amin.into()),
                CArg::Dbl((*hdr).amax.into()),
                CArg::Dbl((*hdr).amean.into()),
            ],
        ));
        for lab_ind in 0..if (*hdr).nlabl > 1 { 2 } else { 1 } {
            mrc_print_label_string(Some(&*hdr), if lab_ind != 0 { (*hdr).nlabl - 1 } else { 0 });
        }
    }
}

pub unsafe fn iiu_trans_header(into_unit: i32, iunit: i32) -> i32 {
    let ih = unsafe { iiu_mrc_header(iunit, "iiuTransHeader", iiu_get_exit_on_error(), 0) };
    let jh = unsafe { iiu_mrc_header(into_unit, "iiuTransHeader", iiu_get_exit_on_error(), 2) };
    if ih.is_null() || jh.is_null() {
        return -1;
    }
    unsafe {
        // `unit_header.c:381-387` saves the output handle, copies the whole
        // input header over it, then puts the handle back.  The block copy
        // duplicates every field bitwise, so the saved handle is written back
        // with `ptr::write` rather than assigned — assigning would drop the
        // copy the block made without it ever having been owned.
        let fp = (*jh).fp.take();
        *jh = (*ih).clone();
        (*jh).fp = fp;
        mrc_init_output_header(&mut *jh);
        if iiu_file_type(into_unit) != IIFILE_MRC {
            (*jh).half_floats = 0
        }
        if iiu_file_type(into_unit) != IIFILE_MRC && iiu_file_type(into_unit) != IIFILE_TIFF {
            (*jh).packed4bits = 0
        }
        iiu_sync_with_mrc_header(into_unit);
        iiu_trans_extended_data(into_unit, iunit)
    }
}

pub fn iiu_ret_cell(i: i32, c: &mut [f32; 6]) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetCell", 1, 0) };
    unsafe {
        *c = [
            (*h).xlen,
            (*h).ylen,
            (*h).zlen,
            (*h).alpha,
            (*h).beta,
            (*h).gamma,
        ];
    }
}
pub fn iiu_alt_cell(i: i32, c: &[f32; 6]) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltCell", 1, 0) };
    unsafe {
        (
            (*h).xlen,
            (*h).ylen,
            (*h).zlen,
            (*h).alpha,
            (*h).beta,
            (*h).gamma,
        ) = (c[0], c[1], c[2], c[3], c[4], c[5]);
    }
}
pub fn iiu_ret_data_type(
    i: i32,
    t: &mut i32,
    l: &mut i32,
    n1: &mut i32,
    n2: &mut i32,
    v1: &mut f32,
    v2: &mut f32,
) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetDataType", 1, 0) };
    unsafe {
        *t = (*h).idtype as i32;
        *l = (*h).lens as i32;
        *n1 = (*h).nd1 as i32;
        *n2 = (*h).nd2 as i32;
        *v1 = 0.01 * (*h).vd1 as f32;
        *v2 = 0.01 * (*h).vd2 as f32;
    }
}
pub fn iiu_alt_data_type(i: i32, t: i32, l: i32, n1: i32, n2: i32, v1: f32, v2: f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetDataType", 1, 0) };
    unsafe {
        (*h).idtype = t as i16;
        (*h).lens = l as i16;
        (*h).nd1 = n1 as i16;
        (*h).nd2 = n2 as i16;
        (*h).vd1 = (v1 * 100.).round() as i16;
        (*h).vd2 = (v2 * 100.).round() as i16;
    }
}

pub fn iiu_ret_size(i: i32, n: &mut [i32; 3], m: &mut [i32; 3], s: &mut [i32; 3]) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetSize", 1, 0) };
    unsafe {
        *n = [(*h).nx, (*h).ny, (*h).nz];
        *m = [(*h).mx, (*h).my, (*h).mz];
        *s = [(*h).nxstart, (*h).nystart, (*h).nzstart];
    }
}
pub fn iiu_alt_size(i: i32, n: &[i32; 3], s: &[i32; 3]) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltSize", 1, 0) };
    unsafe {
        ((*h).nx, (*h).ny, (*h).nz) = (n[0], n[1], n[2]);
        ((*h).nxstart, (*h).nystart, (*h).nzstart) = (s[0], s[1], s[2]);
        iiu_sync_with_mrc_header(i);
    }
}
pub fn iiu_ret_sample(i: i32, m: &mut [i32; 3]) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetSample", 1, 0) };
    unsafe {
        *m = [(*h).mx, (*h).my, (*h).mz];
    }
}
pub fn iiu_alt_sample(i: i32, m: &[i32; 3]) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltSample", 1, 0) };
    unsafe {
        ((*h).mx, (*h).my, (*h).mz) = (m[0], m[1], m[2]);
    }
}
pub unsafe fn iiu_alt_size_samp_cell(i: i32, x: i32, y: i32, z: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltSizeSampCell", 1, 0) };
    unsafe {
        (*h).nx = x;
        (*h).ny = y;
        (*h).nz = z;
        (*h).mx = x;
        (*h).my = y;
        (*h).mz = z;
        (*h).xlen = x as f32;
        (*h).ylen = y as f32;
        (*h).zlen = z as f32;
        (*h).alpha = 90.;
        (*h).beta = 90.;
        (*h).gamma = 90.;
        (*h).nxstart = 0;
        (*h).nystart = 0;
        (*h).nzstart = 0;
        iiu_sync_with_mrc_header(i);
    }
}

pub fn iiu_ret_axis_map(i: i32, p: &mut [i32; 3]) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetAxisMap", 1, 0) };
    unsafe {
        *p = [(*h).mapc, (*h).mapr, (*h).maps];
    }
}
pub fn iiu_alt_axis_map(i: i32, p: &[i32; 3]) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltAxisMap", 1, 0) };
    unsafe {
        ((*h).mapc, (*h).mapr, (*h).maps) = (p[0], p[1], p[2]);
    }
}
pub fn iiu_ret_imod_flags(i: i32, f: &mut i32, im: &mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetImodFlags", 1, 0) };
    unsafe {
        *f = (*h).imod_flags;
        *im = if (*h).imod_stamp == IMOD_MRC_STAMP {
            1
        } else {
            0
        };
    }
}
pub fn iiu_alt_imod_flags(i: i32, f: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltImodFlags", 1, 0) };
    unsafe {
        (*h).imod_flags = f;
    }
}
pub fn iiu_alt_signed(i: i32, f: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltSigned", 1, 0) };
    unsafe {
        (*h).bytes_signed = f;
    }
}
pub fn iiu_ret_mrc_version(i: i32, v: &mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetMRCVersion", 1, 0) };
    unsafe {
        *v = mrc_get_standard_version(Some(&*h));
    }
}
pub fn iiu_alt_mrc_version(i: i32, v: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltMRCVersion", 1, 0) };
    unsafe {
        (*h).nversion = v;
    }
}
pub fn iiu_ret_origin(i: i32, origin: &mut [f32; 3]) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetOrigin", 1, 0) };
    unsafe {
        *origin = [(*h).xorg, (*h).yorg, (*h).zorg];
    }
}
pub fn iiu_alt_origin(i: i32, origin: &[f32; 3]) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltOrigin", 1, 0) };
    unsafe {
        ((*h).xorg, (*h).yorg, (*h).zorg) = (origin[0], origin[1], origin[2]);
    }
}

pub fn iiu_alt_mode(i: i32, mut mode: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltMode", 1, 0) };
    unsafe {
        if mode == MRC_MODE_HALF_FLOAT
            || (mode == MRC_MODE_FLOAT && write_16_bit_mode_for_floats() != 0)
        {
            if iiu_file_type(i) == IIFILE_MRC {
                mode = MRC_MODE_FLOAT;
                (*h).half_floats = 1;
                set_or_clear_flags(
                    &mut *((&mut (*h).iiu_flags as *mut i32).cast::<u32>()),
                    IIUNIT_HALF_FLOATS as u32,
                    1,
                )
            }
        } else if mode == MRC_MODE_FLOAT {
            (*h).half_floats = 0;
            set_or_clear_flags(
                &mut *((&mut (*h).iiu_flags as *mut i32).cast::<u32>()),
                IIUNIT_HALF_FLOATS as u32,
                0,
            )
        }
        (*h).mode = mode;
        iiu_sync_with_mrc_header(i);
    }
}
pub unsafe fn iiu_alt_4_bit_mode(i: i32, d: i32) -> i32 {
    let h = unsafe { iiu_mrc_header(i, "iiuAltMode", 1, 0) };
    unsafe {
        if (*h).mode != MRC_MODE_BYTE {
            return 1;
        }
        if d > 0 && (*h).packed4bits != PACKED_4BIT_MODE {
            if (*h).packed4bits == 0 {
                if (*h).bytes_signed != 0 {
                    return 2;
                }
                if (*h).mx == (*h).nx {
                    (*h).mx *= 2;
                    (*h).xlen *= 2.
                }
                (*h).nx *= 2
            }
            (*h).packed4bits = PACKED_4BIT_MODE;
            set_or_clear_flags(
                &mut *((&mut (*h).iiu_flags as *mut i32).cast::<u32>()),
                IIUNIT_4BIT_MODE as u32,
                1,
            );
            set_or_clear_flags(
                &mut *((&mut (*h).iiu_flags as *mut i32).cast::<u32>()),
                IIUNIT_HALF_XSIZE as u32,
                0,
            );
            return 0;
        }
        if d < 0 && (*h).packed4bits == PACKED_4BIT_MODE {
            if (*h).nx % 2 != 0 {
                return 3;
            }
            (*h).packed4bits = PACKED_HALF_XSIZE;
            set_or_clear_flags(
                &mut *((&mut (*h).iiu_flags as *mut i32).cast::<u32>()),
                IIUNIT_4BIT_MODE as u32,
                0,
            );
            set_or_clear_flags(
                &mut *((&mut (*h).iiu_flags as *mut i32).cast::<u32>()),
                IIUNIT_HALF_XSIZE as u32,
                1,
            );
            return 0;
        }
        -1
    }
}
pub fn iiu_ret_rms(i: i32, v: &mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetRMS", 1, 0) };
    unsafe {
        *v = (*h).rms;
    }
}
pub fn iiu_alt_rms(i: i32, v: f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltRMS", 1, 0) };
    unsafe {
        (*h).rms = v;
    }
}
pub fn iiu_ret_tilt(i: i32, p: &mut [f32; 3]) {
    let h = unsafe { &*iiu_mrc_header(i, "iiuRetTilt", 1, 0) };
    *p = [h.tiltangles[3], h.tiltangles[4], h.tiltangles[5]];
}
pub fn iiu_alt_tilt(i: i32, p: &[f32; 3]) {
    let h = unsafe { &mut *iiu_mrc_header(i, "iiuAltTilt", 1, 0) };
    h.tiltangles[3] = p[0];
    h.tiltangles[4] = p[1];
    h.tiltangles[5] = p[2];
}
pub fn iiu_ret_tilt_orig(i: i32, p: &mut [f32; 3]) {
    let h = unsafe { &*iiu_mrc_header(i, "iiuRetTiltOrig", 1, 0) };
    *p = [h.tiltangles[0], h.tiltangles[1], h.tiltangles[2]];
}
pub fn iiu_alt_tilt_orig(i: i32, p: &[f32; 3]) {
    let h = unsafe { &mut *iiu_mrc_header(i, "iiuAltTiltOrig", 1, 0) };
    h.tiltangles[0] = p[0];
    h.tiltangles[1] = p[1];
    h.tiltangles[2] = p[2];
}
/// Matches C `iiuAltTiltRot` (`unit_header.c`).
pub fn iiu_alt_tilt_rot(iunit: i32, tilt: &[f32; 3]) {
    let mut amat1 = [0.0_f32; 9];
    let mut amat2 = [0.0_f32; 9];
    let mut amat3 = [0.0_f32; 9];
    let hdr = unsafe { &mut *iiu_mrc_header(iunit, "iiuAltTiltRot", 1, 0) };

    /* Convert new and old angles to matrices, then multiply */
    angles_to_matrix(tilt, &mut amat1, 3);
    angles_to_matrix(
        &[hdr.tiltangles[3], hdr.tiltangles[4], hdr.tiltangles[5]],
        &mut amat2,
        3,
    );
    for k in 0..3 {
        for l in 0..3 {
            amat3[l + 3 * k] = 0.;
            for m in 0..3 {
                amat3[l + 3 * k] += amat1[l + 3 * m] * amat2[m + 3 * k];
            }
        }
    }

    /* Convert back to angles */
    let mut result = [0.0; 3];
    icalc_angles(&mut result, &amat3);
    hdr.tiltangles[3..6].copy_from_slice(&result);
}
pub fn iiu_ret_delta(i: i32, p: &mut [f32; 3]) {
    let h = unsafe { &*iiu_mrc_header(i, "iiuRetDelta", 1, 0) };
    p[0] = if h.mx > 0 { h.xlen / h.mx as f32 } else { 1. };
    p[1] = if h.my > 0 { h.ylen / h.my as f32 } else { 1. };
    p[2] = if h.mz > 0 { h.zlen / h.mz as f32 } else { 1. };
}
pub fn iiu_alt_delta(i: i32, p: &[f32; 3]) {
    let h = unsafe { &mut *iiu_mrc_header(i, "iiuAltDelta", 1, 0) };
    h.mx = h.mx.max(1);
    h.xlen = p[0] * h.mx as f32;
    h.my = h.my.max(1);
    h.ylen = p[1] * h.my as f32;
    h.mz = h.mz.max(1);
    h.zlen = p[2] * h.mz as f32;
}
pub fn iiu_ret_space_group(i: i32, p: &mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetSpaceGroup", 1, 0) };
    unsafe {
        *p = (*h).ispg;
    }
}
pub fn iiu_alt_space_group(i: i32, v: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltSpaceGroup", 1, 0) };
    unsafe {
        (*h).ispg = v;
    }
}
pub fn iiu_ret_labels(i: i32, labels: &mut [[u8; MRC_LABEL_SIZE]; MRC_NLABELS], n: &mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetLabels", 1, 0) };
    unsafe {
        *n = (*h).nlabl.clamp(0, MRC_NLABELS as i32);
        for x in 0..*n as usize {
            labels[x] = (*h).labels[x];
        }
    }
}
pub fn iiu_alt_labels(i: i32, labels: &[[u8; MRC_LABEL_SIZE]], n: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltLabels", 1, 0) };
    unsafe {
        (*h).nlabl = n.clamp(0, labels.len().min(MRC_NLABELS) as i32);
        for x in 0..(*h).nlabl as usize {
            (*h).labels[x] = labels[x];
            fix_title_padding(&mut (*h).labels[x]);
        }
        for x in (*h).nlabl as usize..MRC_NLABELS {
            (*h).labels[x] = [0; MRC_LABEL_SIZE];
        }
    }
}
pub fn iiu_trans_labels(into: i32, i: i32) {
    let a = unsafe { iiu_mrc_header(i, "iiuTransLabels", 1, 0) };
    let b = unsafe { iiu_mrc_header(into, "iiuTransLabels", 1, 0) };
    unsafe {
        mrc_head_label_cp(&*a, &mut *b);
    }
}
pub fn iiu_ret_num_extended(i: i32, p: &mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetNumExtended", 1, 0) };
    unsafe {
        *p = (*h).next;
    }
}
pub fn iiu_alt_num_extended(i: i32, n: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltNumExtended", 1, 0) };
    unsafe {
        (*h).next = n;
        (*h).header_size = 1024 + n;
    }
}
pub fn iiu_ret_extended_type(i: i32, values: &mut [i32; 2]) {
    let h = unsafe { iiu_mrc_header(i, "iiRetExtendedType", 1, 0) };
    let mut v = 0;
    let t = unsafe { mrc_get_extended_type(&*h, &mut v) };
    unsafe {
        if t == MRC_EXT_TYPE_NONE || t == MRC_EXT_TYPE_SERI || t == MRC_EXT_TYPE_AGAR {
            *values = [(*h).nint as i32, (*h).nreal as i32]
        } else {
            *values = [-t, v];
        }
    }
}
pub fn iiu_alt_extended_type(i: i32, values: &[i32; 2]) {
    let h = unsafe { iiu_mrc_header(i, "iiAltExtendedType", 1, 0) };
    unsafe {
        ((*h).nint, (*h).nreal) = (values[0] as i16, values[1] as i16);
    }
}
pub fn iiu_ret_header_ext_type(i: i32, values: &mut [i32; 2]) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetHeaderExtType", 1, 0) };
    unsafe {
        values[0] = mrc_get_extended_type(&*h, &mut values[1]);
    }
}
pub fn iiu_ret_extended_data(i: i32, data: &mut Vec<u8>) -> i32 {
    let h = unsafe { iiu_mrc_header(i, "iiRetExtendedData", iiu_get_exit_on_error(), 1) };
    if h.is_null() {
        return -1;
    }
    unsafe {
        if (*h).next == 0 {
            data.clear();
            return 0;
        }
        if mrc_read_extra_header(&mut *h, data) != 0 {
            1
        } else {
            0
        }
    }
}
pub fn iiu_alt_extended_data(i: i32, data: &[u8]) -> i32 {
    let do_exit = unsafe { iiu_get_exit_on_error() };
    let h = unsafe { iiu_mrc_header(i, "iiAltExtendedData", do_exit, 2) };
    if h.is_null() {
        return -1;
    }
    unsafe {
        if iiu_file_type(i) != IIFILE_MRC {
            return 0;
        }
        if (*h).swapped != 0 {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "\nERROR: iiuAltExtendedData - Cannot write extra header data to a \
                 byte-swapped file (unit %d).\n",
                &[CArg::Int(i.into())],
            ));
            if do_exit != 0 {
                std::process::exit(1);
            }
            return -2;
        }
        if data.is_empty() || mrc_write_extra_header(&mut *h, data) != 0 {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "\nERROR: iiAltExtendedData - Writing %d bytes of extended header \
                 data for unit %d.\n",
                &[CArg::Int(data.len() as i64), CArg::Int(i.into())],
            ));
            if do_exit != 0 {
                std::process::exit(1);
            }
            2
        } else {
            0
        }
    }
}
pub fn iiu_trans_extended_data(into: i32, i: i32) -> i32 {
    let a = unsafe { iiu_mrc_header(i, "iiuTransExtendedData", iiu_get_exit_on_error(), 1) };
    let b = unsafe { iiu_mrc_header(into, "iiuTransExtendedData", iiu_get_exit_on_error(), 2) };
    if a.is_null() || b.is_null() {
        return -1;
    }
    unsafe {
        if iiu_file_type(into) == IIFILE_TIFF || iiu_file_type(into) == IIFILE_SHR_MEM {
            return 0;
        }
        if iiu_trans_adoc_sections(into, i) != 0 {
            return 3;
        }
        if (*a).next == 0 {
            iiu_alt_num_extended(into, 0);
            return 0;
        }
        let mut data = Vec::new();
        (*b).nint = (*a).nint;
        (*b).nreal = (*a).nreal;
        mrc_copy_valid_extended_type(&*a, &mut *b);
        let e = iiu_ret_extended_data(i, &mut data);
        let ans = if e != 0 {
            1
        } else {
            iiu_alt_extended_data(into, &data)
        };
        ans
    }
}
pub unsafe fn iiu_trans_valid_ext_type(into: i32, i: i32) {
    unsafe {
        if iiu_file_type(into) == IIFILE_MRC && iiu_file_type(i) == IIFILE_MRC {
            let a = iiu_mrc_header(i, "iiuTransValidExtType", 1, 0);
            let b = iiu_mrc_header(into, "iiuTransValidExtType", 1, 0);
            mrc_copy_valid_extended_type(&*a, &mut *b);
        }
    }
}

// The remaining names are the source's Fortran ABI entry points.  Keeping every
// entry point is important: existing Fortran callers bind these names directly.

#[cfg(test)]
mod tests {
    use super::{
        iiu_alt_delta, iiu_alt_extended_data, iiu_alt_labels, iiu_alt_sample, iiu_alt_size,
        iiu_alt_tilt, iiu_alt_tilt_orig, iiu_create_header, iiu_ret_basic_head, iiu_ret_delta,
        iiu_ret_extended_data, iiu_ret_labels, iiu_ret_sample, iiu_ret_size, iiu_ret_tilt,
        iiu_ret_tilt_orig,
    };
    use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_MODE_FLOAT, MRC_NLABELS};
    use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_open};

    #[test]
    fn creates_and_reports_basic_real_mrc_header() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-unit-header-{}-991.mrc",
            std::process::id()
        ));
        let name = path.to_string_lossy();
        unsafe {
            assert_eq!(iiu_open(991, &name, "NEW"), 0);
            let nxyz = [7, 5, 3];
            let mxyz = [7, 5, 3];
            iiu_create_header(
                991,
                &nxyz,
                &mxyz,
                MRC_MODE_FLOAT,
                &[[0; MRC_LABEL_SIZE]; MRC_NLABELS],
                0,
            );
            let mut out_n = [0; 3];
            let mut out_m = [0; 3];
            let mut out_start = [0; 3];
            iiu_ret_size(991, &mut out_n, &mut out_m, &mut out_start);
            assert_eq!(out_n, nxyz);
            assert_eq!(out_m, mxyz);
            assert_eq!(out_start, [0; 3]);

            let new_size = [9, 6, 4];
            let new_start = [3, 2, 1];
            let new_sample = [18, 12, 8];
            iiu_alt_size(991, &new_size, &new_start);
            iiu_alt_sample(991, &new_sample);
            iiu_ret_size(991, &mut out_n, &mut out_m, &mut out_start);
            assert_eq!(out_n, new_size);
            assert_eq!(out_m, new_sample);
            assert_eq!(out_start, new_start);
            iiu_ret_sample(991, &mut out_m);
            assert_eq!(out_m, new_sample);

            let mut labels = [[0; MRC_LABEL_SIZE]; MRC_NLABELS];
            labels[0][..5].copy_from_slice(b"first");
            labels[1][..6].copy_from_slice(b"second");
            iiu_alt_labels(991, &labels, 2);
            let mut returned_labels = [[0; MRC_LABEL_SIZE]; MRC_NLABELS];
            let mut num_labels = 0;
            iiu_ret_labels(991, &mut returned_labels, &mut num_labels);
            assert_eq!(num_labels, 2);
            assert_eq!(&returned_labels[0][..5], b"first");
            assert_eq!(&returned_labels[1][..6], b"second");
            assert!(returned_labels[0][5..].iter().all(|&byte| byte == b' '));
            iiu_alt_labels(991, &labels[..1], 1);
            returned_labels = [[0; MRC_LABEL_SIZE]; MRC_NLABELS];
            iiu_ret_labels(991, &mut returned_labels, &mut num_labels);
            assert_eq!(num_labels, 1);
            assert_eq!(returned_labels[1], [0; MRC_LABEL_SIZE]);

            let current_tilt = [1.25, -2.5, 3.75];
            let original_tilt = [-4.5, 5.25, -6.0];
            iiu_alt_tilt(991, &current_tilt);
            iiu_alt_tilt_orig(991, &original_tilt);
            let mut out_tilt = [0.0; 3];
            let mut out_original_tilt = [0.0; 3];
            iiu_ret_tilt(991, &mut out_tilt);
            iiu_ret_tilt_orig(991, &mut out_original_tilt);
            assert_eq!(out_tilt, current_tilt);
            assert_eq!(out_original_tilt, original_tilt);

            let delta = [1.5, 2.25, 3.75];
            iiu_alt_delta(991, &delta);
            let mut out_delta = [0.0; 3];
            iiu_ret_delta(991, &mut out_delta);
            assert_eq!(out_delta, delta);

            let extended_data = vec![0x31, 0x42, 0x53, 0x64, 0x75];
            assert_eq!(iiu_alt_extended_data(991, &extended_data), 0);
            let mut returned_extended_data = Vec::new();
            assert_eq!(iiu_ret_extended_data(991, &mut returned_extended_data), 0);
            assert_eq!(returned_extended_data, extended_data);

            let mut mode = -1;
            let mut dmin = 0.;
            let mut dmax = 0.;
            let mut dmean = 0.;
            iiu_ret_basic_head(
                991,
                out_n.as_mut_ptr(),
                out_m.as_mut_ptr(),
                &mut mode,
                &mut dmin,
                &mut dmax,
                &mut dmean,
            );
            assert_eq!(out_n, new_size);
            assert_eq!(out_m, new_sample);
            assert_eq!(mode, MRC_MODE_FLOAT);
            iiu_close(991);
        }
        let _ = std::fs::remove_file(path);
    }
}
