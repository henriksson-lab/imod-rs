//! Translation of `IMOD/libiimod/unit_header.c` and declarations from
//! `IMOD/include/iiunit.h`.
//!
//! The unit table itself belongs to `unit_fileio.c`, which is a later source
//! unit in the ordered port.  Calls to that source unit retain its C ABI here;
//! this file contains the complete header manipulation logic rather than a
//! second, incompatible header table.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::{
    f2c_string, set_or_clear_flags, write_16_bit_mode_for_floats,
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
pub unsafe fn iiuretbasichead(
    iunit: *mut i32,
    nxyz: *mut i32,
    mxyz: *mut i32,
    mode: *mut i32,
    dmin: *mut f32,
    dmax: *mut f32,
    dmean: *mut f32,
) {
    unsafe { iiu_ret_basic_head(*iunit, nxyz, mxyz, mode, dmin, dmax, dmean) }
}

/// Matches C `iiuCreateHeader` (`unit_header.c`).
pub unsafe fn iiu_create_header(
    iunit: i32,
    nxyz: *mut i32,
    mxyz: *mut i32,
    mode: i32,
    labels: *mut i32,
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
        iiu_alt_size(iunit, nxyz, nxyzst.as_ptr() as *mut i32);
        iiu_alt_sample(iunit, mxyz);
        (*hdr).xlen = *mxyz as f32;
        (*hdr).ylen = *mxyz.add(1) as f32;
        (*hdr).zlen = *mxyz.add(2) as f32;
        iiu_alt_labels(iunit, labels, num_labels);
        iiu_sync_with_mrc_header(iunit);
    }
}
pub unsafe fn iiucreateheader(
    iunit: *mut i32,
    nxyz: *mut i32,
    mxyz: *mut i32,
    mode: *mut i32,
    labels: *mut i32,
    num_labels: *mut i32,
) {
    unsafe { iiu_create_header(*iunit, nxyz, mxyz, *mode, labels, *num_labels) }
}
pub unsafe fn icrhdr(
    iunit: *mut i32,
    nxyz: *mut i32,
    mxyz: *mut i32,
    mode: *mut i32,
    labels: *mut i32,
    num_labels: *mut i32,
) {
    unsafe { iiu_create_header(*iunit, nxyz, mxyz, *mode, labels, *num_labels) }
}

pub unsafe fn iiu_write_header(
    iunit: i32,
    label: *mut i32,
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
            iiu_alt_labels(iunit, label, 1)
        } else if lab_flag == 1 {
            (*hdr).nlabl = ((*hdr).nlabl + 1).min(MRC_NLABELS as i32);
            core::ptr::copy_nonoverlapping(
                label.cast::<u8>(),
                (*hdr).labels[((*hdr).nlabl - 1) as usize].as_mut_ptr(),
                MRC_LABEL_SIZE,
            );
            fix_title_padding(&mut (*hdr).labels[((*hdr).nlabl - 1) as usize]);
        } else if lab_flag == 2 {
            (*hdr).nlabl = ((*hdr).nlabl + 1).min(MRC_NLABELS as i32);
            for i in (1..(*hdr).nlabl as usize).rev() {
                (*hdr).labels[i] = (*hdr).labels[i - 1];
            }
            core::ptr::copy_nonoverlapping(
                label.cast::<u8>(),
                (*hdr).labels[0].as_mut_ptr(),
                MRC_LABEL_SIZE,
            );
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
pub unsafe fn iiuwriteheader(
    i: *mut i32,
    l: *mut i32,
    f: *mut i32,
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
) -> i32 {
    unsafe { iiu_write_header(*i, l, *f, *a, *b, *c) }
}
pub unsafe fn iwrhdr(i: *mut i32, l: *mut i32, f: *mut i32, a: *mut f32, b: *mut f32, c: *mut f32) {
    unsafe {
        iiu_write_header(*i, l, *f, *a, *b, *c);
    }
}
pub unsafe fn iiu_write_header_str(
    i: i32,
    label: *const c_char,
    f: i32,
    a: f32,
    b: f32,
    c: f32,
) -> i32 {
    let mut out = [0i32; 20];
    unsafe {
        libc::strncpy(out.as_mut_ptr().cast(), label, MRC_LABEL_SIZE);
        iiu_write_header(i, out.as_mut_ptr(), f, a, b, c)
    }
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
    let label = unsafe { f2c_string(label_str, label_len) };
    if label.is_null() {
        return -1;
    }
    let err = unsafe { iiu_write_header_str(*iunit, label, *lab_flag, *dmin, *dmax, *dmean) };
    unsafe { libc::free(label.cast()) };
    err
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
            // `ImodImageFile.filename` owns its bytes now, so the varargs `%s`
            // takes a terminated copy made here.
            let name = std::ffi::CString::new((*ii_file).filename.clone().unwrap_or_default())
                .unwrap_or_default();
            libc::printf(c"%s: %s\n".as_ptr(), file_prefix, name.as_ptr());
        }
        let (xscale, yscale, zscale) = mrc_get_scale(&*hdr);
        let mode = if (*hdr).half_floats != 0 && (*hdr).mode == MRC_MODE_FLOAT {
            MRC_MODE_HALF_FLOAT
        } else {
            (*hdr).mode
        };
        libc::printf(
            c" Dimensions: %6d %6d %6d   Pixel size:%10.4g %10.4g %10.4g\nMode: %2d               Min, max, mean: %12.5g %12.5g %12.5g\n"
                .as_ptr(),
            (*hdr).nx,
            (*hdr).ny,
            (*hdr).nz,
            xscale as f64,
            yscale as f64,
            zscale as f64,
            mode,
            (*hdr).amin as f64,
            (*hdr).amax as f64,
            (*hdr).amean as f64,
        );
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
pub unsafe fn iiutransheader(i: *mut i32, j: *mut i32) -> i32 {
    unsafe { iiu_trans_header(*i, *j) }
}
pub unsafe fn itrhdr(i: *mut i32, j: *mut i32) {
    unsafe {
        iiu_trans_header(*i, *j);
    }
}

pub unsafe fn iiu_ret_cell(i: i32, c: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetCell", 1, 0) };
    unsafe {
        *c = (*h).xlen;
        *c.add(1) = (*h).ylen;
        *c.add(2) = (*h).zlen;
        *c.add(3) = (*h).alpha;
        *c.add(4) = (*h).beta;
        *c.add(5) = (*h).gamma;
    }
}
pub unsafe fn iiu_alt_cell(i: i32, c: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltCell", 1, 0) };
    unsafe {
        (*h).xlen = *c;
        (*h).ylen = *c.add(1);
        (*h).zlen = *c.add(2);
        (*h).alpha = *c.add(3);
        (*h).beta = *c.add(4);
        (*h).gamma = *c.add(5);
    }
}
pub unsafe fn iiu_ret_data_type(
    i: i32,
    t: *mut i32,
    l: *mut i32,
    n1: *mut i32,
    n2: *mut i32,
    v1: *mut f32,
    v2: *mut f32,
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
pub unsafe fn iiu_alt_data_type(i: i32, t: i32, l: i32, n1: i32, n2: i32, v1: f32, v2: f32) {
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

pub unsafe fn iiu_ret_size(i: i32, n: *mut i32, m: *mut i32, s: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetSize", 1, 0) };
    unsafe {
        *n = (*h).nx;
        *n.add(1) = (*h).ny;
        *n.add(2) = (*h).nz;
        *m = (*h).mx;
        *m.add(1) = (*h).my;
        *m.add(2) = (*h).mz;
        *s = (*h).nxstart;
        *s.add(1) = (*h).nystart;
        *s.add(2) = (*h).nzstart;
    }
}
pub unsafe fn iiu_alt_size(i: i32, n: *mut i32, s: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltSize", 1, 0) };
    unsafe {
        (*h).nx = *n;
        (*h).ny = *n.add(1);
        (*h).nz = *n.add(2);
        (*h).nxstart = *s;
        (*h).nystart = *s.add(1);
        (*h).nzstart = *s.add(2);
        iiu_sync_with_mrc_header(i);
    }
}
pub unsafe fn iiu_ret_sample(i: i32, m: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetSample", 1, 0) };
    unsafe {
        *m = (*h).mx;
        *m.add(1) = (*h).my;
        *m.add(2) = (*h).mz;
    }
}
pub unsafe fn iiu_alt_sample(i: i32, m: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltSample", 1, 0) };
    unsafe {
        (*h).mx = *m;
        (*h).my = *m.add(1);
        (*h).mz = *m.add(2);
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

pub unsafe fn iiu_ret_axis_map(i: i32, p: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetAxisMap", 1, 0) };
    unsafe {
        *p = (*h).mapc;
        *p.add(1) = (*h).mapr;
        *p.add(2) = (*h).maps;
    }
}
pub unsafe fn iiu_alt_axis_map(i: i32, p: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltAxisMap", 1, 0) };
    unsafe {
        (*h).mapc = *p;
        (*h).mapr = *p.add(1);
        (*h).maps = *p.add(2);
    }
}
pub unsafe fn iiu_ret_imod_flags(i: i32, f: *mut i32, im: *mut i32) {
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
pub unsafe fn iiu_alt_imod_flags(i: i32, f: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltImodFlags", 1, 0) };
    unsafe {
        (*h).imod_flags = f;
    }
}
pub unsafe fn iiu_alt_signed(i: i32, f: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltSigned", 1, 0) };
    unsafe {
        (*h).bytes_signed = f;
    }
}
pub unsafe fn iiu_ret_mrc_version(i: i32, v: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetMRCVersion", 1, 0) };
    unsafe {
        *v = mrc_get_standard_version(Some(&*h));
    }
}
pub unsafe fn iiu_alt_mrc_version(i: i32, v: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltMRCVersion", 1, 0) };
    unsafe {
        (*h).nversion = v;
    }
}
pub unsafe fn iiu_ret_origin(i: i32, x: *mut f32, y: *mut f32, z: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetOrigin", 1, 0) };
    unsafe {
        *x = (*h).xorg;
        *y = (*h).yorg;
        *z = (*h).zorg;
    }
}
pub unsafe fn iiu_alt_origin(i: i32, x: f32, y: f32, z: f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltOrigin", 1, 0) };
    unsafe {
        (*h).xorg = x;
        (*h).yorg = y;
        (*h).zorg = z;
    }
}

pub unsafe fn iiu_alt_mode(i: i32, mut mode: i32) {
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
pub unsafe fn iiu_ret_rms(i: i32, v: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetRMS", 1, 0) };
    unsafe {
        *v = (*h).rms;
    }
}
pub unsafe fn iiu_alt_rms(i: i32, v: f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltRMS", 1, 0) };
    unsafe {
        (*h).rms = v;
    }
}
pub unsafe fn iiu_ret_tilt(i: i32, p: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetTilt", 1, 0) };
    unsafe {
        *p = (*h).tiltangles[3];
        *p.add(1) = (*h).tiltangles[4];
        *p.add(2) = (*h).tiltangles[5];
    }
}
pub unsafe fn iiu_alt_tilt(i: i32, p: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltTilt", 1, 0) };
    unsafe {
        (*h).tiltangles[3] = *p;
        (*h).tiltangles[4] = *p.add(1);
        (*h).tiltangles[5] = *p.add(2);
    }
}
pub unsafe fn iiu_ret_tilt_orig(i: i32, p: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetTiltOrig", 1, 0) };
    unsafe {
        *p = (*h).tiltangles[0];
        *p.add(1) = (*h).tiltangles[1];
        *p.add(2) = (*h).tiltangles[2];
    }
}
pub unsafe fn iiu_alt_tilt_orig(i: i32, p: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltTiltOrig", 1, 0) };
    unsafe {
        (*h).tiltangles[0] = *p;
        (*h).tiltangles[1] = *p.add(1);
        (*h).tiltangles[2] = *p.add(2);
    }
}
/// Matches C `iiuAltTiltRot` (`unit_header.c`).
pub unsafe fn iiu_alt_tilt_rot(iunit: i32, tilt: *mut f32) {
    let mut amat1 = [0.0_f32; 9];
    let mut amat2 = [0.0_f32; 9];
    let mut amat3 = [0.0_f32; 9];
    let hdr = unsafe { iiu_mrc_header(iunit, "iiuAltTiltRot", 1, 0) };

    unsafe {
        /* Convert new and old angles to matrices, then multiply */
        angles_to_matrix(&*(tilt as *const [f32; 3]), &mut amat1, 3);
        angles_to_matrix(
            &*((*hdr).tiltangles.as_ptr().add(3) as *const [f32; 3]),
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
        icalc_angles(
            &mut *((*hdr).tiltangles.as_mut_ptr().add(3) as *mut [f32; 3]),
            &amat3,
        );
    }
}
pub unsafe fn iiu_ret_delta(i: i32, p: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetDelta", 1, 0) };
    unsafe {
        *p = if (*h).mx > 0 {
            (*h).xlen / (*h).mx as f32
        } else {
            1.
        };
        *p.add(1) = if (*h).my > 0 {
            (*h).ylen / (*h).my as f32
        } else {
            1.
        };
        *p.add(2) = if (*h).mz > 0 {
            (*h).zlen / (*h).mz as f32
        } else {
            1.
        };
    }
}
pub unsafe fn iiu_alt_delta(i: i32, p: *mut f32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltDelta", 1, 0) };
    unsafe {
        (*h).mx = (*h).mx.max(1);
        (*h).xlen = *p * (*h).mx as f32;
        (*h).my = (*h).my.max(1);
        (*h).ylen = *p.add(1) * (*h).my as f32;
        (*h).mz = (*h).mz.max(1);
        (*h).zlen = *p.add(2) * (*h).mz as f32;
    }
}
pub unsafe fn iiu_ret_space_group(i: i32, p: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetSpaceGroup", 1, 0) };
    unsafe {
        *p = (*h).ispg;
    }
}
pub unsafe fn iiu_alt_space_group(i: i32, v: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltSpaceGroup", 1, 0) };
    unsafe {
        (*h).ispg = v;
    }
}
pub unsafe fn iiu_ret_labels(i: i32, p: *mut i32, n: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetLabels", 1, 0) };
    unsafe {
        *n = (*h).nlabl;
        for x in 0..(*h).nlabl as usize {
            core::ptr::copy_nonoverlapping(
                (*h).labels[x].as_ptr(),
                p.add(20 * x).cast(),
                MRC_LABEL_SIZE,
            )
        }
    }
}
pub unsafe fn iiu_alt_labels(i: i32, p: *mut i32, n: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltLabels", 1, 0) };
    unsafe {
        (*h).nlabl = n.min(MRC_NLABELS as i32);
        for x in 0..(*h).nlabl as usize {
            core::ptr::copy_nonoverlapping(
                p.add(20 * x).cast(),
                (*h).labels[x].as_mut_ptr(),
                MRC_LABEL_SIZE,
            );
            fix_title_padding(&mut (*h).labels[x]);
        }
        for x in (*h).nlabl as usize..MRC_NLABELS {
            (*h).labels[x] = [0; MRC_LABEL_SIZE + 1];
        }
    }
}
pub unsafe fn iiu_trans_labels(into: i32, i: i32) {
    let a = unsafe { iiu_mrc_header(i, "iiuTransLabels", 1, 0) };
    let b = unsafe { iiu_mrc_header(into, "iiuTransLabels", 1, 0) };
    unsafe {
        mrc_head_label_cp(&*a, &mut *b);
    }
}
pub unsafe fn iiu_ret_num_extended(i: i32, p: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetNumExtended", 1, 0) };
    unsafe {
        *p = (*h).next;
    }
}
pub unsafe fn iiu_alt_num_extended(i: i32, n: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuAltNumExtended", 1, 0) };
    unsafe {
        (*h).next = n;
        (*h).header_size = 1024 + n;
    }
}
pub unsafe fn iiu_ret_extended_type(i: i32, a: *mut i32, b: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiRetExtendedType", 1, 0) };
    let mut v = 0;
    let t = unsafe { mrc_get_extended_type(&*h, &mut v) };
    unsafe {
        if t == MRC_EXT_TYPE_NONE || t == MRC_EXT_TYPE_SERI || t == MRC_EXT_TYPE_AGAR {
            *a = (*h).nint as i32;
            *b = (*h).nreal as i32
        } else {
            *a = -t;
            *b = v;
        }
    }
}
pub unsafe fn iiu_alt_extended_type(i: i32, a: i32, b: i32) {
    let h = unsafe { iiu_mrc_header(i, "iiAltExtendedType", 1, 0) };
    unsafe {
        (*h).nint = a as i16;
        (*h).nreal = b as i16;
    }
}
pub unsafe fn iiu_ret_header_ext_type(i: i32, t: *mut i32, v: *mut i32) {
    let h = unsafe { iiu_mrc_header(i, "iiuRetHeaderExtType", 1, 0) };
    unsafe {
        *t = mrc_get_extended_type(&*h, &mut *v);
    }
}
pub unsafe fn iiu_ret_extended_data(i: i32, n: *mut i32, e: *mut i32) -> i32 {
    let h = unsafe { iiu_mrc_header(i, "iiRetExtendedData", iiu_get_exit_on_error(), 1) };
    if h.is_null() {
        return -1;
    }
    unsafe {
        *n = (*h).next;
        if (*h).next == 0 {
            return 0;
        }
        if mrc_read_extra_header(h, &mut e.cast::<u8>()) != 0 {
            1
        } else {
            0
        }
    }
}
unsafe extern "C" {
    #[link_name = "stdout"]
    static mut imod_stdout: *mut libc::FILE;
}

pub unsafe fn iiu_alt_extended_data(i: i32, n: i32, e: *mut i32) -> i32 {
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
            libc::fprintf(
                imod_stdout,
                c"\nERROR: iiuAltExtendedData - Cannot write extra header data to a byte-swapped file (unit %d).\n"
                    .as_ptr(),
                i,
            );
            if do_exit != 0 {
                libc::exit(1);
            }
            return -2;
        }
        if mrc_write_extra_header(h, e.cast(), n) != 0 {
            libc::fprintf(
                imod_stdout,
                c"\nERROR: iiAltExtendedData - Writing %d bytes of extended header data for unit %d.\n"
                    .as_ptr(),
                n,
                i,
            );
            if do_exit != 0 {
                libc::exit(1);
            }
            2
        } else {
            0
        }
    }
}
pub unsafe fn iiu_trans_extended_data(into: i32, i: i32) -> i32 {
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
        let p = libc::malloc(((*a).next as usize + 3) & !3).cast::<i32>();
        if p.is_null() {
            return -1;
        }
        (*b).nint = (*a).nint;
        (*b).nreal = (*a).nreal;
        mrc_copy_valid_extended_type(&*a, &mut *b);
        let mut n = 0;
        let e = iiu_ret_extended_data(i, &mut n, p);
        let ans = if e != 0 {
            1
        } else {
            iiu_alt_extended_data(into, n, p)
        };
        libc::free(p.cast());
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
pub unsafe fn iiuretcell(i: *mut i32, c: *mut f32) {
    unsafe { iiu_ret_cell(*i, c) }
}
pub unsafe fn irtcel(i: *mut i32, c: *mut f32) {
    unsafe { iiu_ret_cell(*i, c) }
}
pub unsafe fn iiualtcell(i: *mut i32, c: *mut f32) {
    unsafe { iiu_alt_cell(*i, c) }
}
pub unsafe fn ialcel(i: *mut i32, c: *mut f32) {
    unsafe { iiu_alt_cell(*i, c) }
}
pub unsafe fn iiuretdatatype(
    i: *mut i32,
    t: *mut i32,
    l: *mut i32,
    n1: *mut i32,
    n2: *mut i32,
    v1: *mut f32,
    v2: *mut f32,
) {
    unsafe { iiu_ret_data_type(*i, t, l, n1, n2, v1, v2) }
}
pub unsafe fn irtdat(
    i: *mut i32,
    t: *mut i32,
    l: *mut i32,
    n1: *mut i32,
    n2: *mut i32,
    v1: *mut f32,
    v2: *mut f32,
) {
    unsafe { iiu_ret_data_type(*i, t, l, n1, n2, v1, v2) }
}
pub unsafe fn iiualtdatatype(
    i: *mut i32,
    t: *mut i32,
    l: *mut i32,
    n1: *mut i32,
    n2: *mut i32,
    v1: *mut f32,
    v2: *mut f32,
) {
    unsafe { iiu_alt_data_type(*i, *t, *l, *n1, *n2, *v1, *v2) }
}
pub unsafe fn ialdat(
    i: *mut i32,
    t: *mut i32,
    l: *mut i32,
    n1: *mut i32,
    n2: *mut i32,
    v1: *mut f32,
    v2: *mut f32,
) {
    unsafe { iiu_alt_data_type(*i, *t, *l, *n1, *n2, *v1, *v2) }
}
pub unsafe fn iiuretsize(i: *mut i32, n: *mut i32, m: *mut i32, s: *mut i32) {
    unsafe { iiu_ret_size(*i, n, m, s) }
}
pub unsafe fn irtsiz(i: *mut i32, n: *mut i32, m: *mut i32, s: *mut i32) {
    unsafe { iiu_ret_size(*i, n, m, s) }
}
pub unsafe fn iiualtsize(i: *mut i32, n: *mut i32, s: *mut i32) {
    unsafe { iiu_alt_size(*i, n, s) }
}
pub unsafe fn ialsiz(i: *mut i32, n: *mut i32, s: *mut i32) {
    unsafe { iiu_alt_size(*i, n, s) }
}
pub unsafe fn iiuretsample(i: *mut i32, m: *mut i32) {
    unsafe { iiu_ret_sample(*i, m) }
}
pub unsafe fn irtsam(i: *mut i32, m: *mut i32) {
    unsafe { iiu_ret_sample(*i, m) }
}
pub unsafe fn iiualtsample(i: *mut i32, m: *mut i32) {
    unsafe { iiu_alt_sample(*i, m) }
}
pub unsafe fn ialsam(i: *mut i32, m: *mut i32) {
    unsafe { iiu_alt_sample(*i, m) }
}
pub unsafe fn iiualtsizesampcell(i: *mut i32, x: *mut i32, y: *mut i32, z: *mut i32) {
    unsafe { iiu_alt_size_samp_cell(*i, *x, *y, *z) }
}
pub unsafe fn ialsiz_sam_cel(i: *mut i32, x: *mut i32, y: *mut i32, z: *mut i32) {
    unsafe { iiu_alt_size_samp_cell(*i, *x, *y, *z) }
}
pub unsafe fn iiuretaxismap(i: *mut i32, p: *mut i32) {
    unsafe { iiu_ret_axis_map(*i, p) }
}
pub unsafe fn irtmap(i: *mut i32, p: *mut i32) {
    unsafe { iiu_ret_axis_map(*i, p) }
}
pub unsafe fn iiualtaxismap(i: *mut i32, p: *mut i32) {
    unsafe { iiu_alt_axis_map(*i, p) }
}
pub unsafe fn ialmap(i: *mut i32, p: *mut i32) {
    unsafe { iiu_alt_axis_map(*i, p) }
}
pub unsafe fn iiuretimodflags(i: *mut i32, a: *mut i32, b: *mut i32) {
    unsafe { iiu_ret_imod_flags(*i, a, b) }
}
pub unsafe fn irtimodflags(i: *mut i32, a: *mut i32, b: *mut i32) {
    unsafe { iiu_ret_imod_flags(*i, a, b) }
}
pub unsafe fn iiualtimodflags(i: *mut i32, a: *mut i32) {
    unsafe { iiu_alt_imod_flags(*i, *a) }
}
pub unsafe fn ialimodflags(i: *mut i32, a: *mut i32) {
    unsafe { iiu_alt_imod_flags(*i, *a) }
}
pub unsafe fn iiualtsigned(i: *mut i32, a: *mut i32) {
    unsafe { iiu_alt_signed(*i, *a) }
}
pub unsafe fn ialsigned(i: *mut i32, a: *mut i32) {
    unsafe { iiu_alt_signed(*i, *a) }
}
pub unsafe fn iiuretmrcversion(i: *mut i32, v: *mut i32) {
    unsafe { iiu_ret_mrc_version(*i, v) }
}
pub unsafe fn iiualtmrcversion(i: *mut i32, v: *mut i32) {
    unsafe { iiu_alt_mrc_version(*i, *v) }
}
pub unsafe fn iiuretorigin(i: *mut i32, x: *mut f32, y: *mut f32, z: *mut f32) {
    unsafe { iiu_ret_origin(*i, x, y, z) }
}
pub unsafe fn irtorg(i: *mut i32, x: *mut f32, y: *mut f32, z: *mut f32) {
    unsafe { iiu_ret_origin(*i, x, y, z) }
}
pub unsafe fn iiualtorigin(i: *mut i32, x: *mut f32, y: *mut f32, z: *mut f32) {
    unsafe { iiu_alt_origin(*i, *x, *y, *z) }
}
pub unsafe fn ialorg(i: *mut i32, x: *mut f32, y: *mut f32, z: *mut f32) {
    unsafe { iiu_alt_origin(*i, *x, *y, *z) }
}
pub unsafe fn ialmod(i: *mut i32, m: *mut i32) {
    unsafe { iiu_alt_mode(*i, *m) }
}
pub unsafe fn iiualtmode(i: *mut i32, m: *mut i32) {
    unsafe { iiu_alt_mode(*i, *m) }
}
pub unsafe fn iiuretrms(i: *mut i32, v: *mut f32) {
    unsafe { iiu_ret_rms(*i, v) }
}
pub unsafe fn irtrms(i: *mut i32, v: *mut f32) {
    unsafe { iiu_ret_rms(*i, v) }
}
pub unsafe fn iiualtrms(i: *mut i32, v: *mut f32) {
    unsafe { iiu_alt_rms(*i, *v) }
}
pub unsafe fn ialrms(i: *mut i32, v: *mut f32) {
    unsafe { iiu_alt_rms(*i, *v) }
}
pub unsafe fn iiurettilt(i: *mut i32, p: *mut f32) {
    unsafe { iiu_ret_tilt(*i, p) }
}
pub unsafe fn irttlt(i: *mut i32, p: *mut f32) {
    unsafe { iiu_ret_tilt(*i, p) }
}
pub unsafe fn iiualttilt(i: *mut i32, p: *mut f32) {
    unsafe { iiu_alt_tilt(*i, p) }
}
pub unsafe fn ialtlt(i: *mut i32, p: *mut f32) {
    unsafe { iiu_alt_tilt(*i, p) }
}
pub unsafe fn iiurettiltorig(i: *mut i32, p: *mut f32) {
    unsafe { iiu_ret_tilt_orig(*i, p) }
}
pub unsafe fn irttlt_orig(i: *mut i32, p: *mut f32) {
    unsafe { iiu_ret_tilt_orig(*i, p) }
}
pub unsafe fn iiualttiltorig(i: *mut i32, p: *mut f32) {
    unsafe { iiu_alt_tilt_orig(*i, p) }
}
pub unsafe fn ialtlt_orig(i: *mut i32, p: *mut f32) {
    unsafe { iiu_alt_tilt_orig(*i, p) }
}
pub unsafe fn iiualttiltrot(i: *mut i32, p: *mut f32) {
    unsafe { iiu_alt_tilt_rot(*i, p) }
}
pub unsafe fn ialtlt_rot(i: *mut i32, p: *mut f32) {
    unsafe { iiu_alt_tilt_rot(*i, p) }
}
pub unsafe fn iiuretdelta(i: *mut i32, p: *mut f32) {
    unsafe { iiu_ret_delta(*i, p) }
}
pub unsafe fn irtdel(i: *mut i32, p: *mut f32) {
    unsafe { iiu_ret_delta(*i, p) }
}
pub unsafe fn iiualtdelta(i: *mut i32, p: *mut f32) {
    unsafe { iiu_alt_delta(*i, p) }
}
pub unsafe fn ialdel(i: *mut i32, p: *mut f32) {
    unsafe { iiu_alt_delta(*i, p) }
}
pub unsafe fn iiuretspacegroup(i: *mut i32, p: *mut i32) {
    unsafe { iiu_ret_space_group(*i, p) }
}
pub unsafe fn iiualtspacegroup(i: *mut i32, p: *mut i32) {
    unsafe { iiu_alt_space_group(*i, *p) }
}
pub unsafe fn iiuretlabels(i: *mut i32, p: *mut i32, n: *mut i32) {
    unsafe { iiu_ret_labels(*i, p, n) }
}
pub unsafe fn irtlab(i: *mut i32, p: *mut i32, n: *mut i32) {
    unsafe { iiu_ret_labels(*i, p, n) }
}
pub unsafe fn iiualtlabels(i: *mut i32, p: *mut i32, n: *mut i32) {
    unsafe { iiu_alt_labels(*i, p, *n) }
}
pub unsafe fn iallab(i: *mut i32, p: *mut i32, n: *mut i32) {
    unsafe { iiu_alt_labels(*i, p, *n) }
}
pub unsafe fn iiutranslabels(i: *mut i32, j: *mut i32) {
    unsafe { iiu_trans_labels(*i, *j) }
}
pub unsafe fn itrlab(i: *mut i32, j: *mut i32) {
    unsafe { iiu_trans_labels(*i, *j) }
}
pub unsafe fn iiuretnumextended(i: *mut i32, n: *mut i32) {
    unsafe { iiu_ret_num_extended(*i, n) }
}
pub unsafe fn irtnbsym(i: *mut i32, n: *mut i32) {
    unsafe { iiu_ret_num_extended(*i, n) }
}
pub unsafe fn iiualtnumextended(i: *mut i32, n: *mut i32) {
    unsafe { iiu_alt_num_extended(*i, *n) }
}
pub unsafe fn ialnbsym(i: *mut i32, n: *mut i32) {
    unsafe { iiu_alt_num_extended(*i, *n) }
}
pub unsafe fn iiuretextendedtype(i: *mut i32, a: *mut i32, b: *mut i32) {
    unsafe { iiu_ret_extended_type(*i, a, b) }
}
pub unsafe fn irtsymtyp(i: *mut i32, a: *mut i32, b: *mut i32) {
    unsafe { iiu_ret_extended_type(*i, a, b) }
}
pub unsafe fn iiualtextendedtype(i: *mut i32, a: *mut i32, b: *mut i32) {
    unsafe { iiu_alt_extended_type(*i, *a, *b) }
}
pub unsafe fn ialsymtyp(i: *mut i32, a: *mut i32, b: *mut i32) {
    unsafe { iiu_alt_extended_type(*i, *a, *b) }
}
pub unsafe fn iiuretheaderexttype(i: *mut i32, a: *mut i32, b: *mut i32) {
    unsafe { iiu_ret_header_ext_type(*i, a, b) }
}
pub unsafe fn irtsym(i: *mut i32, n: *mut i32, e: *mut i32) {
    unsafe {
        let _ = iiu_ret_extended_data(*i, n, e);
    }
}
pub unsafe fn ialsym(i: *mut i32, n: *mut i32, e: *mut i32) {
    unsafe {
        let _ = iiu_alt_extended_data(*i, *n, e);
    }
}
pub unsafe fn itrextra(i: *mut i32, j: *mut i32) {
    unsafe {
        let _ = iiu_trans_extended_data(*i, *j);
    }
}
pub unsafe fn iiutransvalidexttype(i: *mut i32, j: *mut i32) {
    unsafe { iiu_trans_valid_ext_type(*i, *j) }
}

#[cfg(test)]
mod tests {
    use super::{iiu_create_header, iiu_ret_basic_head};
    use crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT;
    use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_open};
    use std::ffi::CString;

    #[test]
    fn creates_and_reports_basic_real_mrc_header() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-unit-header-{}-991.mrc",
            std::process::id()
        ));
        let name = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        unsafe {
            assert_eq!(iiu_open(991, name.as_ptr(), c"NEW".as_ptr()), 0);
            let mut nxyz = [7, 5, 3];
            let mut mxyz = [7, 5, 3];
            iiu_create_header(
                991,
                nxyz.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                MRC_MODE_FLOAT,
                core::ptr::null_mut(),
                0,
            );
            let mut out_n = [0; 3];
            let mut out_m = [0; 3];
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
            assert_eq!(out_n, nxyz);
            assert_eq!(out_m, mxyz);
            assert_eq!(mode, MRC_MODE_FLOAT);
            iiu_close(991);
        }
        let _ = std::fs::remove_file(path);
    }
}
