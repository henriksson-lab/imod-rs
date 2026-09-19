//! Translation of `IMOD/libiimod/mrcsec.c`.
//!
//! This unit is the next bottom-up dependency of `iimrc.c`.  Each entry below
//! is one source function rendered in systematic Rust snake case.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{
    SEEK_CUR, SEEK_SET, b3d_error, b3d_fread, b3d_fseek, b3d_fwrite, b3d_i_max, mrc_huge_seek,
};
use crate::imod::libiimod::halffloat::{imnp_halfbits_to_floatbits, imnp_halfbuf_to_floats};
use crate::imod::libiimod::iimage::IIERR_QUITTING;
use crate::imod::libiimod::iimage::{
    IIFILE_MRC, IIFILE_RAW, IiSectionFunc, ImodImageFile, LineProcData, ii_calling_read_or_write,
    ii_lookup_file_from_fp, ii_read_section_byte_callback, ii_read_section_callback,
    ii_read_section_float_callback, ii_read_section_ushort_callback, ii_save_load_params,
    ii_sync_from_mrc_header,
};
use crate::imod::libiimod::mrcfiles::{LoadInfo, MrcHeader};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MRC_MODE_RGB,
    MRC_MODE_SHORT, MRC_MODE_USHORT, get_byte_map, get_short_map, mrc_get_complex_scale,
    mrc_getdcsize, mrc_mirror_source, mrc_swap_floats, mrc_swap_shorts,
};

const MRSA_BYTE: i32 = 1;
const MRSA_FLOAT: i32 = 2;
const MRSA_USHORT: i32 = 3;
const MRC_RAMP_EXP: i32 = 2;
const MRC_RAMP_LOG: i32 = 3;

pub fn mrc_read_z(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [u8], z: i32) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = li.ymax - li.ymin + 1;
    let mut bytes = 0;
    let mut channels = 0;
    if width < 0
        || rows < 0
        || mrc_getdcsize(hdata.mode, &mut bytes, &mut channels) != 0
        || buf.len()
            < width as usize
                * rows as usize
                * if hdata.half_floats != 0 && hdata.mode == MRC_MODE_FLOAT {
                    2
                } else {
                    (bytes * channels) as usize
                }
    {
        return -1;
    }
    call_ii_or_mrsa(hdata, li, buf, z, 0, 0, Some(ii_read_section_callback))
}
pub fn mrc_read_z_byte(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [u8], z: i32) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = li.ymax - li.ymin + 1;
    if width < 0 || rows < 0 || buf.len() < width as usize * rows as usize {
        return -1;
    }
    call_ii_or_mrsa(hdata, li, buf, z, 0, 1, Some(ii_read_section_byte_callback))
}
pub fn mrc_read_z_ushort(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [u16], z: i32) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = li.ymax - li.ymin + 1;
    if width < 0 || rows < 0 || buf.len() < width as usize * rows as usize {
        return -1;
    }
    call_ii_or_mrsa(
        hdata,
        li,
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), buf.len() * 2) },
        z,
        0,
        3,
        Some(ii_read_section_ushort_callback),
    )
}
pub fn mrc_read_z_float(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [f32], z: i32) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = li.ymax - li.ymin + 1;
    if width < 0 || rows < 0 || buf.len() < width as usize * rows as usize {
        return -1;
    }
    call_ii_or_mrsa(
        hdata,
        li,
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), buf.len() * 4) },
        z,
        0,
        2,
        Some(ii_read_section_float_callback),
    )
}
pub fn mrc_read_y(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [u8], z: i32) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = li.zmax - li.zmin + 1;
    let mut bytes = 0;
    let mut channels = 0;
    if width < 0
        || rows < 0
        || mrc_getdcsize(hdata.mode, &mut bytes, &mut channels) != 0
        || buf.len()
            < width as usize
                * rows as usize
                * if hdata.half_floats != 0 && hdata.mode == MRC_MODE_FLOAT {
                    2
                } else {
                    (bytes * channels) as usize
                }
    {
        return -1;
    }
    call_ii_or_mrsa(hdata, li, buf, z, 1, 0, Some(ii_read_section_callback))
}
pub fn mrc_read_y_byte(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [u8], z: i32) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = li.zmax - li.zmin + 1;
    if width < 0 || rows < 0 || buf.len() < width as usize * rows as usize {
        return -1;
    }
    call_ii_or_mrsa(hdata, li, buf, z, 1, 1, Some(ii_read_section_byte_callback))
}
pub fn mrc_read_y_ushort(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [u16], z: i32) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = li.zmax - li.zmin + 1;
    if width < 0 || rows < 0 || buf.len() < width as usize * rows as usize {
        return -1;
    }
    call_ii_or_mrsa(
        hdata,
        li,
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), buf.len() * 2) },
        z,
        1,
        3,
        Some(ii_read_section_ushort_callback),
    )
}
pub fn mrc_read_y_float(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [f32], z: i32) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = li.zmax - li.zmin + 1;
    if width < 0 || rows < 0 || buf.len() < width as usize * rows as usize {
        return -1;
    }
    call_ii_or_mrsa(
        hdata,
        li,
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), buf.len() * 4) },
        z,
        1,
        2,
        Some(ii_read_section_float_callback),
    )
}
pub fn mrc_read_section(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [u8], z: i32) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    };
    let mut bytes = 0;
    let mut channels = 0;
    if width < 0
        || rows < 0
        || mrc_getdcsize(hdata.mode, &mut bytes, &mut channels) != 0
        || buf.len()
            < width as usize
                * rows as usize
                * if hdata.half_floats != 0 && hdata.mode == MRC_MODE_FLOAT {
                    2
                } else {
                    (bytes * channels) as usize
                }
    {
        return -1;
    }
    call_ii_or_mrsa(
        hdata,
        li,
        buf,
        z,
        if li.axis == 2 { 1 } else { 0 },
        0,
        Some(ii_read_section_callback),
    )
}
pub fn mrc_read_section_byte(
    hdata: &mut MrcHeader,
    li: &mut LoadInfo,
    buf: &mut [u8],
    z: i32,
) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    };
    if width < 0 || rows < 0 || buf.len() < width as usize * rows as usize {
        return -1;
    }
    call_ii_or_mrsa(
        hdata,
        li,
        buf,
        z,
        if li.axis == 2 { 1 } else { 0 },
        1,
        Some(ii_read_section_byte_callback),
    )
}
pub fn mrc_read_section_ushort(
    hdata: &mut MrcHeader,
    li: &mut LoadInfo,
    buf: &mut [u16],
    z: i32,
) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    };
    if width < 0 || rows < 0 || buf.len() < width as usize * rows as usize {
        return -1;
    }
    call_ii_or_mrsa(
        hdata,
        li,
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), buf.len() * 2) },
        z,
        if li.axis == 2 { 1 } else { 0 },
        3,
        Some(ii_read_section_ushort_callback),
    )
}
pub fn mrc_read_section_float(
    hdata: &mut MrcHeader,
    li: &mut LoadInfo,
    buf: &mut [f32],
    z: i32,
) -> i32 {
    let width = li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0);
    let rows = if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    };
    if width < 0 || rows < 0 || buf.len() < width as usize * rows as usize {
        return -1;
    }
    call_ii_or_mrsa(
        hdata,
        li,
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), buf.len() * 4) },
        z,
        if li.axis == 2 { 1 } else { 0 },
        2,
        Some(ii_read_section_float_callback),
    )
}
pub fn call_ii_or_mrsa(
    hdata: &mut MrcHeader,
    li: &mut LoadInfo,
    buf: &mut [u8],
    z: i32,
    read_y: i32,
    type_: i32,
    func: IiSectionFunc,
) -> i32 {
    let mut ii_save = ImodImageFile::default();
    let ii_file =
        unsafe { lookup_ii_file(hdata, li, if read_y != 0 { 2 } else { 3 }, &mut ii_save) };
    if !ii_file.is_null() {
        let Some(func) = func else {
            return -1;
        };
        return unsafe {
            crate::imod::libiimod::iimage::ii_restore_load_params(
                func(ii_file, buf.as_mut_ptr().cast(), z),
                &mut *ii_file,
                &ii_save,
            )
        };
    }
    mrc_read_section_any(hdata, li, buf, z, read_y, type_)
}
pub fn mrc_read_section_any(
    hdata: &mut MrcHeader,
    li: &mut LoadInfo,
    buf: &mut [u8],
    cz: i32,
    read_y: i32,
    mut type_: i32,
) -> i32 {
    let mut fin = hdata.fp.clone().unwrap();
    let mut d = LineProcData::default();
    let mut y_end = if read_y != 0 { li.zmax } else { li.ymax };
    if type_ == 0 && hdata.mode == MRC_MODE_FLOAT && hdata.half_floats != 0 {
        type_ = MRSA_FLOAT;
    }
    d.type_ = type_;
    d.read_y = read_y;
    d.cz = cz;
    d.swapped = hdata.swapped;
    let init = unsafe {
        ii_init_read_section_any(
            hdata,
            li,
            buf.as_mut_ptr(),
            &mut d,
            &mut y_end,
            "mrcReadSectionAny",
        )
    };
    if init != 0 {
        return init;
    }
    let pad_left = li.pad_left.max(0);
    let pad_right = li.pad_right.max(0);
    d.x_dimension = d.xsize + pad_left + pad_right;
    let mut pix_size_buf = [0, 1, 4, 2];
    // `mrcsec.c:230`: entry 0 (MRSA_NOPROC) is filled in from the pixel size of
    // the data actually in the buffer.  Leaving it at 0 makes the inverted-Y
    // setup below fail to advance `bufp` to the last line, so the first line is
    // written before the start of the caller's buffer.
    pix_size_buf[0] = d.pix_size;
    // `mrcsec.c:241-242`: need to buffer if there is padding in X.
    if d.x_dimension > d.xsize {
        d.need_data = 1;
    }
    if hdata.y_inverted != 0 {
        if li.mirror_fft != 0 {
            // `mrcsec.c:264-267`: this combination has no source transform.
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: mrcReadSectionAny - cannot mirror FFT with inverted Y data.\n"
                ),
            );
            return 1;
        }
        let ny = hdata.ny;
        let old_start = d.y_start;
        d.y_start = ny - 1 - y_end;
        y_end = ny - 1 - old_start;
        let lines = (y_end - d.y_start) as isize * d.x_dimension as isize;
        unsafe {
            d.bdata = d.bdata.offset(lines * d.pix_size as isize);
            d.pix_index = d.pix_index.wrapping_add(lines as u32);
            d.bufp_offset += lines * pix_size_buf[type_ as usize] as isize;
        }
        d.delta_y_sign = -1;
    }
    d.bufp_offset += (pix_size_buf[type_ as usize] * pad_left) as isize;
    d.pix_index = d.pix_index.wrapping_add(pad_left as u32);
    let nx = hdata.nx;
    let ny = hdata.ny;
    let nx_seek = if d.packed4bits != 0 { (nx + 1) / 2 } else { nx };
    let target_lines = (2_000_000_f32 / (d.pix_size * nx_seek) as f32 + 0.5) as i32;
    let mut chunk_lines = 1_i32;
    if read_y == 0
        && !(d.convert != 0 && li.mirror_fft != 0)
        && hdata.y_inverted == 0
        && d.xsize as f32 / nx as f32 > 0.5
        && (target_lines > 1 || (d.need_data == 0 && nx == d.xsize))
    {
        if nx > d.xsize {
            d.need_data = 1;
        }
        chunk_lines = (y_end + 1 - d.y_start).min((100_000_000 / nx_seek).max(1));
        if d.need_data != 0 {
            chunk_lines = chunk_lines.min(target_lines).max(1);
        }
    }
    let mut temporary_data = if d.need_data != 0 {
        let bytes = d.pix_size as usize
            * if chunk_lines > 1 {
                (nx_seek * chunk_lines) as usize
            } else if d.packed4bits != 0 {
                ((d.xsize + 4) / 2) as usize
            } else {
                d.xsize as usize
            };
        let mut data = Vec::new();
        if data.try_reserve_exact(bytes).is_err() {
            return 2;
        }
        data.resize(bytes, 0);
        Some(data)
    } else {
        None
    };
    let seek_line = if d.packed4bits != 0 {
        d.x_start / 2
    } else {
        d.x_start * d.pix_size
    };
    let seek_end_x = if d.packed4bits != 0 {
        (nx_seek - (d.x_end + 2) / 2).max(0)
    } else {
        (nx - d.x_end - 1).max(0)
    };
    // `mrcsec.c:320-332`: seekEndY is set for the axis being read before the
    // loop; iiProcessReadLine may then retarget it for FFT mirroring.
    let seek_skip = if read_y != 0 { hdata.section_skip } else { 0 };
    let seek_error = if read_y != 0 {
        d.seek_end_y = ny - 1;
        unsafe {
            mrc_huge_seek(
                &mut fin,
                hdata.header_size + hdata.section_skip * d.y_start,
                0,
                d.cz,
                d.y_start,
                nx_seek,
                ny,
                d.pix_size,
                SEEK_SET,
            )
        }
    } else {
        d.seek_end_y = 0;
        unsafe {
            mrc_huge_seek(
                &mut fin,
                hdata.header_size + hdata.section_skip * d.cz,
                0,
                d.y_start,
                d.cz,
                nx_seek,
                ny,
                d.pix_size,
                SEEK_SET,
            )
        }
    };
    if seek_error != 0 {
        return 3;
    }
    d.line = d.y_start;
    while d.line <= y_end {
        if chunk_lines > 1 {
            let line_end = y_end.min(d.line + chunk_lines - 1);
            let lines = line_end + 1 - d.line;
            let chunk_start = temporary_data.as_deref_mut().map_or_else(
                || unsafe { d.buf.offset(d.bufp_offset) },
                |data| data.as_mut_ptr(),
            );
            if b3d_fread(
                unsafe {
                    core::slice::from_raw_parts_mut(
                        chunk_start.cast::<u8>(),
                        d.pix_size as usize * (lines * nx_seek) as usize,
                    )
                },
                d.pix_size as usize,
                (lines * nx_seek) as usize,
                &mut fin,
            ) != (lines * nx_seek) as usize
            {
                // `mrcsec.c:367`.
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!("ERROR: mrcReadSectionAny - reading data from file.\n"),
                );
                return 3;
            }
            d.bdata = unsafe { chunk_start.add(seek_line as usize) };
            while d.line <= line_end {
                if unsafe {
                    ii_process_read_line(
                        hdata,
                        li,
                        &mut d,
                        core::ptr::null_mut(),
                        core::ptr::null_mut(),
                    )
                } != 0
                {
                    return IIERR_QUITTING;
                }
                d.bdata = unsafe { d.bdata.add((nx_seek * d.pix_size) as usize) };
                d.line += 1;
            }
            continue;
        }
        if seek_line != 0 && b3d_fseek(&mut fin, seek_line, SEEK_CUR) != 0 {
            break;
        }
        let bdata = temporary_data.as_deref_mut().map_or_else(
            || unsafe { d.buf.offset(d.bufp_offset) },
            |data| data.as_mut_ptr(),
        );
        let count = if d.packed4bits != 0 {
            ((d.x_end + 2) / 2 - d.x_start / 2) as usize
        } else {
            d.xsize as usize
        };
        if b3d_fread(
            unsafe {
                core::slice::from_raw_parts_mut(bdata.cast::<u8>(), d.pix_size as usize * count)
            },
            d.pix_size as usize,
            count,
            &mut fin,
        ) != count
        {
            // `mrcsec.c:367`.
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: mrcReadSectionAny - reading data from file.\n"),
            );
            return 3;
        }
        d.bdata = bdata;
        if unsafe {
            ii_process_read_line(
                hdata,
                li,
                &mut d,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        } != 0
        {
            return IIERR_QUITTING;
        }
        // `mrcsec.c:384-386`: only seek when there is something to skip.
        if (seek_end_x != 0 || d.seek_end_y != 0 || seek_skip != 0)
            && unsafe {
                mrc_huge_seek(
                    &mut fin,
                    seek_skip,
                    seek_end_x,
                    d.seek_end_y,
                    0,
                    nx_seek,
                    ny,
                    d.pix_size,
                    SEEK_CUR,
                )
            } != 0
        {
            return 3;
        }
        d.line += 1;
    }
    0
}
pub unsafe fn ii_init_read_section_any(
    hdata: &MrcHeader,
    li: &LoadInfo,
    buf: *mut u8,
    data: &mut LineProcData,
    y_end: &mut i32,
    caller: &str,
) -> i32 {
    let h = hdata;
    let l = li;
    let d = data;
    d.x_start = l.xmin;
    d.x_end = l.xmax;
    d.y_start = if d.read_y != 0 { l.zmin } else { l.ymin };
    let urfy = if d.read_y != 0 { l.zmax } else { l.ymax };
    if l.xmin > l.xmax || d.y_start > urfy {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiInitReadSectionAny - Specification of area to read (x {} to {} y {} to {}) is incorrect\n",
                l.xmin, l.xmax, d.y_start, urfy
            ),
        );
        return 1;
    }
    d.xsize = d.x_end - d.x_start + 1;
    d.byte = (d.type_ == MRSA_BYTE) as i32;
    d.to_short = (d.type_ == MRSA_USHORT) as i32;
    d.map_sbytes = (h.mode == MRC_MODE_BYTE && h.bytes_signed != 0) as i32;
    d.convert = d.byte + d.to_short;
    d.pix_index = 0;
    d.delta_y_sign = 1;
    d.pix_size = 1;
    d.need_data = 0;
    d.packed4bits = (h.packed4bits != 0 && h.mode == MRC_MODE_BYTE) as i32;
    d.half_floats = (h.mode == MRC_MODE_FLOAT && h.half_floats != 0) as i32;
    let eps = if d.to_short != 0 { 0.005 / 256. } else { 0.005 };
    d.do_scale =
        (l.offset <= -1. || l.offset >= 1. || l.slope < 1. - eps || l.slope > 1. + eps) as i32;
    d.bdata = buf;
    d.buf = buf;
    d.bufp_offset = 0;
    d.map.clear();
    if (d.type_ == 0 && d.map_sbytes != 0) || d.packed4bits != 0 {
        d.convert = 1;
    }
    if d.type_ == MRSA_FLOAT
        && !matches!(
            h.mode,
            MRC_MODE_BYTE | MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_FLOAT | MRC_MODE_RGB
        )
    {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: {} - Only real modes can be read as floats\n",
                caller
            ),
        );
        return 1;
    }
    if (d.byte != 0 && l.outmax > 255) || (d.to_short != 0 && l.outmax < 256) {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: {} - outmax ({}) is not in right range for conversion to {}\n",
                caller,
                l.outmax,
                if d.byte != 0 { "bytes" } else { "shorts" }
            ),
        );
        return 1;
    }
    /* This follows the source's four-corner construction exactly: the resulting
    file rectangle is read before ii_process_read_line places direct and reflected data. */
    if l.mirror_fft != 0 && d.convert != 0 {
        d.im_xsize = d.xsize;
        let im_nx = 2 * (h.nx - 1);
        d.im_ymin = if d.read_y != 0 { d.cz } else { l.ymin };
        d.im_ymax = if d.read_y != 0 { d.cz } else { l.ymax };
        let x_start2 = if d.x_start < d.x_end {
            d.x_start + 1
        } else {
            d.x_start
        };
        let (llfx, llfy) = mrc_mirror_source(im_nx, h.ny, d.x_start, d.im_ymin);
        let (ulfx, ulfy) = mrc_mirror_source(im_nx, h.ny, d.x_start, d.im_ymax);
        let (llfx2, llfy2) = mrc_mirror_source(im_nx, h.ny, x_start2, d.im_ymin);
        let (ulfx2, ulfy2) = mrc_mirror_source(im_nx, h.ny, x_start2, d.im_ymax);
        let (lrfx, lrfy) = mrc_mirror_source(im_nx, h.ny, d.x_end, d.im_ymin);
        let (urfx, urfy) = mrc_mirror_source(im_nx, h.ny, d.x_end, d.im_ymax);
        d.x_start = 0;
        if d.x_end < im_nx / 2 {
            d.x_start = lrfx;
        }
        if d.x_start > im_nx / 2 {
            d.x_start = llfx;
        }
        // `mrcsec.c:494`: `d->xEnd = b3dIMax(3, urfx, llfx, llfx2);`  The
        // leading `3` is `b3dIMax`'s argument *count* (`b3dutil.c:1307`), not
        // a value, so this is the maximum of the three corners after it.
        // Including the count floored `xEnd` at 3, which widens `xsize` on any
        // mirrored source whose corners all fall below it.
        d.x_end = b3d_i_max(&[urfx, llfx, llfx2]);
        d.ymin = llfy.min(ulfy).min(lrfy).min(urfy).min(llfy2).min(ulfy2);
        d.ymax = llfy.max(ulfy).max(lrfy).max(urfy).max(llfy2).max(ulfy2);
        d.y_start = if d.read_y != 0 { l.zmin } else { d.ymin };
        *y_end = if d.read_y != 0 { l.zmax } else { d.ymax };
        d.xsize = d.x_end - d.x_start + 1;
        d.toggle_y = -1;
        if d.read_y != 0 && d.ymin < d.ymax {
            d.toggle_y = 0;
            d.cz = d.ymin;
        }
    }
    if d.cz < 0
        || d.cz >= if d.read_y != 0 { h.ny } else { h.nz }
        || d.y_start < 0
        || *y_end >= if d.read_y != 0 { h.nz } else { h.ny }
    {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: {} - Requested area to read is out of range for file size\n",
                caller
            ),
        );
        return 1;
    }
    match h.mode {
        MRC_MODE_BYTE => {
            d.pix_size = 1;
            if (d.byte != 0 && d.do_scale != 0) || d.to_short != 0 {
                d.map = get_byte_map(l.slope, l.offset, l.outmin, l.outmax, h.bytes_signed);
            } else if d.map_sbytes != 0 {
                d.map = get_byte_map(1., 0., 0, 255, 1);
            };
            d.need_data =
                (d.type_ == MRSA_FLOAT || d.type_ == MRSA_USHORT || d.packed4bits != 0) as i32;
        }
        MRC_MODE_SHORT | MRC_MODE_USHORT => {
            d.pix_size = 2;
            if (d.to_short != 0 && (d.do_scale != 0 || h.mode == MRC_MODE_SHORT)) || d.byte != 0 {
                d.map = get_short_map(
                    l.slope,
                    l.offset,
                    l.outmin,
                    l.outmax,
                    l.ramp,
                    d.swapped,
                    (h.mode == MRC_MODE_SHORT) as i32,
                );
            };
            d.need_data = (d.type_ == MRSA_FLOAT || d.type_ == MRSA_BYTE) as i32;
        }
        MRC_MODE_RGB => {
            d.pix_size = 3;
            d.need_data = (d.convert != 0 || d.type_ == MRSA_FLOAT) as i32;
            if d.do_scale != 0 {
                d.map = get_byte_map(l.slope, l.offset, l.outmin, l.outmax, h.bytes_signed);
            }
        }
        MRC_MODE_FLOAT => {
            d.pix_size = if d.half_floats != 0 { 2 } else { 4 };
            d.need_data = d.convert;
            if d.half_floats != 0 {
                d.need_data = (d.to_short == 0) as i32;
            }
        }
        MRC_MODE_COMPLEX_SHORT => {
            d.pix_size = 4;
            if d.convert != 0 {
                return 1;
            }
        }
        MRC_MODE_COMPLEX_FLOAT => {
            d.pix_size = 8;
            d.need_data = d.convert;
        }
        _ => {
            if d.convert != 0 {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!("ERROR: {} - unsupported data type.\n", caller),
                );
                return 1;
            }
        }
    }
    d.bytes_since_check = 0;
    0
}
pub unsafe fn ii_process_read_line(
    hdata: &MrcHeader,
    li: &LoadInfo,
    data: &mut LineProcData,
    bdata_in: *mut u8,
    bufp_in: *mut u8,
) -> i32 {
    let h = hdata;
    let l = li;
    let d = data;
    let passed = !bdata_in.is_null() && !bufp_in.is_null();
    let bdata = if passed { bdata_in } else { d.bdata };
    let mut bufp = if passed {
        bufp_in
    } else {
        unsafe { d.buf.offset(d.bufp_offset) }
    };
    let mut usbufp = bufp.cast::<u16>();
    let mut fbufp = bufp.cast::<f32>();
    let map = d.map.as_ptr();
    let usmap = map.cast::<u16>();
    // `mrcsec.c:614-618` hoists these out of the switch; `outmin`/`outmax` are
    // declared `int` there and promote to float at each `B3DMAX`/`B3DMIN`.
    let slope = l.slope;
    let offset = l.offset;
    let outmin = l.outmin;
    let outmax = l.outmax;
    let n = d.xsize as usize;
    unsafe {
        if d.convert != 0
            || (d.type_ == MRSA_FLOAT && (h.mode != MRC_MODE_FLOAT || d.half_floats != 0))
        {
            match h.mode {
                MRC_MODE_BYTE if d.packed4bits != 0 => {
                    let mut input = 0_usize;
                    let mut output = 0_usize;
                    let mut xsize = n;
                    if d.x_start & 1 != 0 {
                        let value = *bdata.add(input) >> 4;
                        if d.byte != 0 && d.do_scale != 0 {
                            *bufp.add(output) = *map.add(value as usize);
                        } else if d.to_short != 0 {
                            *usbufp.add(output) = *usmap.add(value as usize);
                        } else if d.type_ == MRSA_FLOAT {
                            *fbufp.add(output) = value as f32;
                        } else {
                            *bufp.add(output) = value;
                        }
                        input += 1;
                        output += 1;
                        xsize -= 1;
                    }
                    while input < xsize / 2 {
                        let value = *bdata.add(input);
                        if d.byte != 0 && d.do_scale != 0 {
                            *bufp.add(output) = *map.add((value & 15) as usize);
                            *bufp.add(output + 1) = *map.add((value >> 4) as usize);
                        } else if d.to_short != 0 {
                            *usbufp.add(output) = *usmap.add((value & 15) as usize);
                            *usbufp.add(output + 1) = *usmap.add((value >> 4) as usize);
                        } else if d.type_ == MRSA_FLOAT {
                            *fbufp.add(output) = (value & 15) as f32;
                            *fbufp.add(output + 1) = (value >> 4) as f32;
                        } else {
                            *bufp.add(output) = value & 15;
                            *bufp.add(output + 1) = value >> 4;
                        }
                        input += 1;
                        output += 2;
                    }
                    if xsize & 1 != 0 {
                        let value = *bdata.add(input) & 15;
                        if d.byte != 0 && d.do_scale != 0 {
                            *bufp.add(output) = *map.add(value as usize);
                        } else if d.to_short != 0 {
                            *usbufp.add(output) = *usmap.add(value as usize);
                        } else if d.type_ == MRSA_FLOAT {
                            *fbufp.add(output) = value as f32;
                        } else {
                            *bufp.add(output) = value;
                        }
                    }
                }
                MRC_MODE_BYTE => {
                    if d.type_ == MRSA_FLOAT {
                        for i in 0..n {
                            *fbufp.add(i) = if d.map_sbytes != 0 {
                                *map.add(*bdata.add(i) as usize) as f32
                            } else {
                                *bdata.add(i) as f32
                            };
                        }
                    } else if d.to_short != 0 {
                        for i in 0..n {
                            *usbufp.add(i) = *usmap.add(*bdata.add(i) as usize);
                        }
                    } else if d.do_scale != 0 || d.map_sbytes != 0 {
                        for i in 0..n {
                            *bufp.add(i) = *map.add(*bdata.add(i) as usize);
                        }
                    } else if d.need_data != 0 {
                        core::ptr::copy_nonoverlapping(bdata, bufp, n);
                    }
                }
                MRC_MODE_SHORT | MRC_MODE_USHORT => {
                    let src = bdata.cast::<u16>();
                    if d.swapped != 0 {
                        mrc_swap_shorts(core::slice::from_raw_parts_mut(src.cast::<i16>(), n), n);
                    }
                    if d.type_ == MRSA_FLOAT {
                        for i in 0..n {
                            *fbufp.add(i) = if h.mode == MRC_MODE_SHORT {
                                *src.add(i) as i16 as f32
                            } else {
                                *src.add(i) as f32
                            };
                        }
                    } else if d.byte != 0 {
                        for i in 0..n {
                            *bufp.add(i) = *map.add(*src.add(i) as usize);
                        }
                    } else if d.to_short != 0 && (d.do_scale != 0 || h.mode == MRC_MODE_SHORT) {
                        for i in 0..n {
                            *usbufp.add(i) = *usmap.add(*src.add(i) as usize);
                        }
                    } else if d.need_data != 0 {
                        core::ptr::copy_nonoverlapping(src, usbufp, n);
                    }
                }
                MRC_MODE_RGB => {
                    for i in 0..n {
                        let p = bdata.add(3 * i);
                        let fpixel =
                            0.3 * *p as f32 + 0.59 * *p.add(1) as f32 + 0.11 * *p.add(2) as f32;
                        if d.type_ == MRSA_FLOAT {
                            *fbufp.add(i) = fpixel;
                        } else if d.byte != 0 {
                            *bufp.add(i) = if d.do_scale != 0 {
                                *map.add((fpixel + 0.499) as usize)
                            } else {
                                (fpixel + 0.5) as u8
                            };
                        } else {
                            *usbufp.add(i) = if d.do_scale != 0 {
                                *usmap.add((fpixel + 0.499) as usize)
                            } else {
                                (255. * fpixel + 0.5) as u16
                            };
                        }
                    }
                }
                MRC_MODE_FLOAT => {
                    if d.half_floats != 0 {
                        let src = bdata.cast::<u16>();
                        if d.swapped != 0 {
                            mrc_swap_shorts(core::slice::from_raw_parts_mut(src.cast(), n), n);
                        }
                        if d.type_ == MRSA_FLOAT {
                            imnp_halfbuf_to_floats(
                                core::slice::from_raw_parts(src, n),
                                core::slice::from_raw_parts_mut(fbufp, n),
                                n as i32,
                            );
                        } else if d.byte != 0 {
                            // Half Float to byte (`mrcsec.c:818-838`).
                            // Do unused ramps separately to speed up the regular load
                            if l.ramp == MRC_RAMP_LOG || l.ramp == MRC_RAMP_EXP {
                                for i in 0..n {
                                    let conv =
                                        f32::from_bits(imnp_halfbits_to_floatbits(*src.add(i)));
                                    let mut fpixel = if l.ramp == MRC_RAMP_LOG {
                                        conv.ln() * slope + offset
                                    } else {
                                        conv.exp() * slope + offset
                                    };
                                    if outmin as f32 > fpixel {
                                        fpixel = outmin as f32;
                                    }
                                    if (outmax as f32) < fpixel {
                                        fpixel = outmax as f32;
                                    }
                                    *bufp.add(i) = (fpixel + 0.5) as u8;
                                }
                            } else {
                                for i in 0..n {
                                    let conv =
                                        f32::from_bits(imnp_halfbits_to_floatbits(*src.add(i)));
                                    let mut fpixel = conv * slope + offset;
                                    if outmin as f32 > fpixel {
                                        fpixel = outmin as f32;
                                    }
                                    if (outmax as f32) < fpixel {
                                        fpixel = outmax as f32;
                                    }
                                    *bufp.add(i) = (fpixel + 0.5) as u8;
                                }
                            }
                        } else {
                            // Half Float to ushort (`mrcsec.c:841-860`).
                            if l.ramp == MRC_RAMP_LOG || l.ramp == MRC_RAMP_EXP {
                                for i in 0..n {
                                    let conv =
                                        f32::from_bits(imnp_halfbits_to_floatbits(*src.add(i)));
                                    let mut fpixel = if l.ramp == MRC_RAMP_LOG {
                                        conv.ln() * slope + offset
                                    } else {
                                        conv.exp() * slope + offset
                                    };
                                    if outmin as f32 > fpixel {
                                        fpixel = outmin as f32;
                                    }
                                    if (outmax as f32) < fpixel {
                                        fpixel = outmax as f32;
                                    }
                                    *usbufp.add(i) = (fpixel + 0.5) as u16;
                                }
                            } else {
                                for i in 0..n {
                                    let conv =
                                        f32::from_bits(imnp_halfbits_to_floatbits(*src.add(i)));
                                    let mut fpixel = conv * slope + offset;
                                    if outmin as f32 > fpixel {
                                        fpixel = outmin as f32;
                                    }
                                    if (outmax as f32) < fpixel {
                                        fpixel = outmax as f32;
                                    }
                                    *usbufp.add(i) = (fpixel + 0.5) as u16;
                                }
                            }
                        }
                    } else {
                        let src = bdata.cast::<f32>();
                        if d.swapped != 0 {
                            mrc_swap_floats(core::slice::from_raw_parts_mut(src, n), n);
                        }
                        if d.type_ != MRSA_FLOAT {
                            if d.byte != 0 {
                                // Float to byte (`mrcsec.c:842-861`).
                                // Do unused ramps separately to speed up the regular load
                                if l.ramp == MRC_RAMP_LOG || l.ramp == MRC_RAMP_EXP {
                                    for i in 0..n {
                                        // `mrcsec.c:846` is `(float)log((double)fdata[i])`:
                                        // the double routine, rounded to float.
                                        let mut fpixel = if l.ramp == MRC_RAMP_LOG {
                                            (*src.add(i) as f64).ln() as f32 * slope + offset
                                        } else {
                                            (*src.add(i) as f64).exp() as f32 * slope + offset
                                        };
                                        if outmin as f32 > fpixel {
                                            fpixel = outmin as f32;
                                        }
                                        if (outmax as f32) < fpixel {
                                            fpixel = outmax as f32;
                                        }
                                        *bufp.add(i) = (fpixel + 0.5) as u8;
                                    }
                                } else {
                                    for i in 0..n {
                                        let mut fpixel = *src.add(i) * slope + offset;
                                        if outmin as f32 > fpixel {
                                            fpixel = outmin as f32;
                                        }
                                        if (outmax as f32) < fpixel {
                                            fpixel = outmax as f32;
                                        }
                                        *bufp.add(i) = (fpixel + 0.5) as u8;
                                    }
                                }
                            } else {
                                // Float to ushort (`mrcsec.c:864-883`).
                                if l.ramp == MRC_RAMP_LOG || l.ramp == MRC_RAMP_EXP {
                                    for i in 0..n {
                                        let mut fpixel = if l.ramp == MRC_RAMP_LOG {
                                            (*src.add(i) as f64).ln() as f32 * slope + offset
                                        } else {
                                            (*src.add(i) as f64).exp() as f32 * slope + offset
                                        };
                                        if outmin as f32 > fpixel {
                                            fpixel = outmin as f32;
                                        }
                                        if (outmax as f32) < fpixel {
                                            fpixel = outmax as f32;
                                        }
                                        *usbufp.add(i) = (fpixel + 0.5) as u16;
                                    }
                                } else {
                                    for i in 0..n {
                                        let mut fpixel = *src.add(i) * slope + offset;
                                        if outmin as f32 > fpixel {
                                            fpixel = outmin as f32;
                                        }
                                        if (outmax as f32) < fpixel {
                                            fpixel = outmax as f32;
                                        }
                                        *usbufp.add(i) = (fpixel + 0.5) as u16;
                                    }
                                }
                            }
                        }
                    }
                }
                MRC_MODE_COMPLEX_FLOAT => {
                    let src = bdata.cast::<f32>();
                    if d.swapped != 0 {
                        mrc_swap_floats(core::slice::from_raw_parts_mut(src, 2 * n), 2 * n);
                    }
                    let scale = mrc_get_complex_scale();
                    let mut pix_index = if l.mirror_fft != 0 {
                        0
                    } else {
                        d.pix_index as usize
                    };
                    let fft = if l.mirror_fft != 0 { bdata } else { d.buf };
                    let usfft = fft.cast::<u16>();
                    for i in 0..n {
                        let a = *src.add(2 * i);
                        let b = *src.add(2 * i + 1);
                        let v = ((1. + scale * (a * a + b * b).sqrt()).ln() * l.slope + l.offset)
                            .clamp(l.outmin as f32, l.outmax as f32);
                        if d.byte != 0 {
                            *fft.add(pix_index) = v as u8
                        } else {
                            *usfft.add(pix_index) = v as u16
                        };
                        pix_index += 1;
                    }
                    if l.mirror_fft != 0 {
                        let cury = if d.read_y != 0 { d.cz } else { d.line };
                        let mut ybase = if d.read_y != 0 {
                            d.line - d.y_start
                        } else {
                            cury - d.im_ymin
                        };
                        let x0 = h.nx - 1 + d.x_start;
                        let x1 = h.nx - 1 + d.x_end;
                        if cury >= d.im_ymin && cury <= d.im_ymax && x1 >= l.xmin && x0 <= l.xmax {
                            let x2 = x0.max(l.xmin);
                            let x3 = x1.min(l.xmax);
                            if d.byte != 0 {
                                core::ptr::copy_nonoverlapping(
                                    fft.add((x2 - x0) as usize),
                                    d.buf.add((x2 - l.xmin + ybase * d.im_xsize) as usize),
                                    (x3 + 1 - x2) as usize,
                                );
                            } else {
                                core::ptr::copy_nonoverlapping(
                                    usfft.add((x2 - x0) as usize),
                                    d.buf
                                        .offset(d.bufp_offset)
                                        .cast::<u16>()
                                        .add((x2 - l.xmin + ybase * d.im_xsize) as usize),
                                    (x3 + 1 - x2) as usize,
                                );
                            }
                            if d.x_end == h.nx - 1 && l.xmin == 0 {
                                if d.byte != 0 {
                                    *d.buf.add((ybase * d.im_xsize) as usize) =
                                        *fft.add(d.xsize as usize - 1);
                                } else {
                                    *d.buf
                                        .offset(d.bufp_offset)
                                        .cast::<u16>()
                                        .add((ybase * d.im_xsize) as usize) =
                                        *usfft.add(d.xsize as usize - 1);
                                }
                            }
                        }
                        let cury_mirror = h.ny - cury;
                        let mut x2 = (h.nx - 1 - d.x_end).max(l.xmin);
                        let mut x3 = (h.nx - 1 - d.x_start).min(l.xmax);
                        if x2 == 0 {
                            x2 = 1;
                        }
                        if x3 >= h.nx - 1 {
                            x3 = h.nx - 2;
                        }
                        ybase = if d.read_y != 0 {
                            d.line - d.y_start
                        } else {
                            cury_mirror - d.im_ymin
                        };
                        if cury_mirror >= d.im_ymin && cury_mirror <= d.im_ymax {
                            for i in (0..=(x3 - x2).max(0) as usize).rev() {
                                if d.byte != 0 {
                                    *d.buf.add(
                                        i + x2 as usize - l.xmin as usize
                                            + (ybase * d.im_xsize) as usize,
                                    ) = *fft.add((h.nx - 1 - d.x_start - x2 - i as i32) as usize);
                                } else {
                                    *d.buf.offset(d.bufp_offset).cast::<u16>().add(
                                        i + x2 as usize - l.xmin as usize
                                            + (ybase * d.im_xsize) as usize,
                                    ) = *usfft.add((h.nx - 1 - d.x_start - x2 - i as i32) as usize);
                                }
                            }
                        }
                        /* C's special duplicated bottom-left row (clip-compatible FFT). */
                        if cury_mirror == 1 && d.im_ymin == 0 {
                            let bottom_ybase = if d.read_y != 0 { d.line - d.y_start } else { 0 };
                            for i in (0..=(x3 - x2).max(0) as usize).rev() {
                                let source = (h.nx - 1 - d.x_start - x2 - i as i32) as usize;
                                let destination = i + x2 as usize - l.xmin as usize
                                    + (bottom_ybase * d.im_xsize) as usize;
                                if d.byte != 0 {
                                    *d.buf.add(destination) = *fft.add(source);
                                } else {
                                    *d.buf.offset(d.bufp_offset).cast::<u16>().add(destination) =
                                        *usfft.add(source);
                                }
                            }
                        }
                        if d.toggle_y >= 0 {
                            d.toggle_y = 1 - d.toggle_y;
                            if d.toggle_y != 0 {
                                d.cz = d.ymax;
                                d.seek_end_y = d.ymax - d.ymin - 1;
                                d.line -= 1;
                            } else {
                                d.cz = d.ymin;
                                d.seek_end_y = d.ymin + h.ny - d.ymax - 1;
                            }
                        }
                    } else {
                        d.pix_index =
                            (pix_index as i32 + d.x_dimension * (d.delta_y_sign - 1)) as u32;
                    }
                }
                _ => {}
            }
        } else {
            if d.swapped != 0 {
                match h.mode {
                    MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_COMPLEX_SHORT => mrc_swap_shorts(
                        core::slice::from_raw_parts_mut(
                            bdata.cast(),
                            n * (d.pix_size as usize / 2),
                        ),
                        n * (d.pix_size as usize / 2),
                    ),
                    MRC_MODE_FLOAT | MRC_MODE_COMPLEX_FLOAT => mrc_swap_floats(
                        core::slice::from_raw_parts_mut(
                            bdata.cast(),
                            n * (d.pix_size as usize / 4),
                        ),
                        n * (d.pix_size as usize / 4),
                    ),
                    _ => {}
                }
            }
            if d.need_data != 0 {
                core::ptr::copy_nonoverlapping(bdata, bufp, n * d.pix_size as usize);
            }
        }
        // `mrcsec.c` advances the output pointer separately in each of its three
        // top-level branches, in units of the *output* pixel: every case of the
        // conversion switch (`mrcsec.c:701-996`) steps `bufp` or `usbufp` by
        // `xDimension * deltaYsign`, the conversion-to-float branch
        // (`mrcsec.c:1035-1036`) steps `fbufp` by the same, and only the raw
        // branch (`mrcsec.c:1061`) multiplies by `d->pixSize`.  Sharing the raw
        // branch's `* pixSize` with the conversion branch walked `bufp` past the
        // end of a byte output buffer by the input pixel size (4x for a float
        // file), corrupting the heap beyond it.
        let advance = d.x_dimension * d.delta_y_sign;
        if d.convert != 0 {
            if h.mode == MRC_MODE_COMPLEX_FLOAT {
                // `mrcsec.c:883-996`: complex output is placed through fft/pixIndex
                // and no buffer pointer is advanced.
            } else if d.type_ == MRSA_FLOAT {
                fbufp = fbufp.offset(advance as isize);
                bufp = fbufp.cast();
            } else if d.to_short != 0 {
                usbufp = usbufp.offset(advance as isize);
                bufp = usbufp.cast();
            } else {
                bufp = bufp.offset(advance as isize);
            }
        } else if d.type_ == MRSA_FLOAT {
            fbufp = fbufp.offset(advance as isize);
            bufp = fbufp.cast();
        } else if d.to_short != 0 {
            usbufp = usbufp.offset(advance as isize);
            bufp = usbufp.cast();
        } else {
            bufp = bufp.offset((advance * d.pix_size) as isize);
        }
        if !passed {
            d.bufp_offset = unsafe { bufp.offset_from(d.buf) };
            d.bytes_since_check += (d.xsize * d.pix_size + 100000).min(h.nx * d.pix_size);
            if d.bytes_since_check > 4000000 {
                d.bytes_since_check = 0;
                return crate::imod::libiimod::iimage::ii_check_for_quit(d.cz);
            }
        }
    }
    0
}
pub fn mrc_write_z(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &[u8], z: i32) -> i32 {
    let mut save = ImodImageFile::default();
    let file = unsafe { lookup_ii_file(hdata, li, 3, &mut save) };
    if !file.is_null() {
        unsafe {
            ii_sync_from_mrc_header(&mut *file, hdata);
            if (*file).file == IIFILE_MRC {
                if let Some(header) = (*file).mrc_header.as_mut() {
                    if !core::ptr::eq(header, hdata) {
                        *header = hdata.clone();
                    }
                }
            }
        }
        let answer = unsafe {
            match (*file).write_section {
                Some(func) => func(file, buf.as_ptr().cast_mut().cast(), z),
                None => -1,
            }
        };
        return unsafe {
            crate::imod::libiimod::iimage::ii_restore_load_params(answer, &mut *file, &save)
        };
    }
    mrc_write_section_any(hdata, li, buf, z, hdata.mode)
}
pub fn mrc_write_z_float(hdata: &mut MrcHeader, li: &mut LoadInfo, buf: &mut [f32], z: i32) -> i32 {
    let mut save = ImodImageFile::default();
    let file = unsafe { lookup_ii_file(hdata, li, 3, &mut save) };
    if !file.is_null() {
        unsafe {
            ii_sync_from_mrc_header(&mut *file, hdata);
            if (*file).file == IIFILE_MRC {
                if let Some(header) = (*file).mrc_header.as_mut() {
                    if !core::ptr::eq(header, hdata) {
                        *header = hdata.clone();
                    }
                }
            }
        }
        let answer = unsafe {
            match (*file).write_section_float {
                Some(func) => func(file, buf.as_mut_ptr().cast(), z),
                None => -1,
            }
        };
        return unsafe {
            crate::imod::libiimod::iimage::ii_restore_load_params(answer, &mut *file, &save)
        };
    }
    mrc_write_section_any(
        hdata,
        li,
        unsafe { core::slice::from_raw_parts(buf.as_ptr().cast(), buf.len() * 4) },
        z,
        if hdata.mode == MRC_MODE_COMPLEX_FLOAT || hdata.mode == MRC_MODE_COMPLEX_SHORT {
            hdata.mode
        } else {
            MRC_MODE_FLOAT
        },
    )
}
pub fn mrc_write_section_any(
    hdata: &mut MrcHeader,
    li: &mut LoadInfo,
    buf: &[u8],
    cz: i32,
    buf_mode: i32,
) -> i32 {
    let h = hdata;
    let l = li;
    let mut fin = h.fp.clone().unwrap();
    if l.xmin != 0 || l.xmax != h.nx - 1 {
        // `mrcsec.c:1174-1176`
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrcWriteSectionAny - only full lines can be written\n"),
        );
        return 1;
    }
    let mut bytes_chan = 0;
    let mut channels = 0;
    if mrc_getdcsize(h.mode, &mut bytes_chan, &mut channels) != 0 {
        // `mrcsec.c:1179-1181`
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrcWriteSectionAny - unknown mode.\n"),
        );
        return -1;
    }
    if h.half_floats != 0 && h.mode == MRC_MODE_FLOAT {
        bytes_chan = 2;
    }
    let pix_out = bytes_chan * channels;
    let mut buf_bytes = 0;
    let mut buf_channels = 0;
    if mrc_getdcsize(buf_mode, &mut buf_bytes, &mut buf_channels) != 0 {
        return -1;
    }
    let pix_buf = buf_bytes * buf_channels;
    if h.mode != buf_mode
        && !matches!(
            h.mode,
            MRC_MODE_BYTE | MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_FLOAT
        )
    {
        // `mrcsec.c:1189-1192`
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: mrcWriteSectionAny - floating point data can only be converted to byte/integer modes\n"
            ),
        );
        return 1;
    }
    let pack = h.mode == MRC_MODE_BYTE && h.packed4bits != 0;
    let signed = h.mode == MRC_MODE_BYTE && h.bytes_signed != 0 && !pack;
    // `mrcsec.c:1147` gates halfFloats on the header mode actually being
    // MRC_MODE_FLOAT.  `mrcInitOutputHeader` sets hdata->halfFloats from
    // write16BitModeForFloats() for every mode, so using the raw flag sends
    // byte/short/ushort writes down the half-float conversion path: mode 0
    // then writes two bytes per pixel into an nx-byte line buffer.
    let half_floats = h.mode == MRC_MODE_FLOAT && h.half_floats != 0;
    let nx_seek = if pack { (h.nx + 1) / 2 } else { h.nx };
    let bytes_line = nx_seek * pix_out;
    let need = h.mode != buf_mode
        || half_floats
        || h.swapped != 0 && bytes_chan > 1
        || signed
        || pack
        || h.y_inverted != 0
        || l.pad_left > 0
        || l.pad_right > 0;
    let mut chunk_lines = (l.ymax + 1 - l.ymin).min((100_000_000 / bytes_line).max(1));
    if need {
        chunk_lines =
            ((2_000_000_f32 / bytes_line as f32 + 0.5) as i32).clamp(1, l.ymax + 1 - l.ymin);
    }
    let mut temp = Vec::new();
    if need
        && (temp
            .try_reserve_exact((bytes_line * chunk_lines) as usize)
            .is_err())
    {
        // `mrcsec.c:1210-1213`
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrcWriteSectionAny - getting memory for temporary array.\n"),
        );
        return 2;
    }
    if need {
        temp.resize((bytes_line * chunk_lines) as usize, 0);
    }
    if unsafe {
        mrc_huge_seek(
            &mut fin,
            h.header_size + h.section_skip * cz,
            0,
            l.ymin,
            cz,
            nx_seek,
            h.ny,
            pix_out,
            SEEK_SET,
        )
    } != 0
    {
        // `mrcsec.c:1218-1220`
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrcWriteSectionAny - seeking to write location.\n"),
        );
        return 3;
    }
    let xdim = h.nx + l.pad_left.max(0) + l.pad_right.max(0);
    let mut line = if h.y_inverted != 0 { l.ymax } else { l.ymin };
    let step = if h.y_inverted != 0 { -1 } else { 1 };
    let mut lines_in_chunk = 0_i32;
    let mut chunk_write_ptr: *mut u8 = core::ptr::null_mut();
    while line >= l.ymin && line <= l.ymax {
        let src = unsafe {
            buf.as_ptr().add(
                (line - l.ymin) as usize * xdim as usize * pix_buf as usize
                    + l.pad_left.max(0) as usize * pix_buf as usize,
            )
        };
        let out = if !need {
            src.cast_mut()
        } else {
            unsafe {
                temp.as_mut_ptr()
                    .add((lines_in_chunk * bytes_line) as usize)
            }
        };
        if lines_in_chunk == 0 {
            chunk_write_ptr = out;
        }
        unsafe {
            if h.mode != buf_mode || half_floats {
                crate::imod::libiimod::iimage::ii_convert_line_of_floats(
                    core::slice::from_raw_parts(src.cast(), h.nx as usize),
                    core::slice::from_raw_parts_mut(out, bytes_line as usize),
                    if half_floats {
                        crate::imod::libiimod::mrcfiles::MRC_MODE_HALF_FLOAT
                    } else {
                        h.mode
                    },
                    h.bytes_signed != 0,
                    pack,
                );
            } else if pack {
                for i in 0..(h.nx / 2) as usize {
                    *out.add(i) =
                        ((*src.add(2 * i)).min(15)) | ((*src.add(2 * i + 1)).min(15) << 4);
                }
                if h.nx & 1 != 0 {
                    *out.add((h.nx / 2) as usize) = (*src.add((h.nx - 1) as usize)).min(15);
                }
            } else if signed {
                for i in 0..h.nx as usize {
                    *out.add(i) = (*src.add(i) as i8).wrapping_sub(-128) as u8;
                }
            } else if need {
                core::ptr::copy_nonoverlapping(src, out, (h.nx * pix_out) as usize);
            }
            if h.swapped != 0 && bytes_chan > 1 {
                if bytes_chan == 2 {
                    mrc_swap_shorts(
                        core::slice::from_raw_parts_mut(out.cast(), (h.nx * channels) as usize),
                        (h.nx * channels) as usize,
                    )
                } else {
                    mrc_swap_floats(
                        core::slice::from_raw_parts_mut(out.cast(), (h.nx * channels) as usize),
                        (h.nx * channels) as usize,
                    )
                }
            }
        }
        lines_in_chunk += 1;
        let last_line = if step > 0 {
            line + step > l.ymax
        } else {
            line + step < l.ymin
        };
        if lines_in_chunk == chunk_lines || last_line {
            let data_size = (nx_seek * lines_in_chunk) as usize;
            if b3d_fwrite(
                unsafe {
                    core::slice::from_raw_parts(
                        chunk_write_ptr.cast::<u8>(),
                        pix_out as usize * data_size,
                    )
                },
                pix_out as usize,
                data_size,
                &mut fin,
            ) != data_size
            {
                // `mrcsec.c:1303-1305`
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: mrcWriteSectionAny - writing data ({} bytes) to file (system message: {})\n",
                        data_size,
                        std::io::Error::last_os_error()
                    ),
                );
                return 3;
            }
            lines_in_chunk = 0;
        }
        line += step;
    }
    0
}
pub unsafe fn lookup_ii_file(
    hdata: &mut MrcHeader,
    li: &mut LoadInfo,
    axis: i32,
    ii_save: &mut ImodImageFile,
) -> *mut ImodImageFile {
    unsafe {
        if ii_calling_read_or_write() != 0 {
            return core::ptr::null_mut();
        }
        let Some(ii_file) = hdata.fp.as_ref().and_then(ii_lookup_file_from_fp) else {
            return core::ptr::null_mut();
        };
        if (*ii_file).file == IIFILE_MRC || (*ii_file).file == IIFILE_RAW {
            return core::ptr::null_mut();
        }
        ii_save_load_params(&*ii_file, ii_save);
        (*ii_file).llx = li.xmin;
        (*ii_file).urx = li.xmax;
        if axis == 3 {
            (*ii_file).lly = li.ymin;
            (*ii_file).ury = li.ymax;
        } else {
            (*ii_file).llz = li.zmin;
            (*ii_file).urz = li.zmax;
        }
        (*ii_file).axis = axis;
        (*ii_file).pad_left = li.pad_left;
        (*ii_file).pad_right = li.pad_right;
        (*ii_file).slope = li.slope;
        (*ii_file).offset = li.offset;
        ii_file
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libcfshr::b3dutil::{b3d_get_error, b3d_set_store_error};
    use crate::imod::libiimod::mrcfiles::{
        mrc_head_new, mrc_head_read, mrc_head_write, mrc_init_li,
    };

    #[test]
    fn byte_line_to_float_preserves_unsigned_pixels() {
        unsafe {
            let mut header = MrcHeader::default();
            header.nx = 3;
            header.ny = 1;
            header.nz = 1;
            header.mode = MRC_MODE_BYTE;
            let mut load = LoadInfo::default();
            load.xmin = 0;
            load.xmax = 2;
            load.ymin = 0;
            load.ymax = 0;
            load.zmin = 0;
            load.zmax = 0;
            load.slope = 1.0;
            load.outmax = 255;
            let mut input = [0_u8, 17, 255];
            let mut output = [0.0_f32; 3];
            let mut data = LineProcData::default();
            let mut free_map = 0;
            let mut y_end = 0;
            data.type_ = MRSA_FLOAT;
            data.read_y = 0;
            assert_eq!(
                ii_init_read_section_any(
                    &mut header,
                    &mut load,
                    output.as_mut_ptr().cast(),
                    &mut data,
                    &mut y_end,
                    "test",
                ),
                0
            );
            assert_eq!(
                ii_process_read_line(
                    &mut header,
                    &mut load,
                    &mut data,
                    input.as_mut_ptr(),
                    output.as_mut_ptr().cast(),
                ),
                0
            );
            assert_eq!(output, [0.0, 17.0, 255.0]);
        }
    }

    #[test]
    fn invalid_read_area_reports_the_source_diagnostic() {
        unsafe {
            let mut header = MrcHeader::default();
            header.nx = 3;
            header.ny = 1;
            header.nz = 1;
            header.mode = MRC_MODE_BYTE;
            let mut load = LoadInfo::default();
            load.xmin = 2;
            load.xmax = 1;
            load.ymin = 0;
            load.ymax = 0;
            load.zmin = 0;
            load.zmax = 0;
            load.slope = 1.0;
            load.outmax = 255;
            let mut output = [0_u8; 3];
            let mut data = LineProcData::default();
            let mut free_map = 0;
            let mut y_end = 0;
            b3d_set_store_error(1);
            assert_eq!(
                ii_init_read_section_any(
                    &mut header,
                    &mut load,
                    output.as_mut_ptr(),
                    &mut data,
                    &mut y_end,
                    "mrcReadSectionAny",
                ),
                1
            );
            assert_eq!(
                b3d_get_error(),
                "ERROR: iiInitReadSectionAny - Specification of area to read (x 2 to 1 y 0 to 0) is incorrect\n"
            );
            b3d_set_store_error(0);
        }
    }

    #[test]
    fn packed_four_bit_line_uses_low_then_high_nibbles() {
        unsafe {
            let mut header = MrcHeader::default();
            header.nx = 4;
            header.ny = 1;
            header.nz = 1;
            header.mode = MRC_MODE_BYTE;
            header.packed4bits = 1;
            let mut load = LoadInfo::default();
            load.xmin = 0;
            load.xmax = 3;
            load.ymin = 0;
            load.ymax = 0;
            load.zmin = 0;
            load.zmax = 0;
            load.slope = 1.0;
            load.outmax = 255;
            let mut input = [0x21_u8, 0x43];
            let mut output = [0.0_f32; 4];
            let mut data = LineProcData::default();
            let mut free_map = 0;
            let mut y_end = 0;
            data.type_ = MRSA_FLOAT;
            assert_eq!(
                ii_init_read_section_any(
                    &mut header,
                    &mut load,
                    output.as_mut_ptr().cast(),
                    &mut data,
                    &mut y_end,
                    "test",
                ),
                0
            );
            assert_eq!(
                ii_process_read_line(
                    &mut header,
                    &mut load,
                    &mut data,
                    input.as_mut_ptr(),
                    output.as_mut_ptr().cast(),
                ),
                0
            );
            assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn rgb_float_conversion_retains_weighted_fraction() {
        unsafe {
            let mut header = MrcHeader::default();
            header.nx = 1;
            header.ny = 1;
            header.nz = 1;
            header.mode = MRC_MODE_RGB;
            let mut load = LoadInfo::default();
            load.xmin = 0;
            load.xmax = 0;
            load.ymin = 0;
            load.ymax = 0;
            load.zmin = 0;
            load.zmax = 0;
            load.slope = 1.0;
            load.outmax = 255;
            let mut input = [1_u8, 2, 3];
            let mut output = [0.0_f32; 1];
            let mut data = LineProcData::default();
            let mut free_map = 0;
            let mut y_end = 0;
            data.type_ = MRSA_FLOAT;
            assert_eq!(
                ii_init_read_section_any(
                    &mut header,
                    &mut load,
                    output.as_mut_ptr().cast(),
                    &mut data,
                    &mut y_end,
                    "test",
                ),
                0
            );
            assert_eq!(
                ii_process_read_line(
                    &mut header,
                    &mut load,
                    &mut data,
                    input.as_mut_ptr(),
                    output.as_mut_ptr().cast(),
                ),
                0
            );
            assert_eq!(output, [1.81]);
        }
    }

    #[test]
    fn inverted_y_complex_section_from_real_mrc_starts_at_last_output_row() {
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-mrcsec-inverted-complex-{}.mrc",
                std::process::id()
            ));
            let path = path.to_str().unwrap().to_string();
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(
                mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_COMPLEX_FLOAT),
                0
            );
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let values = [0.0_f32, 0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    unsafe {
                        core::slice::from_raw_parts(
                            values.as_ptr().cast::<u8>(),
                            core::mem::size_of::<f32>() * values.len(),
                        )
                    },
                    core::mem::size_of::<f32>(),
                    values.len(),
                    &mut file,
                ),
                values.len()
            );
            drop(file);

            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "rb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut file, &mut header), 0);
            header.fp = Some(file.clone());
            header.y_inverted = 1;
            let mut load = LoadInfo::default();
            assert_eq!(mrc_init_li(Some(&mut load), None), 0);
            assert_eq!(mrc_init_li(Some(&mut load), Some(&header)), 0);
            load.slope = 10.;
            let mut output = [0_u8; 4];
            assert_eq!(mrc_read_z_byte(&mut header, &mut load, &mut output, 0), 0);
            drop(file);
            assert_eq!(output, [30, 34, 0, 23]);
            std::fs::remove_file(std::path::Path::new(path.as_str())).unwrap();
        }
    }

    #[test]
    fn inverted_y_mirrored_fft_reports_the_source_error() {
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-mrcsec-inverted-mirror-{}.mrc",
                std::process::id()
            ));
            let path = path.to_str().unwrap().to_string();
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(
                mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_COMPLEX_FLOAT),
                0
            );
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let values = [0.0_f32; 8];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    unsafe {
                        core::slice::from_raw_parts(
                            values.as_ptr().cast::<u8>(),
                            core::mem::size_of::<f32>() * values.len(),
                        )
                    },
                    core::mem::size_of::<f32>(),
                    values.len(),
                    &mut file,
                ),
                values.len()
            );
            drop(file);

            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "rb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut file, &mut header), 0);
            header.fp = Some(file.clone());
            header.y_inverted = 1;
            let mut load = LoadInfo::default();
            assert_eq!(mrc_init_li(Some(&mut load), None), 0);
            assert_eq!(mrc_init_li(Some(&mut load), Some(&header)), 0);
            load.mirror_fft = 1;
            let mut output = [0_u8; 4];
            b3d_set_store_error(1);
            assert_eq!(mrc_read_z_byte(&mut header, &mut load, &mut output, 0), 1);
            assert_eq!(
                b3d_get_error(),
                "ERROR: mrcReadSectionAny - cannot mirror FFT with inverted Y data.\n"
            );
            b3d_set_store_error(0);
            drop(file);
            std::fs::remove_file(std::path::Path::new(path.as_str())).unwrap();
        }
    }

    #[test]
    fn inverted_y_write_section_stores_rows_in_reverse() {
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-mrcsec-inverted-write-{}.mrc",
                std::process::id()
            ));
            let path = path.to_str().unwrap().to_string();
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "wb+").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 4, 3, 1, MRC_MODE_FLOAT), 0);
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            header.y_inverted = 1;
            let mut load = LoadInfo::default();
            assert_eq!(mrc_init_li(Some(&mut load), None), 0);
            assert_eq!(mrc_init_li(Some(&mut load), Some(&header)), 0);
            let mut values = [0.0_f32; 12];
            for line in 0..3 {
                for column in 0..4 {
                    values[line * 4 + column] = (10 * line + column) as f32;
                }
            }
            assert_eq!(
                mrc_write_z(
                    &mut header,
                    &mut load,
                    core::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 4),
                    0,
                ),
                0
            );
            {
                use std::io::Write;
                let _ = file.flush();
            }
            let mut stored = [0.0_f32; 12];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    1024 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(
                            stored.as_mut_ptr().cast::<u8>(),
                            4 * (stored.len()),
                        )
                    },
                    4,
                    stored.len(),
                    &mut file,
                ),
                stored.len()
            );
            drop(file);
            // `mrcsec.c:1223-1229`: an inverted-Y header writes the last buffer
            // line into the first file line.
            assert_eq!(
                stored,
                [
                    20.0, 21.0, 22.0, 23.0, 10.0, 11.0, 12.0, 13.0, 0.0, 1.0, 2.0, 3.0
                ]
            );
            std::fs::remove_file(std::path::Path::new(path.as_str())).unwrap();
        }
    }

    #[test]
    fn inverted_y_write_section_spans_several_chunks() {
        unsafe {
            // 1024 float pixels per line puts `chunkLines` (`mrcsec.c:1204-1206`)
            // at 488, so 600 lines are written as three descending chunks.
            let nx = 1024_i32;
            let ny = 600_i32;
            let path = std::env::temp_dir().join(format!(
                "imod-rs-mrcsec-inverted-chunk-{}.mrc",
                std::process::id()
            ));
            let path = path.to_str().unwrap().to_string();
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "wb+").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, nx, ny, 1, MRC_MODE_FLOAT), 0);
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            header.y_inverted = 1;
            let mut load = LoadInfo::default();
            assert_eq!(mrc_init_li(Some(&mut load), None), 0);
            assert_eq!(mrc_init_li(Some(&mut load), Some(&header)), 0);
            let mut values = vec![0.0_f32; (nx * ny) as usize];
            for line in 0..ny {
                for column in 0..nx {
                    values[(line * nx + column) as usize] = line as f32;
                }
            }
            assert_eq!(
                mrc_write_z(
                    &mut header,
                    &mut load,
                    core::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 4),
                    0,
                ),
                0
            );
            {
                use std::io::Write;
                let _ = file.flush();
            }
            let mut stored = vec![0.0_f32; (nx * ny) as usize];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    1024 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(
                            stored.as_mut_ptr().cast::<u8>(),
                            4 * (stored.len()),
                        )
                    },
                    4,
                    stored.len(),
                    &mut file,
                ),
                stored.len()
            );
            drop(file);
            for line in [0, 111, 112, 487, 488, 599] {
                assert_eq!(
                    stored[(line * nx) as usize],
                    (ny - 1 - line) as f32,
                    "file line {line}"
                );
                assert_eq!(
                    stored[(line * nx + nx - 1) as usize],
                    (ny - 1 - line) as f32
                );
            }
            std::fs::remove_file(std::path::Path::new(path.as_str())).unwrap();
        }
    }

    #[test]
    fn padded_write_section_spans_several_chunks() {
        unsafe {
            // Left padding forces the temporary-buffer path (`mrcsec.c:1201`) on a
            // non-inverted write, so the ascending chunk loop is exercised too.
            let nx = 1024_i32;
            let ny = 600_i32;
            let pad = 2_i32;
            let path = std::env::temp_dir().join(format!(
                "imod-rs-mrcsec-padded-chunk-{}.mrc",
                std::process::id()
            ));
            let path = path.to_str().unwrap().to_string();
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "wb+").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, nx, ny, 1, MRC_MODE_FLOAT), 0);
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let mut load = LoadInfo::default();
            assert_eq!(mrc_init_li(Some(&mut load), None), 0);
            assert_eq!(mrc_init_li(Some(&mut load), Some(&header)), 0);
            load.pad_left = pad;
            let xdim = nx + pad;
            let mut values = vec![-1.0_f32; (xdim * ny) as usize];
            for line in 0..ny {
                for column in 0..nx {
                    values[(line * xdim + pad + column) as usize] = line as f32;
                }
            }
            assert_eq!(
                mrc_write_z(
                    &mut header,
                    &mut load,
                    core::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 4),
                    0,
                ),
                0
            );
            {
                use std::io::Write;
                let _ = file.flush();
            }
            let mut stored = vec![0.0_f32; (nx * ny) as usize];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    1024 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(
                            stored.as_mut_ptr().cast::<u8>(),
                            4 * (stored.len()),
                        )
                    },
                    4,
                    stored.len(),
                    &mut file,
                ),
                stored.len()
            );
            drop(file);
            for line in [0, 111, 112, 487, 488, 599] {
                assert_eq!(
                    stored[(line * nx) as usize],
                    line as f32,
                    "file line {line}"
                );
                assert_eq!(stored[(line * nx + nx - 1) as usize], line as f32);
            }
            std::fs::remove_file(std::path::Path::new(path.as_str())).unwrap();
        }
    }

    #[test]
    fn inverted_y_raw_read_fills_the_buffer_backwards_without_underrun() {
        unsafe {
            // `mrcsec.c:230` fills pixSizeBuf[MRSA_NOPROC] with the file pixel
            // size; with a zero there the inverted-Y setup leaves `bufp` at the
            // start of the buffer and the first line lands in front of it.
            let nx = 4_i32;
            let ny = 3_i32;
            let path = std::env::temp_dir().join(format!(
                "imod-rs-mrcsec-inverted-raw-read-{}.mrc",
                std::process::id()
            ));
            let path = path.to_str().unwrap().to_string();
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, nx, ny, 1, MRC_MODE_FLOAT), 0);
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let mut values = [0.0_f32; 12];
            for line in 0..3 {
                for column in 0..4 {
                    values[line * 4 + column] = (10 * line + column) as f32;
                }
            }
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    unsafe {
                        core::slice::from_raw_parts(
                            values.as_ptr().cast::<u8>(),
                            4 * (values.len()),
                        )
                    },
                    4,
                    values.len(),
                    &mut file,
                ),
                values.len()
            );
            drop(file);

            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "rb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut file, &mut header), 0);
            header.fp = Some(file.clone());
            header.y_inverted = 1;
            let mut load = LoadInfo::default();
            assert_eq!(mrc_init_li(Some(&mut load), None), 0);
            assert_eq!(mrc_init_li(Some(&mut load), Some(&header)), 0);
            let guard = 16_usize;
            let mut arena = vec![0xAA_u8; 2 * guard + (nx * ny) as usize * 4];
            assert_eq!(
                mrc_read_z(
                    &mut header,
                    &mut load,
                    &mut arena[guard..guard + (nx * ny) as usize * 4],
                    0,
                ),
                0
            );
            drop(file);
            assert!(arena[..guard].iter().all(|&byte| byte == 0xAA));
            assert!(arena[arena.len() - guard..].iter().all(|&b| b == 0xAA));
            let mut got = [0.0_f32; 12];
            for (value, bytes) in got.iter_mut().zip(arena[guard..guard + 48].chunks_exact(4)) {
                *value = f32::from_ne_bytes(bytes.try_into().unwrap());
            }
            assert_eq!(
                got,
                [
                    20.0, 21.0, 22.0, 23.0, 10.0, 11.0, 12.0, 13.0, 0.0, 1.0, 2.0, 3.0
                ]
            );
            std::fs::remove_file(std::path::Path::new(path.as_str())).unwrap();
        }
    }

    #[test]
    fn float_section_read_as_bytes_stays_inside_the_byte_buffer() {
        unsafe {
            // The conversion cases of `mrcsec.c:701-996` step the output pointer
            // by `xDimension` bytes, not by the file's pixel size; multiplying by
            // `d->pixSize` here wrote four bytes per pixel of stride and ran off
            // the end of the caller's byte buffer.
            let nx = 8_i32;
            let ny = 5_i32;
            let path = std::env::temp_dir().join(format!(
                "imod-rs-mrcsec-float-to-byte-{}.mrc",
                std::process::id()
            ));
            let path = path.to_str().unwrap().to_string();
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, nx, ny, 1, MRC_MODE_FLOAT), 0);
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let values: Vec<f32> = (0..nx * ny).map(|index| index as f32).collect();
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    unsafe {
                        core::slice::from_raw_parts(
                            values.as_ptr().cast::<u8>(),
                            4 * (values.len()),
                        )
                    },
                    4,
                    values.len(),
                    &mut file,
                ),
                values.len()
            );
            drop(file);

            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "rb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut file, &mut header), 0);
            header.fp = Some(file.clone());
            let mut load = LoadInfo::default();
            assert_eq!(mrc_init_li(Some(&mut load), None), 0);
            assert_eq!(mrc_init_li(Some(&mut load), Some(&header)), 0);
            load.slope = 1.0;
            load.offset = 0.0;
            load.outmin = 0;
            load.outmax = 255;
            let guard = 16_usize;
            let mut arena = vec![0xAA_u8; 2 * guard + (nx * ny) as usize];
            assert_eq!(
                mrc_read_z_byte(
                    &mut header,
                    &mut load,
                    &mut arena[guard..guard + (nx * ny) as usize],
                    0,
                ),
                0
            );
            drop(file);
            assert!(arena[..guard].iter().all(|&byte| byte == 0xAA));
            assert!(arena[arena.len() - guard..].iter().all(|&b| b == 0xAA));
            for index in 0..(nx * ny) as usize {
                assert_eq!(arena[guard + index], index as u8, "pixel {index}");
            }
            std::fs::remove_file(std::path::Path::new(path.as_str())).unwrap();
        }
    }

    #[test]
    fn y_slice_read_steps_a_whole_section_between_lines() {
        unsafe {
            // `mrcsec.c:321` sets seekEndY to ny - 1 when reading in Y, so the
            // per-line seek advances to the same row of the next section.
            let nx = 3_i32;
            let ny = 2_i32;
            let nz = 3_i32;
            let path = std::env::temp_dir()
                .join(format!("imod-rs-mrcsec-read-y-{}.mrc", std::process::id()));
            let path = path.to_str().unwrap().to_string();
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "wb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, nx, ny, nz, MRC_MODE_FLOAT), 0);
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let mut values = Vec::new();
            for section in 0..nz {
                for line in 0..ny {
                    for column in 0..nx {
                        values.push((100 * section + 10 * line + column) as f32);
                    }
                }
            }
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    unsafe {
                        core::slice::from_raw_parts(
                            values.as_ptr().cast::<u8>(),
                            4 * (values.len()),
                        )
                    },
                    4,
                    values.len(),
                    &mut file,
                ),
                values.len()
            );
            drop(file);

            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "rb").unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut file, &mut header), 0);
            header.fp = Some(file.clone());
            let mut load = LoadInfo::default();
            assert_eq!(mrc_init_li(Some(&mut load), None), 0);
            assert_eq!(mrc_init_li(Some(&mut load), Some(&header)), 0);
            load.axis = 2;
            let mut got = [0.0_f32; 9];
            let mut got_bytes = [0_u8; 36];
            assert_eq!(mrc_read_y(&mut header, &mut load, &mut got_bytes, 1), 0);
            drop(file);
            for (value, bytes) in got.iter_mut().zip(got_bytes.chunks_exact(4)) {
                *value = f32::from_ne_bytes(bytes.try_into().unwrap());
            }
            assert_eq!(
                got,
                [10.0, 11.0, 12.0, 110.0, 111.0, 112.0, 210.0, 211.0, 212.0]
            );
            std::fs::remove_file(std::path::Path::new(path.as_str())).unwrap();
        }
    }
}
