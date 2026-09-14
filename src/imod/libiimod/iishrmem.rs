//! Translation of `IMOD/libiimod/iishrmem.c`.
//!
//! This module retains the operating-system shared-memory ABI.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, b3d_error, c_format_bytes};
use crate::imod::libcfshr::islice::slice_mode_if_real;
use crate::imod::libiimod::iimage::{
    ImodImageFile, LineProcData, MRSA_BYTE, MRSA_FLOAT, MRSA_NOPROC, MRSA_USHORT,
    ii_convert_line_of_floats, ii_delete, ii_new, ii_sync_from_mrc_header,
};
use crate::imod::libiimod::iimrc::{ii_mrc_fill_header, ii_mrc_set_load_info};
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MrcHeader,
    mrc_getdcsize, mrc_head_new,
};
use crate::imod::libiimod::mrcsec::{ii_init_read_section_any, ii_process_read_line};
#[cfg(windows)]
use core::ffi::c_char;
use core::ffi::c_void;

#[cfg(windows)]
const INVALID_HANDLE_VALUE: *mut c_void = -1_isize as *mut c_void;
#[cfg(windows)]
const PAGE_READWRITE: u32 = 0x04;
#[cfg(windows)]
const FILE_MAP_ALL_ACCESS: u32 = 0x000f_001f;

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateFileMappingA(
        file: *mut c_void,
        attributes: *mut c_void,
        protect: u32,
        maximum_size_high: u32,
        maximum_size_low: u32,
        name: *const c_char,
    ) -> *mut c_void;
    fn OpenFileMappingA(access: u32, inherit_handle: i32, name: *const c_char) -> *mut c_void;
    fn MapViewOfFile(
        mapping: *mut c_void,
        access: u32,
        file_offset_high: u32,
        file_offset_low: u32,
        number_of_bytes_to_map: usize,
    ) -> *mut c_void;
    fn UnmapViewOfFile(base_address: *const c_void) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

pub const IIFILE_SHR_MEM: i32 = 8;
/// C `SHR_MEM_NAME_TAG` (`iimage.h`), represented as normal Rust text.
pub const SHR_MEM_NAME_TAG: &str = "/IMODShrMem_";
pub const SHR_MEM_DATA_OFFSET: usize = 2048;

pub unsafe fn ii_shr_mem_create(filename: &str, ii_file: *mut ImodImageFile) -> i32 {
    unsafe {
        let mut map_file = 0;
        let address = open_and_get_address(filename, "iiShrMemCreate", &mut map_file);
        if address.is_null() {
            return 1;
        }
        (*ii_file).shr_mem_file = map_file;
        (*ii_file).user_data = address.cast();
        (*ii_file).header = core::ptr::null_mut();
        (*ii_file).filename = Some(filename.into());
        (*ii_file).close = Some(shm_close);
        (*ii_file).clean_up = Some(clean_up);
        0
    }
}

pub unsafe fn ii_shr_mem_open(filename: &str, mode: &str) -> *mut ImodImageFile {
    unsafe {
        let ii_file = ii_new();
        if ii_file.is_null() {
            return ii_file;
        }
        (*ii_file).close = Some(shm_close);
        (*ii_file).clean_up = Some(clean_up);
        (*ii_file).reopen = Some(reopen);
        (*ii_file).filename = Some(filename.into());
        (*ii_file).fill_mrc_header = Some(ii_mrc_fill_header);
        (*ii_file).shr_mem_mrc_header = Some(Box::default());
        let header = (*ii_file)
            .shr_mem_mrc_header
            .as_deref_mut()
            .expect("shared-memory header was just allocated")
            as *mut MrcHeader;
        (*ii_file).header = header.cast();
        (*ii_file).user_data =
            open_and_get_address(filename, "iiShrMemOpen", &mut (*ii_file).shr_mem_file).cast();
        if (*ii_file).user_data.is_null() {
            ii_delete(ii_file);
            return core::ptr::null_mut();
        }
        (*ii_file).fmode = mode.chars().take(3).collect();
        (*ii_file).file = IIFILE_SHR_MEM;
        if !mode.contains('w') {
            // A `clone`, not a bitwise copy: see `mrcsec::mrc_write_z`.
            *header = (*(*ii_file).user_data.cast::<MrcHeader>()).clone();
            ii_sync_from_mrc_header(ii_file, header);
        } else {
            mrc_head_new(&mut *header, 1, 1, 1, 0);
            (*header).packed4bits = 0;
            (*header).half_floats = 0;
            // `iishrmem.c:108`: `(FILE *)iiFile->userData`, the shared-memory
            // base address used as an identity token, never for I/O.
            (*header).fp = Some(ImodFile::Token((*ii_file).user_data as usize));
        }
        (*ii_file).fp = Some(ImodFile::Token((*ii_file).user_data as usize));
        (*header).fp = (*ii_file).fp.clone();
        (*ii_file).fill_mrc_header = Some(ii_mrc_fill_header);
        (*ii_file).sync_from_mrc_header = Some(sync_from_mrc_header);
        (*ii_file).write_header = Some(write_header);
        (*ii_file).read_section = Some(read_section);
        (*ii_file).read_section_byte = Some(read_section_byte);
        (*ii_file).read_section_ushort = Some(read_section_ushort);
        (*ii_file).read_section_float = Some(read_section_float);
        (*ii_file).write_section = Some(write_section);
        (*ii_file).write_section_float = Some(write_section_float);
        ii_file
    }
}

pub unsafe fn ii_shr_mem_check_size(filename: &str) -> usize {
    if !filename.starts_with(SHR_MEM_NAME_TAG) {
        return 0;
    }
    // `iishrmem.c:135` is `strtol(filename + strlen(SHR_MEM_NAME_TAG), &endPtr, 10)`:
    // optional leading whitespace and sign, then decimal digits, stopping at
    // the first character that cannot extend the number, with `endPtr` left
    // there.  A name with no digits at all leaves `endPtr` at the start, which
    // is why the `'_'` test below rejects it.
    let base = filename[SHR_MEM_NAME_TAG.len()..].as_bytes();
    let mut pos = 0;
    while pos < base.len() && (base[pos] as char).is_ascii_whitespace() {
        pos += 1;
    }
    let sign_pos = pos;
    if pos < base.len() && (base[pos] == b'+' || base[pos] == b'-') {
        pos += 1;
    }
    let digits_start = pos;
    while pos < base.len() && base[pos].is_ascii_digit() {
        pos += 1;
    }
    let (val, end) = if pos == digits_start {
        (0_i64, sign_pos)
    } else {
        let text = core::str::from_utf8(&base[sign_pos..pos]).unwrap_or("0");
        (text.parse::<i64>().unwrap_or(i64::MAX), pos)
    };
    if base.get(end).copied() != Some(b'_') {
        return 0;
    }
    (1024_i64.wrapping_mul(val)) as usize
}

unsafe fn open_and_get_address(filename: &str, caller: &str, map_file: *mut isize) -> *mut c_void {
    unsafe {
        let mem_size = ii_shr_mem_check_size(filename);
        // `shm_open`/`shm_unlink` are POSIX entry points that take a `char *`
        // and have no Rust standard-library equivalent, so the terminator is
        // added here, at that boundary, and nowhere above it.
        let Ok(cname) = std::ffi::CString::new(filename) else {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: {} - Shared-memory name contains a NUL byte\n",
                    caller
                ),
            );
            return core::ptr::null_mut();
        };
        if mem_size == 0 {
            // `iishrmem.c:163-165`.
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: {} - Filename {} does not have correct form for shared memory, {}/size/name\n",
                    caller, filename, SHR_MEM_NAME_TAG
                ),
            );
            return core::ptr::null_mut();
        }
        #[cfg(windows)]
        {
            let create = caller.contains("Create");
            let mapping = if create {
                CreateFileMappingA(
                    INVALID_HANDLE_VALUE,
                    core::ptr::null_mut(),
                    PAGE_READWRITE,
                    0,
                    mem_size as u32,
                    cname.as_ptr(),
                )
            } else {
                OpenFileMappingA(FILE_MAP_ALL_ACCESS, 0, cname.as_ptr())
            };
            if mapping.is_null() {
                return core::ptr::null_mut();
            }
            let address = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, mem_size);
            if address.is_null() {
                CloseHandle(mapping);
                return core::ptr::null_mut();
            }
            *map_file = mapping as isize;
            return address;
        }
        #[cfg(not(windows))]
        {
            let create = caller.contains("Create");
            if create && libc::shm_unlink(cname.as_ptr()) == 0 {
                use std::io::Write;
                // `iishrmem.c:190`.  Still on the C stream: this program's
                // other output goes through libc stdio, and a Rust write here
                // would reorder a redirected capture.
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "Shared memory file %s already exists, recreating it\n",
                    &[CArg::Str(filename)],
                ));
            }
            let fd = libc::shm_open(
                cname.as_ptr(),
                if create {
                    libc::O_CREAT | libc::O_RDWR | libc::O_EXCL
                } else {
                    libc::O_RDWR
                },
                libc::S_IRUSR | libc::S_IWUSR,
            );
            if fd < 0 {
                // `iishrmem.c:196-198`.
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: {} - Opening memory file for {} - {}\n",
                        caller,
                        filename,
                        std::io::Error::last_os_error()
                    ),
                );
                return core::ptr::null_mut();
            }
            let truncate = cfg!(not(target_os = "macos"));
            if create || truncate {
                if libc::ftruncate(fd, mem_size as libc::off_t) != 0 {
                    // `iishrmem.c:202-204`.
                    b3d_error(
                        Some(&mut ImodFile::Stderr),
                        format_args!(
                            "ERROR: {} - Cannot set shared memory for {} to requested size - {}\n",
                            caller,
                            filename,
                            std::io::Error::last_os_error()
                        ),
                    );
                    libc::shm_unlink(cname.as_ptr());
                    return core::ptr::null_mut();
                }
            }
            let address = libc::mmap(
                core::ptr::null_mut(),
                mem_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if address == libc::MAP_FAILED {
                // `iishrmem.c:211-213`.
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: {} - Cannot get address of shared memory for {} - {}\n",
                        caller,
                        filename,
                        std::io::Error::last_os_error()
                    ),
                );
                libc::shm_unlink(cname.as_ptr());
                return core::ptr::null_mut();
            }
            if create {
                libc::munmap(address, mem_size);
            }
            *map_file = fd as isize;
            address
        }
    }
}

unsafe extern "C" fn shm_close(ii_file: *mut ImodImageFile) {
    unsafe {
        if (*ii_file).user_data.is_null() {
            return;
        }
        #[cfg(windows)]
        {
            UnmapViewOfFile((*ii_file).user_data.cast());
            if (*ii_file).shr_mem_file != 0 {
                CloseHandle((*ii_file).shr_mem_file as *mut c_void);
            }
            (*ii_file).shr_mem_file = 0;
        }
        #[cfg(not(windows))]
        {
            let Some(name) = (*ii_file).filename.clone() else {
                return;
            };
            let size = ii_shr_mem_check_size(&name);
            if size == 0 {
                return;
            }
            if !(*ii_file).header.is_null() {
                libc::munmap((*ii_file).user_data.cast(), size);
            } else {
                if let Ok(cname) = std::ffi::CString::new(name) {
                    libc::shm_unlink(cname.as_ptr());
                }
            }
        }
        (*ii_file).user_data = core::ptr::null_mut();
    }
}
pub unsafe fn ii_shr_mem_remove(filename: &str) -> i32 {
    unsafe {
        #[cfg(windows)]
        {
            let _ = filename;
            1
        }
        #[cfg(not(windows))]
        {
            // The `shm_unlink` boundary; see `open_and_get_address`.
            let Ok(cname) = std::ffi::CString::new(filename) else {
                return -1;
            };
            libc::shm_unlink(cname.as_ptr())
        }
    }
}
unsafe extern "C" fn clean_up(ii_file: *mut ImodImageFile) {
    unsafe {
        // The C implementation frees `iiFile->header` here.  Its Rust storage
        // belongs to the image record, so dropping the owned slot also handles
        // the no-header path without reconstructing ownership from a raw alias.
        (*ii_file).shr_mem_mrc_header = None;
        (*ii_file).header = core::ptr::null_mut();
    }
}
unsafe extern "C" fn reopen(ii_file: *mut ImodImageFile) -> i32 {
    unsafe {
        let name = (*ii_file).filename.clone().unwrap_or_default();
        (*ii_file).user_data =
            open_and_get_address(&name, "iiShrMemOpen", &mut (*ii_file).shr_mem_file).cast();
        if (*ii_file).user_data.is_null() { 1 } else { 0 }
    }
}
unsafe extern "C" fn sync_from_mrc_header(
    ii_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
) -> i32 {
    unsafe {
        if (*ii_file).header.cast::<MrcHeader>() != hdata {
            // A `clone`, not a bitwise copy: see `mrcsec::mrc_write_z`.
            *(*ii_file).header.cast::<MrcHeader>() = (*hdata).clone();
        }
        0
    }
}
unsafe extern "C" fn write_header(ii_file: *mut ImodImageFile) -> i32 {
    unsafe {
        if (*ii_file).user_data.is_null() || (*ii_file).header.is_null() {
            return 1;
        }
        core::ptr::copy_nonoverlapping(
            (*ii_file).header.cast::<MrcHeader>(),
            (*ii_file).user_data.cast(),
            1,
        );
        0
    }
}
unsafe extern "C" fn read_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { shm_read_section_any(in_file, buf, in_section, MRSA_NOPROC) }
}
unsafe extern "C" fn read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { shm_read_section_any(in_file, buf, in_section, MRSA_BYTE) }
}
unsafe extern "C" fn read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { shm_read_section_any(in_file, buf, in_section, MRSA_USHORT) }
}
unsafe extern "C" fn read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { shm_read_section_any(in_file, buf, in_section, MRSA_FLOAT) }
}
unsafe fn shm_read_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    typ: i32,
) -> i32 {
    unsafe {
        let h = (*in_file).header.cast::<MrcHeader>();
        if h.is_null() || (*in_file).user_data.is_null() {
            return 1;
        }
        let mut pix_size_buf = [0, 1, 4, 2];
        let mut d = LineProcData::default();
        let mut li = LoadInfo::default();
        ii_mrc_set_load_info(in_file, &mut li);
        let pad_left = li.pad_left.max(0);
        let pad_right = li.pad_right.max(0);
        li.outmin = (*in_file).smin as i32;
        li.outmax = (*in_file).smax as i32;
        li.mirror_fft = 0;
        d.type_ = typ;
        d.read_y = if li.axis == 2 { 1 } else { 0 };
        let mut y_end = if d.read_y != 0 { li.zmax } else { li.ymax };
        d.cz = in_section;
        d.swapped = 0;
        li.outmin = 0;
        li.outmax = if typ == MRSA_USHORT { 65535 } else { 255 };
        let mut free_map = 0;
        let err = ii_init_read_section_any(
            h,
            &mut li,
            buf.cast(),
            &mut d,
            &mut y_end,
            "shmReadSectionAny",
        );
        if err != 0 {
            return err;
        }
        d.x_dimension = d.xsize + pad_left + pad_right;
        pix_size_buf[0] = d.pix_size;
        let shm_buf = (*in_file).user_data.cast::<u8>().add(
            SHR_MEM_DATA_OFFSET
                + d.pix_size as usize
                    * (*h).nx as usize
                    * ((*h).ny as usize * in_section as usize + d.y_start as usize),
        );
        d.bufp = d
            .bufp
            .add(pix_size_buf[typ as usize] as usize * pad_left as usize);
        d.pix_index += pad_left as u32;
        let lines = y_end + 1 - d.y_start;
        if ((typ == MRSA_FLOAT && (*h).mode == MRC_MODE_FLOAT) || typ == MRSA_NOPROC)
            && pad_left == 0
            && pad_right == 0
            && d.xsize == (*h).nx
            && d.read_y == 0
        {
            core::ptr::copy_nonoverlapping(
                shm_buf,
                buf.cast(),
                d.pix_size as usize * (*h).nx as usize * lines as usize,
            );
            return 0;
        }
        for iy in 0..lines {
            let line = shm_buf
                .add((d.x_start as usize + iy as usize * (*h).nx as usize) * d.pix_size as usize);
            if (typ == MRSA_FLOAT && (*h).mode != MRC_MODE_FLOAT)
                || typ == MRSA_USHORT
                || typ == MRSA_BYTE
            {
                let output = match typ {
                    MRSA_FLOAT => d.bufp.add(iy as usize * d.x_dimension as usize * 4),
                    MRSA_USHORT => d.bufp.add(iy as usize * d.x_dimension as usize * 2),
                    _ => d.bufp.add(iy as usize * d.x_dimension as usize),
                };
                ii_process_read_line(h, &mut li, &mut d, line, output);
            } else {
                let output = d.bufp.add(
                    iy as usize * d.x_dimension as usize * pix_size_buf[typ as usize] as usize,
                );
                core::ptr::copy_nonoverlapping(
                    line,
                    output,
                    d.xsize as usize * d.pix_size as usize,
                );
            }
        }
        0
    }
}
unsafe extern "C" fn write_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { shm_write_section_any(in_file, buf, in_section, 0) }
}
unsafe extern "C" fn write_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { shm_write_section_any(in_file, buf, in_section, 1) }
}
unsafe fn shm_write_section_any(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    from_float: i32,
) -> i32 {
    unsafe {
        let h = (*in_file).header.cast::<MrcHeader>();
        if h.is_null() || (*in_file).user_data.is_null() {
            return 1;
        }
        let mut li = LoadInfo::default();
        ii_mrc_set_load_info(in_file, &mut li);
        let mut buf_mode = (*h).mode;
        let convert =
            from_float > 0 && !matches!((*h).mode, MRC_MODE_COMPLEX_FLOAT | MRC_MODE_FLOAT);
        if convert {
            buf_mode = MRC_MODE_FLOAT;
        }
        if li.xmin != 0 || li.xmax != (*h).nx - 1 {
            return 1;
        }
        let mut bytes_per_chan_out = 0;
        let mut num_chan_out = 0;
        if mrc_getdcsize((*h).mode, &mut bytes_per_chan_out, &mut num_chan_out) != 0
            || (*h).mode == MRC_MODE_COMPLEX_SHORT
        {
            return -1;
        }
        let mut bytes_per_chan_buf = 0;
        let mut num_chan_buf = 0;
        mrc_getdcsize(buf_mode, &mut bytes_per_chan_buf, &mut num_chan_buf);
        let pix_size_out = bytes_per_chan_out * num_chan_out;
        let _pix_size_buf = bytes_per_chan_buf * num_chan_buf;
        if convert && slice_mode_if_real((*h).mode) < 0 {
            return 1;
        }
        let y_start = li.ymin;
        let y_end = li.ymax;
        let chunk_lines = y_end + 1 - y_start;
        let dest = (*in_file).user_data.cast::<u8>().add(
            SHR_MEM_DATA_OFFSET
                + pix_size_out as usize
                    * (*h).nx as usize
                    * ((*h).ny as usize * in_section as usize + y_start as usize),
        );
        if from_float == 0 || (*h).mode == MRC_MODE_FLOAT {
            core::ptr::copy_nonoverlapping(
                buf.cast(),
                dest,
                pix_size_out as usize * (*h).nx as usize * chunk_lines as usize,
            );
            return 0;
        }
        let bytes_signed = ((*h).mode == 0 && (*h).bytes_signed != 0) as i32;
        for iy in 0..chunk_lines {
            ii_convert_line_of_floats(
                buf.cast::<f32>().add(iy as usize * (*h).nx as usize),
                dest.add(pix_size_out as usize * iy as usize * (*h).nx as usize),
                (*h).nx,
                (*h).mode,
                bytes_signed,
                0,
            );
        }
        0
    }
}

#[cfg(all(test, not(windows)))]
mod tests {
    use super::*;

    #[test]
    fn posix_shared_memory_round_trips_a_native_section() {
        unsafe {
            let name = format!("/IMODShrMem_1024_iishrmem_{}", std::process::id());
            let manager = ii_new();
            assert!(!manager.is_null());
            assert_eq!(ii_shr_mem_create(&name, manager), 0);
            let writer = ii_shr_mem_open(&name, "wb+");
            assert!(!writer.is_null());
            let header = (*writer).header.cast::<MrcHeader>();
            let owned_header = (*writer)
                .shr_mem_mrc_header
                .as_deref_mut()
                .expect("shared-memory writer has an owned header")
                as *mut MrcHeader;
            assert_eq!(header, owned_header);
            mrc_head_new(&mut *header, 2, 2, 1, 0);
            ii_sync_from_mrc_header(writer, header);
            assert_eq!(((*writer).write_header.unwrap())(writer), 0);
            let input = [3_i8, 1, 4, 1];
            assert_eq!(
                ((*writer).write_section.unwrap())(writer, input.as_ptr().cast_mut().cast(), 0),
                0
            );
            let reader = ii_shr_mem_open(&name, "rb");
            assert!(!reader.is_null());
            let mut output = [0_i8; 4];
            assert_eq!(
                ((*reader).read_section.unwrap())(reader, output.as_mut_ptr().cast(), 0),
                0
            );
            assert_eq!(output, input);
            ii_delete(reader);
            ii_delete(writer);
            ii_delete(manager);
            assert_ne!(ii_shr_mem_remove(&name), 0);
        }
    }

    #[test]
    fn posix_shared_memory_reads_cropped_padded_floats() {
        unsafe {
            let name = format!("/IMODShrMem_1024_iishrmem_convert_{}", std::process::id());
            let manager = ii_new();
            assert_eq!(ii_shr_mem_create(&name, manager), 0);
            let writer = ii_shr_mem_open(&name, "wb+");
            let header = (*writer).header.cast::<MrcHeader>();
            mrc_head_new(&mut *header, 3, 2, 1, 1);
            ii_sync_from_mrc_header(writer, header);
            assert_eq!(((*writer).write_header.unwrap())(writer), 0);
            let input = [10_i16, 20, 30, 40, 50, 60];
            assert_eq!(
                ((*writer).write_section.unwrap())(writer, input.as_ptr().cast_mut().cast(), 0),
                0
            );
            let reader = ii_shr_mem_open(&name, "rb");
            (*reader).llx = 1;
            (*reader).urx = 2;
            (*reader).pad_left = 1;
            (*reader).pad_right = 1;
            let mut output = [-1_f32; 8];
            assert_eq!(
                ((*reader).read_section_float.unwrap())(reader, output.as_mut_ptr().cast(), 0),
                0
            );
            assert_eq!(output, [-1., 20., 30., -1., -1., 50., 60., -1.]);
            ii_delete(reader);
            ii_delete(writer);
            ii_delete(manager);
            assert_ne!(ii_shr_mem_remove(&name), 0);
        }
    }
}
