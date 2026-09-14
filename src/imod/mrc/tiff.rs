//! Translation of the legacy write path in `IMOD/mrc/tiff.c` and its
//! `b3dtiff.h` contract.
//!
//! The complete reader half of this historical source is a separate lower
//! dependency of `tif2mrc`; this module supplies the paired writer used by
//! `mrc2tif -o` and preserves its deliberately uncompressed classic-TIFF
//! layout.  Its default reader is source-compatible; the optional Rust reader
//! is selected only through `IMOD_RS_TIFF_BACKEND=rust`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::{
    ImodFile, SEEK_CUR, SEEK_END, SEEK_SET, b3d_fread, b3d_fseek, b3d_fwrite, b3d_rewind,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader,
    mrc_head_new,
};

/// C `Tf_header` (`b3dtiff.h`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TfHeader {
    pub byteorder: i16,
    pub version: i16,
    pub first_ifd_offset: i32,
}
/// C `Tf_entry` (`b3dtiff.h`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TfEntry {
    pub tagfield: i16,
    pub ftype: i16,
    pub length: i32,
    pub value: i32,
}
/// C `Im_info` (`b3dtiff.h`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ImInfo {
    pub func: i16,
    pub mag: i16,
    pub tilt: i16,
    pub date: i32,
    pub comment: [i8; 128],
    pub extra: [i8; 128],
}

impl Default for ImInfo {
    /// The all-zero `Tf_info.imageinfo` a C caller gets from its zeroed or
    /// `calloc`ed `Tf_info`; written out because the two 128-byte arrays are
    /// past the length `Default` derives for.
    fn default() -> ImInfo {
        ImInfo {
            func: 0,
            mag: 0,
            tilt: 0,
            date: 0,
            comment: [0; 128],
            extra: [0; 128],
        }
    }
}
/// C `Tf_info` (`b3dtiff.h`).
#[repr(C)]
pub struct TfInfo {
    pub header: TfHeader,
    pub numentries: i16,
    pub directory: [TfEntry; 7],
    pub next_ifd: i32,
    pub imageinfo: ImInfo,
    pub iifile: *mut crate::imod::libiimod::iimage::ImodImageFile,
    pub fp: Option<ImodFile>,
    pub data: *mut u8,
    pub nstrip: i32,
    pub stripoff: *mut i32,
    pub stripsize: *mut i32,
    pub width: i32,
    pub length: i32,
    pub rows_per_strip: i32,
    pub strip_pos: i32,
    pub strip_byte_counts: i32,
    pub bits_per_sample: i32,
    pub photometric_interpretation: i32,
    pub mode: i32,
}

impl Default for TfInfo {
    /// The all-zero `Tf_info` every caller of `tiff_open_file` declares on the
    /// stack and `memset`s.  It cannot be `mem::zeroed` any more: `fp` is an
    /// `Option<ImodFile>`, whose all-zero bit pattern is not a valid `None`,
    /// and `tiff_open_file`'s `(*tif).fp = ...` would then *drop* it
    /// (NATIVE.md 4b/4d).
    fn default() -> TfInfo {
        TfInfo {
            header: TfHeader::default(),
            numentries: 0,
            directory: [TfEntry::default(); 7],
            next_ifd: 0,
            imageinfo: ImInfo::default(),
            iifile: core::ptr::null_mut(),
            fp: None,
            data: core::ptr::null_mut(),
            nstrip: 0,
            stripoff: core::ptr::null_mut(),
            stripsize: core::ptr::null_mut(),
            width: 0,
            length: 0,
            rows_per_strip: 0,
            strip_pos: 0,
            strip_byte_counts: 0,
            bits_per_sample: 0,
            photometric_interpretation: 0,
            mode: 0,
        }
    }
}

static mut SWAP_DATA: i32 = 0;

/// C static `swap` (`tiff.c:35`).
unsafe fn swap(ptr: *mut i8, mut size: u32) {
    unsafe {
        if size % 2 != 0 {
            size -= 1;
        }
        for index in 0..size / 2 {
            let first = ptr.add(index as usize);
            let last = ptr.add((size - 1 - index) as usize);
            let value = *first;
            *first = *last;
            *last = value;
        }
    }
}

/// C `isit_tiff` (`tiff.c:59`).
pub unsafe fn isit_tiff(fp: &mut ImodFile) -> i32 {
    unsafe {
        let mut word = 0u16;
        b3d_rewind(fp);
        if b3d_fread(
            core::slice::from_raw_parts_mut((&mut word as *mut u16).cast::<u8>(), 2 * 1),
            2,
            1,
            fp,
        ) < 1
            || (word != 0x4949 && word != 0x4d4d)
        {
            return 0;
        }
        if b3d_fread(
            core::slice::from_raw_parts_mut((&mut word as *mut u16).cast::<u8>(), 2 * 1),
            2,
            1,
            fp,
        ) < 1
            || (word != 0x0042 && word != 0x4200)
        {
            return 0;
        }
        1
    }
}

/// C `tiffFirstIFD` (`tiff.c:347`).
pub unsafe fn tiff_first_ifd(fp: &mut ImodFile) -> u32 {
    unsafe {
        let mut word = 0u16;
        let mut result = 0u32;
        b3d_rewind(fp);
        if b3d_fread(
            core::slice::from_raw_parts_mut((&mut word as *mut u16).cast::<u8>(), 2 * 1),
            2,
            1,
            fp,
        ) < 1
            || (word != 0x4949 && word != 0x4d4d)
        {
            return 0;
        }
        SWAP_DATA = (word
            != if cfg!(target_endian = "little") {
                0x4949
            } else {
                0x4d4d
            }) as i32;
        if b3d_fread(
            core::slice::from_raw_parts_mut((&mut word as *mut u16).cast::<u8>(), 2 * 1),
            2,
            1,
            fp,
        ) < 1
        {
            return 0;
        }
        if SWAP_DATA != 0 {
            swap((&mut word as *mut u16).cast(), 2);
        }
        b3d_fread(
            core::slice::from_raw_parts_mut((&mut result as *mut u32).cast::<u8>(), 4 * 1),
            4,
            1,
            fp,
        );
        if SWAP_DATA != 0 {
            swap((&mut result as *mut u32).cast(), 4);
        }
        result
    }
}

/// C `read_tiffheader` (`tiff.c:325`).
pub unsafe fn read_tiffheader(fp: &mut ImodFile, header: *mut TfHeader) -> i32 {
    unsafe {
        if b3d_fread(
            core::slice::from_raw_parts_mut(
                (header).cast::<u8>(),
                (core::mem::size_of::<TfHeader>()) * (1),
            ),
            core::mem::size_of::<TfHeader>(),
            1,
            fp,
        ) < 1
        {
            return 0;
        }
        let order = (*header).byteorder as u16;
        if order != 0x4d4d && order != 0x4949 {
            return 0;
        }
        let machine = if cfg!(target_endian = "little") {
            0x4949
        } else {
            0x4d4d
        };
        if order != machine {
            swap(core::ptr::addr_of_mut!((*header).version).cast(), 2);
            swap(
                core::ptr::addr_of_mut!((*header).first_ifd_offset).cast(),
                4,
            );
        }
        (*header).first_ifd_offset = tiff_first_ifd(fp) as i32;
        if (*header).version != 42 { 0 } else { 1 }
    }
}

/// C `tiffIFD` (`tiff.c:379`).
pub unsafe fn tiff_ifd(fp: &mut ImodFile, section: i32) -> u32 {
    unsafe {
        let mut ifd = tiff_first_ifd(fp);
        for _ in 0..section {
            if ifd == 0 {
                return 0;
            }
            let mut entries = 0u16;
            b3d_fseek(fp, (ifd as i64) as i32, SEEK_SET);
            b3d_fread(
                core::slice::from_raw_parts_mut((&mut entries as *mut u16).cast::<u8>(), 2 * 1),
                2,
                1,
                fp,
            );
            if SWAP_DATA != 0 {
                swap((&mut entries as *mut u16).cast(), 2);
            }
            b3d_fseek(
                fp,
                ((ifd + 2 + entries as u32 * 12) as i64) as i32,
                SEEK_SET,
            );
            b3d_fread(
                core::slice::from_raw_parts_mut((&mut ifd as *mut u32).cast::<u8>(), 4 * 1),
                4,
                1,
                fp,
            );
            if SWAP_DATA != 0 {
                swap((&mut ifd as *mut u32).cast(), 4);
            }
        }
        ifd
    }
}

/// C `tiffIFDNumber` (`tiff.c:411`).
pub unsafe fn tiff_ifd_number(fp: &mut ImodFile) -> i32 {
    unsafe {
        let mut ifd = tiff_first_ifd(fp);
        let mut count = 0;
        while ifd != 0 {
            let mut entries = 0u16;
            b3d_fseek(fp, (ifd as i64) as i32, SEEK_SET);
            b3d_fread(
                core::slice::from_raw_parts_mut((&mut entries as *mut u16).cast::<u8>(), 2 * 1),
                2,
                1,
                fp,
            );
            if SWAP_DATA != 0 {
                swap((&mut entries as *mut u16).cast(), 2);
            }
            count += 1;
            b3d_fseek(
                fp,
                ((ifd + 2 + entries as u32 * 12) as i64) as i32,
                SEEK_SET,
            );
            b3d_fread(
                core::slice::from_raw_parts_mut((&mut ifd as *mut u32).cast::<u8>(), 4 * 1),
                4,
                1,
                fp,
            );
            if SWAP_DATA != 0 {
                swap((&mut ifd as *mut u32).cast(), 4);
            }
        }
        count
    }
}

/// C `read_tiffentries` (`tiff.c:432`).
pub unsafe fn read_tiffentries(fp: &mut ImodFile, tif: *mut TfInfo) -> i32 {
    unsafe {
        (*tif).nstrip = 1;
        b3d_fseek(fp, ((*tif).header.first_ifd_offset as i64) as i32, SEEK_SET);
        b3d_fread(
            core::slice::from_raw_parts_mut(
                (core::ptr::addr_of_mut!((*tif).numentries)).cast::<u8>(),
                (2) * (1),
            ),
            2,
            1,
            fp,
        );
        if SWAP_DATA != 0 {
            swap(core::ptr::addr_of_mut!((*tif).numentries).cast(), 2);
        }
        for _ in 0..(*tif).numentries {
            let (mut tag, mut typ, mut len, mut value) = (0u16, 0u16, 0u32, 0u32);
            if b3d_fread(
                core::slice::from_raw_parts_mut((&mut tag as *mut u16).cast::<u8>(), 2 * 1),
                2,
                1,
                fp,
            ) < 1
                || b3d_fread(
                    core::slice::from_raw_parts_mut((&mut typ as *mut u16).cast::<u8>(), 2 * 1),
                    2,
                    1,
                    fp,
                ) < 1
                || b3d_fread(
                    core::slice::from_raw_parts_mut((&mut len as *mut u32).cast::<u8>(), 4 * 1),
                    4,
                    1,
                    fp,
                ) < 1
                || b3d_fread(
                    core::slice::from_raw_parts_mut((&mut value as *mut u32).cast::<u8>(), 4 * 1),
                    4,
                    1,
                    fp,
                ) < 1
            {
                return 0;
            }
            if SWAP_DATA != 0 {
                swap((&mut tag as *mut u16).cast(), 2);
                swap((&mut typ as *mut u16).cast(), 2);
                swap((&mut len as *mut u32).cast(), 4);
                swap((&mut value as *mut u32).cast(), 4);
            }
            if cfg!(target_endian = "little") == false && typ == 3 && len < 3 {
                value >>= 16;
            }
            match tag {
                256 => {
                    (*tif).directory[1].value = value as i32;
                    (*tif).width = value as i32;
                }
                257 => {
                    (*tif).directory[2].value = value as i32;
                    (*tif).length = value as i32;
                }
                258 => {
                    if len == 1 {
                        (*tif).bits_per_sample = value as i32;
                        if value == 16 {
                            (*tif).mode = 2;
                        }
                    } else if len == 3 {
                        let pos = fp.tell();
                        b3d_fseek(fp, (value as i64) as i32, SEEK_SET);
                        let mut bits = 0u16;
                        b3d_fread(
                            core::slice::from_raw_parts_mut(
                                (&mut bits as *mut u16).cast::<u8>(),
                                2 * 1,
                            ),
                            2,
                            1,
                            fp,
                        );
                        if SWAP_DATA != 0 {
                            swap((&mut bits as *mut u16).cast(), 2);
                        }
                        (*tif).bits_per_sample = bits as i32;
                        (*tif).mode = 16;
                        b3d_fseek(fp, (pos) as i32, SEEK_SET);
                    }
                }
                259 if value != 1 => return 0,
                262 => {
                    if value == 3 {
                        return 0;
                    }
                    (*tif).photometric_interpretation = value as i32;
                }
                273 => {
                    (*tif).strip_pos = value as i32;
                    (*tif).nstrip = len as i32;
                }
                278 => (*tif).rows_per_strip = value as i32,
                279 => (*tif).strip_byte_counts = value as i32,
                324 | 325 => return 0,
                _ => {}
            }
        }
        (*tif).stripoff = libc::malloc(core::mem::size_of::<i32>() * (*tif).nstrip as usize).cast();
        (*tif).stripsize =
            libc::malloc(core::mem::size_of::<i32>() * (*tif).nstrip as usize).cast();
        if (*tif).stripoff.is_null() || (*tif).stripsize.is_null() {
            return 0;
        }
        if (*tif).nstrip == 1 {
            *(*tif).stripoff = (*tif).strip_pos;
            *(*tif).stripsize = (*tif).strip_byte_counts;
        } else {
            let pos = fp.tell();
            b3d_fseek(fp, ((*tif).strip_pos as i64) as i32, SEEK_SET);
            b3d_fread(
                core::slice::from_raw_parts_mut(
                    ((*tif).stripoff).cast::<u8>(),
                    (4) * ((*tif).nstrip as usize),
                ),
                4,
                (*tif).nstrip as usize,
                fp,
            );
            b3d_fseek(fp, ((*tif).strip_byte_counts as i64) as i32, SEEK_SET);
            b3d_fread(
                core::slice::from_raw_parts_mut(
                    ((*tif).stripsize).cast::<u8>(),
                    (4) * ((*tif).nstrip as usize),
                ),
                4,
                (*tif).nstrip as usize,
                fp,
            );
            b3d_fseek(fp, (pos) as i32, SEEK_SET);
            if SWAP_DATA != 0 {
                for i in 0..(*tif).nstrip {
                    swap((*tif).stripoff.add(i as usize).cast(), 4);
                    swap((*tif).stripsize.add(i as usize).cast(), 4);
                }
            }
        }
        1
    }
}

/// C `tiff_read_section` (`tiff.c:112`).
pub unsafe fn tiff_read_section(fp: &mut ImodFile, tif: *mut TfInfo, section: i32) -> *mut u8 {
    unsafe {
        if crate::imod::mrc::rust_tiff::contains(tif) {
            return crate::imod::mrc::rust_tiff::read_section(tif, section)
                .unwrap_or(core::ptr::null_mut());
        }
        if (*tif).iifile.is_null() {
            (*tif).header.first_ifd_offset = tiff_ifd(fp, section) as i32;
            if (*tif).header.first_ifd_offset == 0 || read_tiffentries(fp, tif) == 0 {
                return core::ptr::null_mut();
            }
        }
        let x = (*tif).directory[1].value;
        let y = (*tif).directory[2].value;
        let data_size = x as usize * y as usize;
        // `tiff.c:141-145`: the pixel size comes from BitsPerSample and is then
        // overridden for RGB.  Sub-byte data therefore leaves `pixSize` at 0.
        let mut pixel = (*tif).bits_per_sample / 8;
        if (*tif).photometric_interpretation == 2 {
            pixel = 3;
        }
        // `tiff.c:148-152` allocates `(data_size + xsize + ysize) * pixSize`.
        // With `pixSize` 0 that is a zero-byte block which the C then reads and
        // writes far beyond (see the 1-bit branch below), so native `tif2mrc`
        // has no defined output for sub-byte data.  Allocate the block the
        // caller will actually consume, cleared, rather than reproducing the
        // out-of-bounds access.
        let allocated = (data_size + x as usize + y as usize) * pixel.max(1) as usize;
        (*tif).data = if pixel > 0 {
            libc::malloc(allocated)
        } else {
            libc::calloc(allocated, 1)
        }
        .cast();
        if (*tif).data.is_null() {
            return core::ptr::null_mut();
        }
        if !(*tif).iifile.is_null() {
            // `tiff.c` delegates library-backed data to iiReadSection or
            // tiffReadSection; the native iimage dispatch owns that choice.
            if crate::imod::libiimod::iimage::ii_read_section(
                (*tif).iifile,
                (*tif).data.cast(),
                section,
            ) != 0
            {
                libc::free((*tif).data.cast());
                (*tif).data = core::ptr::null_mut();
            }
            return (*tif).data;
        }
        /* binary image. */
        if (*tif).bits_per_sample == 1 {
            // `tiff.c:167-189`.  `pixSize` is 0 on this path, so the C's
            // `fread(&bitdata[dpos], pixSize, tiff->stripsize[i], fp)`
            // transfers nothing and the expansion below runs over the
            // uninitialized `malloc(data_size)` block, writing `data_size`
            // bytes into a zero-byte `tiff->data`.  Native `tif2mrc` therefore
            // aborts in `free()` for any 1-bit TIFF that declares
            // BitsPerSample = 1 (verified: SIGABRT, "free(): invalid pointer").
            // The zero-length reads are kept; `bitdata` is cleared so this
            // produces a deterministic all-zero image instead of a crash.
            let bitdata = libc::calloc(data_size, 1).cast::<u8>();
            if bitdata.is_null() {
                libc::free((*tif).data.cast());
                (*tif).data = core::ptr::null_mut();
                return core::ptr::null_mut();
            }
            let mut dpos = 0_i32;
            for i in 0..(*tif).nstrip {
                b3d_fseek(
                    fp,
                    ((*tif).stripoff.add(i as usize).read()) as i32,
                    SEEK_SET,
                );
                b3d_fread(
                    core::slice::from_raw_parts_mut(
                        (bitdata.wrapping_add(dpos as usize)).cast::<u8>(),
                        (pixel as usize)
                            * ((*tif).stripsize.add(i as usize).read().max(0) as usize),
                    ),
                    pixel as usize,
                    (*tif).stripsize.add(i as usize).read().max(0) as usize,
                    fp,
                );
                dpos += (*tif).stripsize.add(i as usize).read();
            }
            for i in 0..data_size {
                // C reads through `char *bitdata` into an `int cbyte`, so the
                // byte is sign extended before the mask is applied.
                let cbyte = *bitdata.add(i / 8) as i8 as i32;
                let cbit = (i % 8) as i32;
                *(*tif).data.add(i) = if cbyte & ((1 << 7) >> cbit) != 0 {
                    0xff
                } else {
                    0x00
                };
            }
            libc::free(bitdata.cast());
            (*tif).bits_per_sample = 8;
        } else {
            // `tiff.c:190-211`.
            let mut nleft = x * y * pixel;
            let mut dpos = 0_i32;
            for i in 0..(*tif).nstrip {
                b3d_fseek(
                    fp,
                    ((*tif).stripoff.add(i as usize).read()) as i32,
                    SEEK_SET,
                );
                /* DNM 11/17/01: Gatan image did not limit size of last strip */
                // The `max(0)` has no counterpart in the C, which would pass a
                // negative count straight to `fread`.
                let mut realsize = (*tif).stripsize.add(i as usize).read().max(0);
                if realsize > nleft {
                    realsize = nleft;
                }
                /* DNM 12/10/00: was pixSize, needed to be 1 because stripsize
                is in bytes regardless of pixel size */
                b3d_fread(
                    core::slice::from_raw_parts_mut(
                        ((*tif).data.add(dpos as usize)).cast::<u8>(),
                        (1) * (realsize as usize),
                    ),
                    1,
                    realsize as usize,
                    fp,
                );
                /* DNM: swap the bytes if necessary */
                if (*tif).header.byteorder as u16
                    != if cfg!(target_endian = "little") {
                        0x4949
                    } else {
                        0x4d4d
                    }
                    && (pixel == 2 || pixel == 4)
                {
                    let mut at = 0;
                    while at + pixel <= realsize {
                        swap((*tif).data.add((dpos + at) as usize).cast(), pixel as u32);
                        at += pixel;
                    }
                }
                dpos += realsize;
                nleft -= realsize;
            }
        }
        for row in 0..y as usize / 2 {
            let a = (*tif).data.add(row * x as usize * pixel as usize);
            let b = (*tif)
                .data
                .add((y as usize - 1 - row) * x as usize * pixel as usize);
            for j in 0..x as usize * pixel as usize {
                let v = *a.add(j);
                *a.add(j) = *b.add(j);
                *b.add(j) = v;
            }
        }
        (*tif).data
    }
}

/// C `tiff_read_file` (`tiff.c:236`).
pub unsafe fn tiff_read_file(fp: &mut ImodFile, tif: *mut TfInfo) -> *mut u8 {
    unsafe {
        if !(*tif).iifile.is_null() {
            return tiff_read_section(fp, tif, 0);
        }
        b3d_rewind(fp);
        if read_tiffheader(fp, core::ptr::addr_of_mut!((*tif).header)) == 0 {
            core::ptr::null_mut()
        } else {
            tiff_read_section(fp, tif, 0)
        }
    }
}

/// C `tiff_read_mrc` (`tiff.c:96`).
pub unsafe fn tiff_read_mrc(fp: &mut ImodFile, hdata: *mut MrcHeader) -> *mut u8 {
    unsafe {
        let mut tif = TfInfo::default();
        let data = tiff_read_file(fp, &mut tif);
        if data.is_null() {
            return data;
        }
        mrc_head_new(
            &mut *hdata,
            tif.directory[1].value,
            tif.directory[2].value,
            1,
            MRC_MODE_BYTE,
        );
        libc::free(tif.stripoff.cast());
        libc::free(tif.stripsize.cast());
        data
    }
}

/// C `tiff_open_file` (`tiff.c:258`).
pub unsafe fn tiff_open_file(
    filename: *mut i8,
    mode: *mut i8,
    tif: *mut TfInfo,
    any_tif_pixel: i32,
) -> i32 {
    unsafe {
        if tif.is_null() {
            return 1;
        }
        match crate::imod::backends::tiff_backend() {
            Ok(crate::imod::backends::TiffBackend::Rust) => {
                let result = crate::imod::mrc::rust_tiff::open_file(filename, tif, any_tif_pixel);
                if result != 0 {
                    eprintln!(
                        "ERROR: Rust-native backend - Rust TIFF reader could not open or decode requested TIFF"
                    );
                }
                return result;
            }
            Ok(crate::imod::backends::TiffBackend::Parity) => {}
            Err(error) => {
                eprintln!("{error}");
                return 1;
            }
        }
        (*tif).fp = ImodFile::open(
            &core::ffi::CStr::from_ptr(filename).to_string_lossy(),
            &core::ffi::CStr::from_ptr(mode).to_string_lossy(),
        );
        if (*tif).fp.is_none() {
            return 1;
        }
        (*tif).iifile = crate::imod::libiimod::iimage::ii_new();
        if !(*tif).iifile.is_null() {
            (*(*tif).iifile).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(
                &core::ffi::CStr::from_ptr(filename).to_string_lossy(),
                &core::ffi::CStr::from_ptr(mode).to_string_lossy(),
            );
            (*(*tif).iifile).filename = libc::strdup(filename);
            core::ptr::copy_nonoverlapping(mode, (*(*tif).iifile).fmode.as_mut_ptr(), 3);
            (*(*tif).iifile).any_tiff_pix_size = any_tif_pixel;
            if crate::imod::libiimod::iitif::ii_tiff_check((*tif).iifile) != 0 {
                if (*(*tif).iifile).fp.is_none() {
                    (*tif).fp = ImodFile::open(
                        &core::ffi::CStr::from_ptr(filename).to_string_lossy(),
                        &core::ffi::CStr::from_ptr(mode).to_string_lossy(),
                    );
                }
                libc::free((*(*tif).iifile).filename.cast());
                drop(Box::from_raw((*tif).iifile));
                (*tif).iifile = core::ptr::null_mut();
                if (*tif).fp.is_none() {
                    return 1;
                }
                tiff_first_ifd(&mut (*tif).fp.clone().unwrap());
            } else {
                // Follow the successful `iiTIFFCheck` branch in the source
                // before releasing this presently incomplete libtiff reader
                // and continuing with the matching legacy reader below.  The
                // caller uses these properties to size its first chunk before
                // its later `tiff_read_file` call parses the legacy IFD.
                (*tif).bits_per_sample = 8;
                if (*(*tif).iifile).mode == MRC_MODE_SHORT
                    || (*(*tif).iifile).mode == MRC_MODE_USHORT
                {
                    (*tif).bits_per_sample = 16;
                }
                if (*(*tif).iifile).mode == MRC_MODE_FLOAT
                    || (*(*tif).iifile).type_ == crate::imod::libiimod::iimage::IITYPE_UINT
                    || (*(*tif).iifile).type_ == crate::imod::libiimod::iimage::IITYPE_INT
                {
                    (*tif).bits_per_sample = 32;
                }
                (*tif).photometric_interpretation = if (*(*tif).iifile).mode == MRC_MODE_RGB {
                    2
                } else {
                    1
                };
                (*tif).directory[1].value = (*(*tif).iifile).nx;
                (*tif).directory[2].value = (*(*tif).iifile).ny;
                (*tif).width = (*(*tif).iifile).nx;
                (*tif).length = (*(*tif).iifile).ny;
                (*(*tif).iifile).llx = 0;
                (*(*tif).iifile).lly = 0;
                (*(*tif).iifile).urx = -1;
                (*(*tif).iifile).ury = -1;
                // ii_tiff_check in this translation closes the probe FILE and
                // replaces iifile.fp with its libtiff handle.  This legacy source
                // unit still traverses IFDs through tiff->fp, so retain its own
                // ordinary FILE stream alongside the library reader.
                (*tif).fp = ImodFile::open(
                    &core::ffi::CStr::from_ptr(filename).to_string_lossy(),
                    &core::ffi::CStr::from_ptr(mode).to_string_lossy(),
                );
                if (*tif).fp.is_none() {
                    crate::imod::libiimod::iimage::ii_delete((*tif).iifile);
                    (*tif).iifile = core::ptr::null_mut();
                    return 1;
                }
            }
        }
        0
    }
}

/// C `tiff_close_file` (`tiff.c:316`).
pub unsafe fn tiff_close_file(tif: *mut TfInfo) {
    unsafe {
        if tif.is_null() {
            return;
        }
        let _ = crate::imod::mrc::rust_tiff::close_file(tif);
        if !(*tif).iifile.is_null() {
            crate::imod::libiimod::iimage::ii_delete((*tif).iifile);
        }
        (*tif).iifile = core::ptr::null_mut();
        (*tif).fp = None;
    }
}

/// C `tiff_write_entry` (`tiff.c:76`).
pub unsafe fn tiff_write_entry(
    tag: i16,
    type_: i16,
    length: i32,
    mut offset: u32,
    fout: &mut ImodFile,
) {
    unsafe {
        b3d_fwrite(
            core::slice::from_raw_parts(
                (&tag as *const i16).cast::<u8>(),
                (core::mem::size_of::<i16>()) * (1),
            ),
            core::mem::size_of::<i16>(),
            1,
            fout,
        );
        b3d_fwrite(
            core::slice::from_raw_parts(
                (&type_ as *const i16).cast::<u8>(),
                (core::mem::size_of::<i16>()) * (1),
            ),
            core::mem::size_of::<i16>(),
            1,
            fout,
        );
        b3d_fwrite(
            core::slice::from_raw_parts(
                (&length as *const i32).cast::<u8>(),
                (core::mem::size_of::<i32>()) * (1),
            ),
            core::mem::size_of::<i32>(),
            1,
            fout,
        );
        // The C source moves short inline values only on a big-endian host.
        if cfg!(target_endian = "big") && length == 1 {
            if type_ == 1 {
                offset <<= 24;
            } else if type_ == 3 {
                offset <<= 16;
            }
        }
        b3d_fwrite(
            core::slice::from_raw_parts(
                (&offset as *const u32).cast::<u8>(),
                (core::mem::size_of::<u32>()) * (1),
            ),
            core::mem::size_of::<u32>(),
            1,
            fout,
        );
    }
}

/// C `tiff_write_image` (`tiff.c:733`).
pub unsafe fn tiff_write_image(
    fout: &mut ImodFile,
    xsize: i32,
    ysize: i32,
    mode: i32,
    pixels: *mut u8,
    ifd_offset: *mut u32,
    data_offset: *mut u32,
    dmin: f32,
    dmax: f32,
) -> i32 {
    unsafe {
        if pixels.is_null() || ifd_offset.is_null() || data_offset.is_null() {
            return -1;
        }
        if *ifd_offset == 0 {
            let pixel: u32 = if cfg!(target_endian = "big") {
                0x4D4D002A
            } else {
                0x002A4949
            };
            b3d_fwrite(
                core::slice::from_raw_parts((&pixel as *const u32).cast::<u8>(), (4) * (1)),
                4,
                1,
                fout,
            );
            *ifd_offset = 4;
            *data_offset = 8;
        }
        if 0 != 0 {
            return -1;
        }
        let (pixel_size, sample_format) = match mode {
            MRC_MODE_BYTE => (1u32, 0u32),
            MRC_MODE_SHORT => (2, 2),
            MRC_MODE_USHORT => (2, 1),
            MRC_MODE_RGB => (3, 0),
            MRC_MODE_FLOAT => (4, 3),
            _ => return -50,
        };
        let data_size = (xsize as u32).wrapping_mul(ysize as u32);
        let mut ifd = data_size.wrapping_mul(pixel_size);
        let pad = 4i32 - (ifd % 4) as i32;
        ifd = ifd
            .wrapping_add(pad as u32)
            .wrapping_add(4)
            .wrapping_add(*data_offset);
        b3d_fseek(fout, (*ifd_offset as i64) as i32, SEEK_SET);
        if b3d_fwrite(
            core::slice::from_raw_parts((&ifd as *const u32).cast::<u8>(), (4) * (1)),
            4,
            1,
            fout,
        ) == 0
        {
            return -2;
        }
        b3d_fseek(fout, (*data_offset as i64) as i32, SEEK_SET);
        for y in (0..ysize).rev() {
            b3d_fwrite(
                core::slice::from_raw_parts(
                    (pixels.add((xsize * y) as usize * pixel_size as usize)).cast::<u8>(),
                    (pixel_size as usize) * (xsize as usize),
                ),
                pixel_size as usize,
                xsize as usize,
                fout,
            );
            if 0 != 0 {
                return -3;
            }
        }
        let zero = 0u32;
        b3d_fwrite(
            core::slice::from_raw_parts((&zero as *const u32).cast::<u8>(), (4) * (1)),
            4,
            1,
            fout,
        );
        if 0 != 0 {
            return -4;
        }
        b3d_fseek(fout, (ifd as i64) as i32, SEEK_SET);
        if mode != MRC_MODE_RGB {
            let entries: i16 = if mode == MRC_MODE_BYTE { 11 } else { 14 };
            b3d_fwrite(
                core::slice::from_raw_parts((&entries as *const i16).cast::<u8>(), (2) * (1)),
                2,
                1,
                fout,
            );
            tiff_write_entry(254, 4, 1, 0, fout);
            tiff_write_entry(256, 3, 1, xsize as u32, fout);
            tiff_write_entry(257, 3, 1, ysize as u32, fout);
            tiff_write_entry(258, 3, 1, 8 * pixel_size, fout);
            tiff_write_entry(259, 3, 1, 1, fout);
            tiff_write_entry(262, 3, 1, 1, fout);
            tiff_write_entry(273, 4, 1, *data_offset, fout);
            tiff_write_entry(277, 3, 1, 1, fout);
            tiff_write_entry(278, 4, 1, ysize as u32, fout);
            tiff_write_entry(279, 4, 1, data_size.wrapping_mul(pixel_size), fout);
            tiff_write_entry(296, 3, 1, 1, fout);
            if mode != MRC_MODE_BYTE {
                tiff_write_entry(339, 3, 1, sample_format, fout);
                tiff_write_entry(340, 11, 1, dmin.to_bits(), fout);
                tiff_write_entry(341, 11, 1, dmax.to_bits(), fout);
            }
            b3d_fwrite(
                core::slice::from_raw_parts((&zero as *const u32).cast::<u8>(), (4) * (1)),
                4,
                1,
                fout,
            );
            *ifd_offset = ifd + 2 + entries as u32 * 12;
            *data_offset = ifd + 6 + entries as u32 * 12;
        } else {
            let entries: i16 = 12;
            let bps: i16 = 8;
            b3d_fwrite(
                core::slice::from_raw_parts((&entries as *const i16).cast::<u8>(), (2) * (1)),
                2,
                1,
                fout,
            );
            tiff_write_entry(254, 4, 1, 0, fout);
            tiff_write_entry(256, 3, 1, xsize as u32, fout);
            tiff_write_entry(257, 3, 1, ysize as u32, fout);
            tiff_write_entry(258, 3, 3, ifd + 6 + entries as u32 * 12, fout);
            tiff_write_entry(259, 3, 1, 1, fout);
            tiff_write_entry(262, 3, 1, 2, fout);
            tiff_write_entry(273, 4, 1, *data_offset, fout);
            tiff_write_entry(277, 3, 1, 3, fout);
            tiff_write_entry(278, 4, 1, ysize as u32, fout);
            tiff_write_entry(279, 4, 1, data_size * 3, fout);
            tiff_write_entry(284, 3, 1, 1, fout);
            tiff_write_entry(296, 3, 1, 1, fout);
            b3d_fwrite(
                core::slice::from_raw_parts((&zero as *const u32).cast::<u8>(), (4) * (1)),
                4,
                1,
                fout,
            );
            b3d_fwrite(
                core::slice::from_raw_parts((&bps as *const i16).cast::<u8>(), (2) * (1)),
                2,
                1,
                fout,
            );
            b3d_fwrite(
                core::slice::from_raw_parts((&bps as *const i16).cast::<u8>(), (2) * (1)),
                2,
                1,
                fout,
            );
            b3d_fwrite(
                core::slice::from_raw_parts((&bps as *const i16).cast::<u8>(), (2) * (1)),
                2,
                1,
                fout,
            );
            *ifd_offset = ifd + 2 + entries as u32 * 12;
            *data_offset = ifd + 12 + entries as u32 * 12;
        }
        if 0 != 0 { -10 } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_writer_preserves_classic_tiff_orientation_and_ifd_chain() {
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut ifd = 0;
            let mut data = 0;
            // Source writes bottom row first, matching the old mrc2tif path.
            let mut pixels = [1u8, 2, 3, 4];
            assert_eq!(
                tiff_write_image(
                    &mut file,
                    2,
                    2,
                    MRC_MODE_BYTE,
                    pixels.as_mut_ptr(),
                    &mut ifd,
                    &mut data,
                    1.,
                    4.,
                ),
                0
            );
            {
                use std::io::Write;
                let _ = file.flush();
            }
            crate::imod::libcfshr::b3dutil::b3d_rewind(&mut file);
            let mut bytes = [0u8; 16];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(
                            bytes.as_mut_ptr().cast::<u8>(),
                            1 * (bytes.len()),
                        )
                    },
                    1,
                    bytes.len(),
                    &mut file,
                ),
                bytes.len()
            );
            assert_eq!(&bytes[..4], b"II*\0");
            // The first image payload starts at byte 8 and is vertically inverted.
            assert_eq!(&bytes[8..12], &[3, 4, 1, 2]);
            assert_ne!(ifd, 0);
            assert_ne!(data, 0);
            drop(file);
        }
    }

    #[test]
    fn legacy_reader_roundtrips_uncompressed_byte_pixels() {
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut ifd = 0;
            let mut data = 0;
            let mut pixels = [1u8, 2, 3, 4];
            assert_eq!(
                tiff_write_image(
                    &mut file,
                    2,
                    2,
                    MRC_MODE_BYTE,
                    pixels.as_mut_ptr(),
                    &mut ifd,
                    &mut data,
                    1.,
                    4.
                ),
                0
            );
            {
                use std::io::Write;
                let _ = file.flush();
            }
            crate::imod::libcfshr::b3dutil::b3d_rewind(&mut file);
            let mut tif = TfInfo::default();
            let result = tiff_read_file(&mut file, &mut tif);
            assert!(!result.is_null());
            assert_eq!(core::slice::from_raw_parts(result, 4), &[1, 2, 3, 4]);
            libc::free(result.cast());
            libc::free(tif.stripoff.cast());
            libc::free(tif.stripsize.cast());
            drop(file);
        }
    }

    #[test]
    fn legacy_reader_transfers_nothing_for_one_bit_strips() {
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            // Little-endian classic TIFF: 8 by 1, a one-bit grayscale strip
            // at byte 126 holding 10110010, with BitsPerSample = 1.
            let bytes: [u8; 127] = [
                b'I',
                b'I',
                42,
                0,
                8,
                0,
                0,
                0,
                9,
                0,
                0,
                1,
                4,
                0,
                1,
                0,
                0,
                0,
                8,
                0,
                0,
                0,
                1,
                1,
                4,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                2,
                1,
                3,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                3,
                1,
                3,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                6,
                1,
                3,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                17,
                1,
                4,
                0,
                1,
                0,
                0,
                0,
                126,
                0,
                0,
                0,
                21,
                1,
                3,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                22,
                1,
                4,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                23,
                1,
                4,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0b1011_0010,
            ];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    unsafe {
                        core::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), 1 * (bytes.len()))
                    },
                    1,
                    bytes.len(),
                    &mut file,
                ),
                bytes.len()
            );
            crate::imod::libcfshr::b3dutil::b3d_rewind(&mut file);
            let mut tif = TfInfo::default();
            let result = tiff_read_file(&mut file, &mut tif);
            assert!(!result.is_null());
            // `tiff.c:141` sets `pixSize = BitsPerSample / 8`, which is 0 here,
            // so the strip read at `tiff.c:174` is `fread(ptr, 0, stripsize,
            // fp)` and transfers nothing: the packed 0b1011_0010 byte never
            // reaches the expansion loop and every output pixel is 0.  Native
            // `tif2mrc` cannot get this far - the expansion writes xsize*ysize
            // bytes into the zero-byte `malloc` from `tiff.c:150`, and the
            // binary aborts in `free()` ("free(): invalid pointer", exit 134)
            // on every 1-bit TIFF that declares BitsPerSample = 1.  There is no
            // defined native output to match, so this only pins the C's
            // zero-length reads.
            assert_eq!(core::slice::from_raw_parts(result, 8), &[0; 8]);
            libc::free(result.cast());
            libc::free(tif.stripoff.cast());
            libc::free(tif.stripsize.cast());
            drop(file);
        }
    }

    #[test]
    fn open_file_keeps_libtiff_image_handle_for_a_real_classic_tiff() {
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-legacy-tiff-open-{}.tif",
                std::process::id()
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let write_mode = c"wb".as_ptr().cast_mut();
            let mut fout =
                crate::imod::libcfshr::b3dutil::ImodFile::open(&name.to_string_lossy(), "wb")
                    .unwrap();
            let mut ifd = 0;
            let mut data = 0;
            let mut pixels = [1_u8, 2, 3, 4];
            assert_eq!(
                tiff_write_image(
                    &mut fout,
                    2,
                    2,
                    MRC_MODE_BYTE,
                    pixels.as_mut_ptr(),
                    &mut ifd,
                    &mut data,
                    1.,
                    4.
                ),
                0
            );
            drop(fout);

            let mut tif = TfInfo::default();
            let read_mode = c"rb".as_ptr().cast_mut();
            assert_eq!(
                tiff_open_file(name.as_ptr().cast_mut(), read_mode, &mut tif, 0),
                0
            );
            assert!(!tif.iifile.is_null());
            assert_eq!((tif.width, tif.length, tif.bits_per_sample), (2, 2, 8));
            let mut fp = tif.fp.clone().unwrap();
            let result = tiff_read_file(&mut fp, &mut tif);
            assert!(!result.is_null());
            assert_eq!(core::slice::from_raw_parts(result, 4), &[1, 2, 3, 4]);
            libc::free(result.cast());
            tiff_close_file(&mut tif);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn open_file_keeps_libtiff_reader_for_a_multidirectory_classic_tiff_stack() {
        unsafe {
            let path = std::env::temp_dir().join(format!(
                "imod-rs-legacy-tiff-stack-{}.tif",
                std::process::id()
            ));
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            let mut fout =
                crate::imod::libcfshr::b3dutil::ImodFile::open(&name.to_string_lossy(), "wb")
                    .unwrap();
            let mut ifd = 0;
            let mut data = 0;
            let mut first = [1_u8, 2, 3, 4];
            let mut second = [5_u8, 6, 7, 8];
            assert_eq!(
                tiff_write_image(
                    &mut fout,
                    2,
                    2,
                    MRC_MODE_BYTE,
                    first.as_mut_ptr(),
                    &mut ifd,
                    &mut data,
                    1.,
                    4.
                ),
                0
            );
            assert_eq!(
                tiff_write_image(
                    &mut fout,
                    2,
                    2,
                    MRC_MODE_BYTE,
                    second.as_mut_ptr(),
                    &mut ifd,
                    &mut data,
                    5.,
                    8.
                ),
                0
            );
            drop(fout);

            let mut tif = TfInfo::default();
            assert_eq!(
                tiff_open_file(
                    name.as_ptr().cast_mut(),
                    c"rb".as_ptr().cast_mut(),
                    &mut tif,
                    0
                ),
                0
            );
            assert!(!tif.iifile.is_null());
            assert_eq!((*tif.iifile).nz, 2);
            tiff_close_file(&mut tif);
            std::fs::remove_file(path).unwrap();
        }
    }
}
