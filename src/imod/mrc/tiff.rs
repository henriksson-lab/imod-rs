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
    /// C `char comment[128]` / `char extra[128]` (`b3dtiff.h:40-41`), two
    /// fixed-width on-disk fields.  Bytes, not a `String`: the padding past
    /// the text is part of the field (NATIVE.md section 3).
    pub comment: [u8; 128],
    pub extra: [u8; 128],
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
    // C `Tf_info.data` (`b3dtiff.h:51`) is deliberately absent.  It holds the
    // block `tiff_read_section` has just `malloc`ed and returns, and the
    // *caller* frees it (`tif2mrc.c:747`, `:1047`) -- the field is never read
    // again after the call.  With `Vec` ownership the returned buffer is that
    // block, so keeping the field would only leave a duplicate handle to
    // memory the caller already owns.
    /// C `Tf_info.nstrip`, `.stripoff` and `.stripsize` (`b3dtiff.h:52-54`).
    /// The two arrays are `malloc`ed per IFD in `read_tiffentries`
    /// (`tiff.c:534-536`) and freed only by `tiff_read_mrc` (`tiff.c:106-107`),
    /// so the C leaks one pair per section of a multi-page read; a `Vec`
    /// releases the previous pair when the next IFD replaces it.  Nothing
    /// observable depends on that, and the element values are unchanged.
    pub nstrip: i32,
    pub stripoff: Vec<i32>,
    pub stripsize: Vec<i32>,
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
            nstrip: 0,
            stripoff: Vec::new(),
            stripsize: Vec::new(),
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
///
/// The C takes `char *ptr, int size`; the byte range is the argument here, so
/// the length travels with it.  An odd `size` is decremented first, which
/// leaves the last byte of the range in place.
fn swap(bytes: &mut [u8]) {
    let mut size = bytes.len();
    if size % 2 != 0 {
        size -= 1;
    }
    for index in 0..size / 2 {
        bytes.swap(index, size - 1 - index);
    }
}

/// C `isit_tiff` (`tiff.c:59`).
pub fn isit_tiff(fp: &mut ImodFile) -> i32 {
    let mut raw = [0u8; 2];
    b3d_rewind(fp);
    if b3d_fread(&mut raw, 2, 1, fp) < 1 {
        return 0;
    }
    let mut word = u16::from_ne_bytes(raw);
    if word != 0x4949 && word != 0x4d4d {
        return 0;
    }
    if b3d_fread(&mut raw, 2, 1, fp) < 1 {
        return 0;
    }
    word = u16::from_ne_bytes(raw);
    if word != 0x0042 && word != 0x4200 {
        return 0;
    }
    1
}

/// C `tiffFirstIFD` (`tiff.c:347`).
pub unsafe fn tiff_first_ifd(fp: &mut ImodFile) -> u32 {
    unsafe {
        let mut short_raw = [0u8; 2];
        let mut long_raw = [0u8; 4];
        b3d_rewind(fp);
        if b3d_fread(&mut short_raw, 2, 1, fp) < 1 {
            return 0;
        }
        let word = u16::from_ne_bytes(short_raw);
        if word != 0x4949 && word != 0x4d4d {
            return 0;
        }
        SWAP_DATA = (word
            != if cfg!(target_endian = "little") {
                0x4949
            } else {
                0x4d4d
            }) as i32;
        if b3d_fread(&mut short_raw, 2, 1, fp) < 1 {
            return 0;
        }
        if SWAP_DATA != 0 {
            swap(&mut short_raw);
        }
        b3d_fread(&mut long_raw, 4, 1, fp);
        if SWAP_DATA != 0 {
            swap(&mut long_raw);
        }
        u32::from_ne_bytes(long_raw)
    }
}

/// C `read_tiffheader` (`tiff.c:325`).
pub unsafe fn read_tiffheader(fp: &mut ImodFile, header: &mut TfHeader) -> i32 {
    unsafe {
        // `tiff.c:325`: `fread(header, sizeof(Tf_header), 1, fp)` -- the eight
        // bytes of the on-disk header, unpacked field by field rather than
        // read over the struct, so the transfer does not depend on the Rust
        // layout.
        let mut raw = [0u8; 8];
        if b3d_fread(&mut raw, 8, 1, fp) < 1 {
            return 0;
        }
        let mut order_raw = [raw[0], raw[1]];
        let mut version_raw = [raw[2], raw[3]];
        let mut offset_raw = [raw[4], raw[5], raw[6], raw[7]];
        let order = u16::from_ne_bytes(order_raw);
        if order != 0x4d4d && order != 0x4949 {
            header.byteorder = i16::from_ne_bytes(order_raw);
            header.version = i16::from_ne_bytes(version_raw);
            header.first_ifd_offset = i32::from_ne_bytes(offset_raw);
            return 0;
        }
        let machine = if cfg!(target_endian = "little") {
            0x4949
        } else {
            0x4d4d
        };
        if order != machine {
            swap(&mut version_raw);
            swap(&mut offset_raw);
        }
        header.byteorder = i16::from_ne_bytes(order_raw);
        header.version = i16::from_ne_bytes(version_raw);
        header.first_ifd_offset = tiff_first_ifd(fp) as i32;
        if header.version != 42 { 0 } else { 1 }
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
            let mut entries_raw = [0u8; 2];
            b3d_fseek(fp, (ifd as i64) as i32, SEEK_SET);
            b3d_fread(&mut entries_raw, 2, 1, fp);
            if SWAP_DATA != 0 {
                swap(&mut entries_raw);
            }
            let entries = u16::from_ne_bytes(entries_raw);
            b3d_fseek(
                fp,
                ((ifd + 2 + entries as u32 * 12) as i64) as i32,
                SEEK_SET,
            );
            let mut ifd_raw = [0u8; 4];
            b3d_fread(&mut ifd_raw, 4, 1, fp);
            if SWAP_DATA != 0 {
                swap(&mut ifd_raw);
            }
            ifd = u32::from_ne_bytes(ifd_raw);
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
            let mut entries_raw = [0u8; 2];
            b3d_fseek(fp, (ifd as i64) as i32, SEEK_SET);
            b3d_fread(&mut entries_raw, 2, 1, fp);
            if SWAP_DATA != 0 {
                swap(&mut entries_raw);
            }
            let entries = u16::from_ne_bytes(entries_raw);
            count += 1;
            b3d_fseek(
                fp,
                ((ifd + 2 + entries as u32 * 12) as i64) as i32,
                SEEK_SET,
            );
            let mut ifd_raw = [0u8; 4];
            b3d_fread(&mut ifd_raw, 4, 1, fp);
            if SWAP_DATA != 0 {
                swap(&mut ifd_raw);
            }
            ifd = u32::from_ne_bytes(ifd_raw);
        }
        count
    }
}

/// C `read_tiffentries` (`tiff.c:432`).
pub unsafe fn read_tiffentries(fp: &mut ImodFile, tif: &mut TfInfo) -> i32 {
    unsafe {
        (*tif).nstrip = 1;
        b3d_fseek(fp, ((*tif).header.first_ifd_offset as i64) as i32, SEEK_SET);
        let mut numentries_raw = [0u8; 2];
        b3d_fread(&mut numentries_raw, 2, 1, fp);
        if SWAP_DATA != 0 {
            swap(&mut numentries_raw);
        }
        (*tif).numentries = i16::from_ne_bytes(numentries_raw);
        for _ in 0..(*tif).numentries {
            let mut tag_raw = [0u8; 2];
            let mut typ_raw = [0u8; 2];
            let mut len_raw = [0u8; 4];
            let mut value_raw = [0u8; 4];
            if b3d_fread(&mut tag_raw, 2, 1, fp) < 1
                || b3d_fread(&mut typ_raw, 2, 1, fp) < 1
                || b3d_fread(&mut len_raw, 4, 1, fp) < 1
                || b3d_fread(&mut value_raw, 4, 1, fp) < 1
            {
                return 0;
            }
            if SWAP_DATA != 0 {
                swap(&mut tag_raw);
                swap(&mut typ_raw);
                swap(&mut len_raw);
                swap(&mut value_raw);
            }
            let tag = u16::from_ne_bytes(tag_raw);
            let typ = u16::from_ne_bytes(typ_raw);
            let len = u32::from_ne_bytes(len_raw);
            let mut value = u32::from_ne_bytes(value_raw);
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
                        let mut bits_raw = [0u8; 2];
                        b3d_fread(&mut bits_raw, 2, 1, fp);
                        if SWAP_DATA != 0 {
                            swap(&mut bits_raw);
                        }
                        (*tif).bits_per_sample = u16::from_ne_bytes(bits_raw) as i32;
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
        // `tiff.c:534-536`: `malloc(sizeof(int) * tiff->nstrip)` for each.
        // A negative `nstrip` cannot reach here -- it comes from an unsigned
        // IFD length field -- so the cast is the C's own implicit one.
        (*tif).stripoff = vec![0_i32; (*tif).nstrip.max(0) as usize];
        (*tif).stripsize = vec![0_i32; (*tif).nstrip.max(0) as usize];
        if (*tif).nstrip == 1 {
            (*tif).stripoff[0] = (*tif).strip_pos;
            (*tif).stripsize[0] = (*tif).strip_byte_counts;
        } else {
            let pos = fp.tell();
            b3d_fseek(fp, ((*tif).strip_pos as i64) as i32, SEEK_SET);
            let nstrip = (*tif).nstrip.max(0) as usize;
            b3d_fread(
                core::slice::from_raw_parts_mut(
                    (*tif).stripoff.as_mut_ptr().cast::<u8>(),
                    (4) * nstrip,
                ),
                4,
                nstrip,
                fp,
            );
            b3d_fseek(fp, ((*tif).strip_byte_counts as i64) as i32, SEEK_SET);
            b3d_fread(
                core::slice::from_raw_parts_mut(
                    (*tif).stripsize.as_mut_ptr().cast::<u8>(),
                    (4) * nstrip,
                ),
                4,
                nstrip,
                fp,
            );
            b3d_fseek(fp, (pos) as i32, SEEK_SET);
            if SWAP_DATA != 0 {
                for i in 0..nstrip {
                    (*tif).stripoff[i] = (*tif).stripoff[i].swap_bytes();
                    (*tif).stripsize[i] = (*tif).stripsize[i].swap_bytes();
                }
            }
        }
        1
    }
}

/// C `tiff_read_section` (`tiff.c:112`).
pub unsafe fn tiff_read_section(
    fp: &mut ImodFile,
    tif: &mut TfInfo,
    section: i32,
) -> Option<Vec<u8>> {
    unsafe {
        if crate::imod::mrc::rust_tiff::contains(tif) {
            return crate::imod::mrc::rust_tiff::read_section(tif, section);
        }
        if (*tif).iifile.is_null() {
            (*tif).header.first_ifd_offset = tiff_ifd(fp, section) as i32;
            if (*tif).header.first_ifd_offset == 0 || read_tiffentries(fp, tif) == 0 {
                return None;
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
        // The C distinguishes `malloc` from `calloc` only to keep the sub-byte
        // branch's over-read deterministic; a `Vec` is zeroed either way.
        let mut data = vec![0_u8; allocated];
        if !(*tif).iifile.is_null() {
            // `tiff.c` delegates library-backed data to iiReadSection or
            // tiffReadSection; the native iimage dispatch owns that choice.
            if crate::imod::libiimod::iimage::ii_read_section(
                (*tif).iifile,
                data.as_mut_ptr().cast(),
                section,
            ) != 0
            {
                return None;
            }
            return Some(data);
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
            let mut bitdata = vec![0_u8; data_size];
            let mut dpos = 0_i32;
            for i in 0..(*tif).nstrip as usize {
                b3d_fseek(fp, ((*tif).stripoff[i]) as i32, SEEK_SET);
                let count = (*tif).stripsize[i].max(0) as usize;
                b3d_fread(
                    core::slice::from_raw_parts_mut(
                        bitdata.as_mut_ptr().wrapping_add(dpos as usize),
                        (pixel as usize) * count,
                    ),
                    pixel as usize,
                    count,
                    fp,
                );
                dpos += (*tif).stripsize[i];
            }
            for i in 0..data_size {
                // C reads through `char *bitdata` into an `int cbyte`, so the
                // byte is sign extended before the mask is applied.
                let cbyte = bitdata[i / 8] as i8 as i32;
                let cbit = (i % 8) as i32;
                data[i] = if cbyte & ((1 << 7) >> cbit) != 0 {
                    0xff
                } else {
                    0x00
                };
            }
            (*tif).bits_per_sample = 8;
        } else {
            // `tiff.c:190-211`.
            let mut nleft = x * y * pixel;
            let mut dpos = 0_i32;
            for i in 0..(*tif).nstrip as usize {
                b3d_fseek(fp, ((*tif).stripoff[i]) as i32, SEEK_SET);
                /* DNM 11/17/01: Gatan image did not limit size of last strip */
                // The `max(0)` has no counterpart in the C, which would pass a
                // negative count straight to `fread`.
                let mut realsize = (*tif).stripsize[i].max(0);
                if realsize > nleft {
                    realsize = nleft;
                }
                /* DNM 12/10/00: was pixSize, needed to be 1 because stripsize
                is in bytes regardless of pixel size */
                b3d_fread(
                    core::slice::from_raw_parts_mut(
                        data.as_mut_ptr().add(dpos as usize),
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
                        let start = (dpos + at) as usize;
                        swap(&mut data[start..start + pixel as usize]);
                        at += pixel;
                    }
                }
                dpos += realsize;
                nleft -= realsize;
            }
        }
        for row in 0..y as usize / 2 {
            let a = row * x as usize * pixel as usize;
            let b = (y as usize - 1 - row) * x as usize * pixel as usize;
            for j in 0..x as usize * pixel as usize {
                data.swap(a + j, b + j);
            }
        }
        Some(data)
    }
}

/// C `tiff_read_file` (`tiff.c:236`).
pub unsafe fn tiff_read_file(fp: &mut ImodFile, tif: &mut TfInfo) -> Option<Vec<u8>> {
    unsafe {
        if !(*tif).iifile.is_null() {
            return tiff_read_section(fp, tif, 0);
        }
        b3d_rewind(fp);
        if read_tiffheader(fp, &mut (*tif).header) == 0 {
            None
        } else {
            tiff_read_section(fp, tif, 0)
        }
    }
}

/// C `tiff_read_mrc` (`tiff.c:96`).
pub unsafe fn tiff_read_mrc(fp: &mut ImodFile, hdata: &mut MrcHeader) -> Option<Vec<u8>> {
    unsafe {
        let mut tif = TfInfo::default();
        let data = tiff_read_file(fp, &mut tif)?;
        mrc_head_new(
            hdata,
            tif.directory[1].value,
            tif.directory[2].value,
            1,
            MRC_MODE_BYTE,
        );
        // `tiff.c:106-107` frees `tiff.stripoff` and `tiff.stripsize` here;
        // the two `Vec`s are released when `tif` goes out of scope.
        Some(data)
    }
}

/// C `tiff_open_file` (`tiff.c:258`).
pub unsafe fn tiff_open_file(
    filename: &[u8],
    mode: &str,
    tif: &mut TfInfo,
    any_tif_pixel: i32,
) -> i32 {
    unsafe {
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
        let path = String::from_utf8_lossy(filename).into_owned();
        (*tif).fp = ImodFile::open(&path, mode);
        if (*tif).fp.is_none() {
            return 1;
        }
        (*tif).iifile = crate::imod::libiimod::iimage::ii_new();
        if !(*tif).iifile.is_null() {
            (*(*tif).iifile).fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, mode);
            (*(*tif).iifile).filename = Some(filename.to_vec());
            // `tiff.c:274`: `strncpy(tiff->iifile->fmode, mode, 3)`, into a
            // four-byte array that `iiNew` has already cleared.
            for (index, byte) in mode.as_bytes().iter().take(3).enumerate() {
                (*(*tif).iifile).fmode[index] = *byte;
            }
            (*(*tif).iifile).any_tiff_pix_size = any_tif_pixel;
            if crate::imod::libiimod::iitif::ii_tiff_check((*tif).iifile) != 0 {
                if (*(*tif).iifile).fp.is_none() {
                    (*tif).fp = ImodFile::open(&path, mode);
                }
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
                (*tif).fp = ImodFile::open(&path, mode);
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
pub unsafe fn tiff_close_file(tif: &mut TfInfo) {
    unsafe {
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
        b3d_fwrite(&tag.to_ne_bytes(), core::mem::size_of::<i16>(), 1, fout);
        b3d_fwrite(&type_.to_ne_bytes(), core::mem::size_of::<i16>(), 1, fout);
        b3d_fwrite(&length.to_ne_bytes(), core::mem::size_of::<i32>(), 1, fout);
        // The C source moves short inline values only on a big-endian host.
        if cfg!(target_endian = "big") && length == 1 {
            if type_ == 1 {
                offset <<= 24;
            } else if type_ == 3 {
                offset <<= 16;
            }
        }
        b3d_fwrite(&offset.to_ne_bytes(), core::mem::size_of::<u32>(), 1, fout);
    }
}

/// C `tiff_write_image` (`tiff.c:733`).
pub unsafe fn tiff_write_image(
    fout: &mut ImodFile,
    xsize: i32,
    ysize: i32,
    mode: i32,
    pixels: &[u8],
    ifd_offset: &mut u32,
    data_offset: &mut u32,
    dmin: f32,
    dmax: f32,
) -> i32 {
    unsafe {
        if *ifd_offset == 0 {
            let pixel: u32 = if cfg!(target_endian = "big") {
                0x4D4D002A
            } else {
                0x002A4949
            };
            b3d_fwrite(&pixel.to_ne_bytes(), 4, 1, fout);
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
        if b3d_fwrite(&ifd.to_ne_bytes(), 4, 1, fout) == 0 {
            return -2;
        }
        b3d_fseek(fout, (*data_offset as i64) as i32, SEEK_SET);
        for y in (0..ysize).rev() {
            b3d_fwrite(
                core::slice::from_raw_parts(
                    pixels
                        .as_ptr()
                        .add((xsize * y) as usize * pixel_size as usize),
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
        b3d_fwrite(&zero.to_ne_bytes(), 4, 1, fout);
        if 0 != 0 {
            return -4;
        }
        b3d_fseek(fout, (ifd as i64) as i32, SEEK_SET);
        if mode != MRC_MODE_RGB {
            let entries: i16 = if mode == MRC_MODE_BYTE { 11 } else { 14 };
            b3d_fwrite(&entries.to_ne_bytes(), 2, 1, fout);
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
            b3d_fwrite(&zero.to_ne_bytes(), 4, 1, fout);
            *ifd_offset = ifd + 2 + entries as u32 * 12;
            *data_offset = ifd + 6 + entries as u32 * 12;
        } else {
            let entries: i16 = 12;
            let bps: i16 = 8;
            b3d_fwrite(&entries.to_ne_bytes(), 2, 1, fout);
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
            b3d_fwrite(&zero.to_ne_bytes(), 4, 1, fout);
            b3d_fwrite(&bps.to_ne_bytes(), 2, 1, fout);
            b3d_fwrite(&bps.to_ne_bytes(), 2, 1, fout);
            b3d_fwrite(&bps.to_ne_bytes(), 2, 1, fout);
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
                    &pixels,
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
                    &pixels,
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
            let result = tiff_read_file(&mut file, &mut tif).expect("legacy reader returned data");
            assert_eq!(&result[..4], &[1, 2, 3, 4]);
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
            let result = tiff_read_file(&mut file, &mut tif).expect("one-bit reader returned data");
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
            assert_eq!(&result[..8], &[0; 8]);
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
            let name = path.to_str().unwrap().to_owned();
            let mut fout = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "wb").unwrap();
            let mut ifd = 0;
            let mut data = 0;
            let mut pixels = [1_u8, 2, 3, 4];
            assert_eq!(
                tiff_write_image(
                    &mut fout,
                    2,
                    2,
                    MRC_MODE_BYTE,
                    &pixels,
                    &mut ifd,
                    &mut data,
                    1.,
                    4.
                ),
                0
            );
            drop(fout);

            let mut tif = TfInfo::default();
            assert_eq!(tiff_open_file(name.as_bytes(), "rb", &mut tif, 0), 0);
            assert!(!tif.iifile.is_null());
            assert_eq!((tif.width, tif.length, tif.bits_per_sample), (2, 2, 8));
            let mut fp = tif.fp.clone().unwrap();
            let result = tiff_read_file(&mut fp, &mut tif).expect("libtiff reader returned data");
            assert_eq!(&result[..4], &[1, 2, 3, 4]);
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
            let name = path.to_str().unwrap().to_owned();
            let mut fout = crate::imod::libcfshr::b3dutil::ImodFile::open(&name, "wb").unwrap();
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
                    &first,
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
                    &second,
                    &mut ifd,
                    &mut data,
                    5.,
                    8.
                ),
                0
            );
            drop(fout);

            let mut tif = TfInfo::default();
            assert_eq!(tiff_open_file(name.as_bytes(), "rb", &mut tif, 0), 0);
            assert!(!tif.iifile.is_null());
            assert_eq!((*tif.iifile).nz, 2);
            tiff_close_file(&mut tif);
            std::fs::remove_file(path).unwrap();
        }
    }
}
