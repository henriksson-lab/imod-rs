//! Type scaffold for `IMOD/include/mrcfiles.h` and implementation counterpart
//! `IMOD/libiimod/mrcfiles.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{
    CArg, SEEK_CUR, SEEK_SET, b3d_error, b3d_fread, b3d_fseek, b3d_fwrite, b3d_rewind,
    c_format_bytes, data_size_for_mode, extra_is_nbytes_and_flags, fgetline, imod_backup_file,
    invert_mrc_origin_on_output, mrc_huge_seek, read_bytes_signed, set_or_clear_flags,
    write_4_bit_mode_for_bytes, write_16_bit_mode_for_floats, write_bytes_signed,
};
use crate::imod::libiimod::iimage::{
    IIFILE_MRC, IIFILE_RAW, ii_fill_mrc_header, ii_lookup_file_from_fp, ii_sync_from_mrc_header,
    ii_write_header,
};
use crate::imod::libiimod::mrcsec::{mrc_read_z_byte, mrc_read_z_float};
use chrono::{Datelike, Local};
use std::io::{Read, Write};

pub const MRC_IDTYPE_MONO: i32 = 0;
pub const MRC_IDTYPE_TILT: i32 = 1;
pub const MRC_IDTYPE_TILTS: i32 = 2;
pub const MRC_IDTYPE_LINA: i32 = 3;
pub const MRC_IDTYPE_LINS: i32 = 4;
pub const MRC_SCALE_LINEAR: i32 = 1;
pub const MRC_SCALE_POWER: i32 = 2;
pub const MRC_SCALE_LOG: i32 = 3;
pub const MRC_SCALE_BKG: i32 = 4;
pub const MRC_MODE_BYTE: i32 = 0;
pub const MRC_MODE_SHORT: i32 = 1;
pub const MRC_MODE_FLOAT: i32 = 2;
pub const MRC_MODE_COMPLEX_SHORT: i32 = 3;
pub const MRC_MODE_COMPLEX_FLOAT: i32 = 4;
pub const MRC_MODE_USHORT: i32 = 6;
pub const MRC_MODE_HALF_FLOAT: i32 = 12;
pub const MRC_MODE_RGB: i32 = 16;
pub const MRC_MODE_4BIT: i32 = 101;
pub const MRC_RAMP_LIN: i32 = 1;
pub const MRC_RAMP_EXP: i32 = 2;
pub const MRC_RAMP_LOG: i32 = 3;
pub const MRC_NEXTRA: usize = 16;
pub const MRC_MAXCSIZE: usize = 3;
pub const MRC_LABEL_SIZE: usize = 80;
pub const MRC_NLABELS: usize = 10;
pub const MRC_HEADER_SIZE: usize = 1024;
pub const MRC_EXT_TYPE_NONE: i32 = 0;
pub const MRC_EXT_TYPE_SERI: i32 = 1;
pub const MRC_EXT_TYPE_AGAR: i32 = 2;
pub const MRC_EXT_TYPE_FEI: i32 = 3;
pub const MRC_EXT_TYPE_UNKNOWN: i32 = 4;
pub const IMOD_MRC_STAMP: i32 = 1_146_047_817;
pub const MRC_FLAGS_BAD_RMS_NEG: i32 = 8;
pub const MRC_FLAGS_SBYTES: i32 = 1;
pub const MRC_FLAGS_INV_ORIGIN: i32 = 4;
pub const MRC_FLAGS_4BIT_BYTES: i32 = 32;
pub const PACKED_HALF_XSIZE: i32 = 2;
pub const IIUNIT_SWAPPED: i32 = 1;
pub const IIUNIT_BYTES_SIGNED: i32 = 2;
pub const IIUNIT_OLD_STYLE: i32 = 4;
pub const IIUNIT_NINT_BUG: i32 = 8;
pub const IIUNIT_BAD_MAPCRS: i32 = 16;
pub const IIUNIT_4BIT_MODE: i32 = 32;
pub const IIUNIT_HALF_XSIZE: i32 = 64;
pub const IIUNIT_Y_INVERTED: i32 = 128;
pub const IIUNIT_HALF_FLOATS: i32 = 256;
pub const PACKED_4BIT_MODE: i32 = 1;

/// A complex floating-point MRC pixel.
///
/// MRC pixel bytes are read and written explicitly; this Rust value never
/// crosses a foreign ABI boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComplexFloat {
    pub a: f32,
    pub b: f32,
}
/// A complex 16-bit MRC pixel.
///
/// MRC pixel bytes are read and written explicitly; this Rust value never
/// crosses a foreign ABI boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComplexShort {
    pub a: i16,
    pub b: i16,
}

/// MRC header plus the Rust-owned state used while the image is open.
///
/// The first 224 on-disk bytes and the ten labels are explicitly decoded and
/// encoded in [`mrc_head_read`] and [`mrc_head_write`].  The in-memory header
/// therefore has no C-layout or ABI contract.
#[derive(Clone)]
pub struct MrcHeader {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub mode: i32,
    pub nxstart: i32,
    pub nystart: i32,
    pub nzstart: i32,
    pub mx: i32,
    pub my: i32,
    pub mz: i32,
    pub xlen: f32,
    pub ylen: f32,
    pub zlen: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub mapc: i32,
    pub mapr: i32,
    pub maps: i32,
    pub amin: f32,
    pub amax: f32,
    pub amean: f32,
    pub ispg: i32,
    pub next: i32,
    pub creatid: i16,
    pub blank: [u8; 6],
    pub ext_type: [u8; 4],
    pub nversion: i32,
    pub blank2: [u8; 16],
    pub nint: i16,
    pub nreal: i16,
    pub sub: i16,
    pub zfac: i16,
    pub min2: f32,
    pub max2: f32,
    pub min3: f32,
    pub max3: f32,
    pub imod_stamp: i32,
    pub imod_flags: i32,
    pub idtype: i16,
    pub lens: i16,
    pub nd1: i16,
    pub nd2: i16,
    pub vd1: i16,
    pub vd2: i16,
    pub tiltangles: [f32; 6],
    pub xorg: f32,
    pub yorg: f32,
    pub zorg: f32,
    pub cmap: [u8; 4],
    pub stamp: [u8; 4],
    pub rms: f32,
    pub nlabl: i32,
    /// Fixed-width, on-disk MRC labels. The disk format has no C terminator.
    pub labels: [[u8; MRC_LABEL_SIZE]; MRC_NLABELS],
    pub symops: Option<Vec<u8>>,
    /// C `FILE *fp`.  See [`ImodFile`]: a clone shares one `Rc<File>`, hence
    /// one kernel file description and one offset, which is what the source's
    /// aliasing of this field with `ImodImageFile.fp` relies on.  `None` is
    /// the source's NULL.
    pub fp: Option<ImodFile>,
    pub pos: i32,
    pub li: Option<LoadInfo>,
    pub header_size: i32,
    pub section_skip: i32,
    pub swapped: i32,
    pub bytes_signed: i32,
    pub y_inverted: i32,
    pub iiu_flags: i32,
    pub packed4bits: i32,
    pub half_floats: i32,
    pub pathname: Option<String>,
    pub filedesc: Option<String>,
    pub user_data: Option<String>,
}

impl Default for MrcHeader {
    /// The all-zero header the source gets for free: every caller of
    /// `mrc_head_new` (`mrcfiles.c:690`) either declares `MrcHeader` on the
    /// stack or `calloc`s it, and `mrc_head_new` then sets the fields it cares
    /// about.  This is that starting point, written out because a Rust struct
    /// carrying an `Option<ImodFile>` and three `Option<String>`s cannot be
    /// produced by `mem::zeroed` (NATIVE.md 4b).
    ///
    /// One documented consequence, already recorded as unmatchable: the stack
    /// case leaves `labels` uninitialised in native and this zeroes it, so
    /// native MRC output carries stack garbage in the label slots past `nlabl`
    /// where ours carries NULs.
    fn default() -> MrcHeader {
        MrcHeader {
            nx: 0,
            ny: 0,
            nz: 0,
            mode: 0,
            nxstart: 0,
            nystart: 0,
            nzstart: 0,
            mx: 0,
            my: 0,
            mz: 0,
            xlen: 0.,
            ylen: 0.,
            zlen: 0.,
            alpha: 0.,
            beta: 0.,
            gamma: 0.,
            mapc: 0,
            mapr: 0,
            maps: 0,
            amin: 0.,
            amax: 0.,
            amean: 0.,
            ispg: 0,
            next: 0,
            creatid: 0,
            blank: [0; 6],
            ext_type: [0; 4],
            nversion: 0,
            blank2: [0; 16],
            nint: 0,
            nreal: 0,
            sub: 0,
            zfac: 0,
            min2: 0.,
            max2: 0.,
            min3: 0.,
            max3: 0.,
            imod_stamp: 0,
            imod_flags: 0,
            idtype: 0,
            lens: 0,
            nd1: 0,
            nd2: 0,
            vd1: 0,
            vd2: 0,
            tiltangles: [0.; 6],
            xorg: 0.,
            yorg: 0.,
            zorg: 0.,
            cmap: [0; 4],
            stamp: [0; 4],
            rms: 0.,
            nlabl: 0,
            labels: [[0; MRC_LABEL_SIZE]; MRC_NLABELS],
            symops: None,
            fp: None,
            pos: 0,
            li: None,
            header_size: 0,
            section_skip: 0,
            swapped: 0,
            bytes_signed: 0,
            y_inverted: 0,
            iiu_flags: 0,
            packed4bits: 0,
            half_floats: 0,
            pathname: None,
            filedesc: None,
            user_data: None,
        }
    }
}

/// Requested MRC loading geometry and scaling.
#[derive(Clone, Default)]
pub struct LoadInfo {
    pub xmin: i32,
    pub xmax: i32,
    pub ymin: i32,
    pub ymax: i32,
    pub zmin: i32,
    pub zmax: i32,
    pub pad_left: i32,
    pub pad_right: i32,
    pub ramp: i32,
    pub scale: i32,
    pub black: i32,
    pub white: i32,
    pub axis: i32,
    pub slope: f32,
    pub offset: f32,
    pub smin: f32,
    pub smax: f32,
    pub contig: i32,
    pub outmin: i32,
    pub outmax: i32,
    pub mirror_fft: i32,
    pub plist: i32,
    pub opx: f32,
    pub opy: f32,
    pub opz: f32,
    pub px: f32,
    pub py: f32,
    pub pz: f32,
    pub pdz: i32,
    pub pcoords: Option<Vec<i32>>,
}

/// Tilt metadata associated with an MRC image.
#[derive(Clone, Default)]
pub struct TiltInfo {
    pub tilt: Option<Vec<f32>>,
    pub axis_x: f32,
    pub axis_y: f32,
    pub axis_z: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub scale_z: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
}

/// Matches C `sizeCanBe4BitK2SuperRes(int, int)` (`mrcfiles.c:275`).
pub fn size_can_be_4_bit_k2_super_res(nx: i32, ny: i32) -> i32 {
    let allowed_x = [7676, 7680, 11520];
    let allowed_y = [7420, 7424, 8184];
    let tol = 2;
    for ind in 0..allowed_x.len() {
        if ((nx - allowed_x[ind] / 2).abs() <= tol && (ny - allowed_y[ind]).abs() <= tol)
            || ((nx - allowed_y[ind] / 2).abs() <= tol && (ny - allowed_x[ind]).abs() <= tol)
        {
            return 1;
        }
    }
    0
}

/// Matches C `fixTitlePadding(char *)` (`mrcfiles.c:292`).
pub fn fix_title_padding(label: &mut [u8; MRC_LABEL_SIZE]) {
    let mut len = 0;
    while len < MRC_LABEL_SIZE && label[len] != 0 {
        len += 1;
    }
    if len < MRC_LABEL_SIZE {
        label[len..MRC_LABEL_SIZE].fill(b' ');
    }
}

/// Matches C `mrc_test_size(MrcHeader *)` (`mrcfiles.c:307`).
pub fn mrc_test_size(hdata: &MrcHeader) -> i32 {
    if hdata.nx <= 0
        || hdata.ny <= 0
        || hdata.nz <= 0
        || (hdata.nx > 65535 && hdata.ny > 65535 && hdata.nz > 65535)
        || hdata.mapc < 0
        || hdata.mapc > 4
        || hdata.mapr < 0
        || hdata.mapr > 4
        || hdata.maps < 0
        || hdata.maps > 4
    {
        return 1;
    }
    0
}

/// Matches C `mrcGetStandardVersion(MrcHeader *)` (`mrcfiles.c:322`).
pub fn mrc_get_standard_version(hdata: Option<&MrcHeader>) -> i32 {
    let Some(hdata) = hdata else {
        return -1;
    };
    let year = Local::now().year();
    if hdata.nversion >= 20140 && hdata.nversion < (year + 2) * 10 {
        return hdata.nversion;
    }
    0
}

/// Matches C static `extTypeIs(MrcHeader *, char, char, char, char)` (`mrcfiles.c:340`).
fn ext_type_is(hdata: &MrcHeader, c1: u8, c2: u8, c3: u8, c4: u8) -> i32 {
    if hdata.ext_type[0] == c1
        && hdata.ext_type[1] == c2
        && hdata.ext_type[2] == c3
        && (hdata.ext_type[3] == c4 || (c3 != 0 && c4 == 0))
    {
        return 1;
    }
    0
}

/// Matches C `mrcGetExtendedType(MrcHeader *, int *)` (`mrcfiles.c:356`).
pub fn mrc_get_extended_type(hdata: &MrcHeader, version: &mut i32) -> i32 {
    *version = 0;
    if ext_type_is(hdata, b' ', b' ', b' ', b' ') != 0 || ext_type_is(hdata, 0, 0, 0, 0) != 0 {
        return MRC_EXT_TYPE_NONE;
    }
    if ext_type_is(hdata, b'S', b'E', b'R', b'I') != 0 {
        return MRC_EXT_TYPE_SERI;
    }
    if ext_type_is(hdata, b'A', b'G', b'A', b'R') != 0 {
        return MRC_EXT_TYPE_AGAR;
    }
    if ext_type_is(hdata, b'F', b'E', b'I', 0) != 0 {
        *version = hdata.ext_type[3] as i32 - 48;
        return MRC_EXT_TYPE_FEI;
    }
    if mrc_get_standard_version(Some(hdata)) > 0
        && hdata.ext_type[0] >= 65
        && hdata.ext_type[0] <= 90
        && hdata.ext_type[1] >= 65
        && hdata.ext_type[1] <= 90
        && ((hdata.ext_type[2] >= 48 && hdata.ext_type[2] <= 57)
            || (hdata.ext_type[2] >= 65 && hdata.ext_type[2] <= 90))
        && ((hdata.ext_type[3] >= 48 && hdata.ext_type[3] <= 57)
            || hdata.ext_type[3] == 32
            || (hdata.ext_type[3] >= 65 && hdata.ext_type[3] <= 90)
            || hdata.ext_type[3] == 0)
    {
        return MRC_EXT_TYPE_UNKNOWN;
    }
    MRC_EXT_TYPE_NONE
}

/// Matches C `mrc_head_read(FILE *, MrcHeader *)` (`mrcfiles.c:52`).
pub fn mrc_head_read(fin: &mut ImodFile, hdata: &mut MrcHeader) -> i32 {
    if let Some(ii_file) = ii_lookup_file_from_fp(fin) {
        // The opened-file registry is the legacy handle boundary.  Its entry
        // is owned by the image layer for the duration of this call.
        return unsafe { ii_fill_mrc_header(ii_file, hdata) };
    }

    b3d_rewind(fin);
    // `mrc_head_read` (`mrcfiles.c:78`) is `fread(hdata, 4, 56, fin)`: the
    // first 224 bytes of `MRCheader` are the on-disk header word for word.
    // Unpacked field by field here so the transfer does not depend on the
    // struct's layout; `from_ne_bytes` is the C read's native byte order, and
    // `mrc_swap_header` below still does the swapping the source does.
    let mut word = [0u8; 224];
    let words_read = b3d_fread(&mut word, 4, 56, fin);
    if words_read != 56 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: mrc_head_read - reading header data; {} of 56 words read, system error: {}\n",
                words_read,
                // A failed read leaves the operating system's error in the
                // current thread.  This is intentionally Rust's OS-error
                // rendering rather than a `strerror(errno)` FFI round trip.
                std::io::Error::last_os_error()
            ),
        );
        return -1;
    }
    let int_at = |o: usize| i32::from_ne_bytes([word[o], word[o + 1], word[o + 2], word[o + 3]]);
    let flt_at = |o: usize| f32::from_ne_bytes([word[o], word[o + 1], word[o + 2], word[o + 3]]);
    let sht_at = |o: usize| i16::from_ne_bytes([word[o], word[o + 1]]);
    hdata.nx = int_at(0);
    hdata.ny = int_at(4);
    hdata.nz = int_at(8);
    hdata.mode = int_at(12);
    hdata.nxstart = int_at(16);
    hdata.nystart = int_at(20);
    hdata.nzstart = int_at(24);
    hdata.mx = int_at(28);
    hdata.my = int_at(32);
    hdata.mz = int_at(36);
    hdata.xlen = flt_at(40);
    hdata.ylen = flt_at(44);
    hdata.zlen = flt_at(48);
    hdata.alpha = flt_at(52);
    hdata.beta = flt_at(56);
    hdata.gamma = flt_at(60);
    hdata.mapc = int_at(64);
    hdata.mapr = int_at(68);
    hdata.maps = int_at(72);
    hdata.amin = flt_at(76);
    hdata.amax = flt_at(80);
    hdata.amean = flt_at(84);
    hdata.ispg = int_at(88);
    hdata.next = int_at(92);
    hdata.creatid = sht_at(96);
    hdata.blank.copy_from_slice(&word[98..104]);
    hdata.ext_type.copy_from_slice(&word[104..108]);
    hdata.nversion = int_at(108);
    hdata.blank2.copy_from_slice(&word[112..128]);
    hdata.nint = sht_at(128);
    hdata.nreal = sht_at(130);
    hdata.sub = sht_at(132);
    hdata.zfac = sht_at(134);
    hdata.min2 = flt_at(136);
    hdata.max2 = flt_at(140);
    hdata.min3 = flt_at(144);
    hdata.max3 = flt_at(148);
    hdata.imod_stamp = int_at(152);
    hdata.imod_flags = int_at(156);
    hdata.idtype = sht_at(160);
    hdata.lens = sht_at(162);
    hdata.nd1 = sht_at(164);
    hdata.nd2 = sht_at(166);
    hdata.vd1 = sht_at(168);
    hdata.vd2 = sht_at(170);
    for i in 0..6 {
        hdata.tiltangles[i] = flt_at(172 + 4 * i);
    }
    hdata.xorg = flt_at(196);
    hdata.yorg = flt_at(200);
    hdata.zorg = flt_at(204);
    hdata.cmap.copy_from_slice(&word[208..212]);
    hdata.stamp.copy_from_slice(&word[212..216]);
    hdata.rms = flt_at(216);
    hdata.nlabl = int_at(220);
    hdata.swapped = 0;
    hdata.iiu_flags = 0;

    if mrc_test_size(hdata) != 0 {
        hdata.swapped = 1;
    }

    if hdata.cmap[0] != b'M' || hdata.cmap[1] != b'A' || hdata.cmap[2] != b'P' {
        // `mrcfiles.c:96-98`: three `memcpy`s that reinterpret the bytes of
        // three later header words as the origin floats of the old layout, in
        // this order.
        hdata.zorg = f32::from_ne_bytes(hdata.cmap);
        hdata.xorg = f32::from_ne_bytes(hdata.stamp);
        hdata.yorg = hdata.rms;
        hdata.rms = -1.0;
        if hdata.swapped != 0 {
            mrc_swap_floats(core::slice::from_mut(&mut hdata.rms), 1);
        }
        mrc_set_cmap_stamp(hdata);
        hdata.iiu_flags |= IIUNIT_OLD_STYLE;
    }

    if hdata.swapped != 0 {
        mrc_swap_header(hdata);
        if mrc_test_size(hdata) != 0 {
            return 1;
        }
    }

    hdata.header_size = 1024;
    hdata.section_skip = 0;
    hdata.y_inverted = 0;
    hdata.header_size += hdata.next;
    hdata.packed4bits = 0;
    hdata.bytes_signed = read_bytes_signed(
        hdata.imod_stamp,
        hdata.imod_flags,
        hdata.mode,
        hdata.amin,
        hdata.amax,
    );
    if hdata.bytes_signed != 0 {
        hdata.amin += 128.0;
        hdata.amax += 128.0;
        hdata.amean += 128.0;
    }

    if hdata.imod_stamp != IMOD_MRC_STAMP {
        hdata.imod_flags = 0;
    }

    if (hdata.imod_stamp == IMOD_MRC_STAMP && (hdata.imod_flags & MRC_FLAGS_INV_ORIGIN) != 0)
        || mrc_get_standard_version(Some(hdata)) > 0
    {
        hdata.xorg *= -1.0;
        hdata.yorg *= -1.0;
        hdata.zorg *= -1.0;
    }

    let mut ignore_inversion = 0;
    if let Ok(ignore_env) = std::env::var("IMOD_IGNORE_MRC_INVERTED") {
        if !ignore_env.is_empty() {
            // `atoi`: leading whitespace and sign, digits, then stop; no error.
            let digits = ignore_env.trim_start();
            let end = digits
                .char_indices()
                .position(|(i, c)| !(c.is_ascii_digit() || (i == 0 && (c == '-' || c == '+'))))
                .unwrap_or(digits.len());
            ignore_inversion = digits[..end].parse::<i32>().unwrap_or(0);
        }
    }

    let mut version = 0;
    if hdata.imod_stamp != IMOD_MRC_STAMP
        && ((mrc_get_extended_type(hdata, &mut version) == MRC_EXT_TYPE_FEI
            && mrc_get_standard_version(Some(hdata)) == 20140
            && (ignore_inversion & 1) == 0)
            || (hdata.mapr == -2 && (ignore_inversion & 2) == 0))
    {
        hdata.y_inverted = 1;
        if hdata.mapr == -2 {
            hdata.mapr = 2;
        }
        hdata.iiu_flags |= IIUNIT_Y_INVERTED;
    }

    for i in 0..MRC_NLABELS {
        if b3d_fread(&mut hdata.labels[i], MRC_LABEL_SIZE, 1, fin) == 0 {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: mrc_head_read - reading label {}.\n", i),
            );
            return -1;
        }
        if (i as i32) < hdata.nlabl {
            fix_title_padding(&mut hdata.labels[i]);
        }
    }

    hdata.half_floats = 0;
    if hdata.mode == MRC_MODE_HALF_FLOAT {
        hdata.mode = MRC_MODE_FLOAT;
        hdata.half_floats = 1;
        hdata.iiu_flags |= IIUNIT_HALF_FLOATS;
    }

    if hdata.mode == MRC_MODE_4BIT {
        hdata.packed4bits = PACKED_4BIT_MODE;
        hdata.mode = MRC_MODE_BYTE;
        hdata.iiu_flags |= IIUNIT_4BIT_MODE;
    } else if ((hdata.imod_flags & MRC_FLAGS_4BIT_BYTES) != 0 && hdata.mode == MRC_MODE_BYTE)
        || (hdata.mode == MRC_MODE_BYTE
            && hdata.nlabl == 1
            && hdata.labels[0][..hdata.labels[0]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(MRC_LABEL_SIZE)]
                .windows(13)
                .any(|w| w == b"4 bits packed")
            && size_can_be_4_bit_k2_super_res(hdata.nx, hdata.ny) != 0)
    {
        hdata.packed4bits = PACKED_HALF_XSIZE;
        hdata.iiu_flags |= IIUNIT_HALF_XSIZE;
        if hdata.mx == hdata.nx {
            hdata.mx *= 2;
            hdata.xlen *= 2.0;
        }
        hdata.nx *= 2;
        if hdata.bytes_signed != 0 {
            unsafe {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: mrc_head_read - cannot read 4-bit data packed in signed bytes.\n"
                    ),
                );
            }
            return 1;
        }
    }

    if hdata.mode > 31 || hdata.mode < 0 {
        unsafe {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: mrc_head_read - bad file mode {}.\n", hdata.mode),
            );
        }
        return 1;
    }
    if hdata.nlabl > MRC_NLABELS as i32 {
        unsafe {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: mrc_head_read - impossible number of labels, {}.\n",
                    hdata.nlabl
                ),
            );
        }
        return 1;
    }

    if (hdata.mapc + 2) / 3 != 1
        || (hdata.mapr + 2) / 3 != 1
        || (hdata.maps + 2) / 3 != 1
        || hdata.mapc == hdata.mapr
        || hdata.mapr == hdata.maps
        || hdata.mapc == hdata.maps
    {
        hdata.mapc = 1;
        hdata.mapr = 2;
        hdata.maps = 3;
        hdata.iiu_flags |= IIUNIT_BAD_MAPCRS;
    }

    if hdata.mx == 0 || hdata.xlen < 1.0e-5 {
        hdata.mx = 1;
        hdata.my = 1;
        hdata.mz = 1;
        hdata.xlen = 1.0;
        hdata.ylen = 1.0;
        hdata.zlen = 1.0;
    }
    if hdata.zlen < 1.0e-5 {
        hdata.zlen = hdata.mz as f32 * hdata.xlen / hdata.mx as f32;
    }

    if hdata.nint == 128
        && hdata.nreal == 32
        && (hdata.next == 131072 || hdata.labels[0].starts_with(b"Fei "))
    {
        hdata.nint = 0;
        hdata.iiu_flags |= IIUNIT_NINT_BUG;
    }

    let mut datasize = hdata.nx.wrapping_mul(hdata.ny).wrapping_mul(hdata.nz);
    match hdata.mode {
        MRC_MODE_BYTE => {}
        MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_HALF_FLOAT => {
            datasize = datasize.wrapping_mul(2)
        }
        MRC_MODE_FLOAT | MRC_MODE_COMPLEX_SHORT => datasize = datasize.wrapping_mul(4),
        MRC_MODE_COMPLEX_FLOAT => datasize = datasize.wrapping_mul(8),
        MRC_MODE_RGB => datasize = datasize.wrapping_mul(3),
        _ => {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: mrc_head_read - bad file mode {}.\n", hdata.mode),
            );
            return 1;
        }
    }
    let _ = datasize;

    b3d_rewind(fin);
    hdata.fp = Some(fin.clone());
    0
}

/// Matches C `mrc_head_write(FILE *, MrcHeader *)` (`mrcfiles.c:386`).
pub fn mrc_head_write(fout: &mut ImodFile, hdata: &mut MrcHeader) -> i32 {
    if let Some(ii_file) = ii_lookup_file_from_fp(fout) {
        if unsafe { (*ii_file).file } != IIFILE_MRC {
            if unsafe { (*ii_file).file } != IIFILE_RAW {
                unsafe { ii_sync_from_mrc_header(&mut *ii_file, hdata) };
            }
            return unsafe { ii_write_header(&mut *ii_file) };
        }
    }

    hdata.imod_stamp = IMOD_MRC_STAMP;
    let mut imod_flags = hdata.imod_flags as u32;
    set_or_clear_flags(
        &mut imod_flags,
        MRC_FLAGS_SBYTES as u32,
        if hdata.bytes_signed != 0 && hdata.packed4bits == 0 {
            1
        } else {
            0
        },
    );
    hdata.imod_flags = imod_flags as i32;
    hdata.creatid = 0;
    hdata.blank[0] = 0;
    hdata.blank[1] = 0;

    let mut hcopy = hdata.clone();
    if hdata.mode == MRC_MODE_BYTE {
        hcopy.amin = hcopy.amin.max(0.0);
        hcopy.amax = hcopy.amax.min(255.0);
        if hdata.bytes_signed != 0 && hdata.packed4bits == 0 {
            hcopy.amin -= 128.0;
            hcopy.amax -= 128.0;
            hcopy.amean -= 128.0;
        }
    } else if (hdata.mode == MRC_MODE_SHORT || hdata.mode == MRC_MODE_USHORT)
        && hcopy.amin < hcopy.amax
    {
        hcopy.amin = hcopy.amin.max(if hdata.mode == MRC_MODE_USHORT {
            0.0
        } else {
            -32768.0
        });
        hcopy.amax = hcopy.amax.min(if hdata.mode == MRC_MODE_USHORT {
            65535.0
        } else {
            32767.0
        });
    }

    if hdata.packed4bits == PACKED_4BIT_MODE {
        hcopy.mode = MRC_MODE_4BIT;
    } else if hdata.packed4bits == PACKED_HALF_XSIZE {
        if hdata.nx % 2 != 0 {
            unsafe {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: mrc_head_write - Cannot write an odd size in X as 4 bits without using mode 101\n"
                    ),
                );
            }
            return 1;
        }
        if hdata.mx == hdata.nx {
            hcopy.mx /= 2;
            hcopy.xlen /= 2.0;
        }
        hcopy.nx /= 2;
    }

    if hdata.half_floats != 0 && hdata.mode == MRC_MODE_FLOAT {
        hcopy.mode = MRC_MODE_HALF_FLOAT;
    }

    let invert = invert_mrc_origin_on_output() != 0
        || (hcopy.xorg == 0.0 && hcopy.yorg == 0.0 && hcopy.xorg == 0.0);
    let mut imod_flags = hcopy.imod_flags as u32;
    set_or_clear_flags(&mut imod_flags, MRC_FLAGS_INV_ORIGIN as u32, invert as i32);
    hcopy.imod_flags = imod_flags as i32;
    if invert {
        hcopy.xorg *= -1.0;
        hcopy.yorg *= -1.0;
        hcopy.zorg *= -1.0;
    }

    hcopy.nversion = 0;
    if !(hdata.mode == MRC_MODE_BYTE && hdata.bytes_signed == 0)
        && hdata.mode != MRC_MODE_RGB
        && hcopy.mode != MRC_MODE_4BIT
        && (hcopy.imod_flags & MRC_FLAGS_INV_ORIGIN) != 0
    {
        hcopy.nversion = 20140;
        if hcopy.mode == MRC_MODE_HALF_FLOAT {
            hcopy.nversion += 1;
        }
    }

    if hdata.swapped != 0 {
        mrc_swap_header(&mut hcopy);
    }
    b3d_rewind(fout);
    // `mrcfiles.c:493` is `b3dFwrite(&hcopy, 56, 4, fout)` — the same 224-byte
    // prefix as the read, at size 56 count 4 rather than 4 by 56, which is why
    // the failure test below is against 4 and not 56.  Packed field by field,
    // `to_ne_bytes` for the C write's native byte order.
    let mut word = [0u8; 224];
    let mut put_int = |o: usize, v: i32| word[o..o + 4].copy_from_slice(&v.to_ne_bytes());
    put_int(0, hcopy.nx);
    put_int(4, hcopy.ny);
    put_int(8, hcopy.nz);
    put_int(12, hcopy.mode);
    put_int(16, hcopy.nxstart);
    put_int(20, hcopy.nystart);
    put_int(24, hcopy.nzstart);
    put_int(28, hcopy.mx);
    put_int(32, hcopy.my);
    put_int(36, hcopy.mz);
    put_int(64, hcopy.mapc);
    put_int(68, hcopy.mapr);
    put_int(72, hcopy.maps);
    put_int(88, hcopy.ispg);
    put_int(92, hcopy.next);
    put_int(108, hcopy.nversion);
    put_int(152, hcopy.imod_stamp);
    put_int(156, hcopy.imod_flags);
    put_int(220, hcopy.nlabl);
    let mut put_flt = |o: usize, v: f32| word[o..o + 4].copy_from_slice(&v.to_ne_bytes());
    put_flt(40, hcopy.xlen);
    put_flt(44, hcopy.ylen);
    put_flt(48, hcopy.zlen);
    put_flt(52, hcopy.alpha);
    put_flt(56, hcopy.beta);
    put_flt(60, hcopy.gamma);
    put_flt(76, hcopy.amin);
    put_flt(80, hcopy.amax);
    put_flt(84, hcopy.amean);
    put_flt(136, hcopy.min2);
    put_flt(140, hcopy.max2);
    put_flt(144, hcopy.min3);
    put_flt(148, hcopy.max3);
    for i in 0..6 {
        put_flt(172 + 4 * i, hcopy.tiltangles[i]);
    }
    put_flt(196, hcopy.xorg);
    put_flt(200, hcopy.yorg);
    put_flt(204, hcopy.zorg);
    put_flt(216, hcopy.rms);
    let mut put_sht = |o: usize, v: i16| word[o..o + 2].copy_from_slice(&v.to_ne_bytes());
    put_sht(96, hcopy.creatid);
    put_sht(128, hcopy.nint);
    put_sht(130, hcopy.nreal);
    put_sht(132, hcopy.sub);
    put_sht(134, hcopy.zfac);
    put_sht(160, hcopy.idtype);
    put_sht(162, hcopy.lens);
    put_sht(164, hcopy.nd1);
    put_sht(166, hcopy.nd2);
    put_sht(168, hcopy.vd1);
    put_sht(170, hcopy.vd2);
    word[98..104].copy_from_slice(&hcopy.blank);
    word[104..108].copy_from_slice(&hcopy.ext_type);
    word[112..128].copy_from_slice(&hcopy.blank2);
    word[208..212].copy_from_slice(&hcopy.cmap);
    word[212..216].copy_from_slice(&hcopy.stamp);
    if b3d_fwrite(&word, 56, 4, fout) != 4 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrc_head_write - writing header to file\n"),
        );
        return 1;
    }
    for i in 0..MRC_NLABELS {
        if b3d_fwrite(&hdata.labels[i], MRC_LABEL_SIZE, 1, fout) != 1 {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: mrc_head_write - writing header to file\n"),
            );
            return 1;
        }
    }
    0
}

/// Matches C `mrc_swap_shorts(b3dInt16 *, int)` (`mrcfiles.c:2076`).
pub fn mrc_swap_shorts(data: &mut [i16], amt: usize) {
    for value in &mut data[..amt] {
        *value = value.swap_bytes();
    }
}

/// Matches C `mrc_swap_longs(b3dInt32 *, int)` (`mrcfiles.c:2095`).
pub fn mrc_swap_longs(data: &mut [i32], amt: usize) {
    for value in &mut data[..amt] {
        *value = value.swap_bytes();
    }
}

/// Matches the `SWAP_IEEE_FLOATS` C `mrc_swap_floats(b3dFloat *, int)` (`mrcfiles.c:2120`).
pub fn mrc_swap_floats(data: &mut [f32], amt: usize) {
    for value in data.iter_mut().take(amt) {
        *value = f32::from_bits(value.to_bits().swap_bytes());
    }
}

/// Matches C `mrc_swap_header(MrcHeader *)` (`mrcfiles.c:920`).
pub fn mrc_swap_header(hdata: &mut MrcHeader) {
    hdata.nx = hdata.nx.swap_bytes();
    hdata.ny = hdata.ny.swap_bytes();
    hdata.nz = hdata.nz.swap_bytes();
    hdata.mode = hdata.mode.swap_bytes();
    hdata.nxstart = hdata.nxstart.swap_bytes();
    hdata.nystart = hdata.nystart.swap_bytes();
    hdata.nzstart = hdata.nzstart.swap_bytes();
    hdata.mx = hdata.mx.swap_bytes();
    hdata.my = hdata.my.swap_bytes();
    hdata.mz = hdata.mz.swap_bytes();
    for value in [
        &mut hdata.xlen,
        &mut hdata.ylen,
        &mut hdata.zlen,
        &mut hdata.alpha,
        &mut hdata.beta,
        &mut hdata.gamma,
    ] {
        *value = f32::from_bits(value.to_bits().swap_bytes());
    }
    hdata.mapc = hdata.mapc.swap_bytes();
    hdata.mapr = hdata.mapr.swap_bytes();
    hdata.maps = hdata.maps.swap_bytes();
    for value in [&mut hdata.amin, &mut hdata.amax, &mut hdata.amean] {
        *value = f32::from_bits(value.to_bits().swap_bytes());
    }
    hdata.ispg = hdata.ispg.swap_bytes();
    hdata.next = hdata.next.swap_bytes();
    hdata.creatid = hdata.creatid.swap_bytes();
    hdata.nversion = hdata.nversion.swap_bytes();
    hdata.nint = hdata.nint.swap_bytes();
    hdata.nreal = hdata.nreal.swap_bytes();
    hdata.sub = hdata.sub.swap_bytes();
    hdata.zfac = hdata.zfac.swap_bytes();
    for value in [
        &mut hdata.min2,
        &mut hdata.max2,
        &mut hdata.min3,
        &mut hdata.max3,
    ] {
        *value = f32::from_bits(value.to_bits().swap_bytes());
    }
    hdata.imod_stamp = hdata.imod_stamp.swap_bytes();
    hdata.imod_flags = hdata.imod_flags.swap_bytes();
    hdata.idtype = hdata.idtype.swap_bytes();
    hdata.lens = hdata.lens.swap_bytes();
    hdata.nd1 = hdata.nd1.swap_bytes();
    hdata.nd2 = hdata.nd2.swap_bytes();
    hdata.vd1 = hdata.vd1.swap_bytes();
    hdata.vd2 = hdata.vd2.swap_bytes();
    for value in &mut hdata.tiltangles {
        *value = f32::from_bits(value.to_bits().swap_bytes());
    }
    hdata.xorg = f32::from_bits(hdata.xorg.to_bits().swap_bytes());
    hdata.yorg = f32::from_bits(hdata.yorg.to_bits().swap_bytes());
    hdata.zorg = f32::from_bits(hdata.zorg.to_bits().swap_bytes());
    hdata.rms = f32::from_bits(hdata.rms.to_bits().swap_bytes());
    hdata.nlabl = hdata.nlabl.swap_bytes();
}

/// Matches C `mrc_set_cmap_stamp(MrcHeader *)` (`mrcfiles.c:948`).
pub fn mrc_set_cmap_stamp(hdata: &mut MrcHeader) {
    let little_end = if cfg!(target_endian = "little") {
        if hdata.swapped != 0 { 0 } else { 1 }
    } else if hdata.swapped != 0 {
        1
    } else {
        0
    };
    hdata.cmap = *b"MAP ";
    if little_end != 0 {
        hdata.stamp[0] = 16 * 4 + 4;
        hdata.stamp[1] = 16 * 4 + 4;
    } else {
        hdata.stamp[0] = 16 * 1 + 1;
        hdata.stamp[1] = 16 * 1 + 1;
    }
    hdata.stamp[2] = 0;
    hdata.stamp[3] = 0;
}

/// Matches C `mrcInitOutputHeader(MrcHeader *)` (`mrcfiles.c:775`).
pub fn mrc_init_output_header(hdata: &mut MrcHeader) {
    hdata.swapped = 0;
    mrc_set_cmap_stamp(hdata);
    hdata.header_size = 1024;
    hdata.section_skip = 0;
    hdata.y_inverted = 0;
    hdata.imod_flags = MRC_FLAGS_BAD_RMS_NEG;
    hdata.rms = -1.0;
    hdata.bytes_signed = write_bytes_signed();
    hdata.packed4bits = if write_4_bit_mode_for_bytes() != 0 {
        PACKED_4BIT_MODE
    } else {
        0
    };
    hdata.half_floats = write_16_bit_mode_for_floats();
    hdata.next = 0;
    hdata.nint = 0;
    hdata.nreal = 0;
    hdata.nversion = 0;
    hdata.ext_type = [b' '; 4];
}

/// Matches C `mrc_head_new(MrcHeader *, int, int, int, int)` (`mrcfiles.c:690`).
pub fn mrc_head_new(hdata: &mut MrcHeader, x: i32, y: i32, z: i32, mode: i32) -> i32 {
    hdata.nx = x;
    hdata.ny = y;
    hdata.nz = z;
    hdata.mode = mode;
    hdata.nxstart = 0;
    hdata.nystart = 0;
    hdata.nzstart = 0;
    hdata.mx = hdata.nx;
    hdata.my = hdata.ny;
    hdata.mz = hdata.nz;
    hdata.xlen = hdata.nx as f32;
    hdata.ylen = hdata.ny as f32;
    hdata.zlen = hdata.nz as f32;
    hdata.alpha = 90.0;
    hdata.beta = 90.0;
    hdata.gamma = 90.0;
    hdata.mapc = 1;
    hdata.mapr = 2;
    hdata.maps = 3;
    // `mrcfiles.c:31-34` locally defines `FLT_MAX` as `1.e37f` under `#ifndef FLT_MAX`;
    // this unit never includes <float.h>, so `mrcfiles.c:714-716` seeds the sentinels
    // with 1.e37f, not the real C `FLT_MAX`.  Verified against the reference binary.
    hdata.amin = 1.0e37_f32;
    hdata.amax = -1.0e37_f32;
    hdata.amean = 0.0;
    hdata.ispg = 0;
    hdata.next = 0;
    hdata.creatid = 0;
    hdata.nversion = 0;
    hdata.nint = 0;
    hdata.nreal = 0;
    hdata.sub = 0;
    hdata.zfac = 0;
    hdata.min2 = 0.0;
    hdata.max2 = 0.0;
    hdata.min3 = 0.0;
    hdata.max3 = 0.0;
    hdata.imod_stamp = IMOD_MRC_STAMP;
    hdata.imod_flags = MRC_FLAGS_BAD_RMS_NEG;
    hdata.iiu_flags = 0;
    hdata.idtype = 0;
    hdata.lens = 0;
    hdata.nd1 = 0;
    hdata.nd2 = 0;
    hdata.vd1 = 0;
    hdata.vd2 = 0;
    hdata.blank = [0; 6];
    hdata.blank2 = [0; 16];
    hdata.tiltangles = [0.0; 6];
    hdata.rms = -1.0;
    hdata.zorg = 0.0;
    hdata.xorg = 0.0;
    hdata.yorg = 0.0;
    hdata.nlabl = 0;
    mrc_init_output_header(hdata);
    hdata.pathname = None;
    hdata.filedesc = None;
    hdata.user_data = None;
    0
}

/// Matches C `mrc_get_scale(MrcHeader *, float *, float *, float *)` (`mrcfiles.c:812`).
pub fn mrc_get_scale(hdata: &MrcHeader) -> (f32, f32, f32) {
    let mut xs = 0.0;
    let mut ys = 0.0;
    let mut zs = 0.0;
    if hdata.xlen != 0.0 {
        xs = hdata.xlen / hdata.mx as f32;
    }
    if hdata.ylen != 0.0 {
        ys = hdata.ylen / hdata.my as f32;
    }
    if hdata.zlen != 0.0 {
        zs = hdata.zlen / hdata.mz as f32;
    }
    (xs, ys, zs)
}

/// Matches C `mrc_set_scale(MrcHeader *, double, double, double)` (`mrcfiles.c:831`).
pub fn mrc_set_scale(hdata: &mut MrcHeader, x: f64, y: f64, z: f64) {
    if x == 0.0 {
        hdata.xlen = hdata.nx as f32;
        hdata.mx = hdata.nx;
    } else {
        hdata.xlen = (hdata.mx as f64 * x) as f32;
    }
    if y == 0.0 {
        hdata.ylen = hdata.ny as f32;
        hdata.my = hdata.ny;
    } else {
        hdata.ylen = (hdata.my as f64 * y) as f32;
    }
    if z == 0.0 {
        hdata.zlen = hdata.nz as f32;
        hdata.mz = hdata.nz;
    } else {
        hdata.zlen = (hdata.mz as f64 * z) as f32;
    }
}

/// Matches C `mrc_coord_cp(MrcHeader *, MrcHeader *)` (`mrcfiles.c:858`).
pub fn mrc_coord_cp(hout: &mut MrcHeader, hin: &MrcHeader) {
    let (xs, ys, zs) = mrc_get_scale(hin);
    mrc_set_scale(hout, xs as f64, ys as f64, zs as f64);
    hout.tiltangles[3] = hin.tiltangles[3];
    hout.tiltangles[4] = hin.tiltangles[4];
    hout.tiltangles[5] = hin.tiltangles[5];
    hout.xorg = hin.xorg;
    hout.yorg = hin.yorg;
    hout.zorg = hin.zorg;
}

/// Matches C `mrc_head_label(MrcHeader *, const char *)` (`mrcfiles.c:492`).
pub fn mrc_head_label(hdata: &mut MrcHeader, label: &[u8]) -> i32 {
    if hdata.nlabl >= MRC_NLABELS as i32 {
        hdata.nlabl -= 1;
    }
    mrc_fill_label_string(label, &mut hdata.labels[hdata.nlabl as usize]);
    hdata.nlabl += 1;
    0
}

/// Matches C `mrcFillLabelString(const char *, void *)` (`mrcfiles.c:511`).
pub fn mrc_fill_label_string(label: &[u8], out_label: &mut [u8; MRC_LABEL_SIZE]) {
    let mut end_of_label = false;
    let date_len = 25;
    let mut i = 0;
    while i < MRC_LABEL_SIZE - date_len {
        if !end_of_label && i < label.len() && label[i] != 0 {
            out_label[i] = label[i];
        } else {
            end_of_label = true;
        }
        if end_of_label {
            out_label[i] = b' ';
        }
        i += 1;
    }
    // This is a fixed-width on-disk field, not a C string.  Chrono obtains
    // local time without exposing a borrowed `tm`; the final byte preserves
    // the NUL that C `strftime` left in its 25-byte source buffer.
    let date = Local::now().format(" %d-%b-%y  %H:%M:%S    ").to_string();
    debug_assert_eq!(date.len() + 1, date_len);
    out_label[i..i + date.len()].copy_from_slice(date.as_bytes());
    out_label[i + date.len()] = 0;
}

/// Matches C `mrcPrintLabelString(MrcHeader *, int)` (`mrcfiles.c:537`).
pub fn mrc_print_label_string(hdata: Option<&MrcHeader>, label_ind: i32) -> i32 {
    let Some(hdata) = hdata else {
        return 1;
    };
    if label_ind < 0 || label_ind >= hdata.nlabl {
        return 1;
    }
    // Labels can contain arbitrary on-disk bytes, so write their trimmed bytes
    // directly instead of fabricating a temporary C string for `%s`.
    use std::io::Write;
    let label = &hdata.labels[label_ind as usize];
    let end = label
        .iter()
        .rposition(|byte| *byte != b' ')
        .map_or(0, |index| index + 1);
    let _ = ImodFile::Stdout.write_all(&label[..end]);
    let _ = ImodFile::Stdout.write_all(b"\n");
    0
}

/// Matches C `mrc_head_label_cp(MrcHeader *, MrcHeader *)` (`mrcfiles.c:558`).
pub fn mrc_head_label_cp(hin: &MrcHeader, hout: &mut MrcHeader) -> i32 {
    for i in 0..hin.nlabl as usize {
        hout.labels[i] = hin.labels[i];
    }
    hout.nlabl = hin.nlabl;
    0
}

/// Matches C `mrcReadExtraHeader(MrcHeader *, unsigned char **)` (`mrcfiles.c:609`).
///
pub fn mrc_read_extra_header(hin: &mut MrcHeader, ext_data: &mut Vec<u8>) -> i32 {
    if hin.fp.is_none() {
        return 1;
    }
    if hin.next == 0 {
        return -1;
    }
    let mut fin = hin.fp.clone().unwrap();
    if b3d_fseek(&mut fin, MRC_HEADER_SIZE as i32, SEEK_SET) != 0 {
        return 2;
    }
    ext_data.resize(hin.next as usize, 0);
    if b3d_fread(ext_data, 1, hin.next as usize, &mut fin) != hin.next as usize {
        ext_data.clear();
        return 4;
    }
    if hin.swapped != 0 {
        if extra_is_nbytes_and_flags(hin.nint as i32, hin.nreal as i32) != 0 {
            for word in ext_data.chunks_exact_mut(2) {
                word.swap(0, 1);
            }
        } else if hin.nint >= 0 && hin.nreal >= 0 && hin.nint + hin.nreal > 0 {
            let nsecs = hin.next / (4 * (hin.nint as i32 + hin.nreal as i32));
            let mut ind = 0_usize;
            for _ in 0..nsecs {
                if hin.nint != 0 {
                    for word in ext_data[ind..ind + 4 * hin.nint as usize].chunks_exact_mut(4) {
                        word.reverse();
                    }
                }
                ind += 4 * hin.nint as usize;
                if hin.nreal != 0 {
                    for word in ext_data[ind..ind + 4 * hin.nreal as usize].chunks_exact_mut(4) {
                        word.reverse();
                    }
                }
                ind += 4 * hin.nreal as usize;
            }
        }
    }
    0
}

/// Matches C `mrcWriteExtraHeader(MrcHeader *, unsigned char *, int)` (`mrcfiles.c:653`).
pub fn mrc_write_extra_header(hout: &mut MrcHeader, ext_data: &[u8]) -> i32 {
    if hout.fp.is_none() || ext_data.is_empty() {
        return 1;
    }
    let mut fout = hout.fp.clone().unwrap();
    if let Some(file) = ii_lookup_file_from_fp(&fout) {
        if unsafe { (*file).file } != IIFILE_MRC {
            return 6;
        }
    }
    if b3d_fseek(&mut fout, MRC_HEADER_SIZE as i32, SEEK_SET) != 0 {
        return 2;
    }
    if b3d_fwrite(ext_data, 1, ext_data.len(), &mut fout) != ext_data.len() {
        return 5;
    }
    hout.next = ext_data.len() as i32;
    hout.header_size = MRC_HEADER_SIZE as i32 + hout.next;
    0
}

/// Matches C `mrcCopyExtraHeader(MrcHeader *, MrcHeader *)` (`mrcfiles.c:578`).
pub fn mrc_copy_extra_header(hin: &mut MrcHeader, hout: &mut MrcHeader) -> i32 {
    if hout.swapped != 0 {
        return 1;
    }
    let mut ext_data = Vec::new();
    let index = mrc_read_extra_header(hin, &mut ext_data);
    if index != 0 {
        return index.max(0);
    }
    let index = mrc_write_extra_header(hout, &ext_data);
    if index != 0 {
        return index;
    }
    hout.nint = hin.nint;
    hout.nreal = hin.nreal;
    mrc_copy_valid_extended_type(hin, hout);
    0
}

/// Matches C `mrcGetDataMemory(IloadInfo *, size_t, int, int)` (`mrcfiles.c:1562`).
pub fn mrc_get_data_memory(
    li: &mut LoadInfo,
    xysize: usize,
    zsize: i32,
    pixsize: i32,
) -> Option<Vec<Vec<u8>>> {
    let plane_bytes = xysize.checked_mul(pixsize as usize)?;
    let total_planes: usize = zsize.try_into().ok()?;
    if li.contig != 0 {
        // Vecs deliberately model sections independently.  The former contiguous
        // allocation was solely an implementation detail; no Rust caller needs
        // pointer arithmetic across image planes.
        li.contig = 0;
    }
    let mut idata = Vec::new();
    idata.try_reserve_exact(total_planes).ok()?;
    for _ in 0..total_planes {
        let mut plane = Vec::new();
        plane.try_reserve_exact(plane_bytes).ok()?;
        plane.resize(plane_bytes, 0);
        idata.push(plane);
    }
    Some(idata)
}

/// Matches C `mrcCopyValidExtendedType(MrcHeader *, MrcHeader *)` (`mrcfiles.c:676`).
pub fn mrc_copy_valid_extended_type(hin: &MrcHeader, hout: &mut MrcHeader) {
    let mut version = 0;
    let mut ind = mrc_get_extended_type(hin, &mut version);
    if ind != MRC_EXT_TYPE_NONE
        && !(ind == MRC_EXT_TYPE_UNKNOWN
            && extra_is_nbytes_and_flags(hin.nint as i32, hin.nreal as i32) != 0)
    {
        ind = 0;
        while ind < 4 {
            hout.ext_type[ind as usize] = hin.ext_type[ind as usize];
            ind += 1;
        }
    }
}

/// Matches C `mrc_byte_mmm(MrcHeader *, unsigned char **)` (`mrcfiles.c:878`).
pub fn mrc_byte_mmm(hdata: Option<&mut MrcHeader>, idata: Option<&[&[u8]]>) -> i32 {
    let Some(hdata) = hdata else {
        return -1;
    };
    let Some(idata) = idata else {
        return -1;
    };
    let mut mean = 0.0_f64;
    let mut min = idata[0][0] as f64;
    let mut max = idata[0][0] as f64;
    for k in 0..hdata.nz as usize {
        for j in 0..hdata.ny as usize {
            for i in 0..hdata.nx as usize {
                let value = idata[k][i + j * hdata.nx as usize] as f64;
                if value > max {
                    max = value;
                }
                if value < min {
                    min = value;
                }
                mean += value;
            }
        }
    }
    mean /= hdata.nx.wrapping_mul(hdata.ny).wrapping_mul(hdata.nz) as f64;
    hdata.amin = min as f32;
    hdata.amean = mean as f32;
    hdata.amax = max as f32;
    0
}

/// Matches C `mrc_getdcsize(int, int *, int *)` (`mrcfiles.c:2065`).
pub fn mrc_getdcsize(mode: i32, dsize: &mut i32, csize: &mut i32) -> i32 {
    if mode == 99 {
        return -1;
    }
    data_size_for_mode(mode, dsize, csize)
}

/// Matches C `mrcGetComplexScale(void)` (`mrcfiles.c:1720`).
pub fn mrc_get_complex_scale() -> f32 {
    5.0
}

/// Matches C `mrcComplexSminSmax(float, float, float *, float *)` (`mrcfiles.c:1729`).
pub fn mrc_complex_smin_smax(mut in_min: f32, in_max: f32) -> (f32, f32) {
    let mut min_sign = 1.0_f32;
    let kscale = mrc_get_complex_scale();
    if in_min < 0.0 {
        min_sign = -1.0;
        in_min = -in_min;
    }
    let out_min = min_sign * (1.0_f64 + (kscale * in_min) as f64).ln() as f32;
    let out_max = (1.0_f64 + (kscale * in_max) as f64).ln() as f32;
    (out_min, out_max)
}

/// Matches C `mrcMirrorSource(int, int, int, int, int *, int *)` (`mrcfiles.c:1747`).
pub fn mrc_mirror_source(nx: i32, ny: i32, image_x: i32, image_y: i32) -> (i32, i32) {
    let mut file_y = image_y;
    let file_x;
    if image_x == 0 {
        file_x = nx / 2;
    } else if image_x >= nx / 2 {
        file_x = image_x - nx / 2;
    } else {
        file_x = nx / 2 - image_x;
        file_y = if image_y != 0 { ny - image_y } else { ny - 1 };
    }
    (file_x, file_y)
}

/// Matches C `mrcContrastScaling(MrcHeader *, float, float, int, int, int, float *, float *)` (`mrcfiles.c:1270`).
pub fn mrc_contrast_scaling(
    hdata: &MrcHeader,
    smin: f32,
    smax: f32,
    black: i32,
    white: i32,
    ramptype: i32,
) -> (f32, f32) {
    let mut max = hdata.amax;
    let mut min = hdata.amin;
    if smin != smax {
        max = smax;
        min = smin;
    }
    if ramptype == MRC_RAMP_LOG {
        min = (min as f64).ln() as f32;
        max = (max as f64).ln() as f32;
    }
    if ramptype == MRC_RAMP_EXP {
        min = (min as f64).exp() as f32;
        max = (max as f64).exp() as f32;
    }
    if hdata.mode == MRC_MODE_COMPLEX_FLOAT || hdata.mode == MRC_MODE_COMPLEX_SHORT {
        (min, max) = mrc_complex_smin_smax(min, max);
    }
    let mut range = white - black + 1;
    if range == 0 {
        range = 1;
    }
    let rscale = (256.0_f64 / range as f64) as f32;
    let mut slope = if max - min != 0.0 {
        (255.0_f64 / (max - min) as f64) as f32
    } else {
        1.0
    };
    slope *= rscale;
    let offset = (-(((black as f32 as f64 / 255.0) * (max - min) as f64) + min as f64)
        * slope as f64) as f32;
    (slope, offset)
}

/// Matches C `mrc_init_li(IloadInfo *, MrcHeader *)` (`mrcfiles.c:1772`).
pub fn mrc_init_li(li: Option<&mut LoadInfo>, hd: Option<&MrcHeader>) -> i32 {
    let Some(li) = li else {
        return -1;
    };
    if let Some(hd) = hd {
        mrc_fix_li(li, hd.nx, hd.ny, hd.nz);
    } else {
        li.xmin = -1;
        li.xmax = -1;
        li.ymin = -1;
        li.ymax = -1;
        li.zmin = -1;
        li.zmax = -1;
        li.pad_left = 0;
        li.pad_right = 0;
        li.ramp = 0;
        li.black = 0;
        li.white = 255;
        li.axis = 3;
        li.mirror_fft = 0;
        li.smin = 0.0;
        li.smax = 0.0;
        li.contig = 0;
        li.outmin = 0;
        li.outmax = 255;
        li.scale = 1;
        li.slope = 1.0;
        li.offset = 0.0;
        li.plist = 0;
        li.ramp = MRC_RAMP_LIN;
    }
    0
}

/// Matches C `mrc_fix_li(IloadInfo *, int, int, int)` (`mrcfiles.c:1817`).
pub fn mrc_fix_li(li: &mut LoadInfo, nx: i32, ny: i32, nz: i32) -> i32 {
    let mut mx = nx;
    let mut my = ny;
    let mut mz = nz;
    if li.plist != 0 {
        mx = li.px as i32;
        my = li.py as i32;
        mz = li.pz as i32;
        if li.xmin != -1 {
            li.xmin -= li.opx as i32;
        }
        if li.xmax != -1 {
            li.xmax -= li.opx as i32;
        }
        if li.ymin != -1 {
            li.ymin -= li.opy as i32;
        }
        if li.ymax != -1 {
            li.ymax -= li.opy as i32;
        }
        if li.zmin != -1 {
            li.zmin -= li.opz as i32;
        }
        if li.zmax != -1 {
            li.zmax -= li.opz as i32;
        }
    }
    if li.xmax < 0 && li.xmin > 0 && li.xmin < mx {
        li.xmax = mx / 2 + li.xmin / 2;
        li.xmin = li.xmax - li.xmin + 1;
    }
    if li.xmax < 0 || li.xmax > mx - 1 {
        li.xmax = mx - 1;
    }
    if li.xmin < 0 || li.xmin > li.xmax {
        li.xmin = 0;
    }
    if li.ymax < 0 && li.ymin > 0 && li.ymin < my {
        li.ymax = my / 2 + li.ymin / 2;
        li.ymin = li.ymax - li.ymin + 1;
    }
    if li.ymax < 0 || li.ymax > my - 1 {
        li.ymax = my - 1;
    }
    if li.ymin < 0 || li.ymin > li.ymax {
        li.ymin = 0;
    }
    if li.zmax >= mz {
        li.zmax = mz - 1;
    }
    if li.zmin >= mz {
        li.zmin = mz - 1;
    }
    if li.zmax < 0 || li.zmax < li.zmin {
        if li.zmin >= 0 {
            li.zmax = li.zmin;
        } else {
            li.zmax = mz - 1;
        }
    }
    if li.zmin < 0 || li.zmin > li.zmax {
        li.zmin = 0;
    }
    if li.white > 255 || li.white < 1 {
        li.white = 255;
    }
    if li.black < 0 || li.black > li.white {
        li.black = 0;
    }
    if li.axis > 3 || li.axis < 1 {
        li.axis = 3;
    }
    0
}

/// Matches C `mrc_liso(MrcHeader *, IloadInfo *)` (`mrcfiles.c:1899`).
pub fn mrc_liso(hdata: &MrcHeader, li: &mut LoadInfo) {
    let mut max = hdata.amax;
    let mut min = hdata.amin;
    if li.ramp == MRC_RAMP_LOG {
        min = (hdata.amin as f64).ln() as f32;
        max = (hdata.amax as f64).ln() as f32;
    }
    if li.ramp == MRC_RAMP_EXP {
        min = (hdata.amin as f64).exp() as f32;
        max = (hdata.amax as f64).exp() as f32;
    }
    let mut range = li.white - li.black + 1;
    if range == 0 {
        range = 1;
    }
    let rscale = (256.0_f64 / range as f64) as f32;
    li.slope = if max - min != 0.0 {
        (255.0_f64 / (max - min) as f64) as f32
    } else {
        1.0
    };
    li.slope *= rscale;
    li.offset = (-(((li.black as f32 as f64 / 255.0) * (max - min) as f64) + min as f64)
        * li.slope as f64) as f32;
}

/// Matches C `get_loadinfo(MrcHeader *, IloadInfo *)` (`mrcfiles.c:1934`).
///
/// This is an interactive Rust API: the header is read-only and the selected
/// loading geometry is owned by `li`, so neither value needs a C pointer.
pub fn get_loadinfo(hdata: &MrcHeader, li: &mut LoadInfo) -> i32 {
    let mut line = [0_u8; 128];
    // `ImodFile` centralizes the actual C standard-stream boundary in
    // `b3dutil`; this module has no reason to retain its own `FILE *`
    // globals.  Flush the same C stdout stream before prompting.
    let _ = ImodFile::Stdout.flush();
    // The prompts stay on the C stream: `fflush(stdout)` above is what
    // makes them appear before `fgetline` blocks, and a Rust write would
    // not be flushed by it.
    use std::io::Write;
    let _ = ImodFile::Stdout.write_all(b" Enter (min x, max x). (return for default) >");
    fgetline(&mut ImodFile::Stdin, &mut line, 127);
    if line[0] != 0 {
        scan_two_ints(&line, &mut li.xmin, &mut li.xmax);
    } else {
        li.xmin = 0;
        li.xmax = hdata.nx - 1;
    }

    let _ = ImodFile::Stdout.write_all(b" Enter (min y, max y). (return for default)  >");
    fgetline(&mut ImodFile::Stdin, &mut line, 127);
    if line[0] != 0 {
        scan_two_ints(&line, &mut li.ymin, &mut li.ymax);
    } else {
        li.ymin = 0;
        li.ymax = hdata.ny - 1;
    }

    let _ = ImodFile::Stdout.write_all(b" Enter sections (low, high)  >");
    fgetline(&mut ImodFile::Stdin, &mut line, 127);
    if line[0] != 0 {
        scan_two_ints(&line, &mut li.zmin, &mut li.zmax);
    } else {
        li.zmin = 0;
        li.zmax = hdata.nz - 1;
    }
    li.scale = 1;
    1
}

/// The `sscanf(line, "%d%*c%d\n", a, b)` that `get_loadinfo`
/// (`mrcfiles.c:1940`, `:1948`, `:1956`) makes three times: an integer, one
/// suppressed character of any kind, and a second integer.  A field that does
/// not convert leaves its variable alone and ends the scan.
fn scan_two_ints(line: &[u8], first: &mut i32, second: &mut i32) {
    let end = line.iter().position(|b| *b == 0).unwrap_or(line.len());
    let mut pos = 0;
    for is_first in [true, false] {
        let target = if is_first { &mut *first } else { &mut *second };
        while pos < end && (line[pos] as char).is_ascii_whitespace() {
            pos += 1;
        }
        let start = pos;
        if pos < end && (line[pos] == b'+' || line[pos] == b'-') {
            pos += 1;
        }
        let digits = pos;
        while pos < end && line[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos == digits {
            return;
        }
        let text = core::str::from_utf8(&line[start..pos]).unwrap_or("0");
        *target = text.parse::<i32>().unwrap_or(i32::MAX);
        // `%*c` consumes exactly one character, whatever it is, and fails at
        // end of input.
        if is_first {
            if pos >= end {
                return;
            }
            pos += 1;
        }
    }
}

/// Matches C `loadtilts(TiltInfo *, MrcHeader *)` (`mrcfiles.c:1979`).
///
/// The tilt metadata is Rust-owned and the image dimensions are only read,
/// so this keeps the whole interaction in references rather than raw C
/// pointers.
pub fn loadtilts(ti: &mut TiltInfo, hdata: &MrcHeader) -> i32 {
    let mut filename = [0u8; 128];
    let mut tiltflag = 0;
    while tiltflag == 0 {
        {
            use std::io::Write;
            let _ = ImodFile::Stdout.write_all(b"Do you wish to load a tilt info file? (y/n) >");
        }
        match ImodFile::Stdin.getc() {
            121 | 89 => tiltflag = 1,
            110 | 78 => tiltflag = 2,
            _ => {}
        }
        ImodFile::Stdin.getc();
    }
    ti.tilt = Some(vec![0.0f32; hdata.nz as usize]);
    let tilt = ti.tilt.as_mut().unwrap();
    if tiltflag == 2 {
        if hdata.nz < 2 {
            tilt[0] = 0.0;
        } else {
            let tiltoff = -60.0;
            let tslope = (120.0_f64 / (hdata.nz as f64 - 1.0)) as f32;
            for i in 0..hdata.nz {
                tilt[i as usize] = tiltoff + i as f32 * tslope;
            }
        }
        ti.axis_z = (hdata.nz / 2) as f32;
        ti.axis_x = (hdata.nx / 2) as f32;
    }
    if tiltflag == 1 {
        getfilename(&mut filename, "Enter tilt info filename. >");
        let name = String::from_utf8_lossy(
            &filename[..filename
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(filename.len())],
        )
        .into_owned();
        let Some(mut fin) = ImodFile::open(&name, "r") else {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: loadtilts - Couldn't load {}.\n", name),
            );
            return 0;
        };
        // The source reads the whole file with `nz + 2` `fscanf("%f")`
        // calls.  `%f` skips leading whitespace, converts, and stops at the
        // first character that cannot extend a float; on a conversion
        // failure it leaves the text in the stream, so every later call
        // fails too and the target keeps its previous value.  Reading the
        // file once and walking whitespace-separated tokens is that, with
        // the same stopping rule.
        let mut text = String::new();
        let _ = fin.read_to_string(&mut text);
        let mut tokens = text.split_ascii_whitespace();
        let mut scan = || tokens.next().and_then(|t| t.parse::<f32>().ok());
        let tilt = ti.tilt.as_mut().unwrap();
        for i in 0..hdata.nz {
            if let Some(value) = scan() {
                tilt[i as usize] = value;
            }
        }
        if let Some(value) = scan() {
            ti.axis_x = value;
        }
        if let Some(value) = scan() {
            ti.axis_z = value;
        }
    }
    1
}

/// Matches C `mrc_mread_slice(FILE *, MrcHeader *, int, char)` (`mrcfiles.c:983`).
pub fn mrc_mread_slice(
    fin: &mut ImodFile,
    hdata: &mut MrcHeader,
    slice: i32,
    axis: u8,
) -> Option<Vec<u8>> {
    let bsize = match axis as u8 {
        b'x' | b'X' => hdata.ny * hdata.nz,
        b'y' | b'Y' => hdata.nx * hdata.nz,
        b'z' | b'Z' => hdata.nx * hdata.ny,
        _ => {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: mrc_mread_slice - axis error.\n"),
            );
            return None;
        }
    };
    let mut dsize = 0;
    let mut csize = 0;
    if mrc_getdcsize(hdata.mode, &mut dsize, &mut csize) != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrc_mread_slice - unknown mode.\n"),
        );
        return None;
    }
    let mut buf = Vec::new();
    let bytes = (dsize as usize)
        .checked_mul(csize as usize)
        .and_then(|size| size.checked_mul(bsize as usize));
    let Some(bytes) = bytes else {
        return None;
    };
    if buf.try_reserve_exact(bytes).is_err() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrc_mread_slice - couldn't get memory.\n"),
        );
        return None;
    }
    buf.resize(bytes, 0);
    if mrc_read_slice(&mut buf, fin, hdata, slice, axis) == 0 {
        return Some(buf);
    }
    None
}

/// Matches C `mrc_read_slice(void *, FILE *, MrcHeader *, int, char)` (`mrcfiles.c:1037`).
pub fn mrc_read_slice(
    buf: &mut [u8],
    fin: &mut ImodFile,
    hdata: &mut MrcHeader,
    slice: i32,
    axis: u8,
) -> i32 {
    let mut li = LoadInfo::default();
    mrc_init_li(Some(&mut li), None);
    mrc_init_li(Some(&mut li), Some(hdata));
    if matches!(axis as u8, b'z' | b'Z' | b'y' | b'Y') {
        if matches!(axis as u8, b'y' | b'Y') {
            li.axis = 2;
        }
        let fp_save = hdata.fp.take();
        hdata.fp = Some(fin.clone());
        let result = crate::imod::libiimod::mrcsec::mrc_read_section(hdata, &mut li, buf, slice);
        hdata.fp = fp_save;
        return result;
    }
    if let Some(ii_file) = ii_lookup_file_from_fp(fin) {
        // The opened-file registry is a legacy handle boundary.
        if unsafe { (*ii_file).file } != IIFILE_MRC && unsafe { (*ii_file).file } != IIFILE_RAW {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: mrc_read_slice - Cannot read X slice from non-MRC-like file\n"
                ),
            );
            return -1;
        }
    }
    b3d_rewind(fin);
    b3d_fseek(fin, hdata.header_size, SEEK_SET);
    if hdata.packed4bits != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrc_read_slice - Cannot read X slice from 4-bit file\n"),
        );
        return -1;
    }
    let mut dsize = 0;
    let mut csize = 0;
    if mrc_getdcsize(hdata.mode, &mut dsize, &mut csize) != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrc_read_slice - unknown mode.\n"),
        );
        return -1;
    }
    let dcsize = dsize * csize;
    if !matches!(axis as u8, b'x' | b'X') {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrc_read_slice - axis error.\n"),
        );
        return -1;
    }
    if slice >= hdata.nx {
        return -1;
    }
    b3d_fseek(fin, slice.wrapping_mul(dcsize), SEEK_CUR);
    let required = (hdata.ny as usize) * (hdata.nz as usize) * (dcsize as usize);
    if buf.len() < required {
        return -1;
    }
    let mut offset = 0;
    for _ in 0..hdata.nz {
        for _ in 0..hdata.ny {
            if b3d_fread(
                &mut buf[offset..offset + dcsize as usize],
                dcsize as usize,
                1,
                fin,
            ) != 1
            {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!("ERROR: mrc_read_slice x - fread error.\n"),
                );
                return -1;
            }
            offset += dcsize as usize;
            b3d_fseek(fin, dcsize * (hdata.nx - 1), SEEK_CUR);
        }
        if hdata.section_skip != 0 {
            b3d_fseek(fin, hdata.section_skip, SEEK_CUR);
        }
    }
    if hdata.swapped != 0 {
        let word_size = match hdata.mode {
            MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_COMPLEX_SHORT => 2,
            MRC_MODE_FLOAT | MRC_MODE_COMPLEX_FLOAT => 4,
            _ => 0,
        };
        if word_size != 0 {
            for word in buf[..required].chunks_exact_mut(word_size) {
                word.reverse();
            }
        }
    }
    if hdata.mode == MRC_MODE_BYTE && hdata.bytes_signed != 0 {
        for item in &mut buf[..required] {
            *item = item.wrapping_add(128);
        }
    }
    let _ = fin.flush();
    0
}

/// Matches C `mrcReadFloatSlice(b3dFloat *, MrcHeader *, int)` (`mrcfiles.c:1148`).
pub fn mrc_read_float_slice(buf: &mut [f32], hdata: &mut MrcHeader, slice: i32) -> i32 {
    let mut li = LoadInfo::default();
    mrc_init_li(Some(&mut li), None);
    li.xmin = 0;
    li.xmax = hdata.nx - 1;
    li.ymin = 0;
    li.ymax = hdata.ny - 1;
    mrc_read_z_float(hdata, &mut li, buf, slice)
}

/// Matches C `mrc_read_byte(FILE *, MrcHeader *, IloadInfo *, void (*)(const char *))`
/// (`mrcfiles.c:1175`).
pub fn mrc_read_byte(
    fin: &mut ImodFile,
    hdata: &mut MrcHeader,
    li: Option<&mut LoadInfo>,
    func: Option<fn(&[u8])>,
) -> Option<Vec<Vec<u8>>> {
    let mut li_local = LoadInfo::default();
    let li = li.unwrap_or(&mut li_local);
    if li.xmax == 0 && li.ymax == 0 && li.zmax == 0 {
        mrc_init_li(Some(li), None);
        mrc_init_li(Some(li), Some(hdata));
    }
    let fp_save = hdata.fp.take();
    hdata.fp = Some(fin.clone());
    let xsize = li.xmax - li.xmin + 1;
    let ysize = li.ymax - li.ymin + 1;
    let zsize = li.zmax - li.zmin + 1;
    let xysize = xsize as usize * ysize as usize;
    let (slope, offset) =
        mrc_contrast_scaling(hdata, li.smin, li.smax, li.black, li.white, li.ramp);
    li.slope = slope;
    li.offset = offset;
    if let Some(callback) = func {
        // `mrcfiles.c:1205-1210` formats into a 128-byte `char statstr[]`;
        // `c_format_bytes` is that `sprintf`, and the buffer truncation
        // cannot bite because neither line can reach 128 bytes.
        let statstr = if zsize > 1 {
            c_format_bytes(
                "Image size %d x %d, %d sections.\n",
                &[
                    CArg::Int(xsize as i64),
                    CArg::Int(ysize as i64),
                    CArg::Int(zsize as i64),
                ],
            )
        } else {
            c_format_bytes(
                "Image size %d x %d.\n",
                &[CArg::Int(xsize as i64), CArg::Int(ysize as i64)],
            )
        };
        callback(&statstr);
    }
    let mut idata = match mrc_get_data_memory(li, xysize, zsize, 1) {
        Some(data) => data,
        None => {
            hdata.fp = fp_save;
            return None;
        }
    };
    if let Some(callback) = func {
        callback(&c_format_bytes("\nReading Image # %3.3d", &[CArg::Int(1)]));
    }
    for k in 0..zsize {
        if let Some(callback) = func {
            callback(&c_format_bytes(
                "\rReading Image # %3.3d",
                &[CArg::Int((k + 1) as i64)],
            ));
        }
        if mrc_read_z_byte(hdata, li, &mut idata[k as usize], k + li.zmin) != 0 {
            hdata.fp = fp_save;
            return None;
        }
    }
    if let Some(callback) = func {
        callback(b"\n");
    }
    hdata.fp = fp_save;
    Some(idata)
}

/// Matches C `mrc_write_idata(FILE *, MrcHeader *, void **)` (`mrcfiles.c:1333`).
pub fn mrc_write_idata(fout: &mut ImodFile, hdata: &mut MrcHeader, data: &[&[u8]]) -> i32 {
    for k in 0..hdata.nz {
        let result = mrc_write_slice(data[k as usize], fout, hdata, k, b'Z');
        if result != 0 {
            return result;
        }
    }
    0
}

/// Matches C `mrc_write_slice(void *, FILE *, MrcHeader *, int, char)` (`mrcfiles.c:1354`).
pub fn mrc_write_slice(
    buf: &[u8],
    fout: &mut ImodFile,
    hdata: &mut MrcHeader,
    slice: i32,
    axis: u8,
) -> i32 {
    if slice < 0 {
        return -1;
    }
    if matches!(axis as u8, b'z' | b'Z') {
        let mut li = LoadInfo::default();
        mrc_init_li(Some(&mut li), None);
        mrc_init_li(Some(&mut li), Some(hdata));
        let fp_save = hdata.fp.take();
        hdata.fp = Some(fout.clone());
        let result = crate::imod::libiimod::mrcsec::mrc_write_z(hdata, &mut li, buf, slice);
        hdata.fp = fp_save;
        return result;
    }
    if let Some(ii_file) = ii_lookup_file_from_fp(fout) {
        // The opened-file registry is a legacy handle boundary.
        if unsafe { (*ii_file).file } != IIFILE_MRC && unsafe { (*ii_file).file } != IIFILE_RAW {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: mrc_write_slice - Cannot write X or Y slice to non-MRC-like file\n"
                ),
            );
            return -1;
        }
    }
    if hdata.packed4bits != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrc_write_slice - Cannot write X or Y slice to 4-bit file\n"),
        );
        return -1;
    }
    b3d_rewind(fout);
    b3d_fseek(fout, hdata.header_size, SEEK_SET);
    let nx = hdata.nx;
    let ny = hdata.ny;
    let mut dsize = 0;
    let mut csize = 0;
    if mrc_getdcsize(hdata.mode, &mut dsize, &mut csize) != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: mrc_write_slice - unknown mode.\n"),
        );
        return -1;
    }
    let dcsize = dsize * csize;
    let (sxsize, sysize) = match axis as u8 {
        b'x' | b'X' if slice < nx => (ny, hdata.nz),
        b'y' | b'Y' if slice < ny => (nx, hdata.nz),
        b'x' | b'X' | b'y' | b'Y' => return -1,
        _ => {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: mrc_write_slice - axis error.\n"),
            );
            return -1;
        }
    };
    let bytes = (sxsize as usize) * (sysize as usize) * (dcsize as usize);
    if buf.len() < bytes {
        return -1;
    }
    let transformed = (hdata.swapped != 0 && dsize > 1)
        || (hdata.mode == MRC_MODE_BYTE && hdata.bytes_signed != 0);
    let mut owned_data = Vec::new();
    let data: &[u8] = if transformed {
        if owned_data.try_reserve_exact(bytes).is_err() {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: mrc_write_slice - failure to allocate memory.\n"),
            );
            return -1;
        }
        owned_data.extend_from_slice(&buf[..bytes]);
        if hdata.mode == MRC_MODE_BYTE && hdata.bytes_signed != 0 {
            for value in &mut owned_data {
                *value = value.wrapping_sub(128);
            }
        } else {
            let word_size = if dsize == 2 { 2 } else { 4 };
            for word in owned_data.chunks_exact_mut(word_size) {
                word.reverse();
            }
        }
        &owned_data
    } else {
        &buf[..bytes]
    };
    let mut offset = 0;
    match axis as u8 {
        b'x' | b'X' => {
            b3d_fseek(fout, slice.wrapping_mul(dcsize), SEEK_CUR);
            for _ in 0..hdata.nz {
                for _ in 0..ny {
                    if b3d_fwrite(
                        &data[offset..offset + dcsize as usize],
                        dcsize as usize,
                        1,
                        fout,
                    ) != 1
                    {
                        b3d_error(
                            Some(&mut ImodFile::Stderr),
                            format_args!("ERROR: mrc_write_slice x - fwrite error.\n"),
                        );
                        return -1;
                    }
                    offset += dcsize as usize;
                    b3d_fseek(fout, dcsize * (nx - 1), SEEK_CUR);
                }
            }
        }
        b'y' | b'Y' => {
            mrc_huge_seek(fout, 0, 0, slice, 0, nx, ny, dcsize, SEEK_CUR);
            for _ in 0..hdata.nz {
                let row_bytes = (dcsize * nx) as usize;
                if b3d_fwrite(
                    &data[offset..offset + row_bytes],
                    dcsize as usize,
                    nx as usize,
                    fout,
                ) != nx as usize
                {
                    b3d_error(
                        Some(&mut ImodFile::Stderr),
                        format_args!("ERROR: mrc_write_slice y - fwrite error.\n"),
                    );
                    return -1;
                }
                offset += row_bytes;
                mrc_huge_seek(fout, 0, 0, ny - 1, 0, nx, ny, dcsize, SEEK_CUR);
            }
        }
        _ => unreachable!(),
    }
    0
}

/// Matches C `mrcWriteFFT(const char *, float *, int, int, int)` (`mrcfiles.c:1502`).
pub fn mrc_write_fft(
    filename: &[u8],
    fft: &mut [f32],
    nx_real: i32,
    ny_real: i32,
    if_scale: i32,
) -> i32 {
    let mut retval = 1;
    let scale_fac = (1.0_f64 / ((nx_real as f64) * (ny_real as f64)).sqrt()) as f32;
    let mut shift_temp = Vec::<f32>::new();
    if shift_temp
        .try_reserve_exact((2 * nx_real + 4) as usize)
        .is_err()
    {
        return 1;
    }
    shift_temp.resize((2 * nx_real + 4) as usize, 0.);
    crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
        fft,
        &mut shift_temp,
        (nx_real + 2) / 2,
        ny_real,
        0,
    );
    if if_scale != 0 {
        for ind in 0..(nx_real + 2) * ny_real {
            fft[ind as usize] *= scale_fac;
        }
    }
    let mut hdr = MrcHeader::default();
    mrc_head_new(
        &mut hdr,
        (nx_real + 2) / 2,
        ny_real,
        1,
        MRC_MODE_COMPLEX_FLOAT,
    );
    imod_backup_file(&String::from_utf8_lossy(filename));
    let fp = ImodFile::open(&*String::from_utf8_lossy(filename), "wb");
    if let Some(mut fp) = fp {
        retval = 0;
        hdr.amax = -1.0e37_f32;
        hdr.amin = 1.0e37_f32;
        let mut asum = 0.0_f64;
        for ind in (0..(nx_real + 2) * ny_real).step_by(2) {
            let ampl = (fft[ind as usize] * fft[ind as usize]
                + fft[(ind + 1) as usize] * fft[(ind + 1) as usize])
                .sqrt();
            asum += ampl as f64;
            hdr.amin = hdr.amin.min(ampl);
            hdr.amax = hdr.amax.max(ampl);
        }
        hdr.amean = (asum / (0.5 * (nx_real + 2) as f64 * ny_real as f64)) as f32;
        let mut bytes = Vec::with_capacity(fft.len() * core::mem::size_of::<f32>());
        for value in fft.iter() {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        if mrc_head_write(&mut fp, &mut hdr) != 0
            || mrc_write_slice(&bytes, &mut fp, &mut hdr, 0, b'Z') != 0
        {
            retval = 1;
        }
        drop(fp);
    }
    if if_scale != 0 {
        for ind in 0..(nx_real + 2) * ny_real {
            fft[ind as usize] /= scale_fac;
        }
    }
    crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
        fft,
        &mut shift_temp,
        (nx_real + 2) / 2,
        ny_real,
        1,
    );
    retval
}

/// Matches C `get_byte_map(float, float, int, int, int)` (`mrcfiles.c:1634`).
///
/// The source returns a pointer to mutable process-global scratch space.  Its
/// Rust callers own the map instead, so no conversion path retains a borrowed
/// C-style pointer beyond the immediate legacy kernel call.
pub fn get_byte_map(
    slope: f32,
    offset: f32,
    outmin: i32,
    outmax: i32,
    bytes_signed: i32,
) -> Vec<u8> {
    let mut map = vec![0; if outmax > 255 { 512 } else { 256 }];
    let base = if bytes_signed != 0 { 128 } else { 0 };
    for i in 0..256 {
        let mut ival = ((i as f32 * slope + offset) as f64 + 0.5).floor() as i32;
        if ival < outmin {
            ival = outmin;
        }
        if ival > outmax {
            ival = outmax;
        }
        let index = (i + base) % 256;
        if outmax > 255 {
            map[2 * index..2 * index + 2].copy_from_slice(&(ival as u16).to_ne_bytes());
        } else {
            map[index] = ival as u8;
        }
    }
    map
}

/// Matches C `get_short_map(float, float, int, int, int, int, int)` (`mrcfiles.c:1677`).
pub fn get_short_map(
    slope: f32,
    offset: f32,
    outmin: i32,
    outmax: i32,
    ramptype: i32,
    swapbytes: i32,
    signedint: i32,
) -> Vec<u8> {
    let to_short = outmax > 255;
    let mut map = vec![0_u8; 65536 * if to_short { 2 } else { 1 }];
    for i in 0..65536_u32 {
        let mut fpixel = i as f32;
        if i > 32767 && signedint != 0 {
            fpixel = i as i32 as f32 - 65536.0;
        }
        if ramptype == MRC_RAMP_EXP {
            fpixel = (fpixel as f64).exp() as f32;
        }
        if ramptype == MRC_RAMP_LOG {
            fpixel = (fpixel as f64).ln() as f32;
        }
        let mut ival = ((fpixel * slope + offset) as f64 + 0.5).floor() as i32;
        if ival < outmin {
            ival = outmin;
        }
        if ival > outmax {
            ival = outmax;
        }
        let mut index = i as u16;
        if swapbytes != 0 {
            index = index.swap_bytes();
        }
        if to_short {
            map[index as usize * 2..index as usize * 2 + 2]
                .copy_from_slice(&(ival as u16).to_ne_bytes());
        } else {
            map[index as usize] = ival as u8;
        }
    }
    map
}

/// Matches C `getfilename(char *, char *)` (`mrcfiles.c:2046`).
pub fn getfilename(name: &mut [u8], prompt: &str) -> i32 {
    let _ = ImodFile::Stdout.write_all(prompt.as_bytes());
    let _ = ImodFile::Stdout.flush();
    let mut i = 0;
    while i < 255 {
        let c = ImodFile::Stdin.getc();
        if c == -1 || c == b'\n' as i32 {
            break;
        }
        name[i as usize] = c as u8;
        i += 1;
    }
    name[i as usize] = 0;
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loadinfo_integer_scan_updates_owned_fields_with_c_separator_rules() {
        let mut first = -1;
        let mut second = -2;
        scan_two_ints(b"  -12,34\0", &mut first, &mut second);
        assert_eq!((first, second), (-12, 34));

        // `%*c` consumes the comma, but a missing second conversion leaves
        // the existing field untouched, just as `sscanf` does.
        scan_two_ints(b"5,not-an-integer\0", &mut first, &mut second);
        assert_eq!((first, second), (5, 34));
    }

    #[test]
    fn header_runtime_flags_match_source_bit_values() {
        assert_eq!(MRC_FLAGS_INV_ORIGIN, 4);
        assert_eq!(MRC_FLAGS_4BIT_BYTES, 32);
        assert_eq!(PACKED_HALF_XSIZE, 2);
        assert_eq!(IIUNIT_SWAPPED | IIUNIT_HALF_FLOATS, 257);
        assert_eq!(IIUNIT_BYTES_SIGNED | IIUNIT_Y_INVERTED, 130);
    }

    #[test]
    fn mrc_head_read_parses_the_vendored_serialem_stack_header() {
        let path = "IMOD/Etomo/unitTestData/headerTest.st";
        let mut fp = crate::imod::libcfshr::b3dutil::ImodFile::open(path, "rb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut fp, &mut header), 0);
        assert_eq!(
            (header.nx, header.ny, header.nz, header.mode),
            (512, 512, 1, 1)
        );
        assert_eq!((header.mx, header.my, header.mz), (512, 512, 1));
        assert_eq!((header.mapc, header.mapr, header.maps), (1, 2, 3));
        assert_eq!(header.header_size, 2048);
        assert!(header.fp.as_ref().unwrap().ptr_eq(&fp));
        assert_eq!(&header.labels[0][..9], b"SerialEM:");
        drop(fp);
    }

    #[test]
    fn mrc_head_write_round_trips_an_mrc_header() {
        crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(0);
        crate::imod::libcfshr::b3dutil::set_4_bit_output_mode(0);
        let mut fp = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
        let mut written = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut written, 16, 12, 3, MRC_MODE_SHORT), 0);
        written.amin = -12.0;
        written.amax = 300.0;
        written.amean = 42.0;
        written.xorg = 4.0;
        written.yorg = 5.0;
        written.zorg = 6.0;
        written.labels[0][..4].copy_from_slice(b"test");
        written.nlabl = 1;
        assert_eq!(mrc_head_write(&mut fp, &mut written), 0);

        let mut read = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut fp, &mut read), 0);
        assert_eq!(
            (read.nx, read.ny, read.nz, read.mode),
            (16, 12, 3, MRC_MODE_SHORT)
        );
        assert_eq!((read.amin, read.amax, read.amean), (-12.0, 300.0, 42.0));
        assert_eq!((read.xorg, read.yorg, read.zorg), (4.0, 5.0, 6.0));
        assert_eq!(&read.labels[0][..4], b"test");
        drop(fp);
        crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(-1);
    }

    #[test]
    fn size_can_be_4_bit_k2_super_res_uses_the_source_tolerance() {
        assert_eq!(size_can_be_4_bit_k2_super_res(3836, 7422), 1);
        assert_eq!(size_can_be_4_bit_k2_super_res(3835, 7420), 0);
    }

    #[test]
    fn fix_title_padding_fills_the_fixed_width_disk_field() {
        let mut label = [0_u8; MRC_LABEL_SIZE];
        label[..3].copy_from_slice(b"abc");
        fix_title_padding(&mut label);
        assert_eq!(&label[..3], b"abc");
        assert!(label[3..MRC_LABEL_SIZE].iter().all(|value| *value == b' '));
    }

    #[test]
    fn fill_label_string_writes_a_fixed_width_rust_date() {
        let mut label = [0_u8; MRC_LABEL_SIZE];
        mrc_fill_label_string(b"created", &mut label);
        assert_eq!(&label[..7], b"created");
        assert_eq!(label[MRC_LABEL_SIZE - 25], b' ');
        assert_eq!(label[MRC_LABEL_SIZE - 1], 0);
    }

    #[test]
    fn mrc_test_size_checks_all_source_constraints() {
        let mut header = MrcHeader::default();
        header.nx = 1;
        header.ny = 1;
        header.nz = 1;
        header.mapc = 1;
        header.mapr = 2;
        header.maps = 3;
        assert_eq!(mrc_test_size(&header), 0);
        header.mapr = 5;
        assert_eq!(mrc_test_size(&header), 1);
    }

    #[test]
    fn mrc_get_extended_type_recognizes_fei_and_blank() {
        let mut header = MrcHeader::default();
        header.ext_type = *b"FEI2";
        let mut version = -1;
        assert_eq!(
            mrc_get_extended_type(&header, &mut version),
            MRC_EXT_TYPE_FEI
        );
        assert_eq!(version, 2);
        header.ext_type = [0; 4];
        assert_eq!(
            mrc_get_extended_type(&header, &mut version),
            MRC_EXT_TYPE_NONE
        );
        assert_eq!(version, 0);
    }

    #[test]
    fn swap_routines_preserve_source_byte_order_behavior() {
        let mut shorts = [0x1234_i16, -1_i16];
        mrc_swap_shorts(&mut shorts, 2);
        assert_eq!(shorts, [0x3412, -1]);
        let mut longs = [0x12345678_i32];
        mrc_swap_longs(&mut longs, 1);
        assert_eq!(longs, [0x78563412]);
        let mut floats = [f32::from_bits(0x12345678)];
        mrc_swap_floats(&mut floats, 1);
        assert_eq!(floats[0].to_bits(), 0x78563412);
    }

    #[test]
    fn mrc_swap_header_and_stamp_follow_source_groups() {
        let mut header = MrcHeader::default();
        header.nx = 0x12345678;
        header.xlen = f32::from_bits(0x12345678);
        header.nlabl = 0x10203040;
        mrc_swap_header(&mut header);
        assert_eq!(header.nx, 0x78563412);
        assert_eq!(header.xlen.to_bits(), 0x78563412);
        assert_eq!(header.nlabl, 0x40302010);
        header.swapped = 0;
        mrc_set_cmap_stamp(&mut header);
        assert_eq!(header.cmap, *b"MAP ");
        assert_eq!(header.stamp[2..], [0, 0]);
    }

    #[test]
    fn mrc_head_new_and_coordinate_copy_follow_source_defaults() {
        let mut input = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut input, 4, 5, 6, MRC_MODE_FLOAT), 0);
        assert_eq!((input.mx, input.my, input.mz), (4, 5, 6));
        assert_eq!((input.amin, input.amax), (1.0e37_f32, -1.0e37_f32));
        assert_eq!(input.imod_stamp, IMOD_MRC_STAMP);
        assert_eq!(input.ext_type, [b' '; 4]);
        mrc_set_scale(&mut input, 2.0, 3.0, 4.0);
        input.tiltangles[3] = 7.0;
        input.xorg = 8.0;
        let mut output = MrcHeader::default();
        mrc_head_new(&mut output, 8, 10, 12, MRC_MODE_FLOAT);
        mrc_coord_cp(&mut output, &input);
        assert_eq!(mrc_get_scale(&output), (2.0, 3.0, 4.0));
        assert_eq!((output.tiltangles[3], output.xorg), (7.0, 8.0));
    }

    #[test]
    fn labels_fill_copy_and_replace_at_the_source_limit() {
        let mut input = MrcHeader::default();
        mrc_head_new(&mut input, 1, 1, 1, MRC_MODE_FLOAT);
        mrc_head_label(&mut input, b"source label\0");
        assert_eq!(&input.labels[0][..12], b"source label");
        assert_eq!(input.nlabl, 1);
        let mut output = MrcHeader::default();
        assert_eq!(mrc_head_label_cp(&input, &mut output), 0);
        assert_eq!(output.labels[0], input.labels[0]);
        input.nlabl = 10;
        mrc_head_label(&mut input, b"replacement\0");
        assert_eq!(input.nlabl, 10);
        assert_eq!(&input.labels[9][..11], b"replacement");
    }

    #[test]
    fn copy_valid_extended_type_rejects_serialem_unknown_stamp() {
        let mut input = MrcHeader::default();
        input.nversion = 20140;
        input.ext_type = *b"AB12";
        input.nint = 8;
        input.nreal = 3;
        let mut output = MrcHeader::default();
        mrc_copy_valid_extended_type(&input, &mut output);
        assert_eq!(output.ext_type, [0; 4]);
        input.ext_type = *b"SERI";
        mrc_copy_valid_extended_type(&input, &mut output);
        assert_eq!(output.ext_type, *b"SERI");
    }

    #[test]
    fn mrc_byte_mmm_uses_all_source_planes_and_sets_header_statistics() {
        let mut header = MrcHeader::default();
        header.nx = 2;
        header.ny = 2;
        header.nz = 2;
        let first = [2_u8, 9, 1, 4];
        let second = [5_u8, 3, 8, 0];
        assert_eq!(mrc_byte_mmm(Some(&mut header), Some(&[&first, &second])), 0);
        assert_eq!((header.amin, header.amax, header.amean), (0.0, 9.0, 4.0));
        assert_eq!(mrc_byte_mmm(None, Some(&[&first])), -1);
    }

    #[test]
    fn mrc_getdcsize_rejects_slice_max_but_accepts_mrc_modes() {
        let mut dsize = 0;
        let mut csize = 0;
        assert_eq!(mrc_getdcsize(MRC_MODE_RGB, &mut dsize, &mut csize), 0);
        assert_eq!((dsize, csize), (1, 3));
        assert_eq!(mrc_getdcsize(99, &mut dsize, &mut csize), -1);
    }

    #[test]
    fn complex_scaling_mirroring_and_contrast_follow_source_equations() {
        assert_eq!(mrc_get_complex_scale(), 5.0);
        let (out_min, out_max) = mrc_complex_smin_smax(-1.0, 3.0);
        assert_eq!(out_min, -(6.0_f64).ln() as f32);
        assert_eq!(out_max, (16.0_f64).ln() as f32);
        assert_eq!(mrc_mirror_source(8, 6, 0, 2), (4, 2));
        assert_eq!(mrc_mirror_source(8, 6, 2, 0), (2, 5));
        let mut header = MrcHeader::default();
        header.amin = 10.0;
        header.amax = 20.0;
        header.mode = MRC_MODE_FLOAT;
        assert_eq!(
            mrc_contrast_scaling(&header, 0.0, 0.0, 0, 255, 1),
            (25.5, -255.0)
        );
    }

    #[test]
    fn load_info_initialization_and_fixing_follow_source_limits() {
        let mut li = LoadInfo::default();
        assert_eq!(mrc_init_li(Some(&mut li), None), 0);
        assert_eq!(
            (li.xmin, li.xmax, li.zmin, li.zmax, li.white, li.axis),
            (-1, -1, -1, -1, 255, 3)
        );
        li.xmin = 3;
        li.xmax = -1;
        li.ymin = -2;
        li.ymax = 99;
        li.zmin = 8;
        li.zmax = -1;
        li.black = 256;
        li.white = 0;
        li.axis = 4;
        assert_eq!(mrc_fix_li(&mut li, 10, 8, 6), 0);
        assert_eq!((li.xmin, li.xmax), (4, 6));
        assert_eq!((li.ymin, li.ymax, li.zmin, li.zmax), (0, 7, 5, 5));
        assert_eq!((li.black, li.white, li.axis), (0, 255, 3));
    }

    #[test]
    fn mrc_liso_matches_linear_contrast_equation() {
        let mut header = MrcHeader::default();
        header.amin = 10.0;
        header.amax = 20.0;
        let mut li = LoadInfo::default();
        li.black = 0;
        li.white = 255;
        mrc_liso(&header, &mut li);
        assert_eq!((li.slope, li.offset), (25.5, -255.0));
    }

    #[test]
    fn byte_and_short_maps_preserve_scaling_clamping_and_index_rules() {
        let map = get_byte_map(1.0, 0.0, 10, 200, 1);
        let smap = get_byte_map(2.0, 0.0, 0, 500, 0);
        assert_eq!(map[128], 10);
        assert_eq!(map[129], 10);
        assert_eq!(map[127], 200);
        assert_eq!(u16::from_ne_bytes(smap[510..512].try_into().unwrap()), 500);
        let short_map = get_short_map(1.0, 0.0, 0, 255, 1, 1, 0);
        assert_eq!(short_map[0], 0);
        assert_eq!(short_map[0x0100], 1);
    }

    #[test]
    fn read_extra_header_preserves_owned_io_and_swapping_contract() {
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let header_bytes = [0_u8; MRC_HEADER_SIZE];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    &header_bytes,
                    1,
                    header_bytes.len(),
                    &mut file
                ),
                header_bytes.len()
            );
            let extra: Vec<u8> = (0..44).collect();
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(&extra, 1, extra.len(), &mut file),
                extra.len()
            );
            {
                use std::io::Write;
                file.flush().unwrap();
            }

            let mut header = MrcHeader::default();
            header.fp = Some(file.clone());
            header.next = extra.len() as i32;
            header.swapped = 1;
            header.nint = 8;
            header.nreal = 3;
            let mut data = Vec::new();
            assert_eq!(mrc_read_extra_header(&mut header, &mut data), 0);
            for index in (0..extra.len()).step_by(2) {
                assert_eq!(data[index], extra[index + 1]);
                assert_eq!(data[index + 1], extra[index]);
            }
            drop(file);
        }
    }

    #[test]
    fn data_memory_allocation_owns_section_vectors() {
        unsafe {
            let mut separate = LoadInfo::default();
            let mut separate_data = mrc_get_data_memory(&mut separate, 5, 3, 2).unwrap();
            for index in 0..3 {
                separate_data[index][9] = index as u8;
            }

            let mut contiguous = LoadInfo::default();
            contiguous.contig = 1;
            let contiguous_data = mrc_get_data_memory(&mut contiguous, 5, 3, 2).unwrap();
            assert_eq!(contiguous.contig, 0);
            assert_eq!(contiguous_data.len(), 3);
            assert!(contiguous_data.iter().all(|plane| plane.len() == 10));
        }
    }

    #[test]
    fn copy_extra_header_preserves_source_io_metadata_and_non_mrc_rejection() {
        unsafe {
            let mut input_file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut output_file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let disk_header = [0_u8; MRC_HEADER_SIZE];
            let extra = [4_u8, 2, 9, 1, 8, 3];
            for file in [&mut input_file, &mut output_file] {
                assert_eq!(
                    crate::imod::libcfshr::b3dutil::b3d_fwrite(
                        &disk_header,
                        1,
                        disk_header.len(),
                        file
                    ),
                    disk_header.len()
                );
                {
                    use std::io::Write;
                    file.flush().unwrap();
                }
            }
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(&extra, 1, extra.len(), &mut input_file),
                extra.len()
            );
            {
                use std::io::Write;
                input_file.flush().unwrap();
            }

            let mut input = MrcHeader::default();
            input.fp = Some(input_file.clone());
            input.next = extra.len() as i32;
            input.nint = 2;
            input.nreal = 1;
            input.ext_type = *b"AGAR";
            let mut output = MrcHeader::default();
            output.fp = Some(output_file.clone());
            assert_eq!(mrc_copy_extra_header(&mut input, &mut output), 0);
            assert_eq!(
                (output.next, output.header_size, output.nint, output.nreal),
                (6, 1030, 2, 1)
            );
            assert_eq!(output.ext_type, *b"AGAR");
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut output_file,
                    MRC_HEADER_SIZE as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut copied = [0_u8; 6];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(&mut copied, 1, 6, &mut output_file),
                copied.len()
            );
            assert_eq!(copied, extra);

            let image_file = crate::imod::libiimod::iimage::ii_new();
            (*image_file).fp = Some(output_file.clone());
            (*image_file).file = crate::imod::libiimod::iimage::IIFILE_HDF;
            assert_eq!(
                crate::imod::libiimod::iimage::add_to_opened_list(&mut *image_file),
                0
            );
            assert_eq!(mrc_write_extra_header(&mut output, &extra), 6);
            crate::imod::libiimod::iimage::remove_from_opened_list(&mut *image_file);
            crate::imod::libiimod::iimage::ii_delete(image_file);
            drop(input_file);
            drop(output_file);
        }
    }

    #[test]
    fn mrc_header_disk_prefix_offsets_are_explicit_not_rust_layout() {
        unsafe {
            crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(0);
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 7, 11, 13, MRC_MODE_FLOAT), 0);
            header.next = 0x1020_3040;
            header.ext_type = *b"RUST";
            header.nint = 23;
            header.nreal = 29;
            header.min2 = 31.5;
            header.imod_flags = 37;
            header.xorg = 41.5;
            header.yorg = 43.5;
            header.zorg = 47.5;
            header.cmap = *b"TEST";
            header.stamp = [2, 3, 5, 7];
            header.rms = 53.5;
            header.nlabl = 1;
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    0,
                    crate::imod::libcfshr::b3dutil::SEEK_SET,
                ),
                0
            );
            let mut prefix = [0_u8; 224];
            let prefix_len = prefix.len();
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(&mut prefix, 1, prefix_len, &mut file),
                prefix_len
            );
            assert_eq!(i32::from_ne_bytes(prefix[0..4].try_into().unwrap()), 7);
            assert_eq!(i32::from_ne_bytes(prefix[4..8].try_into().unwrap()), 11);
            assert_eq!(i32::from_ne_bytes(prefix[8..12].try_into().unwrap()), 13);
            assert_eq!(
                i32::from_ne_bytes(prefix[92..96].try_into().unwrap()),
                0x1020_3040
            );
            assert_eq!(&prefix[104..108], b"RUST");
            assert_eq!(i16::from_ne_bytes(prefix[128..130].try_into().unwrap()), 23);
            assert_eq!(i16::from_ne_bytes(prefix[130..132].try_into().unwrap()), 29);
            assert_eq!(
                f32::from_ne_bytes(prefix[136..140].try_into().unwrap()),
                31.5
            );
            // `mrc_head_write` clears the inverse-origin flag for this
            // non-inverted output, preserving the other bits.
            assert_eq!(i32::from_ne_bytes(prefix[156..160].try_into().unwrap()), 33);
            assert_eq!(
                f32::from_ne_bytes(prefix[196..200].try_into().unwrap()),
                41.5
            );
            assert_eq!(
                f32::from_ne_bytes(prefix[200..204].try_into().unwrap()),
                43.5
            );
            assert_eq!(
                f32::from_ne_bytes(prefix[204..208].try_into().unwrap()),
                47.5
            );
            assert_eq!(&prefix[208..212], b"TEST");
            assert_eq!(&prefix[212..216], &[2, 3, 5, 7]);
            assert_eq!(
                f32::from_ne_bytes(prefix[216..220].try_into().unwrap()),
                53.5
            );
            assert_eq!(i32::from_ne_bytes(prefix[220..224].try_into().unwrap()), 1);
            crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(-1);
        }
    }

    #[test]
    fn mrc_write_slice_x_writes_the_source_strided_plane_layout() {
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 2, 2, 2, MRC_MODE_BYTE), 0);
            header.bytes_signed = 0;
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let empty = [0_u8; 8];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(&empty, 1, empty.len(), &mut file),
                empty.len()
            );
            let plane = [10_u8, 20, 30, 40];
            assert_eq!(mrc_write_slice(&plane, &mut file, &mut header, 0, b'X'), 0);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    MRC_HEADER_SIZE as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut written = [0_u8; 8];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(&mut written, 1, 8, &mut file),
                written.len()
            );
            assert_eq!(written, [10, 0, 20, 0, 30, 0, 40, 0]);
            drop(file);
        }
    }

    #[test]
    fn mrc_z_slice_round_trip_uses_the_real_file_dispatch_path() {
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 3, 2, 1, MRC_MODE_BYTE), 0);
            header.amin = 0.0;
            header.amax = 255.0;
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let written = [3_u8, 1, 4, 1, 5, 9];
            assert_eq!(
                mrc_write_slice(&written, &mut file, &mut header, 0, b'Z',),
                0
            );
            let mut read = [0_u8; 6];
            assert_eq!(
                mrc_read_slice(&mut read, &mut file, &mut header, 0, b'Z',),
                0
            );
            assert_eq!(read, written);
            drop(file);
        }
    }

    /// `mrcfiles.c:184-192,246-249`: the bad-mode diagnostics end in a real newline and
    /// carry the source wording, as printed by the native `header` command.
    #[test]
    fn mrc_head_read_error_messages_use_source_text_and_a_real_newline() {
        use crate::imod::libcfshr::b3dutil::{b3d_get_error, b3d_set_store_error};
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 4, 4, 1, MRC_MODE_BYTE), 0);
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            // Rewrite the on-disk mode word with an out-of-range value.
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    12 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let bad_mode: i32 = 44;
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    &bad_mode.to_ne_bytes(),
                    4,
                    1,
                    &mut file
                ),
                1
            );
            b3d_set_store_error(1);
            let mut read_back = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut file, &mut read_back), 1);
            assert_eq!(
                b3d_get_error(),
                "ERROR: mrc_head_read - bad file mode 44.\n"
            );

            // An impossible label count reports the source count message.
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    12 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let good_mode: i32 = MRC_MODE_BYTE;
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    &good_mode.to_ne_bytes(),
                    4,
                    1,
                    &mut file
                ),
                1
            );
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    220 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let nlabl: i32 = 11;
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(&nlabl.to_ne_bytes(), 4, 1, &mut file),
                1
            );
            assert_eq!(mrc_head_read(&mut file, &mut read_back), 1);
            assert_eq!(
                b3d_get_error(),
                "ERROR: mrc_head_read - impossible number of labels, 11.\n"
            );
            b3d_set_store_error(0);
            drop(file);
        }
    }

    /// A short header read reports the word count and the current OS error.
    #[test]
    fn mrc_head_read_short_read_reports_the_source_word_count() {
        use crate::imod::libcfshr::b3dutil::{b3d_get_error, b3d_set_store_error};
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let partial = [0_u8; 100];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(&partial, 1, partial.len(), &mut file),
                partial.len()
            );
            b3d_set_store_error(1);
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut file, &mut header), -1);
            let message = b3d_get_error().to_string();
            b3d_set_store_error(0);
            assert!(
                message.starts_with(
                    "ERROR: mrc_head_read - reading header data; 25 of 56 words read, system \
                     error: "
                ),
                "{message}"
            );
            assert!(message.ends_with('\n'), "{message}");
            drop(file);
        }
    }

    /// `mrcfiles.c:426-437`: a 4-bit half-X-size header with an odd X size is refused with
    /// the source message.
    #[test]
    fn mrc_head_write_refuses_odd_x_for_half_size_4bit_output() {
        use crate::imod::libcfshr::b3dutil::{b3d_get_error, b3d_set_store_error};
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 7, 4, 1, MRC_MODE_BYTE), 0);
            header.packed4bits = PACKED_HALF_XSIZE;
            b3d_set_store_error(1);
            assert_eq!(mrc_head_write(&mut file, &mut header), 1);
            assert_eq!(
                b3d_get_error(),
                "ERROR: mrc_head_write - Cannot write an odd size in X as 4 bits without using \
                 mode 101\n"
            );
            b3d_set_store_error(0);

            // An even X size halves nx (and mx/xlen when they track nx) in the written copy.
            let mut even = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut even, 8, 4, 1, MRC_MODE_BYTE), 0);
            even.packed4bits = PACKED_HALF_XSIZE;
            assert_eq!(mrc_head_write(&mut file, &mut even), 0);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    0 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut prefix = [0_i32; 12];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(prefix.as_mut_ptr().cast::<u8>(), 4 * (12))
                    },
                    4,
                    12,
                    &mut file,
                ),
                12
            );
            assert_eq!((prefix[0], prefix[1], prefix[2]), (4, 4, 1));
            assert_eq!(prefix[7], 4);
            assert_eq!(even.nx, 8);
            drop(file);
        }
    }

    /// `mrcfiles.c:714-716` with the local `FLT_MAX` of `mrcfiles.c:31-34`: the min/max
    /// sentinels are 1.e37f, which is what reaches the header of a file whose statistics
    /// are never updated.
    #[test]
    fn mrc_head_new_seeds_the_local_flt_max_sentinels_on_disk() {
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 4, 4, 1, MRC_MODE_FLOAT), 0);
            assert_eq!((header.amin, header.amax), (1.0e37_f32, -1.0e37_f32));
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    76 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut stats = [0.0_f32; 3];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(stats.as_mut_ptr().cast::<u8>(), 4 * (3))
                    },
                    4,
                    3,
                    &mut file,
                ),
                3
            );
            assert_eq!((stats[0], stats[1]), (1.0e37_f32, -1.0e37_f32));
            drop(file);
        }
    }

    /// `mrcfiles.c:442-460`: the written copy carries the inverted origin, the
    /// MRC_FLAGS_INV_ORIGIN bit and nversion 20140, while the caller's header is untouched.
    #[test]
    fn mrc_head_write_inverts_origin_and_stamps_nversion_on_disk() {
        unsafe {
            crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(1);
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 4, 4, 1, MRC_MODE_SHORT), 0);
            header.xorg = 12.5;
            header.yorg = -7.25;
            header.zorg = 3.0;
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            assert_eq!((header.xorg, header.yorg, header.zorg), (12.5, -7.25, 3.0));

            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    108 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut nversion = 0_i32;
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(
                            core::ptr::addr_of_mut!(nversion).cast::<u8>(),
                            4 * 1,
                        )
                    },
                    4,
                    1,
                    &mut file,
                ),
                1
            );
            assert_eq!(nversion, 20140);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    156 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut flags = 0_i32;
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(
                            core::ptr::addr_of_mut!(flags).cast::<u8>(),
                            4 * 1,
                        )
                    },
                    4,
                    1,
                    &mut file,
                ),
                1
            );
            assert_eq!(flags & MRC_FLAGS_INV_ORIGIN, MRC_FLAGS_INV_ORIGIN);
            assert_eq!(flags & MRC_FLAGS_BAD_RMS_NEG, MRC_FLAGS_BAD_RMS_NEG);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    196 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut origin = [0.0_f32; 3];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(origin.as_mut_ptr().cast::<u8>(), 4 * (3))
                    },
                    4,
                    3,
                    &mut file,
                ),
                3
            );
            assert_eq!(origin, [-12.5, 7.25, -3.0]);

            // Reading the file back undoes the inversion (`mrcfiles.c:120-126`).
            let mut read_back = MrcHeader::default();
            assert_eq!(mrc_head_read(&mut file, &mut read_back), 0);
            assert_eq!(
                (read_back.xorg, read_back.yorg, read_back.zorg),
                (12.5, -7.25, 3.0)
            );
            crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(-1);
            drop(file);
        }
    }

    /// `mrcfiles.c:407-423`: byte output clamps to 0/255 and short/ushort output clamps to
    /// the mode range, in the written copy only.
    #[test]
    fn mrc_head_write_clamps_mode_extremes_in_the_written_copy() {
        unsafe {
            crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(0);
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 4, 4, 1, MRC_MODE_BYTE), 0);
            header.bytes_signed = 0;
            header.amin = -40.0;
            header.amax = 900.0;
            header.amean = 12.0;
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            assert_eq!((header.amin, header.amax), (-40.0, 900.0));
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    76 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut stats = [0.0_f32; 3];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(stats.as_mut_ptr().cast::<u8>(), 4 * (3))
                    },
                    4,
                    3,
                    &mut file,
                ),
                3
            );
            assert_eq!(stats, [0.0, 255.0, 12.0]);

            let mut ushort = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut ushort, 4, 4, 1, MRC_MODE_USHORT), 0);
            ushort.amin = -100.0;
            ushort.amax = 90000.0;
            ushort.amean = 5.0;
            assert_eq!(mrc_head_write(&mut file, &mut ushort), 0);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    76 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(stats.as_mut_ptr().cast::<u8>(), 4 * (3))
                    },
                    4,
                    3,
                    &mut file,
                ),
                3
            );
            assert_eq!(stats, [0.0, 65535.0, 5.0]);

            let mut short_header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut short_header, 4, 4, 1, MRC_MODE_SHORT), 0);
            short_header.amin = -40000.0;
            short_header.amax = 40000.0;
            short_header.amean = 5.0;
            assert_eq!(mrc_head_write(&mut file, &mut short_header), 0);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    76 as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(
                    unsafe {
                        core::slice::from_raw_parts_mut(stats.as_mut_ptr().cast::<u8>(), 4 * (3))
                    },
                    4,
                    3,
                    &mut file,
                ),
                3
            );
            assert_eq!(stats, [-32768.0, 32767.0, 5.0]);
            crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(-1);
            drop(file);
        }
    }

    /// `mrcfiles.c:1060-1095`: the X-slice path emits the source diagnostics in source
    /// order - the non-MRC and 4-bit checks precede the mode check, which precedes the axis
    /// check.
    #[test]
    fn mrc_read_slice_reports_x_path_errors_in_source_order() {
        use crate::imod::libcfshr::b3dutil::{b3d_get_error, b3d_set_store_error};
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE), 0);
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let mut buffer = [0_u8; 4];
            b3d_set_store_error(1);

            // 4-bit files are rejected before the mode and axis checks, even for a bad axis.
            header.packed4bits = PACKED_4BIT_MODE;
            assert_eq!(
                mrc_read_slice(&mut buffer, &mut file, &mut header, 0, b'q'),
                -1
            );
            assert_eq!(
                b3d_get_error(),
                "ERROR: mrc_read_slice - Cannot read X slice from 4-bit file\n"
            );
            header.packed4bits = 0;

            // An unsupported mode is reported before the axis check.
            header.mode = 99;
            assert_eq!(
                mrc_read_slice(&mut buffer, &mut file, &mut header, 0, b'q'),
                -1
            );
            assert_eq!(b3d_get_error(), "ERROR: mrc_read_slice - unknown mode.\n");
            header.mode = MRC_MODE_BYTE;

            // Only then does a bad axis produce the axis error.
            assert_eq!(
                mrc_read_slice(&mut buffer, &mut file, &mut header, 0, b'q'),
                -1
            );
            assert_eq!(b3d_get_error(), "ERROR: mrc_read_slice - axis error.\n");

            // An X slice beyond nx returns -1 with no new message.
            assert_eq!(
                mrc_read_slice(&mut buffer, &mut file, &mut header, 5, b'X'),
                -1
            );
            assert_eq!(b3d_get_error(), "ERROR: mrc_read_slice - axis error.\n");
            b3d_set_store_error(0);
            drop(file);
        }
    }

    /// `mrcfiles.c:537-555`: the label print trims trailing blanks and emits through the C
    /// stdout stream, and rejects an out-of-range index.
    #[test]
    fn mrc_print_label_string_trims_trailing_blanks_on_c_stdout() {
        unsafe {
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 4, 4, 1, MRC_MODE_BYTE), 0);
            header.labels[0] = [b' '; MRC_LABEL_SIZE];
            header.labels[0][..5].copy_from_slice(b"label");
            header.nlabl = 1;
            assert_eq!(mrc_print_label_string(None, 0), 1);
            assert_eq!(mrc_print_label_string(Some(&header), -1), 1);
            assert_eq!(mrc_print_label_string(Some(&header), 1), 1);

            let path_buf = std::env::temp_dir()
                .join(format!("imod_rs_mrcfiles_label_{}.txt", std::process::id()));
            let path = std::ffi::CString::new(path_buf.to_str().unwrap()).unwrap();
            let saved = libc::dup(1);
            assert!(saved >= 0);
            let capture = libc::open(
                path.as_ptr(),
                libc::O_CREAT | libc::O_TRUNC | libc::O_RDWR,
                0o600,
            );
            assert!(capture >= 0);
            let _ = ImodFile::Stdout.flush();
            assert!(libc::dup2(capture, 1) >= 0);
            assert_eq!(mrc_print_label_string(Some(&header), 0), 0);
            let _ = ImodFile::Stdout.flush();
            assert!(libc::dup2(saved, 1) >= 0);
            libc::close(saved);
            assert_eq!(libc::lseek(capture, 0, libc::SEEK_SET), 0);
            let mut got = [0_u8; 32];
            let read = libc::read(capture, got.as_mut_ptr().cast(), got.len());
            libc::close(capture);
            libc::unlink(path.as_ptr());
            assert_eq!(&got[..read as usize], b"label\n");
        }
    }

    /// `mrcfiles.c:653-670`: writing an extended header seeks to 1024, writes `next` bytes
    /// and updates both `next` and `headerSize` in the output header.
    #[test]
    fn mrc_write_extra_header_updates_next_and_header_size_on_a_real_file() {
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            assert_eq!(mrc_head_new(&mut header, 4, 4, 1, MRC_MODE_BYTE), 0);
            header.fp = Some(file.clone());
            assert_eq!(mrc_head_write(&mut file, &mut header), 0);
            let extra = [1_u8, 2, 3, 4, 5, 6, 7, 8];
            assert_eq!(mrc_write_extra_header(&mut header, &[]), 1);
            assert_eq!(mrc_write_extra_header(&mut header, &extra), 0);
            assert_eq!(header.next, 8);
            assert_eq!(header.header_size, MRC_HEADER_SIZE as i32 + 8);
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fseek(
                    &mut file,
                    MRC_HEADER_SIZE as i32,
                    crate::imod::libcfshr::b3dutil::SEEK_SET
                ),
                0
            );
            let mut got = [0_u8; 8];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fread(&mut got, 1, 8, &mut file),
                8
            );
            assert_eq!(got, extra);
            drop(file);
        }
    }
}
