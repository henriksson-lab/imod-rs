//! Type scaffold for `IMOD/include/mrcfiles.h` and implementation counterpart
//! `IMOD/libiimod/mrcfiles.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::{
    b3d_error, b3d_shift_bytes, data_size_for_mode, extra_is_nbytes_and_flags, imod_backup_file,
    invert_mrc_origin_on_output, mrc_huge_seek, read_bytes_signed, set_or_clear_flags,
    write_4_bit_mode_for_bytes, write_16_bit_mode_for_floats, write_bytes_signed,
};
use crate::imod::libiimod::iimage::{
    IIFILE_MRC, IIFILE_RAW, ii_fill_mrc_header, ii_lookup_file_from_fp, ii_sync_from_mrc_header,
    ii_write_header,
};
use crate::imod::libiimod::mrcsec::{mrc_read_z_byte, mrc_read_z_float};
use core::ffi::{c_char, c_void};

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
    static mut stdout: *mut libc::FILE;
    static mut stdin: *mut libc::FILE;
}

pub const MRC_MODE_BYTE: i32 = 0;
pub const MRC_MODE_SHORT: i32 = 1;
pub const MRC_MODE_FLOAT: i32 = 2;
pub const MRC_MODE_COMPLEX_SHORT: i32 = 3;
pub const MRC_MODE_COMPLEX_FLOAT: i32 = 4;
pub const MRC_MODE_USHORT: i32 = 6;
pub const MRC_MODE_HALF_FLOAT: i32 = 12;
pub const MRC_MODE_RGB: i32 = 16;
pub const MRC_MODE_4BIT: i32 = 101;
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

/// C `ComplexFloat` (`mrcfiles.h`).
#[repr(C)]
pub struct ComplexFloat {
    pub a: f32,
    pub b: f32,
}
/// C `ComplexShort` (`mrcfiles.h`).
#[repr(C)]
pub struct ComplexShort {
    pub a: i16,
    pub b: i16,
}

/// C `MRCheader` / `MrcHeader` (`mrcfiles.h`), in declaration order.
#[repr(C)]
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
    pub labels: [[u8; MRC_LABEL_SIZE + 1]; MRC_NLABELS],
    pub symops: *mut u8,
    pub fp: *mut c_void,
    pub pos: i32,
    pub li: *mut LoadInfo,
    pub header_size: i32,
    pub section_skip: i32,
    pub swapped: i32,
    pub bytes_signed: i32,
    pub y_inverted: i32,
    pub iiu_flags: i32,
    pub packed4bits: i32,
    pub half_floats: i32,
    pub pathname: *mut c_char,
    pub filedesc: *mut c_char,
    pub user_data: *mut c_char,
}

/// C `LoadInfo` / `IloadInfo` (`mrcfiles.h`).
#[repr(C)]
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
    pub pcoords: *mut i32,
}

/// C `TiltInfo` (`mrcfiles.h`).
#[repr(C)]
pub struct TiltInfo {
    pub tilt: *mut f32,
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

/// C union `FloatBits` (`mrcfiles.h`).
#[repr(C)]
pub union FloatBits {
    pub f: f32,
    pub fbits: u32,
}

static mut BYTE_MAP: [u8; 256] = [0; 256];
static mut BYTE_SMAP: [u16; 256] = [0; 256];

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
pub fn fix_title_padding(label: &mut [u8; MRC_LABEL_SIZE + 1]) {
    label[MRC_LABEL_SIZE] = 0;
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
    let now = unsafe { libc::time(core::ptr::null_mut()) };
    let tmp = unsafe { libc::localtime(&now) };
    let year = unsafe { (*tmp).tm_year + 1900 };
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
pub unsafe fn mrc_head_read(fin: *mut libc::FILE, hdata: *mut MrcHeader) -> i32 {
    if fin.is_null() {
        return -1;
    }

    let ii_file = unsafe { ii_lookup_file_from_fp(fin) };
    if !ii_file.is_null() {
        return unsafe { ii_fill_mrc_header(ii_file, hdata) };
    }

    unsafe { libc::rewind(fin) };
    let words_read = unsafe { libc::fread(hdata.cast(), 4, 56, fin) };
    if words_read != 56 {
        unsafe {
            b3d_error(
                stderr,
                format_args!(
                    "ERROR: mrc_head_read - reading header data; {} of 56 words read\\n",
                    words_read
                ),
            );
        }
        return -1;
    }
    let hdata = unsafe { &mut *hdata };
    hdata.swapped = 0;
    hdata.iiu_flags = 0;

    if mrc_test_size(hdata) != 0 {
        hdata.swapped = 1;
    }

    if hdata.cmap[0] != b'M' || hdata.cmap[1] != b'A' || hdata.cmap[2] != b'P' {
        unsafe {
            core::ptr::copy_nonoverlapping(
                hdata.cmap.as_ptr(),
                core::ptr::addr_of_mut!(hdata.zorg).cast::<u8>(),
                4,
            );
            core::ptr::copy_nonoverlapping(
                hdata.stamp.as_ptr(),
                core::ptr::addr_of_mut!(hdata.xorg).cast::<u8>(),
                4,
            );
            core::ptr::copy_nonoverlapping(
                core::ptr::addr_of!(hdata.rms).cast::<u8>(),
                core::ptr::addr_of_mut!(hdata.yorg).cast::<u8>(),
                4,
            );
        }
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
    let ignore_env = unsafe { libc::getenv(c"IMOD_IGNORE_MRC_INVERTED".as_ptr()) };
    if !ignore_env.is_null() && unsafe { libc::strlen(ignore_env) } > 0 {
        ignore_inversion = unsafe { libc::atoi(ignore_env) };
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
        if unsafe { libc::fread(hdata.labels[i].as_mut_ptr().cast(), MRC_LABEL_SIZE, 1, fin) } == 0
        {
            unsafe {
                b3d_error(
                    stderr,
                    format_args!("ERROR: mrc_head_read - reading label {}.\\n", i),
                );
            }
            hdata.labels[i][MRC_LABEL_SIZE] = 0;
            return -1;
        }
        hdata.labels[i][MRC_LABEL_SIZE] = 0;
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
            && unsafe {
                libc::strstr(hdata.labels[0].as_ptr().cast(), c"4 bits packed".as_ptr()).is_null()
                    == false
            }
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
                    stderr,
                    format_args!(
                        "ERROR: mrc_head_read - cannot read 4-bit data packed in signed bytes.\\n"
                    ),
                );
            }
            return 1;
        }
    }

    if hdata.mode > 31 || hdata.mode < 0 {
        unsafe {
            b3d_error(
                stderr,
                format_args!("ERROR: mrc_head_read - bad file mode {}.\\n", hdata.mode),
            );
        }
        return 1;
    }
    if hdata.nlabl > MRC_NLABELS as i32 {
        unsafe {
            b3d_error(
                stderr,
                format_args!(
                    "ERROR: mrc_head_read - impossible number of labels, {}.\\n",
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
        && (hdata.next == 131072
            || unsafe {
                libc::strstr(hdata.labels[0].as_ptr().cast(), c"Fei ".as_ptr())
                    == hdata.labels[0].as_mut_ptr().cast()
            })
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
            unsafe {
                b3d_error(
                    stderr,
                    format_args!("ERROR: mrc_head_read - bad file mode {}.\\n", hdata.mode),
                );
            }
            return 1;
        }
    }
    let _ = datasize;

    unsafe { libc::rewind(fin) };
    hdata.fp = fin.cast();
    0
}

/// Matches C `mrc_head_write(FILE *, MrcHeader *)` (`mrcfiles.c:386`).
pub unsafe fn mrc_head_write(fout: *mut libc::FILE, hdata: *mut MrcHeader) -> i32 {
    if fout.is_null() {
        return 1;
    }
    let ii_file = unsafe { ii_lookup_file_from_fp(fout) };
    if !ii_file.is_null() && unsafe { (*ii_file).file } != IIFILE_MRC {
        if unsafe { (*ii_file).file } != IIFILE_RAW {
            unsafe { ii_sync_from_mrc_header(ii_file, hdata) };
        }
        return unsafe { ii_write_header(ii_file) };
    }

    let hdata = unsafe { &mut *hdata };
    hdata.imod_stamp = IMOD_MRC_STAMP;
    set_or_clear_flags(
        unsafe { &mut *core::ptr::addr_of_mut!(hdata.imod_flags).cast::<u32>() },
        MRC_FLAGS_SBYTES as u32,
        if hdata.bytes_signed != 0 && hdata.packed4bits == 0 {
            1
        } else {
            0
        },
    );
    hdata.creatid = 0;
    hdata.blank[0] = 0;
    hdata.blank[1] = 0;

    let mut hcopy = unsafe { core::ptr::read(hdata) };
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
                    stderr,
                    format_args!(
                        "ERROR: mrc_head_write - Cannot write an odd size in X as 4 bits without using mode 101\\n"
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
    set_or_clear_flags(
        unsafe { &mut *core::ptr::addr_of_mut!(hcopy.imod_flags).cast::<u32>() },
        MRC_FLAGS_INV_ORIGIN as u32,
        invert as i32,
    );
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
    unsafe { libc::rewind(fout) };
    if unsafe { libc::fwrite(core::ptr::addr_of!(hcopy).cast(), 56, 4, fout) } != 4 {
        unsafe {
            b3d_error(
                stderr,
                format_args!("ERROR: mrc_head_write - writing header to file\\n"),
            );
        }
        return 1;
    }
    for i in 0..MRC_NLABELS {
        if unsafe { libc::fwrite(hdata.labels[i].as_ptr().cast(), MRC_LABEL_SIZE, 1, fout) } != 1 {
            unsafe {
                b3d_error(
                    stderr,
                    format_args!("ERROR: mrc_head_write - writing header to file\\n"),
                );
            }
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
    unsafe {
        let ldata = data.as_mut_ptr().cast::<u8>().add(amt * 4);
        let mut ptr = data.as_mut_ptr().cast::<u8>();
        while ptr < ldata {
            let mut tmp = *ptr;
            *ptr = *ptr.add(3);
            *ptr.add(3) = tmp;
            ptr = ptr.add(1);
            tmp = *ptr;
            *ptr = *ptr.add(1);
            *ptr.add(1) = tmp;
            ptr = ptr.add(3);
        }
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
    hdata.amin = f32::MAX;
    hdata.amax = -f32::MAX;
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
    hdata.pathname = core::ptr::null_mut();
    hdata.filedesc = core::ptr::null_mut();
    hdata.user_data = core::ptr::null_mut();
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
pub fn mrc_fill_label_string(label: &[u8], out_label: &mut [u8; MRC_LABEL_SIZE + 1]) {
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
    let now = unsafe { libc::time(core::ptr::null_mut()) };
    let tmp = unsafe { libc::localtime(&now) };
    let mut date = [0_i8; 25];
    unsafe {
        libc::strftime(
            date.as_mut_ptr(),
            date.len(),
            c" %d-%b-%y  %H:%M:%S    ".as_ptr(),
            tmp,
        );
    }
    for ind in 0..date_len {
        out_label[i + ind] = date[ind] as u8;
    }
}

/// Matches C `mrcPrintLabelString(MrcHeader *, int)` (`mrcfiles.c:537`).
pub fn mrc_print_label_string(hdata: Option<&MrcHeader>, label_ind: i32) -> i32 {
    let Some(hdata) = hdata else {
        return 1;
    };
    if label_ind < 0 || label_ind >= hdata.nlabl {
        return 1;
    }
    let source = &hdata.labels[label_ind as usize];
    let mut end = MRC_LABEL_SIZE;
    while end > 0 && source[end - 1] == b' ' {
        end -= 1;
    }
    println!("{}", String::from_utf8_lossy(&source[..end]));
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
/// The pointer and allocation ownership follow the C interface exactly: when
/// `*ext_data` is null this routine allocates with `libc::malloc`; the caller
/// releases the resulting buffer with `libc::free`.
pub unsafe fn mrc_read_extra_header(hin: *mut MrcHeader, ext_data: *mut *mut u8) -> i32 {
    if hin.is_null() || unsafe { (*hin).fp.is_null() } || ext_data.is_null() {
        return 1;
    }
    if unsafe { (*hin).next } == 0 {
        return -1;
    }
    if unsafe {
        libc::fseek(
            (*hin).fp.cast::<libc::FILE>(),
            MRC_HEADER_SIZE as i64,
            libc::SEEK_SET,
        )
    } != 0
    {
        return 2;
    }
    if unsafe { (*ext_data).is_null() } {
        unsafe { *ext_data = libc::malloc((*hin).next as usize).cast::<u8>() };
    }
    if unsafe { (*ext_data).is_null() } {
        return 3;
    }
    if unsafe {
        libc::fread(
            (*ext_data).cast::<c_void>(),
            1,
            (*hin).next as usize,
            (*hin).fp.cast::<libc::FILE>(),
        )
    } != unsafe { (*hin).next as usize }
    {
        unsafe {
            libc::free(*ext_data.cast());
            *ext_data = core::ptr::null_mut();
        }
        return 4;
    }
    if unsafe { (*hin).swapped } != 0 {
        if extra_is_nbytes_and_flags(unsafe { (*hin).nint as i32 }, unsafe {
            (*hin).nreal as i32
        }) != 0
        {
            let data = unsafe {
                core::slice::from_raw_parts_mut(
                    *ext_data as *mut i16,
                    (*hin).next as usize / core::mem::size_of::<i16>(),
                )
            };
            mrc_swap_shorts(data, unsafe { (*hin).next as usize / 2 });
        } else if unsafe { (*hin).nint >= 0 && (*hin).nreal >= 0 && (*hin).nint + (*hin).nreal > 0 }
        {
            let nsecs =
                unsafe { (*hin).next as i32 / (4 * ((*hin).nint as i32 + (*hin).nreal as i32)) };
            let mut ind = 0_usize;
            for _ in 0..nsecs {
                if unsafe { (*hin).nint } != 0 {
                    let data = unsafe {
                        core::slice::from_raw_parts_mut(
                            (*ext_data).add(ind).cast::<i32>(),
                            (*hin).nint as usize,
                        )
                    };
                    mrc_swap_longs(data, unsafe { (*hin).nint as usize });
                }
                ind += unsafe { 4 * (*hin).nint as usize };
                if unsafe { (*hin).nreal } != 0 {
                    let data = unsafe {
                        core::slice::from_raw_parts_mut(
                            (*ext_data).add(ind).cast::<f32>(),
                            (*hin).nreal as usize,
                        )
                    };
                    mrc_swap_floats(data, unsafe { (*hin).nreal as usize });
                }
                ind += unsafe { 4 * (*hin).nreal as usize };
            }
        }
    }
    0
}

/// Matches C `mrcWriteExtraHeader(MrcHeader *, unsigned char *, int)` (`mrcfiles.c:653`).
pub unsafe fn mrc_write_extra_header(hout: *mut MrcHeader, ext_data: *mut u8, next: i32) -> i32 {
    if hout.is_null() || unsafe { (*hout).fp.is_null() } || ext_data.is_null() || next <= 0 {
        return 1;
    }
    let file = unsafe { ii_lookup_file_from_fp((*hout).fp.cast::<libc::FILE>()) };
    if !file.is_null() && unsafe { (*file).file } != IIFILE_MRC {
        return 6;
    }
    if unsafe {
        libc::fseek(
            (*hout).fp.cast::<libc::FILE>(),
            MRC_HEADER_SIZE as i64,
            libc::SEEK_SET,
        )
    } != 0
    {
        return 2;
    }
    if unsafe {
        libc::fwrite(
            ext_data.cast::<c_void>(),
            1,
            next as usize,
            (*hout).fp.cast::<libc::FILE>(),
        )
    } != next as usize
    {
        return 5;
    }
    unsafe {
        (*hout).next = next;
        (*hout).header_size = MRC_HEADER_SIZE as i32 + next;
    }
    0
}

/// Matches C `mrcCopyExtraHeader(MrcHeader *, MrcHeader *)` (`mrcfiles.c:578`).
pub unsafe fn mrc_copy_extra_header(hin: *mut MrcHeader, hout: *mut MrcHeader) -> i32 {
    if hin.is_null() || hout.is_null() || unsafe { (*hout).swapped } != 0 {
        return 1;
    }
    let mut ext_data = core::ptr::null_mut();
    let index = unsafe { mrc_read_extra_header(hin, &mut ext_data) };
    if index != 0 {
        return index.max(0);
    }
    let index = unsafe { mrc_write_extra_header(hout, ext_data, (*hin).next) };
    unsafe { libc::free(ext_data.cast::<c_void>()) };
    if index != 0 {
        return index;
    }
    unsafe {
        (*hout).nint = (*hin).nint;
        (*hout).nreal = (*hin).nreal;
        mrc_copy_valid_extended_type(&*hin, &mut *hout);
    }
    0
}

/// Matches C `mrcGetDataMemory(IloadInfo *, size_t, int, int)` (`mrcfiles.c:1562`).
pub unsafe fn mrc_get_data_memory(
    li: *mut LoadInfo,
    xysize: usize,
    zsize: i32,
    pixsize: i32,
) -> *mut *mut u8 {
    let mut contig = 0;
    if !li.is_null() {
        contig = unsafe { (*li).contig };
    }
    let idata = unsafe {
        libc::malloc((zsize as usize).wrapping_mul(core::mem::size_of::<*mut u8>()))
            .cast::<*mut u8>()
    };
    if idata.is_null() {
        return core::ptr::null_mut();
    }
    for index in 0..zsize {
        unsafe { *idata.add(index as usize) = core::ptr::null_mut() };
    }

    if contig != 0 {
        let data = unsafe {
            libc::malloc(
                xysize
                    .wrapping_mul(zsize as usize)
                    .wrapping_mul(pixsize as usize),
            )
            .cast::<u8>()
        };
        if data.is_null() {
            unsafe {
                b3d_error(
                    stderr,
                    format_args!(
                        "WARNING: mrcGetDataMemory - Not enough contiguous memory to load image data.\n"
                    ),
                );
                if !li.is_null() {
                    (*li).contig = 0;
                }
            }
        } else {
            for index in 0..zsize {
                unsafe {
                    *idata.add(index as usize) = data.add(
                        xysize
                            .wrapping_mul(index as usize)
                            .wrapping_mul(pixsize as usize),
                    );
                }
            }
            return idata;
        }
    }

    for index in 0..zsize {
        unsafe {
            *idata.add(index as usize) =
                libc::malloc(xysize.wrapping_mul(pixsize as usize)).cast::<u8>();
        }
        if unsafe { (*idata.add(index as usize)).is_null() } {
            unsafe {
                b3d_error(
                    stderr,
                    format_args!(
                        "ERROR: mrcGetDataMemory - Not enough memory for image data after {} sections.\n",
                        index
                    ),
                );
                mrc_free_data_memory(idata, 0, zsize);
            }
            return core::ptr::null_mut();
        }
    }
    idata
}

/// Matches C `mrcFreeDataMemory(unsigned char **, int, int)` (`mrcfiles.c:1613`).
pub unsafe fn mrc_free_data_memory(idata: *mut *mut u8, contig: i32, mut zsize: i32) {
    if contig != 0 {
        zsize = 1;
    }
    for index in 0..zsize {
        if !unsafe { (*idata.add(index as usize)).is_null() } {
            unsafe { libc::free((*idata.add(index as usize)).cast::<c_void>()) };
        }
    }
    unsafe { libc::free(idata.cast::<c_void>()) };
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
    mean /= (hdata.nx * hdata.ny * hdata.nz) as f64;
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
    let out_min = min_sign * (1.0_f64 + kscale as f64 * in_min as f64).ln() as f32;
    let out_max = (1.0_f64 + kscale as f64 * in_max as f64).ln() as f32;
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
    if ramptype == 3 {
        min = (min as f64).ln() as f32;
        max = (max as f64).ln() as f32;
    }
    if ramptype == 2 {
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
    let rscale = 256.0_f32 / range as f32;
    let mut slope = if max - min != 0.0 {
        255.0_f32 / (max - min)
    } else {
        1.0
    };
    slope *= rscale;
    let offset = -(((black as f32 / 255.0) * (max - min)) + min) * slope;
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
        li.ramp = 1;
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
    if li.ramp == 3 {
        min = (hdata.amin as f64).ln() as f32;
        max = (hdata.amax as f64).ln() as f32;
    }
    if li.ramp == 2 {
        min = (hdata.amin as f64).exp() as f32;
        max = (hdata.amax as f64).exp() as f32;
    }
    let mut range = li.white - li.black + 1;
    if range == 0 {
        range = 1;
    }
    let rscale = 256.0_f32 / range as f32;
    li.slope = if max - min != 0.0 {
        255.0_f32 / (max - min)
    } else {
        1.0
    };
    li.slope *= rscale;
    li.offset = -(((li.black as f32 / 255.0) * (max - min)) + min) * li.slope;
}

/// Matches C `get_loadinfo(MrcHeader *, IloadInfo *)` (`mrcfiles.c:1934`).
pub unsafe fn get_loadinfo(hdata: *mut MrcHeader, li: *mut LoadInfo) -> i32 {
    let mut line = [0 as c_char; 128];
    unsafe {
        libc::fflush(stdout);
        libc::fflush(stdin);
        libc::printf(c" Enter (min x, max x). (return for default) >".as_ptr());
        libc::fgets(line.as_mut_ptr(), 128, stdin);
        if line[0] != 0 {
            libc::sscanf(
                line.as_ptr(),
                c"%d%*c%d\n".as_ptr(),
                core::ptr::addr_of_mut!((*li).xmin),
                core::ptr::addr_of_mut!((*li).xmax),
            );
        } else {
            (*li).xmin = 0;
            (*li).xmax = (*hdata).nx - 1;
        }

        libc::printf(c" Enter (min y, max y). (return for default)  >".as_ptr());
        libc::fgets(line.as_mut_ptr(), 128, stdin);
        if line[0] != 0 {
            libc::sscanf(
                line.as_ptr(),
                c"%d%*c%d\n".as_ptr(),
                core::ptr::addr_of_mut!((*li).ymin),
                core::ptr::addr_of_mut!((*li).ymax),
            );
        } else {
            (*li).ymin = 0;
            (*li).ymax = (*hdata).ny - 1;
        }

        libc::printf(c" Enter sections (low, high)  >".as_ptr());
        libc::fgets(line.as_mut_ptr(), 128, stdin);
        if line[0] != 0 {
            libc::sscanf(
                line.as_ptr(),
                c"%d%*c%d\n".as_ptr(),
                core::ptr::addr_of_mut!((*li).zmin),
                core::ptr::addr_of_mut!((*li).zmax),
            );
        } else {
            (*li).zmin = 0;
            (*li).zmax = (*hdata).nz - 1;
        }
        (*li).scale = 1;
    }
    1
}

/// Matches C `loadtilts(TiltInfo *, MrcHeader *)` (`mrcfiles.c:1979`).
pub unsafe fn loadtilts(ti: *mut TiltInfo, hdata: *mut MrcHeader) -> i32 {
    let mut filename = [0 as c_char; 128];
    let mut tiltflag = 0;
    let mut fin: *mut libc::FILE = core::ptr::null_mut();
    unsafe {
        while tiltflag == 0 {
            libc::printf(c"Do you wish to load a tilt info file? (y/n) >".as_ptr());
            match libc::getchar() {
                121 | 89 => tiltflag = 1,
                110 | 78 => tiltflag = 2,
                _ => {}
            }
            libc::getchar();
        }
        (*ti).tilt = libc::malloc((*hdata).nz as usize * core::mem::size_of::<f32>()).cast();
        if (*ti).tilt.is_null() {
            return 0;
        }
        if tiltflag == 2 {
            if (*hdata).nz < 2 {
                *(*ti).tilt = 0.0;
            } else {
                let tiltoff = -60.0;
                let tslope = 120.0 / ((*hdata).nz as f32 - 1.0);
                for i in 0..(*hdata).nz {
                    *(*ti).tilt.add(i as usize) = tiltoff + i as f32 * tslope;
                }
            }
            (*ti).axis_z = ((*hdata).nz / 2) as f32;
            (*ti).axis_x = ((*hdata).nx / 2) as f32;
        }
        if tiltflag == 1 {
            getfilename(
                filename.as_mut_ptr(),
                c"Enter tilt info filename. >".as_ptr(),
            );
            fin = libc::fopen(filename.as_ptr(), c"r".as_ptr());
            if fin.is_null() {
                b3d_error(
                    stderr,
                    format_args!(
                        "ERROR: loadtilts - Couldn't load {}.\n",
                        core::ffi::CStr::from_ptr(filename.as_ptr()).to_string_lossy()
                    ),
                );
                return 0;
            }
            for i in 0..(*hdata).nz {
                libc::fscanf(fin, c"%f".as_ptr(), (*ti).tilt.add(i as usize));
            }
            libc::fscanf(fin, c"%f".as_ptr(), core::ptr::addr_of_mut!((*ti).axis_x));
            libc::fscanf(fin, c"%f".as_ptr(), core::ptr::addr_of_mut!((*ti).axis_z));
        }
    }
    1
}

/// Matches C `mrc_mread_slice(FILE *, MrcHeader *, int, char)` (`mrcfiles.c:983`).
pub unsafe fn mrc_mread_slice(
    fin: *mut libc::FILE,
    hdata: *mut MrcHeader,
    slice: i32,
    axis: c_char,
) -> *mut c_void {
    let bsize = unsafe {
        match axis as u8 {
            b'x' | b'X' => (*hdata).ny * (*hdata).nz,
            b'y' | b'Y' => (*hdata).nx * (*hdata).nz,
            b'z' | b'Z' => (*hdata).nx * (*hdata).ny,
            _ => {
                b3d_error(
                    stderr,
                    format_args!("ERROR: mrc_mread_slice - axis error.\n"),
                );
                return core::ptr::null_mut();
            }
        }
    };
    let mut dsize = 0;
    let mut csize = 0;
    if mrc_getdcsize(unsafe { (*hdata).mode }, &mut dsize, &mut csize) != 0 {
        unsafe {
            b3d_error(
                stderr,
                format_args!("ERROR: mrc_mread_slice - unknown mode.\n"),
            )
        };
        return core::ptr::null_mut();
    }
    let buf = unsafe { libc::malloc((dsize * csize * bsize) as usize) };
    if buf.is_null() {
        unsafe {
            b3d_error(
                stderr,
                format_args!("ERROR: mrc_mread_slice - couldn't get memory.\n"),
            )
        };
        return core::ptr::null_mut();
    }
    if unsafe { mrc_read_slice(buf, fin, hdata, slice, axis) } == 0 {
        return buf;
    }
    unsafe { libc::free(buf) };
    core::ptr::null_mut()
}

/// Matches C `mrc_read_slice(void *, FILE *, MrcHeader *, int, char)` (`mrcfiles.c:1037`).
pub unsafe fn mrc_read_slice(
    buf: *mut c_void,
    fin: *mut libc::FILE,
    hdata: *mut MrcHeader,
    slice: i32,
    axis: c_char,
) -> i32 {
    unsafe {
        let mut li = core::mem::zeroed::<LoadInfo>();
        mrc_init_li(Some(&mut li), None);
        mrc_init_li(Some(&mut li), Some(&*hdata));
        if matches!(axis as u8, b'z' | b'Z' | b'y' | b'Y') {
            if matches!(axis as u8, b'y' | b'Y') {
                li.axis = 2;
            }
            let fp_save = (*hdata).fp;
            (*hdata).fp = fin.cast();
            let result =
                crate::imod::libiimod::mrcsec::mrc_read_section(hdata, &mut li, buf.cast(), slice);
            (*hdata).fp = fp_save;
            return result;
        }
        if !matches!(axis as u8, b'x' | b'X') {
            b3d_error(
                stderr,
                format_args!("ERROR: mrc_read_slice - axis error.\n"),
            );
            return -1;
        }
        let ii_file = ii_lookup_file_from_fp(fin);
        if !ii_file.is_null() && (*ii_file).file != IIFILE_MRC && (*ii_file).file != IIFILE_RAW {
            b3d_error(
                stderr,
                format_args!(
                    "ERROR: mrc_read_slice - Cannot read X slice from non-MRC-like file\n"
                ),
            );
            return -1;
        }
        if (*hdata).packed4bits != 0 {
            b3d_error(
                stderr,
                format_args!("ERROR: mrc_read_slice - Cannot read X slice from 4-bit file\n"),
            );
            return -1;
        }
        let mut dsize = 0;
        let mut csize = 0;
        if mrc_getdcsize((*hdata).mode, &mut dsize, &mut csize) != 0 || slice >= (*hdata).nx {
            return -1;
        }
        libc::rewind(fin);
        libc::fseek(fin, (*hdata).header_size as libc::c_long, libc::SEEK_SET);
        let dcsize = dsize * csize;
        libc::fseek(
            fin,
            slice as libc::c_long * dcsize as libc::c_long,
            libc::SEEK_CUR,
        );
        let mut data = buf.cast::<u8>();
        for _ in 0..(*hdata).nz {
            for _ in 0..(*hdata).ny {
                if libc::fread(data.cast(), dcsize as usize, 1, fin) != 1 {
                    b3d_error(
                        stderr,
                        format_args!("ERROR: mrc_read_slice x - fread error.\n"),
                    );
                    return -1;
                }
                data = data.add(dcsize as usize);
                libc::fseek(
                    fin,
                    (dcsize * ((*hdata).nx - 1)) as libc::c_long,
                    libc::SEEK_CUR,
                );
            }
            if (*hdata).section_skip != 0 {
                libc::fseek(fin, (*hdata).section_skip as libc::c_long, libc::SEEK_CUR);
            }
        }
        if (*hdata).swapped != 0 {
            if matches!(
                (*hdata).mode,
                MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_COMPLEX_SHORT
            ) {
                mrc_swap_shorts(
                    core::slice::from_raw_parts_mut(
                        buf.cast(),
                        ((*hdata).ny * (*hdata).nz * csize) as usize,
                    ),
                    ((*hdata).ny * (*hdata).nz * csize) as usize,
                );
            } else if matches!((*hdata).mode, MRC_MODE_FLOAT | MRC_MODE_COMPLEX_FLOAT) {
                mrc_swap_floats(
                    core::slice::from_raw_parts_mut(
                        buf.cast(),
                        ((*hdata).ny * (*hdata).nz * csize) as usize,
                    ),
                    ((*hdata).ny * (*hdata).nz * csize) as usize,
                );
            }
        }
        if (*hdata).mode == MRC_MODE_BYTE && (*hdata).bytes_signed != 0 {
            for item in core::slice::from_raw_parts_mut(
                buf.cast::<i8>(),
                ((*hdata).ny * (*hdata).nz) as usize,
            ) {
                *item = (*item as i32 + 128) as i8;
            }
        }
        libc::fflush(fin);
    }
    0
}

/// Matches C `mrcReadFloatSlice(b3dFloat *, MrcHeader *, int)` (`mrcfiles.c:1148`).
pub unsafe fn mrc_read_float_slice(buf: *mut f32, hdata: *mut MrcHeader, slice: i32) -> i32 {
    unsafe {
        let mut li = core::mem::zeroed::<LoadInfo>();
        mrc_init_li(Some(&mut li), None);
        li.xmin = 0;
        li.xmax = (*hdata).nx - 1;
        li.ymin = 0;
        li.ymax = (*hdata).ny - 1;
        mrc_read_z_float(hdata, &mut li, buf, slice)
    }
}

/// Matches C `mrc_read_byte(FILE *, MrcHeader *, IloadInfo *, void (*)(const char *))`
/// (`mrcfiles.c:1175`).
pub unsafe fn mrc_read_byte(
    fin: *mut libc::FILE,
    hdata: *mut MrcHeader,
    mut li: *mut LoadInfo,
    func: Option<unsafe extern "C" fn(*const c_char)>,
) -> *mut *mut u8 {
    unsafe {
        if fin.is_null() || hdata.is_null() {
            return core::ptr::null_mut();
        }
        let mut li_local = core::mem::zeroed::<LoadInfo>();
        if li.is_null() {
            li = core::ptr::addr_of_mut!(li_local);
            mrc_init_li(Some(&mut *li), None);
            mrc_init_li(Some(&mut *li), Some(&*hdata));
        }
        let fp_save = (*hdata).fp;
        (*hdata).fp = fin.cast();
        let xsize = (*li).xmax - (*li).xmin + 1;
        let ysize = (*li).ymax - (*li).ymin + 1;
        let zsize = (*li).zmax - (*li).zmin + 1;
        let xysize = xsize as usize * ysize as usize;
        let (slope, offset) = mrc_contrast_scaling(
            &*hdata,
            (*li).smin,
            (*li).smax,
            (*li).black,
            (*li).white,
            (*li).ramp,
        );
        (*li).slope = slope;
        (*li).offset = offset;
        if let Some(callback) = func {
            let mut statstr = [0_i8; 128];
            if zsize > 1 {
                libc::snprintf(
                    statstr.as_mut_ptr(),
                    statstr.len(),
                    c"Image size %d x %d, %d sections.\n".as_ptr(),
                    xsize,
                    ysize,
                    zsize,
                );
            } else {
                libc::snprintf(
                    statstr.as_mut_ptr(),
                    statstr.len(),
                    c"Image size %d x %d.\n".as_ptr(),
                    xsize,
                    ysize,
                );
            }
            callback(statstr.as_ptr());
        }
        let idata = mrc_get_data_memory(li, xysize, zsize, 1);
        if idata.is_null() {
            (*hdata).fp = fp_save;
            return core::ptr::null_mut();
        }
        if let Some(callback) = func {
            let mut statstr = [0_i8; 128];
            libc::snprintf(
                statstr.as_mut_ptr(),
                statstr.len(),
                c"\nReading Image # %3.3d".as_ptr(),
                1,
            );
            callback(statstr.as_ptr());
        }
        for k in 0..zsize {
            if let Some(callback) = func {
                let mut statstr = [0_i8; 128];
                libc::snprintf(
                    statstr.as_mut_ptr(),
                    statstr.len(),
                    c"\rReading Image # %3.3d".as_ptr(),
                    k + 1,
                );
                callback(statstr.as_ptr());
            }
            if mrc_read_z_byte(hdata, li, *idata.add(k as usize), k + (*li).zmin) != 0 {
                mrc_free_data_memory(idata, (*li).contig, zsize);
                (*hdata).fp = fp_save;
                return core::ptr::null_mut();
            }
        }
        if let Some(callback) = func {
            callback(c"\n".as_ptr());
        }
        (*hdata).fp = fp_save;
        idata
    }
}

/// Matches C `mrc_write_idata(FILE *, MrcHeader *, void **)` (`mrcfiles.c:1333`).
pub unsafe fn mrc_write_idata(
    fout: *mut libc::FILE,
    hdata: *mut MrcHeader,
    data: *mut *mut c_void,
) -> i32 {
    unsafe {
        for k in 0..(*hdata).nz {
            let result = mrc_write_slice(*data.add(k as usize), fout, hdata, k, b'Z' as c_char);
            if result != 0 {
                return result;
            }
        }
    }
    0
}

/// Matches C `mrc_write_slice(void *, FILE *, MrcHeader *, int, char)` (`mrcfiles.c:1354`).
pub unsafe fn mrc_write_slice(
    buf: *mut c_void,
    fout: *mut libc::FILE,
    hdata: *mut MrcHeader,
    slice: i32,
    axis: c_char,
) -> i32 {
    if buf.is_null() || slice < 0 {
        return -1;
    }
    if matches!(axis as u8, b'z' | b'Z') {
        unsafe {
            let mut li = core::mem::zeroed::<LoadInfo>();
            mrc_init_li(Some(&mut li), None);
            mrc_init_li(Some(&mut li), Some(&*hdata));
            let fp_save = (*hdata).fp;
            (*hdata).fp = fout.cast();
            let result =
                crate::imod::libiimod::mrcsec::mrc_write_z(hdata, &mut li, buf.cast(), slice);
            (*hdata).fp = fp_save;
            return result;
        }
    }
    unsafe {
        let ii_file = ii_lookup_file_from_fp(fout);
        if !ii_file.is_null() && (*ii_file).file != IIFILE_MRC && (*ii_file).file != IIFILE_RAW {
            b3d_error(
                stderr,
                format_args!(
                    "ERROR: mrc_write_slice - Cannot write X or Y slice to non-MRC-like file\n"
                ),
            );
            return -1;
        }
        if (*hdata).packed4bits != 0 {
            b3d_error(
                stderr,
                format_args!("ERROR: mrc_write_slice - Cannot write X or Y slice to 4-bit file\n"),
            );
            return -1;
        }
        libc::rewind(fout);
        libc::fseek(fout, (*hdata).header_size as libc::c_long, libc::SEEK_SET);
        let nx = (*hdata).nx;
        let ny = (*hdata).ny;
        let mut dsize = 0;
        let mut csize = 0;
        if mrc_getdcsize((*hdata).mode, &mut dsize, &mut csize) != 0 {
            b3d_error(
                stderr,
                format_args!("ERROR: mrc_write_slice - unknown mode.\n"),
            );
            return -1;
        }
        let dcsize = dsize * csize;
        let (sxsize, sysize) = match axis as u8 {
            b'x' | b'X' if slice < nx => (ny, (*hdata).nz),
            b'y' | b'Y' if slice < ny => (nx, (*hdata).nz),
            b'x' | b'X' | b'y' | b'Y' => return -1,
            _ => {
                b3d_error(
                    stderr,
                    format_args!("ERROR: mrc_write_slice - axis error.\n"),
                );
                return -1;
            }
        };
        let bytes_signed = ((*hdata).mode == MRC_MODE_BYTE && (*hdata).bytes_signed != 0) as i32;
        let mut data = buf.cast::<u8>();
        let mut data_orig = core::ptr::null_mut::<u8>();
        if ((*hdata).swapped != 0 && dsize > 1) || bytes_signed != 0 {
            data = libc::malloc((sxsize as usize) * (sysize as usize) * (dcsize as usize)).cast();
            data_orig = data;
            if data.is_null() {
                b3d_error(
                    stderr,
                    format_args!("ERROR: mrc_write_slice - failure to allocate memory.\n"),
                );
                return -1;
            }
            if bytes_signed != 0 {
                b3d_shift_bytes(buf.cast(), data.cast(), sxsize, sysize, 1, 1);
            } else {
                core::ptr::copy_nonoverlapping(
                    buf.cast::<u8>(),
                    data,
                    (sxsize * sysize * dcsize) as usize,
                );
                if dsize == 2 {
                    mrc_swap_shorts(
                        core::slice::from_raw_parts_mut(
                            data.cast(),
                            (sxsize * sysize * csize) as usize,
                        ),
                        (sxsize * sysize * csize) as usize,
                    );
                } else {
                    mrc_swap_floats(
                        core::slice::from_raw_parts_mut(
                            data.cast(),
                            (sxsize * sysize * csize) as usize,
                        ),
                        (sxsize * sysize * csize) as usize,
                    );
                }
            }
        }
        let mut retval = 0;
        match axis as u8 {
            b'x' | b'X' => {
                libc::fseek(fout, (slice * dcsize) as libc::c_long, libc::SEEK_CUR);
                'sections: for _ in 0..(*hdata).nz {
                    for _ in 0..ny {
                        if libc::fwrite(data.cast(), dcsize as usize, 1, fout) != 1 {
                            b3d_error(
                                stderr,
                                format_args!("ERROR: mrc_write_slice x - fwrite error.\n"),
                            );
                            retval = -1;
                            break 'sections;
                        }
                        data = data.add(dcsize as usize);
                        libc::fseek(fout, (dcsize * (nx - 1)) as libc::c_long, libc::SEEK_CUR);
                    }
                }
            }
            b'y' | b'Y' => {
                mrc_huge_seek(fout, 0, 0, slice, 0, nx, ny, dcsize, libc::SEEK_CUR);
                for _ in 0..(*hdata).nz {
                    if libc::fwrite(data.cast(), dcsize as usize, nx as usize, fout) != nx as usize
                    {
                        b3d_error(
                            stderr,
                            format_args!("ERROR: mrc_write_slice y - fwrite error.\n"),
                        );
                        retval = -1;
                        break;
                    }
                    data = data.add((dcsize * nx) as usize);
                    mrc_huge_seek(fout, 0, 0, ny - 1, 0, nx, ny, dcsize, libc::SEEK_CUR);
                }
            }
            _ => unreachable!(),
        }
        if !data_orig.is_null() {
            libc::free(data_orig.cast());
        }
        retval
    }
}

/// Matches C `mrcWriteFFT(const char *, float *, int, int, int)` (`mrcfiles.c:1502`).
pub unsafe fn mrc_write_fft(
    filename: *const c_char,
    fft: *mut f32,
    nx_real: i32,
    ny_real: i32,
    if_scale: i32,
) -> i32 {
    unsafe {
        let mut retval = 1;
        let scale_fac = (1.0_f64 / ((nx_real as f64) * (ny_real as f64)).sqrt()) as f32;
        let shift_temp =
            libc::malloc(((2 * nx_real + 4) as usize) * core::mem::size_of::<f32>()).cast::<f32>();
        if shift_temp.is_null() {
            return 1;
        }
        crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
            fft,
            shift_temp,
            (nx_real + 2) / 2,
            ny_real,
            0,
        );
        if if_scale != 0 {
            for ind in 0..(nx_real + 2) * ny_real {
                *fft.add(ind as usize) *= scale_fac;
            }
        }
        let mut hdr = core::mem::zeroed::<MrcHeader>();
        mrc_head_new(
            &mut hdr,
            (nx_real + 2) / 2,
            ny_real,
            1,
            MRC_MODE_COMPLEX_FLOAT,
        );
        imod_backup_file(filename);
        let fp = libc::fopen(filename, c"wb".as_ptr());
        if !fp.is_null() {
            retval = 0;
            hdr.amax = -1.0e37_f32;
            hdr.amin = 1.0e37_f32;
            let mut asum = 0.0_f64;
            for ind in (0..(nx_real + 2) * ny_real).step_by(2) {
                let ampl = (*fft.add(ind as usize) * *fft.add(ind as usize)
                    + *fft.add((ind + 1) as usize) * *fft.add((ind + 1) as usize))
                .sqrt();
                asum += ampl as f64;
                hdr.amin = hdr.amin.min(ampl);
                hdr.amax = hdr.amax.max(ampl);
            }
            hdr.amean = (asum / (0.5 * (nx_real + 2) as f64 * ny_real as f64)) as f32;
            if mrc_head_write(fp, core::ptr::addr_of_mut!(hdr)) != 0
                || mrc_write_slice(
                    fft.cast(),
                    fp,
                    core::ptr::addr_of_mut!(hdr),
                    0,
                    b'Z' as c_char,
                ) != 0
            {
                retval = 1;
            }
            libc::fclose(fp);
        }
        if if_scale != 0 {
            for ind in 0..(nx_real + 2) * ny_real {
                *fft.add(ind as usize) /= scale_fac;
            }
        }
        crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
            fft,
            shift_temp,
            (nx_real + 2) / 2,
            ny_real,
            1,
        );
        libc::free(shift_temp.cast());
        retval
    }
}

/// Matches C `get_byte_map(float, float, int, int, int)` (`mrcfiles.c:1634`).
pub unsafe fn get_byte_map(
    slope: f32,
    offset: f32,
    outmin: i32,
    outmax: i32,
    bytes_signed: i32,
) -> *mut u8 {
    let base = if bytes_signed != 0 { 128 } else { 0 };
    for i in 0..256 {
        let mut ival = ((i as f32 * slope + offset) as f64 + 0.5).floor() as i32;
        if ival < outmin {
            ival = outmin;
        }
        if ival > outmax {
            ival = outmax;
        }
        if outmax > 255 {
            unsafe { BYTE_SMAP[(i + base) % 256] = ival as u16 };
        } else {
            unsafe { BYTE_MAP[(i + base) % 256] = ival as u8 };
        }
    }
    if outmax > 255 {
        core::ptr::addr_of_mut!(BYTE_SMAP).cast()
    } else {
        core::ptr::addr_of_mut!(BYTE_MAP).cast()
    }
}

/// Matches C `get_short_map(float, float, int, int, int, int, int)` (`mrcfiles.c:1677`).
pub unsafe fn get_short_map(
    slope: f32,
    offset: f32,
    outmin: i32,
    outmax: i32,
    ramptype: i32,
    swapbytes: i32,
    signedint: i32,
) -> *mut u8 {
    let to_short = outmax > 255;
    let map = unsafe { libc::malloc(65536 * if to_short { 2 } else { 1 }).cast::<u8>() };
    if map.is_null() {
        unsafe {
            b3d_error(
                stderr,
                format_args!("ERROR: get_short_map - getting memory"),
            )
        };
        return core::ptr::null_mut();
    }
    let smap = map.cast::<u16>();
    for i in 0..65536_u32 {
        let mut fpixel = i as f32;
        if i > 32767 && signedint != 0 {
            fpixel = i as i32 as f32 - 65536.0;
        }
        if ramptype == 2 {
            fpixel = (fpixel as f64).exp() as f32;
        }
        if ramptype == 3 {
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
            unsafe { *smap.add(index as usize) = ival as u16 };
        } else {
            unsafe { *map.add(index as usize) = ival as u8 };
        }
    }
    map
}

/// Matches C `getfilename(char *, char *)` (`mrcfiles.c:2046`).
pub unsafe fn getfilename(name: *mut c_char, prompt: *const c_char) -> i32 {
    unsafe {
        libc::printf(c"%s".as_ptr(), prompt);
        libc::fflush(stdout);
        let mut i = 0;
        loop {
            let c = libc::getchar();
            if i >= 255 || c == libc::EOF || c == b'\n' as i32 {
                break;
            }
            *name.add(i as usize) = c as c_char;
            i += 1;
        }
        *name.add(i as usize) = 0;
        i
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        unsafe {
            let path = c"IMOD/Etomo/unitTestData/headerTest.st";
            let fp = libc::fopen(path.as_ptr(), c"rb".as_ptr());
            assert!(!fp.is_null());
            let mut header: MrcHeader = core::mem::zeroed();
            assert_eq!(mrc_head_read(fp, &mut header), 0);
            assert_eq!(
                (header.nx, header.ny, header.nz, header.mode),
                (512, 512, 1, 1)
            );
            assert_eq!((header.mx, header.my, header.mz), (512, 512, 1));
            assert_eq!((header.mapc, header.mapr, header.maps), (1, 2, 3));
            assert_eq!(header.header_size, 2048);
            assert_eq!(header.fp, fp.cast());
            assert_eq!(&header.labels[0][..9], b"SerialEM:");
            assert_eq!(libc::fclose(fp), 0);
        }
    }

    #[test]
    fn mrc_head_write_round_trips_an_mrc_header() {
        unsafe {
            crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(0);
            crate::imod::libcfshr::b3dutil::set_4_bit_output_mode(0);
            let fp = libc::tmpfile();
            assert!(!fp.is_null());
            let mut written: MrcHeader = core::mem::zeroed();
            assert_eq!(mrc_head_new(&mut written, 16, 12, 3, MRC_MODE_SHORT), 0);
            written.amin = -12.0;
            written.amax = 300.0;
            written.amean = 42.0;
            written.xorg = 4.0;
            written.yorg = 5.0;
            written.zorg = 6.0;
            written.labels[0][..4].copy_from_slice(b"test");
            written.nlabl = 1;
            assert_eq!(mrc_head_write(fp, &mut written), 0);

            let mut read: MrcHeader = core::mem::zeroed();
            assert_eq!(mrc_head_read(fp, &mut read), 0);
            assert_eq!(
                (read.nx, read.ny, read.nz, read.mode),
                (16, 12, 3, MRC_MODE_SHORT)
            );
            assert_eq!((read.amin, read.amax, read.amean), (-12.0, 300.0, 42.0));
            assert_eq!((read.xorg, read.yorg, read.zorg), (4.0, 5.0, 6.0));
            assert_eq!(&read.labels[0][..4], b"test");
            assert_eq!(libc::fclose(fp), 0);
            crate::imod::libcfshr::b3dutil::override_invert_mrc_origin(-1);
        }
    }

    #[test]
    fn size_can_be_4_bit_k2_super_res_uses_the_source_tolerance() {
        assert_eq!(size_can_be_4_bit_k2_super_res(3836, 7422), 1);
        assert_eq!(size_can_be_4_bit_k2_super_res(3835, 7420), 0);
    }

    #[test]
    fn fix_title_padding_keeps_the_c_terminator_and_fills_to_80() {
        let mut label = [0_u8; MRC_LABEL_SIZE + 1];
        label[..3].copy_from_slice(b"abc");
        fix_title_padding(&mut label);
        assert_eq!(&label[..3], b"abc");
        assert!(label[3..MRC_LABEL_SIZE].iter().all(|value| *value == b' '));
        assert_eq!(label[MRC_LABEL_SIZE], 0);
    }

    #[test]
    fn mrc_test_size_checks_all_source_constraints() {
        let mut header: MrcHeader = unsafe { core::mem::zeroed() };
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
        let mut header: MrcHeader = unsafe { core::mem::zeroed() };
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
        let mut header: MrcHeader = unsafe { core::mem::zeroed() };
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
        let mut input: MrcHeader = unsafe { core::mem::zeroed() };
        assert_eq!(mrc_head_new(&mut input, 4, 5, 6, MRC_MODE_FLOAT), 0);
        assert_eq!((input.mx, input.my, input.mz), (4, 5, 6));
        assert_eq!((input.amin, input.amax), (f32::MAX, -f32::MAX));
        assert_eq!(input.imod_stamp, IMOD_MRC_STAMP);
        assert_eq!(input.ext_type, [b' '; 4]);
        mrc_set_scale(&mut input, 2.0, 3.0, 4.0);
        input.tiltangles[3] = 7.0;
        input.xorg = 8.0;
        let mut output: MrcHeader = unsafe { core::mem::zeroed() };
        mrc_head_new(&mut output, 8, 10, 12, MRC_MODE_FLOAT);
        mrc_coord_cp(&mut output, &input);
        assert_eq!(mrc_get_scale(&output), (2.0, 3.0, 4.0));
        assert_eq!((output.tiltangles[3], output.xorg), (7.0, 8.0));
    }

    #[test]
    fn labels_fill_copy_and_replace_at_the_source_limit() {
        let mut input: MrcHeader = unsafe { core::mem::zeroed() };
        mrc_head_new(&mut input, 1, 1, 1, MRC_MODE_FLOAT);
        mrc_head_label(&mut input, b"source label\0");
        assert_eq!(&input.labels[0][..12], b"source label");
        assert_eq!(input.nlabl, 1);
        let mut output: MrcHeader = unsafe { core::mem::zeroed() };
        assert_eq!(mrc_head_label_cp(&input, &mut output), 0);
        assert_eq!(output.labels[0], input.labels[0]);
        input.nlabl = 10;
        mrc_head_label(&mut input, b"replacement\0");
        assert_eq!(input.nlabl, 10);
        assert_eq!(&input.labels[9][..11], b"replacement");
    }

    #[test]
    fn copy_valid_extended_type_rejects_serialem_unknown_stamp() {
        let mut input: MrcHeader = unsafe { core::mem::zeroed() };
        input.nversion = 20140;
        input.ext_type = *b"AB12";
        input.nint = 8;
        input.nreal = 3;
        let mut output: MrcHeader = unsafe { core::mem::zeroed() };
        mrc_copy_valid_extended_type(&input, &mut output);
        assert_eq!(output.ext_type, [0; 4]);
        input.ext_type = *b"SERI";
        mrc_copy_valid_extended_type(&input, &mut output);
        assert_eq!(output.ext_type, *b"SERI");
    }

    #[test]
    fn mrc_byte_mmm_uses_all_source_planes_and_sets_header_statistics() {
        let mut header: MrcHeader = unsafe { core::mem::zeroed() };
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
        let mut header: MrcHeader = unsafe { core::mem::zeroed() };
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
        let mut li: LoadInfo = unsafe { core::mem::zeroed() };
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
        let mut header: MrcHeader = unsafe { core::mem::zeroed() };
        header.amin = 10.0;
        header.amax = 20.0;
        let mut li: LoadInfo = unsafe { core::mem::zeroed() };
        li.black = 0;
        li.white = 255;
        mrc_liso(&header, &mut li);
        assert_eq!((li.slope, li.offset), (25.5, -255.0));
    }

    #[test]
    fn byte_and_short_maps_preserve_scaling_clamping_and_index_rules() {
        unsafe {
            let map = get_byte_map(1.0, 0.0, 10, 200, 1);
            assert_eq!(*map.add(128), 10);
            assert_eq!(*map.add(129), 10);
            assert_eq!(*map.add(127), 200);
            let smap = get_byte_map(2.0, 0.0, 0, 500, 0).cast::<u16>();
            assert_eq!(*smap.add(255), 500);
            let short_map = get_short_map(1.0, 0.0, 0, 255, 1, 1, 0);
            assert_eq!(*short_map.add(0), 0);
            assert_eq!(*short_map.add(0x0100), 1);
            libc::free(short_map.cast());
        }
    }

    #[test]
    fn read_extra_header_preserves_c_allocation_io_and_swapping_contract() {
        unsafe {
            let file = libc::tmpfile();
            assert!(!file.is_null());
            let header_bytes = [0_u8; MRC_HEADER_SIZE];
            assert_eq!(
                libc::fwrite(header_bytes.as_ptr().cast(), 1, header_bytes.len(), file),
                header_bytes.len()
            );
            let extra: Vec<u8> = (0..44).collect();
            assert_eq!(
                libc::fwrite(extra.as_ptr().cast(), 1, extra.len(), file),
                extra.len()
            );
            assert_eq!(libc::fflush(file), 0);

            let mut header: MrcHeader = core::mem::zeroed();
            header.fp = file.cast();
            header.next = extra.len() as i32;
            header.swapped = 1;
            header.nint = 8;
            header.nreal = 3;
            let mut data = core::ptr::null_mut();
            assert_eq!(mrc_read_extra_header(&mut header, &mut data), 0);
            let result = core::slice::from_raw_parts(data, extra.len());
            for index in (0..extra.len()).step_by(2) {
                assert_eq!(result[index], extra[index + 1]);
                assert_eq!(result[index + 1], extra[index]);
            }
            libc::free(data.cast());
            assert_eq!(libc::fclose(file), 0);
        }
    }

    #[test]
    fn data_memory_allocation_preserves_source_plane_and_contiguous_layouts() {
        unsafe {
            let mut separate: LoadInfo = core::mem::zeroed();
            let separate_data = mrc_get_data_memory(&mut separate, 5, 3, 2);
            assert!(!separate_data.is_null());
            for index in 0..3 {
                assert!(!(*separate_data.add(index)).is_null());
                *(*separate_data.add(index)).add(9) = index as u8;
            }
            mrc_free_data_memory(separate_data, separate.contig, 3);

            let mut contiguous: LoadInfo = core::mem::zeroed();
            contiguous.contig = 1;
            let contiguous_data = mrc_get_data_memory(&mut contiguous, 5, 3, 2);
            assert!(!contiguous_data.is_null());
            assert_eq!(*contiguous_data.add(1), (*contiguous_data).add(10));
            assert_eq!(*contiguous_data.add(2), (*contiguous_data).add(20));
            mrc_free_data_memory(contiguous_data, contiguous.contig, 3);
        }
    }

    #[test]
    fn copy_extra_header_preserves_source_io_metadata_and_non_mrc_rejection() {
        unsafe {
            let input_file = libc::tmpfile();
            let output_file = libc::tmpfile();
            assert!(!input_file.is_null() && !output_file.is_null());
            let disk_header = [0_u8; MRC_HEADER_SIZE];
            let extra = [4_u8, 2, 9, 1, 8, 3];
            for file in [input_file, output_file] {
                assert_eq!(
                    libc::fwrite(disk_header.as_ptr().cast(), 1, disk_header.len(), file),
                    disk_header.len()
                );
                assert_eq!(libc::fflush(file), 0);
            }
            assert_eq!(
                libc::fwrite(extra.as_ptr().cast(), 1, extra.len(), input_file),
                extra.len()
            );
            assert_eq!(libc::fflush(input_file), 0);

            let mut input: MrcHeader = core::mem::zeroed();
            input.fp = input_file.cast();
            input.next = extra.len() as i32;
            input.nint = 2;
            input.nreal = 1;
            input.ext_type = *b"AGAR";
            let mut output: MrcHeader = core::mem::zeroed();
            output.fp = output_file.cast();
            assert_eq!(mrc_copy_extra_header(&mut input, &mut output), 0);
            assert_eq!(
                (output.next, output.header_size, output.nint, output.nreal),
                (6, 1030, 2, 1)
            );
            assert_eq!(output.ext_type, *b"AGAR");
            assert_eq!(
                libc::fseek(output_file, MRC_HEADER_SIZE as i64, libc::SEEK_SET),
                0
            );
            let mut copied = [0_u8; 6];
            assert_eq!(
                libc::fread(copied.as_mut_ptr().cast(), 1, copied.len(), output_file),
                copied.len()
            );
            assert_eq!(copied, extra);

            let image_file = crate::imod::libiimod::iimage::ii_new();
            (*image_file).fp = output_file;
            (*image_file).file = crate::imod::libiimod::iimage::IIFILE_HDF;
            assert_eq!(
                crate::imod::libiimod::iimage::add_to_opened_list(image_file),
                0
            );
            assert_eq!(
                mrc_write_extra_header(&mut output, extra.as_ptr().cast_mut(), extra.len() as i32),
                6
            );
            crate::imod::libiimod::iimage::remove_from_opened_list(image_file);
            libc::free(image_file.cast());
            assert_eq!(libc::fclose(input_file), 0);
            assert_eq!(libc::fclose(output_file), 0);
        }
    }

    #[test]
    fn mrc_header_disk_prefix_offsets_match_the_source_fread_layout() {
        assert_eq!(core::mem::offset_of!(MrcHeader, nx), 0);
        assert_eq!(core::mem::offset_of!(MrcHeader, next), 92);
        assert_eq!(core::mem::offset_of!(MrcHeader, ext_type), 104);
        assert_eq!(core::mem::offset_of!(MrcHeader, nversion), 108);
        assert_eq!(core::mem::offset_of!(MrcHeader, nint), 128);
        assert_eq!(core::mem::offset_of!(MrcHeader, imod_stamp), 152);
        assert_eq!(core::mem::offset_of!(MrcHeader, xorg), 196);
        assert_eq!(core::mem::offset_of!(MrcHeader, cmap), 208);
        assert_eq!(core::mem::offset_of!(MrcHeader, rms), 216);
        assert_eq!(core::mem::offset_of!(MrcHeader, nlabl), 220);
        assert_eq!(core::mem::offset_of!(MrcHeader, labels), 224);
    }

    #[test]
    fn mrc_write_slice_x_writes_the_source_strided_plane_layout() {
        unsafe {
            let file = libc::tmpfile();
            assert!(!file.is_null());
            let mut header: MrcHeader = core::mem::zeroed();
            assert_eq!(mrc_head_new(&mut header, 2, 2, 2, MRC_MODE_BYTE), 0);
            header.bytes_signed = 0;
            assert_eq!(mrc_head_write(file, &mut header), 0);
            let empty = [0_u8; 8];
            assert_eq!(
                libc::fwrite(empty.as_ptr().cast(), 1, empty.len(), file),
                empty.len()
            );
            let plane = [10_u8, 20, 30, 40];
            assert_eq!(
                mrc_write_slice(
                    plane.as_ptr().cast_mut().cast(),
                    file,
                    &mut header,
                    0,
                    b'X' as c_char
                ),
                0
            );
            assert_eq!(
                libc::fseek(file, MRC_HEADER_SIZE as libc::c_long, libc::SEEK_SET),
                0
            );
            let mut written = [0_u8; 8];
            assert_eq!(
                libc::fread(written.as_mut_ptr().cast(), 1, written.len(), file),
                written.len()
            );
            assert_eq!(written, [10, 0, 20, 0, 30, 0, 40, 0]);
            assert_eq!(libc::fclose(file), 0);
        }
    }

    #[test]
    fn mrc_z_slice_round_trip_uses_the_real_file_dispatch_path() {
        unsafe {
            let file = libc::tmpfile();
            assert!(!file.is_null());
            let mut header: MrcHeader = core::mem::zeroed();
            assert_eq!(mrc_head_new(&mut header, 3, 2, 1, MRC_MODE_BYTE), 0);
            header.amin = 0.0;
            header.amax = 255.0;
            assert_eq!(mrc_head_write(file, &mut header), 0);
            let written = [3_u8, 1, 4, 1, 5, 9];
            assert_eq!(
                mrc_write_slice(
                    written.as_ptr().cast_mut().cast(),
                    file,
                    &mut header,
                    0,
                    b'Z' as c_char,
                ),
                0
            );
            let mut read = [0_u8; 6];
            assert_eq!(
                mrc_read_slice(
                    read.as_mut_ptr().cast(),
                    file,
                    &mut header,
                    0,
                    b'Z' as c_char,
                ),
                0
            );
            assert_eq!(read, written);
            assert_eq!(libc::fclose(file), 0);
        }
    }
}
