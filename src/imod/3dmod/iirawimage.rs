//! Translation of `IMOD/3dmod/iirawimage.cpp` and `iirawimage.h`.
//!
//! The native raw-image dialog is an explicit boundary: applications which
//! need the interactive `RawImageForm` install `RAW_IMAGE_DIALOG` before
//! calling [`ii_raw_check`].  Command-line raw-image options set `all_match`,
//! as in the source, and consequently do not cross that boundary.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::imod::libcfshr::b3dutil::{b3drand, b3dsrand};
use crate::imod::libcfshr::islice::slice_create;
use crate::imod::libcfshr::samplemeansd::sample_mean_sd;
use crate::imod::libiimod::iilikemrc::{
    RAW_MODE_BYTE, RAW_MODE_FLOAT, RAW_MODE_SBYTE, RAW_MODE_SHORT, RAW_MODE_USHORT,
    ii_setup_raw_headers,
};
use crate::imod::libiimod::iimage::{
    IIERR_BAD_CALL, IIERR_IO_ERROR, IIERR_NOT_FORMAT, IIFILE_MRC, ImodImageFile, RawImageInfo,
    ii_default_min_max_mean,
};
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT,
    MRC_MODE_USHORT, MrcHeader, mrc_getdcsize, mrc_head_read, mrc_head_write, mrc_init_li,
};
use crate::imod::libiimod::mrcsec::mrc_read_z;
use crate::imod::libiimod::mrcslice::slice_mmm;

/// `RAW_MODE_RGB`, local in the paired source through the raw-image headers.
pub const RAW_MODE_RGB: i32 = 6;
pub const RAW_MODE_COMPLEX_FLOAT: i32 = 5;

/// Resident source `RawImageInfo`, represented safely while it is owned by 3dmod.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawImageState {
    pub type_: i32,
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub swap_bytes: bool,
    pub header_size: i32,
    pub amin: f32,
    pub amax: f32,
    pub scan_min_max: bool,
    pub all_match: bool,
    pub section_skip: i32,
    pub y_inverted: bool,
    pub pixel: f32,
    pub z_pixel: f32,
}

/// C static `info = {0, 64, 64, 1, ...}`.
pub static RAW_IMAGE_INFO: LazyLock<Mutex<RawImageState>> = LazyLock::new(|| {
    Mutex::new(RawImageState {
        type_: RAW_MODE_SBYTE,
        nx: 64,
        ny: 64,
        nz: 1,
        swap_bytes: false,
        header_size: 0,
        amin: 0.,
        amax: 255.,
        scan_min_max: true,
        all_match: false,
        section_skip: 0,
        y_inverted: false,
        pixel: 0.,
        z_pixel: 0.,
    })
});

/// `App->cvi` and `ImodPrefs->loadIntIfEstimate()` fields consumed by this unit.
#[derive(Clone, Copy, Debug, Default)]
pub struct RawImageScanConfig {
    pub scale_scan_type: i32,
    pub switch_to_ushort: bool,
    pub raw_image_store: i32,
    pub store_scan_in_mrc: bool,
    pub load_int_if_estimate: bool,
}

pub static RAW_IMAGE_SCAN_CONFIG: LazyLock<Mutex<RawImageScanConfig>> =
    LazyLock::new(|| Mutex::new(RawImageScanConfig::default()));

/// Native Qt dialog boundary for `RawImageForm::exec`; nonzero accepts the form.
pub type RawImageDialog = fn(&str, &mut RawImageState) -> i32;
pub static RAW_IMAGE_DIALOG: LazyLock<Mutex<Option<RawImageDialog>>> =
    LazyLock::new(|| Mutex::new(None));

/// C `static int seed = 123456` in `iiRawScan`.
static RAW_SCAN_SEED: AtomicI32 = AtomicI32::new(123456);

/// C `iiRawSetSize`.
pub fn ii_raw_set_size(nx: i32, ny: i32, nz: i32) {
    let mut info = RAW_IMAGE_INFO
        .lock()
        .expect("raw image info mutex poisoned");
    info.nx = nx.max(1);
    info.ny = ny.max(1);
    info.nz = nz.max(1);
    info.all_match = true;
}

/// C `iiRawSetMode`.
pub fn ii_raw_set_mode(mode: i32) -> i32 {
    let mut info = RAW_IMAGE_INFO
        .lock()
        .expect("raw image info mutex poisoned");
    match mode {
        -1 => info.type_ = RAW_MODE_SBYTE,
        MRC_MODE_BYTE => info.type_ = RAW_MODE_BYTE,
        MRC_MODE_SHORT => info.type_ = RAW_MODE_SHORT,
        MRC_MODE_USHORT => info.type_ = RAW_MODE_USHORT,
        MRC_MODE_FLOAT => info.type_ = RAW_MODE_FLOAT,
        MRC_MODE_COMPLEX_FLOAT => info.type_ = RAW_MODE_COMPLEX_FLOAT,
        MRC_MODE_RGB => info.type_ = RAW_MODE_RGB,
        _ => return 1,
    }
    0
}

/// C `iiRawSetHeaderSize`.
pub fn ii_raw_set_header_size(size: i32) {
    RAW_IMAGE_INFO
        .lock()
        .expect("raw image info mutex poisoned")
        .header_size = size.max(0);
}

/// C `iiRawSetSwap`.
pub fn ii_raw_set_swap() {
    RAW_IMAGE_INFO
        .lock()
        .expect("raw image info mutex poisoned")
        .swap_bytes = true;
}

/// C `iiRawSetInverted`.
pub fn ii_raw_set_inverted() {
    RAW_IMAGE_INFO
        .lock()
        .expect("raw image info mutex poisoned")
        .y_inverted = true;
}

/// C `iiRawSetScale`.
pub fn ii_raw_set_scale(smin: f32, smax: f32) {
    let mut info = RAW_IMAGE_INFO
        .lock()
        .expect("raw image info mutex poisoned");
    info.amin = smin;
    info.amax = smax;
    info.scan_min_max = false;
}

/// C `iiRawCheck`.
#[unsafe(no_mangle)]
pub unsafe fn ii_raw_check(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() || unsafe { (*in_file).fp.is_none() } {
        return IIERR_BAD_CALL;
    }

    let filename = unsafe { (*in_file).filename.clone() }.unwrap_or_default();
    let str = filename
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(filename.as_str());
    let mut info = *RAW_IMAGE_INFO
        .lock()
        .expect("raw image info mutex poisoned");

    if !info.all_match {
        let dialog = *RAW_IMAGE_DIALOG
            .lock()
            .expect("raw image dialog mutex poisoned");
        if dialog.is_none_or(|dialog| dialog(str, &mut info) == 0) {
            return IIERR_NOT_FORMAT;
        }
        *RAW_IMAGE_INFO
            .lock()
            .expect("raw image info mutex poisoned") = info;
    }
    let mut raw_info = RawImageInfo {
        type_: info.type_,
        nx: info.nx,
        ny: info.ny,
        nz: info.nz,
        swap_bytes: i32::from(info.swap_bytes),
        header_size: info.header_size,
        amin: info.amin,
        amax: info.amax,
        scan_min_max: i32::from(info.scan_min_max),
        all_match: i32::from(info.all_match),
        section_skip: info.section_skip,
        y_inverted: i32::from(info.y_inverted),
        pixel: info.pixel,
        z_pixel: info.z_pixel,
    };
    let err = unsafe { ii_setup_raw_headers(&mut *in_file, &raw_info) };
    if err != 0 {
        return err;
    }
    ii_raw_scan(unsafe { &mut *in_file })
}

/// C `iiRawScan`.
pub fn ii_raw_scan(in_file: &mut ImodImageFile) -> i32 {
    let hdr = in_file.mrc_header.as_deref_mut();
    if in_file.fp.is_none() || hdr.is_none() {
        return 1;
    }
    let hdr = hdr.expect("checked header presence");
    let info = *RAW_IMAGE_INFO
        .lock()
        .expect("raw image info mutex poisoned");
    let mut config = RAW_IMAGE_SCAN_CONFIG
        .lock()
        .expect("raw scan config mutex poisoned");
    let mut amin = info.amin;
    let mut amax = info.amax;
    let mut li = LoadInfo::default();
    mrc_init_li(Some(&mut li), None);
    let mut do_scan = info.scan_min_max;
    let seed = RAW_SCAN_SEED.swap(0, Ordering::Relaxed);
    if seed != 0 {
        b3dsrand(&seed);
    }

    if (hdr.mode == MRC_MODE_BYTE || hdr.mode == MRC_MODE_RGB) && do_scan {
        amin = 0.;
        amax = 255.;
        do_scan = false;
    }
    if (hdr.mode == MRC_MODE_SHORT || hdr.mode == MRC_MODE_USHORT) && config.load_int_if_estimate {
        config.switch_to_ushort = true;
        let mut mean = 0.;
        ii_default_min_max_mean(in_file.type_, &mut amin, &mut amax, &mut mean);
        do_scan = false;
    }

    let mut do_mean_sd = false;
    if do_scan {
        let mut dsize = 0;
        let mut csize = 0;
        mrc_getdcsize(hdr.mode, &mut dsize, &mut csize);
        let border_frac = 0.025_f32;
        let subset_frac = 0.1_f32;
        let rand_frac = 0.1_f32;
        let expand_min_max_frac = 0.1_f32;
        let target_sample = 10000_f32;
        let subset_bytes = 1.0e6_f32;
        let (mut yborder, mut xborder, lines_to_scan, skip_lines) = if config.scale_scan_type != 0 {
            let yborder = (border_frac * hdr.ny as f32) as i32;
            let mut lines = (subset_bytes / (hdr.nx * dsize * csize) as f32) as i32;
            lines = lines.clamp(1, hdr.ny - 2 * yborder);
            (
                yborder,
                (border_frac * hdr.nx as f32) as i32,
                lines,
                (lines as f32 / subset_frac) as i32,
            )
        } else {
            (0, 0, hdr.ny, hdr.ny)
        };
        li.ymin = yborder;
        li.xmin = 0;
        li.xmax = hdr.nx - 1;
        let mut buffer = vec![
            0_u8;
            (hdr.nx as usize)
                .saturating_mul(dsize as usize)
                .saturating_mul(csize as usize)
                .saturating_mul(lines_to_scan as usize)
        ];
        let mode_is_real = matches!(hdr.mode, MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_FLOAT);
        do_mean_sd = mode_is_real && config.scale_scan_type > 1;
        if config.scale_scan_type != 0 && config.load_int_if_estimate {
            config.switch_to_ushort = true;
        }

        amin = 1.0e37;
        amax = -amin;
        let mut z = 0;
        let mut tot_pixels = 0_f64;
        let mut tot_sum = 0_f64;
        let mut tot_sum_sq = 0_f64;
        while z < hdr.nz {
            li.ymax = li.ymin + lines_to_scan - 1;
            if mrc_read_z(&mut *hdr, &mut li, &mut buffer, z) != 0 {
                return IIERR_IO_ERROR;
            }
            if mode_is_real && !do_mean_sd {
                for iy in 0..lines_to_scan {
                    match hdr.mode {
                        MRC_MODE_SHORT => {
                            for ind in xborder..hdr.nx - xborder {
                                let offset = ((iy * hdr.nx + ind) * 2) as usize;
                                let value =
                                    i16::from_ne_bytes(buffer[offset..][..2].try_into().unwrap())
                                        as f32;
                                amin = amin.min(value);
                                amax = amax.max(value);
                            }
                        }
                        MRC_MODE_USHORT => {
                            for ind in xborder..hdr.nx - xborder {
                                let offset = ((iy * hdr.nx + ind) * 2) as usize;
                                let value =
                                    u16::from_ne_bytes(buffer[offset..][..2].try_into().unwrap())
                                        as f32;
                                amin = amin.min(value);
                                amax = amax.max(value);
                            }
                        }
                        MRC_MODE_FLOAT => {
                            for ind in xborder..hdr.nx - xborder {
                                let offset = ((iy * hdr.nx + ind) * 4) as usize;
                                let value =
                                    f32::from_ne_bytes(buffer[offset..][..4].try_into().unwrap());
                                amin = amin.min(value);
                                amax = amax.max(value);
                            }
                        }
                        _ => {}
                    }
                }
            } else if mode_is_real {
                let buf_pixels = (hdr.nx - 2 * xborder) as f32 * lines_to_scan as f32;
                let sample_frac = (target_sample / buf_pixels).min(1.);
                let type_smsd = if hdr.mode == MRC_MODE_FLOAT {
                    6
                } else if hdr.mode == MRC_MODE_SHORT {
                    3
                } else {
                    2
                };
                let mut sample_mean = 0.;
                let mut sample_sd = 0.;
                // `makeLinePointers` becomes the line byte views the
                // translated `sampleMeanSD` takes; they are rebuilt per scan
                // because `buffer` is refilled between calls.
                let line_stride = hdr.nx as usize * (dsize * csize) as usize;
                let lines: Vec<&[u8]> = (0..lines_to_scan as usize)
                    .map(|index| &buffer[(line_stride * index)..])
                    .collect();
                sample_mean_sd(
                    Some(&lines),
                    type_smsd,
                    hdr.nx,
                    lines_to_scan,
                    sample_frac,
                    xborder,
                    0,
                    hdr.nx - 2 * xborder,
                    lines_to_scan,
                    Some(&mut sample_mean),
                    Some(&mut sample_sd),
                );
                let buf_pixels = buf_pixels * sample_frac;
                tot_pixels += buf_pixels as f64;
                tot_sum += sample_mean as f64 * buf_pixels as f64;
                tot_sum_sq += sample_sd as f64 * sample_sd as f64 * (buf_pixels as f64 - 1.)
                    + sample_mean as f64 * sample_mean as f64 * buf_pixels as f64;
            } else {
                let Some(mut slice) = slice_create(hdr.nx, lines_to_scan, hdr.mode) else {
                    return IIERR_IO_ERROR;
                };
                slice.data.copy_from_slice(&buffer);
                slice_mmm(slice.as_mut());
                amin = amin.min(slice.min);
                amax = amax.max(slice.max);
            }
            let mut full_skip = lines_to_scan;
            if config.scale_scan_type != 0 {
                full_skip = (skip_lines as f32 * (1. + rand_frac * (2. * b3drand() - 1.))) as i32;
            }
            let sec_skip = full_skip / hdr.ny;
            let rem_skip = full_skip % hdr.ny;
            z += sec_skip + (li.ymin + rem_skip) / hdr.ny;
            li.ymin = (li.ymin + rem_skip) % hdr.ny;
            if li.ymin < yborder {
                li.ymin = yborder;
            }
            if li.ymin + lines_to_scan > hdr.ny - yborder {
                if hdr.ny - yborder - li.ymin > lines_to_scan / 2 {
                    li.ymin = hdr.ny - yborder - lines_to_scan;
                } else {
                    li.ymin = yborder;
                    z += 1;
                }
            }
        }
        if do_mean_sd {
            in_file.amean = tot_sum as f32 / tot_pixels as f32;
            hdr.amean = in_file.amean;
            let fval =
                (tot_sum_sq - tot_pixels * hdr.amean as f64 * hdr.amean as f64) / (tot_pixels - 1.);
            in_file.rms = fval.max(0.).sqrt() as f32;
            hdr.rms = in_file.rms;
            let pow10 = 10_f32.powf((hdr.amean.abs() + 1.).log10().floor());
            in_file.smin = pow10 * ((hdr.amean / pow10).floor() - 1.);
            in_file.amin = in_file.smin;
            hdr.amin = in_file.smin;
            in_file.smax = pow10 * ((hdr.amean / pow10).floor() - 2.);
            in_file.amax = in_file.smax;
            hdr.amax = in_file.smax;
        }
        let expand = expand_min_max_frac * (amax - amin);
        if config.store_scan_in_mrc && in_file.file == IIFILE_MRC && in_file.filename.is_some() {
            let path = in_file.filename.clone().unwrap_or_default();
            {
                let file = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "rb+");
                if let Some(mut file) = file {
                    let mut temp = MrcHeader::default();
                    if mrc_head_read(&mut file, &mut temp) == 0 {
                        if do_mean_sd {
                            temp.amin = hdr.amin;
                            temp.amax = hdr.amax;
                            temp.amean = hdr.amean;
                            temp.rms = hdr.rms;
                        } else {
                            temp.amin = amin;
                            temp.amax = amax;
                            if config.scale_scan_type != 0 {
                                temp.amin -= expand;
                                temp.amax += expand;
                            }
                            if hdr.amean < hdr.amin.min(hdr.amax)
                                || (hdr.amean == 0. && (hdr.amin == 0. || hdr.amax == 0.))
                            {
                                let pow10 = 10_f32.powf((temp.amin.abs() + 1.).log10().floor());
                                temp.amean = pow10 * ((temp.amin / pow10).floor() - 1.);
                            }
                        }
                        mrc_head_write(&mut file, &mut temp);
                    }
                    drop(file);
                }
            }
        }
        if !do_mean_sd {
            if config.scale_scan_type != 0
                && (config.switch_to_ushort || config.raw_image_store == MRC_MODE_USHORT)
            {
                amin -= expand;
                amax += expand;
            }
            hdr.amin = amin;
            hdr.amax = amax;
        }
    }
    if !do_mean_sd {
        in_file.smin = amin;
        in_file.amin = amin;
        hdr.amin = amin;
        in_file.smax = amax;
        in_file.amax = amax;
        hdr.amax = amax;
        in_file.amean = (amax + amin) / 2.;
        hdr.amean = in_file.amean;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setters_retain_source_state() {
        ii_raw_set_size(0, -3, 2);
        let state = *RAW_IMAGE_INFO.lock().unwrap();
        assert_eq!(
            (state.nx, state.ny, state.nz, state.all_match),
            (1, 1, 2, true)
        );
        assert_eq!(ii_raw_set_mode(MRC_MODE_USHORT), 0);
        ii_raw_set_header_size(-3);
        ii_raw_set_swap();
        ii_raw_set_inverted();
        ii_raw_set_scale(-2., 5.);
        let state = *RAW_IMAGE_INFO.lock().unwrap();
        assert_eq!(
            (
                state.type_,
                state.header_size,
                state.swap_bytes,
                state.y_inverted
            ),
            (RAW_MODE_USHORT, 0, true, true)
        );
        assert_eq!(
            (state.amin, state.amax, state.scan_min_max),
            (-2., 5., false)
        );
    }

    #[test]
    fn invalid_mode_fails_without_changing_type() {
        let type_ = RAW_IMAGE_INFO.lock().unwrap().type_;
        assert_eq!(ii_raw_set_mode(99), 1);
        assert_eq!(RAW_IMAGE_INFO.lock().unwrap().type_, type_);
    }

    #[test]
    fn scan_reads_an_actual_raw_short_stream() {
        unsafe {
            let mut file = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let values = [-7_i16, 3, 12, -2];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(
                    core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), 8),
                    2,
                    values.len(),
                    &mut file
                ),
                values.len()
            );
            crate::imod::libcfshr::b3dutil::b3d_rewind(&mut file);
            let mut image = ImodImageFile::default();
            image.fp = Some(file.clone());
            image.filename = Some("scan.raw".into());
            {
                let mut info = RAW_IMAGE_INFO.lock().unwrap();
                *info = RawImageState {
                    type_: RAW_MODE_SHORT,
                    nx: 2,
                    ny: 2,
                    nz: 1,
                    swap_bytes: false,
                    header_size: 0,
                    amin: 0.,
                    amax: 255.,
                    scan_min_max: true,
                    all_match: true,
                    section_skip: 0,
                    y_inverted: false,
                    pixel: 0.,
                    z_pixel: 0.,
                };
            }
            *RAW_IMAGE_SCAN_CONFIG.lock().unwrap() = RawImageScanConfig::default();
            assert_eq!(ii_raw_check(&mut image), 0);
            assert_eq!((image.amin, image.amax, image.amean), (-7., 12., 2.5));
            (image.clean_up.unwrap())(&mut image);
            drop(file);
        }
    }
}
