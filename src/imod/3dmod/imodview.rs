//! Direct translation boundary for `IMOD/3dmod/imodview.cpp`, `imodview.h`,
//! and `imodviewP.h`.
//!
//! This unit owns `ViewInfo`/`ImodView` image and model state.  Window drawing,
//! Zap, Slicer, Qt, and OpenGL calls remain unavailable until their paired
//! source units are translated; their public calls retain IMOD's no-window
//! return convention instead of creating a substitute view.

use core::ffi::{CStr, c_char};
use core::ptr;

use crate::imod::libiimod::iimage::ImodImageFile;
use crate::imod::libiimod::mrcfiles::{MRC_MODE_RGB, MRC_MODE_USHORT};
use crate::imod::libimod::imodel::{Icont, Imod, Iobj, Ipoint};

pub const IMOD_MM_TOGGLE: i32 = 0;
pub const IMOD_MMOVIE: i32 = 0;
pub const IMOD_MMODEL: i32 = 1;
pub const IMOD_DRAW_IMAGE: i32 = 1;
pub const IMOD_DRAW_XYZ: i32 = 1 << 1;
pub const IMOD_DRAW_MOD: i32 = 1 << 2;
pub const IMOD_DRAW_ALL: i32 = IMOD_DRAW_IMAGE | IMOD_DRAW_XYZ | IMOD_DRAW_MOD;

/// C `ViewInfo` (`3dmod/imodP.h`), with data ownership represented by Rust
/// vectors.  Untranslated C++ service pointers are deliberately opaque.
#[repr(C)]
pub struct ImodView {
    pub idata: *mut *mut u8,
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub xysize: usize,
    pub full_xsize: i32,
    pub full_ysize: i32,
    pub full_zsize: i32,
    pub xmouse: f32,
    pub ymouse: f32,
    pub zmouse: f32,
    pub x_unbin_size: i32,
    pub y_unbin_size: i32,
    pub z_unbin_size: i32,
    pub num_times: i32,
    pub cur_time: i32,
    pub image: *mut ImodImageFile,
    pub image_list: *mut ImodImageFile,
    pub hdr: *mut ImodImageFile,
    pub model_view_vi: i32,
    pub vm_size: i32,
    pub strip_or_tile_cache: i32,
    pub loading_image: i32,
    pub doing_initial_load: i32,
    pub xybin: i32,
    pub zbin: i32,
    pub eer_super_res: i32,
    pub eer_zbinning: i32,
    pub rampbase: i32,
    pub rampsize: i32,
    pub black: i32,
    pub white: i32,
    pub range_low: i32,
    pub range_high: i32,
    pub white_in_range: i32,
    pub black_in_range: i32,
    pub xmovie: i32,
    pub ymovie: i32,
    pub zmovie: i32,
    pub tmovie: i32,
    pub movie_running: i32,
    /// Original `ViewInfo::movierate`, consumed by `moviecon.cpp`.
    pub movierate: i32,
    pub imod: *mut Imod,
    pub extra_obj: Vec<Iobj>,
    pub extra_obj_in_use: Vec<i32>,
    pub num_tilt_angles: i32,
    pub tilt_angles: Vec<f32>,
    pub dim: i32,
    pub obj_moveto: i32,
    pub ghostmode: i32,
    pub ghostlast: i32,
    pub ghostdist: i32,
    pub insertmode: i32,
    pub fastdraw: i32,
    pub drawcursor: i32,
    pub overlay_sec: i32,
    pub overlay_ramp: i32,
    pub which_green: i32,
    pub reverse_overlay: i32,
    pub draw_stipple: i32,
    pub track_mouse_for_plugs: i32,
    pub flippable: i32,
    pub fake_image: i16,
    pub raw_image_store: i16,
    pub ushort_store: i32,
    pub rgb_store: i32,
    pub volume_stack: i32,
    pub pixel_size_varies: i32,
}

impl Default for ImodView {
    fn default() -> Self {
        let mut view = Self {
            idata: ptr::null_mut(),
            xsize: 0,
            ysize: 0,
            zsize: 0,
            xysize: 0,
            full_xsize: 0,
            full_ysize: 0,
            full_zsize: 0,
            xmouse: 0.,
            ymouse: 0.,
            zmouse: 0.,
            x_unbin_size: 0,
            y_unbin_size: 0,
            z_unbin_size: 0,
            num_times: 0,
            cur_time: 0,
            image: ptr::null_mut(),
            image_list: ptr::null_mut(),
            hdr: ptr::null_mut(),
            model_view_vi: 0,
            vm_size: 0,
            strip_or_tile_cache: 0,
            loading_image: 0,
            doing_initial_load: 0,
            xybin: 1,
            zbin: 1,
            eer_super_res: 1,
            eer_zbinning: 10,
            rampbase: 0,
            rampsize: 256,
            black: 0,
            white: 255,
            range_low: 0,
            range_high: 65535,
            white_in_range: 255,
            black_in_range: 0,
            xmovie: 0,
            ymovie: 0,
            zmovie: 0,
            tmovie: 0,
            movie_running: 0,
            movierate: 0,
            imod: ptr::null_mut(),
            extra_obj: Vec::new(),
            extra_obj_in_use: Vec::new(),
            num_tilt_angles: 0,
            tilt_angles: Vec::new(),
            dim: 7,
            obj_moveto: 1,
            ghostmode: 32,
            ghostlast: 35,
            ghostdist: 0,
            insertmode: 0,
            fastdraw: 0,
            drawcursor: 1,
            overlay_sec: 0,
            overlay_ramp: -1,
            which_green: 0,
            reverse_overlay: 0,
            draw_stipple: 0,
            track_mouse_for_plugs: 0,
            flippable: 0,
            fake_image: 0,
            raw_image_store: 0,
            ushort_store: 0,
            rgb_store: 0,
            volume_stack: 0,
            pixel_size_varies: 0,
        };
        start_extra_object_if_none(&mut view);
        view
    }
}

/// `ivwInit` (`imodview.cpp:64`).
pub fn ivw_init(vi: &mut ImodView, modview: bool) {
    *vi = ImodView::default();
    vi.model_view_vi = i32::from(modview);
}

/// `ivwGetPixelBytes` (`imodview.cpp:521`).
pub fn ivw_get_pixel_bytes(mode: i32) -> i32 {
    match mode {
        1 | MRC_MODE_USHORT => 2,
        2 | 3 => 4,
        4 => 8,
        MRC_MODE_RGB => 3,
        _ => 1,
    }
}

/// `ivwAdjustedZIfVolStack` (`imodview.cpp:557`).
pub fn ivw_adjusted_z_if_vol_stack(vi: &ImodView, z_in: i32) -> i32 {
    if vi.volume_stack == 0 {
        z_in
    } else {
        z_in + (vi.cur_time - 1) * vi.zsize
    }
}

/// `ivwBindMouse` (`imodview.cpp:1705`).
pub fn ivw_bind_mouse(vi: &mut ImodView) {
    vi.xmouse = vi.xmouse.max(0.).min((vi.xsize - 1).max(0) as f32);
    vi.ymouse = vi.ymouse.max(0.).min((vi.ysize - 1).max(0) as f32);
    vi.zmouse = vi.zmouse.max(0.).min((vi.zsize - 1).max(0) as f32);
}

/// `ivwGetLocation` (`imodview.cpp:1722`).
pub fn ivw_get_location(vi: &ImodView, x: &mut i32, y: &mut i32, z: &mut i32) {
    *x = vi.xmouse as i32;
    *y = vi.ymouse as i32;
    *z = (vi.zmouse + 0.5).floor() as i32;
}

/// `ivwGetLocationPoint` (`imodview.cpp:1730`).
pub fn ivw_get_location_point(vi: &ImodView, point: &mut Ipoint) {
    *point = Ipoint {
        x: vi.xmouse,
        y: vi.ymouse,
        z: vi.zmouse,
    };
}

/// `ivwSetLocation` (`imodview.cpp:1824`).
pub fn ivw_set_location(vi: &mut ImodView, x: i32, y: i32, z: i32) {
    vi.xmouse = x as f32;
    vi.ymouse = y as f32;
    vi.zmouse = z as f32;
    ivw_bind_mouse(vi);
}

/// `ivwSetLocationPoint` (`imodview.cpp:1834`).
pub fn ivw_set_location_point(vi: &mut ImodView, point: &Ipoint) {
    vi.xmouse = point.x;
    vi.ymouse = point.y;
    vi.zmouse = point.z.round();
    ivw_bind_mouse(vi);
}

/// `ivwGetTime` (`imodview.cpp:1742`).
pub fn ivw_get_time(vi: &ImodView, time: Option<&mut i32>) -> i32 {
    if let Some(time) = time {
        *time = vi.cur_time;
    }
    vi.num_times
}

/// `ivwSetTime` (`imodview.cpp:1753`), excluding Qt dialog updates.
pub fn ivw_set_time(vi: &mut ImodView, time: i32) {
    vi.cur_time = if vi.num_times == 0 {
        0
    } else {
        time.clamp(1, vi.num_times)
    };
    if vi.fake_image == 0 && !vi.image_list.is_null() && vi.cur_time > 0 {
        unsafe {
            vi.image = vi.image_list.add((vi.cur_time - 1) as usize);
            vi.hdr = vi.image;
        }
    }
}

pub fn ivw_get_max_time(vi: &ImodView) -> i32 {
    vi.num_times
}

/// `ivwReadAngleFile` (`imodview.cpp:3455`).
pub unsafe fn ivw_read_angle_file(vi: *mut ImodView, filename: *const c_char) -> i32 {
    if vi.is_null() || filename.is_null() {
        return 1;
    }
    let Ok(contents) = std::fs::read_to_string(CStr::from_ptr(filename).to_string_lossy().as_ref())
    else {
        return 1;
    };
    let mut angles = Vec::new();
    for field in contents.split_whitespace() {
        match field.parse::<f32>() {
            Ok(value) => angles.push(value),
            Err(_) => break,
        }
    }
    (*vi).num_tilt_angles = angles.len() as i32;
    (*vi).tilt_angles = angles;
    0
}

/// `ivwGetTiltAngles` (`imodview.cpp:3488`).  The translated `ImodView`
/// does not yet own `IloadInfo::zmin`, so its loaded angle vector starts at 0.
pub fn ivw_get_tilt_angles(vi: &mut ImodView, number: &mut i32) -> *mut f32 {
    *number = vi.num_tilt_angles.max(0);
    if *number == 0 {
        ptr::null_mut()
    } else {
        vi.tilt_angles.as_mut_ptr()
    }
}

pub unsafe fn ivw_get_time_index_label(vi: *const ImodView, index: i32) -> *const c_char {
    if vi.is_null()
        || index < 1
        || index > (*vi).num_times
        || (*vi).fake_image != 0
        || (*vi).image_list.is_null()
    {
        return c"".as_ptr();
    }
    (*(*vi).image_list.add((index - 1) as usize)).description
}

pub unsafe fn ivw_get_time_label(vi: *const ImodView) -> *const c_char {
    if vi.is_null() || (*vi).image.is_null() {
        c"".as_ptr()
    } else {
        (*(*vi).image).description
    }
}

/// `ivwSetNewContourTime` (`imodview.cpp:1815`).
pub fn ivw_set_new_contour_time(vi: &ImodView, object: &Iobj, contour: &mut Icont) {
    if vi.num_times != 0 && object.flags & (1 << 18) != 0 {
        contour.time = vi.cur_time;
    }
}

/// `ivwGetImageSize` (`imodview.cpp:3578`).
pub fn ivw_get_image_size(vi: &ImodView, x: &mut i32, y: &mut i32, z: &mut i32) {
    *x = vi.xsize;
    *y = vi.ysize;
    *z = vi.zsize;
}
pub fn ivw_get_image_store_mode(vi: &ImodView) -> i32 {
    vi.raw_image_store as i32
}
pub fn ivw_data_in_tile_or_strip_cache(vi: &ImodView) -> bool {
    vi.strip_or_tile_cache != 0
}

/// `ivwGetZSection` (`imodview.cpp:269`).
pub unsafe fn ivw_get_z_section(vi: *mut ImodView, section: i32) -> *mut *mut u8 {
    if vi.is_null() || section < 0 || section >= (*vi).zsize || (*vi).vm_size != 0 {
        ptr::null_mut()
    } else {
        (*vi).idata.add(section as usize)
    }
}
pub unsafe fn ivw_get_current_z_section(vi: *mut ImodView) -> *mut *mut u8 {
    if vi.is_null() {
        ptr::null_mut()
    } else {
        ivw_get_z_section(vi, ((*vi).zmouse + 0.5).floor() as i32)
    }
}
pub unsafe fn ivw_get_z_section_time(vi: *mut ImodView, section: i32, _time: i32) -> *mut *mut u8 {
    ivw_get_z_section(vi, section)
}
pub unsafe fn ivw_get_tile_cached_section(_vi: *mut ImodView, _section: i32) -> *mut *mut u8 {
    ptr::null_mut()
}
pub fn ivw_free_tile_cached_section(_vi: &mut ImodView) {}

/// `ivwGetValue` (`imodview.cpp:942`) for in-memory byte data.
pub unsafe fn ivw_get_value(vi: *const ImodView, x: i32, y: i32, z: i32) -> i32 {
    if vi.is_null()
        || x < 0
        || y < 0
        || z < 0
        || x >= (*vi).xsize
        || y >= (*vi).ysize
        || z >= (*vi).zsize
        || (*vi).idata.is_null()
    {
        return 0;
    }
    let row = *(*vi).idata.add(z as usize);
    if row.is_null() {
        0
    } else {
        *row.add((y * (*vi).xsize + x) as usize) as i32
    }
}
pub unsafe fn ivw_get_file_value(vi: *const ImodView, x: i32, y: i32, z: i32) -> f32 {
    ivw_get_value(vi, x, y, z) as f32
}

/// `ivwCopyImageToByteBuffer` (`imodview.cpp:2078`).
pub unsafe fn ivw_copy_image_to_byte_buffer(
    vi: *const ImodView,
    image: *mut *mut u8,
    buffer: *mut u8,
) -> i32 {
    if vi.is_null() || image.is_null() || buffer.is_null() {
        return 1;
    }
    let pixels = ((*vi).xsize.max(0) as usize) * ((*vi).ysize.max(0) as usize);
    if (*vi).ushort_store != 0 {
        return 1;
    }
    ptr::copy_nonoverlapping(*image, buffer, pixels);
    0
}
pub fn ivw_gray_scale_image_loaded(vi: &ImodView) -> bool {
    !vi.image.is_null() && vi.rgb_store == 0
}

pub unsafe fn ivw_get_model(vi: *const ImodView) -> *mut Imod {
    if vi.is_null() {
        ptr::null_mut()
    } else {
        (*vi).imod
    }
}

/// `startExtraObjectIfNone` (`imodview.cpp:3622`).
pub fn start_extra_object_if_none(vi: &mut ImodView) {
    if vi.extra_obj.is_empty() {
        vi.extra_obj.push(Iobj::default());
        vi.extra_obj_in_use.push(1);
    }
}
pub fn ivw_get_free_extra_object_number(vi: &mut ImodView) -> i32 {
    for index in 1..vi.extra_obj_in_use.len() {
        if vi.extra_obj_in_use[index] == 0 {
            vi.extra_obj_in_use[index] = 1;
            vi.extra_obj[index] = Iobj::default();
            return index as i32;
        }
    }
    start_extra_object_if_none(vi);
    vi.extra_obj.push(Iobj::default());
    vi.extra_obj_in_use.push(1);
    (vi.extra_obj.len() - 1) as i32
}
pub fn ivw_free_extra_object(vi: &mut ImodView, object_number: i32) -> i32 {
    if object_number < 1
        || object_number as usize >= vi.extra_obj.len()
        || vi.extra_obj_in_use[object_number as usize] == 0
    {
        return 1;
    }
    ivw_clear_an_extra_object(vi, object_number);
    vi.extra_obj_in_use[object_number as usize] = 0;
    0
}
pub fn ivw_get_an_extra_object(vi: &mut ImodView, object_number: i32) -> Option<&mut Iobj> {
    if object_number < 0
        || object_number as usize >= vi.extra_obj.len()
        || vi.extra_obj_in_use[object_number as usize] == 0
    {
        None
    } else {
        Some(&mut vi.extra_obj[object_number as usize])
    }
}
pub fn ivw_get_extra_object(vi: &mut ImodView) -> Option<&mut Iobj> {
    ivw_get_an_extra_object(vi, 0)
}
pub fn ivw_clear_extra_object(vi: &mut ImodView) {
    ivw_clear_an_extra_object(vi, 0);
}
pub fn ivw_clear_an_extra_object(vi: &mut ImodView, object_number: i32) {
    if let Some(object) = ivw_get_an_extra_object(vi, object_number) {
        object.cont.clear();
        object.mesh.clear();
        object.store.clear();
    }
}

pub fn ivw_get_movie_model_mode(vi: Option<&ImodView>) -> i32 {
    if vi.is_none()
        || vi.unwrap().imod.is_null()
        || unsafe { (*vi.unwrap().imod).mousemode } == IMOD_MMOVIE
    {
        0
    } else {
        1
    }
}
pub fn ivw_set_movie_model_mode(_vi: &mut ImodView, _mode: i32) { /* model mode UI is in moviecon.cpp */
}
pub fn ivw_enable_stipple(vi: &mut ImodView, enable: i32) {
    vi.draw_stipple = enable;
}
pub fn ivw_track_mouse_for_plugs(vi: &mut ImodView, enable: i32) {
    vi.track_mouse_for_plugs = enable;
}
pub fn ivw_get_ramp(vi: &ImodView, base: &mut i32, size: &mut i32) {
    *base = vi.rampbase;
    *size = vi.rampsize;
}
pub fn ivw_get_contrast_reversed(vi: &ImodView) -> i32 {
    vi.reverse_overlay
}
pub fn ivw_set_overlay_mode(vi: &mut ImodView, section: i32, reverse: i32, green: i32) {
    vi.overlay_sec = section;
    vi.reverse_overlay = reverse;
    vi.which_green = green;
}

/// `ivwBinByN` (`imodview.cpp:4200`).
pub fn ivw_bin_by_n(array: &[u8], nxin: i32, nyin: i32, nbin: i32, brray: &mut [u8]) {
    if nxin <= 0 || nyin <= 0 || nbin <= 0 {
        return;
    }
    let nxout = nxin / nbin;
    let nyout = nyin / nbin;
    let ixofs = (nxin % nbin) / 2;
    let iyofs = (nyin % nbin) / 2;
    for iy in 0..nyout {
        for ix in 0..nxout {
            let mut sum = 0i32;
            for jy in 0..nbin {
                for jx in 0..nbin {
                    let source =
                        ((nbin * iy + iyofs + jy) * nxin + nbin * ix + ixofs + jx) as usize;
                    if source < array.len() {
                        sum += array[source] as i32;
                    }
                }
            }
            let target = (iy * nxout + ix) as usize;
            if target < brray.len() {
                brray[target] = (sum / (nbin * nbin)) as u8;
            }
        }
    }
}

/// `ivwFixUnderSizeCoords` (`imodview.cpp:812`).
pub fn ivw_fix_under_size_coords(
    size: i32,
    nx: i32,
    llx: &mut i32,
    urx: &mut i32,
    offset: &mut i32,
    left_pad: &mut i32,
    right_pad: &mut i32,
) -> bool {
    *offset = (size - nx) / 2;
    *left_pad = 0;
    *right_pad = 0;
    *llx -= *offset;
    *urx -= *offset;
    if *llx < 0 {
        *left_pad = -*llx;
        *llx = 0;
    }
    if *urx >= nx {
        *right_pad = *urx + 1 - nx;
        *urx = nx - 1;
    }
    *urx < 0 || *llx >= nx
}

// These APIs require translated Zap/Slicer/OpenGL window units.  In upstream they return 1
// when no applicable window exists; preserve that result until those units arrive.
pub fn ivw_get_top_zap_zslice(_vi: &ImodView, _z: &mut i32) -> i32 {
    1
}
pub fn ivw_set_top_zap_zslice(_vi: &mut ImodView, _z: i32) -> i32 {
    1
}
pub fn ivw_get_top_zap_zoom(_vi: &ImodView, _zoom: &mut f32) -> i32 {
    1
}
pub fn ivw_set_top_zap_zoom(_vi: &mut ImodView, _zoom: f32, _draw: bool) -> i32 {
    1
}
pub fn ivw_get_top_slicer_zoom(_vi: &ImodView, _zoom: &mut f32) -> i32 {
    1
}
pub fn ivw_get_top_slicer_thickness(_vi: &ImodView, _thickness: &mut i32) -> i32 {
    1
}
pub fn ivw_get_top_zap_mouse(_vi: &ImodView, _point: &mut Ipoint) -> i32 {
    1
}
pub fn ivw_get_top_zap_center(_vi: &ImodView, _x: &mut f32, _y: &mut f32, _z: &mut i32) -> i32 {
    1
}
pub fn ivw_set_top_zap_center(_vi: &mut ImodView, _x: f32, _y: f32, _z: i32, _draw: bool) -> i32 {
    1
}
pub fn ivw_get_top_zap_dev_pixel_ratio(_vi: &ImodView) -> f32 {
    1.
}
pub fn ivw_snapshot_top_zap(_name: &mut String, _format: i32, _check: bool, _full: bool) -> i32 {
    -1
}
pub fn ivw_snapshot_top_slicer(_name: &mut String, _format: i32, _check: bool, _full: bool) -> i32 {
    -1
}
pub fn start_added_arrow(_window_type: i32) -> i32 {
    -1
}
pub fn clear_all_arrows(_window_type: i32, _all_windows: bool) -> i32 {
    -1
}
pub fn ivw_draw(_vi: &mut ImodView, _flags: i32) -> i32 {
    0
}
pub fn ivw_redraw(_vi: &mut ImodView) -> i32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bin_by_three_preserves_upstream_centered_window() {
        let input: Vec<u8> = (0..36).collect();
        let mut output = [0; 4];
        ivw_bin_by_n(&input, 6, 6, 3, &mut output);
        assert_eq!(output, [7, 10, 25, 28]);
    }
    #[test]
    fn location_is_bounded() {
        let mut view = ImodView {
            xsize: 4,
            ysize: 3,
            zsize: 2,
            ..Default::default()
        };
        ivw_set_location(&mut view, 99, -3, 8);
        assert_eq!((view.xmouse, view.ymouse, view.zmouse), (3., 0., 1.));
    }

    #[test]
    fn undersize_coordinates_match_source_centering() {
        let (mut llx, mut urx, mut offset, mut left, mut right) = (0, 7, 0, 0, 0);
        assert!(!ivw_fix_under_size_coords(
            10,
            8,
            &mut llx,
            &mut urx,
            &mut offset,
            &mut left,
            &mut right
        ));
        assert_eq!((llx, urx, offset, left, right), (0, 6, 1, 1, 0));
    }
}
