//! Translation of `IMOD/3dmod/utilities.cpp` and `utilities.h`.
//!
//! The original is a deliberately mixed utility unit.  Its model-independent
//! calculations are implemented below; painting, Qt widgets, GLU tessellation,
//! file dialogs, and application-global services are represented by the explicit
//! [`UtilitiesBoundary`] instead of silently replacing their behaviour.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::icont::ICONT_STIPPLED;
use crate::imod::libimod::imodel::{IMODF_FLIPYZ, IMODF_ROT90X, Icont, Imod, Iobj};
use crate::imod::libimod::iobj::{
    IOBJ_SYM_CIRCLE, IOBJ_SYM_NONE, IOBJ_SYM_SQUARE, IOBJ_SYM_STAR, IOBJ_SYM_TRIANGLE,
    IOBJ_SYMF_FILL, iobj_scat,
};
use crate::imod::three_dmod::imodview::ImodView;

pub const TB_AUTO_RAISE: bool = true;
pub const FLIP_TO_ROTATION: i32 = 0;
pub const ROTATION_TO_FLIP: i32 = 1;

/// `PopupEntry` in `utilities.h`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PopupEntry {
    pub text: &'static str,
    pub key: i32,
    pub ctrl: bool,
    pub shift: bool,
    pub main_index: i16,
}

/// The non-Rust services called by this source unit.
pub trait UtilitiesBoundary {
    fn draw_symbol(&mut self, x: i32, y: i32, symbol: i32, size: i32, filled: bool);
    fn set_stipple(&mut self, enabled: bool);
    fn clear_window(&mut self, color_index: i32);
    fn redraw_model(&mut self);
    fn change_point_size(&mut self);
    fn finish_undo_unit(&mut self);
    fn message(&mut self, text: &str);
    fn flip_yz(&mut self, imod: &mut Imod);
    fn rotate_90_x(&mut self, imod: &mut Imod, inverse: bool);
    fn draw_filled_polygon(&mut self, points: &[crate::imod::libimod::imodel::Ipoint]);
}

/// `utilDrawSymbol`.
pub fn util_draw_symbol(
    boundary: &mut dyn UtilitiesBoundary,
    mx: i32,
    my: i32,
    sym: i32,
    size: i32,
    flags: u32,
) {
    match sym {
        IOBJ_SYM_CIRCLE | IOBJ_SYM_SQUARE | IOBJ_SYM_TRIANGLE => {
            boundary.draw_symbol(mx, my, sym, size, flags & IOBJ_SYMF_FILL != 0)
        }
        IOBJ_SYM_STAR => {}
        IOBJ_SYM_NONE => boundary.draw_symbol(mx, my, sym, 1, true),
        _ => {}
    }
}

/// `utilGetLongestTimeString`; `time_labels` is `ivwGetTimeIndexLabel`.
pub fn util_get_longest_time_string(num_times: i32, time_labels: &[String]) -> String {
    if num_times == 0 {
        return String::new();
    }
    let mut out = " (999)".to_owned();
    let mut max_len = -1_i32;
    let mut tmax = 0usize;
    for time in 1..num_times as usize {
        if let Some(label) = time_labels.get(time) {
            if label.len() as i32 > max_len {
                max_len = label.len() as i32;
                tmax = time;
            }
        }
    }
    if let Some(label) = time_labels.get(tmax) {
        out.push_str(label);
    }
    out
}

/// `utilShrinkFilenameToFit`; `width_of` is QWidget's font metric call.
pub fn util_shrink_filename_to_fit(label: &mut String, width: i32, width_of: impl Fn(&str) -> i32) {
    let len = label.chars().count();
    if width_of(label) > width - 4 {
        for rem in 0..len / 2 - 3 {
            let keep = len / 2 - rem;
            let text = format!(
                "{}...{}",
                label.chars().take(keep).collect::<String>(),
                label.chars().skip(len - keep).collect::<String>()
            );
            if width_of(&text) <= width - 4 {
                *label = text;
                return;
            }
        }
    }
}

/// `utilCurrentPointSize`.
pub fn util_current_point_size(
    obj: Option<&Iobj>,
    min_mod_size: i32,
    min_im_size: i32,
    xybin: i32,
) -> (i32, i32, i32) {
    let (mut mod_size, mut backup_size, mut im_size) =
        (min_mod_size, min_mod_size + 2, min_im_size);
    let Some(obj) = obj else {
        return (mod_size, backup_size, im_size);
    };
    let mut sym_size = if obj.symbol as i32 != IOBJ_SYM_NONE && obj.symsize > 0 {
        obj.symsize as i32
    } else {
        0
    };
    if sym_size == 0 && obj.pdrawsize > 0 {
        sym_size = obj.pdrawsize / xybin;
    }
    if (sym_size - mod_size).abs() < 2 {
        mod_size = sym_size + 2;
    }
    backup_size = mod_size + 2;
    if (sym_size - backup_size).abs() < 2 {
        backup_size = sym_size + 2;
    }
    if (sym_size - im_size).abs() < 2 {
        im_size = sym_size + 2;
    }
    (mod_size, backup_size, im_size)
}

/// `utilEnableStipple`.
pub fn util_enable_stipple(
    boundary: &mut dyn UtilitiesBoundary,
    draw_stipple: i32,
    cont: &Icont,
) -> bool {
    let enabled = draw_stipple != 0 && cont.flags & ICONT_STIPPLED != 0;
    if enabled {
        boundary.set_stipple(true);
    }
    enabled
}
/// `utilDisableStipple`.
pub fn util_disable_stipple(boundary: &mut dyn UtilitiesBoundary, draw_stipple: i32, cont: &Icont) {
    if draw_stipple != 0 && cont.flags & ICONT_STIPPLED != 0 {
        boundary.set_stipple(false);
    }
}
/// `utilCloseKey` (`utilities.cpp:502`); the `Q_OS_MACX` Ctrl-W arm is not
/// compiled on this platform, so only Escape closes.  The Qt key code travels
/// as the plain `int` the rest of the translated input path uses.
pub fn util_close_key(key: i32) -> bool {
    key == 0x0100_0000
}

/// `utilRaiseIfNeeded`.  The source body is compiled only for the legacy
/// macOS Qt path; the Linux winit host has no corresponding raise action.
pub fn util_raise_if_needed() {}

/// `utilNeedToSetCursor`.  This is true only for the legacy macOS Qt path.
pub fn util_need_to_set_cursor() -> bool {
    false
}

/// `utilIgnoreClosing`.  Its warning and ignored close event exist only for
/// the bounded macOS Qt 5.12--5.14 build configuration.
pub fn util_ignore_closing(_closing: bool) -> bool {
    false
}

/// `utilClearWindow`.
pub fn util_clear_window(boundary: &mut dyn UtilitiesBoundary, index: i32) {
    boundary.clear_window(index)
}

/// `utilMouseZaxisRotation`.
pub fn util_mouse_zaxis_rotation(
    winx: i32,
    mx: i32,
    lastmx: i32,
    winy: i32,
    my: i32,
    lastmy: i32,
) -> f32 {
    let (xcen, ycen) = (winx / 2, winy / 2);
    let (mut dx, mut dy) = ((lastmx - xcen) as f64, (winy - 1 - lastmy - ycen) as f64);
    if dx.abs() <= 20. && dy.abs() <= 20. {
        return 0.;
    }
    let start = dy.atan2(dx).to_degrees();
    dx = (mx - xcen) as f64;
    dy = (winy - 1 - my - ycen) as f64;
    if dx.abs() <= 20. && dy.abs() <= 20. {
        return 0.;
    }
    let mut rotation = dy.atan2(dx).to_degrees() - start;
    if rotation < -360. {
        rotation += 360.;
    }
    if rotation > 360. {
        rotation -= 360.;
    }
    rotation as f32
}

/// `utilSetObjFlag`.
pub fn util_set_obj_flag(obj: Option<&mut Iobj>, flag_type: i32, state: bool, flag: u32) {
    let Some(obj) = obj else { return };
    if !(0..=1).contains(&flag_type) {
        return;
    }
    let flags = if flag_type == 1 {
        &mut obj.symflags
    } else {
        return if state {
            obj.flags |= flag
        } else {
            obj.flags &= !flag
        };
    };
    if state {
        *flags |= flag as u8
    } else {
        *flags &= !(flag as u8)
    }
}

/// `utilNextSecWithCont`.
pub fn util_next_sec_with_cont(view: &ImodView, obj: Option<&Iobj>, curz: i32, dir: i32) -> i32 {
    let Some(obj) = obj else { return curz };
    let mut newz = -1;
    for cont in &obj.cont {
        if cont.pts.is_empty()
            || (obj.flags & crate::imod::libimod::iobj::IMOD_OBJFLAG_TIME != 0
                && view.num_times > 1
                && cont.time != view.cur_time)
        {
            continue;
        }
        let lim = if iobj_scat(obj.flags) != 0 {
            cont.pts.len()
        } else {
            1
        };
        for point in cont.pts.iter().take(lim) {
            let z = (point.z.round() as i32).clamp(0, view.zsize - 1);
            let diff = dir * (z - curz);
            if diff > 0 && (newz < 0 || diff < dir * (newz - curz)) {
                newz = z;
            }
        }
    }
    if newz >= 0 { newz } else { curz }
}

/// `utilWheelToPointSizeScaling`.
pub fn util_wheel_to_point_size_scaling(zoom: f32) -> f32 {
    let mut scale = 1. / 1200.;
    if zoom < 4. && zoom >= 2. {
        scale *= 2.;
    } else if zoom < 2. && zoom > 1. {
        scale *= 3.;
    } else if zoom == 1. {
        scale *= 4.;
    } else if zoom < 1. {
        scale *= 5.;
    }
    scale
}

/// `utilWheelChangePointSize`, with selected point supplied directly by input code.
pub fn util_wheel_change_point_size(
    boundary: &mut dyn UtilitiesBoundary,
    cont: &mut Icont,
    point: usize,
    zoom: f32,
    delta: i32,
) {
    if point >= cont.pts.len() {
        return;
    }
    let size = cont.sizes.get(point).copied().unwrap_or(-1.);
    if size == 0. || size < 0. {
        return;
    }
    if cont.sizes.len() <= point {
        cont.sizes.resize(point + 1, -1.);
    }
    cont.sizes[point] = (size + delta as f32 * util_wheel_to_point_size_scaling(zoom)).max(0.);
    boundary.change_point_size();
    boundary.finish_undo_unit();
    boundary.redraw_model();
}

/// `utilIsBandCommitted`.
pub fn util_is_band_committed(
    x: i32,
    y: i32,
    win_x: i32,
    win_y: i32,
    bandmin: i32,
    rb: &mut [i32; 4],
    dragging: &mut [i32; 4],
) -> i32 {
    let (mut x, mut y) = (x, y);
    let (mut dx, mut dy) = (x - rb[0], y - rb[2]);
    let (mut ax, mut ay) = (dx.abs(), dy.abs());
    if dy == 0 && ax >= 6 * bandmin {
        y = rb[2] + if y < win_y / 2 { bandmin } else { -bandmin };
        dy = y - rb[2];
        ay = bandmin;
    } else if dx == 0 && ay >= 6 * bandmin {
        x = rb[0] + if x < win_x / 2 { bandmin } else { -bandmin };
        dx = x - rb[0];
        ax = bandmin;
    }
    if !((ax >= bandmin && ay >= bandmin)
        || (ax >= 3 * bandmin && ay >= (bandmin / 2).max(1))
        || (ay >= 3 * bandmin && ax >= (bandmin / 2).max(1)))
    {
        return 0;
    }
    *dragging = [0; 4];
    if x > rb[0] {
        dragging[1] = 1;
        rb[1] = x
    } else {
        dragging[0] = 1;
        rb[1] = rb[0];
        rb[0] = x;
    }
    if y > rb[2] {
        dragging[3] = 1;
        rb[3] = y
    } else {
        dragging[2] = 1;
        rb[3] = rb[2];
        rb[2] = y;
    }
    1
}

/// `utilAnalyzeBandEdge`.
pub fn util_analyze_band_edge(
    ix: i32,
    iy: i32,
    rb: [i32; 4],
    drag_band: &mut i32,
    dragging: &mut [i32; 4],
) {
    let (x0, x1, y0, y1) = (rb[0], rb[1], rb[2], rb[3]);
    *drag_band = 0;
    *dragging = [0; 4];
    let mut best = 100;
    let mut edge = (-1, -1);
    // Preserve the four source tests' order.  These are `<`, not `<=`, so
    // a cursor exactly equidistant from two corners retains the earlier one.
    for (xedge, yedge, xe, ye) in [
        (x0, y0, 0, 2),
        (x1, y0, 1, 2),
        (x0, y1, 0, 3),
        (x1, y1, 1, 3),
    ] {
        let d = (ix - xedge).pow(2) + (iy - yedge).pow(2);
        if d < best {
            best = d;
            edge = (xe, ye)
        }
    }
    if edge.0 >= 0 {
        *drag_band = 1;
        dragging[edge.0 as usize] = 1;
        dragging[edge.1 as usize] = 1;
        return;
    }
    let mut min = 10;
    let mut found = -1;
    if iy > y0 && iy < y1 && (ix - x0).abs() < min {
        min = (ix - x0).abs();
        found = 0
    };
    if iy > y0 && iy < y1 && (ix - x1).abs() < min {
        min = (ix - x1).abs();
        found = 1
    };
    if ix > x0 && ix < x1 && (iy - y0).abs() < min {
        min = (iy - y0).abs();
        found = 2
    };
    if ix > x0 && ix < x1 && (iy - y1).abs() < min {
        found = 3
    };
    if found >= 0 {
        *drag_band = 1;
        dragging[found as usize] = 1
    }
}

/// `utilTestBandMove`.
pub fn util_test_band_move(x: i32, y: i32, rb: [i32; 4]) -> i32 {
    let (dx0, dx1, dy0, dy1) = (x - rb[0], x - rb[1], y - rb[2], y - rb[3]);
    (((dy0 > 0 && dy1 < 0) && ((dx0).abs() < 10 || (dx1).abs() < 10))
        || ((dx0 > 0 && dx1 < 0) && ((dy0).abs() < 10 || (dy1).abs() < 10))) as i32
}

/// `utilExchangeFlipRotation`.
pub fn util_exchange_flip_rotation(
    boundary: &mut dyn UtilitiesBoundary,
    imod: &mut Imod,
    direction: i32,
) {
    if (direction == FLIP_TO_ROTATION && imod.flags & IMODF_FLIPYZ == 0)
        || (direction == ROTATION_TO_FLIP && imod.flags & IMODF_ROT90X == 0)
    {
        return;
    }
    if direction == FLIP_TO_ROTATION {
        boundary.flip_yz(imod);
        boundary.rotate_90_x(imod, false);
        imod.flags |= IMODF_ROT90X;
        imod.flags &= !IMODF_FLIPYZ;
    } else {
        boundary.rotate_90_x(imod, true);
        boundary.flip_yz(imod);
        imod.flags &= !IMODF_ROT90X;
        imod.flags |= IMODF_FLIPYZ;
    }
}

/// `utilLookupPopupHit`.
pub fn util_lookup_popup_hit(
    index: usize,
    specific: &[PopupEntry],
    num_specific: isize,
) -> Option<(i32, bool, bool)> {
    let count = if num_specific < 0 {
        specific.len()
    } else {
        num_specific as usize
    };
    let entry = if index < count {
        specific.get(index)?
    } else {
        DEFAULT_ACTIONS.get(index - count)?
    };
    Some((entry.key, entry.ctrl, entry.shift))
}

/// `utilUnitZoomForDeviceScaling`.
pub fn util_unit_zoom_for_device_scaling(dev_pix_ratio: f32) -> f32 {
    if dev_pix_ratio >= 1.8 {
        2.
    } else if dev_pix_ratio >= 1.3 {
        1.5
    } else {
        1.
    }
}

/// `utilInitializeScreenChange`.  The source uses `App->DevicePixelRatio`
/// unless its Qt-only `WATCH_DPI_CHANGE` build path is selected; callers with
/// a winit window install the screen callback at their native host instead.
pub fn util_initialize_screen_change(app_device_pixel_ratio: f32) -> f32 {
    app_device_pixel_ratio
}

/// `utilGetNewDevPixRatio`.  `WATCH_DPI_CHANGE` is not enabled for this
/// platform; source therefore returns zero and leaves the current ratio alone.
pub fn util_get_new_dev_pix_ratio() -> f32 {
    0.
}

/// `utilSetZoomOnScreenChange`, with elapsed timer values supplied by the caller.
pub fn util_set_zoom_on_screen_change(
    new_size_change: f32,
    last_size_change: &mut f32,
    screen_changed: &mut bool,
    screen_elapsed: i32,
    resize_elapsed: i32,
    zoom: &mut f32,
    dev_pix_varies: bool,
) -> bool {
    let is_screen = new_size_change == 0.;
    let tiny = new_size_change > 0. && new_size_change > 0.99 && new_size_change < 1.01;
    if !dev_pix_varies {
        return false;
    };
    if !is_screen && !(tiny && resize_elapsed < 2000) {
        *last_size_change = new_size_change
    }
    if (*screen_changed && !is_screen && screen_elapsed < 2000)
        || (!*screen_changed && is_screen && resize_elapsed < 2000)
    {
        if is_screen {
            *screen_changed = true
        };
        if *last_size_change < 0.9 || *last_size_change > 1.1 {
            *zoom *= *last_size_change;
            let zlog = zoom.log10();
            let power = zlog.floor();
            *zoom = 10_f32.powf(power - 1.) * (10_f32.powf(zlog - power) * 10.).round();
            *screen_changed = false;
            *last_size_change = 1.;
            return true;
        }
    } else {
        *screen_changed = is_screen
    };
    false
}

/// `imodwGivenName`.
pub fn imodw_given_name(intro: &str, filein: Option<&str>) -> Option<String> {
    let name = filein?.rsplit('/').next()?;
    if name.is_empty() {
        None
    } else {
        Some(format!(
            "{}{}{}",
            intro,
            if intro.is_empty() { "" } else { " " },
            name
        ))
    }
}
/// `imodwEithername`.
pub fn imodw_either_name(
    intro: &str,
    filein: Option<&str>,
    image_name: Option<&str>,
    model_first: bool,
) -> Option<String> {
    if model_first {
        imodw_given_name(intro, filein).or_else(|| imodw_given_name(intro, image_name))
    } else {
        imodw_given_name(intro, image_name).or_else(|| imodw_given_name(intro, filein))
    }
}
/// `imodCaption`.
pub fn imod_caption(intro: &str, image_name: Option<&str>) -> String {
    let prefix = if intro.is_empty() {
        String::new()
    } else {
        format!("{intro}:")
    };
    imodw_given_name(&prefix, image_name).unwrap_or_else(|| intro.into())
}
/// `utilManageBrowserDir`.
pub fn util_manage_browser_dir(browser_dir: &mut String, filename: &str, only_if_empty: bool) {
    if filename.is_empty() || (only_if_empty && !browser_dir.is_empty()) {
        return;
    }
    if let Some((dir, _)) = filename.rsplit_once('/') {
        *browser_dir = dir.to_owned()
    }
}

/// Qt's popup menu is intentionally a boundary.  These are the source's default rows
/// required by menu units; the full specialized table stays owned by its caller.
pub static DEFAULT_ACTIONS: &[PopupEntry] = &[
    PopupEntry {
        text: "Go to previous object",
        key: 'O' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Go to next object",
        key: 'P' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Go to previous contour",
        key: 'C' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
    PopupEntry {
        text: "Go to next contour",
        key: 'C' as i32,
        ctrl: false,
        shift: true,
        main_index: 0,
    },
    PopupEntry {
        text: "Save model to file",
        key: 'S' as i32,
        ctrl: false,
        shift: false,
        main_index: 0,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_mouse_rotation_and_band() {
        assert_eq!(util_mouse_zaxis_rotation(200, 100, 100, 200, 50, 50), 0.);
        let mut rb = [10, 0, 10, 0];
        let mut drag = [0; 4];
        assert_eq!(
            util_is_band_committed(30, 40, 100, 100, 5, &mut rb, &mut drag),
            1
        );
        assert_eq!((rb, drag), ([10, 30, 10, 40], [0, 1, 0, 1]));
    }
    #[test]
    fn source_title_and_zoom() {
        assert_eq!(
            imodw_given_name("3dmod:", Some("a/b.mrc")).as_deref(),
            Some("3dmod: b.mrc")
        );
        assert_eq!(util_unit_zoom_for_device_scaling(1.3), 1.5);
    }
    #[test]
    fn source_longest_time_label() {
        assert_eq!(
            util_get_longest_time_string(3, &["".into(), "1".into(), "long".into()]),
            " (999)long"
        );
    }

    #[test]
    fn platform_conditional_cursor_and_dpi_utilities_use_the_linux_source_path() {
        assert!(!util_need_to_set_cursor());
        assert!(!util_ignore_closing(false));
        assert_eq!(util_initialize_screen_change(1.5), 1.5);
        assert_eq!(util_get_new_dev_pix_ratio(), 0.);
    }

    #[test]
    fn band_corner_tie_keeps_the_source_first_corner() {
        let mut drag_band = 0;
        let mut dragging = [0; 4];
        // Equidistant from lower-left and lower-right.  C checks lower-left
        // first and strict comparison retains it.
        util_analyze_band_edge(5, 0, [0, 10, 0, 10], &mut drag_band, &mut dragging);
        assert_eq!(drag_band, 1);
        assert_eq!(dragging, [1, 0, 1, 0]);
    }
}
