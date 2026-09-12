//! Translation of `IMOD/3dmod/display.cpp` and `display.h`.
//!
//! `display.cpp` is where 3dmod selects an OpenGL visual, establishes its
//! indexed/RGBA palette, and fans an `imodDraw` request out to every viewer
//! control.  Qt cursor/image objects and the active OpenGL context are kept
//! behind [`DisplayNativeBoundary`]; the source-owned selection and colour
//! policy is retained here.

#![allow(dead_code)]

use crate::imod::libimod::imodel::{Imod, Iobj};
use crate::imod::libimod::iobj::iobj_flag_time;
use crate::imod::three_dmod::control::ivw_control_list_draw;
use crate::imod::three_dmod::imod::{
    IMOD_DRAW_COLORMAP, IMOD_DRAW_IMAGE, IMOD_DRAW_MOD, IMOD_DRAW_RETHINK, IMOD_DRAW_SKIPMODV,
    IMOD_DRAW_XYZ,
};
use crate::imod::three_dmod::imodview::{ImodView, ivw_bind_mouse, ivw_get_location, ivw_set_time};
use crate::imod::three_dmod::preferences::ImodPreferences;

pub const RAMP_MIN: i32 = 16;
pub const RAMP_MAX: i32 = 255;
pub const IMOD_MIN_INDEX: i32 = 1;
const MAX_VISUALS: usize = 32;

/// `ImodGLVisual` (`display.h`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImodGlVisual {
    pub double_buffer: i32,
    pub rgba: i32,
    pub color_bits: i32,
    pub depth_bits: i32,
    pub stereo: i32,
    pub valid_direct: i32,
    pub db_requested: i32,
    pub rgba_requested: i32,
    pub depth_enabled: i32,
    pub alpha: i32,
}

/// `ImodGLRequest` (`display.h`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImodGlRequest {
    pub double_buffer: i32,
    pub rgba: i32,
    pub color_bits: i32,
    pub depth_bits: i32,
    pub stereo: i32,
    pub alpha: i32,
}

const QT_GL_REQUEST_LIST: [ImodGlRequest; 8] = [
    ImodGlRequest {
        double_buffer: 1,
        rgba: 1,
        color_bits: 24,
        depth_bits: 0,
        stereo: 0,
        alpha: 0,
    },
    ImodGlRequest {
        double_buffer: 0,
        rgba: 1,
        color_bits: 24,
        depth_bits: 0,
        stereo: 0,
        alpha: 0,
    },
    ImodGlRequest {
        double_buffer: 1,
        rgba: 0,
        color_bits: 12,
        depth_bits: 0,
        stereo: 0,
        alpha: 0,
    },
    ImodGlRequest {
        double_buffer: 0,
        rgba: 0,
        color_bits: 12,
        depth_bits: 0,
        stereo: 0,
        alpha: 0,
    },
    ImodGlRequest {
        double_buffer: 1,
        rgba: 0,
        color_bits: 8,
        depth_bits: 0,
        stereo: 0,
        alpha: 0,
    },
    ImodGlRequest {
        double_buffer: 0,
        rgba: 0,
        color_bits: 8,
        depth_bits: 0,
        stereo: 0,
        alpha: 0,
    },
    ImodGlRequest {
        double_buffer: 1,
        rgba: 1,
        color_bits: 12,
        depth_bits: 0,
        stereo: 0,
        alpha: 0,
    },
    ImodGlRequest {
        double_buffer: 0,
        rgba: 1,
        color_bits: 12,
        depth_bits: 0,
        stereo: 0,
        alpha: 0,
    },
];

/// Fields owned by `ImodApp` that this paired unit consumes.
#[derive(Clone, Debug)]
pub struct DisplayApplication {
    pub wzoom: i32,
    pub depth: i32,
    pub doublebuffer: i32,
    pub rgba: i32,
    pub qt_enable_depth: i32,
    pub objbase: i32,
    pub base: i32,
    pub device_pixel_ratio: f64,
    /// Value returned by `ImodPrefs->boostModelCursor()`.
    pub model_cursor_boost: i32,
    pub model_cursor_resource: usize,
    pub max_cursor_steps: i32,
    pub background: i32,
    pub foreground: i32,
    pub select: i32,
    pub shadow: i32,
    pub endpoint: i32,
    pub bgnpoint: i32,
    pub curpoint: i32,
    pub ghost: i32,
    pub new_qt_opengl: bool,
    pub visuals: [ImodGlVisual; MAX_VISUALS],
    pub need_to_initialize_vtab: bool,
}

impl Default for DisplayApplication {
    fn default() -> Self {
        Self {
            wzoom: 0,
            depth: 0,
            doublebuffer: 0,
            rgba: 1,
            qt_enable_depth: 0,
            objbase: 0,
            base: 0,
            device_pixel_ratio: 1.,
            model_cursor_boost: 0,
            model_cursor_resource: 0,
            max_cursor_steps: 0,
            background: 1,
            foreground: 2,
            select: 3,
            shadow: 4,
            endpoint: 5,
            bgnpoint: 6,
            curpoint: 7,
            ghost: 8,
            new_qt_opengl: true,
            visuals: [ImodGlVisual {
                valid_direct: -2,
                ..ImodGlVisual::default()
            }; MAX_VISUALS],
            need_to_initialize_vtab: true,
        }
    }
}

/// Data that was static or reached indirectly by `imodDraw`.
#[derive(Clone, Debug, Default)]
pub struct DisplayDrawState {
    pub last_z: i32,
    pub last_time: i32,
    pub colormap_image: bool,
    pub multi_file_z: bool,
    pub ushort_store: bool,
    pub tex_map: bool,
    pub cur_point_extra_obj: bool,
}

/// Qt, GL, `xcramp`, info-window and isosurface operations from this source
/// unit.  They are intentionally direct boundaries, not replacement viewers.
pub trait DisplayNativeBoundary {
    fn set_title(&mut self, title: &str);
    fn mac_m1_cursor_workaround(&mut self) -> bool;
    fn set_model_cursor(&mut self, image: &str, mask: &str, hotspot: Option<(i32, i32)>);
    fn assess_visual(&mut self, request: ImodGlRequest) -> Option<ImodGlVisual>;
    fn color_index(&mut self, index: i32);
    fn color3f(&mut self, red: f32, green: f32, blue: f32);
    fn map_color(&mut self, index: i32, red: i32, green: i32, blue: i32);
    fn ramp_all_init(&mut self, depth: i32, base: i32, size: i32, ushort_store: bool);
    fn ramp_set_levels(&mut self, black: i32, white: i32);
    fn info_set_bw(&mut self, black: i32, white: i32);
    fn draw_colormap(&mut self);
    fn copy_false_colormap(&mut self, z: i32, time: i32, multi_file_z: bool);
    fn ramp(&mut self);
    fn info_bwfloat(&mut self, view: &ImodView, z: i32, time: i32) -> bool;
    fn info_set_ocp(&mut self);
    fn info_set_xyz(&mut self);
    fn imodv_isosurface_update(&mut self, flag: i32) -> bool;
    fn imodv_draw(&mut self);
    fn scale_bar_update(&mut self);
}

/// `imod_display_init`.
pub fn imod_display_init(
    app: &mut DisplayApplication,
    argv: &[String],
    native: &mut dyn DisplayNativeBoundary,
) -> Result<i32, String> {
    app.wzoom = 1;
    app.depth = imod_find_qgl_format(app, argv, native)?;
    imod_setup_cursor(app, native);
    native.set_title("3dmod");
    Ok(0)
}

/// `imodSetupCursor`.
pub fn imod_setup_cursor(app: &mut DisplayApplication, native: &mut dyn DisplayNativeBoundary) {
    const CURSORS: [(&str, &str); 5] = [
        (":/images/cursor.png", ":/images/cursor_mask.png"),
        (":/images/cursor1.5.png", ":/images/cursor_mask1.5.png"),
        (":/images/cursor2.png", ":/images/cursor_mask2.png"),
        (":/images/cursor3.png", ":/images/cursor_mask3.png"),
        (":/images/cursorM1.png", ":/images/cursorM1_mask.png"),
    ];
    let mac_m1 = native.mac_m1_cursor_workaround();
    let set_for_m1 = mac_m1 && app.device_pixel_ratio >= 1.249;
    let mut cursor_ind = if !mac_m1 && app.device_pixel_ratio >= 1.749 {
        2
    } else if !mac_m1 && app.device_pixel_ratio >= 1.249 {
        1
    } else if set_for_m1 {
        CURSORS.len() - 1
    } else {
        0
    };
    app.max_cursor_steps = (CURSORS.len() - 2 - cursor_ind) as i32;
    if !set_for_m1 {
        cursor_ind = (cursor_ind as i32 + app.model_cursor_boost)
            .clamp(0, (CURSORS.len() - 2) as i32) as usize;
    }
    app.model_cursor_resource = cursor_ind;
    let hotspot = set_for_m1.then_some((31, 31));
    native.set_model_cursor(CURSORS[cursor_ind].0, CURSORS[cursor_ind].1, hotspot);
}

/// `imodGetMaxCursorSteps`.
pub fn imod_get_max_cursor_steps(app: &DisplayApplication) -> i32 {
    app.max_cursor_steps
}

/// `imodSetObjectColor`.
pub fn imod_set_object_color(
    app: &DisplayApplication,
    imod: &mut Imod,
    object: i32,
    native: &mut dyn DisplayNativeBoundary,
) {
    let Some(obj) = imod.obj.get_mut(object.max(0) as usize) else {
        return;
    };
    if object < 0 {
        return;
    }
    if app.rgba != 0 {
        native.color3f(obj.red, obj.green, obj.blue);
        return;
    }
    if app.depth <= 8 {
        obj.fgcolor = app.objbase - object;
        native.color_index(app.objbase - object);
    } else {
        obj.fgcolor = app.objbase + object;
        native.color_index(app.objbase + object);
    }
}

/// `mapcolor`.
pub fn mapcolor(
    app: &DisplayApplication,
    color: i32,
    red: i32,
    green: i32,
    blue: i32,
    native: &mut dyn DisplayNativeBoundary,
) -> i32 {
    if app.rgba != 0 {
        return 1;
    }
    native.map_color(color, red, green, blue);
    0
}

/// `imod_color_init`.
pub fn imod_color_init(
    app: &mut DisplayApplication,
    view: &ImodView,
    model: Option<&mut Imod>,
    native: &mut dyn DisplayNativeBoundary,
    prefs: &ImodPreferences,
) -> i32 {
    app.objbase = RAMP_MIN - 1;
    if app.rgba != 0 {
        native.ramp_all_init(app.depth, 0, 256, view.ushort_store != 0);
        native.ramp_set_levels(view.black, view.white);
        return 0;
    }
    if app.depth == 8 {
        native.ramp_all_init(app.depth, RAMP_MIN, RAMP_MAX + 1 - RAMP_MIN, false);
    } else {
        app.objbase = app.base + 257;
        native.ramp_all_init(app.depth, app.base, 256, false);
    }
    if let Some(model) = model {
        imod_cmap(app, model, native, prefs);
    } else {
        map_named_colors(app, native, prefs);
    }
    native.info_set_bw(view.black, view.white);
    native.ramp_set_levels(view.black, view.white);
    0
}

/// `imod_cmap`.
pub fn imod_cmap(
    app: &DisplayApplication,
    model: &mut Imod,
    native: &mut dyn DisplayNativeBoundary,
    prefs: &ImodPreferences,
) {
    if app.rgba != 0 {
        return;
    }
    for (i, obj) in model.obj.iter_mut().enumerate() {
        let (red, green, blue) = (
            (obj.red * 255.) as i32,
            (obj.green * 255.) as i32,
            (obj.blue * 255.) as i32,
        );
        if app.depth == 8 {
            if app.objbase - i as i32 > IMOD_MIN_INDEX {
                mapcolor(app, app.objbase - i as i32, red, green, blue, native);
            }
            obj.fgcolor = app.objbase - i as i32;
        } else {
            mapcolor(app, i as i32 + app.objbase, red, green, blue, native);
            obj.fgcolor = i as i32 + app.objbase;
        }
    }
    map_named_colors(app, native, prefs);
    native.draw_colormap();
}

/// `mapNamedColors`.
pub fn map_named_colors(
    app: &DisplayApplication,
    native: &mut dyn DisplayNativeBoundary,
    prefs: &ImodPreferences,
) {
    for index in [
        app.ghost,
        app.select,
        app.shadow,
        app.endpoint,
        app.bgnpoint,
        app.curpoint,
        app.foreground,
        app.background,
    ] {
        map_one_named_color(app, index, native, prefs);
    }
}

/// `customGhostColor`.
pub fn custom_ghost_color(
    app: &DisplayApplication,
    red: i32,
    green: i32,
    blue: i32,
    native: &mut dyn DisplayNativeBoundary,
) {
    mapcolor(app, app.ghost, red, green, blue, native);
    native.color_index(app.ghost);
    if app.rgba != 0 {
        native.color3f(red as f32 / 255., green as f32 / 255., blue as f32 / 255.);
    }
}

/// `resetGhostColor`.
pub fn reset_ghost_color(
    app: &DisplayApplication,
    native: &mut dyn DisplayNativeBoundary,
    prefs: &ImodPreferences,
) {
    map_one_named_color(app, app.ghost, native, prefs);
}

/// Static `mapOneNamedColor`.
fn map_one_named_color(
    app: &DisplayApplication,
    index: i32,
    native: &mut dyn DisplayNativeBoundary,
    prefs: &ImodPreferences,
) {
    let color = prefs.named_color(index);
    mapcolor(
        app,
        index,
        ((color >> 16) & 255) as i32,
        ((color >> 8) & 255) as i32,
        (color & 255) as i32,
        native,
    );
}

/// Static `rethink`.
fn rethink(view: &mut ImodView) -> i32 {
    if view.imod.is_null() {
        return IMOD_DRAW_MOD;
    }
    unsafe {
        let imod = &mut *view.imod;
        let index = imod.cindex.point;
        if index < 0 {
            return IMOD_DRAW_MOD;
        }
        let Some(obj) = imod.obj.get(imod.cindex.object.max(0) as usize) else {
            return IMOD_DRAW_MOD;
        };
        let Some(contour) = obj.cont.get(imod.cindex.contour.max(0) as usize) else {
            return IMOD_DRAW_MOD;
        };
        let Some(point) = contour.pts.get(index as usize) else {
            return IMOD_DRAW_MOD;
        };
        if iobj_flag_time(obj) != 0 && contour.time != 0 {
            ivw_set_time(view, contour.time);
        }
        view.xmouse = point.x;
        view.ymouse = point.y;
        view.zmouse = point.z;
    }
    ivw_bind_mouse(view);
    IMOD_DRAW_MOD | IMOD_DRAW_XYZ
}

/// `imodDraw`.
pub fn imod_draw(
    app: &DisplayApplication,
    view: &mut ImodView,
    flag: i32,
    state: &mut DisplayDrawState,
    native: &mut dyn DisplayNativeBoundary,
) -> i32 {
    let mut flag = flag;
    if flag & IMOD_DRAW_COLORMAP != 0 {
        ivw_control_list_draw(view, IMOD_DRAW_COLORMAP);
        return 0;
    }
    let (mut cx, mut cy, mut cz) = (0, 0, 0);
    ivw_get_location(view, &mut cx, &mut cy, &mut cz);
    let time = view.cur_time;
    if state.colormap_image && (cz != state.last_z || time != state.last_time) {
        native.copy_false_colormap(cz, time, state.multi_file_z);
        native.ramp();
        state.last_z = cz;
        state.last_time = time;
    }
    if view
        .ctrlist
        .as_ref()
        .is_some_and(|list| !list.list.is_empty())
        && native.info_bwfloat(view, cz, time)
        && app.rgba != 0
    {
        flag |= IMOD_DRAW_IMAGE;
    }
    if flag & IMOD_DRAW_RETHINK != 0 {
        flag |= rethink(view);
    }
    if flag & IMOD_DRAW_MOD != 0 {
        native.info_set_ocp();
    }
    let mut need_modv = false;
    if flag & (IMOD_DRAW_XYZ | IMOD_DRAW_MOD | IMOD_DRAW_IMAGE) != 0 {
        native.info_set_xyz();
        need_modv = native.imodv_isosurface_update(flag);
    }
    ivw_control_list_draw(view, flag);
    if ((flag & IMOD_DRAW_MOD != 0)
        || (flag & IMOD_DRAW_IMAGE != 0 && state.ushort_store && state.tex_map)
        || (flag & IMOD_DRAW_XYZ != 0 && (state.tex_map || state.cur_point_extra_obj))
        || need_modv)
        && flag & IMOD_DRAW_SKIPMODV == 0
    {
        native.imodv_draw();
    }
    native.scale_bar_update();
    0
}

/// Static `imodAssessVisual`.
fn imod_assess_visual(
    app: &mut DisplayApplication,
    index: usize,
    request: ImodGlRequest,
    native: &mut dyn DisplayNativeBoundary,
) {
    let mut visual = native.assess_visual(request).unwrap_or(ImodGlVisual {
        valid_direct: -1,
        ..ImodGlVisual::default()
    });
    visual.db_requested = request.double_buffer;
    visual.rgba_requested = request.rgba;
    visual.depth_enabled = i32::from(request.depth_bits > 0);
    app.visuals[index] = visual;
}

/// `imodFindGLVisual`.
pub fn imod_find_gl_visual(
    app: &mut DisplayApplication,
    request: ImodGlRequest,
    native: &mut dyn DisplayNativeBoundary,
) -> Option<ImodGlVisual> {
    if app.need_to_initialize_vtab {
        app.visuals.iter_mut().for_each(|v| v.valid_direct = -2);
        app.need_to_initialize_vtab = false;
    }
    let index =
        (request.alpha * 16 + request.stereo * 8 + request.double_buffer * 4 + request.rgba * 2)
            as usize;
    if app.visuals[index].valid_direct == -2 {
        imod_assess_visual(app, index, request, native);
    }
    let without_depth = app.visuals[index];
    if request.depth_bits == 0
        && request.stereo == 0
        && without_depth.valid_direct >= 0
        && request.color_bits <= without_depth.color_bits
    {
        return Some(without_depth);
    }
    if app.visuals[index + 1].valid_direct == -2 {
        imod_assess_visual(
            app,
            index + 1,
            ImodGlRequest {
                depth_bits: 1,
                ..request
            },
            native,
        );
    }
    let with_depth = app.visuals[index + 1];
    let no_depth_ok = without_depth.valid_direct >= 0
        && request.color_bits <= without_depth.color_bits
        && request.depth_bits <= without_depth.depth_bits;
    let depth_ok = with_depth.valid_direct >= 0
        && request.color_bits <= with_depth.color_bits
        && request.depth_bits <= with_depth.depth_bits;
    match (no_depth_ok, depth_ok) {
        (false, false) => None,
        (true, false) => Some(without_depth),
        (false, true) => Some(with_depth),
        (true, true) if request.stereo != 0 && without_depth.stereo != with_depth.stereo => {
            Some(if without_depth.stereo != 0 {
                without_depth
            } else {
                with_depth
            })
        }
        (true, true) if request.alpha != 0 && without_depth.alpha != with_depth.alpha => {
            Some(if without_depth.alpha != 0 {
                without_depth
            } else {
                with_depth
            })
        }
        (true, true)
            if without_depth.color_bits < with_depth.color_bits
                || (without_depth.color_bits == with_depth.color_bits
                    && without_depth.depth_bits <= with_depth.depth_bits) =>
        {
            Some(without_depth)
        }
        (true, true) => Some(with_depth),
    }
}

/// `imodFindQGLFormat`.
pub fn imod_find_qgl_format(
    app: &mut DisplayApplication,
    argv: &[String],
    native: &mut dyn DisplayNativeBoundary,
) -> Result<i32, String> {
    if app.new_qt_opengl {
        app.doublebuffer = 1;
        app.rgba = 1;
        app.qt_enable_depth = 1;
        return Ok(24);
    }
    for request in QT_GL_REQUEST_LIST {
        if app.rgba > 0 && request.rgba == 0 || app.rgba < 0 && request.rgba != 0 {
            continue;
        }
        if let Some(visual) = imod_find_gl_visual(app, request, native)
            .filter(|visual| visual.rgba == 0 || visual.color_bits >= 15)
        {
            app.doublebuffer = visual.db_requested;
            app.rgba = visual.rgba_requested;
            app.qt_enable_depth = visual.depth_enabled;
            return Ok(visual.color_bits);
        }
    }
    Err(format!(
        "{}: couldn't get appropriate GL format for Qt windows.",
        argv.first().map_or("3dmod", String::as_str)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::three_dmod::preferences::{
        ImodPrefStruct, PreferencesNativeBoundary, PreferencesSettings,
    };

    #[derive(Default)]
    struct N {
        colors: Vec<(i32, i32, i32, i32)>,
        calls: Vec<String>,
    }
    impl DisplayNativeBoundary for N {
        fn set_title(&mut self, s: &str) {
            self.calls.push(s.into())
        }
        fn mac_m1_cursor_workaround(&mut self) -> bool {
            false
        }
        fn set_model_cursor(&mut self, i: &str, _: &str, _: Option<(i32, i32)>) {
            self.calls.push(i.into())
        }
        fn assess_visual(&mut self, r: ImodGlRequest) -> Option<ImodGlVisual> {
            Some(ImodGlVisual {
                double_buffer: r.double_buffer,
                rgba: r.rgba,
                color_bits: 24,
                depth_bits: 24,
                stereo: r.stereo,
                valid_direct: 1,
                alpha: r.alpha,
                ..Default::default()
            })
        }
        fn color_index(&mut self, _: i32) {}
        fn color3f(&mut self, _: f32, _: f32, _: f32) {}
        fn map_color(&mut self, a: i32, b: i32, c: i32, d: i32) {
            self.colors.push((a, b, c, d))
        }
        fn ramp_all_init(&mut self, _: i32, _: i32, _: i32, _: bool) {}
        fn ramp_set_levels(&mut self, _: i32, _: i32) {}
        fn info_set_bw(&mut self, _: i32, _: i32) {}
        fn draw_colormap(&mut self) {}
        fn copy_false_colormap(&mut self, _: i32, _: i32, _: bool) {}
        fn ramp(&mut self) {}
        fn info_bwfloat(&mut self, _: &ImodView, _: i32, _: i32) -> bool {
            false
        }
        fn info_set_ocp(&mut self) {}
        fn info_set_xyz(&mut self) {}
        fn imodv_isosurface_update(&mut self, _: i32) -> bool {
            false
        }
        fn imodv_draw(&mut self) {}
        fn scale_bar_update(&mut self) {}
    }
    #[derive(Default)]
    struct P;
    impl PreferencesNativeBoundary for P {
        fn set_font(&mut self, _: &str) {}
        fn set_style(&mut self, _: &str) {}
        fn update_dialog(&mut self) {}
        fn update_movie(&mut self) {}
        fn restack_dialogs(&mut self) {}
        fn setup_cursor(&mut self) {}
        fn toggle_model_mode(&mut self) {}
        fn map_named_colors_and_draw(&mut self) {}
    }
    fn prefs() -> ImodPreferences {
        ImodPreferences::new(None, &PreferencesSettings::default(), &mut P)
    }
    #[test]
    fn selects_minimal_visual_and_initializes_display() {
        let mut a = DisplayApplication {
            new_qt_opengl: false,
            ..Default::default()
        };
        let mut n = N::default();
        assert_eq!(imod_display_init(&mut a, &["3dmod".into()], &mut n), Ok(0));
        assert_eq!(a.depth, 24);
        assert_eq!(a.wzoom, 1);
        assert_eq!(n.calls[0], ":/images/cursor.png");
    }
    #[test]
    fn indexed_model_and_named_colors_are_mapped() {
        let mut a = DisplayApplication {
            rgba: 0,
            depth: 8,
            objbase: RAMP_MIN - 1,
            ..Default::default()
        };
        let mut m = Imod::default();
        m.obj.push(Iobj {
            red: 1.,
            green: 0.5,
            blue: 0.,
            ..Default::default()
        });
        let mut n = N::default();
        imod_cmap(&a, &mut m, &mut n, &prefs());
        assert_eq!(m.obj[0].fgcolor, RAMP_MIN - 1);
        assert_eq!(n.colors[0], (RAMP_MIN - 1, 255, 127, 0));
    }
}
