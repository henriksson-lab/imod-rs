//! Translation of `IMOD/3dmod/scalebar.cpp` together with `scalebar.h`.
//!
//! The original keeps the scale-bar parameters and dialog pointers in file
//! statics.  [`ScaleBarState`] is their Rust owner; the Qt/OpenGL/window
//! calls remain on the explicit [`ScaleBarNativeBoundary`] boundary.
#![allow(dead_code)]

/// `ScaleBar` (`scalebar.h`).
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleBar {
    pub draw: bool,
    pub draw_on_snapshots: bool,
    pub white: bool,
    pub min_length: i32,
    pub thickness: i32,
    pub vertical: bool,
    pub position: i32,
    pub indent_x: i32,
    pub indent_y: i32,
    pub use_custom: bool,
    pub custom_val: i32,
    pub use_exact: bool,
    pub exact_val: f32,
    pub color_ramp: bool,
    pub invert_ramp: bool,
    pub last_length: f32,
    pub draw_labels: bool,
    pub label_size: i32,
    pub label_yoffset: i32,
    pub scale_label: f32,
}

impl Default for ScaleBar {
    fn default() -> Self {
        Self {
            draw: true,
            draw_on_snapshots: false,
            white: false,
            min_length: 50,
            thickness: 8,
            vertical: false,
            position: 0,
            indent_x: 20,
            indent_y: 20,
            use_custom: false,
            custom_val: 25,
            use_exact: false,
            exact_val: 0.,
            color_ramp: false,
            invert_ramp: false,
            last_length: 100.,
            draw_labels: false,
            label_size: 14,
            label_yoffset: 2,
            scale_label: 1.,
        }
    }
}

/// File-static `params`, `sNeedNoDia`, `sbDia`, and `sTopWin`.
#[derive(Clone, Debug, Default)]
pub struct ScaleBarState {
    pub params: ScaleBar,
    pub need_no_dia: i32,
    pub dialog_open: bool,
}

/// Concrete Qt, viewer/window, and OpenGL calls made by this source file.
///
/// The scale-bar computation, state changes, timer trigger, and draw sequence
/// are retained below.  A concrete desktop frontend supplies this boundary.
pub trait ScaleBarNativeBoundary {
    fn standalone(&self) -> bool;
    fn loading_image(&self) -> bool;
    fn imodv_closed(&self) -> bool;
    fn pixel_size(&self) -> Option<f32>;
    fn current_pixel_size(&self) -> f32;
    fn xy_bin(&self) -> f32;
    fn units(&self) -> String;
    fn new_qt_opengl(&self) -> bool;
    fn has_gl_widget(&self) -> bool;

    fn raise_dialog(&mut self);
    fn create_dialog(&mut self) -> bool;
    fn set_dialog_title(&mut self);
    fn register_dialog(&mut self);
    fn show_dialog(&mut self);
    fn remove_dialog(&mut self);
    fn dialog_update_values(
        &mut self,
        zap_len: f32,
        multi_zlen: f32,
        slicer_len: f32,
        xyz_len: f32,
        modv_len: f32,
        units: &str,
    );
    fn dialog_start_update_timer(&mut self);

    fn image_cleanup(&mut self);
    fn redraw_imod(&mut self);
    fn redraw_imodv(&mut self);
    fn print_stderr(&mut self, text: &str);

    fn depth_test_enabled(&self) -> bool;
    fn set_depth_test_enabled(&mut self, enabled: bool);
    fn set_ghost_color(&mut self, red: i32, green: i32, blue: i32);
    fn reset_ghost_color(&mut self);
    fn draw_filled_rectangle(&mut self, x: i32, y: i32, width: i32, height: i32);
    fn color_ramp(&self, index: i32) -> (i32, i32, i32);
    fn draw_line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32);
    fn draw_label(
        &mut self,
        x: i32,
        y: i32,
        text: &str,
        color: i32,
        label_size: i32,
        scale_label: f32,
    );

    fn slicer_scale_bar_size(&self) -> Option<f32>;
    fn zap_scale_bar_size(&self) -> Option<f32>;
    fn multi_z_scale_bar_size(&self) -> Option<f32>;
    fn xyz_scale_bar_size(&self) -> f32;
    fn imodv_scale_bar_size(&self) -> f32;
}

/// `scaleBarOpen`.
pub fn scale_bar_open(state: &mut ScaleBarState, native: &mut dyn ScaleBarNativeBoundary) {
    if state.dialog_open {
        native.raise_dialog();
        return;
    }
    if !native.create_dialog() {
        native.print_stderr("Could not open Scale Bar dialog\n");
        return;
    }
    state.dialog_open = true;
    native.set_dialog_title();
    scale_bar_redraw(state, native);
    if native.new_qt_opengl() {
        native.image_cleanup();
        native.redraw_imodv();
    }
    scale_bar_update(state, native);
    native.show_dialog();
    native.register_dialog();
}

/// `scaleBarClosing`.
pub fn scale_bar_closing(state: &mut ScaleBarState, native: &mut dyn ScaleBarNativeBoundary) {
    native.remove_dialog();
    state.dialog_open = false;
    scale_bar_redraw(state, native);
}

/// `setScaleBarWithoutDialog` (declared from `imodview.h`).
pub fn set_scale_bar_without_dialog(state: &mut ScaleBarState, enable: bool) {
    state.need_no_dia += if enable { 1 } else { -1 };
    state.need_no_dia = state.need_no_dia.max(0);
}

/// `scaleBarAssess`.
pub fn scale_bar_assess(
    state: &mut ScaleBarState,
    native: &dyn ScaleBarNativeBoundary,
    winx: i32,
    winy: i32,
    zoom: f32,
    pixlen: &mut i32,
    xst: &mut i32,
    yst: &mut i32,
    xsize: &mut i32,
    ysize: &mut i32,
) -> f32 {
    if !state.params.draw || !(state.dialog_open || state.need_no_dia != 0) {
        return -1.;
    }
    let Some(mut pixsize) = native.pixel_size() else {
        return -1.;
    };
    if !native.standalone() {
        pixsize = native.current_pixel_size();
    }
    pixsize *= if native.standalone() {
        1.
    } else {
        native.xy_bin()
    };
    let minlen = pixsize * state.params.min_length as f32 / zoom;
    let loglen = minlen.log10();
    let expon = loglen.floor();
    let mut normlen = 10_f32.powf(loglen - expon);
    if state.params.use_custom {
        let mut custlen = state.params.custom_val as f32 / 10.;
        if custlen < normlen {
            custlen *= 10.;
        }
        if custlen >= 10. * normlen {
            custlen /= 10.;
        }
        normlen = custlen;
    } else if normlen < 2. {
        normlen = 2.;
    } else if normlen < 5. {
        normlen = 5.;
    } else {
        normlen = 10.;
    }
    let truelen = if state.params.use_exact {
        state.params.exact_val
    } else {
        normlen * 10_f32.powf(expon)
    };
    *pixlen = (truelen * zoom / pixsize).round() as i32;
    *xsize = if state.params.vertical {
        state.params.thickness
    } else {
        *pixlen
    };
    *ysize = if state.params.vertical {
        *pixlen
    } else {
        state.params.thickness
    };
    *xst = state.params.indent_x;
    if state.params.position == 0 || state.params.position == 3 {
        *xst = winx - state.params.indent_x - *xsize;
    }
    *yst = state.params.indent_y;
    if state.params.position == 2 || state.params.position == 3 {
        *yst = winy - state.params.indent_y - *ysize;
    }
    state.params.last_length = truelen;
    truelen
}

/// `scaleBarTestAdjust`.
pub fn scale_bar_test_adjust(
    state: &mut ScaleBarState,
    native: &mut dyn ScaleBarNativeBoundary,
    winx: i32,
    winy: i32,
    zoom: f32,
) {
    let (mut pixlen, mut xst, mut yst, mut xsize, mut ysize) = (0, 0, 0, 0, 0);
    let truelen = scale_bar_assess(
        state,
        native,
        winx,
        winy,
        zoom,
        &mut pixlen,
        &mut xst,
        &mut yst,
        &mut xsize,
        &mut ysize,
    );
    if truelen > 0. && (xst < 0 || yst < 0 || xst + xsize >= winx || yst + ysize >= winy) {
        let min_indent = state.params.indent_x.min(state.params.indent_y);
        while (xst < 0 || yst < 0 || xst + xsize >= winx || yst + ysize >= winy)
            && (state.params.indent_x > min_indent || state.params.indent_y > min_indent)
        {
            if state.params.indent_x > min_indent {
                state.params.indent_x -= 1;
            }
            if state.params.indent_y > min_indent {
                state.params.indent_y -= 1;
            }
            scale_bar_assess(
                state,
                native,
                winx,
                winy,
                zoom,
                &mut pixlen,
                &mut xst,
                &mut yst,
                &mut xsize,
                &mut ysize,
            );
        }
        let length_lim = state.params.min_length / 2;
        while (xst < 0 || yst < 0 || xst + xsize >= winx || yst + ysize >= winy)
            && state.params.min_length > length_lim
        {
            state.params.min_length -= 1;
            scale_bar_assess(
                state,
                native,
                winx,
                winy,
                zoom,
                &mut pixlen,
                &mut xst,
                &mut yst,
                &mut xsize,
                &mut ysize,
            );
        }
        if xst < 0 || yst < 0 || xst + xsize >= winx || yst + ysize >= winy {
            native.print_stderr("Scale bar cannot be adjusted to fit in one panel\n");
            state.params.draw = false;
        } else {
            native.print_stderr("Scale bar position or size was adjusted to fit in one panel\n");
        }
    }
}

/// `scaleBarDraw`.
pub fn scale_bar_draw(
    state: &mut ScaleBarState,
    native: &mut dyn ScaleBarNativeBoundary,
    winx: i32,
    winy: i32,
    zoom: f32,
    background: i32,
) -> f32 {
    let (mut pixlen, mut xst, mut yst, mut xsize, mut ysize) = (0, 0, 0, 0, 0);
    let truelen = scale_bar_assess(
        state,
        native,
        winx,
        winy,
        zoom,
        &mut pixlen,
        &mut xst,
        &mut yst,
        &mut xsize,
        &mut ysize,
    );
    if truelen < 0. {
        return truelen;
    }
    let depth_enabled = native.depth_test_enabled();
    if depth_enabled {
        native.set_depth_test_enabled(false);
    }
    let mut color = if state.params.white { 255 } else { 0 };
    if background != 0 {
        color = if background > 0 { 0 } else { 255 };
    }
    if !state.params.color_ramp {
        native.set_ghost_color(color, color, color);
        native.draw_filled_rectangle(xst, yst, xsize, ysize);
    } else {
        pixlen = pixlen.max(1);
        for i in 0..=pixlen {
            let j = if state.params.invert_ramp {
                pixlen - i
            } else {
                i
            };
            let index = (255. * j as f32 / pixlen as f32).round() as i32;
            let (red, green, blue) = native.color_ramp(index);
            native.set_ghost_color(red, green, blue);
            if state.params.vertical {
                native.draw_line(xst, yst + i, xst + xsize, yst + i);
            } else {
                native.draw_line(xst + i, yst, xst + i, yst + ysize);
            }
        }
    }
    if state.params.draw_labels && native.has_gl_widget() {
        if state.params.color_ramp {
            native.set_ghost_color(color, color, color);
        }
        let text = format!("{} {}", truelen, native.units());
        native.draw_label(
            xst + xsize / 2,
            winy - (yst
                - (state.params.label_yoffset as f32 * state.params.scale_label).round() as i32),
            &text,
            color,
            state.params.label_size,
            state.params.scale_label,
        );
    }
    native.reset_ghost_color();
    if depth_enabled {
        native.set_depth_test_enabled(true);
    }
    if state.dialog_open {
        native.dialog_start_update_timer();
    }
    truelen
}

/// `scaleBarUpdate`.
pub fn scale_bar_update(state: &ScaleBarState, native: &mut dyn ScaleBarNativeBoundary) {
    if (!native.standalone() && native.loading_image()) || native.pixel_size().is_none() {
        return;
    }
    let (mut zap, mut slicer, mut xyz, mut multiz, mut modv) = (-1., -1., -1., -1., -1.);
    scale_bar_all_lengths(
        state,
        native,
        &mut zap,
        &mut slicer,
        &mut xyz,
        &mut multiz,
        &mut modv,
    );
    if state.dialog_open {
        let units = native.units();
        native.dialog_update_values(zap, multiz, slicer, xyz, modv, &units);
    }
}

/// `scaleBarAllLengths` (declared from `imodview.h`).
pub fn scale_bar_all_lengths(
    state: &ScaleBarState,
    native: &dyn ScaleBarNativeBoundary,
    zap_len: &mut f32,
    slicer_len: &mut f32,
    xyz_len: &mut f32,
    multi_zlen: &mut f32,
    modv_len: &mut f32,
) {
    *slicer_len = -1.;
    *zap_len = -1.;
    *multi_zlen = -1.;
    *modv_len = -1.;
    *xyz_len = -1.;
    if !(state.dialog_open || state.need_no_dia != 0) {
        return;
    }
    if !native.standalone() {
        if let Some(value) = native.slicer_scale_bar_size() {
            *slicer_len = value;
        }
        if let Some(value) = native.zap_scale_bar_size() {
            *zap_len = value;
        }
        if let Some(value) = native.multi_z_scale_bar_size() {
            *multi_zlen = value;
        }
        *xyz_len = native.xyz_scale_bar_size();
    }
    if !native.imodv_closed() {
        *modv_len = native.imodv_scale_bar_size();
    }
}

/// `scaleBarGetParams`.
pub fn scale_bar_get_params(state: &mut ScaleBarState) -> &mut ScaleBar {
    &mut state.params
}

/// `scaleBarRedraw`.
pub fn scale_bar_redraw(state: &ScaleBarState, native: &mut dyn ScaleBarNativeBoundary) {
    if !native.standalone() && native.loading_image() {
        return;
    }
    if !native.standalone() {
        native.redraw_imod();
    }
    native.redraw_imodv();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        redraws: i32,
        depth: bool,
        messages: Vec<String>,
    }
    impl ScaleBarNativeBoundary for N {
        fn standalone(&self) -> bool {
            false
        }
        fn loading_image(&self) -> bool {
            false
        }
        fn imodv_closed(&self) -> bool {
            false
        }
        fn pixel_size(&self) -> Option<f32> {
            Some(1.)
        }
        fn current_pixel_size(&self) -> f32 {
            1.
        }
        fn xy_bin(&self) -> f32 {
            1.
        }
        fn units(&self) -> String {
            "nm".into()
        }
        fn new_qt_opengl(&self) -> bool {
            false
        }
        fn has_gl_widget(&self) -> bool {
            false
        }
        fn raise_dialog(&mut self) {}
        fn create_dialog(&mut self) -> bool {
            true
        }
        fn set_dialog_title(&mut self) {}
        fn register_dialog(&mut self) {}
        fn show_dialog(&mut self) {}
        fn remove_dialog(&mut self) {}
        fn dialog_update_values(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: &str) {}
        fn dialog_start_update_timer(&mut self) {}
        fn image_cleanup(&mut self) {}
        fn redraw_imod(&mut self) {
            self.redraws += 1
        }
        fn redraw_imodv(&mut self) {
            self.redraws += 1
        }
        fn print_stderr(&mut self, t: &str) {
            self.messages.push(t.into())
        }
        fn depth_test_enabled(&self) -> bool {
            self.depth
        }
        fn set_depth_test_enabled(&mut self, e: bool) {
            self.depth = e
        }
        fn set_ghost_color(&mut self, _: i32, _: i32, _: i32) {}
        fn reset_ghost_color(&mut self) {}
        fn draw_filled_rectangle(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn color_ramp(&self, _: i32) -> (i32, i32, i32) {
            (0, 0, 0)
        }
        fn draw_line(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn draw_label(&mut self, _: i32, _: i32, _: &str, _: i32, _: i32, _: f32) {}
        fn slicer_scale_bar_size(&self) -> Option<f32> {
            None
        }
        fn zap_scale_bar_size(&self) -> Option<f32> {
            None
        }
        fn multi_z_scale_bar_size(&self) -> Option<f32> {
            None
        }
        fn xyz_scale_bar_size(&self) -> f32 {
            -1.
        }
        fn imodv_scale_bar_size(&self) -> f32 {
            0.
        }
    }
    #[test]
    fn assess_uses_next_standard_length_and_position() {
        let mut state = ScaleBarState {
            dialog_open: true,
            ..Default::default()
        };
        let n = N::default();
        let (mut p, mut x, mut y, mut xs, mut ys) = (0, 0, 0, 0, 0);
        assert_eq!(
            scale_bar_assess(
                &mut state, &n, 200, 100, 1., &mut p, &mut x, &mut y, &mut xs, &mut ys
            ),
            50.
        );
        assert_eq!((p, x, y, xs, ys), (50, 130, 20, 50, 8));
    }
    #[test]
    fn no_dialog_reference_count_is_saturating() {
        let mut s = ScaleBarState::default();
        set_scale_bar_without_dialog(&mut s, false);
        assert_eq!(s.need_no_dia, 0);
        set_scale_bar_without_dialog(&mut s, true);
        assert_eq!(s.need_no_dia, 1);
    }
    #[test]
    fn adjustment_disables_impossible_bar() {
        let mut s = ScaleBarState {
            dialog_open: true,
            ..Default::default()
        };
        s.params.min_length = 100;
        s.params.indent_x = 0;
        s.params.indent_y = 0;
        let mut n = N::default();
        scale_bar_test_adjust(&mut s, &mut n, 10, 10, 1.);
        assert!(!s.params.draw);
        assert_eq!(n.messages.len(), 1);
    }
}
