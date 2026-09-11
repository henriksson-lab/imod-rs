//! Translation of `IMOD/3dmod/form_scalebar.cpp` and `form_scalebar.h`.
#![allow(dead_code)]

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScaleBar {
    pub draw: bool,
    pub draw_on_snapshots: bool,
    pub white: bool,
    pub vertical: bool,
    pub color_ramp: bool,
    pub invert_ramp: bool,
    pub min_length: i32,
    pub thickness: i32,
    pub position: i32,
    pub indent_x: i32,
    pub indent_y: i32,
    pub use_custom: bool,
    pub use_exact: bool,
    pub custom_val: i32,
    pub exact_val: f32,
    pub last_length: f32,
    pub draw_labels: bool,
    pub label_size: i32,
    pub label_yoffset: i32,
}
pub const MIN_LABEL_SIZE: i32 = 1;
/// Native Qt, scale-bar, renderer, and key-routing boundary.
pub trait ScaleBarNativeBoundary {
    fn units(&self) -> String;
    fn standalone(&self) -> bool;
    fn set_attributes_and_signals(&mut self);
    fn set_checked(&mut self, which: i32, value: bool);
    fn set_spin(&mut self, which: i32, value: i32);
    fn set_enabled(&mut self, which: i32, value: bool);
    fn set_position(&mut self, value: i32);
    fn set_units(&mut self, units: &str);
    fn set_exact_text(&mut self, text: &str);
    fn exact_text(&self) -> String;
    fn hide_nonstandalone_values(&mut self);
    fn set_value_text(&mut self, which: i32, text: &str);
    fn kill_timer(&mut self, id: i32);
    fn start_timer(&mut self, msec: i32) -> i32;
    fn redraw(&mut self);
    fn image_cleanup(&mut self);
    fn new_qt_opengl(&self) -> bool;
    fn close_key(&self) -> bool;
    fn close(&mut self);
    fn imodv_key_press(&mut self);
    fn imodv_key_release(&mut self);
    fn ivw_control_key(&mut self, release: bool);
    fn scale_bar_closing(&mut self);
    fn accept_close(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn font_change(&self) -> bool;
}
pub const DRAW: i32 = 0;
pub const DRAW_SNAPS: i32 = 1;
pub const WHITE: i32 = 2;
pub const VERTICAL: i32 = 3;
pub const COLOR: i32 = 4;
pub const INVERT: i32 = 5;
pub const CUSTOM: i32 = 6;
pub const EXACT: i32 = 7;
pub const LABELS: i32 = 8;
pub const LENGTH: i32 = 0;
pub const THICKNESS: i32 = 1;
pub const INDENT_X: i32 = 2;
pub const INDENT_Y: i32 = 3;
pub const CUSTOM_VALUE: i32 = 4;
pub const LABEL_SIZE: i32 = 5;
pub const LABEL_OFFSET: i32 = 6;

/// `ScaleBarForm`.
#[derive(Clone, Debug, Default)]
pub struct ScaleBarForm {
    pub m_params: ScaleBar,
    pub m_timer_id: i32,
}
impl ScaleBarForm {
    /// `ScaleBarForm::ScaleBarForm`.
    pub fn new(params: ScaleBar, native: &mut dyn ScaleBarNativeBoundary) -> Self {
        let mut f = Self {
            m_params: params,
            ..Default::default()
        };
        f.init(native);
        f
    }
    /// `ScaleBarForm::~ScaleBarForm`.
    pub fn destroy(&mut self) {}
    /// `ScaleBarForm::languageChange`.
    pub fn language_change(&mut self) {}
    /// `ScaleBarForm::init`.
    pub fn init(&mut self, n: &mut dyn ScaleBarNativeBoundary) {
        n.set_attributes_and_signals();
        let p = &self.m_params;
        n.set_checked(DRAW, p.draw);
        n.set_checked(DRAW_SNAPS, p.draw_on_snapshots);
        n.set_enabled(DRAW_SNAPS, p.draw);
        n.set_checked(WHITE, p.white);
        n.set_checked(VERTICAL, p.vertical);
        n.set_checked(COLOR, p.color_ramp);
        n.set_checked(INVERT, p.invert_ramp);
        n.set_spin(LENGTH, p.min_length);
        n.set_spin(THICKNESS, p.thickness);
        n.set_spin(INDENT_X, p.indent_x);
        n.set_spin(INDENT_Y, p.indent_y);
        n.set_position(p.position);
        n.set_checked(CUSTOM, p.use_custom);
        n.set_checked(EXACT, p.use_exact);
        n.set_spin(CUSTOM_VALUE, p.custom_val);
        n.set_checked(LABELS, p.draw_labels);
        n.set_spin(LABEL_SIZE, p.label_size);
        n.set_spin(LABEL_OFFSET, p.label_yoffset);
        if n.standalone() {
            n.hide_nonstandalone_values();
        }
        n.set_enabled(CUSTOM_VALUE, p.use_custom && !p.use_exact);
        n.set_enabled(CUSTOM, !p.use_exact);
        n.set_enabled(EXACT, p.use_exact);
        let units = n.units();
        n.set_units(&units);
        let draw_labels = p.draw_labels;
        self.set_exact_val_text(n);
        n.set_enabled(LABEL_SIZE, draw_labels);
        n.set_enabled(LABEL_OFFSET, draw_labels);
    }
    /// `ScaleBarForm::drawToggled`.
    pub fn draw_toggled(&mut self, state: bool, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.draw = state;
        n.set_enabled(DRAW_SNAPS, state);
        n.redraw()
    }
    /// `ScaleBarForm::drawOnSnapsToggled`.
    pub fn draw_on_snaps_toggled(&mut self, state: bool) {
        self.m_params.draw_on_snapshots = state
    }
    /// `ScaleBarForm::whiteToggled`.
    pub fn white_toggled(&mut self, state: bool, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.white = state;
        n.redraw()
    }
    /// `ScaleBarForm::verticalToggled`.
    pub fn vertical_toggled(&mut self, state: bool, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.vertical = state;
        n.redraw()
    }
    /// `ScaleBarForm::colorToggled`.
    pub fn color_toggled(&mut self, state: bool, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.color_ramp = state;
        n.redraw()
    }
    /// `ScaleBarForm::invertToggled`.
    pub fn invert_toggled(&mut self, state: bool, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.invert_ramp = state;
        n.redraw()
    }
    /// `ScaleBarForm::lengthChanged`.
    pub fn length_changed(&mut self, value: i32, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.min_length = value;
        n.redraw()
    }
    /// `ScaleBarForm::thicknessChanged`.
    pub fn thickness_changed(&mut self, value: i32, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.thickness = value;
        n.redraw()
    }
    /// `ScaleBarForm::positionChanged`.
    pub fn position_changed(&mut self, value: i32, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.position = value;
        n.redraw()
    }
    /// `ScaleBarForm::indentXchanged`.
    pub fn indent_xchanged(&mut self, value: i32, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.indent_x = value;
        n.redraw()
    }
    /// `ScaleBarForm::indentYchanged`.
    pub fn indent_ychanged(&mut self, value: i32, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.indent_y = value;
        n.redraw()
    }
    /// `ScaleBarForm::customToggled`.
    pub fn custom_toggled(&mut self, state: bool, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.use_custom = state;
        n.set_enabled(CUSTOM_VALUE, state);
        n.redraw()
    }
    /// `ScaleBarForm::customValChanged`.
    pub fn custom_val_changed(&mut self, value: i32, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.custom_val = value;
        n.redraw()
    }
    /// `ScaleBarForm::exactToggled`.
    pub fn exact_toggled(&mut self, state: bool, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.use_exact = state;
        n.set_enabled(EXACT, state);
        n.set_enabled(CUSTOM, !state);
        n.set_enabled(CUSTOM_VALUE, !state && self.m_params.use_custom);
        if state && self.m_params.exact_val <= 0. {
            self.m_params.exact_val = self.m_params.last_length;
            self.set_exact_val_text(n)
        }
        n.redraw()
    }
    /// `ScaleBarForm::setExactValText`.
    pub fn set_exact_val_text(&mut self, n: &mut dyn ScaleBarNativeBoundary) {
        n.set_exact_text(&format!("{}", self.m_params.exact_val))
    }
    /// `ScaleBarForm::exactValChanged`.
    pub fn exact_val_changed(&mut self, n: &mut dyn ScaleBarNativeBoundary) {
        let value = n.exact_text().parse::<f32>().unwrap_or(0.).max(0.00001);
        self.m_params.exact_val = value;
        self.set_exact_val_text(n);
        n.redraw()
    }
    /// `ScaleBarForm::drawLabelsToggled`.
    pub fn draw_labels_toggled(&mut self, state: bool, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.draw_labels = state;
        n.set_enabled(LABEL_SIZE, state);
        n.set_enabled(LABEL_OFFSET, state);
        if n.new_qt_opengl() {
            n.image_cleanup()
        }
        n.redraw()
    }
    /// `ScaleBarForm::labelSizeChanged`.
    pub fn label_size_changed(&mut self, mut value: i32, n: &mut dyn ScaleBarNativeBoundary) {
        if value < MIN_LABEL_SIZE {
            value = 0
        }
        self.m_params.label_size = value;
        if self.m_params.draw_labels {
            if n.new_qt_opengl() {
                n.image_cleanup()
            }
            n.redraw()
        }
    }
    /// `ScaleBarForm::labelOffsetChanged`.
    pub fn label_offset_changed(&mut self, value: i32, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_params.label_yoffset = value;
        if self.m_params.draw_labels {
            n.redraw()
        }
    }
    /// `ScaleBarForm::updateValues`.
    pub fn update_values(
        &mut self,
        zapv: f32,
        multizv: f32,
        slicerv: f32,
        xyzv: f32,
        modvv: f32,
        units: &str,
        n: &mut dyn ScaleBarNativeBoundary,
    ) {
        if self.m_timer_id != 0 {
            n.kill_timer(self.m_timer_id)
        }
        self.m_timer_id = 0;
        if !n.standalone() {
            for (which, v) in [(0, zapv), (1, multizv), (2, slicerv), (3, xyzv)] {
                let text = if v > 0. {
                    format!("{v} {units}")
                } else {
                    String::new()
                };
                n.set_value_text(which, &text)
            }
        }
        let text = if modvv > 0. {
            format!("{modvv} {units}")
        } else {
            String::new()
        };
        n.set_value_text(4, &text)
    }
    /// `ScaleBarForm::startUpdateTimer`.
    pub fn start_update_timer(&mut self, n: &mut dyn ScaleBarNativeBoundary) {
        if self.m_timer_id != 0 {
            n.kill_timer(self.m_timer_id)
        }
        self.m_timer_id = n.start_timer(100)
    }
    /// `ScaleBarForm::timerEvent`.
    pub fn timer_event(&mut self, n: &mut dyn ScaleBarNativeBoundary) {
        self.m_timer_id = 0;
        n.redraw()
    }
    /// `ScaleBarForm::keyPressEvent`.
    pub fn key_press_event(&mut self, n: &mut dyn ScaleBarNativeBoundary) {
        if n.close_key() {
            n.close()
        } else if n.standalone() {
            n.imodv_key_press()
        } else {
            n.ivw_control_key(false)
        }
    }
    /// `ScaleBarForm::keyReleaseEvent`.
    pub fn key_release_event(&mut self, n: &mut dyn ScaleBarNativeBoundary) {
        if n.standalone() {
            n.imodv_key_release()
        } else {
            n.ivw_control_key(true)
        }
    }
    /// `ScaleBarForm::topCloseEvent`.
    pub fn top_close_event(&mut self, n: &mut dyn ScaleBarNativeBoundary) {
        n.scale_bar_closing();
        n.accept_close()
    }
    /// `ScaleBarForm::topChangeEvent`.
    pub fn top_change_event(&mut self, n: &mut dyn ScaleBarNativeBoundary) {
        n.check_and_set_mac_menu();
        if !n.font_change() {
            return;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        p: ScaleBar,
        redraws: i32,
        text: String,
    }
    impl ScaleBarNativeBoundary for N {
        fn units(&self) -> String {
            "nm".into()
        }
        fn standalone(&self) -> bool {
            false
        }
        fn set_attributes_and_signals(&mut self) {}
        fn set_checked(&mut self, _: i32, _: bool) {}
        fn set_spin(&mut self, _: i32, _: i32) {}
        fn set_enabled(&mut self, _: i32, _: bool) {}
        fn set_position(&mut self, _: i32) {}
        fn set_units(&mut self, _: &str) {}
        fn set_exact_text(&mut self, t: &str) {
            self.text = t.into()
        }
        fn exact_text(&self) -> String {
            self.text.clone()
        }
        fn hide_nonstandalone_values(&mut self) {}
        fn set_value_text(&mut self, _: i32, _: &str) {}
        fn kill_timer(&mut self, _: i32) {}
        fn start_timer(&mut self, _: i32) -> i32 {
            1
        }
        fn redraw(&mut self) {
            self.redraws += 1
        }
        fn image_cleanup(&mut self) {}
        fn new_qt_opengl(&self) -> bool {
            false
        }
        fn close_key(&self) -> bool {
            false
        }
        fn close(&mut self) {}
        fn imodv_key_press(&mut self) {}
        fn imodv_key_release(&mut self) {}
        fn ivw_control_key(&mut self, _: bool) {}
        fn scale_bar_closing(&mut self) {}
        fn accept_close(&mut self) {}
        fn check_and_set_mac_menu(&mut self) {}
        fn font_change(&self) -> bool {
            false
        }
    }
    #[test]
    fn exact_is_clamped_and_redrawn() {
        let mut n = N::default();
        let mut f = ScaleBarForm::new(ScaleBar::default(), &mut n);
        n.text = "0".into();
        f.exact_val_changed(&mut n);
        assert_eq!(f.m_params.exact_val, 0.00001);
        assert!(n.redraws > 0)
    }
}
