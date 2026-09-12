#![allow(dead_code)]
use crate::imod::three_dmod::mv_control::*;
use crate::imod::three_dmod::mv_window::{Key, KeyEvent};
pub trait ImodvControlNativeBoundary {
    fn setup_ui(&mut self);
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_control_signals(&mut self);
    fn set_label(&mut self, which: i32, text: &str);
    fn set_edit_text(&mut self, which: i32, text: &str);
    fn edit_text(&self, which: i32) -> String;
    fn set_slider(&mut self, which: i32, value: i32);
    fn set_checked(&mut self, which: i32, value: bool);
    fn set_enabled(&mut self, which: i32, value: bool);
    fn edit_font_width(&self, which: i32, text: &str) -> i32;
    fn set_edit_fixed_width(&mut self, which: i32, width: i32);
    fn set_focus(&mut self);
    fn format_general_4(&self, value: f32) -> String;
    fn hot_slider_active(&self, ctrl: bool) -> bool;
    fn hot_slider_enabled(&self) -> bool;
    fn hot_slider_key(&self, key: Key) -> bool;
    fn close_key(&self, event: KeyEvent) -> bool;
    fn grab_keyboard(&mut self);
    fn release_keyboard(&mut self);
    fn accept_close(&mut self);
    fn retranslate_ui(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn widget_change_event(&mut self);
    fn imodv_control_zoom(&mut self, value: i32);
    fn imodv_control_scale(&mut self, value: f32);
    fn imodv_control_kick_clips(&mut self, value: bool);
    fn imodv_control_clip(&mut self, which: i32, value: i32, dragging: bool);
    fn imodv_control_zscale(&mut self, value: i32, dragging: bool);
    fn imodv_control_axis_button(&mut self, value: i32);
    fn imodv_control_axis_text(&mut self, axis: i32, value: f32);
    fn imodv_control_start(&mut self);
    fn imodv_control_rate(&mut self, value: i32);
    fn imodv_control_speed(&mut self, value: f32);
    fn imodv_control_inc_speed(&mut self, value: i32);
    fn imodv_draw(&mut self);
    fn imodv_control_closing(&mut self);
    fn imodv_control_quit(&mut self);
    fn imodv_key_press(&mut self, event: KeyEvent);
    fn imodv_key_release(&mut self, event: KeyEvent);
}
pub const FAR_LABEL: i32 = 0;
pub const NEAR_LABEL: i32 = 1;
pub const PERSPECTIVE_LABEL: i32 = 2;
pub const RATE_LABEL: i32 = 3;
pub const ZSCALE_LABEL: i32 = 4;
pub const SCALE_EDIT: i32 = 0;
pub const SPEED_EDIT: i32 = 1;
pub const X_EDIT: i32 = 2;
pub const Y_EDIT: i32 = 3;
pub const Z_EDIT: i32 = 4;
pub const LINK: i32 = 0;
pub const LINK_CENTER: i32 = 1;
pub const DRAW_PLANE: i32 = 2;
pub const LARGE_PLANE: i32 = 3;
pub const KICK: i32 = 4;
pub const NEAR_SLIDER: i32 = 0;
pub const FAR_SLIDER: i32 = 1;
pub const PERSPECTIVE_SLIDER: i32 = 2;
pub const ZSCALE_SLIDER: i32 = 3;
pub const RATE_SLIDER: i32 = 4;
#[derive(Debug, Default)]
pub struct ImodvControlForm {
    pub m_top_win: bool,
    pub m_rate_displayed: i32,
    pub m_ctrl_pressed: bool,
    pub m_zscale_displayed: i32,
    pub m_zscale_pressed: bool,
    pub m_perspective_displayed: i32,
    pub m_perspective_pressed: bool,
    pub m_far_displayed: i32,
    pub m_far_pressed: bool,
    pub m_near_displayed: i32,
    pub m_near_pressed: bool,
    pub m_rate_pressed: bool,
    pub link_to_slicer: i32,
    pub link_slicer_center: i32,
    pub draw_slicer_plane: i32,
    pub standalone: i32,
}
impl ImodvControlForm {
    /// `imodvControlForm::imodvControlForm`.
    pub fn new(
        standalone: i32,
        link: i32,
        center: i32,
        plane: i32,
        n: &mut dyn ImodvControlNativeBoundary,
    ) -> Self {
        let mut f = Self {
            m_top_win: true,
            standalone,
            link_to_slicer: link,
            link_slicer_center: center,
            draw_slicer_plane: plane,
            ..Default::default()
        };
        n.setup_ui();
        f.init(n);
        f
    }
    /// destructor.
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.retranslate_ui()
    }
    /// `init`.
    pub fn init(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_delete_on_close();
        n.set_always_show_tool_tips();
        n.connect_control_signals();
        self.m_near_pressed = false;
        self.m_far_pressed = false;
        self.m_perspective_pressed = false;
        self.m_zscale_pressed = false;
        self.m_rate_pressed = false;
        self.m_ctrl_pressed = false;
        if self.standalone != 0 {
            n.set_enabled(LINK, false);
            n.set_enabled(DRAW_PLANE, false)
        } else {
            n.set_checked(LINK, self.link_to_slicer != 0);
            n.set_checked(DRAW_PLANE, self.draw_slicer_plane & 1 != 0)
        }
        n.set_checked(LINK_CENTER, self.link_slicer_center != 0);
        n.set_checked(LARGE_PLANE, self.draw_slicer_plane & 2 != 0);
        n.set_enabled(
            LINK_CENTER,
            self.standalone == 0 && self.link_to_slicer != 0,
        );
        n.set_enabled(
            LARGE_PLANE,
            self.standalone == 0 && self.draw_slicer_plane & 1 != 0,
        );
        self.set_font_dependent_widths(n)
    }
    pub fn set_font_dependent_widths(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        let mut width = (15 * n.edit_font_width(SCALE_EDIT, "888888")) / 12;
        n.set_edit_fixed_width(SCALE_EDIT, width);
        n.set_edit_fixed_width(SPEED_EDIT, width);
        width = (17 * n.edit_font_width(X_EDIT, "8888888")) / 14;
        n.set_edit_fixed_width(X_EDIT, width);
        n.set_edit_fixed_width(Y_EDIT, width);
        n.set_edit_fixed_width(Z_EDIT, width)
    }
    pub fn display_far_label(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_label(FAR_LABEL, &v.to_string());
        self.m_far_displayed = v
    }
    pub fn display_near_label(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_label(NEAR_LABEL, &v.to_string());
        self.m_near_displayed = v
    }
    pub fn display_perspective_label(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_label(PERSPECTIVE_LABEL, &v.to_string());
        self.m_perspective_displayed = v
    }
    pub fn display_rate_label(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_label(RATE_LABEL, &format!("{:.1}", v as f32 / 10.));
        self.m_rate_displayed = v
    }
    pub fn display_zscale_label(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_label(
            ZSCALE_LABEL,
            &if v < 200 {
                format!("{:.2}", v as f32 / 100.)
            } else {
                format!("{:.1}", v as f32 / 100.)
            },
        );
        self.m_zscale_displayed = v
    }
    pub fn update_slicer_link(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_checked(LINK, v != 0);
        n.set_enabled(LINK_CENTER, v != 0)
    }
    pub fn zoom_down(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_zoom(-1)
    }
    pub fn zoom_up(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_zoom(1)
    }
    pub fn new_scale(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        let value = n
            .edit_text(SCALE_EDIT)
            .parse::<f32>()
            .unwrap_or(0.)
            .max(0.001);
        self.set_scale_text(value, n);
        n.set_focus();
        n.imodv_control_scale(value)
    }
    pub fn kick_box_toggled(&mut self, state: bool, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_kick_clips(state)
    }
    pub fn near_changed(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        if !self.m_near_pressed || n.hot_slider_active(self.m_ctrl_pressed) {
            n.imodv_control_clip(IMODV_CONTROL_NEAR, v, self.m_near_pressed)
        }
    }
    pub fn far_changed(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        if !self.m_far_pressed || n.hot_slider_active(self.m_ctrl_pressed) {
            n.imodv_control_clip(IMODV_CONTROL_FAR, v, self.m_far_pressed)
        }
    }
    pub fn perspective_changed(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        if !self.m_perspective_pressed || n.hot_slider_active(self.m_ctrl_pressed) {
            n.imodv_control_clip(IMODV_CONTROL_FOVY, v, self.m_perspective_pressed)
        }
    }
    pub fn z_scale_changed(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        if !self.m_zscale_pressed || n.hot_slider_active(self.m_ctrl_pressed) {
            n.imodv_control_zscale(v, self.m_zscale_pressed)
        }
    }
    pub fn rotate_xminus(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_axis_button(-IMODV_CONTROL_XAXIS)
    }
    pub fn rotate_xplus(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_axis_button(IMODV_CONTROL_XAXIS)
    }
    pub fn rotate_yminus(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_axis_button(-IMODV_CONTROL_YAXIS)
    }
    pub fn rotate_yplus(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_axis_button(IMODV_CONTROL_YAXIS)
    }
    pub fn rotate_zminus(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_axis_button(-IMODV_CONTROL_ZAXIS)
    }
    pub fn rotate_zplus(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_axis_button(IMODV_CONTROL_ZAXIS)
    }
    pub fn new_xrotation(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_focus();
        n.imodv_control_axis_text(
            IMODV_CONTROL_XAXIS,
            n.edit_text(X_EDIT).parse().unwrap_or(0.),
        )
    }
    pub fn new_yrotation(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_focus();
        n.imodv_control_axis_text(
            IMODV_CONTROL_YAXIS,
            n.edit_text(Y_EDIT).parse().unwrap_or(0.),
        )
    }
    pub fn new_zrotation(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_focus();
        n.imodv_control_axis_text(
            IMODV_CONTROL_ZAXIS,
            n.edit_text(Z_EDIT).parse().unwrap_or(0.),
        )
    }
    pub fn start_stop(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_start()
    }
    pub fn link_box_toggled(&mut self, state: bool, n: &mut dyn ImodvControlNativeBoundary) {
        self.link_to_slicer = state as i32;
        n.set_enabled(LINK_CENTER, state)
    }
    pub fn slicer_plane_toggled(&mut self, state: bool, n: &mut dyn ImodvControlNativeBoundary) {
        if state {
            self.draw_slicer_plane |= 1
        } else {
            self.draw_slicer_plane &= 2
        }
        n.set_enabled(LARGE_PLANE, state);
        n.imodv_draw()
    }
    pub fn large_plane_toggled(&mut self, state: bool, n: &mut dyn ImodvControlNativeBoundary) {
        if state {
            self.draw_slicer_plane |= 2
        } else {
            self.draw_slicer_plane &= 1
        }
        n.imodv_draw()
    }
    pub fn link_center_toggled(&mut self, state: bool) {
        self.link_slicer_center = state as i32
    }
    pub fn rate_changed(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        if !self.m_rate_pressed || n.hot_slider_active(self.m_ctrl_pressed) {
            n.imodv_control_rate(v)
        }
    }
    pub fn new_speed(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        let v = n
            .edit_text(SPEED_EDIT)
            .parse::<f32>()
            .unwrap_or(0.)
            .max(0.1);
        self.set_speed_text(v, n);
        n.set_focus();
        n.imodv_control_speed(v)
    }
    pub fn increase_speed(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_inc_speed(1)
    }
    pub fn decrease_speed(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_inc_speed(-1)
    }
    pub fn set_axis_text(&mut self, axis: i32, v: f32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_edit_text(
            if axis == 1 {
                X_EDIT
            } else if axis == 2 {
                Y_EDIT
            } else {
                Z_EDIT
            },
            &format!("{v:.2}"),
        )
    }
    pub fn set_scale_text(&mut self, v: f32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_edit_text(SCALE_EDIT, &n.format_general_4(v))
    }
    pub fn set_kick_box(&mut self, state: bool, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_checked(KICK, state)
    }
    pub fn set_view_slider(&mut self, which: i32, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        match which {
            IMODV_CONTROL_NEAR => {
                n.set_slider(NEAR_SLIDER, v);
                self.display_near_label(v, n)
            }
            IMODV_CONTROL_FAR => {
                n.set_slider(FAR_SLIDER, v);
                self.display_far_label(v, n)
            }
            IMODV_CONTROL_FOVY => {
                n.set_slider(PERSPECTIVE_SLIDER, v);
                self.display_perspective_label(v, n)
            }
            IMODV_CONTROL_ZSCALE => {
                n.set_slider(ZSCALE_SLIDER, v);
                self.display_zscale_label(v, n)
            }
            _ => {}
        }
    }
    pub fn set_rotation_rate(&mut self, v: i32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_slider(RATE_SLIDER, v);
        self.display_rate_label(v, n)
    }
    pub fn set_speed_text(&mut self, v: f32, n: &mut dyn ImodvControlNativeBoundary) {
        n.set_edit_text(SPEED_EDIT, &n.format_general_4(v))
    }
    pub fn top_close_event(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        n.imodv_control_closing();
        n.accept_close()
    }
    pub fn far_pressed(&mut self) {
        self.m_far_pressed = true
    }
    pub fn far_released(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        self.m_far_pressed = false;
        self.far_changed(self.m_far_displayed, n)
    }
    pub fn near_pressed(&mut self) {
        self.m_near_pressed = true
    }
    pub fn near_released(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        self.m_near_pressed = false;
        self.near_changed(self.m_near_displayed, n)
    }
    pub fn z_scale_released(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        self.m_zscale_pressed = false;
        self.z_scale_changed(self.m_zscale_displayed, n)
    }
    pub fn perspective_pressed(&mut self) {
        self.m_perspective_pressed = true
    }
    pub fn perspective_released(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        self.m_perspective_pressed = false;
        self.perspective_changed(self.m_perspective_displayed, n)
    }
    pub fn z_scale_pressed(&mut self) {
        self.m_zscale_pressed = true
    }
    pub fn rate_pressed(&mut self) {
        self.m_rate_pressed = true
    }
    pub fn rate_released(&mut self, n: &mut dyn ImodvControlNativeBoundary) {
        self.m_rate_pressed = false;
        self.rate_changed(self.m_rate_displayed, n)
    }
    pub fn key_press_event(&mut self, event: KeyEvent, n: &mut dyn ImodvControlNativeBoundary) {
        if n.close_key(event) {
            n.imodv_control_quit()
        } else {
            if n.hot_slider_enabled() && n.hot_slider_key(event.key) {
                self.m_ctrl_pressed = true;
                n.grab_keyboard()
            }
            n.imodv_key_press(event)
        }
    }
    pub fn key_release_event(&mut self, event: KeyEvent, n: &mut dyn ImodvControlNativeBoundary) {
        if n.hot_slider_key(event.key) {
            self.m_ctrl_pressed = false;
            n.release_keyboard()
        }
        n.imodv_key_release(event)
    }
    pub fn top_change_event(&mut self, font: bool, n: &mut dyn ImodvControlNativeBoundary) {
        n.widget_change_event();
        n.check_and_set_mac_menu();
        if font {
            self.set_font_dependent_widths(n)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        calls: Vec<String>,
        edits: [String; 5],
        widths: Vec<(i32, i32)>,
    }
    impl ImodvControlNativeBoundary for N {
        fn setup_ui(&mut self) {}
        fn set_delete_on_close(&mut self) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn connect_control_signals(&mut self) {}
        fn set_label(&mut self, _: i32, _: &str) {}
        fn set_edit_text(&mut self, a: i32, b: &str) {
            self.edits[a as usize] = b.into()
        }
        fn edit_text(&self, a: i32) -> String {
            self.edits[a as usize].clone()
        }
        fn set_slider(&mut self, _: i32, _: i32) {}
        fn set_checked(&mut self, _: i32, _: bool) {}
        fn set_enabled(&mut self, _: i32, _: bool) {}
        fn edit_font_width(&self, _: i32, _: &str) -> i32 {
            8
        }
        fn set_edit_fixed_width(&mut self, which: i32, width: i32) {
            self.widths.push((which, width))
        }
        fn set_focus(&mut self) {
            self.calls.push("focus".into())
        }
        fn format_general_4(&self, value: f32) -> String {
            format!("{value:.4}")
        }
        fn hot_slider_active(&self, _: bool) -> bool {
            false
        }
        fn hot_slider_enabled(&self) -> bool {
            true
        }
        fn hot_slider_key(&self, k: Key) -> bool {
            k == Key::Character('C')
        }
        fn close_key(&self, _: KeyEvent) -> bool {
            false
        }
        fn grab_keyboard(&mut self) {
            self.calls.push("grab".into())
        }
        fn release_keyboard(&mut self) {}
        fn accept_close(&mut self) {}
        fn retranslate_ui(&mut self) {}
        fn check_and_set_mac_menu(&mut self) {}
        fn widget_change_event(&mut self) {}
        fn imodv_control_zoom(&mut self, a: i32) {
            self.calls.push(format!("zoom:{a}"))
        }
        fn imodv_control_scale(&mut self, a: f32) {
            self.calls.push(format!("scale:{a}"))
        }
        fn imodv_control_kick_clips(&mut self, _: bool) {}
        fn imodv_control_clip(&mut self, a: i32, b: i32, c: bool) {
            self.calls.push(format!("clip:{a}:{b}:{c}"))
        }
        fn imodv_control_zscale(&mut self, _: i32, _: bool) {}
        fn imodv_control_axis_button(&mut self, _: i32) {}
        fn imodv_control_axis_text(&mut self, _: i32, _: f32) {}
        fn imodv_control_start(&mut self) {}
        fn imodv_control_rate(&mut self, _: i32) {}
        fn imodv_control_speed(&mut self, _: f32) {}
        fn imodv_control_inc_speed(&mut self, _: i32) {}
        fn imodv_draw(&mut self) {}
        fn imodv_control_closing(&mut self) {}
        fn imodv_control_quit(&mut self) {}
        fn imodv_key_press(&mut self, _: KeyEvent) {}
        fn imodv_key_release(&mut self, _: KeyEvent) {}
    }
    #[test]
    fn dragged_clip_defers_to_release() {
        let mut n = N::default();
        let mut f = ImodvControlForm::new(0, 0, 0, 0, &mut n);
        f.near_pressed();
        f.near_changed(4, &mut n);
        assert!(n.calls.is_empty());
        f.display_near_label(4, &mut n);
        f.near_released(&mut n);
        assert!(n.calls.iter().any(|x| x == "clip:1:4:false"));
    }
    #[test]
    fn plane_bits_follow_source() {
        let mut n = N::default();
        let mut f = ImodvControlForm::new(0, 0, 0, 0, &mut n);
        f.slicer_plane_toggled(true, &mut n);
        f.large_plane_toggled(true, &mut n);
        assert_eq!(f.draw_slicer_plane, 3);
        f.slicer_plane_toggled(false, &mut n);
        assert_eq!(f.draw_slicer_plane, 2);
    }
    #[test]
    fn source_font_widths_and_scale_focus_are_applied() {
        let mut n = N::default();
        n.edits[SCALE_EDIT as usize] = "0".into();
        let mut f = ImodvControlForm::new(0, 0, 0, 0, &mut n);
        assert!(n.widths.contains(&(SCALE_EDIT, 10)));
        assert!(n.widths.contains(&(X_EDIT, 9)));
        f.new_scale(&mut n);
        assert_eq!(n.edits[SCALE_EDIT as usize], "0.0010");
        assert!(n.calls.iter().any(|call| call == "focus"));
    }
}
