#![allow(dead_code)]

use crate::imod::libimod::iobj::{
    IOBJ_SYM_CIRCLE, IOBJ_SYM_NONE, IOBJ_SYM_SQUARE, IOBJ_SYM_TRIANGLE, IOBJ_SYMF_FILL,
};
use crate::imod::libimod::istore::DrawProps;
pub const GEN_STORE_COLOR: i32 = 1;
pub const GEN_STORE_FCOLOR: i32 = 2;
pub const GEN_STORE_TRANS: i32 = 3;
pub const GEN_STORE_3DWIDTH: i32 = 6;
pub const GEN_STORE_2DWIDTH: i32 = 7;
pub const GEN_STORE_SYMTYPE: i32 = 8;
pub const GEN_STORE_SYMSIZE: i32 = 9;
pub const CHANGED_COLOR: i32 = 1;
pub const CHANGED_FCOLOR: i32 = 2;
pub const CHANGED_TRANS: i32 = 4;
pub const CHANGED_3DWIDTH: i32 = 32;
pub const CHANGED_2DWIDTH: i32 = 64;
pub const CHANGED_SYMTYPE: i32 = 128;
pub const CHANGED_SYMSIZE: i32 = 256;
pub const CHANGED_VALUE1: i32 = 512;
pub const LINE_COLOR: i32 = 0;
pub const FILL_COLOR: i32 = 1;
pub const TRANS: i32 = 2;
pub const WIDTH_2D: i32 = 3;
pub const WIDTH_3D: i32 = 4;
pub const SYMTYPE: i32 = 5;
pub const SYMSIZE: i32 = 6;

pub trait FineGrainNativeBoundary {
    fn setup_ui(&mut self);
    fn setup(&mut self);
    fn group(&mut self, i: i32);
    fn enabled(&mut self, i: i32, on: bool);
    fn ds(&mut self, i: i32, s: &str);
    fn color(&mut self, fill: bool, r: i32, g: i32, b: i32);
    fn trans(&mut self, v: i32);
    fn spin(&mut self, i: i32, v: i32);
    fn symbol(&mut self, i: i32);
    fn check(&mut self, i: i32, on: bool);
    fn gap_text(&mut self, s: &str);
    fn gap_tip(&mut self, s: &str);
    fn value(&mut self, s: &str);
    fn format_general_precision(&self, value: f32, precision: i32) -> String;
    fn rounded_style(&self) -> bool;
    fn set_button_width(&mut self, which: i32, rounded: bool, factor: f32, text: &str) -> i32;
    fn ds_label_font_width(&self) -> i32;
    fn set_fixed_width(&mut self, which: i32, width: i32);
    fn retranslate(&mut self);
    fn raise_line(&mut self);
    fn raise_fill(&mut self);
    fn open_line(&mut self, r: i32, g: i32, b: i32);
    fn open_fill(&mut self, r: i32, g: i32, b: i32);
    fn close_line(&mut self);
    fn close_fill(&mut self);
    fn remove_line(&mut self);
    fn remove_fill(&mut self);
    fn pt_cont_surf(&mut self, i: i32);
    fn change_all(&mut self, on: bool);
    fn goto_change(&mut self, previous: bool);
    fn line_color(&mut self, r: i32, g: i32, b: i32);
    fn fill_color(&mut self, r: i32, g: i32, b: i32);
    fn transparency(&mut self, v: i32);
    fn width_2d(&mut self, v: i32);
    fn width_3d(&mut self, v: i32);
    fn symsize(&mut self, v: i32);
    fn symtype(&mut self, v: i32, fill: bool);
    fn end(&mut self, t: i32);
    fn clear(&mut self, t: i32);
    fn gap(&mut self, on: bool);
    fn no_cap(&mut self, on: bool);
    fn connect(&mut self, v: i32);
    fn draw_connect(&mut self, on: bool);
    fn stipple_gaps(&mut self, on: bool);
    fn closing(&mut self);
    fn dump(&mut self);
    fn focus_form(&mut self);
    fn close_key(&self) -> bool;
    fn d_key(&self) -> bool;
    fn hot_flag(&self) -> i32;
    fn hot_slider_active(&self, ctrl_pressed: bool) -> bool;
    fn hot_key(&self) -> bool;
    fn grab_keyboard(&mut self);
    fn release_keyboard(&mut self);
    fn control_key(&mut self, release: bool);
    fn close_top(&mut self);
    fn accept(&mut self);
    fn mac_menu(&mut self);
    fn font_change(&self) -> bool;
    fn widget_change_event(&mut self);
}
#[derive(Clone, Debug)]
pub struct FineGrainForm {
    pub m_top_win: bool,
    pub m_cur_fill_blue: i32,
    pub m_cur_blue: i32,
    pub m_cur_fill_green: i32,
    pub m_cur_green: i32,
    pub m_cur_fill_red: i32,
    pub m_cur_red: i32,
    pub m_last_sym_fill: bool,
    pub m_last_fill_blue: i32,
    pub m_last_fill_green: i32,
    pub m_last_fill_red: i32,
    pub m_last_blue: i32,
    pub m_last_green: i32,
    pub m_last_red: i32,
    pub m_last_trans: i32,
    pub m_last_sym_type: i32,
    pub m_last_symsize: i32,
    pub m_last_3dwidth: i32,
    pub m_last_2dwidth: i32,
    pub m_last_buts: [bool; 7],
    pub m_end_buts: [bool; 7],
    pub m_clear_buts: [bool; 7],
    pub m_type_values: [i32; 7],
    pub m_change_flags: [i32; 7],
    pub m_pt_cont_surf: i32,
    pub m_ctrl_pressed: bool,
    pub m_sym_table: [i32; 4],
    pub m_line_selector: bool,
    pub m_fill_selector: bool,
    pub m_dslabels: [bool; 7],
    pub m_trans_slider: bool,
    pub surf_cont_pt_group: bool,
}
impl Default for FineGrainForm {
    fn default() -> Self {
        Self {
            m_top_win: false,
            m_cur_fill_blue: 0,
            m_cur_blue: 0,
            m_cur_fill_green: 0,
            m_cur_green: 0,
            m_cur_fill_red: 0,
            m_cur_red: 0,
            m_last_sym_fill: false,
            m_last_fill_blue: -1,
            m_last_fill_green: -1,
            m_last_fill_red: -1,
            m_last_blue: -1,
            m_last_green: -1,
            m_last_red: -1,
            m_last_trans: -1,
            m_last_sym_type: -1,
            m_last_symsize: -1,
            m_last_3dwidth: -1,
            m_last_2dwidth: -1,
            m_last_buts: [false; 7],
            m_end_buts: [false; 7],
            m_clear_buts: [false; 7],
            m_type_values: [1, 2, 3, 7, 6, 8, 9],
            m_change_flags: [1, 2, 4, 64, 32, 128, 256],
            m_pt_cont_surf: 0,
            m_ctrl_pressed: false,
            m_sym_table: [
                IOBJ_SYM_NONE,
                IOBJ_SYM_CIRCLE,
                IOBJ_SYM_SQUARE,
                IOBJ_SYM_TRIANGLE,
            ],
            m_line_selector: false,
            m_fill_selector: false,
            m_dslabels: [true; 7],
            m_trans_slider: false,
            surf_cont_pt_group: false,
        }
    }
}
impl FineGrainForm {
    /// `FineGrainForm()` source constructor.
    pub fn new(n: &mut dyn FineGrainNativeBoundary) -> Self {
        let mut s = Self::default();
        s.m_top_win = true;
        n.setup_ui();
        s.init(n);
        s
    }
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.retranslate()
    }
    pub fn init(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.setup();
        self.m_trans_slider = true;
        self.surf_cont_pt_group = true;
        self.set_font_dependent_widths(n)
    }
    pub fn set_font_dependent_widths(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        let rounded = n.rounded_style();
        let dswid = n.set_button_width(0, rounded, 1.3, "Set");
        n.set_fixed_width(1, dswid);
        let cwid = n.set_button_width(2, rounded, 1.25, "Clear");
        let lwid = n.set_button_width(3, rounded, 1.25, "Last");
        let ewid = n.set_button_width(4, rounded, 1.25, "End");
        let dswid = (1.1 * n.ds_label_font_width() as f32 + 0.5) as i32;
        n.set_fixed_width(5, dswid);
        for i in 1..7 {
            n.set_fixed_width(5 + i as i32, dswid);
            n.set_fixed_width(12 + i as i32, lwid);
            n.set_fixed_width(19 + i as i32, ewid);
            n.set_fixed_width(26 + i as i32, cwid);
        }
        let cwid = n.set_button_width(33, rounded, 1.25, "Previous");
        n.set_fixed_width(34, cwid)
    }
    pub fn update(
        &mut self,
        pcs: i32,
        enabled: i32,
        p: &DrawProps,
        flags: i32,
        next: bool,
        prev: bool,
        n: &mut dyn FineGrainNativeBoundary,
    ) {
        let min = [1, 1, 1, 2, 1, 2, 2];
        n.group(pcs);
        n.enabled(20, pcs == 1);
        n.enabled(21, next);
        n.enabled(22, prev);
        for i in 0..7 {
            let changed = flags & self.m_change_flags[i] != 0;
            self.m_end_buts[i] = pcs == 0 && changed && enabled >= min[i];
            self.m_clear_buts[i] = changed && enabled >= min[i];
            n.enabled(30 + i as i32, self.m_end_buts[i]);
            n.enabled(40 + i as i32, self.m_clear_buts[i]);
            n.ds(
                i as i32,
                if changed && enabled >= min[i] {
                    "S"
                } else {
                    "D"
                },
            );
        }
        self.m_last_buts = [
            self.m_last_red >= 0 && enabled > 0,
            self.m_last_fill_red >= 0 && enabled > 0,
            self.m_last_trans >= 0 && enabled > 0,
            self.m_last_2dwidth >= 0 && enabled > 1,
            self.m_last_3dwidth >= 0 && enabled > 0,
            self.m_last_sym_type >= 0 && enabled > 1,
            self.m_last_symsize >= 0 && enabled > 1,
        ];
        for i in 0..7 {
            n.enabled(50 + i as i32, self.m_last_buts[i])
        }
        self.m_cur_red = (255. * p.red) as i32;
        self.m_cur_green = (255. * p.green) as i32;
        self.m_cur_blue = (255. * p.blue) as i32;
        n.color(false, self.m_cur_red, self.m_cur_green, self.m_cur_blue);
        self.m_cur_fill_red = (255. * p.fill_red) as i32;
        self.m_cur_fill_green = (255. * p.fill_green) as i32;
        self.m_cur_fill_blue = (255. * p.fill_blue) as i32;
        n.color(
            true,
            self.m_cur_fill_red,
            self.m_cur_fill_green,
            self.m_cur_fill_blue,
        );
        n.enabled(60, enabled > 0);
        n.enabled(61, enabled > 0);
        n.enabled(62, enabled > 0);
        n.trans(p.trans);
        n.spin(WIDTH_2D, p.linewidth2);
        n.enabled(63, enabled > 1);
        n.spin(WIDTH_3D, p.linewidth);
        n.enabled(64, enabled > 0);
        if let Some(i) = self.m_sym_table.iter().position(|&x| x == p.symtype) {
            n.symbol(i as i32)
        }
        n.enabled(65, enabled > 1);
        n.check(SYMTYPE, p.symflags & IOBJ_SYMF_FILL as i32 != 0);
        n.enabled(66, enabled > 1);
        n.spin(SYMSIZE, p.symsize);
        n.enabled(67, enabled > 1);
        n.check(68, p.gap != 0);
        n.gap_text(if pcs != 0 {
            "Turn off drawing"
        } else {
            "Gap to next point"
        });
        n.enabled(68, enabled > 1);
        n.gap_tip(if pcs != 0 {
            "Do not draw this contour or surface"
        } else {
            "Do not draw line connecting to next point (hot key Ctrl+G)"
        });
        n.enabled(69, pcs == 1);
        n.check(69, p.no_cap != 0);
        n.spin(70, p.connect);
        n.enabled(70, pcs < 2 && enabled > 1);
        let v = if enabled != 0 && flags & CHANGED_VALUE1 != 0 {
            n.format_general_precision(p.value1, 5)
        } else {
            String::new()
        };
        n.value(&v)
    }
    pub fn pt_cont_surf_selected(&mut self, v: i32, n: &mut dyn FineGrainNativeBoundary) {
        n.enabled(20, v == 1);
        n.pt_cont_surf(v)
    }
    /// `FineGrainForm::changeAllToggled`.
    pub fn change_all_toggled(&mut self, on: bool, n: &mut dyn FineGrainNativeBoundary) {
        n.change_all(on)
    }
    /// `FineGrainForm::nextChangeClicked`.
    pub fn next_change_clicked(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.goto_change(false)
    }
    /// `FineGrainForm::prevChangeClicked`.
    pub fn prev_change_clicked(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.goto_change(true)
    }
    /// `FineGrainForm::setLineColor`.
    pub fn set_line_color(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        if self.m_line_selector {
            n.raise_line()
        } else {
            self.m_line_selector = true;
            n.open_line(self.m_cur_red, self.m_cur_green, self.m_cur_blue)
        }
    }
    /// `FineGrainForm::setFillColor`.
    pub fn set_fill_color(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        if self.m_fill_selector {
            n.raise_fill()
        } else {
            self.m_fill_selector = true;
            n.open_fill(
                self.m_cur_fill_red,
                self.m_cur_fill_green,
                self.m_cur_fill_blue,
            )
        }
    }
    /// `FineGrainForm::newLineColor`.
    pub fn new_line_color(&mut self, r: i32, g: i32, b: i32, n: &mut dyn FineGrainNativeBoundary) {
        self.m_last_red = r;
        self.m_last_green = g;
        self.m_last_blue = b;
        n.line_color(r, g, b)
    }
    /// `FineGrainForm::newFillColor`.
    pub fn new_fill_color(&mut self, r: i32, g: i32, b: i32, n: &mut dyn FineGrainNativeBoundary) {
        self.m_last_fill_red = r;
        self.m_last_fill_green = g;
        self.m_last_fill_blue = b;
        n.fill_color(r, g, b)
    }
    /// `FineGrainForm::lineColorDone`.
    pub fn line_color_done(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        if self.m_line_selector {
            n.close_line()
        }
    }
    /// `FineGrainForm::fillColorDone`.
    pub fn fill_color_done(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        if self.m_fill_selector {
            n.close_fill()
        }
    }
    /// `FineGrainForm::lineColorClosing`.
    pub fn line_color_closing(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.remove_line();
        self.m_line_selector = false
    }
    /// `FineGrainForm::fillColorClosing`.
    pub fn fill_color_closing(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.remove_fill();
        self.m_fill_selector = false
    }
    /// `FineGrainForm::applyLastChange`.
    pub fn apply_last_change(&mut self, t: i32, n: &mut dyn FineGrainNativeBoundary) -> i32 {
        if !(0..=6).contains(&t) || !self.m_last_buts[t as usize] {
            return 0;
        }
        match t {
            0 => self.last_line_color(n),
            1 => self.last_fill_color(n),
            2 => self.last_trans(n),
            3 => self.last_2d_width(n),
            4 => self.last_3d_width(n),
            5 => self.last_symtype(n),
            6 => self.last_symsize(n),
            _ => return 0,
        }
        1
    }
    /// `FineGrainForm::lastLineColor`.
    pub fn last_line_color(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.line_color(self.m_last_red, self.m_last_green, self.m_last_blue)
    }
    /// `FineGrainForm::lastFillColor`.
    pub fn last_fill_color(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.fill_color(
            self.m_last_fill_red,
            self.m_last_fill_green,
            self.m_last_fill_blue,
        )
    }
    /// `FineGrainForm::lastTrans`.
    pub fn last_trans(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.transparency(self.m_last_trans)
    }
    /// `FineGrainForm::last2DWidth`.
    pub fn last_2d_width(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.width_2d(self.m_last_2dwidth)
    }
    /// `FineGrainForm::last3DWidth`.
    pub fn last_3d_width(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.width_3d(self.m_last_3dwidth)
    }
    /// `FineGrainForm::lastSymtype`.
    pub fn last_symtype(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.symtype(self.m_last_sym_type, self.m_last_sym_fill)
    }
    /// `FineGrainForm::lastSymsize`.
    pub fn last_symsize(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.symsize(self.m_last_symsize)
    }
    /// `FineGrainForm::endClicked`.
    pub fn end_clicked(&mut self, i: i32, n: &mut dyn FineGrainNativeBoundary) {
        if let Some(&t) = self.m_type_values.get(i as usize) {
            n.end(t)
        }
    }
    /// `FineGrainForm::clearClicked`.
    pub fn clear_clicked(&mut self, i: i32, n: &mut dyn FineGrainNativeBoundary) {
        if let Some(&t) = self.m_type_values.get(i as usize) {
            n.clear(t)
        }
    }
    /// `FineGrainForm::transSliderChanged`.
    pub fn trans_slider_changed(
        &mut self,
        _: i32,
        v: i32,
        dragging: bool,
        n: &mut dyn FineGrainNativeBoundary,
    ) {
        if !dragging || n.hot_slider_active(self.m_ctrl_pressed) {
            n.transparency(v);
            self.m_last_trans = v
        }
    }
    /// `FineGrainForm::width2DChanged`.
    pub fn width_2d_changed(&mut self, v: i32, n: &mut dyn FineGrainNativeBoundary) {
        n.focus_form();
        self.m_last_2dwidth = v;
        n.width_2d(v)
    }
    /// `FineGrainForm::width3DChanged`.
    pub fn width_3d_changed(&mut self, v: i32, n: &mut dyn FineGrainNativeBoundary) {
        n.focus_form();
        self.m_last_3dwidth = v;
        n.width_3d(v)
    }
    /// `FineGrainForm::symsizeChanged`.
    pub fn symsize_changed(&mut self, v: i32, n: &mut dyn FineGrainNativeBoundary) {
        n.focus_form();
        self.m_last_symsize = v;
        n.symsize(v)
    }
    /// `FineGrainForm::symtypeSelected`.
    pub fn symtype_selected(&mut self, i: i32, fill: bool, n: &mut dyn FineGrainNativeBoundary) {
        if let Some(&v) = self.m_sym_table.get(i as usize) {
            self.m_last_sym_type = v;
            self.m_last_sym_fill = fill;
            n.symtype(v, fill)
        }
    }
    /// `FineGrainForm::fillToggled`.
    pub fn fill_toggled(&mut self, on: bool, i: i32, n: &mut dyn FineGrainNativeBoundary) {
        self.symtype_selected(i, on, n)
    }
    /// `FineGrainForm::gapToggled`.
    pub fn gap_toggled(&mut self, on: bool, n: &mut dyn FineGrainNativeBoundary) {
        n.gap(on)
    }
    /// `FineGrainForm::noCapToggled`.
    pub fn no_cap_toggled(&mut self, on: bool, n: &mut dyn FineGrainNativeBoundary) {
        n.no_cap(on)
    }
    /// `FineGrainForm::connectChanged`.
    pub fn connect_changed(&mut self, v: i32, n: &mut dyn FineGrainNativeBoundary) {
        n.focus_form();
        n.connect(v)
    }
    /// `FineGrainForm::drawConnectToggled`.
    pub fn draw_connect_toggled(&mut self, on: bool, n: &mut dyn FineGrainNativeBoundary) {
        n.draw_connect(on)
    }
    /// `FineGrainForm::stippleGapsToggled`.
    pub fn stipple_gaps_toggled(&mut self, on: bool, n: &mut dyn FineGrainNativeBoundary) {
        n.stipple_gaps(on)
    }
    /// `FineGrainForm::topCloseEvent`.
    pub fn top_close_event(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.closing();
        self.line_color_done(n);
        self.fill_color_done(n);
        n.accept()
    }
    /// `FineGrainForm::keyPressEvent`.
    pub fn key_press_event(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        if n.close_key() {
            n.close_top()
        } else if n.d_key() {
            n.dump()
        } else {
            if n.hot_flag() != 0 && n.hot_key() {
                self.m_ctrl_pressed = true;
                n.grab_keyboard()
            }
            n.control_key(false)
        }
    }
    /// `FineGrainForm::keyReleaseEvent`.
    pub fn key_release_event(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        if n.hot_key() {
            self.m_ctrl_pressed = false;
            n.release_keyboard()
        }
        n.control_key(true)
    }
    /// `FineGrainForm::topChangeEvent`.
    pub fn top_change_event(&mut self, n: &mut dyn FineGrainNativeBoundary) {
        n.widget_change_event();
        n.mac_menu();
        if n.font_change() {
            self.set_font_dependent_widths(n)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        e: Vec<String>,
        hot: i32,
    }
    impl FineGrainNativeBoundary for N {
        fn setup_ui(&mut self) {}
        fn setup(&mut self) {}
        fn group(&mut self, _: i32) {}
        fn enabled(&mut self, _: i32, _: bool) {}
        fn ds(&mut self, _: i32, _: &str) {}
        fn color(&mut self, _: bool, _: i32, _: i32, _: i32) {}
        fn trans(&mut self, _: i32) {}
        fn spin(&mut self, _: i32, _: i32) {}
        fn symbol(&mut self, _: i32) {}
        fn check(&mut self, _: i32, _: bool) {}
        fn gap_text(&mut self, _: &str) {}
        fn gap_tip(&mut self, _: &str) {}
        fn value(&mut self, _: &str) {}
        fn format_general_precision(&self, value: f32, _: i32) -> String {
            value.to_string()
        }
        fn rounded_style(&self) -> bool {
            false
        }
        fn set_button_width(&mut self, _: i32, _: bool, _: f32, _: &str) -> i32 {
            0
        }
        fn ds_label_font_width(&self) -> i32 {
            0
        }
        fn set_fixed_width(&mut self, _: i32, _: i32) {}
        fn retranslate(&mut self) {}
        fn raise_line(&mut self) {
            self.e.push("raise".into())
        }
        fn raise_fill(&mut self) {}
        fn open_line(&mut self, r: i32, g: i32, b: i32) {
            self.e.push(format!("open:{r},{g},{b}"))
        }
        fn open_fill(&mut self, _: i32, _: i32, _: i32) {}
        fn close_line(&mut self) {
            self.e.push("close".into())
        }
        fn close_fill(&mut self) {}
        fn remove_line(&mut self) {}
        fn remove_fill(&mut self) {}
        fn pt_cont_surf(&mut self, _: i32) {}
        fn change_all(&mut self, _: bool) {}
        fn goto_change(&mut self, _: bool) {}
        fn line_color(&mut self, r: i32, g: i32, b: i32) {
            self.e.push(format!("line:{r},{g},{b}"))
        }
        fn fill_color(&mut self, _: i32, _: i32, _: i32) {}
        fn transparency(&mut self, v: i32) {
            self.e.push(format!("trans:{v}"))
        }
        fn width_2d(&mut self, _: i32) {}
        fn width_3d(&mut self, _: i32) {}
        fn symsize(&mut self, _: i32) {}
        fn symtype(&mut self, v: i32, f: bool) {
            self.e.push(format!("sym:{v},{f}"))
        }
        fn end(&mut self, _: i32) {}
        fn clear(&mut self, _: i32) {}
        fn gap(&mut self, _: bool) {}
        fn no_cap(&mut self, _: bool) {}
        fn connect(&mut self, _: i32) {}
        fn draw_connect(&mut self, _: bool) {}
        fn stipple_gaps(&mut self, _: bool) {}
        fn closing(&mut self) {}
        fn dump(&mut self) {}
        fn focus_form(&mut self) {}
        fn close_key(&self) -> bool {
            false
        }
        fn d_key(&self) -> bool {
            false
        }
        fn hot_flag(&self) -> i32 {
            self.hot
        }
        fn hot_slider_active(&self, ctrl_pressed: bool) -> bool {
            self.hot != 0 && ctrl_pressed
        }
        fn hot_key(&self) -> bool {
            false
        }
        fn grab_keyboard(&mut self) {}
        fn release_keyboard(&mut self) {}
        fn control_key(&mut self, _: bool) {}
        fn close_top(&mut self) {}
        fn accept(&mut self) {}
        fn mac_menu(&mut self) {}
        fn font_change(&self) -> bool {
            false
        }
        fn widget_change_event(&mut self) {}
    }
    #[test]
    fn selector_lifecycle() {
        let mut n = N::default();
        let mut f = FineGrainForm::new(&mut n);
        assert!(f.m_top_win && f.m_trans_slider && f.surf_cont_pt_group);
        f.m_cur_red = 4;
        f.m_cur_green = 5;
        f.m_cur_blue = 6;
        f.set_line_color(&mut n);
        f.set_line_color(&mut n);
        f.new_line_color(1, 2, 3, &mut n);
        f.line_color_done(&mut n);
        f.line_color_closing(&mut n);
        assert_eq!(n.e, ["open:4,5,6", "raise", "line:1,2,3", "close"]);
        assert!(!f.m_line_selector)
    }
    #[test]
    fn last_and_symbol_mapping() {
        let mut n = N::default();
        let mut f = FineGrainForm::new(&mut n);
        f.m_last_sym_type = IOBJ_SYM_SQUARE;
        f.m_last_sym_fill = true;
        f.m_last_buts[5] = true;
        assert_eq!(f.apply_last_change(5, &mut n), 1);
        f.symtype_selected(3, false, &mut n);
        assert_eq!(n.e, ["sym:2,true", "sym:3,false"])
    }
    #[test]
    fn dragged_trans_requires_hot() {
        let mut n = N::default();
        let mut f = FineGrainForm::new(&mut n);
        f.trans_slider_changed(0, 20, true, &mut n);
        assert_eq!(f.m_last_trans, -1);
        n.hot = 1;
        f.m_ctrl_pressed = true;
        f.trans_slider_changed(0, 30, true, &mut n);
        assert_eq!(n.e, ["trans:30"])
    }
}
