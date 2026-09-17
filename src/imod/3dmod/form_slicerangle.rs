#![allow(dead_code)]
pub const SLAN_COLS: usize = 7;
pub const ANGLE_STRSIZE: usize = 128;
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ipoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlicerAngles {
    pub angles: [f32; 3],
    pub center: Ipoint,
    pub time: i32,
    pub label: String,
}
pub trait SlicerAngleNativeBoundary {
    fn setup_ui(&mut self);
    fn retranslate_ui(&mut self);
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_slicer_angle_signals(&mut self);
    fn set_spin_maximum(&mut self, which: i32, value: i32);
    fn spin_value(&self, which: i32) -> i32;
    fn set_button_enabled(&mut self, which: i32, value: bool);
    fn hide_control(&mut self, which: i32);
    fn resize_for_no_volume_group(&mut self);
    fn set_focus(&mut self);
    fn rounded_style(&self) -> bool;
    fn set_button_width(&mut self, which: i32, rounded: bool, factor: f32, text: &str) -> i32;
    fn set_button_fixed_width(&mut self, which: i32, width: i32);
    fn table_font_width(&self, text: &str) -> i32;
    fn set_table_column_width(&mut self, column: i32, width: i32);
    fn volume_dimensions(&self) -> (i32, i32, i32);
    fn set_time_label(&mut self, text: &str);
    fn select_row(&mut self, row: i32);
    fn row_count(&self) -> i32;
    fn set_table_rows(&mut self, rows: &[SlicerAngles]);
    fn set_table_row(&mut self, row: i32, angle: &SlicerAngles, block: bool);
    fn current_row(&self) -> i32;
    fn focus_table(&mut self);
    fn close(&mut self);
    fn accept_close(&mut self);
    fn help(&mut self, page: &str);
    fn top_slicer_time(&self) -> (i32, bool);
    fn top_slicer_angles(&self) -> Option<([f32; 3], Ipoint, i32)>;
    fn set_top_slicer_angles(&mut self, angles: [f32; 3], center: Ipoint, draw: bool);
    fn display_time(&self) -> i32;
    fn time_label(&self, time: i32) -> String;
    fn undo_model_change(&mut self);
    fn undo_finish(&mut self);
    fn undo_flush(&mut self);
    fn slicer_angles_closing(&mut self);
    fn close_key(&self) -> bool;
    fn ivw_control_key(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn widget_change_event(&mut self);
}
pub const SET_BUTTON: i32 = 0;
pub const DELETE_BUTTON: i32 = 1;
pub const RENUMBER_BUTTON: i32 = 2;
pub const COPY_BUTTON: i32 = 3;
pub const RENUMBER_SPIN: i32 = 0;
pub const COPY_SPIN: i32 = 1;
pub const REMOVE_BUTTON: i32 = 4;
pub const INSERT_BUTTON: i32 = 5;
pub const VOLUME_GROUP: i32 = 6;
pub const TIME_LABEL: i32 = 7;
#[derive(Debug)]
pub struct SlicerAngleForm {
    pub m_ignore_cur_chg: bool,
    pub m_time_inc: i32,
    pub m_max_image_time: i32,
    pub m_max_model_time: i32,
    pub m_cur_row: Vec<i32>,
    pub m_cur_time: i32,
    pub slicer_ang: Vec<SlicerAngles>,
    pub last_dragging: i32,
}
impl SlicerAngleForm {
    /// `SlicerAngleForm()` source constructor.
    pub fn new(
        slicer_ang: Vec<SlicerAngles>,
        max_image_time: i32,
        n: &mut dyn SlicerAngleNativeBoundary,
    ) -> Self {
        n.setup_ui();
        let mut f = Self {
            m_ignore_cur_chg: false,
            m_time_inc: 0,
            m_max_image_time: max_image_time,
            m_max_model_time: 0,
            m_cur_row: Vec::new(),
            m_cur_time: 0,
            slicer_ang,
            last_dragging: 0,
        };
        f.init(n);
        f
    }
    /// `SlicerAngleForm::~SlicerAngleForm`.
    pub fn destroy(&mut self) {
        self.m_cur_row.clear()
    }
    pub fn language_change(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        n.retranslate_ui()
    }
    /// `SlicerAngleForm::init`.
    pub fn init(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        n.set_delete_on_close();
        n.set_always_show_tool_tips();
        n.connect_slicer_angle_signals();
        if self.m_max_image_time == 0 {
            self.m_max_image_time = 1;
            self.m_time_inc = 1
        }
        self.m_max_model_time = self.m_max_image_time;
        for s in &self.slicer_ang {
            self.m_max_model_time = self.m_max_model_time.max(s.time)
        }
        n.set_spin_maximum(RENUMBER_SPIN, self.m_max_model_time);
        n.set_spin_maximum(COPY_SPIN, self.m_max_model_time);
        if self.m_max_model_time == 1 {
            n.hide_control(REMOVE_BUTTON);
            n.hide_control(COPY_BUTTON);
            n.hide_control(RENUMBER_BUTTON);
            n.hide_control(INSERT_BUTTON);
            n.hide_control(RENUMBER_SPIN);
            n.hide_control(COPY_SPIN);
            n.hide_control(VOLUME_GROUP);
            n.resize_for_no_volume_group();
        }
        if self.m_max_image_time == 1 {
            n.hide_control(TIME_LABEL);
        }
        self.m_cur_row = vec![-1; (self.m_max_model_time + 1) as usize];
        let (time, _) = n.top_slicer_time();
        let time = if time < 0 { n.display_time() } else { time } + self.m_time_inc;
        self.set_font_dependent_widths(n);
        self.load_table(time, n);
        self.m_cur_row[time as usize] = -1;
        if n.row_count() > 0 {
            n.select_row(0);
            self.m_cur_row[time as usize] = 0
        }
        self.update_enables(n);
        if self.m_max_model_time > 1 {
            self.set_time_label(n)
        }
        self.set_font_dependent_widths(n)
    }
    pub fn update_enables(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        let rows = n.row_count();
        n.set_button_enabled(SET_BUTTON, rows > 0);
        n.set_button_enabled(DELETE_BUTTON, rows > 0);
        n.set_button_enabled(
            RENUMBER_BUTTON,
            n.spin_value(RENUMBER_SPIN) != self.m_cur_time,
        );
        n.set_button_enabled(COPY_BUTTON, n.spin_value(COPY_SPIN) != self.m_cur_time)
    }
    pub fn change_event(&mut self, font_change: bool, n: &mut dyn SlicerAngleNativeBoundary) {
        n.widget_change_event();
        n.check_and_set_mac_menu();
        if font_change {
            self.set_font_dependent_widths(n)
        }
    }
    pub fn set_font_dependent_widths(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        let rounded = n.rounded_style();
        let mut width = n.set_button_width(SET_BUTTON, rounded, 1.2, "Set Angles");
        n.set_button_fixed_width(DELETE_BUTTON, width);
        n.set_button_width(RENUMBER_BUTTON, rounded, 1.2, "Renumber To");
        n.set_button_width(COPY_BUTTON, rounded, 1.2, "Copy From");
        width = n.set_button_width(REMOVE_BUTTON, rounded, 1.2, "Remove");
        n.set_button_fixed_width(INSERT_BUTTON, width);
        width = n.table_font_width("-90.00") + 8;
        n.set_table_column_width(0, width);
        width = n.table_font_width("-180.00") + 8;
        n.set_table_column_width(1, width);
        n.set_table_column_width(2, width);
        let width2 = (1.2 * n.table_font_width("X cen") as f32).round() as i32 + 8;
        let (xsize, ysize, zsize) = n.volume_dimensions();
        width = n.table_font_width(&format!("{xsize:.2}")) + 8;
        n.set_table_column_width(3, width.max(width2));
        width = n.table_font_width(&format!("{ysize:.2}")) + 8;
        n.set_table_column_width(4, width.max(width2));
        width = n.table_font_width(&format!("{zsize:.2}")) + 8;
        n.set_table_column_width(5, width.max(width2));
        n.set_table_column_width(6, n.table_font_width("abcdefghijklmnop") + 8)
    }
    /// `SlicerAngleForm::setTimeLabel`.
    pub fn set_time_label(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        n.set_time_label(&format!(
            "Time: {} ({})",
            self.m_cur_time,
            n.time_label(self.m_cur_time)
        ))
    }
    /// `SlicerAngleForm::setCurrentOrNewRow`.
    pub fn set_current_or_new_row(
        &mut self,
        time: i32,
        newrow: bool,
        n: &mut dyn SlicerAngleNativeBoundary,
    ) {
        if time + self.m_time_inc != self.m_cur_time {
            self.switch_time(time + self.m_time_inc, false, n)
        }
        if newrow || n.row_count() == 0 {
            self.new_clicked(n)
        } else {
            self.get_ang_clicked(n)
        }
    }
    /// `SlicerAngleForm::setAnglesFromRow`.
    pub fn set_angles_from_row(&mut self, time: i32, n: &mut dyn SlicerAngleNativeBoundary) {
        if time + self.m_time_inc != self.m_cur_time {
            self.switch_time(time + self.m_time_inc, false, n)
        }
        self.set_angles(true, n)
    }
    /// `SlicerAngleForm::getAngClicked`.
    pub fn get_ang_clicked(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        let Some((angles, center, time)) = n.top_slicer_angles() else {
            return;
        };
        if time + self.m_time_inc != self.m_cur_time {
            return;
        };
        let Some(index) = self.find_angles(self.m_cur_row[self.m_cur_time as usize]) else {
            return;
        };
        n.undo_model_change();
        self.slicer_ang[index].angles = angles;
        self.slicer_ang[index].center = center;
        n.undo_finish();
        self.load_row(index, self.m_cur_row[self.m_cur_time as usize], true, n)
    }
    /// `SlicerAngleForm::setAngClicked`.
    pub fn set_ang_clicked(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        self.set_angles(true, n)
    }
    /// `SlicerAngleForm::setAngles`.
    pub fn set_angles(&mut self, draw: bool, n: &mut dyn SlicerAngleNativeBoundary) {
        let (time, _) = n.top_slicer_time();
        if time + self.m_time_inc != self.m_cur_time {
            return;
        };
        let Some(index) = self.find_angles(self.m_cur_row[self.m_cur_time as usize]) else {
            return;
        };
        let s = &self.slicer_ang[index];
        n.set_top_slicer_angles(s.angles, s.center, draw)
    }
    /// `SlicerAngleForm::deleteClicked`.
    pub fn delete_clicked(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        let Some(index) = self.find_angles(self.m_cur_row[self.m_cur_time as usize]) else {
            return;
        };
        n.undo_model_change();
        self.slicer_ang.remove(index);
        n.undo_finish();
        self.load_table(self.m_cur_time, n);
        let row = (n.row_count() - 1).min(self.m_cur_row[self.m_cur_time as usize]);
        self.m_cur_row[self.m_cur_time as usize] = row;
        n.select_row(row);
        self.update_enables(n);
        self.update_top_if_continuous(n)
    }
    /// `SlicerAngleForm::newClicked`.
    pub fn new_clicked(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        let Some((angles, center, time)) = n.top_slicer_angles() else {
            return;
        };
        n.undo_model_change();
        self.slicer_ang.push(SlicerAngles {
            angles,
            center,
            time: time + self.m_time_inc,
            label: String::new(),
        });
        n.undo_finish();
        self.load_table(time + self.m_time_inc, n);
        self.m_cur_row[self.m_cur_time as usize] = n.row_count() - 1;
        n.select_row(self.m_cur_row[self.m_cur_time as usize]);
        self.update_enables(n)
    }
    /// `SlicerAngleForm::removeClicked`.
    pub fn remove_clicked(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        n.undo_model_change();
        let mut changed = false;
        for s in &mut self.slicer_ang {
            if s.time > self.m_cur_time {
                s.time -= 1;
                changed = true
            }
        }
        let old = self.slicer_ang.len();
        self.slicer_ang.retain(|s| s.time != self.m_cur_time);
        changed |= old != self.slicer_ang.len();
        self.finish_or_flush_unit(changed, n);
        self.load_table(self.m_cur_time, n);
        self.update_enables(n);
        self.update_top_if_continuous(n)
    }
    /// `SlicerAngleForm::insertClicked`.
    pub fn insert_clicked(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        n.undo_model_change();
        let mut changed = false;
        for s in &mut self.slicer_ang {
            if s.time >= self.m_cur_time {
                s.time += 1;
                changed = true
            }
        }
        self.finish_or_flush_unit(changed, n);
        self.load_table(self.m_cur_time, n);
        self.update_enables(n);
        self.update_top_if_continuous(n)
    }
    /// `SlicerAngleForm::renumberClicked`.
    pub fn renumber_clicked(&mut self, to_time: i32, n: &mut dyn SlicerAngleNativeBoundary) {
        if to_time == self.m_cur_time {
            return;
        }
        let shift = if to_time > self.m_cur_time { -1 } else { 1 };
        n.undo_model_change();
        let mut changed = false;
        for s in &mut self.slicer_ang {
            if s.time == self.m_cur_time {
                s.time = to_time;
                changed = true
            } else if (s.time > self.m_cur_time && s.time <= to_time)
                || (s.time < self.m_cur_time && s.time >= to_time)
            {
                s.time += shift;
                changed = true
            }
        }
        self.finish_or_flush_unit(changed, n);
        self.load_table(self.m_cur_time, n);
        self.update_enables(n);
        self.update_top_if_continuous(n)
    }
    /// `SlicerAngleForm::copyClicked`.
    pub fn copy_clicked(&mut self, to_time: i32, n: &mut dyn SlicerAngleNativeBoundary) {
        if to_time == self.m_cur_time {
            return;
        }
        n.undo_model_change();
        let mut changed = false;
        let old = self.slicer_ang.len();
        self.slicer_ang.retain(|s| s.time != self.m_cur_time);
        changed |= old != self.slicer_ang.len();
        let copied: Vec<_> = self
            .slicer_ang
            .iter()
            .filter(|s| s.time == to_time)
            .cloned()
            .map(|mut s| {
                s.time = self.m_cur_time;
                s
            })
            .collect();
        changed |= !copied.is_empty();
        self.slicer_ang.extend(copied);
        self.finish_or_flush_unit(changed, n);
        self.load_table(self.m_cur_time, n);
        self.update_enables(n);
        self.update_top_if_continuous(n)
    }
    /// `SlicerAngleForm::renumberChanged`.
    pub fn renumber_changed(&mut self, _: i32, n: &mut dyn SlicerAngleNativeBoundary) {
        n.set_focus();
        self.update_enables(n)
    }
    /// `SlicerAngleForm::copyChanged`.
    pub fn copy_changed(&mut self, _: i32, n: &mut dyn SlicerAngleNativeBoundary) {
        n.set_focus();
        self.update_enables(n)
    }
    /// `SlicerAngleForm::helpClicked`.
    pub fn help_clicked(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        n.help("slicerAngles.html#TOP")
    }
    /// `SlicerAngleForm::cellChanged`.
    pub fn cell_changed(
        &mut self,
        row: i32,
        col: i32,
        text: &str,
        n: &mut dyn SlicerAngleNativeBoundary,
    ) {
        let Some(index) = self.find_angles(row) else {
            return;
        };
        n.undo_model_change();
        let s = &mut self.slicer_ang[index];
        let text = text.trim();
        let val = text.parse::<f32>().unwrap_or(0.);
        match col {
            0 => s.angles[0] = val.clamp(-90., 90.),
            1 | 2 => s.angles[col as usize] = val.clamp(-180., 180.),
            3 => s.center.x = val - 1.,
            4 => s.center.y = val - 1.,
            5 => s.center.z = val - 1.,
            _ => s.label = text.chars().take(ANGLE_STRSIZE - 1).collect(),
        };
        n.undo_finish();
        n.focus_table();
        n.select_row(row);
        self.m_ignore_cur_chg = true;
        if self.m_cur_row[self.m_cur_time as usize] == row {
            self.update_top_if_continuous(n)
        }
    }
    /// `SlicerAngleForm::selectionChanged`.
    pub fn selection_changed(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        if self.m_ignore_cur_chg {
            self.m_ignore_cur_chg = false;
            return;
        }
        self.m_cur_row[self.m_cur_time as usize] = n.current_row();
        self.update_top_if_continuous(n)
    }
    /// `SlicerAngleForm::updateTopIfContinuous`.
    pub fn update_top_if_continuous(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        let (time, continuous) = n.top_slicer_time();
        if time + self.m_time_inc == self.m_cur_time && continuous {
            self.set_angles(true, n)
        }
    }
    /// `SlicerAngleForm::loadTable`.
    pub fn load_table(&mut self, time: i32, n: &mut dyn SlicerAngleNativeBoundary) {
        let rows: Vec<_> = self
            .slicer_ang
            .iter()
            .filter(|s| s.time == time)
            .cloned()
            .collect();
        n.set_table_rows(&rows);
        self.m_cur_time = time
    }
    /// `SlicerAngleForm::loadRow`.
    pub fn load_row(
        &mut self,
        index: usize,
        row: i32,
        block: bool,
        n: &mut dyn SlicerAngleNativeBoundary,
    ) {
        n.set_table_row(row, &self.slicer_ang[index], block)
    }
    /// `SlicerAngleForm::switchTime`.
    pub fn switch_time(
        &mut self,
        newtime: i32,
        do_set: bool,
        n: &mut dyn SlicerAngleNativeBoundary,
    ) {
        self.load_table(newtime, n);
        self.update_enables(n);
        let row = self.m_cur_row[self.m_cur_time as usize].min(n.row_count() - 1);
        self.m_cur_row[self.m_cur_time as usize] = row;
        if row >= 0 {
            n.select_row(row);
            if do_set {
                self.update_top_if_continuous(n)
            }
        }
        self.set_time_label(n)
    }
    /// `SlicerAngleForm::findAngles`.
    pub fn find_angles(&self, row: i32) -> Option<usize> {
        self.slicer_ang
            .iter()
            .enumerate()
            .filter(|(_, s)| s.time == self.m_cur_time)
            .nth(row.max(0) as usize)
            .map(|(i, _)| i)
    }
    /// `SlicerAngleForm::finishOrFlushUnit`.
    pub fn finish_or_flush_unit(&mut self, changed: bool, n: &mut dyn SlicerAngleNativeBoundary) {
        if changed {
            n.undo_finish()
        } else {
            n.undo_flush()
        }
    }
    /// `SlicerAngleForm::newTime`.
    pub fn new_time(&mut self, refresh: bool, n: &mut dyn SlicerAngleNativeBoundary) {
        let (time, _) = n.top_slicer_time();
        let time = if time < 0 { n.display_time() } else { time } + self.m_time_inc;
        if time != self.m_cur_time || refresh {
            self.switch_time(time, true, n)
        }
    }
    /// `SlicerAngleForm::topSlicerDrawing`.
    pub fn top_slicer_drawing(
        &mut self,
        angles: [f32; 3],
        center: Ipoint,
        time: i32,
        dragging: i32,
        continuous: bool,
        n: &mut dyn SlicerAngleNativeBoundary,
    ) {
        if time + self.m_time_inc != self.m_cur_time {
            self.switch_time(time + self.m_time_inc, false, n)
        }
        if continuous {
            if n.row_count() == 0 {
                self.new_clicked(n)
            }
            if let Some(index) = self.find_angles(self.m_cur_row[self.m_cur_time as usize]) {
                if self.slicer_ang[index].angles != angles
                    || self.slicer_ang[index].center != center
                {
                    if self.last_dragging == 0 {
                        n.undo_model_change()
                    }
                    self.slicer_ang[index].angles = angles;
                    self.slicer_ang[index].center = center;
                    self.load_row(index, self.m_cur_row[self.m_cur_time as usize], true, n);
                    if dragging == 0 {
                        n.undo_finish()
                    }
                } else if self.last_dragging != 0 && dragging == 0 {
                    n.undo_finish()
                }
            }
        }
        self.last_dragging = dragging
    }
    /// `SlicerAngleForm::closeEvent`.
    pub fn close_event(&mut self, n: &mut dyn SlicerAngleNativeBoundary) {
        n.slicer_angles_closing();
        n.accept_close()
    }
    /// `SlicerAngleForm::keyPressEvent`.
    pub fn key_press_event(&mut self, navigation: bool, n: &mut dyn SlicerAngleNativeBoundary) {
        if navigation {
            return;
        }
        if n.close_key() {
            n.close()
        } else {
            n.ivw_control_key()
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        rows: Vec<SlicerAngles>,
        time: i32,
        continuous: bool,
        undos: Vec<&'static str>,
        spins: [i32; 2],
        enabled: Vec<(i32, bool)>,
    }
    impl SlicerAngleNativeBoundary for N {
        fn setup_ui(&mut self) {}
        fn retranslate_ui(&mut self) {}
        fn set_delete_on_close(&mut self) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn connect_slicer_angle_signals(&mut self) {}
        fn set_spin_maximum(&mut self, _: i32, _: i32) {}
        fn spin_value(&self, which: i32) -> i32 {
            self.spins[which as usize]
        }
        fn set_button_enabled(&mut self, which: i32, value: bool) {
            self.enabled.push((which, value))
        }
        fn hide_control(&mut self, _: i32) {}
        fn resize_for_no_volume_group(&mut self) {}
        fn set_focus(&mut self) {}
        fn rounded_style(&self) -> bool {
            false
        }
        fn set_button_width(&mut self, _: i32, _: bool, _: f32, _: &str) -> i32 {
            0
        }
        fn set_button_fixed_width(&mut self, _: i32, _: i32) {}
        fn table_font_width(&self, _: &str) -> i32 {
            0
        }
        fn set_table_column_width(&mut self, _: i32, _: i32) {}
        fn volume_dimensions(&self) -> (i32, i32, i32) {
            (0, 0, 0)
        }
        fn set_time_label(&mut self, _: &str) {}
        fn select_row(&mut self, _: i32) {}
        fn row_count(&self) -> i32 {
            self.rows.len() as i32
        }
        fn set_table_rows(&mut self, x: &[SlicerAngles]) {
            self.rows = x.into()
        }
        fn set_table_row(&mut self, _: i32, _: &SlicerAngles, _: bool) {}
        fn current_row(&self) -> i32 {
            0
        }
        fn focus_table(&mut self) {}
        fn close(&mut self) {}
        fn accept_close(&mut self) {}
        fn help(&mut self, _: &str) {}
        fn top_slicer_time(&self) -> (i32, bool) {
            (self.time, self.continuous)
        }
        fn top_slicer_angles(&self) -> Option<([f32; 3], Ipoint, i32)> {
            Some(([1., 2., 3.], Ipoint::default(), self.time))
        }
        fn set_top_slicer_angles(&mut self, _: [f32; 3], _: Ipoint, _: bool) {}
        fn display_time(&self) -> i32 {
            0
        }
        fn time_label(&self, _: i32) -> String {
            String::new()
        }
        fn undo_model_change(&mut self) {
            self.undos.push("change")
        }
        fn undo_finish(&mut self) {
            self.undos.push("finish")
        }
        fn undo_flush(&mut self) {
            self.undos.push("flush")
        }
        fn slicer_angles_closing(&mut self) {}
        fn close_key(&self) -> bool {
            false
        }
        fn ivw_control_key(&mut self) {}
        fn check_and_set_mac_menu(&mut self) {}
        fn widget_change_event(&mut self) {}
    }
    #[test]
    fn insert_remove_and_renumber_follow_source() {
        let mut n = N {
            time: 1,
            ..Default::default()
        };
        let mut f = SlicerAngleForm::new(
            vec![
                SlicerAngles {
                    time: 1,
                    ..Default::default()
                },
                SlicerAngles {
                    time: 2,
                    ..Default::default()
                },
            ],
            2,
            &mut n,
        );
        f.insert_clicked(&mut n);
        assert_eq!(f.slicer_ang[0].time, 2);
        f.m_cur_time = 2;
        f.renumber_clicked(1, &mut n);
        assert!(f.slicer_ang.iter().any(|s| s.time == 1));
        f.remove_clicked(&mut n);
        assert!(f.slicer_ang.iter().all(|s| s.time != 2));
    }
    #[test]
    fn cell_limits_angles_and_offsets_centers() {
        let mut n = N {
            time: 1,
            ..Default::default()
        };
        let mut f = SlicerAngleForm::new(
            vec![SlicerAngles {
                time: 1,
                ..Default::default()
            }],
            1,
            &mut n,
        );
        f.cell_changed(0, 0, "100", &mut n);
        f.cell_changed(0, 3, "4", &mut n);
        assert_eq!(f.slicer_ang[0].angles[0], 90.);
        assert_eq!(f.slicer_ang[0].center.x, 3.);
    }
    #[test]
    fn enable_state_uses_the_source_spin_values() {
        let mut n = N {
            time: 1,
            spins: [2, 1],
            ..Default::default()
        };
        let mut f = SlicerAngleForm::new(vec![], 2, &mut n);
        n.enabled.clear();
        f.update_enables(&mut n);
        assert!(n.enabled.contains(&(RENUMBER_BUTTON, true)));
        assert!(n.enabled.contains(&(COPY_BUTTON, false)));
    }
}
