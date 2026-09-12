//! Translation of `IMOD/3dmod/rotationtool.cpp` and `rotationtool.h`.
#![allow(dead_code)]
pub const FILE_LIST: [&str; 8] = [
    ":/images/plusSign.png",
    ":/images/upRotArrow.png",
    ":/images/rotCCWarrow.png",
    ":/images/leftRotArrow.png",
    ":/images/rightRotArrow.png",
    ":/images/minusSign.png",
    ":/images/downRotArrow.png",
    ":/images/rotCWarrow.png",
];
pub const BUTTON_TIPS: [&str; 8] = [
    "Increase step size",
    "Rotate clockwise around current X axis",
    "Rotate counterclockwise around current Z axis",
    "Rotate clockwise around current Y axis",
    "Rotate counterclockwise around current Y axis",
    "Decrease step size",
    "Rotate counterclockwise around current X axis",
    "Rotate clockwise around current Z axis",
];
pub const STEP_SIGN: [i32; 8] = [1, 0, 0, 0, 0, -1, 0, 0];
pub const ROT_STEPS: [(i32, i32, i32); 8] = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, 0),
    (-1, 0, 0),
    (0, 0, -1),
];
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RotationToolButton {
    pub row: i32,
    pub column: i32,
    pub auto_raise: bool,
    pub fixed_size: i32,
    pub no_focus: bool,
    pub checkable: bool,
    pub checked: bool,
    pub tooltip: Option<String>,
    pub icon: Option<String>,
    pub mapped_index: Option<usize>,
    pub auto_repeat: bool,
    pub auto_repeat_delay: Option<i32>,
    pub auto_repeat_interval: Option<i32>,
}
/// Qt icons/tool buttons/layouts remain the named legacy GUI boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct RotationTool {
    pub icons: Option<[String; 8]>,
    pub first_time: bool,
    pub buttons: Vec<RotationToolButton>,
    pub center_button: Option<usize>,
    pub icon_size: i32,
    pub center_exists: bool,
    pub center_checked: bool,
    pub center_tooltip: Option<String>,
    pub step_label: Option<String>,
    pub auto_raise: bool,
    pub grid_spacing: i32,
    pub rotation_signal: Option<(i32, i32, i32)>,
    pub step_signal: Option<i32>,
    pub center_signal: Option<bool>,
    pub closing_signal: bool,
    pub key_press_signal: bool,
    pub key_release_signal: bool,
}
impl RotationTool {
    pub fn new(
        center_icon: bool,
        center_tip: Option<&str>,
        size: i32,
        auto_raise: bool,
        step_size: f32,
    ) -> Self {
        let mut buttons = Vec::new();
        let mut index = 0;
        let mut center_button = None;
        for row in 0..3 {
            for column in 0..3 {
                if row == 1 && column == 1 {
                    center_button = Some(buttons.len());
                    buttons.push(RotationToolButton {
                        row,
                        column,
                        auto_raise,
                        fixed_size: size,
                        no_focus: true,
                        checkable: true,
                        checked: false,
                        tooltip: center_tip.map(String::from),
                        icon: center_icon.then(|| "center".into()),
                        mapped_index: None,
                        auto_repeat: false,
                        auto_repeat_delay: None,
                        auto_repeat_interval: None,
                    });
                } else {
                    let repeat = STEP_SIGN[index] == 0;
                    buttons.push(RotationToolButton {
                        row,
                        column,
                        auto_raise,
                        fixed_size: size,
                        no_focus: true,
                        checkable: false,
                        checked: false,
                        tooltip: Some(BUTTON_TIPS[index].into()),
                        icon: Some(FILE_LIST[index].into()),
                        mapped_index: Some(index),
                        auto_repeat: repeat,
                        auto_repeat_delay: repeat.then_some(300),
                        auto_repeat_interval: repeat.then_some(100),
                    });
                    index += 1;
                }
            }
        }
        Self {
            icons: Some(FILE_LIST.map(String::from)),
            first_time: false,
            buttons,
            center_button,
            icon_size: size - 4,
            center_exists: true,
            center_checked: false,
            center_tooltip: center_tip.map(String::from),
            step_label: if step_size >= 0.0 {
                Some(format!("Step: {}", step_size))
            } else {
                None
            },
            auto_raise,
            grid_spacing: if auto_raise { 0 } else { 4 },
            rotation_signal: None,
            step_signal: None,
            center_signal: None,
            closing_signal: false,
            key_press_signal: false,
            key_release_signal: false,
        }
    }
    pub fn set_center_state(&mut self, state: bool) {
        if self.center_exists {
            self.center_checked = state
        }
    }
    pub fn set_step_label(&mut self, step: f32) {
        if self.step_label.is_some() {
            self.step_label = Some(format!("Step: {}", step))
        }
    }
    pub fn center_toggled(&mut self, state: bool) {
        self.center_signal = Some(state)
    }
    pub fn button_clicked(&mut self, which: usize) {
        if which >= STEP_SIGN.len() {
            return;
        }
        if STEP_SIGN[which] != 0 {
            self.step_signal = Some(STEP_SIGN[which])
        } else {
            self.rotation_signal = Some(ROT_STEPS[which])
        }
    }
    pub fn close_event(&mut self) {
        self.closing_signal = true
    }
    pub fn key_press_event(&mut self) {
        self.key_press_signal = true
    }
    pub fn key_release_event(&mut self) {
        self.key_release_signal = true
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slots_emit_source_vectors() {
        let mut x = RotationTool::new(false, None, 20, true, 2.0);
        x.button_clicked(0);
        assert_eq!(x.step_signal, Some(1));
        x.button_clicked(3);
        assert_eq!(x.rotation_signal, Some((0, -1, 0)));
        x.set_center_state(true);
        x.center_toggled(true);
        assert_eq!(x.center_signal, Some(true));
    }
}
