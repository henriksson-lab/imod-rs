//! Translation of `IMOD/3dmod/form_mouse.cpp` and `form_mouse.h`.
//!
//! QButtonGroup/widget installation is an explicit native-GUI boundary. The
//! source preference mutations and exact mapping-to-label tables are here.
#![allow(dead_code)]
use core::ffi::c_void;

/// `imod_pref_struct` fields read and written by `MouseForm`.
#[derive(Clone, Debug, Default)]
pub struct ImodPrefStruct {
    pub hot_slider_key: i32,
    pub hot_slider_flag: i32,
    pub mouse_mapping: i32,
    pub modv_swap_left_mid: bool,
}

/// The `Ui::MouseForm` controls passed to `QButtonGroup::addButton` by this
/// source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MouseFormButton {
    CtrlRadioButton,
    ShiftRadioButton,
    AltRadioButton,
    KeyUpRadio,
    KeyDownRadio,
    NeverRadio,
    LmrRadioButton,
    MrlRadioButton,
    RlmRadioButton,
    LrmRadioButton,
    MlrRadioButton,
    RmlRadioButton,
}

/// Direct Qt widgets, `QButtonGroup`, `connect`, and `dia_qtutils` calls.
/// This unit owns neither a substitute UI nor event loop.
pub trait MouseNativeBoundary {
    fn ctrl_string(&self) -> &str;
    fn set_ctrl_text(&mut self, text: &str);
    fn set_ctrl_tool_tip(&mut self, text: &str);
    fn create_button_group(&mut self) -> *mut c_void;
    fn group_add_button(&mut self, group: *mut c_void, button: MouseFormButton, id: i32);
    fn connect_group_clicked(&mut self, group: *mut c_void);
    fn connect_swap_toggled(&mut self);
    fn set_group(&mut self, group: *mut c_void, value: i32);
    fn set_swap_checked(&mut self, value: bool);
    fn set_left_label(&mut self, text: &str);
    fn set_middle_label(&mut self, text: &str);
    fn set_right_label(&mut self, text: &str);
    fn retranslate_ui(&mut self);
}
/// `MouseForm` (`form_mouse.h`).
#[derive(Debug)]
pub struct MouseForm {
    pub m_prefs: ImodPrefStruct,
    pub hot_key_group: *mut c_void,
    pub active_group: *mut c_void,
    pub mouse_group: *mut c_void,
}
impl MouseForm {
    /// `MouseForm::MouseForm`.
    pub fn new(prefs: ImodPrefStruct, native: &mut dyn MouseNativeBoundary) -> Self {
        let mut form = Self {
            m_prefs: prefs,
            hot_key_group: core::ptr::null_mut(),
            active_group: core::ptr::null_mut(),
            mouse_group: core::ptr::null_mut(),
        };
        form.init(native);
        form
    }
    /// `MouseForm::~MouseForm`.
    pub fn destroy(&mut self) {}
    /// `MouseForm::languageChange`.
    pub fn language_change(&mut self, native: &mut dyn MouseNativeBoundary) {
        native.retranslate_ui()
    }
    /// `MouseForm::init`.
    pub fn init(&mut self, native: &mut dyn MouseNativeBoundary) {
        let ctrl = native.ctrl_string().to_owned();
        native.set_ctrl_text(&ctrl);
        native.set_ctrl_tool_tip(&format!(
            "Make {ctrl} key control whether sliders are continuously active"
        ));
        self.hot_key_group = native.create_button_group();
        native.group_add_button(self.hot_key_group, MouseFormButton::CtrlRadioButton, 0);
        native.group_add_button(self.hot_key_group, MouseFormButton::ShiftRadioButton, 1);
        native.group_add_button(self.hot_key_group, MouseFormButton::AltRadioButton, 2);
        self.active_group = native.create_button_group();
        native.group_add_button(self.active_group, MouseFormButton::KeyUpRadio, 0);
        native.group_add_button(self.active_group, MouseFormButton::KeyDownRadio, 1);
        native.group_add_button(self.active_group, MouseFormButton::NeverRadio, 2);
        self.mouse_group = native.create_button_group();
        native.group_add_button(self.mouse_group, MouseFormButton::LmrRadioButton, 0);
        native.group_add_button(self.mouse_group, MouseFormButton::MrlRadioButton, 1);
        native.group_add_button(self.mouse_group, MouseFormButton::RlmRadioButton, 2);
        native.group_add_button(self.mouse_group, MouseFormButton::LrmRadioButton, 3);
        native.group_add_button(self.mouse_group, MouseFormButton::MlrRadioButton, 4);
        native.group_add_button(self.mouse_group, MouseFormButton::RmlRadioButton, 5);
        native.connect_group_clicked(self.hot_key_group);
        native.connect_group_clicked(self.active_group);
        native.connect_group_clicked(self.mouse_group);
        native.connect_swap_toggled();
        self.set_mouse_labels(native);
        self.update(native);
    }
    /// `MouseForm::update`.
    pub fn update(&mut self, native: &mut dyn MouseNativeBoundary) {
        native.set_group(self.hot_key_group, self.m_prefs.hot_slider_key);
        native.set_group(self.active_group, self.m_prefs.hot_slider_flag);
        native.set_group(self.mouse_group, self.m_prefs.mouse_mapping);
        native.set_swap_checked(self.m_prefs.modv_swap_left_mid);
    }
    /// `MouseForm::flagChanged`.
    pub fn flag_changed(&mut self, value: i32) {
        self.m_prefs.hot_slider_flag = value
    }
    /// `MouseForm::keyChanged`.
    pub fn key_changed(&mut self, value: i32) {
        self.m_prefs.hot_slider_key = value
    }
    /// `MouseForm::mappingChanged`.
    pub fn mapping_changed(&mut self, value: i32, native: &mut dyn MouseNativeBoundary) {
        self.m_prefs.mouse_mapping = value;
        self.set_mouse_labels(native)
    }
    /// `MouseForm::swapToggled`.
    pub fn swap_toggled(&mut self, state: bool) {
        self.m_prefs.modv_swap_left_mid = state
    }
    /// `MouseForm::setMouseLabels`.
    pub fn set_mouse_labels(&mut self, native: &mut dyn MouseNativeBoundary) {
        let texts = [
            "Pan, mark\nAttach to pt",
            "Movie up\nAdd point",
            "Movie down\nModify pt",
        ];
        let left = [0, 2, 1, 0, 1, 2];
        let mid = [1, 0, 2, 2, 0, 1];
        let right = [2, 1, 0, 1, 2, 0];
        let index = self.m_prefs.mouse_mapping as usize;
        native.set_left_label(texts[left[index]]);
        native.set_middle_label(texts[mid[index]]);
        native.set_right_label(texts[right[index]]);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        labels: [String; 3],
        groups: Vec<i32>,
        buttons: Vec<(MouseFormButton, i32)>,
        group_connections: usize,
        swap_connection: bool,
    }
    impl MouseNativeBoundary for N {
        fn ctrl_string(&self) -> &str {
            "Ctrl"
        }
        fn set_ctrl_text(&mut self, _: &str) {}
        fn set_ctrl_tool_tip(&mut self, _: &str) {}
        fn create_button_group(&mut self) -> *mut c_void {
            core::ptr::dangling_mut()
        }
        fn group_add_button(&mut self, _: *mut c_void, button: MouseFormButton, id: i32) {
            self.buttons.push((button, id));
        }
        fn connect_group_clicked(&mut self, _: *mut c_void) {
            self.group_connections += 1;
        }
        fn connect_swap_toggled(&mut self) {
            self.swap_connection = true;
        }
        fn set_group(&mut self, _: *mut c_void, x: i32) {
            self.groups.push(x)
        }
        fn set_swap_checked(&mut self, _: bool) {}
        fn set_left_label(&mut self, x: &str) {
            self.labels[0] = x.into()
        }
        fn set_middle_label(&mut self, x: &str) {
            self.labels[1] = x.into()
        }
        fn set_right_label(&mut self, x: &str) {
            self.labels[2] = x.into()
        }
        fn retranslate_ui(&mut self) {}
    }
    #[test]
    fn mapping_labels_match_upstream_tables() {
        let mut n = N::default();
        let mut f = MouseForm::new(ImodPrefStruct::default(), &mut n);
        f.mapping_changed(1, &mut n);
        assert_eq!(
            n.labels,
            [
                "Movie down\nModify pt",
                "Pan, mark\nAttach to pt",
                "Movie up\nAdd point"
            ]
        );
    }
    #[test]
    fn state_handlers_only_change_corresponding_pref() {
        let mut n = N::default();
        let mut f = MouseForm::new(ImodPrefStruct::default(), &mut n);
        f.flag_changed(2);
        f.key_changed(1);
        f.swap_toggled(true);
        assert_eq!(
            (
                f.m_prefs.hot_slider_flag,
                f.m_prefs.hot_slider_key,
                f.m_prefs.modv_swap_left_mid
            ),
            (2, 1, true)
        );
    }

    #[test]
    fn init_installs_the_source_button_groups_and_connections() {
        let mut n = N::default();
        let _form = MouseForm::new(ImodPrefStruct::default(), &mut n);
        assert_eq!(n.buttons.len(), 12);
        assert_eq!(
            n.buttons,
            [
                (MouseFormButton::CtrlRadioButton, 0),
                (MouseFormButton::ShiftRadioButton, 1),
                (MouseFormButton::AltRadioButton, 2),
                (MouseFormButton::KeyUpRadio, 0),
                (MouseFormButton::KeyDownRadio, 1),
                (MouseFormButton::NeverRadio, 2),
                (MouseFormButton::LmrRadioButton, 0),
                (MouseFormButton::MrlRadioButton, 1),
                (MouseFormButton::RlmRadioButton, 2),
                (MouseFormButton::LrmRadioButton, 3),
                (MouseFormButton::MlrRadioButton, 4),
                (MouseFormButton::RmlRadioButton, 5),
            ]
        );
        assert_eq!(n.group_connections, 3);
        assert!(n.swap_connection);
        assert_eq!(n.groups, [0, 0, 0]);
    }
}
