//! Translation of `IMOD/3dmod/form_mouse.cpp` and `form_mouse.h`.
//!
//! QButtonGroup/widget installation is an explicit native-GUI boundary. The
//! source preference mutations and exact mapping-to-label tables are here.
#![allow(dead_code)]
use core::ffi::c_void;

#[derive(Clone, Debug, Default)]
pub struct ImodPrefStruct {
    pub hot_slider_key: i32,
    pub hot_slider_flag: i32,
    pub mouse_mapping: i32,
    pub modv_swap_left_mid: bool,
}
pub trait MouseNativeBoundary {
    fn ctrl_string(&self) -> &str;
    fn set_ctrl_text(&mut self, text: &str);
    fn set_ctrl_tool_tip(&mut self, text: &str);
    fn create_group(&mut self) -> *mut c_void;
    fn group_add_button(&mut self, group: *mut c_void, id: i32);
    fn connect_groups(&mut self);
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
        self.hot_key_group = native.create_group();
        for id in 0..3 {
            native.group_add_button(self.hot_key_group, id)
        }
        self.active_group = native.create_group();
        for id in 0..3 {
            native.group_add_button(self.active_group, id)
        }
        self.mouse_group = native.create_group();
        for id in 0..6 {
            native.group_add_button(self.mouse_group, id)
        }
        native.connect_groups();
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
    /// Static `MouseForm::setMouseLabels`.
    pub fn set_mouse_labels(&mut self, native: &mut dyn MouseNativeBoundary) {
        let texts = [
            "Pan, mark\nAttach to pt",
            "Movie up\nAdd point",
            "Movie down\nModify pt",
        ];
        let left = [0, 2, 1, 0, 1, 2];
        let mid = [1, 0, 2, 2, 0, 1];
        let right = [2, 1, 0, 1, 2, 0];
        let index = self.m_prefs.mouse_mapping.clamp(0, 5) as usize;
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
    }
    impl MouseNativeBoundary for N {
        fn ctrl_string(&self) -> &str {
            "Ctrl"
        }
        fn set_ctrl_text(&mut self, _: &str) {}
        fn set_ctrl_tool_tip(&mut self, _: &str) {}
        fn create_group(&mut self) -> *mut c_void {
            core::ptr::dangling_mut()
        }
        fn group_add_button(&mut self, _: *mut c_void, _: i32) {}
        fn connect_groups(&mut self) {}
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
}
