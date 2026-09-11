//! Translation of `IMOD/3dmod/form_behavior.cpp` together with its header.
//!
//! The native Qt group/buttons and platform environment are direct boundary
//! operations.  The preference fields and form algorithms retain source order.
#![allow(dead_code)]

use core::ffi::c_void;

/// Form-used fields of upstream `imod_pref_struct`.
#[derive(Clone, Debug, Default)]
pub struct ImodPrefStruct {
    pub allow_ctrl_on_mac: bool,
    pub silent_beep: bool,
    pub classic_slicer: bool,
    pub start_in_hq: bool,
    pub arrows_scroll_zap: bool,
    pub start_at_mid_z: bool,
    pub attach_to_on_obj: bool,
    pub slicer_new_surf: bool,
    pub bw_step: i32,
    pub page_step: i32,
    pub autosave_on: bool,
    pub autosave_no_cont_mesh: bool,
    pub autosave_no_iso_mesh: bool,
    pub autosave_interval: i32,
    pub autosave_dir: String,
    pub key_sets_hw_stereo: bool,
    pub no_vert_buf_for_cont: bool,
    pub no_vbo_for_sphere: bool,
    pub exit_when_all_closed: i32,
}

/// Native Qt widgets and path conversion called directly from this unit.
pub trait BehaviorNativeBoundary {
    fn create_exit_group(&mut self) -> *mut c_void;
    fn exit_group_add_button(&mut self, group: *mut c_void, id: i32);
    fn hide_allow_ctrl_on_mac(&mut self);
    fn set_allow_ctrl_enabled(&mut self, enabled: bool);
    fn macos(&self) -> bool;
    fn mac_ctrl_environment_set(&self) -> bool;
    fn connect_exit_group(&mut self);
    fn font_width(&self, text: &str) -> i32;
    fn set_autosave_spin_maximum_width(&mut self, width: i32);
    fn set_checked(&mut self, which: i32, value: bool);
    fn set_spin_box(&mut self, which: i32, value: i32);
    fn set_autosave_dir(&mut self, path: &str);
    fn set_exit_group(&mut self, group: *mut c_void, id: i32);
    fn checked(&self, which: i32) -> bool;
    fn spin_box_value(&self, which: i32) -> i32;
    fn autosave_dir(&self) -> String;
    fn clean_path(&self, path: &str) -> String;
    fn retranslate_ui(&mut self);
}
pub const ALLOW_CTRL: i32 = 0;
pub const SILENCE: i32 = 1;
pub const CLASSIC: i32 = 2;
pub const START_HQ: i32 = 3;
pub const ARROWS: i32 = 4;
pub const MID_Z: i32 = 5;
pub const SELECT_ON: i32 = 6;
pub const SLICER_SURF: i32 = 7;
pub const AUTOSAVE: i32 = 8;
pub const OMIT_CONT: i32 = 9;
pub const OMIT_ISO: i32 = 10;
pub const KEY_HW: i32 = 11;
pub const NO_VB_CONT: i32 = 12;
pub const NO_VB_SPHERE: i32 = 13;
pub const BW_STEP: i32 = 0;
pub const PAGE_STEP: i32 = 1;
pub const AUTOSAVE_INTERVAL: i32 = 2;

/// `BehaviorForm` (`form_behavior.h`).
#[derive(Debug)]
pub struct BehaviorForm {
    pub m_prefs: ImodPrefStruct,
    pub exit_group: *mut c_void,
}
impl BehaviorForm {
    /// `BehaviorForm::BehaviorForm`.
    pub fn new(prefs: ImodPrefStruct, native: &mut dyn BehaviorNativeBoundary) -> Self {
        let mut form = Self {
            m_prefs: prefs,
            exit_group: core::ptr::null_mut(),
        };
        form.init(native);
        form
    }
    /// `BehaviorForm::~BehaviorForm`.
    pub fn destroy(&mut self) {}
    /// `BehaviorForm::languageChange`.
    pub fn language_change(&mut self, native: &mut dyn BehaviorNativeBoundary) {
        native.retranslate_ui()
    }
    /// `BehaviorForm::init`.
    pub fn init(&mut self, native: &mut dyn BehaviorNativeBoundary) {
        self.exit_group = native.create_exit_group();
        native.exit_group_add_button(self.exit_group, 0);
        native.exit_group_add_button(self.exit_group, 1);
        native.exit_group_add_button(self.exit_group, 2);
        if !native.macos() {
            native.hide_allow_ctrl_on_mac()
        }
        native.set_allow_ctrl_enabled(!native.mac_ctrl_environment_set());
        native.connect_exit_group();
        self.set_font_dependent_widths(native);
        self.update(native);
    }
    /// `BehaviorForm::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self, native: &mut dyn BehaviorNativeBoundary) {
        native.set_autosave_spin_maximum_width((6 * 2 + 3) * native.font_width("999999") / (6 * 2));
    }
    /// `BehaviorForm::exitTypeChanged`.
    pub fn exit_type_changed(&mut self, value: i32) {
        self.m_prefs.exit_when_all_closed = value - 1;
    }
    /// `BehaviorForm::update`.
    pub fn update(&mut self, native: &mut dyn BehaviorNativeBoundary) {
        let p = &self.m_prefs;
        native.set_checked(ALLOW_CTRL, p.allow_ctrl_on_mac);
        native.set_checked(SILENCE, p.silent_beep);
        native.set_checked(CLASSIC, p.classic_slicer);
        native.set_checked(START_HQ, p.start_in_hq);
        native.set_checked(ARROWS, p.arrows_scroll_zap);
        native.set_checked(MID_Z, p.start_at_mid_z);
        native.set_checked(SELECT_ON, p.attach_to_on_obj);
        native.set_checked(SLICER_SURF, p.slicer_new_surf);
        native.set_spin_box(BW_STEP, p.bw_step);
        native.set_spin_box(PAGE_STEP, p.page_step);
        native.set_checked(AUTOSAVE, p.autosave_on);
        native.set_checked(OMIT_CONT, p.autosave_no_cont_mesh);
        native.set_checked(OMIT_ISO, p.autosave_no_iso_mesh);
        native.set_spin_box(AUTOSAVE_INTERVAL, p.autosave_interval);
        native.set_autosave_dir(&p.autosave_dir);
        native.set_checked(KEY_HW, p.key_sets_hw_stereo);
        native.set_checked(NO_VB_CONT, p.no_vert_buf_for_cont);
        native.set_checked(NO_VB_SPHERE, p.no_vbo_for_sphere);
        native.set_exit_group(self.exit_group, p.exit_when_all_closed + 1);
    }
    /// `BehaviorForm::unload`.
    pub fn unload(&mut self, native: &mut dyn BehaviorNativeBoundary) {
        let p = &mut self.m_prefs;
        p.allow_ctrl_on_mac = native.checked(ALLOW_CTRL);
        p.silent_beep = native.checked(SILENCE);
        p.classic_slicer = native.checked(CLASSIC);
        p.start_in_hq = native.checked(START_HQ);
        p.arrows_scroll_zap = native.checked(ARROWS);
        p.start_at_mid_z = native.checked(MID_Z);
        p.attach_to_on_obj = native.checked(SELECT_ON);
        p.slicer_new_surf = native.checked(SLICER_SURF);
        p.bw_step = native.spin_box_value(BW_STEP);
        p.page_step = native.spin_box_value(PAGE_STEP);
        p.autosave_on = native.checked(AUTOSAVE);
        p.autosave_no_cont_mesh = native.checked(OMIT_CONT);
        p.autosave_no_iso_mesh = native.checked(OMIT_ISO);
        p.autosave_interval = native.spin_box_value(AUTOSAVE_INTERVAL);
        p.key_sets_hw_stereo = native.checked(KEY_HW);
        p.no_vert_buf_for_cont = native.checked(NO_VB_CONT);
        p.no_vbo_for_sphere = native.checked(NO_VB_SPHERE);
        p.autosave_dir = native.clean_path(&native.autosave_dir());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        checks: [bool; 14],
        spins: [i32; 3],
        path: String,
        calls: Vec<String>,
        group: u8,
    }
    impl BehaviorNativeBoundary for N {
        fn create_exit_group(&mut self) -> *mut c_void {
            &mut self.group as *mut u8 as *mut c_void
        }
        fn exit_group_add_button(&mut self, _: *mut c_void, id: i32) {
            self.calls.push(format!("button:{id}"))
        }
        fn hide_allow_ctrl_on_mac(&mut self) {
            self.calls.push("hide".into())
        }
        fn set_allow_ctrl_enabled(&mut self, x: bool) {
            self.calls.push(format!("enabled:{x}"))
        }
        fn macos(&self) -> bool {
            false
        }
        fn mac_ctrl_environment_set(&self) -> bool {
            false
        }
        fn connect_exit_group(&mut self) {}
        fn font_width(&self, _: &str) -> i32 {
            12
        }
        fn set_autosave_spin_maximum_width(&mut self, _: i32) {}
        fn set_checked(&mut self, x: i32, y: bool) {
            self.checks[x as usize] = y
        }
        fn set_spin_box(&mut self, x: i32, y: i32) {
            self.spins[x as usize] = y
        }
        fn set_autosave_dir(&mut self, x: &str) {
            self.path = x.into()
        }
        fn set_exit_group(&mut self, _: *mut c_void, x: i32) {
            self.calls.push(format!("group:{x}"))
        }
        fn checked(&self, x: i32) -> bool {
            self.checks[x as usize]
        }
        fn spin_box_value(&self, x: i32) -> i32 {
            self.spins[x as usize]
        }
        fn autosave_dir(&self) -> String {
            self.path.clone()
        }
        fn clean_path(&self, x: &str) -> String {
            x.replace("//", "/")
        }
        fn retranslate_ui(&mut self) {}
    }
    #[test]
    fn exit_mapping_matches_source() {
        let mut n = N::default();
        let mut f = BehaviorForm::new(ImodPrefStruct::default(), &mut n);
        f.exit_type_changed(0);
        assert_eq!(f.m_prefs.exit_when_all_closed, -1);
        f.exit_type_changed(2);
        assert_eq!(f.m_prefs.exit_when_all_closed, 1);
    }
    #[test]
    fn unload_uses_native_clean_path() {
        let mut n = N::default();
        let mut f = BehaviorForm::new(ImodPrefStruct::default(), &mut n);
        n.checks[AUTOSAVE as usize] = true;
        n.spins[AUTOSAVE_INTERVAL as usize] = 15;
        n.path = "a//b".into();
        f.unload(&mut n);
        assert!(f.m_prefs.autosave_on);
        assert_eq!(f.m_prefs.autosave_interval, 15);
        assert_eq!(f.m_prefs.autosave_dir, "a/b");
    }
}
