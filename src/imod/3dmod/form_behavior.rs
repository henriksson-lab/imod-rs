//! Translation of `IMOD/3dmod/form_behavior.cpp` and `form_behavior.h`.
#![allow(dead_code)]

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitButton {
    AskExitRadio,
    NeverExitRadio,
    AlwaysExitRadio,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BehaviorCheckBox {
    AllowCtrlOnMacBox,
    SilenceBox,
    ClassicBox,
    StartInHqBox,
    ArrowsScrollZapBox,
    StartAtMidZBox,
    SelectOnCheckBox,
    SlicerNewSurfBox,
    AutosaveEnabledBox,
    OmitContMeshBox,
    OmitIsosurfMeshBox,
    KeyHwStereoBox,
    NoVbForContBox,
    NoVbForSphereBox,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BehaviorSpinBox {
    F1f8StepSpinBox,
    PageStepSpinBox,
    AutosaveSpinBox,
}

/// The `QButtonGroup *` (and other native object) arguments below are opaque
/// native identities, spelled the way `DockingDialogNativeBoundary` spells
/// them: a frontend which owns Qt can use its pointer cast to `usize`, a
/// non-Qt frontend a stable application handle, and `0` is the source's null.
pub trait BehaviorNativeBoundary {
    fn setup_ui(&mut self);
    fn create_exit_group(&mut self) -> usize;
    fn exit_group_add_button(&mut self, group: usize, button: ExitButton, id: i32);
    fn hide_allow_ctrl_on_mac(&mut self);
    fn set_allow_ctrl_enabled(&mut self, enabled: bool);
    fn macos(&self) -> bool;
    fn mac_ctrl_environment_set(&self) -> bool;
    fn qt_version_6_or_later(&self) -> bool;
    fn connect_exit_group_button_clicked(&mut self, group: usize);
    fn connect_exit_group_id_clicked(&mut self, group: usize);
    fn font_width(&self, text: &str) -> i32;
    fn set_autosave_spin_maximum_width(&mut self, width: i32);
    fn set_checked(&mut self, control: BehaviorCheckBox, value: bool);
    fn set_spin_box(&mut self, control: BehaviorSpinBox, value: i32);
    fn set_autosave_dir(&mut self, path: String);
    fn to_native_separators(&self, path: &str) -> String;
    fn set_exit_group(&mut self, group: usize, id: i32);
    fn checked(&self, control: BehaviorCheckBox) -> bool;
    fn spin_box_value(&self, control: BehaviorSpinBox) -> i32;
    fn autosave_dir(&self) -> String;
    fn clean_path(&self, path: &str) -> String;
    fn retranslate_ui(&mut self);
}

#[derive(Debug)]
pub struct BehaviorForm {
    pub m_prefs: ImodPrefStruct,
    pub exit_group: usize,
}

impl BehaviorForm {
    /// `BehaviorForm()` source constructor.
    pub fn new(prefs: ImodPrefStruct, native: &mut dyn BehaviorNativeBoundary) -> Self {
        native.setup_ui();
        let mut form = Self {
            m_prefs: prefs,
            exit_group: 0,
        };
        form.init(native);
        form
    }
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self, native: &mut dyn BehaviorNativeBoundary) {
        native.retranslate_ui();
    }
    pub fn init(&mut self, native: &mut dyn BehaviorNativeBoundary) {
        self.exit_group = native.create_exit_group();
        native.exit_group_add_button(self.exit_group, ExitButton::AskExitRadio, 0);
        native.exit_group_add_button(self.exit_group, ExitButton::NeverExitRadio, 1);
        native.exit_group_add_button(self.exit_group, ExitButton::AlwaysExitRadio, 2);
        if !native.macos() {
            native.hide_allow_ctrl_on_mac();
        }
        native.set_allow_ctrl_enabled(!native.mac_ctrl_environment_set());
        if native.qt_version_6_or_later() {
            native.connect_exit_group_id_clicked(self.exit_group);
        } else {
            native.connect_exit_group_button_clicked(self.exit_group);
        }
        self.set_font_dependent_widths(native);
        self.update(native);
    }
    pub fn set_font_dependent_widths(&mut self, native: &mut dyn BehaviorNativeBoundary) {
        native.set_autosave_spin_maximum_width((6 * 2 + 3) * native.font_width("999999") / (6 * 2));
    }
    pub fn exit_type_changed(&mut self, value: i32) {
        self.m_prefs.exit_when_all_closed = value - 1;
    }
    pub fn update(&mut self, n: &mut dyn BehaviorNativeBoundary) {
        let p = &self.m_prefs;
        n.set_checked(BehaviorCheckBox::AllowCtrlOnMacBox, p.allow_ctrl_on_mac);
        n.set_checked(BehaviorCheckBox::SilenceBox, p.silent_beep);
        n.set_checked(BehaviorCheckBox::ClassicBox, p.classic_slicer);
        n.set_checked(BehaviorCheckBox::StartInHqBox, p.start_in_hq);
        n.set_checked(BehaviorCheckBox::ArrowsScrollZapBox, p.arrows_scroll_zap);
        n.set_checked(BehaviorCheckBox::StartAtMidZBox, p.start_at_mid_z);
        n.set_checked(BehaviorCheckBox::SelectOnCheckBox, p.attach_to_on_obj);
        n.set_checked(BehaviorCheckBox::SlicerNewSurfBox, p.slicer_new_surf);
        n.set_spin_box(BehaviorSpinBox::F1f8StepSpinBox, p.bw_step);
        n.set_spin_box(BehaviorSpinBox::PageStepSpinBox, p.page_step);
        n.set_checked(BehaviorCheckBox::AutosaveEnabledBox, p.autosave_on);
        n.set_checked(BehaviorCheckBox::OmitContMeshBox, p.autosave_no_cont_mesh);
        n.set_checked(BehaviorCheckBox::OmitIsosurfMeshBox, p.autosave_no_iso_mesh);
        n.set_spin_box(BehaviorSpinBox::AutosaveSpinBox, p.autosave_interval);
        n.set_autosave_dir(n.to_native_separators(&p.autosave_dir));
        n.set_checked(BehaviorCheckBox::KeyHwStereoBox, p.key_sets_hw_stereo);
        n.set_checked(BehaviorCheckBox::NoVbForContBox, p.no_vert_buf_for_cont);
        n.set_checked(BehaviorCheckBox::NoVbForSphereBox, p.no_vbo_for_sphere);
        n.set_exit_group(self.exit_group, p.exit_when_all_closed + 1);
    }
    pub fn unload(&mut self, n: &mut dyn BehaviorNativeBoundary) {
        let p = &mut self.m_prefs;
        p.allow_ctrl_on_mac = n.checked(BehaviorCheckBox::AllowCtrlOnMacBox);
        p.silent_beep = n.checked(BehaviorCheckBox::SilenceBox);
        p.classic_slicer = n.checked(BehaviorCheckBox::ClassicBox);
        p.start_in_hq = n.checked(BehaviorCheckBox::StartInHqBox);
        p.arrows_scroll_zap = n.checked(BehaviorCheckBox::ArrowsScrollZapBox);
        p.start_at_mid_z = n.checked(BehaviorCheckBox::StartAtMidZBox);
        p.attach_to_on_obj = n.checked(BehaviorCheckBox::SelectOnCheckBox);
        p.slicer_new_surf = n.checked(BehaviorCheckBox::SlicerNewSurfBox);
        p.bw_step = n.spin_box_value(BehaviorSpinBox::F1f8StepSpinBox);
        p.page_step = n.spin_box_value(BehaviorSpinBox::PageStepSpinBox);
        p.autosave_on = n.checked(BehaviorCheckBox::AutosaveEnabledBox);
        p.autosave_no_cont_mesh = n.checked(BehaviorCheckBox::OmitContMeshBox);
        p.autosave_no_iso_mesh = n.checked(BehaviorCheckBox::OmitIsosurfMeshBox);
        p.autosave_interval = n.spin_box_value(BehaviorSpinBox::AutosaveSpinBox);
        p.key_sets_hw_stereo = n.checked(BehaviorCheckBox::KeyHwStereoBox);
        p.no_vert_buf_for_cont = n.checked(BehaviorCheckBox::NoVbForContBox);
        p.no_vbo_for_sphere = n.checked(BehaviorCheckBox::NoVbForSphereBox);
        p.autosave_dir = n.clean_path(&n.autosave_dir());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native {
        checks: [bool; 14],
        spins: [i32; 3],
        path: String,
        qt6: bool,
        old_connect: usize,
        new_connect: usize,
        hidden: bool,
        enabled: bool,
    }
    impl BehaviorNativeBoundary for Native {
        fn setup_ui(&mut self) {}
        fn create_exit_group(&mut self) -> usize {
            1
        }
        fn exit_group_add_button(&mut self, _: usize, _: ExitButton, _: i32) {}
        fn hide_allow_ctrl_on_mac(&mut self) {
            self.hidden = true;
        }
        fn set_allow_ctrl_enabled(&mut self, x: bool) {
            self.enabled = x;
        }
        fn macos(&self) -> bool {
            false
        }
        fn mac_ctrl_environment_set(&self) -> bool {
            false
        }
        fn qt_version_6_or_later(&self) -> bool {
            self.qt6
        }
        fn connect_exit_group_button_clicked(&mut self, _: usize) {
            self.old_connect += 1;
        }
        fn connect_exit_group_id_clicked(&mut self, _: usize) {
            self.new_connect += 1;
        }
        fn font_width(&self, _: &str) -> i32 {
            12
        }
        fn set_autosave_spin_maximum_width(&mut self, _: i32) {}
        fn set_checked(&mut self, c: BehaviorCheckBox, v: bool) {
            self.checks[c as usize] = v;
        }
        fn set_spin_box(&mut self, c: BehaviorSpinBox, v: i32) {
            self.spins[c as usize] = v;
        }
        fn set_autosave_dir(&mut self, x: String) {
            self.path = x;
        }
        fn to_native_separators(&self, x: &str) -> String {
            x.replace('/', "\\")
        }
        fn set_exit_group(&mut self, _: usize, _: i32) {}
        fn checked(&self, c: BehaviorCheckBox) -> bool {
            self.checks[c as usize]
        }
        fn spin_box_value(&self, c: BehaviorSpinBox) -> i32 {
            self.spins[c as usize]
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
    fn init_selects_the_source_qt_signal_branch_and_platform_controls() {
        let mut n = Native {
            qt6: true,
            ..Default::default()
        };
        let _f = BehaviorForm::new(ImodPrefStruct::default(), &mut n);
        assert_eq!((n.old_connect, n.new_connect), (0, 1));
        assert!(n.hidden);
        assert!(n.enabled);
    }
    #[test]
    fn exit_mapping_and_native_separator_update_match_source() {
        let mut n = Native::default();
        let mut f = BehaviorForm::new(
            ImodPrefStruct {
                autosave_dir: "a/b".into(),
                ..Default::default()
            },
            &mut n,
        );
        assert_eq!(n.path, "a\\b");
        f.exit_type_changed(0);
        assert_eq!(f.m_prefs.exit_when_all_closed, -1);
    }
    #[test]
    fn unload_reads_all_reached_autosave_fields_and_cleans_path() {
        let mut n = Native::default();
        let mut f = BehaviorForm::new(ImodPrefStruct::default(), &mut n);
        n.checks[BehaviorCheckBox::AutosaveEnabledBox as usize] = true;
        n.spins[BehaviorSpinBox::AutosaveSpinBox as usize] = 15;
        n.path = "a//b".into();
        f.unload(&mut n);
        assert!(f.m_prefs.autosave_on);
        assert_eq!(f.m_prefs.autosave_interval, 15);
        assert_eq!(f.m_prefs.autosave_dir, "a/b");
    }
}
