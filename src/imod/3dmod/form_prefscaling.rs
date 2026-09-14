//! Translation of `IMOD/3dmod/form_prefscaling.cpp` and `form_prefscaling.h`.
#![allow(dead_code)]

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImodPrefStruct {
    pub load_ushorts: bool,
    pub load_int_if_mean_sd: bool,
    pub load_int_if_estimate: bool,
    pub eer_super_res: i32,
    pub eer_zbinning: i32,
    pub prefer_mean_sd: bool,
    pub change_mrcstats: bool,
    pub num_sds_for_scaling: i32,
    pub scale_scan_type: i32,
    pub use_ali_piece_coords: i32,
    pub auto_target_mean: i32,
    pub auto_target_sd: i32,
    pub auto_con_at_start: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CurrentImageInfo {
    pub fake_image: bool,
    pub rgb_store: bool,
    pub white: i32,
    pub black: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefScalingButton {
    FullScanButton,
    SubsetMinMaxButton,
    SubsetMeanSdButton,
    UnaliCoordButton,
    AlignedCoordButton,
    AlignedVsCoordButton,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefScalingCheckBox {
    LoadUshortBox,
    LoadIntIfMeanSdBox,
    LoadIntIfEstimateBox,
    PreferMeanSdBox,
    ChangeMrcstatsBox,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefScalingSpinBox {
    SuperResSpinBox,
    EerZbinSpinBox,
    TruncateAtSpinBox,
    AutoMeanSpinBox,
    AutoSdSpinBox,
}

/// The `QButtonGroup *` (and other native object) arguments below are opaque
/// native identities, spelled the way `DockingDialogNativeBoundary` spells
/// them: a frontend which owns Qt can use its pointer cast to `usize`, a
/// non-Qt frontend a stable application handle, and `0` is the source's null.
pub trait PrefScalingNativeBoundary {
    fn setup_ui(&mut self);
    fn create_button_group(&mut self) -> usize;
    fn group_add_button(&mut self, group: usize, button: PrefScalingButton, id: i32);
    fn connect_scan_group_clicked(&mut self, group: usize);
    fn connect_piece_group_clicked(&mut self, group: usize);
    fn connect_auto_mean_value_changed(&mut self);
    fn connect_auto_sd_value_changed(&mut self);
    fn connect_set_target_clicked(&mut self);
    fn set_target_enabled(&mut self, enabled: bool);
    fn set_auto_mean_enabled(&mut self, enabled: bool);
    fn set_auto_sd_enabled(&mut self, enabled: bool);
    fn set_checked(&mut self, control: PrefScalingCheckBox, value: bool);
    fn set_spin_box(&mut self, control: PrefScalingSpinBox, value: i32);
    fn set_group(&mut self, group: usize, value: i32);
    fn set_auto_con_at_start(&mut self, value: i32);
    fn checked(&self, control: PrefScalingCheckBox) -> bool;
    fn spin_box_value(&self, control: PrefScalingSpinBox) -> i32;
    fn auto_con_at_start(&self) -> i32;
    fn info_auto_contrast(&mut self, mean: i32, sd: i32);
    fn info_current_mean_sd(&mut self) -> Option<(f32, f32, f32, f32)>;
    fn retranslate_ui(&mut self);
}

#[derive(Debug)]
pub struct PrefScalingForm {
    pub m_prefs: ImodPrefStruct,
    pub scan_group: usize,
    pub piece_group: usize,
}

impl PrefScalingForm {
    pub fn new(
        prefs: ImodPrefStruct,
        cvi: CurrentImageInfo,
        native: &mut dyn PrefScalingNativeBoundary,
    ) -> Self {
        native.setup_ui();
        let mut form = Self {
            m_prefs: prefs,
            scan_group: 0,
            piece_group: 0,
        };
        form.init(cvi, native);
        form
    }

    pub fn destroy(&mut self) {}

    pub fn language_change(&mut self, native: &mut dyn PrefScalingNativeBoundary) {
        native.retranslate_ui();
    }

    pub fn init(&mut self, cvi: CurrentImageInfo, native: &mut dyn PrefScalingNativeBoundary) {
        self.scan_group = native.create_button_group();
        native.group_add_button(self.scan_group, PrefScalingButton::FullScanButton, 0);
        native.group_add_button(self.scan_group, PrefScalingButton::SubsetMinMaxButton, 1);
        native.group_add_button(self.scan_group, PrefScalingButton::SubsetMeanSdButton, 2);
        self.piece_group = native.create_button_group();
        native.group_add_button(self.piece_group, PrefScalingButton::UnaliCoordButton, 0);
        native.group_add_button(self.piece_group, PrefScalingButton::AlignedCoordButton, 1);
        native.group_add_button(self.piece_group, PrefScalingButton::AlignedVsCoordButton, 2);
        native.connect_scan_group_clicked(self.scan_group);
        native.connect_piece_group_clicked(self.piece_group);
        native.connect_auto_mean_value_changed();
        native.connect_auto_sd_value_changed();
        native.connect_set_target_clicked();
        self.set_font_dependent_widths();
        if cvi.fake_image || cvi.rgb_store {
            native.set_target_enabled(false);
            native.set_auto_mean_enabled(false);
            native.set_auto_sd_enabled(false);
        }
        self.update(native);
    }

    pub fn set_font_dependent_widths(&mut self) {}

    pub fn scan_type_changed(&mut self, value: i32) {
        self.m_prefs.scale_scan_type = value;
    }

    pub fn piece_type_changed(&mut self, value: i32) {
        self.m_prefs.use_ali_piece_coords = value;
    }

    pub fn auto_mean_changed(&mut self, value: i32, native: &mut dyn PrefScalingNativeBoundary) {
        self.m_prefs.auto_target_mean = value;
        native.info_auto_contrast(self.m_prefs.auto_target_mean, self.m_prefs.auto_target_sd);
    }

    pub fn auto_sd_changed(&mut self, value: i32, native: &mut dyn PrefScalingNativeBoundary) {
        self.m_prefs.auto_target_sd = value;
        native.info_auto_contrast(self.m_prefs.auto_target_mean, self.m_prefs.auto_target_sd);
    }

    pub fn set_target_clicked(
        &mut self,
        cvi: CurrentImageInfo,
        native: &mut dyn PrefScalingNativeBoundary,
    ) {
        let Some((image_mean, image_sd, _, _)) = native.info_current_mean_sd() else {
            return;
        };
        let mut range = cvi.white - cvi.black;
        if range <= 0 {
            range = 1;
        }
        // `255.` and `0.5` are double literals in the source, so `imageMean -
        // App->cvi->black` is the only single-precision step: the int converts to
        // float for the subtraction, and that float then widens for the multiply,
        // divide and add, which all happen in double.
        let target_mean =
            (255. * (image_mean - cvi.black as f32) as f64 / range as f64 + 0.5) as i32;
        let target_sd = (255. * image_sd as f64 / range as f64 - 0.5) as i32;
        native.set_spin_box(PrefScalingSpinBox::AutoMeanSpinBox, target_mean);
        native.set_spin_box(PrefScalingSpinBox::AutoSdSpinBox, target_sd);
        self.m_prefs.auto_target_mean = native.spin_box_value(PrefScalingSpinBox::AutoMeanSpinBox);
        self.m_prefs.auto_target_sd = native.spin_box_value(PrefScalingSpinBox::AutoSdSpinBox);
    }

    pub fn update(&mut self, native: &mut dyn PrefScalingNativeBoundary) {
        let p = &self.m_prefs;
        native.set_checked(PrefScalingCheckBox::LoadUshortBox, p.load_ushorts);
        native.set_checked(
            PrefScalingCheckBox::LoadIntIfMeanSdBox,
            p.load_int_if_mean_sd,
        );
        native.set_checked(
            PrefScalingCheckBox::LoadIntIfEstimateBox,
            p.load_int_if_estimate,
        );
        native.set_spin_box(PrefScalingSpinBox::SuperResSpinBox, p.eer_super_res);
        native.set_spin_box(PrefScalingSpinBox::EerZbinSpinBox, p.eer_zbinning);
        native.set_checked(PrefScalingCheckBox::PreferMeanSdBox, p.prefer_mean_sd);
        native.set_checked(PrefScalingCheckBox::ChangeMrcstatsBox, p.change_mrcstats);
        native.set_spin_box(PrefScalingSpinBox::TruncateAtSpinBox, p.num_sds_for_scaling);
        native.set_group(self.scan_group, p.scale_scan_type);
        native.set_group(self.piece_group, p.use_ali_piece_coords);
        native.set_spin_box(PrefScalingSpinBox::AutoMeanSpinBox, p.auto_target_mean);
        native.set_spin_box(PrefScalingSpinBox::AutoSdSpinBox, p.auto_target_sd);
        native.set_auto_con_at_start(p.auto_con_at_start);
    }

    pub fn unload(&mut self, native: &mut dyn PrefScalingNativeBoundary) {
        let p = &mut self.m_prefs;
        p.load_ushorts = native.checked(PrefScalingCheckBox::LoadUshortBox);
        p.load_int_if_mean_sd = native.checked(PrefScalingCheckBox::LoadIntIfMeanSdBox);
        p.load_int_if_estimate = native.checked(PrefScalingCheckBox::LoadIntIfEstimateBox);
        p.eer_super_res = native.spin_box_value(PrefScalingSpinBox::SuperResSpinBox);
        p.eer_zbinning = native.spin_box_value(PrefScalingSpinBox::EerZbinSpinBox);
        p.prefer_mean_sd = native.checked(PrefScalingCheckBox::PreferMeanSdBox);
        p.change_mrcstats = native.checked(PrefScalingCheckBox::ChangeMrcstatsBox);
        p.num_sds_for_scaling = native.spin_box_value(PrefScalingSpinBox::TruncateAtSpinBox);
        p.auto_con_at_start = native.auto_con_at_start();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Native {
        checks: [bool; 5],
        spins: [i32; 5],
        auto_start: i32,
        contrast: Option<(i32, i32)>,
        mean_sd: Option<(f32, f32, f32, f32)>,
        buttons: Vec<(PrefScalingButton, i32)>,
        connections: usize,
        disabled: usize,
    }

    impl PrefScalingNativeBoundary for Native {
        fn setup_ui(&mut self) {}
        fn create_button_group(&mut self) -> usize {
            1
        }
        fn group_add_button(&mut self, _: usize, b: PrefScalingButton, id: i32) {
            self.buttons.push((b, id));
        }
        fn connect_scan_group_clicked(&mut self, _: usize) {
            self.connections += 1;
        }
        fn connect_piece_group_clicked(&mut self, _: usize) {
            self.connections += 1;
        }
        fn connect_auto_mean_value_changed(&mut self) {
            self.connections += 1;
        }
        fn connect_auto_sd_value_changed(&mut self) {
            self.connections += 1;
        }
        fn connect_set_target_clicked(&mut self) {
            self.connections += 1;
        }
        fn set_target_enabled(&mut self, enabled: bool) {
            if !enabled {
                self.disabled += 1;
            }
        }
        fn set_auto_mean_enabled(&mut self, enabled: bool) {
            if !enabled {
                self.disabled += 1;
            }
        }
        fn set_auto_sd_enabled(&mut self, enabled: bool) {
            if !enabled {
                self.disabled += 1;
            }
        }
        fn set_checked(&mut self, c: PrefScalingCheckBox, v: bool) {
            self.checks[c as usize] = v;
        }
        fn set_spin_box(&mut self, c: PrefScalingSpinBox, v: i32) {
            self.spins[c as usize] = v;
        }
        fn set_group(&mut self, _: usize, _: i32) {}
        fn set_auto_con_at_start(&mut self, v: i32) {
            self.auto_start = v;
        }
        fn checked(&self, c: PrefScalingCheckBox) -> bool {
            self.checks[c as usize]
        }
        fn spin_box_value(&self, c: PrefScalingSpinBox) -> i32 {
            self.spins[c as usize]
        }
        fn auto_con_at_start(&self) -> i32 {
            self.auto_start
        }
        fn info_auto_contrast(&mut self, mean: i32, sd: i32) {
            self.contrast = Some((mean, sd));
        }
        fn info_current_mean_sd(&mut self) -> Option<(f32, f32, f32, f32)> {
            self.mean_sd
        }
        fn retranslate_ui(&mut self) {}
    }

    #[test]
    fn init_installs_source_buttons_connections_and_fake_image_disablement() {
        let mut native = Native::default();
        let _form = PrefScalingForm::new(
            ImodPrefStruct::default(),
            CurrentImageInfo {
                fake_image: true,
                ..Default::default()
            },
            &mut native,
        );
        assert_eq!(native.buttons.len(), 6);
        assert_eq!(native.connections, 5);
        assert_eq!(native.disabled, 3);
    }

    #[test]
    fn target_uses_source_rounding_and_current_black_white() {
        let mut native = Native {
            mean_sd: Some((60., 10., 0., 0.)),
            ..Default::default()
        };
        let cvi = CurrentImageInfo {
            black: 10,
            white: 110,
            ..Default::default()
        };
        let mut form = PrefScalingForm::new(ImodPrefStruct::default(), cvi, &mut native);
        form.set_target_clicked(cvi, &mut native);
        assert_eq!(
            (form.m_prefs.auto_target_mean, form.m_prefs.auto_target_sd),
            (128, 25)
        );
    }

    /// `form_prefscaling.cpp:119-120` writes `255.` and `0.5` as double
    /// literals, so `imageMean - App->cvi->black` is the only single-precision
    /// step and the multiply, divide and add that follow are all double. Both
    /// inputs here sit just under a truncation boundary in double and just over
    /// it in an all-`f32` evaluation, which is what the translation used to do.
    #[test]
    fn target_widens_to_double_after_the_single_precision_subtraction() {
        let mut native = Native {
            mean_sd: Some((25558.1, 0., 0., 0.)),
            ..Default::default()
        };
        let cvi = CurrentImageInfo {
            black: 7271,
            white: 7280,
            ..Default::default()
        };
        let mut form = PrefScalingForm::new(ImodPrefStruct::default(), cvi, &mut native);
        form.set_target_clicked(cvi, &mut native);
        assert_eq!(form.m_prefs.auto_target_mean, 518134);
        native.mean_sd = Some((0., 13311.8, 0., 0.));
        let cvi = CurrentImageInfo {
            black: 0,
            white: 3434,
            ..Default::default()
        };
        form.set_target_clicked(cvi, &mut native);
        assert_eq!(form.m_prefs.auto_target_sd, 987);
    }

    #[test]
    fn callbacks_and_unload_follow_source_preference_selection() {
        let mut native = Native::default();
        let mut form = PrefScalingForm::new(
            ImodPrefStruct::default(),
            CurrentImageInfo::default(),
            &mut native,
        );
        form.auto_mean_changed(30, &mut native);
        form.auto_sd_changed(12, &mut native);
        native.checks[PrefScalingCheckBox::LoadUshortBox as usize] = true;
        native.spins[PrefScalingSpinBox::EerZbinSpinBox as usize] = 4;
        native.auto_start = 2;
        form.unload(&mut native);
        assert_eq!(native.contrast, Some((30, 12)));
        assert!(form.m_prefs.load_ushorts);
        assert_eq!(
            (form.m_prefs.eer_zbinning, form.m_prefs.auto_con_at_start),
            (4, 2)
        );
    }
}
