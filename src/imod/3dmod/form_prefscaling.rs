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
/// Native Qt widgets plus `imodInfo*` callbacks, the direct source boundary.
pub trait PrefScalingNativeBoundary {
    fn add_scan_group_button(&mut self, id: i32);
    fn add_piece_group_button(&mut self, id: i32);
    fn connect_signals(&mut self);
    fn set_target_enabled(&mut self, enabled: bool);
    fn set_auto_mean_enabled(&mut self, enabled: bool);
    fn set_auto_sd_enabled(&mut self, enabled: bool);
    fn set_checked(&mut self, which: i32, value: bool);
    fn set_spin_box(&mut self, which: i32, value: i32);
    fn set_group(&mut self, which: i32, value: i32);
    fn set_auto_con_at_start(&mut self, value: i32);
    fn checked(&self, which: i32) -> bool;
    fn spin_box_value(&self, which: i32) -> i32;
    fn auto_con_at_start(&self) -> i32;
    fn info_auto_contrast(&mut self, mean: i32, sd: i32);
    fn info_current_mean_sd(&mut self) -> Option<(f32, f32, f32, f32)>;
    fn retranslate_ui(&mut self);
}
pub const LOAD_USHORTS: i32 = 0;
pub const LOAD_INT_MEAN_SD: i32 = 1;
pub const LOAD_INT_ESTIMATE: i32 = 2;
pub const PREFER_MEAN_SD: i32 = 3;
pub const CHANGE_MRCSTATS: i32 = 4;
pub const SUPER_RES: i32 = 0;
pub const EER_ZBIN: i32 = 1;
pub const TRUNCATE: i32 = 2;
pub const AUTO_MEAN: i32 = 3;
pub const AUTO_SD: i32 = 4;
pub const SCAN_GROUP: i32 = 0;
pub const PIECE_GROUP: i32 = 1;

/// `PrefScalingForm`.
#[derive(Clone, Debug, Default)]
pub struct PrefScalingForm {
    pub m_prefs: ImodPrefStruct,
    pub scan_group: bool,
    pub piece_group: bool,
}
impl PrefScalingForm {
    /// `PrefScalingForm::PrefScalingForm`.
    pub fn new(
        prefs: ImodPrefStruct,
        cvi: CurrentImageInfo,
        native: &mut dyn PrefScalingNativeBoundary,
    ) -> Self {
        let mut form = Self {
            m_prefs: prefs,
            ..Self::default()
        };
        form.init(cvi, native);
        form
    }
    /// `PrefScalingForm::~PrefScalingForm`.
    pub fn destroy(&mut self) {}
    /// `PrefScalingForm::languageChange`.
    pub fn language_change(&mut self, native: &mut dyn PrefScalingNativeBoundary) {
        native.retranslate_ui()
    }
    /// `PrefScalingForm::init`.
    pub fn init(&mut self, cvi: CurrentImageInfo, native: &mut dyn PrefScalingNativeBoundary) {
        native.add_scan_group_button(0);
        native.add_scan_group_button(1);
        native.add_scan_group_button(2);
        native.add_piece_group_button(0);
        native.add_piece_group_button(1);
        native.add_piece_group_button(2);
        self.scan_group = true;
        self.piece_group = true;
        native.connect_signals();
        self.set_font_dependent_widths();
        if cvi.fake_image || cvi.rgb_store {
            native.set_target_enabled(false);
            native.set_auto_mean_enabled(false);
            native.set_auto_sd_enabled(false);
        }
        self.update(native)
    }
    /// `PrefScalingForm::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self) {}
    /// `PrefScalingForm::scanTypeChanged`.
    pub fn scan_type_changed(&mut self, value: i32) {
        self.m_prefs.scale_scan_type = value
    }
    /// `PrefScalingForm::pieceTypeChanged`.
    pub fn piece_type_changed(&mut self, value: i32) {
        self.m_prefs.use_ali_piece_coords = value
    }
    /// `PrefScalingForm::autoMeanChanged`.
    pub fn auto_mean_changed(&mut self, value: i32, native: &mut dyn PrefScalingNativeBoundary) {
        self.m_prefs.auto_target_mean = value;
        native.info_auto_contrast(self.m_prefs.auto_target_mean, self.m_prefs.auto_target_sd)
    }
    /// `PrefScalingForm::autoSDChanged`.
    pub fn auto_sd_changed(&mut self, value: i32, native: &mut dyn PrefScalingNativeBoundary) {
        self.m_prefs.auto_target_sd = value;
        native.info_auto_contrast(self.m_prefs.auto_target_mean, self.m_prefs.auto_target_sd)
    }
    /// `PrefScalingForm::setTargetClicked`.
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
        let target_mean = (255. * (image_mean - cvi.black as f32) / range as f32 + 0.5) as i32;
        let target_sd = (255. * image_sd / range as f32 - 0.5) as i32;
        native.set_spin_box(AUTO_MEAN, target_mean);
        native.set_spin_box(AUTO_SD, target_sd);
        self.m_prefs.auto_target_mean = native.spin_box_value(AUTO_MEAN);
        self.m_prefs.auto_target_sd = native.spin_box_value(AUTO_SD);
    }
    /// `PrefScalingForm::update`.
    pub fn update(&mut self, native: &mut dyn PrefScalingNativeBoundary) {
        let p = &self.m_prefs;
        native.set_checked(LOAD_USHORTS, p.load_ushorts);
        native.set_checked(LOAD_INT_MEAN_SD, p.load_int_if_mean_sd);
        native.set_checked(LOAD_INT_ESTIMATE, p.load_int_if_estimate);
        native.set_spin_box(SUPER_RES, p.eer_super_res);
        native.set_spin_box(EER_ZBIN, p.eer_zbinning);
        native.set_checked(PREFER_MEAN_SD, p.prefer_mean_sd);
        native.set_checked(CHANGE_MRCSTATS, p.change_mrcstats);
        native.set_spin_box(TRUNCATE, p.num_sds_for_scaling);
        native.set_group(SCAN_GROUP, p.scale_scan_type);
        native.set_group(PIECE_GROUP, p.use_ali_piece_coords);
        native.set_spin_box(AUTO_MEAN, p.auto_target_mean);
        native.set_spin_box(AUTO_SD, p.auto_target_sd);
        native.set_auto_con_at_start(p.auto_con_at_start);
    }
    /// `PrefScalingForm::unload`.
    pub fn unload(&mut self, native: &mut dyn PrefScalingNativeBoundary) {
        let p = &mut self.m_prefs;
        p.load_ushorts = native.checked(LOAD_USHORTS);
        p.load_int_if_mean_sd = native.checked(LOAD_INT_MEAN_SD);
        p.load_int_if_estimate = native.checked(LOAD_INT_ESTIMATE);
        p.eer_super_res = native.spin_box_value(SUPER_RES);
        p.eer_zbinning = native.spin_box_value(EER_ZBIN);
        p.prefer_mean_sd = native.checked(PREFER_MEAN_SD);
        p.change_mrcstats = native.checked(CHANGE_MRCSTATS);
        p.num_sds_for_scaling = native.spin_box_value(TRUNCATE);
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
    }
    impl PrefScalingNativeBoundary for Native {
        fn add_scan_group_button(&mut self, _: i32) {}
        fn add_piece_group_button(&mut self, _: i32) {}
        fn connect_signals(&mut self) {}
        fn set_target_enabled(&mut self, _: bool) {}
        fn set_auto_mean_enabled(&mut self, _: bool) {}
        fn set_auto_sd_enabled(&mut self, _: bool) {}
        fn set_checked(&mut self, which: i32, value: bool) {
            self.checks[which as usize] = value
        }
        fn set_spin_box(&mut self, which: i32, value: i32) {
            self.spins[which as usize] = value
        }
        fn set_group(&mut self, _: i32, _: i32) {}
        fn set_auto_con_at_start(&mut self, value: i32) {
            self.auto_start = value
        }
        fn checked(&self, which: i32) -> bool {
            self.checks[which as usize]
        }
        fn spin_box_value(&self, which: i32) -> i32 {
            self.spins[which as usize]
        }
        fn auto_con_at_start(&self) -> i32 {
            self.auto_start
        }
        fn info_auto_contrast(&mut self, mean: i32, sd: i32) {
            self.contrast = Some((mean, sd))
        }
        fn info_current_mean_sd(&mut self) -> Option<(f32, f32, f32, f32)> {
            self.mean_sd
        }
        fn retranslate_ui(&mut self) {}
    }
    #[test]
    fn target_uses_source_rounding_and_current_black_white() {
        let mut n = Native {
            mean_sd: Some((60., 10., 0., 0.)),
            ..Default::default()
        };
        let c = CurrentImageInfo {
            black: 10,
            white: 110,
            ..Default::default()
        };
        let mut f = PrefScalingForm::new(ImodPrefStruct::default(), c, &mut n);
        f.set_target_clicked(c, &mut n);
        assert_eq!(
            (f.m_prefs.auto_target_mean, f.m_prefs.auto_target_sd),
            (128, 25)
        );
    }
    #[test]
    fn auto_callbacks_keep_prefs_and_info_callback_together() {
        let mut n = Native::default();
        let mut f = PrefScalingForm::new(
            ImodPrefStruct::default(),
            CurrentImageInfo::default(),
            &mut n,
        );
        f.auto_mean_changed(30, &mut n);
        f.auto_sd_changed(12, &mut n);
        assert_eq!(n.contrast, Some((30, 12)));
    }
}
