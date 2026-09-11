//! Translation of `IMOD/3dmod/form_appearance.cpp` together with its header.
//!
//! Qt controls/dialogs are explicit native boundary calls. Preference mutation
//! and the zoom/color algorithms remain in this source-organized unit.
#![allow(dead_code)]

pub const MAX_ZOOMS: usize = 22;
pub const MAX_NAMED_COLORS: usize = 12;
pub const IMOD_CURPOINT: i32 = 233;
pub const IMOD_BGNPOINT: i32 = 232;
pub const IMOD_ENDPOINT: i32 = 231;
pub const IMOD_FOREGROUND: i32 = 228;
pub const IMOD_BACKGROUND: i32 = 227;
pub const IMOD_GHOST: i32 = 234;
pub const IMOD_SHADOW: i32 = 230;
pub const IMOD_ARROW: i32 = 226;

/// Form-used subset of upstream `imod_pref_struct`.
#[derive(Clone, Debug)]
pub struct ImodPrefStruct {
    pub min_im_pt_size: i32,
    pub min_im_pt_size_chgd: bool,
    pub min_mod_pt_size: i32,
    pub min_mod_pt_size_chgd: bool,
    pub boost_model_cursor: i32,
    pub slicer_pan_kb: i32,
    pub max_linked_slicers: i32,
    pub speedup_slider: bool,
    pub iso_high_thresh: bool,
    pub iso_box_limit: i32,
    pub iso_box_initial: i32,
    pub font_chgd: bool,
    pub style_key: String,
    pub style_chgd: bool,
    pub named_index: [i32; MAX_NAMED_COLORS],
    pub named_color: [u32; MAX_NAMED_COLORS],
    pub zooms: [f64; MAX_ZOOMS],
    pub zooms_dflt: [f64; MAX_ZOOMS],
    pub zooms_chgd: bool,
}
impl Default for ImodPrefStruct {
    fn default() -> Self {
        Self {
            min_im_pt_size: 1,
            min_im_pt_size_chgd: false,
            min_mod_pt_size: 1,
            min_mod_pt_size_chgd: false,
            boost_model_cursor: 0,
            slicer_pan_kb: 0,
            max_linked_slicers: 0,
            speedup_slider: false,
            iso_high_thresh: false,
            iso_box_limit: 0,
            iso_box_initial: 0,
            font_chgd: false,
            style_key: String::new(),
            style_chgd: false,
            named_index: [0; MAX_NAMED_COLORS],
            named_color: [0; MAX_NAMED_COLORS],
            zooms: [1.; MAX_ZOOMS],
            zooms_dflt: [1.; MAX_ZOOMS],
            zooms_chgd: false,
        }
    }
}

/// Native Qt operations and direct `ImodPreferences` service calls.
pub trait AppearanceNativeBoundary {
    fn connect_appearance_signals(&mut self);
    fn max_cursor_steps(&self) -> i32;
    fn set_zoom_index_maximum(&mut self, maximum: i32);
    fn set_spin_box(&mut self, which: i32, value: i32);
    fn set_checked(&mut self, which: i32, value: bool);
    fn set_enabled(&mut self, which: i32, value: bool);
    fn add_style(&mut self, style: &str);
    fn style_available(&mut self, style: &str) -> bool;
    fn style_list(&self) -> Vec<String>;
    fn set_style_index(&mut self, index: i32);
    fn font_width(&self, text: &str) -> i32;
    fn set_zoom_edit_maximum_width(&mut self, width: i32);
    fn choose_font(&mut self) -> bool;
    fn change_font(&mut self);
    fn point_size_changed(&mut self);
    fn current_color_index(&self) -> i32;
    fn choose_color(&mut self, color: u32) -> Option<u32>;
    fn change_style(&mut self, key: &str);
    fn spin_box_value(&self, which: i32) -> i32;
    fn checked(&self, which: i32) -> bool;
    fn zoom_edit_text(&self) -> String;
    fn set_zoom_edit_text(&mut self, text: &str);
    fn set_default_zoom_label(&mut self, text: &str);
    fn user_canceled(&mut self);
    fn retranslate_ui(&mut self);
}
pub const IMAGE_PT: i32 = 0;
pub const BOOST_CURSOR: i32 = 1;
pub const MODEL_PT: i32 = 2;
pub const VOX_LIMIT: i32 = 3;
pub const MAX_LINKED: i32 = 4;
pub const ISO_LIMIT: i32 = 5;
pub const ISO_INITIAL: i32 = 6;
pub const ZOOM_INDEX: i32 = 7;
pub const LIMIT_SLIDER: i32 = 0;
pub const ISO_HIGH: i32 = 1;

/// `AppearanceForm` (`form_appearance.h`).
#[derive(Debug)]
pub struct AppearanceForm {
    pub m_zoom_val_changed: bool,
    pub m_prefs: ImodPrefStruct,
    pub m_zoom_index: usize,
}
impl AppearanceForm {
    /// `AppearanceForm::AppearanceForm`.
    pub fn new(prefs: ImodPrefStruct, native: &mut dyn AppearanceNativeBoundary) -> Self {
        let mut form = Self {
            m_zoom_val_changed: false,
            m_prefs: prefs,
            m_zoom_index: MAX_ZOOMS / 3,
        };
        form.init(native);
        form
    }
    /// `AppearanceForm::~AppearanceForm`.
    pub fn destroy(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        native.user_canceled();
    }
    /// `AppearanceForm::languageChange`.
    pub fn language_change(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        native.retranslate_ui();
    }
    /// `AppearanceForm::init`.
    pub fn init(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        let max_steps = native.max_cursor_steps();
        native.connect_appearance_signals();
        native.set_zoom_index_maximum(MAX_ZOOMS as i32);
        native.set_spin_box(ZOOM_INDEX, self.m_zoom_index as i32 + 1);
        self.update(native);
        let mut ind = 0;
        for style in native.style_list() {
            if !native.style_available(&style) {
                continue;
            };
            native.add_style(&style);
            if style.eq_ignore_ascii_case(&self.m_prefs.style_key) {
                native.set_style_index(ind)
            };
            ind += 1;
        }
        if max_steps > 0 {
            native.set_spin_box(BOOST_CURSOR, max_steps)
        } else {
            native.set_enabled(BOOST_CURSOR, false);
            native.set_enabled(8, false);
            native.set_enabled(9, false)
        };
        self.set_font_dependent_widths(native);
    }
    /// `AppearanceForm::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        native.set_zoom_edit_maximum_width((6 * 2 + 3) * native.font_width("999999") / (6 * 2));
    }
    /// `AppearanceForm::update`.
    pub fn update(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        native.set_spin_box(IMAGE_PT, self.m_prefs.min_im_pt_size);
        native.set_spin_box(BOOST_CURSOR, self.m_prefs.boost_model_cursor);
        native.set_spin_box(MODEL_PT, self.m_prefs.min_mod_pt_size);
        native.set_spin_box(VOX_LIMIT, self.m_prefs.slicer_pan_kb);
        native.set_spin_box(MAX_LINKED, self.m_prefs.max_linked_slicers);
        native.set_checked(LIMIT_SLIDER, self.m_prefs.speedup_slider);
        native.set_checked(ISO_HIGH, self.m_prefs.iso_high_thresh);
        native.set_spin_box(ISO_LIMIT, self.m_prefs.iso_box_limit);
        native.set_spin_box(ISO_INITIAL, self.m_prefs.iso_box_initial);
        self.display_current_zoom(native);
    }
    /// `AppearanceForm::fontPressed`.
    pub fn font_pressed(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        if !native.choose_font() {
            return;
        };
        self.m_prefs.font_chgd = true;
        native.change_font();
    }
    /// `AppearanceForm::imagePtChanged`.
    pub fn image_pt_changed(&mut self, value: i32, native: &mut dyn AppearanceNativeBoundary) {
        self.m_prefs.min_im_pt_size = value;
        self.m_prefs.min_im_pt_size_chgd = true;
        native.point_size_changed();
    }
    /// `AppearanceForm::modelPtChanged`.
    pub fn model_pt_changed(&mut self, value: i32, native: &mut dyn AppearanceNativeBoundary) {
        self.m_prefs.min_mod_pt_size = value;
        self.m_prefs.min_mod_pt_size_chgd = true;
        native.point_size_changed();
    }
    /// `AppearanceForm::markerColorClicked`.
    pub fn marker_color_clicked(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        let indexes = [
            IMOD_CURPOINT,
            IMOD_BGNPOINT,
            IMOD_ENDPOINT,
            IMOD_FOREGROUND,
            IMOD_BACKGROUND,
            IMOD_GHOST,
            IMOD_SHADOW,
            IMOD_ARROW,
        ];
        let item = indexes[native.current_color_index().clamp(0, 7) as usize];
        let mut which = 0;
        for i in 0..MAX_NAMED_COLORS {
            if self.m_prefs.named_index[i] == item {
                which = i
            }
        }
        let Some(color) = native.choose_color(self.m_prefs.named_color[which]) else {
            return;
        };
        self.m_prefs.named_color[which] = color;
        native.point_size_changed();
    }
    /// `AppearanceForm::styleSelected`.
    pub fn style_selected(&mut self, key: &str, native: &mut dyn AppearanceNativeBoundary) {
        if key.eq_ignore_ascii_case(&self.m_prefs.style_key) {
            return;
        };
        self.m_prefs.style_chgd = true;
        self.m_prefs.style_key = key.into();
        native.change_style(key);
    }
    /// `AppearanceForm::unload`.
    pub fn unload(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        self.m_prefs.slicer_pan_kb = native.spin_box_value(VOX_LIMIT);
        self.m_prefs.max_linked_slicers = native.spin_box_value(MAX_LINKED);
        self.m_prefs.speedup_slider = native.checked(LIMIT_SLIDER);
        self.m_prefs.iso_high_thresh = native.checked(ISO_HIGH);
        self.m_prefs.iso_box_limit = native.spin_box_value(ISO_LIMIT);
        self.m_prefs.iso_box_initial = native.spin_box_value(ISO_INITIAL);
        self.m_prefs.boost_model_cursor = native.spin_box_value(BOOST_CURSOR);
        self.unload_zoom_value(native);
    }
    /// `AppearanceForm::displayCurrentZoom`.
    pub fn display_current_zoom(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        let zoom = self.m_prefs.zooms[self.m_zoom_index];
        let mut str = format!("{zoom:.4}");
        if str.ends_with("00") {
            str.truncate(str.len() - 2)
        };
        native.set_zoom_edit_text(&str);
        let mut label = format!("Default {:.4}", self.m_prefs.zooms_dflt[self.m_zoom_index]);
        if label.ends_with("00") {
            label.truncate(label.len() - 2)
        };
        native.set_default_zoom_label(&label);
        self.m_zoom_val_changed = false;
    }
    /// `AppearanceForm::newZoomIndex`.
    pub fn new_zoom_index(&mut self, value: i32, native: &mut dyn AppearanceNativeBoundary) {
        self.unload_zoom_value(native);
        self.m_zoom_index = (value - 1).clamp(0, MAX_ZOOMS as i32 - 1) as usize;
        self.display_current_zoom(native);
    }
    /// `AppearanceForm::unloadZoomValue`.
    pub fn unload_zoom_value(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        if !self.m_zoom_val_changed {
            return;
        };
        let mut zoom = native.zoom_edit_text().parse::<f64>().unwrap_or(0.);
        zoom = zoom.clamp(0.01, 100.);
        let roundfac = if zoom < 1. { 1000. } else { 100. };
        self.m_prefs.zooms[self.m_zoom_index] = (roundfac * zoom + 0.5).floor() / roundfac;
        self.m_prefs.zooms_chgd = true;
    }
    /// `AppearanceForm::newZoomValue`.
    pub fn new_zoom_value(&mut self) {
        self.m_zoom_val_changed = true;
    }
    /// `AppearanceForm::shiftZoomsDown`.
    pub fn shift_zooms_down(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        self.unload_zoom_value(native);
        for i in 0..MAX_ZOOMS - 1 {
            self.m_prefs.zooms[i] = self.m_prefs.zooms[i + 1]
        }
        self.m_prefs.zooms_chgd = true;
        self.display_current_zoom(native);
    }
    /// `AppearanceForm::shiftZoomsUp`.
    pub fn shift_zooms_up(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        self.unload_zoom_value(native);
        for i in (1..MAX_ZOOMS).rev() {
            self.m_prefs.zooms[i] = self.m_prefs.zooms[i - 1]
        }
        self.m_prefs.zooms_chgd = true;
        self.display_current_zoom(native);
    }
    /// `AppearanceForm::restoreDefaultZooms`.
    pub fn restore_default_zooms(&mut self, native: &mut dyn AppearanceNativeBoundary) {
        self.m_prefs.zooms = self.m_prefs.zooms_dflt;
        self.m_prefs.zooms_chgd = true;
        self.display_current_zoom(native);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        zoom: String,
        calls: Vec<String>,
    }
    impl AppearanceNativeBoundary for N {
        fn connect_appearance_signals(&mut self) {}
        fn max_cursor_steps(&self) -> i32 {
            1
        }
        fn set_zoom_index_maximum(&mut self, _: i32) {}
        fn set_spin_box(&mut self, _: i32, _: i32) {}
        fn set_checked(&mut self, _: i32, _: bool) {}
        fn set_enabled(&mut self, _: i32, _: bool) {}
        fn add_style(&mut self, _: &str) {}
        fn style_available(&mut self, _: &str) -> bool {
            true
        }
        fn style_list(&self) -> Vec<String> {
            vec![]
        }
        fn set_style_index(&mut self, _: i32) {}
        fn font_width(&self, _: &str) -> i32 {
            12
        }
        fn set_zoom_edit_maximum_width(&mut self, _: i32) {}
        fn choose_font(&mut self) -> bool {
            true
        }
        fn change_font(&mut self) {}
        fn point_size_changed(&mut self) {
            self.calls.push("points".into())
        }
        fn current_color_index(&self) -> i32 {
            0
        }
        fn choose_color(&mut self, _: u32) -> Option<u32> {
            Some(3)
        }
        fn change_style(&mut self, _: &str) {}
        fn spin_box_value(&self, _: i32) -> i32 {
            0
        }
        fn checked(&self, _: i32) -> bool {
            false
        }
        fn zoom_edit_text(&self) -> String {
            self.zoom.clone()
        }
        fn set_zoom_edit_text(&mut self, x: &str) {
            self.zoom = x.into()
        }
        fn set_default_zoom_label(&mut self, _: &str) {}
        fn user_canceled(&mut self) {}
        fn retranslate_ui(&mut self) {}
    }
    #[test]
    fn zoom_is_clamped_and_rounded() {
        let mut n = N::default();
        let mut f = AppearanceForm::new(ImodPrefStruct::default(), &mut n);
        n.zoom = ".12345".into();
        f.new_zoom_value();
        f.unload_zoom_value(&mut n);
        assert_eq!(f.m_prefs.zooms[f.m_zoom_index], 0.123);
    }
    #[test]
    fn shifting_retains_source_direction() {
        let mut p = ImodPrefStruct::default();
        p.zooms[0] = 2.;
        p.zooms[1] = 3.;
        let mut n = N::default();
        let mut f = AppearanceForm::new(p, &mut n);
        f.shift_zooms_down(&mut n);
        assert_eq!(f.m_prefs.zooms[0], 3.);
        f.shift_zooms_up(&mut n);
        assert_eq!(f.m_prefs.zooms[1], 3.);
    }
    #[test]
    fn point_change_marks_pref_and_renders() {
        let mut n = N::default();
        let mut f = AppearanceForm::new(ImodPrefStruct::default(), &mut n);
        f.image_pt_changed(7, &mut n);
        assert!(f.m_prefs.min_im_pt_size_chgd);
        assert_eq!(n.calls, ["points"]);
    }
}
