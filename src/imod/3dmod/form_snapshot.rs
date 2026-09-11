//! Translation of `IMOD/3dmod/form_snapshot.cpp` and `form_snapshot.h`.
//! Native Qt controls/signals are direct boundary calls; preference and format
//! enablement behavior is retained here without a synthetic form.
#![allow(dead_code)]
use core::ffi::c_void;
#[derive(Clone, Debug, Default)]
pub struct ImodPrefStruct {
    pub snap_format: String,
    pub snap_quality: i32,
    pub snap_dpi: i32,
    pub scale_snap_dpi: bool,
    pub no_cur_pnt_on_snaps: bool,
    pub tiff_compression: i32,
    pub tiff_jpeg_quality: i32,
    pub jpeg_for_tiff_images: bool,
}
pub trait SnapshotNativeBoundary {
    fn snap_format_list(&self) -> Vec<String>;
    fn snap_format2(&self, format: &str) -> String;
    fn ctrl_string(&self) -> &str;
    fn add_format_items(&mut self, formats: &[String]);
    fn create_tiff_comp_group(&mut self) -> *mut c_void;
    fn group_add_button(&mut self, group: *mut c_void, id: i32);
    fn connect_snapshot_signals(&mut self);
    fn set_format_index(&mut self, index: i32);
    fn set_spin_box(&mut self, which: i32, value: i32);
    fn set_checked(&mut self, which: i32, value: bool);
    fn set_group(&mut self, group: *mut c_void, value: i32);
    fn set_other_formats_label(&mut self, text: &str);
    fn set_enabled(&mut self, which: i32, enabled: bool);
    fn format_text(&self) -> String;
    fn spin_box_value(&self, which: i32) -> i32;
    fn checked(&self, which: i32) -> bool;
    fn checked_group(&self, group: *mut c_void) -> i32;
    fn retranslate_ui(&mut self);
}
pub const QUALITY: i32 = 0;
pub const DPI: i32 = 1;
pub const TIFF_QUALITY: i32 = 2;
pub const SCALE_DPI: i32 = 0;
pub const NO_POINT: i32 = 1;
pub const JPEG_FOR_IMAGE: i32 = 2;
pub const JPEG_FOR_IMAGE_ENABLE: i32 = 3;
pub const TIFF_QUALITY_LABEL: i32 = 4;
pub const TIFF_QUALITY_SPIN: i32 = 5;
/// `SnapshotForm` (`form_snapshot.h`).
#[derive(Debug)]
pub struct SnapshotForm {
    pub m_prefs: ImodPrefStruct,
    pub tiff_comp_group: *mut c_void,
}
impl SnapshotForm {
    /// `SnapshotForm::SnapshotForm`.
    pub fn new(prefs: ImodPrefStruct, n: &mut dyn SnapshotNativeBoundary) -> Self {
        let mut f = Self {
            m_prefs: prefs,
            tiff_comp_group: core::ptr::null_mut(),
        };
        f.init(n);
        f
    }
    /// `SnapshotForm::~SnapshotForm`.
    pub fn destroy(&mut self) {}
    /// `SnapshotForm::languageChange`.
    pub fn language_change(&mut self, n: &mut dyn SnapshotNativeBoundary) {
        n.retranslate_ui()
    }
    /// `SnapshotForm::init`.
    pub fn init(&mut self, n: &mut dyn SnapshotNativeBoundary) {
        let formats = n.snap_format_list();
        n.add_format_items(&formats);
        self.tiff_comp_group = n.create_tiff_comp_group();
        for id in 0..4 {
            n.group_add_button(self.tiff_comp_group, id)
        }
        n.connect_snapshot_signals();
        self.set_font_dependent_widths(n);
        self.update(n)
    }
    /// `SnapshotForm::setFontDependentWidths`.
    pub fn set_font_dependent_widths(&mut self, _: &mut dyn SnapshotNativeBoundary) {}
    /// `SnapshotForm::update`.
    pub fn update(&mut self, n: &mut dyn SnapshotNativeBoundary) {
        let formats = n.snap_format_list();
        let mut item = 0;
        for (i, format) in formats.iter().enumerate() {
            if *format == self.m_prefs.snap_format {
                item = i as i32
            }
        }
        n.set_format_index(item);
        n.set_spin_box(QUALITY, self.m_prefs.snap_quality);
        n.set_spin_box(DPI, self.m_prefs.snap_dpi);
        n.set_checked(SCALE_DPI, self.m_prefs.scale_snap_dpi);
        n.set_checked(NO_POINT, self.m_prefs.no_cur_pnt_on_snaps);
        self.show_other_formats(item, n);
        n.set_group(self.tiff_comp_group, self.m_prefs.tiff_compression);
        n.set_checked(JPEG_FOR_IMAGE, self.m_prefs.jpeg_for_tiff_images);
        n.set_spin_box(TIFF_QUALITY, self.m_prefs.tiff_jpeg_quality);
        self.set_tiff_comp_enables(n)
    }
    /// `SnapshotForm::unload`.
    pub fn unload(&mut self, n: &mut dyn SnapshotNativeBoundary) {
        self.m_prefs.snap_format = n.format_text();
        self.m_prefs.snap_quality = n.spin_box_value(QUALITY);
        self.m_prefs.snap_dpi = n.spin_box_value(DPI);
        self.m_prefs.scale_snap_dpi = n.checked(SCALE_DPI);
        self.m_prefs.no_cur_pnt_on_snaps = n.checked(NO_POINT);
        self.m_prefs.tiff_compression = n.checked_group(self.tiff_comp_group);
        self.m_prefs.tiff_jpeg_quality = n.spin_box_value(TIFF_QUALITY);
        self.m_prefs.jpeg_for_tiff_images = n.checked(JPEG_FOR_IMAGE)
    }
    /// `SnapshotForm::showOtherFormats`.
    pub fn show_other_formats(&mut self, item: i32, n: &mut dyn SnapshotNativeBoundary) {
        let formats = n.snap_format_list();
        let second = formats
            .get(item.max(0) as usize)
            .map(|format| n.snap_format2(format))
            .unwrap_or_default();
        let ctrl = n.ctrl_string();
        let mut label = format!("{ctrl}+S gives TIFF, {ctrl}-Shift+S ");
        if second.is_empty() {
            label.push_str("will not work")
        } else {
            label.push_str("gives ");
            label.push_str(&second)
        }
        n.set_other_formats_label(&label)
    }
    /// `SnapshotForm::tiffCompChanged`.
    pub fn tiff_comp_changed(&mut self, value: i32, n: &mut dyn SnapshotNativeBoundary) {
        self.m_prefs.tiff_compression = value;
        self.set_tiff_comp_enables(n)
    }
    /// `SnapshotForm::useJpegToggled`.
    pub fn use_jpeg_toggled(&mut self, _: bool, n: &mut dyn SnapshotNativeBoundary) {
        self.set_tiff_comp_enables(n)
    }
    /// `SnapshotForm::setTiffCompEnables`.
    pub fn set_tiff_comp_enables(&mut self, n: &mut dyn SnapshotNativeBoundary) {
        let jpeg = n.checked(JPEG_FOR_IMAGE);
        n.set_enabled(JPEG_FOR_IMAGE_ENABLE, self.m_prefs.tiff_compression < 3);
        n.set_enabled(
            TIFF_QUALITY_LABEL,
            self.m_prefs.tiff_compression == 3 || jpeg,
        );
        n.set_enabled(
            TIFF_QUALITY_SPIN,
            self.m_prefs.tiff_compression == 3 || jpeg,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        checks: [bool; 3],
        enabled: Vec<(i32, bool)>,
        label: String,
        group: u8,
    }
    impl SnapshotNativeBoundary for N {
        fn snap_format_list(&self) -> Vec<String> {
            vec!["JPEG".into(), "PNG".into()]
        }
        fn snap_format2(&self, x: &str) -> String {
            if x == "JPEG" {
                "PNG".into()
            } else {
                String::new()
            }
        }
        fn ctrl_string(&self) -> &str {
            "Ctrl"
        }
        fn add_format_items(&mut self, _: &[String]) {}
        fn create_tiff_comp_group(&mut self) -> *mut c_void {
            &mut self.group as *mut u8 as *mut c_void
        }
        fn group_add_button(&mut self, _: *mut c_void, _: i32) {}
        fn connect_snapshot_signals(&mut self) {}
        fn set_format_index(&mut self, _: i32) {}
        fn set_spin_box(&mut self, _: i32, _: i32) {}
        fn set_checked(&mut self, x: i32, y: bool) {
            self.checks[x as usize] = y
        }
        fn set_group(&mut self, _: *mut c_void, _: i32) {}
        fn set_other_formats_label(&mut self, x: &str) {
            self.label = x.into()
        }
        fn set_enabled(&mut self, x: i32, y: bool) {
            self.enabled.push((x, y))
        }
        fn format_text(&self) -> String {
            "JPEG".into()
        }
        fn spin_box_value(&self, _: i32) -> i32 {
            0
        }
        fn checked(&self, x: i32) -> bool {
            self.checks[x as usize]
        }
        fn checked_group(&self, _: *mut c_void) -> i32 {
            2
        }
        fn retranslate_ui(&mut self) {}
    }
    #[test]
    fn enables_match_tiff_compression_rule() {
        let mut n = N::default();
        let mut f = SnapshotForm::new(
            ImodPrefStruct {
                tiff_compression: 3,
                ..Default::default()
            },
            &mut n,
        );
        assert!(
            n.enabled
                .iter()
                .any(|x| *x == (JPEG_FOR_IMAGE_ENABLE, false))
        );
        assert!(n.enabled.iter().any(|x| *x == (TIFF_QUALITY_SPIN, true)));
        f.tiff_comp_changed(2, &mut n);
        assert!(
            n.enabled
                .iter()
                .any(|x| *x == (JPEG_FOR_IMAGE_ENABLE, true))
        );
    }
    #[test]
    fn other_format_label_matches_source() {
        let mut n = N::default();
        let mut f = SnapshotForm::new(ImodPrefStruct::default(), &mut n);
        f.show_other_formats(0, &mut n);
        assert_eq!(n.label, "Ctrl+S gives TIFF, Ctrl-Shift+S gives PNG");
    }
}
