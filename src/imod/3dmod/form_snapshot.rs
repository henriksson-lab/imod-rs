//! Translation of `IMOD/3dmod/form_snapshot.cpp` and `form_snapshot.h`.
#![allow(dead_code)]

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TiffCompressionButton {
    NoCompRadio,
    ZipCompRadio,
    LzwCompRadio,
    JpegCompRadio,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotCheckBox {
    ScaleDpiCheckBox,
    NoCurPntCheckBox,
    JpegForImageCheckBox,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotSpinBox {
    QualitySpinBox,
    DpiSpinBox,
    TiffQualitySpinBox,
}

/// The `QButtonGroup *` (and other native object) arguments below are opaque
/// native identities, spelled the way `DockingDialogNativeBoundary` spells
/// them: a frontend which owns Qt can use its pointer cast to `usize`, a
/// non-Qt frontend a stable application handle, and `0` is the source's null.
pub trait SnapshotNativeBoundary {
    fn setup_ui(&mut self);
    fn snap_format_list(&self) -> Vec<String>;
    fn snap_format2(&self, format: &str) -> String;
    fn ctrl_string(&self) -> &str;
    fn add_format_items(&mut self, formats: &[String]);
    fn create_tiff_comp_group(&mut self) -> usize;
    fn group_add_button(&mut self, group: usize, button: TiffCompressionButton, id: i32);
    fn connect_tiff_comp_group_clicked(&mut self, group: usize);
    fn connect_jpeg_for_image_toggled(&mut self);
    fn connect_format_activated_show_other_formats(&mut self);
    fn set_format_index(&mut self, index: i32);
    fn set_spin_box(&mut self, control: SnapshotSpinBox, value: i32);
    fn set_checked(&mut self, control: SnapshotCheckBox, value: bool);
    fn set_group(&mut self, group: usize, value: i32);
    fn set_other_formats_label(&mut self, text: String);
    fn set_jpeg_for_image_enabled(&mut self, enabled: bool);
    fn set_tiff_jpeg_quality_label_enabled(&mut self, enabled: bool);
    fn set_tiff_quality_enabled(&mut self, enabled: bool);
    fn format_text(&self) -> String;
    fn spin_box_value(&self, control: SnapshotSpinBox) -> i32;
    fn checked(&self, control: SnapshotCheckBox) -> bool;
    fn checked_group(&self, group: usize) -> i32;
    fn retranslate_ui(&mut self);
}

#[derive(Debug)]
pub struct SnapshotForm {
    pub m_prefs: ImodPrefStruct,
    pub tiff_comp_group: usize,
}

impl SnapshotForm {
    pub fn new(prefs: ImodPrefStruct, native: &mut dyn SnapshotNativeBoundary) -> Self {
        native.setup_ui();
        let mut form = Self {
            m_prefs: prefs,
            tiff_comp_group: 0,
        };
        form.init(native);
        form
    }

    pub fn destroy(&mut self) {}

    pub fn language_change(&mut self, native: &mut dyn SnapshotNativeBoundary) {
        native.retranslate_ui();
    }

    pub fn init(&mut self, native: &mut dyn SnapshotNativeBoundary) {
        let formats = native.snap_format_list();
        native.add_format_items(&formats);
        self.tiff_comp_group = native.create_tiff_comp_group();
        native.group_add_button(self.tiff_comp_group, TiffCompressionButton::NoCompRadio, 0);
        native.group_add_button(self.tiff_comp_group, TiffCompressionButton::ZipCompRadio, 1);
        native.group_add_button(self.tiff_comp_group, TiffCompressionButton::LzwCompRadio, 2);
        native.group_add_button(
            self.tiff_comp_group,
            TiffCompressionButton::JpegCompRadio,
            3,
        );
        native.connect_tiff_comp_group_clicked(self.tiff_comp_group);
        native.connect_jpeg_for_image_toggled();
        self.set_font_dependent_widths();
        self.update(native);
    }

    pub fn set_font_dependent_widths(&mut self) {}

    pub fn update(&mut self, native: &mut dyn SnapshotNativeBoundary) {
        let formats = native.snap_format_list();
        let mut item = 0;
        for (index, format) in formats.iter().enumerate() {
            if *format == self.m_prefs.snap_format {
                item = index as i32;
            }
        }
        native.set_format_index(item);
        native.set_spin_box(SnapshotSpinBox::QualitySpinBox, self.m_prefs.snap_quality);
        native.set_spin_box(SnapshotSpinBox::DpiSpinBox, self.m_prefs.snap_dpi);
        native.set_checked(
            SnapshotCheckBox::ScaleDpiCheckBox,
            self.m_prefs.scale_snap_dpi,
        );
        native.set_checked(
            SnapshotCheckBox::NoCurPntCheckBox,
            self.m_prefs.no_cur_pnt_on_snaps,
        );
        self.show_other_formats(item, native);
        native.connect_format_activated_show_other_formats();
        native.set_group(self.tiff_comp_group, self.m_prefs.tiff_compression);
        native.set_checked(
            SnapshotCheckBox::JpegForImageCheckBox,
            self.m_prefs.jpeg_for_tiff_images,
        );
        native.set_spin_box(
            SnapshotSpinBox::TiffQualitySpinBox,
            self.m_prefs.tiff_jpeg_quality,
        );
        self.set_tiff_comp_enables(native);
    }

    pub fn unload(&mut self, native: &mut dyn SnapshotNativeBoundary) {
        self.m_prefs.snap_format = native.format_text();
        self.m_prefs.snap_quality = native.spin_box_value(SnapshotSpinBox::QualitySpinBox);
        self.m_prefs.snap_dpi = native.spin_box_value(SnapshotSpinBox::DpiSpinBox);
        self.m_prefs.scale_snap_dpi = native.checked(SnapshotCheckBox::ScaleDpiCheckBox);
        self.m_prefs.no_cur_pnt_on_snaps = native.checked(SnapshotCheckBox::NoCurPntCheckBox);
        self.m_prefs.tiff_compression = native.checked_group(self.tiff_comp_group);
        self.m_prefs.tiff_jpeg_quality = native.spin_box_value(SnapshotSpinBox::TiffQualitySpinBox);
        self.m_prefs.jpeg_for_tiff_images = native.checked(SnapshotCheckBox::JpegForImageCheckBox);
    }

    pub fn show_other_formats(&mut self, item: i32, native: &mut dyn SnapshotNativeBoundary) {
        let formats = native.snap_format_list();
        let second = native.snap_format2(&formats[item as usize]);
        let ctrl = native.ctrl_string();
        let mut label = format!("{ctrl}+S gives TIFF, {ctrl}-Shift+S ");
        if second != "" {
            label += "gives ";
            label += &second;
        } else {
            label += "will not work";
        }
        native.set_other_formats_label(label);
    }

    pub fn tiff_comp_changed(&mut self, value: i32, native: &mut dyn SnapshotNativeBoundary) {
        self.m_prefs.tiff_compression = value;
        self.set_tiff_comp_enables(native);
    }

    pub fn use_jpeg_toggled(&mut self, _state: bool, native: &mut dyn SnapshotNativeBoundary) {
        self.set_tiff_comp_enables(native);
    }

    pub fn set_tiff_comp_enables(&mut self, native: &mut dyn SnapshotNativeBoundary) {
        native.set_jpeg_for_image_enabled(self.m_prefs.tiff_compression < 3);
        let enabled = self.m_prefs.tiff_compression == 3
            || native.checked(SnapshotCheckBox::JpegForImageCheckBox);
        native.set_tiff_jpeg_quality_label_enabled(enabled);
        native.set_tiff_quality_enabled(enabled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Native {
        checks: [bool; 3],
        enabled: [bool; 3],
        label: String,
        buttons: Vec<(TiffCompressionButton, i32)>,
        tiff_connections: usize,
        jpeg_connections: usize,
        format_connections: usize,
    }

    impl SnapshotNativeBoundary for Native {
        fn setup_ui(&mut self) {}
        fn snap_format_list(&self) -> Vec<String> {
            vec!["JPEG".into(), "PNG".into()]
        }
        fn snap_format2(&self, format: &str) -> String {
            if format == "JPEG" {
                "PNG".into()
            } else {
                String::new()
            }
        }
        fn ctrl_string(&self) -> &str {
            "Ctrl"
        }
        fn add_format_items(&mut self, _: &[String]) {}
        fn create_tiff_comp_group(&mut self) -> usize {
            1
        }
        fn group_add_button(&mut self, _: usize, b: TiffCompressionButton, id: i32) {
            self.buttons.push((b, id));
        }
        fn connect_tiff_comp_group_clicked(&mut self, _: usize) {
            self.tiff_connections += 1;
        }
        fn connect_jpeg_for_image_toggled(&mut self) {
            self.jpeg_connections += 1;
        }
        fn connect_format_activated_show_other_formats(&mut self) {
            self.format_connections += 1;
        }
        fn set_format_index(&mut self, _: i32) {}
        fn set_spin_box(&mut self, _: SnapshotSpinBox, _: i32) {}
        fn set_checked(&mut self, c: SnapshotCheckBox, v: bool) {
            self.checks[c as usize] = v;
        }
        fn set_group(&mut self, _: usize, _: i32) {}
        fn set_other_formats_label(&mut self, text: String) {
            self.label = text;
        }
        fn set_jpeg_for_image_enabled(&mut self, v: bool) {
            self.enabled[0] = v;
        }
        fn set_tiff_jpeg_quality_label_enabled(&mut self, v: bool) {
            self.enabled[1] = v;
        }
        fn set_tiff_quality_enabled(&mut self, v: bool) {
            self.enabled[2] = v;
        }
        fn format_text(&self) -> String {
            "JPEG".into()
        }
        fn spin_box_value(&self, _: SnapshotSpinBox) -> i32 {
            0
        }
        fn checked(&self, c: SnapshotCheckBox) -> bool {
            self.checks[c as usize]
        }
        fn checked_group(&self, _: usize) -> i32 {
            2
        }
        fn retranslate_ui(&mut self) {}
    }

    #[test]
    fn init_and_each_update_follow_source_tiff_and_format_connections() {
        let mut native = Native::default();
        let mut form = SnapshotForm::new(
            ImodPrefStruct {
                tiff_compression: 3,
                ..Default::default()
            },
            &mut native,
        );
        assert_eq!(native.buttons.len(), 4);
        assert_eq!(
            (
                native.tiff_connections,
                native.jpeg_connections,
                native.format_connections
            ),
            (1, 1, 1)
        );
        assert_eq!(native.enabled, [false, true, true]);
        form.update(&mut native);
        assert_eq!(native.format_connections, 2);
    }

    #[test]
    fn other_format_label_and_tiff_enablement_match_source() {
        let mut native = Native::default();
        let mut form = SnapshotForm::new(ImodPrefStruct::default(), &mut native);
        form.show_other_formats(0, &mut native);
        assert_eq!(native.label, "Ctrl+S gives TIFF, Ctrl-Shift+S gives PNG");
        native.checks[SnapshotCheckBox::JpegForImageCheckBox as usize] = true;
        form.tiff_comp_changed(2, &mut native);
        assert_eq!(native.enabled, [true, true, true]);
    }

    #[test]
    fn unload_reads_each_source_preference_control() {
        let mut native = Native::default();
        let mut form = SnapshotForm::new(ImodPrefStruct::default(), &mut native);
        native.checks[SnapshotCheckBox::ScaleDpiCheckBox as usize] = true;
        native.checks[SnapshotCheckBox::NoCurPntCheckBox as usize] = true;
        form.unload(&mut native);
        assert_eq!(form.m_prefs.snap_format, "JPEG");
        assert!(form.m_prefs.scale_snap_dpi);
        assert!(form.m_prefs.no_cur_pnt_on_snaps);
        assert_eq!(form.m_prefs.tiff_compression, 2);
    }
}
