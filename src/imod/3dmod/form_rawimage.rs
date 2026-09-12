//! Translation of `IMOD/3dmod/form_rawimage.cpp` and `form_rawimage.h`.
#![allow(dead_code)]

use core::ffi::c_void;

use crate::imod::libiimod::iimage::RawImageInfo;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawImageDataTypeButton {
    SbyteButton,
    ByteButton,
    IntButton,
    UintButton,
    FloatButton,
    ComplexButton,
    RgbButton,
}

pub trait RawImageNativeBoundary {
    fn setup_ui(&mut self);
    fn set_modal(&mut self, modal: bool);
    fn set_always_show_tool_tips(&mut self);
    fn create_button_group(&mut self) -> *mut c_void;
    fn group_add_button(&mut self, group: *mut c_void, button: RawImageDataTypeButton, id: i32);
    fn connect_ok_accept(&mut self);
    fn connect_cancel_reject(&mut self);
    fn connect_scan_toggled_manage_state(&mut self);
    fn connect_group_clicked_manage_state(&mut self, group: *mut c_void);
    fn retranslate_ui(&mut self);
    fn set_file_label(&mut self, value: String);
    fn set_group(&mut self, group: *mut c_void, value: i32);
    fn set_x_size(&mut self, value: i32);
    fn set_y_size(&mut self, value: i32);
    fn set_z_size(&mut self, value: i32);
    fn set_header_size(&mut self, value: i32);
    fn set_swap_checked(&mut self, value: bool);
    fn set_invert_checked(&mut self, value: bool);
    fn set_match_checked(&mut self, value: bool);
    fn set_scan_checked(&mut self, value: bool);
    fn set_min_text(&mut self, value: String);
    fn set_max_text(&mut self, value: String);
    fn checked_group_id(&self, group: *mut c_void) -> Option<i32>;
    fn x_size(&self) -> i32;
    fn y_size(&self) -> i32;
    fn z_size(&self) -> i32;
    fn header_size(&self) -> i32;
    fn swap_checked(&self) -> bool;
    fn invert_checked(&self) -> bool;
    fn match_checked(&self) -> bool;
    fn scan_checked(&self) -> bool;
    fn min_text(&self) -> String;
    fn max_text(&self) -> String;
    fn set_swap_enabled(&mut self, value: bool);
    fn set_scan_enabled(&mut self, value: bool);
    fn set_min_label_enabled(&mut self, value: bool);
    fn set_min_enabled(&mut self, value: bool);
    fn set_max_label_enabled(&mut self, value: bool);
    fn set_max_enabled(&mut self, value: bool);
}

pub struct RawImageForm {
    pub data_type_group: *mut c_void,
}

impl RawImageForm {
    pub fn new(modal: bool, native: &mut dyn RawImageNativeBoundary) -> Self {
        native.setup_ui();
        native.set_modal(modal);
        native.set_always_show_tool_tips();
        let data_type_group = native.create_button_group();
        native.group_add_button(data_type_group, RawImageDataTypeButton::SbyteButton, 0);
        native.group_add_button(data_type_group, RawImageDataTypeButton::ByteButton, 1);
        native.group_add_button(data_type_group, RawImageDataTypeButton::IntButton, 2);
        native.group_add_button(data_type_group, RawImageDataTypeButton::UintButton, 3);
        native.group_add_button(data_type_group, RawImageDataTypeButton::FloatButton, 4);
        native.group_add_button(data_type_group, RawImageDataTypeButton::ComplexButton, 5);
        native.group_add_button(data_type_group, RawImageDataTypeButton::RgbButton, 6);
        native.connect_ok_accept();
        native.connect_cancel_reject();
        native.connect_scan_toggled_manage_state();
        native.connect_group_clicked_manage_state(data_type_group);
        Self { data_type_group }
    }

    pub fn destroy(&mut self) {}

    pub fn language_change(&mut self, native: &mut dyn RawImageNativeBoundary) {
        native.retranslate_ui();
    }

    pub fn load(
        &mut self,
        file_name: String,
        info: &RawImageInfo,
        native: &mut dyn RawImageNativeBoundary,
    ) {
        native.set_file_label(format!("File: {file_name}"));
        native.set_group(self.data_type_group, info.type_);
        native.set_x_size(info.nx);
        native.set_y_size(info.ny);
        native.set_z_size(info.nz);
        native.set_header_size(info.header_size);
        native.set_swap_checked(info.swap_bytes != 0);
        native.set_invert_checked(info.y_inverted != 0);
        native.set_match_checked(info.all_match != 0);
        native.set_scan_checked(info.scan_min_max != 0);
        native.set_min_text(format!("{:.6}", info.amin));
        native.set_max_text(format!("{:.6}", info.amax));
        self.manage_state(native);
    }

    pub fn unload(&self, info: &mut RawImageInfo, native: &dyn RawImageNativeBoundary) {
        if let Some(ind) = native.checked_group_id(self.data_type_group) {
            info.type_ = ind;
        }
        info.nx = native.x_size();
        info.ny = native.y_size();
        info.nz = native.z_size();
        info.header_size = native.header_size();
        info.swap_bytes = native.swap_checked() as i32;
        info.y_inverted = native.invert_checked() as i32;
        info.all_match = native.match_checked() as i32;
        info.scan_min_max = native.scan_checked() as i32;
        info.amin = native.min_text().parse::<f32>().unwrap_or(0.0);
        info.amax = native.max_text().parse::<f32>().unwrap_or(0.0);
    }

    pub fn manage_state(&mut self, native: &mut dyn RawImageNativeBoundary) {
        let Some(which) = native.checked_group_id(self.data_type_group) else {
            return;
        };
        let enab = which != 6 && !native.scan_checked();
        native.set_swap_enabled(which != 0 && which != 6);
        native.set_scan_enabled(which != 6);
        native.set_min_label_enabled(enab);
        native.set_min_enabled(enab);
        native.set_max_label_enabled(enab);
        native.set_max_enabled(enab);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Native {
        group: Option<i32>,
        x: i32,
        y: i32,
        z: i32,
        header: i32,
        swap: bool,
        invert: bool,
        matching: bool,
        scan: bool,
        min: String,
        max: String,
        enabled: [bool; 6],
        buttons: Vec<(RawImageDataTypeButton, i32)>,
        connections: usize,
        label: String,
    }

    impl RawImageNativeBoundary for Native {
        fn setup_ui(&mut self) {}
        fn set_modal(&mut self, _: bool) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn create_button_group(&mut self) -> *mut c_void {
            core::ptr::dangling_mut()
        }
        fn group_add_button(&mut self, _: *mut c_void, button: RawImageDataTypeButton, id: i32) {
            self.buttons.push((button, id));
        }
        fn connect_ok_accept(&mut self) {
            self.connections += 1;
        }
        fn connect_cancel_reject(&mut self) {
            self.connections += 1;
        }
        fn connect_scan_toggled_manage_state(&mut self) {
            self.connections += 1;
        }
        fn connect_group_clicked_manage_state(&mut self, _: *mut c_void) {
            self.connections += 1;
        }
        fn retranslate_ui(&mut self) {}
        fn set_file_label(&mut self, value: String) {
            self.label = value;
        }
        fn set_group(&mut self, _: *mut c_void, value: i32) {
            self.group = Some(value);
        }
        fn set_x_size(&mut self, value: i32) {
            self.x = value;
        }
        fn set_y_size(&mut self, value: i32) {
            self.y = value;
        }
        fn set_z_size(&mut self, value: i32) {
            self.z = value;
        }
        fn set_header_size(&mut self, value: i32) {
            self.header = value;
        }
        fn set_swap_checked(&mut self, value: bool) {
            self.swap = value;
        }
        fn set_invert_checked(&mut self, value: bool) {
            self.invert = value;
        }
        fn set_match_checked(&mut self, value: bool) {
            self.matching = value;
        }
        fn set_scan_checked(&mut self, value: bool) {
            self.scan = value;
        }
        fn set_min_text(&mut self, value: String) {
            self.min = value;
        }
        fn set_max_text(&mut self, value: String) {
            self.max = value;
        }
        fn checked_group_id(&self, _: *mut c_void) -> Option<i32> {
            self.group
        }
        fn x_size(&self) -> i32 {
            self.x
        }
        fn y_size(&self) -> i32 {
            self.y
        }
        fn z_size(&self) -> i32 {
            self.z
        }
        fn header_size(&self) -> i32 {
            self.header
        }
        fn swap_checked(&self) -> bool {
            self.swap
        }
        fn invert_checked(&self) -> bool {
            self.invert
        }
        fn match_checked(&self) -> bool {
            self.matching
        }
        fn scan_checked(&self) -> bool {
            self.scan
        }
        fn min_text(&self) -> String {
            self.min.clone()
        }
        fn max_text(&self) -> String {
            self.max.clone()
        }
        fn set_swap_enabled(&mut self, value: bool) {
            self.enabled[0] = value;
        }
        fn set_scan_enabled(&mut self, value: bool) {
            self.enabled[1] = value;
        }
        fn set_min_label_enabled(&mut self, value: bool) {
            self.enabled[2] = value;
        }
        fn set_min_enabled(&mut self, value: bool) {
            self.enabled[3] = value;
        }
        fn set_max_label_enabled(&mut self, value: bool) {
            self.enabled[4] = value;
        }
        fn set_max_enabled(&mut self, value: bool) {
            self.enabled[5] = value;
        }
    }

    #[test]
    fn constructor_installs_source_group_and_four_connections() {
        let mut native = Native::default();
        let _form = RawImageForm::new(false, &mut native);
        assert_eq!(native.buttons.len(), 7);
        assert_eq!(native.connections, 4);
    }

    #[test]
    fn load_unload_transfers_source_fields_and_qstring_float_conversion() {
        let mut native = Native::default();
        let mut form = RawImageForm::new(false, &mut native);
        let input = RawImageInfo {
            type_: 4,
            nx: 2,
            ny: 3,
            nz: 4,
            swap_bytes: 1,
            header_size: 8,
            amin: 1.25,
            amax: 9.5,
            scan_min_max: 1,
            all_match: 1,
            section_skip: 0,
            y_inverted: 1,
            pixel: 0.,
            z_pixel: 0.,
        };
        form.load("x.raw".into(), &input, &mut native);
        assert_eq!(native.label, "File: x.raw");
        assert_eq!(native.min, "1.250000");
        native.min = "not a float".into();
        let mut output = RawImageInfo {
            type_: 4,
            nx: 2,
            ny: 3,
            nz: 4,
            swap_bytes: 1,
            header_size: 8,
            amin: 1.25,
            amax: 9.5,
            scan_min_max: 1,
            all_match: 1,
            section_skip: 0,
            y_inverted: 1,
            pixel: 0.,
            z_pixel: 0.,
        };
        form.unload(&mut output, &native);
        assert_eq!(output.amin, 0.0);
        assert_eq!(output.amax, 9.5);
    }

    #[test]
    fn manage_state_retains_no_checked_button_early_return_and_rgb_branch() {
        let mut native = Native::default();
        let mut form = RawImageForm::new(false, &mut native);
        form.manage_state(&mut native);
        assert_eq!(native.enabled, [false; 6]);
        native.group = Some(6);
        form.manage_state(&mut native);
        assert_eq!(native.enabled, [false; 6]);
    }

    #[test]
    fn unload_leaves_type_unchanged_when_group_has_no_checked_button() {
        let mut native = Native {
            x: 11,
            ..Default::default()
        };
        let form = RawImageForm::new(false, &mut native);
        let mut output = RawImageInfo {
            type_: 4,
            nx: 2,
            ny: 3,
            nz: 4,
            swap_bytes: 1,
            header_size: 8,
            amin: 1.25,
            amax: 9.5,
            scan_min_max: 1,
            all_match: 1,
            section_skip: 0,
            y_inverted: 1,
            pixel: 0.,
            z_pixel: 0.,
        };
        form.unload(&mut output, &native);
        assert_eq!(output.type_, 4);
        assert_eq!(output.nx, 11);
    }
}
