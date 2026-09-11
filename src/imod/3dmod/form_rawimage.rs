//! Translation of `IMOD/3dmod/form_rawimage.cpp` and `form_rawimage.h`.
use crate::imod::libiimod::iimage::RawImageInfo;
#[derive(Clone, Debug, Default)]
pub struct RawImageForm {
    pub modal: bool,
    pub tooltips: bool,
    pub accepted: Option<bool>,
    pub file_label: String,
    pub data_type: i32,
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub header_size: i32,
    pub swap: bool,
    pub invert: bool,
    pub all_match: bool,
    pub scan: bool,
    pub amin: f32,
    pub amax: f32,
    pub swap_enabled: bool,
    pub scan_enabled: bool,
    pub limits_enabled: bool,
}
pub fn raw_image_form_new(modal: bool) -> RawImageForm {
    RawImageForm {
        modal,
        tooltips: true,
        ..Default::default()
    }
}
impl RawImageForm {
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self) {}
    pub fn load(&mut self, file: &str, info: &RawImageInfo) {
        self.file_label = format!("File: {file}");
        self.data_type = info.type_;
        self.nx = info.nx;
        self.ny = info.ny;
        self.nz = info.nz;
        self.header_size = info.header_size;
        self.swap = info.swap_bytes != 0;
        self.invert = info.y_inverted != 0;
        self.all_match = info.all_match != 0;
        self.scan = info.scan_min_max != 0;
        self.amin = info.amin;
        self.amax = info.amax;
        self.manage_state()
    }
    pub fn unload(&self, info: &mut RawImageInfo) {
        info.type_ = self.data_type;
        info.nx = self.nx;
        info.ny = self.ny;
        info.nz = self.nz;
        info.header_size = self.header_size;
        info.swap_bytes = self.swap as i32;
        info.y_inverted = self.invert as i32;
        info.all_match = self.all_match as i32;
        info.scan_min_max = self.scan as i32;
        info.amin = self.amin;
        info.amax = self.amax
    }
    pub fn manage_state(&mut self) {
        self.swap_enabled = self.data_type != 0 && self.data_type != 6;
        self.scan_enabled = self.data_type != 6;
        self.limits_enabled = self.data_type != 6 && !self.scan
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn load_unload_source_fields() {
        let mut i = RawImageInfo {
            type_: 4,
            nx: 2,
            ny: 0,
            nz: 0,
            swap_bytes: 0,
            header_size: 0,
            amin: 0.,
            amax: 0.,
            scan_min_max: 1,
            all_match: 0,
            section_skip: 0,
            y_inverted: 0,
            pixel: 0.,
            z_pixel: 0.,
        };
        let mut f = raw_image_form_new(false);
        f.load("x.raw", &i);
        assert!(!f.limits_enabled);
        let mut o = RawImageInfo {
            type_: 0,
            nx: 0,
            ny: 0,
            nz: 0,
            swap_bytes: 0,
            header_size: 0,
            amin: 0.,
            amax: 0.,
            scan_min_max: 0,
            all_match: 0,
            section_skip: 0,
            y_inverted: 0,
            pixel: 0.,
            z_pixel: 0.,
        };
        f.unload(&mut o);
        assert_eq!(o.type_, 4);
    }
}
