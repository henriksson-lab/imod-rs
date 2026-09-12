//! `IMOD/Etomo/src/etomo/ui/swing/TrimvolDisplay.java`.
#![allow(dead_code)]

use super::trimvol_panel::TrimvolPanel;

/// Java `TrimvolDisplay`.
pub trait TrimvolDisplay {
    fn set_swap_yz(&mut self, input: bool);
    fn set_rotate_x(&mut self, input: bool);
    fn set_convert_to_bytes(&mut self, input: bool);
    fn set_section_scale_min(&mut self, input: &str);
    fn set_section_scale_max(&mut self, input: &str);
    fn set_x_min(&mut self, input: &str);
    fn set_x_max(&mut self, input: &str);
    fn set_y_min(&mut self, input: &str);
    fn set_y_max(&mut self, input: &str);
    fn set_z_min(&mut self, input: &str);
    fn set_z_max(&mut self, input: &str);
    fn set_scale_x_min(&mut self, input: &str);
    fn set_scale_y_min(&mut self, input: &str);
    fn set_scale_x_max(&mut self, input: &str);
    fn set_scale_y_max(&mut self, input: &str);
}

impl TrimvolDisplay for TrimvolPanel {
    fn set_swap_yz(&mut self, input: bool) {
        TrimvolPanel::set_swap_yz(self, input)
    }
    fn set_rotate_x(&mut self, input: bool) {
        TrimvolPanel::set_rotate_x(self, input)
    }
    fn set_convert_to_bytes(&mut self, input: bool) {
        TrimvolPanel::set_convert_to_bytes(self, input)
    }
    fn set_section_scale_min(&mut self, input: &str) {
        TrimvolPanel::set_section_scale_min(self, input)
    }
    fn set_section_scale_max(&mut self, input: &str) {
        TrimvolPanel::set_section_scale_max(self, input)
    }
    fn set_x_min(&mut self, input: &str) {
        TrimvolPanel::set_x_min(self, input)
    }
    fn set_x_max(&mut self, input: &str) {
        TrimvolPanel::set_x_max(self, input)
    }
    fn set_y_min(&mut self, input: &str) {
        TrimvolPanel::set_y_min(self, input)
    }
    fn set_y_max(&mut self, input: &str) {
        TrimvolPanel::set_y_max(self, input)
    }
    fn set_z_min(&mut self, input: &str) {
        TrimvolPanel::set_z_min(self, input)
    }
    fn set_z_max(&mut self, input: &str) {
        TrimvolPanel::set_z_max(self, input)
    }
    fn set_scale_x_min(&mut self, input: &str) {
        TrimvolPanel::set_scale_x_min(self, input)
    }
    fn set_scale_y_min(&mut self, input: &str) {
        TrimvolPanel::set_scale_y_min(self, input)
    }
    fn set_scale_x_max(&mut self, input: &str) {
        TrimvolPanel::set_scale_x_max(self, input)
    }
    fn set_scale_y_max(&mut self, input: &str) {
        TrimvolPanel::set_scale_y_max(self, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Display {
        x: String,
        swap: bool,
    }
    impl TrimvolDisplay for Display {
        fn set_swap_yz(&mut self, v: bool) {
            self.swap = v
        }
        fn set_rotate_x(&mut self, _: bool) {}
        fn set_convert_to_bytes(&mut self, _: bool) {}
        fn set_section_scale_min(&mut self, _: &str) {}
        fn set_section_scale_max(&mut self, _: &str) {}
        fn set_x_min(&mut self, v: &str) {
            self.x = v.into()
        }
        fn set_x_max(&mut self, _: &str) {}
        fn set_y_min(&mut self, _: &str) {}
        fn set_y_max(&mut self, _: &str) {}
        fn set_z_min(&mut self, _: &str) {}
        fn set_z_max(&mut self, _: &str) {}
        fn set_scale_x_min(&mut self, _: &str) {}
        fn set_scale_y_min(&mut self, _: &str) {}
        fn set_scale_x_max(&mut self, _: &str) {}
        fn set_scale_y_max(&mut self, _: &str) {}
    }
    #[test]
    fn display_writes_boolean_and_string_values() {
        let mut d = Display::default();
        d.set_swap_yz(true);
        d.set_x_min("2");
        assert!(d.swap);
        assert_eq!(d.x, "2");
    }
}
