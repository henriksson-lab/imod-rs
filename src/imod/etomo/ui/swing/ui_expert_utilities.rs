//! `IMOD/Etomo/src/etomo/ui/swing/UIExpertUtilities.java`.
#![allow(dead_code)]
pub trait MrcHeaderBoundary {
    fn read(&mut self) -> Result<bool, ()>;
    fn x_pixel_spacing(&self) -> f64;
    fn n_columns(&self) -> i32;
}
pub trait ConstTiltParamBoundary {
    fn full_image_x(&self) -> i32;
}
pub trait FidXyzBoundary {
    fn read(&mut self) -> Result<(), ()>;
    fn exists(&self) -> bool;
    fn is_empty(&self) -> bool;
    fn pixel_size_set(&self) -> bool;
    fn pixel_size(&self) -> f64;
}
pub trait FiducialessManager {
    fn set_fiducialess_alignment(&mut self, value: bool);
    fn set_image_rotation(&mut self, value: String);
    fn write_rotation_xf(&mut self, contents: String) -> Result<(), String>;
    fn com_scripts_created(&self) -> bool;
    fn roll_align_com_angles(&mut self);
    fn roll_tilt_com_angles(&mut self);
}
/// Java singleton `UIExpertUtilities.INSTANCE`.
#[derive(Clone, Copy, Debug, Default)]
pub struct UiExpertUtilities;
impl UiExpertUtilities {
    pub const INSTANCE: Self = Self;
    pub fn get_stack_binning<H: MrcHeaderBoundary>(
        &self,
        raw: &mut H,
        stack: &mut H,
        null_if_failed: bool,
    ) -> Option<i32> {
        let default = if null_if_failed { None } else { Some(1) };
        if !raw.read().ok()? || !stack.read().ok()? {
            return default;
        }
        let mut value = default.unwrap_or(i32::MIN);
        if raw.x_pixel_spacing() > 0.0 {
            value = (stack.x_pixel_spacing() / raw.x_pixel_spacing()).round() as i32
        }
        if value != default.unwrap_or(i32::MIN) && value < 1 {
            Some(1)
        } else {
            Some(value)
        }
    }
    pub fn get_stack_binning_from_file_name<H: MrcHeaderBoundary>(
        &self,
        name: Option<&str>,
        raw: &mut H,
        stack: &mut H,
        null_if_failed: bool,
    ) -> Option<i32> {
        if name.is_none_or(|v| v.trim().is_empty()) {
            Some(1)
        } else {
            self.get_stack_binning(raw, stack, null_if_failed)
        }
    }
    pub fn update_fiducialess_params<M: FiducialessManager>(
        &self,
        manager: &mut M,
        image_rotation: &str,
        fiducialess: bool,
    ) -> Result<(), String> {
        if image_rotation.trim().is_empty() {
            return Err("Missing tilt axis rotation value.  Make sure that the aligned stack has been created.".into());
        }
        let angle: f64 = image_rotation
            .parse()
            .map_err(|e: std::num::ParseFloatError| {
                format!("Tilt axis rotation format error: {e}")
            })?;
        manager.set_fiducialess_alignment(fiducialess);
        manager.set_image_rotation(angle.to_string());
        let rads = -angle * std::f64::consts::PI / 180.0;
        manager.write_rotation_xf(format!(
            "{}   {}   {}   {}   0   0\n",
            rads.cos(),
            (-rads).sin(),
            rads.sin(),
            rads.cos()
        ))
    }
    pub fn are_scripts_created<M: FiducialessManager>(&self, manager: &M) -> bool {
        manager.com_scripts_created()
    }
    pub fn get_backward_compatible_align_binning<H: MrcHeaderBoundary, F: FidXyzBoundary>(
        &self,
        raw: &mut H,
        preali: &mut H,
        fid: &mut F,
    ) -> i32 {
        if raw.read().ok() != Some(true) || raw.x_pixel_spacing() <= 0.0 {
            return 1;
        }
        let failed =
            fid.read().is_err() || !fid.exists() || fid.is_empty() || !fid.pixel_size_set();
        if !failed {
            return (fid.pixel_size() / raw.x_pixel_spacing()).round().max(1.0) as i32;
        }
        if preali.read().ok() != Some(true) {
            return 1;
        }
        (preali.x_pixel_spacing() / raw.x_pixel_spacing())
            .round()
            .max(1.0) as i32
    }
    pub fn get_backward_compatible_tilt_binning<H: MrcHeaderBoundary, T: ConstTiltParamBoundary>(
        &self,
        raw: &mut H,
        tilt: &T,
    ) -> i32 {
        if raw.read().ok() != Some(true) {
            return 1;
        }
        let full = tilt.full_image_x();
        if full > 0 {
            (raw.n_columns() as f64 / full as f64).round().max(1.0) as i32
        } else {
            1
        }
    }
    pub fn roll_align_com_angles<M: FiducialessManager>(&self, manager: &mut M) {
        manager.roll_align_com_angles()
    }
    pub fn roll_tilt_com_angles<M: FiducialessManager>(&self, manager: &mut M) {
        manager.roll_tilt_com_angles()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct H {
        ok: bool,
        x: f64,
        n: i32,
    }
    impl MrcHeaderBoundary for H {
        fn read(&mut self) -> Result<bool, ()> {
            Ok(self.ok)
        }
        fn x_pixel_spacing(&self) -> f64 {
            self.x
        }
        fn n_columns(&self) -> i32 {
            self.n
        }
    }
    #[test]
    fn stack_binning_rounds() {
        let mut raw = H {
            ok: true,
            x: 1.0,
            n: 100,
        };
        let mut stack = H {
            ok: true,
            x: 2.6,
            n: 0,
        };
        assert_eq!(
            UiExpertUtilities::INSTANCE.get_stack_binning(&mut raw, &mut stack, false),
            Some(3)
        );
    }
}
