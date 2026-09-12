//! `IMOD/Etomo/src/etomo/ui/swing/VolumeRangePanel.java`.
//!
//! The Swing `GridLayout`, border, and child containers remain an explicit
//! presentation boundary.  This unit retains the Java panel's fields, value
//! transfer order, validation behavior, and rubberband parsing exactly.
#![allow(dead_code)]

use super::{
    etomo_panel::{EtomoPanel, TitledBorder},
    labeled_text_field::LabeledTextField,
};
use crate::imod::etomo::{
    process::imod_process::RUBBERBAND_RESULTS_STRING, ui::field_type::FieldType,
};

/// Java public static `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `TrimvolParam` calls reached by `VolumeRangePanel`.
pub trait VolumeRangeTrimvolParam {
    fn get_x_min(&self) -> String;
    fn get_x_max(&self) -> String;
    fn get_y_min(&self) -> String;
    fn get_y_max(&self) -> String;
    fn get_z_min(&self) -> String;
    fn get_z_max(&self) -> String;
    fn set_x_min(&mut self, value: String);
    fn set_x_max(&mut self, value: String);
    fn set_y_min(&mut self, value: String);
    fn set_y_max(&mut self, value: String);
    fn set_z_min(&mut self, value: String);
    fn set_z_max(&mut self, value: String);
}

/// Java `ConstMetaData` getters used by `setParameters`.
pub trait VolumeRangeConstMetaData {
    fn get_post_trimvol_x_min(&self) -> String;
    fn get_post_trimvol_x_max(&self) -> String;
    fn get_post_trimvol_y_min(&self) -> String;
    fn get_post_trimvol_y_max(&self) -> String;
    fn get_post_trimvol_z_min(&self) -> String;
    fn get_post_trimvol_z_max(&self) -> String;
}

/// Java `MetaData` setters used by the two `getParameters` overloads.
pub trait VolumeRangeMetaData {
    fn set_post_trimvol_x_min(&mut self, value: String);
    fn set_post_trimvol_x_max(&mut self, value: String);
    fn set_post_trimvol_y_min(&mut self, value: String);
    fn set_post_trimvol_y_max(&mut self, value: String);
    fn set_post_trimvol_z_min(&mut self, value: String);
    fn set_post_trimvol_z_max(&mut self, value: String);
    fn set_post_trimvol_new_style_z(&mut self, min: String, max: String);
}

/// Source-visible `GridLayout(3, 2, 5, 5)` setup at the Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VolumeRangePanelLayout {
    pub rows: i32,
    pub columns: i32,
    pub horizontal_gap: i32,
    pub vertical_gap: i32,
    pub child_order: [String; 6],
}

/// Java `VolumeRangePanel` source-visible state.
#[derive(Clone, Debug)]
pub struct VolumeRangePanel {
    pub pnl_root: EtomoPanel,
    pub ltf_x_min: LabeledTextField,
    pub ltf_x_max: LabeledTextField,
    pub ltf_y_min: LabeledTextField,
    pub ltf_y_max: LabeledTextField,
    pub ltf_z_min: LabeledTextField,
    pub ltf_z_max: LabeledTextField,
    pub lock_panel: bool,
    pub layout: Option<VolumeRangePanelLayout>,
}

impl VolumeRangePanel {
    /// Java private `VolumeRangePanel(boolean)`.
    fn new(lock_panel: bool) -> Self {
        Self {
            pnl_root: EtomoPanel::default(),
            ltf_x_min: LabeledTextField::new(FieldType::Integer, "X min: "),
            ltf_x_max: LabeledTextField::new(FieldType::Integer, "X max: "),
            ltf_y_min: LabeledTextField::new(FieldType::Integer, "Y min: "),
            ltf_y_max: LabeledTextField::new(FieldType::Integer, "Y max: "),
            ltf_z_min: LabeledTextField::new(FieldType::Integer, "Z min: "),
            ltf_z_max: LabeledTextField::new(FieldType::Integer, "Z max: "),
            lock_panel,
            layout: None,
        }
    }

    /// Java static `getInstance(boolean)`.
    pub fn get_instance(lock_panel: bool) -> Self {
        let mut instance = Self::new(lock_panel);
        instance.create_panel();
        instance.set_tooltips();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        self.pnl_root.set_border(TitledBorder {
            title: "Volume Range".into(),
        });
        self.layout = Some(VolumeRangePanelLayout {
            rows: 3,
            columns: 2,
            horizontal_gap: 5,
            vertical_gap: 5,
            child_order: [
                "ltfXMin".into(),
                "ltfXMax".into(),
                "ltfYMin".into(),
                "ltfYMax".into(),
                "ltfZMin".into(),
                "ltfZMax".into(),
            ],
        });
    }

    /// Java `getComponent()`; the concrete Swing `Component` remains the
    /// `EtomoPanel` presentation boundary.
    pub fn get_component(&self) -> &EtomoPanel {
        &self.pnl_root
    }

    /// Java `initParameters(TrimvolParam)`.
    pub fn init_parameters<P: VolumeRangeTrimvolParam>(&mut self, param: &P) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min.set_text(&param.get_x_min());
        self.ltf_x_max.set_text(&param.get_x_max());
        self.ltf_y_min.set_text(&param.get_y_min());
        self.ltf_y_max.set_text(&param.get_y_max());
        self.ltf_z_min.set_text(&param.get_z_min());
        self.ltf_z_max.set_text(&param.get_z_max());
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters<M: VolumeRangeConstMetaData>(&mut self, meta_data: &M) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min.set_text(&meta_data.get_post_trimvol_x_min());
        self.ltf_x_max.set_text(&meta_data.get_post_trimvol_x_max());
        self.ltf_y_min.set_text(&meta_data.get_post_trimvol_y_min());
        self.ltf_y_max.set_text(&meta_data.get_post_trimvol_y_max());
        self.ltf_z_min.set_text(&meta_data.get_post_trimvol_z_min());
        self.ltf_z_max.set_text(&meta_data.get_post_trimvol_z_max());
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_parameters<M: VolumeRangeMetaData>(&self, meta_data: &mut M) {
        if self.lock_panel {
            return;
        }
        meta_data.set_post_trimvol_x_min(self.ltf_x_min.get_text());
        meta_data.set_post_trimvol_x_max(self.ltf_x_max.get_text());
        meta_data.set_post_trimvol_y_min(self.ltf_y_min.get_text());
        meta_data.set_post_trimvol_y_max(self.ltf_y_max.get_text());
        meta_data.set_post_trimvol_z_min(self.ltf_z_min.get_text());
        meta_data.set_post_trimvol_z_max(self.ltf_z_max.get_text());
    }

    /// Java `getParametersForTrimvol(MetaData)`.
    pub fn get_parameters_for_trimvol<M: VolumeRangeMetaData>(&self, meta_data: &mut M) {
        if self.lock_panel {
            return;
        }
        meta_data
            .set_post_trimvol_new_style_z(self.ltf_z_min.get_text(), self.ltf_z_max.get_text());
    }

    /// Java `getParameters(TrimvolParam, boolean)`.
    pub fn get_parameters_trimvol<P: VolumeRangeTrimvolParam>(
        &self,
        trimvol_param: &mut P,
        do_validation: bool,
    ) -> bool {
        if self.lock_panel {
            return true;
        }
        let result = (|| {
            trimvol_param.set_x_min(self.ltf_x_min.get_text_validated(do_validation)?);
            trimvol_param.set_x_max(self.ltf_x_max.get_text_validated(do_validation)?);
            trimvol_param.set_y_min(self.ltf_y_min.get_text_validated(do_validation)?);
            trimvol_param.set_y_max(self.ltf_y_max.get_text_validated(do_validation)?);
            trimvol_param.set_z_min(self.ltf_z_min.get_text_validated(do_validation)?);
            trimvol_param.set_z_max(self.ltf_z_max.get_text_validated(do_validation)?);
            Ok::<(), super::labeled_text_field::FieldValidationFailedException>(())
        })();
        result.is_ok()
    }

    /// Java `setXYMinAndMax(Vector)`.
    pub fn set_xy_min_and_max(&mut self, coordinates: Option<&[String]>) {
        let Some(coordinates) = coordinates else {
            return;
        };
        let size = coordinates.len();
        if size == 0 {
            return;
        }
        let mut index = 0;
        while index < size {
            if coordinates[index] == RUBBERBAND_RESULTS_STRING {
                index += 1;
                let Some(value) = coordinates.get(index) else {
                    return;
                };
                self.ltf_x_min.set_text(value);
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_y_min.set_text(&coordinates[index]);
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_x_max.set_text(&coordinates[index]);
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_y_max.set_text(&coordinates[index]);
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_z_min.set_text(&coordinates[index]);
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_z_max.set_text(&coordinates[index]);
                index += 1;
                if index >= size {
                    return;
                }
            }
        }
    }

    /// Java `setXMin(String)`.
    pub fn set_x_min(&mut self, input: &str) {
        self.ltf_x_min.set_text(input);
    }
    /// Java `setXMax(String)`.
    pub fn set_x_max(&mut self, input: &str) {
        self.ltf_x_max.set_text(input);
    }
    /// Java `setYMin(String)`.
    pub fn set_y_min(&mut self, input: &str) {
        self.ltf_y_min.set_text(input);
    }
    /// Java `setYMax(String)`.
    pub fn set_y_max(&mut self, input: &str) {
        self.ltf_y_max.set_text(input);
    }
    /// Java `setZMin(String)`.
    pub fn set_z_min(&mut self, input: &str) {
        self.ltf_z_min.set_text(input);
    }
    /// Java `setZMax(String)`.
    pub fn set_z_max(&mut self, input: &str) {
        self.ltf_z_max.set_text(input);
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&mut self) {
        self.ltf_x_min.set_tool_tip_text(Some(
            "The X coordinate on the left side to retain in the volume.",
        ));
        self.ltf_x_max.set_tool_tip_text(Some(
            "The X coordinate on the right side to retain in the volume.",
        ));
        self.ltf_y_min
            .set_tool_tip_text(Some("The lower Y coordinate to retain in the volume."));
        self.ltf_y_max
            .set_tool_tip_text(Some("The upper Y coordinate to retain in the volume."));
        self.ltf_z_min
            .set_tool_tip_text(Some("The bottom Z slice to retain in the volume."));
        self.ltf_z_max
            .set_tool_tip_text(Some("The top Z slice to retain in the volume."));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Default)]
    struct Values {
        x_min: String,
        x_max: String,
        y_min: String,
        y_max: String,
        z_min: String,
        z_max: String,
        new_style_z: Option<(String, String)>,
    }
    impl VolumeRangeTrimvolParam for Values {
        fn get_x_min(&self) -> String {
            self.x_min.clone()
        }
        fn get_x_max(&self) -> String {
            self.x_max.clone()
        }
        fn get_y_min(&self) -> String {
            self.y_min.clone()
        }
        fn get_y_max(&self) -> String {
            self.y_max.clone()
        }
        fn get_z_min(&self) -> String {
            self.z_min.clone()
        }
        fn get_z_max(&self) -> String {
            self.z_max.clone()
        }
        fn set_x_min(&mut self, value: String) {
            self.x_min = value;
        }
        fn set_x_max(&mut self, value: String) {
            self.x_max = value;
        }
        fn set_y_min(&mut self, value: String) {
            self.y_min = value;
        }
        fn set_y_max(&mut self, value: String) {
            self.y_max = value;
        }
        fn set_z_min(&mut self, value: String) {
            self.z_min = value;
        }
        fn set_z_max(&mut self, value: String) {
            self.z_max = value;
        }
    }
    impl VolumeRangeConstMetaData for Values {
        fn get_post_trimvol_x_min(&self) -> String {
            self.x_min.clone()
        }
        fn get_post_trimvol_x_max(&self) -> String {
            self.x_max.clone()
        }
        fn get_post_trimvol_y_min(&self) -> String {
            self.y_min.clone()
        }
        fn get_post_trimvol_y_max(&self) -> String {
            self.y_max.clone()
        }
        fn get_post_trimvol_z_min(&self) -> String {
            self.z_min.clone()
        }
        fn get_post_trimvol_z_max(&self) -> String {
            self.z_max.clone()
        }
    }
    impl VolumeRangeMetaData for Values {
        fn set_post_trimvol_x_min(&mut self, value: String) {
            self.x_min = value;
        }
        fn set_post_trimvol_x_max(&mut self, value: String) {
            self.x_max = value;
        }
        fn set_post_trimvol_y_min(&mut self, value: String) {
            self.y_min = value;
        }
        fn set_post_trimvol_y_max(&mut self, value: String) {
            self.y_max = value;
        }
        fn set_post_trimvol_z_min(&mut self, value: String) {
            self.z_min = value;
        }
        fn set_post_trimvol_z_max(&mut self, value: String) {
            self.z_max = value;
        }
        fn set_post_trimvol_new_style_z(&mut self, min: String, max: String) {
            self.new_style_z = Some((min, max));
        }
    }

    #[test]
    fn construction_keeps_source_grid_border_and_tooltips() {
        let panel = VolumeRangePanel::get_instance(false);
        assert_eq!(
            panel.get_component().border.as_ref().unwrap().title,
            "Volume Range"
        );
        assert_eq!(panel.layout.as_ref().unwrap().child_order[0], "ltfXMin");
        assert_eq!(
            panel.ltf_z_max.tooltip.as_deref(),
            Some("The top Z slice to retain in the volume.")
        );
    }

    #[test]
    fn parameter_transfer_and_new_style_z_follow_source_order() {
        let input = Values {
            x_min: "1".into(),
            x_max: "2".into(),
            y_min: "3".into(),
            y_max: "4".into(),
            z_min: "5".into(),
            z_max: "6".into(),
            ..Values::default()
        };
        let mut panel = VolumeRangePanel::get_instance(false);
        panel.init_parameters(&input);
        let mut output = Values::default();
        assert!(panel.get_parameters_trimvol(&mut output, true));
        panel.get_parameters_for_trimvol(&mut output);
        assert_eq!((output.x_min, output.z_max), ("1".into(), "6".into()));
        assert_eq!(output.new_style_z, Some(("5".into(), "6".into())));
    }

    #[test]
    fn rubberband_parser_stops_at_each_missing_source_value() {
        let mut panel = VolumeRangePanel::get_instance(false);
        panel.set_xy_min_and_max(Some(&[
            RUBBERBAND_RESULTS_STRING.into(),
            "10".into(),
            "20".into(),
            "30".into(),
            "40".into(),
            "50".into(),
            "60".into(),
        ]));
        assert_eq!(panel.ltf_x_min.get_text(), "10");
        assert_eq!(panel.ltf_y_min.get_text(), "20");
        assert_eq!(panel.ltf_x_max.get_text(), "30");
        assert_eq!(panel.ltf_y_max.get_text(), "40");
        assert_eq!(panel.ltf_z_min.get_text(), "50");
        assert_eq!(panel.ltf_z_max.get_text(), "60");
    }

    #[test]
    fn locked_panel_does_not_transfer_values() {
        let mut panel = VolumeRangePanel::get_instance(true);
        panel.init_parameters(&Values {
            x_min: "1".into(),
            ..Values::default()
        });
        let mut output = Values::default();
        assert!(panel.get_parameters_trimvol(&mut output, true));
        assert!(panel.ltf_x_min.get_text().is_empty());
        assert!(output.x_min.is_empty());
    }
}
