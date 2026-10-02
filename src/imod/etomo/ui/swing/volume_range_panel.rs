//! `IMOD/Etomo/src/etomo/ui/swing/VolumeRangePanel.java`.
//!
//! Java `final class VolumeRangePanel`: the X/Y/Z min and max fields of the
//! trimvol Volume Range box (factored out of `TrimvolPanel`).
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`VolumeRangePanel::get_instance`]; every method takes `&self`.

use std::rc::Rc;

use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::labeled_text_field::LabeledTextField;
use crate::imod::etomo::comscript::trimvol_param::TrimvolParam;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::process::imod_process;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;

/// Java public static final `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `final class VolumeRangePanel`.
pub struct VolumeRangePanel {
    /// Java private final `pnlRoot = new EtomoPanel()`.
    pnl_root: Rc<EtomoPanel>,
    /// Java private final `ltfXMin`.
    ltf_x_min: Rc<LabeledTextField>,
    /// Java private final `ltfXMax`.
    ltf_x_max: Rc<LabeledTextField>,
    /// Java private final `ltfYMin`.
    ltf_y_min: Rc<LabeledTextField>,
    /// Java private final `ltfYMax`.
    ltf_y_max: Rc<LabeledTextField>,
    /// Java private final `ltfZMin`.
    ltf_z_min: Rc<LabeledTextField>,
    /// Java private final `ltfZMax`.
    ltf_z_max: Rc<LabeledTextField>,
    /// Java private final `lockPanel`.  When lockPanel is true, do not load
    /// default, metaData, or comscript values, and do not save to metaData or
    /// comscripts.  LockPanel is set when the data to create this panel
    /// correctly is not available.  Saving data at this point may prevent the
    /// panel from being created correctly when the data is available.
    lock_panel: bool,
}

impl VolumeRangePanel {
    /// Java private constructor `VolumeRangePanel(boolean)`, with the field
    /// initializers.
    fn new(lock_panel: bool) -> Rc<VolumeRangePanel> {
        Rc::new(VolumeRangePanel {
            pnl_root: EtomoPanel::new(),
            ltf_x_min: LabeledTextField::new_field_type_string(FieldType::Integer, Some("X min: ")),
            ltf_x_max: LabeledTextField::new_field_type_string(FieldType::Integer, Some("X max: ")),
            ltf_y_min: LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y min: ")),
            ltf_y_max: LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y max: ")),
            ltf_z_min: LabeledTextField::new_field_type_string(FieldType::Integer, Some("Z min: ")),
            ltf_z_max: LabeledTextField::new_field_type_string(FieldType::Integer, Some("Z max: ")),
            lock_panel,
        })
    }

    /// Java package-private static `getInstance(boolean)`.
    pub fn get_instance(lock_panel: bool) -> Rc<VolumeRangePanel> {
        let instance = VolumeRangePanel::new(lock_panel);
        instance.create_panel();
        instance.set_tooltips();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // Root panel
        // Swing layout: pnlRoot.setLayout(new GridLayout(3, 2, 5, 5)).
        self.pnl_root
            .set_border(&EtchedBorder::new(Some("Volume Range")).get_border());
        let root = self.pnl_root.get_component();
        root.add(&self.ltf_x_min.get_container());
        root.add(&self.ltf_x_max.get_container());
        root.add(&self.ltf_y_min.get_container());
        root.add(&self.ltf_y_max.get_container());
        root.add(&self.ltf_z_min.get_container());
        root.add(&self.ltf_z_max.get_container());
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }

    /// Java package-private `initParameters(TrimvolParam)`.  Set the panel
    /// values with the specified parameters.
    pub fn init_parameters(&self, param: &TrimvolParam) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min.set_text_int(param.get_x_min());
        self.ltf_x_max.set_text_int(param.get_x_max());
        self.ltf_y_min.set_text_int(param.get_y_min());
        self.ltf_y_max.set_text_int(param.get_y_max());
        self.ltf_z_min.set_text_int(param.get_z_min());
        self.ltf_z_max.set_text_int(param.get_z_max());
    }

    /// Java package-private `setParameters(ConstMetaData)`.  Set the panel
    /// values with the specified parameters.
    pub fn set_parameters(&self, meta_data: &dyn ConstMetaData) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min
            .set_text_string(Some(&meta_data.get_post_trimvol_x_min()));
        self.ltf_x_max
            .set_text_string(Some(&meta_data.get_post_trimvol_x_max()));
        self.ltf_y_min
            .set_text_string(Some(&meta_data.get_post_trimvol_y_min()));
        self.ltf_y_max
            .set_text_string(Some(&meta_data.get_post_trimvol_y_max()));
        self.ltf_z_min
            .set_text_string(Some(&meta_data.get_post_trimvol_z_min()));
        self.ltf_z_max
            .set_text_string(Some(&meta_data.get_post_trimvol_z_max()));
    }

    /// Java package-private `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        if self.lock_panel {
            return;
        }
        meta_data.set_post_trimvol_x_min(self.ltf_x_min.get_text_void().as_deref());
        meta_data.set_post_trimvol_x_max(self.ltf_x_max.get_text_void().as_deref());
        meta_data.set_post_trimvol_y_min(self.ltf_y_min.get_text_void().as_deref());
        meta_data.set_post_trimvol_y_max(self.ltf_y_max.get_text_void().as_deref());
        meta_data.set_post_trimvol_z_min(self.ltf_z_min.get_text_void().as_deref());
        meta_data.set_post_trimvol_z_max(self.ltf_z_max.get_text_void().as_deref());
    }

    /// Java package-private `getParametersForTrimvol(MetaData)`.
    pub fn get_parameters_for_trimvol(&self, meta_data: &MetaData) {
        if self.lock_panel {
            return;
        }
        meta_data.set_post_trimvol_new_style_z(
            self.ltf_z_min.get_text_void().as_deref(),
            self.ltf_z_max.get_text_void().as_deref(),
        );
    }

    /// Java package-private `getParameters(TrimvolParam, boolean)`.  Get the
    /// parameter values from the panel.
    pub fn get_parameters_trimvol_param_boolean(
        &self,
        trimvol_param: &mut TrimvolParam,
        do_validation: bool,
    ) -> bool {
        if self.lock_panel {
            return true;
        }
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| {
            trimvol_param.set_x_min(self.ltf_x_min.get_text_boolean(do_validation)?.as_deref());
            trimvol_param.set_x_max(self.ltf_x_max.get_text_boolean(do_validation)?.as_deref());
            trimvol_param.set_y_min(self.ltf_y_min.get_text_boolean(do_validation)?.as_deref());
            trimvol_param.set_y_max(self.ltf_y_max.get_text_boolean(do_validation)?.as_deref());
            trimvol_param.set_z_min(self.ltf_z_min.get_text_boolean(do_validation)?.as_deref());
            trimvol_param.set_z_max(self.ltf_z_max.get_text_boolean(do_validation)?.as_deref());
            Ok::<(), crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException>(())
        })();
        result.is_ok()
    }

    /// Java package-private `setXYMinAndMax(Vector)`.  `None` is Java null.
    pub fn set_xy_min_and_max(&self, coordinates: Option<&[String]>) {
        let Some(coordinates) = coordinates else {
            return;
        };
        let size = coordinates.len();
        if size == 0 {
            return;
        }
        let mut index = 0;
        while index < size {
            let element = &coordinates[index];
            index += 1;
            if imod_process::RUBBERBAND_RESULTS_STRING == element {
                // Upstream bug fixed in translation (VolumeRangePanel.java:176):
                // when the results marker is the last element Java calls
                // `coordinates.get(size)` and throws
                // ArrayIndexOutOfBoundsException.  The translation returns, as the
                // source does after every later element.
                if index >= size {
                    return;
                }
                self.ltf_x_min.set_text_string(Some(&coordinates[index]));
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_y_min.set_text_string(Some(&coordinates[index]));
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_x_max.set_text_string(Some(&coordinates[index]));
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_y_max.set_text_string(Some(&coordinates[index]));
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_z_min.set_text_string(Some(&coordinates[index]));
                index += 1;
                if index >= size {
                    return;
                }
                self.ltf_z_max.set_text_string(Some(&coordinates[index]));
                index += 1;
                if index >= size {
                    return;
                }
            }
        }
    }

    /// Java package-private `setXMin(String)`.
    pub fn set_x_min(&self, input: Option<&str>) {
        self.ltf_x_min.set_text_string(input);
    }

    /// Java package-private `setXMax(String)`.
    pub fn set_x_max(&self, input: Option<&str>) {
        self.ltf_x_max.set_text_string(input);
    }

    /// Java package-private `setYMin(String)`.
    pub fn set_y_min(&self, input: Option<&str>) {
        self.ltf_y_min.set_text_string(input);
    }

    /// Java package-private `setYMax(String)`.
    pub fn set_y_max(&self, input: Option<&str>) {
        self.ltf_y_max.set_text_string(input);
    }

    /// Java package-private `setZMin(String)`.
    pub fn set_z_min(&self, input: Option<&str>) {
        self.ltf_z_min.set_text_string(input);
    }

    /// Java package-private `setZMax(String)`.
    pub fn set_z_max(&self, input: Option<&str>) {
        self.ltf_z_max.set_text_string(input);
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
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
