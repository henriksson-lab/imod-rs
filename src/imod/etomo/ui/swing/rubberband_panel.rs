//! `IMOD/Etomo/src/etomo/ui/swing/RubberbandPanel.java`.
//!
//! Panel to get X and Y (and optionally Z) ranges from a 3dmod rubberband.

use crate::imod::etomo::ui::field::Field;
use std::rc::{Rc, Weak};

use super::etched_border::EtchedBorder;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::rubberband_container::RubberbandContainer;
use super::run_3dmod_button::Run3dmodButton;
use super::spaced_panel::SpacedPanel;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::trimvol_param::TrimvolParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::RUBBERBAND_RESULTS_STRING;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::meta_data::MetaData;
// TODO(unit): needs etomo/type/ParallelMetaData.java - `setXMin`..`setZMax`,
// `getXMin`..`getZMax`, `setNewStyleZ` in the three ParallelMetaData methods below.
use crate::imod::etomo::r#type::parallel_meta_data::ParallelMetaData;
use crate::imod::etomo::ui::field_type::FieldType;

/// Java `RubberbandPanel.rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `final class RubberbandPanel`.
pub struct RubberbandPanel {
    manager: &'static dyn BaseManager,
    pnl_rubberband: Rc<SpacedPanel>,
    /// Java `private final JPanel pnlRange = new JPanel()`.
    pnl_range: Rc<JComponent>,
    ltf_x_min: Rc<LabeledTextField>,
    ltf_x_max: Rc<LabeledTextField>,
    ltf_y_min: Rc<LabeledTextField>,
    ltf_y_max: Rc<LabeledTextField>,
    ltf_z_min: Rc<LabeledTextField>,
    ltf_z_max: Rc<LabeledTextField>,
    btn_rubberband: Rc<MultiLineButton>,
    imod_key: Option<String>,
    x_min_tooltip: Option<String>,
    x_max_tooltip: Option<String>,
    y_min_tooltip: Option<String>,
    y_max_tooltip: Option<String>,
    z_min_tooltip: Option<String>,
    z_max_tooltip: Option<String>,
    btn_imod: Option<Rc<Run3dmodButton>>,
    /// Java `private final RubberbandContainer container; // optional`.  The
    /// container owns this panel, so the back reference is weak.
    container: Option<Weak<dyn RubberbandContainer>>,
    place_buttons: bool,
    lock_panel: bool,
}

impl RubberbandPanel {
    /// Java private `RubberbandPanel(BaseManager, RubberbandContainer, String,
    /// String, String, String, String, String, String, String, String,
    /// Run3dmodButton, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        manager: &'static dyn BaseManager,
        container: Option<Weak<dyn RubberbandContainer>>,
        imod_key: Option<&str>,
        border_label: Option<&str>,
        button_label: Option<&str>,
        x_min_tooltip: Option<&str>,
        x_max_tooltip: Option<&str>,
        y_min_tooltip: Option<&str>,
        y_max_tooltip: Option<&str>,
        z_min_tooltip: Option<&str>,
        z_max_tooltip: Option<&str>,
        btn_imod: Option<Rc<Run3dmodButton>>,
        place_buttons: bool,
        lock_panel: bool,
    ) -> RubberbandPanel {
        // Field initializers.
        let pnl_rubberband = SpacedPanel::get_instance_void();
        let pnl_range = JComponent::new_panel();
        let ltf_x_min =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("X min: "));
        let ltf_x_max =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("X max: "));
        let ltf_y_min =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y min: "));
        let ltf_y_max =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("Y max: "));
        let ltf_z_min =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("Z min: "));
        let ltf_z_max =
            LabeledTextField::new_field_type_string(FieldType::Integer, Some("Z max: "));
        pnl_rubberband.set_border(&EtchedBorder::new(border_label).get_border());
        let btn_rubberband = MultiLineButton::new_string(button_label);
        // Swing layout: pnlRubberband.setBoxLayout(BoxLayout.Y_AXIS).
        let mut pnl_buttons: Option<Rc<SpacedPanel>> = None;
        if let Some(btn_imod) = &btn_imod {
            if place_buttons {
                let buttons = SpacedPanel::get_instance_void();
                // Swing layout: pnlButtons.setBoxLayout(BoxLayout.X_AXIS);
                // pnlButtons.addHorizontalGlue().
                buttons.add_multi_line_button(btn_imod);
                // Swing layout: pnlButtons.addHorizontalGlue().
                buttons.add_multi_line_button(&btn_rubberband);
                // Swing layout: pnlButtons.addHorizontalGlue().
                pnl_buttons = Some(buttons);
            }
            // Swing layout: pnlRange.setLayout(new GridLayout(3, 2, 5, 5)).
        } else {
            // Swing layout: pnlRange.setLayout(new GridLayout(2, 2, 5, 5)).
        }
        pnl_range.add(&ltf_x_min.get_container());
        pnl_range.add(&ltf_x_max.get_container());
        pnl_range.add(&ltf_y_min.get_container());
        pnl_range.add(&ltf_y_max.get_container());
        if btn_imod.is_some() {
            pnl_range.add(&ltf_z_min.get_container());
            pnl_range.add(&ltf_z_max.get_container());
            if place_buttons {
                if let Some(pnl_buttons) = &pnl_buttons {
                    pnl_rubberband.add_container(&pnl_buttons.get_container());
                }
            }
        }
        pnl_rubberband.add_j_panel(&pnl_range);
        if btn_imod.is_none() {
            // Swing layout: btnRubberband.setAlignmentX(Component.CENTER_ALIGNMENT).
            if place_buttons {
                // Swing layout: pnlRubberband.addRigidArea().
                pnl_rubberband.add_multi_line_button(&btn_rubberband);
            }
        }
        let panel = RubberbandPanel {
            manager,
            pnl_rubberband,
            pnl_range,
            ltf_x_min,
            ltf_x_max,
            ltf_y_min,
            ltf_y_max,
            ltf_z_min,
            ltf_z_max,
            btn_rubberband,
            imod_key: imod_key.map(str::to_owned),
            x_min_tooltip: x_min_tooltip.map(str::to_owned),
            x_max_tooltip: x_max_tooltip.map(str::to_owned),
            y_min_tooltip: y_min_tooltip.map(str::to_owned),
            y_max_tooltip: y_max_tooltip.map(str::to_owned),
            z_min_tooltip: z_min_tooltip.map(str::to_owned),
            z_max_tooltip: z_max_tooltip.map(str::to_owned),
            btn_imod,
            container,
            place_buttons,
            lock_panel,
        };
        panel.set_tool_tip_text();
        panel
    }

    /// Java package-private static `getInstance(BaseManager, RubberbandContainer,
    /// String, String, String, String, String, String, String)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance_base_manager_rubberband_container_string_string_string_string_string_string_string(
        manager: &'static dyn BaseManager,
        container: Option<Weak<dyn RubberbandContainer>>,
        imod_key: Option<&str>,
        border_label: Option<&str>,
        button_label: Option<&str>,
        x_min_tooltip: Option<&str>,
        x_max_tooltip: Option<&str>,
        y_min_tooltip: Option<&str>,
        y_max_tooltip: Option<&str>,
    ) -> Rc<RubberbandPanel> {
        let instance = Rc::new(RubberbandPanel::new(
            manager,
            container,
            imod_key,
            border_label,
            button_label,
            x_min_tooltip,
            x_max_tooltip,
            y_min_tooltip,
            y_max_tooltip,
            Some(""),
            Some(""),
            None,
            true,
            false,
        ));
        instance.add_listeners();
        instance
    }

    /// Java package-private static `getInstance(BaseManager, String, String,
    /// String, String, String, String, String, String, String, Run3dmodButton)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance_base_manager_string_string_string_string_string_string_string_string_string_run3dmod_button(
        manager: &'static dyn BaseManager,
        imod_key: Option<&str>,
        border_label: Option<&str>,
        button_label: Option<&str>,
        x_min_tooltip: Option<&str>,
        x_max_tooltip: Option<&str>,
        y_min_tooltip: Option<&str>,
        y_max_tooltip: Option<&str>,
        z_min_tooltip: Option<&str>,
        z_max_tooltip: Option<&str>,
        btn_idmod: Option<Rc<Run3dmodButton>>,
    ) -> Rc<RubberbandPanel> {
        let instance = Rc::new(RubberbandPanel::new(
            manager,
            None,
            imod_key,
            border_label,
            button_label,
            x_min_tooltip,
            x_max_tooltip,
            y_min_tooltip,
            y_max_tooltip,
            z_min_tooltip,
            z_max_tooltip,
            btn_idmod,
            true,
            false,
        ));
        instance.add_listeners();
        instance
    }

    /// Java package-private static `getNoButtonInstance(...)`.
    ///
    /// Does not use the 3dmod button and does not add the rubberband button to
    /// the display.  The rubberband button must be placed in the container to be
    /// available.
    #[allow(clippy::too_many_arguments)]
    pub fn get_no_button_instance(
        manager: &'static dyn BaseManager,
        container: Option<Weak<dyn RubberbandContainer>>,
        imod_key: Option<&str>,
        border_label: Option<&str>,
        button_label: Option<&str>,
        x_min_tooltip: Option<&str>,
        x_max_tooltip: Option<&str>,
        y_min_tooltip: Option<&str>,
        y_max_tooltip: Option<&str>,
        lock_panel: bool,
    ) -> Rc<RubberbandPanel> {
        let instance = Rc::new(RubberbandPanel::new(
            manager,
            container,
            imod_key,
            border_label,
            button_label,
            x_min_tooltip,
            x_max_tooltip,
            y_min_tooltip,
            y_max_tooltip,
            Some(""),
            Some(""),
            None,
            false,
            lock_panel,
        ));
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(self: &Rc<Self>) {
        let listener = RubberbandActionListener::new(self);
        let action_listener: ActionListener =
            Rc::new(move |event: &ActionEvent| listener.action_performed(event));
        self.btn_rubberband.add_action_listener(action_listener);
    }

    /// Java package-private `getRubberbandButtonComponent()`.
    pub fn get_rubberband_button_component(&self) -> Rc<JComponent> {
        self.btn_rubberband.get_component()
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_rubberband.get_container()
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_rubberband.get_container()
    }

    /// Java package-private `buttonAction(ActionEvent)`.
    pub fn button_action(&self, event: &ActionEvent) {
        let command = event.get_action_command();
        // Java compares the two command strings by reference; the listener is
        // registered only on btnRubberband, whose command is the same object.
        if command.map(str::to_owned) == self.btn_rubberband.get_action_command() {
            let coordinates = self
                .manager
                .imod_get_rubberband_coordinates(self.imod_key.as_deref(), Some(AxisID::Only));
            self.set_min_and_max(coordinates.as_deref());
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

    /// Java private `setMinAndMax(Vector)`.
    fn set_min_and_max(&self, coordinates: Option<&[String]>) {
        let Some(coordinates) = coordinates else {
            return;
        };
        let size = coordinates.len();
        if size == 0 {
            return;
        }
        let mut index = 0;
        while index < size {
            let entry = &coordinates[index];
            index += 1;
            if RUBBERBAND_RESULTS_STRING == entry {
                // Upstream bug fixed (RubberbandPanel.java:227): Java reads the
                // element after "Rubberband:" without the bounds check it makes
                // before every later read, so a trailing "Rubberband:" throws
                // ArrayIndexOutOfBoundsException on the EDT.  We return instead,
                // as the later checks do.
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
                if self.btn_imod.is_none() && self.container.is_none() {
                    return;
                }
                if self.btn_imod.is_some() {
                    self.ltf_z_min.set_text_string(Some(&coordinates[index]));
                }
                if let Some(container) = self.container.as_ref().and_then(Weak::upgrade) {
                    container.set_rubberband_container_z_min(Some(&coordinates[index]));
                }
                index += 1;
                if index >= size {
                    return;
                }
                if self.btn_imod.is_some() {
                    self.ltf_z_max.set_text_string(Some(&coordinates[index]));
                }
                if let Some(container) = self.container.as_ref().and_then(Weak::upgrade) {
                    container.set_rubberband_container_z_max(Some(&coordinates[index]));
                }
                return;
            }
        }
    }

    /// Java package-private `setEnabled(boolean)`.
    pub fn set_enabled(&self, enable: bool) {
        self.ltf_x_min.set_enabled(enable);
        self.ltf_x_max.set_enabled(enable);
        self.ltf_y_min.set_enabled(enable);
        self.ltf_y_max.set_enabled(enable);
        self.btn_rubberband.set_enabled(enable);
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_rubberband.set_visible(visible);
    }

    /// Java package-private `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        if self.lock_panel {
            return;
        }
        meta_data.set_post_trimvol_scale_x_min(self.ltf_x_min.get_text_void().as_deref());
        meta_data.set_post_trimvol_scale_x_max(self.ltf_x_max.get_text_void().as_deref());
        meta_data.set_post_trimvol_scale_y_min(self.ltf_y_min.get_text_void().as_deref());
        meta_data.set_post_trimvol_scale_y_max(self.ltf_y_max.get_text_void().as_deref());
        if self.btn_imod.is_some() {
            meta_data.set_post_trimvol_section_scale_min(self.ltf_z_min.get_text_void().as_deref());
            meta_data.set_post_trimvol_section_scale_max(self.ltf_z_max.get_text_void().as_deref());
        }
    }

    /// Java package-private `getParameters(TrimvolParam, boolean)`.  A
    /// `FieldValidationFailedException` returns false.
    pub fn get_parameters_trimvol_param_boolean(
        &self,
        trimvol_param: &mut TrimvolParam,
        do_validation: bool,
    ) -> bool {
        if self.lock_panel {
            return true;
        }
        let Ok(text) = self.ltf_x_min.get_text_boolean(do_validation) else {
            return false;
        };
        trimvol_param.set_x_min(text.as_deref());
        let Ok(text) = self.ltf_x_max.get_text_boolean(do_validation) else {
            return false;
        };
        trimvol_param.set_x_max(text.as_deref());
        let Ok(text) = self.ltf_y_min.get_text_boolean(do_validation) else {
            return false;
        };
        trimvol_param.set_y_min(text.as_deref());
        let Ok(text) = self.ltf_y_max.get_text_boolean(do_validation) else {
            return false;
        };
        trimvol_param.set_y_max(text.as_deref());
        if self.btn_imod.is_some() {
            let Ok(text) = self.ltf_z_min.get_text_boolean(do_validation) else {
                return false;
            };
            trimvol_param.set_z_min(text.as_deref());
            let Ok(text) = self.ltf_z_max.get_text_boolean(do_validation) else {
                return false;
            };
            trimvol_param.set_z_max(text.as_deref());
        }
        true
    }

    /// Java package-private `getScaleParameters(TrimvolParam, boolean)`.  A
    /// `FieldValidationFailedException` returns false.
    pub fn get_scale_parameters(
        &self,
        trimvol_param: &mut TrimvolParam,
        do_validation: bool,
    ) -> bool {
        if self.lock_panel {
            return true;
        }
        {
            let xy_param = trimvol_param.get_scale_xy_param();
            let Ok(text) = self.ltf_x_min.get_text_boolean(do_validation) else {
                return false;
            };
            xy_param.set_x_min_string(text.as_deref());
            let Ok(text) = self.ltf_x_max.get_text_boolean(do_validation) else {
                return false;
            };
            xy_param.set_x_max_string(text.as_deref());
            let Ok(text) = self.ltf_y_min.get_text_boolean(do_validation) else {
                return false;
            };
            xy_param.set_y_min_string(text.as_deref());
            let Ok(text) = self.ltf_y_max.get_text_boolean(do_validation) else {
                return false;
            };
            xy_param.set_y_max_string(text.as_deref());
        }
        if self.btn_imod.is_some() {
            let Ok(text) = self.ltf_z_min.get_text_boolean(do_validation) else {
                return false;
            };
            trimvol_param.set_section_scale_min(text.as_deref());
            let Ok(text) = self.ltf_z_max.get_text_boolean(do_validation) else {
                return false;
            };
            trimvol_param.set_section_scale_max(text.as_deref());
        }
        true
    }

    /// Java package-private `getParameters(ParallelMetaData)`.
    pub fn get_parameters_parallel_meta_data(&self, meta_data: &ParallelMetaData) {
        if self.lock_panel {
            return;
        }
        meta_data.set_x_min(self.ltf_x_min.get_text_void().as_deref());
        meta_data.set_x_max(self.ltf_x_max.get_text_void().as_deref());
        meta_data.set_y_min(self.ltf_y_min.get_text_void().as_deref());
        meta_data.set_y_max(self.ltf_y_max.get_text_void().as_deref());
        if self.btn_imod.is_some() {
            meta_data.set_z_min(self.ltf_z_min.get_text_void().as_deref());
            meta_data.set_z_max(self.ltf_z_max.get_text_void().as_deref());
        }
    }

    /// Java package-private `getParametersForTrimvol(ParallelMetaData)`.
    pub fn get_parameters_for_trimvol(&self, meta_data: &ParallelMetaData) {
        if self.lock_panel {
            return;
        }
        if self.btn_imod.is_some() {
            meta_data.set_new_style_z(
                self.ltf_z_min.get_text_void().as_deref(),
                self.ltf_z_max.get_text_void().as_deref(),
            );
        }
    }

    /// Java package-private `setParameters(TrimvolParam)`.
    pub fn set_parameters_trimvol_param(&self, param: &TrimvolParam) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min.set_text_int(param.get_x_min());
        self.ltf_x_max.set_text_int(param.get_x_max());
        self.ltf_y_min.set_text_int(param.get_y_min());
        self.ltf_y_max.set_text_int(param.get_y_max());
        if self.btn_imod.is_some() {
            self.ltf_z_min.set_text_int(param.get_z_min());
            self.ltf_z_max.set_text_int(param.get_z_max());
        }
    }

    /// Java package-private `initScaleParameters(TrimvolParam)`.
    pub fn init_scale_parameters(&self, trimvol_param: &mut TrimvolParam) {
        if self.lock_panel {
            return;
        }
        {
            let xy_param = trimvol_param.get_scale_xy_param();
            self.ltf_x_min
                .set_text_const_etomo_number(Some(xy_param.get_x_min()));
            self.ltf_x_max
                .set_text_const_etomo_number(Some(xy_param.get_x_max()));
            self.ltf_y_min
                .set_text_const_etomo_number(Some(xy_param.get_y_min()));
            self.ltf_y_max
                .set_text_const_etomo_number(Some(xy_param.get_y_max()));
        }
        if self.btn_imod.is_some() {
            self.ltf_z_min
                .set_text_const_etomo_number(Some(trimvol_param.get_section_scale_min()));
            self.ltf_z_max
                .set_text_const_etomo_number(Some(trimvol_param.get_section_scale_max()));
        }
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min
            .set_text_string(Some(&meta_data.get_post_trimvol_scale_x_min()));
        self.ltf_x_max
            .set_text_string(Some(&meta_data.get_post_trimvol_scale_x_max()));
        self.ltf_y_min
            .set_text_string(Some(&meta_data.get_post_trimvol_scale_y_min()));
        self.ltf_y_max
            .set_text_string(Some(&meta_data.get_post_trimvol_scale_y_max()));
        if self.btn_imod.is_some() {
            self.ltf_z_min
                .set_text_string(Some(&meta_data.get_post_trimvol_section_scale_min()));
            self.ltf_z_max
                .set_text_string(Some(&meta_data.get_post_trimvol_section_scale_max()));
        }
    }

    /// Java package-private `setParameters(ParallelMetaData)`.
    pub fn set_parameters_parallel_meta_data(&self, meta_data: &ParallelMetaData) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min
            .set_text_string(meta_data.get_x_min().as_deref());
        self.ltf_x_max
            .set_text_string(meta_data.get_x_max().as_deref());
        self.ltf_y_min
            .set_text_string(meta_data.get_y_min().as_deref());
        self.ltf_y_max
            .set_text_string(meta_data.get_y_max().as_deref());
        if self.btn_imod.is_some() {
            self.ltf_z_min
                .set_text_string(meta_data.get_z_min().as_deref());
            self.ltf_z_max
                .set_text_string(meta_data.get_z_max().as_deref());
        }
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        self.ltf_x_min
            .set_tool_tip_text(self.x_min_tooltip.as_deref());
        self.ltf_x_max
            .set_tool_tip_text(self.x_max_tooltip.as_deref());
        self.ltf_y_min
            .set_tool_tip_text(self.y_min_tooltip.as_deref());
        self.ltf_y_max
            .set_tool_tip_text(self.y_max_tooltip.as_deref());
        self.ltf_z_min
            .set_tool_tip_text(self.z_min_tooltip.as_deref());
        self.ltf_z_max
            .set_tool_tip_text(self.z_max_tooltip.as_deref());
        self.btn_rubberband.set_tool_tip_text(Some(&format!(
            "After opening the volume in 3dmod, press shift-B in \
             the ZaP window.  Create a rubberband around the contrast \
             range.  Then press this button to retrieve the X{} coordinates.",
            if self.btn_imod.is_none() {
                " and Y"
            } else {
                ", Y, and Z"
            }
        )));
    }
}

/// Java `private static final class RubberbandActionListener implements
/// ActionListener`.
struct RubberbandActionListener {
    /// Java `RubberbandPanel adaptee`; the panel owns the button this listener
    /// is on, so the reference is weak.
    adaptee: Weak<RubberbandPanel>,
}

impl RubberbandActionListener {
    /// Java `RubberbandActionListener(RubberbandPanel)`.
    fn new(panel: &Rc<RubberbandPanel>) -> RubberbandActionListener {
        RubberbandActionListener {
            adaptee: Rc::downgrade(panel),
        }
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, event: &ActionEvent) {
        if let Some(adaptee) = self.adaptee.upgrade() {
            adaptee.button_action(event);
        }
    }
}
