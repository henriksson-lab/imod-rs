//! `IMOD/Etomo/src/etomo/ui/swing/PrenewstPanel.java`.
//!
//! The coarse aligned stack (prenewst / preblend) panel of the coarse
//! alignment dialog.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::blendmont_display::{BlendmontDisplay, BlendmontDisplayException};
use super::check_box::CheckBox;
use super::coarse_align_dialog::CoarseAlignDialog;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::fiducialess_params::FiducialessParams;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_spinner::LabeledSpinner;
use super::newstack_display::{NewstackDisplay, NewstackDisplayException};
use super::panel_header::PanelHeader;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::newst_param::{self, Mode, NewstParam};
use crate::imod::etomo::jdk::MouseEvent;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ChangeEvent, ChangeListener, JComponent,
};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::const_etomo_number::Number;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::view_type::ViewType;

/// Java `final class PrenewstPanel implements ContextMenu, Expandable,
/// Run3dmodButtonContainer, NewstackDisplay, BlendmontDisplay`.
pub struct PrenewstPanel {
    pnl_prenewst: Rc<EtomoPanel>,
    pnl_body: Rc<JComponent>,
    pnl_check_boxes: Rc<JComponent>,
    cb_byte_mode_to_output: Rc<CheckBox>,
    cb_mean_float_densities: Rc<CheckBox>,
    btn_imod: Rc<Run3dmodButton>,

    application_manager: &'static ApplicationManager,
    spin_binning: Rc<LabeledSpinner>,
    axis_id: AxisID,
    btn_coarse_align: Rc<Run3dmodButton>,
    /// Java `actionListener` (a `PrenewstPanelActionListener`).
    action_listener: ActionListener,
    header: Rc<PanelHeader>,
    /// Java `parent`: the owning dialog, which holds this panel.
    parent: Weak<CoarseAlignDialog>,
    dialog_type: DialogType,
    /// Java `cbAntialiasFilter`; null for a montage.
    cb_antialias_filter: Option<Rc<CheckBox>>,
    /// Java `antialiasFilterValue`; null for a montage.
    antialias_filter_value: Option<RefCell<EtomoNumber>>,
}

impl PrenewstPanel {
    /// Java `PrenewstPanel(ApplicationManager, AxisID, DialogType,
    /// CoarseAlignDialog, GlobalExpandButton)` (PrenewstPanel.java:195-280).
    pub fn new(
        application_manager: &'static ApplicationManager,
        id: AxisID,
        dialog_type: DialogType,
        parent: Weak<CoarseAlignDialog>,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<PrenewstPanel> {
        let panel = Rc::new_cyclic(|weak: &Weak<PrenewstPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = weak.clone();
            // Field initializers (PrenewstPanel.java:176-182).
            let pnl_prenewst = EtomoPanel::new();
            let pnl_body = JComponent::new_panel();
            let pnl_check_boxes = JComponent::new_panel();
            let cb_byte_mode_to_output = CheckBox::new_string(Some("Convert to bytes"));
            let cb_mean_float_densities = CheckBox::new_string(Some("Float intensities to mean"));
            let btn_imod = Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                Some("View Aligned Stack In 3dmod"),
                Some(container.clone()),
            );

            // Constructor body.
            let btn_coarse_align = application_manager
                .get_process_result_display_factory(id)
                .get_coarse_align();
            let montage = application_manager.get_meta_data().get_view_type() == ViewType::Montage;
            let pnl_antialias_filter: Option<Rc<JComponent>>;
            let cb_antialias_filter;
            let antialias_filter_value;
            if !montage {
                pnl_antialias_filter = Some(JComponent::new_panel());
                cb_antialias_filter = Some(CheckBox::new_string(Some(
                    "Reduce size with antialiasing filter",
                )));
                antialias_filter_value = Some(RefCell::new(EtomoNumber::new()));
            } else {
                pnl_antialias_filter = None;
                cb_antialias_filter = None;
                antialias_filter_value = None;
            }

            btn_coarse_align.set_container(Some(container.clone()));
            // Swing layout: pnlPrenewst, pnlBody and pnlCheckBoxes BoxLayout Y_AXIS.

            // Construct the binning spinner
            let spin_binning = LabeledSpinner::get_instance_string_int_int_int_int(
                Some("Coarse aligned image stack binning "),
                1,
                1,
                8,
                1,
            );
            // Swing layout: spinBinning.setTextMaxmimumSize(spinner dimension).
            let pnl_binning = JComponent::new_panel();
            // Swing layout: pnlBinning BoxLayout X_AXIS, center aligned, glue.
            pnl_binning.add(&spin_binning.get_container());
            // `UIUtilities.addWithYSpace(p, c)` is `p.add(c)` plus a rigid area
            // (UIUtilities.java:481-484).
            pnl_body.add(&pnl_binning);
            if let Some(cb_antialias_filter) = &cb_antialias_filter {
                let pnl_antialias_filter = pnl_antialias_filter.as_ref().unwrap();
                // Swing layout: pnlAntialiasFilter BoxLayout X_AXIS, glue.
                pnl_antialias_filter.add(&cb_antialias_filter.get_component());
                pnl_body.add(pnl_antialias_filter);
            }
            let expandable: Weak<dyn Expandable> = weak.clone();
            let header;
            if montage {
                header = PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Blendmont"),
                    Some(expandable.clone()),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );
            } else {
                header = PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Newstack"),
                    Some(expandable.clone()),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );
                let pnl_byte_mode_to_output = JComponent::new_panel();
                // Swing layout: pnlByteModeToOutput BoxLayout X_AXIS, center
                // aligned, glue.
                pnl_byte_mode_to_output.add(&cb_byte_mode_to_output.get_component());
                pnl_check_boxes.add(&pnl_byte_mode_to_output);
                pnl_check_boxes.add(&cb_mean_float_densities.get_component());
            }
            pnl_body.add(&pnl_check_boxes);
            btn_coarse_align.set_deferred_3dmod_button_deferred_3dmod_button(Some(
                btn_imod.clone() as Rc<dyn Deferred3dmodButton>,
            ));
            let pnl_buttons = JComponent::new_panel();
            // Swing layout: pnlButtons BoxLayout X_AXIS with glue between buttons.
            pnl_buttons.add(&btn_coarse_align.get_component());
            pnl_buttons.add(&btn_imod.get_component());
            pnl_body.add(&pnl_buttons);

            // Align the UI objects along their left sides
            // Swing layout: center align pnlBody's components, left align
            // pnlCheckBoxes'; pnlPrenewst untitled etched border.
            pnl_prenewst.add(&header);
            pnl_prenewst.get_component().add(&pnl_body);

            // Mouse adapter for context menu
            // Java `actionListener = new PrenewstPanelActionListener(this)`; the
            // class is at PrenewstPanel.java:488-499.
            let adaptee = weak.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });

            PrenewstPanel {
                pnl_prenewst,
                pnl_body,
                pnl_check_boxes,
                cb_byte_mode_to_output,
                cb_mean_float_densities,
                btn_imod,
                application_manager,
                spin_binning,
                axis_id: id,
                btn_coarse_align,
                action_listener,
                header,
                parent,
                dialog_type,
                cb_antialias_filter,
                antialias_filter_value,
            }
        });
        panel
            .btn_coarse_align
            .add_action_listener(panel.action_listener.clone());
        panel
            .btn_imod
            .add_action_listener(panel.action_listener.clone());
        // Java `spinBinning.addChangeListener(new PrenewstBinningChangeListener(this))`;
        // the class is at PrenewstPanel.java:501-512.
        let weak_panel = Rc::downgrade(&panel);
        let change_listener: ChangeListener = Rc::new(move |_event: &ChangeEvent| {
            let Some(panel) = weak_panel.upgrade() else {
                return;
            };
            panel.update_enabled();
        });
        panel.spin_binning.add_change_listener(change_listener);
        // `pnlPrenewst.addMouseListener(new GenericMouseAdapter(this))`: mouse
        // events are not modelled.
        panel.set_tool_tip_text();
        panel
    }

    /// Java `updateAdvanced(boolean)` (PrenewstPanel.java:296-303).
    pub fn update_advanced(&self, state: bool) {
        self.spin_binning.set_visible(state);
        if let Some(cb_antialias_filter) = &self.cb_antialias_filter {
            cb_antialias_filter.set_visible(state);
        }
        self.cb_byte_mode_to_output.set_visible(state);
        self.cb_mean_float_densities.set_visible(state);
    }

    /// Java private `updateEnabled()` (PrenewstPanel.java:305-310).
    fn update_enabled(&self) {
        if let Some(cb_antialias_filter) = &self.cb_antialias_filter {
            let value = self.spin_binning.get_value();
            cb_antialias_filter.set_enabled(value.int_value() > 1);
        }
    }

    /// Java `done()` (PrenewstPanel.java:312-314).
    pub fn done(&self) {
        self.btn_coarse_align
            .remove_action_listener(&self.action_listener);
    }

    /// Java `getPanel()` (PrenewstPanel.java:316-318).
    pub fn get_panel(&self) -> Rc<JComponent> {
        self.pnl_prenewst.get_component()
    }

    /// Java `setAlignmentX(float)` (PrenewstPanel.java:320-322).
    pub fn set_alignment_x(&self, _align: f32) {
        // Swing layout: pnlPrenewst.setAlignmentX(align).
    }

    /// Java `setParameters(ConstMetaData)` (PrenewstPanel.java:324-328).
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        if !meta_data.is_antialias_filter_null(self.dialog_type, self.axis_id) {
            // Java dereferences antialiasFilterValue, which is null for a
            // montage (PrenewstPanel.java:326): a NullPointerException.  Fixed
            // in translation: there is no antialias filter to load for a
            // montage, so the value is skipped.
            if let Some(antialias_filter_value) = &self.antialias_filter_value {
                let antialias_filter =
                    meta_data.get_antialias_filter(self.dialog_type, self.axis_id);
                antialias_filter_value
                    .borrow_mut()
                    .set_const_etomo_number(antialias_filter.as_ref().map(|value| &**value));
            }
        }
    }

    /// Java `setParameters(BaseScreenState)` (PrenewstPanel.java:348-352).
    pub fn set_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        // btnCoarseAlign.setButtonState(screenState.getButtonState(btnCoarseAlign
        // .getButtonStateKey()));
        self.header
            .set_button_states_base_screen_state(Some(screen_state));
    }

    /// Java `getParameters(BaseScreenState)` (PrenewstPanel.java:354-356).
    pub fn get_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.header.get_button_states(Some(screen_state));
    }

    /// Java `getParameters(MetaData)` (PrenewstPanel.java:358-360).
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        let antialias_filter_value = self
            .antialias_filter_value
            .as_ref()
            .map(|value| value.borrow().clone());
        meta_data.set_antialias_filter(
            self.dialog_type,
            self.axis_id,
            antialias_filter_value.as_ref().map(|value| &**value),
        );
    }

    /// Java `getProcessName()` (PrenewstPanel.java:405-410).
    pub fn get_process_name(&self) -> ProcessName {
        if self.application_manager.get_meta_data().get_view_type() == ViewType::Montage {
            return ProcessName::PREBLEND;
        }
        ProcessName::PRENEWST
    }

    /// Java private `setToolTipText()` (PrenewstPanel.java:442-460).  Tooltip
    /// string initialization.
    fn set_tool_tip_text(&self) {
        self.spin_binning.set_tool_tip_text(Some(
            "Binning for the image stack used to generate and fix the fiducial model.",
        ));
        self.cb_byte_mode_to_output.set_tool_tip_text_string(Some(
            &*("Set the storage mode of the output file to bytes.  When unchecked the storage mode is the same as that of the first input file.  This option should be turned off when the dynamic range is still too poor after X ray removal.  Command:  ".to_string()
                + newst_param::DATA_MODE_OPTION
                + " "
                + &newst_param::DATA_MODE_BYTE.to_string()),
        ));
        self.cb_mean_float_densities.set_tool_tip_text_string(Some(
            &*("Adjust densities of sections individually.  Scale sections to common mean and standard deviation.  Command:  ".to_string()
                + newst_param::FLOAT_DENSITIES_OPTION
                + " "
                + &newst_param::FLOAT_DENSITIES_MEAN.to_string()),
        ));
        self.btn_coarse_align.set_tool_tip_text(Some(
            "Use transformations to produce stack of aligned images.",
        ));
        self.btn_imod
            .set_tool_tip_text(Some("Use 3dmod to view the coarsely aligned images."));
        if let Some(cb_antialias_filter) = &self.cb_antialias_filter {
            cb_antialias_filter.set_tool_tip_text_string(Some(
                &*("Use antialiased image reduction instead binning with the ".to_string()
                    + "default filter in Newstack; useful for data from direct detection "
                    + "cameras."),
            ));
        }
    }
}

impl Expandable for PrenewstPanel {
    /// Java `expand(GlobalExpandButton)` (PrenewstPanel.java:282-283).
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java `expand(ExpandButton)` (PrenewstPanel.java:285-294).
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_open_close(button) {
            self.pnl_body.set_visible(button.is_expanded());
        } else if self.header.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.application_manager))
        });
    }
}

impl NewstackDisplay for PrenewstPanel {
    /// Java `getParameters(NewstParam, boolean)` (PrenewstPanel.java:370-403).
    fn get_parameters(
        &self,
        prenewst_params: &mut NewstParam,
        _do_validation: bool,
    ) -> Result<bool, NewstackDisplayException> {
        prenewst_params.set_command_mode(Some(Mode::Prealigned));
        // Java `((Integer) spinBinning.getValue()).intValue()`.
        let binning = self.spin_binning.get_value().int_value();

        // Only explcitly write out the binning if its value is something other than
        // the default of 1 to keep from cluttering up the com script
        if binning > 1 {
            prenewst_params.set_bin_by_factor(Some(Number::Integer(binning)));
        } else {
            prenewst_params.set_bin_by_factor(Some(Number::Integer(i32::MIN)));
        }
        // Save when field is disabled
        // Java dereferences cbAntialiasFilter and antialiasFilterValue, which
        // are null for a montage (PrenewstPanel.java:385-389): a
        // NullPointerException.  Fixed in translation: with no antialias
        // filter check box the filter is off, as an unchecked box would say.
        let antialias_filter = self
            .cb_antialias_filter
            .as_ref()
            .is_some_and(|cb_antialias_filter| cb_antialias_filter.is_selected());
        prenewst_params.set_antialias_filter(antialias_filter);
        if antialias_filter {
            if let Some(antialias_filter_value) = &self.antialias_filter_value {
                prenewst_params.set_antialias_filter_value(&antialias_filter_value.borrow());
            }
        }
        if self.cb_byte_mode_to_output.is_selected() {
            prenewst_params.set_mode_to_output(newst_param::DATA_MODE_BYTE);
        } else {
            prenewst_params.set_mode_to_output(newst_param::DATA_MODE_DEFAULT);
        }
        if self.cb_mean_float_densities.is_selected() {
            prenewst_params.set_float_densities(newst_param::FLOAT_DENSITIES_MEAN);
        } else {
            prenewst_params.set_float_densities(newst_param::FLOAT_DENSITIES_DEFAULT);
        }
        Ok(true)
    }

    /// Java `setParameters(ConstNewstParam)` (PrenewstPanel.java:330-346).
    fn set_parameters(&self, prenewst_params: &dyn ConstNewstParam) {
        let binning = prenewst_params.get_bin_by_factor();
        if binning > 1 {
            self.spin_binning.set_value_int(binning);
        }
        let antialias_filter = !prenewst_params.is_antialias_filter_null();
        // Java dereferences cbAntialiasFilter and antialiasFilterValue, which
        // are null for a montage (PrenewstPanel.java:337-340): a
        // NullPointerException.  Fixed in translation: skipped for a montage.
        if let Some(cb_antialias_filter) = &self.cb_antialias_filter {
            cb_antialias_filter.set_selected_boolean(antialias_filter);
        }
        if antialias_filter {
            if let Some(antialias_filter_value) = &self.antialias_filter_value {
                antialias_filter_value
                    .borrow_mut()
                    .set_string(Some(&prenewst_params.get_antialias_filter()));
            }
        }
        self.cb_byte_mode_to_output.set_selected_boolean(
            prenewst_params.get_mode_to_output() == newst_param::DATA_MODE_BYTE,
        );
        self.cb_mean_float_densities.set_selected_boolean(
            prenewst_params.get_float_densities() == newst_param::FLOAT_DENSITIES_MEAN,
        );
        self.update_enabled();
    }

    /// Java `validate()` (PrenewstPanel.java:462-465).
    fn validate(&self) -> bool {
        true
    }

    /// Java `isFiducialess()` (PrenewstPanel.java:412-415).
    fn is_fiducialess(&self) -> bool {
        // Java `parent.isFiducialess()`; the parent owns this panel, so it is
        // alive whenever the panel is used.
        self.parent
            .upgrade()
            .is_some_and(|parent| parent.is_fiducialess())
    }
}

impl BlendmontDisplay for PrenewstPanel {
    /// Java `getParameters(BlendmontParam, boolean)` (PrenewstPanel.java:417-422).
    fn get_parameters(
        &self,
        blendmont_param: &mut BlendmontParam,
        _do_validation: bool,
    ) -> Result<bool, BlendmontDisplayException> {
        // Java `((Integer) spinBinning.getValue()).intValue()`.
        blendmont_param.set_bin_by_factor_int(self.spin_binning.get_value().int_value());
        Ok(true)
    }

    /// Java `setParameters(BlendmontParam)` (PrenewstPanel.java:362-368).
    fn set_parameters(&self, blendmont_params: &BlendmontParam) {
        let binning = blendmont_params.get_bin_by_factor();
        if !binning.is_null() {
            self.spin_binning.set_value_const_etomo_number(binning);
        }
    }

    /// Java `validate()` (PrenewstPanel.java:462-465).
    fn validate(&self) -> bool {
        true
    }

    /// Java `isFiducialess()` (PrenewstPanel.java:412-415).
    fn is_fiducialess(&self) -> bool {
        self.parent
            .upgrade()
            .is_some_and(|parent| parent.is_fiducialess())
    }
}

impl Run3dmodButtonContainer for PrenewstPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`
    /// (PrenewstPanel.java:476-486).  Executes the action associated with
    /// command.  Deferred3dmodButton is null if it comes from the dialog's
    /// ActionListener.  Otherwise is comes from a Run3dmodButton which called
    /// action(Run3dmodButton, Run3dmoMenuOptions).  In that case it will be
    /// null unless it was set in the Run3dmodButton.
    fn action(
        &self,
        command: &str,
        deferred3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // The manager takes the options by value; a Java null is the default
        // (no options set).
        let menu_options = menu_options.unwrap_or_default();
        if Some(command) == self.btn_coarse_align.get_action_command().as_deref() {
            self.application_manager.coarse_align(
                self.axis_id,
                Some(self.btn_coarse_align.clone() as ProcessResultDisplayHandle),
                None,
                deferred3dmod_button,
                menu_options,
                self.dialog_type,
                self,
                self,
            );
        } else if Some(command) == self.btn_imod.get_action_command().as_deref() {
            self.application_manager
                .imod_coarse_align(self.axis_id, menu_options, None, false);
        }
    }
}

impl ContextMenu for PrenewstPanel {
    /// Java `popUpContextMenu(MouseEvent)` (PrenewstPanel.java:427-437).  Right
    /// mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = ["Newstack".to_string()];
        let man_page = ["newstack.html".to_string()];
        let log_file_label = ["Prenewst".to_string()];
        let log_file = ["prenewst".to_string() + &self.axis_id.get_extension() + ".log"];
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_prenewst.get_component(),
            mouse_event,
            Some("COARSE ALIGNMENT"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label[..]),
            Some(&log_file[..]),
            self.application_manager,
            self.axis_id,
        );
    }
}
