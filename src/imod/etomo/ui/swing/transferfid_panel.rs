//! `IMOD/Etomo/src/etomo/ui/swing/TransferfidPanel.java`.
//!
//! Java `final class TransferfidPanel implements Expandable,
//! Run3dmodButtonContainer`.  An EDT object: created as `Rc<Self>` by
//! [`TransferfidPanel::get_instance`], every method takes `&self`, mutable
//! state lives in `Cell`/`RefCell`.  `this` (handed to `PanelHeader` and to
//! the two `Run3dmodButton`s) is the weak self reference made by
//! `Rc::new_cyclic`.

use crate::imod::etomo::ui::field::Field;
use std::rc::{Rc, Weak};

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::transferfid_param::TransferfidParam;
use crate::imod::etomo::jdk::{ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::mirror_in_x::MirrorInX;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::dataset_files;

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::check_box::CheckBox;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::panel_header::PanelHeader;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::ui_harness;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplay;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java `final class TransferfidPanel implements Expandable, Run3dmodButtonContainer`.
pub struct TransferfidPanel {
    /// Java private final `panelTransferfid`.
    panel_transferfid: Rc<EtomoPanel>,
    /// Java private final `panelTransferfidBody`.
    panel_transferfid_body: Rc<JComponent>,
    /// Java private final `cbRunMidas`.
    cb_run_midas: Rc<CheckBox>,
    /// Java private final `ltfCenterViewA`.
    ltf_center_view_a: Rc<LabeledTextField>,
    /// Java private final `ltfCenterViewB`.
    ltf_center_view_b: Rc<LabeledTextField>,
    /// Java private final `ltfNumberViews`.
    ltf_number_views: Rc<LabeledTextField>,
    /// Java private final `panelSearchDirection`.
    panel_search_direction: Rc<EtomoPanel>,
    /// Java private final `bgSearchDirection`.
    bg_search_direction: Rc<ButtonGroup>,
    /// Java private final `rbSearchBoth`.
    rb_search_both: Rc<RadioButton>,
    /// Java private final `rbSearchPlus90`.
    rb_search_plus90: Rc<RadioButton>,
    /// Java private final `rbSearchMinus90`.
    rb_search_minus90: Rc<RadioButton>,
    /// Java private final `actionListener` (`TransferfidPanelActionListener`).
    action_listener: ActionListener,
    /// Java private final `btn3dmodSeed`.
    btn_3dmod_seed: Rc<Run3dmodButton>,
    /// Java private final `bgMirrorInX`.
    bg_mirror_in_x: Rc<ButtonGroup>,
    /// Java private final `rbMirrorInXAssessBoth`.
    rb_mirror_in_x_assess_both: Rc<RadioButton>,
    /// Java private final `rbMirrorInXAlways`.
    rb_mirror_in_x_always: Rc<RadioButton>,
    /// Java private final `rbMirrorInXNever`.
    rb_mirror_in_x_never: Rc<RadioButton>,
    /// Java private final `pnlOuterMirrorXaxis`.
    pnl_outer_mirror_xaxis: Rc<JComponent>,

    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Java private final `buttonTransferfid`.
    button_transferfid: Rc<Run3dmodButton>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
}

impl TransferfidPanel {
    /// Java private constructor `TransferfidPanel(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<TransferfidPanel> {
        Rc::new_cyclic(|this: &Weak<TransferfidPanel>| {
            // Field initializers, in declaration order.
            let panel_transferfid = EtomoPanel::new();
            let panel_transferfid_body = JComponent::new_panel();
            let cb_run_midas = CheckBox::new_string(Some("Run midas"));
            let ltf_center_view_a = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Center view A: "),
            );
            let ltf_center_view_b = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Center view B: "),
            );
            let ltf_number_views = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Number of views in the search: "),
            );
            let panel_search_direction = EtomoPanel::new();
            let bg_search_direction = ButtonGroup::new();
            let rb_search_both = RadioButton::new_string(Some("Both directions"));
            let rb_search_plus90 = RadioButton::new_string(Some("+90 (CCW) only"));
            let rb_search_minus90 = RadioButton::new_string(Some("-90 (CW) only"));
            // TransferfidPanelActionListener
            let action_listener: ActionListener = {
                let adaptee = this.clone();
                Rc::new(move |event| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                    }
                })
            };
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_3dmod_seed =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Seed Model"),
                    Some(container.clone()),
                );
            let bg_mirror_in_x = ButtonGroup::new();
            let rb_mirror_in_x_assess_both = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(MirrorInX::ASSESS_BOTH),
                Some(&bg_mirror_in_x),
            );
            let rb_mirror_in_x_always = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(MirrorInX::ALWAYS),
                Some(&bg_mirror_in_x),
            );
            let rb_mirror_in_x_never = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(MirrorInX::NEVER),
                Some(&bg_mirror_in_x),
            );
            let pnl_outer_mirror_xaxis = JComponent::new_panel();

            // Constructor body.
            // panels
            let pnl_mirror_xaxis = JComponent::new_panel();
            let expandable: Weak<dyn Expandable> = this.clone();
            let header =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Transfer Fiducials"),
                    Some(expandable),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );
            // Swing layout: panelTransferfidBody BoxLayout Y_AXIS.
            let pnl_run_midas = JComponent::new_panel();
            // Swing layout: pnlRunMidas BoxLayout X_AXIS, CENTER_ALIGNMENT.
            pnl_run_midas.add(&cb_run_midas.get_component());
            // Swing layout: horizontal glue; cbRunMidas RIGHT_ALIGNMENT.
            panel_transferfid_body.add(&pnl_run_midas);

            // Add a horizontal strut to keep the panel a minimum size
            // Swing layout: Box.createHorizontalStrut(300).
            panel_transferfid_body.add(&ltf_center_view_a.get_container());
            panel_transferfid_body.add(&ltf_center_view_b.get_container());
            panel_transferfid_body.add(&ltf_number_views.get_container());

            bg_search_direction.add(&rb_search_both.get_abstract_button());
            bg_search_direction.add(&rb_search_plus90.get_abstract_button());
            bg_search_direction.add(&rb_search_minus90.get_abstract_button());
            let opnl_search_direction = JComponent::new_panel();
            // Swing layout: opnlSearchDirection BoxLayout X_AXIS, CENTER_ALIGNMENT, glue.
            opnl_search_direction.add(&panel_search_direction.get_component());
            // Swing layout: horizontal glue; panelSearchDirection BoxLayout Y_AXIS.
            panel_search_direction
                .set_border(&EtchedBorder::new(Some("Search Direction")).get_border());
            panel_search_direction
                .get_component()
                .add(&rb_search_both.get_component());
            panel_search_direction
                .get_component()
                .add(&rb_search_plus90.get_component());
            panel_search_direction
                .get_component()
                .add(&rb_search_minus90.get_component());
            // Swing layout: panelSearchDirection CENTER_ALIGNMENT; rigid area x0_y1.
            panel_transferfid_body.add(&opnl_search_direction);
            // Swing layout: rigid area x0_y1.
            panel_transferfid_body.add(&pnl_outer_mirror_xaxis);
            // Swing layout: rigid area x0_y5.
            // OuterMirrorInX
            // Swing layout: pnlOuterMirrorXaxis BoxLayout X_AXIS, glue around pnlMirrorXaxis.
            pnl_outer_mirror_xaxis.add(&pnl_mirror_xaxis);
            // MirrorInX
            // Swing layout: pnlMirrorXaxis BoxLayout Y_AXIS; EtchedBorder.
            pnl_mirror_xaxis.set_border_title(
                EtchedBorder::new(Some("Mirroring around X axis"))
                    .get_border()
                    .get_title()
                    .as_deref(),
            );
            pnl_mirror_xaxis.add(&rb_mirror_in_x_assess_both.get_component());
            pnl_mirror_xaxis.add(&rb_mirror_in_x_always.get_component());
            pnl_mirror_xaxis.add(&rb_mirror_in_x_never.get_component());
            //
            let pnl_transferfid = JComponent::new_panel();
            // Swing layout: pnlTransferfid BoxLayout X_AXIS, CENTER_ALIGNMENT, glue.
            let button_transferfid = manager
                .get_process_result_display_factory(axis_id)
                .get_transfer_fiducials();
            button_transferfid.set_container(Some(container));
            // Swing layout: buttonTransferfid CENTER_ALIGNMENT.
            let deferred: Rc<dyn Deferred3dmodButton> = btn_3dmod_seed.clone();
            button_transferfid.set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
            pnl_transferfid.add(&button_transferfid.get_component());
            // Swing layout: horizontal glue.
            pnl_transferfid.add(&btn_3dmod_seed.get_component());
            // Swing layout: horizontal glue.
            panel_transferfid_body.add(&pnl_transferfid);
            // Swing layout: rigid area x0_y5; panelTransferfid BoxLayout Y_AXIS,
            // etched border.
            panel_transferfid.add(&header);
            panel_transferfid
                .get_component()
                .add(&panel_transferfid_body);

            TransferfidPanel {
                panel_transferfid,
                panel_transferfid_body,
                cb_run_midas,
                ltf_center_view_a,
                ltf_center_view_b,
                ltf_number_views,
                panel_search_direction,
                bg_search_direction,
                rb_search_both,
                rb_search_plus90,
                rb_search_minus90,
                action_listener,
                btn_3dmod_seed,
                bg_mirror_in_x,
                rb_mirror_in_x_assess_both,
                rb_mirror_in_x_always,
                rb_mirror_in_x_never,
                pnl_outer_mirror_xaxis,
                header,
                button_transferfid,
                axis_id,
                manager,
                dialog_type,
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType,
    /// GlobalExpandButton)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<TransferfidPanel> {
        let instance = TransferfidPanel::new(manager, axis_id, dialog_type, global_advanced_button);
        // The Java constructor ends with setToolTipText(); it needs the
        // constructed fields, so it runs here, before addListeners() as in Java.
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.panel_transferfid.get_component()
    }

    /// Java private `setup()` (empty in the Java).
    fn setup(&self) {}

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.panel_transferfid.get_component().set_visible(visible);
    }

    /// Java `setParameters()`: set the values of the panel using a
    /// TransferfidParam parameter object.
    pub fn set_parameters_void(&self) {
        let mut params = TransferfidParam::new(self.manager, self.axis_id);
        params.initialize();
        if self.axis_id == AxisID::Second {
            self.manager
                .get_meta_data()
                .get_transferfid_b_fields(&mut params);
        } else {
            self.manager
                .get_meta_data()
                .get_transferfid_a_fields(&mut params);
        }
        self.cb_run_midas
            .set_selected_boolean(params.get_run_midas().is());
        self.ltf_center_view_a
            .set_text_string(Some(&params.get_center_view_a().to_string()));
        self.ltf_center_view_b
            .set_text_string(Some(&params.get_center_view_b().to_string()));
        self.ltf_number_views
            .set_text_string(Some(&params.get_number_views().to_string()));

        if params.get_search_direction().is_null() {
            self.rb_search_both.set_selected_boolean(true);
        }
        if params.get_search_direction().is_negative() {
            self.rb_search_minus90.set_selected_boolean(true);
        }
        if params.get_search_direction().is_positive() {
            self.rb_search_plus90.set_selected_boolean(true);
        }
        let mirror_in_x = params.get_mirror_xaxis();
        if mirror_in_x == MirrorInX::ASSESS_BOTH {
            self.rb_mirror_in_x_assess_both.set_selected_boolean(true);
        }
        if mirror_in_x == MirrorInX::ALWAYS {
            self.rb_mirror_in_x_always.set_selected_boolean(true);
        }
        if mirror_in_x == MirrorInX::NEVER {
            self.rb_mirror_in_x_never.set_selected_boolean(true);
        }
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .set_button_states_base_screen_state(Some(screen_state));
    }

    /// Java `getParameters(BaseScreenState)`.
    pub fn get_parameters_base_screen_state(&self, screen_state: &BaseScreenState) {
        self.header.get_button_states(Some(screen_state));
    }

    /// Java `getParameters(boolean)`.
    pub fn get_parameters_boolean(&self, do_validation: bool) -> bool {
        let mut params = TransferfidParam::new(self.manager, self.axis_id);
        self.get_parameters_transferfid_param_boolean(&mut params, do_validation)
    }

    /// Java `getParameters(TransferfidParam, boolean)`: get the values from the
    /// panel filling in the TransferfidParam object.
    pub fn get_parameters_transferfid_param_boolean(
        &self,
        params: &mut TransferfidParam,
        do_validation: bool,
    ) -> bool {
        // Java try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            params.set_run_midas(self.cb_run_midas.is_selected());
            params.set_center_view_a(
                self.ltf_center_view_a
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            params.set_center_view_b(
                self.ltf_center_view_b
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            if self.rb_search_both.is_selected() {
                params.get_search_direction_mut().reset();
            }
            if self.rb_search_plus90.is_selected() {
                params.set_search_direction(1);
            }
            if self.rb_search_minus90.is_selected() {
                params.set_search_direction(-1);
            }
            params.set_number_views(
                self.ltf_number_views
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            // ((RadioButtonModel) bgMirrorInX.getSelection()).getEnumeratedType()
            // .getValue().  The group always has a selection: MirrorInX.DEFAULT
            // (ASSESS_BOTH) selects itself when its radio button is constructed.
            let enumerated_type = (self.bg_mirror_in_x.get_selection())
                .and_then(|button| button.get_model())
                .and_then(|model| {
                    model
                        .as_any()
                        .downcast_ref::<RadioButtonModel>()
                        .and_then(|model| model.get_enumerated_type())
                })
                .expect("bgMirrorInX has a selected RadioButtonModel");
            params.set_mirror_xaxis(Some(&enumerated_type.get_value()));
            if self.axis_id == AxisID::Second {
                self.manager
                    .get_meta_data()
                    .set_transferfid_b_fields(params);
            } else {
                self.manager
                    .get_meta_data()
                    .set_transferfid_a_fields(params);
            }
            Ok(true)
        })();
        match result {
            Ok(value) => value,
            Err(_) => false,
        }
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.button_transferfid
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_seed
            .add_action_listener(self.action_listener.clone());
    }

    /// Java `done()`.
    pub fn done(&self) {
        self.button_transferfid
            .remove_action_listener(&self.action_listener);
    }

    /// Java `updateAdvanced(boolean)`.
    pub fn update_advanced(&self, is_advanced: bool) {
        self.ltf_center_view_a.set_visible(is_advanced);
        self.ltf_center_view_b.set_visible(is_advanced);
        self.ltf_number_views.set_visible(is_advanced);
        self.pnl_outer_mirror_xaxis.set_visible(is_advanced);
        self.panel_search_direction
            .get_component()
            .set_visible(is_advanced);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, is_enabled: bool) {
        self.button_transferfid.set_enabled(is_enabled);
        self.cb_run_midas.set_enabled(is_enabled);
        self.ltf_center_view_a.set_enabled(is_enabled);
        self.ltf_center_view_b.set_enabled(is_enabled);
        self.ltf_number_views.set_enabled(is_enabled);
        self.rb_search_both.set_enabled(is_enabled);
        self.rb_search_plus90.set_enabled(is_enabled);
        self.rb_search_minus90.set_enabled(is_enabled);
        self.rb_mirror_in_x_assess_both.set_enabled(is_enabled);
        self.rb_mirror_in_x_always.set_enabled(is_enabled);
        self.rb_mirror_in_x_never.set_enabled(is_enabled);
    }

    /// Java private `setToolTipText()`: ToolTip string setup.
    fn set_tool_tip_text(&self) {
        self.cb_run_midas
            .set_tool_tip_text_string(Some("Run Midas to adjust initial alignment manually."));
        self.ltf_center_view_a.set_tool_tip_text(Some(
            "View from A around which to search for the best pair of views.",
        ));
        self.ltf_center_view_b.set_tool_tip_text(Some(
            "View from B around which to search for the best pair of views.",
        ));
        self.ltf_number_views.set_tool_tip_text(Some(
            "Number of views from each axis to consider in searching for best pair.",
        ));
        self.rb_search_both.set_tool_tip_text_string(Some(&format!(
            "{}{}",
            "Try both +90 and -90 degree rotations in searching for best pair of ", "views."
        )));
        self.rb_search_plus90.set_tool_tip_text_string(Some(
            "Try only +90 degree rotations in searching for best pair of views.",
        ));
        self.rb_search_minus90.set_tool_tip_text_string(Some(
            "Try only -90 degree rotations in searching for best pair of views.",
        ));
        self.button_transferfid.set_tool_tip_text(Some(&format!(
            "{}{}",
            "Run Transferfid to make a seed model for this axis from fiducial model for ",
            "the other axis."
        )));
        self.rb_mirror_in_x_always.set_tool_tip_text_string(Some(
            "Mirror one image around the X axis before rotating by 90 degrees.",
        ));
        self.rb_mirror_in_x_assess_both
            .set_tool_tip_text_string(Some(&format!(
                "{}{}",
                "Assess both mirroring one image around the X axis before rotating by 90 degrees, ",
                "and not mirroring.  Use the best method."
            )));
        self.rb_mirror_in_x_never.set_tool_tip_text_string(Some(
            "Do not mirror one image around the X axis before rotating by 90 degrees.",
        ));
    }
}

/// Java `Run3dmodButtonContainer`.
impl Run3dmodButtonContainer for TransferfidPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    ///
    /// Executes the action associated with command.  Deferred3dmodButton is
    /// null if it comes from the dialog's ActionListener.  Otherwise is comes
    /// from a Run3dmodButton which called action(Run3dmodButton,
    /// Run3dmoMenuOptions).  In that case it will be null unless it was set in
    /// the Run3dmodButton.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.button_transferfid.get_action_command().as_deref() {
            let display: Rc<dyn ProcessResultDisplay> = self.button_transferfid.clone();
            self.manager.transferfid(
                self.axis_id,
                Some(display),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options.unwrap_or_default(),
                self.dialog_type,
            );
        } else if Some(command) == self.btn_3dmod_seed.get_action_command().as_deref() {
            let display: Rc<dyn ProcessResultDisplay> = self.btn_3dmod_seed.clone();
            self.manager.imod_seed_model(
                self.axis_id,
                run_3dmod_menu_options.unwrap_or_default(),
                Some(display),
                imod_manager::COARSE_ALIGNED_KEY,
                dataset_files::get_seed_file_name(self.manager, Some(self.axis_id)).as_deref(),
                Some(&dataset_files::get_raw_tilt_file(
                    self.manager,
                    Some(self.axis_id),
                )),
                self.dialog_type,
            );
        }
    }
}

/// Java `Expandable`.
impl Expandable for TransferfidPanel {
    /// Java `expand(GlobalExpandButton)` (empty in the Java).
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_open_close(button) {
            self.panel_transferfid_body
                .set_visible(button.is_expanded());
        } else if self.header.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
    }
}
