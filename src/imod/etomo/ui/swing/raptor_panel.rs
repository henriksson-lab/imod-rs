//! `IMOD/Etomo/src/etomo/ui/swing/RaptorPanel.java`.
//!
//! Java `final class RaptorPanel implements Run3dmodButtonContainer,
//! ContextMenu`: the "Run RAPTOR" panel of the fiducial model dialog.  An EDT
//! object: created as `Rc<Self>` by [`RaptorPanel::get_instance`]; every
//! method takes `&self`.  The inner listener class `RaptorPanelActionListener`
//! is a closure holding a weak reference to the panel.

use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplay;
use crate::imod::etomo::util::event_queue::EdtRef;

use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::radio_button::RadioButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::{SpacedPanel, X_AXIS, Y_AXIS};
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::beadtrack_param::BeadtrackParam;
use crate::imod::etomo::comscript::runraptor_param::RunraptorParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent, MouseEvent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::util::utilities;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `MARK_LABEL`.
const MARK_LABEL: &str = "# of beads to choose";
/// Java private static final `DIAM_LABEL`.
const DIAM_LABEL: &str = "Unbinned Bead diameter";
/// Java package-private static final `RUN_RAPTOR_LABEL`.
pub const RUN_RAPTOR_LABEL: &str = "Run RAPTOR";
/// Java package-private static final `USE_RAPTOR_RESULT_LABEL`.
pub const USE_RAPTOR_RESULT_LABEL: &str = "Use RAPTOR Result as Fiducial Model";

/// Java `final class RaptorPanel implements Run3dmodButtonContainer, ContextMenu`.
pub struct RaptorPanel {
    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `pnlInput = new JPanel()`.
    pnl_input: Rc<JComponent>,
    /// Java private final `btnOpenStack`.
    btn_open_stack: Rc<Run3dmodButton>,
    /// Java private final `ltfMark`.
    ltf_mark: Rc<LabeledTextField>,
    /// Java private final `ltfDiam`.
    ltf_diam: Rc<LabeledTextField>,
    /// Java private final `bgInput = new ButtonGroup()`.
    #[allow(dead_code)]
    bg_input: Rc<ButtonGroup>,
    /// Java private final `rbInputPreali`.
    rb_input_preali: Rc<RadioButton>,
    /// Java private final `rbInputRaw`.
    rb_input_raw: Rc<RadioButton>,
    /// Java private final `btnRaptor`.
    btn_raptor: Rc<Run3dmodButton>,
    /// Java private final `btnOpenRaptorResult`.
    btn_open_raptor_result: Rc<Run3dmodButton>,
    /// Java private final `btnUseRaptorResult`.
    btn_use_raptor_result: Rc<MultiLineButton>,
    /// Java private final `actionListener` (a `RaptorPanelActionListener`).
    action_listener: ActionListener,

    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `dialogType` (stored, never read).
    #[allow(dead_code)]
    dialog_type: DialogType,
    /// Java `this`, for `createPanel`'s `btnRaptor.setContainer(this)` and the
    /// mouse adapter.
    this: Weak<RaptorPanel>,
}

impl RaptorPanel {
    /// Java private constructor `RaptorPanel(ApplicationManager, AxisID,
    /// DialogType)`, with the field initializers.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<RaptorPanel> {
        Rc::new_cyclic(|this: &Weak<RaptorPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            // Field initializers, in declaration order.
            let pnl_root = SpacedPanel::get_instance_void();
            let pnl_input = JComponent::new_panel();
            let btn_open_stack =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Stack in 3dmod"),
                    Some(container.clone()),
                );
            let ltf_mark = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(&format!("{MARK_LABEL}: ")),
            );
            let ltf_diam = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(&format!("{DIAM_LABEL} (in pixels): ")),
            );
            let bg_input = ButtonGroup::new();
            let rb_input_preali = RadioButton::new_string_button_group(
                Some("Run against the coarse aligned stack"),
                Some(&bg_input),
            );
            let rb_input_raw = RadioButton::new_string_button_group(
                Some("Run against the raw stack"),
                Some(&bg_input),
            );
            let btn_open_raptor_result =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open RAPTOR Model in 3dmod"),
                    Some(container.clone()),
                );
            // Java `new RaptorPanelActionListener(this)`: its `actionPerformed`
            // calls `adaptee.action(event.getActionCommand(), null, null)`.
            let adaptee = this.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                let Some(adaptee) = adaptee.upgrade() else {
                    return;
                };
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            });

            // Constructor body.
            let display_factory = manager.get_process_result_display_factory(axis_id);
            // Java casts `(Run3dmodButton) displayFactory.getRaptor()` and
            // `(MultiLineButton) displayFactory.getUseRaptor()`; the factory
            // returns the concrete buttons.
            let btn_raptor = display_factory.get_raptor();
            let btn_use_raptor_result = display_factory.get_use_raptor();
            RaptorPanel {
                pnl_root,
                pnl_input,
                btn_open_stack,
                ltf_mark,
                ltf_diam,
                bg_input,
                rb_input_preali,
                rb_input_raw,
                btn_raptor,
                btn_open_raptor_result,
                btn_use_raptor_result,
                action_listener,
                axis_id,
                manager,
                dialog_type,
                this: this.clone(),
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<RaptorPanel> {
        let instance = RaptorPanel::new(manager, axis_id, dialog_type);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let context_menu: Weak<dyn ContextMenu> = self.this.clone();
        self.pnl_root
            .add_mouse_listener(GenericMouseAdapter::new(context_menu));
        self.btn_open_stack
            .add_action_listener(self.action_listener.clone());
        self.btn_raptor
            .add_action_listener(self.action_listener.clone());
        self.btn_open_raptor_result
            .add_action_listener(self.action_listener.clone());
        self.btn_use_raptor_result
            .add_action_listener(self.action_listener.clone());
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        self.pnl_root.set_box_layout(Y_AXIS);
        self.pnl_root
            .set_border(&EtchedBorder::new(Some(RUN_RAPTOR_LABEL)).get_border());
        // Box.CENTER_ALIGNMENT
        self.pnl_root.set_alignment_x(0.5);
        self.pnl_root.add_j_panel(&self.pnl_input);
        self.pnl_root
            .add_component(&self.btn_open_stack.get_component());
        self.pnl_root.add_container(&self.ltf_mark.get_container());
        self.pnl_root.add_container(&self.ltf_diam.get_container());
        let pnl_raptor_buttons = SpacedPanel::get_instance_void();
        self.pnl_root.add_spaced_panel(&pnl_raptor_buttons);
        // RAPTOR input source panel
        // Swing layout: pnlInput BoxLayout Y_AXIS, etched border, CENTER_ALIGNMENT.
        self.pnl_input.add(&self.rb_input_preali.get_component());
        self.pnl_input.add(&self.rb_input_raw.get_component());
        // RAPTOR button panel
        pnl_raptor_buttons.set_box_layout(X_AXIS);
        pnl_raptor_buttons.add_component(&self.btn_raptor.get_component());
        pnl_raptor_buttons.add_component(&self.btn_open_raptor_result.get_component());
        pnl_raptor_buttons.add_component(&self.btn_use_raptor_result.get_component());
        // set initial values
        self.rb_input_preali.set_selected_boolean(true);
        // Swing layout: btnOpenStack.setAlignmentX(Box.CENTER_ALIGNMENT).
        // raptor button
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_raptor.set_container(Some(container));
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_open_raptor_result.clone();
        self.btn_raptor
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
    }

    /// Java package-private `done()`.
    pub fn done(&self) {
        self.btn_raptor
            .remove_action_listener(&self.action_listener);
        self.btn_open_raptor_result
            .remove_action_listener(&self.action_listener);
        self.btn_use_raptor_result
            .remove_action_listener(&self.action_listener);
    }

    /// Java package-private `setBeadtrackParams(BeadtrackParam)`.
    pub fn set_beadtrack_params(&self, beadtrack_params: &BeadtrackParam) {
        if beadtrack_params.is_bead_diameter_set() {
            self.ltf_diam.set_text_long(utilities::java_lang_math_round(
                beadtrack_params.get_bead_diameter().get_double(),
            ));
        }
    }

    /// Java package-private `getParameters(RunraptorParam, boolean)`.
    pub fn get_parameters_runraptor_param_boolean(
        &self,
        param: &mut RunraptorParam,
        do_validation: bool,
    ) -> bool {
        // Java try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            param.set_use_raw_stack(self.rb_input_raw.is_selected());
            // `getText(boolean)` may return null; `EtomoNumber.set(null)` and
            // `set("")` both make the number null, so "" stands in for null.
            let error_message = param.set_mark(
                self.ltf_mark
                    .get_text_boolean(do_validation)?
                    .as_deref()
                    .unwrap_or(""),
            );
            if let Some(error_message) = error_message {
                let manager: &'static dyn BaseManager = self.manager;
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(manager),
                        &format!("Error in {MARK_LABEL}: {error_message}"),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return Ok(false);
            }
            let error_message = param.set_diam(
                self.ltf_diam
                    .get_text_boolean(do_validation)?
                    .as_deref()
                    .unwrap_or(""),
                self.rb_input_preali.is_selected(),
            );
            if let Some(error_message) = error_message {
                let manager: &'static dyn BaseManager = self.manager;
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(manager),
                        &format!("Error in {DIAM_LABEL}: {error_message}"),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return Ok(false);
            }
            Ok(true)
        })();
        result.unwrap_or(false)
    }

    /// Java package-private `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        if self.axis_id != AxisID::Second {
            meta_data.set_track_raptor_use_raw_stack(self.rb_input_raw.is_selected());
            meta_data.set_track_raptor_mark(self.ltf_mark.get_text_void().as_deref());
            meta_data.set_track_raptor_diam(self.ltf_diam.get_text_void().as_deref());
        }
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters(&self, meta_data: &dyn ConstMetaData) {
        if self.axis_id != AxisID::Second {
            if meta_data.get_track_raptor_use_raw_stack() {
                self.rb_input_raw.set_selected_boolean(true);
            } else {
                self.rb_input_preali.set_selected_boolean(true);
            }
            self.ltf_mark
                .set_text_string(Some(&meta_data.get_track_raptor_mark()));
            let diam = meta_data.get_track_raptor_diam();
            if !diam.is_null() {
                self.ltf_diam.set_text_const_etomo_number(Some(&*diam));
            }
        }
        if self.manager.get_meta_data().get_view_type() == ViewType::Montage {
            self.rb_input_preali.set_selected_boolean(true);
            self.rb_input_raw.set_enabled(false);
            // pnlInput.setVisible(false);
        }
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        self.rb_input_preali
            .set_tool_tip_text_string(Some("Run RAPTOR against the coarsely aligned stack."));
        self.rb_input_raw
            .set_tool_tip_text_string(Some("Run RAPTOR against the raw stack."));
        self.btn_open_stack
            .set_tool_tip_text(Some("Opens the file that RAPTOR will be run against."));
        self.ltf_mark
            .set_tool_tip_text(Some("Number of markers to track."));
        self.ltf_diam
            .set_tool_tip_text(Some("Bead diameter in pixels."));
        self.btn_raptor
            .set_tool_tip_text(Some("Runs the runraptor script"));
        self.btn_open_raptor_result.set_tool_tip_text(Some(
            "Opens the model generated by RAPTOR and the file that RAPTOR was run against.",
        ));
        self.btn_use_raptor_result.set_tool_tip_text(Some(
            "Copies the model generated by RAPTOR to the .fid file.",
        ));
    }
}

impl ContextMenu for RaptorPanel {
    /// Java `popUpContextMenu(MouseEvent)`: right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel: Vec<String> = ["Raptor", "Beadtrack", "3dmod"]
            .iter()
            .map(|label| label.to_string())
            .collect();
        let man_page: Vec<String> = ["raptor.html", "beadtrack.html", "3dmod.html"]
            .iter()
            .map(|page| page.to_string())
            .collect();

        let log_file_label: Vec<String> = vec!["Track".to_string()];
        let mut log_file: Vec<String> = vec![String::new(); 1];
        log_file[0] = format!("track{}.log", self.axis_id.get_extension());

        let manager: &'static dyn BaseManager = self.manager;
        // Java `new ContextPopup(...)`; its constructor throws
        // IllegalArgumentException on mismatched array lengths, which these
        // arrays cannot have.
        if let Err(message) = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root.get_container(),
            mouse_event,
            Some("UsingRaptor"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(log_file_label.as_slice()),
            Some(log_file.as_slice()),
            manager,
            self.axis_id,
        ) {
            eprintln!("java.lang.IllegalArgumentException: {message}");
        }
    }
}

impl Run3dmodButtonContainer for RaptorPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    ///
    /// The action listener passes a null `Run3dmodMenuOptions`; the manager
    /// takes the options by value, and a default instance (every option off)
    /// is what the 3dmod layer makes of null (`ImodProcess` tests each option
    /// only on a non-null object).
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let menu_options = run_3dmod_menu_options.unwrap_or_default();
        if Some(command) == self.btn_open_stack.get_action_command().as_deref() {
            if self.rb_input_raw.is_selected() {
                self.manager.imod_raw_stack(self.axis_id, menu_options);
            } else {
                self.manager
                    .imod_coarse_align(self.axis_id, menu_options, None, false);
            }
        } else if Some(command) == self.btn_raptor.get_action_command().as_deref() {
            self.manager.runraptor(
                Some(self.btn_raptor.clone() as Rc<dyn ProcessResultDisplay>),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                DialogType::FiducialModel,
                self.axis_id,
            );
        } else if Some(command) == self.btn_open_raptor_result.get_action_command().as_deref() {
            self.manager
                .imod_runraptor_result(self.axis_id, run_3dmod_menu_options);
        } else if Some(command) == self.btn_use_raptor_result.get_action_command().as_deref() {
            self.manager.use_runraptor_result(
                Some(self.btn_use_raptor_result.clone() as Rc<dyn ProcessResultDisplay>),
                self.axis_id,
                DialogType::FiducialModel,
            );
        }
    }
}
