//! `IMOD/Etomo/src/etomo/ui/swing/SmoothingAssessmentPanel.java`.
//!
//! Java `final class SmoothingAssessmentPanel implements FlattenWarpDisplay,
//! Run3dmodButtonContainer`: the Smoothing Assessment box of the Flatten
//! panel (Post Processing dialog, and the Tools flatten tool) - a list of
//! smoothing factors and the buttons that run flattenwarp with them and open
//! the resulting model.
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`SmoothingAssessmentPanel::get_post_instance`] /
//! [`SmoothingAssessmentPanel::get_tools_instance`]; every method takes
//! `&self`.  The listener class `SmoothingAssessmentActionListener` is a
//! closure holding a weak reference to the panel.  The parent
//! (`FlattenVolumePanel`, which owns this panel) is held weakly.

use std::rc::{Rc, Weak};

use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::flatten_volume_panel;
use super::flatten_warp_display::FlattenWarpDisplay;
use super::labeled_text_field::LabeledTextField;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::smoothing_assessment_parent::SmoothingAssessmentParent;
use super::spaced_panel::{self, SpacedPanel};
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::flatten_warp_param::{self, FlattenWarpParam};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Java public static final `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java private static final `LAMBDA_FOR_SMOOTHING_LABEL`.
const LAMBDA_FOR_SMOOTHING_LABEL: &str = "Smoothing factors to try";
/// Java static final `FLATTEN_WARP_LABEL`.
pub const FLATTEN_WARP_LABEL: &str = "Run Flattenwarp to Assess Smoothing";

/// Java `final class SmoothingAssessmentPanel`.
pub struct SmoothingAssessmentPanel {
    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `ltfLambdaForSmoothing`.
    ltf_lambda_for_smoothing: Rc<LabeledTextField>,
    /// Java private final `btn3dmod = Run3dmodButton.get3dmodInstance("Open
    /// Assessment in 3dmod", this, FileType.SMOOTHING_ASSESSMENT_OUTPUT_MODEL)`.
    btn_3dmod: Rc<Run3dmodButton>,
    /// Java private final `actionListener = new
    /// SmoothingAssessmentActionListener(this)`.
    action_listener: ActionListener,

    /// Java private final `btnFlattenWarp`.
    btn_flatten_warp: Rc<Run3dmodButton>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `applicationManager`; null for the tools instance.
    application_manager: Option<&'static ApplicationManager>,
    /// Java private final `toolsManager`; null for the post-processing
    /// instance.
    tools_manager: Option<&'static ToolsManager>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `parent` (the owning panel, held weakly).
    parent: Weak<dyn SmoothingAssessmentParent>,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `panelId`.
    panel_id: PanelId,
}

impl SmoothingAssessmentPanel {
    /// The field initializers shared by both Java constructors.  Returns the
    /// `Rc`; `btn_flatten_warp` comes from `make_btn_flatten_warp`, which is
    /// the only statement the two constructors do differently.
    #[allow(clippy::too_many_arguments)]
    fn new_fields(
        manager: &'static dyn BaseManager,
        application_manager: Option<&'static ApplicationManager>,
        tools_manager: Option<&'static ToolsManager>,
        axis_id: AxisID,
        dialog_type: DialogType,
        panel_id: PanelId,
        parent: Weak<dyn SmoothingAssessmentParent>,
        make_btn_flatten_warp: impl FnOnce(Weak<dyn Run3dmodButtonContainer>) -> Rc<Run3dmodButton>,
    ) -> Rc<SmoothingAssessmentPanel> {
        let instance = Rc::new_cyclic(|self_ref: &Weak<SmoothingAssessmentPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            // Field initializers.
            let pnl_root = SpacedPanel::get_instance_void();
            let ltf_lambda_for_smoothing = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointArray,
                Some(&format!("{LAMBDA_FOR_SMOOTHING_LABEL}: ")),
            );
            let smoothing_key: &FileKey = &file_type::CLASS.smoothing_assessment_output_model;
            let btn_3dmod =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container_file_key(
                    Some("Open Assessment in 3dmod"),
                    Some(container.clone()),
                    Some(smoothing_key.clone()),
                );
            // Java `new SmoothingAssessmentActionListener(this)`.
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });
            // Constructor body.
            let btn_flatten_warp = make_btn_flatten_warp(container.clone());
            // btnFlattenWarp.setContainer(this);
            btn_flatten_warp.set_container(Some(container));
            SmoothingAssessmentPanel {
                pnl_root,
                ltf_lambda_for_smoothing,
                btn_3dmod,
                action_listener,
                btn_flatten_warp,
                manager,
                application_manager,
                tools_manager,
                axis_id,
                parent,
                dialog_type,
                panel_id,
            }
        });
        instance
    }

    /// Java private constructor `SmoothingAssessmentPanel(ApplicationManager,
    /// AxisID, DialogType, PanelId, SmoothingAssessmentParent)`.
    fn new_application_manager_axis_id_dialog_type_panel_id_smoothing_assessment_parent(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        panel_id: PanelId,
        parent: Weak<dyn SmoothingAssessmentParent>,
    ) -> Rc<SmoothingAssessmentPanel> {
        SmoothingAssessmentPanel::new_fields(
            manager,
            Some(manager),
            None,
            axis_id,
            dialog_type,
            panel_id,
            parent,
            // btnFlattenWarp = (Run3dmodButton) manager
            //   .getProcessResultDisplayFactory(axisID).getSmoothingAssessment();
            |_container| {
                manager
                    .get_process_result_display_factory(axis_id)
                    .get_smoothing_assessment()
            },
        )
    }

    /// Java private constructor `SmoothingAssessmentPanel(ToolsManager,
    /// AxisID, DialogType, PanelId, SmoothingAssessmentParent)`.
    fn new_tools_manager_axis_id_dialog_type_panel_id_smoothing_assessment_parent(
        manager: &'static ToolsManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        panel_id: PanelId,
        parent: Weak<dyn SmoothingAssessmentParent>,
    ) -> Rc<SmoothingAssessmentPanel> {
        SmoothingAssessmentPanel::new_fields(
            manager,
            None,
            Some(manager),
            axis_id,
            dialog_type,
            panel_id,
            parent,
            // btnFlattenWarp = Run3dmodButton.getDeferred3dmodInstance(
            //   FLATTEN_WARP_LABEL, this);
            |container| {
                Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                    Some(FLATTEN_WARP_LABEL),
                    Some(container),
                )
            },
        )
    }

    /// Java package-private static `getPostInstance(ApplicationManager, AxisID,
    /// DialogType, PanelId, SmoothingAssessmentParent)`.
    pub fn get_post_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        panel_id: PanelId,
        parent: Weak<dyn SmoothingAssessmentParent>,
    ) -> Rc<SmoothingAssessmentPanel> {
        let instance =
            SmoothingAssessmentPanel::new_application_manager_axis_id_dialog_type_panel_id_smoothing_assessment_parent(
                manager,
                axis_id,
                dialog_type,
                panel_id,
                parent,
            );
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java package-private static `getToolsInstance(ToolsManager, AxisID,
    /// DialogType, PanelId, SmoothingAssessmentParent)`.
    pub fn get_tools_instance(
        manager: &'static ToolsManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        panel_id: PanelId,
        parent: Weak<dyn SmoothingAssessmentParent>,
    ) -> Rc<SmoothingAssessmentPanel> {
        let instance =
            SmoothingAssessmentPanel::new_tools_manager_axis_id_dialog_type_panel_id_smoothing_assessment_parent(
                manager,
                axis_id,
                dialog_type,
                panel_id,
                parent,
            );
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.btn_flatten_warp
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod
            .add_action_listener(self.action_listener.clone());
    }

    /// Java package-private `done()`.
    pub fn done(&self) {
        self.btn_flatten_warp
            .remove_action_listener(&self.action_listener);
    }

    /// Java private `createPanel()`.
    fn create_panel(self: &Rc<Self>) {
        // initialize
        let container: Weak<dyn Run3dmodButtonContainer> = Rc::downgrade(self) as Weak<_>;
        self.btn_flatten_warp.set_container(Some(container));
        self.btn_flatten_warp
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_3dmod.clone() as Rc<dyn Deferred3dmodButton>
            ));
        self.ltf_lambda_for_smoothing.set_text_string(Some(
            flatten_warp_param::LAMBDA_FOR_SMOOTHING_ASSESSMENT_DEFAULT,
        ));
        // Local panels
        let pnl_buttons = JComponent::new_panel();
        // root panel
        self.pnl_root.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_root
            .set_border(&EtchedBorder::new(Some("Smoothing Assessment")).get_border());
        self.pnl_root
            .add_container(&self.ltf_lambda_for_smoothing.get_container());
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x0_y5)) (a
        // rigid area through SpacedPanel.add(Component); its spacing is layout).
        self.pnl_root.add_j_panel(&pnl_buttons);
        // Buttons panel
        // Swing layout: pnlButtons.setLayout(new BoxLayout(pnlButtons,
        // BoxLayout.X_AXIS)).
        pnl_buttons.add(&self.btn_flatten_warp.get_component());
        // Swing layout: pnlButtons.add(Box.createRigidArea(FixedDim.x5_y0)).
        pnl_buttons.add(&self.btn_3dmod.get_component());
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters(&self, meta_data: &dyn ConstMetaData) {
        if !meta_data.is_lambda_for_smoothing_list_empty() {
            self.ltf_lambda_for_smoothing
                .set_text_string(Some(&meta_data.get_lambda_for_smoothing_list()));
        }
    }

    /// Java package-private `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_lambda_for_smoothing_list(
            self.ltf_lambda_for_smoothing.get_text_void().as_deref(),
        );
    }

    /// Java private `validateFlattenWarp()`.
    fn validate_flatten_warp(&self) -> bool {
        if self.ltf_lambda_for_smoothing.is_empty() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &format!("{LAMBDA_FOR_SMOOTHING_LABEL} is required."),
                    "Entry Error",
                    Some(self.axis_id),
                )
            });
            return false;
        }
        true
    }

    /// Java package-private `setTooltips()`.
    pub fn set_tooltips(&self) {
        self.btn_flatten_warp
            .set_tool_tip_text(Some("Run flattenwarp with different smoothing factors."));
        self.btn_3dmod
            .set_tool_tip_text(Some("Open model created by flattenwarp."));
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::FLATTEN_WARP),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // `catch (final LockException except) {}`.
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException except)`:
            // `except.printStackTrace()`.
            Err(except) => eprintln!("{}", except),
        }
        // SAFETY: `autodoc` is null or an autodoc the factory keeps for the life
        // of the process.
        let autodoc: Option<&dyn ReadOnlyAutodoc> = if autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*autodoc })
        };
        if autodoc.is_some() {
            self.ltf_lambda_for_smoothing
                .set_tool_tip_text(Some(&format!(
                    "A list of different LambdaForSmoothing values.  {}",
                    etomo_autodoc::get_tooltip(
                        autodoc,
                        Some(flatten_warp_param::LAMBDA_FOR_SMOOTHING_OPTION)
                    )
                    .as_deref()
                    .unwrap_or("null")
                )));
        }
    }
}

impl FlattenWarpDisplay for SmoothingAssessmentPanel {
    /// Java public `getParameters(FlattenWarpParam, boolean)`.
    fn get_parameters(&self, param: &mut FlattenWarpParam, do_validation: bool) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            let mut error_message = param.set_lambda_for_smoothing(
                self.ltf_lambda_for_smoothing
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            if let Some(message) = &error_message {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!("Error in {LAMBDA_FOR_SMOOTHING_LABEL}:  {message}"),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return Ok(false);
            }
            param.set_middle_contour_file(
                file_type::CLASS
                    .smoothing_assessment_output_model
                    .get_file_name(Some(self.manager), Some(self.axis_id))
                    .as_deref(),
            );
            // Rust-only: the parent is held weakly; it owns this panel, so it is
            // always alive while the panel is used.
            let Some(parent) = self.parent.upgrade() else {
                return Ok(false);
            };
            param.set_one_surface(parent.is_one_surface());
            error_message =
                param.set_warp_spacing_x(parent.get_warp_spacing_x(do_validation)?.as_deref());
            if let Some(message) = &error_message {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!(
                            "Error in {}:  {message}",
                            flatten_volume_panel::WARP_SPACING_X_LABEL
                        ),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return Ok(false);
            }
            error_message =
                param.set_warp_spacing_y(parent.get_warp_spacing_y(do_validation)?.as_deref());
            if let Some(message) = &error_message {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!(
                            "Error in {}:  {message}",
                            flatten_volume_panel::WARP_SPACING_Y_LABEL
                        ),
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
}

impl Run3dmodButtonContainer for SmoothingAssessmentPanel {
    /// Java public `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if self.panel_id == PanelId::PostFlattenVolume {
            if Some(command) == self.btn_flatten_warp.get_action_command().as_deref() {
                if self.validate_flatten_warp() {
                    if let Some(application_manager) = self.application_manager {
                        // Java passes a null Run3dmodMenuOptions from the action
                        // listener; ImodState.open replaces null with a new
                        // Run3dmodMenuOptions(), which is the default value.
                        application_manager.flatten_warp(
                            Some(self.btn_flatten_warp.clone() as ProcessResultDisplayHandle),
                            None,
                            deferred_3dmod_button,
                            run_3dmod_menu_options.unwrap_or_default(),
                            self.dialog_type,
                            self.axis_id,
                            self,
                        );
                    }
                }
            } else if Some(command) == self.btn_3dmod.get_action_command().as_deref() {
                if let Some(application_manager) = self.application_manager {
                    application_manager.imod_view_model(
                        self.axis_id,
                        &file_type::CLASS.smoothing_assessment_output_model,
                    );
                }
            } else {
                // Java throws `IllegalStateException("Unknown command " +
                // command)`, which Swing reports on the EDT and survives.  The
                // translation reports it and returns.
                eprintln!("java.lang.IllegalStateException: Unknown command {command}");
            }
        } else if self.panel_id == PanelId::ToolsFlattenVolume {
            if Some(command) == self.btn_flatten_warp.get_action_command().as_deref() {
                if self.validate_flatten_warp() {
                    if let Some(tools_manager) = self.tools_manager {
                        tools_manager.flatten_warp(
                            Some(self.btn_flatten_warp.clone() as ProcessResultDisplayHandle),
                            None,
                            deferred_3dmod_button,
                            run_3dmod_menu_options,
                            Some(self.dialog_type),
                            self.axis_id,
                            self,
                        );
                    }
                }
            } else if Some(command) == self.btn_3dmod.get_action_command().as_deref() {
                if let Some(tools_manager) = self.tools_manager {
                    tools_manager.imod_view_model(
                        self.axis_id,
                        &file_type::CLASS.smoothing_assessment_output_model,
                    );
                }
            } else {
                // Java throws `IllegalStateException`; see above.
                eprintln!("java.lang.IllegalStateException: Unknown command {command}");
            }
        }
    }
}
