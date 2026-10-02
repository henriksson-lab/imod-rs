//! `IMOD/Etomo/src/etomo/ui/swing/FindBeads3dPanel.java`.
//!
//! Java `final class FindBeads3dPanel implements FindBeads3dDisplay,
//! Expandable, Run3dmodButtonContainer`.  An EDT object: created as `Rc<Self>`
//! by [`FindBeads3dPanel::get_instance`]; every method takes `&self`.  The
//! inner listener class `FindBeads3dPanelActionListener` is a closure holding
//! a weak reference to the panel; the private inner enumerated type
//! `StorageThresholdEnum` is [`StorageThresholdEnum`].

use std::fmt;
use std::rc::{Rc, Weak};

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_find_beads3d_param::ConstFindBeads3dParam;
use crate::imod::etomo::comscript::find_beads3d_param::{self, FindBeads3dParam};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::find_beads3d_display::FindBeads3dDisplay;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::newstack_or_blendmont_3d_find_parent::NewstackOrBlendmont3dFindParent;
use super::panel_header::PanelHeader;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::radio_text_field::RadioTextField;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::ui_harness;
use crate::imod::etomo::r#type::const_panel_header_settings::ConstPanelHeaderSettings;

/// Java package-private static final `BEAD_SIZE_LABEL`.
pub const BEAD_SIZE_LABEL: &str = "Bead diameter";

/// Java `final class FindBeads3dPanel`.
pub struct FindBeads3dPanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `actionListener` (`FindBeads3dPanelActionListener`).
    action_listener: ActionListener,
    /// Java private final `pnlBody = SpacedPanel.getInstance(true)`.
    pnl_body: Rc<SpacedPanel>,
    /// Java private final `ltfBeadSize`.
    ltf_bead_size: Rc<LabeledTextField>,
    /// Java private final `ltfMinSpacing`.
    ltf_min_spacing: Rc<LabeledTextField>,
    /// Java private final `ltfGuessNumBeads`.
    ltf_guess_num_beads: Rc<LabeledTextField>,
    /// Java private final `ltfMinRelativeStrength`.
    ltf_min_relative_strength: Rc<LabeledTextField>,
    /// Java private final `ltfThresholdForAveraging`.
    ltf_threshold_for_averaging: Rc<LabeledTextField>,
    /// Java private final `bgStorageThreshold = new ButtonGroup()`.
    bg_storage_threshold: Rc<crate::imod::etomo::jdk::ButtonGroup>,
    /// Java private final `rbStorageThresholdSomeBelow`.
    rb_storage_threshold_some_below: Rc<RadioButton>,
    /// Java private final `rbStorageThresholdOnlyAbove`.
    rb_storage_threshold_only_above: Rc<RadioButton>,
    /// Java private final `rtfStorageThreshold`.
    rtf_storage_threshold: Rc<RadioTextField>,
    /// Java private final `ltfMaxNumBeads`.
    ltf_max_num_beads: Rc<LabeledTextField>,
    /// Java private final `btn3dmodFindBeads3d`.
    btn_3dmod_find_beads3d: Rc<Run3dmodButton>,

    /// Java private final `btnFindBeads3d`.
    btn_find_beads3d: Rc<Run3dmodButton>,
    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `parent` (held weakly: the parent owns this panel).
    parent: Weak<dyn NewstackOrBlendmont3dFindParent>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java `this` (for `btnFindBeads3d.setContainer(this)`).
    this: Weak<FindBeads3dPanel>,
}

impl FindBeads3dPanel {
    /// Java private constructor `FindBeads3dPanel(ApplicationManager,
    /// NewstackOrBlendmont3dFindParent, AxisID, DialogType, GlobalExpandButton)`.
    fn new(
        manager: &'static ApplicationManager,
        parent: Weak<dyn NewstackOrBlendmont3dFindParent>,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<FindBeads3dPanel> {
        Rc::new_cyclic(|this: &Weak<FindBeads3dPanel>| {
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            // FindBeads3dPanelActionListener
            let action_listener: ActionListener = {
                let adaptee = this.clone();
                Rc::new(move |event: &ActionEvent| {
                    if let Some(adaptee) = adaptee.upgrade() {
                        adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                    }
                })
            };
            let pnl_body = SpacedPanel::get_instance_boolean(true);
            let ltf_bead_size = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(&format!("{BEAD_SIZE_LABEL} (pixels): ")),
            );
            let ltf_min_spacing = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Minimum spacing: "),
            );
            let ltf_guess_num_beads = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Estimated number of beads: "),
            );
            let ltf_min_relative_strength = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Minimum peak strength: "),
            );
            let ltf_threshold_for_averaging = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some("Threshold for averaging: "),
            );
            let bg_storage_threshold = crate::imod::etomo::jdk::ButtonGroup::new();
            let rb_storage_threshold_some_below =
                RadioButton::new_string_enumerated_type_button_group(
                    Some("Store some points below threshold"),
                    Some(EnumeratedTypeRef::new(StorageThresholdEnum::SOME_BELOW)),
                    Some(&bg_storage_threshold),
                );
            let rb_storage_threshold_only_above =
                RadioButton::new_string_enumerated_type_button_group(
                    Some("Store only points above threshold"),
                    Some(EnumeratedTypeRef::new(StorageThresholdEnum::ONLY_ABOVE)),
                    Some(&bg_storage_threshold),
                );
            let rtf_storage_threshold = RadioTextField::get_instance_field_type_string_button_group(
                FieldType::FloatingPoint,
                Some("Set threshold for storing: "),
                Some(&bg_storage_threshold),
            );
            let ltf_max_num_beads = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Max points to analyze: "),
            );
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_3dmod_find_beads3d =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View 3D Model on Tomogram"),
                    Some(container),
                );
            // Constructor body.
            let expandable: Weak<dyn Expandable> = this.clone();
            let header =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Find Beads 3d"),
                    Some(expandable),
                    Some(dialog_type),
                    Some(global_advanced_button.clone()),
                );
            // Java casts `(Run3dmodButton) ...getFindBeads3d()`; the factory
            // returns the concrete button.
            let btn_find_beads3d = manager
                .get_process_result_display_factory(axis_id)
                .get_find_beads3d();
            FindBeads3dPanel {
                pnl_root,
                action_listener,
                pnl_body,
                ltf_bead_size,
                ltf_min_spacing,
                ltf_guess_num_beads,
                ltf_min_relative_strength,
                ltf_threshold_for_averaging,
                bg_storage_threshold,
                rb_storage_threshold_some_below,
                rb_storage_threshold_only_above,
                rtf_storage_threshold,
                ltf_max_num_beads,
                btn_3dmod_find_beads3d,
                btn_find_beads3d,
                header,
                manager,
                parent,
                axis_id,
                dialog_type,
                this: this.clone(),
            }
        })
    }

    /// Java static `getInstance(ApplicationManager,
    /// NewstackOrBlendmont3dFindParent, AxisID, DialogType, GlobalExpandButton)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        parent: Weak<dyn NewstackOrBlendmont3dFindParent>,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<FindBeads3dPanel> {
        let instance = FindBeads3dPanel::new(
            manager,
            parent,
            axis_id,
            dialog_type,
            global_advanced_button,
        );
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        self.btn_find_beads3d
            .add_action_listener(self.action_listener.clone());
        self.btn_3dmod_find_beads3d
            .add_action_listener(self.action_listener.clone());
    }

    /// Java `done()`.
    pub fn done(&self) {
        self.btn_find_beads3d
            .remove_action_listener(&self.action_listener);
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // Initialize
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_find_beads3d.set_container(Some(container));
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_3dmod_find_beads3d.clone();
        self.btn_find_beads3d
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        // Local panels
        let pnl_storage_threshold = JComponent::new_panel();
        let pnl_buttons = SpacedPanel::get_instance_void();
        let pnl_a = SpacedPanel::get_instance_void();
        let pnl_b = SpacedPanel::get_instance_void();
        // Root panel
        // Swing layout: pnlRoot BoxLayout Y_AXIS.
        self.pnl_root.add(&self.header.get_container());
        // Swing layout: pnlRoot etched border (untitled).
        self.pnl_root.add(&self.pnl_body.get_container());
        // Body panel
        // Swing layout: pnlBody.setBoxLayout(BoxLayout.Y_AXIS).
        self.pnl_body
            .add_container(&self.ltf_bead_size.get_container());
        self.pnl_body.add_spaced_panel(&pnl_a);
        self.pnl_body.add_spaced_panel(&pnl_b);
        self.pnl_body.add_j_panel(&pnl_storage_threshold);
        self.pnl_body
            .add_container(&self.ltf_max_num_beads.get_container());
        self.pnl_body.add_spaced_panel(&pnl_buttons);
        // Panel A
        // Swing layout: pnlA.setBoxLayout(BoxLayout.X_AXIS).
        pnl_a.add_container(&self.ltf_min_spacing.get_container());
        pnl_a.add_container(&self.ltf_guess_num_beads.get_container());
        // Panel B
        // Swing layout: pnlB.setBoxLayout(BoxLayout.X_AXIS).
        pnl_b.add_container(&self.ltf_min_relative_strength.get_container());
        pnl_b.add_container(&self.ltf_threshold_for_averaging.get_container());
        // Storage threshold panel
        // Swing layout: pnlStorageThreshold GridLayout(3, 2, 3, 3).
        pnl_storage_threshold.set_border_title(
            EtchedBorder::new(Some("Storage Threshold"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_storage_threshold.add(&self.rb_storage_threshold_some_below.get_component());
        pnl_storage_threshold.add(&self.rb_storage_threshold_only_above.get_component());
        pnl_storage_threshold.add(&self.rtf_storage_threshold.get_container());
        // Button panel
        // Swing layout: pnlButtons.setBoxLayout(BoxLayout.X_AXIS).
        pnl_buttons.add_component(&self.btn_find_beads3d.get_component());
        pnl_buttons.add_component(&self.btn_3dmod_find_beads3d.get_component());
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `isAdvanced()`.
    pub fn is_advanced(&self) -> bool {
        self.header.is_advanced()
    }

    /// Java `updateAdvanced(boolean)`.
    pub fn update_advanced(&self, advanced: bool) {
        self.ltf_min_spacing.set_visible(advanced);
        self.ltf_guess_num_beads.set_visible(advanced);
        self.ltf_min_relative_strength.set_visible(advanced);
        self.ltf_threshold_for_averaging.set_visible(advanced);
        self.ltf_max_num_beads.set_visible(advanced);
    }

    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .get_state(Some(screen_state.get_stack_find_beads3d_header_state()));
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .set_state(Some(screen_state.get_stack_find_beads3d_header_state()));
        self.btn_find_beads3d.set_button_state(
            screen_state.get_button_state(self.btn_find_beads3d.get_button_state_key().as_deref()),
        );
    }

    /// Java `setParameters(ConstFindBeads3dParam, boolean)`.
    pub fn set_parameters_const_find_beads3d_param_boolean(
        &self,
        param: &dyn ConstFindBeads3dParam,
        initialize: bool,
    ) {
        if initialize {
            // Bead size starts out as unbinned bead diameter is pixels.
            self.ltf_bead_size
                .set_text_double(self.manager.calc_unbinned_bead_diameter_pixels());
            self.ltf_min_spacing.set_text_double(0.9);
            self.ltf_min_relative_strength.set_text_double(0.05);
        } else {
            self.ltf_bead_size
                .set_text_string(Some(&param.get_bead_size()));
            self.ltf_min_spacing
                .set_text_string(Some(&param.get_min_spacing()));
            self.ltf_guess_num_beads
                .set_text_string(Some(&param.get_guess_num_beads()));
            self.ltf_min_relative_strength
                .set_text_string(Some(&param.get_min_relative_strength()));
            self.ltf_threshold_for_averaging
                .set_text_string(Some(&param.get_threshold_for_averaging()));
            // Set StorageThreshold
            let storage_threshold = param.get_storage_threshold();
            let storage_threshold_enum = StorageThresholdEnum::get_instance(storage_threshold);
            match storage_threshold_enum {
                None => self
                    .rtf_storage_threshold
                    .set_text_const_etomo_number(storage_threshold),
                Some(StorageThresholdEnum::SomeBelow) => self
                    .rb_storage_threshold_some_below
                    .set_selected_boolean(true),
                Some(StorageThresholdEnum::OnlyAbove) => self
                    .rb_storage_threshold_only_above
                    .set_selected_boolean(true),
            }
            self.ltf_max_num_beads
                .set_text_string(Some(&param.get_max_num_beads()));
        }
    }

    /// Java `getBeadSize()`.
    pub fn get_bead_size(&self) -> String {
        // Java `ltfBeadSize.getText()`: a text field's text is never null.
        self.ltf_bead_size.get_text_void().unwrap_or_default()
    }

    /// The body of Java `getParameters(FindBeads3dParam, boolean)`, whose
    /// `FieldValidationFailedException` is caught by the caller.
    fn get_parameters_find_beads3d_param_boolean(
        &self,
        param: &mut FindBeads3dParam,
        do_validation: bool,
    ) -> Result<(), FieldValidationFailedException> {
        let manager: &'static dyn BaseManager = self.manager;
        param.set_input_file(&file_type::CLASS.tilt_3d_find_output);
        param.set_output_file(
            file_type::CLASS
                .find_beads_3d_output_model
                .get_file_name(Some(manager), Some(self.axis_id))
                .as_deref(),
        );
        param.set_bead_size(
            self.ltf_bead_size
                .get_text_boolean(do_validation)?
                .as_deref(),
        );
        param.set_min_spacing(
            self.ltf_min_spacing
                .get_text_boolean(do_validation)?
                .as_deref(),
        );
        param.set_guess_num_beads(
            self.ltf_guess_num_beads
                .get_text_boolean(do_validation)?
                .as_deref(),
        );
        param.set_min_relative_strength(
            self.ltf_min_relative_strength
                .get_text_boolean(do_validation)?
                .as_deref(),
        );
        param.set_threshold_for_averaging(
            self.ltf_threshold_for_averaging
                .get_text_boolean(do_validation)?
                .as_deref(),
        );
        if !Field::is_selected(&*self.rtf_storage_threshold) {
            // ((RadioButton.RadioButtonModel) bgStorageThreshold.getSelection())
            //   .getEnumeratedType().getValue()
            let value = self
                .bg_storage_threshold
                .get_selection()
                .and_then(|button| button.get_model())
                .and_then(|model| {
                    model
                        .as_any()
                        .downcast_ref::<RadioButtonModel>()
                        .and_then(|model| model.get_enumerated_type())
                })
                .map(|enumerated_type| enumerated_type.get_value());
            // Upstream NPE fixed in translation (FindBeads3dPanel.java:237-239):
            // with no storage threshold radio selected (the group is created
            // with SOME_BELOW selected, so this is only reachable after the
            // text radio lost its selection by other means) Java dereferences a
            // null model.  Here the storage threshold is left unset.
            param.set_storage_threshold_const_etomo_number(value.as_ref());
        } else {
            param.set_storage_threshold_string(
                self.rtf_storage_threshold
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
        }
        param.set_max_num_beads(
            self.ltf_max_num_beads
                .get_text_boolean(do_validation)?
                .as_deref(),
        );
        Ok(())
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        let mut autodoc: Option<*mut Autodoc> = None;
        // SAFETY: `AutodocFactory` owns every autodoc it returns for the life
        // of the process, so the pointer stays valid for this method.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::FIND_BEADS_3D),
                self.axis_id,
                false,
            )
        } {
            Ok(instance) => autodoc = Some(instance),
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except)
            Err(except) => eprintln!("{except}"),
        }
        // SAFETY: see above.
        let autodoc: Option<&dyn ReadOnlyAutodoc> =
            autodoc.map(|autodoc| unsafe { &*autodoc } as &dyn ReadOnlyAutodoc);
        self.ltf_bead_size
            .set_tool_tip_text(Some("Size of beads in unbinned pixels."));
        self.ltf_min_spacing.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(find_beads3d_param::MIN_SPACING_TAG))
                .as_deref(),
        );
        self.ltf_guess_num_beads.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(find_beads3d_param::GUESS_NUM_BEADS_TAG))
                .as_deref(),
        );
        self.ltf_min_relative_strength.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(find_beads3d_param::MIN_RELATIVE_STRENGTH_TAG),
            )
            .as_deref(),
        );
        self.ltf_threshold_for_averaging.set_tool_tip_text(
            etomo_autodoc::get_tooltip(
                autodoc,
                Some(find_beads3d_param::THRESHOLD_FOR_AVERAGING_TAG),
            )
            .as_deref(),
        );
        self.rb_storage_threshold_some_below
            .set_tool_tip_text_string(Some(
                "Model will include some points that are probably not beads, because their \
                 relative peak strengths are below the threshold between beads and non-beads",
            ));
        self.rb_storage_threshold_only_above
            .set_tool_tip_text_string(Some(
                "Model will include only the points with relative peak strengths above the \
                 threshold between beads and non-beads",
            ));
        Field::set_tool_tip_text(
            &*self.rtf_storage_threshold,
            Some("Threshold relative peak strength (between 0 and 1) for storing peaks in model"),
        );
        self.ltf_max_num_beads.set_tool_tip_text(
            etomo_autodoc::get_tooltip(autodoc, Some(find_beads3d_param::MAX_NUM_BEADS_TAG))
                .as_deref(),
        );
        self.btn_find_beads3d.set_tool_tip_text(Some(
            "Run findbeads3d to find gold particles in the tomogram.",
        ));
        self.btn_3dmod_find_beads3d
            .set_tool_tip_text(Some("View model of gold particles."));
    }
}

impl FindBeads3dDisplay for FindBeads3dPanel {
    /// Java override `getParameters(FindBeads3dParam, boolean)`.
    fn get_parameters(&self, param: &mut FindBeads3dParam, do_validation: bool) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        self.get_parameters_find_beads3d_param_boolean(param, do_validation)
            .is_ok()
    }

    /// Java override `isFiducialess()`.
    fn is_fiducialess(&self) -> bool {
        // Java dereferences `parent` unconditionally; the parent owns this panel.
        self.parent
            .upgrade()
            .is_some_and(|parent| parent.is_fiducialess())
    }
}

impl Expandable for FindBeads3dPanel {
    /// Java override `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_open_close(button) {
            self.pnl_body.set_visible(button.is_expanded());
        } else if self.header.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::INSTANCE
            .with(|harness| harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager)));
    }

    /// Java override `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}

impl Run3dmodButtonContainer for FindBeads3dPanel {
    /// Java override `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.btn_find_beads3d.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_find_beads3d.clone();
            // The manager takes the options by value; a Java null is the
            // default (no) options.
            self.manager.find_beads3d(
                Some(display),
                None,
                deferred_3dmod_button,
                self.axis_id,
                run_3dmod_menu_options.unwrap_or_default(),
                self.dialog_type,
                self,
            );
        } else if Some(command) == self.btn_3dmod_find_beads3d.get_action_command().as_deref() {
            let manager: &'static dyn BaseManager = self.manager;
            let model = file_type::CLASS
                .find_beads_3d_output_model
                .get_file_name(Some(manager), Some(self.axis_id));
            self.manager.imod_find_beads3d(
                self.axis_id,
                run_3dmod_menu_options.unwrap_or_default(),
                None,
                // Java passes the file type's key; this file type always has one.
                file_type::CLASS
                    .tilt_3d_find_output
                    .get_imod_manager_key()
                    .expect("tilt_3d_find_output has a 3dmod key"),
                model.as_deref(),
                None,
                self.dialog_type,
            );
        }
    }
}

/// Java `private static final class StorageThresholdEnum implements
/// EnumeratedType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageThresholdEnum {
    /// Java `SOME_BELOW = new StorageThresholdEnum(0)`.
    SomeBelow,
    /// Java `ONLY_ABOVE = new StorageThresholdEnum(-1)`.
    OnlyAbove,
}

impl StorageThresholdEnum {
    /// Java `SOME_BELOW`.
    pub const SOME_BELOW: StorageThresholdEnum = StorageThresholdEnum::SomeBelow;
    /// Java `ONLY_ABOVE`.
    pub const ONLY_ABOVE: StorageThresholdEnum = StorageThresholdEnum::OnlyAbove;

    /// Java field `value`: `new EtomoNumber()` then `value.set(int)`.
    fn value(self) -> EtomoNumber {
        let mut value = EtomoNumber::new();
        value.set_int(match self {
            Self::SomeBelow => 0,
            Self::OnlyAbove => -1,
        });
        value
    }

    /// Java private static `getInstance(ConstEtomoNumber)`.
    fn get_instance(storage_threshold: &ConstEtomoNumber) -> Option<StorageThresholdEnum> {
        if Self::SOME_BELOW
            .value()
            .equals_const_etomo_number(Some(storage_threshold))
        {
            return Some(Self::SOME_BELOW);
        }
        if Self::ONLY_ABOVE
            .value()
            .equals_const_etomo_number(Some(storage_threshold))
        {
            return Some(Self::ONLY_ABOVE);
        }
        // Don't return default because some values do not belong in one of these
        // categories.
        None
    }
}

impl fmt::Display for StorageThresholdEnum {
    /// Java inherits `Object.toString()`; nothing reads it.  The value is
    /// shown, as the other enumerated types do.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value(), f)
    }
}

impl EnumeratedType for StorageThresholdEnum {
    /// Java `getValue()`.
    fn get_value(&self) -> ConstEtomoNumber {
        self.value().base
    }

    /// Java `isDefault()`.
    fn is_default(&self) -> bool {
        if *self == Self::SOME_BELOW {
            return true;
        }
        false
    }

    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String> {
        None
    }
}
