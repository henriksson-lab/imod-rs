//! `IMOD/Etomo/src/etomo/ui/swing/TiltAnglePanel.java` (with its file-level
//! class `TiltAngleDialogListener`).
//!
//! An extremely thin GUI (bug# 1052): the decisions live in
//! [`TiltAnglePanelExpert`], which constructs this panel and is handed back to
//! it for the radio buttons' listener.  The panel is an event-dispatch-thread
//! object (`Rc`, `&self`).  It keeps its expert and its parent `SetupDialog`
//! as `Weak` references: the expert owns the panel and the dialog owns the
//! expert (through `SetupDialogExpert`), so strong back references would be
//! `Rc` cycles.

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::image_file_meta_data::ImageFileMetaData;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::swing::etomo_button_group::EtomoButtonGroup;
use crate::imod::etomo::ui::swing::labeled_text_field::LabeledTextField;
use crate::imod::etomo::ui::swing::radio_button::RadioButton;
use crate::imod::etomo::ui::swing::radio_ebutton::RadioEbutton;
use crate::imod::etomo::ui::swing::setup_dialog::SetupDialog;
use crate::imod::etomo::ui::swing::tilt_angle_panel_expert::TiltAnglePanelExpert;
use crate::imod::etomo::ui::swing::tooltip_formatter;
use crate::imod::etomo::util::utilities;
use std::cell::RefCell;
use std::path::Path;
use std::rc::{Rc, Weak};

/// Java `final class TiltAnglePanel`.
pub struct TiltAnglePanel {
    /// Java private final `pnlSource`.
    pnl_source: Rc<JComponent>,
    /// Java private final `bgSource`.
    bg_source: Rc<EtomoButtonGroup>,
    /// Java private final `rbExtract`.
    rb_extract: Rc<RadioButton>,
    /// Java private final `pnlAngle`.
    pnl_angle: Rc<JComponent>,
    /// Java private final `rbSpecify`.
    rb_specify: Rc<RadioEbutton>,
    /// Java private final `ltfMin`.
    ltf_min: Rc<LabeledTextField>,
    /// Java private final `ltfStep`.
    ltf_step: Rc<LabeledTextField>,
    /// Java private final `rbFile`.
    rb_file: Rc<RadioEbutton>,
    /// Java private final `lExcludeViewsMsg`.
    l_exclude_views_msg: Rc<JComponent>,
    /// Java private final `expert`.  Weak: the expert owns this panel.
    expert: Weak<TiltAnglePanelExpert>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private `parent`.  Weak: the dialog (through its expert) owns
    /// this panel's expert.
    parent: RefCell<Option<Weak<SetupDialog>>>,
}

impl TiltAnglePanel {
    /// Java package-private `TiltAnglePanel(ApplicationManager,
    /// TiltAnglePanelExpert, AxisID)`, with the field initialisers.
    ///
    /// The expert constructs the panel while it is itself being constructed
    /// (`panel = new TiltAnglePanel(manager, this, axisID)`), so it passes
    /// the `Weak` of its own `Rc::new_cyclic`.
    pub fn new(
        manager: &'static ApplicationManager,
        expert: Weak<TiltAnglePanelExpert>,
        axis_id: AxisID,
    ) -> Rc<TiltAnglePanel> {
        let pnl_source = JComponent::new_panel();
        let bg_source = EtomoButtonGroup::new();
        let rb_extract = RadioButton::new_string_button_group(
            Some(TiltAngleType::Extract.get_descr()),
            Some(&bg_source.get_button_group()),
        );
        let pnl_angle = JComponent::new_panel();
        let rb_specify =
            RadioEbutton::get_instance(Some(TiltAngleType::Range.get_descr()), Some(&bg_source));
        let ltf_min = LabeledTextField::new_field_type_string(
            FieldType::FloatingPoint,
            Some("Starting angle:"),
        );
        let ltf_step =
            LabeledTextField::new_field_type_string(FieldType::FloatingPoint, Some("Increment:"));
        let rb_file =
            RadioEbutton::get_instance(Some(TiltAngleType::File.get_descr()), Some(&bg_source));
        let l_exclude_views_msg = JComponent::new_label("");
        let panel = Rc::new(TiltAnglePanel {
            pnl_source,
            bg_source,
            rb_extract,
            pnl_angle,
            rb_specify,
            ltf_min,
            ltf_step,
            rb_file,
            l_exclude_views_msg,
            expert: expert.clone(),
            axis_id,
            manager,
            parent: RefCell::new(None),
        });
        let pnl_file = JComponent::new_panel();
        let pnl_extract = JComponent::new_panel();
        let pnl_specify = JComponent::new_panel();
        // Swing layout: pnlAngle BoxLayout X_AXIS.
        panel.pnl_angle.add(&panel.ltf_min.get_component());
        // Swing layout: rigid area x10_y0.
        panel.pnl_angle.add(&panel.ltf_step.get_component());
        // Swing layout: horizontal glue.

        // Swing layout: pnlSource BoxLayout Y_AXIS.
        panel.pnl_source.add(&panel.l_exclude_views_msg);
        panel.pnl_source.add(&pnl_extract);
        panel.pnl_source.add(&pnl_specify);
        panel.pnl_source.add(&panel.pnl_angle);

        panel.pnl_source.add(&pnl_file);
        panel.pnl_source.add(&pnl_file);
        // Extract
        // Swing layout: pnlExtract BoxLayout X_AXIS.
        pnl_extract.add(&panel.rb_extract.get_component());
        // Swing layout: horizontal glue.
        // Specify
        // Swing layout: pnlSpecify BoxLayout X_AXIS.
        pnl_specify.add(&panel.rb_specify.get_component());
        // Swing layout: horizontal glue.
        // File
        // Swing layout: pnlFile BoxLayout X_AXIS.
        pnl_file.add(&panel.rb_file.get_component());
        // Swing layout: horizontal glue.

        // `TiltAngleDialogListener tiltAlignRadioButtonListener = new
        // TiltAngleDialogListener(expert)`, one listener shared by the three
        // radio buttons.
        let tilt_align_radio_button_listener = Rc::new(TiltAngleDialogListener::new(expert));
        let listener = tilt_align_radio_button_listener.clone();
        panel
            .rb_extract
            .add_action_listener(Rc::new(move |event: &ActionEvent| {
                listener.action_performed(event)
            }));
        let listener = tilt_align_radio_button_listener.clone();
        panel
            .rb_file
            .add_action_listener(Rc::new(move |event: &ActionEvent| {
                listener.action_performed(event)
            }));
        let listener = tilt_align_radio_button_listener;
        panel
            .rb_specify
            .add_action_listener(Rc::new(move |event: &ActionEvent| {
                listener.action_performed(event)
            }));
        panel
    }

    /// Java package-private `msgExcludeViewsSucceeded()`.
    pub fn msg_exclude_views_succeeded(&self) {
        if self.rb_specify.is_selected() {
            self.rb_file.set_selected(true);
        }
        self.update_display();
    }

    /// Java package-private `setParent(SetupDialog)`.
    pub fn set_parent(&self, parent: Weak<SetupDialog>) {
        *self.parent.borrow_mut() = Some(parent);
    }

    /// Java package-private `updateDisplay()`.
    pub fn update_display(&self) {
        let specify = self.rb_specify.is_selected() && self.rb_specify.is_enabled();
        self.ltf_min.set_enabled(specify);
        self.ltf_step.set_enabled(specify);
        let parent = self.parent.borrow().as_ref().and_then(Weak::upgrade);
        // Upstream bug fixed in translation (TiltAnglePanel.java:139-143): the
        // source reads `parent.getDatasetName()` before its own
        // `parent != null` test, so a panel with no parent throws a
        // NullPointerException.  With no parent there is no dataset name, so
        // the "no dataset" branch below is taken.
        let dataset_name = parent.as_ref().and_then(|parent| parent.get_dataset_name());
        match (dataset_name, parent) {
            (None, _) => {
                self.rb_specify.disable_warning();
                self.rb_file
                    .set_label(Some(TiltAngleType::File.get_descr()));
                self.rb_file.disable_warning();
            }
            (Some(dataset_name), Some(parent)) => {
                let axis_type = parent.get_axis_type();
                let mut cur_axis_id = self.axis_id;
                if cur_axis_id != AxisID::Second {
                    if axis_type == AxisType::DualAxis {
                        cur_axis_id = AxisID::First;
                    } else {
                        cur_axis_id = AxisID::Only;
                    }
                }
                // Metadata hasn't been created so derive the file name.
                //
                // Java tests `manager.getMetaData()` for null and falls back to
                // a temporary `ImageFileMetaData`; the Rust manager's metadata
                // always exists, so the first branch is the one taken.  The
                // fallback is kept for the case the source guards against.
                let meta_data = Some(self.manager.get_meta_data());
                let image_filename_style;
                let raw_stack_extension;
                if let Some(meta_data) = meta_data {
                    image_filename_style = meta_data.get_image_filename_style();
                    raw_stack_extension = meta_data.get_raw_image_stack_extension();
                } else {
                    let image_file_meta_data = ImageFileMetaData::get_temp_instance();
                    image_filename_style = image_file_meta_data.get_image_filename_style();
                    raw_stack_extension =
                        Some(image_file_meta_data.get_default_raw_image_stack_extension());
                }
                let file_name = file_type::CLASS.raw_tilt_angles.derive_file_name(
                    Some(&dataset_name),
                    Some(axis_type),
                    Some(cur_axis_id),
                    Some(image_filename_style),
                    raw_stack_extension,
                );
                // `new File(parent.getDirectory(), name)`: a null parent
                // directory leaves the name alone.  A null derived name would
                // throw in `new File`; no file can exist without a name, so it
                // is treated as not found.
                let exists = file_name.is_some_and(|file_name| {
                    let file = match parent.get_directory() {
                        Some(directory) => utilities::java_io_file_new(&directory, &file_name),
                        None => file_name,
                    };
                    Path::new(&file).exists()
                });
                if exists || parent.is_remove_exclude_views_msg(self.axis_id) {
                    self.rb_specify.enable_warning(true);
                } else {
                    self.rb_specify.disable_warning();
                }
                if exists {
                    self.rb_file.set_label(Some(&format!(
                        "{} (File was found)",
                        TiltAngleType::File.get_descr()
                    )));
                    self.rb_file.disable_warning();
                } else {
                    self.rb_file.set_label(Some(&format!(
                        "{} (File was not found)",
                        TiltAngleType::File.get_descr()
                    )));
                    self.rb_file.enable_warning(true);
                }
            }
            // `else if (parent != null)` is false: nothing more to do.
            (Some(_), None) => {}
        }
    }

    /// Java package-private `checkpoint()`.
    pub fn checkpoint(&self) {
        self.rb_extract.checkpoint_void();
        self.rb_file.checkpoint();
        self.rb_specify.checkpoint();
    }

    /// Java package-private `updateTemplateValues(DirectiveFileCollection,
    /// AxisID)`.
    pub fn update_template_values(
        &self,
        directive_file_collection: &DirectiveFileCollection,
        axis_id: AxisID,
    ) {
        if directive_file_collection.contains_tilt_angle_spec(Some(axis_id)) {
            let mut tilt_angle_spec = TiltAngleSpec::new();
            directive_file_collection.get_tilt_angle_fields(
                Some(axis_id),
                Some(&mut tilt_angle_spec),
                false,
            );
            if tilt_angle_spec.get_type() == TiltAngleType::Extract {
                self.rb_extract.set_selected_boolean(true);
            } else if tilt_angle_spec.get_type() == TiltAngleType::File {
                self.rb_file.set_selected(true);
            }
        } else if self.rb_extract.is_checkpoint_value() {
            self.rb_extract.set_selected_boolean(true);
        } else if self.rb_file.is_checkpoint_value() {
            self.rb_file.set_selected(true);
        } else if self.rb_specify.is_checkpoint_value() {
            self.rb_specify.set_selected(true);
        } else {
            self.rb_extract.set_selected_boolean(false);
            self.rb_file.set_selected(false);
            self.rb_specify.set_selected(false);
        }
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_source.clone()
    }

    /// Java package-private `setFile(boolean)`.
    pub fn set_file(&self, input: bool) {
        self.rb_file.set_selected(input);
    }

    /// Java package-private `setExtract(boolean)`.
    pub fn set_extract(&self, input: bool) {
        self.rb_extract.set_selected_boolean(input);
    }

    /// Java package-private `setSpecify(boolean)`.
    pub fn set_specify(&self, input: bool) {
        self.rb_specify.set_selected(input);
    }

    /// Java package-private `setMin(double)`.
    pub fn set_min(&self, input: f64) {
        self.ltf_min.set_text_double(input);
    }

    /// Java package-private `setStep(double)`.
    pub fn set_step(&self, input: f64) {
        self.ltf_step.set_text_double(input);
    }

    /// Java package-private `setMinEnabled(boolean)`.
    pub fn set_min_enabled(&self, enable: bool) {
        self.ltf_min.set_enabled(enable);
    }

    /// Java package-private `setStepEnabled(boolean)`.
    pub fn set_step_enabled(&self, enable: bool) {
        self.ltf_step.set_enabled(enable);
    }

    /// Java package-private `isExtractSelected()`.
    pub fn is_extract_selected(&self) -> bool {
        self.rb_extract.is_selected()
    }

    /// Java package-private `isSpecifySelected()`.
    pub fn is_specify_selected(&self) -> bool {
        self.rb_specify.is_selected()
    }

    /// Java package-private `isFileSelected()`.
    pub fn is_file_selected(&self) -> bool {
        self.rb_file.is_selected()
    }

    /// Java package-private `getMin()`.
    pub fn get_min_void(&self) -> String {
        // JTextField.getText() never returns null for a live field.
        self.ltf_min.get_text_void().unwrap_or_default()
    }

    /// Java package-private `getMin(boolean) throws
    /// FieldValidationFailedException`.
    pub fn get_min_boolean(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_min
            .get_text_boolean(do_validation)
            .map(Option::unwrap_or_default)
    }

    /// Java package-private `getStep()`.
    pub fn get_step_void(&self) -> String {
        // JTextField.getText() never returns null for a live field.
        self.ltf_step.get_text_void().unwrap_or_default()
    }

    /// Java package-private `getStep(boolean) throws
    /// FieldValidationFailedException`.
    pub fn get_step_boolean(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_step
            .get_text_boolean(do_validation)
            .map(Option::unwrap_or_default)
    }

    /// Java package-private `setSourceEnabled(boolean)`.
    pub fn set_source_enabled(&self, enable: bool) {
        self.pnl_source.set_enabled(enable);
    }

    /// Java package-private `setAngleEnabled(boolean)`.
    pub fn set_angle_enabled(&self, enable: bool) {
        self.pnl_angle.set_enabled(enable);
    }

    /// Java package-private `setExtractEnabled(boolean)`.
    pub fn set_extract_enabled(&self, enable: bool) {
        self.rb_extract.set_enabled(enable);
    }

    /// Java package-private `setFileEnabled(boolean)`.
    pub fn set_file_enabled(&self, enable: bool) {
        self.rb_file.set_enabled(enable);
    }

    /// Java package-private `setSpecifyEnabled(boolean)`.
    pub fn set_specify_enabled(&self, enable: bool) {
        self.rb_specify.set_enabled(enable);
    }

    /// Java package-private `setSourceTooltip(String)`.
    pub fn set_source_tooltip(&self, tooltip: &str) {
        self.pnl_source
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(Some(tooltip)).as_deref());
    }

    /// Java package-private `setExtractTooltip(String)`.
    pub fn set_extract_tooltip(&self, tooltip: &str) {
        self.rb_extract.set_tool_tip_text_string(Some(tooltip));
    }

    /// Java package-private `setSpecifyTooltip(String)`.
    pub fn set_specify_tooltip(&self, tooltip: &str) {
        self.rb_specify.set_tooltip_string(Some(tooltip));
    }

    /// Java package-private `setMinTooltip(String)`.
    pub fn set_min_tooltip(&self, tooltip: &str) {
        self.ltf_min.set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setStepTooltip(String)`.
    pub fn set_step_tooltip(&self, tooltip: &str) {
        self.ltf_step.set_tool_tip_text(Some(tooltip));
    }

    /// Java package-private `setFileTooltip(String)`.
    pub fn set_file_tooltip(&self, tooltip: &str) {
        self.rb_file.set_tooltip_string(Some(tooltip));
    }

    /// Java package-private `getSpecify()`.
    pub fn get_specify(&self) -> Option<String> {
        Some(self.rb_specify.get_label())
    }
}

/// Java file-level `final class TiltAngleDialogListener implements
/// ActionListener`.
pub struct TiltAngleDialogListener {
    /// Java private final `adaptee`.  Weak: the expert owns the panel whose
    /// buttons hold this listener.
    adaptee: Weak<TiltAnglePanelExpert>,
}

impl TiltAngleDialogListener {
    /// Java package-private `TiltAngleDialogListener(TiltAnglePanelExpert)`.
    pub fn new(adaptee: Weak<TiltAnglePanelExpert>) -> TiltAngleDialogListener {
        TiltAngleDialogListener { adaptee }
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: &ActionEvent) {
        let Some(adaptee) = self.adaptee.upgrade() else {
            return;
        };
        adaptee.set_radio_button_state(event);
    }
}
