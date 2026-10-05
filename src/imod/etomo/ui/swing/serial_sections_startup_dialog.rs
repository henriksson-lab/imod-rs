//! `IMOD/Etomo/src/etomo/ui/swing/SerialSectionsStartupDialog.java`.
//!
//! The modal "Starting Serial Sections" dialog: the stack, its frame type (single
//! frame or montage), the .mdoc piece-list choice, the distortion field and binning.
//! OK saves the choices (`SerialSectionsStartupData`) and starts the manager's
//! `completeStartup` series; Cancel closes the manager.
//!
//! An event dispatch thread object (`Rc`, `&self` methods).  The `JDialog` is the
//! `jdk` stand-in: `display()`'s blocking modal `setVisible(true)` returns at once
//! here, and `EtomoDirector` runs what Java runs after it once the dialog is hidden
//! (`JDialog::after_modal_return`).

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::busy_status_panel::BusyStatusPanel;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::ContextPopup;
use super::file_text_field_interface::FileTextFieldInterface;
use super::file_text_field2::FileTextField2;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::labeled_spinner::LabeledSpinner;
use super::multi_line_button::MultiLineButton;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::result_listener::ResultListener;
use super::swing_component::SwingComponent;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{
    self, ActionEvent, ActionListener, ButtonGroup, JComponent, JDialog, MouseEvent, WindowListener,
};
use crate::imod::etomo::logic::config_tool;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::logic::serial_sections_startup_data::SerialSectionsStartupData;
use crate::imod::etomo::serial_sections_manager::SerialSectionsManager;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::storage::distortion_file_filter::DistortionFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::extension::{self, Extension};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::swing::abstract_radio_button_model::AbstractRadioButtonModel;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::shared_constants;
use crate::imod::etomo::util::utilities;

/// Java private static final `NAME`.
const NAME: &str = "Starting Serial Sections";
/// Java private static final `VIEW_TYPE_LABEL`.
const VIEW_TYPE_LABEL: &str = "Frame Type";

/// Java `public class SerialSectionsStartupDialog implements ContextMenu, UIComponent,
/// SwingComponent, ResultListener`.
pub struct SerialSectionsStartupDialog {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `btnOk`.
    btn_ok: Rc<MultiLineButton>,
    /// Java private final `btnCancel`.
    btn_cancel: Rc<MultiLineButton>,
    /// Java private final `bgViewType`.
    bg_view_type: Rc<ButtonGroup>,
    /// Java private final `rbViewTypeSingle`.
    rb_view_type_single: Rc<RadioButton>,
    /// Java private final `rbViewTypeMontage`.
    rb_view_type_montage: Rc<RadioButton>,
    /// Java private final `spImagesAreBinned`.
    sp_images_are_binned: Rc<LabeledSpinner>,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `cbMdocMetadataFile`.
    cb_mdoc_metadata_file: Rc<CheckBox>,
    /// Java private final `ftfStack`.
    ftf_stack: Rc<FileTextField2>,
    /// Java private final `ftfDistortionField`.
    ftf_distortion_field: Rc<FileTextField2>,
    /// Java private final `dialog`.
    dialog: Rc<JDialog>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static SerialSectionsManager,
    /// Java private final `busyStatusPanel`.
    busy_status_panel: Rc<BusyStatusPanel>,
    /// Java private `startupData`, initially null.  Contains the saved state of the
    /// dialog.
    startup_data: RefCell<Option<SerialSectionsStartupData>>,
    /// `this`, for the listeners.
    this: Weak<SerialSectionsStartupDialog>,
}

/// The dialog as the `ResultListener` `FileTextField2.addResultListener` takes
/// (Java passes `this`).
struct StartupResultListener(Weak<SerialSectionsStartupDialog>);

impl ResultListener for StartupResultListener {
    fn process_result(&mut self, result_origin: &dyn std::any::Any, init: bool) {
        if let Some(dialog) = self.0.upgrade() {
            dialog.process_result(result_origin, init);
        }
    }
}

impl SerialSectionsStartupDialog {
    /// Java private `SerialSectionsStartupDialog(SerialSectionsManager, AxisID)`.
    fn new(
        manager: &'static SerialSectionsManager,
        axis_id: AxisID,
    ) -> Rc<SerialSectionsStartupDialog> {
        let bg_view_type = ButtonGroup::new();
        let rb_view_type_single = RadioButton::new_string_enumerated_type_button_group(
            Some("Single frame"),
            Some(EnumeratedTypeRef::new(ViewType::SingleView)),
            Some(&bg_view_type),
        );
        let rb_view_type_montage = RadioButton::new_string_enumerated_type_button_group(
            Some("Montage"),
            Some(EnumeratedTypeRef::new(ViewType::Montage)),
            Some(&bg_view_type),
        );
        let busy_status_panel = BusyStatusPanel::get_instance(manager, AxisID::Only);
        let ftf_stack = FileTextField2::get_instance(Some(manager), Some("Stack: "));
        let ftf_distortion_field =
            FileTextField2::get_instance(Some(manager), Some("Image distortion field file: "));
        // new JDialog(UIHarness.INSTANCE.getFrame(manager), NAME, true)
        let dialog = JDialog::new(NAME, true);
        dialog.set_default_close_operation(jdk::DO_NOTHING_ON_CLOSE);
        let pnl_root = JComponent::new_panel();
        let field_type = UITestFieldType::PANEL;
        let name = utilities::convert_label_to_name(Some(NAME), field_type.is_unlimited_segments());
        pnl_root.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        Rc::new_cyclic(|this| SerialSectionsStartupDialog {
            pnl_root,
            btn_ok: MultiLineButton::new_string(Some("OK")),
            btn_cancel: MultiLineButton::new_string(Some("Cancel")),
            bg_view_type,
            rb_view_type_single,
            rb_view_type_montage,
            sp_images_are_binned: LabeledSpinner::get_instance_string_int_int_int_int(
                Some("Binning: "),
                1,
                1,
                50,
                1,
            ),
            dialog_type: DialogType::SerialSectionsStartup,
            cb_mdoc_metadata_file: CheckBox::new_string(Some(
                "Extract montage piece list from .mdoc file",
            )),
            ftf_stack,
            ftf_distortion_field,
            dialog,
            axis_id,
            manager,
            busy_status_panel,
            startup_data: RefCell::new(None),
            this: this.clone(),
        })
    }

    /// Java static `getInstance(SerialSectionsManager, AxisID)`.
    pub fn get_instance(
        manager: &'static SerialSectionsManager,
        axis_id: AxisID,
    ) -> Rc<SerialSectionsStartupDialog> {
        let instance = SerialSectionsStartupDialog::new(manager, axis_id);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.ftf_stack.set_absolute_path(true);
        self.ftf_stack.set_origin_etomo_run_dir(true);
        self.ftf_stack.set_text_entry_policy(false);
        self.ftf_distortion_field.set_adjusted_field_width(175.0);
        self.ftf_distortion_field.set_absolute_path(true);
        self.ftf_distortion_field
            .set_origin_string(config_tool::get_distortion_dir(self.manager, None).as_deref());
        self.ftf_distortion_field
            .set_file_filter(Some(Rc::new(DistortionFileFilter::new())));
        self.cb_mdoc_metadata_file.set_enabled(false);
        // panels
        let pnl_stack = JComponent::new_panel();
        let pnl_view_type = JComponent::new_panel();
        let pnl_view_type_x = JComponent::new_panel();
        let pnl_image = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        let pnl_mdoc_metadata = JComponent::new_panel();
        let pnl_status = JComponent::new_panel();
        // dialog
        self.dialog.get_content_pane().add(&self.pnl_root);
        // root
        // Swing layout: BoxLayout Y_AXIS; rigid areas 10, 30, 10, 10, 30.
        self.pnl_root.add(&self.ftf_stack.get_root_panel());
        self.pnl_root.add(&pnl_view_type_x);
        self.pnl_root.add(&pnl_mdoc_metadata);
        self.pnl_root.add(&pnl_image);
        self.pnl_root.add(&pnl_buttons);
        self.pnl_root.add(&pnl_status);
        // stack
        // Swing layout: pnlStack BoxLayout X_AXIS with a 150-pixel rigid area; the
        // panel is never added to the dialog.
        let _ = pnl_stack;
        // view type - x direction
        // Swing layout: BoxLayout X_AXIS, 5-pixel rigid area, horizontal glue.
        pnl_view_type_x.add(&pnl_view_type);
        // view type
        // Swing layout: BoxLayout Y_AXIS.
        pnl_view_type.set_border_title(
            super::etched_border::EtchedBorder::new(Some(VIEW_TYPE_LABEL))
                .get_title()
                .as_deref(),
        );
        pnl_view_type.add(&self.rb_view_type_single.get_component());
        pnl_view_type.add(&self.rb_view_type_montage.get_component());
        // MdocMetadata
        // Swing layout: BoxLayout X_AXIS, 5-pixel rigid area, horizontal glue.
        pnl_mdoc_metadata.add(&self.cb_mdoc_metadata_file.get_component());
        // image
        // Swing layout: BoxLayout X_AXIS, 5-pixel rigid areas between.
        pnl_image.add(&self.ftf_distortion_field.get_root_panel());
        pnl_image.add(&self.sp_images_are_binned.get_container());
        // buttons
        // Swing layout: BoxLayout X_AXIS, a 15-pixel rigid area between.
        pnl_buttons.add(&self.btn_ok.get_component());
        pnl_buttons.add(&self.btn_cancel.get_component());
        // Status
        // Swing layout: BorderLayout, the busy status panel EAST.
        pnl_status.add(&self.busy_status_panel.get_component());
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let this = self.this.clone();
        self.pnl_root.add_mouse_listener(GenericMouseAdapter::new(
            this.clone() as Weak<dyn ContextMenu>
        ));
        self.dialog
            .add_window_listener(Rc::new(SerialSectionsStartupWindowListener {
                dialog: this.clone(),
            }));
        // SerialSectionsStartupActionListener
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(dialog) = this.upgrade() {
                dialog.action(event);
            }
        });
        self.btn_ok.add_action_listener(listener.clone());
        self.btn_cancel.add_action_listener(listener.clone());
        self.rb_view_type_single
            .add_action_listener(listener.clone());
        self.rb_view_type_montage
            .add_action_listener(listener.clone());
        self.cb_mdoc_metadata_file
            .add_action_listener(Some(listener));
        self.ftf_stack
            .add_result_listener(Some(Rc::new(RefCell::new(StartupResultListener(
                self.this.clone(),
            )))));
    }

    /// Java `display()`.
    pub fn display(&self) {
        self.dialog.pack();
        self.dialog.set_visible(true);
    }

    /// The `JDialog` (Rust-only access for `EtomoDirector`'s code after the modal
    /// `display()`; see the module comment).
    pub fn get_dialog(&self) -> Rc<JDialog> {
        self.dialog.clone()
    }

    /// Java private `validate()`.
    fn validate(&self) -> bool {
        if !dataset_tool::validate_dataset_name_input_file_component(
            self.manager,
            Some(self as &dyn crate::imod::etomo::ui::ui_component::UIComponent),
            self.axis_id,
            FileTextFieldInterface::get_file(&*self.ftf_stack).as_deref(),
            DataFileType::SerialSections,
            AxisType::SingleAxis,
        ) {
            return false;
        }
        // Upstream bug fixed in translation (SerialSectionsStartupDialog.java:225):
        // `getStack()` is null when the stack field is empty, and `validateDatasetName`
        // has already refused an empty field, so this is unreachable; a null stack is
        // invalid here instead of a NullPointerException.
        let Some(stack) = self.get_stack() else {
            return false;
        };
        let stack = stack.to_string_lossy().into_owned();
        dataset_tool::validate_view_type(
            self.get_view_type().unwrap_or(ViewType::DEFAULT),
            utilities::java_io_file_get_parent(&stack).as_deref(),
            Some(&utilities::java_io_file_get_name(&stack)),
            self.manager,
            Some(self as &dyn crate::imod::etomo::ui::ui_component::UIComponent),
            self.axis_id,
        )
    }

    /// Java `done()`.  Called when the OK button functionality completes
    /// successfully.
    pub fn done(&self) {
        self.dispose();
        let startup_data = self.startup_data.borrow().clone();
        self.manager.set_startup_data(startup_data.as_ref());
    }

    /// Java `resetSavedState()`.  Throws away the saved state of the dialog.  Called
    /// when the OK button functionality fails.
    pub fn reset_saved_state(&self) {
        *self.startup_data.borrow_mut() = None;
    }

    /// Java private `action(ActionEvent)`.
    fn action(&self, event: &ActionEvent) {
        self.update_display();

        let command = event.get_action_command();
        if command.is_some() && command == self.btn_ok.get_action_command().as_deref() {
            if !self.validate() {
                return;
            }
            if !self.save_state() {
                return;
            }
            self.manager.complete_startup(
                self.this
                    .upgrade()
                    .map(|this| this as Rc<dyn crate::imod::etomo::ui::UiComponent>),
                self.axis_id,
            );
        } else if command.is_some() && command == self.btn_cancel.get_action_command().as_deref() {
            self.reset_saved_state();
            self.dispose();
            self.manager.cancel_startup();
        }
    }

    /// Java `processResult(Object, boolean)`.
    pub fn process_result(&self, _result_origin: &dyn std::any::Any, _init: bool) {
        if self.validate_cb_mdoc_metadata_file() {
            self.cb_mdoc_metadata_file.set_enabled(true);
        } else {
            self.cb_mdoc_metadata_file.set_enabled(false);
        }
    }

    /// Java private `validateCbMdocMetadataFile()`.
    ///
    /// Upstream bug fixed in translation (SerialSectionsStartupDialog.java:278): with
    /// an empty stack field `ftfStack.getFile()` is null and the Java throws
    /// NullPointerException; there is no .mdoc to use then (false).
    fn validate_cb_mdoc_metadata_file(&self) -> bool {
        let Some(stack_file) = FileTextFieldInterface::get_file(&*self.ftf_stack) else {
            return false;
        };
        let stack_file = stack_file.to_string_lossy().into_owned();
        let mdoc_file_path = format!(
            "{}{}{}",
            utilities::java_io_file_get_absolute_path(&stack_file),
            extension::EXTENSION_DIVIDER,
            extension::CLASS.mdoc
        );

        let pl_file_name = Extension::substitute_extension(
            Some(&utilities::java_io_file_get_name(&stack_file)),
            Some(&extension::CLASS.pl),
        )
        .unwrap_or_else(|| "null".to_string());
        let pl_file_path = format!(
            "{}/{}",
            utilities::java_io_file_get_parent(&stack_file).unwrap_or_else(|| "null".to_string()),
            pl_file_name
        );

        let check_mdoc_file = PathBuf::from(mdoc_file_path);
        let checkpl_file_path = PathBuf::from(pl_file_path);

        check_mdoc_file.exists()
            && !checkpl_file_path.exists()
            && self.rb_view_type_montage.is_selected()
    }

    /// Java `getMdocMetadataFileStatus()`.
    pub fn get_mdoc_metadata_file_status(&self) -> bool {
        if self.validate_cb_mdoc_metadata_file() {
            return self.cb_mdoc_metadata_file.is_selected();
        }
        false
    }

    /// Java `getStartupData()`.
    pub fn get_startup_data(&self) -> Option<SerialSectionsStartupData> {
        self.startup_data.borrow().clone()
    }

    /// Java `getDistortionField()`.  Distortion field file or null if unavailable.
    /// If the state has been saved, use the saved value.
    pub fn get_distortion_field(&self) -> Option<PathBuf> {
        if let Some(startup_data) = self.startup_data.borrow().as_ref() {
            return startup_data.get_distortion_field().map(PathBuf::from);
        }
        if !Field::is_empty(&*self.ftf_distortion_field) {
            return FileTextFieldInterface::get_file(&*self.ftf_distortion_field);
        }
        None
    }

    /// Java `getPropertyUserDir()`.
    pub fn get_property_user_dir(&self) -> Option<String> {
        if let Some(startup_data) = self.startup_data.borrow().as_ref() {
            return startup_data
                .get_stack()
                .and_then(|stack| utilities::java_io_file_get_parent(&stack.to_string_lossy()));
        }
        if !Field::is_empty(&*self.ftf_stack) {
            return FileTextFieldInterface::get_file(&*self.ftf_stack)
                .and_then(|file| utilities::java_io_file_get_parent(&file.to_string_lossy()));
        }
        None
    }

    /// Java `getStack()`.  Stack file or null if unavailable.  If the state has been
    /// saved, use the saved value.
    pub fn get_stack(&self) -> Option<PathBuf> {
        if let Some(startup_data) = self.startup_data.borrow().as_ref() {
            return startup_data.get_stack().map(PathBuf::from);
        }
        if !Field::is_empty(&*self.ftf_stack) {
            return FileTextFieldInterface::get_file(&*self.ftf_stack);
        }
        None
    }

    /// `((RadioButton.RadioButtonModel) bgViewType.getSelection()).getEnumeratedType()`.
    /// The group always has a selection (`rbViewTypeSingle` is the enum's default and
    /// selects itself on construction).
    fn selected_view_type(&self) -> Option<EnumeratedTypeRef> {
        self.bg_view_type
            .get_selection()
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioButtonModel>()
                    .and_then(|model| model.get_enumerated_type())
            })
    }

    /// Java `getViewType()`.  If the state has been saved, use the saved value.
    pub fn get_view_type(&self) -> Option<ViewType> {
        if let Some(startup_data) = self.startup_data.borrow().as_ref() {
            return startup_data.get_view_type();
        }
        let selected = self.selected_view_type();
        Some(
            match selected
                .as_ref()
                .and_then(|selected| selected.downcast_ref::<ViewType>())
            {
                Some(view_type) => ViewType::get_instance(*view_type),
                None => ViewType::DEFAULT,
            },
        )
    }

    /// Java `getRootName()`.  Root name or null if unavailable.  If the state has
    /// been saved, use the saved value.
    pub fn get_root_name(&self) -> Option<String> {
        if let Some(startup_data) = self.startup_data.borrow().as_ref() {
            return startup_data.get_root_name();
        }
        if !Field::is_empty(&*self.ftf_stack) {
            let mut temp_startup_data = SerialSectionsStartupData::new(None, None);
            temp_startup_data
                .set_stack(FileTextFieldInterface::get_file(&*self.ftf_stack).as_deref());
            return temp_startup_data.get_root_name();
        }
        None
    }

    /// Java private `saveState()`.  Instanciates, loads, and validates serial
    /// sections startup data member variable.  Sets startup data to null if state is
    /// invalid.  Returns true if valid state.
    fn save_state(&self) -> bool {
        let mut startup_data = SerialSectionsStartupData::new(
            Field::get_quoted_label(&*self.ftf_stack).as_deref(),
            Some(&format!("'{}'", VIEW_TYPE_LABEL)),
        );
        if !Field::is_empty(&*self.ftf_stack) {
            startup_data.set_stack(FileTextFieldInterface::get_file(&*self.ftf_stack).as_deref());
        }
        startup_data.set_view_type(self.selected_view_type().as_ref());

        startup_data.set_mdoc_metadata_file_status(self.get_mdoc_metadata_file_status());

        if !Field::is_empty(&*self.ftf_distortion_field) {
            startup_data.set_distortion_file(
                FileTextFieldInterface::get_file(&*self.ftf_distortion_field).as_deref(),
            );
        }
        startup_data.set_images_are_binned(Some(self.sp_images_are_binned.get_value()));
        let error_message = startup_data.validate();
        *self.startup_data.borrow_mut() = Some(startup_data);
        if let Some(error_message) = error_message {
            let manager = self.manager;
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_ui_component_string_string_axis_id(
                    Some(manager),
                    Some(self as &dyn UIComponent),
                    &error_message,
                    "Entry Error",
                    Some(self.axis_id),
                )
            });
            self.reset_saved_state();
            return false;
        }
        true
    }

    /// Java private `dispose()`.
    fn dispose(&self) {
        self.dialog.set_visible(false);
        self.dialog.dispose();
        self.busy_status_panel.remove_listeners(self.manager);
    }

    /// Java private `windowClosing()`.
    fn window_closing(&self) {
        self.dispose();
        self.manager.cancel_startup();
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        Field::set_tool_tip_text(
            &*self.ftf_stack,
            Some(
                "Stack to be processed.  The stack location will be used as the dataset \
                 directory.",
            ),
        );
        Field::set_tool_tip_text(
            &*self.ftf_distortion_field,
            Some(shared_constants::DISTORTION_FIELD_TOOLTIP),
        );
        self.sp_images_are_binned
            .set_tool_tip_text(Some(shared_constants::IMAGES_ARE_BINNED_TOOLTIP));
        self.rb_view_type_montage
            .set_tool_tip_text_string(Some(shared_constants::VIEW_TYPE_TOOLTIP));
        self.rb_view_type_single
            .set_tool_tip_text_string(Some(shared_constants::VIEW_TYPE_TOOLTIP));
        self.btn_ok.set_tool_tip_text(Some(
            "Creates a dataset in the directory containing the serial sections stack.",
        ));
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        if Field::is_empty(&*self.ftf_stack) {
            return;
        }
        if self.rb_view_type_single.is_selected() {
            self.cb_mdoc_metadata_file.set_enabled(false);
        } else if self.validate_cb_mdoc_metadata_file() {
            self.cb_mdoc_metadata_file.set_enabled(true);
        } else {
            self.cb_mdoc_metadata_file.set_enabled(false);
        }
    }

    /// Java private final `dialogType` (no getter in the source).
    pub fn dialog_type(&self) -> DialogType {
        self.dialog_type
    }
}

impl ContextMenu for SerialSectionsStartupDialog {
    /// Java `popUpContextMenu(MouseEvent)`.  Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let _context_popup = ContextPopup::new_component_mouse_event_base_manager_axis_id_boolean(
            &self.pnl_root,
            mouse_event,
            self.manager,
            self.axis_id,
            true,
        );
    }
}

impl UIComponent for SerialSectionsStartupDialog {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`: the dialog (its content pane here).
    fn get_component(&self) -> Rc<JComponent> {
        self.dialog.get_content_pane()
    }
}

impl SwingComponent for SerialSectionsStartupDialog {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.dialog.get_content_pane()
    }
}

impl crate::imod::etomo::ui::UiComponent for SerialSectionsStartupDialog {}

/// Java private static final class `SerialSectionsStartupWindowListener implements
/// WindowListener`.
struct SerialSectionsStartupWindowListener {
    dialog: Weak<SerialSectionsStartupDialog>,
}

impl WindowListener for SerialSectionsStartupWindowListener {
    fn window_closing(&self) {
        if let Some(dialog) = self.dialog.upgrade() {
            dialog.window_closing();
        }
    }
}
