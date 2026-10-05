//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesDirectiveRow.java`.
//!
//! A row associated with one directive in the Directives Editor (the advanced
//! batchruntomo dataset dialog).  An event dispatch thread object, created as
//! `Rc<Self>`.  Exactly one of the four value fields exists, chosen by the
//! directive's value type.
//!
//! Java `implements DirectiveInterface` (`setValue(boolean)`, `setValue(String)`,
//! `resetValue()`): the methods are inherent here.  The interface is only reached
//! through `DirectivesTable.RowList`'s `DirectiveMapInterface` binding, which only the
//! never-called `DirectivesDialog.setValues(BaseManager)` uses (DEAD_CODE.md); the
//! translated `DirectiveInterface` is a thread-shared trait an event dispatch thread
//! row cannot implement.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::boolean_combo_box_efield::BooleanComboBoxEfield;
use super::button_control_text_efield::ButtonControlTextEfield;
use super::combo_box_efield::ComboBoxEfield;
use super::control_target::ControlTarget;
use super::directives_dialog::DirectivesDialog;
use super::directives_row::DirectivesRow;
use super::directives_section_row::DirectivesSectionRow;
use super::ebutton::Ebutton;
use super::popup::Popup;
use super::select_file_extension::SelectFileExtension;
use super::text_efield::TextEfield;
use super::toggle_ebutton::ToggleEbutton;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{GRID_BAG_REMAINDER, GridBagConstraints, GridBagLayout, JComponent};
use crate::imod::etomo::logic::batch_tool::{self, TemplateValues};
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::directive_adaptor::DirectiveAdaptor;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_descr_choice_list::DirectiveDescrChoiceList;
use crate::imod::etomo::storage::directive_descr_element::DirectiveDescrElement;
use crate::imod::etomo::storage::directive_descr_etomo_column::DirectiveDescrEtomoColumn;
use crate::imod::etomo::storage::directive_file_interface::DirectiveFileInterface;
use crate::imod::etomo::storage::directive_value_type::DirectiveValueType;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_type::FlagType;
use crate::imod::etomo::ui::row_listener::RowListener;
use crate::imod::etomo::ui::ui_component::UIComponent;

thread_local! {
    /// Java private static `select_file_extension`, initially null: the file chooser
    /// definition shared by every file row.
    static SELECT_FILE_EXTENSION: RefCell<Option<Rc<SelectFileExtension>>> =
        const { RefCell::new(None) };
}

/// Java `final class DirectivesDirectiveRow implements DirectivesRow,
/// DirectiveInterface, RowListener`.
pub struct DirectivesDirectiveRow {
    /// Java private final `hTitle`.
    h_title: Rc<Ebutton>,
    /// Java private final `bcbValue`.
    bcb_value: Option<Rc<BooleanComboBoxEfield>>,
    /// Java private final `cbValue`.
    cb_value: Option<Rc<ComboBoxEfield>>,
    /// Java private final `tfValue`.
    tf_value: Option<Rc<TextEfield>>,
    /// Java private final `bctfValue`.
    bctf_value: Option<Rc<ButtonControlTextEfield>>,
    /// Java private final `etomoColumn`.
    #[allow(dead_code)]
    etomo_column: Option<DirectiveDescrEtomoColumn>,
    /// Java private final `valueType`.
    #[allow(dead_code)]
    value_type: DirectiveValueType,
    /// Java private final `directiveDef`.
    directive_def: Option<DirectiveDef>,
    /// Java private final `descriptionAvailable`.
    #[allow(dead_code)]
    description_available: bool,
    /// Java private final `manager`.
    #[allow(dead_code)]
    manager: &'static dyn BaseManager,
    /// Java private final `dialog`.
    dialog: Weak<DirectivesDialog>,
    /// Java private final `hEmpty`.
    h_empty: Option<Rc<Ebutton>>,
    /// Java private final `tbtnOverrideToggle`.
    tbtn_override_toggle: Option<Rc<ToggleEbutton>>,
    /// Java private final `section`.
    section: Rc<DirectivesSectionRow>,

    /// Java private `open`, initially false.
    open: Cell<bool>,
    /// Java private `showForTemplateOnly`, initially true.
    show_for_template_only: Cell<bool>,
    /// Java private `showIncludedOnly`, initially false.
    show_included_only: Cell<bool>,
    /// Java private `debug`, initially false.
    #[allow(dead_code)]
    debug: Cell<bool>,
    /// Java private `showIfSet`, initially false.
    show_if_set: Cell<bool>,
}

/// The section row as the `FlagDisplay` the value fields report to.
fn section_flag_display(section: &Rc<DirectivesSectionRow>) -> Option<Rc<dyn FlagDisplay>> {
    Some(section.clone() as Rc<dyn FlagDisplay>)
}

impl DirectivesDirectiveRow {
    /// Java private `DirectivesDirectiveRow(BaseManager, DirectivesDialog,
    /// DirectivesSectionRow, String[], boolean, boolean, DirectiveDef, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn new_line_array(
        manager: &'static dyn BaseManager,
        dialog: &Rc<DirectivesDialog>,
        section: Rc<DirectivesSectionRow>,
        line_array: Option<&[String]>,
        is_choice_list: bool,
        _dialog_mode: bool,
        directive_def: Option<DirectiveDef>,
        debug: bool,
    ) -> Rc<DirectivesDirectiveRow> {
        let etomo_column = DirectiveDescrElement::get_etomo_column_from_line_array(line_array);
        // Set the title
        let title = if DirectiveDescrElement::is_label(line_array) {
            DirectiveDescrElement::get_label_from_line_array(line_array)
        } else if let Some(directive_def) = directive_def {
            Some(directive_def.to_string())
        } else {
            DirectiveDescrElement::get_name_from_line_array(line_array)
        };
        let h_title = Ebutton::get_header_instance_string(title.as_deref());
        h_title.set_allow_flag_editable_control(false);
        let value_type = DirectiveDescrElement::get_value_type_from_line_array(line_array);
        let section_title = section.get_title();
        let h_title_display: Option<Rc<dyn FlagDisplay>> =
            Some(h_title.clone() as Rc<dyn FlagDisplay>);
        let mut bcb_value = None;
        let mut cb_value = None;
        let mut tf_value = None;
        let mut bctf_value = None;
        let mut h_empty = None;
        let mut tbtn_override_toggle = None;
        if value_type == DirectiveValueType::Boolean {
            let field = BooleanComboBoxEfield::new(title.as_deref());
            field.set_directive_def(directive_def);
            field.add_flag_display(h_title_display.clone());
            field.add_final_flag_display(section_flag_display(&section));
            bcb_value = Some(field);
            h_empty = Some(Ebutton::get_header_instance_void());
        } else if is_choice_list {
            let field = ComboBoxEfield::get_override_instance(title.as_deref(), true);
            field.set_directive_def(directive_def);
            field.add_flag_display(h_title_display.clone());
            field.add_final_flag_display(section_flag_display(&section));
            let toggle = ToggleEbutton::get_override_instance(Some(
                Rc::downgrade(&field) as Weak<dyn ControlTarget>
            ));
            field.add_flag_display(Some(toggle.clone() as Rc<dyn FlagDisplay>));
            tbtn_override_toggle = Some(toggle);
            cb_value = Some(field);
        } else if value_type == DirectiveValueType::File {
            let shared = SELECT_FILE_EXTENSION.with(|extension| extension.borrow().clone());
            let field =
                ButtonControlTextEfield::get_file_override_instance(title.as_deref(), shared, false);
            if SELECT_FILE_EXTENSION.with(|extension| extension.borrow().is_none()) {
                let select_file_extension = field.get_select_file_extension();
                SELECT_FILE_EXTENSION
                    .with(|extension| *extension.borrow_mut() = select_file_extension.clone());
                if let Some(select_file_extension) = select_file_extension {
                    select_file_extension
                        .set_alt_browsing_directory(dialog.get_browsing_directory());
                }
            }
            if directive_def == Some(DirectiveDef::DISTORT)
                || directive_def == Some(DirectiveDef::GRADIENT)
                || directive_def == Some(DirectiveDef::CTF_NOISE)
            {
                field.set_override_file_open_directory(dialog.get_calibration_dir());
                field.set_file_only(true);
                field.set_file_must_exist(true);
                field.set_flag_errors();
            }
            field.set_columns();
            field.set_directive_def(directive_def);
            field.set_location_descr(Some(&section_title));
            field.add_final_flag_display(section_flag_display(&section));
            field.add_flag_display(h_title_display.clone());
            let toggle = ToggleEbutton::get_override_instance(Some(
                Rc::downgrade(&field) as Weak<dyn ControlTarget>
            ));
            field.add_flag_display(Some(toggle.clone() as Rc<dyn FlagDisplay>));
            tbtn_override_toggle = Some(toggle);
            bctf_value = Some(field);
        } else {
            let field = TextEfield::get_override_instance(title.as_deref(), Some(value_type));
            field.set_columns();
            field.set_directive_def(directive_def);
            field.set_flag_errors();
            field.add_flag_display(h_title_display.clone());
            field.add_final_flag_display(section_flag_display(&section));
            field.set_location_descr(Some(&section_title));
            let toggle = ToggleEbutton::get_override_instance(Some(
                Rc::downgrade(&field) as Weak<dyn ControlTarget>
            ));
            field.add_flag_display(Some(toggle.clone() as Rc<dyn FlagDisplay>));
            tbtn_override_toggle = Some(toggle);
            tf_value = Some(field);
        }
        // init
        if let Some(toggle) = &tbtn_override_toggle {
            toggle.set_enabled(false);
        }
        Rc::new(DirectivesDirectiveRow {
            h_title,
            bcb_value,
            cb_value,
            tf_value,
            bctf_value,
            etomo_column,
            value_type,
            directive_def,
            description_available: true,
            manager,
            dialog: Rc::downgrade(dialog),
            h_empty,
            tbtn_override_toggle,
            section,
            open: Cell::new(false),
            show_for_template_only: Cell::new(true),
            show_included_only: Cell::new(false),
            debug: Cell::new(debug),
            show_if_set: Cell::new(false),
        })
    }

    /// Java private `DirectivesDirectiveRow(BaseManager, DirectivesDialog,
    /// DirectivesSectionRow, DirectiveAdaptor, boolean)`.
    fn new_directive(
        manager: &'static dyn BaseManager,
        dialog: &Rc<DirectivesDialog>,
        section: Rc<DirectivesSectionRow>,
        directive: &mut DirectiveAdaptor,
        _dialog_mode: bool,
    ) -> Rc<DirectivesDirectiveRow> {
        let directive_def = directive.get_directive_def();
        // Set the title (Java dereferences the directive def; the table only builds
        // a row for a recognized one).
        let title = directive_def.map(|directive_def| directive_def.to_string());
        let h_title = Ebutton::get_header_instance_string(title.as_deref());
        h_title.set_allow_flag_editable_control(false);
        let value_type = DirectiveValueType::String;
        let tf_value = TextEfield::get_override_instance(title.as_deref(), Some(value_type));
        tf_value.set_columns();
        tf_value.set_directive_def(directive_def);
        tf_value.set_flag_errors();
        tf_value.add_flag_display(Some(h_title.clone() as Rc<dyn FlagDisplay>));
        tf_value.add_final_flag_display(section_flag_display(&section));
        tf_value.set_location_descr(Some(&section.get_title()));
        let tbtn_override_toggle = ToggleEbutton::get_override_instance(Some(
            Rc::downgrade(&tf_value) as Weak<dyn ControlTarget>
        ));
        tf_value.add_flag_display(Some(tbtn_override_toggle.clone() as Rc<dyn FlagDisplay>));
        // init
        tbtn_override_toggle.set_enabled(false);
        Rc::new(DirectivesDirectiveRow {
            h_title,
            bcb_value: None,
            cb_value: None,
            tf_value: Some(tf_value),
            bctf_value: None,
            etomo_column: None,
            value_type,
            directive_def,
            description_available: false,
            manager,
            dialog: Rc::downgrade(dialog),
            h_empty: None,
            tbtn_override_toggle: Some(tbtn_override_toggle),
            section,
            open: Cell::new(false),
            show_for_template_only: Cell::new(true),
            show_included_only: Cell::new(false),
            debug: Cell::new(false),
            show_if_set: Cell::new(false),
        })
    }

    /// Java package-private static `getInstance(BaseManager, DirectivesDialog,
    /// DirectivesSectionRow, String[], JPanel, GridBagLayout, GridBagConstraints,
    /// boolean, DirectiveDef, boolean)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance_line_array(
        manager: &'static dyn BaseManager,
        dialog: &Rc<DirectivesDialog>,
        section: Rc<DirectivesSectionRow>,
        line_array: Option<&[String]>,
        dialog_mode: bool,
        directive_def: Option<DirectiveDef>,
        debug: bool,
    ) -> Rc<DirectivesDirectiveRow> {
        let choice_list = DirectiveDescrElement::get_choice_list_from_line_array(line_array);
        // Create and setup instance
        let instance = DirectivesDirectiveRow::new_line_array(
            manager,
            dialog,
            section.clone(),
            line_array,
            choice_list.is_some(),
            dialog_mode,
            directive_def,
            debug,
        );
        instance.create_panel(choice_list.as_ref());
        instance.add_listeners();
        section.add_directive(instance.clone());
        instance.set_tooltips(line_array);
        instance
    }

    /// Java package-private static `getInstance(BaseManager, DirectivesDialog,
    /// DirectivesSectionRow, DirectiveAdaptor, boolean)`.
    pub fn get_instance_directive(
        manager: &'static dyn BaseManager,
        dialog: &Rc<DirectivesDialog>,
        section: Rc<DirectivesSectionRow>,
        directive: &mut DirectiveAdaptor,
        dialog_mode: bool,
    ) -> Rc<DirectivesDirectiveRow> {
        // Create and setup instance
        let instance =
            DirectivesDirectiveRow::new_directive(manager, dialog, section.clone(), directive, dialog_mode);
        instance.create_panel(None);
        instance.add_listeners();
        section.add_directive(instance.clone());
        instance.set_tooltips(None);
        instance
    }

    fn dialog(&self) -> Option<Rc<DirectivesDialog>> {
        self.dialog.upgrade()
    }

    /// Java package-private `getDirectiveDef()`.
    pub fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def
    }

    /// Java private `createPanel(DirectiveDescrChoiceList)`.
    fn create_panel(&self, choice_list: Option<&DirectiveDescrChoiceList>) {
        // init: `hTitle.setHorizontalAlignment(SwingConstants.RIGHT)`.
        self.h_title.set_horizontal_alignment(4);
        if let (Some(cb_value), Some(choice_list)) = (&self.cb_value, choice_list) {
            cb_value.set_choice_list(choice_list);
        }
        self.row_event();
    }

    /// Java private `addListeners()`.  Listen to the checkboxes in the show panel.
    fn add_listeners(self: &Rc<Self>) {
        if let Some(dialog) = self.dialog() {
            dialog.add_row_listener(self.clone() as Rc<dyn RowListener>);
        }
    }

    /// Java public `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        if let Some(bcb_value) = &self.bcb_value {
            bcb_value.set_editable(editable);
        } else if let Some(cb_value) = &self.cb_value {
            cb_value.set_editable(editable);
        } else if let Some(bctf_value) = &self.bctf_value {
            bctf_value.set_editable(editable);
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.set_editable(editable);
        }
    }

    /// Java package-private `saveAutodoc(WritableAutodoc, boolean, FieldDisplayer,
    /// Map<DirectiveDef, String>, boolean) throws FieldValidationFailedException`.
    pub fn save_autodoc(
        &self,
        autodoc: *mut Autodoc,
        do_validation: bool,
        field_displayer: Option<&dyn FieldDisplayer>,
        template_values: Option<&TemplateValues>,
        validate_only: bool,
    ) -> Result<(), FieldValidationFailedException> {
        if validate_only && !do_validation {
            return Ok(());
        }
        let override_available = self.tbtn_override_toggle.is_some();
        let override_ = override_available
            && self
                .tbtn_override_toggle
                .as_ref()
                .is_some_and(|toggle| toggle.is_selected());
        if let Some(bcb_value) = &self.bcb_value {
            batch_tool::save_text_to_autodoc_efield(
                Some(&**bcb_value),
                bcb_value.get_text().as_deref(),
                autodoc,
                template_values,
                validate_only,
            )?;
        } else if let Some(cb_value) = &self.cb_value {
            batch_tool::save_text_to_autodoc_efield_override(
                &**cb_value,
                cb_value.get_text().as_deref(),
                override_available,
                override_,
                autodoc,
                template_values,
                validate_only,
            )?;
        } else if let Some(bctf_value) = &self.bctf_value {
            let text = bctf_value.get_text_boolean_field_displayer_field_displayer(
                do_validation,
                field_displayer,
                Some(&*self.section as &dyn FieldDisplayer),
            )?;
            batch_tool::save_text_to_autodoc_efield_override(
                &**bctf_value,
                text.as_deref(),
                override_available,
                override_,
                autodoc,
                template_values,
                validate_only,
            )?;
        } else if let Some(tf_value) = &self.tf_value {
            let text = tf_value.get_text_boolean_field_displayer_field_displayer(
                do_validation,
                field_displayer,
                Some(&*self.section as &dyn FieldDisplayer),
            )?;
            batch_tool::save_text_to_autodoc_efield_override(
                &**tf_value,
                text.as_deref(),
                override_available,
                override_,
                autodoc,
                template_values,
                validate_only,
            )?;
        }
        Ok(())
    }

    /// Java package-private `validateMutuallyExclusive(DirectivesDirectiveRow,
    /// DirectivesDirectiveRow, String, FieldDisplayer)`.  Compares the rows, displays
    /// this row, and pops up errMsg; returns false if invalid.
    pub fn validate_mutually_exclusive_rows(
        &self,
        row1: Option<&Rc<DirectivesDirectiveRow>>,
        row2: Option<&Rc<DirectivesDirectiveRow>>,
        err_msg: &str,
        field_displayer: Option<Rc<dyn FieldDisplayer>>,
    ) -> bool {
        // Find out how many of the mutually exclusive fields are set.
        let nfields_set = (if self.is_set() { 1 } else { 0 })
            + (if row1.is_some_and(|row| row.is_set()) { 1 } else { 0 })
            + (if row2.is_some_and(|row| row.is_set()) { 1 } else { 0 });
        if nfields_set > 1 {
            let ui_component = self.get_ui_component();
            Popup::get_error_instance(
                ui_component.as_deref(),
                Some("Mutually Exclusive Fields"),
                Some(err_msg),
                field_displayer,
                Some(self.section.clone() as Rc<dyn FieldDisplayer>),
            )
            .open();
            return false;
        }
        true
    }

    /// Java package-private `validateMutuallyExclusive(boolean, String,
    /// FieldDisplayer)`.
    pub fn validate_mutually_exclusive_set(
        &self,
        field_set: bool,
        err_msg: &str,
        field_displayer: Option<Rc<dyn FieldDisplayer>>,
    ) -> bool {
        if !self.is_empty() && field_set {
            let ui_component = self.get_ui_component();
            Popup::get_error_instance(
                ui_component.as_deref(),
                Some("Mutually Exclusive Fields"),
                Some(err_msg),
                field_displayer,
                Some(self.section.clone() as Rc<dyn FieldDisplayer>),
            )
            .open();
            return false;
        }
        true
    }

    /// Java `setValue(boolean)` (DirectiveInterface).
    pub fn set_value_boolean(&self, value: bool) {
        if let Some(bcb_value) = &self.bcb_value {
            bcb_value.set_selected(value);
        }
    }

    /// Java `setValue(String)` (DirectiveInterface).
    pub fn set_value_string(&self, value: Option<&str>) {
        if let Some(cb_value) = &self.cb_value {
            cb_value.set_text_string(value);
        } else if let Some(bctf_value) = &self.bctf_value {
            bctf_value.set_text_string(value);
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.set_text_string(value);
        }
    }

    /// Java package-private `setValue(DirectiveFileInterface, boolean, Map<DirectiveDef,
    /// String>)`.
    pub fn set_value(
        &self,
        directive_files: &dyn DirectiveFileInterface,
        set_field_highlight_value: bool,
        template_values: Option<&mut TemplateValues>,
    ) {
        let directive_value = if let Some(bcb_value) = &self.bcb_value {
            batch_tool::set_text_value_efield(
                Some(&**bcb_value),
                directive_files,
                set_field_highlight_value,
                template_values,
                true,
            )
        } else if let Some(cb_value) = &self.cb_value {
            batch_tool::set_text_value_efield(
                Some(&**cb_value),
                directive_files,
                set_field_highlight_value,
                template_values,
                true,
            )
        } else if let Some(bctf_value) = &self.bctf_value {
            batch_tool::set_text_value_efield(
                Some(&**bctf_value),
                directive_files,
                set_field_highlight_value,
                template_values,
                true,
            )
        } else if let Some(tf_value) = &self.tf_value {
            batch_tool::set_text_value_efield(
                Some(&**tf_value),
                directive_files,
                set_field_highlight_value,
                template_values,
                true,
            )
        } else {
            None
        };
        // Handle overrides
        let (Some(toggle), Some(directive_value)) = (&self.tbtn_override_toggle, directive_value)
        else {
            return;
        };
        // Never disable the override toggle. Its too complex to keep track of it.
        let override_ = directive_value.is_override();
        // Allow override of template values
        if set_field_highlight_value && !override_ {
            toggle.set_enabled(true);
        } else if directive_value.is_batch() {
            // Set override from batch file
            if override_ {
                toggle.set_enabled(true);
            }
            if toggle.is_enabled() {
                toggle.set_selected(override_);
            }
        }
    }

    /// Java public `getUIComponent()`.
    pub fn get_ui_component(&self) -> Option<Rc<dyn UIComponent>> {
        if let Some(bcb_value) = &self.bcb_value {
            return Some(bcb_value.clone() as Rc<dyn UIComponent>);
        }
        if let Some(cb_value) = &self.cb_value {
            return Some(cb_value.clone() as Rc<dyn UIComponent>);
        }
        if let Some(bctf_value) = &self.bctf_value {
            return Some(bctf_value.clone() as Rc<dyn UIComponent>);
        }
        if let Some(tf_value) = &self.tf_value {
            return Some(tf_value.clone() as Rc<dyn UIComponent>);
        }
        None
    }

    /// Java `resetValue()` (DirectiveInterface).
    pub fn reset_value(&self) {
        self.clear();
    }

    /// Java package-private `restoreFromBackup()`.
    pub fn restore_from_backup(&self) {
        if let Some(bcb_value) = &self.bcb_value {
            bcb_value.restore_from_backup();
        } else if let Some(cb_value) = &self.cb_value {
            cb_value.restore_from_backup();
        } else if let Some(bctf_value) = &self.bctf_value {
            bctf_value.restore_from_backup();
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.restore_from_backup();
        }
    }

    /// Java package-private `isDifferentFromCheckpoint(boolean)`.
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if let Some(bcb_value) = &self.bcb_value {
            return bcb_value.is_different_from_checkpoint(always_check);
        }
        if let Some(cb_value) = &self.cb_value {
            return cb_value.is_different_from_checkpoint(always_check);
        }
        if let Some(bctf_value) = &self.bctf_value {
            return bctf_value.is_different_from_checkpoint(always_check);
        }
        if let Some(tf_value) = &self.tf_value {
            return tf_value.is_different_from_checkpoint(always_check);
        }
        false
    }

    /// Java package-private `getFlagType()`.
    pub fn get_flag_type(&self) -> Option<&'static FlagType> {
        if let Some(bcb_value) = &self.bcb_value {
            return bcb_value.get_flag_type();
        }
        if let Some(cb_value) = &self.cb_value {
            return cb_value.get_flag_type();
        }
        if let Some(bctf_value) = &self.bctf_value {
            return bctf_value.get_flag_type();
        }
        if let Some(tf_value) = &self.tf_value {
            return tf_value.get_flag_type();
        }
        None
    }

    /// Java package-private `clearValue()`.
    pub fn clear_value(&self) {
        self.clear();
    }

    /// Java private `isEmpty()`.
    fn is_empty(&self) -> bool {
        if let Some(bcb_value) = &self.bcb_value {
            return bcb_value.is_empty();
        }
        if let Some(cb_value) = &self.cb_value {
            return cb_value.is_empty();
        }
        if let Some(bctf_value) = &self.bctf_value {
            return bctf_value.is_empty();
        }
        if let Some(tf_value) = &self.tf_value {
            return tf_value.is_empty();
        }
        true
    }

    /// Java private `isSet()`.
    fn is_set(&self) -> bool {
        if let Some(bcb_value) = &self.bcb_value {
            return !bcb_value.is_empty() && !bcb_value.is_override() && bcb_value.is_selected();
        }
        if let Some(cb_value) = &self.cb_value {
            return !cb_value.is_empty() && !cb_value.is_override();
        }
        if let Some(bctf_value) = &self.bctf_value {
            return !bctf_value.is_empty() && !bctf_value.is_override();
        }
        if let Some(tf_value) = &self.tf_value {
            return !tf_value.is_empty() && !tf_value.is_override();
        }
        true
    }

    /// Java package-private `getValue()`.
    pub fn get_value(&self) -> Option<String> {
        if let Some(bcb_value) = &self.bcb_value {
            return bcb_value.get_text();
        }
        if let Some(cb_value) = &self.cb_value {
            return cb_value.get_text();
        }
        if let Some(bctf_value) = &self.bctf_value {
            return bctf_value.get_text_void();
        }
        if let Some(tf_value) = &self.tf_value {
            return tf_value.get_text_void();
        }
        None
    }

    /// Java private `isTemplateValue()`.
    fn is_template_value(&self) -> bool {
        if let Some(bcb_value) = &self.bcb_value {
            return bcb_value.is_template_value();
        }
        if let Some(cb_value) = &self.cb_value {
            return cb_value.is_template_value();
        }
        if let Some(bctf_value) = &self.bctf_value {
            return bctf_value.is_template_value();
        }
        if let Some(tf_value) = &self.tf_value {
            return tf_value.is_template_value();
        }
        false
    }

    /// Java private `isOverride()`.
    fn is_override(&self) -> bool {
        if let Some(bcb_value) = &self.bcb_value {
            return bcb_value.is_override();
        }
        if let Some(cb_value) = &self.cb_value {
            return cb_value.is_override();
        }
        if let Some(bctf_value) = &self.bctf_value {
            return bctf_value.is_override();
        }
        if let Some(tf_value) = &self.tf_value {
            return tf_value.is_override();
        }
        false
    }

    /// Java private `isEnabled()`.
    #[allow(dead_code)]
    fn is_enabled(&self) -> bool {
        if let Some(bcb_value) = &self.bcb_value {
            return bcb_value.is_enabled();
        }
        if let Some(cb_value) = &self.cb_value {
            return cb_value.is_enabled();
        }
        if let Some(bctf_value) = &self.bctf_value {
            return bctf_value.is_enabled();
        }
        if let Some(tf_value) = &self.tf_value {
            return tf_value.is_enabled();
        }
        true
    }

    /// Java private `isEditable()`.
    #[allow(dead_code)]
    fn is_editable(&self) -> bool {
        if let Some(bcb_value) = &self.bcb_value {
            return bcb_value.is_editable();
        }
        if let Some(cb_value) = &self.cb_value {
            return cb_value.is_editable();
        }
        if let Some(bctf_value) = &self.bctf_value {
            return bctf_value.is_editable();
        }
        if let Some(tf_value) = &self.tf_value {
            return tf_value.is_editable();
        }
        true
    }

    /// Java package-private `checkpoint()`.
    pub fn checkpoint(&self) {
        if let Some(bcb_value) = &self.bcb_value {
            bcb_value.checkpoint();
        } else if let Some(cb_value) = &self.cb_value {
            cb_value.checkpoint();
        } else if let Some(bctf_value) = &self.bctf_value {
            bctf_value.checkpoint();
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.checkpoint();
        }
    }

    /// Java package-private `clearTemplateValue()`.
    pub fn clear_template_value(&self) {
        if let Some(bcb_value) = &self.bcb_value {
            bcb_value.clear_template_value();
        } else if let Some(cb_value) = &self.cb_value {
            cb_value.clear_template_value();
        } else if let Some(bctf_value) = &self.bctf_value {
            bctf_value.clear_template_value();
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.clear_template_value();
        }
    }

    /// Java package-private `clear()`.
    pub fn clear(&self) {
        if let Some(bcb_value) = &self.bcb_value {
            bcb_value.clear();
        } else if let Some(cb_value) = &self.cb_value {
            cb_value.clear();
        } else if let Some(bctf_value) = &self.bctf_value {
            bctf_value.clear();
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.clear();
        }
    }

    /// Java package-private `backup()`.
    pub fn backup(&self) {
        if let Some(bcb_value) = &self.bcb_value {
            bcb_value.backup();
        } else if let Some(cb_value) = &self.cb_value {
            cb_value.backup();
        } else if let Some(bctf_value) = &self.bctf_value {
            bctf_value.backup();
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.backup();
        }
    }

    /// Java package-private `canDisplay()`.  Returns true if the row is allowed by the
    /// show panel.
    pub fn can_display(&self) -> bool {
        let show_included_only = self.show_included_only.get();
        let show_if_set = self.show_if_set.get();
        if show_included_only || show_if_set {
            let set = !self.is_empty() || self.is_override();
            if show_included_only && set && !self.is_template_value() {
                return true;
            }
            if show_if_set && set && !show_included_only {
                return true;
            }
            return false;
        }
        true
    }

    /// Java private `updateVisible()`.
    fn update_visible(&self) {
        let visible = self.open.get() && self.can_display();
        self.h_title.set_visible(visible);
        if let Some(bcb_value) = &self.bcb_value {
            bcb_value.set_visible(visible);
        } else if let Some(cb_value) = &self.cb_value {
            cb_value.set_visible(visible);
        } else if let Some(bctf_value) = &self.bctf_value {
            bctf_value.set_visible(visible);
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.set_visible(visible);
        }
        if let Some(h_empty) = &self.h_empty {
            h_empty.set_visible(visible);
        }
        if let Some(toggle) = &self.tbtn_override_toggle {
            toggle.set_visible(visible);
        }
    }

    /// Java package-private `setOpen(boolean)`.
    pub fn set_open(&self, open: bool) {
        self.open.set(open);
        self.update_visible();
    }

    /// Java private `setTooltips(String[])`.
    fn set_tooltips(&self, line_array: Option<&[String]>) {
        let note = DirectiveDescrElement::get_note(line_array);
        self.h_title.set_tooltip(Some(&format!(
            "{} - {} - {}{}",
            self.directive_def
                .map(|directive_def| directive_def.to_string())
                .unwrap_or_else(|| "null".to_owned()),
            DirectiveDescrElement::get_description_from_line_array(line_array)
                .unwrap_or_else(|| "null".to_owned()),
            DirectiveDescrElement::get_value_type_from_line_array(line_array),
            match note {
                Some(note) if !note.is_empty() => format!(" - {}", note),
                _ => String::new(),
            }
        )));
    }

    /// `field.add(pnlTable, layout, constraints)`: the field's `add` and the
    /// `layout.setConstraints` it makes.
    fn add_component(
        add: impl FnOnce(&Rc<JComponent>),
        component: Rc<JComponent>,
        pnl_table: &Rc<JComponent>,
        layout: &GridBagLayout,
        constraints: &GridBagConstraints,
    ) {
        add(pnl_table);
        layout.set_constraints(&component, constraints);
    }
}

impl std::fmt::Display for DirectivesDirectiveRow {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.h_title.get_text())
    }
}

impl DirectivesRow for DirectivesDirectiveRow {
    /// Java `statusChanged(BatchRunTomoStatus)`.
    fn status_changed(&self, status: Option<BatchRunTomoStatus>) {
        self.set_editable(status.is_none() || status == Some(BatchRunTomoStatus::Open));
    }

    /// Java `remove()`.
    fn remove(&self) {
        self.h_title.remove();
        if let Some(bcb_value) = &self.bcb_value {
            bcb_value.remove();
        } else if let Some(cb_value) = &self.cb_value {
            cb_value.remove();
        } else if let Some(bctf_value) = &self.bctf_value {
            bctf_value.remove();
        } else if let Some(tf_value) = &self.tf_value {
            tf_value.remove();
        }
        if let Some(h_empty) = &self.h_empty {
            h_empty.remove();
        }
        if let Some(toggle) = &self.tbtn_override_toggle {
            toggle.remove();
        }
    }

    /// Java `display(JPanel, GridBagLayout, GridBagConstraints)`.
    fn display(
        &self,
        pnl_table: &Rc<JComponent>,
        layout: &GridBagLayout,
        constraints: &mut GridBagConstraints,
    ) {
        constraints.gridwidth = 1;
        Self::add_component(
            |panel| self.h_title.add(panel),
            self.h_title.get_component(),
            pnl_table,
            layout,
            constraints,
        );
        if let Some(bcb_value) = &self.bcb_value {
            Self::add_component(
                |panel| bcb_value.add(panel),
                bcb_value.get_component(),
                pnl_table,
                layout,
                constraints,
            );
        } else if let Some(cb_value) = &self.cb_value {
            Self::add_component(
                |panel| cb_value.add(panel),
                cb_value.get_component(),
                pnl_table,
                layout,
                constraints,
            );
        } else if let Some(bctf_value) = &self.bctf_value {
            Self::add_component(
                |panel| bctf_value.add(panel),
                bctf_value.get_component(),
                pnl_table,
                layout,
                constraints,
            );
        } else if let Some(tf_value) = &self.tf_value {
            Self::add_component(
                |panel| tf_value.add(panel),
                tf_value.get_component(),
                pnl_table,
                layout,
                constraints,
            );
        }
        constraints.gridwidth = GRID_BAG_REMAINDER;
        if let Some(toggle) = &self.tbtn_override_toggle {
            Self::add_component(
                |panel| toggle.add(panel),
                toggle.get_component(),
                pnl_table,
                layout,
                constraints,
            );
        } else if let Some(h_empty) = &self.h_empty {
            Self::add_component(
                |panel| h_empty.add(panel),
                h_empty.get_component(),
                pnl_table,
                layout,
                constraints,
            );
        }
    }
}

impl RowListener for DirectivesDirectiveRow {
    /// Java `rowEvent()`.
    fn row_event(&self) {
        if let Some(dialog) = self.dialog() {
            self.show_for_template_only
                .set(dialog.is_show_for_template_only());
            self.show_included_only.set(dialog.is_show_included_only());
            self.show_if_set.set(dialog.is_show_if_set());
        }
        self.update_visible();
    }
}
