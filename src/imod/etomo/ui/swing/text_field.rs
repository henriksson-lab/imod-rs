//! `IMOD/Etomo/src/etomo/ui/swing/TextField.java`.
//!
//! A `JTextField` wrapper that names itself (uitest `tf.` names), validates its text
//! and keeps backup, default, checkpoint and field-highlight settings.  Sizes, fonts,
//! columns and alignment are Swing layout and are not modelled (`// Swing layout:`
//! comments).  Java registers this object as the text field's `FocusListener` when a
//! field highlight is first used; [`TextField::focus_lost`] is what it runs.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::Rc;

use super::colors;
use super::swing_component::SwingComponent;
use super::tooltip_formatter;
use super::ui_utilities;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{
    Color, Dimension, FocusEvent, FocusListener, FontMetrics, JComponent,
};
use crate::imod::etomo::logic::autodoc_attribute_retriever;
use crate::imod::etomo::logic::field_validator::FieldValidator;
use crate::imod::etomo::logic::validation_set::ValidationSet;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_setting_interface::FieldSettingInterface;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::text_field_setting::TextFieldSetting;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `TextField`.
pub struct TextField {
    /// Java `textField`, a `JTextField`.
    text_field: Rc<JComponent>,
    /// Java `fieldType`.
    field_type: FieldType,
    /// Java `locationDescr`.
    location_descr: Option<String>,
    /// Java `reference`.
    reference: RefCell<Option<String>>,
    /// Java `required`.
    required: Cell<bool>,
    /// Java `fileMustExist`.
    file_must_exist: Cell<bool>,
    /// Java `origForeground`.
    orig_foreground: Cell<Option<Color>>,
    /// Java `directiveDef`.
    directive_def: Cell<Option<DirectiveDef>>,
    // Never reassign TextFieldSetting to null. If null means that they have never been
    // used, less updating is required.
    /// Java `backup`.
    backup: RefCell<Option<Rc<TextFieldSetting>>>,
    /// Java `defaultValue`.
    default_value: RefCell<Option<Rc<TextFieldSetting>>>,
    /// Java `checkpoint`.
    checkpoint: RefCell<Option<Rc<TextFieldSetting>>>,
    /// Java `fieldHighlight`.
    field_highlight: RefCell<Option<Rc<TextFieldSetting>>>,
    /// Java `fontMetrics`.
    font_metrics: RefCell<Option<FontMetrics>>,
    /// Java `fixedSize`.
    fixed_size: Cell<Option<Dimension>>,
    /// Java `debug`.
    debug: Cell<bool>,
    /// Java `tooltip`.
    tooltip: RefCell<Option<String>>,
    /// Java `alternateTooltip`.
    alternate_tooltip: RefCell<Option<String>>,
    /// Java `validationSet`.  Java keeps the caller's object (shared); Rust keeps a copy.
    validation_set: RefCell<Option<ValidationSet>>,
    /// Java `overridableFieldDisplayer1`.
    overridable_field_displayer1: RefCell<Option<Rc<dyn FieldDisplayer>>>,
    /// Java `overridableFieldDisplayer2`.
    overridable_field_displayer2: RefCell<Option<Rc<dyn FieldDisplayer>>>,
    /// Java `unformattedTooltip`.
    unformatted_tooltip: RefCell<Option<String>>,
    /// `this`, for registering this object as the text field's `FocusListener`.
    self_ref: std::rc::Weak<TextField>,
}

impl TextField {
    /// Java `TextField(FieldType, String, String)`.
    pub fn new(
        field_type: FieldType,
        reference: Option<&str>,
        location_descr: Option<&str>,
    ) -> Rc<TextField> {
        let text_field = Rc::new_cyclic(|self_ref| TextField {
            self_ref: self_ref.clone(),
            text_field: JComponent::new_text_field(),
            field_type,
            location_descr: location_descr.map(str::to_owned),
            reference: RefCell::new(reference.map(str::to_owned)),
            required: Cell::new(false),
            file_must_exist: Cell::new(false),
            orig_foreground: Cell::new(None),
            directive_def: Cell::new(None),
            backup: RefCell::new(None),
            default_value: RefCell::new(None),
            checkpoint: RefCell::new(None),
            field_highlight: RefCell::new(None),
            font_metrics: RefCell::new(None),
            fixed_size: Cell::new(None),
            debug: Cell::new(false),
            tooltip: RefCell::new(None),
            alternate_tooltip: RefCell::new(None),
            validation_set: RefCell::new(None),
            overridable_field_displayer1: RefCell::new(None),
            overridable_field_displayer2: RefCell::new(None),
            unformatted_tooltip: RefCell::new(None),
        });
        text_field.set_name(reference);
        // Swing layout: set the maximum height of the text field box to twice the font
        // size (the larger of a JLabel(reference)'s and the text field's), since it is
        // not set by default.
        text_field
    }

    /// Java `setColumns(int)`.
    pub fn set_columns_int(&self, _columns: i32) {
        // Swing layout: textField.setColumns(columns).
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.debug.set(input);
    }

    /// Java `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        if let Some(fixed_size) = self.fixed_size.get() {
            return fixed_size.width;
        }
        if self.font_metrics.borrow().is_none() {
            *self.font_metrics.borrow_mut() =
                ui_utilities::get_font_metrics_j_component(&self.text_field);
        }
        ui_utilities::get_preferred_width_string_font_metrics(
            Some(&self.text_field.get_text()),
            self.font_metrics.borrow().as_ref(),
        )
    }

    /// Java `setPreformattedTooltip(String)`.  Sets a preformatted tooltip.  Handles
    /// alternate tooltips.
    pub fn set_preformatted_tooltip(&self, tooltip: Option<&str>) {
        self.text_field.set_tool_tip_text(tooltip);
        // If the alternate tooltip is in use, store this tooltip as the main one.
        if self.alternate_tooltip.borrow().is_some() {
            *self.tooltip.borrow_mut() = tooltip.map(str::to_owned);
        }
    }

    /// Java `setAlternateTooltipText(String)`.  Stores a second tooltip.  Does not
    /// switch to the second tooltip.
    pub fn set_alternate_tooltip_text(&self, text: Option<&str>) {
        // Save the current tooltip as the main tooltip.
        if self.tooltip.borrow().is_none() {
            let cur_tooltip = self.text_field.get_tool_tip_text();
            if let Some(cur_tooltip) = cur_tooltip.filter(|tooltip| tooltip != "") {
                *self.tooltip.borrow_mut() = Some(cur_tooltip);
            }
        }
        *self.alternate_tooltip.borrow_mut() = tooltip_formatter::INSTANCE.format(text);
    }

    /// Java `switchTooltips(boolean)`.  Switch to/from the alternate tooltip.
    pub fn switch_tooltips(&self, alternate: bool) {
        if alternate {
            let alternate_tooltip = self.alternate_tooltip.borrow().clone();
            self.text_field
                .set_tool_tip_text(alternate_tooltip.as_deref());
        } else {
            let tooltip = self.tooltip.borrow().clone();
            self.text_field.set_tool_tip_text(tooltip.as_deref());
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.text_field.clone()
    }

    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&self, _alignment_x: f32) {
        // Swing layout: textField.setAlignmentX(alignmentX).
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.text_field.set_enabled(enabled);
        if enabled {
            self.update_field_highlight();
        }
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.text_field.set_editable(editable);
    }

    /// Java `isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.text_field.is_editable()
    }

    /// Java `getDocument()`.  The stand-in keeps the document listeners on the text
    /// field itself, so the text field stands for its document.
    pub fn get_document(&self) -> Rc<JComponent> {
        self.text_field.clone()
    }

    /// Java `setColumns()`.
    pub fn set_columns_void(&self) {
        // Swing layout: textField.setColumns(fieldType.getColumns()).
        let _ = self.field_type.get_columns();
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.text_field.set_visible(visible);
    }

    /// Java `checkpoint()`: saves the current text as the checkpoint.
    pub fn checkpoint_void(&self) {
        if self.checkpoint.borrow().is_none() {
            *self.checkpoint.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let text = self.get_text_void();
        let checkpoint = self.checkpoint.borrow().clone().unwrap();
        checkpoint.set_string(text.as_deref());
    }

    /// Java `checkpoint(String)`.
    pub fn checkpoint_string(&self, value: Option<&str>) {
        if self.checkpoint.borrow().is_none() {
            *self.checkpoint.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let checkpoint = self.checkpoint.borrow().clone().unwrap();
        checkpoint.set_string(value);
    }

    /// Java `resetToCheckpoint()`.
    pub fn reset_to_checkpoint(&self) {
        let checkpoint = self.checkpoint.borrow().clone();
        let Some(checkpoint) = checkpoint.filter(|checkpoint| checkpoint.is_set()) else {
            return;
        };
        self.set_text_string(checkpoint.get_value().as_deref());
    }

    /// Java `focusGained(FocusEvent)`.
    pub fn focus_gained(&self) {}

    /// Java `focusLost(FocusEvent)`.
    pub fn focus_lost(&self) {
        self.update_field_highlight();
    }

    /// Java `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&self, listener: FocusListener) {
        self.text_field.add_focus_listener(listener);
    }

    /// Java `removeFocusListener(FocusListener)` on the text field.
    pub fn remove_focus_listener(&self, listener: &FocusListener) {
        self.text_field.remove_focus_listener(listener);
    }

    /// `this` as a `FocusListener` (Java `textField.addFocusListener(this)`).
    fn this_focus_listener(&self) -> FocusListener {
        let this = self.self_ref.clone();
        Rc::new(move |event: &FocusEvent| {
            if let Some(this) = this.upgrade() {
                if event.gained {
                    this.focus_gained();
                } else {
                    this.focus_lost();
                }
            }
        })
    }

    /// Java `updateFieldHighlight()`.  If the field highlight is in use, use the field
    /// highlight color on the foreground of the text field if the value of the text
    /// field equals the field highlight value.  Save the original foreground.
    /// Otherwise try to restore the original foreground.
    pub fn update_field_highlight(&self) {
        // To avoid constantly updating the foreground color, assuming that
        // fieldHighlight is never reassigned to null
        let field_highlight = self.field_highlight.borrow().clone();
        let Some(field_highlight) = field_highlight else {
            return;
        };
        if !self.text_field.is_enabled() {
            return;
        }
        if field_highlight.is_set()
            && field_highlight.equals_string(Some(&self.text_field.get_text()))
        {
            // save the original color
            if self.orig_foreground.get().is_none() {
                let mut orig_foreground = self.text_field.get_foreground();
                if orig_foreground.is_none() {
                    // Color.BLACK
                    orig_foreground = Some((0, 0, 0));
                }
                self.orig_foreground.set(orig_foreground);
            }
            self.text_field
                .set_foreground(Some(colors::FIELD_HIGHLIGHT));
        } else if let Some(orig_foreground) = self.orig_foreground.get() {
            // field highlight has been turned off or field highlight doesn't match
            self.text_field.set_foreground(Some(orig_foreground));
        }
    }

    /// Java `setReference(String)`.
    pub fn set_reference(&self, reference: Option<&str>) {
        *self.reference.borrow_mut() = reference.map(str::to_owned);
        self.set_name(reference);
    }

    /// Java `setText(File)`.
    pub fn set_text_file(&self, file: &Path) {
        let absolute_path = std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf());
        self.text_field.set_text(&absolute_path.to_string_lossy());
        self.update_field_highlight();
    }

    /// Java `setText(String)`.
    pub fn set_text_string(&self, text: Option<&str>) {
        self.text_field.set_text(text.unwrap_or(""));
        self.update_field_highlight();
    }

    /// Java `setRequired(boolean)`.  Sets a validation setting.
    pub fn set_required(&self, required: bool) {
        self.required.set(required);
    }

    /// Java `setFileMustExist(boolean)`.
    pub fn set_file_must_exist(&self, file_must_exist: bool) {
        self.file_must_exist.set(file_must_exist);
    }

    /// Java `getText(boolean)`.
    pub fn get_text_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer_field_displayer(do_validation, None, None)
    }

    /// Java `validatePairedArrays(Field)`.  Calls two field validation for arrays that
    /// need to have a one to one correspondence.
    pub fn validate_paired_arrays(
        &self,
        field2: Option<&dyn Field>,
    ) -> Result<(), FieldValidationFailedException> {
        let Some(field2) = field2 else {
            return Ok(());
        };
        if !self.text_field.is_enabled() || !field2.is_enabled() {
            return Ok(());
        }
        FieldValidator::validate_paired_arrays(
            Some(&self.text_field.get_text()),
            Some(self),
            &self.get_description(),
            field2.get_text_void().as_deref(),
            &field2.get_description(),
        )
    }

    /// Java private `getQuotedReference()`.
    fn get_quoted_reference(&self) -> Option<String> {
        utilities::quote_label(self.reference.borrow().as_deref())
    }

    // Java `getFont()`: fonts are not modelled.

    /// Java `getMaximumSize()`.  Sizes are not modelled; zero is returned.
    pub fn get_maximum_size(&self) -> Dimension {
        // Swing layout: textField.getMaximumSize().
        Dimension {
            width: 0,
            height: 0,
        }
    }

    /// Java `setMaximumSize(Dimension)`.
    pub fn set_maximum_size(&self, _size: Dimension) {
        // Swing layout: textField.setMaximumSize(size).
    }

    /// Java `setTextPreferredWidth(double)`.
    pub fn set_text_preferred_width(&self, _min_width: f64) {
        // Swing layout: prefSize = textField.getPreferredSize();
        // prefSize.setSize(minWidth, prefSize.getHeight());
        // textField.setPreferredSize(prefSize).
    }

    /// Java `setTextPreferredSize(Dimension)`.
    pub fn set_text_preferred_size(&self, size: Dimension) {
        self.fixed_size.set(Some(size));
        // Swing layout: textField.setPreferredSize(size); textField.setMaximumSize(size).
    }

    /// Java `setSize(Dimension)`.
    pub fn set_size(&self, _size: Dimension) {
        // Swing layout: textField.setSize(size).
    }

    /// Java `setPreferredWidth(int)`.
    pub fn set_preferred_width(&self, width: i32) {
        // Swing layout: textField.getPreferredSize() is not modelled.
        let _dim = ui_utilities::calc_new_text_field_size(None, width, true);
        // Swing layout: textField.setPreferredSize(dim); textField.setMaximumSize(dim).
    }

    /// Java `getSize()`.  Sizes are not modelled; zero is returned.
    pub fn get_size(&self) -> Dimension {
        // Swing layout: textField.getSize().
        Dimension {
            width: 0,
            height: 0,
        }
    }

    /// Java `getPreferredSize()`.  Sizes are not modelled; zero is returned.
    pub fn get_preferred_size(&self) -> Dimension {
        // Swing layout: textField.getPreferredSize().
        Dimension {
            width: 0,
            height: 0,
        }
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.text_field.is_visible()
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, reference: Option<&str>) {
        let field_type = &ui_test_field_type::TEXT_FIELD;
        let name = utilities::convert_label_to_name(reference, field_type.is_unlimited_segments());
        self.text_field.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.text_field.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `setValidationSet(ValidationSet)`.
    pub fn set_validation_set(&self, input: Option<&ValidationSet>) {
        let mut validation_set = self.validation_set.borrow_mut();
        match validation_set.as_mut() {
            None => *validation_set = input.cloned(),
            Some(validation_set) => {
                if input.is_some() {
                    validation_set.copy(input);
                } else {
                    validation_set.clear();
                }
            }
        }
    }

    /// Java `setNumberMustBePositive(boolean)`.
    pub fn set_number_must_be_positive(&self, input: bool) {
        let mut validation_set = self.validation_set.borrow_mut();
        if validation_set.is_none() {
            *validation_set = Some(ValidationSet::new(Some(self.field_type), None));
        }
        validation_set
            .as_mut()
            .unwrap()
            .set_number_must_be_positive(input);
    }

    /// Java `setOverridableFieldDisplayers(FieldDisplayer, FieldDisplayer)`.
    pub fn set_overridable_field_displayers(
        &self,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) {
        *self.overridable_field_displayer1.borrow_mut() = field_displayer1;
        *self.overridable_field_displayer2.borrow_mut() = field_displayer2;
    }
}

impl Field for TextField {
    /// Java `isBoolean()`.
    fn is_boolean(&self) -> bool {
        false
    }

    /// Java `isDebug()`.
    fn is_debug(&self) -> bool {
        self.debug.get() || ARGUMENTS.lock().unwrap().is_debug()
    }

    /// Java `isText()`.
    fn is_text(&self) -> bool {
        true
    }

    /// Java `equalsSelectedStringValue(String)`.  No selectedStringValue is
    /// implemented so all non-empty values are true.
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }

    /// Java `setToolTipText(String)`.
    fn set_tool_tip_text(&self, text: Option<&str>) {
        self.set_preformatted_tooltip(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `setUnformattedTooltip(String)`.
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String> {
        *self.unformatted_tooltip.borrow_mut() = text.map(str::to_owned);
        self.unformatted_tooltip.borrow().clone()
    }

    /// Java `hasUnformattedTooltip()`.
    fn has_unformatted_tooltip(&self) -> bool {
        self.unformatted_tooltip.borrow().is_some()
    }

    /// Java synchronized `useUnformattedTooltip(String, String)`.  Use
    /// unformattedTooltip to build a tooltip, and then delete unformattedTooltip.
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>) {
        let unformatted_tooltip = self.unformatted_tooltip.borrow().clone();
        self.set_tool_tip_text(
            tooltip_formatter::INSTANCE
                .build_tooltip(unformatted_tooltip.as_deref(), param_descr, directive_descr)
                .as_deref(),
        );
        *self.unformatted_tooltip.borrow_mut() = None;
    }

    /// Java `setTooltip(Field)`.
    fn set_tooltip(&self, field: Option<&dyn Field>) {
        if let Some(field) = field {
            self.set_preformatted_tooltip(field.get_tooltip().as_deref());
        }
    }

    /// Java `getTooltip()`.  Gets the tooltip that is currently in use.
    fn get_tooltip(&self) -> Option<String> {
        self.text_field.get_tool_tip_text()
    }

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool {
        self.text_field.is_enabled()
    }

    /// Java `isDifferentFromCheckpoint(boolean)`.  When alwaysCheck is false, return
    /// false when the field is disabled or invisible.  Returns true if the text field is
    /// different from the checkpoint.
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.text_field.is_enabled() || !self.text_field.is_visible()) {
            return false;
        }
        let checkpoint = self.checkpoint.borrow().clone();
        match checkpoint {
            None => true,
            Some(checkpoint) => !checkpoint.equals_string(self.get_text_void().as_deref()),
        }
    }

    /// Java `backup()`.
    fn backup(&self) {
        if self.debug.get() {
            let checkpoint = self.checkpoint.borrow().clone();
            println!(
                "checkpoint:{}.textField:{}",
                checkpoint.map_or("null".to_string(), |checkpoint| {
                    // Object.toString(): the identity hash is not reproducible.
                    format!(
                        "etomo.ui.TextFieldSetting@{:x}",
                        Rc::as_ptr(&checkpoint) as usize
                    )
                }),
                self.text_field.get_text()
            );
        }
        if self.backup.borrow().is_none() {
            *self.backup.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let backup = self.backup.borrow().clone().unwrap();
        backup.set_string(Some(&self.text_field.get_text()));
    }

    /// Java `restoreFromBackup()`.  If the field was backed up, make the backup value
    /// the displayed value, and turn off the back up.
    fn restore_from_backup(&self) {
        let backup = self.backup.borrow().clone();
        if let Some(backup) = backup.filter(|backup| backup.is_set()) {
            self.set_text_string(backup.get_value().as_deref());
            backup.reset();
        }
    }

    /// Java `clear()`.
    fn clear(&self) {
        self.set_text_string(Some(""));
    }

    /// Java `isSelected()`.
    fn is_selected(&self) -> bool {
        false
    }

    /// Java `setDirectiveDef(DirectiveDef)`.
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>) {
        self.directive_def.set(directive_def);
    }

    /// Java `getDirectiveDef()`.
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        self.directive_def.get()
    }

    /// Java `useDefaultValue()`.
    fn use_default_value(&self) {
        let Some(directive_def) = self.directive_def.get() else {
            let default_value = self.default_value.borrow().clone();
            if let Some(default_value) =
                default_value.filter(|default_value| default_value.is_set())
            {
                default_value.reset();
            }
            return;
        };
        // only search for default value once
        if self.default_value.borrow().is_none() {
            let default_value = Rc::new(TextFieldSetting::new_field_type(self.field_type));
            let value = autodoc_attribute_retriever::INSTANCE
                .get_default_value(Some(directive_def.clone()));
            if let Some(value) = value {
                // if default value has been found, set it in the field setting
                default_value.set_string(Some(&value));
            }
            *self.default_value.borrow_mut() = Some(default_value);
        }
        let default_value = self.default_value.borrow().clone().unwrap();
        if default_value.is_set() {
            self.set_text_string(default_value.get_value().as_deref());
        }
    }

    /// Java `equalsDefaultValue()`.
    fn equals_default_value_void(&self) -> bool {
        let default_value = self.default_value.borrow().clone();
        default_value.is_some_and(|default_value| {
            default_value.is_set() && default_value.equals_string(self.get_text_void().as_deref())
        })
    }

    /// Java `equalsDefaultValue(String)`.
    fn equals_default_value_string(&self, value: Option<&str>) -> bool {
        let default_value = self.default_value.borrow().clone();
        default_value.is_some_and(|default_value| {
            default_value.is_set() && default_value.equals_string(value)
        })
    }

    /// Java `checkpoint()`.  Saves the current text as the checkpoint.
    fn checkpoint(&self) {
        self.checkpoint_void();
    }

    /// Java `getCheckpoint()` (Java returns the `TextFieldSetting` itself).
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        self.checkpoint
            .borrow()
            .clone()
            .map(|checkpoint| checkpoint as Rc<dyn FieldSettingInterface>)
    }

    /// Java `setCheckpoint(FieldSettingInterface)`.
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>) {
        if self.checkpoint.borrow().is_none()
            && input.is_some_and(|input| input.is_set() && input.is_text())
        {
            *self.checkpoint.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
        }
        let checkpoint = self.checkpoint.borrow().clone();
        if let Some(checkpoint) = checkpoint {
            checkpoint.copy(input);
        }
    }

    /// Java `isFieldHighlightSet()`.
    fn is_field_highlight_set(&self) -> bool {
        self.field_highlight
            .borrow()
            .as_ref()
            .is_some_and(|field_highlight| field_highlight.is_set())
    }

    /// Java `setFieldHighlight(FieldSettingInterface)`.
    fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    ) {
        if self.field_highlight.borrow().is_none()
            && input.is_some_and(|input| input.is_set() && input.is_text())
        {
            *self.field_highlight.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
            self.text_field
                .add_focus_listener(self.this_focus_listener());
        }
        let field_highlight = self.field_highlight.borrow().clone();
        if let Some(field_highlight) = field_highlight {
            field_highlight.copy(input);
            self.update_field_highlight();
        }
    }

    /// Java `setFieldHighlight(String)`.
    fn set_field_highlight_string(&self, value: Option<&str>) {
        if self.field_highlight.borrow().is_none() {
            *self.field_highlight.borrow_mut() =
                Some(Rc::new(TextFieldSetting::new_field_type(self.field_type)));
            self.text_field
                .add_focus_listener(self.this_focus_listener());
        }
        let field_highlight = self.field_highlight.borrow().clone().unwrap();
        field_highlight.set_string(value);
        self.update_field_highlight();
    }

    /// Java `setFieldHighlight(boolean)`.
    fn set_field_highlight_boolean(&self, _value: bool) {}

    /// Java `getFieldHighlight()` (Java returns the `TextFieldSetting` itself).
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>> {
        self.field_highlight
            .borrow()
            .clone()
            .map(|field_highlight| field_highlight as Rc<dyn FieldSettingInterface>)
    }

    /// Java `clearFieldHighlight()`.
    fn clear_field_highlight(&self) {
        let field_highlight = self.field_highlight.borrow().clone();
        if let Some(field_highlight) =
            field_highlight.filter(|field_highlight| field_highlight.is_set())
        {
            field_highlight.reset();
            self.update_field_highlight();
        }
    }

    /// Java `equalsFieldHighlight()`.
    fn equals_field_highlight_void(&self) -> bool {
        let field_highlight = self.field_highlight.borrow().clone();
        field_highlight.is_some_and(|field_highlight| {
            field_highlight.is_set()
                && field_highlight.equals_string(self.get_text_void().as_deref())
        })
    }

    /// Java `equalsFieldHighlight(String)`.
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool {
        let field_highlight = self.field_highlight.borrow().clone();
        field_highlight.is_some_and(|field_highlight| {
            field_highlight.is_set() && field_highlight.equals_string(value)
        })
    }

    /// Java `isRequired()`.
    fn is_required(&self) -> bool {
        self.required.get() && self.text_field.is_enabled()
    }

    /// Java `getText(boolean, FieldDisplayer)`.  Validates and returns text in text
    /// field.
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.get_text_boolean_field_displayer_field_displayer(do_validation, field_displayer1, None)
    }

    /// Java `getText(boolean, FieldDisplayer, FieldDisplayer)`.
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        mut field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        mut field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        let mut text = Some(self.text_field.get_text());
        if do_validation && self.text_field.is_enabled() {
            if field_displayer1.is_none() {
                field_displayer1 = self.overridable_field_displayer1.borrow().clone();
            }
            if field_displayer2.is_none() {
                field_displayer2 = self.overridable_field_displayer2.borrow().clone();
            }
            let descr = self
                .get_quoted_reference()
                .unwrap_or_else(|| "null".to_string())
                + &match &self.location_descr {
                    None => String::new(),
                    Some(location_descr) => " in ".to_string() + location_descr,
                };
            let validation_set = self.validation_set.borrow().clone();
            text = FieldValidator::validate_text_string_field_type_ui_component_string_boolean_boolean_boolean_validation_set_field_displayer_field_displayer(
                text.as_deref(),
                Some(self.field_type),
                Some(self),
                Some(&descr),
                self.required.get(),
                self.file_must_exist.get(),
                false,
                validation_set.as_ref(),
                field_displayer1.as_deref(),
                field_displayer2.as_deref(),
            )?;
        }
        Ok(text)
    }

    /// Java `getDescription()`.
    fn get_description(&self) -> String {
        self.get_quoted_reference()
            .unwrap_or_else(|| "null".to_string())
            + &match &self.location_descr {
                None => String::new(),
                Some(location_descr) => " in ".to_string() + location_descr,
            }
    }

    /// Java `setValue(Field)`.
    fn set_value_field(&self, input: Option<&dyn Field>) {
        match input {
            None => self.clear(),
            Some(input) => self.set_text_string(input.get_text_void().as_deref()),
        }
    }

    /// Java `setValue(String)`.
    fn set_value_string(&self, value: Option<&str>) {
        self.set_text_string(value);
    }

    /// Java `setValue(boolean)`.
    fn set_value_boolean(&self, _value: bool) {}

    /// Java `getText()`: return text without validation.
    fn get_text_void(&self) -> Option<String> {
        self.get_text_boolean_field_displayer_field_displayer(false, None, None)
            .unwrap_or(None)
    }

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool {
        let text = self.get_text_void();
        text.as_deref()
            .is_none_or(java_lang_string_matches_whitespace)
    }

    /// Java `getName()`.
    fn get_name(&self) -> Option<String> {
        self.text_field.get_name()
    }

    /// Java `getQuotedLabel()`.
    fn get_quoted_label(&self) -> Option<String> {
        self.get_quoted_reference()
    }
}

impl UIComponent for TextField {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.text_field.clone()
    }
}

impl SwingComponent for TextField {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.text_field.clone()
    }
}
