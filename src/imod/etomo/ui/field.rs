//! `IMOD/Etomo/src/etomo/ui/Field.java`.
//!
//! An interface to allow the generic handling of GUI fields.  Implementers are EDT
//! objects (`Rc`, `&self` methods).  Java's overloaded members carry the parameter-type
//! suffix (`setValue(Field)` -> `set_value_field`, `getText()` -> `get_text_void`).
//!
//! `getCheckpoint()` and `getFieldHighlight()` return the setting the field holds.  A
//! Rust field keeps its setting inside a cell and cannot lend it out, so the setting is
//! returned as an owned snapshot; every Java caller only reads it.
use std::rc::Rc;

use super::field_displayer::FieldDisplayer;
use super::field_setting_interface::FieldSettingInterface;
use super::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::storage::directive_def::DirectiveDef;

/// Java `Field`.
pub trait Field {
    /// Java `isDebug()`.
    fn is_debug(&self) -> bool;

    /// Java `getName()`.
    fn get_name(&self) -> Option<String>;

    /// Java `isBoolean()`.
    fn is_boolean(&self) -> bool;

    /// Java `isText()`.  Returns true if the field contains any kind of text -
    /// including spinners.
    fn is_text(&self) -> bool;

    /// Java `getQuotedLabel()`.
    fn get_quoted_label(&self) -> Option<String>;

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;

    /// Java `clear()`.  Resets value - does not change other settings.
    fn clear(&self);

    /// Java `setValue(Field)`.
    fn set_value_field(&self, from: Option<&dyn Field>);

    /// Java `setValue(String)`.  No effect in boolean-only fields.
    fn set_value_string(&self, text: Option<&str>);

    /// Java `setValue(boolean)`.  No effect in text-only fields.
    fn set_value_boolean(&self, bool_: bool);

    /// Java `isEmpty()`.
    fn is_empty(&self) -> bool;

    /// Java `isSelected()`.  True if this a binary control (toggle button, radio
    /// button, checkbox) and it is selected.
    fn is_selected(&self) -> bool;

    /// Java `isRequired()`.  True if the field must currently contain text (not
    /// required when disabled).
    fn is_required(&self) -> bool;

    /// Java `getText()`.
    fn get_text_void(&self) -> Option<String>;

    /// Java `getText(boolean, FieldDisplayer)`.
    fn get_text_boolean_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getText(boolean, FieldDisplayer, FieldDisplayer)`.
    fn get_text_boolean_field_displayer_field_displayer(
        &self,
        do_validation: bool,
        field_displayer1: Option<Rc<dyn FieldDisplayer>>,
        field_displayer2: Option<Rc<dyn FieldDisplayer>>,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getDirectiveDef()`.  (The source's comment above this member - "Returns
    /// true if there is a non-empty, non-whitespace value.  Boolean fields are never
    /// empty." - is a stray describing `isEmpty`.)
    fn get_directive_def(&self) -> Option<DirectiveDef>;

    /// Java `useDefaultValue()`.  Use DefaultFinder to find and set a default value.
    /// DefaultFinder requires DirectiveDef, and only works for comparam directives.
    fn use_default_value(&self);

    /// Java `equalsDefaultValue()`.
    fn equals_default_value_void(&self) -> bool;

    /// Java `equalsDefaultValue(String)`.
    fn equals_default_value_string(&self, value: Option<&str>) -> bool;

    /// Java `backup()`.
    fn backup(&self);

    /// Java `restoreFromBackup()`.  Set the value from the backup, and delete the
    /// backup.
    fn restore_from_backup(&self);

    /// Java `checkpoint()`.
    fn checkpoint(&self);

    /// Java `setCheckpoint(FieldSettingInterface)`.
    fn set_checkpoint(&self, input: Option<&dyn FieldSettingInterface>);

    /// Java `getCheckpoint()`.
    fn get_checkpoint(&self) -> Option<Rc<dyn FieldSettingInterface>>;

    /// Java `isDifferentFromCheckpoint(boolean)`.
    fn is_different_from_checkpoint(&self, always_check: bool) -> bool;

    /// Java `isFieldHighlightSet()`.
    fn is_field_highlight_set(&self) -> bool;

    /// Java `clearFieldHighlight()`.
    fn clear_field_highlight(&self);

    /// Java `setFieldHighlight(FieldSettingInterface)`.
    fn set_field_highlight_field_setting_interface(
        &self,
        input: Option<&dyn FieldSettingInterface>,
    );

    /// Java `setFieldHighlight(String)`.
    fn set_field_highlight_string(&self, input: Option<&str>);

    /// Java `setFieldHighlight(boolean)`.  No effect in text-only fields.
    fn set_field_highlight_boolean(&self, input: bool);

    /// Java `getFieldHighlight()`.
    fn get_field_highlight(&self) -> Option<Rc<dyn FieldSettingInterface>>;

    /// Java `equalsFieldHighlight()`.
    fn equals_field_highlight_void(&self) -> bool;

    /// Java `equalsFieldHighlight(String)`.
    fn equals_field_highlight_string(&self, value: Option<&str>) -> bool;

    /// Java `setToolTipText(String)`.
    fn set_tool_tip_text(&self, tooltip: Option<&str>);

    /// Java `setTooltip(Field)`.
    fn set_tooltip(&self, field: Option<&dyn Field>);

    /// Java `getTooltip()`.
    fn get_tooltip(&self) -> Option<String>;

    /// Java `equalsSelectedStringValue(String)`.  Compares with selectedStringValue - a
    /// value associated with the field being selected.  A boolean field make be
    /// selected when its associated directive has a specific value.  The
    /// selectedStringValue is this directive value.  If selectedStringValue is not set,
    /// it defaults to any non-empty value.
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool;

    /// Java `setDirectiveDef(DirectiveDef)`.
    fn set_directive_def(&self, directive_def: Option<DirectiveDef>);

    /// Java `getDescription()`.
    fn get_description(&self) -> String;

    /// Java `setUnformattedTooltip(String)`.  Returns the new unformattedTooltip value.
    fn set_unformatted_tooltip(&self, text: Option<&str>) -> Option<String>;

    /// Java `hasUnformattedTooltip()`.
    fn has_unformatted_tooltip(&self) -> bool;

    /// Java `useUnformattedTooltip(String, String)`.
    fn use_unformatted_tooltip(&self, param_descr: Option<&str>, directive_descr: Option<&str>);
}
