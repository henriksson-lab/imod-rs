//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesDirectiveRow.java`.
//!
//! The owning table, dialog, section row, directive storage, autodoc writer,
//! validation pop-up and Swing layout are neighbouring source units.  This
//! module preserves the row's construction branches and forwarding order, and
//! keeps each of those calls at an explicit boundary.
#![allow(dead_code)]

use std::collections::HashMap;

use crate::imod::etomo::base_manager::BaseManager;

use super::appearance_extension::FlagType;
use super::batch_run_tomo_step_panel::BatchRunTomoStatus;
use super::cell::{CellGridBagConstraintsBoundary, CellGridBagLayoutBoundary, CellPanelBoundary};
use super::directives_row::DirectivesRow;
use super::toggle_ebutton::ToggleEbutton;

/// Java `DirectiveValueType` as read from a directive-description row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectiveValueType {
    Boolean,
    File,
    String,
    Integer,
    FloatingPoint,
}

/// Java `DirectiveDef`, whose storage identity is used by this row.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DirectiveDef(pub String);
impl std::fmt::Display for DirectiveDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Java `DirectiveValue` returned by `BatchTool.setTextValue`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectiveValue {
    pub override_value: bool,
    pub batch: bool,
}

/// Calls from this row to the real `DirectivesDialog` source unit.
pub trait DirectivesDialogBoundary {
    fn is_show_for_template_only(&self) -> bool;
    fn is_show_included_only(&self) -> bool;
    fn is_show_if_set(&self) -> bool;
    fn add_row_listener(&mut self);
}

/// Calls from this row to the real `DirectivesSectionRow` source unit.
pub trait DirectivesSectionRowBoundary {
    fn get_title(&self) -> String;
    fn add_directive(&mut self);
}

/// Java `Ebutton` state reached by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EbuttonBoundary {
    pub text: String,
    pub visible: bool,
    pub enabled: bool,
    pub selected: bool,
    pub tooltip: Option<String>,
    pub horizontal_alignment_right: bool,
    pub removed: bool,
}
impl EbuttonBoundary {
    pub fn get_header_instance(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            visible: true,
            enabled: true,
            ..Default::default()
        }
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
    pub fn remove(&mut self) {
        self.removed = true;
    }
}

/// Source-visible common field operations.  The concrete combo/text/file
/// widgets remain their own Java source units; this state records exactly the
/// messages `DirectivesDirectiveRow` sends to one of them.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectiveFieldBoundary {
    pub text: String,
    pub empty: bool,
    pub selected: bool,
    pub override_value: bool,
    pub template_value: bool,
    pub enabled: bool,
    pub editable: bool,
    pub visible: bool,
    pub backup: Option<String>,
    pub checkpoint: Option<String>,
    pub removed: bool,
    pub directive_def: Option<DirectiveDef>,
    pub location_descr: Option<String>,
    pub flag_errors: bool,
    /// Java `getFlagType()` from the field's appearance extension.
    pub flag_type: Option<FlagType>,
}
impl DirectiveFieldBoundary {
    fn new() -> Self {
        Self {
            empty: true,
            enabled: true,
            editable: true,
            visible: true,
            ..Default::default()
        }
    }
    fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.empty = self.text.is_empty();
    }
    fn clear(&mut self) {
        self.text.clear();
        self.empty = true;
    }
    fn backup(&mut self) {
        self.backup = Some(self.text.clone());
    }
    fn checkpoint(&mut self) {
        self.checkpoint = Some(self.text.clone());
    }
    fn restore_from_backup(&mut self) {
        if let Some(text) = self.backup.clone() {
            self.set_text(text);
        }
    }
}

/// Java final `DirectivesDirectiveRow`.
pub struct DirectivesDirectiveRow {
    pub h_title: EbuttonBoundary,
    pub bcb_value: Option<DirectiveFieldBoundary>,
    pub cb_value: Option<DirectiveFieldBoundary>,
    pub tf_value: Option<DirectiveFieldBoundary>,
    pub bctf_value: Option<DirectiveFieldBoundary>,
    pub etomo_column: Option<bool>,
    pub value_type: DirectiveValueType,
    pub directive_def: DirectiveDef,
    pub description_available: bool,
    pub manager: Option<&'static dyn BaseManager>,
    pub h_empty: Option<EbuttonBoundary>,
    pub tbtn_override_toggle: Option<ToggleEbutton>,
    pub open: bool,
    pub show_for_template_only: bool,
    pub show_included_only: bool,
    pub debug: bool,
    pub show_if_set: bool,
}

impl DirectivesDirectiveRow {
    /// Java private description-row constructor.  `title`, `value_type`, and
    /// `is_choice_list` are the values extracted by `DirectiveDescrElement`.
    pub fn new_description(
        manager: Option<&'static dyn BaseManager>,
        title: String,
        section_title: String,
        value_type: DirectiveValueType,
        is_choice_list: bool,
        directive_def: DirectiveDef,
        debug: bool,
    ) -> Self {
        let mut h_title = EbuttonBoundary::get_header_instance(title);
        h_title.enabled = false; // Java setAllowFlagEditableControl(false) boundary.
        let mut row = Self {
            h_title,
            bcb_value: None,
            cb_value: None,
            tf_value: None,
            bctf_value: None,
            etomo_column: None,
            value_type,
            directive_def,
            description_available: true,
            manager,
            h_empty: None,
            tbtn_override_toggle: None,
            open: false,
            show_for_template_only: true,
            show_included_only: false,
            debug,
            show_if_set: false,
        };
        if value_type == DirectiveValueType::Boolean {
            row.bcb_value = Some(DirectiveFieldBoundary::new());
            row.h_empty = Some(EbuttonBoundary::get_header_instance(""));
        } else if is_choice_list {
            row.cb_value = Some(DirectiveFieldBoundary::new());
            row.tbtn_override_toggle = Some(ToggleEbutton::get_override_instance(None));
        } else if value_type == DirectiveValueType::File {
            let mut field = DirectiveFieldBoundary::new();
            field.location_descr = Some(section_title);
            row.bctf_value = Some(field);
            row.tbtn_override_toggle = Some(ToggleEbutton::get_override_instance(None));
        } else {
            let mut field = DirectiveFieldBoundary::new();
            field.location_descr = Some(section_title);
            field.flag_errors = true;
            row.tf_value = Some(field);
            row.tbtn_override_toggle = Some(ToggleEbutton::get_override_instance(None));
        }
        if let Some(toggle) = &mut row.tbtn_override_toggle {
            toggle.set_enabled(false);
        }
        row
    }

    /// Java private `DirectiveAdaptor` constructor.
    pub fn new_adaptor(
        manager: Option<&'static dyn BaseManager>,
        title: String,
        section_title: String,
        directive_def: DirectiveDef,
    ) -> Self {
        let mut row = Self::new_description(
            manager,
            title,
            section_title,
            DirectiveValueType::String,
            false,
            directive_def,
            false,
        );
        row.description_available = false;
        row
    }

    /// Java static `getInstance` post-construction sequence.
    pub fn get_instance<D: DirectivesDialogBoundary, S: DirectivesSectionRowBoundary>(
        mut row: Self,
        dialog: &mut D,
        section: Option<&mut S>,
        choice_list_present: bool,
        tooltip: String,
    ) -> Self {
        row.create_panel(choice_list_present);
        row.add_listeners(dialog);
        if let Some(section) = section {
            section.add_directive();
        }
        row.set_tooltips(tooltip);
        row
    }

    /// Java `toString()`.
    pub fn to_string_value(&self) -> String {
        self.h_title.text.clone()
    }
    pub fn get_directive_def(&self) -> &DirectiveDef {
        &self.directive_def
    }

    /// Java private `createPanel(DirectiveDescrChoiceList)`.
    pub fn create_panel(&mut self, _choice_list_present: bool) {
        self.h_title.horizontal_alignment_right = true;
        self.row_event_values(true, false, false);
    }
    /// Java private `addListeners()`.
    pub fn add_listeners<D: DirectivesDialogBoundary>(&mut self, dialog: &mut D) {
        dialog.add_row_listener();
    }
    /// Java `statusChanged(BatchRunTomoStatus)`; `None` is Java null.
    pub fn status_changed_nullable(&mut self, status: Option<BatchRunTomoStatus>) {
        self.set_editable(status.is_none_or(|value| value == BatchRunTomoStatus::Open));
    }
    pub fn set_editable(&mut self, editable: bool) {
        if let Some(v) = &mut self.bcb_value {
            v.editable = editable;
        } else if let Some(v) = &mut self.cb_value {
            v.editable = editable;
        } else if let Some(v) = &mut self.bctf_value {
            v.editable = editable;
        } else if let Some(v) = &mut self.tf_value {
            v.editable = editable;
        }
    }
    /// Java `setValue(boolean)`.
    pub fn set_value_boolean(&mut self, value: bool) {
        if let Some(v) = &mut self.bcb_value {
            v.selected = value;
            v.empty = false;
            v.text = if value { "1".into() } else { "0".into() };
        }
    }
    /// Java `setValue(String)`.
    pub fn set_value_string(&mut self, value: impl Into<String>) {
        let value = value.into();
        if let Some(v) = &mut self.cb_value {
            v.set_text(value);
        } else if let Some(v) = &mut self.bctf_value {
            v.set_text(value);
        } else if let Some(v) = &mut self.tf_value {
            v.set_text(value);
        }
    }
    /// Java `setValue(DirectiveFileInterface, boolean, Map)` after the
    /// storage `BatchTool.setTextValue` call has returned its `DirectiveValue`.
    pub fn set_value_from_directive_value(
        &mut self,
        value: Option<DirectiveValue>,
        set_field_highlight_value: bool,
    ) {
        let Some(value) = value else { return };
        let Some(toggle) = &mut self.tbtn_override_toggle else {
            return;
        };
        if set_field_highlight_value && !value.override_value {
            toggle.set_enabled(true);
        } else if value.batch {
            if value.override_value {
                toggle.set_enabled(true);
            }
            if toggle.is_enabled() {
                toggle.set_selected(value.override_value);
            }
        }
    }
    pub fn reset_value(&mut self) {
        self.clear_value();
    }
    pub fn restore_from_backup(&mut self) {
        if let Some(v) = &mut self.bcb_value {
            v.restore_from_backup();
        } else if let Some(v) = &mut self.cb_value {
            v.restore_from_backup();
        } else if let Some(v) = &mut self.bctf_value {
            v.restore_from_backup();
        } else if let Some(v) = &mut self.tf_value {
            v.restore_from_backup();
        }
    }
    pub fn is_different_from_checkpoint(&self, _always_check: bool) -> bool {
        if let Some(v) = &self.bcb_value {
            return v.checkpoint.as_ref().is_some_and(|x| x != &v.text);
        }
        if let Some(v) = &self.cb_value {
            return v.checkpoint.as_ref().is_some_and(|x| x != &v.text);
        }
        if let Some(v) = &self.bctf_value {
            return v.checkpoint.as_ref().is_some_and(|x| x != &v.text);
        }
        self.tf_value
            .as_ref()
            .is_some_and(|v| v.checkpoint.as_ref().is_some_and(|x| x != &v.text))
    }
    pub fn clear_value(&mut self) {
        if let Some(v) = &mut self.bcb_value {
            v.clear();
        } else if let Some(v) = &mut self.cb_value {
            v.clear();
        } else if let Some(v) = &mut self.bctf_value {
            v.clear();
        } else if let Some(v) = &mut self.tf_value {
            v.clear();
        }
    }
    /// Java `getFlagType()`.
    pub fn get_flag_type(&self) -> Option<FlagType> {
        if let Some(value) = &self.bcb_value {
            return value.flag_type;
        }
        if let Some(value) = &self.cb_value {
            return value.flag_type;
        }
        if let Some(value) = &self.bctf_value {
            return value.flag_type;
        }
        self.tf_value.as_ref().and_then(|value| value.flag_type)
    }
    pub fn checkpoint(&mut self) {
        if let Some(v) = &mut self.bcb_value {
            v.checkpoint();
        } else if let Some(v) = &mut self.cb_value {
            v.checkpoint();
        } else if let Some(v) = &mut self.bctf_value {
            v.checkpoint();
        } else if let Some(v) = &mut self.tf_value {
            v.checkpoint();
        }
    }
    pub fn clear_template_value(&mut self) {
        if let Some(v) = &mut self.bcb_value {
            v.template_value = false;
        } else if let Some(v) = &mut self.cb_value {
            v.template_value = false;
        } else if let Some(v) = &mut self.bctf_value {
            v.template_value = false;
        } else if let Some(v) = &mut self.tf_value {
            v.template_value = false;
        }
    }
    pub fn clear(&mut self) {
        self.clear_value();
    }
    pub fn backup(&mut self) {
        if let Some(v) = &mut self.bcb_value {
            v.backup();
        } else if let Some(v) = &mut self.cb_value {
            v.backup();
        } else if let Some(v) = &mut self.bctf_value {
            v.backup();
        } else if let Some(v) = &mut self.tf_value {
            v.backup();
        }
    }
    pub fn remove(&mut self) {
        self.h_title.remove();
        if let Some(v) = &mut self.bcb_value {
            v.removed = true;
        } else if let Some(v) = &mut self.cb_value {
            v.removed = true;
        } else if let Some(v) = &mut self.bctf_value {
            v.removed = true;
        } else if let Some(v) = &mut self.tf_value {
            v.removed = true;
        }
        if let Some(v) = &mut self.h_empty {
            v.remove();
        }
        if let Some(v) = &mut self.tbtn_override_toggle {
            v.remove();
        }
    }
    /// Java `rowEvent()` after the three dialog queries.
    pub fn row_event_values(
        &mut self,
        show_for_template_only: bool,
        show_included_only: bool,
        show_if_set: bool,
    ) {
        self.show_for_template_only = show_for_template_only;
        self.show_included_only = show_included_only;
        self.show_if_set = show_if_set;
        self.update_visible();
    }
    pub fn row_event<D: DirectivesDialogBoundary>(&mut self, dialog: &D) {
        self.row_event_values(
            dialog.is_show_for_template_only(),
            dialog.is_show_included_only(),
            dialog.is_show_if_set(),
        );
    }
    /// Java `canDisplay()` including the intentionally unused historical
    /// template-only condition.
    pub fn can_display(&self) -> bool {
        if self.show_included_only || self.show_if_set {
            let set = !self.is_empty() || self.is_override();
            return (self.show_included_only && set && !self.is_template_value())
                || (self.show_if_set && set && !self.show_included_only);
        }
        true
    }
    pub fn set_open(&mut self, open: bool) {
        self.open = open;
        self.update_visible();
    }
    pub fn get_value(&self) -> Option<String> {
        if let Some(v) = &self.bcb_value {
            Some(v.text.clone())
        } else if let Some(v) = &self.cb_value {
            Some(v.text.clone())
        } else if let Some(v) = &self.bctf_value {
            Some(v.text.clone())
        } else {
            self.tf_value.as_ref().map(|v| v.text.clone())
        }
    }
    pub fn validate_mutually_exclusive_rows(
        &self,
        row1: Option<&Self>,
        row2: Option<&Self>,
    ) -> bool {
        !(self.is_set() as u8
            + row1.is_some_and(Self::is_set) as u8
            + row2.is_some_and(Self::is_set) as u8
            > 1)
    }
    pub fn validate_mutually_exclusive_field(&self, field_set: bool) -> bool {
        self.is_empty() || !field_set
    }
    fn is_empty(&self) -> bool {
        self.field().is_none_or(|v| v.empty)
    }
    fn is_set(&self) -> bool {
        self.field().map_or(true, |v| {
            !v.empty && !v.override_value && (self.bcb_value.is_none() || v.selected)
        })
    }
    fn is_template_value(&self) -> bool {
        self.field().is_some_and(|v| v.template_value)
    }
    fn is_override(&self) -> bool {
        self.field().is_some_and(|v| v.override_value)
    }
    fn field(&self) -> Option<&DirectiveFieldBoundary> {
        self.bcb_value
            .as_ref()
            .or(self.cb_value.as_ref())
            .or(self.bctf_value.as_ref())
            .or(self.tf_value.as_ref())
    }
    /// Java private `updateVisible()`.
    fn update_visible(&mut self) {
        let visible = self.open && self.can_display();
        self.h_title.set_visible(visible);
        if let Some(v) = &mut self.bcb_value {
            v.visible = visible;
        } else if let Some(v) = &mut self.cb_value {
            v.visible = visible;
        } else if let Some(v) = &mut self.bctf_value {
            v.visible = visible;
        } else if let Some(v) = &mut self.tf_value {
            v.visible = visible;
        }
        if let Some(v) = &mut self.h_empty {
            v.set_visible(visible);
        }
        if let Some(v) = &mut self.tbtn_override_toggle {
            v.set_visible(visible);
        }
    }
    /// Java private `setTooltips(String[])`, after description-element values
    /// have been extracted by the storage unit.
    pub fn set_tooltips(&mut self, description: String) {
        self.h_title.tooltip = Some(format!("{} - {description}", self.directive_def));
    }
}

impl DirectivesRow for DirectivesDirectiveRow {
    /// Java `display(JPanel, GridBagLayout, GridBagConstraints)`.  The three
    /// Swing objects are presentation-boundary handles; the row already owns
    /// all component insertion order (title, value, then toggle/empty).
    fn display(
        &mut self,
        _pnl_table: &mut CellPanelBoundary,
        _layout: &mut CellGridBagLayoutBoundary,
        _constraints: &mut CellGridBagConstraintsBoundary,
    ) {
        // Java sets gridwidth to 1 for title/value then REMAINDER for the
        // trailing toggle or empty header.  The opaque boundary owns those
        // actual GridBag mutations.
    }

    fn remove(&mut self) {
        DirectivesDirectiveRow::remove(self);
    }

    fn status_changed(&mut self, status: BatchRunTomoStatus) {
        self.status_changed_nullable(Some(status));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn visibility_follows_open_and_included_only_value_rules() {
        let def = DirectiveDef("Test".into());
        let mut row = DirectivesDirectiveRow::new_description(
            None,
            "Test".into(),
            "Section".into(),
            DirectiveValueType::String,
            false,
            def,
            false,
        );
        row.set_open(true);
        assert!(row.h_title.visible);
        row.row_event_values(true, true, false);
        assert!(!row.h_title.visible);
        row.set_value_string("value");
        row.row_event_values(true, true, false);
        assert!(row.h_title.visible);
        row.tf_value.as_mut().unwrap().template_value = true;
        row.row_event_values(true, true, false);
        assert!(!row.h_title.visible);
    }
    #[test]
    fn override_toggle_only_changes_for_the_source_cases() {
        let mut row = DirectivesDirectiveRow::new_description(
            None,
            "Choice".into(),
            "S".into(),
            DirectiveValueType::String,
            true,
            DirectiveDef("d".into()),
            false,
        );
        row.set_value_from_directive_value(
            Some(DirectiveValue {
                override_value: false,
                batch: false,
            }),
            true,
        );
        assert!(row.tbtn_override_toggle.as_ref().unwrap().is_enabled());
        row.set_value_from_directive_value(
            Some(DirectiveValue {
                override_value: true,
                batch: true,
            }),
            false,
        );
        assert!(row.tbtn_override_toggle.as_ref().unwrap().is_selected());
    }
    #[test]
    fn constructors_preserve_the_four_java_value_widget_branches() {
        let def = DirectiveDef("d".into());
        let boolean = DirectivesDirectiveRow::new_description(
            None,
            "b".into(),
            "s".into(),
            DirectiveValueType::Boolean,
            false,
            def.clone(),
            false,
        );
        assert!(boolean.bcb_value.is_some() && boolean.h_empty.is_some());
        let file = DirectivesDirectiveRow::new_description(
            None,
            "f".into(),
            "s".into(),
            DirectiveValueType::File,
            false,
            def,
            false,
        );
        assert!(file.bctf_value.is_some() && file.tbtn_override_toggle.is_some());
    }
}
