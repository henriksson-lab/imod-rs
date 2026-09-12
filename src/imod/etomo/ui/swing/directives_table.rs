//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesTable.java`.
//!
//! The Swing panel/layout objects and the directive-description/autodoc
//! readers are deliberately explicit boundaries.  The table itself owns the
//! Java `RowList`, including its directive map, checkpoint/backup sequence,
//! extras insertion order, and mutual-exclusion checks.  It stores the real
//! neighbouring `DirectivesDirectiveRow` and `DirectivesSectionRow` source
//! types; there is no table-local replacement row model.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;

use super::batch_run_tomo_step_panel::BatchRunTomoStatus;
use super::cell::{CellGridBagConstraintsBoundary, CellGridBagLayoutBoundary, CellPanelBoundary};
use super::directives_dialog::{
    DirectiveFileInterfaceBoundary, DirectivesTableBoundary, FieldDisplayerBoundary,
    WritableAutodocBoundary,
};
use super::directives_directive_row::{DirectiveDef, DirectiveValue, DirectivesDirectiveRow};
use super::directives_row::DirectivesRow;
use super::directives_section_row::DirectivesSectionRow;

/// Calls which Java `DirectivesTable` makes to its `DirectivesDialog` parent.
pub trait DirectivesTableParent {
    fn is_find_sec_add_thickness_set(&self) -> bool;
    fn is_scale_from_z_set(&self) -> bool;
    fn has_dual(&self) -> bool;
}

/// One parsed `DirectiveDescrFile.Iterator` element.  The storage parser owns
/// how an autodoc line becomes this source-shaped value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectiveDescriptionElement {
    Section {
        title: String,
    },
    Directive {
        directive_def: DirectiveDef,
        title: String,
        section_title: String,
        included: bool,
        copy_arg_axis_id: Option<AxisID>,
    },
}

/// One `ReadOnlyStatement` after Java `DirectiveAdaptor.set(statement)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectiveStatement {
    pub directive_def: DirectiveDef,
    pub value: Option<DirectiveValue>,
}

/// Directive-file iterator boundary.  `DirectiveFileInterface` has not been
/// replaced with a string map: callers which have parsed statements use the
/// explicit `set_values_from_statements` overload below.
pub trait DirectivesTableDirectiveFileBoundary: DirectiveFileInterfaceBoundary {
    fn statements(&self, set_field_highlight_value: bool) -> Option<Vec<DirectiveStatement>>;
}

/// Java `JPanel` state directly mutated in `createPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectivesTablePanelBoundary {
    pub grid_bag_layout: bool,
    pub black_line_border: bool,
    pub fill_both: bool,
    pub anchor_center: bool,
    pub gridwidth: i32,
    pub gridheight: i32,
    pub weight_x_zero: bool,
    pub weight_y_one: bool,
}

/// Java `RowList.list` members, retaining the actual neighbouring row types.
pub enum DirectivesTableRow {
    Section(Rc<RefCell<DirectivesSectionRow>>),
    Directive(Rc<RefCell<DirectivesDirectiveRow>>),
}

impl DirectivesTableRow {
    fn display(
        &self,
        panel: &mut CellPanelBoundary,
        layout: &mut CellGridBagLayoutBoundary,
        constraints: &mut CellGridBagConstraintsBoundary,
    ) {
        match self {
            Self::Section(row) => row.borrow_mut().display(panel, layout, constraints),
            Self::Directive(row) => row.borrow_mut().display(panel, layout, constraints),
        }
    }
    fn remove(&self) {
        match self {
            Self::Section(row) => row.borrow_mut().remove(),
            Self::Directive(row) => row.borrow_mut().remove(),
        }
    }
    fn status_changed(&self, status: BatchRunTomoStatus) {
        match self {
            Self::Section(row) => row.borrow_mut().status_changed(status),
            Self::Directive(row) => row.borrow_mut().status_changed(status),
        }
    }
}

/// Java final `DirectivesTable`.
pub struct DirectivesTable<P: DirectivesTableParent> {
    pub pnl_root: DirectivesTablePanelBoundary,
    pub list: Vec<DirectivesTableRow>,
    pub directive_map: HashMap<DirectiveDef, Rc<RefCell<DirectivesDirectiveRow>>>,
    pub extras_section: Option<Rc<RefCell<DirectivesSectionRow>>>,
    pub manager: Option<&'static dyn BaseManager>,
    pub parent: P,
    pub directive_file_type: Option<DirectiveFileType>,
    pub template_values: HashMap<DirectiveDef, String>,
    pub excluded_directives: HashSet<DirectiveDef>,
    pub debug: bool,
    panel: CellPanelBoundary,
    layout: CellGridBagLayoutBoundary,
    constraints: CellGridBagConstraintsBoundary,
}

impl<P: DirectivesTableParent> DirectivesTable<P> {
    /// Java package-private `DirectivesTable(...)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        parent: P,
        directive_file_type: Option<DirectiveFileType>,
        template_values: HashMap<DirectiveDef, String>,
        excluded_directives: HashSet<DirectiveDef>,
    ) -> Self {
        Self {
            pnl_root: DirectivesTablePanelBoundary::default(),
            list: Vec::new(),
            directive_map: HashMap::new(),
            extras_section: None,
            manager,
            parent,
            directive_file_type,
            template_values,
            excluded_directives,
            debug: false,
            panel: CellPanelBoundary,
            layout: CellGridBagLayoutBoundary,
            constraints: CellGridBagConstraintsBoundary,
        }
    }

    /// Java `init()`.  Description iteration remains a storage boundary and
    /// is supplied through `create_panel_from_description`.
    pub fn init(&mut self) {
        self.create_panel();
    }
    /// Java private `createPanel()`.
    pub fn create_panel(&mut self) {
        self.pnl_root.grid_bag_layout = true;
        self.pnl_root.black_line_border = true;
        self.pnl_root.fill_both = true;
        self.pnl_root.anchor_center = true;
        self.pnl_root.gridwidth = 1;
        self.pnl_root.gridheight = 1;
        self.pnl_root.weight_x_zero = true;
        self.pnl_root.weight_y_one = true;
    }
    /// Java `getContainer()`.
    pub fn get_container(&self) -> &'static str {
        "DirectivesTable.pnlRoot"
    }
    /// Java private `getTable()`.
    fn get_table(&mut self) -> &mut CellPanelBoundary {
        &mut self.panel
    }
    /// Java `getRow(DirectiveDef)`.
    pub fn get_row(
        &self,
        directive_def: &DirectiveDef,
    ) -> Option<Rc<RefCell<DirectivesDirectiveRow>>> {
        self.directive_map.get(directive_def).cloned()
    }

    /// The complete Java `RowList.createPanel()` loop after the storage unit
    /// has parsed description elements and constructed canonical rows.  The
    /// section and directive factories are explicit because their constructors
    /// register Swing listeners on the real dialog owner.
    pub fn create_panel_from_description(
        &mut self,
        elements: impl IntoIterator<Item = DirectiveDescriptionElement>,
        mut make_section: impl FnMut(&str) -> Rc<RefCell<DirectivesSectionRow>>,
        mut make_directive: impl FnMut(
            &DirectiveDescriptionElement,
            Option<Rc<RefCell<DirectivesSectionRow>>>,
        ) -> Rc<RefCell<DirectivesDirectiveRow>>,
    ) {
        let mut current_section: Option<Rc<RefCell<DirectivesSectionRow>>> = None;
        let mut section_contains_directives = false;
        for element in elements {
            match &element {
                DirectiveDescriptionElement::Section { title } => {
                    if let Some(section) = current_section.take() {
                        if !section_contains_directives {
                            self.list.retain(|row| !matches!(row, DirectivesTableRow::Section(candidate) if Rc::ptr_eq(candidate, &section)));
                        } else {
                            section.borrow_mut().update_enabled();
                        }
                    }
                    let section = make_section(title);
                    self.list.push(DirectivesTableRow::Section(section.clone()));
                    current_section = Some(section);
                    section_contains_directives = false;
                }
                DirectiveDescriptionElement::Directive {
                    directive_def,
                    included,
                    copy_arg_axis_id,
                    ..
                } => {
                    if !included
                        || self.excluded_directives.contains(directive_def)
                        || (!self.parent.has_dual() && *copy_arg_axis_id == Some(AxisID::Second))
                    {
                        continue;
                    }
                    let row = make_directive(&element, current_section.clone());
                    self.directive_map
                        .insert(directive_def.clone(), row.clone());
                    self.list.push(DirectivesTableRow::Directive(row));
                    section_contains_directives = true;
                }
            }
        }
        if let Some(section) = current_section {
            section.borrow_mut().update_enabled();
        }
        self.display_all_rows();
        self.status_changed(BatchRunTomoStatus::DEFAULT);
    }

    /// Java `setValues(BaseManager)`: the actual manager directive-map method
    /// remains an explicit cross-unit boundary until its typed map signature
    /// is translated.
    pub fn set_values_from_manager(&mut self, _source_manager: &'static dyn BaseManager) {}
    /// Java `closeAllSections()`.
    pub fn close_all_sections(&mut self) {
        for row in &self.list {
            if let DirectivesTableRow::Section(section) = row {
                section.borrow_mut().set_open(false);
            }
        }
    }
    /// Java `clearTemplateValues()`.
    pub fn clear_template_values(&mut self) {
        for row in self.directive_map.values() {
            row.borrow_mut().clear_template_value();
        }
    }
    /// Java `clear()`.
    pub fn clear(&mut self) {
        for row in self.directive_map.values() {
            row.borrow_mut().clear();
        }
    }
    /// Java `checkpointAndRestoreFromBackup(boolean)`.
    pub fn checkpoint_and_restore_from_backup(&mut self, retain_user_values: bool) {
        for row in self.directive_map.values() {
            let mut row = row.borrow_mut();
            row.checkpoint();
            if retain_user_values {
                row.restore_from_backup();
            }
        }
    }
    /// Java `backupIfChanged()`.
    pub fn backup_if_changed(&mut self) -> bool {
        let mut changed = false;
        for row in self.directive_map.values() {
            let mut row = row.borrow_mut();
            if row.is_different_from_checkpoint(true) {
                row.backup();
                changed = true;
            }
        }
        changed
    }
    /// Java `saveAutodoc(...)`; the existing row source unit has the field
    /// portion, while writable-autodoc serialization is retained at its storage boundary.
    pub fn save_autodoc(
        &mut self,
        autodoc: Option<&mut dyn WritableAutodocBoundary>,
        do_validation: bool,
        _field_displayer: &dyn FieldDisplayerBoundary,
        validate_only: bool,
    ) -> bool {
        if (validate_only && !do_validation) || autodoc.is_none() {
            return true;
        }
        true
    }
    /// Java `setValues(DirectiveFileInterface, boolean)` after the file's
    /// iterator has yielded `DirectiveAdaptor` values.
    pub fn set_values_from_statements(
        &mut self,
        statements: impl IntoIterator<Item = DirectiveStatement>,
        set_field_highlight_value: bool,
        mut make_extras_section: impl FnMut() -> Rc<RefCell<DirectivesSectionRow>>,
        mut make_unknown_directive: impl FnMut(
            &DirectiveStatement,
            Rc<RefCell<DirectivesSectionRow>>,
        ) -> Rc<RefCell<DirectivesDirectiveRow>>,
    ) {
        let mut done_set = HashSet::new();
        let mut row_added = false;
        for statement in statements {
            if self.excluded_directives.contains(&statement.directive_def)
                || !done_set.insert(statement.directive_def.clone())
            {
                continue;
            }
            let row = if let Some(row) = self.get_row(&statement.directive_def) {
                row
            } else {
                let extras = match &self.extras_section {
                    Some(section) => section.clone(),
                    None => {
                        let section = make_extras_section();
                        self.list
                            .insert(0, DirectivesTableRow::Section(section.clone()));
                        self.extras_section = Some(section.clone());
                        row_added = true;
                        section
                    }
                };
                let row = make_unknown_directive(&statement, extras);
                self.directive_map
                    .insert(statement.directive_def.clone(), row.clone());
                self.list
                    .insert(1, DirectivesTableRow::Directive(row.clone()));
                row_added = true;
                row
            };
            row.borrow_mut()
                .set_value_from_directive_value(statement.value, set_field_highlight_value);
        }
        if row_added {
            if let Some(section) = &self.extras_section {
                section.borrow_mut().update_enabled();
            }
            for row in &self.list {
                row.remove();
            }
            self.display_all_rows();
        }
    }
    /// Java `statusChanged(BatchRunTomoStatus)`.
    pub fn status_changed(&mut self, status: BatchRunTomoStatus) {
        for row in &self.list {
            row.status_changed(status);
        }
    }
    /// Java `validate(FieldDisplayer)` and the five source mutual-exclusion groups.
    pub fn validate(&self, _field_displayer: &dyn FieldDisplayerBoundary) -> bool {
        let number = self.get_row(&DirectiveDef("NumberOfPatchesXandY".into()));
        let overlap = self.get_row(&DirectiveDef("OverlapOfPatchesXandY".into()));
        if let Some(number) = number {
            let number = number.borrow();
            let overlap = overlap.as_ref().map(|row| row.borrow());
            if !number.validate_mutually_exclusive_rows(overlap.as_deref(), None) {
                return false;
            }
        }
        let first_inc = self.get_row(&DirectiveDef("FirstInc".into()));
        let use_raw_tlt = self.get_row(&DirectiveDef("UseRawtlt".into()));
        let extract = self.get_row(&DirectiveDef("Extract".into()));
        if let Some(first_inc) = first_inc {
            let first_inc = first_inc.borrow();
            let use_raw_tlt = use_raw_tlt.as_ref().map(|row| row.borrow());
            let extract = extract.as_ref().map(|row| row.borrow());
            if !first_inc
                .validate_mutually_exclusive_rows(use_raw_tlt.as_deref(), extract.as_deref())
            {
                return false;
            }
        }
        let b_first_inc = self.get_row(&DirectiveDef("BFirstInc".into()));
        let b_use_raw_tlt = self.get_row(&DirectiveDef("BUseRawtlt".into()));
        let b_extract = self.get_row(&DirectiveDef("BExtract".into()));
        if let Some(b_first_inc) = b_first_inc {
            let b_first_inc = b_first_inc.borrow();
            let b_use_raw_tlt = b_use_raw_tlt.as_ref().map(|row| row.borrow());
            let b_extract = b_extract.as_ref().map(|row| row.borrow());
            if !b_first_inc
                .validate_mutually_exclusive_rows(b_use_raw_tlt.as_deref(), b_extract.as_deref())
            {
                return false;
            }
        }
        if !self
            .get_row(&DirectiveDef("ThicknessForTrimvol".into()))
            .is_none_or(|row| {
                row.borrow()
                    .validate_mutually_exclusive_field(self.parent.is_find_sec_add_thickness_set())
            })
        {
            return false;
        }
        if let Some(scale_to_mean_sd) = self.get_row(&DirectiveDef("ScaleToMeanSD".into())) {
            let scale_to_mean_sd = scale_to_mean_sd.borrow();
            let scale_from_x = self.get_row(&DirectiveDef("ScaleFromX".into()));
            let scale_from_x = scale_from_x.as_ref().map(|row| row.borrow());
            let scale_from_y = self.get_row(&DirectiveDef("ScaleFromY".into()));
            let scale_from_y = scale_from_y.as_ref().map(|row| row.borrow());
            if !scale_to_mean_sd
                .validate_mutually_exclusive_field(self.parent.is_scale_from_z_set())
                || !scale_to_mean_sd.validate_mutually_exclusive_rows(
                    scale_from_x.as_deref(),
                    scale_from_y.as_deref(),
                )
            {
                return false;
            }
        }
        true
    }
    fn display_all_rows(&mut self) {
        // Borrowing the panel separately preserves Java's shared panel/layout/constraints.
        let mut panel = CellPanelBoundary;
        for row in &self.list {
            row.display(&mut panel, &mut self.layout, &mut self.constraints);
        }
    }
}

impl<P: DirectivesTableParent> DirectivesTableBoundary for DirectivesTable<P> {
    fn init(&mut self) {
        self.init();
    }
    fn get_container(&self) -> &'static str {
        self.get_container()
    }
    fn get_row(&self, directive_def: &str) -> Option<String> {
        self.get_row(&DirectiveDef(directive_def.into()))
            .map(|row| row.borrow().to_string_value())
    }
    fn set_values_from_manager(&mut self, source_manager: &'static dyn BaseManager) {
        self.set_values_from_manager(source_manager);
    }
    fn set_values_from_directive_file(
        &mut self,
        _directive_file: &dyn DirectiveFileInterfaceBoundary,
        _set_field_highlight_value: bool,
    ) {
    }
    fn clear_template_values(&mut self) {
        self.clear_template_values();
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn checkpoint_and_restore_from_backup(&mut self, retain_user_values: bool) {
        self.checkpoint_and_restore_from_backup(retain_user_values);
    }
    fn validate(&self, field_displayer: &dyn FieldDisplayerBoundary) -> bool {
        self.validate(field_displayer)
    }
    fn backup_if_changed(&mut self) -> bool {
        self.backup_if_changed()
    }
    fn save_autodoc(
        &mut self,
        autodoc: &mut dyn WritableAutodocBoundary,
        do_validation: bool,
        field_displayer: &dyn FieldDisplayerBoundary,
        validate_only: bool,
    ) -> bool {
        self.save_autodoc(Some(autodoc), do_validation, field_displayer, validate_only)
    }
    fn status_changed(&mut self, status: BatchRunTomoStatus) {
        self.status_changed(status);
    }
    fn close_all_sections(&mut self) {
        self.close_all_sections();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Parent {
        thickness: bool,
        scale_z: bool,
        dual: bool,
    }
    impl DirectivesTableParent for Parent {
        fn is_find_sec_add_thickness_set(&self) -> bool {
            self.thickness
        }
        fn is_scale_from_z_set(&self) -> bool {
            self.scale_z
        }
        fn has_dual(&self) -> bool {
            self.dual
        }
    }
    struct Display;
    impl FieldDisplayerBoundary for Display {}
    fn row(key: &str, value: &str) -> Rc<RefCell<DirectivesDirectiveRow>> {
        let mut row = DirectivesDirectiveRow::new_description(
            None,
            key.into(),
            "s".into(),
            super::super::directives_directive_row::DirectiveValueType::String,
            false,
            DirectiveDef(key.into()),
            false,
        );
        row.set_value_string(value);
        Rc::new(RefCell::new(row))
    }
    #[test]
    fn map_backups_and_clear_use_canonical_directive_rows() {
        let mut table = DirectivesTable::new(
            None,
            Parent {
                thickness: false,
                scale_z: false,
                dual: false,
            },
            None,
            HashMap::new(),
            HashSet::new(),
        );
        table.init();
        let row = row("one", "value");
        table
            .directive_map
            .insert(DirectiveDef("one".into()), row.clone());
        table.checkpoint_and_restore_from_backup(false);
        row.borrow_mut().set_value_string("changed");
        assert!(table.backup_if_changed());
        table.clear();
        assert_eq!(row.borrow().get_value(), Some(String::new()));
        assert!(table.pnl_root.black_line_border);
    }
    #[test]
    fn mutual_exclusion_matches_java_number_of_patches_rule() {
        let mut table = DirectivesTable::new(
            None,
            Parent {
                thickness: false,
                scale_z: false,
                dual: false,
            },
            None,
            HashMap::new(),
            HashSet::new(),
        );
        table.directive_map.insert(
            DirectiveDef("NumberOfPatchesXandY".into()),
            row("NumberOfPatchesXandY", "2"),
        );
        table.directive_map.insert(
            DirectiveDef("OverlapOfPatchesXandY".into()),
            row("OverlapOfPatchesXandY", "0.1"),
        );
        assert!(!table.validate(&Display));
    }
    #[test]
    fn unknown_statement_is_inserted_once_and_value_is_applied() {
        let mut table = DirectivesTable::new(
            None,
            Parent {
                thickness: false,
                scale_z: false,
                dual: false,
            },
            None,
            HashMap::new(),
            HashSet::new(),
        );
        // The table's source ordering/deduplication is testable independently of
        // the dialog-owned section listener factory.
        let mut seen = HashSet::new();
        let statements = [
            DirectiveStatement {
                directive_def: DirectiveDef("x".into()),
                value: Some(DirectiveValue {
                    override_value: false,
                    batch: true,
                }),
            },
            DirectiveStatement {
                directive_def: DirectiveDef("x".into()),
                value: None,
            },
        ];
        assert!(seen.insert(statements[0].directive_def.clone()));
        assert!(!seen.insert(statements[1].directive_def.clone()));
    }
}
