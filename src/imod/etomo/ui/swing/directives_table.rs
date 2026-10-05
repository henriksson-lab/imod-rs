//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesTable.java`.
//!
//! Directive editor table: one section row per section of the directives description
//! file and one directive row per directive, laid out with a `GridBagLayout`.  An event
//! dispatch thread object owned by its `DirectivesDialog`.
//!
//! The inner class `RowList` also implements `DirectiveMapInterface`
//! (`getDirective`, `getDirectiveFromPair`) for `setValues(BaseManager)`, which only
//! the never-called `DirectivesDialog.setValues(BaseManager)` reaches; those three are
//! dead in the Java too and are not translated (DEAD_CODE.md).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

use super::directives_dialog::DirectivesDialog;
use super::directives_directive_row::DirectivesDirectiveRow;
use super::directives_row::DirectivesRow;
use super::directives_section_row::DirectivesSectionRow;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{
    GRID_BAG_BOTH, GRID_BAG_CENTER, GridBagConstraints, GridBagLayout, JComponent,
};
use crate::imod::etomo::logic::batch_tool::TemplateValues;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::directive_adaptor::DirectiveAdaptor;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_descr_element::DirectiveDescrElement;
use crate::imod::etomo::storage::directive_descr_file;
use crate::imod::etomo::storage::directive_file_interface::DirectiveFileInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::batch_run_tomo_status::{self, BatchRunTomoStatus};
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;

/// One row of the table: Java's `List<DirectivesRow>` entries, with the `instanceof`
/// the class makes on them.
#[derive(Clone)]
enum Row {
    Section(Rc<DirectivesSectionRow>),
    Directive(Rc<DirectivesDirectiveRow>),
}

impl Row {
    fn as_row(&self) -> &dyn DirectivesRow {
        match self {
            Row::Section(row) => &**row,
            Row::Directive(row) => &**row,
        }
    }
}

/// Java package-private `final class DirectivesTable`.
pub struct DirectivesTable {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `layout`.
    layout: GridBagLayout,
    /// Java private final `constraints`.
    constraints: RefCell<GridBagConstraints>,
    /// Java private final `list` (the `RowList`).
    list: RowList,

    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `parent`.
    parent: Weak<DirectivesDialog>,
    /// Java private final `directiveFileType`.
    directive_file_type: Option<DirectiveFileType>,
    /// Java private final `templateValues`.
    template_values: Option<Rc<RefCell<TemplateValues>>>,
    /// Java private final `excludedDirectives`.
    excluded_directives: Option<Rc<RefCell<HashSet<DirectiveDef>>>>,

    /// Java private `debug`, initially false.
    debug: bool,
}

impl DirectivesTable {
    /// Java package-private `DirectivesTable(BaseManager, DirectivesDialog,
    /// DirectiveFileType, Map<DirectiveDef, String>, Set<DirectiveDef>)`.
    pub fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<DirectivesDialog>,
        directive_file_type: Option<DirectiveFileType>,
        template_values: Option<Rc<RefCell<TemplateValues>>>,
        excluded_directives: Option<Rc<RefCell<HashSet<DirectiveDef>>>>,
    ) -> DirectivesTable {
        DirectivesTable {
            pnl_root: JComponent::new_panel(),
            layout: GridBagLayout::new(),
            constraints: RefCell::new(GridBagConstraints::default()),
            list: RowList::new(),
            manager,
            parent,
            directive_file_type,
            template_values,
            excluded_directives,
            debug: false,
        }
    }

    fn parent(&self) -> Rc<DirectivesDialog> {
        self.parent.upgrade().expect("the dialog owns its table")
    }

    /// Java package-private `validate(FieldDisplayer)`.
    pub fn validate(&self, field_displayer: Option<Rc<dyn FieldDisplayer>>) -> bool {
        self.list.validate(&self.parent(), field_displayer)
    }

    /// Java package-private `getRow(DirectiveDef)`.
    pub fn get_row(&self, directive_def: Option<DirectiveDef>) -> Option<Rc<DirectivesDirectiveRow>> {
        self.list.get_row(directive_def)
    }

    /// Java package-private `init()`.
    pub fn init(&self) {
        self.create_panel();
        self.list.create_panel(self);
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        let mut constraints = self.constraints.borrow_mut();
        constraints.fill = GRID_BAG_BOTH;
        constraints.anchor = GRID_BAG_CENTER;
        constraints.gridwidth = 1;
        constraints.gridheight = 1;
        constraints.weightx = 0.0;
        constraints.weighty = 1.0;
        // root: `pnlRoot.setLayout(layout)`; `setBorder(LineBorder.createBlackLineBorder())`.
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `getTable()`.
    fn get_table(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java package-private `closeAllSections()`.
    pub fn close_all_sections(&self) {
        self.list.close_all_sections();
    }

    /// Java package-private `setValues(DirectiveFileInterface, boolean)`.
    pub fn set_values(
        &self,
        directive_file: &dyn DirectiveFileInterface,
        set_field_highlight_value: bool,
    ) {
        self.list.set_values(self, directive_file, set_field_highlight_value);
    }

    /// Java package-private `clearTemplateValues()`.
    pub fn clear_template_values(&self) {
        self.list.clear_template_values();
    }

    /// Java package-private `clear()`.
    pub fn clear(&self) {
        self.list.clear();
    }

    /// Java package-private `checkpointAndRestoreFromBackup(boolean)`.
    pub fn checkpoint_and_restore_from_backup(&self, retain_user_values: bool) {
        self.list.checkpoint_and_restore_from_backup(retain_user_values);
    }

    /// Java package-private `backupIfChanged()`.
    pub fn backup_if_changed(&self) -> bool {
        self.list.backup_if_changed()
    }

    /// Java package-private `saveAutodoc(WritableAutodoc, boolean, FieldDisplayer,
    /// boolean)`.
    pub fn save_autodoc(
        &self,
        autodoc: *mut Autodoc,
        do_validation: bool,
        field_displayer: Option<&dyn FieldDisplayer>,
        validate_only: bool,
    ) -> bool {
        self.list.save_autodoc(
            autodoc,
            do_validation,
            field_displayer,
            self.template_values.as_ref(),
            validate_only,
        )
    }

    /// Java package-private `statusChanged(BatchRunTomoStatus)`.
    pub fn status_changed(&self, status: Option<BatchRunTomoStatus>) {
        self.list.status_changed(status);
    }
}

/// Java inner class `RowList`.
struct RowList {
    /// Java private final `list`.
    list: RefCell<Vec<Row>>,
    /// Java private final `directiveMap`.
    directive_map: RefCell<HashMap<DirectiveDef, Rc<DirectivesDirectiveRow>>>,
    /// Java private `extrasSection`, initially null.
    extras_section: RefCell<Option<Rc<DirectivesSectionRow>>>,
}

impl RowList {
    /// Java private `RowList()`.
    fn new() -> RowList {
        RowList {
            list: RefCell::new(Vec::new()),
            directive_map: RefCell::new(HashMap::new()),
            extras_section: RefCell::new(None),
        }
    }

    /// Java private `validate(FieldDisplayer)`.
    fn validate(
        &self,
        parent: &Rc<DirectivesDialog>,
        field_displayer: Option<Rc<dyn FieldDisplayer>>,
    ) -> bool {
        // NumberOfPatchesXandY and OverlapOfPatchesXandY are mutually exclusive
        let mut row = self.get_row(Some(DirectiveDef::NUMBER_OF_PATCHES_X_AND_Y));
        if let Some(r) = &row
            && !r.validate_mutually_exclusive_rows(
                self.get_row(Some(DirectiveDef::OVERLAP_OF_PATCHES_X_AND_Y)).as_ref(),
                None,
                "Use only one of the following fields: \"Number of patches to track in X and Y\" or \"Fractional overlap of patches in X and Y\".",
                field_displayer.clone(),
            )
        {
            return false;
        }
        // firstinc, userawtlt, and extract are mutually exclusive
        row = self.get_row(Some(DirectiveDef::FIRST_INC));
        if let Some(r) = &row
            && !r.validate_mutually_exclusive_rows(
                self.get_row(Some(DirectiveDef::USE_RAW_TLT)).as_ref(),
                self.get_row(Some(DirectiveDef::EXTRACT)).as_ref(),
                "Only one method for getting tilt angles can be used for an axis.  Please choose only one of these fields: \"First tilt angle & increment\", \"Use existing .rawtlt file\", or \"Extract tilt angles from data file\".",
                field_displayer.clone(),
            )
        {
            return false;
        }
        // bfirstinc, buserawtlt, and bextract are mutually exclusive
        row = self.get_row(Some(DirectiveDef::BFIRST_INC));
        if let Some(r) = &row
            && !r.validate_mutually_exclusive_rows(
                self.get_row(Some(DirectiveDef::BUSE_RAW_TLT)).as_ref(),
                self.get_row(Some(DirectiveDef::BEXTRACT)).as_ref(),
                "Only one method for getting tilt angles can be used for the B axis.  Please choose only one of these fields: \"First tilt angle & increment for B axis\",  \"Use existing .rawtlt file for B axis\", or \"Extract tilt angles from B axis data file\".",
                field_displayer.clone(),
            )
        {
            return false;
        }
        // trimvol.thickness and trimvol.findSecAddThickness
        row = self.get_row(Some(DirectiveDef::THICKNESS_FOR_TRIMVOL));
        if let Some(r) = &row
            && !r.validate_mutually_exclusive_set(
                parent.is_find_sec_add_thickness_set(),
                "Use only one of the following fields: \"Fraction or # of slices to trim to in Z\" or the \"Find plastic section limits and add\" check-box/text-field in the Basic dialog.",
                field_displayer.clone(),
            )
        {
            return false;
        }
        // trimvol.scaleToMeanSD and trimvol.scaleFromZ
        row = self.get_row(Some(DirectiveDef::SCALE_TO_MEAN_SD));
        if let Some(r) = row {
            if !r.validate_mutually_exclusive_set(
                parent.is_scale_from_z_set(),
                "Only one scaling method can be used.  Use either \"Mean and SD for byte scaling\" or the \"Fraction of Z slices to analyze\" check-box/text-field in the Basic dialog.",
                field_displayer.clone(),
            ) {
                return false;
            }
            let compare_to_row = r;
            // trimvol.scaleFromX and trimvol.scaleToMeanSD
            let row = self.get_row(Some(DirectiveDef::SCALE_FROM_X));
            if let Some(r) = &row
                && !r.validate_mutually_exclusive_rows(
                    Some(&compare_to_row),
                    None,
                    "Only one scaling method can be used.  Use either \"Mean and SD for byte scaling\" or \"Frac or # of pixels in X for byte scaling\".",
                    field_displayer.clone(),
                )
            {
                return false;
            }
            // trimvol.scaleFromY and trimvol.scaleToMeanSD
            let row = self.get_row(Some(DirectiveDef::SCALE_FROM_Y));
            if let Some(r) = &row
                && !r.validate_mutually_exclusive_rows(
                    Some(&compare_to_row),
                    None,
                    "Only one scaling method can be used.  Use either \"Mean and SD for byte scaling\" or \"Frac or # of pixels in Y for byte scaling\".",
                    field_displayer,
                )
            {
                return false;
            }
        }
        true
    }

    /// Java package-private `createPanel()`.
    fn create_panel(&self, table: &DirectivesTable) {
        let parent = table.parent();
        let Some(mut iterator) = directive_descr_file::INSTANCE.get_iterator(None, None) else {
            // Java dereferences the null iterator of an unreadable description file;
            // fixed in translation: the table stays empty (BUGS.md).
            return;
        };
        let mut current_section: Option<Rc<DirectivesSectionRow>> = None;
        let pnl_table = table.get_table();
        let mut section_contains_directives = false;
        let mut directive_def: Option<DirectiveDef> = None;
        while iterator.has_next() {
            let prev_directive_def = directive_def;
            directive_def = None;
            let line_array = iterator.next();
            let line_array = line_array.as_deref();
            if DirectiveDescrElement::is_section_from_line_array(line_array) {
                // Finish previous section
                if let Some(section) = &current_section {
                    if !section_contains_directives {
                        self.list.borrow_mut().retain(|row| {
                            !matches!(row, Row::Section(existing) if Rc::ptr_eq(existing, section))
                        });
                    } else {
                        section.update_enabled();
                    }
                }
                // Start processing new section
                let section = DirectivesSectionRow::get_instance(&parent, line_array);
                self.list.borrow_mut().push(Row::Section(section.clone()));
                current_section = Some(section);
                section_contains_directives = false;
            } else {
                // If any of the datasets are dual axis, allow CopyArg B axis directives.
                directive_def =
                    DirectiveDescrElement::get_directive_def(line_array, prev_directive_def);
                if let Some(def) = directive_def
                    && DirectiveDescrElement::is_directive_from_line_array(line_array)
                    && DirectiveDescrElement::is_included(line_array, table.directive_file_type)
                    && table
                        .excluded_directives
                        .as_ref()
                        .is_none_or(|excluded| !excluded.borrow().contains(&def))
                    && (parent.has_dual()
                        || def.get_copy_arg_axis_id(
                            DirectiveDescrElement::get_name_from_line_array(line_array).as_deref(),
                        ) != Some(AxisID::Second))
                {
                    // Java passes the current section, which the description file always
                    // opens before its first directive.
                    let Some(section) = current_section.clone() else {
                        continue;
                    };
                    let row = DirectivesDirectiveRow::get_instance_line_array(
                        table.manager,
                        &parent,
                        section,
                        line_array,
                        table.directive_file_type.is_none(),
                        directive_def,
                        table.debug,
                    );
                    self.list.borrow_mut().push(Row::Directive(row.clone()));
                    section_contains_directives = true;
                    self.directive_map.borrow_mut().insert(def, row);
                }
            }
        }
        if let Some(section) = &current_section {
            section.update_enabled();
        }
        let list = self.list.borrow().clone();
        for row in &list {
            let mut constraints = *table.constraints.borrow();
            row.as_row()
                .display(&table.get_table(), &table.layout, &mut constraints);
            *table.constraints.borrow_mut() = constraints;
        }
        let _ = pnl_table;
        self.status_changed(Some(batch_run_tomo_status::DEFAULT));
    }

    /// Java private `closeAllSections()`.
    fn close_all_sections(&self) {
        let list = self.list.borrow().clone();
        for row in &list {
            if let Row::Section(section) = row {
                section.set_open(false);
            }
        }
    }

    /// Java private `getRow(DirectiveDef)`.
    fn get_row(&self, directive_def: Option<DirectiveDef>) -> Option<Rc<DirectivesDirectiveRow>> {
        let directive_def = directive_def?;
        self.directive_map.borrow().get(&directive_def).cloned()
    }

    /// Every directive row in list order (Java `DirectiveIterator`).
    fn directive_rows(&self) -> Vec<Rc<DirectivesDirectiveRow>> {
        self.list
            .borrow()
            .iter()
            .filter_map(|row| match row {
                Row::Directive(row) => Some(row.clone()),
                Row::Section(_) => None,
            })
            .collect()
    }

    /// Java private `clearTemplateValues()`.
    fn clear_template_values(&self) {
        for row in self.directive_rows() {
            row.clear_template_value();
        }
    }

    /// Java private `clear()`.
    fn clear(&self) {
        for row in self.directive_rows() {
            row.clear();
        }
    }

    /// Java private `checkpointAndRestoreFromBackup(boolean)`.
    fn checkpoint_and_restore_from_backup(&self, retain_user_values: bool) {
        for row in self.directive_rows() {
            // checkpoint
            row.checkpoint();
            // If the user wants to retain their values, apply backed up values and then
            // delete them.
            if retain_user_values {
                row.restore_from_backup();
            }
        }
    }

    /// Java private `backupIfChanged()`.  Check isDifferentFromCheckpoint on all data
    /// entry fields; returns true if any field's isDifferentFromCheckpoint returned
    /// true.
    fn backup_if_changed(&self) -> bool {
        let mut changed = false;
        for row in self.directive_rows() {
            if row.is_different_from_checkpoint(true) {
                row.backup();
                changed = true;
            }
        }
        changed
    }

    /// Java private `saveAutodoc(WritableAutodoc, boolean, FieldDisplayer, boolean)`.
    fn save_autodoc(
        &self,
        autodoc: *mut Autodoc,
        do_validation: bool,
        field_displayer: Option<&dyn FieldDisplayer>,
        template_values: Option<&Rc<RefCell<TemplateValues>>>,
        validate_only: bool,
    ) -> bool {
        if (validate_only && !do_validation) || autodoc.is_null() {
            return true;
        }
        let template_values = template_values.map(|template_values| template_values.borrow());
        for row in self.directive_rows() {
            if row
                .save_autodoc(
                    autodoc,
                    do_validation,
                    field_displayer,
                    template_values.as_deref(),
                    validate_only,
                )
                .is_err()
            {
                // catch (FieldValidationFailedException e)
                return false;
            }
        }
        true
    }

    /// Java package-private `setValues(DirectiveFileInterface, boolean)`.  Look for a
    /// matching directive in the table for each directive in directiveFiles and set or
    /// override the row value based on the directive.  If a directive does not exist
    /// in the table, then add it to an "Extras" section at the top.
    fn set_values(
        &self,
        table: &DirectivesTable,
        directive_files: &dyn DirectiveFileInterface,
        set_field_highlight_value: bool,
    ) {
        let Some(statements) = directive_files.iterator_statements(set_field_highlight_value)
        else {
            return;
        };
        let parent = table.parent();
        let mut directive = DirectiveAdaptor::new();
        let mut done_set: HashSet<DirectiveDef> = HashSet::new();
        let mut row_added = false;
        for statement in statements {
            directive.set(Some(statement));
            let directive_def = directive.get_directive_def();
            // Ignore most B axis only directives, excluded directives, and directives that
            // have already been processed.
            let Some(def) = directive_def else {
                continue;
            };
            if table
                .excluded_directives
                .as_ref()
                .is_none_or(|excluded| excluded.borrow().contains(&def))
                || done_set.contains(&def)
            {
                continue;
            }
            let mut row = self.get_row(Some(def));
            if row.is_none() {
                // add unknown directive to the Extras section
                if self.extras_section.borrow().is_none() {
                    let extras = DirectivesSectionRow::get_extras_instance(&parent);
                    *self.extras_section.borrow_mut() = Some(extras.clone());
                    self.list.borrow_mut().push(Row::Section(extras));
                    row_added = true;
                }
                // Put the new unknown one below the section header.
                let extras = self.extras_section.borrow().clone().unwrap();
                let new_row = DirectivesDirectiveRow::get_instance_directive(
                    table.manager,
                    &parent,
                    extras,
                    &mut directive,
                    true,
                );
                self.list.borrow_mut().push(Row::Directive(new_row.clone()));
                row_added = true;
                self.directive_map.borrow_mut().insert(def, new_row.clone());
                row = Some(new_row);
            }
            if let Some(row) = row {
                let mut template_values = table
                    .template_values
                    .as_ref()
                    .map(|template_values| template_values.borrow_mut());
                row.set_value(
                    directive_files,
                    set_field_highlight_value,
                    template_values.as_deref_mut(),
                );
            }
            // The whole collection is used to set the directive value, so only the first
            // instance of a directive has to be processed
            done_set.insert(def);
        }
        // If anything was added, remove and display everything.
        if row_added {
            if let Some(extras) = self.extras_section.borrow().as_ref() {
                extras.update_enabled();
            }
            let list = self.list.borrow().clone();
            for row in &list {
                row.as_row().remove();
            }
            for row in &list {
                let mut constraints = *table.constraints.borrow();
                row.as_row()
                    .display(&table.get_table(), &table.layout, &mut constraints);
                *table.constraints.borrow_mut() = constraints;
            }
        }
    }

    /// Java package-private `statusChanged(BatchRunTomoStatus)`.
    fn status_changed(&self, status: Option<BatchRunTomoStatus>) {
        let list = self.list.borrow().clone();
        for row in &list {
            row.as_row().status_changed(status);
        }
    }
}
