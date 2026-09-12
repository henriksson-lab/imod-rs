//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesSectionRow.java`.
//!
//! `Ebutton`, `JPanel`, `GridBagLayout`, `GridBagConstraints`, and the
//! directive-row source unit are GUI/source-unit boundaries here.  The state
//! and transitions performed by `DirectivesSectionRow` remain in this file.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use super::appearance_extension::{FlagDisplay, FlagType};
use super::batch_run_tomo_step_panel::BatchRunTomoStatus;
use super::cell::{CellGridBagConstraintsBoundary, CellGridBagLayoutBoundary, CellPanelBoundary};
use super::directives_dialog::{
    DirectivesDialog, DirectivesDialogParent, DirectivesTableBoundary, SectionListener,
};
use super::directives_directive_row::DirectivesDirectiveRow;
use super::directives_row::DirectivesRow;

/// Calls made by this source unit to the neighbouring
/// `DirectivesDirectiveRow` source unit.
pub trait DirectivesDirectiveRowBoundary {
    /// Java `DirectivesDirectiveRow.setOpen(boolean)`.
    fn set_open(&mut self, open: bool);
    /// Java `DirectivesDirectiveRow.getFlagType()`.
    fn get_flag_type(&self) -> Option<FlagType>;
    /// Java `DirectivesDirectiveRow.canDisplay()`.
    fn can_display(&self) -> bool;
}

/// Direct implementation of the calls above once the neighbouring source unit
/// is present.  This is the Java `List<DirectivesDirectiveRow>` relationship,
/// rather than a replacement row model.
impl DirectivesDirectiveRowBoundary for DirectivesDirectiveRow {
    fn set_open(&mut self, open: bool) {
        DirectivesDirectiveRow::set_open(self, open);
    }

    fn get_flag_type(&self) -> Option<FlagType> {
        DirectivesDirectiveRow::get_flag_type(self)
    }

    fn can_display(&self) -> bool {
        DirectivesDirectiveRow::can_display(self)
    }
}

/// Direct calls to Swing's `Ebutton` from this source unit.  Rendering and
/// native listener dispatch remain at that boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectivesSectionEbuttonBoundary {
    pub text: String,
    pub selected: bool,
    pub enabled: bool,
    pub respond_to_non_error_flags: bool,
    pub allow_flag_editable_control: bool,
    pub action_listener_count: usize,
    pub removed: bool,
    pub flag_type: Option<FlagType>,
    pub add_count: usize,
}

impl DirectivesSectionEbuttonBoundary {
    /// Java `Ebutton.getHeaderInstance()`.
    pub fn get_header_instance() -> Self {
        Self {
            text: String::new(),
            selected: false,
            enabled: true,
            respond_to_non_error_flags: true,
            allow_flag_editable_control: true,
            action_listener_count: 0,
            removed: false,
            flag_type: None,
            add_count: 0,
        }
    }
    /// Java `Ebutton.getOpenCloseInstance(String)`.
    pub fn get_open_close_instance(text: impl Into<String>) -> Self {
        let mut instance = Self::get_header_instance();
        instance.text = text.into();
        instance
    }
    /// Java `Ebutton.add(JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn add(
        &mut self,
        _panel: &mut CellPanelBoundary,
        _layout: &mut CellGridBagLayoutBoundary,
        _constraints: &mut CellGridBagConstraintsBoundary,
    ) {
        self.add_count += 1;
    }
    /// Java `Ebutton.remove()`.
    pub fn remove(&mut self) {
        self.removed = true;
    }
    /// Java `Ebutton.addActionListener(ActionListener)`.
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }
    /// Java `Ebutton.setFlag(FlagType)`.
    pub fn set_flag(&mut self, flag_type: Option<FlagType>) {
        self.flag_type = flag_type;
    }
}

/// Java final package-private `DirectivesSectionRow`.
pub struct DirectivesSectionRow {
    pub h_empty: DirectivesSectionEbuttonBoundary,
    pub directive_list: Vec<Rc<RefCell<dyn DirectivesDirectiveRowBoundary>>>,
    pub btn_section: DirectivesSectionEbuttonBoundary,
    cur_error_flag_type: Option<FlagType>,
    /// The Java `GridBagConstraints.gridwidth` writes made by `display`.
    /// The native constraints object is deliberately an external GUI boundary.
    pub display_gridwidths: Vec<i32>,
}

impl DirectivesSectionRow {
    /// Java private `DirectivesSectionRow(String[])`.
    fn new(line_array: &[String]) -> Self {
        let mut btn_section = DirectivesSectionEbuttonBoundary::get_open_close_instance(
            line_array.first().cloned().unwrap_or_default(),
        );
        btn_section.respond_to_non_error_flags = false;
        btn_section.allow_flag_editable_control = false;
        Self {
            h_empty: DirectivesSectionEbuttonBoundary::get_header_instance(),
            directive_list: Vec::new(),
            btn_section,
            cur_error_flag_type: None,
            display_gridwidths: Vec::new(),
        }
    }
    /// Java private `DirectivesSectionRow(String)`.
    fn new_with_title(title: &str) -> Self {
        let mut btn_section = DirectivesSectionEbuttonBoundary::get_open_close_instance(title);
        btn_section.respond_to_non_error_flags = false;
        btn_section.allow_flag_editable_control = false;
        Self {
            h_empty: DirectivesSectionEbuttonBoundary::get_header_instance(),
            directive_list: Vec::new(),
            btn_section,
            cur_error_flag_type: None,
            display_gridwidths: Vec::new(),
        }
    }
    /// Java static `getInstance(DirectivesDialog, String[], JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn get_instance<T: DirectivesTableBoundary, P: DirectivesDialogParent>(
        parent: &mut DirectivesDialog<T, P>,
        line_array: &[String],
        _panel: &mut CellPanelBoundary,
        _layout: &mut CellGridBagLayoutBoundary,
        _constraints: &mut CellGridBagConstraintsBoundary,
    ) -> Rc<RefCell<Self>> {
        let instance = Rc::new(RefCell::new(Self::new(line_array)));
        Self::add_listeners(parent, instance.clone());
        instance
    }
    /// Java static `getExtrasInstance(DirectivesDialog)`.
    pub fn get_extras_instance<T: DirectivesTableBoundary, P: DirectivesDialogParent>(
        parent: &mut DirectivesDialog<T, P>,
    ) -> Rc<RefCell<Self>> {
        let instance = Rc::new(RefCell::new(Self::new_with_title("Unknown Directives")));
        Self::add_listeners(parent, instance.clone());
        instance
    }
    /// Java private `addListeners(DirectivesDialog)`.
    fn add_listeners<T: DirectivesTableBoundary, P: DirectivesDialogParent>(
        dialog: &mut DirectivesDialog<T, P>,
        instance: Rc<RefCell<Self>>,
    ) {
        instance.borrow_mut().btn_section.add_action_listener();
        dialog.add_section_listener(instance);
    }
    /// Java `getTitle()`.
    pub fn get_title(&self) -> &str {
        &self.btn_section.text
    }
    /// Java `setOpen(boolean)`.
    pub fn set_open(&mut self, open: bool) {
        if self.btn_section.selected == open {
            return;
        }
        self.btn_section.selected = open;
        for directive in &self.directive_list {
            directive.borrow_mut().set_open(open);
        }
    }
    /// Java `addDirective(DirectivesDirectiveRow)`.
    pub fn add_directive(&mut self, directive: Rc<RefCell<dyn DirectivesDirectiveRowBoundary>>) {
        self.directive_list.push(directive);
    }
    /// Java `hasDirectives()`.
    pub fn has_directives(&self) -> bool {
        !self.directive_list.is_empty()
    }
    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self) {
        self.update_open();
    }
    /// Java `display()` from `FieldDisplayer`.
    pub fn display_field(&mut self) {
        if !self.btn_section.selected {
            self.btn_section.selected = true;
            self.action_performed();
        }
    }
    /// Java private `updateOpen()`.
    fn update_open(&mut self) {
        let open = self.btn_section.selected;
        for directive in &self.directive_list {
            directive.borrow_mut().set_open(open);
        }
    }
    /// Java `sectionEvent()`.
    pub fn section_event(&mut self) {
        self.update_enabled();
    }
    /// Java private `isProcessFlag(FlagType)`.
    fn is_process_flag(&self, flag_type: Option<FlagType>) -> bool {
        let flag_type = flag_type.filter(|flag_type| flag_type.is_error());
        if flag_type.is_none() && self.cur_error_flag_type.is_none() {
            return false;
        }
        flag_type.is_none() || self.cur_error_flag_type.is_none()
    }
    /// Java `updateEnabled()`.
    pub fn update_enabled(&mut self) {
        let can_display = self
            .directive_list
            .iter()
            .any(|directive| directive.borrow().can_display());
        if !can_display && self.btn_section.selected {
            self.btn_section.selected = false;
            self.update_open();
        }
        self.btn_section.enabled = can_display;
    }
}

impl DirectivesRow for DirectivesSectionRow {
    /// Java `display(JPanel, GridBagLayout, GridBagConstraints)`.
    fn display(
        &mut self,
        pnl_table: &mut CellPanelBoundary,
        layout: &mut CellGridBagLayoutBoundary,
        constraints: &mut CellGridBagConstraintsBoundary,
    ) {
        self.display_gridwidths.push(1);
        self.btn_section.add(pnl_table, layout, constraints);
        // Java `GridBagConstraints.REMAINDER`.
        self.display_gridwidths.push(0);
        self.h_empty.add(pnl_table, layout, constraints);
    }
    /// Java `remove()`.
    fn remove(&mut self) {
        self.btn_section.remove();
        self.h_empty.remove();
    }
    /// Java `statusChanged(BatchRunTomoStatus) {}`.
    fn status_changed(&mut self, _status: BatchRunTomoStatus) {}
}

impl SectionListener for DirectivesSectionRow {
    fn section_event(&mut self) {
        self.section_event();
    }
}

impl FlagDisplay for DirectivesSectionRow {
    /// Java `setFlag(FlagType)`.
    fn set_flag(&mut self, flag_type: Option<FlagType>) {
        if !self.is_process_flag(flag_type) {
            return;
        }
        let error_flag = self.directive_list.iter().find_map(|directive| {
            directive
                .borrow()
                .get_flag_type()
                .filter(|flag| flag.is_error())
        });
        if self.is_process_flag(error_flag) {
            self.cur_error_flag_type = error_flag;
            self.btn_section.set_flag(self.cur_error_flag_type);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Directive {
        open: Vec<bool>,
        flag_type: Option<FlagType>,
        can_display: bool,
    }
    impl DirectivesDirectiveRowBoundary for Directive {
        fn set_open(&mut self, open: bool) {
            self.open.push(open);
        }
        fn get_flag_type(&self) -> Option<FlagType> {
            self.flag_type
        }
        fn can_display(&self) -> bool {
            self.can_display
        }
    }
    #[test]
    fn open_close_and_field_display_propagate_to_directives() {
        let mut row = DirectivesSectionRow::new_with_title("Section");
        let directive = Rc::new(RefCell::new(Directive {
            can_display: true,
            ..Directive::default()
        }));
        row.add_directive(directive.clone());
        row.set_open(true);
        row.set_open(true);
        row.action_performed();
        row.btn_section.selected = false;
        row.display_field();
        assert_eq!(directive.borrow().open, vec![true, true, true]);
        assert!(row.btn_section.selected);
        assert!(row.has_directives());
    }
    #[test]
    fn flags_and_visibility_follow_source_rules() {
        let mut row = DirectivesSectionRow::new_with_title("Section");
        let directive = Rc::new(RefCell::new(Directive {
            flag_type: Some(FlagType::ERROR),
            ..Directive::default()
        }));
        row.add_directive(directive.clone());
        row.set_flag(Some(FlagType::ERROR));
        assert_eq!(row.btn_section.flag_type, Some(FlagType::ERROR));
        row.btn_section.selected = true;
        directive.borrow_mut().can_display = false;
        row.update_enabled();
        assert!(!row.btn_section.enabled);
        assert!(!row.btn_section.selected);
        assert_eq!(directive.borrow().open, vec![false]);
    }
    #[test]
    fn table_display_remove_and_status_keep_source_boundary_calls() {
        let mut row = DirectivesSectionRow::new_with_title("Section");
        let mut panel = CellPanelBoundary;
        let mut layout = CellGridBagLayoutBoundary;
        let mut constraints = CellGridBagConstraintsBoundary;
        row.display(&mut panel, &mut layout, &mut constraints);
        row.status_changed(BatchRunTomoStatus::Running);
        row.remove();
        assert_eq!(row.display_gridwidths, vec![1, 0]);
        assert_eq!(row.btn_section.add_count, 1);
        assert_eq!(row.h_empty.add_count, 1);
        assert!(row.btn_section.removed);
        assert!(row.h_empty.removed);
    }
}
