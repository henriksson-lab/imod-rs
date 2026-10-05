//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesSectionRow.java`.
//!
//! A section header row of the directives table: an open/close button over the
//! directive rows of one section of the directives description file.  An event
//! dispatch thread object, created as `Rc<Self>`.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::directives_dialog::DirectivesDialog;
use super::directives_directive_row::DirectivesDirectiveRow;
use super::directives_row::DirectivesRow;
use super::ebutton::Ebutton;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, GRID_BAG_REMAINDER, GridBagConstraints, GridBagLayout,
    JComponent,
};
use crate::imod::etomo::storage::directive_descr_element::DirectiveDescrElement;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_type::FlagType;
use crate::imod::etomo::ui::section_listener::SectionListener;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java `final class DirectivesSectionRow implements DirectivesRow, ActionListener,
/// SectionListener, FlagDisplay, FieldDisplayer`.
pub struct DirectivesSectionRow {
    /// Java private final `hEmpty`.
    h_empty: Rc<Ebutton>,
    /// Java private final `directiveList`.
    directive_list: RefCell<Vec<Rc<DirectivesDirectiveRow>>>,
    /// Java private final `btnSection`.
    btn_section: Rc<Ebutton>,
    /// Java private `curErrorFlagType`, initially null.
    cur_error_flag_type: Cell<Option<&'static FlagType>>,
    /// Java `this`.
    this: Weak<DirectivesSectionRow>,
}

impl DirectivesSectionRow {
    /// The field initialisers with the given section button.
    fn construct(btn_section: Rc<Ebutton>) -> Rc<DirectivesSectionRow> {
        btn_section.set_respond_to_non_error_flags(false);
        btn_section.set_allow_flag_editable_control(false);
        Rc::new_cyclic(|this| DirectivesSectionRow {
            h_empty: Ebutton::get_header_instance_void(),
            directive_list: RefCell::new(Vec::new()),
            btn_section,
            cur_error_flag_type: Cell::new(None),
            this: this.clone(),
        })
    }

    /// Java private `DirectivesSectionRow(String[])`.
    fn new_line_array(line_array: Option<&[String]>) -> Rc<DirectivesSectionRow> {
        DirectivesSectionRow::construct(Ebutton::get_open_close_instance(
            DirectiveDescrElement::get_section_header_from_line_array(line_array).as_deref(),
        ))
    }

    /// Java private `DirectivesSectionRow(String)`.
    fn new_title(title: &str) -> Rc<DirectivesSectionRow> {
        DirectivesSectionRow::construct(Ebutton::get_open_close_instance(Some(title)))
    }

    /// Java package-private static `getInstance(DirectivesDialog, String[], JPanel,
    /// GridBagLayout, GridBagConstraints)`.
    pub fn get_instance(
        parent: &Rc<DirectivesDialog>,
        line_array: Option<&[String]>,
    ) -> Rc<DirectivesSectionRow> {
        let instance = DirectivesSectionRow::new_line_array(line_array);
        instance.add_listeners(parent);
        instance
    }

    /// Java package-private static `getExtrasInstance(DirectivesDialog)`.
    pub fn get_extras_instance(parent: &Rc<DirectivesDialog>) -> Rc<DirectivesSectionRow> {
        let instance = DirectivesSectionRow::new_title("Unknown Directives");
        instance.add_listeners(parent);
        instance
    }

    /// Java private `addListeners(DirectivesDialog)`.
    fn add_listeners(&self, dialog: &Rc<DirectivesDialog>) {
        let this = self.this.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(row) = this.upgrade() {
                row.action_performed(event);
            }
        });
        self.btn_section
            .add_action_listener_action_listener(Some(listener));
        if let Some(this) = self.this.upgrade() {
            dialog.add_section_listener(this as Rc<dyn SectionListener>);
        }
    }

    /// Java package-private `getTitle()`.
    pub fn get_title(&self) -> String {
        self.btn_section.get_text()
    }

    /// Java package-private `setOpen(boolean)`.
    pub fn set_open(&self, open: bool) {
        if self.btn_section.is_selected() == open {
            return;
        }
        self.btn_section.set_selected(open);
        let directive_list = self.directive_list.borrow().clone();
        for directive in directive_list {
            directive.set_open(open);
        }
    }

    /// Java package-private `addDirective(DirectivesDirectiveRow)`.
    pub fn add_directive(&self, directive: Rc<DirectivesDirectiveRow>) {
        self.directive_list.borrow_mut().push(directive);
    }

    /// Java package-private `hasDirectives()`.
    pub fn has_directives(&self) -> bool {
        !self.directive_list.borrow().is_empty()
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.update_open();
    }

    /// Java `display()` (FieldDisplayer).
    pub fn display_void(&self) {
        if !self.btn_section.is_selected() {
            self.btn_section.do_click();
        }
    }

    /// Java private `updateOpen()`.
    fn update_open(&self) {
        let open = self.btn_section.is_selected();
        let directive_list = self.directive_list.borrow().clone();
        for directive in directive_list {
            directive.set_open(open);
        }
    }

    /// Java private `isProcessFlag(FlagType)`.  Returns true if flagType is not equal
    /// to curErrorFlagType.  Not error flag types are treated as null flags because
    /// this class does not react to them.  Error flags are all processed in the same
    /// way so they are treated as the same flag.
    fn is_process_flag(&self, mut flag_type: Option<&'static FlagType>) -> bool {
        if let Some(flag) = flag_type
            && !flag.is_error()
        {
            flag_type = None;
        }
        let cur_error_flag_type = self.cur_error_flag_type.get();
        if flag_type.is_none() && cur_error_flag_type.is_none() {
            return false;
        }
        flag_type.is_none() || cur_error_flag_type.is_none()
    }

    /// Java package-private `updateEnabled()`.  Allows the section be disabled if no
    /// rows are visible.
    pub fn update_enabled(&self) {
        let can_display = self
            .directive_list
            .borrow()
            .iter()
            .any(|directive| directive.can_display());
        if !can_display && self.btn_section.is_selected() {
            self.btn_section.set_selected(false);
            self.update_open();
        }
        self.btn_section.set_enabled(can_display);
    }
}

impl DirectivesRow for DirectivesSectionRow {
    /// Java `remove()`.
    fn remove(&self) {
        self.btn_section.remove();
        self.h_empty.remove();
    }

    /// Java `display(JPanel, GridBagLayout, GridBagConstraints)`.
    fn display(
        &self,
        pnl_table: &Rc<JComponent>,
        layout: &GridBagLayout,
        constraints: &mut GridBagConstraints,
    ) {
        constraints.gridwidth = 1;
        self.btn_section.add(pnl_table);
        layout.set_constraints(&self.btn_section.get_component(), constraints);
        constraints.gridwidth = GRID_BAG_REMAINDER;
        self.h_empty.add(pnl_table);
        layout.set_constraints(&self.h_empty.get_component(), constraints);
    }

    /// Java `statusChanged(BatchRunTomoStatus)`: empty.
    fn status_changed(&self, _status: Option<BatchRunTomoStatus>) {}
}

impl SectionListener for DirectivesSectionRow {
    /// Java `sectionEvent()`.
    fn section_event(&self) {
        self.update_enabled();
    }
}

impl FlagDisplay for DirectivesSectionRow {
    /// Java `setFlag(FlagType)`.  Set a flag if any of the rows has an error flagType.
    /// Only do this if the flag has changed.
    fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        // If this change isn't significant, don't investigate it.
        if !self.is_process_flag(flag_type) {
            return;
        }
        let mut error_flag = None;
        for directive in self.directive_list.borrow().iter() {
            let flag_type = directive.get_flag_type();
            if let Some(flag) = flag_type
                && flag.is_error()
            {
                error_flag = Some(flag);
                break;
            }
        }
        // Call setFlag if something has changed.
        // Don't care which error flag it is - they all work the same for btnSection.
        if self.is_process_flag(error_flag) {
            self.cur_error_flag_type.set(error_flag);
            self.btn_section.set_flag(error_flag);
        }
    }
}

impl FieldDisplayer for DirectivesSectionRow {
    /// Java `display()`.
    fn display_void(&self) {
        DirectivesSectionRow::display_void(self);
    }

    /// Java `display(UIComponent)`.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {
        DirectivesSectionRow::display_void(self);
    }
}
