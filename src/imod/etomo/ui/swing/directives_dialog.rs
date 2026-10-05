//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesDialog.java`.
//!
//! The main dialog for the directives editor: the advanced view of a batchruntomo
//! dataset dialog, listing every directive of the directives description file.  An
//! event dispatch thread object, created as `Rc<Self>` by
//! [`DirectivesDialog::get_instance`].
//!
//! The public `setValues(BaseManager)` (`table.setValues(sourceManager)` ->
//! `sourceManager.updateDirectiveMap(...)`) has no caller in the Java and is not
//! translated (DEAD_CODE.md); nor is the commented-out `getInstance` overload.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use super::batch_run_tomo_dataset_dialog::BatchRunTomoDatasetDialog;
use super::check_box_efield::CheckBoxEfield;
use super::directives_directive_row::DirectivesDirectiveRow;
use super::directives_table::DirectivesTable;
use super::ebutton::Ebutton;
use super::etched_border::EtchedBorder;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::logic::batch_tool::TemplateValues;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file_interface::DirectiveFileInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::batch_run_tomo_status::BatchRunTomoStatus;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::row_listener::RowListener;
use crate::imod::etomo::ui::section_listener::SectionListener;

/// Java `public final class DirectivesDialog implements ActionListener`.
pub struct DirectivesDialog {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `btnCloseAllSections`.
    btn_close_all_sections: Rc<Ebutton>,
    /// Java private final `cbShowForTemplateOnly`.
    cb_show_for_template_only: Rc<CheckBoxEfield>,
    /// Java private final `cbShowIfSet`.
    cb_show_if_set: Rc<CheckBoxEfield>,
    /// Java private final `cbShowIncludedOnly`.
    cb_show_included_only: Rc<CheckBoxEfield>,
    /// Java private final `calibrationDir`.
    calibration_dir: Option<PathBuf>,

    /// Java private final `table`.
    table: DirectivesTable,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `directiveFileType`.
    directive_file_type: Option<DirectiveFileType>,
    /// Java private final `btnSave`.
    btn_save: Option<Rc<Ebutton>>,
    /// Java private final `btnCancel`.
    btn_cancel: Option<Rc<Ebutton>>,
    /// Java private final `cbShowUnchanged`.
    cb_show_unchanged: Option<Rc<CheckBoxEfield>>,
    /// Java private final `browsingDirectory`.
    browsing_directory: Option<Weak<dyn BrowsingDirectory>>,
    /// Java private final `parent`.
    parent: Weak<BatchRunTomoDatasetDialog>,

    /// Java private `rowListeners`, initially null.
    row_listeners: RefCell<Option<Vec<Rc<dyn RowListener>>>>,
    /// Java private `sectionListeners`, initially null.
    section_listeners: RefCell<Option<Vec<Rc<dyn SectionListener>>>>,
    /// Java `this`.
    this: Weak<DirectivesDialog>,
    /// Set once the panel is built (the rows need the dialog's `Rc`).
    created: Cell<bool>,
}

impl DirectivesDialog {
    /// Java private `DirectivesDialog(BaseManager, BatchRunTomoDatasetDialog,
    /// DirectiveFileType, Map<DirectiveDef, String>, Set<DirectiveDef>,
    /// BrowsingDirectory)`.
    fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<BatchRunTomoDatasetDialog>,
        directive_file_type: Option<DirectiveFileType>,
        template_values: Option<Rc<RefCell<TemplateValues>>>,
        excluded_directives: Option<Rc<RefCell<HashSet<DirectiveDef>>>>,
        browsing_directory: Option<Weak<dyn BrowsingDirectory>>,
    ) -> Rc<DirectivesDialog> {
        let (btn_save, btn_cancel, cb_show_unchanged) = if directive_file_type.is_some() {
            (
                Some(Ebutton::get_single_line_instance(Some("Save"))),
                Some(Ebutton::get_single_line_instance(Some("Cancel"))),
                Some(CheckBoxEfield::get_instance(Some("Show unchanged"))),
            )
        } else {
            (None, None, None)
        };
        Rc::new_cyclic(|this: &Weak<DirectivesDialog>| DirectivesDialog {
            pnl_root: JComponent::new_panel(),
            btn_close_all_sections: Ebutton::get_single_line_instance(Some("Close All Sections")),
            cb_show_for_template_only: CheckBoxEfield::get_instance(Some(
                "Only items saved to template by default",
            )),
            cb_show_if_set: CheckBoxEfield::get_instance(Some("Only items containing a value")),
            cb_show_included_only: CheckBoxEfield::get_instance(Some(
                "Only items output to batch files",
            )),
            calibration_dir: etomo_director::INSTANCE.get_imod_calib_directory(),
            table: DirectivesTable::new(
                manager,
                this.clone(),
                directive_file_type,
                template_values,
                excluded_directives,
            ),
            manager,
            directive_file_type,
            btn_save,
            btn_cancel,
            cb_show_unchanged,
            browsing_directory,
            parent,
            row_listeners: RefCell::new(None),
            section_listeners: RefCell::new(None),
            this: this.clone(),
            created: Cell::new(false),
        })
    }

    /// Java package-private static `getInstance(BatchRunTomoManager,
    /// BatchRunTomoDatasetDialog, Map<DirectiveDef, String>, Set<DirectiveDef>, Ebutton,
    /// boolean, BrowsingDirectory)`.
    pub fn get_instance(
        manager: &'static BatchRunTomoManager,
        parent: Weak<BatchRunTomoDatasetDialog>,
        template_values: Option<Rc<RefCell<TemplateValues>>>,
        excluded_directives: Option<Rc<RefCell<HashSet<DirectiveDef>>>>,
        btn_basic: Option<&Rc<Ebutton>>,
        shift_button: bool,
        browsing_dir: Option<Weak<dyn BrowsingDirectory>>,
    ) -> Rc<DirectivesDialog> {
        let instance = DirectivesDialog::new(
            manager,
            parent,
            None,
            template_values,
            excluded_directives,
            browsing_dir,
        );
        instance.create_panel(btn_basic, shift_button);
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    fn parent(&self) -> Rc<BatchRunTomoDatasetDialog> {
        self.parent
            .upgrade()
            .expect("the dataset dialog owns its directives dialog")
    }

    /// Java package-private `isFindSecAddThicknessSet()`.
    pub fn is_find_sec_add_thickness_set(&self) -> bool {
        self.parent().is_find_sec_add_thickness_set()
    }

    /// Java package-private `isScaleFromZSet()`.
    pub fn is_scale_from_z_set(&self) -> bool {
        self.parent().is_scale_from_z_set()
    }

    /// Java package-private `validate(FieldDisplayer)`.
    pub fn validate(&self, field_displayer: Option<Rc<dyn FieldDisplayer>>) -> bool {
        self.table.validate(field_displayer)
    }

    /// Java package-private `hasDual()`.
    pub fn has_dual(&self) -> bool {
        self.parent().has_dual()
    }

    /// Java private `createPanel(Ebutton, boolean)`.
    fn create_panel(&self, btn_basic: Option<&Rc<Ebutton>>, _shift_basic_button: bool) {
        // panels
        let pnl_control = JComponent::new_panel();
        let pnl_show = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        let pnl_save = if self.btn_save.is_some() {
            Some(JComponent::new_panel())
        } else {
            None
        };
        // init
        let mut _index = -1;
        if let Some(directive_file_type) = self.directive_file_type {
            _index = directive_file_type.get_index();
        }
        if let Some(directive_file_type) = self.directive_file_type {
            if directive_file_type.is_batch() {
                self.cb_show_for_template_only.set_enabled(false);
                self.cb_show_for_template_only.set_visible(false);
            } else {
                self.cb_show_for_template_only.set_selected(true);
            }
        }
        self.created.set(true);
        self.table.init();
        // root: `ToolTipManager.sharedInstance().registerComponent(pnlRoot)`.
        self.pnl_root.add(&pnl_control);
        self.pnl_root.add(&self.table.get_container());
        // control
        pnl_control.add(&pnl_show);
        pnl_control.add(&pnl_buttons);
        // Buttons
        if let Some(pnl_save) = &pnl_save {
            pnl_buttons.add(pnl_save);
        }
        if let Some(btn_basic) = btn_basic {
            pnl_buttons.add(&btn_basic.get_component());
        }
        pnl_buttons.add(&self.btn_close_all_sections.get_component());
        // show
        pnl_show.set_border_title(
            EtchedBorder::new(Some("Which Directives to Show"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        if let Some(cb_show_unchanged) = &self.cb_show_unchanged {
            pnl_show.add(&cb_show_unchanged.get_component());
        }
        // pnlShow.add(cbShowForTemplateOnly.getComponent());
        pnl_show.add(&self.cb_show_if_set.get_component());
        pnl_show.add(&self.cb_show_included_only.get_component());
        // Save
        if let (Some(pnl_save), Some(btn_save), Some(btn_cancel)) =
            (&pnl_save, &self.btn_save, &self.btn_cancel)
        {
            pnl_save.add(&btn_save.get_component());
            pnl_save.add(&btn_cancel.get_component());
        }
    }

    /// The `ActionListener` Java registers as `this`.
    fn action_listener(&self) -> ActionListener {
        let this = self.this.clone();
        Rc::new(move |event: &ActionEvent| {
            if let Some(dialog) = this.upgrade() {
                dialog.action_performed(Some(event));
            }
        })
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let listener = self.action_listener();
        self.cb_show_included_only.add_action_listener(listener.clone());
        self.cb_show_for_template_only
            .add_action_listener(listener.clone());
        self.cb_show_if_set.add_action_listener(listener.clone());
        self.btn_close_all_sections
            .add_action_listener_action_listener(Some(listener.clone()));
        if let (Some(btn_save), Some(btn_cancel)) = (&self.btn_save, &self.btn_cancel) {
            btn_save.add_action_listener_action_listener(Some(listener.clone()));
            btn_cancel.add_action_listener_action_listener(Some(listener));
        }
    }

    /// Java package-private `getRow(DirectiveDef)`.
    pub fn get_row(&self, directive_def: Option<DirectiveDef>) -> Option<Rc<DirectivesDirectiveRow>> {
        self.table.get_row(directive_def)
    }

    /// Java package-private `setValues(DirectiveFileInterface, boolean)`.
    pub fn set_values(&self, directive_file: &dyn DirectiveFileInterface, set_field_highlight_value: bool) {
        self.table.set_values(directive_file, set_field_highlight_value);
    }

    /// Java package-private `clearTemplateValues()`.
    pub fn clear_template_values(&self) {
        self.table.clear_template_values();
    }

    /// Java package-private `clear()`.
    pub fn clear(&self) {
        self.table.clear();
    }

    /// Java package-private `checkpointAndRestoreFromBackup(boolean)`.
    pub fn checkpoint_and_restore_from_backup(&self, retain_user_values: bool) {
        self.table.checkpoint_and_restore_from_backup(retain_user_values);
    }

    /// Java package-private `isShowForTemplateOnly()`.
    pub fn is_show_for_template_only(&self) -> bool {
        self.cb_show_for_template_only.is_selected() && self.cb_show_for_template_only.is_enabled()
    }

    /// Java package-private `isShowIfSet()`.
    pub fn is_show_if_set(&self) -> bool {
        self.cb_show_if_set.is_selected() && self.cb_show_if_set.is_enabled()
    }

    /// Java package-private `isShowIncludedOnly()`.
    pub fn is_show_included_only(&self) -> bool {
        self.cb_show_included_only.is_selected() && self.cb_show_included_only.is_enabled()
    }

    /// Java package-private `addRowListener(RowListener)`.  Allows the row to react to
    /// cbShowIncludedOnly actions after checkbox has been modified.
    pub fn add_row_listener(&self, listener: Rc<dyn RowListener>) {
        self.row_listeners
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(listener);
    }

    /// Java package-private `addSectionListener(SectionListener)`.  Allows the sections
    /// to react to checkbox actions after the row changes have completed.
    pub fn add_section_listener(&self, listener: Rc<dyn SectionListener>) {
        self.section_listeners
            .borrow_mut()
            .get_or_insert_with(Vec::new)
            .push(listener);
    }

    /// Java package-private `isSelected(String)`.
    pub fn is_selected(&self, action_command: Option<&str>) -> bool {
        let Some(action_command) = action_command else {
            return false;
        };
        let action_command = Some(action_command.to_owned());
        if action_command == self.cb_show_included_only.get_action_command() {
            return self.cb_show_included_only.is_selected();
        }
        if action_command == self.cb_show_for_template_only.get_action_command() {
            return self.cb_show_for_template_only.is_selected();
        }
        if action_command == self.cb_show_if_set.get_action_command() {
            return self.cb_show_if_set.is_selected();
        }
        false
    }

    /// Java package-private `backupIfChanged()`.
    pub fn backup_if_changed(&self) -> bool {
        self.table.backup_if_changed()
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
        self.table
            .save_autodoc(autodoc, do_validation, field_displayer, validate_only)
    }

    /// Java public `setAdvanced(boolean)`: empty.
    pub fn set_advanced(&self, _advanced: bool) {}

    /// Java package-private `statusChanged(BatchRunTomoStatus)`.
    pub fn status_changed(&self, status: Option<BatchRunTomoStatus>) {
        self.table.status_changed(status);
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        if let Some(cb_show_unchanged) = &self.cb_show_unchanged {
            cb_show_unchanged.set_tooltip(Some("Show directives whose values have not changed."));
        }
        self.cb_show_for_template_only.set_tooltip(Some(
            "Show only directives that are usually included in a directive file.",
        ));
        self.cb_show_if_set
            .set_tooltip(Some("Show only directives that have a value or are overridden."));
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java public `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        let Some(event) = event else {
            return;
        };
        let action_command = event.get_action_command().map(str::to_owned);
        if self.btn_close_all_sections.get_action_command() == action_command {
            self.table.close_all_sections();
        } else if self
            .btn_save
            .as_ref()
            .is_some_and(|btn_save| btn_save.get_action_command() == action_command)
        {
            ui_harness::with(|harness| {
                harness.save_base_manager_axis_id(Some(self.manager), Some(AxisID::Only))
            });
        } else if self
            .btn_cancel
            .as_ref()
            .is_some_and(|btn_cancel| btn_cancel.get_action_command() == action_command)
        {
            ui_harness::with(|harness| harness.cancel(Some(self.manager)));
        } else if self.cb_show_included_only.get_action_command() == action_command {
            let enable = !self.cb_show_included_only.is_selected();
            if let Some(cb_show_unchanged) = &self.cb_show_unchanged {
                cb_show_unchanged.set_enabled(enable);
            }
            self.cb_show_for_template_only.set_enabled(enable);
            self.send_events();
        } else if self.cb_show_for_template_only.get_action_command() == action_command
            || self.cb_show_if_set.get_action_command() == action_command
        {
            self.send_events();
        }
    }

    /// Java package-private `sendEvents()`.
    pub fn send_events(&self) {
        let row_listeners = self.row_listeners.borrow().clone();
        if let Some(row_listeners) = row_listeners {
            for listener in row_listeners {
                listener.row_event();
            }
        }
        let section_listeners = self.section_listeners.borrow().clone();
        if let Some(section_listeners) = section_listeners {
            for listener in section_listeners {
                listener.section_event();
            }
        }
    }

    /// Java package-private `getCalibrationDir()`.
    pub fn get_calibration_dir(&self) -> Option<PathBuf> {
        self.calibration_dir.clone()
    }

    /// Java package-private `getBrowsingDirectory()`.
    pub fn get_browsing_directory(&self) -> Option<Rc<dyn BrowsingDirectory>> {
        self.browsing_directory.as_ref().and_then(Weak::upgrade)
    }
}
