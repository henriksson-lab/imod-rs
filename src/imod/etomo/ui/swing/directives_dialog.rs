//! `IMOD/Etomo/src/etomo/ui/swing/DirectivesDialog.java`.
//!
//! Swing containers, `DirectivesTable`, directive storage, and the global
//! `UIHarness` are neighbouring source units / application boundaries.  This
//! unit retains the dialog's control construction, ordering, filtering rules,
//! listener ordering, and direct calls to those boundaries.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use super::batch_run_tomo_step_panel::BatchRunTomoStatus;
use super::check_box_efield::CheckBoxEfield;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;

/// Java `RowListener`.
pub trait RowListener {
    fn row_event(&mut self);
}

/// Java `SectionListener`.
pub trait SectionListener {
    fn section_event(&mut self);
}

/// Direct calls from this source unit to `BatchRunTomoDatasetDialog`.
pub trait DirectivesDialogParent {
    fn is_find_sec_add_thickness_set(&self) -> bool;
    fn is_scale_from_z_set(&self) -> bool;
    fn has_dual(&self) -> bool;
}

/// The `DirectiveFileInterface` storage input is owned by its untranslated
/// source unit.  Its identity is deliberately retained instead of replacing
/// directive-file semantics with strings.
pub trait DirectiveFileInterfaceBoundary {}

/// `WritableAutodoc` is an autodoc-storage boundary at this source unit.
pub trait WritableAutodocBoundary {}

/// The `FieldDisplayer` validation UI boundary.
pub trait FieldDisplayerBoundary {}

/// Direct `DirectivesTable` calls.  Its data model remains in its own source
/// unit, so an adapter preserves every call made by `DirectivesDialog`.
pub trait DirectivesTableBoundary {
    fn init(&mut self);
    fn get_container(&self) -> &'static str;
    fn get_row(&self, directive_def: &str) -> Option<String>;
    fn set_values_from_manager(&mut self, source_manager: &'static dyn BaseManager);
    fn set_values_from_directive_file(
        &mut self,
        directive_file: &dyn DirectiveFileInterfaceBoundary,
        set_field_highlight_value: bool,
    );
    fn clear_template_values(&mut self);
    fn clear(&mut self);
    fn checkpoint_and_restore_from_backup(&mut self, retain_user_values: bool);
    fn validate(&self, field_displayer: &dyn FieldDisplayerBoundary) -> bool;
    fn backup_if_changed(&mut self) -> bool;
    fn save_autodoc(
        &mut self,
        autodoc: &mut dyn WritableAutodocBoundary,
        do_validation: bool,
        field_displayer: &dyn FieldDisplayerBoundary,
        validate_only: bool,
    ) -> bool;
    fn status_changed(&mut self, status: BatchRunTomoStatus);
    fn close_all_sections(&mut self);
}

/// Java `UIHarness.INSTANCE.save/cancel` calls.
pub trait DirectivesDialogUiHarness {
    fn save(&mut self, manager: &'static dyn BaseManager, axis_id: AxisID);
    fn cancel(&mut self, manager: &'static dyn BaseManager);
}

/// Source-visible `Ebutton` state.  Actual button painting and listener
/// registration are GUI-boundary work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EbuttonBoundary {
    pub text: String,
    pub action_command: String,
    pub action_listener_count: usize,
}

impl EbuttonBoundary {
    /// Java `Ebutton.getSingleLineInstance(String)`.
    pub fn get_single_line_instance(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            action_command: text.clone(),
            text,
            action_listener_count: 0,
        }
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }
}

/// Source-visible JPanel/BoxLayout state, including the exact add order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirectivesDialogLayout {
    pub root_box_layout_y_axis: bool,
    pub control_box_layout_x_axis: bool,
    pub buttons_box_layout_y_axis: bool,
    pub show_box_layout_y_axis: bool,
    pub save_box_layout_x_axis: bool,
    pub root_order: Vec<String>,
    pub control_order: Vec<String>,
    pub buttons_order: Vec<String>,
    pub show_order: Vec<String>,
    pub save_order: Vec<String>,
    pub show_border: Option<String>,
    pub tooltip_manager_registered: bool,
    pub show_template_only_visible: bool,
    pub rigid_area_x143_y0: bool,
    pub shifted_basic_button: bool,
}

/// Java final `DirectivesDialog`.
pub struct DirectivesDialog<T: DirectivesTableBoundary, P: DirectivesDialogParent> {
    pub pnl_root: DirectivesDialogLayout,
    pub btn_close_all_sections: EbuttonBoundary,
    pub cb_show_for_template_only: CheckBoxEfield,
    pub cb_show_if_set: CheckBoxEfield,
    pub cb_show_included_only: CheckBoxEfield,
    pub calibration_dir: Option<PathBuf>,
    pub table: T,
    pub manager: &'static dyn BaseManager,
    pub directive_file_type: Option<DirectiveFileType>,
    pub btn_save: Option<EbuttonBoundary>,
    pub btn_cancel: Option<EbuttonBoundary>,
    pub cb_show_unchanged: Option<CheckBoxEfield>,
    pub browsing_directory: Rc<dyn BrowsingDirectory>,
    pub parent: P,
    pub row_listeners: Option<Vec<Rc<RefCell<dyn RowListener>>>>,
    pub section_listeners: Option<Vec<Rc<RefCell<dyn SectionListener>>>>,
}

impl<T: DirectivesTableBoundary, P: DirectivesDialogParent> DirectivesDialog<T, P> {
    /// Java private `DirectivesDialog(...)`.  `DirectivesTable` is supplied by
    /// the adjacent source-unit adapter, preserving Java's one-table instance.
    pub fn new(
        manager: &'static dyn BaseManager,
        parent: P,
        directive_file_type: Option<DirectiveFileType>,
        browsing_directory: Rc<dyn BrowsingDirectory>,
        calibration_dir: Option<PathBuf>,
        table: T,
    ) -> Self {
        let mut cb_show_for_template_only =
            CheckBoxEfield::get_instance("Only items saved to template by default");
        cb_show_for_template_only.check_box.action_command =
            Some("Only items saved to template by default".to_owned());
        let mut cb_show_if_set = CheckBoxEfield::get_instance("Only items containing a value");
        cb_show_if_set.check_box.action_command = Some("Only items containing a value".to_owned());
        let mut cb_show_included_only =
            CheckBoxEfield::get_instance("Only items output to batch files");
        cb_show_included_only.check_box.action_command =
            Some("Only items output to batch files".to_owned());
        let cb_show_unchanged = directive_file_type.map(|_| {
            let mut checkbox = CheckBoxEfield::get_instance("Show unchanged");
            checkbox.check_box.action_command = Some("Show unchanged".to_owned());
            checkbox
        });
        Self {
            pnl_root: DirectivesDialogLayout::default(),
            btn_close_all_sections: EbuttonBoundary::get_single_line_instance("Close All Sections"),
            cb_show_for_template_only,
            cb_show_if_set,
            cb_show_included_only,
            calibration_dir,
            table,
            manager,
            directive_file_type,
            btn_save: directive_file_type
                .map(|_| EbuttonBoundary::get_single_line_instance("Save")),
            btn_cancel: directive_file_type
                .map(|_| EbuttonBoundary::get_single_line_instance("Cancel")),
            cb_show_unchanged,
            browsing_directory,
            parent,
            row_listeners: None,
            section_listeners: None,
        }
    }

    /// Java package-private static `getInstance(BatchRunTomoManager, ...)`.
    /// Template maps and exclusions are owned by `DirectivesTable`; the table
    /// adapter is passed in after constructing it from those source arguments.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        parent: P,
        browsing_directory: Rc<dyn BrowsingDirectory>,
        calibration_dir: Option<PathBuf>,
        table: T,
        btn_basic: Option<EbuttonBoundary>,
        shift_button: bool,
    ) -> Self {
        let mut instance = Self::new(
            manager,
            parent,
            None,
            browsing_directory,
            calibration_dir,
            table,
        );
        instance.create_panel(btn_basic.as_ref(), shift_button);
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java `isFindSecAddThicknessSet`.
    pub fn is_find_sec_add_thickness_set(&self) -> bool {
        self.parent.is_find_sec_add_thickness_set()
    }
    /// Java `isScaleFromZSet`.
    pub fn is_scale_from_z_set(&self) -> bool {
        self.parent.is_scale_from_z_set()
    }
    /// Java `hasDual`.
    pub fn has_dual(&self) -> bool {
        self.parent.has_dual()
    }
    /// Java `validate(FieldDisplayer)`.
    pub fn validate(&self, field_displayer: &dyn FieldDisplayerBoundary) -> bool {
        self.table.validate(field_displayer)
    }

    /// Java private `createPanel(Ebutton, boolean)`.
    pub fn create_panel(&mut self, btn_basic: Option<&EbuttonBoundary>, shift_basic_button: bool) {
        if let Some(directive_file_type) = self.directive_file_type {
            let _index = directive_file_type.get_index();
            if directive_file_type.is_batch() {
                self.cb_show_for_template_only.set_enabled(false);
                self.cb_show_for_template_only.set_visible(false);
            } else {
                self.cb_show_for_template_only.set_selected(true);
            }
        }
        self.table.init();
        self.pnl_root.root_box_layout_y_axis = true;
        self.pnl_root.control_box_layout_x_axis = true;
        self.pnl_root.buttons_box_layout_y_axis = true;
        self.pnl_root.show_box_layout_y_axis = true;
        self.pnl_root.save_box_layout_x_axis = self.btn_save.is_some();
        self.pnl_root.tooltip_manager_registered = true;
        self.pnl_root.root_order = vec!["pnlControl".into(), self.table.get_container().into()];
        self.pnl_root.control_order = vec![
            "pnlShow".into(),
            "FixedDim.x143_y0".into(),
            "pnlButtons".into(),
        ];
        self.pnl_root.rigid_area_x143_y0 = true;
        if self.btn_save.is_some() {
            self.pnl_root.buttons_order.push("pnlSave".into());
            self.pnl_root.save_order = vec![
                "horizontalGlue".into(),
                "btnSave".into(),
                "horizontalGlue".into(),
                "btnCancel".into(),
                "horizontalGlue".into(),
            ];
        }
        if btn_basic.is_some() {
            if shift_basic_button {
                self.pnl_root.buttons_order.push("FixedDim.x0_y5".into());
            }
            self.pnl_root.buttons_order.push("btnBasic".into());
        }
        self.pnl_root.shifted_basic_button = btn_basic.is_some() && shift_basic_button;
        self.pnl_root
            .buttons_order
            .extend(["verticalGlue".into(), "btnCloseAllSections".into()]);
        self.pnl_root.show_border = Some("Which Directives to Show".into());
        if self.cb_show_unchanged.is_some() {
            self.pnl_root.show_order.push("cbShowUnchanged".into());
        }
        self.pnl_root
            .show_order
            .extend(["cbShowIfSet".into(), "cbShowIncludedOnly".into()]);
        self.pnl_root.show_template_only_visible = self.cb_show_for_template_only.is_visible();
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.cb_show_included_only.add_action_listener();
        self.cb_show_for_template_only.add_action_listener();
        self.cb_show_if_set.add_action_listener();
        self.btn_close_all_sections.add_action_listener();
        if let Some(button) = &mut self.btn_save {
            button.add_action_listener();
        }
        if let Some(button) = &mut self.btn_cancel {
            button.add_action_listener();
        }
    }

    /// Java `getRow(DirectiveDef)`; directive identity remains storage-owned.
    pub fn get_row(&self, directive_def: &str) -> Option<String> {
        self.table.get_row(directive_def)
    }
    /// Java `setValues(BaseManager)`.
    pub fn set_values(&mut self, source_manager: &'static dyn BaseManager) {
        self.table.set_values_from_manager(source_manager);
    }
    /// Java `setValues(DirectiveFileInterface, boolean)`.
    pub fn set_values_from_directive_file(
        &mut self,
        directive_file: &dyn DirectiveFileInterfaceBoundary,
        set_field_highlight_value: bool,
    ) {
        self.table
            .set_values_from_directive_file(directive_file, set_field_highlight_value);
    }
    /// Java `clearTemplateValues`.
    pub fn clear_template_values(&mut self) {
        self.table.clear_template_values();
    }
    /// Java `clear`.
    pub fn clear(&mut self) {
        self.table.clear();
    }
    /// Java `checkpointAndRestoreFromBackup`.
    pub fn checkpoint_and_restore_from_backup(&mut self, retain_user_values: bool) {
        self.table
            .checkpoint_and_restore_from_backup(retain_user_values);
    }
    /// Java `isShowForTemplateOnly`.
    pub fn is_show_for_template_only(&self) -> bool {
        self.cb_show_for_template_only.is_selected() && self.cb_show_for_template_only.is_enabled()
    }
    /// Java `isShowIfSet`.
    pub fn is_show_if_set(&self) -> bool {
        self.cb_show_if_set.is_selected() && self.cb_show_if_set.is_enabled()
    }
    /// Java `isShowIncludedOnly`.
    pub fn is_show_included_only(&self) -> bool {
        self.cb_show_included_only.is_selected() && self.cb_show_included_only.is_enabled()
    }

    /// Java `addRowListener`.
    pub fn add_row_listener(&mut self, listener: Rc<RefCell<dyn RowListener>>) {
        self.row_listeners
            .get_or_insert_with(Vec::new)
            .push(listener);
    }
    /// Java `addSectionListener`.
    pub fn add_section_listener(&mut self, listener: Rc<RefCell<dyn SectionListener>>) {
        self.section_listeners
            .get_or_insert_with(Vec::new)
            .push(listener);
    }

    /// Java `isSelected(String)`.
    pub fn is_selected(&self, action_command: Option<&str>) -> bool {
        let Some(action_command) = action_command else {
            return false;
        };
        if self.cb_show_included_only.get_action_command() == Some(action_command) {
            return self.cb_show_included_only.is_selected();
        }
        if self.cb_show_for_template_only.get_action_command() == Some(action_command) {
            return self.cb_show_for_template_only.is_selected();
        }
        if self.cb_show_if_set.get_action_command() == Some(action_command) {
            return self.cb_show_if_set.is_selected();
        }
        false
    }

    /// Java `backupIfChanged`.
    pub fn backup_if_changed(&mut self) -> bool {
        self.table.backup_if_changed()
    }
    /// Java `saveAutodoc`.
    pub fn save_autodoc(
        &mut self,
        autodoc: &mut dyn WritableAutodocBoundary,
        do_validation: bool,
        field_displayer: &dyn FieldDisplayerBoundary,
        validate_only: bool,
    ) -> bool {
        self.table
            .save_autodoc(autodoc, do_validation, field_displayer, validate_only)
    }
    /// Java `setAdvanced(boolean)`.
    pub fn set_advanced(&mut self, _advanced: bool) {}
    /// Java `statusChanged(BatchRunTomoStatus)`.
    pub fn status_changed(&mut self, status: BatchRunTomoStatus) {
        self.table.status_changed(status);
    }

    /// Java private `setTooltips`.
    pub fn set_tooltips(&mut self) {
        if let Some(checkbox) = &mut self.cb_show_unchanged {
            checkbox.set_tooltip("Show directives whose values have not changed.");
        }
        self.cb_show_for_template_only
            .set_tooltip("Show only directives that are usually included in a directive file.");
        self.cb_show_if_set
            .set_tooltip("Show only directives that have a value or are overridden.");
    }

    /// Java `getComponent`.
    pub fn get_component(&self) -> &DirectivesDialogLayout {
        &self.pnl_root
    }
    /// Java `getContainer`.
    pub fn get_container(&self) -> &DirectivesDialogLayout {
        &self.pnl_root
    }

    /// Java `actionPerformed(ActionEvent)`.  `UIHarness.INSTANCE` is supplied
    /// by the caller as its direct application boundary.
    pub fn action_performed(
        &mut self,
        action_command: Option<&str>,
        ui_harness: &mut dyn DirectivesDialogUiHarness,
    ) {
        let Some(action_command) = action_command else {
            return;
        };
        if self.btn_close_all_sections.action_command == action_command {
            self.table.close_all_sections();
        } else if self
            .btn_save
            .as_ref()
            .is_some_and(|button| button.action_command == action_command)
        {
            ui_harness.save(self.manager, AxisID::Only);
        } else if self
            .btn_cancel
            .as_ref()
            .is_some_and(|button| button.action_command == action_command)
        {
            ui_harness.cancel(self.manager);
        } else if self.cb_show_included_only.get_action_command() == Some(action_command) {
            let enable = !self.cb_show_included_only.is_selected();
            if let Some(checkbox) = &mut self.cb_show_unchanged {
                checkbox.set_enabled(enable);
            }
            self.cb_show_for_template_only.set_enabled(enable);
            self.send_events();
        } else if self.cb_show_for_template_only.get_action_command() == Some(action_command)
            || self.cb_show_if_set.get_action_command() == Some(action_command)
        {
            self.send_events();
        }
    }

    /// Java `sendEvents`; rows are notified before sections.
    pub fn send_events(&mut self) {
        if let Some(listeners) = &self.row_listeners {
            for listener in listeners {
                listener.borrow_mut().row_event();
            }
        }
        if let Some(listeners) = &self.section_listeners {
            for listener in listeners {
                listener.borrow_mut().section_event();
            }
        }
    }
    /// Java `getCalibrationDir`.
    pub fn get_calibration_dir(&self) -> Option<&std::path::Path> {
        self.calibration_dir.as_deref()
    }
    /// Java `getBrowsingDirectory`.
    pub fn get_browsing_directory(&self) -> &dyn BrowsingDirectory {
        self.browsing_directory.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    use std::cell::Cell;

    #[derive(Default)]
    struct Table {
        init: usize,
        close: usize,
        events: Vec<BatchRunTomoStatus>,
    }
    impl DirectivesTableBoundary for Table {
        fn init(&mut self) {
            self.init += 1;
        }
        fn get_container(&self) -> &'static str {
            "table"
        }
        fn get_row(&self, _: &str) -> Option<String> {
            None
        }
        fn set_values_from_manager(&mut self, _: &'static dyn BaseManager) {}
        fn set_values_from_directive_file(
            &mut self,
            _: &dyn DirectiveFileInterfaceBoundary,
            _: bool,
        ) {
        }
        fn clear_template_values(&mut self) {}
        fn clear(&mut self) {}
        fn checkpoint_and_restore_from_backup(&mut self, _: bool) {}
        fn validate(&self, _: &dyn FieldDisplayerBoundary) -> bool {
            true
        }
        fn backup_if_changed(&mut self) -> bool {
            true
        }
        fn save_autodoc(
            &mut self,
            _: &mut dyn WritableAutodocBoundary,
            _: bool,
            _: &dyn FieldDisplayerBoundary,
            _: bool,
        ) -> bool {
            true
        }
        fn status_changed(&mut self, status: BatchRunTomoStatus) {
            self.events.push(status);
        }
        fn close_all_sections(&mut self) {
            self.close += 1;
        }
    }
    struct Parent;
    impl DirectivesDialogParent for Parent {
        fn is_find_sec_add_thickness_set(&self) -> bool {
            true
        }
        fn is_scale_from_z_set(&self) -> bool {
            false
        }
        fn has_dual(&self) -> bool {
            true
        }
    }
    #[derive(Default)]
    struct Browsing;
    impl BrowsingDirectory for Browsing {
        fn get_browsing_dir(&self) -> Option<PathBuf> {
            None
        }
        fn set_browsing_dir(&self, _: Option<&std::path::Path>) {}
    }
    #[derive(Default)]
    struct Harness {
        save: usize,
        cancel: usize,
    }
    impl DirectivesDialogUiHarness for Harness {
        fn save(&mut self, _: &'static dyn BaseManager, _: AxisID) {
            self.save += 1
        }
        fn cancel(&mut self, _: &'static dyn BaseManager) {
            self.cancel += 1
        }
    }
    struct Counter(Rc<Cell<usize>>);
    impl RowListener for Counter {
        fn row_event(&mut self) {
            self.0.set(self.0.get() + 1)
        }
    }
    impl SectionListener for Counter {
        fn section_event(&mut self) {
            self.0.set(self.0.get() + 1)
        }
    }
    fn manager() -> &'static dyn BaseManager {
        DirectiveEditorManager::new(None, None, None, None)
    }
    fn dialog(file_type: Option<DirectiveFileType>) -> DirectivesDialog<Table, Parent> {
        DirectivesDialog::new(
            manager(),
            Parent,
            file_type,
            Rc::new(Browsing),
            Some(PathBuf::from("calib")),
            Table::default(),
        )
    }
    #[test]
    fn source_panel_construction_preserves_type_dependent_filters_and_order() {
        let mut value = dialog(Some(DirectiveFileType::Batch));
        value.create_panel(None, false);
        assert_eq!(value.table.init, 1);
        assert!(!value.cb_show_for_template_only.is_enabled());
        assert!(!value.cb_show_for_template_only.is_visible());
        assert_eq!(value.pnl_root.root_order, vec!["pnlControl", "table"]);
        assert_eq!(
            value.pnl_root.show_order,
            vec!["cbShowUnchanged", "cbShowIfSet", "cbShowIncludedOnly"]
        );
    }
    #[test]
    fn template_type_selects_template_filter_and_save_controls() {
        let mut value = dialog(Some(DirectiveFileType::Scope));
        value.create_panel(
            Some(&EbuttonBoundary::get_single_line_instance("Basic")),
            true,
        );
        assert!(value.is_show_for_template_only());
        assert!(value.pnl_root.shifted_basic_button);
        assert_eq!(value.pnl_root.save_order.len(), 5);
    }
    #[test]
    fn included_only_disables_filters_then_notifies_rows_before_sections() {
        let mut value = dialog(Some(DirectiveFileType::User));
        let sequence = Rc::new(Cell::new(0));
        value.add_row_listener(Rc::new(RefCell::new(Counter(sequence.clone()))));
        value.add_section_listener(Rc::new(RefCell::new(Counter(sequence.clone()))));
        value.cb_show_included_only.set_selected(true);
        let mut harness = Harness::default();
        let action_command = value
            .cb_show_included_only
            .get_action_command()
            .map(str::to_owned);
        value.action_performed(action_command.as_deref(), &mut harness);
        assert!(!value.cb_show_for_template_only.is_enabled());
        assert!(!value.cb_show_unchanged.unwrap().is_enabled());
        assert_eq!(sequence.get(), 2);
    }
    #[test]
    fn commands_delegate_to_table_and_ui_harness_and_null_event_is_ignored() {
        let mut value = dialog(Some(DirectiveFileType::User));
        let mut harness = Harness::default();
        value.action_performed(None, &mut harness);
        value.action_performed(Some("Close All Sections"), &mut harness);
        value.action_performed(Some("Save"), &mut harness);
        value.action_performed(Some("Cancel"), &mut harness);
        assert_eq!(value.table.close, 1);
        assert_eq!((harness.save, harness.cancel), (1, 1));
    }
    #[test]
    fn direct_parent_and_table_delegations_are_retained() {
        let mut value = dialog(None);
        assert!(value.is_find_sec_add_thickness_set());
        assert!(!value.is_scale_from_z_set());
        assert!(value.has_dual());
        value.status_changed(BatchRunTomoStatus::Done);
        assert_eq!(value.table.events, vec![BatchRunTomoStatus::Done]);
        assert_eq!(
            value.get_calibration_dir(),
            Some(std::path::Path::new("calib"))
        );
    }
}
