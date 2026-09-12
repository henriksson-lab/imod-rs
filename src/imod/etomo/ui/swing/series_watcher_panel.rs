//! `IMOD/Etomo/src/etomo/ui/swing/SeriesWatcherPanel.java`.
//!
//! Swing layout, autodoc lookup, directive-file lookup, and `BatchRunTomoManager`
//! dispatch remain explicit integration boundaries.  This source unit retains
//! the panel-owned values, match-string construction, enablement, checkpoint,
//! action, and expansion behaviour.
#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};

use crate::imod::etomo::{
    r#type::{axis_id::AxisID, axis_type::AxisType, dialog_type::DialogType},
    ui::field_type::FieldType,
    util::utilities,
};

use super::{
    check_box::CheckBox,
    labeled_text_field::{FieldValidationFailedException, LabeledTextField},
    panel_header::{ExpandButton, Expandable, PanelHeader, PanelHeaderState},
    radio_button::{RadioButton, RadioButtonGroup},
};

pub const MINIMUM_TILT_RANGE_DEFAULT: &str = "40";
pub const MINIMUM_NUMBER_OF_VIEWS_DEFAULT: &str = "12";
pub const MINUMUM_AGE_OF_STACKS_DEFAULT: &str = "300";

/// Java `SeriesWatcherParent` boundary.
pub trait SeriesWatcherParent {
    fn is_series_watcher_on(&self) -> bool;
}

/// Java `BatchRunTomoMetaData` members used by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoMetaDataBoundary {
    pub dual_axis: Option<bool>,
    pub two_surfaces: Option<bool>,
    pub mpoe_root_name: Option<String>,
    pub mpoe_ext: Option<String>,
    pub mpoe_include_combine: bool,
    pub mpoe_a_only: bool,
    pub mpoe_separate_b: bool,
    pub minimum_tilt_range: Option<String>,
    pub minimum_number_of_views: Option<String>,
    pub minimum_age_of_stacks: Option<String>,
}

/// Java `SeriesWatcherParam` members used by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SeriesWatcherParamBoundary {
    pub dual_axis: Option<bool>,
    pub two_surfaces: Option<bool>,
    pub match_pattern_or_ext: Option<String>,
    pub minimum_tilt_range: Option<String>,
    pub minimum_number_of_views: Option<String>,
    pub minimum_age_of_stacks: Option<String>,
}

/// Java `UserConfiguration.getSingleAxis()` value at the director boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UserConfigurationBoundary {
    pub single_axis: bool,
}

/// Java `BatchTool` directive application calls owned by `setValues`.
pub trait SeriesWatcherDirectiveFileCollection {
    fn set_dual_axis(&self, check_box: &mut CheckBox);
    fn set_two_surfaces(&self, check_box: &mut CheckBox);
}

/// Java `UIHarness.INSTANCE.pack(axisID, manager)` boundary.
pub trait SeriesWatcherUiHarness<M> {
    fn pack(&mut self, axis_id: AxisID, manager: &M);
}

/// Swing structures locally created by Java `createPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SeriesWatcherPanelLayout {
    pub pnl_root_visible: bool,
    pub pnl_root_order: Vec<&'static str>,
    pub pnl_header_etched_border: bool,
    pub pnl_header_order: Vec<&'static str>,
    pub pnl_body_order: Vec<&'static str>,
    pub pnl_settings_order: Vec<&'static str>,
    pub pnl_match_pattern_or_ext_order: Vec<&'static str>,
    pub pnl_mpoe_include_combine_order: Vec<&'static str>,
    pub pnl_mpoe_a_only_order: Vec<&'static str>,
    pub pnl_mpoe_separate_b_order: Vec<&'static str>,
    pub pnl_match_string_order: Vec<&'static str>,
}

/// Java package-private final `SeriesWatcherPanel`.
pub struct SeriesWatcherPanel<P: SeriesWatcherParent> {
    pub layout: SeriesWatcherPanelLayout,
    pub cb_dual_axis: CheckBox,
    pub cb_two_surfaces: CheckBox,
    pub bg_match_pattern_or_ext: Rc<RefCell<RadioButtonGroup>>,
    pub ltf_mpoe_root_name: LabeledTextField,
    pub ltf_mpoe_ext: LabeledTextField,
    pub rb_mpoe_include_combine: RadioButton,
    pub rb_mpoe_a_only: RadioButton,
    pub rb_mpoe_separate_b: RadioButton,
    pub l_match_string_label: String,
    pub l_match_string: String,
    /// Java assigns both labels this same tooltip in `setTooltips`.
    pub l_match_string_label_tooltip: Option<String>,
    pub l_match_string_tooltip: Option<String>,
    /// Java makes `lMatchString` two font points larger than `ltfMpoeExt`.
    pub l_match_string_font_size_delta: i32,
    pub ltf_minimum_tilt_range: LabeledTextField,
    pub ltf_minimum_number_of_views: LabeledTextField,
    pub ltf_minimum_age_of_stacks: LabeledTextField,
    pub axis_id: AxisID,
    pub parent: P,
    pub header: PanelHeader,
    pub advanced: bool,
    pub listener_count: usize,
    pub autodoc_tooltips_loaded: bool,
}

impl<P: SeriesWatcherParent> SeriesWatcherPanel<P> {
    /// Java private `SeriesWatcherPanel(BatchRunTomoManager, AxisID, DialogType, SeriesWatcherParent)`.
    fn new(axis_id: AxisID, dialog_type: DialogType, parent: P) -> Self {
        let bg_match_pattern_or_ext = Rc::new(RefCell::new(RadioButtonGroup::new()));
        Self {
            layout: SeriesWatcherPanelLayout::default(),
            cb_dual_axis: CheckBox::new_with_text("Dual axis"),
            cb_two_surfaces: CheckBox::new_with_text("Fiducials on 2 surfaces"),
            bg_match_pattern_or_ext: bg_match_pattern_or_ext.clone(),
            ltf_mpoe_root_name: LabeledTextField::new(
                FieldType::String,
                "Do stacks with root names matching ",
            ),
            ltf_mpoe_ext: LabeledTextField::new(FieldType::String, "Extension: "),
            rb_mpoe_include_combine: RadioButton::new_in_group(
                "Do both axes in one run OR Do B axis and finish combine",
                bg_match_pattern_or_ext.clone(),
            ),
            rb_mpoe_a_only: RadioButton::new_in_group(
                "Do A axis only",
                bg_match_pattern_or_ext.clone(),
            ),
            rb_mpoe_separate_b: RadioButton::new_in_group(
                "Do two runs: A axis first; then B axis and combine",
                bg_match_pattern_or_ext,
            ),
            l_match_string_label: "File name match string: ".into(),
            l_match_string: String::new(),
            l_match_string_label_tooltip: None,
            l_match_string_tooltip: None,
            l_match_string_font_size_delta: 2,
            ltf_minimum_tilt_range: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Minimum range of tilt angles (degrees): ",
            ),
            ltf_minimum_number_of_views: LabeledTextField::new(
                FieldType::Integer,
                "Minimum number of views: ",
            ),
            ltf_minimum_age_of_stacks: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Minimum time to wait if no .openTS (sec): ",
            ),
            axis_id,
            parent,
            // The constructor callback is a native widget boundary.  `new` has
            // exactly the choices made by `getAdvancedBasicOnlyInstance`.
            header: PanelHeader::new(
                "Series Watching Mode",
                true,
                false,
                dialog_type,
                false,
                true,
                true,
                false,
                true,
            ),
            advanced: false,
            listener_count: 0,
            autodoc_tooltips_loaded: false,
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType, parent: P) -> Self {
        let mut instance = Self::new(axis_id, dialog_type, parent);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tooltips();
        instance
    }

    /// Java `retrieveScreenStateFromDialog(BatchRunTomoScreenState)`.
    pub fn retrieve_screen_state_from_dialog(&self, state: &mut PanelHeaderState) {
        self.header.get_state(Some(state));
    }

    /// Java `applyScreenStateToDialog(BatchRunTomoScreenState)`.
    pub fn apply_screen_state_to_dialog(&mut self, state: &PanelHeaderState) {
        self.header.set_state(Some(state));
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        self.cb_dual_axis.set_directive_def(Some("dual"));
        self.cb_two_surfaces
            .set_directive_def(Some("surfaces-to-analyze"));
        self.ltf_mpoe_root_name.set_preferred_width(100, None);
        self.ltf_mpoe_ext.set_preferred_width(50, None);
        self.rb_mpoe_separate_b.set_selected(true);
        self.ltf_minimum_tilt_range
            .set_text(MINIMUM_TILT_RANGE_DEFAULT);
        self.ltf_minimum_number_of_views
            .set_text(MINIMUM_NUMBER_OF_VIEWS_DEFAULT);
        self.ltf_minimum_age_of_stacks
            .set_text(MINUMUM_AGE_OF_STACKS_DEFAULT);
        self.layout = SeriesWatcherPanelLayout {
            pnl_root_visible: true,
            pnl_root_order: vec!["rigidArea(0,15)", "pnlHeader", "glue"],
            pnl_header_etched_border: true,
            pnl_header_order: vec!["header", "pnlBody"],
            pnl_body_order: vec![
                "rigidArea(0,5)",
                "pnlSettings",
                "rigidArea(0,15)",
                "pnlMatchPatternOrExt",
                "pnlMpoeSeparateB",
                "pnlMpoeIncludeCombine",
                "pnlMpoeAOnly",
                "rigidArea(0,10)",
                "pnlMatchString",
                "rigidArea(0,15)",
                "ltfMinimumTiltRange",
                "ltfMinimumNumberOfViews",
                "ltfMinimumAgeOfStacks",
            ],
            pnl_settings_order: vec!["cbDualAxis", "horizontalStrut(3)", "cbTwoSurfaces", "glue"],
            pnl_match_pattern_or_ext_order: vec![
                "ltfMpoeRootName",
                "horizontalStrut(10)",
                "ltfMpoeExt",
                "horizontalGlue",
            ],
            pnl_mpoe_include_combine_order: vec!["rbMpoeIncludeCombine", "glue"],
            pnl_mpoe_a_only_order: vec!["rbMpoeAOnly", "glue"],
            pnl_mpoe_separate_b_order: vec!["rbMpoeSeparateB", "glue"],
            pnl_match_string_order: vec!["lMatchStringLabel", "lMatchString", "glue"],
        };
        self.build_match_pattern_or_ext();
        self.update_display();
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.layout.pnl_root_visible = visible;
    }

    /// Java `setParameters(BatchRunTomoMetaData)`.
    pub fn set_parameters_metadata(&mut self, meta_data: &BatchRunTomoMetaDataBoundary) {
        if let Some(value) = meta_data.dual_axis {
            self.cb_dual_axis.set_selected(value);
        }
        if let Some(value) = meta_data.two_surfaces {
            self.cb_two_surfaces.set_selected(value);
        }
        self.ltf_mpoe_root_name
            .set_text(meta_data.mpoe_root_name.as_deref().unwrap_or_default());
        self.ltf_mpoe_ext
            .set_text(meta_data.mpoe_ext.as_deref().unwrap_or_default());
        self.rb_mpoe_include_combine
            .set_selected(meta_data.mpoe_include_combine);
        self.rb_mpoe_a_only.set_selected(meta_data.mpoe_a_only);
        self.rb_mpoe_separate_b
            .set_selected(meta_data.mpoe_separate_b);
        self.build_match_pattern_or_ext();
        if let Some(value) = &meta_data.minimum_tilt_range {
            self.ltf_minimum_tilt_range.set_text(value);
        }
        if let Some(value) = &meta_data.minimum_number_of_views {
            self.ltf_minimum_number_of_views.set_text(value);
        }
        if let Some(value) = &meta_data.minimum_age_of_stacks {
            self.ltf_minimum_age_of_stacks.set_text(value);
        }
        self.update_display();
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        self.cb_dual_axis.set_editable(editable);
        self.cb_two_surfaces.set_editable(editable);
        self.ltf_mpoe_root_name.set_editable(editable);
        self.ltf_mpoe_ext.set_editable(editable);
        self.rb_mpoe_include_combine.set_editable(editable);
        self.rb_mpoe_a_only.set_editable(editable);
        self.rb_mpoe_separate_b.set_editable(editable);
        self.ltf_minimum_tilt_range.set_editable(editable);
        self.ltf_minimum_number_of_views.set_editable(editable);
        self.ltf_minimum_age_of_stacks.set_editable(editable);
    }

    /// Java `getParameters(BatchRunTomoMetaData)`.
    pub fn get_parameters_metadata(&self, meta_data: &mut BatchRunTomoMetaDataBoundary) {
        if meta_data.dual_axis.is_some() {
            meta_data.dual_axis = Some(self.cb_dual_axis.is_selected());
        }
        if meta_data.two_surfaces.is_some() {
            meta_data.two_surfaces = Some(self.cb_two_surfaces.is_selected());
        }
        meta_data.mpoe_root_name = Some(self.ltf_mpoe_root_name.get_text());
        meta_data.mpoe_ext = Some(self.ltf_mpoe_ext.get_text());
        meta_data.mpoe_include_combine = self.rb_mpoe_include_combine.is_selected();
        meta_data.mpoe_a_only = self.rb_mpoe_a_only.is_selected();
        meta_data.mpoe_separate_b = self.rb_mpoe_separate_b.is_selected();
        meta_data.minimum_tilt_range = Some(self.ltf_minimum_tilt_range.get_text());
        meta_data.minimum_number_of_views = Some(self.ltf_minimum_number_of_views.get_text());
        meta_data.minimum_age_of_stacks = Some(self.ltf_minimum_age_of_stacks.get_text());
    }

    /// Java `setParameters(SeriesWatcherParam)`.
    pub fn set_parameters_series_watcher(&mut self, param: &SeriesWatcherParamBoundary) {
        if let Some(value) = param.dual_axis {
            self.cb_dual_axis.set_selected(value);
        }
        if let Some(value) = param.two_surfaces {
            self.cb_two_surfaces.set_selected(value);
        }
        if let Some(value) = &param.minimum_tilt_range {
            self.ltf_minimum_tilt_range.set_text(value);
        }
        if let Some(value) = &param.minimum_number_of_views {
            self.ltf_minimum_number_of_views.set_text(value);
        }
        if let Some(value) = &param.minimum_age_of_stacks {
            self.ltf_minimum_age_of_stacks.set_text(value);
        }
    }

    /// Java `setParameters(UserConfiguration)`.
    pub fn set_parameters_user_configuration(
        &mut self,
        user_configuration: UserConfigurationBoundary,
    ) {
        if user_configuration.single_axis {
            self.cb_dual_axis.set_selected(false);
        }
    }

    /// Java `backupIfChanged(boolean)`.
    pub fn backup_if_changed(&mut self, only_advanced_dataset_dialog: bool) -> bool {
        let mut changed = false;
        if !only_advanced_dataset_dialog {
            if self.cb_dual_axis.is_different_from_checkpoint(true) {
                self.cb_dual_axis.backup();
                changed = true;
            }
            if self.cb_two_surfaces.is_different_from_checkpoint(true) {
                self.cb_two_surfaces.backup();
                changed = true;
            }
        }
        changed
    }

    /// Java `applyValues(boolean, boolean, DirectiveFileCollection, String, boolean)`.
    pub fn apply_values<D: SeriesWatcherDirectiveFileCollection>(
        &mut self,
        init: bool,
        retain_user_values: bool,
        directive_file_collection: &D,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
        user_configuration: UserConfigurationBoundary,
        meta_data: &BatchRunTomoMetaDataBoundary,
    ) {
        if only_stack_id_dataset_dialog.is_none() {
            if !init {
                self.cb_dual_axis.clear();
                self.cb_two_surfaces.clear();
            }
            self.set_parameters_user_configuration(user_configuration);
            self.set_values(directive_file_collection, only_advanced_dataset_dialog);
            self.cb_dual_axis.checkpoint();
            self.cb_two_surfaces.checkpoint();
            if retain_user_values {
                self.cb_dual_axis.restore_from_backup();
                self.cb_two_surfaces.restore_from_backup();
            } else {
                self.set_parameters_metadata(meta_data);
            }
            self.update_display();
        }
    }

    /// Java `setValues(DirectiveFileCollection, boolean)`.
    pub fn set_values<D: SeriesWatcherDirectiveFileCollection>(
        &mut self,
        directive_file_collection: &D,
        only_advanced_dataset_dialog: bool,
    ) {
        if !only_advanced_dataset_dialog {
            directive_file_collection.set_dual_axis(&mut self.cb_dual_axis);
            directive_file_collection.set_two_surfaces(&mut self.cb_two_surfaces);
            self.update_display();
        }
    }

    /// Java `getParameters(SeriesWatcherParam, boolean)`.
    pub fn get_parameters_series_watcher(
        &self,
        param: &mut SeriesWatcherParamBoundary,
        do_validation: bool,
    ) -> bool {
        let result: Result<(), FieldValidationFailedException> = (|| {
            param.dual_axis = Some(self.cb_dual_axis.is_selected());
            param.two_surfaces = Some(self.cb_two_surfaces.is_selected());
            param.match_pattern_or_ext = Some(self.l_match_string.clone());
            param.minimum_tilt_range = Some(
                self.ltf_minimum_tilt_range
                    .get_text_validated(do_validation)?,
            );
            param.minimum_number_of_views = Some(
                self.ltf_minimum_number_of_views
                    .get_text_validated(do_validation)?,
            );
            param.minimum_age_of_stacks = Some(
                self.ltf_minimum_age_of_stacks
                    .get_text_validated(do_validation)?,
            );
            Ok(())
        })();
        result.is_ok()
    }

    /// Java `getAxisType()`.
    pub fn get_axis_type(&self) -> AxisType {
        if self.cb_dual_axis.is_selected() {
            AxisType::DualAxis
        } else {
            AxisType::SingleAxis
        }
    }
    /// Java `isTwoSurfaces()`.
    pub fn is_two_surfaces(&self) -> bool {
        self.cb_two_surfaces.is_selected()
    }
    /// Java `isAOnly()`.
    pub fn is_a_only(&self) -> bool {
        self.rb_mpoe_a_only.is_enabled() && self.rb_mpoe_a_only.is_selected()
    }

    /// Java private `updateDisplay()`.
    pub fn update_display(&mut self) {
        let series_watcher_on = self.parent.is_series_watcher_on();
        let dual_axis = self.cb_dual_axis.is_selected();
        self.cb_dual_axis.set_enabled(series_watcher_on);
        self.cb_two_surfaces.set_enabled(series_watcher_on);
        self.ltf_mpoe_root_name.set_enabled(series_watcher_on);
        self.ltf_mpoe_ext.set_enabled(series_watcher_on);
        self.rb_mpoe_include_combine
            .set_enabled(series_watcher_on && dual_axis);
        self.rb_mpoe_a_only
            .set_enabled(series_watcher_on && dual_axis);
        self.rb_mpoe_separate_b
            .set_enabled(series_watcher_on && dual_axis);
        self.ltf_minimum_tilt_range.set_enabled(series_watcher_on);
        self.ltf_minimum_number_of_views
            .set_enabled(series_watcher_on);
        self.ltf_minimum_age_of_stacks
            .set_enabled(series_watcher_on);
        self.ltf_minimum_tilt_range.set_visible(self.advanced);
        self.ltf_minimum_number_of_views.set_visible(self.advanced);
        self.ltf_minimum_age_of_stacks.set_visible(self.advanced);
    }

    /// Java `focusGained(FocusEvent)`, deliberately empty.
    pub fn focus_gained(&mut self) {}
    /// Java `focusLost(FocusEvent)`.
    pub fn focus_lost(&mut self) {
        self.build_match_pattern_or_ext();
    }

    /// Java private `buildMatchPatternOrExt()`.
    pub fn build_match_pattern_or_ext(&mut self) {
        let mut pattern = String::new();
        let root_name = self.ltf_mpoe_root_name.get_text();
        if !root_name.is_empty() {
            pattern.push_str(&root_name);
            if !utilities::contains_wildcard(Some(&root_name)) {
                pattern.push('*');
            }
        } else {
            pattern.push('*');
        }
        if self.cb_dual_axis.is_selected() {
            if self.rb_mpoe_include_combine.is_selected() {
                pattern.push_str(&AxisID::Second.get_extension());
            } else if self.rb_mpoe_a_only.is_selected() {
                pattern.push_str(&AxisID::First.get_extension());
            } else if self.rb_mpoe_separate_b.is_selected() {
                pattern.push_str(
                    utilities::get_regular_expression_class(Some(
                        &(AxisID::First.get_extension() + &AxisID::Second.get_extension()),
                    ))
                    .as_deref()
                    .unwrap_or_default(),
                );
            }
        }
        pattern.push('.');
        let ext = self.ltf_mpoe_ext.get_text();
        let ext = ext.strip_prefix('.').unwrap_or(&ext);
        if ext.is_empty() {
            pattern.push_str("mrc");
        } else {
            pattern.push_str(ext);
        }
        self.l_match_string = pattern;
    }

    /// Java private `addListeners()`.
    fn add_listeners(&mut self) {
        self.cb_dual_axis.add_action_listener();
        self.ltf_mpoe_root_name.add_action_listener();
        self.ltf_mpoe_root_name.add_focus_listener();
        self.ltf_mpoe_ext.add_action_listener();
        self.ltf_mpoe_ext.add_focus_listener();
        self.rb_mpoe_include_combine.add_action_listener();
        self.rb_mpoe_a_only.add_action_listener();
        self.rb_mpoe_separate_b.add_action_listener();
        self.listener_count = 8;
    }

    /// Java `actionPerformed(ActionEvent)` with its native event object represented by an optional command.
    pub fn action_performed(&mut self, command: Option<&str>) {
        let Some(command) = command else { return };
        if self.cb_dual_axis.get_action_command() == Some(command)
            || self.ltf_mpoe_root_name.get_action_command() == command
            || self.ltf_mpoe_ext.get_action_command() == command
            || self.rb_mpoe_include_combine.get_action_command() == command
            || self.rb_mpoe_a_only.get_action_command() == command
            || self.rb_mpoe_separate_b.get_action_command() == command
        {
            self.build_match_pattern_or_ext();
        }
        self.update_display();
    }

    /// Java private `setTooltips()`.  Autodoc calls remain a boundary; callers
    /// may replace these values after `AutodocFactory` has supplied them.
    fn set_tooltips(&mut self) {
        self.ltf_mpoe_root_name.set_tool_tip_text(Some("Enter letters to match in the root name, or leave blank to match all eligible files.  Wild cards can be used: '*' to match any set of characters, '?' to match one character, or [list] to match any one character in the list, which can include ranges like a-z and 0-9.  If NO wild cards are used, '*' will be added after the entry, otherwise not. Do not include  axis letter for dual axis; if entry ends with a or b that will be considered part of the data set root name."));
        self.ltf_mpoe_ext
            .set_tool_tip_text(Some("Enter the filename extension without the '.'."));
        self.rb_mpoe_include_combine.set_tool_tip_text(Some("Wait for B axis to be present; if A was already processed, runs B axis and combine; if A is still present, does both axes and combine in one run."));
        self.rb_mpoe_a_only.set_tool_tip_text(Some(
            "Start data set with A axis only when it appears and ignore B axis stack",
        ));
        self.rb_mpoe_separate_b.set_tool_tip_text(Some("Start data set with A axis when it appears; do a second run to finish data set when B axis stack appears"));
        let tooltip =
            "This is the full string being entered with the -match option to Serieswatcher.";
        self.l_match_string_label_tooltip = Some(tooltip.into());
        self.l_match_string_tooltip = Some(tooltip.into());
    }

    /// Java `getComponent()` at the native Swing component boundary.
    pub fn get_component(&self) -> &SeriesWatcherPanelLayout {
        &self.layout
    }

    /// Java `expand(ExpandButton)` with the source's `UIHarness.pack` call.
    pub fn expand<M, H: SeriesWatcherUiHarness<M>>(
        &mut self,
        button: &ExpandButton,
        ui_harness: &mut H,
        manager: &M,
    ) {
        if self.header.equals_advanced_basic(button) {
            self.advanced = button.is_expanded();
            self.update_display();
            ui_harness.pack(self.axis_id, manager);
        }
    }
    /// Java `expand(GlobalExpandButton)`, deliberately empty.
    pub fn expand_global_button(&mut self) {}
}

impl<P: SeriesWatcherParent> Expandable for SeriesWatcherPanel<P> {
    fn expand_expand_button(&mut self, button: &ExpandButton) {
        if self.header.equals_advanced_basic(button) {
            self.advanced = button.is_expanded();
            self.update_display();
        }
    }
    fn expand_global_button(&mut self, _: &super::process_dialog::GlobalExpandButton) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone, Copy)]
    struct Parent(bool);
    impl SeriesWatcherParent for Parent {
        fn is_series_watcher_on(&self) -> bool {
            self.0
        }
    }
    #[test]
    fn match_string_tracks_root_extension_and_axis_choice() {
        let mut panel =
            SeriesWatcherPanel::get_instance(AxisID::Only, DialogType::BatchRunTomo, Parent(true));
        panel.cb_dual_axis.set_selected(true);
        panel.ltf_mpoe_root_name.set_text("set");
        panel.ltf_mpoe_ext.set_text(".st");
        panel.rb_mpoe_separate_b.set_selected(true);
        panel.build_match_pattern_or_ext();
        assert_eq!(panel.l_match_string, "set*[ab].st");
        panel.rb_mpoe_a_only.set_selected(true);
        panel.build_match_pattern_or_ext();
        assert_eq!(panel.l_match_string, "set*a.st");
    }
    #[test]
    fn display_uses_parent_axis_and_advanced_rules() {
        let mut panel =
            SeriesWatcherPanel::get_instance(AxisID::Only, DialogType::BatchRunTomo, Parent(false));
        assert!(!panel.cb_dual_axis.is_enabled());
        assert!(!panel.ltf_minimum_tilt_range.is_visible());
        panel.parent = Parent(true);
        panel.cb_dual_axis.set_selected(true);
        panel.advanced = true;
        panel.update_display();
        assert!(panel.rb_mpoe_a_only.is_enabled());
        assert!(panel.ltf_minimum_tilt_range.is_visible());
    }
    #[test]
    fn metadata_and_series_watcher_param_keep_source_values() {
        let mut panel =
            SeriesWatcherPanel::get_instance(AxisID::Only, DialogType::BatchRunTomo, Parent(true));
        let meta = BatchRunTomoMetaDataBoundary {
            dual_axis: Some(true),
            mpoe_root_name: Some("x".into()),
            mpoe_ext: Some("rec".into()),
            mpoe_a_only: true,
            ..Default::default()
        };
        panel.set_parameters_metadata(&meta);
        let mut param = SeriesWatcherParamBoundary::default();
        assert!(panel.get_parameters_series_watcher(&mut param, true));
        assert_eq!(param.match_pattern_or_ext.as_deref(), Some("x*a.rec"));
    }
}
