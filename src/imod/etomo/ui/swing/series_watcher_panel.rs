//! `IMOD/Etomo/src/etomo/ui/swing/SeriesWatcherPanel.java`.
//!
//! Contains parameters for the serieswatcher process: the "Series Watching Mode" panel
//! of the batchruntomo dialog's Run tab.  An event dispatch thread object, created as
//! `Rc<Self>` by [`SeriesWatcherPanel::get_instance`].

use std::cell::Cell;
use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::labeled_text_field::LabeledTextField;
use super::panel_header::PanelHeader;
use super::radio_button::RadioButton;
use super::series_watcher_parent::SeriesWatcherParent;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::batch_run_tomo_manager::BatchRunTomoManager;
use crate::imod::etomo::comscript::series_watcher_param::{self, SeriesWatcherParam};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, FocusEvent, FocusListener, JComponent,
};
use crate::imod::etomo::logic::batch_tool;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::directive_def::{self, DirectiveDef};
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::batch_run_tomo_meta_data::BatchRunTomoMetaData;
use crate::imod::etomo::r#type::batch_run_tomo_screen_state::BatchRunTomoScreenState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::util::utilities;

/// Java `final class SeriesWatcherPanel implements Expandable, ActionListener,
/// FocusListener`.
pub struct SeriesWatcherPanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<JComponent>,
    /// Java private final `pnlHeader`.
    pnl_header: Rc<JComponent>,
    /// Java private final `pnlBody`.
    pnl_body: Rc<JComponent>,
    /// Java private final `cbDualAxis`.
    cb_dual_axis: Rc<CheckBox>,
    /// Java private final `cbTwoSurfaces`.
    cb_two_surfaces: Rc<CheckBox>,
    /// Java private final `bgMatchPatternOrExt`.
    #[allow(dead_code)]
    bg_match_pattern_or_ext: Rc<ButtonGroup>,
    /// Java private final `ltfMpoeRootName`.
    ltf_mpoe_root_name: Rc<LabeledTextField>,
    /// Java private final `ltfMpoeExt`.
    ltf_mpoe_ext: Rc<LabeledTextField>,
    /// Java private final `rbMpoeIncludeCombine`.
    rb_mpoe_include_combine: Rc<RadioButton>,
    /// Java private final `rbMpoeAOnly`.
    rb_mpoe_a_only: Rc<RadioButton>,
    /// Java private final `rbMpoeSeparateB`.
    rb_mpoe_separate_b: Rc<RadioButton>,
    /// Java private final `lMatchStringLabel`.
    l_match_string_label: Rc<JComponent>,
    /// Java private final `lMatchString`.
    l_match_string: Rc<JComponent>,
    /// Java private final `ltfMinimumTiltRange`.
    ltf_minimum_tilt_range: Rc<LabeledTextField>,
    /// Java private final `ltfMinimumNumberOfViews`.
    ltf_minimum_number_of_views: Rc<LabeledTextField>,
    /// Java private final `ltfMinimumAgeOfStacks`.
    ltf_minimum_age_of_stacks: Rc<LabeledTextField>,

    /// Java private final `manager`.
    manager: &'static BatchRunTomoManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `parent`.
    parent: Weak<dyn SeriesWatcherParent>,
    /// Java private final `header`.
    header: Rc<PanelHeader>,

    /// Java private `advanced`, initially false.
    advanced: Cell<bool>,
    /// Java `this`.
    this: Weak<SeriesWatcherPanel>,
}

impl SeriesWatcherPanel {
    /// Java private `SeriesWatcherPanel(BatchRunTomoManager, AxisID, DialogType,
    /// SeriesWatcherParent)`, with the field initialisers.
    fn new(
        manager: &'static BatchRunTomoManager,
        axis_id: AxisID,
        dialog_type: Option<DialogType>,
        parent: Weak<dyn SeriesWatcherParent>,
    ) -> Rc<SeriesWatcherPanel> {
        let bg_match_pattern_or_ext = ButtonGroup::new();
        Rc::new_cyclic(|this: &Weak<SeriesWatcherPanel>| {
            let expandable: Weak<dyn Expandable> = this.clone();
            SeriesWatcherPanel {
                pnl_root: JComponent::new_panel(),
                pnl_header: JComponent::new_panel(),
                pnl_body: JComponent::new_panel(),
                cb_dual_axis: CheckBox::new_string(Some("Dual axis")),
                cb_two_surfaces: CheckBox::new_string(Some("Fiducials on 2 surfaces")),
                ltf_mpoe_root_name: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some("Do stacks with root names matching "),
                ),
                ltf_mpoe_ext: LabeledTextField::new_field_type_string(
                    FieldType::String,
                    Some("Extension: "),
                ),
                rb_mpoe_include_combine: RadioButton::new_string_button_group(
                    Some("Do both axes in one run OR Do B axis and finish combine"),
                    Some(&bg_match_pattern_or_ext),
                ),
                rb_mpoe_a_only: RadioButton::new_string_button_group(
                    Some("Do A axis only"),
                    Some(&bg_match_pattern_or_ext),
                ),
                rb_mpoe_separate_b: RadioButton::new_string_button_group(
                    Some("Do two runs: A axis first; then B axis and combine"),
                    Some(&bg_match_pattern_or_ext),
                ),
                bg_match_pattern_or_ext: bg_match_pattern_or_ext.clone(),
                l_match_string_label: JComponent::new_label("File name match string: "),
                l_match_string: JComponent::new_label(""),
                ltf_minimum_tilt_range: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Minimum range of tilt angles (degrees): "),
                ),
                ltf_minimum_number_of_views: LabeledTextField::new_field_type_string(
                    FieldType::Integer,
                    Some("Minimum number of views: "),
                ),
                ltf_minimum_age_of_stacks: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("Minimum time to wait if no .openTS (sec): "),
                ),
                manager,
                axis_id,
                parent,
                header: PanelHeader::get_advanced_basic_only_instance(
                    Some("Series Watching Mode"),
                    Some(expandable),
                    dialog_type,
                    None,
                    false,
                ),
                advanced: Cell::new(false),
                this: this.clone(),
            }
        })
    }

    /// Java package-private static `getInstance(BatchRunTomoManager, AxisID,
    /// DialogType, SeriesWatcherParent)`.
    pub fn get_instance(
        manager: &'static BatchRunTomoManager,
        axis_id: AxisID,
        dialog_type: Option<DialogType>,
        parent: Weak<dyn SeriesWatcherParent>,
    ) -> Rc<SeriesWatcherPanel> {
        let instance = SeriesWatcherPanel::new(manager, axis_id, dialog_type, parent);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tooltips();
        instance
    }

    /// Java public `retrieveScreenStateFromDialog(BatchRunTomoScreenState)`.
    pub fn retrieve_screen_state_from_dialog(&self, screen_state: &BatchRunTomoScreenState) {
        self.header
            .get_state(Some(screen_state.get_series_watcher_header_state()));
    }

    /// Java public `applyScreenStateToDialog(BatchRunTomoScreenState)`.
    pub fn apply_screen_state_to_dialog(&self, screen_state: &BatchRunTomoScreenState) {
        self.header
            .set_state(Some(screen_state.get_series_watcher_header_state()));
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // panels
        let pnl_settings = JComponent::new_panel();
        let pnl_match_pattern_or_ext = JComponent::new_panel();
        let pnl_mpoe_include_combine = JComponent::new_panel();
        let pnl_mpoe_a_only = JComponent::new_panel();
        let pnl_mpoe_separate_b = JComponent::new_panel();
        let pnl_match_string = JComponent::new_panel();
        // init
        self.cb_dual_axis.set_directive_def(Some(DirectiveDef::DUAL));
        self.cb_two_surfaces
            .set_directive_def(Some(DirectiveDef::SURFACES_TO_ANALYZE));
        self.ltf_mpoe_root_name.set_preferred_width(100);
        self.ltf_mpoe_ext.set_preferred_width(50);
        self.rb_mpoe_separate_b.set_selected_boolean(true);
        self.ltf_minimum_tilt_range
            .set_text_double(series_watcher_param::MINIMUM_TILT_RANGE_DEFAULT);
        self.ltf_minimum_number_of_views
            .set_text_int(series_watcher_param::MINIMUM_NUMBER_OF_VIEWS_DEFAULT);
        self.ltf_minimum_age_of_stacks
            .set_text_double(series_watcher_param::MINUMUM_AGE_OF_STACKS_DEFAULT);
        // Emphasize match string: `lMatchString.setFont(...)` with the font size raised
        // by 2.  Fonts are not modelled by the Swing stand-in.
        // screen state
        self.apply_screen_state_to_dialog(self.manager.get_batch_run_tomo_screen_state());
        // Root
        self.pnl_root.add(&self.pnl_header);
        // Header
        self.pnl_header.add(&self.header.get_component());
        self.pnl_header.add(&self.pnl_body);
        // Body
        self.pnl_body.add(&pnl_settings);
        self.pnl_body.add(&pnl_match_pattern_or_ext);
        self.pnl_body.add(&pnl_mpoe_separate_b);
        self.pnl_body.add(&pnl_mpoe_include_combine);
        self.pnl_body.add(&pnl_mpoe_a_only);
        self.pnl_body.add(&pnl_match_string);
        self.pnl_body
            .add(&self.ltf_minimum_tilt_range.get_component());
        self.pnl_body
            .add(&self.ltf_minimum_number_of_views.get_component());
        self.pnl_body
            .add(&self.ltf_minimum_age_of_stacks.get_component());
        // Settings
        pnl_settings.add(&self.cb_dual_axis.get_component());
        pnl_settings.add(&self.cb_two_surfaces.get_component());
        // MatchPatternOrExt
        pnl_match_pattern_or_ext.add(&self.ltf_mpoe_root_name.get_component());
        pnl_match_pattern_or_ext.add(&self.ltf_mpoe_ext.get_component());
        // MpoeIncludeCombine
        pnl_mpoe_include_combine.add(&self.rb_mpoe_include_combine.get_component());
        // MpoeAOnly
        pnl_mpoe_a_only.add(&self.rb_mpoe_a_only.get_component());
        // MpoeSeparateB
        pnl_mpoe_separate_b.add(&self.rb_mpoe_separate_b.get_component());
        // MatchString
        pnl_match_string.add(&self.l_match_string_label);
        pnl_match_string.add(&self.l_match_string);
        //
        self.build_match_pattern_or_ext();
        self.update_display();
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }

    /// Java package-private `setParameters(BatchRunTomoMetaData)`.
    pub fn set_parameters_meta_data(&self, meta_data: &BatchRunTomoMetaData) {
        if meta_data.is_dual_axis_set() {
            self.cb_dual_axis
                .set_selected_boolean(meta_data.is_dual_axis());
        }
        if meta_data.is_two_surfaces_set() {
            self.cb_two_surfaces
                .set_selected_boolean(meta_data.is_two_surfaces());
        }
        self.ltf_mpoe_root_name
            .set_text_string(Some(&meta_data.get_mpoe_root_name()));
        self.ltf_mpoe_ext
            .set_text_string(Some(&meta_data.get_mpoe_ext()));
        self.rb_mpoe_include_combine
            .set_selected_boolean(meta_data.is_mpoe_include_combine());
        self.rb_mpoe_a_only
            .set_selected_boolean(meta_data.is_mpoe_a_only());
        self.rb_mpoe_separate_b
            .set_selected_boolean(meta_data.is_mpoe_separate_b());
        self.build_match_pattern_or_ext();
        if meta_data.is_minimum_tilt_range_set() {
            self.ltf_minimum_tilt_range
                .set_text_string(Some(&meta_data.get_minimum_tilt_range()));
        }
        if meta_data.is_minimum_number_of_views_set() {
            self.ltf_minimum_number_of_views
                .set_text_string(Some(&meta_data.get_minimum_number_of_views()));
        }
        if meta_data.is_minimum_age_of_stacks_set() {
            self.ltf_minimum_age_of_stacks
                .set_text_string(Some(&meta_data.get_minimum_age_of_stacks()));
        }
        self.update_display();
    }

    /// Java package-private `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
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

    /// Java package-private `getParameters(BatchRunTomoMetaData)`.  The dual-axis and
    /// two-surfaces values are only stored when already set, as in the Java (kept
    /// native; see BUGS.md).
    pub fn get_parameters_meta_data(&self, meta_data: &BatchRunTomoMetaData) {
        if meta_data.is_dual_axis_set() {
            meta_data.set_dual_axis(self.cb_dual_axis.is_selected());
        }
        if meta_data.is_two_surfaces_set() {
            meta_data.set_two_surfaces(self.cb_two_surfaces.is_selected());
        }
        meta_data.set_mpoe_root_name(Field::get_text_void(&*self.ltf_mpoe_root_name).as_deref());
        meta_data.set_mpoe_ext(Field::get_text_void(&*self.ltf_mpoe_ext).as_deref());
        meta_data.set_mpoe_include_combine(self.rb_mpoe_include_combine.is_selected());
        meta_data.set_mpoe_a_only(self.rb_mpoe_a_only.is_selected());
        meta_data.set_mpoe_separate_b(self.rb_mpoe_separate_b.is_selected());
        meta_data.set_minimum_tilt_range(
            Field::get_text_void(&*self.ltf_minimum_tilt_range).as_deref(),
        );
        meta_data.set_minimum_number_of_views(
            Field::get_text_void(&*self.ltf_minimum_number_of_views).as_deref(),
        );
        meta_data.set_minimum_age_of_stacks(
            Field::get_text_void(&*self.ltf_minimum_age_of_stacks).as_deref(),
        );
    }

    /// Java public `setParameters(SeriesWatcherParam)`.
    pub fn set_parameters_param(&self, param: &SeriesWatcherParam) {
        if param.is_dual_axis_set() {
            self.cb_dual_axis.set_selected_boolean(param.is_dual_axis());
        }
        if param.is_two_surfaces_set() {
            self.cb_two_surfaces
                .set_selected_boolean(param.is_two_surfaces());
        }
        if param.is_minimum_tilt_range_set() {
            self.ltf_minimum_tilt_range
                .set_text_string(Some(&param.get_minimum_tilt_range()));
        }
        if param.is_minimum_number_of_views_set() {
            self.ltf_minimum_number_of_views
                .set_text_string(Some(&param.get_minimum_number_of_views()));
        }
        if param.is_minimum_age_of_stacks_set() {
            self.ltf_minimum_age_of_stacks
                .set_text_string(Some(&param.get_minimum_age_of_stacks()));
        }
    }

    /// Java package-private `setParameters(UserConfiguration)`.
    pub fn set_parameters_user_configuration(&self, user_configuration: &UserConfiguration) {
        if user_configuration.get_single_axis() {
            self.cb_dual_axis.set_selected_boolean(false);
        }
    }

    /// Java package-private `backupIfChanged(boolean)`.  Check
    /// isDifferentFromCheckpoint on all data entry fields that are loaded from directive
    /// files; returns true if any field's isDifferentFromCheckpoint returned true.
    pub fn backup_if_changed(&self, only_advanced_dataset_dialog: bool) -> bool {
        let mut changed = false;
        if !only_advanced_dataset_dialog {
            if self.cb_dual_axis.is_different_from_checkpoint_boolean(true) {
                self.cb_dual_axis.backup();
                changed = true;
            }
            if self.cb_two_surfaces.is_different_from_checkpoint_boolean(true) {
                self.cb_two_surfaces.backup();
                changed = true;
            }
        }
        changed
    }

    /// Java package-private `applyValues(boolean, boolean, DirectiveFileCollection,
    /// String, boolean)`.
    pub fn apply_values(
        &self,
        init: bool,
        retain_user_values: bool,
        directive_file_collection: &DirectiveFileCollection,
        only_stack_id_dataset_dialog: Option<&str>,
        only_advanced_dataset_dialog: bool,
    ) {
        let only_dataset_dialog = only_stack_id_dataset_dialog.is_some();
        if !only_dataset_dialog {
            // to apply values and highlights, start with a clean slate
            if !init {
                self.cb_dual_axis.clear();
                self.cb_two_surfaces.clear();
            }
            // no default values to apply to table
            // Apply settings values
            etomo_director::INSTANCE.with_user_configuration(|user_configuration| {
                self.set_parameters_user_configuration(user_configuration)
            });
            // Apply the directive collection values
            self.set_values(directive_file_collection, only_advanced_dataset_dialog);
            // checkpoint template/starting batch
            self.cb_dual_axis.checkpoint_void();
            self.cb_two_surfaces.checkpoint_void();
            // If the user wants to retain their values, apply backed up values and then
            // delete them.
            if retain_user_values {
                self.cb_dual_axis.restore_from_backup();
                self.cb_two_surfaces.restore_from_backup();
            } else {
                self.set_parameters_meta_data(self.manager.get_meta_data());
                self.manager.set_series_watcher_parameters();
            }
            self.update_display();
        }
    }

    /// Java package-private `setValues(DirectiveFileCollection, boolean)`.  Called when
    /// the template has changed; only changes fields that exist in the directive file
    /// collection.
    pub fn set_values(
        &self,
        directive_file_collection: &DirectiveFileCollection,
        only_advanced_dataset_dialog: bool,
    ) {
        if !only_advanced_dataset_dialog {
            batch_tool::set_boolean_value_missing(
                Some(&*self.cb_dual_axis),
                directive_file_collection,
                false,
                false,
                None,
            );
            batch_tool::set_boolean_value_from_selected_text(
                Some(&*self.cb_two_surfaces),
                Some(directive_def::SURFACES_TO_ANALYZE_DEFAULT),
                directive_file_collection,
                false,
                None,
            );
            self.update_display();
        }
    }

    /// Java public `getParameters(SeriesWatcherParam, boolean)`.
    pub fn get_parameters_param(&self, param: &mut SeriesWatcherParam, do_validation: bool) -> bool {
        // try
        let result = (|| {
            // Fields that are only disabled when serieswatcher is not in use are always
            // saved since the com file will be saved either way.
            param.set_dual_axis(self.cb_dual_axis.is_selected());
            param.set_two_surfaces(self.cb_two_surfaces.is_selected());
            param.set_match_pattern_or_ext(Some(&self.l_match_string.get_text()));
            let displayer = Some(self.header.get_advanced_field_displayer());
            param.set_minimum_tilt_range(
                Field::get_text_boolean_field_displayer(
                    &*self.ltf_minimum_tilt_range,
                    do_validation,
                    displayer.clone(),
                )?
                .as_deref(),
            );
            param.set_minimum_number_of_views(
                Field::get_text_boolean_field_displayer(
                    &*self.ltf_minimum_number_of_views,
                    do_validation,
                    displayer.clone(),
                )?
                .as_deref(),
            );
            param.set_minimum_age_of_stacks(
                Field::get_text_boolean_field_displayer(
                    &*self.ltf_minimum_age_of_stacks,
                    do_validation,
                    displayer,
                )?
                .as_deref(),
            );
            Ok::<(), crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException>(())
        })();
        // catch (final FieldValidationFailedException e)
        result.is_ok()
    }

    /// Java package-private `getAxisType()`.
    pub fn get_axis_type(&self) -> AxisType {
        if self.cb_dual_axis.is_selected() {
            return AxisType::DualAxis;
        }
        AxisType::SingleAxis
    }

    /// Java package-private `isTwoSurfaces()`.
    pub fn is_two_surfaces(&self) -> bool {
        self.cb_two_surfaces.is_selected()
    }

    /// Java package-private `isAOnly()`.
    pub fn is_a_only(&self) -> bool {
        self.rb_mpoe_a_only.is_enabled() && self.rb_mpoe_a_only.is_selected()
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        let series_watcher_on = self
            .parent
            .upgrade()
            .is_some_and(|parent| parent.is_series_watcher_on());
        let dual_axis = self.cb_dual_axis.is_selected();
        self.cb_dual_axis.set_enabled(series_watcher_on);
        self.cb_two_surfaces.set_enabled(series_watcher_on);
        self.ltf_mpoe_root_name.set_enabled(series_watcher_on);
        self.ltf_mpoe_ext.set_enabled(series_watcher_on);
        self.rb_mpoe_include_combine
            .set_enabled(series_watcher_on && dual_axis);
        self.rb_mpoe_a_only.set_enabled(series_watcher_on && dual_axis);
        self.rb_mpoe_separate_b
            .set_enabled(series_watcher_on && dual_axis);
        self.ltf_minimum_tilt_range.set_enabled(series_watcher_on);
        self.ltf_minimum_number_of_views.set_enabled(series_watcher_on);
        self.ltf_minimum_age_of_stacks.set_enabled(series_watcher_on);
        // Update advanced
        let advanced = self.advanced.get();
        self.ltf_minimum_tilt_range.set_visible(advanced);
        self.ltf_minimum_number_of_views.set_visible(advanced);
        self.ltf_minimum_age_of_stacks.set_visible(advanced);
    }

    /// Java `focusGained(FocusEvent)`: empty.
    pub fn focus_gained(&self) {}

    /// Java `focusLost(FocusEvent)`.  Assuming that only the MatchPatternOrExt text
    /// fields added a focus listener.
    pub fn focus_lost(&self) {
        self.build_match_pattern_or_ext();
    }

    /// Java private `buildMatchPatternOrExt()`.  Builds matchPatternOrExt from the root
    /// and extension, and information on whether the B axis is available.
    fn build_match_pattern_or_ext(&self) {
        let mut pattern = String::new();
        // Build left side of pattern.
        let root_name = Field::get_text_void(&*self.ltf_mpoe_root_name);
        if let Some(root_name) = &root_name {
            pattern.push_str(root_name);
            if !utilities::contains_wildcard(Some(root_name)) {
                pattern.push_str(utilities::ZERO_OR_MORE_WILDCARD);
            }
        } else {
            // No root name.
            pattern.push_str(utilities::ZERO_OR_MORE_WILDCARD);
        }
        if self.cb_dual_axis.is_selected() {
            if self.rb_mpoe_include_combine.is_selected() {
                pattern.push_str(&AxisID::Second.get_extension());
            } else if self.rb_mpoe_a_only.is_selected() {
                pattern.push_str(&AxisID::First.get_extension());
            } else if self.rb_mpoe_separate_b.is_selected() {
                pattern.push_str(
                    &utilities::get_regular_expression_class(Some(&format!(
                        "{}{}",
                        AxisID::First.get_extension(),
                        AxisID::Second.get_extension()
                    )))
                    .unwrap_or_else(|| "null".to_owned()),
                );
            }
        }
        // Build right side of pattern.
        let delimiter = ".";
        let default_ext = "mrc";
        pattern.push_str(delimiter);
        match Field::get_text_void(&*self.ltf_mpoe_ext) {
            None => pattern.push_str(default_ext),
            Some(mut ext) => {
                if ext.starts_with(delimiter) {
                    // Get rid of the delimiter to make things simpler.
                    ext = ext[1..].to_owned();
                }
                if ext.is_empty() {
                    pattern.push_str(default_ext);
                } else {
                    pattern.push_str(&ext);
                }
            }
        }
        self.l_match_string.set_text(&pattern);
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let action_listener = self.get_action_listener();
        let this = self.this.clone();
        let focus_listener: FocusListener = Rc::new(move |event: &FocusEvent| {
            if let Some(panel) = this.upgrade() {
                if event.gained {
                    panel.focus_gained();
                } else {
                    panel.focus_lost();
                }
            }
        });
        self.cb_dual_axis
            .add_action_listener(Some(action_listener.clone()));
        // MatchPatternOrExt
        self.ltf_mpoe_root_name
            .add_action_listener(action_listener.clone());
        self.ltf_mpoe_root_name
            .add_focus_listener(focus_listener.clone());
        self.ltf_mpoe_ext.add_action_listener(action_listener.clone());
        self.ltf_mpoe_ext.add_focus_listener(focus_listener);
        self.rb_mpoe_include_combine
            .add_action_listener(action_listener.clone());
        self.rb_mpoe_a_only
            .add_action_listener(action_listener.clone());
        self.rb_mpoe_separate_b.add_action_listener(action_listener);
        //
        // If focusListeners are required for something other then MatchPatternOrExt, an
        // inner class should be used.
    }

    /// This panel as the `ActionListener` Java registers as `this` (the dialog also
    /// adds it to its series watcher checkbox).
    pub fn get_action_listener(&self) -> ActionListener {
        let this = self.this.clone();
        Rc::new(move |event: &ActionEvent| {
            if let Some(panel) = this.upgrade() {
                panel.action_performed(Some(event));
            }
        })
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        let Some(event) = event else {
            return;
        };
        let Some(command) = event.get_action_command() else {
            return;
        };
        let command = Some(command.to_owned());
        if command == self.cb_dual_axis.get_action_command()
            || command.as_deref() == Some(self.ltf_mpoe_root_name.get_action_command().as_str())
            || command.as_deref() == Some(self.ltf_mpoe_ext.get_action_command().as_str())
            || command == self.rb_mpoe_include_combine.get_action_command()
            || command == self.rb_mpoe_a_only.get_action_command()
            || command == self.rb_mpoe_separate_b.get_action_command()
        {
            self.build_match_pattern_or_ext();
        }
        self.update_display();
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        Field::set_tool_tip_text(&*self.ltf_mpoe_root_name, Some(
            "Enter letters to match in the root name, or leave blank to match all eligible files.  Wild cards can be used: '*' to match any set of characters, '?' to match one character, or [list] to match any one character in the list, which can include ranges like a-z and 0-9.  If NO wild cards are used, '*' will be added after the entry, otherwise not. Do not include  axis letter for dual axis; if entry ends with a or b that will be considered part of the data set root name.",
        ));
        Field::set_tool_tip_text(
            &*self.ltf_mpoe_ext,
            Some("Enter the filename extension without the '.'."),
        );
        self.rb_mpoe_include_combine.set_tool_tip_text_string(Some(
            "Wait for B axis to be present; if A was already processed, runs B axis and combine; if A is still present, does both axes and combine in one run.",
        ));
        self.rb_mpoe_a_only.set_tool_tip_text_string(Some(
            "Start data set with A axis only when it appears and ignore B axis stack",
        ));
        self.rb_mpoe_separate_b.set_tool_tip_text_string(Some(
            "Start data set with A axis when it appears; do a second run to finish data set when B axis stack appears",
        ));
        let tooltip =
            "This is the full string being entered with the -match option to Serieswatcher.";
        self.l_match_string_label.set_tool_tip_text(Some(tooltip));
        self.l_match_string.set_tool_tip_text(Some(tooltip));

        let manager: &'static dyn BaseManager = self.manager;
        // try
        match unsafe {
            autodoc_factory::get_instance(
                Some(manager),
                Some(autodoc_factory::SERIES_WATCHER),
                self.axis_id,
                false,
            )
        } {
            Ok(autodoc) => {
                let autodoc = unsafe { autodoc.as_ref() }
                    .map(|autodoc| autodoc as &dyn ReadOnlyAutodoc);
                self.cb_dual_axis.set_tool_tip_text_string(
                    etomo_autodoc::get_tooltip(autodoc, Some(series_watcher_param::DUAL_AXIS_KEY))
                        .as_deref(),
                );
                self.cb_two_surfaces.set_tool_tip_text_string(
                    etomo_autodoc::get_tooltip(
                        autodoc,
                        Some(series_watcher_param::TWO_SURFACES_KEY),
                    )
                    .as_deref(),
                );
                Field::set_tool_tip_text(
                    &*self.ltf_minimum_tilt_range,
                    etomo_autodoc::get_tooltip(
                        autodoc,
                        Some(series_watcher_param::MINIMUM_TILT_RANGE_KEY),
                    )
                    .as_deref(),
                );
                Field::set_tool_tip_text(
                    &*self.ltf_minimum_number_of_views,
                    etomo_autodoc::get_tooltip(
                        autodoc,
                        Some(series_watcher_param::MINIMUM_NUMBER_OF_VIEWS_KEY),
                    )
                    .as_deref(),
                );
                Field::set_tool_tip_text(
                    &*self.ltf_minimum_age_of_stacks,
                    etomo_autodoc::get_tooltip(
                        autodoc,
                        Some(series_watcher_param::MINIMUM_AGE_OF_STACKS_KEY),
                    )
                    .as_deref(),
                );
            }
            Err(LogFileError::Lock(_)) => {}
            Err(e) => eprintln!("{e}"),
        }
    }

    /// Java public `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }
}

impl Expandable for SeriesWatcherPanel {
    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_advanced_basic(button) {
            self.advanced.set(button.is_expanded());
            self.update_display();
            let manager: &'static dyn BaseManager = self.manager;
            ui_harness::with(|harness| {
                harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
            });
        }
    }

    /// Java `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}
}
