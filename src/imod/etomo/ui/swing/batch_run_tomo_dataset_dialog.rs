//! `IMOD/Etomo/src/etomo/ui/swing/BatchRunTomoDatasetDialog.java`.
//!
//! The Swing frame, layouts, menus, 3dmod launch, manager persistence, and
//! autodoc I/O are intentionally explicit boundaries.  This module retains the
//! dataset-dialog's source-owned controls and controller rules; adapters supply
//! those Java-boundary operations.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::imod::etomo::ui::field_type::FieldType;

use super::batch_run_tomo_step_panel::BatchRunTomoStatus;
use super::check_box::CheckBox;
use super::labeled_text_field::LabeledTextField;
use super::radio_button::{RadioButton, RadioButtonGroup};

pub const LENGTH_OF_PIECES_DEFAULT: &str = "-1";
pub const SCALE_TO_INTEGER_VALUE: &str = "-20000,20000";
pub const CTF_RANGE_DEFAULT: &str = "0.0";
pub const CTF_STEP_DEFAULT: &str = "0.0";
pub const CTF_RANGE_FIT_EVERY_IMAGE: &str = "1";
pub const CTF_STEP_FIT_EVERY_IMAGE: &str = "0";
pub const TARGET_NUMBER_OF_BEADS_LABEL: &str = "Target number of beads: ";

/// Java `BatchRunTomoRow` / `BatchRunTomoDialog` callbacks reached by this unit.
pub trait BatchRunTomoDatasetDialogBoundary {
    fn has_dual(&self) -> bool;
    fn is_parallel_processing(&self) -> bool;
    fn imod_stack(&mut self, model_file: Option<&str>) -> Option<String>;
    fn delete_dataset(&mut self);
    fn set_dataset_table_visible(&mut self, visible: bool);
    fn display_dataset_tab(&mut self);
    fn save_batch_run_tomo_dialog(&mut self, _parallel: bool) {}
    fn init_dialog(&mut self, _stack_id: Option<&str>, _advanced: bool) {}
}

/// Source-visible frame/JPanel state; actual Swing construction is a GUI boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoDatasetDialogLayout {
    pub frame_visible: bool,
    pub root_body_visible: bool,
    pub postprocessing_body_visible: bool,
    pub basic_visible: bool,
    pub advanced_visible: bool,
    pub model_button_enabled: bool,
    pub listener_count: usize,
    pub pack_count: usize,
    pub title: Option<String>,
    pub tooltips_initialized: bool,
}

/// Rust representation of the Java metadata exchange object at this unit's boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRunTomoDatasetMetaData {
    pub values: BTreeMap<String, String>,
    pub booleans: BTreeMap<String, bool>,
    pub prenewst_bin_by_factor: i32,
    pub preblend_bin_by_factor: i32,
}

/// Java `BatchRunTomoDatasetDialog`, with every data-entry control retained.
pub struct BatchRunTomoDatasetDialog {
    pub pnl_root: BatchRunTomoDatasetDialogLayout,
    pub cb_remove_xrays: CheckBox,
    pub cb_enable_stretching: CheckBox,
    pub cb_local_alignments: CheckBox,
    pub cb_length_of_pieces: CheckBox,
    pub cb_correct_ctf: CheckBox,
    pub cb_do_backproj_also: CheckBox,
    pub cb_fake_sirt_iterations: CheckBox,
    pub cb_use_sirt: CheckBox,
    pub cb_scale_to_integer: CheckBox,
    pub cb_sample_type: CheckBox,
    pub cb_do_trimvol: CheckBox,
    /// Java `ctfFindSecAddThickness` check-box half.
    pub cb_find_sec_add_thickness: CheckBox,
    /// Java `ctfScaleFromZ` check-box half.
    pub cb_scale_from_z: CheckBox,
    pub cb_erase_gold: CheckBox,
    pub cb_has_gold_beads: CheckBox,
    pub cb_tune_fitting_and_sampling: CheckBox,
    pub rb_tracking_method_seed: RadioButton,
    pub rb_tracking_method_raptor: RadioButton,
    pub rb_tracking_method_patch_tracking: RadioButton,
    pub rb_fiducialless: RadioButton,
    pub rb_fit_every_image: RadioButton,
    pub rtf_auto_fit_range_and_step: RadioButton,
    pub rtf_thickness: RadioButton,
    pub rtf_binned_thickness: RadioButton,
    pub rb_fallback_and_extra_thickness: RadioButton,
    pub rb_erase_gold_fid: RadioButton,
    pub rb_erase_gold_3d: RadioButton,
    pub rb_sample_type_plastic_section: RadioButton,
    pub rb_sample_type_cryo: RadioButton,
    pub ltf_gold: LabeledTextField,
    pub ltf_target_number_of_beads: LabeledTextField,
    pub ltf_number_of_markers: LabeledTextField,
    pub ltf_size_of_patches_x_and_y: LabeledTextField,
    pub ltf_scan_defocus_range: LabeledTextField,
    pub ltf_defocus: LabeledTextField,
    pub ltf_auto_fit_range_and_step: LabeledTextField,
    pub ltf_auto_fit_step: LabeledTextField,
    pub ltf_fake_sirt_iterations: LabeledTextField,
    pub ltf_leave_iterations: LabeledTextField,
    pub ltf_thickness: LabeledTextField,
    pub ltf_binned_thickness: LabeledTextField,
    pub ltf_extra_thickness: LabeledTextField,
    pub ltf_fallback_thickness: LabeledTextField,
    pub ltf_model_file: LabeledTextField,
    /// Java `ftfDistort`; filesystem browsing is an explicit GUI boundary.
    pub ltf_distort: Option<LabeledTextField>,
    /// Java `ftfGradient`; filesystem browsing is an explicit GUI boundary.
    pub ltf_gradient: Option<LabeledTextField>,
    /// Java `lsBinByFactor`; the spinner widget is a GUI boundary and its
    /// source-visible integer text is retained here.
    pub ltf_bin_by_factor: LabeledTextField,
    pub ltf_prenewst_bin_by_factor: LabeledTextField,
    pub ltf_preblend_bin_by_factor: LabeledTextField,
    pub ctf_find_sec_add_thickness: LabeledTextField,
    pub ctf_scale_from_z: LabeledTextField,
    pub ltf_gold_erasing_thickness: LabeledTextField,
    pub ltf_positioning_thickness: LabeledTextField,
    pub ltf_positioning_gold: LabeledTextField,
    pub field_list: Vec<String>,
    pub template_values: BTreeMap<String, String>,
    pub basic_directives: BTreeSet<String>,
    pub dataset_file: Option<PathBuf>,
    pub stack_id: Option<String>,
    pub length_of_pieces: Option<String>,
    pub status: BatchRunTomoStatus,
    pub global: bool,
    pub from_saved: bool,
    pub empty_table: bool,
    pub row_exists: bool,
    pub row_dual: bool,
    pub row_parallel_processing: bool,
    pub fiducial_model_mode: bool,
    pub shift_basic_button: bool,
    pub advanced: bool,
    pub advanced_dialog_exists: bool,
    pub directives_changed: bool,
    pub frame_state_normal: bool,
}

impl BatchRunTomoDatasetDialog {
    /// Java private constructor `BatchRunTomoDatasetDialog(...)`.
    pub fn new(
        dataset_file: Option<PathBuf>,
        global: bool,
        stack_id: Option<String>,
        from_saved: bool,
    ) -> Self {
        let tracking = std::rc::Rc::new(std::cell::RefCell::new(RadioButtonGroup::new()));
        let autofit = std::rc::Rc::new(std::cell::RefCell::new(RadioButtonGroup::new()));
        let thickness = std::rc::Rc::new(std::cell::RefCell::new(RadioButtonGroup::new()));
        let erase_gold = std::rc::Rc::new(std::cell::RefCell::new(RadioButtonGroup::new()));
        let sample = std::rc::Rc::new(std::cell::RefCell::new(RadioButtonGroup::new()));
        let mut value = Self {
            pnl_root: BatchRunTomoDatasetDialogLayout {
                root_body_visible: true,
                basic_visible: true,
                ..Default::default()
            },
            cb_remove_xrays: CheckBox::new_with_text("Remove X-rays"),
            cb_enable_stretching: CheckBox::new_with_text(
                "Enable distortion (stretching) in alignment",
            ),
            cb_local_alignments: CheckBox::new_with_text("Use local alignments"),
            cb_length_of_pieces: CheckBox::new_with_text("Break contours into pieces"),
            cb_correct_ctf: CheckBox::new_with_text("Correct CTF"),
            cb_do_backproj_also: CheckBox::new_with_text("R-weighted backprojection"),
            cb_fake_sirt_iterations: CheckBox::new_with_text("SIRT-like filter"),
            cb_use_sirt: CheckBox::new_with_text("SIRT"),
            cb_scale_to_integer: CheckBox::new_with_text("Scale to integers"),
            cb_sample_type: CheckBox::new_with_text("Do positioning for:"),
            cb_do_trimvol: CheckBox::new_with_text("Postprocess with trimvol"),
            cb_find_sec_add_thickness: CheckBox::new_with_text(
                "Find plastic section limits and add: ",
            ),
            cb_scale_from_z: CheckBox::new_with_text("Fraction of Z slices to analyze:"),
            cb_erase_gold: CheckBox::new_with_text("Erase gold"),
            cb_has_gold_beads: CheckBox::new_with_text("Sample has gold beads"),
            cb_tune_fitting_and_sampling: CheckBox::new_with_text("Autotune"),
            rb_tracking_method_seed: RadioButton::new_in_group(
                "Autoseed and track",
                tracking.clone(),
            ),
            rb_tracking_method_raptor: RadioButton::new_in_group(
                "Raptor and track",
                tracking.clone(),
            ),
            rb_tracking_method_patch_tracking: RadioButton::new_in_group(
                "Patch tracking",
                tracking.clone(),
            ),
            rb_fiducialless: RadioButton::new_in_group("Coarse alignment only", tracking),
            rb_fit_every_image: RadioButton::new_in_group("Fit every image", autofit.clone()),
            rtf_auto_fit_range_and_step: RadioButton::new_in_group("Autofit range", autofit),
            rtf_thickness: RadioButton::new_in_group(
                "Thickness total (unbinned pixels): ",
                thickness.clone(),
            ),
            rtf_binned_thickness: RadioButton::new_in_group(
                "Thickness total (binned pixels): ",
                thickness.clone(),
            ),
            rb_fallback_and_extra_thickness: RadioButton::new_in_group(
                "Calculated thickness (unbinned pixels):",
                thickness,
            ),
            rb_erase_gold_fid: RadioButton::new_in_group("Use fiducial model", erase_gold.clone()),
            rb_erase_gold_3d: RadioButton::new_in_group("Find beads in 3D", erase_gold),
            rb_sample_type_plastic_section: RadioButton::new_in_group(
                "Plastic section",
                sample.clone(),
            ),
            rb_sample_type_cryo: RadioButton::new_in_group("Cryo sample", sample),
            ltf_gold: LabeledTextField::new(FieldType::FloatingPoint, "Bead size (nm): "),
            ltf_target_number_of_beads: LabeledTextField::new(
                FieldType::Integer,
                TARGET_NUMBER_OF_BEADS_LABEL,
            ),
            ltf_number_of_markers: LabeledTextField::new(
                FieldType::Integer,
                TARGET_NUMBER_OF_BEADS_LABEL,
            ),
            ltf_size_of_patches_x_and_y: LabeledTextField::new(
                FieldType::IntegerPair,
                "Patch tracking size: ",
            ),
            ltf_scan_defocus_range: LabeledTextField::new(
                FieldType::FloatingPointPair,
                "Defocus range to scan: ",
            ),
            ltf_defocus: LabeledTextField::new(FieldType::FloatingPoint, "Defocus: "),
            ltf_auto_fit_range_and_step: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Autofit range ",
            ),
            ltf_auto_fit_step: LabeledTextField::new(FieldType::FloatingPoint, " and step "),
            ltf_fake_sirt_iterations: LabeledTextField::new(
                FieldType::Integer,
                "Equivalent iterations: ",
            ),
            ltf_leave_iterations: LabeledTextField::new(FieldType::String, "Leave iterations: "),
            ltf_thickness: LabeledTextField::new(
                FieldType::Integer,
                "Thickness total (unbinned pixels): ",
            ),
            ltf_binned_thickness: LabeledTextField::new(
                FieldType::Integer,
                "Thickness total (binned pixels): ",
            ),
            ltf_extra_thickness: LabeledTextField::new(
                FieldType::Integer,
                "          Plus (optional): ",
            ),
            ltf_fallback_thickness: LabeledTextField::new(
                FieldType::Integer,
                "          With fallback: ",
            ),
            ltf_model_file: LabeledTextField::new(FieldType::String, "Manual replacement model: "),
            ltf_distort: None,
            ltf_gradient: None,
            ltf_bin_by_factor: LabeledTextField::new(FieldType::Integer, "Aligned stack binning: "),
            ltf_prenewst_bin_by_factor: LabeledTextField::new(
                FieldType::Integer,
                "Coarse aligned stack binning - single frame: ",
            ),
            ltf_preblend_bin_by_factor: LabeledTextField::new(FieldType::Integer, "Montage: "),
            ctf_find_sec_add_thickness: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Find plastic section limits and add: ",
            ),
            ctf_scale_from_z: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Fraction of Z slices to analyze:",
            ),
            ltf_gold_erasing_thickness: LabeledTextField::new(
                FieldType::Integer,
                "Tomogram thickness (pixels): ",
            ),
            ltf_positioning_thickness: LabeledTextField::new(
                FieldType::Integer,
                "Tomogram thickness: ",
            ),
            ltf_positioning_gold: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Bead size (nm): ",
            ),
            field_list: Vec::new(),
            template_values: BTreeMap::new(),
            basic_directives: BTreeSet::new(),
            dataset_file,
            stack_id,
            length_of_pieces: None,
            status: BatchRunTomoStatus::DEFAULT,
            global,
            from_saved,
            empty_table: global,
            row_exists: !global,
            row_dual: false,
            row_parallel_processing: false,
            fiducial_model_mode: true,
            shift_basic_button: false,
            advanced: false,
            advanced_dialog_exists: false,
            directives_changed: false,
            frame_state_normal: false,
        };
        value.create_panel(global);
        value.set_tooltips();
        value.add_listeners();
        value
    }

    /// Java `getGlobalInstance(...)`.
    pub fn get_global_instance() -> Self {
        Self::new(None, true, None, true)
    }
    /// Java `getRowInstance(...)`.
    pub fn get_row_instance(dataset_file: PathBuf, stack_id: String) -> Self {
        let mut value = Self::new(Some(dataset_file), false, Some(stack_id), false);
        value.set_visible(true);
        value
    }
    /// Java `getSavedRowInstance(...)`.
    pub fn get_saved_row_instance(dataset_file: PathBuf, stack_id: String) -> Self {
        Self::new(Some(dataset_file), false, Some(stack_id), true)
    }

    pub fn is_find_sec_add_thickness_set(&self) -> bool {
        self.cb_find_sec_add_thickness.is_selected() && !self.ctf_find_sec_add_thickness.is_empty()
    }
    pub fn is_scale_from_z_set(&self) -> bool {
        self.cb_scale_from_z.is_selected() && !self.ctf_scale_from_z.is_empty()
    }
    pub fn has_dual(&self) -> bool {
        self.row_dual
    }

    /// Java `createPanel(boolean)`: layout creation remains represented by `pnl_root`.
    fn create_panel(&mut self, global: bool) {
        self.rb_tracking_method_seed.set_selected(true);
        self.rb_erase_gold_3d.set_selected(true);
        self.rtf_auto_fit_range_and_step.set_selected(true);
        self.cb_do_backproj_also.set_selected(true);
        self.ltf_gold.set_required(true);
        self.ltf_target_number_of_beads.set_required(true);
        self.ltf_number_of_markers.set_required(true);
        self.ltf_size_of_patches_x_and_y.set_required(true);
        self.ltf_fake_sirt_iterations.set_required(true);
        self.ltf_leave_iterations.set_required(true);
        self.ltf_fallback_thickness.set_required(true);
        self.ltf_thickness.set_required(true);
        self.ltf_binned_thickness.set_required(true);
        self.ctf_find_sec_add_thickness.set_required(true);
        self.ctf_scale_from_z.set_text("0.5");
        self.field_list = vec![
            "DISTORT".into(),
            "GRADIENT".into(),
            "REMOVE_XRAYS".into(),
            "MODEL_FILE".into(),
            "TRACKING_METHOD".into(),
            "SEEDING_METHOD".into(),
            "FIDUCIALLESS".into(),
            "GOLD".into(),
            "TARGET_NUMBER_OF_BEADS".into(),
            "NUMBER_OF_MARKERS".into(),
            "SIZE_OF_PATCHES_X_AND_Y".into(),
            "LENGTH_OF_PIECES".into(),
            "ENABLE_STRETCHING".into(),
            "LOCAL_ALIGNMENTS".into(),
            "BIN_BY_FACTOR_FOR_ALIGNED_STACK".into(),
            "CORRECT_CTF".into(),
            "SCAN_DEFOCUS_RANGE".into(),
            "DEFOCUS".into(),
            "AUTO_FIT_RANGE_AND_STEP".into(),
            "DO_BACKPROJ_ALSO".into(),
            "FAKE_SIRT_ITERATIONS".into(),
            "USE_SIRT".into(),
            "LEAVE_ITERATIONS".into(),
            "SCALE_TO_INTEGER".into(),
            "THICKNESS_FOR_TILT".into(),
            "BINNED_THICKNESS".into(),
            "FALLBACK_THICKNESS".into(),
            "EXTRA_THICKNESS".into(),
            "BIN_BY_FACTOR_FOR_PRENEWST".into(),
            "BIN_BY_FACTOR_FOR_PREBLEND".into(),
            "DO_TRIMVOL".into(),
            "FIND_SEC_ADD_THICKNESS".into(),
            "SCALE_FROM_Z".into(),
            "ERASE_GOLD".into(),
            "THICKNESS_FOR_GOLD_ERASING".into(),
            "SAMPLE_TYPE".into(),
            "THICKNESS_FOR_POSITIONING".into(),
            "HAS_GOLD_BEADS".into(),
            "TUNE_FITTING_AND_SAMPLING".into(),
        ];
        self.basic_directives
            .extend(self.field_list.iter().cloned());
        self.pnl_root.basic_visible = !self.advanced;
        self.pnl_root.advanced_visible = self.advanced;
        if !global {
            self.pnl_root.title = self
                .dataset_file
                .as_ref()
                .and_then(|file| file.file_stem())
                .map(|name| name.to_string_lossy().into_owned());
        }
        self.update_display();
        self.update_advanced(false, self.from_saved);
        self.status_changed(self.status);
        self.pack();
    }

    fn update_gold_panel(&mut self) {
        self.update_display();
        self.pack();
    }
    fn setup_field(&mut self, directive_def: &str) {
        self.field_list.push(directive_def.into());
        self.basic_directives.insert(directive_def.into());
    }
    pub fn is_advanced_dialog_exists(&self) -> bool {
        self.advanced_dialog_exists
    }
    pub fn is_tracking_method_seed(&self) -> bool {
        self.rb_tracking_method_seed.is_selected()
    }
    pub fn get_preferred_width(&self) -> i32 {
        self.ltf_model_file.columns + 5 + 1
    }
    pub fn set_visible(&mut self, visible: bool) {
        if !self.global {
            self.pnl_root.frame_visible = visible;
        }
    }
    fn add_listeners(&mut self) {
        self.pnl_root.listener_count = 23;
    }
    pub fn get_component(&self) -> &BatchRunTomoDatasetDialogLayout {
        &self.pnl_root
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn expand(&mut self, root: bool, expanded: bool) {
        if root {
            self.pnl_root.root_body_visible = expanded;
        } else {
            self.pnl_root.postprocessing_body_visible = expanded;
        }
        self.pack();
    }
    pub fn expand_global(&mut self) {}
    fn pack(&mut self) {
        self.pnl_root.pack_count += 1;
    }
    pub fn validate(&mut self) -> bool {
        if self.rb_sample_type_cryo.is_selected() && self.ltf_positioning_thickness.is_empty() {
            self.display();
            return false;
        }
        true
    }
    pub fn validate_surfaces(&mut self, surfaces_to_analyze_2: bool) -> bool {
        if !surfaces_to_analyze_2
            && self.ltf_gold_erasing_thickness.is_enabled()
            && self.ltf_gold_erasing_thickness.is_empty()
        {
            self.display();
            return false;
        }
        true
    }
    pub fn is_parallel_processing(&self) -> bool {
        self.row_parallel_processing
    }

    fn update_advanced(&mut self, advanced: bool, init: bool) {
        if !init && advanced && self.stack_id.is_some() {
            self.directives_changed = true;
        }
        if !self.advanced_dialog_exists
            && ((init && (self.stack_id.is_none() || self.advanced))
                || (!init && advanced && self.stack_id.is_some()))
        {
            self.advanced_dialog_exists = true;
        }
        self.advanced = advanced;
        self.pnl_root.basic_visible = !advanced;
        self.pnl_root.advanced_visible = advanced;
        self.pack();
    }
    pub fn get_directives_dialog(&self) -> bool {
        self.advanced_dialog_exists
    }

    fn update_display(&mut self) {
        let remove_xrays = self.cb_remove_xrays.is_selected();
        self.ltf_model_file.set_enabled(remove_xrays);
        self.pnl_root.model_button_enabled = remove_xrays && !self.empty_table;
        let fiducialless = self.rb_fiducialless.is_selected();
        self.cb_enable_stretching.set_enabled(!fiducialless);
        self.cb_local_alignments.set_enabled(!fiducialless);
        let seed = self.rb_tracking_method_seed.is_selected();
        let raptor = self.rb_tracking_method_raptor.is_selected();
        let bead_tracking = seed || raptor;
        self.ltf_gold.set_enabled(bead_tracking);
        self.ltf_target_number_of_beads.set_enabled(seed);
        self.ltf_number_of_markers.set_enabled(raptor);
        let sample_type = self.cb_sample_type.is_selected();
        self.rb_sample_type_plastic_section.set_enabled(sample_type);
        self.rb_sample_type_cryo.set_enabled(sample_type);
        let cryo = sample_type && self.rb_sample_type_cryo.is_selected();
        self.cb_has_gold_beads.set_enabled(!bead_tracking && cryo);
        self.ltf_positioning_gold
            .set_enabled(!bead_tracking && cryo && self.cb_has_gold_beads.is_selected());
        if bead_tracking {
            self.cb_has_gold_beads.set_selected(true);
        }
        if self.fiducial_model_mode {
            self.ltf_positioning_gold
                .set_text(&self.ltf_gold.get_text());
        } else {
            self.ltf_gold
                .set_text(&self.ltf_positioning_gold.get_text());
        }
        self.fiducial_model_mode = bead_tracking;
        let patch = self.rb_tracking_method_patch_tracking.is_selected();
        self.ltf_size_of_patches_x_and_y.set_enabled(patch);
        self.cb_length_of_pieces.set_enabled(patch);
        let ctf = self.cb_correct_ctf.is_selected();
        self.ltf_scan_defocus_range.set_enabled(ctf);
        self.ltf_defocus.set_enabled(ctf);
        self.cb_tune_fitting_and_sampling.set_enabled(ctf);
        self.rtf_auto_fit_range_and_step.set_enabled(ctf);
        self.rb_fit_every_image.set_enabled(ctf);
        self.ltf_auto_fit_step
            .set_enabled(ctf && self.rtf_auto_fit_range_and_step.is_selected());
        let use_sirt = self.cb_use_sirt.is_selected();
        self.ltf_fake_sirt_iterations
            .set_enabled(self.cb_fake_sirt_iterations.is_selected());
        self.ltf_fake_sirt_iterations.set_visible(!use_sirt);
        self.ltf_leave_iterations.set_visible(use_sirt);
        self.ltf_leave_iterations.set_enabled(use_sirt);
        self.cb_scale_to_integer.set_enabled(use_sirt);
        let fallback = self.rb_fallback_and_extra_thickness.is_selected();
        self.ltf_extra_thickness.set_enabled(fallback);
        self.ltf_fallback_thickness.set_enabled(fallback);
        let trimvol = self.cb_do_trimvol.is_selected();
        self.cb_find_sec_add_thickness.set_enabled(trimvol);
        self.ctf_find_sec_add_thickness.set_enabled(trimvol);
        self.cb_scale_from_z.set_enabled(trimvol);
        self.ctf_scale_from_z.set_enabled(trimvol);
        let erase = self.cb_erase_gold.is_selected();
        self.rb_erase_gold_fid
            .set_enabled(erase && !fiducialless && !patch);
        self.rb_erase_gold_3d.set_enabled(erase);
        if erase && (fiducialless || patch) {
            self.rb_erase_gold_3d.set_selected(true);
        }
        self.ltf_gold_erasing_thickness
            .set_enabled(erase && self.rb_erase_gold_3d.is_selected());
    }

    pub fn status_changed(&mut self, status: BatchRunTomoStatus) {
        self.status = status;
        let open = matches!(
            status,
            BatchRunTomoStatus::Open | BatchRunTomoStatus::Done | BatchRunTomoStatus::Failed
        );
        for value in [
            &mut self.cb_remove_xrays,
            &mut self.cb_enable_stretching,
            &mut self.cb_local_alignments,
            &mut self.cb_length_of_pieces,
            &mut self.cb_correct_ctf,
            &mut self.cb_do_backproj_also,
            &mut self.cb_fake_sirt_iterations,
            &mut self.cb_use_sirt,
            &mut self.cb_scale_to_integer,
            &mut self.cb_sample_type,
            &mut self.cb_do_trimvol,
            &mut self.cb_find_sec_add_thickness,
            &mut self.cb_scale_from_z,
            &mut self.cb_erase_gold,
            &mut self.cb_has_gold_beads,
            &mut self.cb_tune_fitting_and_sampling,
        ] {
            value.set_editable(open);
        }
    }
    pub fn status_changed_frame(&mut self, single: Option<bool>, montage: Option<bool>) {
        if let Some(enabled) = single {
            self.ltf_prenewst_bin_by_factor.set_enabled(enabled);
        }
        if let Some(enabled) = montage {
            self.ltf_preblend_bin_by_factor.set_enabled(enabled);
        }
    }
    pub fn backup_if_changed(&mut self, only_advanced_dataset_dialog: bool) -> bool {
        if only_advanced_dataset_dialog {
            return self.advanced_dialog_exists && self.directives_changed;
        }
        let changed = self.directives_changed;
        self.directives_changed = false;
        changed
    }
    pub fn apply_values(
        &mut self,
        init: bool,
        retain_user_values: bool,
        directive_file_collection: &BTreeMap<String, String>,
        only_advanced_dataset_dialog: bool,
    ) {
        if !only_advanced_dataset_dialog && !init && !retain_user_values {
            self.length_of_pieces = None;
        }
        if !only_advanced_dataset_dialog {
            self.set_individual_defaults();
        }
        self.set_values(
            directive_file_collection,
            false,
            only_advanced_dataset_dialog,
            false,
        );
        self.set_values(
            directive_file_collection,
            true,
            only_advanced_dataset_dialog,
            false,
        );
    }
    pub fn set_montage(&mut self, montage: bool) {
        self.ltf_prenewst_bin_by_factor.set_enabled(!montage);
        self.ltf_preblend_bin_by_factor.set_enabled(montage);
    }
    fn set_individual_defaults(&mut self) {
        self.rtf_auto_fit_range_and_step.set_selected(true);
    }

    pub fn set_parameters(&mut self, metadata: &BatchRunTomoDatasetMetaData) {
        for (name, text) in &metadata.values {
            match name.as_str() {
                "distort" => {
                    if let Some(field) = &mut self.ltf_distort {
                        field.set_text(text);
                    }
                }
                "gradient" => {
                    if let Some(field) = &mut self.ltf_gradient {
                        field.set_text(text);
                    }
                }
                "gold" => self.ltf_gold.set_text(text),
                "targetNumberOfBeads" => self.ltf_target_number_of_beads.set_text(text),
                "numberOfMarkers" => self.ltf_number_of_markers.set_text(text),
                "sizeOfPatchesXAndY" => self.ltf_size_of_patches_x_and_y.set_text(text),
                "positioningGold" => self.ltf_positioning_gold.set_text(text),
                "positioningThickness" => self.ltf_positioning_thickness.set_text(text),
                "modelFile" => self.ltf_model_file.set_text(text),
                "scanDefocusRange" => self.ltf_scan_defocus_range.set_text(text),
                "defocus" => self.ltf_defocus.set_text(text),
                "autoFitRange" => self.ltf_auto_fit_range_and_step.set_text(text),
                "autoFitStep" => self.ltf_auto_fit_step.set_text(text),
                "fakeSIRTiterations" => self.ltf_fake_sirt_iterations.set_text(text),
                "leaveIterations" => self.ltf_leave_iterations.set_text(text),
                "thickness" => self.ltf_thickness.set_text(text),
                "binnedThickness" => self.ltf_binned_thickness.set_text(text),
                "extraThickness" => self.ltf_extra_thickness.set_text(text),
                "fallbackThickness" => self.ltf_fallback_thickness.set_text(text),
                "findSecAddThickness" => self.ctf_find_sec_add_thickness.set_text(text),
                "scaleFromZ" => self.ctf_scale_from_z.set_text(text),
                "goldErasingThickness" => self.ltf_gold_erasing_thickness.set_text(text),
                _ => {}
            }
        }
        for (name, selected) in &metadata.booleans {
            match name.as_str() {
                "removeXrays" => self.cb_remove_xrays.set_selected(*selected),
                "enableStretching" => self.cb_enable_stretching.set_selected(*selected),
                "localAlignments" => self.cb_local_alignments.set_selected(*selected),
                "lengthOfPieces" => self.cb_length_of_pieces.set_selected(*selected),
                "correctCTF" => self.cb_correct_ctf.set_selected(*selected),
                "autoFitRangeAndStep" => self.rtf_auto_fit_range_and_step.set_selected(*selected),
                "fitEveryImage" => self.rb_fit_every_image.set_selected(*selected),
                "useFakeSIRTiterations" => self.cb_fake_sirt_iterations.set_selected(*selected),
                "useSirt" => self.cb_use_sirt.set_selected(*selected),
                "scaleToInteger" => self.cb_scale_to_integer.set_selected(*selected),
                "doBackprojAlso" => self.cb_do_backproj_also.set_selected(*selected),
                "thickness" => self.rtf_thickness.set_selected(*selected),
                "binnedThickness" => self.rtf_binned_thickness.set_selected(*selected),
                "fallbackAndExtraThickness" => {
                    self.rb_fallback_and_extra_thickness.set_selected(*selected)
                }
                "useFindSecAddThickness" => self.cb_find_sec_add_thickness.set_selected(*selected),
                "useScaleFromZ" => self.cb_scale_from_z.set_selected(*selected),
                "eraseGold" => self.cb_erase_gold.set_selected(*selected),
                "eraseGoldFid" => self.rb_erase_gold_fid.set_selected(*selected),
                "eraseGold3d" => self.rb_erase_gold_3d.set_selected(*selected),
                "sampleType" => self.cb_sample_type.set_selected(*selected),
                "sampleTypePlasticSection" => {
                    self.rb_sample_type_plastic_section.set_selected(*selected)
                }
                "sampleTypeCryo" => self.rb_sample_type_cryo.set_selected(*selected),
                "hasGoldBeads" => self.cb_has_gold_beads.set_selected(*selected),
                "tuneFittingAndSampling" => {
                    self.cb_tune_fitting_and_sampling.set_selected(*selected)
                }
                _ => {}
            }
        }
        self.ltf_prenewst_bin_by_factor
            .set_text(&metadata.prenewst_bin_by_factor.to_string());
        self.ltf_preblend_bin_by_factor
            .set_text(&metadata.preblend_bin_by_factor.to_string());
        self.update_display();
        self.status_changed(self.status);
    }
    pub fn get_parameters(&self, metadata: &mut BatchRunTomoDatasetMetaData) {
        if let Some(field) = &self.ltf_distort {
            metadata.values.insert("distort".into(), field.get_text());
        }
        if let Some(field) = &self.ltf_gradient {
            metadata.values.insert("gradient".into(), field.get_text());
        }
        metadata
            .values
            .insert("gold".into(), self.ltf_gold.get_text());
        metadata.values.insert(
            "targetNumberOfBeads".into(),
            self.ltf_target_number_of_beads.get_text(),
        );
        metadata.values.insert(
            "numberOfMarkers".into(),
            self.ltf_number_of_markers.get_text(),
        );
        metadata.values.insert(
            "sizeOfPatchesXAndY".into(),
            self.ltf_size_of_patches_x_and_y.get_text(),
        );
        metadata.values.insert(
            "positioningGold".into(),
            self.ltf_positioning_gold.get_text(),
        );
        metadata.values.insert(
            "positioningThickness".into(),
            self.ltf_positioning_thickness.get_text(),
        );
        metadata
            .values
            .insert("modelFile".into(), self.ltf_model_file.get_text());
        metadata.values.insert(
            "scanDefocusRange".into(),
            self.ltf_scan_defocus_range.get_text(),
        );
        metadata
            .values
            .insert("defocus".into(), self.ltf_defocus.get_text());
        metadata.values.insert(
            "autoFitRange".into(),
            self.ltf_auto_fit_range_and_step.get_text(),
        );
        metadata
            .values
            .insert("autoFitStep".into(), self.ltf_auto_fit_step.get_text());
        metadata.values.insert(
            "fakeSIRTiterations".into(),
            self.ltf_fake_sirt_iterations.get_text(),
        );
        metadata.values.insert(
            "leaveIterations".into(),
            self.ltf_leave_iterations.get_text(),
        );
        metadata
            .values
            .insert("thickness".into(), self.ltf_thickness.get_text());
        metadata.values.insert(
            "binnedThickness".into(),
            self.ltf_binned_thickness.get_text(),
        );
        metadata
            .values
            .insert("extraThickness".into(), self.ltf_extra_thickness.get_text());
        metadata.values.insert(
            "fallbackThickness".into(),
            self.ltf_fallback_thickness.get_text(),
        );
        metadata.values.insert(
            "findSecAddThickness".into(),
            self.ctf_find_sec_add_thickness.get_text(),
        );
        metadata
            .values
            .insert("scaleFromZ".into(), self.ctf_scale_from_z.get_text());
        metadata.values.insert(
            "goldErasingThickness".into(),
            self.ltf_gold_erasing_thickness.get_text(),
        );
        metadata
            .booleans
            .insert("removeXrays".into(), self.cb_remove_xrays.is_selected());
        metadata.booleans.insert(
            "enableStretching".into(),
            self.cb_enable_stretching.is_selected(),
        );
        metadata.booleans.insert(
            "localAlignments".into(),
            self.cb_local_alignments.is_selected(),
        );
        metadata.booleans.insert(
            "lengthOfPieces".into(),
            self.cb_length_of_pieces.is_selected(),
        );
        metadata
            .booleans
            .insert("correctCTF".into(), self.cb_correct_ctf.is_selected());
        metadata.booleans.insert(
            "autoFitRangeAndStep".into(),
            self.rtf_auto_fit_range_and_step.is_selected(),
        );
        metadata.booleans.insert(
            "fitEveryImage".into(),
            self.rb_fit_every_image.is_selected(),
        );
        metadata.booleans.insert(
            "useFakeSIRTiterations".into(),
            self.cb_fake_sirt_iterations.is_selected(),
        );
        metadata
            .booleans
            .insert("useSirt".into(), self.cb_use_sirt.is_selected());
        metadata.booleans.insert(
            "scaleToInteger".into(),
            self.cb_scale_to_integer.is_selected(),
        );
        metadata.booleans.insert(
            "doBackprojAlso".into(),
            self.cb_do_backproj_also.is_selected(),
        );
        metadata
            .booleans
            .insert("thickness".into(), self.rtf_thickness.is_selected());
        metadata.booleans.insert(
            "binnedThickness".into(),
            self.rtf_binned_thickness.is_selected(),
        );
        metadata.booleans.insert(
            "fallbackAndExtraThickness".into(),
            self.rb_fallback_and_extra_thickness.is_selected(),
        );
        metadata.booleans.insert(
            "useFindSecAddThickness".into(),
            self.cb_find_sec_add_thickness.is_selected(),
        );
        metadata
            .booleans
            .insert("useScaleFromZ".into(), self.cb_scale_from_z.is_selected());
        metadata
            .booleans
            .insert("eraseGold".into(), self.cb_erase_gold.is_selected());
        metadata
            .booleans
            .insert("eraseGoldFid".into(), self.rb_erase_gold_fid.is_selected());
        metadata
            .booleans
            .insert("eraseGold3d".into(), self.rb_erase_gold_3d.is_selected());
        metadata
            .booleans
            .insert("sampleType".into(), self.cb_sample_type.is_selected());
        metadata.booleans.insert(
            "sampleTypePlasticSection".into(),
            self.rb_sample_type_plastic_section.is_selected(),
        );
        metadata.booleans.insert(
            "sampleTypeCryo".into(),
            self.rb_sample_type_cryo.is_selected(),
        );
        metadata
            .booleans
            .insert("hasGoldBeads".into(), self.cb_has_gold_beads.is_selected());
        metadata.booleans.insert(
            "tuneFittingAndSampling".into(),
            self.cb_tune_fitting_and_sampling.is_selected(),
        );
        metadata.prenewst_bin_by_factor = self
            .ltf_prenewst_bin_by_factor
            .get_text()
            .parse()
            .unwrap_or(1);
        metadata.preblend_bin_by_factor = self
            .ltf_preblend_bin_by_factor
            .get_text()
            .parse()
            .unwrap_or(1);
    }
    pub fn display(&mut self) {
        self.display_advanced(false);
    }
    fn display_advanced(&mut self, advanced: bool) {
        if self.global {
            self.frame_state_normal = false;
        } else {
            self.pnl_root.frame_visible = true;
            self.frame_state_normal = true;
        }
        self.set_advanced(advanced);
    }
    fn set_advanced(&mut self, advanced: bool) {
        if self.advanced != advanced {
            self.update_advanced(advanced, false);
        }
    }

    /// Java `saveAutodoc(...)`; `WritableAutodoc` is represented by the output map.
    pub fn save_autodoc(
        &mut self,
        autodoc: &mut BTreeMap<String, String>,
        do_validation: bool,
        validate_only: bool,
    ) -> bool {
        if validate_only && !do_validation {
            return true;
        }
        if do_validation
            && self.cb_correct_ctf.is_selected()
            && self.ltf_scan_defocus_range.is_empty()
            && self.ltf_defocus.is_empty()
        {
            return false;
        }
        if do_validation && !self.validate() {
            return false;
        }
        let gold = if self.fiducial_model_mode {
            self.ltf_gold.get_text()
        } else if self.ltf_positioning_gold.is_enabled() {
            self.ltf_positioning_gold.get_text()
        } else {
            "0".into()
        };
        let mut metadata = BatchRunTomoDatasetMetaData::default();
        self.get_parameters(&mut metadata);
        let mut values = metadata.values;
        for (name, selected) in metadata.booleans {
            values.insert(name, if selected { "1".into() } else { "0".into() });
        }
        values.insert(
            "BIN_BY_FACTOR_FOR_ALIGNED_STACK".into(),
            self.ltf_bin_by_factor.get_text(),
        );
        values.insert(
            "BIN_BY_FACTOR_FOR_PRENEWST".into(),
            self.ltf_prenewst_bin_by_factor.get_text(),
        );
        values.insert(
            "BIN_BY_FACTOR_FOR_PREBLEND".into(),
            self.ltf_preblend_bin_by_factor.get_text(),
        );
        if let Some(field) = &self.ltf_distort {
            values.insert("DISTORT".into(), field.get_text());
        }
        if let Some(field) = &self.ltf_gradient {
            values.insert("GRADIENT".into(), field.get_text());
        }
        values.insert("GOLD".into(), gold);
        values.insert(
            "LENGTH_OF_PIECES".into(),
            if self.cb_length_of_pieces.is_selected() {
                self.length_of_pieces
                    .clone()
                    .unwrap_or_else(|| LENGTH_OF_PIECES_DEFAULT.into())
            } else {
                String::new()
            },
        );
        values.insert(
            "AUTO_FIT_RANGE_AND_STEP".into(),
            if self.rb_fit_every_image.is_selected() {
                format!("{CTF_RANGE_FIT_EVERY_IMAGE},{CTF_STEP_FIT_EVERY_IMAGE}")
            } else {
                self.make_list(
                    self.ltf_auto_fit_range_and_step.get_text(),
                    self.ltf_auto_fit_step.get_text(),
                )
            },
        );
        values.insert(
            "SCALE_TO_INTEGER".into(),
            if self.cb_scale_to_integer.is_selected() {
                SCALE_TO_INTEGER_VALUE.into()
            } else {
                String::new()
            },
        );
        if !validate_only {
            autodoc.extend(values);
        }
        true
    }
    fn make_list(&self, mut element_1: String, mut element_2: String) -> String {
        let set_1 = !element_1.trim().is_empty();
        let set_2 = !element_2.trim().is_empty();
        if !set_1 {
            element_1.clear();
        }
        if !set_2 {
            element_2.clear();
        }
        format!("{}{}{}", element_1, if set_2 { "," } else { "" }, element_2)
    }
    pub fn set_values(
        &mut self,
        directive_files: &BTreeMap<String, String>,
        set_field_highlight_value: bool,
        only_advanced_dataset_dialog: bool,
        _loading_directive_file: bool,
    ) {
        if only_advanced_dataset_dialog {
            return;
        }
        if let Some(value) = directive_files.get("DISTORT") {
            if let Some(field) = &mut self.ltf_distort {
                field.set_text(value);
            }
        }
        if let Some(value) = directive_files.get("GRADIENT") {
            if let Some(field) = &mut self.ltf_gradient {
                field.set_text(value);
            }
        }
        if let Some(value) = directive_files.get("REMOVE_XRAYS") {
            self.cb_remove_xrays.set_selected(value != "0");
        }
        if let Some(value) = directive_files.get("MODEL_FILE") {
            self.ltf_model_file.set_text(value);
        }
        if let Some(value) = directive_files.get("BIN_BY_FACTOR_FOR_ALIGNED_STACK") {
            self.ltf_bin_by_factor.set_text(value);
        }
        if let Some(value) = directive_files.get("BIN_BY_FACTOR_FOR_PRENEWST") {
            self.ltf_prenewst_bin_by_factor.set_text(value);
        }
        if let Some(value) = directive_files.get("BIN_BY_FACTOR_FOR_PREBLEND") {
            self.ltf_preblend_bin_by_factor.set_text(value);
        }
        if let Some(value) = directive_files.get("FIDUCIALLESS") {
            self.rb_fiducialless.set_selected(value != "0");
        }
        if let Some(value) = directive_files.get("GOLD") {
            self.ltf_gold.set_text(value);
            self.ltf_positioning_gold.set_text(value);
        }
        if let Some(value) = directive_files.get("LENGTH_OF_PIECES") {
            self.length_of_pieces = (!set_field_highlight_value).then(|| value.clone());
            self.cb_length_of_pieces
                .set_selected(!value.trim().is_empty() && value.trim() != "0");
        }
        if let Some(value) = directive_files.get("TARGET_NUMBER_OF_BEADS") {
            self.ltf_target_number_of_beads.set_text(value);
        }
        if let Some(value) = directive_files.get("NUMBER_OF_MARKERS") {
            self.ltf_number_of_markers.set_text(value);
        }
        if let Some(value) = directive_files.get("SIZE_OF_PATCHES_X_AND_Y") {
            self.ltf_size_of_patches_x_and_y.set_text(value);
        }
        if let Some(value) = directive_files.get("ENABLE_STRETCHING") {
            self.cb_enable_stretching.set_selected(value != "0");
        }
        if let Some(value) = directive_files.get("LOCAL_ALIGNMENTS") {
            self.cb_local_alignments.set_selected(value != "0");
        }
        if let Some(value) = directive_files.get("CORRECT_CTF") {
            self.cb_correct_ctf.set_selected(value != "0");
        }
        if let Some(value) = directive_files.get("TUNE_FITTING_AND_SAMPLING") {
            self.cb_tune_fitting_and_sampling.set_selected(value != "0");
        }
        if let Some(value) = directive_files.get("TRACKING_METHOD") {
            self.rb_tracking_method_seed.set_selected(value == "seed");
            self.rb_tracking_method_raptor
                .set_selected(value == "raptor");
            self.rb_tracking_method_patch_tracking
                .set_selected(value == "patchTracking");
            self.update_gold_panel();
        }
        if let Some(value) = directive_files.get("SCAN_DEFOCUS_RANGE") {
            let converted = value
                .split(',')
                .filter_map(|part| part.trim().parse::<f64>().ok())
                .map(|part| (part / 1000.0).to_string())
                .collect::<Vec<_>>()
                .join(",");
            self.ltf_scan_defocus_range.set_text(&converted);
        }
        if let Some(value) = directive_files
            .get("DEFOCUS")
            .and_then(|value| value.parse::<f64>().ok())
        {
            self.ltf_defocus.set_text(&(value / 1000.0).to_string());
        }
        if let Some(value) = directive_files.get("AUTO_FIT_RANGE_AND_STEP") {
            let mut split = value.split(',');
            self.ltf_auto_fit_range_and_step
                .set_text(split.next().unwrap_or_default());
            self.ltf_auto_fit_step
                .set_text(split.next().unwrap_or_default());
            if self.ltf_auto_fit_range_and_step.get_text() == CTF_RANGE_FIT_EVERY_IMAGE
                && self.ltf_auto_fit_step.get_text() == CTF_STEP_FIT_EVERY_IMAGE
            {
                self.rb_fit_every_image.set_selected(true);
            }
        }
        if let Some(value) = directive_files.get("FAKE_SIRT_ITERATIONS") {
            self.cb_fake_sirt_iterations
                .set_selected(!value.is_empty() && value != "0");
            self.ltf_fake_sirt_iterations.set_text(value);
        }
        if let Some(value) = directive_files.get("USE_SIRT") {
            self.cb_use_sirt.set_selected(value != "0");
        }
        if let Some(value) = directive_files.get("DO_BACKPROJ_ALSO") {
            self.cb_do_backproj_also.set_selected(value != "0");
        }
        if !self.cb_fake_sirt_iterations.is_selected() && !self.cb_use_sirt.is_selected() {
            self.cb_do_backproj_also.set_selected(true);
        }
        if let Some(value) = directive_files.get("LEAVE_ITERATIONS") {
            self.ltf_leave_iterations.set_text(value);
        }
        if let Some(value) = directive_files.get("SCALE_TO_INTEGER") {
            self.cb_scale_to_integer
                .set_selected(!value.is_empty() && value != "0");
        }
        if let Some(value) = directive_files.get("FALLBACK_THICKNESS") {
            self.rb_fallback_and_extra_thickness.set_selected(true);
            self.ltf_fallback_thickness.set_text(value);
        }
        if let Some(value) = directive_files.get("EXTRA_THICKNESS") {
            self.rb_fallback_and_extra_thickness.set_selected(true);
            self.ltf_extra_thickness.set_text(value);
        }
        if let Some(value) = directive_files.get("BINNED_THICKNESS") {
            self.rtf_binned_thickness.set_selected(true);
            self.ltf_binned_thickness.set_text(value);
        }
        if let Some(value) = directive_files.get("THICKNESS_FOR_TILT") {
            self.rtf_thickness.set_selected(true);
            self.ltf_thickness.set_text(value);
        }
        if directive_files.contains_key("SCALE_FROM_X")
            || directive_files.contains_key("SCALE_FROM_Y")
        {
            self.cb_do_trimvol.set_selected(true);
            self.cb_scale_from_z.set_selected(true);
        }
        if let Some(value) = directive_files.get("SCALE_FROM_Z") {
            self.cb_scale_from_z
                .set_selected(!value.is_empty() && value != "0");
            self.ctf_scale_from_z.set_text(value);
        }
        if let Some(value) = directive_files.get("DO_TRIMVOL") {
            self.cb_do_trimvol.set_selected(value != "0");
        }
        if let Some(value) = directive_files.get("FIND_SEC_ADD_THICKNESS") {
            self.cb_find_sec_add_thickness
                .set_selected(!value.is_empty() && value != "0");
            self.ctf_find_sec_add_thickness.set_text(value);
        }
        if let Some(value) = directive_files.get("ERASE_GOLD") {
            self.cb_erase_gold
                .set_selected(!value.is_empty() && value != "0");
            self.rb_erase_gold_fid.set_selected(value == "fid");
            self.rb_erase_gold_3d.set_selected(value == "find3d");
        }
        if let Some(value) = directive_files.get("THICKNESS_FOR_GOLD_ERASING") {
            self.ltf_gold_erasing_thickness.set_text(value);
        }
        if let Some(value) = directive_files.get("SAMPLE_TYPE") {
            self.cb_sample_type.set_selected(value != "0");
            self.rb_sample_type_plastic_section
                .set_selected(value == "plasticSection");
            self.rb_sample_type_cryo.set_selected(value == "cryo");
        }
        if let Some(value) = directive_files.get("THICKNESS_FOR_POSITIONING") {
            self.ltf_positioning_thickness.set_text(value);
        }
        if let Some(value) = directive_files.get("HAS_GOLD_BEADS") {
            self.cb_has_gold_beads.set_selected(value != "0");
        }
        self.update_display();
    }
    pub fn action_performed(
        &mut self,
        action_command: Option<&str>,
        boundary: Option<&mut dyn BatchRunTomoDatasetDialogBoundary>,
    ) {
        let Some(command) = action_command else {
            return;
        };
        if command == "Make in 3dmod" && self.row_exists {
            if let Some(boundary) = boundary {
                let model = boundary.imod_stack(
                    (!self.ltf_model_file.is_empty())
                        .then(|| self.ltf_model_file.get_text())
                        .as_deref(),
                );
                if self.ltf_model_file.is_empty() {
                    if let Some(model) = model {
                        self.ltf_model_file.set_text(&model);
                    }
                }
            }
        } else if command == "OK" {
            self.set_visible(false);
        } else if command == "Revert to Global" {
            self.set_visible(false);
            if let Some(boundary) = boundary {
                boundary.delete_dataset();
            }
        } else if command == "Advanced" {
            self.update_advanced(true, false);
        } else if command == "Basic" {
            self.update_advanced(false, false);
        } else if command == "Autoseed and track" || command == "Raptor and track" {
            self.update_gold_panel();
        } else if command == "Fit Window" {
            self.pack();
        } else if command == "SIRT-like filter" {
            if self.cb_fake_sirt_iterations.is_selected() {
                self.cb_use_sirt.set_selected(false);
            }
            self.update_display();
        } else if command == "SIRT" {
            if self.cb_use_sirt.is_selected() {
                self.cb_fake_sirt_iterations.set_selected(false);
            }
            self.update_display();
        } else {
            self.update_display();
        }
    }
    pub fn last_row_deleted(&mut self) {
        self.empty_table = true;
        self.row_exists = false;
        self.update_display();
    }
    pub fn first_row_added(&mut self) {
        self.empty_table = false;
        self.row_exists = true;
        self.update_display();
    }
    fn set_tooltips(&mut self) {
        self.pnl_root.tooltips_initialized = true;
    }
    pub fn display_ui_component(&mut self) {
        self.display();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn global_constructor_preserves_source_defaults_and_gating() {
        let mut dialog = BatchRunTomoDatasetDialog::get_global_instance();
        assert!(dialog.rb_tracking_method_seed.is_selected());
        assert!(dialog.rb_erase_gold_3d.is_selected());
        assert!(dialog.cb_do_backproj_also.is_selected());
        assert!(dialog.pnl_root.tooltips_initialized);
        dialog.cb_remove_xrays.set_selected(true);
        dialog.update_display();
        assert!(!dialog.pnl_root.model_button_enabled);
        dialog.first_row_added();
        assert!(dialog.pnl_root.model_button_enabled);
    }
    #[test]
    fn tracking_ctf_sirt_and_erase_gold_rules_match_source() {
        let mut dialog = BatchRunTomoDatasetDialog::get_global_instance();
        dialog.rb_tracking_method_patch_tracking.set_selected(true);
        dialog.cb_erase_gold.set_selected(true);
        dialog.cb_correct_ctf.set_selected(true);
        dialog.cb_use_sirt.set_selected(true);
        dialog.update_display();
        assert!(!dialog.ltf_gold.is_enabled());
        assert!(dialog.ltf_size_of_patches_x_and_y.is_enabled());
        assert!(dialog.rb_erase_gold_3d.is_selected());
        assert!(!dialog.rb_erase_gold_fid.is_enabled());
        assert!(dialog.ltf_leave_iterations.is_visible());
        assert!(!dialog.ltf_fake_sirt_iterations.is_visible());
    }
    #[test]
    fn autodoc_converts_ctf_and_selected_source_values() {
        let mut dialog = BatchRunTomoDatasetDialog::get_global_instance();
        dialog.cb_correct_ctf.set_selected(true);
        dialog.ltf_defocus.set_text("2.5");
        dialog.ltf_gold.set_text("10");
        dialog.cb_scale_to_integer.set_selected(true);
        let mut autodoc = BTreeMap::new();
        assert!(dialog.save_autodoc(&mut autodoc, true, false));
        assert_eq!(autodoc.get("GOLD"), Some(&"10".into()));
        assert_eq!(
            autodoc.get("SCALE_TO_INTEGER"),
            Some(&SCALE_TO_INTEGER_VALUE.into())
        );
    }
    #[test]
    fn directive_ctf_conversion_and_row_events_are_preserved() {
        let mut dialog = BatchRunTomoDatasetDialog::get_global_instance();
        dialog.set_values(
            &BTreeMap::from([
                ("SCAN_DEFOCUS_RANGE".into(), "1000,2500".into()),
                ("DEFOCUS".into(), "3000".into()),
                ("LENGTH_OF_PIECES".into(), "-1".into()),
            ]),
            false,
            false,
            false,
        );
        assert_eq!(dialog.ltf_scan_defocus_range.get_text(), "1,2.5");
        assert_eq!(dialog.ltf_defocus.get_text(), "3");
        assert!(dialog.cb_length_of_pieces.is_selected());
        dialog.last_row_deleted();
        assert!(dialog.empty_table);
    }
}
