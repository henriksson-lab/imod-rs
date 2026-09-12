//! `IMOD/Etomo/src/etomo/ui/swing/SerialSectionsDialog.java`.
//!
//! Swing construction, header reads, autodoc lookup, and process execution stay at
//! their direct boundaries. This unit owns the source dialog selections, parameter
//! transfers, tab transitions, action routing, and popup construction.
#![allow(dead_code)]

use super::context_menu::{ContextMenu, MouseEvent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::{axis_id::AxisID, dialog_type::DialogType, view_type::ViewType};

pub const DIALOG_TYPE: DialogType = DialogType::SerialSections;
pub const SHIFT_LABEL: &str = "Shift in ";
pub const SIZE_LABEL: &str = "Size in ";
pub const PIECE_TO_PIECE_DIFFERENCES_ONLY: i32 = 1;
pub const GRADIENT_WITHIN_PIECES_ALSO: i32 = 2;
pub const SUM_PIECES_FOR_GRADIENT: i32 = 1;

/// Java private static `Tab`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tab {
    InitialBlend,
    Align,
    MakeAlignedStack,
}
impl Tab {
    pub const NUM_TABS: usize = 3;
    pub fn index(self) -> usize {
        match self {
            Self::InitialBlend => 0,
            Self::Align => 1,
            Self::MakeAlignedStack => 2,
        }
    }
    pub fn title(self) -> &'static str {
        match self {
            Self::InitialBlend => "Initial Blend",
            Self::Align => "Align",
            Self::MakeAlignedStack => "Make Stack",
        }
    }
    pub fn get_instance(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::InitialBlend),
            1 => Some(Self::Align),
            2 => Some(Self::MakeAlignedStack),
            _ => None,
        }
    }
    pub fn get_default_instance(view_type: ViewType) -> Self {
        if view_type == ViewType::Montage {
            Self::InitialBlend
        } else {
            Self::Align
        }
    }
    pub fn equals(self, index: usize) -> bool {
        self.index() == index
    }
}

/// Source operation made on `SerialSectionsManager` by `action`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SerialSectionsManagerOperation {
    Preblend,
    MidasFixEdges,
    Align,
    ImodRaw,
    ImodPreblend,
    ImodPrealign,
    ImodAlign,
}

/// Direct `SerialSectionsManager`, transform-check, and `UIHarness` collaborators.
pub trait SerialSectionsDialogApplicationManager {
    fn serial_sections_operation(
        &mut self,
        operation: SerialSectionsManagerOperation,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
    fn check_up_to_date_edge_functions_file(
        &mut self,
        axis_id: AxisID,
        tab_label: &str,
        button_label: &str,
    );
    fn pack(&mut self, axis_id: AxisID);
    fn move_sub_frame(&mut self);
}

/// Exact `ContextPopup` construction inputs from Java `popUpContextMenu`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SerialSectionsContextPopup {
    pub mouse_event: MouseEvent,
    pub anchor: Option<&'static str>,
    pub guide: &'static str,
    pub man_page_label: Option<[&'static str; 3]>,
    pub man_page: Option<[&'static str; 3]>,
    pub log_file_label: Option<&'static str>,
    pub log_file: Option<&'static str>,
    pub graph_serial_sections_mean_max: bool,
}

/// Value state passed to/from untranslated metadata and comscript collaborators.
#[derive(Clone, Debug, PartialEq)]
pub struct SerialSectionsDialogParameters {
    pub robust_fit_criterion: Option<String>,
    pub midas_binning: i32,
    pub number_to_fit_global_alignment: bool,
    pub use_reference_section: bool,
    pub reference_section: i32,
    pub size_x: String,
    pub size_y: String,
    pub shift_x: String,
    pub shift_y: String,
    pub preblend_very_sloppy_montage: bool,
    pub preblend_weight_for_expected_shifts: bool,
    pub preblend_weight_distance: Option<String>,
    pub preblend_em_grid_map_filter: bool,
    pub preblend_high_frequency_filter_cutoff: Option<String>,
    pub tab: Option<usize>,
    pub other_sum_gradient_file: String,
    pub bin_by_factor: i32,
    pub fill_with_zero: bool,
    pub hybrid_fits: Option<HybridFits>,
    pub reference_section_in_xftoxg: Option<i32>,
}
impl Default for SerialSectionsDialogParameters {
    fn default() -> Self {
        Self {
            robust_fit_criterion: None,
            midas_binning: 1,
            number_to_fit_global_alignment: false,
            use_reference_section: false,
            reference_section: 0,
            size_x: String::new(),
            size_y: String::new(),
            shift_x: String::new(),
            shift_y: String::new(),
            preblend_very_sloppy_montage: false,
            preblend_weight_for_expected_shifts: false,
            preblend_weight_distance: None,
            preblend_em_grid_map_filter: false,
            preblend_high_frequency_filter_cutoff: None,
            tab: None,
            other_sum_gradient_file: String::new(),
            bin_by_factor: 1,
            fill_with_zero: false,
            hybrid_fits: None,
            reference_section_in_xftoxg: None,
        }
    }
}

/// Java `XftoxgParam.HybridFits`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HybridFits {
    Rotation,
    Translations,
    TranslationsRotations,
}

/// Java `SerialSectionsDialog` fields, excluding only Swing-owned component handles.
pub struct SerialSectionsDialog {
    pub axis_id: AxisID,
    pub view_type: ViewType,
    pub pnl_root_created: bool,
    pub pnl_tab_array: [bool; Tab::NUM_TABS],
    pub pnl_tab_body_array: [bool; Tab::NUM_TABS],
    pub tab_enabled: [bool; Tab::NUM_TABS],
    pub selected_tab_index: usize,
    pub cur_tab: Option<Tab>,
    pub reference_section_problem: bool,
    pub reference_section_max: Option<i32>,
    pub reference_section_checkbox_enabled: bool,
    pub reference_section_selected: bool,
    pub reference_section: i32,
    pub preblend_very_sloppy_montage: bool,
    pub preblend_very_sloppy_montage_enabled: bool,
    pub preblend_read_in_xcorrs: bool,
    pub robust_fit_selected: bool,
    pub robust_fit_criterion: String,
    pub midas_binning: i32,
    pub size_x: String,
    pub size_y: String,
    pub shift_x: String,
    pub shift_y: String,
    pub bin_by_factor: i32,
    pub fill_with_zero: bool,
    pub no_options_selected: bool,
    pub hybrid_fits: Option<HybridFits>,
    pub number_to_fit_global_alignment: bool,
    pub none_intensity_correction_selected: bool,
    pub piece_to_piece_differences_selected: bool,
    pub gradient_within_pieces_selected: bool,
    pub none_initial_gradient_selected: bool,
    pub planar_fit_selected: bool,
    pub gradient_file_selected: bool,
    pub gradient_file: String,
    pub gradient_file_enabled: bool,
    pub em_grid_map_filter_selected: bool,
    pub em_grid_map_filter_enabled: bool,
    pub high_frequency_filter_cutoff: String,
    pub high_frequency_filter_cutoff_enabled: bool,
    pub weight_for_expected_shifts_selected: bool,
    pub weight_for_expected_shifts_enabled: bool,
    pub default_distance_selected: bool,
    pub pixel_distance_selected: bool,
    pub pixel_distance: String,
    pub pixel_distance_enabled: bool,
    pub pixels_label_enabled: bool,
    pub listeners_added: bool,
    pub auto_alignment_controller_set: bool,
    pub auto_alignment_process_changed: bool,
    pub context_popup: Option<SerialSectionsContextPopup>,
}

impl SerialSectionsDialog {
    /// Java private `SerialSectionsDialog(SerialSectionsManager, AxisID)`.
    pub fn new(axis_id: AxisID, view_type: ViewType) -> Self {
        Self {
            axis_id,
            view_type,
            pnl_root_created: false,
            pnl_tab_array: [false; Tab::NUM_TABS],
            pnl_tab_body_array: [false; Tab::NUM_TABS],
            tab_enabled: [true; Tab::NUM_TABS],
            selected_tab_index: 0,
            cur_tab: None,
            reference_section_problem: false,
            reference_section_max: None,
            reference_section_checkbox_enabled: false,
            reference_section_selected: false,
            reference_section: 0,
            preblend_very_sloppy_montage: false,
            preblend_very_sloppy_montage_enabled: true,
            preblend_read_in_xcorrs: false,
            robust_fit_selected: false,
            robust_fit_criterion: String::new(),
            midas_binning: 1,
            size_x: String::new(),
            size_y: String::new(),
            shift_x: String::new(),
            shift_y: String::new(),
            bin_by_factor: 1,
            fill_with_zero: false,
            no_options_selected: false,
            hybrid_fits: None,
            number_to_fit_global_alignment: false,
            none_intensity_correction_selected: true,
            piece_to_piece_differences_selected: false,
            gradient_within_pieces_selected: false,
            none_initial_gradient_selected: true,
            planar_fit_selected: false,
            gradient_file_selected: false,
            gradient_file: String::new(),
            gradient_file_enabled: false,
            em_grid_map_filter_selected: false,
            em_grid_map_filter_enabled: true,
            high_frequency_filter_cutoff: "0.25".into(),
            high_frequency_filter_cutoff_enabled: false,
            weight_for_expected_shifts_selected: false,
            weight_for_expected_shifts_enabled: true,
            default_distance_selected: true,
            pixel_distance_selected: false,
            pixel_distance: String::new(),
            pixel_distance_enabled: false,
            pixels_label_enabled: false,
            listeners_added: false,
            auto_alignment_controller_set: false,
            auto_alignment_process_changed: false,
            context_popup: None,
        }
    }
    /// Java static `getInstance`.
    pub fn get_instance(axis_id: AxisID, view_type: ViewType) -> Self {
        let mut instance = Self::new(axis_id, view_type);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }
    pub fn set_auto_alignment_controller(&mut self) {
        self.auto_alignment_controller_set = true;
    }
    /// Java `createPanel`; actual layout is the GUI boundary.
    pub fn create_panel(&mut self) {
        self.pnl_root_created = true;
        self.pnl_tab_array = [true; Tab::NUM_TABS];
        self.pnl_tab_body_array = [true; Tab::NUM_TABS];
        self.tab_enabled[Tab::InitialBlend.index()] = self.view_type == ViewType::Montage;
        self.selected_tab_index = Tab::get_default_instance(self.view_type).index();
        self.cb_defaults_after_panel_creation();
    }
    /// Source initialization tail in `createPanel`.
    pub fn cb_defaults_after_panel_creation(&mut self) {
        self.none_intensity_correction_selected = true;
        self.none_initial_gradient_selected = true;
        self.weight_for_expected_shifts_selected = false;
        self.default_distance_selected = true;
        self.em_grid_map_filter_selected = false;
        self.high_frequency_filter_cutoff = "0.25".into();
        self.update_display();
    }
    pub fn get_root_container(&self) -> bool {
        self.pnl_root_created
    }
    pub fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }
    pub fn get_dialog_type(&self) -> DialogType {
        DIALOG_TYPE
    }
    pub fn add_listeners(&mut self) {
        self.listeners_added = true;
    }
    pub fn msg_process_ended(&mut self) {
        self.auto_alignment_process_changed = true;
    }
    /// Java `getParameters(SerialSectionsMetaData, boolean)`.
    pub fn get_parameters_serial_sections_metadata(
        &self,
        parameters: &mut SerialSectionsDialogParameters,
        _do_validation: bool,
    ) -> bool {
        parameters.robust_fit_criterion = self
            .robust_fit_selected
            .then(|| self.robust_fit_criterion.clone());
        parameters.midas_binning = self.midas_binning;
        parameters.number_to_fit_global_alignment = self.number_to_fit_global_alignment;
        parameters.use_reference_section =
            self.reference_section_selected && self.reference_section_checkbox_enabled;
        parameters.reference_section = self.reference_section;
        parameters.size_x = self.size_x.clone();
        parameters.size_y = self.size_y.clone();
        parameters.shift_x = self.shift_x.clone();
        parameters.shift_y = self.shift_y.clone();
        parameters.preblend_very_sloppy_montage = self.preblend_very_sloppy_montage;
        parameters.preblend_weight_for_expected_shifts = self.weight_for_expected_shifts_selected;
        parameters.preblend_weight_distance = self
            .pixel_distance_selected
            .then(|| self.pixel_distance.clone());
        parameters.preblend_em_grid_map_filter = self.em_grid_map_filter_selected;
        parameters.preblend_high_frequency_filter_cutoff = self
            .em_grid_map_filter_selected
            .then(|| self.high_frequency_filter_cutoff.clone());
        parameters.tab = self.cur_tab.map(Tab::index);
        parameters.other_sum_gradient_file = self.gradient_file.clone();
        true
    }
    /// Java `setParameters(ConstSerialSectionsMetaData)`.
    pub fn set_parameters_serial_sections_metadata(
        &mut self,
        parameters: &SerialSectionsDialogParameters,
    ) {
        self.robust_fit_selected = parameters.robust_fit_criterion.is_some();
        self.robust_fit_criterion = parameters.robust_fit_criterion.clone().unwrap_or_default();
        self.midas_binning = parameters.midas_binning;
        self.number_to_fit_global_alignment = parameters.number_to_fit_global_alignment;
        self.reference_section_selected = parameters.use_reference_section;
        self.reference_section = parameters.reference_section;
        self.size_x = parameters.size_x.clone();
        self.size_y = parameters.size_y.clone();
        self.shift_x = parameters.shift_x.clone();
        self.shift_y = parameters.shift_y.clone();
        self.preblend_very_sloppy_montage = parameters.preblend_very_sloppy_montage;
        self.weight_for_expected_shifts_selected = parameters.preblend_weight_for_expected_shifts;
        self.pixel_distance_selected = parameters.preblend_weight_distance.is_some();
        if let Some(value) = &parameters.preblend_weight_distance {
            self.pixel_distance = value.clone();
        }
        self.em_grid_map_filter_selected = parameters.preblend_em_grid_map_filter;
        if let Some(value) = &parameters.preblend_high_frequency_filter_cutoff {
            self.high_frequency_filter_cutoff = value.clone();
        }
        self.gradient_file = parameters.other_sum_gradient_file.clone();
        self.change_tab(
            parameters
                .tab
                .unwrap_or_else(|| Tab::get_default_instance(self.view_type).index()),
        );
        self.update_display();
    }
    pub fn is_preblend_robust_fitting(&self) -> bool {
        self.robust_fit_selected
    }
    pub fn get_preblend_robust_fitting(&self) -> String {
        self.robust_fit_criterion.clone()
    }
    pub fn is_fix_intensity_from_edges(&self) -> bool {
        self.none_intensity_correction_selected
            || self.piece_to_piece_differences_selected
            || self.gradient_within_pieces_selected
    }
    pub fn get_fix_intensity_from_edges(&self) -> Option<i32> {
        if self.piece_to_piece_differences_selected {
            Some(PIECE_TO_PIECE_DIFFERENCES_ONLY)
        } else if self.gradient_within_pieces_selected {
            Some(GRADIENT_WITHIN_PIECES_ALSO)
        } else {
            None
        }
    }
    pub fn is_sum_pieces_for_gradient(&self) -> bool {
        self.none_initial_gradient_selected || self.planar_fit_selected
    }
    pub fn get_sum_pieces_for_gradient(&self) -> Option<i32> {
        self.planar_fit_selected.then_some(SUM_PIECES_FOR_GRADIENT)
    }
    pub fn is_other_sum_gradient_file(&self) -> bool {
        self.gradient_file_selected
    }
    pub fn get_other_sum_gradient_file(&self) -> String {
        self.gradient_file.clone()
    }
    pub fn get_parameters_midas(&self, parameters: &mut SerialSectionsDialogParameters) {
        parameters.midas_binning = self.midas_binning;
    }
    pub fn set_preblend_read_in_xcorrs(&mut self, read_in_xcorrs: bool) {
        self.preblend_read_in_xcorrs = read_in_xcorrs;
        self.update_display();
    }
    pub fn set_preblend_parameters(&mut self, parameters: &SerialSectionsDialogParameters) {
        self.preblend_very_sloppy_montage = parameters.preblend_very_sloppy_montage;
        self.robust_fit_selected = parameters.robust_fit_criterion.is_some();
        self.robust_fit_criterion = parameters.robust_fit_criterion.clone().unwrap_or_default();
        self.em_grid_map_filter_selected = parameters.preblend_em_grid_map_filter;
        if let Some(value) = &parameters.preblend_high_frequency_filter_cutoff {
            self.high_frequency_filter_cutoff = value.clone();
        }
        self.weight_for_expected_shifts_selected =
            parameters.preblend_weight_for_expected_shifts && !self.em_grid_map_filter_selected;
        if let Some(value) = &parameters.preblend_weight_distance {
            if value == "1" {
                self.default_distance_selected = true;
                self.pixel_distance_selected = false;
            } else {
                self.default_distance_selected = false;
                self.pixel_distance_selected = true;
                self.pixel_distance = value.clone();
            }
        }
        if self.gradient_file_selected {
            self.gradient_file = parameters.other_sum_gradient_file.clone();
        }
        self.update_display();
    }
    /// Java `getPreblendParameters(BlendmontParam, boolean)`.
    pub fn get_preblend_parameters(
        &self,
        parameters: &mut SerialSectionsDialogParameters,
        _do_validation: bool,
    ) -> bool {
        parameters.preblend_very_sloppy_montage =
            self.preblend_very_sloppy_montage_enabled && self.preblend_very_sloppy_montage;
        parameters.robust_fit_criterion = self
            .robust_fit_selected
            .then(|| self.robust_fit_criterion.clone());
        if self.preblend_read_in_xcorrs {
            parameters.preblend_em_grid_map_filter = false;
            parameters.preblend_high_frequency_filter_cutoff = None;
            parameters.preblend_weight_for_expected_shifts = false;
            parameters.preblend_weight_distance = None;
        } else {
            parameters.preblend_em_grid_map_filter = self.em_grid_map_filter_selected;
            parameters.preblend_high_frequency_filter_cutoff = self
                .em_grid_map_filter_selected
                .then(|| self.high_frequency_filter_cutoff.clone());
            let use_weight =
                self.em_grid_map_filter_selected || self.weight_for_expected_shifts_selected;
            parameters.preblend_weight_for_expected_shifts = use_weight;
            parameters.preblend_weight_distance = use_weight.then(|| {
                if self.default_distance_selected {
                    "1".into()
                } else {
                    self.pixel_distance.clone()
                }
            });
        }
        parameters.other_sum_gradient_file = if self.gradient_file_selected {
            self.gradient_file.clone()
        } else {
            String::new()
        };
        true
    }
    /// Java `getBlendParameters` and `getParameters(NewstParam, boolean)`.
    pub fn get_parameters_blend_or_newst(
        &self,
        parameters: &mut SerialSectionsDialogParameters,
        _do_validation: bool,
    ) -> bool {
        parameters.size_x = self.size_x.clone();
        parameters.size_y = self.size_y.clone();
        parameters.shift_x = self.shift_x.clone();
        parameters.shift_y = self.shift_y.clone();
        parameters.bin_by_factor = self.bin_by_factor;
        parameters.fill_with_zero = self.fill_with_zero;
        true
    }
    pub fn set_blend_parameters(&mut self, parameters: &SerialSectionsDialogParameters) {
        self.bin_by_factor = parameters.bin_by_factor;
        self.fill_with_zero = parameters.fill_with_zero;
    }
    pub fn set_parameters_newst(&mut self, parameters: &SerialSectionsDialogParameters) {
        self.shift_x = parameters.shift_x.clone();
        self.shift_y = parameters.shift_y.clone();
        self.bin_by_factor = parameters.bin_by_factor;
        self.fill_with_zero = parameters.fill_with_zero;
    }
    /// Java `getParameters(XftoxgParam)`.
    pub fn get_parameters_xftoxg(&self, parameters: &mut SerialSectionsDialogParameters) {
        parameters.hybrid_fits = if self.no_options_selected {
            None
        } else {
            self.hybrid_fits
        };
        parameters.number_to_fit_global_alignment = !self.no_options_selected
            && self.hybrid_fits.is_none()
            && self.number_to_fit_global_alignment;
        parameters.reference_section_in_xftoxg = (self.reference_section_selected
            && self.reference_section_checkbox_enabled)
            .then_some(self.reference_section);
    }
    pub fn set_parameters_xftoxg(&mut self, parameters: &SerialSectionsDialogParameters) {
        self.no_options_selected =
            parameters.hybrid_fits.is_none() && !parameters.number_to_fit_global_alignment;
        self.hybrid_fits = parameters.hybrid_fits;
        self.number_to_fit_global_alignment = parameters.number_to_fit_global_alignment;
        self.reference_section_selected = parameters.reference_section_in_xftoxg.is_some();
        if let Some(value) = parameters.reference_section_in_xftoxg {
            self.reference_section = value;
        }
        self.update_display();
    }
    /// Java `action` direct manager dispatch.
    pub fn action<M: SerialSectionsDialogApplicationManager>(
        &mut self,
        manager: &mut M,
        operation: SerialSectionsManagerOperation,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        manager.serial_sections_operation(operation, self.axis_id, run_3dmod_menu_options);
        self.update_display();
    }
    /// Java `updateDisplay`.
    pub fn update_display(&mut self) {
        self.reference_section_checkbox_enabled =
            self.number_to_fit_global_alignment && !self.reference_section_problem;
        self.preblend_very_sloppy_montage_enabled = !self.preblend_read_in_xcorrs;
        self.gradient_file_enabled = self.gradient_file_selected;
        self.em_grid_map_filter_enabled = !self.preblend_read_in_xcorrs;
        self.high_frequency_filter_cutoff_enabled =
            !self.preblend_read_in_xcorrs && self.em_grid_map_filter_selected;
        self.weight_for_expected_shifts_enabled =
            !self.preblend_read_in_xcorrs && !self.em_grid_map_filter_selected;
        self.pixel_distance_enabled = !self.preblend_read_in_xcorrs
            && (self.weight_for_expected_shifts_selected || self.em_grid_map_filter_selected);
        self.pixels_label_enabled = self.pixel_distance_enabled && self.pixel_distance_selected;
    }
    pub fn change_tab(&mut self, new_tab_index: usize) {
        self.selected_tab_index = new_tab_index;
        self.change_tab_current();
    }
    pub fn change_tab_current(&mut self) {
        let new_tab = Tab::get_instance(self.selected_tab_index);
        if new_tab != self.cur_tab {
            self.cur_tab = new_tab;
        }
    }
    pub fn tab_change_warning<M: SerialSectionsDialogApplicationManager>(&self, manager: &mut M) {
        if self.cur_tab == Some(Tab::InitialBlend) {
            manager.check_up_to_date_edge_functions_file(
                self.axis_id,
                "Initial Blend",
                "Make Blended Stack",
            );
        }
    }
    /// Java `popUpContextMenu`, retaining `ContextPopup` construction values.
    pub fn pop_up_context_menu_with_dataset(
        &mut self,
        mouse_event: MouseEvent,
        stack_is_not_one_by: bool,
    ) {
        self.context_popup = Some(match self.cur_tab {
            Some(Tab::InitialBlend) => SerialSectionsContextPopup {
                mouse_event,
                anchor: Some("Blending"),
                guide: "serialalign.html",
                man_page_label: Some(["Blendmont", "Midas", "3dmod"]),
                man_page: Some(["blendmont.html", "midas.html", "3dmod.html"]),
                log_file_label: Some("Preblend"),
                log_file: Some("preblend.log"),
                graph_serial_sections_mean_max: stack_is_not_one_by,
            },
            Some(Tab::Align) => SerialSectionsContextPopup {
                mouse_event,
                anchor: Some("Aligning"),
                guide: "serialalign.html",
                man_page_label: Some(["Xfalign", "Midas", "3dmod"]),
                man_page: Some(["xfalign.html", "midas.html", "3dmod.html"]),
                log_file_label: Some("Xfalign"),
                log_file: Some("xfalign.log"),
                graph_serial_sections_mean_max: false,
            },
            Some(Tab::MakeAlignedStack) => {
                let montage = self.view_type == ViewType::Montage;
                SerialSectionsContextPopup {
                    mouse_event,
                    anchor: Some("Images"),
                    guide: "serialalign.html",
                    man_page_label: Some(if montage {
                        ["Blendmont", "Xftoxg", "3dmod"]
                    } else {
                        ["Colornewst", "Xftoxg", "3dmod"]
                    }),
                    man_page: Some(if montage {
                        ["blendmont.html", "xftoxg.html", "3dmod.html"]
                    } else {
                        ["colornewst.html", "xftoxg.html", "3dmod.html"]
                    }),
                    log_file_label: Some(if montage { "Blend" } else { "Newst" }),
                    log_file: Some(if montage { "blend.log" } else { "newst.log" }),
                    graph_serial_sections_mean_max: false,
                }
            }
            None => SerialSectionsContextPopup {
                mouse_event,
                anchor: None,
                guide: "serialalign.html",
                man_page_label: None,
                man_page: None,
                log_file_label: None,
                log_file: None,
                graph_serial_sections_mean_max: false,
            },
        });
    }
    pub fn set_tooltips(&mut self) {}
    pub fn display(&self) {}
    pub fn display_component(&self) {}
}
impl ContextMenu for SerialSectionsDialog {
    fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        self.pop_up_context_menu_with_dataset(mouse_event, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn construction_uses_source_defaults() {
        let dialog = SerialSectionsDialog::get_instance(AxisID::Only, ViewType::SingleView);
        assert!(!dialog.tab_enabled[Tab::InitialBlend.index()]);
        assert_eq!(dialog.selected_tab_index, Tab::Align.index());
        assert_eq!(dialog.high_frequency_filter_cutoff, "0.25");
        assert!(dialog.listeners_added);
    }
    #[test]
    fn display_dependency_rules_match_read_in_xcorrs_and_peak_options() {
        let mut dialog = SerialSectionsDialog::get_instance(AxisID::Only, ViewType::Montage);
        dialog.em_grid_map_filter_selected = true;
        dialog.weight_for_expected_shifts_selected = true;
        dialog.pixel_distance_selected = true;
        dialog.update_display();
        assert!(dialog.high_frequency_filter_cutoff_enabled);
        assert!(dialog.pixel_distance_enabled);
        assert!(dialog.pixels_label_enabled);
        dialog.set_preblend_read_in_xcorrs(true);
        assert!(!dialog.preblend_very_sloppy_montage_enabled);
        assert!(!dialog.high_frequency_filter_cutoff_enabled);
        assert!(!dialog.pixel_distance_enabled);
    }
    #[test]
    fn preblend_parameter_transfer_resets_filters_for_existing_xcorrs() {
        let mut dialog = SerialSectionsDialog::get_instance(AxisID::Only, ViewType::Montage);
        dialog.robust_fit_selected = true;
        dialog.robust_fit_criterion = "1.5".into();
        dialog.em_grid_map_filter_selected = true;
        dialog.set_preblend_read_in_xcorrs(true);
        let mut parameters = SerialSectionsDialogParameters::default();
        assert!(dialog.get_preblend_parameters(&mut parameters, true));
        assert_eq!(parameters.robust_fit_criterion.as_deref(), Some("1.5"));
        assert!(!parameters.preblend_em_grid_map_filter);
        assert_eq!(parameters.preblend_weight_distance, None);
    }
    #[test]
    fn context_popup_tracks_all_three_source_tabs() {
        let mut dialog = SerialSectionsDialog::get_instance(AxisID::Only, ViewType::Montage);
        dialog.change_tab(Tab::InitialBlend.index());
        dialog.pop_up_context_menu_with_dataset(
            MouseEvent {
                x: 2,
                y: 3,
                right_mouse_button: true,
            },
            true,
        );
        assert_eq!(
            dialog.context_popup.as_ref().unwrap().log_file,
            Some("preblend.log")
        );
        assert!(
            dialog
                .context_popup
                .as_ref()
                .unwrap()
                .graph_serial_sections_mean_max
        );
        dialog.change_tab(Tab::MakeAlignedStack.index());
        dialog.pop_up_context_menu_with_dataset(MouseEvent::default(), false);
        assert_eq!(
            dialog.context_popup.as_ref().unwrap().log_file,
            Some("blend.log")
        );
    }
}
