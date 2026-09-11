//! `IMOD/Etomo/src/etomo/ui/SharedStrings.java`.
//!
//! Strings shared by more than one place in the interface.  The Java class declares
//! nothing but `public static final String` constants, so the module is the same list
//! of `pub const`s, in source order and with the source's names.
#![allow(dead_code)]

/// Java `D_PHI_LABEL`.
pub const D_PHI_LABEL: &str = "Phi";
/// Java `D_THETA_LABEL`.
pub const D_THETA_LABEL: &str = "Theta";
/// Java `D_PSI_LABEL`.
pub const D_PSI_LABEL: &str = "Psi";
/// Java `EDGE_SHIFT_LABEL`.
pub const EDGE_SHIFT_LABEL: &str = "Edge shift";
/// Java `FLG_FAIR_REFERENCE_LABEL`.
pub const FLG_FAIR_REFERENCE_LABEL: &str = "Multiparticle reference";
/// Java `INIT_MOTL_LABEL`.
pub const INIT_MOTL_LABEL: &str = "Initial Motive List";
/// Java `INIT_MOTL_X_AND_Z_AXIS_LABEL`.
pub const INIT_MOTL_X_AND_Z_AXIS_LABEL: &str = "Align particle Y axes";
/// Java `INIT_MOTL_RANDOM_ROTATIONS`.
pub const INIT_MOTL_RANDOM_ROTATIONS: &str = "Uniform random rotations";
/// Java `INIT_MOTL_RANDOM_AXIAL_ROTATIONS`.
pub const INIT_MOTL_RANDOM_AXIAL_ROTATIONS: &str = "Random axial (Y) rotations";
/// Java `SAMPLE_SPHERE_LABEL`.
pub const SAMPLE_SPHERE_LABEL: &str = "Spherical Sampling for Theta and Psi";
/// Java `YAXIS_TYPE_LABEL`.
pub const YAXIS_TYPE_LABEL: &str = "Particle Y Axis";
/// Java `YAXIS_TYPE_Y_AXIS_LABEL`.
pub const YAXIS_TYPE_Y_AXIS_LABEL: &str = "Tomogram Y axis";
/// Java `YAXIS_TYPE_PARTICLE_MODEL_LABEL`.
pub const YAXIS_TYPE_PARTICLE_MODEL_LABEL: &str = "Particle model points";
/// Java `YAXIS_TYPE_CONTOUR_LABEL`.
pub const YAXIS_TYPE_CONTOUR_LABEL: &str = "End points of contour";
/// Java `N_WEIGHT_GROUP_LABEL`.
pub const N_WEIGHT_GROUP_LABEL: &str = "Weight groups";
/// Java `DEBUG_LEVEL_LABEL`.
pub const DEBUG_LEVEL_LABEL: &str = "Debug level";
/// Java `FLG_ALIGN_AVERAGES_LABEL`.
pub const FLG_ALIGN_AVERAGES_LABEL: &str = "Align averages to have their Y axes vertical";
/// Java `FLG_REMOVE_DUPLICATES_LABEL`.
pub const FLG_REMOVE_DUPLICATES_LABEL: &str = "Remove duplicates";
/// Java `BANDPASS_FILTERING_LABEL`.
pub const BANDPASS_FILTERING_LABEL: &str = "Bandpass filtering";
/// Java `FLG_ABS_VALUE_LABEL`.
pub const FLG_ABS_VALUE_LABEL: &str = "Use absolute value of cross-correlation";
/// Java `FLG_STRICT_SEARCH_LIMITS_LABEL`.
pub const FLG_STRICT_SEARCH_LIMITS_LABEL: &str = "Strict search limit checking";
/// Java `FLG_NO_REFERENCE_REFINEMENT_LABEL`.
pub const FLG_NO_REFERENCE_REFINEMENT_LABEL: &str = "No reference refinement (template matching)";
/// Java `CSV_FILES_LABEL`.
pub const CSV_FILES_LABEL: &str = "User supplied csv files";
/// Java `FLG_RANDOMIZE_LABEL`.
pub const FLG_RANDOMIZE_LABEL: &str = "Randomized particle selection";
/// Java `FLG_VOL_NAMES_ARE_TEMPLATES_LABEL`.
pub const FLG_VOL_NAMES_ARE_TEMPLATES_LABEL: &str = "File names are templates";
/// Java `STACK_TOOLTIP`.
pub const STACK_TOOLTIP: &str = "The A or B axis image stack for this dataset";
/// Java `IMOD_A_TOOLTIP`.
pub const IMOD_A_TOOLTIP: &str = "Opens the A axis stack";
/// Java `IMOD_B_TOOLTIP`.
pub const IMOD_B_TOOLTIP: &str = "Opens the B axis stack";
/// Java `EDIT_DATASET_TOOLTIP`.
pub const EDIT_DATASET_TOOLTIP: &str = "When \"Set\", dataset-specific values are in use.";
/// Java `DUAL_TOOLTIP`.
pub const DUAL_TOOLTIP: &str = "Dataset is dual axis.";
/// Java `MONTAGE_TOOLTIP`.
pub const MONTAGE_TOOLTIP: &str = "Dataset is a montage.";
/// Java `SKIP_TOOLTIP`.
pub const SKIP_TOOLTIP: &str = "Views to exclude from A or only tilt series";
/// Java `BSKIP_TOOLTIP`.
pub const BSKIP_TOOLTIP: &str = "Views to exclude from B tilt series";
/// Java `RAW_BOUNDARY_MODEL`.
pub const RAW_BOUNDARY_MODEL: &str = "Use a model drawn on raw stack to specify areas to include for autoseeding or patch tracking.  Open stack, or A stack for dual axis, to create model.";
/// Java `SURFACES_TO_ANALYZE_2_TOOLTIP`.
pub const SURFACES_TO_ANALYZE_2_TOOLTIP: &str =
    "Select beads on two surfaces when autoseeding and assume beads on two surfaces when aligning.";
/// Java `STEP_TOOLTIP`.
pub const STEP_TOOLTIP: &str = "Latest step successfully completed";
/// Java `TRUE_STRING`.
pub const TRUE_STRING: &str = "Yes";
/// Java `FALSE_STRING`.
pub const FALSE_STRING: &str = "No";
/// Java `OVERRIDE_TEXT`.
pub const OVERRIDE_TEXT: &str = ">OVERRIDE<";
/// Java `SIRT_LIKE_FILTER_RADIO_BUTTON_TOOLTIP`.
pub const SIRT_LIKE_FILTER_RADIO_BUTTON_TOOLTIP: &str =
    "Use a radial filter that produces the same result as iterating with SIRT";
/// Java `GAUSSIAN_FILTER_RADIO_BUTTON_TOOLTIP`.
pub const GAUSSIAN_FILTER_RADIO_BUTTON_TOOLTIP: &str = "Filter high frequencies with a Gaussian starting at the cutoff anfd falling off with the given sigma";
/// Java `HAMMING_LIKE_FILTER_RADIO_BUTTON_TOOLTIP`.
pub const HAMMING_LIKE_FILTER_RADIO_BUTTON_TOOLTIP: &str = "Filter high frequencies with a filter very similar to a Hamming window, starting at the given frequency";
/// Java `EXACT_FILTER_RADIO_BUTTON_TOOLTIP`.
pub const EXACT_FILTER_RADIO_BUTTON_TOOLTIP: &str =
    "Use 'exact filter' functions of Harauz and van Heel for the radial filter";
/// Java `CTF_CORRECTION_LABEL`.
pub const CTF_CORRECTION_LABEL: &str = "CTF Correction";
/// Java `FINAL_ALIGNED_STACK_LABEL`.
pub const FINAL_ALIGNED_STACK_LABEL: &str = "Final Aligned Stack";
/// Java `CTF_PLOTTER_LABEL`.
pub const CTF_PLOTTER_LABEL: &str = "CTF Plotter";
/// Java `EXPECTED_DEFOCUS_LABEL`.
pub const EXPECTED_DEFOCUS_LABEL: &str = "expected defocus";
/// Java `DOSE_WEIGHTING_LABEL`.
pub const DOSE_WEIGHTING_LABEL: &str = "Dose Weighting";
/// Java `_2D_FILTER_LABEL`.
pub const _2D_FILTER_LABEL: &str = "2D Filter";
/// Java `EXCLUDE_LIST_LABEL`.
pub const EXCLUDE_LIST_LABEL: &str = "Exclude particles ";
/// Java `INCLUDE_LIST_LABEL`.
pub const INCLUDE_LIST_LABEL: &str = "Include particles ";
/// Java `SELECT_CLASS_ID_LABEL`.
pub const SELECT_CLASS_ID_LABEL: &str = "Average only members of classes ";
/// Java `FLG_ELEVATION_COMPENSATION_LABEL`.
pub const FLG_ELEVATION_COMPENSATION_LABEL: &str = "Elevation Compensation";
/// Java `FLG_FRM_LABEL`.
pub const FLG_FRM_LABEL: &str = "Fast rotational matching";
/// Java `FLG_ALLOW_MASKED_CORRELATION_LABEL`.
pub const FLG_ALLOW_MASKED_CORRELATION_LABEL: &str = "Masked correlation computation";
/// Java `FLG_FILTER_REF_ONLY_LABEL`.
pub const FLG_FILTER_REF_ONLY_LABEL: &str = "Filter reference only";
/// Java `FLG_SEARCH_ALONG_PARTICLE_AXES_LABEL`.
pub const FLG_SEARCH_ALONG_PARTICLE_AXES_LABEL: &str = "Search along particle axes";
/// Java `FLG_FP_WEDGE_MASK_LABEL`.
pub const FLG_FP_WEDGE_MASK_LABEL: &str = "Floating point wedge mask";
/// Java `YAXIS_SYMMETRY_LABEL`.
pub const YAXIS_SYMMETRY_LABEL: &str = "Per iteration c<N> axial search order ";
/// Java `FLG_USE_EXTRACTED_PARTICLES_LABEL`.
pub const FLG_USE_EXTRACTED_PARTICLES_LABEL: &str = "Use previously extracted particles";
/// Java `CN_SYMMETRIC_AVERAGING_LABEL`.
pub const CN_SYMMETRIC_AVERAGING_LABEL: &str = "c<N> symmetric averaging with N";
/// Java `FLG_CN_MASKING_LABEL`.
pub const FLG_CN_MASKING_LABEL: &str = "Masking during c<N> averaging";
/// Java `USER_COMMANDS_LABEL`.
pub const USER_COMMANDS_LABEL: &str = "User commands ";

// TODO(unit): needs etomo/storage/CpuAdoc.java - `PARALLEL_PROCESSING_REQUIRED_MESSAGE`
// concatenates `CpuAdoc.FILE_NAME` and `CpuAdoc.MAN_PAGE` around
// `ProcessName.MAN.toString()`, and those two constants have no module yet.
