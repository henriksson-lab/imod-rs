//! `IMOD/Etomo/src/etomo/storage/DirectiveDef.java`.
//!
//! Describes directives as required.  Each instance should be unique.
//!
//! **Shape.**  A Java `DirectiveDef` is a shared, mutable object: the preconstructed
//! `public static final` instances and every instance `getInstance` builds are kept in
//! the static `MAP`, and `getInstance` later corrects `aOnlyDirective` /
//! `bOnlyDirective` on instances already handed out, while `loadDirectiveDescr` fills
//! in the description fields lazily.  Here every instance's state lives in one
//! process-global registry, in construction order, and [`DirectiveDef`] is a `Copy`
//! handle (the index into that registry), so it is `Send + Sync` and every holder sees
//! the same object, as a Java reference does.  The 132 preconstructed instances are
//! associated constants with their Java names; the registry builds them first, in
//! declaration order, so the first-wins rule of `MAP.put` behaves as in the source.
//!
//! `equals` and `hashCode` compare standard keys; see [`DirectiveDef::equals`].
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use regex::Regex;

use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::storage::directive_attribute::Match;
use crate::imod::etomo::storage::directive_descr_file;
use crate::imod::etomo::storage::directive_file::{Comfile, Command, Module};
use crate::imod::etomo::storage::directive_type::DirectiveType;
use crate::imod::etomo::storage::directive_value_type::DirectiveValueType;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;
use crate::imod::etomo::util::utilities::java_lang_string_split;

/// Java `RUN_TIME_ANY_AXIS_TAG`.
pub const RUN_TIME_ANY_AXIS_TAG: &str = "any";
/// Java `RUN_TIME_A_AXIS_TAG`.
pub const RUN_TIME_A_AXIS_TAG: &str = "a";
/// Java `B_AXIS_TAG`.
pub const B_AXIS_TAG: &str = "b";

/// Java private `RUN_TIME_BIN_BY_FACTOR_NAME`.
const RUN_TIME_BIN_BY_FACTOR_NAME: &str = "binByFactor";
/// Java private `COM_PARAM_BIN_BY_FACTOR_NAME`.
const COM_PARAM_BIN_BY_FACTOR_NAME: &str = "BinByFactor";
/// Java private `BINNING_NAME`.
const BINNING_NAME: &str = "binning";
/// Java private `THICKNESS_NAME`.
const THICKNESS_NAME: &str = "thickness";
/// Java `TRUE_VALUE`.
pub const TRUE_VALUE: &str = "1";
/// Java `FALSE_VALUE`.
pub const FALSE_VALUE: &str = "0";
/// Java `SAMPLE_TYPE_PLASTIC_SECTION_VALUE`.
pub const SAMPLE_TYPE_PLASTIC_SECTION_VALUE: &str = "1";
/// Java `SURFACES_TO_ANALYZE_DEFAULT`.
pub const SURFACES_TO_ANALYZE_DEFAULT: &str = "2";

/// The fields of one Java `DirectiveDef` object.
#[derive(Clone, Debug)]
struct DirectiveDefFields {
    /// Java field `directiveType`.  Every constructor call in the source passes a
    /// non-null type (`getInstance` returns before constructing when the type is not
    /// recognized, because the name then stays null).
    directive_type: DirectiveType,
    /// Java field `module`.
    module: Option<String>,
    /// Java field `comfile`.
    comfile: Option<String>,
    /// Java field `command`.
    command: Option<String>,
    /// Java field `name`: required, and never null or empty for a constructed instance.
    name: String,
    /// Java field `preconstructed`: should only be true when the instance is a static
    /// member of this class.
    preconstructed: bool,
    /// Java field `aOnlyDirective`: do not change if preconstructed is true.  True for A
    /// axis CopyArg if B axis CopyArg twin exists.  Example setupset.copyarg.skip.
    a_only_directive: bool,
    /// Java field `bOnlyDirective`: do not change if preconstructed is true.  True for B
    /// axis CopyArg.  Example setupset.copyarg.bskip.
    b_only_directive: bool,
    /// Java field `bool`.
    bool: bool,
    /// Java field `templateA`.
    template_a: bool,
    /// Java field `templateB`.
    template_b: bool,
    /// Java field `batchA`.
    batch_a: bool,
    /// Java field `batchB`.
    batch_b: bool,
    /// Java field `directiveDescrLoaded`.
    directive_descr_loaded: bool,
    /// Java field `description`.
    description: Option<String>,
}

/// Every Java `DirectiveDef` object, and the static `MAP`.
struct Registry {
    /// The objects, in construction order; a [`DirectiveDef`] is an index here.
    defs: Vec<DirectiveDefFields>,
    /// Java static `MAP`: standard key to instance.
    map: HashMap<String, usize>,
}

/// The registry, with the preconstructed instances built in declaration order.
static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    let mut registry = Registry {
        defs: Vec::new(),
        map: HashMap::new(),
    };
    preconstruct(&mut registry);
    Mutex::new(registry)
});

/// Java `DirectiveDef`: a handle to one shared instance.
#[derive(Clone, Copy, Debug)]
pub struct DirectiveDef(usize);

/// Java `directive.split("\\" + AutodocTokenizer.SEPARATOR_CHAR)`, which drops trailing
/// empty strings.
fn split_directive(directive: &str) -> Vec<String> {
    java_lang_string_split(
        directive,
        &Regex::new(&regex::escape(SEPARATOR_CHAR)).unwrap(),
    )
}

/// Java `String.substring(1)` of a name: throws for an empty string, which a
/// constructed name never is.
fn substring_1(name: &str) -> String {
    name.chars().skip(1).collect()
}

/// The preconstructed `public static final` instances, in declaration order.
fn preconstruct(registry: &mut Registry) {
    // copyarg
    // BINNING
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, BINNING_NAME);
    // CS
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "Cs");
    // CTF_NOISE
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "ctfnoise");
    // DEFOCUS
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "defocus");
    // DISTORT
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "distort");
    // DUAL
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "dual");
    // EXTRACT
    DirectiveDef::new_copy_arg_a(registry, DirectiveType::COPY_ARG, "extract", true);
    // BEXTRACT
    DirectiveDef::get_b_instance(registry, DirectiveType::COPY_ARG, "bextract");
    // FIRST_INC
    DirectiveDef::new_copy_arg_a(registry, DirectiveType::COPY_ARG, "firstinc", true);
    // BFIRST_INC
    DirectiveDef::get_b_instance(registry, DirectiveType::COPY_ARG, "bfirstinc");
    // FOCUS
    DirectiveDef::new_copy_arg_a(registry, DirectiveType::COPY_ARG, "focus", true);
    // BFOCUS
    DirectiveDef::get_b_instance(registry, DirectiveType::COPY_ARG, "bfocus");
    // GOLD
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "gold");
    // GRADIENT
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "gradient");
    // HALF_FLOAT
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "halffloat");
    // MONTAGE
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "montage");
    // NAME
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "name");
    // PIXEL
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "pixel");
    // ROTATION
    DirectiveDef::new_copy_arg_a(registry, DirectiveType::COPY_ARG, "rotation", true);
    // BROTATION
    DirectiveDef::get_b_instance(registry, DirectiveType::COPY_ARG, "brotation");
    // SKIP
    DirectiveDef::new_copy_arg_a(registry, DirectiveType::COPY_ARG, "skip", true);
    // BSKIP
    DirectiveDef::get_b_instance(registry, DirectiveType::COPY_ARG, "bskip");
    // STACK_EXT
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "stackext");
    // TWODIR
    DirectiveDef::new_copy_arg_a(registry, DirectiveType::COPY_ARG, "twodir", true);
    // BTWODIR
    DirectiveDef::get_b_instance(registry, DirectiveType::COPY_ARG, "btwodir");
    // USE_RAW_TLT
    DirectiveDef::new_copy_arg_a(registry, DirectiveType::COPY_ARG, "userawtlt", true);
    // BUSE_RAW_TLT
    DirectiveDef::get_b_instance(registry, DirectiveType::COPY_ARG, "buserawtlt");
    // VOLTAGE
    DirectiveDef::new_name(registry, DirectiveType::COPY_ARG, "voltage");
    // DOSESYM
    DirectiveDef::new_copy_arg_a(registry, DirectiveType::COPY_ARG, "dosesym", true);
    // BDOSESYM
    DirectiveDef::get_b_instance(registry, DirectiveType::COPY_ARG, "bdosesym");
    // CURRENT_B_STACK_EXT
    DirectiveDef::new_name(registry, DirectiveType::SETUP_SET, "currentBStackExt");
    // CURRENT_STACK_EXT
    DirectiveDef::new_name(registry, DirectiveType::SETUP_SET, "currentStackExt");
    // DATASET_DIRECTORY
    DirectiveDef::new_name(registry, DirectiveType::SETUP_SET, "datasetDirectory");
    // SCAN_HEADER
    DirectiveDef::new_name(registry, DirectiveType::SETUP_SET, "scanHeader");
    // SCOPE_TEMPLATE
    DirectiveDef::new_name(registry, DirectiveType::SETUP_SET, "scopeTemplate");
    // SYSTEM_TEMPLATE
    DirectiveDef::new_name(registry, DirectiveType::SETUP_SET, "systemTemplate");
    // USER_TEMPLATE
    DirectiveDef::new_name(registry, DirectiveType::SETUP_SET, "userTemplate");
    // BIN_BY_FACTOR_FOR_ALIGNED_STACK
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::ALIGNED_STACK),
        RUN_TIME_BIN_BY_FACTOR_NAME,
    );
    // CORRECT_CTF
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::ALIGNED_STACK),
        "correctCTF",
    );
    // ERASE_GOLD
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::ALIGNED_STACK),
        "eraseGold",
    );
    // FILTER_STACK
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::ALIGNED_STACK),
        "filterStack",
    );
    // LINEAR_INTERPOLATION
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::ALIGNED_STACK),
        "linearInterpolation",
    );
    // SIZE_IN_X_AND_Y
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::ALIGNED_STACK),
        "sizeInXandY",
    );
    // NUMBER_OF_RUNS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::BEAD_TRACKING),
        "numberOfRuns",
    );
    // DO_SIRT_IF_BOTH
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::COMBINE),
        "doSIRTifBoth",
    );
    // EXTRA_TARGETS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::COMBINE),
        "extraTargets",
    );
    // FINAL_PATCH_SIZE
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::COMBINE),
        "finalPatchSize",
    );
    // FIND_SEC_BOX_SIZE
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::COMBINE),
        "findSecBoxSize",
    );
    // FIND_SEC_NUM_SCALES
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::COMBINE),
        "findSecNumScales",
    );
    // LOW_FROM_BOTH_RADIUS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::COMBINE),
        "lowFromBothRadius",
    );
    // MATCH_A_TO_B_THICK_RATIO
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::COMBINE),
        "matchAtoBThickRatio",
    );
    // PATCH_SIZE
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::COMBINE),
        "patchSize",
    );
    // WEDGE_REDUCTION
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::COMBINE),
        "wedgeReduction",
    );
    // CORRECT_FOR_X_AXIS_TILT
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::CTF_CORRECTION),
        "correctForXAxisTilt",
    );
    // AUTO_FIT_RANGE_AND_STEP
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::CTF_PLOTTING),
        "autoFitRangeAndStep",
    );
    // DELETE_OLD_FILES
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::EXCLUDE_VIEWS),
        "deleteOldFiles",
    );
    // FIDUCIALLESS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::FIDUCIALS),
        "fiducialless",
    );
    // SEEDING_METHOD
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::FIDUCIALS),
        "seedingMethod",
    );
    // TRACKING_METHOD
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::FIDUCIALS),
        "trackingMethod",
    );
    // BINNING_FOR_GOLD_ERASING
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::GOLD_ERASING),
        BINNING_NAME,
    );
    // EXTRA_DIAMETER
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::GOLD_ERASING),
        "extraDiameter",
    );
    // THICKNESS_FOR_GOLD_ERASING
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::GOLD_ERASING),
        THICKNESS_NAME,
    );
    // ITERATIONS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::NAD),
        "iterations",
    );
    // K_VALUE
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::NAD),
        "Kvalue",
    );
    // CHUNK_MEMORY_MB
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::NAD),
        "chunkMemoryMB",
    );
    // ADJUST_TILT_ANGLES
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::PATCH_TRACKING),
        "adjustTiltAngles",
    );
    // CONTOUR_PIECES
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::PATCH_TRACKING),
        "contourPieces",
    );
    // RAW_BOUNDARY_MODEL_FOR_PATCH_TRACKING
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::PATCH_TRACKING),
        "rawBoundaryModel",
    );
    // BIN_BY_FACTOR_FOR_POSITIONING
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::POSITIONING),
        RUN_TIME_BIN_BY_FACTOR_NAME,
    );
    // CENTER_ON_GOLD
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::POSITIONING),
        "centerOnGold",
    );
    // HAS_GOLD_BEADS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::POSITIONING),
        "hasGoldBeads",
    );
    // SAMPLE_TYPE
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::POSITIONING),
        "sampleType",
    );
    // THICKNESS_FOR_POSITIONING
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::POSITIONING),
        THICKNESS_NAME,
    );
    // WHOLE_TOMOGRAM
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::POSITIONING),
        "wholeTomogram",
    );
    // DO_TRIMVOL
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::POSTPROCESS),
        "doTrimvol",
    );
    // ARCHIVE_ORIGINAL
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::PREPROCESSING),
        "archiveOriginal",
    );
    // DARK_EXCLUDE_RATIO
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::PREPROCESSING),
        "darkExcludeRatio",
    );
    // DARK_EXCLUDE_FRACTION
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::PREPROCESSING),
        "darkExcludeFraction",
    );
    // END_EXCLUDE_CRITERION
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::PREPROCESSING),
        "endExcludeCriterion",
    );
    // REMOVE_EXCLUDED_VIEWS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::PREPROCESSING),
        "removeExcludedViews",
    );
    // REMOVE_XRAYS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::PREPROCESSING),
        "removeXrays",
    );
    // NUMBER_OF_MARKERS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RAPTOR),
        "numberOfMarkers",
    );
    // USE_ALIGNED_STACK
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RAPTOR),
        "useAlignedStack",
    );
    // BINNED_THICKNESS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RECONSTRUCTION),
        "binnedThickness",
    );
    // DO_BACKPROJ_ALSO
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RECONSTRUCTION),
        "doBackprojAlso",
    );
    // EXTRA_THICKNESS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RECONSTRUCTION),
        "extraThickness",
    );
    // FALLBACK_THICKNESS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RECONSTRUCTION),
        "fallbackThickness",
    );
    // USE_SIRT
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RECONSTRUCTION),
        "useSirt",
    );
    // MIN_MEASUREMENT_RATIO
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RESTRICT_ALIGN),
        "minMeasurementRatio",
    );
    // ORDER_OF_RESTRICTIONS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RESTRICT_ALIGN),
        "orderOfRestrictions",
    );
    // SKIP_BEAM_TILT_WITH_ONE_ROT
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RESTRICT_ALIGN),
        "skipBeamTiltWithOneRot",
    );
    // TARGET_MEASUREMENT_RATIO
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RESTRICT_ALIGN),
        "targetMeasurementRatio",
    );
    // RAW_BOUNDARY_MODEL_FOR_SEED_FINDING
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::SEED_FINDING),
        "rawBoundaryModel",
    );
    // ENABLE_STRETCHING
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TILT_ALIGNMENT),
        "enableStretching",
    );
    // DO_A_OR_B_OF_DUAL_AXIS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "doAorBofDualAxis",
    );
    // DO_SIRT_IF_BOTH_FOR_TRIMVOL
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "doSIRTifBoth",
    );
    // FIND_SEC_ADD_THICKNESS
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "findSecAddThickness",
    );
    // REORIENT
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "reorient",
    );
    // SIZE_IN_X
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "sizeInX",
    );
    // SIZE_IN_Y
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "sizeInY",
    );
    // SCALE_FROM_X
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "scaleFromX",
    );
    // SCALE_FROM_Y
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "scaleFromY",
    );
    // SCALE_FROM_Z
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "scaleFromZ",
    );
    // SCALE_TO_MEAN_SD
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        "scaleToMeanSD",
    );
    // THICKNESS_FOR_TRIMVOL
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::TRIMVOL),
        THICKNESS_NAME,
    );
    // REPLACE_STEP_NINE
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::REPLACE_STEP),
        "9",
    );
    // REPLACE_STEP_THIRTEEN
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::REPLACE_STEP),
        "13",
    );
    // RUN_AFTER_STEP_NINE
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RUN_AFTER_STEP),
        "9",
    );
    // RUN_AFTER_STEP_THIRTEEN
    DirectiveDef::new_run_time(
        registry,
        DirectiveType::RUN_TIME,
        Some(&Module::RUN_AFTER_STEP),
        "13",
    );
    // LOCAL_ALIGNMENTS
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::ALIGN),
        Some(&Command::TILTALIGN),
        "LocalAlignments",
    );
    // SURFACES_TO_ANALYZE
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::ALIGN),
        Some(&Command::TILTALIGN),
        "SurfacesToAnalyze",
    );
    // TARGET_NUMBER_OF_BEADS
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::AUTOFIDSEED),
        Some(&Command::AUTOFIDSEED),
        "TargetNumberOfBeads",
    );
    // MIN_GUESS_NUM_BEADS
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::AUTOFIDSEED),
        Some(&Command::AUTOFIDSEED),
        "MinGuessNumBeads",
    );
    // TARGET_DENSITY_OF_BEADS
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::AUTOFIDSEED),
        Some(&Command::AUTOFIDSEED),
        "TargetDensityOfBeads",
    );
    // TWO_SURFACES
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::AUTOFIDSEED),
        Some(&Command::AUTOFIDSEED),
        "TwoSurfaces",
    );
    // MODEL_FILE
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::ERASER),
        Some(&Command::CCDERASER),
        "ModelFile",
    );
    // SLAB_THICKNESS_IN_NM
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::CTF_3D_SETUP),
        Some(&Command::CTF_3D_SETUP),
        "SlabThicknessInNm",
    );
    // SCAN_DEFOCUS_RANGE
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::CTF_PLOTTER),
        Some(&Command::CTF_PLOTTER),
        "ScanDefocusRange",
    );
    // TUNE_FITTING_AND_SAMPLING
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::CTF_PLOTTER),
        Some(&Command::CTF_PLOTTER),
        "TuneFittingAndSampling",
    );
    // EXPAND_CIRCLE_ITERATIONS
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::GOLD_ERASER),
        Some(&Command::CCDERASER),
        "ExpandCircleIterations",
    );
    // BIN_BY_FACTOR_FOR_PREBLEND
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::PREBLEND),
        Some(&Command::BLENDMONT),
        COM_PARAM_BIN_BY_FACTOR_NAME,
    );
    // BIN_BY_FACTOR_FOR_PRENEWST
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::PRENEWST),
        Some(&Command::NEWSTACK),
        COM_PARAM_BIN_BY_FACTOR_NAME,
    );
    // LEAVE_ITERATIONS
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::SIRTSETUP),
        Some(&Command::SIRTSETUP),
        "LeaveIterations",
    );
    // SCALE_TO_INTEGER
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::SIRTSETUP),
        Some(&Command::SIRTSETUP),
        "ScaleToInteger",
    );
    // FAKE_SIRT_ITERATIONS
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::TILT),
        Some(&Command::TILT),
        "FakeSIRTiterations",
    );
    // THICKNESS_FOR_TILT
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::TILT),
        Some(&Command::TILT),
        "THICKNESS",
    );
    // LOCAL_AREA_TARGET_SIZE
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::TRACK),
        Some(&Command::BEADTRACK),
        "LocalAreaTargetSize",
    );
    // LENGTH_OF_PIECES
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::XCORR_PT),
        Some(&Command::IMODCHOPCONTS),
        "LengthOfPieces",
    );
    // SEARCH_MAG_CHANGES
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::XCORR),
        Some(&Command::TILTXCORR),
        "SearchMagChanges",
    );
    // NUMBER_OF_PATCHES_X_AND_Y
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::XCORR_PT),
        Some(&Command::TILTXCORR),
        "NumberOfPatchesXandY",
    );
    // OVERLAP_OF_PATCHES_X_AND_Y
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::XCORR_PT),
        Some(&Command::TILTXCORR),
        "OverlapOfPatchesXandY",
    );
    // SIZE_OF_PATCHES_X_AND_Y
    DirectiveDef::new_com_param(
        registry,
        DirectiveType::COM_PARAM,
        Some(&Comfile::XCORR_PT),
        Some(&Command::TILTXCORR),
        "SizeOfPatchesXandY",
    );
}

impl DirectiveDef {
    /// Java `BINNING = new DirectiveDef(DirectiveType.COPY_ARG, BINNING_NAME)`.
    pub const BINNING: DirectiveDef = DirectiveDef(0);
    /// Java `CS = new DirectiveDef(DirectiveType.COPY_ARG, "Cs")`.
    pub const CS: DirectiveDef = DirectiveDef(1);
    /// Java `CTF_NOISE = new DirectiveDef(DirectiveType.COPY_ARG, "ctfnoise")`.
    pub const CTF_NOISE: DirectiveDef = DirectiveDef(2);
    /// Java `DEFOCUS = new DirectiveDef(DirectiveType.COPY_ARG, "defocus")`.
    pub const DEFOCUS: DirectiveDef = DirectiveDef(3);
    /// Java `DISTORT = new DirectiveDef(DirectiveType.COPY_ARG, "distort")`.
    pub const DISTORT: DirectiveDef = DirectiveDef(4);
    /// Java `DUAL = new DirectiveDef(DirectiveType.COPY_ARG, "dual")`.
    pub const DUAL: DirectiveDef = DirectiveDef(5);
    /// Java `EXTRACT = new DirectiveDef(DirectiveType.COPY_ARG, "extract", true)`.
    pub const EXTRACT: DirectiveDef = DirectiveDef(6);
    /// Java `BEXTRACT = getBInstance(DirectiveType.COPY_ARG, "bextract")`.
    pub const BEXTRACT: DirectiveDef = DirectiveDef(7);
    /// Java `FIRST_INC = new DirectiveDef(DirectiveType.COPY_ARG, "firstinc", true)`.
    pub const FIRST_INC: DirectiveDef = DirectiveDef(8);
    /// Java `BFIRST_INC = getBInstance(DirectiveType.COPY_ARG, "bfirstinc")`.
    pub const BFIRST_INC: DirectiveDef = DirectiveDef(9);
    /// Java `FOCUS = new DirectiveDef(DirectiveType.COPY_ARG, "focus", true)`.
    pub const FOCUS: DirectiveDef = DirectiveDef(10);
    /// Java `BFOCUS = getBInstance(DirectiveType.COPY_ARG, "bfocus")`.
    pub const BFOCUS: DirectiveDef = DirectiveDef(11);
    /// Java `GOLD = new DirectiveDef(DirectiveType.COPY_ARG, "gold")`.
    pub const GOLD: DirectiveDef = DirectiveDef(12);
    /// Java `GRADIENT = new DirectiveDef(DirectiveType.COPY_ARG, "gradient")`.
    pub const GRADIENT: DirectiveDef = DirectiveDef(13);
    /// Java `HALF_FLOAT = new DirectiveDef(DirectiveType.COPY_ARG, "halffloat")`.
    pub const HALF_FLOAT: DirectiveDef = DirectiveDef(14);
    /// Java `MONTAGE = new DirectiveDef(DirectiveType.COPY_ARG, "montage")`.
    pub const MONTAGE: DirectiveDef = DirectiveDef(15);
    /// Java `NAME = new DirectiveDef(DirectiveType.COPY_ARG, "name")`.
    pub const NAME: DirectiveDef = DirectiveDef(16);
    /// Java `PIXEL = new DirectiveDef(DirectiveType.COPY_ARG, "pixel")`.
    pub const PIXEL: DirectiveDef = DirectiveDef(17);
    /// Java `ROTATION = new DirectiveDef(DirectiveType.COPY_ARG, "rotation", true)`.
    pub const ROTATION: DirectiveDef = DirectiveDef(18);
    /// Java `BROTATION = getBInstance(DirectiveType.COPY_ARG, "brotation")`.
    pub const BROTATION: DirectiveDef = DirectiveDef(19);
    /// Java `SKIP = new DirectiveDef(DirectiveType.COPY_ARG, "skip", true)`.
    pub const SKIP: DirectiveDef = DirectiveDef(20);
    /// Java `BSKIP = getBInstance(DirectiveType.COPY_ARG, "bskip")`.
    pub const BSKIP: DirectiveDef = DirectiveDef(21);
    /// Java `STACK_EXT = new DirectiveDef(DirectiveType.COPY_ARG, "stackext")`.
    pub const STACK_EXT: DirectiveDef = DirectiveDef(22);
    /// Java `TWODIR = new DirectiveDef(DirectiveType.COPY_ARG, "twodir", true)`.
    pub const TWODIR: DirectiveDef = DirectiveDef(23);
    /// Java `BTWODIR = getBInstance(DirectiveType.COPY_ARG, "btwodir")`.
    pub const BTWODIR: DirectiveDef = DirectiveDef(24);
    /// Java `USE_RAW_TLT = new DirectiveDef(DirectiveType.COPY_ARG, "userawtlt", true)`.
    pub const USE_RAW_TLT: DirectiveDef = DirectiveDef(25);
    /// Java `BUSE_RAW_TLT = getBInstance(DirectiveType.COPY_ARG, "buserawtlt")`.
    pub const BUSE_RAW_TLT: DirectiveDef = DirectiveDef(26);
    /// Java `VOLTAGE = new DirectiveDef(DirectiveType.COPY_ARG, "voltage")`.
    pub const VOLTAGE: DirectiveDef = DirectiveDef(27);
    /// Java `DOSESYM = new DirectiveDef(DirectiveType.COPY_ARG, "dosesym", true)`.
    pub const DOSESYM: DirectiveDef = DirectiveDef(28);
    /// Java `BDOSESYM = getBInstance(DirectiveType.COPY_ARG, "bdosesym")`.
    pub const BDOSESYM: DirectiveDef = DirectiveDef(29);
    /// Java `CURRENT_B_STACK_EXT = new DirectiveDef(DirectiveType.SETUP_SET, "currentBStackExt")`.
    pub const CURRENT_B_STACK_EXT: DirectiveDef = DirectiveDef(30);
    /// Java `CURRENT_STACK_EXT = new DirectiveDef(DirectiveType.SETUP_SET, "currentStackExt")`.
    pub const CURRENT_STACK_EXT: DirectiveDef = DirectiveDef(31);
    /// Java `DATASET_DIRECTORY = new DirectiveDef(DirectiveType.SETUP_SET, "datasetDirectory")`.
    pub const DATASET_DIRECTORY: DirectiveDef = DirectiveDef(32);
    /// Java `SCAN_HEADER = new DirectiveDef(DirectiveType.SETUP_SET, "scanHeader")`.
    pub const SCAN_HEADER: DirectiveDef = DirectiveDef(33);
    /// Java `SCOPE_TEMPLATE = new DirectiveDef(DirectiveType.SETUP_SET, "scopeTemplate")`.
    pub const SCOPE_TEMPLATE: DirectiveDef = DirectiveDef(34);
    /// Java `SYSTEM_TEMPLATE = new DirectiveDef(DirectiveType.SETUP_SET, "systemTemplate")`.
    pub const SYSTEM_TEMPLATE: DirectiveDef = DirectiveDef(35);
    /// Java `USER_TEMPLATE = new DirectiveDef(DirectiveType.SETUP_SET, "userTemplate")`.
    pub const USER_TEMPLATE: DirectiveDef = DirectiveDef(36);
    /// Java `BIN_BY_FACTOR_FOR_ALIGNED_STACK = new DirectiveDef( DirectiveType.RUN_TIME, Module.ALIGNED_STACK, RUN_TIME_BIN_BY_FACTOR_NAME)`.
    pub const BIN_BY_FACTOR_FOR_ALIGNED_STACK: DirectiveDef = DirectiveDef(37);
    /// Java `CORRECT_CTF = new DirectiveDef(DirectiveType.RUN_TIME, Module.ALIGNED_STACK, "correctCTF")`.
    pub const CORRECT_CTF: DirectiveDef = DirectiveDef(38);
    /// Java `ERASE_GOLD = new DirectiveDef(DirectiveType.RUN_TIME, Module.ALIGNED_STACK, "eraseGold")`.
    pub const ERASE_GOLD: DirectiveDef = DirectiveDef(39);
    /// Java `FILTER_STACK = new DirectiveDef(DirectiveType.RUN_TIME, Module.ALIGNED_STACK, "filterStack")`.
    pub const FILTER_STACK: DirectiveDef = DirectiveDef(40);
    /// Java `LINEAR_INTERPOLATION = new DirectiveDef(DirectiveType.RUN_TIME, Module.ALIGNED_STACK, "linearInterpolation")`.
    pub const LINEAR_INTERPOLATION: DirectiveDef = DirectiveDef(41);
    /// Java `SIZE_IN_X_AND_Y = new DirectiveDef(DirectiveType.RUN_TIME, Module.ALIGNED_STACK, "sizeInXandY")`.
    pub const SIZE_IN_X_AND_Y: DirectiveDef = DirectiveDef(42);
    /// Java `NUMBER_OF_RUNS = new DirectiveDef(DirectiveType.RUN_TIME, Module.BEAD_TRACKING, "numberOfRuns")`.
    pub const NUMBER_OF_RUNS: DirectiveDef = DirectiveDef(43);
    /// Java `DO_SIRT_IF_BOTH = new DirectiveDef(DirectiveType.RUN_TIME, Module.COMBINE, "doSIRTifBoth")`.
    pub const DO_SIRT_IF_BOTH: DirectiveDef = DirectiveDef(44);
    /// Java `EXTRA_TARGETS = new DirectiveDef(DirectiveType.RUN_TIME, Module.COMBINE, "extraTargets")`.
    pub const EXTRA_TARGETS: DirectiveDef = DirectiveDef(45);
    /// Java `FINAL_PATCH_SIZE = new DirectiveDef(DirectiveType.RUN_TIME, Module.COMBINE, "finalPatchSize")`.
    pub const FINAL_PATCH_SIZE: DirectiveDef = DirectiveDef(46);
    /// Java `FIND_SEC_BOX_SIZE = new DirectiveDef(DirectiveType.RUN_TIME, Module.COMBINE, "findSecBoxSize")`.
    pub const FIND_SEC_BOX_SIZE: DirectiveDef = DirectiveDef(47);
    /// Java `FIND_SEC_NUM_SCALES = new DirectiveDef(DirectiveType.RUN_TIME, Module.COMBINE, "findSecNumScales")`.
    pub const FIND_SEC_NUM_SCALES: DirectiveDef = DirectiveDef(48);
    /// Java `LOW_FROM_BOTH_RADIUS = new DirectiveDef(DirectiveType.RUN_TIME, Module.COMBINE, "lowFromBothRadius")`.
    pub const LOW_FROM_BOTH_RADIUS: DirectiveDef = DirectiveDef(49);
    /// Java `MATCH_A_TO_B_THICK_RATIO = new DirectiveDef(DirectiveType.RUN_TIME, Module.COMBINE, "matchAtoBThickRatio")`.
    pub const MATCH_A_TO_B_THICK_RATIO: DirectiveDef = DirectiveDef(50);
    /// Java `PATCH_SIZE = new DirectiveDef(DirectiveType.RUN_TIME, Module.COMBINE, "patchSize")`.
    pub const PATCH_SIZE: DirectiveDef = DirectiveDef(51);
    /// Java `WEDGE_REDUCTION = new DirectiveDef(DirectiveType.RUN_TIME, Module.COMBINE, "wedgeReduction")`.
    pub const WEDGE_REDUCTION: DirectiveDef = DirectiveDef(52);
    /// Java `CORRECT_FOR_X_AXIS_TILT = new DirectiveDef( DirectiveType.RUN_TIME, Module.CTF_CORRECTION, "correctForXAxisTilt")`.
    pub const CORRECT_FOR_X_AXIS_TILT: DirectiveDef = DirectiveDef(53);
    /// Java `AUTO_FIT_RANGE_AND_STEP = new DirectiveDef(DirectiveType.RUN_TIME, Module.CTF_PLOTTING, "autoFitRangeAndStep")`.
    pub const AUTO_FIT_RANGE_AND_STEP: DirectiveDef = DirectiveDef(54);
    /// Java `DELETE_OLD_FILES = new DirectiveDef(DirectiveType.RUN_TIME, Module.EXCLUDE_VIEWS, "deleteOldFiles")`.
    pub const DELETE_OLD_FILES: DirectiveDef = DirectiveDef(55);
    /// Java `FIDUCIALLESS = new DirectiveDef(DirectiveType.RUN_TIME, Module.FIDUCIALS, "fiducialless")`.
    pub const FIDUCIALLESS: DirectiveDef = DirectiveDef(56);
    /// Java `SEEDING_METHOD = new DirectiveDef(DirectiveType.RUN_TIME, Module.FIDUCIALS, "seedingMethod")`.
    pub const SEEDING_METHOD: DirectiveDef = DirectiveDef(57);
    /// Java `TRACKING_METHOD = new DirectiveDef(DirectiveType.RUN_TIME, Module.FIDUCIALS, "trackingMethod")`.
    pub const TRACKING_METHOD: DirectiveDef = DirectiveDef(58);
    /// Java `BINNING_FOR_GOLD_ERASING = new DirectiveDef(DirectiveType.RUN_TIME, Module.GOLD_ERASING, BINNING_NAME)`.
    pub const BINNING_FOR_GOLD_ERASING: DirectiveDef = DirectiveDef(59);
    /// Java `EXTRA_DIAMETER = new DirectiveDef(DirectiveType.RUN_TIME, Module.GOLD_ERASING, "extraDiameter")`.
    pub const EXTRA_DIAMETER: DirectiveDef = DirectiveDef(60);
    /// Java `THICKNESS_FOR_GOLD_ERASING = new DirectiveDef(DirectiveType.RUN_TIME, Module.GOLD_ERASING, THICKNESS_NAME)`.
    pub const THICKNESS_FOR_GOLD_ERASING: DirectiveDef = DirectiveDef(61);
    /// Java `ITERATIONS = new DirectiveDef(DirectiveType.RUN_TIME, Module.NAD, "iterations")`.
    pub const ITERATIONS: DirectiveDef = DirectiveDef(62);
    /// Java `K_VALUE = new DirectiveDef(DirectiveType.RUN_TIME, Module.NAD, "Kvalue")`.
    pub const K_VALUE: DirectiveDef = DirectiveDef(63);
    /// Java `CHUNK_MEMORY_MB = new DirectiveDef(DirectiveType.RUN_TIME, Module.NAD, "chunkMemoryMB")`.
    pub const CHUNK_MEMORY_MB: DirectiveDef = DirectiveDef(64);
    /// Java `ADJUST_TILT_ANGLES = new DirectiveDef(DirectiveType.RUN_TIME, Module.PATCH_TRACKING, "adjustTiltAngles")`.
    pub const ADJUST_TILT_ANGLES: DirectiveDef = DirectiveDef(65);
    /// Java `CONTOUR_PIECES = new DirectiveDef(DirectiveType.RUN_TIME, Module.PATCH_TRACKING, "contourPieces")`.
    pub const CONTOUR_PIECES: DirectiveDef = DirectiveDef(66);
    /// Java `RAW_BOUNDARY_MODEL_FOR_PATCH_TRACKING = new DirectiveDef(DirectiveType.RUN_TIME, Module.PATCH_TRACKING, "rawBoundaryModel")`.
    pub const RAW_BOUNDARY_MODEL_FOR_PATCH_TRACKING: DirectiveDef = DirectiveDef(67);
    /// Java `BIN_BY_FACTOR_FOR_POSITIONING = new DirectiveDef( DirectiveType.RUN_TIME, Module.POSITIONING, RUN_TIME_BIN_BY_FACTOR_NAME)`.
    pub const BIN_BY_FACTOR_FOR_POSITIONING: DirectiveDef = DirectiveDef(68);
    /// Java `CENTER_ON_GOLD = new DirectiveDef(DirectiveType.RUN_TIME, Module.POSITIONING, "centerOnGold")`.
    pub const CENTER_ON_GOLD: DirectiveDef = DirectiveDef(69);
    /// Java `HAS_GOLD_BEADS = new DirectiveDef(DirectiveType.RUN_TIME, Module.POSITIONING, "hasGoldBeads")`.
    pub const HAS_GOLD_BEADS: DirectiveDef = DirectiveDef(70);
    /// Java `SAMPLE_TYPE = new DirectiveDef(DirectiveType.RUN_TIME, Module.POSITIONING, "sampleType")`.
    pub const SAMPLE_TYPE: DirectiveDef = DirectiveDef(71);
    /// Java `THICKNESS_FOR_POSITIONING = new DirectiveDef(DirectiveType.RUN_TIME, Module.POSITIONING, THICKNESS_NAME)`.
    pub const THICKNESS_FOR_POSITIONING: DirectiveDef = DirectiveDef(72);
    /// Java `WHOLE_TOMOGRAM = new DirectiveDef(DirectiveType.RUN_TIME, Module.POSITIONING, "wholeTomogram")`.
    pub const WHOLE_TOMOGRAM: DirectiveDef = DirectiveDef(73);
    /// Java `DO_TRIMVOL = new DirectiveDef(DirectiveType.RUN_TIME, Module.POSTPROCESS, "doTrimvol")`.
    pub const DO_TRIMVOL: DirectiveDef = DirectiveDef(74);
    /// Java `ARCHIVE_ORIGINAL = new DirectiveDef(DirectiveType.RUN_TIME, Module.PREPROCESSING, "archiveOriginal")`.
    pub const ARCHIVE_ORIGINAL: DirectiveDef = DirectiveDef(75);
    /// Java `DARK_EXCLUDE_RATIO = new DirectiveDef(DirectiveType.RUN_TIME, Module.PREPROCESSING, "darkExcludeRatio")`.
    pub const DARK_EXCLUDE_RATIO: DirectiveDef = DirectiveDef(76);
    /// Java `DARK_EXCLUDE_FRACTION = new DirectiveDef(DirectiveType.RUN_TIME, Module.PREPROCESSING, "darkExcludeFraction")`.
    pub const DARK_EXCLUDE_FRACTION: DirectiveDef = DirectiveDef(77);
    /// Java `END_EXCLUDE_CRITERION = new DirectiveDef(DirectiveType.RUN_TIME, Module.PREPROCESSING, "endExcludeCriterion")`.
    pub const END_EXCLUDE_CRITERION: DirectiveDef = DirectiveDef(78);
    /// Java `REMOVE_EXCLUDED_VIEWS = new DirectiveDef(DirectiveType.RUN_TIME, Module.PREPROCESSING, "removeExcludedViews")`.
    pub const REMOVE_EXCLUDED_VIEWS: DirectiveDef = DirectiveDef(79);
    /// Java `REMOVE_XRAYS = new DirectiveDef(DirectiveType.RUN_TIME, Module.PREPROCESSING, "removeXrays")`.
    pub const REMOVE_XRAYS: DirectiveDef = DirectiveDef(80);
    /// Java `NUMBER_OF_MARKERS = new DirectiveDef(DirectiveType.RUN_TIME, Module.RAPTOR, "numberOfMarkers")`.
    pub const NUMBER_OF_MARKERS: DirectiveDef = DirectiveDef(81);
    /// Java `USE_ALIGNED_STACK = new DirectiveDef(DirectiveType.RUN_TIME, Module.RAPTOR, "useAlignedStack")`.
    pub const USE_ALIGNED_STACK: DirectiveDef = DirectiveDef(82);
    /// Java `BINNED_THICKNESS = new DirectiveDef(DirectiveType.RUN_TIME, Module.RECONSTRUCTION, "binnedThickness")`.
    pub const BINNED_THICKNESS: DirectiveDef = DirectiveDef(83);
    /// Java `DO_BACKPROJ_ALSO = new DirectiveDef(DirectiveType.RUN_TIME, Module.RECONSTRUCTION, "doBackprojAlso")`.
    pub const DO_BACKPROJ_ALSO: DirectiveDef = DirectiveDef(84);
    /// Java `EXTRA_THICKNESS = new DirectiveDef(DirectiveType.RUN_TIME, Module.RECONSTRUCTION, "extraThickness")`.
    pub const EXTRA_THICKNESS: DirectiveDef = DirectiveDef(85);
    /// Java `FALLBACK_THICKNESS = new DirectiveDef(DirectiveType.RUN_TIME, Module.RECONSTRUCTION, "fallbackThickness")`.
    pub const FALLBACK_THICKNESS: DirectiveDef = DirectiveDef(86);
    /// Java `USE_SIRT = new DirectiveDef(DirectiveType.RUN_TIME, Module.RECONSTRUCTION, "useSirt")`.
    pub const USE_SIRT: DirectiveDef = DirectiveDef(87);
    /// Java `MIN_MEASUREMENT_RATIO = new DirectiveDef( DirectiveType.RUN_TIME, Module.RESTRICT_ALIGN, "minMeasurementRatio")`.
    pub const MIN_MEASUREMENT_RATIO: DirectiveDef = DirectiveDef(88);
    /// Java `ORDER_OF_RESTRICTIONS = new DirectiveDef( DirectiveType.RUN_TIME, Module.RESTRICT_ALIGN, "orderOfRestrictions")`.
    pub const ORDER_OF_RESTRICTIONS: DirectiveDef = DirectiveDef(89);
    /// Java `SKIP_BEAM_TILT_WITH_ONE_ROT = new DirectiveDef( DirectiveType.RUN_TIME, Module.RESTRICT_ALIGN, "skipBeamTiltWithOneRot")`.
    pub const SKIP_BEAM_TILT_WITH_ONE_ROT: DirectiveDef = DirectiveDef(90);
    /// Java `TARGET_MEASUREMENT_RATIO = new DirectiveDef( DirectiveType.RUN_TIME, Module.RESTRICT_ALIGN, "targetMeasurementRatio")`.
    pub const TARGET_MEASUREMENT_RATIO: DirectiveDef = DirectiveDef(91);
    /// Java `RAW_BOUNDARY_MODEL_FOR_SEED_FINDING = new DirectiveDef(DirectiveType.RUN_TIME, Module.SEED_FINDING, "rawBoundaryModel")`.
    pub const RAW_BOUNDARY_MODEL_FOR_SEED_FINDING: DirectiveDef = DirectiveDef(92);
    /// Java `ENABLE_STRETCHING = new DirectiveDef(DirectiveType.RUN_TIME, Module.TILT_ALIGNMENT, "enableStretching")`.
    pub const ENABLE_STRETCHING: DirectiveDef = DirectiveDef(93);
    /// Java `DO_A_OR_B_OF_DUAL_AXIS = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "doAorBofDualAxis")`.
    pub const DO_A_OR_B_OF_DUAL_AXIS: DirectiveDef = DirectiveDef(94);
    /// Java `DO_SIRT_IF_BOTH_FOR_TRIMVOL = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "doSIRTifBoth")`.
    pub const DO_SIRT_IF_BOTH_FOR_TRIMVOL: DirectiveDef = DirectiveDef(95);
    /// Java `FIND_SEC_ADD_THICKNESS = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "findSecAddThickness")`.
    pub const FIND_SEC_ADD_THICKNESS: DirectiveDef = DirectiveDef(96);
    /// Java `REORIENT = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "reorient")`.
    pub const REORIENT: DirectiveDef = DirectiveDef(97);
    /// Java `SIZE_IN_X = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "sizeInX")`.
    pub const SIZE_IN_X: DirectiveDef = DirectiveDef(98);
    /// Java `SIZE_IN_Y = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "sizeInY")`.
    pub const SIZE_IN_Y: DirectiveDef = DirectiveDef(99);
    /// Java `SCALE_FROM_X = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "scaleFromX")`.
    pub const SCALE_FROM_X: DirectiveDef = DirectiveDef(100);
    /// Java `SCALE_FROM_Y = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "scaleFromY")`.
    pub const SCALE_FROM_Y: DirectiveDef = DirectiveDef(101);
    /// Java `SCALE_FROM_Z = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "scaleFromZ")`.
    pub const SCALE_FROM_Z: DirectiveDef = DirectiveDef(102);
    /// Java `SCALE_TO_MEAN_SD = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, "scaleToMeanSD")`.
    pub const SCALE_TO_MEAN_SD: DirectiveDef = DirectiveDef(103);
    /// Java `THICKNESS_FOR_TRIMVOL = new DirectiveDef(DirectiveType.RUN_TIME, Module.TRIMVOL, THICKNESS_NAME)`.
    pub const THICKNESS_FOR_TRIMVOL: DirectiveDef = DirectiveDef(104);
    /// Java `REPLACE_STEP_NINE = new DirectiveDef(DirectiveType.RUN_TIME, Module.REPLACE_STEP, "9")`.
    pub const REPLACE_STEP_NINE: DirectiveDef = DirectiveDef(105);
    /// Java `REPLACE_STEP_THIRTEEN = new DirectiveDef(DirectiveType.RUN_TIME, Module.REPLACE_STEP, "13")`.
    pub const REPLACE_STEP_THIRTEEN: DirectiveDef = DirectiveDef(106);
    /// Java `RUN_AFTER_STEP_NINE = new DirectiveDef(DirectiveType.RUN_TIME, Module.RUN_AFTER_STEP, "9")`.
    pub const RUN_AFTER_STEP_NINE: DirectiveDef = DirectiveDef(107);
    /// Java `RUN_AFTER_STEP_THIRTEEN = new DirectiveDef(DirectiveType.RUN_TIME, Module.RUN_AFTER_STEP, "13")`.
    pub const RUN_AFTER_STEP_THIRTEEN: DirectiveDef = DirectiveDef(108);
    /// Java `LOCAL_ALIGNMENTS = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.ALIGN, Command.TILTALIGN, "LocalAlignments")`.
    pub const LOCAL_ALIGNMENTS: DirectiveDef = DirectiveDef(109);
    /// Java `SURFACES_TO_ANALYZE = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.ALIGN, Command.TILTALIGN, "SurfacesToAnalyze")`.
    pub const SURFACES_TO_ANALYZE: DirectiveDef = DirectiveDef(110);
    /// Java `TARGET_NUMBER_OF_BEADS = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.AUTOFIDSEED, Command.AUTOFIDSEED, "TargetNumberOfBeads")`.
    pub const TARGET_NUMBER_OF_BEADS: DirectiveDef = DirectiveDef(111);
    /// Java `MIN_GUESS_NUM_BEADS = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.AUTOFIDSEED, Command.AUTOFIDSEED, "MinGuessNumBeads")`.
    pub const MIN_GUESS_NUM_BEADS: DirectiveDef = DirectiveDef(112);
    /// Java `TARGET_DENSITY_OF_BEADS = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.AUTOFIDSEED, Command.AUTOFIDSEED, "TargetDensityOfBeads")`.
    pub const TARGET_DENSITY_OF_BEADS: DirectiveDef = DirectiveDef(113);
    /// Java `TWO_SURFACES = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.AUTOFIDSEED, Command.AUTOFIDSEED, "TwoSurfaces")`.
    pub const TWO_SURFACES: DirectiveDef = DirectiveDef(114);
    /// Java `MODEL_FILE = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.ERASER, Command.CCDERASER, "ModelFile")`.
    pub const MODEL_FILE: DirectiveDef = DirectiveDef(115);
    /// Java `SLAB_THICKNESS_IN_NM = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.CTF_3D_SETUP, Command.CTF_3D_SETUP, "SlabThicknessInNm")`.
    pub const SLAB_THICKNESS_IN_NM: DirectiveDef = DirectiveDef(116);
    /// Java `SCAN_DEFOCUS_RANGE = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.CTF_PLOTTER, Command.CTF_PLOTTER, "ScanDefocusRange")`.
    pub const SCAN_DEFOCUS_RANGE: DirectiveDef = DirectiveDef(117);
    /// Java `TUNE_FITTING_AND_SAMPLING = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.CTF_PLOTTER, Command.CTF_PLOTTER, "TuneFittingAndSampling")`.
    pub const TUNE_FITTING_AND_SAMPLING: DirectiveDef = DirectiveDef(118);
    /// Java `EXPAND_CIRCLE_ITERATIONS = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.GOLD_ERASER, Command.CCDERASER, "ExpandCircleIterations")`.
    pub const EXPAND_CIRCLE_ITERATIONS: DirectiveDef = DirectiveDef(119);
    /// Java `BIN_BY_FACTOR_FOR_PREBLEND = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.PREBLEND, Command.BLENDMONT, COM_PARAM_BIN_BY_FACTOR_NAME)`.
    pub const BIN_BY_FACTOR_FOR_PREBLEND: DirectiveDef = DirectiveDef(120);
    /// Java `BIN_BY_FACTOR_FOR_PRENEWST = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.PRENEWST, Command.NEWSTACK, COM_PARAM_BIN_BY_FACTOR_NAME)`.
    pub const BIN_BY_FACTOR_FOR_PRENEWST: DirectiveDef = DirectiveDef(121);
    /// Java `LEAVE_ITERATIONS = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.SIRTSETUP, Command.SIRTSETUP, "LeaveIterations")`.
    pub const LEAVE_ITERATIONS: DirectiveDef = DirectiveDef(122);
    /// Java `SCALE_TO_INTEGER = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.SIRTSETUP, Command.SIRTSETUP, "ScaleToInteger")`.
    pub const SCALE_TO_INTEGER: DirectiveDef = DirectiveDef(123);
    /// Java `FAKE_SIRT_ITERATIONS = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.TILT, Command.TILT, "FakeSIRTiterations")`.
    pub const FAKE_SIRT_ITERATIONS: DirectiveDef = DirectiveDef(124);
    /// Java `THICKNESS_FOR_TILT = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.TILT, Command.TILT, "THICKNESS")`.
    pub const THICKNESS_FOR_TILT: DirectiveDef = DirectiveDef(125);
    /// Java `LOCAL_AREA_TARGET_SIZE = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.TRACK, Command.BEADTRACK, "LocalAreaTargetSize")`.
    pub const LOCAL_AREA_TARGET_SIZE: DirectiveDef = DirectiveDef(126);
    /// Java `LENGTH_OF_PIECES = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.XCORR_PT, Command.IMODCHOPCONTS, "LengthOfPieces")`.
    pub const LENGTH_OF_PIECES: DirectiveDef = DirectiveDef(127);
    /// Java `SEARCH_MAG_CHANGES = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.XCORR, Command.TILTXCORR, "SearchMagChanges")`.
    pub const SEARCH_MAG_CHANGES: DirectiveDef = DirectiveDef(128);
    /// Java `NUMBER_OF_PATCHES_X_AND_Y = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.XCORR_PT, Command.TILTXCORR, "NumberOfPatchesXandY")`.
    pub const NUMBER_OF_PATCHES_X_AND_Y: DirectiveDef = DirectiveDef(129);
    /// Java `OVERLAP_OF_PATCHES_X_AND_Y = new DirectiveDef(DirectiveType.COM_PARAM, Comfile.XCORR_PT, Command.TILTXCORR, "OverlapOfPatchesXandY")`.
    pub const OVERLAP_OF_PATCHES_X_AND_Y: DirectiveDef = DirectiveDef(130);
    /// Java `SIZE_OF_PATCHES_X_AND_Y = new DirectiveDef( DirectiveType.COM_PARAM, Comfile.XCORR_PT, Command.TILTXCORR, "SizeOfPatchesXandY")`.
    pub const SIZE_OF_PATCHES_X_AND_Y: DirectiveDef = DirectiveDef(131);
}

impl DirectiveDef {
    /// Java private general constructor
    /// `DirectiveDef(DirectiveType, String, String, String, String, boolean, boolean,
    /// boolean)`.  Saves to MAP if the instance doesn't have a duplicate key.  Saves
    /// directives with a separate B directive twice.
    fn construct(
        registry: &mut Registry,
        directive_type: DirectiveType,
        module: Option<&str>,
        comfile: Option<&str>,
        command: Option<&str>,
        name: &str,
        a_only_directive: bool,
        b_only_directive: bool,
        preconstructed: bool,
    ) -> DirectiveDef {
        let fields = DirectiveDefFields {
            directive_type,
            module: module.map(|module| module.to_string()),
            comfile: comfile.map(|comfile| comfile.to_string()),
            command: command.map(|command| command.to_string()),
            name: name.to_string(),
            preconstructed,
            a_only_directive,
            b_only_directive,
            bool: false,
            template_a: false,
            template_b: false,
            batch_a: false,
            batch_b: false,
            directive_descr_loaded: false,
            description: None,
        };
        let index = registry.defs.len();
        registry.defs.push(fields);
        let descr_key = DirectiveDef::get_standard_key_static(
            directive_type,
            module,
            comfile,
            command,
            Some(name),
        );
        if let Some(descr_key) = descr_key {
            if !registry.map.contains_key(&descr_key) {
                registry.map.insert(descr_key, index);
            }
        }
        DirectiveDef(index)
    }

    /// Java private `DirectiveDef(DirectiveType, String)`.  Constructor for
    /// precontructed setupset.copyarg and setupset.  Keep private.
    fn new_name(
        registry: &mut Registry,
        directive_type: DirectiveType,
        name: &str,
    ) -> DirectiveDef {
        DirectiveDef::construct(
            registry,
            directive_type,
            None,
            None,
            None,
            name,
            false,
            false,
            true,
        )
    }

    /// Java private `DirectiveDef(DirectiveType, String, boolean)`.  Constructor for
    /// precontructed setupset.copyarg.  Keep private.
    fn new_copy_arg_a(
        registry: &mut Registry,
        directive_type: DirectiveType,
        name: &str,
        a_only_directive: bool,
    ) -> DirectiveDef {
        DirectiveDef::construct(
            registry,
            directive_type,
            None,
            None,
            None,
            name,
            a_only_directive,
            false,
            true,
        )
    }

    /// Java private static `getBInstance(DirectiveType, String)`.  getInstance function
    /// for precontructed setupset.copyarg.  Keep private.
    fn get_b_instance(
        registry: &mut Registry,
        directive_type: DirectiveType,
        name: &str,
    ) -> DirectiveDef {
        DirectiveDef::construct(
            registry,
            directive_type,
            None,
            None,
            None,
            name,
            false,
            true,
            true,
        )
    }

    /// Java private `DirectiveDef(DirectiveType, Module, String)`.  Constructor for
    /// preconstructed runtime.  Keep private.
    fn new_run_time(
        registry: &mut Registry,
        directive_type: DirectiveType,
        module: Option<&Module>,
        name: &str,
    ) -> DirectiveDef {
        let module = module.map(|module| module.to_string());
        DirectiveDef::construct(
            registry,
            directive_type,
            module.as_deref(),
            None,
            None,
            name,
            false,
            false,
            true,
        )
    }

    /// Java private `DirectiveDef(DirectiveType, Comfile, Command, String)`.
    /// Constructor for preconstructed comparam.  Keep private.
    fn new_com_param(
        registry: &mut Registry,
        directive_type: DirectiveType,
        comfile: Option<&Comfile>,
        command: Option<&Command>,
        name: &str,
    ) -> DirectiveDef {
        let comfile = comfile.map(|comfile| comfile.to_string());
        let command = command.map(|command| command.to_string());
        DirectiveDef::construct(
            registry,
            directive_type,
            None,
            comfile.as_deref(),
            command.as_deref(),
            name,
            false,
            false,
            true,
        )
    }

    /// The Java object's fields, read under the registry lock.
    fn fields(self) -> DirectiveDefFields {
        REGISTRY.lock().unwrap().defs[self.0].clone()
    }

    /// Java package-private `getAxisIDInstance(AxisID)`.  A null pairAxisID is NOT
    /// treated as axis A.  When it is null the same directiveDef will be returned.
    ///
    /// The source returns `getInstance(switchStandardKey(pairAxisID))`, which is null
    /// only when the paired key cannot be built; callers then dereference it.  Fixed in
    /// translation: this instance is returned in that case.
    pub(crate) fn get_axis_id_instance(self, pair_axis_id: Option<AxisID>) -> DirectiveDef {
        let fields = self.fields();
        if pair_axis_id.is_none()
            || fields.directive_type != DirectiveType::COPY_ARG
            || !fields.a_only_directive && !fields.b_only_directive
            || (pair_axis_id != Some(AxisID::Second) && fields.a_only_directive)
            || (pair_axis_id == Some(AxisID::Second) && fields.b_only_directive)
        {
            return self;
        }
        match DirectiveDef::get_instance(self.switch_standard_key(pair_axis_id).as_deref()) {
            None => self,
            Some(directive_def) => directive_def,
        }
    }

    /// Java static `getInstance(String)`.
    pub fn get_instance(directive: Option<&str>) -> Option<DirectiveDef> {
        DirectiveDef::get_instance_private(directive, None)
    }

    /// Java static `getInstanceFromCsv(String, DirectiveDef)`.
    pub fn get_instance_from_csv(
        directive: Option<&str>,
        prev_directive_def_from_csv: Option<DirectiveDef>,
    ) -> Option<DirectiveDef> {
        DirectiveDef::get_instance_private(directive, prev_directive_def_from_csv)
    }

    /// Java private static `getInstance(String, DirectiveDef)`.  Gets or builds a
    /// directiveDef from any directive string.  Works with directives from an autodoc
    /// file, or from directives.csv.  Saves the instance in MAP under the standard key if
    /// it is new.  Returns an existing directiveDef if it already exists.  The directive
    /// type must be known.  The module, comfile, and command are treated as strings and
    /// don't have to be recognized.  Assumes that a comfile ending in a or b has an axis
    /// letter.
    ///
    /// Can create directiveDefs that are not in the directives.csv file.
    ///
    /// Corrects aOnlyDirective and bOnlyDirective when both the A and B members of a
    /// CopyArg pair are present.  PrevDirectiveDefFromCsv makes the correction process
    /// slightly more reliable if the directives are coming from the directives.csv file.
    ///
    /// Accepted forms:
    /// setupset.copyarg.name, setupset.copyarg.bname, setupset.name,
    /// runtime.module.any.name, runtime.module.name, runtime.module..name,
    /// runtime.module.a.name, runtime.module.b.name, comparam.comfile.command.name,
    /// comparam.comfilea.command.name, comparam.comfileb.command.name
    fn get_instance_private(
        directive: Option<&str>,
        prev_directive_def_from_csv: Option<DirectiveDef>,
    ) -> Option<DirectiveDef> {
        let directive = directive?;
        let mut guard = REGISTRY.lock().unwrap();
        let registry = &mut *guard;
        // Construct or find directiveDef.
        // First see if the directive string is a key to a saved directive
        let mut directive_def = registry.map.get(directive).copied();
        if directive_def.is_none() {
            // So far can't find it, pull the fields out of the directive string..
            let array = split_directive(directive);
            let directive_type = DirectiveType::get_instance_from_array(Some(array.as_slice()));
            let mut module: Option<String> = None;
            let mut comfile: Option<String> = None;
            let mut command: Option<String> = None;
            let mut name: Option<String> = None;
            let mut index: usize;
            if directive_type == Some(DirectiveType::COPY_ARG) {
                index = 2;
                if array.len() > index {
                    name = Some(array[index].clone());
                }
            } else if directive_type == Some(DirectiveType::SETUP_SET) {
                // name
                index = 1;
                if array.len() > index {
                    name = Some(array[index].clone());
                }
            } else if directive_type == Some(DirectiveType::RUN_TIME) {
                // module
                index = 1;
                if array.len() > index {
                    module = Some(array[index].clone());
                }
                // Check for optional axisID
                index += 1;
                if array.len() > index + 1
                    && (array[index].is_empty()
                        || array[index] == RUN_TIME_ANY_AXIS_TAG
                        || array[index] == RUN_TIME_A_AXIS_TAG
                        || array[index] == B_AXIS_TAG)
                {
                    // found axisID - will be replaced with "any" in the key
                    index += 1;
                }
                // name
                if array.len() > index {
                    name = Some(array[index].clone());
                }
            } else if directive_type == Some(DirectiveType::COM_PARAM) {
                // comfile
                index = 1;
                if array.len() > index {
                    let mut value = array[index].clone();
                    // check for axisID - the a or b addition to comfile names
                    if value.chars().count() > 1
                        && (value.ends_with(&AxisID::First.get_extension())
                            || value.ends_with(&AxisID::Second.get_extension()))
                    {
                        // found axisID - stripping from comfile
                        let length = value.chars().count();
                        value = value.chars().take(length - 1).collect();
                    }
                    comfile = Some(value);
                }
                // command
                index += 1;
                if array.len() > index {
                    command = Some(array[index].clone());
                }
                // name
                index += 1;
                if array.len() > index {
                    name = Some(array[index].clone());
                }
            }
            let name = match name {
                Some(name) if !name.is_empty() => name,
                _ => return None,
            };
            // `name` is only set when the type was recognized.
            let directive_type = directive_type.unwrap();
            // Continue constructing or finding directiveDef
            //
            // With the axis information turned off, the function getSimpleStandardKey
            // will get the default key based on fields from the directives string.
            let key = DirectiveDef::get_standard_key_static(
                directive_type,
                module.as_deref(),
                comfile.as_deref(),
                command.as_deref(),
                Some(&name),
            );
            directive_def = key.and_then(|key| registry.map.get(&key).copied());
            if directive_def.is_none() {
                directive_def = Some(
                    DirectiveDef::construct(
                        registry,
                        directive_type,
                        module.as_deref(),
                        comfile.as_deref(),
                        command.as_deref(),
                        &name,
                        false,
                        false,
                        false,
                    )
                    .0,
                );
            }
        }
        let directive_def = directive_def.unwrap();
        let def = registry.defs[directive_def].clone();
        // The preconstructed ones don't have to be corrected. Only CopyArg has paired
        // directives.
        if def.preconstructed || def.directive_type != DirectiveType::COPY_ARG {
            return Some(DirectiveDef(directive_def));
        }
        // If there are paired A and B copyArg directives available then correct them.
        let mut a_directive_def: Option<usize> = None;
        let mut b_directive_def: Option<usize> = None;
        let prev_matches = match prev_directive_def_from_csv {
            None => false,
            Some(prev) => def.name == format!("{}{}", B_AXIS_TAG, registry.defs[prev.0].name),
        };
        if prev_matches {
            // In the directives.csv file, A axis copyArg directives always comes right
            // before the corresponding B directive.
            a_directive_def = prev_directive_def_from_csv.map(|prev| prev.0);
            b_directive_def = Some(directive_def);
        } else {
            // Look for a matching B directive
            b_directive_def = DirectiveDef::get_standard_key_static(
                def.directive_type,
                def.module.as_deref(),
                def.comfile.as_deref(),
                def.command.as_deref(),
                Some(&format!("{}{}", B_AXIS_TAG, def.name)),
            )
            .and_then(|key| registry.map.get(&key).copied());
            if b_directive_def.is_some() {
                a_directive_def = Some(directive_def);
            } else if def.name.starts_with(B_AXIS_TAG) {
                // Look for a matching A directive
                a_directive_def = DirectiveDef::get_standard_key_static(
                    def.directive_type,
                    def.module.as_deref(),
                    def.comfile.as_deref(),
                    def.command.as_deref(),
                    Some(&substring_1(&def.name)),
                )
                .and_then(|key| registry.map.get(&key).copied());
                if a_directive_def.is_some() {
                    b_directive_def = Some(directive_def);
                }
            }
        }
        // If a pair was found, correct them.
        if let (Some(a), Some(b)) = (a_directive_def, b_directive_def) {
            if !registry.defs[a].preconstructed {
                registry.defs[a].a_only_directive = true;
            }
            if !registry.defs[b].preconstructed {
                registry.defs[b].b_only_directive = true;
            }
        }
        Some(DirectiveDef(directive_def))
    }

    /// Java package-private `isAOnlyDirective`.
    pub(crate) fn is_a_only_directive(self) -> bool {
        self.fields().a_only_directive
    }

    /// Java package-private `isBOnlyDirective`.
    pub(crate) fn is_b_only_directive(self) -> bool {
        self.fields().b_only_directive
    }

    /// Java `switchStandardKey(AxisID)`.  Returns the standard key to this directive,
    /// unless this directive is part of a pair.  In that case it returns the key of the
    /// dirctive that cooresponds to pairAxisID if pairAxisID is not null.
    pub fn switch_standard_key(self, pair_axis_id: Option<AxisID>) -> Option<String> {
        let fields = self.fields();
        DirectiveDef::get_standard_key_static(
            fields.directive_type,
            fields.module.as_deref(),
            fields.comfile.as_deref(),
            fields.command.as_deref(),
            Some(&self.get_pair_name(pair_axis_id)),
        )
    }

    /// Java private `getPairName(AxisID)`.  If this is a paired dirctive (copyArg and
    /// representing a directive that has an A axis and B axis form), return the name
    /// corresponding to pairAxisID.  A null pairAxisID returns the name of this
    /// directiveDef.
    fn get_pair_name(self, pair_axis_id: Option<AxisID>) -> String {
        let fields = self.fields();
        if pair_axis_id.is_none()
            || fields.directive_type != DirectiveType::COPY_ARG
            || (!fields.a_only_directive && !fields.b_only_directive)
            || (fields.a_only_directive && pair_axis_id != Some(AxisID::Second))
            || (fields.b_only_directive && pair_axis_id == Some(AxisID::Second))
        {
            return fields.name;
        }
        if pair_axis_id != Some(AxisID::Second) {
            return substring_1(&fields.name);
        }
        format!("{}{}", B_AXIS_TAG, fields.name)
    }

    /// Java private static `getStandardKey(DirectiveType, String, String, String,
    /// String)`.  Returns the key to the MAP in this class and and the one in
    /// DirectiveDescrFile.  This class's hashCode also uses this key.
    ///
    /// Descr key formats: setupset.copyarg.name, setupset.copyarg.bname, setupset.name,
    /// runtime.module.any.name, comparam.comfile.command.name
    fn get_standard_key_static(
        directive_type: DirectiveType,
        module: Option<&str>,
        comfile: Option<&str>,
        command: Option<&str>,
        name: Option<&str>,
    ) -> Option<String> {
        let name = match name {
            Some(name) if !name.is_empty() => name,
            _ => return None,
        };
        if directive_type == DirectiveType::COPY_ARG {
            return Some(format!(
                "{}{}{}{}{}",
                DirectiveType::SETUP_SET,
                SEPARATOR_CHAR,
                directive_type,
                SEPARATOR_CHAR,
                name
            ));
        }
        if directive_type == DirectiveType::SETUP_SET {
            return Some(format!("{}{}{}", directive_type, SEPARATOR_CHAR, name));
        }
        if directive_type == DirectiveType::RUN_TIME {
            let module = match module {
                Some(module) if !module.is_empty() => module,
                _ => return None,
            };
            return Some(format!(
                "{}{}{}{}{}{}{}",
                directive_type,
                SEPARATOR_CHAR,
                module,
                SEPARATOR_CHAR,
                RUN_TIME_ANY_AXIS_TAG,
                SEPARATOR_CHAR,
                name
            ));
        }
        if directive_type == DirectiveType::COM_PARAM {
            let (comfile, command) = match (comfile, command) {
                (Some(comfile), Some(command)) if !comfile.is_empty() && !command.is_empty() => {
                    (comfile, command)
                }
                _ => return None,
            };
            return Some(format!(
                "{}{}{}{}{}{}{}",
                directive_type,
                SEPARATOR_CHAR,
                comfile,
                SEPARATOR_CHAR,
                command,
                SEPARATOR_CHAR,
                name
            ));
        }
        None
    }

    /// Java `getStandardKey()`.  Returns a key to the MAP.  Always returns the A version
    /// of a copyarg directive key.
    pub fn get_standard_key(self) -> Option<String> {
        let fields = self.fields();
        DirectiveDef::get_standard_key_static(
            fields.directive_type,
            fields.module.as_deref(),
            fields.comfile.as_deref(),
            fields.command.as_deref(),
            Some(&fields.name),
        )
    }

    /// Java `getCopyArgAxisID(String)`.  Returns the Axis of copyArgDirective, if this
    /// instance is part of a pair of CopyArg directives.  Returns null instead of A
    /// axis.  Also return null in case of failure.
    ///
    /// Fixed in translation: DirectiveDef.java:767 calls
    /// `DirectiveType.getInstance(copyArgDirective)` before the null check further down,
    /// a NullPointerException for a null directive.  A null directive returns null.
    pub fn get_copy_arg_axis_id(self, copy_arg_directive: Option<&str>) -> Option<AxisID> {
        let fields = self.fields();
        let copy_arg_directive = copy_arg_directive?;
        if fields.directive_type != DirectiveType::COPY_ARG
            || DirectiveType::get_instance(copy_arg_directive) != Some(fields.directive_type)
        {
            return None;
        }
        // If this directive has a separate b directive, or is a b directive, use the
        // string parameter to find out the axisID to use.
        if fields.directive_type == DirectiveType::COPY_ARG
            && (fields.a_only_directive || fields.b_only_directive)
        {
            let array = split_directive(copy_arg_directive);
            let name_index = 2;
            if array.len() > name_index
                && array[name_index].chars().count() > B_AXIS_TAG.chars().count()
                && array[name_index].starts_with(B_AXIS_TAG)
            {
                return Some(AxisID::Second);
            }
        }
        None
    }

    /// Java package-private `getDirectiveType`.
    pub(crate) fn get_directive_type(self) -> DirectiveType {
        self.fields().directive_type
    }

    /// Java package-private `loadDirectiveDescr`.  Load information about directive from
    /// the directive.csv file.
    ///
    /// `DirectiveDescrFile` builds `DirectiveDef`s itself while it loads, so it is
    /// consulted with the registry unlocked and the results are stored afterwards.
    pub(crate) fn load_directive_descr(self) -> bool {
        let fields = self.fields();
        if fields.directive_descr_loaded {
            return true;
        }
        let mut bool = fields.bool;
        let mut template_a = fields.template_a;
        let mut template_b = fields.template_b;
        let mut batch_a = fields.batch_a;
        let mut batch_b = fields.batch_b;
        let mut element = directive_descr_file::INSTANCE
            .get(self.switch_standard_key(Some(AxisID::First)).as_deref());
        if let Some(element) = &element {
            let r#type = element.get_value_type();
            if r#type == Some(DirectiveValueType::Boolean) {
                bool = true;
            }
            template_a = element.is_template();
            batch_a = element.is_batch();
        }
        if fields.directive_type == DirectiveType::COPY_ARG {
            element = directive_descr_file::INSTANCE
                .get(self.switch_standard_key(Some(AxisID::Second)).as_deref());
            if let Some(element) = &element {
                template_b = element.is_template();
                batch_b = element.is_batch();
            }
        }
        if !fields.a_only_directive && !fields.b_only_directive {
            template_b = template_a;
            batch_b = batch_a;
        }
        let description;
        let loaded;
        match &element {
            Some(element) => {
                description = element.get_description();
                loaded = true;
            }
            None => {
                eprintln!("\nError:  Unknown directive: {}\n", self);
                description = fields.description.clone();
                loaded = false;
            }
        }
        {
            let mut registry = REGISTRY.lock().unwrap();
            let def = &mut registry.defs[self.0];
            def.bool = bool;
            def.template_a = template_a;
            def.template_b = template_b;
            def.batch_a = batch_a;
            def.batch_b = batch_b;
            def.description = description;
            def.directive_descr_loaded = loaded;
        }
        loaded
    }

    /// Java package-private `hasSecondaryMatch(AxisID)`.
    pub(crate) fn has_secondary_match(self, _axis_id: Option<AxisID>) -> bool {
        if self.fields().directive_type == DirectiveType::COPY_ARG {
            return false;
        }
        true
    }

    /// Java `isCopyArg`.
    pub fn is_copy_arg(self) -> bool {
        self.fields().directive_type == DirectiveType::COPY_ARG
    }

    /// Java `isComparam`.
    pub fn is_comparam(self) -> bool {
        self.fields().directive_type == DirectiveType::COM_PARAM
    }

    /// Java `isRuntime`.
    pub fn is_runtime(self) -> bool {
        self.fields().directive_type == DirectiveType::RUN_TIME
    }

    /// Java `isBoolean`.
    pub fn is_boolean(self) -> bool {
        self.load_directive_descr();
        self.fields().bool
    }

    /// Java static `getBooleanValue(boolean)`.
    pub fn get_boolean_value_bool(bool: bool) -> String {
        if bool {
            return TRUE_VALUE.to_string();
        }
        FALSE_VALUE.to_string()
    }

    /// Java static `convertToBoolean(String)`.  Returns true if value equals
    /// TRUE_VALUE.  Otherwise returns false.
    pub fn convert_to_boolean(value: Option<&str>) -> bool {
        match value {
            None => false,
            Some(value) => value == TRUE_VALUE,
        }
    }

    /// Java static `isValidBooleanValue(String)`.
    pub fn is_valid_boolean_value(value: Option<&str>) -> bool {
        let value = match value {
            None => return false,
            Some(value) => java_lang_string_trim(value),
        };
        value == TRUE_VALUE || value == FALSE_VALUE
    }

    /// Java static `getBooleanValue(String)`.
    pub fn get_boolean_value_string(value: Option<&str>) -> String {
        if DirectiveDef::is_valid_boolean_value(value) {
            return value.unwrap().to_string();
        }
        if value.is_none() || value == Some("") {
            return FALSE_VALUE.to_string();
        }
        TRUE_VALUE.to_string()
    }

    /// Java `getTooltip`.  Gets a tooltip from the description in the directive
    /// description file.  No tooltip is loaded for comparam directives.
    pub fn get_tooltip(self) -> String {
        self.load_directive_descr();
        format!(
            "{} ({})",
            self.fields().description.as_deref().unwrap_or("null"),
            self
        )
    }

    /// Java `getUnformattedTooltip`.
    pub fn get_unformatted_tooltip(self) -> Option<String> {
        self.load_directive_descr();
        self.fields().description
    }

    /// Java package-private `isTemplate(AxisID)`.
    pub(crate) fn is_template(self, axis_id: Option<AxisID>) -> bool {
        self.load_directive_descr();
        let fields = self.fields();
        if axis_id == Some(AxisID::Second) {
            return fields.template_b;
        }
        fields.template_a
    }

    /// Java package-private `isBatch(AxisID)`.
    pub(crate) fn is_batch(self, axis_id: Option<AxisID>) -> bool {
        self.load_directive_descr();
        let fields = self.fields();
        if axis_id == Some(AxisID::Second) {
            return fields.batch_b;
        }
        fields.batch_a
    }

    /// Java package-private `getName(Match, AxisID)`.  Get name according to which
    /// match is being done.  How well an attribute matches an axis.  Primary overrides
    /// secondary match even in a lower precedence directive file.
    ///
    /// Copyarg: No Secondary match for CopyArg.  Paired directives are only used on
    /// their own axis.  Unpaired directives are a First level match for both axes.  A
    /// null pairAxisID causes the name of this directive to be returned.
    pub(crate) fn get_name_for_match(
        self,
        r#match: Match,
        pair_axis_id: Option<AxisID>,
    ) -> Option<String> {
        let fields = self.fields();
        if fields.directive_type != DirectiveType::COPY_ARG {
            return Some(fields.name);
        }
        // Get the name for copyArg directives.
        if r#match == Match::Secondary {
            return None;
        }
        if pair_axis_id.is_none() || !fields.a_only_directive && !fields.b_only_directive {
            return Some(fields.name);
        }
        // Get the name for paired copyArg directives.
        if (fields.a_only_directive && pair_axis_id != Some(AxisID::Second))
            || (fields.b_only_directive && pair_axis_id == Some(AxisID::Second))
        {
            return Some(fields.name);
        }
        None
    }

    /// Java `getModule`.  Module name for runtime directives.
    pub fn get_module(self) -> Option<String> {
        self.fields().module
    }

    /// Java package-private `getAxis(Match, AxisID)`.  Returns the axis tag.
    ///
    /// Runtime and Comparam:
    /// axis = null:  Primary match: any.  Secondary match: a, does not match b
    /// axis = only: Primary match: any.  Secondary match: a, does not match b
    /// axis = first: Primary match: a.  Secondary match: any, does not match b
    /// axis = second: Primary match: b.  Secondary match: any, does not match a
    pub(crate) fn get_axis(self, r#match: Match, axis_id: Option<AxisID>) -> Option<String> {
        if self.fields().directive_type != DirectiveType::RUN_TIME {
            return None;
        }
        if r#match == Match::Primary {
            if axis_id == Some(AxisID::First) {
                return Some(RUN_TIME_A_AXIS_TAG.to_string());
            }
            if axis_id == Some(AxisID::Second) {
                return Some(B_AXIS_TAG.to_string());
            }
        } else if r#match == Match::Secondary {
            if axis_id.is_none() || axis_id == Some(AxisID::Only) {
                return Some(RUN_TIME_A_AXIS_TAG.to_string());
            }
        }
        Some(RUN_TIME_ANY_AXIS_TAG.to_string())
    }

    /// Java package-private `getComfile(Match, AxisID)`.  Same functionality as getAxis,
    /// but for comparam instead of runtime.
    pub(crate) fn get_comfile(self, r#match: Match, axis_id: Option<AxisID>) -> Option<String> {
        let fields = self.fields();
        if fields.directive_type != DirectiveType::COM_PARAM {
            return None;
        }
        let comfile = fields.comfile.as_deref().unwrap_or("null");
        if r#match == Match::Primary {
            if axis_id == Some(AxisID::First) {
                return Some(format!("{}{}", comfile, AxisID::First.get_extension()));
            }
            if axis_id == Some(AxisID::Second) {
                return Some(format!("{}{}", comfile, AxisID::Second.get_extension()));
            }
        } else if r#match == Match::Secondary {
            if axis_id.is_none() || axis_id == Some(AxisID::Only) {
                return Some(format!("{}{}", comfile, AxisID::First.get_extension()));
            }
        }
        fields.comfile
    }

    /// Java `getCommand`.  The command name element for comparam directives.
    pub fn get_command(self) -> Option<String> {
        self.fields().command
    }

    /// Java `getName()`.
    pub fn get_name(self) -> String {
        self.fields().name
    }

    /// Java `getName(AxisID)`.  Get the correct name for the axis without having to use
    /// the other axis DirectiveDef instance.  A null pairAxisID always returns the name
    /// from this directiveDef.
    pub fn get_name_for_axis(self, pair_axis_id: Option<AxisID>) -> String {
        let fields = self.fields();
        if pair_axis_id.is_none() || fields.directive_type != DirectiveType::COPY_ARG {
            return fields.name;
        }
        if fields.a_only_directive && pair_axis_id == Some(AxisID::Second) {
            return format!("{}{}", B_AXIS_TAG, fields.name);
        }
        if fields.b_only_directive && pair_axis_id != Some(AxisID::Second) {
            return substring_1(&fields.name);
        }
        fields.name
    }

    /// Java `getDirective`.  Returns the full directive string with the axis tag.
    pub fn get_directive(self) -> String {
        format!(
            "{}{}{}",
            self.get_prefix(),
            self.get_axis_tag(),
            self.get_postfix()
        )
    }

    /// Java `hashCode`: `String.hashCode` of the standard key, or of `toString()` when
    /// there is no key.
    pub fn hash_code(self) -> i32 {
        let key = self.get_standard_key().unwrap_or_else(|| self.to_string());
        let mut hash: i32 = 0;
        for unit in key.encode_utf16() {
            hash = hash.wrapping_mul(31).wrapping_add(unit as i32);
        }
        hash
    }

    /// Java `equals(Object)`.
    ///
    /// Fixed in translation: DirectiveDef.java:1087 compares `hashCode()` values, so two
    /// directives whose standard keys collide under `String.hashCode` compare equal.
    /// The keys themselves (or `toString()` where there is no key) are compared here,
    /// which is what the hash stands for.
    pub fn equals(self, object: Option<DirectiveDef>) -> bool {
        let object = match object {
            None => return false,
            Some(object) => object,
        };
        if self.0 == object.0 {
            return true;
        }
        self.identity_key() == object.identity_key()
    }

    /// The string `hashCode` is computed from.
    fn identity_key(self) -> String {
        self.get_standard_key().unwrap_or_else(|| self.to_string())
    }

    /// Java package-private `equalsName(String, AxisID)`.
    pub(crate) fn equals_name(self, name: Option<&str>, pair_axis_id: Option<AxisID>) -> bool {
        Some(self.get_name_for_axis(pair_axis_id).as_str()) == name
    }

    /// Java private `getAxisTag`.  Gets the default axis tag.
    fn get_axis_tag(self) -> String {
        if self.fields().directive_type == DirectiveType::RUN_TIME {
            return RUN_TIME_ANY_AXIS_TAG.to_string();
        }
        String::new()
    }

    /// Java private `getPrefix`.  Creates the directive string up to the axis tag, or
    /// the whole directive string for directives with no axis tag.  The source's final
    /// branch, for a type that is none of the four, cannot be reached with a
    /// `DirectiveType` value.
    fn get_prefix(self) -> String {
        let fields = self.fields();
        match fields.directive_type {
            DirectiveType::CopyArg => format!(
                "{}{}{}",
                fields.directive_type.get_key(),
                SEPARATOR_CHAR,
                fields.name
            ),
            DirectiveType::SetupSet => format!(
                "{}{}{}",
                fields.directive_type.get_key(),
                SEPARATOR_CHAR,
                fields.name
            ),
            DirectiveType::RunTime => format!(
                "{}{}{}{}",
                fields.directive_type.get_key(),
                SEPARATOR_CHAR,
                fields.module.as_deref().unwrap_or("null"),
                SEPARATOR_CHAR
            ),
            DirectiveType::ComParam => format!(
                "{}{}{}",
                fields.directive_type.get_key(),
                SEPARATOR_CHAR,
                fields.comfile.as_deref().unwrap_or("null")
            ),
        }
    }

    /// Java private `getPostfix`.  Creates the directive string after the axis tag.
    /// Returns nothing for directives with no axis tag.
    fn get_postfix(self) -> String {
        let fields = self.fields();
        if fields.directive_type == DirectiveType::COPY_ARG {
            return String::new();
        }
        if fields.directive_type == DirectiveType::SETUP_SET {
            return String::new();
        }
        if fields.directive_type == DirectiveType::RUN_TIME {
            return format!("{}{}", SEPARATOR_CHAR, fields.name);
        }
        if fields.directive_type == DirectiveType::COM_PARAM {
            return format!(
                "{}{}{}{}",
                SEPARATOR_CHAR,
                fields.command.as_deref().unwrap_or("null"),
                SEPARATOR_CHAR,
                fields.name
            );
        }
        String::new()
    }
}

/// Java `toString`.  Returns the directive string with no axis tag.  Each instance
/// should return a unique string.
impl std::fmt::Display for DirectiveDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.get_prefix(), self.get_postfix())
    }
}

/// Java `equals(Object)`; see [`DirectiveDef::equals`].
impl PartialEq for DirectiveDef {
    fn eq(&self, other: &DirectiveDef) -> bool {
        self.equals(Some(*other))
    }
}

impl Eq for DirectiveDef {}

/// Java `hashCode`: consistent with `equals`.
impl std::hash::Hash for DirectiveDef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity_key().hash(state);
    }
}
