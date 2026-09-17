//! `IMOD/Etomo/src/etomo/type/FileType.java`.
//!
//! A class that can describe the types of files used in Etomo.  Gives a description of
//! the name where possible.  Includes the `ImodManager` key if it exists.  Gives the
//! location of the file if it is in a subdirectory rather then in the main dataset
//! directory.  The file types with name descriptions are stored in a list; unit tests
//! are used to prevent name collisions of these files.
//!
//! **Representation.**  `FileType` extends `etomo/type/FileKey.java`; the superclass is
//! held as the `file_key` field and reached through `Deref`.  The `Object[]
//! fileNamePattern` arrays hold `String`, `Variable`, `ExtensionMarker`, `Extension` and
//! `FileType` elements, so they are `Vec<PatternElement>` here.  The
//! `FileType`-to-`FileType` references (`subFileType`, `singleFileType`, `dualFileType`,
//! `subdir`, and a pattern's `FileType` elements) are `Arc<FileType>`, and
//! `parentFileType` - which `constructDifferentDualSingleImageFileInstance` writes back
//! into its two children, forming a cycle Java's collector handles - is a `Weak`.
//!
//! **Statics.**  The `public static final FileType` singletons (FileType.java:318-1037)
//! are built together in `CLASS`, a single `LazyLock`, because Java runs their
//! initialisers in declaration order the first time the class is touched and each
//! constructor registers the instance in `namedFileTypeList`; one lazy static each would
//! register them in call order instead.  `etomo/type/extension.rs` uses the same
//! modelling.  Three deprecated singletons are the exception - see the `TODO(unit)` in
//! `CLASS`.
//!
//! **Frontier.**  The unit's remaining blockers are named by the `// TODO(unit):`
//! comments below.  `FileType.Variable.toFormattedString` and the five
//! `VariableTestTool` wrappers need `java.text.DecimalFormat` with a run-time pattern
//! and `BigDecimal.setScale(HALF_UP)`, which is a JDK class rather than an etomo unit.
#![allow(dead_code)]

use std::sync::{Arc, LazyLock, Mutex, Weak};

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::base_meta_data::BaseMetaData;
use super::extension::{self, Extension};
use super::extension_marker::ExtensionMarker;
use super::file_key::FileKey;
use super::image_file_meta_data::ImageFileMetaData;
use super::image_filename_style::ImageFilenameStyle;
use super::process_name::ProcessName;
use super::status::Status;
use super::validation_type::ValidationType;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::util::utilities;
use crate::imod::etomo::util::utilities::java_io_file_get_absolute_path;

/// Java `COM_DIR`.
pub const COM_DIR: &str = "com";
/// Java `CHUNK_NUMBER_DIVIDER`.
pub const CHUNK_NUMBER_DIVIDER: &str = "-";
/// Java `CHUNK_NUMBER_DIVIDER_VOL`.
pub const CHUNK_NUMBER_DIVIDER_VOL: &str = "-vol";

/// Java `INSTANCES`, a `Map<String, FileTypeCollection>` keyed by name.  Nothing in the
/// vendored source reads or writes it, and `FileTypeCollection` is an empty marker
/// interface (`etomo/type/FileTypeCollection.java`), so no concrete type implements it.
static INSTANCES: LazyLock<
    Mutex<
        std::collections::HashMap<
            String,
            Box<dyn crate::imod::etomo::r#type::file_type_collection::FileTypeCollection + Send>,
        >,
    >,
> = LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

/// Java `namedFileTypeList` (deprecated 2/1/2019).  Every constructor whose `unnamed`
/// argument is false appends to it, and `getInstance` and `iterator` read it.
static NAMED_FILE_TYPE_LIST: LazyLock<Mutex<Vec<Arc<FileType>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// Java `DEBUG`: `EtomoDirector.INSTANCE.getArguments().isDebug()`, read once when the
/// class is initialised.
static DEBUG: LazyLock<bool> = LazyLock::new(|| {
    crate::imod::etomo::etomo_director::ARGUMENTS
        .lock()
        .unwrap()
        .is_debug()
});

/// One element of a Java `Object[] fileNamePattern`.  The element types the source puts
/// in those arrays are `String`, `Variable`, `ExtensionMarker`, `Extension`, `FileType`
/// and `null`; `buildPattern`, `joinPatterns` and `toString(Object[])` all branch on
/// `instanceof`, so the element type is part of the behaviour.
#[derive(Clone, Debug)]
pub enum PatternElement {
    /// A `java.lang.String` element.
    Str(String),
    /// A `Variable` element.
    Variable(Variable),
    /// An `ExtensionMarker` element.
    ExtensionMarker(ExtensionMarker),
    /// An `Extension` element.
    Extension(&'static Extension),
    /// A `FileType` element.
    FileType(Arc<FileType>),
    /// A `null` element, which `joinPatterns` writes into the tail of its result.
    Null,
}

impl PatternElement {
    /// `Object.toString()` on a pattern element.  `buildPattern`'s final `else if
    /// (pattern[i] != null)` branch appends this.
    fn to_string_element(&self) -> String {
        match self {
            PatternElement::Str(value) => value.clone(),
            PatternElement::Variable(variable) => variable.to_string(),
            PatternElement::ExtensionMarker(extension_marker) => extension_marker.to_string(),
            PatternElement::Extension(extension) => extension.to_string(),
            PatternElement::FileType(file_type) => file_type.to_string(),
            PatternElement::Null => "null".to_string(),
        }
    }
}

/// Java nested class `Variable`.  The numeric integer match strings allow longer
/// integers and require the number mentioned - which is the format.  The numbers are
/// file numbers and there may be situations where the number files is larger then
/// anticipated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Variable {
    /// Java field `regex`.
    regex: String,
    /// Java field `dualRegex`.
    dual_regex: Option<String>,
    /// Java field `validationType`.
    validation_type: ValidationType,
    /// Java field `padTo`: number or leading or trailing zeros.
    pad_to: Option<i32>,
    /// Java field `round`.
    round: bool,
    /// Java field `maxInteger`.
    max_integer: Option<i32>,
}

impl Variable {
    /// Java `DATASET`.  See Tomography Guide 1.5. File Format and Naming Conventions.
    pub fn dataset() -> Variable {
        Variable::get_string_instance(
            &("[^\\s`'\"!#$%&*(){};/?\\|".to_string()
                + (if utilities::is_mac_os() || utilities::is_windows_os() {
                    // Java concatenates a `char` here, which widens to its decimal value
                    // in `char + String`; the source's expression is
                    // `(Utilities.isMacOS() || Utilities.isWindowsOS() ? ':' : "")`, a
                    // conditional whose branches are `char` and `String`, so the `char`
                    // is boxed to Character and its toString is used.
                    ":"
                } else {
                    ""
                })
                + "]+"),
            None,
        )
    }

    /// Java `AXIS`.  a, b, or nothing.
    pub fn axis() -> Variable {
        Variable::get_string_instance("[ab]?", Some("[ab]"))
    }

    /// Java `DATASET_AND_AXIS`.  Dataset followed by axis.
    pub fn dataset_and_axis() -> Variable {
        let dataset = Variable::dataset();
        let axis = Variable::axis();
        Variable::get_string_instance(
            &(dataset.regex.clone() + &axis.regex),
            Some(&(dataset.regex + axis.dual_regex.as_deref().unwrap_or("null"))),
        )
    }

    /// Java `ORIG_RAW_IMAGE_EXTENSION`.
    pub fn orig_raw_image_extension() -> Variable {
        Variable::get_string_instance(
            &("\\".to_string()
                + extension::EXTENSION_DIVIDER
                + &Extension::get_image_input_regex()),
            None,
        )
    }

    /// Java `ONE_DIGIT_INTEGER`.  Original regex: "\\d\\d+". Match n, nn, etc.
    pub fn one_digit_integer() -> Variable {
        Variable::get_integer_instance(1, Some(9))
    }

    /// Java `TWO_DIGIT_INTEGER`.  Original regex: "\\d\\d+". Match nn, nnn, etc.
    fn two_digit_integer() -> Variable {
        Variable::get_integer_instance(2, Some(99))
    }

    /// Java `THREE_DIGIT_INTEGER`.  Original regex: "\\d\\d\\d+". Match nnn, nnnn, etc.
    fn three_digit_integer() -> Variable {
        Variable::get_integer_instance(3, Some(999))
    }

    /// Java `FOUR_DIGIT_INTEGER`.  Original regex: "\\d\\d\\d\\d+". Match nnnn,
    /// nnnnn, etc.
    fn four_digit_integer() -> Variable {
        Variable::get_integer_instance(4, Some(9999))
    }

    /// Java `ANY_INTEGER`.  "\\d+". Match n, nn, nnn, nnnn, nnnnn, nnnnnn, etc.
    fn any_integer() -> Variable {
        Variable::get_integer_instance_any()
    }

    /// Java `FLOAT`.  Original regex: "\\d+\\.\\d+". Match n.n, nn.nn, etc.
    /// Padding: 0.0
    fn float() -> Variable {
        Variable::get_float_instance("\\d+\\.\\d+")
    }

    /// Java `PRECISION_TWO_FLOAT`.  Original regex: "\\d+\\.\\d\\d". Match n.nn,
    /// nn.nn, etc.
    fn precision_two_float() -> Variable {
        Variable::get_float_with_rounding_instance("\\d+", 2)
    }

    /// Java `PRECISION_THREE_FLOAT`.  Original regex: "\\d+\\.\\d\\d\\d". Match
    /// n.nnn, nn.nnn, etc.
    fn precision_three_float() -> Variable {
        Variable::get_float_with_rounding_instance("\\d+", 3)
    }

    /// Java `PRECISION_THREE_FRACTION`.  Original regex: "0\\.\\d\\d\\d". Match 0.nnn.
    fn precision_three_fraction() -> Variable {
        Variable::get_float_with_rounding_instance("0", 3)
    }

    /// Java `Variable(String, String, ValidationType, Integer, Integer, boolean)`.
    fn new(
        regex: &str,
        dual_regex: Option<&str>,
        validation_type: ValidationType,
        pad_to: Option<i32>,
        max_integer: Option<i32>,
        round: bool,
    ) -> Variable {
        Variable {
            regex: regex.to_string(),
            dual_regex: dual_regex.map(|dual_regex| dual_regex.to_string()),
            validation_type,
            pad_to,
            round,
            max_integer,
        }
    }

    /// Java `selectIntegerInstance`.
    pub fn select_integer_instance(num_digits: i32) -> Option<Variable> {
        match num_digits {
            1 => return Some(Variable::one_digit_integer()),
            2 => return Some(Variable::two_digit_integer()),
            3 => return Some(Variable::three_digit_integer()),
            4 => return Some(Variable::four_digit_integer()),
            _ => {}
        }
        None
    }

    /// Java `getStringInstance`.
    fn get_string_instance(regex: &str, dual_regex: Option<&str>) -> Variable {
        Variable::new(regex, dual_regex, ValidationType::String, None, None, false)
    }

    /// Java `getIntegerInstance` (int, Integer overload).  Creates a regular
    /// expression of an integer with a minimum but no maximum number of digits.
    /// Equivalent to:
    /// 2: \\d\\d+
    /// 3: \\d\\d\\d+
    /// 4: \\d\\d\\d\\d+
    fn get_integer_instance(pad_to: i32, max_integer: Option<i32>) -> Variable {
        Variable::new(
            &format!("\\d{{{},}}", pad_to),
            None,
            ValidationType::Integer,
            Some(pad_to),
            max_integer,
            true,
        )
    }

    /// Java `getIntegerInstance` (no-argument overload).
    fn get_integer_instance_any() -> Variable {
        Variable::new("\\d+", None, ValidationType::Integer, None, None, true)
    }

    /// Java `getFloatWithRoundingInstance`.  Equivalent to:
    /// ("\\d+",3): \\d+\\.\\d\\d\\d
    /// ("0", 3  ): 0\\.\\d\\d\\d
    fn get_float_with_rounding_instance(prefix: &str, pad_to: i32) -> Variable {
        Variable::new(
            &format!("{}\\.\\d{{{}}}", prefix, pad_to),
            None,
            ValidationType::FloatingPoint,
            Some(pad_to),
            None,
            true,
        )
    }

    /// Java `getFloatInstance`.  A basic float with no maximum precision.
    fn get_float_instance(regex: &str) -> Variable {
        Variable::new(
            regex,
            None,
            ValidationType::FloatingPoint,
            Some(1),
            None,
            false,
        )
    }

    /// Java `toFormattedString(Number)`.  Rust formatting supplies the same
    /// fixed-width DecimalFormat patterns; the explicit half-up helper avoids
    /// the platform's default tie-breaking rule.
    pub fn to_formatted_string(&self, value: Option<f64>) -> Option<String> {
        let value = value?;
        if !self.is_numeric() || self.pad_to.is_none_or(|pad| pad < 1) {
            return Some(value.to_string());
        }
        if !self.round {
            return Some(value.to_string());
        }
        let digits = self.pad_to.unwrap() as usize;
        if self.is_integer() {
            return Some(((value + 0.5).floor() as i64).to_string());
        }
        let scale = 10_f64.powi(digits as i32);
        let rounded = if value >= 0.0 {
            (value * scale + 0.5).floor()
        } else {
            (value * scale - 0.5).ceil()
        } / scale;
        Some(format!("{rounded:.digits$}"))
    }

    // Formerly untranslated: Java's private `toFormattedString(Number)` builds a
    // `java.text.DecimalFormat` pattern at run time and rounds through
    // `java.math.BigDecimal.setScale(RoundingMode.HALF_UP)`.  Its `Converter.toDouble`
    // call is now translated (`etomo::logic::converter::to_double`), but no model of
    // DecimalFormat with an arbitrary pattern exists -
    // `utilities::java_text_decimal_format_three_fraction_digits` covers only the one
    // fixed pattern it was written for.  The blocker is a JDK class, not an etomo
    // source unit, so this carries no TODO(unit) marker.  `getFileNameFromPattern`
    // calls it to turn `numeric1`/`numeric2` into `formattedNumeric1`/
    // `formattedNumeric2`, so that transfer is left out there too.

    /// Java `isNumeric`.
    fn is_numeric(&self) -> bool {
        self.validation_type.is_numeric()
    }

    /// Java `isInteger`.
    fn is_integer(&self) -> bool {
        self.validation_type.is_integer()
    }

    /// Java `getMaxInteger`.
    pub fn get_max_integer(&self) -> Option<i32> {
        self.max_integer
    }
}

/// Java `toString` on the nested `Variable` class.
impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.regex)
    }
}

/// Java package-private `VariableTestTool` formatting entry points.
pub struct VariableTestTool;
impl VariableTestTool {
    #[allow(non_snake_case)]
    pub fn toOneDigitIntegerFormattedString(value: Option<f64>) -> Option<String> {
        Variable::one_digit_integer().to_formatted_string(value)
    }
    #[allow(non_snake_case)]
    pub fn toTwoDigitIntegerFormattedString(value: Option<f64>) -> Option<String> {
        Variable::two_digit_integer().to_formatted_string(value)
    }
    #[allow(non_snake_case)]
    pub fn toFloatFormattedString(value: Option<f64>) -> Option<String> {
        Variable::float().to_formatted_string(value)
    }
    #[allow(non_snake_case)]
    pub fn toPrecisionThreeFloatFormattedString(value: Option<f64>) -> Option<String> {
        Variable::precision_three_float().to_formatted_string(value)
    }
    #[allow(non_snake_case)]
    pub fn toPrecisionThreeFractionFormattedString(value: Option<f64>) -> Option<String> {
        Variable::precision_three_fraction().to_formatted_string(value)
    }
}

// Untranslated: Java's nested `VariableTestTool` class
// (`toOneDigitIntegerFormattedString`, `toTwoDigitIntegerFormattedString`,
// `toFloatFormattedString`, `toPrecisionThreeFloatFormattedString`,
// `toPrecisionThreeFractionFormattedString`) delegates to
// `Variable.toFormattedString`, which is blocked above on java.text.DecimalFormat.

/// Java `containsValidDatasetName`.  Checks a file name, or path, to see if it contains
/// a valid dataset name.  `err_msg`, when it is not null, collects a message if the
/// return value is false.
pub fn contains_valid_dataset_name(file_path: Option<&str>, err_msg: Option<&mut String>) -> bool {
    let file_path = match file_path {
        None => {
            if let Some(err_msg) = err_msg {
                err_msg.push_str("No input image file path found.  ");
            }
            return false;
        }
        Some(file_path) if file_path.is_empty() => {
            if let Some(err_msg) = err_msg {
                err_msg.push_str("No input image file path found.  ");
            }
            return false;
        }
        Some(file_path) => file_path,
    };
    let mut file_name = utilities::java_io_file_get_name(file_path);
    if file_name.contains(extension::EXTENSION_DIVIDER) {
        file_name = utilities::remove_extension(Some(&file_name)).unwrap_or_default();
    }
    let dataset_and_axis = Variable::dataset_and_axis();
    if !java_lang_string_matches(&file_name, &dataset_and_axis.regex) {
        if let Some(err_msg) = err_msg {
            err_msg.push_str(&format!(
                "file name {} contains illegal characters[s].  See Tomography Guide 1.5. File Format and Naming Conventions.  ",
                file_path
            ));
        }
        return false;
    }
    true
}

/// Java `DynamicLocation`.  For template file names, this is part of the file name that
/// changes.  The source declares three private singletons and a private constructor, so
/// the class is only ever compared by identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynamicLocation {
    /// Java `AFTER_EXTENSION`.  After extension example: .srecnn
    AfterExtension,
    /// Java `AFTER_TYPE_STRING`.  After type string example: dateset_slfinnnn.mrc
    AfterTypeString,
    /// Java `AFTER_TYPE_STRING_AND_AFTER_MIDDLE_PIECE`.  After type string and middle
    /// piece example: dataset_gfc0.xxx-f0.xxx.mrc
    AfterTypeStringAndAfterMiddlePiece,
}

/// The `public static final FileType` singletons (FileType.java:318-1037).  Java runs
/// these initialisers in declaration order the first time the class is touched, and
/// each constructor registers the instance in `namedFileTypeList`, so they are built
/// together in one `LazyLock` - the same modelling `etomo/type/extension.rs` uses -
/// rather than one lazy static each, which would register them in call order.
pub struct ClassStatics {
    /// Java `ORIG_COMS_DIR`.
    pub orig_coms_dir: Arc<FileType>,
    /// Java `FIDUCIAL_3D_MODEL`.
    pub fiducial_3d_model: Arc<FileType>,
    /// Java `SERIES_WATCHER_PROJECT`.
    pub series_watcher_project: Arc<FileType>,
    /// Java `BATCH_RUN_TOMO_GLOBAL_AUTODOC`.
    pub batch_run_tomo_global_autodoc: Arc<FileType>,
    /// Java `DEFAULT_BATCH_RUN_TOMO_AUTODOC`.
    pub default_batch_run_tomo_autodoc: Arc<FileType>,
    /// Java `LOCAL_BATCH_DIRECTIVE_FILE`.
    pub local_batch_directive_file: Arc<FileType>,
    /// Java `LOCAL_SCOPE_TEMPLATE`.
    pub local_scope_template: Arc<FileType>,
    /// Java `LOCAL_SYSTEM_TEMPLATE`.
    pub local_system_template: Arc<FileType>,
    /// Java `LOCAL_USER_TEMPLATE`.
    pub local_user_template: Arc<FileType>,
    /// Java `ALIGNED_STACK_OLD`.
    pub aligned_stack_old: Arc<FileType>,
    /// Java `ALIGNED_STACK`.
    pub aligned_stack: Arc<FileType>,
    /// Java `NEWST_OR_BLEND_3D_FIND_OUTPUT_OLD`.
    pub newst_or_blend_3d_find_output_old: Arc<FileType>,
    /// Java `NEWST_OR_BLEND_3D_FIND_OUTPUT`.
    pub newst_or_blend_3d_find_output: Arc<FileType>,
    /// Java `CTF_CORRECTED_STACK_OLD`.
    pub ctf_corrected_stack_old: Arc<FileType>,
    /// Java `CTF_CORRECTED_STACK`.
    pub ctf_corrected_stack: Arc<FileType>,
    /// Java `ERASED_BEADS_STACK_OLD`.
    pub erased_beads_stack_old: Arc<FileType>,
    /// Java `ERASED_BEADS_STACK`.
    pub erased_beads_stack: Arc<FileType>,
    /// Java `MTF_FILTERED_STACK_OLD`.
    pub mtf_filtered_stack_old: Arc<FileType>,
    /// Java `MTF_FILTERED_STACK`.
    pub mtf_filtered_stack: Arc<FileType>,
    /// Java `TRANSFORMED_REFINING_MODEL`.
    pub transformed_refining_model: Arc<FileType>,
    /// Java `XCORR_BLEND_OUTPUT_OLD`.
    pub xcorr_blend_output_old: Arc<FileType>,
    /// Java `XCORR_BLEND_OUTPUT`.
    pub xcorr_blend_output: Arc<FileType>,
    /// Java `CHECK_FILE`.
    pub check_file: Arc<FileType>,
    /// Java `SERIES_WATCHER_CHECK_FILE`.
    pub series_watcher_check_file: Arc<FileType>,
    /// Java `ALIGN_COMSCRIPT`.
    pub align_comscript: Arc<FileType>,
    /// Java `AUTOFIDSEED_COMSCRIPT`.
    pub autofidseed_comscript: Arc<FileType>,
    /// Java `BATCH_RUN_TOMO_COMSCRIPT`.
    pub batch_run_tomo_comscript: Arc<FileType>,
    /// Java `BLEND_COMSCRIPT`.
    pub blend_comscript: Arc<FileType>,
    /// Java `COPYTOMOCOMS_COMSCRIPT`.
    pub copytomocoms_comscript: Arc<FileType>,
    /// Java `CRYO_POSITION_COMSCRIPT`.
    pub cryo_position_comscript: Arc<FileType>,
    /// Java `CTF_3D_SETUP_COMSCRIPT`.
    pub ctf_3d_setup_comscript: Arc<FileType>,
    /// Java `CTF_CORRECTION_COMSCRIPT`.
    pub ctf_correction_comscript: Arc<FileType>,
    /// Java `FIND_BEADS_3D_COMSCRIPT`.
    pub find_beads_3d_comscript: Arc<FileType>,
    /// Java `FLATTEN_COMSCRIPT`.
    pub flatten_comscript: Arc<FileType>,
    /// Java `FLATTEN_TOOL_COMSCRIPT`.
    pub flatten_tool_comscript: Arc<FileType>,
    /// Java `GOLD_ERASER_COMSCRIPT`.
    pub gold_eraser_comscript: Arc<FileType>,
    /// Java `MULTIFILT_SETUP_COMSCRIPT`.
    pub multifilt_setup_comscript: Arc<FileType>,
    /// Java `MTF_FILTER_COMSCRIPT`.
    pub mtf_filter_comscript: Arc<FileType>,
    /// Java `NEWST_COMSCRIPT`.
    pub newst_comscript: Arc<FileType>,
    /// Java `PREBLEND_COMSCRIPT`.
    pub preblend_comscript: Arc<FileType>,
    /// Java `SIRTSETUP_COMSCRIPT`.
    pub sirtsetup_comscript: Arc<FileType>,
    /// Java `SLOPPY_BLEND_COMSCRIPT`.
    pub sloppy_blend_comscript: Arc<FileType>,
    /// Java `TILT_COMSCRIPT`.
    pub tilt_comscript: Arc<FileType>,
    /// Java `TILT_FOR_SIRT_COMSCRIPT`.
    pub tilt_for_sirt_comscript: Arc<FileType>,
    /// Java `TILT_FOR_POS_SAMPLE_COMSCRIPT`.
    pub tilt_for_pos_sample_comscript: Arc<FileType>,
    /// Java `TRACK_COMSCRIPT`.
    pub track_comscript: Arc<FileType>,
    /// Java `TRACK_ADJUSTED_COMSCRIPT`.
    pub track_adjusted_comscript: Arc<FileType>,
    /// Java `TRACK_ORIG_COMSCRIPT`.
    pub track_orig_comscript: Arc<FileType>,
    /// Java `CROSS_CORRELATION_COMSCRIPT`.
    pub cross_correlation_comscript: Arc<FileType>,
    /// Java `PATCH_TRACKING_COMSCRIPT`.
    pub patch_tracking_comscript: Arc<FileType>,
    /// Java `DIRECTIVES_DESCR`.
    pub directives_descr: Arc<FileType>,
    /// Java `JOIN_WARP_2_MODEL_COMSCRIPT`.
    pub join_warp_2_model_comscript: Arc<FileType>,
    /// Java `ALIGN_FRAMES_OUTPUT_COMSCRIPT`.
    pub align_frames_output_comscript: Arc<FileType>,
    /// Java `SUBTOMO_SETUP_COMSCRIPT`.
    pub subtomo_setup_comscript: Arc<FileType>,
    /// Java `ALT_TOMO_SETUP_COMSCRIPT`.
    pub alt_tomo_setup_comscript: Arc<FileType>,
    /// Java `RESTRICT_ALIGN_COMSCRIPT`.
    pub restrict_align_comscript: Arc<FileType>,
    /// Java `REDUCE_FILT_VOL_COMSCRIPT`.
    pub reduce_filt_vol_comscript: Arc<FileType>,
    /// Java `SERIES_WATCHER_COMSCRIPT`.
    pub series_watcher_comscript: Arc<FileType>,
    /// Java `DISTORTION_CORRECTED_STACK_OLD`.
    pub distortion_corrected_stack_old: Arc<FileType>,
    /// Java `DISTORTION_CORRECTED_STACK`.
    pub distortion_corrected_stack: Arc<FileType>,
    /// Java `AUTOFIDSEED_DIR`.
    pub autofidseed_dir: Arc<FileType>,
    /// Java `BATCH_RUN_TOMO_PROJECT`.
    pub batch_run_tomo_project: Arc<FileType>,
    /// Java `PIECE_SHIFTS`.
    pub piece_shifts: Arc<FileType>,
    /// Java `MANUAL_REPLACEMENT_MODEL`.
    pub manual_replacement_model: Arc<FileType>,
    /// Java `FIDUCIAL_MODEL`.
    pub fiducial_model: Arc<FileType>,
    /// Java `CCD_ERASER_BEADS_INPUT_MODEL`.
    pub ccd_eraser_beads_input_model: Arc<FileType>,
    /// Java `FIDUCIAL_NO_GAPS_MODEL`.
    pub fiducial_no_gaps_model: Arc<FileType>,
    /// Java `FIDUCIAL_PATCH_TRACKING_MODEL`.
    pub fiducial_patch_tracking_model: Arc<FileType>,
    /// Java `FLATTEN_TOOL_OUTPUT_OLD`.
    pub flatten_tool_output_old: Arc<FileType>,
    /// Java `FLATTEN_TOOL_OUTPUT`.
    pub flatten_tool_output: Arc<FileType>,
    /// Java `EXCLUDE_VIEWS_INFO`.
    pub exclude_views_info: Arc<FileType>,
    /// Java `NAD_TEST_INPUT_OLD`.
    pub nad_test_input_old: Arc<FileType>,
    /// Java `NAD_TEST_INPUT`.
    pub nad_test_input: Arc<FileType>,
    /// Java `JOIN_OLD`.
    pub join_old: Arc<FileType>,
    /// Java `JOIN`.
    pub join: Arc<FileType>,
    /// Java `MODELED_JOIN_OLD`.
    pub modeled_join_old: Arc<FileType>,
    /// Java `MODELED_JOIN`.
    pub modeled_join: Arc<FileType>,
    /// Java `TRIAL_JOIN_OLD`.
    pub trial_join_old: Arc<FileType>,
    /// Java `TRIAL_JOIN`.
    pub trial_join: Arc<FileType>,
    /// Java `BATCH_RUN_TOMO_LOG`.
    pub batch_run_tomo_log: Arc<FileType>,
    /// Java `ALT_TOMO_SETUP_LOG`.
    pub alt_tomo_setup_log: Arc<FileType>,
    /// Java `BATCH_RUN_TOMO_DATASET_LOG`.
    pub batch_run_tomo_dataset_log: Arc<FileType>,
    /// Java `CTF_3D_FINISH_LOG`.
    pub ctf_3d_finish_log: Arc<FileType>,
    /// Java `CTF_3D_SETUP_LOG`.
    pub ctf_3d_setup_log: Arc<FileType>,
    /// Java `ERASER_LOG`.
    pub eraser_log: Arc<FileType>,
    /// Java `TILT_ALIGN_LOG`.
    pub tilt_align_log: Arc<FileType>,
    /// Java `GPU_TEST_LOG`.
    pub gpu_test_log: Arc<FileType>,
    /// Java `PREBLEND_LOG`.
    pub preblend_log: Arc<FileType>,
    /// Java `PROJECT_LOG`.
    pub project_log: Arc<FileType>,
    /// Java `SERIES_WATCHER_BRT_ROW_LOG`.
    pub series_watcher_brt_row_log: Arc<FileType>,
    /// Java `ALIGN_ANGLES_LOG`.
    pub align_angles_log: Arc<FileType>,
    /// Java `ALIGN_ERROR_LOG`.
    pub align_error_log: Arc<FileType>,
    /// Java `ALIGN_ROBUST_LOG`.
    pub align_robust_log: Arc<FileType>,
    /// Java `ALIGN_SOLUTION_LOG`.
    pub align_solution_log: Arc<FileType>,
    /// Java `CROSS_CORRELATION_LOG`.
    pub cross_correlation_log: Arc<FileType>,
    /// Java `ALIGN_FRAMES_LOG`.
    pub align_frames_log: Arc<FileType>,
    /// Java `RESTRICT_ALIGN_LOG`.
    pub restrict_align_log: Arc<FileType>,
    /// Java `SUBTOMO_SETUP_LOG`.
    pub subtomo_setup_log: Arc<FileType>,
    /// Java `CTF_CORRECTION_LOG`.
    pub ctf_correction_log: Arc<FileType>,
    /// Java `GOLD_ERASER_LOG`.
    pub gold_eraser_log: Arc<FileType>,
    /// Java `MTF_FILTER_LOG`.
    pub mtf_filter_log: Arc<FileType>,
    /// Java `REDUCE_FILT_VOL_LOG`.
    pub reduce_filt_vol_log: Arc<FileType>,
    /// Java `SERIES_WATCHER_LOG`.
    pub series_watcher_log: Arc<FileType>,
    /// Java `MTF_FILTER_MDOC`.
    pub mtf_filter_mdoc: Arc<FileType>,
    /// Java `FIND_BEADS_3D_OUTPUT_MODEL`.
    pub find_beads_3d_output_model: Arc<FileType>,
    /// Java `AUTOFIDSEED_BOUNDARY_MODEL`.
    pub autofidseed_boundary_model: Arc<FileType>,
    /// Java `AUTO_ALIGN_BOUNDARY_MODEL`.
    pub auto_align_boundary_model: Arc<FileType>,
    /// Java `SMOOTHING_ASSESSMENT_OUTPUT_MODEL`.
    pub smoothing_assessment_output_model: Arc<FileType>,
    /// Java `CLUSTERED_ELONGATED_MODEL`.
    pub clustered_elongated_model: Arc<FileType>,
    /// Java `FLATTEN_WARP_INPUT_MODEL`.
    pub flatten_warp_input_model: Arc<FileType>,
    /// Java `PATCH_VECTOR_MODEL`.
    pub patch_vector_model: Arc<FileType>,
    /// Java `PATCH_VECTOR_CCC_MODEL`.
    pub patch_vector_ccc_model: Arc<FileType>,
    /// Java `PATCH_TRACKING_BOUNDARY_MODEL`.
    pub patch_tracking_boundary_model: Arc<FileType>,
    /// Java `BATCH_RUN_TOMO_BOUNDARY_MODEL`.
    pub batch_run_tomo_boundary_model: Arc<FileType>,
    /// Java `TOMOPITCH_MODEL`.
    pub tomopitch_model: Arc<FileType>,
    /// Java `ALIGNED_STACK_MRC_OLD`.
    pub aligned_stack_mrc_old: Arc<FileType>,
    /// Java `ALIGNED_STACK_MRC`.
    pub aligned_stack_mrc: Arc<FileType>,
    /// Java `PREBLEND_OUTPUT_MRC_OLD`.
    pub preblend_output_mrc_old: Arc<FileType>,
    /// Java `PREBLEND_OUTPUT_MRC`.
    pub preblend_output_mrc: Arc<FileType>,
    /// Java `MUTLIFILT_EXACT_OBJECT_SIZES_OUTPUT_TEMPLATE_OLD`.
    pub mutlifilt_exact_object_sizes_output_template_old: Arc<FileType>,
    /// Java `MUTLIFILT_EXACT_OBJECT_SIZES_OUTPUT_TEMPLATE`.
    pub mutlifilt_exact_object_sizes_output_template: Arc<FileType>,
    /// Java `MUTLIFILT_GAUSSIAN_OUTPUT_TEMPLATE_OLD`.
    pub mutlifilt_gaussian_output_template_old: Arc<FileType>,
    /// Java `MUTLIFILT_GAUSSIAN_OUTPUT_TEMPLATE`.
    pub mutlifilt_gaussian_output_template: Arc<FileType>,
    /// Java `MUTLIFILT_HAMMING_LIKE_STARTS_OUTPUT_TEMPLATE_OLD`.
    pub mutlifilt_hamming_like_starts_output_template_old: Arc<FileType>,
    /// Java `MUTLIFILT_HAMMING_LIKE_STARTS_OUTPUT_TEMPLATE`.
    pub mutlifilt_hamming_like_starts_output_template: Arc<FileType>,
    /// Java `MUTLIFILT_FAKE_SIRT_ITERATIONS_OUTPUT_TEMPLATE_OLD`.
    pub mutlifilt_fake_sirt_iterations_output_template_old: Arc<FileType>,
    /// Java `MUTLIFILT_FAKE_SIRT_ITERATIONS_OUTPUT_TEMPLATE`.
    pub mutlifilt_fake_sirt_iterations_output_template: Arc<FileType>,
    /// Java `TEST_NAD`.
    pub test_nad: Arc<FileType>,
    /// Java `ANISOTROPIC_DIFFUSION_OUTPUT_OLD`.
    pub anisotropic_diffusion_output_old: Arc<FileType>,
    /// Java `ANISOTROPIC_DIFFUSION_OUTPUT`.
    pub anisotropic_diffusion_output: Arc<FileType>,
    /// Java `PIECE_LIST`.
    pub piece_list: Arc<FileType>,
    /// Java `PREALIGNED_STACK_OLD`.
    pub prealigned_stack_old: Arc<FileType>,
    /// Java `PREALIGNED_STACK`.
    pub prealigned_stack: Arc<FileType>,
    /// Java `PRE_TRANSFORMATION_LIST`.
    pub pre_transformation_list: Arc<FileType>,
    /// Java `PRE_XG`.
    pub pre_xg: Arc<FileType>,
    /// Java `MATLAB_PARAM_FILE`.
    pub matlab_param_file: Arc<FileType>,
    /// Java `RAW_TILT_ANGLES`.
    pub raw_tilt_angles: Arc<FileType>,
    /// Java `TRIM_VOL_OUTPUT_OLD`.
    pub trim_vol_output_old: Arc<FileType>,
    /// Java `TRIM_VOL_OUTPUT`.
    pub trim_vol_output: Arc<FileType>,
    /// Java `PROCESSCHUNKS_REC`.
    pub processchunks_rec: Arc<FileType>,
    /// Java `PROCESSCHUNKS_MRC`.
    pub processchunks_mrc: Arc<FileType>,
    /// Java `PROCESSCHUNKS_VOL_MRC`.
    pub processchunks_vol_mrc: Arc<FileType>,
    /// Java `TILT_3D_FIND_OUTPUT_OLD`.
    pub tilt_3d_find_output_old: Arc<FileType>,
    /// Java `TILT_3D_FIND_OUTPUT`.
    pub tilt_3d_find_output: Arc<FileType>,
    /// Java `BOTTOM_SAMPLE_OLD`.
    pub bottom_sample_old: Arc<FileType>,
    /// Java `BOTTOM_SAMPLE`.
    pub bottom_sample: Arc<FileType>,
    /// Java `CRYO_POSITION_OUTPUT_OLD`.
    pub cryo_position_output_old: Arc<FileType>,
    /// Java `CRYO_POSITION_OUTPUT`.
    pub cryo_position_output: Arc<FileType>,
    /// Java `TILT_OUTPUT_DUAL_OLD`.
    pub tilt_output_dual_old: Arc<FileType>,
    /// Java `TILT_OUTPUT_SINGLE_OLD`.
    pub tilt_output_single_old: Arc<FileType>,
    /// Java `TILT_OUTPUT_SINGLE`.
    pub tilt_output_single: Arc<FileType>,
    /// Java `TILT_OUTPUT_DUAL`.
    pub tilt_output_dual: Arc<FileType>,
    /// Java `TILT_OUTPUT_OLD`.
    pub tilt_output_old: Arc<FileType>,
    /// Java `TILT_OUTPUT`.
    pub tilt_output: Arc<FileType>,
    /// Java `CTF_3D_OUTPUT`.
    pub ctf_3d_output: Arc<FileType>,
    /// Java `FLATTEN_OUTPUT_OLD`.
    pub flatten_output_old: Arc<FileType>,
    /// Java `FLATTEN_OUTPUT`.
    pub flatten_output: Arc<FileType>,
    /// Java `MIDDLE_SAMPLE_OLD`.
    pub middle_sample_old: Arc<FileType>,
    /// Java `MIDDLE_SAMPLE`.
    pub middle_sample: Arc<FileType>,
    /// Java `COMBINED_VOLUME_OLD`.
    pub combined_volume_old: Arc<FileType>,
    /// Java `COMBINED_VOLUME`.
    pub combined_volume: Arc<FileType>,
    /// Java `TOP_SAMPLE_OLD`.
    pub top_sample_old: Arc<FileType>,
    /// Java `TOP_SAMPLE`.
    pub top_sample: Arc<FileType>,
    /// Java `JOIN_SAMPLE_AVERAGES_OLD`.
    pub join_sample_averages_old: Arc<FileType>,
    /// Java `JOIN_SAMPLE_AVERAGES`.
    pub join_sample_averages: Arc<FileType>,
    /// Java `JOIN_SAMPLE_OLD`.
    pub join_sample_old: Arc<FileType>,
    /// Java `JOIN_SAMPLE`.
    pub join_sample: Arc<FileType>,
    /// Java `SEED_MODEL`.
    pub seed_model: Arc<FileType>,
    /// Java `SIRT_SCALED_OUTPUT_TEMPLATE_OLD`.
    pub sirt_scaled_output_template_old: Arc<FileType>,
    /// Java `SIRT_SCALED_OUTPUT_TEMPLATE`.
    pub sirt_scaled_output_template: Arc<FileType>,
    /// Java `SIRT_SUBAREA_SCALED_OUTPUT_TEMPLATE_OLD`.
    pub sirt_subarea_scaled_output_template_old: Arc<FileType>,
    /// Java `SIRT_SUBAREA_SCALED_OUTPUT_TEMPLATE`.
    pub sirt_subarea_scaled_output_template: Arc<FileType>,
    /// Java `SQUEEZE_VOL_OUTPUT_OLD`.
    pub squeeze_vol_output_old: Arc<FileType>,
    /// Java `SQUEEZE_VOL_OUTPUT`.
    pub squeeze_vol_output: Arc<FileType>,
    /// Java `SIRT_OUTPUT_TEMPLATE_OLD`.
    pub sirt_output_template_old: Arc<FileType>,
    /// Java `SIRT_OUTPUT_TEMPLATE`.
    pub sirt_output_template: Arc<FileType>,
    /// Java `SIRT_SUBAREA_OUTPUT_TEMPLATE_OLD`.
    pub sirt_subarea_output_template_old: Arc<FileType>,
    /// Java `SIRT_SUBAREA_OUTPUT_TEMPLATE`.
    pub sirt_subarea_output_template: Arc<FileType>,
    /// Java `RAW_STACK`.
    pub raw_stack: Arc<FileType>,
    /// Java `STATS_LOG_OLD`.
    pub stats_log_old: Arc<FileType>,
    /// Java `PROCESSCHUNKS_LOG`.
    pub processchunks_log: Arc<FileType>,
    /// Java `STATS_LOG`.
    pub stats_log: Arc<FileType>,
    /// Java `FIXED_XRAYS_STACK`.
    pub fixed_xrays_stack: Arc<FileType>,
    /// Java `FIXED_STATS_LOG_OLD`.
    pub fixed_stats_log_old: Arc<FileType>,
    /// Java `FIXED_STATS_LOG`.
    pub fixed_stats_log: Arc<FileType>,
    /// Java `ORIGINAL_RAW_STACK`.
    pub original_raw_stack: Arc<FileType>,
    /// Java `FULL_VSR`.
    pub full_vsr: Arc<FileType>,
    /// Java `SUB_VSR`.
    pub sub_vsr: Arc<FileType>,
    /// Java `TILT_ANGLES`.
    pub tilt_angles: Arc<FileType>,
    /// Java `WARP_XG`.
    pub warp_xg: Arc<FileType>,
    /// Java `EDGE_FUNCTIONS_X`.
    pub edge_functions_x: Arc<FileType>,
    /// Java `LOCAL_TRANSFORMATION_LIST`.
    pub local_transformation_list: Arc<FileType>,
    /// Java `AUTO_LOCAL_TRANSFORMATION_LIST`.
    pub auto_local_transformation_list: Arc<FileType>,
    /// Java `EMPTY_LOCAL_TRANSFORMATION_LIST`.
    pub empty_local_transformation_list: Arc<FileType>,
    /// Java `MIDAS_LOCAL_TRANSFORMATION_LIST`.
    pub midas_local_transformation_list: Arc<FileType>,
    /// Java `GLOBAL_TRANSFORMATION_LIST`.
    pub global_transformation_list: Arc<FileType>,
    /// Java `ALT_STACK_ROOTNAME_EVEN_FILE`.
    pub alt_stack_rootname_even_file: Arc<FileType>,
    /// Java `ALT_STACK_ROOTNAME_ODD_FILE`.
    pub alt_stack_rootname_odd_file: Arc<FileType>,
    /// Java `ALT_STACK_EVEN_TOMOGRAM`.
    pub alt_stack_even_tomogram: Arc<FileType>,
    /// Java `ALT_STACK_ODD_TOMOGRAM`.
    pub alt_stack_odd_tomogram: Arc<FileType>,
    /// Java `ALT_STACK_EVEN_FULL_TOMOGRAM`.
    pub alt_stack_even_full_tomogram: Arc<FileType>,
    /// Java `ALT_STACK_ODD_FULL_TOMOGRAM`.
    pub alt_stack_odd_full_tomogram: Arc<FileType>,
    /// Java `ALT_STACK_TOMOGRAM`.
    pub alt_stack_tomogram: Arc<FileType>,
    /// Java `REDUCE_FILT_VOL_OUTPUT_FILE`.
    pub reduce_filt_vol_output_file: Arc<FileType>,
    /// Java `FLATTEN_REDUCE_FILT_VOL_FILE`.
    pub flatten_reduce_filt_vol_file: Arc<FileType>,
}

/// The single instance of the class's static file-type state.
pub static CLASS: LazyLock<ClassStatics> = LazyLock::new(|| {
    let orig_coms_dir = FileType::construct_instance(false, false, Some("origcoms"), Some(""));
    let fiducial_3d_model = FileType::construct_imod_instance(
        true,
        true,
        Some(""),
        Some(".3dmod"),
        Some(imod_manager::FIDUCIAL_MODEL_KEY),
    );
    let series_watcher_project = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset()),
        PatternElement::Str(".".to_string()),
        PatternElement::Str(extension::CLASS.ebt.to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.active),
    ]));
    let batch_run_tomo_global_autodoc =
        FileType::construct_instance(true, false, Some(""), Some(".adoc"));
    let default_batch_run_tomo_autodoc = FileType::construct_imod_dir_instance(
        false,
        false,
        Some("batchDefaults"),
        Some(".adoc"),
        Some(COM_DIR),
    );
    let local_batch_directive_file =
        FileType::construct_instance(false, false, Some("batchDirective"), Some(".adoc"));
    let local_scope_template =
        FileType::construct_instance(false, false, Some("scopeTemplate"), Some(".adoc"));
    let local_system_template =
        FileType::construct_instance(false, false, Some("systemTemplate"), Some(".adoc"));
    let local_user_template =
        FileType::construct_instance(false, false, Some("userTemplate"), Some(".adoc"));
    let aligned_stack_old = FileType::construct_described_imod_image_file_instance(
        true,
        true,
        Some(""),
        Some(".ali"),
        Some(imod_manager::FINE_ALIGNED_KEY),
        Some("the final aligned stack"),
    );
    let aligned_stack = FileType::construct_instance_pattern_key_descr(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.ali),
        ]),
        Some(imod_manager::FINE_ALIGNED_KEY),
        Some("the final aligned stack"),
    );
    let newst_or_blend_3d_find_output_old = FileType::construct_imod_image_file_instance(
        true,
        true,
        Some("_3dfind"),
        Some(".ali"),
        Some(imod_manager::FINE_ALIGNED_3D_FIND_KEY),
    );
    let newst_or_blend_3d_find_output = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_3dfind".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.ali),
        ]),
        Some(imod_manager::FINE_ALIGNED_3D_FIND_KEY),
    );
    let ctf_corrected_stack_old = FileType::construct_imod_image_file_instance(
        true,
        true,
        Some("_ctfcorr"),
        Some(".ali"),
        Some(imod_manager::CTF_CORRECTION_KEY),
    );
    let ctf_corrected_stack = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_ctfcorr".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.ali),
        ]),
        Some(imod_manager::CTF_CORRECTION_KEY),
    );
    let erased_beads_stack_old = FileType::construct_imod_image_file_instance(
        true,
        true,
        Some("_erase"),
        Some(".ali"),
        Some(imod_manager::ERASED_FIDUCIALS_KEY),
    );
    let erased_beads_stack = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_erase".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.ali),
        ]),
        Some(imod_manager::ERASED_FIDUCIALS_KEY),
    );
    let mtf_filtered_stack_old = FileType::construct_imod_image_file_instance(
        true,
        true,
        Some("_filt"),
        Some(".ali"),
        Some(imod_manager::MTF_FILTER_KEY),
    );
    let mtf_filtered_stack = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_filt".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.ali),
        ]),
        Some(imod_manager::MTF_FILTER_KEY),
    );
    let transformed_refining_model = FileType::construct_imod_instance(
        true,
        false,
        Some("_refine"),
        Some(".alimod"),
        Some(imod_manager::TRANSFORMED_MODEL_KEY),
    );
    let xcorr_blend_output_old =
        FileType::construct_image_file_instance(true, true, Some(""), Some(".bl"));
    let xcorr_blend_output = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.bl),
    ]));
    let check_file = FileType::construct_instance(true, true, Some(""), Some(".cmds"));
    let series_watcher_check_file = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str(
            ProcessName::SERIES_WATCHER
                .get_text()
                .unwrap_or("")
                .to_string(),
        ),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.cmds),
    ]));
    let align_comscript = FileType::construct_instance(false, true, Some("align"), Some(".com"));
    let autofidseed_comscript =
        FileType::construct_instance(false, true, Some("autofidseed"), Some(".com"));
    let batch_run_tomo_comscript =
        FileType::construct_instance(true, false, Some(""), Some(".com"));
    let blend_comscript = FileType::construct_instance(false, true, Some("blend"), Some(".com"));
    let copytomocoms_comscript =
        FileType::construct_instance(false, false, Some("copytomocoms"), Some(".com"));
    let cryo_position_comscript =
        FileType::construct_instance(false, true, Some("cryoposition"), Some(".com"));
    let ctf_3d_setup_comscript =
        FileType::construct_instance(false, true, Some("ctf3dsetup"), Some(".com"));
    let ctf_correction_comscript =
        FileType::construct_instance(false, true, Some("ctfcorrection"), Some(".com"));
    let find_beads_3d_comscript =
        FileType::construct_instance(false, true, Some("findbeads3d"), Some(".com"));
    let flatten_comscript =
        FileType::construct_instance(false, false, Some("flatten"), Some(".com"));
    let flatten_tool_comscript =
        FileType::construct_instance(true, false, Some("_flatten"), Some(".com"));
    let gold_eraser_comscript = FileType::construct_instance(
        false,
        true,
        ProcessName::GOLD_ERASER.get_text(),
        Some(".com"),
    );
    let multifilt_setup_comscript =
        FileType::construct_instance(false, true, Some("multifiltsetup"), Some(".com"));
    let mtf_filter_comscript =
        FileType::construct_instance(false, true, Some("mtffilter"), Some(".com"));
    let newst_comscript = FileType::construct_instance(false, true, Some("newst"), Some(".com"));
    let preblend_comscript =
        FileType::construct_instance(false, true, Some("preblend"), Some(".com"));
    let sirtsetup_comscript =
        FileType::construct_instance(false, true, Some("sirtsetup"), Some(".com"));
    let sloppy_blend_comscript = FileType::construct_imod_dir_instance(
        false,
        false,
        Some("sloppyblend"),
        Some(".com"),
        Some(COM_DIR),
    );
    let tilt_comscript = FileType::construct_instance(false, true, Some("tilt"), Some(".com"));
    let tilt_for_sirt_comscript =
        FileType::construct_instance(false, true, Some("tilt"), Some("_for_sirt.com"));
    let tilt_for_pos_sample_comscript = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("tilt_sample".to_string()),
        PatternElement::Variable(Variable::axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.com),
    ]));
    let track_comscript = FileType::construct_instance(false, true, Some("track"), Some(".com"));
    let track_adjusted_comscript = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("track".to_string()),
        PatternElement::Variable(Variable::axis()),
        PatternElement::Str("_adjusted".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.com),
    ]));
    let track_orig_comscript = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("track".to_string()),
        PatternElement::Variable(Variable::axis()),
        PatternElement::Str("_orig".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.com),
    ]));
    let cross_correlation_comscript =
        FileType::construct_instance(false, true, Some("xcorr"), Some(".com"));
    let patch_tracking_comscript =
        FileType::construct_instance(false, true, Some("xcorr_pt"), Some(".com"));
    let directives_descr = FileType::construct_imod_dir_instance(
        false,
        false,
        Some("directives"),
        Some(".csv"),
        Some(COM_DIR),
    );
    let join_warp_2_model_comscript = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str(ProcessName::JOIN_WARP_2_MODEL.to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.com),
    ]));
    let align_frames_output_comscript = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset()),
        PatternElement::Str("_af".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.com),
    ]));
    let subtomo_setup_comscript =
        FileType::construct_instance(false, true, Some("subtomosetup"), Some(".com"));
    let alt_tomo_setup_comscript =
        FileType::construct_instance(false, true, Some("alttomosetup"), Some(".com"));
    let restrict_align_comscript =
        FileType::construct_instance(false, true, Some("restrictalign"), Some(".com"));
    let reduce_filt_vol_comscript =
        FileType::construct_instance(false, false, Some("reducefiltvol"), Some(".com"));
    let series_watcher_comscript = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str(
            ProcessName::SERIES_WATCHER
                .get_text()
                .unwrap_or("")
                .to_string(),
        ),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.com),
    ]));
    let distortion_corrected_stack_old =
        FileType::construct_image_file_instance(true, true, Some(""), Some(".dcst"));
    let distortion_corrected_stack = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.dcst),
    ]));
    let autofidseed_dir =
        FileType::construct_instance(false, true, Some("autofidseed"), Some(".dir"));
    let batch_run_tomo_project = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.ebt),
    ]));
    let piece_shifts = FileType::construct_instance(true, true, Some(""), Some(".ecd"));
    let manual_replacement_model =
        FileType::construct_instance(true, true, Some(""), Some(".erase"));
    let fiducial_model = FileType::construct_instance(true, true, Some(""), Some(".fid"));
    let ccd_eraser_beads_input_model =
        FileType::construct_instance(true, true, Some("_erase"), Some(".fid"));
    let fiducial_no_gaps_model =
        FileType::construct_instance(true, true, Some("_nogaps"), Some(".fid"));
    let fiducial_patch_tracking_model =
        FileType::construct_instance(true, true, Some("_pt"), Some(".fid"));
    let flatten_tool_output_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some(""),
        Some(".flat"),
        Some(imod_manager::FLATTEN_TOOL_OUTPUT_KEY),
    );
    let flatten_tool_output = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.flat),
        ]),
        Some(imod_manager::FLATTEN_TOOL_OUTPUT_KEY),
    );
    let exclude_views_info =
        FileType::construct_versioned_instance(true, true, Some("_cutviews"), Some(".info"));
    let nad_test_input_old = FileType::construct_imod_image_file_instance_in_subdirectory(
        false,
        false,
        Some("test"),
        Some(".input"),
        Some(imod_manager::TEST_VOLUME_KEY),
        None,
    );
    let nad_test_input = FileType::construct_instance_directory(
        Some(DirectoryType::nad_subdir()),
        Some(vec![
            PatternElement::Str("test".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.input),
        ]),
        Some(imod_manager::TEST_VOLUME_KEY),
    );
    let join_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some(""),
        Some(".join"),
        Some(imod_manager::JOIN_KEY),
    );
    let join = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.join),
        ]),
        Some(imod_manager::JOIN_KEY),
    );
    let modeled_join_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some("_modeled"),
        Some(".join"),
        Some(imod_manager::MODELED_JOIN_KEY),
    );
    let modeled_join = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::Str("_modeled".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.join),
        ]),
        Some(imod_manager::MODELED_JOIN_KEY),
    );
    let trial_join_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some("_trial"),
        Some(".join"),
        Some(imod_manager::TRIAL_JOIN_KEY),
    );
    let trial_join = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::Str("_trial".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.join),
        ]),
        Some(imod_manager::TRIAL_JOIN_KEY),
    );
    let batch_run_tomo_log = FileType::construct_instance(true, false, Some(""), Some(".log"));
    let alt_tomo_setup_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("alttomosetup".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let batch_run_tomo_dataset_log =
        FileType::construct_instance(false, false, Some("batchruntomo"), Some(".log"));
    let ctf_3d_finish_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("ctf3d-finish".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let ctf_3d_setup_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("ctf3dsetup".to_string()),
        PatternElement::Variable(Variable::axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let eraser_log = FileType::construct_instance(false, true, Some("eraser"), Some(".log"));
    let tilt_align_log = FileType::construct_instance(false, true, Some("align"), Some(".log"));
    let gpu_test_log = FileType::construct_instance(false, false, Some("gputest"), Some(".log"));
    let preblend_log = FileType::construct_instance(false, true, Some("preblend"), Some(".log"));
    let project_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset()),
        PatternElement::Str("_project".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let series_watcher_brt_row_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("swbrt_".to_string()),
        PatternElement::Variable(Variable::dataset()),
        PatternElement::Str(".".to_string()),
        PatternElement::Variable(Variable::any_integer()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let align_angles_log =
        FileType::construct_instance(false, true, Some("taAngles"), Some(".log"));
    let align_error_log = FileType::construct_instance(false, true, Some("taError"), Some(".log"));
    let align_robust_log =
        FileType::construct_instance(false, true, Some("taRobust"), Some(".log"));
    let align_solution_log =
        FileType::construct_instance(false, true, Some("taSolution"), Some(".log"));
    let cross_correlation_log =
        FileType::construct_instance(false, true, Some("xcorr"), Some(".log"));
    let align_frames_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset()),
        PatternElement::Str("_af".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let restrict_align_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("restrictalign".to_string()),
        PatternElement::Variable(Variable::axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let subtomo_setup_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("subtomosetup".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let ctf_correction_log =
        FileType::construct_instance(false, true, Some("ctfcorrection"), Some(".log"));
    let gold_eraser_log =
        FileType::construct_instance(false, true, Some("golderaser"), Some(".log"));
    let mtf_filter_log = FileType::construct_instance(false, true, Some("mtffilter"), Some(".log"));
    let reduce_filt_vol_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("reducefiltvol".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let series_watcher_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str(
            ProcessName::SERIES_WATCHER
                .get_text()
                .unwrap_or("")
                .to_string(),
        ),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let mtf_filter_mdoc = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::Variable(Variable::orig_raw_image_extension()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.mdoc),
    ]));
    let find_beads_3d_output_model =
        FileType::construct_instance(true, true, Some("_3dfind"), Some(".mod"));
    let autofidseed_boundary_model =
        FileType::construct_instance(true, true, Some("_afsbound"), Some(".mod"));
    let auto_align_boundary_model =
        FileType::construct_instance(true, false, Some("_bound"), Some(".mod"));
    let smoothing_assessment_output_model = FileType::construct_imod_instance(
        true,
        true,
        Some("_checkflat"),
        Some(".mod"),
        Some(imod_manager::SMOOTHING_ASSESSMENT_KEY),
    );
    let clustered_elongated_model = FileType::construct_imod_instance_in_subdirectory(
        false,
        false,
        Some("clusterElong"),
        Some(".mod"),
        None,
        Some(autofidseed_dir.clone()),
    );
    let flatten_warp_input_model =
        FileType::construct_instance(true, false, Some("_flat"), Some(".mod"));
    let patch_vector_model = FileType::construct_imod_instance(
        false,
        false,
        Some("patch_vector"),
        Some(".mod"),
        Some(imod_manager::PATCH_VECTOR_MODEL_KEY),
    );
    let patch_vector_ccc_model = FileType::construct_imod_instance(
        false,
        false,
        Some("patch_vector_ccc"),
        Some(".mod"),
        Some(imod_manager::PATCH_VECTOR_CCC_MODEL_KEY),
    );
    let patch_tracking_boundary_model =
        FileType::construct_instance(true, true, Some("_ptbound"), Some(".mod"));
    let batch_run_tomo_boundary_model =
        FileType::construct_instance(true, false, Some("_rawbound"), Some(".mod"));
    let tomopitch_model =
        FileType::construct_instance(false, true, Some("tomopitch"), Some(".mod"));
    let aligned_stack_mrc_old = FileType::construct_imod_image_file_instance(
        true,
        true,
        Some("_ali"),
        Some(".mrc"),
        Some(imod_manager::ALIGNED_STACK_KEY),
    );
    let aligned_stack_mrc = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_ali".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.mrc),
        ]),
        Some(imod_manager::ALIGNED_STACK_KEY),
    );
    let preblend_output_mrc_old = FileType::construct_imod_image_file_instance(
        true,
        true,
        Some("_preblend"),
        Some(".mrc"),
        Some(imod_manager::PREBLEND_KEY),
    );
    let preblend_output_mrc = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_preblend".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.mrc),
        ]),
        Some(imod_manager::PREBLEND_KEY),
    );
    let mutlifilt_exact_object_sizes_output_template_old =
        FileType::construct_template_image_file_instance_descr(
            true,
            true,
            Some("_efos"),
            Some(".mrc"),
            Some(DynamicLocation::AfterTypeString),
            Some("Exact Filter Trials"),
        );
    let mutlifilt_exact_object_sizes_output_template =
        FileType::construct_instance_pattern_key_descr(
            Some(vec![
                PatternElement::Variable(Variable::dataset_and_axis()),
                PatternElement::Str("_efos".to_string()),
                PatternElement::Variable(Variable::two_digit_integer()),
                PatternElement::ExtensionMarker(ExtensionMarker::Image),
                PatternElement::Extension(&extension::CLASS.mrc),
            ]),
            None,
            Some("Exact Filter Trials"),
        );
    let mutlifilt_gaussian_output_template_old =
        FileType::construct_template_image_file_instance_middle_piece(
            true,
            true,
            Some("_gfc"),
            Some("-f"),
            Some(".mrc"),
            Some(DynamicLocation::AfterTypeStringAndAfterMiddlePiece),
            Some("Standard Gaussian Filter Trials"),
        );
    let mutlifilt_gaussian_output_template = FileType::construct_instance_pattern_key_descr(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_gfc".to_string()),
            PatternElement::Variable(Variable::precision_three_float()),
            PatternElement::Str("-f".to_string()),
            PatternElement::Variable(Variable::precision_three_float()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.mrc),
        ]),
        None,
        Some("Standard Gaussian Filter Trials"),
    );
    let mutlifilt_hamming_like_starts_output_template_old =
        FileType::construct_template_image_file_instance_descr(
            true,
            true,
            Some("_hlfs"),
            Some(".mrc"),
            Some(DynamicLocation::AfterTypeString),
            Some("Hamming-Like Filter Trials"),
        );
    let mutlifilt_hamming_like_starts_output_template =
        FileType::construct_instance_pattern_key_descr(
            Some(vec![
                PatternElement::Variable(Variable::dataset_and_axis()),
                PatternElement::Str("_hlfs".to_string()),
                PatternElement::Variable(Variable::precision_three_fraction()),
                PatternElement::ExtensionMarker(ExtensionMarker::Image),
                PatternElement::Extension(&extension::CLASS.mrc),
            ]),
            None,
            Some("Hamming-Like Filter Trials"),
        );
    let mutlifilt_fake_sirt_iterations_output_template_old =
        FileType::construct_template_image_file_instance_descr(
            true,
            true,
            Some("_slfi"),
            Some(".mrc"),
            Some(DynamicLocation::AfterTypeString),
            Some("SIRT-like Filter Trials"),
        );
    let mutlifilt_fake_sirt_iterations_output_template =
        FileType::construct_instance_pattern_key_descr(
            Some(vec![
                PatternElement::Variable(Variable::dataset_and_axis()),
                PatternElement::Str("_slfi".to_string()),
                PatternElement::Variable(Variable::two_digit_integer()),
                PatternElement::ExtensionMarker(ExtensionMarker::Image),
                PatternElement::Extension(&extension::CLASS.mrc),
            ]),
            None,
            Some("SIRT-like Filter Trials"),
        );
    let test_nad = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Str("test.K".to_string()),
            PatternElement::Variable(Variable::float()),
            PatternElement::Str("-".to_string()),
            PatternElement::Variable(Variable::three_digit_integer()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.mrc),
        ]),
        Some(imod_manager::TRIMMED_VOLUME_KEY),
    );
    let anisotropic_diffusion_output_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some(""),
        Some(".nad"),
        Some(imod_manager::ANISOTROPIC_DIFFUSION_VOLUME_KEY),
    );
    let anisotropic_diffusion_output = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.nad),
        ]),
        Some(imod_manager::ANISOTROPIC_DIFFUSION_VOLUME_KEY),
    );
    let piece_list = FileType::construct_instance(true, true, Some(""), Some(".pl"));
    let prealigned_stack_old = FileType::construct_imod_image_file_instance(
        true,
        true,
        Some(""),
        Some(".preali"),
        Some(imod_manager::COARSE_ALIGNED_KEY),
    );
    let prealigned_stack = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.preali),
        ]),
        Some(imod_manager::COARSE_ALIGNED_KEY),
    );
    let pre_transformation_list =
        FileType::construct_instance(true, true, Some(""), Some(".prexf"));
    let pre_xg = FileType::construct_imod_instance(true, true, Some(""), Some(".prexg"), None);
    let matlab_param_file = FileType::construct_instance(true, false, Some(""), Some(".prm"));
    let raw_tilt_angles = FileType::construct_instance(true, true, Some(""), Some(".rawtlt"));
    let trim_vol_output_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some(""),
        Some(".rec"),
        Some(imod_manager::TRIMMED_VOLUME_KEY),
    );
    let trim_vol_output = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::TRIMMED_VOLUME_KEY),
    );
    let processchunks_rec = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str(CHUNK_NUMBER_DIVIDER.to_string()),
            PatternElement::Variable(Variable::three_digit_integer()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::TRIMMED_VOLUME_KEY),
    );
    let processchunks_mrc = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str(CHUNK_NUMBER_DIVIDER.to_string()),
            PatternElement::Variable(Variable::one_digit_integer()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.mrc),
        ]),
        Some(imod_manager::SUBTOMO_SETUP_KEY),
    );
    let processchunks_vol_mrc = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str(CHUNK_NUMBER_DIVIDER_VOL.to_string()),
            PatternElement::Variable(Variable::one_digit_integer()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.mrc),
        ]),
        Some(imod_manager::SUBTOMO_SETUP_KEY),
    );
    let tilt_3d_find_output_old = FileType::construct_imod_image_file_instance(
        true,
        true,
        Some("_3dfind"),
        Some(".rec"),
        Some(imod_manager::FULL_VOLUME_3D_FIND_KEY),
    );
    let tilt_3d_find_output = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_3dfind".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::FULL_VOLUME_3D_FIND_KEY),
    );
    let bottom_sample_old =
        FileType::construct_image_file_instance(false, true, Some("bot"), Some(".rec"));
    let bottom_sample = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("bot".to_string()),
        PatternElement::Variable(Variable::axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.rec),
    ]));
    let cryo_position_output_old =
        FileType::construct_image_file_instance(true, true, Some("_cpos"), Some(".rec"));
    let cryo_position_output = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::Str("_cpos".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.rec),
    ]));
    let tilt_output_dual_old =
        FileType::construct_image_file_instance(true, true, Some(""), Some(".rec"));
    let tilt_output_single_old =
        FileType::construct_image_file_instance(true, true, Some("_full"), Some(".rec"));
    let tilt_output_single = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::Str("_full".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.rec),
    ]));
    let tilt_output_dual = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.rec),
    ]));
    let tilt_output_old = FileType::construct_different_dual_single_image_file_instance(
        tilt_output_single_old.clone(),
        tilt_output_dual_old.clone(),
        Some(imod_manager::FULL_VOLUME_KEY),
        Some("the tomogram"),
    );
    let tilt_output = FileType::construct_instance_two_patterns(
        tilt_output_dual.file_name_pattern.clone(),
        tilt_output_single.file_name_pattern.clone(),
        Some(imod_manager::FULL_VOLUME_KEY),
        Some("the tomogram"),
    );
    let ctf_3d_output = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_3dctf".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::CTF_3D_KEY),
    );
    let flatten_output_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some("_flat"),
        Some(".rec"),
        Some(imod_manager::FLAT_VOLUME_KEY),
    );
    let flatten_output = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::Str("_flat".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::FLAT_VOLUME_KEY),
    );
    let middle_sample_old =
        FileType::construct_image_file_instance(false, true, Some("mid"), Some(".rec"));
    let middle_sample = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("mid".to_string()),
        PatternElement::Variable(Variable::axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.rec),
    ]));
    let combined_volume_old = FileType::construct_imod_image_file_instance(
        false,
        false,
        Some("sum"),
        Some(".rec"),
        Some(imod_manager::COMBINED_TOMOGRAM_KEY),
    );
    let combined_volume = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Str("sum".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::COMBINED_TOMOGRAM_KEY),
    );
    let top_sample_old =
        FileType::construct_image_file_instance(false, true, Some("top"), Some(".rec"));
    let top_sample = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Str("top".to_string()),
        PatternElement::Variable(Variable::axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.rec),
    ]));
    let join_sample_averages_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some(""),
        Some(".sampavg"),
        Some(imod_manager::JOIN_SAMPLE_AVERAGES_KEY),
    );
    let join_sample_averages = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.sampavg),
        ]),
        Some(imod_manager::JOIN_SAMPLE_AVERAGES_KEY),
    );
    let join_sample_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some(""),
        Some(".sample"),
        Some(imod_manager::JOIN_SAMPLES_KEY),
    );
    let join_sample = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.sample),
        ]),
        Some(imod_manager::JOIN_SAMPLES_KEY),
    );
    let seed_model = FileType::construct_instance(true, true, Some(""), Some(".seed"));
    let sirt_scaled_output_template_old = FileType::construct_derived_template_image_file_instance(
        Some(tilt_output_old.clone()),
        Some(".sint"),
        Some(imod_manager::SIRT_KEY),
        Some(DynamicLocation::AfterExtension),
    );
    let sirt_scaled_output_template = FileType::construct_instance_joined(
        tilt_output.file_name_pattern.clone(),
        tilt_output.single_axis_file_name_pattern.clone(),
        Some(vec![
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.sint),
            PatternElement::Variable(Variable::two_digit_integer()),
        ]),
        Some(imod_manager::SIRT_KEY),
    );
    let sirt_subarea_scaled_output_template_old = FileType::construct_template_image_file_instance(
        true,
        true,
        Some("_sub"),
        Some(".sint"),
        Some(DynamicLocation::AfterExtension),
    );
    let sirt_subarea_scaled_output_template = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::Str("_sub".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.sint),
        PatternElement::Variable(Variable::two_digit_integer()),
    ]));
    let squeeze_vol_output_old = FileType::construct_imod_image_file_instance(
        true,
        false,
        Some(""),
        Some(".sqz"),
        Some(imod_manager::SQUEEZED_VOLUME_KEY),
    );
    let squeeze_vol_output = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.sqz),
        ]),
        Some(imod_manager::SQUEEZED_VOLUME_KEY),
    );
    let sirt_output_template_old = FileType::construct_derived_template_image_file_instance(
        Some(tilt_output_old.clone()),
        Some(".srec"),
        Some(imod_manager::SIRT_KEY),
        Some(DynamicLocation::AfterExtension),
    );
    let sirt_output_template = FileType::construct_instance_joined(
        tilt_output.file_name_pattern.clone(),
        tilt_output.single_axis_file_name_pattern.clone(),
        Some(vec![
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.srec),
            PatternElement::Variable(Variable::two_digit_integer()),
        ]),
        Some(imod_manager::SIRT_KEY),
    );
    let sirt_subarea_output_template_old = FileType::construct_template_image_file_instance(
        true,
        true,
        Some("_sub"),
        Some(".srec"),
        Some(DynamicLocation::AfterExtension),
    );
    let sirt_subarea_output_template = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::Str("_sub".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.srec),
        PatternElement::Variable(Variable::two_digit_integer()),
    ]));
    // TODO(unit): needs etomo/logic/DatasetTool.java - Java `RAW_STACK_OLD` is
    // `FileType.constructTwoImodRawImageFileInstance(...)` with `DatasetTool.STANDARD_DATASET_EXT`, and that
    // constant has no module.
    let raw_stack = FileType::construct_instance_two_keys(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::ExtensionMarker(ExtensionMarker::InputImage),
            PatternElement::Extension(&extension::CLASS.st),
        ]),
        Some(imod_manager::RAW_STACK_KEY),
        Some(imod_manager::PREVIEW_KEY),
        None,
    );
    let stats_log_old = FileType::construct_instance(true, true, Some(".st_stats"), Some(".log"));
    // TODO(unit): needs etomo/logic/DatasetTool.java - Java `FIXED_XRAYS_STACK_OLD` is
    // `FileType.constructImodImageFileInstance(...)` with `DatasetTool.STANDARD_DATASET_EXT`, and that
    // constant has no module.
    let processchunks_log = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::Str(CHUNK_NUMBER_DIVIDER.to_string()),
        PatternElement::Variable(Variable::three_digit_integer()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.log),
    ]));
    let stats_log = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::FileType(raw_stack.clone()),
            PatternElement::Str("_stats".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Generic),
            PatternElement::Extension(&extension::CLASS.log),
        ]),
        Some(imod_manager::JOIN_KEY),
    );
    let fixed_xrays_stack = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_fixed".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::InputImage),
            PatternElement::Extension(&extension::CLASS.st),
        ]),
        Some(imod_manager::ERASED_STACK_KEY),
    );
    let fixed_stats_log_old =
        FileType::construct_instance(true, true, Some("_fixed.st_stats"), Some(".log"));
    let fixed_stats_log = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::FileType(fixed_xrays_stack.clone()),
            PatternElement::Str("_stats".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Generic),
            PatternElement::Extension(&extension::CLASS.log),
        ]),
        Some(imod_manager::JOIN_KEY),
    );
    // TODO(unit): needs etomo/logic/DatasetTool.java - Java `ORIGINAL_RAW_STACK_OLD` is
    // `FileType.constructRawImageFileInstance(...)` with `DatasetTool.STANDARD_DATASET_EXT`, and that
    // constant has no module.
    let original_raw_stack = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::Str("_orig".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::InputImage),
        PatternElement::Extension(&extension::CLASS.st),
    ]));
    let full_vsr = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::Str("_full".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.vsr),
        PatternElement::Variable(Variable::two_digit_integer()),
    ]));
    let sub_vsr = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::Str("_sub".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::Image),
        PatternElement::Extension(&extension::CLASS.vsr),
        PatternElement::Variable(Variable::two_digit_integer()),
    ]));
    let tilt_angles = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset_and_axis()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.tlt),
    ]));
    let warp_xg = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset()),
        PatternElement::ExtensionMarker(ExtensionMarker::Generic),
        PatternElement::Extension(&extension::CLASS.warpxg),
    ]));
    let edge_functions_x = FileType::construct_instance(true, true, Some(""), Some(".xef"));
    let local_transformation_list =
        FileType::construct_instance(true, false, Some(""), Some(".xf"));
    let auto_local_transformation_list =
        FileType::construct_instance(true, false, Some("_auto"), Some(".xf"));
    let empty_local_transformation_list =
        FileType::construct_instance(true, false, Some("_empty"), Some(".xf"));
    let midas_local_transformation_list =
        FileType::construct_instance(true, false, Some("_midas"), Some(".xf"));
    let global_transformation_list =
        FileType::construct_instance(true, false, Some(""), Some(".xg"));
    let alt_stack_rootname_even_file = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset()),
        PatternElement::Str("_even".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::InputImage),
        PatternElement::Extension(&extension::CLASS.st),
    ]));
    let alt_stack_rootname_odd_file = FileType::construct_instance_pattern(Some(vec![
        PatternElement::Variable(Variable::dataset()),
        PatternElement::Str("_odd".to_string()),
        PatternElement::ExtensionMarker(ExtensionMarker::InputImage),
        PatternElement::Extension(&extension::CLASS.st),
    ]));
    let alt_stack_even_tomogram = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_even".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY),
    );
    let alt_stack_odd_tomogram = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_odd".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY),
    );
    let alt_stack_even_full_tomogram = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_even_full".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY),
    );
    let alt_stack_odd_full_tomogram = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::Str("_odd_full".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY),
    );
    let alt_stack_tomogram = FileType::construct_instance_pattern_key(
        Some(vec![
            PatternElement::Variable(Variable::dataset_and_axis()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.rec),
        ]),
        Some(imod_manager::ALT_TOMO_SETUP_TOMOGRAM_KEY),
    );
    let reduce_filt_vol_output_file = FileType::construct_instance_pattern_key_descr(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::Str("_red".to_string()),
            PatternElement::Variable(Variable::precision_two_float()),
            PatternElement::Str("filt".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.mrc),
        ]),
        Some(imod_manager::REDUCED_FILTERED_VOLUME_KEY),
        Some("Reducefiltvol output image file"),
    );
    let flatten_reduce_filt_vol_file = FileType::construct_instance_pattern_key_descr(
        Some(vec![
            PatternElement::Variable(Variable::dataset()),
            PatternElement::Str("_red".to_string()),
            PatternElement::Variable(Variable::precision_two_float()),
            PatternElement::Str("filt".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Image),
            PatternElement::Extension(&extension::CLASS.mrc),
        ]),
        Some(imod_manager::FLATTEN_REDUCE_FILT_VOL_KEY),
        Some("Flatten input file for reducefiltvol"),
    );
    ClassStatics {
        orig_coms_dir,
        fiducial_3d_model,
        series_watcher_project,
        batch_run_tomo_global_autodoc,
        default_batch_run_tomo_autodoc,
        local_batch_directive_file,
        local_scope_template,
        local_system_template,
        local_user_template,
        aligned_stack_old,
        aligned_stack,
        newst_or_blend_3d_find_output_old,
        newst_or_blend_3d_find_output,
        ctf_corrected_stack_old,
        ctf_corrected_stack,
        erased_beads_stack_old,
        erased_beads_stack,
        mtf_filtered_stack_old,
        mtf_filtered_stack,
        transformed_refining_model,
        xcorr_blend_output_old,
        xcorr_blend_output,
        check_file,
        series_watcher_check_file,
        align_comscript,
        autofidseed_comscript,
        batch_run_tomo_comscript,
        blend_comscript,
        copytomocoms_comscript,
        cryo_position_comscript,
        ctf_3d_setup_comscript,
        ctf_correction_comscript,
        find_beads_3d_comscript,
        flatten_comscript,
        flatten_tool_comscript,
        gold_eraser_comscript,
        multifilt_setup_comscript,
        mtf_filter_comscript,
        newst_comscript,
        preblend_comscript,
        sirtsetup_comscript,
        sloppy_blend_comscript,
        tilt_comscript,
        tilt_for_sirt_comscript,
        tilt_for_pos_sample_comscript,
        track_comscript,
        track_adjusted_comscript,
        track_orig_comscript,
        cross_correlation_comscript,
        patch_tracking_comscript,
        directives_descr,
        join_warp_2_model_comscript,
        align_frames_output_comscript,
        subtomo_setup_comscript,
        alt_tomo_setup_comscript,
        restrict_align_comscript,
        reduce_filt_vol_comscript,
        series_watcher_comscript,
        distortion_corrected_stack_old,
        distortion_corrected_stack,
        autofidseed_dir,
        batch_run_tomo_project,
        piece_shifts,
        manual_replacement_model,
        fiducial_model,
        ccd_eraser_beads_input_model,
        fiducial_no_gaps_model,
        fiducial_patch_tracking_model,
        flatten_tool_output_old,
        flatten_tool_output,
        exclude_views_info,
        nad_test_input_old,
        nad_test_input,
        join_old,
        join,
        modeled_join_old,
        modeled_join,
        trial_join_old,
        trial_join,
        batch_run_tomo_log,
        alt_tomo_setup_log,
        batch_run_tomo_dataset_log,
        ctf_3d_finish_log,
        ctf_3d_setup_log,
        eraser_log,
        tilt_align_log,
        gpu_test_log,
        preblend_log,
        project_log,
        series_watcher_brt_row_log,
        align_angles_log,
        align_error_log,
        align_robust_log,
        align_solution_log,
        cross_correlation_log,
        align_frames_log,
        restrict_align_log,
        subtomo_setup_log,
        ctf_correction_log,
        gold_eraser_log,
        mtf_filter_log,
        reduce_filt_vol_log,
        series_watcher_log,
        mtf_filter_mdoc,
        find_beads_3d_output_model,
        autofidseed_boundary_model,
        auto_align_boundary_model,
        smoothing_assessment_output_model,
        clustered_elongated_model,
        flatten_warp_input_model,
        patch_vector_model,
        patch_vector_ccc_model,
        patch_tracking_boundary_model,
        batch_run_tomo_boundary_model,
        tomopitch_model,
        aligned_stack_mrc_old,
        aligned_stack_mrc,
        preblend_output_mrc_old,
        preblend_output_mrc,
        mutlifilt_exact_object_sizes_output_template_old,
        mutlifilt_exact_object_sizes_output_template,
        mutlifilt_gaussian_output_template_old,
        mutlifilt_gaussian_output_template,
        mutlifilt_hamming_like_starts_output_template_old,
        mutlifilt_hamming_like_starts_output_template,
        mutlifilt_fake_sirt_iterations_output_template_old,
        mutlifilt_fake_sirt_iterations_output_template,
        test_nad,
        anisotropic_diffusion_output_old,
        anisotropic_diffusion_output,
        piece_list,
        prealigned_stack_old,
        prealigned_stack,
        pre_transformation_list,
        pre_xg,
        matlab_param_file,
        raw_tilt_angles,
        trim_vol_output_old,
        trim_vol_output,
        processchunks_rec,
        processchunks_mrc,
        processchunks_vol_mrc,
        tilt_3d_find_output_old,
        tilt_3d_find_output,
        bottom_sample_old,
        bottom_sample,
        cryo_position_output_old,
        cryo_position_output,
        tilt_output_dual_old,
        tilt_output_single_old,
        tilt_output_single,
        tilt_output_dual,
        tilt_output_old,
        tilt_output,
        ctf_3d_output,
        flatten_output_old,
        flatten_output,
        middle_sample_old,
        middle_sample,
        combined_volume_old,
        combined_volume,
        top_sample_old,
        top_sample,
        join_sample_averages_old,
        join_sample_averages,
        join_sample_old,
        join_sample,
        seed_model,
        sirt_scaled_output_template_old,
        sirt_scaled_output_template,
        sirt_subarea_scaled_output_template_old,
        sirt_subarea_scaled_output_template,
        squeeze_vol_output_old,
        squeeze_vol_output,
        sirt_output_template_old,
        sirt_output_template,
        sirt_subarea_output_template_old,
        sirt_subarea_output_template,
        raw_stack,
        stats_log_old,
        processchunks_log,
        stats_log,
        fixed_xrays_stack,
        fixed_stats_log_old,
        fixed_stats_log,
        original_raw_stack,
        full_vsr,
        sub_vsr,
        tilt_angles,
        warp_xg,
        edge_functions_x,
        local_transformation_list,
        auto_local_transformation_list,
        empty_local_transformation_list,
        midas_local_transformation_list,
        global_transformation_list,
        alt_stack_rootname_even_file,
        alt_stack_rootname_odd_file,
        alt_stack_even_tomogram,
        alt_stack_odd_tomogram,
        alt_stack_even_full_tomogram,
        alt_stack_odd_full_tomogram,
        alt_stack_tomogram,
        reduce_filt_vol_output_file,
        flatten_reduce_filt_vol_file,
    }
});

/// Java `FileType`.
#[derive(Debug)]
pub struct FileType {
    /// The `FileKey` superclass state.
    file_key: FileKey,
    /// Java field `usesDataset`.
    uses_dataset: bool,
    /// Java field `usesAxisID`.
    uses_axis_id: bool,
    /// Java field `typeString`.
    type_string: Option<String>,
    /// Java field `extension`.
    extension: Option<String>,
    /// Java field `composite`.
    composite: bool,
    /// Java field `inSubdirectory`.
    in_subdirectory: bool,
    /// Java field `subFileType`.
    sub_file_type: Option<Arc<FileType>>,
    /// Java field `singleFileType`.
    single_file_type: Option<Arc<FileType>>,
    /// Java field `dualFileType`.
    dual_file_type: Option<Arc<FileType>>,
    /// Java field `unnamed`.
    unnamed: bool,
    /// Java field `template`.
    template: bool,
    /// Java field `inImodSubdirectory`.
    in_imod_subdirectory: Option<String>,
    /// Java field `subdir`.
    subdir: Option<Arc<FileType>>,
    /// Java field `versioned`.
    versioned: bool,
    /// Java field `templateOnlyMiddlePiece`.
    template_only_middle_piece: Option<String>,
    /// Java field `dynamicLocation`.
    dynamic_location: Option<DynamicLocation>,
    /// Java field `imageFile`.
    image_file: bool,
    /// Java field `expandedImageExtensionSet`.  For raw images which can be tif files.
    expanded_image_extension_set: bool,
    /// Java field `fileNamePattern`.
    file_name_pattern: Option<Vec<PatternElement>>,
    /// Java field `directory`.
    directory: Option<Arc<DirectoryType>>,
    /// Java field `singleAxisFileNamePattern`.
    single_axis_file_name_pattern: Option<Vec<PatternElement>>,
    /// Java field `parentFileType`, which
    /// `constructDifferentDualSingleImageFileInstance` writes back into its children.
    parent_file_type: Mutex<Weak<FileType>>,
}

impl std::ops::Deref for FileType {
    type Target = FileKey;
    fn deref(&self) -> &FileKey {
        &self.file_key
    }
}

impl FileType {
    /// Java `FileType()`.  Creates an empty instance which is not added to the
    /// collection.  This constructor should not be used, and exists only to allow the
    /// class to be inherited (see `etomo.plugin.demo`).
    pub fn new_empty() -> FileType {
        let instance = FileType {
            file_key: FileKey::new_with_descr(None, None, None),
            uses_dataset: false,
            uses_axis_id: false,
            type_string: None,
            extension: None,
            composite: false,
            in_subdirectory: false,
            sub_file_type: None,
            single_file_type: None,
            dual_file_type: None,
            unnamed: true,
            template: false,
            in_imod_subdirectory: None,
            subdir: None,
            versioned: false,
            template_only_middle_piece: None,
            dynamic_location: None,
            image_file: false,
            expanded_image_extension_set: false,
            file_name_pattern: None,
            directory: None,
            single_axis_file_name_pattern: None,
            parent_file_type: Mutex::new(Weak::new()),
        };
        // `if (!unnamed) { namedFileTypeList.add(this); }` - unnamed is true here.
        instance
    }

    /// Java `FileType(boolean, boolean, String, String, String, String, String, boolean,
    /// boolean, FileType, FileType, FileType, boolean, boolean, String, FileType,
    /// boolean, String, DynamicLocation, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        imod_manager_key: Option<&str>,
        imod_manager_key2: Option<&str>,
        descr: Option<&str>,
        composite: bool,
        in_subdirectory: bool,
        sub_file_type: Option<Arc<FileType>>,
        single_file_type: Option<Arc<FileType>>,
        dual_file_type: Option<Arc<FileType>>,
        unnamed: bool,
        template: bool,
        in_imod_subdirectory: Option<&str>,
        subdir: Option<Arc<FileType>>,
        versioned: bool,
        template_only_middle_piece: Option<&str>,
        dynamic_location: Option<DynamicLocation>,
        image_file: bool,
        expanded_image_extension_set: bool,
    ) -> Arc<FileType> {
        let instance = Arc::new(FileType {
            file_key: FileKey::new_with_descr(imod_manager_key, imod_manager_key2, descr),
            uses_dataset,
            uses_axis_id,
            type_string: type_string.map(|value| value.to_string()),
            extension: extension.map(|value| value.to_string()),
            composite,
            in_subdirectory,
            sub_file_type,
            single_file_type,
            dual_file_type,
            unnamed,
            template,
            in_imod_subdirectory: in_imod_subdirectory.map(|value| value.to_string()),
            subdir,
            versioned,
            template_only_middle_piece: template_only_middle_piece.map(|value| value.to_string()),
            dynamic_location,
            image_file,
            expanded_image_extension_set,
            file_name_pattern: None,
            directory: None,
            single_axis_file_name_pattern: None,
            parent_file_type: Mutex::new(Weak::new()),
        });
        if !unnamed {
            NAMED_FILE_TYPE_LIST.lock().unwrap().push(instance.clone());
        }
        instance
    }

    /// `FileType(DirectoryType, Object[], Object[], String, String, String)` as a
    /// subclass `super(...)` call: `DirectoryType` and `NumberedFileType` both use it.
    pub(crate) fn new_with_pattern_for_subclass(
        directory: Option<Arc<DirectoryType>>,
        file_name_pattern: Option<Vec<PatternElement>>,
        single_axis_file_name_pattern: Option<Vec<PatternElement>>,
        imod_manager_key: Option<&str>,
        imod_manager_key2: Option<&str>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new_with_pattern(
            directory,
            file_name_pattern,
            single_axis_file_name_pattern,
            imod_manager_key,
            imod_manager_key2,
            descr,
        )
    }

    /// Java `FileType(DirectoryType, Object[], Object[], String, String, String)`.  New
    /// file types using a generic pattern.
    fn new_with_pattern(
        directory: Option<Arc<DirectoryType>>,
        file_name_pattern: Option<Vec<PatternElement>>,
        single_axis_file_name_pattern: Option<Vec<PatternElement>>,
        imod_manager_key: Option<&str>,
        imod_manager_key2: Option<&str>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        let instance = Arc::new(FileType {
            file_key: FileKey::new_with_descr(imod_manager_key, imod_manager_key2, descr),
            directory,
            file_name_pattern,
            single_axis_file_name_pattern,
            //
            uses_dataset: false,
            uses_axis_id: false,
            type_string: None,
            extension: None,
            composite: false,
            in_subdirectory: false,
            sub_file_type: None,
            single_file_type: None,
            dual_file_type: None,
            unnamed: true,
            template: false,
            in_imod_subdirectory: None,
            subdir: None,
            versioned: false,
            template_only_middle_piece: None,
            dynamic_location: None,
            image_file: false,
            expanded_image_extension_set: false,
            parent_file_type: Mutex::new(Weak::new()),
        });
        // `if (!unnamed) { namedFileTypeList.add(this); }` - unnamed is true here.
        instance
    }

    /// Java `constructInstance(Object[], String, String)`.
    fn construct_instance_pattern_key_descr(
        file_name_pattern: Option<Vec<PatternElement>>,
        imod_manager_key: Option<&str>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new_with_pattern(None, file_name_pattern, None, imod_manager_key, None, descr)
    }

    /// Java `constructInstance(Object[], String)`.
    fn construct_instance_pattern_key(
        file_name_pattern: Option<Vec<PatternElement>>,
        imod_manager_key: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new_with_pattern(None, file_name_pattern, None, imod_manager_key, None, None)
    }

    /// Java `constructInstance(Object[], Object[], String, String)`.
    fn construct_instance_two_patterns(
        file_name_pattern: Option<Vec<PatternElement>>,
        single_axis_file_name_pattern: Option<Vec<PatternElement>>,
        imod_manager_key: Option<&str>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new_with_pattern(
            None,
            file_name_pattern,
            single_axis_file_name_pattern,
            imod_manager_key,
            None,
            descr,
        )
    }

    /// Java `constructInstance(Object[], Object[], Object[], String)`.
    fn construct_instance_joined(
        source_file_name_pattern: Option<Vec<PatternElement>>,
        source_single_axis_file_name_pattern: Option<Vec<PatternElement>>,
        suffix_pattern: Option<Vec<PatternElement>>,
        imod_manager_key: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new_with_pattern(
            None,
            FileType::join_patterns(source_file_name_pattern, suffix_pattern.clone()),
            FileType::join_patterns(source_single_axis_file_name_pattern, suffix_pattern),
            imod_manager_key,
            None,
            None,
        )
    }

    /// Java `constructInstance(Object[])`.
    fn construct_instance_pattern(file_name_pattern: Option<Vec<PatternElement>>) -> Arc<FileType> {
        FileType::new_with_pattern(None, file_name_pattern, None, None, None, None)
    }

    /// Java `constructInstance(DirectoryType, Object[], String)`.
    fn construct_instance_directory(
        directory: Option<Arc<DirectoryType>>,
        file_name_pattern: Option<Vec<PatternElement>>,
        imod_manager_key: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new_with_pattern(
            directory,
            file_name_pattern,
            None,
            imod_manager_key,
            None,
            None,
        )
    }

    /// Java `constructInstance(Object[], String, String, String)`.
    fn construct_instance_two_keys(
        file_name_pattern: Option<Vec<PatternElement>>,
        imod_manager_key: Option<&str>,
        imod_manager_key2: Option<&str>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new_with_pattern(
            None,
            file_name_pattern,
            None,
            imod_manager_key,
            imod_manager_key2,
            descr,
        )
    }

    // old file types

    /// Java `constructVersionedInstance`.
    fn construct_versioned_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            true,
            None,
            None,
            false,
            false,
        )
    }

    /// Java `constructTemplateImageFileInstance(boolean, boolean, String, String,
    /// DynamicLocation)`.
    fn construct_template_image_file_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        dynamic_location: Option<DynamicLocation>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            true,
            None,
            None,
            false,
            None,
            dynamic_location,
            true,
            false,
        )
    }

    /// Java `constructTemplateImageFileInstance(boolean, boolean, String, String,
    /// DynamicLocation, String)`.
    fn construct_template_image_file_instance_descr(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        dynamic_location: Option<DynamicLocation>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            None,
            None,
            descr,
            false,
            false,
            None,
            None,
            None,
            false,
            true,
            None,
            None,
            false,
            None,
            dynamic_location,
            true,
            false,
        )
    }

    /// Java `constructTemplateImageFileInstance(boolean, boolean, String, String, String,
    /// DynamicLocation, String)`.
    #[allow(clippy::too_many_arguments)]
    fn construct_template_image_file_instance_middle_piece(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        template_only_middle_piece: Option<&str>,
        extension: Option<&str>,
        dynamic_location: Option<DynamicLocation>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            None,
            None,
            descr,
            false,
            false,
            None,
            None,
            None,
            false,
            true,
            None,
            None,
            false,
            template_only_middle_piece,
            dynamic_location,
            true,
            false,
        )
    }

    /// Java `constructDifferentDualSingleImageFileInstance`.  Get file types that are
    /// quite different in dual and single (BBa.rec and BBa_full.rec).
    fn construct_different_dual_single_image_file_instance(
        single_file_type: Arc<FileType>,
        dual_file_type: Arc<FileType>,
        imod_manager_key: Option<&str>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        let instance = FileType::new(
            false,
            false,
            None,
            None,
            imod_manager_key,
            None,
            descr,
            true,
            false,
            None,
            Some(single_file_type.clone()),
            Some(dual_file_type.clone()),
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            true,
            false,
        );
        // Child file types are not valid by themselves
        *single_file_type.parent_file_type.lock().unwrap() = Arc::downgrade(&instance);
        *dual_file_type.parent_file_type.lock().unwrap() = Arc::downgrade(&instance);
        instance
    }

    /// Java `constructInstance(boolean, boolean, String, String)`.
    pub fn construct_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            false,
            false,
        )
    }

    /// Java `constructImageFileInstance`.
    pub fn construct_image_file_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            true,
            false,
        )
    }

    /// Java `constructRawImageFileInstance`.
    pub fn construct_raw_image_file_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            true,
            true,
        )
    }

    /// Java `constructInstance(boolean, boolean, String, String, String)`.
    pub fn construct_instance_descr(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            None,
            None,
            descr,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            false,
            false,
        )
    }

    /// Java `constructDerivedTemplateImageFileInstance`.  For use when everything is
    /// coming from another file type, except the extension (BBa.srec, BBa_full.srec,
    /// BBa.sint, BBa_full.sint).
    pub fn construct_derived_template_image_file_instance(
        sub_file_type: Option<Arc<FileType>>,
        extension: Option<&str>,
        imod_manager_key: Option<&str>,
        dynamic_location: Option<DynamicLocation>,
    ) -> Arc<FileType> {
        FileType::new(
            false,
            false,
            None,
            extension,
            imod_manager_key,
            None,
            None,
            true,
            false,
            sub_file_type,
            None,
            None,
            false,
            true,
            None,
            None,
            false,
            None,
            dynamic_location,
            true,
            false,
        )
    }

    /// Java `constructImodInstance`.
    fn construct_imod_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        imod_manager_key: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            imod_manager_key,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            false,
            false,
        )
    }

    /// Java `constructImodImageFileInstance`.
    fn construct_imod_image_file_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        imod_manager_key: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            imod_manager_key,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            true,
            false,
        )
    }

    /// Java `constructImodInstanceInSubdirectory`.
    fn construct_imod_instance_in_subdirectory(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        imod_manager_key: Option<&str>,
        subdir: Option<Arc<FileType>>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            imod_manager_key,
            None,
            None,
            false,
            true,
            None,
            None,
            None,
            false,
            false,
            None,
            subdir,
            false,
            None,
            None,
            false,
            false,
        )
    }

    /// Java `constructImodImageFileInstanceInSubdirectory`.
    fn construct_imod_image_file_instance_in_subdirectory(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        imod_manager_key: Option<&str>,
        subdir: Option<Arc<FileType>>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            imod_manager_key,
            None,
            None,
            false,
            true,
            None,
            None,
            None,
            false,
            false,
            None,
            subdir,
            false,
            None,
            None,
            true,
            false,
        )
    }

    /// Java `constructTwoImodRawImageFileInstance`.
    fn construct_two_imod_raw_image_file_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        imod_manager_key: Option<&str>,
        imod_manager_key2: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            imod_manager_key,
            imod_manager_key2,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            true,
            true,
        )
    }

    /// Java `constructDescribedImodImageFileInstance`.
    fn construct_described_imod_image_file_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        imod_manager_key: Option<&str>,
        descr: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            imod_manager_key,
            None,
            descr,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            false,
            None,
            None,
            true,
            false,
        )
    }

    /// Java `constructUnamedImageFileInstance`.  Not added to the collection and can't
    /// be retrieved via `getInstance`.
    fn construct_unamed_image_file_instance(imod_manager_key: Option<&str>) -> Arc<FileType> {
        FileType::new(
            false,
            false,
            Some(""),
            Some(""),
            imod_manager_key,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            true,
            false,
            None,
            None,
            false,
            None,
            None,
            true,
            false,
        )
    }

    /// Java `constructUnamedImageFileInstanceInSubdirectory`.  Not added to the
    /// collection and can't be retrieved via `getInstance`.
    fn construct_unamed_image_file_instance_in_subdirectory(
        imod_manager_key: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            false,
            false,
            Some(""),
            Some(""),
            imod_manager_key,
            None,
            None,
            false,
            true,
            None,
            None,
            None,
            true,
            false,
            None,
            None,
            false,
            None,
            None,
            true,
            false,
        )
    }

    /// Java `constructIMODDirInstance`.
    pub fn construct_imod_dir_instance(
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
        imod_subdirectory: Option<&str>,
    ) -> Arc<FileType> {
        FileType::new(
            uses_dataset,
            uses_axis_id,
            type_string,
            extension,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            imod_subdirectory,
            None,
            false,
            None,
            None,
            false,
            false,
        )
    }

    /// Java `getInstance(AxisType, boolean, boolean, String, String)` (deprecated
    /// 2/1/2019).  Get a `FileType` instance from its name description.
    pub fn get_instance(
        axis_type: Option<AxisType>,
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
    ) -> Option<Arc<FileType>> {
        // Java touches `FileType` here, which runs the class's static initialisers and
        // so fills `namedFileTypeList`; forcing `CLASS` is that initialisation.
        LazyLock::force(&CLASS);
        let list = NAMED_FILE_TYPE_LIST.lock().unwrap();
        for file_type in list.iter() {
            if file_type.equals_name_description(
                axis_type,
                uses_dataset,
                uses_axis_id,
                type_string,
                extension,
            ) {
                let parent = file_type.parent_file_type.lock().unwrap().upgrade();
                if let Some(parent) = parent {
                    // This is a child file type which is not valid by itself
                    return Some(parent);
                }
                return Some(file_type.clone());
            }
        }
        None
    }

    // TODO(unit): needs etomo/BaseManager.java and etomo/type/BaseMetaData.java - Java
    // `getInstance(BaseManager, AxisID, boolean, boolean, String)` (deprecated) builds
    // its fixed pattern from `manager.getBaseMetaData().getName()` and its axis type
    // from `manager.getBaseMetaData().getAxisType()`.  The `equals` overload it calls,
    // `equals(AxisType, String, boolean, boolean, String, String)`, is translated below.

    /// Java `equals(AxisType, String, boolean, boolean, String, String)` (deprecated
    /// 2/2/2019).  Returns true if the file name can be matched to the fixedPattern,
    /// typeString, axisPattern, and extension.
    pub fn equals_file_name(
        &self,
        axis_type: Option<AxisType>,
        file_name: &str,
        uses_dataset: bool,
        uses_axis_id: bool,
        fixed_pattern: &str,
        axis_pattern: &str,
    ) -> bool {
        if self.composite {
            // Handle type files which are based on another file type but have their own
            // extension.
            if self.sub_file_type.is_some() && self.extension.is_some() {
                return self
                    .sub_file_type
                    .as_ref()
                    .unwrap()
                    .equals_file_name_extension(
                        axis_type,
                        file_name,
                        uses_dataset,
                        uses_axis_id,
                        fixed_pattern,
                        axis_pattern,
                        &java_util_regex_pattern_quote(self.extension.as_deref().unwrap()),
                    );
            }
            // Handle file types with single and dual file types instead of descriptions.
            return self.get_child_file_type(axis_type).equals_file_name(
                axis_type,
                file_name,
                uses_dataset,
                uses_axis_id,
                fixed_pattern,
                axis_pattern,
            );
        }
        uses_dataset == self.uses_dataset
            && uses_axis_id == self.uses_axis_id
            && java_lang_string_matches(
                file_name,
                &(fixed_pattern.to_string()
                    + &java_util_regex_pattern_quote(
                        self.type_string.as_deref().unwrap_or("null"),
                    )
                    + axis_pattern
                    + &java_util_regex_pattern_quote(self.extension.as_deref().unwrap_or("null"))),
            )
    }

    /// Java `equals(AxisType, String, boolean, boolean, String, String, String)`
    /// (deprecated 2/2/2019).  Returns true if the file name can be matched to the
    /// fixedPattern, typeString, axisPattern, and extensionPattern.
    #[allow(clippy::too_many_arguments)]
    pub fn equals_file_name_extension(
        &self,
        axis_type: Option<AxisType>,
        file_name: &str,
        uses_dataset: bool,
        uses_axis_id: bool,
        fixed_pattern: &str,
        axis_pattern: &str,
        extension_pattern: &str,
    ) -> bool {
        if self.composite {
            // Handle type files which are based on another file type but have their own
            // extension.
            if let Some(sub_file_type) = self.sub_file_type.as_ref() {
                return sub_file_type.equals_file_name_extension(
                    axis_type,
                    file_name,
                    uses_dataset,
                    uses_axis_id,
                    fixed_pattern,
                    axis_pattern,
                    extension_pattern,
                );
            }
            // Handle file types with single and dual file types instead of descriptions.
            return self
                .get_child_file_type(axis_type)
                .equals_file_name_extension(
                    axis_type,
                    file_name,
                    uses_dataset,
                    uses_axis_id,
                    fixed_pattern,
                    axis_pattern,
                    extension_pattern,
                );
        }
        uses_dataset == self.uses_dataset
            && uses_axis_id == self.uses_axis_id
            && java_lang_string_matches(
                file_name,
                &(fixed_pattern.to_string()
                    + &java_util_regex_pattern_quote(
                        self.type_string.as_deref().unwrap_or("null"),
                    )
                    + axis_pattern
                    + extension_pattern),
            )
    }

    /// Java `equals(AxisType, FileType)` (deprecated 2/2/2019).  Compares the member
    /// variables that make a file type unique: usesDataset, usesAxisID, typeString, and
    /// extension.
    pub fn equals_file_type(&self, axis_type: Option<AxisType>, file_type: &FileType) -> bool {
        self.equals_name_description(
            axis_type,
            file_type.uses_dataset,
            file_type.uses_axis_id,
            file_type.type_string.as_deref(),
            file_type.extension.as_deref(),
        )
    }

    /// Java `equals(AxisType, boolean, boolean, String, String)` (deprecated 2/2/2019).
    /// Returns true if the dataset, axis, type string and extension are equal.
    pub fn equals_name_description(
        &self,
        axis_type: Option<AxisType>,
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
        extension: Option<&str>,
    ) -> bool {
        if self.composite {
            // Handle type files which are based on another file type but have their own
            // extension.
            if self.sub_file_type.is_some() && self.extension.is_some() {
                return self.extension.as_deref() == extension
                    && self.sub_file_type.as_ref().unwrap().equals_type_string(
                        axis_type,
                        uses_dataset,
                        uses_axis_id,
                        type_string,
                    );
            }
            // Handle file types with single and dual file types instead of descriptions.
            return self.get_child_file_type(axis_type).equals_name_description(
                axis_type,
                uses_dataset,
                uses_axis_id,
                type_string,
                extension,
            );
        }
        self.uses_dataset == uses_dataset
            && self.uses_axis_id == uses_axis_id
            && self.type_string.as_deref() == type_string
            && self.extension.as_deref() == extension
    }

    /// Java `equals(AxisType, boolean, boolean, String)` (deprecated 2/2/2019).  Returns
    /// true if the dataset and axis settings, and type string are equal.  Ignores the
    /// extension.
    pub fn equals_type_string(
        &self,
        axis_type: Option<AxisType>,
        uses_dataset: bool,
        uses_axis_id: bool,
        type_string: Option<&str>,
    ) -> bool {
        if self.composite {
            // Handle type files which are based on another file type
            if let Some(sub_file_type) = self.sub_file_type.as_ref() {
                return sub_file_type.equals_type_string(
                    axis_type,
                    uses_dataset,
                    uses_axis_id,
                    type_string,
                );
            }
            // Handle file types with single and dual file types instead of descriptions.
            return self.get_child_file_type(axis_type).equals_type_string(
                axis_type,
                uses_dataset,
                uses_axis_id,
                type_string,
            );
        }
        if self.uses_dataset == uses_dataset
            && self.uses_axis_id == uses_axis_id
            && self.type_string.as_deref() == type_string
        {
            return true;
        }
        false
    }

    /// Java `getChildFileType`.  If no manager is available, assumes single axis.
    fn get_child_file_type(&self, axis_type: Option<AxisType>) -> &FileType {
        if self.composite {
            if self.single_file_type.is_some() && self.dual_file_type.is_some() {
                if axis_type == Some(AxisType::DualAxis) {
                    return self.dual_file_type.as_ref().unwrap();
                }
                return self.single_file_type.as_ref().unwrap();
            } else if let Some(sub_file_type) = self.sub_file_type.as_ref() {
                return sub_file_type;
            }
        }
        self
    }

    /// Java `hasFixedName` (deprecated 8/3/2019, will be reduced to private).  Return
    /// true if the name can be generated.  All files with a fileNamePattern can be
    /// generated.
    pub fn has_fixed_name(&self, axis_type: Option<AxisType>) -> bool {
        if self.file_name_pattern.is_some() {
            return true;
        }
        if self.composite {
            if self.sub_file_type.is_some() && self.extension.is_some() {
                return self.extension.as_deref() != Some("")
                    || self
                        .sub_file_type
                        .as_ref()
                        .unwrap()
                        .has_fixed_name(axis_type);
            }
            return self
                .get_child_file_type(axis_type)
                .has_fixed_name(axis_type);
        }
        self.uses_axis_id
            || self.uses_dataset
            || self.extension.as_deref() != Some("")
            || self.type_string.as_deref() != Some("")
    }

    /// Java `isInSubdirectory`.
    pub fn is_in_subdirectory(&self) -> bool {
        self.in_subdirectory
    }

    /// Java `getExtension(AxisType)` (deprecated 8/3/2019).  Get the extension for files
    /// with no pattern.
    pub fn get_extension_for_axis_type(&self, axis_type: Option<AxisType>) -> Option<String> {
        if self.file_name_pattern.is_some() {
            // Don't have this information - this function should be private
            return None;
        }
        if self.composite {
            if self.sub_file_type.is_some() && self.extension.is_some() {
                return self.extension.clone();
            }
            return self
                .get_child_file_type(axis_type)
                .get_extension_for_axis_type(axis_type);
        }
        self.extension.clone()
    }

    /// Java `getFile(BaseManager, AxisID, Number, Number)`.
    pub fn get_file_numeric(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        numeric1: Option<&str>,
        numeric2: Option<&str>,
    ) -> Option<std::path::PathBuf> {
        self.get_file_full(
            manager, None, None, None, None, axis_id, None, None, None, numeric1, None, numeric2,
        )
    }

    /// Java `getFile(BaseManager, AxisID)`.
    pub fn get_file(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> Option<std::path::PathBuf> {
        self.get_file_full(
            manager, None, None, None, None, axis_id, None, None, None, None, None, None,
        )
    }

    /// Java `getFile(BaseManager, String, AxisType, AxisID, String)`.
    // ALIGN_FRAMES
    pub fn get_file_with_property_user_dir(
        &self,
        manager: Option<&'static dyn BaseManager>,
        rootname: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        property_use_dir: Option<&str>,
    ) -> Option<std::path::PathBuf> {
        self.get_file_full(
            manager,
            None,
            None,
            rootname,
            axis_type,
            axis_id,
            property_use_dir,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Java `getFile(String, AxisID, String)`.
    pub fn get_file_from_root_name(
        &self,
        rootname: Option<&str>,
        axis_id: Option<AxisID>,
        property_use_dir: Option<&str>,
    ) -> Option<std::path::PathBuf> {
        self.get_file_full(
            None,
            None,
            None,
            rootname,
            None,
            axis_id,
            property_use_dir,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Java package-private `getFile(BaseManager, BaseMetaData, ImageFilenameStyle,
    /// String, AxisType, AxisID, String, String, String, Number, String, Number)`.
    ///
    /// Pass in either manager, metaData, or rootName and axisType.  If manager is null,
    /// pass in propertyUserDir and fileSubdirectoryName if necessary.
    ///
    /// The `numeric1`/`numeric2` parameters are the already-formatted strings; see the
    /// note on `Variable::to_formatted_string` above for why the `Number` transfer is
    /// left out.
    #[allow(clippy::too_many_arguments)]
    pub fn get_file_full(
        &self,
        manager: Option<&'static dyn BaseManager>,
        meta_data: Option<&dyn BaseMetaData>,
        override_image_filename_style: Option<ImageFilenameStyle>,
        root_name: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        property_user_dir: Option<&str>,
        file_subdirectory_name: Option<&str>,
        formatted_numeric1: Option<&str>,
        numeric1: Option<&str>,
        formatted_numeric2: Option<&str>,
        numeric2: Option<&str>,
    ) -> Option<std::path::PathBuf> {
        let mut meta_data = meta_data;
        let mut axis_type = axis_type;
        let mut root_name_string: Option<String> = root_name.map(|name| name.to_string());
        let mut property_user_dir_string: Option<String> =
            property_user_dir.map(|dir| dir.to_string());
        let mut file_subdirectory_name_string: Option<String> =
            file_subdirectory_name.map(|name| name.to_string());
        if meta_data.is_none() {
            if let Some(manager) = manager {
                meta_data = manager.get_base_meta_data();
            }
        }
        if axis_type.is_none() {
            axis_type = match meta_data {
                Some(meta_data) => Some(meta_data.base().get_axis_type()),
                None => Some(AxisType::SingleAxis),
            };
        }
        if root_name_string.is_none() {
            root_name_string = match meta_data {
                Some(meta_data) => meta_data.get_name(),
                None => Some(String::new()),
            };
        }
        if self.file_name_pattern.is_none() && !self.has_fixed_name(axis_type) {
            return None;
        }
        let file_name = self.get_file_name_full(
            manager,
            meta_data,
            override_image_filename_style,
            root_name_string.as_deref(),
            axis_type,
            axis_id,
            false,
            None,
            formatted_numeric1,
            numeric1,
            formatted_numeric2,
            numeric2,
        );
        let file_name = match file_name {
            None => return None,
            Some(file_name) if file_name.is_empty() => return None,
            Some(file_name) => file_name,
        };
        if self.file_name_pattern.is_some() {
            return Some(std::path::PathBuf::from(java_io_file_get_absolute_path(
                &utilities::java_io_file_new(
                    &self
                        .get_directory(
                            manager,
                            meta_data,
                            root_name_string.as_deref(),
                            axis_type,
                            axis_id,
                            property_user_dir_string.as_deref(),
                            formatted_numeric1,
                            numeric1,
                            formatted_numeric2,
                            numeric2,
                        )
                        .map(|dir| dir.to_string_lossy().to_string())
                        .unwrap_or("null".to_string()),
                    &file_name,
                ),
            )));
        }
        // Make the file. The file may be in a subdirectory under IMOD_DIR, in a
        // subdirectory below the dataset location, or in the dataset.
        if let Some(in_imod_subdirectory) = &self.in_imod_subdirectory {
            let imod_directory = etomo_director::INSTANCE
                .lock()
                .unwrap()
                .get_imod_directory()
                .map(|dir| dir.to_string_lossy().to_string());
            return Some(std::path::PathBuf::from(java_io_file_get_absolute_path(
                &utilities::java_io_file_new(
                    &utilities::java_io_file_new(
                        &imod_directory.unwrap_or("null".to_string()),
                        in_imod_subdirectory,
                    ),
                    &file_name,
                ),
            )));
        }
        if property_user_dir_string.is_none() {
            if let Some(manager) = manager {
                property_user_dir_string = manager.get_property_user_dir();
            }
        }
        if self.in_subdirectory {
            if file_subdirectory_name_string.is_none() {
                if let Some(manager) = manager {
                    file_subdirectory_name_string = manager.get_file_subdirectory_name();
                }
            }
            if let Some(subdir) = &self.subdir {
                return Some(std::path::PathBuf::from(java_io_file_get_absolute_path(
                    &utilities::java_io_file_new(
                        &subdir
                            .get_file_name_full(
                                manager,
                                meta_data,
                                override_image_filename_style,
                                root_name_string.as_deref(),
                                axis_type,
                                axis_id,
                                false,
                                None,
                                formatted_numeric1,
                                numeric1,
                                formatted_numeric2,
                                numeric2,
                            )
                            .unwrap_or("null".to_string()),
                        &file_name,
                    ),
                )));
            }
            if manager.is_some() && file_subdirectory_name_string.is_some() {
                return Some(std::path::PathBuf::from(java_io_file_get_absolute_path(
                    &utilities::java_io_file_new(
                        &utilities::java_io_file_new(
                            &property_user_dir_string
                                .clone()
                                .unwrap_or("null".to_string()),
                            file_subdirectory_name_string.as_deref().unwrap(),
                        ),
                        &file_name,
                    ),
                )));
            }
        }
        let mut dir = property_user_dir_string.clone();
        if property_user_dir_string.is_none() {
            dir = etomo_director::INSTANCE
                .lock()
                .unwrap()
                .original_user_dir
                .clone();
        }
        Some(std::path::PathBuf::from(java_io_file_get_absolute_path(
            &utilities::java_io_file_new(&dir.unwrap_or("null".to_string()), &file_name),
        )))
    }

    /// Java private `getDirectory`.  Call when pattern is set.  Only uses the directory
    /// member variable.
    #[allow(clippy::too_many_arguments)]
    fn get_directory(
        &self,
        manager: Option<&'static dyn BaseManager>,
        meta_data: Option<&dyn BaseMetaData>,
        root_name: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        property_user_dir: Option<&str>,
        formatted_numeric1: Option<&str>,
        numeric1: Option<&str>,
        formatted_numeric2: Option<&str>,
        numeric2: Option<&str>,
    ) -> Option<std::path::PathBuf> {
        if let Some(directory) = &self.directory {
            return directory.get_file_full(
                manager,
                meta_data,
                None,
                root_name,
                axis_type,
                axis_id,
                property_user_dir,
                None,
                formatted_numeric1,
                numeric1,
                formatted_numeric2,
                numeric2,
            );
        }
        let mut directory_name = property_user_dir.map(|dir| dir.to_string());
        if directory_name.is_none() {
            if let Some(manager) = manager {
                directory_name = manager.get_property_user_dir();
            }
        }
        if directory_name.is_none() {
            directory_name = etomo_director::INSTANCE
                .lock()
                .unwrap()
                .original_user_dir
                .clone();
        }
        Some(std::path::PathBuf::from(
            directory_name.unwrap_or("null".to_string()),
        ))
    }

    /// Java `exists(BaseManager, AxisID)`.
    pub fn exists(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> bool {
        let file = self.get_file_full(
            manager, None, None, None, None, axis_id, None, None, None, None, None, None,
        );
        if let Some(file) = file {
            return file.exists();
        }
        false
    }

    /// Java `lastModified(BaseManager, AxisID)`.
    pub fn last_modified(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> i64 {
        let file = self.get_file_full(
            manager, None, None, None, None, axis_id, None, None, None, None, None, None,
        );
        if let Some(file) = file {
            // `File.lastModified()`, milliseconds since the epoch, 0 when unavailable.
            return utilities::java_io_file_last_modified(&file.to_string_lossy());
        }
        -1
    }

    /// Java `exists(BaseManager, BaseMetaData, AxisID)`.  Note that the source ignores
    /// its `metaData` parameter and passes null on.
    pub fn exists_with_meta_data(
        &self,
        manager: Option<&'static dyn BaseManager>,
        meta_data: Option<&dyn BaseMetaData>,
        axis_id: Option<AxisID>,
    ) -> bool {
        let _ = meta_data;
        let file = self.get_file_full(
            manager, None, None, None, None, axis_id, None, None, None, None, None, None,
        );
        if let Some(file) = file {
            return file.exists();
        }
        false
    }

    /// Java `getRoot`.  Return the file name, stripped of its extension.
    pub fn get_root(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> String {
        // Template OK at least for now - the current template types have extra text after
        // the extension.
        let file_name = self
            .get_file_name_full(
                manager, None, None, None, None, axis_id, true, None, None, None, None, None,
            )
            // The source dereferences the result without a null check.
            .expect("java.lang.NullPointerException");
        let index = file_name.rfind('.').map(|index| index as i64).unwrap_or(-1);
        // `fileName.substring(0, index)`, which throws for a name with no period.
        file_name[..index as usize].to_string()
    }

    /// Java `getTemplate`.
    pub fn get_template(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> Option<String> {
        self.get_file_name_full(
            manager, None, None, None, None, axis_id, true, None, None, None, None, None,
        )
    }

    /// Java `getFileName(BaseManager, AxisID)`, which overrides
    /// `FileKey.getFileName(BaseManager, AxisID)`.
    pub fn get_file_name(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> Option<String> {
        self.get_file_name_full(
            manager, None, None, None, None, axis_id, false, None, None, None, None, None,
        )
    }

    /// Java `getFileName(BaseManager, AxisID, String, String)`.
    pub fn get_file_name_formatted_numeric(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        formatted_numeric1: Option<&str>,
        formatted_numeric2: Option<&str>,
    ) -> Option<String> {
        self.get_file_name_full(
            manager,
            None,
            None,
            None,
            None,
            axis_id,
            false,
            None,
            formatted_numeric1,
            None,
            formatted_numeric2,
            None,
        )
    }

    /// Java `getFileName(BaseManager, AxisID, Number, Number)`.
    pub fn get_file_name_numeric(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        numeric1: Option<&str>,
        numeric2: Option<&str>,
    ) -> Option<String> {
        self.get_file_name_full(
            manager, None, None, None, None, axis_id, false, None, None, numeric1, None, numeric2,
        )
    }

    /// Java `getFileName(BaseManager, String, AxisType, AxisID)`.
    pub fn get_file_name_with_axis_type(
        &self,
        manager: Option<&'static dyn BaseManager>,
        root_name: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
    ) -> Option<String> {
        self.get_file_name_full(
            manager, None, None, root_name, axis_type, axis_id, false, None, None, None, None, None,
        )
    }

    /// Java `getFileName(BaseManager, BaseMetaData, AxisID)`.
    pub fn get_file_name_with_meta_data(
        &self,
        manager: Option<&'static dyn BaseManager>,
        meta_data: Option<&dyn BaseMetaData>,
        axis_id: Option<AxisID>,
    ) -> Option<String> {
        self.get_file_name_full(
            manager, meta_data, None, None, None, axis_id, false, None, None, None, None, None,
        )
    }

    /// Java `getFileName(BaseManager, String, AxisID)`.
    pub fn get_file_name_with_root_name(
        &self,
        manager: Option<&'static dyn BaseManager>,
        root_name: Option<&str>,
        axis_id: Option<AxisID>,
    ) -> Option<String> {
        self.get_file_name_full(
            manager, None, None, root_name, None, axis_id, false, None, None, None, None, None,
        )
    }

    /// Java `getFile(BaseManager, String, AxisType, AxisID)`.
    pub fn get_file_with_axis_type(
        &self,
        manager: Option<&'static dyn BaseManager>,
        rootname: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
    ) -> Option<std::path::PathBuf> {
        self.get_file_full(
            manager, None, None, rootname, axis_type, axis_id, None, None, None, None, None, None,
        )
    }

    /// Java `getFile(BaseManager, File, String, AxisType, AxisID)`.  Builds a file in a
    /// directory.
    pub fn get_file_in_dir(
        &self,
        manager: Option<&'static dyn BaseManager>,
        dir: Option<&std::path::Path>,
        root_name: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
    ) -> Option<std::path::PathBuf> {
        let file_name = self.get_file_name_full(
            manager, None, None, root_name, axis_type, axis_id, false, None, None, None, None, None,
        );
        if let Some(file_name) = file_name {
            return Some(std::path::PathBuf::from(utilities::java_io_file_new(
                &dir.map(|dir| dir.to_string_lossy().to_string())
                    .unwrap_or("null".to_string()),
                &file_name,
            )));
        }
        None
    }

    /// Java private `getFileName(BaseManager, BaseMetaData, ImageFilenameStyle, String,
    /// AxisType, AxisID, boolean, String, String, Number, String, Number)`.
    ///
    /// Pass in either manager, metaData, or rootName and axisType.
    #[allow(clippy::too_many_arguments)]
    fn get_file_name_full(
        &self,
        manager: Option<&'static dyn BaseManager>,
        meta_data: Option<&dyn BaseMetaData>,
        override_image_filename_style: Option<ImageFilenameStyle>,
        root_name: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        template_ok: bool,
        version: Option<&str>,
        formatted_numeric1: Option<&str>,
        numeric1: Option<&str>,
        formatted_numeric2: Option<&str>,
        numeric2: Option<&str>,
    ) -> Option<String> {
        let mut meta_data = meta_data;
        let mut axis_type = axis_type;
        let mut root_name_string: Option<String> = root_name.map(|name| name.to_string());
        if meta_data.is_none() {
            if let Some(manager) = manager {
                meta_data = manager.get_base_meta_data();
            }
        }
        if axis_type.is_none() {
            axis_type = match meta_data {
                Some(meta_data) => Some(meta_data.base().get_axis_type()),
                None => Some(AxisType::SingleAxis),
            };
        }
        if root_name_string.is_none() {
            root_name_string = match meta_data {
                Some(meta_data) => meta_data.get_name(),
                None => Some(String::new()),
            };
        }
        if self.file_name_pattern.is_none() && !self.has_fixed_name(axis_type) {
            return None;
        }
        let mut include_extension = true;
        if self.template {
            if !template_ok && *DEBUG {
                eprintln!("Warning:  Getting the file name of template {}", self);
            }
            if self.dynamic_location == Some(DynamicLocation::AfterTypeString)
                || self.dynamic_location
                    == Some(DynamicLocation::AfterTypeStringAndAfterMiddlePiece)
            {
                include_extension = false;
            }
        }
        if self.composite && (self.sub_file_type.is_none() || self.extension.is_none()) {
            return self.get_child_file_type(axis_type).get_file_name_full(
                manager,
                meta_data,
                override_image_filename_style,
                root_name_string.as_deref(),
                axis_type,
                axis_id,
                true,
                version,
                formatted_numeric1,
                numeric1,
                formatted_numeric2,
                numeric2,
            );
        }
        if self.file_name_pattern.is_some() {
            let orig_raw_image_stack_extension;
            let image_filename_style;
            let raw_image_stack_extension;
            // The `ImageFileMetaData` branch builds a fresh instance, which the borrow
            // below must outlive, so it is bound here.
            let image_file_meta_data;
            match meta_data {
                Some(meta_data) => {
                    orig_raw_image_stack_extension = Extension::get_instance(
                        &meta_data
                            .base()
                            .get_orig_raw_image_stack_extension()
                            .to_string(),
                    );
                    image_filename_style = Some(meta_data.base().get_image_filename_style());
                    raw_image_stack_extension =
                        Some(meta_data.base().get_raw_image_stack_extension());
                }
                None => {
                    orig_raw_image_stack_extension = None;
                    image_file_meta_data =
                        ImageFileMetaData::new(false, override_image_filename_style, true);
                    image_filename_style = Some(image_file_meta_data.get_image_filename_style());
                    raw_image_stack_extension =
                        Some(image_file_meta_data.get_default_raw_image_stack_extension());
                }
            }
            return self.get_file_name_from_pattern(
                root_name_string.as_deref(),
                axis_type,
                axis_id,
                formatted_numeric1,
                formatted_numeric2,
                orig_raw_image_stack_extension,
                image_filename_style,
                raw_image_stack_extension,
            );
        }
        Some(format!(
            "{}{}",
            self.get_left_side(root_name_string.as_deref(), axis_type, axis_id, version)
                .unwrap_or("null".to_string()),
            if include_extension {
                self.extension.clone().unwrap_or("null".to_string())
            } else {
                String::new()
            }
        ))
    }

    /// Java `joinPatterns`.  Returns the concatenation of the left side of
    /// `patternForLeftSide` with all of `suffixPattern`.
    ///
    /// Important: if the entire file name is needed on the left side, more recent
    /// functionality allows just adding the `FileType` instance to the pattern.
    fn join_patterns(
        pattern_for_left_side: Option<Vec<PatternElement>>,
        suffix_pattern: Option<Vec<PatternElement>>,
    ) -> Option<Vec<PatternElement>> {
        if pattern_for_left_side.is_none() && suffix_pattern.is_none() {
            return None;
        }
        match &pattern_for_left_side {
            None => return suffix_pattern,
            Some(pattern_for_left_side) if pattern_for_left_side.is_empty() => {
                return suffix_pattern;
            }
            Some(_) => {}
        }
        let pattern_for_left_side = pattern_for_left_side.unwrap();
        // Construct the new pattern array with the largest possible length.  Java's
        // `suffixPattern.length` throws NullPointerException when suffixPattern is null,
        // which the source's own null check below implies cannot happen here.
        let suffix_len = suffix_pattern
            .as_ref()
            .map(|suffix| suffix.len())
            .unwrap_or(0);
        let mut pattern: Vec<PatternElement> =
            vec![PatternElement::Null; pattern_for_left_side.len() + suffix_len];
        let mut last_index: i64 = -1;
        for i in 0..pattern_for_left_side.len() {
            // Don't use the extension from patternForLeftSide
            if matches!(pattern_for_left_side[i], PatternElement::ExtensionMarker(_)) {
                break;
            }
            pattern[i] = pattern_for_left_side[i].clone();
            last_index = i as i64;
        }
        // Add the suffix
        if let Some(suffix_pattern) = suffix_pattern {
            if !suffix_pattern.is_empty() {
                for element in suffix_pattern.iter() {
                    last_index += 1;
                    pattern[last_index as usize] = element.clone();
                }
            }
        }
        // Set nulls for the elements left over from not using the patternForLeftSide
        // extension.
        for i in (last_index + 1) as usize..pattern.len() {
            pattern[i] = PatternElement::Null;
        }
        Some(pattern)
    }

    /// Java `toString(Object[])`.  Convert the file name pattern to create a string that
    /// identifies the file type instance.
    fn to_string_pattern(file_name_pattern: Option<&[PatternElement]>) -> Option<String> {
        let file_name_pattern = match file_name_pattern {
            None => return None,
            Some(file_name_pattern) => file_name_pattern,
        };
        let mut builder = String::new();
        for element in file_name_pattern.iter() {
            match element {
                PatternElement::Str(value) if !utilities::is_empty(Some(value)) => {
                    builder.push_str(value);
                }
                PatternElement::ExtensionMarker(_) => builder.push('.'),
                PatternElement::Extension(extension) => builder.push_str(&extension.to_string()),
                _ => {}
            }
        }
        Some(builder)
    }

    /// Java `getRegex`.  Returns a file name search string (regular expression).
    /// Requires `fileNamePattern`.  All fields are optional.
    ///
    /// * `root_name` - when null uses a search pattern
    /// * `axis_type` - when null or NOT_SET looks for both types - not set from manager
    /// * `axis_id` - when null looks for both axes if axisType is null or dual
    pub fn get_regex(
        &self,
        root_name: Option<&str>,
        mut axis_type: Option<AxisType>,
        mut axis_id: Option<AxisID>,
        image_filename_style: Option<super::image_filename_style::ImageFilenameStyle>,
        raw_stack_extenson: Option<&Extension>,
    ) -> Option<String> {
        self.file_name_pattern.as_ref()?;
        // Allow regular expression to contain patterns instead of specific axis
        // information.
        if axis_type == Some(AxisType::NotSet) {
            axis_type = None;
        }
        if axis_id.is_some() {
            axis_id = Some(FileType::correct_axis_id(axis_type, axis_id));
        }
        let mut first_axis_type: Option<AxisType> = None;
        let mut pattern_axis_type: Option<AxisType> = None;
        if self.single_axis_file_name_pattern.is_some() {
            if axis_type.is_none() {
                // Will have to "or" the two patterns together so the regex can match both
                // axis types.  Tell the builder which axis type it is starting with.
                first_axis_type = Some(AxisType::DualAxis);
            } else if axis_type == Some(AxisType::SingleAxis) {
                pattern_axis_type = Some(AxisType::SingleAxis);
            }
        }
        let mut builder = FileNameBuilder::new(
            true,
            first_axis_type,
            root_name,
            axis_type,
            axis_id,
            None,
            None,
            None,
        );
        let regex = self.build_pattern(
            &mut builder,
            pattern_axis_type,
            image_filename_style,
            raw_stack_extenson,
        );
        if first_axis_type.is_none() || !builder.start_second_pattern() {
            return Some(regex);
        }
        // Add the single axis pattern
        Some(self.build_pattern(
            &mut builder,
            Some(AxisType::SingleAxis),
            image_filename_style,
            raw_stack_extenson,
        ))
    }

    /// Java `getFileNameFromPattern`.  Returns a file name.  Requires
    /// `fileNamePattern`.
    ///
    /// * `root_name` - when null uses an empty root name
    /// * `axis_type` - defaults to single axis
    /// * `axis_id` - `AxisID.SECOND` overrides axisType, otherwise follows axisType and
    ///   defaults to FIRST when dual
    /// * `formatted_numeric1` - one of the numeric parameters is required if the file has
    ///   a variable piece
    /// * `formatted_numeric2` - required if the file has a second variable piece
    ///
    /// Note: the source also accepts raw `Number numeric1`/`numeric2` and converts them
    /// with `Variable.toFormattedString`, which needs `etomo/logic/Converter.java`; see
    /// the `TODO(unit)` on `Variable`.  Only the already-formatted parameters are
    /// accepted here.
    #[allow(clippy::too_many_arguments)]
    pub fn get_file_name_from_pattern(
        &self,
        root_name: Option<&str>,
        mut axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        formatted_numeric1: Option<&str>,
        formatted_numeric2: Option<&str>,
        orig_raw_image_stack_extension: Option<&Extension>,
        image_filename_style: Option<super::image_filename_style::ImageFilenameStyle>,
        raw_stack_extenson: Option<&Extension>,
    ) -> Option<String> {
        self.file_name_pattern.as_ref()?;
        if axis_type.is_none() || axis_type == Some(AxisType::NotSet) {
            axis_type = Some(AxisType::SingleAxis);
        }
        let axis_id = Some(FileType::correct_axis_id(axis_type, axis_id));
        let mut pattern_axis_type: Option<AxisType> = None;
        if self.single_axis_file_name_pattern.is_some() && axis_type == Some(AxisType::SingleAxis) {
            pattern_axis_type = Some(AxisType::SingleAxis);
        }
        let mut builder = FileNameBuilder::new(
            false,
            None,
            root_name,
            axis_type,
            axis_id,
            formatted_numeric1,
            formatted_numeric2,
            orig_raw_image_stack_extension,
        );
        Some(self.build_pattern(
            &mut builder,
            pattern_axis_type,
            image_filename_style,
            raw_stack_extenson,
        ))
    }

    /// Java `selectPattern`.
    fn select_pattern(&self, pattern_axis_type: Option<AxisType>) -> Option<&Vec<PatternElement>> {
        if self.single_axis_file_name_pattern.is_some()
            && pattern_axis_type == Some(AxisType::SingleAxis)
        {
            return self.single_axis_file_name_pattern.as_ref();
        }
        self.file_name_pattern.as_ref()
    }

    /// Java `buildPattern`.  `fileNamePattern` is required for this function.
    fn build_pattern(
        &self,
        builder: &mut FileNameBuilder,
        pattern_axis_type: Option<AxisType>,
        image_filename_style: Option<super::image_filename_style::ImageFilenameStyle>,
        raw_stack_extenson: Option<&Extension>,
    ) -> String {
        // Java dereferences the selected pattern unconditionally; a NullPointerException
        // here means the caller broke the "fileNamePattern is required" contract.
        let pattern = match self.select_pattern(pattern_axis_type) {
            None => return builder.to_string(),
            Some(pattern) => pattern.clone(),
        };
        let mut extension_marker: Option<ExtensionMarker> = None;
        let mut add_extension_from_settings = false;
        for i in 0..pattern.len() {
            match &pattern[i] {
                PatternElement::FileType(file_type) => {
                    file_type.build_pattern(
                        builder,
                        pattern_axis_type,
                        image_filename_style,
                        raw_stack_extenson,
                    );
                }
                PatternElement::Variable(variable) => {
                    builder.append_variable(variable);
                }
                PatternElement::ExtensionMarker(marker) => {
                    // There can be multiple extensions at the end of the file name
                    // because one file name can be derived from another.  Each extension
                    // needs to be processed the same way.
                    extension_marker = Some(*marker);
                    if *marker == ExtensionMarker::InputImage {
                        // The standard extension is always required for input images.
                        // Substitute the correct extension.
                        add_extension_from_settings = true;
                        break;
                    }
                    // Place the correct divider for this file name.
                    if *marker == ExtensionMarker::Image
                        && marker.uses_standard_extension(image_filename_style)
                    {
                        // Use the standard extension
                        add_extension_from_settings = true;
                        if !marker.is_compatible(i as i32, Some(&pattern)) {
                            // First standardize old style file name
                            // (dataset.flat => dataset_flat).
                            builder.append_string(Some(extension::STANDARDIZATION_DIVIDER));
                        } else {
                            // The standard extension is required when the file style is
                            // standard.  Substitute the correct extension.
                            break;
                        }
                    } else {
                        builder.append_string(Some(extension::EXTENSION_DIVIDER));
                    }
                }
                PatternElement::Null => {}
                element => {
                    builder.append_string(Some(&element.to_string_element()));
                }
            }
        }
        // Add the standard extension if necessary.
        if add_extension_from_settings {
            if extension_marker == Some(ExtensionMarker::Image) {
                // This doesn't make much sense if this was old style, but
                // addExtensionFromSettings shouldn't be true in cases like that.
                let extension = image_filename_style.map(|image_filename_style| {
                    image_filename_style.get_default_raw_image_stack_extension()
                });
                if let Some(extension) = extension {
                    builder.append_string(Some(extension::EXTENSION_DIVIDER));
                    builder.append_string(Some(&extension.to_string()));
                }
            } else if extension_marker == Some(ExtensionMarker::InputImage) {
                builder.append_string(Some(extension::EXTENSION_DIVIDER));
                builder.append_string(Some(
                    &raw_stack_extenson
                        .map(|extension| extension.to_string())
                        .unwrap_or_else(|| "null".to_string()),
                ));
            }
        }
        builder.done();
        builder.to_string()
    }

    /// Java `getLeftSide`.  Get the typeString with the dataset and axis letter added as
    /// necessary.  For example, the left side of BBa_fixed.st is "BBa_fixed", the left
    /// side of tilta.com is "tilta", the left side of tilt.com is "tilt", and the left
    /// side of tilta_for_sirt.com is "tilta".  `version` is always at the end of the left
    /// side.
    pub fn get_left_side(
        &self,
        root_name: Option<&str>,
        mut axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        version: Option<&str>,
    ) -> Option<String> {
        let root_name = root_name.unwrap_or("");
        let version = version.unwrap_or("");
        if axis_type.is_none() || axis_type == Some(AxisType::NotSet) {
            axis_type = Some(AxisType::SingleAxis);
        }
        if !self.has_fixed_name(axis_type) {
            return None;
        }
        if self.composite {
            return self.get_child_file_type(axis_type).get_left_side(
                Some(root_name),
                axis_type,
                axis_id,
                Some(version),
            );
        }
        let type_string = self.type_string.as_deref().unwrap_or("null");
        if !self.uses_dataset && !self.uses_axis_id {
            // Example: flatten.com
            return Some(type_string.to_string() + version);
        }
        let mut axis_id_extension = String::new();
        if self.uses_axis_id {
            axis_id_extension = FileType::correct_axis_id(axis_type, axis_id).get_extension();
        }
        if self.uses_dataset {
            // With the dataset the axis follows the dataset
            // Example: BBa_erase.fid
            return Some(root_name.to_string() + &axis_id_extension + type_string + version);
        }
        // Without the dataset the axis follows the left extension
        // Example: tilta.com
        Some(type_string.to_string() + &axis_id_extension + version)
    }

    /// Java `deriveFileName`.  Derive a file name with the same type as this instance,
    /// but with a different root name and/or a different axis type as the manager
    /// parameter.
    pub fn derive_file_name(
        &self,
        root_name: Option<&str>,
        axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        image_filename_style: Option<super::image_filename_style::ImageFilenameStyle>,
        raw_stack_extenson: Option<&Extension>,
    ) -> Option<String> {
        if self.file_name_pattern.is_some() {
            return self.get_file_name_from_pattern(
                root_name,
                axis_type,
                axis_id,
                None,
                None,
                None,
                image_filename_style,
                raw_stack_extenson,
            );
        }
        // Java concatenates two possibly-null Strings, which prints "null".
        Some(
            self.get_left_side(root_name, axis_type, axis_id, None)
                .unwrap_or_else(|| "null".to_string())
                + &self
                    .get_extension_for_axis_type(axis_type)
                    .unwrap_or_else(|| "null".to_string()),
        )
    }

    /// Java `getTypeString(AxisType)` (deprecated 8/2/2019).  Returns the non-generic
    /// part of the left side of the file name.  For example, the type string for
    /// BBa_fixed.st is "_fixed", the type string for tilta.com is "tilt", and the type
    /// string for tilt.com is "tilt".
    pub fn get_type_string_for_axis_type(&self, axis_type: Option<AxisType>) -> Option<String> {
        if self.file_name_pattern.is_some() {
            return None;
        }
        if self.composite {
            return self
                .get_child_file_type(axis_type)
                .get_type_string_for_axis_type(axis_type);
        }
        self.type_string.clone()
    }

    /// Java `getTypeString()`.
    ///
    /// WARNING: Does not get the child file type.  Ignores the composite setting.  Just
    /// returns the string in this instance.
    pub fn get_type_string(&self) -> Option<&str> {
        self.type_string.as_deref()
    }

    /// Java `getMiddlePiece`.  Templates only.  Returns a part of the file name between
    /// the type string and the extension.  Example: in the Gaussian filter output file
    /// template of format "dataset_gfc0.xxx-f0.xxx.mrc", "-f" is the middle piece.
    pub fn get_middle_piece(&self) -> Option<&str> {
        self.template_only_middle_piece.as_deref()
    }

    /// Java `getExtension()`.  Returns the extension.
    ///
    /// WARNING: Does not get the child file type.  Ignores the composite setting.  Just
    /// returns the string in this instance.
    pub fn get_extension(&self) -> Option<&str> {
        self.extension.as_deref()
    }

    /// Java `usesDataset`.
    pub fn uses_dataset(&self) -> bool {
        self.uses_dataset
    }

    /// Java `usesAxisID`.
    pub fn uses_axis_id(&self) -> bool {
        self.uses_axis_id
    }

    /// Java `correctAxisID`.  A null axisID or an ONLY axisID is sometimes used to
    /// signify a FIRST axisID in a dual axis dataset.  A similar problem may exist for
    /// single axis datasets.  The axisID must be corrected to get a valid file name.
    ///
    /// This is not true for Tomogram Combination file names, which do not have an axisID
    /// letter (equivalent to `AxisID.ONLY`).  However these files would have the
    /// `usesAxisID` member variable set to false, so that is not a problem.
    fn correct_axis_id(axis_type: Option<AxisType>, axis_id: Option<AxisID>) -> AxisID {
        if axis_id == Some(AxisID::Second) {
            return AxisID::Second;
        }
        if axis_type == Some(AxisType::DualAxis) {
            return AxisID::First;
        }
        AxisID::Only
    }

    /// Java `iterator` (deprecated 3/14/2019).  Returns `namedFileTypeList.iterator()`.
    pub fn iterator() -> Vec<Arc<FileType>> {
        // As in `getInstance`: naming the class runs its static initialisers.
        LazyLock::force(&CLASS);
        NAMED_FILE_TYPE_LIST.lock().unwrap().clone()
    }

    /// Java `getDescription`, overriding `FileKey.getDescription`.
    pub fn get_description(self: &Arc<FileType>) -> Option<String> {
        let retval = self.file_key.get_description();
        if retval.is_some() {
            return retval;
        }
        if Arc::ptr_eq(self, &CLASS.fiducial_3d_model) {
            return Some("FIDUCIAL_3D_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.aligned_stack) {
            return Some("ALIGNED_STACK".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.xcorr_blend_output) {
            return Some("XCORR_BLEND_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.distortion_corrected_stack) {
            return Some("DISTORTION_CORRECTED_STACK".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.fiducial_model) {
            return Some("FIDUCIAL_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.flatten_tool_output) {
            return Some("FLATTEN_TOOL_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.join) {
            return Some("JOIN".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.anisotropic_diffusion_output) {
            return Some("ANISOTROPIC_DIFFUSION_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.prealigned_stack) {
            return Some("PREALIGNED_STACK".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.raw_tilt_angles) {
            return Some("RAW_TILT_ANGLES".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.trim_vol_output) {
            return Some("TRIM_VOL_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.join_sample_averages) {
            return Some("JOIN_SAMPLE_AVERAGES".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.join_sample) {
            return Some("JOIN_SAMPLE".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.squeeze_vol_output) {
            return Some("SQUEEZE_VOL_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.newst_or_blend_3d_find_output) {
            return Some("NEWST_OR_BLEND_3D_FIND_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.find_beads_3d_output_model) {
            return Some("FIND_BEADS_3D_OUTPUT_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.tilt_3d_find_output) {
            return Some("TILT_3D_FIND_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.smoothing_assessment_output_model) {
            return Some("SMOOTHING_ASSESSMENT_OUTPUT_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.ctf_corrected_stack) {
            return Some("CTF_CORRECTED_STACK".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.ctf_correction_comscript) {
            return Some("CTF_CORRECTION_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.erased_beads_stack) {
            return Some("ERASED_BEADS_STACK".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.ccd_eraser_beads_input_model) {
            return Some("CCD_ERASER_BEADS_INPUT_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.mtf_filtered_stack) {
            return Some("MTF_FILTERED_STACK".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.find_beads_3d_comscript) {
            return Some("FIND_BEADS_3D_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.fixed_xrays_stack) {
            return Some("FIXED_XRAYS_STACK".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.flatten_warp_input_model) {
            return Some("FLATTEN_WARP_INPUT_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.flatten_comscript) {
            return Some("FLATTEN_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.flatten_output) {
            return Some("FLATTEN_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.flatten_tool_comscript) {
            return Some("FLATTEN_TOOL_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.tilt_output) {
            return Some("TILT_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.sirt_scaled_output_template) {
            return Some("SIRT_SCALED_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.sirt_output_template) {
            return Some("SIRT_OUTPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.modeled_join) {
            return Some("MODELED_JOIN".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.mtf_filter_comscript) {
            return Some("MTF_FILTER_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.original_raw_stack) {
            return Some("ORIGINAL_RAW_STACK".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.patch_vector_model) {
            return Some("PATCH_VECTOR_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.patch_vector_ccc_model) {
            return Some("PATCH_VECTOR_CCC_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.patch_tracking_boundary_model) {
            return Some("PATCH_TRACKING_BOUNDARY_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.transformed_refining_model) {
            return Some("TRANSFORMED_REFINING_MODEL".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.sirtsetup_comscript) {
            return Some("SIRTSETUP_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.combined_volume) {
            return Some("COMBINED_VOLUME".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.nad_test_input) {
            return Some("NAD_TEST_INPUT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.tilt_comscript) {
            return Some("TILT_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.tilt_for_sirt_comscript) {
            return Some("TILT_FOR_SIRT_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.track_comscript) {
            return Some("TRACK_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.trial_join) {
            return Some("TRIAL_JOIN".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.cross_correlation_comscript) {
            return Some("CROSS_CORRELATION_COMSCRIPT".to_string());
        }
        if Arc::ptr_eq(self, &CLASS.patch_tracking_comscript) {
            return Some("PATCH_TRACKING_COMSCRIPT".to_string());
        }
        // The source's last six tests - `this == AVERAGED_VOLUMES`,
        // `NAD_TEST_VARYING_ITERATIONS`, `NAD_TEST_VARYING_K`, `POSITIONING_SAMPLE`,
        // `REFERENCE_VOLUMES` and `TRIAL_TOMOGRAM` (FileType.java:2782-2799) - name
        // `FileKey` singletons (FileKey.java:7-16), not `FileType` ones.  `FileType`
        // extends `FileKey`, so Java compiles the comparison, but a `FileType` instance
        // is never one of those six objects: all six arms are dead code, and Rust's
        // types say the same thing.
        self.get_file_format_descr()
    }

    /// Java `getFileFormatDescr`.
    pub fn get_file_format_descr(&self) -> Option<String> {
        let axis_type = Some(AxisType::SingleAxis);
        let root_name = "";
        if self.file_name_pattern.is_none() && !self.has_fixed_name(axis_type) {
            return self.file_key.get_description();
        }
        let mut include_extension = true;
        if self.template
            && (self.dynamic_location == Some(DynamicLocation::AfterTypeString)
                || self.dynamic_location
                    == Some(DynamicLocation::AfterTypeStringAndAfterMiddlePiece))
        {
            include_extension = false;
        }
        if self.composite && (self.sub_file_type.is_none() || self.extension.is_none()) {
            return self.get_child_file_type(axis_type).get_file_format_descr();
        }
        if self.file_name_pattern.is_some() {
            let mut pattern = self.file_name_pattern.as_ref();
            if self.single_axis_file_name_pattern.is_some() {
                pattern = self.single_axis_file_name_pattern.as_ref();
            }
            // Note: the source calls `pattern.toString()` on the *array object*, not on
            // its contents, so the value is `java.lang.Object.toString()` for an array -
            // `[Ljava.lang.Object;@<hash>`.  The hash is the JVM's identity hash code;
            // see the identity-hash note in `etomo/type/ConstEtomoNumber.java`'s module.
            // A process-local counter stands in for it so the shape matches, and the
            // selected pattern is therefore read but not printed, exactly as in Java.
            let _selected_pattern = pattern;
            static IDENTITY_HASH: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(1);
            return Some(format!(
                "[Ljava.lang.Object;@{:x}",
                IDENTITY_HASH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
        }
        Some(
            self.get_left_side(Some(root_name), axis_type, None, None)
                .unwrap_or_else(|| "null".to_string())
                + if include_extension {
                    self.extension.as_deref().unwrap_or("null")
                } else {
                    ""
                },
        )
    }
}

/// Java `toString()`.  Attempt to create a string that identifies the file type
/// instance.
impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut builder = String::new();
        // Handle old contruction style
        if !utilities::is_empty(self.type_string.as_deref()) {
            builder.push_str(self.type_string.as_deref().unwrap());
        }
        if !utilities::is_empty(self.template_only_middle_piece.as_deref()) {
            builder.push_str(self.template_only_middle_piece.as_deref().unwrap());
        }
        if !utilities::is_empty(self.extension.as_deref()) {
            builder.push_str(self.extension.as_deref().unwrap());
        }
        if !builder.is_empty() {
            return f.write_str(&builder);
        }
        // Handle new file name pattern.
        if self.file_name_pattern.is_some() {
            builder.push_str(
                &FileType::to_string_pattern(self.file_name_pattern.as_deref()).unwrap_or_default(),
            );
        }
        let mut single_axis: Option<String> = None;
        if self.single_axis_file_name_pattern.is_some() {
            single_axis =
                FileType::to_string_pattern(self.single_axis_file_name_pattern.as_deref());
        }
        if !builder.is_empty() {
            if !utilities::is_empty(single_axis.as_deref()) {
                builder.push_str(&(" / ".to_string() + single_axis.as_deref().unwrap()));
            }
            return f.write_str(&builder);
        }
        // `super.toString()` is `java.lang.Object.toString()`; see the identity-hash note
        // in `etomo/type/ConstEtomoNumber.java`'s module.
        static IDENTITY_HASH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
        write!(
            f,
            "etomo.type.FileType@{:x}",
            IDENTITY_HASH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )
    }
}

/// Java nested class `FileNameBuilder`.  Builds either a file name or a regular
/// expression that matches the file name.
struct FileNameBuilder {
    /// Java field `builder`.
    builder: String,
    /// Java field `buildRegex`: when true a regular expression is built, otherwise the
    /// file name is built.
    build_regex: bool,
    /// Java field `firstAxisType`: when set, expects two patterns (or'ed together);
    /// ignored when `buildRegex` is false.
    first_axis_type: Option<AxisType>,
    /// Java field `rootName`: when null uses the dataset regex.
    root_name: String,
    /// Java field `origRawStackExtension`.  The extension includes the "."!
    orig_raw_stack_extension: String,
    /// Java field `formattedNumeric1`: when null adds a regex.
    formatted_numeric1: Option<String>,
    /// Java field `formattedNumeric2`: when null adds a regex.
    formatted_numeric2: Option<String>,
    /// Java field `startQuote`.
    start_quote: &'static str,
    /// Java field `endQuote`.
    end_quote: &'static str,
    /// Java field `axisIDExtension`, initialised to "" (single axis pattern).
    axis_id_extension: String,
    /// Java field `numericIndex`.
    numeric_index: i32,
    /// Java field `endRegex`, initialised to "$".
    end_regex: String,
}

impl FileNameBuilder {
    /// Java `FileNameBuilder(boolean, AxisType, String, AxisType, AxisID, String, String,
    /// Extension)`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        build_regex: bool,
        first_axis_type: Option<AxisType>,
        root_name: Option<&str>,
        mut axis_type: Option<AxisType>,
        axis_id: Option<AxisID>,
        formatted_numeric1: Option<&str>,
        formatted_numeric2: Option<&str>,
        orig_raw_image_stack_extension: Option<&Extension>,
    ) -> FileNameBuilder {
        let mut builder = String::new();
        // Quotes for creating a pattern match string.
        let start_quote = if build_regex { "\\Q" } else { "" };
        let end_quote = if build_regex { "\\E" } else { "" };
        let this_first_axis_type: Option<AxisType>;
        if build_regex {
            if first_axis_type == Some(AxisType::NotSet) {
                this_first_axis_type = None;
            } else {
                this_first_axis_type = first_axis_type;
            }
            if this_first_axis_type.is_some() {
                axis_type = this_first_axis_type;
                // Putting the two patterns in a required non-capturing group.
                builder.push_str("(?:");
            }
            builder.push('^');
        } else {
            // ignored when building a file name
            this_first_axis_type = None;
        }
        let this_root_name: String;
        match root_name {
            None => {
                if build_regex {
                    // If the rootName is missing use pattern match string for it
                    this_root_name = Variable::dataset().regex;
                } else {
                    this_root_name = String::new();
                }
            }
            Some(root_name) => {
                this_root_name = start_quote.to_string() + root_name + end_quote;
            }
        }
        //
        // The origRawStackExtension is currently only used in MTF_FILTER_MDOC.  It comes
        // from MetaData - and only from interfaces which use a single raw stack (in other
        // words tomogram reconstruction).  It is the extension of the raw stack file the
        // user picked.
        let this_orig_raw_stack_extension: String;
        match orig_raw_image_stack_extension {
            None => {
                if build_regex {
                    // If the origRawStackExtension is missing use pattern match string
                    // for it
                    this_orig_raw_stack_extension = Variable::orig_raw_image_extension().regex;
                } else {
                    this_orig_raw_stack_extension = String::new();
                }
            }
            Some(orig_raw_image_stack_extension) => {
                this_orig_raw_stack_extension = start_quote.to_string()
                    + extension::EXTENSION_DIVIDER
                    + &orig_raw_image_stack_extension.to_string()
                    + end_quote;
            }
        }
        let mut axis_id_extension = String::new();
        if !build_regex || axis_id.is_some() {
            // Java calls `axisID.getExtension()` unconditionally here, so a null axisID
            // with buildRegex off is a NullPointerException in the source.
            axis_id_extension = axis_id
                .map(|axis_id| axis_id.get_extension())
                .unwrap_or_default();
        } else if axis_type == Some(AxisType::DualAxis) {
            axis_id_extension = Variable::axis().dual_regex.unwrap_or_default();
        } else if axis_type.is_none() {
            // match both single and dual.
            axis_id_extension = Variable::axis().regex;
        }
        FileNameBuilder {
            builder,
            build_regex,
            first_axis_type: this_first_axis_type,
            root_name: this_root_name,
            orig_raw_stack_extension: this_orig_raw_stack_extension,
            formatted_numeric1: formatted_numeric1.map(|value| value.to_string()),
            formatted_numeric2: formatted_numeric2.map(|value| value.to_string()),
            start_quote,
            end_quote,
            axis_id_extension,
            numeric_index: 0,
            end_regex: "$".to_string(),
        }
    }

    /// Java `getRootName`.
    fn get_root_name(&self) -> &str {
        &self.root_name
    }

    /// Java `startSecondPattern`.
    fn start_second_pattern(&mut self) -> bool {
        if self.first_axis_type.is_none() || self.first_axis_type == Some(AxisType::NotSet) {
            // There is no non-capturing group set up - do not add a second pattern.
            return false;
        }
        self.builder.push('|');
        // End pattern and non-capturing group when done is called.
        self.end_regex = "$)".to_string();
        if self.first_axis_type == Some(AxisType::DualAxis) {
            self.axis_id_extension = String::new();
        } else if self.first_axis_type == Some(AxisType::SingleAxis) {
            self.axis_id_extension = Variable::axis().dual_regex.unwrap_or_default();
        }
        true
    }

    /// Java `append(Variable)`.
    fn append_variable(&mut self, variable: &Variable) {
        if variable.is_numeric() {
            if self.numeric_index == 0 && self.formatted_numeric1.is_some() {
                self.numeric_index += 1;
                let value = self.start_quote.to_string()
                    + self.formatted_numeric1.as_deref().unwrap()
                    + self.end_quote;
                self.builder.push_str(&value);
            } else if self.numeric_index == 1 && self.formatted_numeric2.is_some() {
                self.numeric_index += 1;
                let value = self.start_quote.to_string()
                    + self.formatted_numeric2.as_deref().unwrap()
                    + self.end_quote;
                self.builder.push_str(&value);
            } else if self.build_regex {
                // Use the variable's pattern match string if the corresponding parameter
                // is not set.
                self.builder.push_str(&variable.regex);
            }
        } else if *variable == Variable::dataset() {
            let value = self.root_name.clone();
            self.builder.push_str(&value);
        } else if *variable == Variable::axis() {
            let value = self.axis_id_extension.clone();
            self.builder.push_str(&value);
        } else if *variable == Variable::dataset_and_axis() {
            let root_name = self.root_name.clone();
            self.builder.push_str(&root_name);
            let axis_id_extension = self.axis_id_extension.clone();
            self.builder.push_str(&axis_id_extension);
        } else if *variable == Variable::orig_raw_image_extension() {
            let value = self.orig_raw_stack_extension.clone();
            self.builder.push_str(&value);
        }
    }

    /// Java `append(String)`.
    fn append_string(&mut self, string: Option<&str>) {
        if let Some(string) = string {
            if !string.is_empty() {
                let value = self.start_quote.to_string() + string + self.end_quote;
                self.builder.push_str(&value);
            }
        }
    }

    /// Java `done`.
    fn done(&mut self) {
        if self.build_regex {
            let end_regex = self.end_regex.clone();
            self.builder.push_str(&end_regex);
        }
    }
}

/// Java `toString` on the nested `FileNameBuilder` class.
impl std::fmt::Display for FileNameBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.builder)
    }
}

/// `java.util.regex.Pattern.quote(String)`: wraps the literal in `\Q` ... `\E`, doubling
/// any `\E` the literal itself contains.
fn java_util_regex_pattern_quote(literal: &str) -> String {
    let mut quoted = String::from("\\Q");
    let mut rest = literal;
    while let Some(index) = rest.find("\\E") {
        quoted.push_str(&rest[..index]);
        quoted.push_str("\\E\\\\E\\Q");
        rest = &rest[index + 2..];
    }
    quoted.push_str(rest);
    quoted.push_str("\\E");
    quoted
}

/// `java.lang.String.matches(String)`, which anchors at both ends.  An invalid pattern
/// throws `PatternSyntaxException` in Java; here it returns false.
///
/// Java's `\Q`...`\E` literal quoting has no counterpart in the `regex` crate, and
/// every `equals` overload in this unit builds its pattern out of `Pattern.quote`, so
/// the quoted spans are expanded to escaped literals before the pattern is compiled.
/// The `\E\\E\Q` sequence `Pattern.quote` emits for a literal `\E` passes straight
/// through, because outside a quoted span `\\E` already means "backslash then E" in
/// both dialects.
///
/// One dialect difference remains: Java's `\s`, `\d` and `\w` are ASCII-only, while
/// the crate's are Unicode-aware.  That shows only for non-ASCII input; the patterns
/// this unit builds are file names.
fn java_lang_string_matches(string: &str, pattern: &str) -> bool {
    let mut translated = String::with_capacity(pattern.len());
    let mut rest = pattern;
    while let Some(index) = rest.find("\\Q") {
        translated.push_str(&rest[..index]);
        let body = &rest[index + 2..];
        match body.find("\\E") {
            Some(end) => {
                translated.push_str(&regex::escape(&body[..end]));
                rest = &body[end + 2..];
            }
            None => {
                // An unterminated \Q quotes to the end of the pattern.
                translated.push_str(&regex::escape(body));
                rest = "";
            }
        }
    }
    translated.push_str(rest);
    match regex::Regex::new(&format!("\\A(?:{})\\z", translated)) {
        Ok(regex) => regex.is_match(string),
        Err(_) => false,
    }
}

/// `IMOD/Etomo/src/etomo/type/DirectoryType.java`.  A package-private `FileType`
/// subclass, held here by composition as `FileType` holds `FileKey`.
#[derive(Debug)]
pub struct DirectoryType {
    /// The `FileType` superclass state.
    file_type: Arc<FileType>,
}

impl std::ops::Deref for DirectoryType {
    type Target = FileType;
    fn deref(&self) -> &FileType {
        &self.file_type
    }
}

impl DirectoryType {
    /// Java `NAD_SUBDIR`.
    pub fn nad_subdir() -> Arc<DirectoryType> {
        DirectoryType::construct_instance(vec![
            PatternElement::Str("naddir".to_string()),
            PatternElement::ExtensionMarker(ExtensionMarker::Generic),
            PatternElement::Variable(Variable::dataset()),
        ])
    }

    /// Java `DirectoryType(Object[])`.
    fn new(pattern: Vec<PatternElement>) -> Arc<DirectoryType> {
        Arc::new(DirectoryType {
            file_type: FileType::new_with_pattern(None, Some(pattern), None, None, None, None),
        })
    }

    /// Java `constructInstance(Object[])`.
    fn construct_instance(pattern: Vec<PatternElement>) -> Arc<DirectoryType> {
        DirectoryType::new(pattern)
    }
}

#[cfg(test)]
mod statics_tests {
    use super::*;

    /// One line per `public static final FileType` singleton, in declaration order,
    /// in the format the reference JVM harness prints through reflection.  Run with
    /// `IMOD_RS_ETOMO_PROBE` set to dump the table for a fresh comparison; the
    /// committed assertions below are the lines that pin what the generator-built
    /// declarations had to get right.
    fn dump(name: &str, ft: &Arc<FileType>) -> String {
        fn opt(value: Option<&str>) -> String {
            match value {
                None => "null".to_string(),
                Some(value) => value.to_string(),
            }
        }
        fn brief(ft: &Option<Arc<FileType>>) -> String {
            match ft {
                None => "null".to_string(),
                Some(ft) => ft.to_string(),
            }
        }
        format!(
            "{}\t{}|key={}|key2={}|descr={}|usesDataset={}|usesAxisID={}|typeString={}\
|extension={}|composite={}|inSubdirectory={}|unnamed={}|template={}\
|inImodSubdirectory={}|versioned={}|templateOnlyMiddlePiece={}|imageFile={}\
|expandedImageExtensionSet={}|dynamicLocation={}|subFileType={}|singleFileType={}\
|dualFileType={}|subdir={}|directory={}|parentFileType={}",
            name,
            ft,
            opt(ft.file_key.get_imod_manager_key()),
            opt(ft.file_key.get_imod_manager_key2()),
            opt(ft.file_key.get_descr()),
            ft.uses_dataset,
            ft.uses_axis_id,
            opt(ft.type_string.as_deref()),
            opt(ft.extension.as_deref()),
            ft.composite,
            ft.in_subdirectory,
            ft.unnamed,
            ft.template,
            opt(ft.in_imod_subdirectory.as_deref()),
            ft.versioned,
            opt(ft.template_only_middle_piece.as_deref()),
            ft.image_file,
            ft.expanded_image_extension_set,
            match ft.dynamic_location {
                None => "null",
                Some(DynamicLocation::AfterExtension) => "AFTER_EXTENSION",
                Some(DynamicLocation::AfterTypeString) => "AFTER_TYPE_STRING",
                Some(DynamicLocation::AfterTypeStringAndAfterMiddlePiece) =>
                    "AFTER_TYPE_STRING_AND_AFTER_MIDDLE_PIECE",
            },
            brief(&ft.sub_file_type),
            brief(&ft.single_file_type),
            brief(&ft.dual_file_type),
            brief(&ft.subdir),
            match &ft.directory {
                None => "null".to_string(),
                Some(directory) => directory.to_string(),
            },
            match ft.parent_file_type.lock().unwrap().upgrade() {
                None => "null".to_string(),
                Some(parent) => parent.to_string(),
            },
        )
    }

    fn table() -> Vec<String> {
        vec![
            dump("ORIG_COMS_DIR", &CLASS.orig_coms_dir),
            dump("FIDUCIAL_3D_MODEL", &CLASS.fiducial_3d_model),
            dump("SERIES_WATCHER_PROJECT", &CLASS.series_watcher_project),
            dump(
                "BATCH_RUN_TOMO_GLOBAL_AUTODOC",
                &CLASS.batch_run_tomo_global_autodoc,
            ),
            dump(
                "DEFAULT_BATCH_RUN_TOMO_AUTODOC",
                &CLASS.default_batch_run_tomo_autodoc,
            ),
            dump(
                "LOCAL_BATCH_DIRECTIVE_FILE",
                &CLASS.local_batch_directive_file,
            ),
            dump("LOCAL_SCOPE_TEMPLATE", &CLASS.local_scope_template),
            dump("LOCAL_SYSTEM_TEMPLATE", &CLASS.local_system_template),
            dump("LOCAL_USER_TEMPLATE", &CLASS.local_user_template),
            dump("ALIGNED_STACK_OLD", &CLASS.aligned_stack_old),
            dump("ALIGNED_STACK", &CLASS.aligned_stack),
            dump(
                "NEWST_OR_BLEND_3D_FIND_OUTPUT_OLD",
                &CLASS.newst_or_blend_3d_find_output_old,
            ),
            dump(
                "NEWST_OR_BLEND_3D_FIND_OUTPUT",
                &CLASS.newst_or_blend_3d_find_output,
            ),
            dump("CTF_CORRECTED_STACK_OLD", &CLASS.ctf_corrected_stack_old),
            dump("CTF_CORRECTED_STACK", &CLASS.ctf_corrected_stack),
            dump("ERASED_BEADS_STACK_OLD", &CLASS.erased_beads_stack_old),
            dump("ERASED_BEADS_STACK", &CLASS.erased_beads_stack),
            dump("MTF_FILTERED_STACK_OLD", &CLASS.mtf_filtered_stack_old),
            dump("MTF_FILTERED_STACK", &CLASS.mtf_filtered_stack),
            dump(
                "TRANSFORMED_REFINING_MODEL",
                &CLASS.transformed_refining_model,
            ),
            dump("XCORR_BLEND_OUTPUT_OLD", &CLASS.xcorr_blend_output_old),
            dump("XCORR_BLEND_OUTPUT", &CLASS.xcorr_blend_output),
            dump("CHECK_FILE", &CLASS.check_file),
            dump(
                "SERIES_WATCHER_CHECK_FILE",
                &CLASS.series_watcher_check_file,
            ),
            dump("ALIGN_COMSCRIPT", &CLASS.align_comscript),
            dump("AUTOFIDSEED_COMSCRIPT", &CLASS.autofidseed_comscript),
            dump("BATCH_RUN_TOMO_COMSCRIPT", &CLASS.batch_run_tomo_comscript),
            dump("BLEND_COMSCRIPT", &CLASS.blend_comscript),
            dump("COPYTOMOCOMS_COMSCRIPT", &CLASS.copytomocoms_comscript),
            dump("CRYO_POSITION_COMSCRIPT", &CLASS.cryo_position_comscript),
            dump("CTF_3D_SETUP_COMSCRIPT", &CLASS.ctf_3d_setup_comscript),
            dump("CTF_CORRECTION_COMSCRIPT", &CLASS.ctf_correction_comscript),
            dump("FIND_BEADS_3D_COMSCRIPT", &CLASS.find_beads_3d_comscript),
            dump("FLATTEN_COMSCRIPT", &CLASS.flatten_comscript),
            dump("FLATTEN_TOOL_COMSCRIPT", &CLASS.flatten_tool_comscript),
            dump("GOLD_ERASER_COMSCRIPT", &CLASS.gold_eraser_comscript),
            dump(
                "MULTIFILT_SETUP_COMSCRIPT",
                &CLASS.multifilt_setup_comscript,
            ),
            dump("MTF_FILTER_COMSCRIPT", &CLASS.mtf_filter_comscript),
            dump("NEWST_COMSCRIPT", &CLASS.newst_comscript),
            dump("PREBLEND_COMSCRIPT", &CLASS.preblend_comscript),
            dump("SIRTSETUP_COMSCRIPT", &CLASS.sirtsetup_comscript),
            dump("SLOPPY_BLEND_COMSCRIPT", &CLASS.sloppy_blend_comscript),
            dump("TILT_COMSCRIPT", &CLASS.tilt_comscript),
            dump("TILT_FOR_SIRT_COMSCRIPT", &CLASS.tilt_for_sirt_comscript),
            dump(
                "TILT_FOR_POS_SAMPLE_COMSCRIPT",
                &CLASS.tilt_for_pos_sample_comscript,
            ),
            dump("TRACK_COMSCRIPT", &CLASS.track_comscript),
            dump("TRACK_ADJUSTED_COMSCRIPT", &CLASS.track_adjusted_comscript),
            dump("TRACK_ORIG_COMSCRIPT", &CLASS.track_orig_comscript),
            dump(
                "CROSS_CORRELATION_COMSCRIPT",
                &CLASS.cross_correlation_comscript,
            ),
            dump("PATCH_TRACKING_COMSCRIPT", &CLASS.patch_tracking_comscript),
            dump("DIRECTIVES_DESCR", &CLASS.directives_descr),
            dump(
                "JOIN_WARP_2_MODEL_COMSCRIPT",
                &CLASS.join_warp_2_model_comscript,
            ),
            dump(
                "ALIGN_FRAMES_OUTPUT_COMSCRIPT",
                &CLASS.align_frames_output_comscript,
            ),
            dump("SUBTOMO_SETUP_COMSCRIPT", &CLASS.subtomo_setup_comscript),
            dump("ALT_TOMO_SETUP_COMSCRIPT", &CLASS.alt_tomo_setup_comscript),
            dump("RESTRICT_ALIGN_COMSCRIPT", &CLASS.restrict_align_comscript),
            dump(
                "REDUCE_FILT_VOL_COMSCRIPT",
                &CLASS.reduce_filt_vol_comscript,
            ),
            dump("SERIES_WATCHER_COMSCRIPT", &CLASS.series_watcher_comscript),
            dump(
                "DISTORTION_CORRECTED_STACK_OLD",
                &CLASS.distortion_corrected_stack_old,
            ),
            dump(
                "DISTORTION_CORRECTED_STACK",
                &CLASS.distortion_corrected_stack,
            ),
            dump("AUTOFIDSEED_DIR", &CLASS.autofidseed_dir),
            dump("BATCH_RUN_TOMO_PROJECT", &CLASS.batch_run_tomo_project),
            dump("PIECE_SHIFTS", &CLASS.piece_shifts),
            dump("MANUAL_REPLACEMENT_MODEL", &CLASS.manual_replacement_model),
            dump("FIDUCIAL_MODEL", &CLASS.fiducial_model),
            dump(
                "CCD_ERASER_BEADS_INPUT_MODEL",
                &CLASS.ccd_eraser_beads_input_model,
            ),
            dump("FIDUCIAL_NO_GAPS_MODEL", &CLASS.fiducial_no_gaps_model),
            dump(
                "FIDUCIAL_PATCH_TRACKING_MODEL",
                &CLASS.fiducial_patch_tracking_model,
            ),
            dump("FLATTEN_TOOL_OUTPUT_OLD", &CLASS.flatten_tool_output_old),
            dump("FLATTEN_TOOL_OUTPUT", &CLASS.flatten_tool_output),
            dump("EXCLUDE_VIEWS_INFO", &CLASS.exclude_views_info),
            dump("NAD_TEST_INPUT_OLD", &CLASS.nad_test_input_old),
            dump("NAD_TEST_INPUT", &CLASS.nad_test_input),
            dump("JOIN_OLD", &CLASS.join_old),
            dump("JOIN", &CLASS.join),
            dump("MODELED_JOIN_OLD", &CLASS.modeled_join_old),
            dump("MODELED_JOIN", &CLASS.modeled_join),
            dump("TRIAL_JOIN_OLD", &CLASS.trial_join_old),
            dump("TRIAL_JOIN", &CLASS.trial_join),
            dump("BATCH_RUN_TOMO_LOG", &CLASS.batch_run_tomo_log),
            dump("ALT_TOMO_SETUP_LOG", &CLASS.alt_tomo_setup_log),
            dump(
                "BATCH_RUN_TOMO_DATASET_LOG",
                &CLASS.batch_run_tomo_dataset_log,
            ),
            dump("CTF_3D_FINISH_LOG", &CLASS.ctf_3d_finish_log),
            dump("CTF_3D_SETUP_LOG", &CLASS.ctf_3d_setup_log),
            dump("ERASER_LOG", &CLASS.eraser_log),
            dump("TILT_ALIGN_LOG", &CLASS.tilt_align_log),
            dump("GPU_TEST_LOG", &CLASS.gpu_test_log),
            dump("PREBLEND_LOG", &CLASS.preblend_log),
            dump("PROJECT_LOG", &CLASS.project_log),
            dump(
                "SERIES_WATCHER_BRT_ROW_LOG",
                &CLASS.series_watcher_brt_row_log,
            ),
            dump("ALIGN_ANGLES_LOG", &CLASS.align_angles_log),
            dump("ALIGN_ERROR_LOG", &CLASS.align_error_log),
            dump("ALIGN_ROBUST_LOG", &CLASS.align_robust_log),
            dump("ALIGN_SOLUTION_LOG", &CLASS.align_solution_log),
            dump("CROSS_CORRELATION_LOG", &CLASS.cross_correlation_log),
            dump("ALIGN_FRAMES_LOG", &CLASS.align_frames_log),
            dump("RESTRICT_ALIGN_LOG", &CLASS.restrict_align_log),
            dump("SUBTOMO_SETUP_LOG", &CLASS.subtomo_setup_log),
            dump("CTF_CORRECTION_LOG", &CLASS.ctf_correction_log),
            dump("GOLD_ERASER_LOG", &CLASS.gold_eraser_log),
            dump("MTF_FILTER_LOG", &CLASS.mtf_filter_log),
            dump("REDUCE_FILT_VOL_LOG", &CLASS.reduce_filt_vol_log),
            dump("SERIES_WATCHER_LOG", &CLASS.series_watcher_log),
            dump("MTF_FILTER_MDOC", &CLASS.mtf_filter_mdoc),
            dump(
                "FIND_BEADS_3D_OUTPUT_MODEL",
                &CLASS.find_beads_3d_output_model,
            ),
            dump(
                "AUTOFIDSEED_BOUNDARY_MODEL",
                &CLASS.autofidseed_boundary_model,
            ),
            dump(
                "AUTO_ALIGN_BOUNDARY_MODEL",
                &CLASS.auto_align_boundary_model,
            ),
            dump(
                "SMOOTHING_ASSESSMENT_OUTPUT_MODEL",
                &CLASS.smoothing_assessment_output_model,
            ),
            dump(
                "CLUSTERED_ELONGATED_MODEL",
                &CLASS.clustered_elongated_model,
            ),
            dump("FLATTEN_WARP_INPUT_MODEL", &CLASS.flatten_warp_input_model),
            dump("PATCH_VECTOR_MODEL", &CLASS.patch_vector_model),
            dump("PATCH_VECTOR_CCC_MODEL", &CLASS.patch_vector_ccc_model),
            dump(
                "PATCH_TRACKING_BOUNDARY_MODEL",
                &CLASS.patch_tracking_boundary_model,
            ),
            dump(
                "BATCH_RUN_TOMO_BOUNDARY_MODEL",
                &CLASS.batch_run_tomo_boundary_model,
            ),
            dump("TOMOPITCH_MODEL", &CLASS.tomopitch_model),
            dump("ALIGNED_STACK_MRC_OLD", &CLASS.aligned_stack_mrc_old),
            dump("ALIGNED_STACK_MRC", &CLASS.aligned_stack_mrc),
            dump("PREBLEND_OUTPUT_MRC_OLD", &CLASS.preblend_output_mrc_old),
            dump("PREBLEND_OUTPUT_MRC", &CLASS.preblend_output_mrc),
            dump(
                "MUTLIFILT_EXACT_OBJECT_SIZES_OUTPUT_TEMPLATE_OLD",
                &CLASS.mutlifilt_exact_object_sizes_output_template_old,
            ),
            dump(
                "MUTLIFILT_EXACT_OBJECT_SIZES_OUTPUT_TEMPLATE",
                &CLASS.mutlifilt_exact_object_sizes_output_template,
            ),
            dump(
                "MUTLIFILT_GAUSSIAN_OUTPUT_TEMPLATE_OLD",
                &CLASS.mutlifilt_gaussian_output_template_old,
            ),
            dump(
                "MUTLIFILT_GAUSSIAN_OUTPUT_TEMPLATE",
                &CLASS.mutlifilt_gaussian_output_template,
            ),
            dump(
                "MUTLIFILT_HAMMING_LIKE_STARTS_OUTPUT_TEMPLATE_OLD",
                &CLASS.mutlifilt_hamming_like_starts_output_template_old,
            ),
            dump(
                "MUTLIFILT_HAMMING_LIKE_STARTS_OUTPUT_TEMPLATE",
                &CLASS.mutlifilt_hamming_like_starts_output_template,
            ),
            dump(
                "MUTLIFILT_FAKE_SIRT_ITERATIONS_OUTPUT_TEMPLATE_OLD",
                &CLASS.mutlifilt_fake_sirt_iterations_output_template_old,
            ),
            dump(
                "MUTLIFILT_FAKE_SIRT_ITERATIONS_OUTPUT_TEMPLATE",
                &CLASS.mutlifilt_fake_sirt_iterations_output_template,
            ),
            dump("TEST_NAD", &CLASS.test_nad),
            dump(
                "ANISOTROPIC_DIFFUSION_OUTPUT_OLD",
                &CLASS.anisotropic_diffusion_output_old,
            ),
            dump(
                "ANISOTROPIC_DIFFUSION_OUTPUT",
                &CLASS.anisotropic_diffusion_output,
            ),
            dump("PIECE_LIST", &CLASS.piece_list),
            dump("PREALIGNED_STACK_OLD", &CLASS.prealigned_stack_old),
            dump("PREALIGNED_STACK", &CLASS.prealigned_stack),
            dump("PRE_TRANSFORMATION_LIST", &CLASS.pre_transformation_list),
            dump("PRE_XG", &CLASS.pre_xg),
            dump("MATLAB_PARAM_FILE", &CLASS.matlab_param_file),
            dump("RAW_TILT_ANGLES", &CLASS.raw_tilt_angles),
            dump("TRIM_VOL_OUTPUT_OLD", &CLASS.trim_vol_output_old),
            dump("TRIM_VOL_OUTPUT", &CLASS.trim_vol_output),
            dump("PROCESSCHUNKS_REC", &CLASS.processchunks_rec),
            dump("PROCESSCHUNKS_MRC", &CLASS.processchunks_mrc),
            dump("PROCESSCHUNKS_VOL_MRC", &CLASS.processchunks_vol_mrc),
            dump("TILT_3D_FIND_OUTPUT_OLD", &CLASS.tilt_3d_find_output_old),
            dump("TILT_3D_FIND_OUTPUT", &CLASS.tilt_3d_find_output),
            dump("BOTTOM_SAMPLE_OLD", &CLASS.bottom_sample_old),
            dump("BOTTOM_SAMPLE", &CLASS.bottom_sample),
            dump("CRYO_POSITION_OUTPUT_OLD", &CLASS.cryo_position_output_old),
            dump("CRYO_POSITION_OUTPUT", &CLASS.cryo_position_output),
            dump("TILT_OUTPUT_DUAL_OLD", &CLASS.tilt_output_dual_old),
            dump("TILT_OUTPUT_SINGLE_OLD", &CLASS.tilt_output_single_old),
            dump("TILT_OUTPUT_SINGLE", &CLASS.tilt_output_single),
            dump("TILT_OUTPUT_DUAL", &CLASS.tilt_output_dual),
            dump("TILT_OUTPUT_OLD", &CLASS.tilt_output_old),
            dump("TILT_OUTPUT", &CLASS.tilt_output),
            dump("CTF_3D_OUTPUT", &CLASS.ctf_3d_output),
            dump("FLATTEN_OUTPUT_OLD", &CLASS.flatten_output_old),
            dump("FLATTEN_OUTPUT", &CLASS.flatten_output),
            dump("MIDDLE_SAMPLE_OLD", &CLASS.middle_sample_old),
            dump("MIDDLE_SAMPLE", &CLASS.middle_sample),
            dump("COMBINED_VOLUME_OLD", &CLASS.combined_volume_old),
            dump("COMBINED_VOLUME", &CLASS.combined_volume),
            dump("TOP_SAMPLE_OLD", &CLASS.top_sample_old),
            dump("TOP_SAMPLE", &CLASS.top_sample),
            dump("JOIN_SAMPLE_AVERAGES_OLD", &CLASS.join_sample_averages_old),
            dump("JOIN_SAMPLE_AVERAGES", &CLASS.join_sample_averages),
            dump("JOIN_SAMPLE_OLD", &CLASS.join_sample_old),
            dump("JOIN_SAMPLE", &CLASS.join_sample),
            dump("SEED_MODEL", &CLASS.seed_model),
            dump(
                "SIRT_SCALED_OUTPUT_TEMPLATE_OLD",
                &CLASS.sirt_scaled_output_template_old,
            ),
            dump(
                "SIRT_SCALED_OUTPUT_TEMPLATE",
                &CLASS.sirt_scaled_output_template,
            ),
            dump(
                "SIRT_SUBAREA_SCALED_OUTPUT_TEMPLATE_OLD",
                &CLASS.sirt_subarea_scaled_output_template_old,
            ),
            dump(
                "SIRT_SUBAREA_SCALED_OUTPUT_TEMPLATE",
                &CLASS.sirt_subarea_scaled_output_template,
            ),
            dump("SQUEEZE_VOL_OUTPUT_OLD", &CLASS.squeeze_vol_output_old),
            dump("SQUEEZE_VOL_OUTPUT", &CLASS.squeeze_vol_output),
            dump("SIRT_OUTPUT_TEMPLATE_OLD", &CLASS.sirt_output_template_old),
            dump("SIRT_OUTPUT_TEMPLATE", &CLASS.sirt_output_template),
            dump(
                "SIRT_SUBAREA_OUTPUT_TEMPLATE_OLD",
                &CLASS.sirt_subarea_output_template_old,
            ),
            dump(
                "SIRT_SUBAREA_OUTPUT_TEMPLATE",
                &CLASS.sirt_subarea_output_template,
            ),
            dump("RAW_STACK", &CLASS.raw_stack),
            dump("STATS_LOG_OLD", &CLASS.stats_log_old),
            dump("PROCESSCHUNKS_LOG", &CLASS.processchunks_log),
            dump("STATS_LOG", &CLASS.stats_log),
            dump("FIXED_XRAYS_STACK", &CLASS.fixed_xrays_stack),
            dump("FIXED_STATS_LOG_OLD", &CLASS.fixed_stats_log_old),
            dump("FIXED_STATS_LOG", &CLASS.fixed_stats_log),
            dump("ORIGINAL_RAW_STACK", &CLASS.original_raw_stack),
            dump("FULL_VSR", &CLASS.full_vsr),
            dump("SUB_VSR", &CLASS.sub_vsr),
            dump("TILT_ANGLES", &CLASS.tilt_angles),
            dump("WARP_XG", &CLASS.warp_xg),
            dump("EDGE_FUNCTIONS_X", &CLASS.edge_functions_x),
            dump(
                "LOCAL_TRANSFORMATION_LIST",
                &CLASS.local_transformation_list,
            ),
            dump(
                "AUTO_LOCAL_TRANSFORMATION_LIST",
                &CLASS.auto_local_transformation_list,
            ),
            dump(
                "EMPTY_LOCAL_TRANSFORMATION_LIST",
                &CLASS.empty_local_transformation_list,
            ),
            dump(
                "MIDAS_LOCAL_TRANSFORMATION_LIST",
                &CLASS.midas_local_transformation_list,
            ),
            dump(
                "GLOBAL_TRANSFORMATION_LIST",
                &CLASS.global_transformation_list,
            ),
            dump(
                "ALT_STACK_ROOTNAME_EVEN_FILE",
                &CLASS.alt_stack_rootname_even_file,
            ),
            dump(
                "ALT_STACK_ROOTNAME_ODD_FILE",
                &CLASS.alt_stack_rootname_odd_file,
            ),
            dump("ALT_STACK_EVEN_TOMOGRAM", &CLASS.alt_stack_even_tomogram),
            dump("ALT_STACK_ODD_TOMOGRAM", &CLASS.alt_stack_odd_tomogram),
            dump(
                "ALT_STACK_EVEN_FULL_TOMOGRAM",
                &CLASS.alt_stack_even_full_tomogram,
            ),
            dump(
                "ALT_STACK_ODD_FULL_TOMOGRAM",
                &CLASS.alt_stack_odd_full_tomogram,
            ),
            dump("ALT_STACK_TOMOGRAM", &CLASS.alt_stack_tomogram),
            dump(
                "REDUCE_FILT_VOL_OUTPUT_FILE",
                &CLASS.reduce_filt_vol_output_file,
            ),
            dump(
                "FLATTEN_REDUCE_FILT_VOL_FILE",
                &CLASS.flatten_reduce_filt_vol_file,
            ),
        ]
    }

    /// Values captured byte-for-byte from a reflection harness run against the
    /// reference JVM (see the module header of `tests/etomo_file_type_probe.rs` for how
    /// that runtime is built).  The whole 205-line table was diffed and matched, apart
    /// from four lines whose Java text is an `Object.toString()` identity hash - the
    /// documented non-achievable - so these are the representative lines kept here,
    /// one per constructor shape the declarations use.
    #[test]
    fn jvm_verified_statics() {
        assert_eq!(table().len(), 205);
        assert_eq!(
            dump("ORIG_COMS_DIR", &CLASS.orig_coms_dir),
            "ORIG_COMS_DIR	origcoms|key=null|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=origcoms|extension=|composite=false|inSubdirectory=false|unnamed=false|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("FIDUCIAL_3D_MODEL", &CLASS.fiducial_3d_model),
            "FIDUCIAL_3D_MODEL	.3dmod|key=fiducial model|key2=null|descr=null|usesDataset=true|usesAxisID=true|typeString=|extension=.3dmod|composite=false|inSubdirectory=false|unnamed=false|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("SERIES_WATCHER_PROJECT", &CLASS.series_watcher_project),
            "SERIES_WATCHER_PROJECT	.ebt.active|key=null|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump(
                "DEFAULT_BATCH_RUN_TOMO_AUTODOC",
                &CLASS.default_batch_run_tomo_autodoc
            ),
            "DEFAULT_BATCH_RUN_TOMO_AUTODOC	batchDefaults.adoc|key=null|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=batchDefaults|extension=.adoc|composite=false|inSubdirectory=false|unnamed=false|template=false|inImodSubdirectory=com|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("ALIGNED_STACK", &CLASS.aligned_stack),
            "ALIGNED_STACK	.ali|key=fine aligned|key2=null|descr=the final aligned stack|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump(
                "NEWST_OR_BLEND_3D_FIND_OUTPUT",
                &CLASS.newst_or_blend_3d_find_output
            ),
            "NEWST_OR_BLEND_3D_FIND_OUTPUT	_3dfind.ali|key=fine aligned for findbeads3d|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("CTF_CORRECTED_STACK", &CLASS.ctf_corrected_stack),
            "CTF_CORRECTED_STACK	_ctfcorr.ali|key=CtfCorrection|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("RAW_STACK", &CLASS.raw_stack),
            "RAW_STACK	.st|key=raw stack|key2=preview|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("STATS_LOG", &CLASS.stats_log),
            "STATS_LOG	_stats.log|key=join|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("FIXED_XRAYS_STACK", &CLASS.fixed_xrays_stack),
            "FIXED_XRAYS_STACK	_fixed.st|key=erased stack|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("ORIGINAL_RAW_STACK", &CLASS.original_raw_stack),
            "ORIGINAL_RAW_STACK	_orig.st|key=null|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("PREALIGNED_STACK", &CLASS.prealigned_stack),
            "PREALIGNED_STACK	.preali|key=coarse aligned|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("TILT_OUTPUT", &CLASS.tilt_output),
            "TILT_OUTPUT	.rec / _full.rec|key=full volume|key2=null|descr=the tomogram|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("TRIM_VOL_OUTPUT", &CLASS.trim_vol_output),
            "TRIM_VOL_OUTPUT	.rec|key=trimmed volume|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("JOIN", &CLASS.join),
            "JOIN	.join|key=join|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("TRIAL_JOIN", &CLASS.trial_join),
            "TRIAL_JOIN	_trial.join|key=TrialJoinKey|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("MODELED_JOIN", &CLASS.modeled_join),
            "MODELED_JOIN	_modeled.join|key=modeled join|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("JOIN_SAMPLE", &CLASS.join_sample),
            "JOIN_SAMPLE	.sample|key=joinSamples|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("JOIN_SAMPLE_AVERAGES", &CLASS.join_sample_averages),
            "JOIN_SAMPLE_AVERAGES	.sampavg|key=joinSampleAverages|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("NAD_TEST_INPUT", &CLASS.nad_test_input),
            "NAD_TEST_INPUT	test.input|key=TestVolume|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=naddir.|parentFileType=null"
        );
        assert_eq!(
            dump("PROCESSCHUNKS_LOG", &CLASS.processchunks_log),
            "PROCESSCHUNKS_LOG	-.log|key=null|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("SQUEEZE_VOL_OUTPUT", &CLASS.squeeze_vol_output),
            "SQUEEZE_VOL_OUTPUT	.sqz|key=SqueezedVolume|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump(
                "CLUSTERED_ELONGATED_MODEL",
                &CLASS.clustered_elongated_model
            ),
            "CLUSTERED_ELONGATED_MODEL	clusterElong.mod|key=null|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=clusterElong|extension=.mod|composite=false|inSubdirectory=true|unnamed=false|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=autofidseed.dir|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump("GOLD_ERASER_COMSCRIPT", &CLASS.gold_eraser_comscript),
            "GOLD_ERASER_COMSCRIPT	golderaser.com|key=null|key2=null|descr=null|usesDataset=false|usesAxisID=true|typeString=golderaser|extension=.com|composite=false|inSubdirectory=false|unnamed=false|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
        assert_eq!(
            dump(
                "SERIES_WATCHER_CHECK_FILE",
                &CLASS.series_watcher_check_file
            ),
            "SERIES_WATCHER_CHECK_FILE	serieswatcher.cmds|key=null|key2=null|descr=null|usesDataset=false|usesAxisID=false|typeString=null|extension=null|composite=false|inSubdirectory=false|unnamed=true|template=false|inImodSubdirectory=null|versioned=false|templateOnlyMiddlePiece=null|imageFile=false|expandedImageExtensionSet=false|dynamicLocation=null|subFileType=null|singleFileType=null|dualFileType=null|subdir=null|directory=null|parentFileType=null"
        );
    }

    #[test]
    fn etomo_file_type_statics_probe() {
        if std::env::var("IMOD_RS_ETOMO_PROBE").is_err() {
            return;
        }
        for line in table() {
            println!("{}", line);
        }
    }
}
