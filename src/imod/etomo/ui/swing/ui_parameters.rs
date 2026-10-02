//! `IMOD/Etomo/src/etomo/ui/swing/UIParameters.java`.
//!
//! The font-size-dependent sizes of eTomo's fields and buttons.  A process-wide
//! singleton, as in Java (`instance`, guarded by `CONSTRUCT`).  The sizes feed Swing
//! layout only, but they are computed state that callers read, so they are kept.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};

use super::ui_utilities;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{Dimension, FontMetrics};
use crate::imod::etomo::util::utilities;

/// Java private static `instance` (created once; Java's `CONSTRUCT` lock is the
/// `OnceLock`).
static INSTANCE: OnceLock<UIParameters> = OnceLock::new();
/// Java `DEFAULT_FONT_SIZE`.
pub const DEFAULT_FONT_SIZE: f64 = 12.0;
/// Java private `DEFAULT_CHECKBOX_HEIGHT`.
const DEFAULT_CHECKBOX_HEIGHT: f64 = 21.0;
/// Java private `DEFAULT_TEXT_HEIGHT`.
const DEFAULT_TEXT_HEIGHT: i32 = 15;
/// Java private `DEFAULT_MAX_CHARS_WIDTH`: `Utilities.isMacOS() ? 125 : 139`.
static DEFAULT_MAX_CHARS_WIDTH: LazyLock<i32> =
    LazyLock::new(|| if utilities::is_mac_os() { 125 } else { 139 });
/// Java private `LONG_BUTTON_LINES`.
const LONG_BUTTON_LINES: [&str; 12] = [
    "Align SerialSections ",
    " Anisotropic Diffusion",
    " Replacement Model",
    "Track with Fiducial ",
    "Use RAPTOR Result ",
    "Delete Intermediate ",
    "Run Flattenwarp to ",
    " Values Test Results",
    "Run with Different ",
    " Iteration Test Results",
    " Alignment to Midas",
    " Transformed Model",
];
/// Java private static `adjustedButtonSize`.
static ADJUSTED_BUTTON_SIZE: AtomicBool = AtomicBool::new(false);
/// Java private static `buttonFontMetrics`.
static BUTTON_FONT_METRICS: Mutex<Option<FontMetrics>> = Mutex::new(None);

/// The instance fields of Java `UIParameters`.
struct Fields {
    dim_button: Dimension,
    dim_button_single_line: Dimension,
    dim_narrow_button: Dimension,
    dim_spinner: Dimension,
    dim_file_field: Dimension,
    dim_file_chooser: Dimension,
    font_size_adjustment: f64,
    numeric_width: i32,
    wide_numeric_width: i32,
    sections_width: i32,
    integer_triplet_width: i32,
    integer_doublet_width: i32,
    integer_width: i32,
    four_digit_width: i32,
    list_width: i32,
    file_width: i32,
    checkbox_height: f64,
}

/// Java `UIParameters`.
pub struct UIParameters {
    fields: Mutex<Fields>,
}

/// `java.awt.Dimension.setSize(double, double)` (a JDK method, not an eTomo one):
/// `width = (int) Math.ceil(width)`, same for height.
fn dimension_set_size(dimension: &mut Dimension, width: f64, height: f64) {
    dimension.width = width.ceil() as i32;
    dimension.height = height.ceil() as i32;
}

impl UIParameters {
    /// Java private `UIParameters()`.
    fn new() -> UIParameters {
        let zero = Dimension {
            width: 0,
            height: 0,
        };
        UIParameters {
            fields: Mutex::new(Fields {
                dim_button: zero,
                dim_button_single_line: zero,
                dim_narrow_button: zero,
                dim_spinner: zero,
                dim_file_field: zero,
                dim_file_chooser: zero,
                font_size_adjustment: 1.0,
                numeric_width: 0,
                wide_numeric_width: 0,
                sections_width: 0,
                integer_triplet_width: 0,
                integer_doublet_width: 0,
                integer_width: 0,
                four_digit_width: 40,
                list_width: 0,
                file_width: 0,
                checkbox_height: DEFAULT_CHECKBOX_HEIGHT,
            }),
        }
    }

    /// Java `createInstance(double)`.  Calculates sizes using the fontSize parameter.
    /// Creates the instance if necessary.
    pub fn create_instance(font_size: f64) {
        let instance = INSTANCE.get_or_init(UIParameters::new);
        instance.calc_sizes(font_size);
    }

    /// Java `getInstance()`.  Calls createInstance if the instance is null.  Never
    /// returns null.
    pub fn get_instance_void() -> &'static UIParameters {
        if INSTANCE.get().is_none() {
            UIParameters::create_instance(DEFAULT_FONT_SIZE);
        }
        INSTANCE.get().unwrap()
    }

    /// Java `getInstance(FontMetrics)`.  Returns the singleton instance with a better
    /// button size based on font metrics.
    pub fn get_instance_font_metrics(button_font_metrics: Option<FontMetrics>) -> &'static UIParameters {
        *BUTTON_FONT_METRICS.lock().unwrap() = button_font_metrics.clone();
        if INSTANCE.get().is_none() {
            UIParameters::create_instance(DEFAULT_FONT_SIZE);
            return INSTANCE.get().unwrap();
        }
        let instance = INSTANCE.get().unwrap();
        if !ADJUSTED_BUTTON_SIZE.load(Ordering::SeqCst) {
            if let Some(button_font_metrics) = button_font_metrics.as_ref() {
                instance.improve_button_size(Some(button_font_metrics));
            }
        }
        instance
    }

    /// Java `adjustSize(Dimension, int)`.  Returns fieldSize with the width based on
    /// newWidth and the font size adjustment.  If fieldSize is null it gets the height
    /// from folderButton in FixedDim.
    pub fn adjust_size(field_size: Option<Dimension>, new_width: i32) -> Dimension {
        let mut field_size = match field_size {
            Some(field_size) => field_size,
            None => {
                let mut field_size = Dimension {
                    width: 0,
                    height: 0,
                };
                field_size.height = ui_utilities::get_scaled_folder_button_dimension().height;
                field_size
            }
        };
        field_size.width = new_width
            * crate::imod::etomo::util::utilities::java_lang_math_round(
                UIParameters::get_instance_void().get_font_size_adjustment(),
            ) as i32;
        field_size
    }

    /// Java `needButtonFontMetrics()`.  Only the multi-line button size is
    /// recalculated based on font metrics.
    pub fn need_button_font_metrics() -> bool {
        !ADJUSTED_BUTTON_SIZE.load(Ordering::SeqCst)
    }

    /// Java `getButtonDimension()`.  Return the size of a standard button (a copy).
    pub fn get_button_dimension(&self) -> Dimension {
        self.fields.lock().unwrap().dim_button
    }

    /// Java `getButtonSingleLineDimension()`.
    pub fn get_button_single_line_dimension(&self) -> Dimension {
        self.fields.lock().unwrap().dim_button_single_line
    }

    /// Java `getNarrowButtonDimension()`.
    pub fn get_narrow_button_dimension(&self) -> Dimension {
        // Return a safe copy of the Dimension
        self.fields.lock().unwrap().dim_narrow_button
    }

    /// Java `getSpinnerDimension()`.
    pub fn get_spinner_dimension(&self) -> Dimension {
        self.fields.lock().unwrap().dim_spinner
    }

    /// Java `getFileFieldDimension()`.
    pub fn get_file_field_dimension(&self) -> Dimension {
        self.fields.lock().unwrap().dim_file_field
    }

    /// Java `getFileChooserDimension()`.
    pub fn get_file_chooser_dimension(&self) -> Dimension {
        self.fields.lock().unwrap().dim_file_chooser
    }

    /// Java `getNumericWidth()`.
    pub fn get_numeric_width(&self) -> i32 {
        self.fields.lock().unwrap().numeric_width
    }

    /// Java `getWideNumericWidth()`.
    pub fn get_wide_numeric_width(&self) -> i32 {
        self.fields.lock().unwrap().wide_numeric_width
    }

    /// Java `getSectionsWidth()`.
    pub fn get_sections_width(&self) -> i32 {
        self.fields.lock().unwrap().sections_width
    }

    /// Java `getIntegerTripletWidth()`.
    pub fn get_integer_triplet_width(&self) -> i32 {
        self.fields.lock().unwrap().integer_triplet_width
    }

    /// Java `getIntegerDoubletWidth()`.
    pub fn get_integer_doublet_width(&self) -> i32 {
        self.fields.lock().unwrap().integer_doublet_width
    }

    /// Java `getIntegerWidth()`.
    pub fn get_integer_width(&self) -> i32 {
        self.fields.lock().unwrap().integer_width
    }

    /// Java `getFourDigitWidth()`.
    pub fn get_four_digit_width(&self) -> i32 {
        self.fields.lock().unwrap().four_digit_width
    }

    /// Java `getListWidth()`.
    pub fn get_list_width(&self) -> i32 {
        self.fields.lock().unwrap().list_width
    }

    /// Java `getFileWidth()`.
    pub fn get_file_width(&self) -> i32 {
        self.fields.lock().unwrap().file_width
    }

    /// Java `getFontSizeAdjustment()`.  Get the amount to adjust a fields based on the
    /// current font size.
    pub fn get_font_size_adjustment(&self) -> f64 {
        self.fields.lock().unwrap().font_size_adjustment
    }

    /// Java private `calcSizes(double)`.  Sets size of objects given the current UI
    /// state.
    fn calc_sizes(&self, font_size: f64) {
        ADJUSTED_BUTTON_SIZE.store(false, Ordering::SeqCst);
        {
            let mut f = self.fields.lock().unwrap();
            // Create a temporary check box and get its height
            if !ARGUMENTS.lock().unwrap().is_headless() {
                // Swing layout: new JCheckBox().getPreferredSize().getHeight() - the
                // stand-in has no preferred sizes, so the default height is used.
                f.checkbox_height = DEFAULT_CHECKBOX_HEIGHT;
            } else {
                f.checkbox_height = DEFAULT_CHECKBOX_HEIGHT;
            }
            f.font_size_adjustment = font_size / DEFAULT_FONT_SIZE;
            let (h, a) = (f.checkbox_height, f.font_size_adjustment);
            dimension_set_size(&mut f.dim_button, 7.0 * h * a, 2.0 * h * a);
            dimension_set_size(&mut f.dim_button_single_line, 7.0 * h * a, 1.25 * h * a);
            dimension_set_size(&mut f.dim_narrow_button, 4.0 * h * a, 1.25 * h * a);
            dimension_set_size(&mut f.dim_spinner, 2.0 * h * a, 1.05 * h * a);
            dimension_set_size(&mut f.dim_file_field, 20.0 * h * a, 2.0 * h * a);
            dimension_set_size(&mut f.dim_file_chooser, 400.0 * a, 400.0 * a);
            f.numeric_width = (40.0 * a) as i32;
            f.wide_numeric_width = (50.0 * a) as i32;
            f.sections_width = (75.0 * a) as i32;
            f.integer_triplet_width = (75.0 * a) as i32;
            f.integer_doublet_width = (50.0 * a) as i32;
            f.integer_width = (30.0 * a) as i32;
            f.four_digit_width = (40.0 * a) as i32;
            f.list_width = (140.0 * a) as i32;
            f.file_width = (210.0 * a) as i32;
        }
        // Adjust the button size if possible
        let button_font_metrics = BUTTON_FONT_METRICS.lock().unwrap().clone();
        if button_font_metrics.is_some() {
            // Java `instance.improveButtonSize(...)`: `instance` is this object.
            self.improve_button_size(button_font_metrics.as_ref());
        }
    }

    /// Java private `improveButtonSize(FontMetrics)`.  Create a better button size
    /// based on font metrics and the longest lines of text in the multiline buttons.
    fn improve_button_size(&self, font_metrics: Option<&FontMetrics>) {
        let Some(font_metrics) = font_metrics else {
            return;
        };
        ADJUSTED_BUTTON_SIZE.store(true, Ordering::SeqCst);
        let text_height = font_metrics.get_height();
        let height_adjustment = text_height as f64 / DEFAULT_TEXT_HEIGHT as f64;
        let mut max_chars_width = 0;
        for line in LONG_BUTTON_LINES.iter() {
            let chars: Vec<char> = line.chars().collect();
            max_chars_width = max_chars_width.max(font_metrics.chars_width(&chars, 0, chars.len()));
        }
        let width_adjustment = max_chars_width as f64 / *DEFAULT_MAX_CHARS_WIDTH as f64;
        let mut f = self.fields.lock().unwrap();
        let h = f.checkbox_height;
        dimension_set_size(&mut f.dim_button, 7.0 * h * width_adjustment, 2.0 * h * height_adjustment);
    }
}
