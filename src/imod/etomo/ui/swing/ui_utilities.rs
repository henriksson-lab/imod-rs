//! `IMOD/Etomo/src/etomo/ui/swing/UIUtilities.java`.
//!
//! Static helpers for sizing, aligning and choosing files.  Component sizes,
//! alignment, insets, backgrounds and icons are Swing layout/painting and are not
//! modelled by the stand-in (`jdk.rs`); each Java statement that reads or writes one
//! is kept as a `// Swing layout:` comment, and where Java adds a component's inset or
//! icon size into a computed width the comment says so.  Font metrics are modelled
//! (`JComponent::get_font_metrics`), since several fields derive text from them.

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Mutex;

use super::colors::{self, ColorUIResource};
use super::file_chooser::{self, FileChooser};
use super::fixed_dim;
use super::ui_parameters::{self, UIParameters};
use crate::imod::etomo::etomo_director::{self, EtomoDirector};
use crate::imod::etomo::jdk::{
    self, Color, ComponentKind, Dimension, FileFilter, FontMetrics, JComponent, JFileChooser,
};
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::util::utilities;

/// Java private `ESTIMATED_MENU_HEIGHT`.
const ESTIMATED_MENU_HEIGHT: i32 = 60;
/// Java private static `screenSize`.
static SCREEN_SIZE: Mutex<Option<Dimension>> = Mutex::new(None);
/// Java private static `folderButtonDimension`.
static FOLDER_BUTTON_DIMENSION: Mutex<Option<Dimension>> = Mutex::new(None);
/// Java `DEFAULT_TEXT_FIELD_HEIGHT`.  Default sizes for font size 14.
pub const DEFAULT_TEXT_FIELD_HEIGHT: i32 = 17;
/// Java `DEFAULT_COMBO_BOX_HEIGHT`.
pub const DEFAULT_COMBO_BOX_HEIGHT: i32 = 26;
/// Java `DEFAULT_SINGLE_LINE_BUTTON_HEIGHT`.
pub const DEFAULT_SINGLE_LINE_BUTTON_HEIGHT: i32 = 27;
/// Java `DEFAULT_PROGRESS_BAR_HEIGHT`.
pub const DEFAULT_PROGRESS_BAR_HEIGHT: i32 = 18;

/// Java `chooseFile(Component, File, BrowsingDirectory, FileFilter)`.
pub fn choose_file(
    component: Option<&Rc<JComponent>>,
    dir: Option<PathBuf>,
    browsing_directory: Option<Rc<dyn BrowsingDirectory>>,
    file_filter: Option<Rc<dyn FileFilter>>,
) -> Option<PathBuf> {
    let chooser = FileChooser::new_base_manager_file_browsing_directory(
        None,
        dir.as_deref(),
        browsing_directory.as_deref(),
    );
    chooser.set_dialog_title(Some(" File Chooser"));
    // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
    // .getFileChooserDimension()).
    let _ = UIParameters::get_instance_void().get_file_chooser_dimension();
    chooser.set_file_filter(file_filter);
    let return_val = chooser.show_open_dialog(component);
    let mut file = None;
    if return_val == file_chooser::APPROVE_OPTION {
        file = chooser.get_selected_file();
        if let (Some(browsing_directory), Some(file)) = (browsing_directory.as_ref(), file.as_ref())
        {
            browsing_directory.set_browsing_dir(file.parent());
        }
    }
    file
}

/// Java `scaleByFontSize(Dimension)`.  Returns a dimension that was scaled by the
/// ratio of the user font size with the default font size.  Does not return null
/// unless it receives a null parameter.
pub fn scale_by_font_size_dimension(dimension: Option<Dimension>) -> Option<Dimension> {
    let dimension = match dimension {
        Some(dimension) if etomo_director::EtomoDirector::is_user_preference_loaded() => dimension,
        other => return other,
    };
    let font_size = etomo_director::EtomoDirector::get_user_font_size();
    if font_size <= etomo_director::SCALE_IMAGES_ABOVE_FONT_SIZE
        && font_size >= etomo_director::SCALE_IMAGES_BELOW_FONT_SIZE
    {
        return Some(dimension);
    }
    // Scale by font size.
    let ratio = font_size as f32 / etomo_director::SCALE_IMAGES_ABOVE_FONT_SIZE as f32;
    let mut scaled_dimension = Dimension {
        width: 0,
        height: 0,
    };
    let width = java_math_round_f32(ratio * dimension.width as f32).max(1);
    let height = java_math_round_f32(ratio * dimension.height as f32).max(1);
    scaled_dimension.width = width;
    scaled_dimension.height = height;
    Some(scaled_dimension)
}

/// Java `getScaledFolderButtonDimension()` (synchronized).  Returns the folder button
/// dimension scaled based on font size.  Does not return null.
pub fn get_scaled_folder_button_dimension() -> Dimension {
    let mut folder_button_dimension = FOLDER_BUTTON_DIMENSION.lock().unwrap();
    if let Some(dimension) = *folder_button_dimension {
        return dimension;
    }
    let scaled_dimension = scale_by_font_size_dimension(Some(fixed_dim::folderButton)).unwrap();
    // Kept as in the Java: the dimension is cached only while the user preferences
    // are *not* loaded (`UIUtilities.java:93`).
    if !etomo_director::EtomoDirector::is_user_preference_loaded() {
        *folder_button_dimension = Some(scaled_dimension);
    }
    scaled_dimension
}

/// Java private `calcNewSize(Dimension, int, boolean, int)`.  Returns the preferred
/// size with the new width (optionally scaled).  If preferredSize is null, it returns a
/// size containing a scaled defaultHeight.
fn calc_new_size(
    preferred_size: Option<Dimension>,
    width: i32,
    adjust_by_font: bool,
    default_height: i32,
) -> Dimension {
    let mut preferred_size = match preferred_size {
        Some(preferred_size) => preferred_size,
        None => {
            let mut preferred_size = Dimension {
                width: 0,
                height: 0,
            };
            preferred_size.height = scale_by_font_size_int(default_height);
            preferred_size
        }
    };
    if adjust_by_font && width > 0 {
        preferred_size.width = scale_by_font_size_int(width);
    } else {
        preferred_size.width = width;
    }
    preferred_size
}

/// Java `calcNewTextFieldSize(Dimension, int, boolean)`.
pub fn calc_new_text_field_size(
    preferred_size: Option<Dimension>,
    width: i32,
    adjust_by_font: bool,
) -> Dimension {
    calc_new_size(
        preferred_size,
        width,
        adjust_by_font,
        DEFAULT_TEXT_FIELD_HEIGHT,
    )
}

/// Java `setStringAndPrefer(JProgressBar, String, boolean)`.  Set the string and set a
/// new preferred width in the progress bar.  Returns true if the width changed.
///
/// The progress bar's preferred size and width are layout and are not modelled; the
/// stand-in has no preferred size, so Java's `preferredSize == null` branch is taken
/// with a zero width.
pub fn set_string_and_prefer(
    progress_bar: Option<&Rc<JComponent>>,
    string: Option<&str>,
    increase_only: bool,
) -> bool {
    let Some(progress_bar) = progress_bar else {
        return false;
    };
    let Some(string) = string.filter(|string| !string.is_empty()) else {
        return false;
    };
    let mut modified = false;
    // Get the current preferred size before the string is set.
    // Swing layout: progressBar.getPreferredSize() - not modelled (null), so
    // preferredSize = new Dimension(); preferredSize.width = progressBar.getWidth() (0).
    let mut preferred_size = Dimension {
        width: 0,
        height: 0,
    };
    let min_height = scale_by_font_size_int(DEFAULT_PROGRESS_BAR_HEIGHT);
    if preferred_size.height < min_height {
        preferred_size.height = min_height;
        modified = true;
    }
    // Set the string.
    let string = " ".to_string() + string + " ";
    progress_bar.set_string(Some(&string));
    let mut retval = false;
    // Get the new number of pixels for width.
    let font_metrics = get_font_metrics_j_progress_bar(progress_bar);
    if let Some(font_metrics) = font_metrics {
        let new_width = font_metrics.string_width(&string);
        // Update preferred size.
        let width = preferred_size.width;
        // Look for reasons not to use this preferred size or the new width.
        if preferred_size.height > 0
            && new_width > 0
            && new_width != width
            && (!increase_only || new_width >= width)
        {
            preferred_size.width = new_width;
            // Swing layout: progressBar.setPreferredSize(preferredSize).
            modified = true;
            retval = true;
        }
    }
    if modified {
        // Swing layout: progressBar.revalidate(); progressBar.repaint().
    }
    retval
}

/// Java `calcNewComboBoxSize(Dimension, int, boolean)`.
pub fn calc_new_combo_box_size(
    preferred_size: Option<Dimension>,
    width: i32,
    adjust_by_font: bool,
) -> Dimension {
    calc_new_size(
        preferred_size,
        width,
        adjust_by_font,
        DEFAULT_COMBO_BOX_HEIGHT,
    )
}

/// Java `scaleByFontSize(int)`.  Returns an integer that was scaled by the ratio of the
/// user font size with the default font size.
pub fn scale_by_font_size_int(size: i32) -> i32 {
    if size <= 0 || !etomo_director::EtomoDirector::is_user_preference_loaded() {
        return size;
    }
    let font_size = etomo_director::EtomoDirector::get_user_font_size();
    // Scale by font size.
    let ratio = font_size as f32 / ui_parameters::DEFAULT_FONT_SIZE as f32;
    java_math_round_f32(ratio * size as f32).max(1)
}

/// Java `getFontMetrics(AbstractButton)`.
pub fn get_font_metrics_abstract_button(button: &Rc<JComponent>) -> Option<FontMetrics> {
    // Swing painting: button.getGraphics().getFontMetrics(button.getFont()) with
    // graphics.dispose(); the stand-in has no Graphics, so Java's fallback is used.
    Some(button.get_font_metrics())
}

/// Java `getFontMetrics(JProgressBar)`.
pub fn get_font_metrics_j_progress_bar(progress_bar: &Rc<JComponent>) -> Option<FontMetrics> {
    // Swing painting: progressBar.getGraphics().getFontMetrics(...) - no Graphics in
    // the stand-in, so Java's fallback is used.
    Some(progress_bar.get_font_metrics())
}

/// Java `getFontMetrics(JComponent)`.
pub fn get_font_metrics_j_component(component: &Rc<JComponent>) -> Option<FontMetrics> {
    // Swing painting: component.getGraphics().getFontMetrics(...) - no Graphics in the
    // stand-in, so Java's fallback is used.
    Some(component.get_font_metrics())
}

/// Java `getPreferredJComponentWidth(JComponent, String)`.
pub fn get_preferred_j_component_width(component: &Rc<JComponent>, text: Option<&str>) -> i32 {
    let font_metrics = get_font_metrics_j_component(component);
    get_preferred_width_string_font_metrics(text, font_metrics.as_ref())
}

/// Java `getPreferredSize(AbstractButton, String)`.
pub fn get_preferred_size(button: &Rc<JComponent>, text: Option<&str>) -> Dimension {
    let font_metrics = get_font_metrics_abstract_button(button);
    Dimension {
        width: get_preferred_width_abstract_button_string_font_metrics(
            button,
            text,
            font_metrics.as_ref(),
        ),
        height: get_preferred_height(button, text, font_metrics.as_ref()),
    }
}

/// Java `getMaxWidthIndex(AbstractButton, String[])`.
pub fn get_max_width_index(button: &Rc<JComponent>, text: Option<&[Option<String>]>) -> i32 {
    let mut index = -1;
    let Some(text) = text else {
        return index;
    };
    let mut max_width = 0;
    for (i, text) in text.iter().enumerate() {
        let width = get_preferred_width_abstract_button_string(button, text.as_deref());
        if max_width < width {
            index = i as i32;
            max_width = width;
        }
    }
    index
}

/// Java public `getPreferredWidth(AbstractButton, String)`.
pub fn get_preferred_width_abstract_button_string(
    button: &Rc<JComponent>,
    text: Option<&str>,
) -> i32 {
    get_preferred_width_abstract_button_string_font_metrics(
        button,
        text,
        get_font_metrics_abstract_button(button).as_ref(),
    )
}

/// Java private `getPreferredWidth(AbstractButton, String, FontMetrics)`.
fn get_preferred_width_abstract_button_string_font_metrics(
    _button: &Rc<JComponent>,
    text: Option<&str>,
    font_metrics: Option<&FontMetrics>,
) -> i32 {
    let mut width = 0;
    // Swing layout: width += insets.left + insets.right (button.getInsets(null)) - not
    // modelled.
    // Swing painting: if (button.getIcon() != null) width += icon.getIconWidth() - icons
    // are not modelled.
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        // Swing painting: if icon != null, width += button.getIconTextGap().
        if let Some(font_metrics) = font_metrics {
            // Add some padding, because the the text is most likely bolded.
            width += font_metrics.string_width(text) + font_metrics.char_width('W');
        }
    }
    // Java 7 is underestimating this width. It estimated the width of a ButtonCell
    // labeled "Open" to be 42 - and this cuts off the text. Java 8 estimated the width to
    // be 43, which worked.
    let mut correction = 0;
    if width > 0 && utilities::is_java7() {
        // This number works for normal sized labels as long as the resulting correction
        // is at least one.
        let java7_off_by = 0.019f32;
        correction = java_math_round_f32(width as f32 * java7_off_by);
        if correction == 0 {
            correction = 1;
        }
    }
    width + correction
}

/// Java private `getPreferredHeight(AbstractButton, String, FontMetrics)`.
fn get_preferred_height(
    _button: &Rc<JComponent>,
    text: Option<&str>,
    font_metrics: Option<&FontMetrics>,
) -> i32 {
    // Swing layout: height = insets.top + insets.bottom (button.getInsets(null)) - not
    // modelled.
    let mut height = 0;
    let mut text_height = 0;
    if text.is_some_and(|text| !text.is_empty()) {
        if let Some(font_metrics) = font_metrics {
            text_height = font_metrics.get_height();
        }
    }
    // Swing painting: iconHeight = button.getIcon().getIconHeight() - icons are not
    // modelled.
    let icon_height = 0;
    height += text_height.max(icon_height);
    height
}

/// Java `getPreferredWidth(JLabel, String)`.
pub fn get_preferred_width_j_label_string(label: &Rc<JComponent>, text: Option<&str>) -> i32 {
    let font_metrics = get_font_metrics_j_component(label);
    // Swing layout: width = insets.left + insets.right (label.getInsets(null)) - not
    // modelled.
    let mut width = 0;
    // Swing painting: if (label.getIcon() != null) width += icon.getIconWidth().
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        // Swing painting: if icon != null, width += label.getIconTextGap().
        if let Some(font_metrics) = font_metrics {
            // Add padding singe the text is most likely bold.
            width += font_metrics.string_width(text) + font_metrics.char_width('W');
        }
    }
    width
}

/// Java `getPreferredWidth(String, FontMetrics)`.
pub fn get_preferred_width_string_font_metrics(
    text: Option<&str>,
    font_metrics: Option<&FontMetrics>,
) -> i32 {
    let mut width = 0;
    if let (Some(font_metrics), Some(text)) = (font_metrics, text.filter(|text| !text.is_empty())) {
        // Add padding singe the text is most likely bold.
        width += font_metrics.string_width(text) + font_metrics.char_width('W');
    }
    width
}

/// Java `addWithXSpace(Container, Component)`.  Add a component to a container
/// followed by the default value of x space.
pub fn add_with_x_space(panel: &Rc<JComponent>, component: &Rc<JComponent>) {
    panel.add(component);
    // Swing layout: panel.add(Box.createRigidArea(FixedDim.x5_y0)).
}

/// Java `addWithYSpace(Container, Component)`.
pub fn add_with_y_space(panel: &Rc<JComponent>, component: &Rc<JComponent>) {
    panel.add(component);
    // Swing layout: panel.add(Box.createRigidArea(FixedDim.x0_y5)).
}

/// Java `addWithSpace(Container, Component, Dimension)`.
pub fn add_with_space(panel: &Rc<JComponent>, component: &Rc<JComponent>, _dim: Dimension) {
    panel.add(component);
    // Swing layout: panel.add(Box.createRigidArea(dim)).
}

/// Java `alignComponentsX(Container, float)`.
pub fn align_components_x(container: &Rc<JComponent>, _alignment: f32) {
    for _child in container.get_components() {
        // Swing layout: ((JComponent) child).setAlignmentX(alignment).
    }
}

/// Java `shrinkWrapHorizontal(Container)`.  Used to stop horizontal panel bloat.
/// Component minimum/preferred sizes and insets are layout and are not modelled, so
/// only the container walk remains.
pub fn shrink_wrap_horizontal(container: Option<&Rc<JComponent>>) {
    let Some(container) = container else {
        return;
    };
    // Swing layout: dim = container.getPreferredSize(); for each JComponent child,
    // componentDim = child.getMinimumSize() (return if null) and track the widest;
    // if maxComponentWidth > 0, dim.width = maxComponentWidth + insets.left +
    // insets.right and container.setPreferredSize(dim); container.setMaximumSize(dim).
    let _children = container.get_components();
}

/// Java `alignAllComponentsX(Container, float)`.
pub fn align_all_components_x(container: &Rc<JComponent>, alignment: f32) {
    for child in container.get_components() {
        // Swing layout: jcomp.setAlignmentX(alignment).
        align_all_components_x(&child, alignment);
    }
}

/// Java `alignComponentsY(Container, float)`.
pub fn align_components_y(container: &Rc<JComponent>, _alignment: f32) {
    for _child in container.get_components() {
        // Swing layout: jcomp.setAlignmentY(alignment).
    }
}

/// Java `setButtonSizeAll(Container, Dimension)`.  Set the button sizes (preferred and
/// maximum) of all buttons in a container to the same size.
pub fn set_button_size_all(container: &Rc<JComponent>, _size: Dimension) {
    for child in container.get_components() {
        if matches!(
            child.kind(),
            ComponentKind::Button | ComponentKind::ToggleButton | ComponentKind::RadioButton
        ) {
            // Swing layout: btn.setPreferredSize(size); btn.setMaximumSize(size).
        }
    }
}

/// Java `getDefaultUIResource(Object, String)`.  The look and feel's `UIManager`
/// defaults are not modelled, so no resource is found.
pub fn get_default_ui_resource(target: Option<&str>, name: Option<&str>) -> Option<String> {
    if target.is_none() || name.is_none() {
        return None;
    }
    // Swing look and feel: search UIManager.getDefaults() for a key named `name` whose
    // value is an instance of target's class - not modelled.
    None
}

/// Java `getScreenSize()`.
pub fn get_screen_size() -> Dimension {
    let mut screen_size = SCREEN_SIZE.lock().unwrap();
    if screen_size.is_none() {
        // Toolkit.getDefaultToolkit().getScreenSize()
        let mut size = jdk::get_screen_size();
        size.height -= ESTIMATED_MENU_HEIGHT;
        *screen_size = Some(size);
    }
    screen_size.unwrap()
}

/// Java `highlightJTextComponents(boolean, Container)`.
pub fn highlight_j_text_components(highlight: bool, container: &Rc<JComponent>) {
    let component_list = container.get_components();
    for component in component_list {
        if matches!(
            component.kind(),
            ComponentKind::TextField | ComponentKind::TextArea
        ) {
            if highlight {
                // Swing painting: setBackground(Colors.HIGHLIGHT_BACKGROUND).
                let _ = colors::HIGHLIGHT_BACKGROUND;
            } else {
                // Swing painting: setBackground(Colors.BACKGROUND).
                let _ = colors::BACKGROUND;
            }
        }
        highlight_j_text_components(highlight, &component);
    }
}

/// Java `printComponents(Container)`.  Java prints each component's class; the
/// stand-in prints its kind.
pub fn print_components(container: &Rc<JComponent>) {
    let component_list = container.get_components();
    if component_list.is_empty() {
        println!();
        return;
    }
    println!(":");
    for component in component_list {
        print!("{:?}", component.kind());
        // Every component of the stand-in is a Container.
        print_components(&component);
    }
}

/// Java `divideColor(Color, int)`.
pub fn divide_color(color: Color, divisor: i32) -> ColorUIResource {
    (
        (color.0 as i32 / divisor) as u8,
        (color.1 as i32 / divisor) as u8,
        (color.2 as i32 / divisor) as u8,
    )
}

/// Java `isFontGreaterThanDefaultSize()`.
pub fn is_font_greater_than_default_size() -> bool {
    if etomo_director::EtomoDirector::get_user_font_size() as f64 > ui_parameters::DEFAULT_FONT_SIZE
    {
        return true;
    }
    false
}

/// Java `isFontLessThanDefaultSize()`.
pub fn is_font_less_than_default_size() -> bool {
    if (etomo_director::INSTANCE.with_user_configuration(|c| c.get_font_size()) as f64)
        < ui_parameters::DEFAULT_FONT_SIZE
    {
        return true;
    }
    false
}

/// `java.lang.Math.round(float)`, returning an `int` (the JDK body, not the javadoc's
/// `floor(a + 0.5f)` wording; see `logic/converter.rs`, which holds the same routine
/// privately).
pub fn java_math_round_f32(a: f32) -> i32 {
    // FloatConsts.SIGNIFICAND_WIDTH = 24, EXP_BIAS = 127,
    // EXP_BIT_MASK = 0x7F800000, SIGNIF_BIT_MASK = 0x007FFFFF.
    let int_bits = a.to_bits() as i32;
    let biased_exp = (int_bits & 0x7F800000i32) >> (24 - 1);
    let shift = (24 - 2 + 127) - biased_exp;
    if (shift & -32) == 0 {
        // shift >= 0 && shift < 32
        let mut r = (int_bits & 0x007FFFFFi32) | (0x007FFFFFi32 + 1);
        if int_bits < 0 {
            r = r.wrapping_neg();
        }
        ((r >> shift) + 1) >> 1
    } else {
        // a is a NaN, an infinity, or is already an integer of magnitude at least 2^23.
        a as i32
    }
}
