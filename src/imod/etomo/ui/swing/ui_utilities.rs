//! `IMOD/Etomo/src/etomo/ui/swing/UIUtilities.java`.
//!
//! Java puts the small, reusable pieces of Swing layout policy in this static
//! class.  The actual native widget calls are represented by the state below:
//! this keeps the source algorithms (including their deliberately odd sizing
//! rules) testable without making a second GUI toolkit part of eTomo.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{LazyLock, Mutex};

use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::util::utilities;

use super::panel::Dimension;
use super::{
    file_chooser::{FileChooser, FileChooserReturnValue},
    file_text_field_interface::FileFilter,
};

pub const ESTIMATED_MENU_HEIGHT: i32 = 60;
pub const DEFAULT_TEXT_FIELD_HEIGHT: i32 = 17;
pub const DEFAULT_COMBO_BOX_HEIGHT: i32 = 26;
pub const DEFAULT_SINGLE_LINE_BUTTON_HEIGHT: i32 = 27;
pub const DEFAULT_PROGRESS_BAR_HEIGHT: i32 = 18;
pub const DEFAULT_FONT_SIZE: i32 = 12;
pub const FOLDER_BUTTON: Dimension = Dimension {
    width: 22,
    height: 22,
};
pub const X5_Y0: Dimension = Dimension {
    width: 5,
    height: 0,
};
pub const X0_Y5: Dimension = Dimension {
    width: 0,
    height: 5,
};

/// Java `java.awt.Insets`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Insets {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
}

/// Source-visible `Icon` dimensions; painting is a native GUI boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Icon {
    pub width: i32,
    pub height: i32,
}

/// The three FontMetrics values used by this source unit.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FontMetrics {
    pub average_char_width: i32,
    pub wide_char_width: i32,
    pub height: i32,
}
impl FontMetrics {
    /// Java `FontMetrics.stringWidth(String)` represented without an AWT graphics context.
    pub fn string_width(&self, text: &str) -> i32 {
        text.chars().count() as i32 * self.average_char_width
    }
    /// Java `FontMetrics.charWidth('W')`.
    pub fn char_width_w(&self) -> i32 {
        self.wide_char_width
    }
}

/// State used by Java `AbstractButton` calls in this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AbstractButton {
    pub insets: Insets,
    pub icon: Option<Icon>,
    pub icon_text_gap: i32,
    pub font_metrics: Option<FontMetrics>,
    pub preferred_size: Option<Dimension>,
    pub maximum_size: Option<Dimension>,
    pub is_check_box: bool,
}

/// State used by Java `JLabel` calls in this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Label {
    pub insets: Insets,
    pub icon: Option<Icon>,
    pub icon_text_gap: i32,
    pub font_metrics: Option<FontMetrics>,
}

/// State used by Java `JProgressBar` calls in this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProgressBar {
    pub preferred_size: Option<Dimension>,
    pub width: i32,
    pub string: Option<String>,
    pub font_metrics: Option<FontMetrics>,
    pub revalidate_count: u64,
    pub repaint_count: u64,
}

/// Java `Color` / `ColorUIResource` RGB components.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Color {
    pub red: i32,
    pub green: i32,
    pub blue: i32,
}

/// Source-visible Swing component state needed by the recursive layout methods.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JComponent {
    pub alignment_x: f32,
    pub alignment_y: f32,
    pub minimum_size: Option<Dimension>,
    pub background: Option<Color>,
    pub font_metrics: Option<FontMetrics>,
}

/// Java `Component` cases touched by `UIUtilities`.
#[derive(Clone, Debug, PartialEq)]
pub enum Component {
    Button(AbstractButton),
    JComponent(JComponent),
    TextComponent(JComponent),
    Container(Box<Container>),
    RigidArea(Dimension),
    Other(String),
}

/// Java `Container` state used by the source methods.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Container {
    pub children: Vec<Component>,
    pub insets: Option<Insets>,
    pub preferred_size: Option<Dimension>,
    pub maximum_size: Option<Dimension>,
}

/// Direct `UIParameters.getFileChooserDimension()` boundary.
pub trait UiParametersBoundary {
    fn get_file_chooser_dimension(&self) -> Dimension;
}

/// Direct `Toolkit.getDefaultToolkit().getScreenSize()` boundary.
pub trait ToolkitBoundary {
    fn get_screen_size(&self) -> Dimension;
}

static SCREEN_SIZE: LazyLock<Mutex<Option<Dimension>>> = LazyLock::new(|| Mutex::new(None));
static FOLDER_BUTTON_DIMENSION: LazyLock<Mutex<Option<Dimension>>> =
    LazyLock::new(|| Mutex::new(None));

/// Rust namespace for Java's final static `UIUtilities` class.
pub struct UiUtilities;

impl UiUtilities {
    /// Java `chooseFile(Component, File, BrowsingDirectory, FileFilter)`.
    ///
    /// `JFileChooser.showOpenDialog` is a native Swing interaction.  The Rust
    /// renderer supplies its result through `selected_file`; no selection is
    /// the Java `CANCEL_OPTION` path.
    pub fn choose_file<P: UiParametersBoundary>(
        component: Option<&Component>,
        dir: Option<&Path>,
        browsing_directory: Option<&dyn BrowsingDirectory>,
        file_filter: Option<Rc<dyn FileFilter>>,
        ui_parameters: &P,
        selected_file: Option<&Path>,
    ) -> Option<PathBuf> {
        let mut chooser = FileChooser::new_with_manager(
            None,
            None,
            dir.and_then(Path::to_str),
            browsing_directory,
        );
        chooser.set_dialog_title(Some(" File Chooser"));
        chooser.set_preferred_size(ui_parameters.get_file_chooser_dimension());
        chooser.set_file_filter(file_filter);
        let _ = component;
        if chooser.show_open_dialog(selected_file) == FileChooserReturnValue::ApproveOption {
            let file = chooser.get_selected_file();
            if let (Some(directory), Some(file)) = (browsing_directory, file.as_ref()) {
                directory.set_browsing_dir(file.parent());
            }
            return file;
        }
        None
    }

    /// Java `scaleByFontSize(Dimension)`.
    pub fn scale_by_font_size_dimension(
        dimension: Option<Dimension>,
        user_font_size: Option<i32>,
    ) -> Option<Dimension> {
        let Some(dimension) = dimension else {
            return None;
        };
        let Some(font_size) = user_font_size else {
            return Some(dimension);
        };
        if (11..=14).contains(&font_size) {
            return Some(dimension);
        }
        Some(Dimension {
            width: ((font_size as f32 / 14.0 * dimension.width as f32).round() as i32).max(1),
            height: ((font_size as f32 / 14.0 * dimension.height as f32).round() as i32).max(1),
        })
    }

    /// Java synchronized `getScaledFolderButtonDimension()`.
    pub fn get_scaled_folder_button_dimension(user_font_size: Option<i32>) -> Dimension {
        let mut cached = FOLDER_BUTTON_DIMENSION.lock().unwrap();
        if let Some(dimension) = *cached {
            return dimension;
        }
        let dimension =
            Self::scale_by_font_size_dimension(Some(FOLDER_BUTTON), user_font_size).unwrap();
        if user_font_size.is_none() {
            *cached = Some(dimension);
        }
        dimension
    }

    /// Java private `calcNewSize(Dimension, int, boolean, int)`.
    pub fn calc_new_size(
        preferred_size: Option<Dimension>,
        width: i32,
        adjust_by_font: bool,
        default_height: i32,
        user_font_size: Option<i32>,
    ) -> Dimension {
        let mut preferred_size = preferred_size.unwrap_or(Dimension {
            width: 0,
            height: Self::scale_by_font_size_int(default_height, user_font_size),
        });
        preferred_size.width = if adjust_by_font && width > 0 {
            Self::scale_by_font_size_int(width, user_font_size)
        } else {
            width
        };
        preferred_size
    }

    /// Java `calcNewTextFieldSize(Dimension, int, boolean)`.
    pub fn calc_new_text_field_size(
        preferred_size: Option<Dimension>,
        width: i32,
        adjust_by_font: bool,
        user_font_size: Option<i32>,
    ) -> Dimension {
        Self::calc_new_size(
            preferred_size,
            width,
            adjust_by_font,
            DEFAULT_TEXT_FIELD_HEIGHT,
            user_font_size,
        )
    }

    /// Java `setStringAndPrefer(JProgressBar, String, boolean)`.
    pub fn set_string_and_prefer(
        progress_bar: Option<&mut ProgressBar>,
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
        let mut preferred_size = progress_bar.preferred_size.unwrap_or(Dimension {
            width: progress_bar.width,
            height: 0,
        });
        if preferred_size.height < DEFAULT_PROGRESS_BAR_HEIGHT {
            preferred_size.height = DEFAULT_PROGRESS_BAR_HEIGHT;
            modified = true;
        }
        let string = format!(" {string} ");
        progress_bar.string = Some(string.clone());
        let mut retval = false;
        if let Some(font_metrics) = progress_bar.font_metrics {
            let new_width = font_metrics.string_width(&string);
            let width = preferred_size.width;
            if preferred_size.height > 0
                && new_width > 0
                && new_width != width
                && (!increase_only || new_width >= width)
            {
                preferred_size.width = new_width;
                progress_bar.preferred_size = Some(preferred_size);
                modified = true;
                retval = true;
            }
        }
        if modified {
            progress_bar.revalidate_count += 1;
            progress_bar.repaint_count += 1;
        }
        retval
    }

    /// Java `calcNewComboBoxSize(Dimension, int, boolean)`.
    pub fn calc_new_combo_box_size(
        preferred_size: Option<Dimension>,
        width: i32,
        adjust_by_font: bool,
        user_font_size: Option<i32>,
    ) -> Dimension {
        Self::calc_new_size(
            preferred_size,
            width,
            adjust_by_font,
            DEFAULT_COMBO_BOX_HEIGHT,
            user_font_size,
        )
    }

    /// Java `scaleByFontSize(int)`.
    pub fn scale_by_font_size_int(size: i32, user_font_size: Option<i32>) -> i32 {
        if size <= 0 {
            return size;
        }
        let Some(font_size) = user_font_size else {
            return size;
        };
        ((font_size as f32 / DEFAULT_FONT_SIZE as f32 * size as f32).round() as i32).max(1)
    }

    /// Java `getFontMetrics(AbstractButton)`; graphics retrieval is a widget boundary.
    pub fn get_font_metrics_button(button: &AbstractButton) -> Option<FontMetrics> {
        button.font_metrics
    }
    /// Java `getFontMetrics(JProgressBar)`; graphics retrieval is a widget boundary.
    pub fn get_font_metrics_progress_bar(progress_bar: &ProgressBar) -> Option<FontMetrics> {
        progress_bar.font_metrics
    }
    /// Java `getFontMetrics(JComponent)`; graphics retrieval is a widget boundary.
    pub fn get_font_metrics_component(component: &JComponent) -> Option<FontMetrics> {
        component.font_metrics
    }

    /// Java `getPreferredJComponentWidth(JComponent, String)`.
    pub fn get_preferred_jcomponent_width(component: &JComponent, text: Option<&str>) -> i32 {
        Self::get_preferred_width_font_metrics(text, component.font_metrics)
    }

    /// Java `getPreferredSize(AbstractButton, String)`.
    pub fn get_preferred_size(button: &AbstractButton, text: Option<&str>) -> Dimension {
        let font_metrics = Self::get_font_metrics_button(button);
        Dimension {
            width: Self::get_preferred_width_button_with_font_metrics(button, text, font_metrics),
            height: Self::get_preferred_height(button, text, font_metrics),
        }
    }

    /// Java `getMaxWidthIndex(AbstractButton, String[])`.
    pub fn get_max_width_index(button: &AbstractButton, text: Option<&[String]>) -> i32 {
        let Some(text) = text else { return -1 };
        let mut index = -1;
        let mut max_width = 0;
        for (i, value) in text.iter().enumerate() {
            let width = Self::get_preferred_width_button(button, Some(value));
            if max_width < width {
                index = i as i32;
                max_width = width;
            }
        }
        index
    }

    /// Java public `getPreferredWidth(AbstractButton, String)`.
    pub fn get_preferred_width_button(button: &AbstractButton, text: Option<&str>) -> i32 {
        Self::get_preferred_width_button_with_font_metrics(
            button,
            text,
            Self::get_font_metrics_button(button),
        )
    }

    /// Java private overload `getPreferredWidth(AbstractButton, String, FontMetrics)`.
    pub fn get_preferred_width_button_with_font_metrics(
        button: &AbstractButton,
        text: Option<&str>,
        font_metrics: Option<FontMetrics>,
    ) -> i32 {
        let mut width = button.insets.left + button.insets.right;
        if let Some(icon) = button.icon {
            width += icon.width;
        }
        if let Some(text) = text.filter(|text| !text.is_empty()) {
            if button.icon.is_some() {
                width += button.icon_text_gap;
            }
            if let Some(font_metrics) = font_metrics {
                width += font_metrics.string_width(text) + font_metrics.char_width_w();
            }
        }
        let mut correction = 0;
        if width > 0 && utilities::is_java7() {
            correction = (width as f32 * 0.019).round() as i32;
            if correction == 0 {
                correction = 1;
            }
        }
        width + correction
    }

    /// Java private `getPreferredHeight(AbstractButton, String, FontMetrics)`.
    pub fn get_preferred_height(
        button: &AbstractButton,
        text: Option<&str>,
        font_metrics: Option<FontMetrics>,
    ) -> i32 {
        let mut height = button.insets.top + button.insets.bottom;
        let text_height = if text.is_some_and(|text| !text.is_empty()) {
            font_metrics.map_or(0, |font_metrics| font_metrics.height)
        } else {
            0
        };
        let icon_height = button.icon.map_or(0, |icon| icon.height);
        height + text_height.max(icon_height)
    }

    /// Java `getPreferredWidth(JLabel, String)`.
    pub fn get_preferred_width_label(label: &Label, text: Option<&str>) -> i32 {
        let mut width = label.insets.left + label.insets.right;
        if let Some(icon) = label.icon {
            width += icon.width;
        }
        if let Some(text) = text.filter(|text| !text.is_empty()) {
            if label.icon.is_some() {
                width += label.icon_text_gap;
            }
            if let Some(font_metrics) = label.font_metrics {
                width += font_metrics.string_width(text) + font_metrics.char_width_w();
            }
        }
        width
    }

    /// Java private overload `getPreferredWidth(String, FontMetrics)`.
    pub fn get_preferred_width_font_metrics(
        text: Option<&str>,
        font_metrics: Option<FontMetrics>,
    ) -> i32 {
        match (text.filter(|text| !text.is_empty()), font_metrics) {
            (Some(text), Some(font_metrics)) => {
                font_metrics.string_width(text) + font_metrics.char_width_w()
            }
            _ => 0,
        }
    }

    /// Java `addWithXSpace(Container, Component)`.
    pub fn add_with_x_space(panel: &mut Container, component: Component) {
        panel.children.push(component);
        panel.children.push(Component::RigidArea(X5_Y0));
    }
    /// Java `addWithYSpace(Container, Component)`.
    pub fn add_with_y_space(panel: &mut Container, component: Component) {
        panel.children.push(component);
        panel.children.push(Component::RigidArea(X0_Y5));
    }
    /// Java `addWithSpace(Container, Component, Dimension)`.
    pub fn add_with_space(panel: &mut Container, component: Component, dimension: Dimension) {
        panel.children.push(component);
        panel.children.push(Component::RigidArea(dimension));
    }

    /// Java `alignComponentsX(Container, float)`.
    pub fn align_components_x(container: &mut Container, alignment: f32) {
        for child in &mut container.children {
            if let Component::JComponent(component) | Component::TextComponent(component) = child {
                component.alignment_x = alignment;
            }
        }
    }

    /// Java `shrinkWrapHorizontal(Container)`.
    pub fn shrink_wrap_horizontal(container: Option<&mut Container>) {
        let Some(container) = container else { return };
        let Some(mut dimension) = container.preferred_size else {
            return;
        };
        let mut max_component_width = 0;
        for child in &container.children {
            let minimum_size = match child {
                Component::JComponent(component) | Component::TextComponent(component) => {
                    component.minimum_size
                }
                _ => continue,
            };
            let Some(minimum_size) = minimum_size else {
                return;
            };
            if max_component_width < minimum_size.width {
                max_component_width = minimum_size.width;
            }
        }
        if max_component_width > 0 {
            let insets = container.insets.unwrap_or_default();
            dimension.width = max_component_width + insets.left + insets.right;
            container.preferred_size = Some(dimension);
            container.maximum_size = Some(dimension);
        }
    }

    /// Java recursive `alignAllComponentsX(Container, float)`.
    pub fn align_all_components_x(container: &mut Container, alignment: f32) {
        for child in &mut container.children {
            match child {
                Component::JComponent(component) | Component::TextComponent(component) => {
                    component.alignment_x = alignment
                }
                Component::Container(container) => {
                    Self::align_all_components_x(container, alignment)
                }
                _ => {}
            }
        }
    }

    /// Java `alignComponentsY(Container, float)`.
    pub fn align_components_y(container: &mut Container, alignment: f32) {
        for child in &mut container.children {
            if let Component::JComponent(component) | Component::TextComponent(component) = child {
                component.alignment_y = alignment;
            }
        }
    }

    /// Java `setButtonSizeAll(Container, Dimension)`.
    pub fn set_button_size_all(container: &mut Container, size: Dimension) {
        for child in &mut container.children {
            if let Component::Button(button) = child
                && !button.is_check_box
            {
                button.preferred_size = Some(size);
                button.maximum_size = Some(size);
            }
        }
    }

    /// Java `getDefaultUIResource(Object, String)`.
    ///
    /// UIManager's look-and-feel map is a native UI boundary.  Keeping the
    /// `isInstance` condition here makes a supplied map follow Java exactly.
    pub fn get_default_ui_resource<T: Clone + 'static>(
        target_matches_value: impl Fn(&T) -> bool,
        defaults: &[(String, T)],
        name: Option<&str>,
    ) -> Option<T> {
        let name = name?;
        for (key, value) in defaults {
            if key == name && target_matches_value(value) {
                return Some(value.clone());
            }
        }
        None
    }

    /// Java `getScreenSize()`.
    pub fn get_screen_size<T: ToolkitBoundary>(toolkit: &T) -> Dimension {
        let mut screen_size = SCREEN_SIZE.lock().unwrap();
        if let Some(value) = *screen_size {
            return value;
        }
        let mut value = toolkit.get_screen_size();
        value.height -= ESTIMATED_MENU_HEIGHT;
        *screen_size = Some(value);
        value
    }

    /// Java recursive `highlightJTextComponents(boolean, Container)`.
    pub fn highlight_jtext_components(highlight: bool, container: &mut Container) {
        for component in &mut container.children {
            match component {
                Component::TextComponent(text) => {
                    text.background = Some(if highlight {
                        Color {
                            red: 204,
                            green: 255,
                            blue: 255,
                        }
                    } else {
                        Color {
                            red: 255,
                            green: 255,
                            blue: 255,
                        }
                    })
                }
                Component::Container(container) => {
                    Self::highlight_jtext_components(highlight, container)
                }
                _ => {}
            }
        }
    }

    /// Java `printComponents(Container)`.
    pub fn print_components(container: &Container) {
        if container.children.is_empty() {
            println!();
            return;
        }
        println!(":");
        for component in &container.children {
            let class = match component {
                Component::Button(_) => "class javax.swing.AbstractButton",
                Component::JComponent(_) => "class javax.swing.JComponent",
                Component::TextComponent(_) => "class javax.swing.text.JTextComponent",
                Component::Container(_) => "class java.awt.Container",
                Component::RigidArea(_) => "class javax.swing.Box$Filler",
                Component::Other(name) => name,
            };
            print!("{class}");
            if let Component::Container(container) = component {
                Self::print_components(container);
            } else {
                println!();
            }
        }
    }

    /// Java `divideColor(Color, int)`.
    pub fn divide_color(color: Color, divisor: i32) -> Color {
        Color {
            red: color.red / divisor,
            green: color.green / divisor,
            blue: color.blue / divisor,
        }
    }

    /// Java `isFontGreaterThanDefaultSize()`.
    pub fn is_font_greater_than_default_size(user_font_size: i32) -> bool {
        user_font_size > DEFAULT_FONT_SIZE
    }
    /// Java `isFontLessThanDefaultSize()`.
    pub fn is_font_less_than_default_size(user_font_size: i32) -> bool {
        user_font_size < DEFAULT_FONT_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Parameters;

    impl UiParametersBoundary for Parameters {
        fn get_file_chooser_dimension(&self) -> Dimension {
            Dimension {
                width: 300,
                height: 200,
            }
        }
    }

    #[derive(Default)]
    struct Directory(RefCell<Option<PathBuf>>);

    impl BrowsingDirectory for Directory {
        fn get_browsing_dir(&self) -> Option<PathBuf> {
            self.0.borrow().clone()
        }

        fn set_browsing_dir(&self, file: Option<&Path>) {
            *self.0.borrow_mut() = file.map(Path::to_path_buf);
        }
    }

    #[test]
    fn source_font_scaling_has_its_two_different_threshold_policies() {
        assert_eq!(
            UiUtilities::scale_by_font_size_dimension(
                Some(Dimension {
                    width: 10,
                    height: 20
                }),
                Some(12)
            ),
            Some(Dimension {
                width: 10,
                height: 20
            })
        );
        assert_eq!(UiUtilities::scale_by_font_size_int(10, Some(12)), 10);
        assert_eq!(UiUtilities::scale_by_font_size_int(10, Some(14)), 12);
    }
    #[test]
    fn button_size_includes_insets_icon_gap_and_bold_padding() {
        let button = AbstractButton {
            insets: Insets {
                left: 2,
                right: 3,
                ..Default::default()
            },
            icon: Some(Icon {
                width: 8,
                height: 7,
            }),
            icon_text_gap: 4,
            font_metrics: Some(FontMetrics {
                average_char_width: 5,
                wide_char_width: 8,
                height: 12,
            }),
            ..Default::default()
        };
        assert_eq!(
            UiUtilities::get_preferred_width_button(&button, Some("go")),
            35
        );
        assert_eq!(
            UiUtilities::get_preferred_size(&button, Some("go")).height,
            12
        );
    }
    #[test]
    fn progress_bar_keeps_java_empty_string_and_increase_only_behavior() {
        let mut bar = ProgressBar {
            width: 10,
            font_metrics: Some(FontMetrics {
                average_char_width: 5,
                wide_char_width: 0,
                height: 10,
            }),
            ..Default::default()
        };
        assert!(!UiUtilities::set_string_and_prefer(
            Some(&mut bar),
            Some(""),
            false
        ));
        assert!(UiUtilities::set_string_and_prefer(
            Some(&mut bar),
            Some("long"),
            true
        ));
        assert_eq!(bar.string.as_deref(), Some(" long "));
        assert_eq!(
            bar.preferred_size,
            Some(Dimension {
                width: 30,
                height: 18
            })
        );
    }
    #[test]
    fn recursive_operations_observe_java_component_type_checks() {
        let mut container = Container {
            children: vec![
                Component::TextComponent(JComponent::default()),
                Component::Container(Box::new(Container {
                    children: vec![Component::TextComponent(JComponent::default())],
                    ..Default::default()
                })),
            ],
            ..Default::default()
        };
        UiUtilities::highlight_jtext_components(true, &mut container);
        UiUtilities::align_all_components_x(&mut container, 0.5);
        let Component::TextComponent(first) = &container.children[0] else {
            panic!()
        };
        assert_eq!(
            first.background,
            Some(Color {
                red: 204,
                green: 255,
                blue: 255
            })
        );
        assert_eq!(first.alignment_x, 0.5);
    }

    #[test]
    fn choose_file_updates_browsing_directory_only_after_approval() {
        let directory = Directory::default();
        assert_eq!(
            UiUtilities::choose_file(
                None,
                Some(Path::new("/tmp")),
                Some(&directory),
                None,
                &Parameters,
                Some(Path::new("/tmp/etomo/example.rec")),
            ),
            Some(PathBuf::from("/tmp/etomo/example.rec"))
        );
        assert_eq!(
            directory.get_browsing_dir(),
            Some(PathBuf::from("/tmp/etomo"))
        );
        assert_eq!(
            UiUtilities::choose_file(None, None, Some(&directory), None, &Parameters, None),
            None
        );
        assert_eq!(
            directory.get_browsing_dir(),
            Some(PathBuf::from("/tmp/etomo"))
        );
    }
}
