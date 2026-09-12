//! `IMOD/Etomo/src/etomo/ui/swing/InputCell.java`.
#![allow(dead_code)]

use super::colors;
use super::ui_utilities::Color;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::util::utilities;

/// Java abstract `InputCell` component operations at the native GUI boundary.
pub trait InputCellComponent {
    fn set_background(&mut self, color: Color);
    fn set_name(&mut self, name: String);
    fn is_enabled(&self) -> bool;
}

/// Java package-private abstract `InputCell` state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputCell {
    pub header_background: bool,
    pub highlight: bool,
    pub warning: bool,
    pub error: bool,
    pub plain_font: Option<String>,
    pub italic_font: Option<String>,
    pub jpanel_container: bool,
    pub initialized: bool,
    pub table_header: Option<String>,
    pub row_header: Option<String>,
    pub column_header: Option<String>,
    pub debug: bool,
    pub run_highlight: bool,
}
impl InputCell {
    /// Java `InputCell()`.
    pub fn new() -> Self {
        Self::new_with(false, false)
    }
    /// Java `InputCell(boolean, boolean)`.
    pub fn new_with(header_background: bool, debug: bool) -> Self {
        Self {
            header_background,
            highlight: false,
            warning: false,
            error: false,
            plain_font: None,
            italic_font: None,
            jpanel_container: false,
            initialized: false,
            table_header: None,
            row_header: None,
            column_header: None,
            debug,
            run_highlight: false,
        }
    }
    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn add(&mut self) {
        self.jpanel_container = true;
    }
    /// Java `remove()`.
    pub fn remove(&mut self) {
        if self.jpanel_container {
            self.jpanel_container = false;
        }
    }
    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
    /// Java `isDebug()`.
    pub fn is_debug(&self) -> bool {
        self.debug
    }
    /// Java `equalsSelectedStringValue(String)`.
    pub fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }
    /// Java `setHighlight(boolean)`.
    pub fn set_highlight<C: InputCellComponent>(&mut self, highlight: bool, component: &mut C) {
        self.highlight = highlight;
        self.set_background(component);
    }
    /// Java `setWarning(boolean)`.
    pub fn set_warning<C: InputCellComponent>(&mut self, warning: bool, component: &mut C) {
        if warning && self.error {
            self.warning = false;
            self.set_error(false, component);
        }
        self.warning = warning;
        self.set_background(component);
    }
    /// Java `setError(boolean)`.
    pub fn set_error<C: InputCellComponent>(&mut self, error: bool, component: &mut C) {
        if error && self.warning {
            self.error = false;
            self.set_warning(false, component);
        }
        self.error = error;
        self.set_background(component);
    }
    /// Java `setRunHighlight(boolean)`.
    pub fn set_run_highlight<C: InputCellComponent>(
        &mut self,
        run_highlight: bool,
        component: &mut C,
    ) {
        self.run_highlight = run_highlight;
        self.set_background(component);
    }
    /// Java `setBackground()`.
    pub fn set_background<C: InputCellComponent>(&self, component: &mut C) {
        let enabled = component.is_enabled();
        let color = if self.error {
            if enabled {
                colors::CELL_ERROR_BACKGROUND
            } else {
                colors::CELL_ERROR_BACKGROUND_NOT_EDITABLE
            }
        } else if self.warning {
            if enabled {
                colors::WARNING_BACKGROUND
            } else {
                colors::WARNING_BACKGROUND_NOT_EDITABLE
            }
        } else if self.run_highlight {
            if enabled {
                colors::RUN_HIGHLIGHT_BACKGROUND
            } else {
                colors::RUN_HIGHLIGHT_BACKGROUND_NOT_EDITABLE
            }
        } else if self.highlight {
            if enabled {
                colors::HIGHLIGHT_BACKGROUND
            } else {
                colors::HIGHLIGHT_BACKGROUND_NOT_EDITABLE
            }
        } else if enabled {
            if self.header_background {
                colors::HEADER_BACKGROUND
            } else {
                colors::BACKGROUND
            }
        } else {
            colors::Colors::get_cell_not_editable_background()
        };
        component.set_background(color);
    }
    /// Java `setFont()`.
    pub fn set_font(&mut self, plain_font: impl Into<String>, italic_font: impl Into<String>) {
        self.plain_font = Some(plain_font.into());
        self.italic_font = Some(italic_font.into());
    }
    /// Java `isHeaderBackground()`.
    pub fn is_header_background(&self) -> bool {
        self.header_background
    }
    /// Java `setHeaders(String, HeaderCell, HeaderCell)`.
    pub fn set_headers<C: InputCellComponent>(
        &mut self,
        table_header: &str,
        row_header: &str,
        column_header: &str,
        field_type: &str,
        component: &mut C,
    ) {
        self.table_header = Some(table_header.into());
        self.row_header = Some(row_header.into());
        self.column_header = Some(column_header.into());
        self.set_name(field_type, component);
    }
    /// Java `msgLabelChanged()`.
    pub fn msg_label_changed<C: InputCellComponent>(&self, field_type: &str, component: &mut C) {
        self.set_name(field_type, component);
    }
    /// Java `convertLabelToName(boolean)`.
    pub fn convert_label_to_name(&self, unlimited_segments: bool) -> Option<String> {
        utilities::convert_label_to_name_three(
            self.table_header.as_deref(),
            self.row_header.as_deref(),
            self.column_header.as_deref(),
            unlimited_segments,
        )
    }
    /// Java `setName()`.
    pub fn set_name<C: InputCellComponent>(&self, field_type: &str, component: &mut C) {
        let name = self.convert_label_to_name(true).unwrap_or_default();
        component.set_name(format!("{field_type}{SEPARATOR_CHAR}{name}"));
    }
}
impl Default for InputCell {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Boundary {
        enabled: bool,
        background: Option<Color>,
        name: String,
    }
    impl InputCellComponent for Boundary {
        fn set_background(&mut self, color: Color) {
            self.background = Some(color);
        }
        fn set_name(&mut self, name: String) {
            self.name = name;
        }
        fn is_enabled(&self) -> bool {
            self.enabled
        }
    }
    #[test]
    fn error_precedes_warning_and_disabled_color() {
        let mut input = InputCell::new();
        let mut component = Boundary::default();
        input.set_warning(true, &mut component);
        input.set_error(true, &mut component);
        assert!(input.error);
        assert!(!input.warning);
        assert_eq!(
            component.background,
            Some(colors::CELL_ERROR_BACKGROUND_NOT_EDITABLE)
        );
    }
    #[test]
    fn headers_create_source_test_name() {
        let mut input = InputCell::new();
        let mut component = Boundary {
            enabled: true,
            ..Default::default()
        };
        input.set_headers("T", "R", "C", "sp", &mut component);
        assert_eq!(component.name, "sp.t-r-c");
    }
}
