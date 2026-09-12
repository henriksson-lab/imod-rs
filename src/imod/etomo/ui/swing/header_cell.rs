//! `IMOD/Etomo/src/etomo/ui/swing/HeaderCell.java`.
//!
//! `JButton`, `JToggleButton`, Swing borders, action listeners, and GridBag
//! placement are native GUI concerns. The state those calls expose to the
//! Java source is retained in `HeaderCellButtonBoundary`.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::util::utilities;

use super::cell::{
    Cell, CellGridBagConstraintsBoundary, CellGridBagLayoutBoundary, CellPanelBoundary, CellState,
    TableField, TableState,
};
use super::colors::{self, ColorUiResource};
use super::panel::Dimension;
use super::tooltip_formatter::TooltipFormatter;
use super::ui_utilities::{AbstractButton, Color, FontMetrics, Insets, UiUtilities};

/// Java `Border` identity and insets used by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeaderCellBorder {
    pub name: String,
    pub insets: Insets,
}
impl HeaderCellBorder {
    /// Java `BorderFactory.createEtchedBorder()`.
    pub fn create_etched_border() -> Self {
        Self {
            name: "EtchedBorder".into(),
            insets: Insets::default(),
        }
    }
}

/// Source-observable Java `AbstractButton` state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeaderCellButtonBoundary {
    pub abstract_button: AbstractButton,
    pub toggle: bool,
    pub text: Option<String>,
    pub name: Option<String>,
    pub focusable: bool,
    pub content_area_filled: bool,
    pub border: Option<HeaderCellBorder>,
    pub preferred_size: Dimension,
    pub width: i32,
    pub height: i32,
    pub background: ColorUiResource,
    pub foreground: ColorUiResource,
    pub enabled: bool,
    pub visible: bool,
    pub border_painted: bool,
    pub selected: bool,
    pub tooltip: Option<String>,
    pub action_listener_count: usize,
}
impl HeaderCellButtonBoundary {
    /// Java `new JButton()` / `new JToggleButton()` construction path.
    fn new(toggle: bool, text: Option<String>) -> Self {
        Self {
            abstract_button: AbstractButton {
                insets: Insets::default(),
                ..Default::default()
            },
            toggle,
            text,
            name: None,
            focusable: true,
            content_area_filled: true,
            border: Some(HeaderCellBorder::create_etched_border()),
            preferred_size: Dimension::default(),
            width: 0,
            height: 0,
            background: HEADER_BACKGROUND,
            foreground: colors::FOREGROUND,
            enabled: true,
            visible: true,
            border_painted: true,
            selected: false,
            tooltip: None,
            action_listener_count: 0,
        }
    }
}

/// Java `UITestFieldType` values selected only by `HeaderCell`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeaderCellUiTestFieldType {
    HeaderCell,
    MiniButton,
}
impl HeaderCellUiTestFieldType {
    /// Java `UITestFieldType.toString()`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HeaderCell => "hc",
            Self::MiniButton => "mb",
        }
    }
    /// Java `UITestFieldType.isUnlimitedSegments()`.
    pub fn is_unlimited_segments(self) -> bool {
        true
    }
}

/// Java `HeaderCell.background`.
pub const HEADER_BACKGROUND: ColorUiResource = Color {
    red: 204,
    green: 204,
    blue: 204,
};
/// Java `HeaderCell.GREYOUT`.
pub const GREYOUT: ColorUiResource = Color {
    red: 51,
    green: 51,
    blue: 51,
};
/// Java `HeaderCell.warningBackground`.
pub const WARNING_BACKGROUND: ColorUiResource = Color {
    red: 204,
    green: 204,
    blue: 153,
};

/// Java `HeaderCell.children` entry. A frontend connects an actual `Cell`'s
/// `msgLabelChanged` dispatch here; this records source messages at the GUI boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HeaderCellChildBoundary {
    pub msg_label_changed_count: usize,
}

/// Native handle for the Java `TableField` inherited through `Cell`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HeaderCellTableField;
impl TableField for HeaderCellTableField {}

/// Native handle for the Java `TableState` inherited through `Cell`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeaderCellTableState {
    pub display: bool,
    pub gridwidth: i32,
}
impl Default for HeaderCellTableState {
    fn default() -> Self {
        Self {
            display: true,
            gridwidth: Self::DEFAULT_GRIDWIDTH,
        }
    }
}
impl TableState<HeaderCellTableField> for HeaderCellTableState {
    fn is_display(&self, _table_field: Option<&HeaderCellTableField>) -> bool {
        self.display
    }
    fn get_gridwidth(&self, _table_field: Option<&HeaderCellTableField>) -> i32 {
        self.gridwidth
    }
}

/// Java package-private final `HeaderCell`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeaderCell {
    /// Java superclass `Cell.tableField` / `Cell.tableState` state.
    pub cell_state: CellState<HeaderCellTableField, HeaderCellTableState>,
    pub ui_test_field_type: Option<HeaderCellUiTestFieldType>,
    pub cell: HeaderCellButtonBoundary,
    pub jpanel_container: bool,
    /// Java `text`; empty is retained separately from the constructor's null.
    pub text: String,
    pub text_is_null: bool,
    pub control_color: bool,
    pub pad: String,
    pub children: Option<Vec<HeaderCellChildBoundary>>,
    pub table_header: Option<String>,
    pub row_header: Option<String>,
    pub column_header: Option<String>,
    pub font_metrics: Option<FontMetrics>,
}
impl Default for HeaderCell {
    /// Java `HeaderCell()`.
    fn default() -> Self {
        Self::new_internal(None, -1, true, false, None)
    }
}
impl HeaderCell {
    /// Java private `HeaderCell(String, int, boolean, boolean, String)`.
    fn new_internal(
        text: Option<String>,
        width: i32,
        control_color: bool,
        toggle: bool,
        reference: Option<String>,
    ) -> Self {
        let ui_test_field_type = if toggle {
            Some(HeaderCellUiTestFieldType::MiniButton)
        } else if reference.is_some() || text.is_some() {
            Some(HeaderCellUiTestFieldType::HeaderCell)
        } else {
            None
        };
        let text_is_null = text.is_none();
        let visible_text = text.unwrap_or_default();
        let button_text = (!text_is_null).then(|| format!("<html><b>{visible_text}</b>"));
        let mut cell = HeaderCellButtonBoundary::new(toggle, button_text);
        if !toggle {
            cell.focusable = false;
            cell.content_area_filled = false;
        }
        if width > 0 {
            let mut size = cell.preferred_size;
            size.width = width;
            cell.preferred_size = size;
            cell.abstract_button.preferred_size = Some(size);
            cell.width = width;
        }
        let mut value = Self {
            cell_state: CellState::default(),
            ui_test_field_type,
            cell,
            jpanel_container: false,
            text: visible_text,
            text_is_null,
            control_color,
            pad: String::new(),
            children: None,
            table_header: None,
            row_header: None,
            column_header: None,
            font_metrics: None,
        };
        if let Some(reference) = reference {
            value.set_name(Some(&reference));
        } else if !value.text_is_null {
            let text = value.text.clone();
            value.set_name(Some(&text));
        }
        value
    }
    /// Java `HeaderCell(String)`.
    pub fn new(text: impl Into<String>) -> Self {
        Self::new_internal(Some(text.into()), -1, true, false, None)
    }
    /// Java `HeaderCell(String, boolean)`.
    pub fn new_with_control_color(text: impl Into<String>, control_color: bool) -> Self {
        Self::new_internal(Some(text.into()), -1, control_color, false, None)
    }
    /// Java `HeaderCell(int)`.
    pub fn new_with_width(width: i32) -> Self {
        Self::new_internal(None, width, true, false, None)
    }
    /// Java `HeaderCell(String, int)`.
    pub fn new_with_text_width(text: impl Into<String>, width: i32) -> Self {
        Self::new_internal(Some(text.into()), width, true, false, None)
    }
    /// Compatibility spelling used by table-source units before this canonical unit.
    pub fn with_width(text: impl Into<String>, width: usize) -> Self {
        Self::new_with_text_width(text, width as i32)
    }
    /// Java static `getNamedInstance(String, String, String)`.
    pub fn get_named_instance(
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) -> Self {
        Self::new_internal(
            None,
            -1,
            true,
            false,
            utilities::concatenate(reference1, reference2, reference3, Some(" ")),
        )
    }
    /// Java static `getToggleInstance(String, int)`.
    pub fn get_toggle_instance(text: impl Into<String>, width: i32) -> Self {
        Self::new_internal(Some(text.into()), width, false, true, None)
    }
    /// Java `toString()`.
    pub fn to_string(&self) -> &str {
        &self.text
    }
    /// Java `getButton()`.
    pub fn get_button(&self) -> &HeaderCellButtonBoundary {
        &self.cell
    }
    /// Java `setFocusable(boolean)`.
    pub fn set_focusable(&mut self, input: bool) {
        self.cell.focusable = input;
    }
    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    /// Java `getComponent()`.
    pub fn get_component(&self) -> &HeaderCellButtonBoundary {
        &self.cell
    }
    /// Java `getName()`.
    pub fn get_name(&self) -> Option<&str> {
        self.cell.name.as_deref()
    }
    /// Java `setWarning(boolean, String)`.
    pub fn set_warning_with_tool_tip(&mut self, warning: bool, tool_tip_text: impl AsRef<str>) {
        self.set_warning(warning);
        self.set_tool_tip_text(tool_tip_text);
    }
    /// Java `setWarning(boolean)`.
    pub fn set_warning(&mut self, warning: bool) {
        if self.control_color {
            self.cell.background = if warning {
                WARNING_BACKGROUND
            } else {
                HEADER_BACKGROUND
            };
        }
    }
    /// Java `setBorderPainted(boolean)`.
    pub fn set_border_painted(&mut self, border_painted: bool) {
        self.cell.border_painted = border_painted;
    }
    /// Java overridden `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enable: bool) {
        if self.control_color {
            self.cell.foreground = if enable {
                colors::FOREGROUND
            } else {
                colors::CELL_DISABLED_FOREGROUND
            };
        } else {
            self.cell.enabled = enable;
        }
    }
    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.cell.visible = visible;
    }
    /// Java `setBorder(Border)`.
    pub fn set_border(&mut self, border: Option<HeaderCellBorder>) {
        self.cell.border = border;
    }
    /// Java `setSelected(boolean)`.
    pub fn set_selected(&mut self, selected: bool) {
        self.cell.selected = selected;
    }
    /// Java `addActionListener(ActionListener)`; listener dispatch is native GUI work.
    pub fn add_action_listener(&mut self) {
        self.cell.action_listener_count += 1;
    }
    /// Java overridden `add(JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn add(&mut self) {
        self.jpanel_container = true;
    }
    /// Java `TableComponent.getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        UiUtilities::get_preferred_width_button(&self.cell.abstract_button, self.get_text())
    }
    /// Java `remove()`.
    pub fn remove(&mut self) {
        if self.jpanel_container {
            self.jpanel_container = false;
        }
    }
    /// Java `getText()`.
    pub fn get_text(&self) -> Option<&str> {
        (!self.text_is_null).then_some(self.text.as_str())
    }
    /// Java `getInt()`.
    pub fn get_int(&self) -> i32 {
        EtomoNumber::new().set_string(self.get_text()).get_int()
    }
    /// Java `setText(String)`.
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.text_is_null = false;
        self.cell.text = Some(self.format_text());
        if let Some(children) = &mut self.children {
            for child in children {
                child.msg_label_changed_count += 1;
            }
        }
    }
    /// Java `setText(int)`.
    pub fn set_text_int(&mut self, text: i32) {
        self.set_text(text.to_string());
    }
    /// Java `setForeground(Color)`.
    pub fn set_foreground(&mut self, color: Color) {
        self.cell.foreground = color;
    }
    /// Java `addChild(Cell)`.
    pub fn add_child(&mut self, child: HeaderCellChildBoundary) {
        self.children.get_or_insert_default().push(child);
    }
    /// Java `setText()`.
    pub fn clear_text(&mut self) {
        self.text.clear();
        self.set_text("");
    }
    /// Java overridden `msgLabelChanged()`.
    pub fn msg_label_changed(&mut self) {
        self.set_name(None);
    }
    /// Java `setName(String)`.
    pub fn set_name(&mut self, reference: Option<&str>) {
        let Some(ui_test_field_type) = self.ui_test_field_type else {
            return;
        };
        let name = if let Some(reference) = reference {
            utilities::convert_label_to_name(
                Some(reference),
                ui_test_field_type.is_unlimited_segments(),
            )
        } else if self.table_header.is_none()
            && self.row_header.is_none()
            && self.column_header.is_none()
        {
            utilities::convert_label_to_name(
                self.get_text(),
                ui_test_field_type.is_unlimited_segments(),
            )
        } else {
            utilities::convert_label_to_name_three(
                self.table_header.as_deref(),
                self.row_header.as_deref(),
                self.column_header.as_deref(),
                ui_test_field_type.is_unlimited_segments(),
            )
        };
        let Some(name) = name.filter(|name| !name.is_empty()) else {
            return;
        };
        self.cell.name = Some(format!(
            "{}{SEPARATOR_CHAR}{name}",
            ui_test_field_type.as_str()
        ));
        if ARGUMENTS.lock().expect("arguments lock").is_print_names() {
            println!(
                "{}{SEPARATOR_CHAR}{name} {DEFAULT_DELIMITER} ",
                ui_test_field_type.as_str()
            );
        }
    }
    /// Java `setTableHeader(String)`.
    pub fn set_table_header(&mut self, input: Option<&str>) {
        self.table_header = input.map(str::to_owned);
    }
    /// Java `setRowHeader(HeaderCell)`.
    pub fn set_row_header(&mut self, input: Option<&HeaderCell>) {
        self.row_header = input.and_then(|header| header.get_text().map(str::to_owned));
    }
    /// Java `setColumnHeader(HeaderCell)`.
    pub fn set_column_header(&mut self, input: Option<&HeaderCell>) {
        self.column_header = input.and_then(|header| header.get_text().map(str::to_owned));
    }
    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.cell.selected
    }
    /// Java `getHeight()`.
    pub fn get_height(&self) -> i32 {
        self.cell.height
            + self
                .cell
                .border
                .as_ref()
                .expect("java.lang.NullPointerException")
                .insets
                .bottom
            - 1
    }
    /// Java `getWidth()`.
    pub fn get_width(&self) -> i32 {
        self.cell.width
    }
    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&mut self, text: impl AsRef<str>) {
        self.cell.tooltip = TooltipFormatter::instance().format(Some(text.as_ref()));
    }
    /// Java `addTooltip(String)`.
    pub fn add_tooltip(&mut self, text: Option<&str>) {
        let Some(text) = text else { return };
        match self.cell.tooltip.clone() {
            None => self.set_tool_tip_text(text),
            Some(tooltip) => {
                let formatted = TooltipFormatter::instance()
                    .format(Some(text))
                    .unwrap_or_default();
                self.cell.tooltip = Some(format!("{tooltip} & {formatted}"));
            }
        }
    }
    /// Java `pad()`.
    pub fn pad(&mut self) {
        if self.text_is_null {
            return;
        }
        self.pad = " ".into();
        self.cell.text = Some(self.format_text());
    }
    /// Java private `formatText()`.
    fn format_text(&self) -> String {
        format!("<html><b>{}{}</b>", self.text, self.pad)
    }
}

impl Cell<HeaderCellTableField, HeaderCellTableState> for HeaderCell {
    fn set_enabled(&mut self, enable: bool) {
        HeaderCell::set_enabled(self, enable);
    }
    fn msg_label_changed(&mut self) {
        HeaderCell::msg_label_changed(self);
    }
    fn add(
        &mut self,
        _panel: &mut CellPanelBoundary,
        _layout: &mut CellGridBagLayoutBoundary,
        _constraints: &mut CellGridBagConstraintsBoundary,
    ) {
        HeaderCell::add(self);
    }
    fn cell_state(&self) -> &CellState<HeaderCellTableField, HeaderCellTableState> {
        &self.cell_state
    }
    fn cell_state_mut(&mut self) -> &mut CellState<HeaderCellTableField, HeaderCellTableState> {
        &mut self.cell_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_selects_source_widget_and_test_name() {
        let header = HeaderCell::new("Angular Search");
        assert!(!header.cell.toggle);
        assert!(!header.cell.focusable);
        assert!(!header.cell.content_area_filled);
        assert_eq!(
            header.cell.text.as_deref(),
            Some("<html><b>Angular Search</b>")
        );
        assert_eq!(header.get_name(), Some("hc.angular-search"));
        let toggle = HeaderCell::get_toggle_instance("Open", 29);
        assert!(toggle.cell.toggle);
        assert!(toggle.cell.focusable);
        assert!(toggle.cell.content_area_filled);
        assert_eq!(toggle.get_name(), Some("mb.open"));
        assert_eq!(toggle.get_width(), 29);
    }
    #[test]
    fn warning_enable_and_padding_follow_control_color_branch() {
        let mut header = HeaderCell::new("Header");
        header.set_warning(true);
        assert_eq!(header.cell.background, WARNING_BACKGROUND);
        header.set_enabled(false);
        assert_eq!(header.cell.foreground, colors::CELL_DISABLED_FOREGROUND);
        assert!(header.cell.enabled);
        header.pad();
        assert_eq!(header.cell.text.as_deref(), Some("<html><b>Header </b>"));
        let mut toggle = HeaderCell::get_toggle_instance("Run", 0);
        toggle.set_warning(true);
        assert_eq!(toggle.cell.background, HEADER_BACKGROUND);
        toggle.set_enabled(false);
        assert!(!toggle.cell.enabled);
    }
    #[test]
    fn label_changes_rebuild_table_name_and_notify_children() {
        let mut header = HeaderCell::new("Ignored");
        let row = HeaderCell::new("Row #1");
        let column = HeaderCell::new("Column");
        header.set_table_header(Some("Table"));
        header.set_row_header(Some(&row));
        header.set_column_header(Some(&column));
        header.msg_label_changed();
        assert_eq!(header.get_name(), Some("hc.table-row-#1-column"));
        header.add_child(HeaderCellChildBoundary::default());
        header.set_text_int(12);
        assert_eq!(header.get_text(), Some("12"));
        assert_eq!(
            header.children.as_ref().unwrap()[0].msg_label_changed_count,
            1
        );
    }
    #[test]
    fn tooltip_is_formatted_then_appended_as_in_source() {
        let mut header = HeaderCell::new("Header");
        header.add_tooltip(Some("first"));
        header.add_tooltip(Some("second"));
        assert_eq!(
            header.cell.tooltip.as_deref(),
            Some("<html>first & <html>second")
        );
    }
    #[test]
    fn inherited_cell_table_state_dispatch_is_retained() {
        let mut header = HeaderCell::new("Header");
        Cell::<HeaderCellTableField, HeaderCellTableState>::set_table_state(
            &mut header,
            Some(HeaderCellTableField),
            Some(HeaderCellTableState {
                display: false,
                gridwidth: 3,
            }),
        );
        assert!(!Cell::<HeaderCellTableField, HeaderCellTableState>::is_display(&header));
        assert_eq!(
            Cell::<HeaderCellTableField, HeaderCellTableState>::get_gridwidth(&header),
            3
        );
    }
}
