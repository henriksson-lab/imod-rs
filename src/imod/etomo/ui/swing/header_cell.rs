//! `IMOD/Etomo/src/etomo/ui/swing/HeaderCell.java`.
//!
//! Java `final class HeaderCell extends Cell implements TableComponent,
//! UIComponent, SwingComponent`: a table header cell drawn as a button.  The
//! `Cell` base is embedded as `base`; the abstract `Cell` methods are the
//! `CellVirtual` implementation.  Painting (fonts, borders, background colours,
//! sizes) is Swing layout and is recorded as comments.

use std::cell::{Cell as StdCell, RefCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::cell::{Cell, CellVirtual};
use super::swing_component::SwingComponent;
use super::tooltip_formatter;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionListener, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `HeaderCell`.
pub struct HeaderCell {
    /// Java superclass `Cell`.
    base: Cell,
    /// Java `uiTestFieldType`.
    ui_test_field_type: Option<UITestFieldType>,
    /// Java `cell` (an `AbstractButton`: `JButton` or `JToggleButton`).
    cell: Rc<JComponent>,
    /// Java `jpanelContainer`.
    jpanel_container: RefCell<Option<Rc<JComponent>>>,
    /// Java `text`.
    text: RefCell<Option<String>>,
    /// Java `controlColor`.
    control_color: bool,
    /// Java `pad`.
    pad: RefCell<String>,
    /// Java `children` (`List` of `Cell`).
    children: RefCell<Option<Vec<Weak<dyn CellVirtual>>>>,
    /// Java `tableHeader`.
    table_header: RefCell<Option<String>>,
    /// Java `rowHeader`.
    row_header: RefCell<Option<Rc<HeaderCell>>>,
    /// Java `columnHeader`.
    column_header: RefCell<Option<Rc<HeaderCell>>>,
    /// Java `setBorderPainted` state (painting only; kept as state).
    border_painted: StdCell<bool>,
}

impl Deref for HeaderCell {
    type Target = Cell;
    fn deref(&self) -> &Cell {
        &self.base
    }
}

impl std::fmt::Display for HeaderCell {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.text.borrow().as_deref().unwrap_or("null"))
    }
}

impl HeaderCell {
    /// Java private `HeaderCell(String, int, boolean, boolean, String)`.
    fn new_private(
        text: Option<&str>,
        width: i32,
        control_color: bool,
        toggle: bool,
        reference: Option<&str>,
    ) -> Rc<HeaderCell> {
        let ui_test_field_type = if toggle {
            Some(UITestFieldType::MINI_BUTTON)
        } else if reference.is_some() || text.is_some() {
            Some(UITestFieldType::HEADER_CELL)
        } else {
            None
        };
        let cell = if toggle {
            JComponent::new_toggle_button("")
        } else {
            JComponent::new_button("")
        };
        let header_cell = Rc::new(HeaderCell {
            base: Cell::new(),
            ui_test_field_type,
            cell,
            jpanel_container: RefCell::new(None),
            text: RefCell::new(text.map(str::to_owned)),
            control_color,
            pad: RefCell::new(String::new()),
            children: RefCell::new(None),
            table_header: RefCell::new(None),
            row_header: RefCell::new(None),
            column_header: RefCell::new(None),
            border_painted: StdCell::new(true),
        });
        if text.is_some() {
            header_cell.cell.set_text(&header_cell.format_text());
        }
        // Swing layout: a non-toggle cell is not focusable and does not fill its
        // content area; every cell gets an etched border; a positive width sets the
        // preferred size.
        let _ = width;
        if reference.is_some() {
            header_cell.set_name(reference);
        } else if text.is_some() {
            header_cell.set_name(text);
        }
        header_cell
    }

    /// Java `getButton()`.
    pub fn get_button(&self) -> Rc<JComponent> {
        self.cell.clone()
    }

    /// Java `setFocusable(boolean)`: focus is not modelled.
    pub fn set_focusable(&self, _input: bool) {}

    /// Java `HeaderCell()`.
    pub fn new_void() -> Rc<HeaderCell> {
        Self::new_private(None, -1, true, false, None)
    }

    /// Java `getNamedInstance(String, String, String)`.
    pub fn get_named_instance(
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    ) -> Rc<HeaderCell> {
        let reference = utilities::concatenate(reference1, reference2, reference3, Some(" "));
        Self::new_private(None, -1, true, false, reference.as_deref())
    }

    /// Java `HeaderCell(String)`.
    pub fn new_string(text: Option<&str>) -> Rc<HeaderCell> {
        Self::new_private(text, -1, true, false, None)
    }

    /// Java `HeaderCell(String, boolean)`.
    pub fn new_string_boolean(text: Option<&str>, control_color: bool) -> Rc<HeaderCell> {
        Self::new_private(text, -1, control_color, false, None)
    }

    /// Java `HeaderCell(int)`.
    pub fn new_int(width: i32) -> Rc<HeaderCell> {
        Self::new_private(None, width, true, false, None)
    }

    /// Java `HeaderCell(String, int)`.
    pub fn new_string_int(text: Option<&str>, width: i32) -> Rc<HeaderCell> {
        Self::new_private(text, width, true, false, None)
    }

    /// Java `getToggleInstance(String, int)`.
    pub fn get_toggle_instance(text: Option<&str>, width: i32) -> Rc<HeaderCell> {
        Self::new_private(text, width, false, true, None)
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.cell.clone()
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.cell.get_name()
    }

    /// Java `setWarning(boolean, String)`.
    pub fn set_warning_boolean_string(&self, warning: bool, tool_tip_text: Option<&str>) {
        self.set_warning_boolean(warning);
        self.set_tool_tip_text(tool_tip_text);
    }

    /// Java `setWarning(boolean)`: sets the warning or normal background when the
    /// cell controls its colour.  Background colour is painting only.
    pub fn set_warning_boolean(&self, warning: bool) {
        if !self.control_color {
            return;
        }
        // Swing painting: `cell.setBackground(warning ? warningBackground : background)`.
        let _ = warning;
    }

    /// Java `setBorderPainted(boolean)`.
    pub fn set_border_painted(&self, border_painted: bool) {
        self.border_painted.set(border_painted);
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.cell.set_visible(visible);
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        self.cell.set_selected(selected);
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, action_listener: ActionListener) {
        self.cell.add_action_listener(action_listener);
    }

    /// Java `getPreferredWidth()`.
    pub fn get_preferred_width(&self) -> i32 {
        super::ui_utilities::get_preferred_width_abstract_button_string(
            &self.cell,
            self.get_text().as_deref(),
        )
    }

    /// Java `remove()`.
    pub fn remove(&self) {
        let container = self.jpanel_container.borrow_mut().take();
        if let Some(container) = container {
            container.remove(&self.cell);
        }
    }

    /// Java `getText()`.
    pub fn get_text(&self) -> Option<String> {
        self.text.borrow().clone()
    }

    /// Java `getInt()`.
    pub fn get_int(&self) -> i32 {
        let mut number = EtomoNumber::new();
        number.set_string(self.text.borrow().as_deref());
        number.get_int()
    }

    /// Java `setText(String)`.
    pub fn set_text_string(&self, text: Option<&str>) {
        *self.text.borrow_mut() = text.map(str::to_owned);
        self.cell.set_text(&self.format_text());
        let children: Vec<_> = self
            .children
            .borrow()
            .iter()
            .flatten()
            .filter_map(Weak::upgrade)
            .collect();
        for child in children {
            child.msg_label_changed();
        }
    }

    /// Java `setText(int)`.
    pub fn set_text_int(&self, text: i32) {
        *self.text.borrow_mut() = Some(text.to_string());
        self.cell.set_text(&self.format_text());
        let children: Vec<_> = self
            .children
            .borrow()
            .iter()
            .flatten()
            .filter_map(Weak::upgrade)
            .collect();
        for child in children {
            child.msg_label_changed();
        }
    }

    /// Java `setForeground(Color)`.
    pub fn set_foreground(&self, color: Option<(u8, u8, u8)>) {
        self.cell.set_foreground(color);
    }

    /// Java `addChild(Cell)`.
    pub fn add_child(&self, child: Weak<dyn CellVirtual>) {
        let mut children = self.children.borrow_mut();
        children.get_or_insert_with(Vec::new).push(child);
    }

    /// Java `setText()`.
    pub fn set_text_void(&self) {
        *self.text.borrow_mut() = Some(String::new());
        self.set_text_string(Some(""));
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, reference: Option<&str>) {
        let prefix;
        let unlimited_segments;
        if let Some(ui_test_field_type) = &self.ui_test_field_type {
            prefix = format!("{ui_test_field_type}{SEPARATOR_CHAR}");
            unlimited_segments = ui_test_field_type.is_unlimited_segments();
        } else {
            return;
        }
        let name;
        if reference.is_some() {
            name = utilities::convert_label_to_name(reference, unlimited_segments);
        } else if self.table_header.borrow().is_none()
            && self.row_header.borrow().is_none()
            && self.column_header.borrow().is_none()
        {
            name =
                utilities::convert_label_to_name(self.text.borrow().as_deref(), unlimited_segments);
        } else {
            let row_text = self
                .row_header
                .borrow()
                .as_ref()
                .and_then(|row_header| row_header.get_text());
            let column_text = self
                .column_header
                .borrow()
                .as_ref()
                .and_then(|column_header| column_header.get_text());
            name = utilities::convert_label_to_name_three(
                self.table_header.borrow().as_deref(),
                row_text.as_deref(),
                column_text.as_deref(),
                unlimited_segments,
            );
        }
        let Some(name) = name.filter(|name| !name.is_empty()) else {
            return;
        };
        self.cell.set_name(Some(&format!("{prefix}{name}")));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{}{}{} {} ",
                self.ui_test_field_type.as_ref().unwrap(),
                SEPARATOR_CHAR,
                name,
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `setTableHeader(String)`.
    pub fn set_table_header(&self, input: Option<&str>) {
        *self.table_header.borrow_mut() = input.map(str::to_owned);
    }

    /// Java `setRowHeader(HeaderCell)`.
    pub fn set_row_header(&self, input: Option<Rc<HeaderCell>>) {
        *self.row_header.borrow_mut() = input;
    }

    /// Java `setColumnHeader(HeaderCell)`.
    pub fn set_column_header(&self, input: Option<Rc<HeaderCell>>) {
        *self.column_header.borrow_mut() = input;
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.cell.is_selected()
    }

    /// Java `getHeight()`: a Swing size; not modelled.
    pub fn get_height(&self) -> i32 {
        0
    }

    /// Java `getWidth()`: a Swing size; not modelled.
    pub fn get_width(&self) -> i32 {
        0
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.cell
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `addTooltip(String)`.
    pub fn add_tooltip(&self, text: Option<&str>) {
        if text.is_none() {
            return;
        }
        let tooltip = self.cell.get_tool_tip_text();
        match tooltip {
            None => self.set_tool_tip_text(text),
            Some(tooltip) => self.cell.set_tool_tip_text(Some(&format!(
                "{} & {}",
                tooltip,
                tooltip_formatter::INSTANCE
                    .format(text)
                    .unwrap_or_else(|| "null".to_owned())
            ))),
        }
    }

    /// Java `pad()`.
    pub fn pad(&self) {
        if self.text.borrow().is_none() {
            return;
        }
        *self.pad.borrow_mut() = " ".to_owned();
        self.cell.set_text(&self.format_text());
    }

    /// Java private `formatText()`.
    fn format_text(&self) -> String {
        format!(
            "<html><b>{}{}</b>",
            self.text.borrow().as_deref().unwrap_or("null"),
            self.pad.borrow()
        )
    }
}

impl CellVirtual for HeaderCell {
    fn cell(&self) -> &Cell {
        &self.base
    }

    /// Java `setEnabled(boolean)`.
    fn set_enabled(&self, enable: bool) {
        if self.control_color {
            // Swing painting: the foreground switches between `Colors.FOREGROUND`
            // and `Colors.CELL_DISABLED_FOREGROUND`; the button stays enabled.
            let _ = enable;
        } else {
            // push button
            self.cell.set_enabled(enable);
        }
    }

    /// Java `msgLabelChanged()`.
    fn msg_label_changed(&self) {
        self.set_name(None);
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.
    fn add(&self, panel: &Rc<JComponent>) {
        // Swing layout: `layout.setConstraints(cell, constraints)`.
        panel.add(&self.cell);
        *self.jpanel_container.borrow_mut() = Some(panel.clone());
    }
}

impl SwingComponent for HeaderCell {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.cell.clone()
    }
}

impl UIComponent for HeaderCell {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.cell.clone()
    }
}
