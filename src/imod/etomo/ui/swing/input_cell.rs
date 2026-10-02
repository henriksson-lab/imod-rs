//! `IMOD/Etomo/src/etomo/ui/swing/InputCell.java`.
//!
//! Java `abstract class InputCell extends Cell`: a table cell holding an input
//! component.  Uses lazy construction: the inheritor calls `setBackground`,
//! `setForeground` and `setFont` from its constructor.
//!
//! A concrete cell embeds [`InputCell`] as its `base` field (with
//! `Deref<Target = InputCell>`), implements [`InputCellVirtual`] (the abstract and
//! overridden methods) and [`CellVirtual`], and right after allocating itself calls
//! [`InputCell::set_this`] so that `InputCell`'s own methods can dispatch to the
//! subclass (`isEnabled`, `getComponent`, `setBackground(ColorUIResource)`, ...).
//! Code that holds a Java `InputCell` holds `Rc<dyn InputCellVirtual>`.

use std::cell::{Cell as StdCell, RefCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::cell::{Cell, CellVirtual};
use super::colors::{self, ColorUIResource};
use super::header_cell::HeaderCell;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::util::clean_print::CleanPrint;
use crate::imod::etomo::util::utilities;

/// The abstract and overridable methods of Java `InputCell`.
pub trait InputCellVirtual: CellVirtual {
    /// The embedded `InputCell` (Java `this` seen as an `InputCell`).
    fn input_cell(&self) -> &InputCell;

    /// Java abstract `getComponent()`.
    fn get_component(&self) -> Rc<JComponent>;

    /// Java abstract `getFieldType()`.
    fn get_field_type(&self) -> &'static UITestFieldType;

    /// Java abstract `getWidth()`.
    fn get_width(&self) -> i32;

    /// Java abstract `setToolTipText(String)`.
    fn set_tool_tip_text(&self, tool_tip_text: Option<&str>);

    /// Java abstract `getText()`.
    fn get_text(&self) -> Option<String>;

    /// Java abstract `setName(String, String, String)`.
    fn set_name_string_string_string(
        &self,
        reference1: Option<&str>,
        reference2: Option<&str>,
        reference3: Option<&str>,
    );

    /// Java abstract `getName()`.
    fn get_name(&self) -> Option<String>;

    /// Java abstract `setLocked(boolean)`.
    fn set_locked(&self, locked: bool);

    /// Java abstract `setEditable(boolean)`.
    fn set_editable(&self, editable: bool);

    /// Java abstract `isLocked()`.
    fn is_locked(&self) -> bool;

    /// Java abstract `isEditable()`.
    fn is_editable(&self) -> bool;

    /// Java abstract `isEnabled()`.
    fn is_enabled(&self) -> bool;

    /// Java `setDebug(boolean)` (overridden by `FieldCell`).
    fn set_debug(&self, input: bool) {
        self.input_cell().set_debug_super(input);
    }

    /// Java `isDebug()` (overridden by `CheckBoxCell`).
    fn is_debug(&self) -> bool {
        self.input_cell().is_debug_super()
    }

    /// Java `equalsSelectedStringValue(String)` (overridden by `CheckBoxCell`).
    fn equals_selected_string_value(&self, value: Option<&str>) -> bool {
        self.input_cell().equals_selected_string_value_super(value)
    }

    /// Java `setBackground(ColorUIResource)` (overridden by `SpinnerCell`).
    fn set_background_color_ui_resource(&self, color: ColorUIResource) {
        self.input_cell()
            .set_background_color_ui_resource_super(color);
    }
}

/// Java `InputCell`.
pub struct InputCell {
    /// Java superclass `Cell`.
    base: Cell,
    /// Java `this`, seen through the subclass's overrides.
    this: RefCell<Option<Weak<dyn InputCellVirtual>>>,
    /// Java `cleanPrint`.
    #[allow(dead_code)]
    clean_print: CleanPrint,
    /// Java `headerBackground`.
    header_background: bool,
    /// Java `highlight`.
    highlight: StdCell<bool>,
    /// Java `warning`.
    warning: StdCell<bool>,
    /// Java `error`.
    error: StdCell<bool>,
    // Java `plainFont` and `italicFont`: fonts are not modelled (see `set_font`).
    /// Java `jpanelContainer`.
    jpanel_container: RefCell<Option<Rc<JComponent>>>,
    /// Java `initialized` (never read by the source).
    #[allow(dead_code)]
    initialized: StdCell<bool>,
    /// Java `tableHeader`.
    table_header: RefCell<Option<String>>,
    /// Java `rowHeader`.
    row_header: RefCell<Option<Rc<HeaderCell>>>,
    /// Java `columnHeader`.
    column_header: RefCell<Option<Rc<HeaderCell>>>,
    /// Java `debug`.
    debug: StdCell<bool>,
    /// Java `runHighlight`.
    run_highlight: StdCell<bool>,
}

impl Deref for InputCell {
    type Target = Cell;

    fn deref(&self) -> &Cell {
        &self.base
    }
}

impl InputCell {
    /// Java `InputCell()`.
    pub fn new_void() -> InputCell {
        InputCell::new_boolean_boolean(false, false)
    }

    /// Java `InputCell(boolean, boolean)`.
    pub fn new_boolean_boolean(header_background: bool, debug: bool) -> InputCell {
        InputCell {
            base: Cell::new(),
            this: RefCell::new(None),
            clean_print: CleanPrint::get_instance_with_label(Some("InputCell")),
            header_background,
            highlight: StdCell::new(false),
            warning: StdCell::new(false),
            error: StdCell::new(false),
            jpanel_container: RefCell::new(None),
            initialized: StdCell::new(false),
            table_header: RefCell::new(None),
            row_header: RefCell::new(None),
            column_header: RefCell::new(None),
            debug: StdCell::new(debug),
            run_highlight: StdCell::new(false),
        }
    }

    /// Connects this `InputCell` to the subclass that embeds it (Java `this`).  The
    /// subclass calls it right after allocating itself, before running the rest of
    /// its constructor.
    pub fn set_this(&self, this: Weak<dyn InputCellVirtual>) {
        *self.this.borrow_mut() = Some(this);
    }

    /// Java `this` as the subclass.
    fn this(&self) -> Rc<dyn InputCellVirtual> {
        self.this
            .borrow()
            .as_ref()
            .and_then(Weak::upgrade)
            .expect("InputCell: set_this was not called by the subclass constructor")
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)` (implements
    /// `Cell.add`).
    pub fn add(&self, panel: &Rc<JComponent>) {
        let this = self.this();
        // Swing layout: layout.setConstraints(getComponent(), constraints).
        panel.add(&this.get_component());
        *self.jpanel_container.borrow_mut() = Some(panel.clone());
    }

    /// Java final `remove()`.
    pub fn remove(&self) {
        let container = self.jpanel_container.borrow().clone();
        if let Some(container) = container {
            container.remove(&self.this().get_component());
            *self.jpanel_container.borrow_mut() = None;
        }
    }

    /// Java `setDebug(boolean)` (the class's body).
    pub fn set_debug_super(&self, input: bool) {
        self.debug.set(input);
    }

    /// Java `isDebug()` (the class's body).
    pub fn is_debug_super(&self) -> bool {
        self.debug.get()
    }

    /// Java `equalsSelectedStringValue(String)` (the class's body).  True value is
    /// not implemented so all non-empty values are true.
    pub fn equals_selected_string_value_super(&self, value: Option<&str>) -> bool {
        value.is_some_and(|value| !value.is_empty())
    }

    /// Java final `setHighlight(boolean)`.
    pub fn set_highlight(&self, highlight: bool) {
        self.highlight.set(highlight);
        self.set_background_void();
    }

    /// Java final `setWarning(boolean)`.
    pub fn set_warning_boolean(&self, warning: bool) {
        // if switching from warning to error, turn off error first
        if warning && self.error.get() {
            self.warning.set(false); // prevent recursion
            self.set_error_boolean(false);
        }
        self.warning.set(warning);
        self.set_background_void();
    }

    /// Java final `setWarning(boolean, String)`.
    pub fn set_warning_boolean_string(&self, warning: bool, tooltip: Option<&str>) {
        self.set_warning_boolean(warning);
        self.this().set_tool_tip_text(tooltip);
    }

    /// Java final `setError(boolean, String)`.
    pub fn set_error_boolean_string(&self, error: bool, tooltip: Option<&str>) {
        self.set_error_boolean(error);
        self.this().set_tool_tip_text(tooltip);
    }

    /// Java final `setError(boolean)`.
    pub fn set_error_boolean(&self, error: bool) {
        // if switching from warning to error, turn off warning first
        if error && self.warning.get() {
            self.error.set(false); // prevent recursion
            self.set_warning_boolean(false);
        }
        self.error.set(error);
        self.set_background_void();
    }

    /// Java `setRunHighlight(boolean)`.
    pub fn set_run_highlight(&self, run_highlight: bool) {
        self.run_highlight.set(run_highlight);
        self.set_background_void();
    }

    /// Java `setBackground()`.  Order of precedence: 1. error, 2. warning,
    /// 3. runHighlight, 4. highlight.
    pub fn set_background_void(&self) {
        let this = self.this();
        if self.error.get() {
            if this.is_enabled() {
                this.set_background_color_ui_resource(colors::CELL_ERROR_BACKGROUND);
            } else {
                this.set_background_color_ui_resource(colors::CELL_ERROR_BACKGROUND_NOT_EDITABLE);
            }
        } else if self.warning.get() {
            if this.is_enabled() {
                this.set_background_color_ui_resource(colors::WARNING_BACKGROUND);
            } else {
                this.set_background_color_ui_resource(colors::WARNING_BACKGROUND_NOT_EDITABLE);
            }
        } else if self.run_highlight.get() {
            if this.is_enabled() {
                this.set_background_color_ui_resource(colors::RUN_HIGHLIGHT_BACKGROUND);
            } else {
                this.set_background_color_ui_resource(
                    colors::RUN_HIGHLIGHT_BACKGROUND_NOT_EDITABLE,
                );
            }
        } else if self.highlight.get() {
            if this.is_enabled() {
                this.set_background_color_ui_resource(colors::HIGHLIGHT_BACKGROUND);
            } else {
                this.set_background_color_ui_resource(colors::HIGHLIGHT_BACKGROUND_NOT_EDITABLE);
            }
        } else if this.is_enabled() {
            if self.header_background {
                this.set_background_color_ui_resource(colors::HEADER_BACKGROUND);
            } else {
                this.set_background_color_ui_resource(colors::BACKGROUND);
            }
        } else {
            this.set_background_color_ui_resource(colors::get_cell_not_editable_background());
        }
    }

    /// Java `setFont()`.
    pub fn set_font(&self) {
        // Swing painting: plainFont = getComponent().getFont(); italicFont = new
        // Font(plainFont.getFontName(), Font.ITALIC, plainFont.getSize()).  Fonts are
        // not modelled and neither font is read anywhere else.
    }

    /// Java final `isHeaderBackground()`.
    pub fn is_header_background(&self) -> bool {
        self.header_background
    }

    /// Java `setBackground(ColorUIResource)` (the class's body).
    pub fn set_background_color_ui_resource_super(&self, _color: ColorUIResource) {
        // Swing painting: getComponent().setBackground(color).  Backgrounds are not
        // modelled by the jdk stand-in.
        let _ = self.this().get_component();
    }

    /// Java `setHeaders(String, HeaderCell, HeaderCell)`.
    pub fn set_headers(
        &self,
        table_header: Option<&str>,
        row_header: &Rc<HeaderCell>,
        column_header: &Rc<HeaderCell>,
    ) {
        *self.table_header.borrow_mut() = table_header.map(str::to_owned);
        *self.row_header.borrow_mut() = Some(row_header.clone());
        *self.column_header.borrow_mut() = Some(column_header.clone());
        self.set_name_void();
        let this = self.this();
        let child: Rc<dyn CellVirtual> = this.clone();
        row_header.add_child(Rc::downgrade(&child));
        column_header.add_child(Rc::downgrade(&child));
    }

    /// Java `msgLabelChanged()` (implements `Cell.msgLabelChanged`).  Message from
    /// row header or column header that their label has changed.
    pub fn msg_label_changed(&self) {
        self.set_name_void();
    }

    /// Java `convertLabelToName(boolean)`.
    pub fn convert_label_to_name(&self, unlimited_segments: bool) -> Option<String> {
        let table_header = self.table_header.borrow().clone();
        let row_header = self.row_header.borrow().clone();
        let column_header = self.column_header.borrow().clone();
        let row_text = row_header
            .as_ref()
            .and_then(|row_header| row_header.get_text());
        let column_text = column_header
            .as_ref()
            .and_then(|column_header| column_header.get_text());
        utilities::convert_label_to_name_three(
            table_header.as_deref(),
            row_text.as_deref(),
            column_text.as_deref(),
            unlimited_segments,
        )
    }

    /// Java `setName()`.  Build the name out of table header, row header, and column
    /// header.
    pub fn set_name_void(&self) {
        let this = self.this();
        let field_type = this.get_field_type();
        let name = self.convert_label_to_name(field_type.is_unlimited_segments());
        // Java string concatenation of a null name gives "null".
        this.get_component().set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                this.get_component().get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }
}
