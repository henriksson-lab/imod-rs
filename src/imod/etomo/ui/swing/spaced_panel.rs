//! `IMOD/Etomo/src/etomo/ui/swing/SpacedPanel.java`.
//!
//! A panel that puts rigid-area spacing between the components added to it.
//! Box layouts, rigid areas, glue, alignment and backgrounds are Swing layout
//! and are recorded as comments; the component tree (outer panel -> inner
//! panel -> `EtomoPanel`) and the order of added components are kept.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use super::check_box::CheckBox;
use super::etomo_panel::EtomoPanel;
use super::file_text_field::FileTextField;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel_header::PanelHeader;
use super::radio_button::RadioButton;
use super::spaced_text_field::SpacedTextField;
use super::spinner::Spinner;
use super::text_field::TextField;
use super::tooltip_formatter;
use crate::imod::etomo::jdk::{JComponent, TitledBorder};

/// Java `BoxLayout.X_AXIS`.
pub const X_AXIS: i32 = 0;
/// Java `BoxLayout.Y_AXIS`.
pub const Y_AXIS: i32 = 1;

/// Java `SpacedPanel`.
pub struct SpacedPanel {
    /// Java `panel`.
    panel: Rc<EtomoPanel>,
    /// Java `innerPanel`.
    inner_panel: Rc<JComponent>,
    /// Java `outerPanel`.
    outer_panel: Rc<JComponent>,
    /// Java `layoutSet`.
    layout_set: Cell<bool>,
    /// Java `previousComponentWasSpaced`.
    previous_component_was_spaced: Cell<bool>,
    /// Java `componentAlignmentX`.
    component_alignment_x: RefCell<Option<f32>>,
    /// Java `axis`.
    axis: Cell<i32>,
}

impl SpacedPanel {
    /// Java `getInstance()`.
    pub fn get_instance_void() -> Rc<SpacedPanel> {
        Self::new(false, false)
    }

    /// Java `getInstance(boolean)`.
    pub fn get_instance_boolean(y_axis_padding: bool) -> Rc<SpacedPanel> {
        Self::new(y_axis_padding, false)
    }

    /// Java `getFocusableInstance()`.
    pub fn get_focusable_instance_void() -> Rc<SpacedPanel> {
        Self::new(false, true)
    }

    /// Java `getFocusableInstance(boolean)`.
    pub fn get_focusable_instance_boolean(y_axis_padding: bool) -> Rc<SpacedPanel> {
        Self::new(y_axis_padding, true)
    }

    /// Java private `SpacedPanel(boolean, boolean)`.
    fn new(y_axis_padding: bool, focusable: bool) -> Rc<SpacedPanel> {
        // panels.  A focusable panel is Java's private `FocusablePanel extends JPanel`
        // (`setFocusable(true)`); focus is not modelled.
        let _ = focusable;
        let outer_panel = JComponent::new_panel();
        let inner_panel = JComponent::new_panel();
        let panel = EtomoPanel::new();
        // Swing layout: outerPanel BoxLayout Y_AXIS, innerPanel BoxLayout X_AXIS.
        // innerPanel: rigid area x5, panel, rigid area x5.
        inner_panel.add(&panel.get_component());
        // outerPanel: optional rigid area y5 (yAxisPadding), innerPanel, rigid area y5.
        let _ = y_axis_padding;
        outer_panel.add(&inner_panel);
        Rc::new(SpacedPanel {
            panel,
            inner_panel,
            outer_panel,
            layout_set: Cell::new(false),
            previous_component_was_spaced: Cell::new(false),
            component_alignment_x: RefCell::new(None),
            axis: Cell::new(0),
        })
    }

    /// Java `requestFocus()`: focus is not modelled.
    pub fn request_focus(&self) {}

    /// Java `setBackground(Color)`: painting only.
    pub fn set_background(&self, _color: (u8, u8, u8)) {}

    /// Java `setBoxLayout(int)`.
    pub fn set_box_layout(&self, axis: i32) {
        self.axis.set(axis);
        // Swing layout: `panel.setLayout(new BoxLayout(panel, axis))`.
        self.layout_set.set(true);
    }

    /// Java private `addSpacing()`.  The rigid areas themselves are layout; the
    /// `previousComponentWasSpaced` bookkeeping is kept.
    fn add_spacing(&self) {
        if !self.layout_set.get() {
            return;
        }
        if self.axis.get() == X_AXIS {
            // Swing layout: `panel.add(Box.createRigidArea(FixedDim.x5_y0))`.
        } else if self.axis.get() == Y_AXIS && !self.previous_component_was_spaced.get() {
            // Swing layout: `panel.add(Box.createRigidArea(FixedDim.x0_y5))`.
        }
        if self.previous_component_was_spaced.get() {
            self.previous_component_was_spaced.set(false);
        }
    }

    /// Java `remove(SpacedPanel)`.
    pub fn remove(&self, panel: &SpacedPanel) {
        self.panel.get_component().remove(&panel.get_container());
    }

    /// Java `add(Container)`.
    pub fn add_container(&self, container: &Rc<JComponent>) {
        self.panel.get_component().add(container);
        self.add_spacing();
    }

    /// Java `add(Component)`.
    pub fn add_component(&self, component: &Rc<JComponent>) {
        self.panel.get_component().add(component);
        self.add_spacing();
    }

    /// Java `add(SpacedPanel)`.
    pub fn add_spaced_panel(&self, spaced_panel: &SpacedPanel) {
        self.panel
            .get_component()
            .add(&spaced_panel.get_container());
        self.add_spacing();
        if let Some(alignment) = *self.component_alignment_x.borrow() {
            spaced_panel.set_component_alignment_x(alignment);
        }
        self.previous_component_was_spaced.set(true);
    }

    /// Java `add(JScrollPane)`.
    pub fn add_j_scroll_pane(&self, j_scroll_pane: &Rc<JComponent>) {
        self.panel.get_component().add(j_scroll_pane);
        self.add_spacing();
        // Swing layout: `jScrollPane.setAlignmentX(componentAlignmentX)` when set.
    }

    /// Java `add(JPanel)`.
    pub fn add_j_panel(&self, j_panel: &Rc<JComponent>) {
        self.panel.get_component().add(j_panel);
        self.add_spacing();
        // Swing layout: `jPanel.setAlignmentX(componentAlignmentX)` when set.
    }

    /// Java `add(JComboBox)`.
    pub fn add_j_combo_box(&self, j_combo_box: &Rc<JComponent>) {
        self.panel.get_component().add(j_combo_box);
        self.add_spacing();
        // Swing layout: `jComboBox.setAlignmentX(componentAlignmentX)` when set.
    }

    /// Java `add(FileTextField)`.
    pub fn add_file_text_field(&self, file_text_field: &FileTextField) {
        self.panel
            .get_component()
            .add(&file_text_field.get_container());
        self.add_spacing();
        if let Some(alignment) = *self.component_alignment_x.borrow() {
            file_text_field.set_alignment_x(alignment);
        }
    }

    /// Java `add(LabeledTextField)`.
    pub fn add_labeled_text_field(&self, labeled_text_field: &LabeledTextField) {
        self.panel
            .get_component()
            .add(&labeled_text_field.get_container());
        self.add_spacing();
        if let Some(alignment) = *self.component_alignment_x.borrow() {
            labeled_text_field.set_alignment_x(alignment);
        }
    }

    /// Java `add(Spinner)`.
    pub fn add_spinner(&self, spinner: &Spinner) {
        self.panel.get_component().add(&spinner.get_container());
        self.add_spacing();
        if let Some(alignment) = *self.component_alignment_x.borrow() {
            spinner.set_alignment_x(alignment);
        }
    }

    /// Java `add(TextField)`.
    pub fn add_text_field(&self, text_field: &TextField) {
        self.panel.get_component().add(&text_field.get_component());
        self.add_spacing();
        if let Some(alignment) = *self.component_alignment_x.borrow() {
            text_field.set_alignment_x(alignment);
        }
    }

    /// Java `add(JLabel)`.
    pub fn add_j_label(&self, j_label: &Rc<JComponent>) {
        self.panel.get_component().add(j_label);
        self.add_spacing();
        // Swing layout: `jLabel.setAlignmentX(componentAlignmentX)` when set.
    }

    /// Java `add(RadioButton)`.
    pub fn add_radio_button(&self, radio_button: &RadioButton) {
        self.panel
            .get_component()
            .add(&radio_button.get_component());
        self.add_spacing();
        if let Some(alignment) = *self.component_alignment_x.borrow() {
            radio_button.set_alignment_x(alignment);
        }
    }

    /// Java `add(CheckBox)`.
    pub fn add_check_box(&self, check_box: &CheckBox) {
        self.panel.get_component().add(&check_box.get_component());
        self.add_spacing();
        if let Some(alignment) = *self.component_alignment_x.borrow() {
            check_box.set_alignment_x(alignment);
        }
    }

    /// Java `add(SpacedTextField)`.
    pub fn add_spaced_text_field(&self, spaced_text_field: &SpacedTextField) {
        self.panel
            .get_component()
            .add(&spaced_text_field.get_container());
        self.add_spacing();
        if let Some(alignment) = *self.component_alignment_x.borrow() {
            spaced_text_field.set_alignment_x(alignment);
        }
        self.previous_component_was_spaced.set(true);
    }

    /// Java `add(PanelHeader)`.
    pub fn add_panel_header(&self, panel_header: &PanelHeader) {
        // `panel.add(panelHeader)` is EtomoPanel's `add(PanelHeader)` overload,
        // which also names the panel.
        self.panel.add(panel_header);
        self.add_spacing();
    }

    /// Java `add(LabeledSpinner)`.
    pub fn add_labeled_spinner(&self, labeled_spinner: &LabeledSpinner) {
        self.panel
            .get_component()
            .add(&labeled_spinner.get_container());
        self.add_spacing();
        if let Some(alignment) = *self.component_alignment_x.borrow() {
            labeled_spinner.set_alignment_x(alignment);
        }
    }

    /// Java `add(MultiLineButton)`.
    pub fn add_multi_line_button(&self, multi_line_button: &MultiLineButton) {
        self.panel
            .get_component()
            .add(&multi_line_button.get_component());
        self.add_spacing();
        // Swing layout: `multiLineButton.setAlignmentX(componentAlignmentX)` when set.
    }

    /// Java `add(JButton)`.
    pub fn add_j_button(&self, button: &Rc<JComponent>) {
        self.panel.get_component().add(button);
        self.add_spacing();
        // Swing layout: `button.setAlignmentX(componentAlignmentX)` when set.
    }

    /// Java `addMouseListener(MouseListener)`.
    pub fn add_mouse_listener(&self, mouse_listener: Rc<dyn crate::imod::etomo::jdk::MouseListener>) {
        self.panel.get_component().add_mouse_listener(mouse_listener);
    }

    /// Java `addHorizontalGlue()`.
    pub fn add_horizontal_glue(&self) {
        // Swing layout: `panel.add(Box.createHorizontalGlue())`.
    }

    /// Java `addRigidArea()`.
    pub fn add_rigid_area_void(&self) {
        // Swing layout: a 5-pixel rigid area along the axis.
    }

    /// Java `addRigidArea(Dimension)`.
    pub fn add_rigid_area_dimension(&self, _dim: (i32, i32)) {
        // Swing layout: a rigid area of `dim` along the axis.
    }

    /// Java `removeAll()`.
    pub fn remove_all(&self) {
        self.panel.get_component().remove_all();
    }

    /// Java `setComponentAlignmentX(float)`.
    pub fn set_component_alignment_x(&self, component_alignment_x: f32) {
        *self.component_alignment_x.borrow_mut() = Some(component_alignment_x);
    }

    /// Java `alignComponentsX(float)`.
    pub fn align_components_x(&self, alignment: f32) {
        self.set_component_alignment_x(alignment);
        // Swing layout: `setAlignmentX(alignment)` on every child of the outer,
        // inner and content panels (the private `alignComponentsX(JPanel, float)`).
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.panel
            .get_component()
            .set_tool_tip_text(tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.outer_panel.clone()
    }

    /// Java `getJPanel()`.
    pub fn get_j_panel(&self) -> Rc<JComponent> {
        self.get_container()
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.panel.get_component().get_name()
    }

    /// Java `setName(String)`: `panel.setName(name)`, which dispatches to
    /// EtomoPanel's `setName` override (the uitest panel name).
    pub fn set_name(&self, name: Option<&str>) {
        self.panel.set_name(name);
    }

    /// Java `setAlignmentX(float)`: layout only.
    pub fn set_alignment_x(&self, _alignment_x: f32) {}

    /// Java `setBorder(TitledBorder)`.  (Java's `setBorder(Border)` overload is
    /// painting only; the titled one names the panel through EtomoPanel.)
    pub fn set_border(&self, border: &TitledBorder) {
        self.panel.set_border(border);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.panel.get_component().set_enabled(enabled);
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.get_container().set_visible(visible);
    }
}
