//! `IMOD/Etomo/src/etomo/ui/swing/SpacedPanel.java`.
//!
//! Swing owns the actual `JPanel`, `BoxLayout`, `Box`, borders, focus, and
//! listener objects.  This source unit retains the ordered component tree and
//! the source's spacing/alignment decisions at that GUI boundary.
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

use super::abstract_frame::ComponentState;
use super::check_box::CheckBox;
use super::etomo_panel::{EtomoPanel, TitledBorder};
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::panel::Dimension;
use super::panel_header::PanelHeader;
use super::radio_button::RadioButton;
use super::spaced_text_field::SpacedTextField;
use super::ui_utilities::{Color, X0_Y5, X5_Y0};

/// Java `BoxLayout.X_AXIS`.
pub const X_AXIS: i32 = 0;
/// Java `BoxLayout.Y_AXIS`.
pub const Y_AXIS: i32 = 1;
static NEXT_CONTAINER_ID: AtomicUsize = AtomicUsize::new(1);

/// State held by the Swing `JPanel` instances constructed by this unit.
#[derive(Clone, Debug, Default)]
pub struct JPanel {
    pub layout_axis: Option<i32>,
    pub children: Vec<SpacedPanelChild>,
    pub background: Option<Color>,
    pub alignment_x: f32,
    pub visible: bool,
    pub enabled: bool,
    pub focusable: bool,
}

/// Java's private `FocusablePanel`.
#[derive(Clone, Debug, Default)]
pub struct FocusablePanel {
    pub panel: JPanel,
}

impl FocusablePanel {
    /// Java `FocusablePanel()`.
    pub fn new() -> Self {
        Self {
            panel: JPanel {
                focusable: true,
                visible: true,
                enabled: true,
                ..Default::default()
            },
        }
    }
}

/// GUI-boundary forms of Swing components not otherwise owned by a translated
/// source unit.  Their exact insertion order is what `SpacedPanel` observes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Container {
    pub component: ComponentState,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JScrollPane {
    pub alignment_x: f32,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JComboBox {
    pub alignment_x: f32,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JLabel {
    pub text: Option<String>,
    pub alignment_x: f32,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JButton {
    pub text: Option<String>,
    pub alignment_x: f32,
}
/// Java `javax.swing.border.Border` state used by this source unit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Border {
    pub description: Option<String>,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FileTextField {
    pub alignment_x: f32,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Spinner {
    pub alignment_x: f32,
}
/// GUI boundary for Java's separate package-private `TextField` source unit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextField {
    pub alignment_x: f32,
}

/// A source-visible entry in the three nested Swing panels.
#[derive(Clone, Debug)]
pub enum SpacedPanelChild {
    RigidArea(Dimension),
    HorizontalGlue,
    Container(Container),
    Component(ComponentState),
    SpacedPanel(Box<SpacedPanel>),
    JScrollPane(JScrollPane),
    JPanel(JPanel),
    JComboBox(JComboBox),
    FileTextField(FileTextField),
    LabeledTextField(LabeledTextField),
    Spinner(Spinner),
    TextField(TextField),
    JLabel(JLabel),
    RadioButton(RadioButton),
    CheckBox(CheckBox),
    SpacedTextField(SpacedTextField),
    PanelHeader(PanelHeader),
    LabeledSpinner(LabeledSpinner),
    MultiLineButton(MultiLineButton),
    JButton(JButton),
}

/// Java final `SpacedPanel` fields and methods.
#[derive(Clone, Debug)]
pub struct SpacedPanel {
    pub panel: EtomoPanel,
    pub inner_panel: JPanel,
    pub outer_panel: JPanel,
    pub layout_set: bool,
    pub previous_component_was_spaced: bool,
    pub component_alignment_x: Option<f32>,
    pub axis: i32,
    pub mouse_listener_count: usize,
    pub focused: bool,
    /// The Swing outer-container identity used by Java `remove(SpacedPanel)`.
    pub container_id: usize,
    /// Children directly added to Java's `panel`.  `EtomoPanel` predates this
    /// source unit and stores only generic component state, so the precise
    /// `Box`/widget ordering belongs here.
    pub panel_children: Vec<SpacedPanelChild>,
    pub panel_background: Option<Color>,
    pub panel_layout_axis: Option<i32>,
    pub tooltip: Option<String>,
    pub border: Option<Border>,
    pub titled_border: Option<TitledBorder>,
}

impl SpacedPanel {
    /// Java package-private `getInstance()`.
    pub fn get_instance() -> Self {
        Self::new(false, false)
    }

    /// Java `getInstance(boolean)`.
    pub fn get_instance_y_axis_padding(y_axis_padding: bool) -> Self {
        Self::new(y_axis_padding, false)
    }

    /// Java package-private `getFocusableInstance()`.
    pub fn get_focusable_instance() -> Self {
        Self::new(false, true)
    }

    /// Java package-private `getFocusableInstance(boolean)`.
    pub fn get_focusable_instance_y_axis_padding(y_axis_padding: bool) -> Self {
        Self::new(y_axis_padding, true)
    }

    /// Java private `SpacedPanel(boolean, boolean)`.
    fn new(y_axis_padding: bool, focusable: bool) -> Self {
        let mut inner_panel = JPanel {
            layout_axis: Some(X_AXIS),
            visible: true,
            enabled: true,
            ..Default::default()
        };
        inner_panel
            .children
            .push(SpacedPanelChild::RigidArea(X5_Y0));
        inner_panel
            .children
            .push(SpacedPanelChild::Component(ComponentState::default()));
        inner_panel
            .children
            .push(SpacedPanelChild::RigidArea(X5_Y0));
        let mut outer_panel = if focusable {
            FocusablePanel::new().panel
        } else {
            JPanel {
                visible: true,
                enabled: true,
                ..Default::default()
            }
        };
        outer_panel.layout_axis = Some(Y_AXIS);
        if y_axis_padding {
            outer_panel
                .children
                .push(SpacedPanelChild::RigidArea(X0_Y5));
        }
        outer_panel
            .children
            .push(SpacedPanelChild::JPanel(inner_panel.clone()));
        outer_panel
            .children
            .push(SpacedPanelChild::RigidArea(X0_Y5));
        Self {
            panel: EtomoPanel::default(),
            inner_panel,
            outer_panel,
            layout_set: false,
            previous_component_was_spaced: false,
            component_alignment_x: None,
            axis: 0,
            mouse_listener_count: 0,
            focused: false,
            container_id: NEXT_CONTAINER_ID.fetch_add(1, Ordering::Relaxed),
            panel_children: Vec::new(),
            panel_background: None,
            panel_layout_axis: None,
            tooltip: None,
            border: None,
            titled_border: None,
        }
    }

    /// Java `requestFocus()`.
    pub fn request_focus(&mut self) {
        self.focused = true;
    }

    /// Java `setBackground(Color)`.
    pub fn set_background(&mut self, color: Color) {
        self.panel_background = Some(color);
        self.inner_panel.background = Some(color);
        self.outer_panel.background = Some(color);
    }

    /// Java `setBoxLayout(int)`.
    pub fn set_box_layout(&mut self, axis: i32) {
        self.axis = axis;
        self.panel_layout_axis = Some(axis);
        self.layout_set = true;
    }

    /// Java private `addSpacing()`.
    fn add_spacing(&mut self) {
        if !self.layout_set {
            return;
        }
        if self.axis == X_AXIS {
            self.panel_children.push(SpacedPanelChild::RigidArea(X5_Y0));
        } else if self.axis == Y_AXIS && !self.previous_component_was_spaced {
            self.panel_children.push(SpacedPanelChild::RigidArea(X0_Y5));
        }
        if self.previous_component_was_spaced {
            self.previous_component_was_spaced = false;
        }
    }

    /// Java `remove(SpacedPanel)`.
    pub fn remove(&mut self, spaced_panel: &SpacedPanel) {
        if let Some(index) = self.panel_children.iter().position(|child| {
            matches!(child, SpacedPanelChild::SpacedPanel(candidate) if candidate.container_id == spaced_panel.container_id)
        }) {
            self.panel_children.remove(index);
        }
    }

    /// Java overloaded `add(Container)`.
    pub fn add_container(&mut self, container: Container) {
        self.panel_children
            .push(SpacedPanelChild::Container(container));
        self.add_spacing();
    }

    /// Java overloaded `add(Component)`.
    pub fn add_component(&mut self, component: ComponentState) {
        self.panel_children
            .push(SpacedPanelChild::Component(component));
        self.add_spacing();
    }

    /// Java overloaded `add(SpacedPanel)`.
    pub fn add_spaced_panel(&mut self, mut spaced_panel: SpacedPanel) {
        if let Some(alignment) = self.component_alignment_x {
            spaced_panel.set_component_alignment_x(alignment);
        }
        self.panel_children
            .push(SpacedPanelChild::SpacedPanel(Box::new(spaced_panel)));
        self.add_spacing();
        self.previous_component_was_spaced = true;
    }

    /// Java overloaded `add(JScrollPane)`.
    pub fn add_j_scroll_pane(&mut self, mut j_scroll_pane: JScrollPane) {
        if let Some(alignment) = self.component_alignment_x {
            j_scroll_pane.alignment_x = alignment;
        }
        self.panel_children
            .push(SpacedPanelChild::JScrollPane(j_scroll_pane));
        self.add_spacing();
    }

    /// Java overloaded `add(JPanel)`.
    pub fn add_j_panel(&mut self, mut j_panel: JPanel) {
        if let Some(alignment) = self.component_alignment_x {
            j_panel.alignment_x = alignment;
        }
        self.panel_children.push(SpacedPanelChild::JPanel(j_panel));
        self.add_spacing();
    }

    /// Java overloaded `add(JComboBox)`.
    pub fn add_j_combo_box(&mut self, mut j_combo_box: JComboBox) {
        if let Some(alignment) = self.component_alignment_x {
            j_combo_box.alignment_x = alignment;
        }
        self.panel_children
            .push(SpacedPanelChild::JComboBox(j_combo_box));
        self.add_spacing();
    }

    /// Java overloaded `add(FileTextField)`.
    pub fn add_file_text_field(&mut self, mut file_text_field: FileTextField) {
        if let Some(alignment) = self.component_alignment_x {
            file_text_field.alignment_x = alignment;
        }
        self.panel_children
            .push(SpacedPanelChild::FileTextField(file_text_field));
        self.add_spacing();
    }

    /// Java overloaded `add(LabeledTextField)`.
    pub fn add_labeled_text_field(&mut self, mut labeled_text_field: LabeledTextField) {
        if let Some(alignment) = self.component_alignment_x {
            labeled_text_field.set_alignment_x(alignment);
        }
        self.panel_children
            .push(SpacedPanelChild::LabeledTextField(labeled_text_field));
        self.add_spacing();
    }

    /// Java overloaded `add(Spinner)`.
    pub fn add_spinner(&mut self, mut spinner: Spinner) {
        if let Some(alignment) = self.component_alignment_x {
            spinner.alignment_x = alignment;
        }
        self.panel_children.push(SpacedPanelChild::Spinner(spinner));
        self.add_spacing();
    }

    /// Java overloaded `add(TextField)`.
    pub fn add_text_field(&mut self, mut text_field: TextField) {
        if let Some(alignment) = self.component_alignment_x {
            text_field.alignment_x = alignment;
        }
        self.panel_children
            .push(SpacedPanelChild::TextField(text_field));
        self.add_spacing();
    }

    /// Java overloaded `add(JLabel)`.
    pub fn add_j_label(&mut self, mut j_label: JLabel) {
        if let Some(alignment) = self.component_alignment_x {
            j_label.alignment_x = alignment;
        }
        self.panel_children.push(SpacedPanelChild::JLabel(j_label));
        self.add_spacing();
    }

    /// Java overloaded `add(RadioButton)`.
    pub fn add_radio_button(&mut self, mut radio_button: RadioButton) {
        if let Some(alignment) = self.component_alignment_x {
            radio_button.set_alignment_x(alignment);
        }
        self.panel_children
            .push(SpacedPanelChild::RadioButton(radio_button));
        self.add_spacing();
    }

    /// Java overloaded `add(CheckBox)`.
    pub fn add_check_box(&mut self, mut check_box: CheckBox) {
        if let Some(alignment) = self.component_alignment_x {
            check_box.set_alignment_x(alignment);
        }
        self.panel_children
            .push(SpacedPanelChild::CheckBox(check_box));
        self.add_spacing();
    }

    /// Java overloaded `add(SpacedTextField)`.
    pub fn add_spaced_text_field(&mut self, mut spaced_text_field: SpacedTextField) {
        if let Some(alignment) = self.component_alignment_x {
            spaced_text_field.set_alignment_x(alignment);
        }
        self.panel_children
            .push(SpacedPanelChild::SpacedTextField(spaced_text_field));
        self.add_spacing();
        self.previous_component_was_spaced = true;
    }

    /// Java overloaded `add(PanelHeader)`.
    pub fn add_panel_header(&mut self, panel_header: PanelHeader) {
        self.panel_children
            .push(SpacedPanelChild::PanelHeader(panel_header));
        self.add_spacing();
    }

    /// Java overloaded `add(LabeledSpinner)`.
    pub fn add_labeled_spinner(&mut self, mut labeled_spinner: LabeledSpinner) {
        if let Some(alignment) = self.component_alignment_x {
            labeled_spinner.set_alignment_x(alignment);
        }
        self.panel_children
            .push(SpacedPanelChild::LabeledSpinner(labeled_spinner));
        self.add_spacing();
    }

    /// Java overloaded `add(MultiLineButton)`.
    pub fn add_multi_line_button(&mut self, mut multi_line_button: MultiLineButton) {
        if let Some(alignment) = self.component_alignment_x {
            multi_line_button.set_alignment_x(alignment);
        }
        self.panel_children
            .push(SpacedPanelChild::MultiLineButton(multi_line_button));
        self.add_spacing();
    }

    /// Java overloaded `add(JButton)`.
    pub fn add_j_button(&mut self, mut button: JButton) {
        if let Some(alignment) = self.component_alignment_x {
            button.alignment_x = alignment;
        }
        self.panel_children.push(SpacedPanelChild::JButton(button));
        self.add_spacing();
    }

    /// Java `addMouseListener(MouseListener)`.
    pub fn add_mouse_listener(&mut self) {
        self.mouse_listener_count += 1;
    }
    /// Java `addHorizontalGlue()`.
    pub fn add_horizontal_glue(&mut self) {
        self.panel_children.push(SpacedPanelChild::HorizontalGlue);
    }
    /// Java `addRigidArea()`.
    pub fn add_rigid_area(&mut self) {
        if self.axis == X_AXIS {
            self.panel_children.push(SpacedPanelChild::RigidArea(X5_Y0));
        } else if self.axis == Y_AXIS {
            self.panel_children.push(SpacedPanelChild::RigidArea(X0_Y5));
        }
    }
    /// Java `addRigidArea(Dimension)`.
    pub fn add_rigid_area_dimension(&mut self, dimension: Dimension) {
        if self.axis == X_AXIS || self.axis == Y_AXIS {
            self.panel_children
                .push(SpacedPanelChild::RigidArea(dimension));
        }
    }
    /// Java `removeAll()`.
    pub fn remove_all(&mut self) {
        self.panel_children.clear();
    }
    /// Java `setComponentAlignmentX(float)`.
    pub fn set_component_alignment_x(&mut self, component_alignment_x: f32) {
        self.component_alignment_x = Some(component_alignment_x);
    }
    /// Java `alignComponentsX(float)`.
    pub fn align_components_x(&mut self, alignment: f32) {
        self.set_component_alignment_x(alignment);
        self.outer_panel.alignment_x = alignment;
        self.inner_panel.alignment_x = alignment;
    }
    /// Java `setToolTipText(String)`; formatter remains a presentation boundary.
    pub fn set_tool_tip_text(&mut self, text: Option<&str>) {
        self.tooltip = text.map(str::to_owned);
    }
    /// Java private overloaded `alignComponentsX(JPanel, float)`.
    fn align_components_x_j_panel(panel: &mut JPanel, alignment: f32) {
        panel.alignment_x = alignment;
    }
    /// Java `getContainer()`.
    pub fn get_container(&self) -> &JPanel {
        &self.outer_panel
    }
    /// Java package-private `getJPanel()`.
    pub fn get_j_panel(&self) -> &JPanel {
        self.get_container()
    }
    /// Java `getName()`.
    pub fn get_name(&self) -> Option<&str> {
        self.panel.name.as_deref()
    }
    /// Java `setName(String)`.
    pub fn set_name(&mut self, name: &str) {
        self.panel.set_name(name);
    }
    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&mut self, alignment_x: f32) {
        self.outer_panel.alignment_x = alignment_x;
    }
    /// Java overloaded `setBorder(Border)`.
    pub fn set_border(&mut self, border: Border) {
        self.border = Some(border);
    }
    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.outer_panel.enabled = enabled;
    }
    /// Java overloaded `setBorder(TitledBorder)`.
    pub fn set_titled_border(&mut self, border: TitledBorder) {
        self.titled_border = Some(border.clone());
        self.panel.set_border(border);
    }
    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.outer_panel.visible = visible;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_has_java_nested_padding_and_optional_y_padding() {
        let panel = SpacedPanel::get_instance_y_axis_padding(true);
        assert_eq!(panel.inner_panel.layout_axis, Some(X_AXIS));
        assert_eq!(panel.outer_panel.layout_axis, Some(Y_AXIS));
        assert_eq!(panel.outer_panel.children.len(), 3);
        assert_eq!(panel.inner_panel.children.len(), 3);
    }

    #[test]
    fn y_spacing_skips_once_after_spaced_child() {
        let mut panel = SpacedPanel::get_instance();
        panel.set_box_layout(Y_AXIS);
        panel.add_spaced_text_field(SpacedTextField::new(
            crate::imod::etomo::ui::field_type::FieldType::String,
            "Field",
        ));
        assert!(panel.previous_component_was_spaced);
        panel.add_j_label(JLabel::default());
        assert!(!panel.previous_component_was_spaced);
        assert_eq!(panel.panel_children.len(), 3);
    }

    #[test]
    fn alignment_is_applied_before_source_component_is_inserted() {
        let mut panel = SpacedPanel::get_instance();
        panel.set_component_alignment_x(0.25);
        panel.add_j_button(JButton::default());
        assert!(
            matches!(panel.panel_children.last(), Some(SpacedPanelChild::JButton(button)) if button.alignment_x == 0.25)
        );
    }
}
