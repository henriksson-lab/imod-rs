//! `IMOD/Etomo/src/etomo/ui/swing/SpacedLabel.java`.
#![allow(dead_code)]

use super::fixed_dim::FixedDim;

/// Java package-private final `SpacedLabel`, including all three Swing widgets.
#[derive(Clone, Debug, PartialEq)]
pub struct SpacedLabel {
    pub label: String,
    pub label_tooltip: Option<String>,
    pub label_panel_tooltip: Option<String>,
    pub y_axis_panel_tooltip: Option<String>,
    pub label_visible: bool,
    pub label_panel_visible: bool,
    pub y_axis_panel_visible: bool,
    pub label_alignment_x: f32,
    pub label_panel_alignment_x: f32,
    pub y_axis_panel_alignment_x: f32,
    pub label_panel_rigid_areas: [(i32, i32); 2],
    pub y_axis_rigid_area: (i32, i32),
}

impl SpacedLabel {
    /// Java `SpacedLabel(String)`.
    pub fn new(label: &str) -> Self {
        Self {
            label: label.trim().to_owned(),
            label_tooltip: None,
            label_panel_tooltip: None,
            y_axis_panel_tooltip: None,
            label_visible: true,
            label_panel_visible: true,
            y_axis_panel_visible: true,
            label_alignment_x: 0.5,
            label_panel_alignment_x: 0.5,
            y_axis_panel_alignment_x: 0.5,
            label_panel_rigid_areas: [
                (FixedDim::x5_y0.width, FixedDim::x5_y0.height),
                (FixedDim::x5_y0.width, FixedDim::x5_y0.height),
            ],
            y_axis_rigid_area: (FixedDim::x0_y5.width, FixedDim::x0_y5.height),
        }
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&mut self, tooltip: Option<&str>) {
        let tooltip = tooltip.map(str::to_owned);
        self.label_tooltip = tooltip.clone();
        self.label_panel_tooltip = tooltip.clone();
        self.y_axis_panel_tooltip = tooltip;
    }

    /// Java `getContainer()`; `true` identifies Java's non-null `yAxisPanel`.
    pub fn get_container_is_y_axis_panel(&self) -> bool {
        true
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.y_axis_panel_visible = visible;
    }

    /// Java `setAlignmentX(float)`.
    pub fn set_alignment_x(&mut self, alignment_x: f32) {
        self.label_alignment_x = alignment_x;
        self.label_panel_alignment_x = alignment_x;
        self.y_axis_panel_alignment_x = alignment_x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_nested_widgets_receive_tooltip_and_alignment() {
        let mut label = SpacedLabel::new("  Name  ");
        label.set_tool_tip_text(Some("tip"));
        label.set_alignment_x(0.0);
        assert_eq!(label.label, "Name");
        assert_eq!(label.label_tooltip.as_deref(), Some("tip"));
        assert_eq!(label.label_panel_tooltip.as_deref(), Some("tip"));
        assert_eq!(label.y_axis_panel_alignment_x, 0.0);
    }
}
