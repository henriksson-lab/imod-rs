//! `IMOD/Etomo/src/etomo/ui/swing/SpacedLabel.java`.
//!
//! A label padded by rigid areas: five pixels either side and five below.

use std::rc::Rc;

use crate::imod::etomo::jdk::JComponent;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `final class SpacedLabel`.
pub struct SpacedLabel {
    /// Java `label`.
    label: Rc<JComponent>,
    /// Java `labelPanel`.
    label_panel: Rc<JComponent>,
    /// Java `yAxisPanel`.
    y_axis_panel: Rc<JComponent>,
}

impl SpacedLabel {
    /// Java `SpacedLabel(String)`.
    pub fn new(label: &str) -> SpacedLabel {
        // label = label.trim(): Java trims code points <= ' '.
        let label = label.trim_matches(|c: char| c <= ' ');
        let label = JComponent::new_label(label);
        // panels
        let y_axis_panel = JComponent::new_panel();
        // Swing layout: yAxisPanel.setLayout(new BoxLayout(yAxisPanel, BoxLayout.Y_AXIS)).
        let label_panel = JComponent::new_panel();
        // Swing layout: labelPanel.setLayout(new BoxLayout(labelPanel, BoxLayout.X_AXIS)).
        // labelPanel
        // Swing layout: labelPanel.add(Box.createRigidArea(FixedDim.x5_y0)).
        label_panel.add(&label);
        // Swing layout: labelPanel.add(Box.createRigidArea(FixedDim.x5_y0)).
        // yPanel
        y_axis_panel.add(&label_panel);
        // Swing layout: yAxisPanel.add(Box.createRigidArea(FixedDim.x0_y5)).
        SpacedLabel {
            label,
            label_panel,
            y_axis_panel,
        }
    }

    /// Java final `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, tool_tip_text: Option<&str>) {
        self.label.set_tool_tip_text(tool_tip_text);
        self.label_panel.set_tool_tip_text(tool_tip_text);
        self.y_axis_panel.set_tool_tip_text(tool_tip_text);
    }

    /// Java final `getContainer()`.  (`yAxisPanel` is never null after construction,
    /// so the Java's `labelPanel` fallback is not reachable.)
    pub fn get_container(&self) -> Rc<JComponent> {
        self.y_axis_panel.clone()
    }

    /// Java final `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.get_container().set_visible(visible);
    }

    /// Java final `setAlignmentX(float)`.
    pub fn set_alignment_x(&self, _alignment_x: f32) {
        // Swing layout: label, labelPanel and yAxisPanel .setAlignmentX(alignmentX).
    }
}
