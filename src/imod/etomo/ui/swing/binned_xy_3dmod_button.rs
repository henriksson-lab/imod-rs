//! `IMOD/Etomo/src/etomo/ui/swing/BinnedXY3dmodButton.java`.
//!
//! `JPanel`, `BoxLayout`, `Box`, `JLabel`, etched borders, action listeners,
//! `TooltipFormatter`, and the eventual `Run3dmodButtonContainer` dispatch are
//! native GUI/application boundaries.  This unit retains the source's lazy
//! component hierarchy and all state passed across those boundaries.
#![allow(dead_code)]

use super::labeled_spinner::LabeledSpinner;
use super::multi_line_button::MultiLineButton;

/// Java `Run3dmodButtonContainer`, which is called by the unported right-click
/// 3dmod menu adapter owned by `Run3dmodButton`.
pub trait Run3dmodButtonContainer {}

/// Java `Run3dmodButton` state used directly by this source unit.  The
/// context-menu and manager dispatch remain the explicit GUI/application
/// boundary of the separately translated `Run3dmodButton.java` source unit.
#[derive(Clone, Debug, PartialEq)]
pub struct Run3dmodButton {
    pub multi_line_button: MultiLineButton,
    pub deferred: bool,
    pub container_attached: bool,
    pub mouse_listener_count: usize,
}

impl Run3dmodButton {
    /// Java `Run3dmodButton.get3dmodInstance(String, Run3dmodButtonContainer)`.
    pub fn get_3dmod_instance<C: Run3dmodButtonContainer>(label: &str, _container: &C) -> Self {
        let mut multi_line_button = MultiLineButton::new_with_label(Some(label));
        // Java `AbstractButton` defaults its action command to the button text.
        multi_line_button.set_action_command(Some(label));
        multi_line_button.add_mouse_listener();
        Self {
            multi_line_button,
            deferred: false,
            container_attached: true,
            mouse_listener_count: 1,
        }
    }

    /// Java inherited `getComponent`.
    pub fn get_component(&self) -> &MultiLineButton {
        &self.multi_line_button
    }

    /// Java inherited `setToolTipText`.
    pub fn set_tool_tip_text(&mut self, text: &str) {
        self.multi_line_button.set_tool_tip_text(Some(text));
    }

    /// Java inherited `addActionListener`.
    pub fn add_action_listener(&mut self) {
        self.multi_line_button.add_action_listener();
    }

    /// Java inherited `getActionCommand`.
    pub fn get_action_command(&self) -> Option<&str> {
        self.multi_line_button.get_action_command()
    }

    /// Java inherited `setEnabled`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.multi_line_button.set_enabled(enabled);
    }
}

/// Source-visible, lazily-created Swing panel tree returned by `getContainer`.
#[derive(Clone, Debug, PartialEq)]
pub struct BinnedXY3dmodButtonContainer {
    pub box_layout_axis: i32,
    pub alignment_x: f32,
    pub leading_horizontal_glue_count: usize,
    pub trailing_horizontal_glue_count: usize,
    pub border_box_layout_axis: i32,
    pub border_etched: bool,
    pub border_alignment_x: f32,
    pub spinner_box_layout_axis: i32,
    pub spinner_alignment_x: f32,
    pub button_box_layout_axis: i32,
    pub button_alignment_x: f32,
}

/// Java package-private final `BinnedXY3dmodButton`.
#[derive(Clone, Debug)]
pub struct BinnedXY3dmodButton {
    pub button: Run3dmodButton,
    pub sp_binning_xy: LabeledSpinner,
    pub label: String,
    pub label_enabled: bool,
    pub label_tooltip: Option<String>,
    pub panel: Option<BinnedXY3dmodButtonContainer>,
}

impl BinnedXY3dmodButton {
    /// Java package-private constructor.
    pub fn new<C: Run3dmodButtonContainer>(label: &str, container: &C) -> Self {
        Self {
            sp_binning_xy: LabeledSpinner::get_instance("Open binned by ", 1, 1, 50, 1),
            label: " in X and Y".into(),
            button: Run3dmodButton::get_3dmod_instance(label, container),
            label_enabled: true,
            label_tooltip: None,
            panel: None,
        }
    }

    /// Java `getContainer`.  Widget creation is intentionally lazy and its
    /// exact BoxLayout/glue/border insertion sequence is retained as state.
    pub fn get_container(&mut self) -> &BinnedXY3dmodButtonContainer {
        if self.panel.is_none() {
            self.panel = Some(BinnedXY3dmodButtonContainer {
                box_layout_axis: 0,
                alignment_x: 0.5,
                leading_horizontal_glue_count: 3,
                trailing_horizontal_glue_count: 3,
                border_box_layout_axis: 1,
                border_etched: true,
                border_alignment_x: 0.5,
                spinner_box_layout_axis: 0,
                spinner_alignment_x: 0.5,
                button_box_layout_axis: 0,
                button_alignment_x: 0.5,
            });
        }
        self.panel.as_ref().expect("Java panel assigned above")
    }

    /// Java `getButton` returns its `Deferred3dmodButton` implementation.
    pub fn get_button(&self) -> &Run3dmodButton {
        &self.button
    }

    /// Java `setSpinnerToolTipText`; `TooltipFormatter` remains the native
    /// presentation boundary and receives the same unformatted source text.
    pub fn set_spinner_tool_tip_text(&mut self, text: &str) {
        self.sp_binning_xy.set_tool_tip_text(Some(text));
        self.label_tooltip = Some(text.into());
    }

    /// Java `setButtonToolTipText`.
    pub fn set_button_tool_tip_text(&mut self, text: &str) {
        self.button.set_tool_tip_text(text);
    }

    /// Java `addActionListener`.
    pub fn add_action_listener(&mut self) {
        self.button.add_action_listener();
    }

    /// Java `getActionCommand`.
    pub fn get_action_command(&self) -> Option<&str> {
        self.button.get_action_command()
    }

    /// Java `getBinningInXandY`.
    pub fn get_binning_in_x_and_y(&self) -> i32 {
        self.sp_binning_xy.get_value()
    }

    /// Java `setEnabled`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.sp_binning_xy.set_enabled(enabled);
        self.label_enabled = enabled;
        self.button.set_enabled(enabled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Container;
    impl Run3dmodButtonContainer for Container {}

    #[test]
    fn source_constructor_and_lazy_container_preserve_widget_state() {
        let mut button = BinnedXY3dmodButton::new("Open in 3dmod", &Container);
        assert_eq!(button.sp_binning_xy.get_value(), 1);
        assert_eq!(button.sp_binning_xy.minimum, 1);
        assert_eq!(button.sp_binning_xy.maximum, 50);
        assert_eq!(button.label, " in X and Y");
        assert!(button.panel.is_none());
        let panel = button.get_container();
        assert_eq!(panel.box_layout_axis, 0);
        assert_eq!(panel.leading_horizontal_glue_count, 3);
        assert_eq!(panel.trailing_horizontal_glue_count, 3);
        assert!(panel.border_etched);
        assert_eq!(panel.border_box_layout_axis, 1);
    }

    #[test]
    fn source_tooltips_actions_binning_and_enabled_state_are_forwarded() {
        let mut button = BinnedXY3dmodButton::new("Open in 3dmod", &Container);
        button.set_spinner_tool_tip_text("bin images");
        button.set_button_tool_tip_text("open images");
        button.add_action_listener();
        button.sp_binning_xy.set_value_int(4);
        button.set_enabled(false);
        assert_eq!(button.label_tooltip.as_deref(), Some("bin images"));
        assert_eq!(button.sp_binning_xy.tooltip.as_deref(), Some("bin images"));
        assert_eq!(
            button.button.multi_line_button.button.tooltip.as_deref(),
            Some("open images")
        );
        assert_eq!(
            button.button.multi_line_button.button.action_listener_count,
            1
        );
        assert_eq!(button.get_action_command(), Some("Open in 3dmod"));
        assert_eq!(button.get_binning_in_x_and_y(), 4);
        assert!(!button.sp_binning_xy.is_enabled());
        assert!(!button.label_enabled);
        assert!(!button.button.multi_line_button.is_enabled());
    }
}
