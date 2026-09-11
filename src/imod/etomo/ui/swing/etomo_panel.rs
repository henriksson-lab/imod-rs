//! `IMOD/Etomo/src/etomo/ui/swing/EtomoPanel.java`.
//!
//! Swing owns the actual `JPanel`, `TitledBorder`, and `PanelHeader` widgets.
//! Their source-observable title, child insertion, and uitest naming behavior
//! is retained here for the optional native GUI harness.
#![allow(dead_code)]

use super::abstract_frame::ComponentState;
use super::tool_panel::ToolPanel;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

/// Rust state at the direct `javax.swing.border.TitledBorder` boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TitledBorder {
    pub title: String,
}

/// Direct source dependency used by `EtomoPanel.add(PanelHeader)`.
///
/// The complete `PanelHeader.java` unit owns expander controls and is translated
/// separately.  These are exactly the two values this source unit reads.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PanelHeader {
    pub container: ComponentState,
    pub title: String,
}

impl PanelHeader {
    /// Java `getContainer()`.
    pub fn get_container(&self) -> &ComponentState {
        &self.container
    }

    /// Java `getTitle()`.
    pub fn get_title(&self) -> &str {
        &self.title
    }
}

/// Java package-private `EtomoPanel` fields and methods.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EtomoPanel {
    pub component: ComponentState,
    pub name: Option<String>,
    pub border: Option<TitledBorder>,
    pub children: Vec<ComponentState>,
}

impl EtomoPanel {
    /// Java `getUIComponent()`.
    pub fn get_ui_component(&self) -> &Self {
        self
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> &ComponentState {
        &self.component
    }

    /// Java `setBorder(TitledBorder)`.
    pub fn set_border(&mut self, border: TitledBorder) {
        self.border = Some(border.clone());
        self.set_name(&border.title);
    }

    /// Java overloaded `add(PanelHeader)`.  Rust's overload-free spelling
    /// identifies the source argument type while preserving its exact order.
    pub fn add_panel_header(&mut self, panel_header: &PanelHeader) {
        self.children.push(panel_header.get_container().clone());
        self.set_name(panel_header.get_title());
    }

    /// Java `setName(String)`.
    pub fn set_name(&mut self, text: &str) {
        let name = utilities::convert_label_to_name(Some(text), true).unwrap_or_default();
        self.name = Some(format!("pnl{SEPARATOR_CHAR}{name}"));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {DEFAULT_DELIMITER} ",
                self.name.as_deref().unwrap_or_default()
            );
        }
    }
}

impl ToolPanel for EtomoPanel {
    fn get_component(&self) -> &ComponentState {
        &self.component
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titled_border_assigns_source_panel_uitest_name() {
        let mut panel = EtomoPanel::default();
        panel.set_border(TitledBorder {
            title: "My panel: 2".into(),
        });
        assert_eq!(panel.name.as_deref(), Some("pnl.my-panel"));
        assert_eq!(
            panel.border.as_ref().map(|border| border.title.as_str()),
            Some("My panel: 2")
        );
    }

    #[test]
    fn panel_header_is_added_before_its_title_names_the_panel() {
        let mut panel = EtomoPanel::default();
        let header = PanelHeader {
            container: ComponentState {
                height: 17,
                ..Default::default()
            },
            title: "Header title".into(),
        };
        panel.add_panel_header(&header);
        assert_eq!(panel.children, vec![header.container]);
        assert_eq!(panel.name.as_deref(), Some("pnl.header-title"));
    }
}
