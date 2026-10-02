//! `IMOD/Etomo/src/etomo/ui/swing/TabbedPane.java`.
//!
//! Java `final class TabbedPane extends JTabbedPane`: names itself (uitest `tb.`)
//! from its first tab's title.  The `JTabbedPane` is [`TabbedPane::get_component`];
//! other inherited members are called on it.

use std::rc::Rc;

use super::spaced_panel::SpacedPanel;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::util::utilities;

/// Java `TabbedPane`.
pub struct TabbedPane {
    /// The `JTabbedPane` this class extends.
    component: Rc<JComponent>,
}

impl TabbedPane {
    /// Java implicit constructor.
    pub fn new() -> Rc<TabbedPane> {
        Rc::new(TabbedPane {
            component: JComponent::new_tabbed_pane(),
        })
    }

    /// The `JTabbedPane`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }

    /// Java `addTab(String, Component)` (override).
    pub fn add_tab_string_component(&self, title: &str, component: &Rc<JComponent>) {
        self.component.add_tab(title, component);
        if self.component.get_tab_count() == 1 {
            self.set_name(Some(title));
        }
    }

    /// Java `addTab(String, SpacedPanel)`.
    pub fn add_tab_string_spaced_panel(&self, title: &str, spaced_panel: &SpacedPanel) {
        self.component.add_tab(title, &spaced_panel.get_container());
        if self.component.get_tab_count() == 1 {
            self.set_name(Some(title));
        }
    }

    /// Java `setTitleAt(int, String)` (override).
    pub fn set_title_at(&self, index: usize, title: &str) {
        self.component.set_title_at(index, title);
        if index == 0 {
            self.set_name(Some(title));
        }
    }

    /// Java `setName(String)` (override).
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = UITestFieldType::TAB;
        let name = utilities::convert_label_to_name(text, field_type.is_unlimited_segments());
        self.component.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.component.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }
}
