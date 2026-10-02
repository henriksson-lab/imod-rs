//! `IMOD/Etomo/src/etomo/ui/swing/EtomoPanel.java`.
//!
//! Java `class EtomoPanel extends JPanel implements UIComponent, SwingComponent`: a
//! panel that names itself (uitest `pnl.` names) from its titled border or its panel
//! header.  The `JPanel` is [`EtomoPanel::get_component`]; inherited `JPanel` members
//! (`add(Component)`, `setLayout`, ...) are called on it.

use std::rc::Rc;

use super::panel_header::PanelHeader;
use super::swing_component::SwingComponent;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{JComponent, TitledBorder};
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `EtomoPanel`.
pub struct EtomoPanel {
    /// The `JPanel` this class extends.
    component: Rc<JComponent>,
}

impl EtomoPanel {
    /// Java default constructor.
    pub fn new() -> Rc<EtomoPanel> {
        Rc::new(EtomoPanel {
            component: JComponent::new_panel(),
        })
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }

    /// Java `setBorder(TitledBorder)`.
    pub fn set_border(&self, border: &TitledBorder) {
        // super.setBorder(border): the stand-in keeps a titled border's title.
        self.component.set_border_title(border.get_title().as_deref());
        self.set_name(border.get_title().as_deref());
    }

    /// Java `add(PanelHeader)`.
    pub fn add(&self, panel_header: &PanelHeader) {
        self.component.add(&panel_header.get_container());
        self.set_name(panel_header.get_title().as_deref());
    }

    /// Java `setName(String)` (override).
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = &ui_test_field_type::PANEL;
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

impl UIComponent for EtomoPanel {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }
}

impl SwingComponent for EtomoPanel {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }
}
