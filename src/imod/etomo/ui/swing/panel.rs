//! `IMOD/Etomo/src/etomo/ui/swing/Panel.java`.
//!
//! Java `final class Panel extends JPanel`: a panel whose maximum size is scaled by the
//! font size.  The `JPanel` is [`Panel::get_component`].

use std::rc::Rc;

use super::ui_utilities;
use crate::imod::etomo::jdk::{Dimension, JComponent};

/// Java `Panel`.
pub struct Panel {
    /// The `JPanel` this class extends.
    component: Rc<JComponent>,
}

impl Panel {
    /// Java default constructor.
    pub fn new() -> Rc<Panel> {
        Rc::new(Panel {
            component: JComponent::new_panel(),
        })
    }

    /// The `JPanel` this class extends.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.component.clone()
    }

    /// Java `setMaximumSize(Dimension)` (override).  Java scales the caller's
    /// `Dimension` in place, so the parameter is `&mut`.
    pub fn set_maximum_size(&self, maximum_size: Option<&mut Dimension>) {
        if let Some(maximum_size) = maximum_size {
            maximum_size.width = ui_utilities::scale_by_font_size_int(maximum_size.width);
            maximum_size.height = ui_utilities::scale_by_font_size_int(maximum_size.height);
        }
        // Swing layout: super.setMaximumSize(maximumSize).
    }
}
