//! `IMOD/Etomo/src/etomo/ui/swing/TextComponentAppearanceExtension.java`.
#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};

use super::appearance_extension::{AppearanceExtension, ComponentBoundary};

/// Java `JTextComponent` state at the native presentation boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextComponentBoundary {
    pub component: Rc<RefCell<ComponentBoundary>>,
    pub editable: bool,
}

impl TextComponentBoundary {
    /// Java `JTextComponent.isEditable()`.
    pub fn is_editable(&self) -> bool {
        self.editable
    }
    /// Java `JTextComponent.setEditable(boolean)`.
    pub fn set_editable(&mut self, editable: bool) {
        self.editable = editable;
    }
}

/// Java package-private `TextComponentAppearanceExtension`.
pub struct TextComponentAppearanceExtension {
    pub appearance_extension: AppearanceExtension,
    pub text_component: Rc<RefCell<TextComponentBoundary>>,
}

impl TextComponentAppearanceExtension {
    /// Java `TextComponentAppearanceExtension(JTextComponent, boolean, boolean)`.
    pub fn new(
        text_component: Rc<RefCell<TextComponentBoundary>>,
        enabled_field: bool,
        editable_component: bool,
    ) -> Self {
        let editable = text_component.borrow().is_editable();
        let component = text_component.borrow().component.clone();
        Self {
            appearance_extension: AppearanceExtension::new_with(
                component,
                enabled_field,
                editable_component,
                editable,
            ),
            text_component,
        }
    }

    /// Java overridden `setComponentEditable(boolean)`.
    pub fn set_component_editable(&mut self, editable: bool) {
        if !editable || self.appearance_extension.editable_component {
            self.text_component.borrow_mut().set_editable(editable);
        }
    }

    /// Java overridden `isNativeSetEditable()`.
    pub fn is_native_set_editable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ineditable_component_never_becomes_editable() {
        let text_component = Rc::new(RefCell::new(TextComponentBoundary {
            component: Rc::new(RefCell::new(ComponentBoundary::default())),
            editable: false,
        }));
        let mut extension =
            TextComponentAppearanceExtension::new(text_component.clone(), true, false);
        extension.set_component_editable(true);
        assert!(!text_component.borrow().editable);
        assert!(extension.is_native_set_editable());
    }
}
