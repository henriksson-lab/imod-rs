//! `IMOD/Etomo/src/etomo/ui/swing/GridBagExtension.java`.
//!
//! Remembers the panel a field was added to so that it can be removed again.  The
//! `GridBagLayout` and `GridBagConstraints` parameters are layout only and are not
//! modelled.

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::jdk::JComponent;

/// Java `GridBagExtension`.
pub struct GridBagExtension {
    /// Java `parent`.
    parent: RefCell<Option<Rc<JComponent>>>,
}

impl GridBagExtension {
    /// Java `GridBagExtension()`.
    pub fn new() -> Rc<GridBagExtension> {
        Rc::new(GridBagExtension {
            parent: RefCell::new(None),
        })
    }

    /// Java `add(Component, JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn add(&self, field: &Rc<JComponent>, panel: &Rc<JComponent>) {
        // Swing layout: layout.setConstraints(field, constraints).
        panel.add(field);
        *self.parent.borrow_mut() = Some(panel.clone());
    }

    /// Java `remove(Component)`.
    pub fn remove(&self, field: &Rc<JComponent>) {
        let parent = self.parent.borrow().clone();
        if let Some(parent) = parent {
            parent.remove(field);
            *self.parent.borrow_mut() = None;
        }
    }
}
