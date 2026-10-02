//! `IMOD/Etomo/src/etomo/ui/swing/EfieldContainer.java`: holds a field and its
//! optional label and control components, creating a panel only when one is
//! needed.
//!
//! `final class EfieldContainer`.  The components are jdk stand-in
//! [`JComponent`]s; `root instanceof JPanel` is `root.kind() ==
//! ComponentKind::Panel`.  The grid-bag layout and its constraints are Swing
//! layout and are recorded as comments only.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::imod::etomo::jdk::{ComponentKind, JComponent};

// Java `FIELD_INSETS = new Insets(0, 0, 0, -1)` and
// `COMPONENT_INSETS = new Insets(0, 0, 0, -1)`: grid-bag constraint insets
// (Swing layout).

/// Java `EfieldContainer`.
pub struct EfieldContainer {
    // Java `gridBaglayout` and `constraints`: the GridBagLayout and its
    // GridBagConstraints when `useGridBag` (Swing layout, not modelled).
    use_grid_bag: bool,

    root: RefCell<Option<Rc<JComponent>>>,
    lock: Cell<bool>,
}

impl EfieldContainer {
    /// Java `EfieldContainer(boolean modifiable, boolean useGridBag, Component
    /// label, Container field, Component fieldControl)`.  If label and/or
    /// fieldControl are present create a JPanel and add to it in order: label,
    /// field, fieldControl.  If only field is present then no JPanel is created
    /// and the field is returned in getComponent.
    /// - `modifiable`: field components will be added after the field is placed
    ///   in a panel
    /// - `use_grid_bag`: format field with grid bag layout
    /// - `field`: required
    /// - `label`, `field_control`: optional
    pub fn new(
        modifiable: bool,
        use_grid_bag: bool,
        label: Option<&Rc<JComponent>>,
        field: &Rc<JComponent>,
        field_control: Option<&Rc<JComponent>>,
    ) -> EfieldContainer {
        let instance = EfieldContainer {
            use_grid_bag,
            root: RefCell::new(None),
            lock: Cell::new(false),
        };
        // Swing layout: when useGridBag, gridBaglayout = new GridBagLayout() and
        // constraints = new GridBagConstraints(); otherwise both null.
        if !modifiable && label.is_none() && field_control.is_none() {
            *instance.root.borrow_mut() = Some(field.clone());
        } else {
            instance.create_panel();
            let root = instance.root.borrow().clone().unwrap();
            // label
            if let Some(label) = label {
                // Swing layout (useGridBag): gridBaglayout.setConstraints(label,
                // constraints).
                root.add(label);
            }
            // field
            // Swing layout (useGridBag): constraints.insets = FIELD_INSETS;
            // gridBaglayout.setConstraints(field, constraints).
            root.add(field);
            // fieldControl
            if let Some(field_control) = field_control {
                // Swing layout (useGridBag): gridBaglayout.setConstraints(
                // fieldControl, constraints).
                root.add(field_control);
            }
        }
        instance
    }

    /// Java `add(Component)`.  Add a component.  Creates a JPanel to hold field and
    /// components if necessary.
    ///
    /// # Panics
    /// Java throws `IllegalStateException` (an unchecked "Serious Software Error")
    /// when the field has already been placed in a panel by `getComponent`.
    pub fn add(&self, component: Option<&Rc<JComponent>>) {
        let Some(component) = component else {
            return;
        };
        if self.lock.get() {
            // Java appends `component.toString()` and `root.toString()` (Swing's
            // class/bounds dump, not modelled); kind and name stand in.
            let root = self.root.borrow().clone();
            panic!(
                "ERROR:  Serious Software Error.  Attempting to add component {:?} {:?} to a \
                 non-existent container.  A container cannot be created for the {:?} {:?} field \
                 because the field has already been placed in a panel.",
                component.kind(),
                component.get_name(),
                root.as_ref().map(|root| root.kind()),
                root.as_ref().and_then(|root| root.get_name())
            );
        }
        let root_is_panel = self
            .root
            .borrow()
            .as_ref()
            .is_some_and(|root| root.kind() == ComponentKind::Panel);
        if !root_is_panel {
            // Did not need a JPanel until now. Set root to a new JPanel and add the
            // field to it.
            let field = self.root.borrow().clone();
            self.create_panel();
            // field
            // Swing layout (useGridBag): constraints.insets = FIELD_INSETS;
            // gridBaglayout.setConstraints(field, constraints).
            if let Some(field) = &field {
                self.root.borrow().clone().unwrap().add(field);
            }
            // component
            // Swing layout (useGridBag): constraints.insets = COMPONENT_INSETS;
            // gridBaglayout.setConstraints(component, constraints).
        }
        self.root.borrow().clone().unwrap().add(component);
    }

    /// Java `createPanel()`.
    fn create_panel(&self) {
        *self.root.borrow_mut() = Some(JComponent::new_panel());
        if self.use_grid_bag {
            // Swing layout: root.setLayout(gridBaglayout); constraints.fill = BOTH,
            // weightx = weighty = 0.0, gridheight = gridwidth = 1.
        } else {
            // Swing layout: root.setLayout(new BoxLayout(root, BoxLayout.X_AXIS)).
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        let root = self.root.borrow().clone().unwrap();
        if !self.lock.get() && root.kind() != ComponentKind::Panel {
            self.lock.set(true);
        }
        root
    }
}
