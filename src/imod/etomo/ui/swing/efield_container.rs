//! `IMOD/Etomo/src/etomo/ui/swing/EfieldContainer.java`.
//!
//! Swing `Component`, `Container`, `JPanel`, `BoxLayout`, and `GridBagLayout`
//! calls are retained as explicit native-GUI boundary state.  The conditional
//! panel construction and the `getComponent` lock are source behavior.
#![allow(dead_code)]

/// Java `java.awt.Insets` values used by `EfieldContainer`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EfieldInsets {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
}

/// Opaque native Swing `Component` boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EfieldComponentBoundary {
    /// Java `Component.toString()` boundary value.
    pub to_string_value: String,
}

/// Opaque native Swing `Container` boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EfieldContainerBoundary {
    /// Java `Container.toString()` boundary value.
    pub to_string_value: String,
}

/// A child inserted with Java `Container.add(Component)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EfieldContainerChild {
    Component(EfieldComponentBoundary),
    Container(EfieldContainerBoundary),
}

/// The subset of a Java `GridBagConstraints` observed by this source unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EfieldGridBagConstraintsBoundary {
    pub insets: EfieldInsets,
    pub fill_both: bool,
    pub weight_x: i32,
    pub weight_y: i32,
    pub grid_height: i32,
    pub grid_width: i32,
}

/// One Java `GridBagLayout.setConstraints` call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EfieldGridBagSetConstraintsBoundary {
    pub child: EfieldContainerChild,
    pub constraints: EfieldGridBagConstraintsBoundary,
}

/// Native `JPanel` state constructed by Java private `createPanel()`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EfieldPanelBoundary {
    pub layout: &'static str,
    pub children: Vec<EfieldContainerChild>,
    pub grid_bag_set_constraints: Vec<EfieldGridBagSetConstraintsBoundary>,
}

/// Java `root`, which is either the original `field` container or a `JPanel`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EfieldContainerRoot {
    Field(EfieldContainerBoundary),
    Panel(EfieldPanelBoundary),
}

/// Java package-private final `EfieldContainer`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EfieldContainer {
    pub grid_bag_layout_created: bool,
    pub constraints: Option<EfieldGridBagConstraintsBoundary>,
    pub use_grid_bag: bool,
    pub root: Option<EfieldContainerRoot>,
    pub lock: bool,
}

impl EfieldContainer {
    pub const FIELD_INSETS: EfieldInsets = EfieldInsets {
        top: 0,
        left: 0,
        bottom: 0,
        right: -1,
    };
    pub const COMPONENT_INSETS: EfieldInsets = EfieldInsets {
        top: 0,
        left: 0,
        bottom: 0,
        right: -1,
    };

    /// Java `EfieldContainer(boolean, boolean, Component, Container, Component)`.
    pub fn new(
        modifiable: bool,
        use_grid_bag: bool,
        label: Option<EfieldComponentBoundary>,
        field: EfieldContainerBoundary,
        field_control: Option<EfieldComponentBoundary>,
    ) -> Self {
        let mut value = Self {
            grid_bag_layout_created: use_grid_bag,
            constraints: use_grid_bag.then_some(EfieldGridBagConstraintsBoundary {
                insets: EfieldInsets {
                    top: 0,
                    left: 0,
                    bottom: 0,
                    right: 0,
                },
                fill_both: false,
                weight_x: 0,
                weight_y: 0,
                grid_height: 1,
                grid_width: 1,
            }),
            use_grid_bag,
            root: None,
            lock: false,
        };
        if !modifiable && label.is_none() && field_control.is_none() {
            value.root = Some(EfieldContainerRoot::Field(field));
        } else {
            value.create_panel();
            if let Some(label) = label {
                if use_grid_bag {
                    let constraints = value.constraints.clone().unwrap();
                    if let Some(EfieldContainerRoot::Panel(root)) = &mut value.root {
                        root.grid_bag_set_constraints
                            .push(EfieldGridBagSetConstraintsBoundary {
                                child: EfieldContainerChild::Component(label.clone()),
                                constraints,
                            });
                    }
                }
                if let Some(EfieldContainerRoot::Panel(root)) = &mut value.root {
                    root.children.push(EfieldContainerChild::Component(label));
                }
            }
            if use_grid_bag {
                value.constraints.as_mut().unwrap().insets = Self::FIELD_INSETS;
                let constraints = value.constraints.clone().unwrap();
                if let Some(EfieldContainerRoot::Panel(root)) = &mut value.root {
                    root.grid_bag_set_constraints
                        .push(EfieldGridBagSetConstraintsBoundary {
                            child: EfieldContainerChild::Container(field.clone()),
                            constraints,
                        });
                }
            }
            if let Some(EfieldContainerRoot::Panel(root)) = &mut value.root {
                root.children.push(EfieldContainerChild::Container(field));
            }
            if let Some(field_control) = field_control {
                if use_grid_bag {
                    let constraints = value.constraints.clone().unwrap();
                    if let Some(EfieldContainerRoot::Panel(root)) = &mut value.root {
                        root.grid_bag_set_constraints
                            .push(EfieldGridBagSetConstraintsBoundary {
                                child: EfieldContainerChild::Component(field_control.clone()),
                                constraints,
                            });
                    }
                }
                if let Some(EfieldContainerRoot::Panel(root)) = &mut value.root {
                    root.children
                        .push(EfieldContainerChild::Component(field_control));
                }
            }
        }
        value
    }

    /// Java `add(Component)`.
    pub fn add(&mut self, component: Option<EfieldComponentBoundary>) {
        let Some(component) = component else {
            return;
        };
        if self.lock {
            panic!(
                "ERROR:  Serious Software Error.  Attempting to add component {} to a non-existent container.  A container cannot be created for the {} field because the field has already been placed in a panel.",
                component.to_string_value,
                match self.root.as_ref().unwrap() {
                    EfieldContainerRoot::Field(value) => &value.to_string_value,
                    EfieldContainerRoot::Panel(_) => "JPanel",
                },
            );
        }
        if matches!(self.root, Some(EfieldContainerRoot::Field(_))) {
            let field = match self.root.take().unwrap() {
                EfieldContainerRoot::Field(field) => field,
                EfieldContainerRoot::Panel(_) => unreachable!(),
            };
            self.create_panel();
            if self.use_grid_bag {
                self.constraints.as_mut().unwrap().insets = Self::FIELD_INSETS;
                let constraints = self.constraints.clone().unwrap();
                if let Some(EfieldContainerRoot::Panel(root)) = &mut self.root {
                    root.grid_bag_set_constraints
                        .push(EfieldGridBagSetConstraintsBoundary {
                            child: EfieldContainerChild::Container(field.clone()),
                            constraints,
                        });
                }
            }
            if let Some(EfieldContainerRoot::Panel(root)) = &mut self.root {
                root.children.push(EfieldContainerChild::Container(field));
            }
            if self.use_grid_bag {
                self.constraints.as_mut().unwrap().insets = Self::COMPONENT_INSETS;
                let constraints = self.constraints.clone().unwrap();
                if let Some(EfieldContainerRoot::Panel(root)) = &mut self.root {
                    root.grid_bag_set_constraints
                        .push(EfieldGridBagSetConstraintsBoundary {
                            child: EfieldContainerChild::Component(component.clone()),
                            constraints,
                        });
                }
            }
        }
        if let Some(EfieldContainerRoot::Panel(root)) = &mut self.root {
            root.children
                .push(EfieldContainerChild::Component(component));
        }
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        self.root = Some(EfieldContainerRoot::Panel(EfieldPanelBoundary {
            layout: if self.use_grid_bag {
                "GridBagLayout"
            } else {
                "BoxLayout.X_AXIS"
            },
            children: Vec::new(),
            grid_bag_set_constraints: Vec::new(),
        }));
        if self.use_grid_bag {
            let constraints = self.constraints.as_mut().unwrap();
            constraints.fill_both = true;
            constraints.weight_x = 0;
            constraints.weight_y = 0;
            constraints.grid_height = 1;
            constraints.grid_width = 1;
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&mut self) -> &EfieldContainerRoot {
        if !self.lock && matches!(self.root, Some(EfieldContainerRoot::Field(_))) {
            self.lock = true;
        }
        self.root.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field() -> EfieldContainerBoundary {
        EfieldContainerBoundary {
            to_string_value: "field".into(),
        }
    }
    fn component(value: &str) -> EfieldComponentBoundary {
        EfieldComponentBoundary {
            to_string_value: value.into(),
        }
    }

    #[test]
    fn unmodified_unlabeled_field_is_returned_and_locked() {
        let mut value = EfieldContainer::new(false, false, None, field(), None);
        assert!(matches!(
            value.get_component(),
            EfieldContainerRoot::Field(_)
        ));
        assert!(value.lock);
    }

    #[test]
    #[should_panic(expected = "Attempting to add component control")]
    fn get_component_locks_a_raw_field_against_late_addition() {
        let mut value = EfieldContainer::new(false, false, None, field(), None);
        value.get_component();
        value.add(Some(component("control")));
    }

    #[test]
    fn late_addition_wraps_raw_field_in_grid_bag_panel() {
        let mut value = EfieldContainer::new(false, true, None, field(), None);
        value.add(Some(component("control")));
        let EfieldContainerRoot::Panel(root) = value.get_component() else {
            panic!("expected panel");
        };
        assert_eq!(root.layout, "GridBagLayout");
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.grid_bag_set_constraints.len(), 2);
        assert_eq!(
            root.grid_bag_set_constraints[1].constraints.insets,
            EfieldContainer::COMPONENT_INSETS
        );
    }

    #[test]
    fn constructor_preserves_label_field_and_control_order() {
        let mut value = EfieldContainer::new(
            true,
            false,
            Some(component("label")),
            field(),
            Some(component("control")),
        );
        let EfieldContainerRoot::Panel(root) = value.get_component() else {
            panic!("expected panel");
        };
        assert_eq!(root.layout, "BoxLayout.X_AXIS");
        assert_eq!(
            root.children,
            vec![
                EfieldContainerChild::Component(component("label")),
                EfieldContainerChild::Container(field()),
                EfieldContainerChild::Component(component("control")),
            ]
        );
    }
}
