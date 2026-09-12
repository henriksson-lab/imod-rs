//! `IMOD/Etomo/src/etomo/ui/swing/SwingComponent.java`.
#![allow(dead_code)]

use super::abstract_frame::ComponentState;

/// Java public `SwingComponent` interface.
///
/// `ComponentState` is the direct native-`java.awt.Component` boundary used
/// throughout the translated Swing source.
pub trait SwingComponent {
    /// Java `getComponent()`.
    fn get_component(&self) -> &ComponentState;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ComponentHolder(ComponentState);

    impl SwingComponent for ComponentHolder {
        fn get_component(&self) -> &ComponentState {
            &self.0
        }
    }

    #[test]
    fn component_contract_returns_the_component_boundary() {
        let component = ComponentHolder(ComponentState {
            height: 19,
            ..Default::default()
        });
        assert_eq!(component.get_component().height, 19);
    }
}
