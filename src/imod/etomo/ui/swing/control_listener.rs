//! `IMOD/Etomo/src/etomo/ui/swing/ControlListener.java`.
//!
//! The Java source is a package-private callback interface.  Implementors
//! provide the callback at the GUI boundary; dispatch remains with the Swing
//! control which owns the listener list.
#![allow(dead_code)]

/// Java package-private `ControlListener`.
pub trait ControlListener {
    /// Java `controlEvent()`.
    fn control_event(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Listener {
        event_count: usize,
    }

    impl ControlListener for Listener {
        fn control_event(&mut self) {
            self.event_count += 1;
        }
    }

    #[test]
    fn control_event_is_implemented_by_the_listener() {
        let mut listener = Listener::default();
        listener.control_event();
        listener.control_event();
        assert_eq!(listener.event_count, 2);
    }
}
