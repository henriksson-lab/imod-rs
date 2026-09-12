//! `IMOD/Etomo/src/etomo/ui/swing/FileContainer.java`.
#![allow(dead_code)]

/// Java package-private `FileContainer`.
pub trait FileContainer {
    /// Java `fixIncorrectPaths(boolean)`.
    fn fix_incorrect_paths(&mut self, choose_path_every_row: bool);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Container(Option<bool>);

    impl FileContainer for Container {
        fn fix_incorrect_paths(&mut self, choose_path_every_row: bool) {
            self.0 = Some(choose_path_every_row);
        }
    }

    #[test]
    fn interface_preserves_boolean_argument() {
        let mut container = Container::default();
        container.fix_incorrect_paths(true);
        assert_eq!(container.0, Some(true));
    }
}
