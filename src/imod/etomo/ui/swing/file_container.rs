//! `IMOD/Etomo/src/etomo/ui/swing/FileContainer.java`.
//!
//! The implementor (`PeetDialog`) is an event dispatch thread object reached through
//! `Rc`, so the method takes `&self`.

/// Java package-private `interface FileContainer`.
pub trait FileContainer {
    /// Java `fixIncorrectPaths(boolean)`.
    fn fix_incorrect_paths(&self, choose_path_every_row: bool);
}
