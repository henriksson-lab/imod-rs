//! `IMOD/Etomo/src/etomo/ui/swing/Highlightable.java`.
//!
//! An event dispatch thread object (a table row or a table) that responds to a
//! `HighlighterButton`; every method takes `&self` (the objects are `Rc`-shared Swing
//! objects with interior mutability).

/// Java package-private `interface Highlightable`.
pub trait Highlightable {
    /// Java `highlight(boolean)`.
    fn highlight(&self, highlight: bool);
}
