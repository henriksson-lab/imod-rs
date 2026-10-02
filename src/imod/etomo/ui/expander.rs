//! `IMOD/Etomo/src/etomo/ui/Expander.java`.

/// Java `Expander`.  Implementers are EDT objects (`Rc`, `&self` methods).
pub trait Expander {
    /// Java `isExpanded()`.
    fn is_expanded(&self) -> bool;
}
