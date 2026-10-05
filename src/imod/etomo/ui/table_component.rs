//! `IMOD/Etomo/src/etomo/ui/TableComponent.java`.

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public interface TableComponent`.
pub trait TableComponent {
    /// Java `getPreferredWidth()`.
    fn get_preferred_width(&self) -> i32;
}
