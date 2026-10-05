//! `IMOD/Etomo/src/etomo/type/FrameType.java`.

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class FrameType`: two identity-compared instances.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameType {
    /// Java `FrameType.Main`.
    Main,
    /// Java `FrameType.Sub`.
    Sub,
}

impl std::fmt::Display for FrameType {
    /// Java `toString()`.  "Unknown" is unreachable: only the two instances exist.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameType::Main => f.write_str("Main"),
            FrameType::Sub => f.write_str("Sub"),
        }
    }
}
