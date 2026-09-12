//! `IMOD/Etomo/src/etomo/ui/swing/Constants.java`.
//!
//! This package-private Java source unit has no GUI behavior.  It centralizes
//! the guide-document anchor shared by the Swing source units; rendering and
//! guide launching remain at their respective native-GUI boundaries.
#![allow(dead_code)]

/// Java public static final `Constants.rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java package-private static final `Constants.TOP_ANCHOR`.
pub(crate) const TOP_ANCHOR: &str = "#TOP";

#[cfg(test)]
mod tests {
    use super::{RCSID, TOP_ANCHOR};

    #[test]
    fn constants_match_the_java_source() {
        assert_eq!(RCSID, "$Id:$");
        assert_eq!(TOP_ANCHOR, "#TOP");
    }
}
