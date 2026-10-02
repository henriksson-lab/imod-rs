//! `IMOD/Etomo/src/etomo/storage/DirectiveDescrEtomoColumn.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a `Copy` struct with associated
//! constants; identity comparison is field equality (the tags are distinct).

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java final `DirectiveDescrEtomoColumn`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectiveDescrEtomoColumn {
    /// Java private final field `tag`.
    tag: &'static str,
    /// Java private final field `recommended`.
    recommended: bool,
}

impl DirectiveDescrEtomoColumn {
    /// Java `NE`: no effect on etomo and not saved (template).
    pub const NE: DirectiveDescrEtomoColumn = DirectiveDescrEtomoColumn::new("NE", false);
    /// Java package-private `NES`: no effect on etomo, optionally saved (template).
    pub(crate) const NES: DirectiveDescrEtomoColumn = DirectiveDescrEtomoColumn::new("NES", true);
    /// Java `SD`: saved by default (template).
    pub const SD: DirectiveDescrEtomoColumn = DirectiveDescrEtomoColumn::new("SD", true);
    /// Java `SO`: save optional (template).
    pub const SO: DirectiveDescrEtomoColumn = DirectiveDescrEtomoColumn::new("SO", true);

    /// Java private `DirectiveDescrEtomoColumn(String, boolean)`.
    const fn new(tag: &'static str, recommended: bool) -> DirectiveDescrEtomoColumn {
        DirectiveDescrEtomoColumn { tag, recommended }
    }

    /// Java package-private static `getInstance(String)`.
    pub(crate) fn get_instance(input: &str) -> Option<DirectiveDescrEtomoColumn> {
        if input == Self::NE.tag {
            return Some(Self::NE);
        }
        if input == Self::NES.tag {
            return Some(Self::NES);
        }
        if input == Self::SD.tag {
            return Some(Self::SD);
        }
        if input == Self::SO.tag {
            return Some(Self::SO);
        }
        None
    }

    /// Java `isRecommended()`.
    pub fn is_recommended(&self) -> bool {
        self.recommended
    }
}

/// Java `toString()`.
impl std::fmt::Display for DirectiveDescrEtomoColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tag)
    }
}
