//! `IMOD/Etomo/src/etomo/ui/swing/GridBagExtension.java`.
#![allow(dead_code)]

/// Java final `GridBagExtension`; the component/panel/layout calls terminate at
/// the native Swing boundary while this source-owned parent relationship remains.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GridBagExtension {
    pub parent: Option<usize>,
    pub added: bool,
    pub removed: bool,
    pub constraints: Option<(i32, i32)>,
}

impl GridBagExtension {
    /// Java package-private `GridBagExtension()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Java `add(Component, JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn add(&mut self, parent: usize, constraints: (i32, i32)) {
        self.parent = Some(parent);
        self.added = true;
        self.removed = false;
        self.constraints = Some(constraints);
    }

    /// Java `remove(Component)`.
    pub fn remove(&mut self) {
        if self.parent.is_some() {
            self.removed = true;
            self.parent = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn remove_only_removes_a_previously_added_parent() {
        let mut extension = GridBagExtension::new();
        extension.remove();
        assert!(!extension.removed);
        extension.add(7, (3, 4));
        extension.remove();
        assert!(extension.removed);
        assert_eq!(extension.parent, None);
    }
}
