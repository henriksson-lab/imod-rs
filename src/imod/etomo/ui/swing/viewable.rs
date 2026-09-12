//! `IMOD/Etomo/src/etomo/ui/swing/Viewable.java`.

/// Java package-private `Viewable`.
///
/// `JComponent[]` is represented by an associated native presentation type;
/// the table retains ownership of the focusable components.
pub trait Viewable {
    type FocusableParent;
    /// Java `msgViewportPaged()`.
    fn msg_viewport_paged(&mut self);
    /// Java `size()`.
    fn size(&self) -> usize;
    /// Java `getFocusableParents()`.
    fn get_focusable_parents(&self) -> Vec<Self::FocusableParent>;
}

#[cfg(test)]
mod tests {
    use super::Viewable;
    struct Table {
        paged: bool,
    }
    impl Viewable for Table {
        type FocusableParent = usize;
        fn msg_viewport_paged(&mut self) {
            self.paged = true
        }
        fn size(&self) -> usize {
            4
        }
        fn get_focusable_parents(&self) -> Vec<usize> {
            vec![1, 2]
        }
    }
    #[test]
    fn table_contract_preserves_paging_size_and_focus_parents() {
        let mut table = Table { paged: false };
        table.msg_viewport_paged();
        assert!(table.paged);
        assert_eq!(table.size(), 4);
        assert_eq!(table.get_focusable_parents(), vec![1, 2]);
    }
}
