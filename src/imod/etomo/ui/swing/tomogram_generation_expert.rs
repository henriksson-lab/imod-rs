//! `IMOD/Etomo/src/etomo/ui/swing/TomogramGenerationExpert.java`.
//!
//! `TomogramGenerationExpert` is a concrete Java coordinator.  Its dialog
//! boundary is a trait because Java passes the expert into
//! `TomogramGenerationDialog`, which invokes this source-visible method.

/// Java `TomogramGenerationExpert.doneDialog()`.
///
/// This is the canonical expert-to-dialog interface and retains Java's
/// zero-argument, void signature. Process/comscript collaborators are owned
/// by the concrete expert coordinator rather than invented here.
pub trait TomogramGenerationExpert {
    fn done_dialog(&mut self);
}

#[cfg(test)]
mod tests {
    use super::TomogramGenerationExpert;
    struct Expert(bool);
    impl TomogramGenerationExpert for Expert {
        fn done_dialog(&mut self) {
            self.0 = true;
        }
    }
    #[test]
    fn dialog_facing_signature_is_void_and_zero_argument() {
        let mut expert = Expert(false);
        expert.done_dialog();
        assert!(expert.0);
    }
}
