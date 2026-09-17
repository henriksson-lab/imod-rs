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

/// Native coordinator state for `TomogramGenerationExpert`.  Process launch,
/// comscript I/O, and concrete dialog widgets are supplied by the application
/// layer; this unit owns the source ordering and lifecycle decisions.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TomogramGenerationExpertCoordinator {
    pub dialog_open: bool,
    pub sirt_checkpointed: bool,
    pub tilt_state_set: bool,
    pub parameter_transfer_count: u32,
    pub sirt_succeeded: bool,
}

impl TomogramGenerationExpertCoordinator {
    #[allow(non_snake_case)]
    pub fn openDialog(&mut self) {
        self.dialog_open = true;
    }
    #[allow(non_snake_case)]
    pub fn msgSirtsetupSucceeded(&mut self) {
        self.sirtCheckpoint();
    }
    #[allow(non_snake_case)]
    pub fn sirtCheckpoint(&mut self) {
        if self.dialog_open {
            self.sirt_checkpointed = true;
        }
    }
    #[allow(non_snake_case)]
    pub fn msgSirtSucceeded(&mut self) {
        self.sirt_succeeded = true;
    }
    #[allow(non_snake_case)]
    pub fn startNextProcess(&mut self, process: &str) -> bool {
        match process {
            "processchunks" | "sirtDone" => {
                if process == "sirtDone" {
                    self.msgSirtSucceeded();
                }
                true
            }
            _ => false,
        }
    }
    #[allow(non_snake_case)]
    pub fn reconnectTilt(&self) -> bool {
        self.dialog_open
    }
    #[allow(non_snake_case)]
    pub fn setTiltState(&mut self) {
        if self.dialog_open {
            self.tilt_state_set = true;
        }
    }
    #[allow(non_snake_case)]
    pub fn setParameters(&mut self) {
        self.parameter_transfer_count += 1;
    }
    #[allow(non_snake_case)]
    pub fn getParameters(&mut self) {
        self.parameter_transfer_count += 1;
    }
    /// The concrete tilt widget belongs to the native dialog adapter.
    #[allow(non_snake_case)]
    pub fn getTiltDisplay(&self) -> bool {
        self.dialog_open
    }
    #[allow(non_snake_case)]
    pub fn doneDialog(&mut self) {
        self.dialog_open = false;
    }
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
