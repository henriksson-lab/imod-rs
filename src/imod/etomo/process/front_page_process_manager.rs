//! `IMOD/Etomo/src/etomo/process/FrontPageProcessManager.java`.
//!
//! The front page's process manager.  The Java subclass adds nothing to
//! `BaseProcessManager` (it overrides no hook), so no
//! `BaseProcessManagerHooks` are installed and the base class's own bodies
//! run.

use super::base_process_manager::BaseProcessManager;
use crate::imod::etomo::base_manager::BaseManager;

/// Java public final class `FrontPageProcessManager extends
/// BaseProcessManager`.
pub struct FrontPageProcessManager {
    /// The Java superclass part.
    pub base: BaseProcessManager,
}

impl std::ops::Deref for FrontPageProcessManager {
    type Target = BaseProcessManager;
    fn deref(&self) -> &BaseProcessManager {
        &self.base
    }
}

impl FrontPageProcessManager {
    /// Java `FrontPageProcessManager(BaseManager)`.  The manager keeps it for
    /// the program's lifetime, as the Java does, so it is leaked here like
    /// every other process manager (processes hold `&'static
    /// BaseProcessManager`); the owning `FrontPageManager` roots it.
    pub fn new(manager: &'static dyn BaseManager) -> &'static FrontPageProcessManager {
        Box::leak(Box::new(FrontPageProcessManager {
            base: BaseProcessManager::new(manager),
        }))
    }
}
