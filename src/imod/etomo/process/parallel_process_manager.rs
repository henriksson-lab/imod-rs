//! `IMOD/Etomo/src/etomo/process/ParallelProcessManager.java`.
//!
//! This coordinator is deliberately kept as a separate source unit from
//! `ParallelManager`.  The command parameter and process-runtime classes named by
//! the Java signatures have not been translated yet, so their Rust argument slots
//! remain explicit null-equivalent boundaries.  In particular, this code does not
//! claim that a background process was started before `BackgroundProcess.java` and
//! the applicable command-detail classes are available.
#![allow(dead_code)]

use std::convert::Infallible;

use crate::imod::etomo::parallel_manager::ParallelManager;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;

/// Java final `ParallelProcessManager extends BaseProcessManager`.
pub struct ParallelProcessManager {
    /// Java superclass state.
    base: BaseProcessManager,
    /// Java final `manager`.
    manager: &'static ParallelManager,
}

impl ParallelProcessManager {
    /// Java `ParallelProcessManager(ParallelManager)`.
    pub fn new(manager: &'static ParallelManager) -> Self {
        Self {
            // `BaseProcessManager` still has an unported `BaseManager` reference
            // representation.  The concrete manager is retained in this unit's own
            // source field, exactly as Java does.
            base: BaseProcessManager::new(None),
            manager,
        }
    }

    /// Java `trimVolume(TrimvolParam, ProcessSeries)`.
    pub fn trim_volume(
        &self,
        trimvol_param: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (trimvol_param, process_series);
        // TODO(unit): TrimvolParam.java, ProcessSeries.java's concrete manager
        // reference, BackgroundProcess.java, and BaseProcessManager's typed
        // startBackgroundProcess overload.  The Java body starts TRIMVOL on ONLY
        // then returns BackgroundProcess.getName().
        None
    }

    /// Java `anisotropicDiffusion(AnisotropicDiffusionParam, ProcessSeries)`.
    pub fn anisotropic_diffusion(
        &self,
        param: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, process_series);
        // TODO(unit): AnisotropicDiffusionParam.java, BackgroundProcess.java, and
        // BaseProcessManager's typed startBackgroundProcess overload.  The Java body
        // starts ANISOTROPIC_DIFFUSION on ONLY then returns getName().
        None
    }

    /// Java `chunksetup(ChunksetupParam, ProcessSeries)`.
    pub fn chunksetup(
        &self,
        param: Option<Infallible>,
        process_series: Option<Infallible>,
    ) -> Option<String> {
        let _ = (param, process_series);
        // TODO(unit): ChunksetupParam.java, BackgroundProcess.java, and
        // BaseProcessManager's typed startBackgroundProcess overload.  The Java body
        // starts CHUNKSETUP on ONLY then returns getName().
        None
    }

    /// Java override `postProcess(BackgroundProcess)`.
    pub fn post_process_background(&self, process: Option<Infallible>) {
        self.base.post_process_background(None);
        let _ = process;
        // TODO(unit): BackgroundProcess.java, CommandDetails.java,
        // AnisotropicDiffusionParam.java, ChunksetupParam.java, and
        // ParallelState.java.  The source first returns for null CommandDetails;
        // ANISOTROPIC_DIFFUSION copies K_VALUE and ITERATION_LIST to state; CHUNKSETUP
        // supplies stdout and ONE_LINE_COMMAND_PROGRAM to the manager.
    }

    /// Java package-private `getManager()`.
    pub fn get_manager(&self) -> &'static ParallelManager {
        self.manager
    }

    /// Java override `postProcess(DetachedProcess)`.
    pub fn post_process_detached(&self, process: Option<Infallible>) {
        // Source order matters: BaseProcessManager saves processchunks resume data
        // before this manager examines the detached command.
        self.base.post_process_detached(None);
        let _ = process;
        // TODO(unit): DetachedProcess.java, Command.java, CommandDetails.java,
        // AnisotropicDiffusionParam.java, and ParallelState.java.  For a PROCESSCHUNKS
        // command whose anisotropic-diffusion subcommand has VARYING_K mode, Java
        // copies K_VALUE_LIST and ITERATION into ParallelState.
    }
}

#[cfg(test)]
mod tests {
    use super::ParallelProcessManager;
    use crate::imod::etomo::parallel_manager::ParallelManager;

    #[test]
    fn retains_the_parallel_manager_reference() {
        let manager = ParallelManager::new();
        let process_manager = ParallelProcessManager::new(manager);
        assert!(std::ptr::eq(process_manager.get_manager(), manager));
        assert!(std::ptr::eq(
            manager.parallel_process_manager().get_manager(),
            manager
        ));
    }

    #[test]
    fn unavailable_process_types_do_not_report_started_processes() {
        let process_manager = ParallelProcessManager::new(ParallelManager::new());
        assert_eq!(process_manager.trim_volume(None, None), None);
        assert_eq!(process_manager.anisotropic_diffusion(None, None), None);
        assert_eq!(process_manager.chunksetup(None, None), None);
    }
}
