//! `IMOD/Etomo/src/etomo/logic/UserEnv.java`.
//!
//! The Java methods take an `AxisID` that may be null; callers here pass either an
//! `AxisID` or an `Option<AxisID>`, so the parameter is `impl Into<Option<AxisID>>`.
//! `Network`'s translation takes a non-null `AxisID`; a null one is passed on as
//! `AxisID::Only` (the axis the source's own callers pass).  Deviation only for a null
//! axis, where the Java passes null down to `CpuAdoc`/`Node`.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java static `isParallelProcessing(BaseManager, AxisID, String)`.
pub fn is_parallel_processing(
    _manager: &'static dyn BaseManager,
    _axis_id: impl Into<Option<AxisID>>,
    _property_user_dir: Option<&str>,
) -> bool {
    etomo_director::ARGUMENTS.lock().unwrap().is_cpus()
        || !etomo_director::INSTANCE.with_user_configuration(|c| c.get_no_parallel_processing())
}

/// Java static `isGpuProcessing(BaseManager, AxisID, String)`.
pub fn is_gpu_processing(
    manager: &'static dyn BaseManager,
    axis_id: impl Into<Option<AxisID>>,
    property_user_dir: Option<&str>,
) -> bool {
    let axis_id = axis_id.into();
    etomo_director::ARGUMENTS.lock().unwrap().is_gpus()
        || (is_gpu_processing_enabled(manager, axis_id, property_user_dir)
            && etomo_director::INSTANCE.with_user_configuration(|c| c.get_gpu_processing_default()))
}

/// Java static `isGpuProcessingEnabled(BaseManager, AxisID, String)`.
pub fn is_gpu_processing_enabled(
    manager: &'static dyn BaseManager,
    axis_id: impl Into<Option<AxisID>>,
    property_user_dir: Option<&str>,
) -> bool {
    let axis_id = axis_id.into().unwrap_or(AxisID::Only);
    Network::is_non_local_host_gpu_processing_enabled(manager, axis_id, property_user_dir)
        || Network::is_local_host_gpu_processing_enabled(manager, axis_id, property_user_dir)
}
