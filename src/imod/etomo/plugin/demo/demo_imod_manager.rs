//! `IMOD/Etomo/src/etomo/plugin/demo/DemoImodManager.java`.
//!
//! Allows this plugin to run 3dmod.  `DemoImodManager extends BaseImodManager`: the
//! superclass is embedded as `base` (reached through `Deref`), and the one override,
//! `newImodState`, is the [`BaseImodManagerHooks`] implementation handed to it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::process::base_imod_manager::{
    BaseImodManager, BaseImodManagerHooks, ImodManagerException,
};
use crate::imod::etomo::process::imod_state::ImodState;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `public static final String DEMO_KEY = new String("Demo output files")`.
pub const DEMO_KEY: &str = "Demo output files";

/// Java `final class DemoImodManager extends BaseImodManager`.
pub struct DemoImodManager {
    /// The `BaseImodManager` superclass.
    base: BaseImodManager,
}

impl std::ops::Deref for DemoImodManager {
    type Target = BaseImodManager;
    fn deref(&self) -> &BaseImodManager {
        &self.base
    }
}

/// The subclass's override of `newImodState`.
struct DemoImodManagerHooks;

impl DemoImodManager {
    /// Java package-private `DemoImodManager(BaseManager)`.  The manager lives as long
    /// as the dataset (the base class's request handler needs `&'static`).
    pub fn new(manager: &'static dyn BaseManager) -> &'static DemoImodManager {
        let instance: &'static DemoImodManager = Box::leak(Box::new(DemoImodManager {
            base: BaseImodManager::new(manager, Arc::new(DemoImodManagerHooks)),
        }));
        instance.base.start_request_handler();
        instance
    }
}

impl BaseImodManagerHooks for DemoImodManagerHooks {
    /// Java protected override `newImodState(String, String, AxisID, String, File,
    /// String[], String, File[])`.
    ///
    /// Java returns null for any other key or a null axis, and its callers then throw
    /// `NullPointerException` (`newVector(null)`, `imodState.setSwapYZ`).  Fixed in
    /// translation: the error abandons the action as `ImodManager`'s unknown-key error
    /// does.
    fn new_imod_state(
        &self,
        base: &BaseImodManager,
        key: &str,
        _file_extension: Option<&str>,
        axis_id: Option<AxisID>,
        _dataset_name: Option<&str>,
        file: Option<&Path>,
        _file_name_array: Option<&[String]>,
        _subdir_name: Option<&str>,
        _file_list: Option<&[PathBuf]>,
    ) -> Result<ImodState, ImodManagerException> {
        if key == DEMO_KEY
            && let Some(axis_id) = axis_id
        {
            return Ok(DemoImodManagerHooks::new_demo(base, file, axis_id));
        }
        Err(ImodManagerException::Runtime(format!(
            "{} cannot be created with axisID={}",
            key,
            match axis_id {
                Some(axis_id) => axis_id.get_extension(),
                None => "null".to_string(),
            }
        )))
    }
}

impl DemoImodManagerHooks {
    /// Java private `newDemo(File, AxisID)`.  The Java ignores `file`.
    fn new_demo(base: &BaseImodManager, _file: Option<&Path>, axis_id: AxisID) -> ImodState {
        ImodState::new_base_manager_axis_id(base.manager, Some(axis_id))
    }
}
