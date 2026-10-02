//! `IMOD/Etomo/src/etomo/logic/CombineTool.java`.
//!
//! Description: Logic for tomogram combination
//!
//! Copyright: Copyright 2015 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::util::imodinfo::Imodinfo;

/// Java `CombineTool`, a class of static methods.
pub struct CombineTool;

impl CombineTool {
    /// Java static `isInvertYLimits(BaseManager)`.
    pub fn is_invert_y_limits(manager: &'static dyn BaseManager) -> bool {
        file_type::CLASS.orig_coms_dir.exists(Some(manager), None)
    }

    /// Java static `getInitialVolumeMatchingInitValue(ApplicationManager)`.
    pub fn get_initial_volume_matching_init_value(manager: &'static ApplicationManager) -> bool {
        let meta_data = manager.get_meta_data();
        meta_data.is_fiducialess_alignment(AxisID::First)
            || meta_data.is_fiducialess_alignment(AxisID::Second)
            || Imodinfo::new(&file_type::CLASS.fiducial_model)
                .is_patch_tracking(manager, AxisID::First)
    }
}
