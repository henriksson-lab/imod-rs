//! `IMOD/Etomo/src/etomo/type/ConstPeetMetaData.java`.
//!
//! The read-only face of `PeetMetaData`.  The `ConstEtomoNumber` getters return a
//! copy of the number (the shared meta data keeps its fields behind locks).

use super::axis_type::AxisType;
use super::const_etomo_number::ConstEtomoNumber;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public interface ConstPeetMetaData`.
pub trait ConstPeetMetaData {
    /// Java `getName()`.
    fn get_name(&self) -> Option<String>;

    /// Java `getInitMotlFile(int)`.
    fn get_init_motl_file(&self, key: i32) -> Option<String>;

    /// Java `getTiltRangeMultiAxesFile(int)`.
    fn get_tilt_range_multi_axes_file(&self, key: i32) -> Option<String>;

    /// Java `getTiltRangeMin(int)`.
    fn get_tilt_range_min(&self, key: i32) -> Option<String>;

    /// Java `getTiltRangeMax(int)`.
    fn get_tilt_range_max(&self, key: i32) -> Option<String>;

    /// Java `getAxisType()`.
    fn get_axis_type(&self) -> AxisType;

    /// Java `getReferenceFile()`.
    fn get_reference_file(&self) -> Option<String>;

    /// Java `getReferenceParticle()`.
    fn get_reference_particle(&self) -> ConstEtomoNumber;

    /// Java `getReferenceVolume()`.
    fn get_reference_volume(&self) -> ConstEtomoNumber;

    /// Java `getReferenceMultiparticleLevel()`.
    fn get_reference_multiparticle_level(&self) -> i32;

    /// Java `getEdgeShift()`.
    fn get_edge_shift(&self) -> ConstEtomoNumber;

    /// Java `isFlgWedgeWeight()`.
    fn is_flg_wedge_weight(&self) -> bool;

    /// Java `getMaskModelPtsZRotation()`.
    fn get_mask_model_pts_z_rotation(&self) -> ConstEtomoNumber;

    /// Java `getMaskModelPtsYRotation()`.
    fn get_mask_model_pts_y_rotation(&self) -> Option<String>;

    /// Java `getMaskTypeVolume()`.
    fn get_mask_type_volume(&self) -> Option<String>;

    /// Java `getNWeightGroup()`.
    fn get_n_weight_group(&self) -> ConstEtomoNumber;

    /// Java `isTiltRange()`.
    fn is_tilt_range(&self) -> bool;

    /// Java `isManualCylinderOrientation()`.
    fn is_manual_cylinder_orientation(&self) -> bool;

    /// Java `isTiltRangeMultiAxes()`.
    fn is_tilt_range_multi_axes(&self) -> bool;

    /// Java `getCylinderHeight()`.
    fn get_cylinder_height(&self) -> Option<String>;

    /// Java `getMaskBlurStdDev()`.
    fn get_mask_blur_std_dev(&self) -> Option<String>;

    /// Java `getLowCutoffCutoff(int)`.
    fn get_low_cutoff_cutoff(&self, key: i32) -> Option<String>;

    /// Java `getLowCutoffSigma(int)`.
    fn get_low_cutoff_sigma(&self, key: i32) -> Option<String>;

    /// Java `isLowCutoff()`.
    fn is_low_cutoff(&self) -> bool;

    /// Java `isFlgAlignAverages()`.
    fn is_flg_align_averages(&self) -> bool;

    /// Java `getCNSymmetricAveraging()`.
    fn get_cn_symmetric_averaging(&self) -> ConstEtomoNumber;

    /// Java `isCNSymmetricAveraging()`.
    fn is_cn_symmetric_averaging(&self) -> bool;
}
