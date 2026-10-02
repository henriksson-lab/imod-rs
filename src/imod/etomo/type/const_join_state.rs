//! `IMOD/Etomo/src/etomo/type/ConstJoinState.java`.
//!
//! The interface's `ConstEtomoNumber` and `ScriptParameter` getters return the
//! implementing state's own instances in Java.  `JoinState` is shared across threads and
//! keeps each field behind its own lock (see `join_state.rs`), so these getters return
//! a copy of the field taken under its lock; every caller only reads it.  The
//! `IntKeyList.Walker` getters return a walker borrowing the state's (locked) list, which
//! - like Java's walker - reads the live list one synchronised call at a time.
//!
//! `getRefineTrial` is declared `ConstEtomoNumber` but returns an `EtomoBoolean2`, whose
//! `is()`/`isNull()` overrides Java dispatches to virtually; the copy is therefore an
//! `EtomoBoolean2`, so `get_refine_trial().is()` keeps the override.
#![allow(dead_code)]

use super::const_etomo_number::ConstEtomoNumber;
use super::etomo_boolean2::EtomoBoolean2;
use super::int_key_list::Walker;
use super::script_parameter::ScriptParameter;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java interface `ConstJoinState`.
pub trait ConstJoinState {
    /// Java `getJoinStartListWalker(boolean)`.
    fn get_join_start_list_walker(&self, trial: bool) -> Walker<'_>;

    /// Java `getJoinEndListWalker(boolean)`.
    fn get_join_end_list_walker(&self, trial: bool) -> Walker<'_>;

    /// Java `getJoinShiftInX(boolean)`.
    fn get_join_shift_in_x(&self, trial: bool) -> ConstEtomoNumber;

    /// Java `getJoinShiftInY(boolean)`.
    fn get_join_shift_in_y(&self, trial: bool) -> ConstEtomoNumber;

    /// Java `isJoinLocalFits(boolean)`.
    fn is_join_local_fits(&self, trial: bool) -> bool;

    /// Java `getJoinShiftInXParameter(boolean)`.
    fn get_join_shift_in_x_parameter(&self, trial: bool) -> ScriptParameter;

    /// Java `getJoinShiftInYParameter(boolean)`.
    fn get_join_shift_in_y_parameter(&self, trial: bool) -> ScriptParameter;

    /// Java `getJoinTrialBinning()`.
    fn get_join_trial_binning(&self) -> ConstEtomoNumber;

    /// Java `getJoinAlignmentRefSection(boolean)`.
    fn get_join_alignment_ref_section(&self, trial: bool) -> ConstEtomoNumber;

    /// Java `getJoinSizeInX(boolean)`.
    fn get_join_size_in_x(&self, trial: bool) -> ConstEtomoNumber;

    /// Java `getJoinSizeInY(boolean)`.
    fn get_join_size_in_y(&self, trial: bool) -> ConstEtomoNumber;

    /// Java `getJoinSizeInXParameter(boolean)`.
    fn get_join_size_in_x_parameter(&self, trial: bool) -> ScriptParameter;

    /// Java `getJoinSizeInYParameter(boolean)`.
    fn get_join_size_in_y_parameter(&self, trial: bool) -> ScriptParameter;

    /// Java `getRefineTrial()`.
    fn get_refine_trial(&self) -> EtomoBoolean2;

    /// Java `getRefineStartListWalker()`.
    fn get_refine_start_list_walker(&self) -> Walker<'_>;

    /// Java `getRefineEndListWalker()`.
    fn get_refine_end_list_walker(&self) -> Walker<'_>;

    /// Java `getXfModelOutputFile()`.
    fn get_xf_model_output_file(&self) -> Option<String>;
}
