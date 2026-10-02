//! `IMOD/Etomo/src/etomo/type/Run3dmodMenuOptions.java`.
//!
//! **Value, not reference.**  The Java object is mutable and passed by reference, so
//! `ImodState.open` (`setNoOptions`, `orGlobalOptions`, `setAllowBinningInZ`) changes
//! the caller's instance as a side effect.  The translation passes it by value
//! (`Option<Run3dmodMenuOptions>` where the source can pass null): the callee's changes
//! do not leak back into an options object the caller may reuse for a later open, which
//! in the source could carry one 3dmod's `noOptions` clearing into the next.
//!
//! The fields are `pub` because the translated Swing menus build an instance with a
//! struct literal; the Java fields are private and are changed through the setters.

use crate::imod::etomo::ui::swing::ui_harness;

/// Java `Run3dmodMenuOptions`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Run3dmodMenuOptions {
    /// Java field `startupWindow`, initialised to false.
    pub startup_window: bool,
    /// Java field `binBy2`, initialised to false.
    pub bin_by_2: bool,
    /// Java field `allowBinningInZ`, initialised to false.
    pub allow_binning_in_z: bool,
    /// Java field `noOptions`, initialised to false.
    pub no_options: bool,
}

impl Run3dmodMenuOptions {
    /// Java `Run3dmodMenuOptions()`.
    pub fn new() -> Run3dmodMenuOptions {
        Run3dmodMenuOptions::default()
    }

    /// Java `setNoOptions`.
    pub fn set_no_options(&mut self, no_options: bool) {
        self.no_options = no_options;
        if no_options {
            self.startup_window = false;
            self.bin_by_2 = false;
        }
    }

    /// Java `orGlobalOptions`.
    pub fn or_global_options(&mut self) {
        if self.no_options {
            return;
        }
        let startup_window = ui_harness::INSTANCE
            .with(|ui_harness| ui_harness.is_3dmod_startup_window());
        if startup_window {
            self.startup_window = true;
        }
        let bin_by_2 =
            ui_harness::INSTANCE.with(|ui_harness| ui_harness.is_3dmod_bin_by_2());
        if bin_by_2 {
            self.bin_by_2 = true;
        }
    }

    /// Java `setAllowBinningInZ`.
    pub fn set_allow_binning_in_z(&mut self, allow_binning_in_z: bool) {
        self.allow_binning_in_z = allow_binning_in_z;
    }

    /// Java `setStartupWindow`.
    pub fn set_startup_window(&mut self, startup_window: bool) {
        if self.no_options {
            return;
        }
        self.startup_window = startup_window;
    }

    /// Java `setBinBy2`.
    pub fn set_bin_by_2(&mut self, bin_by_2: bool) {
        if self.no_options {
            return;
        }
        self.bin_by_2 = bin_by_2;
    }

    /// Java `isBinBy2`.
    pub fn is_bin_by_2(&self) -> bool {
        self.bin_by_2
    }

    /// Java `isAllowBinningInZ`.
    pub fn is_allow_binning_in_z(&self) -> bool {
        self.allow_binning_in_z
    }

    /// Java `isStartupWindow`.
    pub fn is_startup_window(&self) -> bool {
        self.startup_window
    }
}

/// Java `toString`.
impl std::fmt::Display for Run3dmodMenuOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[startupWindow={},binBy2={},\nallowBinningInZ={},noOptions={}]",
            self.startup_window, self.bin_by_2, self.allow_binning_in_z, self.no_options
        )
    }
}
