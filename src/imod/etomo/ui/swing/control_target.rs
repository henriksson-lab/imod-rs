//! `IMOD/Etomo/src/etomo/ui/swing/ControlTarget.java`.

use std::path::{Path, PathBuf};

use super::control_state::ControlState;

/// Java package-private `interface ControlTarget`.
pub trait ControlTarget {
    /// Java `clear()`.
    fn clear(&self);

    /// Java `setText(File)`.
    fn set_text_file(&self, file: Option<&Path>);

    /// Java `setText(File[])`.
    fn set_text_file_array(&self, files: Option<&[PathBuf]>);

    /// Java `getLabel()`.
    fn get_label(&self) -> Option<String>;

    /// Java `setComponentControl(boolean, ControlState)`.
    fn set_component_control(&self, control: bool, state: Option<&'static ControlState>);

    /// Java `setEnableControl(boolean, ControlState)`.
    fn set_enable_control(&self, control: bool, state: Option<&'static ControlState>);

    /// Java `sendControlEvent()`.
    fn send_control_event(&self);

    /// Java `isLocalDir(String)`.
    fn is_local_dir(&self, current_directory: Option<&str>) -> bool;
}
