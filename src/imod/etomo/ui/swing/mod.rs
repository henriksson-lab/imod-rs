//! Translations of `IMOD/Etomo/src/etomo/ui/swing` source units.
pub mod abstract_frame;
pub mod abstract_parallel_dialog;
pub mod abstract_tilt_panel;
pub mod axis_process_panel;
pub mod axis_progress_panel;
pub mod busy_status_panel;
pub mod etomo_frame;
pub mod etomo_menu;
pub mod etomo_panel;
pub mod main_frame;
pub mod main_panel;
pub mod manager_frame;
pub mod panel;
pub mod parallel_dialog;
pub mod parallel_panel;
pub mod process_control_panel;
pub mod process_dialog;
pub mod progress_panel;
pub mod scroll_panel;
pub mod settings_dialog;
pub mod sub_frame;
pub mod tilt_panel;
pub mod token;
pub mod tool_panel;
pub mod tools_dialog;
pub mod window_switch;

// `UIHarness.java` owns the application-frame boundary.  It depends on Slint
// only for the explicitly optional GUI build, just as Java only constructs its
// `MainFrame` when it is not headless.
#[cfg(feature = "gui")]
pub mod ui_harness;
