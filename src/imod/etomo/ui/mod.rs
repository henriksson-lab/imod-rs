//! Non-GUI translations of `IMOD/Etomo/src/etomo/ui` source units.
/// Java `etomo.ui.UIComponent` frontend-identity boundary.
pub trait UiComponent: std::any::Any {}

pub mod browsing_directory;
pub mod directive_display_settings;
pub mod field_type;
pub mod log_properties;
pub mod queue_table_data_event;
pub mod queue_table_event;
pub mod queue_table_listener;
pub mod shared_strings;
pub mod standard_bar_string;
pub mod swing;
pub mod setup_recon_interface;
pub mod boolean_efield_interface;
pub mod boolean_field_interface;
pub mod boolean_field_setting;
pub mod boolean_flag_extension;
pub mod boolean_flag_origin;
pub mod boolean_state_extension;
pub mod boolean_text_field_interface;
pub mod expander;
pub mod field;
pub mod field_displayer;
pub mod field_setting_bundle;
pub mod field_setting_interface;
pub mod field_validation_failed_exception;
pub mod flag_display;
pub mod flag_origin_listener;
pub mod flag_type;
pub mod processor_table_field;
pub mod run_3dmod_menu_target;
pub mod table_field;
pub mod text_field_interface;
pub mod text_field_setting;
pub mod text_flag_extension;
pub mod text_flag_origin;
pub mod text_state_extension;
pub mod ui_component;
pub mod value_manipulation_field;
pub mod value_manipulation_listener;
pub mod setup_recon_ui_harness;
pub mod front_page_ui_harness;
