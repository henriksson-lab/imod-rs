//! Non-GUI translations of `IMOD/Etomo/src/etomo/ui` source units.
/// Java `etomo.ui.UIComponent` frontend-identity boundary.
pub trait UiComponent: std::any::Any {}

pub mod browsing_directory;
pub mod field_type;
pub mod log_properties;
pub mod queue_table_event;
pub mod queue_table_listener;
pub mod shared_strings;
pub mod standard_bar_string;
pub mod swing;
