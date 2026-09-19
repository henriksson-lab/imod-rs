//! `IMOD/Etomo/src/etomo/ui/swing/SettingsDialog.java`.
//!
//! `UserConfiguration`, TemplatePanel, Network/CpuAdoc, and concrete Swing
//! controls remain source-unit boundaries. This module owns the exact dialog
//! field values and SettingsDialog's transformations/actions around them.
#![allow(dead_code)]
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::storable::Storable;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const TITLE: &str = "Etomo Settings";
pub const CANCEL: &str = "Cancel";
pub const APPLY: &str = "Apply";
pub const DONE: &str = "Done";
/// Direct data boundary corresponding to the source's `UserConfiguration` getters/setters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserConfigurationValues {
    pub tooltips_initial_delay_ms: i32,
    pub tooltips_dismiss_delay_ms: i32,
    pub auto_fit: bool,
    pub native_laf: bool,
    pub advanced_dialogs: bool,
    pub compact_display: bool,
    pub font_size: i32,
    pub font_family: String,
    pub single_axis: bool,
    pub montage: bool,
    pub no_parallel_processing: bool,
    pub gpu_processing_default: bool,
    pub remove_excluded_views: bool,
    pub tilt_angles_rawtlt_file: bool,
    pub swap_y_and_z: bool,
    pub set_fei_pixel_size: bool,
    pub parallel_processing: bool,
    pub gpu_processing: bool,
    pub cpus: String,
    pub local_gpus: String,
    pub parallel_table_size: String,
    pub join_table_size: String,
    pub peet_table_size: String,
    pub batch_table_size: String,
    pub user_template_dir: Option<PathBuf>,
    pub smtp_server: String,
}

impl Default for UserConfigurationValues {
    fn default() -> Self {
        Self {
            // These are the values the director has historically exposed
            // before a user configuration is loaded.  Keeping them here also
            // means an absent .etomo file produces a usable settings dialog.
            tooltips_initial_delay_ms: 1_000,
            tooltips_dismiss_delay_ms: 4_000,
            auto_fit: false,
            native_laf: false,
            advanced_dialogs: false,
            compact_display: false,
            font_size: 12,
            font_family: "Dialog".to_owned(),
            single_axis: false,
            montage: false,
            no_parallel_processing: false,
            gpu_processing_default: false,
            remove_excluded_views: false,
            tilt_angles_rawtlt_file: false,
            swap_y_and_z: false,
            set_fei_pixel_size: false,
            parallel_processing: false,
            gpu_processing: false,
            cpus: String::new(),
            local_gpus: String::new(),
            parallel_table_size: String::new(),
            join_table_size: String::new(),
            peet_table_size: String::new(),
            batch_table_size: String::new(),
            user_template_dir: None,
            smtp_server: String::new(),
        }
    }
}

/// Persistent subset of Java's `UserConfiguration` used by the translated
/// director/settings path.  A deterministic `Settings.` namespace prevents
/// settings from colliding with project parameter files.
impl Storable for UserConfigurationValues {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        macro_rules! put {
            ($name:literal, $value:expr) => {
                properties.insert(concat!("Settings.", $name).to_owned(), $value.to_string());
            };
        }
        put!("TooltipsInitialDelay", self.tooltips_initial_delay_ms);
        put!("TooltipsDismissDelay", self.tooltips_dismiss_delay_ms);
        put!("AutoFit", self.auto_fit);
        put!("NativeLookAndFeel", self.native_laf);
        put!("AdvancedDialogs", self.advanced_dialogs);
        put!("CompactDisplay", self.compact_display);
        put!("FontSize", self.font_size);
        put!("FontFamily", self.font_family);
        put!("ParallelProcessing", self.parallel_processing);
        put!("GpuProcessing", self.gpu_processing);
        put!("Cpus", self.cpus);
        put!("LocalGpus", self.local_gpus);
        put!("SmtpServer", self.smtp_server);
        if let Some(directory) = &self.user_template_dir {
            put!("UserTemplateDir", directory.display());
        }
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, _prepend: &str) {
        self.store(properties);
    }

    fn load(&mut self, properties: &BTreeMap<String, String>) {
        let get = |name| {
            properties
                .get(&format!("Settings.{name}"))
                .map(String::as_str)
        };
        macro_rules! read {
            ($field:ident, $name:literal, $type:ty) => {
                if let Some(value) = get($name).and_then(|value| value.parse::<$type>().ok()) {
                    self.$field = value;
                }
            };
        }
        read!(tooltips_initial_delay_ms, "TooltipsInitialDelay", i32);
        read!(tooltips_dismiss_delay_ms, "TooltipsDismissDelay", i32);
        read!(auto_fit, "AutoFit", bool);
        read!(native_laf, "NativeLookAndFeel", bool);
        read!(advanced_dialogs, "AdvancedDialogs", bool);
        read!(compact_display, "CompactDisplay", bool);
        read!(font_size, "FontSize", i32);
        read!(parallel_processing, "ParallelProcessing", bool);
        read!(gpu_processing, "GpuProcessing", bool);
        for (name, field) in [
            ("FontFamily", &mut self.font_family),
            ("Cpus", &mut self.cpus),
            ("LocalGpus", &mut self.local_gpus),
            ("SmtpServer", &mut self.smtp_server),
        ] {
            if let Some(value) = get(name) {
                *field = value.to_owned();
            }
        }
        self.user_template_dir = get("UserTemplateDir").map(PathBuf::from);
    }

    fn load_with_prepend(&mut self, properties: &BTreeMap<String, String>, _prepend: &str) {
        self.load(properties);
    }
}
/// Java private static `FontFamilies`.
#[derive(Clone, Debug, Default)]
pub struct FontFamilies {
    pub usable: Vec<String>,
    pub default_index: isize,
}
impl FontFamilies {
    pub fn new(available: &[String]) -> Self {
        let mut usable = vec![];
        let mut default_index = -1;
        for name in available {
            if name.contains('\'') {
                eprintln!("Removing unusable font family:{name}");
            } else {
                if name.eq_ignore_ascii_case("dialog") {
                    default_index = usable.len() as isize;
                }
                usable.push(name.clone());
            }
        }
        Self {
            usable,
            default_index,
        }
    }
    pub fn get_font_families(&self) -> &[String] {
        &self.usable
    }
    pub fn get_index(&self, name: &str) -> isize {
        self.usable
            .iter()
            .position(|v| v.eq_ignore_ascii_case(name))
            .map_or(self.default_index, |i| i as isize)
    }
    pub fn get_name(&self, index: isize) -> Option<&str> {
        self.usable.get(index.max(0) as usize).map(String::as_str)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettingsAction {
    Cancel,
    Apply,
    Done,
    ParallelProcessingChanged,
    GpuProcessingChanged,
}
/// Fields and implemented behavior of Java's final `SettingsDialog`.
pub struct SettingsDialog {
    pub font_families: FontFamilies,
    pub selected_font_index: isize,
    pub font_size: String,
    pub tooltips_initial_delay: String,
    pub tooltips_dismiss_delay: String,
    pub native_laf: bool,
    pub advanced_dialogs: bool,
    pub auto_fit: bool,
    pub compact_display: bool,
    pub parallel_processing: bool,
    pub gpu_processing: bool,
    pub cpus: String,
    pub local_gpus: String,
    pub single_axis: bool,
    pub montage: bool,
    pub no_parallel_processing: bool,
    pub gpu_processing_default: bool,
    pub tilt_angles_rawtlt_file: bool,
    pub swap_y_and_z: bool,
    pub parallel_table_size: String,
    pub join_table_size: String,
    pub peet_table_size: String,
    pub batch_table_size: String,
    pub set_fei_pixel_size: bool,
    pub user_template_dir: Option<PathBuf>,
    pub smtp_server: String,
    pub remove_excluded_views: bool,
    pub cpu_adoc_viable: bool,
    pub manager: &'static dyn BaseManager,
    pub property_user_dir: String,
    pub visible: bool,
    pub cpus_enabled: bool,
    pub gpu_enabled: bool,
    pub local_gpus_enabled: bool,
    pub parallel_enabled: bool,
    pub auto_fit_enabled: bool,
    pub applied: bool,
    pub saved: bool,
    pub closed: bool,
}
impl SettingsDialog {
    fn new(
        manager: &'static dyn BaseManager,
        property_user_dir: impl Into<String>,
        available_fonts: &[String],
        cpu_adoc_viable: bool,
    ) -> Self {
        let fonts = FontFamilies::new(available_fonts);
        Self {
            selected_font_index: fonts.default_index,
            font_families: fonts,
            font_size: String::new(),
            tooltips_initial_delay: String::new(),
            tooltips_dismiss_delay: String::new(),
            native_laf: false,
            advanced_dialogs: false,
            auto_fit: false,
            compact_display: false,
            parallel_processing: false,
            gpu_processing: false,
            cpus: String::new(),
            local_gpus: String::new(),
            single_axis: false,
            montage: false,
            no_parallel_processing: false,
            gpu_processing_default: false,
            tilt_angles_rawtlt_file: false,
            swap_y_and_z: false,
            parallel_table_size: String::new(),
            join_table_size: String::new(),
            peet_table_size: String::new(),
            batch_table_size: String::new(),
            set_fei_pixel_size: false,
            user_template_dir: None,
            smtp_server: String::new(),
            remove_excluded_views: false,
            cpu_adoc_viable,
            manager,
            property_user_dir: property_user_dir.into(),
            visible: false,
            cpus_enabled: false,
            gpu_enabled: false,
            local_gpus_enabled: false,
            parallel_enabled: true,
            auto_fit_enabled: false,
            applied: false,
            saved: false,
            closed: false,
        }
    }
    /// `getInstance(BaseManager, String)`; font discovery is native presentation input.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        property_user_dir: impl Into<String>,
        available_fonts: &[String],
        cpu_adoc_viable: bool,
    ) -> Self {
        let mut dialog = Self::new(manager, property_user_dir, available_fonts, cpu_adoc_viable);
        dialog.build_dialog();
        dialog.set_tooltips();
        dialog.add_listeners();
        dialog
    }
    fn build_dialog(&mut self) {
        self.visible = true;
        self.auto_fit_enabled = false;
    }
    pub fn set_scroll_bar_increments(&self) -> (i32, i32) {
        (10, 50)
    }
    /// `setParametersFromNetwork`; network discovery inputs are explicit because Network.java is unported.
    pub fn set_parameters_from_network(
        &mut self,
        external: bool,
        local_cpus: Option<i32>,
        local_gpus: Option<i32>,
    ) -> bool {
        if !external {
            return false;
        }
        self.parallel_enabled = false;
        self.parallel_processing = true;
        match local_cpus {
            Some(v) => self.cpus = v.to_string(),
            None => self.cpus.clear(),
        }
        match local_gpus {
            Some(v) if v > 0 => {
                self.gpu_processing = true;
                self.local_gpus = v.to_string();
            }
            _ => {
                self.gpu_processing = false;
                if self.cpu_adoc_viable {
                    self.local_gpus.clear();
                }
            }
        }
        true
    }
    pub fn update_display(&mut self) {
        self.cpus_enabled = self.parallel_enabled && self.parallel_processing;
        self.gpu_enabled = !self.cpu_adoc_viable && self.parallel_processing;
        self.local_gpus_enabled =
            self.parallel_processing && self.gpu_enabled && self.gpu_processing;
    }
    fn add_listeners(&mut self) {}
    /// `setParameters(UserConfiguration)`.
    pub fn set_parameters(&mut self, config: &UserConfigurationValues) {
        self.tooltips_initial_delay = (config.tooltips_initial_delay_ms / 1000).to_string();
        self.tooltips_dismiss_delay = (config.tooltips_dismiss_delay_ms / 1000).to_string();
        self.auto_fit = config.auto_fit;
        self.native_laf = config.native_laf;
        self.advanced_dialogs = config.advanced_dialogs;
        self.compact_display = config.compact_display;
        self.selected_font_index = self.font_families.get_index(&config.font_family);
        self.font_size = config.font_size.to_string();
        self.single_axis = config.single_axis;
        self.montage = config.montage;
        self.no_parallel_processing = config.no_parallel_processing;
        self.gpu_processing_default = config.gpu_processing_default;
        self.remove_excluded_views = config.remove_excluded_views;
        self.tilt_angles_rawtlt_file = config.tilt_angles_rawtlt_file;
        self.swap_y_and_z = config.swap_y_and_z;
        self.set_fei_pixel_size = config.set_fei_pixel_size;
        self.parallel_processing = config.parallel_processing;
        self.gpu_processing = config.gpu_processing;
        self.cpus = config.cpus.clone();
        self.local_gpus = config.local_gpus.clone();
        self.parallel_table_size = config.parallel_table_size.clone();
        self.join_table_size = config.join_table_size.clone();
        self.peet_table_size = config.peet_table_size.clone();
        self.batch_table_size = config.batch_table_size.clone();
        self.smtp_server = config.smtp_server.clone();
        self.user_template_dir = config.user_template_dir.clone();
        self.update_display();
    }
    pub fn equals_user_template_dir(&self, input: Option<&std::path::Path>) -> bool {
        self.user_template_dir.as_deref() == input
    }
    pub fn get_user_template_dir(&self) -> Option<&std::path::Path> {
        self.user_template_dir.as_deref()
    }
    /// `getParameters(UserConfiguration)`.
    pub fn get_parameters(&self, config: &mut UserConfigurationValues) -> Result<(), String> {
        config.tooltips_initial_delay_ms = (self
            .tooltips_initial_delay
            .parse::<f64>()
            .map_err(|_| "invalid initial tooltip delay")?
            * 1000.) as i32;
        config.tooltips_dismiss_delay_ms = (self
            .tooltips_dismiss_delay
            .parse::<f64>()
            .map_err(|_| "invalid dismiss tooltip delay")?
            * 1000.) as i32;
        config.auto_fit = self.auto_fit;
        config.native_laf = self.native_laf;
        config.advanced_dialogs = self.advanced_dialogs;
        config.compact_display = self.compact_display;
        config.font_size = self.font_size.parse().map_err(|_| "invalid font size")?;
        config.font_family = self
            .font_families
            .get_name(self.selected_font_index)
            .unwrap_or("")
            .into();
        config.single_axis = self.single_axis;
        config.montage = self.montage;
        config.no_parallel_processing = self.no_parallel_processing;
        config.gpu_processing_default = self.gpu_processing_default;
        config.remove_excluded_views = self.remove_excluded_views;
        config.tilt_angles_rawtlt_file = self.tilt_angles_rawtlt_file;
        config.swap_y_and_z = self.swap_y_and_z;
        config.set_fei_pixel_size = self.set_fei_pixel_size;
        config.parallel_processing = self.parallel_processing;
        config.gpu_processing = self.gpu_processing;
        config.cpus = self.cpus.clone();
        config.local_gpus = self.local_gpus.clone();
        config.parallel_table_size = self.parallel_table_size.clone();
        config.join_table_size = self.join_table_size.clone();
        config.peet_table_size = self.peet_table_size.clone();
        config.batch_table_size = self.batch_table_size.clone();
        config.user_template_dir = self.user_template_dir.clone();
        config.smtp_server = self.smtp_server.clone();
        Ok(())
    }
    pub fn is_appearance_setting_changed(&self, config: &UserConfigurationValues) -> bool {
        config.native_laf != self.native_laf
            || config.compact_display != self.compact_display
            || config.single_axis != self.single_axis
            || config.montage != self.montage
            || config.no_parallel_processing != self.no_parallel_processing
            || config.gpu_processing_default != self.gpu_processing_default
            || config.remove_excluded_views != self.remove_excluded_views
            || config.tilt_angles_rawtlt_file != self.tilt_angles_rawtlt_file
            || config.swap_y_and_z != self.swap_y_and_z
            || config.set_fei_pixel_size != self.set_fei_pixel_size
            || config.font_size.to_string() != self.font_size
            || self
                .font_families
                .get_name(self.selected_font_index)
                .unwrap_or("")
                != config.font_family
            || config.parallel_processing != self.parallel_processing
            || config.gpu_processing != self.gpu_processing
            || config.cpus != self.cpus
            || config.parallel_table_size != self.parallel_table_size
            || config.join_table_size != self.join_table_size
            || config.peet_table_size != self.peet_table_size
            || config.batch_table_size != self.batch_table_size
            || config.smtp_server != self.smtp_server
    }
    /// `action(String)`, representing EtomoDirector calls with dialog-owned state.
    pub fn action(&mut self, command: &str) -> Option<SettingsAction> {
        let action = match command {
            CANCEL => {
                self.closed = true;
                SettingsAction::Cancel
            }
            APPLY => {
                self.applied = true;
                SettingsAction::Apply
            }
            DONE => {
                self.applied = true;
                self.saved = true;
                self.closed = true;
                SettingsAction::Done
            }
            "Enable parallel processing" => SettingsAction::ParallelProcessingChanged,
            "Enable graphics processing" => SettingsAction::GpuProcessingChanged,
            _ => return None,
        };
        self.update_display();
        Some(action)
    }
    pub fn set_tooltips(&mut self) {}
}
pub struct SettingsDialogListener;
impl SettingsDialogListener {
    pub fn action_performed(dialog: &mut SettingsDialog, command: &str) -> Option<SettingsAction> {
        dialog.action(command)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::directive_editor_manager::DirectiveEditorManager;
    fn dialog() -> SettingsDialog {
        SettingsDialog::get_instance(
            DirectiveEditorManager::new(None, None, None, None),
            "/tmp",
            &["Dialog".into(), "O'Reilly".into()],
            false,
        )
    }
    #[test]
    fn font_filter_and_config_roundtrip_follow_source() {
        let mut d = dialog();
        assert_eq!(d.font_families.get_font_families(), ["Dialog"]);
        let c = UserConfigurationValues {
            tooltips_initial_delay_ms: 1500,
            tooltips_dismiss_delay_ms: 2000,
            font_size: 12,
            font_family: "Dialog".into(),
            cpus: "8".into(),
            ..Default::default()
        };
        d.set_parameters(&c);
        let mut result = UserConfigurationValues::default();
        d.get_parameters(&mut result).unwrap();
        assert_eq!(result.tooltips_initial_delay_ms, 1000);
        assert_eq!(result.cpus, "8");
    }
    #[test]
    fn action_done_applies_saves_and_closes() {
        let mut d = dialog();
        assert_eq!(d.action(DONE), Some(SettingsAction::Done));
        assert!(d.applied && d.saved && d.closed);
    }
    #[test]
    fn network_control_has_source_precedence() {
        let mut d = dialog();
        assert!(d.set_parameters_from_network(true, Some(16), Some(2)));
        d.update_display();
        assert!(!d.parallel_enabled && d.parallel_processing);
        // `dialog()` builds this with `cpuAdocViable == false`, and the network
        // path only ever calls `cbGpuProcessing.setSelected(...)`
        // (`SettingsDialog.java:275,281`) -- it never touches the *enabled*
        // state.  That is decided solely by `updateDisplay`:
        // `cbGpuProcessing.setEnabled(!cpuAdocViable && cbParallelProcessing
        // .isSelected())` (`SettingsDialog.java:300`), which is `!false && true`
        // here.  The Java comment on the line above it says why: "When
        // IMOD_PROCESSORS is in use, the GPUs can still be set from this
        // dialog."  This assertion used to read `!d.gpu_enabled`, which is the
        // opposite of the source.
        assert!(d.gpu_enabled);
        // `ltfNumberOfLocalGPUs.setEnabled(cbParallelProcessing.isSelected()
        // && cbGpuProcessing.isEnabled() && cbGpuProcessing.isSelected())`
        // (`SettingsDialog.java:301-302`); `localHostGpus > 0` selected the box
        // and set the count (`:275-276`).
        assert!(d.gpu_processing && d.local_gpus_enabled);
        assert_eq!(d.local_gpus, "2");
        assert_eq!(d.cpus, "16");
    }
}
