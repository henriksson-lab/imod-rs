//! Source-mirrored non-GUI command line model from `etomo/Arguments.java`.
#![allow(dead_code)]

use super::storage::autodoc::autodoc_factory;
use super::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use super::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use super::storage::autodoc::read_only_section::ReadOnlySection;
use super::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use super::storage::autodoc::section::Section;
use super::storage::data_file_filter::DataFileFilter;
use super::storage::log_file::LogFileError;
use super::r#type::axis_id::AxisID;
use super::r#type::axis_type::AxisType;
use super::r#type::debug_level::DebugLevel;
use super::r#type::etomo_autodoc;
use super::r#type::image_filename_style::ImageFilenameStyle;
use super::r#type::view_type::ViewType;
use std::path::{Path, PathBuf};

pub const DIRECTIVE_TAG: &str = "-directive";
pub const SELFTEST_TAG: &str = "-selftest";
pub const TEST_TAG: &str = "-test";
const HEADLESS_TAG: &str = "-headless";
const HELP_TAGS: [&str; 4] = ["--help", "-help", "--h", "-h"];
pub const DEBUG_TAG: &str = "-debug";
const MEMORY_TAG: &str = "-memory";
const NEWSTUFF_TAG: &str = "-newstuff";
const TIMESTAMP_TAG: &str = "-timestamp";
pub const USER_TEMPLATE_LOC_TAG: &str = "-userTemplateLoc";
pub const IMAGE_FILENAME_STYLE_TAG: &str = "-namingstyle";
pub const MOVE_B_TAG: &str = "-moveb";
pub const NAMES_TAG: &str = "-names";

/// `Arguments.TestLevel`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TestLevel {
    Default,
    NoWait,
    Stress,
    Fail,
}
impl TestLevel {
    pub fn param_value(self) -> i32 {
        match self {
            Self::Default => 1,
            Self::NoWait => 2,
            Self::Stress => 3,
            Self::Fail => 4,
        }
    }
    pub fn get_instance(value: &str) -> Option<Self> {
        value.parse().ok().and_then(Self::get_instance_int)
    }
    pub fn get_instance_int(value: i32) -> Option<Self> {
        match value {
            1 => Some(Self::Default),
            2 => Some(Self::NoWait),
            3 => Some(Self::Stress),
            4 => Some(Self::Fail),
            _ => None,
        }
    }
    pub fn get_param_value(self) -> String {
        self.param_value().to_string()
    }
    pub fn is_sleep_allowed(self) -> bool {
        !matches!(self, Self::NoWait | Self::Fail)
    }
    pub fn is_blocking_ids(self) -> bool {
        self.get_num_blocking_ids() != 0
    }
    pub fn get_num_blocking_ids(self) -> i32 {
        match self {
            Self::Stress => 2,
            Self::Fail => 1,
            _ => 0,
        }
    }
    pub fn get_descr(self) -> &'static str {
        match self {
            Self::Default => "Default test",
            Self::NoWait => "No Wait test",
            Self::Stress => "Stress test",
            Self::Fail => "Fail test",
        }
    }
}
impl std::fmt::Display for TestLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.get_descr())
    }
}

/// `Arguments.GrabItParameter`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrabItParameter {
    Batchruntomo,
    CopyTomoComs,
    ParallelProcessing,
}
impl GrabItParameter {
    pub fn get_instance(value: Option<&str>) -> Option<Self> {
        match value {
            Some("b") => Some(Self::Batchruntomo),
            Some("c") => Some(Self::CopyTomoComs),
            Some("p") => Some(Self::ParallelProcessing),
            _ => None,
        }
    }
}

/// All state fields in Java `Arguments`.
#[derive(Clone, Debug)]
pub struct Arguments {
    pub param_file_name_list: Vec<String>,
    pub debug: bool,
    pub debug_level: DebugLevel,
    pub headless: bool,
    pub test: i32,
    pub create: bool,
    pub self_test: bool,
    pub newstuff: bool,
    pub display_memory: bool,
    pub help: bool,
    pub print_names: bool,
    pub i_display_memory: i32,
    pub axis: bool,
    pub at_axis: Option<AxisType>,
    pub frame: bool,
    pub vt_frame: Option<ViewType>,
    pub scan: bool,
    pub raw_image_stack: bool,
    pub s_raw_image_stack: Option<String>,
    pub dir: bool,
    pub f_dir: Option<PathBuf>,
    pub exit: bool,
    pub listen: bool,
    pub auto_close_3dmod: bool,
    pub ignore_loc: bool,
    pub recon_automation: bool,
    pub ignore_settings: bool,
    pub actions: bool,
    pub directive: bool,
    pub f_directive: Option<PathBuf>,
    pub error_message_list: Vec<String>,
    pub warning_message_list: Vec<String>,
    pub fiducial: bool,
    pub en_fiducial: Option<f64>,
    pub fiducial_invalid_reason: String,
    pub cpus: bool,
    pub gpus: bool,
    pub from_brt: bool,
    pub user_template_loc: Option<PathBuf>,
    pub plugin: bool,
    pub grab_it: bool,
    pub image_filename_style: Option<ImageFilenameStyle>,
    pub grab_it_parameter: Option<GrabItParameter>,
    pub move_b: bool,
    pub computer_section: bool,
    pub computer_section_value: Option<String>,
    pub queue_section: bool,
    pub queue_section_value: Option<String>,
    pub no_load: bool,
    pub debug_always_on: bool,
}
impl Default for Arguments {
    fn default() -> Self {
        Self {
            param_file_name_list: vec![],
            debug: false,
            debug_level: DebugLevel::Off,
            headless: false,
            test: TestLevel::Default.param_value(),
            create: false,
            self_test: false,
            newstuff: false,
            display_memory: false,
            help: false,
            print_names: false,
            i_display_memory: 0,
            axis: false,
            at_axis: None,
            frame: false,
            vt_frame: None,
            scan: false,
            raw_image_stack: false,
            s_raw_image_stack: None,
            dir: false,
            f_dir: None,
            exit: false,
            listen: false,
            auto_close_3dmod: false,
            ignore_loc: false,
            recon_automation: false,
            ignore_settings: false,
            actions: false,
            directive: false,
            f_directive: None,
            error_message_list: vec![],
            warning_message_list: vec![],
            fiducial: false,
            en_fiducial: None,
            fiducial_invalid_reason: String::new(),
            cpus: false,
            gpus: false,
            from_brt: false,
            user_template_loc: None,
            plugin: false,
            grab_it: false,
            image_filename_style: None,
            grab_it_parameter: None,
            move_b: false,
            computer_section: false,
            computer_section_value: None,
            queue_section: false,
            queue_section_value: None,
            no_load: false,
            debug_always_on: false,
        }
    }
}
impl Arguments {
    pub fn new() -> Self {
        Self::default()
    }
    /// Java package-private static `printHelpMessage()`.
    ///
    /// Walks the sections of `IMOD/autodoc/etomo.adoc` in *file* order, across
    /// collection types - which is what `autodoc.getSectionLocation()` plus
    /// `autodoc.nextSection(sectionLocation)` do and what the C `libcfshr` autodoc
    /// cannot.
    pub fn print_help_message() {
        // Java's `ReadOnlyAutodoc autodoc = null` and the try/catch around the body.
        let autodoc = unsafe {
            autodoc_factory::get_instance(None, Some(autodoc_factory::ETOMO), AxisID::Only, false)
        };
        let autodoc = match autodoc {
            Ok(autodoc) => autodoc,
            Err(e) => {
                match e {
                    // `catch (final LockException except) {}`.
                    LogFileError::Lock(_) => {}
                    _ => {
                        // `except.printStackTrace()`; see etomo/util/stack_trace.rs.
                        eprintln!("{}", e);
                        println!("\nFor more information run 'man etomo'.");
                    }
                }
                return;
            }
        };
        if !autodoc.is_null() {
            let autodoc: &dyn ReadOnlyAutodoc = unsafe { &*autodoc };
            let mut dash = "-";
            if !unsafe { autodoc.get_attribute(Some(etomo_autodoc::DOUBLE_DASH_ATTRIBUTE_NAME)) }
                .is_null()
            {
                dash = "--";
            }
            // Get this information in order
            let section_location = autodoc.get_section_location();
            if let Some(mut section_location) = section_location {
                let mut section;
                let mut attribute;
                let mut attribute_value;
                loop {
                    section = unsafe { autodoc.next_section(Some(&mut section_location)) };
                    if section.is_null() {
                        break;
                    }
                    let section: &Section = unsafe { &*section };
                    let section_type = ReadOnlySection::get_type(section);
                    if section_type == etomo_autodoc::HEADER_SECTION_NAME {
                        attribute = unsafe {
                            ReadOnlySection::get_attribute(
                                section,
                                Some(etomo_autodoc::USAGE_ATTRIBUTE_NAME),
                            )
                        };
                        if !attribute.is_null() {
                            attribute_value = unsafe { (*attribute).get_value() };
                            if let Some(attribute_value) = attribute_value {
                                // Print section header
                                println!("\n{}", attribute_value);
                            }
                        }
                    } else if section_type == etomo_autodoc::FIELD_SECTION_NAME {
                        // Print parameter
                        print!(
                            "{}{}",
                            dash,
                            ReadOnlySectionList::get_name(section).unwrap_or("null".to_string())
                        );
                        // Look for short parameter name
                        attribute = unsafe {
                            ReadOnlySection::get_attribute(
                                section,
                                Some(etomo_autodoc::SHORT_ATTRIBUTE_NAME),
                            )
                        };
                        if !attribute.is_null() {
                            attribute_value = unsafe { (*attribute).get_value() };
                            if let Some(attribute_value) = attribute_value {
                                print!(" OR {}{}", dash, attribute_value);
                                if ReadOnlySectionList::get_name(section)
                                    == Some("help".to_string())
                                {
                                    if dash == "--" {
                                        print!(" OR -{}", attribute_value);
                                    } else {
                                        print!(" OR --{}", attribute_value);
                                    }
                                }
                            }
                        }
                        // Look for value description
                        attribute = unsafe {
                            ReadOnlySection::get_attribute(
                                section,
                                Some(etomo_autodoc::FORMAT_ATTRIBUTE_NAME),
                            )
                        };
                        attribute_value = if attribute.is_null() {
                            None
                        } else {
                            unsafe { (*attribute).get_value() }
                        };
                        if let Some(attribute_value) = attribute_value {
                            println!(
                                "   {}",
                                Arguments::strip_manpage_formatting(&attribute_value)
                            );
                        } else {
                            attribute = unsafe {
                                ReadOnlySection::get_attribute(
                                    section,
                                    Some(etomo_autodoc::TYPE_ATTRIBUTE_NAME),
                                )
                            };
                            if !attribute.is_null() {
                                attribute_value = unsafe { (*attribute).get_value() };
                                if let Some(attribute_value) = attribute_value {
                                    if attribute_value == etomo_autodoc::BOOLEAN_TYPE {
                                        println!();
                                    }
                                    if attribute_value == etomo_autodoc::FLOAT_TYPE {
                                        println!("   {}", "Floating point");
                                    }
                                    if attribute_value == etomo_autodoc::INTEGER_TYPE {
                                        println!("   {}", "Integer");
                                    }
                                }
                            }
                        }
                        // Look for parameter description
                        attribute = unsafe {
                            ReadOnlySection::get_attribute(
                                section,
                                Some(etomo_autodoc::USAGE_ATTRIBUTE_NAME),
                            )
                        };
                        attribute_value = if attribute.is_null() {
                            None
                        } else {
                            unsafe { (*attribute).get_value() }
                        };
                        if let Some(attribute_value) = attribute_value {
                            println!("     {}", attribute_value);
                        } else {
                            attribute = unsafe {
                                ReadOnlySection::get_attribute(
                                    section,
                                    Some(etomo_autodoc::MANPAGE_ATTRIBUTE_NAME),
                                )
                            };
                            if !attribute.is_null() {
                                attribute_value = unsafe { (*attribute).get_value() };
                                if let Some(attribute_value) = attribute_value {
                                    println!(
                                        "     {}",
                                        Arguments::strip_manpage_formatting(&attribute_value)
                                    );
                                }
                            }
                        }
                    }
                }
                return;
            }
        }
    }
    pub fn strip_manpage_formatting(input: &str) -> String {
        if input.contains("\\f") {
            input
                .replace("\\fB", "")
                .replace("\\fI", "")
                .replace("\\fR", "")
        } else {
            input.to_owned()
        }
    }
    pub fn is_recon_automation(&self) -> bool {
        self.recon_automation
    }
    pub fn is_headless(&self) -> bool {
        self.headless
    }
    pub fn is_debug(&mut self) -> bool {
        if self.debug_always_on && !self.debug {
            self.start_debug();
        }
        if self.debug && !self.debug_level.is_on() {
            self.start_debug();
        }
        self.debug
    }
    pub fn reset_debug(&mut self) {
        self.debug = false;
        self.debug_level = DebugLevel::Off;
    }
    pub fn start_debug(&mut self) {
        if !self.debug_level.is_on() {
            self.debug_level = DebugLevel::Standard;
        }
        self.debug = true;
    }
    pub fn is_self_test(&self) -> bool {
        self.self_test
    }
    pub fn is_demo(&self) -> bool {
        false
    }
    pub fn is_directive(&self) -> bool {
        self.directive
    }
    pub fn get_directive(&self) -> Option<&Path> {
        self.f_directive.as_deref()
    }
    pub fn get_axis(&self) -> Option<AxisType> {
        self.at_axis
    }
    pub fn set_test(&mut self, value: Option<&str>) {
        self.test = value
            .and_then(TestLevel::get_instance)
            .map_or(0, TestLevel::param_value);
        if value.is_none() {
            self.test = TestLevel::Default.param_value();
        }
    }
    pub fn is_test(&self) -> bool {
        self.get_test_level().is_some()
    }
    pub fn get_test_level(&self) -> Option<TestLevel> {
        TestLevel::get_instance_int(self.test)
    }
    pub fn is_no_wait_test(&self) -> bool {
        self.get_test_level() == Some(TestLevel::NoWait)
    }
    pub fn is_stress_test(&self) -> bool {
        self.get_test_level() == Some(TestLevel::Stress)
    }
    pub fn is_fail_test(&self) -> bool {
        self.get_test_level() == Some(TestLevel::Fail)
    }
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
    pub fn is_scan(&self) -> bool {
        self.scan
    }
    pub fn is_create(&self) -> bool {
        self.create
    }
    pub fn is_exit(&self) -> bool {
        self.exit
    }
    pub fn is_from_brt(&self) -> bool {
        self.from_brt
    }
    pub fn is_plugin(&self) -> bool {
        self.plugin
    }
    pub fn is_grab_it(&self) -> bool {
        self.grab_it
    }
    pub fn is_move_b(&self) -> bool {
        self.move_b
    }
    pub fn is_no_load(&self) -> bool {
        self.no_load
    }
    pub fn is_computer_section(&self) -> bool {
        self.computer_section
    }
    pub fn get_computer_section(&self) -> Option<&str> {
        self.computer_section_value.as_deref()
    }
    pub fn is_queue_section(&self) -> bool {
        self.queue_section
    }
    pub fn get_queue_section(&self) -> Option<&str> {
        self.queue_section_value.as_deref()
    }
    pub fn get_image_filename_style(&self) -> Option<ImageFilenameStyle> {
        self.image_filename_style
    }
    pub fn get_grab_it_parameter(&self) -> Option<GrabItParameter> {
        self.grab_it_parameter
    }
    pub fn is_print_names(&self) -> bool {
        self.print_names
    }
    pub fn is_listen(&self) -> bool {
        self.listen
    }
    pub fn is_auto_close_3dmod(&self) -> bool {
        self.auto_close_3dmod
    }
    pub fn is_newstuff(&self) -> bool {
        self.newstuff
    }
    pub fn get_frame(&self) -> Option<ViewType> {
        self.vt_frame
    }
    pub fn get_dataset(&self) -> Option<&str> {
        self.s_raw_image_stack.as_deref()
    }
    pub fn get_raw_image_stack(&self) -> Option<&str> {
        self.s_raw_image_stack.as_deref()
    }
    pub fn get_debug_level(&mut self) -> DebugLevel {
        if self.debug_always_on && !self.debug {
            self.start_debug();
        }
        if self.debug_level.is_on() && !self.debug {
            self.debug = true;
        }
        self.debug_level
    }
    pub fn get_dir(&self) -> Option<&Path> {
        self.f_dir.as_deref()
    }
    pub fn get_fiducial(&self) -> Option<f64> {
        self.en_fiducial
    }
    pub fn is_ignore_loc(&self) -> bool {
        self.ignore_loc
    }
    pub fn is_ignore_settings(&self) -> bool {
        self.ignore_settings
    }
    pub fn is_actions(&self) -> bool {
        self.actions
    }
    pub fn is_cpus(&self) -> bool {
        self.cpus
    }
    pub fn is_gpus(&self) -> bool {
        self.gpus
    }
    pub fn get_user_template_loc(&self) -> Option<&Path> {
        self.user_template_loc.as_deref()
    }
    /// Java `parse(String[])`, preserving accepted one- and two-dash tags.
    pub fn parse(&mut self, args: &[String]) {
        self.test = 0;
        self.self_test = false;
        let mut i = 0;
        while i < args.len() {
            let argument = &args[i];
            if !argument.starts_with('-') {
                self.param_file_name_list.push(argument.clone());
                i += 1;
                continue;
            }
            if !self.param_file_name_list.is_empty() {
                self.error_message_list.push(format!(
                    "WARNING:  unknown argument(s), {:?}, ignored.",
                    self.param_file_name_list
                ));
                self.param_file_name_list.clear();
            }
            let next = args.get(i + 1).map(String::as_str);
            let two_dash_tag = |tag: &str| argument == tag || argument == &format!("-{tag}");
            if HELP_TAGS.contains(&argument.as_str()) {
                self.help = true;
            } else if two_dash_tag(TEST_TAG) {
                if let Some(value) =
                    next.filter(|value| !value.starts_with("--") && self.is_numeric_integer(value))
                {
                    self.set_test(Some(value));
                    i += 1;
                } else {
                    self.test = TestLevel::Default.param_value();
                }
            } else if two_dash_tag(HEADLESS_TAG) {
                self.headless = true;
            } else if two_dash_tag(SELFTEST_TAG) {
                self.self_test = true;
            } else if two_dash_tag(NAMES_TAG) {
                self.print_names = true;
            } else if two_dash_tag(MEMORY_TAG) {
                self.display_memory = true;
                if let Some(value) = next.filter(|value| !value.starts_with("--")) {
                    self.i_display_memory = value.parse::<i32>().unwrap_or(0).abs();
                    i += 1;
                }
            } else if two_dash_tag(DEBUG_TAG) {
                self.debug = true;
                self.debug_level = DebugLevel::Standard;
                if let Some(value) =
                    next.filter(|value| !value.starts_with("--") && self.is_numeric_integer(value))
                {
                    self.debug_level = DebugLevel::get_instance(value);
                    if matches!(self.debug_level, DebugLevel::Off | DebugLevel::Limited) {
                        self.debug = false;
                    }
                    i += 1;
                }
            } else if two_dash_tag(TIMESTAMP_TAG) { /* Java Utilities.setTimestamp(true): JVM logging boundary. */
            } else if two_dash_tag(NEWSTUFF_TAG) {
                self.newstuff = true;
            } else if argument == "-dataset" || two_dash_tag("-rawimagestack") {
                self.raw_image_stack = true;
                self.recon_automation = true;
                if let Some(value) = next {
                    self.s_raw_image_stack = Some(value.to_owned());
                    i += 1;
                }
            } else if two_dash_tag("-dir") {
                self.dir = true;
                self.recon_automation = true;
                if let Some(value) = next {
                    self.f_dir = Some(PathBuf::from(value));
                    i += 1;
                }
            } else if two_dash_tag("-axis") {
                self.axis = true;
                self.recon_automation = true;
                if let Some(value) = next {
                    self.at_axis = AxisType::from_string(value);
                    if self.at_axis.is_some() {
                        i += 1;
                    }
                }
            } else if two_dash_tag("-grabit") {
                self.grab_it = true;
                if let Some(value) = next.filter(|value| !value.starts_with("--")) {
                    self.grab_it_parameter = GrabItParameter::get_instance(Some(value));
                    if self.grab_it_parameter.is_none() {
                        println!("ERROR: unknown grabit parameter");
                    }
                    i += 1;
                }
            } else if two_dash_tag(MOVE_B_TAG) {
                self.move_b = true;
            } else if two_dash_tag("-noload") {
                self.no_load = true;
            } else if two_dash_tag(IMAGE_FILENAME_STYLE_TAG) {
                if let Some(value) = next.filter(|value| !value.starts_with("--")) {
                    self.image_filename_style = ImageFilenameStyle::get_instance(value, true);
                    i += 1;
                }
            } else if two_dash_tag("-frame") {
                self.frame = true;
                self.recon_automation = true;
                if let Some(value) = next {
                    self.vt_frame = ViewType::from_string(value);
                    if self.vt_frame.is_some() {
                        i += 1;
                    }
                }
            } else if two_dash_tag("-fiducial") {
                self.fiducial = true;
                self.recon_automation = true;
                if let Some(value) = next {
                    match value.parse::<f64>() {
                        Ok(value) => {
                            self.en_fiducial = Some(value);
                            self.fiducial_invalid_reason.clear();
                            i += 1;
                        }
                        Err(error) => self.fiducial_invalid_reason = error.to_string(),
                    }
                }
            } else if two_dash_tag("-scan") {
                self.recon_automation = true;
                self.scan = true;
            } else if two_dash_tag("-create") {
                self.recon_automation = true;
                self.create = true;
            } else if two_dash_tag("-exit") {
                self.exit = true;
            } else if two_dash_tag("-fg") {
                self.recon_automation = true;
            } else if two_dash_tag("-listen") {
                self.listen = true;
            } else if two_dash_tag("-autoclose3dmod") {
                self.auto_close_3dmod = true;
            } else if two_dash_tag("-ignoreloc") {
                self.ignore_loc = true;
            } else if two_dash_tag("-ignoresettings") {
                self.ignore_settings = true;
            } else if two_dash_tag("-actions") {
                self.actions = true;
            } else if two_dash_tag(DIRECTIVE_TAG) {
                self.directive = true;
                self.recon_automation = true;
                self.headless = true;
                self.create = true;
                self.exit = true;
                if let Some(value) = next {
                    self.f_directive = Some(PathBuf::from(value));
                    i += 1;
                }
            } else if two_dash_tag("-cpus") {
                self.cpus = true;
                if next.is_some_and(|value| !value.starts_with("--")) {
                    i += 1;
                }
            } else if two_dash_tag("-gpus") {
                self.gpus = true;
                if next.is_some_and(|value| !value.starts_with("--")) {
                    i += 1;
                }
            } else if two_dash_tag("-computersection") {
                self.computer_section = true;
                self.computer_section_value = next
                    .filter(|value| !value.starts_with('-'))
                    .map(str::to_owned);
                if self.computer_section_value.is_some() {
                    i += 1;
                }
            } else if two_dash_tag("-queuesection") {
                self.queue_section = true;
                self.queue_section_value = next
                    .filter(|value| !value.starts_with('-'))
                    .map(str::to_owned);
                if self.queue_section_value.is_some() {
                    i += 1;
                }
            } else if two_dash_tag("-fromBRT") {
                self.from_brt = true;
            } else if two_dash_tag(USER_TEMPLATE_LOC_TAG) {
                if let Some(value) = next {
                    self.user_template_loc = Some(PathBuf::from(value));
                    i += 1;
                }
            } else if two_dash_tag("-plugin") {
                self.plugin = true;
            } else {
                self.error_message_list.push(format!("WARNING:  unknown argument, {argument}, ignored. Please run 'etomo --help' for further assistance."));
            }
            i += 1;
        }
    }
    pub fn is_numeric_integer(&self, input: &str) -> bool {
        input.parse::<i32>().is_ok()
    }
    /// Java `validate(UIComponent)`, with Java dialog presentation retained as a JVM GUI boundary.
    pub fn validate(&mut self) -> bool {
        for string in &self.param_file_name_list {
            let file = Path::new(string);
            if !file.exists() {
                self.error_message_list.push(format!(
                    "File parameter, {}, does not exist.,string:{string}",
                    file.canonicalize()
                        .unwrap_or_else(|_| file.to_path_buf())
                        .display()
                ));
            } else {
                if !file.is_file() {
                    self.error_message_list.push(format!(
                        "File parameter, {}, is a directory.",
                        file.canonicalize()
                            .unwrap_or_else(|_| file.to_path_buf())
                            .display()
                    ));
                }
                let file_filter = DataFileFilter::new();
                if !file_filter.accept(file) {
                    self.error_message_list.push(format!(
                        "File parameter, {}, is not a {}.",
                        file.canonicalize()
                            .unwrap_or_else(|_| file.to_path_buf())
                            .display(),
                        file_filter.get_description()
                    ));
                }
            }
        }
        if self.directive {
            match &self.f_directive {
                None => self
                    .error_message_list
                    .push(format!("Missing {DIRECTIVE_TAG} parameter value.")),
                Some(file) => {
                    if file.extension().is_none_or(|extension| extension != "adoc") {
                        self.error_message_list.push(format!(
                            "{DIRECTIVE_TAG} parameter value, {}, has the wrong extension.",
                            file.canonicalize()
                                .unwrap_or_else(|_| file.to_path_buf())
                                .display()
                        ));
                    }
                    if !file.exists() {
                        self.error_message_list.push(format!(
                            "{DIRECTIVE_TAG} parameter value, {}, does not exist.",
                            file.canonicalize()
                                .unwrap_or_else(|_| file.to_path_buf())
                                .display()
                        ));
                    }
                }
            };
            if self.axis || self.raw_image_stack || self.dir || self.fiducial || self.frame {
                self.warning_message_list.push("The parameters: -axis, -rawimagestack, -dir, -fiducial, and -frame are ignored when -directive is used.".to_owned());
            }
        }
        if self.axis && self.at_axis.is_none() {
            self.error_message_list
                .push("Missing or invalid -axis parameter value.".to_owned());
        }
        if self.raw_image_stack && self.s_raw_image_stack.is_none() {
            self.error_message_list
                .push("Missing -rawimagestack parameter value.".to_owned());
        }
        if self.dir {
            match &self.f_dir {
                None => self
                    .error_message_list
                    .push("Missing -dir parameter value.".to_owned()),
                Some(file) if !file.exists() => self.error_message_list.push(format!(
                    "-dir parameter value, {}, does not exist.",
                    file.canonicalize()
                        .unwrap_or_else(|_| file.to_path_buf())
                        .display()
                )),
                Some(file) if !file.is_dir() => self.error_message_list.push(format!(
                    "-dir parameter value, {}, is not a directory.",
                    file.canonicalize()
                        .unwrap_or_else(|_| file.to_path_buf())
                        .display()
                )),
                _ => {}
            }
        }
        if self.fiducial && self.en_fiducial.is_none() {
            if self.fiducial_invalid_reason.is_empty() {
                self.error_message_list
                    .push("Missing -fiducial parameter value.".to_owned());
            } else {
                self.error_message_list.push(format!(
                    "-fiducial parameter value is an invalid number: {}",
                    self.fiducial_invalid_reason
                ));
            }
        } else if self.en_fiducial.is_some_and(|value| value < 0.) {
            self.error_message_list.push(format!(
                "-fiducial parameter value, {}, cannot be negative.",
                self.en_fiducial.unwrap()
            ));
        }
        if self.frame && self.vt_frame.is_none() {
            self.error_message_list
                .push("Missing or invalid -frame parameter value.".to_owned());
        }
        self.error_message_list.is_empty()
    }
    pub fn get_param_file_name_list(&self) -> &[String] {
        &self.param_file_name_list
    }
    pub fn is_help(&self) -> bool {
        self.help
    }
    pub fn get_display_memory_interval(&self) -> i32 {
        self.i_display_memory
    }
    pub fn is_display_memory(&self) -> bool {
        self.display_memory
    }
}

#[cfg(test)]
mod tests {
    use super::{Arguments, TestLevel};
    #[test]
    fn parses_batch_relevant_directive() {
        let mut arguments = Arguments::new();
        arguments.parse(&[
            "-directive".into(),
            "run.adoc".into(),
            "-cpus".into(),
            "parallel".into(),
        ]);
        assert!(arguments.is_directive());
        assert!(arguments.is_headless());
        assert!(arguments.is_cpus());
        assert!(arguments.is_create());
    }
    #[test]
    fn parses_optional_test_level() {
        let mut arguments = Arguments::new();
        arguments.parse(&["--test".into(), "3".into()]);
        assert_eq!(arguments.get_test_level(), Some(TestLevel::Stress));
    }
}
