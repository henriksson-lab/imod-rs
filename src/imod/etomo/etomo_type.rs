//! Non-GUI type units used by `etomo/Arguments.java`.
#![allow(dead_code)]

/// `etomo.type.AxisType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisType {
    SingleAxis,
    DualAxis,
    NotSet,
}
impl AxisType {
    pub fn get_instance(line: Option<&str>) -> Option<Self> {
        let line = line?;
        if line.contains("Single Axis") {
            Some(Self::SingleAxis)
        } else if line.contains("Dual Axis") {
            Some(Self::DualAxis)
        } else if line.contains("Not Set") {
            Some(Self::NotSet)
        } else {
            None
        }
    }
    pub fn from_string(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Single Axis") || name.eq_ignore_ascii_case("single") {
            Some(Self::SingleAxis)
        } else if name.eq_ignore_ascii_case("Dual Axis") || name.eq_ignore_ascii_case("dual") {
            Some(Self::DualAxis)
        } else if name.eq_ignore_ascii_case("Not Set") {
            Some(Self::NotSet)
        } else {
            None
        }
    }
    pub fn get_value(self) -> &'static str {
        match self {
            Self::SingleAxis => "single",
            Self::DualAxis => "dual",
            Self::NotSet => "",
        }
    }
}
impl std::fmt::Display for AxisType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::SingleAxis => "Single Axis",
            Self::DualAxis => "Dual Axis",
            Self::NotSet => "Not Set",
        })
    }
}

/// `etomo.type.ViewType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewType {
    SingleView,
    Montage,
}
impl ViewType {
    pub fn get_param_value(self) -> &'static str {
        match self {
            Self::SingleView => "single",
            Self::Montage => "montage",
        }
    }
    pub fn get_value(self) -> i32 {
        match self {
            Self::SingleView => 0,
            Self::Montage => 1,
        }
    }
    pub fn is_default(self) -> bool {
        self == Self::SingleView
    }
    pub fn get_label(self) -> Option<&'static str> {
        None
    }
    pub fn from_string(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("Single View") || name.eq_ignore_ascii_case("single") {
            Some(Self::SingleView)
        } else if name.eq_ignore_ascii_case("Montage") || name.eq_ignore_ascii_case("montage") {
            Some(Self::Montage)
        } else {
            None
        }
    }
    pub fn get_instance(value: Self) -> Self {
        value
    }
}
impl std::fmt::Display for ViewType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::SingleView => "Single View",
            Self::Montage => "Montage",
        })
    }
}

/// `etomo.type.DebugLevel`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DebugLevel {
    Off,
    Limited,
    Standard,
    Extra,
    Verbose,
    ExtraVerbose,
}
impl DebugLevel {
    pub fn get_instance(value: &str) -> Self {
        match value.parse::<i32>().ok() {
            Some(0) => Self::Off,
            Some(-1) => Self::Limited,
            Some(1) => Self::Standard,
            Some(2) => Self::Extra,
            Some(3) => Self::Verbose,
            Some(4) => Self::ExtraVerbose,
            _ => Self::Standard,
        }
    }
    pub fn get_off_instance() -> Self {
        Self::Off
    }
    pub fn get_value(self) -> i32 {
        match self {
            Self::Off => 0,
            Self::Limited => -1,
            Self::Standard => 1,
            Self::Extra => 2,
            Self::Verbose => 3,
            Self::ExtraVerbose => 4,
        }
    }
    pub fn is_limited(self) -> bool {
        self == Self::Limited
    }
    pub fn is_on(self) -> bool {
        self != Self::Off && self != Self::Limited
    }
    pub fn is_extra(self) -> bool {
        matches!(self, Self::Extra | Self::Verbose | Self::ExtraVerbose)
    }
    pub fn is_verbose(self) -> bool {
        matches!(self, Self::Verbose | Self::ExtraVerbose)
    }
    pub fn is_extra_verbose(self) -> bool {
        self == Self::ExtraVerbose
    }
    pub fn ge(self, other: Self) -> bool {
        self.get_value() >= other.get_value()
    }
}
impl std::fmt::Display for DebugLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Limited => "limited",
            Self::Standard => "standard",
            Self::Extra => "extra",
            Self::Verbose => "verbose",
            Self::ExtraVerbose => "extraVerbose",
        })
    }
}

/// `etomo.type.ImageFilenameStyle`, including its Java default (`MRC`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageFilenameStyle {
    Old,
    Mrc,
    Hdf,
}
impl ImageFilenameStyle {
    pub fn get_instance(string: &str, allow_default: bool) -> Option<Self> {
        if string.eq_ignore_ascii_case("OLD") {
            Some(Self::Old)
        } else if string.eq_ignore_ascii_case("MRC") {
            Some(Self::Mrc)
        } else if string.eq_ignore_ascii_case("HDF") {
            Some(Self::Hdf)
        } else if allow_default {
            Some(Self::Mrc)
        } else {
            None
        }
    }
    pub fn is_valid_value(value: Option<i32>) -> bool {
        matches!(value, Some(0..=2))
    }
    pub fn get_instance_from_index(index: i32) -> Self {
        match index {
            0 => Self::Old,
            1 => Self::Mrc,
            2 => Self::Hdf,
            _ => Self::Mrc,
        }
    }
    pub fn get_instance_from_property_value(value: &str) -> Option<Self> {
        Self::get_instance(value, false)
    }
    pub fn get_index(self) -> i32 {
        match self {
            Self::Old => 0,
            Self::Mrc => 1,
            Self::Hdf => 2,
        }
    }
    pub fn is_standard(self) -> bool {
        self != Self::Old
    }
    pub fn get_value(self) -> i32 {
        match self {
            Self::Old => 0,
            Self::Mrc => 1,
            Self::Hdf => 2,
        }
    }
    pub fn get_property_value(self) -> &'static str {
        match self {
            Self::Old => "OLD",
            Self::Mrc => "MRC",
            Self::Hdf => "HDF",
        }
    }
    pub fn equals_string(self, value: &str) -> bool {
        value == self.get_value().to_string()
            || value.eq_ignore_ascii_case(self.get_property_value())
            || matches!(
                (self, value),
                (Self::Old, "st") | (Self::Mrc, "mrc") | (Self::Hdf, "hdf")
            )
    }
    pub fn get_default_raw_image_stack_extension(self) -> &'static str {
        match self {
            Self::Old => "st",
            Self::Mrc => "mrc",
            Self::Hdf => "hdf",
        }
    }
}
impl std::fmt::Display for ImageFilenameStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_value())
    }
}

/// `etomo.type.InterfaceType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterfaceType {
    BatchRunTomo,
    Join,
    Peet,
    Pp,
    Recon,
    SerialSections,
    Tools,
    DirectiveEditor,
    FrontPage,
}
impl InterfaceType {
    pub fn get_instance(name: Option<&str>) -> Option<Self> {
        match name {
            Some("recon") => Some(Self::Recon),
            Some("join") => Some(Self::Join),
            Some("pp") => Some(Self::Pp),
            Some("batchRunTomo") => Some(Self::BatchRunTomo),
            Some("peet") => Some(Self::Peet),
            _ => None,
        }
    }
    pub fn equals(self, other: Self) -> bool {
        self == other
    }
}
impl std::fmt::Display for InterfaceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::BatchRunTomo => "batchRunTomo",
            Self::Join => "join",
            Self::Peet => "peet",
            Self::Pp => "pp",
            Self::Recon => "recon",
            Self::SerialSections => "serialSections",
            Self::Tools => "tools",
            Self::DirectiveEditor => "directiveEditor",
            Self::FrontPage => "frontPage",
        })
    }
}

/// `etomo.type.DataFileType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataFileType {
    Recon,
    Join,
    Parallel,
    BatchRunTomo,
    Peet,
    SerialSections,
    Tools,
    DirectiveEditor,
}
impl DataFileType {
    pub fn extension(self) -> Option<&'static str> {
        match self {
            Self::Recon => Some(".edf"),
            Self::Join => Some(".ejf"),
            Self::Parallel => Some(".epp"),
            Self::BatchRunTomo => Some(".ebt"),
            Self::Peet => Some(".epe"),
            Self::SerialSections => Some(".ess"),
            Self::Tools => None,
            Self::DirectiveEditor => Some(".adoc"),
        }
    }
    pub fn has_axis_type(self) -> bool {
        self == Self::Recon
    }
    pub fn get_instance(file_name: Option<&str>) -> Option<Self> {
        let extension = file_name?
            .rsplit_once('.')
            .map_or(file_name?, |(_, extension)| extension)
            .trim();
        [
            Self::Recon,
            Self::Join,
            Self::Parallel,
            Self::BatchRunTomo,
            Self::Peet,
            Self::SerialSections,
        ]
        .into_iter()
        .find(|value| {
            value
                .extension()
                .is_some_and(|suffix| suffix.trim_start_matches('.') == extension)
        })
    }
    pub fn get_interface_type(self) -> Option<InterfaceType> {
        match self {
            Self::Recon => Some(InterfaceType::Recon),
            Self::Join => Some(InterfaceType::Join),
            Self::Parallel => Some(InterfaceType::Pp),
            Self::BatchRunTomo => Some(InterfaceType::BatchRunTomo),
            Self::Peet => Some(InterfaceType::Peet),
            Self::SerialSections => Some(InterfaceType::SerialSections),
            Self::Tools => Some(InterfaceType::Tools),
            Self::DirectiveEditor => None,
        }
    }
}
impl std::fmt::Display for DataFileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Recon => "Reconstruction",
            Self::Join => "Join",
            Self::Parallel => "Parallel",
            Self::BatchRunTomo => "Batch Run Tomo",
            Self::Peet => "PEET",
            Self::SerialSections => "Serial Sections",
            Self::Tools => "Tools",
            Self::DirectiveEditor => self.extension().unwrap_or(""),
        })
    }
}
