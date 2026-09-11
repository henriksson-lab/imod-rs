//! `IMOD/Etomo/src/etomo/type/DataFileType.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton; Java's identity comparisons (`this == RECON`)
//! become variant matches.
#![allow(dead_code)]

use super::interface_type::InterfaceType;

/// Java `DataFileType`.  Describes the types of data files and rules about directory
/// sharing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataFileType {
    /// Java `RECON`, constructed with extension ".edf" and hasAxisType true.
    Recon,
    /// Java `JOIN`, constructed with extension ".ejf" and hasAxisType false.
    Join,
    /// Java `PARALLEL`, constructed with extension ".epp" and hasAxisType false.
    Parallel,
    /// Java `BATCH_RUN_TOMO`, constructed with extension ".ebt" and hasAxisType false.
    BatchRunTomo,
    /// Java `PEET`, constructed with extension ".epe" and hasAxisType false.
    Peet,
    /// Java `SERIAL_SECTIONS`, constructed with extension ".ess" and hasAxisType false.
    SerialSections,
    /// Java `TOOLS`, constructed with extension null and hasAxisType false.
    Tools,
    /// Java `DIRECTIVE_EDITOR`, constructed with extension ".adoc" and hasAxisType
    /// false.
    DirectiveEditor,
}

impl DataFileType {
    /// Java `rcsid`.
    pub const RCSID: &'static str = "$Id$";

    /// Java public field `extension`.
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

    /// Java public field `hasAxisType`.  HasAxisType is true when it is possible for
    /// the data file type be a dual axis.
    pub fn has_axis_type(self) -> bool {
        match self {
            Self::Recon => true,
            Self::Join => false,
            Self::Parallel => false,
            Self::BatchRunTomo => false,
            Self::Peet => false,
            Self::SerialSections => false,
            Self::Tools => false,
            Self::DirectiveEditor => false,
        }
    }

    /// Java `getInstance`.  Return a DataFileType instance based on the extension of
    /// fileName.  Cannot return the TOOLS instance because it has no extension
    /// associated with it.
    pub fn get_instance(file_name: Option<&str>) -> Option<DataFileType> {
        let file_name = match file_name {
            None => return None,
            Some(file_name) => file_name,
        };
        let mut ext = file_name;
        let ext_index = file_name.rfind('.');
        if let Some(ext_index) = ext_index {
            ext = file_name[ext_index..].trim();
        }
        if ext == Self::Recon.extension().unwrap() {
            return Some(Self::Recon);
        }
        if ext == Self::Join.extension().unwrap() {
            return Some(Self::Join);
        }
        if ext == Self::Parallel.extension().unwrap() {
            return Some(Self::Parallel);
        }
        if ext == Self::BatchRunTomo.extension().unwrap() {
            return Some(Self::BatchRunTomo);
        }
        if ext == Self::Peet.extension().unwrap() {
            return Some(Self::Peet);
        }
        if ext == Self::SerialSections.extension().unwrap() {
            return Some(Self::SerialSections);
        }
        None
    }

    /// Java `getInterfaceType`.
    pub fn get_interface_type(self) -> Option<InterfaceType> {
        if self == Self::Recon {
            return Some(InterfaceType::Recon);
        }
        if self == Self::Join {
            return Some(InterfaceType::Join);
        }
        if self == Self::Parallel {
            return Some(InterfaceType::Pp);
        }
        if self == Self::BatchRunTomo {
            return Some(InterfaceType::BatchRunTomo);
        }
        if self == Self::Peet {
            return Some(InterfaceType::Peet);
        }
        if self == Self::SerialSections {
            return Some(InterfaceType::SerialSections);
        }
        if self == Self::Tools {
            return Some(InterfaceType::Tools);
        }
        None
    }
}

/// Java `toString`.
impl std::fmt::Display for DataFileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Self::Recon {
            return f.write_str("Reconstruction");
        }
        if *self == Self::Join {
            return f.write_str("Join");
        }
        if *self == Self::Parallel {
            return f.write_str("Parallel");
        }
        if *self == Self::BatchRunTomo {
            return f.write_str("Batch Run Tomo");
        }
        if *self == Self::Peet {
            return f.write_str("PEET");
        }
        if *self == Self::SerialSections {
            return f.write_str("Serial Sections");
        }
        if *self == Self::Tools {
            return f.write_str("Tools");
        }
        // Java returns the `extension` field, which is null only for TOOLS - handled
        // above - so DIRECTIVE_EDITOR's ".adoc" is what reaches here.
        f.write_str(self.extension().unwrap_or(""))
    }
}
