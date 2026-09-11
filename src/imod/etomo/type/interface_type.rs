//! `IMOD/Etomo/src/etomo/type/InterfaceType.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final` singleton.
#![allow(dead_code)]

/// Java `InterfaceType`.  Represents each etomo interface.  String parameter is used
/// in cpu.adoc.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterfaceType {
    /// Java `BATCH_RUN_TOMO`.
    BatchRunTomo,
    /// Java `JOIN`.
    Join,
    /// Java `PEET`.
    Peet,
    /// Java `PP`.
    Pp,
    /// Java `RECON`.
    Recon,
    /// Java `SERIAL_SECTIONS`.
    SerialSections,
    /// Java `TOOLS`.
    Tools,
    /// Java `DIRECTIVE_EDITOR`.
    DirectiveEditor,
    /// Java `FRONT_PAGE`.  Not used in cpu.adoc.
    FrontPage,
}

impl InterfaceType {
    /// Java field `name`, set by the private `InterfaceType(String)` constructor.
    fn name(self) -> &'static str {
        match self {
            Self::BatchRunTomo => "batchRunTomo",
            Self::Join => "join",
            Self::Peet => "peet",
            Self::Pp => "pp",
            Self::Recon => "recon",
            Self::SerialSections => "serialSections",
            Self::Tools => "tools",
            Self::DirectiveEditor => "directiveEditor",
            Self::FrontPage => "frontPage",
        }
    }

    /// Java `getInstance`.
    pub fn get_instance(name: Option<&str>) -> Option<InterfaceType> {
        let name = match name {
            None => return None,
            Some(name) => name,
        };
        if name == Self::Recon.name() {
            return Some(Self::Recon);
        }
        if name == Self::Join.name() {
            return Some(Self::Join);
        }
        if name == Self::Pp.name() {
            return Some(Self::Pp);
        }
        if name == Self::BatchRunTomo.name() {
            return Some(Self::BatchRunTomo);
        }
        if name == Self::Peet.name() {
            return Some(Self::Peet);
        }
        None
    }

    /// Java `equals`.
    pub fn equals(self, interface_type: InterfaceType) -> bool {
        self == interface_type
    }
}

/// Java `toString`.
impl std::fmt::Display for InterfaceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
