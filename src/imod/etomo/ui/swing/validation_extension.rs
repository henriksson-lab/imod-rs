//! `IMOD/Etomo/src/etomo/ui/swing/ValidationExtension.java`.
#![allow(dead_code)]

/// Java public final `ValidationExtension`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ValidationExtension {
    location_descr: Option<String>,
    file_only: bool,
    file_must_exist: bool,
    must_be_positive: bool,
    required: bool,
}

impl ValidationExtension {
    /// Java package-private `ValidationExtension()`.
    pub fn new() -> Self {
        Self::default()
    }
    /// Java `setRequired(boolean)`.
    pub fn set_required(&mut self, required: bool) {
        self.required = required;
    }
    /// Java `getLocationAddon()`.
    pub fn get_location_addon(&self) -> String {
        self.location_descr
            .as_ref()
            .map_or_else(String::new, |value| format!(" in {value}"))
    }
    /// Java `setLocationDescr(String)`.
    pub fn set_location_descr(&mut self, location_descr: Option<String>) {
        self.location_descr = location_descr;
    }
    /// Java `setFileOnly(boolean)`.
    pub fn set_file_only(&mut self, file_only: bool) {
        self.file_only = file_only;
    }
    /// Java `setFileMustExist(boolean)`.
    pub fn set_file_must_exist(&mut self, file_must_exist: bool) {
        self.file_must_exist = file_must_exist;
    }
    /// Java `setMustBePositive(boolean)`.
    pub fn set_must_be_positive(&mut self, must_be_positive: bool) {
        self.must_be_positive = must_be_positive;
    }
    /// Java `isRequired()`.
    pub fn is_required(&self) -> bool {
        self.required
    }
    /// Java `isFileOnly()`.
    pub fn is_file_only(&self) -> bool {
        self.file_only
    }
    /// Java `isFileMustExist()`.
    pub fn is_file_must_exist(&self) -> bool {
        self.file_must_exist
    }
    /// Java `isMustBePositive()`.
    pub fn is_must_be_positive(&self) -> bool {
        self.must_be_positive
    }
}
