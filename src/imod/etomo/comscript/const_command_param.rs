//! `IMOD/Etomo/src/etomo/comscript/ConstCommandParam.java`.

/// Java package-private `ConstCommandParam`.
pub trait ConstCommandParam {
    /// Java `isParseComments`.
    fn is_parse_comments(&self) -> bool;
    /// Java `getProcessNameString`.
    fn get_process_name_string(&self) -> Option<String>;
    /// Java `getCommand`.
    fn get_command(&self) -> Option<String>;
}
