//! `IMOD/Etomo/src/etomo/comscript/ConstGotoParam.java`.
//!
//! Java package-private class `ConstGotoParam`, which `GotoParam` extends; translated
//! as a struct `GotoParam` embeds (`GotoParam.base`, reached through `Deref`).

/// Java package-private `DELIMITER`.
pub const DELIMITER: char = ':';
/// Java package-private `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "goto";

/// Java `ConstGotoParam`.
#[derive(Clone, Debug)]
pub struct ConstGotoParam {
    /// Java field `label`, initialised to "".
    pub label: Option<String>,
}

impl Default for ConstGotoParam {
    fn default() -> Self {
        ConstGotoParam::new()
    }
}

impl ConstGotoParam {
    /// Java package-private `ConstGotoParam()`.
    pub fn new() -> ConstGotoParam {
        let mut param = ConstGotoParam {
            label: Some(String::new()),
        };
        param.reset();
        param
    }

    /// Java package-private `reset`.
    pub fn reset(&mut self) {
        self.label = Some(String::new());
    }

    /// Java package-private `getLabel`.
    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}
