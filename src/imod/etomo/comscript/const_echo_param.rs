//! `IMOD/Etomo/src/etomo/comscript/ConstEchoParam.java`.
//!
//! Java package-private class `ConstEchoParam`, which `EchoParam` extends; translated
//! as a struct `EchoParam` embeds (`EchoParam.base`, reached through `Deref`).

/// Java package-private `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "echo";

/// Java `ConstEchoParam`.
#[derive(Clone, Debug)]
pub struct ConstEchoParam {
    /// Java field `string` (a `StringBuffer`).
    pub string: String,
}

impl Default for ConstEchoParam {
    fn default() -> Self {
        ConstEchoParam::new()
    }
}

impl ConstEchoParam {
    /// Java package-private `ConstEchoParam()`.
    pub fn new() -> ConstEchoParam {
        let mut param = ConstEchoParam {
            string: String::new(),
        };
        param.reset();
        param
    }

    /// Java package-private `reset`.
    pub fn reset(&mut self) {
        self.string = String::new();
    }

    /// Java `getString`.
    pub fn get_string(&self) -> String {
        self.string.clone()
    }
}
