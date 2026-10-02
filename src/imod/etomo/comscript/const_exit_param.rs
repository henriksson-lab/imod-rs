//! `IMOD/Etomo/src/etomo/comscript/ConstExitParam.java`.
//!
//! Java package-private class `ConstExitParam`, which `ExitParam` extends; translated
//! as a struct `ExitParam` embeds (`ExitParam.base`, reached through `Deref`).

/// Java package-private `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "exit";

/// Java `ConstExitParam`.
#[derive(Clone, Debug)]
pub struct ConstExitParam {
    /// Java field `resultValue`.
    pub result_value: i32,
}

impl Default for ConstExitParam {
    fn default() -> Self {
        ConstExitParam::new()
    }
}

impl ConstExitParam {
    /// Java package-private `ConstExitParam()`.
    pub fn new() -> ConstExitParam {
        let mut param = ConstExitParam { result_value: 0 };
        param.reset();
        param
    }

    /// Java package-private `reset`.
    pub fn reset(&mut self) {
        self.result_value = 0;
    }

    /// Java `getResultValue`.
    pub fn get_result_value(&self) -> i32 {
        self.result_value
    }
}
