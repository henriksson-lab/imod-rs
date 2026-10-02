//! `IMOD/Etomo/src/etomo/comscript/ConstMatchshiftsParam.java`.
//!
//! A package-private Java class with state, so a struct here; `MatchshiftsParam`
//! holds it as its `base` through `Deref`/`DerefMut`.

/// Java private static `command`.
const COMMAND: &str = "matchshifts";

/// Java `ConstMatchshiftsParam`.
#[derive(Clone, Debug)]
pub struct ConstMatchshiftsParam {
    /// Java `rootName1`.
    pub(crate) root_name1: Option<String>,
    /// Java `rootName2`.
    pub(crate) root_name2: Option<String>,
    /// Java `xDim`.
    pub(crate) x_dim: i32,
    /// Java `yDim`.
    pub(crate) y_dim: i32,
    /// Java `zDim`.
    pub(crate) z_dim: i32,
    /// Java `xfIn`.
    pub(crate) xf_in: Option<String>,
    /// Java `xfOut`.
    pub(crate) xf_out: Option<String>,
}

impl ConstMatchshiftsParam {
    /// Java `ConstMatchshiftsParam()`.
    pub(crate) fn new() -> ConstMatchshiftsParam {
        let mut instance = ConstMatchshiftsParam {
            root_name1: None,
            root_name2: None,
            x_dim: 0,
            y_dim: 0,
            z_dim: 0,
            xf_in: None,
            xf_out: None,
        };
        instance.reset();
        instance
    }

    /// Java package-private `reset`.
    pub(crate) fn reset(&mut self) {
        self.root_name1 = Some(String::new());
        self.root_name2 = Some(String::new());
        self.x_dim = i32::MIN;
        self.y_dim = i32::MIN;
        self.z_dim = i32::MIN;
        self.xf_in = Some(String::new());
        self.xf_out = Some(String::new());
    }

    /// Java package-private `getCommand`.
    pub(crate) fn get_command(&self) -> String {
        COMMAND.to_owned()
    }
}
