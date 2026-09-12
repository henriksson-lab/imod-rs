//! `IMOD/Etomo/src/etomo/LocalArguments.java`.
//!
//! This deliberately remains a distinct type rather than an alias for
//! `Arguments`: Java uses it to identify the local command-line construction
//! path while inheriting the complete `Arguments` state.

#![allow(dead_code)]

use std::path::PathBuf;

use super::arguments::Arguments;

/// Java `LocalArguments extends Arguments`.
#[derive(Clone, Debug, Default)]
pub struct LocalArguments {
    pub arguments: Arguments,
}

impl LocalArguments {
    /// Java `setRawImageStack(String)`.
    pub fn set_raw_image_stack(&mut self, input: impl Into<String>) {
        self.arguments.s_raw_image_stack = Some(input.into());
    }

    /// Java `setDir(String)`.
    pub fn set_dir(&mut self, input: impl Into<PathBuf>) {
        self.arguments.dir = true;
        self.arguments.f_dir = Some(input.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setters_retain_the_inherited_arguments_state() {
        let mut arguments = LocalArguments::default();
        arguments.set_raw_image_stack("sum.mrc");
        arguments.set_dir("frames");
        assert_eq!(arguments.arguments.get_raw_image_stack(), Some("sum.mrc"));
        assert!(arguments.arguments.dir);
        assert_eq!(
            arguments.arguments.get_dir(),
            Some(std::path::Path::new("frames"))
        );
    }
}
