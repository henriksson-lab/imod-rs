//! `IMOD/Etomo/src/etomo/ui/swing/NewstackDisplay.java`.
#![allow(dead_code)]

use std::io;

use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

/// The checked exceptions declared by Java `NewstackDisplay.getParameters`.
#[derive(Debug)]
pub enum NewstackDisplayException {
    /// Java `FortranInputSyntaxException`.
    FortranInputSyntaxException(FortranInputSyntaxException),
    /// Java `etomo.util.InvalidParameterException`.
    InvalidParameterException(InvalidParameterException),
    /// Java `IOException`.
    Io(io::Error),
}

impl std::fmt::Display for NewstackDisplayException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FortranInputSyntaxException(exception) => exception.fmt(formatter),
            Self::InvalidParameterException(exception) => exception.fmt(formatter),
            Self::Io(exception) => exception.fmt(formatter),
        }
    }
}

impl std::error::Error for NewstackDisplayException {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FortranInputSyntaxException(exception) => Some(exception),
            Self::InvalidParameterException(exception) => Some(exception),
            Self::Io(exception) => Some(exception),
        }
    }
}

/// Java `NewstackDisplay`.
pub trait NewstackDisplay {
    /// Java `NewstParam`, retained at its original com-script boundary.
    type NewstParam;

    /// Java `getParameters(NewstParam, boolean)`.
    fn get_parameters(
        &self,
        newst_param: &mut Self::NewstParam,
        do_validation: bool,
    ) -> Result<bool, NewstackDisplayException>;

    /// Java `setParameters(ConstNewstParam)`.
    fn set_parameters(&mut self, param: &Self::NewstParam);

    /// Java `validate()`.
    fn validate(&self) -> bool;

    /// Java `isFiducialess()`.
    fn is_fiducialess(&self) -> bool;
}

impl NewstackDisplay for super::newstack_panel::NewstackPanel {
    type NewstParam = super::newstack_or_blendmont_panel::NewstParam;

    fn get_parameters(
        &self,
        newst_param: &mut Self::NewstParam,
        do_validation: bool,
    ) -> Result<bool, NewstackDisplayException> {
        Ok(self
            .newstack_or_blendmont_panel
            .get_newst_parameters(newst_param, do_validation))
    }

    fn set_parameters(&mut self, param: &Self::NewstParam) {
        self.newstack_or_blendmont_panel.set_newst_parameters(param);
    }

    fn validate(&self) -> bool {
        self.newstack_or_blendmont_panel.validate()
    }

    fn is_fiducialess(&self) -> bool {
        self.newstack_or_blendmont_panel.is_fiducialess()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Display {
        parameter: String,
        fiducialess: bool,
    }

    impl NewstackDisplay for Display {
        type NewstParam = String;

        fn get_parameters(
            &self,
            newst_param: &mut Self::NewstParam,
            do_validation: bool,
        ) -> Result<bool, NewstackDisplayException> {
            *newst_param = self.parameter.clone();
            Ok(do_validation)
        }

        fn set_parameters(&mut self, param: &Self::NewstParam) {
            self.parameter = param.clone();
        }

        fn validate(&self) -> bool {
            true
        }

        fn is_fiducialess(&self) -> bool {
            self.fiducialess
        }
    }

    #[test]
    fn display_contract_preserves_all_four_java_operations() {
        let mut display = Display {
            parameter: "input".into(),
            fiducialess: true,
        };
        let mut parameter = String::new();
        assert!(display.get_parameters(&mut parameter, true).unwrap());
        assert_eq!(parameter, "input");
        display.set_parameters(&"output".into());
        assert_eq!(display.parameter, "output");
        assert!(display.validate());
        assert!(display.is_fiducialess());
    }

    #[test]
    fn checked_exception_variants_preserve_each_source_error_channel() {
        let syntax = NewstackDisplayException::FortranInputSyntaxException(
            FortranInputSyntaxException::new("syntax"),
        );
        let invalid = NewstackDisplayException::InvalidParameterException(
            InvalidParameterException::new("parameter"),
        );
        let io = NewstackDisplayException::Io(io::Error::other("io"));
        assert_eq!(syntax.to_string(), "syntax");
        assert_eq!(
            invalid.to_string(),
            "etomo.util.InvalidParameterException: parameter"
        );
        assert_eq!(io.to_string(), "io");
    }
}
