//! `IMOD/Etomo/src/etomo/ui/swing/BlendmontDisplay.java`.
//!
//! The Java source is the small contract between a Blendmont parameter owner
//! and its Swing display. Its parameter implementation stays at the original
//! `etomo.comscript` boundary; the associated type avoids inventing a second
//! parameter representation in this interface translation.

#![allow(dead_code)]

use std::io;

use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::invalid_parameter_exception::InvalidParameterException;

/// The checked exceptions declared by Java `BlendmontDisplay.getParameters`.
#[derive(Debug)]
pub enum BlendmontDisplayException {
    /// Java `FortranInputSyntaxException`.
    FortranInputSyntaxException(FortranInputSyntaxException),
    /// Java `InvalidParameterException`.
    InvalidParameterException(InvalidParameterException),
    /// Java `IOException`.
    Io(io::Error),
}

impl std::fmt::Display for BlendmontDisplayException {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FortranInputSyntaxException(exception) => exception.fmt(formatter),
            Self::InvalidParameterException(exception) => exception.fmt(formatter),
            Self::Io(exception) => exception.fmt(formatter),
        }
    }
}

impl std::error::Error for BlendmontDisplayException {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FortranInputSyntaxException(exception) => Some(exception),
            Self::InvalidParameterException(exception) => Some(exception),
            Self::Io(exception) => Some(exception),
        }
    }
}

/// Java `BlendmontDisplay`.
pub trait BlendmontDisplay {
    /// Java `BlendmontParam`, kept at its original com-script boundary.
    type BlendmontParam;

    /// Java `getParameters(BlendmontParam, boolean)`.
    fn get_parameters(
        &self,
        blendmont_param: &mut Self::BlendmontParam,
        do_validation: bool,
    ) -> Result<bool, BlendmontDisplayException>;

    /// Java `setParameters(BlendmontParam)`.
    fn set_parameters(&mut self, param: &Self::BlendmontParam);

    /// Java `validate()`.
    fn validate(&self) -> bool;

    /// Java `isFiducialess()`.
    fn is_fiducialess(&self) -> bool;
}

impl BlendmontDisplay for super::blendmont_panel::BlendmontPanel {
    type BlendmontParam = super::newstack_or_blendmont_panel::BlendmontParam;

    fn get_parameters(
        &self,
        blendmont_param: &mut Self::BlendmontParam,
        do_validation: bool,
    ) -> Result<bool, BlendmontDisplayException> {
        Ok(self
            .newstack_or_blendmont_panel
            .get_blendmont_parameters(blendmont_param, do_validation))
    }

    fn set_parameters(&mut self, param: &Self::BlendmontParam) {
        self.newstack_or_blendmont_panel
            .set_blendmont_parameters(param);
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
    use std::io;

    use super::*;
    use crate::imod::etomo::r#type::axis_id::AxisID;
    use crate::imod::etomo::r#type::dialog_type::DialogType;
    use crate::imod::etomo::ui::swing::blendmont_panel::BlendmontPanel;
    use crate::imod::etomo::ui::swing::newstack_or_blendmont_panel::{
        BlendmontParam, GlobalExpandButton,
    };

    struct Display {
        parameter: String,
        fiducialess: bool,
    }

    impl BlendmontDisplay for Display {
        type BlendmontParam = String;

        fn get_parameters(
            &self,
            blendmont_param: &mut Self::BlendmontParam,
            do_validation: bool,
        ) -> Result<bool, BlendmontDisplayException> {
            *blendmont_param = self.parameter.clone();
            Ok(do_validation)
        }

        fn set_parameters(&mut self, param: &Self::BlendmontParam) {
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
    fn blendmont_panel_implements_the_source_display_contract() {
        let mut panel = BlendmontPanel::get_instance(
            AxisID::First,
            DialogType::FinalAlignedStack,
            &GlobalExpandButton::get_instance("Advanced", "Basic"),
        );
        let input = BlendmontParam {
            fiducialess: true,
            ..Default::default()
        };
        // `BlendmontParam` itself does not carry this UI setting through Java
        // `NewstackAndBlendmontParamPanel.setParameters`; it is set from the
        // corresponding metadata path.
        panel
            .newstack_or_blendmont_panel
            .set_fiducialess_alignment(true);
        let display: &mut dyn BlendmontDisplay<BlendmontParam = BlendmontParam> = &mut panel;
        display.set_parameters(&input);
        let mut output = BlendmontParam::default();
        assert!(display.get_parameters(&mut output, true).unwrap());
        assert!(display.validate());
        assert!(display.is_fiducialess());
    }

    #[test]
    fn checked_exception_variants_preserve_each_source_error_channel() {
        let syntax = BlendmontDisplayException::FortranInputSyntaxException(
            FortranInputSyntaxException::new("syntax"),
        );
        let invalid = BlendmontDisplayException::InvalidParameterException(
            InvalidParameterException::new(Some("parameter")),
        );
        let io = BlendmontDisplayException::Io(io::Error::other("io"));
        assert_eq!(syntax.to_string(), "syntax");
        assert_eq!(
            invalid.to_string(),
            "etomo.comscript.InvalidParameterException: parameter"
        );
        assert_eq!(io.to_string(), "io");
    }
}
