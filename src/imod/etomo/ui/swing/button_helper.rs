//! `IMOD/Etomo/src/etomo/ui/swing/ButtonHelper.java`.
#![allow(dead_code)]

/// Native boundary for `AbstractButton` sizing.
pub trait ButtonHelperButton {
    fn set_preferred_size(&mut self, size: (i32, i32));
    fn set_maximum_size(&mut self, size: (i32, i32));
    fn set_minimum_size(&mut self, size: (i32, i32));
}
/// Java package-private final `ButtonHelper`.
pub struct ButtonHelper;
impl ButtonHelper {
    pub const RCSID: &str = "$$Id$$";
    /// Java `MULTI_LINE_BUTTON_DIM`; supplied by `UIParameters` at the GUI boundary.
    pub const MULTI_LINE_BUTTON_DIM: (i32, i32) = (0, 0);
    /// Java static `format(String)`.
    pub fn format(text: Option<&str>) -> Option<String> {
        let text = text?;
        if text.to_ascii_lowercase().starts_with("<html>") {
            Some(text.to_owned())
        } else {
            Some(format!("<html><b>{text}</b>"))
        }
    }
    /// Java static `setStandardSize(AbstractButton)`.
    pub fn set_standard_size<B: ButtonHelperButton>(button: &mut B, dimension: (i32, i32)) {
        button.set_preferred_size(dimension);
        button.set_maximum_size(dimension);
        button.set_minimum_size(dimension);
    }
    /// Java static `printDefaultUIResource(String)`; UI defaults enumeration is native-GUI owned.
    pub fn print_default_ui_resource(
        type_name: Option<&str>,
        defaults: &[(String, String)],
    ) -> Vec<String> {
        match type_name {
            None => Self::print_default_ui_resources(defaults),
            Some(type_name) => defaults
                .iter()
                .filter(|(key, _)| {
                    key.split('.')
                        .any(|segment| segment.eq_ignore_ascii_case(type_name))
                })
                .map(|(key, value)| format!("{key}={value}"))
                .collect(),
        }
    }
    /// Java static `printDefaultUIResource()`.
    pub fn print_default_ui_resources(defaults: &[(String, String)]) -> Vec<String> {
        defaults
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn format_matches_java() {
        assert_eq!(ButtonHelper::format(None), None);
        assert_eq!(
            ButtonHelper::format(Some("<HTML>x")),
            Some("<HTML>x".to_owned())
        );
        assert_eq!(
            ButtonHelper::format(Some("x")),
            Some("<html><b>x</b>".to_owned())
        );
    }
}
