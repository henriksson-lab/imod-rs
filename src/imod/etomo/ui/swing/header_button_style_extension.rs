//! `IMOD/Etomo/src/etomo/ui/swing/HeaderButtonStyleExtension.java`.
#![allow(dead_code)]

use std::sync::{Arc, LazyLock, Mutex};

use super::button_style_extension::ButtonStyleExtension;

pub trait HeaderStyleButton {
    fn set_text(&mut self, text: String);
    fn set_focusable(&mut self, focusable: bool);
    fn set_content_area_filled(&mut self, filled: bool);
    fn set_etched_border(&mut self);
}

pub struct HeaderButtonStyleExtension {
    pub button_style_extension: ButtonStyleExtension<&'static str>,
}

static INSTANCE: LazyLock<Mutex<Option<Arc<HeaderButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));

impl HeaderButtonStyleExtension {
    /// Java private `HeaderButtonStyleExtension()`.
    fn new() -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(false, None, None, None, None, false),
        }
    }

    /// Java static `getInstance()`.
    pub fn get_instance() -> Arc<Self> {
        let mut instance = INSTANCE
            .lock()
            .expect("HeaderButtonStyleExtension mutex poisoned");
        if instance.is_none() {
            *instance = Some(Arc::new(Self::new()));
        }
        Arc::clone(instance.as_ref().expect("Java INSTANCE assigned above"))
    }

    /// Java overridden final `setup(AbstractButton, String, boolean)`.
    pub fn setup<B: HeaderStyleButton>(
        &self,
        button: Option<&mut B>,
        label: Option<&str>,
        _debug: bool,
    ) {
        let Some(button) = button else { return };
        if let Some(label) = label {
            button.set_text(format!("{label}: "));
        }
        button.set_focusable(false);
        button.set_content_area_filled(false);
        button.set_etched_border();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Button {
        text: Option<String>,
        focusable: bool,
        filled: bool,
        etched: bool,
    }
    impl HeaderStyleButton for Button {
        fn set_text(&mut self, text: String) {
            self.text = Some(text)
        }
        fn set_focusable(&mut self, value: bool) {
            self.focusable = value
        }
        fn set_content_area_filled(&mut self, value: bool) {
            self.filled = value
        }
        fn set_etched_border(&mut self) {
            self.etched = true
        }
    }
    #[test]
    fn setup_is_noninteractive_header() {
        let mut button = Button {
            focusable: true,
            filled: true,
            ..Default::default()
        };
        HeaderButtonStyleExtension::new().setup(Some(&mut button), Some("Head"), false);
        assert_eq!(button.text, Some("Head: ".to_owned()));
        assert!(!button.focusable);
        assert!(!button.filled);
        assert!(button.etched);
    }
}
