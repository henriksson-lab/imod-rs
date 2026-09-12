//! `IMOD/Etomo/src/etomo/ui/swing/XButtonStyleExtension.java`.
#![allow(dead_code)]
use super::button_style_extension::{ButtonStyleExtension, CompleteIconBoundary};
use std::sync::{Arc, LazyLock, Mutex};
pub type XCompleteIconBoundary = CompleteIconBoundary<&'static str>;
pub struct XButtonStyleExtension {
    pub button_style_extension: ButtonStyleExtension<&'static str>,
}
static INSTANCE: LazyLock<Mutex<Option<Arc<XButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));
impl XButtonStyleExtension {
    fn new() -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(
                false,
                Some(XCompleteIconBoundary {
                    image: "x.png",
                    selected_image: None,
                    pressed_image: Some("x-pressed.png"),
                    rollover_image: None,
                    image_observer: None,
                    debug: false,
                    icon_size: None,
                }),
                Some(XCompleteIconBoundary {
                    image: "xBlue.png",
                    selected_image: None,
                    pressed_image: Some("xBlue-pressed.png"),
                    rollover_image: None,
                    image_observer: None,
                    debug: false,
                    icon_size: None,
                }),
                Some(XCompleteIconBoundary {
                    image: "xRed.png",
                    selected_image: None,
                    pressed_image: Some("xRed-pressed.png"),
                    rollover_image: None,
                    image_observer: None,
                    debug: false,
                    icon_size: None,
                }),
                None,
                false,
            ),
        }
    }
    pub fn get_instance() -> Arc<Self> {
        let mut instance = INSTANCE
            .lock()
            .expect("XButtonStyleExtension mutex poisoned");
        if instance.is_none() {
            *instance = Some(Arc::new(Self::new()));
        }
        Arc::clone(instance.as_ref().expect("Java INSTANCE assigned above"))
    }
}
