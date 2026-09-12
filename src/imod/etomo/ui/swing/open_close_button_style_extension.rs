//! `IMOD/Etomo/src/etomo/ui/swing/OpenCloseButtonStyleExtension.java`.
#![allow(dead_code)]
use super::button_style_extension::{
    ButtonStyleButton, ButtonStyleExtension, CompleteIconBoundary,
};
use std::sync::{Arc, LazyLock, Mutex};
pub type OpenCloseCompleteIconBoundary = CompleteIconBoundary<&'static str>;
pub struct OpenCloseButtonStyleExtension {
    pub button_style_extension: ButtonStyleExtension<&'static str>,
}
static INSTANCE: LazyLock<Mutex<Option<Arc<OpenCloseButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));
impl OpenCloseButtonStyleExtension {
    fn new() -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(
                true,
                Some(OpenCloseCompleteIconBoundary {
                    image: "openSmall.png",
                    selected_image: Some("closeSmall.png"),
                    pressed_image: None,
                    rollover_image: None,
                    image_observer: None,
                    debug: false,
                    icon_size: None,
                }),
                None,
                Some(OpenCloseCompleteIconBoundary {
                    image: "openRedSmall.png",
                    selected_image: Some("closeRedSmall.png"),
                    pressed_image: None,
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
            .expect("OpenCloseButtonStyleExtension mutex poisoned");
        if instance.is_none() {
            *instance = Some(Arc::new(Self::new()));
        }
        Arc::clone(instance.as_ref().expect("Java INSTANCE assigned above"))
    }
    /// Java overridden `setup(AbstractButton, String, boolean)`.
    pub fn setup<B: ButtonStyleButton<&'static str>>(
        &mut self,
        button: Option<&mut B>,
        label: Option<&str>,
        debug: bool,
    ) {
        self.button_style_extension.setup(button, label, debug);
    }
}
