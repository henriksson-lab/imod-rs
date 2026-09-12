//! `IMOD/Etomo/src/etomo/ui/swing/ImodButtonStyleExtension.java`.
#![allow(dead_code)]
use super::button_style_extension::{
    ButtonStyleExtension, CompleteIconBoundary, ImageObserverBoundary,
};
use std::sync::{Arc, LazyLock, Mutex};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImodScaledImage {
    Imod,
    ImodPressed,
    ImodRollover,
}
pub type ImodCompleteIconBoundary = CompleteIconBoundary<ImodScaledImage>;
pub struct ImodButtonStyleExtension {
    pub button_style_extension: ButtonStyleExtension<ImodScaledImage>,
}
static INSTANCE: LazyLock<Mutex<Option<Arc<ImodButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));
impl ImodButtonStyleExtension {
    fn new(image_observer: Option<Arc<dyn ImageObserverBoundary>>) -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(
                false,
                Some(ImodCompleteIconBoundary {
                    image: ImodScaledImage::Imod,
                    selected_image: None,
                    pressed_image: Some(ImodScaledImage::ImodPressed),
                    rollover_image: Some(ImodScaledImage::ImodRollover),
                    image_observer,
                    debug: false,
                    icon_size: None,
                }),
                None,
                None,
                None,
                true,
            ),
        }
    }
    pub fn get_instance(image_observer: Option<Arc<dyn ImageObserverBoundary>>) -> Arc<Self> {
        let mut instance = INSTANCE
            .lock()
            .expect("ImodButtonStyleExtension mutex poisoned");
        if instance.is_none() {
            *instance = Some(Arc::new(Self::new(image_observer)));
        }
        Arc::clone(instance.as_ref().expect("Java INSTANCE assigned above"))
    }
}
