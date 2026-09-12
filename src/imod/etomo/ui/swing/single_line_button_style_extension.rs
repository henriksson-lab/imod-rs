//! `IMOD/Etomo/src/etomo/ui/swing/SingleLineButtonStyleExtension.java`.
#![allow(dead_code)]
use super::button_style_extension::ButtonStyleExtension;
use std::sync::{Arc, LazyLock, Mutex};
pub trait SingleLineStyleButton {
    fn set_preferred_size(&mut self, size: (i32, i32));
    fn set_maximum_size(&mut self, size: (i32, i32));
}
pub struct SingleLineButtonStyleExtension {
    pub button_style_extension: ButtonStyleExtension<&'static str>,
}
static INSTANCE: LazyLock<Mutex<Option<Arc<SingleLineButtonStyleExtension>>>> =
    LazyLock::new(|| Mutex::new(None));
impl SingleLineButtonStyleExtension {
    fn new() -> Self {
        Self {
            button_style_extension: ButtonStyleExtension::new(false, None, None, None, None, false),
        }
    }
    pub fn get_instance() -> Arc<Self> {
        let mut instance = INSTANCE
            .lock()
            .expect("SingleLineButtonStyleExtension mutex poisoned");
        if instance.is_none() {
            *instance = Some(Arc::new(Self::new()));
        }
        Arc::clone(instance.as_ref().expect("Java INSTANCE assigned above"))
    }
    /// Java overload `setup(AbstractButton)`.
    pub fn setup<B: SingleLineStyleButton>(&self, button: &mut B, size: (i32, i32)) {
        button.set_preferred_size(size);
        button.set_maximum_size(size);
    }
}
