//! `IMOD/Etomo/src/etomo/ui/swing/Deferred3dmodButton.java`.
//!
//! A button that knows how to open 3dmod on a process's output, so the
//! process (or a process series) can run that 3dmod after it finishes.
//! Handles are `Rc<dyn Deferred3dmodButton>` (EDT objects, `&self` methods).

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::file_key::FileKey;

/// Java `Deferred3dmodButton`.
pub trait Deferred3dmodButton {
    /// Java `action(Run3dmodMenuOptions)`.
    fn action(&self, menu_options: Run3dmodMenuOptions);

    /// Java `getOutputImageFileKey()`.
    fn get_output_image_file_key(&self) -> Option<FileKey>;
}
