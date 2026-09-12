//! `IMOD/Etomo/src/etomo/ui/swing/Deferred3dmodButton.java`.
//!
//! This interface is the deferred-viewer boundary.  A process-launching Swing
//! control receives it and may invoke the same 3dmod action after its process
//! completes; the actual event-loop and process scheduling remain owned by the
//! concrete button and manager source units.
#![allow(dead_code)]

use crate::imod::etomo::{process::imod_process::Run3dmodMenuOptions, r#type::file_key::FileKey};

/// Java `Deferred3dmodButton`.
///
/// Java reference values may be null.  `FileKey` is therefore returned as an
/// `Option`, while an actual invocation has a concrete
/// `Run3dmodMenuOptions` object just as the Java interface declares.
pub trait Deferred3dmodButton {
    /// Java `action(Run3dmodMenuOptions)`.
    fn action(&mut self, menu_options: Run3dmodMenuOptions);

    /// Java `getOutputImageFileKey()`.
    fn get_output_image_file_key(&self) -> Option<&FileKey>;
}

#[cfg(test)]
mod tests {
    use crate::imod::etomo::{
        process::imod_process::Run3dmodMenuOptions, r#type::file_key::FileKey,
    };

    use super::Deferred3dmodButton;

    struct Button {
        output_image_file_key: Option<FileKey>,
        received_menu_options: Option<Run3dmodMenuOptions>,
    }

    impl Deferred3dmodButton for Button {
        fn action(&mut self, menu_options: Run3dmodMenuOptions) {
            self.received_menu_options = Some(menu_options);
        }

        fn get_output_image_file_key(&self) -> Option<&FileKey> {
            self.output_image_file_key.as_ref()
        }
    }

    #[test]
    fn action_receives_the_menu_options_object() {
        let options = Run3dmodMenuOptions {
            bin_by_2: true,
            allow_binning_in_z: true,
            startup_window: false,
        };
        let mut button = Button {
            output_image_file_key: None,
            received_menu_options: None,
        };

        button.action(options);

        assert_eq!(button.received_menu_options, Some(options));
    }

    #[test]
    fn output_image_file_key_keeps_java_nullability() {
        let mut button = Button {
            output_image_file_key: None,
            received_menu_options: None,
        };
        assert_eq!(button.get_output_image_file_key(), None);

        button.output_image_file_key = Some(FileKey::new(Some("output")));
        assert_eq!(
            button
                .get_output_image_file_key()
                .and_then(FileKey::get_imod_manager_key),
            Some("output")
        );
    }
}
