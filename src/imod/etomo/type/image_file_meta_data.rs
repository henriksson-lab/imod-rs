//! `IMOD/Etomo/src/etomo/type/ImageFileMetaData.java`.
//!
//! Protects image file settings.  Does not have its own prepend.  (Actually "ImageFile"
//! is basically its prepend, but there's only one property being saved so it doesn't
//! need special `createPrepend` functionality.)
//!
//! Java `Properties` is modelled by a deterministic `BTreeMap<String, String>`, as
//! `etomo/storage/storable.rs` does.
#![allow(dead_code)]

use super::image_filename_style::{ImageFilenameStyle, ImageFilenameStyleException};
use super::image_output_format::ImageOutputFormat;
use super::imod_output_format::{self, ImodOutputFormat};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::util::environment_variable;
use crate::imod::etomo::util::stack_trace::StackTrace;
use crate::imod::etomo::util::utilities::create_property_key;
use std::collections::BTreeMap;

/// Java `IMAGE_FILENAME_STYLE_KEY`.
const IMAGE_FILENAME_STYLE_KEY: &str = "ImageFile.ImageFilenameStyle";

/// Java `ImageFileMetaData`.
pub struct ImageFileMetaData {
    /// Java field `envImodOutputFormat`.
    env_imod_output_format: Option<ImodOutputFormat>,
    /// Java field `newDataset`.
    new_dataset: bool,
    /// Java field `imageFilenameStyle`.  Set from command line and environment when the
    /// dataset is new.  After that should always come from the dataset.  When null the
    /// default (`ImageFilenameStyle.OLD`) is returned.
    image_filename_style: Option<ImageFilenameStyle>,
}

impl ImageFileMetaData {
    /// Java package-private
    /// `ImageFileMetaData(boolean, ImageFilenameStyle, boolean)`.  Sets
    /// `imageFilenameStyle` for a new dataset.  For an existing dataset this value will
    /// be overridden in the `load()` function.
    pub fn new(
        force_old_style: bool,
        override_image_filename_style: Option<ImageFilenameStyle>,
        new_dataset: bool,
    ) -> ImageFileMetaData {
        let env_imod_output_format = ImageFileMetaData::get_env_imod_output_format();
        let image_filename_style;
        // Set imageFilenameStyle
        // Set imageFilenameStyle to the new dataset value.
        if !force_old_style && override_image_filename_style.is_none() {
            // Parameter overrides the environment variables.
            let mut style = etomo_director::ARGUMENTS
                .lock()
                .unwrap()
                .get_image_filename_style();
            if style.is_none() {
                // Set from environment variable(s).
                // ETOMO_NAMING_STYLE overrides IMOD_OUTPUT_FORMAT for image file name
                // style.  ETOMO_NAMING_STYLE allows OLD style to be used with HDF output
                // format.
                style = ImageFileMetaData::get_env_image_filename_style();
            }
            if style.is_none() {
                // Derive imageFilenameStyle from imageOutputFormat or use the default
                // imageFilenameStyle.
                style = Some(ImageFilenameStyle::get_instance_from_imod_output_format(
                    env_imod_output_format,
                ));
            }
            image_filename_style = style;
        } else if override_image_filename_style.is_some() {
            // Use the override image filename style parameter. This is most likely from
            // the test manager.
            image_filename_style = override_image_filename_style;
        } else {
            // ForceOnlyStyle: Some managers such as PEET are old style only.
            image_filename_style = Some(ImageFilenameStyle::Old);
        }
        ImageFileMetaData {
            env_imod_output_format,
            new_dataset,
            image_filename_style,
        }
    }

    /// Java `getTempInstance`.
    pub fn get_temp_instance() -> ImageFileMetaData {
        ImageFileMetaData::new(false, None, false)
    }

    /// Java package-private `load(BaseManager, Properties, String)`.
    pub fn load(
        &mut self,
        manager: Option<&'static dyn BaseManager>,
        props: &BTreeMap<String, String>,
        parent_prepend: Option<&str>,
    ) {
        if self.new_dataset {
            // New dataset - use the settings from the constructor.
            if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
                eprintln!(
                    "INFO:Dataset in {} is new.",
                    match manager {
                        None => "null".to_string(),
                        Some(manager) => manager
                            .get_property_user_dir()
                            .unwrap_or("null".to_string()),
                    }
                );
            }
        } else {
            // reset
            self.image_filename_style = None;
            // load
            self.image_filename_style =
                ImageFilenameStyle::load(props, parent_prepend, Some(IMAGE_FILENAME_STYLE_KEY));
            if self.image_filename_style.is_none() {
                eprintln!(
                    "\nINFO: The {} property is missing from the dataset file.  Unless it is\nrepaired, old-style names will be assumed.  If the repair fails, then set this property to the correct\nvalue ({} or {}).  Further messages will show whether or not the correction succeeded.\n",
                    self.get_image_filename_style_key(parent_prepend)
                        .unwrap_or("null".to_string()),
                    ImageFilenameStyle::Mrc.get_property_value(),
                    ImageFilenameStyle::Hdf.get_property_value()
                );
            }
        }
    }

    /// Java package-private `wasImageFilenameStyleLoaded`.  Returns the real state of
    /// `imageFilenameStyle`.
    pub fn was_image_filename_style_loaded(&self) -> bool {
        self.image_filename_style.is_some()
    }

    /// Java package-private `getImageFilenameStyleKey`.
    pub fn get_image_filename_style_key(&self, parent_prepend: Option<&str>) -> Option<String> {
        create_property_key(parent_prepend, Some(IMAGE_FILENAME_STYLE_KEY))
    }

    /// Java package-private `correctImageFilenameStyle`.
    ///
    /// WARNING: changing `imageFilenameStyle` can make a dataset incompatible with its
    /// files.  Only use this function to fix this incompatibility.  The function will
    /// not allow `imageFilenameStyle` to change unless it is null, and will not allow
    /// `imageFilenameStyle` to be set to null.  [Bug# 2386, Bug# 2403]
    pub fn correct_image_filename_style(
        &mut self,
        parent_prepend: Option<&str>,
        new_image_filename_style: Option<ImageFilenameStyle>,
    ) -> bool {
        let key = self
            .get_image_filename_style_key(parent_prepend)
            .unwrap_or("null".to_string());
        if new_image_filename_style == self.image_filename_style {
            if let Some(new_image_filename_style) = new_image_filename_style {
                eprintln!(
                    "\nINFO: {} already equals {}.  No correction necessary.\n",
                    key,
                    new_image_filename_style.get_property_value()
                );
                return true;
            } else {
                eprintln!(
                    "\nWARNING: Unable to correct {} with a null value.  No correction made.\n",
                    key
                );
                return false;
            }
        }
        let mut warning: Option<String> = None;
        if new_image_filename_style.is_none() {
            // Bug# 2403
            warning = Some(format!(
                "\nWARNING: Attempt to delete {} (currently set to {}) blocked.  [Bug# 2386, Bug# 2403]\n",
                key,
                self.image_filename_style.unwrap().get_property_value()
            ));
        } else if let Some(image_filename_style) = self.image_filename_style {
            // Bug# 2403
            warning = Some(format!(
                "\nWARNING: Attempt to change {} from {} to {} blocked.  [Bug# 2386, Bug# 2403]\n",
                key,
                image_filename_style.get_property_value(),
                new_image_filename_style.unwrap().get_property_value()
            ));
        }
        let warning = match warning {
            None => {
                self.image_filename_style = new_image_filename_style;
                // Bug# 2403
                eprintln!(
                    "\nINFO: Corrected empty {}property.  Value was set to {}.  [Bug# 2386, Bug# 2403]\n",
                    key,
                    self.image_filename_style.unwrap().get_property_value()
                );
                return true;
            }
            Some(warning) => warning,
        };
        println!("{}", warning);
        eprintln!("{}", warning);
        // `new ImageFilenameStyle.ImageFilenameStyleException(warning).printStackTrace()`
        // - the exception is constructed only to print its own stack trace.
        let exception = ImageFilenameStyleException::new(&warning);
        let _ = &exception;
        StackTrace::new_with_thread(None, None).print(Some(&exception.to_string()), true);
        false
    }

    /// Java private static `getEnvImodOutputFormat`.  Retrieves the value of
    /// environment variable `IMOD_OUTPUT_FORMAT`.  Returns null if `IMOD_OUTPUT_FORMAT`
    /// does not exist, "" if it is empty.
    fn get_env_imod_output_format() -> Option<ImodOutputFormat> {
        // TODO 2325 Remember this environment variable and its setting
        if !environment_variable::INSTANCE.exists(None, None, imod_output_format::ENV_VAR, None) {
            return None;
        }
        ImodOutputFormat::get_instance(&environment_variable::INSTANCE.get_value(
            None,
            None,
            imod_output_format::ENV_VAR,
            None,
        ))
    }

    /// Java private static `getEnvImageFilenameStyle`.
    fn get_env_image_filename_style() -> Option<ImageFilenameStyle> {
        if !environment_variable::INSTANCE.exists(None, None, ImageFilenameStyle::ENV_VAR, None) {
            return None;
        }
        ImageFilenameStyle::get_instance(
            &environment_variable::INSTANCE.get_value(
                None,
                None,
                ImageFilenameStyle::ENV_VAR,
                None,
            ),
            true,
        )
    }

    /// Java package-private `store(Properties, String)`.  Store using the prepend that
    /// was passed in - this class does not have its own prepend.
    pub fn store(&self, props: &mut BTreeMap<String, String>, parent_prepend: Option<&str>) {
        // The source dereferences `imageFilenameStyle` without a null check, so a null
        // field throws a NullPointerException here.
        self.image_filename_style.unwrap().store(
            props,
            parent_prepend,
            Some(IMAGE_FILENAME_STYLE_KEY),
        );
    }

    /// Java `getImageFilenameStyle`.  Returns `imageFilenameStyle` or
    /// `ImageFilenameStyle.OLD`.  Never returns null.
    pub fn get_image_filename_style(&self) -> ImageFilenameStyle {
        if let Some(image_filename_style) = self.image_filename_style {
            return image_filename_style;
        }
        ImageFilenameStyle::Old
    }

    /// Java package-private `getImageOutputFormat`.
    pub fn get_image_output_format(&self) -> ImageOutputFormat {
        if self.image_filename_style.is_none()
            || self.image_filename_style == Some(ImageFilenameStyle::Old)
        {
            // In non-standard datasets the ImageOutputFormat is based on the
            // IMOD_OUTPUT_FORMAT environment variable, or is the default.
            return ImageOutputFormat::get_instance_from_imod_output_format(
                self.env_imod_output_format,
            );
        }
        // In standard datasets the ImageOutputFormat is based on the imageFilenameStyle.
        ImageOutputFormat::get_instance_from_image_filename_style(self.image_filename_style)
    }

    /// Java `getDefaultRawImageStackExtension`.  Does not return null.
    pub fn get_default_raw_image_stack_extension(&self) -> &'static Extension {
        if let Some(image_filename_style) = self.image_filename_style {
            return image_filename_style.get_default_raw_image_stack_extension();
        }
        ImageFilenameStyle::Old.get_default_raw_image_stack_extension()
    }

    /// Java package-private `isOldImageFilenameStyle`.
    pub fn is_old_image_filename_style(&self) -> bool {
        self.image_filename_style.is_none()
            || self.image_filename_style == Some(ImageFilenameStyle::Old)
    }
}
