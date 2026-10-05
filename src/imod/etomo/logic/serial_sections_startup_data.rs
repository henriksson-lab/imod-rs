//! `IMOD/Etomo/src/etomo/logic/SerialSectionsStartupData.java`.
//!
//! The saved state of the Serial Sections startup dialog: the stack, view type,
//! .mdoc piece-list choice, distortion field and binning, with the validation and
//! the com-script settings derived from them.  A plain value object, owned by the
//! startup dialog (an event dispatch thread object).

use std::path::{Path, PathBuf};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::Number;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::swing::radio_button_interface::EnumeratedTypeRef;
use crate::imod::etomo::util::utilities;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `String.matches("\\s*")`.
fn matches_whitespace(string: &str) -> bool {
    string
        .chars()
        .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
}

/// Java `public final class SerialSectionsStartupData`.
#[derive(Clone)]
pub struct SerialSectionsStartupData {
    /// Java private final `stackLabel`.
    stack_label: String,
    /// Java private final `viewTypeLabel`.
    view_type_label: String,
    /// Java private `stack`, initially null.
    stack: Option<PathBuf>,
    /// Java private `viewType`, initially null.
    view_type: Option<ViewType>,
    /// Java private `mdocMetadataFileStatus`, initially false.
    mdoc_metadata_file_status: bool,
    /// Java private `distortionField`, initially null.
    distortion_field: Option<PathBuf>,
    /// Java private `imagesAreBinned`, initially null.
    images_are_binned: Option<Number>,
}

impl SerialSectionsStartupData {
    /// Java `SerialSectionsStartupData(String, String)`.
    pub fn new(
        stack_label: Option<&str>,
        view_type_label: Option<&str>,
    ) -> SerialSectionsStartupData {
        let stack_label = match stack_label {
            Some(stack_label) if !matches_whitespace(stack_label) => stack_label.to_string(),
            _ => "stack".to_string(),
        };
        let view_type_label = match view_type_label {
            Some(view_type_label) if !matches_whitespace(view_type_label) => {
                view_type_label.to_string()
            }
            _ => "view type".to_string(),
        };
        SerialSectionsStartupData {
            stack_label,
            view_type_label,
            stack: None,
            view_type: None,
            mdoc_metadata_file_status: false,
            distortion_field: None,
            images_are_binned: None,
        }
    }

    /// Java `setStack(File)`.
    pub fn set_stack(&mut self, input: Option<&Path>) {
        self.stack = input.map(Path::to_path_buf);
    }

    /// Java `setViewType(EnumeratedType)`: `ViewType.getInstance(input)`.
    pub fn set_view_type(&mut self, input: Option<&EnumeratedTypeRef>) {
        self.view_type = Some(
            match input.and_then(|input| input.downcast_ref::<ViewType>()) {
                Some(view_type) => ViewType::get_instance(*view_type),
                None => ViewType::DEFAULT,
            },
        );
    }

    /// Java `setMdocMetadataFileStatus(boolean)`.
    pub fn set_mdoc_metadata_file_status(&mut self, input: bool) {
        self.mdoc_metadata_file_status = input;
    }

    /// Java `setDistortionFile(File)`.
    pub fn set_distortion_file(&mut self, input: Option<&Path>) {
        self.distortion_field = input.map(Path::to_path_buf);
    }

    /// Java `setImagesAreBinned(Number)`.  Sets imagesAreBinned if input is not 1.
    ///
    /// Upstream bug fixed in translation (SerialSectionsStartupData.java:77): a null
    /// input throws NullPointerException on `intValue()`; it is stored as null here.
    pub fn set_images_are_binned(&mut self, input: Option<Number>) {
        match input {
            Some(input) if input.int_value() == 1 => self.images_are_binned = None,
            input => self.images_are_binned = input,
        }
    }

    /// Java `getPreblendParameters(BlendmontParam, BaseManager)`.
    pub fn get_preblend_parameters(
        &self,
        param: &mut BlendmontParam,
        manager: &'static dyn BaseManager,
    ) {
        self.get_parameters_blendmont(param, manager);
        param.set_image_input_file_file(self.stack.as_deref());
        param.set_image_output_file(
            &file_type::CLASS.preblend_output_mrc,
            self.get_root_name().as_deref(),
            Some(AxisType::SingleAxis),
        );
    }

    /// Java `getBlendParameters(BlendmontParam, BaseManager)`.
    pub fn get_blend_parameters(
        &self,
        param: &mut BlendmontParam,
        manager: &'static dyn BaseManager,
    ) {
        self.get_parameters_blendmont(param, manager);
        param.set_from_scratch(true);
        let meta_data = manager.get_base_meta_data();
        param.set_image_input_file(
            file_type::CLASS
                .preblend_output_mrc
                .derive_file_name(
                    self.get_root_name().as_deref(),
                    Some(AxisType::SingleAxis),
                    Some(AxisID::Only),
                    meta_data.map(|meta_data| meta_data.base().get_image_filename_style()),
                    meta_data.and_then(|meta_data| meta_data.get_raw_image_stack_extension()),
                )
                .as_deref(),
        );
        param.set_image_output_file(
            &file_type::CLASS.aligned_stack_mrc,
            self.get_root_name().as_deref(),
            Some(AxisType::SingleAxis),
        );
    }

    /// Java private `getParameters(BlendmontParam, BaseManager)`.
    fn get_parameters_blendmont(
        &self,
        param: &mut BlendmontParam,
        manager: &'static dyn BaseManager,
    ) {
        match &self.distortion_field {
            None => param.reset_distortion_field(),
            Some(distortion_field) => param.set_distortion_field(Some(
                &utilities::java_io_file_get_name(&distortion_field.to_string_lossy()),
            )),
        }
        let meta_data = manager.get_base_meta_data();
        param.set_piece_list_input(
            file_type::CLASS
                .piece_list
                .derive_file_name(
                    self.get_root_name().as_deref(),
                    Some(AxisType::SingleAxis),
                    Some(AxisID::Only),
                    meta_data.map(|meta_data| meta_data.base().get_image_filename_style()),
                    meta_data.and_then(|meta_data| meta_data.get_raw_image_stack_extension()),
                )
                .as_deref(),
        );
        param.set_root_name_for_edges(self.get_root_name().as_deref());
        param.set_images_are_binned(self.images_are_binned);
        param.set_adjust_origin(true);
    }

    /// Java `getParameters(NewstParam, BaseManager)`.
    pub fn get_parameters_newst(&self, param: &mut NewstParam, _manager: &'static dyn BaseManager) {
        match &self.stack {
            None => param.reset_input_file(),
            Some(stack) => param.set_input_file(Some(&utilities::java_io_file_get_name(
                &stack.to_string_lossy(),
            ))),
        }
        param.set_output_file_derived(
            &file_type::CLASS.aligned_stack_mrc,
            self.get_root_name().as_deref(),
            Some(AxisType::SingleAxis),
        );
        match &self.distortion_field {
            None => param.reset_distortion_field(),
            Some(distortion_field) => param.set_distortion_field(Some(
                &utilities::java_io_file_get_name(&distortion_field.to_string_lossy()),
            )),
        }
        param.set_images_are_binned(self.images_are_binned);
        param.set_adjust_origin(true);
    }

    /// Java `validate()`.  Returns an error message or null if valid.
    pub fn validate(&self) -> Option<String> {
        let Some(stack) = &self.stack else {
            return Some(format!("Missing required entry: {}.", self.stack_label));
        };
        if !stack.exists() {
            return Some(format!("{}doesn't exist.", self.stack_label));
        }
        if !utilities::java_io_file_can_read(&stack.to_string_lossy()) {
            return Some(format!("{}is not readable.", self.stack_label));
        }
        if self.view_type.is_none() {
            return Some(format!("Missing required entry: {}.", self.view_type_label));
        }
        None
    }

    /// Java `getRootName()`.  The root name is the file name of the stack member
    /// variable, minus the extension.
    pub fn get_root_name(&self) -> Option<String> {
        let stack = self.stack.as_ref()?;
        let name = utilities::java_io_file_get_name(&stack.to_string_lossy());
        match name.rfind('.') {
            None => Some(name),
            Some(index) => Some(name[..index].to_string()),
        }
    }

    /// Java `getParamFile()`.  Builds and returns the param file.
    pub fn get_param_file(&self) -> Option<PathBuf> {
        let stack = self.stack.as_ref()?;
        let root_name = self.get_root_name()?;
        let child = format!(
            "{}{}",
            root_name,
            DataFileType::SerialSections.extension().unwrap_or("null")
        );
        Some(PathBuf::from(
            match utilities::java_io_file_get_parent(&stack.to_string_lossy()) {
                None => child,
                Some(parent) => utilities::java_io_file_new(&parent, &child),
            },
        ))
    }

    /// Java `getViewType()`.
    pub fn get_view_type(&self) -> Option<ViewType> {
        self.view_type
    }

    /// Java `getStack()`.
    pub fn get_stack(&self) -> Option<&Path> {
        self.stack.as_deref()
    }

    /// Java `getMdocMetadataFile()`.
    pub fn get_mdoc_metadata_file(&self) -> bool {
        self.mdoc_metadata_file_status
    }

    /// Java `getDistortionField()`.
    pub fn get_distortion_field(&self) -> Option<&Path> {
        self.distortion_field.as_deref()
    }

    /// Java `getImagesAreBinned()`.
    pub fn get_images_are_binned(&self) -> Option<Number> {
        self.images_are_binned
    }
}
