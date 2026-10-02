//! `IMOD/Etomo/src/etomo/comscript/ExtractpiecesParam.java`.
//!
//! Builds the `extractpieces <stack> <piece list> [-MdocMetadataFile]` command line.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::util::dataset_files;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `COMMAND_NAME`.
pub const COMMAND_NAME: &str = "extractpieces";

/// Java private static `enableMdocMetadataFile`, shared by every instance.
static ENABLE_MDOC_METADATA_FILE: AtomicBool = AtomicBool::new(false);

/// Java `ExtractpiecesParam`.
pub struct ExtractpiecesParam {
    axis_id: AxisID,
    manager: &'static dyn BaseManager,
    /// Use when the manager setup is incomplete.
    raw_stack_file_name: Option<String>,
    /// Use when the manager setup is incomplete.
    root_name: Option<String>,
    /// Use when the manager setup is incomplete.
    axis_type: Option<AxisType>,
    command_array: Option<Vec<String>>,
}

impl ExtractpiecesParam {
    /// Java `ExtractpiecesParam(BaseManager, AxisID)`.
    pub fn new(manager: &'static dyn BaseManager, axis_id: AxisID) -> ExtractpiecesParam {
        ExtractpiecesParam {
            axis_id,
            manager,
            raw_stack_file_name: None,
            root_name: None,
            axis_type: None,
            command_array: None,
        }
    }

    /// Java `ExtractpiecesParam(String, String, AxisType, BaseManager, AxisID)`.  Use this
    /// constructor when the manager setup is incomplete.
    pub fn new_with_raw_stack(
        raw_stack_file_name: Option<&str>,
        root_name: Option<&str>,
        axis_type: Option<AxisType>,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
    ) -> ExtractpiecesParam {
        ExtractpiecesParam {
            axis_id,
            manager,
            raw_stack_file_name: raw_stack_file_name.map(|s| s.to_string()),
            root_name: root_name.map(|s| s.to_string()),
            axis_type,
            command_array: None,
        }
    }

    /// Java `getCommand()`.
    pub fn get_command(&mut self) -> Vec<String> {
        if self.command_array.is_none() {
            self.build_command();
        }
        self.command_array.clone().unwrap_or_default()
    }

    /// Java private `buildCommand()`.
    fn build_command(&mut self) {
        let mut command: Vec<String> = Vec::new();
        command.push(format!(
            "{}{}",
            base_manager::get_imod_bin_path().unwrap_or_else(|| "null".to_string()),
            COMMAND_NAME
        ));
        match &self.raw_stack_file_name {
            None => {
                command.push(
                    dataset_files::get_stack_name(self.manager, Some(self.axis_id))
                        .unwrap_or_else(|| "null".to_string()),
                );
                command.push(dataset_files::get_piece_list_file_name(
                    self.manager,
                    Some(self.axis_id),
                ));
            }
            Some(raw_stack_file_name) => {
                command.push(raw_stack_file_name.clone());
                // ExtractpiecesParam.java:79-81 dereferences `manager.getBaseMetaData()`
                // without a null check (NullPointerException when the manager has no
                // metadata).  Fixed in translation: with no metadata the style and
                // extension are passed as null, which `deriveFileName` accepts.
                let meta_data = self.manager.get_base_meta_data();
                let image_filename_style =
                    meta_data.map(|meta_data| meta_data.base().get_image_filename_style());
                let raw_image_stack_extension =
                    meta_data.and_then(|meta_data| meta_data.get_raw_image_stack_extension());
                command.push(
                    file_type::CLASS
                        .piece_list
                        .derive_file_name(
                            self.root_name.as_deref(),
                            self.axis_type,
                            Some(self.axis_id),
                            image_filename_style,
                            raw_image_stack_extension,
                        )
                        .unwrap_or_else(|| "null".to_string()),
                );
            }
        }

        if ENABLE_MDOC_METADATA_FILE.load(Ordering::SeqCst) {
            command.push("-MdocMetadataFile".to_string());
        }

        let command_size = command.len();
        let mut command_array = vec![String::new(); command_size];
        for i in 0..command_size {
            command_array[i] = command[i].clone();
        }
        self.command_array = Some(command_array);
    }

    /// Java `setMdocMetadataFileEnabled(boolean)`.  Sets the static, class-wide flag.
    pub fn set_mdoc_metadata_file_enabled(&self, cb_mdoc_metadata_file: bool) {
        ENABLE_MDOC_METADATA_FILE.store(cb_mdoc_metadata_file, Ordering::SeqCst);
    }
}
