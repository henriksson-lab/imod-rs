//! `IMOD/Etomo/src/etomo/type/DirectiveFileType.java`.
#![allow(dead_code)]

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type::CLASS;
use std::path::PathBuf;

/// Java `DirectiveFileType` typesafe enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectiveFileType {
    /// Java `BATCH_DEFAULTS`.
    BatchDefaults,
    /// Java `SCOPE`.
    Scope,
    /// Java `SYSTEM`.
    System,
    /// Java `USER`.
    User,
    /// Java `BATCH`.
    Batch,
}

/// Java `NUM`.
pub const NUM: i32 = 5;

impl DirectiveFileType {
    /// Java field `index`.
    pub fn get_index(self) -> i32 {
        match self {
            Self::BatchDefaults => 0,
            Self::Scope => 1,
            Self::System => 2,
            Self::User => 3,
            Self::Batch => 4,
        }
    }

    /// Java field `string` / `toString`.
    pub fn string(self) -> &'static str {
        match self {
            Self::BatchDefaults => "Batch Defaults",
            Self::Scope => "Scope",
            Self::System => "System",
            Self::User => "User",
            Self::Batch => "Batch",
        }
    }

    /// Java `getLabel`.
    pub fn get_label(self) -> &'static str {
        match self {
            Self::BatchDefaults => "Batch Defaults",
            Self::Scope => "Scope Template",
            Self::System => "System Template",
            Self::User => "User Template",
            Self::Batch => "Batch Directive File",
        }
    }

    /// Java `getInstance(String)`.
    pub fn get_instance(label: Option<&str>) -> Option<Self> {
        let label = label?;
        [
            Self::BatchDefaults,
            Self::Scope,
            Self::System,
            Self::User,
            Self::Batch,
        ]
        .into_iter()
        .find(|file_type| label == file_type.get_label())
    }

    /// Java `getInstance(int)`.
    pub fn get_instance_from_index(input: i32) -> Option<Self> {
        [
            Self::BatchDefaults,
            Self::Scope,
            Self::System,
            Self::User,
            Self::Batch,
        ]
        .into_iter()
        .find(|file_type| input == file_type.get_index())
    }

    /// Java static `exists`.
    pub fn exists(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        index: i32,
    ) -> bool {
        let file_type = match Self::get_instance_from_index(index) {
            Some(file_type) => file_type,
            None => return false,
        };
        let file = match file_type.get_local_file(manager, axis_id) {
            Some(file) => file,
            None => return false,
        };
        file.exists()
    }

    /// Java `isBatch`.
    pub fn is_batch(self) -> bool {
        self == Self::Batch
    }

    /// Java `isTemplate`.
    pub fn is_template(self) -> bool {
        !self.is_batch()
    }

    /// Java `getLocalFile`.
    pub fn get_local_file(
        self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
    ) -> Option<PathBuf> {
        let file_type = match self {
            Self::BatchDefaults => &CLASS.default_batch_run_tomo_autodoc,
            Self::Scope => &CLASS.local_scope_template,
            Self::System => &CLASS.local_system_template,
            Self::User => &CLASS.local_user_template,
            Self::Batch => &CLASS.local_batch_directive_file,
        };
        file_type.get_file(manager, axis_id)
    }

    /// Java static `toString(int)`.
    pub fn to_string_from_index(index: i32) -> Option<&'static str> {
        let file_type = Self::get_instance_from_index(index)?;
        if file_type == Self::BatchDefaults {
            return None;
        }
        Some(file_type.string())
    }

    /// Java static `getLabel(int)`.
    pub fn get_label_from_index(index: i32) -> Option<&'static str> {
        let file_type = Self::get_instance_from_index(index)?;
        if file_type == Self::BatchDefaults {
            return None;
        }
        Some(file_type.get_label())
    }
}

impl std::fmt::Display for DirectiveFileType {
    /// Java `toString`.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.string())
    }
}

#[cfg(test)]
mod tests {
    use super::{DirectiveFileType, NUM};

    #[test]
    fn source_singletons_have_their_java_indices_and_labels() {
        assert_eq!(NUM, 5);
        assert_eq!(
            DirectiveFileType::get_instance(Some("Scope Template")),
            Some(DirectiveFileType::Scope)
        );
        assert_eq!(
            DirectiveFileType::get_instance_from_index(4),
            Some(DirectiveFileType::Batch)
        );
        assert_eq!(DirectiveFileType::to_string_from_index(0), None);
        assert_eq!(DirectiveFileType::get_label_from_index(0), None);
    }
}
