//! `IMOD/Etomo/src/etomo/type/DialogType.java`.
//!
//! Java's typesafe-enum pattern is mirrored as a Rust enum with one variant per
//! `public static final DialogType` singleton, as `etomo/type/data_file_type.rs` does;
//! Java's identity comparisons become variant matches.  The three instance fields
//! (`name`, `index`, `dataFileType`) are all assigned by the private constructor from
//! the singleton's own arguments, so each is a per-variant accessor; `name` is
//! `toString(dataFileType, index)`, exactly what the constructor stores.
//!
//! Java `Properties` is modelled by a deterministic `BTreeMap<String, String>`, the
//! representation `etomo/type/string_property.rs` and `etomo/type/const_etomo_number.rs`
//! already use.
#![allow(dead_code)]

use std::collections::BTreeMap;

use super::data_file_type::DataFileType;
use super::interface_type::InterfaceType;
use crate::imod::etomo::ui::shared_strings;

/// Java private static `PROPERTIES_KEY`.
const PROPERTIES_KEY: &str = "DialogType";

/// Java private static `setupIndex`.
const SETUP_INDEX: i32 = 0;
/// Java private static `preProcessingIndex`.
const PRE_PROCESSING_INDEX: i32 = 1;
/// Java private static `coarseAlignmentIndex`.
const COARSE_ALIGNMENT_INDEX: i32 = 2;
/// Java private static `fiducialModelIndex`.
const FIDUCIAL_MODEL_INDEX: i32 = 3;
/// Java private static `fineAlignmentIndex`.
const FINE_ALIGNMENT_INDEX: i32 = 4;
/// Java private static `tomogramPositioningIndex`.
const TOMOGRAM_POSITIONING_INDEX: i32 = 5;
/// Java private static `finalAlignedStackIndex`.
const FINAL_ALIGNED_STACK_INDEX: i32 = 6;
/// Java private static `tomogramGenerationIndex`.
const TOMOGRAM_GENERATION_INDEX: i32 = 7;
/// Java private static `tomogramCombinationIndex`.
const TOMOGRAM_COMBINATION_INDEX: i32 = 8;
/// Java private static `postProcessingIndex`.
const POST_PROCESSING_INDEX: i32 = 9;
/// Java private static `cleanUpIndex`.
const CLEAN_UP_INDEX: i32 = 10;

/// Java public static `TOTAL_RECON`.
pub const TOTAL_RECON: i32 = CLEAN_UP_INDEX + 1;

/// Java private static `joinIndex`.
const JOIN_INDEX: i32 = 0;

/// Java public static `TOTAL_JOIN`.
pub const TOTAL_JOIN: i32 = JOIN_INDEX + 1;

/// Java private static `parallelIndex`.
const PARALLEL_INDEX: i32 = 0;
/// Java private static `anisotropicDiffusionIndex`.
const ANISOTROPIC_DIFFUSION_INDEX: i32 = 1;

/// Java public static `TOTAL_PARALLEL`.
pub const TOTAL_PARALLEL: i32 = ANISOTROPIC_DIFFUSION_INDEX + 1;

/// Java private static `peetIndex`.
const PEET_INDEX: i32 = 1;

/// Java public static `TOTAL_PEET`.
pub const TOTAL_PEET: i32 = PEET_INDEX + 1;

/// Java private static `serialSectionsIndex`.
const SERIAL_SECTIONS_INDEX: i32 = 1;

/// Java public static `TOTAL_SERIAL_SECTIONS`.
pub const TOTAL_SERIAL_SECTIONS: i32 = SERIAL_SECTIONS_INDEX + 1;

/// Java private static `batchRunTomoIndex`.
const BATCH_RUN_TOMO_INDEX: i32 = 0;

/// Java public static `TOTAL_BATCH_RUN_TOMO`.
pub const TOTAL_BATCH_RUN_TOMO: i32 = BATCH_RUN_TOMO_INDEX + 1;

// Storable names cannot be changed without handling the resulting backwards
// compatibility errors.
/// Java private static `SETUP_RECON_NAME`.
const SETUP_RECON_NAME: &str = "SetupRecon";
/// Java private static `PRE_PROCESSING_NAME`.
const PRE_PROCESSING_NAME: &str = "PreProc";
/// Java private static `COARSE_ALIGNMENT_NAME`.
const COARSE_ALIGNMENT_NAME: &str = "CoarseAlign";
/// Java private static `FIDUCIAL_MODEL_NAME`.
const FIDUCIAL_MODEL_NAME: &str = "FidModel";
/// Java private static `FINE_ALIGNMENT_NAME`.
const FINE_ALIGNMENT_NAME: &str = "FineAlign";
/// Java private static `TOMOGRAM_POSITIONING_NAME`.
const TOMOGRAM_POSITIONING_NAME: &str = "TomoPos";
/// Java private static `FINAL_ALIGNED_STACK_NAME`.
const FINAL_ALIGNED_STACK_NAME: &str = "FinalStack";
/// Java private static `TOMOGRAM_GENERATION_NAME`.
const TOMOGRAM_GENERATION_NAME: &str = "TomoGen";
/// Java private static `TOMOGRAM_COMBINATION_NAME`.
const TOMOGRAM_COMBINATION_NAME: &str = "Combine";
/// Java private static `POST_PROCESSING_NAME`.
const POST_PROCESSING_NAME: &str = "PostProc";
/// Java private static `CLEAN_UP_NAME`.
const CLEAN_UP_NAME: &str = "CleanUp";
/// Java private static `JOIN_NAME`.
const JOIN_NAME: &str = "Join";
/// Java private static `PARALLEL_NAME`.
const PARALLEL_NAME: &str = "Parallel";
/// Java private static `ANISOTROPIC_DIFFUSION_NAME`.
const ANISOTROPIC_DIFFUSION_NAME: &str = "AnisotropicDiffusion";
/// Java private static `BATCH_RUN_TOMO_NAME`.
const BATCH_RUN_TOMO_NAME: &str = "BatchRunTomo";
/// Java private static `PEET_STARTUP_NAME`.
const PEET_STARTUP_NAME: &str = "PeetStart";
/// Java private static `PEET_NAME`.
const PEET_NAME: &str = "Peet";
/// Java private static `TOOLS_NAME`.
const TOOLS_NAME: &str = "Tools";
/// Java private static `DIRECTIVE_EDITOR_NAME`.
const DIRECTIVE_EDITOR_NAME: &str = "DirectiveEditor";

/// Java `DialogType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DialogType {
    /// Java `SETUP_RECON = new DialogType(DataFileType.RECON, setupIndex)`.
    SetupRecon,
    /// Java `PRE_PROCESSING = new DialogType(DataFileType.RECON, preProcessingIndex)`.
    PreProcessing,
    /// Java `COARSE_ALIGNMENT =
    /// new DialogType(DataFileType.RECON, coarseAlignmentIndex)`.
    CoarseAlignment,
    /// Java `FIDUCIAL_MODEL = new DialogType(DataFileType.RECON, fiducialModelIndex)`.
    FiducialModel,
    /// Java `FINE_ALIGNMENT = new DialogType(DataFileType.RECON, fineAlignmentIndex)`.
    FineAlignment,
    /// Java `TOMOGRAM_POSITIONING =
    /// new DialogType(DataFileType.RECON, tomogramPositioningIndex)`.
    TomogramPositioning,
    /// Java `FINAL_ALIGNED_STACK =
    /// new DialogType(DataFileType.RECON, finalAlignedStackIndex)`.
    FinalAlignedStack,
    /// Java `TOMOGRAM_GENERATION =
    /// new DialogType(DataFileType.RECON, tomogramGenerationIndex)`.
    TomogramGeneration,
    /// Java `TOMOGRAM_COMBINATION =
    /// new DialogType(DataFileType.RECON, tomogramCombinationIndex)`.
    TomogramCombination,
    /// Java `POST_PROCESSING = new DialogType(DataFileType.RECON, postProcessingIndex)`.
    PostProcessing,
    /// Java `CLEAN_UP = new DialogType(DataFileType.RECON, cleanUpIndex)`.
    CleanUp,
    /// Java `JOIN = new DialogType(DataFileType.JOIN, joinIndex)`.
    Join,
    /// Java `PARALLEL = new DialogType(DataFileType.PARALLEL, parallelIndex)`.
    Parallel,
    /// Java `ANISOTROPIC_DIFFUSION =
    /// new DialogType(DataFileType.PARALLEL, anisotropicDiffusionIndex)`.
    AnisotropicDiffusion,
    /// Java `PEET_STARTUP = new DialogType(DataFileType.PEET, 0)`.
    PeetStartup,
    /// Java `PEET = new DialogType(DataFileType.PEET, peetIndex)`.
    Peet,
    /// Java `SERIAL_SECTIONS_STARTUP =
    /// new DialogType(DataFileType.SERIAL_SECTIONS, 0)`.
    SerialSectionsStartup,
    /// Java `SERIAL_SECTIONS =
    /// new DialogType(DataFileType.SERIAL_SECTIONS, serialSectionsIndex)`.
    SerialSections,
    /// Java `TOOLS = new DialogType(DataFileType.TOOLS, 0)`.
    Tools,
    /// Java `DIRECTIVE_EDITOR = new DialogType(DataFileType.DIRECTIVE_EDITOR, 0)`.
    DirectiveEditor,
    /// Java `BATCH_RUN_TOMO = new DialogType(DataFileType.BATCH_RUN_TOMO, 0)`.
    BatchRunTomo,
}

impl DialogType {
    /// Java private final field `dataFileType`, the private constructor's first
    /// argument.
    fn data_file_type(self) -> Option<DataFileType> {
        match self {
            Self::SetupRecon
            | Self::PreProcessing
            | Self::CoarseAlignment
            | Self::FiducialModel
            | Self::FineAlignment
            | Self::TomogramPositioning
            | Self::FinalAlignedStack
            | Self::TomogramGeneration
            | Self::TomogramCombination
            | Self::PostProcessing
            | Self::CleanUp => Some(DataFileType::Recon),
            Self::Join => Some(DataFileType::Join),
            Self::Parallel | Self::AnisotropicDiffusion => Some(DataFileType::Parallel),
            Self::PeetStartup | Self::Peet => Some(DataFileType::Peet),
            Self::SerialSectionsStartup | Self::SerialSections => {
                Some(DataFileType::SerialSections)
            }
            Self::Tools => Some(DataFileType::Tools),
            Self::DirectiveEditor => Some(DataFileType::DirectiveEditor),
            Self::BatchRunTomo => Some(DataFileType::BatchRunTomo),
        }
    }

    /// Java private final field `index`, the private constructor's second argument.
    fn index_field(self) -> i32 {
        match self {
            Self::SetupRecon => SETUP_INDEX,
            Self::PreProcessing => PRE_PROCESSING_INDEX,
            Self::CoarseAlignment => COARSE_ALIGNMENT_INDEX,
            Self::FiducialModel => FIDUCIAL_MODEL_INDEX,
            Self::FineAlignment => FINE_ALIGNMENT_INDEX,
            Self::TomogramPositioning => TOMOGRAM_POSITIONING_INDEX,
            Self::FinalAlignedStack => FINAL_ALIGNED_STACK_INDEX,
            Self::TomogramGeneration => TOMOGRAM_GENERATION_INDEX,
            Self::TomogramCombination => TOMOGRAM_COMBINATION_INDEX,
            Self::PostProcessing => POST_PROCESSING_INDEX,
            Self::CleanUp => CLEAN_UP_INDEX,
            Self::Join => JOIN_INDEX,
            Self::Parallel => PARALLEL_INDEX,
            Self::AnisotropicDiffusion => ANISOTROPIC_DIFFUSION_INDEX,
            Self::PeetStartup => 0,
            Self::Peet => PEET_INDEX,
            Self::SerialSectionsStartup => 0,
            Self::SerialSections => SERIAL_SECTIONS_INDEX,
            Self::Tools => 0,
            Self::DirectiveEditor => 0,
            Self::BatchRunTomo => 0,
        }
    }

    /// Java private final field `name`, which the private constructor
    /// `DialogType(DataFileType, int)` sets to `toString(dataFileType, index)`.
    fn name(self) -> String {
        self.to_string_of(self.data_file_type(), self.index_field())
    }

    /// Java `getInterfaceType`.
    pub fn get_interface_type(self) -> Option<InterfaceType> {
        let data_file_type = match self.data_file_type() {
            None => return None,
            Some(data_file_type) => data_file_type,
        };
        data_file_type.get_interface_type()
    }

    /// Java `getIndex`.
    pub fn get_index(self) -> i32 {
        self.index_field()
    }

    /// Java `getStorableName()`.  Storable names cannot be changed without handling the
    /// resulting backwards compatibility errors.
    pub fn get_storable_name(self) -> String {
        self.get_storable_name_of(self.data_file_type(), self.index_field())
    }

    /// Java `getCompactLabel()`.
    pub fn get_compact_label(self) -> String {
        self.get_compact_label_of(self.data_file_type(), self.index_field())
    }

    /// Java `toIndex`.
    pub fn to_index(self) -> i32 {
        self.index_field()
    }

    /// Java private `toString(DataFileType, int)`.
    fn to_string_of(self, data_file_type: Option<DataFileType>, index: i32) -> String {
        if data_file_type == Some(DataFileType::Recon) {
            match index {
                SETUP_INDEX => return "Setup Tomogram".to_string(),
                PRE_PROCESSING_INDEX => return "Pre-processing".to_string(),
                COARSE_ALIGNMENT_INDEX => return "Coarse Alignment".to_string(),
                FIDUCIAL_MODEL_INDEX => return "Fiducial Model Gen.".to_string(),
                FINE_ALIGNMENT_INDEX => return "Fine Alignment".to_string(),
                TOMOGRAM_POSITIONING_INDEX => return "Tomogram Positioning".to_string(),
                FINAL_ALIGNED_STACK_INDEX => {
                    return shared_strings::FINAL_ALIGNED_STACK_LABEL.to_string();
                }
                TOMOGRAM_GENERATION_INDEX => return "Tomogram Generation".to_string(),
                TOMOGRAM_COMBINATION_INDEX => return "Tomogram Combination".to_string(),
                POST_PROCESSING_INDEX => return "Post-processing".to_string(),
                CLEAN_UP_INDEX => return "Clean Up".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Join) {
            match index {
                JOIN_INDEX => return "Join".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Parallel) {
            match index {
                PARALLEL_INDEX => return "Parallel".to_string(),
                ANISOTROPIC_DIFFUSION_INDEX => return "Anisotropic Diffusion".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::BatchRunTomo) {
            match index {
                BATCH_RUN_TOMO_INDEX => return "Batch Run Tomo".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Peet) {
            match index {
                0 => return "PEET Startup".to_string(),
                PEET_INDEX => return "PEET".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Tools) {
            match index {
                0 => return "Tools".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::DirectiveEditor) {
            match index {
                0 => return "Directive Editor".to_string(),
                _ => {}
            }
        }
        String::new()
    }

    /// Java private `getCompactLabel(DataFileType, int)`.  Return a name without spaces.
    /// All storable names must be unique to DialogType.
    fn get_compact_label_of(self, data_file_type: Option<DataFileType>, index: i32) -> String {
        if data_file_type == Some(DataFileType::Recon) {
            match index {
                SETUP_INDEX => return "Setup".to_string(),
                PRE_PROCESSING_INDEX => return "Pre".to_string(),
                COARSE_ALIGNMENT_INDEX => return "Coarse".to_string(),
                FIDUCIAL_MODEL_INDEX => return "Track".to_string(),
                FINE_ALIGNMENT_INDEX => return "Fine".to_string(),
                TOMOGRAM_POSITIONING_INDEX => return "Pos".to_string(),
                FINAL_ALIGNED_STACK_INDEX => return "Stack".to_string(),
                TOMOGRAM_GENERATION_INDEX => return "Gen".to_string(),
                TOMOGRAM_COMBINATION_INDEX => return "Comb".to_string(),
                POST_PROCESSING_INDEX => return "Post".to_string(),
                CLEAN_UP_INDEX => return "Clean".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Parallel) {
            match index {
                PARALLEL_INDEX => return "Para".to_string(),
                ANISOTROPIC_DIFFUSION_INDEX => return "NAD".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::BatchRunTomo) {
            match index {
                BATCH_RUN_TOMO_INDEX => return "BRT".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Peet) {
            match index {
                0 => return "PEET-Start".to_string(),
                PEET_INDEX => return "PEET".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Tools) {
            match index {
                0 => return "Tools".to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::DirectiveEditor) {
            match index {
                0 => return "Dir-Ed".to_string(),
                _ => {}
            }
        }
        String::new()
    }

    /// Java private `getStorableName(DataFileType, int)`.  Storable names cannot be
    /// changed without handling the resulting backwards compatibility errors.  Return a
    /// name without spaces.  All storable names must be unique to DialogType.
    fn get_storable_name_of(self, data_file_type: Option<DataFileType>, index: i32) -> String {
        if data_file_type == Some(DataFileType::Recon) {
            match index {
                SETUP_INDEX => return SETUP_RECON_NAME.to_string(),
                PRE_PROCESSING_INDEX => return PRE_PROCESSING_NAME.to_string(),
                COARSE_ALIGNMENT_INDEX => return COARSE_ALIGNMENT_NAME.to_string(),
                FIDUCIAL_MODEL_INDEX => return FIDUCIAL_MODEL_NAME.to_string(),
                FINE_ALIGNMENT_INDEX => return FINE_ALIGNMENT_NAME.to_string(),
                TOMOGRAM_POSITIONING_INDEX => {
                    return TOMOGRAM_POSITIONING_NAME.to_string();
                }
                FINAL_ALIGNED_STACK_INDEX => return FINAL_ALIGNED_STACK_NAME.to_string(),
                TOMOGRAM_GENERATION_INDEX => return TOMOGRAM_GENERATION_NAME.to_string(),
                TOMOGRAM_COMBINATION_INDEX => {
                    return TOMOGRAM_COMBINATION_NAME.to_string();
                }
                POST_PROCESSING_INDEX => return POST_PROCESSING_NAME.to_string(),
                CLEAN_UP_INDEX => return CLEAN_UP_NAME.to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Parallel) {
            match index {
                PARALLEL_INDEX => return PARALLEL_NAME.to_string(),
                ANISOTROPIC_DIFFUSION_INDEX => {
                    return ANISOTROPIC_DIFFUSION_NAME.to_string();
                }
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::BatchRunTomo) {
            match index {
                BATCH_RUN_TOMO_INDEX => return BATCH_RUN_TOMO_NAME.to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Peet) {
            match index {
                0 => return PEET_STARTUP_NAME.to_string(),
                PEET_INDEX => return PEET_NAME.to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::Tools) {
            match index {
                0 => return TOOLS_NAME.to_string(),
                _ => {}
            }
        } else if data_file_type == Some(DataFileType::DirectiveEditor) {
            match index {
                0 => return DIRECTIVE_EDITOR_NAME.to_string(),
                _ => {}
            }
        }
        String::new()
    }

    /// Java `equals(String)`.
    pub fn equals(self, storable_name: Option<&str>) -> bool {
        let storable_name = match storable_name {
            None => return false,
            Some(storable_name) => storable_name,
        };
        self.get_storable_name_of(self.data_file_type(), self.index_field()) == storable_name
    }

    /// Java `store(Properties)`.
    pub fn store(self, props: &mut BTreeMap<String, String>) {
        props.insert(
            PROPERTIES_KEY.to_string(),
            self.get_storable_name_of(self.data_file_type(), self.index_field()),
        );
    }

    /// Java private static `createKey`.
    fn create_key(prepend: &str) -> String {
        if prepend.ends_with('.') {
            return format!("{}{}", prepend, PROPERTIES_KEY);
        }
        format!("{}.{}", prepend, PROPERTIES_KEY)
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(self, props: &mut BTreeMap<String, String>, prepend: &str) {
        props.insert(
            Self::create_key(prepend),
            self.get_storable_name_of(self.data_file_type(), self.index_field()),
        );
    }

    /// Java static `remove(Properties, String)`.
    pub fn remove(props: &mut BTreeMap<String, String>, prepend: &str) {
        props.remove(&Self::create_key(prepend));
    }

    /// Java static `getInstance(String)`.
    pub fn get_instance(storable_name: Option<&str>) -> Option<DialogType> {
        let storable_name = match storable_name {
            None => return None,
            Some(storable_name) => storable_name,
        };
        if storable_name == SETUP_RECON_NAME {
            return Some(Self::SetupRecon);
        }
        if storable_name == PRE_PROCESSING_NAME {
            return Some(Self::PreProcessing);
        }
        if storable_name == COARSE_ALIGNMENT_NAME {
            return Some(Self::CoarseAlignment);
        }
        if storable_name == FIDUCIAL_MODEL_NAME {
            return Some(Self::FiducialModel);
        }
        if storable_name == FINE_ALIGNMENT_NAME {
            return Some(Self::FineAlignment);
        }
        if storable_name == TOMOGRAM_POSITIONING_NAME {
            return Some(Self::TomogramPositioning);
        }
        if storable_name == FINAL_ALIGNED_STACK_NAME {
            return Some(Self::FinalAlignedStack);
        }
        if storable_name == TOMOGRAM_GENERATION_NAME {
            return Some(Self::TomogramGeneration);
        }
        if storable_name == TOMOGRAM_COMBINATION_NAME {
            return Some(Self::TomogramCombination);
        }
        if storable_name == POST_PROCESSING_NAME {
            return Some(Self::PostProcessing);
        }
        if storable_name == CLEAN_UP_NAME {
            return Some(Self::CleanUp);
        }
        if storable_name == PARALLEL_NAME {
            return Some(Self::Parallel);
        }
        if storable_name == ANISOTROPIC_DIFFUSION_NAME {
            return Some(Self::AnisotropicDiffusion);
        }
        if storable_name == BATCH_RUN_TOMO_NAME {
            return Some(Self::BatchRunTomo);
        }
        if storable_name == PEET_STARTUP_NAME {
            return Some(Self::PeetStartup);
        }
        if storable_name == PEET_NAME {
            return Some(Self::Peet);
        }
        if storable_name == TOOLS_NAME {
            return Some(Self::Tools);
        }
        if storable_name == DIRECTIVE_EDITOR_NAME {
            return Some(Self::DirectiveEditor);
        }
        None
    }

    /// Java static `load(Properties, String)`.  Load property value without a default.
    pub fn load(props: &BTreeMap<String, String>, prepend: &str) -> Option<DialogType> {
        Self::get_instance(props.get(&Self::create_key(prepend)).map(|x| x.as_str()))
    }

    /// Java static `load(DataFileType, Properties)`.
    pub fn load_with_data_file_type(
        data_file_type: Option<DataFileType>,
        props: &BTreeMap<String, String>,
    ) -> Option<DialogType> {
        let default_type = Self::get_default(data_file_type);
        if let Some(default_type) = default_type {
            let default_string = default_type.to_string();
            return Self::get_instance(Some(
                props
                    .get(PROPERTIES_KEY)
                    .map(|x| x.as_str())
                    .unwrap_or(&default_string),
            ));
        }
        Self::get_instance(props.get(PROPERTIES_KEY).map(|x| x.as_str()))
    }

    /// Java static `getDefault(DataFileType)`.
    pub fn get_default(data_file_type: Option<DataFileType>) -> Option<DialogType> {
        if data_file_type == Some(DataFileType::Parallel) {
            return Some(Self::Parallel);
        }
        if data_file_type == Some(DataFileType::BatchRunTomo) {
            return Some(Self::BatchRunTomo);
        }
        if data_file_type == Some(DataFileType::Peet) {
            return Some(Self::Peet);
        }
        if data_file_type == Some(DataFileType::DirectiveEditor) {
            return Some(Self::DirectiveEditor);
        }
        None
    }
}

/// Java `toString()`.  Returns a string representation of the object: the `name` field
/// the private constructor computed.
impl std::fmt::Display for DialogType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}
