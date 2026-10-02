//! `IMOD/Etomo/src/etomo/process/ProcessOutputStrings.java`: the tags eTomo
//! looks for in the output of the programs it runs.
#![allow(dead_code)]

/// Java `END_PARAMETERS_TAG`.
pub const END_PARAMETERS_TAG: &str = "*** End of entries ***";

/// Java `START_PARAMETERS_TAG`.
pub const START_PARAMETERS_TAG: &str = "*** Entries to program";

/// Java `OPEN_BRACKET`.
pub const OPEN_BRACKET: char = '[';

/// Java `SUCCESS_TAG`.
pub const SUCCESS_TAG: &str = "SUCCESSFULLY COMPLETED";

/// Java `LOG_TAG`.
pub const LOG_TAG: &str = "[:LOG]";

/// Java `BRT_START_PARAMETERS_TAG`.
pub const BRT_START_PARAMETERS_TAG: &str = "*** Entries to program batchruntomo ***";

/// Java `BRT_DATASET_MSG_ID`.
pub const BRT_DATASET_MSG_ID: &str = "[brt1]";

/// Java `BRT_CURRENT_DATASET_TAG`.
pub const BRT_CURRENT_DATASET_TAG: &str = "set ";

/// Java `BRT_CREATED_DATASET_DIRECTORY`.
pub const BRT_CREATED_DATASET_DIRECTORY: &str = "Created dataset directory";

/// Java `BRT_DATASET_TAG`.
pub const BRT_DATASET_TAG: &str = "Starting data set";

/// Java `BRT_STEP_TAG`.
pub const BRT_STEP_TAG: &str = "running ";

/// Java `BRT_STEP_SUCCESS_MSG_ID`.
pub const BRT_STEP_SUCCESS_MSG_ID: &str = "[brt3]";

/// Java `BRT_STEP_SUCCESS_TAG`.
pub const BRT_STEP_SUCCESS_TAG: &str = "Successfully finished";

/// Java `BRT_DATASET_COMPLETED_MSG_ID`.
pub const BRT_DATASET_COMPLETED_MSG_ID: &str = "[brt4]";

/// Java `BRT_DATASET_SUCCESS_TAG`.
pub const BRT_DATASET_SUCCESS_TAG: &str = "Completed dataset";

/// Java `BRT_TIME_STAMP_TAG`.
pub const BRT_TIME_STAMP_TAG: &str = " at ";

/// Java `BRT_BATCH_RUN_TOMO_ERROR_TAG`.
pub const BRT_BATCH_RUN_TOMO_ERROR_TAG: &str = "ERROR: batchruntomo -";

/// Java `BRT_AXIS_B_TAG`.
pub const BRT_AXIS_B_TAG: &str = "Starting axis B";

/// Java `BRT_LOG_TAGS`.
pub const BRT_LOG_TAGS: &[&str] = &[ "Final align -", "AUTOPATCHFIT - Refinematch found a good ", "AUTOPATCHFIT - Findwarp found a good " ];

/// Java `BRT_KILLING_MSG_ID`.
pub const BRT_KILLING_MSG_ID: &str = "[brt5]";

/// Java `BRT_KILLING_TAG`.
pub const BRT_KILLING_TAG: &str = "RECEIVED SIGNAL TO QUIT, JUST EXITING";

/// Java `BRT_PAUSED_MSG_ID`.
pub const BRT_PAUSED_MSG_ID: &str = "[brt6]";

/// Java `BRT_PAUSED_TAG`.
pub const BRT_PAUSED_TAG: &str = "Exiting after finishing dataset as requested";

/// Java `BRT_STARTING_DATASET_MSG_ID`.
pub const BRT_STARTING_DATASET_MSG_ID: &str = "[brt7]";

/// Java `BRT_STARTING_DATASET_TAG`.
pub const BRT_STARTING_DATASET_TAG: &str = "Beginning to process directive file";

/// Java `BRT_REACHED_STEP_TAG`.
pub const BRT_REACHED_STEP_TAG: &str = "Reached step";

/// Java `BRT_ABORT_SET_TAG`.
pub const BRT_ABORT_SET_TAG: &str = "ABORT SET:";

/// Java `BRT_ABORT_AXIS_TAG`.
pub const BRT_ABORT_AXIS_TAG: &str = "ABORT AXIS:";

/// Java `BRT_ENDING_STEP_PARAM_TAG`.
pub const BRT_ENDING_STEP_PARAM_TAG: &str = "EndingStep =";

/// Java `BRT_STARTING_STEP_PARAM_TAG`.
pub const BRT_STARTING_STEP_PARAM_TAG: &str = "StartingStep =";

/// Java `BRT_ROOT_NAME_PARAM_TAG`.
pub const BRT_ROOT_NAME_PARAM_TAG: &str = "RootName =";

/// Java `BRT_CURRENT_LOCATION_PARAM_TAG`.
pub const BRT_CURRENT_LOCATION_PARAM_TAG: &str = "CurrentLocation =";

/// Java `BRT_FILE_LOCATION_TAG`.
pub const BRT_FILE_LOCATION_TAG: &str = "to:";

/// Java `BRT_DELIVERED_MSG_ID`.
pub const BRT_DELIVERED_MSG_ID: &str = "[brt8]";

/// Java `BRT_DELIVERED_TAG`.
pub const BRT_DELIVERED_TAG: &str = "Delivered stack";

/// Java `BRT_RENAMED_MSG_ID`.
pub const BRT_RENAMED_MSG_ID: &str = "[brt9]";

/// Java `BRT_RENAMED_TAG`.
pub const BRT_RENAMED_TAG: &str = "Renamed stack from";

/// Java `BRT_ABORT_TAG`.
pub const BRT_ABORT_TAG: &str = "ABORT";

/// Java `BRT_DATASET_DIR_NOT_UNIQUE_ERR1`.
pub const BRT_DATASET_DIR_NOT_UNIQUE_ERR1: &str = "Dataset location";

/// Java `BRT_DATASET_DIR_NOT_UNIQUE_ERR2`.
pub const BRT_DATASET_DIR_NOT_UNIQUE_ERR2: &str = " must be unique";

/// Java `BRT_STARTED_DATASET_TAG`.
pub const BRT_STARTED_DATASET_TAG: &str = "[brt11]";

/// Java `BRT_DATASET_LOCATION_TAG`.
pub const BRT_DATASET_LOCATION_TAG: &str = "[brt13]";

/// Java `BRT_DATASET_LOCATION_START_TAG`.
pub const BRT_DATASET_LOCATION_START_TAG: &str = "In:";

/// Java `BRT_DATASET_LOG_CLOSED_TAG`.
pub const BRT_DATASET_LOG_CLOSED_TAG: &str = "[brt12]";

/// Java `BRT_START_AXIS_MSG_ID`.
pub const BRT_START_AXIS_MSG_ID: &str = "[brt14]";

/// Java `BRT_START_AXIS`.
pub const BRT_START_AXIS: &str = "Starting axis";

/// Java `BRT_TRANSFER_FID_B`.
pub const BRT_TRANSFER_FID_B: &str = "from A to B";

/// Java `BRT_TRANSFER_FID_A`.
pub const BRT_TRANSFER_FID_A: &str = "from B to A";

/// Java `SRW_SERIES_WATCHER_ERROR_TAG`.
pub const SRW_SERIES_WATCHER_ERROR_TAG: &str = "ERROR: serieswatcher -";

/// Java `SRW_ABORT_TAG`.
pub const SRW_ABORT_TAG: &str = "ABORT";

/// Java `SRW_CODE`.
pub const SRW_CODE: &str = "SRW";

/// Java `SRW_STARTING_TO_PROCESS_SET`.
pub const SRW_STARTING_TO_PROCESS_SET: &str = "[SRW1]";

/// Java `SRW_LOG_LOCATION`.
pub const SRW_LOG_LOCATION: &str = "[SRW2]";

/// Java `SRW_FINISHED_PROCESSING_SET`.
pub const SRW_FINISHED_PROCESSING_SET: &str = "[SRW3]";

/// Java `SRW_PROCESS_FINISHED`.
pub const SRW_PROCESS_FINISHED: &str = "[SRW4]";

/// Java `SRW_PROCESS_KILLED`.
pub const SRW_PROCESS_KILLED: &str = "[SRW5]";

/// Java `SRW_ROOT_NAME`.
pub const SRW_ROOT_NAME: &str = "[SRW6]";

/// Java `SPLIT_BATCH_PROCESSCHUNKS_MACHINE_LIST_MSG_ID`.
pub const SPLIT_BATCH_PROCESSCHUNKS_MACHINE_LIST_MSG_ID: &str = "[SPB1]";

/// Java `PROCESSCHUNK_RUNNING_COMSCRIPT_MSG_ID`.
pub const PROCESSCHUNK_RUNNING_COMSCRIPT_MSG_ID: &str = "[PRC1]";

/// Java `PROCESSCHUNK_FAILED_NEED_RESTART_MSG_ID`.
pub const PROCESSCHUNK_FAILED_NEED_RESTART_MSG_ID: &str = "[PRC2]";

/// Java `RESTRICT_ALIGN_RERUNNING_TILT_ALIGN_MSG_ID`.
pub const RESTRICT_ALIGN_RERUNNING_TILT_ALIGN_MSG_ID: &str = "[rsa1]";

/// Java `REDUCE_FILT_VOL_NOT_ENOUGH_MEMORY_ERROR_TAG`.
pub const REDUCE_FILT_VOL_NOT_ENOUGH_MEMORY_ERROR_TAG: &str = "[MTF1]";

/// Java `CHUNK_SETUP_OUTPUT_FILE`.
pub const CHUNK_SETUP_OUTPUT_FILE: &str = "[CHS1]";

/// Java `BRT_ETOMO_TAG_OLD` (deprecated - capitalization changed).
pub const BRT_ETOMO_TAG_OLD: &str = "starting eTomo with log in";

/// Java `BRT_ETOMO_TAG`.
pub const BRT_ETOMO_TAG: &str = "starting Etomo with log in";

/// Java `BRT_STEP_MSG_ID`.
pub const BRT_STEP_MSG_ID: &str = "[brt2]";

/// Java `BRT_STEP_END_TAG`.
pub const BRT_STEP_END_TAG: &str = ".com";
