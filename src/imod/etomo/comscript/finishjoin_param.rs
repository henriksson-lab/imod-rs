//! `IMOD/Etomo/src/etomo/comscript/FinishjoinParam.java`.
//!
//! The `finishjoin` command line (`python -u <scripts>finishjoin -PID ...`), built when
//! the param is constructed: from the join meta data for the Join tab's modes
//! (`genOptions`), and from the join state and the boundary table for the Rejoin and
//! Model tabs (`genRejoinOptions`).  The values each run used are kept as fields so that
//! `JoinProcessManager.postProcess` can save them in the `JoinState`.

use std::path::PathBuf;

use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::field_interface::{self, FieldInterface};
use super::process_details::{Hashtable, ProcessDetails};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::storage::loggable::{Loggable, LoggableException};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::const_int_key_list::ConstIntKeyList;
use crate::imod::etomo::r#type::const_join_meta_data::ConstJoinMetaData;
use crate::imod::etomo::r#type::const_join_state::ConstJoinState;
use crate::imod::etomo::r#type::const_section_table_row_data::ConstSectionTableRowData;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::int_key_list::IntKeyList;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;

/// Java `public static final String SIZE_TAG`.
pub const SIZE_TAG: &str = "Maximum size required:";
/// Java `public static final String OFFSET_TAG`.
pub const OFFSET_TAG: &str = "Offset needed to center:";
/// Java `public static final int SIZE_IN_X_INDEX`.
pub const SIZE_IN_X_INDEX: usize = 3;
/// Java `public static final int SIZE_IN_Y_INDEX`.
pub const SIZE_IN_Y_INDEX: usize = 4;
/// Java `public static final int OFFSET_IN_X_INDEX`.
pub const OFFSET_IN_X_INDEX: usize = 4;
/// Java `public static final int OFFSET_IN_Y_INDEX`.
pub const OFFSET_IN_Y_INDEX: usize = 5;

/// Java private static final `PROCESS_NAME`.
const PROCESS_NAME: ProcessName = ProcessName::FINISHJOIN;
/// Java `public static final String COMMAND_NAME`.
pub const COMMAND_NAME: &str = "finishjoin";

/// Java nested `Fields implements FieldInterface`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fields {
    /// Java `ALIGNMENT_REF_SECTION`.
    AlignmentRefSection,
    /// Java `SIZE_IN_X`.
    SizeInX,
    /// Java `SIZE_IN_Y`.
    SizeInY,
    /// Java `SHIFT_IN_X`.
    ShiftInX,
    /// Java `SHIFT_IN_Y`.
    ShiftInY,
    /// Java `BINNING`.
    Binning,
    /// Java `JOIN_START_LIST`.
    JoinStartList,
    /// Java `JOIN_END_LIST`.
    JoinEndList,
    /// Java `REFINE_START_LIST`.
    RefineStartList,
    /// Java `REFINE_END_LIST`.
    RefineEndList,
    /// Java `USE_EVERY_N_SLICES`.
    UseEveryNSlices,
    /// Java `LOCAL_FITS`.
    LocalFits,
}

impl FieldInterface for Fields {}

/// Java nested `public final static class Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// Java `FINISH_JOIN = new Mode("FinishJoin")`.
    FinishJoin,
    /// Java `MAX_SIZE = new Mode("MaxSize")`.
    MaxSize,
    /// Java `TRIAL = new Mode("Trial")`.
    Trial,
    /// Java `REJOIN = new Mode("Rejoin")`.
    Rejoin,
    /// Java `TRIAL_REJOIN = new Mode("TrialRejoin")`.
    TrialRejoin,
    /// Java `SUPPRESS_EXECUTION = new Mode("SuppressExecution")`.
    SuppressExecution,
}

/// Java `Mode.toString()`: the key.
impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::FinishJoin => "FinishJoin",
            Mode::MaxSize => "MaxSize",
            Mode::Trial => "Trial",
            Mode::Rejoin => "Rejoin",
            Mode::TrialRejoin => "TrialRejoin",
            Mode::SuppressExecution => "SuppressExecution",
        })
    }
}

impl CommandMode for Mode {}

/// Java `public final class FinishjoinParam implements CommandDetails`.
pub struct FinishjoinParam {
    /// Java private final `debug`.
    debug: i32,
    /// Java private final `commandArray`.
    command_array: Vec<String>,
    /// Java private `rootName`.
    root_name: Option<String>,
    /// Java private `outputFile`.
    output_file: Option<PathBuf>,
    /// Java private `mode`.
    mode: Mode,
    // set in genOptions
    /// Java private `alignmentRefSection`, initially null.
    alignment_ref_section: Option<EtomoNumber>,
    /// Java private `sizeInX`, initially null.
    size_in_x: Option<ScriptParameter>,
    /// Java private `sizeInY`, initially null.
    size_in_y: Option<ScriptParameter>,
    /// Java private `shiftInX`, initially null.
    shift_in_x: Option<ScriptParameter>,
    /// Java private `shiftInY`, initially null.
    shift_in_y: Option<ScriptParameter>,
    /// Java private `joinStartList`, initially null.
    join_start_list: Option<IntKeyList>,
    /// Java private `joinEndList`, initially null.
    join_end_list: Option<IntKeyList>,
    /// Java private `binning`, initially null.
    binning: Option<ScriptParameter>,
    /// Java private `useEveryNSlices`, initially null.
    use_every_n_slices: Option<EtomoNumber>,
    // set in getRejoinOptions
    /// Java private `refineStartList`, initially null.
    refine_start_list: Option<IntKeyList>,
    /// Java private `refineEndList`, initially null.
    refine_end_list: Option<IntKeyList>,
    /// Java private `localFits`, initially null.
    local_fits: Option<EtomoBoolean2>,
}

impl FinishjoinParam {
    /// Java `FinishjoinParam(JoinManager, Mode)`.
    pub fn new(manager: &'static JoinManager, mode: Mode) -> FinishjoinParam {
        FinishjoinParam::new_with_debug(manager, mode, 1)
    }

    /// Java `FinishjoinParam(JoinManager, Mode, int)`.
    pub fn new_with_debug(
        manager: &'static JoinManager,
        mode: Mode,
        debug: i32,
    ) -> FinishjoinParam {
        let mut param = FinishjoinParam {
            debug,
            command_array: Vec::new(),
            root_name: manager
                .get_base_meta_data()
                .and_then(|meta_data| meta_data.get_name()),
            output_file: file_type::CLASS.join.get_file(Some(manager), None),
            mode,
            alignment_ref_section: None,
            size_in_x: None,
            size_in_y: None,
            shift_in_x: None,
            shift_in_y: None,
            join_start_list: None,
            join_end_list: None,
            binning: None,
            use_every_n_slices: None,
            refine_start_list: None,
            refine_end_list: None,
            local_fits: None,
        };
        let options =
            if mode == Mode::Rejoin || mode == Mode::SuppressExecution || mode == Mode::TrialRejoin
            {
                param.gen_rejoin_options(manager)
            } else {
                param.gen_options(manager)
            };
        if mode == Mode::SuppressExecution && debug >= 1 {
            eprint!("SUPPRESS_EXECUTION:");
        }
        let command_size = 4;
        let mut command_array = vec![String::new(); options.len() + command_size];
        command_array[0] = "python".to_string();
        command_array[1] = "-u".to_string();
        command_array[2] = format!(
            "{}{}",
            etomo_director::INSTANCE
                .get_python_script_path()
                .unwrap_or_else(|| "null".to_string()),
            COMMAND_NAME
        );
        command_array[3] = "-PID".to_string();
        for i in 0..options.len() {
            command_array[i + command_size] = options[i].clone();
        }
        if debug >= 1 {
            let mut buffer = String::new();
            for i in 0..command_array.len() {
                buffer.push_str(&command_array[i]);
                if i < command_array.len() - 1 {
                    buffer.push(' ');
                }
            }
            eprintln!("{buffer}");
        }
        param.command_array = command_array;
        param
    }

    /// Java static `getOutputName(JoinManager)`.
    pub fn get_output_name(manager: &'static JoinManager) -> Option<String> {
        file_type::CLASS
            .join
            .get_file_name(Some(manager), Some(AxisID::Only))
    }

    /// Java static `getShift(String)`.  The Java `IllegalArgumentException` is the
    /// `Err` message.
    pub fn get_shift(offset: Option<&str>) -> Result<i32, String> {
        let mut offset_number = EtomoNumber::new_with_type(Some(Type::Integer));
        if offset_number.set_string(offset).is_valid() {
            return Ok(offset_number.get_int().wrapping_mul(-1));
        }
        Err(format!(
            "{}: {}",
            offset_number.get_description(),
            offset_number.get_invalid_reason()
        ))
    }

    /// Java private `genOptions()`.  For calls from the Join tab.  Generates options
    /// and sets the member variables (except refine start and end lists).  For rejoin
    /// see genRejoinOptions.
    fn gen_options(&mut self, manager: &'static JoinManager) -> Vec<String> {
        let mut options: Vec<String> = Vec::new();
        // options.add("-P");
        let meta_data = manager.get_const_meta_data();
        if meta_data.is_use_alignment_ref_section() {
            let alignment_ref_section =
                EtomoNumber::new_from_instance(Some(&meta_data.get_alignment_ref_section()));
            options.push("-r".to_string());
            options.push(alignment_ref_section.to_string());
            self.alignment_ref_section = Some(alignment_ref_section);
        }
        // Add optional size
        let size_in_x =
            ScriptParameter::new_from_instance(Some(&meta_data.get_size_in_x_parameter()));
        let size_in_y =
            ScriptParameter::new_from_instance(Some(&meta_data.get_size_in_y_parameter()));
        if size_in_x.is_not_null_and_not_default() || size_in_y.is_not_null_and_not_default() {
            options.push("-s".to_string());
            // both numbers must exist
            options.push(format!("{},{}", size_in_x, size_in_y));
        }
        self.size_in_x = Some(size_in_x);
        self.size_in_y = Some(size_in_y);
        // Add optional offset
        let shift_in_x =
            ScriptParameter::new_from_instance(Some(&meta_data.get_shift_in_x_parameter()));
        let shift_in_y =
            ScriptParameter::new_from_instance(Some(&meta_data.get_shift_in_y_parameter()));
        if shift_in_x.is_not_null_and_not_default() || shift_in_y.is_not_null_and_not_default() {
            options.push("-o".to_string());
            // both numbers must exist
            // offset is a negative shift
            options.push(format!(
                "{},{}",
                shift_in_x.get_int().wrapping_mul(-1),
                shift_in_y.get_int().wrapping_mul(-1)
            ));
        }
        self.shift_in_x = Some(shift_in_x);
        self.shift_in_y = Some(shift_in_y);
        let mut local_fits = EtomoBoolean2::new();
        if meta_data.is_local_fits() {
            local_fits.set_boolean(true);
            options.push("-l".to_string());
        }
        self.local_fits = Some(local_fits);
        if self.mode == Mode::MaxSize {
            options.push("-m".to_string());
        }
        if self.mode == Mode::Trial {
            options.push("-t".to_string());
            let use_every_n_slices =
                EtomoNumber::new_from_instance(Some(&meta_data.get_use_every_n_slices()));
            options.push(use_every_n_slices.to_string());
            self.use_every_n_slices = Some(use_every_n_slices);
            let binning =
                ScriptParameter::new_from_instance(Some(&meta_data.get_trial_binning_parameter()));
            if binning.is_not_null_and_not_default() {
                options.push("-b".to_string());
                options.push(binning.to_string());
            }
            self.binning = Some(binning);
        }
        options.push(self.root_name.clone().unwrap_or_else(|| "null".to_string()));
        // Fixed in translation: the source dereferences a null section table
        // (`sectionData.size()`), a NullPointerException; no rows are added here.
        let section_data = meta_data.get_section_table_data().unwrap_or_default();
        let section_data_size = section_data.len();
        let mut join_start_list = IntKeyList::get_string_instance();
        let mut join_end_list = IntKeyList::get_string_instance();
        for i in 0..section_data_size {
            let data: &dyn ConstSectionTableRowData = &*section_data[i];
            let join_final_start = data.get_join_final_start().to_string();
            let join_final_end = data.get_join_final_end().to_string();
            join_start_list.put_string(i as i32, Some(&join_final_start));
            join_end_list.put_string(i as i32, Some(&join_final_end));
            // both numbers must exist
            options.push(format!("{},{}", join_final_start, join_final_end));
        }
        self.join_start_list = Some(join_start_list);
        self.join_end_list = Some(join_end_list);
        options
    }

    /// Java private `genRejoinOptions()`.  For calls from the Model and Rejoin tabs.
    /// Generates the options using the state.  Saves refine start and end lists.
    fn gen_rejoin_options(&mut self, manager: &'static JoinManager) -> Vec<String> {
        let mut options: Vec<String> = Vec::new();
        // options.add("-P");
        let state = manager.get_state();
        let trial = state.get_refine_trial().is();
        if !state.get_join_alignment_ref_section(trial).is_null() {
            options.push("-r".to_string());
            options.push(state.get_join_alignment_ref_section(trial).to_string());
        }
        // Add optional size
        if state
            .get_join_size_in_x_parameter(trial)
            .is_not_null_and_not_default()
            || state
                .get_join_size_in_y_parameter(trial)
                .is_not_null_and_not_default()
        {
            options.push("-s".to_string());
            // both numbers must exist
            options.push(format!(
                "{},{}",
                state.get_join_size_in_x(trial),
                state.get_join_size_in_y(trial)
            ));
        }
        // Add optional offset
        if state
            .get_join_shift_in_x_parameter(trial)
            .is_not_null_and_not_default()
            || state
                .get_join_shift_in_y_parameter(trial)
                .is_not_null_and_not_default()
        {
            options.push("-o".to_string());
            // both numbers must exist
            // offset is a negative shift
            options.push(format!(
                "{},{}",
                state.get_join_shift_in_x(trial).get_int().wrapping_mul(-1),
                state.get_join_shift_in_y(trial).get_int().wrapping_mul(-1)
            ));
        }
        if state.is_join_local_fits(trial) {
            options.push("-l".to_string());
        }
        if self.mode == Mode::TrialRejoin {
            let meta_data = manager.get_const_meta_data();
            options.push("-t".to_string());
            let use_every_n_slices =
                EtomoNumber::new_from_instance(Some(&meta_data.get_rejoin_use_every_n_slices()));
            options.push(use_every_n_slices.to_string());
            self.use_every_n_slices = Some(use_every_n_slices);
            let rejoin_binning = ScriptParameter::new_from_instance(Some(
                &meta_data.get_rejoin_trial_binning_parameter(),
            ));
            if rejoin_binning.is_not_null_and_not_default() {
                options.push("-b".to_string());
                options.push(rejoin_binning.to_string());
            }
        }
        options.push("-gaps".to_string());
        options.push("-xform".to_string());
        options.push(dataset_files::get_refine_join_xg_file_name(manager));
        options.push(self.root_name.clone().unwrap_or_else(|| "null".to_string()));
        let meta_data = manager.get_const_meta_data();
        // Java's unused local `sectionData`.
        let _section_data = meta_data.get_section_table_data();
        // The first start and end: start comes from join final start and end comes
        // from the boundary table
        let mut start_list_walker = state.get_join_start_list_walker(trial);
        let end_list_walker = state.get_join_end_list_walker(trial);
        let gap_start_list = meta_data.get_boundary_row_start_list();
        let gap_end_list = meta_data.get_boundary_row_end_list();
        let mut gap_start_list_walker = gap_start_list.get_walker();
        let mut gap_end_list_walker = gap_end_list.get_walker();
        // Make sure that lists are valid
        let num_rows = start_list_walker.size();
        // Build the refine start and end list, so it can be sent to the join state
        let mut refine_start_list = IntKeyList::get_number_instance();
        let mut refine_end_list = IntKeyList::get_number_instance();
        // `IntKeyList.add(ConstEtomoNumber)`, which takes a null number as a null
        // value.
        let add_number = |list: &mut IntKeyList, number: &Option<EtomoNumber>| match number {
            Some(number) => list.add_etomo_number(number),
            None => list.add_string(None),
        };
        let display = |number: &Option<EtomoNumber>| match number {
            None => "null".to_string(),
            Some(number) => number.to_string(),
        };
        if num_rows >= 2
            && num_rows == end_list_walker.size()
            && num_rows == gap_start_list_walker.size() + 1
            && num_rows == gap_end_list_walker.size() + 1
        {
            // Add the first first and end pair. Start comes from join final start and
            // end comes from the boundary table
            let mut start = start_list_walker.next_etomo_number();
            let mut end = gap_end_list_walker.next_etomo_number();
            options.push(format!("{},{}", display(&start), display(&end)));
            // set the first key (first row index) from the join start list. Then add
            // the rows in order, incrementing the key by 1 each time
            refine_start_list.reset_with_start_key(1);
            refine_end_list.reset_with_start_key(1);
            add_number(&mut refine_start_list, &start);
            add_number(&mut refine_end_list, &end);
            // The middle start and end pairs come from the boundary table
            // while look should end when gapEndListWalker runs out of values (its
            // ahead of gapStartListWalker by 1).
            while gap_end_list_walker.has_next() {
                start = gap_start_list_walker.next_etomo_number();
                end = gap_end_list_walker.next_etomo_number();
                options.push(format!("{},{}", display(&start), display(&end)));
                add_number(&mut refine_start_list, &start);
                add_number(&mut refine_end_list, &end);
            }
            // The last start and end: start comes from the boundary table and end comes
            // from the last join final end.
            start = gap_start_list_walker.next_etomo_number();
            end = end_list_walker.get_last_etomo_number();
            options.push(format!("{},{}", display(&start), display(&end)));
            add_number(&mut refine_start_list, &start);
            add_number(&mut refine_end_list, &end);
        } else if num_rows >= 2
            && num_rows == end_list_walker.size()
            && num_rows > gap_start_list_walker.size() + 1
            && num_rows > gap_end_list_walker.size() + 1
        {
            ui_harness::open_message_dialog_from_process(
                Some(manager),
                "The dataset file may be corrupted.  If this process fails, exit and rerun \
                 Etomo, then go to the Join tab and run Finish Join to fix the problem.",
                "Etomo Warning",
                None,
            );
        }
        self.refine_start_list = Some(refine_start_list);
        self.refine_end_list = Some(refine_end_list);
        options
    }
}

impl Command for FinishjoinParam {
    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        AxisID::Only
    }

    /// Java `getCommandArray()`.
    fn get_command_array(&self) -> Option<Vec<String>> {
        Some(self.command_array.clone())
    }

    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        let mut buffer = String::new();
        for element in &self.command_array {
            buffer.push_str(&format!("{element} "));
        }
        Some(buffer)
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        Some(COMMAND_NAME.to_string())
    }

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        Some(PROCESS_NAME)
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        Some(COMMAND_NAME.to_string())
    }

    /// Java `getCommandOutputFile()`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        self.output_file.clone()
    }

    /// Java `getCommandInputFile()`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        None
    }

    /// Java `getCommandMode()`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        Some(&self.mode)
    }

    /// Java `isMessageReporter()`.
    fn is_message_reporter(&self) -> bool {
        false
    }

    /// Java `getSubcommandDetails()`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName()`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `getOutputImageFileType()` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<std::sync::Arc<FileType>> {
        match self.mode {
            Mode::FinishJoin => Some(std::sync::Arc::clone(&file_type::CLASS.join)),
            Mode::MaxSize => None,
            Mode::Trial => Some(std::sync::Arc::clone(&file_type::CLASS.trial_join)),
            Mode::Rejoin => Some(std::sync::Arc::clone(&file_type::CLASS.join)),
            Mode::TrialRejoin => Some(std::sync::Arc::clone(&file_type::CLASS.trial_join)),
            Mode::SuppressExecution => None,
        }
    }

    /// Java `getOutputImageFileKey()`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        match self.mode {
            Mode::FinishJoin => Some(FileKey::clone(&file_type::CLASS.join)),
            Mode::MaxSize => None,
            Mode::Trial => Some(FileKey::clone(&file_type::CLASS.trial_join)),
            Mode::Rejoin => Some(FileKey::clone(&file_type::CLASS.join)),
            Mode::TrialRejoin => Some(FileKey::clone(&file_type::CLASS.trial_join)),
            Mode::SuppressExecution => None,
        }
    }

    /// Java `getOutputImageFileType2()` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<std::sync::Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2()`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `command instanceof ProcessDetails`: this is a `CommandDetails`.
    fn get_process_details(&self) -> Option<&dyn ProcessDetails> {
        Some(self)
    }
}

impl Loggable for FinishjoinParam {
    /// Java `getName()`.
    fn get_name(&self) -> String {
        COMMAND_NAME.to_string()
    }

    /// Java `getLogMessage()`, which returns null; there is no message.
    fn get_log_message(&self) -> Result<Vec<Option<String>>, LoggableException> {
        Ok(Vec::new())
    }
}

/// Every getter the source does not answer throws `IllegalArgumentException("field="
/// + field)`.  Fixed in translation: the value is unavailable (`None`).
impl ProcessDetails for FinishjoinParam {
    /// Java `getEtomoNumber(FieldInterface)`.
    fn get_etomo_number(&self, field: &dyn FieldInterface) -> Option<ConstEtomoNumber> {
        // `ScriptParameter` field read as `ConstEtomoNumber`.
        let as_const_etomo_number = |field: &Option<ScriptParameter>| {
            field.as_ref().map(|field| {
                let number: &ConstEtomoNumber = field;
                number.clone()
            })
        };
        match field_interface::as_field::<Fields>(field) {
            Some(Fields::AlignmentRefSection) => self.alignment_ref_section.as_ref().map(|field| {
                let number: &ConstEtomoNumber = field;
                number.clone()
            }),
            Some(Fields::SizeInX) => as_const_etomo_number(&self.size_in_x),
            Some(Fields::SizeInY) => as_const_etomo_number(&self.size_in_y),
            Some(Fields::ShiftInX) => as_const_etomo_number(&self.shift_in_x),
            Some(Fields::ShiftInY) => as_const_etomo_number(&self.shift_in_y),
            Some(Fields::Binning) => as_const_etomo_number(&self.binning),
            Some(Fields::UseEveryNSlices) => self.use_every_n_slices.as_ref().map(|field| {
                let number: &ConstEtomoNumber = field;
                number.clone()
            }),
            Some(Fields::LocalFits) => self.local_fits.as_ref().map(|field| {
                let number: &ConstEtomoNumber = field;
                number.clone()
            }),
            _ => None,
        }
    }

    /// Java `getIntKeyList(FieldInterface)`.
    fn get_int_key_list(&self, field: &dyn FieldInterface) -> Option<IntKeyList> {
        match field_interface::as_field::<Fields>(field) {
            Some(Fields::JoinStartList) => self.join_start_list.clone(),
            Some(Fields::JoinEndList) => self.join_end_list.clone(),
            Some(Fields::RefineStartList) => self.refine_start_list.clone(),
            Some(Fields::RefineEndList) => self.refine_end_list.clone(),
            _ => None,
        }
    }

    /// Java `getString(FieldInterface)`.
    fn get_string(&self, _field: &dyn FieldInterface) -> Option<String> {
        None
    }

    /// Java `getIntValue(FieldInterface)`.
    fn get_int_value(&self, _field: &dyn FieldInterface) -> Option<i32> {
        None
    }

    /// Java `getIteratorElementList(FieldInterface)`.
    fn get_iterator_element_list(
        &self,
        _field: &dyn FieldInterface,
    ) -> Option<crate::imod::etomo::r#type::iterator_element_list::IteratorElementList> {
        None
    }

    /// Java `getBooleanValue(FieldInterface)`.
    fn get_boolean_value(&self, _field: &dyn FieldInterface) -> Option<bool> {
        None
    }

    /// Java `getStringArray(FieldInterface)`.
    fn get_string_array(&self, _field: &dyn FieldInterface) -> Option<Vec<String>> {
        None
    }

    /// Java `getHashtable(FieldInterface)`.
    fn get_hashtable(&self, _field: &dyn FieldInterface) -> Option<Hashtable> {
        None
    }

    /// Java `getDoubleValue(FieldInterface)`.
    fn get_double_value(&self, _field: &dyn FieldInterface) -> Option<f64> {
        None
    }
}
