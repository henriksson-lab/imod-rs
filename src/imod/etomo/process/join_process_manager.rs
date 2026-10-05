//! `IMOD/Etomo/src/etomo/process/JoinProcessManager.java`.
//!
//! The process manager of the Join interface (`JoinManager`).  It embeds the
//! `BaseProcessManager` superclass as `base` and installs itself as the base's
//! [`BaseProcessManagerHooks`] for its `postProcess`/`errorProcess` overrides.
//!
//! **Threads.**  The overrides run on the process thread.  The join state is shared
//! (its fields carry their own locks) and is updated there, as in the source; the
//! calls that reach the join dialog (`setMode`, `addSection`, `setSize`, `setShift`,
//! `updateJoinDialogDisplay`, `postProcess`, `abortAddSection`) are made on the event
//! dispatch thread, in the source's order.

use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use regex::Regex;

use super::background_process::{BackgroundProcess, BackgroundProcessInit};
use super::base_process_manager::{AxisBusyException, BaseProcessManager, BaseProcessManagerHooks};
use super::com_script_process::ComScriptProcess;
use super::process_interface::{ProcessSeriesRef, SystemProcessInterface as _};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::clip_param::{self, ClipParam};
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::command_mode;
use crate::imod::etomo::comscript::finishjoin_param::{self, FinishjoinParam};
use crate::imod::etomo::comscript::joinwarp2model_param::Joinwarp2modelParam;
use crate::imod::etomo::comscript::makejoincom_param::{self, MakejoincomParam};
use crate::imod::etomo::comscript::remapmodel_param::RemapmodelParam;
use crate::imod::etomo::comscript::start_join_param::{self, StartJoinParam};
use crate::imod::etomo::comscript::xfjointomo_param::XfjointomoParam;
use crate::imod::etomo::comscript::xfmodel_param;
use crate::imod::etomo::comscript::xftoxg_param::XftoxgParam;
use crate::imod::etomo::join_manager::JoinManager;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::xfjointomo_log::XfjointomoLog;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::join_state::JoinState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::event_queue;
use crate::imod::etomo::util::utilities;

/// Java private static final `START_JOIN_COMSCRIPT_NAME`.
const START_JOIN_COMSCRIPT_NAME: &str = "startjoin.com";

/// Runs `job` on the event dispatch thread: at once when already there (the source's
/// synchronous call), else posted.  Rust-only threading plumbing (see the module
/// header).
fn on_edt(job: impl FnOnce() + Send + 'static) {
    if event_queue::is_dispatch_thread() {
        job();
    } else {
        event_queue::invoke_later(job);
    }
}

/// Java `public final class JoinProcessManager extends BaseProcessManager`.
pub struct JoinProcessManager {
    /// The `BaseProcessManager` superclass.
    pub base: BaseProcessManager,
    /// Java private final `state`.
    state: &'static JoinState,
    /// Java private final `manager`.
    manager: &'static JoinManager,
}

impl JoinProcessManager {
    /// Java `JoinProcessManager(JoinManager, JoinState)`.  The manager keeps it for
    /// the run, and the base class's start functions take `&'static self`.
    pub fn new(join_mgr: &'static JoinManager, state: &'static JoinState) -> &'static Self {
        let process_manager: &'static JoinProcessManager = Box::leak(Box::new(Self {
            base: BaseProcessManager::new(join_mgr),
            state,
            manager: join_mgr,
        }));
        process_manager.base.set_hooks(process_manager);
        process_manager
    }

    // <p>Updates done</p>

    /// Java `makejoincom(MakejoincomParam, ProcessSeries) throws AxisBusyException`.
    /// Run makejoincom.
    pub fn makejoincom(
        &'static self,
        makejoincom_param: Arc<MakejoincomParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            makejoincom_param as Arc<dyn Command + Send + Sync>,
            true,
            AxisID::Only,
            Some(ProcessName::MAKEJOINCOM),
            None,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java `remapmodel(RemapmodelParam, ProcessSeries) throws AxisBusyException`.
    pub fn remapmodel(
        &'static self,
        param: Arc<RemapmodelParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            param as Arc<dyn Command + Send + Sync>,
            false,
            AxisID::Only,
            Some(ProcessName::REMAPMODEL),
            None,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java `xftoxg(XftoxgParam, ProcessSeries) throws AxisBusyException`.
    pub fn xftoxg(
        &'static self,
        param: Arc<XftoxgParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            param as Arc<dyn Command + Send + Sync>,
            false,
            AxisID::Only,
            Some(ProcessName::XFTOXG),
            None,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java `xfmodel(XfmodelParam, AxisID, ProcessResultDisplay, ProcessSeries)`, the
    /// inherited `BaseProcessManager` member.
    pub fn xfmodel(
        &'static self,
        param: Arc<dyn Command + Send + Sync>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        self.base.xfmodel(param, axis_id, None, process_series)
    }

    /// Java `createNewFile(String)`, the inherited `BaseProcessManager` member.
    pub fn create_new_file(&self, absolute_path: &str) {
        self.base.create_new_file(absolute_path);
    }

    /// Java `saveFinishjoinState(FinishjoinParam, ProcessSeries)`.  Run the post
    /// process functionality for finishjoin without running the process.
    pub fn save_finishjoin_state(
        &'static self,
        finishjoin_param: Arc<FinishjoinParam>,
        process_series: Option<ProcessSeriesRef>,
    ) {
        let mut init = BackgroundProcessInit::new(
            self.manager,
            &self.base,
            AxisID::Only,
            Some(ProcessName::FINISHJOIN),
            process_series,
        );
        init.command = Some(finishjoin_param as Arc<dyn Command + Send + Sync>);
        init.is_command_details = true;
        let background_process = BackgroundProcess::get_instance(init);
        self.post_process_background(&self.base, &background_process);
    }

    /// Java `finishjoin(FinishjoinParam, ProcessSeries) throws AxisBusyException`.
    /// Run finishjoin.
    pub fn finishjoin(
        &'static self,
        finishjoin_param: Arc<FinishjoinParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            finishjoin_param as Arc<dyn Command + Send + Sync>,
            true,
            AxisID::Only,
            Some(ProcessName::FINISHJOIN),
            None,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java `xfjointomo(XfjointomoParam, ProcessSeries) throws AxisBusyException`.
    pub fn xfjointomo(
        &'static self,
        xfjointomo_param: &mut XfjointomoParam,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        XfjointomoLog::get_instance(self.manager, AxisID::Only).reset();
        let background_process = self.base.start_background_process_array_display(
            xfjointomo_param.get_command_array(),
            AxisID::Only,
            None,
            Some(ProcessName::XFJOINTOMO),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `rotx(ClipParam, ProcessSeries) throws AxisBusyException`.  Run clip rotx.
    pub fn rotx(
        &'static self,
        clipyz_param: Arc<ClipParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            clipyz_param as Arc<dyn Command + Send + Sync>,
            true,
            AxisID::Only,
            Some(ProcessName::CLIP),
            None,
            process_series,
            false,
            true, // POPUP_CHUNK_WARNINGS_DEFAULT
        )?;
        Ok(background_process.get_name())
    }

    /// Java `startjoin(StartJoinParam, ProcessSeries) throws AxisBusyException`.  Run
    /// the startjoin com file.
    pub fn startjoin(
        &'static self,
        param: Arc<StartJoinParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            param as Arc<dyn Command + Send + Sync>,
            true,
            None,
            AxisID::Only,
            None,
            process_series,
            None,
        )?;
        Ok(com_script_process.get_name())
    }

    /// Java package-private `getManager()`.
    pub fn get_manager(&self) -> &'static dyn BaseManager {
        self.manager
    }

    /// Java `joinwarp2model(Joinwarp2modelParam, ProcessSeries) throws
    /// AxisBusyException`.  Run joinwarp2model.  The param is not read (it was saved to
    /// the com file by the caller).
    pub fn joinwarp2model(
        &'static self,
        _param: &Joinwarp2modelParam,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_file_type(
            &file_type::CLASS
                .join_warp_2_model_comscript
                .get_file_name(Some(self.manager), Some(AxisID::Only))
                .unwrap_or_else(|| "null".to_string()),
            AxisID::Only,
            process_series,
            Some(&file_type::CLASS.join_warp_2_model_comscript),
        )?;
        Ok(com_script_process.get_name())
    }

    /// Java `pause(AxisID)`, the inherited `BaseProcessManager` member.
    pub fn pause(&self, axis_id: AxisID) -> bool {
        self.base.pause(axis_id)
    }
}

/// The `\\s+` pattern of the source's `line.split("\\s+")` (Java's `\s` is ASCII
/// whitespace).
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?-u:\s)+").unwrap());

impl BaseProcessManagerHooks for JoinProcessManager {
    /// Java override `postProcess(ComScriptProcess)`.
    fn post_process_com_script(&self, _base: &BaseProcessManager, process: &ComScriptProcess) {
        let command_name = process.get_com_script_name();
        let process_details = process
            .get_command_details()
            .and_then(|details| details.get_process_details());
        if command_name == START_JOIN_COMSCRIPT_NAME {
            self.state.set_sample_produced(true);
            let manager = self.manager;
            on_edt(move || {
                manager.set_mode();
            });
            // Fixed in translation: a com script started without its details makes
            // the source dereference null (`processDetails.getBooleanValue`); nothing
            // more is saved here.
            if let Some(process_details) = process_details
                && process_details
                    .get_boolean_value(&start_join_param::Fields::Rotate)
                    .unwrap_or(false)
            {
                self.state.set_total_rows(
                    process_details
                        .get_int_value(&start_join_param::Fields::TotalRows)
                        .unwrap_or(0),
                );
                self.state.set_rotation_angles_list(
                    process_details.get_hashtable(&start_join_param::Fields::RotationAnglesList),
                );
            }
        }
    }

    /// Java override `postProcess(BackgroundProcess)`.  Non-generic post processing
    /// for a successful BackgroundProcess.
    fn post_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_background_base(process);
        let Some(command_name) = process.get_command_name() else {
            return;
        };
        let command_details = process.get_command_details().cloned();
        let process_details = process
            .get_command_details()
            .and_then(|details| details.get_process_details());
        let command = process.get_command();
        let manager = self.manager;
        if command_name == clip_param::PROCESS_NAME.to_string() {
            let Some(command) = command else {
                return;
            };
            let output_file = command.get_command_output_file();
            on_edt(move || manager.add_section(output_file.as_deref()));
        } else if command_name == finishjoin_param::COMMAND_NAME {
            let Some(command) = command else {
                return;
            };
            let mode = command.get_command_mode();
            if command_mode::equals_mode(mode, &finishjoin_param::Mode::MaxSize) {
                let std_output = process.get_std_output();
                if let Some(std_output) = std_output {
                    for line in std_output.iter() {
                        if line.contains(finishjoin_param::SIZE_TAG) {
                            let line_array = utilities::java_lang_string_split(line, &WHITESPACE);
                            let size_in_x =
                                line_array.get(finishjoin_param::SIZE_IN_X_INDEX).cloned();
                            let size_in_y =
                                line_array.get(finishjoin_param::SIZE_IN_Y_INDEX).cloned();
                            on_edt(move || {
                                manager.set_size(size_in_x.as_deref(), size_in_y.as_deref())
                            });
                        } else if line.contains(finishjoin_param::OFFSET_TAG) {
                            let line_array = utilities::java_lang_string_split(line, &WHITESPACE);
                            let shift_in_x = FinishjoinParam::get_shift(
                                line_array
                                    .get(finishjoin_param::OFFSET_IN_X_INDEX)
                                    .map(String::as_str),
                            );
                            let shift_in_y = FinishjoinParam::get_shift(
                                line_array
                                    .get(finishjoin_param::OFFSET_IN_Y_INDEX)
                                    .map(String::as_str),
                            );
                            match (shift_in_x, shift_in_y) {
                                (Ok(shift_in_x), Ok(shift_in_y)) => {
                                    on_edt(move || manager.set_shift(shift_in_x, shift_in_y))
                                }
                                // An uncaught IllegalArgumentException on the process
                                // thread: its handler prints it.
                                (Err(message), _) | (_, Err(message)) => {
                                    eprintln!("java.lang.IllegalArgumentException: {message}")
                                }
                            }
                        }
                    }
                }
                return;
            } else if command_mode::equals_mode(mode, &finishjoin_param::Mode::Trial)
                || command_mode::equals_mode(mode, &finishjoin_param::Mode::FinishJoin)
            {
                if let Some(process_details) = process_details {
                    let trial = command_mode::equals_mode(mode, &finishjoin_param::Mode::Trial);
                    self.state.set_join_alignment_ref_section(
                        trial,
                        process_details
                            .get_etomo_number(&finishjoin_param::Fields::AlignmentRefSection)
                            .as_ref(),
                    );
                    self.state.set_join_size_in_x(
                        trial,
                        process_details
                            .get_etomo_number(&finishjoin_param::Fields::SizeInX)
                            .as_ref(),
                    );
                    self.state.set_join_size_in_y(
                        trial,
                        process_details
                            .get_etomo_number(&finishjoin_param::Fields::SizeInY)
                            .as_ref(),
                    );
                    self.state.set_join_shift_in_x(
                        trial,
                        process_details
                            .get_etomo_number(&finishjoin_param::Fields::ShiftInX)
                            .as_ref(),
                    );
                    self.state.set_join_shift_in_y(
                        trial,
                        process_details
                            .get_etomo_number(&finishjoin_param::Fields::ShiftInY)
                            .as_ref(),
                    );
                    self.state.set_join_local_fits(
                        trial,
                        process_details
                            .get_etomo_number(&finishjoin_param::Fields::LocalFits)
                            .as_ref(),
                    );
                    let join_start_list =
                        process_details.get_int_key_list(&finishjoin_param::Fields::JoinStartList);
                    self.state.set_join_start_list(
                        trial,
                        join_start_list
                            .as_ref()
                            .map(|list| list as &dyn crate::imod::etomo::r#type::const_int_key_list::ConstIntKeyList),
                    );
                    let join_end_list =
                        process_details.get_int_key_list(&finishjoin_param::Fields::JoinEndList);
                    self.state.set_join_end_list(
                        trial,
                        join_end_list
                            .as_ref()
                            .map(|list| list as &dyn crate::imod::etomo::r#type::const_int_key_list::ConstIntKeyList),
                    );
                    self.state.set_current_join_version(trial);
                    if command_mode::equals_mode(mode, &finishjoin_param::Mode::Trial) {
                        self.state.set_join_trial_binning(
                            process_details
                                .get_etomo_number(&finishjoin_param::Fields::Binning)
                                .as_ref(),
                        );
                        self.state.set_join_trial_use_every_n_slices(
                            process_details
                                .get_etomo_number(&finishjoin_param::Fields::UseEveryNSlices)
                                .as_ref(),
                        );
                    }
                }
                on_edt(move || manager.update_join_dialog_display());
            } else if command_mode::equals_mode(mode, &finishjoin_param::Mode::Rejoin)
                || command_mode::equals_mode(mode, &finishjoin_param::Mode::SuppressExecution)
            {
                // Fixed in translation: a process without its details makes the source
                // dereference null; the lists are not changed here.
                if let Some(process_details) = process_details {
                    let refine_start_list = process_details
                        .get_int_key_list(&finishjoin_param::Fields::RefineStartList);
                    self.state
                        .set_refine_start_list(refine_start_list.as_ref().map(|list| {
                        list as &dyn crate::imod::etomo::r#type::const_int_key_list::ConstIntKeyList
                    }));
                    let refine_end_list =
                        process_details.get_int_key_list(&finishjoin_param::Fields::RefineEndList);
                    self.state
                        .set_refine_end_list(refine_end_list.as_ref().map(|list| {
                        list as &dyn crate::imod::etomo::r#type::const_int_key_list::ConstIntKeyList
                    }));
                }
                on_edt(move || manager.update_join_dialog_display());
            } else if command_mode::equals_mode(mode, &finishjoin_param::Mode::TrialRejoin) {
                if let Some(process_details) = process_details {
                    self.state.set_refine_trial_use_every_n_slices(
                        process_details
                            .get_etomo_number(&finishjoin_param::Fields::UseEveryNSlices)
                            .as_ref(),
                    );
                }
            }
        } else if command_name == makejoincom_param::COMMAND_NAME {
            on_edt(move || manager.post_process(Some(&command_name), command_details.as_ref()));
        } else if command_name == ProcessName::XFJOINTOMO.to_string() {
            base.write_log_file(
                process,
                process.get_axis_id(),
                &dataset_files::get_log_name(
                    manager,
                    Some(process.get_axis_id()),
                    process.get_process_name().as_ref(),
                ),
            );
            match XfjointomoLog::get_instance(manager, AxisID::Only).gaps_exist() {
                Ok(gaps_exist) => self.state.set_gaps_exist(gaps_exist),
                Err(LogFileError::Lock(_)) => {
                    // if not sure whether gaps exist, run remapmodel
                    self.state.set_gaps_exist(true);
                }
                Err(e) => {
                    eprintln!("{e:?}");
                    // if not sure whether gaps exist, run remapmodel
                    self.state.set_gaps_exist(true);
                }
            }
            on_edt(move || {
                manager.post_process(Some(&command_name), command_details.as_ref());
                manager.update_join_dialog_display();
            });
        } else if command_name == *xfmodel_param::COMMAND_NAME {
            self.state.set_xf_model_output_file(
                process_details
                    .and_then(|details| details.get_string(&xfmodel_param::Fields::OutputFile))
                    .as_deref(),
            );
        }
    }

    /// Java override `errorProcess(BackgroundProcess)`.
    ///
    /// Fixed in translation (JoinProcessManager.java:255): the source compares the
    /// command name with the `ProcessName` object (`commandName.equals(
    /// ClipParam.PROCESS_NAME)`), which is never equal to a `String`, so a failed
    /// `clip rotx` never deleted its partial output or re-enabled Add Section.  The
    /// command name is compared with the process name's string here.
    fn error_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        let Some(command_name) = process.get_command_name() else {
            return;
        };
        let manager = self.manager;
        if command_name == makejoincom_param::COMMAND_NAME {
            self.state.set_sample_produced(false);
            on_edt(move || {
                manager.set_mode();
            });
        } else if command_name == clip_param::PROCESS_NAME.to_string() {
            let Some(command) = process.get_command() else {
                return;
            };
            let output_file: Option<PathBuf> = command.get_command_output_file();
            // A partially created flip file can cause an error when it is opened.
            if let Some(output_file) = output_file {
                let _ = std::fs::remove_file(output_file);
            }
            on_edt(move || manager.abort_add_section());
        } else if command_name == ProcessName::XFJOINTOMO.to_string() {
            base.write_log_file(
                process,
                process.get_axis_id(),
                &dataset_files::get_log_name(
                    manager,
                    Some(process.get_axis_id()),
                    process.get_process_name().as_ref(),
                ),
            );
        }
    }

    /// Java override `errorProcess(ComScriptProcess)`.
    fn error_process_com_script(&self, _base: &BaseProcessManager, process: &ComScriptProcess) {
        let command_name = process.get_com_script_name();
        if command_name == START_JOIN_COMSCRIPT_NAME {
            self.state.set_sample_produced(false);
            let manager = self.manager;
            on_edt(move || {
                manager.set_mode();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_whitespace_split_keeps_the_leading_empty_token() {
        assert_eq!(
            utilities::java_lang_string_split("Maximum size required:   1024   980", &WHITESPACE),
            vec!["Maximum", "size", "required:", "1024", "980"]
        );
        assert_eq!(
            utilities::java_lang_string_split("  Offset needed to center:  -3  4", &WHITESPACE),
            vec!["", "Offset", "needed", "to", "center:", "-3", "4"]
        );
    }
}
