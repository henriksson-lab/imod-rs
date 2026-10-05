//! `IMOD/Etomo/src/etomo/process/ProcessManager.java`.
//!
//! This object manages the execution of com scripts in the background and the
//! opening and sending messages to imod.  It also provides an interface to
//! executing some simple command sequences.
//!
//! **Shape.**  `ProcessManager extends BaseProcessManager`: the base is the
//! embedded [`ProcessManager::base`], the overridden `postProcess`/
//! `errorProcess` hooks are this struct's [`BaseProcessManagerHooks`], and the
//! object lives for the program (the `ApplicationManager` keeps it), so the
//! processes it starts hold `&'static` references to it.
//!
//! Parameter objects arrive as `Arc`s: the process thread keeps reading them
//! (`Command`), as in the Java.  A Java `ConstXxxParam` argument is the
//! concrete `XxxParam` here.

use crate::imod::etomo::process::process_messages::MessagesArray;
use super::align_log_generator::{self, AlignLogGenerator};
use super::background_process::BackgroundProcess;
use super::base_process_manager::{
    AxisBusyException, BaseProcessManager, BaseProcessManagerHooks, SystemProcessException,
};
use super::blendmont_process_monitor::BlendmontProcessMonitor;
use super::ccd_eraser_process_monitor::CCDEraserProcessMonitor;
use super::com_script_process::ComScriptProcess;
use super::combine_process_monitor::CombineProcessMonitor;
use super::ctf_correction_monitor::CtfCorrectionMonitor;
use super::matchvol1_process_monitor::Matchvol1ProcessMonitor;
use super::monitor::ProcessMonitor;
use super::monitor::{Monitor, ProcessMonitor as _};
use super::mtffilter_process_monitor::MtffilterProcessMonitor;
use super::newst_process_monitor::NewstProcessMonitor;
use super::prenewst_process_monitor::PrenewstProcessMonitor;
use super::process_data::ProcessData;
use super::process_interface::{ProcessResultDisplayRef, ProcessSeriesRef, SystemProcessInterface};
use super::process_messages::{MessageType, ProcessMessages};
use super::process_output_strings;
use super::processchunks_volcombine_monitor::ProcesschunksVolcombineMonitor;
use super::reconnect_process::ReconnectProcess;
use super::system_program::SystemProgram;
use super::tilt_process_monitor::TiltProcessMonitor;
use super::tilt3d_find_process_monitor::Tilt3dFindProcessMonitor;
use super::tiltxcorr_process_watcher::TiltxcorrProcessWatcher;
use super::xcorr_process_watcher::XcorrProcessWatcher;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::comscript::alt_tomo_setup_param::{self, AltTomoSetupParam};
use crate::imod::etomo::comscript::archiveorig_param::{self, ArchiveorigParam};
use crate::imod::etomo::comscript::autofidseed_param::AutofidseedParam;
use crate::imod::etomo::comscript::batchruntomo_param::{self, BatchruntomoParam};
use crate::imod::etomo::comscript::beadtrack_param::{self, BeadtrackParam};
use crate::imod::etomo::comscript::blendmont_param::{self, BlendmontParam};
use crate::imod::etomo::comscript::ccd_eraser_param::{self, CCDEraserParam};
use crate::imod::etomo::comscript::clip_param::{self, ClipParam};
use crate::imod::etomo::comscript::combine_comscript_state::{self, CombineComscriptState};
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::command_mode::{CommandMode, equals_mode};
use crate::imod::etomo::comscript::const_ctf_phase_flip_param::ConstCtfPhaseFlipParam;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::const_split_correction_param::ConstSplitCorrectionParam;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::const_tiltalign_param;
use crate::imod::etomo::comscript::copy_tomo_coms::{self, CopyTomoComs};
use crate::imod::etomo::comscript::cryo_position_param::CryoPositionParam;
use crate::imod::etomo::comscript::ctf_phase_flip_param::CtfPhaseFlipParam;
use crate::imod::etomo::comscript::ctf3d_setup_param::Ctf3dSetupParam;
use crate::imod::etomo::comscript::extractmagrad_param::ExtractmagradParam;
use crate::imod::etomo::comscript::extractpieces_param::ExtractpiecesParam;
use crate::imod::etomo::comscript::extracttilts_param::ExtracttiltsParam;
use crate::imod::etomo::comscript::field_interface::FieldInterface;
use crate::imod::etomo::comscript::find_beads3d_param::FindBeads3dParam;
use crate::imod::etomo::comscript::find_section_param::FindSectionParam;
use crate::imod::etomo::comscript::flatten_warp_param::FlattenWarpParam;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::comscript::mtf_filter_param::MTFFilterParam;
use crate::imod::etomo::comscript::multifilt_setup_param::MultifiltSetupParam;
use crate::imod::etomo::comscript::newst_param::{self, NewstParam};
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::comscript::reduce_filt_vol_param::{self, ReduceFiltVolParam};
use crate::imod::etomo::comscript::restrictalign_param::RestrictalignParam;
use crate::imod::etomo::comscript::setup_combine::SetupCombine;
use crate::imod::etomo::comscript::sirtsetup_param::{self, SirtsetupParam};
use crate::imod::etomo::comscript::splitcombine_param::SplitcombineParam;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::squeezevol_param::{self, SqueezevolParam};
use crate::imod::etomo::comscript::subtomo_setup_param::SubtomoSetupParam;
use crate::imod::etomo::comscript::tilt_param::{self, TiltParam};
use crate::imod::etomo::comscript::tiltalign_param::TiltalignParam;
use crate::imod::etomo::comscript::tiltxcorr_param::TiltxcorrParam;
use crate::imod::etomo::comscript::transferfid_param::TransferfidParam;
use crate::imod::etomo::comscript::trimvol_param::{self, TrimvolParam};
use crate::imod::etomo::comscript::warp_vol_param::{self, WarpVolParam};
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::autofidseed_log::AutofidseedLog;
use crate::imod::etomo::storage::flatten_warp_log::FlattenWarpLog;
use crate::imod::etomo::storage::log_file::{Handle, LogFile, LogFileError};
use crate::imod::etomo::storage::track_log::TrackLog;
use crate::imod::etomo::storage::transfer_fid_log::TransferFidLog;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::pos_sample_type::PosSampleType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::run_type::RunType;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::swing::parallel_progress_display::ParallelProgressDisplay;
use crate::imod::etomo::ui::swing::text_page_window::TextPageWindow;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::swing::ui_parameters;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::event_queue;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

/// Java final `ProcessManager extends BaseProcessManager`.
pub struct ProcessManager {
    /// The `BaseProcessManager` superclass.
    pub base: BaseProcessManager,
    /// Java `appManager`, the base class's `manager` cast to its concrete type.
    app_manager: &'static ApplicationManager,
    /// Java `transferfidCommandLine`: save the transferfid command line so
    /// that we can identify when process is complete.
    transferfid_command_line: Mutex<Option<String>>,
}

/// The errors `runCommand` and its callers declare
/// (`SystemProcessException, LogFileException, IOException, LockException`).
#[derive(Debug)]
pub enum RunCommandError {
    SystemProcess(SystemProcessException),
    LogFile(LogFileError),
}

impl std::fmt::Display for RunCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunCommandError::SystemProcess(e) => write!(f, "{e}"),
            RunCommandError::LogFile(e) => write!(f, "{}", e.get_message()),
        }
    }
}

impl From<LogFileError> for RunCommandError {
    fn from(e: LogFileError) -> RunCommandError {
        RunCommandError::LogFile(e)
    }
}

/// `ComScriptProcess.getName()` of a started com script.
fn name_of(process: Arc<ComScriptProcess>) -> String {
    process.get_name()
}

/// A param as the `Command` a process thread reads.
fn command_of<P: Command + Send + Sync + 'static>(
    param: &Arc<P>,
) -> Arc<dyn Command + Send + Sync> {
    Arc::clone(param) as Arc<dyn Command + Send + Sync>
}

/// A monitor as the `ProcessMonitor` the start functions take.
fn monitor_of<M: ProcessMonitor + 'static>(monitor: &Arc<M>) -> Option<Arc<dyn ProcessMonitor>> {
    Some(Arc::clone(monitor) as Arc<dyn ProcessMonitor>)
}

impl ProcessManager {
    /// Java `ProcessManager(ApplicationManager)`.  The manager keeps it for the
    /// program's lifetime, as the Java does.
    pub fn new(app_mgr: &'static ApplicationManager) -> &'static ProcessManager {
        let process_manager: &'static ProcessManager = Box::leak(Box::new(ProcessManager {
            base: BaseProcessManager::new(app_mgr),
            app_manager: app_mgr,
            transferfid_command_line: Mutex::new(None),
        }));
        process_manager.base.set_hooks(process_manager);
        process_manager
    }

    /// Java `setupCtfPlotterComScript`.
    pub fn setup_ctf_plotter_com_script(
        &self,
        ctf_phase_flip_param: &CtfPhaseFlipParam,
        axis_id: AxisID,
    ) {
        let mut copy_tomo_coms = CopyTomoComs::new(self.app_manager, false, false);
        copy_tomo_coms.set_ctf_files(copy_tomo_coms::CtfFilesValue::CtfPlotter);
        copy_tomo_coms.set_voltage(Some(ctf_phase_flip_param.get_voltage()));
        copy_tomo_coms
            .set_spherical_aberration(Some(ctf_phase_flip_param.get_spherical_aberration()));
        self.setup_com_scripts_private(&mut copy_tomo_coms, axis_id, None);
    }

    /// Java `setupCtfCorrectionComScript`.
    pub fn setup_ctf_correction_com_script(&self, axis_id: AxisID) {
        let mut copy_tomo_coms = CopyTomoComs::new(self.app_manager, false, false);
        copy_tomo_coms.set_ctf_files(copy_tomo_coms::CtfFilesValue::CtfCorrection);
        self.setup_com_scripts_private(&mut copy_tomo_coms, axis_id, None);
    }

    /// Java `setupComScripts(AxisID, CopyTomoComs, AxisType)`.
    pub fn setup_com_scripts<'a>(
        &self,
        axis_id: AxisID,
        param: &'a mut CopyTomoComs,
        axis_type: Option<AxisType>,
    ) -> Option<MutexGuard<'a, ProcessMessages>> {
        if etomo_director::ARGUMENTS.lock().unwrap().is_debug() {
            eprintln!("copytomocoms command line: {}", param.get_command_line());
        }
        self.app_manager.save_storables(Some(axis_id));
        self.setup_com_scripts_private(param, axis_id, axis_type)
    }

    /// Java private `setupComScripts(CopyTomoComs, AxisID, AxisType)`: run the
    /// copytomocoms script.
    ///
    /// Java returns `copytomocoms`'s own `ProcessMessages`; the Rust messages live
    /// behind the `SystemProgram`'s lock, so the guard is returned.
    ///
    /// Upstream bug fixed in translation (ProcessManager.java, `setupComScripts`):
    /// `getProcessMessages()` is null when copytomocoms was never built, and
    /// `messages.isEmpty` then throws NullPointerException.  Here that returns null.
    fn setup_com_scripts_private<'a>(
        &self,
        copy_tomo_coms: &'a mut CopyTomoComs,
        axis_id: AxisID,
        axis_type: Option<AxisType>,
    ) -> Option<MutexGuard<'a, ProcessMessages>> {
        if !copy_tomo_coms.setup() {
            return None;
        }
        let copy_tomo_coms: &'a CopyTomoComs = copy_tomo_coms;
        let exit_value = copy_tomo_coms.run();
        // process messages
        let messages = copy_tomo_coms.get_process_messages()?;
        if !messages.is_empty(Some(MessageType::Info)) {
            // smallest signed 16-bit integer (amount that needs to be added to
            // make everything positive).
            let info_value = "32768";
            for i in 0..messages.size(MessageType::Info) {
                if messages
                    .get(MessageType::Info, i)
                    .is_some_and(|message| message.contains(info_value))
                {
                    self.app_manager
                        .get_meta_data()
                        .set_gen_log(AxisID::First, Some(info_value));
                    if axis_type == Some(AxisType::DualAxis) {
                        self.app_manager
                            .get_meta_data()
                            .set_gen_log(AxisID::Second, Some(info_value));
                    }
                }
            }
        }
        if !messages.is_empty(Some(MessageType::Error)) {
            let mut error_message = String::from("Error running Copytomocoms");
            for i in 0..messages.size(MessageType::Error) {
                error_message.push_str(&format!(
                    "\n{}",
                    messages.get(MessageType::Error, i).unwrap_or("null")
                ));
            }
            self.open_message_dialog(&error_message, "Copytomocoms Error", axis_id);
        }
        for i in 0..messages.size(MessageType::Warning) {
            self.open_message_dialog(
                messages.get(MessageType::Warning, i).unwrap_or("null"),
                "Copytomocoms Warning",
                axis_id,
            );
        }
        if exit_value != 0 {
            if let Some(std_error_string) = copy_tomo_coms.get_std_error_string()
                && !std_error_string.is_empty()
            {
                self.open_message_dialog(&std_error_string, "Copytomocoms Error", axis_id);
            }
            return None;
        }
        Some(messages)
    }

    /// `UIHarness.INSTANCE.openMessageDialog(appManager, message, title,
    /// axisID)`.
    fn open_message_dialog(&self, message: &str, title: &str, axis_id: AxisID) {
        ui_harness::post_message_dialog(
            Some(self.app_manager),
            message.to_owned(),
            title.to_owned(),
            Some(axis_id),
        );
    }

    /// Java `batchruntomo`: returns true if the process succeeded.
    pub fn batchruntomo(&self, axis_id: AxisID, param: &mut BatchruntomoParam) -> bool {
        if !param.is_valid() {
            return false;
        }
        let exit_value = param.run();
        let mut retval = true;
        let mut title_prefix = "Batchruntomo";
        // process messages
        if equals_mode(param.get_command_mode(), &batchruntomo_param::Mode::Rename) {
            title_prefix = "Raw Image Stack";
            if let Some(output) = param.get_std_output() {
                for line in &output {
                    if line.contains(super::process_output_strings::BRT_RENAMED_MSG_ID) {
                        let file = super::log_feed_monitor::get_file_from_output(
                            line,
                            super::process_output_strings::BRT_RENAMED_MSG_ID,
                            None,
                        );
                        if let Some(file) = file {
                            // The raw image stack has been renamed
                            self.app_manager
                                .set_raw_image_stack_extension_string(Some(&file));
                            break;
                        }
                    }
                }
            }
        } else if equals_mode(
            param.get_command_mode(),
            &batchruntomo_param::Mode::Validation,
        ) {
            title_prefix = "Template Validation";
            // Upstream bug fixed in translation (ProcessManager.java:228-229): a null
            // `getProcessMessages()` (batchruntomo never built) throws
            // NullPointerException; here the validation fails instead.
            let Some(messages) = param.get_process_messages() else {
                return false;
            };
            retval = messages.is_empty(Some(MessageType::Error));
            if !retval {
                let mut error_message = String::from(
                    "The template validation has failed because of invalid directive(s).\nBatchruntomo error message:",
                );
                for i in 0..messages.size(MessageType::Error) {
                    error_message.push_str(&format!(
                        "\n{}",
                        messages.get(MessageType::Error, i).unwrap_or("null")
                    ));
                }
                self.open_message_dialog(&error_message, "Template Validation Error", axis_id);
            }
            for i in 0..messages.size(MessageType::Warning) {
                self.open_message_dialog(
                    messages.get(MessageType::Warning, i).unwrap_or("null"),
                    "Template Validation Warning",
                    axis_id,
                );
            }
        }
        if exit_value != 0 {
            retval = false;
            let message = param
                .get_std_error_string()
                .or_else(|| param.get_std_output_string())
                .unwrap_or_else(|| "null".to_owned());
            self.open_message_dialog(&message, &format!("{title_prefix} Error"), axis_id);
        }
        retval
    }

    /// Java `makecomfile`.
    pub fn makecomfile(&self, axis_id: AxisID, param: &mut MakecomfileParam) -> bool {
        if !param.setup() {
            return false;
        }
        let exit_value = param.run();
        // process messages
        // Upstream bug fixed in translation (ProcessManager.java, `makecomfile`): a
        // null `getProcessMessages()` throws NullPointerException; here it returns
        // false.
        let Some(messages) = param.get_process_messages() else {
            return false;
        };
        let err = !messages.is_empty(Some(MessageType::Error));
        if err {
            let mut error_message = String::from("Error running Makecomfile");
            for i in 0..messages.size(MessageType::Error) {
                error_message.push_str(&format!(
                    "\n{}",
                    messages.get(MessageType::Error, i).unwrap_or("null")
                ));
            }
            self.open_message_dialog(&error_message, "Makecomfile Error", axis_id);
        }
        for i in 0..messages.size(MessageType::Warning) {
            self.open_message_dialog(
                messages.get(MessageType::Warning, i).unwrap_or("null"),
                "Makecomfile Warning",
                axis_id,
            );
        }
        if exit_value != 0 {
            let msg = param
                .get_std_error_string()
                .filter(|msg| !msg.is_empty())
                .or_else(|| param.get_std_output_string())
                .unwrap_or_else(|| "null".to_owned());
            self.open_message_dialog(&msg, "Makecomfile Error", axis_id);
            return false;
        }
        err
    }

    /// Java `eraser`: erase the specified pixels.
    pub fn eraser(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        ccd_eraser_param: Option<Arc<CCDEraserParam>>,
    ) -> Result<String, AxisBusyException> {
        // Create the process monitor
        let ccd_eraser_process_monitor = CCDEraserProcessMonitor::new(self.app_manager, axis_id);
        // Create the required command string
        let command = format!("eraser{}.com", axis_id.get_extension());
        // Start the com script in the background
        let com_script_process = self.base.start_com_script_command(
            &command,
            monitor_of(&ccd_eraser_process_monitor),
            axis_id,
            process_result_display,
            ccd_eraser_param.as_ref().map(command_of),
            false,
            process_series,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `goldEraser`.
    pub fn gold_eraser(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        ccd_eraser_param: Arc<CCDEraserParam>,
    ) -> Result<String, AxisBusyException> {
        let ccd_eraser_process_monitor = CCDEraserProcessMonitor::new(self.app_manager, axis_id);
        let command = format!("golderaser{}.com", axis_id.get_extension());
        let com_script_process = self.base.start_com_script_command(
            &command,
            monitor_of(&ccd_eraser_process_monitor),
            axis_id,
            process_result_display,
            Some(command_of(&ccd_eraser_param)),
            false,
            process_series,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `clipStats`: run clip stats.
    pub fn clip_stats(
        &'static self,
        param: Arc<ClipParam>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            command_of(&param),
            false,
            axis_id,
            Some(ProcessName::CLIP),
            None,
            process_series,
            false,
            true,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `findSection`.
    pub fn find_section(
        &'static self,
        param: Arc<FindSectionParam>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            command_of(&param),
            false,
            axis_id,
            Some(ProcessName::FIND_SECTION),
            None,
            process_series,
            false,
            true,
        )?;
        Ok(background_process.get_name())
    }

    /// Java private `getDatasetName`.
    fn get_dataset_name(&self) -> String {
        self.app_manager.get_meta_data().get_dataset_name()
    }

    /// Java `crossCorrelate(BlendmontParam, ...)`: calculate the
    /// cross-correlation for the specified axis.
    pub fn cross_correlate(
        &'static self,
        param: Arc<BlendmontParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        // Create the process monitor
        let xcorr_process_watcher = XcorrProcessWatcher::new(self.app_manager, axis_id, true);
        // Start the com script in the background
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true, // static type implements CommandDetails: Java picks startComScript(CommandDetails, ...)
            monitor_of(&xcorr_process_watcher),
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `tiltxcorr`.
    #[allow(clippy::too_many_arguments)]
    pub fn tiltxcorr(
        &'static self,
        param: Arc<TiltxcorrParam>,
        process_name: ProcessName,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        run_tiltxcorr: bool,
        break_contours: bool,
    ) -> Result<String, AxisBusyException> {
        // Create the process monitor
        let tiltxcorr_process_watcher = TiltxcorrProcessWatcher::new_process_name(
            self.app_manager,
            axis_id,
            process_name,
            run_tiltxcorr,
            break_contours,
        );
        // Start the com script in the background
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            false,
            monitor_of(&tiltxcorr_process_watcher),
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `autofidseed`.
    pub fn autofidseed(
        &'static self,
        param: Arc<AutofidseedParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            false,
            None,
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `makeDistortionCorrectedStack`.
    pub fn make_distortion_corrected_stack(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        // Create the required tiltalign command
        let process_name =
            BlendmontParam::get_process_name_for_mode(blendmont_param::Mode::Undistort);
        let command = process_name.get_comscript(axis_id);
        // Start the com script in the background
        let blendmont_process_monitor = BlendmontProcessMonitor::new(
            self.app_manager,
            axis_id,
            blendmont_param::Mode::Undistort,
        );
        let com_script_process = self.base.start_com_script_resumable(
            &command,
            monitor_of(&blendmont_process_monitor),
            axis_id,
            process_result_display,
            process_series,
            process_name.resumable,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `coarseAlign`: calculate the coarse alignment for the specified
    /// axis.
    pub fn coarse_align(
        &'static self,
        param: Arc<NewstParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        // Create the required tiltalign command
        let command = format!("prenewst{}.com", axis_id.get_extension());
        // Start the com script in the background
        let prenewst_process_monitor = PrenewstProcessMonitor::new(self.app_manager, axis_id);
        let com_script_process = self.base.start_com_script_command(
            &command,
            monitor_of(&prenewst_process_monitor),
            axis_id,
            process_result_display,
            Some(command_of(&param)),
            false,
            process_series,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `preblend`: run preblend comscript.
    pub fn preblend(
        &'static self,
        param: Arc<BlendmontParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let blendmont_process_monitor = BlendmontProcessMonitor::new(
            self.app_manager,
            axis_id,
            blendmont_param::Mode::Preblend,
        );
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true, // static type implements CommandDetails: Java picks startComScript(CommandDetails, ...)
            monitor_of(&blendmont_process_monitor),
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `blend`: run blend comscript.
    pub fn blend(
        &'static self,
        blendmont_param: Arc<BlendmontParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let blendmont_process_monitor =
            BlendmontProcessMonitor::new(self.app_manager, axis_id, blendmont_param.get_mode());
        let com_script_process = self.base.start_com_script_param(
            command_of(&blendmont_param),
            true, // static type implements CommandDetails: Java picks startComScript(CommandDetails, ...)
            monitor_of(&blendmont_process_monitor),
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `generatePreXG`: generate the XG transform file.
    pub fn generate_pre_xg(&self, axis_id: AxisID) -> Result<(), RunCommandError> {
        let xftoxg = vec![
            format!(
                "{}xftoxg",
                base_manager::get_imod_bin_path().unwrap_or_default()
            ),
            "-NumberToFit".to_owned(),
            "0".to_owned(),
            format!(
                "{}{}.prexf",
                self.get_dataset_name(),
                axis_id.get_extension()
            ),
            format!(
                "{}{}.prexg",
                self.get_dataset_name(),
                axis_id.get_extension()
            ),
        ];
        self.run_command(xftoxg, axis_id, None)
    }

    /// Java `generateNonFidXF`: run both xftoxg and xproduct to create the
    /// _nonfid.xf for specified axis.
    pub fn generate_non_fid_xf(&self, axis_id: AxisID) -> Result<(), RunCommandError> {
        let xfproduct = vec![
            format!(
                "{}xfproduct",
                base_manager::get_imod_bin_path().unwrap_or_default()
            ),
            format!(
                "{}{}.prexg",
                self.get_dataset_name(),
                axis_id.get_extension()
            ),
            format!("rotation{}.xf", axis_id.get_extension()),
            format!(
                "{}{}_nonfid.xf",
                self.get_dataset_name(),
                axis_id.get_extension()
            ),
        ];
        self.run_command(xfproduct, axis_id, None)
    }

    /// Java `setupNonFiducialAlign`: copy the _nonfid.xf to .xf file and the
    /// .rawtlt to the .tlt file.
    pub fn setup_non_fiducial_align(&self, axis_id: AxisID) -> Result<(), RunCommandError> {
        let working_directory = self.app_manager.get_property_user_dir().unwrap_or_default();
        let axis_dataset = format!("{}{}", self.get_dataset_name(), axis_id.get_extension());
        let nonfid_xf = Path::new(&working_directory).join(format!("{axis_dataset}_nonfid.xf"));
        let xf = Path::new(&working_directory).join(format!("{axis_dataset}.xf"));
        utilities::copy_file(
            Some(self.app_manager),
            Some(axis_id),
            Some(&nonfid_xf),
            Some(&xf),
            false,
            false,
            false,
        )?;
        let rawtlt = Path::new(&working_directory).join(format!("{axis_dataset}.rawtlt"));
        let tlt = Path::new(&working_directory).join(format!("{axis_dataset}.tlt"));
        if !rawtlt.exists() {
            self.app_manager
                .make_rawtlt_file(axis_id)
                .map_err(|e| RunCommandError::SystemProcess(SystemProcessException(e)))?;
        }
        utilities::copy_file(
            Some(self.app_manager),
            Some(axis_id),
            Some(&rawtlt),
            Some(&tlt),
            false,
            false,
            false,
        )?;
        Ok(())
    }

    /// Java `setupFiducialAlign`: if they exist copy the _fid.xf to .xf and
    /// _fid.tlt to .tlt.
    pub fn setup_fiducial_align(
        &self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> Result<bool, LogFileError> {
        if self
            .app_manager
            .is_axis_busy(axis_id, process_result_display)
        {
            return Ok(false);
        }
        let working_directory = self.app_manager.get_property_user_dir().unwrap_or_default();
        let axis_dataset = format!("{}{}", self.get_dataset_name(), axis_id.get_extension());
        let file =
            |suffix: &str| Path::new(&working_directory).join(format!("{axis_dataset}{suffix}"));
        let copy = |from: &Path, to: &Path| {
            utilities::copy_file(
                Some(self.app_manager),
                Some(axis_id),
                Some(from),
                Some(to),
                false,
                false,
                false,
            )
        };
        // Files to be managed
        let xf = file(".xf");
        let fid_xf = file("_fid.xf");
        let nonfid_xf = file("_nonfid.xf");
        let tlt = file(".tlt");
        let fid_tlt = file("_fid.tlt");
        let tltxf = file(".tltxf");
        if tltxf.exists() {
            // Align{|a|b}.com shows evidence of being run
            if utilities::file_exists(self.app_manager, Some("_fid.xf"), Some(axis_id)) {
                // A recent align.com (or equivalent) has created the _fid.xf and
                // _fid.tlt (protected) transform and tilt files
                copy(&fid_xf, &xf)?;
                copy(&fid_tlt, &tlt)?;
            } else if nonfid_xf.exists() {
                // An older align.com that just wrote out an .xf and .tlt was
                // run; if the nonfid.xf was run it overwrote the the original
                // data: delete the xf and tlt so that an error occurs
                let _ = std::fs::remove_file(&xf);
                let _ = std::fs::remove_file(&tlt);
            } else {
                // No nonfid{|a|b}.xf exists so the .xf and .tlt came from
                // align; create the protected copies
                copy(&xf, &fid_xf)?;
                copy(&tlt, &fid_tlt)?;
            }
        } else {
            // Align has not been run, delete any .xf and .tlt file so that they
            // are not accidentally used
            let _ = std::fs::remove_file(&xf);
            let _ = std::fs::remove_file(&tlt);
        }
        Ok(true)
    }

    /// Java `midasRawStack` (deprecated 3/7/2020, no caller): run midas on the
    /// specified raw stack.
    pub fn midas_raw_stack(&self, axis_id: AxisID, image_rotation: f64) {
        BaseProcessManager::start_system_program_thread(
            self.get_midas_raw_stack_command_line(
                &self.get_dataset_name(),
                axis_id,
                image_rotation,
            ),
            axis_id,
            Some(self.app_manager),
        );
    }

    /// Java `getMidasRawStackCommandLine` (deprecated, test only).
    pub fn get_midas_raw_stack_command_line(
        &self,
        dataset_name: &str,
        axis_id: AxisID,
        image_rotation: f64,
    ) -> Vec<String> {
        let stack = file_type::CLASS
            .raw_stack
            .get_file_name(Some(self.app_manager), Some(axis_id));
        let xform = format!("{dataset_name}{}.prexf", axis_id.get_extension());
        // Midas must not fork.
        vec![
            format!(
                "{}midas",
                base_manager::get_imod_bin_path().unwrap_or_default()
            ),
            "-D".to_owned(),
            "-a".to_owned(),
            crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                -1.0 * image_rotation,
            ),
            "-t".to_owned(),
            dataset_files::get_raw_tilt_name(self.app_manager, Some(axis_id)),
            stack.unwrap_or_else(|| "null".to_owned()),
            xform,
        ]
    }

    /// Java `midasBlendStack` (deprecated 3/7/2020, no caller).
    pub fn midas_blend_stack(&self, axis_id: AxisID, image_rotation: f64) {
        BaseProcessManager::start_system_program_thread(
            self.get_midas_blend_stack_command_line(
                &self.get_dataset_name(),
                axis_id,
                image_rotation,
            ),
            axis_id,
            Some(self.app_manager),
        );
    }

    /// Java `getMidasBlendStackCommandLine` (deprecated, test only).
    pub fn get_midas_blend_stack_command_line(
        &self,
        dataset_name: &str,
        axis_id: AxisID,
        image_rotation: f64,
    ) -> Vec<String> {
        let stack = file_type::CLASS
            .xcorr_blend_output
            .get_file_name(Some(self.app_manager), Some(axis_id));
        let xform = format!("{dataset_name}{}.prexf", axis_id.get_extension());
        // Midas must not fork.
        vec![
            format!(
                "{}midas",
                base_manager::get_imod_bin_path().unwrap_or_default()
            ),
            "-D".to_owned(),
            "-a".to_owned(),
            crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string(
                -1.0 * image_rotation,
            ),
            "-t".to_owned(),
            dataset_files::get_raw_tilt_name(self.app_manager, Some(axis_id)),
            stack.unwrap_or_else(|| "null".to_owned()),
            xform,
        ]
    }

    /// Java `midasFixEdges` (deprecated 3/7/2020, no caller).
    pub fn midas_fix_edges(&self, axis_id: AxisID) {
        BaseProcessManager::start_system_program_thread(
            self.get_midas_fix_edges_command_line(&self.get_dataset_name(), axis_id),
            axis_id,
            Some(self.app_manager),
        );
    }

    /// Java `getMidasFixEdgesCommandLine` (deprecated, test only).
    pub fn get_midas_fix_edges_command_line(
        &self,
        dataset_name: &str,
        axis_id: AxisID,
    ) -> Vec<String> {
        let file_type = if self.app_manager.get_meta_data().is_distortion_correction() {
            &file_type::CLASS.distortion_corrected_stack
        } else {
            &file_type::CLASS.raw_stack
        };
        let stack = file_type.get_file_name(Some(self.app_manager), Some(axis_id));
        let xform = format!("{dataset_name}{}.ecd", axis_id.get_extension());
        // Midas must not fork.
        vec![
            format!(
                "{}midas",
                base_manager::get_imod_bin_path().unwrap_or_default()
            ),
            "-D".to_owned(),
            "-p ".to_owned(),
            format!("{dataset_name}{}.pl", axis_id.get_extension()),
            "-b".to_owned(),
            "0".to_owned(),
            "-q".to_owned(),
            stack.unwrap_or_else(|| "null".to_owned()),
            xform,
        ]
    }

    /// Java `fiducialModelTrack`: run the appropriate track com file.
    pub fn fiducial_model_track(
        &'static self,
        param: Arc<BeadtrackParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true, // static type implements CommandDetails: Java picks startComScript(CommandDetails, ...)
            None,
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `fineAlignment`: run the appropriate align com file.
    pub fn fine_alignment(
        &'static self,
        param: Arc<TiltalignParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        // Create the required tiltalign command (unused in the Java)
        let _command = format!("align{}.com", axis_id.get_extension());
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true, // static type implements CommandDetails: Java picks startComScript(CommandDetails, ...)
            None,
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `generateAlignLogs`: generate the split align log file.
    pub fn generate_align_logs(&self, axis_id: AxisID) {
        let mut alignment_log_generator = AlignLogGenerator::new(
            self.app_manager,
            axis_id,
            align_log_generator::Mode::TiltAlignLogs,
        );
        if alignment_log_generator.run().is_err() {
            self.open_message_dialog("Unable to create alignlog files", "Alignlog Error", axis_id);
        }
    }

    /// Java `generateAlignLogForProjectLog`.
    pub fn generate_align_log_for_project_log(&self, axis_id: AxisID) -> AlignLogGenerator {
        let mut alignment_log_generator = AlignLogGenerator::new(
            self.app_manager,
            axis_id,
            align_log_generator::Mode::ProjectLog,
        );
        if alignment_log_generator.run().is_err() {
            self.open_message_dialog(
                "Unable to add to the project log",
                "Alignlog Error",
                axis_id,
            );
        }
        alignment_log_generator
    }

    /// Java `copyFiducialAlignFiles`: copy the fiducial align files to the new
    /// protected names.
    pub fn copy_fiducial_align_files(&self, axis_id: AxisID) {
        let working_directory = self.app_manager.get_property_user_dir().unwrap_or_default();
        let axis_dataset = format!("{}{}", self.get_dataset_name(), axis_id.get_extension());
        let file =
            |suffix: &str| Path::new(&working_directory).join(format!("{axis_dataset}{suffix}"));
        let result = (|| -> Result<(), LogFileError> {
            if utilities::file_exists(self.app_manager, Some(".xf"), Some(axis_id)) {
                utilities::copy_file(
                    Some(self.app_manager),
                    Some(axis_id),
                    Some(&file(".xf")),
                    Some(&file("_fid.xf")),
                    false,
                    false,
                    false,
                )?;
            }
            if utilities::file_exists(self.app_manager, Some(".tlt"), Some(axis_id)) {
                utilities::copy_file(
                    Some(self.app_manager),
                    Some(axis_id),
                    Some(&file(".tlt")),
                    Some(&file("_fid.tlt")),
                    false,
                    false,
                    false,
                )?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("{}", e.get_message());
            self.open_message_dialog(
                "Unable to copy protected align files:",
                "Align Error",
                axis_id,
            );
        }
    }

    /// Java `transferFiducials`: run the transferfid script.
    pub fn transfer_fiducials(
        &'static self,
        transferfid_param: Arc<TransferfidParam>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let mut axis_id = AxisID::Second;
        // Run transferfid on the destination axis.
        if transferfid_param.get_b_to_a().is() {
            axis_id = AxisID::First;
        }
        let background_process = self.base.start_background_process_array_display(
            transferfid_param.get_command(),
            axis_id,
            process_result_display,
            Some(ProcessName::TRANSFERFID),
            process_series,
        )?;
        *self.transferfid_command_line.lock().unwrap() =
            Some(background_process.get_command_line());
        Ok(background_process.get_name())
    }

    /// Java `createSample`: run the appropriate sample com file.
    pub fn create_sample(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        param: Arc<TiltParam>,
    ) -> Result<String, AxisBusyException> {
        // Create the required sample command
        let command = format!("{}{}.com", ProcessName::SAMPLE, axis_id.get_extension());
        let com_script_process = self.base.start_com_script_command(
            &command,
            None,
            axis_id,
            process_result_display,
            Some(command_of(&param)),
            false,
            process_series,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `tomopitch`: run the appropriate tomopitch com file.
    pub fn tomopitch(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let command = format!("tomopitch{}.com", axis_id.get_extension());
        let com_script_process = self.base.start_com_script_resumable(
            &command,
            None,
            axis_id,
            process_result_display,
            process_series,
            ProcessName::TOMOPITCH.resumable,
        )?;
        Ok(name_of(com_script_process))
    }

    // Java override `processchunks(AxisID, ProcesschunksParam, ...)`: see
    // `BaseProcessManagerHooks::processchunks` below.

    /// Java `newst`: run the appropriate newst com (or newst_3dfind.com).
    pub fn newst(
        &'static self,
        newst_param: Arc<NewstParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        process_name: ProcessName,
    ) -> Result<String, AxisBusyException> {
        let newst_process_monitor = NewstProcessMonitor::new(
            self.app_manager,
            axis_id,
            process_name,
            Arc::clone(&newst_param) as Arc<dyn ConstNewstParam + Send + Sync>,
        );
        let com_script_process = self.base.start_com_script_param(
            command_of(&newst_param),
            true, // static type implements CommandDetails: Java picks startComScript(CommandDetails, ...)
            monitor_of(&newst_process_monitor),
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `cryoPosition`.
    pub fn cryo_position(
        &'static self,
        param: Arc<CryoPositionParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            false,
            None,
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `findBeads3d`.
    pub fn find_beads3d(
        &'static self,
        param: Arc<FindBeads3dParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true, // static type implements CommandDetails: Java picks startComScript(CommandDetails, ...)
            None,
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `mtffilter`: run the appropriate mtffilter com file.
    pub fn mtffilter(
        &'static self,
        param: Arc<MTFFilterParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let command = format!("mtffilter{}.com", axis_id.get_extension());
        let mtffilter_process_monitor = MtffilterProcessMonitor::new(self.app_manager, axis_id);
        let com_script_process = self.base.start_com_script_command(
            &command,
            monitor_of(&mtffilter_process_monitor),
            axis_id,
            process_result_display,
            Some(command_of(&param)),
            false,
            process_series,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `ctfPlotter`.
    pub fn ctf_plotter(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) {
        let command = ProcessName::CTF_PLOTTER.get_comscript(axis_id);
        // Start the com script in the background
        self.base
            .start_non_blocking_com_script(&command, axis_id, process_result_display);
    }

    /// Java `ctfCorrection`.
    pub fn ctf_correction(
        &'static self,
        param: Arc<CtfPhaseFlipParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let command = ProcessName::CTF_CORRECTION.get_comscript(axis_id);
        let monitor = CtfCorrectionMonitor::new(self.app_manager, axis_id);
        let com_script_process = self.base.start_com_script_command(
            &command,
            monitor_of(&monitor),
            axis_id,
            process_result_display,
            Some(command_of(&param)),
            false,
            process_series,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `reconnectTilt(AxisID, ProcessResultDisplay, ProcessSeries)`.
    pub fn reconnect_tilt(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> bool {
        let monitor = TiltProcessMonitor::get_reconnect_instance(self.app_manager, axis_id);
        let process_monitor: Arc<dyn ProcessMonitor> = monitor.clone();
        let process = match ReconnectProcess::get_instance(
            self.app_manager,
            &self.base,
            Some(process_monitor),
            Some(self.base.axis_process_data.get_saved_process_data(axis_id)),
            axis_id,
            process_series,
        ) {
            Ok(process) => process,
            Err(e) => {
                eprintln!("{}", e.get_message());
                ui_harness::open_message_dialog_from_process(
                    Some(self.app_manager),
                    &format!("Unable to reconnect to processchunks.\n{}", e.get_message()),
                    "Reconnect Failure",
                    Some(axis_id),
                );
                return false;
            }
        };
        process.set_process_result_display(process_result_display);
        let thread_process = Arc::clone(&process);
        std::thread::spawn(move || thread_process.run());
        self.base
            .axis_process_data
            .map_axis_thread(Some(process.as_process()), axis_id);
        self.base.axis_process_data.map_axis_process_monitor(
            None,
            Some(monitor as Arc<dyn Monitor>),
            axis_id,
        );
        true
    }

    /// Java `sirtsetup`.
    pub fn sirtsetup(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        param: Arc<SirtsetupParam>,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true,
            None,
            axis_id,
            process_result_display,
            process_series,
            processing_method,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `multifiltSetup`.
    pub fn multifilt_setup(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        param: Arc<MultifiltSetupParam>,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true,
            None,
            axis_id,
            process_result_display,
            process_series,
            processing_method,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `ctf3dSetup`.
    pub fn ctf3d_setup(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        param: Arc<Ctf3dSetupParam>,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true,
            None,
            axis_id,
            process_result_display,
            process_series,
            processing_method,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `subtomoSetup`.
    pub fn subtomo_setup(
        &'static self,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
        param: Arc<SubtomoSetupParam>,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true,
            None,
            axis_id,
            None,
            process_series,
            processing_method,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `altTomoSetup`.
    pub fn alt_tomo_setup(
        &'static self,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
        param: Arc<AltTomoSetupParam>,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true,
            None,
            axis_id,
            None,
            process_series,
            processing_method,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `tilt`: run the appropriate tilt com file (sampling a whole
    /// tomogram).
    pub fn tilt(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        param: Arc<TiltParam>,
        process_title: Option<&str>,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<String, AxisBusyException> {
        // Instantiate the process monitor
        let tilt_process_monitor =
            TiltProcessMonitor::new(self.app_manager, axis_id, ProcessName::TILT);
        tilt_process_monitor
            .subclass
            .set_process_title(process_title);
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true,
            monitor_of(&tilt_process_monitor),
            axis_id,
            process_result_display,
            process_series,
            processing_method,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `tilt3dFind`: run the appropriate tilt_3dfind com file.
    #[allow(clippy::too_many_arguments)]
    pub fn tilt3d_find(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        param: Arc<TiltParam>,
        process_title: Option<&str>,
        process_name: ProcessName,
        processing_method: Option<ProcessingMethod>,
    ) -> Result<String, AxisBusyException> {
        let monitor = Tilt3dFindProcessMonitor::new(
            self.app_manager,
            axis_id,
            process_name,
            Arc::clone(&param) as Arc<dyn ConstTiltParam + Send + Sync>,
        );
        monitor.subclass.set_process_title(process_title);
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true,
            monitor_of(&monitor),
            axis_id,
            process_result_display,
            process_series,
            processing_method,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `tilt3dFindReproject`: run tilt_3dfind in reproject mode.
    pub fn tilt3d_find_reproject(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        param: Arc<TiltParam>,
        _process_title: Option<&str>,
        _process_name: ProcessName,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            true,
            None,
            axis_id,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `splittilt`: run splittilt.
    pub fn splittilt(
        &'static self,
        param: Arc<SplittiltParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_display(
            param.get_command(),
            axis_id,
            process_result_display,
            Some(ProcessName::SPLITTILT),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `splitCorrection`.
    pub fn split_correction(
        &'static self,
        param: Arc<dyn ConstSplitCorrectionParam + Send + Sync>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_display(
            param.get_command(),
            axis_id,
            process_result_display,
            Some(ProcessName::SPLIT_CORRECTION),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `extracttilts`: run extracttilts.
    pub fn extracttilts(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_force(
            ExtracttiltsParam::new(self.app_manager, axis_id).get_command(),
            axis_id,
            true,
            process_result_display,
            process_series,
            Some(ProcessName::EXTRACTTILTS),
        )?;
        Ok(background_process.get_name())
    }

    /// Java `extractpieces`: run extractpieces.
    pub fn extractpieces(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_force(
            ExtractpiecesParam::new(self.app_manager, axis_id).get_command(),
            axis_id,
            true,
            process_result_display,
            process_series,
            Some(ProcessName::EXTRACTPIECES),
        )?;
        Ok(background_process.get_name())
    }

    /// Java `extractmagrad`: run extractmagrad.
    pub fn extractmagrad(
        &'static self,
        param: Arc<ExtractmagradParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_force(
            param.get_command(),
            axis_id,
            true,
            process_result_display,
            process_series,
            Some(ProcessName::EXTRACTMAGRAD),
        )?;
        Ok(background_process.get_name())
    }

    /// Java `splitcombine`: run splitcombine.
    pub fn splitcombine(
        &'static self,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_display(
            SplitcombineParam::new().get_command(),
            AxisID::Only,
            process_result_display,
            Some(ProcessName::SPLITCOMBINE),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `setupCombineScripts`: execute the setupcombine script.
    pub fn setup_combine_scripts(
        &self,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> std::io::Result<bool> {
        let mut setup_combine = match SetupCombine::get_instance(self.app_manager) {
            Ok(setup_combine) => setup_combine,
            Err(e) => {
                self.open_message_dialog(&e.to_string(), "Setup Combine Error", AxisID::Only);
                return Ok(false);
            }
        };
        self.app_manager.save_storables(Some(AxisID::Only));
        let exit_value = setup_combine.run();
        let messages = setup_combine.get_process_messages();
        for i in 0..messages.size(MessageType::Error) {
            self.open_message_dialog(
                messages.get(MessageType::Error, i).unwrap_or("null"),
                "Setup Combine Error",
                AxisID::Only,
            );
        }
        for i in 0..messages.size(MessageType::Warning) {
            self.open_message_dialog(
                messages.get(MessageType::Warning, i).unwrap_or("null"),
                "Setup Combine Warning",
                AxisID::Only,
            );
        }
        let state = self.app_manager.get_state();
        if exit_value != 0 {
            self.open_message_dialog(
                &format!("Setup combine failed.  Exit value = {exit_value}"),
                "Setup Combine Failed",
                AxisID::Only,
            );
            if let Some(display) = &process_result_display {
                display.get().msg_process_failed();
            }
            state.set_combine_match_mode(None);
            state.set_combine_scripts_created(false);
            return Ok(false);
        }
        if let Some(display) = &process_result_display {
            display.get().msg_process_succeeded();
        }
        state.set_combine_match_mode(setup_combine.get_match_mode());
        state.set_combine_scripts_created(true);
        Ok(true)
    }

    /// Java `setupCombineOnlyMakeCombineCom`.
    pub fn setup_combine_only_make_combine_com(&self) -> std::io::Result<bool> {
        let mut setup_combine =
            match SetupCombine::get_only_make_combine_com_instance(self.app_manager) {
                Ok(setup_combine) => setup_combine,
                Err(e) => {
                    self.open_message_dialog(&e.to_string(), "Setup Combine Error", AxisID::Only);
                    return Ok(false);
                }
            };
        let exit_value = setup_combine.run();
        let messages = setup_combine.get_process_messages();
        for i in 0..messages.size(MessageType::Error) {
            self.open_message_dialog(
                messages.get(MessageType::Error, i).unwrap_or("null"),
                "Setup Combine Error",
                AxisID::Only,
            );
        }
        for i in 0..messages.size(MessageType::Warning) {
            self.open_message_dialog(
                messages.get(MessageType::Warning, i).unwrap_or("null"),
                "Setup Combine Warning",
                AxisID::Only,
            );
        }
        if exit_value != 0 {
            self.open_message_dialog(
                &format!(
                    "Setup combine failed.  Copy combine.com from $IMOD_DIR/com.  Exit value = {exit_value}"
                ),
                "Setup Combine Failed",
                AxisID::Only,
            );
            return Ok(false);
        }
        Ok(true)
    }

    /// Java `modelToPatch`: run the imod2patch command.
    pub fn model_to_patch(&self, axis_id: AxisID) -> Result<(), RunCommandError> {
        let patch_out = LogFile::get_instance_user_dir(
            &self.app_manager.get_property_user_dir().unwrap_or_default(),
            dataset_files::PATCH_OUT,
            Some(self.app_manager.get_emergency_monitor(Some(axis_id))),
        )?;
        patch_out.backup()?;
        // Convert the new patchvector.mod
        let command = "imod2patch";
        let imod2patch = vec![
            command.to_owned(),
            dataset_files::PATCH_VECTOR_MODEL.to_owned(),
            dataset_files::PATCH_OUT.to_owned(),
        ];
        self.run_command(imod2patch, axis_id, Some(&patch_out))
    }

    /// Java `combine(CombineComscriptState, ProcessResultDisplay,
    /// ProcessSeries)`: run the combine com file.
    pub fn combine(
        &'static self,
        combine_comscript_state: CombineComscriptState,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        // Create the required combine command
        let comscript = format!("{}.com", combine_comscript_state::COMSCRIPT_NAME);
        let combine_comscript_state = Arc::new(combine_comscript_state);
        let combine_process_monitor = CombineProcessMonitor::new(
            self.app_manager,
            AxisID::Only,
            Arc::clone(&combine_comscript_state),
            process_result_display,
        );
        // Start the com script in the background
        let com_script_process = self.base.start_background_com_script(
            &comscript,
            combine_process_monitor,
            AxisID::Only,
            Some(combine_comscript_state),
            Some(combine_comscript_state::COMSCRIPT_WATCHED_FILE),
            process_series,
            ProcessName::COMBINE.resumable,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java private `solvematch`: run the solvematch com file (no caller).
    fn solvematch(
        &'static self,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let command = "solvematch.com";
        let com_script_process = self.base.start_com_script_series_resumable(
            command,
            None,
            AxisID::Only,
            process_series,
            ProcessName::SOLVEMATCH.resumable,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java private `matchvol1`: run the matchvol1 com file (no caller).
    fn matchvol1(
        &'static self,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let command = "matchvol1.com";
        let com_script_process = self.base.start_com_script_series_resumable(
            command,
            None,
            AxisID::Only,
            process_series,
            ProcessName::MATCHVOL1.resumable,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `matchorwarp`: run the matchorwarp com file.
    pub fn matchorwarp(
        &'static self,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let command = "matchorwarp.com";
        let com_script_process = self.base.start_com_script_series_resumable(
            command,
            None,
            AxisID::Only,
            process_series,
            ProcessName::MATCHORWARP.resumable,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `trimVolume`: run trimvol.
    pub fn trim_volume(
        &'static self,
        trimvol_param: Arc<TrimvolParam>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command_display(
            command_of(&trimvol_param),
            AxisID::Only,
            process_result_display,
            Some(ProcessName::TRIMVOL),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `excludeViews`.
    pub fn exclude_views(
        &'static self,
        param: Arc<dyn Command + Send + Sync>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command_multi_line(
            param,
            axis_id,
            None,
            Some(ProcessName::EXCLUDE_VIEWS),
            process_series,
            true,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `archiveOrig`: run archiveorig.
    pub fn archive_orig(
        &'static self,
        param: Arc<ArchiveorigParam>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command(
            command_of(&param),
            false,
            AxisID::Only,
            Some(ProcessName::ARCHIVEORIG),
            None,
            process_series,
            true,
            true,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `restrictalign`.
    pub fn restrictalign(
        &'static self,
        param: Arc<RestrictalignParam>,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&param),
            false,
            None,
            axis_id,
            None,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `flatten`: run the appropriate flatten com file.
    pub fn flatten(
        &'static self,
        param: Arc<WarpVolParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let command = format!("{}{}.com", ProcessName::FLATTEN, axis_id.get_extension());
        let monitor =
            Matchvol1ProcessMonitor::get_flatten_instance(self.app_manager, axis_id, None);
        let com_script_process = self.base.start_com_script_command(
            &command,
            monitor_of(&monitor),
            axis_id,
            process_result_display,
            Some(command_of(&param)),
            false,
            process_series,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java `flattenWarp`.
    pub fn flatten_warp(
        &'static self,
        param: Arc<FlattenWarpParam>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        axis_id: AxisID,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_display(
            param.get_command_array(),
            axis_id,
            process_result_display,
            Some(param.get_process_name()),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `runraptor(RunraptorParam, ProcessResultDisplay, ProcessSeries,
    /// AxisID)`: runs the runraptor command line as a background process and
    /// returns the process's name.
    pub fn runraptor(
        &'static self,
        param: &mut crate::imod::etomo::comscript::runraptor_param::RunraptorParam,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        axis_id: AxisID,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_array_display(
            param.get_command_array(),
            axis_id,
            process_result_display,
            Some(param.get_process_name()),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `squeezeVolume`: run squeezevol (no caller; replaced by
    /// reducefiltvol).
    pub fn squeeze_volume(
        &'static self,
        squeezevol_param: Arc<SqueezevolParam>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let background_process = self.base.start_background_process_command_display(
            command_of(&squeezevol_param),
            AxisID::Only,
            process_result_display,
            Some(ProcessName::SQUEEZEVOL),
            process_series,
        )?;
        Ok(background_process.get_name())
    }

    /// Java `reduceFiltVol`.
    pub fn reduce_filt_vol(
        &'static self,
        reduce_filt_vol_param: Arc<ReduceFiltVolParam>,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
    ) -> Result<String, AxisBusyException> {
        let com_script_process = self.base.start_com_script_param(
            command_of(&reduce_filt_vol_param),
            false,
            None,
            AxisID::Only,
            process_result_display,
            process_series,
            None,
        )?;
        Ok(name_of(com_script_process))
    }

    /// Java private `showLogFile`: puts a log file into a window and displays
    /// it.
    fn show_log_file(&self, log_file: &Path) {
        // Show a log file window to the user.  The window is a Swing object:
        // built on the event dispatch thread (this runs on the process thread).
        let log_file = log_file.to_path_buf();
        event_queue::invoke_later(move || {
            // TextPageWindow(): the font size comes from the first UIManager
            // FontUIResource, which is not modelled; UIParameters' default stands in.
            let log_file_window = TextPageWindow::new(ui_parameters::DEFAULT_FONT_SIZE as i32);
            let visible = log_file_window.set_file_from_file(&log_file);
            log_file_window.set_visible(visible);
        });
    }

    /// Java private `runCommand`: execute the command and arguments in
    /// commandArray immediately.
    fn run_command(
        &self,
        command_array: Vec<String>,
        axis_id: AxisID,
        log_file: Option<&Arc<Handle>>,
    ) -> Result<(), RunCommandError> {
        let system_program = SystemProgram::new_array(
            Some(self.app_manager),
            self.app_manager.get_property_user_dir(),
            Some(command_array),
            axis_id,
        );
        system_program.set_working_directory(Some(PathBuf::from(
            self.app_manager.get_property_user_dir().unwrap_or_default(),
        )));
        let log_writing_id = match log_file {
            Some(log_file) => Some(log_file.open_for_writing()?),
            None => None,
        };
        system_program.run();
        if let (Some(log_file), Some(log_writing_id)) = (log_file, &log_writing_id) {
            log_file.close_id(Some(log_writing_id));
        }
        if system_program.get_exit_value() != 0 {
            let mut message = String::new();
            // Copy any stderr output to the message
            if let Some(stderr) = system_program.get_std_error() {
                for line in &stderr {
                    message = format!("{message}{line}\n");
                }
            }
            // Also scan stdout for ERROR: lines
            let mut found_error = false;
            if let Some(std_output) = system_program.get_std_output() {
                for line in &std_output {
                    if !found_error {
                        if line.contains("ERROR:") {
                            found_error = true;
                            message.push_str(line);
                        }
                    } else {
                        message.push_str(line);
                    }
                }
            }
            return Err(RunCommandError::SystemProcess(SystemProcessException(
                message,
            )));
        }
        Ok(())
    }

    /// Java private `postProcess(String processName, AxisID)`: post process
    /// for processes that may or may not be done with processchunks.
    fn post_process_name(&self, process_name: Option<&str>, axis_id: AxisID) {
        let Some(process_name) = process_name else {
            return;
        };
        if ProcessName::TILT_3D_FIND.equals_with_axis(process_name, axis_id) {
            self.app_manager.copy_tilt3d_find_reproject_com(axis_id);
        } else if ProcessName::CTF_CORRECTION.equals_with_axis(process_name, axis_id) {
            self.app_manager
                .get_state()
                .set_use_ctf_correction_warning(axis_id, true);
        } else if ProcessName::ALT_TOMO_PROCESS_CHUNKS.equals_with_axis(process_name, axis_id) {
            self.app_manager.log_simple_message_newline(
                Some(self.app_manager.get_alt_tomo_setup_log_message().as_str()),
                true,
            );
        }
    }

    /// Java private `setInvalidEdgeFunctions`.
    fn set_invalid_edge_functions(
        &self,
        command: Option<&Arc<dyn Command + Send + Sync>>,
        succeeded: bool,
    ) {
        let Some(command) = command else {
            return;
        };
        if self.app_manager.get_meta_data().get_view_type() == ViewType::Montage
            && command.get_command_name().as_deref() == Some(blendmont_param::COMMAND_NAME)
            && (equals_mode(command.get_command_mode(), &blendmont_param::Mode::Xcorr)
                || equals_mode(command.get_command_mode(), &blendmont_param::Mode::Preblend))
        {
            self.app_manager
                .get_state()
                .set_invalid_edge_functions(command.get_axis_id(), !succeeded);
        }
    }

    /// Java `getManager`.
    pub fn get_manager(&self) -> &'static dyn BaseManager {
        self.app_manager
    }
}

/// `details.getBooleanValue(field)` with Java's `false` for an unset value.
fn boolean_of(
    details: Option<&Arc<dyn Command + Send + Sync>>,
    field: &dyn FieldInterface,
) -> bool {
    details
        .and_then(|details| details.get_process_details())
        .and_then(|details| details.get_boolean_value(field))
        .unwrap_or(false)
}

/// `commandDetails.getCommandMode() == mode`.
fn mode_is<M: CommandMode + PartialEq>(
    details: Option<&Arc<dyn Command + Send + Sync>>,
    mode: &M,
) -> bool {
    equals_mode(details.and_then(|details| details.get_command_mode()), mode)
}

impl BaseProcessManagerHooks for ProcessManager {
    /// Java override `processchunks(AxisID, ProcesschunksParam,
    /// ParallelProgressDisplay, ProcessResultDisplay, ProcessSeries, boolean,
    /// ProcessingMethod, boolean, RunType, ProcessData, List<ProcessMessages>)`.
    #[allow(clippy::too_many_arguments)]
    fn processchunks(
        &self,
        base: &'static BaseProcessManager,
        axis_id: AxisID,
        param: Arc<ProcesschunksParam>,
        parallel_progress_display: &dyn ParallelProgressDisplay,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: Option<ProcessSeriesRef>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
        multi_line_messages: bool,
        run_type: Option<RunType>,
        managed_process_data: Option<Arc<Mutex<ProcessData>>>,
        messages_array: Option<MessagesArray>,
    ) -> Result<String, AxisBusyException> {
        if param.equals_root_name(Some(ProcessName::VOLCOMBINE), Some(axis_id)) {
            return base.processchunks_monitor(
                ProcesschunksVolcombineMonitor::new(
                    base.manager,
                    axis_id,
                    param.get_root_name().as_deref(),
                    Some(param.get_computer_map().into_iter().collect()),
                ),
                axis_id,
                param,
                parallel_progress_display,
                process_result_display,
                process_series,
                popup_chunk_warnings,
                processing_method,
                managed_process_data,
            );
        }
        base.processchunks_base(
            axis_id,
            param,
            parallel_progress_display,
            process_result_display,
            process_series,
            popup_chunk_warnings,
            processing_method,
            multi_line_messages,
            run_type,
            managed_process_data,
            messages_array,
        )
    }

    /// Java override `postProcess(DetachedProcess)`.
    fn post_process_detached(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_detached_base(process);
        // A null command or command name is the Java's caught
        // NullPointerException.
        if let Some(command) = process.get_command()
            && command.get_command_name().as_deref()
                == Some(ProcessName::PROCESSCHUNKS.to_string().as_str())
            && let Some(command_details) = process.get_command_details()
        {
            self.post_process_name(
                command_details.get_subcommand_process_name().as_deref(),
                process.get_axis_id(),
            );
        }
    }

    /// Java override `postProcess(ComScriptProcess)`.
    fn post_process_com_script(&self, _base: &BaseProcessManager, script: &ComScriptProcess) {
        // Script specific post processing
        let process_name = script.get_process_name();
        let process_details = script.get_command();
        let command_details = script.get_command_details();
        let command = script.get_command();
        let state = self.app_manager.get_state();
        let axis_id = script.get_axis_id();
        if process_name == Some(ProcessName::ALIGN) {
            self.generate_align_logs(axis_id);
            state.set_made_z_factors(
                axis_id,
                boolean_of(
                    process_details,
                    &const_tiltalign_param::Fields::UseOutputZFactorFile,
                ),
            );
            state.set_used_local_alignments(
                axis_id,
                boolean_of(
                    process_details,
                    &const_tiltalign_param::Fields::LocalAlignments,
                ),
            );
            self.app_manager.set_tilt_state(axis_id);
            let double = |field: &dyn FieldInterface| {
                process_details
                    .and_then(|details| details.get_process_details())
                    .and_then(|details| details.get_double_value(field))
                    .unwrap_or(0.0)
            };
            state.set_align_axis_z_shift(
                axis_id,
                double(&const_tiltalign_param::Fields::AxisZShift),
            );
            state.set_align_angle_offset(
                axis_id,
                double(&const_tiltalign_param::Fields::AngleOffset),
            );
            self.app_manager.post_process(
                axis_id,
                process_name.clone(),
                process_details.cloned(),
                script.get_process_result_display(),
            );
            // The project log is a Swing window: logged on the event dispatch
            // thread (this runs on the process thread).
            let align_log = self.generate_align_log_for_project_log(axis_id);
            let app_manager = self.app_manager;
            event_queue::invoke_later(move || {
                app_manager.log_message_loggable(Some(&align_log), Some(axis_id));
            });
        } else if process_name == Some(ProcessName::ERASER) {
            if let Some(command) = command
                && equals_mode(command.get_command_mode(), &ccd_eraser_param::Mode::XRays)
            {
                state.set_use_fixed_stack_warning(axis_id, true);
            }
        } else if process_name == Some(ProcessName::MTFFILTER) {
            state.set_use_filtered_stack_warning(axis_id, true);
        } else if process_name == Some(ProcessName::CTF_CORRECTION) {
            state.set_use_ctf_correction_warning(axis_id, true);
        } else if process_name == Some(ProcessName::TOMOPITCH) {
            self.app_manager.set_tomopitch_output(axis_id);
        } else if process_name == Some(ProcessName::NEWST) {
            if mode_is(command_details, &newst_param::Mode::FullAlignedStack) {
                let details = command_details.and_then(|details| details.get_process_details());
                state.set_newst_fiducialess_alignment(
                    axis_id,
                    boolean_of(command_details, &newst_param::Field::FiducialessAlignment),
                );
                self.app_manager.set_tilt_state(axis_id);
                self.app_manager.update_aligned_stack_binning(axis_id);
                state.set_stack_use_linear_interpolation(
                    axis_id,
                    boolean_of(command_details, &newst_param::Field::UseLinearInterpolation),
                );
                state.set_stack_user_size_to_output_in_x_and_y(
                    axis_id,
                    details
                        .and_then(|d| d.get_string(&newst_param::Field::UserSizeToOutputInXAndY))
                        .as_deref(),
                );
                state.set_stack_image_rotation(
                    axis_id,
                    details
                        .and_then(|d| d.get_etomo_number(&newst_param::Field::ImageRotation))
                        .as_ref(),
                );
            }
        } else if process_name == Some(ProcessName::BLEND) {
            if mode_is(command_details, &blendmont_param::Mode::Blend) {
                let details = command_details.and_then(|details| details.get_process_details());
                state.set_stack_use_linear_interpolation(
                    axis_id,
                    boolean_of(
                        command_details,
                        &blendmont_param::Field::LinearInterpolation,
                    ),
                );
                state.set_stack_user_size_to_output_in_x_and_y(
                    axis_id,
                    details
                        .and_then(|d| {
                            d.get_string(&blendmont_param::Field::UserSizeToOutputInXAndY)
                        })
                        .as_deref(),
                );
                state.set_stack_image_rotation(
                    axis_id,
                    details
                        .and_then(|d| d.get_etomo_number(&blendmont_param::Field::ImageRotation))
                        .as_ref(),
                );
                state.set_newst_fiducialess_alignment(
                    axis_id,
                    boolean_of(command_details, &blendmont_param::Field::Fiducialess),
                );
            }
        } else if process_name == Some(ProcessName::UNDISTORT) {
            self.app_manager.set_enabled_fix_edges_with_midas(axis_id);
        } else if process_name == Some(ProcessName::XCORR) {
            self.set_invalid_edge_functions(script.get_command(), true);
            if command_details.is_some() && mode_is(command_details, &blendmont_param::Mode::Xcorr)
            {
                state.set_xcorr_blendmont_was_run(axis_id, true);
            }
        } else if process_name == Some(ProcessName::PREBLEND) {
            self.set_invalid_edge_functions(script.get_command(), true);
        } else if process_name == Some(ProcessName::SAMPLE) {
            self.app_manager.tomogram_positioning_post_process(
                axis_id,
                process_details.cloned(),
                PosSampleType::Samples,
            );
        } else if process_name == Some(ProcessName::TILT) {
            if mode_is(command_details, &tilt_param::Mode::Whole) {
                self.app_manager.tomogram_positioning_post_process(
                    axis_id,
                    process_details.cloned(),
                    PosSampleType::Whole,
                );
            }
            if !mode_is(command_details, &tilt_param::Mode::Sample) {
                state.set_adjust_origin(
                    axis_id,
                    boolean_of(process_details, &tilt_param::Field::AdjustOrigin),
                );
            }
        } else if process_name == Some(ProcessName::NEWST_3D_FIND)
            || process_name == Some(ProcessName::BLEND_3D_FIND)
        {
            state.set_stack_using_newst_or_blend_3d_find_output(axis_id, true);
        } else if process_name == Some(ProcessName::GOLD_ERASER) {
            state.set_use_erased_stack_warning(axis_id, true);
        } else if process_name == Some(ProcessName::TRACK) {
            let fiducial_file =
                dataset_files::get_fiducial_model_file(self.app_manager, Some(axis_id));
            if fiducial_file.exists() {
                let modified = std::fs::metadata(&fiducial_file)
                    .and_then(|metadata| metadata.modified())
                    .ok()
                    .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |duration| duration.as_millis() as i64);
                state.set_fid_file_last_modified(axis_id, modified);
                // Logged on the event dispatch thread, where the project log is.
                let app_manager = self.app_manager;
                let user_dir = app_manager.get_property_user_dir();
                event_queue::invoke_later(move || {
                    app_manager.log_message_loggable(
                        Some(&TrackLog::get_instance(
                            Some(app_manager),
                            axis_id,
                            user_dir.as_deref(),
                        )),
                        Some(axis_id),
                    );
                });
            } else {
                state.reset_fid_file_last_modified(axis_id);
            }
            state.set_track_light_beads(
                axis_id,
                boolean_of(command_details, &beadtrack_param::Field::LightBeads),
            );
        } else if process_name == Some(ProcessName::SIRTSETUP) {
            if boolean_of(process_details, &sirtsetup_param::Field::Subarea) {
                let details = process_details.and_then(|details| details.get_process_details());
                state.set_gen_sirtsetup_subarea_size(
                    axis_id,
                    details
                        .and_then(|d| d.get_string(&sirtsetup_param::Field::SubareaSize))
                        .as_deref(),
                );
                state.set_gen_sirtsetupy_offset_of_subarea(
                    axis_id,
                    details
                        .and_then(|d| d.get_int_value(&sirtsetup_param::Field::YOffsetOfSubset))
                        .unwrap_or(0),
                );
            }
            self.app_manager.msg_sirtsetup_succeeded(axis_id);
        } else if process_name == Some(ProcessName::AUTOFIDSEED) {
            self.app_manager.msg_autofidseed_succeeded(axis_id);
            // Logged on the event dispatch thread, where the project log is.
            let app_manager = self.app_manager;
            let user_dir = app_manager.get_property_user_dir();
            event_queue::invoke_later(move || {
                app_manager.log_message_loggable(
                    Some(&AutofidseedLog::get_instance(
                        app_manager,
                        axis_id,
                        user_dir.as_deref(),
                    )),
                    Some(axis_id),
                );
            });
            state.set_seeding_done(axis_id, true);
        } else if process_name == Some(ProcessName::CRYO_POSITION) {
            self.app_manager.rename_cryo_position_output(axis_id);
            // `command.getSubcommandDetails()`: the command of a cryoposition run
            // is a CryoPositionParam, whose subcommand details are its tilt
            // param; handed over as the shared object.
            let subcommand_details = command
                .cloned()
                .and_then(|command| {
                    (command as Arc<dyn std::any::Any + Send + Sync>)
                        .downcast::<CryoPositionParam>()
                        .ok()
                })
                .and_then(|param| param.get_subcommand_details_shared())
                .map(|tilt_param| tilt_param as Arc<dyn Command + Send + Sync>);
            self.app_manager.tomogram_positioning_post_process(
                axis_id,
                subcommand_details,
                PosSampleType::WholeCryo,
            );
        } else if process_name == Some(ProcessName::ALT_TOMO_SETUP) {
            if mode_is(command_details, &alt_tomo_setup_param::Mode::AltTomoSetup) {
                let details = process_details.and_then(|details| details.get_process_details());
                state.set_alt_tomo_preprocess_for_extremes(boolean_of(
                    process_details,
                    &alt_tomo_setup_param::Field::PreprocessForExtremes,
                ));
                state.set_alt_tomo_correct_ctf(boolean_of(
                    process_details,
                    &alt_tomo_setup_param::Field::CorrectCtf,
                ));
                state.set_alt_tomo_erase_fiducials(boolean_of(
                    process_details,
                    &alt_tomo_setup_param::Field::EraseFiducials,
                ));
                state.set_alt_tomo_filter_in_2d(boolean_of(
                    process_details,
                    &alt_tomo_setup_param::Field::FilterIn2d,
                ));
                state.set_alt_tomo_trim_vol_checked(boolean_of(
                    process_details,
                    &alt_tomo_setup_param::Field::TrimVolume,
                ));
                state.set_alt_tomo_rootname_to_process(
                    details
                        .and_then(|d| d.get_string(&alt_tomo_setup_param::Field::RootnameToProcess))
                        .as_deref(),
                );
                state.set_alt_tomo_even_and_odd_pairs(boolean_of(
                    process_details,
                    &alt_tomo_setup_param::Field::EvenAndOddPairs,
                ));
                state.set_alt_tomo_axis_to_process(
                    details
                        .and_then(|d| d.get_string(&alt_tomo_setup_param::Field::AxisToProcess))
                        .as_deref(),
                );
            }
        } else if process_name == Some(ProcessName::RESTRICTALIGN) {
            // Logged on the event dispatch thread, where the project log is.
            let app_manager = self.app_manager;
            event_queue::invoke_later(move || {
                app_manager.log_message_until_with_keyword(
                    Some(&file_type::CLASS.restrict_align_log),
                    Some(process_output_strings::RESTRICT_ALIGN_RERUNNING_TILT_ALIGN_MSG_ID),
                    Some("restrictalign:"),
                    Some(process_output_strings::SUCCESS_TAG),
                    Some(axis_id),
                );
                if app_manager.log_message_with_keyword_file_type(
                    Some(&file_type::CLASS.restrict_align_log),
                    Some(process_output_strings::RESTRICT_ALIGN_RERUNNING_TILT_ALIGN_MSG_ID),
                    None,
                    Some(axis_id),
                ) {
                    app_manager.log_message_loggable(
                        Some(
                            &app_manager
                                .get_process_mgr()
                                .generate_align_log_for_project_log(axis_id),
                        ),
                        Some(axis_id),
                    );
                }
            });
        } else if process_name == Some(ProcessName::REDUCE_FILT_VOL) {
            let reduce_filt_vol_log_file = file_type::CLASS
                .reduce_filt_vol_log
                .get_file(Some(self.app_manager), Some(axis_id));
            if let Some(file) = reduce_filt_vol_log_file
                && let Ok(text) = std::fs::read_to_string(&file)
            {
                for st in text.lines() {
                    if st.contains(
                        super::process_output_strings::REDUCE_FILT_VOL_NOT_ENOUGH_MEMORY_ERROR_TAG,
                    ) {
                        self.app_manager
                            .log_message(Some(reduce_filt_vol_param::NOT_ENOUGH_MEMORY_MSG));
                        self.open_message_dialog(
                            reduce_filt_vol_param::NOT_ENOUGH_MEMORY_MSG,
                            "Not Enough Memory",
                            axis_id,
                        );
                        break;
                    }
                }
            }
            state.set_reduce_filt_vol_flipped(boolean_of(
                process_details,
                &reduce_filt_vol_param::Field::IsReduceFiltVolFlipped,
            ));
        } else if process_name == Some(ProcessName::FLATTEN) {
            state.set_flatten_flipped(boolean_of(
                process_details,
                &warp_vol_param::Field::IsFlattenFlipped,
            ));
        } else if let Some(process_name) = &process_name {
            // For processes that can also be done with processchunks.
            self.post_process_name(Some(&process_name.to_string()), axis_id);
        }
    }

    /// Java override `errorProcess(ComScriptProcess)`.
    fn error_process_com_script(&self, _base: &BaseProcessManager, script: &ComScriptProcess) {
        let process_name = script.get_process_name();
        let command_details = script.get_command_details();
        let axis_id = script.get_axis_id();
        let state = self.app_manager.get_state();
        if process_name == Some(ProcessName::XCORR) {
            self.set_invalid_edge_functions(script.get_command(), false);
            if command_details.is_some() && mode_is(command_details, &blendmont_param::Mode::Xcorr)
            {
                // Do not allow tomodataplots -type 4 to unless you are sure that
                // blendmont ran.
                state.set_xcorr_blendmont_was_run(axis_id, false);
            }
        } else if process_name == Some(ProcessName::PREBLEND) {
            self.set_invalid_edge_functions(script.get_command(), false);
        } else if process_name == Some(ProcessName::COMBINE) {
            self.app_manager.error_process(
                axis_id,
                process_name.clone(),
                script.get_command().cloned(),
            );
        }
        if process_name == Some(ProcessName::NEWST)
            && mode_is(command_details, &newst_param::Mode::FullAlignedStack)
        {
            self.app_manager.update_aligned_stack_binning(axis_id);
        }
    }

    /// Java override `postProcess(BackgroundProcess)`.
    fn post_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        base.post_process_background_base(process);
        let transferfid_command_line = self.transferfid_command_line.lock().unwrap().clone();
        if Some(process.get_command_line()) == transferfid_command_line {
            base.write_log_file(
                process,
                process.get_axis_id(),
                dataset_files::TRANSFER_FID_LOG,
            );
            self.app_manager
                .get_state()
                .set_seeding_done(process.get_axis_id(), true);
            // Logged on the event dispatch thread, where the project log is.
            let app_manager = self.app_manager;
            let axis_id = process.get_axis_id();
            let user_dir = app_manager.get_property_user_dir();
            event_queue::invoke_later(move || {
                app_manager.log_message_loggable(
                    Some(&TransferFidLog::get_instance(
                        Some(app_manager),
                        axis_id,
                        user_dir.as_deref(),
                    )),
                    Some(axis_id),
                );
            });
        } else if process.get_process_name() == Some(ProcessName::FLATTEN_WARP) {
            let std_output = process.get_std_output();
            // Built and logged on the event dispatch thread, where the project
            // log is.
            let app_manager = self.app_manager;
            let axis_id = process.get_axis_id();
            event_queue::invoke_later(move || {
                let flatten_warp_log = FlattenWarpLog::new();
                flatten_warp_log.set_log(std_output);
                app_manager.log_message_loggable(Some(&flatten_warp_log), Some(axis_id));
            });
        } else if process.get_process_name() == Some(ProcessName::RUNRAPTOR) {
            self.app_manager
                .get_state()
                .set_use_raptor_result_warning(true);
        } else if process.get_process_name() == Some(ProcessName::EXCLUDE_VIEWS) {
            self.app_manager
                .msg_exclude_views_succeeded(process.get_axis_id());
        } else {
            let Some(command_name) = process.get_command_name() else {
                return;
            };
            let command = process.get_command();
            let state = self.app_manager.get_state();
            if command_name == trimvol_param::COMMAND_NAME {
                state.set_trimvol_flipped(
                    boolean_of(command, &trimvol_param::Fields::SwapYz)
                        || boolean_of(command, &trimvol_param::Fields::RotateX),
                );
                let meta_data = self.app_manager.get_meta_data();
                let input_file_name = TrimvolParam::get_input_file_name_for(
                    self.app_manager,
                    ConstMetaData::get_axis_type(meta_data),
                    Some(&meta_data.get_name()),
                );
                let mrc_header = MRCHeader::get_instance_in_dir(
                    self.app_manager.get_property_user_dir().as_deref(),
                    input_file_name.as_deref(),
                    Some(AxisID::Only),
                );
                // Upstream bug fixed in translation (ProcessManager.java:1743-1749): a
                // null header (no input file name) throws NullPointerException; here the
                // sizes are not set.
                if let Some(mrc_header) = mrc_header {
                    let mut mrc_header = mrc_header.borrow_mut();
                    match mrc_header.read_with_manager(self.app_manager) {
                        Ok(true) => {
                            state
                                .set_post_proc_trim_vol_input_n_columns(mrc_header.get_n_columns());
                            state.set_post_proc_trim_vol_input_n_rows(mrc_header.get_n_rows());
                            state.set_post_proc_trim_vol_input_n_sections(
                                mrc_header.get_n_sections(),
                            );
                        }
                        Ok(false) => {}
                        // `catch (IOException | InvalidParameterException e)`
                        Err(e) => eprintln!("{e}"),
                    }
                }
            } else if command_name == squeezevol_param::COMMAND_NAME {
                state.set_squeezevol_flipped(boolean_of(
                    command,
                    &squeezevol_param::Fields::Flipped,
                ));
            } else if command_name == archiveorig_param::command_name() {
                self.app_manager
                    .delete_original_stack(command.cloned(), process.get_std_output());
            } else if let Some(command) = command
                && command.get_process_name() == Some(ProcessName::CLIP)
                && equals_mode(command.get_command_mode(), &clip_param::Mode::Stats)
            {
                let log_file_name = format!(
                    "{}_stats.log",
                    command
                        .get_command_input_file()
                        .and_then(|file| file
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned()))
                        .unwrap_or_else(|| "null".to_owned())
                );
                base.write_log_file(process, process.get_axis_id(), &log_file_name);
                self.show_log_file(
                    &Path::new(&self.app_manager.get_property_user_dir().unwrap_or_default())
                        .join(&log_file_name),
                );
            }
        }
    }

    /// Java override `errorProcess(BackgroundProcess)`.
    fn error_process_background(&self, base: &BaseProcessManager, process: &BackgroundProcess) {
        let transferfid_command_line = self.transferfid_command_line.lock().unwrap().clone();
        if Some(process.get_command_line()) == transferfid_command_line {
            base.write_log_file(
                process,
                process.get_axis_id(),
                dataset_files::TRANSFER_FID_LOG,
            );
            self.show_log_file(
                &Path::new(&self.app_manager.get_property_user_dir().unwrap_or_default())
                    .join(dataset_files::TRANSFER_FID_LOG),
            );
        }
    }
}
