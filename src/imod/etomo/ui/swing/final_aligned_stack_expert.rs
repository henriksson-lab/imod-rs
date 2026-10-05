//! `IMOD/Etomo/src/etomo/ui/swing/FinalAlignedStackExpert.java`.
//!
//! Java `public final class FinalAlignedStackExpert extends ReconUIExpert`.
//! The superclass is the embedded [`ReconUIExpert`] `base` (reached through
//! `Deref`); the abstract methods are [`ReconUIExpertVirtual`] and the
//! interface is [`UIExpert`].  The expert lives on the event dispatch thread
//! as an `Rc`; every method takes `&self`, the `dialog` field is a
//! `RefCell<Option<Rc<..>>>` cloned out before each use, and no borrow is
//! held across a call to the manager or back into the dialog.
//!
//! Java private final `comScriptMgr` (`manager.getComScriptManager()`) is not
//! stored: the manager hands out a guard that must only be held for the
//! statement that uses it, so each Java `comScriptMgr.x(...)` is
//! `self.manager.get_com_script_manager().x(...)`.

use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::ccd_eraser_param;
use crate::imod::etomo::comscript::const_ctf_phase_flip_param::ConstCtfPhaseFlipParam;
use crate::imod::etomo::comscript::const_ctf_plotter_param::ConstCtfPlotterParam;
use crate::imod::etomo::comscript::const_mtf_filter_param::ConstMTFFilterParam;
use crate::imod::etomo::comscript::ctf_phase_flip_param::CtfPhaseFlipParam;
use crate::imod::etomo::comscript::ctf_plotter_param::CtfPlotterParam;
use crate::imod::etomo::comscript::mtf_filter_param::MTFFilterParam;
use crate::imod::etomo::comscript::processchunks_param::OutputImageFileKey;
use crate::imod::etomo::comscript::split_correction_param::SplitCorrectionParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::process_series::{Process, ProcessSeries, ProcessSeriesHandle};
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::simple_defocus_file;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result::ProcessResult;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::process_track::ProcessTrack;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::event_queue::EdtRef;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::blendmont_display::BlendmontDisplayException;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::final_aligned_stack_dialog::{self, FinalAlignedStackDialog, Tab};
use super::main_tomogram_panel::MainTomogramPanel;
use super::process_dialog::{DialogExitState, ProcessDialogVirtual};
use super::process_display::ProcessDisplay;
use super::recon_ui_expert::{ReconUIExpert, ReconUIExpertVirtual};
use super::ui_expert::UIExpert;
use super::ui_expert_utilities::UIExpertUtilities;
use super::ui_harness;

/// Java `public final class FinalAlignedStackExpert extends ReconUIExpert`.
pub struct FinalAlignedStackExpert {
    /// The `ReconUIExpert` superclass.
    base: ReconUIExpert,
    /// The Java `this` handed to `FinalAlignedStackDialog.getInstance`.
    this: Weak<FinalAlignedStackExpert>,
    /// Java private final `state`.
    state: &'static TomogramState,
    /// Java private final `screenState`.
    screen_state: &'static ReconScreenState,

    /// Java private `dialog`, initially null.
    dialog: RefCell<Option<Rc<FinalAlignedStackDialog>>>,
    /// Java private `advanced`.
    advanced: Cell<bool>,
    /// Java private `enableFiltering`.
    enable_filtering: Cell<bool>,
    /// Java private `curTab`, initially `FinalAlignedStackDialog.Tab.DEFAULT`.
    cur_tab: Cell<Tab>,
}

impl Deref for FinalAlignedStackExpert {
    type Target = ReconUIExpert;
    fn deref(&self) -> &ReconUIExpert {
        &self.base
    }
}

impl FinalAlignedStackExpert {
    /// Java `FinalAlignedStackExpert(ApplicationManager, MainTomogramPanel,
    /// ProcessTrack, AxisID)` (FinalAlignedStackExpert.java:65).
    pub fn new(
        manager: &'static ApplicationManager,
        main_panel: Option<Rc<MainTomogramPanel>>,
        process_track: Option<&'static ProcessTrack>,
        axis_id: AxisID,
    ) -> Rc<FinalAlignedStackExpert> {
        let instance = Rc::new_cyclic(|this| FinalAlignedStackExpert {
            base: ReconUIExpert::new(
                manager,
                main_panel,
                process_track,
                axis_id,
                DialogType::FinalAlignedStack,
            ),
            this: this.clone(),
            // comScriptMgr = manager.getComScriptManager(): see the module docs.
            state: manager.get_state(),
            screen_state: manager.get_screen_state(axis_id),
            dialog: RefCell::new(None),
            advanced: Cell::new(false),
            enable_filtering: Cell::new(false),
            cur_tab: Cell::new(Tab::DEFAULT),
        });
        let this: Weak<dyn ReconUIExpertVirtual> =
            Rc::downgrade(&instance) as Weak<dyn ReconUIExpertVirtual>;
        instance.base.set_this(this);
        instance
    }

    /// The Java `dialog` field read (null is `None`), cloned out so no borrow
    /// of the field is held.
    fn dialog(&self) -> Option<Rc<FinalAlignedStackDialog>> {
        self.dialog.borrow().clone()
    }

    /// Java public `updateDialog()`.
    pub fn update_dialog(&self) {
        let manager: &'static dyn BaseManager = self.manager;
        self.update_filter(
            file_type::CLASS
                .aligned_stack
                .exists(Some(manager), Some(self.axis_id)),
        );
    }

    /// Java private `updateCtfPlotterCom(boolean)`.
    fn update_ctf_plotter_com(&self, do_validation: bool) -> Option<CtfPlotterParam> {
        let mut param = self
            .manager
            .get_com_script_manager()
            .get_ctf_plotter_param(self.axis_id);
        if !self.get_parameters_ctf_plotter_param_boolean(&mut param, do_validation) {
            return None;
        }
        self.manager
            .get_com_script_manager()
            .save_ctf_plotter(&param, self.axis_id);
        Some(param)
    }

    /// Java private `updateCtfCorrectionCom(boolean)`.
    fn update_ctf_correction_com(&self, do_validation: bool) -> Option<CtfPhaseFlipParam> {
        let mut param = self
            .manager
            .get_com_script_manager()
            .get_ctf_phase_flip_param(self.axis_id);
        if !self.get_parameters_ctf_phase_flip_param_boolean(&mut param, do_validation) {
            return None;
        }
        self.manager
            .get_com_script_manager()
            .save_ctf_phase_flip(&param, self.axis_id);
        Some(param)
    }

    /// Java private `updateMTFFilterCom(boolean)`: update the mtffilter.com
    /// from the FinalAlignedStackDialog.  Returns the parameters if
    /// successful.
    fn update_mtf_filter_com(&self, do_validation: bool) -> Option<MTFFilterParam> {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        // Set a reference to the correct object
        if self.dialog().is_none() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(manager),
                    "Can not update mtffilter?.com without an active final aligned stack dialog",
                    "Program logic error",
                    Some(axis_id),
                )
            });
            return None;
        }
        // try { ... } catch (NumberFormatException | FortranInputSyntaxException
        // except) - two catch blocks with the same body.
        let mut mtf_filter_param = self
            .manager
            .get_com_script_manager()
            .get_mtf_filter_param(axis_id);
        match self.get_parameters_mtf_filter_param_boolean(&mut mtf_filter_param, do_validation) {
            Ok(false) => return None,
            Ok(true) => {}
            Err(except) => {
                let mut error_message: Vec<String> = vec![String::new(); 3];
                error_message[0] = "MTF Filter Parameter Syntax Error".to_string();
                error_message[1] = format!("Axis: {}", axis_id.get_extension());
                error_message[2] = except.to_string();
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(manager),
                        &error_message,
                        "MTF Filter Parameter Syntax Error",
                        Some(axis_id),
                    )
                });
                return None;
            }
        }
        // was metaData.getDatasetName() + AxisID.ONLY.getExtension() + ".ali";
        let input_file_name = file_type::CLASS
            .aligned_stack
            .get_file_name(Some(manager), Some(axis_id));
        // was metaData.getDatasetName() + AxisID.ONLY.getExtension() + "_filt.ali";
        let output_file_name = file_type::CLASS
            .mtf_filtered_stack
            .get_file_name(Some(manager), Some(axis_id));
        mtf_filter_param.set_input_file(input_file_name.as_deref());
        mtf_filter_param.set_output_file(output_file_name.as_deref());
        self.manager
            .get_com_script_manager()
            .save_mtf_filter(&mtf_filter_param, axis_id);
        Some(mtf_filter_param)
    }

    /// Java package-private `mtffilter(ProcessResultDisplay, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn mtffilter(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self.manager,
                self.axis_id,
                Some(self.dialog_type),
                Some("mtffilter"),
            ),
        };
        if self.dialog().is_none() {
            process_series.borrow().end_series();
            return;
        }
        self.send_msg_process_starting(process_result_display.as_ref());
        let Some(param) = self.update_mtf_filter_com(true) else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        self.set_dialog_state(ProcessState::InProgress);
        let result = self.manager.mtffilter(
            Arc::new(param),
            self.axis_id,
            process_result_display.clone(),
            Some(process_series),
        );
        self.send_msg(result, process_result_display.as_ref());
    }

    /// Java package-private `ctfPlotter(ProcessResultDisplay)`.
    pub fn ctf_plotter(&self, process_result_display: Option<ProcessResultDisplayHandle>) {
        if self.dialog().is_none() {
            return;
        }
        self.send_msg_process_starting(process_result_display.as_ref());
        let param = self.update_ctf_plotter_com(true);
        if param.is_none() {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            return;
        }
        self.set_dialog_state(ProcessState::InProgress);
        let result = self
            .manager
            .ctf_plotter(self.axis_id, process_result_display.clone());
        self.send_msg(result, process_result_display.as_ref());
    }

    /// Java package-private `createSimpleDefocusFile(boolean)`: create or
    /// reuse a _simple.defocus file.  Returns true if it succeeds.
    pub fn create_simple_defocus_file(&self, do_validation: bool) -> bool {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        // Upstream bug fixed in translation (FinalAlignedStackExpert.java:494):
        // `dialog` is read without a null check (NullPointerException when there
        // is no dialog).  A missing dialog is a failure here, the same answer as
        // a field that fails validation.
        let Some(dialog) = self.dialog() else {
            return false;
        };
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let Ok(expected_defocus) = dialog.get_expected_defocus(do_validation) else {
            return false;
        };
        let expected_defocus_in_nanometers =
            utilities::convert_microns_to_nanometers(expected_defocus.as_deref());
        let Ok(phase_shift_in_degrees) = dialog.get_phase_shift_in_degrees(do_validation) else {
            return false;
        };
        // A Java text field's text is never null.
        let phase_shift_in_degrees = phase_shift_in_degrees.unwrap_or_default();
        if simple_defocus_file::is_up_to_date(
            manager,
            axis_id,
            &expected_defocus_in_nanometers,
            &phase_shift_in_degrees,
        ) {
            return true;
        }
        if !simple_defocus_file::write_file(
            manager,
            axis_id,
            &expected_defocus_in_nanometers,
            &phase_shift_in_degrees,
        ) {
            let message = format!(
                "{} may not be up to date.  Continue?",
                dataset_files::get_simple_defocus_file_name(manager, Some(axis_id))
            );
            if !ui_harness::with(|harness| {
                harness.open_yes_no_dialog_base_manager_string_axis_id(
                    Some(manager),
                    &message,
                    Some(axis_id),
                )
            }) {
                return false;
            }
        }
        true
    }

    /// Java package-private `ctfCorrection(ProcessResultDisplay,
    /// ProcessSeries, Deferred3dmodButton, Run3dmodMenuOptions,
    /// ProcessingMethod)`.
    pub fn ctf_correction_process_result_display_process_series_deferred_3dmod_button_run_3dmod_menu_options_processing_method(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        correction_processing_method: ProcessingMethod,
    ) {
        let Some(dialog) = self.dialog() else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self.manager,
                self.axis_id,
                Some(self.dialog_type),
                Some("ctfCorrection"),
            ),
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        if dialog.is_use_expected_defocus() && !self.create_simple_defocus_file(true) {
            process_series.borrow().end_series();
            return;
        }
        if !correction_processing_method.is_local() {
            self.splitcorrection(
                process_result_display,
                Some(process_series),
                correction_processing_method,
            );
        } else {
            self.ctf_correction_process_result_display_process_series(
                process_result_display,
                Some(process_series),
            );
        }
    }

    /// Java package-private `ctfCorrection(ProcessResultDisplay, ProcessSeries)`.
    pub fn ctf_correction_process_result_display_process_series(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        if self.dialog().is_none() {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        self.send_msg_process_starting(process_result_display.as_ref());
        let Some(param) = self.update_ctf_correction_com(true) else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        self.set_dialog_state(ProcessState::InProgress);
        let result = self.manager.ctf_correction(
            Arc::new(param),
            self.axis_id,
            process_result_display.clone(),
            process_series,
        );
        self.send_msg(result, process_result_display.as_ref());
    }

    /// Java private `updateSplitCorrectionParam(boolean)`.
    fn update_split_correction_param(&self, do_validation: bool) -> Option<SplitCorrectionParam> {
        self.dialog()?;
        let mut param = SplitCorrectionParam::new(self.axis_id);
        if !self.get_parameters_split_correction_param_boolean(&mut param, do_validation) {
            return None;
        }
        Some(param)
    }

    /// Java package-private `getParameters(SplitCorrectionParam, boolean)`.
    pub fn get_parameters_split_correction_param_boolean(
        &self,
        param: &mut SplitCorrectionParam,
        do_validation: bool,
    ) -> bool {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        // Upstream bug fixed in translation (FinalAlignedStackExpert.java:578):
        // `getParallelPanel().getCPUsSelected(...)` has no null check
        // (NullPointerException when the axis has no parallel panel).  A missing
        // panel is treated like a field that fails validation.
        let Some(parallel_panel) = self.get_parallel_panel() else {
            return false;
        };
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let Ok(cpus) = parallel_panel.get_cpus_selected(do_validation) else {
            return false;
        };
        param.set_cpus(cpus.as_deref());
        let header = MRCHeader::get_instance_in_dir(
            self.manager.get_property_user_dir().as_deref(),
            file_type::CLASS
                .aligned_stack
                .get_file_name(Some(manager), Some(axis_id))
                .as_deref(),
            Some(axis_id),
        );
        if let Some(header) = header {
            // try { if (header.read(manager)) ... } catch (IOException |
            // InvalidParameterException e) { e.printStackTrace(); }
            let read = header.borrow_mut().read_with_manager(manager);
            match read {
                Ok(true) => {
                    let n_sections = header.borrow().get_n_sections();
                    param.set_max_z(n_sections);
                }
                Ok(false) => {}
                Err(e) => eprintln!("{e}"),
            }
        }
        true
    }

    /// Java package-private `splitcorrection(ProcessResultDisplay,
    /// ProcessSeries, ProcessingMethod)`.
    pub fn splitcorrection(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        correction_processing_method: ProcessingMethod,
    ) {
        if self.dialog().is_none() {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        self.send_msg_process_starting(process_result_display.as_ref());
        if self.update_ctf_correction_com(true).is_none() {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let Some(param) = self.update_split_correction_param(true) else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        self.set_dialog_state(ProcessState::InProgress);
        let process_result = self.manager.split_correction(
            self.axis_id,
            process_result_display.clone(),
            process_series,
            Arc::new(param),
            self.dialog_type,
            Some(correction_processing_method),
        );
        if process_result.is_some() {
            self.send_msg(process_result, process_result_display.as_ref());
        }
    }

    /// Java package-private `useCtfCorrection(ProcessResultDisplay)`: replace
    /// the full aligned stack with the ctf corrected full aligned stack created
    /// by ctfcorrection.com.
    pub fn use_ctf_correction(&self, process_result_display: Option<ProcessResultDisplayHandle>) {
        if self.manager.use_file_as_full_aligned_stack(
            process_result_display,
            &file_type::CLASS.ctf_corrected_stack,
            final_aligned_stack_dialog::CTF_CORRECTION_LABEL,
            self.axis_id,
            self.dialog_type,
        ) {
            self.state
                .set_use_ctf_correction_warning(self.axis_id, false);
        }
    }

    /// Java package-private `useMtfFilter(ProcessResultDisplay)`: replace the
    /// full aligned stack with the filtered full aligned stack created from
    /// mtffilter.
    pub fn use_mtf_filter(&self, process_result_display: Option<ProcessResultDisplayHandle>) {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        if self.dialog().is_none() {
            return;
        }
        self.send_msg_process_starting(process_result_display.as_ref());
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        if self
            .manager
            .is_axis_busy(axis_id, process_result_display_ref)
        {
            return;
        }
        self.start_progress_bar("Using filtered full aligned stack", axis_id);
        // Java `FileType.getFile` never returns null; an unresolvable file is
        // one that does not exist.
        let mtf_filtered_stack: PathBuf = file_type::CLASS
            .mtf_filtered_stack
            .get_file(Some(manager), Some(axis_id))
            .unwrap_or_default();
        if !mtf_filtered_stack.exists() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(manager),
                    "The filtered full aligned stack doesn't exist.  Create the filtered full aligned stack first",
                    "Filtered full aligned stack missing",
                    Some(axis_id),
                )
            });
            self.stop_progress_bar_axis_id(axis_id);
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            return;
        }
        self.set_dialog_state(ProcessState::InProgress);
        if file_type::CLASS
            .aligned_stack
            .get_file(Some(manager), Some(axis_id))
            .as_deref()
            .is_some_and(Path::exists)
            && mtf_filtered_stack.exists()
        {
            if !utilities::is_valid_stack_file(&mtf_filtered_stack, manager, Some(axis_id)) {
                let message = format!(
                    "{} is not a valid MRC file.",
                    utilities::java_io_file_get_name(&mtf_filtered_stack.to_string_lossy())
                );
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(manager),
                        &message,
                        "Entry Error",
                        Some(axis_id),
                    )
                });
                self.stop_progress_bar_axis_id(axis_id);
                self.send_msg(
                    Some(ProcessResult::FAILED_TO_START),
                    process_result_display.as_ref(),
                );
                return;
            }
            match self
                .manager
                .backup_image_file(Some(&file_type::CLASS.aligned_stack), Some(axis_id))
            {
                Ok(()) => {}
                // catch (final LockException e)
                Err(LogFileError::Lock(_)) => {
                    self.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        ProcessEndState::FileLockFailure,
                    );
                    self.send_msg(Some(ProcessResult::FAILED), process_result_display.as_ref());
                    return;
                }
                // catch (final IOException | LogFileException except)
                Err(except) => {
                    let message = format!(
                        "Unable to backup {}\n{}",
                        file_type::CLASS
                            .aligned_stack
                            .get_file_name(Some(manager), Some(axis_id))
                            .unwrap_or_else(|| "null".to_string()),
                        except
                    );
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(manager),
                            &message,
                            "File Rename Error (1)",
                            Some(axis_id),
                        )
                    });
                    self.stop_progress_bar_axis_id(axis_id);
                    self.send_msg(Some(ProcessResult::FAILED), process_result_display.as_ref());
                    return;
                }
            }
        }
        match self.manager.rename_image_file(
            Some(&file_type::CLASS.mtf_filtered_stack),
            Some(&file_type::CLASS.aligned_stack),
            Some(axis_id),
        ) {
            Ok(()) => {}
            // catch (final LockException e)
            Err(LogFileError::Lock(_)) => {
                self.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    ProcessEndState::FileLockFailure,
                );
                self.send_msg(Some(ProcessResult::FAILED), process_result_display.as_ref());
                return;
            }
            // catch (final IOException | LogFileException except)
            Err(except) => {
                // System.err.println(except.getClass()); except.printStackTrace();
                eprintln!("{except:?}");
                let message = except.to_string();
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(manager),
                        &message,
                        "File Rename Error (2)",
                        Some(axis_id),
                    )
                });
                self.stop_progress_bar_axis_id(axis_id);
                self.send_msg(Some(ProcessResult::FAILED), process_result_display.as_ref());
                return;
            }
        }
        self.stop_progress_bar_axis_id(axis_id);
        self.send_msg(
            Some(ProcessResult::SUCCEEDED),
            process_result_display.as_ref(),
        );
        self.state.set_use_filtered_stack_warning(axis_id, false);
    }

    /// Java private `updateFilter(boolean)`.
    fn update_filter(&self, enable: bool) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        self.enable_filtering.set(enable);
        dialog.set_filter_button_enabled(self.enable_filtering.get());
        dialog.set_view_filter_button_enabled(self.enable_filtering.get());
        self.enable_use_filter();
    }

    /// Java public `updateAlignedStackBinning()`.
    pub fn update_aligned_stack_binning(&self) {
        if let Some(dialog) = self.dialog() {
            dialog.update_aligned_stack_binning();
        }
    }

    /// Java package-private `enableUseFilter()`.
    pub fn enable_use_filter(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        if !self.enable_filtering.get() {
            dialog.set_use_filter_enabled(false);
            return;
        }
        let starting_and_ending_z: String = dialog.get_starting_and_ending_z();
        // Java `startingAndEndingZ.matches("\\s+")`: the whole string is one or
        // more whitespace characters ([ \t\n\x0B\f\r]).
        if starting_and_ending_z.is_empty()
            || starting_and_ending_z
                .chars()
                .all(|c| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
        {
            // btnFilter.setSelected(false);
            dialog.set_use_filter_enabled(true);
        } else {
            dialog.set_use_filter_enabled(false);
        }
    }

    /// Java package-private `getConfigDir()`.
    pub fn get_config_dir(&self) -> PathBuf {
        let calib_dir: Option<PathBuf> = etomo_director::INSTANCE.get_imod_calib_directory();
        // Upstream bug fixed in translation (FinalAlignedStackExpert.java:756):
        // `calibDir.exists()` on a null calibration directory (IMOD_CALIB_DIR
        // unset) throws a NullPointerException out of the dialog constructor.
        // A missing calibration directory is treated as one that does not
        // exist, so the working directory is used.
        if let Some(calib_dir) = calib_dir.filter(|calib_dir| calib_dir.exists()) {
            let dir = calib_dir.join("CTFnoise");
            if dir.exists() {
                return dir;
            }
            return calib_dir;
        }
        PathBuf::from(self.manager.get_property_user_dir().unwrap_or_default())
    }

    /// Java private `getParameters(MTFFilterParam, boolean) throws
    /// FortranInputSyntaxException`.
    fn get_parameters_mtf_filter_param_boolean(
        &self,
        mtf_filter_param: &mut MTFFilterParam,
        do_validation: bool,
    ) -> Result<bool, final_aligned_stack_dialog::MTFFilterParametersException> {
        let manager: &'static dyn BaseManager = self.manager;
        mtf_filter_param.set_pixel_size(
            self.meta_data.get_pixel_size()
                * f64::from(utilities::get_stack_binning_for_file_type(
                    manager,
                    self.axis_id,
                    &file_type::CLASS.aligned_stack,
                )),
        );

        if let Some(dialog) = self.dialog() {
            return dialog.get_parameters_mtf_filter_param_boolean(mtf_filter_param, do_validation);
        }
        Ok(false)
    }

    /// Java private `setParameters(ConstCtfPlotterParam)`.
    fn set_parameters_const_ctf_plotter_param(&self, param: &dyn ConstCtfPlotterParam) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_config_file(Some(&param.get_config_file()));
        if param.is_scan_defocus_range_low() || param.is_scan_defocus_range_high() {
            dialog.set_scan_defocus_range(Some(&format!(
                "{},{}",
                self.scale_scan_defocus_range_double(param.get_scan_defocus_range_low()),
                self.scale_scan_defocus_range_double(param.get_scan_defocus_range_high())
            )));
        }
        dialog.set_expected_defocus(&utilities::convert_nanometers_to_microns(
            param.get_expected_defocus(),
        ));
        dialog.set_phase_shift_in_degrees(Some(&param.get_phase_shift_in_degrees()));
        dialog.set_offset_to_add(param.get_offset_to_add());
    }

    /// Java private `getParameters(CtfPlotterParam, boolean)`.
    fn get_parameters_ctf_plotter_param_boolean(
        &self,
        param: &mut CtfPlotterParam,
        do_validation: bool,
    ) -> bool {
        let Some(dialog) = self.dialog() else {
            return false;
        };
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        macro_rules! field {
            ($e:expr) => {
                match $e {
                    Ok(value) => value,
                    Err(_) => return false,
                }
            };
        }
        param.set_voltage(field!(dialog.get_voltage(do_validation)).as_deref());
        param.set_spherical_aberration(
            field!(dialog.get_spherical_aberration(do_validation)).as_deref(),
        );
        param.set_invert_tilt_angles(dialog.get_invert_tilt_angles());
        param.set_amplitude_contrast(
            field!(dialog.get_amplitude_contrast(do_validation)).as_deref(),
        );
        let scan_defocus_range_microns = field!(dialog.get_scan_defocus_range(do_validation));
        let expected_defocus = field!(dialog.get_expected_defocus(do_validation));
        if utilities::is_empty(scan_defocus_range_microns.as_deref())
            && utilities::is_empty(expected_defocus.as_deref())
            && do_validation
        {
            let manager: &'static dyn BaseManager = self.manager;
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(manager),
                    "Either scan defocus range or expected defocus needs to be filled in. One, the other, or both are acceptable",
                    "Etomo Error",
                    Some(self.axis_id),
                )
            });
            return false;
        }
        if !utilities::is_empty(scan_defocus_range_microns.as_deref()) {
            // Java `scanDefocusRangeMicrons.replaceAll("\\s", "")`.
            let microns_no_whitespace: String = scan_defocus_range_microns
                .as_deref()
                .unwrap_or("")
                .chars()
                .filter(|c| !matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
                .collect();
            // Java `split(",")`, which drops trailing empty strings.
            let mut arr_microns: Vec<&str> = microns_no_whitespace.split(',').collect();
            while arr_microns.last().is_some_and(|value| value.is_empty()) {
                arr_microns.pop();
            }
            // try { ... } catch (final FortranInputSyntaxException e) {
            // e.printStackTrace(); }
            //
            // Upstream bug fixed in translation
            // (FinalAlignedStackExpert.java:816-817): `arrMicrons[0]` and
            // `arrMicrons[1]` are indexed unchecked, so a range with one value
            // throws ArrayIndexOutOfBoundsException out of the save.  A missing
            // element is passed as null here, which `setScanDefocusRange`
            // rejects as a syntax error (printed, as the source's own catch
            // does).
            let low = arr_microns
                .first()
                .map(|value| self.scale_scan_defocus_range_string(Some(value)));
            let high = arr_microns
                .get(1)
                .map(|value| self.scale_scan_defocus_range_string(Some(value)));
            if let Err(e) = param.set_scan_defocus_range(low.as_deref(), high.as_deref()) {
                eprintln!("{e}");
            }
        } else {
            param.reset_scan_defocus_range();
        }
        param.set_expected_defocus(Some(&utilities::convert_microns_to_nanometers(
            field!(dialog.get_expected_defocus(do_validation)).as_deref(),
        )));
        param.set_phase_shift_in_degrees(
            field!(dialog.get_phase_shift_in_degrees(do_validation)).as_deref(),
        );
        param.set_offset_to_add(field!(dialog.get_offset_to_add(do_validation)).as_deref());
        param.set_config_file(dialog.get_config_file().as_deref());
        true
    }

    // <p>updates done</p>

    /// Java public `setTiltState()`.
    pub fn set_tilt_state(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_tilt_state(self.state, self.meta_data);
    }

    /// Java private `setParameters(ConstCtfPhaseFlipParam, boolean)`.
    fn set_parameters_const_ctf_phase_flip_param_boolean(
        &self,
        param: &dyn ConstCtfPhaseFlipParam,
        initialize: bool,
    ) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.set_voltage(param.get_voltage());
        dialog.set_spherical_aberration(param.get_spherical_aberration());
        dialog.set_invert_tilt_angles(param.get_invert_tilt_angles());
        dialog.set_amplitude_contrast(param.get_amplitude_contrast());
        dialog.set_use_expected_defocus(
            param
                .get_defocus_file()
                .ends_with(Some(dataset_files::SIMPLE_DEFOCUS_EXT)),
        );
        dialog.set_interpolation_width(param.get_interpolation_width());
        dialog.set_ctf_phase_flip_x_axis_tilt(Some(&param.get_x_axis_tilt()), false);
        dialog.set_scale_by_ctf_power(Some(&param.get_scale_by_ctf_power()));
        dialog.set_minimum_zero_spacing(Some(&param.get_minimum_zero_spacing()));
        dialog.set_defocus_tol(param.get_defocus_tol());
        dialog.set_parameters_const_ctf_phase_flip_param_boolean(param, initialize);
        dialog.update_ctf_plotter();
    }

    /// Java private `getParameters(CtfPhaseFlipParam, boolean)`.
    fn get_parameters_ctf_phase_flip_param_boolean(
        &self,
        param: &mut CtfPhaseFlipParam,
        do_validation: bool,
    ) -> bool {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        let Some(dialog) = self.dialog() else {
            return false;
        };
        dialog.get_parameters_ctf_phase_flip_param(param);
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        macro_rules! field {
            ($e:expr) => {
                match $e {
                    Ok(value) => value,
                    Err(_) => return false,
                }
            };
        }
        param.set_voltage(field!(dialog.get_voltage(do_validation)).as_deref());
        param.set_spherical_aberration(
            field!(dialog.get_spherical_aberration(do_validation)).as_deref(),
        );
        param.set_invert_tilt_angles(dialog.get_invert_tilt_angles());
        param.set_amplitude_contrast(
            field!(dialog.get_amplitude_contrast(do_validation)).as_deref(),
        );
        if dialog.is_use_expected_defocus() {
            param.set_defocus_file(Some(&dataset_files::get_simple_defocus_file_name(
                manager,
                Some(axis_id),
            )));
        } else {
            param.set_defocus_file(Some(&dataset_files::get_ctf_plotter_file_name(
                manager,
                Some(axis_id),
            )));
        }
        param.set_interpolation_width(
            field!(dialog.get_interpolation_width(do_validation)).as_deref(),
        );
        param.set_x_axis_tilt(
            field!(dialog.get_ctf_phase_flip_x_axis_tilt(do_validation)).as_deref(),
        );
        param.set_scale_by_ctf_power(
            field!(dialog.get_scale_by_ctf_power(do_validation)).as_deref(),
        );
        param.set_minimum_zero_spacing(
            field!(dialog.get_minimum_zero_spacing(do_validation)).as_deref(),
        );
        param.set_defocus_tol(field!(dialog.get_defocus_tol(do_validation)).as_deref());
        param.set_output_file_name(
            file_type::CLASS
                .ctf_corrected_stack
                .get_file_name(Some(manager), Some(axis_id))
                .as_deref(),
        );
        let setup_pixel_size = self.meta_data.get_pixel_size();
        param.set_pixel_size(
            setup_pixel_size
                * f64::from(utilities::get_stack_binning_for_file_type(
                    manager,
                    axis_id,
                    &file_type::CLASS.aligned_stack,
                )),
        );
        param.update_unbinned_pixel_size(setup_pixel_size);
        true
    }

    /// Java private `setParameters(ConstMTFFilterParam)`.
    fn set_parameters_const_mtf_filter_param(
        &self,
        mtf_filter_param: &dyn ConstMTFFilterParam,
    ) -> bool {
        if let Some(dialog) = self.dialog() {
            dialog.set_parameters_const_mtf_filter_param(mtf_filter_param);
            self.enable_use_filter();
            return true;
        }
        false
    }

    /// Java private `scaleScanDefocusRange(Double)`.
    fn scale_scan_defocus_range_double(&self, input: Option<f64>) -> String {
        let microns: ConstEtomoNumber = utilities::convert_nanometers_to_microns_double(input);
        microns.to_string()
    }

    /// Java private `scaleScanDefocusRange(String)`.
    fn scale_scan_defocus_range_string(&self, input: Option<&str>) -> String {
        utilities::convert_microns_to_nanometers(input)
    }
}

impl ReconUIExpertVirtual for FinalAlignedStackExpert {
    /// Java package-private override `doneDialog()`.
    fn done_dialog_void(&self) {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        let Some(dialog) = self.dialog() else {
            return;
        };
        self.cur_tab.set(dialog.get_cur_tab());
        let exit_state = dialog.get_exit_state();
        if exit_state != DialogExitState::Cancel
            && !self.manager.is_exiting()
            && exit_state != DialogExitState::Postpone
        {
            if self.state.is_use_ctf_correction_warning(axis_id) {
                // The use button wasn't pressed and the user is moving on to the next
                // dialog. Don't put this message in the log.
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        None,
                        &format!(
                            "To use the CTF correction go back to Final Aligned Stack and press the \"{}\" button in the {} tab.",
                            final_aligned_stack_dialog::USE_CTF_CORRECTION_LABEL,
                            final_aligned_stack_dialog::CTF_TAB_LABEL
                        ),
                        "Entry Warning",
                        Some(axis_id),
                    )
                });
                // Only warn once.
                self.state.set_use_ctf_correction_warning(axis_id, false);
            }
            if self.state.is_use_erased_stack_warning(axis_id) {
                // The use button wasn't pressed and the user is moving on to the next
                // dialog. Don't put this message in the log.
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        None,
                        &format!(
                            "To use the stack with the erased beads go back to Final Aligned Stack and press the \"{}\" button in the {} tab.",
                            FinalAlignedStackDialog::get_use_erased_stack_label(),
                            FinalAlignedStackDialog::get_erased_stack_tab_label()
                        ),
                        "Entry Warning",
                        Some(axis_id),
                    )
                });
                // Only warn once.
                self.state.set_use_erased_stack_warning(axis_id, false);
            }
            if self.state.is_use_filtered_stack_warning(axis_id) {
                // The use button wasn't pressed and the user is moving on to the next
                // dialog. Don't put this message in the log.
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        None,
                        &format!(
                            "To use the MTF filtered stack go back to Final Aligned Stack and press the \"{}\" button in the {} tab.",
                            final_aligned_stack_dialog::USE_FILTERED_STACK_LABEL,
                            final_aligned_stack_dialog::MTF_FILTER_TAB_LABEL
                        ),
                        "Entry Warning",
                        Some(axis_id),
                    )
                });
                // Only warn once.
                self.state.set_use_filtered_stack_warning(axis_id, false);
            }
        }
        if exit_state == DialogExitState::Execute {
            manager.close_imod(
                Some(imod_manager::MTF_FILTER_KEY),
                Some(axis_id),
                Some("MTF filtered stack"),
                false,
            );
            manager.close_imod(
                Some(imod_manager::CTF_CORRECTION_KEY),
                Some(axis_id),
                Some("CTF corrected stack"),
                false,
            );
            manager.close_imod(
                Some(imod_manager::ERASED_FIDUCIALS_KEY),
                Some(axis_id),
                Some("Erased beads stack"),
                false,
            );
            manager.close_imod(
                Some(imod_manager::FINE_ALIGNED_3D_FIND_KEY),
                Some(axis_id),
                Some("Aligned stack for 3d find"),
                false,
            );
            manager.close_imod(
                Some(imod_manager::FULL_VOLUME_3D_FIND_KEY),
                Some(axis_id),
                Some("tomogram 3d find"),
                false,
            );
        }
        if exit_state != DialogExitState::Cancel {
            self.save_dialog_void();
        }
        // Clean up the existing dialog
        self.leave_dialog(exit_state);
        // Hold onto the finished dialog (don't set dialog to null) in case anything
        // is running that needs it or there are next processes that need it.
    }

    /// Java package-private override `saveDialog()`.
    fn save_dialog_void(&self) {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        let Some(dialog) = self.dialog() else {
            return;
        };
        self.advanced.set(dialog.is_advanced());
        // Get the user input data from the dialog box
        // try { ... } catch (final FortranInputSyntaxException e)
        if let Err(e) = dialog.get_parameters_meta_data(self.meta_data) {
            let message = e.get_message().unwrap_or("null").to_string();
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(manager),
                    &message,
                    "Data File Error",
                )
            });
        }
        dialog.get_parameters_recon_screen_state(self.screen_state);
        let fiducialess_params = dialog.get_fiducialess_params();
        UIExpertUtilities::INSTANCE
            .update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
                self.manager,
                &*fiducialess_params,
                axis_id,
                false,
            );
        if self.meta_data.get_view_type() == ViewType::Montage {
            // try { ... } catch (FortranInputSyntaxException |
            // InvalidParameterException | IOException e) - three catch blocks with
            // the same body.  Java hands the displays over without a null check;
            // both are present for a montage (the dialog built a BlendmontPanel
            // and the erase-gold panel a Blendmont3dFindPanel).
            let result: Result<(), BlendmontDisplayException> = (|| {
                if let Some(blendmont_display) = dialog.get_blendmont_display() {
                    self.manager
                        .update_blend_com(&*blendmont_display, axis_id, false, false)?;
                }
                if let Some(blendmont3d_find_display) = dialog.get_blendmont3d_find_display() {
                    self.manager.update_blend3d_find_com(
                        &*blendmont3d_find_display,
                        axis_id,
                        false,
                        false,
                    )?;
                }
                Ok(())
            })();
            if let Err(e) = result {
                let message = e.to_string();
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(manager),
                        &message,
                        "Update Com Error",
                    )
                });
            }
        } else {
            let newstack_display = dialog.get_newstack_display();
            self.manager
                .update_newst_com(newstack_display.as_deref(), axis_id, false, false);
            let newstack3d_find_display = dialog.get_newstack3d_find_display();
            self.manager.update_newst3d_find_com(
                newstack3d_find_display.as_deref(),
                axis_id,
                false,
                false,
            );
        }
        self.update_mtf_filter_com(false);
        self.update_ctf_plotter_com(false);
        self.update_ctf_correction_com(false);
        let tilt3d_find_display = dialog.get_tilt3d_find_display();
        self.manager
            .update_tilt3d_find_com(tilt3d_find_display.as_deref(), axis_id, false);
        // Java passes the displays without a null check; the erase-gold panel
        // always has them.
        if let Some(find_beads3d_display) = dialog.get_find_beads3d_display() {
            self.manager
                .update_find_beads3d_com(&*find_beads3d_display, axis_id, false);
        }
        if let Some(ccd_eraser_beads_display) = dialog.get_ccd_eraser_beads_display() {
            self.manager
                .update_gold_eraser_param(&*ccd_eraser_beads_display, axis_id, false);
        }
        self.manager.save_storables(Some(axis_id));
    }

    /// Java package-private override `getDialog()`.
    fn get_dialog(&self) -> Option<Rc<dyn ProcessDialogVirtual>> {
        self.dialog()
            .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>)
    }
}

impl UIExpert for FinalAlignedStackExpert {
    /// Java override `openDialog()`: open the final aligned stack dialog.
    fn open_dialog(&self) {
        if !self.can_show_dialog() {
            return;
        }
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        let meta_data = self.meta_data;
        let action_message = self
            .manager
            .set_current_dialog_type(Some(self.dialog_type), Some(axis_id));
        let existing = self.dialog();
        if self.show_dialog(
            existing.as_ref().map(|dialog| dialog.process_dialog()),
            action_message.as_deref(),
        ) {
            return;
        }
        // Create the dialog and show it.
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("FinalAlignedStackDialog"),
            Some(utilities::STARTED_STATUS),
        );
        let dialog = FinalAlignedStackDialog::get_instance(
            self.manager,
            self.this.clone(),
            axis_id,
            self.cur_tab.get(),
        );
        *self.dialog.borrow_mut() = Some(dialog.clone());
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("FinalAlignedStackDialog"),
            Some(utilities::FINISHED_STATUS),
        );

        dialog.initialize();
        dialog.set_parameters_const_meta_data(meta_data);

        // Find out if this is the first time this dialog was opened. Tilt3dfind is now the
        // only com file that's created when this dialog is opened.
        let property_user_dir = self.manager.get_property_user_dir().unwrap_or_default();
        let tilt3d_find_com = PathBuf::from(utilities::java_io_file_new(
            &property_user_dir,
            &ProcessName::TILT_3D_FIND.get_comscript(axis_id),
        ));
        let mut new_tilt3d_find_com_and_init = false;
        if !tilt3d_find_com.exists() {
            new_tilt3d_find_com_and_init = true;
        }

        // no longer managing image size

        // Read in the newst{|a|b}.com parameters. WARNING this needs to be done
        // before reading the tilt paramers below so that the GUI knows how to
        // correctly scale the dimensions - from when full alignment and tilt where
        // on the same dialog
        if meta_data.get_view_type() == ViewType::Montage {
            self.manager.get_com_script_manager().load_blend(axis_id);
            let blend_param = self
                .manager
                .get_com_script_manager()
                .get_blend_param(axis_id);
            dialog.set_parameters_blendmont_param(&blend_param);
            // if blend_3dfind.com doesn't exist copy blend.com to blend_3dfind.com.
            let blend3d_find_com = PathBuf::from(utilities::java_io_file_new(
                &property_user_dir,
                &ProcessName::BLEND_3D_FIND.get_comscript(axis_id),
            ));
            // try { ... } catch (IOException | LogFileException | LockException e)
            let result: Result<(), LogFileError> = (|| {
                if !blend3d_find_com.exists() {
                    utilities::copy_file(
                        Some(manager),
                        Some(axis_id),
                        Some(&PathBuf::from(utilities::java_io_file_new(
                            &property_user_dir,
                            &ProcessName::BLEND.get_comscript(axis_id),
                        ))),
                        Some(&blend3d_find_com),
                        false,
                        false,
                        false,
                    )?;
                }
                self.manager
                    .get_com_script_manager()
                    .load_blend3d_find(axis_id);
                let blend_param = self
                    .manager
                    .get_com_script_manager()
                    .get_blend_param_from_blend3d_find(axis_id);
                dialog.set_erase_gold_parameters_blendmont_param(&blend_param);
                Ok(())
            })();
            if result.is_err() {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(manager),
                        "Unable to copy to blend.com to blend_3d_find.com.  Will not be able to use findbeads3d when erasing gold.",
                        "Etomo Error",
                    )
                });
            }
        } else {
            self.manager.get_com_script_manager().load_newst(axis_id);
            let newst_param = self
                .manager
                .get_com_script_manager()
                .get_newst_com_newst_param(axis_id);
            dialog.set_parameters_const_newst_param(&newst_param); // TEMP newstparam
            // if newst_3dfind.com doesn't exist copy newst.com to newst_3dfind.com.
            let newst3d_find_com = PathBuf::from(utilities::java_io_file_new(
                &property_user_dir,
                &ProcessName::NEWST_3D_FIND.get_comscript(axis_id),
            ));
            // try { ... } catch (IOException | LogFileException | LockException e)
            let result: Result<(), LogFileError> = (|| {
                if !newst3d_find_com.exists() {
                    utilities::copy_file(
                        Some(manager),
                        Some(axis_id),
                        Some(&PathBuf::from(utilities::java_io_file_new(
                            &property_user_dir,
                            &ProcessName::NEWST.get_comscript(axis_id),
                        ))),
                        Some(&newst3d_find_com),
                        false,
                        false,
                        false,
                    )?;
                }
                self.manager
                    .get_com_script_manager()
                    .load_newst3d_find(axis_id);
                let newst_param = self
                    .manager
                    .get_com_script_manager()
                    .get_newst_param_from_newst3d_find(axis_id);
                dialog.set_erase_gold_parameters_newst_param(&newst_param);
                Ok(())
            })();
            if result.is_err() {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string(
                        Some(manager),
                        "Unable to copy to newst.com to newst_3d_find.com.  Will not be able to use findbeads3d when erasing gold.",
                        "Etomo Error",
                    )
                });
            }
        }
        // if tilt_3dfind.com doesn't exist copy tilt.com to tilt_3dfind.com.
        // try { ... } catch (IOException | LogFileException | LockException e) -
        // the catch also covers the IOException `setParameters(ConstTiltParam,
        // boolean)` declares.
        let result: Result<(), String> = (|| {
            if new_tilt3d_find_com_and_init {
                utilities::copy_file(
                    Some(manager),
                    Some(axis_id),
                    Some(&PathBuf::from(utilities::java_io_file_new(
                        &property_user_dir,
                        &ProcessName::TILT.get_comscript(axis_id),
                    ))),
                    Some(&tilt3d_find_com),
                    false,
                    false,
                    false,
                )
                .map_err(|e| e.to_string())?;
            }
            self.manager
                .get_com_script_manager()
                .load_tilt3d_find(axis_id);
            let tilt_param = self
                .manager
                .get_com_script_manager()
                .get_tilt_param_from_tilt3d_find(axis_id);
            dialog
                .set_parameters_const_tilt_param_boolean(&tilt_param, new_tilt3d_find_com_and_init)
                .map_err(|e| e.to_string())?;
            Ok(())
        })();
        if result.is_err() {
            // The source reuses the blend message here (sic).
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(manager),
                    "Unable to copy to blend.com to blend_3d_find.com.  Will not be able to use findbeads3d when erasing gold.",
                    "Etomo Error",
                )
            });
        }
        // Backwards compatibility
        // If track light beads isn't be saved to state, get light beads from
        // track.com.
        self.manager.get_com_script_manager().load_track(axis_id);
        self.manager
            .get_com_script_manager()
            .load_find_beads3d(axis_id);
        let find_beads3d_param = self
            .manager
            .get_com_script_manager()
            .get_find_beads3d_param(axis_id);
        dialog.set_parameters_const_find_beads3d_param_boolean(
            &find_beads3d_param,
            new_tilt3d_find_com_and_init,
        );
        self.manager
            .get_com_script_manager()
            .load_tilt3d_find_reproject(axis_id);

        // Get the align{|a|b}.com parameters
        self.manager.get_com_script_manager().load_align(axis_id);
        let tiltalign_param = self
            .manager
            .get_com_script_manager()
            .get_tiltalign_param(axis_id);
        dialog.set_parameters_const_tiltalign_param_boolean(
            &tiltalign_param,
            new_tilt3d_find_com_and_init,
        );

        // Load gold erase comfile
        let gold_eraser_loaded = self
            .manager
            .get_com_script_manager()
            .load_gold_eraser(axis_id, false);
        if gold_eraser_loaded {
            let ccd_eraser_param = self
                .manager
                .get_com_script_manager()
                .get_ccd_eraser_param_from_gold_eraser(
                    axis_id,
                    Some(ccd_eraser_param::Mode::Beads),
                );
            dialog.set_parameters_const_ccd_eraser_param(&ccd_eraser_param);
        }

        // backward compatibility
        // Try loading ctfcorrection.com first. If it isn't there, copy it with
        // copytomocoms and load it.-++
        // Then try loading ctfplotter.com. If it isn't there, copy it with
        // copytomocoms, using the voltage, etc from ctfcorrection.com.
        // Ignore ctfplotter.param.
        let mut new_ctf_correction = false;
        let ctf_correction_loaded = self
            .manager
            .get_com_script_manager()
            .load_ctf_correction(axis_id, false);
        if !ctf_correction_loaded {
            new_ctf_correction = true;
            self.manager.setup_ctf_correction_com_script(axis_id);
            self.manager
                .get_com_script_manager()
                .load_ctf_correction(axis_id, true);
        }
        let ctf_phase_flip_param = self
            .manager
            .get_com_script_manager()
            .get_ctf_phase_flip_param(axis_id);
        self.set_parameters_const_ctf_phase_flip_param_boolean(
            &ctf_phase_flip_param,
            new_ctf_correction || new_tilt3d_find_com_and_init,
        );
        let ctf_plotter_loaded = self
            .manager
            .get_com_script_manager()
            .load_ctf_plotter(axis_id, false);
        if !ctf_plotter_loaded {
            // Get the voltage, etc from ctfcorrection.com, since it has been loaded, and it
            // is somewhat more likely that it was updated
            self.manager
                .setup_ctf_plotter_com_script(axis_id, &ctf_phase_flip_param);
            self.manager
                .get_com_script_manager()
                .load_ctf_plotter(axis_id, true);
        }
        let mut param = self
            .manager
            .get_com_script_manager()
            .get_ctf_plotter_param(axis_id);
        if meta_data.is_stack_ctf_auto_fit_range_and_step_set(axis_id) {
            param.set_auto_fit_range_and_step(
                &meta_data.get_stack_ctf_auto_fit_range_and_step(axis_id),
            );
        }
        self.set_parameters_const_ctf_plotter_param(&param);
        self.manager
            .get_com_script_manager()
            .save_ctf_plotter(&param, axis_id);
        dialog.set_parameters_recon_screen_state(self.screen_state);
        self.manager
            .get_com_script_manager()
            .load_mtf_filter(axis_id);
        let mtf_filter_param = self
            .manager
            .get_com_script_manager()
            .get_mtf_filter_param(axis_id);
        self.set_parameters_const_mtf_filter_param(&mtf_filter_param);
        // updateDialog()
        self.update_filter(
            file_type::CLASS
                .aligned_stack
                .exists(Some(manager), Some(axis_id)),
        );

        // Set the fidcialess state and tilt axis angle
        // From updateFiducialessParams
        dialog.set_fiducialess_alignment(meta_data.is_fiducialess_alignment(axis_id));
        dialog.set_image_rotation(Some(&meta_data.get_image_rotation(axis_id).to_string()));
        dialog.set_tilt_state(self.state, meta_data);
        dialog.set_override_parameters(meta_data);
        // Load CTF phase flip XAxisTilt from tilt.com if it unavailable.
        if dialog.is_ctf_phase_flip_x_axis_tilt_empty() {
            self.manager.get_com_script_manager().load_tilt(axis_id);
            let tilt_param = self
                .manager
                .get_com_script_manager()
                .get_tilt_param(axis_id);
            dialog.set_tilt_com_parameters(&tilt_param);
        }
        self.open_dialog_process_dialog_string(dialog.process_dialog(), action_message.as_deref());
    }

    /// Java override `startNextProcess(ProcessSeries.Process,
    /// ProcessResultDisplay, ProcessSeries, DialogType, ProcessDisplay)`:
    /// start the next process specified by the nextProcess string.  Returns
    /// true if the process is recognized.
    fn start_next_process(
        &self,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: Option<DialogType>,
        display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        let manager: &'static dyn BaseManager = self.manager;
        let axis_id = self.axis_id;
        if process.equals_string(Some(&ProcessName::PROCESSCHUNKS.to_string())) {
            let dialog = self.dialog();
            self.processchunks(
                manager,
                dialog
                    .as_ref()
                    .map(|dialog| dialog.process_dialog() as &dyn AbstractParallelDialog),
                process_result_display,
                process_series,
                // Java `process.getSubprocessName().toString()`: a null
                // subprocess name would throw there and prints "null" here.
                &format!(
                    "{}{}",
                    process
                        .get_subprocess_name()
                        .map_or_else(|| "null".to_owned(), |name| name.to_string()),
                    axis_id.get_extension()
                ),
                process
                    .get_output_image_file_key()
                    .cloned()
                    .map(OutputImageFileKey::FileKey),
                process.get_processing_method(),
                false,
            );
            return true;
        }
        if process.equals_string(Some(&ProcessName::TILT_3D_FIND.to_string())) {
            // `(TiltDisplay) display`.
            let tilt_display = display
                .as_deref()
                .and_then(|display| display.as_tilt_display());
            // Java passes null for the Run3dmodMenuOptions; with a null
            // Deferred3dmodButton the process series ignores the options, so
            // the no-options value is equivalent.
            // Fixed in translation: a null dialog type or processing method makes
            // Java's tilt3dFindAction throw NullPointerException
            // (`tiltProcessingMethod.isLocal()`); the process is not started.
            if let (Some(dialog_type), Some(processing_method)) =
                (dialog_type, process.get_processing_method())
            {
                self.manager.tilt3d_find_action(
                    process_result_display,
                    process_series,
                    None,
                    Run3dmodMenuOptions::default(),
                    tilt_display,
                    axis_id,
                    dialog_type,
                    processing_method,
                );
            }
            return true;
        }
        false
    }

    /// Java inherited final `ReconUIExpert.saveAction()`.
    fn save_action(&self) {
        self.base.save_action();
    }

    /// Java inherited final `ReconUIExpert.saveDialog(DialogExitState)`.
    fn save_dialog(&self, exit_state: DialogExitState) {
        self.base.save_dialog_dialog_exit_state(exit_state);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
