//! `IMOD/Etomo/src/etomo/ui/swing/TomogramPositioningExpert.java`.
//!
//! Java `public final class TomogramPositioningExpert extends ReconUIExpert`.
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
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::{self, BlendmontParam, ConvertError};
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::cryo_position_param::CryoPositionParam;
use crate::imod::etomo::comscript::find_section_param::FindSectionParam;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::comscript::newst_param::{self, NewstParam, SetSizeToOutputInXandYError};
use crate::imod::etomo::comscript::process_details::ProcessDetails;
use crate::imod::etomo::comscript::tilt_param::{self, TiltParam};
use crate::imod::etomo::comscript::tiltalign_param::TiltalignParam;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::process_series::{Process, ProcessSeries, ProcessSeriesHandle};
use crate::imod::etomo::storage::tomopitch_log::TomopitchLog;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::Number;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::pos_sample_type::PosSampleType;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result::ProcessResult;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::process_track::ProcessTrack;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::utilities;

use super::deferred_3dmod_button::Deferred3dmodButton;
use super::main_tomogram_panel::MainTomogramPanel;
use super::process_dialog::{DialogExitState, ProcessDialogVirtual};
use super::process_display::ProcessDisplay;
use super::recon_ui_expert::{ReconUIExpert, ReconUIExpertVirtual};
use super::text_page_window::TextPageWindow;
use super::tomogram_positioning_dialog::TomogramPositioningDialog;
use super::ui_expert::UIExpert;
use super::ui_expert_utilities::UIExpertUtilities;
use super::ui_harness;

/// Java package-private static final `SAMPLE_TOMOGRAMS_LABEL`.
pub const SAMPLE_TOMOGRAMS_LABEL: &str = "Create Sample Tomograms";

/// Java `public final class TomogramPositioningExpert extends ReconUIExpert`.
pub struct TomogramPositioningExpert {
    /// The `ReconUIExpert` superclass.
    base: ReconUIExpert,
    /// The Java `this` handed to `TomogramPositioningDialog.getInstance`.
    this: Weak<TomogramPositioningExpert>,
    /// Java private final `state`.
    state: &'static TomogramState,
    /// Java private final `axisType`.
    axis_type: AxisType,
    /// Java private `dialog`.
    dialog: RefCell<Option<Rc<TomogramPositioningDialog>>>,
    /// Java private `advanced`.
    advanced: Cell<bool>,
}

impl Deref for TomogramPositioningExpert {
    type Target = ReconUIExpert;
    fn deref(&self) -> &ReconUIExpert {
        &self.base
    }
}

impl TomogramPositioningExpert {
    /// Java `TomogramPositioningExpert(ApplicationManager, MainTomogramPanel,
    /// ProcessTrack, AxisID, AxisType)` (TomogramPositioningExpert.java:65).
    pub fn new(
        manager: &'static ApplicationManager,
        main_panel: Option<Rc<MainTomogramPanel>>,
        process_track: Option<&'static ProcessTrack>,
        axis_id: AxisID,
        axis_type: AxisType,
    ) -> Rc<TomogramPositioningExpert> {
        let instance = Rc::new_cyclic(|this| TomogramPositioningExpert {
            base: ReconUIExpert::new(
                manager,
                main_panel,
                process_track,
                axis_id,
                DialogType::TomogramPositioning,
            ),
            this: this.clone(),
            axis_type,
            // comScriptMgr = manager.getComScriptManager(): see the module docs.
            state: manager.get_state(),
            dialog: RefCell::new(None),
            advanced: Cell::new(false),
        });
        let this: Weak<dyn ReconUIExpertVirtual> =
            Rc::downgrade(&instance) as Weak<dyn ReconUIExpertVirtual>;
        instance.base.set_this(this);
        instance
    }

    /// The Java `dialog` field read (null is `None`), cloned out so no borrow
    /// of the field is held.
    fn dialog(&self) -> Option<Rc<TomogramPositioningDialog>> {
        self.dialog.borrow().clone()
    }

    /// Java `postProcess(ProcessDetails, TomogramState, PosSampleType)`
    /// (TomogramPositioningExpert.java:101): post processing for sample and
    /// tilt.  ProcessDetails is required in both cases.
    ///
    /// Called from the manager's post-processing, which posts it to the EDT
    /// (the dialog is an EDT object).
    pub fn post_process(
        &self,
        process_details: &dyn ProcessDetails,
        state: &TomogramState,
        pos_sample_type: PosSampleType,
    ) {
        // `TiltParam` answers every field read here; Java's
        // IllegalArgumentException for another field is `None`, read as 0 /
        // false.
        state.set_pos_sample_type(self.axis_id, Some(pos_sample_type));
        if pos_sample_type == PosSampleType::WholeCryo {
            state.set_sample_x_axis_tilt(self.axis_id, 0.0);
        } else {
            state.set_sample_x_axis_tilt(
                self.axis_id,
                process_details
                    .get_double_value(&tilt_param::Field::XAxisTilt)
                    .unwrap_or_default(),
            );
        }
        let fiducialess = process_details
            .get_boolean_value(&tilt_param::Field::Fiducialess)
            .unwrap_or_default();
        state.set_sample_fiducialess(self.axis_id, fiducialess);
        if !fiducialess {
            state.set_sample_axis_z_shift_const_etomo_number(
                self.axis_id,
                Some(&*state.get_align_axis_z_shift(self.axis_id)),
            );
            state.set_sample_angle_offset_const_etomo_number(
                self.axis_id,
                Some(&*state.get_align_angle_offset(self.axis_id)),
            );
        } else {
            // no alignment for fidless
            state.set_sample_axis_z_shift_double(
                self.axis_id,
                process_details
                    .get_double_value(&tilt_param::Field::ZShift)
                    .unwrap_or_default(),
            );
            state.set_sample_angle_offset_double(
                self.axis_id,
                process_details
                    .get_double_value(&tilt_param::Field::TiltAngleOffset)
                    .unwrap_or_default(),
            );
        }
        if let Some(dialog) = self.dialog() {
            dialog.update_display();
        }
    }

    /// Java package-private `sampleAction(ProcessResultDisplay, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions)` (TomogramPositioningExpert.java:202).
    pub fn sample_action(
        &self,
        sample: Option<ProcessResultDisplayHandle>,
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
                Some("sampleAction"),
            ),
        };
        let Some(dialog) = self.dialog() else {
            process_series.borrow().end_series();
            return;
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        if !dialog.is_sample_type_cryo() {
            if dialog.is_whole_tomogram() {
                self.whole_tomogram(sample, Some(process_series));
            } else {
                self.create_sample(sample, Some(process_series));
            }
        } else {
            self.cryo_position(sample, Some(process_series));
        }
    }

    /// Java package-private `createBoundary(Run3dmodMenuOptions)`
    /// (TomogramPositioningExpert.java:225).
    pub fn create_boundary(&self, menu_options: Option<Run3dmodMenuOptions>) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        // ApplicationManager takes Run3dmodMenuOptions by value; a null (from the
        // local action listener) is the default, empty menu options.
        let menu_options = menu_options.unwrap_or_default();
        if dialog.is_whole_tomogram() {
            self.manager.imod_full_sample(self.axis_id, menu_options);
        } else {
            self.manager.imod_sample(self.axis_id, menu_options);
        }
    }

    /// Java package-private `fiducialessAction()` (TomogramPositioningExpert.java:264).
    pub fn fiducialess_action(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.update_display();
        // Save tilt param.
        let mut tilt_param = self
            .manager
            .get_com_script_manager()
            .get_tilt_param(self.axis_id);
        dialog.set_parameters_fiducialess(&mut tilt_param, self.meta_data);
        self.manager
            .update_exclude_list(&mut tilt_param, self.axis_id);
        self.manager
            .get_com_script_manager()
            .save_tilt(&tilt_param, self.axis_id);
        self.meta_data
            .set_fiducialess(self.axis_id, tilt_param.is_fiducialess());
        ui_harness::INSTANCE.with(|harness| {
            harness.pack_axis_id_base_manager(
                Some(self.axis_id),
                Some(self.manager as &'static dyn BaseManager),
            )
        });
    }

    /// Java `createSample(ProcessResultDisplay, ProcessSeries)`
    /// (TomogramPositioningExpert.java:303): run the sample com script.
    pub fn create_sample(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self.manager,
                self.axis_id,
                Some(self.dialog_type),
                Some("createSample"),
            ),
        };
        self.send_msg_process_starting(process_result_display.as_ref());
        // Make sure that we have an active positioning dialog
        let Some(dialog) = self.dialog() else {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Can not update sample.com without an active positioning dialog",
                "Program logic error",
                Some(self.axis_id),
            );
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        };
        // Get the user input data from the dialog box
        if !UIExpertUtilities::INSTANCE
            .update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
                self.manager,
                &*dialog,
                self.axis_id,
                true,
            )
        {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        }
        let Some(tilt_param) = self.update_tomo_pos_tilt_com(true, true) else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        };
        self.manager
            .get_com_script_manager()
            .load_tilt(self.axis_id);
        self.set_dialog_state(ProcessState::InProgress);
        let aligned_stack: &FileKey = &file_type::CLASS.aligned_stack;
        self.manager
            .close_imod_file_key(Some(aligned_stack), Some(self.axis_id), true);
        if dialog.is_sample_type_auto() {
            process_series
                .borrow_mut()
                .set_next_process(Some(&ProcessName::FIND_SECTION.to_string()), None);
        }
        let result = self.manager.create_sample(
            self.axis_id,
            process_result_display.clone(),
            Some(process_series),
            tilt_param,
        );
        self.send_msg(result, process_result_display.as_ref());
    }

    /// Java `cryoPosition(ProcessResultDisplay, ProcessSeries)`
    /// (TomogramPositioningExpert.java:341).
    pub fn cryo_position(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self.manager,
                self.axis_id,
                Some(self.dialog_type),
                Some("cryoPosition"),
            ),
        };
        self.send_msg_process_starting(process_result_display.as_ref());
        // Make sure that we have an active positioning dialog
        let Some(dialog) = self.dialog() else {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Can not run cryposition without an active positioning dialog",
                "Program logic error",
                Some(self.axis_id),
            );
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        };
        // Get the user input data from the dialog box
        if !UIExpertUtilities::INSTANCE
            .update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
                self.manager,
                &*dialog,
                self.axis_id,
                true,
            )
        {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        }
        let Some(tilt_param) = self.update_tomo_pos_tilt_com(true, true) else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        };
        let param = self.update_cryo_position_com(true, process_result_display.as_ref());
        // Post process requires information from tilt.com.
        let Some(mut param) = param else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        };
        let tilt_param: Arc<dyn ConstTiltParam + Send + Sync> = tilt_param;
        param.set_tilt_param(Some(tilt_param));
        self.set_dialog_state(ProcessState::InProgress);
        let aligned_stack: &FileKey = &file_type::CLASS.aligned_stack;
        self.manager
            .close_imod_file_key(Some(aligned_stack), Some(self.axis_id), true);
        process_series
            .borrow_mut()
            .set_next_process_task(POST_CRYO_POSITION.with(|task| task.clone()));
        let result = self.manager.cryo_position(
            self.axis_id,
            process_result_display.clone(),
            Some(process_series),
            Arc::new(param),
        );
        self.send_msg(result, process_result_display.as_ref());
    }

    /// Java `makeCryoPositionComfile(AxisID)` (TomogramPositioningExpert.java:384).
    /// The parameter shadows the field, as in the Java.
    pub fn make_cryo_position_comfile(&self, axis_id: AxisID) -> bool {
        let Some(dialog) = self.dialog() else {
            return false;
        };
        let mut param = MakecomfileParam::new(
            self.manager,
            axis_id,
            file_type::CLASS.cryo_position_comscript.clone(),
        );
        dialog.get_parameters_makecomfile_param_boolean(&mut param, true);
        self.manager.makecomfile(axis_id, &mut param)
    }

    /// Java `wholeTomogram(ProcessResultDisplay, ProcessSeries)`
    /// (TomogramPositioningExpert.java:399): create a whole tomogram for
    /// positioning the tomogram in the volume.
    pub fn whole_tomogram(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self.manager,
                self.axis_id,
                Some(self.dialog_type),
                Some("wholeTomogram"),
            ),
        };
        self.send_msg_process_starting(process_result_display.as_ref());
        // Make sure that we have an active positioning dialog
        let Some(dialog) = self.dialog() else {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Can not save comscripts without an active positioning dialog",
                "Program logic error",
                Some(self.axis_id),
            );
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        };
        // Get the user input from the dialog
        if !UIExpertUtilities::INSTANCE
            .update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
                self.manager,
                &*dialog,
                self.axis_id,
                true,
            )
        {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        }
        let mut newst_param: Option<NewstParam> = None;
        let mut blendmont_param: Option<BlendmontParam> = None;
        if self.meta_data.get_view_type() != ViewType::Montage {
            newst_param = self.update_newst_com();
            if newst_param.is_none() {
                self.send_msg(
                    Some(ProcessResult::FAILED_TO_START),
                    process_result_display.as_ref(),
                );
                process_series.borrow().end_series();
                return;
            }
        } else {
            // Java catches FortranInputSyntaxException, InvalidParameterException
            // and IOException separately, each with the same dialog;
            // `ConvertError` carries the first as `FortranInputSyntax` and the
            // other two as `MontagesizeRead`.
            match self.update_blend_com() {
                Ok(param) => blendmont_param = param,
                Err(e) => {
                    ui_harness::open_message_dialog_from_process(
                        Some(self.manager),
                        &e.to_string(),
                        "Update Com Error",
                        None,
                    );
                }
            }
            if blendmont_param.is_none() {
                self.send_msg(
                    Some(ProcessResult::FAILED_TO_START),
                    process_result_display.as_ref(),
                );
                process_series.borrow().end_series();
                return;
            }
        }
        if self.update_tomo_pos_tilt_com(true, true).is_none() {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            process_series.borrow().end_series();
            return;
        }
        self.set_dialog_state(ProcessState::InProgress);
        let process_result = if self.meta_data.get_view_type() != ViewType::Montage {
            self.manager
                .whole_tomogram_axis_id_process_result_display_process_series_const_newst_param(
                    self.axis_id,
                    process_result_display.clone(),
                    Some(process_series.clone()),
                    // Non-null: checked above.
                    Arc::new(newst_param.unwrap()),
                )
        } else {
            self.manager
                .whole_tomogram_axis_id_process_result_display_process_series_blendmont_param(
                    self.axis_id,
                    process_result_display.clone(),
                    Some(process_series.clone()),
                    // Non-null: checked above.
                    Arc::new(blendmont_param.unwrap()),
                )
        };
        if process_result.is_some() {
            self.send_msg(process_result, process_result_display.as_ref());
            return;
        }
        process_series
            .borrow_mut()
            .set_next_process(Some(&ProcessName::TILT.to_string()), None);
        if dialog.is_sample_type_auto() {
            process_series
                .borrow_mut()
                .set_last_process(Some(&ProcessName::FIND_SECTION.to_string()));
        }
    }

    /// Java `tomopitch(ProcessResultDisplay, ProcessSeries)`
    /// (TomogramPositioningExpert.java:475).
    pub fn tomopitch(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        self.send_msg_process_starting(process_result_display.as_ref());
        if self.dialog().is_none() {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Can not save comscript without an active positioning dialog",
                "Program logic error",
                Some(self.axis_id),
            );
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        if !self.update_tomopitch_com(true) {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        self.set_dialog_state(ProcessState::InProgress);
        let result =
            self.manager
                .tomopitch(self.axis_id, process_result_display.clone(), process_series);
        self.send_msg(result, process_result_display.as_ref());
    }

    /// Java private `openTomopitchLog()` (TomogramPositioningExpert.java:505):
    /// open the tomopitch log file.
    fn open_tomopitch_log(&self) {
        let log_file_name =
            dataset_files::get_tomopitch_log_file_name(self.manager, Some(self.axis_id));
        // TextPageWindow(): the font size comes from the first UIManager
        // FontUIResource, which is not modelled; UIParameters' default stands in.
        let log_file_window = TextPageWindow::new(super::ui_parameters::DEFAULT_FONT_SIZE as i32);
        let file = format!(
            "{}{}{}",
            self.manager
                .get_property_user_dir()
                .unwrap_or_else(|| "null".to_owned()),
            std::path::MAIN_SEPARATOR,
            log_file_name
        );
        let visible = log_file_window.set_file_from_file_name(file);
        log_file_window.set_visible(visible);
    }

    /// Java `setTomopitchOutput()` (TomogramPositioningExpert.java:518).  If
    /// dialog is null, then the tomopitch output will not end up on the tomo
    /// pos dialog.  But this is unlikely to happen because tomopitch runs very
    /// fast.  The values will have to be save somewhere else if this is a
    /// problem.
    ///
    /// Called by the manager from the process thread; the manager posts it
    /// to the EDT (the dialog is an EDT object).
    pub fn set_tomopitch_output(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        let log = TomopitchLog::new(self.manager, self.axis_id);
        if !dialog.set_parameters_tomopitch_log(&log) {
            self.open_tomopitch_log();
        }
    }

    /// Java `finalAlign(ProcessResultDisplay, ProcessSeries)`
    /// (TomogramPositioningExpert.java:528).
    pub fn final_align(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        self.send_msg_process_starting(process_result_display.as_ref());
        if self.dialog().is_none() {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Can not save comscript without an active positioning dialog",
                "Program logic error",
                Some(self.axis_id),
            );
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let Some(tiltalign_param) = self.update_align_com(true) else {
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
        let result = self.manager.final_align(
            self.axis_id,
            process_result_display.clone(),
            process_series,
            Arc::new(tiltalign_param),
        );
        self.send_msg(result, process_result_display.as_ref());
    }

    /// Java private `sampleTilt(ProcessResultDisplay, ProcessSeries)`
    /// (TomogramPositioningExpert.java:559): whole tomogram.
    fn sample_tilt(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        self.manager
            .get_com_script_manager()
            .load_tilt(self.axis_id);
        let mut tilt_param = self
            .manager
            .get_com_script_manager()
            .get_tilt_param(self.axis_id);
        tilt_param.set_command_mode(tilt_param::Mode::Whole);
        tilt_param.set_fiducialess(self.meta_data.is_fiducialess(self.axis_id));
        let result = self.manager.sample_tilt(
            self.axis_id,
            process_result_display.clone(),
            process_series,
            Arc::new(tilt_param),
        );
        self.send_msg(result, process_result_display.as_ref());
    }

    /// Java private `findSection(ProcessResultDisplay, ProcessSeries)`
    /// (TomogramPositioningExpert.java:569).
    fn find_section(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        // Upstream bug fixed in translation: TomogramPositioningExpert.java:573
        // dereferences `dialog` without a null check, so a find-section
        // process reached with no dialog throws a NullPointerException out of
        // the process-series callback.  Here a missing dialog fails the
        // process to start instead: the message goes to the display and the
        // series ends, as the other entry points of this class do.
        let Some(dialog) = self.dialog() else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.as_ref(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let result = self.manager.find_section(
            self.axis_id,
            process_result_display.clone(),
            process_series,
            Arc::new(FindSectionParam::new(
                self.manager,
                self.axis_id,
                dialog.is_whole_tomogram(),
            )),
        );
        self.send_msg(result, process_result_display.as_ref());
    }

    /// Java private `updateAlignCom(boolean)` (TomogramPositioningExpert.java:577).
    fn update_align_com(&self, do_validation: bool) -> Option<TiltalignParam> {
        let dialog = self.dialog()?;
        // Java also catches NumberFormatException here ("Tiltalign Parameter
        // Syntax Error", "Axis: " + axisID.getExtension(), message); none of
        // the translated callees raise it.
        let mut tiltalign_param = self
            .manager
            .get_com_script_manager()
            .get_tiltalign_param(self.axis_id);
        if !dialog.get_align_params(&mut tiltalign_param, self.meta_data, do_validation) {
            return None;
        }
        self.roll_align_com_angles();
        self.manager
            .get_com_script_manager()
            .save_align(&tiltalign_param, self.axis_id);
        // update xfproduct in align.com
        let mut xfproduct_param = self
            .manager
            .get_com_script_manager()
            .get_xfproduct_in_align(self.axis_id);
        if let Err(except) = xfproduct_param.set_scale_shifts(
            UIExpertUtilities::INSTANCE.get_stack_binning_base_manager_axis_id_file_type(
                self.manager,
                self.axis_id,
                &file_type::CLASS.prealigned_stack,
            ),
        ) {
            except.print_stack_trace();
            let error_message = [
                "Xfproduct Parameter Syntax Error".to_string(),
                format!("Axis: {}", self.axis_id.get_extension()),
                except.to_string(),
            ];
            ui_harness::open_message_dialog_array_from_process(
                Some(self.manager),
                &error_message,
                "Xfproduct Parameter Syntax Error",
                Some(self.axis_id),
            );
            return None;
        }
        self.manager
            .get_com_script_manager()
            .save_xfproduct_in_align(&xfproduct_param, self.axis_id);
        Some(tiltalign_param)
    }

    /// Java private `updateTomoPosTiltCom(boolean, boolean)`
    /// (TomogramPositioningExpert.java:621): update the tilt{|a|b}.com file
    /// with sample parameters for the specified axis.
    fn update_tomo_pos_tilt_com(
        &self,
        positioning: bool,
        do_validation: bool,
    ) -> Option<Arc<TiltParam>> {
        // Make sure that we have an active positioning dialog
        let Some(dialog) = self.dialog() else {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Can not update sample.com without an active positioning dialog",
                "Program logic error",
                Some(self.axis_id),
            );
            return None;
        };
        // Get the current tilt parameters, make any user changes and save the
        // parameters back to the tilt{|a|b}.com
        // Java also catches NumberFormatException here ("Tilt Parameter Syntax
        // Error", "Axis: " + axisID.getExtension(), message); none of the
        // translated callees raise it.
        let mut tilt_param = self
            .manager
            .get_com_script_manager()
            .get_tilt_param(self.axis_id);
        tilt_param.set_fiducialess(self.meta_data.is_fiducialess(self.axis_id));
        if !dialog.get_tilt_params(&mut tilt_param, self.meta_data, do_validation) {
            return None;
        }
        // if not postioning then just saving tilt.com to continue, so want the
        // final thickness instead of the sample thickness.
        if positioning && !dialog.get_tilt_params_for_sample(&mut tilt_param, do_validation) {
            return None;
        }
        // get the command mode right
        if !dialog.is_whole_tomogram() {
            tilt_param.set_command_mode(tilt_param::Mode::Sample);
        } else {
            tilt_param.set_command_mode(tilt_param::Mode::Whole);
        }
        dialog.get_parameters_meta_data(self.meta_data);
        let output_file_name = file_type::CLASS
            .tilt_output
            .get_file_name(Some(self.manager), Some(self.axis_id));
        // outputFileName = metaData.getDatasetName() + "_full.rec";
        // outputFileName = metaData.getDatasetName() + axisID.getExtension() + ".rec";
        tilt_param.set_output_file(output_file_name.as_deref());
        if self.meta_data.get_view_type() == ViewType::Montage {
            // binning is currently always 1 and correct size should be coming from
            // copytomocoms
            // tiltParam.setMontageFullImage(propertyUserDir,
            // tomogramPositioningDialog.getBinning());
        }
        self.roll_tilt_com_angles();
        self.manager
            .update_exclude_list(&mut tilt_param, self.axis_id);
        self.manager
            .get_com_script_manager()
            .save_tilt(&tilt_param, self.axis_id);
        self.meta_data
            .set_fiducialess(self.axis_id, tilt_param.is_fiducialess());
        Some(Arc::new(tilt_param))
    }

    /// Java private `updateTomopitchCom(boolean)` (TomogramPositioningExpert.java:689):
    /// update the tomopitch{|a|b}.com file with sample parameters for the
    /// specified axis.
    fn update_tomopitch_com(&self, do_validation: bool) -> bool {
        // Make sure that we have an active positioning dialog
        let Some(dialog) = self.dialog() else {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Can not update tomopitch.com without an active positioning dialog",
                "Program logic error",
                Some(self.axis_id),
            );
            return false;
        };
        // Get the current tilt parameters, make any user changes and save the
        // parameters back to the tilt{|a|b}.com
        // Java also catches NumberFormatException here ("Tomopitch Parameter
        // Syntax Error", "Axis: " + axisID.getExtension(), message); none of
        // the translated callees raise it.
        let mut tomopitch_param = self
            .manager
            .get_com_script_manager()
            .get_tomopitch_param(self.axis_id);
        if !dialog.get_tomopitch_param(&mut tomopitch_param, self.meta_data, do_validation) {
            return false;
        }
        self.manager
            .get_com_script_manager()
            .save_tomopitch(&tomopitch_param, self.axis_id);
        true
    }

    /// Java private `updateNewstCom()` (TomogramPositioningExpert.java:727):
    /// update the newst{|a|b}.com scripts with the parameters from the
    /// tomogram positioning dialog.
    fn update_newst_com(&self) -> Option<NewstParam> {
        let dialog = self.dialog()?;
        // Get the whole tomogram positions state
        self.meta_data
            .set_whole_tomogram_sample(self.axis_id, dialog.is_whole_tomogram());
        let mut newst_param = self
            .manager
            .get_com_script_manager()
            .get_newst_com_newst_param(self.axis_id);
        self.get_newst_param(&mut newst_param);
        self.manager
            .get_com_script_manager()
            .save_newst(&newst_param, self.axis_id);
        Some(newst_param)
    }

    /// Java private `updateBlendCom() throws FortranInputSyntaxException,
    /// InvalidParameterException, IOException` (TomogramPositioningExpert.java:748):
    /// update the blend{|a|b}.com scripts with the parameters from the
    /// tomogram positioning dialog.
    fn update_blend_com(&self) -> Result<Option<BlendmontParam>, ConvertError> {
        let Some(dialog) = self.dialog() else {
            return Ok(None);
        };
        // Get the whole tomogram positions state
        self.meta_data
            .set_whole_tomogram_sample(self.axis_id, dialog.is_whole_tomogram());
        let mut blendmont_param = self
            .manager
            .get_com_script_manager()
            .get_blend_param(self.axis_id);
        self.get_parameters_blendmont_param(&mut blendmont_param)?;
        self.manager
            .get_com_script_manager()
            .save_blend(&blendmont_param, self.axis_id);
        Ok(Some(blendmont_param))
    }

    /// Java private `updateCryoPositionCom(boolean, ProcessResultDisplay)`
    /// (TomogramPositioningExpert.java:761).
    fn update_cryo_position_com(
        &self,
        for_run: bool,
        process_result_display: Option<&ProcessResultDisplayHandle>,
    ) -> Option<CryoPositionParam> {
        let dialog = self.dialog()?;
        if !self
            .manager
            .get_com_script_manager()
            .load_cryo_position(self.axis_id, false)
        {
            if !self.make_cryo_position_comfile(self.axis_id) && for_run {
                self.send_msg(Some(ProcessResult::FAILED_TO_START), process_result_display);
            }
            self.manager
                .get_com_script_manager()
                .load_cryo_position(self.axis_id, true);
        }
        let mut param = self
            .manager
            .get_com_script_manager()
            .get_cryo_position_param(self.axis_id, self.axis_type);
        if !dialog.get_parameters_cryo_position_param_boolean(&mut param, for_run) {
            return None;
        }
        self.manager
            .get_com_script_manager()
            .save_cryo_position(&param, self.axis_id);
        Some(param)
    }

    /// Java package-private `rollAlignComAngles()` (TomogramPositioningExpert.java:780).
    pub fn roll_align_com_angles(&self) {
        if let Some(dialog) = self.dialog() {
            dialog.roll_align_com_angles();
        }
    }

    /// Java package-private `rollTiltComAngles()` (TomogramPositioningExpert.java:786).
    pub fn roll_tilt_com_angles(&self) {
        if let Some(dialog) = self.dialog() {
            dialog.roll_tilt_com_angles();
        }
    }

    /// Java private `getNewstParam(NewstParam)` (TomogramPositioningExpert.java:796):
    /// get the newst.com parameters from the dialog.
    fn get_newst_param(&self, newst_param: &mut NewstParam) {
        let dialog = self.dialog();
        let mut binning = 1;
        if let Some(dialog) = &dialog {
            binning = dialog.get_binning();
        }
        // Only whole tomogram can change binning
        // Only explcitly write out the binning if its value is something other than
        // the default of 1 to keep from cluttering up the com script
        if binning == 1 {
            newst_param.set_bin_by_factor(Some(Number::Integer(i32::MIN)));
        } else {
            newst_param.set_bin_by_factor(Some(Number::Integer(binning)));
        }
        if let Some(dialog) = &dialog {
            dialog.update_meta_data(self.meta_data);
        }
        // Make sure the size output is removed, it was only there as a
        // copytomocoms template
        newst_param.set_command_mode(Some(newst_param::Mode::WholeTomogramSample));
        // Upstream bug fixed in translation: TomogramPositioningExpert.java:818
        // calls `dialog.getBinning()` without the null check the method makes
        // just above, so a null dialog throws a NullPointerException here.
        // The binning already read (1 without a dialog) is used instead; with
        // a dialog it is the same value.
        match newst_param.set_size_to_output_in_xand_y(
            "",
            binning,
            self.meta_data.get_image_rotation(self.axis_id).get_double(),
            None,
        ) {
            Ok(_) => {}
            // Java InvalidParameterException / IOException.
            Err(SetSizeToOutputInXandYError::HeaderRead(message)) => {
                eprintln!("{message}");
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!("Unable to update newst com: {message}"),
                    "Etomo Error",
                    Some(self.axis_id),
                );
            }
            // Java FortranInputSyntaxException, caught by the outer try.
            Err(SetSizeToOutputInXandYError::FortranInputSyntax(e)) => {
                e.print_stack_trace();
            }
        }
    }

    /// Java private `getParameters(BlendmontParam) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`
    /// (TomogramPositioningExpert.java:841).
    fn get_parameters_blendmont_param(
        &self,
        blendmont_param: &mut BlendmontParam,
    ) -> Result<(), ConvertError> {
        let dialog = self.dialog();
        let mut binning = 1;
        if let Some(dialog) = &dialog {
            binning = dialog.get_binning();
        }
        blendmont_param.set_bin_by_factor_int(binning);
        if let Some(dialog) = &dialog {
            dialog.update_meta_data(self.meta_data);
        }
        blendmont_param.set_mode(blendmont_param::Mode::WholeTomogramSample);
        blendmont_param.set_blendmont_state(&self.state.get_invalid_edge_functions(self.axis_id));
        blendmont_param.reset_starting_and_ending_xand_y();
        blendmont_param.convert_to_starting_and_ending_xand_y(
            "",
            self.meta_data.get_image_rotation(self.axis_id).get_double(),
            None,
        )?;
        Ok(())
    }
}

impl ReconUIExpertVirtual for TomogramPositioningExpert {
    /// Java package-private override `doneDialog()` (TomogramPositioningExpert.java:238).
    fn done_dialog_void(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        let exit_state = dialog.get_exit_state();
        if exit_state == DialogExitState::Execute {
            let sample_fiducialess = self.state.get_sample_fiducialess(self.axis_id);
            if sample_fiducialess
                .as_ref()
                .is_none_or(|sample_fiducialess| !sample_fiducialess.is())
                && dialog.is_tomopitch_button()
                && dialog.is_align_button_enabled()
                && !dialog.is_align_button()
            {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    "ERROR:  Final alignment is not done or is out of date.  Run \
                     final alignment in positioning before continuing.",
                    "User Error",
                    Some(self.axis_id),
                );
            }
            self.manager.close_imod(
                Some(imod_manager::SAMPLE_KEY),
                Some(self.axis_id),
                Some("sample reconstruction"),
                false,
            );
        }
        if exit_state != DialogExitState::Cancel {
            self.save_dialog_void();
        }
        self.leave_dialog(exit_state);
        // Hold onto the finished dialog in case anything is running that needs it or
        // there are next processes that need it.
    }

    /// Java package-private override `saveDialog()` (TomogramPositioningExpert.java:279).
    fn save_dialog_void(&self) {
        let Some(dialog) = self.dialog() else {
            return;
        };
        dialog.get_parameters_meta_data(self.meta_data);
        self.advanced.set(dialog.is_advanced());
        // Get all of the parameters from the panel
        let sample_fiducialess = self.state.get_sample_fiducialess(self.axis_id);
        if sample_fiducialess
            .as_ref()
            .is_none_or(|sample_fiducialess| !sample_fiducialess.is())
        {
            self.update_align_com(false);
        }
        self.update_tomo_pos_tilt_com(false, false);
        self.update_tomopitch_com(false);
        self.update_cryo_position_com(false, None);
        UIExpertUtilities::INSTANCE
            .update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
                self.manager,
                &*dialog,
                self.axis_id,
                false,
            );
        if self.meta_data.get_view_type() != ViewType::Montage {
            self.update_newst_com();
        }
        self.manager.save_storables(Some(self.axis_id));
    }

    /// Java package-private override `getDialog()` (TomogramPositioningExpert.java:859).
    fn get_dialog(&self) -> Option<Rc<dyn ProcessDialogVirtual>> {
        self.dialog()
            .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>)
    }
}

impl UIExpert for TomogramPositioningExpert {
    /// Java override `openDialog()` (TomogramPositioningExpert.java:134): open
    /// the tomogram positioning dialog.
    fn open_dialog(&self) {
        if !self.can_show_dialog() {
            return;
        }
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
        // Create a new dialog panel and map it the generic reference
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("TomogramPositioningDialog"),
            Some(utilities::STARTED_STATUS),
        );
        let dialog = TomogramPositioningDialog::get_instance(
            self.manager,
            self.this.clone(),
            axis_id,
            self.axis_type,
            meta_data.get_view_type(),
        );
        *self.dialog.borrow_mut() = Some(dialog.clone());
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("TomogramPositioningDialog"),
            Some(utilities::FINISHED_STATUS),
        );
        // Read in the meta data parameters. WARNING this needs to be done
        // before reading the tilt paramers below so that the GUI knows how to
        // correctly scale the dimensions
        dialog.set_parameters_const_meta_data(meta_data);
        if meta_data.get_view_type() != ViewType::Montage {
            self.manager.get_com_script_manager().load_newst(axis_id);
        } else {
            self.manager.get_com_script_manager().load_blend(axis_id);
        }

        // Get the align{|a|b}.com parameters
        self.manager.get_com_script_manager().load_align(axis_id);
        let mut tiltalign_param = self
            .manager
            .get_com_script_manager()
            .get_tiltalign_param(axis_id);
        if meta_data.get_view_type() != ViewType::Montage {
            // upgrade and save param to comscript
            UIExpertUtilities::INSTANCE.upgrade_old_align_com(
                self.manager,
                axis_id,
                &mut tiltalign_param,
            );
        }
        dialog.set_align_param(&tiltalign_param);

        // Get the tilt{|a|b}.com parameters
        self.manager.get_com_script_manager().load_tilt(axis_id);
        let mut tilt_param = self
            .manager
            .get_com_script_manager()
            .get_tilt_param(axis_id);
        tilt_param.set_fiducialess(meta_data.is_fiducialess(axis_id));
        dialog.set_tilt_param(&tilt_param, !meta_data.is_pos_exists(axis_id));
        // If this is a montage, then binning can only be 1, so no need to upgrade
        if meta_data.get_view_type() != ViewType::Montage {
            // upgrade and save param to comscript
            UIExpertUtilities::INSTANCE.upgrade_old_tilt_com(
                self.manager,
                axis_id,
                &mut tilt_param,
            );
        }

        // Get the tomopitch{|a|b}.com parameters
        self.manager
            .get_com_script_manager()
            .load_tomopitch(axis_id);
        let tomopitch_param = self
            .manager
            .get_com_script_manager()
            .get_tomopitch_param(axis_id);
        dialog.set_tomopitch_param(&tomopitch_param);

        // Set the whole tomogram sampling state, fidcialess state, and tilt axis
        // angle
        dialog.set_whole_tomogram(meta_data.is_whole_tomogram_sample(axis_id));

        dialog.set_fiducialess(meta_data);
        dialog.set_image_rotation(Some(&meta_data.get_image_rotation(axis_id).to_string()));
        dialog.set_button_state(self.manager.get_screen_state(axis_id));
        self.fiducialess_action();
        // cryoPosition
        if self
            .manager
            .get_com_script_manager()
            .load_cryo_position(axis_id, false)
        {
            // Java tests the returned param for null; the Rust
            // `getCryoPositionParam` always returns one.
            let param = self
                .manager
                .get_com_script_manager()
                .get_cryo_position_param(axis_id, self.axis_type);
            dialog.set_parameters_cryo_position_param(&param);
        }
        self.open_dialog_process_dialog_string(dialog.process_dialog(), action_message.as_deref());
        meta_data.set_pos_exists(axis_id, true);
    }

    /// Java override `startNextProcess(ProcessSeries.Process,
    /// ProcessResultDisplay, ProcessSeries, DialogType, ProcessDisplay)`
    /// (TomogramPositioningExpert.java:80): start the next process specified
    /// by the nextProcess string.  Returns true if the process is recognized.
    fn start_next_process(
        &self,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        _dialog_type: Option<DialogType>,
        _display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        // whole tomogram
        if process.equals_string(Some(&ProcessName::TILT.to_string())) {
            self.sample_tilt(process_result_display, process_series);
            return true;
        }
        if process.equals_string(Some(&ProcessName::FIND_SECTION.to_string())) {
            self.find_section(process_result_display, process_series);
            return true;
        }
        // Fixed in translation (BUGS.md): cryoPosition queues Task.POST_CRYO_POSITION
        // and nothing in the Java handles it, so the series never ends and the axis
        // stays busy.  Its post-processing is done in ProcessManager.postProcess, so the
        // task only continues (and so ends) the series.
        if POST_CRYO_POSITION.with(|task| process.equals_task(task.as_ref())) {
            if let Some(process_series) = &process_series {
                ProcessSeries::start_next_process(process_series, self.axis_id);
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

/// Java `public static final class Task implements TaskInterface`
/// (TomogramPositioningExpert.java:863).
///
/// The Java's one instance, `POST_CRYO_POSITION`, is a private static
/// singleton compared by identity.  A process series holds its tasks as
/// `Rc<dyn TaskInterface>` on the event dispatch thread, so the singleton is a
/// thread-local `Rc` created once per thread (only the EDT uses it); identity
/// is `Rc::ptr_eq`.
pub struct Task {
    /// Java private final `descr`.
    descr: String,
}

impl Task {
    /// Java private `Task(String)`.
    fn new(descr: &str) -> Task {
        Task {
            descr: descr.to_string(),
        }
    }
}

impl TaskInterface for Task {
    /// Java override `getDescr()`.
    fn get_descr(&self) -> Option<String> {
        Some(self.descr.clone())
    }

    /// Java override `okToDrop()`.
    fn ok_to_drop(&self) -> bool {
        false
    }
}

thread_local! {
    /// Java private static final `Task.POST_CRYO_POSITION`.
    pub static POST_CRYO_POSITION: Rc<dyn TaskInterface> = Rc::new(Task::new("post cryo position"));
}
