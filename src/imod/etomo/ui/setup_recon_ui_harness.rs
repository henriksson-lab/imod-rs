//! `IMOD/Etomo/src/etomo/ui/SetupReconUIHarness.java`.
//!
//! Class to handle the tomogram reconstruction setup, with or without a
//! front-end.  Handles headless, directive-based automation as well as an
//! interface based setup - with or without parameter-based automation.
//!
//! The harness lives on the event dispatch thread with the `SetupDialogExpert`
//! it creates (`util/event_queue.rs`).  It hands `this` to the expert, so it is
//! built with `Rc::new_cyclic` and keeps a `this: Weak<Self>`; its methods take
//! `&self` and no `RefCell` borrow is held across a call out to the manager,
//! the expert or the dialog.
//!
//! Java's `SetupReconInterface` is implemented by both `SetupDialog` and
//! `DirectiveFileCollection`; `getSetupReconInterface` hands back whichever is
//! current as an `Rc<dyn SetupReconInterface>`.

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::arguments::DIRECTIVE_TAG;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::exclude_views_param::ExcludeViewsParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::local_arguments::LocalArguments;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::logic::seeding_method::{self, SeedingMethod};
use crate::imod::etomo::logic::tracking_method::TrackingMethod;
use crate::imod::etomo::logic::validation_set::ValidationSet;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file::DirectiveFile;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::r#type::erase_gold::EraseGold;
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::setup_recon_interface::{
    DirectiveFileCollectionHandle, SetupReconInterface,
};
use crate::imod::etomo::ui::swing::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::ui::swing::process_dialog::DialogExitState;
use crate::imod::etomo::ui::swing::setup_dialog_expert::SetupDialogExpert;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::mrc_header::{self, MRCHeader};
use crate::imod::etomo::util::utilities;
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;

/// Java private static final `NO_GUI_ERROR_TITLE`.
const NO_GUI_ERROR_TITLE: &str = "No GUI";

/// Java private static final `NO_GUI_ERROR_MESSAGE`: "GUI not found.  To run
/// automation without the GUI, use the " + `Arguments.DIRECTIVE_TAG` + "
/// option.".
fn no_gui_error_message() -> String {
    format!("GUI not found.  To run automation without the GUI, use the {DIRECTIVE_TAG} option.")
}

/// Java `public final class SetupReconUIHarness`.
pub struct SetupReconUIHarness {
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `binningValidationSet`.
    binning_validation_set: Arc<ValidationSet>,
    /// Java private `expert`.
    expert: RefCell<Option<Rc<SetupDialogExpert>>>,
    /// Java private `directiveFileCollection`.
    directive_file_collection: RefCell<Option<DirectiveFileCollectionHandle>>,
    /// Java private `setFEIPixelSize`.
    set_fei_pixel_size: Cell<bool>,
    /// Java private `header`.
    header: RefCell<Option<std::sync::Arc<crate::imod::etomo::util::mrc_header::SharedMRCHeader>>>,
    /// Rust-only: the `this` the Java hands to `SetupDialogExpert.getInstance`.
    this: Weak<SetupReconUIHarness>,
}

impl SetupReconUIHarness {
    /// Java `SetupReconUIHarness(ApplicationManager, AxisID)`.
    pub fn new(manager: &'static ApplicationManager, axis_id: AxisID) -> Rc<SetupReconUIHarness> {
        let mut binning_validation_set = ValidationSet::new_numeric(Type::Double);
        binning_validation_set.set_minimum(0.5);
        binning_validation_set.set_maximum(50.0);
        Rc::new_cyclic(|this| SetupReconUIHarness {
            manager,
            axis_id,
            binning_validation_set: Arc::new(binning_validation_set),
            expert: RefCell::new(None),
            directive_file_collection: RefCell::new(None),
            set_fei_pixel_size: Cell::new(false),
            header: RefCell::new(None),
            this: this.clone(),
        })
    }

    fn expert(&self) -> Option<Rc<SetupDialogExpert>> {
        self.expert.borrow().clone()
    }

    fn directive_file_collection(&self) -> Option<DirectiveFileCollectionHandle> {
        self.directive_file_collection.borrow().clone()
    }

    /// Java `doAutomation(LocalArguments)`: runs the doAutomation function in
    /// the dialog expert, or handles an automation directive file.
    pub fn do_automation(&self, local_arguments: Option<&LocalArguments>) {
        let is_directive = etomo_director::ARGUMENTS.lock().unwrap().is_directive();
        if !is_directive {
            if let Some(expert) = self.expert() {
                expert.do_automation(local_arguments);
            } else {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!("{} (1)", no_gui_error_message()),
                    NO_GUI_ERROR_TITLE,
                    None,
                );
            }
            return;
        }
        // Headless automation using directives
        let is_from_brt = etomo_director::ARGUMENTS.lock().unwrap().is_from_brt();
        // Java shares the one `binningValidationSet` object; the collection
        // owns a copy of the (never modified) set.
        let directive_file_collection: DirectiveFileCollectionHandle =
            Rc::new(RefCell::new(if is_from_brt {
                DirectiveFileCollection::get_batch_instance_with_validation_set(
                    self.manager,
                    Some(self.axis_id),
                    Some((*self.binning_validation_set).clone()),
                )
            } else {
                DirectiveFileCollection::new_with_validation_set(
                    self.manager,
                    Some(self.axis_id),
                    Some((*self.binning_validation_set).clone()),
                )
            }));
        *self.directive_file_collection.borrow_mut() = Some(directive_file_collection.clone());
        let batch_directive_file =
            DirectiveFile::get_arg_instance(self.manager, Some(self.axis_id)).map(Arc::new);
        directive_file_collection
            .borrow_mut()
            .setup(batch_directive_file);
        self.do_directive_automation();
    }

    /// Java private `doDirectiveAutomation()`.
    fn do_directive_automation(&self) {
        let Some(directive_file_collection) = self.directive_file_collection() else {
            self.manager.end_automation(false);
            // Upstream bug fixed in translation (SetupReconUIHarness.java:136-142):
            // `endAutomation(false)` exits only when etomo was started with
            // `-exit`; otherwise the source carries on and dereferences the null
            // collection (NullPointerException).  The automation stops here.
            return;
        };
        // Java `manager.getConstMetaData()` is the manager's `metaData`.
        etomo_director::INSTANCE.with_user_configuration(|user_configuration| {
            self.initialize_fields(self.manager.get_meta_data(), user_configuration)
        });
        let mut axis_type = AxisType::SingleAxis;
        let dual = directive_file_collection
            .borrow()
            .is_value(Some(DirectiveDef::DUAL));
        if dual {
            axis_type = AxisType::DualAxis;
        }
        let property_user_dir = self.get_property_user_dir().unwrap_or_default();
        let name = directive_file_collection
            .borrow()
            .get_value(Some(DirectiveDef::NAME));
        // Odd but kept: when the name is invalid and `endAutomation` does not
        // exit, the source goes on to finish the setup anyway.
        if !dataset_tool::validate_dataset_name(
            self.manager,
            None,
            self.axis_id,
            Path::new(&property_user_dir),
            name.as_deref(),
            DataFileType::Recon,
            axis_type,
            true,
        ) {
            self.manager.end_automation(false);
        }
        let scan_header = {
            let collection = directive_file_collection.borrow();
            collection.is_value(Some(DirectiveDef::SCAN_HEADER))
                && (!collection.contains(Some(DirectiveDef::PIXEL))
                    || !collection.contains(Some(DirectiveDef::ROTATION)))
        };
        if scan_header {
            self.scan_header_action(&directive_file_collection, false);
        }
        let dataset_directory = directive_file_collection
            .borrow()
            .get_value(Some(DirectiveDef::DATASET_DIRECTORY));
        if dual {
            self.manager.done_setup_dialog(
                directive_file_collection.borrow().is_value_axis(
                    Some(DirectiveDef::REMOVE_EXCLUDED_VIEWS),
                    Some(AxisID::First),
                ),
                directive_file_collection.borrow().is_value_axis(
                    Some(DirectiveDef::REMOVE_EXCLUDED_VIEWS),
                    Some(AxisID::Second),
                ),
                dataset_directory.as_deref(),
                dual,
                None,
            );
        } else {
            self.manager.done_setup_dialog(
                directive_file_collection.borrow().is_value_axis(
                    Some(DirectiveDef::REMOVE_EXCLUDED_VIEWS),
                    Some(AxisID::Only),
                ),
                false,
                dataset_directory.as_deref(),
                dual,
                None,
            );
        }
    }

    /// Java `getParameters(ExcludeViewsParam, AxisID, boolean, boolean)`.
    pub fn get_parameters(
        &self,
        param: &mut ExcludeViewsParam,
        axis_id: AxisID,
        dual_axis: bool,
        do_validation: bool,
    ) -> bool {
        let Some(setup_interface) = self.get_setup_recon_interface() else {
            return false;
        };
        setup_interface.get_parameters(param, axis_id, dual_axis, do_validation)
    }

    /// Java `getParametersForRenameInputImageFiles(BatchruntomoParam)`.
    pub fn get_parameters_for_rename_input_image_files(
        &self,
        param: &mut BatchruntomoParam,
    ) -> bool {
        let is_from_brt = etomo_director::ARGUMENTS.lock().unwrap().is_from_brt();
        if is_from_brt {
            // The batch run of batchruntomo will take care of this.
            return false;
        }
        let Some(meta_data) = self.manager.get_base_meta_data() else {
            return false;
        };
        // RootName
        param.set_root_name(meta_data.get_dataset_name().as_deref());
        // NamingStyle
        param.set_naming_style(meta_data.base().get_image_filename_style());
        let mut raw_image_stack: Option<String> = None;
        // StackExtension
        let mut extension: Option<&'static Extension> = None;
        let directive_file_collection = self.directive_file_collection();
        if let Some(directive_file_collection) = &directive_file_collection {
            // For running directive-based automation from the command line
            // rather then from batchruntomo - not the usual method.
            extension = directive_file_collection
                .borrow()
                .get_value(Some(DirectiveDef::STACK_EXT))
                .as_deref()
                .and_then(Extension::get_instance);
        } else if let Some(expert) = self.expert() {
            raw_image_stack = expert.get_raw_image_stack();
            extension = raw_image_stack.as_deref().and_then(Extension::get_instance);
        }
        if extension.is_none_or(|extension| !extension.is_input_image_file()) {
            let message = match extension {
                None => "No raw image extension has been chosen.".to_owned(),
                Some(extension) => {
                    format!("{extension}is not a valid raw image stack extension.")
                }
            } + "  Unable to rename raw image files.  ";
            let title = "Unable to Setup Reconstruction";
            if directive_file_collection.is_some() {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!(
                        "{message}A valid {} directive batch directive is required when \
                         running etomo with directive automation.",
                        DirectiveDef::STACK_EXT
                            .get_command()
                            .as_deref()
                            .unwrap_or("null")
                    ),
                    title,
                    None,
                );
                self.manager.end_automation(false);
            } else {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!(
                        "{message}Valid raw image stack extensions are {}.",
                        Extension::get_input_image_file_descr()
                    ),
                    title,
                    None,
                );
            }
            return false;
        }
        param.set_stack_extension(extension);
        // AxisOfExtension
        param.set_axis_of_extension(AxisID::get_instance_from_file_name(
            self.get_axis_type(),
            raw_image_stack.as_deref(),
        ));
        true
    }

    /// Java `getSetupDialogExpert(AxisProgressPanel)`: called by the manager
    /// when not headless.
    pub fn get_setup_dialog_expert(
        &self,
        progress_panel: Option<Rc<AxisProgressPanel>>,
    ) -> Option<Rc<SetupDialogExpert>> {
        let is_headless = etomo_director::ARGUMENTS.lock().unwrap().is_headless();
        if is_headless {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                &format!("{} (2)", no_gui_error_message()),
                NO_GUI_ERROR_TITLE,
                None,
            );
            return None;
        }
        if self.expert().is_none() {
            // Upstream bug fixed in translation (SetupReconUIHarness.java:216 ->
            // SetupDialog constructor): a null progressPanel is dereferenced by
            // the SetupDialog constructor (NullPointerException); no expert is
            // created then and null is returned.
            let Some(progress_panel) = progress_panel else {
                return None;
            };
            let distortion_dir = dataset_files::get_distortion_dir(
                Some(self.manager),
                self.manager.get_property_user_dir().as_deref(),
                Some(self.axis_id),
            );
            let expert = SetupDialogExpert::get_instance(
                self.manager,
                self.this.clone(),
                distortion_dir.is_some_and(|distortion_dir| distortion_dir.exists()),
                self.binning_validation_set.clone(),
                progress_panel,
            );
            *self.expert.borrow_mut() = Some(expert);
        }
        self.expert()
    }

    /// Java `freeDialog()`.
    pub fn free_dialog(&self) {
        *self.expert.borrow_mut() = None;
    }

    /// Java `msgExcludeViewsSucceeded(AxisID, boolean, boolean)`.
    pub fn msg_exclude_views_succeeded(
        &self,
        axis_id: AxisID,
        process_starting: bool,
        process_done: bool,
    ) {
        if let Some(setup_interface) = self.get_setup_recon_interface() {
            setup_interface.msg_exclude_views_succeeded(axis_id, process_starting, process_done);
        }
    }

    /// Java private `getSetupReconInterface()`.
    fn get_setup_recon_interface(&self) -> Option<Rc<dyn SetupReconInterface>> {
        if let Some(directive_file_collection) = self.directive_file_collection() {
            return Some(Rc::new(directive_file_collection) as Rc<dyn SetupReconInterface>);
        }
        if let Some(expert) = self.expert() {
            return Some(expert.get_setup_recon_interface());
        }
        ui_harness::open_message_dialog_from_process(
            Some(self.manager),
            &format!("{} (3)", no_gui_error_message()),
            NO_GUI_ERROR_TITLE,
            None,
        );
        None
    }

    /// Java `getExitState()`.
    pub fn get_exit_state(&self) -> Option<DialogExitState> {
        if self.directive_file_collection().is_some() {
            return Some(DialogExitState::Execute);
        }
        if let Some(expert) = self.expert() {
            return Some(expert.get_exit_state());
        }
        ui_harness::open_message_dialog_from_process(
            Some(self.manager),
            &format!("{} (4)", no_gui_error_message()),
            NO_GUI_ERROR_TITLE,
            None,
        );
        None
    }

    /// Java `getViewsToSkip(AxisID, boolean)`.
    pub fn get_views_to_skip(&self, axis_id: AxisID, do_validation: bool) -> Option<String> {
        if let Some(directive_file_collection) = self.directive_file_collection() {
            return directive_file_collection
                .borrow()
                .get_skip(Some(axis_id), do_validation);
        }
        if let Some(expert) = self.expert() {
            return expert.get_views_to_skip(axis_id, do_validation);
        }
        None
    }

    /// Java `checkForSharedDirectory(Extension)`.  This is functionality is
    /// mostly duplicated by the validate dataset functions in the logic
    /// package.  Not worth duplicating for headless automation.
    pub fn check_for_shared_directory(&self, raw_image_stack_extension: &Extension) -> bool {
        if let Some(expert) = self.expert() {
            return expert.check_for_shared_directory(raw_image_stack_extension);
        }
        false
    }

    /// Java `getWorkingDirectory()`.
    pub fn get_working_directory(&self) -> Option<PathBuf> {
        if self.directive_file_collection().is_some() {
            return Some(PathBuf::from(
                self.get_property_user_dir().unwrap_or_default(),
            ));
        }
        if let Some(expert) = self.expert() {
            return expert.get_working_directory();
        }
        ui_harness::open_message_dialog_from_process(
            Some(self.manager),
            &format!("{} (5)", no_gui_error_message()),
            NO_GUI_ERROR_TITLE,
            None,
        );
        None
    }

    /// Java `getRawImageStack()`.
    pub fn get_raw_image_stack(&self) -> Option<String> {
        let Some(expert) = self.expert() else {
            if let Some(directive_file_collection) = self.directive_file_collection() {
                return directive_file_collection.borrow().get_raw_image_stack();
            }
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                &format!("{} (6)", no_gui_error_message()),
                NO_GUI_ERROR_TITLE,
                None,
            );
            return None;
        };
        expert.get_raw_image_stack()
    }

    /// Java `getRawStackExtension(boolean)`.
    pub fn get_raw_stack_extension(&self, _do_validation: bool) -> Option<&'static Extension> {
        let mut extension: Option<&'static Extension> = None;
        match self.expert() {
            None => {
                if let Some(directive_file_collection) = self.directive_file_collection() {
                    extension = directive_file_collection
                        .borrow()
                        .get_stack_ext()
                        .as_deref()
                        .and_then(Extension::get_instance);
                }
                // Odd but kept: the source pops the "No GUI" message here even
                // when the directive collection supplied the extension.
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!("{} (7)", no_gui_error_message()),
                    NO_GUI_ERROR_TITLE,
                    None,
                );
            }
            Some(expert) => {
                extension = expert
                    .get_raw_image_stack()
                    .as_deref()
                    .and_then(Extension::get_instance);
            }
        }
        let Some(extension) = extension else {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Missing raw image stack extension.",
                "Missing Extension",
                None,
            );
            return None;
        };
        if !extension.is_input_image_file() {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                &format!("{extension} is not a valid raw image stack extension."),
                "Invalid Extension",
                None,
            );
            return None;
        }
        Some(extension)
    }

    /// Java `getCurrentStackExt()`.
    pub fn get_current_stack_ext(&self) -> Option<String> {
        if let Some(directive_file_collection) = self.directive_file_collection() {
            return directive_file_collection.borrow().get_current_stack_ext();
        }
        None
    }

    /// Java `getMetaData()`.
    pub fn get_meta_data(&self) -> Option<MetaData> {
        let setup_interface = self.get_setup_recon_interface()?;
        let meta_data = MetaData::new(Some(self.manager), self.manager.get_log_properties(), true);
        meta_data.set_axis_type(self.get_axis_type().unwrap_or(AxisType::NotSet));
        // The dataset name needs to be set after the axis type so the metadata
        // object modifies the file ending correctly (if a file name is used).
        meta_data.set_dataset_name(
            setup_interface
                .get_raw_image_stack()
                .as_deref()
                .unwrap_or_default(),
        );
        Some(meta_data)
    }

    /// Java `getAxisType()`.
    pub fn get_axis_type(&self) -> Option<AxisType> {
        let setup_interface = self.get_setup_recon_interface()?;
        if setup_interface.is_single_axis_selected() {
            Some(AxisType::SingleAxis)
        } else {
            Some(AxisType::DualAxis)
        }
    }

    /// Java private `getViewType(SetupReconInterface)`.
    fn get_view_type(&self, setup_interface: &dyn SetupReconInterface) -> ViewType {
        if setup_interface.is_single_view_selected() {
            ViewType::SingleView
        } else {
            ViewType::Montage
        }
    }

    /// Java `getPropertyUserDir()`: get the directory in which the user wants
    /// to create the dataset.
    pub fn get_property_user_dir(&self) -> Option<String> {
        let directive_file_collection = self.directive_file_collection();
        if let Some(directive_file_collection) = directive_file_collection
            .as_ref()
            .filter(|dfc| dfc.borrow().contains(Some(DirectiveDef::DATASET_DIRECTORY)))
        {
            return directive_file_collection
                .borrow()
                .get_value(Some(DirectiveDef::DATASET_DIRECTORY));
        } else if let Some(expert) = self.expert() {
            if let Some(dir) = expert.get_dir() {
                return Some(utilities::java_io_file_get_absolute_path(
                    &dir.to_string_lossy(),
                ));
            }
        }
        // Java `if (manager == null) return null;` - the manager is never
        // null here.
        self.manager.get_property_user_dir()
    }

    /// Java `isFloatModeInput()`.
    pub fn is_float_mode_input(&self) -> bool {
        self.header
            .borrow()
            .as_ref()
            .is_some_and(|header| header.borrow().get_mode() == mrc_header::FLOATING_POINT_MODE)
    }

    /// Java `scanHeaderAction(SetupReconInterface, boolean)`.
    pub fn scan_header_action(
        &self,
        setup_interface: &dyn SetupReconInterface,
        load_header_only: bool,
    ) {
        let raw_image_stack = setup_interface.get_raw_image_stack();
        let header = self.read_mrc_header(raw_image_stack.as_deref(), load_header_only);
        *self.header.borrow_mut() = header.clone();
        let Some(header) = header.filter(|_| !load_header_only) else {
            return; // false;
        };
        let header = header.borrow();
        // Set the image rotation if available
        let image_rotation = header.get_image_rotation();
        if !image_rotation.is_null() {
            setup_interface.set_image_rotation(Some(&image_rotation.to_string()));
        }
        // set the pixel size if available
        let mut x_pixel_size = header.get_x_pixel_size().get_double();
        let y_pixel_size = header.get_y_pixel_size().get_double();
        if x_pixel_size.is_nan() || y_pixel_size.is_nan() {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Pixel size is not defined in the image file header",
                "Pixel size is missing",
                Some(AxisID::Only),
            );
            return; // false;
        }
        if x_pixel_size != y_pixel_size {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "X & Y pixels sizes are different, don't know what to do",
                "Pixel sizes are different",
                Some(AxisID::Only),
            );
            return; // false;
        }
        if x_pixel_size == 1.0 {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                "Pixel size is not defined in the image file header",
                "Pixel size is missing",
                Some(AxisID::Only),
            );
            return; // false;
        }
        x_pixel_size /= 10.0;
        setup_interface.set_pixel_size(
            utilities::java_lang_math_round(x_pixel_size * 1000000.0) as f64 / 1000000.0,
        );
        setup_interface.set_binning(Some(&header.get_binning()));
        let mut twodir = header.get_twodir();
        if !twodir.is_null() {
            setup_interface.set_twodir(AxisID::First, twodir.get_double());
        }
        let mut dose_sym = header.get_dose_sym();
        if !dose_sym.is_null() {
            setup_interface.set_dose_sym(AxisID::First, dose_sym.get_double());
        }
        drop(header);
        // B stack
        let b_stack = dataset_tool::get_b_axis_input_image_file(
            raw_image_stack.as_deref(),
            setup_interface.is_dual_axis_selected(),
        );
        if let Some(b_stack) = b_stack {
            let header_b = self.read_mrc_header(Some(&b_stack), false);
            if let Some(header_b) = header_b {
                let header_b = header_b.borrow();
                twodir = header_b.get_twodir();
                if !twodir.is_null() {
                    setup_interface.set_twodir(AxisID::Second, twodir.get_double());
                }
                dose_sym = header_b.get_dose_sym();
                if !dose_sym.is_null() {
                    setup_interface.set_dose_sym(AxisID::Second, dose_sym.get_double());
                }
            }
        }
        // true;
    }

    /// Java private `readMRCHeader(String, boolean)`: construct and read an
    /// MRCHeader object.  MRCHeader saves instances, and an existing instance
    /// will be used if it already exists.
    fn read_mrc_header(
        &self,
        input_image_file_path: Option<&str>,
        no_popups: bool,
    ) -> Option<std::sync::Arc<crate::imod::etomo::util::mrc_header::SharedMRCHeader>> {
        // Run header on the dataset to the extract whatever information is
        // available
        if input_image_file_path.is_none_or(str::is_empty) {
            if !no_popups {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    "Raw image stack has not been entered",
                    "Missing Raw Image Stack",
                    Some(AxisID::Only),
                );
            }
            return None;
        }
        let header = MRCHeader::get_instance(input_image_file_path, Some(AxisID::Only))?;
        let result = header.borrow_mut().read_with_manager(self.manager);
        match result {
            Ok(false) => return None,
            Ok(true) => {}
            // `catch (final InvalidParameterException except)`.
            Err(crate::imod::etomo::util::mrc_header::ReadError::InvalidParameter(message)) => {
                if !no_popups {
                    ui_harness::open_message_dialog_from_process(
                        Some(self.manager),
                        &message,
                        "Invalid Parameter Exception",
                        Some(AxisID::Only),
                    );
                }
            }
            // `catch (final IOException except)`; the unchecked NumberFormatException
            // of a header whose size is not a number is reported the same way.
            Err(message) => {
                if !no_popups {
                    ui_harness::open_message_dialog_from_process(
                        Some(self.manager),
                        &message.to_string(),
                        "IO Exception",
                        Some(AxisID::Only),
                    );
                }
            }
        }
        Some(header)
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        let Some(setup_interface) = self.get_setup_recon_interface() else {
            return false;
        };
        let error_message_title = "Setup Dialog Error";
        let input_image_file = setup_interface.get_raw_image_stack();
        let mut err_msg = String::new();
        if !dataset_tool::is_valid_input_image_file(
            input_image_file.as_deref().unwrap_or_default(),
            setup_interface.is_dual_axis_selected(),
            Some(&mut err_msg),
        ) {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                &format!("Raw image stack is not valid.  {err_msg}"),
                error_message_title,
                Some(AxisID::Only),
            );
            return false;
        }
        // validate image distortion field file name
        // optional
        // file must exist
        let distortion_file_text = setup_interface.get_distortion_file();
        if let Some(distortion_file_text) = distortion_file_text.filter(|text| text != "") {
            if !Path::new(&distortion_file_text).exists() {
                let distortion_file_name = utilities::java_io_file_get_name(&distortion_file_text);
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!(
                        "The image distortion field file {distortion_file_name} does not exist."
                    ),
                    error_message_title,
                    Some(AxisID::Only),
                );
                return false;
            }
        }
        // validate mag gradient field file name
        // optional
        // file must exist
        let mag_gradient_file_text = setup_interface.get_mag_gradient_file();
        if let Some(mag_gradient_file_text) = mag_gradient_file_text.filter(|text| text != "") {
            if !Path::new(&mag_gradient_file_text).exists() {
                let mag_gradient_file_name =
                    utilities::java_io_file_get_name(&mag_gradient_file_text);
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!(
                        "The mag gradients correction file {mag_gradient_file_name} does not exist."
                    ),
                    error_message_title,
                    Some(AxisID::Only),
                );
                return false;
            }
        }
        if !dataset_tool::validate_view_type(
            if setup_interface.is_single_view_selected() {
                ViewType::SingleView
            } else {
                ViewType::Montage
            },
            self.get_property_user_dir().as_deref(),
            input_image_file.as_deref(),
            self.manager,
            None,
            AxisID::Only,
        ) {
            return false;
        }
        if !setup_interface.validate_tilt_angle(AxisID::First, error_message_title) {
            return false;
        }
        if !setup_interface.validate_tilt_angle(AxisID::Second, error_message_title) {
            return false;
        }
        true
    }

    /// Java `isDirectiveDrivenAutomation()`.
    pub fn is_directive_driven_automation(&self) -> bool {
        self.directive_file_collection.borrow().is_some()
    }

    /// Java `getDirectiveFileCollection()`.
    pub fn get_directive_file_collection(&self) -> Option<DirectiveFileCollectionHandle> {
        if let Some(directive_file_collection) = self.directive_file_collection() {
            return Some(directive_file_collection);
        }
        // Upstream bug fixed in translation (SetupReconUIHarness.java:599):
        // `expert.getDirectiveFileCollection()` throws a NullPointerException
        // when there is neither a collection nor an expert; null is returned.
        self.expert()
            .and_then(|expert| expert.get_directive_file_collection())
    }

    /// Java `getFields(boolean)`.
    pub fn get_fields(&self, do_validation: bool) -> Option<MetaData> {
        self.get_setup_recon_interface()?;
        let meta_data = self.get_meta_data()?;
        if self.get_fields_meta_data(&meta_data, do_validation) {
            return Some(meta_data);
        }
        None
    }

    /// Java `getFields(MetaData, boolean)`.
    pub fn get_fields_meta_data(&self, meta_data: &MetaData, do_validation: bool) -> bool {
        let Some(setup_interface) = self.get_setup_recon_interface() else {
            return false;
        };
        match self.get_fields_try(&*setup_interface, meta_data, do_validation) {
            Ok(value) => value,
            // `catch (final FieldValidationFailedException e)`.
            Err(FieldValidationFailedException { .. }) => false,
        }
    }

    /// The body of Java `getFields(MetaData, boolean)`'s outer `try` block;
    /// `Err` is the `FieldValidationFailedException` it catches.
    fn get_fields_try(
        &self,
        setup_interface: &dyn SetupReconInterface,
        meta_data: &MetaData,
        do_validation: bool,
    ) -> Result<bool, FieldValidationFailedException> {
        let axis_type = self.get_axis_type();
        meta_data.set_backup_directory(setup_interface.get_backup_directory().as_deref());
        meta_data.set_distortion_file(setup_interface.get_distortion_file().as_deref());
        meta_data.set_mag_gradient_file(setup_interface.get_mag_gradient_file().as_deref());
        let property_user_dir = self.get_property_user_dir();
        meta_data.set_default_parallel(
            setup_interface.is_parallel_process_selected(property_user_dir.as_deref()),
        );
        let property_user_dir = self.get_property_user_dir();
        meta_data.set_default_gpu_processing(
            setup_interface.is_gpu_processing_selected(property_user_dir.as_deref()),
        );
        meta_data.set_adjusted_focus_a(setup_interface.is_adjusted_focus_selected(AxisID::First));
        meta_data.set_adjusted_focus_b(setup_interface.is_adjusted_focus_selected(AxisID::Second));
        meta_data.set_view_type(self.get_view_type(setup_interface));
        let mut current_field;
        current_field = "Image Rotation";
        meta_data.set_image_rotation(
            setup_interface
                .get_image_rotation(AxisID::First, do_validation)?
                .as_deref(),
            AxisID::First,
        );
        if !meta_data.get_image_rotation(AxisID::First).is_valid() {
            ui_harness::open_message_dialog_from_process(
                Some(self.manager),
                &format!("{current_field} must be numeric."),
                "Setup Dialog Error",
                Some(AxisID::Only),
            );
            return Ok(false);
        }
        // The inner `try { ... } catch (NumberFormatException e)`: the
        // metadata setters here parse their strings without throwing (an
        // unparsable number becomes NaN); only `getTiltAngleFields` can reach
        // the catch arm, and its `Err` is handled at each call below.
        current_field = "Pixel Size";
        meta_data.set_pixel_size_string(setup_interface.get_pixel_size(do_validation)?.as_deref());
        current_field = "Half Float Mode Output";
        meta_data.set_half_float_mode_output(setup_interface.get_half_float_mode_output());
        current_field = "Fiducial Diameter";
        meta_data.set_fiducial_diameter_string(
            setup_interface
                .get_fiducial_diameter(do_validation)?
                .as_deref(),
        );
        if axis_type == Some(AxisType::DualAxis) {
            meta_data.set_image_rotation(
                setup_interface
                    .get_image_rotation(AxisID::Second, do_validation)?
                    .as_deref(),
                AxisID::Second,
            );
        }
        current_field = "Axis A starting and step angles";
        // Java passes the metadata's own `TiltAngleSpec`, which the call fills
        // in place; the Rust metadata hands out a copy, so it is stored back.
        let mut tilt_angle_spec_a = meta_data.get_tilt_angle_spec_a();
        let filled = setup_interface.get_tilt_angle_fields(
            AxisID::First,
            &mut tilt_angle_spec_a,
            do_validation,
        );
        meta_data.set_tilt_angle_spec_a(tilt_angle_spec_a);
        match filled {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            // `catch (NumberFormatException e)`.
            Err(_) => {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!("{current_field} must be numeric."),
                    "Setup Dialog Error",
                    Some(AxisID::Only),
                );
                return Ok(false);
            }
        }
        current_field = "Axis B starting and step angles";
        let mut tilt_angle_spec_b = meta_data.get_tilt_angle_spec_b();
        let filled = setup_interface.get_tilt_angle_fields(
            AxisID::Second,
            &mut tilt_angle_spec_b,
            do_validation,
        );
        meta_data.set_tilt_angle_spec_b(tilt_angle_spec_b);
        match filled {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            // `catch (NumberFormatException e)`.
            Err(_) => {
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!("{current_field} must be numeric."),
                    "Setup Dialog Error",
                    Some(AxisID::Only),
                );
                return Ok(false);
            }
        }
        let _ = current_field;
        meta_data.set_binning(setup_interface.get_binning(do_validation)?.as_deref());
        meta_data.set_exclude_projections(
            setup_interface
                .get_exclude_list(AxisID::First, do_validation)?
                .as_deref(),
            AxisID::First,
        );
        meta_data.set_exclude_projections(
            setup_interface
                .get_exclude_list(AxisID::Second, do_validation)?
                .as_deref(),
            AxisID::Second,
        );
        meta_data.set_is_twodir(AxisID::First, setup_interface.is_twodir(AxisID::First));
        meta_data.set_twodir(
            AxisID::First,
            setup_interface
                .get_twodir(AxisID::First, do_validation)?
                .as_deref(),
        );
        meta_data.set_is_twodir(AxisID::Second, setup_interface.is_twodir(AxisID::Second));
        meta_data.set_twodir(
            AxisID::Second,
            setup_interface
                .get_twodir(AxisID::Second, do_validation)?
                .as_deref(),
        );
        meta_data.set_is_dose_sym(AxisID::First, setup_interface.is_dose_sym(AxisID::First));
        meta_data.set_dose_sym(
            AxisID::First,
            setup_interface
                .get_dose_sym(AxisID::First, do_validation)?
                .as_deref(),
        );
        meta_data.set_is_dose_sym(AxisID::Second, setup_interface.is_dose_sym(AxisID::Second));
        meta_data.set_dose_sym(
            AxisID::Second,
            setup_interface
                .get_dose_sym(AxisID::Second, do_validation)?
                .as_deref(),
        );
        if axis_type == Some(AxisType::DualAxis) {
            let property_user_dir = self.get_property_user_dir();
            let b_stack = dataset_files::get_stack_with_dir(
                self.manager,
                property_user_dir.as_deref(),
                Some(meta_data as &dyn BaseMetaData),
                Some(AxisID::Second),
            );
            meta_data.set_b_stack_processed_boolean(b_stack.exists());
        }
        meta_data.set_set_fei_pixel_size(self.set_fei_pixel_size.get());
        // Upstream bug fixed in translation (SetupReconUIHarness.java:690-711):
        // the source dereferences `directiveFileCollection` before its own
        // `!= null` test, so a null collection throws a NullPointerException.
        // With no collection there are no directive settings to copy, and the
        // fields read so far stand.
        let Some(directive_file_collection) = setup_interface.get_directive_file_collection()
        else {
            return Ok(true);
        };
        let mut directive_file = directive_file_collection
            .borrow()
            .get_directive_file(DirectiveFileType::Scope);
        if let Some(directive_file) = &directive_file {
            meta_data.set_orig_scope_template(
                directive_file
                    .get_file()
                    .map(|file| file.to_string_lossy().into_owned())
                    .as_deref(),
            );
            self.save_directive_file(Some(directive_file), meta_data);
        }
        directive_file = directive_file_collection
            .borrow()
            .get_directive_file(DirectiveFileType::System);
        if let Some(directive_file) = &directive_file {
            meta_data.set_orig_system_template(
                directive_file
                    .get_file()
                    .map(|file| file.to_string_lossy().into_owned())
                    .as_deref(),
            );
            self.save_directive_file(Some(directive_file), meta_data);
        }
        directive_file = directive_file_collection
            .borrow()
            .get_directive_file(DirectiveFileType::User);
        if let Some(directive_file) = &directive_file {
            meta_data.set_orig_user_template(
                directive_file
                    .get_file()
                    .map(|file| file.to_string_lossy().into_owned())
                    .as_deref(),
            );
            self.save_directive_file(Some(directive_file), meta_data);
        }
        self.save_directive_file(
            directive_file_collection
                .borrow()
                .get_directive_file(DirectiveFileType::Batch)
                .as_ref(),
            meta_data,
        );
        let mut value: Option<String>;
        if axis_type == Some(AxisType::DualAxis) {
            // The brotation is ignored by the setup reconstruction dialog, so
            // it can be overidden from the templates
            value = directive_file_collection
                .borrow()
                .get_image_rotation(Some(AxisID::Second), false);
            if let Some(value) = &value {
                meta_data.set_image_rotation(Some(value), AxisID::Second);
            }
        }
        value = directive_file_collection
            .borrow()
            .get_value(Some(DirectiveDef::PATCH_SIZE));
        if let Some(value) = &value {
            meta_data.set_patch_type_or_xyz(Some(value));
        }
        value = directive_file_collection
            .borrow()
            .get_value(Some(DirectiveDef::EXTRA_TARGETS));
        if let Some(value) = &value {
            meta_data.set_extra_residual_targets(Some(value));
        }
        value = directive_file_collection
            .borrow()
            .get_value(Some(DirectiveDef::FINAL_PATCH_SIZE));
        if let Some(value) = &value {
            meta_data.set_auto_patch_final_size(Some(value));
        }
        value = directive_file_collection
            .borrow()
            .get_value(Some(DirectiveDef::WEDGE_REDUCTION));
        if let Some(value) = &value {
            meta_data.set_wedge_reduction_fraction(Some(value));
        }
        value = directive_file_collection
            .borrow()
            .get_value(Some(DirectiveDef::LOW_FROM_BOTH_RADIUS));
        if let Some(value) = &value {
            meta_data.set_low_from_both_radius(Some(value));
        }
        let mut axis_id = AxisID::First;
        value = directive_file_collection
            .borrow()
            .get_value_axis(Some(DirectiveDef::EXPAND_CIRCLE_ITERATIONS), Some(axis_id));
        if let Some(value) = &value {
            meta_data.set_use_final_stack_expand_circle_iterations(axis_id, true);
            meta_data.set_final_stack_expand_circle_iterations_string(axis_id, Some(value));
        }
        axis_id = AxisID::Second;
        value = directive_file_collection
            .borrow()
            .get_value_axis(Some(DirectiveDef::EXPAND_CIRCLE_ITERATIONS), Some(axis_id));
        if let Some(value) = &value {
            meta_data.set_use_final_stack_expand_circle_iterations(axis_id, true);
            meta_data.set_final_stack_expand_circle_iterations_string(axis_id, Some(value));
        }
        Ok(true)
    }

    /// Java private `saveDirectiveFile(DirectiveFile, MetaData)`.
    fn save_directive_file(
        &self,
        directive_file: Option<&Arc<DirectiveFile>>,
        meta_data: &MetaData,
    ) {
        let Some(directive_file) = directive_file else {
            return;
        };
        if directive_file.contains_axis(Some(DirectiveDef::USE_ALIGNED_STACK), Some(AxisID::First))
        {
            meta_data.set_track_raptor_use_raw_stack(
                directive_file
                    .is_value_axis(Some(DirectiveDef::USE_ALIGNED_STACK), Some(AxisID::First)),
            );
        }
        if directive_file.contains_axis(Some(DirectiveDef::NUMBER_OF_MARKERS), Some(AxisID::First))
        {
            meta_data.set_track_raptor_mark(
                directive_file
                    .get_value_axis(Some(DirectiveDef::NUMBER_OF_MARKERS), Some(AxisID::First))
                    .as_deref(),
            );
        }
        if directive_file.contains(Some(DirectiveDef::SLAB_THICKNESS_IN_NM)) {
            meta_data.set_ctf_3d_setup_slab_thickness_in_nm_set(true);
        }
        self.save_directive_file_axis(directive_file, meta_data, AxisID::First);
        self.save_directive_file_axis(directive_file, meta_data, AxisID::Second);
    }

    /// Java private `saveDirectiveFile(DirectiveFile, MetaData, AxisID)`.
    fn save_directive_file_axis(
        &self,
        directive_file: &DirectiveFile,
        meta_data: &MetaData,
        axis_id: AxisID,
    ) {
        if directive_file.contains_axis(Some(DirectiveDef::FIDUCIALLESS), Some(axis_id)) {
            let value =
                directive_file.is_value_axis(Some(DirectiveDef::FIDUCIALLESS), Some(axis_id));
            meta_data.set_fiducialess(axis_id, value);
            meta_data.set_fiducialess_alignment(axis_id, value);
        }
        // Defaults are set in metadata for seeding method - only set for
        // non-default situation.
        if directive_file.contains_axis(Some(DirectiveDef::SEEDING_METHOD), Some(axis_id)) {
            let seeding_method: Option<SeedingMethod> = SeedingMethod::get_instance(
                directive_file
                    .get_value_axis(Some(DirectiveDef::SEEDING_METHOD), Some(axis_id))
                    .as_deref(),
            );
            if seeding_method == Some(seeding_method::MANUAL) {
                meta_data.set_track_seed_model_manual(true, axis_id);
                meta_data.set_track_seed_model_auto(false, axis_id);
                meta_data.set_track_seed_model_transfer(false, axis_id);
            } else if axis_id == AxisID::First {
                if seeding_method == Some(seeding_method::TRANSFER_FID) {
                    meta_data.set_track_seed_model_transfer(true, axis_id);
                    meta_data.set_track_seed_model_manual(false, axis_id);
                    meta_data.set_track_seed_model_auto(false, axis_id);
                }
            } else if axis_id == AxisID::Second
                && seeding_method == Some(seeding_method::AUTO_FID_SEED)
            {
                meta_data.set_track_seed_model_auto(true, axis_id);
                meta_data.set_track_seed_model_manual(false, axis_id);
                meta_data.set_track_seed_model_transfer(false, axis_id);
            }
        }
        if directive_file.contains_axis(Some(DirectiveDef::TRACKING_METHOD), Some(axis_id)) {
            meta_data.set_track_method(
                axis_id,
                TrackingMethod::to_meta_data_value(
                    directive_file
                        .get_value_axis(Some(DirectiveDef::TRACKING_METHOD), Some(axis_id))
                        .as_deref(),
                ),
            );
        }
        if directive_file.contains_axis(Some(DirectiveDef::SIZE_IN_X_AND_Y), Some(axis_id)) {
            if let Err(e) = meta_data.set_size_to_output_in_x_and_y(
                axis_id,
                directive_file
                    .get_value_axis(Some(DirectiveDef::SIZE_IN_X_AND_Y), Some(axis_id))
                    .as_deref(),
            ) {
                let file = directive_file.get_file();
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!(
                        "Invalid directive file{}.  Invalid directive: {}.  {}",
                        match file {
                            Some(file) => format!(
                                ": {}",
                                utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                            ),
                            None => String::new(),
                        },
                        DirectiveDef::SIZE_IN_X_AND_Y,
                        e
                    ),
                    "Invalid Directive",
                    None,
                );
            }
        }
        if directive_file.contains_axis(
            Some(DirectiveDef::BIN_BY_FACTOR_FOR_ALIGNED_STACK),
            Some(axis_id),
        ) {
            meta_data.set_stack_binning_string(
                axis_id,
                directive_file
                    .get_value_axis(
                        Some(DirectiveDef::BIN_BY_FACTOR_FOR_ALIGNED_STACK),
                        Some(axis_id),
                    )
                    .as_deref(),
            );
        }
        if directive_file.contains_axis(Some(DirectiveDef::CORRECT_FOR_X_AXIS_TILT), Some(axis_id))
        {
            meta_data.set_use_stack_ctf_phase_flip_x_axis_tilt(
                axis_id,
                directive_file
                    .is_value_axis(Some(DirectiveDef::CORRECT_FOR_X_AXIS_TILT), Some(axis_id)),
            );
        }
        if directive_file.contains_axis(Some(DirectiveDef::AUTO_FIT_RANGE_AND_STEP), Some(axis_id))
        {
            if let Err(e) = meta_data.set_stack_ctf_auto_fit_range_and_step(
                axis_id,
                directive_file
                    .get_value_axis(Some(DirectiveDef::AUTO_FIT_RANGE_AND_STEP), Some(axis_id))
                    .as_deref(),
            ) {
                // Upstream bug fixed in translation (SetupReconUIHarness.java:845):
                // `directiveFile.getFile().getAbsolutePath()` throws a
                // NullPointerException for a directive file with no file; "null"
                // is printed for the path instead.
                ui_harness::open_message_dialog_from_process(
                    Some(self.manager),
                    &format!(
                        "Invalid directive file: {}.  Invalid directive: {}.  {}",
                        match directive_file.get_file() {
                            Some(file) =>
                                utilities::java_io_file_get_absolute_path(&file.to_string_lossy()),
                            None => "null".to_owned(),
                        },
                        DirectiveDef::AUTO_FIT_RANGE_AND_STEP,
                        e
                    ),
                    "Invalid Directive",
                    None,
                );
            }
        }
        if directive_file.contains_axis(Some(DirectiveDef::BINNING_FOR_GOLD_ERASING), Some(axis_id))
        {
            meta_data.set_stack_3d_find_binning_string(
                axis_id,
                directive_file
                    .get_value_axis(Some(DirectiveDef::BINNING_FOR_GOLD_ERASING), Some(axis_id))
                    .as_deref(),
            );
        }
        // GoldErasingThickness overrides the .com file
        if directive_file.contains_axis(
            Some(DirectiveDef::THICKNESS_FOR_GOLD_ERASING),
            Some(axis_id),
        ) {
            meta_data.set_stack_3d_find_thickness(
                axis_id,
                directive_file
                    .get_value_axis(
                        Some(DirectiveDef::THICKNESS_FOR_GOLD_ERASING),
                        Some(axis_id),
                    )
                    .as_deref(),
            );
        }
        if directive_file.contains_axis(Some(DirectiveDef::WHOLE_TOMOGRAM), Some(axis_id)) {
            meta_data.set_whole_tomogram_sample(
                axis_id,
                directive_file.is_value_axis(Some(DirectiveDef::WHOLE_TOMOGRAM), Some(axis_id)),
            );
        }
        if directive_file.contains_axis(Some(DirectiveDef::USE_SIRT), Some(axis_id)) {
            meta_data.set_gen_back_projection(
                axis_id,
                !directive_file.is_value_axis(Some(DirectiveDef::USE_SIRT), Some(axis_id)),
            );
        }
        if directive_file
            .contains_axis(Some(DirectiveDef::THICKNESS_FOR_POSITIONING), Some(axis_id))
        {
            meta_data.set_sample_thickness(
                axis_id,
                directive_file
                    .get_value_axis(Some(DirectiveDef::THICKNESS_FOR_POSITIONING), Some(axis_id))
                    .as_deref(),
            );
        }
        if directive_file.contains_axis(
            Some(DirectiveDef::BIN_BY_FACTOR_FOR_POSITIONING),
            Some(axis_id),
        ) {
            meta_data.set_pos_binning_string(
                axis_id,
                directive_file
                    .get_value_axis(
                        Some(DirectiveDef::BIN_BY_FACTOR_FOR_POSITIONING),
                        Some(axis_id),
                    )
                    .as_deref(),
            );
        }
        if directive_file.contains(Some(DirectiveDef::TARGET_MEASUREMENT_RATIO)) {
            meta_data.set_target_measurement_ratio(
                axis_id,
                directive_file
                    .get_value_axis(Some(DirectiveDef::TARGET_MEASUREMENT_RATIO), Some(axis_id))
                    .as_deref(),
            );
        }
        if directive_file.contains(Some(DirectiveDef::MIN_MEASUREMENT_RATIO)) {
            meta_data.set_min_measurement_ratio(
                axis_id,
                directive_file
                    .get_value_axis(Some(DirectiveDef::MIN_MEASUREMENT_RATIO), Some(axis_id))
                    .as_deref(),
            );
        }
        if directive_file.contains(Some(DirectiveDef::ORDER_OF_RESTRICTIONS)) {
            meta_data.set_order_of_restrictions(
                axis_id,
                directive_file
                    .get_value_axis(Some(DirectiveDef::ORDER_OF_RESTRICTIONS), Some(axis_id))
                    .as_deref(),
            );
        }
        if directive_file.contains(Some(DirectiveDef::SKIP_BEAM_TILT_WITH_ONE_ROT)) {
            meta_data.set_skip_beam_tilt_with_one_rot(
                axis_id,
                directive_file.is_value_axis(
                    Some(DirectiveDef::SKIP_BEAM_TILT_WITH_ONE_ROT),
                    Some(axis_id),
                ),
            );
        }
        if directive_file.contains_axis(Some(DirectiveDef::SAMPLE_TYPE), Some(axis_id)) {
            meta_data.set_sample_type_string(
                axis_id,
                directive_file
                    .get_value_axis(Some(DirectiveDef::SAMPLE_TYPE), Some(axis_id))
                    .as_deref(),
            );
        }
        if directive_file.contains_axis(Some(DirectiveDef::HAS_GOLD_BEADS), Some(axis_id)) {
            meta_data.set_has_gold_beads(
                axis_id,
                directive_file.is_value_axis(Some(DirectiveDef::HAS_GOLD_BEADS), Some(axis_id)),
            );
        }
        if directive_file.contains(Some(DirectiveDef::SLAB_THICKNESS_IN_NM)) {
            meta_data.set_gen_back_projection(axis_id, false);
            meta_data.set_gen_filter_trials(axis_id, false);
            meta_data.set_gen_sirt(axis_id, false);
        }
        if directive_file.contains(Some(DirectiveDef::ERASE_GOLD)) {
            meta_data.set_erase_gold_model_use_fid_erase_gold(
                axis_id,
                EraseGold::get_instance(
                    directive_file
                        .get_value_axis(Some(DirectiveDef::ERASE_GOLD), Some(axis_id))
                        .as_deref(),
                ),
            );
        }
    }

    /// Java `initializeFields(ConstMetaData, UserConfiguration)`.
    pub fn initialize_fields(
        &self,
        meta_data: &dyn ConstMetaData,
        user_config: &UserConfiguration,
    ) {
        if let Some(setup_interface) = self.get_setup_recon_interface() {
            setup_interface.init_tilt_angle_fields(
                AxisID::First,
                &meta_data.get_tilt_angle_spec_a(),
                user_config,
            );
            setup_interface.init_tilt_angle_fields(
                AxisID::Second,
                &meta_data.get_tilt_angle_spec_b(),
                user_config,
            );
        }
        if let Some(expert) = self.expert() {
            expert.initialize_fields(meta_data, user_config);
        }
        self.set_fei_pixel_size
            .set(user_config.is_set_fei_pixel_size());
    }
}
