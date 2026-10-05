//! Initial executable ownership slice of `etomo/ApplicationManager.java`.
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::base_manager::BaseManagerBase;
use crate::imod::etomo::comscript::alt_tomo_setup_param;
use crate::imod::etomo::comscript::alt_tomo_setup_param::AltTomoSetupParam;
use crate::imod::etomo::comscript::archiveorig_param;
use crate::imod::etomo::comscript::archiveorig_param::ArchiveorigParam;
use crate::imod::etomo::comscript::autofidseed_param::AutofidseedParam;
use crate::imod::etomo::comscript::batchruntomo_param::BatchruntomoParam;
use crate::imod::etomo::comscript::beadtrack_param::BeadtrackParam;
use crate::imod::etomo::comscript::blendmont_param;
use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::ccd_eraser_param;
use crate::imod::etomo::comscript::ccd_eraser_param::CCDEraserParam;
use crate::imod::etomo::comscript::clip_param::ClipParam;
use crate::imod::etomo::comscript::com_script_manager::ComScriptManager;
use crate::imod::etomo::comscript::com_script_util::ComScriptUtil;
use crate::imod::etomo::comscript::combine_comscript_state;
use crate::imod::etomo::comscript::combine_comscript_state::CombineComscriptState;
use crate::imod::etomo::comscript::command::Command;
use crate::imod::etomo::comscript::command_details::CommandDetails;
use crate::imod::etomo::comscript::command_mode::equals_mode;
use crate::imod::etomo::comscript::const_combine_params::ConstCombineParams;
use crate::imod::etomo::comscript::const_ctf_phase_flip_param::ConstCtfPhaseFlipParam;
use crate::imod::etomo::comscript::const_ctf_plotter_param::ConstCtfPlotterParam;
use crate::imod::etomo::comscript::const_mtf_filter_param::ConstMTFFilterParam;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::const_set_param;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::const_tiltalign_param::ConstTiltalignParam;
use crate::imod::etomo::comscript::const_tiltxcorr_param::ConstTiltxcorrParam;
use crate::imod::etomo::comscript::copy_tomo_coms::CopyTomoComs;
use crate::imod::etomo::comscript::cryo_position_param::CryoPositionParam;
use crate::imod::etomo::comscript::ctf_phase_flip_param::CtfPhaseFlipParam;
use crate::imod::etomo::comscript::ctf3d_setup_param::Ctf3dSetupParam;
use crate::imod::etomo::comscript::exclude_views_param::ExcludeViewsParam;
use crate::imod::etomo::comscript::extractmagrad_param;
use crate::imod::etomo::comscript::extractmagrad_param::ExtractmagradParam;
use crate::imod::etomo::comscript::extractpieces_param;
use crate::imod::etomo::comscript::extracttilts_param;
use crate::imod::etomo::comscript::find_beads3d_param::FindBeads3dParam;
use crate::imod::etomo::comscript::find_section_param::FindSectionParam;
use crate::imod::etomo::comscript::flatten_warp_param::FlattenWarpParam;
use crate::imod::etomo::comscript::fortran_input_string::FortranInputString;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::imodchopconts_param;
use crate::imod::etomo::comscript::imodchopconts_param::ImodchopcontsParam;
use crate::imod::etomo::comscript::makecomfile_param::MakecomfileParam;
use crate::imod::etomo::comscript::midas_param;
use crate::imod::etomo::comscript::midas_param::MidasParam;
use crate::imod::etomo::comscript::mtf_filter_param::MTFFilterParam;
use crate::imod::etomo::comscript::multifilt_setup_param::MultifiltSetupParam;
use crate::imod::etomo::comscript::newst_param;
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::comscript::processchunks_param::OutputImageFileKey;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::comscript::reduce_filt_vol_param::ReduceFiltVolParam;
use crate::imod::etomo::comscript::restrictalign_param::RestrictalignParam;
use crate::imod::etomo::comscript::set_env_param::SetEnvParam;
use crate::imod::etomo::comscript::sirtsetup_param::SirtsetupParam;
use crate::imod::etomo::comscript::solvematch_param::SolvematchParam;
use crate::imod::etomo::comscript::split_correction_param::SplitCorrectionParam;
use crate::imod::etomo::comscript::splitcombine_param;
use crate::imod::etomo::comscript::splittilt_param;
use crate::imod::etomo::comscript::splittilt_param::SplittiltParam;
use crate::imod::etomo::comscript::subtomo_setup_param;
use crate::imod::etomo::comscript::subtomo_setup_param::SubtomoSetupParam;
use crate::imod::etomo::comscript::tilt_param::TiltParam;
use crate::imod::etomo::comscript::tiltalign_log::TiltalignLog;
use crate::imod::etomo::comscript::tiltalign_param::TiltalignParam;
use crate::imod::etomo::comscript::tiltxcorr_param;
use crate::imod::etomo::comscript::tiltxcorr_param::TiltxcorrParam;
use crate::imod::etomo::comscript::tomodataplots_param;
use crate::imod::etomo::comscript::transferfid_param::TransferfidParam;
use crate::imod::etomo::comscript::trimvol_param;
use crate::imod::etomo::comscript::trimvol_param::TrimvolParam;
use crate::imod::etomo::comscript::warp_vol_param::WarpVolParam;
use crate::imod::etomo::comscript::xfmodel_param::XfmodelParam;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::local_arguments::LocalArguments;
use crate::imod::etomo::logic::converter;
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::logic::seeding_method;
use crate::imod::etomo::logic::seeding_method::SeedingMethod;
use crate::imod::etomo::logic::tomogram_tool::TomogramTool;
use crate::imod::etomo::logic::tracking_method;
use crate::imod::etomo::logic::tracking_method::TrackingMethod;
use crate::imod::etomo::logic::trimvol_input_file_state::TrimvolInputFileState;
use crate::imod::etomo::logic::trimvol_reorientation::TrimvolReorientation;
use crate::imod::etomo::process::base_imod_manager::ImodError;
use crate::imod::etomo::process::base_imod_manager::ImodManagerError;
use crate::imod::etomo::process::base_imod_manager::ImodManagerException;
use crate::imod::etomo::process::base_process_manager::BaseProcessManager;
use crate::imod::etomo::process::continuous_listener_target::ContinuousListenerTarget;
use crate::imod::etomo::process::emergency_monitor::EmergencyMonitor;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::BeadFixerMode;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process::process_data::ProcessData;
use crate::imod::etomo::process::process_interface::ProcessResultDisplayRef;
use crate::imod::etomo::process::process_interface::ProcessSeriesRef;
use crate::imod::etomo::process::process_manager::ProcessManager;
use crate::imod::etomo::process::process_manager::RunCommandError;
use crate::imod::etomo::process::process_messages::MessageType;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::process_series::Process;
use crate::imod::etomo::process_series::ProcessSeries;
use crate::imod::etomo::process_series::ProcessSeriesHandle;
use crate::imod::etomo::storage::batch_run_tomo_log::BatchRunTomoLog;
use crate::imod::etomo::storage::cpu_adoc;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file::DirectiveFile;
use crate::imod::etomo::storage::directive_map::DirectiveMap;
use crate::imod::etomo::storage::log_file::LockException;
use crate::imod::etomo::storage::log_file::LogFile;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::storable::Storable;
use crate::imod::etomo::storage::xray_stack_archive_filter::XrayStackArchiveFilter;
use crate::imod::etomo::task_interface::TaskInterface;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::base_process_track::BaseProcessTrack;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::base_state::BaseState;
use crate::imod::etomo::r#type::combine_process_type;
use crate::imod::etomo::r#type::combine_process_type::CombineProcessType;
use crate::imod::etomo::r#type::const_etomo_number;
use crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber;
use crate::imod::etomo::r#type::const_etomo_number::Number;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_double_to_string;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::const_string_parameter::ConstStringParameter;
use crate::imod::etomo::r#type::dialog_exit_state::DialogExitState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::dialog_type::TOTAL_RECON;
use crate::imod::etomo::r#type::directive_file_type::DirectiveFileType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::etomo_state;
use crate::imod::etomo::r#type::extension::Extension;
use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::r#type::file_key;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::imod_output_format;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::match_mode::MatchMode;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::pos_sample_type::PosSampleType;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result::ProcessResult;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::process_track::ProcessTrack;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::UiComponent;
use crate::imod::etomo::ui::setup_recon_ui_harness::SetupReconUIHarness;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::swing::abstract_parallel_dialog::AbstractParallelDialog;
use crate::imod::etomo::ui::swing::alignment_estimation_dialog::AlignmentEstimationDialog;
use crate::imod::etomo::ui::swing::alt_stack_display::AltStackDisplay;
use crate::imod::etomo::ui::swing::axis_progress_panel::AxisProgressPanel;
use crate::imod::etomo::ui::swing::bead_track_display::BeadTrackDisplay;
use crate::imod::etomo::ui::swing::bead_track_display::BeadTrackDisplayException;
use crate::imod::etomo::ui::swing::beadtrack_panel;
use crate::imod::etomo::ui::swing::blendmont_display::BlendmontDisplay;
use crate::imod::etomo::ui::swing::blendmont_display::BlendmontDisplayException;
use crate::imod::etomo::ui::swing::ccd_eraser_display::CcdEraserDisplay;
use crate::imod::etomo::ui::swing::clean_up_dialog::CleanUpDialog;
use crate::imod::etomo::ui::swing::coarse_align_dialog::CoarseAlignDialog;
use crate::imod::etomo::ui::swing::coarse_align_display::CoarseAlignDisplay;
use crate::imod::etomo::ui::swing::ctf3d_setup_display::Ctf3dSetupDisplay;
use crate::imod::etomo::ui::swing::deferred_3dmod_button::Deferred3dmodButton;
use crate::imod::etomo::ui::swing::fiducial_model_dialog::FiducialModelDialog;
use crate::imod::etomo::ui::swing::fiducialess_params::FiducialessParams;
use crate::imod::etomo::ui::swing::final_aligned_stack_dialog;
use crate::imod::etomo::ui::swing::final_aligned_stack_dialog::FinalAlignedStackDialog;
use crate::imod::etomo::ui::swing::final_aligned_stack_expert::FinalAlignedStackExpert;
use crate::imod::etomo::ui::swing::find_beads3d_display::FindBeads3dDisplay;
use crate::imod::etomo::ui::swing::flatten_warp_display::FlattenWarpDisplay;
use crate::imod::etomo::ui::swing::main_panel::MainPanelVirtual;
use crate::imod::etomo::ui::swing::main_tomogram_panel::MainTomogramPanel;
use crate::imod::etomo::ui::swing::multifilt_setup_display::MultifiltSetupDisplay;
use crate::imod::etomo::ui::swing::newstack_display::NewstackDisplay;
use crate::imod::etomo::ui::swing::newstack_display::NewstackDisplayException;
use crate::imod::etomo::ui::swing::parallel_panel::ParallelPanel;
use crate::imod::etomo::ui::swing::post_processing_dialog::PostProcessingDialog;
use crate::imod::etomo::ui::swing::pre_processing_dialog::PreProcessingDialog;
use crate::imod::etomo::ui::swing::process_dialog::ProcessDialog;
use crate::imod::etomo::ui::swing::process_dialog::ProcessDialogVirtual;
use crate::imod::etomo::ui::swing::process_display::ProcessDisplay;
use crate::imod::etomo::ui::swing::process_result_display_factory::ProcessResultDisplayFactory;
use crate::imod::etomo::ui::swing::reduce_filt_vol_display::ReduceFiltVolDisplay;
use crate::imod::etomo::ui::swing::setup_dialog_expert::SetupDialogExpert;
use crate::imod::etomo::ui::swing::sirtsetup_display::SirtsetupDisplay;
use crate::imod::etomo::ui::swing::subtomo_setup_display::SubtomoSetupDisplay;
use crate::imod::etomo::ui::swing::tilt_display::TiltDisplay;
use crate::imod::etomo::ui::swing::tilt_display::TiltDisplayException;
use crate::imod::etomo::ui::swing::tiltalign_panel::TiltalignParamsException;
use crate::imod::etomo::ui::swing::tiltxcorr_display::TiltXcorrDisplay;
use crate::imod::etomo::ui::swing::tomogram_combination_dialog;
use crate::imod::etomo::ui::swing::tomogram_combination_dialog::TomogramCombinationDialog;
use crate::imod::etomo::ui::swing::tomogram_generation_expert;
use crate::imod::etomo::ui::swing::tomogram_generation_expert::TomogramGenerationExpert;
use crate::imod::etomo::ui::swing::tomogram_positioning_expert::TomogramPositioningExpert;
use crate::imod::etomo::ui::swing::trial_tilt_display::TrialTiltDisplay;
use crate::imod::etomo::ui::swing::ui_expert::UIExpert;
use crate::imod::etomo::ui::swing::ui_expert_utilities::UIExpertUtilities;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::swing::ui_harness::INSTANCE as UI_HARNESS;
use crate::imod::etomo::ui::swing::warp_vol_display::WarpVolDisplay;
use crate::imod::etomo::util::dataset_files;
use crate::imod::etomo::util::event_queue;
use crate::imod::etomo::util::event_queue::EdtCell;
use crate::imod::etomo::util::event_queue::EdtRef;
use crate::imod::etomo::util::event_queue::ReentrantGuard;
use crate::imod::etomo::util::event_queue::ReentrantLock;
use crate::imod::etomo::util::event_queue::invoke_and_wait;
use crate::imod::etomo::util::event_queue::invoke_later;
use crate::imod::etomo::util::imodinfo::Imodinfo;
use crate::imod::etomo::util::mrc_header::MRCHeader;
use crate::imod::etomo::util::utilities;
use std::any::Any;
use std::convert::Infallible;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
pub struct ApplicationManager {
    base: BaseManagerBase,
    name: String,
    /// Java `metaData`.  Reassigned by `doneSetupDialog`
    /// (`ApplicationManager.java:544`), so each value is leaked and rooted in
    /// `ROOTS`, the way the collector keeps any object a caller still holds.
    meta_data: Mutex<Option<&'static MetaData>>,
    /// Java `processTrack` (`createProcessTrack`).
    process_track: Mutex<Option<&'static ProcessTrack>>,
    /// Java `final processMgr`.
    process_mgr: OnceLock<&'static ProcessManager>,
    /// Java `state` (`createState`).
    state: Mutex<Option<&'static TomogramState>>,
    advanced_a: Mutex<[bool; TOTAL_RECON as usize]>,
    advanced_b: Mutex<[bool; TOTAL_RECON as usize]>,
    /// Java `screenStateA`, created on first use.
    screen_state_a: Mutex<Option<&'static ReconScreenState>>,
    /// Java `screenStateB`, created on first use.
    screen_state_b: Mutex<Option<&'static ReconScreenState>>,
    /// Java `reconnectRunA`.
    reconnect_run_a: Mutex<bool>,
    /// Java `reconnectRunB`.
    reconnect_run_b: Mutex<bool>,
    /// Java package-private `comScriptMgr`.  Java reaches it from the EDT and
    /// from monitor threads with no lock; the `Rc`s inside it make an unlocked
    /// collision undefined here, so it is only handed out behind
    /// `com_script_lock` (see `get_com_script_manager`).
    com_script_mgr: Mutex<Option<ComScriptManagerRef>>,
    com_script_lock: ReentrantLock,
    /// Java `setupDialogExpert`.
    setup_dialog_expert: EdtCell<Rc<SetupDialogExpert>>,
    /// Java `preProcDialogA`.
    pre_proc_dialog_a: EdtCell<Rc<PreProcessingDialog>>,
    /// Java `preProcDialogB`.
    pre_proc_dialog_b: EdtCell<Rc<PreProcessingDialog>>,
    /// Java `coarseAlignDialogA`.
    coarse_align_dialog_a: EdtCell<Rc<CoarseAlignDialog>>,
    /// Java `coarseAlignDialogB`.
    coarse_align_dialog_b: EdtCell<Rc<CoarseAlignDialog>>,
    /// Java `fiducialModelDialogA`.
    fiducial_model_dialog_a: EdtCell<Rc<FiducialModelDialog>>,
    /// Java `fiducialModelDialogB`.
    fiducial_model_dialog_b: EdtCell<Rc<FiducialModelDialog>>,
    /// Java `fineAlignmentDialogA`.
    fine_alignment_dialog_a: EdtCell<Rc<AlignmentEstimationDialog>>,
    /// Java `fineAlignmentDialogB`.
    fine_alignment_dialog_b: EdtCell<Rc<AlignmentEstimationDialog>>,
    /// Java `tomogramPositioningExpertA`.
    tomogram_positioning_expert_a: EdtCell<Rc<TomogramPositioningExpert>>,
    /// Java `tomogramPositioningExpertB`.
    tomogram_positioning_expert_b: EdtCell<Rc<TomogramPositioningExpert>>,
    /// Java `finalAlignedStackExpertA`.
    final_aligned_stack_expert_a: EdtCell<Rc<FinalAlignedStackExpert>>,
    /// Java `finalAlignedStackExpertB`.
    final_aligned_stack_expert_b: EdtCell<Rc<FinalAlignedStackExpert>>,
    /// Java `tomogramGenerationExpertA`.
    tomogram_generation_expert_a: EdtCell<Rc<TomogramGenerationExpert>>,
    /// Java `tomogramGenerationExpertB`.
    tomogram_generation_expert_b: EdtCell<Rc<TomogramGenerationExpert>>,
    /// Java `tomogramCombinationDialog`.
    tomogram_combination_dialog: EdtCell<Rc<TomogramCombinationDialog>>,
    /// Java `postProcessingDialog`.
    post_processing_dialog: EdtCell<Rc<PostProcessingDialog>>,
    /// Java `cleanUpDialog`.
    clean_up_dialog: EdtCell<Rc<CleanUpDialog>>,
    /// Java `mainPanel` (null when headless).
    main_panel: EdtCell<Rc<MainTomogramPanel>>,
    /// Java `processResultDisplayFactoryA`.
    process_result_display_factory_a: EdtCell<Rc<ProcessResultDisplayFactory>>,
    /// Java `processResultDisplayFactoryB`.
    process_result_display_factory_b: EdtCell<Rc<ProcessResultDisplayFactory>>,
    /// Java `setupReconUIHarness`.
    setup_recon_ui_harness: EdtCell<Rc<SetupReconUIHarness>>,
    /// Java `emergencyMonitorA`.
    emergency_monitor_a: Mutex<Option<Arc<EmergencyMonitor>>>,
    /// Java `emergencyMonitorB`.
    emergency_monitor_b: Mutex<Option<Arc<EmergencyMonitor>>>,
}

/// A `ComScriptManager` reference that crosses threads.
///
/// SAFETY: `ComScriptManager` holds `Rc`/`RefCell` state and is not `Sync`.
/// The only way to reach the referent is `get_com_script_manager`, which holds
/// `com_script_lock` for the life of the returned guard, so no two threads
/// touch it at once; every `Rc` it creates stays inside it or inside a value
/// dropped before the guard is.
#[derive(Clone, Copy)]
struct ComScriptManagerRef(&'static ComScriptManager);
unsafe impl Send for ComScriptManagerRef {}
unsafe impl Sync for ComScriptManagerRef {}

/// `ComScriptManager` access held for one call (Java's unguarded
/// `comScriptMgr` field read).
pub struct ComScriptManagerGuard {
    manager: &'static ComScriptManager,
    _lock: ReentrantGuard<'static>,
}

impl std::ops::Deref for ComScriptManagerGuard {
    type Target = ComScriptManager;
    fn deref(&self) -> &ComScriptManager {
        self.manager
    }
}

/// Roots the per-manager objects Java replaces over a run (`metaData`,
/// `state`, `comScriptMgr`, ...): each is leaked on creation and kept here so
/// a reference a caller still holds stays valid, as it would in Java.
struct Roots {
    meta_data: Vec<&'static MetaData>,
    state: Vec<&'static TomogramState>,
    process_track: Vec<&'static ProcessTrack>,
    screen_state: Vec<&'static ReconScreenState>,
    com_script_mgr: Vec<usize>,
}
static ROOTS: Mutex<Roots> = Mutex::new(Roots {
    meta_data: Vec::new(),
    state: Vec::new(),
    process_track: Vec::new(),
    screen_state: Vec::new(),
    com_script_mgr: Vec::new(),
});

/// Owns every `ApplicationManager` this module builds.  Java's owner is the collector, by
/// way of `EtomoDirector.managerList`, which keeps each manager for the run;
/// the translation hands out `&'static Self`, so without a root here the
/// allocation is unreachable the moment the constructor returns.
static INSTANCES: std::sync::Mutex<Vec<&'static ApplicationManager>> =
    std::sync::Mutex::new(Vec::new());

/// Java `manager instanceof ApplicationManager` (and the cast), for a manager
/// held as `BaseManager`.
pub fn instance_of(manager: &'static dyn BaseManager) -> Option<&'static ApplicationManager> {
    INSTANCES.lock().unwrap().iter().copied().find(|m| {
        std::ptr::addr_eq(
            *m as *const ApplicationManager,
            manager as *const dyn BaseManager,
        )
    })
}

impl ApplicationManager {
    /// Java `ApplicationManager(String, AxisID)`: retain the dataset identity
    /// and run the shared manager construction before typed metadata/process
    /// state is attached by the subsequent translation units.
    pub fn new(param_file_name: Option<&str>, axis_id: Option<AxisID>) -> &'static Self {
        let manager = Box::leak(Box::new(Self {
            base: BaseManagerBase::initial(),
            name: param_file_name
                .filter(|name| !name.is_empty())
                .unwrap_or("Setup Tomogram")
                .to_owned(),
            meta_data: Mutex::new(None),
            process_track: Mutex::new(None),
            process_mgr: OnceLock::new(),
            state: Mutex::new(None),
            advanced_a: Mutex::new([false; TOTAL_RECON as usize]),
            advanced_b: Mutex::new([false; TOTAL_RECON as usize]),
            screen_state_a: Mutex::new(None),
            screen_state_b: Mutex::new(None),
            reconnect_run_a: Mutex::new(false),
            reconnect_run_b: Mutex::new(false),
            com_script_mgr: Mutex::new(None),
            com_script_lock: ReentrantLock::new(),
            setup_dialog_expert: EdtCell::new(),
            pre_proc_dialog_a: EdtCell::new(),
            pre_proc_dialog_b: EdtCell::new(),
            coarse_align_dialog_a: EdtCell::new(),
            coarse_align_dialog_b: EdtCell::new(),
            fiducial_model_dialog_a: EdtCell::new(),
            fiducial_model_dialog_b: EdtCell::new(),
            fine_alignment_dialog_a: EdtCell::new(),
            fine_alignment_dialog_b: EdtCell::new(),
            tomogram_positioning_expert_a: EdtCell::new(),
            tomogram_positioning_expert_b: EdtCell::new(),
            final_aligned_stack_expert_a: EdtCell::new(),
            final_aligned_stack_expert_b: EdtCell::new(),
            tomogram_generation_expert_a: EdtCell::new(),
            tomogram_generation_expert_b: EdtCell::new(),
            tomogram_combination_dialog: EdtCell::new(),
            post_processing_dialog: EdtCell::new(),
            clean_up_dialog: EdtCell::new(),
            main_panel: EdtCell::new(),
            process_result_display_factory_a: EdtCell::new(),
            process_result_display_factory_b: EdtCell::new(),
            setup_recon_ui_harness: EdtCell::new(),
            emergency_monitor_a: Mutex::new(None),
            emergency_monitor_b: Mutex::new(None),
        }));
        INSTANCES.lock().unwrap().push(manager);
        // Java field initialiser `comScriptMgr = new ComScriptManager(this)`;
        // `super()` then calls `createProcessTrack` and `createComScriptManager`
        // (the second ComScriptManager replaces the first, as in Java).
        manager.set_com_script_manager(ComScriptManager::new(manager));
        manager.base_manager();
        manager.set_meta_data(MetaData::new(
            Some(manager),
            manager.get_log_properties(),
            param_file_name.is_none_or(str::is_empty),
        ));
        manager.create_state();
        let _ = manager.process_mgr.set(ProcessManager::new(manager));
        manager.initialize_ui_parameters_from_name(param_file_name, axis_id);
        // Update from batchruntomo
        manager.get_meta_data().move_batchruntomo_settings();
        manager.get_state().move_batchruntomo_settings();
        manager.initialize_advanced();
        // ApplicationManager.java:291-310.
        // Open the etomo data file if one was found on the command line
        let param_file_name_is_empty = param_file_name.is_none_or(|name| name == "");
        let loaded_param_file = *manager.base().loaded_param_file.lock().unwrap();
        let new_dataset =
            (!param_file_name_is_empty && !loaded_param_file) || param_file_name_is_empty;
        if new_dataset {
            manager
                .setup_recon_ui_harness
                .set(Some(SetupReconUIHarness::new(manager, AxisID::Only)));
        }
        let is_headless = etomo_director::ARGUMENTS.lock().unwrap().is_headless();
        if !is_headless {
            if !param_file_name_is_empty {
                manager
                    .get_imod_manager()
                    .set_meta_data_const_meta_data(manager.get_meta_data());
                if *manager.base().loaded_param_file.lock().unwrap() {
                    manager.open_processing_panel();
                    manager.process_batch_run_tomo_log();
                    if let Some(main_panel) = manager.main_panel.get() {
                        main_panel.set_status_bar_text(
                            manager.base().param_file.lock().unwrap().as_deref(),
                            Some(manager.get_meta_data()),
                            manager.base().log_window.get().as_ref(),
                        );
                    }
                }
            }
            if new_dataset {
                manager.open_setup_dialog();
            }
        }
        //
        manager
    }

    /// Stores a new `metaData`, rooting it (Java assignment `metaData = ...`).
    fn set_meta_data(&self, meta_data: MetaData) {
        let meta_data: &'static MetaData = Box::leak(Box::new(meta_data));
        ROOTS.lock().unwrap().meta_data.push(meta_data);
        *self.meta_data.lock().unwrap() = Some(meta_data);
    }

    /// Stores a new `comScriptMgr` (Java assignment `comScriptMgr = ...`).
    fn set_com_script_manager(&self, com_script_mgr: ComScriptManager) {
        let _lock = self.com_script_lock.lock();
        let com_script_mgr: &'static ComScriptManager = Box::leak(Box::new(com_script_mgr));
        ROOTS
            .lock()
            .unwrap()
            .com_script_mgr
            .push(com_script_mgr as *const ComScriptManager as usize);
        *self.com_script_mgr.lock().unwrap() = Some(ComScriptManagerRef(com_script_mgr));
    }

    /// Java field read `comScriptMgr` / `getComScriptManager`: the manager,
    /// locked against a concurrent access from another thread for as long as
    /// the guard lives.
    pub fn get_com_script_manager(&'static self) -> ComScriptManagerGuard {
        let lock = self.com_script_lock.lock();
        let manager = self.com_script_mgr.lock().unwrap().expect("comScriptMgr").0;
        ComScriptManagerGuard {
            manager,
            _lock: lock,
        }
    }

    /// Java field read `processMgr`.
    pub fn get_process_mgr(&self) -> &'static ProcessManager {
        self.process_mgr.get().expect("processMgr")
    }

    /// Java field read `processTrack` (typed; `BaseManager.getProcessTrack`
    /// returns the same object).
    pub fn get_recon_process_track(&self) -> &'static ProcessTrack {
        self.process_track.lock().unwrap().expect("processTrack")
    }
}

impl BaseManager for ApplicationManager {
    fn base(&self) -> &BaseManagerBase {
        &self.base
    }
    fn this(&'static self) -> &'static dyn BaseManager {
        self
    }

    // ---- impl BaseManager for ApplicationManager ----
    //
    // Overrides in this range.  Plain `fn` items for the integrator to move into the trait
    // impl (replacing the existing `get_emergency_monitor` and `get_interface_type`).

    /// Java `getEmergencyMonitor(AxisID)` (ApplicationManager.java:314), `@Override`.  Can
    /// insert a error message into the progress bar with needing a process monitor.
    ///
    /// The source's double-checked `if (emergencyMonitorX == null) synchronized (this) { if
    /// (emergencyMonitorX == null) ... }`: the field's own lock serialises the test and the
    /// assignment, so the two tests collapse into one.
    fn get_emergency_monitor(&'static self, axis_id: Option<AxisID>) -> Arc<EmergencyMonitor> {
        if axis_id == Some(AxisID::Second) {
            let mut emergency_monitor_b = self.emergency_monitor_b.lock().unwrap();
            if emergency_monitor_b.is_none() {
                *emergency_monitor_b = Some(Arc::new(EmergencyMonitor::new(Some(self), axis_id)));
            }
            return Arc::clone(emergency_monitor_b.as_ref().unwrap());
        }

        // A axis
        let mut emergency_monitor_a = self.emergency_monitor_a.lock().unwrap();
        if emergency_monitor_a.is_none() {
            *emergency_monitor_a = Some(Arc::new(EmergencyMonitor::new(Some(self), axis_id)));
        }
        Arc::clone(emergency_monitor_a.as_ref().unwrap())
    }

    /// Java `doAutomation(LocalArguments)` (ApplicationManager.java:341), `@Override`.  Does
    /// the setup for a reconstruction dataset using automation.
    fn do_automation(&self, local_arguments: Option<&LocalArguments>) {
        if let Some(setup_recon_ui_harness) = self.setup_recon_ui_harness.get() {
            setup_recon_ui_harness.do_automation(local_arguments);
        }
        let is_directive = etomo_director::ARGUMENTS.lock().unwrap().is_directive();
        if !is_directive {
            // super.doAutomation(localArguments)
            self.do_automation_super(local_arguments);
        }
    }

    /// Java `getProcessResultDisplayFactoryInterface(AxisID)` (ApplicationManager.java:365),
    /// `@Override`.
    fn get_process_result_display_factory_interface(
        &self,
        axis_id: Option<AxisID>,
    ) -> Option<Rc<ProcessResultDisplayFactory>> {
        // Java passes the (possibly null) axis on; getProcessResultDisplayFactory maps
        // everything but SECOND to the A factory.
        Some(self.get_process_result_display_factory(axis_id.unwrap_or(AxisID::Only)))
    }

    /// Java `kill(AxisID)` (ApplicationManager.java:847), `@Override`.
    fn kill(&self, axis_id: Option<AxisID>) {
        // super.kill(axisID)
        self.kill_super(axis_id);
        if let Some(setup_dialog_expert) = self.setup_dialog_expert.get() {
            setup_dialog_expert.update_display(true);
        }
    }

    /// Java `getInterfaceType()` (ApplicationManager.java:1026), `@Override`.
    fn get_interface_type(&self) -> Option<InterfaceType> {
        Some(InterfaceType::Recon)
    }

    // ---- impl BaseManager for ApplicationManager ----
    //
    // Overrides of `BaseManager` methods declared in this range.  Plain `fn` items
    // for the integrator to move into `impl BaseManager for ApplicationManager`.

    /// Java `@Override canRunTomodataplots(TaskInterface, AxisID)`
    /// (ApplicationManager.java:2723).
    fn can_run_tomodataplots(
        &self,
        task: Option<&dyn TaskInterface>,
        axis_id: Option<AxisID>,
    ) -> bool {
        let is_coarse_mean_max = task.is_some_and(|task| {
            (task as &dyn std::any::Any)
                .downcast_ref::<tomodataplots_param::Task>()
                .is_some_and(|task| *task == tomodataplots_param::Task::CoarseMeanMax)
        });
        // `TomogramState.isXcorrBlendmontWasRun` reads axis B only for AxisID.SECOND; a
        // null axis reads axis A, as `AxisID::Only` does.
        if is_coarse_mean_max
            && self.get_meta_data().get_view_type() == ViewType::Montage
            && !self
                .get_state()
                .is_xcorr_blendmont_was_run(axis_id.unwrap_or(AxisID::Only))
        {
            return false;
        }
        true
    }

    /// Java `@Override isBeadfixerDiameterAvailable()` (ApplicationManager.java:3297).
    fn is_beadfixer_diameter_available(&self) -> bool {
        true
    }

    /// Java `@Override getBeadfixerDiameter(AxisID)` (ApplicationManager.java:3302).
    ///
    /// NEEDS `&'static self`: `UIExpertUtilities.getStackBinning` takes the manager as a
    /// `&'static dyn BaseManager`.
    fn get_beadfixer_diameter(&'static self, axis_id: Option<AxisID>) -> Option<i32> {
        let meta_data = self.get_meta_data();
        // Java passes the nullable axis through; `getStackBinning` needs one, and every
        // caller passes the axis of an open 3dmod.
        let axis_id = axis_id.unwrap_or(AxisID::Only);
        Some(utilities::java_lang_math_round(
            meta_data.get_fiducial_diameter()
                / meta_data.get_pixel_size()
                / UIExpertUtilities::INSTANCE.get_stack_binning_base_manager_axis_id_file_type(
                    self,
                    axis_id,
                    &file_type::CLASS.prealigned_stack,
                ) as f64,
        ) as i32)
    }

    /// Java `@Override exitProgram(AxisID)` (ApplicationManager.java:3571).  Call
    /// BaseManager.exitProgram().  Call saveDialog.  Return the value of
    /// BaseManager.exitProgram().  To guarantee that etomo can always exit, catch all
    /// unrecognized Exceptions and Errors and return true.
    fn exit_program(&'static self, axis_id: Option<AxisID>) -> bool {
        // Java `catch (final Throwable e) { e.printStackTrace(); return true; }`: a panic
        // anywhere below is caught the same way.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if self.exit_program_super(axis_id) {
                self.end_threads();
                let _ = self.save_param_file();
                return true;
            }
            false
        }))
        .unwrap_or(true)
    }

    /// Java `@Override save() throws LogFileException, IOException, LockException`
    /// (ApplicationManager.java:3587).
    ///
    /// NEEDS `&'static self` (it calls `saveDialogs`, which opens dialogs with `this`)
    /// and a `Result` for the three checked exceptions.
    fn save(&'static self) -> Result<bool, LogFileError> {
        self.save_super()?;
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.done();
        }
        self.save_dialogs();
        Ok(true)
    }

    /// Java `@Override saveAll(StringBuffer)` (ApplicationManager.java:3832).  Save the
    /// param file and the open dialogs and return a timestamp.  If returning null, popup a
    /// dialog containing errmsg before leaving.
    ///
    /// NEEDS `&'static self` (it calls `saveDialogs`) and
    /// `save_param_file() -> Result<bool, LogFileError>`.
    fn save_all(&'static self, errmsg: &mut String) -> Option<String> {
        let timestamp = utilities::get_date_time_stamp();
        match self.save_param_file() {
            Ok(_) => {}
            Err(LogFileError::Lock(_)) => {}
            // Java `catch (final LogFileException | IOException e)`.
            Err(e) => {
                eprintln!("{e:?}");
                errmsg.push_str(&format!(
                    "Unable to save dataset.  Directive values will be out of date.\n{e}"
                ));
            }
        }
        self.save_dialogs();
        Some(timestamp)
    }

    // ---- impl BaseManager for ApplicationManager ----

    /// Java `@Override processSeriesSucceeded(AxisID, ProcessName)`
    /// (ApplicationManager.java:4630).  Runs on the event dispatch thread
    /// (`BaseManager.processDone`).
    fn process_series_succeeded(&self, axis_id: Option<AxisID>, process_name: Option<ProcessName>) {
        if process_name == Some(ProcessName::TRANSFERFID) {
            let fiducial_model_dialog = if axis_id == Some(AxisID::Second) {
                self.fiducial_model_dialog_b.get()
            } else {
                self.fiducial_model_dialog_a.get()
            };
            let Some(fiducial_model_dialog) = fiducial_model_dialog else {
                return;
            };
            fiducial_model_dialog.update_display();
        }
    }

    // ---- impl BaseManager for ApplicationManager ----

    /// Java `@Override boolean isTomosnapshotThumbnail()` (ApplicationManager.java:9122).
    fn is_tomosnapshot_thumbnail(&self) -> bool {
        true
    }

    // Section wrapper only: the integrator moves these fns into the existing
    // `impl BaseManager for ApplicationManager` block.
    //
    // The overrides in this range.  Receivers are `&'static self` where the body passes
    // `this` on (ProcessSeries, FileType, Utilities) — see NEEDS.

    /// Java `startNextProcess(UIComponent, AxisID, ProcessSeries.Process,
    /// ProcessResultDisplay, ProcessSeries, DialogType, ProcessDisplay)`
    /// (ApplicationManager.java:9703), `@Override`.  Start the next process specified by
    /// the nextProcess string; returns true if the process is recognized.
    #[allow(clippy::too_many_arguments)]
    fn start_next_process(
        &'static self,
        ui_component: Option<Rc<dyn UiComponent>>,
        axis_id: AxisID,
        process: &Process,
        process_result_display: Option<ProcessResultDisplayRef>,
        process_series: &ProcessSeriesHandle,
        dialog_type: Option<DialogType>,
        display: Option<Rc<dyn ProcessDisplay>>,
    ) -> bool {
        if self.start_next_process_super(
            ui_component,
            axis_id,
            process,
            process_result_display.clone(),
            process_series,
            dialog_type,
            display.clone(),
        ) {
            return true;
        }
        // The manager's own methods take the EDT handle of the display.
        let process_result_display_handle: Option<ProcessResultDisplayHandle> =
            process_result_display
                .as_ref()
                .map(|display| Rc::clone(display.get()));
        let ui_expert =
            dialog_type.and_then(|dialog_type| self.get_ui_expert(Some(dialog_type), axis_id));
        if let Some(ui_expert) = ui_expert
            && ui_expert.start_next_process(
                process,
                process_result_display_handle.clone(),
                Some(Rc::clone(process_series)),
                dialog_type,
                display.clone(),
            )
        {
            return true;
        }
        if process.equals_string(Some("checkUpdateFiducialModel")) {
            self.check_update_fiducial_model(
                axis_id,
                process_result_display_handle,
                Some(Rc::clone(process_series)),
            );
            return true;
        }
        if process.equals_string(Some(archiveorig_param::command_name().as_str())) {
            self.archive_original_stack_axis_id_process_series_dialog_type(
                Some(AxisID::Second),
                Some(Rc::clone(process_series)),
                dialog_type,
            );
            return true;
        }
        if process.equals_string(Some(ProcessName::PROCESSCHUNKS.to_string().as_str()))
            && process.get_subprocess_name() == Some(ProcessName::VOLCOMBINE)
        {
            self.processchunks_volcombine(
                process_result_display_handle,
                Some(Rc::clone(process_series)),
                process.get_processing_method(),
            );
            return true;
        }
        if process.equals_task(&Task::Processchunks) {
            self.processchunks_process_series_command_axis_id(
                Some(Rc::clone(process_series)),
                process.get_command().cloned(),
                axis_id,
            );
            return true;
        }
        if process.equals_string(Some(splitcombine_param::COMMAND_NAME)) {
            self.splitcombine_process_series_deferred3dmod_button_run3dmod_menu_options_dialog_type_processing_method(
            Some(Rc::clone(process_series)),
            None,
            None,
            dialog_type,
            process.get_processing_method(),
        );
            return true;
        }
        if process.equals_string(Some(extractpieces_param::COMMAND_NAME)) {
            self.extractpieces(
                axis_id,
                process_result_display_handle,
                Some(Rc::clone(process_series)),
                dialog_type,
                self.get_meta_data().get_view_type(),
            );
            return true;
        }
        if process.equals_string(Some(extractmagrad_param::COMMAND_NAME)) {
            self.extractmagrad(axis_id, process_result_display_handle, process_series);
            return true;
        }
        if process.equals_string(Some(ProcessName::XCORR.to_string().as_str())) {
            // `(TiltXcorrDisplay) display`.
            // Fixed in translation: Java dereferences a null display inside
            // tiltxcorr (NullPointerException); the process is not started.
            if let Some(display) = display
                .as_deref()
                .and_then(|display| display.as_tilt_xcorr_display())
            {
                self.tiltxcorr(
                    axis_id,
                    process_result_display_handle,
                    None,
                    // Java passes a null Run3dmodMenuOptions with a null
                    // Deferred3dmodButton; the options are never read without the button.
                    Run3dmodMenuOptions::default(),
                    Some(Rc::clone(process_series)),
                    dialog_type,
                    display,
                    true,
                    ProcessName::XCORR, /*was: FileType.CROSS_CORRELATION_COMSCRIPT*/
                    true,
                    false,
                );
            }
            return true;
        }
        if process.equals_string(Some(ProcessName::ERASER.to_string().as_str())) {
            // `(CcdEraserDisplay) display`.
            // Fixed in translation: Java dereferences a null display inside
            // eraser (NullPointerException); the process is not started.
            if let Some(display) = display
                .as_deref()
                .and_then(|display| display.as_ccd_eraser_display())
            {
                self.eraser(
                    axis_id,
                    process_result_display_handle,
                    Some(Rc::clone(process_series)),
                    dialog_type,
                    display,
                );
            }
            return true;
        }
        if process.equals_task(&Task::ReloadAlignCom) {
            self.reload_align_com(axis_id, Some(Rc::clone(process_series)));
            return true;
        }
        if process.equals_task(&Task::CopyTomoComs) {
            self.copy_tomo_coms(Some(&Rc::clone(process_series)));
            return true;
        }
        if process.equals_task(&Task::SetupReconFailed) {
            self.msg_setup_recon_failed(Some(&Rc::clone(process_series)));
            return true;
        }
        if process.equals_task(&Task::ExcludeViewsA) {
            self.exclude_views(
                AxisID::First,
                process_series,
                None,
                process.get_axis_type(),
                None,
                process.get_command().cloned(),
            );
            return true;
        }
        if process.equals_task(&Task::ExcludeViewsB) {
            self.exclude_views(
                AxisID::Second,
                process_series,
                None,
                process.get_axis_type(),
                None,
                process.get_command().cloned(),
            );
            return true;
        }
        if process.equals_task(&Task::ValidateDirectiveFiles) {
            self.validate_directive_files(process_series, axis_id);
            return true;
        }
        if process.equals_task(&Task::SetRawStackExtension) {
            self.set_raw_image_stack_extension_process_series_axis_id(process_series, axis_id);
            return true;
        }
        if process.equals_task(&Task::SetOrigRawStackExtension) {
            self.set_orig_raw_image_stack_extension(Some(&Rc::clone(process_series)), axis_id);
            return true;
        }
        if process.equals_task(&Task::RenameInputImageFiles) {
            self.rename_input_image_file(process_series, axis_id);
            return true;
        }
        false
    }

    /// Java `updateDialog(ProcessName, AxisID)` (ApplicationManager.java:9800),
    /// `@Override`.  Runs on the event dispatch thread (`BaseManager.processDone`).
    fn update_dialog(&'static self, process_name: Option<ProcessName>, axis_id: Option<AxisID>) {
        self.update_dialog_fiducial_model_dialog_axis_id(
            self.fiducial_model_dialog_b.get(),
            AxisID::Second,
        );
        self.update_dialog_fiducial_model_dialog_axis_id(
            self.fiducial_model_dialog_a.get(),
            AxisID::First,
        );
        if process_name == Some(ProcessName::NEWST) || process_name == Some(ProcessName::BLEND) {
            // `((FinalAlignedStackExpert) getUIExpert(DialogType.FINAL_ALIGNED_STACK,
            // axisID)).updateDialog()`; a null axisID selects the A expert in
            // `getUIExpert` (it is not SECOND).
            if let Some(expert) = self.get_ui_expert(
                Some(DialogType::FinalAlignedStack),
                axis_id.unwrap_or(AxisID::Only),
            ) && let Some(expert) = expert.as_any().downcast_ref::<FinalAlignedStackExpert>()
            {
                expert.update_dialog();
            }
        }
    }

    /// Java `createComScriptManager()` (ApplicationManager.java:9912), `@Override`.
    // REPLACES existing
    fn create_com_script_manager(&self) {
        let this: &'static ApplicationManager = INSTANCES
            .lock()
            .unwrap()
            .iter()
            .copied()
            .find(|m| std::ptr::eq(*m, self))
            .expect("constructed ApplicationManager");
        this.set_com_script_manager(ComScriptManager::new(this));
    }

    /// Java `createMainPanel()` (ApplicationManager.java:9917), `@Override`.
    // REPLACES existing
    fn create_main_panel(&self) {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_headless() {
            // `new MainTomogramPanel(this)` needs the `&'static` manager (see
            // `create_com_script_manager`).
            let this: &'static ApplicationManager = INSTANCES
                .lock()
                .unwrap()
                .iter()
                .copied()
                .find(|m| std::ptr::eq(*m, self))
                .expect("constructed ApplicationManager");
            self.main_panel.set(Some(MainTomogramPanel::new(this)));
        }
    }

    /// Java `getViewType()` (ApplicationManager.java:9924), `@Override`.
    // REPLACES existing (the inherent `get_view_type`)
    fn get_view_type(&self) -> ViewType {
        self.get_meta_data().get_view_type()
    }

    /// Java `setParamFile(File)` (ApplicationManager.java:9936), `@Override`.  Set the
    /// data set parameter file.  This also updates the mainframe data parameters.
    fn set_param_file_from(&self, param_file: Option<&Path>) -> bool {
        if !self.set_param_file_from_super(param_file) {
            return false;
        }
        // Update main window information and status bar
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_status_bar_text(
                param_file,
                Some(self.get_meta_data() as &dyn BaseMetaData),
                self.get_log_window().as_ref(),
            );
        }
        true
    }

    /// Java `createProcessTrack()` (ApplicationManager.java:9946), `@Override`.
    // REPLACES existing
    fn create_process_track(&self) {
        let process_track: &'static ProcessTrack = Box::leak(Box::new(ProcessTrack::new()));
        ROOTS.lock().unwrap().process_track.push(process_track);
        *self.process_track.lock().unwrap() = Some(process_track);
    }

    /// Java `getBaseState()` (ApplicationManager.java:9959), `@Override`.
    fn get_base_state(&self) -> Option<&'static dyn BaseState> {
        self.state
            .lock()
            .unwrap()
            .map(|state| state as &'static dyn BaseState)
    }

    /// Java `getBaseMetaData()` (ApplicationManager.java:10128), `@Override`.
    // REPLACES existing
    fn get_base_meta_data(&self) -> Option<&dyn BaseMetaData> {
        self.meta_data
            .lock()
            .unwrap()
            .map(|meta_data| meta_data as &dyn BaseMetaData)
    }

    /// Java `getMainPanel()` (ApplicationManager.java:10133), `@Override`: returns
    /// `mainPanel` (null in headless mode).
    // REPLACES existing
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.main_panel
            .get()
            .map(|main_panel| main_panel as Rc<dyn MainPanelVirtual>)
    }

    /// Java package-private `getProcessTrack(Storable[], int)`
    /// (ApplicationManager.java:10155), `@Override`.
    fn get_process_track_into(
        &self,
        storable: Option<&mut [Option<&'static dyn Storable>]>,
        index: i32,
    ) {
        let Some(storable) = storable else {
            return;
        };
        storable[index as usize] = self
            .process_track
            .lock()
            .unwrap()
            .map(|process_track| process_track as &'static dyn Storable);
    }

    /// Java package-private `getProcessTrack()` (ApplicationManager.java:10163),
    /// `@Override`.
    fn get_process_track(&self) -> Option<&'static dyn BaseProcessTrack> {
        self.process_track
            .lock()
            .unwrap()
            .map(|process_track| process_track as &'static dyn BaseProcessTrack)
    }

    /// Java `getProcessManager()` (ApplicationManager.java:11156), `@Override`.
    // REPLACES existing
    fn get_process_manager(&self) -> Option<&'static BaseProcessManager> {
        self.process_mgr.get().map(|process_mgr| &process_mgr.base)
    }

    /// Java package-private `getStorables(int)` (ApplicationManager.java:11161),
    /// `@Override`.  Slots before `offset` stay null, as in the Java array.
    // REPLACES existing
    fn get_storables_with_offset(&self, offset: i32) -> Option<Vec<Option<&'static dyn Storable>>> {
        let meta_data = self.get_meta_data();
        let mut array_size = 5 + offset;
        let mut dual_axis = true;
        if meta_data.base().get_axis_type() == AxisType::SingleAxis {
            dual_axis = false;
            array_size -= 1;
        }
        let mut storables: Vec<Option<&'static dyn Storable>> = vec![None; array_size as usize];
        let mut index = offset as usize;
        storables[index] = Some(meta_data as &'static dyn Storable);
        index += 1;
        storables[index] = Some(self.get_state() as &'static dyn Storable);
        index += 1;
        storables[index] = Some(self.get_recon_process_track() as &'static dyn Storable);
        index += 1;
        storables[index] = Some(self.get_screen_state(AxisID::First) as &'static dyn Storable);
        index += 1;
        if dual_axis {
            storables[index] = Some(self.get_screen_state(AxisID::Second) as &'static dyn Storable);
        }
        Some(storables)
    }

    /// Java `getBaseScreenState(AxisID)` (ApplicationManager.java:11194), `@Override`.
    fn get_base_screen_state(&self, axis_id: Option<AxisID>) -> Option<&'static BaseScreenState> {
        // `getScreenState(null)` takes the A branch (null is not SECOND).
        let screen_state: &'static ReconScreenState =
            self.get_screen_state(axis_id.unwrap_or(AxisID::First));
        Some(&**screen_state)
    }

    /// Java `canChangeParamFileName()` (ApplicationManager.java:11199), `@Override`.
    fn can_change_param_file_name(&self) -> bool {
        // if the param file hasn't been loaded, any param name that is added while
        // be overwritten when Setup Tomogram is complete, so don't allow the user
        // to
        // do a Save As.
        *self.base().loaded_param_file.lock().unwrap()
    }

    /// Java `canSaveDirectives()` (ApplicationManager.java:11208), `@Override`.
    fn can_save_directives(&self) -> bool {
        *self.base().loaded_param_file.lock().unwrap()
    }

    /// Java `updateDirectiveMap(DirectiveMap, StringBuffer)` (ApplicationManager.java:3854),
    /// reached through `BaseManager::update_directive_map_directive_map` (see the
    /// upstream-bug note there).
    fn update_directive_map_directive_map(
        &'static self,
        directive_map: &DirectiveMap,
        errmsg: &mut String,
    ) {
        ApplicationManager::update_directive_map(self, directive_map, errmsg);
    }

    /// Java `getName()` (ApplicationManager.java:11213), `@Override`.
    // REPLACES existing (which returned the constructor argument; the Java reads metaData)
    fn get_name(&self) -> Option<String> {
        let meta_data = *self.meta_data.lock().unwrap();
        match meta_data {
            None => Some(MetaData::get_new_file_title().to_string()),
            Some(meta_data) => Some(meta_data.get_name()),
        }
    }
}

/// `ApplicationManager implements ContinuousListenerTarget`
/// (etomo/process/ContinuousListenerTarget.java).
impl ContinuousListenerTarget for ApplicationManager {
    /// Java `@Override getContinuousMessage(String, AxisID)` (ApplicationManager.java:3321).
    ///
    /// Called from the 3dmod listener thread; neither call below touches a dialog field
    /// or `mainPanel` directly (`generateAlignLogs` posts its own popup).
    fn get_continuous_message(&self, message: &str, axis_id: Option<AxisID>) {
        let axis_id = axis_id.unwrap_or(AxisID::Only);
        if message.contains("Tiltalign ran with exit code 0") {
            self.get_process_mgr().generate_align_logs(axis_id);
            let generator = self
                .get_process_mgr()
                .generate_align_log_for_project_log(axis_id);
            self.log_message_loggable(Some(&generator), Some(axis_id));
        }
    }
}

// RAPTOR (`ApplicationManager.java:9408-9572`).
impl ApplicationManager {
    /// Java package-private `updateRunraptorParam(AxisID, boolean)`.
    pub fn update_runraptor_param(
        &'static self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<crate::imod::etomo::comscript::runraptor_param::RunraptorParam> {
        let mut param =
            crate::imod::etomo::comscript::runraptor_param::RunraptorParam::new(self, axis_id);
        // `(FiducialModelDialog) getDialog(DialogType.FIDUCIAL_MODEL, axisID)`.
        let dialog = if axis_id == AxisID::Second {
            self.fiducial_model_dialog_b.get()
        } else {
            self.fiducial_model_dialog_a.get()
        };
        let Some(dialog) = dialog else {
            UI_HARNESS.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    "Unable to get information from the fiducial model dialog.",
                    "Etomo Error",
                    Some(axis_id),
                )
            });
            return None;
        };
        if !dialog.get_parameters_runraptor_param_boolean(&mut param, do_validation) {
            return None;
        }
        Some(param)
    }

    /// Java `runraptor(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, AxisID)`: execute runraptor.
    pub fn runraptor(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        axis_id: AxisID,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("runraptor")),
        };
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let Some(mut param) = self.update_runraptor_param(axis_id, true) else {
            process_series.borrow().end_series();
            return;
        };
        // Attempt to backup the output file
        match crate::imod::etomo::storage::log_file::LogFile::get_instance_file(
            Some(&dataset_files::get_raptor_fiducial_model(
                self,
                Some(axis_id),
            )),
            Some(self.get_emergency_monitor(Some(axis_id))),
        )
        .and_then(|output_file| output_file.backup())
        {
            Ok(_) | Err(crate::imod::etomo::storage::log_file::LogFileError::Lock(_)) => {}
            Err(e) => eprintln!("{e}"),
        }
        // Run process
        // `if (processTrack != null)`: the process track always exists here.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        let thread_name = match self.get_process_mgr().runraptor(
            &mut param,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series))) as ProcessSeriesRef),
            axis_id,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec!["Can not execute runraptor".to_owned(), e.0.clone()];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Running runraptor"),
                axis_id,
                Some(&ProcessName::RUNRAPTOR),
            );
        }
    }

    /// Java `imodRunraptorResult(AxisID, Run3dmodMenuOptions)`: open 3dmod with the
    /// runraptor result.
    pub fn imod_runraptor_result(
        &'static self,
        axis_id: AxisID,
        menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let imod_manager_key = imod_manager::COARSE_ALIGNED_KEY;
        let model = dataset_files::get_raptor_fiducial_model_name(self, Some(axis_id));
        let result = (|| -> Result<(), ImodManagerException> {
            let tilt_file = dataset_files::get_raw_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                self.get_imod_manager().set_tilt_file(
                    imod_manager_key,
                    Some(axis_id),
                    tilt_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                self.get_imod_manager()
                    .reset_tilt_file(imod_manager_key, Some(axis_id))?;
            }
            self.get_imod_manager()
                .open_string_axis_id_string_boolean_run3dmod_menu_options(
                    imod_manager_key,
                    Some(axis_id),
                    Some(&model),
                    true,
                    menu_options,
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
                return;
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!(
                            "Can't open 3dmod on {imod_manager_key} with model: {model}\n{except}"
                        ),
                        "Can't Open 3dmod",
                        Some(axis_id),
                    )
                });
                return;
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler
            // prints it, and the rest of the method does not run.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
                return;
            }
        }
        self.get_state().set_fixed_fiducials(axis_id, true);
    }

    /// Java `useRunraptorResult(ProcessResultDisplay, AxisID, DialogType)`: replace
    /// `.fid` with `_raptor.fid`.
    pub fn use_runraptor_result(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        use crate::imod::etomo::storage::log_file::{LogFile, LogFileError};
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        if self.is_axis_busy(axis_id, process_result_display_ref.clone()) {
            return;
        }
        let raptor_dataset_file = dataset_files::get_raptor_fiducial_model(self, Some(axis_id));
        let raptor_name = raptor_dataset_file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !raptor_dataset_file.exists() {
            UI_HARNESS.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &format!("{raptor_name} does not exist"),
                    "Entry Error",
                    Some(axis_id),
                )
            });
            return;
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_progress_bar_string_int_boolean_axis_id(
                Some("Using RAPTOR result as fiducial model"),
                1,
                false,
                axis_id,
            );
        }
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        // Back up .fid file
        let fid_file = dataset_files::get_fiducial_model_file(self, Some(axis_id));
        match LogFile::get_instance_file(
            Some(&fid_file),
            Some(self.get_emergency_monitor(Some(axis_id))),
        )
        .and_then(|fid_handle| fid_handle.backup())
        {
            Ok(_) => {}
            Err(LogFileError::Lock(_)) => {
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return;
            }
            Err(e) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string(
                        Some(self),
                        &format!(
                            "Unable to backup {}",
                            dataset_files::get_fiducial_model_name(self, Some(axis_id))
                        ),
                        "File Error",
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return;
            }
        }
        // Rename _raptor.fid file
        match LogFile::get_instance_file(
            Some(&raptor_dataset_file),
            Some(self.get_emergency_monitor(Some(axis_id))),
        )
        .and_then(|raptor_file| {
            raptor_file.rename(
                Some(self),
                Some(axis_id),
                Some(&fid_file),
                false,
                false,
                false,
            )
        }) {
            Ok(true) => {}
            Ok(false) => {
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string(
                        Some(self),
                        &format!("Unable to rename {raptor_name}"),
                        "File Error",
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return;
            }
            Err(LogFileError::Lock(_)) => {
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return;
            }
            Err(e) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string(
                        Some(self),
                        &format!("Unable to rename {raptor_name}"),
                        "File Error",
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return;
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id(axis_id);
        }
        self.send_msg_process_succeeded(process_result_display_ref.as_ref());
        self.get_state().set_use_raptor_result_warning(false);
    }
}

// ApplicationManager.java, the rest of the class (spliced from the staged parts 1-6).

impl ApplicationManager {
    /// Java private `initializeAdvanced()` (ApplicationManager.java:356).  Initialize
    /// advancedA and advancedB, which remember the advanced state of dialogs when the
    /// user switches from one to another.  This is necessary because the done function
    /// is run each time a user switches dialogs.
    ///
    /// Replaces the existing `initialize_advanced(bool)`: the constructor's call becomes
    /// `manager.initialize_advanced()`.
    fn initialize_advanced(&self) {
        let is_advanced = etomo_director::INSTANCE.get_advanced();
        let mut advanced_a = self.advanced_a.lock().unwrap();
        let mut advanced_b = self.advanced_b.lock().unwrap();
        for i in 0..TOTAL_RECON as usize {
            advanced_a[i] = is_advanced;
            advanced_b[i] = is_advanced;
        }
    }

    /// Java `getProcessResultDisplayFactory(AxisID)` (ApplicationManager.java:370).
    pub fn get_process_result_display_factory(
        &self,
        axis_id: AxisID,
    ) -> Rc<ProcessResultDisplayFactory> {
        if axis_id == AxisID::Second {
            if !self.process_result_display_factory_b.is_some() {
                self.process_result_display_factory_b.set(Some(
                    ProcessResultDisplayFactory::get_instance(
                        &**self.get_screen_state(axis_id),
                        axis_id,
                        self.get_meta_data().get_axis_type(),
                    ),
                ));
            }
            return self.process_result_display_factory_b.get().unwrap();
        }
        if !self.process_result_display_factory_a.is_some() {
            self.process_result_display_factory_a.set(Some(
                ProcessResultDisplayFactory::get_instance(
                    &**self.get_screen_state(axis_id),
                    axis_id,
                    self.get_meta_data().get_axis_type(),
                ),
            ));
        }
        self.process_result_display_factory_a.get().unwrap()
    }

    /// Java `isAdvanced(DialogType, AxisID)` (ApplicationManager.java:392).  Checks the
    /// advanced state of a dialog.
    pub fn is_advanced(&self, dialog_type: DialogType, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.advanced_b.lock().unwrap()[dialog_type.to_index() as usize];
        }
        self.advanced_a.lock().unwrap()[dialog_type.to_index() as usize]
    }

    /// Java `setAdvanced(DialogType, AxisID, boolean)` (ApplicationManager.java:406).
    /// Sets the advanced state of a dialog.  The A entry is set whatever the axis, as
    /// in the source.
    pub fn set_advanced_dialog_type_axis_id_boolean(
        &self,
        dialog_type: DialogType,
        axis_id: AxisID,
        advanced: bool,
    ) {
        if axis_id == AxisID::Second {
            self.advanced_b.lock().unwrap()[dialog_type.to_index() as usize] = advanced;
        }
        self.advanced_a.lock().unwrap()[dialog_type.to_index() as usize] = advanced;
    }

    /// Java `setAdvanced(DialogType, boolean)` (ApplicationManager.java:420).  Sets the
    /// advanced state of a dialog.  Assumes the A axis (or single axis).
    pub fn set_advanced_dialog_type_boolean(&self, dialog_type: DialogType, advanced: bool) {
        self.advanced_a.lock().unwrap()[dialog_type.to_index() as usize] = advanced;
    }

    /// Java `isNewManager()` (ApplicationManager.java:428).  Finds out whether the
    /// manager is new, which means that it has no .edf file.  If setupDialog is not
    /// null, then the manager is new.
    pub fn is_new_manager(&self) -> bool {
        self.setup_dialog_expert.is_some()
    }

    /// Java `isSetupChanged()` (ApplicationManager.java:438).  Check if setup dialog has
    /// been modified by the user.  Return true if there is text in the dataset field.
    pub fn is_setup_changed(&self) -> bool {
        let Some(setup_dialog_expert) = self.setup_dialog_expert.get() else {
            return false;
        };
        let raw_image_stack = setup_dialog_expert.get_raw_image_stack();
        raw_image_stack.is_some_and(|raw_image_stack| {
            !const_etomo_number::java_lang_string_matches_whitespace(&raw_image_stack)
        })
    }

    /// Java `openSetupDialog()` (ApplicationManager.java:449).  Open the setup dialog.
    pub fn open_setup_dialog(&'static self) {
        // Open the dialog in the appropriate mode for the current state of
        // processing
        let action_message =
            self.set_current_dialog_type(Some(DialogType::SetupRecon), Some(AxisID::Only));
        let setup_recon_ui_harness = self
            .setup_recon_ui_harness
            .get()
            .expect("openSetupDialog: setupReconUIHarness is created for a new dataset");
        if !self.setup_dialog_expert.is_some() {
            utilities::timestamp_process_container_status(
                Some("new"),
                Some("SetupDialog"),
                Some(utilities::STARTED_STATUS),
            );
            let progress_panel = self
                .main_panel
                .get()
                .map(|main_panel| main_panel.map_axis_progress_panel(AxisID::Only));
            self.setup_dialog_expert
                .set(setup_recon_ui_harness.get_setup_dialog_expert(progress_panel.flatten()));
            utilities::timestamp_process_container_status(
                Some("new"),
                Some("SetupDialog"),
                Some(utilities::FINISHED_STATUS),
            );
            etomo_director::INSTANCE.with_user_configuration(|user_config| {
                setup_recon_ui_harness.initialize_fields(self.get_meta_data(), user_config)
            });
        }
        if let Some(main_panel) = self.main_panel.get() {
            // Fixed in translation: Java passes a null expert when SetupReconUIHarness
            // could not build one, and openSetupPanel dereferences it.
            if let Some(setup_dialog_expert) = self.setup_dialog_expert.get() {
                main_panel.open_setup_panel(&setup_dialog_expert);
            }
            // Swing layout: if (!GraphicsEnvironment.isHeadless()) center the frame on the
            // screen - Toolkit.getScreenSize(), mainPanel.getSize(),
            // mainPanel.setLocation((screen - frame) / 2).
        }
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java package-private `updateBatchruntomo(boolean, DirectiveFile, AxisID)`
    /// (ApplicationManager.java:473).
    fn update_batchruntomo(
        &'static self,
        directive_driven_automation: bool,
        directive_file: Option<&DirectiveFile>,
        axis_id: AxisID,
    ) -> Option<BatchruntomoParam> {
        let directive_file = directive_file?;
        let mut param = BatchruntomoParam::get_validation_instance(self, axis_id);
        param.set_validation_type(directive_driven_automation);
        param.add_directive_file_directive_file(Some(directive_file));
        if param.is_valid() {
            return Some(param);
        }
        // If the batchruntomo is invalid, it just means that no directive files where added
        // to it and there is nothing to do.
        None
    }

    /// Java package-private `updateBatchruntomoForRenameInputImageFiles(AxisID)`
    /// (ApplicationManager.java:489).
    fn update_batchruntomo_for_rename_input_image_files(
        &'static self,
        axis_id: AxisID,
    ) -> Option<BatchruntomoParam> {
        let is_from_brt = etomo_director::ARGUMENTS.lock().unwrap().is_from_brt();
        if is_from_brt {
            // Batchruntomo will take care of this.
            return None;
        }
        let mut param = BatchruntomoParam::get_rename_input_image_files_instance(self, axis_id);
        let setup_recon_ui_harness = self
            .setup_recon_ui_harness
            .get()
            .expect("setupReconUIHarness exists while setup runs");
        if setup_recon_ui_harness.get_parameters_for_rename_input_image_files(&mut param) {
            return Some(param);
        }
        None
    }

    /// Java `doneSetupDialog(boolean, boolean, String, boolean, AxisProgressPanel)`
    /// (ApplicationManager.java:506).  Close message from the setup dialog window.
    pub fn done_setup_dialog(
        &'static self,
        mut remove_excluded_views_a: bool,
        mut remove_excluded_views_b: bool,
        dataset_dir: Option<&str>,
        dual_axis: bool,
        progress_panel: Option<Rc<AxisProgressPanel>>,
    ) {
        let axis_type = if dual_axis {
            AxisType::DualAxis
        } else {
            AxisType::SingleAxis
        };
        if let Some(progress_panel) = &progress_panel {
            progress_panel.correct_axis_id(axis_type);
            progress_panel.set_visible(true);
        }
        let setup_recon_ui_harness = self
            .setup_recon_ui_harness
            .get()
            .expect("doneSetupDialog: setupReconUIHarness exists while setup runs");
        let exit_state = setup_recon_ui_harness.get_exit_state();
        if exit_state == Some(DialogExitState::Cancel) {
            etomo_director::INSTANCE.close_current_manager(Some(AxisID::Only), false);
            return;
        }
        let mut axis_id = AxisID::Only;
        // set Failure process
        let process_series = ProcessSeries::new(
            self,
            axis_id,
            Some(DialogType::SetupRecon),
            Some("doneSetupDialog"),
        );
        process_series
            .borrow_mut()
            .set_fail_process(Rc::new(Task::SetupReconFailed));
        // Get the selected exit button
        if !setup_recon_ui_harness.is_valid() {
            ProcessSeries::start_fail_process(&process_series, axis_id);
            return;
        }
        // Set the current working directory for the application saving the
        // old user.dir property until the meta data is valid
        let old_user_dir = self.get_property_user_dir();
        let property_user_dir = setup_recon_ui_harness
            .get_working_directory()
            .map(|dir| utilities::java_io_file_get_absolute_path(&dir.to_string_lossy()));
        *self.base().property_user_dir.lock().unwrap() = property_user_dir.clone();
        if property_user_dir
            .as_deref()
            .is_some_and(|dir| dir.ends_with(' '))
        {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &format!(
                        "The directory, {}, cannot be used because it ends with a space.",
                        property_user_dir.as_deref().unwrap_or("null")
                    ),
                    "Unusable Directory Name",
                    Some(AxisID::Only),
                )
            });
            *self.base().property_user_dir.lock().unwrap() = old_user_dir;
            ProcessSeries::start_fail_process(&process_series, axis_id);
            return;
        }
        // Set up metadata and param file - for EXECUTE and POSTPONE
        let do_validation = exit_state == Some(DialogExitState::Execute);
        // Fixed in translation (ApplicationManager.java:544, BUGS.md): Java assigns the
        // result to the field even when it is null (a field failed validation, say), and
        // every later `metaData` dereference - getScreenState, isDualAxis, the save on
        // exit - then throws NullPointerException.  The field is replaced only by a
        // non-null result; the failure path below is otherwise the source's.
        let fields = setup_recon_ui_harness.get_fields(do_validation);
        let fields_are_null = fields.is_none();
        if let Some(fields) = fields {
            self.set_meta_data(fields);
        }
        self.copy_directive_files();
        if fields_are_null {
            ProcessSeries::start_fail_process(&process_series, axis_id);
            return;
        }
        let meta_data = self.get_meta_data();
        if BaseMetaData::is_valid(meta_data) {
            let raw_stack_extension = setup_recon_ui_harness.get_raw_stack_extension(do_validation);
            let Some(raw_stack_extension) = raw_stack_extension else {
                ProcessSeries::start_fail_process(&process_series, axis_id);
                return;
            };
            if setup_recon_ui_harness.check_for_shared_directory(raw_stack_extension) {
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!(
                            "This directory ({}) is already being used by an .edf file.  Either open the existing .edf file or create a new directory for the new reconstruction.",
                            self.get_property_user_dir().as_deref().unwrap_or("null")
                        ),
                        "WARNING:  CANNOT PROCEED",
                        Some(AxisID::Only),
                    )
                });
                ProcessSeries::start_fail_process(&process_series, axis_id);
                return;
            }
            meta_data.set_raw_image_stack_extension(Some(raw_stack_extension));
            meta_data.set_orig_raw_image_stack_extension(Some(raw_stack_extension));
            self.get_recon_process_track()
                .set_setup_state(ProcessState::InProgress);
            // final initialization of IMOD manager
            self.get_imod_manager()
                .set_meta_data_const_meta_data(meta_data);
            // set paramFile so meta data can be saved
            let param_file = PathBuf::from(utilities::java_io_file_new(
                self.get_property_user_dir().as_deref().unwrap_or(""),
                &meta_data.get_meta_data_file_name(),
            ));
            *self.base().param_file.lock().unwrap() = Some(param_file.clone());
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_status_bar_text(
                    Some(&param_file),
                    Some(meta_data),
                    self.base().log_window.get().as_ref(),
                );
            }
            if etomo_director::INSTANCE
                .with_user_configuration(|user_config| user_config.get_swap_y_and_z())
            {
                meta_data.set_post_trimvol_swap_yz(true);
            }
            let data_file =
                utilities::java_io_file_get_absolute_path(&param_file.to_string_lossy());
            etomo_director::INSTANCE.with_user_configuration_mut(|user_config| {
                user_config.put_data_file(Some(&data_file))
            });
            *self.base().loaded_param_file.lock().unwrap() = true;
            self.get_state().initialize();
        } else {
            let error_message = vec![
                "Setup Parameter Error".to_owned(),
                BaseMetaData::base(meta_data).get_invalid_reason(),
            ];
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                    Some(self),
                    &error_message,
                    "Setup Parameter Error",
                    Some(AxisID::Only),
                )
            });
            *self.base().property_user_dir.lock().unwrap() = old_user_dir;
            ProcessSeries::start_fail_process(&process_series, axis_id);
            return;
        }
        if exit_state == Some(DialogExitState::Execute) {
            // Set processes
            process_series
                .borrow_mut()
                .add_process(Rc::new(Task::ValidateDirectiveFiles));
            process_series
                .borrow_mut()
                .add_process(Rc::new(Task::SetRawStackExtension));
            process_series
                .borrow_mut()
                .add_process(Rc::new(Task::SetOrigRawStackExtension));
            if !setup_recon_ui_harness.is_directive_driven_automation() {
                process_series
                    .borrow_mut()
                    .add_process(Rc::new(Task::RenameInputImageFiles));
            }
            // remove excluded views Axis A
            let views_to_exclude_a =
                setup_recon_ui_harness.get_views_to_skip(axis_id, do_validation);
            remove_excluded_views_a = remove_excluded_views_a
                && views_to_exclude_a.as_deref().is_some_and(|views| {
                    !const_etomo_number::java_lang_string_matches_whitespace(views)
                });
            if remove_excluded_views_a {
                if dual_axis {
                    axis_id = AxisID::First;
                }
                self.get_process_mgr().base.unblock_axis(axis_id);
                let param = self.update_exclude_views_param(
                    axis_id,
                    dataset_dir,
                    dual_axis,
                    views_to_exclude_a.as_deref(),
                    true,
                );
                process_series.borrow_mut().add_process_command_axis_type(
                    Rc::new(Task::ExcludeViewsA),
                    param,
                    Some(axis_type),
                );
            }
            // remove excluded views Axis B
            axis_id = AxisID::Second;
            let views_to_exclude_b =
                setup_recon_ui_harness.get_views_to_skip(axis_id, do_validation);
            remove_excluded_views_b = remove_excluded_views_b
                && dual_axis
                && views_to_exclude_b.as_deref().is_some_and(|views| {
                    !const_etomo_number::java_lang_string_matches_whitespace(views)
                });
            if remove_excluded_views_b {
                self.get_process_mgr().base.unblock_axis(axis_id);
                let param = self.update_exclude_views_param(
                    axis_id,
                    dataset_dir,
                    dual_axis,
                    views_to_exclude_b.as_deref(),
                    true,
                );
                process_series.borrow_mut().add_process_command_axis_type(
                    Rc::new(Task::ExcludeViewsB),
                    param,
                    Some(axis_type),
                );
            }
            // copytomocoms
            process_series
                .borrow_mut()
                .set_last_process_task(Rc::new(Task::CopyTomoComs));
        }
        ProcessSeries::start_next_process(&process_series, AxisID::Only);
    }

    /// Java private `validateDirectiveFiles(ProcessSeries, AxisID)`
    /// (ApplicationManager.java:638).  Runs validation on either the automation directive
    /// file or the three templates.  `process_series` is required and should contain a
    /// fail process.
    ///
    /// `setupReconUIHarness.getDirectiveFileCollection()` may return null in the
    /// translation (the harness's own NullPointerException fix,
    /// SetupReconUIHarness.java:599); a missing collection reads as a missing directive
    /// file, for which `updateBatchruntomo` returns null.
    fn validate_directive_files(
        &'static self,
        process_series: &ProcessSeriesHandle,
        axis_id: AxisID,
    ) {
        let setup_recon_ui_harness = self
            .setup_recon_ui_harness
            .get()
            .expect("validateDirectiveFiles: setupReconUIHarness exists while setup runs");
        // Automation
        let is_from_brt = etomo_director::ARGUMENTS.lock().unwrap().is_from_brt();
        if setup_recon_ui_harness.is_directive_driven_automation() && !is_from_brt {
            // Etomo is responsible for running the validation of the directive files.
            let directive_file = setup_recon_ui_harness
                .get_directive_file_collection()
                .and_then(|collection| {
                    collection
                        .borrow()
                        .get_directive_file(DirectiveFileType::Batch)
                });
            let param = self.update_batchruntomo(true, directive_file.as_deref(), axis_id);
            if let Some(mut param) = param
                && !self.get_process_mgr().batchruntomo(axis_id, &mut param)
            {
                ProcessSeries::start_fail_process(process_series, axis_id);
                return;
            }
        } else if !setup_recon_ui_harness.is_directive_driven_automation() {
            // Run validation for each template
            let directive_file = setup_recon_ui_harness
                .get_directive_file_collection()
                .and_then(|collection| {
                    collection
                        .borrow()
                        .get_directive_file(DirectiveFileType::Scope)
                });
            let param = self.update_batchruntomo(false, directive_file.as_deref(), axis_id);
            if let Some(mut param) = param
                && !self.get_process_mgr().batchruntomo(axis_id, &mut param)
            {
                ProcessSeries::start_fail_process(process_series, axis_id);
                return;
            }
            let directive_file = setup_recon_ui_harness
                .get_directive_file_collection()
                .and_then(|collection| {
                    collection
                        .borrow()
                        .get_directive_file(DirectiveFileType::System)
                });
            let param = self.update_batchruntomo(false, directive_file.as_deref(), axis_id);
            if let Some(mut param) = param
                && !self.get_process_mgr().batchruntomo(axis_id, &mut param)
            {
                ProcessSeries::start_fail_process(process_series, axis_id);
                return;
            }
            let directive_file = setup_recon_ui_harness
                .get_directive_file_collection()
                .and_then(|collection| {
                    collection
                        .borrow()
                        .get_directive_file(DirectiveFileType::User)
                });
            let param = self.update_batchruntomo(false, directive_file.as_deref(), axis_id);
            if let Some(mut param) = param
                && !self.get_process_mgr().batchruntomo(axis_id, &mut param)
            {
                ProcessSeries::start_fail_process(process_series, axis_id);
                return;
            }
        }
        ProcessSeries::start_next_process(process_series, axis_id);
    }

    /// Java private `renameInputImageFile(ProcessSeries, AxisID)`
    /// (ApplicationManager.java:675).  When etomo was started from batchruntomo nothing
    /// happens, and the series is neither continued nor ended - as in the source.
    fn rename_input_image_file(
        &'static self,
        process_series: &ProcessSeriesHandle,
        axis_id: AxisID,
    ) {
        let is_from_brt = etomo_director::ARGUMENTS.lock().unwrap().is_from_brt();
        if !is_from_brt {
            let param = self.update_batchruntomo_for_rename_input_image_files(axis_id);
            if let Some(mut param) = param
                && self.get_process_mgr().batchruntomo(axis_id, &mut param)
            {
                ProcessSeries::start_next_process(process_series, axis_id);
            } else {
                ProcessSeries::start_fail_process(process_series, axis_id);
            }
        }
    }

    /// Java private `setRawImageStackExtension(ProcessSeries, AxisID)`
    /// (ApplicationManager.java:688).
    fn set_raw_image_stack_extension_process_series_axis_id(
        &'static self,
        process_series: &ProcessSeriesHandle,
        axis_id: AxisID,
    ) {
        let setup_recon_ui_harness = self
            .setup_recon_ui_harness
            .get()
            .expect("setRawImageStackExtension: setupReconUIHarness exists while setup runs");
        self.set_raw_image_stack_extension_string(
            setup_recon_ui_harness.get_raw_image_stack().as_deref(),
        );
        ProcessSeries::start_next_process(process_series, axis_id);
    }

    /// Java private `setOrigRawImageStackExtension(ProcessSeries, AxisID)`
    /// (ApplicationManager.java:694).
    fn set_orig_raw_image_stack_extension(
        &'static self,
        process_series: Option<&ProcessSeriesHandle>,
        axis_id: AxisID,
    ) {
        let setup_recon_ui_harness = self
            .setup_recon_ui_harness
            .get()
            .expect("setOrigRawImageStackExtension: setupReconUIHarness exists while setup runs");
        let raw_image_stack_file_name = setup_recon_ui_harness.get_raw_image_stack();
        let Some(raw_image_stack_file_name) = raw_image_stack_file_name else {
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let extension = Extension::get_instance(&raw_image_stack_file_name);
        if extension.is_none_or(|extension| !extension.is_input_image_file()) {
            eprintln!("Error:  invalid extension for raw image stack: {raw_image_stack_file_name}");
            if let Some(process_series) = process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        self.get_meta_data()
            .set_orig_raw_image_stack_extension(extension);
        // Java calls `processSeries.startNextProcess(axisID)` unguarded after testing the
        // series for null twice above; the only caller (startNextProcess) always passes
        // one.
        if let Some(process_series) = process_series {
            ProcessSeries::start_next_process(process_series, axis_id);
        }
    }

    /// Java `setRawImageStackExtension(String)` (ApplicationManager.java:720).  Needs to
    /// be usable during setup - before the param file is marked as loaded.
    pub fn set_raw_image_stack_extension_string(&self, raw_image_stack_file_name: Option<&str>) {
        let Some(raw_image_stack_file_name) = raw_image_stack_file_name else {
            return;
        };
        let extension = Extension::get_instance(raw_image_stack_file_name);
        if extension.is_none_or(|extension| !extension.is_input_image_file()) {
            eprintln!("Error:  invalid extension for raw image stack: {raw_image_stack_file_name}");
            return;
        }
        self.get_meta_data()
            .set_raw_image_stack_extension(extension);
    }

    /// Java private `excludeViews(AxisID, ProcessSeries, String, AxisType, String,
    /// Command)` (ApplicationManager.java:733).
    fn exclude_views(
        &'static self,
        mut axis_id: AxisID,
        process_series: &ProcessSeriesHandle,
        dataset_dir: Option<&str>,
        axis_type: Option<AxisType>,
        views_to_exclude: Option<&str>,
        param: Option<Arc<dyn Command + Send + Sync>>,
    ) {
        let _ = (dataset_dir, views_to_exclude);
        // Make sure that the post processing panel is open
        if !self.setup_recon_ui_harness.is_some() {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    "Setup Reconstruction dialog not open",
                    "Program logic error",
                    Some(AxisID::Only),
                )
            });
            ProcessSeries::start_fail_process(process_series, axis_id);
            return;
        }
        if axis_id != AxisID::Second {
            axis_id = if axis_type == Some(AxisType::DualAxis) {
                AxisID::First
            } else {
                AxisID::Only
            };
        }
        // force previews to exit
        let result = (|| -> Result<(), ImodError> {
            self.get_imod_manager()
                .quit_all(imod_manager::PREVIEW_KEY, Some(axis_id))?;
            self.release_file();
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodError::AxisType(except)) => eprintln!("{except:?}"),
            Err(ImodError::Io(e)) => eprintln!("{e:?}"),
            Err(ImodError::SystemProcess(e)) => eprintln!("{e:?}"),
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
        if let Some(setup_dialog_expert) = self.setup_dialog_expert.get() {
            setup_dialog_expert.show_progress_panel();
        }
        let Some(param) = param else {
            // Param is created and passed to ProcessSeries for this call, so it should not be
            // null.
            ProcessSeries::start_fail_process(process_series, axis_id);
            return;
        };
        // Raw tilt file is required for exclude views.
        if let Some(setup_dialog_expert) = self.setup_dialog_expert.get()
            && setup_dialog_expert.get_tilt_angle_type(axis_id) == Some(TiltAngleType::Range)
        {
            // The user wants to create or recreate the raw tilt file from a range. This
            // file is needed by excludeviews.
            if let Err(e) = self.make_rawtlt_file(axis_id) {
                // catch IOException, catch InvalidParameterException: both print.
                eprintln!("{e:?}");
            }
            if !file_type::CLASS
                .raw_tilt_angles
                .exists(Some(self), Some(axis_id))
            {
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!(
                            "Missing raw tilt file: {}",
                            file_type::CLASS
                                .raw_tilt_angles
                                .get_file_name(Some(self), Some(axis_id))
                                .as_deref()
                                .unwrap_or("null")
                        ),
                        "Unable to execute command",
                        Some(AxisID::Only),
                    )
                });
                ProcessSeries::start_fail_process(process_series, axis_id);
                return;
            }
        }
        let thread_name = match self.get_process_mgr().exclude_views(
            param,
            axis_id,
            Some(Arc::new(EdtRef::new(Rc::clone(process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    "Can not execute excludeviews command".to_owned(),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(AxisID::Only),
                    )
                });
                ProcessSeries::start_fail_process(process_series, axis_id);
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_process_can_be_killed(false, axis_id);
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(
                    &format!(
                        "Removing excluded views{}",
                        if axis_id == AxisID::First || axis_id == AxisID::Second {
                            format!(" for {axis_id} axis")
                        } else {
                            String::new()
                        }
                    )
                    .as_str(),
                ),
                axis_id,
                Some(&ProcessName::EXCLUDE_VIEWS),
            );
        }
    }

    /// Java private `updateExcludeViewsParam(AxisID, String, boolean, String, boolean)`
    /// (ApplicationManager.java:815).
    fn update_exclude_views_param(
        &'static self,
        axis_id: AxisID,
        dataset_dir: Option<&str>,
        dual_axis: bool,
        views_to_exclude: Option<&str>,
        do_validation: bool,
    ) -> Option<Arc<dyn Command + Send + Sync>> {
        let setup_recon_ui_harness = self.setup_recon_ui_harness.get()?;
        // Get the metadata excludeviews param.
        let mut param = ExcludeViewsParam::new(axis_id, dataset_dir);
        if !setup_recon_ui_harness.get_parameters(&mut param, axis_id, dual_axis, do_validation) {
            return None;
        }
        param.set_views_to_exclude(views_to_exclude);
        Some(Arc::new(param))
    }

    /// Java private `msgSetupReconFailed(ProcessSeries)` (ApplicationManager.java:829).
    fn msg_setup_recon_failed(&'static self, process_series: Option<&ProcessSeriesHandle>) {
        if let Some(setup_dialog_expert) = self.setup_dialog_expert.get() {
            setup_dialog_expert.msg_setup_recon_failed();
        }
        if let Some(process_series) = process_series {
            process_series.borrow().end_series();
        }
        self.end_automation(false);
    }

    /// Java `endAutomation(boolean)` (ApplicationManager.java:839).
    pub fn end_automation(&self, success: bool) {
        let (is_directive, is_exit) = {
            let arguments = etomo_director::ARGUMENTS.lock().unwrap();
            (arguments.is_directive(), arguments.is_exit())
        };
        if is_directive && is_exit {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.exit(Some(AxisID::Only), if success { 0 } else { 1 })
            });
        }
    }

    /// Java private `copyTomoComs(ProcessSeries)` (ApplicationManager.java:854).
    fn copy_tomo_coms(&'static self, process_series: Option<&ProcessSeriesHandle>) -> bool {
        let setup_recon_ui_harness = self
            .setup_recon_ui_harness
            .get()
            .expect("copyTomoComs: setupReconUIHarness exists while setup runs");
        setup_recon_ui_harness.get_fields_meta_data(self.get_meta_data(), true);
        // copytomocoms
        let mut param = self.update_copytomocoms();
        // Run copytomocoms on the command line
        let messages = self.get_process_mgr().setup_com_scripts(
            AxisID::Only,
            &mut param,
            Some(self.get_meta_data().get_axis_type()),
        );
        let Some(messages) = messages else {
            if let Some(setup_dialog_expert) = self.setup_dialog_expert.get() {
                setup_dialog_expert.update_display(true);
            }
            if let Some(process_series) = process_series {
                ProcessSeries::start_fail_process(process_series, AxisID::Only);
            }
            // Java `mainPanel.stopProgressBar(null)`: MainPanel maps a null axis to the A
            // (only) panel.  Skipped when headless (see the head of this part).
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.stop_progress_bar_axis_id(AxisID::Only);
            }
            return false;
        };
        // Send a specific INFO: message to the project log
        if !messages.is_empty(Some(MessageType::Info)) {
            let info_messages = messages.match_messages(
                MessageType::Info,
                &[
                    "Setting logarithm offset",
                    "Pixel spacing",
                    "Setting up dual axis set without B stack",
                ],
            );
            if let Some(info_messages) = &info_messages
                && info_messages.len() != 0
            {
                self.log_message_list(
                    Some(info_messages),
                    Some("Copytomocoms"),
                    Some(AxisID::Only),
                );
            }
        }
        // Create the .rawtlt file if the angle type is range. This makes it
        // easy to display titl angles in 3dmod.
        if self.get_meta_data().get_tilt_angle_spec_a().get_type() == TiltAngleType::Range {
            let axis_type = self.get_meta_data().get_axis_type();
            let mut result = self.make_rawtlt_file(if axis_type == AxisType::DualAxis {
                AxisID::First
            } else {
                AxisID::Only
            });
            if result.is_ok() && axis_type == AxisType::DualAxis {
                result = self.make_rawtlt_file(AxisID::Second);
            }
            if let Err(e) = result {
                // catch IOException, catch InvalidParameterException: both print.
                eprintln!("{e:?}");
            }
        }
        self.complete_setup_state();
        if let Some(setup_dialog_expert) = self.setup_dialog_expert.get() {
            setup_dialog_expert.update_display(true);
        }
        self.close_setup_dialog();
        self.end_automation(true);
        // Java `mainPanel.stopProgressBar(null)`; see above.
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id(AxisID::Only);
        }
        // Java `processSeries.endSeries()` unguarded; the caller always passes a series.
        if let Some(process_series) = process_series {
            process_series.borrow().end_series();
        }
        true
    }

    /// Java private `completeSetupState()` (ApplicationManager.java:908).
    fn complete_setup_state(&'static self) {
        self.get_recon_process_track()
            .set_setup_state(ProcessState::Complete);
        self.get_meta_data().set_com_script_created(true);
        let dataset_name = self.get_meta_data().get_dataset_name();
        let _ = etomo_director::INSTANCE.rename_current_manager(dataset_name);
        self.close_imods(
            Some(imod_manager::PREVIEW_KEY),
            AxisID::First,
            Some("Axis A preview stack"),
        );
        self.close_imods(
            Some(imod_manager::PREVIEW_KEY),
            AxisID::Second,
            Some("Axis B preview stack"),
        );
    }

    /// Java private `closeSetupDialog()` (ApplicationManager.java:916).
    fn close_setup_dialog(&'static self) {
        if let Some(setup_dialog_expert) = self.setup_dialog_expert.get() {
            setup_dialog_expert.set_displayed(false);
        }
        // Switch the main window to the procesing panel
        self.open_processing_panel();
        // Free the dialog
        self.setup_recon_ui_harness
            .get()
            .expect("closeSetupDialog: setupReconUIHarness exists while setup runs")
            .free_dialog();
        self.save_storables(Some(AxisID::Only));
        if self.setup_dialog_expert.is_some() {
            self.setup_dialog_expert.set(None);
        }
    }

    /// Java package-private `updateCopytomocoms()` (ApplicationManager.java:935).  Setup
    /// the copytomocoms command.  Create a comscript if this is directive-driven
    /// automation.
    fn update_copytomocoms(&'static self) -> CopyTomoComs {
        let setup_recon_ui_harness = self
            .setup_recon_ui_harness
            .get()
            .expect("updateCopytomocoms: setupReconUIHarness exists while setup runs");
        let is_create = etomo_director::ARGUMENTS.lock().unwrap().is_create();
        let mut param = CopyTomoComs::new(
            self,
            is_create,
            setup_recon_ui_harness.is_directive_driven_automation(),
        );
        param.set_directive_file_collection(setup_recon_ui_harness.get_directive_file_collection());
        param
    }

    /// Java private `copyDirectiveFiles()` (ApplicationManager.java:947).  Copy the
    /// directive and template files chosen by the user during setup.  If a type of
    /// template was not chosen, delete the corresponding type in the dataset.
    fn copy_directive_files(&'static self) {
        let setup_recon_ui_harness = self
            .setup_recon_ui_harness
            .get()
            .expect("copyDirectiveFiles: setupReconUIHarness exists while setup runs");
        // `getDirectiveFileCollection()` returns null in the translation where Java throws
        // NullPointerException (SetupReconUIHarness.java:599, fixed there); with no
        // collection there is nothing to copy and nothing is deleted.
        let Some(directive_file_collection) =
            setup_recon_ui_harness.get_directive_file_collection()
        else {
            return;
        };
        let mut directive_file: Option<Arc<DirectiveFile>> = None;
        let mut template: &Arc<file_type::FileType>;
        let mut file: Option<PathBuf>;
        // try
        let result: Result<(), LogFileError> = 'try_block: {
            template = &file_type::CLASS.local_scope_template;
            directive_file = directive_file_collection
                .borrow()
                .get_directive_file(DirectiveFileType::Scope);
            if let Some(directive_file) = &directive_file {
                file = directive_file.get_file();
                if let Some(file) = &file
                    && let Err(e) = utilities::copy_file_to_file_type(
                        Some(file),
                        template,
                        Some(self),
                        Some(AxisID::Only),
                        false,
                        false,
                        false,
                    )
                {
                    break 'try_block Err(e);
                }
            } else {
                let _ = utilities::delete_file_type(self, Some(AxisID::Only), template);
            }
            template = &file_type::CLASS.local_system_template;
            directive_file = directive_file_collection
                .borrow()
                .get_directive_file(DirectiveFileType::System);
            if let Some(directive_file) = &directive_file {
                file = directive_file.get_file();
                if let Some(file) = &file
                    && let Err(e) = utilities::copy_file_to_file_type(
                        Some(file),
                        template,
                        Some(self),
                        Some(AxisID::Only),
                        true,
                        false,
                        false,
                    )
                {
                    break 'try_block Err(e);
                }
            } else {
                let _ = utilities::delete_file_type(self, Some(AxisID::Only), template);
            }
            template = &file_type::CLASS.local_user_template;
            directive_file = directive_file_collection
                .borrow()
                .get_directive_file(DirectiveFileType::User);
            if let Some(directive_file) = &directive_file {
                file = directive_file.get_file();
                if let Some(file) = &file
                    && let Err(e) = utilities::copy_file_to_file_type(
                        Some(file),
                        template,
                        Some(self),
                        Some(AxisID::Only),
                        true,
                        false,
                        false,
                    )
                {
                    break 'try_block Err(e);
                }
            } else {
                let _ = utilities::delete_file_type(self, Some(AxisID::Only), template);
            }
            template = &file_type::CLASS.local_batch_directive_file;
            directive_file = directive_file_collection
                .borrow()
                .get_directive_file(DirectiveFileType::Batch);
            if let Some(directive_file) = &directive_file {
                file = directive_file.get_file();
                if let Some(file) = &file
                    && let Err(e) = utilities::copy_file_to_file_type(
                        Some(file),
                        template,
                        Some(self),
                        Some(AxisID::Only),
                        true,
                        false,
                        false,
                    )
                {
                    break 'try_block Err(e);
                }
            } else {
                let _ = utilities::delete_file_type(self, Some(AxisID::Only), template);
            }
            Ok(())
        };
        let _ = template;
        // catch (final IOException | LogFileException | LockException e)
        if let Err(e) = result {
            eprintln!("{e:?}");
            if let Some(directive_file) = &directive_file {
                file = directive_file.get_file();
                if let Some(file) = &file {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string(
                            Some(self),
                            &format!(
                                "Unable to copy {} to dataset.",
                                utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                            ),
                            "Unable to Copy File",
                        )
                    });
                } else {
                    eprintln!("Warning: in copyDirectiveFiles directive file was not set.");
                }
            }
        }
    }

    /// Java `setupCtfCorrectionComScript(AxisID)` (ApplicationManager.java:1016).
    pub fn setup_ctf_correction_com_script(&self, axis_id: AxisID) {
        self.get_process_mgr()
            .setup_ctf_correction_com_script(axis_id);
    }

    /// Java `setupCtfPlotterComScript(AxisID, CtfPhaseFlipParam)`
    /// (ApplicationManager.java:1020).
    pub fn setup_ctf_plotter_com_script(
        &self,
        axis_id: AxisID,
        ctf_phase_flip_param: &CtfPhaseFlipParam,
    ) {
        self.get_process_mgr()
            .setup_ctf_plotter_com_script(ctf_phase_flip_param, axis_id);
    }

    /// Java private `openProcessingPanel()` (ApplicationManager.java:1033).  Open the
    /// main window in processing mode MUST run reconnect for all axis.
    fn open_processing_panel(&'static self) {
        let Some(main_panel) = self.main_panel.get() else {
            return;
        };
        main_panel.show_processing_panel(self.get_meta_data().get_axis_type());
        main_panel.update_all_processing_states(self.get_recon_process_track());
        self.set_panel();
        if self.get_meta_data().get_axis_type() == AxisType::DualAxis {
            self.reconnect(
                Some(
                    self.get_axis_process_data()
                        .get_saved_process_data(AxisID::First),
                ),
                AxisID::First,
                false,
            );
            self.reconnect(
                Some(
                    self.get_axis_process_data()
                        .get_saved_process_data(AxisID::Second),
                ),
                AxisID::Second,
                false,
            );
        } else {
            self.reconnect(
                Some(
                    self.get_axis_process_data()
                        .get_saved_process_data(AxisID::Only),
                ),
                AxisID::Only,
                false,
            );
        }
    }

    /// Java private `isReconnectRun(AxisID)` (ApplicationManager.java:1049).  This class's
    /// own `reconnectRunA/B`, not BaseManager's private fields of the same name.
    fn is_reconnect_run(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return *self.reconnect_run_b.lock().unwrap();
        }
        *self.reconnect_run_a.lock().unwrap()
    }

    /// Java private `setReconnectRun(AxisID)` (ApplicationManager.java:1056).
    fn set_reconnect_run(&self, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.reconnect_run_b.lock().unwrap() = true;
        } else {
            *self.reconnect_run_a.lock().unwrap() = true;
        }
    }

    /// Java `reconnect(ProcessData, AxisID, boolean)` (ApplicationManager.java:1073).
    /// Attempts to reconnect to a currently running process.  Only run once per axis.
    /// Only attempts one reconnect.  Must run super.reconnect first.  Returns true if a
    /// reconnect was attempted.
    ///
    /// Not an override: BaseManager's `reconnect` takes a fourth parameter, and is called
    /// here as `BaseManager::reconnect`.
    pub fn reconnect(
        &'static self,
        process_data: Option<Arc<Mutex<ProcessData>>>,
        axis_id: AxisID,
        multi_line_messages: bool,
    ) -> bool {
        if BaseManager::reconnect(
            self,
            process_data.clone(),
            Some(axis_id),
            multi_line_messages,
            None,
        ) {
            return true;
        }
        if self.is_reconnect_run(axis_id) {
            return false;
        }
        self.set_reconnect_run(axis_id);
        let Some(process_data) = process_data else {
            return false;
        };
        // Each read of the shared ProcessData holds its lock for the statement only:
        // reconnectTilt below reaches the same object through getSavedProcessData.
        let process_name = process_data.lock().unwrap().get_process_name();
        if process_name == Some(ProcessName::TILT) {
            let is_on_different_host = process_data.lock().unwrap().is_on_different_host();
            if is_on_different_host
                && !self
                    .reconnect_to_different_host(Some(&Arc::clone(&process_data)), Some(axis_id))
            {
                return false;
            }
            let is_running = process_data.lock().unwrap().is_running();
            if is_running {
                let process_data_string = process_data.lock().unwrap().to_source_string();
                eprintln!("\nAttempting to reconnect in Axis {axis_id}\n{process_data_string}");
                // `(TomogramGenerationExpert) getUIExpert(TOMOGRAM_GENERATION, axisID)`:
                // getUIExpert creates the expert on first use and returns this field, so the
                // cast is the field read after the call.
                let _ = self.get_ui_expert(Some(DialogType::TomogramGeneration), axis_id);
                let tomogram_generation_expert = if axis_id == AxisID::Second {
                    self.tomogram_generation_expert_b.get()
                } else {
                    self.tomogram_generation_expert_a.get()
                }
                .expect("getUIExpert creates the TomogramGenerationExpert");
                // processData.getProcessName(), which is TILT here.
                if !tomogram_generation_expert.reconnect_tilt(ProcessName::TILT) {
                    eprintln!("\nReconnect in Axis{axis_id} failed");
                }
                return true;
            }
        }
        false
    }

    /// Java `reconnectTilt(AxisID, ProcessName, ProcessResultDisplay)`
    /// (ApplicationManager.java:1103).
    pub fn reconnect_tilt(
        &'static self,
        axis_id: AxisID,
        process_name: ProcessName,
        display: Option<ProcessResultDisplayHandle>,
    ) -> bool {
        let display: Option<ProcessResultDisplayRef> =
            display.map(|display| Arc::new(EdtRef::new(display)));
        // Java reads the process data into a local it never uses.
        let _process_data = self.get_process_mgr().base.get_process_data(axis_id);
        let ret = self
            .get_process_mgr()
            .reconnect_tilt(axis_id, display, None);
        self.set_thread_name(Some(&process_name.to_string()), Some(axis_id));
        ret
    }

    /// Java `openPreProcDialog(AxisID)` (ApplicationManager.java:1114).  Open the
    /// pre-processing dialog.
    pub fn open_pre_proc_dialog(&'static self, axis_id: AxisID) {
        // Check to see if the com files are present otherwise pop up a dialog
        // box informing the user to run the setup process
        if !UIExpertUtilities::INSTANCE.are_scripts_created(self, self.get_meta_data(), axis_id) {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(axis_id);
            }
            return;
        }
        let action_message =
            self.set_current_dialog_type(Some(DialogType::PreProcessing), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.select_button(axis_id, "Pre-processing");
        }
        // Java TODO: When a panel is overwriten by another should it be nulled and
        // closed or left and and reshown when needed?
        // Problem with stale data for align and tilt info since they are on
        // multiple panels
        // Check to see if the dialog panel is already open
        let pre_proc_dialog_a = self.pre_proc_dialog_a.get();
        let pre_proc_dialog_b = self.pre_proc_dialog_b.get();
        if self.show_if_exists(
            pre_proc_dialog_a.as_deref().map(|dialog| &**dialog),
            pre_proc_dialog_b.as_deref().map(|dialog| &**dialog),
            axis_id,
            action_message.as_deref(),
        ) {
            return;
        }
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("PreProcessingDialog"),
            Some(utilities::STARTED_STATUS),
        );
        let pre_proc_dialog = PreProcessingDialog::new(self, axis_id);
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("PreProcessingDialog"),
            Some(utilities::FINISHED_STATUS),
        );
        if axis_id == AxisID::Second {
            self.pre_proc_dialog_b
                .set(Some(Rc::clone(&pre_proc_dialog)));
        } else {
            self.pre_proc_dialog_a
                .set(Some(Rc::clone(&pre_proc_dialog)));
        }
        // Load the required ccderaser{|a|b}.com files
        // Fill in the parameters and set it to the appropriate state
        self.get_com_script_manager().load_eraser(axis_id);
        let ccd_eraser_param = self
            .get_com_script_manager()
            .get_ccd_eraser_param(axis_id, Some(ccd_eraser_param::Mode::XRays));
        pre_proc_dialog.set_ccd_eraser_params(&ccd_eraser_param);
        pre_proc_dialog.set_parameters(&**self.get_screen_state(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&pre_proc_dialog.get_container(), axis_id);
        }
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `donePreProcDialog(AxisID)` (ApplicationManager.java:1155).  Closes
    /// preprocessing dialog and deletes dialog.
    pub fn done_pre_proc_dialog(&'static self, axis_id: AxisID) {
        let pre_proc_dialog = if axis_id == AxisID::Second {
            self.pre_proc_dialog_b.get()
        } else {
            self.pre_proc_dialog_a.get()
        };
        // Java passes the field on unchecked; the dialog's own done button is the only
        // caller, so it exists.
        if let Some(pre_proc_dialog) = &pre_proc_dialog {
            self.save_pre_proc_dialog(pre_proc_dialog, axis_id);
        }
        // Clean up the existing dialog
        if axis_id == AxisID::Second {
            self.pre_proc_dialog_b.set(None);
        } else {
            self.pre_proc_dialog_a.set(None);
        }
        drop(pre_proc_dialog);
    }

    /// Java `savePreProcDialog(PreProcessingDialog, AxisID)`
    /// (ApplicationManager.java:1180).  Updates comscripts and edf file.
    pub fn save_pre_proc_dialog(
        &'static self,
        pre_proc_dialog: &Rc<PreProcessingDialog>,
        axis_id: AxisID,
    ) {
        self.set_advanced_dialog_type_axis_id_boolean(
            pre_proc_dialog.get_dialog_type(),
            axis_id,
            pre_proc_dialog.is_advanced(),
        );

        // Keep dialog box open until we get good info or it is cancelled
        let exit_state = pre_proc_dialog.get_exit_state();
        if exit_state == DialogExitState::Cancel {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(axis_id);
            }
        } else {
            if !self.is_exiting()
                && exit_state != DialogExitState::Postpone
                && self.get_state().is_use_fixed_stack_warning(axis_id)
            {
                // Only warn once.
                self.get_state().set_use_fixed_stack_warning(axis_id, false);
                // The use button wasn't pressed and the user is moving on to the next
                // dialog. Don't put this message in the log.
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        None,
                        &format!(
                            "To use the fixed stack go back to Pre-processing and press the \"{}\" button.",
                            PreProcessingDialog::get_use_fixed_stack_label()
                        ),
                        "Entry Warning",
                        Some(axis_id),
                    )
                });
            }
            pre_proc_dialog.get_parameters(&**self.get_screen_state(axis_id));
            self.update_eraser_com(
                &*pre_proc_dialog.get_ccd_eraser_display(),
                axis_id,
                false,
                false,
            );
            if exit_state == DialogExitState::Execute {
                self.get_recon_process_track()
                    .set_pre_processing_state(ProcessState::Complete, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_pre_processing_state(ProcessState::Complete, axis_id);
                }
                // Go to the coarse align dialog by default
                self.open_coarse_align_dialog(axis_id);
                // If there are raw stack imod processes open ask the user if they
                // should be closed.
                BaseManager::close_imod(
                    self,
                    Some(imod_manager::RAW_STACK_KEY),
                    Some(axis_id),
                    Some("raw stack"),
                    false,
                );
                BaseManager::close_imod(
                    self,
                    Some(imod_manager::ERASED_STACK_KEY),
                    Some(axis_id),
                    Some("fixed stack"),
                    false,
                );
            } else if exit_state != DialogExitState::Save {
                self.get_recon_process_track()
                    .set_pre_processing_state(ProcessState::InProgress, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_pre_processing_state(ProcessState::InProgress, axis_id);
                    // Go to the coarse align dialog by default
                    main_panel.show_blank_process(axis_id);
                }
            }
            self.save_storables(Some(axis_id));
        }
    }

    /// Java `imodManualErase(AxisID, Run3dmodMenuOptions, DialogType)`
    /// (ApplicationManager.java:1226).  Open 3dmod to create the manual erase model.
    pub fn imod_manual_erase(
        &'static self,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
    ) {
        let erase_model_name = format!(
            "{}{}.erase",
            self.get_meta_data().get_dataset_name(),
            axis_id.get_extension()
        );
        let result = (|| -> Result<(), ImodError> {
            self.get_imod_manager().set_interpolation(
                imod_manager::RAW_STACK_KEY,
                Some(axis_id),
                false,
            )?;
            if self.get_meta_data().get_view_type() == ViewType::Montage {
                self.get_imod_manager()
                    .set_montage_separation(imod_manager::RAW_STACK_KEY, Some(axis_id))?;
                self.get_imod_manager()
                    .set_piece_list_file_name_string_axis_id_string(
                        imod_manager::RAW_STACK_KEY,
                        Some(axis_id),
                        Some(&format!(
                            "{}{}.pl",
                            self.get_meta_data().get_dataset_name(),
                            axis_id.get_extension()
                        )),
                    )?;
            }
            self.get_imod_manager()
                .open_string_axis_id_string_boolean_run3dmod_menu_options(
                    imod_manager::RAW_STACK_KEY,
                    Some(axis_id),
                    Some(&erase_model_name),
                    true,
                    Some(menu_options),
                )?;
            self.get_recon_process_track().set_state_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_state_process_state_axis_id_dialog_type(
                    ProcessState::InProgress,
                    axis_id,
                    dialog_type,
                );
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.0,
                        "Cannot open 3dmod on raw stack",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "Axis type problem in 3dmod erase",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java private `updateEraserCom(CcdEraserDisplay, AxisID, boolean, boolean)`
    /// (ApplicationManager.java:1267).  Get the eraser script parameters from the CCD
    /// eraser panel and write them out the eraser{|a|b}.com script.  The Java return
    /// type is `Command`; the object is always the CCDEraserParam, which is what
    /// `ProcessManager::eraser` takes.
    fn update_eraser_com(
        &'static self,
        display: &dyn CcdEraserDisplay,
        axis_id: AxisID,
        trial_mode: bool,
        do_validation: bool,
    ) -> Option<Arc<CCDEraserParam>> {
        // Get the user input data from the dialog box. The CCDEraserParam
        // is first initialized from the currently loaded com script to
        // provide deafault values for those not handled by the dialog box
        // get function needs some error checking
        let mut ccd_eraser_param = self.get_com_script_manager().get_ccd_eraser_param(
            axis_id,
            Some(if trial_mode {
                ccd_eraser_param::Mode::XRaysTrial
            } else {
                ccd_eraser_param::Mode::XRays
            }),
        );
        if !display.get_parameters(&mut ccd_eraser_param, do_validation) {
            return None;
        }
        ccd_eraser_param.set_trial_mode(trial_mode);
        self.get_com_script_manager()
            .save_eraser(&ccd_eraser_param, axis_id);
        Some(Arc::new(ccd_eraser_param))
    }

    /// Java private `eraser(AxisID, ProcessResultDisplay, ProcessSeries, DialogType,
    /// CcdEraserDisplay)` (ApplicationManager.java:1288).  Run the eraser script for the
    /// specified axis.
    fn eraser(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: Option<DialogType>,
        display: &dyn CcdEraserDisplay,
    ) {
        let process_result_display: Option<ProcessResultDisplayRef> =
            process_result_display.map(|display| Arc::new(EdtRef::new(display)));
        let process_series: Option<ProcessSeriesRef> =
            process_series.map(|process_series| Arc::new(EdtRef::new(process_series)));
        let Some(param) = self.update_eraser_com(display, axis_id, false, true) else {
            return;
        };
        // Java's setState calls compare the (possibly null) dialog type with each
        // constant, so a null dialog type changes nothing.
        if let Some(dialog_type) = dialog_type {
            self.get_recon_process_track().set_state_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_state_process_state_axis_id_dialog_type(
                    ProcessState::InProgress,
                    axis_id,
                    dialog_type,
                );
            }
        }
        let thread_name = match self.get_process_mgr().eraser(
            axis_id,
            process_result_display,
            process_series,
            Some(param),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!("Can not execute eraser{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
    }

    /// Java `findXrays(AxisID, ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, CcdEraserDisplay)` (ApplicationManager.java:1317).
    /// Run CCDeraser in trial mode.
    pub fn find_xrays(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        display: &dyn CcdEraserDisplay,
    ) {
        let process_result_display: Option<ProcessResultDisplayRef> =
            process_result_display.map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("findXrays")),
        };
        self.send_msg_process_starting(process_result_display.as_ref());
        if self
            .update_eraser_com(display, axis_id, true, true)
            .is_none()
        {
            self.send_msg_process_failed_to_start(process_result_display.as_ref());
            process_series.borrow().end_series();
            return;
        }
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        // Trial find X-rays only creates a model, does not change the image file.
        let thread_name = match self.get_process_mgr().eraser(
            axis_id,
            process_result_display,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            None,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!("Can not execute eraser{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
    }

    /// Java `imodXrayModel(AxisID, Run3dmodMenuOptions)` (ApplicationManager.java:1353).
    /// Open 3dmod to view the xray model on the raw stack.
    pub fn imod_xray_model(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        let x_ray_model = format!(
            "{}{}_peak.mod",
            self.get_meta_data().get_dataset_name(),
            axis_id.get_extension()
        );
        let result = (|| -> Result<(), ImodError> {
            self.get_imod_manager().set_preserve_contrast(
                imod_manager::RAW_STACK_KEY,
                Some(axis_id),
                true,
            )?;
            if self.get_meta_data().get_view_type() == ViewType::Montage {
                self.get_imod_manager()
                    .set_montage_separation(imod_manager::RAW_STACK_KEY, Some(axis_id))?;
                self.get_imod_manager()
                    .set_piece_list_file_name_string_axis_id_string(
                        imod_manager::RAW_STACK_KEY,
                        Some(axis_id),
                        Some(&format!(
                            "{}{}.pl",
                            self.get_meta_data().get_dataset_name(),
                            axis_id.get_extension()
                        )),
                    )?;
            }
            self.get_imod_manager()
                .open_string_axis_id_string_run3dmod_menu_options(
                    imod_manager::RAW_STACK_KEY,
                    Some(axis_id),
                    Some(&x_ray_model),
                    Some(menu_options),
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.0,
                        "Problem opening coarse stack",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodErasedStack(AxisID, Run3dmodMenuOptions)` (ApplicationManager.java:1383).
    /// Open 3dmod to view the erased stack.
    pub fn imod_erased_stack(
        &'static self,
        axis_id: AxisID,
        run_3dmod_menu_options: Run3dmodMenuOptions,
    ) {
        if utilities::get_file_must_exist_file_type(
            self,
            true,
            axis_id,
            &file_type::CLASS.fixed_xrays_stack,
            "erased stack",
        )
        .is_none()
        {
            return;
        }
        let result = (|| -> Result<(), ImodError> {
            if self.get_meta_data().get_view_type() == ViewType::Montage {
                self.get_imod_manager()
                    .set_montage_separation(imod_manager::ERASED_STACK_KEY, Some(axis_id))?;
                self.get_imod_manager()
                    .set_piece_list_file_name_string_axis_id_string(
                        imod_manager::ERASED_STACK_KEY,
                        Some(axis_id),
                        Some(&format!(
                            "{}{}.pl",
                            self.get_meta_data().get_dataset_name(),
                            axis_id.get_extension()
                        )),
                    )?;
            }
            let tilt_file = dataset_files::get_raw_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                self.get_imod_manager().set_tilt_file(
                    imod_manager::ERASED_STACK_KEY,
                    Some(axis_id),
                    Some(&utilities::java_io_file_get_name(
                        &tilt_file.to_string_lossy(),
                    )),
                )?;
            } else {
                self.get_imod_manager()
                    .reset_tilt_file(imod_manager::ERASED_STACK_KEY, Some(axis_id))?;
            }
            self.get_imod_manager()
                .open_string_axis_id_run3dmod_menu_options(
                    imod_manager::ERASED_STACK_KEY,
                    Some(axis_id),
                    Some(run_3dmod_menu_options),
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.0,
                        "Problem opening erased stack",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `archiveOriginalStack(ProcessSeries, DialogType)`
    /// (ApplicationManager.java:1419).
    pub fn archive_original_stack_process_series_dialog_type(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: DialogType,
    ) {
        if self.get_process_mgr().base.in_use(AxisID::Only, None, true) {
            return;
        }
        self.archive_original_stack_axis_id_process_series_dialog_type(
            None,
            process_series,
            Some(dialog_type),
        );
    }

    /// Java private `archiveOriginalStack(AxisID, ProcessSeries, DialogType)`
    /// (ApplicationManager.java:1433).  Archive the original stacks during clean up.
    /// `current_axis_id` is the stack to archive.
    fn archive_original_stack_axis_id_process_series_dialog_type(
        &'static self,
        current_axis_id: Option<AxisID>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: Option<DialogType>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self,
                AxisID::First,
                dialog_type,
                Some("archiveOriginalStack"),
            ),
        };
        // figure out which original stack to archive
        let mut stack_axis_id = current_axis_id;
        if stack_axis_id.is_none() {
            if self.get_meta_data().get_axis_type() == AxisType::DualAxis {
                stack_axis_id = Some(AxisID::First);
            } else {
                stack_axis_id = Some(AxisID::Only);
            }
        }
        let stack_axis_id = stack_axis_id.unwrap();
        // set next process to archiveorig so that the second axis can be done
        if stack_axis_id == AxisID::First {
            process_series
                .borrow_mut()
                .set_next_process(Some(&archiveorig_param::command_name()), None);
        }
        // else {
        // resetNextProcess(AxisID.ONLY);
        // }
        // check for original stack
        let original_stack = utilities::get_file_must_exist_file_type(
            self,
            false,
            stack_axis_id,
            &file_type::CLASS.original_raw_stack,
            "original stack",
        );
        // mustExist is false, so the file is never null.
        if !original_stack.is_some_and(|original_stack| original_stack.exists()) {
            if stack_axis_id == AxisID::First {
                // Nothing to do on the first axis, so move on to the second axis
                ProcessSeries::start_next_process_display(&process_series, AxisID::Only, None);
                return;
            } else {
                process_series.borrow().end_series();
                return;
            }
        }
        // set progress bar and process state
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Archiving {stack_axis_id} stack").as_str()),
                AxisID::Only,
                Some(&ProcessName::ARCHIVEORIG),
            );
        }
        self.get_recon_process_track()
            .set_clean_up_state(ProcessState::InProgress);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_clean_up_state(ProcessState::InProgress);
        }
        // create param
        let param = ArchiveorigParam::new(self, stack_axis_id);
        // run process
        match self.get_process_mgr().archive_orig(
            Arc::new(param),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => self.set_thread_name(Some(&thread_name), Some(AxisID::Only)),
            Err(e) => {
                // resetNextProcess(AxisID.ONLY);
                eprintln!("{e:?}");
                let message = vec![
                    format!(
                        "Can not execute {} command",
                        archiveorig_param::command_name()
                    ),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(AxisID::Only),
                    )
                });
            }
        }
    }

    /// Java `deleteOriginalStack(Command, String[])` (ApplicationManager.java:1492).
    /// Called by `ProcessManager.postProcess` on the process thread.
    ///
    /// Fixed in translation (BUGS.md):
    /// * ApplicationManager.java:1533-1536 - the `throw new IllegalStateException` sits
    ///   inside the final `else`, so an error recorded by the first two branches (the
    ///   original stack is missing; no output) is dropped: with a missing stack Java goes
    ///   on to offer to delete a file that does not exist, and with a null output it
    ///   throws NullPointerException at `output.length`.  The error is now reported
    ///   whenever one was recorded.  Java's uncaught IllegalStateException ends the
    ///   method (and the process thread's postProcess) with a stack trace; the
    ///   translation prints the exception and returns, since a panic would take the
    ///   process thread down.
    /// * ApplicationManager.java:1538 - `new String[output.length - 2 + 3]` is too short
    ///   for an output of fewer than two lines (ArrayIndexOutOfBoundsException, or
    ///   NegativeArraySizeException for an empty one).  The message is built as a list,
    ///   which has exactly the source's lines for every longer output.
    pub fn delete_original_stack(
        &'static self,
        archiveorig_param: Option<Arc<dyn Command + Send + Sync>>,
        output: Option<Vec<String>>,
    ) {
        // Java casts `archiveorigParam.getCommandMode()` unchecked; postProcess passes the
        // archiveorig command it just matched by name.
        let Some(archiveorig_param) = archiveorig_param else {
            return;
        };
        let axis_id;
        let mode = archiveorig_param.get_command_mode();
        if equals_mode(mode, &archiveorig_param::Mode::AxisA) {
            axis_id = AxisID::First;
        } else if equals_mode(mode, &archiveorig_param::Mode::AxisB) {
            axis_id = AxisID::Second;
        } else if equals_mode(mode, &archiveorig_param::Mode::AxisOnly) {
            axis_id = AxisID::Only;
        } else {
            return;
        }
        // mustExist is false, so the file is never null.
        let original_stack = utilities::get_file_must_exist_file_type(
            self,
            false,
            axis_id,
            &file_type::CLASS.original_raw_stack,
            "",
        )
        .expect("Utilities.getFile with mustExist false never returns null");
        let err_tag = "Unexpected result from running archiveorig";
        let mut err_mess = String::new();
        if !original_stack.exists() {
            err_mess.push_str(err_tag);
            err_mess.push_str(" - original stack doesn't exist.");
        } else if output.is_none() {
            err_mess.push_str(err_tag);
            err_mess.push_str(" - no output returned.");
        } else {
            let output = output.as_deref().unwrap();
            let success_message = format!(
                "It is now safe to delete {}",
                utilities::java_io_file_get_name(&original_stack.to_string_lossy())
            );
            let mut i = 0;
            while i < output.len() {
                if output[i] == success_message {
                    break;
                }
                i += 1;
            }
            if i >= output.len() {
                err_mess.push_str(err_tag);
                err_mess.push_str(" - success message missing from output:");
                i = 0;
                while i < output.len() {
                    err_mess.push_str(&output[i]);
                    i += 1;
                }
            }
        }
        if !err_mess.is_empty() {
            eprintln!("java.lang.IllegalStateException: {err_mess}");
            return;
        }
        // Unreachable: a null output recorded an error above.
        let Some(output) = output else {
            return;
        };
        let original_stack_path =
            utilities::java_io_file_get_absolute_path(&original_stack.to_string_lossy());
        let mut message: Vec<String> = Vec::new();
        message.push(format!(
            "Result of {}:\n",
            archiveorig_param
                .get_command_line()
                .unwrap_or_else(|| "null".to_owned())
        ));
        let mut i = 2;
        while i < output.len() {
            message.push(format!("{}\n", output[i]));
            i += 1;
        }
        message.push("\n".to_owned());
        message.push(format!("Delete {original_stack_path}?"));
        // Posted: Java calls this on the process thread.  Java blocks the process thread
        // on the modal delete dialog, so the rest of the method runs on the EDT and the
        // process thread waits for it.
        invoke_and_wait(move || {
            if ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_delete_dialog(Some(self), &message, Some(AxisID::Only))
            }) {
                eprintln!("Deleting {original_stack_path}");
                let _ = std::fs::remove_file(&original_stack);
                if let Some(clean_up_dialog) = self.clean_up_dialog.get() {
                    clean_up_dialog.set_archive_fields();
                }
            }
            self.update_archive_display();
        });
    }

    /// Java `getArchiveInfo(AxisID)` (ApplicationManager.java:1556).
    pub fn get_archive_info(&'static self, axis_id: AxisID) -> Option<String> {
        let stack = utilities::get_file_must_exist_file_type(
            self,
            false,
            axis_id,
            &file_type::CLASS.raw_stack,
            "",
        );
        let original_stack = utilities::get_file_must_exist_file_type(
            self,
            false,
            axis_id,
            &file_type::CLASS.original_raw_stack,
            "",
        );
        let xray_stack =
            utilities::get_file_must_exist_extension(self, false, axis_id, "_xray.st.gz", "");
        // mustExist is false throughout, so none of the three is null and this
        // IllegalStateException is never thrown.
        if stack.is_none() && original_stack.is_none() || xray_stack.is_none() {
            panic!("Unable to get file information");
        }
        let stack = stack.expect("mustExist false");
        let original_stack = original_stack.expect("mustExist false");
        let xray_stack = xray_stack.expect("mustExist false");
        if stack.exists() && !original_stack.exists() && xray_stack.exists() {
            return Some(utilities::java_io_file_get_name(&stack.to_string_lossy()));
        }
        None
    }

    /// Java `updateArchiveDisplay()` (ApplicationManager.java:1570).
    pub fn update_archive_display(&'static self) {
        let Some(clean_up_dialog) = self.clean_up_dialog.get() else {
            return;
        };
        if self.get_meta_data().get_axis_type() == AxisType::SingleAxis {
            clean_up_dialog.update_archive_display(
                dataset_files::get_original_stack(self, Some(AxisID::Only))
                    .is_some_and(|original_stack| original_stack.exists()),
            );
        } else {
            clean_up_dialog.update_archive_display(
                dataset_files::get_original_stack(self, Some(AxisID::First))
                    .is_some_and(|original_stack| original_stack.exists())
                    || dataset_files::get_original_stack(self, Some(AxisID::Second))
                        .is_some_and(|original_stack| original_stack.exists()),
            );
        }
    }

    /// Java `replaceRawStack(AxisID, ProcessResultDisplay, DialogType)`
    /// (ApplicationManager.java:1590).  Replace the raw stack with the fixed stack created
    /// from eraser.
    pub fn replace_raw_stack(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        dialog_type: DialogType,
    ) {
        let process_result_display: Option<ProcessResultDisplayRef> =
            process_result_display.map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display.as_ref());
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_progress_bar_string_int_boolean_axis_id(
                Some("Using fixed stack"),
                1,
                false,
                axis_id,
            );
        }
        let fixed_xrays_file = file_type::CLASS
            .fixed_xrays_stack
            .get_file(Some(self), Some(axis_id))
            .unwrap_or_else(|| PathBuf::from("null"));
        if !fixed_xrays_file.exists() {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    "Unable to rename.  Fixed stack does not exist.",
                    "Entry Error",
                    Some(axis_id),
                )
            });
            self.send_msg_process_failed_to_start(process_result_display.as_ref());
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    Some(ProcessEndState::Failed),
                );
            }
            return;
        }
        if !utilities::is_valid_stack_file(&fixed_xrays_file, self, Some(axis_id)) {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &format!(
                        "{} is not a valid MRC file.",
                        utilities::java_io_file_get_name(&fixed_xrays_file.to_string_lossy())
                    ),
                    "Entry Error",
                    Some(axis_id),
                )
            });
            self.send_msg_process_failed_to_start(process_result_display.as_ref());
            return;
        }
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        // Rename the fixed stack to the raw stack file name and save the original
        // raw stack to _orig.st if that does not already exist
        // try {
        //   try {
        let inner_result = if !file_type::CLASS
            .original_raw_stack
            .get_file(Some(self), Some(axis_id))
            .is_some_and(|file| file.exists())
        {
            BaseManager::rename_image_file(
                self,
                Some(&*file_type::CLASS.raw_stack),
                Some(&*file_type::CLASS.original_raw_stack),
                Some(axis_id),
            )
        } else {
            Ok(())
        };
        let outer_result: Result<(), LogFileError> = match inner_result {
            //   } catch (final LockException e) {
            Err(LogFileError::Lock(_)) => {
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        Some(ProcessEndState::FileLockFailure),
                    );
                }
                return;
            }
            Err(e) => Err(e),
            Ok(()) => {
                //   try {
                let inner_result = match self.rename_xray_stack(axis_id) {
                    Ok(true) => BaseManager::rename_image_file(
                        self,
                        Some(&*file_type::CLASS.fixed_xrays_stack),
                        Some(&*file_type::CLASS.raw_stack),
                        Some(axis_id),
                    ),
                    Ok(false) => Ok(()),
                    Err(e) => Err(LogFileError::Lock(e)),
                };
                match inner_result {
                    //   } catch (final LockException e) {
                    Err(LogFileError::Lock(_)) => {
                        if let Some(main_panel) = self.main_panel.get() {
                            main_panel.stop_progress_bar_axis_id_process_end_state(
                                axis_id,
                                Some(ProcessEndState::FileLockFailure),
                            );
                        }
                        return;
                    }
                    Err(e) => Err(e),
                    Ok(()) => {
                        self.send_msg_process_succeeded(process_result_display.as_ref());
                        Ok(())
                    }
                }
            }
        };
        // } catch (IOException | LogFileException except) {
        if outer_result.is_err() {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    Some(ProcessEndState::FileLockFailure),
                );
            }
            self.send_msg_process_failed(process_result_display.as_ref());
            return;
        }
        self.get_state().set_use_fixed_stack_warning(axis_id, false);
        // An _orig.st file may have been created, so refresh the Clean Up dialog's
        // archive fields.
        if let Some(clean_up_dialog) = self.clean_up_dialog.get() {
            clean_up_dialog.set_archive_fields();
        }
        self.update_archive_display();
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id(axis_id);
        }
    }

    /// Java private `renameXrayStack(AxisID) throws LockException`
    /// (ApplicationManager.java:1655).  Renames the _xray.st.gz file to
    /// {dataset}{axisID}_xray.st.gz.{number}.  Numbers start at 1 and go up.  Returns
    /// false if rename fails.
    ///
    /// Fixed in translation (ApplicationManager.java:1676-1678, BUGS.md): Java throws an
    /// uncaught IllegalStateException when the archive name it chose already exists,
    /// abandoning replaceRawStack with its progress bar still running.  The translation
    /// reports the same message and returns false, the method's own "rename failed"
    /// result, so the fixed stack is not renamed over the raw stack.
    fn rename_xray_stack(&'static self, axis_id: AxisID) -> Result<bool, LockException> {
        let property_user_dir = self.get_property_user_dir().unwrap_or_default();
        let xray_stack = PathBuf::from(utilities::java_io_file_new(
            &property_user_dir,
            &format!(
                "{}{}_xray.st.gz",
                self.get_meta_data().get_dataset_name(),
                axis_id.get_extension()
            ),
        ));
        if !xray_stack.exists() {
            return Ok(true);
        }
        let user_dir = PathBuf::from(&property_user_dir);
        // Java `userDir.list(new XrayStackArchiveFilter())`: null when the directory
        // cannot be listed.
        let filter = XrayStackArchiveFilter::new();
        let xray_stack_archives: Option<Vec<String>> = match std::fs::read_dir(&user_dir) {
            Err(_) => None,
            Ok(entries) => {
                let mut names = Vec::new();
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if filter.accept(&user_dir, &name) {
                        names.push(name);
                    }
                }
                Some(names)
            }
        };
        let mut file_number = 0;
        if let Some(xray_stack_archives) = &xray_stack_archives {
            let mut new_file_number = EtomoNumber::new_with_type(Some(Type::Integer));
            for xray_stack_archive in xray_stack_archives {
                // `substring(lastIndexOf('.') + 1)`; -1 + 1 is the whole name.
                let start = xray_stack_archive.rfind('.').map_or(0, |index| index + 1);
                new_file_number.set_string(Some(&xray_stack_archive[start..]));
                if new_file_number.is_valid() && !new_file_number.is_null() {
                    file_number = file_number.max(new_file_number.get_int());
                }
            }
        }
        let xray_stack_archive = PathBuf::from(utilities::java_io_file_new(
            &property_user_dir,
            &format!(
                "{}{}_xray.st.gz.{}",
                self.get_meta_data().get_dataset_name(),
                axis_id.get_extension(),
                file_number + 1
            ),
        ));
        if xray_stack_archive.exists() {
            eprintln!(
                "java.lang.IllegalStateException: {} should not exist.",
                utilities::java_io_file_get_name(&xray_stack_archive.to_string_lossy())
            );
            return Ok(false);
        }
        match utilities::rename_file(
            Some(self),
            Some(axis_id),
            Some(&xray_stack),
            Some(&xray_stack_archive),
            false,
            false,
            false,
        ) {
            Ok(_) => {}
            // LockException is declared, not caught.
            Err(LogFileError::Lock(e)) => return Err(e),
            // catch (final IOException | LogFileException e)
            Err(e) => {
                eprintln!("{e:?}");
                let xray_stack_name =
                    utilities::java_io_file_get_name(&xray_stack.to_string_lossy());
                let xray_stack_archive_name =
                    utilities::java_io_file_get_name(&xray_stack_archive.to_string_lossy());
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!(
                            "{e}\nUnable to move {xray_stack_name} to {xray_stack_archive_name}.\nCannot continue.\nFirst run \"mv {xray_stack_name} {xray_stack_archive_name}\" from the command line."
                        ),
                        "Rename Failed",
                        Some(axis_id),
                    )
                });
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Java `openFilesInImod(AxisID, String, File[], Run3dmodMenuOptions)`
    /// (ApplicationManager.java:1696).
    pub fn open_files_in_imod_axis_id_string_file_array_run3dmod_menu_options(
        &'static self,
        axis_id: AxisID,
        key: &str,
        file_list: &[PathBuf],
        menu_options: Run3dmodMenuOptions,
    ) {
        let result = (|| -> Result<(), ImodError> {
            let preview_number = self.get_imod_manager().new_imod_string_axis_id_file_array(
                key,
                Some(axis_id),
                Some(file_list),
            )?;
            self.get_imod_manager()
                .open_string_axis_id_int_run3dmod_menu_options(
                    key,
                    Some(axis_id),
                    preview_number,
                    Some(menu_options),
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.0,
                        &format!("Problem opening {key}"),
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `openFilesInImod(AxisID, String, String[], String, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:1717).
    pub fn open_files_in_imod_axis_id_string_string_array_string_run3dmod_menu_options(
        &'static self,
        axis_id: AxisID,
        key: &str,
        file_name_array: &[String],
        subdir_name: Option<&str>,
        menu_options: Run3dmodMenuOptions,
    ) {
        let result = (|| -> Result<(), ImodError> {
            self.get_imod_manager()
                .open_axis_id_string_string_array_run3dmod_menu_options_string(
                    Some(axis_id),
                    key,
                    Some(file_name_array),
                    Some(menu_options),
                    subdir_name,
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.0,
                        &format!("Problem opening {key}"),
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `openEvenOddFilesInImod(AxisID, String, Run3dmodMenuOptions, boolean)`
    /// (ApplicationManager.java:1738).
    pub fn open_even_odd_files_in_imod(
        &'static self,
        axis_id: AxisID,
        key: &str,
        menu_options: Run3dmodMenuOptions,
        is_alt_tomo_trim_vol_checked: bool,
    ) {
        let result = (|| -> Result<(), ImodError> {
            self.get_imod_manager().set_swap_yz_string_axis_id_boolean(
                key,
                Some(axis_id),
                !is_alt_tomo_trim_vol_checked || !self.is_trimvol_flipped(),
            )?;
            self.get_imod_manager()
                .open_string_axis_id_run3dmod_menu_options(
                    key,
                    Some(axis_id),
                    Some(menu_options),
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.0,
                        &format!("Problem opening {key}"),
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodPreview(String, AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:1764).  Bring up a preview 3dmod.
    pub fn imod_preview(
        &'static self,
        file_extension: &str,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
    ) {
        let Some(setup_dialog_expert) = self.setup_dialog_expert.get() else {
            return;
        };
        let key = imod_manager::PREVIEW_KEY;
        // Java dereferences the result unchecked; with an open setup dialog the harness
        // always has a setup interface, so it is never null here.
        let Some(preview_meta_data) = self
            .setup_recon_ui_harness
            .get()
            .expect("imodPreview: setupReconUIHarness exists while the setup dialog is open")
            .get_meta_data()
        else {
            return;
        };
        self.get_imod_manager()
            .set_preview_meta_data(&preview_meta_data);
        let preview_working_dir = preview_meta_data.get_valid_dataset_directory(
            &setup_dialog_expert
                .get_working_directory()
                .map(|dir| utilities::java_io_file_get_absolute_path(&dir.to_string_lossy()))
                .unwrap_or_else(|| "null".to_owned()),
        );
        let Some(preview_working_dir) = preview_working_dir else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &BaseMetaData::base(&preview_meta_data).get_invalid_reason(),
                    "Raw Image Stack",
                    Some(axis_id),
                )
            });
            return;
        };
        let result = (|| -> Result<(), ImodError> {
            let preview_number = self.get_imod_manager().new_imod_string_string_axis_id(
                key,
                Some(file_extension),
                Some(axis_id),
            )?;
            self.get_imod_manager().set_working_directory(
                key,
                Some(axis_id),
                preview_number,
                Some(PathBuf::from(&preview_working_dir)),
            )?;
            if file_type::CLASS.piece_list.exists_with_meta_data(
                Some(self),
                Some(&preview_meta_data),
                Some(axis_id),
            ) {
                self.get_imod_manager()
                    .set_piece_list_file_name_string_axis_id_int_string(
                        key,
                        Some(axis_id),
                        preview_number,
                        file_type::CLASS
                            .piece_list
                            .get_file_name_with_meta_data(
                                Some(self),
                                Some(&preview_meta_data),
                                Some(axis_id),
                            )
                            .as_deref(),
                    )?;
            }
            self.get_imod_manager()
                .open_string_axis_id_int_run3dmod_menu_options(
                    key,
                    Some(axis_id),
                    preview_number,
                    Some(menu_options),
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.0,
                        "Problem opening raw stack",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `openCoarseAlignDialog(AxisID)` (ApplicationManager.java:1806).  Open the
    /// coarse alignment dialog.
    pub fn open_coarse_align_dialog(&'static self, axis_id: AxisID) {
        // Check to see if the com files are present otherwise pop up a dialog
        // box informing the user to run the setup process
        if !UIExpertUtilities::INSTANCE.are_scripts_created(self, self.get_meta_data(), axis_id) {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(axis_id);
            }
            return;
        }
        let action_message =
            self.set_current_dialog_type(Some(DialogType::CoarseAlignment), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.select_button(axis_id, "Coarse Alignment");
        }
        let coarse_align_dialog_a = self.coarse_align_dialog_a.get();
        let coarse_align_dialog_b = self.coarse_align_dialog_b.get();
        if self.show_if_exists(
            coarse_align_dialog_a.as_deref().map(|dialog| &**dialog),
            coarse_align_dialog_b.as_deref().map(|dialog| &**dialog),
            axis_id,
            action_message.as_deref(),
        ) {
            return;
        }
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("CoarseAlignDialog"),
            Some(utilities::STARTED_STATUS),
        );
        self.get_com_script_manager().load_xcorr(axis_id);
        let tiltxcorr_param: TiltxcorrParam =
            self.get_com_script_manager().get_tiltxcorr_param(axis_id);
        self.get_meta_data().set_orig_views_with_mag_changes(
            axis_id,
            !tiltxcorr_param.is_views_with_mag_changes_null(),
        );
        let coarse_align_dialog = CoarseAlignDialog::get_instance(
            self,
            axis_id,
            self.get_meta_data().is_orig_views_with_mag_changes(axis_id),
        );
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("CoarseAlignDialog"),
            Some(utilities::FINISHED_STATUS),
        );
        if axis_id == AxisID::Second {
            self.coarse_align_dialog_b
                .set(Some(Rc::clone(&coarse_align_dialog)));
        } else {
            self.coarse_align_dialog_a
                .set(Some(Rc::clone(&coarse_align_dialog)));
        }
        coarse_align_dialog.set_parameters_const_meta_data(self.get_meta_data());
        // Create the dialog box
        self.get_com_script_manager().load_undistort(axis_id);
        coarse_align_dialog.set_cross_correlation_params(&tiltxcorr_param);
        if self.get_meta_data().get_view_type() == ViewType::Montage {
            self.get_com_script_manager().load_preblend(axis_id);
        } else {
            self.get_com_script_manager().load_prenewst(axis_id);
        }

        if self.get_meta_data().get_view_type() == ViewType::Montage {
            let blendmont_param: BlendmontParam =
                self.get_com_script_manager().get_preblend_param(axis_id);
            coarse_align_dialog.set_params(&blendmont_param);
        } else {
            let prenewst_param = self.get_com_script_manager().get_prenewst_param(axis_id);
            coarse_align_dialog.set_prenewst_params(&prenewst_param);
        }

        coarse_align_dialog
            .set_fiducialess_alignment(self.get_meta_data().is_fiducialess_alignment(axis_id));
        coarse_align_dialog.set_image_rotation(Some(
            &self
                .get_meta_data()
                .get_image_rotation(axis_id)
                .to_string()
                .as_str(),
        ));
        coarse_align_dialog.set_parameters_recon_screen_state(self.get_screen_state(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&coarse_align_dialog.get_container(), axis_id);
        }
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `doneCoarseAlignDialog(AxisID)` (ApplicationManager.java:1865).  Get the
    /// parameters from the coarse align process dialog box.
    pub fn done_coarse_align_dialog(&'static self, axis_id: AxisID) {
        // Set a reference to the correct object
        let coarse_align_dialog = self.map_coarse_align_dialog(axis_id);
        // Java passes the field on unchecked; the dialog's own done button is the only
        // caller, so it exists.
        if let Some(coarse_align_dialog) = &coarse_align_dialog {
            self.save_coarse_align_dialog(coarse_align_dialog, axis_id);
        }
        // Clean up the existing dialog
        if axis_id == AxisID::Second {
            self.coarse_align_dialog_b.set(None);
        } else {
            self.coarse_align_dialog_a.set(None);
        }
        drop(coarse_align_dialog);
    }

    /// Java `saveCoarseAlignDialog(CoarseAlignDialog, AxisID)`
    /// (ApplicationManager.java:1882).  Get the parameters from the coarse align process
    /// dialog box.
    pub fn save_coarse_align_dialog(
        &'static self,
        coarse_align_dialog: &Rc<CoarseAlignDialog>,
        axis_id: AxisID,
    ) {
        self.set_advanced_dialog_type_axis_id_boolean(
            coarse_align_dialog.get_dialog_type(),
            axis_id,
            coarse_align_dialog.is_advanced(),
        );
        let exit_state = coarse_align_dialog.get_exit_state();
        if exit_state == DialogExitState::Cancel {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(axis_id);
            }
        } else {
            coarse_align_dialog.get_parameters_base_screen_state(self.get_screen_state(axis_id));
            // Get the user input data from the dialog box
            self.update_xcorr_com(
                &*coarse_align_dialog.get_tilt_xcorr_display(),
                axis_id,
                false,
                false,
                true,
            );
            self.update_blendmont_in_xcorr_com(axis_id);
            // try
            let result = if self.get_meta_data().get_view_type() != ViewType::Montage {
                coarse_align_dialog.get_parameters_meta_data(self.get_meta_data());
                self.update_prenewst_com(
                    &*coarse_align_dialog.get_newstack_display(),
                    axis_id,
                    false,
                    false,
                )
                .map(|_| ())
            } else {
                Ok(())
            };
            // catch (InvalidParameterException e), catch (IOException e): identical arms.
            if let Err(e) = result {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!("Unable to update prenewst com:  {e}"),
                        "Etomo Error",
                        Some(axis_id),
                    )
                });
            }
            UIExpertUtilities::INSTANCE
                .update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
                    self,
                    &**coarse_align_dialog,
                    axis_id,
                    false,
                );
            if exit_state == DialogExitState::Execute {
                self.get_recon_process_track()
                    .set_coarse_alignment_state(ProcessState::Complete, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_coarse_align_state(ProcessState::Complete, axis_id);
                }
                // Go to the fiducial model dialog by default
                if self.get_meta_data().is_fiducialess_alignment(axis_id) {
                    if let Some(ui_expert) =
                        self.get_ui_expert(Some(DialogType::TomogramPositioning), axis_id)
                    {
                        ui_expert.open_dialog();
                    }
                    // Check to see if the user wants to keep any coarse aligned imods
                    // open
                    BaseManager::close_imod(
                        self,
                        Some(imod_manager::COARSE_ALIGNED_KEY),
                        Some(axis_id),
                        Some("coarsely aligned stack"),
                        false,
                    );
                } else {
                    self.open_fiducial_model_dialog(axis_id);
                }
            } else if exit_state != DialogExitState::Save {
                self.get_recon_process_track()
                    .set_coarse_alignment_state(ProcessState::InProgress, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_coarse_align_state(ProcessState::InProgress, axis_id);
                    main_panel.show_blank_process(axis_id);
                }
            }
            self.save_storables(Some(axis_id));
        }
    }

    /// Java `tiltxcorr(AxisID, ProcessResultDisplay, Deferred3dmodButton,
    /// Run3dmodMenuOptions, ProcessSeries, DialogType, TiltXcorrDisplay, boolean,
    /// ProcessName, boolean, boolean)` (ApplicationManager.java:1940).  Get the
    /// parameters from the display and run the cross correlation script.
    #[allow(clippy::too_many_arguments)]
    pub fn tiltxcorr(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: Option<DialogType>,
        display: &dyn TiltXcorrDisplay,
        use_blendmont: bool,
        process_name: ProcessName,
        run_tiltxcorr: bool,
        break_contours: bool,
    ) {
        let process_result_display: Option<ProcessResultDisplayRef> =
            process_result_display.map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display.as_ref());
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, dialog_type, Some("tiltxcorr")),
        };
        // Get the parameters from the dialog box
        let tiltxcorr_param = self.update_xcorr_com(display, axis_id, true, true, run_tiltxcorr);
        let Some(tiltxcorr_param) = tiltxcorr_param else {
            self.send_msg_process_failed_to_start(process_result_display.as_ref());
            process_series.borrow().end_series();
            return;
        };
        // Java's setState calls compare the (possibly null) dialog type with each
        // constant, so a null dialog type changes nothing.
        if let Some(dialog_type) = dialog_type {
            self.get_recon_process_track().set_state_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_state_process_state_axis_id_dialog_type(
                    ProcessState::InProgress,
                    axis_id,
                    dialog_type,
                );
            }
        }
        let thread_name;
        // try
        let result = {
            let mut blendmont_param: Option<Arc<BlendmontParam>> = None;
            if use_blendmont {
                blendmont_param = self.update_blendmont_in_xcorr_com(axis_id).map(Arc::new);
            }
            match blendmont_param {
                None => {
                    // The xcorr.com file containing a blendmont command is going to run. If
                    // blendmont runs, this value will be turned back on.
                    self.get_state().set_xcorr_blendmont_was_run(axis_id, false);
                    self.get_process_mgr().tiltxcorr(
                        Arc::new(tiltxcorr_param),
                        process_name,
                        axis_id,
                        process_result_display.clone(),
                        Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
                        run_tiltxcorr,
                        break_contours,
                    )
                }
                Some(blendmont_param) => self.get_process_mgr().cross_correlate(
                    blendmont_param,
                    axis_id,
                    process_result_display.clone(),
                    Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
                ),
            }
        };
        match result {
            Ok(name) => thread_name = name,
            // catch (final AxisBusyException e)
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!("Can not execute xcorr{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return;
            }
        }
        if deferred_3dmod_button.is_some() {
            process_series
                .borrow_mut()
                .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        }
        self.set_thread_name(Some(&thread_name), Some(axis_id));
    }

    /// Java `cleanupAutofidseed(AxisID)` (ApplicationManager.java:1993).
    pub fn cleanup_autofidseed(&'static self, axis_id: AxisID) {
        if self.get_process_mgr().base.in_use(AxisID::Only, None, true) {
            return;
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Deleting temporary directory"),
                axis_id,
                Some(&ProcessName::AUTOFIDSEED),
            );
        }
        if !utilities::delete_file_type(self, Some(axis_id), &file_type::CLASS.autofidseed_dir) {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    Some(ProcessEndState::Failed),
                );
            }
        } else if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id(axis_id);
        }
    }

    /// Java private `processBatchRunTomoLog()` (ApplicationManager.java:2011).
    /// Sets process tracking button states from the batchruntomo log.
    fn process_batch_run_tomo_log(&'static self) {
        let log = BatchRunTomoLog::new();
        let Some(mut iterator) = log.seek(self, self.get_meta_data()) else {
            return;
        };
        while iterator.has_next() {
            let Some(process_signal) = iterator.next() else {
                continue;
            };
            let process_state = process_signal.get_process_state();
            let dialog_type = process_signal.get_dialog_type();
            let axis_id = process_signal.get_axis_id();
            // Java's setState compares a null dialog type with each constant and
            // changes nothing; a null state is never produced by the log reader, and a
            // null axis is not SECOND, so it reads as the A axis.
            let (Some(process_state), Some(dialog_type)) = (process_state, dialog_type) else {
                continue;
            };
            let axis_id_value = axis_id.unwrap_or(AxisID::First);
            if axis_id == Some(AxisID::Second) {
                self.get_recon_process_track().set_state_dialog_type(
                    process_state,
                    axis_id_value,
                    dialog_type,
                );
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_state_process_state_axis_id_dialog_type(
                        process_state,
                        axis_id_value,
                        dialog_type,
                    );
                }
            } else {
                self.get_recon_process_track().set_state_dialog_type(
                    process_state,
                    axis_id_value,
                    dialog_type,
                );
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_state_process_state_axis_id_dialog_type(
                        process_state,
                        axis_id_value,
                        dialog_type,
                    );
                }
            }
        }
        iterator.done();
    }

    /// Java `autofidseed(AxisID, ProcessResultDisplay, Deferred3dmodButton,
    /// Run3dmodMenuOptions, ProcessSeries, DialogType, FiducialModelDialog, boolean)`
    /// (ApplicationManager.java:2040).  Get the parameters from the display and run the
    /// autofidseed script.
    #[allow(clippy::too_many_arguments)]
    pub fn autofidseed(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: DialogType,
        display: &Rc<FiducialModelDialog>,
        just_find_shifts_near_zero: bool,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("autofidseed")),
        };
        // Get the parameters from the dialog box
        let Some(param) =
            self.update_autofidseed_com(display, axis_id, just_find_shifts_near_zero, true)
        else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        let bead_track_display = display.get_bead_track_display();
        if self
            .update_track_com(Some(&*bead_track_display), axis_id, true)
            .is_none()
        {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let thread_name = match self.get_process_mgr().autofidseed(
            Arc::new(param),
            axis_id,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!("Can not execute autofidseed{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed(process_result_display_ref.as_ref());
                return;
            }
        };
        if deferred_3dmod_button.is_some() {
            process_series
                .borrow_mut()
                .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        }
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Autofidseed"),
                axis_id,
                Some(&ProcessName::AUTOFIDSEED),
            );
        }
    }

    /// Java `preCrossCorrelate(AxisID, ProcessResultDisplay, ProcessSeries, DialogType,
    /// TiltXcorrDisplay)` (ApplicationManager.java:2092).  If the b stack hasn't been
    /// processed, run extracttilts before running crossCorrelate().
    pub fn pre_cross_correlate(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: DialogType,
        display: Rc<dyn TiltXcorrDisplay>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new_with_process_display(
                self,
                axis_id,
                Some(dialog_type),
                Some(Rc::clone(&display) as Rc<dyn ProcessDisplay>),
                Some("preCrossCorrelate"),
            ),
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        if self.get_meta_data().get_view_type() == ViewType::Montage
            && !dataset_tool::is_one_by(
                self.get_property_user_dir().as_deref(),
                file_type::CLASS
                    .raw_stack
                    .get_file_name(Some(self), Some(axis_id))
                    .as_deref(),
                self,
                axis_id,
            )
        {
            process_series
                .borrow_mut()
                .add_process(Rc::new(tomodataplots_param::Task::CoarseMeanMax));
        }
        if axis_id == AxisID::Second && self.process_b_stack() {
            process_series
                .borrow_mut()
                .set_last_process(Some(&ProcessName::XCORR.to_string()));
            self.extracttilts(axis_id, process_result_display, &process_series);
            return;
        }
        // Java passes a null Run3dmodMenuOptions with a null Deferred3dmodButton;
        // the options are never read without the button.
        self.tiltxcorr(
            axis_id,
            process_result_display,
            None,
            Run3dmodMenuOptions::default(),
            Some(process_series),
            Some(dialog_type),
            &*display,
            true,
            ProcessName::XCORR, /*was: FileType.CROSS_CORRELATION_COMSCRIPT*/
            true,
            false,
        );
    }

    /// Java `preEraser(AxisID, ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, CcdEraserDisplay)`
    /// (ApplicationManager.java:2121).  If the b stack hasn't been processed, run
    /// extracttilts before running eraser().
    #[allow(clippy::too_many_arguments)]
    pub fn pre_eraser(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        display: Rc<dyn CcdEraserDisplay>,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new_with_process_display(
                self,
                axis_id,
                Some(dialog_type),
                Some(Rc::clone(&display) as Rc<dyn ProcessDisplay>),
                Some("preEraser"),
            ),
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        if axis_id == AxisID::Second && self.process_b_stack() {
            process_series
                .borrow_mut()
                .set_last_process(Some(&ProcessName::ERASER.to_string()));
            self.extracttilts(axis_id, process_result_display, &process_series);
            return;
        }
        self.eraser(
            axis_id,
            process_result_display,
            Some(process_series),
            Some(dialog_type),
            &*display,
        );
    }

    /// Java private `processBStack()` (ApplicationManager.java:2139).
    fn process_b_stack(&'static self) -> bool {
        let axis_id = AxisID::Second;
        let meta_data = self.get_meta_data();
        let b_stack_processed = meta_data.get_b_stack_processed();
        if b_stack_processed.is_none() || b_stack_processed.as_ref().unwrap().is() {
            return false;
        }
        let b_stack = dataset_files::get_stack_with_dir(
            self,
            self.get_property_user_dir().as_deref(),
            Some(meta_data as &dyn BaseMetaData),
            Some(axis_id),
        );
        if !b_stack.exists() {
            return false;
        }
        self.get_com_script_manager().load_tilt(axis_id);
        let mut param = self.get_com_script_manager().get_tilt_param(axis_id);
        param.set_fiducialess(meta_data.is_fiducialess(axis_id));
        if meta_data.get_view_type() == ViewType::Montage {
            param.set_montage_full_image();
        } else {
            param.set_full_image(&b_stack);
        }
        UIExpertUtilities::INSTANCE.roll_tilt_com_angles(self, axis_id);
        self.get_com_script_manager().save_tilt(&param, axis_id);
        meta_data.set_fiducialess(axis_id, param.is_fiducialess());
        meta_data.set_b_stack_processed_boolean(true);
        self.save_storables(Some(axis_id));
        true
    }

    /// Java private `extracttilts(AxisID, ProcessResultDisplay, ProcessSeries)`
    /// (ApplicationManager.java:2166).
    ///
    /// `processSeries` is never null here: the first statement dereferences it, and
    /// both callers pass a series they have just created or checked, so the source's
    /// later `processSeries != null` tests are always true and are written as such.
    fn extracttilts(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: &ProcessSeriesHandle,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        process_series
            .borrow_mut()
            .set_next_process(Some(extractpieces_param::COMMAND_NAME), None);
        if self.get_meta_data().get_tilt_angle_spec_a().get_type() != TiltAngleType::Extract {
            ProcessSeries::start_next_process_display(
                process_series,
                axis_id,
                process_result_display_ref,
            );
            return;
        }
        let raw_tilt_file = dataset_files::get_raw_tilt(self, Some(axis_id));
        if raw_tilt_file.exists() {
            ProcessSeries::start_next_process_display(
                process_series,
                axis_id,
                process_result_display_ref,
            );
            return;
        }
        let thread_name = match self.get_process_mgr().extracttilts(
            axis_id,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!("Can not execute {}", extracttilts_param::COMMAND_NAME),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                ProcessSeries::start_next_process_display(
                    process_series,
                    axis_id,
                    process_result_display_ref,
                );
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", extracttilts_param::COMMAND_NAME).as_str()),
                axis_id,
                Some(&ProcessName::EXTRACTTILTS),
            );
        }
    }

    /// Java `extractpieces(AxisID, ProcessResultDisplay, ProcessSeries, DialogType,
    /// ViewType)` (ApplicationManager.java:2199).
    pub fn extractpieces(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: Option<DialogType>,
        view_type: ViewType,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, dialog_type, Some("extractpieces")),
        };
        process_series
            .borrow_mut()
            .set_next_process(Some(extractmagrad_param::COMMAND_NAME), None);
        // `processSeries != null` is always true from here on.
        if view_type != ViewType::Montage {
            ProcessSeries::start_next_process_display(
                &process_series,
                axis_id,
                process_result_display_ref,
            );
            return;
        }
        let piece_list_file = dataset_files::get_piece_list_file(self, Some(axis_id));
        if piece_list_file.exists() {
            ProcessSeries::start_next_process_display(
                &process_series,
                axis_id,
                process_result_display_ref,
            );
            return;
        }
        let thread_name = match self.get_process_mgr().extractpieces(
            axis_id,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!("Can not execute {}", extractpieces_param::COMMAND_NAME),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                ProcessSeries::start_next_process_display(
                    &process_series,
                    axis_id,
                    process_result_display_ref,
                );
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", extractpieces_param::COMMAND_NAME).as_str()),
                axis_id,
                Some(&ProcessName::EXTRACTPIECES),
            );
        }
    }

    /// Java private `extractmagrad(AxisID, ProcessResultDisplay, ProcessSeries)`
    /// (ApplicationManager.java:2235).
    ///
    /// `processSeries` is never null here: the only caller is `startNextProcess`
    /// (ApplicationManager.java:9744), which passes the running series.  The source's
    /// `processSeries != null` tests are therefore always true.
    fn extractmagrad(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: &ProcessSeriesHandle,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        let meta_data = self.get_meta_data();
        // `MetaData.getMagGradientFile` never returns null (a null or blank value is
        // ""), so the source's null test is always false and `processSeries != null`
        // is always true.
        let mag_gradient_file_name = meta_data.get_mag_gradient_file();
        if java_lang_string_matches_whitespace(&mag_gradient_file_name) {
            ProcessSeries::start_next_process_display(
                process_series,
                axis_id,
                process_result_display_ref,
            );
            return;
        }
        let mag_gradient_file = dataset_files::get_mag_gradient(self, Some(axis_id));
        if mag_gradient_file.exists() {
            ProcessSeries::start_next_process_display(
                process_series,
                axis_id,
                process_result_display_ref,
            );
            return;
        }
        let mut param = ExtractmagradParam::new(self, axis_id);
        let image_rotation = meta_data.get_image_rotation(axis_id);
        param.set_rotation_angle(Some(&*image_rotation));
        param.set_gradient_table(Some(&meta_data.get_mag_gradient_file()));
        let thread_name = match self.get_process_mgr().extractmagrad(
            Arc::new(param),
            axis_id,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!("Can not execute {}", extractmagrad_param::COMMAND_NAME),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                ProcessSeries::start_next_process_display(
                    process_series,
                    axis_id,
                    process_result_display_ref,
                );
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", extractmagrad_param::COMMAND_NAME).as_str()),
                axis_id,
                Some(&ProcessName::EXTRACTMAGRAD),
            );
        }
    }

    /// Java `makeDistortionCorrectedStack(AxisID, ProcessResultDisplay, ProcessSeries)`
    /// (ApplicationManager.java:2277).  Run undistort.com.
    pub fn make_distortion_corrected_stack(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.update_undistort_com(axis_id);
        self.get_recon_process_track()
            .set_coarse_alignment_state(ProcessState::InProgress, axis_id);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_coarse_align_state(ProcessState::InProgress, axis_id);
        }
        let thread_name = match self.get_process_mgr().make_distortion_corrected_stack(
            axis_id,
            process_result_display_ref,
            process_series.map(|process_series| Arc::new(EdtRef::new(process_series))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!("Can not execute undistort{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
    }

    /// Java `clipStats(AxisID, FileType, ProcessSeries, DialogType,
    /// TomodataplotsParam.Task, String)` (ApplicationManager.java:2299).
    // TEMP alignframes
    pub fn clip_stats(
        &'static self,
        axis_id: AxisID,
        input_file_type: &Arc<FileType>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: DialogType,
        task: tomodataplots_param::Task,
        tomodataplots_input_file_absolute_path: Option<&str>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("clipStats")),
        };
        let clip_param = ClipParam::get_stats_instance(
            self,
            axis_id,
            &input_file_type
                .get_file(Some(self), Some(axis_id))
                .unwrap_or_default(),
            std::path::Path::new(&self.get_property_user_dir().unwrap_or_default()),
        );
        let thread_name = match self.get_process_mgr().clip_stats(
            Arc::new(clip_param),
            axis_id,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(except) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!("Can't run clip stats\n{}", except.0),
                        "SystemProcessException",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        // TEMP alignframes
        process_series
            .borrow_mut()
            .set_next_process_task_parameter(Rc::new(task), tomodataplots_input_file_absolute_path);
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id(Some("clip stats "), axis_id);
        }
    }

    /// Java `coarseAlign(AxisID, ProcessResultDisplay, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions, DialogType, BlendmontDisplay,
    /// NewstackDisplay)` (ApplicationManager.java:2326).  Run the coarse alignment
    /// script.
    #[allow(clippy::too_many_arguments)]
    pub fn coarse_align(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        blendmont_display: &dyn BlendmontDisplay,
        newstack_display: &dyn NewstackDisplay,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("coarseAlign")),
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let process_name: ProcessName;
        let mut blendmont_param: Option<BlendmontParam> = None;
        let mut prenewst_param: Option<NewstParam> = None;
        if self.get_meta_data().get_view_type() == ViewType::Montage {
            match self.update_preblend_com(blendmont_display, axis_id, true, true) {
                Ok(Some(param)) => blendmont_param = Some(param),
                Ok(None) => {
                    process_series.borrow().end_series();
                    return;
                }
                // The source has three identical arms, for FortranInputSyntaxException,
                // InvalidParameterException and IOException.
                Err(e) => {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string(
                            Some(self),
                            &e.to_string(),
                            "Update Com Error",
                        )
                    });
                    // Upstream bug fixed (ApplicationManager.java:2339-2354): Java
                    // reports the error and carries on with a null blendmontParam, which
                    // `processMgr.preblend` then dereferences (NullPointerException on
                    // the event thread).  The series is ended here instead, as the
                    // `blendmontParam == null` path above does.
                    process_series.borrow().end_series();
                    return;
                }
            }
            process_name =
                BlendmontParam::get_process_name_for_mode(blendmont_param::Mode::Preblend);
        } else {
            match self.update_prenewst_com(newstack_display, axis_id, true, true) {
                Ok(param) => {
                    prenewst_param = param;
                    if prenewst_param.is_none() {
                        self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                        process_series.borrow().end_series();
                        return;
                    }
                    process_name = ProcessName::PRENEWST;
                }
                // The source has two identical arms, for InvalidParameterException and
                // IOException (updatePrenewstCom handles FortranInputSyntaxException
                // itself).
                Err(e) => {
                    eprintln!("{e:?}");
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self),
                            &format!("Unable to update prenewst com:  {e}"),
                            "Etomo Error",
                            Some(axis_id),
                        )
                    });
                    process_series.borrow().end_series();
                    return;
                }
            }
        }
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let process_series_ref: Option<ProcessSeriesRef> =
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series))));
        let result = if self.get_meta_data().get_view_type() == ViewType::Montage {
            self.get_process_mgr().preblend(
                Arc::new(blendmont_param.unwrap()),
                axis_id,
                process_result_display_ref,
                process_series_ref,
            )
        } else {
            self.get_process_mgr().coarse_align(
                Arc::new(prenewst_param.unwrap()),
                axis_id,
                process_result_display_ref,
                process_series_ref,
            )
        };
        let thread_name = match result {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!(
                        "Can not execute {}{}.com",
                        process_name,
                        axis_id.get_extension()
                    ),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        process_series
            .borrow_mut()
            .set_next_process(Some("checkUpdateFiducialModel"), None);
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        self.set_thread_name(Some(&thread_name), Some(axis_id));
    }

    /// Java `imodRawStack(AxisID, Run3dmodMenuOptions)` (ApplicationManager.java:2411).
    /// Open 3dmod to view the coarsely aligned stack.
    pub fn imod_raw_stack(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        let result = (|| -> Result<(), ImodManagerError> {
            let imod_manager = self.get_imod_manager();
            imod_manager.set_open_log_off(imod_manager::RAW_STACK_KEY, Some(axis_id))?;
            let tilt_file = dataset_files::get_raw_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                imod_manager.set_tilt_file(
                    imod_manager::RAW_STACK_KEY,
                    Some(axis_id),
                    tilt_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                imod_manager.reset_tilt_file(imod_manager::RAW_STACK_KEY, Some(axis_id))?;
            }
            imod_manager.open_string_axis_id_run3dmod_menu_options(
                imod_manager::RAW_STACK_KEY,
                Some(axis_id),
                Some(menu_options),
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "Problem opening raw stack",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodSortedModels(AxisID, Run3dmodMenuOptions, List<String>)`
    /// (ApplicationManager.java:2438).
    pub fn imod_sorted_models(
        &'static self,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
        model_name_list: &[String],
    ) {
        let result = self
            .get_imod_manager()
            .open_string_axis_id_list_run3dmod_menu_options(
                imod_manager::SORTED_MODELS_KEY,
                Some(axis_id),
                Some(model_name_list.to_vec()),
                Some(menu_options),
            );
        match result {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "Problem opening coarse stack",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodCoarseAlign(AxisID, Run3dmodMenuOptions, String, boolean)`
    /// (ApplicationManager.java:2461).  Open 3dmod to view the coarsely aligned stack.
    pub fn imod_coarse_align(
        &'static self,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
        model_name: Option<&str>,
        preserve_contrast: bool,
    ) {
        let result = (|| -> Result<(), ImodManagerError> {
            let imod_manager = self.get_imod_manager();
            imod_manager.set_open_log_off(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))?;
            let tilt_file = dataset_files::get_raw_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                imod_manager.set_tilt_file(
                    imod_manager::COARSE_ALIGNED_KEY,
                    Some(axis_id),
                    tilt_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                imod_manager.reset_tilt_file(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))?;
            }
            imod_manager.set_preserve_contrast(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                preserve_contrast,
            )?;
            if model_name.is_none() {
                imod_manager.open_string_axis_id_run3dmod_menu_options(
                    imod_manager::COARSE_ALIGNED_KEY,
                    Some(axis_id),
                    Some(menu_options),
                )?;
            } else {
                imod_manager.open_string_axis_id_string_boolean_run3dmod_menu_options(
                    imod_manager::COARSE_ALIGNED_KEY,
                    Some(axis_id),
                    model_name,
                    true,
                    Some(menu_options),
                )?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "Problem opening coarse stack",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `midasRawStack(AxisID, ProcessResultDisplay, CoarseAlignDisplay)`
    /// (ApplicationManager.java:2501).  Run midas on the raw stack.
    pub fn midas_raw_stack(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        display: &dyn CoarseAlignDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Upstream bug fixed (ApplicationManager.java:2504-2505): Java passes
        // `mapCoarseAlignDialog(axisID)` straight to `updateFiducialessParams`, which
        // dereferences it; with no coarse alignment dialog for the axis that is a
        // NullPointerException.  Here a missing dialog is the same early return as a
        // failed update.
        let Some(coarse_align_dialog) = self.map_coarse_align_dialog(axis_id) else {
            return;
        };
        if !UIExpertUtilities::INSTANCE
            .update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
                self,
                &*coarse_align_dialog,
                axis_id,
                true,
            )
        {
            return;
        }
        // processMgr.midasBlendStack(axisID,
        // metaData.getImageRotation(axisID).getDouble());
        // processMgr.midasRawStack(axisID, metaData.getImageRotation(axisID).getDouble());
        let mut param = MidasParam::new(self, axis_id, midas_param::Mode::RawStack);
        self.get_parameters(Some(&mut param), axis_id);
        display.get_coarse_align_parameters(&mut param);
        let param = Arc::new(param);
        match self
            .get_process_mgr()
            .base
            .midas(Arc::clone(&param) as Arc<dyn Command + Send + Sync>)
        {
            Ok(_) => {
                self.get_recon_process_track()
                    .set_coarse_alignment_state(ProcessState::InProgress, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_coarse_align_state(ProcessState::InProgress, axis_id);
                }
                self.send_msg_process_succeeded(process_result_display_ref.as_ref());
            }
            Err(_) => {
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        "Unable open midas on the raw stack.  ",
                        "Unable to Run Process",
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            }
        }
    }

    /// Java `midasFixEdges(AxisID, ProcessResultDisplay)` (ApplicationManager.java:2531).
    /// Run fix edges in Midas.
    pub fn midas_fix_edges(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let mut param = MidasParam::new(self, axis_id, midas_param::Mode::FixEdges);
        self.get_parameters(Some(&mut param), axis_id);
        let param = Arc::new(param);
        match self
            .get_process_mgr()
            .base
            .midas(Arc::clone(&param) as Arc<dyn Command + Send + Sync>)
        {
            Ok(_) => {
                self.get_recon_process_track()
                    .set_coarse_alignment_state(ProcessState::InProgress, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_coarse_align_state(ProcessState::InProgress, axis_id);
                }
                self.send_msg_process_succeeded(process_result_display_ref.as_ref());
            }
            Err(_) => {
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!(
                            "Unable open midas on {}.  ",
                            param.get_input_file_name().as_deref().unwrap_or("null")
                        ),
                        "Unable to Run Process",
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            }
        }
    }

    /// Java `getParameters(MidasParam, AxisID)` (ApplicationManager.java:2551).
    pub fn get_parameters(&'static self, param: Option<&mut MidasParam>, axis_id: AxisID) -> bool {
        let Some(param) = param else {
            return false;
        };
        let meta_data = self.get_meta_data();
        let mode = param.get_mode();
        if mode == midas_param::Mode::FixEdges {
            let file_type: &Arc<FileType> = if meta_data.is_distortion_correction() {
                &file_type::CLASS.distortion_corrected_stack
            } else {
                &file_type::CLASS.raw_stack
            };
            param.set_input_file_name(
                file_type
                    .get_file_name(Some(self), Some(axis_id))
                    .as_deref(),
            );
            param.set_binning(Some(Number::Integer(0)));
        } else if mode == midas_param::Mode::RawStack {
            if meta_data.get_view_type() == ViewType::Montage {
                param.set_input_file_name(
                    file_type::CLASS
                        .xcorr_blend_output
                        .get_file_name(Some(self), Some(axis_id))
                        .as_deref(),
                );
            } else {
                param.set_input_file_name(
                    file_type::CLASS
                        .raw_stack
                        .get_file_name(Some(self), Some(axis_id))
                        .as_deref(),
                );
            }
            param.set_image_rotation(meta_data.get_image_rotation(axis_id).get_double());
        }
        true
    }

    /// Java private `syncTiltxcorrParam(TiltxcorrParam, TiltxcorrParam, boolean)`
    /// (ApplicationManager.java:2586).  Keep xcorr.com and xcorr_pt.com up todate with
    /// each other by making sure that they have the same angleOffset.
    fn sync_tiltxcorr_param(
        &self,
        from_param: &TiltxcorrParam,
        to_param: &mut TiltxcorrParam,
        init: bool,
    ) {
        to_param.set_angle_offset(Some(&from_param.get_angle_offset()));
        to_param.set_tilt_angle_spec(from_param.get_tilt_angle_spec());
        if init {
            to_param.set_skip_views(Some(&from_param.get_skip_views()));
        }
    }

    /// Java private `updateXcorrCom(TiltXcorrDisplay, AxisID, boolean, boolean, boolean)`
    /// (ApplicationManager.java:2603).  Get the required parameters from the dialog box
    /// and update the xcorr.com script.  Handles xcorr.com and xcorr_pt.com.  Use
    /// syncTiltxcorrParam to keep the inactive .com file update to date.
    ///
    /// Returns the param if successful in getting the parameters and saving the com
    /// script, None if not successful.
    fn update_xcorr_com(
        &'static self,
        display: &dyn TiltXcorrDisplay,
        axis_id: AxisID,
        validate: bool,
        do_field_validation: bool,
        run_tiltxcorr: bool,
    ) -> Option<TiltxcorrParam> {
        let panel_id = display.get_panel_id();
        let result = (|| -> Result<Option<TiltxcorrParam>, FortranInputSyntaxException> {
            let mut tilt_xcorr_param: Option<TiltxcorrParam> = None;
            let mut to_param: Option<TiltxcorrParam> = None;
            if panel_id == PanelId::CrossCorrelation {
                tilt_xcorr_param = Some(self.get_com_script_manager().get_tiltxcorr_param(axis_id));
                if self.get_com_script_manager().load_xcorr_pt(axis_id, false) {
                    // xcorr_pt.com exists
                    to_param = Some(
                        self.get_com_script_manager()
                            .get_tiltxcorr_param_from_xcorr_pt(axis_id),
                    );
                }
            } else if panel_id == PanelId::PatchTracking {
                tilt_xcorr_param = Some(
                    self.get_com_script_manager()
                        .get_tiltxcorr_param_from_xcorr_pt(axis_id),
                );
                self.get_com_script_manager().load_xcorr(axis_id);
                to_param = Some(self.get_com_script_manager().get_tiltxcorr_param(axis_id));
            }
            // Upstream bug fixed (ApplicationManager.java:2622): for any other panel id
            // Java's `tiltXcorrParam` is still null and `setValidate` throws a
            // NullPointerException.  Here that is the method's failure return.
            let Some(mut tilt_xcorr_param) = tilt_xcorr_param else {
                return Ok(None);
            };
            tilt_xcorr_param.set_validate(validate);
            if !display.get_parameters(&mut tilt_xcorr_param, do_field_validation)? {
                return Ok(None);
            }
            if let Some(to_param) = to_param.as_mut() {
                self.sync_tiltxcorr_param(&tilt_xcorr_param, to_param, false);
            }
            if panel_id == PanelId::CrossCorrelation {
                self.get_com_script_manager()
                    .save_xcorr_tiltxcorr(&tilt_xcorr_param, axis_id);
                if let Some(to_param) = to_param.as_ref() {
                    self.get_com_script_manager()
                        .save_xcorr_pt_tiltxcorr(to_param, axis_id);
                }
            } else if panel_id == PanelId::PatchTracking {
                let mut imodchopconts_param = self
                    .get_com_script_manager()
                    .get_imodchopconts_param(axis_id);
                if !display
                    .get_parameters_imodchopconts(&mut imodchopconts_param, do_field_validation)
                {
                    return Ok(None);
                }
                self.get_com_script_manager()
                    .save_xcorr_pt_imodchopconts(&imodchopconts_param, axis_id);
                let goto_param = self
                    .get_com_script_manager()
                    .get_goto_param_from_xcorr_pt(axis_id, true);
                if let Some(mut goto_param) = goto_param {
                    if run_tiltxcorr {
                        goto_param.set_label(Some(tiltxcorr_param::GOTO_LABEL));
                    } else {
                        goto_param.set_label(Some(imodchopconts_param::GOTO_LABEL));
                    }
                    self.get_com_script_manager()
                        .save_xcorr_pt_goto(&goto_param, axis_id);
                }
                self.get_com_script_manager()
                    .save_xcorr_pt_tiltxcorr(&tilt_xcorr_param, axis_id);
                if let Some(to_param) = to_param.as_ref() {
                    self.get_com_script_manager()
                        .save_xcorr_tiltxcorr(to_param, axis_id);
                }
            }
            Ok(Some(tilt_xcorr_param))
        })();
        match result {
            Ok(tilt_xcorr_param) => tilt_xcorr_param,
            Err(except) => {
                eprintln!("{except:?}");
                let error_message = vec![
                    "Xcorr Parameter Syntax Error".to_string(),
                    except.get_message().unwrap_or("null").to_string(),
                    format!("New value: {}", except.get_new_string()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Xcorr Parameter Syntax Error",
                        Some(axis_id),
                    )
                });
                None
            } // Java also catches NumberFormatException ("Xcorr Align Parameter Syntax
              // Error", axisID.getExtension(), except.getMessage()).  It is unchecked and
              // the translated getters report no such error, so the arm has no Rust
              // counterpart.
        }
    }

    /// Java private `updateAutofidseedCom(FiducialModelDialog, AxisID, boolean, boolean)`
    /// (ApplicationManager.java:2687).  Get the required parameters from the dialog box
    /// and update the autofidseed.com script.  Returns the param if successful in
    /// getting the parameters and saving the com script, None if not successful.
    fn update_autofidseed_com(
        &'static self,
        display: &Rc<FiducialModelDialog>,
        axis_id: AxisID,
        just_find_shifts_near_zero: bool,
        do_validation: bool,
    ) -> Option<AutofidseedParam> {
        let mut param = self.get_com_script_manager().get_autofidseed_param(axis_id);
        self.get_com_script_manager()
            .load_autofidseed(axis_id, false);
        match display.get_parameters_autofidseed_param_boolean_boolean(
            &mut param,
            just_find_shifts_near_zero,
            do_validation,
        ) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(except) => {
                eprintln!("{except:?}");
                let error_message = vec![
                    "Autofidseed Parameter Syntax Error".to_string(),
                    axis_id.get_extension(),
                    except.get_message().unwrap_or("null").to_string(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Autofidseed Parameter Syntax Error",
                        Some(axis_id),
                    )
                });
                return None;
            }
        }
        // Java also catches NumberFormatException with the same message; it is
        // unchecked and has no Rust counterpart.
        self.get_com_script_manager()
            .save_autofidseed(&param, axis_id);
        Some(param)
    }

    /// Java private `updateBlendmontInXcorrCom(AxisID)` (ApplicationManager.java:2740).
    /// Update the blendmont command in the xcorr comscript.  If blendmont does not have
    /// to be run, update the goto param to skip blendmont; return blendmont param if
    /// blendmont has to be run.
    fn update_blendmont_in_xcorr_com(&'static self, axis_id: AxisID) -> Option<BlendmontParam> {
        let mut running_blendmont = false;
        let mut blendmont_param: Option<BlendmontParam> = None;
        // handle montaging
        if self.get_meta_data().get_view_type() == ViewType::Montage {
            let mut param = self
                .get_com_script_manager()
                .get_blendmont_param_from_tiltxcorr(axis_id);
            let goto_param = self
                .get_com_script_manager()
                .get_goto_param_from_tiltxcorr(axis_id);
            running_blendmont =
                param.set_blendmont_state(&self.get_state().get_invalid_edge_functions(axis_id));
            // Upstream bug fixed (ApplicationManager.java:2744-2755): Java dereferences
            // the goto param without a null check; an xcorr.com without a goto command
            // is a NullPointerException there.  Here the label update and its save are
            // skipped when there is no goto command.
            if let Some(mut goto_param) = goto_param {
                if running_blendmont {
                    goto_param.set_label(Some(blendmont_param::GOTO_LABEL));
                } else {
                    goto_param.set_label(Some(tiltxcorr_param::GOTO_LABEL));
                }
                self.get_com_script_manager()
                    .save_xcorr_goto(&goto_param, axis_id);
            }
            self.get_com_script_manager()
                .save_xcorr_blendmont(&param, axis_id);
            blendmont_param = Some(param);
        }
        if !running_blendmont {
            return None;
        }
        blendmont_param
    }

    /// Java private `updateUndistortCom(AxisID)` (ApplicationManager.java:2769).  Update
    /// undistort.com from xcorr.com.
    fn update_undistort_com(&'static self, axis_id: AxisID) {
        let mut blendmont_param = self
            .get_com_script_manager()
            .get_blendmont_param_from_tiltxcorr(axis_id);
        blendmont_param.set_mode(blendmont_param::Mode::Undistort);
        blendmont_param.set_blendmont_state(&self.get_state().get_invalid_edge_functions(axis_id));
        self.get_com_script_manager()
            .save_xcorr_to_undistort(&blendmont_param, axis_id);
    }

    /// Java private `updatePrenewstCom(NewstackDisplay, AxisID, boolean, boolean) throws
    /// InvalidParameterException, IOException` (ApplicationManager.java:2784).  Get the
    /// prenewst parameters from the dialog box and update the prenewst com script as
    /// well as the align com script since binning of the pre-aligned stack has an
    /// affect on the scaling of the fiducial model.
    ///
    /// The error is `NewstackDisplayException`, whose `FortranInputSyntaxException`
    /// variant this method handles itself (as the source's inner catch does), so only
    /// the `InvalidParameterException` and `IOException` variants reach the caller.
    fn update_prenewst_com(
        &'static self,
        display: &dyn NewstackDisplay,
        axis_id: AxisID,
        validate: bool,
        do_validation: bool,
    ) -> Result<Option<NewstParam>, NewstackDisplayException> {
        let mut prenewst_param = self.get_com_script_manager().get_prenewst_param(axis_id);
        prenewst_param.set_validate(validate);
        match display.get_parameters(&mut prenewst_param, do_validation) {
            Ok(true) => {}
            Ok(false) => return Ok(None),
            Err(NewstackDisplayException::FortranInputSyntaxException(except)) => {
                let error_message = vec![
                    "prenewst Parameter Syntax Error".to_string(),
                    format!("Axis: {}", axis_id.get_extension()),
                    except.get_message().unwrap_or("null").to_string(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Prenewst Parameter Syntax Error",
                        Some(axis_id),
                    )
                });
                return Ok(None);
            }
            Err(e) => return Err(e),
        }
        self.get_com_script_manager()
            .save_prenewst(&prenewst_param, axis_id);
        Ok(Some(prenewst_param))
    }

    /// Java private `updatePreblendCom(BlendmontDisplay, AxisID, boolean, boolean)
    /// throws FortranInputSyntaxException, InvalidParameterException, IOException`
    /// (ApplicationManager.java:2813).  Get the set the blendmont parameters and update
    /// the preblend com script.
    fn update_preblend_com(
        &'static self,
        display: &dyn BlendmontDisplay,
        axis_id: AxisID,
        validate: bool,
        do_validation: bool,
    ) -> Result<Option<BlendmontParam>, BlendmontDisplayException> {
        // `validate` is declared and unused in the source as well.
        let _ = validate;
        let mut preblend_param = self.get_com_script_manager().get_preblend_param(axis_id);
        if !display.get_parameters(&mut preblend_param, do_validation)? {
            return Ok(None);
        }
        preblend_param.set_blendmont_state(&self.get_state().get_invalid_edge_functions(axis_id));
        self.get_com_script_manager()
            .save_preblend(&preblend_param, axis_id);
        Ok(Some(preblend_param))
    }

    /// Java private `updateTiltCom(AltStackDisplay, AxisID)` (ApplicationManager.java:2825).
    fn update_tilt_com_alt_stack_display_axis_id(
        &'static self,
        display: &dyn AltStackDisplay,
        axis_id: AxisID,
    ) -> bool {
        let mut tilt_param = self.get_com_script_manager().get_tilt_param(axis_id);
        if !display.get_parameters_tilt_param(&mut tilt_param) {
            return false;
        }
        self.get_com_script_manager()
            .save_tilt(&tilt_param, axis_id);
        true
    }

    /// Java private `mapCoarseAlignDialog(AxisID)` (ApplicationManager.java:2840).
    /// Return the CoarseAlignDialog associated the specified AxisID.
    fn map_coarse_align_dialog(&self, axis_id: AxisID) -> Option<Rc<CoarseAlignDialog>> {
        if axis_id == AxisID::Second {
            return self.coarse_align_dialog_b.get();
        }
        self.coarse_align_dialog_a.get()
    }

    /// Java `openFiducialModelDialog(AxisID)` (ApplicationManager.java:2851).  Open the
    /// fiducial model generation dialog.
    pub fn open_fiducial_model_dialog(&'static self, axis_id: AxisID) {
        let meta_data = self.get_meta_data();
        // Check to see if the com files are present otherwise pop up a dialog
        // box informing the user to run the setup process
        if !UIExpertUtilities::INSTANCE.are_scripts_created(self, meta_data, axis_id) {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(axis_id);
            }
            return;
        }
        let action_message =
            self.set_current_dialog_type(Some(DialogType::FiducialModel), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.select_button(axis_id, "Fiducial Model Gen.");
        }
        let fiducial_model_dialog_a = self.fiducial_model_dialog_a.get();
        let fiducial_model_dialog_b = self.fiducial_model_dialog_b.get();
        if self.show_if_exists(
            fiducial_model_dialog_a.as_deref().map(|dialog| &**dialog),
            fiducial_model_dialog_b.as_deref().map(|dialog| &**dialog),
            axis_id,
            action_message.as_deref(),
        ) {
            return;
        }
        // Create a new dialog panel and map it the generic reference
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("FiducialModelDialog"),
            Some(utilities::STARTED_STATUS),
        );
        let fiducial_model_dialog =
            FiducialModelDialog::get_instance(self, axis_id, meta_data.get_axis_type());
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("FiducialModelDialog"),
            Some(utilities::FINISHED_STATUS),
        );
        if axis_id == AxisID::Second {
            self.fiducial_model_dialog_b
                .set(Some(Rc::clone(&fiducial_model_dialog)));
        } else {
            self.fiducial_model_dialog_a
                .set(Some(Rc::clone(&fiducial_model_dialog)));
        }
        self.update_dialog_fiducial_model_dialog_axis_id(
            Some(Rc::clone(&fiducial_model_dialog)),
            axis_id,
        );
        // Load the required track{|a|b}.com files, fill in the dialog box
        // params
        // and set it to the appropriate state
        self.get_com_script_manager().load_track(axis_id);
        // Create a default transferfid object to populate the alignment dialog
        fiducial_model_dialog.set_transfer_fid_params();
        let mut beadtrack_param = self.get_com_script_manager().get_beadtrack_param(axis_id);
        fiducial_model_dialog.set_beadtrack_params(&mut beadtrack_param, false);
        fiducial_model_dialog.set_parameters_const_meta_data(meta_data);
        // Try to load xcorr_pt comscript. If it isn't there, set it up and then
        // load. it.
        let mut tilt_xcorr_pt_param: Option<TiltxcorrParam> = None;
        if !self.get_com_script_manager().load_xcorr_pt(axis_id, false) {
            let mut makecomfile_param = MakecomfileParam::new(
                self,
                axis_id,
                Arc::clone(&file_type::CLASS.patch_tracking_comscript),
            );
            self.makecomfile(axis_id, &mut makecomfile_param);
            self.get_com_script_manager().load_xcorr_pt(axis_id, true);
            // Sync from xcorr.com to xcorr_pt.com
            self.get_com_script_manager().load_xcorr(axis_id);
            let mut pt_param = self
                .get_com_script_manager()
                .get_tiltxcorr_param_from_xcorr_pt(axis_id);
            let tilt_xcorr_param = self.get_com_script_manager().get_tiltxcorr_param(axis_id);
            self.sync_tiltxcorr_param(&tilt_xcorr_param, &mut pt_param, true);
            pt_param.set_partial_save(true);
            self.get_com_script_manager()
                .save_xcorr_pt_tiltxcorr(&pt_param, axis_id);
            tilt_xcorr_pt_param = Some(pt_param);
        }
        let mut imodchopconts_param: Option<ImodchopcontsParam> = None;
        if tilt_xcorr_pt_param.is_none() {
            let pt_param = self
                .get_com_script_manager()
                .get_tiltxcorr_param_from_xcorr_pt(axis_id);
            // Backwards compatibility
            let goto_param = self
                .get_com_script_manager()
                .get_goto_param_from_xcorr_pt(axis_id, false);
            if goto_param.is_none() {
                // This is an old version of xcorr_pt.com - create the new version.
                let _ = utilities::delete_file_type(
                    self,
                    Some(axis_id),
                    &file_type::CLASS.patch_tracking_comscript,
                );
                let mut makecomfile_param = MakecomfileParam::new(
                    self,
                    axis_id,
                    Arc::clone(&file_type::CLASS.patch_tracking_comscript),
                );
                self.makecomfile(axis_id, &mut makecomfile_param);
                self.get_com_script_manager().load_xcorr_pt(axis_id, true);
                self.get_com_script_manager()
                    .save_xcorr_pt_tiltxcorr(&pt_param, axis_id);
                // Transfer the old xcorr data that now belongs in the imodchopconts
                // command.
                let backward_param =
                    ImodchopcontsParam::get_backward_compatable_instance(&pt_param);
                self.get_com_script_manager()
                    .save_xcorr_pt_imodchopconts(&backward_param, axis_id);
                imodchopconts_param = Some(backward_param);
            }
            tilt_xcorr_pt_param = Some(pt_param);
        }
        let imodchopconts_param = match imodchopconts_param {
            Some(imodchopconts_param) => imodchopconts_param,
            None => self
                .get_com_script_manager()
                .get_imodchopconts_param(axis_id),
        };
        fiducial_model_dialog
            .set_parameters_const_tiltxcorr_param(tilt_xcorr_pt_param.as_ref().unwrap());
        fiducial_model_dialog.set_parameters_imodchopconts_param(&imodchopconts_param);
        // Autofidseed
        if !self
            .get_com_script_manager()
            .load_autofidseed(axis_id, false)
        {
            let mut makecom_file_param = MakecomfileParam::new(
                self,
                axis_id,
                Arc::clone(&file_type::CLASS.autofidseed_comscript),
            );
            self.makecomfile(axis_id, &mut makecom_file_param);
            self.get_com_script_manager()
                .load_autofidseed(axis_id, true);
        }
        let autofidseed_param = self.get_com_script_manager().get_autofidseed_param(axis_id);
        fiducial_model_dialog.set_parameters_autofidseed_param_boolean(&autofidseed_param, false);
        fiducial_model_dialog.set_parameters_recon_screen_state(self.get_screen_state(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&fiducial_model_dialog.get_container(), axis_id);
        }
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `doneFiducialModelDialog(AxisID)` (ApplicationManager.java:2943).  Save
    /// comscripts and the .edf file and delete the dialog.
    pub fn done_fiducial_model_dialog(&'static self, axis_id: AxisID) {
        // Set a reference to the correct object
        let fiducial_model_dialog = if axis_id == AxisID::Second {
            self.fiducial_model_dialog_b.get()
        } else {
            self.fiducial_model_dialog_a.get()
        };
        // Upstream bug fixed (ApplicationManager.java:2952): with no dialog for the
        // axis Java passes null to `saveFiducialModelDialog`, which dereferences it.
        // Here there is nothing to save and only the clean-up runs.
        if let Some(fiducial_model_dialog) = fiducial_model_dialog.as_ref() {
            self.save_fiducial_model_dialog(fiducial_model_dialog, axis_id);
        }
        // Clean up the existing dialog
        if axis_id == AxisID::Second {
            self.fiducial_model_dialog_b.set(None);
        } else {
            self.fiducial_model_dialog_a.set(None);
        }
        // `fiducialModelDialog = null;`
        drop(fiducial_model_dialog);
    }

    /// Java `saveFiducialModelDialog(FiducialModelDialog, AxisID)`
    /// (ApplicationManager.java:2966).  Save comscripts and the .edf file and delete
    /// the dialog.
    pub fn save_fiducial_model_dialog(
        &'static self,
        fiducial_model_dialog: &Rc<FiducialModelDialog>,
        axis_id: AxisID,
    ) {
        self.set_advanced_dialog_type_axis_id_boolean(
            fiducial_model_dialog.get_dialog_type(),
            axis_id,
            fiducial_model_dialog.is_advanced(),
        );
        let exit_state = fiducial_model_dialog.get_exit_state();
        if exit_state == DialogExitState::Cancel {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(axis_id);
            }
        } else {
            if !self.is_exiting()
                && exit_state != DialogExitState::Postpone
                && self.get_state().is_use_raptor_result_warning()
            {
                // The use button wasn't pressed and the user is moving on to the next
                // dialog. Don't put this message in the log.
                ui_harness::INSTANCE.with(|ui_harness| ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    None,
                    &format!(
                        "To use the RAPTOR result go back to Fiducial Model and press the \"{}\" button.",
                        FiducialModelDialog::get_use_raptor_result_label()
                    ),
                    "Entry Warning",
                    Some(axis_id),
                ));
                // Only warn once.
                self.get_state().set_use_raptor_result_warning(false);
            }
            fiducial_model_dialog.get_parameters_base_screen_state(self.get_screen_state(axis_id));
            fiducial_model_dialog.get_transfer_fid_params_boolean(false);
            fiducial_model_dialog.get_parameters_meta_data(self.get_meta_data());
            // Get the user input data from the dialog box
            let bead_track_display = fiducial_model_dialog.get_bead_track_display();
            self.update_track_com(Some(&*bead_track_display), axis_id, false);
            let tiltxcorr_display = fiducial_model_dialog.get_tiltxcorr_display();
            self.update_xcorr_com(&*tiltxcorr_display, axis_id, false, false, true);
            self.update_autofidseed_com(fiducial_model_dialog, axis_id, false, false);
            if exit_state == DialogExitState::Execute {
                self.get_recon_process_track()
                    .set_fiducial_model_state(ProcessState::Complete, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_fiducial_model_state(ProcessState::Complete, axis_id);
                }
                self.open_fine_alignment_dialog(axis_id);
            } else if exit_state != DialogExitState::Save {
                self.get_recon_process_track()
                    .set_fiducial_model_state(ProcessState::InProgress, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_fiducial_model_state(ProcessState::InProgress, axis_id);
                    main_panel.show_blank_process(axis_id);
                }
            }
            self.save_storables(Some(axis_id));
        }
    }

    /// Java `imodFindBeads3d(AxisID, Run3dmodMenuOptions, ProcessResultDisplay, String,
    /// String, File, DialogType)` (ApplicationManager.java:3011).  Open 3dmod in seed
    /// mode with model and tilt file.
    #[allow(clippy::too_many_arguments)]
    pub fn imod_find_beads3d(
        &'static self,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
        process_result_display: Option<ProcessResultDisplayHandle>,
        key: &str,
        model: Option<&str>,
        tilt_file: Option<&std::path::Path>,
        dialog_type: DialogType,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let result = (|| -> Result<(), ImodManagerError> {
            let imod_manager = self.get_imod_manager();
            imod_manager.set_preserve_contrast(key, Some(axis_id), true)?;
            imod_manager.set_open_bead_fixer(key, Some(axis_id), true)?;
            imod_manager.set_beadfixer_mode(key, Some(axis_id), Some(BeadFixerMode::SeedMode))?;
            imod_manager.set_open_log_off(key, Some(axis_id))?;
            imod_manager.set_delete_all_sections(key, Some(axis_id), true)?;
            if tilt_file.is_some() && tilt_file.unwrap().exists() {
                imod_manager.set_tilt_file(
                    key,
                    Some(axis_id),
                    tilt_file
                        .unwrap()
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                imod_manager.reset_tilt_file(key, Some(axis_id))?;
            }
            imod_manager.open_string_axis_id_string_boolean_run3dmod_menu_options(
                key,
                Some(axis_id),
                model,
                true,
                Some(menu_options),
            )?;
            self.get_recon_process_track().set_state_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_state_process_state_axis_id_dialog_type(
                    ProcessState::InProgress,
                    axis_id,
                    dialog_type,
                );
            }
            self.send_msg_process_succeeded(process_result_display_ref.as_ref());
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        &format!(
                            "Can't open 3dmod on {} with model: {}",
                            key,
                            model.unwrap_or("null")
                        ),
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodSeedModel(AxisID, Run3dmodMenuOptions, ProcessResultDisplay, String,
    /// String, File, DialogType)` (ApplicationManager.java:3053).  Open 3dmod in seed
    /// mode with model and tilt file.
    #[allow(clippy::too_many_arguments)]
    pub fn imod_seed_model(
        &'static self,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
        process_result_display: Option<ProcessResultDisplayHandle>,
        key: &str,
        model: Option<&str>,
        tilt_file: Option<&std::path::Path>,
        dialog_type: DialogType,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let result = (|| -> Result<(), ImodManagerError> {
            let imod_manager = self.get_imod_manager();
            imod_manager.set_open_contours(key, Some(axis_id), true)?;
            imod_manager.set_preserve_contrast(key, Some(axis_id), true)?;
            imod_manager.set_open_bead_fixer(key, Some(axis_id), true)?;
            imod_manager.set_auto_center(key, Some(axis_id), true)?;
            imod_manager.set_beadfixer_mode(key, Some(axis_id), Some(BeadFixerMode::SeedMode))?;
            imod_manager.set_open_log_off(key, Some(axis_id))?;
            if tilt_file.is_some() && tilt_file.unwrap().exists() {
                imod_manager.set_tilt_file(
                    key,
                    Some(axis_id),
                    tilt_file
                        .unwrap()
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                imod_manager.reset_tilt_file(key, Some(axis_id))?;
            }
            imod_manager.open_string_axis_id_string_boolean_run3dmod_menu_options(
                key,
                Some(axis_id),
                model,
                true,
                Some(menu_options),
            )?;
            self.get_recon_process_track().set_state_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_state_process_state_axis_id_dialog_type(
                    ProcessState::InProgress,
                    axis_id,
                    dialog_type,
                );
            }
            self.send_msg_process_succeeded(process_result_display_ref.as_ref());
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        &format!(
                            "Can't open 3dmod on {} with model: {}",
                            key,
                            model.unwrap_or("null")
                        ),
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java private `okToFiducialModelTrack(AxisID)` (ApplicationManager.java:3093).
    fn ok_to_fiducial_model_track(&'static self, axis_id: AxisID) -> bool {
        let fiducial_file = dataset_files::get_fiducial_model_file(self, Some(axis_id));
        if !fiducial_file.exists() {
            return true;
        }
        let fiducial_last_modified =
            utilities::java_io_file_last_modified(&fiducial_file.to_string_lossy());
        let fiducial_last_tracked = self.get_state().get_fid_file_last_modified(axis_id);
        let mut seed_last_modified: i64 = 0;
        let seed_file = dataset_files::get_seed_file(self, Some(axis_id));
        if seed_file.exists() {
            seed_last_modified =
                utilities::java_io_file_last_modified(&seed_file.to_string_lossy());
        }
        // If the fid file is more recent and it is not the product of tracking
        // then ask before overwriting it by tracking
        if fiducial_last_modified > seed_last_modified
            && !fiducial_last_tracked.is_null()
            && fiducial_last_modified != fiducial_last_tracked.get_long()
        {
            return ui_harness::INSTANCE.with(|ui_harness| ui_harness.open_yes_no_warning_dialog(
                Some(self),
                &format!(
                    "If you track fiducials now, you will overwrite the fiducial file you just modified.\nFirst press the {} button.\n\nWARNING:  If you press Yes, you will overwrite your fiducial file.",
                    beadtrack_panel::USE_MODEL_LABEL
                ),
                Some(axis_id),
            ));
        }
        true
    }

    /// Java private `okToMakeFiducialModelSeedModel(AxisID)`
    /// (ApplicationManager.java:3118).
    fn ok_to_make_fiducial_model_seed_model(&'static self, axis_id: AxisID) -> bool {
        let seed_file = dataset_files::get_seed_file(self, Some(axis_id));
        if !seed_file.exists() {
            return true;
        }
        let seed_last_modified =
            utilities::java_io_file_last_modified(&seed_file.to_string_lossy());
        let seed_last_copied = self.get_state().get_seed_file_last_modified(axis_id);
        let mut fiducial_last_modified: i64 = 0;
        let fiducial_file = dataset_files::get_fiducial_model_file(self, Some(axis_id));
        if fiducial_file.exists() {
            fiducial_last_modified =
                utilities::java_io_file_last_modified(&fiducial_file.to_string_lossy());
        }
        // If the seed file is more recent then the fid file
        // then ask before overwriting it by copying the fid file into the seed
        // file.
        if seed_last_modified > fiducial_last_modified
            && seed_last_modified != seed_last_copied.get_long()
        {
            return ui_harness::INSTANCE.with(|ui_harness| ui_harness.open_yes_no_warning_dialog(
                Some(self),
                &format!(
                    "If you copy the fiducial file to the seed file now, you will overwrite the seed file you just modified.\nFirst press the {} button.\n\nWARNING:  If you press Yes, you will overwrite your seed file.",
                    beadtrack_panel::TRACK_LABEL
                ),
                Some(axis_id),
            ));
        }
        true
    }

    /// Java `fiducialModelTrack(AxisID, ProcessResultDisplay, ProcessSeries, DialogType,
    /// BeadTrackDisplay)` (ApplicationManager.java:3148).  Get the beadtrack parameters
    /// from the fiducial model dialog and run the track com script.
    pub fn fiducial_model_track(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: DialogType,
        display: &dyn BeadTrackDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let Some(beadtrack_param) = self.update_track_com(Some(display), axis_id, true) else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            if let Some(process_series) = process_series.as_ref() {
                process_series.borrow().end_series();
            }
            return;
        };
        if !self.ok_to_fiducial_model_track(axis_id) {
            if let Some(process_series) = process_series.as_ref() {
                process_series.borrow().end_series();
            }
            return;
        }
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let thread_name = match self.get_process_mgr().fiducial_model_track(
            Arc::new(beadtrack_param),
            axis_id,
            process_result_display_ref,
            process_series.map(|process_series| Arc::new(EdtRef::new(process_series))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = vec![
                    format!("Can not execute track{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Tracking fiducials"),
                axis_id,
                Some(&ProcessName::TRACK),
            );
        }
    }

    /// Java `makeFiducialModelSeedModel(AxisID)` (ApplicationManager.java:3190).  Using
    /// Fiducial Model as Seed.
    pub fn make_fiducial_model_seed_model(&'static self, axis_id: AxisID) -> bool {
        let mut retval = true;
        let meta_data = self.get_meta_data();
        // Java string concatenation writes a null propertyUserDir as "null".
        let property_user_dir = self
            .get_property_user_dir()
            .unwrap_or_else(|| "null".to_string());
        let seed_model_filename = format!(
            "{}{}{}{}.seed",
            property_user_dir,
            std::path::MAIN_SEPARATOR,
            meta_data.get_dataset_name(),
            axis_id.get_extension()
        );
        let seed_model = std::path::PathBuf::from(&seed_model_filename);
        let fiducial_model_filename = format!(
            "{}{}{}{}.fid",
            property_user_dir,
            std::path::MAIN_SEPARATOR,
            meta_data.get_dataset_name(),
            axis_id.get_extension()
        );
        let fiducial_model = std::path::PathBuf::from(&fiducial_model_filename);
        if !fiducial_model.exists() {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &format!(
                        "Fiducial model does not exist.  {} failed.",
                        beadtrack_panel::USE_MODEL_LABEL
                    ),
                    "Process Failed",
                    Some(axis_id),
                )
            });
            return false;
        }
        if !self.ok_to_make_fiducial_model_seed_model(axis_id) {
            return false;
        } /* if (seedModel.exists() && seedModel.lastModified() > fiducialModel.lastModified())
         * { String[] message = new String[3]; message[0] = "WARNING: The seed model file is
         * more recent the fiducial model file"; message[1] = "To avoid losing your changes
         * to the seed model file,"; message[2] = "track fiducials before pressing Use
         * Fiducial Model as Seed."; uiHarness.openMessageDialog(message, "Use Fiducial Model
         * as Seed Failed", axisID); return; } */
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_progress_bar_string_int_boolean_axis_id(
                Some("Using Fiducial Model as Seed"),
                1,
                false,
                axis_id,
            );
        }
        self.get_recon_process_track()
            .set_fiducial_model_state(ProcessState::InProgress, axis_id);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_fiducial_model_state(ProcessState::InProgress, axis_id);
        }
        let orig_seed_model_filename = format!(
            "{}{}{}{}_orig.seed",
            property_user_dir,
            std::path::MAIN_SEPARATOR,
            meta_data.get_dataset_name(),
            axis_id.get_extension()
        );
        let orig_seed_model = std::path::PathBuf::from(&orig_seed_model_filename);
        // backup original seed model file
        if seed_model.exists() && orig_seed_model.exists() {
            self.backup_file(Some(&seed_model), Some(axis_id));
        }
        let rename = (|| -> Result<(), LogFileError> {
            if seed_model.exists() && !orig_seed_model.exists() {
                utilities::rename_file(
                    Some(self),
                    Some(axis_id),
                    Some(&seed_model),
                    Some(&orig_seed_model),
                    false,
                    false,
                    false,
                )?;
            }
            // rename fiducial model file to seed model file
            utilities::rename_file(
                Some(self),
                Some(axis_id),
                Some(&fiducial_model),
                Some(&seed_model),
                false,
                false,
                false,
            )?;
            Ok(())
        })();
        match rename {
            Ok(()) => {}
            Err(LogFileError::Lock(_)) => {
                retval = false;
            }
            // Java `catch (final IOException | LogFileException except)`.
            Err(except) => {
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "File Rename Error (4)",
                        Some(axis_id),
                    )
                });
                retval = false;
            }
        }
        let quit = (|| -> Result<(), ImodManagerError> {
            let imod_manager = self.get_imod_manager();
            if imod_manager
                .is_open_string_axis_id(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))?
            {
                let seed_model_name = seed_model
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if Some(seed_model_name)
                    == imod_manager
                        .get_model_name(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))?
                {
                    if !etomo_director::ARGUMENTS
                        .lock()
                        .unwrap()
                        .is_auto_close_3dmod()
                    {
                        let message = vec![
                            format!(
                                "The old seed model file is open in 3dmod.{}",
                                self.get_file_lock_message(Some("  "))
                            ),
                            "Should it be closed?".to_string(),
                        ];
                        if ui_harness::INSTANCE.with(|ui_harness| {
                            ui_harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                                Some(self),
                                &message,
                                Some(axis_id),
                            )
                        }) {
                            imod_manager.quit_string_axis_id(
                                imod_manager::COARSE_ALIGNED_KEY,
                                Some(axis_id),
                            )?;
                        }
                    } else {
                        imod_manager
                            .quit_string_axis_id(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))?;
                    }
                }
            }
            Ok(())
        })();
        match quit {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(e)) => {
                eprintln!("{e:?}");
                eprintln!("Axis type exception in replaceRawStack");
                retval = false;
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
                retval = false;
            }
            Err(ImodManagerError::SystemProcess(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "System Process Exception",
                        Some(axis_id),
                    )
                });
                retval = false;
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
        let seed_file = dataset_files::get_seed_file(self, Some(axis_id));
        if seed_file.exists() {
            self.get_state().set_seed_file_last_modified(
                axis_id,
                utilities::java_io_file_last_modified(&seed_file.to_string_lossy()),
            );
        } else {
            self.get_state().reset_seed_file_last_modified(axis_id);
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id(axis_id);
        }
        self.get_state().set_seeding_done(axis_id, true);
        if axis_id == AxisID::Second {
            self.update_dialog_fiducial_model_dialog_axis_id(
                self.fiducial_model_dialog_a.get().as_ref().cloned(),
                AxisID::First,
            );
            if let Some(fiducial_model_dialog_b) = self.fiducial_model_dialog_b.get() {
                fiducial_model_dialog_b.update_display();
            }
        } else {
            self.update_dialog_fiducial_model_dialog_axis_id(
                self.fiducial_model_dialog_b.get().as_ref().cloned(),
                AxisID::Second,
            );
            if let Some(fiducial_model_dialog_a) = self.fiducial_model_dialog_a.get() {
                fiducial_model_dialog_a.update_display();
            }
        }
        retval
    }

    /// Java `calcBinnedBeadDiameterPixels(AxisID, FileType, int)`
    /// (ApplicationManager.java:3308).
    pub fn calc_binned_bead_diameter_pixels(
        &'static self,
        axis_id: AxisID,
        file_type: &Arc<FileType>,
        digits_after_decimal: i32,
    ) -> f64 {
        let meta_data = self.get_meta_data();
        let adj = 10f64.powf(digits_after_decimal as f64);
        (utilities::java_lang_math_round(
            meta_data.get_fiducial_diameter()
                / meta_data.get_pixel_size()
                / utilities::get_stack_binning_for_file_type(self, axis_id, file_type) as f64
                * adj,
        )) as f64
            / adj
    }

    /// Java `calcUnbinnedBeadDiameterPixels()` (ApplicationManager.java:3315).
    pub fn calc_unbinned_bead_diameter_pixels(&self) -> f64 {
        let meta_data = self.get_meta_data();
        (utilities::java_lang_math_round(
            meta_data.get_fiducial_diameter() / meta_data.get_pixel_size() * 100.0,
        )) as f64
            / 100.0
    }

    /// Java `imodFixFiducials(AxisID, Run3dmodMenuOptions, ProcessResultDisplay,
    /// ImodProcess.BeadFixerMode, String)` (ApplicationManager.java:3331).  Open 3dmod
    /// with the new fidcuial model.
    pub fn imod_fix_fiducials(
        &'static self,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
        process_result_display: Option<ProcessResultDisplayHandle>,
        beadfixer_mode: BeadFixerMode,
        skip_list: Option<&str>,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .as_ref()
            .map(|display| Arc::new(EdtRef::new(Rc::clone(display))));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // if fix fiducials has already run, don't change auto center
        let set_auto_center = !self.get_state().is_fixed_fiducials(axis_id);
        let fiducial_model = format!(
            "{}{}.fid",
            self.get_meta_data().get_dataset_name(),
            axis_id.get_extension()
        );
        let result = (|| -> Result<(), ImodManagerError> {
            let imod_manager = self.get_imod_manager();
            imod_manager.set_open_bead_fixer(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                true,
            )?;
            if set_auto_center {
                imod_manager.set_auto_center(
                    imod_manager::COARSE_ALIGNED_KEY,
                    Some(axis_id),
                    false,
                )?;
            }
            imod_manager.set_skip_list(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                skip_list,
            )?;
            imod_manager.set_beadfixer_mode(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                Some(beadfixer_mode),
            )?;
            let residual_mode = beadfixer_mode == BeadFixerMode::ResidualMode
                || beadfixer_mode == BeadFixerMode::PatchTrackingResidualMode;
            imod_manager.set_open_log(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                residual_mode,
                Some(&dataset_files::get_log_name(
                    self,
                    Some(axis_id),
                    Some(&ProcessName::ALIGN),
                )),
            )?;
            let tilt_file = dataset_files::get_raw_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                imod_manager.set_tilt_file(
                    imod_manager::COARSE_ALIGNED_KEY,
                    Some(axis_id),
                    tilt_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                imod_manager.reset_tilt_file(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))?;
            }
            if residual_mode {
                // Listen for message from 3dmod that the align logs have to be redone.
                imod_manager.set_continuous_listener_target(
                    imod_manager::COARSE_ALIGNED_KEY,
                    Some(axis_id),
                    Some(Arc::new(self) as Arc<dyn ContinuousListenerTarget>),
                )?;
            }
            if beadfixer_mode == BeadFixerMode::PatchTrackingResidualMode {
                imod_manager
                    .set_open_model_view(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))?;
            }
            imod_manager.set_preserve_contrast(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                true,
            )?;
            imod_manager.open_string_axis_id_string_boolean_run3dmod_menu_options(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                Some(&fiducial_model),
                true,
                Some(menu_options),
            )?;
            self.send_msg_process_succeeded(process_result_display_ref.as_ref());
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return;
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        &format!(
                            "Can't open 3dmod on coarse aligned stack with model: {fiducial_model}"
                        ),
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return;
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
        self.get_state().set_fixed_fiducials(axis_id, true);
    }

    /// Java private `updateTiltCom(ConstTiltalignParam, AxisID)`
    /// (ApplicationManager.java:3396).  Update the tilt parameters dependent on the
    /// align script - local alignments file - the exclude list.
    fn update_tilt_com_const_tiltalign_param_axis_id(
        &'static self,
        tiltalign_param: &ConstTiltalignParam,
        current_axis: AxisID,
    ) {
        let meta_data = self.get_meta_data();
        self.get_com_script_manager().load_tilt(current_axis);
        let mut tilt_param = self.get_com_script_manager().get_tilt_param(current_axis);
        tilt_param.set_fiducialess(meta_data.is_fiducialess(current_axis));
        let align_file_extension = format!("{}local.xf", current_axis.get_extension());
        if tiltalign_param.get_local_alignments().is() {
            tilt_param.set_local_align_file(Some(&format!(
                "{}{}",
                meta_data.get_dataset_name(),
                align_file_extension
            )));
        } else {
            tilt_param.set_local_align_file(Some(""));
        }
        UIExpertUtilities::INSTANCE.roll_tilt_com_angles(self, current_axis);
        self.update_exclude_list(&mut tilt_param, current_axis);
        self.get_com_script_manager()
            .save_tilt(&tilt_param, current_axis);
        meta_data.set_fiducialess(current_axis, tilt_param.is_fiducialess());
    }

    /// Java private `updateTrackCom(BeadTrackDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:3417).  Update the specified track com script.
    fn update_track_com(
        &'static self,
        display: Option<&dyn BeadTrackDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<BeadtrackParam> {
        let Some(display) = display else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    "Can not update track?.com without an active display",
                    "Program logic error",
                    Some(axis_id),
                )
            });
            return None;
        };
        let mut beadtrack_param = self.get_com_script_manager().get_beadtrack_param(axis_id);
        match display.get_parameters(&mut beadtrack_param, do_validation) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(BeadTrackDisplayException::FortranInputSyntaxException(except)) => {
                eprintln!("{except:?}");
                let error_message = vec![
                    "Beadtrack Parameter Syntax Error".to_string(),
                    except.get_message().unwrap_or("null").to_string(),
                    format!("New value: {}", except.get_new_string()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Beadtrack Parameter Syntax Error",
                        Some(axis_id),
                    )
                });
                return None;
            }
            Err(BeadTrackDisplayException::InvalidEtomoNumberException(_)) => {
                return None;
            }
        }
        // Java also catches NumberFormatException ("Beadtrack Parameter Syntax Error",
        // axisID.getExtension(), except.getMessage()); it is unchecked and has no Rust
        // counterpart.
        self.get_com_script_manager()
            .save_track(&beadtrack_param, axis_id);
        Some(beadtrack_param)
    }

    /// Java `openFineAlignmentDialog(AxisID)` (ApplicationManager.java:3462).  Open the
    /// alignment estimation dialog.
    pub fn open_fine_alignment_dialog(&'static self, axis_id: AxisID) {
        let meta_data = self.get_meta_data();
        // Check to see if the com files are present otherwise pop up a dialog
        // box informing the user to run the setup process
        if !UIExpertUtilities::INSTANCE.are_scripts_created(self, meta_data, axis_id) {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(axis_id);
            }
            return;
        }
        let action_message =
            self.set_current_dialog_type(Some(DialogType::FineAlignment), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.select_button(axis_id, "Fine Alignment");
        }
        let fine_alignment_dialog_a = self.fine_alignment_dialog_a.get();
        let fine_alignment_dialog_b = self.fine_alignment_dialog_b.get();
        if self.show_if_exists(
            fine_alignment_dialog_a.as_deref().map(|dialog| &**dialog),
            fine_alignment_dialog_b.as_deref().map(|dialog| &**dialog),
            axis_id,
            action_message.as_deref(),
        ) {
            return;
        }
        // Create a new dialog panel and map it the generic reference
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("AlignmentEstimationDialog"),
            Some(utilities::STARTED_STATUS),
        );
        let fine_alignment_dialog = AlignmentEstimationDialog::new(self, axis_id);
        utilities::timestamp_process_container_status(
            Some("new"),
            Some("AlignmentEstimationDialog"),
            Some(utilities::FINISHED_STATUS),
        );
        if axis_id == AxisID::Second {
            self.fine_alignment_dialog_b
                .set(Some(Rc::clone(&fine_alignment_dialog)));
        } else {
            self.fine_alignment_dialog_a
                .set(Some(Rc::clone(&fine_alignment_dialog)));
        }

        // Load the required align{|a|b}.com files, fill in the dialog box
        // params and set it to the appropriate state
        self.get_com_script_manager().load_align(axis_id);
        let mut tiltalign_param = self.get_com_script_manager().get_tiltalign_param(axis_id);
        // If this is a montage, then binning can only be 1, so no need to upgrade
        if meta_data.get_view_type() != ViewType::Montage {
            // upgrade and save param to comscript
            UIExpertUtilities::INSTANCE.upgrade_old_align_com(self, axis_id, &mut tiltalign_param);
        }
        fine_alignment_dialog.set_default_parameters();
        fine_alignment_dialog.set_parameters_const_meta_data(meta_data);
        fine_alignment_dialog.set_tiltalign_params(&tiltalign_param);

        // Restrict align
        let mut is_make_com_file_valid = true;
        if !self
            .get_com_script_manager()
            .load_restrict_align(axis_id, false)
        {
            let mut makecom_file_param = MakecomfileParam::new(
                self,
                axis_id,
                Arc::clone(&file_type::CLASS.restrict_align_comscript),
            );
            if !fine_alignment_dialog
                .get_parameters_makecomfile_param_boolean(&mut makecom_file_param, true)
            {
                is_make_com_file_valid = false;
                // Java `Thread.dumpStack()`: prints the calling Java thread's stack
                // trace; a Rust backtrace is neither the same frames nor the same text.
                eprintln!("Invalid field for restrict align from fine alignment");
            } else {
                self.makecomfile(axis_id, &mut makecom_file_param);
                self.get_com_script_manager()
                    .load_restrict_align(axis_id, true);
            }
        }
        if is_make_com_file_valid {
            let restrictalign_param = self
                .get_com_script_manager()
                .get_restrict_align_param(axis_id);
            fine_alignment_dialog.set_restrictalign_params(&restrictalign_param);
        }

        // Handle patch tracking.
        let mut imodinfo = Imodinfo::new(&file_type::CLASS.fiducial_model);
        if imodinfo.is_patch_tracking(self, axis_id) {
            // Patch tracking creates fiducials on one side only. If the .fid file was
            // created by patch tracking, default to 1 surface if this is the first
            // time
            // opening fine alignment. In any case tell the dialog that patch tracking
            // was used.
            if !meta_data.is_fine_exists(axis_id) {
                fine_alignment_dialog.set_surfaces_to_analyze(1);
            }
            fine_alignment_dialog.set_patch_tracking(true);
        }
        // Java `getBaseScreenState(axisID)`, which this class overrides to return
        // `getScreenState(axisID)` (ApplicationManager.java:11194).
        fine_alignment_dialog.set_parameters_base_screen_state(self.get_screen_state(axis_id));
        meta_data.set_fine_exists(axis_id, true);
        // Create a default transferfid object to populate the alignment dialog
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&fine_alignment_dialog.get_container(), axis_id);
        }
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `saveCurrentDialog(AxisID)` (ApplicationManager.java:3549).  Calls
    /// saveAction for the dialog type.  SaveAction() is a way to call the dialog's done
    /// function without pressing an exit button.
    pub fn save_current_dialog(&'static self, axis_id: AxisID) {
        let current_dialog_type = self.get_current_dialog_type(Some(axis_id));
        // handle dialogs with experts
        if let Some(expert) = self.get_ui_expert(current_dialog_type, axis_id) {
            expert.save_action();
            return;
        }
        // handle accessible dialogs
        if let Some(dialog) = self.get_dialog(current_dialog_type, axis_id) {
            dialog.save_action();
        }
    }

    /// Java `saveDialogs()` (ApplicationManager.java:3600).  Saves all dialogs that are
    /// not set to null.  Called by exitProgram().
    pub fn save_dialogs(&'static self) {
        if self.meta_data.lock().unwrap().is_none() {
            return;
        }
        let first_axis_id = if self.get_meta_data().get_axis_type() == AxisType::DualAxis {
            AxisID::First
        } else {
            AxisID::Only
        };
        if let Some(pre_proc_dialog_a) = self.pre_proc_dialog_a.get() {
            self.save_pre_proc_dialog(&pre_proc_dialog_a, first_axis_id);
        }
        if let Some(pre_proc_dialog_b) = self.pre_proc_dialog_b.get() {
            self.save_pre_proc_dialog(&pre_proc_dialog_b, AxisID::Second);
        }
        if let Some(coarse_align_dialog_a) = self.coarse_align_dialog_a.get() {
            self.save_coarse_align_dialog(&coarse_align_dialog_a, first_axis_id);
        }
        if let Some(coarse_align_dialog_b) = self.coarse_align_dialog_b.get() {
            self.save_coarse_align_dialog(&coarse_align_dialog_b, AxisID::Second);
        }
        if let Some(fiducial_model_dialog_a) = self.fiducial_model_dialog_a.get() {
            self.save_fiducial_model_dialog(&fiducial_model_dialog_a, first_axis_id);
        }
        if let Some(fiducial_model_dialog_b) = self.fiducial_model_dialog_b.get() {
            self.save_fiducial_model_dialog(&fiducial_model_dialog_b, AxisID::Second);
        }
        if let Some(fine_alignment_dialog_a) = self.fine_alignment_dialog_a.get() {
            self.save_alignment_estimation_dialog(&fine_alignment_dialog_a, first_axis_id);
        }
        if let Some(fine_alignment_dialog_b) = self.fine_alignment_dialog_b.get() {
            self.save_alignment_estimation_dialog(&fine_alignment_dialog_b, AxisID::Second);
        }

        // `getUIExpert` creates the expert when it does not exist for these three
        // dialog types, so the source's unchecked dereference cannot see null.
        if let Some(expert) =
            self.get_ui_expert(Some(DialogType::TomogramPositioning), first_axis_id)
        {
            expert.save_dialog(DialogExitState::Save);
        }
        if let Some(expert) =
            self.get_ui_expert(Some(DialogType::TomogramPositioning), AxisID::Second)
        {
            expert.save_dialog(DialogExitState::Save);
        }
        if let Some(expert) = self.get_ui_expert(Some(DialogType::FinalAlignedStack), first_axis_id)
        {
            expert.save_dialog(DialogExitState::Save);
        }
        if let Some(expert) =
            self.get_ui_expert(Some(DialogType::FinalAlignedStack), AxisID::Second)
        {
            expert.save_dialog(DialogExitState::Save);
        }
        if let Some(expert) =
            self.get_ui_expert(Some(DialogType::TomogramGeneration), first_axis_id)
        {
            expert.save_dialog(DialogExitState::Save);
        }
        if let Some(expert) =
            self.get_ui_expert(Some(DialogType::TomogramGeneration), AxisID::Second)
        {
            expert.save_dialog(DialogExitState::Save);
        }

        if let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() {
            self.save_tomogram_combination_dialog(Some(&tomogram_combination_dialog));
        }
        if let Some(post_processing_dialog) = self.post_processing_dialog.get() {
            self.save_post_processing(&post_processing_dialog);
        }
        if let Some(clean_up_dialog) = self.clean_up_dialog.get() {
            self.save_clean_up(&clean_up_dialog);
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer, boolean)`
    /// (ApplicationManager.java:3655).
    fn update_directive_directive_map_directive_def_string_buffer_boolean(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: bool,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_boolean(value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, AxisID, StringBuffer,
    /// boolean)` (ApplicationManager.java:3666).
    fn update_directive_directive_map_directive_def_axis_id_string_buffer_boolean(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        pair_axis_id: AxisID,
        errmsg: &mut String,
        value: bool,
    ) {
        if let Some(directive) =
            map.get_directive_from_pair(Some(*directive_def), Some(pair_axis_id))
        {
            directive.set_value_boolean(value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer, boolean,
    /// boolean)` (ApplicationManager.java:3677).
    fn update_directive_directive_map_directive_def_string_buffer_boolean_boolean(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: bool,
        default_value: bool,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_boolean(value);
            directive.set_default_value_boolean(default_value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, AxisID, StringBuffer,
    /// boolean, boolean)` (ApplicationManager.java:3689).
    fn update_directive_directive_map_directive_def_axis_id_string_buffer_boolean_boolean(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        pair_axis_id: AxisID,
        errmsg: &mut String,
        value: bool,
        default_value: bool,
    ) {
        if let Some(directive) =
            map.get_directive_from_pair(Some(*directive_def), Some(pair_axis_id))
        {
            directive.set_value_boolean(value);
            directive.set_default_value_boolean(default_value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer,
    /// ConstEtomoNumber)` (ApplicationManager.java:3702).
    fn update_directive_directive_map_directive_def_string_buffer_const_etomo_number(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: Option<&ConstEtomoNumber>,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_const_etomo_number(value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, AxisID, StringBuffer,
    /// ConstEtomoNumber)` (ApplicationManager.java:3713).
    fn update_directive_directive_map_directive_def_axis_id_string_buffer_const_etomo_number(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        pair_axis_id: AxisID,
        errmsg: &mut String,
        value: Option<&ConstEtomoNumber>,
    ) {
        if let Some(directive) =
            map.get_directive_from_pair(Some(*directive_def), Some(pair_axis_id))
        {
            directive.set_value_const_etomo_number(value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer,
    /// ConstEtomoNumber, ConstEtomoNumber)` (ApplicationManager.java:3724).
    fn update_directive_directive_map_directive_def_string_buffer_const_etomo_number_const_etomo_number(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: Option<&ConstEtomoNumber>,
        default_value: Option<&ConstEtomoNumber>,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_const_etomo_number(value);
            directive.set_default_value_const_etomo_number(default_value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer,
    /// ConstStringParameter)` (ApplicationManager.java:3737).
    fn update_directive_directive_map_directive_def_string_buffer_const_string_parameter(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: Option<&dyn ConstStringParameter>,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_const_string_parameter(value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer, double)`
    /// (ApplicationManager.java:3748).
    fn update_directive_directive_map_directive_def_string_buffer_double(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: f64,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_double(value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, AxisID, StringBuffer,
    /// double[])` (ApplicationManager.java:3759).
    fn update_directive_directive_map_directive_def_axis_id_string_buffer_double_array(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        pair_axis_id: AxisID,
        errmsg: &mut String,
        value: &[f64],
    ) {
        if let Some(directive) =
            map.get_directive_from_pair(Some(*directive_def), Some(pair_axis_id))
        {
            directive.set_value_double_array(Some(value));
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer,
    /// FortranInputString)` (ApplicationManager.java:3770).
    fn update_directive_directive_map_directive_def_string_buffer_fortran_input_string(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: Option<&FortranInputString>,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_fortran_input_string(value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer, int,
    /// int)` (ApplicationManager.java:3781).
    fn update_directive_directive_map_directive_def_string_buffer_int_int(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: i32,
        default_value: i32,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_int(value);
            directive.set_default_value_int(default_value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer, String)`
    /// (ApplicationManager.java:3793).
    fn update_directive_directive_map_directive_def_string_buffer_string(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: Option<&str>,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_string(value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, AxisID, StringBuffer,
    /// String)` (ApplicationManager.java:3804).
    fn update_directive_directive_map_directive_def_axis_id_string_buffer_string(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        pair_axis_id: AxisID,
        errmsg: &mut String,
        value: Option<&str>,
    ) {
        if let Some(directive) =
            map.get_directive_from_pair(Some(*directive_def), Some(pair_axis_id))
        {
            directive.set_value_string(value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java private `updateDirective(DirectiveMap, DirectiveDef, StringBuffer, String,
    /// String)` (ApplicationManager.java:3815).
    fn update_directive_directive_map_directive_def_string_buffer_string_string(
        &self,
        map: &DirectiveMap,
        directive_def: &DirectiveDef,
        errmsg: &mut String,
        value: Option<&str>,
        default_value: Option<&str>,
    ) {
        if let Some(directive) = map.get_directive_directive_def(Some(*directive_def)) {
            directive.set_value_string(value);
            directive.set_default_value_string(default_value);
        } else {
            errmsg.push_str(&format!("Missing directive: {directive_def}.  "));
        }
    }

    /// Java `updateDirectiveMap(DirectiveMap, StringBuffer)` (ApplicationManager.java:3854).
    /// Save the dataset file and the open dialogs in the dataset.  Update the setup and
    /// runtime parameters in the directive maps.
    ///
    /// Not an override: `BaseManager.updateDirectiveMap` takes a
    /// `DirectiveMapInterface`, this method a `DirectiveMap`.
    pub fn update_directive_map(&'static self, directive_map: &DirectiveMap, errmsg: &mut String) {
        let meta_data = self.get_meta_data();
        // Save directive data from the current dataset.
        let dual_axis = meta_data.get_axis_type() == AxisType::DualAxis;
        let first_axis_id = if dual_axis {
            AxisID::First
        } else {
            AxisID::Only
        };
        let mut cur_axis_id = first_axis_id;
        let _axis_type = meta_data.get_axis_type();
        let montage = meta_data.get_view_type() == ViewType::Montage;
        // setupset.copyarg directvies
        self.update_directive_directive_map_directive_def_string_buffer_boolean_boolean(
            directive_map,
            &DirectiveDef::DUAL,
            errmsg,
            meta_data.get_axis_type() == AxisType::DualAxis,
            true,
        );
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::MONTAGE,
            errmsg,
            montage,
        );
        self.update_directive_directive_map_directive_def_string_buffer_double(
            directive_map,
            &DirectiveDef::PIXEL,
            errmsg,
            meta_data.get_pixel_size(),
        );
        self.update_directive_directive_map_directive_def_string_buffer_double(
            directive_map,
            &DirectiveDef::GOLD,
            errmsg,
            meta_data.get_fiducial_diameter(),
        );
        self.update_directive_directive_map_directive_def_axis_id_string_buffer_const_etomo_number(
            directive_map,
            &DirectiveDef::ROTATION,
            cur_axis_id,
            errmsg,
            Some(&*meta_data.get_image_rotation(cur_axis_id)),
        );
        if dual_axis {
            cur_axis_id = AxisID::Second;
            self.update_directive_directive_map_directive_def_axis_id_string_buffer_const_etomo_number(
                directive_map,
                &DirectiveDef::ROTATION,
                cur_axis_id,
                errmsg,
                Some(&*meta_data.get_image_rotation(cur_axis_id)),
            );
        }
        cur_axis_id = first_axis_id;
        let mut tilt_angle_spec = meta_data.get_tilt_angle_spec_a();
        if tilt_angle_spec.get_type() == TiltAngleType::Range {
            self.update_directive_directive_map_directive_def_axis_id_string_buffer_double_array(
                directive_map,
                &DirectiveDef::FIRST_INC,
                cur_axis_id,
                errmsg,
                &[
                    tilt_angle_spec.get_range_min(),
                    tilt_angle_spec.get_range_step(),
                ],
            );
        } else if tilt_angle_spec.get_type() == TiltAngleType::File {
            self.update_directive_directive_map_directive_def_axis_id_string_buffer_boolean(
                directive_map,
                &DirectiveDef::USE_RAW_TLT,
                cur_axis_id,
                errmsg,
                true,
            );
        } else if tilt_angle_spec.get_type() == TiltAngleType::Extract {
            self.update_directive_directive_map_directive_def_axis_id_string_buffer_boolean_boolean(
                directive_map,
                &DirectiveDef::EXTRACT,
                cur_axis_id,
                errmsg,
                true,
                true,
            );
        }
        if dual_axis {
            cur_axis_id = AxisID::Second;
            tilt_angle_spec = meta_data.get_tilt_angle_spec_b();
            if tilt_angle_spec.get_type() == TiltAngleType::Range {
                self.update_directive_directive_map_directive_def_axis_id_string_buffer_double_array(
                    directive_map,
                    &DirectiveDef::BFIRST_INC,
                    cur_axis_id,
                    errmsg,
                    &[tilt_angle_spec.get_range_min(), tilt_angle_spec.get_range_step()],
                );
            } else if tilt_angle_spec.get_type() == TiltAngleType::File {
                self.update_directive_directive_map_directive_def_axis_id_string_buffer_boolean(
                    directive_map,
                    &DirectiveDef::BUSE_RAW_TLT,
                    cur_axis_id,
                    errmsg,
                    true,
                );
            } else if tilt_angle_spec.get_type() == TiltAngleType::Extract {
                self.update_directive_directive_map_directive_def_axis_id_string_buffer_boolean_boolean(
                    directive_map,
                    &DirectiveDef::BEXTRACT,
                    cur_axis_id,
                    errmsg,
                    true,
                    true,
                );
            }
        }
        cur_axis_id = first_axis_id;
        self.update_directive_directive_map_directive_def_axis_id_string_buffer_string(
            directive_map,
            &DirectiveDef::SKIP,
            cur_axis_id,
            errmsg,
            Some(&meta_data.get_exclude_projections_a()),
        );
        if meta_data.is_twodir(cur_axis_id) {
            self.update_directive_directive_map_directive_def_axis_id_string_buffer_string(
                directive_map,
                &DirectiveDef::TWODIR,
                cur_axis_id,
                errmsg,
                Some(&meta_data.get_twodir(cur_axis_id)),
            );
        } else {
            self.update_directive_directive_map_directive_def_axis_id_string_buffer_string(
                directive_map,
                &DirectiveDef::TWODIR,
                cur_axis_id,
                errmsg,
                Some(""),
            );
        }
        if dual_axis {
            cur_axis_id = AxisID::Second;
            self.update_directive_directive_map_directive_def_axis_id_string_buffer_string(
                directive_map,
                &DirectiveDef::SKIP,
                cur_axis_id,
                errmsg,
                Some(&meta_data.get_exclude_projections_b()),
            );
            if meta_data.is_twodir(cur_axis_id) {
                self.update_directive_directive_map_directive_def_axis_id_string_buffer_string(
                    directive_map,
                    &DirectiveDef::TWODIR,
                    cur_axis_id,
                    errmsg,
                    Some(&meta_data.get_twodir(cur_axis_id)),
                );
            } else {
                self.update_directive_directive_map_directive_def_axis_id_string_buffer_string(
                    directive_map,
                    &DirectiveDef::TWODIR,
                    cur_axis_id,
                    errmsg,
                    Some(""),
                );
            }
        } else {
            // Single axis: `curAxisID` is still the first axis, so this rewrites the
            // same TWODIR directive to "" (as the source does).
            self.update_directive_directive_map_directive_def_axis_id_string_buffer_string(
                directive_map,
                &DirectiveDef::TWODIR,
                cur_axis_id,
                errmsg,
                Some(""),
            );
        }
        cur_axis_id = first_axis_id;
        self.update_directive_directive_map_directive_def_string_buffer_string(
            directive_map,
            &DirectiveDef::DISTORT,
            errmsg,
            Some(&meta_data.get_distortion_file()),
        );
        self.update_directive_directive_map_directive_def_string_buffer_string_string(
            directive_map,
            &DirectiveDef::BINNING,
            errmsg,
            Some(&meta_data.get_binning()),
            Some("1"),
        );
        self.update_directive_directive_map_directive_def_string_buffer_string(
            directive_map,
            &DirectiveDef::GRADIENT,
            errmsg,
            Some(&meta_data.get_mag_gradient_file()),
        );
        self.update_directive_directive_map_directive_def_axis_id_string_buffer_boolean(
            directive_map,
            &DirectiveDef::FOCUS,
            cur_axis_id,
            errmsg,
            meta_data.get_adjusted_focus_a().is(),
        );
        if dual_axis {
            cur_axis_id = AxisID::Second;
            self.update_directive_directive_map_directive_def_axis_id_string_buffer_boolean(
                directive_map,
                &DirectiveDef::FOCUS,
                cur_axis_id,
                errmsg,
                meta_data.get_adjusted_focus_b().is(),
            );
        }
        cur_axis_id = first_axis_id;
        let mut ctf_plotter_param = None;
        if self
            .get_com_script_manager()
            .load_ctf_plotter(first_axis_id, false)
        {
            // `getCtfPlotterParam` never returns null in the translation, so the
            // source's null test is always true.
            let param = self
                .get_com_script_manager()
                .get_ctf_plotter_param(first_axis_id);
            self.update_directive_directive_map_directive_def_string_buffer_const_etomo_number(
                directive_map,
                &DirectiveDef::DEFOCUS,
                errmsg,
                param.get_expected_defocus(),
            );
            ctf_plotter_param = Some(param);
        }
        if self
            .get_com_script_manager()
            .load_ctf_correction(first_axis_id, false)
        {
            let ctf_phase_flip_param = self
                .get_com_script_manager()
                .get_ctf_phase_flip_param(first_axis_id);
            self.update_directive_directive_map_directive_def_string_buffer_const_etomo_number(
                directive_map,
                &DirectiveDef::VOLTAGE,
                errmsg,
                Some(ctf_phase_flip_param.get_voltage()),
            );
            self.update_directive_directive_map_directive_def_string_buffer_const_etomo_number(
                directive_map,
                &DirectiveDef::CS,
                errmsg,
                Some(ctf_phase_flip_param.get_spherical_aberration()),
            );
        }
        if let Some(ctf_plotter_param) = ctf_plotter_param.as_ref() {
            self.update_directive_directive_map_directive_def_string_buffer_string(
                directive_map,
                &DirectiveDef::CTF_NOISE,
                errmsg,
                Some(&ctf_plotter_param.get_config_file()),
            );
        }
        // setupset directives

        self.update_directive_directive_map_directive_def_string_buffer_string(
            directive_map,
            &DirectiveDef::SCOPE_TEMPLATE,
            errmsg,
            Some(&meta_data.get_orig_scope_template()),
        );
        self.update_directive_directive_map_directive_def_string_buffer_string(
            directive_map,
            &DirectiveDef::SYSTEM_TEMPLATE,
            errmsg,
            Some(&meta_data.get_orig_system_template()),
        );
        self.update_directive_directive_map_directive_def_string_buffer_string(
            directive_map,
            &DirectiveDef::USER_TEMPLATE,
            errmsg,
            Some(&meta_data.get_orig_user_template()),
        );

        // runtime
        // prepend = DirectiveType.RUNTIME.toString() + AutodocTokenizer.SEPARATOR_CHAR;

        // Preprocessing
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::REMOVE_XRAYS,
            errmsg,
            file_type::CLASS
                .eraser_log
                .get_file(Some(self), Some(cur_axis_id))
                .is_some_and(|file| file.exists()),
        );
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::ARCHIVE_ORIGINAL,
            errmsg,
            file_type::CLASS
                .original_raw_stack
                .get_file(Some(self), Some(cur_axis_id))
                .is_some_and(|file| file.exists()),
        );
        // Fiducials module
        // Coarse alignment
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::FIDUCIALLESS,
            errmsg,
            meta_data.is_fiducialess(cur_axis_id),
        );
        // Tracking choices
        let tracking_method_value =
            TrackingMethod::to_directive_value(Some(&meta_data.get_track_method(cur_axis_id)));
        let seed_value = tracking_method::SEED.get_value();
        self.update_directive_directive_map_directive_def_string_buffer_const_etomo_number_const_etomo_number(
            directive_map,
            &DirectiveDef::TRACKING_METHOD,
            errmsg,
            tracking_method_value.as_ref().map(|value| &**value),
            Some(&*seed_value),
        );
        self.update_directive_directive_map_directive_def_string_buffer_string_string(
            directive_map,
            &DirectiveDef::SEEDING_METHOD,
            errmsg,
            SeedingMethod::to_directive_value(meta_data, cur_axis_id).as_deref(),
            Some(&seeding_method::MANUAL.get_value().to_string()),
        );
        // Beadtracking
        // numberOfRuns - cannot update
        // Auto seed finding
        // rawBoundaryModel - cannot update
        // RAPTOR parameters
        let _ = meta_data.get_track_raptor_use_raw_stack();
        self.update_directive_directive_map_directive_def_string_buffer_boolean_boolean(
            directive_map,
            &DirectiveDef::USE_ALIGNED_STACK,
            errmsg,
            !meta_data.get_track_raptor_use_raw_stack(),
            true,
        );
        self.update_directive_directive_map_directive_def_string_buffer_string(
            directive_map,
            &DirectiveDef::NUMBER_OF_MARKERS,
            errmsg,
            Some(&meta_data.get_track_raptor_mark()),
        );
        // Patch tracking
        // rawBoundaryModel - cannot update
        // contourPieces - cannot update
        // adjustTiltAngles - cannot update
        // Alignment
        // enableStretching - cannot update
        // Tomogram Positioning - Positioning module
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::WHOLE_TOMOGRAM,
            errmsg,
            meta_data.is_whole_tomogram_sample(cur_axis_id),
        );
        self.update_directive_directive_map_directive_def_string_buffer_int_int(
            directive_map,
            &DirectiveDef::BIN_BY_FACTOR_FOR_POSITIONING,
            errmsg,
            meta_data.get_pos_binning(cur_axis_id),
            3,
        );
        self.update_directive_directive_map_directive_def_string_buffer_const_etomo_number(
            directive_map,
            &DirectiveDef::THICKNESS_FOR_POSITIONING,
            errmsg,
            Some(&*meta_data.get_sample_thickness(cur_axis_id)),
        );
        // Aligned stack module
        // Aligned stack choices
        let process_result_display_factory = self.get_process_result_display_factory(cur_axis_id);
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::CORRECT_CTF,
            errmsg,
            self.get_screen_state(cur_axis_id).get_button_state(
                process_result_display_factory
                    .get_ctf_correction()
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::ERASE_GOLD,
            errmsg,
            self.get_screen_state(cur_axis_id).get_button_state(
                process_result_display_factory
                    .get_ccd_eraser_beads()
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::FILTER_STACK,
            errmsg,
            self.get_screen_state(cur_axis_id).get_button_state(
                process_result_display_factory
                    .get_filter()
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
        // Aligned Stack Parameters
        let linear_interpolation: bool;
        let bin_by_factor: i32;
        if montage {
            self.get_com_script_manager().load_blend(cur_axis_id);
            let param = self.get_com_script_manager().get_blend_param(cur_axis_id);
            linear_interpolation = param.is_linear_interpolation();
            bin_by_factor = param.get_bin_by_factor().get_int();
        } else {
            self.get_com_script_manager().load_newst(cur_axis_id);
            let param = self
                .get_com_script_manager()
                .get_newst_com_newst_param(cur_axis_id);
            linear_interpolation = param.is_linear_interpolation();
            bin_by_factor = param.get_bin_by_factor();
        }
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::LINEAR_INTERPOLATION,
            errmsg,
            linear_interpolation,
        );
        self.update_directive_directive_map_directive_def_string_buffer_int_int(
            directive_map,
            &DirectiveDef::BIN_BY_FACTOR_FOR_ALIGNED_STACK,
            errmsg,
            bin_by_factor,
            1,
        );
        self.update_directive_directive_map_directive_def_string_buffer_fortran_input_string(
            directive_map,
            &DirectiveDef::SIZE_IN_X_AND_Y,
            errmsg,
            Some(&meta_data.get_size_to_output_in_x_and_y(cur_axis_id)),
        );
        // CTFplotting module
        self.update_directive_directive_map_directive_def_string_buffer_fortran_input_string(
            directive_map,
            &DirectiveDef::AUTO_FIT_RANGE_AND_STEP,
            errmsg,
            Some(&meta_data.get_stack_ctf_auto_fit_range_and_step(cur_axis_id)),
        );
        // GoldErasing module
        // Java resolves `updateDirective(..., int)` to the `double` overload (there is
        // no single-int overload), so the binning is widened to double.
        self.update_directive_directive_map_directive_def_string_buffer_double(
            directive_map,
            &DirectiveDef::BINNING_FOR_GOLD_ERASING,
            errmsg,
            meta_data.get_stack_3d_find_binning(cur_axis_id) as f64,
        );
        // extraDiameter - cannot update
        self.update_directive_directive_map_directive_def_string_buffer_string(
            directive_map,
            &DirectiveDef::THICKNESS_FOR_GOLD_ERASING,
            errmsg,
            Some(&meta_data.get_stack_3d_find_thickness(cur_axis_id)),
        );
        // Reconstruction
        // extraThickness - cannot update
        // binnedThickness - cannot update
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::USE_SIRT,
            errmsg,
            self.get_screen_state(cur_axis_id).get_button_state(
                process_result_display_factory
                    .get_use_sirt()
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
        self.update_directive_directive_map_directive_def_string_buffer_boolean(
            directive_map,
            &DirectiveDef::DO_BACKPROJ_ALSO,
            errmsg,
            self.get_screen_state(cur_axis_id).get_button_state(
                process_result_display_factory
                    .get_tilt(DialogType::TomogramGeneration)
                    .get_button_state_key()
                    .as_deref(),
            ),
        );
        // Postprocessing
        // Trimvol module
        // module = "Trimvol" + AutodocTokenizer.SEPARATOR_CHAR
        // + DirectiveFile.RUN_TIME_ANY_AXIS_NAME + AutodocTokenizer.SEPARATOR_CHAR;
        // reorient
        // `TrimvolReorientation.toDirectiveValue` returns int; as above, Java resolves
        // the call to the `double` overload.
        self.update_directive_directive_map_directive_def_string_buffer_double(
            directive_map,
            &DirectiveDef::REORIENT,
            errmsg,
            TrimvolReorientation::to_directive_value(meta_data) as f64,
        );
        // thickness - cannot update
        // sizeInX - cannot update
        // sizeInY - cannot update
        // scaleFromX - cannot update
        // scaleFromY - cannot update
        // scaleFromZ - cannot update
    }

    /// Java `getDialog(DialogType, AxisID)` (ApplicationManager.java:4094).
    fn get_dialog(
        &'static self,
        dialog_type: Option<DialogType>,
        axis_id: AxisID,
    ) -> Option<Rc<dyn ProcessDialogVirtual>> {
        let Some(dialog_type) = dialog_type else {
            return None;
        };
        if dialog_type == DialogType::PreProcessing {
            if axis_id == AxisID::Second {
                return self
                    .pre_proc_dialog_b
                    .get()
                    .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
            } else {
                return self
                    .pre_proc_dialog_a
                    .get()
                    .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
            }
        } else if dialog_type == DialogType::CoarseAlignment {
            return self
                .map_coarse_align_dialog(axis_id)
                .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
        } else if dialog_type == DialogType::FiducialModel {
            if axis_id == AxisID::Second {
                return self
                    .fiducial_model_dialog_b
                    .get()
                    .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
            } else {
                return self
                    .fiducial_model_dialog_a
                    .get()
                    .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
            }
        } else if dialog_type == DialogType::FineAlignment {
            if axis_id == AxisID::Second {
                return self
                    .fine_alignment_dialog_b
                    .get()
                    .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
            } else {
                return self
                    .fine_alignment_dialog_a
                    .get()
                    .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
            }
        } else if dialog_type == DialogType::TomogramCombination {
            return self
                .tomogram_combination_dialog
                .get()
                .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
        } else if dialog_type == DialogType::PostProcessing {
            return self
                .post_processing_dialog
                .get()
                .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
        } else if dialog_type == DialogType::CleanUp {
            return self
                .clean_up_dialog
                .get()
                .map(|dialog| dialog as Rc<dyn ProcessDialogVirtual>);
        }
        None
    }

    /// Java `doneAlignmentEstimationDialog(AxisID)` (ApplicationManager.java:4140).
    pub fn done_alignment_estimation_dialog(&'static self, axis_id: AxisID) {
        // Set a reference to the correct object
        let mut fine_alignment_dialog = if axis_id == AxisID::Second {
            self.fine_alignment_dialog_b.get()
        } else {
            self.fine_alignment_dialog_a.get()
        };
        // Java calls `saveAlignmentEstimationDialog(fineAlignmentDialog, axisID)`
        // unconditionally (ApplicationManager.java:4148), which throws a
        // NullPointerException when the dialog is not open.  Fixed in translation:
        // a null dialog is not saved; the clean up below still runs.
        if let Some(fine_alignment_dialog) = &fine_alignment_dialog {
            self.save_alignment_estimation_dialog(fine_alignment_dialog, axis_id);
        }
        // Clean up the existing dialog
        if axis_id == AxisID::Second {
            self.fine_alignment_dialog_b.set(None);
        } else {
            self.fine_alignment_dialog_a.set(None);
        }
        fine_alignment_dialog = None;
        let _ = fine_alignment_dialog;
    }

    /// Java `saveAlignmentEstimationDialog(AlignmentEstimationDialog, AxisID)`
    /// (ApplicationManager.java:4163).
    pub fn save_alignment_estimation_dialog(
        &'static self,
        fine_alignment_dialog: &Rc<AlignmentEstimationDialog>,
        axis_id: AxisID,
    ) {
        self.set_advanced_dialog_type_axis_id_boolean(
            fine_alignment_dialog.get_dialog_type(),
            axis_id,
            fine_alignment_dialog.is_advanced(),
        );
        let exit_state = fine_alignment_dialog.get_exit_state();
        if exit_state == DialogExitState::Cancel {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(axis_id);
            }
        } else {
            fine_alignment_dialog.get_parameters_base_screen_state(self.get_screen_state(axis_id));
            // Get the user input data from the dialog box
            self.update_align_com(axis_id, false);
            self.update_restrictalign_com(axis_id, false);
            if exit_state == DialogExitState::Postpone {
                self.get_recon_process_track()
                    .set_fine_alignment_state(ProcessState::InProgress, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_fine_alignment_state(ProcessState::InProgress, axis_id);
                }
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.show_blank_process(axis_id);
                }
            } else if exit_state != DialogExitState::Save {
                self.get_recon_process_track()
                    .set_fine_alignment_state(ProcessState::Complete, axis_id);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_fine_alignment_state(ProcessState::Complete, axis_id);
                }
                // Check to see if the user wants to keep any coarse aligned imods
                // open
                self.close_imod(
                    Some(imod_manager::COARSE_ALIGNED_KEY),
                    Some(axis_id),
                    Some("coarsely aligned stack"),
                    false,
                );
                self.close_imod(
                    Some(imod_manager::FIDUCIAL_MODEL_KEY),
                    Some(axis_id),
                    Some("fiducial model"),
                    false,
                );
                // `getUIExpert(TOMOGRAM_POSITIONING, ...)` never returns null.
                if let Some(expert) =
                    self.get_ui_expert(Some(DialogType::TomogramPositioning), axis_id)
                {
                    expert.open_dialog();
                }
            }
            self.save_storables(Some(axis_id));
        }
    }

    /// Java `closeImods(String, AxisID, String)` (ApplicationManager.java:4195).
    ///
    /// Not overloaded in `ApplicationManager`, so it keeps the plain name; it
    /// shadows `BaseManager::close_imods` (a different Java overload), which callers
    /// in this module reach as `BaseManager::close_imods(self, ...)`.
    pub fn close_imods(
        &'static self,
        key: Option<&str>,
        axis_id: AxisID,
        description: Option<&str>,
    ) {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            eprintln!(
                "closeImods:key:{},axisID:{},description:{}",
                key.unwrap_or("null"),
                axis_id,
                description.unwrap_or("null")
            );
        }
        // Check to see if the user wants to keep any imods open. Don't log message.
        let result = (|| -> Result<(), ImodManagerException> {
            // A null key is never open (BaseImodManager.get finds no state).
            let Some(key) = key else {
                return Ok(());
            };
            if self
                .get_imod_manager()
                .is_open_string_axis_id(key, Some(axis_id))?
            {
                if !etomo_director::ARGUMENTS
                    .lock()
                    .unwrap()
                    .is_auto_close_3dmod()
                {
                    let message = vec![
                        format!(
                            "{}(s) are open in 3dmod.{}",
                            description.unwrap_or("null"),
                            self.get_file_lock_message(Some("  "))
                        ),
                        "Should they be closed?".to_owned(),
                    ];
                    if UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                            None,
                            &message,
                            Some(axis_id),
                        )
                    }) {
                        self.get_imod_manager().quit_all(key, Some(axis_id))?;
                        self.release_file();
                    }
                } else {
                    self.get_imod_manager().quit_all(key, Some(axis_id))?;
                    self.release_file();
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "System Process Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java private `closeImod(String, String, boolean)` (ApplicationManager.java:4234).
    fn close_imod_string_string_boolean(
        &'static self,
        key: Option<&str>,
        description: Option<&str>,
        file_lock_risk: bool,
    ) {
        let _ = file_lock_risk;
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            eprintln!(
                "closeImod:key:{},description:{}",
                key.unwrap_or("null"),
                description.unwrap_or("null")
            );
        }
        let Some(key) = key else {
            return;
        };
        // Check to see if the user wants to keep any imods open
        let result = (|| -> Result<(), ImodManagerException> {
            if self.get_imod_manager().is_open_string(key)? {
                if !etomo_director::ARGUMENTS
                    .lock()
                    .unwrap()
                    .is_auto_close_3dmod()
                {
                    let message = vec![
                        format!(
                            "The {} is open in 3dmod.{}",
                            description.unwrap_or("null"),
                            self.get_file_lock_message(Some("  "))
                        ),
                        "Should it be closed?".to_owned(),
                    ];
                    if UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                            None,
                            &message,
                            Some(AxisID::Only),
                        )
                    }) {
                        self.get_imod_manager().quit_string(key)?;
                        self.release_file();
                    }
                } else {
                    self.get_imod_manager().quit_string(key)?;
                    self.release_file();
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(AxisID::Only),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(AxisID::Only),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "System Process Exception",
                        Some(AxisID::Only),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `closeImod(String, AxisID, AxisID, String)` (ApplicationManager.java:4286).
    /// Allow a 3dmod to be closed even though it has an axis different from the
    /// axis of the frame where the message should be popped up.
    pub fn close_imod_string_axis_id_axis_id_string(
        &'static self,
        key: Option<&str>,
        frame_axis_id: AxisID,
        imod_axis_id: AxisID,
        description: Option<&str>,
    ) {
        if etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            eprintln!(
                "closeImod:key:{},frameAxisID:{},imodAxisID:{},description:{}",
                key.unwrap_or("null"),
                frame_axis_id,
                imod_axis_id,
                description.unwrap_or("null")
            );
        }
        // Check to see if the user wants to keep any imods open
        let result = (|| -> Result<(), ImodManagerException> {
            // A null key is never open (BaseImodManager.get finds no state).
            let Some(key) = key else {
                return Ok(());
            };
            if self
                .get_imod_manager()
                .is_open_string_axis_id(key, Some(imod_axis_id))?
            {
                if !etomo_director::ARGUMENTS
                    .lock()
                    .unwrap()
                    .is_auto_close_3dmod()
                {
                    let message = vec![
                        format!(
                            "The {} is open in 3dmod.{}",
                            description.unwrap_or("null"),
                            self.get_file_lock_message(Some("  "))
                        ),
                        "Should it be closed?".to_owned(),
                    ];
                    if UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_yes_no_dialog_base_manager_string_array_axis_id(
                            None,
                            &message,
                            Some(frame_axis_id),
                        )
                    }) {
                        self.get_imod_manager()
                            .quit_string_axis_id(key, Some(imod_axis_id))?;
                        self.release_file();
                    }
                } else {
                    self.get_imod_manager()
                        .quit_string_axis_id(key, Some(imod_axis_id))?;
                    self.release_file();
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(frame_axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(frame_axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "System Process Exception",
                        Some(frame_axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `fineAlignment(AxisID, ProcessResultDisplay, ProcessSeries)`
    /// (ApplicationManager.java:4336).  Execute the fine alignment script
    /// (align.com) for the appropriate axis.  This will also reset the
    /// fiducialess alignment flag; setFiducialAlign does not need to be called
    /// because the necessary copies are called at the end of the align script.
    pub fn fine_alignment(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Set a reference to the correct object
        let fine_alignment_dialog: Option<Rc<AlignmentEstimationDialog>> = self
            .get_dialog(Some(DialogType::FineAlignment), axis_id)
            .and_then(|dialog| {
                (dialog as Rc<dyn Any>)
                    .downcast::<AlignmentEstimationDialog>()
                    .ok()
            });
        if fine_alignment_dialog
            .as_ref()
            .is_some_and(|dialog| !dialog.is_valid())
        {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let Some(tiltalign_param) = self.update_align_com(axis_id, true) else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        self.get_recon_process_track()
            .set_fine_alignment_state(ProcessState::InProgress, axis_id);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_fine_alignment_state(ProcessState::InProgress, axis_id);
        }
        let thread_name = match self.get_process_mgr().fine_alignment(
            Arc::new(tiltalign_param),
            axis_id,
            process_result_display_ref.clone(),
            process_series
                .clone()
                .map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute align{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return;
            }
        };
        self.get_meta_data()
            .set_fiducialess_alignment(axis_id, false);
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Aligning stack"),
                axis_id,
                Some(&ProcessName::ALIGN),
            );
        }
    }

    /// Java `restrictalign(AxisID, ProcessSeries)` (ApplicationManager.java:4377).
    pub fn restrictalign(
        &'static self,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        // Set a reference to the correct object
        let dialog: Option<Rc<AlignmentEstimationDialog>> = self
            .get_dialog(Some(DialogType::FineAlignment), axis_id)
            .and_then(|dialog| {
                (dialog as Rc<dyn Any>)
                    .downcast::<AlignmentEstimationDialog>()
                    .ok()
            });
        if dialog.as_ref().is_some_and(|dialog| !dialog.is_valid()) {
            return;
        }
        if self.update_align_com(axis_id, true).is_none() {
            return;
        }
        let Some(param) = self.update_restrictalign_com(axis_id, true) else {
            return;
        };
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => {
                let process_series = ProcessSeries::new(
                    self,
                    axis_id,
                    Some(DialogType::FineAlignment),
                    Some("restrictalign"),
                );
                process_series
                    .borrow_mut()
                    .set_last_process_task(Rc::new(Task::ReloadAlignCom));
                process_series
            }
        };
        self.get_recon_process_track()
            .set_fine_alignment_state(ProcessState::InProgress, axis_id);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_fine_alignment_state(ProcessState::InProgress, axis_id);
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Restricting alignment variables"),
                axis_id,
                Some(&ProcessName::RESTRICTALIGN),
            );
        }
        match self.get_process_mgr().restrictalign(
            Arc::new(param),
            axis_id,
            Some(Arc::new(EdtRef::new(process_series)) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => self.set_thread_name(Some(&thread_name), Some(axis_id)),
            Err(e) => {
                eprintln!("{e}");
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        Some(ProcessEndState::Failed),
                    );
                }
                let message = vec!["Can not execute restrictalign".to_owned(), e.0.clone()];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute script",
                        Some(axis_id),
                    )
                });
                return;
            }
        }
    }

    /// Java private `reloadAlignCom(AxisID, ProcessSeries)` (ApplicationManager.java:4414).
    fn reload_align_com(
        &'static self,
        axis_id: AxisID,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id(Some("Reloading align"), axis_id);
        }
        self.get_com_script_manager().reset_align(axis_id);
        let dialog: Option<Rc<AlignmentEstimationDialog>> = self
            .get_dialog(Some(DialogType::FineAlignment), axis_id)
            .and_then(|dialog| {
                (dialog as Rc<dyn Any>)
                    .downcast::<AlignmentEstimationDialog>()
                    .ok()
            });
        match dialog {
            None => self.open_fine_alignment_dialog(axis_id),
            Some(dialog) => {
                self.get_com_script_manager().load_align(axis_id);
                let param = self.get_com_script_manager().get_tiltalign_param(axis_id);
                // If this is a montage, then binning can only be 1, so no need to upgrade
                dialog.set_tiltalign_params(&param);
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel
                .stop_progress_bar_axis_id_process_end_state(axis_id, Some(ProcessEndState::Done));
        }
        if let Some(process_series) = &process_series {
            process_series.borrow().end_series();
        }
    }

    /// Java `imodViewResiduals(AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:4437).  Open 3dmod with the new fidcuial model.
    pub fn imod_view_residuals(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        let fiducial_model = format!(
            "{}{}.resmod",
            self.get_meta_data().get_dataset_name(),
            axis_id.get_extension()
        );
        let result = (|| -> Result<(), ImodManagerException> {
            let imod_manager = self.get_imod_manager();
            imod_manager.set_preserve_contrast(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                true,
            )?;
            imod_manager.set_beadfixer_mode(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                Some(BeadFixerMode::ResidualMode),
            )?;
            imod_manager.set_open_log_off(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))?;
            let tilt_file = dataset_files::get_raw_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                imod_manager.set_tilt_file(
                    imod_manager::COARSE_ALIGNED_KEY,
                    Some(axis_id),
                    tilt_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                imod_manager.reset_tilt_file(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))?;
            }
            imod_manager.open_string_axis_id_string_run3dmod_menu_options(
                imod_manager::COARSE_ALIGNED_KEY,
                Some(axis_id),
                Some(&fiducial_model),
                Some(menu_options),
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        &format!(
                            "Can't open 3dmod on coarse aligned stack with model: {fiducial_model}"
                        ),
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodViewModel(AxisID, FileType)` (ApplicationManager.java:4475).
    /// Open 3dmodv with a model.
    pub fn imod_view_model(&'static self, axis_id: AxisID, model_file_type: &FileType) {
        // Fixed in translation: a file type with no 3dmod key makes Java throw
        // NullPointerException in ImodManager.getPrivateKey; nothing is opened.
        let Some(key) = model_file_type.get_imod_manager_key() else {
            return;
        };
        let result = (|| -> Result<(), ImodManagerException> {
            self.get_imod_manager().open_string_axis_id_string(
                key,
                Some(axis_id),
                model_file_type
                    .get_file_name(Some(self), Some(axis_id))
                    .as_deref(),
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        &format!(
                            "Can't open 3dmod on {}",
                            model_file_type
                                .get_file_name(Some(self), Some(axis_id))
                                .unwrap_or("null".to_owned())
                        ),
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodFineAlign3dFind(AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:4495).
    pub fn imod_fine_align3d_find(
        &'static self,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
    ) {
        // Get the correct ImodManager key. The file to bring up is whatever file
        // was last used as input for tilt_3dfind.
        let key: Option<&'static str> = if self
            .get_state()
            .is_stack_using_newst_or_blend_3d_find_output(axis_id)
        {
            file_type::CLASS
                .newst_or_blend_3d_find_output
                .get_imod_manager_key()
        } else {
            file_type::CLASS.aligned_stack.get_imod_manager_key()
        };
        let Some(key) = key else {
            return;
        };
        let result = (|| -> Result<(), ImodManagerException> {
            let tilt_file = dataset_files::get_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                self.get_imod_manager().set_tilt_file(
                    key,
                    Some(axis_id),
                    tilt_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                self.get_imod_manager()
                    .reset_tilt_file(key, Some(axis_id))?;
            }
            self.get_imod_manager()
                .open_string_axis_id_run3dmod_menu_options(
                    key,
                    Some(axis_id),
                    Some(menu_options),
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        &format!("Can't open 3dmod on {}", key),
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodFineAlign(AxisID, Run3dmodMenuOptions)` (ApplicationManager.java:4538).
    /// Open 3dmod to view the fine aligned stack or the fine aligned stack for
    /// findbeads3d.
    pub fn imod_fine_align(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        let result = (|| -> Result<(), ImodManagerException> {
            let tilt_file = dataset_files::get_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                self.get_imod_manager().set_tilt_file(
                    imod_manager::FINE_ALIGNED_KEY,
                    Some(axis_id),
                    tilt_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                self.get_imod_manager()
                    .reset_tilt_file(imod_manager::FINE_ALIGNED_KEY, Some(axis_id))?;
            }
            self.get_imod_manager()
                .open_string_axis_id_run3dmod_menu_options(
                    imod_manager::FINE_ALIGNED_KEY,
                    Some(axis_id),
                    Some(menu_options),
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        &format!("Can't open 3dmod on {}", imod_manager::FINE_ALIGNED_KEY),
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodMTFFilter(AxisID, Run3dmodMenuOptions)` (ApplicationManager.java:4570).
    /// Open 3dmod to view the MTF filter results.
    pub fn imod_mtf_filter(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        let result = (|| -> Result<(), ImodManagerException> {
            let tilt_file = dataset_files::get_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                self.get_imod_manager().set_tilt_file(
                    imod_manager::MTF_FILTER_KEY,
                    Some(axis_id),
                    tilt_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                self.get_imod_manager()
                    .reset_tilt_file(imod_manager::MTF_FILTER_KEY, Some(axis_id))?;
            }
            self.get_imod_manager()
                .open_string_axis_id_run3dmod_menu_options(
                    imod_manager::MTF_FILTER_KEY,
                    Some(axis_id),
                    Some(menu_options),
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "Can't open 3dmod on MTF filter results",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodCtfCorrection(AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:4596).
    pub fn imod_ctf_correction(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        let result = (|| -> Result<(), ImodManagerException> {
            let tilt_file = dataset_files::get_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                self.get_imod_manager().set_tilt_file(
                    imod_manager::CTF_CORRECTION_KEY,
                    Some(axis_id),
                    tilt_file
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .as_deref(),
                )?;
            } else {
                self.get_imod_manager()
                    .reset_tilt_file(imod_manager::CTF_CORRECTION_KEY, Some(axis_id))?;
            }
            self.get_imod_manager()
                .open_string_axis_id_run3dmod_menu_options(
                    imod_manager::CTF_CORRECTION_KEY,
                    Some(axis_id),
                    Some(menu_options),
                )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        &format!(
                            "Can't open 3dmod on {} results",
                            ProcessName::CTF_CORRECTION
                        ),
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `coordFileExists()` (ApplicationManager.java:4625).
    pub fn coord_file_exists(&'static self) -> bool {
        dataset_files::get_transfer_fid_coord_file(self).exists()
    }

    /// Java `msgExcludeViewsSucceeded(AxisID)` (ApplicationManager.java:4651).
    /// The exclude views value must be removed because, if excludeviews was rerun
    /// with the values from the previous run, more views will be removed.
    pub fn msg_exclude_views_succeeded(&'static self, axis_id: AxisID) {
        // Posted: Java calls this on the process thread.
        invoke_later(move || {
            if let Some(setup_recon_ui_harness) = self.setup_recon_ui_harness.get() {
                setup_recon_ui_harness.msg_exclude_views_succeeded(axis_id, true, false);
            }
            self.get_meta_data().reset_exclude_projections(axis_id);
        });
    }

    /// Java `transferfid(AxisID, ProcessResultDisplay, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions, DialogType)`
    /// (ApplicationManager.java:4663).  Transfer the fiducial to the specified axis.
    pub fn transferfid(
        &'static self,
        dest_axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, dest_axis_id, Some(dialog_type), Some("transferfid")),
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Set a reference to the correct object
        let fiducial_model_dialog;
        let from_axis;
        if dest_axis_id == AxisID::Second {
            fiducial_model_dialog = self.fiducial_model_dialog_b.get();
            from_axis = AxisID::First;
        } else {
            fiducial_model_dialog = self.fiducial_model_dialog_a.get();
            from_axis = AxisID::Second;
        }
        if dest_axis_id != AxisID::Only
            && !utilities::file_exists(
                self,
                Some("fid.xyz"),
                Some(if dest_axis_id == AxisID::First {
                    AxisID::Second
                } else {
                    AxisID::First
                }),
            )
        {
            UI_HARNESS.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &format!(
                        "It is recommended that you run Fine Alignment on axis {} at least once",
                        if dest_axis_id == AxisID::First {
                            "B"
                        } else {
                            "A"
                        }
                    ),
                    "Warning",
                    Some(dest_axis_id),
                )
            });
        }
        let Some(fiducial_model_dialog) = fiducial_model_dialog else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        // Pull values used in autofidseed from the "from" axis.
        if self
            .get_com_script_manager()
            .load_autofidseed(from_axis, false)
        {
            let from_autofidseed_param = self
                .get_com_script_manager()
                .get_autofidseed_param(from_axis);
            fiducial_model_dialog
                .set_parameters_autofidseed_param_boolean(&from_autofidseed_param, true);
        }
        self.get_com_script_manager().load_track(from_axis);
        let mut from_track_param = self.get_com_script_manager().get_beadtrack_param(from_axis);
        fiducial_model_dialog.set_beadtrack_params(&mut from_track_param, true);
        //
        let mut transferfid_param = TransferfidParam::new(self, dest_axis_id);
        // Setup the default parameters depending upon the axis to transfer
        // the fiducials from
        let dataset_name = self.get_meta_data().get_dataset_name();
        transferfid_param.set_dataset_name(Some(&dataset_name));
        if dest_axis_id == AxisID::First {
            transferfid_param.set_b_to_a(true);
        } else {
            transferfid_param.set_b_to_a(false);
        }
        // Get any user specified changes
        if !fiducial_model_dialog
            .get_transfer_fid_params_transferfid_param_boolean(&mut transferfid_param, true)
        {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        let thread_name = match self.get_process_mgr().transfer_fiducials(
            Arc::new(transferfid_param),
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series))) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    "Can not execute transferfid command".to_owned(),
                    e.0.clone(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute command",
                        Some(dest_axis_id),
                    )
                });
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(dest_axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Transferring fiducials"),
                dest_axis_id,
                Some(&ProcessName::TRANSFERFID),
            );
        }
        self.update_dialog_fiducial_model_dialog_axis_id(
            Some(Rc::clone(&fiducial_model_dialog)),
            dest_axis_id,
        );
    }

    /// Java private `updateRestrictalignCom(AxisID, boolean)`
    /// (ApplicationManager.java:4741).
    fn update_restrictalign_com(
        &'static self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<RestrictalignParam> {
        let dialog: Rc<AlignmentEstimationDialog> = self
            .get_dialog(Some(DialogType::FineAlignment), axis_id)
            .and_then(|dialog| {
                (dialog as Rc<dyn Any>)
                    .downcast::<AlignmentEstimationDialog>()
                    .ok()
            })?;
        // `if (param == null) return null`: the translated
        // `ComScriptManager.getRestrictAlignParam` always returns a param.
        let mut param = self
            .get_com_script_manager()
            .get_restrict_align_param(axis_id);
        // From directive files
        param.set_skip_beam_tilt_with_one_rot(Some(
            &self
                .get_meta_data()
                .get_skip_beam_tilt_with_one_rot(axis_id),
        ));
        if !dialog.get_parameters_restrictalign_param_boolean(&mut param, do_validation) {
            return None;
        }
        self.get_com_script_manager()
            .save_restrict_align(&param, axis_id);

        Some(param)
    }

    /// Java private `updateAlignCom(AxisID, boolean)` (ApplicationManager.java:4767).
    /// Updates the align{|a|b}.com scripts with the parameters from the alignment
    /// estimation dialog.  This also updates the local alignment state of the
    /// appropriate tilt files.
    fn update_align_com(
        &'static self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<TiltalignParam> {
        let fine_alignment_dialog: Option<Rc<AlignmentEstimationDialog>> = self
            .get_dialog(Some(DialogType::FineAlignment), axis_id)
            .and_then(|dialog| {
                (dialog as Rc<dyn Any>)
                    .downcast::<AlignmentEstimationDialog>()
                    .ok()
            });
        let Some(fine_alignment_dialog) = fine_alignment_dialog else {
            UI_HARNESS.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    "Can not update align?.com without an active alignment dialog",
                    "Program logic error",
                    Some(axis_id),
                )
            });
            return None;
        };
        let result = (|| -> Result<Option<TiltalignParam>, TiltalignParamsException> {
            let mut tiltalign_param = self.get_com_script_manager().get_tiltalign_param(axis_id);
            fine_alignment_dialog.get_parameters_meta_data(self.get_meta_data());
            if !fine_alignment_dialog.get_tiltalign_params(&mut tiltalign_param, do_validation)? {
                return Ok(None);
            }
            UIExpertUtilities::INSTANCE.roll_align_com_angles(self, axis_id);
            self.get_com_script_manager()
                .save_align(&tiltalign_param, axis_id);
            // Update the tilt.com script with the dependent parameters
            self.update_tilt_com_const_tiltalign_param_axis_id(&tiltalign_param, axis_id);
            // update xfproduct in align.com
            let mut xfproduct_param = self
                .get_com_script_manager()
                .get_xfproduct_in_align(axis_id);
            xfproduct_param
                .set_scale_shifts(
                    UIExpertUtilities::INSTANCE.get_stack_binning_base_manager_axis_id_file_type(
                        self,
                        axis_id,
                        &file_type::CLASS.prealigned_stack,
                    ),
                )
                .map_err(TiltalignParamsException::FortranInputSyntaxException)?;
            self.get_com_script_manager()
                .save_xfproduct_in_align(&xfproduct_param, axis_id);
            if fine_alignment_dialog.get_exit_state() != DialogExitState::Save {
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_fine_alignment_state(ProcessState::InProgress, axis_id);
                }
            }
            Ok(Some(tiltalign_param))
        })();
        match result {
            Ok(tiltalign_param) => tiltalign_param,
            // catch (final NumberFormatException except)
            Err(TiltalignParamsException::NumberFormatException(message)) => {
                let error_message = vec!["Tiltalign Parameter Syntax Error".to_owned(), message];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Tiltalign Parameter Syntax Error",
                        Some(axis_id),
                    )
                });
                None
            }
            // catch (final FortranInputSyntaxException except)
            Err(TiltalignParamsException::FortranInputSyntaxException(except)) => {
                let error_message = vec![
                    "Tiltalign Parameter Syntax Error".to_owned(),
                    except.get_new_string().to_owned(),
                    except.get_message().unwrap_or("null").to_owned(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Tiltalign Parameter Syntax Error",
                        Some(axis_id),
                    )
                });
                None
            }
        }
    }

    /// Java `getUIExpert(DialogType, AxisID)` (ApplicationManager.java:4817).
    /// `dialogType` is nullable in Java (`getCurrentDialogType`), and a null type
    /// returns null.
    pub fn get_ui_expert(
        &'static self,
        dialog_type: Option<DialogType>,
        axis_id: AxisID,
    ) -> Option<Rc<dyn UIExpert>> {
        if dialog_type == Some(DialogType::TomogramPositioning) {
            if axis_id == AxisID::Second {
                if !self.tomogram_positioning_expert_b.is_some() {
                    self.tomogram_positioning_expert_b
                        .set(Some(TomogramPositioningExpert::new(
                            self,
                            self.main_panel.get(),
                            Some(self.get_recon_process_track()),
                            axis_id,
                            self.get_meta_data().get_axis_type(),
                        )));
                }
                return self
                    .tomogram_positioning_expert_b
                    .get()
                    .map(|expert| expert as Rc<dyn UIExpert>);
            }
            if !self.tomogram_positioning_expert_a.is_some() {
                self.tomogram_positioning_expert_a
                    .set(Some(TomogramPositioningExpert::new(
                        self,
                        self.main_panel.get(),
                        Some(self.get_recon_process_track()),
                        axis_id,
                        self.get_meta_data().get_axis_type(),
                    )));
            }
            return self
                .tomogram_positioning_expert_a
                .get()
                .map(|expert| expert as Rc<dyn UIExpert>);
        } else if dialog_type == Some(DialogType::FinalAlignedStack) {
            if axis_id == AxisID::Second {
                if !self.final_aligned_stack_expert_b.is_some() {
                    self.final_aligned_stack_expert_b
                        .set(Some(FinalAlignedStackExpert::new(
                            self,
                            self.main_panel.get(),
                            Some(self.get_recon_process_track()),
                            axis_id,
                        )));
                }
                return self
                    .final_aligned_stack_expert_b
                    .get()
                    .map(|expert| expert as Rc<dyn UIExpert>);
            }
            if !self.final_aligned_stack_expert_a.is_some() {
                self.final_aligned_stack_expert_a
                    .set(Some(FinalAlignedStackExpert::new(
                        self,
                        self.main_panel.get(),
                        Some(self.get_recon_process_track()),
                        axis_id,
                    )));
            }
            return self
                .final_aligned_stack_expert_a
                .get()
                .map(|expert| expert as Rc<dyn UIExpert>);
        } else if dialog_type == Some(DialogType::TomogramGeneration) {
            if axis_id == AxisID::Second {
                if !self.tomogram_generation_expert_b.is_some() {
                    self.tomogram_generation_expert_b
                        .set(Some(TomogramGenerationExpert::new(
                            self,
                            self.main_panel.get(),
                            Some(self.get_recon_process_track()),
                            axis_id,
                        )));
                }
                return self
                    .tomogram_generation_expert_b
                    .get()
                    .map(|expert| expert as Rc<dyn UIExpert>);
            }
            if !self.tomogram_generation_expert_a.is_some() {
                self.tomogram_generation_expert_a
                    .set(Some(TomogramGenerationExpert::new(
                        self,
                        self.main_panel.get(),
                        Some(self.get_recon_process_track()),
                        axis_id,
                    )));
            }
            return self
                .tomogram_generation_expert_a
                .get()
                .map(|expert| expert as Rc<dyn UIExpert>);
        }
        None
    }

    /// Java `getSetupDialogExpert()` (ApplicationManager.java:4863).
    pub fn get_setup_dialog_expert(&self) -> Option<Rc<SetupDialogExpert>> {
        if !etomo_director::ARGUMENTS.lock().unwrap().is_test() {
            // Java `throw new IllegalStateException("only for testing")`: a
            // deliberate guard against a test-only accessor, kept.
            panic!("only for testing");
        }
        self.setup_dialog_expert.get()
    }

    /// Java `createSample(AxisID, ProcessResultDisplay, ProcessSeries,
    /// ConstTiltParam)` (ApplicationManager.java:4873).  Run the sample com script.
    pub fn create_sample(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        tilt_param: Arc<TiltParam>,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        // Make sure we have a current prexg and _nonfid.xf if fiducialess is
        // selected
        if self.get_meta_data().is_fiducialess_alignment(axis_id) {
            self.generate_fiducialess_transforms(axis_id);
        } else {
            match self
                .get_process_mgr()
                .setup_fiducial_align(axis_id, process_result_display_ref.clone())
            {
                Ok(true) => {}
                Ok(false) => {
                    if let Some(process_series) = &process_series {
                        process_series.borrow().end_series();
                    }
                    return Some(ProcessResult::FAILED_TO_START);
                }
                Err(except) => {
                    eprintln!("{except}");
                    let message = vec![
                        "Problem copying fiducial alignment files".to_owned(),
                        except.get_message(),
                    ];
                    UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                            Some(self),
                            &message,
                            "Unable to copy fiducial alignment files",
                            Some(axis_id),
                        )
                    });
                }
            }
        }

        let thread_name = match self.get_process_mgr().create_sample(
            axis_id,
            process_result_display_ref.clone(),
            process_series
                .clone()
                .map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
            tilt_param,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute sample{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Creating sample tomogram"),
                axis_id,
                Some(&ProcessName::SAMPLE),
            );
        }
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java `wholeTomogram(AxisID, ProcessResultDisplay, ProcessSeries,
    /// ConstNewstParam)` (ApplicationManager.java:4923).  Create a whole tomogram
    /// for positioning the tomogram in the volume.
    pub fn whole_tomogram_axis_id_process_result_display_process_series_const_newst_param(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: Arc<NewstParam>,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        // Make sure we have a current prexg and _nonfid.xf if fiducialess is
        // selected
        if self.get_meta_data().is_fiducialess_alignment(axis_id) {
            self.generate_fiducialess_transforms(axis_id);
        } else {
            match self
                .get_process_mgr()
                .setup_fiducial_align(axis_id, process_result_display_ref.clone())
            {
                Ok(true) => {}
                Ok(false) => {
                    if let Some(process_series) = &process_series {
                        process_series.borrow().end_series();
                    }
                    return Some(ProcessResult::FAILED_TO_START);
                }
                Err(except) => {
                    eprintln!("{except}");
                    let message = vec![
                        "Problem copying fiducial alignment files".to_owned(),
                        except.get_message(),
                    ];
                    UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                            Some(self),
                            &message,
                            "Unable to copy fiducial alignment files",
                            Some(axis_id),
                        )
                    });
                }
            }
        }
        let thread_name = match self.get_process_mgr().newst(
            param,
            axis_id,
            process_result_display_ref.clone(),
            process_series
                .clone()
                .map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
            ProcessName::NEWST,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute newst{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java `findSection(AxisID, ProcessResultDisplay, ProcessSeries,
    /// FindSectionParam)` (ApplicationManager.java:4966).
    pub fn find_section(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: Arc<FindSectionParam>,
    ) -> Option<ProcessResult> {
        let _ = process_result_display;
        let thread_name = match self.get_process_mgr().find_section(
            param,
            axis_id,
            process_series
                .clone()
                .map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => thread_name,
            Err(except) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!("Can't run findsection\n{}", except.0),
                        "SystemProcessException",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id(Some("Find Section"), axis_id);
        }
        None
    }

    /// Java `cryoPosition(AxisID, ProcessResultDisplay, ProcessSeries,
    /// CryoPositionParam)` (ApplicationManager.java:4984).
    pub fn cryo_position(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: Arc<CryoPositionParam>,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let thread_name = match self.get_process_mgr().cryo_position(
            param,
            axis_id,
            process_result_display_ref.clone(),
            process_series
                .clone()
                .map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = vec!["Can not execute cryoposition".to_owned(), e.0.clone()];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Running cryoposition"),
                axis_id,
                Some(&ProcessName::CRYO_POSITION),
            );
        }
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java `renameCryoPositionOutput(AxisID)` (ApplicationManager.java:5006).
    pub fn rename_cryo_position_output(&'static self, axis_id: AxisID) {
        let cryo_position_output = file_type::CLASS
            .cryo_position_output
            .get_file(Some(self), Some(axis_id));
        let tilt_output = file_type::CLASS
            .tilt_output
            .get_file(Some(self), Some(axis_id));
        match utilities::rename_file(
            Some(self),
            Some(axis_id),
            cryo_position_output.as_deref(),
            tilt_output.as_deref(),
            false,
            false,
            false,
        ) {
            Ok(_) => {}
            // `catch (final LockException e) {}`
            Err(LogFileError::Lock(_)) => {}
            Err(e) => {
                eprintln!("{e}");
                // Posted: Java calls this on the process thread.
                ui_harness::post_message_dialog(
                    Some(self),
                    format!(
                        "Unable to rename {} to {}.  In order to use the automated positioning functionality going forward, please perform this rename.",
                        cryo_position_output
                            .as_deref()
                            .and_then(|file| file.file_name())
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or("null".to_owned()),
                        tilt_output
                            .as_deref()
                            .and_then(|file| file.file_name())
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or("null".to_owned())
                    ),
                    "Rename Failed".to_owned(),
                    Some(axis_id),
                );
            }
        }
    }

    /// Java `wholeTomogram(AxisID, ProcessResultDisplay, ProcessSeries,
    /// BlendmontParam)` (ApplicationManager.java:5028).  Create a whole tomogram
    /// for positioning the tomogram in the volume.
    pub fn whole_tomogram_axis_id_process_result_display_process_series_blendmont_param(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: Arc<BlendmontParam>,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        // Make sure we have a current prexg and _nonfid.xf if fiducialess is
        // selected
        if self.get_meta_data().is_fiducialess_alignment(axis_id) {
            self.generate_fiducialess_transforms(axis_id);
        } else {
            match self
                .get_process_mgr()
                .setup_fiducial_align(axis_id, process_result_display_ref.clone())
            {
                Ok(true) => {}
                Ok(false) => {
                    if let Some(process_series) = &process_series {
                        process_series.borrow().end_series();
                    }
                    return Some(ProcessResult::FAILED_TO_START);
                }
                Err(except) => {
                    eprintln!("{except}");
                    let message = vec![
                        "Problem copying fiducial alignment files".to_owned(),
                        except.get_message(),
                    ];
                    UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                            Some(self),
                            &message,
                            "Unable to copy fiducial alignment files",
                            Some(axis_id),
                        )
                    });
                }
            }
        }
        let thread_name = match self.get_process_mgr().blend(
            param,
            axis_id,
            process_result_display_ref.clone(),
            process_series
                .clone()
                .map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute newst{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java `imodSample(AxisID, Run3dmodMenuOptions)` (ApplicationManager.java:5076).
    /// Open 3dmod on the sample volume for the specified axis along with the
    /// tomopitch model.
    pub fn imod_sample(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        let result = (|| -> Result<(), ImodManagerException> {
            // It is safe to use open contours in all cases.
            self.get_imod_manager().set_open_contours(
                imod_manager::SAMPLE_KEY,
                Some(axis_id),
                true,
            )?;
            self.get_imod_manager()
                .set_point_limit(imod_manager::SAMPLE_KEY, Some(axis_id), 2)?;
            self.get_imod_manager()
                .open_string_axis_id_run3dmod_menu_options(
                    imod_manager::SAMPLE_KEY,
                    Some(axis_id),
                    Some(menu_options),
                )?;
            self.get_recon_process_track()
                .set_tomogram_positioning_state(ProcessState::InProgress, axis_id);
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_tomogram_positioning_state(ProcessState::InProgress, axis_id);
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "Problem opening sample reconstruction",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodFullSample(AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:5105).  Open 3dmod on the full volume along with
    /// the tomopitch model.
    pub fn imod_full_sample(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        let tomopitch_model_name = format!("tomopitch{}.mod", axis_id.get_extension());
        let result = (|| -> Result<(), ImodManagerException> {
            self.get_imod_manager().set_open_contours(
                imod_manager::FULL_VOLUME_KEY,
                Some(axis_id),
                true,
            )?;
            self.get_imod_manager().set_point_limit(
                imod_manager::FULL_VOLUME_KEY,
                Some(axis_id),
                2,
            )?;
            self.get_imod_manager()
                .open_string_axis_id_string_boolean_run3dmod_menu_options(
                    imod_manager::FULL_VOLUME_KEY,
                    Some(axis_id),
                    Some(&tomopitch_model_name),
                    true,
                    Some(menu_options),
                )?;
            self.get_recon_process_track()
                .set_tomogram_positioning_state(ProcessState::InProgress, axis_id);
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_tomogram_positioning_state(ProcessState::InProgress, axis_id);
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "Problem opening sample reconstruction",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `tomopitch(AxisID, ProcessResultDisplay, ProcessSeries)`
    /// (ApplicationManager.java:5133).
    pub fn tomopitch(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let thread_name = match self.get_process_mgr().tomopitch(
            axis_id,
            process_result_display_ref.clone(),
            process_series
                .clone()
                .map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute tomopitch{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Finding sample position"),
                axis_id,
                Some(&ProcessName::TOMOPITCH),
            );
        }
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java `postProcess(AxisID, ProcessName, ProcessDetails, ProcessResultDisplay)`
    /// (ApplicationManager.java:5160).  Post processing after a successful process.
    ///
    /// `processDetails` arrives as the process's command (`ComScriptProcess
    /// .getCommand()`); the Java body does not read it.
    pub fn post_process(
        &'static self,
        axis_id: AxisID,
        process_name: Option<ProcessName>,
        process_details: Option<Arc<dyn Command + Send + Sync>>,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) {
        let _ = process_details;
        // Posted: Java calls this on the process thread.  The identity test
        // against the factory's button and the dialog call both need the event
        // dispatch thread.
        invoke_later(move || {
            if process_name == Some(ProcessName::ALIGN)
                && match &process_result_display {
                    None => false,
                    Some(process_result_display) => std::ptr::addr_eq(
                        Rc::as_ptr(process_result_display.get()),
                        Rc::as_ptr(
                            &self
                                .get_process_result_display_factory(axis_id)
                                .get_compute_alignment(),
                        ),
                    ),
                }
            {
                match self
                    .get_imod_manager()
                    .reopen_log(imod_manager::COARSE_ALIGNED_KEY, Some(axis_id))
                {
                    Ok(()) => {}
                    Err(ImodManagerException::AxisType(e)) => {
                        UI_HARNESS.with(|ui_harness| {
                            ui_harness.open_message_dialog_base_manager_string_string(
                                Some(self),
                                &format!("Unable to reopen log file.\n{e}"),
                                "3dmod Error",
                            )
                        });
                    }
                    Err(ImodManagerException::Io(e)) => {
                        UI_HARNESS.with(|ui_harness| {
                            ui_harness.open_message_dialog_base_manager_string_string(
                                Some(self),
                                &format!("Unable to reopen log file.\n{e}"),
                                "3dmod Error",
                            )
                        });
                    }
                    Err(ImodManagerException::SystemProcess(e)) => {
                        UI_HARNESS.with(|ui_harness| {
                            ui_harness.open_message_dialog_base_manager_string_string(
                                Some(self),
                                &format!("Unable to reopen log file.\n{e}"),
                                "3dmod Error",
                            )
                        });
                    }
                    // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
                    Err(ImodManagerException::Runtime(message)) => {
                        eprintln!("{message}");
                    }
                }
            } else if process_name == Some(ProcessName::PATCHCORR) {
                if let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() {
                    // backwards compatibility - patchcorr used to create patch_vector.mod
                    tomogram_combination_dialog.update_patch_vector_model_display();
                }
            }
        });
    }

    /// Java `tomogramPositioningPostProcess(AxisID, ProcessDetails, PosSampleType)`
    /// (ApplicationManager.java:5190).
    pub fn tomogram_positioning_post_process(
        &'static self,
        axis_id: AxisID,
        process_details: Option<Arc<dyn Command + Send + Sync>>,
        pos_sample_type: PosSampleType,
    ) {
        // Posted: Java calls this on the process thread.
        invoke_later(move || {
            let expert: Option<Rc<TomogramPositioningExpert>> = self
                .get_ui_expert(Some(DialogType::TomogramPositioning), axis_id)
                .and_then(|expert| {
                    (expert as Rc<dyn Any>)
                        .downcast::<TomogramPositioningExpert>()
                        .ok()
                });
            // `getUIExpert(TOMOGRAM_POSITIONING, ...)` never returns null.
            // Fixed in translation: "ProcessDetails is required" - Java dereferences a
            // null one inside postProcess (NullPointerException); nothing is updated.
            if let Some(expert) = expert
                && let Some(process_details) = process_details
                    .as_ref()
                    .and_then(|command| command.get_process_details())
            {
                expert.post_process(process_details, self.get_state(), pos_sample_type);
            }
        });
    }

    /// Java `errorProcess(AxisID, ProcessName, ProcessDetails)`
    /// (ApplicationManager.java:5203).  Processing done after an unsuccessful
    /// process.
    pub fn error_process(
        &'static self,
        axis_id: AxisID,
        process_name: Option<ProcessName>,
        process_details: Option<Arc<dyn Command + Send + Sync>>,
    ) {
        let _ = (axis_id, process_details);
        // Posted: Java calls this on the process thread.
        invoke_later(move || {
            if process_name == Some(ProcessName::COMBINE) {
                if let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() {
                    tomogram_combination_dialog.update_patch_vector_model_display();
                }
            }
        });
    }

    /// Java `msgPatchVectorCreated()` (ApplicationManager.java:5212).
    pub fn msg_patch_vector_created(&'static self) {
        // Posted: Java calls this on the process thread.
        invoke_later(move || {
            if let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() {
                tomogram_combination_dialog.update_patch_vector_model_display();
            }
        });
    }

    /// Java `setTomopitchOutput(AxisID)` (ApplicationManager.java:5218).
    pub fn set_tomopitch_output(&'static self, axis_id: AxisID) {
        // Posted: Java calls this on the process thread.
        invoke_later(move || {
            // `getUIExpert(TOMOGRAM_POSITIONING, ...)` never returns null.
            if let Some(expert) = self
                .get_ui_expert(Some(DialogType::TomogramPositioning), axis_id)
                .and_then(|expert| {
                    (expert as Rc<dyn Any>)
                        .downcast::<TomogramPositioningExpert>()
                        .ok()
                })
            {
                expert.set_tomopitch_output();
            }
        });
    }

    /// Java `setTiltState(AxisID)` (ApplicationManager.java:5223).
    pub fn set_tilt_state(&'static self, axis_id: AxisID) {
        // Posted: Java calls this on the process thread.
        invoke_later(move || {
            if let Some(expert) = self
                .get_ui_expert(Some(DialogType::FinalAlignedStack), axis_id)
                .and_then(|expert| {
                    (expert as Rc<dyn Any>)
                        .downcast::<FinalAlignedStackExpert>()
                        .ok()
                })
            {
                expert.set_tilt_state();
            }
            if let Some(expert) = self
                .get_ui_expert(Some(DialogType::TomogramGeneration), axis_id)
                .and_then(|expert| {
                    (expert as Rc<dyn Any>)
                        .downcast::<TomogramGenerationExpert>()
                        .ok()
                })
            {
                expert.set_tilt_state();
            }
        });
    }

    /// Java `finalAlign(AxisID, ProcessResultDisplay, ProcessSeries,
    /// ConstTiltalignParam)` (ApplicationManager.java:5236).  Compute the final
    /// alignment from the updated parameters in the positioning dialog.
    pub fn final_align(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        tiltalign_param: Arc<TiltalignParam>,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let thread_name = match self.get_process_mgr().fine_alignment(
            tiltalign_param,
            axis_id,
            process_result_display_ref.clone(),
            process_series
                .clone()
                .map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => {
                self.get_meta_data()
                    .set_fiducialess_alignment(axis_id, false);
                thread_name
            }
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute align{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Calculating final alignment"),
                axis_id,
                Some(&ProcessName::ALIGN),
            );
        }
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java private `generateFiducialessTransforms(AxisID)`
    /// (ApplicationManager.java:5264).  Generate the prexg and _nonfid.xf for the
    /// specified axis and setup the transform files for fiducialless mode.
    fn generate_fiducialess_transforms(&'static self, axis_id: AxisID) {
        match self.get_process_mgr().generate_pre_xg(axis_id) {
            Ok(()) => {}
            // `catch (final LockException except) {}`
            Err(RunCommandError::LogFile(LogFileError::Lock(_))) => {}
            Err(except) => {
                eprintln!("{except}");
                let message = vec!["Unable to generate prexg".to_owned(), except.to_string()];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to generate prexg",
                        Some(axis_id),
                    )
                });
            }
        }
        match self.get_process_mgr().generate_non_fid_xf(axis_id) {
            Ok(()) => {}
            // `catch (final LockException except) {}`
            Err(RunCommandError::LogFile(LogFileError::Lock(_))) => {}
            Err(except) => {
                eprintln!("{except}");
                let message = vec![
                    "Unable to generate _nonfid.xf".to_owned(),
                    except.to_string(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to generate _nonfid.xf",
                        Some(axis_id),
                    )
                });
            }
        }
        // Java has two arms with the same dialog: `IOException | LogFileException |
        // LockException` (which also prints the stack trace) and
        // `InvalidParameterException`.  The translated `setupNonFiducialAlign`
        // reports `makeRawtltFile`'s IOException/InvalidParameterException as
        // `RunCommandError::SystemProcess`, so every error takes the first arm.
        match self.get_process_mgr().setup_non_fiducial_align(axis_id) {
            Ok(()) => {}
            Err(except) => {
                eprintln!("{except}");
                let message = vec![
                    "Unable to setup fiducialless align files".to_owned(),
                    except.to_string(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to setup fiducialless align files",
                        Some(axis_id),
                    )
                });
            }
        }
    }

    /// Java `makeRawtltFile(AxisID) throws IOException, InvalidParameterException`
    /// (ApplicationManager.java:5307).  `Err` carries the thrown exception's
    /// message (the caller, `ProcessManager.setupNonFiducialAlign`, rethrows it).
    pub fn make_rawtlt_file(&'static self, axis_id: AxisID) -> Result<(), String> {
        let property_user_dir = self.get_property_user_dir().unwrap_or("null".to_owned());
        let rawtlt = PathBuf::from(utilities::java_io_file_new(
            &property_user_dir,
            &format!(
                "{}{}.rawtlt",
                self.get_meta_data().get_dataset_name(),
                axis_id.get_extension()
            ),
        ));
        // backing up .rawtlt, which is currently unnecessary because this function
        // is only called when .rawtlt doesn't exist
        if let Err(e) = utilities::rename_file(
            Some(self),
            Some(axis_id),
            Some(&rawtlt),
            Some(&PathBuf::from(utilities::java_io_file_new(
                &property_user_dir,
                &format!(
                    "{}{}.rawtlt~",
                    self.get_meta_data().get_dataset_name(),
                    axis_id.get_extension()
                ),
            ))),
            false,
            false,
            false,
        ) {
            eprintln!("{}", e.get_message());
            eprintln!("{e}");
        }

        let rawtlt_absolute_path =
            utilities::java_io_file_get_absolute_path(&rawtlt.to_string_lossy());
        // Java's `catch (IOException)` and `catch (InvalidParameterException)`
        // arms show the same dialog and rethrow; `MRCHeader.read` reports both as
        // one message here, so both take the IOException arm (the other arm only
        // adds a stack trace and closes the writer, which dropping it does).
        let result = (|| -> Result<std::io::BufWriter<std::fs::File>, String> {
            let mut buffered_writer = std::io::BufWriter::new(
                std::fs::File::create(&rawtlt).map_err(|except| except.to_string())?,
            );
            let tilt_angle_spec = if axis_id == AxisID::Second {
                self.get_meta_data().get_tilt_angle_spec_b()
            } else {
                self.get_meta_data().get_tilt_angle_spec_a()
            };
            let starting_angle = tilt_angle_spec.get_range_min();
            let step = tilt_angle_spec.get_range_step();
            // `MRCHeader.getInstance(BaseManager, AxisID, FileType)` never returns
            // null.
            let raw_stack_header = MRCHeader::get_instance_from_file_type(
                self,
                Some(axis_id),
                &file_type::CLASS.raw_stack,
            )
            .expect("MRCHeader.getInstance");
            let read = raw_stack_header.borrow_mut().read_with_manager(self)?;
            if !read {
                let message = format!("Unable to create {rawtlt_absolute_path}");
                ui_harness::post_message_dialog(
                    Some(self),
                    message.clone(),
                    "Unable to create raw tilt file".to_owned(),
                    Some(axis_id),
                );
                drop(buffered_writer);
                // `throw new IOException(message)`, caught by the IOException arm
                // below as in Java.
                return Err(message);
            }
            let sections = raw_stack_header.borrow().get_n_sections();
            for cur_section in 0..sections {
                buffered_writer
                    .write_all(
                        java_lang_double_to_string(starting_angle + (step * cur_section as f64))
                            .as_bytes(),
                    )
                    .map_err(|except| except.to_string())?;
                buffered_writer
                    .write_all(b"\n")
                    .map_err(|except| except.to_string())?;
            }
            Ok(buffered_writer)
        })();
        let buffered_writer = match result {
            Ok(buffered_writer) => buffered_writer,
            Err(except) => {
                let message = vec![
                    format!("Unable to create {rawtlt_absolute_path}"),
                    except.clone(),
                ];
                // Posted: Java calls this on the process thread.
                invoke_later(move || {
                    UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                            Some(self),
                            &message,
                            "Unable to create raw tilt file",
                            Some(axis_id),
                        )
                    });
                });
                return Err(except);
            }
        };

        // `bufferedWriter.close()` outside the try: an IOException here
        // propagates without a dialog.
        buffered_writer
            .into_inner()
            .map_err(|except| except.error().to_string())?;
        Ok(())
    }

    /// Java `setEnabledFixEdgesWithMidas(AxisID)` (ApplicationManager.java:5379).
    pub fn set_enabled_fix_edges_with_midas(&'static self, axis_id: AxisID) {
        // Posted: Java calls this on the process thread.
        invoke_later(move || {
            let Some(coarse_align_dialog) = self.map_coarse_align_dialog(axis_id) else {
                return;
            };
            coarse_align_dialog.set_enabled_fix_edges_midas_button();
        });
    }

    /// Java `updateNewstCom(NewstackDisplay, AxisID, boolean, boolean)`
    /// (ApplicationManager.java:5393).  Update the newst.com from the display.
    /// Reads metaData.
    pub fn update_newst_com(
        &'static self,
        display: Option<&dyn NewstackDisplay>,
        axis_id: AxisID,
        validate: bool,
        do_validation: bool,
    ) -> Option<NewstParam> {
        // Set a reference to the correct object
        let Some(display) = display else {
            UI_HARNESS.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    "Can not update newst?.com without an active display",
                    "Program logic error",
                    Some(axis_id),
                )
            });
            return None;
        };
        let mut newst_param = self
            .get_com_script_manager()
            .get_newst_com_newst_param(axis_id);
        newst_param.set_validate(validate);
        // Make sure the size output is removed, it was only there for a
        // copytomocoms template
        newst_param.set_command_mode(Some(newst_param::Mode::FullAlignedStack));
        newst_param
            .set_output_image_file_key(Some(FileKey::clone(&file_type::CLASS.aligned_stack)));
        newst_param
            .set_fiducialess_alignment(self.get_meta_data().is_fiducialess_alignment(axis_id));
        // Java's `catch (NumberFormatException)` arm (the same dialog as the
        // FortranInputSyntaxException arm) has no Rust counterpart: the translated
        // callees report parse failures through their results.
        match display.get_parameters(&mut newst_param, do_validation) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(NewstackDisplayException::FortranInputSyntaxException(except)) => {
                let error_message = vec![
                    "newst Parameter Syntax Error".to_owned(),
                    format!("Axis: {}", axis_id.get_extension()),
                    except.get_message().unwrap_or("null").to_owned(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Newst Parameter Syntax Error",
                        Some(axis_id),
                    )
                });
                return None;
            }
            Err(NewstackDisplayException::InvalidParameterException(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!("Unable to update newst com:  {}", e.get_message()),
                        "Etomo Error",
                        Some(axis_id),
                    )
                });
                return None;
            }
            Err(NewstackDisplayException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!("Unable to update newst com:  {e}"),
                        "Etomo Error",
                        Some(axis_id),
                    )
                });
                return None;
            }
        }
        self.get_com_script_manager()
            .save_newst(&newst_param, axis_id);
        Some(newst_param)
    }

    /// Java `updateNewst3dFindCom(NewstackDisplay, AxisID, boolean, boolean)`
    /// (ApplicationManager.java:5455).  Update the newst3dfind.com from the
    /// display.  Reads metaData.
    pub fn update_newst3d_find_com(
        &'static self,
        display: Option<&dyn NewstackDisplay>,
        axis_id: AxisID,
        validate: bool,
        do_validation: bool,
    ) -> Option<NewstParam> {
        // Java ignores `validate` here and always calls `setValidate(true)`; kept.
        let _ = validate;
        // Set a reference to the correct object
        let Some(display) = display else {
            UI_HARNESS.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    "Can not update newst3dfind?.com without an active display",
                    "Program logic error",
                    Some(axis_id),
                )
            });
            return None;
        };
        let mut newst_param = self
            .get_com_script_manager()
            .get_newst_param_from_newst3d_find(axis_id);
        newst_param.set_validate(true);
        // Java's `catch (NumberFormatException)` arm has no Rust counterpart (see
        // `update_newst_com`).
        match display.get_parameters(&mut newst_param, do_validation) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(NewstackDisplayException::FortranInputSyntaxException(except)) => {
                let error_message = vec![
                    "newst Parameter Syntax Error".to_owned(),
                    format!("Axis: {}", axis_id.get_extension()),
                    except.get_message().unwrap_or("null").to_owned(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Newst Parameter Syntax Error",
                        Some(axis_id),
                    )
                });
                return None;
            }
            Err(NewstackDisplayException::InvalidParameterException(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!("Unable to update newst 3dfind com:  {}", e.get_message()),
                        "Etomo Error",
                        Some(axis_id),
                    )
                });
                return None;
            }
            Err(NewstackDisplayException::Io(e)) => {
                eprintln!("{e}");
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!("Unable to update newst 3dfind com:  {e}"),
                        "Etomo Error",
                        Some(axis_id),
                    )
                });
                return None;
            }
        }
        self.get_com_script_manager()
            .save_newst3d_find_newst(&newst_param, axis_id);
        Some(newst_param)
    }

    /// Java `updateBlendCom(BlendmontDisplay, AxisID, boolean, boolean) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`
    /// (ApplicationManager.java:5512).  Get the set the blendmont parameters and
    /// update the blend.com script.
    pub fn update_blend_com(
        &'static self,
        display: &dyn BlendmontDisplay,
        axis_id: AxisID,
        validate: bool,
        do_validation: bool,
    ) -> Result<Option<BlendmontParam>, BlendmontDisplayException> {
        let mut blend_param = self.get_com_script_manager().get_blend_param(axis_id);
        blend_param.set_validate(validate);
        if !display.get_parameters(&mut blend_param, do_validation)? {
            return Ok(None);
        }
        blend_param.set_mode(blendmont_param::Mode::Blend);
        blend_param.set_blendmont_state(&self.get_state().get_invalid_edge_functions(axis_id));
        self.get_com_script_manager()
            .save_blend(&blend_param, axis_id);
        Ok(Some(blend_param))
    }

    /// Java `updateBlend3dFindCom(BlendmontDisplay, AxisID, boolean, boolean)
    /// throws FortranInputSyntaxException, InvalidParameterException, IOException`
    /// (ApplicationManager.java:5533).  Get the set the blendmont parameters and
    /// update the blend3dfin.com script.
    pub fn update_blend3d_find_com(
        &'static self,
        display: &dyn BlendmontDisplay,
        axis_id: AxisID,
        validate: bool,
        do_validation: bool,
    ) -> Result<Option<BlendmontParam>, BlendmontDisplayException> {
        // Update blendmont
        let mut blend_param = self
            .get_com_script_manager()
            .get_blend_param_from_blend3d_find(axis_id);
        blend_param.set_validate(validate);
        if !display.get_parameters(&mut blend_param, do_validation)? {
            return Ok(None);
        }
        blend_param.set_mode(blendmont_param::Mode::Blend3dFind);
        blend_param.set_blendmont_state(&self.get_state().get_invalid_edge_functions(axis_id));
        blend_param
            .set_image_output_file_for_3d_find(&file_type::CLASS.newst_or_blend_3d_find_output);
        self.get_com_script_manager()
            .save_blend3d_find_blendmont(&blend_param, axis_id);
        // Update mrctaper
        let mut mrc_taper_param = self
            .get_com_script_manager()
            .get_mrc_taper_param_from_blend3d_find(axis_id);
        mrc_taper_param.set_input_file(
            file_type::CLASS
                .newst_or_blend_3d_find_output
                .get_file_name(Some(self), Some(axis_id))
                .as_deref(),
        );
        self.get_com_script_manager()
            .save_blend3d_find_mrc_taper(&mrc_taper_param, axis_id);
        Ok(Some(blend_param))
    }

    /// Java `blend3dFind(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// AxisID, Run3dmodMenuOptions, DialogType, BlendmontDisplay)`
    /// (ApplicationManager.java:5567).  Run blend_3dfind.com.
    #[allow(clippy::too_many_arguments)]
    pub fn blend3d_find(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        axis_id: AxisID,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        blendmont_display: &dyn BlendmontDisplay,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("blend3dFind")),
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Get the user input from the dialog
        if blendmont_display.is_fiducialess()
            && !UIExpertUtilities::INSTANCE
                .update_fiducialess_params_application_manager_string_boolean_axis_id(
                    self,
                    &self
                        .get_state()
                        .get_stack_image_rotation(axis_id)
                        .to_string(),
                    self.get_state().is_newst_fiducialess_alignment(axis_id),
                    axis_id,
                )
        {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        let mut param: Option<BlendmontParam> = None;
        match self.update_blend3d_find_com(blendmont_display, axis_id, true, true) {
            Ok(None) => {
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                process_series.borrow().end_series();
                return;
            }
            Ok(Some(blend_param)) => param = Some(blend_param),
            Err(e) => {
                // The three Java arms (FortranInputSyntaxException,
                // InvalidParameterException, IOException) show the same dialog.
                let message = match &e {
                    BlendmontDisplayException::FortranInputSyntaxException(e) => {
                        e.get_message().unwrap_or("null").to_owned()
                    }
                    BlendmontDisplayException::InvalidParameterException(e) => {
                        e.get_message().to_owned()
                    }
                    BlendmontDisplayException::Io(e) => e.to_string(),
                };
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string(
                        Some(self),
                        &message,
                        "Update Com Error",
                    )
                });
            }
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        // Make sure we have a current prexg and _nonfid.xf if fiducialess is
        // selected
        if self.get_meta_data().is_fiducialess_alignment(axis_id) {
            self.generate_fiducialess_transforms(axis_id);
        } else {
            match self
                .get_process_mgr()
                .setup_fiducial_align(axis_id, process_result_display_ref.clone())
            {
                Ok(true) => {}
                Ok(false) => {
                    self.send_msg(
                        Some(ProcessResult::FAILED_TO_START),
                        process_result_display.clone(),
                    );
                    process_series.borrow().end_series();
                    return;
                }
                Err(except) => {
                    eprintln!("{except}");
                    let message = vec![
                        "Problem copying fiducial alignment files".to_owned(),
                        except.get_message(),
                    ];
                    UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                            Some(self),
                            &message,
                            "Unable to copy fiducial alignment files",
                            Some(axis_id),
                        )
                    });
                }
            }
        }
        // Java passes `param` to `processMgr.blend` even when `updateBlend3dFindCom`
        // threw above and `param` is still null (ApplicationManager.java:5591-5640);
        // `ProcessManager.blend` then throws a NullPointerException
        // (`blendmontParam.getMode()`, ProcessManager.java:458).  Fixed in
        // translation: with no param the process fails to start, as when the
        // update returns null.
        let Some(param) = param else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        let mut thread_name: Option<String> = None;
        match self.get_process_mgr().blend(
            Arc::new(param),
            axis_id,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series))) as ProcessSeriesRef),
        ) {
            Ok(name) => thread_name = Some(name),
            Err(e) => {
                eprintln!("{e}");
                let message = vec!["Can not execute blend_3dfind".to_owned(), e.0.clone()];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                self.send_msg(
                    Some(ProcessResult::FAILED_TO_START),
                    process_result_display.clone(),
                );
            }
        }
        self.set_thread_name(thread_name.as_deref(), Some(axis_id));
    }

    /// Java `blend(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton, AxisID,
    /// Run3dmodMenuOptions, DialogType, FiducialessParams, BlendmontDisplay)`
    /// (ApplicationManager.java:5654).  Run blend.com.
    #[allow(clippy::too_many_arguments)]
    pub fn blend(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        axis_id: AxisID,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        fiducialess_params: &dyn FiducialessParams,
        blendmont_display: &dyn BlendmontDisplay,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("blend")),
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Get the user input from the dialog
        if !UIExpertUtilities::INSTANCE
            .update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
                self,
                fiducialess_params,
                axis_id,
                true,
            )
        {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        let mut param: Option<BlendmontParam> = None;
        match self.update_blend_com(blendmont_display, axis_id, true, true) {
            Ok(None) => {
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                process_series.borrow().end_series();
                return;
            }
            Ok(Some(blend_param)) => param = Some(blend_param),
            Err(e) => {
                // The three Java arms (FortranInputSyntaxException,
                // InvalidParameterException, IOException) show the same dialog.
                let message = match &e {
                    BlendmontDisplayException::FortranInputSyntaxException(e) => {
                        e.get_message().unwrap_or("null").to_owned()
                    }
                    BlendmontDisplayException::InvalidParameterException(e) => {
                        e.get_message().to_owned()
                    }
                    BlendmontDisplayException::Io(e) => e.to_string(),
                };
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string(
                        Some(self),
                        &message,
                        "Update Com Error",
                    )
                });
            }
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        // Make sure we have a current prexg and _nonfid.xf if fiducialess is
        // selected
        if self.get_meta_data().is_fiducialess_alignment(axis_id) {
            self.generate_fiducialess_transforms(axis_id);
        } else {
            match self
                .get_process_mgr()
                .setup_fiducial_align(axis_id, process_result_display_ref.clone())
            {
                Ok(true) => {}
                Ok(false) => {
                    self.send_msg(
                        Some(ProcessResult::FAILED_TO_START),
                        process_result_display.clone(),
                    );
                    process_series.borrow().end_series();
                    return;
                }
                Err(except) => {
                    eprintln!("{except}");
                    let message = vec![
                        "Problem copying fiducial alignment files".to_owned(),
                        except.get_message(),
                    ];
                    UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                            Some(self),
                            &message,
                            "Unable to copy fiducial alignment files",
                            Some(axis_id),
                        )
                    });
                }
            }
        }
        // Java passes `param` to `processMgr.blend` even when `updateBlendCom` threw
        // above and `param` is still null (ApplicationManager.java:5677-5716);
        // `ProcessManager.blend` then throws a NullPointerException
        // (`blendmontParam.getMode()`, ProcessManager.java:458).  Fixed in
        // translation: with no param the process fails to start, as when the
        // update returns null.
        let Some(param) = param else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        let mut thread_name: Option<String> = None;
        match self.get_process_mgr().blend(
            Arc::new(param),
            axis_id,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series))) as ProcessSeriesRef),
        ) {
            Ok(name) => thread_name = Some(name),
            Err(e) => {
                eprintln!("{e}");
                let message = vec!["Can not execute blend".to_owned(), e.0.clone()];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                self.send_msg(
                    Some(ProcessResult::FAILED_TO_START),
                    process_result_display.clone(),
                );
            }
        }
        self.set_thread_name(thread_name.as_deref(), Some(axis_id));
    }

    /// Java `equalsBinning(AxisID, int, FileType)` (ApplicationManager.java:5728).
    pub fn equals_binning(
        &'static self,
        axis_id: AxisID,
        binning: i32,
        file_type: &Arc<FileType>,
    ) -> bool {
        let file_binning = UIExpertUtilities::INSTANCE
            .get_stack_binning_base_manager_axis_id_file_type(self, axis_id, file_type);
        binning == file_binning
    }

    /// Java `newst(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton, AxisID,
    /// Run3dmodMenuOptions, DialogType, FiducialessParams, NewstackDisplay,
    /// ProcessName)` (ApplicationManager.java:5746).  Run newst.com.
    #[allow(clippy::too_many_arguments)]
    pub fn newst(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        axis_id: AxisID,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        fiducialess_params: &dyn FiducialessParams,
        newstack_display: Option<&dyn NewstackDisplay>,
        process_name: ProcessName,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("newst")),
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Get the user input from the dialog
        if !UIExpertUtilities::INSTANCE
            .update_fiducialess_params_application_manager_fiducialess_params_axis_id_boolean(
                self,
                fiducialess_params,
                axis_id,
                true,
            )
        {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        let Some(param) = self.update_newst_com(newstack_display, axis_id, true, true) else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        // Make sure we have a current prexg and _nonfid.xf if fiducialess is
        // selected
        if self.get_meta_data().is_fiducialess_alignment(axis_id) {
            self.generate_fiducialess_transforms(axis_id);
        } else {
            match self
                .get_process_mgr()
                .setup_fiducial_align(axis_id, process_result_display_ref.clone())
            {
                Ok(true) => {}
                Ok(false) => {
                    self.send_msg(
                        Some(ProcessResult::FAILED_TO_START),
                        process_result_display.clone(),
                    );
                    process_series.borrow().end_series();
                    return;
                }
                Err(except) => {
                    eprintln!("{except}");
                    let message = vec![
                        "Problem copying fiducial alignment files".to_owned(),
                        except.get_message(),
                    ];
                    UI_HARNESS.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                            Some(self),
                            &message,
                            "Unable to copy fiducial alignment files",
                            Some(axis_id),
                        )
                    });
                }
            }
        }
        let mut thread_name: Option<String> = None;
        match self.get_process_mgr().newst(
            Arc::new(param),
            axis_id,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series))) as ProcessSeriesRef),
            process_name,
        ) {
            Ok(name) => thread_name = Some(name),
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute newst{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                UI_HARNESS.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(axis_id),
                    )
                });
                self.send_msg(
                    Some(ProcessResult::FAILED_TO_START),
                    process_result_display.clone(),
                );
            }
        }
        self.set_thread_name(thread_name.as_deref(), Some(axis_id));
    }

    /// Java `newst3dFind(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// AxisID, Run3dmodMenuOptions, DialogType, NewstackDisplay)`
    /// (ApplicationManager.java:5822).  Run newst_3dfind.com.
    #[allow(clippy::too_many_arguments)]
    pub fn newst3d_find(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        axis_id: AxisID,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        newstack_display: &dyn NewstackDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("newst3dFind")),
        };
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Get the user input from the dialog
        if newstack_display.is_fiducialess()
            && !UIExpertUtilities::INSTANCE
                .update_fiducialess_params_application_manager_string_boolean_axis_id(
                    self,
                    &self
                        .get_state()
                        .get_stack_image_rotation(axis_id)
                        .to_string(),
                    self.get_state().is_newst_fiducialess_alignment(axis_id),
                    axis_id,
                )
        {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        // `ConstNewstParam param = null;`
        let param = self.update_newst3d_find_com(Some(newstack_display), axis_id, true, true);
        let Some(param) = param else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        // Make sure we have a current prexg and _nonfid.xf if fiducialess is
        // selected
        if self.get_meta_data().is_fiducialess_alignment(axis_id) {
            self.generate_fiducialess_transforms(axis_id);
        } else {
            match self
                .get_process_mgr()
                .setup_fiducial_align(axis_id, process_result_display_ref.clone())
            {
                Ok(true) => {}
                Ok(false) => {
                    self.send_msg(
                        Some(ProcessResult::FailedToStart),
                        process_result_display.clone(),
                    );
                    process_series.borrow().end_series();
                    return;
                }
                // `catch (final IOException | LogFileException | LockException except)`
                Err(except) => {
                    eprintln!("{except}");
                    let message = vec![
                        "Problem copying fiducial alignment files".to_string(),
                        except.get_message(),
                    ];
                    ui_harness::open_message_dialog_array_from_process(
                        Some(self),
                        &message,
                        "Unable to copy fiducial alignment files",
                        Some(axis_id),
                    );
                }
            }
        }
        let mut thread_name: Option<String> = None;
        match self.get_process_mgr().newst(
            Arc::new(param),
            axis_id,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series))) as ProcessSeriesRef),
            ProcessName::NEWST_3D_FIND,
        ) {
            Ok(name) => thread_name = Some(name),
            // `catch (final AxisBusyException e)`
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute newst{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(axis_id),
                );
                self.send_msg(
                    Some(ProcessResult::FailedToStart),
                    process_result_display.clone(),
                );
            }
        }
        self.set_thread_name(thread_name.as_deref(), Some(axis_id));
    }

    /// Java `findBeads3d(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// AxisID, Run3dmodMenuOptions, DialogType, FindBeads3dDisplay)`
    /// (ApplicationManager.java:5899).  Run newst_3dfind.com.
    #[allow(clippy::too_many_arguments)]
    pub fn find_beads3d(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        axis_id: AxisID,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        display: &dyn FindBeads3dDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("findBeads3d")),
        };
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Get the user input from the dialog
        if display.is_fiducialess()
            && !UIExpertUtilities::INSTANCE
                .update_fiducialess_params_application_manager_string_boolean_axis_id(
                    self,
                    &self
                        .get_state()
                        .get_stack_image_rotation(axis_id)
                        .to_string(),
                    self.get_state().is_newst_fiducialess_alignment(axis_id),
                    axis_id,
                )
        {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        // `ConstFindBeads3dParam param = null;`
        let param = self.update_find_beads3d_com(display, axis_id, true);
        let Some(param) = param else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let mut thread_name: Option<String> = None;
        match self.get_process_mgr().find_beads3d(
            Arc::new(param),
            axis_id,
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series))) as ProcessSeriesRef),
        ) {
            Ok(name) => thread_name = Some(name),
            // `catch (final AxisBusyException e)`
            Err(e) => {
                eprintln!("{e}");
                // The source's message names newst, as written.
                let message = vec![
                    format!("Can not execute newst{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(axis_id),
                );
                self.send_msg(
                    Some(ProcessResult::FailedToStart),
                    process_result_display.clone(),
                );
            }
        }
        self.set_thread_name(thread_name.as_deref(), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&ProcessName::FIND_BEADS_3D.to_string().as_str()),
                axis_id,
                Some(&ProcessName::FIND_BEADS_3D),
            );
        }
    }

    /// Java `setProcessState(ProcessState, AxisID, DialogType)`
    /// (ApplicationManager.java:5944).
    pub fn set_process_state(
        &self,
        process_state: ProcessState,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        // `if (processTrack != null)`: never null after construction.
        self.get_recon_process_track()
            .set_state_dialog_type(process_state, axis_id, dialog_type);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                process_state,
                axis_id,
                dialog_type,
            );
        }
    }

    /// Java `updateFindBeads3dCom(FindBeads3dDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:5952).
    pub fn update_find_beads3d_com(
        &'static self,
        display: &dyn FindBeads3dDisplay,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<FindBeads3dParam> {
        let mut param = self
            .get_com_script_manager()
            .get_find_beads3d_param(axis_id);
        if !self.get_state().is_track_light_beads_null(axis_id) {
            param.set_light_beads(self.get_state().is_track_light_beads(axis_id));
        } else {
            // backwards compatibility
            // Get light beads from track.com.
            let _beadtrack_param = self.get_com_script_manager().get_beadtrack_param(axis_id);
        }
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        self.get_com_script_manager()
            .save_find_beads3d(&param, axis_id);
        Some(param)
    }

    /// Java `sendMsg(ProcessResult, ProcessResultDisplay)`
    /// (ApplicationManager.java:5970).
    pub fn send_msg(
        &self,
        display_state: Option<ProcessResult>,
        process_result_display: Option<ProcessResultDisplayHandle>,
    ) {
        let (Some(display_state), Some(process_result_display)) =
            (display_state, process_result_display)
        else {
            return;
        };
        process_result_display.msg_process_result(display_state);
    }

    /// Java `isAxisBusy(AxisID, ProcessResultDisplay)`
    /// (ApplicationManager.java:5978).  Called from process threads by
    /// `ProcessManager`, so the display arrives as a `ProcessResultDisplayRef`.
    pub fn is_axis_busy(
        &self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayRef>,
    ) -> bool {
        self.get_process_mgr()
            .base
            .in_use(axis_id, process_result_display, true)
    }

    /// Java `mtffilter(ConstMTFFilterParam, AxisID, ProcessResultDisplay,
    /// ProcessSeries)` (ApplicationManager.java:5985).
    pub fn mtffilter(
        &'static self,
        param: Arc<MTFFilterParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) -> Option<ProcessResult> {
        let thread_name = match self.get_process_mgr().mtffilter(
            param,
            axis_id,
            process_result_display
                .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef),
            process_series.map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => thread_name,
            // `catch (final AxisBusyException e)`
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute mtffilter{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(axis_id),
                );
                return Some(ProcessResult::FailedToStart);
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java `ctfPlotter(AxisID, ProcessResultDisplay)` (ApplicationManager.java:6004).
    pub fn ctf_plotter(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
    ) -> Option<ProcessResult> {
        // `try { processMgr.ctfPlotter(axisID, processResultDisplay); } catch
        // (SystemProcessException e) { ... "Can not execute " +
        // ProcessName.CTF_PLOTTER.getComscript(axisID) ... return FAILED_TO_START; }`:
        // the translated `ProcessManager::ctf_plotter` starts the non-blocking com
        // script and reports no failure, so the catch arm has no Rust counterpart.
        self.get_process_mgr().ctf_plotter(
            axis_id,
            process_result_display
                .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef),
        );
        self.get_recon_process_track()
            .set_final_aligned_stack_state(ProcessState::InProgress, axis_id);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_final_aligned_stack_state(ProcessState::InProgress, axis_id);
        }
        // No way to know when this is done, so I'm assuming that it succeeded.
        None
    }

    /// Java `ctfCorrection(ConstCtfPhaseFlipParam, AxisID, ProcessResultDisplay,
    /// ProcessSeries)` (ApplicationManager.java:6023).
    pub fn ctf_correction(
        &'static self,
        param: Arc<CtfPhaseFlipParam>,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) -> Option<ProcessResult> {
        let thread_name = match self.get_process_mgr().ctf_correction(
            param,
            axis_id,
            process_result_display
                .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef),
            process_series.map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
        ) {
            Ok(thread_name) => thread_name,
            // `catch (final AxisBusyException e)`
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!(
                        "Can not execute {}",
                        ProcessName::CTF_CORRECTION.get_comscript(axis_id)
                    ),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(axis_id),
                );
                return Some(ProcessResult::FailedToStart);
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        self.get_recon_process_track()
            .set_final_aligned_stack_state(ProcessState::InProgress, axis_id);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_final_aligned_stack_state(ProcessState::InProgress, axis_id);
        }
        None
    }

    /// Java `sampleTilt(AxisID, ProcessResultDisplay, ProcessSeries, TiltParam)`
    /// (ApplicationManager.java:6045).
    pub fn sample_tilt(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        tilt_param: Arc<TiltParam>,
    ) -> Option<ProcessResult> {
        let thread_name = match self.get_process_mgr().tilt(
            axis_id,
            process_result_display
                .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef),
            process_series.map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
            tilt_param,
            None,
            None,
        ) {
            Ok(thread_name) => thread_name,
            // `catch (final AxisBusyException e)`
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute tilt{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(axis_id),
                );
                return Some(ProcessResult::FailedToStart);
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java `reprojectModelAction(ProcessResultDisplay, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions, TiltDisplay, AxisID, DialogType)`
    /// (ApplicationManager.java:6065).
    #[allow(clippy::too_many_arguments)]
    pub fn reproject_model_action(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        if display.is_none() {
            return;
        }
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self,
                axis_id,
                Some(dialog_type),
                Some("reprojectModelAction"),
            ),
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        self.reproject_model(
            process_result_display,
            Some(process_series),
            display,
            axis_id,
            dialog_type,
        );
    }

    /// Java `tilt3dFindAction(ProcessResultDisplay, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions, TiltDisplay, AxisID, DialogType,
    /// ProcessingMethod)` (ApplicationManager.java:6079).
    #[allow(clippy::too_many_arguments)]
    pub fn tilt3d_find_action(
        &'static self,
        tilt: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
        tilt_processing_method: ProcessingMethod,
    ) {
        if display.is_none() {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("tilt3dFindAction")),
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        if !tilt_processing_method.is_local() {
            self.splittilt3d_find(
                tilt,
                Some(process_series),
                display,
                axis_id,
                dialog_type,
                tilt_processing_method,
            );
        } else {
            self.tilt3d_find(
                tilt,
                Some(process_series),
                display,
                axis_id,
                dialog_type,
                tilt_processing_method,
            );
        }
    }

    /// Java `tiltAction(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, TiltDisplay, AxisID, DialogType, ProcessingMethod)`
    /// (ApplicationManager.java:6103).
    #[allow(clippy::too_many_arguments)]
    pub fn tilt_action(
        &'static self,
        tilt: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
        tilt_processing_method: ProcessingMethod,
    ) {
        if display.is_none() {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("tiltAction")),
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        if !tilt_processing_method.is_local() {
            self.splittilt_process_result_display_process_series_tilt_display_axis_id_dialog_type_processing_method(
                tilt,
                Some(process_series),
                display,
                axis_id,
                dialog_type,
                tilt_processing_method,
            );
        } else {
            self.tilt(
                tilt,
                Some(process_series),
                display,
                axis_id,
                dialog_type,
                tilt_processing_method,
            );
        }
    }

    /// Java `imodTestVolume(Run3dmodMenuOptions, AxisID, TrialTiltDisplay)`
    /// (ApplicationManager.java:6131).  Open 3dmod on the current test volume.
    pub fn imod_test_volume_run3dmod_menu_options_axis_id_trial_tilt_display(
        &'static self,
        menu_options: Run3dmodMenuOptions,
        axis_id: AxisID,
        display: Option<&dyn TrialTiltDisplay>,
    ) {
        let Some(display) = display else {
            return;
        };
        self.imod_test_volume_axis_id_run3dmod_menu_options_string(
            axis_id,
            menu_options,
            display.get_trial_tomogram_name().as_deref(),
        );
    }

    /// Java `commitTestVolume(ProcessResultDisplay, AxisID, TrialTiltDisplay)`
    /// (ApplicationManager.java:6139).
    pub fn commit_test_volume_process_result_display_axis_id_trial_tilt_display(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        axis_id: AxisID,
        display: Option<&dyn TrialTiltDisplay>,
    ) {
        let Some(display) = display else {
            return;
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        self.send_msg(
            self.commit_test_volume_axis_id_process_result_display_string(
                axis_id,
                process_result_display.clone(),
                display.get_trial_tomogram_name().as_deref(),
            ),
            process_result_display,
        );
    }

    /// Java `trialAction(ProcessResultDisplay, ProcessSeries, TrialTiltDisplay,
    /// AxisID, DialogType, ProcessingMethod)` (ApplicationManager.java:6150).
    pub fn trial_action(
        &'static self,
        trial: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        display: Option<&dyn TrialTiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
        tilt_processing_method: ProcessingMethod,
    ) {
        let Some(display) = display else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("trialAction")),
        };
        let trial_tomogram_name = display.get_trial_tomogram_name();
        // Upstream bug fixed (ApplicationManager.java:6163): the source tests
        // `trialTomogramName == ""`, a reference comparison that is false for any
        // name read from the text field, so an empty name was never caught here.
        // The translation compares the contents, which is the evident intent.
        if trial_tomogram_name.as_deref() == Some("") {
            let error_message = vec![
                "Missing trial tomogram filename:".to_string(),
                "A filename for the trial tomogram must be entered in the Trial".to_string()
                    + " tomogram filename edit box.",
            ];
            ui_harness::open_message_dialog_array_from_process(
                Some(self),
                &error_message,
                "Tilt Parameter Syntax Error",
                Some(axis_id),
            );
            process_series.borrow().end_series();
            return;
        }
        if !display.contains_trial_tomogram_name(trial_tomogram_name.as_deref()) {
            display.add_trial_tomogram_name(trial_tomogram_name.as_deref());
        }
        if !tilt_processing_method.is_local() {
            self.split_trial_tilt(
                trial,
                Some(process_series),
                Some(display),
                axis_id,
                dialog_type,
                tilt_processing_method,
            );
        } else {
            self.trial_tilt(
                trial,
                Some(process_series),
                Some(display),
                axis_id,
                dialog_type,
                tilt_processing_method,
            );
        }
    }

    /// Java private `splittilt3dFind(ProcessResultDisplay, ProcessSeries,
    /// TiltDisplay, AxisID, DialogType, ProcessingMethod)`
    /// (ApplicationManager.java:6185).
    fn splittilt3d_find(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
        tilt_processing_method: ProcessingMethod,
    ) {
        let Some(display) = display else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let tilt_param = self.update_tilt3d_find_com(Some(display), axis_id, true);
        if tilt_param.is_none() {
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let splittilt_param = self.update_splittilt_param(Some(display), axis_id, true);
        let Some(splittilt_param) = splittilt_param else {
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        // `if (processTrack != null)`: never null after construction.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let process_result = self
            .splittilt_axis_id_process_result_display_process_series_splittilt_param_dialog_type(
                axis_id,
                process_result_display.clone(),
                process_series.clone(),
                splittilt_param,
                dialog_type,
            );
        if process_result.is_none() {
            // The source dereferences `processSeries` here unchecked; every caller
            // passes one (`tilt3dFindAction` creates it).
            if let Some(process_series) = &process_series {
                process_series
                    .borrow_mut()
                    .set_next_process_output_file_type(
                        Some(&ProcessName::PROCESSCHUNKS.to_string()),
                        Some(ProcessName::TILT_3D_FIND),
                        Some(file_type::CLASS.tilt_3d_find_output.as_ref()),
                        Some(tilt_processing_method),
                    );
            }
        }
        self.send_msg(process_result, process_result_display);
    }

    /// Java private `splittilt(ProcessResultDisplay, ProcessSeries, TiltDisplay,
    /// AxisID, DialogType, ProcessingMethod)` (ApplicationManager.java:6224).
    fn splittilt_process_result_display_process_series_tilt_display_axis_id_dialog_type_processing_method(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
        tilt_processing_method: ProcessingMethod,
    ) {
        let Some(display) = display else {
            return;
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let tilt_param =
            self.update_tilt_com_tilt_display_axis_id_boolean(Some(display), axis_id, true);
        if tilt_param.is_none() {
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let splittilt_param = self.update_splittilt_param(Some(display), axis_id, true);
        let Some(splittilt_param) = splittilt_param else {
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        // `if (processTrack != null)`: never null after construction.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let process_result = self
            .splittilt_axis_id_process_result_display_process_series_splittilt_param_dialog_type(
                axis_id,
                process_result_display.clone(),
                process_series.clone(),
                splittilt_param,
                dialog_type,
            );
        if process_result.is_none() {
            // The source dereferences `processSeries` here unchecked; every caller
            // passes one (`tiltAction` creates it).
            if let Some(process_series) = &process_series {
                process_series
                    .borrow_mut()
                    .set_next_process_output_file_type(
                        Some(&ProcessName::PROCESSCHUNKS.to_string()),
                        Some(ProcessName::TILT),
                        Some(file_type::CLASS.tilt_output.as_ref()),
                        Some(tilt_processing_method),
                    );
            }
        }
        self.send_msg(process_result, process_result_display);
    }

    /// Java private `splitTrialTilt(ProcessResultDisplay, ProcessSeries,
    /// TrialTiltDisplay, AxisID, DialogType, ProcessingMethod)`
    /// (ApplicationManager.java:6260).
    fn split_trial_tilt(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        display: Option<&dyn TrialTiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
        tilt_processing_method: ProcessingMethod,
    ) {
        let Some(display) = display else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let tilt_param = self.update_trial_tilt_com(Some(display), axis_id, true);
        let Some(tilt_param) = tilt_param else {
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let splittilt_param =
            self.update_splittilt_param(Some(display as &dyn TiltDisplay), axis_id, true);
        let Some(splittilt_param) = splittilt_param else {
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        // `if (processTrack != null)`: never null after construction.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let process_result = self
            .splittilt_axis_id_process_result_display_process_series_splittilt_param_dialog_type(
                axis_id,
                process_result_display.clone(),
                process_series.clone(),
                splittilt_param,
                dialog_type,
            );
        if process_result.is_none() {
            // The source dereferences `processSeries` here unchecked; every caller
            // passes one (`trialAction` creates it).
            if let Some(process_series) = &process_series {
                process_series.borrow_mut().set_next_process_subprocess(
                    Some(&ProcessName::PROCESSCHUNKS.to_string()),
                    Some(ProcessName::TILT),
                    Some(tilt_processing_method),
                );
            }
            self.close_imod_file_key_and_name(
                Some(&*file_key::TRIAL_TOMOGRAM),
                Some(&tilt_param.get_output_file()),
                Some(axis_id),
                true,
            );
        }
        self.send_msg(process_result, process_result_display);
    }

    /// Java private `reprojectModel(ProcessResultDisplay, ProcessSeries,
    /// TiltDisplay, AxisID, DialogType)` (ApplicationManager.java:6304).  Run the
    /// tilt_3dfind command script for the specified axis.
    fn reproject_model(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        let Some(display) = display else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let tilt3d_find_reproject_com = PathBuf::from(utilities::java_io_file_new(
            &self.get_property_user_dir().unwrap_or_default(),
            &ProcessName::TILT_3D_FIND_REPROJECT.get_comscript(axis_id),
        ));
        if !tilt3d_find_reproject_com.exists() {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                &format!(
                    "{} does not exist.  Run {} before running {}",
                    utilities::java_io_file_get_name(&tilt3d_find_reproject_com.to_string_lossy()),
                    FinalAlignedStackDialog::get_tilt3d_find_button_label(),
                    FinalAlignedStackDialog::get_reproject_model_button_label()
                ),
                "Entry Error",
                Some(axis_id),
            );
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let param = self.update_tilt3d_find_reproject_com(Some(display), axis_id, true);
        let Some(param) = param else {
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        // `if (processTrack != null)`: never null after construction.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        self.send_msg(
            self.tilt3d_find_reproject_process(
                axis_id,
                process_result_display.clone(),
                process_series,
                Arc::new(param),
                Some("Reprojecting Model"),
                ProcessName::TILT_3D_FIND_REPROJECT,
            ),
            process_result_display,
        );
    }

    /// Java private `tilt3dFind(ProcessResultDisplay, ProcessSeries, TiltDisplay,
    /// AxisID, DialogType, ProcessingMethod)` (ApplicationManager.java:6347).  Run
    /// the tilt_3dfind command script for the specified axis.
    fn tilt3d_find(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
        processing_method: ProcessingMethod,
    ) {
        let Some(display) = display else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let param = self.update_tilt3d_find_com(Some(display), axis_id, true);
        let Some(param) = param else {
            // The source does not end the series on this path.
            self.send_msg(Some(ProcessResult::FailedToStart), process_result_display);
            return;
        };
        // `if (processTrack != null)`: never null after construction.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        self.send_msg(
            self.tilt3d_find_process(
                axis_id,
                process_result_display.clone(),
                process_series,
                Arc::new(param),
                None,
                ProcessName::TILT_3D_FIND,
                Some(processing_method),
            ),
            process_result_display,
        );
    }

    /// Java private `tilt(ProcessResultDisplay, ProcessSeries, TiltDisplay, AxisID,
    /// DialogType, ProcessingMethod)` (ApplicationManager.java:6373).  Run the tilt
    /// command script for the specified axis.
    fn tilt(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
        processing_method: ProcessingMethod,
    ) {
        let Some(display) = display else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let param = self.update_tilt_com_tilt_display_axis_id_boolean(Some(display), axis_id, true);
        let Some(param) = param else {
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        // `if (processTrack != null)`: never null after construction.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        self.send_msg(
            self.tilt_process(
                axis_id,
                process_result_display.clone(),
                process_series,
                Arc::new(param),
                None,
                Some(processing_method),
            ),
            process_result_display,
        );
    }

    /// Java private `trialTilt(ProcessResultDisplay, ProcessSeries,
    /// TrialTiltDisplay, AxisID, DialogType, ProcessingMethod)`
    /// (ApplicationManager.java:6404).  Start a tilt process in trial mode.
    fn trial_tilt(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        display: Option<&dyn TrialTiltDisplay>,
        axis_id: AxisID,
        dialog_type: DialogType,
        processing_method: ProcessingMethod,
    ) {
        let Some(display) = display else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let param = self.update_trial_tilt_com(Some(display), axis_id, true);
        let Some(param) = param else {
            self.send_msg(
                Some(ProcessResult::FailedToStart),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        // `if (processTrack != null)`: never null after construction.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        self.send_msg(
            self.tilt_process(
                axis_id,
                process_result_display.clone(),
                process_series,
                Arc::new(param),
                None,
                Some(processing_method),
            ),
            process_result_display,
        );
    }

    /// Java private `tilt3dFindReprojectProcess(AxisID, ProcessResultDisplay,
    /// ProcessSeries, ConstTiltParam, String, ProcessName)`
    /// (ApplicationManager.java:6438).  Tilt process initiator.  Since tilt can be
    /// started from multiple points in the process chain we need separate the
    /// execution from the parameter collection and state updating.
    fn tilt3d_find_reproject_process(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: Arc<TiltParam>,
        process_title: Option<&str>,
        process_name: ProcessName,
    ) -> Option<ProcessResult> {
        let thread_name = match self.get_process_mgr().tilt3d_find_reproject(
            axis_id,
            process_result_display
                .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef),
            process_series.map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
            param,
            process_title,
            process_name,
        ) {
            Ok(thread_name) => thread_name,
            // `catch (final AxisBusyException e)`
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute tilt_3dfind{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(axis_id),
                );
                return Some(ProcessResult::FailedToStart);
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&ProcessName::TILT_3D_FIND_REPROJECT.to_string().as_str()),
                axis_id,
                Some(&ProcessName::TILT_3D_FIND_REPROJECT),
            );
        }
        None
    }

    /// Java private `tilt3dFindProcess(AxisID, ProcessResultDisplay, ProcessSeries,
    /// ConstTiltParam, String, ProcessName, ProcessingMethod)`
    /// (ApplicationManager.java:6468).  Tilt_3dfind process initiator.
    #[allow(clippy::too_many_arguments)]
    fn tilt3d_find_process(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: Arc<TiltParam>,
        process_title: Option<&str>,
        process_name: ProcessName,
        processing_method: Option<ProcessingMethod>,
    ) -> Option<ProcessResult> {
        let thread_name = match self.get_process_mgr().tilt3d_find(
            axis_id,
            process_result_display
                .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef),
            process_series.map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
            param,
            process_title,
            process_name,
            processing_method,
        ) {
            Ok(thread_name) => thread_name,
            // `catch (final AxisBusyException e)`
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute tilt_3dfind{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(axis_id),
                );
                return Some(ProcessResult::FailedToStart);
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java private `tiltProcess(AxisID, ProcessResultDisplay, ProcessSeries,
    /// ConstTiltParam, String, ProcessingMethod)` (ApplicationManager.java:6496).
    /// Tilt process initiator.
    fn tilt_process(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: Arc<TiltParam>,
        process_title: Option<&str>,
        processing_method: Option<ProcessingMethod>,
    ) -> Option<ProcessResult> {
        self.close_imod_file_key_and_name(
            Some(&*file_key::TRIAL_TOMOGRAM),
            Some(&param.get_output_file()),
            Some(axis_id),
            true,
        );
        let thread_name = match self.get_process_mgr().tilt(
            axis_id,
            process_result_display
                .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef),
            process_series.map(|series| Arc::new(EdtRef::new(series)) as ProcessSeriesRef),
            param,
            process_title,
            processing_method,
        ) {
            Ok(thread_name) => thread_name,
            // `catch (final AxisBusyException e)`
            Err(e) => {
                eprintln!("{e}");
                let message = vec![
                    format!("Can not execute tilt{}.com", axis_id.get_extension()),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(axis_id),
                );
                return Some(ProcessResult::FailedToStart);
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        None
    }

    /// Java private `getParametersForTilt3dFindReproject(TiltParam, AxisID)`
    /// (ApplicationManager.java:6525).  Update the non-dialog dependent parts of the
    /// tilt param in tilt_3dfind_reproject.com.
    fn get_parameters_for_tilt3d_find_reproject(
        &'static self,
        param: &mut TiltParam,
        axis_id: AxisID,
    ) {
        param.set_project_model(&file_type::CLASS.find_beads_3d_output_model);
        param.set_output_file(
            file_type::CLASS
                .ccd_eraser_beads_input_model
                .get_file_name(Some(self), Some(axis_id))
                .as_deref(),
        );
        param.set_process_name(ProcessName::TILT_3D_FIND_REPROJECT);
    }

    /// Java `copyTilt3dFindReprojectCom(AxisID)` (ApplicationManager.java:6539).
    /// Backup tilt_3dfind_reproject.com.  Copy tilt_3dfind_reproject.com from
    /// tilt_3dfind.com.  Modify it so that it is correct for reprojecting and save
    /// it.  Called on a process thread; it touches no dialog.
    pub fn copy_tilt3d_find_reproject_com(&'static self, axis_id: AxisID) {
        let tilt3d_find_com = PathBuf::from(utilities::java_io_file_new(
            &self.get_property_user_dir().unwrap_or_default(),
            &ProcessName::TILT_3D_FIND.get_comscript(axis_id),
        ));
        let tilt3d_find_reproject_com = PathBuf::from(utilities::java_io_file_new(
            &self.get_property_user_dir().unwrap_or_default(),
            &ProcessName::TILT_3D_FIND_REPROJECT.get_comscript(axis_id),
        ));
        if !tilt3d_find_com.exists() {
            eprintln!(
                "ERROR:  {} does not exist.  Unable to create {}",
                utilities::java_io_file_get_name(&tilt3d_find_com.to_string_lossy()),
                utilities::java_io_file_get_name(&tilt3d_find_reproject_com.to_string_lossy())
            );
            return;
        }
        if tilt3d_find_reproject_com.exists() {
            // Backup tilt_3dfind_reproject.com
            match LogFile::get_instance_file(
                Some(&tilt3d_find_reproject_com),
                Some(self.get_emergency_monitor(Some(axis_id))),
            )
            .and_then(|log_file| log_file.backup())
            {
                Ok(_) => {}
                // `catch (final LockException e) {}`
                Err(LogFileError::Lock(_)) => {}
                // `catch (final LogFileException | IOException e)`
                Err(e) => eprintln!("{e}"),
            }
        }
        // Copy file
        match utilities::copy_file(
            Some(self),
            Some(axis_id),
            Some(&PathBuf::from(utilities::java_io_file_new(
                &self.get_property_user_dir().unwrap_or_default(),
                &ProcessName::TILT_3D_FIND.get_comscript(axis_id),
            ))),
            Some(&tilt3d_find_reproject_com),
            false,
            false,
            false,
        ) {
            Ok(()) => {
                self.get_com_script_manager()
                    .load_tilt3d_find_reproject(axis_id);
            }
            // `catch (final IOException | LogFileException | LockException e)`
            Err(e) => {
                eprintln!("{e}");
                eprintln!(
                    "ERROR:  Unable to create {}",
                    utilities::java_io_file_get_name(&tilt3d_find_reproject_com.to_string_lossy())
                );
            }
        }
        // Update the tilt command in the com file.
        // `TiltParam tiltParam = null; try { ... } catch (NumberFormatException)`:
        // none of the translated calls below reports a number format error.
        let mut tilt_param = self
            .get_com_script_manager()
            .get_tilt_param_from_tilt3d_find_reproject(axis_id);
        self.get_parameters_for_tilt3d_find_reproject(&mut tilt_param, axis_id);
        self.update_exclude_list(&mut tilt_param, axis_id);
        self.get_com_script_manager()
            .save_tilt3d_find_reproject(&tilt_param, axis_id);
    }

    /// Java `updateTilt3dFindReprojectCom(TiltDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:6591).  Update the tilt_3dfind_reproject.com.
    pub fn update_tilt3d_find_reproject_com(
        &'static self,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<TiltParam> {
        // `TiltParam tiltParam = null;`
        let mut tilt_param = self
            .get_com_script_manager()
            .get_tilt_param_from_tilt3d_find_reproject(axis_id);
        self.get_parameters_for_tilt3d_find_reproject(&mut tilt_param, axis_id);
        if let Some(display) = display {
            // The source ignores the returned boolean.
            if let Err(except) = display.get_parameters(&mut tilt_param, do_validation) {
                match except {
                    TiltDisplayException::NumberFormat(_)
                    | TiltDisplayException::InvalidParameter(_) => {
                        eprintln!("{except}");
                        let error_message = vec![
                            "Tilt Parameter Syntax Error".to_string(),
                            format!("Axis: {}", axis_id.get_extension()),
                            except.to_string(),
                        ];
                        ui_harness::open_message_dialog_array_from_process(
                            Some(self),
                            &error_message,
                            "Tilt Parameter Syntax Error",
                            Some(axis_id),
                        );
                        return None;
                    }
                    TiltDisplayException::Io(_) => {
                        let error_message = vec![
                            "Tilt Parameter".to_string(),
                            format!("Axis: {}", axis_id.get_extension()),
                            except.to_string(),
                        ];
                        ui_harness::open_message_dialog_array_from_process(
                            Some(self),
                            &error_message,
                            "Tilt Parameter",
                            Some(axis_id),
                        );
                        return None;
                    }
                }
            }
        }
        self.update_exclude_list(&mut tilt_param, axis_id);
        self.get_com_script_manager()
            .save_tilt3d_find_reproject(&tilt_param, axis_id);
        Some(tilt_param)
    }

    /// Java `updateExcludeList(TiltParam, AxisID)` (ApplicationManager.java:6634).
    pub fn update_exclude_list(&'static self, param: &mut TiltParam, axis_id: AxisID) {
        // `TiltalignLog.getInstance`,
        // `exists`, `isSuccess` and `getExcludeList`; written against the
        // Java-derived API.
        let log = TiltalignLog::get_instance(self, axis_id);
        if !log.exists() || !log.is_success() {
            self.get_com_script_manager().load_align(axis_id);
            let tiltalign_param = self.get_com_script_manager().get_tiltalign_param(axis_id);
            param.set_exclude_list(Some(&tiltalign_param.get_exclude_list()));
        } else {
            param.set_exclude_list(log.get_exclude_list().as_deref());
        }
    }

    /// Java `updateTilt3dFindCom(TiltDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:6652).  Update the tilt_3dfind.com from the
    /// TomogramGenerationDialog.
    pub fn update_tilt3d_find_com(
        &'static self,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<TiltParam> {
        // Set a reference to the correct object
        let Some(display) = display else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update tilt_3dfind?.com without an active display",
                "Program logic error",
                Some(axis_id),
            );
            return None;
        };
        // `TiltParam tiltParam = null;`
        let mut tilt_param = self
            .get_com_script_manager()
            .get_tilt_param_from_tilt3d_find(axis_id);
        tilt_param.set_fiducialess(self.get_meta_data().is_fiducialess(axis_id));
        match display.get_parameters(&mut tilt_param, do_validation) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(
                except @ (TiltDisplayException::NumberFormat(_)
                | TiltDisplayException::InvalidParameter(_)),
            ) => {
                eprintln!("{except}");
                let error_message = vec![
                    "Tilt Parameter Syntax Error".to_string(),
                    format!("Axis: {}", axis_id.get_extension()),
                    except.to_string(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &error_message,
                    "Tilt Parameter Syntax Error",
                    Some(axis_id),
                );
                return None;
            }
            Err(e @ TiltDisplayException::Io(_)) => {
                let error_message = vec![
                    "Tilt Parameter".to_string(),
                    format!("Axis: {}", axis_id.get_extension()),
                    e.to_string(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &error_message,
                    "Tilt Parameter",
                    Some(axis_id),
                );
                return None;
            }
        }
        self.update_exclude_list(&mut tilt_param, axis_id);
        self.get_com_script_manager()
            .save_tilt3d_find(&tilt_param, axis_id);
        Some(tilt_param)
    }

    /// Java `updateTiltCom(TiltDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:6708).  Update the tilt.com from the
    /// TomogramGenerationDialog.
    pub fn update_tilt_com_tilt_display_axis_id_boolean(
        &'static self,
        display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<TiltParam> {
        // Set a reference to the correct object
        let Some(display) = display else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update tilt?.com without an active tomogram generation dialog",
                "Program logic error",
                Some(axis_id),
            );
            return None;
        };
        // `TiltParam tiltParam = null;`
        let mut tilt_param = self.get_com_script_manager().get_tilt_param(axis_id);
        tilt_param.set_fiducialess(self.get_meta_data().is_fiducialess(axis_id));
        match display.get_parameters(&mut tilt_param, do_validation) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(
                except @ (TiltDisplayException::NumberFormat(_)
                | TiltDisplayException::InvalidParameter(_)),
            ) => {
                eprintln!("{except}");
                let error_message = vec![
                    "Tilt Parameter Syntax Error".to_string(),
                    format!("Axis: {}", axis_id.get_extension()),
                    except.to_string(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &error_message,
                    "Tilt Parameter Syntax Error",
                    Some(axis_id),
                );
                return None;
            }
            Err(e @ TiltDisplayException::Io(_)) => {
                let error_message = vec![
                    "Tilt Parameter".to_string(),
                    format!("Axis: {}", axis_id.get_extension()),
                    e.to_string(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &error_message,
                    "Tilt Parameter",
                    Some(axis_id),
                );
                return None;
            }
        }
        let output_file_name = file_type::CLASS
            .tilt_output
            .get_file_name(Some(self), Some(axis_id));
        // outputFileName = metaData.getDatasetName() + "_full.rec";
        // outputFileName = metaData.getDatasetName() + axisID.getExtension() + ".rec";
        tilt_param.set_output_file(output_file_name.as_deref());
        if self.get_meta_data().get_view_type() == ViewType::Montage {
            // binning is currently always 1 and correct size should be coming from
            // copytomocoms
            // tiltParam.setMontageFullImage(propertyUserDir,
            // tomogramGenerationDialog.getBinning());
        }
        UIExpertUtilities::INSTANCE.roll_tilt_com_angles(self, axis_id);
        self.update_exclude_list(&mut tilt_param, axis_id);
        self.get_com_script_manager()
            .save_tilt(&tilt_param, axis_id);
        self.get_meta_data()
            .set_fiducialess(axis_id, tilt_param.is_fiducialess());
        Some(tilt_param)
    }

    /// Java `updateMultifiltSetupCom(MultifiltSetupDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:6770).
    pub fn update_multifilt_setup_com(
        &'static self,
        display: Option<&dyn MultifiltSetupDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<MultifiltSetupParam> {
        // Set a reference to the correct object
        let Some(display) = display else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                &format!(
                    "Can not update {} without an active tomogram generation dialog",
                    file_type::CLASS
                        .multifilt_setup_comscript
                        .get_file_name(Some(self), Some(axis_id))
                        .unwrap_or_else(|| "null".to_string())
                ),
                "Program logic error",
                Some(axis_id),
            );
            return None;
        };
        // `MultifiltSetupParam param = null;`
        let mut param = self
            .get_com_script_manager()
            .get_multifilt_setup_param(axis_id);
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        self.get_com_script_manager()
            .save_multifilt_setup_multifilt_setup(&param, axis_id);
        // Set the IMOD_OUTPUT_FORMAT environment variable in new image file name style
        // datasets only.
        if !self.get_meta_data().base().is_old_image_filename_style() {
            let set_env_param = self
                .get_com_script_manager()
                .get_set_env_param_from_multifilt_setup(axis_id, imod_output_format::ENV_VAR);
            if set_env_param.is_none() {
                // Keep the existing setting if it was set. If not then add it.
                let mut set_env_param = SetEnvParam::new(Some(imod_output_format::ENV_VAR));
                set_env_param.set_value(Some(
                    &self
                        .get_meta_data()
                        .base()
                        .get_image_output_format()
                        .to_string(),
                ));
                self.get_com_script_manager().save_multifilt_setup_set_env(
                    &set_env_param,
                    axis_id,
                    imod_output_format::ENV_VAR,
                );
            }
        }
        Some(param)
    }

    /// Java `updateCtf3dSetupCom(Ctf3dSetupDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:6801).
    pub fn update_ctf3d_setup_com(
        &'static self,
        display: Option<&dyn Ctf3dSetupDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<Ctf3dSetupParam> {
        // Set a reference to the correct object
        let Some(display) = display else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                &format!(
                    "Can not update {} without an active tomogram generation dialog",
                    file_type::CLASS
                        .ctf_3d_setup_comscript
                        .get_file_name(Some(self), Some(axis_id))
                        .unwrap_or_else(|| "null".to_string())
                ),
                "Program logic error",
                Some(axis_id),
            );
            return None;
        };
        // `Ctf3dSetupParam param = null;`
        let mut param = self.get_com_script_manager().get_ctf3d_setup_param(axis_id);
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        // ParallelPanel
        let parallel_panel = self
            .main_panel
            .get()
            .and_then(|main_panel| main_panel.get_parallel_panel(axis_id));
        let Some(parallel_panel) = parallel_panel else {
            if do_validation {
                ParallelPanel::reset_parameters(&mut param);
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    Some(axis_id),
                );
            }
            return None;
        };
        // The source ignores the returned boolean.
        parallel_panel.get_parameters_ctf3d_setup_param_boolean_boolean(
            &mut param,
            display.is_run_slabs_in_parallel(),
            true,
        );
        self.get_com_script_manager()
            .save_ctf3d_setup(&param, axis_id);
        Some(param)
    }

    /// Java `updateSubtomoSetupCom(SubtomoSetupDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:6832).
    pub fn update_subtomo_setup_com(
        &'static self,
        display: Option<&dyn SubtomoSetupDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<SubtomoSetupParam> {
        // Set a reference to the correct object
        let Some(display) = display else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                &format!(
                    "Can not update {} without an active tomogram generation dialog",
                    file_type::CLASS
                        .subtomo_setup_comscript
                        .get_file_name(Some(self), Some(axis_id))
                        .unwrap_or_else(|| "null".to_string())
                ),
                "Program logic error",
                Some(axis_id),
            );
            return None;
        };
        // `SubtomoSetupParam param = null;`
        let mut param = self
            .get_com_script_manager()
            .get_subtomo_setup_param(axis_id);
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        self.get_com_script_manager().save_subtomo_setup(&param);
        Some(param)
    }

    /// Java `updateAltTomoSetupCom(AltStackDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:6851).
    pub fn update_alt_tomo_setup_com(
        &'static self,
        display: Option<&dyn AltStackDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<AltTomoSetupParam> {
        // Set a reference to the correct object
        let Some(display) = display else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                &format!(
                    "Can not update {} without an active tomogram generation dialog",
                    file_type::CLASS
                        .alt_tomo_setup_comscript
                        .get_file_name_with_axis_type(
                            Some(self),
                            None,
                            Some(AxisType::SingleAxis),
                            Some(AxisID::Only),
                        )
                        .unwrap_or_else(|| "null".to_string())
                ),
                "Program logic error",
                Some(axis_id),
            );
            return None;
        };
        // `AltTomoSetupParam param = null;`
        let mut param = self
            .get_com_script_manager()
            .get_alt_tomo_setup_param(AxisID::Only);
        if !display.get_parameters_alt_tomo_setup_param_boolean(&mut param, do_validation) {
            return None;
        }
        param.reset_just_restore_initial_set();
        param.set_command_mode(alt_tomo_setup_param::Mode::AltTomoSetup);
        self.get_com_script_manager()
            .save_alt_tomo_setup(&param, AxisID::Only);
        Some(param)
    }

    /// Java `updateAltTomoSetupComRestoreSwappedFiles(AltStackDisplay, AxisID,
    /// boolean)` (ApplicationManager.java:6874).
    pub fn update_alt_tomo_setup_com_restore_swapped_files(
        &'static self,
        display: Option<&dyn AltStackDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<AltTomoSetupParam> {
        // Set a reference to the correct object
        let Some(display) = display else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                &format!(
                    "Can not update {} without an active tomogram generation dialog",
                    file_type::CLASS
                        .alt_tomo_setup_comscript
                        .get_file_name_with_axis_type(
                            Some(self),
                            None,
                            Some(AxisType::SingleAxis),
                            Some(AxisID::Only),
                        )
                        .unwrap_or_else(|| "null".to_string())
                ),
                "Program logic error",
                Some(axis_id),
            );
            return None;
        };
        // `AltTomoSetupParam param = null;`
        let mut param = self
            .get_com_script_manager()
            .get_alt_tomo_setup_param(AxisID::Only);
        if !display.get_parameters_alt_tomo_setup_param_boolean(&mut param, do_validation) {
            return None;
        }

        param.set_just_restore_initial_set(true);
        let tomogram_state = self.get_state();
        param.reset_rootname_to_process();
        param.reset_even_and_odd_pairs();
        param.reset_axis_to_process();
        param.set_rootname_to_process(Some(&tomogram_state.get_alt_tomo_rootname_to_process()));
        param.set_even_and_odd_pairs(tomogram_state.is_alt_tomo_even_and_odd_pairs());
        param.set_axis_to_process(Some(&tomogram_state.get_alt_tomo_axis_to_process()));
        param.set_command_mode(alt_tomo_setup_param::Mode::JustRestoreInitialSet);
        self.get_com_script_manager()
            .save_alt_tomo_setup(&param, AxisID::Only);
        Some(param)
    }

    /// Java `updateTrialTiltCom(TrialTiltDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:6911).  Update the tilt.com from the
    /// TomogramGenerationDialog.  Use the trial tomogram filename specified in the
    /// TomogramGenerationDialog.
    pub fn update_trial_tilt_com(
        &'static self,
        display: Option<&dyn TrialTiltDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<TiltParam> {
        // Set a reference to the correct object
        let Some(display) = display else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update tilt?.com without an active tomogram generation dialog",
                "Program logic error",
                Some(axis_id),
            );
            return None;
        };
        // `TiltParam tiltParam = null;`
        let mut tilt_param = self.get_com_script_manager().get_tilt_param(axis_id);
        tilt_param.set_fiducialess(self.get_meta_data().is_fiducialess(axis_id));
        match display.get_parameters(&mut tilt_param, do_validation) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(
                except @ (TiltDisplayException::NumberFormat(_)
                | TiltDisplayException::InvalidParameter(_)),
            ) => {
                eprintln!("{except}");
                let error_message = vec![
                    "Tilt Parameter Syntax Error".to_string(),
                    format!("Axis: {}", axis_id.get_extension()),
                    except.to_string(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &error_message,
                    "Tilt Parameter Syntax Error",
                    Some(axis_id),
                );
                return None;
            }
            Err(e @ TiltDisplayException::Io(_)) => {
                let error_message = vec![
                    "Tilt Parameter".to_string(),
                    format!("Axis: {}", axis_id.get_extension()),
                    e.to_string(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &error_message,
                    "Tilt Parameter",
                    Some(axis_id),
                );
                return None;
            }
        }
        let trial_tomogram_name = display.get_trial_tomogram_name();
        tilt_param.set_output_file(trial_tomogram_name.as_deref());

        if self.get_meta_data().get_view_type() == ViewType::Montage {
            // binning is currently always 1 and correct size should be coming from
            // copytomocoms
            // tiltParam.setMontageFullImage(propertyUserDir,
            // tomogramGenerationDialog.getBinning());
        }
        UIExpertUtilities::INSTANCE.roll_tilt_com_angles(self, axis_id);
        self.update_exclude_list(&mut tilt_param, axis_id);
        self.get_com_script_manager()
            .save_tilt(&tilt_param, axis_id);
        self.get_meta_data()
            .set_fiducialess(axis_id, tilt_param.is_fiducialess());
        Some(tilt_param)
    }

    /// Java private `updateSplittiltParam(TiltDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:6972).
    fn update_splittilt_param(
        &self,
        tilt_display: Option<&dyn TiltDisplay>,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<SplittiltParam> {
        let tilt_display = tilt_display?;
        let mut param = SplittiltParam::new(axis_id);
        if !tilt_display.get_parameters_splittilt(&mut param, do_validation) {
            return None;
        }
        param.set_separate_chunks(cpu_adoc::INSTANCE.is_separate_chunks());
        Some(param)
    }

    /// Java `imodFullVolume(AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:6991).  Open 3dmod to view the tomogram.
    pub fn imod_full_volume(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        match self
            .get_imod_manager()
            .open_string_axis_id_run3dmod_menu_options(
                imod_manager::FULL_VOLUME_KEY,
                Some(axis_id),
                Some(menu_options),
            ) {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.0,
                    "Can't open 3dmod with the tomogram",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imod(FileType, AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:7016).  Open 3dmod to view a file.
    pub fn imod_file_type_axis_id_run3dmod_menu_options(
        &'static self,
        file_type: &FileType,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
    ) {
        // Fixed in translation: a file type with no 3dmod key makes Java throw
        // NullPointerException in ImodManager.getPrivateKey; nothing is opened.
        let Some(key) = file_type.get_imod_manager_key() else {
            return;
        };
        match self
            .get_imod_manager()
            .open_string_axis_id_run3dmod_menu_options(key, Some(axis_id), Some(menu_options))
        {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.0,
                    "Can't open 3dmod",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imod(FileType, FileType, AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:7041).  Open 3dmod to view a file with a model.
    pub fn imod_file_type_file_type_axis_id_run3dmod_menu_options(
        &'static self,
        file_type: &FileType,
        model_file_type: &FileType,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
    ) {
        // Fixed in translation: a file type with no 3dmod key makes Java throw
        // NullPointerException in ImodManager.getPrivateKey; nothing is opened.
        let Some(key) = file_type.get_imod_manager_key() else {
            return;
        };
        match self
            .get_imod_manager()
            .open_string_axis_id_string_run3dmod_menu_options(
                key,
                Some(axis_id),
                model_file_type
                    .get_file_name(Some(self), Some(axis_id))
                    .as_deref(),
                Some(menu_options),
            ) {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.0,
                    "Can't open 3dmod",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodReprojectModel(AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:7067).  Open 3dmod to view a file with a model.
    pub fn imod_reproject_model(&'static self, axis_id: AxisID, menu_options: Run3dmodMenuOptions) {
        let file_type: &FileType = if self
            .get_state()
            .is_stack_using_newst_or_blend_3d_find_output(axis_id)
        {
            &file_type::CLASS.newst_or_blend_3d_find_output
        } else {
            &file_type::CLASS.aligned_stack
        };
        // Fixed in translation: a file type with no 3dmod key makes Java throw
        // NullPointerException in ImodManager.getPrivateKey; nothing is opened.
        let Some(key) = file_type.get_imod_manager_key() else {
            return;
        };
        let imod_manager = self.get_imod_manager();
        match (|| -> Result<(), ImodManagerError> {
            imod_manager.set_open_bead_fixer(key, Some(axis_id), true)?;
            imod_manager.set_auto_center(key, Some(axis_id), true)?;
            imod_manager.open_string_axis_id_string_run3dmod_menu_options(
                key,
                Some(axis_id),
                file_type::CLASS
                    .ccd_eraser_beads_input_model
                    .get_file_name(Some(self), Some(axis_id))
                    .as_deref(),
                Some(menu_options),
            )?;
            Ok(())
        })() {
            Ok(()) => {}
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.0,
                    "Can't open 3dmod",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodTestVolume(AxisID, Run3dmodMenuOptions, String)`
    /// (ApplicationManager.java:7102).  Open 3dmod on the current test volume.
    pub fn imod_test_volume_axis_id_run3dmod_menu_options_string(
        &'static self,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
        trial_tomogram_name: Option<&str>,
    ) {
        let imod_manager = self.get_imod_manager();
        match (|| -> Result<(), ImodManagerError> {
            imod_manager.new_imod_string_axis_id_string(
                imod_manager::TRIAL_TOMOGRAM_KEY,
                Some(axis_id),
                trial_tomogram_name,
            )?;
            imod_manager.open_string_axis_id_run3dmod_menu_options(
                imod_manager::TRIAL_TOMOGRAM_KEY,
                Some(axis_id),
                Some(menu_options),
            )?;
            Ok(())
        })() {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                // The source builds this array and never shows it.
                let _message = [
                    format!(
                        "Unable to open specified tomogram:{}",
                        trial_tomogram_name.unwrap_or("null")
                    ),
                    "Does it exist in the working directory?".to_string(),
                ];
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!("{}\nCan't open 3dmod with the tomogram", except.0),
                    "Cannot Open 3dmod",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `commitTestVolume(AxisID, ProcessResultDisplay, String)`
    /// (ApplicationManager.java:7127).
    pub fn commit_test_volume_axis_id_process_result_display_string(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        trial_tomogram_name: Option<&str>,
    ) -> Option<ProcessResult> {
        let _ = process_result_display;
        // Check to see if the trial tomogram exist
        let trial_tomogram_file = PathBuf::from(utilities::java_io_file_new(
            &self.get_property_user_dir().unwrap_or_default(),
            trial_tomogram_name.unwrap_or("null"),
        ));
        if !trial_tomogram_file.exists() {
            let message = vec![
                format!(
                    "The specified tomogram does not exist:{}",
                    trial_tomogram_name.unwrap_or("null")
                ),
                "It must be calculated before commiting".to_string(),
            ];
            ui_harness::open_message_dialog_array_from_process(
                Some(self),
                &message,
                "Can't rename tomogram",
                Some(axis_id),
            );
            return Some(ProcessResult::FailedToStart);
        }
        // rename the trial tomogram to the output filename of appropriate
        // tilt.com
        let output_file: &FileType = &file_type::CLASS.tilt_output;
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_progress_bar_string_int_boolean_axis_id(
                Some(
                    &format!(
                        "Using trial tomogram: {}",
                        trial_tomogram_name.unwrap_or("null")
                    )
                    .as_str(),
                ),
                1,
                false,
                axis_id,
            );
        }
        if output_file
            .get_file(Some(self), Some(axis_id))
            .is_some_and(|file| file.exists())
            && trial_tomogram_file.exists()
        {
            match self.backup_image_file(Some(output_file), Some(axis_id)) {
                Ok(()) => {}
                // `catch (final LockException e)`
                Err(LogFileError::Lock(_)) => {
                    if let Some(main_panel) = self.main_panel.get() {
                        main_panel.stop_progress_bar_axis_id_process_end_state(
                            axis_id,
                            Some(ProcessEndState::FileLockFailure),
                        );
                    }
                    return Some(ProcessResult::Failed);
                }
                // `catch (final IOException | LogFileException except)`
                Err(except) => {
                    ui_harness::open_message_dialog_from_process(
                        Some(self),
                        &format!(
                            "Unable to backup {}\n{}",
                            utilities::java_io_file_get_absolute_path(
                                &output_file
                                    .get_file(Some(self), Some(axis_id))
                                    .map(|file| file.to_string_lossy().into_owned())
                                    .unwrap_or_default()
                            ),
                            except.get_message()
                        ),
                        "File Rename Error (5)",
                        Some(axis_id),
                    );
                    if let Some(main_panel) = self.main_panel.get() {
                        main_panel.stop_progress_bar_axis_id(axis_id);
                    }
                    return Some(ProcessResult::Failed);
                }
            }
        }
        match self.rename_image_file_from_key(
            Some(&*file_key::TRIAL_TOMOGRAM),
            Some(&trial_tomogram_file),
            Some(output_file),
            Some(axis_id),
            true,
        ) {
            Ok(()) => {}
            // `catch (final LockException e)`
            Err(LogFileError::Lock(_)) => {
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        Some(ProcessEndState::FileLockFailure),
                    );
                }
                return Some(ProcessResult::Failed);
            }
            // `catch (final IOException | LogFileException except)`
            Err(except) => {
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.get_message(),
                    "File Rename Error (6)",
                    Some(axis_id),
                );
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id(axis_id);
                }
                return Some(ProcessResult::Failed);
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id(axis_id);
        }
        Some(ProcessResult::Succeeded)
    }

    /// Java `deleteIntermediateImageStacks(AxisID, ProcessResultDisplay)`
    /// (ApplicationManager.java:7183).  Delete the pre-aligned and aligned stack
    /// for the specified axis.
    pub fn delete_intermediate_image_stacks(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
    ) {
        let process_result_display: Option<ProcessResultDisplayRef> = process_result_display
            .map(|display| Arc::new(EdtRef::new(display)) as ProcessResultDisplayRef);
        self.send_msg_process_starting(process_result_display.as_ref());
        if self.is_axis_busy(axis_id, process_result_display.clone()) {
            return;
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_progress_bar_string_int_boolean_axis_id(
                Some("Deleting aligned image stack"),
                1,
                false,
                axis_id,
            );
        }
        // `Utilities.deleteFileType` reports its own failure (a message dialog) and
        // returns a boolean the source ignores.
        let _ = utilities::delete_file_type(self, Some(axis_id), &file_type::CLASS.aligned_stack);
        let _ =
            utilities::delete_file_type(self, Some(axis_id), &file_type::CLASS.tilt_3d_find_output);
        let _ = utilities::delete_file_type(
            self,
            Some(axis_id),
            &file_type::CLASS.newst_or_blend_3d_find_output,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id(axis_id);
        }
        self.send_msg_process_succeeded(process_result_display.as_ref());
    }

    /// Java `updateAlignedStackBinning(AxisID)` (ApplicationManager.java:7197).
    pub fn update_aligned_stack_binning(&'static self, axis_id: AxisID) {
        // Posted: Java calls this on the process thread.
        event_queue::invoke_later(move || {
            if axis_id == AxisID::Second {
                if let Some(expert) = self.final_aligned_stack_expert_b.get() {
                    expert.update_aligned_stack_binning();
                }
            } else if let Some(expert) = self.final_aligned_stack_expert_a.get() {
                expert.update_aligned_stack_binning();
            }
        });
    }

    /// Java `openTomogramCombinationDialog()` (ApplicationManager.java:7212).  Open
    /// the tomogram combination dialog.
    pub fn open_tomogram_combination_dialog(&'static self) {
        // Check to see if the com files are present otherwise pop up a dialog
        // box informing the user to run the setup process
        if !UIExpertUtilities::INSTANCE.are_scripts_created(
            self,
            self.get_meta_data(),
            AxisID::Only,
        ) {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(AxisID::Only);
            }
            return;
        }
        // Verify that this process is applicable
        if self.get_meta_data().get_axis_type() == AxisType::SingleAxis {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "This step is valid only for a dual axis tomogram",
                "Invalid tomogram combination selection",
                Some(AxisID::Only),
            );
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(AxisID::Only);
            }
            return;
        }
        let action_message = self
            .set_current_dialog_type(Some(DialogType::TomogramCombination), Some(AxisID::First));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.select_button(AxisID::First, "Tomogram Combination");
        }
        let mut init = false;
        if !self.tomogram_combination_dialog.is_some() {
            // Get the setupcombine parameters and set the default patch
            // boundaries if
            // they have not already been set
            // `final CombineParams combineParams = metaData.getCombineParams();` - the
            // metadata's own object, locked for each statement that uses it.
            if !self
                .get_meta_data()
                .get_combine_params()
                .is_patch_boundary_set()
                && !self.get_meta_data().is_new_batchruntomo_combine_settings()
                && !self.get_state().is_new_batchruntomo_combine_settings()
            {
                eprintln!("No NewBatchruntomoCombineSettings");
                init = true;
                // The first time combine is opened for this dataset, set tomogram size
                TomogramTool::save_tomogram_size(self, AxisID::First, AxisID::Only);
                TomogramTool::save_tomogram_size(self, AxisID::Second, AxisID::Only);
                let rec_file_name;
                let match_mode = self.get_meta_data().get_combine_params().get_match_mode();
                if match_mode.is_none() || match_mode == Some(MatchMode::BToA) {
                    rec_file_name = file_type::CLASS
                        .tilt_output_dual
                        .get_file_name(Some(self), Some(AxisID::First))
                        .unwrap_or_default();
                    // recFileName = metaData.getDatasetName() + "a.rec";
                } else {
                    rec_file_name = file_type::CLASS
                        .tilt_output_dual
                        .get_file_name(Some(self), Some(AxisID::Second))
                        .unwrap_or_default();
                    // recFileName = metaData.getDatasetName() + "b.rec";
                }
                let result = self
                    .get_meta_data()
                    .get_combine_params()
                    .set_default_patch_boundaries(&rec_file_name);
                // `catch (final IOException except)`.  A `NumberFormatException` is
                // unchecked in Java and escapes the method; here it takes the
                // `InvalidParameterException` arm below (BUGS.md, fixed in translation).
                if let Err(crate::imod::etomo::util::mrc_header::ReadError::Io(except)) = &result {
                    ui_harness::open_message_dialog_from_process(
                        Some(self),
                        except,
                        &format!("IO Error: {}", rec_file_name),
                        Some(AxisID::Only),
                    );
                    // Delete the dialog
                    self.tomogram_combination_dialog.set(None);
                    if let Some(main_panel) = self.main_panel.get() {
                        main_panel.show_blank_process(AxisID::Only);
                    }
                    return;
                }
                // `catch (final InvalidParameterException except)`.
                if let Err(except) = result {
                    let except = except.to_string();
                    // Upstream bug fixed (ApplicationManager.java:7264): the source
                    // concatenates the `String[]` itself (`detailedMessage + "\n"`),
                    // which prints the array's identity hash (`[Ljava.lang.String;@...`)
                    // instead of its lines.  The translation joins the four lines.
                    let detailed_message = [
                        "Unable to set default patch boundaries".to_string(),
                        "Are both tomograms computed and available?".to_string(),
                        String::new(),
                        except,
                    ];
                    ui_harness::open_message_dialog_from_process(
                        Some(self),
                        &format!(
                            "{}\nInvalid parameter: {}",
                            detailed_message.join("\n"),
                            rec_file_name
                        ),
                        "Invalid Parameter",
                        Some(AxisID::Only),
                    );
                    // Delete the dialog
                    self.tomogram_combination_dialog.set(None);
                    if let Some(main_panel) = self.main_panel.get() {
                        main_panel.show_blank_process(AxisID::Only);
                    }
                    return;
                }
            }
            if self.get_meta_data().is_new_batchruntomo_combine_settings()
                || self.get_state().is_new_batchruntomo_combine_settings()
            {
                // If batchruntomo data is in the metadata, then combine was run by batchruntomo.
                // The batchruntomo data must override other meta data. Transfer the batchruntomo
                // data to properties managed by etomo, and remove the original Batchruntomo data.
                self.get_meta_data().reset_batchruntomo_combine_settings();
                self.get_state().reset_batchruntomo_combine_settings();
                self.get_state().set_combine_scripts_created(true);
                self.get_state()
                    .set_combine_match_mode(self.get_meta_data().get_match_mode());
            }
            if !self.tomogram_combination_dialog.is_some() {
                utilities::timestamp_process_container_status(
                    Some("new"),
                    Some("TomogramCombinationDialog"),
                    Some(utilities::STARTED_STATUS),
                );
                self.tomogram_combination_dialog
                    .set(Some(TomogramCombinationDialog::new(self)));
                utilities::timestamp_process_container_status(
                    Some("new"),
                    Some("TomogramCombinationDialog"),
                    Some(utilities::FINISHED_STATUS),
                );
            }
            let tomogram_combination_dialog = self
                .tomogram_combination_dialog
                .get()
                .expect("tomogramCombinationDialog");
            // Fill in the dialog box params and set it to the appropriate state
            {
                let combine_params = self.get_meta_data().get_combine_params();
                tomogram_combination_dialog
                    .set_combine_params(&*combine_params as &dyn ConstCombineParams, init);
            }
            // TODO combine.com must have dualvolmatch command
            self.backwards_compatibility_combine_scripts_exist();
            // If setupcombine has been run load the com scripts, otherwise disable
            // the
            // apropriate panels in the tomogram combination dialog
            // tomogramCombinationDialog.enableCombineTabs(combineScriptsExist());
            if self.get_state().is_combine_scripts_created() {
                // Check to see if a solvematch.com file exists and load it if so
                // otherwise load the correct old solvematch* file
                let solvematch = PathBuf::from(utilities::java_io_file_new(
                    &self.get_property_user_dir().unwrap_or_default(),
                    "solvematch.com",
                ));
                if solvematch.exists() {
                    self.load_solvematch_void();
                } else {
                    // For backward compatibility using fiducialMatch instead of
                    // modelBased. ModelBased was not updated when fiducialMatch was
                    // changed and ficucialMatch wasn't update when modelBased was
                    // changed.
                    // But it looks like modelBased was never changed. In version 3.2.6
                    // CombineParam.setModelBased() was never called. So fiducialMatch is
                    // probably correct in earlier versions.
                    let mut model_based = false;
                    let fiducial_match = self
                        .get_meta_data()
                        .get_combine_params()
                        .get_fiducial_match();
                    if fiducial_match == Some(FiducialMatch::UseModel)
                        || fiducial_match == Some(FiducialMatch::UseModelOnly)
                    {
                        model_based = true;
                    }
                    self.load_solvematch_boolean(model_based);
                }
                self.load_dualvolmatch();
                self.load_matchvol1();
                self.load_patchcorr();
                self.load_matchorwarp();
                self.load_volcombine();
                let combine_comscript_state = self.load_combine_comscript();
                tomogram_combination_dialog.synchronize(
                    tomogram_combination_dialog::LBL_SETUP,
                    true, /* false */
                );
                // Upstream bug fixed (ApplicationManager.java:7335): the source
                // dereferences the result of `loadCombineComscript()`, which returns
                // null when combine.com cannot be read, and throws a
                // NullPointerException that leaves the dialog half opened.  The
                // translation skips the dualvolmatch upgrade when there is no state.
                if let Some(combine_comscript_state) = combine_comscript_state
                    && !combine_comscript_state
                        .is_dualvolmatch_present(&self.get_com_script_manager())
                    && self.setup_combine_com_only()
                {
                    self.load_combine_comscript();
                }
            } else {
                // force the user to set Z values on a new combine
                // make sure not destroying user entries by checking for patchcorr.com
                if !dataset_files::get_axis_only_com_file(self, Some(&ProcessName::PATCHCORR))
                    .exists()
                {
                    tomogram_combination_dialog.set_z_min("");
                    tomogram_combination_dialog.set_z_max("");
                }
            }
        }
        let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() else {
            return;
        };
        tomogram_combination_dialog.show();
        tomogram_combination_dialog.set_parameters_const_meta_data(self.get_meta_data());
        tomogram_combination_dialog
            .set_parameters_recon_screen_state(self.get_screen_state(AxisID::Only));
        // Show the process panel
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&tomogram_combination_dialog.get_container(), AxisID::First);
        }
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `imodMatchingModel(boolean, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:7363).  Open the matching models in the 3dmod
    /// reconstruction instances.
    pub fn imod_matching_model(&'static self, bin_by2: bool, menu_options: Run3dmodMenuOptions) {
        if !self.tomogram_combination_dialog.is_some() {
            return;
        }

        // FIXME do we want to be updating here break model (maybe it is saving
        // the bin by 2 state?
        if !self.update_combine_params(false, true) {
            return;
        }

        let imod_manager = self.get_imod_manager();
        match (|| -> Result<(), ImodManagerError> {
            if bin_by2 {
                imod_manager.set_binning_string_axis_id_int(
                    imod_manager::FULL_VOLUME_KEY,
                    Some(AxisID::First),
                    2,
                )?;
                imod_manager.set_binning_string_axis_id_int(
                    imod_manager::FULL_VOLUME_KEY,
                    Some(AxisID::Second),
                    2,
                )?;
            }
            imod_manager.set_open_contours(
                imod_manager::FULL_VOLUME_KEY,
                Some(AxisID::First),
                true,
            )?;
            imod_manager.set_open_contours(
                imod_manager::FULL_VOLUME_KEY,
                Some(AxisID::Second),
                true,
            )?;
            imod_manager.open_string_axis_id_string_boolean_run3dmod_menu_options(
                imod_manager::FULL_VOLUME_KEY,
                Some(AxisID::First),
                Some(&format!(
                    "{}{}.matmod",
                    self.get_meta_data().get_dataset_name(),
                    AxisID::First.get_extension()
                )),
                true,
                Some(menu_options),
            )?;
            imod_manager.open_string_axis_id_string_boolean_run3dmod_menu_options(
                imod_manager::FULL_VOLUME_KEY,
                Some(AxisID::Second),
                Some(&format!(
                    "{}{}.matmod",
                    self.get_meta_data().get_dataset_name(),
                    AxisID::Second.get_extension()
                )),
                true,
                Some(menu_options),
            )?;
            Ok(())
        })() {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!(
                        "{}\nCan't open 3dmod on tomograms for matching models",
                        except.0
                    ),
                    "Cannot Open 3dmod",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(AxisID::Only),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodMatchCheck(Run3dmodMenuOptions)` (ApplicationManager.java:7408).
    /// Open the matchcheck results in 3dmod.
    pub fn imod_match_check(&'static self, menu_options: Run3dmodMenuOptions) {
        match self
            .get_imod_manager()
            .open_string_run3dmod_menu_options(imod_manager::MATCH_CHECK_KEY, Some(menu_options))
        {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!(
                        "{}\nCan't open 3dmod on matchcheck.mat or matchcheck.rec",
                        except.0
                    ),
                    "Cannot Open 3dmod",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(AxisID::Only),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodPatchRegionModel(Run3dmodMenuOptions)`
    /// (ApplicationManager.java:7432).  Open the patch region models in tomogram
    /// being matched to.
    pub fn imod_patch_region_model(&'static self, menu_options: Run3dmodMenuOptions) {
        // FIXME do we want to be updating here break model
        // Get the latest combine parameters from the dialog
        if !self.update_combine_params(false, true) {}
        let axis_id;
        let match_mode = self.get_meta_data().get_combine_params().get_match_mode();
        if match_mode.is_none() || match_mode == Some(MatchMode::BToA) {
            axis_id = AxisID::First;
        } else {
            axis_id = AxisID::Second;
        }
        eprintln!("axisID={axis_id}");
        match self
            .get_imod_manager()
            .open_string_axis_id_string_boolean_run3dmod_menu_options(
                imod_manager::FULL_VOLUME_KEY,
                Some(axis_id),
                Some("patch_region.mod"),
                true,
                Some(menu_options),
            ) {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!(
                        "{}\nCan't open 3dmod on tomogram for patch region models",
                        except.0
                    ),
                    "Cannot Open 3dmod",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodModel(FileType, FileType, AxisID, Run3dmodMenuOptions, boolean,
    /// boolean)` (ApplicationManager.java:7471).  Open a model for editing.
    pub fn imod_model(
        &'static self,
        file_type: &FileType,
        model_file_type: &FileType,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
        model_mode: bool,
        use_raw_tilt_file: bool,
    ) {
        // Fixed in translation: a file type with no 3dmod key makes Java throw
        // NullPointerException in ImodManager.getPrivateKey; nothing is opened.
        let Some(imod_manager_key) = file_type.get_imod_manager_key() else {
            return;
        };
        let imod_manager = self.get_imod_manager();
        match (|| -> Result<(), ImodManagerError> {
            if use_raw_tilt_file {
                let tilt_file = dataset_files::get_raw_tilt_file(self, Some(axis_id));
                if tilt_file.exists() {
                    imod_manager.set_tilt_file(
                        imod_manager_key,
                        Some(axis_id),
                        Some(&utilities::java_io_file_get_name(
                            &tilt_file.to_string_lossy(),
                        )),
                    )?;
                } else {
                    imod_manager.reset_tilt_file(imod_manager_key, Some(axis_id))?;
                }
            }
            imod_manager.open_string_axis_id_string_boolean_run3dmod_menu_options(
                imod_manager_key,
                Some(axis_id),
                model_file_type
                    .get_file_name(Some(self), Some(axis_id))
                    .as_deref(),
                model_mode,
                Some(menu_options),
            )?;
            Ok(())
        })() {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.0,
                    &format!(
                        "Can't open 3dmod on {} for {}",
                        file_type
                            .get_file_name(Some(self), Some(axis_id))
                            .unwrap_or_else(|| "null".to_string()),
                        model_file_type
                            .get_file_name(Some(self), Some(axis_id))
                            .unwrap_or_else(|| "null".to_string())
                    ),
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodClusteredElongatedModel(AxisID, Run3dmodMenuOptions)`
    /// (ApplicationManager.java:7508).  Open clustered elongated model.
    pub fn imod_clustered_elongated_model(
        &'static self,
        axis_id: AxisID,
        menu_options: Run3dmodMenuOptions,
    ) {
        let file_type: &FileType = &file_type::CLASS.prealigned_stack;
        let model = format!(
            "{}{}{}",
            file_type::CLASS
                .autofidseed_dir
                .get_file_name(Some(self), Some(axis_id))
                .unwrap_or_else(|| "null".to_string()),
            std::path::MAIN_SEPARATOR,
            utilities::java_io_file_get_name(
                &file_type::CLASS
                    .clustered_elongated_model
                    .get_file(Some(self), Some(axis_id))
                    .map(|file| file.to_string_lossy().into_owned())
                    .unwrap_or_default()
            )
        );
        // Fixed in translation: a file type with no 3dmod key makes Java throw
        // NullPointerException in ImodManager.getPrivateKey; nothing is opened.
        let Some(imod_manager_key) = file_type.get_imod_manager_key() else {
            return;
        };
        let imod_manager = self.get_imod_manager();
        match (|| -> Result<(), ImodManagerError> {
            let tilt_file = dataset_files::get_raw_tilt_file(self, Some(axis_id));
            if tilt_file.exists() {
                imod_manager.set_tilt_file(
                    imod_manager_key,
                    Some(axis_id),
                    Some(&utilities::java_io_file_get_name(
                        &tilt_file.to_string_lossy(),
                    )),
                )?;
            } else {
                imod_manager.reset_tilt_file(imod_manager_key, Some(axis_id))?;
            }
            imod_manager.set_open_surf_cont_point(imod_manager_key, Some(axis_id), true)?;
            imod_manager.set_preserve_contrast(imod_manager_key, Some(axis_id), true)?;
            imod_manager.open_string_axis_id_string_run3dmod_menu_options(
                imod_manager_key,
                Some(axis_id),
                Some(&model),
                Some(menu_options),
            )?;
            Ok(())
        })() {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.0,
                    &format!(
                        "Can't open 3dmod on {} for {}",
                        file_type
                            .get_file_name(Some(self), Some(axis_id))
                            .unwrap_or_else(|| "null".to_string()),
                        model
                    ),
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodPatchVectorModel(String)` (ApplicationManager.java:7546).  Open the
    /// patch vector models in 3dmod.
    pub fn imod_patch_vector_model(&'static self, key: &str) {
        match self.get_imod_manager().open_string(key) {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!(
                        "{}\nCan't open 3dmod on tomogram for patch vector model",
                        except.0
                    ),
                    "Cannot Open 3dmod",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(AxisID::Only),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodMatchedToTomogram(Run3dmodMenuOptions)`
    /// (ApplicationManager.java:7570).  Open the tomogram being matched to.
    pub fn imod_matched_to_tomogram(&'static self, menu_options: Run3dmodMenuOptions) {
        let axis_id;
        let match_mode = self.get_meta_data().get_combine_params().get_match_mode();
        if match_mode.is_none() || match_mode == Some(MatchMode::BToA) {
            axis_id = AxisID::First;
        } else {
            axis_id = AxisID::Second;
        }
        match self
            .get_imod_manager()
            .open_string_axis_id_run3dmod_menu_options(
                imod_manager::FULL_VOLUME_KEY,
                Some(axis_id),
                Some(menu_options),
            ) {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!(
                        "{}\nCan't open 3dmod on tomogram being matched to",
                        except.0
                    ),
                    "Cannot Open 3dmod",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `doneTomogramCombinationDialog()` (ApplicationManager.java:7602).
    /// Tomogram combination done method, move on to the post processing window.
    pub fn done_tomogram_combination_dialog(&'static self) {
        if self.can_combine(true) {
            let dialog = self.tomogram_combination_dialog.get();
            self.save_tomogram_combination_dialog(dialog.as_ref());
        } else {
            // Java dereferences the field without a null check.
            if let Some(dialog) = self.tomogram_combination_dialog.get() {
                dialog.remove_listeners();
            }
            self.tomogram_combination_dialog.set(None);
        }
    }

    /// Java private `canCombine(boolean)` (ApplicationManager.java:7612).
    fn can_combine(&'static self, silent: bool) -> bool {
        if !file_type::CLASS
            .tilt_output
            .exists(Some(self), Some(AxisID::First))
            || !file_type::CLASS
                .tilt_output
                .exists(Some(self), Some(AxisID::Second))
        {
            let message = "Cannot combine.  One or more tomograms is missing.  Tomogram Generation should be run.";
            if !silent {
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    message,
                    "Unable to Combine",
                    None,
                );
            } else {
                eprintln!("Warning: {message}");
            }
            return false;
        }
        true
    }

    /// Java `saveTomogramCombinationDialog(TomogramCombinationDialog)`
    /// (ApplicationManager.java:7631).  Tomogram combination done method, move on to
    /// the post processing window.
    pub fn save_tomogram_combination_dialog(
        &'static self,
        tomogram_combination_dialog: Option<&Rc<TomogramCombinationDialog>>,
    ) {
        // if the dialog isn't displayed, it wouldn't have been changed since it was
        // last saved
        let Some(tomogram_combination_dialog) = tomogram_combination_dialog else {
            return;
        };
        if !tomogram_combination_dialog.is_displayed() {
            return;
        }
        self.set_advanced_dialog_type_boolean(
            tomogram_combination_dialog.get_dialog_type(),
            tomogram_combination_dialog.is_advanced(),
        );
        let exit_state = tomogram_combination_dialog.get_exit_state();
        if exit_state == DialogExitState::Cancel {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(AxisID::Only);
            }
        } else {
            tomogram_combination_dialog.synchronize_from_current_tab();
            // Update the com script and metadata info from the tomogram
            // combination dialog box. Since there are multiple pages and scripts
            // associated with the postpone button get the ones that are appropriate
            self.update_combine_params(false, true);
            tomogram_combination_dialog.get_parameters_meta_data(self.get_meta_data());
            tomogram_combination_dialog
                .get_parameters_recon_screen_state(self.get_screen_state(AxisID::Only));
            if !tomogram_combination_dialog.is_changed(self.get_state()) {
                self.update_solvematch_com(false);
                self.update_dualvolmatch_com(false);
                self.update_matchvol1_com(false);
                self.update_patchcorr_com(false);
                self.update_matchorwarp_com(false, false);
                self.update_volcombine_com(false);
            }
            if exit_state == DialogExitState::Postpone {
                self.get_recon_process_track()
                    .set_tomogram_combination_state(ProcessState::InProgress);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_tomogram_combination_state(ProcessState::InProgress);
                    main_panel.show_blank_process(AxisID::Only);
                }
            } else if exit_state != DialogExitState::Save {
                self.get_recon_process_track()
                    .set_tomogram_combination_state(ProcessState::Complete);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_tomogram_combination_state(ProcessState::Complete);
                }
                self.close_imod(
                    Some(imod_manager::FULL_VOLUME_KEY),
                    Some(AxisID::First),
                    Some("axis A full volume"),
                    false,
                );
                self.close_imod_string_axis_id_axis_id_string(
                    Some(imod_manager::FULL_VOLUME_KEY),
                    AxisID::First,
                    AxisID::Second,
                    Some("axis B full volume"),
                );
                self.close_imod_string_axis_id_axis_id_string(
                    Some(imod_manager::MATCH_CHECK_KEY),
                    AxisID::First,
                    AxisID::Second,
                    Some("match check volume"),
                );
                self.close_imod_string_string_boolean(
                    Some(imod_manager::PATCH_VECTOR_MODEL_KEY),
                    Some("patch vector model"),
                    false,
                );
                self.open_post_processing_dialog();
            }
            self.save_storables(Some(AxisID::Only));
        }
    }

    /// Java `backwardsCompatibilityCombineScriptsExist()` (ApplicationManager.java:7685).
    /// Check to see if the combine scripts exist.
    pub fn backwards_compatibility_combine_scripts_exist(&'static self) {
        let combine_scripts_created_value =
            self.get_state().get_combine_scripts_created().get_int();
        if combine_scripts_created_value == etomo_state::FALSE_VALUE {
            return;
        }
        // Check if combineScriptsCreated was not set. Also check if it is true, to
        // make sure that it is still true.
        // `new File(propertyUserDir, name)`: a null parent gives the bare name, as
        // `PathBuf::new().join(name)` does.
        let property_user_dir: PathBuf = self
            .get_property_user_dir()
            .map(PathBuf::from)
            .unwrap_or_default();
        let solvematchshift = property_user_dir.join("solvematchshift.com");
        let solvematchmod = property_user_dir.join("solvematchmod.com");
        let solvematch = property_user_dir.join("solvematch.com");
        let matchvol1 = property_user_dir.join("matchvol1.com");
        let matchorwarp = property_user_dir.join("matchorwarp.com");
        let patchcorr = property_user_dir.join("patchcorr.com");
        let volcombine = property_user_dir.join("volcombine.com");
        let warpvol = property_user_dir.join("warpvol.com");
        if (solvematch.exists() || (solvematchshift.exists() && solvematchmod.exists()))
            && matchvol1.exists()
            && matchorwarp.exists()
            && patchcorr.exists()
            && volcombine.exists()
            && warpvol.exists()
        {
            self.get_state().set_combine_scripts_created(true);
        } else {
            self.get_state().reset_combine_scripts_created();
        }
    }

    /// Java `createCombineScripts(ProcessResultDisplay)` (ApplicationManager.java:7718).
    /// Run the setupcombine script with the current combine parameters stored in
    /// metaData object.  updateCombineCom is called first to get the currect
    /// parameters from the dialog.
    pub fn create_combine_scripts(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
    ) -> bool {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let run_msg = "Creating combine scripts";
        eprintln!("{run_msg}.");
        if self.is_axis_busy(AxisID::Only, process_result_display_ref.clone()) {
            eprintln!("{run_msg} failed.");
            return false;
        }
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(run_msg),
                AxisID::Only,
                Some(&ProcessName::SOLVEMATCH),
            );
        }
        if !self.update_combine_params(true, false) {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    AxisID::Only,
                    Some(ProcessEndState::Failed),
                );
            }
            return false;
        }
        match self
            .get_process_mgr()
            .setup_combine_scripts(process_result_display_ref.clone())
        {
            Ok(false) => {
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
                }
                return false;
            }
            Ok(true) => {
                self.get_recon_process_track()
                    .set_tomogram_combination_state(ProcessState::InProgress);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_tomogram_combination_state(ProcessState::InProgress);
                }
            }
            Err(except) => {
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
                }
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!("Can't run setupcombine\n{except}"),
                    "Setupcombine IOException",
                    Some(AxisID::Only),
                );
                return false;
            }
        }
        // Reload the initial and final match paramaters from the newly created
        // scripts

        // Reload all of the parameters into the ComScriptManager
        self.load_solvematch_void();
        self.load_dualvolmatch();
        self.load_matchvol1();
        self.load_patchcorr();
        self.load_matchorwarp();
        self.load_volcombine();
        self.load_combine_comscript();
        TomogramTool::save_tomogram_size(self, AxisID::First, AxisID::Only);
        TomogramTool::save_tomogram_size(self, AxisID::Second, AxisID::Only);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id_process_end_state(
                AxisID::Only,
                Some(ProcessEndState::Done),
            );
        }
        true
    }

    /// Java `setupCombineComOnly()` (ApplicationManager.java:7763).
    pub fn setup_combine_com_only(&'static self) -> bool {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Combine.com out of date - recreating"),
                AxisID::Only,
                Some(&ProcessName::SETUPCOMBINE),
            );
        }
        match self.get_process_mgr().setup_combine_only_make_combine_com() {
            Ok(true) => {}
            Ok(false) => {
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
                }
                return false;
            }
            Err(except) => {
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
                }
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!(
                        "Can't run setupcombine.  Copy combine.com from $IMOD_DIR/com.\n{except}"
                    ),
                    "Setupcombine IOException",
                    Some(AxisID::Only),
                );
                return false;
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id_process_end_state(
                AxisID::Only,
                Some(ProcessEndState::Done),
            );
        }
        true
    }

    /// Java private `updateCombineParams(boolean, boolean)` (ApplicationManager.java:7792).
    /// Update the combine parameters from the a specified tab of the calling dialog;
    /// assumes that the dialog is synchronized.
    fn update_combine_params(&'static self, do_validation: bool, silent: bool) -> bool {
        if !self.can_combine(silent) {
            return false;
        }
        let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update combine.com without an active tomogram combination dialog",
                "Program logic error",
                Some(AxisID::Only),
            );
            return false;
        };
        {
            // start with existing combine params and update it from the screen
            let mut combine_params = self.get_meta_data().get_combine_params();
            match tomogram_combination_dialog
                .get_combine_params(&mut *combine_params, do_validation)
            {
                Ok(true) => {}
                Ok(false) => return false,
                // catch (NumberFormatException except)
                Err(except) => {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self),
                            &except,
                            "Number format error",
                            Some(AxisID::Only),
                        )
                    });
                    return false;
                }
            }
            let rec_file_name: Option<String>;
            let match_mode = combine_params.get_match_mode();
            if match_mode.is_none() || match_mode == Some(MatchMode::BToA) {
                rec_file_name = file_type::CLASS
                    .tilt_output_dual
                    .get_file_name(Some(self), Some(AxisID::First));
                // recFileName = metaData.getDatasetName() + "a.rec";
            } else {
                rec_file_name = file_type::CLASS
                    .tilt_output_dual
                    .get_file_name(Some(self), Some(AxisID::Second));
                // recFileName = metaData.getDatasetName() + "b.rec";
            }
            let rec_file_name_display = rec_file_name.as_deref().unwrap_or("null");
            // `CombineParams.setMaxPatchZMax(String)` throws InvalidParameterException
            // or IOException; the translation reports both as one message.  The only
            // IOException the method itself throws is "file does not exist"; every
            // other message comes from `MRCHeader.read`, whose two exceptions it cannot
            // tell apart, and takes the InvalidParameterException arm.
            if let Err(except) = combine_params.set_max_patch_z_max_from_file(rec_file_name_display)
            {
                if except == "file does not exist" {
                    ui_harness::open_message_dialog_from_process(
                        Some(self),
                        &except,
                        &format!("IO Error: {rec_file_name_display}"),
                        Some(AxisID::Only),
                    );
                } else {
                    let detailed_message = [
                        "Unable to get max patch Z boundary".to_string(),
                        "Are both tomograms computed and available?".to_string(),
                        String::new(),
                        except,
                    ];
                    // Upstream bug (ApplicationManager.java:7828): Java concatenates the
                    // String[] itself (`detailedMessage + "\n" + ...`), so the dialog
                    // shows the array's identity ("[Ljava.lang.String;@1b6d3586")
                    // instead of the four lines.  Fixed in translation: the lines are
                    // joined with newlines.
                    ui_harness::open_message_dialog_from_process(
                        Some(self),
                        &format!(
                            "{}\nInvalid parameter: {rec_file_name_display}",
                            detailed_message.join("\n")
                        ),
                        "Invalid Parameter",
                        Some(AxisID::Only),
                    );
                }
                // Delete the dialog
                drop(combine_params);
                self.tomogram_combination_dialog.set(None);
                return false;
            }
            if self.get_state().is_combine_scripts_created() && !combine_params.is_valid(true) {
                let invalid_reasons = combine_params.get_invalid_reasons();
                drop(combine_params);
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &invalid_reasons,
                    "Invalid combine parameters",
                    Some(AxisID::Only),
                );
                return false;
            }
        }
        self.save_storables(Some(AxisID::Only));
        true
    }

    /// Java `loadSolvematch()` (ApplicationManager.java:7859).  Load the solvematch com
    /// script into the tomogram combination dialog.
    pub fn load_solvematch_void(&'static self) {
        self.get_com_script_manager().load_solvematch();
        let solvematch = self.get_com_script_manager().get_solvematch();
        // Java dereferences the field without a null check.
        if let Some(dialog) = self.tomogram_combination_dialog.get() {
            dialog.set_solvematch_params(&solvematch);
        }
    }

    /// Java `loadDualvolmatch()` (ApplicationManager.java:7864).
    pub fn load_dualvolmatch(&'static self) {
        self.get_com_script_manager().load_dualvolmatch();
        let dualvolmatch = self.get_com_script_manager().get_dualvolmatch();
        // Java dereferences the field without a null check.
        if let Some(dialog) = self.tomogram_combination_dialog.get() {
            dialog.set_dualvolmatch_params(&dualvolmatch);
        }
    }

    /// Java `loadSolvematch(boolean)` (ApplicationManager.java:7873).  Merge
    /// solvematchshift and solvematchmod com scripts in solvematch and load
    /// solvematch into the tomogram combination dialog.
    pub fn load_solvematch_boolean(&'static self, model_based: bool) {
        // get solvematchshift param
        self.get_com_script_manager().load_solvematchshift();
        let solvematchshift_param = self.get_com_script_manager().get_solvematchshift();
        // get solvematchmod param
        self.get_com_script_manager().load_solvematchmod();
        let solvematchmod_param = self.get_com_script_manager().get_solvematchmod();
        // merge shift and mod into solvematch param
        let mut solvematch_param = SolvematchParam::new(self);
        solvematch_param.merge_solvematchshift(&solvematchshift_param, model_based);
        solvematch_param.merge_solvematchmod(&solvematchmod_param, model_based);
        // update the dialog from the same source
        // Java dereferences the field without a null check.
        if let Some(dialog) = self.tomogram_combination_dialog.get() {
            dialog.set_solvematch_params(&solvematch_param);
        }
        // create the new solvematch com script
        self.get_com_script_manager().load_solvematch();
        self.get_com_script_manager()
            .save_solvematch_solvematch(&solvematch_param);
        // add matchshifts to the solvematch com script
        let matchshifts_param = self
            .get_com_script_manager()
            .get_matchshifts_from_solvematchshifts();
        self.get_com_script_manager()
            .save_solvematch_matchshifts(&matchshifts_param);
    }

    /// Java `updateSolvematchCom(boolean)` (ApplicationManager.java:7900).  Update the
    /// solvematch com file from the tomogramCombinationDialog.
    pub fn update_solvematch_com(&'static self, do_validation: bool) -> bool {
        let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update solvematch.com without an active tomogram generation dialog",
                "Program logic error",
                Some(AxisID::Only),
            );
            return false;
        };
        self.get_com_script_manager().load_solvematch();
        let mut solvematch_param = self.get_com_script_manager().get_solvematch();
        match tomogram_combination_dialog
            .get_solvematch_params(&mut solvematch_param, do_validation)
        {
            Ok(true) => {}
            Ok(false) => return false,
            // catch (final NumberFormatException except)
            Err(except) => {
                let error_message = ["Solvematch Parameter Syntax Error".to_string(), except];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Solvematch Parameter Syntax Error",
                        Some(AxisID::Only),
                    )
                });
                return false;
            }
        }
        self.get_com_script_manager()
            .save_solvematch_solvematch(&solvematch_param);
        true
    }

    /// Java `updateDualvolmatchCom(boolean)` (ApplicationManager.java:7926).
    pub fn update_dualvolmatch_com(&'static self, do_validation: bool) -> bool {
        let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update dualvolmatch.com without an active tomogram generation dialog",
                "Program logic error",
                Some(AxisID::Only),
            );
            return false;
        };
        // Java also catches NumberFormatException here ("Dualvolmatch Parameter Syntax
        // Error", the exception's message); none of the translated callees raise it.
        self.get_com_script_manager().load_dualvolmatch();
        let mut param = self.get_com_script_manager().get_dualvolmatch();
        if !tomogram_combination_dialog
            .get_parameters_dualvolmatch_param_boolean(&mut param, do_validation)
        {
            return false;
        }
        self.get_com_script_manager().save_dualvolmatch(&param);
        true
    }

    /// Java `updateMatchvol1Com(boolean)` (ApplicationManager.java:7957).  Update the
    /// matchvol1 com file from the tomogramCombinationDialog.
    pub fn update_matchvol1_com(&'static self, do_validation: bool) -> bool {
        let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update matchvol1.com without an active tomogram generation dialog",
                "Program logic error",
                Some(AxisID::Only),
            );
            return false;
        };
        // Java also catches NumberFormatException here ("Matchvol1 Parameter Syntax
        // Error", the exception's message); none of the translated callees raise it.
        self.get_com_script_manager().load_matchvol1();
        let mut param = self.get_com_script_manager().get_matchvol_param();
        if !tomogram_combination_dialog
            .get_parameters_matchvol_param_boolean(&mut param, do_validation)
        {
            return false;
        }
        self.get_com_script_manager().save_matchvol(&param);
        true
    }

    // Java ApplicationManager.java:7992-8010 is a commented-out `createNewSolvematch`;
    // nothing to translate.

    /// Java private `loadMatchvol1()` (ApplicationManager.java:8007).  Load the
    /// matchvol1 com script into the tomogram combination dialog.
    fn load_matchvol1(&'static self) {
        self.get_com_script_manager().load_matchvol1();
        let param = self.get_com_script_manager().get_matchvol_param();
        // Java dereferences the field without a null check.
        if let Some(dialog) = self.tomogram_combination_dialog.get() {
            dialog.set_parameters_matchvol_param(&param);
        }
    }

    /// Java private `loadPatchcorr()` (ApplicationManager.java:8015).  Load the
    /// patchcorr com script into the tomogram combination dialog.
    fn load_patchcorr(&'static self) {
        self.get_com_script_manager().load_patchcorr();
        let param = self.get_com_script_manager().get_patchcrawl3_d();
        // Java dereferences the field without a null check.
        if let Some(dialog) = self.tomogram_combination_dialog.get() {
            dialog.set_patchcrawl3_d_params(&param);
        }
    }

    /// Java private `loadVolcombine()` (ApplicationManager.java:8020).
    fn load_volcombine(&'static self) {
        self.get_com_script_manager().load_volcombine();
        // Java dereferences the dialog field without a null check.
        let dialog = self.tomogram_combination_dialog.get();
        // try to load reduction factor
        let mut set_param = self.get_com_script_manager().get_set_param_from_volcombine(
            const_set_param::COMBINEFFT_REDUCTION_FACTOR_NAME,
            Type::Double,
        );
        if let Some(dialog) = &dialog {
            dialog.set_reduction_factor_params(set_param.as_ref());
            dialog.enable_reduction_factor(
                set_param
                    .as_ref()
                    .is_some_and(|set_param| set_param.is_valid()),
            );
        }
        // try to load low from both radius
        set_param = self
            .get_com_script_manager()
            .get_set_param_from_volcombine_previous_command(
                const_set_param::COMBINEFFT_LOW_FROM_BOTH_RADIUS_NAME,
                const_set_param::COMBINEFFT_LOW_FROM_BOTH_RADIUS_TYPE,
                Some(const_set_param::COMMAND_NAME),
            );
        if let Some(dialog) = &dialog {
            dialog.set_low_from_both_radius_params(set_param.as_ref());
            dialog.enable_low_from_both_radius(
                set_param
                    .as_ref()
                    .is_some_and(|set_param| set_param.is_valid()),
            );
        }
    }

    /// Java private `updatePatchcorrCom(boolean)` (ApplicationManager.java:8043).
    /// Update the patchcorr.com script from the information in the tomogram
    /// combination dialog box.
    fn update_patchcorr_com(&'static self, do_validation: bool) -> bool {
        // Set a reference to the correct object
        let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update patchcorr.com without an active tomogram generation dialog",
                "Program logic error",
                Some(AxisID::Only),
            );
            return false;
        };
        let mut patchcrawl3_d_param = self.get_com_script_manager().get_patchcrawl3_d();
        match tomogram_combination_dialog
            .get_patchcrawl3_d_params(&mut patchcrawl3_d_param, do_validation)
        {
            Ok(true) => {}
            Ok(false) => return false,
            // catch (final NumberFormatException except)
            Err(except) => {
                let error_message = ["Patchcorr Parameter Syntax Error".to_string(), except];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message,
                        "Patchcorr Parameter Syntax Error",
                        Some(AxisID::Only),
                    )
                });
                return false;
            }
        }
        self.get_com_script_manager()
            .save_patchcorr(&patchcrawl3_d_param);
        true
    }

    /// Java private `updateVolcombineCom(boolean)` (ApplicationManager.java:8076).
    /// Update the volcombine.com script from the information in the tomogram
    /// combination dialog box.
    fn update_volcombine_com(&'static self, do_validation: bool) -> bool {
        // Set a reference to the correct object
        let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update volcombine.com without an active tomogram generation dialog",
                "Program logic error",
                Some(AxisID::Only),
            );
            return false;
        };
        // Java also catches NumberFormatException here ("Volcombine Parameter Syntax
        // Error", the exception's message); none of the translated callees raise it.
        //
        // Make sure the reduction factor set command is available in
        // volcombine.com
        let mut set_param = self.get_com_script_manager().get_set_param_from_volcombine(
            const_set_param::COMBINEFFT_REDUCTION_FACTOR_NAME,
            Type::Double,
        );
        let mut set_param_is_valid = set_param
            .as_ref()
            .is_some_and(|set_param| set_param.is_valid());
        tomogram_combination_dialog.enable_reduction_factor(set_param_is_valid);
        if !tomogram_combination_dialog
            .get_reduction_factor_param(set_param.as_mut(), do_validation)
        {
            return false;
        }
        if set_param_is_valid && let Some(set_param) = &set_param {
            self.get_com_script_manager().save_volcombine(set_param);
        }
        // Make sure the low from both radius set command is available in
        // volcombine.com
        set_param = self
            .get_com_script_manager()
            .get_set_param_from_volcombine_previous_command(
                const_set_param::COMBINEFFT_LOW_FROM_BOTH_RADIUS_NAME,
                const_set_param::COMBINEFFT_LOW_FROM_BOTH_RADIUS_TYPE,
                Some(const_set_param::COMMAND_NAME),
            );
        set_param_is_valid = set_param
            .as_ref()
            .is_some_and(|set_param| set_param.is_valid());
        tomogram_combination_dialog.enable_low_from_both_radius(set_param_is_valid);
        if !tomogram_combination_dialog
            .get_low_from_both_radius_param(set_param.as_mut(), do_validation)
        {
            return false;
        }
        if set_param_is_valid && let Some(set_param) = &set_param {
            self.get_com_script_manager()
                .save_volcombine_previous_command(set_param, const_set_param::COMMAND_NAME);
        }
        true
    }

    /// Java private `loadMatchorwarp()` (ApplicationManager.java:8125).  Load the
    /// matchorwarp com script into the tomogram combination dialog.
    fn load_matchorwarp(&'static self) {
        self.get_com_script_manager().load_matchorwarp();
        let param = self.get_com_script_manager().get_matchorwar_param();
        // Java dereferences the field without a null check.
        if let Some(dialog) = self.tomogram_combination_dialog.get() {
            dialog.set_matchorwarp_params(&param);
        }
    }

    /// Java private `getCombineComscript()` (ApplicationManager.java:8130).
    fn get_combine_comscript(&'static self) -> Option<CombineComscriptState> {
        let initial_volume_matching = self.get_meta_data().is_initial_volume_matching();
        let mut combine_comscript_state = self
            .get_com_script_manager()
            .get_combine_comscript(initial_volume_matching);
        if combine_comscript_state.is_none() {
            if let Err(e) = ComScriptUtil::use_template(
                self,
                combine_comscript_state::COMSCRIPT_NAME,
                AxisType::DualAxis,
                AxisID::Only,
                true,
            ) {
                eprintln!("{e:?}");
                let message = [
                    "Unable to copy combine com script".to_string(),
                    "Check file and directory permissions".to_string(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Can't create a new comscript",
                    Some(AxisID::Only),
                );
                return None;
            }
            self.get_com_script_manager().load_combine();
            combine_comscript_state = self
                .get_com_script_manager()
                .get_combine_comscript(initial_volume_matching);
        }
        combine_comscript_state
    }

    // Java `getComScriptManager()` (ApplicationManager.java:8159) returns
    // `comScriptMgr`; it already exists in application_manager.rs as
    // `get_com_script_manager`, which returns the locked guard.

    /// Java private `loadCombineComscript()` (ApplicationManager.java:8166).  Load the
    /// combine com script.
    fn load_combine_comscript(&'static self) -> Option<CombineComscriptState> {
        self.get_com_script_manager().load_combine();
        let combine_comscript_state = self.get_combine_comscript()?;
        // Java dereferences the field without a null check.
        if let Some(dialog) = self.tomogram_combination_dialog.get() {
            dialog.set_run_volcombine(combine_comscript_state.is_run_volcombine());
            dialog.update_display();
        }
        Some(combine_comscript_state)
    }

    /// Java private `updateCombineComscriptState(CombineProcessType)`
    /// (ApplicationManager.java:8181).
    fn update_combine_comscript_state(
        &'static self,
        start_command: CombineProcessType,
    ) -> Option<CombineComscriptState> {
        let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update combine.com without an active tomogram generation dialog",
                "Program logic error",
                Some(AxisID::Only),
            );
            return None;
        };
        // set goto a the start of combine
        let mut combine_comscript_state = self.get_combine_comscript()?;
        // set first command to run
        combine_comscript_state
            .set_start_command(start_command.get_index(), &self.get_com_script_manager());
        // set last command to run:
        // Volcombine is the last command to run in the combine script if parallel
        // processing is not used and either the the "stop before running
        // volcombine"
        // checkbox is off or "Restart at volcombine" was pressed.
        if start_command == CombineProcessType::VOLCOMBINE
            || tomogram_combination_dialog.is_run_volcombine()
        {
            // Pop up call close imod dialogs before running process since the
            // processes
            // before volcombine can take awhile.
            combine_comscript_state
                .set_output_image_file_type(Some(Arc::clone(&file_type::CLASS.combined_volume)));
            if tomogram_combination_dialog
                .get_run_processing_method()
                .is_local()
            {
                combine_comscript_state.set_end_command(
                    CombineProcessType::VOLCOMBINE.get_index(),
                    &self.get_com_script_manager(),
                );
            } else {
                combine_comscript_state.reset_output_image_file_type();
                combine_comscript_state.set_end_command(
                    CombineProcessType::get_instance_index(
                        combine_process_type::VOLCOMBINE_INDEX - 1,
                        false,
                    )
                    .expect("CombineProcessType index")
                    .get_index(),
                    &self.get_com_script_manager(),
                );
            }
        } else {
            combine_comscript_state.reset_output_image_file_type();
            combine_comscript_state.set_end_command(
                CombineProcessType::get_instance_index(
                    combine_process_type::VOLCOMBINE_INDEX - 1,
                    false,
                )
                .expect("CombineProcessType index")
                .get_index(),
                &self.get_com_script_manager(),
            );
        }
        Some(combine_comscript_state)
    }

    /// Java private `updateMatchorwarpCom(boolean, boolean)`
    /// (ApplicationManager.java:8233).  Update the matchorwarp.com script from the
    /// information in the tomogram combination dialog box.
    fn update_matchorwarp_com(&'static self, trial_mode: bool, do_validation: bool) -> bool {
        // Set a reference to the correct object
        let Some(tomogram_combination_dialog) = self.tomogram_combination_dialog.get() else {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Can not update matchorwarp.com without an active tomogram generation dialog",
                "Program logic error",
                Some(AxisID::Only),
            );
            return false;
        };
        // Java also catches NumberFormatException here ("Matchorwarp Parameter Syntax
        // Error", the exception's message); none of the translated callees raise it.
        let mut matchorwarp_param = self.get_com_script_manager().get_matchorwar_param();
        if !tomogram_combination_dialog
            .get_matchorwarp_params(&mut matchorwarp_param, do_validation)
        {
            return false;
        }
        matchorwarp_param.set_trial_mode(trial_mode);
        self.get_com_script_manager()
            .save_matchorwarp(&matchorwarp_param);
        true
    }

    /// Java private `checkCPUsIfParallelProcessSet(boolean, boolean)`
    /// (ApplicationManager.java:8262).
    fn check_cpus_if_parallel_process_set(
        &'static self,
        parallel_process: bool,
        no_vol_combine: bool,
    ) -> bool {
        let main_panel = self.main_panel.get();
        if parallel_process && !no_vol_combine {
            // Java dereferences `mainPanel` without a null check.
            if let Some(main_panel) = &main_panel {
                match main_panel.get_cpus_selected_int(AxisID::Only, false) {
                    Ok(cpus) if cpus <= 0 => {
                        ui_harness::open_message_dialog_from_process(
                            Some(self),
                            "No cores selected",
                            "Parallel Processing Failed",
                            Some(AxisID::Only),
                        );
                        main_panel.stop_progress_bar_axis_id_process_end_state(
                            AxisID::Only,
                            Some(ProcessEndState::Failed),
                        );
                        return false;
                    }
                    Ok(_) => {}
                    Err(_) => {
                        // `catch (final FieldValidationFailedException e)`
                        main_panel.stop_progress_bar_axis_id_process_end_state(
                            AxisID::Only,
                            Some(ProcessEndState::Failed),
                        );
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Java `combine(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, ProcessingMethod, boolean, boolean, boolean)`
    /// (ApplicationManager.java:8283).  Initiate the combine process from the
    /// beginning.
    #[allow(clippy::too_many_arguments)]
    pub fn combine(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        volcombine_processing_method: ProcessingMethod,
        initial_volume_matching: bool,
        parallel_process: bool,
        no_vol_combine: bool,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let process_series = process_series.unwrap_or_else(|| {
            ProcessSeries::new(self, AxisID::First, Some(dialog_type), Some("combine"))
        });
        if !self.check_cpus_if_parallel_process_set(parallel_process, no_vol_combine) {
            return;
        }
        // FIXME: what are the necessary updates
        // Update the scripts from the dialog panel
        if !self.update_combine_params(true, false) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        let Some(combine_comscript_state) =
            self.update_combine_comscript_state(if initial_volume_matching {
                CombineProcessType::DUALVOLMATCH
            } else {
                CombineProcessType::SOLVEMATCH
            })
        else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        // Don't fail on a failed com file that won't be run.
        if !self.update_solvematch_com(true) && !initial_volume_matching {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_dualvolmatch_com(true) && initial_volume_matching {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_matchvol1_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_patchcorr_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_matchorwarp_com(false, true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_volcombine_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }

        self.get_recon_process_track()
            .set_tomogram_combination_state(ProcessState::InProgress);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_tomogram_combination_state(ProcessState::InProgress);
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        let process_series_ref: ProcessSeriesRef =
            Arc::new(EdtRef::new(Rc::clone(&process_series)));
        let thread_name = match self.get_process_mgr().combine(
            combine_comscript_state,
            process_result_display_ref.clone(),
            Some(process_series_ref),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = ["Can not execute combine.com".to_string(), e.0.clone()];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(AxisID::Only),
                );
                return;
            }
        };
        // Set the next process to execute when this is finished
        if !volcombine_processing_method.is_local()
            && self
                .tomogram_combination_dialog
                .get()
                .is_some_and(|dialog| dialog.is_run_volcombine())
        {
            process_series.borrow_mut().set_next_process(
                Some(splitcombine_param::COMMAND_NAME),
                Some(volcombine_processing_method),
            );
        }
        self.set_background_thread_name(
            Some(&thread_name),
            AxisID::First,
            Some(combine_comscript_state::COMSCRIPT_NAME),
        );
    }

    /// Java `showPane(String, CombineProcessType)` (ApplicationManager.java:8370).
    /// Called by the combine monitor on a process thread.
    pub fn show_pane(&'static self, comscript: &str, combine_process_type: CombineProcessType) {
        if comscript == combine_comscript_state::COMSCRIPT_NAME {
            // Posted: Java calls this on the process thread.
            event_queue::invoke_later(move || {
                // Java dereferences the field without a null check.
                if let Some(dialog) = self.tomogram_combination_dialog.get() {
                    dialog.show_pane(combine_process_type);
                }
            });
        }
    }

    /// Java `matchvol1Combine(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, ProcessingMethod, boolean, boolean)`
    /// (ApplicationManager.java:8380).  Execute the matchvol1 com script and put
    /// patchcorr in the execution queue.
    #[allow(clippy::too_many_arguments)]
    pub fn matchvol1_combine(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        volcombine_processing_method: ProcessingMethod,
        parallel_process: bool,
        no_vol_combine: bool,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = process_series.unwrap_or_else(|| {
            ProcessSeries::new(
                self,
                AxisID::Only,
                Some(dialog_type),
                Some("matchvol1Combine"),
            )
        });
        if !self.check_cpus_if_parallel_process_set(parallel_process, no_vol_combine) {
            return;
        }
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // FIXME: what are the necessary updates
        // Update the scripts from the dialog panel
        if !self.update_combine_params(true, false) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_matchvol1_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_patchcorr_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_matchorwarp_com(false, true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_volcombine_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_combine_params(true, false) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        let Some(combine_comscript_state) =
            self.update_combine_comscript_state(CombineProcessType::MATCHVOL1)
        else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        self.get_recon_process_track()
            .set_tomogram_combination_state(ProcessState::InProgress);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_tomogram_combination_state(ProcessState::InProgress);
        }
        // Check to see if solve.xf exists first
        let solve_xf: PathBuf = self
            .get_property_user_dir()
            .map(PathBuf::from)
            .unwrap_or_default()
            .join("solve.xf");
        if !solve_xf.exists() {
            // nextProcess = "";
            let message = [
                "Can not execute combine.com".to_string(),
                "solve.xf must exist in the working".to_string(),
            ];
            ui_harness::open_message_dialog_array_from_process(
                Some(self),
                &message,
                "Unable to execute com script",
                Some(AxisID::Only),
            );
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        let process_series_ref: ProcessSeriesRef =
            Arc::new(EdtRef::new(Rc::clone(&process_series)));
        let thread_name = match self.get_process_mgr().combine(
            combine_comscript_state,
            process_result_display_ref.clone(),
            Some(process_series_ref),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = ["Can not execute combine.com".to_string(), e.0.clone()];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(AxisID::Only),
                );
                return;
            }
        };
        // Set the next process to execute when this is finished
        if !volcombine_processing_method.is_local()
            && self
                .tomogram_combination_dialog
                .get()
                .is_some_and(|dialog| dialog.is_run_volcombine())
        {
            process_series.borrow_mut().set_next_process(
                Some(splitcombine_param::COMMAND_NAME),
                Some(volcombine_processing_method),
            );
        }
        self.set_background_thread_name(
            Some(&thread_name),
            AxisID::First,
            Some(combine_comscript_state::COMSCRIPT_NAME),
        );
    }

    /// Java `patchcorrCombine(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, ProcessingMethod, boolean, boolean)`
    /// (ApplicationManager.java:8475).  Initiate the combine process from patchcorr
    /// step.
    #[allow(clippy::too_many_arguments)]
    pub fn patchcorr_combine(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        volcombine_processing_method: ProcessingMethod,
        parallel_process: bool,
        no_vol_combine: bool,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = process_series.unwrap_or_else(|| {
            ProcessSeries::new(
                self,
                AxisID::First,
                Some(dialog_type),
                Some("patchcorrCombine"),
            )
        });
        // Check for CPU's selected if parallel processing is enabled
        if !self.check_cpus_if_parallel_process_set(parallel_process, no_vol_combine) {
            return;
        }

        self.send_msg_process_starting(process_result_display_ref.as_ref());
        if !self.update_combine_params(true, false) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        let Some(combine_comscript_state) =
            self.update_combine_comscript_state(CombineProcessType::PATCHCORR)
        else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        if !self.update_patchcorr_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_matchorwarp_com(false, true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_volcombine_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }

        self.get_recon_process_track()
            .set_tomogram_combination_state(ProcessState::InProgress);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_tomogram_combination_state(ProcessState::InProgress);
        }

        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        let process_series_ref: ProcessSeriesRef =
            Arc::new(EdtRef::new(Rc::clone(&process_series)));
        let thread_name = match self.get_process_mgr().combine(
            combine_comscript_state,
            process_result_display_ref.clone(),
            Some(process_series_ref),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = ["Can not execute patchcorr.com".to_string(), e.0.clone()];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(AxisID::Only),
                );
                return;
            }
        };
        // Set the next process to execute when this is finished
        if !volcombine_processing_method.is_local()
            && self
                .tomogram_combination_dialog
                .get()
                .is_some_and(|dialog| dialog.is_run_volcombine())
        {
            process_series.borrow_mut().set_next_process(
                Some(splitcombine_param::COMMAND_NAME),
                Some(volcombine_processing_method),
            );
        }
        self.set_background_thread_name(
            Some(&thread_name),
            AxisID::First,
            Some(combine_comscript_state::COMSCRIPT_NAME),
        );
    }

    /// Java `matchorwarpCombine(ProcessResultDisplay, ProcessSeries,
    /// Deferred3dmodButton, Run3dmodMenuOptions, DialogType, ProcessingMethod, boolean,
    /// boolean)` (ApplicationManager.java:8549).  Initiate the combine process from
    /// matchorwarp step.
    #[allow(clippy::too_many_arguments)]
    pub fn matchorwarp_combine(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        volcombine_processing_method: ProcessingMethod,
        parallel_process: bool,
        no_vol_combine: bool,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = process_series.unwrap_or_else(|| {
            ProcessSeries::new(
                self,
                AxisID::First,
                Some(dialog_type),
                Some("matchorwarpCombine"),
            )
        });
        // Check for CPU's selected if parallel processing is enabled
        if !self.check_cpus_if_parallel_process_set(parallel_process, no_vol_combine) {
            return;
        }

        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let Some(combine_comscript_state) =
            self.update_combine_comscript_state(CombineProcessType::MATCHORWARP)
        else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        if !self.update_matchorwarp_com(false, true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_volcombine_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        self.get_recon_process_track()
            .set_tomogram_combination_state(ProcessState::InProgress);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_tomogram_combination_state(ProcessState::InProgress);
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        let process_series_ref: ProcessSeriesRef =
            Arc::new(EdtRef::new(Rc::clone(&process_series)));
        let thread_name = match self.get_process_mgr().combine(
            combine_comscript_state,
            process_result_display_ref.clone(),
            Some(process_series_ref),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = ["Can not execute combine.com".to_string(), e.0.clone()];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(AxisID::Only),
                );
                return;
            }
        };
        // Set the next process to execute when this is finished
        if !volcombine_processing_method.is_local()
            && self
                .tomogram_combination_dialog
                .get()
                .is_some_and(|dialog| dialog.is_run_volcombine())
        {
            process_series.borrow_mut().set_next_process(
                Some(splitcombine_param::COMMAND_NAME),
                Some(volcombine_processing_method),
            );
        }
        self.set_background_thread_name(
            Some(&thread_name),
            AxisID::First,
            Some(combine_comscript_state::COMSCRIPT_NAME),
        );
    }

    /// Java `matchorwarpTrial(ProcessSeries)` (ApplicationManager.java:8611).
    /// Initiate the combine process from matchorwarp step.
    pub fn matchorwarp_trial(&'static self, process_series: Option<ProcessSeriesHandle>) {
        if self.update_matchorwarp_com(true, true) {
            self.get_recon_process_track()
                .set_tomogram_combination_state(ProcessState::InProgress);
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.set_tomogram_combination_state(ProcessState::InProgress);
            }
            // Set the next process to execute when this is finished
            // nextProcess = next;
            let process_series_ref: Option<ProcessSeriesRef> =
                process_series.map(|process_series| Arc::new(EdtRef::new(process_series)));
            let thread_name = match self.get_process_mgr().matchorwarp(process_series_ref) {
                Ok(thread_name) => thread_name,
                Err(e) => {
                    eprintln!("{e:?}");
                    let message = ["Can not execute matchorwarp.com".to_string(), e.0.clone()];
                    ui_harness::open_message_dialog_array_from_process(
                        Some(self),
                        &message,
                        "Unable to execute com script",
                        Some(AxisID::Only),
                    );
                    return;
                }
            };
            self.set_thread_name(Some(&thread_name), Some(AxisID::First));
            // FIXME why show the final pane when the button that calls this function
            // on the final pane?
            // Java dereferences the field without a null check.
            if let Some(dialog) = self.tomogram_combination_dialog.get() {
                dialog.show_pane(CombineProcessType::MATCHORWARP);
            }
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.start_progress_bar_string_axis_id_process_name(
                    Some("matchorwarp"),
                    AxisID::First,
                    Some(&ProcessName::MATCHORWARP),
                );
            }
        }
    }

    /// Java `volcombine(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType)` (ApplicationManager.java:8641).  Execute the
    /// combine script starting at volcombine.
    pub fn volcombine(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = process_series.unwrap_or_else(|| {
            ProcessSeries::new(self, AxisID::First, Some(dialog_type), Some("volcombine"))
        });
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let combine_comscript_state =
            self.update_combine_comscript_state(CombineProcessType::VOLCOMBINE);
        if !self.update_combine_params(true, false) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        if !self.update_volcombine_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        self.get_recon_process_track()
            .set_tomogram_combination_state(ProcessState::InProgress);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_tomogram_combination_state(ProcessState::InProgress);
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        // Upstream bug (ApplicationManager.java:8648-8668): the result of
        // `updateCombineComscriptState` is never checked, so a null state (no dialog, or
        // combine.com could not be read or copied) reaches `processMgr.combine` and
        // throws a NullPointerException after the process track and main panel were
        // already marked in progress.  Fixed in translation: the start is reported as
        // failed and the series ended, as every sibling combine method does for a null
        // state.
        let Some(combine_comscript_state) = combine_comscript_state else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        let process_series_ref: ProcessSeriesRef =
            Arc::new(EdtRef::new(Rc::clone(&process_series)));
        let thread_name = match self.get_process_mgr().combine(
            combine_comscript_state,
            process_result_display_ref.clone(),
            Some(process_series_ref),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = ["Can not execute combine.com".to_string(), e.0.clone()];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute com script",
                    Some(AxisID::Only),
                );
                return;
            }
        };
        self.set_background_thread_name(
            Some(&thread_name),
            AxisID::First,
            Some(combine_comscript_state::COMSCRIPT_NAME),
        );
    }

    /// Java `modelToPatch()` (ApplicationManager.java:8685).  Convert the patch.mod to
    /// patch.out.
    pub fn model_to_patch(&'static self) {
        if let Some(main_panel) = self.main_panel.get() {
            main_panel
                .start_progress_bar_string_axis_id(Some("Replacing patch vectors"), AxisID::Only);
        }
        match self.get_process_mgr().model_to_patch(AxisID::Only) {
            Ok(()) => {}
            Err(RunCommandError::LogFile(LogFileError::Lock(_))) => {
                // `catch (final LockException except)`
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
                }
                return;
            }
            Err(except) => {
                // `catch (final SystemProcessException | LogFileException | IOException)`
                eprintln!("{except:?}");
                let error_message = [
                    "Unable to convert patch_vector.mod to patch.out".to_string(),
                    except.to_string(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &error_message,
                    "Patch vector model error",
                    Some(AxisID::Only),
                );
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        AxisID::Only,
                        Some(ProcessEndState::Failed),
                    );
                }
                return;
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.stop_progress_bar_axis_id_process_end_state(
                AxisID::Only,
                Some(ProcessEndState::Done),
            );
        }
    }

    /// Java `openPostProcessingDialog()` (ApplicationManager.java:8710).  Open the post
    /// processing dialog.
    pub fn open_post_processing_dialog(&'static self) {
        // Check to see if the com files are present otherwise pop up a dialog
        // box informing the user to run the setup process
        if !UIExpertUtilities::INSTANCE.are_scripts_created(
            self,
            self.get_meta_data(),
            AxisID::Only,
        ) {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(AxisID::Only);
            }
            return;
        }
        // Open the dialog in the appropriate mode for the current state of
        // processing
        let action_message =
            self.set_current_dialog_type(Some(DialogType::PostProcessing), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.select_button(AxisID::Only, &DialogType::PostProcessing.to_string());
        }
        let mut dialog_exists = true;
        let create_dialog = !self.post_processing_dialog.is_some();
        let mut trimvol_input_file_state: Option<TrimvolInputFileState> = None;
        if create_dialog {
            // Set the appropriate input and output files
            // Set the initial values from TrimvolParam only when postProcessingDialog in new.
            let input_file_name = TrimvolParam::get_input_file_name_for(
                self,
                self.get_meta_data().get_axis_type(),
                Some(&self.get_meta_data().get_name()),
            );
            match TrimvolInputFileState::get_post_processing_instance(
                self,
                AxisID::Only,
                input_file_name.as_deref(),
                self.get_state(),
            ) {
                Ok(state) => trimvol_input_file_state = Some(state),
                Err(except) => {
                    // `catch (final InvalidParameterException except)`.  The translation
                    // reports `MRCHeader.read`'s InvalidParameterException and
                    // IOException as one message, so both take this arm.  Java's
                    // IOException arm (ApplicationManager.java:8749-8757: message dialog
                    // titled "IO Error: <input file>", then
                    // `postProcessingDialog.buttonCancelAction(null)` on the dialog that
                    // is null here - an upstream NullPointerException) cannot be told
                    // apart.
                    let detailed_message = [
                        "Unable to set trimvol range".to_string(),
                        "Does the reconstruction file exist yet?".to_string(),
                        String::new(),
                        except,
                    ];
                    // Upstream bug (ApplicationManager.java:8741): Java concatenates the
                    // String[] itself, so the dialog shows the array's identity.  Fixed
                    // in translation: the lines are joined with newlines.
                    ui_harness::open_message_dialog_from_process(
                        Some(self),
                        &format!(
                            "{}\nInvalid parameter: {}",
                            detailed_message.join("\n"),
                            TrimvolParam::get_input_file_name_for(
                                self,
                                self.get_meta_data().get_axis_type(),
                                Some(&self.get_meta_data().get_name()),
                            )
                            .as_deref()
                            .unwrap_or("null")
                        ),
                        "Invalid Parameter",
                        Some(AxisID::Only),
                    );
                    // Delete the dialog
                    self.post_processing_dialog.set(None);
                    if let Some(main_panel) = self.main_panel.get() {
                        main_panel.show_blank_process(AxisID::Only);
                    }
                    return;
                }
            }
            utilities::timestamp_process_container_status(
                Some("new"),
                Some("PostProcessingDialog"),
                Some(utilities::STARTED_STATUS),
            );
            self.post_processing_dialog
                .set(Some(PostProcessingDialog::get_instance(
                    self,
                    trimvol_input_file_state
                        .as_ref()
                        .is_some_and(|state| state.is_input_file_missing()),
                )));
            utilities::timestamp_process_container_status(
                Some("new"),
                Some("PostProcessingDialog"),
                Some(utilities::FINISHED_STATUS),
            );
        }
        // The dialog exists from here on (created above, or already open).
        let Some(post_processing_dialog) = self.post_processing_dialog.get() else {
            return;
        };

        let mut is_com_file_already_exists = true;
        if !self
            .get_com_script_manager()
            .load_reduce_filt_vol(AxisID::Only, false)
        {
            is_com_file_already_exists = false;
            let squeezevol_param = self.get_meta_data().get_squeezevol_param();
            if let Some(squeezevol_param) = squeezevol_param.as_ref() {
                post_processing_dialog.set_parameters_const_squeezevol_param(squeezevol_param);
            }
        }
        // SubtomoSetup
        if self.is_dual_axis() || self.get_view_type() == ViewType::Montage {
            post_processing_dialog
                .set_parameters_recon_screen_state(self.get_screen_state(AxisID::Only));
            post_processing_dialog
                .set_parameters_const_meta_data_boolean(self.get_meta_data(), dialog_exists);
        } else {
            if !self
                .get_com_script_manager()
                .load_subtomo_setup(AxisID::Only, false)
            {
                let file = file_type::CLASS
                    .subtomo_setup_comscript
                    .get_file(Some(self), Some(AxisID::Only))
                    .unwrap_or_default();
                // `File.getAbsolutePath()`.
                let absolute_path = std::path::absolute(&file).unwrap_or(file);
                BaseProcessManager::touch(&absolute_path.to_string_lossy(), Some(self));
                self.get_com_script_manager()
                    .load_subtomo_setup(AxisID::Only, true);
            }
            let subtomo_setup_param = self
                .get_com_script_manager()
                .get_subtomo_setup_param(AxisID::Only);

            post_processing_dialog
                .set_parameters_recon_screen_state(self.get_screen_state(AxisID::Only));
            post_processing_dialog
                .set_parameters_const_meta_data_boolean(self.get_meta_data(), dialog_exists);
            post_processing_dialog.set_parameters_subtomo_setup_param(&subtomo_setup_param);
        }
        // `createDialog && postProcessingDialog != null`; the input file state is set
        // whenever the dialog was created.
        if create_dialog && let Some(trimvol_input_file_state) = &trimvol_input_file_state {
            let restore_to_defaults =
                post_processing_dialog.set_startup_warnings(trimvol_input_file_state);
            // Set the dialog from TrimvolParam defaults.
            dialog_exists = self.get_meta_data().is_post_exists();
            if !dialog_exists || restore_to_defaults {
                let mut trimvol_param =
                    TrimvolParam::new(self, Some(trimvol_param::Mode::PostProcessing));
                trimvol_param.set_default_range(trimvol_input_file_state, dialog_exists);
                post_processing_dialog.init_parameters(&mut trimvol_param);
            }
            if !trimvol_input_file_state.is_input_file_missing() {
                self.get_meta_data().set_post_exists(true);
            }
            self.save_storables(Some(AxisID::Only));
        }
        self.get_com_script_manager().load_flatten(AxisID::Only);
        // Set from flatten.com after meta data. Flatten.com is not created by
        // copytomocoms and may be empty.
        if self
            .get_com_script_manager()
            .is_warp_vol_param_in_flatten(AxisID::Only)
        {
            let warp_vol_param = self
                .get_com_script_manager()
                .get_warp_vol_param_from_flatten(AxisID::Only);
            post_processing_dialog.set_parameters_const_warp_vol_param(&warp_vol_param);
        }
        // To avoid disturbing the special functionality that decides whether the trimvol
        // output is valid or out of date, add the batchruntomo values as they appear after
        // the
        // dialog is set up.
        let trimvol_display = post_processing_dialog.get_trimvol_display();
        self.get_meta_data()
            .move_post_trimvol_batchruntomo_settings(Some(&*trimvol_display));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_process(&post_processing_dialog.get_container(), AxisID::Only);
        }
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
        // AltStack dialog
        let tilt_param: TiltParam;
        if self.is_dual_axis() {
            self.get_com_script_manager().load_tilt(AxisID::First);
            tilt_param = self.get_com_script_manager().get_tilt_param(AxisID::First);
        } else {
            self.get_com_script_manager().load_tilt(AxisID::Only);
            tilt_param = self.get_com_script_manager().get_tilt_param(AxisID::Only);
        }
        post_processing_dialog.set_parameters_tilt_param_boolean(&tilt_param, !dialog_exists);
        crate::imod::etomo::ui::swing::process_interface::ProcessInterface::set_method(
            &*post_processing_dialog,
            crate::imod::etomo::ui::swing::process_interface::ProcessInterface::get_processing_method(
                &*post_processing_dialog,
            ),
        );

        // ReduceFiltVol
        if !is_com_file_already_exists {
            let mut makecom_file_param = MakecomfileParam::new(
                self,
                AxisID::Only,
                Arc::clone(&file_type::CLASS.reduce_filt_vol_comscript),
            );
            if !post_processing_dialog
                .get_parameters_makecomfile_param_boolean(&mut makecom_file_param, true)
            {
                eprintln!("Invalid field for reduce/filter volume");
            } else {
                self.makecomfile(AxisID::Only, &mut makecom_file_param);
                self.get_com_script_manager()
                    .load_reduce_filt_vol(AxisID::Only, true);
            }
        }
        let reduce_filt_vol_param = self
            .get_com_script_manager()
            .get_reduce_filt_vol_param(AxisID::Only);
        post_processing_dialog.set_parameters_reduce_filt_vol_param_boolean_boolean(
            &reduce_filt_vol_param,
            !dialog_exists,
            is_com_file_already_exists,
        );
    }

    /// Java `openCleanUpDialog()` (ApplicationManager.java:8856).  Open the clean up
    /// dialog.
    pub fn open_clean_up_dialog(&'static self) {
        // Check to see if the com files are present otherwise pop up a dialog
        // box informing the user to run the setup process
        if !UIExpertUtilities::INSTANCE.are_scripts_created(
            self,
            self.get_meta_data(),
            AxisID::Only,
        ) {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(AxisID::Only);
            }
            return;
        }
        // Open the dialog in the appropriate mode for the current state of
        // processing
        let action_message =
            self.set_current_dialog_type(Some(DialogType::CleanUp), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.select_button(AxisID::Only, "Clean Up");
        }
        if !self.clean_up_dialog.is_some() {
            utilities::timestamp_process_container_status(
                Some("new"),
                Some("CleanUpDialog"),
                Some(utilities::STARTED_STATUS),
            );
            self.clean_up_dialog.set(Some(CleanUpDialog::new(self)));
            utilities::timestamp_process_container_status(
                Some("new"),
                Some("CleanUpDialog"),
                Some(utilities::FINISHED_STATUS),
            );
        }
        self.update_archive_display();
        if let (Some(main_panel), Some(clean_up_dialog)) =
            (self.main_panel.get(), self.clean_up_dialog.get())
        {
            main_panel.show_process(&clean_up_dialog.get_container(), AxisID::Only);
        }
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
    }

    /// Java `donePostProcessing()` (ApplicationManager.java:8882).  Close the post
    /// processing dialog panel.
    pub fn done_post_processing(&'static self) {
        // Java passes the field without a null check and `savePostProcessing`
        // dereferences it.
        if let Some(post_processing_dialog) = self.post_processing_dialog.get() {
            self.save_post_processing(&post_processing_dialog);
        }
        self.post_processing_dialog.set(None);
    }

    /// Java `savePostProcessing(PostProcessingDialog)` (ApplicationManager.java:8890).
    /// Close the post processing dialog panel.
    pub fn save_post_processing(&'static self, post_processing_dialog: &Rc<PostProcessingDialog>) {
        self.set_advanced_dialog_type_boolean(
            post_processing_dialog.get_dialog_type(),
            post_processing_dialog.is_advanced(),
        );
        let exit_state = post_processing_dialog.get_exit_state();
        if exit_state == DialogExitState::Cancel {
            if let Some(main_panel) = self.main_panel.get() {
                main_panel.show_blank_process(AxisID::Only);
            }
        } else {
            let flatten_warp_display = post_processing_dialog.get_flatten_warp_display();
            self.update_flatten_warp_param(&*flatten_warp_display, AxisID::Only, false);
            let reduce_filt_vol_display = post_processing_dialog.get_reduce_filt_vol_display();
            self.update_reduce_filt_vol_param(&*reduce_filt_vol_display, AxisID::Only, false);
            if self.is_dual_axis() || self.get_view_type() == ViewType::Montage {
                // do nothing
            } else {
                let subtomo_setup_display = post_processing_dialog.get_subtomo_setup_display();
                self.update_subtomo_setup_com(
                    subtomo_setup_display.as_deref(),
                    AxisID::Only,
                    false,
                );
            }
            let alt_stack_display = post_processing_dialog.get_alt_stack_display();
            let alt_stack_axis_id = alt_stack_display.get_axis_id();
            match alt_stack_axis_id {
                None => {
                    self.update_tilt_com_alt_stack_display_axis_id(
                        &*alt_stack_display,
                        AxisID::First,
                    );
                    self.get_com_script_manager().load_tilt(AxisID::Second);
                    self.update_tilt_com_alt_stack_display_axis_id(
                        &*alt_stack_display,
                        AxisID::Second,
                    );
                }
                Some(alt_stack_axis_id) => {
                    self.update_tilt_com_alt_stack_display_axis_id(
                        &*alt_stack_display,
                        alt_stack_axis_id,
                    );
                }
            }
            post_processing_dialog.get_parameters_meta_data(self.get_meta_data());
            post_processing_dialog.get_parameters_for_trimvol(self.get_meta_data());
            if exit_state == DialogExitState::Postpone {
                self.get_recon_process_track()
                    .set_post_processing_state(ProcessState::InProgress);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_post_processing_state(ProcessState::InProgress);
                    main_panel.show_blank_process(AxisID::Only);
                }
            } else if exit_state != DialogExitState::Save {
                self.get_recon_process_track()
                    .set_post_processing_state(ProcessState::Complete);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_post_processing_state(ProcessState::Complete);
                }
                self.close_imod_string_string_boolean(
                    Some(imod_manager::COMBINED_TOMOGRAM_KEY),
                    Some("full tomogram"),
                    false,
                );
                self.close_imod_string_string_boolean(
                    Some(imod_manager::TRIMMED_VOLUME_KEY),
                    Some("trimmed volume"),
                    false,
                );
                self.close_imod_string_string_boolean(
                    Some(imod_manager::FLAT_VOLUME_KEY),
                    Some("flattened volume"),
                    false,
                );
                self.close_imod_string_string_boolean(
                    Some(imod_manager::SQUEEZED_VOLUME_KEY),
                    Some("squeezed volume"),
                    false,
                );
                self.close_imod_string_string_boolean(
                    Some(imod_manager::SUBTOMO_SETUP_KEY),
                    Some("subtomograms"),
                    false,
                );
                self.close_imod_string_string_boolean(
                    Some(imod_manager::ALT_TOMO_SETUP_TOMOGRAM_KEY),
                    Some("altstack tomograms"),
                    false,
                );
                self.close_imod_string_string_boolean(
                    Some(imod_manager::ALT_TOMO_SETUP_EVEN_ODD_TOMOGRAM_KEY),
                    Some("altstack even/odd tomograms"),
                    false,
                );
                self.close_imod_string_string_boolean(
                    Some(imod_manager::ALT_TOMO_SETUP_EVEN_ODD_FULL_TOMOGRAM_KEY),
                    Some("altstack even/odd full tomograms"),
                    false,
                );
                self.open_clean_up_dialog();
            }
            self.save_storables(Some(AxisID::Only));
        }
    }

    /// Java `doneCleanUp()` (ApplicationManager.java:8948).  Close the clean up dialog
    /// panel.
    pub fn done_clean_up(&'static self) {
        // Java passes the field without a null check and `saveCleanUp` dereferences
        // it.
        if let Some(clean_up_dialog) = self.clean_up_dialog.get() {
            self.save_clean_up(&clean_up_dialog);
        }
        self.clean_up_dialog.set(None);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.show_blank_process(AxisID::Only);
        }
    }

    /// Java `saveCleanUp(CleanUpDialog)` (ApplicationManager.java:8957).  Close the
    /// clean up dialog panel.
    pub fn save_clean_up(&'static self, clean_up_dialog: &Rc<CleanUpDialog>) {
        self.set_advanced_dialog_type_boolean(
            clean_up_dialog.get_dialog_type(),
            clean_up_dialog.is_advanced(),
        );
        let exit_state = clean_up_dialog.get_exit_state();
        if exit_state != DialogExitState::Cancel {
            if exit_state == DialogExitState::Postpone {
                self.get_recon_process_track()
                    .set_clean_up_state(ProcessState::InProgress);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_clean_up_state(ProcessState::InProgress);
                }
            } else if exit_state != DialogExitState::Save {
                self.get_recon_process_track()
                    .set_clean_up_state(ProcessState::Complete);
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.set_clean_up_state(ProcessState::Complete);
                }
            }
            self.save_storables(Some(AxisID::Only));
        }
    }

    /// Java `imodCombinedTomogram(Run3dmodMenuOptions)` (ApplicationManager.java:8976).
    /// Open the combined (or full) volume in 3dmod.
    pub fn imod_combined_tomogram(&'static self, menu_options: Run3dmodMenuOptions) {
        match self
            .get_imod_manager()
            .open_string_axis_id_run3dmod_menu_options(
                imod_manager::COMBINED_TOMOGRAM_KEY,
                Some(AxisID::Only),
                Some(menu_options),
            ) {
            Ok(_) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "Cannot open 3dmod on the trimmed tomogram",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(AxisID::Only),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(AxisID::Only),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `updateLog(String, AxisID)` (ApplicationManager.java:9003).  Calls
    /// ProcessManager.generateAlignLogs() when the ta files are out of date.  Returns
    /// true if any log files where changed.
    pub fn update_log(&'static self, command_name: &str, axis_id: AxisID) -> bool {
        let _ = command_name;
        let align_log_file = match LogFile::get_instance_process_name(
            &self.get_property_user_dir().unwrap_or_default(),
            axis_id,
            ProcessName::ALIGN,
            Some(self.get_emergency_monitor(Some(axis_id))),
        ) {
            Ok(align_log_file) => align_log_file,
            Err(e) => {
                // `catch (final FileException e)` and `catch (final IOException e)`
                eprintln!("{e:?}");
                return false;
            }
        };
        // File alignLog = new File(propertyUserDir, alignLogName);
        let ta_error_log_name = format!("taError{}.log", axis_id.get_extension());
        let ta_error_log: PathBuf = self
            .get_property_user_dir()
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(&ta_error_log_name);
        if !align_log_file.exists() {
            return false;
        }
        if !ta_error_log.exists()
            || utilities::java_io_file_last_modified(&ta_error_log.to_string_lossy())
                < align_log_file.last_modified()
        {
            self.get_process_mgr().generate_align_logs(axis_id);
            return true;
        }
        false
    }

    /// Java `generateAlignLogs(AxisID)` (ApplicationManager.java:9030).
    pub fn generate_align_logs(&self, axis_id: AxisID) {
        self.get_process_mgr().generate_align_logs(axis_id);
    }

    /// Java `imodTrimmedVolume(Run3dmodMenuOptions, AxisID)`
    /// (ApplicationManager.java:9037).  Open the trimmed volume in 3dmod.
    pub fn imod_trimmed_volume(&'static self, menu_options: Run3dmodMenuOptions, axis_id: AxisID) {
        let mut trimvol_param = TrimvolParam::new(self, Some(trimvol_param::Mode::PostProcessing));
        // Java dereferences the dialog field without a null check.
        let Some(post_processing_dialog) = self.post_processing_dialog.get() else {
            return;
        };
        if !post_processing_dialog.get_parameters_trimvol_param_boolean(&mut trimvol_param, true) {
            return;
        }
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<(), ImodManagerError> {
            imod_manager.set_swap_yz_string_axis_id_boolean(
                imod_manager::TRIMMED_VOLUME_KEY,
                Some(axis_id),
                !trimvol_param.is_swap_yz() && !trimvol_param.is_rotate_x(),
            )?;
            imod_manager.set_start_new_contours_at_new_z(
                imod_manager::TRIMMED_VOLUME_KEY,
                Some(axis_id),
                false,
            )?;
            imod_manager.open_string_axis_id_run3dmod_menu_options(
                imod_manager::TRIMMED_VOLUME_KEY,
                Some(axis_id),
                Some(menu_options),
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!("{except}\nCan't open 3dmod on the trimmed tomogram"),
                    "Cannot Open 3dmod",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodFlatten(Run3dmodMenuOptions, AxisID)` (ApplicationManager.java:9073).
    /// Open flatten.com output.
    pub fn imod_flatten(&'static self, menu_options: Run3dmodMenuOptions, axis_id: AxisID) {
        let key = imod_manager::FLAT_VOLUME_KEY;
        match self
            .get_imod_manager()
            .open_string_axis_id_run3dmod_menu_options(key, Some(axis_id), Some(menu_options))
        {
            Ok(_) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!("{except}\nCannot open 3dmod on the {key}"),
                    "Cannot Open 3dmod",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `isFlipped(FileType)` (ApplicationManager.java:9094).
    pub fn is_flipped(&self, file_type: Option<&Arc<FileType>>) -> bool {
        let Some(file_type) = file_type else {
            return false;
        };
        // Java compares the singletons by identity.
        if Arc::ptr_eq(file_type, &file_type::CLASS.trim_vol_output) {
            return self.is_trimvol_flipped();
        } else if Arc::ptr_eq(file_type, &file_type::CLASS.reduce_filt_vol_output_file)
            || Arc::ptr_eq(file_type, &file_type::CLASS.flatten_reduce_filt_vol_file)
        {
            return self.is_reduce_filt_vol_flipped();
        }
        // `throw new IllegalStateException(...)`: a programming-error assertion, kept.
        panic!("Unknown file type {file_type}");
    }

    /// Java `isSqueezevolFlipped()` (ApplicationManager.java:9114).  Return true if the
    /// result of squeezevol is flipped.  If squeezevol hasn't been done, return true
    /// if the result of trimvol is flipped.
    pub fn is_squeezevol_flipped(&self) -> bool {
        let squeezevol_flipped = self.get_state().get_squeezevol_flipped();
        if !squeezevol_flipped.is_null() {
            return squeezevol_flipped.is();
        }
        self.is_trimvol_flipped()
    }

    /// Java `isTrimvolFlipped()` (ApplicationManager.java:9132).  Return true if the
    /// result of trimvol is flipped.
    pub fn is_trimvol_flipped(&self) -> bool {
        let trimvol_flipped = self.get_state().get_trimvol_flipped();
        if trimvol_flipped.is_null() {
            return self.get_state().get_backward_compatible_trimvol_flipped();
        }
        trimvol_flipped.is()
    }

    /// Java `isReduceFiltVolFlipped()` (ApplicationManager.java:9139).
    pub fn is_reduce_filt_vol_flipped(&self) -> bool {
        self.get_state().get_reduce_filt_vol_flipped().is()
    }

    /// Java `isFlattenFlipped()` (ApplicationManager.java:9143).
    pub fn is_flatten_flipped(&self) -> bool {
        self.get_state().is_flatten_flipped()
    }

    /// Java `isResultSetFlattenFlipped()` (ApplicationManager.java:9147).
    pub fn is_result_set_flatten_flipped(&self) -> bool {
        self.get_state().is_result_set_flatten_flipped()
    }

    /// Java `imodMakeSurfaceModel(Run3dmodMenuOptions, AxisID, int, FileType, File)`
    /// (ApplicationManager.java:9151).
    pub fn imod_make_surface_model(
        &'static self,
        menu_options: Run3dmodMenuOptions,
        axis_id: AxisID,
        binning: i32,
        file_type: &Arc<FileType>,
        file: Option<&Path>,
    ) {
        // Pick ImodManager key
        // Need to look at tomogram edge on. Use -Y, unless using squeezevol and it
        // is not flipped.
        let key: Option<String> = file_type.get_imod_manager_key().map(str::to_owned);
        let use_swap_yz = self.is_flipped(Some(file_type));
        let imod_manager = self.get_imod_manager();
        let key_ref = key.as_deref();
        let result = (|| -> Result<(), ImodManagerError> {
            if Arc::ptr_eq(file_type, &file_type::CLASS.trim_vol_output) {
                imod_manager.set_swap_yz_string_axis_id_boolean(
                    key_ref.unwrap_or("null"),
                    Some(axis_id),
                    use_swap_yz,
                )?;
            } else {
                imod_manager.set_swap_yz_string_axis_id_file_boolean(
                    key_ref.unwrap_or("null"),
                    Some(axis_id),
                    file,
                    use_swap_yz,
                )?;
            }
            imod_manager.set_open_contours(key_ref.unwrap_or("null"), Some(axis_id), true)?;
            imod_manager.set_start_new_contours_at_new_z(
                key_ref.unwrap_or("null"),
                Some(axis_id),
                true,
            )?;
            imod_manager.set_binning_xy_string_int(key_ref.unwrap_or("null"), binning)?;
            let model = file_type::CLASS
                .flatten_warp_input_model
                .get_file_name(Some(self), Some(axis_id));
            match file {
                None => {
                    imod_manager.open_string_axis_id_string_boolean_run3dmod_menu_options(
                        key_ref.unwrap_or("null"),
                        Some(axis_id),
                        model.as_deref(),
                        true,
                        Some(menu_options),
                    )?;
                }
                Some(file) => {
                    imod_manager.open_string_axis_id_file_string_boolean_run3dmod_menu_options(
                        key_ref.unwrap_or("null"),
                        Some(axis_id),
                        Some(file),
                        model.as_deref(),
                        true,
                        Some(menu_options),
                    )?;
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &format!(
                        "{except}\nCan't open 3dmod on the {}",
                        key_ref.unwrap_or("null")
                    ),
                    "Cannot Open 3dmod",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodSqueezedVolume(Run3dmodMenuOptions, AxisID)`
    /// (ApplicationManager.java:9196).  Open the squeezed volume in 3dmod.
    pub fn imod_squeezed_volume(&'static self, menu_options: Run3dmodMenuOptions, axis_id: AxisID) {
        // Make sure that the post processing panel is open
        if !self.post_processing_dialog.is_some() {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Post processing dialog not open",
                "Program logic error",
                Some(axis_id),
            );
            return;
        }
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<(), ImodManagerError> {
            imod_manager.set_swap_yz_string_axis_id_boolean(
                imod_manager::SQUEEZED_VOLUME_KEY,
                Some(axis_id),
                !self.is_squeezevol_flipped(),
            )?;
            imod_manager.set_start_new_contours_at_new_z(
                imod_manager::SQUEEZED_VOLUME_KEY,
                Some(axis_id),
                false,
            )?;
            imod_manager.open_string_axis_id_run3dmod_menu_options(
                imod_manager::SQUEEZED_VOLUME_KEY,
                Some(axis_id),
                Some(menu_options),
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "Can't open 3dmod on the squeezed tomogram",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `imodReducedFilteredVolume(Run3dmodMenuOptions, AxisID, File)`
    /// (ApplicationManager.java:9226).
    pub fn imod_reduced_filtered_volume(
        &'static self,
        menu_options: Run3dmodMenuOptions,
        axis_id: AxisID,
        file: Option<&Path>,
    ) {
        // Make sure that the post processing panel is open
        if !self.post_processing_dialog.is_some() {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Post processing dialog not open",
                "Program logic error",
                Some(axis_id),
            );
            return;
        }
        let imod_manager = self.get_imod_manager();
        let result = (|| -> Result<(), ImodManagerError> {
            imod_manager.set_swap_yz_string_axis_id_file_boolean(
                imod_manager::REDUCED_FILTERED_VOLUME_KEY,
                Some(axis_id),
                file,
                !self.is_reduce_filt_vol_flipped(),
            )?;
            imod_manager.set_start_new_contours_at_new_z(
                imod_manager::REDUCED_FILTERED_VOLUME_KEY,
                Some(axis_id),
                false,
            )?;
            imod_manager.open_string_axis_id_file_run3dmod_menu_options(
                imod_manager::REDUCED_FILTERED_VOLUME_KEY,
                Some(axis_id),
                file,
                Some(menu_options),
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(ImodManagerError::SystemProcess(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "Can't open 3dmod on the reduced/filtered tomogram",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::AxisType(except)) => {
                eprintln!("{except:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &except.to_string(),
                    "AxisType problem",
                    Some(axis_id),
                );
            }
            Err(ImodManagerError::Io(e)) => {
                eprintln!("{e:?}");
                ui_harness::open_message_dialog_from_process(
                    Some(self),
                    &e.to_string(),
                    "IO Exception",
                    Some(axis_id),
                );
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `trimVolume(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType)` (ApplicationManager.java:9260).  Execute
    /// trimvol.
    pub fn trim_volume(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = process_series.unwrap_or_else(|| {
            ProcessSeries::new(self, AxisID::First, Some(dialog_type), Some("trimVolume"))
        });
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Make sure that the post processing panel is open
        if !self.post_processing_dialog.is_some() {
            ui_harness::open_message_dialog_from_process(
                Some(self),
                "Post processing dialog not open",
                "Program logic error",
                Some(AxisID::Only),
            );
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        let Some(trimvol_param) = self.update_trimvol_param() else {
            process_series.borrow().end_series();
            return;
        };
        // Start the trimvol process
        self.get_recon_process_track()
            .set_post_processing_state(ProcessState::InProgress);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_post_processing_state(ProcessState::InProgress);
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        let process_series_ref: ProcessSeriesRef =
            Arc::new(EdtRef::new(Rc::clone(&process_series)));
        let thread_name = match self.get_process_mgr().trim_volume(
            Arc::new(trimvol_param),
            process_result_display_ref.clone(),
            Some(process_series_ref),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = ["Can not execute trimvol command".to_string(), e.0.clone()];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute command",
                    Some(AxisID::Only),
                );
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Trimming volume"),
                AxisID::Only,
                Some(&ProcessName::TRIMVOL),
            );
        }
    }

    /// Java package-private final `updateWarpVolParam(WarpVolDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:9302).
    pub fn update_warp_vol_param(
        &'static self,
        display: &dyn WarpVolDisplay,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<WarpVolParam> {
        let mut param = self
            .get_com_script_manager()
            .get_warp_vol_param_from_flatten(axis_id);
        // `if (display == null)`: "Unable to get information from the display.",
        // "Etomo Error".  The display is a reference here, so the branch cannot be
        // taken.
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        self.get_com_script_manager().save_flatten(&param, axis_id);
        Some(param)
    }

    /// Java package-private `updateFlattenWarpParam(FlattenWarpDisplay, AxisID,
    /// boolean)` (ApplicationManager.java:9317).
    pub fn update_flatten_warp_param(
        &'static self,
        display: &dyn FlattenWarpDisplay,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<FlattenWarpParam> {
        let _ = axis_id;
        let mut param = FlattenWarpParam::new(self);
        // `if (display == null)`: "Unable to get information from the display.",
        // "Etomo Error".  The display is a reference here, so the branch cannot be
        // taken.
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        Some(param)
    }

    /// Java `flatten(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, AxisID, WarpVolDisplay)`
    /// (ApplicationManager.java:9334).  Execute flatten.com.
    #[allow(clippy::too_many_arguments)]
    pub fn flatten(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        axis_id: AxisID,
        display: &dyn WarpVolDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = process_series.unwrap_or_else(|| {
            ProcessSeries::new(self, axis_id, Some(dialog_type), Some("flatten"))
        });
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let Some(param) = self.update_warp_vol_param(display, axis_id, true) else {
            process_series.borrow().end_series();
            return;
        };
        // Run process
        // `if (processTrack != null)`: the track is created with the manager.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        let process_series_ref: ProcessSeriesRef =
            Arc::new(EdtRef::new(Rc::clone(&process_series)));
        let thread_name = match self.get_process_mgr().flatten(
            Arc::new(param),
            axis_id,
            process_result_display_ref.clone(),
            Some(process_series_ref),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = [
                    format!("Can not execute {}", ProcessName::FLATTEN),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute command",
                    Some(axis_id),
                );
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
    }

    /// Java `flattenWarp(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, AxisID, FlattenWarpDisplay)`
    /// (ApplicationManager.java:9371).  Execute flattenwarp.
    #[allow(clippy::too_many_arguments)]
    pub fn flatten_warp(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Run3dmodMenuOptions,
        dialog_type: DialogType,
        axis_id: AxisID,
        display: &dyn FlattenWarpDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = process_series.unwrap_or_else(|| {
            ProcessSeries::new(self, axis_id, Some(dialog_type), Some("flattenWarp"))
        });
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let Some(param) = self.update_flatten_warp_param(display, axis_id, true) else {
            process_series.borrow().end_series();
            return;
        };
        let param = Arc::new(param);
        // Run process
        // `if (processTrack != null)`: the track is created with the manager.
        self.get_recon_process_track().set_state_dialog_type(
            ProcessState::InProgress,
            axis_id,
            dialog_type,
        );
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, Some(run_3dmod_menu_options));
        let process_series_ref: ProcessSeriesRef =
            Arc::new(EdtRef::new(Rc::clone(&process_series)));
        let thread_name = match self.get_process_mgr().flatten_warp(
            Arc::clone(&param),
            process_result_display_ref.clone(),
            Some(process_series_ref),
            axis_id,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e:?}");
                let message = [
                    format!("Can not execute {}", param.get_process_name()),
                    e.0.clone(),
                ];
                ui_harness::open_message_dialog_array_from_process(
                    Some(self),
                    &message,
                    "Unable to execute command",
                    Some(axis_id),
                );
                return;
            }
        };
        self.set_thread_name(Some(&thread_name), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", param.get_process_name()).as_str()),
                axis_id,
                Some(&param.get_process_name()),
            );
        }
    }

    /// Java `reduceFiltVol(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, ReduceFiltVolDisplay)`
    /// (ApplicationManager.java:9577).  Execute reducefiltvol.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_filt_vol(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        display: &dyn ReduceFiltVolDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(
                self,
                AxisID::First,
                Some(dialog_type),
                Some("reduceFiltVol"),
            ),
        };
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Make sure that the post processing panel is open
        if !self.post_processing_dialog.is_some() {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    "Post processing dialog not open",
                    "Program logic error",
                    Some(AxisID::Only),
                )
            });
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        let Some(reduce_filt_vol_param) =
            self.update_reduce_filt_vol_param(display, AxisID::Only, true)
        else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        // Start the trimvol process
        self.get_recon_process_track()
            .set_post_processing_state(ProcessState::InProgress);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_post_processing_state(ProcessState::InProgress);
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        let thread_name = match self.get_process_mgr().reduce_filt_vol(
            Arc::new(reduce_filt_vol_param),
            process_result_display_ref.clone(),
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some("Can not execute reducefiltvol command".to_string()),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(AxisID::Only),
                    )
                });
                return;
            }
        };
        self.set_thread_name(Some(thread_name.as_str()), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some("Reduce/filter volume"),
                AxisID::Only,
                Some(&ProcessName::REDUCE_FILT_VOL),
            );
        }
    }

    /// Java private `updateTrimvolParam()` (ApplicationManager.java:9629).  Updates
    /// ConstMetaData.trimvolParam with data from postProcessingDialog; returns the
    /// updated TrimvolParam.
    fn update_trimvol_param(&'static self) -> Option<TrimvolParam> {
        // Get trimvol param data from dialog.
        let mut param = TrimvolParam::new(self, Some(trimvol_param::Mode::PostProcessing));
        // FIXED (upstream NPE, ApplicationManager.java:9632): Java dereferences a null
        // `postProcessingDialog`; here a missing dialog is the method's own failure
        // return (null).
        let Some(post_processing_dialog) = self.post_processing_dialog.get() else {
            return None;
        };
        if !post_processing_dialog.get_parameters_trimvol_param_boolean(&mut param, true) {
            return None;
        }
        let meta_data = self.get_meta_data();
        post_processing_dialog.get_parameters_for_trimvol(meta_data);
        // Add input and output files.
        param.set_input_file_name_for(
            meta_data.base().get_axis_type(),
            Some(meta_data.get_dataset_name().as_str()),
        );
        param.set_output_file_name(
            &file_type::CLASS
                .trim_vol_output
                .get_file_name(Some(self), Some(AxisID::First))
                .unwrap_or_default(), /*was: metaData.getDatasetName() + ".rec"*/
        );
        if meta_data.base().get_axis_type() == AxisType::SingleAxis
            && !self.get_state().is_adjust_origin(AxisID::Only)
        {
            param.set_keep_same_origin(true);
        }
        param.set_old_flipped_coordinates_scaling(
            meta_data.is_post_trimvol_new_style_z(),
            meta_data.is_post_trimvol_scaling_new_style_z(),
        );
        Some(param)
    }

    /// Java private `updateReduceFiltVolParam(ReduceFiltVolDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:9649).
    fn update_reduce_filt_vol_param(
        &'static self,
        display: &dyn ReduceFiltVolDisplay,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<ReduceFiltVolParam> {
        let mut param = self
            .get_com_script_manager()
            .get_reduce_filt_vol_param(axis_id);
        match display.get_parameters(&mut param, do_validation) {
            Ok(true) => {}
            Ok(false) => return None,
            Err(except) => {
                except.print_stack_trace();
                let error_message: [Option<String>; 3] =
                    [except.get_message().map(str::to_string), None, None];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &error_message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "ReduceFiltVol Parameter Syntax Error",
                        Some(AxisID::Only),
                    )
                });
                return None;
            }
        }
        self.get_com_script_manager()
            .save_reduce_filt_vol(&param, axis_id);
        Some(param)
    }

    //
    // Utility functions
    //

    /// Java private `showIfExists(ProcessDialog, ProcessDialog, AxisID, String)`
    /// (ApplicationManager.java:9673).
    fn show_if_exists(
        &self,
        panel_a: Option<&ProcessDialog>,
        panel_b: Option<&ProcessDialog>,
        axis_id: AxisID,
        action_message: Option<&str>,
    ) -> bool {
        if axis_id == AxisID::Second {
            match panel_b {
                None => return false,
                Some(panel_b) => {
                    if let Some(main_panel) = self.main_panel.get() {
                        main_panel.show_process(&panel_b.get_container(), axis_id);
                    }
                }
            }
        } else {
            match panel_a {
                None => return false,
                Some(panel_a) => {
                    if let Some(main_panel) = self.main_panel.get() {
                        main_panel.show_process(&panel_a.get_container(), axis_id);
                    }
                }
            }
        }
        if let Some(action_message) = action_message {
            eprintln!("{action_message}");
        }
        true
    }

    /// Java private `updateDialog(FiducialModelDialog, AxisID)`
    /// (ApplicationManager.java:9809).
    fn update_dialog_fiducial_model_dialog_axis_id(
        &'static self,
        dialog: Option<Rc<FiducialModelDialog>>,
        axis_id: AxisID,
    ) {
        // `Thread.sleep(100)`; an InterruptedException is ignored.
        std::thread::sleep(std::time::Duration::from_millis(100));
        let prealis_exist = file_type::CLASS
            .prealigned_stack
            .exists(Some(self), Some(AxisID::First))
            && file_type::CLASS
                .prealigned_stack
                .exists(Some(self), Some(AxisID::Second));
        let fid_exists;
        if axis_id == AxisID::First {
            fid_exists = utilities::file_exists(self, Some(".fid"), Some(AxisID::Second));
        } else {
            fid_exists = utilities::file_exists(self, Some(".fid"), Some(AxisID::First));
        }
        if let Some(dialog) = dialog {
            dialog.set_transferfid_enabled(prealis_exist && fid_exists);
            dialog.update_enabled();
        }
    }

    /// Java private `setBackgroundThreadName(String, AxisID, String)`
    /// (ApplicationManager.java:9829).
    fn set_background_thread_name(
        &self,
        name: Option<&str>,
        axis_id: AxisID,
        process_name: Option<&str>,
    ) {
        self.set_thread_name(name, Some(axis_id));
        if axis_id == AxisID::Second {
            // Java `throw new IllegalStateException("No Axis B background processes
            // exist.")`: an unchecked exception that Swing's event thread reports and
            // survives, after `setThreadName` has already run.  Reported the same way
            // here instead of unwinding.
            eprintln!("java.lang.IllegalStateException: No Axis B background processes exist.");
            return;
        } else {
            *self.base().background_process_a.lock().unwrap() = true;
            *self.base().background_process_name_a.lock().unwrap() =
                process_name.map(str::to_string);
        }
    }

    // Test helper functions

    /// Java package-private `getThreadName(AxisID)` (ApplicationManager.java:9848).
    /// Return the currently executing thread name for the specified axis.
    pub fn get_thread_name(&self, axis_id: AxisID) -> String {
        if axis_id == AxisID::Second {
            return self.base().thread_name_b.lock().unwrap().clone();
        }
        self.base().thread_name_a.lock().unwrap().clone()
    }

    /// Java package-private `checkUpdateFiducialModel(AxisID, ProcessResultDisplay,
    /// ConstProcessSeries)` (ApplicationManager.java:9855).
    pub fn check_update_fiducial_model(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let mut fid_xyz = UIExpertUtilities::INSTANCE.get_fid_xyz(self, axis_id);
        let prealigned_header = MRCHeader::get_instance_from_file_type(
            self,
            Some(axis_id),
            &file_type::CLASS.prealigned_stack,
        );
        let rawstack_header = MRCHeader::get_instance_from_file_type(
            self,
            Some(axis_id),
            &file_type::CLASS.raw_stack,
        );
        // `MRCHeader.getInstance` never returns null in Java; a None here (no key file)
        // takes the same exit as a failed read.
        let (Some(preali_header), Some(rawstack_header)) = (prealigned_header, rawstack_header)
        else {
            self.process_done_secondary(
                Some(axis_id),
                process_result_display_ref.clone(),
                process_series.as_ref(),
            );
            return;
        };
        // try { fidXyz.read(); if (!prealiHeader.read(this) || !rawstackHeader.read(this))
        // ... } catch (IOException) {...} catch (InvalidParameterException) {
        // printStackTrace; ... }.  The translated readers report both exceptions as one
        // `Err(String)`, so the InvalidParameterException stack trace is not
        // distinguishable and is not printed.
        if fid_xyz.read().is_err() {
            self.process_done_secondary(
                Some(axis_id),
                process_result_display_ref.clone(),
                process_series.as_ref(),
            );
            return;
        }
        let read_ok = match preali_header.borrow_mut().read_with_manager(self) {
            Err(_) => None,
            Ok(false) => Some(false),
            Ok(true) => match rawstack_header.borrow_mut().read_with_manager(self) {
                Err(_) => None,
                Ok(read) => Some(read),
            },
        };
        match read_ok {
            None | Some(false) => {
                self.process_done_secondary(
                    Some(axis_id),
                    process_result_display_ref.clone(),
                    process_series.as_ref(),
                );
                return;
            }
            Some(true) => {}
        }
        if !fid_xyz.exists() {
            self.process_done_secondary(
                Some(axis_id),
                process_result_display_ref.clone(),
                process_series.as_ref(),
            );
            return;
        }
        // if fidXyz.getPixelSize() is 1, then the binning used in align.com must
        // have been 1, if the preali binning is also 1, then no error message
        // should
        // be sent. preali binning is preali pixel spacing / .st pixel spacing
        let fid_xyz_pixel_size_set = fid_xyz.is_pixel_size_set();
        if !fid_xyz_pixel_size_set {
            let a = preali_header.borrow().get_x_pixel_spacing()
                / rawstack_header.borrow().get_x_pixel_spacing();
            // `Math.round(double)`, the JDK body (not the javadoc's floor(a + 0.5)).
            let rounded: i64 = {
                let long_bits = a.to_bits() as i64;
                let biased_exp = (long_bits & 0x7FF0_0000_0000_0000) >> (53 - 1); // EXP_BIT_MASK
                let shift = (53 - 2 + 1023) - biased_exp; // SIGNIFICAND_WIDTH - 2 + EXP_BIAS
                if (shift & -64) == 0 {
                    let mut r = (long_bits & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000;
                    if long_bits < 0 {
                        r = -r;
                    }
                    ((r >> shift) + 1) >> 1
                } else {
                    a as i64
                }
            };
            if rounded == 1 {
                self.process_done_secondary(Some(axis_id), process_result_display_ref, None);
                return;
            }
        }
        if !fid_xyz_pixel_size_set
            || fid_xyz.get_pixel_size() != preali_header.borrow().get_x_pixel_spacing()
        {
            // if (getStackBinning(axisID, ".preali") !=
            // getBackwardCompatibleAlignBinning(axisID)) {
            let title = "Prealigned image stack binning has changed";
            let message: [Option<String>; 4] = [
                Some(
                    "The prealigned image stack binning has changed.  You must:".to_string(),
                ),
                Some(
                    "    1. Go  to Fiducial Model Gen. and Press Fix Fiducial Model to open the fiducial model."
                        .to_string(),
                ),
                Some("    2. Save the fiducial model by pressing \"s\".".to_string()),
                Some(format!(
                    "    3. Go to Fine Alignment and press Compute Alignment to rerun align{}.com.",
                    axis_id.get_extension()
                )),
            ];
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                    Some(self),
                    &message
                        .iter()
                        .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                        .collect::<Vec<String>>(),
                    title,
                    Some(axis_id),
                )
            });
        }
        self.process_done_secondary(
            Some(axis_id),
            process_result_display_ref,
            process_series.as_ref(),
        );
    }

    /// Java private `createState()` (ApplicationManager.java:9950).
    // REPLACES existing
    fn create_state(&'static self) {
        let state: &'static TomogramState = Box::leak(Box::new(TomogramState::new(Some(self))));
        ROOTS.lock().unwrap().state.push(state);
        *self.state.lock().unwrap() = Some(state);
    }

    /// Java `getState()` (ApplicationManager.java:9954).
    // REPLACES existing
    pub fn get_state(&self) -> &'static TomogramState {
        self.state.lock().unwrap().expect("state")
    }

    /// Java `getMetaData()` (ApplicationManager.java:9963).
    // REPLACES existing
    pub fn get_meta_data(&self) -> &'static MetaData {
        self.meta_data.lock().unwrap().expect("metaData")
    }

    /// Java `getConstMetaData()` (ApplicationManager.java:9967).
    // REPLACES existing
    pub fn get_const_meta_data(&self) -> &'static MetaData {
        self.get_meta_data()
    }

    /// Java `xfmodel(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, AxisID, DialogType)` (ApplicationManager.java:9971).
    #[allow(clippy::too_many_arguments)]
    pub fn xfmodel_process_result_display_process_series_deferred3dmod_button_run3dmod_menu_options_axis_id_dialog_type(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("xfmodel")),
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let param = XfmodelParam::new(self, axis_id);
        let process_track = *self.process_track.lock().unwrap();
        if let Some(process_track) = process_track {
            process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let process_result = self
            .xfmodel_axis_id_process_result_display_process_series_xfmodel_param_dialog_type(
                axis_id,
                process_result_display.clone(),
                Some(process_series),
                param,
                dialog_type,
            );
        self.send_msg(process_result, process_result_display);
    }

    /// Java `seedEraseFiducialModel(Run3dmodMenuOptions, AxisID, DialogType)`
    /// (ApplicationManager.java:9990).
    pub fn seed_erase_fiducial_model(
        &'static self,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        if !file_type::CLASS
            .aligned_stack
            .exists(Some(self), Some(axis_id))
        {
            let message = format!(
                "To see this model, build the aligned stack in the {} tab.",
                final_aligned_stack_dialog::FINAL_ALIGNED_STACK_TAB_LABEL
            );
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string(
                    Some(self),
                    &message,
                    "Aligned Stack Required",
                )
            });
            return;
        }
        self.imod_seed_model(
            axis_id,
            run_3dmod_menu_options.unwrap_or_default(),
            None,
            imod_manager::FINE_ALIGNED_KEY,
            file_type::CLASS
                .ccd_eraser_beads_input_model
                .get_file_name(Some(self), Some(axis_id))
                .as_deref(),
            None,
            dialog_type,
        );
    }

    /// Java private `xfmodel(AxisID, ProcessResultDisplay, ProcessSeries, XfmodelParam,
    /// DialogType)` (ApplicationManager.java:10003).
    fn xfmodel_axis_id_process_result_display_process_series_xfmodel_param_dialog_type(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: XfmodelParam,
        dialog_type: DialogType,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("xfmodel")),
        };
        let thread_name = match self.get_process_mgr().base.xfmodel(
            Arc::new(param),
            axis_id,
            process_result_display_ref,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", ProcessName::XFMODEL)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::XFMODEL).as_str()),
                axis_id,
                Some(&ProcessName::XFMODEL),
            );
        }
        None
    }

    /// Java `useFileAsFullAlignedStack(ProcessResultDisplay, FileType, String, AxisID,
    /// DialogType)` (ApplicationManager.java:10033).  Replace the full aligned stack
    /// with the output of CTF correction or bead erase; returns true if succeeded.
    pub fn use_file_as_full_aligned_stack(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        use_file_type: &Arc<FileType>,
        run_button_label: &str,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> bool {
        self.use_image_file(
            process_result_display,
            use_file_type,
            None,
            &file_type::CLASS.aligned_stack,
            run_button_label,
            axis_id,
            dialog_type,
        )
    }

    /// Java `useCcdEraser(ProcessResultDisplay, AxisID, DialogType, String)`
    /// (ApplicationManager.java:10040).
    pub fn use_ccd_eraser(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        axis_id: AxisID,
        dialog_type: DialogType,
        ccd_eraser_label: &str,
    ) {
        self.use_file_as_full_aligned_stack(
            process_result_display,
            &file_type::CLASS.erased_beads_stack,
            ccd_eraser_label,
            axis_id,
            dialog_type,
        );
        self.get_state()
            .set_use_erased_stack_warning(axis_id, false);
    }

    /// Java `imodErasedFiducials(Run3dmodMenuOptions, AxisID)`
    /// (ApplicationManager.java:10047).
    pub fn imod_erased_fiducials(
        &'static self,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
    ) {
        self.imod_open_axis(
            Some(axis_id),
            Some(imod_manager::ERASED_FIDUCIALS_KEY),
            file_type::CLASS
                .ccd_eraser_beads_input_model
                .get_file_name(Some(self), Some(axis_id))
                .as_deref(),
            run_3dmod_menu_options,
            false,
        );
    }

    /// Java `updateGoldEraserParam(CcdEraserDisplay, AxisID, boolean)`
    /// (ApplicationManager.java:10054).
    pub fn update_gold_eraser_param(
        &'static self,
        display: &dyn CcdEraserDisplay,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Option<CCDEraserParam> {
        if !self
            .get_com_script_manager()
            .load_gold_eraser(axis_id, false)
        {
            let mut makecom_file_param = MakecomfileParam::new(
                self,
                axis_id,
                Arc::clone(&file_type::CLASS.gold_eraser_comscript),
            );
            if !display.get_parameters_makecomfile(&mut makecom_file_param, true) {
                return None;
            }
            self.makecomfile(axis_id, &mut makecom_file_param);
            self.get_com_script_manager()
                .load_gold_eraser(axis_id, true);
        }
        let mut param = self
            .get_com_script_manager()
            .get_gold_eraser_param(axis_id, Some(ccd_eraser_param::Mode::Beads));
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        self.get_com_script_manager()
            .save_gold_eraser(&param, axis_id);
        Some(param)
    }

    /// Java `makecomfile(AxisID, MakecomfileParam)` (ApplicationManager.java:10074).
    pub fn makecomfile(&self, axis_id: AxisID, param: &mut MakecomfileParam) -> bool {
        self.get_process_mgr().makecomfile(axis_id, param)
    }

    /// Java `goldEraser(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, AxisID, DialogType, CcdEraserDisplay)`
    /// (ApplicationManager.java:10078).
    #[allow(clippy::too_many_arguments)]
    pub fn gold_eraser_process_result_display_process_series_deferred3dmod_button_run3dmod_menu_options_axis_id_dialog_type_ccd_eraser_display(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        axis_id: AxisID,
        dialog_type: DialogType,
        display: &dyn CcdEraserDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("goldEraser")),
        };
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let Some(param) = self.update_gold_eraser_param(display, axis_id, true) else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        };
        let process_track = *self.process_track.lock().unwrap();
        if let Some(process_track) = process_track {
            process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let process_result = self
            .gold_eraser_axis_id_process_result_display_process_series_ccd_eraser_param_dialog_type(
                axis_id,
                process_result_display.clone(),
                Some(process_series),
                param,
                dialog_type,
            );
        self.send_msg(process_result, process_result_display);
    }

    /// Java private `goldEraser(AxisID, ProcessResultDisplay, ProcessSeries,
    /// CCDEraserParam, DialogType)` (ApplicationManager.java:10102).
    fn gold_eraser_axis_id_process_result_display_process_series_ccd_eraser_param_dialog_type(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: CCDEraserParam,
        dialog_type: DialogType,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("goldEraser")),
        };
        let thread_name = match self.get_process_mgr().gold_eraser(
            axis_id,
            process_result_display_ref,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            Arc::new(param),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", ProcessName::GOLD_ERASER)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::GOLD_ERASER).as_str()),
                axis_id,
                Some(&ProcessName::GOLD_ERASER),
            );
        }
        None
    }

    /// Java `openNextDialog(AxisID, DialogType)` (ApplicationManager.java:10137).
    pub fn open_next_dialog(&'static self, axis_id: AxisID, dialog_type: DialogType) {
        if dialog_type == DialogType::TomogramPositioning {
            // Java dereferences the expert without a null check; for this dialog type
            // getUIExpert never returns null.
            if let Some(expert) = self.get_ui_expert(Some(DialogType::FinalAlignedStack), axis_id) {
                expert.open_dialog();
            }
        } else if dialog_type == DialogType::FinalAlignedStack {
            if let Some(expert) = self.get_ui_expert(Some(DialogType::TomogramGeneration), axis_id)
            {
                expert.open_dialog();
            }
        } else if dialog_type == DialogType::TomogramGeneration {
            if self.is_dual_axis() {
                if let Some(main_panel) = self.main_panel.get() {
                    main_panel.show_blank_process(axis_id);
                }
            } else {
                self.open_post_processing_dialog();
            }
        }
    }

    /// Java private `splittilt(AxisID, ProcessResultDisplay, ProcessSeries,
    /// SplittiltParam, DialogType)` (ApplicationManager.java:10167).
    fn splittilt_axis_id_process_result_display_process_series_splittilt_param_dialog_type(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: SplittiltParam,
        dialog_type: DialogType,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("splittilt")),
        };
        let thread_name = match self.get_process_mgr().splittilt(
            Arc::new(param),
            axis_id,
            process_result_display_ref,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", splittilt_param::COMMAND_NAME)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", splittilt_param::COMMAND_NAME).as_str()),
                axis_id,
                Some(&ProcessName::SPLITTILT),
            );
        }
        None
    }

    /// Java `splitCorrection(AxisID, ProcessResultDisplay, ProcessSeries,
    /// ConstSplitCorrectionParam, DialogType, ProcessingMethod)`
    /// (ApplicationManager.java:10192).
    pub fn split_correction(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        param: Arc<SplitCorrectionParam>,
        dialog_type: DialogType,
        correction_processing_method: Option<ProcessingMethod>,
    ) -> Option<ProcessResult> {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("splitCorrection")),
        };
        let thread_name = match self.get_process_mgr().split_correction(
            param,
            axis_id,
            process_result_display_ref,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", ProcessName::SPLIT_CORRECTION)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                return Some(ProcessResult::FAILED_TO_START);
            }
        };
        process_series
            .borrow_mut()
            .set_next_process_output_file_type(
                Some(ProcessName::PROCESSCHUNKS.to_string().as_str()),
                Some(ProcessName::CTF_CORRECTION),
                Some(&*file_type::CLASS.ctf_corrected_stack),
                correction_processing_method,
            );
        self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::SPLIT_CORRECTION).as_str()),
                axis_id,
                Some(&ProcessName::SPLIT_CORRECTION),
            );
        }
        None
    }

    /// Java `updateSirtSetupCom(AxisID, SirtsetupDisplay, boolean)`
    /// (ApplicationManager.java:10228).  Updates sirtsetup.com.
    pub fn update_sirt_setup_com(
        &'static self,
        axis_id: AxisID,
        display: &dyn SirtsetupDisplay,
        do_validation: bool,
    ) -> Option<SirtsetupParam> {
        // `getMainPanel().getParallelPanel(axisID)`.
        let parallel_panel = self
            .main_panel
            .get()
            .and_then(|main_panel| main_panel.get_parallel_panel(axis_id));
        let Some(parallel_panel) = parallel_panel else {
            if do_validation {
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
            }
            return None;
        };
        // Java computes `sirtSetupComFile` and never reads it.
        let _sirt_setup_com_file = file_type::CLASS
            .sirtsetup_comscript
            .get_file(Some(self), Some(axis_id));
        let mut param = self.get_com_script_manager().get_sirtsetup_param(axis_id);
        if !display.get_parameters(&mut param, do_validation) {
            return None;
        }
        if !parallel_panel.get_parameters_sirtsetup_param_boolean(&mut param, do_validation) {
            return None;
        }
        self.get_com_script_manager()
            .save_sirtsetup(&param, axis_id);
        Some(param)
    }

    /// Java `msgSirtsetupSucceeded(AxisID)` (ApplicationManager.java:10251).
    pub fn msg_sirtsetup_succeeded(&'static self, axis_id: AxisID) {
        // Posted: Java calls this on the process thread.
        invoke_later(move || {
            if let Some(expert) = self.get_ui_expert(Some(DialogType::TomogramGeneration), axis_id)
                && let Some(expert) = expert.as_any().downcast_ref::<TomogramGenerationExpert>()
            {
                expert.msg_sirtsetup_succeeded();
            }
        });
    }

    /// Java `msgAutofidseedSucceeded(AxisID)` (ApplicationManager.java:10256).
    pub fn msg_autofidseed_succeeded(&'static self, axis_id: AxisID) {
        // Posted: Java calls this on the process thread.
        invoke_later(move || {
            let dialog = if axis_id == AxisID::Second {
                self.fiducial_model_dialog_b.get()
            } else {
                self.fiducial_model_dialog_a.get()
            };
            // FIXED (upstream NPE, ApplicationManager.java:10257): Java calls
            // `updateEnabled` on a null dialog when the fiducial model dialog has been
            // closed before autofidseed finishes; here nothing is updated.
            if let Some(dialog) = dialog {
                dialog.update_enabled();
            }
        });
    }

    /// Java `useTrackAdjustedComfile(AxisID, ProcessResultDisplay)`
    /// (ApplicationManager.java:10261).
    pub fn use_track_adjusted_comfile(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let main_panel = self.main_panel.get();
        if let Some(main_panel) = &main_panel {
            main_panel.start_progress_bar_string_axis_id(
                Some(&format!("Using the adjusted {} comfile", ProcessName::TRACK).as_str()),
                axis_id,
            );
        }
        if !file_type::CLASS
            .track_adjusted_comscript
            .exists(Some(self), Some(axis_id))
        {
            // Nothing to do.
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state_string(
                    axis_id,
                    Some(ProcessEndState::Cancelled),
                    Some(
                        format!(
                            "{} is not available",
                            file_type::CLASS
                                .track_adjusted_comscript
                                .get_file_name(Some(self), Some(axis_id))
                                .unwrap_or_else(|| "null".to_string())
                        )
                        .as_str(),
                    ),
                );
            }
            return;
        }
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        if file_type::CLASS
            .track_comscript
            .exists(Some(self), Some(axis_id))
        {
            match utilities::rename_file(
                Some(self),
                Some(axis_id),
                file_type::CLASS
                    .track_comscript
                    .get_file(Some(self), Some(axis_id))
                    .as_deref(),
                file_type::CLASS
                    .track_orig_comscript
                    .get_file(Some(self), Some(axis_id))
                    .as_deref(),
                false,
                false,
                false,
            ) {
                Ok(_) => {}
                Err(LogFileError::Lock(_)) => {
                    if let Some(main_panel) = &main_panel {
                        main_panel.stop_progress_bar_axis_id_process_end_state(
                            axis_id,
                            Some(ProcessEndState::FileLockFailure),
                        );
                    }
                    return;
                }
                Err(_) => {
                    let message = format!(
                        "Unable to back up {} to {}.",
                        file_type::CLASS
                            .track_comscript
                            .get_file_name(Some(self), Some(axis_id))
                            .unwrap_or_else(|| "null".to_string()),
                        file_type::CLASS
                            .track_orig_comscript
                            .get_file_name(Some(self), Some(axis_id))
                            .unwrap_or_else(|| "null".to_string())
                    );
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self),
                            &message,
                            "Unable to Backup Comscript",
                            Some(axis_id),
                        )
                    });
                    self.send_msg(Some(ProcessResult::FAILED_TO_START), process_result_display);
                    if let Some(main_panel) = &main_panel {
                        main_panel.stop_progress_bar_axis_id_process_end_state(
                            axis_id,
                            Some(ProcessEndState::Failed),
                        );
                    }
                    return;
                }
            }
        }
        match utilities::rename_file(
            Some(self),
            Some(axis_id),
            file_type::CLASS
                .track_adjusted_comscript
                .get_file(Some(self), Some(axis_id))
                .as_deref(),
            file_type::CLASS
                .track_comscript
                .get_file(Some(self), Some(axis_id))
                .as_deref(),
            false,
            true,
            true,
        ) {
            Ok(_) => {}
            Err(LogFileError::Lock(_)) => {
                if let Some(main_panel) = &main_panel {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        Some(ProcessEndState::FileLockFailure),
                    );
                }
                return;
            }
            Err(_) => {
                let message = format!(
                    "Unable to copy {} to {}.",
                    file_type::CLASS
                        .track_adjusted_comscript
                        .get_file_name(Some(self), Some(axis_id))
                        .unwrap_or_else(|| "null".to_string()),
                    file_type::CLASS
                        .track_comscript
                        .get_file_name(Some(self), Some(axis_id))
                        .unwrap_or_else(|| "null".to_string())
                );
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &message,
                        "Unable to Copy Comscript",
                        Some(axis_id),
                    )
                });
                self.send_msg(Some(ProcessResult::FAILED_TO_START), process_result_display);
                if let Some(main_panel) = &main_panel {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        Some(ProcessEndState::Failed),
                    );
                }
                return;
            }
        }
        self.send_msg(Some(ProcessResult::SUCCEEDED), process_result_display);
        if let Some(main_panel) = &main_panel {
            main_panel
                .stop_progress_bar_axis_id_process_end_state(axis_id, Some(ProcessEndState::Done));
        }
    }

    /// Java `sirtsetup(AxisID, ProcessResultDisplay, ProcessSeries, DialogType,
    /// ProcessingMethod, SirtsetupDisplay)` (ApplicationManager.java:10314).
    #[allow(clippy::too_many_arguments)]
    pub fn sirtsetup(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: DialogType,
        processing_method: Option<ProcessingMethod>,
        display: &dyn SirtsetupDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // `((TomogramGenerationExpert) getUIExpert(dialogType, axisID)).getTiltDisplay()`.
        let tilt_display = self
            .get_ui_expert(Some(dialog_type), axis_id)
            .and_then(|expert| {
                expert
                    .as_any()
                    .downcast_ref::<TomogramGenerationExpert>()
                    .and_then(|expert| expert.get_tilt_display())
            });
        if self
            .update_tilt_com_tilt_display_axis_id_boolean(tilt_display.as_deref(), axis_id, true)
            .is_none()
        {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let Some(param) = self.update_sirt_setup_com(axis_id, display, true) else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.clone(),
            );
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        // Copy from tilt_for_sirt.com to tilt.com when resuming in case tilt.com was
        // overwritten by Tomo Pos.
        // This can go
        if !param.is_start_from_zero() {
            if self.is_axis_busy(axis_id, process_result_display_ref.clone()) {
                return;
            }
            if let Err(e) = utilities::copy_file_file_types(
                &file_type::CLASS.tilt_for_sirt_comscript,
                &file_type::CLASS.tilt_comscript,
                Some(self),
                Some(axis_id),
                false,
                false,
                false,
            ) {
                eprintln!("{e}");
            }
        }
        let process_track = *self.process_track.lock().unwrap();
        if let Some(process_track) = process_track {
            process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("sirtsetup")),
        };
        let thread_name = match self.get_process_mgr().sirtsetup(
            axis_id,
            process_result_display_ref,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            Arc::new(param),
            processing_method,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", ProcessName::SIRTSETUP)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                self.send_msg(Some(ProcessResult::FAILED_TO_START), process_result_display);
                return;
            }
        };
        process_series.borrow_mut().set_next_process_subprocess(
            Some(ProcessName::PROCESSCHUNKS.to_string().as_str()),
            Some(ProcessName::TILT_SIRT),
            processing_method,
        );
        process_series
            .borrow_mut()
            .set_last_process(Some(tomogram_generation_expert::SIRT_DONE));
        self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::SIRTSETUP).as_str()),
                axis_id,
                Some(&ProcessName::SIRTSETUP),
            );
        }
    }

    /// Java `multifiltSetup(AxisID, ProcessResultDisplay, ProcessSeries, DialogType,
    /// ProcessingMethod, MultifiltSetupDisplay)` (ApplicationManager.java:10381).
    #[allow(clippy::too_many_arguments)]
    pub fn multifilt_setup(
        &'static self,
        axis_id: AxisID,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        dialog_type: DialogType,
        processing_method: Option<ProcessingMethod>,
        display: &dyn MultifiltSetupDisplay,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        let tilt_display = self
            .get_ui_expert(Some(dialog_type), axis_id)
            .and_then(|expert| {
                expert
                    .as_any()
                    .downcast_ref::<TomogramGenerationExpert>()
                    .and_then(|expert| expert.get_tilt_display())
            });
        if self
            .update_tilt_com_tilt_display_axis_id_boolean(tilt_display.as_deref(), axis_id, true)
            .is_none()
        {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.clone(),
            );
            // FIXED (upstream NPE, ApplicationManager.java:10390): Java calls
            // `processSeries.endSeries()` before the `processSeries == null` check
            // below, so a null series throws; `sirtsetup` guards the same call.
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        }
        let Some(param) = self.update_multifilt_setup_com(Some(display), axis_id, true) else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.clone(),
            );
            // FIXED (upstream NPE, ApplicationManager.java:10396), as above.
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return;
        };
        let process_track = *self.process_track.lock().unwrap();
        if let Some(process_track) = process_track {
            process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("multifiltSetup")),
        };
        let thread_name = match self.get_process_mgr().multifilt_setup(
            axis_id,
            process_result_display_ref,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            Arc::new(param),
            processing_method,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", ProcessName::MULTIFILT_SETUP)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                self.send_msg(Some(ProcessResult::FAILED_TO_START), process_result_display);
                return;
            }
        };
        process_series.borrow_mut().set_next_process_subprocess(
            Some(ProcessName::PROCESSCHUNKS.to_string().as_str()),
            Some(ProcessName::TILT_MULTIFILT),
            processing_method,
        );
        self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::MULTIFILT_SETUP).as_str()),
                axis_id,
                Some(&ProcessName::MULTIFILT_SETUP),
            );
        }
    }

    /// Java `ctf3dSetup(ProcessResultDisplay, ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, Ctf3dSetupDisplay, AxisID, DialogType, ProcessingMethod)`
    /// (ApplicationManager.java:10428).
    #[allow(clippy::too_many_arguments)]
    pub fn ctf3d_setup(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        display: &dyn Ctf3dSetupDisplay,
        axis_id: AxisID,
        dialog_type: DialogType,
        processing_method: Option<ProcessingMethod>,
    ) {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, axis_id, Some(dialog_type), Some("ctf3dSetup")),
        };
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // Ctf3dSetup
        let Some(param) = self.update_ctf3d_setup_com(Some(display), axis_id, true) else {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.clone(),
            );
            process_series.borrow().end_series();
            return;
        };

        // Tilt
        let tilt_display = self
            .get_ui_expert(Some(dialog_type), axis_id)
            .and_then(|expert| {
                expert
                    .as_any()
                    .downcast_ref::<TomogramGenerationExpert>()
                    .and_then(|expert| expert.get_tilt_display())
            });
        let tilt_param = self.update_tilt_com_tilt_display_axis_id_boolean(
            tilt_display.as_deref(),
            axis_id,
            true,
        );
        if tilt_param.is_none() {
            self.send_msg(
                Some(ProcessResult::FAILED_TO_START),
                process_result_display.clone(),
            );
            process_series.borrow().end_series();
            return;
        }
        // Load ctfphaseflip command from CtfCorrection comfile.
        if !self
            .get_com_script_manager()
            .load_ctf_correction(axis_id, false)
        {
            self.setup_ctf_correction_com_script(axis_id);
            self.get_com_script_manager()
                .load_ctf_correction(axis_id, true);
        }
        // validations
        // input file
        if !display.is_use_unaligned_images()
            && !file_type::CLASS
                .aligned_stack
                .exists(Some(self), Some(axis_id))
        {
            let message = format!(
                "{} does not exist.",
                file_type::CLASS
                    .aligned_stack
                    .get_file_name(Some(self), Some(axis_id))
                    .unwrap_or_else(|| "null".to_string())
            );
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_ui_component_string_string(
                    Some(self),
                    display.get_use_unaligned_images_ui_component(),
                    &message,
                    "File Not Found",
                )
            });
            return;
        }
        // CTF Correction validation
        // `comScriptMgr.getCtfPhaseFlipParam` always returns a param here, so Java's
        // `ctfPhaseFlipParam != null` test is always true.
        let mut ctf_phase_flip_param = self
            .get_com_script_manager()
            .get_ctf_phase_flip_param(axis_id);
        {
            let x_axis_tilt_string = ctf_phase_flip_param.get_x_axis_tilt();
            let x_axis_tilt = converter::to_double(Some(x_axis_tilt_string.as_str()));
            if let Some(x_axis_tilt) = x_axis_tilt
                && x_axis_tilt != 0.0
            {
                ctf_phase_flip_param.set_x_axis_tilt(Some(x_axis_tilt_string.as_str()));
            }
            // defocusFile
            let defocus_file_parameter = ctf_phase_flip_param.get_defocus_file();
            if !defocus_file_parameter.is_empty() {
                let defocus_file_parameter_string = defocus_file_parameter.to_string();
                // `new File(propertyUserDir, defocusFileParameter.toString())`.
                let defocus_file = match self.get_property_user_dir() {
                    Some(property_user_dir) => PathBuf::from(utilities::java_io_file_new(
                        &property_user_dir,
                        &defocus_file_parameter_string,
                    )),
                    None => PathBuf::from(&defocus_file_parameter_string),
                };
                let file_name = defocus_file
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let simple_defocus_file = file_name.ends_with(dataset_files::SIMPLE_DEFOCUS_EXT);
                let simple_defocus_msg = format!(
                    "It is recommended that you run {} in {} rather then using the {}.",
                    shared_strings::CTF_PLOTTER_LABEL,
                    shared_strings::FINAL_ALIGNED_STACK_LABEL,
                    shared_strings::EXPECTED_DEFOCUS_LABEL
                );
                let ui_component = display.get_ctf_correction_ui_component();
                if !defocus_file.exists() {
                    let errmsg = format!(
                        "Missing defocus file {}.  {}",
                        file_name,
                        if simple_defocus_file {
                            simple_defocus_msg.clone()
                        } else {
                            format!(
                                "Please run {} in {}.",
                                shared_strings::CTF_PLOTTER_LABEL,
                                shared_strings::FINAL_ALIGNED_STACK_LABEL
                            )
                        }
                    );
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_ui_component_string_string(
                            Some(self),
                            ui_component,
                            &errmsg,
                            "File Not Found",
                        )
                    });
                    return;
                } else if simple_defocus_file {
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_ui_component_string_string(
                            Some(self),
                            ui_component,
                            &format!("WARNING:  {simple_defocus_msg}"),
                            "Not Recommended",
                        )
                    });
                }
            }
        }
        // gold erasing
        if display.is_erase_fiducials() {
            let uicomponent = display.get_erase_fiducials_ui_component();
            if !file_type::CLASS
                .gold_eraser_comscript
                .exists(Some(self), Some(axis_id))
            {
                let message = format!(
                    "{} is required for gold erasing.",
                    file_type::CLASS
                        .gold_eraser_comscript
                        .get_file_name(Some(self), Some(axis_id))
                        .unwrap_or_else(|| "null".to_string())
                );
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_ui_component_string_string(
                        Some(self),
                        uicomponent,
                        &message,
                        "Missing File",
                    )
                });
                return;
            }
            // `getCCDEraserParamFromGoldEraser` always returns a param here, so Java's
            // `ccdEraserParam != null` test is always true.
            let ccd_eraser_param = self
                .get_com_script_manager()
                .get_ccd_eraser_param_from_gold_eraser(
                    axis_id,
                    Some(ccd_eraser_param::Mode::Beads),
                );
            let model_file_name = ccd_eraser_param.get_model_file();
            if let Some(model_file_name) = model_file_name {
                let model_file = match self.get_property_user_dir() {
                    Some(property_user_dir) => PathBuf::from(utilities::java_io_file_new(
                        &property_user_dir,
                        &model_file_name,
                    )),
                    None => PathBuf::from(&model_file_name),
                };
                if !model_file.exists() {
                    let message = format!(
                        "{} must either exist or be removed from {}.",
                        model_file_name,
                        ProcessName::GOLD_ERASER.get_comscript(axis_id)
                    );
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_ui_component_string_string(
                            Some(self),
                            uicomponent,
                            &message,
                            "Missing File",
                        )
                    });
                    return;
                }
            }
        }
        if display.is_filter_in_2d() {
            let mtf_filter_param = self.get_com_script_manager().get_mtf_filter_param(axis_id);
            let low_pass_radiu_sigma = mtf_filter_param.is_low_pass_radius_sigma_set();
            let uicomponent = display.get_filter_in_2d_ui_component();
            let mtf_filter_comfile = ProcessName::MTFFILTER.get_comscript(axis_id);
            if !low_pass_radiu_sigma
                && !mtf_filter_param.is_type_of_dose_file_set()
                && !mtf_filter_param.is_fixed_image_dose_set()
            {
                let message = format!(
                    "No dose weighting value or file set in  {}  Please modify {}.",
                    mtf_filter_comfile,
                    shared_strings::FINAL_ALIGNED_STACK_LABEL
                );
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_ui_component_string_string(
                        Some(self),
                        uicomponent.clone(),
                        &message,
                        "Missing parameter(s)",
                    )
                });
                return;
            }
            let dose_weighting_file = mtf_filter_param.get_dose_weighting_file();
            if !dose_weighting_file.is_empty() && !Path::new(&dose_weighting_file).exists() {
                let message = format!(
                    "{} must either exist or be removed from {}.",
                    dose_weighting_file, mtf_filter_comfile
                );
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_ui_component_string_string(
                        Some(self),
                        uicomponent.clone(),
                        &message,
                        "Missing File",
                    )
                });
                return;
            }
            if low_pass_radiu_sigma {
                let message = format!(
                    "WARNING:  It is recommended that you use  {}.  See {} in {}.",
                    shared_strings::DOSE_WEIGHTING_LABEL,
                    shared_strings::_2D_FILTER_LABEL,
                    shared_strings::FINAL_ALIGNED_STACK_LABEL
                );
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_ui_component_string_string(
                        Some(self),
                        uicomponent.clone(),
                        &message,
                        "Not Recommended",
                    )
                });
            }
            let process_track = *self.process_track.lock().unwrap();
            if let Some(process_track) = process_track {
                process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
            }
        }
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        let thread_name = match self.get_process_mgr().ctf3d_setup(
            axis_id,
            process_result_display_ref,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            Arc::new(param),
            processing_method,
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", ProcessName::CTF_3D_SETUP)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                self.send_msg(Some(ProcessResult::FAILED_TO_START), process_result_display);
                return;
            }
        };
        process_series.borrow_mut().set_next_process_subprocess(
            Some(ProcessName::PROCESSCHUNKS.to_string().as_str()),
            Some(ProcessName::CTF_3D),
            processing_method,
        );

        self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::CTF_3D_SETUP).as_str()),
                axis_id,
                Some(&ProcessName::CTF_3D_SETUP),
            );
        }
    }

    /// Java `subtomoSetup(AxisID, DialogType, ProcessingMethod, SubtomoSetupDisplay)`
    /// (ApplicationManager.java:10586).
    pub fn subtomo_setup(
        &'static self,
        axis_id: AxisID,
        dialog_type: DialogType,
        processing_method: Option<ProcessingMethod>,
        display: &dyn SubtomoSetupDisplay,
    ) {
        let Some(param) = self.update_subtomo_setup_com(Some(display), axis_id, true) else {
            return;
        };
        let process_track = *self.process_track.lock().unwrap();
        if let Some(process_track) = process_track {
            process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
        }
        let main_panel = self.main_panel.get();
        if let Some(main_panel) = &main_panel {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        let process_series =
            ProcessSeries::new(self, axis_id, Some(dialog_type), Some("subtomoSetup"));
        if let Some(main_panel) = &main_panel {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::SUBTOMO_SETUP).as_str()),
                axis_id,
                Some(&ProcessName::SUBTOMO_SETUP),
            );
        }

        let thread_name: String;
        let process_end_state: Option<ProcessEndState> = None;

        // try {
        let mut processchunks_param =
            ProcesschunksParam::get_instance(self, axis_id, Some("subtomo_coms/tilt-sub"), None);
        let parallel_panel = main_panel
            .as_ref()
            .and_then(|main_panel| main_panel.get_parallel_panel(AxisID::Only));
        let Some(parallel_panel) = parallel_panel else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    Some(axis_id),
                )
            });
            process_series.borrow().end_series();
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    Some(ProcessEndState::Failed),
                );
            }
            return;
        };
        let gpu_found: bool;
        if parallel_panel.get_parameters_processchunks_param_boolean(&mut processchunks_param, true)
        {
            if param.get_when_to_use_gpu() == subtomo_setup_param::WHEN_TO_USE_GPU_VAL_2 {
                gpu_found = processchunks_param.reorder_computer_map_gpu_first();

                if !gpu_found {
                    let no_gpu_error_msg = "None of the selected machines has a GPU for doing CTF correction. To use this option, please choose at least one machine that has a GPU.";
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self),
                            no_gpu_error_msg,
                            "No GPU selected",
                            Some(axis_id),
                        )
                    });
                    process_series.borrow().end_series();
                    if let Some(main_panel) = &main_panel {
                        main_panel.stop_progress_bar_axis_id_process_end_state(
                            axis_id,
                            Some(ProcessEndState::Failed),
                        );
                    }
                    return;
                }
            }
        } else {
            process_series.borrow().end_series();
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    Some(ProcessEndState::Failed),
                );
            }
            return;
        }

        // `if (processchunksParam != null)`: `getInstance` never returns null.
        process_series.borrow_mut().set_next_process_task_command(
            Rc::new(Task::Processchunks),
            Arc::new(processchunks_param),
        );
        match self.get_process_mgr().subtomo_setup(
            axis_id,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            Arc::new(param),
            processing_method,
        ) {
            Ok(name) => {
                thread_name = name;
                // `if (threadName != null)`: a started process always has a name.
                self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
            }
            // } catch (final AxisBusyException e) {
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", ProcessName::SUBTOMO_SETUP)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                if let Some(main_panel) = &main_panel {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        Some(ProcessEndState::Failed),
                    );
                }
                return;
            }
        }

        if process_end_state == Some(ProcessEndState::Failed) {
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    Some(ProcessEndState::Failed),
                );
            }
            process_series.borrow().end_series();
        }
    }

    /// Java `altTomoSetup(AxisID, DialogType, ProcessingMethod, AltStackDisplay)`
    /// (ApplicationManager.java:10665).
    pub fn alt_tomo_setup(
        &'static self,
        axis_id: AxisID,
        dialog_type: DialogType,
        processing_method: Option<ProcessingMethod>,
        display: &dyn AltStackDisplay,
    ) {
        let Some(mut param) = self.update_alt_tomo_setup_com(Some(display), axis_id, true) else {
            return;
        };
        let process_track = *self.process_track.lock().unwrap();
        if let Some(process_track) = process_track {
            process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
        }
        let main_panel = self.main_panel.get();
        if let Some(main_panel) = &main_panel {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }

        if let Some(main_panel) = &main_panel {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::ALT_TOMO_SETUP).as_str()),
                axis_id,
                Some(&ProcessName::ALT_TOMO_SETUP),
            );
        }

        let process_series =
            ProcessSeries::new(self, axis_id, Some(dialog_type), Some("altTomoSetup"));
        let mut processchunks_param = ProcesschunksParam::get_instance_process_name(
            self,
            axis_id,
            ProcessName::ALT_TOMO_PROCESS_CHUNKS,
            None,
        );
        let parallel_panel = main_panel
            .as_ref()
            .and_then(|main_panel| main_panel.get_parallel_panel(axis_id));
        let Some(parallel_panel) = parallel_panel else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    Some(axis_id),
                )
            });
            process_series.borrow().end_series();
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    Some(ProcessEndState::Failed),
                );
            }
            return;
        };
        // `processchunksParam != null` is always true: `getInstance` never returns null.
        if parallel_panel.get_parameters_processchunks_param_boolean(&mut processchunks_param, true)
        {
            process_series.borrow_mut().set_next_process_task_command(
                Rc::new(Task::Processchunks),
                Arc::new(processchunks_param),
            );
        } else {
            process_series.borrow().end_series();
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    Some(ProcessEndState::Failed),
                );
            }
            return;
        }

        self.close_alt_tomo_setup_imod_files(&param, axis_id);
        // try {
        parallel_panel.get_parameters_alt_tomo_setup_param_boolean(&mut param, true);
        let thread_name = match self.get_process_mgr().alt_tomo_setup(
            AxisID::Only,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
            Arc::new(param),
            processing_method,
        ) {
            Ok(thread_name) => thread_name,
            // } catch (final AxisBusyException e) {
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", ProcessName::ALT_TOMO_SETUP)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                if let Some(main_panel) = &main_panel {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        Some(ProcessEndState::Failed),
                    );
                }
                return;
            }
        };

        self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
    }

    /// Java `altTomoSetupRestoreSwappedFiles(AxisID, DialogType, ProcessingMethod,
    /// AltStackDisplay)` (ApplicationManager.java:10723).
    pub fn alt_tomo_setup_restore_swapped_files(
        &'static self,
        axis_id: AxisID,
        dialog_type: DialogType,
        processing_method: Option<ProcessingMethod>,
        display: &dyn AltStackDisplay,
    ) {
        let Some(mut param) =
            self.update_alt_tomo_setup_com_restore_swapped_files(Some(display), axis_id, true)
        else {
            return;
        };

        let process_track = *self.process_track.lock().unwrap();
        if let Some(process_track) = process_track {
            process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
        }
        let main_panel = self.main_panel.get();
        if let Some(main_panel) = &main_panel {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }

        if let Some(main_panel) = &main_panel {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", ProcessName::ALT_TOMO_SETUP).as_str()),
                axis_id,
                Some(&ProcessName::ALT_TOMO_SETUP),
            );
        }

        let parallel_panel = main_panel
            .as_ref()
            .and_then(|main_panel| main_panel.get_parallel_panel(axis_id));
        let Some(parallel_panel) = parallel_panel else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    Some(axis_id),
                )
            });
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    axis_id,
                    Some(ProcessEndState::Failed),
                );
            }
            return;
        };
        // try {
        parallel_panel.get_parameters_alt_tomo_setup_param_boolean(&mut param, true);
        let thread_name = match self.get_process_mgr().alt_tomo_setup(
            AxisID::Only,
            None,
            Arc::new(param),
            processing_method,
        ) {
            Ok(thread_name) => thread_name,
            // } catch (final AxisBusyException e) {
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!("Can not execute {}", ProcessName::ALT_TOMO_SETUP)),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(axis_id),
                    )
                });
                if let Some(main_panel) = &main_panel {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        Some(ProcessEndState::Failed),
                    );
                }
                return;
            }
        };

        self.set_thread_name(Some(thread_name.as_str()), Some(axis_id));
    }

    /// Java package-private `closeAltTomoSetupImodFiles(AltTomoSetupParam, AxisID)`
    /// (ApplicationManager.java:10766).
    pub fn close_alt_tomo_setup_imod_files(
        &'static self,
        param: &AltTomoSetupParam,
        axis_id: AxisID,
    ) {
        if self.is_dual_axis() {
            let is_axis_to_process = param.is_axis_to_process();
            let param_axis_to_process = param.get_axis_to_process();
            // FIXED (upstream bug, ApplicationManager.java:10770 and :10777): Java
            // compares `paramAxisToProcess == "A"` / `== "B"` by reference.  The value
            // comes from `StringParameter.toString()`, never the interned literal, so
            // the test is always false and only the `!isAxisToProcess` arm could close
            // anything.  Compared by value here, as the code evidently intends.
            if !is_axis_to_process || param_axis_to_process == "A" {
                self.close_imod_key_file(
                    Some(imod_manager::ALT_TOMO_SETUP_TOMOGRAM_KEY),
                    file_type::CLASS
                        .alt_stack_tomogram
                        .get_file_with_property_user_dir(
                            Some(self),
                            Some(param.get_rootname_to_process().as_str()),
                            Some(AxisType::DualAxis),
                            Some(AxisID::First),
                            self.get_property_user_dir().as_deref(),
                        )
                        .as_deref(),
                    Some(AxisID::First),
                    file_type::CLASS
                        .alt_stack_tomogram
                        .get_description()
                        .as_deref(),
                    true,
                );
            }
            if !is_axis_to_process || param_axis_to_process == "B" {
                self.close_imod_key_file(
                    Some(imod_manager::ALT_TOMO_SETUP_TOMOGRAM_KEY),
                    file_type::CLASS
                        .alt_stack_tomogram
                        .get_file_with_property_user_dir(
                            Some(self),
                            Some(param.get_rootname_to_process().as_str()),
                            Some(AxisType::DualAxis),
                            Some(AxisID::Second),
                            self.get_property_user_dir().as_deref(),
                        )
                        .as_deref(),
                    Some(AxisID::Second),
                    file_type::CLASS
                        .alt_stack_tomogram
                        .get_description()
                        .as_deref(),
                    true,
                );
            }
        } else {
            let is_alt_tomo_trim_vol_checked = param.is_trim_volume();
            if self.get_state().is_alt_tomo_even_and_odd_pairs() {
                if !is_alt_tomo_trim_vol_checked {
                    self.close_imod_file_key(
                        Some(&file_type::CLASS.alt_stack_even_full_tomogram),
                        Some(axis_id),
                        true,
                    );
                    self.close_imod_file_key(
                        Some(&file_type::CLASS.alt_stack_odd_full_tomogram),
                        Some(axis_id),
                        true,
                    );
                } else {
                    self.close_imod_file_key(
                        Some(&file_type::CLASS.alt_stack_even_tomogram),
                        Some(axis_id),
                        true,
                    );
                    self.close_imod_file_key(
                        Some(&file_type::CLASS.alt_stack_odd_tomogram),
                        Some(axis_id),
                        true,
                    );
                }
            } else if !is_alt_tomo_trim_vol_checked {
                self.close_imod_key_file(
                    Some(imod_manager::ALT_TOMO_SETUP_TOMOGRAM_KEY),
                    file_type::CLASS
                        .tilt_output_single
                        .get_file_with_property_user_dir(
                            Some(self),
                            Some(param.get_rootname_to_process().as_str()),
                            Some(AxisType::SingleAxis),
                            Some(axis_id),
                            self.get_property_user_dir().as_deref(),
                        )
                        .as_deref(),
                    Some(axis_id),
                    file_type::CLASS
                        .tilt_output_single
                        .get_description()
                        .as_deref(),
                    true,
                );
            } else {
                self.close_imod_key_file(
                    Some(imod_manager::ALT_TOMO_SETUP_TOMOGRAM_KEY),
                    file_type::CLASS
                        .alt_stack_tomogram
                        .get_file_with_property_user_dir(
                            Some(self),
                            Some(param.get_rootname_to_process().as_str()),
                            Some(AxisType::SingleAxis),
                            Some(axis_id),
                            self.get_property_user_dir().as_deref(),
                        )
                        .as_deref(),
                    Some(axis_id),
                    file_type::CLASS
                        .alt_stack_tomogram
                        .get_description()
                        .as_deref(),
                    true,
                );
            }
        }
    }

    /// Java `getAltTomoSetupLogMessage()` (ApplicationManager.java:10812).  Called on a
    /// process thread; reads only `TomogramState`.
    pub fn get_alt_tomo_setup_log_message(&self) -> String {
        let state = self.get_state();
        let mut log_message = String::new();
        if state.is_alt_tomo_even_and_odd_pairs() {
            log_message += "Even and odd pairs processed with ";
        } else {
            log_message += &format!(
                "Stack {} processed with ",
                state.get_alt_tomo_rootname_to_process()
            );
        }
        let mut preprocessing_steps: Vec<String> = Vec::new();
        if state.is_alt_tomo_preprocess_for_extremes() {
            preprocessing_steps.push("preprocessing".to_string());
        }
        if state.is_alt_tomo_correct_ctf() {
            preprocessing_steps.push("CTF correction".to_string());
        }
        if state.is_alt_tomo_erase_fiducials() {
            preprocessing_steps.push("gold erasing".to_string());
        }
        if state.is_alt_tomo_filter_in_2d() {
            preprocessing_steps.push("2D filtering".to_string());
        }
        if state.is_alt_tomo_trim_vol_checked() {
            preprocessing_steps.push("trimming".to_string());
        }

        if preprocessing_steps.is_empty() {
            log_message += "no pre-processing steps to run.";
        } else {
            // `ArrayList.toString()`: "[a, b, c]".
            let str_steps_to_run = format!("[{}]", preprocessing_steps.join(", "));
            log_message += &str_steps_to_run[1..str_steps_to_run.len() - 1];
            log_message += ".";
        }

        log_message
    }

    /// Java `openCtf3d(AxisID, Run3dmodMenuOptions)` (ApplicationManager.java:10849).
    pub fn open_ctf3d(
        &'static self,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        match self
            .get_imod_manager()
            .open_string_axis_id_run3dmod_menu_options(
                imod_manager::CTF_3D_KEY,
                Some(axis_id),
                run_3dmod_menu_options,
            ) {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "Can't open 3dmod with this tomogram",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `openAlternativeTomogram(File, AxisID, AxisType, Run3dmodMenuOptions,
    /// String, String, boolean)` (ApplicationManager.java:10869).  `axisType` and
    /// `rootname` are unused in the Java body.
    #[allow(clippy::too_many_arguments)]
    pub fn open_alternative_tomogram(
        &'static self,
        file: &Path,
        axis_id: AxisID,
        axis_type: AxisType,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        key: &str,
        rootname: Option<&str>,
        is_alt_tomo_trim_vol_checked: bool,
    ) {
        let _ = (axis_type, rootname);
        let result = self
            .get_imod_manager()
            .set_swap_yz_string_axis_id_file_boolean(
                key,
                Some(axis_id),
                Some(file),
                !is_alt_tomo_trim_vol_checked || !self.is_trimvol_flipped(),
            )
            .and_then(|()| {
                self.get_imod_manager()
                    .open_string_axis_id_file_run3dmod_menu_options(
                        key,
                        Some(axis_id),
                        Some(file),
                        run_3dmod_menu_options,
                    )
            });
        match result {
            Ok(()) => {}
            Err(ImodManagerException::AxisType(except)) => {
                eprintln!("{except}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "AxisType problem",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::SystemProcess(except)) => {
                eprintln!("{except}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "Can't open 3dmod with this tomogram",
                        Some(axis_id),
                    )
                });
            }
            Err(ImodManagerException::Io(e)) => {
                eprintln!("{e}");
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &e.to_string(),
                        "IO Exception",
                        Some(axis_id),
                    )
                });
            }
            // An uncaught RuntimeException from BaseImodManager: Swing's handler prints it.
            Err(ImodManagerException::Runtime(message)) => {
                eprintln!("{message}");
            }
        }
    }

    /// Java `useCtf3d(ProcessResultDisplay, String, AxisID, DialogType)`
    /// (ApplicationManager.java:10892).
    pub fn use_ctf3d(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        run_button_label: &str,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> bool {
        let output_file_type = &file_type::CLASS.tilt_output;
        self.use_image_file(
            process_result_display,
            &file_type::CLASS.ctf_3d_output,
            None,
            output_file_type,
            run_button_label,
            axis_id,
            dialog_type,
        )
    }

    /// Java `resume(AxisID, ProcesschunksParam, ProcessResultDisplay, ProcessSeries,
    /// CommandDetails, boolean, ProcessingMethod)` (ApplicationManager.java:10909).
    /// Override resume to add a last process for tilt_sirt.
    #[allow(clippy::too_many_arguments)]
    pub fn resume(
        &'static self,
        axis_id: AxisID,
        param: Option<Arc<ProcesschunksParam>>,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        subcommand_details: Option<Arc<dyn CommandDetails + Send + Sync>>,
        popup_chunk_warnings: bool,
        processing_method: Option<ProcessingMethod>,
    ) {
        let mut process_series = process_series;
        let mut root_name: Option<String> = None;
        if let Some(param) = &param {
            root_name = param.get_root_name();
        }
        if root_name.is_none() {
            root_name = self
                .get_meta_data()
                .base()
                .get_current_processchunks_root_name(Some(axis_id));
        }
        // FIXED (upstream NPE, ApplicationManager.java:10919): Java calls
        // `rootName.indexOf` on a null root name (no param root name and none saved in
        // the metadata); here a null root name does not contain "sirt".
        if root_name
            .as_deref()
            .is_some_and(|root_name| root_name.contains("sirt"))
            && process_series.is_none()
        {
            let new_series = ProcessSeries::new(
                self,
                axis_id,
                Some(DialogType::TomogramGeneration),
                Some("resume"),
            );
            new_series
                .borrow_mut()
                .set_last_process(Some(tomogram_generation_expert::SIRT_DONE));
            process_series = Some(new_series);
        }
        BaseManager::resume(
            self,
            Some(axis_id),
            param,
            process_result_display,
            process_series,
            subcommand_details,
            popup_chunk_warnings,
            processing_method,
            false,
            Some(DialogType::TomogramGeneration),
        );
    }

    /// Java `useSirt(ProcessResultDisplay, File, String, AxisID, DialogType)`
    /// (ApplicationManager.java:10931).
    pub fn use_sirt(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        use_file: Option<PathBuf>,
        run_button_label: &str,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> bool {
        let output_file_type = &file_type::CLASS.tilt_output;
        self.use_image_file(
            process_result_display,
            &file_type::CLASS.sirt_output_template,
            use_file,
            output_file_type,
            run_button_label,
            axis_id,
            dialog_type,
        )
    }

    /// Java `useImageFile(ProcessResultDisplay, FileType, File, FileType, String, AxisID,
    /// DialogType)` (ApplicationManager.java:10945).  Replace the full aligned stack
    /// with the output of CTF correction, bead erase, or SIRT; returns true if
    /// succeeded.
    #[allow(clippy::too_many_arguments)]
    pub fn use_image_file(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        use_file_type: &Arc<FileType>,
        use_file: Option<PathBuf>,
        output_file_type: &Arc<FileType>,
        run_button_label: &str,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> bool {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        let mut use_file = use_file;
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        if self.is_axis_busy(axis_id, process_result_display_ref.clone()) {
            return false;
        }
        let mut rename_with_file_type = false;
        if use_file.is_none() {
            rename_with_file_type = true;
            use_file = use_file_type.get_file(Some(self), Some(axis_id));
        }
        let Some(use_file) = use_file else {
            self.send_msg(Some(ProcessResult::FAILED_TO_START), process_result_display);
            return false;
        };
        let use_file_name = use_file
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let main_panel = self.main_panel.get();
        if let Some(main_panel) = &main_panel {
            main_panel.start_progress_bar_string_axis_id(
                Some(
                    &format!(
                        "Using {} as {}",
                        use_file_name,
                        output_file_type
                            .get_description()
                            .unwrap_or_else(|| "null".to_string())
                    )
                    .as_str(),
                ),
                axis_id,
            );
        }
        if !use_file.exists() {
            let message = format!(
                "{use_file_name} doesn't exist.  Press {run_button_label} to create this file."
            );
            let title = format!("{run_button_label} Output Missing");
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &message,
                    &title,
                    Some(axis_id),
                )
            });
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id(axis_id);
            }
            self.send_msg(Some(ProcessResult::FAILED_TO_START), process_result_display);
            return false;
        }
        let process_track = *self.process_track.lock().unwrap();
        if let Some(process_track) = process_track {
            process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
        }
        if let Some(main_panel) = &main_panel {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        // Java dereferences `outputFileType.getFile(this, axisID)` without a null check;
        // a null file is taken as one that does not exist.
        if output_file_type
            .get_file(Some(self), Some(axis_id))
            .is_some_and(|file| file.exists())
            && use_file.exists()
        {
            if !utilities::is_valid_stack_file(use_file.as_path(), self, Some(axis_id)) {
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &format!("{use_file_name} is not a valid MRC file."),
                        "Entry Error",
                        Some(axis_id),
                    )
                });
                if let Some(main_panel) = &main_panel {
                    main_panel.stop_progress_bar_axis_id(axis_id);
                }
                self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
                return false;
            }
            match self.backup_image_file(Some(output_file_type), Some(axis_id)) {
                Ok(()) => {}
                Err(LogFileError::Lock(_)) => {
                    if let Some(main_panel) = &main_panel {
                        main_panel.stop_progress_bar_axis_id_process_end_state(
                            axis_id,
                            Some(ProcessEndState::FileLockFailure),
                        );
                    }
                    self.send_msg(Some(ProcessResult::FAILED), process_result_display);
                    return false;
                }
                Err(except) => {
                    let message = format!(
                        "Unable to backup {}\n{}",
                        output_file_type
                            .get_file_name(Some(self), Some(axis_id))
                            .unwrap_or_else(|| "null".to_string()),
                        except
                    );
                    ui_harness::INSTANCE.with(|ui_harness| {
                        ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self),
                            &message,
                            "File Rename Error (7)",
                            Some(axis_id),
                        )
                    });
                    if let Some(main_panel) = &main_panel {
                        main_panel.stop_progress_bar_axis_id(axis_id);
                    }
                    self.send_msg(Some(ProcessResult::FAILED), process_result_display);
                    return false;
                }
            }
        }
        let rename_result = if rename_with_file_type {
            self.rename_image_file(Some(use_file_type), Some(output_file_type), Some(axis_id))
        } else {
            self.rename_image_file_from_key(
                Some(use_file_type),
                Some(use_file.as_path()),
                Some(output_file_type),
                Some(axis_id),
                false,
            )
        };
        match rename_result {
            Ok(()) => {}
            Err(LogFileError::Lock(_)) => {
                if let Some(main_panel) = &main_panel {
                    main_panel.stop_progress_bar_axis_id_process_end_state(
                        axis_id,
                        Some(ProcessEndState::FileLockFailure),
                    );
                }
                self.send_msg(Some(ProcessResult::FAILED), process_result_display);
                return false;
            }
            Err(except) => {
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self),
                        &except.to_string(),
                        "File Rename Error (8)",
                        Some(axis_id),
                    )
                });
                if let Some(main_panel) = &main_panel {
                    main_panel.stop_progress_bar_axis_id(axis_id);
                }
                self.send_msg(Some(ProcessResult::FAILED), process_result_display);
                return false;
            }
        }
        if let Some(main_panel) = &main_panel {
            main_panel.stop_progress_bar_axis_id(axis_id);
        }
        self.send_msg(Some(ProcessResult::SUCCEEDED), process_result_display);
        true
    }

    /// Java `splitcombine(ProcessSeries, Deferred3dmodButton, Run3dmodMenuOptions,
    /// DialogType, ProcessingMethod, boolean, boolean)` (ApplicationManager.java:11025).
    #[allow(clippy::too_many_arguments)]
    pub fn splitcombine_process_series_deferred3dmod_button_run3dmod_menu_options_dialog_type_processing_method_boolean_boolean(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
        volcombine_processing_method: Option<ProcessingMethod>,
        parallel_process: bool,
        no_vol_combine: bool,
    ) {
        // Check for CPU's selected if parallel processing is enabled
        if !self.check_cpus_if_parallel_process_set(parallel_process, no_vol_combine) {
            return;
        }
        self.splitcombine_process_series_deferred3dmod_button_run3dmod_menu_options_dialog_type_processing_method(
            process_series,
            deferred_3dmod_button,
            run_3dmod_menu_options,
            dialog_type,
            volcombine_processing_method,
        );
    }

    /// Java private `splitcombine(ProcessSeries, Deferred3dmodButton,
    /// Run3dmodMenuOptions, DialogType, ProcessingMethod)`
    /// (ApplicationManager.java:11038).
    fn splitcombine_process_series_deferred3dmod_button_run3dmod_menu_options_dialog_type_processing_method(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: Option<DialogType>,
        volcombine_processing_method: Option<ProcessingMethod>,
    ) {
        let process_series = match process_series {
            Some(process_series) => process_series,
            None => ProcessSeries::new(self, AxisID::First, dialog_type, Some("splitcombine")),
        };

        let process_result_display: Option<ProcessResultDisplayHandle> = Some(
            self.get_process_result_display_factory(AxisID::Only)
                .get_restart_volcombine(),
        );
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        if !self.update_volcombine_com(true) {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            process_series.borrow().end_series();
            return;
        }
        self.get_recon_process_track()
            .set_tomogram_combination_state(ProcessState::InProgress);
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.set_tomogram_combination_state(ProcessState::InProgress);
        }
        process_series
            .borrow_mut()
            .set_run_3dmod_deferred(deferred_3dmod_button, run_3dmod_menu_options);
        let thread_name = match self.get_process_mgr().splitcombine(
            process_result_display_ref,
            Some(Arc::new(EdtRef::new(Rc::clone(&process_series)))),
        ) {
            Ok(thread_name) => thread_name,
            Err(e) => {
                eprintln!("{e}");
                let message: [Option<String>; 2] = [
                    Some(format!(
                        "Can not execute {}",
                        splitcombine_param::COMMAND_NAME
                    )),
                    Some(e.0.clone()),
                ];
                ui_harness::INSTANCE.with(|ui_harness| {
                    ui_harness.open_message_dialog_base_manager_string_array_string_axis_id(
                        Some(self),
                        &message
                            .iter()
                            .map(|line| line.clone().unwrap_or_else(|| "null".to_string()))
                            .collect::<Vec<String>>(),
                        "Unable to execute command",
                        Some(AxisID::Only),
                    )
                });
                return;
            }
        };
        process_series.borrow_mut().set_next_process_subprocess(
            Some(ProcessName::PROCESSCHUNKS.to_string().as_str()),
            Some(ProcessName::VOLCOMBINE),
            volcombine_processing_method,
        );
        self.set_thread_name(Some(thread_name.as_str()), Some(AxisID::Only));
        if let Some(main_panel) = self.main_panel.get() {
            main_panel.start_progress_bar_string_axis_id_process_name(
                Some(&format!("Running {}", splitcombine_param::COMMAND_NAME).as_str()),
                AxisID::Only,
                Some(&ProcessName::SPLITCOMBINE),
            );
        }
    }

    /// Java private `processchunksVolcombine(ProcessResultDisplay, ProcessSeries,
    /// ProcessingMethod)` (ApplicationManager.java:11077).
    fn processchunks_volcombine(
        &'static self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        processing_method: Option<ProcessingMethod>,
    ) -> bool {
        // CloseImod, including the one for processchunks volcombine is take care
        // when combine.com is updated.
        let dialog: Option<Rc<dyn AbstractParallelDialog>> = self
            .tomogram_combination_dialog
            .get()
            .map(|dialog| dialog as Rc<dyn AbstractParallelDialog>);
        self.processchunks_axis_id_dialog_type_abstract_parallel_dialog_process_result_display_process_series_process_name_file_type_processing_method(
            AxisID::Only,
            DialogType::TomogramCombination,
            dialog,
            process_result_display,
            process_series,
            ProcessName::VOLCOMBINE,
            &file_type::CLASS.combined_volume,
            processing_method,
        )
    }

    /// Java private `processchunks(AxisID, DialogType, AbstractParallelDialog,
    /// ProcessResultDisplay, ProcessSeries, ProcessName, FileType, ProcessingMethod)`
    /// (ApplicationManager.java:11091).  Run processchunks.
    #[allow(clippy::too_many_arguments)]
    fn processchunks_axis_id_dialog_type_abstract_parallel_dialog_process_result_display_process_series_process_name_file_type_processing_method(
        &'static self,
        axis_id: AxisID,
        dialog_type: DialogType,
        dialog: Option<Rc<dyn AbstractParallelDialog>>,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        process_name: ProcessName,
        output_image_file_type: &Arc<FileType>,
        processing_method: Option<ProcessingMethod>,
    ) -> bool {
        let process_result_display_ref: Option<ProcessResultDisplayRef> = process_result_display
            .clone()
            .map(|display| Arc::new(EdtRef::new(display)));
        self.send_msg_process_starting(process_result_display_ref.as_ref());
        // FIXED (upstream NPE, ApplicationManager.java:11097/:11110/:11117): Java calls
        // `processSeries.endSeries()` on a null series; each call is guarded here.
        let Some(dialog) = dialog else {
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return false;
        };
        let mut param = ProcesschunksParam::get_instance_process_name(
            self,
            axis_id,
            process_name,
            Some(OutputImageFileKey::FileType(Arc::clone(
                output_image_file_type,
            ))),
        );
        let main_panel = self.main_panel.get();
        let parallel_panel = main_panel
            .as_ref()
            .and_then(|main_panel| main_panel.get_parallel_panel(axis_id));
        let Some(parallel_panel) = parallel_panel else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    Some(axis_id),
                )
            });
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    AxisID::Only,
                    Some(ProcessEndState::Failed),
                );
            }
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return false;
        };
        dialog.get_parameters(&mut param);
        if !parallel_panel.get_parameters_processchunks_param_boolean(&mut param, true) {
            if let Some(main_panel) = &main_panel {
                main_panel.stop_progress_bar_axis_id_process_end_state(
                    AxisID::Only,
                    Some(ProcessEndState::Failed),
                );
            }
            self.send_msg_process_failed_to_start(process_result_display_ref.as_ref());
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return false;
        }
        let process_track = *self.process_track.lock().unwrap();
        if let Some(process_track) = process_track {
            process_track.set_state_dialog_type(ProcessState::InProgress, axis_id, dialog_type);
        }
        if let Some(main_panel) = &main_panel {
            main_panel.set_state_process_state_axis_id_dialog_type(
                ProcessState::InProgress,
                axis_id,
                dialog_type,
            );
        }
        // param should never be set to resume
        parallel_panel.reset_results();
        BaseManager::processchunks(
            self,
            Some(axis_id),
            Some(Arc::new(param)),
            process_result_display,
            process_series,
            true,
            processing_method,
            false,
            Some(dialog_type),
            None,
            None,
            None,
        )
    }

    /// Java package-private `processchunks(ProcessSeries, Command, AxisID)`
    /// (ApplicationManager.java:11130).
    pub fn processchunks_process_series_command_axis_id(
        &'static self,
        process_series: Option<ProcessSeriesHandle>,
        command: Option<Arc<dyn Command + Send + Sync>>,
        axis_id: AxisID,
    ) -> bool {
        // `command instanceof ProcesschunksParam` (false for null).
        let param: Option<Arc<ProcesschunksParam>> = command.and_then(|command| {
            (command as Arc<dyn std::any::Any + Send + Sync>)
                .downcast::<ProcesschunksParam>()
                .ok()
        });
        let Some(param) = param else {
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return false;
        };
        let parallel_panel = self
            .main_panel
            .get()
            .and_then(|main_panel| main_panel.get_parallel_panel(axis_id));
        let Some(parallel_panel) = parallel_panel else {
            ui_harness::INSTANCE.with(|ui_harness| {
                ui_harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self),
                    &shared_strings::PARALLEL_PROCESSING_REQUIRED_MESSAGE,
                    "Unable to execute command",
                    Some(axis_id),
                )
            });
            if let Some(process_series) = &process_series {
                process_series.borrow().end_series();
            }
            return false;
        };
        parallel_panel.reset_results();
        BaseManager::processchunks(
            self,
            Some(axis_id),
            Some(param),
            None,
            process_series,
            true,
            None,
            false,
            None,
            None,
            None,
            None,
        );
        true
    }

    /// Java `getScreenState(AxisID)` (ApplicationManager.java:11180).
    // REPLACES existing
    pub fn get_screen_state(&self, axis_id: AxisID) -> &'static ReconScreenState {
        let slot = if axis_id == AxisID::Second {
            &self.screen_state_b
        } else {
            &self.screen_state_a
        };
        let mut slot = slot.lock().unwrap();
        if slot.is_none() {
            let screen_state: &'static ReconScreenState = Box::leak(Box::new(
                ReconScreenState::new(axis_id, self.get_meta_data().base().get_axis_type()),
            ));
            ROOTS.lock().unwrap().screen_state.push(screen_state);
            *slot = Some(screen_state);
        }
        slot.unwrap()
    }
}

// ---- end impl BaseManager for ApplicationManager ----

/// Java `public static final class ApplicationManager.Task implements TaskInterface`
/// (ApplicationManager.java:11220).  Its ten instances are private static finals,
/// all constructed with `Task(String)`, i.e. `droppable = false`; translated as an
/// enum (the same shape as `base_manager::Task`).  `ProcessSeries.Process.equals(Task)`
/// compares by concrete type and description (`process_series.rs`), which identifies
/// these variants exactly as Java's reference comparison does.
///
/// Module-level item: the integrator places it at the top level of
/// application_manager.rs (as `pub enum Task`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    /// Java `RELOAD_ALIGN_COM`.
    ReloadAlignCom,
    /// Java `COPY_TOMO_COMS`.
    CopyTomoComs,
    /// Java `EXCLUDE_VIEWS_A`.
    ExcludeViewsA,
    /// Java `EXCLUDE_VIEWS_B`.
    ExcludeViewsB,
    /// Java `SETUP_RECON_FAILED`.
    SetupReconFailed,
    /// Java `VALIDATE_DIRECTIVE_FILES`.
    ValidateDirectiveFiles,
    /// Java `SET_RAW_STACK_EXTENSION`.
    SetRawStackExtension,
    /// Java `SET_ORIG_RAW_STACK_EXTENSION`.
    SetOrigRawStackExtension,
    /// Java `RENAME_INPUT_IMAGE_FILES`.
    RenameInputImageFiles,
    /// Java `PROCESSCHUNKS`.
    Processchunks,
}

impl Task {
    /// Java field `descr` (the constructor argument).
    fn descr(self) -> &'static str {
        match self {
            Task::ReloadAlignCom => "RELOAD_ALIGN_COM",
            Task::CopyTomoComs => "COPY_TOMO_COMS",
            Task::ExcludeViewsA => "EXCLUDE_VIEWS_A",
            Task::ExcludeViewsB => "EXCLUDE_VIEWS_B",
            Task::SetupReconFailed => "SETUP_RECON_FAILED",
            Task::ValidateDirectiveFiles => "VALIDATE_DIRECTIVE_FILES",
            Task::SetRawStackExtension => "SET_RAW_STACK_EXTENSION",
            Task::SetOrigRawStackExtension => "SET_ORIG_RAW_STACK_EXTENSION",
            Task::RenameInputImageFiles => "RENAME_INPUT_IMAGE_FILES",
            Task::Processchunks => "PROCESSCHUNKS",
        }
    }

    /// Java field `droppable`: every instance is built with `Task(String)`, which
    /// passes `false`.
    fn droppable(self) -> bool {
        false
    }
}

impl std::fmt::Display for Task {
    /// Java `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.descr())
    }
}

impl TaskInterface for Task {
    /// Java `okToDrop()`.
    fn ok_to_drop(&self) -> bool {
        self.droppable()
    }

    /// Java `getDescr()`.
    fn get_descr(&self) -> Option<String> {
        Some(self.descr().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::util::event_queue::invoke_and_wait;

    #[test]
    fn advanced_state_preserves_source_axis_update_rules() {
        // Managers, their dialogs and panels live on the event dispatch thread.
        invoke_and_wait(|| {
            let manager = ApplicationManager::new(None, Some(AxisID::Only));
            manager.set_advanced_dialog_type_axis_id_boolean(
                DialogType::SetupRecon,
                AxisID::Second,
                true,
            );
            // Java's source sets B and then unconditionally sets A.
            assert!(manager.is_advanced(DialogType::SetupRecon, AxisID::Second));
            assert!(manager.is_advanced(DialogType::SetupRecon, AxisID::First));
            manager.set_advanced_dialog_type_boolean(DialogType::SetupRecon, false);
            assert!(!manager.is_advanced(DialogType::SetupRecon, AxisID::First));
            assert!(manager.is_advanced(DialogType::SetupRecon, AxisID::Second));
        });
    }

    #[test]
    fn setup_changed_matches_source_whitespace_rule() {
        invoke_and_wait(|| {
            // A dataset file that does not exist yet is a new dataset: the setup
            // dialog opens (ApplicationManager.java:291-310), and an untouched
            // dataset field is not a change.
            let manager = ApplicationManager::new(Some("data.edf"), Some(AxisID::Only));
            assert!(manager.is_new_manager());
            assert!(!manager.is_setup_changed());
        });
    }
}
