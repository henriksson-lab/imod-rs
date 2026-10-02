//! `IMOD/Etomo/src/etomo/ui/swing/InitialCombinePanel.java`.
//!
//! The Initial Match tab of `TomogramCombinationDialog`: the Initial Match
//! copy of `SolvematchPanel` and the Matchvol1 panel (initial match size and
//! the "Restart at Matchvol1" button).
//!
//! Java `class InitialCombinePanel implements ContextMenu,
//! InitialCombineFields, Run3dmodButtonContainer, Expandable`: an EDT object
//! created as `Rc<Self>` by [`InitialCombinePanel::get_instance`]; every
//! method takes `&self`.  The dialog is held weakly (it owns its panels).  The
//! inner class `ButtonActionListener` is a closure holding a weak reference to
//! the panel, kept in `button_action` so `removeListeners` can remove it.

use std::cell::Cell;
use std::rc::{Rc, Weak};

use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etomo_panel::EtomoPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::global_expand_button::GlobalExpandButton;
use super::initial_combine_fields::InitialCombineFields;
use super::labeled_text_field::LabeledTextField;
use super::panel_header::PanelHeader;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::solvematch_panel::SolvematchPanel;
use super::spaced_panel::{self, SpacedPanel};
use super::tomogram_combination_dialog::{self, TomogramCombinationDialog};
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_solvematch_param::ConstSolvematchParam;
use crate::imod::etomo::comscript::dualvolmatch_param::DualvolmatchParam;
use crate::imod::etomo::comscript::matchvol_param::MatchvolParam;
use crate::imod::etomo::comscript::solvematch_param::SolvematchParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent, MouseEvent, MouseListener};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::match_mode::MatchMode;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::{self, ReconScreenState};
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::util::mrc_header::MRCHeader;

/// Java `class InitialCombinePanel implements ContextMenu,
/// InitialCombineFields, Run3dmodButtonContainer, Expandable`.
pub struct InitialCombinePanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `btnMatchvolRestart`.
    btn_matchvol_restart: Rc<Run3dmodButton>,
    /// Java private final `pnlMatchvol1Body = SpacedPanel.getInstance(true)`.
    pnl_matchvol1_body: Rc<SpacedPanel>,
    /// Java private final `matchvol1Header`.
    matchvol1_header: Rc<PanelHeader>,
    /// Java private final `ltfOutputSizeY`.
    ltf_output_size_y: Rc<LabeledTextField>,
    /// Java private final `lOutputSizeYInfo = new JLabel()`.
    l_output_size_y_info: Rc<JComponent>,
    /// Java private final `buttonAction = new ButtonActionListener(this)`.
    button_action: ActionListener,
    /// Java private final `tomogramCombinationDialog` (held weakly).
    tomogram_combination_dialog: Weak<TomogramCombinationDialog>,
    /// Java private final `applicationManager`.
    application_manager: &'static ApplicationManager,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `pnlSolvematch`.
    pnl_solvematch: Rc<SolvematchPanel>,

    /// Java private `matchMode = null`.
    match_mode: Cell<Option<MatchMode>>,

    /// Java `this`.
    this: Weak<InitialCombinePanel>,
}

impl InitialCombinePanel {
    /// Java private constructor `InitialCombinePanel(TomogramCombinationDialog,
    /// ApplicationManager, DialogType, GlobalExpandButton)`.  Default
    /// constructor.
    fn new(
        parent: Weak<TomogramCombinationDialog>,
        app_mgr: &'static ApplicationManager,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<InitialCombinePanel> {
        Rc::new_cyclic(|this: &Weak<InitialCombinePanel>| {
            let expandable: Weak<dyn Expandable> = this.clone();
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            let pnl_matchvol1_body = SpacedPanel::get_instance_boolean(true);
            let ltf_output_size_y = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some("Initial match size: "),
            );
            let l_output_size_y_info = JComponent::new_label("");
            // Java `new ButtonActionListener(this)`.
            let listenee = this.clone();
            let button_action: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(listenee) = listenee.upgrade() {
                    listenee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });
            // Constructor body.
            // this.dialogType = dialogType; tomogramCombinationDialog = parent;
            // applicationManager = appMgr;
            global_advanced_button.register_expandable(expandable.clone());
            let matchvol1_header =
                PanelHeader::get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
                    Some("Matchvol1"),
                    Some(expandable),
                    Some(DialogType::TomogramCombination),
                    Some(global_advanced_button.clone()),
                );
            // (Run3dmodButton) appMgr.getProcessResultDisplayFactory(AxisID.ONLY)
            // .getRestartMatchvol1()
            let btn_matchvol_restart = app_mgr
                .get_process_result_display_factory(AxisID::Only)
                .get_restart_matchvol1();
            let pnl_solvematch = SolvematchPanel::get_instance(
                parent.clone(),
                tomogram_combination_dialog::LBL_INITIAL,
                app_mgr,
                recon_screen_state::COMBINE_INITIAL_SOLVEMATCH_HEADER_GROUP.as_str(),
                dialog_type,
                true,
                Some(global_advanced_button),
            );
            InitialCombinePanel {
                pnl_root,
                btn_matchvol_restart,
                pnl_matchvol1_body,
                matchvol1_header,
                ltf_output_size_y,
                l_output_size_y_info,
                button_action,
                tomogram_combination_dialog: parent,
                application_manager: app_mgr,
                dialog_type,
                pnl_solvematch,
                match_mode: Cell::new(None),
                this: this.clone(),
            }
        })
    }

    /// Java package-private static `getInstance(TomogramCombinationDialog,
    /// ApplicationManager, DialogType, GlobalExpandButton)`.
    pub fn get_instance(
        parent: Weak<TomogramCombinationDialog>,
        manager: &'static ApplicationManager,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<InitialCombinePanel> {
        let instance =
            InitialCombinePanel::new(parent, manager, dialog_type, global_advanced_button);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // panels
        let pnl_matchvol1 = EtomoPanel::new();
        // init
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_matchvol_restart.set_container(Some(container));
        // Swing layout: lOutputSizeYInfo.setAlignmentX(Component.CENTER_ALIGNMENT);
        // btnMatchvolRestart.setAlignmentX(Component.CENTER_ALIGNMENT).
        // root
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.Y_AXIS)).
        self.pnl_root.add(&self.pnl_solvematch.get_container());
        self.pnl_root.add(&pnl_matchvol1.get_component());
        // Matchvol1
        // Swing layout: pnlMatchvol1.setBorder(BorderFactory.createEtchedBorder());
        // pnlMatchvol1.setLayout(new BoxLayout(pnlMatchvol1, BoxLayout.Y_AXIS)).
        pnl_matchvol1.add(&self.matchvol1_header);
        pnl_matchvol1
            .get_component()
            .add(&self.pnl_matchvol1_body.get_container());
        // Matchvol1Body
        self.pnl_matchvol1_body.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_matchvol1_body
            .add_labeled_text_field(&self.ltf_output_size_y);
        self.pnl_matchvol1_body
            .add_j_label(&self.l_output_size_y_info);
        self.pnl_matchvol1_body
            .add_multi_line_button(&self.btn_matchvol_restart);
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // Bind the UI objects to their ActionListeners
        self.btn_matchvol_restart
            .add_action_listener(self.button_action.clone());
        // Mouse listener for context menu
        let context_menu: Weak<dyn ContextMenu> = self.this.clone();
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.pnl_root.add_mouse_listener(mouse_adapter);
    }

    /// Java package-private `removeListeners()`.
    pub fn remove_listeners(&self) {
        self.btn_matchvol_restart
            .remove_action_listener(&self.button_action);
    }

    /// Java package-private `setDeferred3dmodButtons()`.
    pub fn set_deferred_3dmod_buttons(&self) {
        if let Some(dialog) = self.tomogram_combination_dialog.upgrade() {
            let deferred: Rc<dyn Deferred3dmodButton> = dialog.get_imod_combined_button();
            self.btn_matchvol_restart
                .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        }
        self.pnl_solvematch.set_deferred_3dmod_buttons();
    }

    /// Java package-private `getProcessingMethod()`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        ProcessingMethod::LocalCpu
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java package-private `updateAdvanced(boolean)`.
    pub fn update_advanced(&self, state: bool) {
        self.update_matchvol1_advanced(state);
    }

    /// Java package-private `updateDisplay()`.
    pub fn update_display(&self) {
        self.pnl_solvematch.update_display();
    }

    /// Java package-private `updateMatchvol1Advanced(boolean)`.
    pub fn update_matchvol1_advanced(&self, advanced: bool) {
        self.ltf_output_size_y.set_visible(advanced);
        self.l_output_size_y_info.set_visible(advanced);
    }

    /// Java package-private final `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_solvematch.set_visible(visible);
    }

    /// Java package-private `getParameters(MatchvolParam, boolean)`.
    pub fn get_parameters_matchvol_param_boolean(
        &self,
        param: &mut MatchvolParam,
        do_validation: bool,
    ) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        match self.ltf_output_size_y.get_text_boolean(do_validation) {
            Ok(text) => {
                param.set_output_size_y(text.as_deref());
                true
            }
            Err(_) => false,
        }
    }

    /// Java public `setParameters(MatchvolParam)`.
    pub fn set_parameters_matchvol_param(&self, param: &MatchvolParam) {
        self.ltf_output_size_y
            .set_text_int(param.get_output_size_y());
    }

    /// Java package-private `setSolvematchParams(ConstSolvematchParam)`.  Set
    /// the solvematch parameters from the ConstSolvematchParam object.
    pub fn set_solvematch_params(&self, solvematch_param: &ConstSolvematchParam) {
        self.pnl_solvematch
            .set_parameters_const_solvematch_param(solvematch_param);
    }

    /// Java package-private `setDualvolmatchParams(DualvolmatchParam)`.
    pub fn set_dualvolmatch_params(&self, param: &DualvolmatchParam) {
        self.pnl_solvematch.set_parameters_dualvolmatch_param(param);
    }

    /// Java package-private `getSolvematchParams(SolvematchParam, boolean)`.
    /// Get the solvematch parameters from the UI.  `Err` carries the message
    /// of the NumberFormatException `SolvematchPanel.getParameters` lets
    /// through.
    pub fn get_solvematch_params(
        &self,
        solvematch_param: &mut SolvematchParam,
        do_validation: bool,
    ) -> Result<bool, String> {
        self.pnl_solvematch
            .get_parameters_solvematch_param_boolean(solvematch_param, do_validation)
    }

    /// Java package-private `getParameters(DualvolmatchParam, boolean)`.
    pub fn get_parameters_dualvolmatch_param_boolean(
        &self,
        param: &mut DualvolmatchParam,
        do_validation: bool,
    ) -> bool {
        self.pnl_solvematch
            .get_parameters_dualvolmatch_param_boolean(param, do_validation)
    }

    /// Java package-private `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.pnl_solvematch
            .get_parameters_recon_screen_state(screen_state);
    }

    /// Java package-private final `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.pnl_solvematch
            .set_parameters_recon_screen_state(screen_state);
        self.btn_matchvol_restart.set_button_state(
            screen_state
                .get_button_state(self.btn_matchvol_restart.get_button_state_key().as_deref()),
        );
    }

    /// Java private `setToolTipText()`.  Initialize the tooltip text for the
    /// axis panel objects.
    fn set_tool_tip_text(&self) {
        let text = "Thickness to make initial matching volume, which may need to be \
                    thicker than the final matching volume to contain all the material \
                    needed for patch correlations.";
        self.ltf_output_size_y.set_tool_tip_text(Some(text));
        self.l_output_size_y_info.set_tool_tip_text(Some(text));
        self.btn_matchvol_restart.set_tool_tip_text(Some(
            "Resume and make first matching volume, despite a small displacement \
             between the match check volumes",
        ));
    }
}

impl InitialCombineFields for InitialCombinePanel {
    /// Java public override `setMatchMode(MatchMode)`.
    fn set_match_mode(&self, match_mode: Option<MatchMode>) {
        if self.match_mode.get() == match_mode {
            return;
        }
        self.match_mode.set(match_mode);
        // set lOutputSizeYInfo
        let mut to_axis_id = AxisID::First;
        let mut from_axis_id = AxisID::Second;
        if match_mode == Some(MatchMode::AToB) {
            to_axis_id = AxisID::Second;
            from_axis_id = AxisID::First;
        }
        let manager: &'static dyn BaseManager = self.application_manager;
        // MRCHeader.getInstance(BaseManager, AxisID, FileType) never returns null.
        let Some(to_header) = MRCHeader::get_instance_from_file_type(
            manager,
            Some(to_axis_id),
            &file_type::CLASS.tilt_output,
        ) else {
            return;
        };
        let Some(from_header) = MRCHeader::get_instance_from_file_type(
            manager,
            Some(from_axis_id),
            &file_type::CLASS.tilt_output,
        ) else {
            return;
        };
        let mut to_y: i32 = -1;
        // try { if (!toHeader.read(applicationManager)) return; }
        // catch (InvalidParameterException | IOException e) { e.printStackTrace(); }
        let read = to_header.borrow_mut().read_with_manager(manager);
        match read {
            Ok(false) => return,
            Ok(true) => {}
            Err(e) => eprintln!("{e}"),
        }
        to_y = to_header.borrow().get_n_rows();
        let mut from_y: i32 = -1;
        let read = from_header.borrow_mut().read_with_manager(manager);
        match read {
            Ok(false) => return,
            Ok(true) => {}
            Err(e) => eprintln!("{e}"),
        }
        from_y = from_header.borrow().get_n_rows();
        self.l_output_size_y_info.set_text(&format!(
            "Original {} size is {}.  Final size will be {}",
            from_axis_id.get_extension().to_uppercase(),
            from_y,
            to_y
        ));
    }

    /// Java public override `getMatchMode()`.  Since match mode isn't modified
    /// in initial tab, always return null.
    fn get_match_mode(&self) -> Option<MatchMode> {
        None
    }

    /// Java public override `isUseCorrespondingPoints()`.
    fn is_use_corresponding_points(&self) -> bool {
        self.pnl_solvematch.is_use_corresponding_points()
    }

    /// Java public override `setUseCorrespondingPoints(boolean)`.
    fn set_use_corresponding_points(&self, use_: bool) {
        self.pnl_solvematch.set_use_corresponding_points(use_);
    }

    /// Java public override `isEnabled()`.
    fn is_enabled(&self) -> bool {
        match self.tomogram_combination_dialog.upgrade() {
            Some(dialog) => dialog.is_tab_enabled(tomogram_combination_dialog::LBL_INITIAL),
            None => false,
        }
    }

    /// Java public override `isInitialVolumeMatching()`.
    fn is_initial_volume_matching(&self) -> bool {
        self.pnl_solvematch.is_initial_volume_matching()
    }

    /// Java public override `setInitialVolumeMatching(boolean)`.
    fn set_initial_volume_matching(&self, input: bool) {
        self.pnl_solvematch.set_initial_volume_matching(input);
    }

    // InitialCombineFields interface pass-thru
    /// Java public override `getSurfacesOrModels()`.
    fn get_surfaces_or_models(&self) -> FiducialMatch {
        self.pnl_solvematch.get_surfaces_or_models()
    }

    /// Java public override `setSurfacesOrModels(FiducialMatch)`.
    fn set_surfaces_or_models(&self, state: FiducialMatch) {
        self.pnl_solvematch.set_surfaces_or_models(state);
    }

    /// Java public override `isBinBy2()`.
    fn is_bin_by2(&self) -> bool {
        self.pnl_solvematch.is_bin_by2()
    }

    /// Java public override `setBinBy2(boolean)`.
    fn set_bin_by2(&self, state: bool) {
        self.pnl_solvematch.set_bin_by2(state);
    }

    /// Java public override `setUseList(String)`.
    fn set_use_list(&self, use_list: Option<&str>) {
        self.pnl_solvematch.set_use_list(use_list);
    }

    /// Java public override `setFiducialMatchListA(String)`.
    fn set_fiducial_match_list_a(&self, fiducial_match_list_a: Option<&str>) {
        self.pnl_solvematch
            .set_fiducial_match_list_a(fiducial_match_list_a);
    }

    /// Java public override `getUseList(boolean)`.
    fn get_use_list_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.pnl_solvematch.get_use_list_boolean(do_validation)
    }

    /// Java public override `getUseList()`.
    fn get_use_list_void(&self) -> Option<String> {
        self.pnl_solvematch.get_use_list_void()
    }

    /// Java public override `getFiducialMatchListA(boolean)`.
    fn get_fiducial_match_list_a_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.pnl_solvematch
            .get_fiducial_match_list_a_boolean(do_validation)
    }

    /// Java public override `getFiducialMatchListA()`.
    fn get_fiducial_match_list_a_void(&self) -> Option<String> {
        self.pnl_solvematch.get_fiducial_match_list_a_void()
    }

    /// Java public override `setFiducialMatchListB(String)`.
    fn set_fiducial_match_list_b(&self, fiducial_match_list_b: Option<&str>) {
        self.pnl_solvematch
            .set_fiducial_match_list_b(fiducial_match_list_b);
    }

    /// Java public override `getFiducialMatchListB(boolean)`.
    fn get_fiducial_match_list_b_boolean(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.pnl_solvematch
            .get_fiducial_match_list_b_boolean(do_validation)
    }

    /// Java public override `getFiducialMatchListB()`.
    fn get_fiducial_match_list_b_void(&self) -> Option<String> {
        self.pnl_solvematch.get_fiducial_match_list_b_void()
    }
}

impl Expandable for InitialCombinePanel {
    /// Java public override `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.update_advanced(button.is_expanded());
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(AxisID::Only), Some(manager))
        });
    }

    /// Java public override `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        let expanded = button.is_expanded();
        // matchvol1Header is never null here (final, set by the constructor).
        if self.matchvol1_header.equals_open_close(button) {
            self.pnl_matchvol1_body.set_visible(expanded);
        } else if self.matchvol1_header.equals_advanced_basic(button) {
            self.update_matchvol1_advanced(expanded);
        }
        let manager: &'static dyn BaseManager = self.application_manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(AxisID::Only), Some(manager))
        });
    }
}

impl ContextMenu for InitialCombinePanel {
    /// Java public override `popUpContextMenu(MouseEvent)`.  Right mouse button
    /// context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = ["Solvematch".to_string(), "Matchshifts".to_string()];
        let man_page = [
            "solvematch.html".to_string(),
            "matchshifts.html".to_string(),
        ];
        let log_file_label = ["Transferfid".to_string(), "Solvematch".to_string()];
        let log_file = ["transferfid.log".to_string(), "solvematch.log".to_string()];

        let manager: &'static dyn BaseManager = self.application_manager;
        // The Java constructor's IllegalArgumentException (mismatched arrays)
        // cannot occur: the arrays are built in label/value pairs.
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root,
            mouse_event,
            Some("Initial Problems in Combining"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            manager,
            AxisID::Only,
        );
    }
}

impl Run3dmodButtonContainer for InitialCombinePanel {
    /// Java public override `action(String, Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // The dialog owns this panel, so it is alive while the panel is.
        let Some(dialog) = self.tomogram_combination_dialog.upgrade() else {
            return;
        };
        // Synchronize this panel with the others
        dialog.synchronize(tomogram_combination_dialog::LBL_INITIAL, true);

        if Some(command) == self.btn_matchvol_restart.get_action_command().as_deref() {
            let display: ProcessResultDisplayHandle = self.btn_matchvol_restart.clone();
            // A null Run3dmodMenuOptions becomes `new Run3dmodMenuOptions()` where
            // the 3dmod state opens (ImodState.open).
            self.application_manager.matchvol1_combine(
                Some(display),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options.unwrap_or_default(),
                self.dialog_type,
                dialog.get_run_processing_method(),
                dialog.is_parallel(),
                !dialog.is_run_volcombine(),
            );
        }
    }
}
