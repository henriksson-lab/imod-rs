//! `IMOD/Etomo/src/etomo/ui/swing/Beads3dFindPanel.java`.
//!
//! Java `final class Beads3dFindPanel implements
//! NewstackOrBlendmont3dFindParent, Tilt3dFindParent, Expandable`: panel to
//! use findbeads3d to find all the beads in an existing or newly created
//! aligned stack which is used to generate a tomogram.  An EDT object:
//! created as `Rc<Self>` by [`Beads3dFindPanel::get_instance`]; every method
//! takes `&self`.
//!
//! The Java field `newstackOrBlendmont3dFindPanel` has the abstract type
//! `NewstackOrBlendmont3dFindPanel` and is cast to the concrete subclass where
//! the Java casts it; here it is [`NewstackOrBlendmont3dFindPanelRef`], one
//! variant per concrete class, which derefs to the shared superclass part.

use std::io;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::const_find_beads3d_param::ConstFindBeads3dParam;
use crate::imod::etomo::comscript::const_tilt_param::ConstTiltParam;
use crate::imod::etomo::comscript::const_tiltalign_param::ConstTiltalignParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process_series::ProcessSeries;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::r#type::recon_screen_state::ReconScreenState;
use crate::imod::etomo::r#type::tomogram_state::TomogramState;
use crate::imod::etomo::r#type::view_type::ViewType;

use super::blendmont_3d_find_panel::Blendmont3dFindPanel;
use super::blendmont_display::BlendmontDisplay;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::erase_gold_panel::EraseGoldPanel;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::final_aligned_stack_dialog;
use super::find_beads3d_display::FindBeads3dDisplay;
use super::find_beads3d_panel::FindBeads3dPanel;
use super::global_expand_button::GlobalExpandButton;
use super::newstack_3d_find_panel::Newstack3dFindPanel;
use super::newstack_display::NewstackDisplay;
use super::newstack_or_blendmont_3d_find_panel::{
    NewstackOrBlendmont3dFindPanel, NewstackOrBlendmont3dFindPanelVirtual,
};
use super::newstack_or_blendmont_3d_find_parent::NewstackOrBlendmont3dFindParent;
use super::newstack_or_blendmont_panel;
use super::panel_header::PanelHeader;
use super::process_display::ProcessDisplay;
use super::reproject_model_panel::ReprojectModelPanel;
use super::spaced_panel::SpacedPanel;
use super::tilt_display::TiltDisplay;
use super::tilt3d_find_panel::Tilt3dFindPanel;
use super::tilt3d_find_parent::Tilt3dFindParent;
use super::tomogram_generation_parent::TomogramGenerationParent;
use super::ui_harness;

/// The Java field type `NewstackOrBlendmont3dFindPanel` (abstract): the
/// concrete instance the constructor chose.
#[derive(Clone)]
pub enum NewstackOrBlendmont3dFindPanelRef {
    /// A `Newstack3dFindPanel` (non-montage).
    Newstack(Rc<Newstack3dFindPanel>),
    /// A `Blendmont3dFindPanel` (montage).
    Blendmont(Rc<Blendmont3dFindPanel>),
}

impl Deref for NewstackOrBlendmont3dFindPanelRef {
    type Target = NewstackOrBlendmont3dFindPanel;
    fn deref(&self) -> &NewstackOrBlendmont3dFindPanel {
        match self {
            Self::Newstack(panel) => panel,
            Self::Blendmont(panel) => panel,
        }
    }
}

impl NewstackOrBlendmont3dFindPanelRef {
    /// Java virtual `runProcess(ProcessResultDisplay, ProcessSeries,
    /// Run3dmodMenuOptions)`, dispatched to the subclass.
    fn run_process(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<crate::imod::etomo::process_series::ProcessSeriesHandle>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        match self {
            Self::Newstack(panel) => NewstackOrBlendmont3dFindPanelVirtual::run_process(
                &**panel,
                process_result_display,
                process_series,
                run_3dmod_menu_options,
            ),
            Self::Blendmont(panel) => NewstackOrBlendmont3dFindPanelVirtual::run_process(
                &**panel,
                process_result_display,
                process_series,
                run_3dmod_menu_options,
            ),
        }
    }
}

/// Java `final class Beads3dFindPanel`.
pub struct Beads3dFindPanel {
    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `pnlGenerateTomogramBody = new JPanel()`.
    pnl_generate_tomogram_body: Rc<JComponent>,
    /// Java private final `newstackOrBlendmont3dFindPanel`.
    newstack_or_blendmont_3d_find_panel: NewstackOrBlendmont3dFindPanelRef,
    /// Java private final `tilt3dFindPanel`.
    tilt3d_find_panel: Rc<Tilt3dFindPanel>,
    /// Java private final `findBeads3dPanel`.
    find_beads3d_panel: Rc<FindBeads3dPanel>,
    /// Java private final `reprojectModelPanel`.
    reproject_model_panel: Rc<ReprojectModelPanel>,
    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `header`.
    header: Rc<PanelHeader>,
    /// Java private final `parent` (held weakly: the parent owns this panel).
    parent: Weak<EraseGoldPanel>,
}

impl Beads3dFindPanel {
    /// Java package-private constructor `Beads3dFindPanel(ApplicationManager,
    /// EraseGoldPanel, AxisID, DialogType, GlobalExpandButton)`.
    fn new(
        manager: &'static ApplicationManager,
        parent: Weak<EraseGoldPanel>,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<Beads3dFindPanel> {
        Rc::new_cyclic(|this: &Weak<Beads3dFindPanel>| {
            // Field initializers.
            let pnl_root = SpacedPanel::get_instance_void();
            let pnl_generate_tomogram_body = JComponent::new_panel();
            // Constructor body.
            let expandable: Weak<dyn Expandable> = this.clone();
            let header = PanelHeader::get_instance(
                Some("Align Stack and Create Tomogram"),
                Some(expandable),
                Some(dialog_type),
            );
            let nob_parent: Weak<dyn NewstackOrBlendmont3dFindParent> = this.clone();
            let newstack_or_blendmont_3d_find_panel = if manager.get_meta_data().get_view_type()
                == ViewType::Montage
            {
                NewstackOrBlendmont3dFindPanelRef::Blendmont(Blendmont3dFindPanel::get_instance(
                    manager,
                    axis_id,
                    dialog_type,
                    nob_parent.clone(),
                ))
            } else {
                NewstackOrBlendmont3dFindPanelRef::Newstack(Newstack3dFindPanel::get_instance(
                    manager,
                    axis_id,
                    dialog_type,
                    nob_parent.clone(),
                ))
            };
            let tilt3d_find_parent: Weak<dyn Tilt3dFindParent> = this.clone();
            let tilt3d_find_panel = Tilt3dFindPanel::get_instance(
                manager,
                axis_id,
                dialog_type,
                tilt3d_find_parent,
                Some(newstack_or_blendmont_3d_find_panel.get3dmod_button()),
            );
            let find_beads3d_panel = FindBeads3dPanel::get_instance(
                manager,
                nob_parent,
                axis_id,
                dialog_type,
                global_advanced_button,
            );
            let reproject_model_panel =
                ReprojectModelPanel::get_instance(manager, axis_id, dialog_type);
            Beads3dFindPanel {
                pnl_root,
                pnl_generate_tomogram_body,
                newstack_or_blendmont_3d_find_panel,
                tilt3d_find_panel,
                find_beads3d_panel,
                reproject_model_panel,
                manager,
                axis_id,
                dialog_type,
                header,
                parent,
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, EraseGoldPanel, AxisID,
    /// DialogType, GlobalExpandButton)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        parent: Weak<EraseGoldPanel>,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<Beads3dFindPanel> {
        let instance = Beads3dFindPanel::new(
            manager,
            parent,
            axis_id,
            dialog_type,
            global_advanced_button,
        );
        instance.create_panel();
        instance
    }

    /// Java `reregisterProcessingMethodMediator()`.
    pub fn reregister_processing_method_mediator(&self) {
        self.tilt3d_find_panel
            .reregister_processing_method_mediator();
    }

    /// Java `getProcessingMethod()`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        self.tilt3d_find_panel.get_processing_method()
    }

    /// Java `done()`.
    pub fn done(&self) {
        self.tilt3d_find_panel.done();
        self.find_beads3d_panel.done();
        self.reproject_model_panel.done();
    }

    /// Java `updateAdvanced(boolean)`.
    pub fn update_advanced(&self, advanced: bool) {
        self.find_beads3d_panel.update_advanced(advanced);
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // Local panels
        let pnl_generate_tomogram = JComponent::new_panel();
        // Root panel
        // Swing layout: pnlRoot.setBoxLayout(BoxLayout.Y_AXIS).
        self.pnl_root.add_j_panel(&pnl_generate_tomogram);
        self.pnl_root
            .add_component(&self.find_beads3d_panel.get_component());
        self.pnl_root
            .add_component(&self.reproject_model_panel.get_component());
        // Generate tomogram panel
        // Swing layout: pnlGenerateTomogram BoxLayout Y_AXIS, untitled etched
        // border.
        pnl_generate_tomogram.add(&self.header.get_container());
        pnl_generate_tomogram.add(&self.pnl_generate_tomogram_body);
        // Generate tomogram body panel
        // Swing layout: pnlGenerateTomogramBody BoxLayout Y_AXIS.
        self.pnl_generate_tomogram_body
            .add(&self.newstack_or_blendmont_3d_find_panel.get_component());
        self.pnl_generate_tomogram_body
            .add(&self.tilt3d_find_panel.get_root());
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java `isAdvanced()`.
    pub fn is_advanced(&self) -> bool {
        self.find_beads3d_panel.is_advanced()
    }

    /// Java `getNewstack3dFindDisplay()`.
    pub fn get_newstack3d_find_display(&self) -> Option<Rc<dyn NewstackDisplay>> {
        if self.manager.get_meta_data().get_view_type() != ViewType::Montage {
            // (NewstackDisplay) newstackOrBlendmont3dFindPanel
            if let NewstackOrBlendmont3dFindPanelRef::Newstack(panel) =
                &self.newstack_or_blendmont_3d_find_panel
            {
                return Some(panel.clone() as Rc<dyn NewstackDisplay>);
            }
        }
        None
    }

    /// Java `getBlendmont3dFindDisplay()`.
    pub fn get_blendmont3d_find_display(&self) -> Option<Rc<dyn BlendmontDisplay>> {
        if self.manager.get_meta_data().get_view_type() == ViewType::Montage {
            // (BlendmontDisplay) newstackOrBlendmont3dFindPanel
            if let NewstackOrBlendmont3dFindPanelRef::Blendmont(panel) =
                &self.newstack_or_blendmont_3d_find_panel
            {
                return Some(panel.clone() as Rc<dyn BlendmontDisplay>);
            }
        }
        None
    }

    /// Java `getTilt3dFindDisplay()`.
    pub fn get_tilt3d_find_display(&self) -> Option<Rc<dyn TiltDisplay>> {
        Some(self.tilt3d_find_panel.clone() as Rc<dyn TiltDisplay>)
    }

    /// Java `getFindBeads3dDisplay()`.
    pub fn get_find_beads3d_display(&self) -> Option<Rc<dyn FindBeads3dDisplay>> {
        Some(self.find_beads3d_panel.clone() as Rc<dyn FindBeads3dDisplay>)
    }

    /// Java `setTiltState(TomogramState, ConstMetaData)`.
    pub fn set_tilt_state(&self, state: &TomogramState, meta_data: &dyn ConstMetaData) {
        self.tilt3d_find_panel.set_state(state, meta_data);
    }

    /// Java `setParameters(ConstTiltParam, boolean) throws
    /// FileNotFoundException, IOException`.
    pub fn set_parameters_const_tilt_param_boolean(
        &self,
        param: &dyn ConstTiltParam,
        initialize: bool,
    ) -> Result<(), io::Error> {
        self.tilt3d_find_panel
            .set_parameters_const_tilt_param_boolean(param, initialize);
        Ok(())
    }

    /// Java `setParameters(ConstFindBeads3dParam, boolean)`.
    pub fn set_parameters_const_find_beads3d_param_boolean(
        &self,
        param: &dyn ConstFindBeads3dParam,
        initialize: bool,
    ) {
        self.find_beads3d_panel
            .set_parameters_const_find_beads3d_param_boolean(param, initialize);
    }

    /// Java `initialize()`.
    pub fn initialize(&self) {
        self.newstack_or_blendmont_3d_find_panel.initialize();
    }

    /// Java `setParameters(ConstTiltalignParam, boolean)`.
    pub fn set_parameters_const_tiltalign_param_boolean(
        &self,
        param: &ConstTiltalignParam,
        initialize: bool,
    ) {
        self.tilt3d_find_panel
            .set_parameters_const_tiltalign_param_boolean(param, initialize);
    }

    /// Java `setOverrideParameters(ConstMetaData)`.
    pub fn set_override_parameters(&self, meta_data: &dyn ConstMetaData) {
        self.tilt3d_find_panel.set_override_parameters(meta_data);
    }

    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .set_state(Some(screen_state.get_stack_align_and_tilt_header_state()));
        self.find_beads3d_panel
            .set_parameters_recon_screen_state(screen_state);
        self.reproject_model_panel.set_parameters(screen_state);
    }

    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_recon_screen_state(&self, screen_state: &ReconScreenState) {
        self.header
            .get_state(Some(screen_state.get_stack_align_and_tilt_header_state()));
        self.find_beads3d_panel
            .get_parameters_recon_screen_state(screen_state);
    }

    /// Java `setParameters(BlendmontParam)`.
    pub fn set_parameters_blendmont_param(&self, param: &BlendmontParam) {
        if self.manager.get_meta_data().get_view_type() == ViewType::Montage {
            // ((Blendmont3dFindPanel) newstackOrBlendmont3dFindPanel).setParameters(param)
            if let NewstackOrBlendmont3dFindPanelRef::Blendmont(panel) =
                &self.newstack_or_blendmont_3d_find_panel
            {
                BlendmontDisplay::set_parameters(&**panel, param);
            }
        }
    }

    /// Java `setParameters(NewstParam)`.
    pub fn set_parameters_newst_param(&self, param: &NewstParam) {
        if self.manager.get_meta_data().get_view_type() != ViewType::Montage {
            // ((Newstack3dFindPanel) newstackOrBlendmont3dFindPanel).setParameters(param)
            if let NewstackOrBlendmont3dFindPanelRef::Newstack(panel) =
                &self.newstack_or_blendmont_3d_find_panel
            {
                NewstackDisplay::set_parameters(&**panel, param);
            }
        }
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }

    /// Java `validate()`.
    pub fn validate(&self) -> bool {
        NewstackOrBlendmont3dFindPanel::validate(&self.newstack_or_blendmont_3d_find_panel)
    }

    /// Java `getParameters(MetaData) throws FortranInputSyntaxException`.
    pub fn get_parameters_meta_data(
        &self,
        meta_data: &MetaData,
    ) -> Result<(), FortranInputSyntaxException> {
        NewstackOrBlendmont3dFindPanel::get_parameters(
            &self.newstack_or_blendmont_3d_find_panel,
            meta_data,
        );
        self.tilt3d_find_panel.get_parameters_meta_data(meta_data)?;
        Ok(())
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        NewstackOrBlendmont3dFindPanel::set_parameters(
            &self.newstack_or_blendmont_3d_find_panel,
            meta_data,
        );
        self.tilt3d_find_panel
            .set_parameters_const_meta_data(meta_data);
    }

    /// Java override `getBeadSize()` (NewstackOrBlendmont3dFindParent).
    pub fn get_bead_size(&self) -> String {
        self.find_beads3d_panel.get_bead_size()
    }

    /// Java override `isFiducialess()` (NewstackOrBlendmont3dFindParent).
    pub fn is_fiducialess(&self) -> bool {
        // Java dereferences `parent` unconditionally; the parent owns this panel.
        self.parent
            .upgrade()
            .is_some_and(|parent| parent.is_fiducialess())
    }
}

impl Expandable for Beads3dFindPanel {
    /// Java override `expand(GlobalExpandButton)`: empty.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java override `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.header.equals_open_close(button) {
            self.pnl_generate_tomogram_body
                .set_visible(button.is_expanded());
        }
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::INSTANCE
            .with(|harness| harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager)));
    }
}

impl TomogramGenerationParent for Beads3dFindPanel {
    /// Java override `isMultifilt()`.
    fn is_multifilt(&self) -> bool {
        false
    }

    /// Java override `isCtf3d()`.
    fn is_ctf3d(&self) -> bool {
        false
    }

    /// Java override `isBackProjection()`.
    fn is_back_projection(&self) -> bool {
        true
    }

    /// Java override `isMethodPlugin()`.
    fn is_method_plugin(&self) -> bool {
        false
    }

    /// Java override `isSirt()`.
    fn is_sirt(&self) -> bool {
        false
    }
}

impl NewstackOrBlendmont3dFindParent for Beads3dFindPanel {
    /// Java override `getBeadSize()`.
    fn get_bead_size(&self) -> String {
        Beads3dFindPanel::get_bead_size(self)
    }

    /// Java override `isFiducialess()`.
    fn is_fiducialess(&self) -> bool {
        Beads3dFindPanel::is_fiducialess(self)
    }
}

impl Tilt3dFindParent for Beads3dFindPanel {
    /// Java override `tilt3dFindAction(ProcessResultDisplay,
    /// Deferred3dmodButton, Run3dmodMenuOptions, ProcessingMethod)`.
    fn tilt3d_find_action(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        tiltp_processing_method: ProcessingMethod,
    ) {
        let manager: &'static dyn BaseManager = self.manager;
        let aligned_stack = &file_type::CLASS.aligned_stack;
        // The parent (this class) is responsible for running tilt_3dfind because it
        // may have to run newst/blend_3dfind first.
        // Validate to make sure that the binned bead size in pixels is not too
        // small.
        if !aligned_stack.exists(Some(manager), Some(self.axis_id))
            && Beads3dFindPanel::is_fiducialess(self)
        {
            // 3D find can't handle a missing _ali file in a fiducialess dataset.
            ui_harness::INSTANCE.with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(manager),
                    &format!(
                        "Please go to the {} tab and press {}.",
                        final_aligned_stack_dialog::FINAL_ALIGNED_STACK_TAB_LABEL,
                        newstack_or_blendmont_panel::RUN_BUTTON_LABEL
                    ),
                    "Please Build Aligned Stack",
                    Some(self.axis_id),
                )
            });
            return;
        }
        if !self.validate() {
            return;
        }
        // If the full aligned stack does not exist, or the binning of the full
        // aligned stack is different from the binning requested here, run
        // newst/blend_3dfind.com and then run tilt_3dfind.
        if !aligned_stack.exists(Some(manager), Some(self.axis_id))
            || !self.manager.equals_binning(
                self.axis_id,
                self.newstack_or_blendmont_3d_find_panel.get_binning(),
                aligned_stack,
            )
        {
            let process_display: Rc<dyn ProcessDisplay> = self.tilt3d_find_panel.clone();
            let process_series = ProcessSeries::new_with_process_display(
                manager,
                self.axis_id,
                Some(self.dialog_type),
                Some(process_display),
                Some("tilt3dFindAction"),
            );
            process_series.borrow_mut().set_next_process(
                Some(&ProcessName::TILT_3D_FIND.to_string()),
                Some(tiltp_processing_method),
            );
            self.newstack_or_blendmont_3d_find_panel.run_process(
                process_result_display,
                Some(process_series),
                run_3dmod_menu_options,
            );
        } else {
            self.manager
                .get_state()
                .set_stack_using_newst_or_blend_3d_find_output(self.axis_id, false);
            // Just run tilt_3dfind.
            self.tilt3d_find_panel.tilt3d_find_action(
                process_result_display,
                deferred_3dmod_button,
                run_3dmod_menu_options,
            );
        }
    }
}
