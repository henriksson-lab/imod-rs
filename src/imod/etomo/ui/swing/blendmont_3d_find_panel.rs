//! `IMOD/Etomo/src/etomo/ui/swing/Blendmont3dFindPanel.java`.
//!
//! Java `final class Blendmont3dFindPanel extends NewstackOrBlendmont3dFindPanel
//! implements BlendmontDisplay`.  The superclass is the embedded `base`
//! (reached through `Deref`); the abstract members are
//! [`NewstackOrBlendmont3dFindPanelVirtual`] and `Run3dmodButtonContainer`.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::blendmont_display::{BlendmontDisplay, BlendmontDisplayException};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::newstack_or_blendmont_3d_find_panel::{
    NewstackOrBlendmont3dFindPanel, NewstackOrBlendmont3dFindPanelVirtual,
};
use super::newstack_or_blendmont_3d_find_parent::NewstackOrBlendmont3dFindParent;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::blendmont_param::{self, BlendmontParam, ConvertError};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process_series::ProcessSeriesHandle;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `final class Blendmont3dFindPanel`.
pub struct Blendmont3dFindPanel {
    /// The `NewstackOrBlendmont3dFindPanel` superclass.
    base: NewstackOrBlendmont3dFindPanel,
}

impl Deref for Blendmont3dFindPanel {
    type Target = NewstackOrBlendmont3dFindPanel;
    fn deref(&self) -> &NewstackOrBlendmont3dFindPanel {
        &self.base
    }
}

impl Blendmont3dFindPanel {
    /// Java private constructor `Blendmont3dFindPanel(ApplicationManager,
    /// AxisID, DialogType, NewstackOrBlendmont3dFindParent)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: Weak<dyn NewstackOrBlendmont3dFindParent>,
    ) -> Rc<Blendmont3dFindPanel> {
        Rc::new_cyclic(|this: &Weak<Blendmont3dFindPanel>| {
            let this: Weak<dyn NewstackOrBlendmont3dFindPanelVirtual> = this.clone();
            Blendmont3dFindPanel {
                base: NewstackOrBlendmont3dFindPanel::new(
                    manager,
                    axis_id,
                    dialog_type,
                    parent,
                    this,
                ),
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType,
    /// NewstackOrBlendmont3dFindParent)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: Weak<dyn NewstackOrBlendmont3dFindParent>,
    ) -> Rc<Blendmont3dFindPanel> {
        let instance = Blendmont3dFindPanel::new(manager, axis_id, dialog_type, parent);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }
}

impl BlendmontDisplay for Blendmont3dFindPanel {
    /// Java `getParameters(BlendmontParam, boolean) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`.
    fn get_parameters(
        &self,
        param: &mut BlendmontParam,
        _do_validation: bool,
    ) -> Result<bool, BlendmontDisplayException> {
        param.set_bin_by_factor_int(self.get_binning());
        param.set_mode(blendmont_param::Mode::Blend3dFind);
        // Opt out of validation because state should be correct sinces it's data is from a
        // process that ran.
        match param.convert_to_starting_and_ending_xand_y(
            &self
                .manager
                .get_state()
                .get_stack_user_size_to_output_in_x_and_y(self.axis_id),
            self.manager
                .get_meta_data()
                .get_image_rotation(self.axis_id)
                .get_double(),
            None,
        ) {
            // The boolean result is ignored by the Java.
            Ok(_) => {}
            Err(ConvertError::FortranInputSyntax(e)) => {
                return Err(BlendmontDisplayException::FortranInputSyntaxException(e));
            }
            // InvalidParameterException / IOException from reading the montage size.
            Err(ConvertError::MontagesizeRead(message)) => {
                return Err(BlendmontDisplayException::InvalidParameterException(
                    InvalidParameterException::new(&message),
                ));
            }
        }
        Ok(true)
    }

    /// Java `setParameters(BlendmontParam)`: empty.
    fn set_parameters(&self, _param: &BlendmontParam) {}

    /// Java inherited public final `validate()`.
    fn validate(&self) -> bool {
        self.base.validate()
    }

    /// Java inherited public `isFiducialess()`.
    fn is_fiducialess(&self) -> bool {
        self.base.is_fiducialess()
    }
}

impl NewstackOrBlendmont3dFindPanelVirtual for Blendmont3dFindPanel {
    /// Java `runProcess(ProcessResultDisplay, ProcessSeries, Run3dmodMenuOptions)`.
    fn run_process(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // A Java null Run3dmodMenuOptions is the empty option set.
        self.manager.blend3d_find(
            process_result_display,
            process_series,
            None,
            self.axis_id,
            run_3dmod_menu_options.unwrap_or_default(),
            self.dialog_type,
            self,
        );
    }
}

impl Run3dmodButtonContainer for Blendmont3dFindPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    /// Executes the action associated with command.  Deferred3dmodButton is
    /// null if it comes from the dialog's ActionListener.  Otherwise is comes
    /// from a Run3dmodButton which called action(Run3dmodButton,
    /// Run3dmoMenuOptions).  In that case it will be null unless it was set in
    /// the Run3dmodButton.
    fn action(
        &self,
        command: &str,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.get3dmod_full_button_action_command().as_deref() {
            // A Java null Run3dmodMenuOptions is the empty option set.
            self.manager
                .imod_fine_align3d_find(self.axis_id, run_3dmod_menu_options.unwrap_or_default());
        }
    }
}
