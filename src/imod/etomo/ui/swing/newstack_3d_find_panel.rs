//! `IMOD/Etomo/src/etomo/ui/swing/Newstack3dFindPanel.java`.
//!
//! Java `final class Newstack3dFindPanel extends NewstackOrBlendmont3dFindPanel
//! implements NewstackDisplay`.  The superclass is the embedded `base`
//! (reached through `Deref`); the abstract members are
//! [`NewstackOrBlendmont3dFindPanelVirtual`] and `Run3dmodButtonContainer`.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::deferred_3dmod_button::Deferred3dmodButton;
use super::newstack_display::{NewstackDisplay, NewstackDisplayException};
use super::newstack_or_blendmont_3d_find_panel::{
    NewstackOrBlendmont3dFindPanel, NewstackOrBlendmont3dFindPanelVirtual,
};
use super::newstack_or_blendmont_3d_find_parent::NewstackOrBlendmont3dFindParent;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::newst_param::{self, NewstParam, SetSizeToOutputInXandYError};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process_series::ProcessSeriesHandle;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Number;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `final class Newstack3dFindPanel`.
pub struct Newstack3dFindPanel {
    /// The `NewstackOrBlendmont3dFindPanel` superclass.
    base: NewstackOrBlendmont3dFindPanel,
}

impl Deref for Newstack3dFindPanel {
    type Target = NewstackOrBlendmont3dFindPanel;
    fn deref(&self) -> &NewstackOrBlendmont3dFindPanel {
        &self.base
    }
}

impl Newstack3dFindPanel {
    /// Java private constructor `Newstack3dFindPanel(ApplicationManager,
    /// AxisID, DialogType, NewstackOrBlendmont3dFindParent)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: Weak<dyn NewstackOrBlendmont3dFindParent>,
    ) -> Rc<Newstack3dFindPanel> {
        Rc::new_cyclic(|this: &Weak<Newstack3dFindPanel>| {
            let this: Weak<dyn NewstackOrBlendmont3dFindPanelVirtual> = this.clone();
            Newstack3dFindPanel {
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
    ) -> Rc<Newstack3dFindPanel> {
        let instance = Newstack3dFindPanel::new(manager, axis_id, dialog_type, parent);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }
}

impl NewstackDisplay for Newstack3dFindPanel {
    /// Java `getParameters(NewstParam, boolean) throws
    /// FortranInputSyntaxException, InvalidParameterException, IOException`.
    fn get_parameters(
        &self,
        newst_param: &mut NewstParam,
        _do_validation: bool,
    ) -> Result<bool, NewstackDisplayException> {
        newst_param.set_command_mode(Some(newst_param::Mode::FullAlignedStack));
        newst_param.set_fiducialess_alignment(
            self.manager
                .get_meta_data()
                .is_fiducialess_alignment(self.axis_id),
        );
        let binning = self.get_binning();
        // Only explicitly write out the binning if its value is something other than
        // the default of 1 to keep from cluttering up the com script
        if binning > 1 {
            newst_param.set_bin_by_factor(Some(Number::Integer(binning)));
        } else {
            newst_param.set_bin_by_factor(Some(Number::Integer(i32::MIN)));
        }
        // Get the rest of the parameters from current state of the final stack
        let state = self.manager.get_state();
        newst_param.set_linear_interpolation(state.is_stack_use_linear_interpolation(self.axis_id));
        // State values should be valid - opt out of validate
        match newst_param.set_size_to_output_in_xand_y(
            &state.get_stack_user_size_to_output_in_x_and_y(self.axis_id),
            self.get_binning(),
            self.manager
                .get_meta_data()
                .get_image_rotation(self.axis_id)
                .get_double(),
            None,
        ) {
            // The boolean result is ignored by the Java.
            Ok(_) => {}
            Err(SetSizeToOutputInXandYError::FortranInputSyntax(e)) => {
                return Err(NewstackDisplayException::FortranInputSyntaxException(e));
            }
            // InvalidParameterException / IOException from reading the header.
            Err(SetSizeToOutputInXandYError::HeaderRead(message)) => {
                return Err(NewstackDisplayException::InvalidParameterException(
                    InvalidParameterException::new(&message),
                ));
            }
        }
        // Set output file because this file was copied from newst.com
        // Java builds a local `Vector outputFile` holding this name and never
        // uses it; the lookup is kept.
        let manager: &'static dyn BaseManager = self.manager;
        let mut output_file: Vec<Option<String>> = Vec::new();
        output_file.push(
            file_type::CLASS
                .newst_or_blend_3d_find_output
                .get_file_name(Some(manager), Some(self.axis_id)),
        );
        let _ = output_file;
        newst_param.set_output_file(&file_type::CLASS.newst_or_blend_3d_find_output);
        newst_param.set_process_name(ProcessName::NEWST_3D_FIND);
        Ok(true)
    }

    /// Java `setParameters(ConstNewstParam)`: empty.
    fn set_parameters(&self, _param: &dyn ConstNewstParam) {}

    /// Java inherited public final `validate()`.
    fn validate(&self) -> bool {
        self.base.validate()
    }

    /// Java inherited public `isFiducialess()`.
    fn is_fiducialess(&self) -> bool {
        self.base.is_fiducialess()
    }
}

impl NewstackOrBlendmont3dFindPanelVirtual for Newstack3dFindPanel {
    /// Java `runProcess(ProcessResultDisplay, ProcessSeries, Run3dmodMenuOptions)`.
    fn run_process(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // A Java null Run3dmodMenuOptions is the empty option set.
        self.manager.newst3d_find(
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

impl Run3dmodButtonContainer for Newstack3dFindPanel {
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
