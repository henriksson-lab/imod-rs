//! `IMOD/Etomo/src/etomo/ui/swing/NewstackPanel.java`.
//!
//! Java `final class NewstackPanel extends NewstackOrBlendmontPanel`.  The
//! superclass is the embedded `base` (reached through `Deref`); the abstract
//! members are [`NewstackOrBlendmontPanelVirtual`] and
//! `Run3dmodButtonContainer`.  `Expandable`, `NewstackDisplay` and
//! `BlendmontDisplay` are implemented by the Java superclass; here they are
//! implemented on the subclass by forwarding to `base`, so that `this` (an
//! `Rc<NewstackPanel>`) can be handed out as those interfaces as in Java.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::blendmont_display::{BlendmontDisplay, BlendmontDisplayException};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::newstack_display::{NewstackDisplay, NewstackDisplayException};
use super::newstack_or_blendmont_panel::{
    NewstackOrBlendmontPanel, NewstackOrBlendmontPanelVirtual,
};
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::comscript::blendmont_param::BlendmontParam;
use crate::imod::etomo::comscript::const_newst_param::ConstNewstParam;
use crate::imod::etomo::comscript::newst_param::NewstParam;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `final class NewstackPanel extends NewstackOrBlendmontPanel`.
pub struct NewstackPanel {
    /// The `NewstackOrBlendmontPanel` superclass.
    base: NewstackOrBlendmontPanel,
}

impl Deref for NewstackPanel {
    type Target = NewstackOrBlendmontPanel;
    fn deref(&self) -> &NewstackOrBlendmontPanel {
        &self.base
    }
}

impl NewstackPanel {
    /// Java private constructor `NewstackPanel(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton)`.  The superclass constructor calls the
    /// abstract `getHeaderTitle()`; its value ("Newstack") is passed in (see
    /// `newstack_or_blendmont_panel.rs`).
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<NewstackPanel> {
        Rc::new_cyclic(|this: &Weak<NewstackPanel>| {
            let this: Weak<dyn NewstackOrBlendmontPanelVirtual> = this.clone();
            NewstackPanel {
                base: NewstackOrBlendmontPanel::new(
                    manager,
                    axis_id,
                    dialog_type,
                    global_advanced_button,
                    this,
                    HEADER_TITLE,
                ),
            }
        })
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType,
    /// GlobalExpandButton)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<NewstackPanel> {
        let instance = NewstackPanel::new(manager, axis_id, dialog_type, global_advanced_button);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }
}

/// The value of Java `getHeaderTitle()`.
const HEADER_TITLE: &str = "Newstack";

impl NewstackOrBlendmontPanelVirtual for NewstackPanel {
    /// Java `getHeaderTitle()`.
    fn get_header_title(&self) -> String {
        HEADER_TITLE.to_string()
    }
}

impl Run3dmodButtonContainer for NewstackPanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    /// Executes the action associated with command.  Deferred3dmodButton is
    /// null if it comes from the dialog's ActionListener.  Otherwise is comes
    /// from a Run3dmodButton which called action(Run3dmodButton,
    /// Run3dmoMenuOptions).  In that case it will be null unless it was set in
    /// the Run3dmodButton.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.get_run_process_button_action_command().as_deref() {
            let fiducialess_params = self.get_fiducialess_params();
            // A Java null Run3dmodMenuOptions is the empty option set.
            self.manager.newst(
                Some(self.get_run_process_result_display()),
                None,
                deferred_3dmod_button,
                self.axis_id,
                run_3dmod_menu_options.unwrap_or_default(),
                self.dialog_type,
                &*fiducialess_params,
                Some(self as &dyn NewstackDisplay),
                ProcessName::NEWST,
            );
        } else if Some(command) == self.get3dmod_full_button_action_command().as_deref() {
            self.manager
                .imod_fine_align(self.axis_id, run_3dmod_menu_options.unwrap_or_default());
        }
    }
}

impl Expandable for NewstackPanel {
    /// Java inherited `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        self.base.expand_expand_button(button);
    }

    /// Java inherited `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.base.expand_global_expand_button(button);
    }
}

impl NewstackDisplay for NewstackPanel {
    /// Java inherited final `getParameters(NewstParam, boolean)`.
    fn get_parameters(
        &self,
        newst_param: &mut NewstParam,
        do_validation: bool,
    ) -> Result<bool, NewstackDisplayException> {
        NewstackDisplay::get_parameters(&self.base, newst_param, do_validation)
    }

    /// Java inherited final `setParameters(ConstNewstParam)`.
    fn set_parameters(&self, newst_param: &dyn ConstNewstParam) {
        NewstackDisplay::set_parameters(&self.base, newst_param);
    }

    /// Java inherited `validate()`.
    fn validate(&self) -> bool {
        NewstackDisplay::validate(&self.base)
    }

    /// Java inherited `isFiducialess()`.
    fn is_fiducialess(&self) -> bool {
        NewstackDisplay::is_fiducialess(&self.base)
    }
}

impl BlendmontDisplay for NewstackPanel {
    /// Java inherited final `getParameters(BlendmontParam, boolean)`.
    fn get_parameters(
        &self,
        param: &mut BlendmontParam,
        do_validation: bool,
    ) -> Result<bool, BlendmontDisplayException> {
        BlendmontDisplay::get_parameters(&self.base, param, do_validation)
    }

    /// Java inherited final `setParameters(BlendmontParam)`.
    fn set_parameters(&self, param: &BlendmontParam) {
        BlendmontDisplay::set_parameters(&self.base, param);
    }

    /// Java inherited `validate()`.
    fn validate(&self) -> bool {
        BlendmontDisplay::validate(&self.base)
    }

    /// Java inherited `isFiducialess()`.
    fn is_fiducialess(&self) -> bool {
        BlendmontDisplay::is_fiducialess(&self.base)
    }
}
