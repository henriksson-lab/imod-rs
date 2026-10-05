//! `IMOD/Etomo/src/etomo/plugin/TomoGenMethodPlugin.java`.
//!
//! Please read all of the information in `etomo.plugin.Plugin`.  This interface is
//! integrated with `TomogramGenerationDialog`.  It represents an alternative method of
//! creating a tomogram.  It can be used to attach one panel to
//! `TomogramGenerationDialog` (along-side the SIRT radio button).  It can also be used
//! to supply a customized child of `TiltPanel`.  An example of the implementation of
//! this interface is in `etomo.plugin.demo`.  This interface is based on the
//! requirements of the Ettention plugin.
//!
//! The class implementing this plugin will be managed by `TomogramGenerationExpert`.
//! There will be one instance per axis.  Only one plugin of this type will be loaded by
//! `PluginFactory`.  For a given axis, an instance of this class will exist from the
//! time the Tomogram Generation dialog is first opened, until the dataset is closed.  A
//! new instance of `TomogramGenerationDialog` is constructed each time the dialog is
//! opened, while the instance of `TomogramGenerationExpert` remains.
//!
//! `TomogramGenerationDialog` will obtain a pointer to the plugin's panel, and will
//! create a radio button to select it.  It will treat the plugin's panel as one of its
//! own.  See `etomo.plugin.PluginPanel`.  (The rest of the Java javadoc - tips and the
//! classes a plugin may inherit - is not repeated here; see the source.)
//!
//! The dialog and the expert hand themselves to the plugin while they are being
//! constructed, so the plugin receives weak references (the dialog owns the plugin's
//! panel, the expert owns the plugin).

use std::rc::{Rc, Weak};

use super::plugin::Plugin;
use super::plugin_panel::PluginPanel;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::ui::swing::global_expand_button::GlobalExpandButton;
use crate::imod::etomo::ui::swing::tilt_panel::TiltPanelVirtual;
use crate::imod::etomo::ui::swing::tomogram_generation_dialog::TomogramGenerationDialog;
use crate::imod::etomo::ui::swing::tomogram_generation_expert::TomogramGenerationExpert;
use crate::imod::etomo::ui::swing::tomogram_generation_parent::TomogramGenerationParent;

/// Java `public interface TomoGenMethodPlugin extends Plugin`.
pub trait TomoGenMethodPlugin: Plugin {
    /// Java `initTomoGenMethod(ApplicationManager, TomogramGenerationExpert)`.  Will be
    /// called after `Plugin.init`.
    fn init_tomo_gen_method(
        &self,
        manager: &'static ApplicationManager,
        expert: Weak<TomogramGenerationExpert>,
    );

    /// Java `getPanel(TomogramGenerationDialog, GlobalExpandButton)`.  Called while
    /// `TomogramGenerationDialog` is being constructed.  Construct a new instance of the
    /// panel in this function.  `None` is Java null.
    fn get_panel(
        &self,
        parent: Weak<TomogramGenerationDialog>,
        btn_advanced_dialog: &Rc<GlobalExpandButton>,
    ) -> Option<Rc<dyn PluginPanel>>;

    /// Java `hasCustomTiltPanel()`.  Called while `TomogramGenerationDialog` is being
    /// constructed.  Return true if the plugin returns its own instance of a
    /// `TiltPanel` child class.
    fn has_custom_tilt_panel(&self) -> bool;

    /// Java `getTiltPanel(TomogramGenerationParent, GlobalExpandButton)`.  Called while
    /// `TomogramGenerationDialog` is being constructed.  Return an instance of a
    /// `TiltPanel` child class or null (`None`).
    fn get_tilt_panel(
        &self,
        parent: Weak<dyn TomogramGenerationParent>,
        btn_advanced_dialog: &Rc<GlobalExpandButton>,
    ) -> Option<Rc<dyn TiltPanelVirtual>>;
}
