//! `IMOD/Etomo/src/etomo/plugin/demo/DemoProcessResultDisplayFactory.java`.
//!
//! Stores and manages process result displays (buttons).  This class can insert its
//! display list into the main process result display list.  This allows the displays
//! created here to interact with etomo's buttons.  To retrieve the displays created
//! here, place getters in this class.
//!
//! This class can also give each of its displays a unique ID.  This allows the buttons
//! to be retrieved generically by their saved ID.  Button IDs are saved when a process
//! is running when eTomo exits.
//!
//! `DemoProcessResultDisplayFactory extends AbstractProcessResultDisplayFactory`: the
//! superclass is embedded as `base` and reached through `Deref`.  An
//! event-dispatch-thread object.

use std::ops::Deref;
use std::rc::Rc;

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_result_display::{
    ProcessResultDisplay, ProcessResultDisplayHandle,
};
use crate::imod::etomo::ui::swing::abstract_process_result_display_factory::AbstractProcessResultDisplayFactory;
use crate::imod::etomo::ui::swing::multi_line_button::MultiLineButton;
use crate::imod::etomo::ui::swing::run_3dmod_button::Run3dmodButton;

/// Java `final class DemoProcessResultDisplayFactory extends
/// AbstractProcessResultDisplayFactory`.
pub struct DemoProcessResultDisplayFactory {
    /// The `AbstractProcessResultDisplayFactory` superclass.
    base: AbstractProcessResultDisplayFactory,
    /// Java private final `etomoPluginDemo`.  etomoPluginDemo is a button with a
    /// right-click menu.  It can run a process and then cause another button to run
    /// 3dmod.
    etomo_plugin_demo: ProcessResultDisplayHandle,
    /// Java private final `unselectDependencies`.
    unselect_dependencies: ProcessResultDisplayHandle,
}

impl Deref for DemoProcessResultDisplayFactory {
    type Target = AbstractProcessResultDisplayFactory;
    fn deref(&self) -> &AbstractProcessResultDisplayFactory {
        &self.base
    }
}

impl DemoProcessResultDisplayFactory {
    /// Java package-private `DemoProcessResultDisplayFactory(AxisID, AxisType)`.
    fn new(
        axis_id: Option<AxisID>,
        axis_type: Option<AxisType>,
    ) -> DemoProcessResultDisplayFactory {
        // Field initializers.
        let etomo_plugin_demo: ProcessResultDisplayHandle =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some("Run EtomoPluginDemo"),
                Some(DialogType::TomogramGeneration),
            );
        let unselect_dependencies: ProcessResultDisplayHandle =
            MultiLineButton::get_toggle_button_instance_string_dialog_type(
                Some("Unselect Dependencies"),
                Some(DialogType::TomogramGeneration),
            );
        // super("etomo.plugin.demo.ProcessResultDisplayFactory" + (axisType ==
        // AxisType.DUAL_AXIS ? axisID.toString() : ""))
        let suffix = if axis_type == Some(AxisType::DualAxis) {
            match axis_id {
                Some(axis_id) => axis_id.to_string(),
                None => "null".to_string(),
            }
        } else {
            String::new()
        };
        DemoProcessResultDisplayFactory {
            base: AbstractProcessResultDisplayFactory::new(format!(
                "etomo.plugin.demo.ProcessResultDisplayFactory{suffix}"
            )),
            etomo_plugin_demo,
            unselect_dependencies,
        }
    }

    /// Java package-private static `getInstance(ApplicationManager, AxisID, AxisType)`.
    /// Inserts data from this class to into `ProcessResultDisplayFactory`.  Call this
    /// function once per axis and manager.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: Option<AxisID>,
        axis_type: Option<AxisType>,
    ) -> Rc<DemoProcessResultDisplayFactory> {
        let instance = Rc::new(DemoProcessResultDisplayFactory::new(axis_id, axis_type));
        instance.initialize(manager, axis_id);
        instance
    }

    /// Java private `initialize(ApplicationManager, AxisID)`.  Build one or more
    /// dependency lists and insert them into the main factory's list.  See
    /// `ProcessResultDisplayFactory` for more information on dependency lists.
    fn initialize(&self, manager: &'static ApplicationManager, axis_id: Option<AxisID>) {
        // The manager's factory and screen state are per axis; Java's null axisID is
        // the single (A) axis there.
        let axis_id = axis_id.unwrap_or(AxisID::Only);
        // Build a global dependency list in this instance.
        let mut id = 0;
        self.add_dependency(Some(&self.etomo_plugin_demo), id);
        id += 1;
        self.add_dependency(Some(&self.unselect_dependencies), id);
        let _ = id;

        // Insert this instance's global dependency list into the manager factory's
        // global dependency list. Placing these two buttons before the first TomoGen
        // button.
        let main_factory = manager.get_process_result_display_factory(axis_id);
        main_factory.insert_after(
            Some(&self.base),
            Some(&(main_factory.get_use_filtered_stack() as ProcessResultDisplayHandle)),
        );
        // Resets this instance's global list - no effect on the displays.
        self.reset();

        // Multiple lists can be built, added, and then reset. But each display can only
        // be used once. This is because global lists are composed of displays linked
        // together with their own next pointers.

        // The individual buttons can be modified after the dependency list was added.

        // Setting the etomoPluginDemo display to only affect the unselectDependencies
        // display. This is how the creation of temporary files is handled. The button
        // that creates the temporary file does not use the global dependecy list, and
        // so has no effect any button other then the Use... button it is paired with.

        // The unselectDependencies display will affect all displays that come after it
        // on the manager factory's global dependency list. Both buttons will be
        // affected by displays preceding them.

        // setUseGlobalDependencyList has the opposite functionality as the
        // addDependents function, which as been removed. If addDependents was used,
        // simply the delete the calls. The addDpendents functionality is now the
        // default state.
        self.etomo_plugin_demo.set_use_global_dependency_list(false);
        self.etomo_plugin_demo
            .add_dependent_display(self.unselect_dependencies.clone());

        // Set display states
        if let Some(screen_state) = manager.get_base_screen_state(Some(axis_id)) {
            self.etomo_plugin_demo.set_screen_state(screen_state);
            self.unselect_dependencies.set_screen_state(screen_state);
        }
    }

    /// Java package-private `getEtomoPluginDemo()`.
    pub fn get_etomo_plugin_demo(&self) -> ProcessResultDisplayHandle {
        self.etomo_plugin_demo.clone()
    }

    /// Java package-private `getUnselectDependencies()`.
    pub fn get_unselect_dependencies(&self) -> ProcessResultDisplayHandle {
        self.unselect_dependencies.clone()
    }
}
