//! `IMOD/Etomo/src/etomo/plugin/PluginFactory.java`.
//!
//! A factory that uses the Service Loader to return plugins that correspond to
//! specific interfaces found in the `etomo.plugin` packages.  See `etomo.plugin.Plugin`
//! for more information.  (The Java javadoc's step-by-step instructions for building a
//! plugin jar are not repeated here; see the source.)
//!
//! **The jar boundary.**  Java's `ServiceLoader.load(TomoGenMethodPlugin.class)` scans
//! the class path for `META-INF/services/etomo.plugin.TomoGenMethodPlugin` entries and
//! instantiates the classes they name.  The `etomo` launcher puts `etomo.jar`,
//! `$IMOD_DIR/Plugins/*` and `$IMOD_DIR/imodplug/etomo/*` on that class path
//! (`pysrc/etomo`), and `etomo.jar` itself declares no such service, so with no plugin
//! jar installed the iterator is empty, `pickPlugin` returns null and no method plugin
//! is used.  A Rust program cannot load the Java classes in a plugin jar, so here the
//! service iterator is always that empty one: the translation behaves exactly as Java
//! does when no plugin jar is present.  A jar that is present is not loaded (Java would
//! offer it with the "Allow plugin?" question).  The built-in demo plugin
//! (`loadDemoPlugin`, `etomo -plugin`) is translated and is not affected.
//!
//! `loadBasicPlugin(UIComponent)` has no caller in the source and is not translated
//! (`DEAD_CODE.md`).

use std::rc::Rc;

use super::demo::demo_plugin_manager::DemoPluginManager;
use super::plugin::Plugin;
use super::tomo_gen_method_plugin::TomoGenMethodPlugin;
use crate::imod::etomo::etomo_director::INSTANCE as ETOMO_DIRECTOR;
use crate::imod::etomo::ui::swing::popup::Popup;
use crate::imod::etomo::ui::swing::ui_harness;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java `public final class PluginFactory` (static methods only).
pub struct PluginFactory;

impl PluginFactory {
    /// Java public static `loadDemoPlugin(UIComponent)`.  Returns a plugin that has been
    /// incorporated into the etomo jar file.  Plugin is self-contained in its own
    /// package.
    pub fn load_demo_plugin(
        _ui_component: Option<&dyn UIComponent>,
    ) -> Option<Rc<dyn TomoGenMethodPlugin>> {
        let plugin: Rc<dyn TomoGenMethodPlugin> = DemoPluginManager::new();
        PluginFactory::print_info(Some(&*plugin), "TomoGenMethodPlugin");
        Some(plugin)
    }

    /// Java public static `loadTomoGenMethodPlugin(UIComponent)`.  Gets one plugin
    /// corresponding to the `TomoGenMethodPlugin` interface.
    pub fn load_tomo_gen_method_plugin(
        ui_component: Option<&dyn UIComponent>,
    ) -> Option<Rc<dyn TomoGenMethodPlugin>> {
        // `ServiceLoader.load(TomoGenMethodPlugin.class).iterator()`: the jar boundary
        // (see the module comment) - no Java plugin class can be instantiated here, so
        // the iterator is the one Java gets with no plugin jar on the class path.
        let mut loaded_plugins = std::iter::empty::<Option<Rc<dyn TomoGenMethodPlugin>>>();
        let plugin = PluginFactory::pick_plugin(&mut loaded_plugins, ui_component);
        PluginFactory::print_info(
            plugin.as_deref().map(|p| p as &dyn Plugin),
            "TomoGenMethodPlugin",
        );
        plugin
    }

    /// Java private static `printInfo(Plugin, String)`.
    fn print_info(plugin: Option<&dyn Plugin>, interface_name: &str) {
        if let Some(plugin) = plugin {
            eprintln!(
                "Successfully loaded plugin that implements the {} interface.\nPlugin information:{},{},{}",
                interface_name,
                plugin.get_title().as_deref().unwrap_or("null"),
                plugin.get_version().as_deref().unwrap_or("null"),
                plugin.get_description().as_deref().unwrap_or("null")
            );
        }
    }

    /// Java private static `pickPlugin(Iterator, UIComponent)`.  Java's
    /// `catch (ServiceConfigurationError e)` (a malformed provider entry in a jar) has
    /// no counterpart: the iterator does not read jars (see the module comment).
    fn pick_plugin(
        loaded_plugins: &mut dyn Iterator<Item = Option<Rc<dyn TomoGenMethodPlugin>>>,
        ui_component: Option<&dyn UIComponent>,
    ) -> Option<Rc<dyn TomoGenMethodPlugin>> {
        let mut list: Option<Vec<Rc<dyn TomoGenMethodPlugin>>> = None;
        for plugin in loaded_plugins {
            // Only use plugins that work with this niche.
            let Some(plugin) = plugin else {
                continue;
            };
            let key = plugin.get_key();
            // Check for plugins that have already been allowed or excluded.
            if ETOMO_DIRECTOR
                .with_user_configuration(|configuration| configuration.has_plugin(key.as_deref()))
            {
                // Does the user want this plugin?
                if ETOMO_DIRECTOR.with_user_configuration(|configuration| {
                    configuration.is_plugin(key.as_deref())
                }) {
                    // Plugin has been verified.
                    return Some(plugin);
                }
                list.get_or_insert_with(Vec::new).push(plugin);
            } else {
                // No information on the this plugin so ask the user.
                let popup = Popup::get_yes_no_instance(
                    ui_component,
                    Some("Allow plugin?"),
                    Some(&format!(
                        "An external plugin for eTomo was found.\n{} version: {}\n{}\n\nUse this plugin?",
                        plugin.get_title().as_deref().unwrap_or("null"),
                        plugin.get_version().as_deref().unwrap_or("null"),
                        plugin.get_description().as_deref().unwrap_or("null")
                    )),
                    Some("Do not ask about this plugin again."),
                );
                ui_harness::INSTANCE.with(|ui_harness| ui_harness.open_popup(&popup));
                let mut allowed = false;
                if popup.is_yes() {
                    allowed = true;
                }
                if popup.is_checkbox_selected() {
                    ETOMO_DIRECTOR.with_user_configuration_mut(|configuration| {
                        configuration.set_plugin(key.as_deref(), allowed)
                    });
                }
                if allowed {
                    return Some(plugin);
                }
            }
        }
        // Java keeps the excluded plugins in `list` and never reads it.
        let _ = list;
        None
    }
}
