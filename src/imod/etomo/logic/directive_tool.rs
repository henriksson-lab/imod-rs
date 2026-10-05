//! `IMOD/Etomo/src/etomo/logic/DirectiveTool.java`.
//!
//! Decides, for the directive editor, whether a directive's include check box should be
//! toggled and whether the directive should be visible, from the editor's display
//! settings (`DirectiveDisplaySettings`, the `DirectiveEditorDialog`).  An event
//! dispatch thread object shared by the editor's panels as `Rc<DirectiveTool>`; the
//! display settings are held weakly because the dialog owns the tool.

use std::cell::Cell;
use std::rc::Weak;

use crate::imod::etomo::etomo_director;
use crate::imod::etomo::storage::directive::Directive;
use crate::imod::etomo::storage::directive_descr_etomo_column::DirectiveDescrEtomoColumn;
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::directive_file_type::{self, DirectiveFileType};
use crate::imod::etomo::ui::directive_display_settings::DirectiveDisplaySettings;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public final class DirectiveTool`.
pub struct DirectiveTool {
    /// Java private final `type`.
    r#type: Option<DirectiveFileType>,
    /// Java private final `fileTypeExists`.
    file_type_exists: bool,
    /// Java private final `displaySettings`.
    display_settings: Weak<dyn DirectiveDisplaySettings>,
    /// Java private `debug`, initialised from the arguments' debug level.
    debug: Cell<DebugLevel>,
}

impl DirectiveTool {
    /// Java `DirectiveTool(DirectiveFileType, boolean, DirectiveDisplaySettings)`.
    pub fn new(
        r#type: Option<DirectiveFileType>,
        file_type_exists: bool,
        display_settings: Weak<dyn DirectiveDisplaySettings>,
    ) -> DirectiveTool {
        DirectiveTool {
            r#type,
            file_type_exists,
            display_settings,
            debug: Cell::new(etomo_director::ARGUMENTS.lock().unwrap().get_debug_level()),
        }
    }

    /// The display settings (the dialog, which outlives every use of its tool).
    fn display_settings(&self) -> std::rc::Rc<dyn DirectiveDisplaySettings> {
        self.display_settings
            .upgrade()
            .expect("DirectiveTool used after its DirectiveDisplaySettings")
    }

    /// Java `setDebug(DebugLevel)`.
    pub fn set_debug(&self, input: DebugLevel) {
        self.debug.set(input);
    }

    /// Java `resetDebug()`.
    pub fn reset_debug(&self) {
        self.debug
            .set(etomo_director::ARGUMENTS.lock().unwrap().get_debug_level());
    }

    /// Java `isToggleDirectiveIncluded(Directive, boolean)`.  Returns true if the include
    /// checkbox value needs to be changed.
    pub fn is_toggle_directive_included(
        &self,
        directive: Option<&Directive>,
        include_checked: bool,
    ) -> bool {
        let Some(directive) = directive else {
            return include_checked != false;
        };
        if !self.is_matches_type(directive) {
            return include_checked != false;
        }
        let display_settings = self.display_settings();
        // Check the display settings in order of precedence.
        let mut include = false;
        let mut exclude = false;
        // if the directive is in a current directive file then the include/exclude
        // settings can affect it.
        let mut matching_index = -1;
        if let Some(r#type) = self.r#type {
            matching_index = r#type.get_index();
        }
        // Check all but the settings for the file type that matches the type of the
        // editor.
        for i in 0..directive_file_type::NUM {
            // Override the lower precedence settings. If the directive, or the any axis
            // version of the directive is in the file, then it may have an effect.
            if i != matching_index && directive.is_in_directive_file(i) {
                // Include and exclude cannot be set at the same time (but they can both
                // be off).
                if display_settings.is_include(i) {
                    include = true;
                    exclude = false;
                } else if display_settings.is_exclude(i) {
                    include = false;
                    exclude = true;
                }
            }
        }
        // the directive file matching the type of the editor has the highest precedence.
        if directive.is_in_directive_file(matching_index) {
            if display_settings.is_include(matching_index) {
                include = true;
                exclude = false;
            } else if display_settings.is_exclude(matching_index) {
                include = false;
                exclude = true;
            }
        }
        if include {
            return include_checked != true;
        }
        if exclude {
            return include_checked != false;
        }
        if !self.file_type_exists {
            return include_checked
                != (self.r#type == Some(DirectiveFileType::User)
                    && directive.get_etomo_column() == Some(DirectiveDescrEtomoColumn::SD)
                    && self.is_matches_type(directive)
                    && directive.get_values().is_changed());
        }
        include_checked != false
    }

    /// Java `isDirectiveVisible(Directive, boolean, boolean)`.  Used whenever necessary
    /// to show or hide directives.  Keeps everything where include is checked visible,
    /// and anything that the user has changed.
    pub fn is_directive_visible(
        &self,
        directive: &Directive,
        included_in_gui: bool,
        changed_in_gui: bool,
    ) -> bool {
        let display_settings = self.display_settings();
        if display_settings.is_show_only_included() {
            return included_in_gui;
        }
        // Included and changed directives should always be visible
        if included_in_gui || changed_in_gui {
            return true;
        }
        // Hide batch-only directives in a template editor. Hide template-only directives
        // in a batch file editor.
        if !self.is_matches_type(directive) {
            return false;
        }
        // Hide unchanged and hidden directives, unless the display settings say
        // otherwise.  Batch directives and undefined directives are never considered
        // hidden.
        let etomo_column = directive.get_etomo_column();
        directive.set_debug(self.debug.get());
        let retval = (display_settings.is_show_unchanged() || directive.get_values().is_changed())
            && (display_settings.is_show_hidden()
                || (etomo_column.is_some() && etomo_column != Some(DirectiveDescrEtomoColumn::NE))
                || self.r#type == Some(DirectiveFileType::Batch)
                || directive.get_description().is_none());
        directive.reset_debug();
        retval
    }

    /// Java private `isMatchesType(Directive)`.  Distinguishes between batch directives
    /// and template directives.
    fn is_matches_type(&self, directive: &Directive) -> bool {
        let template = directive.is_template();
        let batch = directive.is_batch();
        (!template && !batch)
            || (batch && self.r#type == Some(DirectiveFileType::Batch))
            || (template && self.r#type != Some(DirectiveFileType::Batch))
    }
}
