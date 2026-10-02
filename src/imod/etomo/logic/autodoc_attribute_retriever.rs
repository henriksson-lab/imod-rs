//! `IMOD/Etomo/src/etomo/logic/AutodocAttributeRetriever.java`.
//!
//! Retrieves default values and tooltips for directive-backed fields from the program
//! autodocs, falling back to progDefaults.adoc and the directive description file.
//!
//! **Shape.**  Java has one `INSTANCE` holding two mutable caches.  The autodocs those
//! caches point at are owned by `AutodocFactory`, whose registry and arena are
//! per-thread (`storage/autodoc/autodoc_factory.rs`), so an autodoc pointer is only
//! meaningful on the thread that loaded it.  The caches therefore live in
//! `thread_local!` statics next to the factory's own, and `INSTANCE` is a field-less
//! value every caller can share (the callers are the Swing fields, on the event
//! dispatch thread).  The source's `synchronized (this)` double-checked load of
//! `progDefaultsAutodoc` needs no lock on a per-thread cache.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;

use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use crate::imod::etomo::storage::autodoc::section::Section;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::ui::field::Field;

/// Java `AutodocAttributeRetriever`.  The instance fields are the thread-local statics
/// below (see the module comment).
pub struct AutodocAttributeRetriever {}

/// Java `INSTANCE`.
pub static INSTANCE: AutodocAttributeRetriever = AutodocAttributeRetriever {};

thread_local! {
    /// Java private `progDefaultsAutodoc`, initialised to null.
    static PROG_DEFAULTS_AUTODOC: Cell<*mut Autodoc> = const { Cell::new(std::ptr::null_mut()) };
    /// Java private `failedAutodocNames`, initialised to null.  Avoid trying to open
    /// nonexistent autodocs multiple times.
    static FAILED_AUTODOC_NAMES: RefCell<Option<HashSet<String>>> = const { RefCell::new(None) };
}

impl AutodocAttributeRetriever {
    /// Java `getDefaultValue(DirectiveDef)`.  Gets the default value from the autodoc
    /// corresponding to the directiveDef, or from progDefaults as a fallback.
    pub fn get_default_value(&self, directive_def: Option<DirectiveDef>) -> Option<String> {
        let directive_def = directive_def?;
        let autodoc_name = self.get_autodoc_name(Some(directive_def))?;
        let field_name = self.get_field_name(directive_def)?;
        let mut value: Option<String> = None;
        // If the autodoc has not been loaded, look for directiveDef.command in
        // progDefaults.adoc to see if the autodoc has defaults before loading it. If
        // progDefefault.adoc doesn't show any defaults for the command, return null.
        let autodoc = self.get_autodoc(&autodoc_name, true);
        if !autodoc.is_null() {
            // SAFETY: a non-null autodoc is owned by the factory's arena on this thread
            // for the life of the thread.
            let autodoc: &Autodoc = unsafe { &*autodoc };
            let field_section = unsafe {
                autodoc.get_section(Some(etomo_autodoc::FIELD_SECTION_NAME), Some(&field_name))
            };
            if !field_section.is_null() {
                // SAFETY: sections are owned by their autodoc.
                let field_section: &Section = unsafe { &*field_section };
                let attribute = unsafe { field_section.get_attribute(Some("default")) };
                if !attribute.is_null() {
                    // SAFETY: attributes are owned by their section.
                    value = unsafe { (*attribute).get_value() };
                    if value.is_some() {
                        return value;
                    }
                }
            }
        }
        if value.is_none() {
            // Fallback - if the default was not found in the autodoc, try to find it in
            // progDefaults.
            let command_section = self.get_prog_defaults_command_section(&autodoc_name);
            if command_section.is_null() {
                return None;
            }
            // SAFETY: the section is owned by the progDefaults autodoc.
            let command_section: &Section = unsafe { &*command_section };
            let attribute = unsafe { command_section.get_attribute(Some(&field_name)) };
            if !attribute.is_null() {
                // SAFETY: attributes are owned by their section.
                return unsafe { (*attribute).get_value() };
            }
        }
        None
    }

    /// Java `getTooltip(DirectiveDef)`.  Gets the tooltip from the autodoc corresponding
    /// to the directiveDef.  The fallback is the description from the directive def
    /// file.
    pub fn get_tooltip(&self, directive_def: Option<DirectiveDef>) -> Option<String> {
        let directive_def = directive_def?;
        let tooltip: Option<String>;
        let autodoc_name = self.get_autodoc_name(Some(directive_def));
        if let Some(autodoc_name) = autodoc_name {
            let field_name = self.get_field_name(directive_def);
            if let Some(field_name) = field_name {
                let autodoc = self.get_autodoc(&autodoc_name, false);
                if !autodoc.is_null() {
                    // SAFETY: owned by the factory's arena on this thread.
                    tooltip = unsafe {
                        etomo_autodoc::get_tooltip(
                            Some(unsafe { &*autodoc } as &dyn ReadOnlyAutodoc),
                            Some(&field_name),
                        )
                    };
                    if let Some(tooltip) = tooltip {
                        return Some(format!("{} ({})", tooltip, directive_def));
                    }
                }
            }
        }
        Some(directive_def.get_tooltip())
    }

    /// Java `setToolTipText(Field)`.  Handles the complexity of setting a three part,
    /// formatted tooltip.  This works for both simple and combo fields (such as a
    /// checkbox and text field combo).
    pub fn set_tool_tip_text(&self, field: Option<&dyn Field>) {
        let Some(field) = field else {
            return;
        };
        // Get descriptions to add to the tooltip from the directive. Also add an
        // unformatted tooltip if then isn't one.
        let directive_def = field.get_directive_def();
        let mut param_descr: Option<String> = None;
        let mut directive_descr: Option<String> = None;
        if let Some(directive_def) = directive_def {
            let mut autodoc: *mut Autodoc = std::ptr::null_mut();
            let mut field_name: Option<String> = None;
            let autodoc_name = self.get_autodoc_name(Some(directive_def));
            if let Some(autodoc_name) = autodoc_name {
                field_name = self.get_field_name(directive_def);
                if field_name.is_some() {
                    autodoc = self.get_autodoc(&autodoc_name, false);
                }
            }
            // If there's no preset tooltip, set one from the directive.
            if !field.has_unformatted_tooltip() {
                if !autodoc.is_null() {
                    // SAFETY: owned by the factory's arena on this thread.
                    let unformatted_tooltip = unsafe {
                        etomo_autodoc::get_unformatted_tooltip(
                            Some(unsafe { &*autodoc } as &dyn ReadOnlyAutodoc),
                            field_name.as_deref(),
                        )
                    };
                    field.set_unformatted_tooltip(unformatted_tooltip.as_deref());
                } else {
                    field.set_unformatted_tooltip(
                        directive_def.get_unformatted_tooltip().as_deref(),
                    );
                }
            }
            // Get the parameter description.
            if !autodoc.is_null() && field_name.is_some() {
                // SAFETY: owned by the factory's arena on this thread.
                param_descr = unsafe {
                    etomo_autodoc::get_source_tooltip_string_autodoc(
                        Some(unsafe { &*autodoc } as &dyn ReadOnlyAutodoc),
                        field_name.as_deref(),
                    )
                };
            }
            // Get the directive description
            directive_descr = Some(directive_def.to_string());
        }
        // Set the tool tip.
        field.use_unformatted_tooltip(param_descr.as_deref(), directive_descr.as_deref());
    }

    /// Java private `getAutodocName(DirectiveDef)`.
    fn get_autodoc_name(&self, directive_def: Option<DirectiveDef>) -> Option<String> {
        let directive_def = directive_def?;
        let mut autodoc_name: Option<String> = None;
        if directive_def.is_comparam() {
            autodoc_name = directive_def.get_command();
        } else if directive_def.is_runtime() {
            // Warning: not all of the directive modules have a corresponding .adoc file.
            let command = directive_def.get_module();
            if let Some(command) = command {
                // `String.toLowerCase()` (default locale; module names are ASCII).
                autodoc_name = Some(command.to_lowercase());
            }
        }
        if let Some(name) = &autodoc_name
            && FAILED_AUTODOC_NAMES.with_borrow(|failed_autodoc_names| {
                failed_autodoc_names
                    .as_ref()
                    .is_some_and(|failed_autodoc_names| failed_autodoc_names.contains(name))
            })
        {
            return None;
        }
        autodoc_name
    }

    /// Java private `getFieldName(DirectiveDef)`.
    fn get_field_name(&self, directive_def: DirectiveDef) -> Option<String> {
        let mut name = directive_def.get_name();
        if !directive_def.is_runtime() || name.is_empty() {
            return Some(name);
        }
        // Runtime names start with a small letter. Capitalize it so it matches the
        // parameter name, which will be capitalized.
        let mut chars = name.chars();
        let first = chars.next().unwrap();
        let first_letter: String = first.to_uppercase().collect();
        if name.chars().count() > 1 {
            name = first_letter + chars.as_str();
        } else {
            name = first_letter;
        }
        Some(name)
    }

    // <p>Updates done</p>

    /// Java private `getAutodoc(String, boolean)`.  Null is a null pointer.
    fn get_autodoc(&self, autodoc_name: &str, if_autodoc_loaded: bool) -> *mut Autodoc {
        if if_autodoc_loaded && !autodoc_factory::is_loaded(autodoc_name) {
            return std::ptr::null_mut();
        }
        // SAFETY: the factory owns the returned autodoc for the life of this thread.
        match unsafe { autodoc_factory::get_instance_name(None, Some(autodoc_name)) } {
            Ok(autodoc) => {
                // SAFETY: as above; a non-null pointer is live.
                if autodoc.is_null() || unsafe { (*autodoc).to_string() } == "" {
                    FAILED_AUTODOC_NAMES.with_borrow_mut(|failed_autodoc_names| {
                        if failed_autodoc_names.is_none() {
                            *failed_autodoc_names = Some(HashSet::new());
                        }
                        let failed_autodoc_names = failed_autodoc_names.as_mut().unwrap();
                        if !failed_autodoc_names.contains(autodoc_name) {
                            failed_autodoc_names.insert(autodoc_name.to_string());
                        }
                    });
                    std::ptr::null_mut()
                } else {
                    autodoc
                }
            }
            // `catch (final FileNotFoundException e)`
            Err(LogFileError::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => {
                FAILED_AUTODOC_NAMES.with_borrow_mut(|failed_autodoc_names| {
                    if failed_autodoc_names.is_none() {
                        *failed_autodoc_names = Some(HashSet::new());
                    }
                    let failed_autodoc_names = failed_autodoc_names.as_mut().unwrap();
                    if !failed_autodoc_names.contains(autodoc_name) {
                        failed_autodoc_names.insert(autodoc_name.to_string());
                    }
                });
                std::ptr::null_mut()
            }
            // `catch (final LockException e) {}`
            Err(LogFileError::Lock(_)) => std::ptr::null_mut(),
            // `catch (final LogFileException | IOException e)`
            Err(e) => {
                // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                eprintln!("{}", e);
                std::ptr::null_mut()
            }
        }
    }

    /// Java private `getProgDefaultsCommandSection(String)`.  Returns a command section
    /// in the progDefaults autodoc (null pointer for null).
    fn get_prog_defaults_command_section(&self, command: &str) -> *mut Section {
        if PROG_DEFAULTS_AUTODOC.get().is_null() {
            // `synchronized (this)`: see the module comment.
            if PROG_DEFAULTS_AUTODOC.get().is_null() {
                // SAFETY: the factory owns the returned autodoc for the life of this
                // thread.
                match unsafe {
                    autodoc_factory::get_com_instance(Some(autodoc_factory::PROG_DEFAULTS))
                } {
                    Ok(autodoc) => PROG_DEFAULTS_AUTODOC.set(autodoc),
                    // `catch (final LockException e) {}`
                    Err(LogFileError::Lock(_)) => return std::ptr::null_mut(),
                    // `catch (final LogFileException | IOException e)`
                    Err(e) => {
                        // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
                        eprintln!("{}", e);
                        return std::ptr::null_mut();
                    }
                }
            }
        }
        let prog_defaults_autodoc = PROG_DEFAULTS_AUTODOC.get();
        if !prog_defaults_autodoc.is_null() {
            // SAFETY: owned by the factory's arena on this thread.
            return unsafe { (*prog_defaults_autodoc).get_section(Some("Program"), Some(command)) };
        }
        std::ptr::null_mut()
    }
}
