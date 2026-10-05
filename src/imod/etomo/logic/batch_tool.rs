//! `IMOD/Etomo/src/etomo/logic/BatchTool.java`.
//!
//! Static helpers for the batchruntomo dialogs: merging directive files into a batch
//! file, and moving values between directive files, fields and writable autodocs.
//!
//! **Representation.**  Only static members.  Java overloads carry suffixes naming
//! the parameter that distinguishes them.  The fields are event dispatch thread
//! objects (passed as trait objects); the autodocs are the autodoc registry's raw
//! pointers (see `storage/autodoc/autodoc_factory.rs`); a thrown
//! `FieldValidationFailedException` is the `Err` of the result.  The template value
//! maps are the dialogs' `Map<DirectiveDef, String>`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::dataset_tool;
use super::numeric_comparison_strategy;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::writable_autodoc::WritableAutodoc;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::storage::directive_file_interface::DirectiveFileInterface;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::name_value_pair_list::NameValuePairList;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::ui::boolean_field_interface::BooleanFieldInterface;
use crate::imod::etomo::ui::boolean_text_field_interface::BooleanTextFieldInterface;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
use crate::imod::etomo::ui::swing::abstract_radio_button_model::AbstractRadioButtonModel;
use crate::imod::etomo::ui::swing::radio_button::RadioButtonModel;
use crate::imod::etomo::ui::swing::radio_button_interface::EnumeratedTypeRef;
use crate::imod::etomo::ui::swing::text_efield_interface::TextEfieldInterface;
use crate::imod::etomo::ui::text_field_interface::TextFieldInterface;
use crate::imod::etomo::storage::directive_value::DirectiveValue;

/// The dialogs' `Map<DirectiveDef, String>` of template values.
pub type TemplateValues = HashMap<DirectiveDef, Option<String>>;

/// Java private static `debug`, initially false.
static DEBUG: AtomicBool = AtomicBool::new(false);

/// Java `setDebug(boolean)`.
pub fn set_debug(input: bool) {
    DEBUG.store(input, Ordering::Relaxed);
}

/// Java `isDebug()`.
pub fn is_debug() -> bool {
    DEBUG.load(Ordering::Relaxed)
}

/// `new NameValuePairList(AutodocFactory.getAutodocInstance(manager, file))`.
fn autodoc_list(
    manager: &'static dyn BaseManager,
    file: &std::path::Path,
) -> Result<NameValuePairList, LogFileError> {
    // SAFETY: the factory keeps the autodoc alive for the thread's lifetime.
    unsafe {
        let autodoc = autodoc_factory::get_autodoc_instance(Some(manager), Some(file))?;
        Ok(NameValuePairList::new_autodoc(autodoc))
    }
}

/// Java `catch (LogFileException | IOException e) { e.printStackTrace(); } catch
/// (LockException e) {}`.
fn report(e: LogFileError) {
    if !matches!(e, LogFileError::Lock(_)) {
        eprintln!("{}", e);
    }
}

/// Java `mergeTemplates(BaseManager, File[])`.  Combine templates.  Merge templates in
/// order of precedence.  Call this once per save.
///
/// Formula (`<` : merge when name is missing on left side - higher priority + lower
/// priority):
///
/// templates =  templateUser < templateSystem < templateScope < batchDefaults
pub fn merge_templates(
    manager: &'static dyn BaseManager,
    template_files: Option<&[Option<PathBuf>]>,
) -> Option<NameValuePairList> {
    // Merge templates together in order of decreasing priority (highest first)
    // templates = templateUser + templateSystem + templateScope
    let mut templates: Option<NameValuePairList> = None;
    if let Some(template_files) = template_files {
        for template_file in template_files.iter().rev() {
            if let Some(template_file) = template_file {
                match autodoc_list(manager, template_file) {
                    Ok(list) => match &mut templates {
                        None => templates = Some(list),
                        Some(templates) => templates.merge(Some(&list)),
                    },
                    Err(e) => report(e),
                }
            }
        }
    }
    // + batchDefaults
    let batch_default = file_type::CLASS
        .default_batch_run_tomo_autodoc
        .get_file(Some(manager), Some(AxisID::Only));
    if let Some(batch_default) = batch_default
        && batch_default.exists()
    {
        match autodoc_list(manager, &batch_default) {
            Ok(list) => match &mut templates {
                None => templates = Some(list),
                Some(templates) => templates.merge(Some(&list)),
            },
            Err(e) => report(e),
        }
    }
    templates
}

// Updates done

/// Java `createBatchFile(BaseManager, Autodoc, boolean, NameValuePairList,
/// NameValuePairList, Set<DirectiveDef>, NameValuePairList, NameValuePairList)`.
/// Combine dialog data with autodocs.  Data and autodocs are merged in order of
/// precedence.  Then templates are subtracted when they match both name and value.
/// Call for global dialog and each row.
///
/// Formula (`<` merge when name is missing on left side, `--` subtract from left side
/// when both name and value match, `-` subtract from left side when the directive
/// matches):
///
/// templates =  templateUser < templateSystem < templateScope < batchDefaults
///
/// If the advanced dialog has not been created:
/// directiveFiles = (batchFile - basidDirectives) < (startingBatchFile - basicDirectives) < templates
///
/// If the advanced dialog has been created:
/// directiveFiles = templates
///
/// batchState =  (dialogData < directiveFiles) -- templates
///
/// # Safety
/// `dialog_data` must point to a live `Autodoc`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_batch_file(
    _manager: &'static dyn BaseManager,
    dialog_data: *mut Autodoc,
    advanced_dialog_exists: bool,
    global_batch_list: Option<&NameValuePairList>,
    loaded_batch_list: Option<&mut NameValuePairList>,
    basic_directives: Option<&HashSet<DirectiveDef>>,
    advanced_starting_batch_list: Option<&NameValuePairList>,
    templates: Option<&NameValuePairList>,
) -> NameValuePairList {
    //
    // directiveFiles - represents a merging of all of the directives files.
    let mut directive_files: Option<NameValuePairList> = None;
    // Included the advanced directives by getting them from files. This is unnecessary
    // if the advanced dialog is open.
    if !advanced_dialog_exists {
        // Start with the advanced directives from the global batch list (if this is row
        // level). Or start with the loaded batch file if this is the global level.
        if let Some(global_batch_list) = global_batch_list {
            directive_files = Some(NameValuePairList::new_copy(Some(global_batch_list)));
        } else if let Some(loaded_batch_list) = loaded_batch_list {
            loaded_batch_list.subtract(basic_directives);
            directive_files = Some(NameValuePairList::new_copy(Some(loaded_batch_list)));
        }
        // The advanced part of the starting batch
        if let Some(advanced_starting_batch_list) = advanced_starting_batch_list {
            match &mut directive_files {
                None => {
                    directive_files =
                        Some(NameValuePairList::new_copy(Some(advanced_starting_batch_list)))
                }
                Some(directive_files) => directive_files.merge(Some(advanced_starting_batch_list)),
            }
        }
    }
    // templates
    match &mut directive_files {
        None => directive_files = Some(NameValuePairList::new_copy(templates)),
        Some(directive_files) => directive_files.merge(templates),
    }
    //
    // savebatchList - combine the dialog data with the directive files
    // merge directiveFiles
    let mut savebatch_list = unsafe { NameValuePairList::new_autodoc(dialog_data) };
    if let Some(directive_files) = &directive_files {
        savebatch_list.merge(Some(directive_files));
    }
    let Some(templates) = templates else {
        return savebatch_list;
    };
    // Substract the templates and batch default since these directives do not have to
    // be saved.
    savebatch_list.subtract_list(Some(templates), Some(&numeric_comparison_strategy::INSTANCE));
    savebatch_list
}

// private static final class NumericComparisonStrategy implements ComparisonStrategy {}

/// Java `setTextValue(TextFieldInterface, DirectiveFileInterface, boolean,
/// Map<DirectiveDef, String>)`.  Returns true if directiveFiles contains
/// directiveDef.
pub fn set_text_value(
    text_field_td: Option<&dyn TextFieldInterface>,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> bool {
    let Some(text_field_td) = text_field_td else {
        return false;
    };
    let directive_def = text_field_td.get_directive_def();
    if !directive_files.contains_template_ignore(directive_def, set_field_highlight_value, true) {
        return false;
    }
    let directive_value =
        directive_files.get_value_template_ignore(directive_def, set_field_highlight_value, true);
    let Some(directive_value) = directive_value else {
        return false;
    };
    let value = directive_value.get_value();
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(template_values, directive_def, value.as_deref());
        }
        text_field_td.set_field_highlight_string(value.as_deref());
    } else {
        text_field_td.set_value_string(value.as_deref());
    }
    true
}

/// Java `Map.put(directiveDef, value)` (a Java `HashMap` holds a null value).
fn put(template_values: &mut TemplateValues, directive_def: DirectiveDef, value: Option<&str>) {
    template_values.insert(directive_def, value.map(str::to_owned));
}

/// Java `setTextValue(TextEfieldInterface, DirectiveFileInterface, boolean,
/// Map<DirectiveDef, String>, boolean)`.  Returns the directiveValue if directiveFiles
/// contains directiveDef.
pub fn set_text_value_efield(
    text_field_td: Option<&dyn TextEfieldInterface>,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
    include_override: bool,
) -> Option<Box<dyn DirectiveValue>> {
    let text_field_td = text_field_td?;
    let directive_def = text_field_td.get_directive_def();
    if !directive_files.contains_template_ignore_override(
        directive_def,
        set_field_highlight_value,
        true,
        include_override,
    ) {
        return None;
    }
    let directive_value = directive_files.get_value_template_ignore_override(
        directive_def,
        set_field_highlight_value,
        true,
        include_override,
    )?;
    let value = directive_value.get_value();
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(template_values, directive_def, value.as_deref());
        }
        text_field_td.set_field_highlight(value.as_deref());
    } else {
        // !!!set override here if it was found in a batch file
        text_field_td.set_text(value.as_deref());
    }
    Some(directive_value)
}

/// Java `setTextValue(TextFieldInterface, DirectiveFileInterface, boolean,
/// Map<DirectiveDef, String>, AxisID)`.
pub fn set_text_value_axis(
    text_field_td: Option<&dyn TextFieldInterface>,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
    axis_id: Option<AxisID>,
) -> bool {
    let Some(text_field_td) = text_field_td else {
        return false;
    };
    let directive_def = text_field_td.get_directive_def();
    if !directive_files.contains_axis_template_ignore(
        directive_def,
        axis_id,
        set_field_highlight_value,
        true,
    ) {
        return false;
    }
    let directive_value = directive_files.get_value_axis_template_ignore(
        directive_def,
        axis_id,
        set_field_highlight_value,
        true,
    );
    let Some(directive_value) = directive_value else {
        return false;
    };
    let value = directive_value.get_value();
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(template_values, directive_def, value.as_deref());
        }
        text_field_td.set_field_highlight_string(value.as_deref());
    } else {
        text_field_td.set_value_string(value.as_deref());
    }
    true
}

/// Java `setTextValues(TextFieldInterface, TextFieldInterface, DirectiveFileInterface,
/// boolean, Map<DirectiveDef, String>)`.  Set text fields to directive array value.
/// Returns true if directiveFiles contains directiveDef.
pub fn set_text_values(
    text_field_ad0: Option<&dyn TextFieldInterface>,
    text_field_ad1: &dyn TextFieldInterface,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> bool {
    let Some(text_field_ad0) = text_field_ad0 else {
        return false;
    };
    let directive_def = text_field_ad0.get_directive_def();
    if !directive_files.contains_template_ignore(directive_def, set_field_highlight_value, true) {
        return false;
    }
    let directive_value =
        directive_files.get_value_template_ignore(directive_def, set_field_highlight_value, true);
    let Some(directive_value) = directive_value else {
        return false;
    };
    let value = directive_value.get_value().unwrap_or_default();
    let mut value0: Option<String> = None;
    let mut value1: Option<String> = None;
    // `value.split("\\s*,\\s*")`, which drops trailing empty strings.
    let value_array = crate::imod::etomo::util::utilities::java_lang_string_split(
        &value,
        &regex::Regex::new(r"\s*,\s*").unwrap(),
    );
    if !value_array.is_empty() {
        value0 = Some(value_array[0].clone());
        if value_array.len() > 1 {
            value1 = Some(value_array[1].clone());
        }
    }
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(template_values, directive_def, Some(&value));
        }
        text_field_ad0.set_field_highlight_string(value0.as_deref());
        text_field_ad1.set_field_highlight_string(value1.as_deref());
    } else {
        text_field_ad0.set_value_string(value0.as_deref());
        text_field_ad1.set_value_string(value1.as_deref());
    }
    true
}

/// Java `setTextValue(TextFieldInterface, String, String, boolean, Map<DirectiveDef,
/// String>)`.
pub fn set_text_value_derived(
    text_field_td: Option<&dyn TextFieldInterface>,
    directive_value: Option<&str>,
    derived_value: Option<&str>,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) {
    let Some(text_field_td) = text_field_td else {
        return;
    };
    let Some(directive_value) = directive_value else {
        return;
    };
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) =
            (template_values, text_field_td.get_directive_def())
        {
            put(template_values, directive_def, Some(directive_value));
        }
        if derived_value.is_some() {
            text_field_td.set_field_highlight_string(derived_value);
        }
    } else {
        text_field_td.set_value_string(derived_value);
    }
}

/// Java `setTextValue(TextFieldInterface, String, boolean, Map<DirectiveDef,
/// String>)`.
pub fn set_text_value_string(
    text_field_td: Option<&dyn TextFieldInterface>,
    directive_value: Option<&str>,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) {
    let Some(text_field_td) = text_field_td else {
        return;
    };
    let Some(directive_value) = directive_value else {
        return;
    };
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) =
            (template_values, text_field_td.get_directive_def())
        {
            put(template_values, directive_def, Some(directive_value));
        }
        text_field_td.set_field_highlight_string(Some(directive_value));
    } else {
        text_field_td.set_value_string(Some(directive_value));
    }
}

/// Java `setBooleanValue(BooleanFieldInterface, DirectiveFileInterface, boolean,
/// Map<DirectiveDef, String>)`.  Returns true if directiveFiles contains directiveDef.
pub fn set_boolean_value(
    boolean_field_bd: Option<&dyn BooleanFieldInterface>,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> bool {
    let Some(boolean_field_bd) = boolean_field_bd else {
        return false;
    };
    let directive_def = boolean_field_bd.get_directive_def();
    if !directive_files.contains_template_ignore(directive_def, set_field_highlight_value, true) {
        return false;
    }
    let selected =
        directive_files.is_value_template_ignore(directive_def, set_field_highlight_value, true);
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(
                template_values,
                directive_def,
                Some(&DirectiveDef::get_boolean_value_bool(selected)),
            );
        }
        boolean_field_bd.set_field_highlight_boolean(selected);
    } else {
        boolean_field_bd.set_value_boolean(selected);
    }
    true
}

/// Java `setBooleanValue(BooleanFieldInterface, DirectiveFileInterface, boolean,
/// boolean, Map<DirectiveDef, String>)`.  Returns true if directiveFiles contains
/// directiveDef.
pub fn set_boolean_value_missing(
    boolean_field_bd: Option<&dyn BooleanFieldInterface>,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    missing_field_sets_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> bool {
    let Some(boolean_field_bd) = boolean_field_bd else {
        return false;
    };
    let directive_def = boolean_field_bd.get_directive_def();
    if !directive_files.contains_template_ignore(directive_def, set_field_highlight_value, true) {
        if missing_field_sets_value {
            boolean_field_bd.set_value_boolean(false);
        }
        return false;
    }
    let selected =
        directive_files.is_value_template_ignore(directive_def, set_field_highlight_value, true);
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(
                template_values,
                directive_def,
                Some(&DirectiveDef::get_boolean_value_bool(selected)),
            );
        }
        boolean_field_bd.set_field_highlight_boolean(selected);
    } else {
        boolean_field_bd.set_value_boolean(selected);
    }
    true
}

/// Java `setBooleanValueFromSelectedText(BooleanFieldInterface, String,
/// DirectiveFileInterface, boolean, Map<DirectiveDef, String>)`.
pub fn set_boolean_value_from_selected_text(
    boolean_field_td: Option<&dyn BooleanFieldInterface>,
    selected_text: Option<&str>,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> bool {
    let Some(boolean_field_td) = boolean_field_td else {
        return false;
    };
    let directive_def = boolean_field_td.get_directive_def();
    if !directive_files.contains_template_ignore(directive_def, set_field_highlight_value, true) {
        return false;
    }
    let mut selected = false;
    let directive_value =
        directive_files.get_value_template_ignore(directive_def, set_field_highlight_value, true);
    let Some(directive_value) = directive_value else {
        return false;
    };
    let value = directive_value.get_value();
    if value.is_some() && value.as_deref() == selected_text {
        selected = true;
    }
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(template_values, directive_def, value.as_deref());
        }
        boolean_field_td.set_field_highlight_boolean(selected);
    } else {
        boolean_field_td.set_value_boolean(selected);
    }
    true
}

/// Java `setNegatedBooleanValue(BooleanFieldInterface, DirectiveFileInterface,
/// boolean, Map<DirectiveDef, String>)`.  Sets boolean field to the inverse of the
/// directive value.  Returns true if directiveFiles contains directiveDef.
pub fn set_negated_boolean_value(
    boolean_field_bd: Option<&dyn BooleanFieldInterface>,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> bool {
    let Some(boolean_field_bd) = boolean_field_bd else {
        return false;
    };
    let directive_def = boolean_field_bd.get_directive_def();
    if !directive_files.contains_template_ignore(directive_def, set_field_highlight_value, true) {
        return false;
    }
    let selected =
        directive_files.is_value_template_ignore(directive_def, set_field_highlight_value, true);
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(
                template_values,
                directive_def,
                Some(&DirectiveDef::get_boolean_value_bool(selected)),
            );
        }
        boolean_field_bd.set_field_highlight_boolean(!selected);
    } else {
        boolean_field_bd.set_value_boolean(!selected);
    }
    true
}

/// Java `setBooleanValue(BooleanFieldInterface, boolean, boolean)`.  Unable to add an
/// entry to templateValues, as the directive value is not known.
pub fn set_boolean_value_selected(
    boolean_field: Option<&dyn BooleanFieldInterface>,
    selected: bool,
    set_field_highlight_value: bool,
) {
    let Some(boolean_field) = boolean_field else {
        return;
    };
    if set_field_highlight_value {
        boolean_field.set_field_highlight_boolean(selected);
    } else {
        boolean_field.set_value_boolean(selected);
    }
}

/// Java `setBooleanValue(BooleanFieldInterface, boolean, boolean, boolean)`.  Unable to
/// add an entry to templateValues, as the directive value is not known.
pub fn set_boolean_value_selected_highlight(
    boolean_field: Option<&dyn BooleanFieldInterface>,
    selected: bool,
    highlight: bool,
    set_field_highlight_value: bool,
) {
    let Some(boolean_field) = boolean_field else {
        return;
    };
    if set_field_highlight_value {
        if highlight {
            boolean_field.set_field_highlight_boolean(selected);
        }
    } else {
        boolean_field.set_value_boolean(selected);
    }
}

/// Java `setBooleanValueIfEquals(BooleanFieldInterface, String, int, boolean)`.
/// Unable to add an entry to templateValues, as the directive value is not known.
/// Returns field exists and element0 == element1.
pub fn set_boolean_value_if_equals(
    boolean_field: Option<&dyn BooleanFieldInterface>,
    element0: Option<&str>,
    element1: i32,
    set_field_highlight_value: bool,
) -> bool {
    let Some(boolean_field) = boolean_field else {
        return false;
    };
    let mut selected = false;
    // `Double.parseDouble(element0)`: a null element0 throws NullPointerException in
    // Java (not caught); every caller passes a field's text, which is never null.
    if let Some(element0) = element0
        && let Ok(value) =
            crate::imod::etomo::r#type::const_etomo_number::java_lang_double_value_of(element0)
    {
        selected = value == element1 as f64;
        if set_field_highlight_value {
            boolean_field.set_field_highlight_boolean(selected);
        }
    }
    if !set_field_highlight_value {
        boolean_field.set_value_boolean(selected);
    }
    selected
}

/// Java `setBooleanValueFromText(BooleanFieldInterface, DirectiveFileInterface,
/// boolean, Map<DirectiveDef, String>)`.  Returns the directive value.
pub fn set_boolean_value_from_text(
    boolean_field_td: Option<&dyn BooleanFieldInterface>,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> Option<String> {
    let boolean_field_td = boolean_field_td?;
    let directive_def = boolean_field_td.get_directive_def();
    if !directive_files.contains_template_ignore_override(
        directive_def,
        set_field_highlight_value,
        true,
        true,
    ) {
        return None;
    }
    let directive_value =
        directive_files.get_value_template_ignore(directive_def, set_field_highlight_value, true);
    // If directive value is null then its an override because this is a string
    // directive.
    let mut value: Option<String> = None;
    if let Some(directive_value) = directive_value {
        value = directive_value.get_value();
    }
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def), Some(value)) =
            (template_values, directive_def, value.as_deref())
        {
            put(
                template_values,
                directive_def,
                Some(&DirectiveDef::get_boolean_value_string(Some(value))),
            );
        }
        boolean_field_td.set_field_highlight_boolean(value.is_some());
    }
    if !set_field_highlight_value {
        boolean_field_td.set_value_boolean(value.is_some());
    }
    value
}

/// Java `setBooleanValueFromUnselectedText(BooleanFieldInterface, String,
/// DirectiveFileInterface, boolean, Map<DirectiveDef, String>)`.  Returns the
/// directive value.
pub fn set_boolean_value_from_unselected_text(
    boolean_field_td: Option<&dyn BooleanFieldInterface>,
    unselected_text: &str,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> Option<String> {
    let boolean_field_td = boolean_field_td?;
    let directive_def = boolean_field_td.get_directive_def();
    if !directive_files.contains_template_ignore(directive_def, set_field_highlight_value, true) {
        return None;
    }
    let mut selected = false;
    let directive_value =
        directive_files.get_value_template_ignore(directive_def, set_field_highlight_value, true)?;
    // `value.equals(unselectedText)` throws NullPointerException for a null value in
    // Java; a null value is treated as different here.
    let value = directive_value.get_value();
    if value.as_deref() != Some(unselected_text) {
        selected = true;
    }
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(
                template_values,
                directive_def,
                Some(&DirectiveDef::get_boolean_value_string(value.as_deref())),
            );
        }
        boolean_field_td.set_field_highlight_boolean(selected);
    } else {
        boolean_field_td.set_value_boolean(selected);
    }
    value
}

/// Java `setBooleanTextValue(BooleanTextFieldInterface, DirectiveFileInterface,
/// boolean, Map<DirectiveDef, String>)`.  Returns true if directiveFiles contains
/// directive.
pub fn set_boolean_text_value(
    boolean_text_field_td: Option<&dyn BooleanTextFieldInterface>,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> bool {
    let Some(boolean_text_field_td) = boolean_text_field_td else {
        return false;
    };
    let directive_def = boolean_text_field_td.get_directive_def();
    if !directive_files.contains_template_ignore(directive_def, set_field_highlight_value, true) {
        return false;
    }
    let directive_value =
        directive_files.get_value_template_ignore(directive_def, set_field_highlight_value, true);
    let Some(directive_value) = directive_value else {
        return false;
    };
    let value = directive_value.get_value();
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(
                template_values,
                directive_def,
                Some(&DirectiveDef::get_boolean_value_string(value.as_deref())),
            );
        }
        boolean_text_field_td.set_field_highlight_string(value.as_deref());
        boolean_text_field_td.set_field_highlight_boolean(true);
    } else {
        boolean_text_field_td.set_value_string(value.as_deref());
        boolean_text_field_td.set_value_boolean(true);
    }
    true
}

/// Java `setBooleanTextValue(BooleanTextFieldInterface, String, boolean)`.  Returns
/// true if text is not empty.
pub fn set_boolean_text_value_string(
    boolean_text_field: Option<&dyn BooleanTextFieldInterface>,
    text: Option<&str>,
    set_field_highlight_value: bool,
) -> bool {
    let Some(boolean_text_field) = boolean_text_field else {
        return false;
    };
    let Some(text) = text else {
        return false;
    };
    if java_lang_string_matches_whitespace(text) {
        return false;
    }
    if set_field_highlight_value {
        boolean_text_field.set_field_highlight_string(Some(text));
        boolean_text_field.set_field_highlight_boolean(true);
    } else {
        boolean_text_field.set_value_string(Some(text));
        boolean_text_field.set_value_boolean(true);
    }
    true
}

/// Java `setBooleanAndTextValue(BooleanFieldInterface, TextFieldInterface,
/// DirectiveFileInterface, boolean, Map<DirectiveDef, String>)`.  Assumes the two
/// fields have the same directive.  Returns true if directiveFiles contains directive.
pub fn set_boolean_and_text_value(
    boolean_field_td: Option<&dyn BooleanFieldInterface>,
    text_field_td: &dyn TextFieldInterface,
    directive_files: &dyn DirectiveFileInterface,
    set_field_highlight_value: bool,
    template_values: Option<&mut TemplateValues>,
) -> bool {
    let Some(boolean_field_td) = boolean_field_td else {
        return false;
    };
    let directive_def = boolean_field_td.get_directive_def();
    if !directive_files.contains_template_ignore(directive_def, set_field_highlight_value, true) {
        return false;
    }
    let directive_value =
        directive_files.get_value_template_ignore(directive_def, set_field_highlight_value, true);
    let Some(directive_value) = directive_value else {
        return false;
    };
    let value = directive_value.get_value();
    if set_field_highlight_value {
        if let (Some(template_values), Some(directive_def)) = (template_values, directive_def) {
            put(template_values, directive_def, value.as_deref());
        }
        boolean_field_td.set_field_highlight_boolean(true);
        text_field_td.set_field_highlight_string(value.as_deref());
    } else {
        boolean_field_td.set_value_boolean(true);
        text_field_td.set_value_string(value.as_deref());
    }
    true
}

/// `autodoc.addNameValuePairAttribute(name, value)` on a writable autodoc pointer.
fn add_name_value_pair(autodoc: *mut Autodoc, name: &str, value: &str) {
    // SAFETY: the dialogs pass live autodocs from the autodoc factory.
    unsafe {
        if let Some(autodoc) = autodoc.as_mut() {
            autodoc.add_name_value_pair_attribute(Some(name), Some(value));
        }
    }
}

/// Java `saveTextToAutodoc(boolean, DirectiveDef, String, String, WritableAutodoc,
/// Map<DirectiveDef, String>, boolean)`.  `validateOnly`: nothing is done.
pub fn save_text_to_autodoc(
    enabled: bool,
    directive_def: Option<DirectiveDef>,
    text: Option<&str>,
    default_text: Option<&str>,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    if validate_only {
        return Ok(());
    }
    let mut text: Option<String> = text.map(str::to_owned);
    if text.is_none() && default_text.is_some() {
        text = default_text.map(str::to_owned);
    }
    if text.is_none() || !enabled || java_lang_string_matches_whitespace(text.as_deref().unwrap())
    {
        text = Some(String::new());
    }
    let text = text.unwrap();
    // Java dereferences directiveDef unguarded below; every caller passes one.
    let Some(directive_def) = directive_def else {
        return Ok(());
    };
    if let Some(template_values) = template_values {
        if text.is_empty() && !template_values.contains_key(&directive_def) {
            return Ok(());
        }
        let template_value = template_values.get(&directive_def).cloned().flatten();
        if Some(&text) == template_value.as_ref() {
            return Ok(());
        }
    }
    add_name_value_pair(autodoc, &directive_def.get_directive(), &text);
    Ok(())
}

/// Java `saveNotTextToAutodoc(DirectiveDef, EnumeratedType, WritableAutodoc,
/// Map<DirectiveDef, String>, boolean)`.  Save text which causes a function to for
/// performed.  This is only necessary when the template value is set to something
/// else.  Do not save if there is no template value.
pub fn save_not_text_to_autodoc(
    directive_def: DirectiveDef,
    enumerated_type: Option<&EnumeratedTypeRef>,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    if validate_only {
        return Ok(());
    }
    let mut template_value: Option<String> = None;
    if let Some(template_values) = template_values {
        if enumerated_type.is_none() || !template_values.contains_key(&directive_def) {
            return Ok(());
        }
        template_value = template_values.get(&directive_def).cloned().flatten();
    }
    // Java dereferences enumeratedType unguarded when templateValues is null.
    let Some(enumerated_type) = enumerated_type else {
        return Ok(());
    };
    let text = enumerated_type.to_string();
    if template_value.is_some_and(|template_value| template_value == text) {
        return Ok(());
    }
    add_name_value_pair(autodoc, &directive_def.get_directive(), &text);
    Ok(())
}

/// Java `saveTextToAutodoc(Field, String, WritableAutodoc, Map<DirectiveDef, String>,
/// boolean)`.
pub fn save_text_to_autodoc_field(
    field_td: Option<&dyn Field>,
    text: Option<&str>,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    let Some(field_td) = field_td else {
        return Ok(());
    };
    save_text_to_autodoc(
        field_td.is_enabled(),
        field_td.get_directive_def(),
        text,
        None,
        autodoc,
        template_values,
        validate_only,
    )
}

/// Java `saveTextToAutodoc(TextEfieldInterface, String, WritableAutodoc,
/// Map<DirectiveDef, String>, boolean)`.
pub fn save_text_to_autodoc_efield(
    field_td: Option<&dyn TextEfieldInterface>,
    text: Option<&str>,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    let Some(field_td) = field_td else {
        return Ok(());
    };
    save_text_to_autodoc(
        field_td.is_enabled(),
        field_td.get_directive_def(),
        text,
        None,
        autodoc,
        template_values,
        validate_only,
    )
}

/// Java `saveTextToAutodoc(TextEfieldInterface, String, boolean, boolean,
/// WritableAutodoc, Map<DirectiveDef, String>, boolean)`.
#[allow(clippy::too_many_arguments)]
pub fn save_text_to_autodoc_efield_override(
    field_td: &dyn TextEfieldInterface,
    text: Option<&str>,
    override_available: bool,
    override_: bool,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    if !override_available {
        save_text_to_autodoc(
            field_td.is_enabled(),
            field_td.get_directive_def(),
            text,
            None,
            autodoc,
            template_values,
            validate_only,
        )
    } else {
        save_text_to_autodoc_override(
            field_td.is_enabled(),
            field_td.get_directive_def(),
            text,
            override_,
            autodoc,
            template_values,
            validate_only,
        )
    }
}

/// Java `saveTextToAutodoc(boolean, DirectiveDef, String, boolean, WritableAutodoc,
/// Map<DirectiveDef, String>, boolean)`.  `override`: use this instead of text, and
/// always override if it is on.
pub fn save_text_to_autodoc_override(
    enabled: bool,
    directive_def: Option<DirectiveDef>,
    text: Option<&str>,
    override_: bool,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    if validate_only || !enabled {
        // Field is not in use
        return Ok(());
    }
    let text_value: String;
    if !override_ {
        match text {
            Some(value) if !java_lang_string_matches_whitespace(value) => {
                text_value = value.to_owned()
            }
            // Not override and no text - nothing to do
            _ => return Ok(()),
        }
    } else {
        // override
        text_value = String::new();
    }
    let text = text_value;
    let Some(directive_def) = directive_def else {
        return Ok(());
    };
    // Keep values that are the same as template values out of the output.
    if !override_
        && template_values.is_some_and(|template_values| {
            template_values.contains_key(&directive_def)
                && Some(&text) == template_values.get(&directive_def).cloned().flatten().as_ref()
        })
    {
        return Ok(());
    }
    add_name_value_pair(autodoc, &directive_def.get_directive(), &text);
    Ok(())
}

/// Java `saveBooleanTextToAutodoc(BooleanFieldInterface, String, WritableAutodoc,
/// Map<DirectiveDef, String>, boolean)`.
pub fn save_boolean_text_to_autodoc(
    boolean_field_td: Option<&dyn BooleanFieldInterface>,
    text: Option<&str>,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    let Some(boolean_field_td) = boolean_field_td else {
        return Ok(());
    };
    save_text_to_autodoc(
        boolean_field_td.is_enabled() && boolean_field_td.is_selected(),
        boolean_field_td.get_directive_def(),
        text,
        None,
        autodoc,
        template_values,
        validate_only,
    )
}

/// Java `saveBooleanTextToAutodoc(BooleanFieldInterface, String, WritableAutodoc,
/// boolean)`.
pub fn save_boolean_text_to_autodoc_no_template(
    boolean_field_td: Option<&dyn BooleanFieldInterface>,
    text: Option<&str>,
    autodoc: *mut Autodoc,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    let Some(boolean_field_td) = boolean_field_td else {
        return Ok(());
    };
    save_text_to_autodoc(
        boolean_field_td.is_enabled() && boolean_field_td.is_selected(),
        boolean_field_td.get_directive_def(),
        text,
        None,
        autodoc,
        None,
        validate_only,
    )
}

/// Java `saveBooleanTextToAutodoc(BooleanFieldInterface, WritableAutodoc, String,
/// String, boolean)`.
pub fn save_boolean_text_to_autodoc_selected(
    boolean_field_td: Option<&dyn BooleanFieldInterface>,
    autodoc: *mut Autodoc,
    selected_text: Option<&str>,
    unselected_text: Option<&str>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    let Some(boolean_field_td) = boolean_field_td else {
        return Ok(());
    };
    if boolean_field_td.is_selected() {
        save_text_to_autodoc(
            boolean_field_td.is_enabled(),
            boolean_field_td.get_directive_def(),
            selected_text,
            None,
            autodoc,
            None,
            validate_only,
        )
    } else {
        save_text_to_autodoc(
            boolean_field_td.is_enabled(),
            boolean_field_td.get_directive_def(),
            unselected_text,
            None,
            autodoc,
            None,
            validate_only,
        )
    }
}

/// Java `saveBooleanTextToAutodoc(BooleanFieldInterface, String, String,
/// WritableAutodoc, Map<DirectiveDef, String>, boolean)`.
pub fn save_boolean_text_to_autodoc_default(
    boolean_field_td: Option<&dyn BooleanFieldInterface>,
    text: Option<&str>,
    default_text: Option<&str>,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    let Some(boolean_field_td) = boolean_field_td else {
        return Ok(());
    };
    save_text_to_autodoc(
        boolean_field_td.is_enabled() && boolean_field_td.is_selected(),
        boolean_field_td.get_directive_def(),
        text,
        default_text,
        autodoc,
        template_values,
        validate_only,
    )
}

/// Java `saveBooleanTextToAutodoc(BooleanTextFieldInterface, WritableAutodoc, boolean,
/// FieldDisplayer, Map<DirectiveDef, String>, boolean)`.  `validateOnly`: save is not
/// done.
pub fn save_boolean_text_field_to_autodoc(
    boolean_text_field_td: Option<&dyn BooleanTextFieldInterface>,
    autodoc: *mut Autodoc,
    do_validation: bool,
    field_displayer: Option<Rc<dyn FieldDisplayer>>,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    let Some(boolean_text_field_td) = boolean_text_field_td else {
        return Ok(());
    };
    let enabled = boolean_text_field_td.is_enabled() && boolean_text_field_td.is_selected();
    let directive_def = boolean_text_field_td.get_directive_def();
    let text =
        boolean_text_field_td.get_text_boolean_field_displayer(do_validation, field_displayer)?;
    save_text_to_autodoc(
        enabled,
        directive_def,
        text.as_deref(),
        None,
        autodoc,
        template_values,
        validate_only,
    )
}

/// Java `saveTextToAutodoc(TextFieldInterface, WritableAutodoc, boolean,
/// FieldDisplayer, Map<DirectiveDef, String>, boolean)`.  `validateOnly`: save is not
/// done.
pub fn save_text_field_to_autodoc(
    text_field_td: Option<&dyn TextFieldInterface>,
    autodoc: *mut Autodoc,
    do_validation: bool,
    field_displayer: Option<Rc<dyn FieldDisplayer>>,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    let Some(text_field_td) = text_field_td else {
        return Ok(());
    };
    let enabled = text_field_td.is_enabled();
    let directive_def = text_field_td.get_directive_def();
    let text = text_field_td.get_text_boolean_field_displayer(do_validation, field_displayer)?;
    save_text_to_autodoc(
        enabled,
        directive_def,
        text.as_deref(),
        None,
        autodoc,
        template_values,
        validate_only,
    )
}

/// Java `saveTextToAutodoc(RadioButton.RadioButtonModel, WritableAutodoc,
/// Map<DirectiveDef, String>, boolean)`.  Do not override if no enumerated type is
/// retrieved.  Some radio buttons in the group may not have an enumerated type, and
/// will have to be handled differently.  Returns true if an enumerated type was
/// retrieved.
pub fn save_radio_button_model_to_autodoc(
    model_td: Option<&RadioButtonModel>,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<bool, FieldValidationFailedException> {
    let Some(model_td) = model_td else {
        return Ok(false);
    };
    let enumerated_type = model_td.get_enumerated_type();
    let Some(enumerated_type) = enumerated_type else {
        return Ok(false);
    };
    let mut text: Option<String> = None;
    let number = enumerated_type.get_value();
    if !number.is_null() {
        text = Some(number.to_string());
    }
    // Java dereferences the field unguarded; a model without a live button has none.
    if let Some(field) = model_td.get_field() {
        save_text_to_autodoc(
            field.is_enabled(),
            field.get_directive_def(),
            text.as_deref(),
            None,
            autodoc,
            template_values,
            validate_only,
        )?;
    }
    Ok(true)
}

/// Java `saveTextToAutodoc(boolean, DirectiveDef, EnumeratedType, WritableAutodoc,
/// Map<DirectiveDef, String>, boolean)`.  `validateOnly`: nothing is done.
pub fn save_enumerated_type_to_autodoc(
    enabled: bool,
    directive_def: Option<DirectiveDef>,
    enumerated_type: Option<&EnumeratedTypeRef>,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    let mut text: Option<String> = None;
    if let Some(enumerated_type) = enumerated_type {
        let number = enumerated_type.get_value();
        if !number.is_null() {
            text = Some(number.to_string());
        }
    }
    save_text_to_autodoc(
        enabled,
        directive_def,
        text.as_deref(),
        None,
        autodoc,
        template_values,
        validate_only,
    )
}

/// Java `saveBooleanToAutodoc(BooleanFieldInterface, WritableAutodoc,
/// Map<DirectiveDef, String>, boolean)`.  `validateOnly`: nothing is done.
pub fn save_boolean_to_autodoc(
    boolean_field_bd: Option<&dyn BooleanFieldInterface>,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    if validate_only {
        return Ok(());
    }
    let Some(boolean_field_bd) = boolean_field_bd else {
        return Ok(());
    };
    save_boolean_selected_to_autodoc(
        Some(boolean_field_bd),
        boolean_field_bd.is_enabled() && boolean_field_bd.is_selected(),
        autodoc,
        template_values,
        validate_only,
    )?;
    Ok(())
}

/// Java `saveBooleanToAutodoc(Field, boolean, WritableAutodoc, Map<DirectiveDef,
/// String>, boolean)`.  Returns true if field is selected.
pub fn save_boolean_selected_to_autodoc(
    field_bd: Option<&dyn Field>,
    selected: bool,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<bool, FieldValidationFailedException> {
    let Some(field_bd) = field_bd else {
        return Ok(false);
    };
    let directive_def = field_bd.get_directive_def();
    if let Some(template_values) = template_values {
        let contains = directive_def.is_some_and(|def| template_values.contains_key(&def));
        if !selected && !contains {
            return Ok(selected);
        }
        let template_value = DirectiveDef::convert_to_boolean(
            directive_def
                .and_then(|def| template_values.get(&def).cloned().flatten())
                .as_deref(),
        );
        if selected == template_value {
            return Ok(selected);
        }
    }
    if validate_only {
        return Ok(selected);
    }
    if let Some(directive_def) = directive_def {
        add_name_value_pair(
            autodoc,
            &directive_def.get_directive(),
            &DirectiveDef::get_boolean_value_bool(selected),
        );
    }
    Ok(selected)
}

/// Java `overrideInAutodoc(DirectiveDef, WritableAutodoc, Map<DirectiveDef, String>,
/// boolean)`.  `directiveDef` must be a text directive; `validateOnly`: nothing is done.
pub fn override_in_autodoc(
    directive_def: DirectiveDef,
    autodoc: *mut Autodoc,
    template_values: Option<&TemplateValues>,
    validate_only: bool,
) -> Result<(), FieldValidationFailedException> {
    if validate_only
        || template_values.is_some_and(|template_values| template_values.contains_key(&directive_def))
    {
        // not there, or already overridden.
        return Ok(());
    }
    add_name_value_pair(autodoc, &directive_def.get_directive(), "");
    Ok(())
}

/// Java `getModelFileName(BaseManager, FileType, String, boolean)`.
pub fn get_model_file_name(
    manager: &'static dyn BaseManager,
    file_type: Option<&FileType>,
    stack: &str,
    dual_axis: bool,
) -> Option<String> {
    let file_type = file_type?;
    file_type.get_file_name_with_axis_type(
        Some(manager),
        dataset_tool::get_dataset_name(Some(stack), dual_axis).as_deref(),
        Some(if dual_axis {
            AxisType::DualAxis
        } else {
            AxisType::SingleAxis
        }),
        None,
    )
}
