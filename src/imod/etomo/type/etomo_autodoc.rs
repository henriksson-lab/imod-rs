//! `IMOD/Etomo/src/etomo/type/EtomoAutodoc.java`.
//!
//! Description:
//!
//! Copyright: Copyright 2005 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! Java's `private EtomoAutodoc() {}` only prevents instantiation of the utility class;
//! a Rust module needs no equivalent.  `debug` is a mutable static of the class, so it
//! is an `AtomicBool` here.
#![allow(dead_code)]

use super::const_etomo_number::ConstEtomoNumber;
use super::substitution_string::SubstitutionString;
use crate::imod::etomo::storage::autodoc::attribute::Attribute;
use crate::imod::etomo::storage::autodoc::read_only_attribute::ReadOnlyAttribute;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use crate::imod::etomo::ui::swing::token::{self, Token};
use crate::imod::etomo::util::primative_tokenizer::PrimativeTokenizer;
use std::sync::atomic::{AtomicBool, Ordering};

/// Java `HEADER_SECTION_NAME`.
pub const HEADER_SECTION_NAME: &str = "SectionHeader";
/// Java `FIELD_SECTION_NAME`.
pub const FIELD_SECTION_NAME: &str = "Field";
/// Java `REQUIRED_ATTRIBUTE_NAME`.
pub const REQUIRED_ATTRIBUTE_NAME: &str = "required";
/// Java `TYPE_ATTRIBUTE_NAME`.
pub const TYPE_ATTRIBUTE_NAME: &str = "type";
/// Java `SHORT_ATTRIBUTE_NAME`.
pub const SHORT_ATTRIBUTE_NAME: &str = "short";
/// Java `FORMAT_ATTRIBUTE_NAME`.
pub const FORMAT_ATTRIBUTE_NAME: &str = "format";
/// Java `USAGE_ATTRIBUTE_NAME`.
pub const USAGE_ATTRIBUTE_NAME: &str = "usage";
/// Java `MANPAGE_ATTRIBUTE_NAME`.
pub const MANPAGE_ATTRIBUTE_NAME: &str = "manpage";
/// Java `DEFAULT_KEY`.
pub const DEFAULT_KEY: &str = "default";
/// Java `BOOLEAN_TYPE`.
pub const BOOLEAN_TYPE: &str = "B";
/// Java `FLOAT_TYPE`.
pub const FLOAT_TYPE: &str = "F";
/// Java `INTEGER_TYPE`.
pub const INTEGER_TYPE: &str = "I";
/// Java `COMMENT_KEY`.
pub const COMMENT_KEY: &str = "comment";
/// Java `REQUIRED_TRUE_VALUE`.
pub const REQUIRED_TRUE_VALUE: i32 = 1;
/// Java `VAR_TAG`.
pub const VAR_TAG: char = '%';
/// Java `NEW_LINE_CHAR`.
pub const NEW_LINE_CHAR: char = '^';
/// Java private `TOOLTIP_ATTRIBUTE_NAME`.
const TOOLTIP_ATTRIBUTE_NAME: &str = "tooltip";
/// Java `DOUBLE_DASH_ATTRIBUTE_NAME`.
pub const DOUBLE_DASH_ATTRIBUTE_NAME: &str = "DoubleDashOptions";
/// Java private `FORMAT_CHAR`.
const FORMAT_CHAR: char = '\\';
/// Java private `FORMAT_STRINGS`.
const FORMAT_STRINGS: [&str; 3] = ["fI", "fB", "fR"];

/// Java private static `debug`, initialised to false.
static DEBUG: AtomicBool = AtomicBool::new(false);

/// Java `getTooltip(String, ReadOnlySection, boolean)`.
///
pub fn get_tooltip_add_source(
    autodoc_name: Option<&str>,
    section: &dyn ReadOnlySection,
    add_source: bool,
) -> Option<String> {
    let mut text: Option<String> = None;
    let mut attribute: *mut Attribute =
        unsafe { section.get_attribute(Some(TOOLTIP_ATTRIBUTE_NAME)) };
    if attribute.is_null() || {
        text = unsafe { (*attribute).get_multi_line_value() };
        text.is_none()
    } {
        attribute = unsafe { section.get_attribute(Some(USAGE_ATTRIBUTE_NAME)) };
    }
    if attribute.is_null() || {
        text = unsafe { (*attribute).get_multi_line_value() };
        text.is_none()
    } {
        attribute = unsafe { section.get_attribute(Some(COMMENT_KEY)) };
    }
    if attribute.is_null() || {
        text = unsafe { (*attribute).get_multi_line_value() };
        text.is_none()
    } {
        attribute = unsafe { section.get_attribute(Some(MANPAGE_ATTRIBUTE_NAME)) };
        if attribute.is_null() {
            // Java dereferences the null attribute here and throws; the only caller that
            // can see it, `getTooltip(String, ReadOnlySection, String)`, catches it.
            panic!("java.lang.NullPointerException");
        }
        text = unsafe { (*attribute).get_multi_line_value() };
    }
    if let Some(text_value) = text {
        let mut text_value = remove_formatting(Some(text_value.trim()))?;
        attribute = unsafe { section.get_attribute(Some(DEFAULT_KEY)) };
        if !attribute.is_null() {
            let default_value = unsafe { (*attribute).get_value() };
            if let Some(default_value) = default_value {
                text_value = SubstitutionString::substitute_variable(
                    Some(&text_value),
                    Some(DEFAULT_KEY),
                    Some(&default_value),
                )?;
            }
        }
        if add_source {
            let source = format!(
                "({})",
                match get_source_tooltip_string(autodoc_name, section) {
                    None => "null".to_string(),
                    Some(source) => source,
                }
            );
            if text_value.ends_with('.') {
                return Some(text_value[0..text_value.len() - 1].to_string() + " " + &source + ".");
            }
            return Some(text_value + " " + &source + ".");
        } else {
            return Some(text_value);
        }
    }
    None
}

/// Java `getSourceTooltipString(String, ReadOnlySection)`.
///
pub fn get_source_tooltip_string(
    autodoc_name: Option<&str>,
    section: &dyn ReadOnlySection,
) -> Option<String> {
    Some(format!(
        "{}: {}",
        autodoc_name.unwrap_or("null"),
        match ReadOnlySectionList::get_name(section) {
            None => "null".to_string(),
            Some(name) => name,
        }
    ))
}

/// Java `getTooltip(String, ReadOnlySection, String)`.
///
pub fn get_tooltip_enum_value_name(
    autodoc_name: Option<&str>,
    section: &dyn ReadOnlySection,
    enum_value_name: Option<&str>,
) -> Option<String> {
    // Java's `try`/`catch (NullPointerException)`: any of the three `getAttribute`
    // results may be null, and the catch falls through to the three-argument overload.
    let enum_attribute = unsafe { section.get_attribute(Some("enum")) };
    let enum_tooltip = if enum_attribute.is_null() {
        None
    } else {
        let value_attribute = unsafe { (*enum_attribute).get_attribute_by_name(enum_value_name) };
        if value_attribute.is_null() {
            None
        } else {
            let tooltip_attribute =
                unsafe { (*value_attribute).get_attribute_by_name(Some(TOOLTIP_ATTRIBUTE_NAME)) };
            if tooltip_attribute.is_null() {
                None
            } else {
                Some(unsafe { (*tooltip_attribute).get_multi_line_value() })
            }
        }
    };
    if let Some(enum_tooltip) = enum_tooltip {
        if let Some(enum_tooltip) = enum_tooltip {
            let enum_tooltip = remove_formatting(Some(enum_tooltip.trim()))?;
            let source = format!(
                "({}:  {} {})",
                autodoc_name.unwrap_or("null"),
                match ReadOnlySectionList::get_name(section) {
                    None => "null".to_string(),
                    Some(name) => name,
                },
                enum_value_name.unwrap_or("null")
            );
            if enum_tooltip.ends_with('.') {
                return Some(
                    enum_tooltip[0..enum_tooltip.len() - 1].to_string() + " " + &source + ".",
                );
            }
            return Some(enum_tooltip + " " + &source + ".");
        }
        return get_tooltip_add_source(autodoc_name, section, true);
    }
    get_tooltip_add_source(autodoc_name, section, true)
}

/// Java `getTooltip(ReadOnlyAutodoc, String)`.
///
/// # Safety
/// `autodoc` must be null or point to a live `ReadOnlyAutodoc`.
pub unsafe fn get_tooltip(
    autodoc: *const dyn ReadOnlyAutodoc,
    field_name: Option<&str>,
) -> Option<String> {
    if autodoc.is_null() || field_name.is_none() {
        return None;
    }
    let autodoc: &dyn ReadOnlyAutodoc = unsafe { &*autodoc };
    let section = autodoc.get_section(Some(FIELD_SECTION_NAME), field_name);
    if section.is_null() {
        if DEBUG.load(Ordering::Relaxed) {
            println!("EtomoAutodoc.getTooltip:section is null");
        }
        return None;
    }
    get_tooltip_add_source(
        Some(&autodoc.get_autodoc_name()),
        unsafe { &*section },
        true,
    )
}

/// Java `getUnformattedTooltip(ReadOnlyAutodoc, String)`.
///
/// # Safety
/// See `get_tooltip`.
pub unsafe fn get_unformatted_tooltip(
    autodoc: *const dyn ReadOnlyAutodoc,
    field_name: Option<&str>,
) -> Option<String> {
    if autodoc.is_null() || field_name.is_none() {
        return None;
    }
    let autodoc: &dyn ReadOnlyAutodoc = unsafe { &*autodoc };
    let section = autodoc.get_section(Some(FIELD_SECTION_NAME), field_name);
    if section.is_null() {
        if DEBUG.load(Ordering::Relaxed) {
            println!("EtomoAutodoc.getTooltip:section is null");
        }
        return None;
    }
    get_tooltip_add_source(
        Some(&autodoc.get_autodoc_name()),
        unsafe { &*section },
        false,
    )
}

/// Java `getSourceTooltipString(ReadOnlyAutodoc, String)`.
///
/// # Safety
/// See `get_tooltip`.
pub unsafe fn get_source_tooltip_string_autodoc(
    autodoc: *const dyn ReadOnlyAutodoc,
    field_name: Option<&str>,
) -> Option<String> {
    if autodoc.is_null() || field_name.is_none() {
        return None;
    }
    let autodoc: &dyn ReadOnlyAutodoc = unsafe { &*autodoc };
    let section = autodoc.get_section(Some(FIELD_SECTION_NAME), field_name);
    if section.is_null() {
        if DEBUG.load(Ordering::Relaxed) {
            println!("EtomoAutodoc.getSourceTooltipString:section is null");
        }
        return None;
    }
    get_source_tooltip_string(Some(&autodoc.get_autodoc_name()), unsafe { &*section })
}

/// Java `getTooltip(ReadOnlyAutodoc, String, boolean)`.
///
/// # Safety
/// See `get_tooltip`.
pub unsafe fn get_tooltip_autodoc_add_source(
    autodoc: *const dyn ReadOnlyAutodoc,
    field_name: Option<&str>,
    add_source: bool,
) -> Option<String> {
    if autodoc.is_null() || field_name.is_none() {
        return None;
    }
    let autodoc: &dyn ReadOnlyAutodoc = unsafe { &*autodoc };
    let section = autodoc.get_section(Some(FIELD_SECTION_NAME), field_name);
    if section.is_null() {
        if DEBUG.load(Ordering::Relaxed) {
            println!("EtomoAutodoc.getTooltip:section is null");
        }
        return None;
    }
    get_tooltip_add_source(
        Some(&autodoc.get_autodoc_name()),
        unsafe { &*section },
        add_source,
    )
}

/// Java `removeFormatting(String)`.  Removes formatting strings.
pub fn remove_formatting(value: Option<&str>) -> Option<String> {
    let value = value?;
    let mut tokenizer =
        PrimativeTokenizer::get_string_instance(value, DEBUG.load(Ordering::Relaxed));
    let mut tooltip = String::new();
    let mut token: *mut Token = std::ptr::null_mut();
    // Java's `catch (LogFileException)` prints the stack trace and returns `value`, and
    // its `catch (IOException | LockException)` returns `value`.
    if let Err(e) = tokenizer.initialize() {
        if let crate::imod::etomo::storage::log_file::LogFileError::LogFile(_) = e {
            // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
            eprintln!("{}", e);
        }
        return Some(value.to_string());
    }
    // Remove:
    // ^ followed by whitespace at start of line
    // \fB
    // \fI
    // \fR
    // Replace EOL with a space
    let mut eol = true;
    let mut new_line = false;
    let mut format = false;
    loop {
        token = unsafe { tokenizer.next(token) };
        if unsafe { (*token).is(token::Type::Eof) } {
            break;
        }
        let mut token_value = unsafe { (*token).get_value() }.map(|value| value.to_string());
        //
        // Process booleans from the previous iteration.
        //
        // Remove indent
        if new_line {
            new_line = false;
            // Remove indent.
            if unsafe { (*token).is(token::Type::Whitespace) } {
                token_value = Some("".to_string());
            }
        }
        // Remove new line character
        if eol {
            eol = false;
            // Remove the formatting character ('^' at the beginning of the line).
            if unsafe { (*token).equals_type_and_char(token::Type::Symbol, NEW_LINE_CHAR as u16) } {
                new_line = true;
                token_value = Some("".to_string());
            }
        }
        // Remove specific formatting string
        if format {
            format = false;
            let mut found = false;
            // Look for formatting string
            if unsafe { (*token).is(token::Type::Alphanum) } {
                let string = unsafe { (*token).get_value() }.map(|value| value.to_string());
                if let Some(string) = string {
                    for format_string in FORMAT_STRINGS.iter() {
                        // If specific formatting string is found, then omit it.
                        if string.starts_with(format_string) {
                            found = true;
                            // Token may contain more then the formatting string.
                            let len = format_string.len();
                            if string.len() > len {
                                token_value = Some(string[len..].to_string());
                            } else {
                                token_value = Some("".to_string());
                            }
                        }
                    }
                }
            }
            if !found {
                // Failed to find a specific formatting string - add back the "\" from the
                // previous iteration, which was omitted in case it was part of a
                // formatting string.
                tooltip.push(FORMAT_CHAR);
            }
        }
        //
        // Finished processing booleans from the previous iteration.
        //
        // Convert end of line to a space.
        if unsafe { (*token).is(token::Type::Eol) } {
            eol = true;
            token_value = Some(" ".to_string());
        }
        // Remove formatting string
        if unsafe { (*token).equals_type_and_char(token::Type::Symbol, FORMAT_CHAR as u16) } {
            format = true;
            // Omit "\", in case it is part of a specific formatting string.
            token_value = Some("".to_string());
        }
        // `StringBuilder.append(String)` renders a null argument as "null".
        tooltip.push_str(token_value.as_deref().unwrap_or("null"));
    }
    if !token.is_null() {
        drop(unsafe { Box::from_raw(token) });
    }
    Some(tooltip)
}

/// Java `format(String)`.  Uses the '^' formatting to format the value.
pub fn format(value: Option<&str>) -> Option<Vec<String>> {
    let value = value?;
    let mut tokenizer =
        PrimativeTokenizer::get_string_instance(value, DEBUG.load(Ordering::Relaxed));
    let mut list: Vec<String> = Vec::new();
    let mut buffer = String::new();
    let mut start_of_line = true;
    let mut first_token = true;
    let mut token: *mut Token = std::ptr::null_mut();
    // Java's `catch (LogFileException)` prints the stack trace and returns
    // `new String[] { value }`, and its `catch (IOException | LockException)` returns
    // the same.
    if let Err(e) = tokenizer.initialize() {
        if let crate::imod::etomo::storage::log_file::LogFileError::LogFile(_) = e {
            // `e.printStackTrace()`; see etomo/util/stack_trace.rs.
            eprintln!("{}", e);
        }
        return Some(vec![value.to_string()]);
    }
    token = unsafe { tokenizer.next(token) };
    while !token.is_null() && !unsafe { (*token).is(token::Type::Eof) } {
        if start_of_line {
            start_of_line = false;
            // Handle new-line character ('^' at the start of the line).
            if unsafe { (*token).equals_type_and_char(token::Type::Symbol, NEW_LINE_CHAR as u16) } {
                list.push(buffer.clone());
                buffer = String::new();
                start_of_line = false;
                token = unsafe { tokenizer.next(token) };
                continue;
            } else if !first_token {
                // wasn't really the end of the line so convert EOL to a space.
                buffer.push(' ');
            } else {
                first_token = false;
            }
            // still need to process the current token
        }
        if unsafe { (*token).is(token::Type::Eol) } {
            // Wait to convert end-of-line to a space. If the next token is '^', then this
            // really is the end of a line.
            start_of_line = true;
        } else {
            buffer.push_str(unsafe { (*token).get_value() }.unwrap_or("null"));
            start_of_line = false;
        }
        token = unsafe { tokenizer.next(token) };
    }
    if !token.is_null() {
        drop(unsafe { Box::from_raw(token) });
    }
    if !buffer.is_empty() {
        list.push(buffer);
    }
    if list.is_empty() {
        return Some(Vec::new());
    }
    if list.len() == 1 {
        return Some(vec![list[0].clone()]);
    }
    Some(list)
}

/// Java `getTooltip(String, ReadOnlySection, ConstEtomoNumber)`.
///
pub fn get_tooltip_const_etomo_number(
    autodoc_name: Option<&str>,
    section: &dyn ReadOnlySection,
    enum_value_name: &ConstEtomoNumber,
) -> Option<String> {
    get_tooltip_enum_value_name(autodoc_name, section, Some(&enum_value_name.to_string()))
}

/// Java `getTooltip(String, ReadOnlySection, int)`.
///
pub fn get_tooltip_int(
    autodoc_name: Option<&str>,
    section: &dyn ReadOnlySection,
    enum_value_name: i32,
) -> Option<String> {
    get_tooltip_enum_value_name(autodoc_name, section, Some(&enum_value_name.to_string()))
}

/// Java `setDebug(boolean)`.
pub fn set_debug(debug: bool) {
    DEBUG.store(debug, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_formatting_preserves_plain_tooltip_text() {
        assert_eq!(
            remove_formatting(Some("plain tooltip text")),
            Some("plain tooltip text".to_string())
        );
    }
}
