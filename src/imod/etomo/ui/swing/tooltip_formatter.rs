//! `IMOD/Etomo/src/etomo/ui/swing/TooltipFormatter.java`.
//!
//! Formats tooltips as HTML, wrapping them at `N_COLUMNS`.  Java indexes strings by
//! UTF-16 code unit; this translation indexes by `char`, which is the same for every
//! character in the Basic Multilingual Plane (all eTomo tooltip text).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_trim;

/// Java `static final TooltipFormatter INSTANCE`.
pub static INSTANCE: TooltipFormatter = TooltipFormatter {
    debug: AtomicBool::new(false),
};

/// Java private `N_COLUMNS`.
const N_COLUMNS: usize = 60;
/// Java private `SPLIT_CHAR`.
const SPLIT_CHAR: char = ' ';
/// Java private `HTML_TAG`.
const HTML_TAG: &str = "<html>";
/// Java private `LINE_BREAK_TAG` (unused in the Java; `format` writes the literal).
#[allow(dead_code)]
const LINE_BREAK_TAG: &str = "<br>";

/// Java `TooltipFormatter` (a singleton; see [`INSTANCE`]).
pub struct TooltipFormatter {
    /// Java `debug`.
    debug: AtomicBool,
}

impl TooltipFormatter {
    /// Java `buildTooltip(String, String, String)`.  Build a tooltip out of unformatted
    /// text, and field and directive names.  Returns an unformatted tooltip.  Will
    /// return just the field and directive names if `unformatted_text` is null.
    /// Returns null if all parameters are null.
    pub fn build_tooltip(
        &self,
        unformatted_text: Option<&str>,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) -> Option<String> {
        if unformatted_text.is_none() && param_descr.is_none() && directive_descr.is_none() {
            return None;
        }
        if param_descr.is_none() && directive_descr.is_none() {
            return unformatted_text.map(str::to_owned);
        }
        // Temporarily remove the period to make room for the descriptions.
        let mut unformatted_text = unformatted_text.map(str::to_owned);
        if let Some(text) = unformatted_text.take() {
            let mut text = java_lang_string_trim(&text).to_owned();
            if text.ends_with('.') {
                text.pop();
            }
            unformatted_text = Some(text);
        }
        // Build description
        let descr = param_descr.unwrap_or("").to_string()
            + if param_descr.is_some() && directive_descr.is_some() {
                ", "
            } else {
                ""
            }
            + directive_descr.unwrap_or("");
        let Some(unformatted_text) = unformatted_text else {
            // Nothing else available. Return descriptions.
            return Some(descr + ".");
        };
        Some(unformatted_text + " (" + &descr + ")" + ".")
    }

    /// Java `format(String)`.  Format the raw string into an HTML string appropriate
    /// for tooltips.
    pub fn format(&self, raw_string: Option<&str>) -> Option<String> {
        let raw_string: Vec<char> = raw_string?.chars().collect();
        let mut splitting = true;
        // Is rawString long enough to split?
        let mut html_format = String::from(HTML_TAG);
        let raw_string_len = raw_string.len();
        if raw_string_len <= N_COLUMNS {
            // RawString fits in N_COLUMNS.
            html_format.push_str(&self.convert_to_html(&raw_string));
            splitting = false;
        }
        let mut idx_start = 0;
        while splitting {
            let bounds = idx_start + N_COLUMNS + 1;
            // Does the unformatted part of rawString fit in N_COLUMNS?
            if bounds > raw_string_len {
                html_format.push_str(&self.convert_to_html(&raw_string[idx_start..]));
                break;
            }
            // Find the last space in the unformatted part of rawString that fits in
            // N_COLUMNS.
            let substring_space_index = raw_string[idx_start..bounds]
                .iter()
                .rposition(|&c| c == SPLIT_CHAR);
            let Some(substring_space_index) = substring_space_index else {
                // No space found. Use the next available space outside of the bounds.
                let outside_space_index = raw_string[idx_start..]
                    .iter()
                    .position(|&c| c == SPLIT_CHAR)
                    .map(|index| index + idx_start);
                let Some(outside_space_index) = outside_space_index else {
                    // No spaces left in the unformatted part of rawString. Stop splitting.
                    html_format.push_str(&self.convert_to_html(&raw_string[idx_start..]));
                    break;
                };
                // Use the next available space outside of the bounds.
                html_format.push_str(
                    &(self.convert_to_html(&raw_string[idx_start..outside_space_index])
                        + "<br>"),
                );
                // Move to the unformatted part of rawString, skipping the space
                idx_start = outside_space_index + 1;
                continue;
            };
            // insideSpaceIndex was created from a substring. Adjust insideSpaceIndex to
            // rawString.
            let inside_space_index = substring_space_index + idx_start;
            // Split at insideSpaceIndex.
            html_format.push_str(
                &(self.convert_to_html(&raw_string[idx_start..inside_space_index]) + "<br>"),
            );
            idx_start = inside_space_index + 1;
        }
        Some(html_format)
    }

    /// Java `formatForRegressionTest(String)`, `@deprecated` because it loops forever
    /// for some inputs.
    ///
    /// Fixed in translation (`TooltipFormatter.java:143-148`): when the next
    /// `N_COLUMNS` characters hold no space but a space follows later, Java sets
    /// `idxStop = N_COLUMNS` and then neither appends nor advances `idxStart`, so the
    /// loop never ends.  Here that branch splits at the `N_COLUMNS` boundary, as the
    /// assignment evidently intended (append the first `N_COLUMNS` characters and a
    /// break, then continue after them).
    pub fn format_for_regression_test(&self, raw_string: Option<&str>) -> Option<String> {
        let raw_string: Vec<char> = raw_string?.chars().collect();
        let mut html_format = String::from("<html>");
        let mut splitting = true;
        let mut idx_start = 0usize;
        while splitting {
            let idx_search = idx_start + N_COLUMNS;
            // Are we past the end of the string
            if idx_search as i64 >= raw_string.len() as i64 - 1 {
                html_format.push_str(&self.convert_to_html(&raw_string[idx_start..]));
                splitting = false;
            } else {
                let sub_string = &raw_string[idx_start..idx_search];
                let idx_stop = sub_string.iter().rposition(|&c| c == ' ');
                // All one word!
                match idx_stop {
                    None => {
                        if raw_string[idx_start..].contains(&' ') {
                            let idx_stop = N_COLUMNS;
                            html_format.push_str(
                                &(self
                                    .convert_to_html(&raw_string[idx_start..idx_start + idx_stop])
                                    + "<br>"),
                            );
                            idx_start += idx_stop;
                        } else {
                            // No more spaces in this tooltip - place the rest on one line.
                            html_format.push_str(&self.convert_to_html(&raw_string[idx_start..]));
                            splitting = false;
                        }
                    }
                    Some(idx_stop) => {
                        html_format.push_str(
                            &(self.convert_to_html(&raw_string[idx_start..idx_start + idx_stop])
                                + "<br>"),
                        );
                        idx_start = idx_start + idx_stop + 1;
                    }
                }
            }
        }
        Some(html_format)
    }

    /// Java private `convertToHtml(String)`.  Convert ">" and "<" to the html
    /// versions, unless they are part of `<b>` or `</b>`.  Go backwards through the
    /// string so that htmlMask is not invalidated.
    fn convert_to_html(&self, raw_string: &[char]) -> String {
        let raw: String = raw_string.iter().collect();
        if self.debug.load(Ordering::Relaxed) {
            println!("TooltipFormatter.convertToHtml:rawString={}", raw);
        }
        let mut buffer: Vec<char> = raw_string.to_vec();
        let mut index: i64 = raw_string.len() as i64;
        let mut html_mask = HtmlMask::new();
        html_mask.mask(Some(&raw));
        while index >= 0 {
            // Java max(buffer.lastIndexOf("<", index - 1), buffer.lastIndexOf(">", index - 1)).
            let last_index_of = |target: char| -> i64 {
                let from = index - 1;
                if from < 0 {
                    return -1;
                }
                let from = (from as usize).min(buffer.len().saturating_sub(1));
                if buffer.is_empty() {
                    return -1;
                }
                buffer[..=from]
                    .iter()
                    .rposition(|&c| c == target)
                    .map_or(-1, |i| i as i64)
            };
            index = last_index_of('<').max(last_index_of('>'));
            if index != -1 && !html_mask.is_masked(index) {
                let c = buffer.remove(index as usize);
                if c == '<' {
                    buffer.splice(index as usize..index as usize, "<html>&lt".chars());
                } else if c == '>' {
                    buffer.splice(index as usize..index as usize, "<html>&gt".chars());
                }
            }
        }
        let buffer: String = buffer.into_iter().collect();
        if self.debug.load(Ordering::Relaxed) {
            println!("TooltipFormatter.convertToHtml:buffer.toString()={}", buffer);
        }
        buffer
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, debug: bool) {
        self.debug.store(debug, Ordering::Relaxed);
    }

    /// Java `isDebug()`.
    pub fn is_debug(&self) -> bool {
        self.debug.load(Ordering::Relaxed)
    }
}

/// Java private static nested `HtmlMask`.
struct HtmlMask {
    /// Java `tagArray`.  Use only lower case in tagArray.
    tag_array: [&'static str; 2],
    /// Java `maskMap`.
    mask_map: Option<Vec<bool>>,
}

impl HtmlMask {
    /// Java private `HtmlMask()`.
    fn new() -> HtmlMask {
        HtmlMask {
            tag_array: ["<b>", "</b>"],
            mask_map: None,
        }
    }

    /// Java private `mask(String)`.  maskMap is constructed to be the same length as
    /// string.  Put a true at each index which corresponds to a place in string which
    /// should be masked.  Not case sensitive.
    fn mask(&mut self, string: Option<&str>) {
        let Some(string) = string else {
            self.mask_map = None;
            return;
        };
        // Java toLowerCase keeps the length for these characters; lower-case per char.
        let string: Vec<char> = string
            .chars()
            .map(|c| c.to_lowercase().next().unwrap_or(c))
            .collect();
        let mut mask_map = vec![false; string.len()];
        let mut i_tag = 0;
        while i_tag < self.tag_array.len() {
            let tag: Vec<char> = self.tag_array[i_tag].chars().collect();
            let mut i_find: i64 = 0;
            while i_find != -1 && (i_find as usize) < string.len() {
                let from = i_find as usize;
                i_find = string[from..]
                    .windows(tag.len())
                    .position(|window| window == tag.as_slice())
                    .map_or(-1, |i| (i + from) as i64);
                if i_find != -1 {
                    for i in i_find as usize..i_find as usize + tag.len() {
                        mask_map[i] = true;
                    }
                    i_find += tag.len() as i64;
                }
            }
            i_tag += 1;
        }
        self.mask_map = Some(mask_map);
    }

    /// Java private `isMasked(int)`.
    fn is_masked(&self, index: i64) -> bool {
        match &self.mask_map {
            None => false,
            Some(mask_map) => {
                if index < 0 || index as usize >= mask_map.len() {
                    return false;
                }
                mask_map[index as usize]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_escaped_except_bold_tags() {
        assert_eq!(
            INSTANCE.format(Some("a <b>b</b> < c")).as_deref(),
            Some("<html>a <b>b</b> <html>&lt c")
        );
        let long = "word ".repeat(20);
        let formatted = INSTANCE.format(Some(&long)).unwrap();
        assert!(formatted.contains("<br>"));
    }
}
