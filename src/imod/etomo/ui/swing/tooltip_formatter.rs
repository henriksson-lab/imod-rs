//! `IMOD/Etomo/src/etomo/ui/swing/TooltipFormatter.java`.
#![allow(dead_code)]

/// Java package-private `TooltipFormatter`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TooltipFormatter {
    debug: bool,
}

impl TooltipFormatter {
    pub const N_COLUMNS: usize = 60;
    pub const SPLIT_CHAR: char = ' ';
    pub const HTML_TAG: &'static str = "<html>";
    pub const LINE_BREAK_TAG: &'static str = "<br>";

    /// Java singleton `TooltipFormatter.INSTANCE`.
    pub fn instance() -> Self {
        Self::default()
    }

    /// Java `buildTooltip(String, String, String)`.
    pub fn build_tooltip(
        &self,
        unformatted_tooltip: Option<&str>,
        param_descr: Option<&str>,
        directive_descr: Option<&str>,
    ) -> Option<String> {
        if unformatted_tooltip.is_none() && param_descr.is_none() && directive_descr.is_none() {
            return None;
        }
        if param_descr.is_none() && directive_descr.is_none() {
            return unformatted_tooltip.map(str::to_owned);
        }
        let unformatted_tooltip = unformatted_tooltip.map(|text| {
            let text = text.trim();
            text.strip_suffix('.').unwrap_or(text).to_owned()
        });
        let descr = format!(
            "{}{}{}",
            param_descr.unwrap_or_default(),
            if param_descr.is_some() && directive_descr.is_some() {
                ", "
            } else {
                ""
            },
            directive_descr.unwrap_or_default()
        );
        Some(match unformatted_tooltip {
            None => format!("{descr}."),
            Some(text) => format!("{text} ({descr})."),
        })
    }

    /// Java `format(String)`.
    pub fn format(&self, raw_string: Option<&str>) -> Option<String> {
        let raw_string = raw_string?;
        let mut splitting = true;
        let mut html_format = String::from(Self::HTML_TAG);
        let raw_string_len = raw_string.len();
        if raw_string_len <= Self::N_COLUMNS {
            html_format.push_str(&self.convert_to_html(raw_string));
            splitting = false;
        }
        let mut idx_start = 0;
        while splitting {
            let bounds = idx_start + Self::N_COLUMNS + 1;
            if bounds > raw_string_len {
                html_format.push_str(&self.convert_to_html(&raw_string[idx_start..]));
                break;
            }
            let substring_space_index = raw_string[idx_start..bounds].rfind(Self::SPLIT_CHAR);
            let Some(substring_space_index) = substring_space_index else {
                let outside_space_index = raw_string[idx_start..]
                    .find(Self::SPLIT_CHAR)
                    .map(|index| idx_start + index);
                let Some(outside_space_index) = outside_space_index else {
                    html_format.push_str(&self.convert_to_html(&raw_string[idx_start..]));
                    break;
                };
                html_format
                    .push_str(&self.convert_to_html(&raw_string[idx_start..outside_space_index]));
                html_format.push_str(Self::LINE_BREAK_TAG);
                idx_start = outside_space_index + 1;
                continue;
            };
            let inside_space_index = substring_space_index + idx_start;
            html_format.push_str(&self.convert_to_html(&raw_string[idx_start..inside_space_index]));
            html_format.push_str(Self::LINE_BREAK_TAG);
            idx_start = inside_space_index + 1;
        }
        Some(html_format)
    }

    /// Java deprecated `formatForRegressionTest(String)`.
    pub fn format_for_regression_test(&self, raw_string: Option<&str>) -> Option<String> {
        let raw_string = raw_string?;
        let mut html_format = String::from(Self::HTML_TAG);
        let mut splitting = true;
        let mut idx_start = 0;
        while splitting {
            let idx_search = idx_start + Self::N_COLUMNS;
            if idx_search >= raw_string.len().saturating_sub(1) {
                html_format.push_str(&self.convert_to_html(&raw_string[idx_start..]));
                splitting = false;
            } else {
                let substring = &raw_string[idx_start..idx_search];
                let mut idx_stop = substring.rfind(Self::SPLIT_CHAR);
                if idx_stop.is_none() {
                    if raw_string[idx_start..].contains(Self::SPLIT_CHAR) {
                        idx_stop = Some(Self::N_COLUMNS);
                    } else {
                        html_format.push_str(&self.convert_to_html(&raw_string[idx_start..]));
                        splitting = false;
                    }
                }
                if let Some(idx_stop) = idx_stop {
                    html_format.push_str(
                        &self.convert_to_html(&raw_string[idx_start..idx_start + idx_stop]),
                    );
                    html_format.push_str(Self::LINE_BREAK_TAG);
                    idx_start += idx_stop + 1;
                }
            }
        }
        Some(html_format)
    }

    /// Java private `convertToHtml(String)`.
    fn convert_to_html(&self, raw_string: &str) -> String {
        if self.debug {
            println!("TooltipFormatter.convertToHtml:rawString={raw_string}");
        }
        let mut buffer = raw_string.to_owned();
        let mut index = raw_string.len();
        let mut html_mask = HtmlMask::new();
        html_mask.mask(Some(raw_string));
        while index > 0 {
            let less = buffer[..index].rfind('<');
            let greater = buffer[..index].rfind('>');
            let Some(found) = less.into_iter().chain(greater).max() else {
                break;
            };
            index = found;
            if !html_mask.is_masked(index) {
                let character = buffer.remove(index);
                if character == '<' {
                    buffer.insert_str(index, "<html>&lt");
                } else if character == '>' {
                    buffer.insert_str(index, "<html>&gt");
                }
            }
        }
        if self.debug {
            println!("TooltipFormatter.convertToHtml:buffer.toString()={buffer}");
        }
        buffer
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Java `isDebug()`.
    pub fn is_debug(&self) -> bool {
        self.debug
    }
}

/// Java private static final nested `HtmlMask`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct HtmlMask {
    tag_array: [&'static str; 2],
    mask_map: Option<Vec<bool>>,
}

impl HtmlMask {
    /// Java private `HtmlMask()`.
    fn new() -> Self {
        Self {
            tag_array: ["<b>", "</b>"],
            mask_map: None,
        }
    }

    /// Java private `mask(String)`.
    fn mask(&mut self, string: Option<&str>) {
        let Some(string) = string else {
            self.mask_map = None;
            return;
        };
        let string = string.to_lowercase();
        let mut mask_map = vec![false; string.len()];
        for tag in self.tag_array {
            let mut find = 0;
            while find < string.len() {
                let Some(found) = string[find..].find(tag).map(|index| find + index) else {
                    break;
                };
                for index in found..found + tag.len() {
                    mask_map[index] = true;
                }
                find = found + tag.len();
            }
        }
        self.mask_map = Some(mask_map);
    }

    /// Java private `isMasked(int)`.
    fn is_masked(&self, index: usize) -> bool {
        self.mask_map
            .as_ref()
            .and_then(|map| map.get(index))
            .copied()
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_format_preserves_but_escapes_other_angles() {
        let formatter = TooltipFormatter::instance();
        assert_eq!(
            formatter.format(Some("<b>x</b> < y > z")).as_deref(),
            Some("<html><b>x</b> <html>&lt y <html>&gt z")
        );
    }
    #[test]
    fn source_build_tooltip_strips_terminal_period() {
        assert_eq!(
            TooltipFormatter::instance()
                .build_tooltip(Some("Value."), Some("param"), Some("directive"))
                .as_deref(),
            Some("Value (param, directive).")
        );
    }
}
