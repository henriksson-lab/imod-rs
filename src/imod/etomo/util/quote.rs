//! `IMOD/Etomo/src/etomo/util/Quote.java`.
//!
//! Finds a quote at the beginning and end of a string.  Can find quotes inside
//! brackets.  Java's three static instances, compared by identity, are the three
//! `pub static` items, handed around as `&'static Quote`.

use super::bracket::Bracket;

/// Java `public class Quote`.
#[derive(Debug)]
pub struct Quote {
    /// Java private final `quote`.
    quote: char,
    /// Java private final `descr`.
    descr: &'static str,
}

/// Java `SINGLE = new Quote('\'', "single quotes")`.
pub static SINGLE: Quote = Quote::new('\'', "single quotes");
/// Java `DOUBLE = new Quote('"', "double quotes")`.
pub static DOUBLE: Quote = Quote::new('"', "double quotes");
/// Java `BACK_QUOTE = new Quote('`', "back quotes")`.
pub static BACK_QUOTE: Quote = Quote::new('`', "back quotes");

impl Quote {
    /// Java private `Quote(Character, String)`.
    const fn new(quote: char, descr: &'static str) -> Quote {
        Quote { quote, descr }
    }

    /// Java `getInstance(Character)`.
    pub fn get_instance_character(symbol: Option<char>) -> Option<&'static Quote> {
        let symbol = symbol?;
        if SINGLE.quote == symbol {
            return Some(&SINGLE);
        }
        if DOUBLE.quote == symbol {
            return Some(&DOUBLE);
        }
        if BACK_QUOTE.quote == symbol {
            return Some(&BACK_QUOTE);
        }
        None
    }

    /// Java `getLeftInstance(String, boolean)`.
    pub fn get_left_instance(
        text: Option<&str>,
        may_be_inside_bracket: bool,
    ) -> Option<&'static Quote> {
        Quote::get_instance(text, true, may_be_inside_bracket)
    }

    /// Java `getRightInstance(String, boolean)`.
    pub fn get_right_instance(
        text: Option<&str>,
        may_be_inside_bracket: bool,
    ) -> Option<&'static Quote> {
        Quote::get_instance(text, false, may_be_inside_bracket)
    }

    /// Java private static `getInstance(String, boolean, boolean)`.  Return the
    /// instance that matches either the opening or closing quote mark in the text.
    /// Return null if the text doesn't start/end with a quote mark.
    fn get_instance(
        text: Option<&str>,
        left_quote: bool,
        may_be_inside_bracket: bool,
    ) -> Option<&'static Quote> {
        let text = text?;
        // Java `String.trim()`: strips code units <= ' '.
        let text = text.trim_matches(|c: char| (c as u32) <= 0x20);
        let chars: Vec<char> = text.chars().collect();
        let size = chars.len() as i32;
        let mut bracketed = false;
        if may_be_inside_bracket {
            bracketed = Bracket::get_instance(Some(text), left_quote).is_some();
        }
        let bracket_size = if bracketed { 1 } else { 0 };
        let mut min_size = 1 + bracket_size;
        let mut index = bracket_size;
        if !left_quote {
            min_size += 1;
            index = size - 1 - bracket_size;
        }
        if size < min_size {
            return None;
        }
        let possible_quote = chars[index as usize];
        if possible_quote == SINGLE.quote {
            return Some(&SINGLE);
        }
        if possible_quote == DOUBLE.quote {
            return Some(&DOUBLE);
        }
        if possible_quote == BACK_QUOTE.quote {
            return Some(&BACK_QUOTE);
        }
        None
    }

    /// Java `getDescr()`.
    pub fn get_descr(&self) -> String {
        format!("{}...{} ({})", self.quote, self.quote, self.descr)
    }
}

/// Java `toString()`.
impl std::fmt::Display for Quote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.quote)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_quotes_inside_brackets() {
        assert!(std::ptr::eq(
            Quote::get_left_instance(Some("{'a'}"), true).unwrap(),
            &SINGLE
        ));
        assert!(std::ptr::eq(
            Quote::get_right_instance(Some("\"a\""), false).unwrap(),
            &DOUBLE
        ));
        assert!(Quote::get_right_instance(Some("'"), false).is_none());
        assert_eq!(SINGLE.get_descr(), "'...' (single quotes)");
    }
}
