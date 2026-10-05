//! `IMOD/Etomo/src/etomo/util/Bracket.java`.
//!
//! Finds a bracket at the beginning and end of a string.  Java's three static
//! instances, compared by identity, are the three `pub static` items; the class is
//! handed around as `&'static Bracket` and compared with `std::ptr::eq`.

/// Java `public final class Bracket`.
#[derive(Debug)]
pub struct Bracket {
    /// Java private final `open`.
    open: char,
    /// Java private final `close`.
    close: char,
    /// Java private final `descr`.
    descr: &'static str,
}

/// Java `CURLY = new Bracket('{', '}', "curly braces")`.
pub static CURLY: Bracket = Bracket::new('{', '}', "curly braces");
/// Java `SQUARE = new Bracket('[', ']', "square brackets")`.
pub static SQUARE: Bracket = Bracket::new('[', ']', "square brackets");
/// Java `PARENTHESIS = new Bracket('(', ')', "parenthesis")`.
pub static PARENTHESIS: Bracket = Bracket::new('(', ')', "parenthesis");

impl Bracket {
    /// Java private `Bracket(Character, Character, String)`.
    const fn new(open: char, close: char, descr: &'static str) -> Bracket {
        Bracket { open, close, descr }
    }

    /// Java `getOpenInstance(Character)`.
    pub fn get_open_instance_character(text: Option<char>) -> Option<&'static Bracket> {
        let text = text?;
        Bracket::get_instance(Some(&text.to_string()), true)
    }

    /// Java `getCloseInstance(Character)`.
    pub fn get_close_instance_character(text: Option<char>) -> Option<&'static Bracket> {
        let text = text?;
        Bracket::get_instance(Some(&text.to_string()), false)
    }

    /// Java `getOpenInstance(String)`.
    pub fn get_open_instance_string(text: Option<&str>) -> Option<&'static Bracket> {
        Bracket::get_instance(text, true)
    }

    /// Java `getCloseInstance(String)`.
    pub fn get_close_instance_string(text: Option<&str>) -> Option<&'static Bracket> {
        Bracket::get_instance(text, false)
    }

    /// Java package-private static `getInstance(String, boolean)`.  Return the
    /// instance that matches either the open or close bracket in the text.  Return
    /// null if the text doesn't start/end with a bracket.
    pub fn get_instance(text: Option<&str>, open: bool) -> Option<&'static Bracket> {
        let text = text?;
        // Java `String.trim()`: strips code units <= ' '.
        let text: Vec<char> = text
            .trim_matches(|c: char| (c as u32) <= 0x20)
            .chars()
            .collect();
        let size = text.len();
        if size == 0 {
            return None;
        }
        let mut index = 0;
        if !open {
            index = size - 1;
        }
        let possible_bracket = text[index];
        if open {
            if possible_bracket == CURLY.open {
                return Some(&CURLY);
            }
            if possible_bracket == SQUARE.open {
                return Some(&SQUARE);
            }
            if possible_bracket == PARENTHESIS.open {
                return Some(&PARENTHESIS);
            }
        } else {
            if possible_bracket == CURLY.close {
                return Some(&CURLY);
            }
            if possible_bracket == SQUARE.close {
                return Some(&SQUARE);
            }
            if possible_bracket == PARENTHESIS.close {
                return Some(&PARENTHESIS);
            }
        }
        None
    }

    /// Java `getOpen()`.
    pub fn get_open(&self) -> String {
        self.open.to_string()
    }

    /// Java `getClose()`.
    pub fn get_close(&self) -> String {
        self.close.to_string()
    }

    /// Java `getEmpty()`.
    pub fn get_empty(&self) -> String {
        self.get_open() + &self.get_close()
    }

    /// Java `getDescr()`.
    pub fn get_descr(&self) -> String {
        format!(
            "{} and {} ({})",
            self.get_open(),
            self.get_close(),
            self.descr
        )
    }
}

/// Java `toString()`.
impl std::fmt::Display for Bracket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.get_descr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_brackets_at_either_end() {
        assert!(std::ptr::eq(
            Bracket::get_open_instance_string(Some("  {a")).unwrap(),
            &CURLY
        ));
        assert!(std::ptr::eq(
            Bracket::get_close_instance_string(Some("a] ")).unwrap(),
            &SQUARE
        ));
        assert!(Bracket::get_open_instance_string(Some("a")).is_none());
        assert_eq!(CURLY.get_descr(), "{ and } (curly braces)");
    }
}
