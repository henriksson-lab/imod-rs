//! `IMOD/Etomo/src/etomo/ui/swing/ColoredStateText.java`.
//!
//! A list of states, each with a colour and optionally a label, one of which is
//! selected.  A plain value object held by `ProcessControlPanel`.

use crate::imod::etomo::jdk::Color;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `class ColoredStateText`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColoredStateText {
    /// Java `labels` (null when constructed from colours only).
    labels: Option<Vec<String>>,
    /// Java `colors`.
    colors: Vec<Color>,
    /// Java `nItems`.
    n_items: i32,
    /// Java `currentSelected`.
    current_selected: i32,
}

impl ColoredStateText {
    /// Java `ColoredStateText(String[], Color[]) throws InvalidParameterException`.
    pub fn new_string_array_color_array(
        labels: &[&str],
        colors: &[Color],
    ) -> Result<ColoredStateText, InvalidParameterException> {
        let n_items = labels.len() as i32;
        if n_items != colors.len() as i32 {
            return Err(InvalidParameterException::new(
                "The length of the labels and colors arrays do not match",
            ));
        }
        Ok(ColoredStateText {
            labels: Some(labels.iter().map(|label| (*label).to_owned()).collect()),
            colors: colors.to_vec(),
            n_items,
            current_selected: -1,
        })
    }

    /// Java `ColoredStateText(Color[])`.
    pub fn new_color_array(colors: &[Color]) -> ColoredStateText {
        ColoredStateText {
            labels: None,
            colors: colors.to_vec(),
            n_items: colors.len() as i32,
            current_selected: -1,
        }
    }

    /// Java `setSelected(int) throws InvalidParameterException`.
    ///
    /// Upstream bug fixed in translation (`ColoredStateText.java:62`): the Java test is
    /// `index < 0 & index >= nItems`, which can never be true, so an out-of-range index
    /// was accepted and the next `getSelectedText`/`getSelectedColor` threw
    /// `ArrayIndexOutOfBoundsException`.  The evident intent (the message says "Index
    /// out of range") is `||`, which is what this tests.
    pub fn set_selected(&mut self, index: i32) -> Result<(), InvalidParameterException> {
        if index < 0 || index >= self.n_items {
            return Err(InvalidParameterException::new(&format!(
                "Index out of range, nItems: {} index: {}",
                self.n_items, index
            )));
        }
        self.current_selected = index;
        Ok(())
    }

    /// Java `getSelected()`.
    pub fn get_selected(&self) -> i32 {
        self.current_selected
    }

    /// Java `getSelectedText()`.
    ///
    /// Upstream bug fixed in translation (`ColoredStateText.java:77`): before any
    /// `setSelected`, `currentSelected` is -1 and the Java throws
    /// `ArrayIndexOutOfBoundsException`; this returns null (`None`) instead.
    pub fn get_selected_text(&self) -> Option<String> {
        let labels = self.labels.as_ref()?;
        if self.current_selected < 0 {
            return None;
        }
        labels.get(self.current_selected as usize).cloned()
    }

    /// Java `getSelectedColor()`.
    ///
    /// Upstream bug fixed in translation (`ColoredStateText.java:81`): before any
    /// `setSelected`, `currentSelected` is -1 and the Java throws
    /// `ArrayIndexOutOfBoundsException`; this returns `None` instead.
    pub fn get_selected_color(&self) -> Option<Color> {
        if self.current_selected < 0 {
            return None;
        }
        self.colors.get(self.current_selected as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_and_read() {
        let mut state =
            ColoredStateText::new_string_array_color_array(&["a", "b"], &[(1, 1, 1), (2, 2, 2)])
                .unwrap();
        assert_eq!(state.get_selected_color(), None);
        assert!(state.set_selected(2).is_err());
        state.set_selected(1).unwrap();
        assert_eq!(state.get_selected_text().as_deref(), Some("b"));
        assert_eq!(state.get_selected_color(), Some((2, 2, 2)));
    }
}
