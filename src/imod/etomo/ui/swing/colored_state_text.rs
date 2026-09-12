//! `IMOD/Etomo/src/etomo/ui/swing/ColoredStateText.java`.
//!
//! `java.awt.Color` is represented by the shared GUI colour value.  Widget
//! painting remains at the GUI adapter boundary; this unit has no widget of
//! its own and retains the source's label, colour, and selected-index state.
#![allow(dead_code)]

use crate::imod::etomo::ui::swing::ui_utilities::Color;
use crate::imod::etomo::util::invalid_parameter_exception::InvalidParameterException;

/// Java package-private `ColoredStateText`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColoredStateText {
    /// Java nullable `labels` array.
    pub labels: Option<Vec<String>>,
    /// Java `colors` array.
    pub colors: Vec<Color>,
    /// Java `nItems`.
    pub n_items: i32,
    /// Java `currentSelected`.
    pub current_selected: i32,
}

impl ColoredStateText {
    /// Java `ColoredStateText(String[], Color[])`.
    pub fn new_with_labels(
        labels: Vec<String>,
        colors: Vec<Color>,
    ) -> Result<Self, InvalidParameterException> {
        let n_items = labels.len() as i32;
        if n_items != colors.len() as i32 {
            return Err(InvalidParameterException::new(
                "The length of the labels and colors arrays do not match",
            ));
        }
        Ok(Self {
            labels: Some(labels),
            colors,
            n_items,
            current_selected: -1,
        })
    }

    /// Java `ColoredStateText(Color[])`.
    pub fn new(colors: Vec<Color>) -> Self {
        Self {
            n_items: colors.len() as i32,
            labels: None,
            colors,
            current_selected: -1,
        }
    }

    /// Java `setSelected(int)`.
    ///
    /// The Java condition uses non-short-circuit boolean `&` between mutually
    /// exclusive comparisons.  It consequently accepts every index; retaining
    /// that source behaviour is important for parity.
    pub fn set_selected(&mut self, index: i32) -> Result<(), InvalidParameterException> {
        if (index < 0) & (index >= self.n_items) {
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
    pub fn get_selected_text(&self) -> Option<&str> {
        let labels = self.labels.as_ref()?;
        Some(&labels[self.current_selected as usize])
    }

    /// Java `getSelectedColor()`.
    pub fn get_selected_color(&self) -> Color {
        self.colors[self.current_selected as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: Color = Color {
        red: 255,
        green: 0,
        blue: 0,
    };
    const GREEN: Color = Color {
        red: 0,
        green: 255,
        blue: 0,
    };

    #[test]
    fn labeled_constructor_preserves_the_parallel_arrays() {
        let state = ColoredStateText::new_with_labels(
            vec!["Not started".into(), "Complete".into()],
            vec![RED, GREEN],
        )
        .unwrap();

        assert_eq!(state.n_items, 2);
        assert_eq!(state.current_selected, -1);
        assert_eq!(
            state
                .labels
                .as_deref()
                .map(|labels| labels.iter().map(String::as_str).collect::<Vec<_>>()),
            Some(vec!["Not started", "Complete"])
        );
        assert_eq!(state.colors, vec![RED, GREEN]);
    }

    #[test]
    fn labeled_constructor_rejects_different_array_lengths() {
        let error =
            ColoredStateText::new_with_labels(vec!["Only label".into()], vec![RED]).unwrap();
        assert_eq!(error.n_items, 1);

        let error =
            ColoredStateText::new_with_labels(vec!["Only label".into()], vec![]).unwrap_err();
        assert_eq!(
            error.get_message(),
            "The length of the labels and colors arrays do not match"
        );
    }

    #[test]
    fn selection_reads_the_source_label_and_colour_arrays() {
        let mut state = ColoredStateText::new_with_labels(
            vec!["Not started".into(), "Complete".into()],
            vec![RED, GREEN],
        )
        .unwrap();

        state.set_selected(1).unwrap();

        assert_eq!(state.get_selected(), 1);
        assert_eq!(state.get_selected_text(), Some("Complete"));
        assert_eq!(state.get_selected_color(), GREEN);
    }

    #[test]
    fn colour_only_constructor_returns_null_selected_text() {
        let mut state = ColoredStateText::new(vec![RED]);
        state.set_selected(0).unwrap();

        assert_eq!(state.get_selected_text(), None);
        assert_eq!(state.get_selected_color(), RED);
    }

    #[test]
    fn source_and_condition_accepts_out_of_range_indices() {
        let mut state = ColoredStateText::new(vec![RED]);

        assert!(state.set_selected(-1).is_ok());
        assert_eq!(state.get_selected(), -1);
        assert!(state.set_selected(1).is_ok());
        assert_eq!(state.get_selected(), 1);
    }
}
