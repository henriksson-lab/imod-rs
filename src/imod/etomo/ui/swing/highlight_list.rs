//! `IMOD/Etomo/src/etomo/ui/swing/HighlightList.java`.
#![allow(dead_code)]

use super::fixed_dim::FixedDim;
use super::panel::Dimension;

/// Java `HighlightList` label state at the native Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HighlightListLabel {
    pub text: String,
    pub foreground: (u8, u8, u8),
    pub mouse_listener_count: usize,
}

/// Java package-private `HighlightList`.
pub struct HighlightList {
    pub labels: Vec<HighlightListLabel>,
    pub n_items: usize,
    pub current_selected: isize,
    pub unselected: (u8, u8, u8),
    pub selected: (u8, u8, u8),
    pub rigid_areas: Vec<Dimension>,
    pub panel_mouse_listener_count: usize,
}

impl HighlightList {
    /// Java `HighlightList(String[])`.
    pub fn new(items: &[String]) -> Self {
        let unselected = (128, 128, 128);
        Self {
            labels: items
                .iter()
                .map(|item| HighlightListLabel {
                    text: item.clone(),
                    foreground: unselected,
                    mouse_listener_count: 0,
                })
                .collect(),
            n_items: items.len(),
            current_selected: -1,
            unselected,
            selected: (160, 0, 64),
            rigid_areas: vec![FixedDim::x0_y5; items.len()],
            panel_mouse_listener_count: 0,
        }
    }

    /// Java `getPanel()` native JPanel boundary.
    pub fn get_panel(&self) -> &[HighlightListLabel] {
        &self.labels
    }

    /// Java `setSelected(int)`; source intentionally does not deselect on an
    /// invalid index.
    pub fn set_selected(&mut self, index: isize) {
        if index >= 0 && index < self.n_items as isize {
            if self.current_selected != -1 {
                self.labels[self.current_selected as usize].foreground = self.unselected;
            }
            self.current_selected = index;
            self.labels[index as usize].foreground = self.selected;
        }
    }

    /// Java `getSelected()`.
    pub fn get_selected(&self) -> isize {
        self.current_selected
    }

    /// Java `getSelectedText()`; Java indexes directly and therefore fails if
    /// no valid selection has been made.
    pub fn get_selected_text(&self) -> String {
        self.labels[self.current_selected as usize].text.clone()
    }

    /// Java `addMouseListener(MouseListener)` native listener boundary.
    pub fn add_mouse_listener(&mut self) {
        self.panel_mouse_listener_count += 1;
        for label in &mut self.labels {
            label.mouse_listener_count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HighlightList;

    #[test]
    fn selection_preserves_java_colour_and_listener_order() {
        let mut list = HighlightList::new(&["one".to_owned(), "two".to_owned()]);
        list.add_mouse_listener();
        list.set_selected(0);
        list.set_selected(1);
        list.set_selected(4);
        assert_eq!(list.get_selected(), 1);
        assert_eq!(list.get_selected_text(), "two");
        assert_eq!(list.labels[0].foreground, (128, 128, 128));
        assert_eq!(list.labels[1].foreground, (160, 0, 64));
        assert_eq!(list.panel_mouse_listener_count, 1);
        assert_eq!(list.labels[0].mouse_listener_count, 1);
    }
}
