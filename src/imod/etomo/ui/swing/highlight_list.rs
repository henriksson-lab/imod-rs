//! `IMOD/Etomo/src/etomo/ui/swing/HighlightList.java`.
//!
//! A vertical list of labels, one of which is shown selected (in a highlight
//! colour).

use std::cell::Cell;
use std::rc::Rc;

use crate::imod::etomo::jdk::{Color, JComponent, MouseListener};

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java package-private `class HighlightList`.
pub struct HighlightList {
    /// Java `panel`.
    panel: Rc<JComponent>,
    /// Java `labels`.
    labels: Vec<Rc<JComponent>>,
    /// Java `nItems`.
    n_items: i32,
    /// Java `currentSelected`.
    current_selected: Cell<i32>,
    /// Java `unselected`.
    unselected: Color,
    /// Java `selected`.
    selected: Color,
}

impl HighlightList {
    /// Java `HighlightList(String[])`.
    pub fn new(items: &[&str]) -> HighlightList {
        let panel = JComponent::new_panel();
        let unselected: Color = (128, 128, 128);
        let selected: Color = (160, 0, 64);
        let n_items = items.len() as i32;
        // Swing layout: panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS)).
        let mut labels = Vec::with_capacity(items.len());
        for i in 0..n_items as usize {
            let label = JComponent::new_label(items[i]);
            label.set_text(items[i]);
            label.set_foreground(Some(unselected));
            panel.add(&label);
            // Swing layout: panel.add(Box.createRigidArea(FixedDim.x0_y5)).
            labels.push(label);
        }
        HighlightList {
            panel,
            labels,
            n_items,
            current_selected: Cell::new(-1),
            unselected,
            selected,
        }
    }

    /// Java `getPanel()`.
    pub fn get_panel(&self) -> Rc<JComponent> {
        self.panel.clone()
    }

    /// Java `setSelected(int)`.
    pub fn set_selected(&self, index: i32) {
        if index >= 0 && index < self.n_items {
            //
            //  Deselect the current item
            //
            if self.current_selected.get() != -1 {
                self.labels[self.current_selected.get() as usize]
                    .set_foreground(Some(self.unselected));
            }
            //
            //  Select the new item
            //
            self.current_selected.set(index);
            self.labels[index as usize].set_foreground(Some(self.selected));
        }
    }

    /// Java `getSelected()`.
    pub fn get_selected(&self) -> i32 {
        self.current_selected.get()
    }

    /// Java `getSelectedText()`.
    ///
    /// Upstream bug fixed in translation (`HighlightList.java:74`): before any item is
    /// selected `currentSelected` is -1 and the Java throws
    /// `ArrayIndexOutOfBoundsException`; this returns null (`None`) instead.
    pub fn get_selected_text(&self) -> Option<String> {
        let current_selected = self.current_selected.get();
        if current_selected < 0 {
            return None;
        }
        Some(self.labels[current_selected as usize].get_text())
    }

    /// Java `addMouseListener(MouseListener)`.
    pub fn add_mouse_listener(&self, listener: Rc<dyn MouseListener>) {
        self.panel.add_mouse_listener(listener.clone());
        for label in &self.labels {
            label.add_mouse_listener(listener.clone());
        }
    }
}
