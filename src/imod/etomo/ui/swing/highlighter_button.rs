//! `IMOD/Etomo/src/etomo/ui/swing/HighlighterButton.java`.
//!
//! The "=>" toggle at the start of a table row: selecting it highlights the row
//! (its parent) and turns off the other rows' highlighters in the same group (the
//! table).  An event dispatch thread object, created as `Rc<Self>`; the action
//! listener is a closure holding a weak reference to the button.
//!
//! **Groups.**  Java keeps a static `HashedLists` from each group to the
//! highlighters in it, keyed by the group object.  Groups and buttons are EDT
//! objects, so the lists are a thread-local keyed by the group's address; the lists
//! hold weak references (Java's static lists keep the buttons alive for the run,
//! which only matters to the groups that still exist).

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::cell::CellVirtual;
use super::header_cell::HeaderCell;
use super::highlightable::Highlightable;
use super::ui_parameters::UIParameters;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, GridBagConstraints, GridBagLayout, JComponent};
use crate::imod::etomo::util::utilities;

thread_local! {
    /// Java private static final `groupLists = new HashedLists()`.
    static GROUP_LISTS: RefCell<Vec<(*const (), Vec<Weak<HighlighterButton>>)>> =
        const { RefCell::new(Vec::new()) };
}

/// The address a group is keyed by (Java's `Hashtable` uses the group object).
fn group_key(group: &Weak<dyn Highlightable>) -> *const () {
    group.as_ptr() as *const ()
}

/// Java package-private `final class HighlighterButton`.
pub struct HighlighterButton {
    /// Java private final `parent`.
    parent: Weak<dyn Highlightable>,
    /// Java private final `group`.
    group: Option<Weak<dyn Highlightable>>,
    /// Java private final `cell`.
    cell: Rc<HeaderCell>,
    /// Rust-only: Java `this` for the listeners and the group lists.
    self_ref: Weak<HighlighterButton>,
}

impl HighlighterButton {
    /// Java private `HighlighterButton(Highlightable, Highlightable)`.  Lazy
    /// constructor.
    fn new(
        parent: Weak<dyn Highlightable>,
        group: Option<Weak<dyn Highlightable>>,
    ) -> Rc<HighlighterButton> {
        let this = Rc::new_cyclic(|self_ref: &Weak<HighlighterButton>| {
            // button
            let cell = HeaderCell::get_toggle_instance(
                Some("=>"),
                (40.0 * UIParameters::get_instance_void().get_font_size_adjustment()) as i32,
            );
            HighlighterButton {
                parent,
                group: group.clone(),
                cell,
                self_ref: self_ref.clone(),
            }
        });
        // group
        if let Some(group) = &this.group {
            let key = group_key(group);
            GROUP_LISTS.with(|group_lists| {
                let mut group_lists = group_lists.borrow_mut();
                match group_lists.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, list)) => list.push(Rc::downgrade(&this)),
                    None => group_lists.push((key, vec![Rc::downgrade(&this)])),
                }
            });
        }
        // Swing painting: cell.setBorder(BorderFactory.createBevelBorder(RAISED)).
        CellVirtual::set_enabled(&*this.cell, true);
        this.cell.add_action_listener(this.hb_action_listener());
        this.cell.set_focusable(true);
        this.set_tool_tip_text_void();
        this
    }

    /// Java private static final class `HBActionListener`.
    fn hb_action_listener(&self) -> ActionListener {
        let highlighter_button = self.self_ref.clone();
        Rc::new(move |_event: &ActionEvent| {
            if let Some(highlighter_button) = highlighter_button.upgrade() {
                highlighter_button.action();
            }
        })
    }

    /// Java static `getInstance(Highlightable, Highlightable)`.
    pub fn get_instance(
        parent: Weak<dyn Highlightable>,
        group: Option<Weak<dyn Highlightable>>,
    ) -> Rc<HighlighterButton> {
        let instance = HighlighterButton::new(parent, group);
        instance.add_listeners();
        instance
    }

    /// Java `setHeaders(String, HeaderCell, HeaderCell)`.
    pub fn set_headers(
        &self,
        table_header: Option<&str>,
        row_header: &Rc<HeaderCell>,
        column_header: &Rc<HeaderCell>,
    ) {
        self.cell.set_table_header(table_header);
        self.cell.set_row_header(Some(Rc::clone(row_header)));
        self.cell.set_column_header(Some(Rc::clone(column_header)));
        self.cell.set_name(None);
        let cell: Rc<dyn CellVirtual> = self.cell.clone();
        row_header.add_child(Rc::downgrade(&cell));
        column_header.add_child(Rc::downgrade(&cell));
    }

    /// Java `isHighlighted()`.
    pub fn is_highlighted(&self) -> bool {
        self.cell.is_selected()
    }

    /// Java `getWidth()`.
    pub fn get_width(&self) -> i32 {
        self.cell.get_width()
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.cell.set_tool_tip_text(text);
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.
    pub fn add(
        &self,
        panel: &Rc<JComponent>,
        layout: &GridBagLayout,
        constraints: &mut GridBagConstraints,
    ) {
        let old_weightx = constraints.weightx;
        constraints.weightx = 0.0;
        CellVirtual::add(&*self.cell, panel);
        layout.set_constraints(&self.cell.get_component(), constraints);
        constraints.weightx = old_weightx;
    }

    /// Java `remove()`.
    pub fn remove(&self) {
        self.cell.remove();
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, select: bool) {
        self.cell.set_selected(select);
        self.action();
    }

    /// Java private `action()`.
    fn action(&self) {
        let highlight = self.cell.is_selected();
        if let Some(parent) = self.parent.upgrade() {
            parent.highlight(highlight);
        }
        let Some(group) = &self.group else {
            return;
        };
        // If turning on the highlight, all other highlighters in the group must be
        // turned off
        let key = group_key(group);
        let list: Option<Vec<Weak<HighlighterButton>>> = GROUP_LISTS.with(|group_lists| {
            group_lists
                .borrow()
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, list)| list.clone())
        });
        let Some(list) = list else {
            panic!("Should be in the list.  group={:p}", key);
        };
        for highlighter_button in list.iter().filter_map(Weak::upgrade) {
            if !std::ptr::eq(&*highlighter_button, self) {
                highlighter_button.turn_off_highlight();
            }
        }
        // The group may also need to respond to the highlight
        if let Some(group) = group.upgrade() {
            group.highlight(highlight);
        }
    }

    /// Java final `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.cell.get_component()
    }

    /// Java `setForeground()`, empty.
    pub fn set_foreground(&self) {}

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text_void(&self) {
        let mask = if utilities::is_mac_os() {
            "[alt option]"
        } else {
            "[Alt]"
        };
        self.cell.set_tool_tip_text(Some(&format!(
            "Press to highlight row.  Hotkeys: {mask}+[Up_Arrow] and {mask}+[Down_Arrow]"
        )));
    }

    /// Java private `addListeners()`.  (The constructor has already added one
    /// `HBActionListener`; the source adds a second, so a click runs `action()`
    /// twice, which is idempotent.)
    fn add_listeners(&self) {
        self.cell.add_action_listener(self.hb_action_listener());
    }

    /// Java private `turnOffHighlight()`.
    fn turn_off_highlight(&self) {
        self.cell.set_selected(false);
        if let Some(parent) = self.parent.upgrade() {
            parent.highlight(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct Target(Cell<i32>, Cell<bool>);
    impl Highlightable for Target {
        fn highlight(&self, highlight: bool) {
            self.0.set(self.0.get() + 1);
            self.1.set(highlight);
        }
    }

    #[test]
    fn selecting_one_row_turns_the_other_off() {
        let group: Rc<dyn Highlightable> = Rc::new(Target(Cell::new(0), Cell::new(false)));
        let row1 = Rc::new(Target(Cell::new(0), Cell::new(false)));
        let row2 = Rc::new(Target(Cell::new(0), Cell::new(false)));
        let row1_dyn: Rc<dyn Highlightable> = row1.clone();
        let row2_dyn: Rc<dyn Highlightable> = row2.clone();
        let button1 =
            HighlighterButton::get_instance(Rc::downgrade(&row1_dyn), Some(Rc::downgrade(&group)));
        let button2 =
            HighlighterButton::get_instance(Rc::downgrade(&row2_dyn), Some(Rc::downgrade(&group)));
        button1.set_selected(true);
        assert!(button1.is_highlighted() && row1.1.get());
        button2.get_component().do_click();
        assert!(button2.is_highlighted() && row2.1.get());
        assert!(!button1.is_highlighted() && !row1.1.get());
    }
}
