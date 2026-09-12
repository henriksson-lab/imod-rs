//! `IMOD/Etomo/src/etomo/ui/swing/HighlighterButton.java`.
//!
//! `HeaderCell`, Swing listeners, GridBag placement, and platform detection are
//! explicit GUI boundaries.  This source unit retains toggle, group, tooltip,
//! listener, and header-identity state.
#![allow(dead_code)]

use super::highlightable::Highlightable;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HighlighterButton {
    pub selected: bool,
    pub displayed: bool,
    pub enabled: bool,
    pub focusable: bool,
    pub width: i32,
    pub tooltip: String,
    pub table_header: Option<String>,
    pub row_header: Option<String>,
    pub column_header: Option<String>,
    pub border_raised: bool,
    pub listener_count: usize,
    pub parent_key: usize,
    pub group_key: Option<usize>,
}

impl HighlighterButton {
    /// Java private constructor plus `getInstance` listener installation.
    pub fn get_instance(parent_key: usize, group_key: Option<usize>) -> Self {
        let mut button = Self {
            enabled: true,
            focusable: true,
            width: 40,
            border_raised: true,
            parent_key,
            group_key,
            ..Default::default()
        };
        button.set_tool_tip_text_default(false);
        button.add_listeners();
        button
    }
    pub fn set_headers(&mut self, table_header: &str, row_header: &str, column_header: &str) {
        self.table_header = Some(table_header.to_owned());
        self.row_header = Some(row_header.to_owned());
        self.column_header = Some(column_header.to_owned());
    }
    pub fn is_highlighted(&self) -> bool {
        self.selected
    }
    pub fn get_width(&self) -> i32 {
        self.width
    }
    pub fn set_tool_tip_text(&mut self, text: impl Into<String>) {
        self.tooltip = text.into();
    }
    pub fn add(&mut self) {
        self.displayed = true;
    }
    pub fn remove(&mut self) {
        self.displayed = false;
    }
    /// Java `setSelected`; `parent.highlight` is deliberately performed by the
    /// caller owning that parent because native listener dispatch is a boundary.
    pub fn set_selected(&mut self, select: bool) -> bool {
        self.selected = select;
        self.action()
    }
    pub fn action(&self) -> bool {
        self.selected
    }
    pub fn get_component(&self) -> bool {
        self.displayed
    }
    /// Java has an empty body.
    pub fn set_foreground(&mut self) {}
    pub fn set_tool_tip_text_default(&mut self, mac_os: bool) {
        let mask = if mac_os { "[alt option]" } else { "[Alt]" };
        self.tooltip =
            format!("Press to highlight row.  Hotkeys: {mask}+[Up_Arrow] and {mask}+[Down_Arrow]");
    }
    pub fn add_listeners(&mut self) {
        self.listener_count += 1;
    }
    pub fn turn_off_highlight(&mut self, parent: &mut dyn Highlightable) {
        self.selected = false;
        parent.highlight(false);
    }
    /// Java `HBActionListener.actionPerformed` observable dispatch.
    pub fn action_performed(
        &self,
        parent: &mut dyn Highlightable,
        group: Option<&mut dyn Highlightable>,
    ) {
        let highlight = self.action();
        parent.highlight(highlight);
        if let Some(group) = group {
            group.highlight(highlight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Target(Vec<bool>);
    impl Highlightable for Target {
        fn highlight(&mut self, highlight: bool) {
            self.0.push(highlight);
        }
    }
    #[test]
    fn source_toggle_listener_and_headers_are_retained() {
        let mut button = HighlighterButton::get_instance(1, Some(2));
        button.set_headers("Table", "1", "Column");
        assert_eq!(button.listener_count, 1);
        assert!(button.set_selected(true));
        let mut parent = Target::default();
        let mut group = Target::default();
        button.action_performed(&mut parent, Some(&mut group));
        assert_eq!(parent.0, vec![true]);
        assert_eq!(group.0, vec![true]);
        button.turn_off_highlight(&mut parent);
        assert!(!button.is_highlighted());
    }
}
