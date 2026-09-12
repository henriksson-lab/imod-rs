//! `IMOD/Etomo/src/etomo/ui/swing/HighlightableTable.java`.
#![allow(dead_code)]

/// Native Swing InputMap/ActionMap registration retained by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HighlightFocusableParent {
    pub focusable: bool,
    pub focused_keys: Vec<String>,
    pub window_keys: Vec<String>,
    pub ancestor_keys: Vec<String>,
    pub actions: Vec<String>,
}

/// Java abstract `HighlightableTable`.
pub trait HighlightableTable {
    /// Java abstract `highlightUpActionPerformed()`.
    fn highlight_up_action_performed(&mut self);
    /// Java abstract `highlightDownActionPerformed()`.
    fn highlight_down_action_performed(&mut self);
    /// Java abstract `getFocusableParents()` explicit Swing boundary.
    fn get_focusable_parents(&mut self) -> Option<&mut [HighlightFocusableParent]>;
    /// Java private final `uniqueKey` supplied by each derived table.
    fn unique_key(&self) -> &str;

    /// Java `initHighlightHotkeys()`.
    fn init_highlight_hotkeys(&mut self) {
        let unique_key = self.unique_key().to_owned();
        let Some(parents) = self.get_focusable_parents() else {
            return;
        };
        for parent in parents {
            parent.focusable = true;
            for (suffix, key) in [("ALT_UP", "ALT+UP"), ("ALT_DOWN", "ALT+DOWN")] {
                let action = format!("{unique_key}{suffix}");
                parent.focused_keys.push(key.to_owned());
                parent.window_keys.push(key.to_owned());
                parent.ancestor_keys.push(key.to_owned());
                parent.actions.push(action);
            }
        }
    }

    /// Java private `HighlightUpAction.actionPerformed(ActionEvent)`.
    fn highlight_up_action(&mut self) {
        self.highlight_up_action_performed();
    }

    /// Java private `HighlightDownAction.actionPerformed(ActionEvent)`.
    fn highlight_down_action(&mut self) {
        self.highlight_down_action_performed();
    }
}

#[cfg(test)]
mod tests {
    use super::{HighlightFocusableParent, HighlightableTable};

    struct Table {
        key: String,
        parents: Vec<HighlightFocusableParent>,
        up: usize,
        down: usize,
    }
    impl HighlightableTable for Table {
        fn highlight_up_action_performed(&mut self) {
            self.up += 1;
        }
        fn highlight_down_action_performed(&mut self) {
            self.down += 1;
        }
        fn get_focusable_parents(&mut self) -> Option<&mut [HighlightFocusableParent]> {
            Some(&mut self.parents)
        }
        fn unique_key(&self) -> &str {
            &self.key
        }
    }

    #[test]
    fn installs_source_alt_actions_in_all_three_input_maps() {
        let mut table = Table {
            key: "row.".to_owned(),
            parents: vec![HighlightFocusableParent::default()],
            up: 0,
            down: 0,
        };
        table.init_highlight_hotkeys();
        assert!(table.parents[0].focusable);
        assert_eq!(table.parents[0].actions, ["row.ALT_UP", "row.ALT_DOWN"]);
        assert_eq!(table.parents[0].focused_keys, ["ALT+UP", "ALT+DOWN"]);
        table.highlight_up_action();
        table.highlight_down_action();
        assert_eq!((table.up, table.down), (1, 1));
    }
}
