//! `IMOD/Etomo/src/etomo/ui/swing/HighlightableTable.java`.
//!
//! The base of a table whose rows can be highlighted: it binds Alt+Up and Alt+Down
//! on the table's focusable parents to moving the highlight up and down.  Adds to
//! the action maps of up to three components.
//!
//! **Representation.**  The Java abstract class is the [`HighlightableTable`] state
//! (embedded by the subclass) plus the [`HighlightableTableVirtual`] trait for its
//! abstract methods.  The key bindings go to the `jdk` stand-in's per-component
//! binding table (`JComponent::put_key_binding`), which is what
//! `getInputMap(...).put` and `getActionMap().put` amount to.

use std::rc::{Rc, Weak};

use super::highlightable::Highlightable;
use crate::imod::etomo::jdk::JComponent;

/// The abstract methods of Java `HighlightableTable`.
pub trait HighlightableTableVirtual: Highlightable {
    /// The embedded `HighlightableTable` (Java `this` seen as the base class).
    fn highlightable_table(&self) -> &HighlightableTable;

    /// Java abstract `highlightUpActionPerformed()`.
    fn highlight_up_action_performed(&self);

    /// Java abstract `highlightDownActionPerformed()`.
    fn highlight_down_action_performed(&self);

    /// Java abstract `getFocusableParents()`.
    fn get_focusable_parents(&self) -> Vec<Option<Rc<JComponent>>>;
}

/// Java package-private `abstract class HighlightableTable implements Highlightable`.
pub struct HighlightableTable {
    /// Java private final `uniqueKey`.
    unique_key: String,
}

impl HighlightableTable {
    /// Java package-private `HighlightableTable(String)`.
    pub fn new(unique_key: &str) -> HighlightableTable {
        HighlightableTable {
            unique_key: unique_key.to_owned(),
        }
    }

    /// Java package-private `initHighlightHotkeys()`.  `this` is the subclass object.
    pub fn init_highlight_hotkeys(&self, this: Weak<dyn HighlightableTableVirtual>) {
        self.add_listeners(this);
    }

    /// Java private `addAction(InputMap[], ActionMap, int, String, AbstractAction)`.
    /// Optionally add a key stroke action for one parent.  Each of the three input
    /// maps (`WHEN_FOCUSED`, `WHEN_IN_FOCUSED_WINDOW`,
    /// `WHEN_ANCESTOR_OF_FOCUSED_COMPONENT`) maps the stroke to the same action on the
    /// one parent, which is one binding here.
    fn add_action(
        &self,
        parent: &Rc<JComponent>,
        key_stroke: &str,
        key: &str,
        action: Rc<dyn Fn()>,
    ) {
        // `key = uniqueKey + key`: the action map key, which only names the action.
        let _key = format!("{}{}", self.unique_key, key);
        parent.put_key_binding(key_stroke, action);
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self, this: Weak<dyn HighlightableTableVirtual>) {
        let Some(table) = this.upgrade() else {
            return;
        };
        let focusable_parents = table.get_focusable_parents();
        // Java `HighlightUpAction` / `HighlightDownAction`.
        let up_table = this.clone();
        let up_action: Rc<dyn Fn()> = Rc::new(move || {
            if let Some(table) = up_table.upgrade() {
                table.highlight_up_action_performed();
            }
        });
        let down_table = this.clone();
        let down_action: Rc<dyn Fn()> = Rc::new(move || {
            if let Some(table) = down_table.upgrade() {
                table.highlight_down_action_performed();
            }
        });
        for parent in focusable_parents.iter() {
            let Some(parent) = parent else {
                continue;
            };
            parent.set_focusable(true);
            self.add_action(parent, "alt UP", "ALT_UP", up_action.clone());
            self.add_action(parent, "alt DOWN", "ALT_DOWN", down_action.clone());
        }
    }
}
