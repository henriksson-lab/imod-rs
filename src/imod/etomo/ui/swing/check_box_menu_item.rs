//! `IMOD/Etomo/src/etomo/ui/swing/CheckBoxMenuItem.java`.
//!
//! `JCheckBoxMenuItem` rendering and event dispatch stay at the explicit Swing
//! boundary.  This source unit owns the self-naming state inherited from that
//! widget.
#![allow(dead_code)]

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::util::utilities;

/// Source-observable inherited `JCheckBoxMenuItem` state; painting and native
/// listener dispatch belong to Swing.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct JCheckBoxMenuItemBoundary {
    pub text: Option<String>,
    pub name: Option<String>,
    pub selected: bool,
    pub enabled: bool,
    pub visible: bool,
}

/// Java package-private final `CheckBoxMenuItem`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckBoxMenuItem {
    pub check_box_menu_item: JCheckBoxMenuItemBoundary,
}

impl CheckBoxMenuItem {
    /// Java `CheckBoxMenuItem()`.
    pub fn new() -> Self {
        Self {
            check_box_menu_item: JCheckBoxMenuItemBoundary {
                enabled: true,
                visible: true,
                ..Default::default()
            },
        }
    }

    /// Java `CheckBoxMenuItem(String)`.
    pub fn with_text(text: &str) -> Self {
        let mut value = Self::new();
        value.set_text(text);
        value
    }

    /// Java overridden `setText(String)`.
    pub fn set_text(&mut self, text: &str) {
        self.check_box_menu_item.text = Some(text.into());
        self.set_name(text);
    }

    /// Java overridden `setName(String)`.
    pub fn set_name(&mut self, text: &str) {
        // Java `UITestFieldType.CHECK_BOX_MENU_ITEM.toString()`.
        const FIELD_TYPE: &str = "cbmn";

        let name = utilities::convert_label_to_name(Some(text), true).unwrap_or_default();
        self.check_box_menu_item.name = Some(format!("{FIELD_TYPE}{SEPARATOR_CHAR}{name}"));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {DEFAULT_DELIMITER} ",
                self.check_box_menu_item.name.as_deref().unwrap_or_default()
            );
        }
    }
}

impl Default for CheckBoxMenuItem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_constructor_sets_the_source_owned_name() {
        let item = CheckBoxMenuItem::with_text("Open 3dmod Binned by 2");

        assert_eq!(
            item.check_box_menu_item.text.as_deref(),
            Some("Open 3dmod Binned by 2")
        );
        assert_eq!(
            item.check_box_menu_item.name.as_deref(),
            Some("cbmn.open-3dmod-binned-by-2")
        );
    }

    #[test]
    fn set_text_renames_the_menu_item() {
        let mut item = CheckBoxMenuItem::new();
        item.set_text("Open 3dmod with Startup Window");

        assert_eq!(
            item.check_box_menu_item.name.as_deref(),
            Some("cbmn.open-3dmod-with-startup-window")
        );
    }

    #[test]
    fn set_name_does_not_change_inherited_text() {
        let mut item = CheckBoxMenuItem::with_text("Old");
        item.set_name("New Item");

        assert_eq!(item.check_box_menu_item.text.as_deref(), Some("Old"));
        assert_eq!(
            item.check_box_menu_item.name.as_deref(),
            Some("cbmn.new-item")
        );
    }
}
