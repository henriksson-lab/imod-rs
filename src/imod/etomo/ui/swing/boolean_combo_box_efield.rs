//! `IMOD/Etomo/src/etomo/ui/swing/BooleanComboBoxEfield.java`.
//!
//! This source class extends the separately translated `ComboBoxEfield`.
//! Composition exposes that exact inherited source object at the Rust
//! inheritance boundary; the constructor and the two methods below are the
//! complete members declared by `BooleanComboBoxEfield.java`.
#![allow(dead_code)]

use super::combo_box_efield::{ComboBoxEfield, ComboBoxOption};
use crate::imod::etomo::ui::shared_strings::{FALSE_STRING, TRUE_STRING};

const EMPTY_INDEX: i32 = 0;
const TRUE_INDEX: i32 = 1;
const FALSE_INDEX: i32 = 2;

/// Java package-private final `BooleanComboBoxEfield`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanComboBoxEfield {
    /// The Java superclass object created by `super(label, false, false)`.
    pub combo_box_efield: ComboBoxEfield,
}

impl BooleanComboBoxEfield {
    /// Java `BooleanComboBoxEfield(String)`.
    pub fn new(label: &str) -> Self {
        let mut combo_box_efield = ComboBoxEfield::new(label, false, false);
        combo_box_efield.add_item(ComboBoxOption {
            value: Some("1".into()),
            descr: Some(TRUE_STRING.into()),
            include_value: false,
        });
        combo_box_efield.add_item(ComboBoxOption {
            value: Some("0".into()),
            descr: Some(FALSE_STRING.into()),
            include_value: false,
        });
        combo_box_efield.set_choice_list_set(true);
        Self { combo_box_efield }
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&mut self, value: bool) {
        if value {
            self.combo_box_efield.set_selected_index(TRUE_INDEX);
        } else {
            self.combo_box_efield.set_selected_index(FALSE_INDEX);
        }
        self.combo_box_efield.update_flag_extension();
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.combo_box_efield.get_selected_index() == TRUE_INDEX
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::ui::swing::combo_box_efield::ComboBoxItem;

    #[test]
    fn constructor_uses_the_canonical_superclass_and_boolean_option_sequence() {
        let field = BooleanComboBoxEfield::new("Use alignment");
        let base = &field.combo_box_efield;

        assert_eq!(base.label, "Use alignment");
        assert!(!base.include_value);
        assert!(base.control_component.is_none());
        assert_eq!(base.get_selected_index(), EMPTY_INDEX);
        assert!(base.choice_list_set);
        assert_eq!(base.combo_box_items.len(), 3);
        assert_eq!(base.combo_box_items[0], ComboBoxItem::Empty);
        assert_eq!(
            base.combo_box_items[1],
            ComboBoxItem::Option(ComboBoxOption {
                value: Some("1".into()),
                descr: Some(TRUE_STRING.into()),
                include_value: false,
            })
        );
        assert_eq!(
            base.combo_box_items[2],
            ComboBoxItem::Option(ComboBoxOption {
                value: Some("0".into()),
                descr: Some(FALSE_STRING.into()),
                include_value: false,
            })
        );
    }

    #[test]
    fn set_selected_uses_source_indices_and_the_superclass_flag_dispatch() {
        let mut field = BooleanComboBoxEfield::new("Use alignment");
        field.combo_box_efield.create_flag_extension();

        field.set_selected(true);
        assert_eq!(field.combo_box_efield.get_selected_index(), TRUE_INDEX);
        assert!(field.is_selected());
        assert_eq!(
            field
                .combo_box_efield
                .flag_extension
                .as_ref()
                .map(|extension| extension.update_count),
            Some(1)
        );

        field.set_selected(false);
        assert_eq!(field.combo_box_efield.get_selected_index(), FALSE_INDEX);
        assert!(!field.is_selected());
        assert_eq!(
            field
                .combo_box_efield
                .flag_extension
                .as_ref()
                .map(|extension| extension.update_count),
            Some(2)
        );
    }

    #[test]
    fn set_selected_does_not_create_the_optional_superclass_flag_extension() {
        let mut field = BooleanComboBoxEfield::new("Use alignment");

        field.set_selected(true);

        assert!(field.combo_box_efield.flag_extension.is_none());
        assert_eq!(field.combo_box_efield.get_selected_index(), TRUE_INDEX);
    }
}
