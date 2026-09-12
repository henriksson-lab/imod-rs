//! `IMOD/Etomo/src/etomo/ui/swing/BooleanComboBoxEfield.java`.
//!
//! `JComboBox` is a Swing-owned widget and is retained as an explicit GUI
//! boundary. The state below is exactly the inherited `ComboBoxEfield` state
//! read or changed by this source unit: its null/empty item, two `Option`
//! items, selection, and `TextFlagExtension.update` notification.
#![allow(dead_code)]

/// Java `Option` values installed by this source unit.
///
/// `Option.java` is a separate source unit; this boundary representation keeps
/// its `value` and `descr` arguments intact until the generic combo-box unit is
/// translated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanComboBoxOption {
    /// Java `Option.value`.
    pub value: String,
    /// Java `Option.descr`.
    pub descr: String,
}

/// Java package-private final `BooleanComboBoxEfield`.
///
/// Java inheritance is represented by source-visible inherited fields because
/// `ComboBoxEfield.java` is a separately pending source unit. Swing selection
/// and flag-display dispatch remain explicit GUI boundaries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanComboBoxEfield {
    /// Java inherited `label` constructor argument.
    pub label: String,
    /// Java inherited `includeValue`, passed as false to `super`.
    pub include_value: bool,
    /// Java inherited absent `ControlComponentModule`, passed as false.
    pub include_control_component: bool,
    /// Java `comboBox` item sequence; `None` is Java's initial null item.
    pub items: Vec<Option<BooleanComboBoxOption>>,
    /// Java `comboBox.getSelectedIndex()`; JComboBox initially selects zero.
    pub selected_index: i32,
    /// Java inherited `choiceListSet` after `setChoiceListSet(true)`.
    pub choice_list_set: bool,
    /// Java inherited nullable `flagExtension` presence.
    pub flag_extension_present: bool,
    /// Dispatches from `updateFlagExtension` to `TextFlagExtension.update`.
    pub flag_extension_update_count: usize,
}

impl BooleanComboBoxEfield {
    /// Java `BooleanComboBoxEfield(String)`.
    pub fn new(label: &str) -> Self {
        const EMPTY_INDEX: i32 = 0;
        const TRUE_VALUE: &str = "1";
        const FALSE_VALUE: &str = "0";
        const TRUE_STRING: &str = "Yes";
        const FALSE_STRING: &str = "No";

        Self {
            label: label.into(),
            include_value: false,
            include_control_component: false,
            items: vec![
                None,
                Some(BooleanComboBoxOption {
                    value: TRUE_VALUE.into(),
                    descr: TRUE_STRING.into(),
                }),
                Some(BooleanComboBoxOption {
                    value: FALSE_VALUE.into(),
                    descr: FALSE_STRING.into(),
                }),
            ],
            selected_index: EMPTY_INDEX,
            choice_list_set: true,
            flag_extension_present: false,
            flag_extension_update_count: 0,
        }
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&mut self, value: bool) {
        const TRUE_INDEX: i32 = 1;
        const FALSE_INDEX: i32 = 2;

        if value {
            self.selected_index = TRUE_INDEX;
        } else {
            self.selected_index = FALSE_INDEX;
        }
        self.update_flag_extension();
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        const TRUE_INDEX: i32 = 1;

        self.selected_index == TRUE_INDEX
    }

    /// Java inherited `updateFlagExtension` called by `setSelected`.
    pub fn update_flag_extension(&mut self) {
        if self.flag_extension_present {
            self.flag_extension_update_count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_preserves_super_and_boolean_option_sequence() {
        let field = BooleanComboBoxEfield::new("Use alignment");

        assert_eq!(field.label, "Use alignment");
        assert!(!field.include_value);
        assert!(!field.include_control_component);
        assert_eq!(field.selected_index, 0);
        assert!(field.choice_list_set);
        assert_eq!(field.items.len(), 3);
        assert_eq!(field.items[0], None);
        assert_eq!(field.items[1].as_ref().unwrap().value, "1");
        assert_eq!(field.items[1].as_ref().unwrap().descr, "Yes");
        assert_eq!(field.items[2].as_ref().unwrap().value, "0");
        assert_eq!(field.items[2].as_ref().unwrap().descr, "No");
    }

    #[test]
    fn set_selected_uses_source_indices_and_notifies_existing_flag_extension() {
        let mut field = BooleanComboBoxEfield::new("Use alignment");
        field.flag_extension_present = true;

        field.set_selected(true);
        assert_eq!(field.selected_index, 1);
        assert!(field.is_selected());
        assert_eq!(field.flag_extension_update_count, 1);

        field.set_selected(false);
        assert_eq!(field.selected_index, 2);
        assert!(!field.is_selected());
        assert_eq!(field.flag_extension_update_count, 2);
    }

    #[test]
    fn set_selected_does_not_construct_a_missing_flag_extension() {
        let mut field = BooleanComboBoxEfield::new("Use alignment");

        field.set_selected(true);

        assert!(!field.flag_extension_present);
        assert_eq!(field.flag_extension_update_count, 0);
    }
}
