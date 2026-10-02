//! `IMOD/Etomo/src/etomo/ui/swing/EtomoButtonGroup.java`.
//!
//! Extension of ButtonGroup which allows a button to be selected based on
//! its EnumeratedType.
//!
//! Java `extends ButtonGroup`: the `javax.swing.ButtonGroup` is the `group`
//! field (deref).  Java's `add(AbstractButton)` reads the button's
//! `ButtonModel`; the `jdk.rs` stand-in keeps selection on the component and
//! has no model object, so the caller (`RadioEbutton`, the only one) passes
//! the `AbstractRadioButtonModel` it installed with `setModel` alongside the
//! button, and the group keeps the button with it so that selecting the
//! model selects the button.

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use super::radio_button_interface::EnumeratedTypeRef;
use crate::imod::etomo::jdk::{ButtonGroup, JComponent};

use super::abstract_radio_button_model::AbstractRadioButtonModel;

/// Java package-private final `EtomoButtonGroup`.
pub struct EtomoButtonGroup {
    /// The Java `ButtonGroup` this class extends.
    group: Rc<ButtonGroup>,
    /// Java final `buttonModelList`
    /// (`HashMap<EnumeratedType, AbstractRadioButtonModel>`), with the
    /// model's button.  Java's keys are the enumerated-type singletons,
    /// compared by identity, which is `EnumeratedTypeRef`'s `==`.
    button_model_list: RefCell<
        Vec<(
            EnumeratedTypeRef,
            Rc<dyn AbstractRadioButtonModel>,
            Rc<JComponent>,
        )>,
    >,
}

impl Deref for EtomoButtonGroup {
    type Target = ButtonGroup;
    fn deref(&self) -> &ButtonGroup {
        &self.group
    }
}

impl EtomoButtonGroup {
    /// Java package-private `EtomoButtonGroup()`.
    pub fn new() -> Rc<EtomoButtonGroup> {
        Rc::new(EtomoButtonGroup {
            group: ButtonGroup::new(),
            button_model_list: RefCell::new(Vec::new()),
        })
    }

    /// The Java `ButtonGroup` itself.
    pub fn get_button_group(&self) -> Rc<ButtonGroup> {
        self.group.clone()
    }

    /// Java `add(AbstractButton)` (overrides `ButtonGroup.add`).  `model` is
    /// Java's `button.getModel()` when that is an `AbstractRadioButtonModel`.
    pub fn add(&self, button: &Rc<JComponent>, model: Option<Rc<dyn AbstractRadioButtonModel>>) {
        self.group.add(button);
        // Save the button model if possible.
        if let Some(radio_button_model) = model {
            let enumerated_type = radio_button_model.get_enumerated_type();
            if let Some(enumerated_type) = enumerated_type {
                let mut button_model_list = self.button_model_list.borrow_mut();
                // Java `HashMap.put`: an equal key's entry is replaced.
                let existing = button_model_list
                    .iter()
                    .position(|(key, _, _)| *key == enumerated_type);
                let entry = (enumerated_type, radio_button_model, button.clone());
                match existing {
                    Some(index) => button_model_list[index] = entry,
                    None => button_model_list.push(entry),
                }
            }
        }
    }

    /// Java `setSelected(EnumeratedType)`: `super.setSelected(model, true)`,
    /// which ignores a null model.
    pub fn set_selected(&self, enumerated_type: Option<&EnumeratedTypeRef>) {
        let Some(enumerated_type) = enumerated_type else {
            return;
        };
        let button = self
            .button_model_list
            .borrow()
            .iter()
            .find(|(key, _, _)| key == enumerated_type)
            .map(|(_, _, button)| button.clone());
        if let Some(button) = button {
            button.set_selected(true);
        }
    }
}
