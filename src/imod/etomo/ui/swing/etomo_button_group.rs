//! `IMOD/Etomo/src/etomo/ui/swing/EtomoButtonGroup.java`.
//!
//! `AbstractButton`, `ButtonGroup`, and `ButtonModel` are Swing objects.  Their
//! model installation and selected-state delivery remain at that GUI boundary;
//! this source unit retains the source-owned enumerated-type to radio-model map
//! and its exact add/select order.
#![allow(dead_code)]

use std::{cell::RefCell, collections::HashMap, hash::Hash, rc::Rc};

use super::abstract_radio_button_model::AbstractRadioButtonModel;

/// Java `ButtonModel` reference held by an `AbstractButton`.
///
/// The `Other` case is the Java `instanceof AbstractRadioButtonModel` false
/// branch.  Radio models are retained by shared identity because Java's
/// `ButtonGroup` and `HashMap` both retain object references, not copies.
#[derive(Debug)]
pub enum ButtonModelBoundary<M> {
    Other,
    AbstractRadioButton(Rc<RefCell<M>>),
}

impl<M> Clone for ButtonModelBoundary<M> {
    fn clone(&self) -> Self {
        match self {
            Self::Other => Self::Other,
            Self::AbstractRadioButton(model) => Self::AbstractRadioButton(model.clone()),
        }
    }
}

/// Java `AbstractButton` at the Swing boundary.
#[derive(Clone, Debug)]
pub struct AbstractButtonBoundary<M> {
    pub model: ButtonModelBoundary<M>,
}

impl<M> AbstractButtonBoundary<M> {
    /// Java `AbstractButton.getModel` result retained for a test or GUI adapter.
    pub fn new(model: ButtonModelBoundary<M>) -> Self {
        Self { model }
    }
}

/// Direct `ButtonGroup` calls made by this class.
///
/// Native Swing owns exclusive selection and listener notification.  The
/// retained state is the observable argument stream crossing that boundary.
#[derive(Clone, Debug)]
pub struct ButtonGroupBoundary<M> {
    pub models: Vec<ButtonModelBoundary<M>>,
    pub selected_model: Option<Rc<RefCell<M>>>,
    pub set_selected_calls: Vec<(Option<Rc<RefCell<M>>>, bool)>,
}

impl<M> Default for ButtonGroupBoundary<M> {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            selected_model: None,
            set_selected_calls: Vec::new(),
        }
    }
}

impl<M> ButtonGroupBoundary<M> {
    /// Java `ButtonGroup.add(AbstractButton)`.
    pub fn add(&mut self, button: &AbstractButtonBoundary<M>) {
        self.models.push(button.model.clone());
    }

    /// Java `ButtonGroup.setSelected(ButtonModel, boolean)`.
    pub fn set_selected(&mut self, model: Option<Rc<RefCell<M>>>, selected: bool) {
        self.set_selected_calls.push((model.clone(), selected));
        if selected {
            if let Some(model) = model {
                self.selected_model = Some(model);
            }
        }
    }
}

/// Java package-private final `EtomoButtonGroup`.
pub struct EtomoButtonGroup<M>
where
    M: AbstractRadioButtonModel,
    M::EnumeratedType: Clone + Eq + Hash,
{
    /// Inherited Java `ButtonGroup` state at the Swing boundary.
    pub button_group: ButtonGroupBoundary<M>,
    /// Java final `buttonModelList`.
    pub button_model_list: HashMap<M::EnumeratedType, Rc<RefCell<M>>>,
}

impl<M> EtomoButtonGroup<M>
where
    M: AbstractRadioButtonModel,
    M::EnumeratedType: Clone + Eq + Hash,
{
    /// Java `EtomoButtonGroup()`.
    pub fn new() -> Self {
        Self {
            button_group: ButtonGroupBoundary::default(),
            button_model_list: HashMap::new(),
        }
    }

    /// Java overridden `add(AbstractButton)`.
    pub fn add(&mut self, button: &AbstractButtonBoundary<M>) {
        self.button_group.add(button);
        if let ButtonModelBoundary::AbstractRadioButton(radio_button_model) = &button.model {
            let enumerated_type = radio_button_model.borrow().get_enumerated_type().cloned();
            if let Some(enumerated_type) = enumerated_type {
                self.button_model_list
                    .insert(enumerated_type, radio_button_model.clone());
            }
        }
    }

    /// Java `setSelected(EnumeratedType)`.
    pub fn set_selected(&mut self, enumerated_type: Option<&M::EnumeratedType>) {
        let Some(enumerated_type) = enumerated_type else {
            return;
        };
        self.button_group
            .set_selected(self.button_model_list.get(enumerated_type).cloned(), true);
    }
}

impl<M> Default for EtomoButtonGroup<M>
where
    M: AbstractRadioButtonModel,
    M::EnumeratedType: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{AbstractButtonBoundary, ButtonModelBoundary, EtomoButtonGroup};
    use crate::imod::etomo::ui::swing::abstract_radio_button_model::{
        AbstractRadioButtonModel, ToggleButtonModelBoundary,
    };
    use std::{cell::RefCell, rc::Rc};

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    struct TestEnumeratedType(&'static str);

    struct TestRadioButtonModel {
        enumerated_type: Option<TestEnumeratedType>,
    }

    impl ToggleButtonModelBoundary for TestRadioButtonModel {}

    impl AbstractRadioButtonModel for TestRadioButtonModel {
        type EnumeratedType = TestEnumeratedType;

        fn get_enumerated_type(&self) -> Option<&Self::EnumeratedType> {
            self.enumerated_type.as_ref()
        }
    }

    #[test]
    fn add_delegates_to_button_group_and_maps_a_radio_model_with_an_enumerated_type() {
        let model = Rc::new(RefCell::new(TestRadioButtonModel {
            enumerated_type: Some(TestEnumeratedType("first")),
        }));
        let button =
            AbstractButtonBoundary::new(ButtonModelBoundary::AbstractRadioButton(model.clone()));
        let mut group = EtomoButtonGroup::new();

        group.add(&button);

        assert_eq!(group.button_group.models.len(), 1);
        assert!(Rc::ptr_eq(
            group
                .button_model_list
                .get(&TestEnumeratedType("first"))
                .unwrap(),
            &model
        ));
    }

    #[test]
    fn add_does_not_map_a_non_radio_model_or_a_radio_model_without_an_enumerated_type() {
        let no_type = Rc::new(RefCell::new(TestRadioButtonModel {
            enumerated_type: None,
        }));
        let other = AbstractButtonBoundary::new(ButtonModelBoundary::Other);
        let no_type_button =
            AbstractButtonBoundary::new(ButtonModelBoundary::AbstractRadioButton(no_type));
        let mut group = EtomoButtonGroup::new();

        group.add(&other);
        group.add(&no_type_button);

        assert_eq!(group.button_group.models.len(), 2);
        assert!(group.button_model_list.is_empty());
    }

    #[test]
    fn set_selected_ignores_null_and_selects_the_model_saved_by_the_same_enumerated_type() {
        let model = Rc::new(RefCell::new(TestRadioButtonModel {
            enumerated_type: Some(TestEnumeratedType("selected")),
        }));
        let button =
            AbstractButtonBoundary::new(ButtonModelBoundary::AbstractRadioButton(model.clone()));
        let mut group = EtomoButtonGroup::new();
        group.add(&button);

        group.set_selected(None);
        assert!(group.button_group.set_selected_calls.is_empty());

        group.set_selected(Some(&TestEnumeratedType("selected")));
        assert!(Rc::ptr_eq(
            group.button_group.selected_model.as_ref().unwrap(),
            &model
        ));
        assert_eq!(group.button_group.set_selected_calls.len(), 1);
        assert!(group.button_group.set_selected_calls[0].1);
    }

    #[test]
    fn set_selected_for_an_unmapped_type_preserves_the_parent_null_model_call() {
        let mut group = EtomoButtonGroup::<TestRadioButtonModel>::new();

        group.set_selected(Some(&TestEnumeratedType("absent")));

        assert_eq!(group.button_group.set_selected_calls.len(), 1);
        assert!(group.button_group.set_selected_calls[0].0.is_none());
        assert!(group.button_group.selected_model.is_none());
    }
}
