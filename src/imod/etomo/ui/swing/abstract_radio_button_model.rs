//! `IMOD/Etomo/src/etomo/ui/swing/AbstractRadioButtonModel.java`.
//!
//! ```java
//! abstract class AbstractRadioButtonModel extends JToggleButton.ToggleButtonModel {
//!   public abstract EnumeratedType getEnumeratedType();
//! }
//! ```
//!
//! The Swing superclass `JToggleButton.ToggleButtonModel` is the jdk stand-in's
//! button-model hook (`crate::imod::etomo::jdk::ButtonModel`): the selected
//! state itself lives in the `JComponent`, which calls the model's
//! `set_selected` after it has done what `ToggleButtonModel.setSelected` does
//! (group bookkeeping, state change, item/change events).  A subclass's
//! `setSelected` override that begins with `super.setSelected(selected)` is
//! therefore the trait method's body without that first statement.
//!
//! `ButtonGroup.getSelection()` returns the selected member's model, which
//! callers cast (`(RadioButton.RadioButtonModel) bg.getSelection()`); the
//! Rust callers reach it with `button.get_model()` and
//! `model.as_any().downcast_ref::<RadioButtonModel>()`.

use crate::imod::etomo::jdk::ButtonModel;
use crate::imod::etomo::ui::swing::radio_button_interface::EnumeratedTypeRef;

/// Java `AbstractRadioButtonModel`.
pub trait AbstractRadioButtonModel: ButtonModel {
    /// Java `getEnumeratedType()`.
    fn get_enumerated_type(&self) -> Option<EnumeratedTypeRef>;
}
