//! `IMOD/Etomo/src/etomo/ui/swing/BooleanComboBoxEfield.java`.
//!
//! A `ComboBoxEfield` with the choices empty / Yes / No.  `final class
//! BooleanComboBoxEfield extends ComboBoxEfield`: the superclass is the `base` field,
//! built with this object as its `this` (see `combo_box_efield.rs`); the interfaces the
//! superclass implements are bound to the superclass's bodies.

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::combo_box_efield::{ComboBoxEfield, ComboBoxEfieldVirtual};
use super::control_state::ControlState;
use super::control_target::ControlTarget;
use super::swing_component::SwingComponent;
use super::text_efield_interface::TextEfieldInterface;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::storage::directive_attribute;
use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::option::Option as TypeOption;
use crate::imod::etomo::ui::flag_origin_listener::FlagOriginListener;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::text_flag_origin::TextFlagOrigin;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::ui::value_manipulation_field::ValueManipulationField;
use crate::imod::etomo::ui::value_manipulation_listener::ValueManipulationListener;

/// Java private static final `EMPTY_INDEX`.
#[allow(dead_code)]
const EMPTY_INDEX: i32 = 0;
/// Java private static final `TRUE_INDEX`.
const TRUE_INDEX: i32 = 1;
/// Java private static final `FALSE_INDEX`.
const FALSE_INDEX: i32 = 2;

/// Java `final class BooleanComboBoxEfield extends ComboBoxEfield`.
pub struct BooleanComboBoxEfield {
    /// Java superclass `ComboBoxEfield`.
    base: ComboBoxEfield,
}

impl Deref for BooleanComboBoxEfield {
    type Target = ComboBoxEfield;
    fn deref(&self) -> &ComboBoxEfield {
        &self.base
    }
}

impl BooleanComboBoxEfield {
    /// Java package-private `BooleanComboBoxEfield(String)`.
    pub fn new(label: Option<&str>) -> Rc<BooleanComboBoxEfield> {
        let instance = Rc::new_cyclic(|this: &Weak<BooleanComboBoxEfield>| BooleanComboBoxEfield {
            // super(label, false, false)
            base: ComboBoxEfield::new(this.clone(), label, false, false),
        });
        instance.base.add_item(Some(TypeOption::new_string_string(
            Some(directive_attribute::TRUE_VALUE),
            Some(shared_strings::TRUE_STRING),
        )));
        instance.base.add_item(Some(TypeOption::new_string_string(
            Some(directive_attribute::FALSE_VALUE),
            Some(shared_strings::FALSE_STRING),
        )));
        instance.base.set_choice_list_set(true);
        instance
    }

    /// Java package-private `setSelected(boolean)`.
    pub fn set_selected(&self, value: bool) {
        if value {
            self.base.set_selected_index(TRUE_INDEX);
        } else {
            self.base.set_selected_index(FALSE_INDEX);
        }
        self.base.update_flag_extension();
    }

    /// Java package-private `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.base.get_selected_index() == TRUE_INDEX
    }
}

impl ComboBoxEfieldVirtual for BooleanComboBoxEfield {
    fn combo_box_efield(&self) -> &ComboBoxEfield {
        &self.base
    }
}

// ---- the superclass's interfaces, bound to its bodies ----

impl SwingComponent for BooleanComboBoxEfield {
    fn get_component(&self) -> Rc<JComponent> {
        ComboBoxEfield::get_component(&self.base)
    }
}

impl UIComponent for BooleanComboBoxEfield {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        ComboBoxEfield::get_component(&self.base)
    }
}

impl TextFlagOrigin for BooleanComboBoxEfield {
    fn equals(&self, value: Option<&str>) -> bool {
        ComboBoxEfield::equals(&self.base, value)
    }
    fn add_flag_origin_listener(&self, listener: Rc<dyn FlagOriginListener>) {
        ComboBoxEfield::add_flag_origin_listener(&self.base, listener)
    }
    fn is_valid(&self) -> bool {
        ComboBoxEfield::is_valid(&self.base)
    }
}

impl TextEfieldInterface for BooleanComboBoxEfield {
    fn get_directive_def(&self) -> Option<DirectiveDef> {
        ComboBoxEfield::get_directive_def(&self.base)
    }
    fn is_enabled(&self) -> bool {
        ComboBoxEfield::is_enabled(&self.base)
    }
    fn is_visible(&self) -> bool {
        ComboBoxEfield::is_visible(&self.base)
    }
    fn get_text(&self) -> Option<String> {
        ComboBoxEfield::get_text(&self.base)
    }
    fn set_text(&self, text: Option<&str>) {
        ComboBoxEfield::set_text_string(&self.base, text)
    }
    fn set_field_highlight(&self, text: Option<&str>) {
        ComboBoxEfield::set_field_highlight(&self.base, text)
    }
    fn set_template_value(&self) {
        ComboBoxEfield::set_template_value(&self.base)
    }
    fn equals(&self, string: Option<&str>) -> bool {
        ComboBoxEfield::equals(&self.base, string)
    }
    fn set_debug(&self, debug: bool) {
        ComboBoxEfield::set_debug(&self.base, debug)
    }
}

impl ValueManipulationField for BooleanComboBoxEfield {
    fn add_value_manipulation_listener(&self, listener: Rc<dyn ValueManipulationListener>) {
        ComboBoxEfield::add_value_manipulation_listener(&self.base, listener)
    }
    fn is_empty(&self) -> bool {
        ComboBoxEfield::is_empty(&self.base)
    }
    fn set_text(&self, text: Option<&str>) {
        ComboBoxEfield::set_text_string(&self.base, text)
    }
}

impl ControlTarget for BooleanComboBoxEfield {
    fn clear(&self) {
        ComboBoxEfield::clear(&self.base)
    }
    fn set_text_file(&self, file: Option<&Path>) {
        ComboBoxEfield::set_text_file(&self.base, file)
    }
    fn set_text_file_array(&self, files: Option<&[PathBuf]>) {
        ComboBoxEfield::set_text_file_array(&self.base, files)
    }
    fn get_label(&self) -> Option<String> {
        ComboBoxEfield::get_label(&self.base)
    }
    fn set_component_control(&self, control: bool, state: Option<&'static ControlState>) {
        ComboBoxEfield::set_component_control(&self.base, control, state)
    }
    fn set_enable_control(&self, control: bool, state: Option<&'static ControlState>) {
        ComboBoxEfield::set_enable_control(&self.base, control, state)
    }
    /// Java `@Override sendControlEvent()` (empty, in the superclass).
    fn send_control_event(&self) {}
    fn is_local_dir(&self, current_directory: Option<&str>) -> bool {
        ComboBoxEfield::is_local_dir(&self.base, current_directory)
    }
}
