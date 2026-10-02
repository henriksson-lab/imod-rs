//! `IMOD/Etomo/src/etomo/ui/swing/TextComponentAppearanceExtension.java`.
//!
//! An extension of AppearanceExtension which uses `JTextComponent.setEditable()`.
//!
//! Java `extends AppearanceExtension`: the superclass is embedded as `base` (with
//! `Deref`), and the two overridden methods are in the
//! [`AppearanceExtensionVirtual`] implementation.  The superclass constructor already
//! dispatches `setComponentEditable` to this class, so construction follows
//! `AppearanceExtension`'s split: `construct`, `set_this`, `constructor_body`.

use std::ops::Deref;
use std::rc::{Rc, Weak};

use super::appearance_extension::{AppearanceExtension, AppearanceExtensionVirtual};
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::ui::flag_display::FlagDisplay;
use crate::imod::etomo::ui::flag_type::FlagType;

/// Java `final class TextComponentAppearanceExtension extends AppearanceExtension`.
pub struct TextComponentAppearanceExtension {
    base: AppearanceExtension,
}

impl Deref for TextComponentAppearanceExtension {
    type Target = AppearanceExtension;
    fn deref(&self) -> &AppearanceExtension {
        &self.base
    }
}

impl TextComponentAppearanceExtension {
    /// Java `TextComponentAppearanceExtension(JTextComponent, boolean, boolean)`.
    /// `textComponent` is required.
    pub fn new(
        text_component: &Rc<JComponent>,
        enabled_field: bool,
        editable_component: bool,
    ) -> Rc<TextComponentAppearanceExtension> {
        // super(textComponent, enabledField, editableComponent, textComponent.isEditable())
        let extension = Rc::new(TextComponentAppearanceExtension {
            base: AppearanceExtension::construct(
                text_component,
                enabled_field,
                editable_component,
                text_component.is_editable(),
            ),
        });
        extension
            .base
            .set_this(Rc::downgrade(&extension) as Weak<dyn AppearanceExtensionVirtual>);
        extension.base.constructor_body();
        extension
    }
}

impl AppearanceExtensionVirtual for TextComponentAppearanceExtension {
    fn appearance_extension(&self) -> &AppearanceExtension {
        &self.base
    }

    /// Java `@Override setComponentEditable(boolean)`.  Make the component editable or
    /// ineditable.
    fn set_component_editable(&self, editable: bool) {
        // Ineditable components are never editable, even when the field is editable.
        if !editable || self.base.editable_component {
            // ((JTextComponent) component).setEditable(editable)
            self.base.component.set_editable(editable);
        }
    }

    /// Java `@Override isNativeSetEditable()`.
    fn is_native_set_editable(&self) -> bool {
        true
    }
}

impl FlagDisplay for TextComponentAppearanceExtension {
    /// Java `setFlag(FlagType)`, inherited from `AppearanceExtension`.
    fn set_flag(&self, flag_type: Option<&'static FlagType>) {
        self.base.set_flag(flag_type);
    }
}
