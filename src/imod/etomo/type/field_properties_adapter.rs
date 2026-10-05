//! `IMOD/Etomo/src/etomo/type/FieldPropertiesAdapter.java`.
//!
//! Adapter field properties from metadata to/from field settings.

use std::rc::Rc;

use super::field_properties::FieldProperties;
use super::field_settings::FieldSettings;
use crate::imod::etomo::util::event_queue::{self, EdtRef};

/// Java `public final class FieldPropertiesAdapter`.  Only static members.
pub struct FieldPropertiesAdapter;

impl FieldPropertiesAdapter {
    /// Java static `applyTo(FieldProperties, FieldSettings)`.
    pub fn apply_to_field(props: Option<&FieldProperties>, field: Option<&dyn FieldSettings>) -> i32 {
        Self::apply_to_impl(props, field)
    }

    /// Java static `applyToLater(FieldProperties, FieldSettings)`.  The field is an
    /// event dispatch thread object; the properties are read when the posted
    /// `ApplyToFieldRunnable` runs, from the copy taken here (the source passes the
    /// metadata's instance).
    pub fn apply_to_later(props: Option<&FieldProperties>, field: Option<Rc<dyn FieldSettings>>) {
        let (Some(props), Some(field)) = (props, field) else {
            return;
        };
        let runnable = ApplyToFieldRunnable::new(props.clone(), EdtRef::new(field));
        event_queue::invoke_later(move || {
            let mut runnable = runnable;
            runnable.run();
        });
    }

    /// Java package-private static `applyToImpl(FieldProperties, FieldSettings)`.
    /// Returns the absolute value of the number of field properties that could be
    /// modified.  Returns 0 if a parameter is null.  Returns a negative number if props
    /// and field are not compatible.
    pub fn apply_to_impl(props: Option<&FieldProperties>, field: Option<&dyn FieldSettings>) -> i32 {
        let (Some(props), Some(field)) = (props, field) else {
            return 0;
        };
        let mut num_applied = 0;
        let error = false;
        // Toggle button property
        if props.is_toggle_button() {
            let selected_property = props.get_selected_property();
            if let Some(selected_property) = selected_property {
                field.set_selected(selected_property);
                num_applied += 1;
            }
        }
        // Universal properties
        let editable_property = props.get_editable_property();
        if let Some(editable_property) = editable_property {
            field.set_editable(editable_property);
            num_applied += 1;
        }
        let enabled_property = props.get_enabled_property();
        if let Some(enabled_property) = enabled_property {
            field.set_enabled(enabled_property);
            num_applied += 1;
        }
        if error { -num_applied } else { num_applied }
    }

    /// Java static `applyTo(FieldSettings, FieldProperties)`.  Returns the absolute value
    /// of the number of field properties that could be modified.  Returns 0 if a
    /// parameter is null.  Returns a negative number if props and field are not
    /// compatible.
    pub fn apply_to_props(field: Option<&dyn FieldSettings>, props: Option<&mut FieldProperties>) -> i32 {
        let (Some(field), Some(props)) = (field, props) else {
            return 0;
        };
        let mut num_applied = 0;
        let error = false;
        // Toggle button property
        if props.is_toggle_button() {
            props.set_selected_property(field.is_selected());
            num_applied += 1;
        }
        // Universal properties
        props.set_editable_property(field.is_editable());
        num_applied += 1;
        props.set_enabled_property(field.is_enabled());
        num_applied += 1;
        if error { -num_applied } else { num_applied }
    }
}

/// Java private static final nested `ApplyToFieldRunnable implements Runnable`.
struct ApplyToFieldRunnable {
    /// Java private final `props`.
    props: FieldProperties,
    /// Java private final `field`.
    field: EdtRef<dyn FieldSettings>,
    /// Java private `numApplied`, initially 0.
    num_applied: i32,
}

impl ApplyToFieldRunnable {
    /// Java package-private `ApplyToFieldRunnable(FieldProperties, FieldSettings)`.
    fn new(props: FieldProperties, field: EdtRef<dyn FieldSettings>) -> ApplyToFieldRunnable {
        ApplyToFieldRunnable {
            props,
            field,
            num_applied: 0,
        }
    }

    /// Java `run()`.
    fn run(&mut self) {
        self.num_applied =
            FieldPropertiesAdapter::apply_to_impl(Some(&self.props), Some(&**self.field.get()));
    }
}
