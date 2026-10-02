//! `IMOD/Etomo/src/etomo/ui/swing/RadioButtonInterface.java`.
//!
//! The package-private interface a radio-button wrapper implements so that its
//! `RadioButton.RadioButtonModel` (or `RadioEbutton.RadioEButtonModel`) can
//! report the selection and hand back the enumerated type and the field.
//! Implementers are EDT objects (`Rc`, `&self` methods).
//!
//! It also holds [`EnumeratedTypeRef`], the Rust form of a Java reference to an
//! `EnumeratedType` instance.  The Java enumerated types are singletons compared
//! with `==` (`RadioButton.equals(EnumeratedType)`) and cast back to their
//! concrete class by callers (`(FilterType) model.getEnumeratedType()`); the
//! Rust enumerated types are `Copy` enums, so a reference carries the value
//! type-erased twice - as the `EnumeratedType` trait object and as `Any` - with
//! `==` being "same concrete type and equal value", which is exactly Java's
//! singleton identity.

use std::any::Any;
use std::ops::Deref;
use std::rc::Rc;

use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::ui::field::Field;

/// A Java `EnumeratedType` reference (see the module documentation).
#[derive(Clone)]
pub struct EnumeratedTypeRef {
    value: Rc<dyn EnumeratedType>,
    any: Rc<dyn Any>,
    same: fn(&dyn Any, &dyn Any) -> bool,
}

impl EnumeratedTypeRef {
    /// A reference to the instance `value`.
    pub fn new<E: EnumeratedType + PartialEq + 'static>(value: E) -> EnumeratedTypeRef {
        fn same<T: PartialEq + 'static>(a: &dyn Any, b: &dyn Any) -> bool {
            match (a.downcast_ref::<T>(), b.downcast_ref::<T>()) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
        }
        let value = Rc::new(value);
        EnumeratedTypeRef {
            value: value.clone(),
            any: value,
            same: same::<E>,
        }
    }

    /// The Java cast `(E) enumeratedType`: the concrete instance, or `None` when
    /// the reference is of another class (where Java would throw
    /// `ClassCastException`).
    pub fn downcast_ref<E: 'static>(&self) -> Option<&E> {
        self.any.downcast_ref::<E>()
    }
}

impl Deref for EnumeratedTypeRef {
    type Target = dyn EnumeratedType;
    fn deref(&self) -> &Self::Target {
        &*self.value
    }
}

/// Java `==` on two `EnumeratedType` references.
impl PartialEq for EnumeratedTypeRef {
    fn eq(&self, other: &EnumeratedTypeRef) -> bool {
        (self.same)(&*self.any, &*other.any)
    }
}

/// Java `RadioButtonInterface`.
pub trait RadioButtonInterface {
    /// Java `msgSelected()`.
    fn msg_selected(&self);

    /// Java `getEnumeratedType()`.
    fn get_enumerated_type(&self) -> Option<EnumeratedTypeRef>;

    /// Java `isEnabled()`.
    fn is_enabled(&self) -> bool;

    /// Java `getField()`: the implementer itself, as a `Field`.
    fn get_field(&self) -> Option<Rc<dyn Field>>;
}
