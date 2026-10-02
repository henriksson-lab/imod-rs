//! `IMOD/Etomo/src/etomo/process/ContinuousListenerTarget.java`.

use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java interface `ContinuousListenerTarget`.
///
/// The target receives messages from `ImodProcess`'s continuous listener thread, so an
/// implementor is shared across threads.  `axisID` is nullable in the source (several
/// `ImodProcess` constructors never set it), hence `Option`.
pub trait ContinuousListenerTarget: Send + Sync {
    /// Java `getContinuousMessage(String, AxisID)`.
    fn get_continuous_message(&self, message: &str, axis_id: Option<AxisID>);
}

/// A Java reference to a (leaked, `'static`) target, handed to a 3dmod state that
/// keeps it.
impl<T: ContinuousListenerTarget + ?Sized> ContinuousListenerTarget for &'static T {
    fn get_continuous_message(&self, message: &str, axis_id: Option<AxisID>) {
        (**self).get_continuous_message(message, axis_id)
    }
}
