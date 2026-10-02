//! `IMOD/Etomo/src/etomo/type/BaseState.java`.
//!
//! The abstract parent of the dataset "state" objects (`TomogramState`, ...).
//!
//! Java's `store(Properties, String)` and `load(Properties, String)` bodies only run
//! `prepend = createPrepend(prepend)` and discard the result (the `// reset` and
//! `// load` sections are empty); subclasses call them as `super.store`/`super.load`,
//! which is a no-op, and say so at the call site.  The abstract `createPrepend` and the
//! `equals(BaseState)` default are the trait below; `Storable` is its supertrait, as
//! the Java class implements it.

use crate::imod::etomo::storage::storable::{Storable, StorableValue};

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java abstract class `BaseState`.
pub trait BaseState: Storable {
    /// Java abstract package-private `createPrepend(String)`.
    fn create_prepend(&self, prepend: &str) -> String;

    /// Java `equals(BaseState)`.
    fn equals(&self, input: &dyn BaseState) -> bool {
        let _ = input;
        true
    }
}
