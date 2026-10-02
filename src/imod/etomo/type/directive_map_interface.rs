//! `IMOD/Etomo/src/etomo/type/DirectiveMapInterface.java`.
//!
//! The implementing class (`storage::DirectiveMap`) overloads `getDirective`, so the
//! interface's `getDirective(DirectiveDef)` carries the overload suffix
//! `get_directive_directive_def`.

use std::sync::Arc;

use crate::imod::etomo::storage::directive_def::DirectiveDef;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::directive_interface::DirectiveInterface;

/// Java `DirectiveMapInterface`.
pub trait DirectiveMapInterface: Send + Sync {
    /// Java `getDirectiveFromPair(DirectiveDef, AxisID)`.  Gets the directive based on
    /// pairAxisID and directiveDef.  If directiveDef is the member of a pair of
    /// directives (CopyArg with an A and a B form), then the directive corresponding to
    /// pairAxisID is returned.
    fn get_directive_from_pair(
        &self,
        directive_def: Option<DirectiveDef>,
        pair_axis_id: Option<AxisID>,
    ) -> Option<Arc<dyn DirectiveInterface>>;

    /// Java `getDirective(DirectiveDef)`.
    fn get_directive_directive_def(
        &self,
        directive_def: Option<DirectiveDef>,
    ) -> Option<Arc<dyn DirectiveInterface>>;
}
