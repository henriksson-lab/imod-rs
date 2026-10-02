//! `IMOD/Etomo/src/etomo/comscript/TiltalignSolution.java`.
#![allow(dead_code)]

use super::fortran_input_string::FortranInputString;
use super::string_list::StringList;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `TiltalignSolution`.  The fields are public in the source.
#[derive(Clone, Debug)]
pub struct TiltalignSolution {
    /// Java field `type`, default-initialised to 0.
    pub r#type: i32,
    /// Java field `referenceView`.
    pub reference_view: FortranInputString,
    /// Java field `params`.
    pub params: FortranInputString,
    /// Java field `additionalGroups`.
    pub additional_groups: StringList,
}

impl TiltalignSolution {
    /// Java `TiltalignSolution()`.
    pub fn new() -> TiltalignSolution {
        let mut params = FortranInputString::new(2);
        params.set_integer_type_index(0, true);
        params.set_integer_type_index(1, true);
        let mut reference_view = FortranInputString::new(1);
        reference_view.set_integer_type_index(0, true);
        let additional_groups = StringList::new_with_n_elements(0);
        TiltalignSolution {
            r#type: 0,
            reference_view,
            params,
            additional_groups,
        }
    }

    /// Java `TiltalignSolution(TiltalignSolution)`, the copy constructor.
    ///
    /// Deviation: the source assigns `referenceView = src.referenceView`, so the copy
    /// shares the source's `FortranInputString` object rather than copying it (only
    /// `params` and `additionalGroups` are copied).  A Rust value field cannot alias, so
    /// the reference view is cloned here.  Nothing in the source calls this constructor.
    pub fn new_from(src: &TiltalignSolution) -> TiltalignSolution {
        TiltalignSolution {
            r#type: src.r#type,
            reference_view: src.reference_view.clone(),
            params: FortranInputString::new_from_instance(&src.params),
            additional_groups: StringList::new_from(&src.additional_groups),
        }
    }
}
