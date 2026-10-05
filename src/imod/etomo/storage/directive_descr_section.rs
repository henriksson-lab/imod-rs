//! `IMOD/Etomo/src/etomo/storage/DirectiveDescrSection.java`.
//!
//! A section of the directives description file (`directives.csv`): its header and the
//! keys of the directives listed under it.  Built by `DirectiveEditorBuilder` and read by
//! the directive editor's panels, all on the event dispatch thread; shared as
//! `Rc<DirectiveDescrSection>` with its mutable fields in cells.

use std::cell::{Cell, RefCell};

use crate::imod::etomo::storage::directive::Directive;
use crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java `public final class DirectiveDescrSection`.
pub struct DirectiveDescrSection {
    /// Java private final `nameArray = new ArrayList<String>()`.
    name_array: RefCell<Vec<String>>,
    /// Java private final `header`.
    header: Option<String>,
    /// Java private `containsEditableDirectives`, initialised to true.
    contains_editable_directives: Cell<bool>,
}

impl DirectiveDescrSection {
    /// Java `DirectiveDescrSection(String)`.
    pub fn new(header: Option<&str>) -> DirectiveDescrSection {
        DirectiveDescrSection {
            name_array: RefCell::new(Vec::new()),
            header: header.map(str::to_string),
            contains_editable_directives: Cell::new(true),
        }
    }

    /// Java `add(Directive)`.
    pub fn add_directive(&self, directive: Option<&Directive>) {
        let Some(directive) = directive else {
            return;
        };
        if !directive.is_valid() {
            return;
        }
        // `nameArray.add(directive.getKey())`: an ArrayList accepts null.
        self.name_array
            .borrow_mut()
            .push(directive.get_key().unwrap_or_else(|| "null".to_string()));
    }

    /// Java `add(String)`.
    pub fn add_string(&self, directive_name: Option<&str>) {
        if let Some(directive_name) = directive_name
            && !java_lang_string_matches_whitespace(directive_name)
        {
            self.name_array
                .borrow_mut()
                .push(directive_name.to_string());
        }
    }

    /// Java `size()`.
    pub fn size(&self) -> i32 {
        self.name_array.borrow().len() as i32
    }

    /// Java `setContainsEditableDirectives(boolean)`.
    pub fn set_contains_editable_directives(&self, input: bool) {
        self.contains_editable_directives.set(input);
    }

    /// Java `isContainsEditableDirectives()`.
    pub fn is_contains_editable_directives(&self) -> bool {
        self.contains_editable_directives.get()
    }

    /// Java `nameIterator()`.  A copy of the list: nothing adds to a section while its
    /// names are iterated.
    pub fn name_iterator(&self) -> std::vec::IntoIter<String> {
        self.name_array.borrow().clone().into_iter()
    }
}

/// Java `toString()`: the header.
impl std::fmt::Display for DirectiveDescrSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.header.as_deref().unwrap_or("null"))
    }
}
