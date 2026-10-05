//! `IMOD/Etomo/src/etomo/storage/DirectiveFileInterface.java`.
//!
//! Implemented by `DirectiveFile` and `DirectiveFileCollection`, whose inherent
//! methods carry the bodies; the overloads carry descriptive suffixes naming their
//! extra parameters, as on the implementors.  `setDebug` is inherent on the
//! implementors (it needs `&mut`) and is not reached through the interface by any
//! caller.  `iterator(boolean)` returns each implementor's own iterator type, so the
//! interface method [`DirectiveFileInterface::iterator_statements`] hands back the
//! statements that iterator yields, in order (null where the Java returns a null
//! iterator).

use super::directive_def::DirectiveDef;
use super::directive_file::DirectiveFile;
use super::directive_file_collection::DirectiveFileCollection;
use super::autodoc::statement::Statement;
use super::directive_value::DirectiveValue;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java `public interface DirectiveFileInterface`.
pub trait DirectiveFileInterface {
    /// Java `contains(DirectiveDef)`.
    fn contains(&self, directive_def: Option<DirectiveDef>) -> bool;

    /// Java `getValue(DirectiveDef)`.
    fn get_value(&self, directive_def: Option<DirectiveDef>) -> Option<String>;

    /// Java `getValue(DirectiveDef, int)`.
    fn get_value_index(&self, directive_def: Option<DirectiveDef>, index: i32) -> Option<String>;

    /// Java `getValue(DirectiveDef, boolean)`.
    fn get_value_template(&self, directive_def: Option<DirectiveDef>, template_only: bool) -> Option<String>;

    /// Java `getValue(DirectiveDef, boolean, boolean)`.
    fn get_value_template_ignore(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool) -> Option<Box<dyn DirectiveValue>>;

    /// Java `getValue(DirectiveDef, boolean, boolean, boolean)`.
    fn get_value_template_ignore_override(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool, include_override: bool) -> Option<Box<dyn DirectiveValue>>;

    /// Java `contains(DirectiveDef, boolean)`.
    fn contains_template(&self, directive_def: Option<DirectiveDef>, template_only: bool) -> bool;

    /// Java `contains(DirectiveDef, boolean, boolean)`.
    fn contains_template_ignore(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool) -> bool;

    /// Java `contains(DirectiveDef, boolean, boolean, boolean)`.
    fn contains_template_ignore_override(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool, include_override: bool) -> bool;

    /// Java `isValue(DirectiveDef, boolean, boolean)`.
    fn is_value_template_ignore(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool) -> bool;

    /// Java `isValue(DirectiveDef)`.
    fn is_value(&self, directive_def: Option<DirectiveDef>) -> bool;

    /// Java `contains(DirectiveDef, AxisID, boolean)`.
    fn contains_axis_template(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool) -> bool;

    /// Java `contains(DirectiveDef, AxisID, boolean, boolean)`.
    fn contains_axis_template_ignore(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool, ignore_file_type: bool) -> bool;

    /// Java `isValue(DirectiveDef, AxisID, boolean)`.
    fn is_value_axis_template(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool) -> bool;

    /// Java `getValue(DirectiveDef, AxisID, boolean, boolean)`.
    fn get_value_axis_template_ignore(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool, ignore_file_type: bool) -> Option<Box<dyn DirectiveValue>>;


    /// Java `iterator(boolean)`: the statements the iterator yields.
    fn iterator_statements(&self, template_only: bool) -> Option<Vec<*mut dyn Statement>>;
}

impl DirectiveFileInterface for DirectiveFile {
    fn iterator_statements(&self, template_only: bool) -> Option<Vec<*mut dyn Statement>> {
        DirectiveFile::iterator(self, template_only).map(|iterator| iterator.collect())
    }

    fn contains(&self, directive_def: Option<DirectiveDef>) -> bool {
        DirectiveFile::contains(self, directive_def)
    }

    fn get_value(&self, directive_def: Option<DirectiveDef>) -> Option<String> {
        DirectiveFile::get_value(self, directive_def)
    }

    fn get_value_index(&self, directive_def: Option<DirectiveDef>, index: i32) -> Option<String> {
        DirectiveFile::get_value_index(self, directive_def, index)
    }

    fn get_value_template(&self, directive_def: Option<DirectiveDef>, template_only: bool) -> Option<String> {
        DirectiveFile::get_value_template(self, directive_def, template_only)
    }

    fn get_value_template_ignore(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool) -> Option<Box<dyn DirectiveValue>> {
        DirectiveFile::get_value_template_ignore(self, directive_def, template_only, ignore_file_type)
            .map(|value| Box::new(value) as Box<dyn DirectiveValue>)
    }

    fn get_value_template_ignore_override(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool, include_override: bool) -> Option<Box<dyn DirectiveValue>> {
        DirectiveFile::get_value_template_ignore_override(self, directive_def, template_only, ignore_file_type, include_override)
            .map(|value| Box::new(value) as Box<dyn DirectiveValue>)
    }

    fn contains_template(&self, directive_def: Option<DirectiveDef>, template_only: bool) -> bool {
        DirectiveFile::contains_template(self, directive_def, template_only)
    }

    fn contains_template_ignore(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool) -> bool {
        DirectiveFile::contains_template_ignore(self, directive_def, template_only, ignore_file_type)
    }

    fn contains_template_ignore_override(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool, include_override: bool) -> bool {
        DirectiveFile::contains_template_ignore_override(self, directive_def, template_only, ignore_file_type, include_override)
    }

    fn is_value_template_ignore(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool) -> bool {
        DirectiveFile::is_value_template_ignore(self, directive_def, template_only, ignore_file_type)
    }

    fn is_value(&self, directive_def: Option<DirectiveDef>) -> bool {
        DirectiveFile::is_value(self, directive_def)
    }

    fn contains_axis_template(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool) -> bool {
        DirectiveFile::contains_axis_template(self, directive_def, axis_id, template_only)
    }

    fn contains_axis_template_ignore(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool, ignore_file_type: bool) -> bool {
        DirectiveFile::contains_axis_template_ignore(self, directive_def, axis_id, template_only, ignore_file_type)
    }

    fn is_value_axis_template(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool) -> bool {
        DirectiveFile::is_value_axis_template(self, directive_def, axis_id, template_only)
    }

    fn get_value_axis_template_ignore(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool, ignore_file_type: bool) -> Option<Box<dyn DirectiveValue>> {
        DirectiveFile::get_value_axis_template_ignore(self, directive_def, axis_id, template_only, ignore_file_type)
            .map(|value| Box::new(value) as Box<dyn DirectiveValue>)
    }

}

impl DirectiveFileInterface for DirectiveFileCollection {
    fn iterator_statements(&self, template_only: bool) -> Option<Vec<*mut dyn Statement>> {
        let mut iterator = DirectiveFileCollection::iterator(self, template_only);
        let mut statements = Vec::new();
        while iterator.has_next() {
            if let Some(statement) = iterator.next() {
                statements.push(statement);
            }
        }
        Some(statements)
    }

    fn contains(&self, directive_def: Option<DirectiveDef>) -> bool {
        DirectiveFileCollection::contains(self, directive_def)
    }

    fn get_value(&self, directive_def: Option<DirectiveDef>) -> Option<String> {
        DirectiveFileCollection::get_value(self, directive_def)
    }

    fn get_value_index(&self, directive_def: Option<DirectiveDef>, index: i32) -> Option<String> {
        DirectiveFileCollection::get_value_index(self, directive_def, index)
    }

    fn get_value_template(&self, directive_def: Option<DirectiveDef>, template_only: bool) -> Option<String> {
        DirectiveFileCollection::get_value_template(self, directive_def, template_only)
    }

    fn get_value_template_ignore(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool) -> Option<Box<dyn DirectiveValue>> {
        DirectiveFileCollection::get_value_template_ignore(self, directive_def, template_only, ignore_file_type)
    }

    fn get_value_template_ignore_override(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool, include_override: bool) -> Option<Box<dyn DirectiveValue>> {
        DirectiveFileCollection::get_value_template_ignore_override(self, directive_def, template_only, ignore_file_type, include_override)
    }

    fn contains_template(&self, directive_def: Option<DirectiveDef>, template_only: bool) -> bool {
        DirectiveFileCollection::contains_template(self, directive_def, template_only)
    }

    fn contains_template_ignore(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool) -> bool {
        DirectiveFileCollection::contains_template_ignore(self, directive_def, template_only, ignore_file_type)
    }

    fn contains_template_ignore_override(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool, include_override: bool) -> bool {
        DirectiveFileCollection::contains_template_ignore_override(self, directive_def, template_only, ignore_file_type, include_override)
    }

    fn is_value_template_ignore(&self, directive_def: Option<DirectiveDef>, template_only: bool, ignore_file_type: bool) -> bool {
        DirectiveFileCollection::is_value_template_ignore(self, directive_def, template_only, ignore_file_type)
    }

    fn is_value(&self, directive_def: Option<DirectiveDef>) -> bool {
        DirectiveFileCollection::is_value(self, directive_def)
    }

    fn contains_axis_template(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool) -> bool {
        DirectiveFileCollection::contains_axis_template(self, directive_def, axis_id, template_only)
    }

    fn contains_axis_template_ignore(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool, ignore_file_type: bool) -> bool {
        DirectiveFileCollection::contains_axis_template_ignore(self, directive_def, axis_id, template_only, ignore_file_type)
    }

    fn is_value_axis_template(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool) -> bool {
        DirectiveFileCollection::is_value_axis_template(self, directive_def, axis_id, template_only)
    }

    fn get_value_axis_template_ignore(&self, directive_def: Option<DirectiveDef>, axis_id: Option<AxisID>, template_only: bool, ignore_file_type: bool) -> Option<Box<dyn DirectiveValue>> {
        DirectiveFileCollection::get_value_axis_template_ignore(self, directive_def, axis_id, template_only, ignore_file_type)
    }

}
