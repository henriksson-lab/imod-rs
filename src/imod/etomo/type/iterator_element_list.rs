//! `IMOD/Etomo/src/etomo/type/IteratorElementList.java`.
//!
//! A list of `IteratorElement`s parsed from text such as "2,4 - 9,10".

use std::collections::BTreeMap;

use super::axis_id::AxisID;
use super::iterator_element::IteratorElement;
use super::iterator_parser::IteratorParser;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::ui::swing::anisotropic_diffusion_dialog;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class IteratorElementList`.
pub struct IteratorElementList {
    /// Java private final `list`: list of elements.  List may be cleared and reused.
    list: Vec<IteratorElement>,
    /// Java private final `manager`.
    manager: Option<&'static dyn BaseManager>,
    /// Java private final `axisID`.
    axis_id: Option<AxisID>,
    /// Java private final `label`.
    label: Option<String>,
    /// Java private `parser`, initially null.
    parser: Option<IteratorParser>,
}

impl IteratorElementList {
    /// Java `IteratorElementList(BaseManager, AxisID, String)`.
    pub fn new(
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        label: Option<&str>,
    ) -> IteratorElementList {
        IteratorElementList {
            list: Vec::new(),
            manager,
            axis_id,
            label: label.map(str::to_owned),
            parser: None,
        }
    }

    /// Java `setList(String)`.  Sets a new list by parsing input.
    pub fn set_list_string(&mut self, input: Option<&str>) {
        self.list.clear();
        let Some(input) = input else {
            return;
        };
        if input
            .chars()
            .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r'))
        {
            return;
        }
        let mut parser = self.parser.take().unwrap_or_else(|| {
            IteratorParser::new(
                self.manager,
                self.axis_id,
                Some(anisotropic_diffusion_dialog::ITERATION_LIST_LABEL),
            )
        });
        parser.parse(Some(input), Some(self));
        self.parser = Some(parser);
    }

    /// Java `setList(IteratorElementList)`.  Adds the elements in input.list to list.
    /// The elements are IteratorElements, which are immutable.
    pub fn set_list(&mut self, input: Option<&IteratorElementList>) {
        self.list.clear();
        let Some(input) = input else {
            return;
        };
        for element in &input.list {
            self.list.push(element.clone());
        }
    }

    /// Java `add(IteratorElement)`.
    pub fn add(&mut self, element: IteratorElement) {
        self.list.push(element);
    }

    /// Java `isValid()`.  If the parser hasn't been used - always valid.  If the
    /// parser has been used return parser.isValid.
    pub fn is_valid(&self) -> bool {
        match &self.parser {
            None => true,
            Some(parser) => parser.is_valid(),
        }
    }

    /// Java `getExpandedList()`.  Returns a list of strings containing all the
    /// integers specified by the numbers and ranges in list.  (Java adds a null
    /// `getNumber()` to the list; it prints as "null".)
    pub fn get_expanded_list(&self) -> Vec<Option<String>> {
        let mut expanded_list = Vec::new();
        for element in &self.list {
            if !element.is_range() {
                expanded_list.push(element.get_number());
            } else {
                expanded_list.extend(element.get_range().into_iter().map(Some));
            }
        }
        expanded_list
    }

    /// Java `store(Properties, String)`.
    pub fn store(&self, props: Option<&mut BTreeMap<String, String>>, prepend: Option<&str>) {
        let Some(props) = props else {
            return;
        };
        // Java computes makePropertyKey(prepend) and then recomputes the same key.
        let properties_key = self.make_property_key(prepend);
        if self.list.is_empty() {
            props.remove(&properties_key);
        } else {
            props.insert(properties_key, self.to_string());
        }
    }

    /// Java `load(Properties, String)`.
    pub fn load(&mut self, props: Option<&BTreeMap<String, String>>, prepend: Option<&str>) {
        match props {
            None => self.list.clear(),
            Some(props) => {
                let value = props.get(&self.make_property_key(prepend)).cloned();
                self.set_list_string(value.as_deref());
            }
        }
    }

    /// Java private `makePropertyKey(String)`.
    fn make_property_key(&self, prepend: Option<&str>) -> String {
        let label = self.label.as_deref().unwrap_or("null");
        match prepend {
            Some(prepend)
                if !prepend
                    .chars()
                    .all(|c| matches!(c, ' ' | '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r')) =>
            {
                format!("{prepend}.{label}")
            }
            _ => label.to_owned(),
        }
    }
}

impl Clone for IteratorElementList {
    /// Not a Java member: a list a param hands to the state (Java passes the
    /// reference and `ParallelState.setTestIterationList` copies its elements).
    fn clone(&self) -> IteratorElementList {
        IteratorElementList {
            list: self.list.clone(),
            manager: self.manager,
            axis_id: self.axis_id,
            label: self.label.clone(),
            parser: None,
        }
    }
}

/// Java `toString()`.
impl std::fmt::Display for IteratorElementList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iterator = self.list.iter();
        if let Some(element) = iterator.next() {
            write!(f, "{element}")?;
        }
        for element in iterator {
            write!(f, ",{element}")?;
        }
        Ok(())
    }
}
