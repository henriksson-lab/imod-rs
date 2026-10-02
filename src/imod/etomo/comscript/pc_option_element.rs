//! `IMOD/Etomo/src/etomo/comscript/PcOptionElement.java`.
//!
//! The `pc-option` maps built by `CpuAdoc.load` and `Node.load` are Java
//! `LinkedHashMap<String, PcOptionElement>`s: insertion-ordered, keyed by the option
//! name.  [`PcOptionsMap`] keeps that order as a vector of entries.

use crate::imod::etomo::storage::pc_option_type::PcOptionType;

/// Java `LinkedHashMap<String, PcOptionElement>`, in insertion order.  Keys are unique.
pub type PcOptionsMap = Vec<(String, PcOptionElement)>;

/// Java `PcOptionElement`.
#[derive(Clone, Debug)]
pub struct PcOptionElement {
    /// Java private final field `key`.
    key: Option<String>,
    /// Java private field `computer`, initialised to null.
    computer: Option<String>,
    /// Java private field `queue`, initialised to null.
    queue: Option<String>,
    /// Java private field `both`, initialised to null.
    both: Option<String>,
}

impl PcOptionElement {
    /// Java `PcOptionElement(String, PcOptionType, String)`.  A null type sets `both`.
    pub fn new(key: Option<&str>, r#type: Option<PcOptionType>, value: Option<&str>) -> Self {
        let mut element = PcOptionElement {
            key: key.map(str::to_string),
            computer: None,
            queue: None,
            both: None,
        };
        element.set_pc_option_element(r#type, value);
        element
    }

    /// Java `getComputer`.
    pub fn get_computer(&self) -> Option<String> {
        if self.computer.is_some() {
            self.computer.clone()
        } else {
            self.both.clone()
        }
    }

    /// Java `getQueue`.
    pub fn get_queue(&self) -> Option<String> {
        if self.queue.is_some() {
            self.queue.clone()
        } else {
            self.both.clone()
        }
    }

    /// Java `getOption`.
    pub fn get_option(&self) -> Option<String> {
        self.key.clone()
    }

    /// Java `setPcOptionElement(PcOptionType, String)`.
    pub fn set_pc_option_element(&mut self, r#type: Option<PcOptionType>, value: Option<&str>) {
        if r#type == Some(PcOptionType::PcOptionTypeComputer) {
            self.computer = value.map(str::to_string);
        } else if r#type == Some(PcOptionType::PcOptionTypeQueue) {
            self.queue = value.map(str::to_string);
        } else {
            self.both = value.map(str::to_string);
        }
    }

    /// Java `getValue(PcOptionType)`.
    pub fn get_value(&self, r#type: PcOptionType) -> Option<String> {
        if r#type == PcOptionType::PcOptionTypeQueue {
            self.get_queue()
        } else {
            self.get_computer()
        }
    }
}
