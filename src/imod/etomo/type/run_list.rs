//! `IMOD/Etomo/src/etomo/type/RunList.java`.
//!
//! List of an ID and run status.  Created on the event dispatch thread by the
//! batchruntomo dialog and handed to the batchruntomo monitors, which run on their
//! own threads, so it is shared through an `Arc` and each method locks the list.

use std::sync::Mutex;

use super::run_status::RunStatus;

/// Java `public final class RunList`.
#[derive(Default)]
pub struct RunList {
    /// Java private final `list`.
    list: Mutex<Vec<Element>>,
}

impl RunList {
    /// Java `RunList()`.
    pub fn new() -> RunList {
        RunList {
            list: Mutex::new(Vec::new()),
        }
    }

    /// Java `add(String, RunStatus, boolean)`.  Adds a new element to the list if stackID
    /// is not null.
    pub fn add(&self, stack_id: Option<&str>, run_status: Option<RunStatus>, dual: bool) {
        if let Some(stack_id) = stack_id {
            self.list
                .lock()
                .unwrap()
                .push(Element::new(stack_id, run_status, dual));
        }
    }

    /// Java `size()`.
    pub fn size(&self) -> i32 {
        self.list.lock().unwrap().len() as i32
    }

    /// Java `setRootName(String, String)`.
    pub fn set_root_name(&self, stack_id: Option<&str>, root_name: Option<&str>) {
        let mut list = self.list.lock().unwrap();
        if let Some(element) = Self::get_in(&mut list, stack_id) {
            element.root_name = root_name.map(str::to_owned);
        }
    }

    // <p>Updates done</p>

    /// Java `setStackLocation(String, String)`.
    pub fn set_stack_location(&self, stack_id: Option<&str>, stack_location: Option<&str>) {
        let mut list = self.list.lock().unwrap();
        if let Some(element) = Self::get_in(&mut list, stack_id) {
            element.stack_location = stack_location.map(str::to_owned);
        }
    }

    /// Java `get(String)`'s search, over the locked list.  Java compares
    /// `element.stackID.equals(stackID)`; a null `stackID` matches nothing.
    fn get_in<'a>(list: &'a mut [Element], stack_id: Option<&str>) -> Option<&'a mut Element> {
        list.iter_mut()
            .find(|element| Some(element.stack_id.as_str()) == stack_id)
    }

    /// Java `get(String)`: a copy of the element (Java returns the element; every
    /// caller only reads it).
    pub fn get(&self, stack_id: Option<&str>) -> Option<Element> {
        let mut list = self.list.lock().unwrap();
        Self::get_in(&mut list, stack_id).map(|element| element.clone())
    }

    /// Java `hasDual()`.
    pub fn has_dual(&self) -> bool {
        self.list.lock().unwrap().iter().any(|element| element.dual)
    }

    /// Java `size(RunStatus)`.  Returns the number of elements containing the parameter
    /// runStatus.
    pub fn size_run_status(&self, run_status: Option<RunStatus>) -> i32 {
        let mut size = 0;
        for element in self.list.lock().unwrap().iter() {
            if element.run_status == run_status {
                size += 1;
            }
        }
        size
    }

    /// Java `equalsRunStatus(int, RunStatus)`.  Return true if element run status matches
    /// parameter.
    pub fn equals_run_status(&self, index: i32, run_status: Option<RunStatus>) -> bool {
        let list = self.list.lock().unwrap();
        if index < 0 || index as usize >= list.len() {
            return false;
        }
        list[index as usize].run_status == run_status
    }

    /// Java `getNumDone()`.  Returns the number of completed elements.
    pub fn get_num_done(&self) -> i32 {
        let mut num_done = 0;
        for element in self.list.lock().unwrap().iter() {
            if element.run_status == Some(RunStatus::Ran) {
                num_done += 1;
            }
        }
        num_done
    }
}

/// Java `toString()`.
impl std::fmt::Display for RunList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let list = self.list.lock().unwrap();
        let elements: Vec<String> = list.iter().map(|element| element.to_string()).collect();
        write!(f, "[list:[{}]]", elements.join(", "))
    }
}

/// Java `public static final class Element`.
#[derive(Clone, Debug)]
pub struct Element {
    /// Java private final `stackID`.
    stack_id: String,
    /// Java private final `runStatus`.
    run_status: Option<RunStatus>,
    /// Java private final `dual`.
    dual: bool,
    /// Java private `rootName`, initially null.
    root_name: Option<String>,
    /// Java private `stackLocation`, initially null.
    stack_location: Option<String>,
}

impl Element {
    /// Java private `Element(String, RunStatus, boolean)`.
    fn new(stack_id: &str, run_status: Option<RunStatus>, dual: bool) -> Element {
        Element {
            stack_id: stack_id.to_owned(),
            run_status,
            dual,
            root_name: None,
            stack_location: None,
        }
    }

    /// Java `getRootName()`.
    pub fn get_root_name(&self) -> Option<&str> {
        self.root_name.as_deref()
    }

    /// Java `getStackLocation()`.
    pub fn get_stack_location(&self) -> Option<&str> {
        self.stack_location.as_deref()
    }
}

/// Java `Element.toString()`.
impl std::fmt::Display for Element {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[stackID:{},runStatus:{},rootName:{},stackLocation:{}]",
            self.stack_id,
            self.run_status
                .map_or("null".to_owned(), |run_status| run_status.to_string()),
            self.root_name.as_deref().unwrap_or("null"),
            self.stack_location.as_deref().unwrap_or("null")
        )
    }
}
