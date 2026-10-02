//! `IMOD/Etomo/src/etomo/ui/swing/AbstractProcessResultDisplayFactory.java`.
//!
//! Description: A single direction link list of ProcessResultDisplays, which
//! labels each display with an ID that must be unique to the dataset.
//!
//! The list is the displays' own `next` links (`ProcessResultDisplay.setNext`
//! / `getNext`), exactly as in the Java; this struct holds only the head and
//! the tail.  A concrete factory embeds it as field `base` and derefs to it.
//! Factories and displays are event-dispatch-thread objects: every method
//! takes `&self` and the list ends sit in `RefCell`s, borrowed only for the
//! statement that reads or writes them.

use crate::imod::etomo::process::process_result_display_factory_interface::ProcessResultDisplayFactoryInterface;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use std::cell::RefCell;
use std::rc::Rc;

/// Java `public abstract class AbstractProcessResultDisplayFactory implements
/// ProcessResultDisplayFactoryInterface`.
pub struct AbstractProcessResultDisplayFactory {
    /// Java private final `factoryID`.
    ///
    /// factoryID must be unique within the dataset. If there are multiple
    /// instances of the same class of factory in the dataset (as in one for
    /// each axis), they must have different IDs.
    ///
    /// The factory ID will be saved to files as process data, so it must be
    /// the same each time etomo runs. However, because process data is
    /// short-lived, this ID doesn't have to stable from version to version.
    factory_id: String,
    // Dependency list
    /// Java private `head`.
    head: RefCell<Option<ProcessResultDisplayHandle>>,
    /// Java private `tail`.
    tail: RefCell<Option<ProcessResultDisplayHandle>>,
}

impl AbstractProcessResultDisplayFactory {
    /// Java protected `AbstractProcessResultDisplayFactory(String)`.
    pub fn new(factory_id: String) -> AbstractProcessResultDisplayFactory {
        AbstractProcessResultDisplayFactory {
            factory_id,
            head: RefCell::new(None),
            tail: RefCell::new(None),
        }
    }

    /// Java protected synchronized `addDependency(ProcessResultDisplay, int)`.
    /// Adds display to this instance's link list.  Does not follow display's
    /// links.  Sets the display's displayID and factoryID.
    ///
    /// Java `synchronized`: the factory is used only on the event dispatch
    /// thread, so there is nothing to lock.
    pub fn add_dependency(&self, display: Option<&ProcessResultDisplayHandle>, display_id: i32) {
        let Some(display) = display else {
            eprintln!("Error: display is null");
            // Thread.dumpStack(): no Rust counterpart for a Java stack dump.
            return;
        };

        // Add display to the end of the linked list.
        display.set_id(display_id, self.factory_id.clone());
        let tail = self.tail.borrow().clone();
        if let Some(tail) = tail {
            tail.set_next(Some(display.clone()));
        }
        *self.tail.borrow_mut() = Some(display.clone());
        display.set_next(None);
        if self.head.borrow().is_none() {
            // New list
            let tail = self.tail.borrow().clone();
            *self.head.borrow_mut() = tail;
        }
    }

    /// Java `getProcessResultDisplay(int, String)`
    /// (`ProcessResultDisplayFactoryInterface`).
    pub fn get_process_result_display(
        &self,
        display_id: i32,
        factory_id: &str,
    ) -> Option<ProcessResultDisplayHandle> {
        // This link list represents the buttons that interact with each other
        // in an interface - maybe fifty elements at most. There is no reason
        // to do a faster search for such a small number of elements.
        let head = self.head.borrow().clone();
        let Some(head) = head else {
            // DependencyID not found.
            return None;
        };
        let mut current = Some(head);
        while let Some(display) = current {
            if display.equals_id(display_id, factory_id) {
                return Some(display);
            }
            current = display.get_next();
        }
        None
    }

    //
    // Plugin functions

    /// Java protected `reset()`: just deletes the link list.  For plugin
    /// factories - if they have displays that need to be inserted in
    /// different places in the manager's list.
    pub fn reset(&self) {
        *self.head.borrow_mut() = None;
        *self.tail.borrow_mut() = None;
    }

    /// Java `insertAfter(AbstractProcessResultDisplayFactory,
    /// ProcessResultDisplay)`.  For plugin factories - allows the insertion of
    /// a link list from another factory.  This allows plugin buttons to
    /// interact with etomo button.  Assumes that the display ID has already
    /// been set.  Replaces the factory ID.
    ///
    /// `factory`: its list is inserted.  `start_insert`: element from the link
    /// list starting from `this.head`; the list is inserted after it.
    pub fn insert_after(
        &self,
        factory: Option<&AbstractProcessResultDisplayFactory>,
        start_insert: Option<&ProcessResultDisplayHandle>,
    ) {
        let Some(factory) = factory else {
            eprintln!("Error: factory is null");
            // Thread.dumpStack(): no Rust counterpart for a Java stack dump.
            return;
        };
        let Some(start_insert) = start_insert else {
            eprintln!("Error: startInsert is null");
            // Thread.dumpStack(): no Rust counterpart for a Java stack dump.
            return;
        };
        let factory_head = factory.head.borrow().clone();
        let Some(factory_head) = factory_head else {
            // Java prints the factory's `Object.toString()`
            // (`<class>@<hash>`); the class name and the factory ID stand in
            // for it.
            eprintln!(
                "Warning: Factory, AbstractProcessResultDisplayFactory({}), does not contain a list",
                factory.factory_id
            );
            // Thread.dumpStack(): no Rust counterpart for a Java stack dump.
            return;
        };
        // Insert display
        let end_insert = start_insert.get_next();
        // Overrides the previous factory ID.
        let mut current = factory_head;
        current.set_factory_id(self.factory_id.clone());
        start_insert.set_next(Some(current.clone()));
        // Move to the end of display's link list
        while let Some(next) = current.get_next() {
            current = next;
            current.set_factory_id(self.factory_id.clone());
        }
        // Attach startInsert.next to the end of display's link list.
        current.set_next(end_insert);
    }
}

impl ProcessResultDisplayFactoryInterface for AbstractProcessResultDisplayFactory {
    /// Java `@Override getProcessResultDisplay(int, String)`.
    fn get_process_result_display(
        &self,
        display_id: i32,
        factory_id: &str,
    ) -> Option<ProcessResultDisplayHandle> {
        AbstractProcessResultDisplayFactory::get_process_result_display(
            self, display_id, factory_id,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::r#type::file_key::FileKey;
    use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
    use crate::imod::etomo::r#type::process_result::ProcessResult;
    use std::cell::Cell;

    /// A display holding only the link and ID state the factory touches.
    #[derive(Default)]
    struct Link {
        next: RefCell<Option<ProcessResultDisplayHandle>>,
        id: Cell<i32>,
        factory: RefCell<String>,
    }

    impl crate::imod::etomo::r#type::process_result_display::ProcessResultDisplay for Link {
        fn as_any_rc(self: Rc<Self>) -> Rc<dyn std::any::Any> {
            self
        }
        fn get_output_image_file_key(&self) -> Option<FileKey> {
            None
        }
        fn set_next(&self, display: Option<ProcessResultDisplayHandle>) {
            *self.next.borrow_mut() = display;
        }
        fn set_use_global_dependency_list(&self, _input: bool) {}
        fn get_next(&self) -> Option<ProcessResultDisplayHandle> {
            self.next.borrow().clone()
        }
        fn set_debug(&self, _input: bool) {}
        fn dump_state(&self) {}
        fn get_original_state(&self) -> bool {
            false
        }
        fn set_original_state(&self, _original_state: bool) {}
        fn set_process_done(&self, _done: bool) {}
        fn set_screen_state(
            &self,
            _screen_state: &'static crate::imod::etomo::r#type::base_screen_state::BaseScreenState,
        ) {
        }
        fn msg_process_result(&self, _display_state: ProcessResult) {}
        fn msg_process_end_state(&self, _end_state: ProcessEndState) {}
        fn msg_process_starting(&self) {}
        fn msg_process_succeeded(&self) {}
        fn msg_process_failed(&self) {}
        fn msg_process_failed_to_start(&self) {}
        fn msg_secondary_process(&self) {}
        fn add_dependent_display(&self, _dependent_display: ProcessResultDisplayHandle) {}
        fn add_failure_display(&self, _failure_display: ProcessResultDisplayHandle) {}
        fn add_success_display(&self, _success_display: ProcessResultDisplayHandle) {}
        fn equals_id(&self, display_id: i32, factory_id: &str) -> bool {
            self.id.get() == display_id && *self.factory.borrow() == factory_id
        }
        fn set_id(&self, display_id: i32, factory_id: String) {
            self.id.set(display_id);
            *self.factory.borrow_mut() = factory_id;
        }
        fn get_display_id(&self) -> i32 {
            self.id.get()
        }
        fn get_factory_id(&self) -> Option<String> {
            Some(self.factory.borrow().clone())
        }
        fn set_factory_id(&self, factory_id: String) {
            *self.factory.borrow_mut() = factory_id;
        }
        fn get_button_state_key(&self) -> Option<String> {
            None
        }
    }

    #[test]
    fn list_links_ids_and_inserts_a_plugin_list() {
        let main = AbstractProcessResultDisplayFactory::new("main".to_owned());
        let first: ProcessResultDisplayHandle = Rc::new(Link::default());
        let last: ProcessResultDisplayHandle = Rc::new(Link::default());
        main.add_dependency(Some(&first), 0);
        main.add_dependency(Some(&last), 1);
        assert!(Rc::ptr_eq(
            &main.get_process_result_display(1, "main").unwrap(),
            &last
        ));
        let plugin = AbstractProcessResultDisplayFactory::new("plugin".to_owned());
        let inserted: ProcessResultDisplayHandle = Rc::new(Link::default());
        plugin.add_dependency(Some(&inserted), 7);
        main.insert_after(Some(&plugin), Some(&first));
        assert!(Rc::ptr_eq(&first.get_next().unwrap(), &inserted));
        assert!(Rc::ptr_eq(&inserted.get_next().unwrap(), &last));
        assert!(main.get_process_result_display(7, "main").is_some());
    }
}
