//! `IMOD/Etomo/src/etomo/ui/swing/AbstractProcessResultDisplayFactory.java`.
//!
//! Java stores `ProcessResultDisplay` object links directly.  Rust display widgets are
//! owned by their concrete factory fields, so the same singly linked list is represented
//! by its ordered stable display IDs.  The concrete factory supplies its typed display
//! fields when resolving an ID; no second set of display state is created here.
#![allow(dead_code)]

use super::multi_line_button::MultiLineButton;

/// Java `AbstractProcessResultDisplayFactory`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AbstractProcessResultDisplayFactory {
    /// Java final `factoryID`.
    pub factory_id: String,
    /// Java `head`, represented by its stable display ID.
    pub head: Option<i32>,
    /// Java `tail`, represented by its stable display ID.
    pub tail: Option<i32>,
    /// Rust ownership-safe representation of Java's singly linked display list.
    pub dependency_order: Vec<i32>,
}

impl AbstractProcessResultDisplayFactory {
    /// Java protected constructor `AbstractProcessResultDisplayFactory(String)`.
    pub fn new(factory_id: String) -> Self {
        Self {
            factory_id,
            head: None,
            tail: None,
            dependency_order: Vec::new(),
        }
    }

    /// Java protected synchronized `addDependency(ProcessResultDisplay, int)`.
    ///
    /// The caller owns each concrete display field.  `previous_display` is the Rust
    /// ownership bridge for Java's stored `tail` object reference, so this method can
    /// preserve the source `tail.setNext(display)` transition without duplicating widgets.
    pub fn add_dependency(
        &mut self,
        display: Option<&mut MultiLineButton>,
        display_id: i32,
        previous_display: Option<&mut MultiLineButton>,
    ) -> bool {
        let Some(display) = display else {
            eprintln!("Error: display is null");
            return false;
        };

        display.set_id(display_id, Some(&self.factory_id));
        if self.tail.is_some() {
            if let Some(previous_display) = previous_display {
                previous_display.set_next(true);
            }
        }
        self.tail = Some(display_id);
        display.set_next(false);
        if self.head.is_none() {
            self.head = self.tail;
        }
        self.dependency_order.push(display_id);
        true
    }

    /// Java `getProcessResultDisplay(int, String)`.
    pub fn get_process_result_display<'a>(
        &self,
        displays: impl IntoIterator<Item = &'a MultiLineButton>,
        display_id: i32,
        factory_id: Option<&str>,
    ) -> Option<&'a MultiLineButton> {
        if self.head.is_none() {
            return None;
        }
        let displays = displays.into_iter().collect::<Vec<_>>();
        for linked_display_id in &self.dependency_order {
            if *linked_display_id == display_id {
                if let Some(display) = displays
                    .iter()
                    .copied()
                    .find(|display| display.equals_id(*linked_display_id, factory_id))
                {
                    return Some(display);
                }
            }
        }
        None
    }

    /// Java protected `reset()`.
    pub fn reset(&mut self) {
        self.head = None;
        self.tail = None;
        self.dependency_order.clear();
    }

    /// Java `insertAfter(AbstractProcessResultDisplayFactory, ProcessResultDisplay)`.
    ///
    /// `end_insert` and `factory_displays` are Rust ownership bridges for Java's
    /// `startInsert.getNext()` and linked object traversal.  They let this method mutate
    /// the real widgets rather than a duplicate representation.
    pub fn insert_after(
        &mut self,
        factory: Option<&mut AbstractProcessResultDisplayFactory>,
        start_insert: Option<&mut MultiLineButton>,
        end_insert: Option<&mut MultiLineButton>,
        factory_displays: &mut [&mut MultiLineButton],
    ) -> bool {
        let Some(factory) = factory else {
            eprintln!("Error: factory is null");
            return false;
        };
        let Some(start_insert) = start_insert else {
            eprintln!("Error: startInsert is null");
            return false;
        };
        if factory.head.is_none() {
            eprintln!("Warning: Factory does not contain a list");
            return false;
        }
        let Some(insert_index) = self
            .dependency_order
            .iter()
            .position(|display_id| *display_id == start_insert.get_display_id())
        else {
            return false;
        };

        for display in factory_displays.iter_mut() {
            display.set_factory_id(Some(&self.factory_id));
        }
        start_insert.set_next(true);
        if let Some(display) = factory_displays.last_mut() {
            display.set_next(end_insert.is_some());
        }
        self.dependency_order.splice(
            insert_index + 1..insert_index + 1,
            factory.dependency_order.iter().copied(),
        );
        // Java intentionally does not update `tail` here.  Plugin factories reset their
        // temporary list after insertion, and later additions retain the original source
        // behavior even when insertion occurred after that list's tail.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_dependency_labels_and_resolves_the_linked_displays() {
        let mut factory = AbstractProcessResultDisplayFactory::new("factory".to_owned());
        let mut first = MultiLineButton::new_with_label(Some("first"));
        let mut second = MultiLineButton::new_with_label(Some("second"));
        assert!(factory.add_dependency(Some(&mut first), 4, None));
        assert!(factory.add_dependency(Some(&mut second), 9, Some(&mut first)));
        assert_eq!(factory.head, Some(4));
        assert_eq!(factory.tail, Some(9));
        assert_eq!(factory.dependency_order, vec![4, 9]);
        assert!(first.get_next());
        assert!(!second.get_next());
        assert_eq!(
            factory
                .get_process_result_display([&first, &second], 9, Some("factory"))
                .unwrap()
                .get_text(),
            Some("second")
        );
        assert!(
            factory
                .get_process_result_display([&first, &second], 9, Some("other"))
                .is_none()
        );
    }

    #[test]
    fn reset_and_insert_after_preserve_source_error_and_factory_id_paths() {
        let mut main = AbstractProcessResultDisplayFactory::new("main".to_owned());
        let mut first = MultiLineButton::new_with_label(Some("first"));
        let mut last = MultiLineButton::new_with_label(Some("last"));
        main.add_dependency(Some(&mut first), 1, None);
        main.add_dependency(Some(&mut last), 2, Some(&mut first));
        let mut plugin = AbstractProcessResultDisplayFactory::new("plugin".to_owned());
        let mut plugin_display = MultiLineButton::new_with_label(Some("plugin"));
        plugin.add_dependency(Some(&mut plugin_display), 7, None);
        assert!(main.insert_after(
            Some(&mut plugin),
            Some(&mut first),
            Some(&mut last),
            &mut [&mut plugin_display]
        ));
        assert_eq!(main.dependency_order, vec![1, 7, 2]);
        assert_eq!(main.tail, Some(2));
        assert_eq!(plugin_display.get_factory_id(), Some("main"));
        assert!(first.get_next());
        assert!(plugin_display.get_next());
        main.reset();
        assert_eq!(main.head, None);
        assert_eq!(main.tail, None);
        assert!(main.dependency_order.is_empty());
        assert!(!main.insert_after(None, Some(&mut first), None, &mut []));
    }
}
