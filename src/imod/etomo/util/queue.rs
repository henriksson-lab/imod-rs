//! `IMOD/Etomo/src/etomo/util/Queue.java`.
//!
//! A first-in first-out list with an optional size limit.  Java's `synchronized`
//! `pop`/`push` guard an object owned here by one holder, which `&mut self` covers.

use crate::imod::etomo::util::stack_trace::StackTrace;

/// Java `public final class Queue<E>`.
#[derive(Clone, Debug)]
pub struct Queue<E> {
    /// Java private final `list`.
    list: Vec<E>,
    /// Java private final `sizeLimit`.
    size_limit: i32,
}

impl<E> Default for Queue<E> {
    fn default() -> Queue<E> {
        Queue::new()
    }
}

impl<E> Queue<E> {
    /// Java `Queue()`.
    pub fn new() -> Queue<E> {
        Queue {
            list: Vec::new(),
            size_limit: -1,
        }
    }

    /// Java `Queue(int)`.  Limits the size of the queue; when the limit is reached the
    /// first element is popped before the next is pushed.
    pub fn new_size_limit(size_limit: i32) -> Queue<E> {
        if size_limit == 0 {
            eprintln!(
                "Warning:  attempt to limit etomo.util.Queue size to 0.  Ignoring size limit"
            );
            // `Thread.dumpStack()`.
            StackTrace::new_with_thread(None, None).print(Some("Stack trace"), false);
            return Queue {
                list: Vec::new(),
                size_limit: -1,
            };
        }
        Queue {
            list: Vec::new(),
            size_limit,
        }
    }

    /// Java `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Java `pop()`.  Removes and returns the first element, or null.
    pub fn pop(&mut self) -> Option<E> {
        if !self.list.is_empty() {
            return Some(self.list.remove(0));
        }
        None
    }

    /// Java `peek()`.  Returns the first element, or null.
    pub fn peek(&self) -> Option<&E> {
        if !self.list.is_empty() {
            return self.list.first();
        }
        None
    }

    /// Java `push(E)`.  A null element is not pushed.
    pub fn push(&mut self, element: Option<E>) {
        if let Some(element) = element {
            if self.size_limit > 0 && self.list.len() as i32 == self.size_limit {
                self.pop();
            }
            self.list.push(element);
        }
    }

    /// Java `clear()`.
    pub fn clear(&mut self) {
        self.list.clear();
    }
}

/// Java `toString()`: `ArrayList.toString()`.
impl<E: std::fmt::Display> std::fmt::Display for Queue<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (i, element) in self.list.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{element}")?;
        }
        f.write_str("]")
    }
}
