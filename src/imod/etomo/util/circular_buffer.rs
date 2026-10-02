//! `IMOD/Etomo/src/etomo/util/CircularBuffer.java`.
//!
//! The Java class holds a raw `Vector` of `Object`s; it is generic over the element
//! type here, with a Java null element as `None`.
//!
//! `CircularBuffer(0)` sets `iHead` to -1, and `put` then writes element 0 of an empty
//! `Vector` (`ArrayIndexOutOfBoundsException`); `get` reads element -1.  A negative
//! size throws `IllegalArgumentException` from `new Vector`.  UserConfiguration builds
//! one from the user's `NMRUFiles` property, so those sizes are reachable from a
//! settings file.  Fixed in translation: a negative size is 0, `put` into an empty
//! buffer stores nothing and `get` from it returns null.
#![allow(dead_code)]

/// Java `CircularBuffer`.
#[derive(Clone, Debug)]
pub struct CircularBuffer<T> {
    /// Java package-private field `buffer`.
    pub(crate) buffer: Vec<Option<T>>,
    /// Java package-private field `iHead`.
    pub(crate) i_head: i32,
}

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

impl<T: Clone + PartialEq> CircularBuffer<T> {
    /// Java `CircularBuffer(int)`.
    pub fn new(n_elements: i32) -> CircularBuffer<T> {
        let n_elements = n_elements.max(0);
        // Need to also set the size, even though the space is allocated for
        // nElements, the buffer thinks it's length is zero.
        CircularBuffer {
            buffer: vec![None; n_elements as usize],
            i_head: n_elements - 1,
        }
    }

    /// Java `size()`.  Return the number of elements available in the circular buffer.
    pub fn size(&self) -> i32 {
        self.buffer.len() as i32
    }

    /// Java `put(Object)`.  Place an object into the next position on circular buffer.
    pub fn put(&mut self, obj: Option<T>) {
        if self.buffer.is_empty() {
            return;
        }
        if self.i_head == self.buffer.len() as i32 - 1 {
            self.i_head = 0;
        } else {
            self.i_head += 1;
        }
        self.buffer[self.i_head as usize] = obj;
    }

    /// Java `get()`.  Get an object from current position in the buffer and move the
    /// position down one element.
    pub fn get(&mut self) -> Option<T> {
        if self.buffer.is_empty() {
            return None;
        }
        let i_current = self.i_head;
        if self.i_head == 0 {
            self.i_head = self.buffer.len() as i32 - 1;
        } else {
            self.i_head -= 1;
        }
        self.buffer[i_current as usize].clone()
    }

    /// Java `search(Object)`.  Search the list to see if the specified object is on it.
    /// (`bufferObj.equals(null)` is false, so a null `obj` is never found.)
    pub fn search(&self, obj: Option<&T>) -> i32 {
        for i in 0..self.buffer.len() {
            if let Some(buffer_obj) = &self.buffer[i]
                && Some(buffer_obj) == obj
            {
                return i as i32;
            }
        }
        -1
    }
}

impl<T: std::fmt::Display> CircularBuffer<T> {
    /// Java package-private `paramString()`.
    pub(crate) fn param_string(&self) -> String {
        let mut string_buffer = format!("\niHead={}", self.i_head);
        let element = |index: usize| match &self.buffer[index] {
            None => "null".to_string(),
            Some(value) => value.to_string(),
        };
        let mut index = self.i_head;
        while index >= 0 && (index as usize) < self.buffer.len() {
            string_buffer.push_str(&format!(",\nbuffer[{}]={}", index, element(index as usize)));
            index += 1;
        }
        index = 0;
        while index < self.i_head && (index as usize) < self.buffer.len() {
            string_buffer.push_str(&format!(",\nbuffer[{}]={}", index, element(index as usize)));
            index += 1;
        }
        string_buffer
    }
}

/// Java `toString()`.
impl<T: std::fmt::Display> std::fmt::Display for CircularBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "etomo.util.CircularBuffer[{}]", self.param_string())
    }
}
