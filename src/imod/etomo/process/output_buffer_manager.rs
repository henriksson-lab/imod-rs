//! `IMOD/Etomo/src/etomo/process/OutputBufferManager.java`.
//!
//! `SystemProgram` owns the actual pipe-reader threads in Rust.  This unit
//! owns the Java class's independent concern: retaining their lines, applying
//! an optional key phrase, and providing one primary plus multiple monitor
//! output lists without losing records between polls.

use std::collections::HashMap;

/// Source `OutputBufferManager`.  Listener keys are source `Object` identity
/// values represented by stable caller-supplied strings at this Rust boundary.
#[derive(Clone, Debug, Default)]
pub struct OutputBufferManager {
    output_list: Vec<String>,
    collect_output: bool,
    key_phrase: Option<String>,
    /// The source's first listener aliases `outputList`; later listeners own
    /// independent lists in `listenerList`.
    first_listener: Option<String>,
    listener_list: HashMap<String, Vec<String>>,
    process_done: bool,
    debug: bool,
    print_messages: bool,
}

impl OutputBufferManager {
    pub const MESSAGE_TOKEN: &'static str = "MESSAGE:";

    pub fn new() -> Self {
        Self {
            collect_output: true,
            ..Default::default()
        }
    }

    pub fn with_key_phrase(key_phrase: impl Into<String>) -> Self {
        Self {
            key_phrase: Some(key_phrase.into()),
            ..Self::new()
        }
    }

    /// A line delivered by the pipe reader in `run`.
    pub fn add(&mut self, line: impl Into<String>) {
        let line = line.into();
        if self
            .key_phrase
            .as_deref()
            .is_some_and(|key_phrase| !line.contains(key_phrase))
        {
            return;
        }
        if self.first_listener.is_none() && self.listener_list.is_empty() {
            self.output_list.push(line.clone());
        } else if self.first_listener.is_some() {
            self.output_list.push(line.clone());
        }
        for listener in self.listener_list.values_mut() {
            listener.push(line.clone());
        }
    }

    pub fn set_collect_output(&mut self, collect_output: bool) {
        self.collect_output = collect_output;
    }

    pub fn set_process_done(&mut self, process_done: bool) {
        self.process_done = process_done;
    }

    pub fn is_process_done(&self) -> bool {
        self.process_done
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    pub fn set_print_messages(&mut self, print_messages: bool) {
        self.print_messages = print_messages;
    }

    /// Java `printToErr`.  `HashedArray` prints its keys in insertion order;
    /// listener keys at this boundary are strings, so emit the primary key
    /// first and the remaining registered keys.  The map does not preserve
    /// Java's insertion order, but this is diagnostic-only output and no
    /// process routing depends on it.
    pub fn print_to_err(&self) {
        if let Some(listener) = &self.first_listener {
            eprintln!("{listener}");
        }
        let mut listeners = self.listener_list.keys().collect::<Vec<_>>();
        listeners.sort_unstable();
        for listener in listeners {
            eprintln!("{listener}");
        }
    }

    pub fn size(&self) -> usize {
        self.output_list.len()
    }

    pub fn get_line(&self, index: usize) -> Option<&str> {
        self.output_list.get(index).map(String::as_str)
    }

    /// Java `get()`: return the primary list and clear it only for the
    /// intermittent-process (`collectOutput == false`) path.
    pub fn get(&mut self) -> Vec<String> {
        let output = self.output_list.clone();
        if !self.collect_output {
            self.output_list.clear();
        }
        output
    }

    /// Java `get(listenerKey)`.  The first listener receives buffered output
    /// accumulated before registration; later listeners begin empty and see
    /// only subsequent lines.
    pub fn get_for_listener(&mut self, listener_key: impl Into<String>) -> Vec<String> {
        let listener_key = listener_key.into();
        if self.first_listener.is_none() && self.listener_list.is_empty() {
            self.first_listener = Some(listener_key.clone());
            let output = self.output_list.clone();
            if !self.collect_output {
                self.output_list.clear();
            }
            return output;
        }
        if self.first_listener.as_deref() == Some(listener_key.as_str()) {
            return self.get();
        }
        let listener = self.listener_list.entry(listener_key).or_default();
        let output = listener.clone();
        if !self.collect_output {
            listener.clear();
        }
        output
    }

    pub fn clear(&mut self) {
        self.output_list.clear();
        for listener in self.listener_list.values_mut() {
            listener.clear();
        }
    }

    /// Java `dropListener`: dropping the primary listener clears the shared
    /// output list before releasing its key.
    pub fn drop_listener(&mut self, listener_key: &str) {
        if self.first_listener.as_deref() == Some(listener_key) {
            self.output_list.clear();
            self.first_listener = None;
        }
        if let Some(mut listener) = self.listener_list.remove(listener_key) {
            listener.clear();
        }
    }

    pub fn listener_count(&self) -> usize {
        usize::from(self.first_listener.is_some()) + self.listener_list.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_listener_receives_pre_registration_output_and_later_one_starts_empty() {
        let mut manager = OutputBufferManager::new();
        manager.add("before");
        assert_eq!(manager.get_for_listener("one"), ["before"]);
        assert!(manager.get_for_listener("two").is_empty());
        manager.add("after");
        assert_eq!(manager.get_for_listener("one"), ["before", "after"]);
        assert_eq!(manager.get_for_listener("two"), ["after"]);
    }

    #[test]
    fn non_collecting_listener_clears_only_its_own_output() {
        let mut manager = OutputBufferManager::new();
        let _ = manager.get_for_listener("one");
        let _ = manager.get_for_listener("two");
        manager.set_collect_output(false);
        manager.add("line");
        assert_eq!(manager.get_for_listener("two"), ["line"]);
        assert!(manager.get_for_listener("two").is_empty());
        assert_eq!(manager.get_for_listener("one"), ["line"]);
    }

    #[test]
    fn key_phrase_and_drop_follow_source_lifecycle() {
        let mut manager = OutputBufferManager::with_key_phrase("keep");
        manager.add("discard");
        manager.add("keep this");
        assert_eq!(manager.get_for_listener("one"), ["keep this"]);
        manager.drop_listener("one");
        assert_eq!(manager.listener_count(), 0);
        assert!(manager.get().is_empty());
    }

    #[test]
    fn print_to_err_is_safe_for_empty_and_registered_listener_sets() {
        let mut manager = OutputBufferManager::new();
        manager.print_to_err();
        let _ = manager.get_for_listener("primary");
        let _ = manager.get_for_listener("secondary");
        manager.print_to_err();
    }
}
