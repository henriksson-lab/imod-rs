//! `IMOD/Etomo/src/etomo/storage/NameValuePairList.java`.
//!
//! An ordered list of the name/value pairs of an autodoc, used by the batchruntomo
//! interface to merge and subtract directive files.  Event dispatch thread objects;
//! the pairs are never modified, so lists share them through `Rc`.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::autodoc::autodoc::Autodoc;
use super::autodoc::autodoc_tokenizer;
use super::autodoc::read_only_statement::ReadOnlyStatement;
use super::autodoc::read_only_statement_list::ReadOnlyStatementList;
use super::autodoc::statement;
use super::directive_def::DirectiveDef;
use super::log_file::{Handle, LogFileError};
use crate::imod::etomo::logic::comparison_strategy::ComparisonStrategy;

/// Java private static final `DEBUG_PRINT = null`.
const DEBUG_PRINT: Option<&[&str]> = None;
/// Java private static final `DEBUG_PRINT_MUST_BE_NULL = null`.
const DEBUG_PRINT_MUST_BE_NULL: Option<&[bool]> = None;

/// Java private static `DebugPrintDumpedStack`, initially false.
static DEBUG_PRINT_DUMPED_STACK: AtomicBool = AtomicBool::new(false);

/// Java `public final class NameValuePairList`.
pub struct NameValuePairList {
    /// Java private final `list`.
    list: Vec<Rc<Pair>>,
    /// Java private final `map`.
    map: HashMap<String, Rc<Pair>>,
}

impl NameValuePairList {
    /// Java `NameValuePairList(Autodoc)`.
    ///
    /// # Safety
    /// `autodoc` must point to a live `Autodoc` (the autodoc registry's pointers are).
    pub unsafe fn new_autodoc(autodoc: *mut Autodoc) -> NameValuePairList {
        let mut instance = NameValuePairList {
            list: Vec::new(),
            map: HashMap::new(),
        };
        let autodoc: &Autodoc = unsafe { &*autodoc };
        let mut loc = ReadOnlyStatementList::get_statement_location(autodoc);
        // load global autodoc statements
        loop {
            let statement = unsafe { ReadOnlyStatementList::next_statement(autodoc, loc.as_mut()) };
            if statement.is_null() {
                break;
            }
            let statement = unsafe { &*statement };
            if statement.get_type() != statement::Type::NameValuePair {
                continue;
            }
            let pair = Rc::new(Pair::new(
                statement.get_left_side().unwrap_or_default(),
                statement.get_right_side(),
            ));
            // Clobber entry with a preexisting name.  Upstream bug fixed in translation
            // (NameValuePairList.java:58-61): the source calls `list.remove(pair)` on
            // the *new* pair, which is not in the list, so the earlier entry stayed and
            // the name was listed twice; the earlier entry is removed here.
            if let Some(existing) = instance.map.get(&pair.name).cloned() {
                instance.list.retain(|listed| !Rc::ptr_eq(listed, &existing));
            }
            // Add entry
            instance.list.push(pair.clone());
            instance.map.insert(pair.name.clone(), pair);
        }
        instance.debug_print("A", None);
        instance
    }

    /// Java `NameValuePairList(NameValuePairList)`, the copy constructor.
    pub fn new_copy(input: Option<&NameValuePairList>) -> NameValuePairList {
        let mut instance = NameValuePairList {
            list: Vec::new(),
            map: HashMap::new(),
        };
        let Some(input) = input else {
            return instance;
        };
        for pair in &input.list {
            // Pairs can be shared because they are never modified.
            instance.list.push(pair.clone());
            instance.map.insert(pair.name.clone(), pair.clone());
        }
        instance
    }

    /// Java private `debugPrint(String, NameValuePairList)`.  Prints selected pairs.
    fn debug_print(&self, label: &str, name_value_pair_list: Option<&NameValuePairList>) {
        let Some(debug_print) = DEBUG_PRINT else {
            return;
        };
        if debug_print.is_empty() {
            return;
        }
        let mut pair: Vec<Option<Rc<Pair>>> = Vec::new();
        let mut print = false;
        for name in debug_print {
            let found = match name_value_pair_list {
                None => self.get(name),
                Some(list) => list.get(name),
            };
            if found.is_some() {
                print = true;
            }
            pair.push(found);
        }
        // If all values are null, then don't print
        if !print {
            return;
        }
        print!("{}:", label);
        let mut dump_stack = false;
        for (i, name) in debug_print.iter().enumerate() {
            print!(
                "{}{}",
                match &pair[i] {
                    Some(pair) => pair.to_string(),
                    None => format!("{}:null", name),
                },
                if i < debug_print.len() - 1 { "," } else { "" }
            );
            if let Some(must_be_null) = DEBUG_PRINT_MUST_BE_NULL
                && !must_be_null.is_empty()
                && !dump_stack
                && must_be_null[i]
                && pair[i].is_some()
            {
                dump_stack = true;
            }
        }
        println!();
        // Attempt to dump stack when there's an error.
        if dump_stack {
            match name_value_pair_list {
                None => self.dump_stack(),
                Some(list) => list.dump_stack(),
            }
        }
    }

    /// Java private `dumpStack()`.  Dump stack once for this instance.
    fn dump_stack(&self) {
        if !DEBUG_PRINT_DUMPED_STACK.load(Ordering::Relaxed) {
            // Thread.dumpStack(): no Rust counterpart for the Java thread's stack.
            DEBUG_PRINT_DUMPED_STACK.store(true, Ordering::Relaxed);
        }
    }

    /// Java `merge(NameValuePairList)`.  Add pairs to this instance from merge, where
    /// the name is not in this instance.
    pub fn merge(&mut self, merge: Option<&NameValuePairList>) {
        let Some(merge) = merge else {
            return;
        };
        for merge_pair in &merge.list {
            if !self.map.contains_key(&merge_pair.name) {
                // pair doesn't exist in instance - add it
                self.list.push(merge_pair.clone());
                self.map.insert(merge_pair.name.clone(), merge_pair.clone());
            }
        }
    }

    /// Java `subtract(NameValuePairList, ComparisonStrategy)`.  Remove sub pairs from
    /// this instance, when the name and the value are the same.
    pub fn subtract_list(
        &mut self,
        sub: Option<&NameValuePairList>,
        strategy: Option<&dyn ComparisonStrategy>,
    ) {
        let Some(sub) = sub else {
            return;
        };
        for sub_pair in &sub.list {
            let pair = self.map.get(&sub_pair.name).cloned();
            if let Some(pair) = pair
                && pair.equals_value(sub_pair, strategy)
            {
                // identical - remove it
                self.list.retain(|listed| !Rc::ptr_eq(listed, &pair));
                self.map.remove(&pair.name);
            }
        }
    }

    /// Java `subtract(Set<DirectiveDef>)`.  Removes pairs from this instance where the
    /// left side matches an element in directiveDefSet.
    pub fn subtract(&mut self, directive_def_set: Option<&HashSet<DirectiveDef>>) {
        self.debug_print("J", None);
        let Some(directive_def_set) = directive_def_set else {
            return;
        };
        // Go through the list backwards and strip off any pair that matches.
        let len = self.list.len();
        for i in (0..len).rev() {
            let pair = self.list[i].clone();
            if DirectiveDef::get_instance(Some(&pair.name))
                .is_some_and(|directive_def| directive_def_set.contains(&directive_def))
            {
                self.list.remove(i);
                self.map.remove(&pair.name);
            }
        }
        self.debug_print("K", None);
    }

    /// Java `write(LogFile.Handle)`.  Write list to a file.
    pub fn write(&self, log_file: Option<&Arc<Handle>>) {
        let Some(log_file) = log_file else {
            return;
        };
        self.debug_print("L", None);
        let id = match log_file.open_writer() {
            Ok(id) => Some(id),
            Err(LogFileError::Lock(_)) => None,
            Err(e) => {
                eprintln!("{}", e);
                None
            }
        };
        if let Some(id) = &id {
            let result: Result<(), LogFileError> = (|| {
                for pair in &self.list {
                    match &pair.value {
                        None => log_file.write(
                            Some(&format!(
                                "{} {} ",
                                pair.name,
                                autodoc_tokenizer::DEFAULT_DELIMITER
                            )),
                            id,
                        )?,
                        Some(value) => log_file.write(
                            Some(&format!(
                                "{} {} {}",
                                pair.name,
                                autodoc_tokenizer::DEFAULT_DELIMITER,
                                value
                            )),
                            id,
                        )?,
                    }
                    log_file.new_line(id)?;
                }
                Ok(())
            })();
            match result {
                Ok(()) | Err(LogFileError::Lock(_)) => {}
                Err(e) => eprintln!("{}", e),
            }
        }
        log_file.close_id(id.as_deref());
    }

    /// Java package-private `size()`.
    pub fn size(&self) -> usize {
        self.list.len()
    }

    /// Java package-private `get(int)`.
    pub fn get_index(&self, index: usize) -> Rc<Pair> {
        self.list[index].clone()
    }

    /// Java `get(String)`.
    pub fn get(&self, name: &str) -> Option<Rc<Pair>> {
        self.map.get(name).cloned()
    }
}

/// Java `toString()`.
impl std::fmt::Display for NameValuePairList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pairs: Vec<String> = self.list.iter().map(|pair| pair.to_string()).collect();
        write!(f, "[list:[{}]]", pairs.join(", "))
    }
}

/// Java package-private `static final class Pair`.  Name/value pair.
pub struct Pair {
    /// Java package-private final `name` (required).
    pub name: String,
    /// Java package-private final `value` (may be null).
    pub value: Option<String>,
}

impl Pair {
    /// Java private `Pair(String, String)`.
    fn new(name: String, value: Option<String>) -> Pair {
        Pair { name, value }
    }

    /// Java private `equalsValue(Pair, ComparisonStrategy)`.  True if values are equal,
    /// or are both null.  `strategy` (optional) changes the comparison algorithm.
    fn equals_value(&self, pair: &Pair, strategy: Option<&dyn ComparisonStrategy>) -> bool {
        if let Some(strategy) = strategy
            && let Ok(equal) = strategy.equals(self.value.as_deref(), pair.value.as_deref())
        {
            return equal;
        }
        (self.value.is_none() && pair.value.is_none())
            || (self.value.is_some() && self.value == pair.value)
    }

    /// Java package-private `getName()`.
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Java package-private `getValue()`.
    pub fn get_value(&self) -> Option<&str> {
        self.value.as_deref()
    }
}

/// Java `Pair.toString()`.
impl std::fmt::Display for Pair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[name:{},value:{}]",
            self.name,
            self.value.as_deref().unwrap_or("null")
        )
    }
}
