//! `IMOD/Etomo/src/etomo/util/StackTrace.java`.
//!
//! Prints stack traces with the non-etomo lines eliminated for readability.  Create an
//! instance to store and print one stack trace.  Use `print_all` to print a stack trace
//! for each thread that is running in the etomo code.
//!
//! **The captured frames are a JVM fact.**  The class's whole input is
//! `Thread.getStackTrace()`, an array of `java.lang.StackTraceElement` whose
//! `toString()` reads `etomo.storage.LogFile$Handle.<init>(LogFile.java:1730)`.  This
//! process has no Java frames, so the captured array is empty and `preprocess` is given
//! nothing: that is the same input a Java thread running no etomo code presents, and
//! every method below behaves exactly as the source does for it.  `containsString`,
//! `isStarting` and `isExiting` are therefore false, `createLineList` produces no lines,
//! and `print` emits only its title line.  The line-list processing itself is translated
//! in full and is correct for any line list.
//!
//! `Thread.getId()` is a JVM-assigned `long` counting from 1.  Rust's `ThreadId` is an
//! opaque value; its `as_u64` is unstable, so the id is rendered from the `Debug` form,
//! which is the same *kind* of fact but not the same number.
#![allow(dead_code)]

/// Java `StackTrace`.
#[derive(Debug)]
pub struct StackTrace {
    /// Java field `title`.
    title: Option<String>,
    /// Java field `threadId`.
    thread_id: Option<i64>,
    /// Java field `stackTrace`.  See the module header: this process contributes no
    /// `java.lang.StackTraceElement`s, so the captured array is empty.
    stack_trace: Option<Vec<String>>,
    /// Java field `lineList`.
    line_list: Option<Vec<String>>,
}

/// `java.lang.Thread.currentThread().getId()`.  See the module header for why this is
/// not the JVM's number.
fn java_lang_thread_get_id() -> i64 {
    let debug = format!("{:?}", std::thread::current().id());
    // `ThreadId(3)`
    let digits: String = debug.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse().unwrap_or(0)
}

impl StackTrace {
    /// Java `StackTrace(String, Thread)`.  A `thread` of `None` is the source's null
    /// argument, which captures the current thread.
    pub fn new_with_thread(title: Option<&str>, thread: Option<i64>) -> StackTrace {
        match thread {
            None => StackTrace {
                title: title.map(|title| title.to_string()),
                thread_id: Some(java_lang_thread_get_id()),
                stack_trace: Some(Vec::new()),
                line_list: None,
            },
            Some(thread) => StackTrace {
                title: title.map(|title| title.to_string()),
                thread_id: Some(thread),
                stack_trace: Some(Vec::new()),
                line_list: None,
            },
        }
    }

    /// Java `StackTrace(String)`.
    pub fn new_with_title(title: Option<&str>) -> StackTrace {
        StackTrace::new_with_thread(title, None)
    }

    /// Java `StackTrace()`.
    pub fn new() -> StackTrace {
        StackTrace::new_with_thread(None, None)
    }

    /// Java `getThreadId`.
    pub fn get_thread_id(&self) -> Option<i64> {
        self.thread_id
    }

    /// Java `print(boolean)`.
    pub fn print_all_lines(&mut self, add_all_lines: bool) {
        self.print(None, add_all_lines);
    }

    /// Java `print(String, boolean)`.  Print this instance's stack trace.
    pub fn print(&mut self, new_title: Option<&str>, add_all_lines: bool) {
        self.create_line_list(add_all_lines);
        StackTrace::print_static(
            new_title,
            self.title.as_deref(),
            self.thread_id,
            self.line_list.as_deref(),
        );
    }

    /// Java `createLineList`.
    fn create_line_list(&mut self, add_all_lines: bool) {
        // Convert stackTrace to lineList.
        if self.line_list.is_none() && self.stack_trace.is_some() {
            self.line_list =
                StackTrace::preprocess(self.stack_trace.as_deref(), true, add_all_lines);
            self.stack_trace = None;
        }
    }

    /// Java `isStarting`.
    pub fn is_starting(&mut self) -> bool {
        // etomo.EtomoDirector.main(EtomoDirector.java:131)
        self.contains_string(Some("etomo.EtomoDirector.main("))
            || self.contains_string(Some("etomo.EtomoDirector.setup("))
    }

    /// Java `isExiting`.
    pub fn is_exiting(&mut self) -> bool {
        // etomo.BaseManager.exitProgram(BaseManager.java:1642)
        // etomo.ApplicationManager.exitProgram(ApplicationManager.java:3541)
        self.contains_string(Some("Manager.exitProgram("))
            || self.contains_string(Some("etomo.ui.swing.UIHarness.exit("))
    }

    /// Java `containsString`.
    pub fn contains_string(&mut self, search_string: Option<&str>) -> bool {
        let search_string = match search_string {
            None => return false,
            Some(search_string) if search_string.is_empty() => return false,
            Some(search_string) => search_string,
        };
        self.create_line_list(false);
        let line_list = match self.line_list.as_ref() {
            None => return false,
            Some(line_list) => line_list,
        };
        for line in line_list.iter() {
            if line.contains(search_string) {
                return true;
            }
        }
        false
    }

    /// Java `printAll`.  Print a stack trace for each thread that is running in the
    /// etomo code.
    ///
    /// Deviation: `Thread.getAllStackTraces()` enumerates every live JVM thread.  There
    /// is no equivalent enumeration here, so only the banner the source always prints is
    /// produced; see the module header.
    pub fn print_all(_add_all_lines: bool) {
        eprintln!("\nAll Threads");
    }

    /// Java `preprocess`.  Gets the stack ready to be printed; returns a list of lines to
    /// be printed.
    fn preprocess(
        stack_trace_array: Option<&[String]>,
        add_any_stack: bool,
        add_all_lines: bool,
    ) -> Option<Vec<String>> {
        let stack_trace_array = match stack_trace_array {
            None => return None,
            Some(stack_trace_array) => stack_trace_array,
        };

        // Look for the first etomo line.
        let mut line_array: Vec<String> = Vec::new();
        let mut etomo_line_found = false;
        let mut index = 0usize;
        while index < stack_trace_array.len() {
            let line = &stack_trace_array[index];
            etomo_line_found = line.starts_with("etomo.");
            if etomo_line_found {
                break;
            }
            // Add the possibly useful non-etomo lines to lineArray.
            line_array.push(line.clone());
            index += 1;
        }

        // If no etomo line is found, then printing depends on addAnyStack.  If
        // addAnyStack is off, then there's nothing to print.  If addAnyStack is on and no
        // stack can be skipped, then there's no preprocessing to be done, because all the
        // lines have to be included.  Just return lineArray, which is full because no
        // etomo line was found.
        if !etomo_line_found {
            if !add_any_stack && !add_all_lines {
                return None;
            }
            return Some(line_array);
        }

        // The stack contains etomo lines, so all non etomo lines can be removed.
        if !add_all_lines {
            line_array.clear();
        }
        // Continue looking for and saving etomo lines.
        while index < stack_trace_array.len() {
            let line = &stack_trace_array[index];
            if add_all_lines || line.starts_with("etomo.") {
                line_array.push(line.clone());
            }
            index += 1;
        }
        Some(line_array)
    }

    /// Java `print(String, String, Long, List<String>)`.  Print a preprocessed stack
    /// trace.  If there are no lines to print, only the thread ID will be printed.
    fn print_static(
        new_title: Option<&str>,
        orig_title: Option<&str>,
        thread_id: Option<i64>,
        line_list: Option<&[String]>,
    ) {
        // Print title.
        if new_title.is_some() || orig_title.is_some() || thread_id.is_none() {
            let mut title = String::new();
            let mut adding_to_title = false;
            if let Some(new_title) = new_title {
                title.push_str(&(new_title.to_string() + " "));
                adding_to_title = true;
            }
            if let Some(orig_title) = orig_title {
                title.push_str(&(orig_title.to_string() + " "));
                adding_to_title = true;
            }
            if let Some(thread_id) = thread_id {
                if adding_to_title {
                    title.push('(');
                }
                title.push_str(&format!("Thread ID:{}", thread_id));
                if adding_to_title {
                    title.push(')');
                }
            }
            eprintln!("\n{}", title);
        }
        // Print line list.
        let line_list = match line_list {
            None => return,
            Some(line_list) if line_list.is_empty() => return,
            Some(line_list) => line_list,
        };
        for line in line_list.iter() {
            eprintln!("{}", line);
        }
    }
}
