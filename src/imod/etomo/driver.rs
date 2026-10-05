//! Click driver for the translated eTomo: the Rust counterpart of
//! `/big/henriksson/etomoproc/driver/src/EtomoDriver.java`.
//!
//! **Not a translated unit.**  Upstream IMOD has no such program; this is
//! Rust-only test plumbing, like `comrun.rs`, reachable as the command
//! `etomodriver SCRIPT OUTDIR [etomo args]`.  It runs the same click scripts
//! as the Java driver against the translated component tree (`jdk.rs`), so a
//! run of the Rust eTomo can be compared step by step with the Java reference
//! run.
//!
//! It starts `EtomoDirector` with its setup queued on the event dispatch
//! thread ([`EtomoDirector::main_headless_edt`]), then executes the script on
//! the calling thread.  Every action runs on the EDT (`invoke_later`, then a
//! settle of three EDT round trips, as the Java driver does), components are
//! found by name among the showing components of the windows the harness owns
//! (the main frame with its menu bar, the sub frame, the current manager's log
//! window), and every `JOptionPane` popup is answered by the `UIHarness` popup
//! hook with the Java driver's rules: a matching `popup.<text>=<button>` rule,
//! else the pane's initial value, else OK/Yes, else the first button.  The
//! hook runs synchronously on the EDT where the Java pane is shown, which
//! stands in for the Java driver's popup-watcher thread; the watcher thread
//! here only logs newly opened windows.
//!
//! Script language and outputs (`driver.log`, `popups.log`, `exec.log`,
//! `dump-<label>.txt`, `$DRIVER_SNAPSHOT`) are those of `EtomoDriver.java`;
//! see its head comment.  One addition (also in the Java driver copy under
//! `/big/henriksson/gui2/parallel/driver/`): `chooser=<path>` makes the next
//! `FileChooser` that is shown select `<path>` and approve (one-shot); a chooser
//! shown with nothing pending is cancelled.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director::{self, EtomoDirector};
use crate::imod::etomo::jdk::{ComponentKind, JComponent};
use crate::imod::etomo::ui::swing::etomo_frame;
use crate::imod::etomo::ui::swing::ui_harness::{self, PopupAnswer, PopupRequest};
use crate::imod::etomo::util::event_queue;
use crate::imod::libcfshr::b3dutil;

/// The driver's open logs and state shared with the popup hook and the window
/// watcher.
struct DriverState {
    log: Option<File>,
    popup_log: Option<File>,
    out_dir: PathBuf,
    /// `popup.<key>=<button>` rules, in first-insertion order (Java
    /// `LinkedHashMap.put` keeps a replaced key's position).
    popup_rules: Vec<(String, String)>,
    handled_popups: Vec<String>,
    popup_count: usize,
    /// `chooser=<path>`: the file the next shown `FileChooser` selects and
    /// approves (one-shot); with none pending a chooser is cancelled.
    chooser_file: Option<String>,
}

static STATE: Mutex<DriverState> = Mutex::new(DriverState {
    log: None,
    popup_log: None,
    out_dir: PathBuf::new(),
    popup_rules: Vec::new(),
    handled_popups: Vec::new(),
    popup_count: 0,
    chooser_file: None,
});

/// `HH:mm:ss.SSS` in local time (Java `SimpleDateFormat`).
fn timestamp() -> String {
    chrono::Local::now().format("%H:%M:%S%.3f").to_string()
}

/// Java `log(String)`: driver.log and standard output.
fn log(text: &str) {
    let line = format!("{} {}", timestamp(), text);
    let mut state = STATE.lock().unwrap();
    if let Some(file) = state.log.as_mut() {
        let _ = writeln!(file, "{}", line);
        let _ = file.flush();
    }
    println!("[driver] {}", line);
}

fn popup_log(text: &str) {
    let mut state = STATE.lock().unwrap();
    if let Some(file) = state.popup_log.as_mut() {
        let _ = writeln!(file, "{}", text);
        let _ = file.flush();
    }
}

fn out_dir() -> PathBuf {
    STATE.lock().unwrap().out_dir.clone()
}

/// The `etomodriver` command: `etomodriver SCRIPT OUTDIR [etomo args]`.
/// Returns the exit status.
pub fn etomodriver(args: &[String]) -> i32 {
    if args.len() < 2 {
        eprintln!("usage: etomodriver <script> <outdir> [etomo args]");
        return 2;
    }
    let script = args[0].clone();
    let out = PathBuf::from(&args[1]);
    let _ = std::fs::create_dir_all(&out);
    {
        let mut state = STATE.lock().unwrap();
        let open = |name: &str| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(out.join(name))
                .ok()
        };
        state.log = open("driver.log");
        state.popup_log = open("popups.log");
        state.out_dir = out.clone();
    }
    let etomo_args: Vec<String> = args[2..].to_vec();
    log(&format!(
        "start: script={} etomo args={}",
        script,
        etomo_args.join(" ")
    ));
    // The popup hook lives on the event dispatch thread's UIHarness.
    event_queue::invoke_and_wait(|| {
        ui_harness::with(|harness| harness.set_popup_hook(Some(Box::new(handle_popup))));
        // `chooser=<path>`: answer the next file chooser with that file.
        crate::imod::etomo::ui::swing::file_chooser::set_dialog_responder(Some(std::rc::Rc::new(
            |chooser: &crate::imod::etomo::ui::swing::file_chooser::FileChooser, _parent| {
                let file = STATE.lock().unwrap().chooser_file.take();
                match file {
                    Some(file) => {
                        popup_log(&format!("{} CHOOSER -> {}", timestamp(), file));
                        log(&format!("chooser: {file}"));
                        chooser.set_selected_file(Some(std::path::Path::new(&file)));
                        chooser.approve_selection();
                    }
                    None => {
                        log("chooser: none pending, cancelled");
                        chooser.cancel_selection();
                    }
                }
            },
        )));
    });
    std::thread::Builder::new()
        .name("popup-watcher".to_owned())
        .spawn(window_watcher)
        .expect("starting the window watcher");
    EtomoDirector::main_headless_edt(&etomo_args);
    let status = match run_script(&script) {
        Ok(()) => 0,
        Err(error) => {
            log(&format!("SCRIPT FAILED: {}", error));
            dump("failure");
            1
        }
    };
    log(&format!("done, exit {}", status));
    status
}

// ---------------------------------------------------------------- script

/// Java `String.trim()`: strips code units `<= ' '`.
fn java_trim(text: &str) -> &str {
    text.trim_matches(|c: char| c <= ' ')
}

fn run_script(script: &str) -> Result<(), String> {
    let file = File::open(script).map_err(|e| format!("java.io.FileNotFoundException: {e}"))?;
    let variable = regex::Regex::new(r"\$\{([A-Za-z_0-9]+)\}").unwrap();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_no = index + 1;
        let line = line.map_err(|e| e.to_string())?;
        let line = java_trim(&line);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // ${VAR} is replaced by the environment variable VAR
        let line = variable
            .replace_all(line, |captures: &regex::Captures| {
                std::env::var(&captures[1]).unwrap_or_default()
            })
            .into_owned();
        log(&format!("> {}   (line {})", line, line_no));
        let (key, value) = match line.find('=') {
            Some(eq) => (
                java_trim(&line[..eq]).to_owned(),
                Some(java_trim(&line[eq + 1..]).to_owned()),
            ),
            None => (line.clone(), None),
        };
        execute(&key, value.as_deref())?;
    }
    Ok(())
}

fn parse_long(value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("java.lang.NumberFormatException: For input string: \"{value}\""))
}

/// `name[,secs]` with a default number of seconds.
fn name_and_secs(value: &str, default: u64) -> Result<(String, u64), String> {
    match value.rfind(',') {
        Some(c) if c > 0 => Ok((value[..c].to_owned(), parse_long(&value[c + 1..])?)),
        _ => Ok((value.to_owned(), default)),
    }
}

fn execute(key: &str, value: Option<&str>) -> Result<(), String> {
    let value_str = value.unwrap_or("");
    if let Some(rule) = key.strip_prefix("popup.") {
        let mut state = STATE.lock().unwrap();
        match state.popup_rules.iter_mut().find(|(k, _)| k == rule) {
            Some(entry) => entry.1 = value_str.to_owned(),
            None => state
                .popup_rules
                .push((rule.to_owned(), value_str.to_owned())),
        }
        return Ok(());
    }
    match key {
        "chooser" => {
            STATE.lock().unwrap().chooser_file = Some(value_str.to_owned());
            return Ok(());
        }
        "sleep" => {
            std::thread::sleep(Duration::from_millis(parse_long(value_str)?));
            return Ok(());
        }
        "log" => return Ok(()),
        "dump" => {
            dump(value_str);
            return Ok(());
        }
        "exit" => {
            log("exit requested");
            let status = if value_str.is_empty() {
                0
            } else {
                value_str.parse::<i32>().map_err(|_| {
                    format!("java.lang.NumberFormatException: For input string: \"{value_str}\"")
                })?
            };
            b3dutil::exit(status);
        }
        "snapshot" => {
            settle();
            let snap = std::env::var("DRIVER_SNAPSHOT").unwrap_or_else(|_| "null".to_owned());
            let run_dir = std::env::current_dir()
                .map(|d| d.to_string_lossy().into_owned())
                .unwrap_or_default();
            run_shell(&format!(
                "\"{}\" \"{}\" \"{}\" \"{}\"",
                snap,
                run_dir,
                value_str,
                out_dir().display()
            ));
            return Ok(());
        }
        "exec" => {
            run_shell(value_str);
            return Ok(());
        }
        "menu" => {
            // `menu=<text>[#N]`: click the N-th (1-based, default 1) menu item with
            // that text in the windows' menu bars, in window order.
            let (text, nth) = match value_str.rsplit_once('#') {
                Some((text, n)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
                    (text.to_owned(), n.parse::<usize>().unwrap_or(1).max(1))
                }
                _ => (value_str.to_owned(), 1),
            };
            /// The menu items with `text`, in window order.
            fn menu_items(text: &str) -> Vec<Rc<JComponent>> {
                fn walk(node: &Rc<JComponent>, text: &str, out: &mut Vec<Rc<JComponent>>) {
                    if matches!(
                        node.kind(),
                        ComponentKind::MenuItem | ComponentKind::CheckBoxMenuItem
                    ) && node.get_text() == text
                    {
                        out.push(node.clone());
                    }
                    for child in node.get_components() {
                        walk(&child, text, out);
                    }
                }
                let mut items = Vec::new();
                for window in windows() {
                    for root in &window.roots {
                        walk(root, text, &mut items);
                    }
                }
                items
            }
            let text_edt = text.clone();
            let found = event_queue::invoke_and_wait(move || menu_items(&text_edt).len() >= nth);
            if found {
                // Clicked from a later event, as the Java driver's `invokeLater`
                // (a popup the click opens must not block the driver).
                event_queue::invoke_later(move || {
                    if let Some(item) = menu_items(&text).into_iter().nth(nth - 1) {
                        item.do_click();
                    }
                });
            }
            if !found {
                return Err(format!(
                    "java.lang.RuntimeException: no menu item {value_str}"
                ));
            }
            settle();
            return Ok(());
        }
        "rclick" => {
            // `rclick=<name>[.N]`: a right mouse press on the N-th showing component
            // with that name, delivered as Swing does (`jdk::dispatch_mouse_pressed`).
            let (name, index) = match value_str.rsplit_once('.') {
                Some((name, n)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
                    (name.to_owned(), n.parse::<usize>().unwrap_or(0))
                }
                _ => (value_str.to_owned(), 0),
            };
            find_with_retry(&name, index, 30)?;
            event_queue::invoke_later(move || match find_all(&name).get(index) {
                Some(component) => crate::imod::etomo::jdk::dispatch_mouse_pressed(
                    component,
                    &crate::imod::etomo::jdk::MouseEvent {
                        button: 3,
                        popup_trigger: true,
                        x: 0,
                        y: 0,
                    },
                ),
                None => log(&format!("ACTION FAILED on {}: component vanished", name)),
            });
            settle();
            return Ok(());
        }
        "popupdump" => {
            // `popupdump=<label>`: log the showing popup menu's items.
            let text = event_queue::invoke_and_wait(|| {
                let Some((menu, _, _)) = crate::imod::etomo::jdk::showing_popup_menu() else {
                    return "POPUP none".to_owned();
                };
                let mut text = format!("POPUP '{}'", menu.get_text());
                for item in menu.get_components() {
                    match item.kind() {
                        ComponentKind::Separator => text.push_str("\n  ---"),
                        ComponentKind::MenuItem => text.push_str(&format!(
                            "\n  {} enabled={} name={}",
                            item.get_text(),
                            item.is_enabled(),
                            item.get_name().unwrap_or_else(|| "null".to_owned())
                        )),
                        _ => {}
                    }
                }
                text
            });
            log(&format!("{value_str} {text}"));
            return Ok(());
        }
        "popupclose" => {
            // `popupclose`: Escape closes a showing popup menu.
            event_queue::invoke_and_wait(|| {
                if let Some((menu, _, _)) = crate::imod::etomo::jdk::showing_popup_menu() {
                    menu.set_visible(false);
                }
            });
            settle();
            return Ok(());
        }
        "pmn" => {
            // `pmn=<text>`: choose the item with that text in the showing popup menu
            // (`BasicMenuItemUI`: the menu path is cleared, then the item clicked).
            let wanted = value_str.to_owned();
            let found = event_queue::invoke_and_wait(move || {
                crate::imod::etomo::jdk::showing_popup_menu().is_some_and(|(menu, _, _)| {
                    menu.get_components().into_iter().any(|item| {
                        item.kind() == ComponentKind::MenuItem && item.get_text() == wanted
                    })
                })
            });
            if !found {
                return Err(format!(
                    "java.lang.RuntimeException: no popup menu item {value_str}"
                ));
            };
            let wanted = value_str.to_owned();
            event_queue::invoke_later(move || {
                if let Some((menu, _, _)) = crate::imod::etomo::jdk::showing_popup_menu()
                    && let Some(item) = menu.get_components().into_iter().find(|item| {
                        item.kind() == ComponentKind::MenuItem && item.get_text() == wanted
                    })
                {
                    menu.set_visible(false);
                    item.do_click();
                }
            });
            settle();
            return Ok(());
        }
        "wait.name" => {
            let (name, secs) = name_and_secs(value_str, 60)?;
            find_with_retry(&name, 0, secs)?;
            return Ok(());
        }
        "wait.process" => {
            let (mut secs, mut start_secs) = (3600, 30);
            if !value_str.is_empty() {
                let parts: Vec<&str> = value_str.split(',').collect();
                secs = parse_long(java_trim(parts[0]))?;
                if parts.len() > 1 {
                    start_secs = parse_long(java_trim(parts[1]))?;
                }
            }
            return wait_process(secs, start_secs);
        }
        "wait.popup" => {
            let (name, secs) = name_and_secs(value_str, 60)?;
            let end = Instant::now() + Duration::from_secs(secs);
            while Instant::now() < end {
                {
                    let mut state = STATE.lock().unwrap();
                    if let Some(at) = state.handled_popups.iter().position(|p| p.contains(&name)) {
                        state.handled_popups.remove(at);
                        return Ok(());
                    }
                }
                std::thread::sleep(Duration::from_millis(250));
            }
            return Err(format!(
                "java.lang.RuntimeException: popup {} did not appear",
                name
            ));
        }
        _ => {}
    }
    // component commands
    let Some(dot) = key.find('.') else {
        return Err(format!("java.lang.RuntimeException: unknown command {key}"));
    };
    let kind = key[..dot].to_owned();
    let mut name = key.to_owned();
    let mut index = 0usize;
    let last = key.rfind('.').unwrap();
    if last > dot {
        let suffix = &key[last + 1..];
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            name = key[..last].to_owned();
            index = suffix.parse().unwrap_or(0);
        }
    }
    let find_index = if kind == "tb" { 0 } else { index };
    find_with_retry(&name, find_index, 30)?;
    let value = value.map(str::to_owned);
    event_queue::invoke_later(move || {
        // The component is looked up again on the EDT: components are EDT
        // objects and cannot cross threads.  Java hands the found Component
        // to the EDT runnable; the search is the same one.
        let found = find_all(&name);
        match found.get(find_index) {
            Some(component) => act(&kind, component, value.as_deref(), index),
            None => log(&format!("ACTION FAILED on {}: component vanished", name)),
        }
    });
    // Let the action's own event cascade run.
    settle();
    Ok(())
}

/// Java `getText().replaceAll("<[^>]*>", "").trim()`.
fn strip_html(text: &str) -> String {
    let tags = regex::Regex::new("<[^>]*>").unwrap();
    java_trim(&tags.replace_all(text, "")).to_owned()
}

fn act(kind: &str, component: &Rc<JComponent>, value: Option<&str>, index: usize) {
    let name = component.get_name().unwrap_or_else(|| "null".to_owned());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match kind {
        "bn" | "rb" | "ctb" | "cttb" => component.do_click(),
        "cb" => {
            let want = matches!(
                value,
                None | Some("") | Some("on") | Some("1") | Some("true")
            );
            if component.is_selected() != want {
                component.do_click();
            }
        }
        "tf" => {
            component.set_text(value.unwrap_or(""));
            if component.kind() == ComponentKind::TextField {
                // JTextField.postActionEvent
                component.fire_action_performed();
            }
        }
        "sp" => {
            let value = value.unwrap_or("");
            let mut target = value.parse::<f64>().ok();
            if let Some(model) = component.get_spinner_model() {
                if value == "up" {
                    target = Some(model.value + model.step_size)
                        .filter(|v| model.maximum.is_none_or(|max| *v <= max));
                } else if value == "down" {
                    target = Some(model.value - model.step_size)
                        .filter(|v| model.minimum.is_none_or(|min| *v >= min));
                }
            }
            match target {
                Some(target) => component.set_spinner_value(target),
                None => log(&format!(
                    "ACTION FAILED on {}: java.lang.IllegalArgumentException: illegal value",
                    name
                )),
            }
        }
        "cbb" => {
            for i in 0..component.get_item_count() {
                if let Some(item) = component.get_item_at(i)
                    && Some(item.as_str()) == value
                {
                    component.set_selected_index(i as i32);
                    return;
                }
            }
            if component.is_editable() {
                // rows2 addition: an editable combo box takes the text
                // (`JComboBox.setSelectedItem`, as its editor commits it).
                component.set_selected_item(value);
                return;
            }
            log(&format!(
                "WARNING: combo item {} not found in {}",
                value.unwrap_or("null"),
                name
            ));
        }
        "tb" => component.set_selected_tab(index as i32),
        "mb" => {
            let mut buttons = vec![component.clone()];
            collect(component, &mut buttons);
            for button in buttons {
                if matches!(
                    button.kind(),
                    ComponentKind::Button
                        | ComponentKind::ToggleButton
                        | ComponentKind::CheckBox
                        | ComponentKind::RadioButton
                        | ComponentKind::MenuItem
                        | ComponentKind::CheckBoxMenuItem
                        | ComponentKind::Menu
                ) && Some(strip_html(&button.get_text()).as_str()) == value
                    && button.is_showing()
                {
                    button.do_click();
                    return;
                }
            }
            log(&format!(
                "WARNING: no button '{}' in {}",
                value.unwrap_or("null"),
                name
            ));
        }
        _ => log(&format!("WARNING: unknown component type {}", kind)),
    }));
    if let Err(panic) = result {
        let message = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_else(|| "panic".to_owned());
        log(&format!("ACTION FAILED on {}: {}", name, message));
    }
}

// ---------------------------------------------------------------- windows

/// One top-level window the harness owns.
struct Window {
    class: &'static str,
    title: String,
    showing: bool,
    /// The window's roots in AWT tree order: content pane, then menu bar.
    roots: Vec<Rc<JComponent>>,
}

/// Java `Window.getWindows()`: the main frame, the sub frame and the current
/// manager's log window.  EDT only.
fn windows() -> Vec<Window> {
    let mut out = Vec::new();
    if let Some(main_frame) = ui_harness::with(|harness| harness.get_main_frame()) {
        let mut roots = vec![main_frame.get_content_pane()];
        roots.extend(main_frame.get_j_menu_bar());
        out.push(Window {
            class: "etomo.ui.swing.MainFrame",
            title: main_frame.get_title(),
            showing: main_frame.is_visible(),
            roots,
        });
    }
    if let Some(sub_frame) = etomo_frame::sub_frame() {
        let mut roots = vec![sub_frame.get_content_pane()];
        roots.extend(sub_frame.get_j_menu_bar());
        out.push(Window {
            class: "etomo.ui.swing.SubFrame",
            title: sub_frame.get_title(),
            showing: sub_frame.is_visible(),
            roots,
        });
    }
    // The showing `JDialog`s (a modal startup dialog), as `Window.getWindows()`
    // lists them.
    for dialog in crate::imod::etomo::jdk::showing_dialogs() {
        out.push(Window {
            class: "javax.swing.JDialog",
            title: dialog.get_title(),
            showing: dialog.is_visible(),
            roots: vec![dialog.get_content_pane()],
        });
    }
    if let Some(manager) = etomo_director::INSTANCE.get_current_manager_for_driver()
        && let Some(log_window) = manager.get_log_window()
    {
        out.push(Window {
            class: "etomo.ui.swing.LogWindow$LogFrame",
            title: log_window.get_frame_title(),
            showing: log_window.is_frame_visible(),
            roots: std::iter::once(log_window.get_frame_content_pane())
                .chain(log_window.get_frame_j_menu_bar())
                .collect(),
        });
    }
    // The log-file frames `ContextPopup` opens, in the order they were created.
    for window in crate::imod::etomo::ui::swing::tabbed_text_window::get_windows() {
        out.push(Window {
            class: "etomo.ui.swing.TabbedTextWindow",
            title: window.get_title().unwrap_or_default(),
            showing: window.is_visible(),
            roots: vec![window.get_content_pane()],
        });
    }
    for window in crate::imod::etomo::ui::swing::text_page_window::get_windows() {
        out.push(Window {
            class: "etomo.ui.swing.TextPageWindow",
            title: window.get_title(),
            showing: window.is_visible(),
            roots: vec![window.get_content_pane()],
        });
    }
    // The `ManagerFrame`s (the directive editor), in the order they were created, as
    // `Window.getWindows()` lists them.
    for (_, manager_frame) in ui_harness::with(|harness| harness.get_manager_frames()) {
        let mut roots = vec![manager_frame.get_content_pane()];
        roots.extend(manager_frame.get_j_menu_bar());
        out.push(Window {
            class: "etomo.ui.swing.ManagerFrame",
            title: manager_frame.get_title(),
            showing: manager_frame.is_visible(),
            roots,
        });
    }
    out
}

/// Java `collect(Container, List)`: every descendant, depth first.
/// A `JMenu`'s items live in its popup menu, which is not a child of the
/// menu in AWT, so they are not collected.
fn collect(component: &Rc<JComponent>, out: &mut Vec<Rc<JComponent>>) {
    if component.kind() == ComponentKind::Menu {
        return;
    }
    for child in component
        .get_components()
        .into_iter()
        .chain(component.get_tab_title_components())
    {
        out.push(child.clone());
        collect(&child, out);
    }
}

fn all_components(window: &Window) -> Vec<Rc<JComponent>> {
    let mut all = Vec::new();
    for root in &window.roots {
        all.push(root.clone());
        collect(root, &mut all);
    }
    all
}

/// Java `findAll(String)`: the showing components (in all showing windows)
/// with this name, or this prefix for a name ending in `*`.  EDT only.
fn find_all(name: &str) -> Vec<Rc<JComponent>> {
    let mut showing = Vec::new();
    for window in windows() {
        if !window.showing {
            continue;
        }
        for component in all_components(&window) {
            if let Some(n) = component.get_name()
                && component.is_showing()
                && (n == name || (name.ends_with('*') && n.starts_with(&name[..name.len() - 1])))
            {
                showing.push(component);
            }
        }
    }
    showing
}

fn find_with_retry(name: &str, index: usize, secs: u64) -> Result<(), String> {
    let end = Instant::now() + Duration::from_secs(secs);
    loop {
        let wanted = name.to_owned();
        let found = event_queue::invoke_and_wait(move || {
            let list = find_all(&wanted);
            list.get(index).is_some_and(|c| c.is_enabled())
        });
        if found {
            return Ok(());
        }
        if Instant::now() > end {
            dump(&format!("notfound-{}", name.replace('/', "_")));
            return Err(format!(
                "java.lang.RuntimeException: component {}[{}] not found (or not enabled) after {}s",
                name, index, secs
            ));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

// ---------------------------------------------------------------- processes

/// Java `waitProcess(long, long)`: waits for `lb.busy` or `bn.kill-process`
/// to come on, then until both are off for 3 s, and logs the progress bar.
fn wait_process(secs: u64, start_secs: u64) -> Result<(), String> {
    let start = Instant::now();
    let mut started = false;
    while start.elapsed() < Duration::from_secs(start_secs) {
        if busy() {
            started = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    if !started {
        log("WARNING: no process started (busy signal never came on)");
    }
    let mut quiet_since: Option<Instant> = None;
    loop {
        if start.elapsed() > Duration::from_secs(secs) {
            return Err(format!(
                "java.lang.RuntimeException: process did not finish in {}s",
                secs
            ));
        }
        if busy() {
            quiet_since = None;
        } else if let Some(since) = quiet_since {
            if since.elapsed() > Duration::from_millis(3000) {
                break;
            }
        } else {
            quiet_since = Some(Instant::now());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    let (label, bar) = event_queue::invoke_and_wait(|| {
        let mut label = None;
        let mut bar = None;
        for c in find_all("the-progress-bar-label") {
            label = Some(c.get_text());
        }
        for c in find_all("the-progress-bar") {
            bar = c.get_string();
        }
        (label, bar)
    });
    log(&format!(
        "process finished after {}s: label='{}' progress='{}'",
        start.elapsed().as_millis() as f64 / 1000.0,
        label.as_deref().unwrap_or("null"),
        bar.as_deref().unwrap_or("null")
    ));
    Ok(())
}

fn busy() -> bool {
    event_queue::invoke_and_wait(|| {
        find_all("lb.busy").iter().any(|c| c.is_enabled())
            || find_all("bn.kill-process").iter().any(|c| c.is_enabled())
    })
}

// ---------------------------------------------------------------- popups

/// Java `PopupWatcher.handle(Dialog, JOptionPane)`, as the `UIHarness` popup
/// hook.  Runs on the EDT where the pane is shown.
fn handle_popup(request: &PopupRequest) -> PopupAnswer {
    // The Java driver's watcher finds a popup on its next 300 ms scan, so a
    // popup stays up for a while there; answering at once let a process
    // thread blocked on the dialog go on before eTomo's 100 ms monitor poll
    // (`startComScriptMonitor`) had mapped its monitor, a race the Java
    // timing never meets.
    std::thread::sleep(Duration::from_millis(300));
    let mut text = String::new();
    for line in &request.message {
        text.push_str(line);
        text.push('\n');
    }
    let name = request.name.clone();
    let title = request.title.clone();
    let buttons = &request.options;
    let mut choice: Option<String> = None;
    {
        let state = STATE.lock().unwrap();
        for (k, v) in &state.popup_rules {
            if name.as_deref().is_some_and(|n| n.contains(k.as_str()))
                || title.as_deref().is_some_and(|t| t.contains(k.as_str()))
                || text.contains(k.as_str())
            {
                choice = Some(v.clone());
                break;
            }
        }
    }
    if choice.is_none() {
        choice = request.initial_value.clone();
    }
    let mut press: Option<usize> = None;
    for (i, b) in buttons.iter().enumerate() {
        if choice.as_deref() == Some(b.as_str()) {
            press = Some(i);
        }
    }
    if press.is_none() {
        for pref in ["OK", "Ok", "Yes", "YES"] {
            for (i, b) in buttons.iter().enumerate() {
                if press.is_none() && pref == b {
                    press = Some(i);
                }
            }
        }
    }
    // The root pane's default button: JOptionPane makes the initial value's
    // button the default, which the initial-value rule above already chose.
    if press.is_none() && !buttons.is_empty() {
        press = Some(0);
    }
    let labels: String = buttons.iter().map(|b| format!("[{}]", b)).collect();
    let pressed = press.map_or("(none)".to_owned(), |i| buttons[i].clone());
    let count = {
        let mut state = STATE.lock().unwrap();
        state.popup_count += 1;
        state.popup_count
    };
    popup_log(&format!(
        "=== popup #{} at {}\nname: {}\ntitle: {}\nbuttons: {}\npressed: {}\ntext:\n{}",
        count,
        timestamp(),
        name.as_deref().unwrap_or("null"),
        title.as_deref().unwrap_or("null"),
        labels,
        pressed,
        text
    ));
    log(&format!(
        "popup #{} '{}' ({}) -> {}: {}",
        count,
        title.as_deref().unwrap_or("null"),
        name.as_deref().unwrap_or("null"),
        pressed,
        text.replace('\n', " ")
    ));
    STATE.lock().unwrap().handled_popups.push(format!(
        "{}|{}",
        name.as_deref().unwrap_or("null"),
        title.as_deref().unwrap_or("null")
    ));
    match press {
        Some(i) => PopupAnswer::Selected(i),
        None => PopupAnswer::Closed,
    }
}

/// The window half of Java's `PopupWatcher`: every 300 ms, logs each newly
/// shown window.  (Popups are answered by [`handle_popup`].)
fn window_watcher() {
    let mut seen: HashSet<String> = HashSet::new();
    let mut frame_count = 0usize;
    loop {
        std::thread::sleep(Duration::from_millis(300));
        let shown: Vec<(String, String)> = event_queue::invoke_and_wait(|| {
            windows()
                .into_iter()
                .filter(|w| w.showing)
                .map(|w| (w.class.to_owned(), w.title))
                .collect()
        });
        for (class, title) in shown {
            if seen.insert(class.clone()) {
                popup_log(&format!(
                    "{} WINDOW opened: class={} title='{}' name=frame{}",
                    timestamp(),
                    class,
                    title,
                    frame_count
                ));
                frame_count += 1;
                log(&format!("window opened: {} '{}'", class, title));
            }
        }
    }
}

// ---------------------------------------------------------------- utils

/// Java `settle()`: waits until the EDT has drained what was queued so far.
fn settle() {
    for _ in 0..3 {
        std::thread::sleep(Duration::from_millis(300));
        event_queue::invoke_and_wait(|| ());
    }
}

/// Java `runShell(String)`: `bash -c`, output appended to `exec.log`.
fn run_shell(command: &str) {
    log(&format!("exec: {}", command));
    let exec_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_dir().join("exec.log"));
    let status = match exec_log {
        Ok(file) => {
            let err = file.try_clone();
            let mut child = std::process::Command::new("bash");
            child.arg("-c").arg(command).stdout(file);
            if let Ok(err) = err {
                child.stderr(err);
            }
            child.status().map(|s| s.code().unwrap_or(-1))
        }
        Err(e) => Err(e),
    };
    match status {
        Ok(rc) => log(&format!("exec rc={}", rc)),
        Err(e) => log(&format!("exec failed: {}", e)),
    }
}

/// The Java class simple name a dump prints for a component kind.
fn simple_name(kind: ComponentKind) -> &'static str {
    match kind {
        ComponentKind::Panel => "JPanel",
        ComponentKind::Label => "JLabel",
        ComponentKind::Button => "JButton",
        ComponentKind::ToggleButton => "JToggleButton",
        ComponentKind::CheckBox => "JCheckBox",
        ComponentKind::RadioButton => "JRadioButton",
        ComponentKind::TextField => "JTextField",
        ComponentKind::TextArea => "JTextArea",
        ComponentKind::Spinner => "JSpinner",
        ComponentKind::ComboBox => "JComboBox",
        ComponentKind::TabbedPane => "JTabbedPane",
        ComponentKind::ScrollPane => "JScrollPane",
        ComponentKind::MenuItem => "JMenuItem",
        ComponentKind::CheckBoxMenuItem => "JCheckBoxMenuItem",
        ComponentKind::Menu => "Menu",
        ComponentKind::PopupMenu => "JPopupMenu",
        ComponentKind::ProgressBar => "JProgressBar",
        ComponentKind::Separator => "JPopupMenu$Separator",
        ComponentKind::Other => "Component",
    }
}

/// Java `dump(String)`: every named component of every window.
fn dump(label: &str) {
    let text = event_queue::invoke_and_wait(|| {
        let mut sb = String::new();
        for window in windows() {
            sb.push_str(&format!(
                "WINDOW {} '{}' showing={}\n",
                window.class, window.title, window.showing
            ));
            for c in all_components(&window) {
                let Some(name) = c.get_name() else {
                    continue;
                };
                if name.starts_with("null.") {
                    continue;
                }
                sb.push_str(&format!(
                    "  {}\t{}\tshowing={}\tenabled={}",
                    name,
                    simple_name(c.kind()),
                    window.showing && c.is_showing(),
                    c.is_enabled()
                ));
                match c.kind() {
                    ComponentKind::Button
                    | ComponentKind::ToggleButton
                    | ComponentKind::CheckBox
                    | ComponentKind::RadioButton
                    | ComponentKind::MenuItem
                    | ComponentKind::CheckBoxMenuItem
                    | ComponentKind::Menu => sb.push_str(&format!(
                        "\ttext='{}' selected={}",
                        c.get_text(),
                        c.is_selected()
                    )),
                    ComponentKind::TextField | ComponentKind::TextArea | ComponentKind::Label => {
                        sb.push_str(&format!("\ttext='{}'", c.get_text()))
                    }
                    ComponentKind::Spinner => {
                        let value = c.get_spinner_value();
                        let integer = c.get_spinner_model().is_some_and(|m| m.integer);
                        if integer {
                            sb.push_str(&format!("\tvalue={}", value as i64));
                        } else {
                            sb.push_str(&format!("\tvalue={:?}", value));
                        }
                    }
                    ComponentKind::ComboBox => sb.push_str(&format!(
                        "\tselected={}",
                        c.get_selected_item().as_deref().unwrap_or("null")
                    )),
                    ComponentKind::ProgressBar => sb.push_str(&format!(
                        "\tstring='{}'",
                        c.get_string().as_deref().unwrap_or("null")
                    )),
                    ComponentKind::TabbedPane => {
                        sb.push_str("\ttabs=");
                        for i in 0..c.get_tab_count() {
                            sb.push_str(&format!("[{}]", c.get_title_at(i).unwrap_or_default()));
                        }
                        sb.push_str(&format!(" selected={}", c.get_selected_tab()));
                    }
                    _ => {}
                }
                sb.push('\n');
            }
        }
        sb
    });
    let path: PathBuf = out_dir().join(format!("dump-{}.txt", label));
    match std::fs::write(Path::new(&path), text) {
        Ok(()) => log(&format!("dumped components to dump-{}.txt", label)),
        Err(e) => log(&format!("dump failed: {}", e)),
    }
}
