//! Rust-only plumbing: **not a translated unit** (like `comrun.rs`).
//!
//! Connects the Slint main window to the translated Swing component tree
//! (`etomo/jdk.rs`).  The Slint widgets of `gui/etomo-ui-common` report every
//! click, toggle and edit through the `EtomoBridge` global as `(kind, label)`
//! -- the Swing widget class prefix (`bn`, `tf`, `cb`, `rb`, `sp`, `mb`) and the
//! label the Java passes to the widget -- and ask for the value to show the same
//! way.  eTomo names every widget from exactly that pair
//! (`Utilities.convertLabelToName`, the jfcUnit names its uitests click), so a
//! pair resolves to the one showing component of that name under the main
//! frame, and the event is delivered to it on the event dispatch thread (the
//! Slint event loop thread, see `event_queue::install_slint_edt`) as a Swing
//! click or edit would be.  Every listener, manager action and process launch
//! after that is the translated Java.
//!
//! The process-panel buttons (`ProcessControlPanel`, kind `pcp`) are found by
//! their action command, the `DialogType` label they were built with; their
//! state is read back from the button text, as `ColoredStateText` writes it.
//!
//! A timer re-reads the tree every [`REFRESH_MS`] milliseconds: it bumps the
//! global's `generation` (so every bound value is queried again), sets which
//! manager panel and dialog the window shows, and gives every shown
//! `TextPageWindow` (a log file window) its own Slint window.
#![cfg(feature = "gui")]

use std::cell::RefCell;
use std::rc::Rc;

use etomo_ui_main_window::{EtomoBridge, MainFrameWindow, OptionPaneWindow, TextPageWindowWindow};
use slint::ComponentHandle;

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ComponentKind, JComponent, named_components};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::ui::swing::etomo_menu::ToolType;
use crate::imod::etomo::ui::swing::text_page_window::{self, TextPageWindow};
use crate::imod::etomo::ui::swing::ui_harness::{self, PopupAnswer, PopupRequest};
use crate::imod::etomo::util::utilities;

/// How often the window re-reads the component tree.
pub const REFRESH_MS: u64 = 150;

/// The showing components under the main frame, with their names.
fn showing_components() -> Vec<(String, Rc<JComponent>)> {
    let Some(root) = ui_harness::with(|harness| harness.get_main_frame_root()) else {
        return Vec::new();
    };
    named_components(&root)
        .into_iter()
        .filter(|(_, component)| component.is_showing())
        .collect()
}

/// The uitest name eTomo gives a `kind` widget labelled `label`.  Kinds
/// "raw" (and the progress-bar kinds "pb", "pbf") name a component by its exact
/// Swing name (`ProgressPanel.NAME`).
fn names_for(kind: &str, label: &str) -> Vec<String> {
    if matches!(kind, "raw" | "pb" | "pbf") {
        return vec![label.to_owned()];
    }
    let mut names = Vec::new();
    for unlimited in [false, true] {
        if let Some(name) = utilities::convert_label_to_name(Some(label), unlimited) {
            let name = format!("{kind}.{name}");
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

/// The showing component a Slint `(kind, label)` pair stands for.  A label
/// may list alternatives separated by `|`.  A label ending in `#N` names the N-th (1-based, in component-tree order) of several
/// showing components with the same name -- the A and B axis copies of a field
/// that the Java builds with the same label.
fn find(kind: &str, label: &str) -> Option<Rc<JComponent>> {
    let found = find_unlogged(kind, label);
    // `ETOMO_BRIDGE_DEBUG=1`: report every pair that matches nothing, with the
    // showing names, to find a Slint widget whose label differs from the Java.
    if found.is_none() && std::env::var_os("ETOMO_BRIDGE_DEBUG").is_some() {
        eprintln!("slint_bridge: no showing component for ({kind}, {label:?})");
    }
    found
}

fn find_unlogged(kind: &str, label: &str) -> Option<Rc<JComponent>> {
    // "Advanced|Basic": a button whose label (and so its name) changes; the
    // first alternative that matches.
    if label.contains('|') {
        return label
            .split('|')
            .find_map(|alternative| find_unlogged(kind, alternative));
    }
    if let Some((base, occurrence)) = label.rsplit_once('#')
        && let Ok(occurrence) = occurrence.parse::<usize>()
        && occurrence >= 1
    {
        return find_all(kind, base).into_iter().nth(occurrence - 1);
    }
    find_all(kind, label).into_iter().next()
}

/// Every showing component a Slint `(kind, label)` pair can stand for, in
/// component-tree order.
fn find_all(kind: &str, label: &str) -> Vec<Rc<JComponent>> {
    let components = showing_components();
    if kind == "pcp" {
        // `ProcessControlPanel`: the toggle button whose action command is the
        // `DialogType` label.
        return components
            .into_iter()
            .map(|(_, component)| component)
            .filter(|component| {
                component.kind() == ComponentKind::ToggleButton
                    && component.get_action_command().as_deref() == Some(label)
            })
            .collect();
    }
    if kind == "fbn" {
        // The file-chooser button beside the text field labelled `label`
        // (`FileTextField`, `FileTextField2`, `ButtonControlTextEfield`): the
        // first button in the field's panel.
        return find_all("tf", label)
            .into_iter()
            .filter_map(|field| {
                field
                    .get_parent()?
                    .get_components()
                    .into_iter()
                    .find(|component| component.kind() == ComponentKind::Button)
            })
            .collect();
    }
    if matches!(kind, "mn" | "cbmn") {
        // A menu item, by its text, anywhere under the menu bar (the menus
        // are not showing until opened).
        let wanted = if kind == "mn" {
            [ComponentKind::MenuItem, ComponentKind::Menu]
        } else {
            [
                ComponentKind::CheckBoxMenuItem,
                ComponentKind::CheckBoxMenuItem,
            ]
        };
        let Some(menu_bar) = ui_harness::with(|harness| harness.get_main_frame_menu_bar()) else {
            return Vec::new();
        };
        let mut found = Vec::new();
        fn walk(
            node: &Rc<JComponent>,
            label: &str,
            wanted: &[ComponentKind],
            found: &mut Vec<Rc<JComponent>>,
        ) {
            if wanted.contains(&node.kind()) && node.get_text() == label {
                found.push(node.clone());
            }
            for child in node.get_components() {
                walk(&child, label, wanted, found);
            }
        }
        walk(&menu_bar, label, &wanted, &mut found);
        return found;
    }
    let names = names_for(kind, label);
    let exact: Vec<_> = components
        .iter()
        .filter(|(n, _)| names.contains(n))
        .map(|(_, component)| component.clone())
        .collect();
    if !exact.is_empty() {
        return exact;
    }
    // A name that carries a state suffix (`bn.<label>-not-started`).
    components
        .iter()
        .filter(|(n, _)| names.iter().any(|name| n.starts_with(&format!("{name}-"))))
        .map(|(_, component)| component.clone())
        .collect()
}

/// `ColoredStateText`'s index from a process button's text.
fn process_state(component: &JComponent) -> i32 {
    let text = component.get_text();
    if text.contains("In Progress") {
        1
    } else if text.contains("Complete") {
        2
    } else {
        0
    }
}

/// The Slint name of a reconstruction dialog type (the `dialog` property).
fn dialog_name(dialog_type: DialogType) -> &'static str {
    match dialog_type {
        DialogType::PreProcessing => "PRE_PROCESSING",
        DialogType::CoarseAlignment => "COARSE_ALIGNMENT",
        DialogType::FiducialModel => "FIDUCIAL_MODEL",
        DialogType::FineAlignment => "FINE_ALIGNMENT",
        DialogType::TomogramPositioning => "TOMOGRAM_POSITIONING",
        DialogType::FinalAlignedStack => "FINAL_ALIGNED_STACK",
        DialogType::TomogramGeneration => "TOMOGRAM_GENERATION",
        DialogType::TomogramCombination => "TOMOGRAM_COMBINATION",
        DialogType::PostProcessing => "POST_PROCESSING",
        DialogType::CleanUp => "CLEAN_UP",
        _ => "",
    }
}

/// Sets which manager panel, axis type and dialog the window shows, from the
/// director's current manager.
fn update_view(window: &MainFrameWindow) {
    let manager: Option<&'static dyn BaseManager> = etomo_director::INSTANCE.get_current_manager();
    let Some(manager) = manager else {
        window.set_view("".into());
        return;
    };
    match manager.get_interface_type() {
        Some(InterfaceType::FrontPage) => {
            window.set_view("front-page".into());
        }
        Some(InterfaceType::Recon) => {
            window.set_view("recon".into());
            let main_panel = manager.get_main_panel();
            if let Some(main_panel) = main_panel.as_ref() {
                window.set_status_bar_text(main_panel.main_panel().get_status_bar_text().into());
            }
            let showing_setup = main_panel
                .as_ref()
                .is_some_and(|main_panel| main_panel.main_panel().is_showing_setup());
            window.set_showing_setup(showing_setup);
            let axis_type = manager
                .get_base_meta_data()
                .map(|meta_data| meta_data.base().get_axis_type())
                .unwrap_or(AxisType::NotSet);
            window.set_axis_type(
                match axis_type {
                    AxisType::DualAxis => "DUAL_AXIS",
                    _ => "SINGLE_AXIS",
                }
                .into(),
            );
            let dialog_type = manager.get_current_dialog_type(Some(AxisID::First));
            window.set_dialog(dialog_type.map(dialog_name).unwrap_or("").into());
        }
        Some(InterfaceType::Tools) => {
            window.set_view("tools".into());
            if let Some(main_panel) = manager.get_main_panel() {
                window.set_status_bar_text(main_panel.main_panel().get_status_bar_text().into());
            }
            window.set_tool_type(match manager.get_tool_type() {
                Some(ToolType::GpuTiltTest) => 1,
                Some(ToolType::AlignFrames) => 2,
                _ => 0,
            });
        }
        _ => {
            window.set_view("".into());
        }
    }
}

thread_local! {
    /// The Slint window drawn for each shown `TextPageWindow`.
    static PAGE_WINDOWS: RefCell<Vec<(Rc<TextPageWindow>, TextPageWindowWindow)>> =
        const { RefCell::new(Vec::new()) };
}

/// Opens a Slint window for every newly shown `TextPageWindow`, and closes the
/// ones whose `TextPageWindow` was hidden or disposed.  Closing the Slint window
/// is the window system's `WINDOW_CLOSING` event (`processWindowEvent`).
fn update_page_windows() {
    let shown = text_page_window::get_windows();
    PAGE_WINDOWS.with(|page_windows| {
        let mut page_windows = page_windows.borrow_mut();
        page_windows.retain(|(page, slint_window)| {
            let keep = page.is_visible() && shown.iter().any(|s| Rc::ptr_eq(s, page));
            if !keep {
                let _ = slint_window.hide();
            }
            keep
        });
        for page in shown {
            if !page.is_visible() || page_windows.iter().any(|(p, _)| Rc::ptr_eq(p, &page)) {
                continue;
            }
            let Ok(slint_window) = TextPageWindowWindow::new() else {
                continue;
            };
            slint_window.set_title_text(page.get_title().into());
            slint_window.set_page_text(page.get_text().into());
            let closing = Rc::downgrade(&page);
            slint_window.window().on_close_requested(move || {
                if let Some(page) = closing.upgrade() {
                    page.process_window_closing();
                    // DO_NOTHING_ON_CLOSE keeps the frame up.
                    if page.is_visible() {
                        return slint::CloseRequestResponse::KeepWindowShown;
                    }
                }
                slint::CloseRequestResponse::HideWindow
            });
            let _ = slint_window.show();
            page_windows.push((page, slint_window));
        }
    });
}

/// Wires `window`'s `EtomoBridge` to the component tree.  Called by
/// `UIHarness.createMainFrame` on the event dispatch thread.
pub fn install(window: &MainFrameWindow) -> slint::Timer {
    let bridge = window.global::<EtomoBridge>();
    bridge.on_activated(|kind, label| {
        if let Some(component) = find(&kind, &label) {
            if component.is_enabled() {
                component.do_click();
            }
        }
    });
    bridge.on_toggled(|kind, label, value| {
        let Some(component) = find(&kind, &label) else {
            return;
        };
        if !component.is_enabled() {
            return;
        }
        if kind.as_str() == "sp" {
            // The spinner's arrow buttons: one step up (true) or down.
            if let Some(model) = component.get_spinner_model() {
                let mut next = component.get_spinner_value()
                    + if value {
                        model.step_size
                    } else {
                        -model.step_size
                    };
                if let Some(maximum) = model.maximum {
                    next = next.min(maximum);
                }
                if let Some(minimum) = model.minimum {
                    next = next.max(minimum);
                }
                component.set_spinner_value(next);
            }
        } else if component.is_selected() != value {
            component.do_click();
        }
    });
    bridge.on_edited(|kind, label, text| {
        if std::env::var_os("ETOMO_BRIDGE_DEBUG").is_some() {
            eprintln!("slint_bridge: edited ({kind}, {label:?}) {text:?}");
        }
        let Some(component) = find(&kind, &label) else {
            return;
        };
        if kind.as_str() == "sp" {
            if let Ok(value) = text.trim().parse::<f64>() {
                component.set_spinner_value(value);
            }
        } else {
            component.set_text(&text);
        }
    });
    bridge.on_focus_changed(|kind, label, gained| {
        if let Some(component) = find(&kind, &label) {
            component.fire_focus_changed(gained);
        }
    });
    bridge.on_items(|kind, label, _generation| {
        let items: Vec<slint::SharedString> = match find(&kind, &label) {
            None => Vec::new(),
            Some(component) => (0..component.get_item_count())
                .map(|index| component.get_item_at(index).unwrap_or_default().into())
                .collect(),
        };
        slint::ModelRc::new(slint::VecModel::from(items))
    });
    bridge.on_selected(|kind, label, index| {
        if let Some(component) = find(&kind, &label)
            && component.is_enabled()
        {
            component.set_selected_index(index);
        }
    });
    bridge.on_text_value(
        |kind, label, default, _generation| match find(&kind, &label) {
            None => default,
            Some(component) => match kind.as_str() {
                "pcp" => process_state(&component).to_string().into(),
                "cbb" => component.get_selected_item().unwrap_or_default().into(),
                // A progress bar's painted string.
                "pb" => component.get_string().unwrap_or_default().into(),
                // A progress bar's filled fraction, -1 when indeterminate.
                "pbf" => {
                    if component.is_indeterminate() {
                        "-1".into()
                    } else {
                        let range = (component.get_maximum() - component.get_minimum()).max(1);
                        format!(
                            "{}",
                            (component.get_value() - component.get_minimum()) as f32 / range as f32
                        )
                        .into()
                    }
                }
                "sp" => {
                    let value = component.get_spinner_value();
                    if component
                        .get_spinner_model()
                        .is_some_and(|model| model.integer)
                    {
                        format!("{}", value as i64).into()
                    } else {
                        format!("{value}").into()
                    }
                }
                _ => component.get_text().into(),
            },
        },
    );
    bridge.on_bool_value(
        |kind, label, default, _generation| match find(&kind, &label) {
            None => default,
            Some(component) => component.is_selected(),
        },
    );
    bridge.on_enabled_value(
        |kind, label, default, _generation| match find(&kind, &label) {
            None => default,
            Some(component) => component.is_enabled(),
        },
    );
    if std::env::var_os("ETOMO_BRIDGE_DEBUG").is_some() {
        bridge.on_activated(|kind, label| {
            eprintln!("slint_bridge: activated ({kind}, {label:?})");
            for (name, component) in showing_components() {
                eprintln!("slint_bridge:   showing {name} {:?}", component.kind());
            }
            if let Some(component) = find(&kind, &label)
                && component.is_enabled()
            {
                component.do_click();
            }
        });
    }
    bridge.set_wired(true);
    update_view(window);
    let timer = slint::Timer::default();
    let weak = window.as_weak();
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(REFRESH_MS),
        move || {
            if let Some(window) = weak.upgrade() {
                let bridge = window.global::<EtomoBridge>();
                bridge.set_generation(bridge.get_generation().wrapping_add(1));
                update_view(&window);
                update_page_windows();
            }
        },
    );
    timer
}

/// Shows a `JOptionPane` popup as a modal dialog and returns the answer, or
/// `None` when the dialog could not be shown.  The dialog is drawn by a child
/// process of this binary (`imod etomo-popup`, [`etomo_popup`]), so the main
/// window's event dispatch thread waits for the answer as Swing's modal
/// `dialog.setVisible(true)` does.
pub fn present_popup(request: &PopupRequest) -> Option<PopupAnswer> {
    use std::io::Write;
    let exe = std::env::current_exe().ok()?;
    let initial = request
        .initial_value
        .as_ref()
        .and_then(|initial| request.options.iter().position(|option| option == initial))
        .map_or(-1, |index| index as i64);
    let mut child = std::process::Command::new(exe)
        .arg("etomo-popup")
        .arg(request.title.as_deref().unwrap_or(""))
        .arg(request.message_type.to_string())
        .arg(initial.to_string())
        .args(&request.options)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    if let Some(mut stdin) = child.stdin.take() {
        for line in &request.message {
            let _ = writeln!(stdin, "{line}");
        }
    }
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    let answer = String::from_utf8_lossy(&output.stdout);
    Some(match answer.trim().parse::<usize>() {
        Ok(index) if index < request.options.len() => PopupAnswer::Selected(index),
        _ => PopupAnswer::Closed,
    })
}

/// Rust-only command `etomo-popup TITLE MESSAGE_TYPE INITIAL [OPTION...]`:
/// shows one `JOptionPane` dialog with the message lines read from standard
/// input, and prints the index of the button pressed (nothing when the dialog
/// was closed).  [`present_popup`] runs it.
pub fn etomo_popup(arguments: &[String]) -> i32 {
    use std::io::Read;
    let mut message = String::new();
    let _ = std::io::stdin().read_to_string(&mut message);
    let Ok(window) = OptionPaneWindow::new() else {
        return 1;
    };
    let title = arguments.first().cloned().unwrap_or_default();
    let message_type = arguments.get(1).and_then(|t| t.parse().ok()).unwrap_or(-1);
    let initial = arguments.get(2).and_then(|t| t.parse().ok()).unwrap_or(-1);
    let mut options: Vec<slint::SharedString> = arguments
        .iter()
        .skip(3)
        .map(|option| option.into())
        .collect();
    if options.is_empty() {
        options.push("OK".into());
    }
    let lines: Vec<slint::SharedString> = message.lines().map(Into::into).collect();
    window.set_title_text(title.into());
    window.set_message_type(message_type);
    window.set_initial(initial);
    window.set_message_lines(slint::ModelRc::new(slint::VecModel::from(lines)));
    window.set_options(slint::ModelRc::new(slint::VecModel::from(options)));
    let chosen = std::rc::Rc::new(std::cell::Cell::new(None::<i32>));
    let chosen_by_click = chosen.clone();
    let weak = window.as_weak();
    window.on_chosen(move |index| {
        chosen_by_click.set(Some(index));
        if let Some(window) = weak.upgrade() {
            let _ = window.hide();
        }
    });
    if window.run().is_err() {
        return 1;
    }
    if let Some(index) = chosen.get() {
        println!("{index}");
    }
    0
}
