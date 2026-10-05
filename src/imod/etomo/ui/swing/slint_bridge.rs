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

use etomo_ui_main_window::{
    BridgedPopupItem, BridgedTableCell, BridgedTableRow, DirectivePanelData, DirectiveSectionData,
    EtomoBridge, FileChooserWindow, HtmlLine, HtmlRun, MainFrameWindow, OptionPaneWindow,
    ParallelPanelRow, SubFrameWindow, TabbedTextWindowWindow, TextPageWindowWindow,
};
use slint::{ComponentHandle, Model};

use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{ComponentKind, GRID_BAG_REMAINDER, JComponent, named_components};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::ui::swing::etomo_menu::ToolType;
use crate::imod::etomo::ui::swing::file_chooser::{self, FileChooser};
use crate::imod::etomo::ui::swing::tabbed_text_window::TabbedTextWindow;
use crate::imod::etomo::ui::swing::text_page_window::{self, TextPageWindow};
use crate::imod::etomo::ui::swing::ui_harness::{self, PopupAnswer, PopupRequest};
use crate::imod::etomo::util::utilities;

/// How often the window re-reads the component tree.
pub const REFRESH_MS: u64 = 150;

/// Which Swing frame a Slint window stands for: the main frame (with the
/// showing modal dialogs and `ManagerFrame`s over it), or the `SubFrame` that
/// shows axis B when both axes are shown.  Each Slint window has its own
/// `EtomoBridge`, and its lookups resolve in its own frame, as the two Swing
/// frames hold different components with the same names (the A and B copies).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    Main,
    Sub,
}

thread_local! {
    /// The frame the bridge callback being run resolves in.
    static SCOPE: std::cell::Cell<Scope> = const { std::cell::Cell::new(Scope::Main) };
    /// The frame whose window asked for the showing context menu.
    static POPUP_SCOPE: std::cell::Cell<Scope> = const { std::cell::Cell::new(Scope::Main) };
}

/// Sets [`SCOPE`] for the life of the guard.
struct ScopeGuard(Scope);

impl ScopeGuard {
    fn enter(scope: Scope) -> ScopeGuard {
        ScopeGuard(SCOPE.with(|current| current.replace(scope)))
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPE.with(|current| current.set(self.0));
    }
}

/// The visible `SubFrame` (axis B when both axes are shown).
fn visible_sub_frame() -> Option<Rc<crate::imod::etomo::ui::swing::sub_frame::SubFrame>> {
    crate::imod::etomo::ui::swing::etomo_frame::sub_frame().filter(|frame| frame.is_visible())
}

/// The roots the current [`SCOPE`] searches: the showing modal dialogs, the
/// visible `ManagerFrame`s and the main frame's root, or the `SubFrame`'s
/// content pane.
fn scope_roots() -> Vec<Rc<JComponent>> {
    if SCOPE.with(|scope| scope.get()) == Scope::Sub {
        return visible_sub_frame()
            .map(|frame| vec![frame.get_content_pane()])
            .unwrap_or_default();
    }
    // A showing modal `JDialog` (a startup dialog) comes first: while it shows,
    // it is the window the user acts in.
    let mut roots: Vec<Rc<JComponent>> = crate::imod::etomo::jdk::showing_dialogs()
        .into_iter()
        .rev()
        .map(|dialog| dialog.get_content_pane())
        .collect();
    // A showing `ManagerFrame` (the directive editor) is drawn over the main
    // frame, so its widgets come before the main frame's.
    roots.extend(showing_manager_frame_roots());
    if let Some(root) = ui_harness::with(|harness| harness.get_main_frame_root()) {
        roots.push(root);
    }
    roots
}

/// The menu bar of the current [`SCOPE`]'s frame.
fn scope_menu_bar() -> Option<Rc<JComponent>> {
    if SCOPE.with(|scope| scope.get()) == Scope::Sub {
        return visible_sub_frame().and_then(|frame| frame.get_j_menu_bar());
    }
    ui_harness::with(|harness| harness.get_main_frame_menu_bar())
}

/// The showing components under the main frame, with their names.
fn showing_components() -> Vec<(String, Rc<JComponent>)> {
    scope_roots()
        .iter()
        .flat_map(named_components)
        .filter(|(_, component)| component.is_showing())
        .collect()
}

/// The content panes of the visible `ManagerFrame`s, newest first.
fn showing_manager_frame_roots() -> Vec<Rc<JComponent>> {
    ui_harness::with(|harness| harness.get_manager_frames())
        .into_iter()
        .rev()
        .filter(|(_, frame)| frame.is_visible())
        .map(|(_, frame)| frame.get_content_pane())
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
    if kind == "tbi" {
        // A `TabbedPane` that has a tab titled `label`.
        return components
            .into_iter()
            .map(|(_, component)| component)
            .filter(|component| {
                component.kind() == ComponentKind::TabbedPane
                    && tab_index(component, label).is_some()
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
    if matches!(kind, "mbo" | "mba" | "mbm") {
        // One of a `PanelHeader`'s `ExpandButton`s (all three are named "mb." +
        // the title), told apart by the symbol it shows, as
        // `ExpandButton.Type.equals` does: open/close, advanced/basic,
        // more/less.
        let symbols: [&str; 2] = match kind {
            "mbo" => ["-", "+"],
            "mba" => ["B", "A"],
            _ => ["<", ">"],
        };
        return find_all("mb", label)
            .into_iter()
            .filter(|button| symbols.contains(&strip_tags(&button.get_text()).as_str()))
            .collect();
    }
    if kind == "lbp" {
        // An unnamed `JLabel`, by the fixed start or end of its text.
        return components_showing()
            .into_iter()
            .filter(|component| {
                component.kind() == ComponentKind::Label && {
                    let text = component.get_text();
                    text.starts_with(label) || text.ends_with(label)
                }
            })
            .collect();
    }
    if kind == "lnx" {
        // An unnamed `JLabel` whose text the Java sets at run time, by the
        // fixed text of the label just before it in the same panel
        // (`SeriesWatcherPanel.lMatchString`, after "File name match string: ").
        return components_showing()
            .into_iter()
            .filter(|component| {
                component.kind() == ComponentKind::Label && component.get_text() == label
            })
            .filter_map(|component| {
                let siblings = component.get_parent()?.get_components();
                let position = siblings
                    .iter()
                    .position(|sibling| Rc::ptr_eq(sibling, &component))?;
                siblings
                    .get(position + 1)
                    .filter(|next| next.kind() == ComponentKind::Label)
                    .cloned()
            })
            .collect();
    }
    if kind == "lsx" {
        // An unnamed `JList` in a `JScrollPane`, by the fixed text of the label
        // just before the scroll pane in the same panel
        // (`SettingsDialog.listFontFamily`, after "Font family:").
        return components_showing()
            .into_iter()
            .filter(|component| {
                component.kind() == ComponentKind::Label && component.get_text() == label
            })
            .filter_map(|component| {
                let siblings = component.get_parent()?.get_components();
                let position = siblings
                    .iter()
                    .position(|sibling| Rc::ptr_eq(sibling, &component))?;
                siblings
                    .get(position + 1)
                    .filter(|next| next.kind() == ComponentKind::ScrollPane)?
                    .get_components()
                    .into_iter()
                    .next()
            })
            .collect();
    }
    if kind == "acb" {
        // A check box with no name (`new CheckBox()`), by its action command:
        // the directive editor's include/exclude check boxes.
        return components_showing()
            .into_iter()
            .filter(|component| {
                component.kind() == ComponentKind::CheckBox
                    && component.get_name().is_none()
                    && component.get_action_command().as_deref() == Some(label)
            })
            .collect();
    }
    if kind == "fmn" {
        // A menu item, by its text, under the showing `ManagerFrame`'s menu bar.
        let mut found = Vec::new();
        for (_, frame) in ui_harness::with(|harness| harness.get_manager_frames())
            .into_iter()
            .rev()
            .filter(|(_, frame)| frame.is_visible())
        {
            if let Some(menu_bar) = frame.get_j_menu_bar() {
                fn walk(node: &Rc<JComponent>, label: &str, found: &mut Vec<Rc<JComponent>>) {
                    if node.kind() == ComponentKind::MenuItem && node.get_text() == label {
                        found.push(node.clone());
                    }
                    for child in node.get_components() {
                        walk(&child, label, found);
                    }
                }
                walk(&menu_bar, label, &mut found);
            }
        }
        return found;
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
        let Some(menu_bar) = scope_menu_bar() else {
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
    let suffixed: Vec<_> = components
        .iter()
        .filter(|(n, _)| names.iter().any(|name| n.starts_with(&format!("{name}-"))))
        .map(|(_, component)| component.clone())
        .collect();
    if !suffixed.is_empty() || kind != "bn" {
        return suffixed;
    }
    // A plain `new JButton(label)` has no name (the dialog buttons of
    // `SettingsDialog`, `MainFrame_AboutBox`): the showing unnamed button with
    // that text.
    components_showing()
        .into_iter()
        .filter(|component| {
            component.kind() == ComponentKind::Button
                && component.get_name().is_none()
                && component.get_text() == label
        })
        .collect()
}

/// Whether the Slint widget `(kind, label)` is to be drawn: the Swing component
/// it stands for is showing (`Component.isShowing()`, the jdk stand-in's
/// `is_showing`: it and all its ancestors visible, and on the selected tab).
/// A widget whose component exists but does not show -- an advanced field in
/// basic mode, a panel hidden with `setVisible(false)`, a body under a closed
/// `PanelHeader` -- or that the Java does not build in this configuration is not
/// drawn.  Kind "brd" is a panel by its titled border: drawn only while such a
/// panel shows, so a titled panel the Java builds lazily is not drawn before it
/// exists.  Kind "ph" (a `PanelHeader`: its `ExpandButton`s "mb" or its title
/// `HeaderCell` "hc") is hidden only when such a component exists and none of
/// them shows, since a title-only header has no name of its own to miss.
fn visible_value(kind: &str, label: &str) -> bool {
    if label.is_empty() {
        return true;
    }
    match kind {
        "brd" => {
            // A titled panel the Java has not built (yet) is not drawn: several
            // panels are built lazily (on first selection, in advanced mode, ...),
            // and Swing has nothing to show until then.
            for component in components_all() {
                if component.get_border_title().as_deref() == Some(label) && component.is_showing()
                {
                    return true;
                }
            }
            if std::env::var_os("ETOMO_BRIDGE_DEBUG").is_some()
                && !components_all()
                    .iter()
                    .any(|component| component.get_border_title().as_deref() == Some(label))
            {
                eprintln!("slint_bridge: no titled panel at all for ({kind}, {label:?})");
            }
            false
        }
        "ph" => {
            if find_unlogged("mb", label).is_some() || find_unlogged("hc", label).is_some() {
                return true;
            }
            let names: Vec<String> = names_for("mb", label)
                .into_iter()
                .chain(names_for("hc", label))
                .collect();
            !components_all().iter().any(|component| {
                component
                    .get_name()
                    .is_some_and(|name| names.contains(&name))
            })
        }
        _ => {
            let shown = find_unlogged(kind, label).is_some();
            // `ETOMO_BRIDGE_DEBUG=1`: report a widget whose component does not
            // exist at all (neither showing nor hidden) -- the Java does not
            // build it here, or the Slint label differs from the Java's.
            if !shown && std::env::var_os("ETOMO_BRIDGE_DEBUG").is_some() {
                let base = label.rsplit_once('#').map_or(label, |(base, _)| base);
                let names: Vec<String> = base.split('|').flat_map(|l| names_for(kind, l)).collect();
                let exists = components_all().iter().any(|component| {
                    component.get_name().is_some_and(|name| {
                        names
                            .iter()
                            .any(|n| name == *n || name.starts_with(&format!("{n}-")))
                    })
                });
                if !exists
                    && !matches!(
                        kind,
                        "pcp"
                            | "tbi"
                            | "fbn"
                            | "acb"
                            | "mn"
                            | "cbmn"
                            | "fmn"
                            | "mbo"
                            | "mba"
                            | "mbm"
                    )
                {
                    eprintln!("slint_bridge: no component at all for ({kind}, {label:?})");
                }
            }
            shown
        }
    }
}

/// Every component, showing or not, under the showing dialogs, the visible
/// `ManagerFrame`s and the main frame, in tree order.
fn components_all() -> Vec<Rc<JComponent>> {
    let roots = scope_roots();
    fn walk(node: &Rc<JComponent>, out: &mut Vec<Rc<JComponent>>) {
        out.push(node.clone());
        for child in node
            .get_components()
            .into_iter()
            .chain(node.get_tab_title_components())
        {
            walk(&child, out);
        }
    }
    let mut out = Vec::new();
    for root in &roots {
        walk(root, &mut out);
    }
    out
}

/// Every showing component, named or not, under the showing dialogs, the
/// visible `ManagerFrame`s and the main frame, in tree order.
fn components_showing() -> Vec<Rc<JComponent>> {
    let roots = scope_roots();
    fn walk(node: &Rc<JComponent>, out: &mut Vec<Rc<JComponent>>) {
        if !node.is_showing() {
            return;
        }
        out.push(node.clone());
        for child in node.get_components() {
            walk(&child, out);
        }
    }
    let mut out = Vec::new();
    for root in &roots {
        walk(root, &mut out);
    }
    out
}

/// The index of the tab titled `title` in a tabbed pane.
fn tab_index(tabbed_pane: &JComponent, title: &str) -> Option<usize> {
    (0..tabbed_pane.get_tab_count())
        .find(|&index| tabbed_pane.get_title_at(index).as_deref() == Some(title))
}

/// The panel of the GridBag table `table`: under the showing component whose
/// titled border is `table` (else whose name is `table`), the first showing
/// panel, depth first, whose children carry GridBag constraints.
///
/// This is the generic table reader the Slint `BridgedTable` uses.  A Swing
/// table in eTomo is a `JPanel` with a `GridBagLayout` whose cells
/// (`HeaderCell`, `FieldCell`, `CheckBoxCell`, `SpinnerCell`,
/// `HighlighterButton`, ...) are added in row order, each row ending in a
/// cell whose `gridwidth` is `REMAINDER`; the translated cells record their
/// constraints on their component (`GridBagLayout.setConstraints`), so the
/// rows can be read back from the component tree whatever the table is
/// (join section and boundary tables, the serial-sections, batch and PEET
/// tables).
fn table_panel(table: &str) -> Option<Rc<JComponent>> {
    // "child:<name>": the showing panel with a direct child of that uitest name
    // (the `BatchRunTomoTable` grid, whose views sit under an untitled border
    // in two of its three tabs: "child:hc.#").
    if let Some(name) = table.strip_prefix("child:") {
        return components_showing().into_iter().find_map(|component| {
            (component.get_name().as_deref() == Some(name))
                .then(|| component.get_parent())
                .flatten()
        });
    }
    // "<title>/..": the walk starts that many ancestors above the titled panel
    // (a table that is an untitled sibling of a titled panel:
    // `DirectivesDialog`'s table beside "Which Directives to Show").
    if let Some(base) = table.strip_suffix("/..") {
        let mut levels = 1;
        let mut base = base;
        while let Some(shorter) = base.strip_suffix("/..") {
            levels += 1;
            base = shorter;
        }
        let roots = scope_roots();
        fn find_titled_up(node: &Rc<JComponent>, title: &str) -> Option<Rc<JComponent>> {
            if !node.is_showing() {
                return None;
            }
            if node.get_border_title().as_deref() == Some(title) {
                return Some(node.clone());
            }
            node.get_components()
                .iter()
                .find_map(|child| find_titled_up(child, title))
        }
        let mut node = roots.iter().find_map(|root| find_titled_up(root, base))?;
        for _ in 0..levels {
            node = node.get_parent()?;
        }
        fn walk_up(node: &Rc<JComponent>) -> Option<Rc<JComponent>> {
            if !node.is_showing() {
                return None;
            }
            if node
                .get_components()
                .iter()
                .any(|child| child.get_layout_constraints().is_some())
            {
                return Some(node.clone());
            }
            node.get_components().iter().find_map(walk_up)
        }
        return walk_up(&node);
    }
    let components = showing_components();
    // The titled border may sit on an unnamed panel (a plain `new JPanel()`,
    // as VolumeTable and IterationTable use), so every showing node under the
    // showing dialogs and the main frame is searched for it, not only the
    // named ones.
    let roots = scope_roots();
    fn find_titled(node: &Rc<JComponent>, table: &str) -> Option<Rc<JComponent>> {
        if !node.is_showing() {
            return None;
        }
        if node.get_border_title().as_deref() == Some(table) {
            return Some(node.clone());
        }
        node.get_components()
            .iter()
            .find_map(|child| find_titled(child, table))
    }
    let titled = roots.iter().find_map(|root| find_titled(root, table));
    let root = titled
        .as_ref()
        .or_else(|| {
            components
                .iter()
                .find(|(name, _)| name == table)
                .map(|(_, component)| component)
        })?
        .clone();
    fn walk(node: &Rc<JComponent>) -> Option<Rc<JComponent>> {
        if !node.is_showing() {
            return None;
        }
        if node
            .get_components()
            .iter()
            .any(|child| child.get_layout_constraints().is_some())
        {
            return Some(node.clone());
        }
        node.get_components().iter().find_map(walk)
    }
    walk(&root)
}

/// The rows of the GridBag table `table`, for `BridgedTable`.
fn table_rows(table: &str) -> Vec<BridgedTableRow> {
    let Some(panel) = table_panel(table) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    let mut cells: Vec<BridgedTableCell> = Vec::new();
    for (index, child) in panel.get_components().iter().enumerate() {
        let Some(constraints) = child.get_layout_constraints() else {
            continue;
        };
        if child.is_showing() {
            let name = child.get_name().unwrap_or_default();
            let kind = if name.starts_with("hc.") {
                "hc"
            } else {
                match child.kind() {
                    ComponentKind::Button => "bn",
                    ComponentKind::ToggleButton => "tb",
                    ComponentKind::TextField | ComponentKind::TextArea => "tf",
                    ComponentKind::CheckBox => "cb",
                    ComponentKind::RadioButton => "rb",
                    ComponentKind::Spinner => "sp",
                    ComponentKind::Label => "lb",
                    _ => "",
                }
            };
            let text = if kind == "sp" {
                let value = child.get_spinner_value();
                if child.get_spinner_model().is_some_and(|model| model.integer) {
                    format!("{}", value as i64)
                } else {
                    format!("{value}")
                }
            } else {
                strip_tags(&child.get_text())
            };
            cells.push(BridgedTableCell {
                index: index as i32,
                kind: kind.into(),
                text: text.into(),
                enabled: child.is_enabled(),
                editable: child.is_editable(),
                selected: child.is_selected(),
                span: constraints.gridwidth,
            });
        }
        if constraints.gridwidth == GRID_BAG_REMAINDER && !cells.is_empty() {
            rows.push(BridgedTableRow {
                cells: slint::ModelRc::new(slint::VecModel::from(std::mem::take(&mut cells))),
            });
        }
    }
    if !cells.is_empty() {
        rows.push(BridgedTableRow {
            cells: slint::ModelRc::new(slint::VecModel::from(cells)),
        });
    }
    // A REMAINDER cell extends to the grid's last column (`GridBagLayout`), so
    // it spans the columns its row's other cells leave of the widest row: the
    // Section Table's "Rotation Angles" header covers X, Y and Z.
    use slint::Model;
    let columns = |row: &BridgedTableRow| row.cells.iter().map(|c| c.span.max(1)).sum::<i32>();
    let width = rows.iter().map(columns).max().unwrap_or(0);
    for row in &rows {
        let used: i32 = row
            .cells
            .iter()
            .filter(|c| c.span != GRID_BAG_REMAINDER)
            .map(|c| c.span.max(1))
            .sum();
        for i in 0..row.cells.row_count() {
            let mut cell = row.cells.row_data(i).unwrap();
            if cell.span == GRID_BAG_REMAINDER {
                cell.span = (width - used).max(1);
                row.cells.set_row_data(i, cell);
            }
        }
    }
    rows
}

thread_local! {
    /// One persistent row model per bridged table, and per row its cell model:
    /// a refresh updates the rows and cells that changed in place, so the
    /// Slint repeaters keep their elements (a field being typed in keeps its
    /// focus) instead of being rebuilt every refresh.
    static TABLE_MODELS: RefCell<
        std::collections::HashMap<
            String,
            (
                Rc<slint::VecModel<BridgedTableRow>>,
                Vec<Rc<slint::VecModel<BridgedTableCell>>>,
            ),
        >,
    > = RefCell::new(std::collections::HashMap::new());
}

/// The persistent model of the GridBag table `table`, brought up to date with
/// the Swing tree ([`table_rows`]).
fn table_model(table: &str) -> slint::ModelRc<BridgedTableRow> {
    use slint::Model;
    let rows = table_rows(table);
    TABLE_MODELS.with(|models| {
        let mut models = models.borrow_mut();
        let (row_model, cell_models) = models
            .entry(table.to_owned())
            .or_insert_with(|| (Rc::new(slint::VecModel::default()), Vec::new()));
        for (index, row) in rows.iter().enumerate() {
            let cells: Vec<BridgedTableCell> = row.cells.iter().collect();
            if index < cell_models.len() {
                let cell_model = &cell_models[index];
                if cell_model.row_count() != cells.len() {
                    cell_model.set_vec(cells);
                } else {
                    for (cell_index, cell) in cells.into_iter().enumerate() {
                        if cell_model.row_data(cell_index).as_ref() != Some(&cell) {
                            cell_model.set_row_data(cell_index, cell);
                        }
                    }
                }
            } else {
                let cell_model = Rc::new(slint::VecModel::from(cells));
                row_model.push(BridgedTableRow {
                    cells: slint::ModelRc::from(cell_model.clone()),
                });
                cell_models.push(cell_model);
            }
        }
        while cell_models.len() > rows.len() {
            cell_models.pop();
            row_model.remove(cell_models.len());
        }
        slint::ModelRc::from(row_model.clone())
    })
}

/// Cell `index` of the GridBag table `table`.
fn table_cell(table: &str, index: i32) -> Option<Rc<JComponent>> {
    table_panel(table)?
        .get_components()
        .into_iter()
        .nth(usize::try_from(index).ok()?)
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

/// The labels of a showing `MainFrame_AboutBox` ("About" `JDialog`), in the
/// order its constructor adds them: the eTomo line, the version, the two
/// `imodinfo` lines when there are more than two, the authors, then the IMOD
/// and PEET versions when known.
fn update_about_box(window: &MainFrameWindow) {
    let Some(dialog) = crate::imod::etomo::jdk::showing_dialogs()
        .into_iter()
        .rev()
        .find(|dialog| dialog.get_title() == "About")
    else {
        return;
    };
    fn labels(node: &Rc<JComponent>, out: &mut Vec<String>) {
        if node.kind() == ComponentKind::Label {
            out.push(node.get_text());
        }
        for child in node.get_components() {
            labels(&child, out);
        }
    }
    let mut texts = Vec::new();
    labels(&dialog.get_content_pane(), &mut texts);
    window.set_about_version_text(texts.get(1).cloned().unwrap_or_default().into());
    let authors = texts
        .iter()
        .position(|text| text.starts_with("Written by: "))
        .unwrap_or(texts.len());
    window.set_about_show_imod_info(authors >= 4);
    if authors >= 4 {
        window.set_about_imod_info_1(texts[2].clone().into());
        window.set_about_imod_info_2(texts[3].clone().into());
    }
    let imod_version = texts
        .iter()
        .find_map(|text| text.strip_prefix("IMOD Version: "));
    window.set_about_show_imod_version(imod_version.is_some());
    window.set_about_imod_version(imod_version.unwrap_or_default().into());
    let peet_version = texts
        .iter()
        .find_map(|text| text.strip_prefix("PEET Version: "));
    window.set_about_show_peet_version(peet_version.is_some());
    window.set_about_peet_version(peet_version.unwrap_or_default().into());
}

/// Sets which manager panel, axis type and dialog the window shows, from the
/// director's current manager.
fn update_view(window: &MainFrameWindow) {
    // A showing modal `JDialog` (a project's startup dialog) is drawn over the
    // frame, by its title.
    window.set_startup_dialog(
        crate::imod::etomo::jdk::showing_dialogs()
            .last()
            .map(|dialog| dialog.get_title())
            .unwrap_or_default()
            .into(),
    );
    // A dataset's own `BatchRunTomoDatasetDialog` frame (`getRowInstance`, the
    // only dialog with a "Revert to Global" button).
    window.set_startup_dialog_batch_dataset(
        crate::imod::etomo::jdk::showing_dialogs()
            .last()
            .is_some_and(|dialog| {
                fn has_named(node: &Rc<JComponent>, name: &str) -> bool {
                    node.get_name().as_deref() == Some(name)
                        || node
                            .get_components()
                            .iter()
                            .any(|child| has_named(child, name))
                }
                has_named(&dialog.get_content_pane(), "bn.revert-to-global")
            }),
    );
    update_about_box(window);
    // `MainFrame.setTitle`: "<manager> - Etomo", prefixed "A Axis - " / "B Axis - "
    // for a dual axis.
    if let Some(main_frame) = ui_harness::with(|harness| harness.get_main_frame()) {
        let title = main_frame.get_title();
        if window.get_frame_title().as_str() != title {
            window.set_frame_title(title.into());
        }
    }
    window.set_peet_available(!find_all("mn", "PEET Help").is_empty());
    update_manager_frame(window);
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
            // `MainPanel.showAxisB` puts axis B in `panelCenter` (and clears
            // `showingAxisA`); `showAxisA` and `showBothAxis` keep axis A there.
            let showing_b = axis_type == AxisType::DualAxis
                && main_panel
                    .as_ref()
                    .is_some_and(|main_panel| !main_panel.main_panel().is_showing_axis_a());
            window.set_showing_axis(if showing_b { "B" } else { "A" }.into());
            let dialog_type = manager.get_current_dialog_type(Some(if showing_b {
                AxisID::Second
            } else {
                AxisID::First
            }));
            window.set_dialog(dialog_type.map(dialog_name).unwrap_or("").into());
            // `AxisProcessPanel.parallelStatusPanel` of the shown axis.
            let parallel_status_showing = main_panel.as_ref().is_some_and(|main_panel| {
                main_panel
                    .main_panel()
                    .get_parallel_status_panel(if showing_b {
                        AxisID::Second
                    } else {
                        AxisID::First
                    })
                    .is_some_and(|panel| panel.is_showing())
            });
            window.set_show_parallel_panel(parallel_status_showing);
            if parallel_status_showing {
                window.set_parallel_rows(parallel_rows_model());
            }
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
            // `ToolsDialog.taTaskLog`: the last showing text area in a scroll
            // pane (`createPanel` adds `scrTaskLog` after the tool panel).
            let task_log = components_showing()
                .into_iter()
                .filter(|component| {
                    component.kind() == ComponentKind::TextArea
                        && component
                            .get_parent()
                            .is_some_and(|parent| parent.kind() == ComponentKind::ScrollPane)
                })
                .last()
                .map(|text_area| text_area.get_text())
                .unwrap_or_default();
            if window.get_tools_task_log().as_str() != task_log {
                window.set_tools_task_log(task_log.into());
            }
        }
        Some(InterfaceType::Join) => {
            window.set_view("join".into());
            if let Some(main_panel) = manager.get_main_panel() {
                window.set_status_bar_text(main_panel.main_panel().get_status_bar_text().into());
            }
        }
        Some(InterfaceType::Peet) => {
            window.set_view("peet".into());
            let main_panel = manager.get_main_panel();
            if let Some(main_panel) = main_panel.as_ref() {
                window.set_status_bar_text(main_panel.main_panel().get_status_bar_text().into());
            }
            // `PeetManager.openPeetDialog` shows the `PeetDialog` in `panelDialog`
            // once the startup dialog is done: its "PEET" border panel shows.
            window.set_peet_dialog_shown(find_unlogged("tbi", "Setup").is_some());
            let parallel_status_showing = main_panel.as_ref().is_some_and(|main_panel| {
                main_panel
                    .main_panel()
                    .get_parallel_status_panel(AxisID::Only)
                    .is_some_and(|panel| panel.is_showing())
            });
            window.set_show_parallel_panel(parallel_status_showing);
            let rows: Vec<ParallelPanelRow> = processor_table_rows()
                .into_iter()
                .map(|(computer, selected, cells)| ParallelPanelRow {
                    computer: computer.into(),
                    selected,
                    cells: slint::ModelRc::new(slint::VecModel::from(
                        cells
                            .into_iter()
                            .map(Into::into)
                            .collect::<Vec<slint::SharedString>>(),
                    )),
                })
                .collect();
            window.set_parallel_rows(slint::ModelRc::new(slint::VecModel::from(rows)));
        }
        Some(InterfaceType::SerialSections) => {
            window.set_view("serial-sections".into());
            if let Some(main_panel) = manager.get_main_panel() {
                window.set_status_bar_text(main_panel.main_panel().get_status_bar_text().into());
            }
            // `SerialSectionsManager` shows the `SerialSectionsDialog` in
            // `panelDialog` once the startup dialog is done: its tab pane.
            window
                .set_serial_sections_dialog_shown(find_unlogged("tbi", "Initial Blend").is_some());
        }
        Some(InterfaceType::BatchRunTomo) => {
            window.set_view("batch".into());
            let main_panel = manager.get_main_panel();
            if let Some(main_panel) = main_panel.as_ref() {
                window.set_status_bar_text(main_panel.main_panel().get_status_bar_text().into());
            }
            // `BatchRunTomoManager` shows the `BatchRunTomoDialog` in
            // `panelDialog`: its tab pane.
            window.set_batch_dialog_shown(find_unlogged("tbi", "Batch Setup").is_some());
            // The Run tab's `parallelStatusPanel` (`altParallelLoc`).
            let parallel_status_showing = main_panel.as_ref().is_some_and(|main_panel| {
                main_panel
                    .main_panel()
                    .get_parallel_status_panel(AxisID::Only)
                    .is_some_and(|panel| panel.is_showing())
            });
            window.set_show_parallel_panel(parallel_status_showing);
            if parallel_status_showing {
                window.set_parallel_rows(parallel_rows_model());
            }
        }
        Some(InterfaceType::Pp) => {
            window.set_view("parallel".into());
            let main_panel = manager.get_main_panel();
            if let Some(main_panel) = main_panel.as_ref() {
                window.set_status_bar_text(main_panel.main_panel().get_status_bar_text().into());
            }
            // The process `ParallelManager` showed in `panelDialog`, by a widget
            // only that dialog has.
            let parallel_dialog = if find_unlogged("bn", "Run Chunksetup").is_some() {
                "PARALLEL"
            } else if find_unlogged("bn", "Extract Test Volume").is_some() {
                "ANISOTROPIC_DIFFUSION"
            } else if find_unlogged("bn", "Generic Parallel Process").is_some() {
                "chooser"
            } else {
                ""
            };
            window.set_parallel_dialog(parallel_dialog.into());
            let parallel_status_showing = main_panel.as_ref().is_some_and(|main_panel| {
                main_panel
                    .main_panel()
                    .get_parallel_status_panel(AxisID::Only)
                    .is_some_and(|panel| panel.is_showing())
            });
            window.set_show_parallel_panel(parallel_status_showing);
            let rows: Vec<ParallelPanelRow> = processor_table_rows()
                .into_iter()
                .map(|(computer, selected, cells)| ParallelPanelRow {
                    computer: computer.into(),
                    selected,
                    cells: slint::ModelRc::new(slint::VecModel::from(
                        cells
                            .into_iter()
                            .map(Into::into)
                            .collect::<Vec<slint::SharedString>>(),
                    )),
                })
                .collect();
            window.set_parallel_rows(slint::ModelRc::new(slint::VecModel::from(rows)));
        }
        _ => {
            window.set_view("".into());
        }
    }
}

thread_local! {
    /// The directive editor's three section columns as last given to the window
    /// (plain mirror of the Slint models), so a model is only replaced when the
    /// shown sections or rows change and a field being typed in keeps its focus.
    static DIRECTIVE_COLUMNS: RefCell<Vec<Vec<DirectiveSectionMirror>>> =
        const { RefCell::new(Vec::new()) };
}

/// A plain copy of a `DirectiveSectionData`: title, show-check-box occurrence,
/// shown, and the rows (include text, include occurrence, value kind, value label).
type DirectiveSectionMirror = (String, String, bool, Vec<(String, String, String, String)>);

/// The "directive-editor" view: a visible `ManagerFrame` (only
/// `DirectiveEditorManager` lives in one, `isInManagerFrame`) is drawn over the
/// main frame with its title, status bar and `DirectiveEditorDialog`.
fn update_manager_frame(window: &MainFrameWindow) {
    let frame = ui_harness::with(|harness| harness.get_manager_frames())
        .into_iter()
        .rev()
        .find(|(_, frame)| frame.is_visible());
    let Some((manager, frame)) = frame else {
        window.set_manager_frame("".into());
        return;
    };
    let title = frame.get_title();
    // An empty title would hide the frame; Swing shows an untitled frame.
    window.set_manager_frame(if title.is_empty() {
        " ".into()
    } else {
        title.into()
    });
    if let Some(main_panel) = manager.get_main_panel() {
        window.set_manager_frame_status_bar_text(
            main_panel.main_panel().get_status_bar_text().into(),
        );
    }
    let columns = directive_editor_columns(manager);
    let total: usize = columns.iter().map(Vec::len).sum();
    let remainder = total % 3;
    window.set_directive_column_1_extra_row(remainder > 0);
    window.set_directive_column_2_extra_row(remainder > 1);
    window.set_directive_column_3_extra_row(false);
    DIRECTIVE_COLUMNS.with_borrow_mut(|previous| {
        if previous.len() != 3 {
            *previous = vec![Vec::new(), Vec::new(), Vec::new()];
            window.set_directive_column_1(directive_column_model(&[]));
            window.set_directive_column_2(directive_column_model(&[]));
            window.set_directive_column_3(directive_column_model(&[]));
        }
        for (index, column) in columns.into_iter().enumerate().take(3) {
            if previous[index] == column {
                continue;
            }
            let model = directive_column_model(&column);
            match index {
                0 => window.set_directive_column_1(model),
                1 => window.set_directive_column_2(model),
                _ => window.set_directive_column_3(model),
            }
            previous[index] = column;
        }
    });
}

/// The Slint model of one section column.
fn directive_column_model(
    column: &[DirectiveSectionMirror],
) -> slint::ModelRc<DirectiveSectionData> {
    let sections: Vec<DirectiveSectionData> = column
        .iter()
        .map(
            |(title, show_occurrence, shown, rows)| DirectiveSectionData {
                title: title.as_str().into(),
                show_occurrence: show_occurrence.as_str().into(),
                shown: *shown,
                directives: slint::ModelRc::new(slint::VecModel::from(
                    rows.iter()
                        .map(
                            |(include_text, include_occurrence, value_kind, value_label)| {
                                DirectivePanelData {
                                    include_text: include_text.as_str().into(),
                                    include_occurrence: include_occurrence.as_str().into(),
                                    value_kind: value_kind.as_str().into(),
                                    value_label: value_label.as_str().into(),
                                }
                            },
                        )
                        .collect::<Vec<_>>(),
                )),
            },
        )
        .collect();
    slint::ModelRc::new(slint::VecModel::from(sections))
}

/// The sections of the directive editor `manager`'s `DirectiveEditorDialog`, read
/// from the component tree `createPanel` built: `pnlRoot` holds `pnlSource`,
/// `pnlControl` and `pnlSections`, whose three columns (children 0, 2 and 4;
/// separators between) hold one panel per section with its `cbShow`, then the
/// sections' `pnlRoot`s (titled borders).  A section's rows are the showing
/// `DirectivePanel`s of its `pnlDirectives` (`pnlRoot` > `pnlBody` > second
/// child), each `cbInclude` then `cbValue` or `tfValue` (+ the file button).
/// Every widget's bridge label carries "#N" when N showing widgets share its
/// name, counted in the order the bridge searches.
fn directive_editor_columns(manager: &'static dyn BaseManager) -> Vec<Vec<DirectiveSectionMirror>> {
    let Some(container) =
        crate::imod::etomo::directive_editor_manager::dialog_container_of(manager)
    else {
        return Vec::new();
    };
    let showing = showing_components();
    // The occurrence of `component` among the showing components of its name.
    let occurrence = |component: &Rc<JComponent>| -> String {
        let Some(name) = component.get_name() else {
            return String::new();
        };
        let same: Vec<&Rc<JComponent>> = showing
            .iter()
            .filter(|(other, _)| *other == name)
            .map(|(_, other)| other)
            .collect();
        if same.len() <= 1 {
            return String::new();
        }
        match same.iter().position(|other| Rc::ptr_eq(other, component)) {
            Some(index) => format!("#{}", index + 1),
            None => String::new(),
        }
    };
    let Some(pnl_sections) = container.get_components().get(2).cloned() else {
        return Vec::new();
    };
    let mut columns = Vec::new();
    for column in pnl_sections
        .get_components()
        .into_iter()
        .filter(|child| child.kind() == ComponentKind::Panel)
    {
        let children = column.get_components();
        let show_check_boxes: Vec<Rc<JComponent>> = children
            .iter()
            .filter(|child| child.get_border_title().is_none())
            .filter_map(|child| child.get_components().first().cloned())
            .filter(|child| child.kind() == ComponentKind::CheckBox)
            .collect();
        let section_roots: Vec<Rc<JComponent>> = children
            .iter()
            .filter(|child| child.get_border_title().is_some())
            .cloned()
            .collect();
        let mut sections = Vec::new();
        for (show_check_box, section_root) in show_check_boxes.iter().zip(section_roots.iter()) {
            let shown = section_root.is_showing();
            let mut rows = Vec::new();
            if shown
                && let Some(pnl_body) = section_root.get_components().first().cloned()
                && let Some(pnl_directives) = pnl_body.get_components().get(1).cloned()
            {
                for directive_panel in pnl_directives.get_components() {
                    if !directive_panel.is_showing() {
                        continue;
                    }
                    let parts = directive_panel.get_components();
                    let (Some(cb_include), Some(value)) = (parts.first(), parts.get(1)) else {
                        continue;
                    };
                    let include_text = cb_include.get_text();
                    // `" -  " + title + ": "`: the value widget is named after the
                    // title.
                    let title = include_text
                        .strip_prefix(" -  ")
                        .and_then(|rest| rest.strip_suffix(": "))
                        .unwrap_or("")
                        .to_string();
                    let value_kind = if value.kind() == ComponentKind::ComboBox {
                        "combo"
                    } else if parts.len() > 2 {
                        "file"
                    } else {
                        "text"
                    };
                    rows.push((
                        include_text,
                        occurrence(cb_include),
                        value_kind.to_string(),
                        format!("{}{}", title, occurrence(value)),
                    ));
                }
            }
            sections.push((
                show_check_box.get_text(),
                occurrence(show_check_box),
                shown,
                rows,
            ));
        }
        columns.push(sections);
    }
    columns
}

/// The rows of the showing `ProcessorTable` as the Slint `ParallelPanel` model.
fn parallel_rows_model() -> slint::ModelRc<ParallelPanelRow> {
    let rows: Vec<ParallelPanelRow> = processor_table_rows()
        .into_iter()
        .map(|(computer, selected, cells)| ParallelPanelRow {
            computer: computer.into(),
            selected,
            cells: slint::ModelRc::new(slint::VecModel::from(
                cells
                    .into_iter()
                    .map(Into::into)
                    .collect::<Vec<slint::SharedString>>(),
            )),
        })
        .collect();
    slint::ModelRc::new(slint::VecModel::from(rows))
}

/// The rows of the showing `ProcessorTable` (`ParallelPanel`), for the parallel
/// view: each row starts with its computer check box ("cb.computer", HTML text
/// whose tags are dropped) and runs to the next row's check box; the cells
/// between are the row's displayed field and spinner cells, in table order.
fn processor_table_rows() -> Vec<(String, bool, Vec<String>)> {
    let check_boxes = find_all("cb", "Computer");
    let mut rows = Vec::new();
    for check_box in &check_boxes {
        let mut cells = Vec::new();
        if let Some(parent) = check_box.get_parent() {
            let children = parent.get_components();
            if let Some(start) = children
                .iter()
                .position(|child| Rc::ptr_eq(child, check_box))
            {
                for child in children.iter().skip(start + 1) {
                    if check_boxes.iter().any(|other| Rc::ptr_eq(other, child)) {
                        break;
                    }
                    if !child.is_showing() {
                        continue;
                    }
                    cells.push(match child.kind() {
                        ComponentKind::Spinner => format!("{}", child.get_spinner_value() as i64),
                        _ => strip_tags(&child.get_text()),
                    });
                }
            }
        }
        rows.push((
            strip_tags(&check_box.get_text()),
            check_box.is_selected(),
            cells,
        ));
    }
    rows
}

/// A Swing HTML label's text without its tags, with the character entities
/// Swing's HTML renderer decodes (`&gt`, `&lt`, `&amp`, `&nbsp`, with or
/// without the closing `;`).
fn strip_tags(text: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    for (entity, character) in [("&gt", ">"), ("&lt", "<"), ("&nbsp", " "), ("&amp", "&")] {
        out = out
            .replace(&format!("{entity};"), character)
            .replace(entity, character);
    }
    out
}

thread_local! {
    /// The Slint window drawn for each shown `TextPageWindow`.
    static PAGE_WINDOWS: RefCell<Vec<(Rc<TextPageWindow>, TextPageWindowWindow)>> =
        const { RefCell::new(Vec::new()) };
}

/// Opens a Slint window for every newly shown `TextPageWindow`, and closes the
/// ones whose `TextPageWindow` was hidden or disposed.  Closing the Slint window
/// is the window system's `WINDOW_CLOSING` event (`processWindowEvent`).
thread_local! {
    /// The visible `TabbedTextWindow`s and the Slint windows drawing them.
    static TABBED_WINDOWS: RefCell<Vec<(Rc<TabbedTextWindow>, TabbedTextWindowWindow)>> =
        const { RefCell::new(Vec::new()) };
}

/// A `TabbedTextWindow`'s `JTabbedPane`: its tab titles, the selected tab and
/// that tab's editor-pane text (scroll pane -> editor pane).
fn tabbed_text_contents(window: &TabbedTextWindow) -> (Vec<String>, i32, String) {
    let Some(tab_pane) = window
        .get_content_pane()
        .get_components()
        .into_iter()
        .find(|component| component.kind() == ComponentKind::TabbedPane)
    else {
        return (Vec::new(), -1, String::new());
    };
    let titles = (0..tab_pane.get_tab_count())
        .map(|index| tab_pane.get_title_at(index).unwrap_or_default())
        .collect();
    let selected = tab_pane.get_selected_index();
    let text = usize::try_from(selected)
        .ok()
        .and_then(|index| tab_pane.get_component_at(index))
        .and_then(|scroll_pane| {
            scroll_pane
                .get_components()
                .into_iter()
                .find(|component| component.kind() == ComponentKind::TextArea)
        })
        .map(|editor_pane| editor_pane.get_text())
        .unwrap_or_default();
    (titles, selected, text)
}

/// Draws each visible `TabbedTextWindow` in a Slint window of its own; a press
/// on a tab selects it, and closing the window disposes the frame
/// (`DISPOSE_ON_CLOSE`).
fn update_tabbed_text_windows() {
    let shown = crate::imod::etomo::ui::swing::tabbed_text_window::get_windows();
    TABBED_WINDOWS.with(|tabbed_windows| {
        let mut tabbed_windows = tabbed_windows.borrow_mut();
        tabbed_windows.retain(|(window, slint_window)| {
            let keep = window.is_visible() && shown.iter().any(|s| Rc::ptr_eq(s, window));
            if !keep {
                let _ = slint_window.hide();
            }
            keep
        });
        for window in shown {
            if !window.is_visible() || tabbed_windows.iter().any(|(w, _)| Rc::ptr_eq(w, &window)) {
                continue;
            }
            let Ok(slint_window) = TabbedTextWindowWindow::new() else {
                continue;
            };
            let refresh = |window: &TabbedTextWindow, slint_window: &TabbedTextWindowWindow| {
                let (titles, selected, text) = tabbed_text_contents(window);
                slint_window.set_tabs(slint::ModelRc::new(slint::VecModel::from(
                    titles
                        .into_iter()
                        .map(slint::SharedString::from)
                        .collect::<Vec<_>>(),
                )));
                slint_window.set_current_tab(selected);
                slint_window.set_page_text(text.into());
            };
            slint_window.set_title_text(window.get_title().unwrap_or_default().into());
            refresh(&window, &slint_window);
            let pressed = (Rc::downgrade(&window), slint_window.as_weak());
            slint_window.on_tab_pressed(move |index| {
                let (Some(window), Some(slint_window)) = (pressed.0.upgrade(), pressed.1.upgrade())
                else {
                    return;
                };
                if let Some(tab_pane) = window
                    .get_content_pane()
                    .get_components()
                    .into_iter()
                    .find(|component| component.kind() == ComponentKind::TabbedPane)
                {
                    tab_pane.set_selected_tab(index);
                }
                refresh(&window, &slint_window);
            });
            let closing = Rc::downgrade(&window);
            slint_window.window().on_close_requested(move || {
                if let Some(window) = closing.upgrade() {
                    window.dispose();
                }
                slint::CloseRequestResponse::HideWindow
            });
            let _ = slint_window.show();
            tabbed_windows.push((window, slint_window));
        }
    });
}

fn update_page_windows() {
    update_tabbed_text_windows();
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

thread_local! {
    /// Where (window coordinates) the last right press asked for a context menu.
    static POPUP_AT: std::cell::Cell<(f32, f32)> = const { std::cell::Cell::new((0.0, 0.0)) };
}

/// The component a right press on the Slint widget `(kind, label)` lands on.
/// Kind "brd" is the showing panel whose titled border is `label`.  Kind ""
/// (the dialog's background, or an untitled group box) is the outermost
/// showing component with a mouse listener outside the process panel: the
/// Slint window does not know which nested Swing panel lies under the
/// pointer there, and the dialog's root panel is the one that covers it.
/// `jdk::dispatch_mouse_pressed` then walks up to the nearest listener, as
/// Swing's `LightweightDispatcher` picks the deepest component that has one.
fn context_target(kind: &str, label: &str) -> Option<Rc<JComponent>> {
    if kind == "brd" && !label.is_empty() {
        return components_showing()
            .into_iter()
            .find(|component| component.get_border_title().as_deref() == Some(label));
    }
    if !kind.is_empty() && kind != "brd" {
        return find(kind, label);
    }
    fn in_process_panel(component: &Rc<JComponent>) -> bool {
        // A `ProcessControlPanel`: its run button is a toggle button with a
        // mouse listener.
        (component.kind() == ComponentKind::ToggleButton && component.has_mouse_listeners())
            || component.get_components().iter().any(in_process_panel)
    }
    components_showing().into_iter().find(|component| {
        component.has_mouse_listeners() && !in_process_panel(component) && {
            let mut ancestor = component.get_parent();
            let mut outermost = true;
            while let Some(parent) = ancestor {
                if parent.has_mouse_listeners() {
                    outermost = false;
                    break;
                }
                ancestor = parent.get_parent();
            }
            outermost
        }
    })
}

/// The showing `JPopupMenu` as `EtomoBridge.popup-*`.
fn update_popup<T>(window: &T, scope: Scope)
where
    T: ComponentHandle + 'static,
    for<'a> EtomoBridge<'a>: slint::Global<'a, T>,
{
    let bridge = window.global::<EtomoBridge>();
    let showing = crate::imod::etomo::jdk::showing_popup_menu()
        .filter(|_| POPUP_SCOPE.with(|popup_scope| popup_scope.get()) == scope);
    let Some((menu, _, _)) = showing else {
        if bridge.get_popup_shown() {
            bridge.set_popup_shown(false);
        }
        return;
    };
    let items: Vec<BridgedPopupItem> = menu
        .get_components()
        .into_iter()
        .map(|item| BridgedPopupItem {
            text: strip_tags(&item.get_text()).into(),
            enabled: item.is_enabled(),
            separator: item.kind() == ComponentKind::Separator,
        })
        .collect();
    let (x, y) = POPUP_AT.with(|at| at.get());
    bridge.set_popup_x(x);
    bridge.set_popup_y(y);
    // Replaced only when the items change: a new model rebuilds the drawn
    // items, which would lose a press in progress.
    let current = bridge.get_popup_items();
    let unchanged = bridge.get_popup_shown()
        && current.row_count() == items.len()
        && items
            .iter()
            .enumerate()
            .all(|(index, item)| current.row_data(index).as_ref() == Some(item));
    if !unchanged {
        bridge.set_popup_items(slint::ModelRc::new(slint::VecModel::from(items)));
    }
    bridge.set_popup_shown(true);
}

/// Wires `window`'s `EtomoBridge` to the component tree.  Called by
/// `UIHarness.createMainFrame` on the event dispatch thread.
thread_local! {
    /// The Slint window drawing the visible `SubFrame` (axis B when both axes
    /// are shown).
    static SUB_FRAME_WINDOW: RefCell<Option<SubFrameWindow>> = const { RefCell::new(None) };
}

/// Shows the `SubFrame` window while the Swing `SubFrame` is visible (Java's
/// "Both" axis view), with its title, status bar and axis B's current dialog,
/// next to the main window (`SubFrame.moveSubFrame`); hides it otherwise.
/// Closing it is the window system's `WINDOW_CLOSING`
/// (`SubFrame.processWindowEvent`: `mainFrame.showAxisA()`).
fn update_sub_frame_window(main_window: &MainFrameWindow) {
    let sub_frame = visible_sub_frame();
    SUB_FRAME_WINDOW.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(sub_frame) = sub_frame else {
            if let Some(window) = slot.as_ref() {
                let _ = window.hide();
            }
            return;
        };
        if slot.is_none() {
            let Ok(window) = SubFrameWindow::new() else {
                return;
            };
            install_bridge(&window, Scope::Sub);
            window.on_parallel_row_toggled(|row| {
                let _scope = ScopeGuard::enter(Scope::Sub);
                if let Some(check_box) = find_all("cb", "Computer")
                    .into_iter()
                    .nth(row.max(0) as usize)
                    && check_box.is_enabled()
                {
                    check_box.do_click();
                }
            });
            window.window().on_close_requested(|| {
                if let Some(sub_frame) = crate::imod::etomo::ui::swing::etomo_frame::sub_frame() {
                    sub_frame.process_window_event(
                        crate::imod::etomo::ui::swing::abstract_frame::WINDOW_CLOSING,
                    );
                }
                slint::CloseRequestResponse::HideWindow
            });
            *slot = Some(window);
        }
        let window = slot.as_ref().unwrap();
        window.set_title_text(sub_frame.get_title().into());
        window.set_peet_available(main_window.get_peet_available());
        let manager: Option<&'static dyn BaseManager> =
            etomo_director::INSTANCE.get_current_manager();
        if let Some(manager) = manager {
            if let Some(main_panel) = manager.get_main_panel() {
                window.set_status_bar_text(main_panel.main_panel().get_status_bar_text().into());
            }
            let dialog_type = manager.get_current_dialog_type(Some(AxisID::Second));
            window.set_dialog(dialog_type.map(dialog_name).unwrap_or("").into());
            let _scope = ScopeGuard::enter(Scope::Sub);
            let parallel_status_showing = manager.get_main_panel().is_some_and(|main_panel| {
                main_panel
                    .main_panel()
                    .get_parallel_status_panel(AxisID::Second)
                    .is_some_and(|panel| panel.is_showing())
            });
            window.set_show_parallel_panel(parallel_status_showing);
            if parallel_status_showing {
                window.set_parallel_rows(parallel_rows_model());
            }
        }
        let bridge = window.global::<EtomoBridge>();
        bridge.set_generation(bridge.get_generation().wrapping_add(1));
        update_popup(window, Scope::Sub);
        if !window.window().is_visible() {
            // `moveSubFrame`: to the right of the main frame.
            let main_position = main_window.window().position();
            let main_size = main_window.window().size();
            window.window().set_position(slint::PhysicalPosition::new(
                main_position.x + main_size.width as i32,
                main_position.y,
            ));
            let _ = window.show();
        }
    });
}

/// The showing `JFileChooser` the Java added to a panel (`CleanupPanel`).
fn embedded_chooser() -> Option<Rc<FileChooser>> {
    components_showing()
        .iter()
        .filter(|component| component.kind() == ComponentKind::Other)
        .find_map(file_chooser::chooser_for_component)
}

/// The embedded chooser's list: (name, is directory) pairs for its current
/// directory under its current filter.
fn embedded_chooser_entries(chooser: &FileChooser) -> Vec<(String, bool)> {
    let dir = chooser.get_current_directory().unwrap_or_default();
    file_chooser::list_entries(chooser, &dir, chooser.get_file_filter().as_ref())
}

/// Which of the embedded chooser's entries are selected
/// (`getSelectedFiles`, or `getSelectedFile` when multi-selection is off).
fn embedded_chooser_selection(chooser: &FileChooser) -> Vec<bool> {
    let dir = chooser.get_current_directory().unwrap_or_default();
    let selected: Vec<std::path::PathBuf> = if chooser.is_multi_selection_enabled() {
        chooser.get_selected_files()
    } else {
        chooser.get_selected_file().into_iter().collect()
    };
    embedded_chooser_entries(chooser)
        .iter()
        .map(|(name, _)| selected.contains(&dir.join(name)))
        .collect()
}

/// Wires one window's `EtomoBridge` global to the component tree: every
/// callback resolves its `(kind, label)` pairs in `scope` (the main window's
/// frames, or the `SubFrame`'s), as each Swing frame has its own components.
fn install_bridge<T>(window: &T, scope: Scope)
where
    T: ComponentHandle + 'static,
    for<'a> EtomoBridge<'a>: slint::Global<'a, T>,
{
    let bridge = window.global::<EtomoBridge>();
    bridge.on_activated(move |kind, label| {
        let _scope = ScopeGuard::enter(scope);
        {
            if let Some(component) = find(&kind, &label) {
                if kind.as_str() == "tbi" {
                    // Choosing a tab: an enabled one is selected, as Swing does.
                    if let Some(index) = tab_index(&component, &label)
                        && component.is_enabled_at(index)
                    {
                        component.set_selected_tab(index as i32);
                    }
                } else if component.is_enabled() {
                    component.do_click();
                }
            }
        }
    });
    bridge.on_table_rows(move |table, _generation| {
        let _scope = ScopeGuard::enter(scope);
        table_model(&table)
    });
    bridge.on_table_cell_activated(move |table, index| {
        let _scope = ScopeGuard::enter(scope);
        {
            if let Some(cell) = table_cell(&table, index)
                && cell.is_enabled()
            {
                cell.do_click();
            }
        }
    });
    bridge.on_table_cell_toggled(move |table, index, value| {
        let _scope = ScopeGuard::enter(scope);
        {
            if let Some(cell) = table_cell(&table, index)
                && cell.is_enabled()
                && cell.is_selected() != value
            {
                cell.do_click();
            }
        }
    });
    bridge.on_table_cell_edited(move |table, index, text| {
        let _scope = ScopeGuard::enter(scope);
        {
            let Some(cell) = table_cell(&table, index) else {
                return;
            };
            if cell.kind() == ComponentKind::Spinner {
                if let Ok(value) = text.trim().parse::<f64>() {
                    cell.set_spinner_value(value);
                }
            } else {
                cell.set_text(&text);
            }
        }
    });
    bridge.on_toggled(move |kind, label, value| {
        let _scope = ScopeGuard::enter(scope);
        {
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
        }
    });
    bridge.on_edited(move |kind, label, text| {
        let _scope = ScopeGuard::enter(scope);
        {
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
            } else if kind.as_str() == "cbb" {
                // An editable `JComboBox`'s editor commits its text with
                // `setSelectedItem` (`BasicComboBoxUI`: Enter or focus lost).
                if component.is_editable() && component.is_enabled() {
                    component.set_selected_item(Some(&text));
                }
            } else {
                component.set_text(&text);
            }
        }
    });
    bridge.on_focus_changed(move |kind, label, gained| {
        let _scope = ScopeGuard::enter(scope);
        {
            if let Some(component) = find(&kind, &label) {
                component.fire_focus_changed(gained);
            }
        }
    });
    bridge.on_items(move |kind, label, _generation| {
        let _scope = ScopeGuard::enter(scope);
        {
            let items: Vec<slint::SharedString> = match find(&kind, &label) {
                None => Vec::new(),
                Some(component) => (0..component.get_item_count())
                    .map(|index| component.get_item_at(index).unwrap_or_default().into())
                    .collect(),
            };
            slint::ModelRc::new(slint::VecModel::from(items))
        }
    });
    bridge.on_selected(move |kind, label, index| {
        let _scope = ScopeGuard::enter(scope);
        {
            if let Some(component) = find(&kind, &label)
                && component.is_enabled()
            {
                component.set_selected_index(index);
            }
        }
    });
    bridge.on_text_value(move |kind, label, default, _generation| {
        let _scope = ScopeGuard::enter(scope);
        match find(&kind, &label) {
            None => default,
            Some(component) => match kind.as_str() {
                "pcp" => process_state(&component).to_string().into(),
                "tbi" => component.get_selected_tab().to_string().into(),
                "cbb" => component.get_selected_item().unwrap_or_default().into(),
                // A `JList`'s selected index (-1: none).
                "lsx" => component.get_selected_index().to_string().into(),
                // A `PanelHeader` `ExpandButton`'s symbol ("<html>&lt" is "<").
                "mbo" | "mba" | "mbm" => strip_tags(&component.get_text()).into(),
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
        }
    });
    bridge.on_bool_value(move |kind, label, default, _generation| {
        let _scope = ScopeGuard::enter(scope);
        match find(&kind, &label) {
            None => default,
            Some(component) => component.is_selected(),
        }
    });
    bridge.on_visible_value(move |kind, label, _default, _generation| {
        let _scope = ScopeGuard::enter(scope);
        visible_value(&kind, &label)
    });
    bridge.on_enabled_value(move |kind, label, default, _generation| {
        let _scope = ScopeGuard::enter(scope);
        match find(&kind, &label) {
            None => default,
            Some(component) if kind.as_str() == "tbi" => {
                tab_index(&component, &label).is_some_and(|index| component.is_enabled_at(index))
            }
            Some(component) => component.is_enabled(),
        }
    });
    if std::env::var_os("ETOMO_BRIDGE_DEBUG").is_some() {
        bridge.on_activated(move |kind, label| {
            let _scope = ScopeGuard::enter(scope);
            {
                eprintln!("slint_bridge: activated ({kind}, {label:?})");
                for (name, component) in showing_components() {
                    eprintln!("slint_bridge:   showing {name} {:?}", component.kind());
                }
                if let Some(component) = find(&kind, &label)
                    && component.is_enabled()
                {
                    component.do_click();
                }
            }
        });
    }
    // A right mouse press: Swing's mouse dispatch to the pressed component (see
    // `context_target`), which opens a `ContextPopup` or `Run3dmodMenu`.
    bridge.on_context_requested(move |kind, label, x, y| {
        let _scope = ScopeGuard::enter(scope);
        {
            let Some(component) = context_target(&kind, &label) else {
                return;
            };
            POPUP_AT.with(|at| at.set((x, y)));
            POPUP_SCOPE.with(|popup_scope| popup_scope.set(scope));
            crate::imod::etomo::jdk::dispatch_mouse_pressed(
                &component,
                &crate::imod::etomo::jdk::MouseEvent {
                    button: 3,
                    popup_trigger: true,
                    x: x as i32,
                    y: y as i32,
                },
            );
        }
    });
    // Choosing a popup menu item: `BasicMenuItemUI` clears the menu selection
    // path (hiding the popup) and then clicks the item.
    let popup_window = window.as_weak();
    bridge.on_popup_item_chosen(move |index| {
        let _scope = ScopeGuard::enter(scope);
        {
            let Some((menu, _, _)) = crate::imod::etomo::jdk::showing_popup_menu() else {
                return;
            };
            let item = menu.get_components().into_iter().nth(index.max(0) as usize);
            menu.set_visible(false);
            // Hidden at once: the item's action may block this thread (a modal
            // popup) before the next refresh.
            if let Some(window) = popup_window.upgrade() {
                window.global::<EtomoBridge>().set_popup_shown(false);
            }
            if let Some(item) = item
                && item.kind() != ComponentKind::Separator
                && item.is_enabled()
            {
                // Clicked a moment later, so the window repaints without the menu
                // first (Swing repaints inside the modal loop an item's action may
                // enter; this event loop cannot be nested).
                slint::Timer::single_shot(std::time::Duration::from_millis(50), move || {
                    item.do_click();
                });
            }
        }
    });
    bridge.on_popup_cancelled(move || {
        let _scope = ScopeGuard::enter(scope);
        {
            if let Some((menu, _, _)) = crate::imod::etomo::jdk::showing_popup_menu() {
                menu.set_visible(false);
            }
        }
    });
    bridge.on_chooser_directory(move |_generation| {
        let _scope = ScopeGuard::enter(scope);
        embedded_chooser()
            .and_then(|chooser| chooser.get_current_directory())
            .map(|dir| dir.to_string_lossy().into_owned())
            .unwrap_or_default()
            .into()
    });
    bridge.on_chooser_entries(move |_generation| {
        let _scope = ScopeGuard::enter(scope);
        let entries: Vec<slint::SharedString> = embedded_chooser()
            .map(|chooser| embedded_chooser_entries(&chooser))
            .unwrap_or_default()
            .into_iter()
            .map(|(name, is_dir)| if is_dir { format!("{name}/") } else { name }.into())
            .collect();
        slint::ModelRc::new(slint::VecModel::from(entries))
    });
    bridge.on_chooser_selected(move |_generation| {
        let _scope = ScopeGuard::enter(scope);
        let selected = embedded_chooser()
            .map(|chooser| embedded_chooser_selection(&chooser))
            .unwrap_or_default();
        slint::ModelRc::new(slint::VecModel::from(selected))
    });
    bridge.on_chooser_filters(move |_generation| {
        let _scope = ScopeGuard::enter(scope);
        let mut filters: Vec<slint::SharedString> = vec!["All Files".into()];
        if let Some(chooser) = embedded_chooser() {
            filters.extend(
                chooser
                    .get_choosable_file_filters()
                    .iter()
                    .map(|filter| filter.get_description().unwrap_or_default().into()),
            );
        }
        slint::ModelRc::new(slint::VecModel::from(filters))
    });
    bridge.on_chooser_current_filter(move |_generation| {
        let _scope = ScopeGuard::enter(scope);
        let Some(chooser) = embedded_chooser() else {
            return 0;
        };
        let current = chooser.get_file_filter();
        current
            .and_then(|current| {
                chooser
                    .get_choosable_file_filters()
                    .iter()
                    .position(|filter| Rc::ptr_eq(filter, &current))
            })
            .map_or(0, |index| index as i32 + 1)
    });
    bridge.on_chooser_file_name(move |_generation| {
        let _scope = ScopeGuard::enter(scope);
        let Some(chooser) = embedded_chooser() else {
            return "".into();
        };
        let names: Vec<String> = embedded_chooser_entries(&chooser)
            .into_iter()
            .zip(embedded_chooser_selection(&chooser))
            .filter(|(_, selected)| *selected)
            .map(|((name, _), _)| name)
            .collect();
        if names.len() > 1 {
            names
                .iter()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(" ")
                .into()
        } else {
            names.first().cloned().unwrap_or_default().into()
        }
    });
    // A click on a list entry: the list's selection model, which the chooser's
    // `setSelectedFiles` / `setSelectedFile` follow; directories are not
    // selectable in a files-only chooser.
    bridge.on_chooser_entry_clicked(move |index, control| {
        let _scope = ScopeGuard::enter(scope);
        let Some(chooser) = embedded_chooser() else {
            return;
        };
        let entries = embedded_chooser_entries(&chooser);
        let Some((name, is_dir)) = entries.get(index.max(0) as usize).cloned() else {
            return;
        };
        if is_dir && chooser.get_file_selection_mode() == file_chooser::FILES_ONLY {
            return;
        }
        let dir = chooser.get_current_directory().unwrap_or_default();
        let path = dir.join(&name);
        if chooser.is_multi_selection_enabled() {
            let mut selected = chooser.get_selected_files();
            if control {
                if let Some(position) = selected.iter().position(|file| *file == path) {
                    selected.remove(position);
                } else {
                    selected.push(path);
                }
            } else {
                selected = vec![path];
            }
            chooser.set_selected_files(&selected);
        } else {
            chooser.set_selected_file(Some(&path));
        }
    });
    // A double click on a directory enters it (`setCurrentDirectory`).
    bridge.on_chooser_entry_entered(move |index| {
        let _scope = ScopeGuard::enter(scope);
        let Some(chooser) = embedded_chooser() else {
            return;
        };
        let entries = embedded_chooser_entries(&chooser);
        if let Some((name, true)) = entries.get(index.max(0) as usize).cloned() {
            let dir = chooser
                .get_current_directory()
                .unwrap_or_default()
                .join(name);
            chooser.set_current_directory(Some(&dir));
            chooser.set_selected_files(&[]);
        }
    });
    bridge.on_chooser_filter_chosen(move |index| {
        let _scope = ScopeGuard::enter(scope);
        let Some(chooser) = embedded_chooser() else {
            return;
        };
        // 0 is "All Files" (the look and feel's accept-all filter).
        let filter = (index as usize)
            .checked_sub(1)
            .and_then(|index| chooser.get_choosable_file_filters().get(index).cloned());
        chooser.set_file_filter(filter);
    });
    bridge.on_chooser_up(move || {
        let _scope = ScopeGuard::enter(scope);
        if let Some(chooser) = embedded_chooser() {
            let parent = chooser
                .get_current_directory()
                .and_then(|dir| dir.parent().map(|parent| parent.to_path_buf()));
            if let Some(parent) = parent {
                chooser.set_current_directory(Some(&parent));
            }
        }
    });
    bridge.set_wired(true);
}

pub fn install(window: &MainFrameWindow) -> slint::Timer {
    install_bridge(window, Scope::Main);
    // A click on a `ProcessorTable` row's computer check box.
    window.on_parallel_row_toggled(|row| {
        if let Some(check_box) = find_all("cb", "Computer")
            .into_iter()
            .nth(row.max(0) as usize)
            && check_box.is_enabled()
        {
            check_box.do_click();
        }
    });
    // The About box's OK button.
    window.on_about_ok(|| {
        if std::env::var_os("ETOMO_BRIDGE_DEBUG").is_some() {
            eprintln!("slint_bridge: about-ok");
        }
        if let Some(dialog) = crate::imod::etomo::jdk::showing_dialogs()
            .into_iter()
            .rev()
            .find(|dialog| dialog.get_title() == "About")
        {
            fn ok_button(node: &Rc<JComponent>) -> Option<Rc<JComponent>> {
                if node.kind() == ComponentKind::Button && node.get_text() == "OK" {
                    return Some(node.clone());
                }
                node.get_components().iter().find_map(ok_button)
            }
            if let Some(button) = ok_button(&dialog.get_content_pane()) {
                button.do_click();
            }
        }
    });
    // The close button of a showing modal `JDialog` (a startup dialog): the
    // window system's `WINDOW_CLOSING` to the newest one.
    window.on_startup_dialog_closing(|| {
        if let Some(dialog) = crate::imod::etomo::jdk::showing_dialogs().pop() {
            dialog.process_window_closing();
        }
    });
    // The close button of the showing `ManagerFrame` (the directive editor):
    // the window system's `WINDOW_CLOSING` (`ManagerFrame.processWindowEvent`).
    window.on_manager_frame_closing(|| {
        if let Some((_, frame)) = ui_harness::with(|harness| harness.get_manager_frames())
            .into_iter()
            .rev()
            .find(|(_, frame)| frame.is_visible())
        {
            frame.process_window_event(
                crate::imod::etomo::ui::swing::abstract_frame::WINDOW_CLOSING,
            );
        }
    });
    // `JFileChooser.showDialog`: the Slint chooser, unless a driver or test
    // already answers choosers.
    if !file_chooser::has_dialog_responder() {
        file_chooser::set_dialog_responder(Some(Rc::new(|chooser, _parent| {
            present_file_chooser(chooser);
        })));
    }
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
                run_debug_commands(&window);
                update_view(&window);
                update_popup(&window, Scope::Main);
                update_sub_frame_window(&window);
                update_page_windows();
            }
        },
    );
    timer
}

/// `ETOMO_BRIDGE_COMMANDS=<file>` (verification only): each refresh runs and
/// deletes the file's lines, so a check script can reach a widget by name
/// instead of by screen position.  Each line is delivered through the
/// window's own `EtomoBridge` callbacks, exactly as the Slint widget would
/// deliver it:
/// - `<kind>\t<label>` or `click\t<kind>\t<label>`: the widget's click
///   (`activated`; kind "tbi" chooses a tab);
/// - `edit\t<kind>\t<label>\t<text>`: focus gained, the text typed
///   (`edited`), focus lost;
/// - `toggle\t<kind>\t<label>\t<0|1>`: a check box or radio button click
///   (`toggled`);
/// - `select\t<kind>\t<label>\t<index>`: a combo box choice (`selected`).
///
/// An operation written `B.<op>` goes to the `SubFrame` window (axis B when
/// both axes are shown).  `ETOMO_BRIDGE_STATUS=<file>` additionally rewrites
/// that file on every refresh with the busy state the click drivers wait on
/// (`lb.busy` or `bn.kill-process` enabled in either window) and the progress
/// texts.
fn run_debug_commands(main_window: &MainFrameWindow) {
    if let Some(status_path) = std::env::var_os("ETOMO_BRIDGE_STATUS") {
        let mut busy = false;
        let mut progress = Vec::new();
        for scope in [Scope::Main, Scope::Sub] {
            let _scope = ScopeGuard::enter(scope);
            busy |= find_all("raw", "lb.busy").iter().any(|c| c.is_enabled())
                || find_all("raw", "bn.kill-process")
                    .iter()
                    .any(|c| c.is_enabled());
            for name in ["the-progress-bar-label", "the-progress-bar"] {
                for component in find_all("raw", name) {
                    progress.push(component.get_text());
                }
            }
        }
        let text = format!("busy={}\nprogress={}\n", busy as i32, progress.join(" | "));
        let temporary = std::path::PathBuf::from(format!("{}.tmp", status_path.to_string_lossy()));
        if std::fs::write(&temporary, text).is_ok() {
            let _ = std::fs::rename(&temporary, &status_path);
        }
    }
    let Some(path) = std::env::var_os("ETOMO_BRIDGE_COMMANDS") else {
        return;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    let _ = std::fs::remove_file(&path);
    for line in text.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        let (operation, arguments) = match fields.as_slice() {
            [kind, label] => ("click", vec![*kind, *label]),
            [operation, rest @ ..] => (*operation, rest.to_vec()),
            _ => continue,
        };
        let (sub, operation) = match operation.strip_prefix("B.") {
            Some(operation) => (true, operation),
            None => (false, operation),
        };
        let deliver = |bridge: EtomoBridge| match (operation, arguments.as_slice()) {
            ("click", [kind, label]) => bridge.invoke_activated((*kind).into(), (*label).into()),
            ("edit", [kind, label, text]) => {
                bridge.invoke_focus_changed((*kind).into(), (*label).into(), true);
                bridge.invoke_edited((*kind).into(), (*label).into(), (*text).into());
                bridge.invoke_focus_changed((*kind).into(), (*label).into(), false);
            }
            ("toggle", [kind, label, value]) => {
                bridge.invoke_toggled((*kind).into(), (*label).into(), *value == "1")
            }
            ("select", [kind, label, index]) => {
                if let Ok(index) = index.parse::<i32>() {
                    bridge.invoke_selected((*kind).into(), (*label).into(), index);
                }
            }
            _ => eprintln!("slint_bridge: bad command line {line:?}"),
        };
        if sub {
            let window = SUB_FRAME_WINDOW.with(|slot| slot.borrow().as_ref().map(|w| w.as_weak()));
            if let Some(window) = window.and_then(|weak| weak.upgrade()) {
                deliver(window.global::<EtomoBridge>());
            } else {
                eprintln!("slint_bridge: no SubFrame window for {line:?}");
            }
        } else {
            deliver(main_window.global::<EtomoBridge>());
        }
    }
}

/// Shows a `JOptionPane` popup as a modal dialog and returns the answer, or
/// `None` when the dialog could not be shown.  The dialog is drawn by a child
/// process of this binary (`imod etomo-popup`, [`etomo_popup`]), so the main
/// window's event dispatch thread waits for the answer as Swing's modal
/// `dialog.setVisible(true)` does.
pub fn present_popup(request: &PopupRequest) -> Option<PopupAnswer> {
    use std::io::Write;
    // Verification only: with `ETOMO_BRIDGE_STATUS=<file>`, every popup is
    // also appended to `<file>.popups` (title, options, message), as the
    // click drivers' popups.log records it.
    if let Some(status_path) = std::env::var_os("ETOMO_BRIDGE_STATUS")
        && let Ok(mut log) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}.popups", status_path.to_string_lossy()))
    {
        let _ = writeln!(
            log,
            "POPUP title={:?} options={:?}\n{}\n",
            request.title.as_deref().unwrap_or(""),
            request.options,
            request.message.join("\n")
        );
    }
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
    // `BasicOptionPaneUI.addMessageComponents`: each line of the message is
    // one `JLabel`; a label whose text starts with "<html>" renders it as
    // Swing's basic HTML (`BasicHTML.isHTMLString`).
    let lines: Vec<HtmlLine> = message.lines().flat_map(swing_label_lines).collect();
    window.set_title_text(title.into());
    window.set_message_type(message_type);
    window.set_initial(initial);
    window.set_message_html(slint::ModelRc::new(slint::VecModel::from(lines)));
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

/// One `JLabel` of a `JOptionPane` message as the lines it draws: a plain
/// label is one line of its text as it is; a label whose text starts with
/// "<html>" (`BasicHTML.isHTMLString`, any case) is rendered as Swing's basic
/// HTML ([`swing_html_lines`]).
fn swing_label_lines(text: &str) -> Vec<HtmlLine> {
    if text.len() >= 6 && text.is_char_boundary(6) && text[..6].eq_ignore_ascii_case("<html>") {
        return swing_html_lines(text);
    }
    vec![HtmlLine {
        runs: slint::ModelRc::new(slint::VecModel::from(vec![html_run(
            text,
            &HtmlStyle::default(),
        )])),
        center: false,
        indent: 0.0,
        gap: 0.0,
    }]
}

/// The character style in effect at a point of an HTML text.
#[derive(Clone, Default)]
struct HtmlStyle {
    bold: bool,
    italic: bool,
    underline: bool,
    color: Option<slint::Color>,
}

fn html_run(text: &str, style: &HtmlStyle) -> HtmlRun {
    HtmlRun {
        text: text.into(),
        bold: style.bold,
        italic: style.italic,
        underline: style.underline,
        colored: style.color.is_some(),
        color: style.color.unwrap_or_default(),
    }
}

/// An HTML color value as Swing's CSS parser reads it: "#rrggbb", "rgb(r, g,
/// b)" or one of the HTML 3.2 color names.
fn html_color(value: &str) -> Option<slint::Color> {
    let value = value.trim().trim_matches(|c| c == '"' || c == '\'');
    if let Some(hex) = value.strip_prefix('#') {
        let rgb = u32::from_str_radix(hex, 16).ok()?;
        return Some(slint::Color::from_rgb_u8(
            (rgb >> 16) as u8,
            (rgb >> 8) as u8,
            rgb as u8,
        ));
    }
    let lower = value.to_ascii_lowercase();
    if let Some(inner) = lower
        .strip_prefix("rgb(")
        .and_then(|rest| rest.strip_suffix(')'))
    {
        let parts: Vec<u8> = inner
            .split(',')
            .filter_map(|part| part.trim().parse().ok())
            .collect();
        if let [r, g, b] = parts.as_slice() {
            return Some(slint::Color::from_rgb_u8(*r, *g, *b));
        }
        return None;
    }
    let rgb = match lower.as_str() {
        "black" => 0x000000,
        "silver" => 0xc0c0c0,
        "gray" | "grey" => 0x808080,
        "white" => 0xffffff,
        "maroon" => 0x800000,
        "red" => 0xff0000,
        "purple" => 0x800080,
        "fuchsia" => 0xff00ff,
        "green" => 0x008000,
        "lime" => 0x00ff00,
        "olive" => 0x808000,
        "yellow" => 0xffff00,
        "navy" => 0x000080,
        "blue" => 0x0000ff,
        "teal" => 0x008080,
        "aqua" => 0x00ffff,
        "orange" => 0xffa500,
        _ => return None,
    };
    Some(slint::Color::from_rgb_u8(
        (rgb >> 16) as u8,
        (rgb >> 8) as u8,
        rgb as u8,
    ))
}

/// The value of attribute `name` in a tag's attribute text (`color=red`,
/// `align="center"`), if present.
fn html_attribute(attributes: &str, name: &str) -> Option<String> {
    let lower = attributes.to_ascii_lowercase();
    let mut start = 0;
    while let Some(found) = lower[start..].find(name) {
        let index = start + found;
        start = index + name.len();
        let before_ok = index == 0 || !lower.as_bytes()[index - 1].is_ascii_alphanumeric();
        let rest = lower[start..].trim_start();
        if !before_ok || !rest.starts_with('=') {
            continue;
        }
        let offset = attributes.len() - rest.len() + 1;
        let value = attributes[offset..].trim_start();
        let value = if let Some(quoted) = value.strip_prefix('"') {
            quoted.split('"').next().unwrap_or("")
        } else if let Some(quoted) = value.strip_prefix('\'') {
            quoted.split('\'').next().unwrap_or("")
        } else {
            value.split(char::is_whitespace).next().unwrap_or("")
        };
        return Some(value.to_owned());
    }
    None
}

/// Renders an HTML label text the way Swing's basic HTML support
/// (`javax.swing.text.html`, the subset a `JLabel` uses) lays it out, for the
/// tags eTomo's messages and buttons use: `<b>`/`<strong>`, `<i>`/`<em>`,
/// `<u>`, `<font color>`, a `style` attribute's `color` / `font-weight`,
/// `<br>`, `<p>` and `<div>` (a paragraph gap; `align=center`), `<center>`,
/// `<ul>`/`<ol>` with `<li>` (indented bullets / numbers), `<h1>`..`<h6>` (bold),
/// `<hr>`, entities (`&lt;` `&gt;` `&amp;` `&quot;` `&nbsp;` `&#N;`, also
/// without the `;` as Swing's parser accepts them), whitespace collapsed to
/// single spaces.  Unknown tags are ignored, as Swing ignores them.
fn swing_html_lines(text: &str) -> Vec<HtmlLine> {
    struct Builder {
        lines: Vec<HtmlLine>,
        runs: Vec<HtmlRun>,
        center: bool,
        indent: f32,
        gap: f32,
        pending_space: bool,
    }
    impl Builder {
        fn push_text(&mut self, text: &str, style: &HtmlStyle) {
            for (index, word) in text.split(' ').enumerate() {
                if index > 0 {
                    self.pending_space = true;
                }
                if word.is_empty() {
                    continue;
                }
                let mut piece = String::new();
                if self.pending_space && !self.runs.is_empty() {
                    piece.push(' ');
                }
                self.pending_space = false;
                piece.push_str(word);
                if let Some(last) = self.runs.last_mut()
                    && last.bold == style.bold
                    && last.italic == style.italic
                    && last.underline == style.underline
                    && last.colored == style.color.is_some()
                    && (!last.colored || Some(last.color) == style.color)
                {
                    last.text = format!("{}{}", last.text, piece).into();
                } else {
                    self.runs.push(html_run(&piece, style));
                }
            }
        }
        fn break_line(&mut self, force: bool) {
            if self.runs.is_empty() && !force {
                return;
            }
            let runs = std::mem::take(&mut self.runs);
            self.lines.push(HtmlLine {
                runs: slint::ModelRc::new(slint::VecModel::from(runs)),
                center: self.center,
                indent: self.indent,
                gap: self.gap,
            });
            self.gap = 0.0;
            self.pending_space = false;
        }
    }
    let mut builder = Builder {
        lines: Vec::new(),
        runs: Vec::new(),
        center: false,
        indent: 0.0,
        gap: 0.0,
        pending_space: false,
    };
    // Open inline elements, each with the style it started from.
    let mut styles: Vec<(String, HtmlStyle)> = Vec::new();
    let mut style = HtmlStyle::default();
    // Open lists: (ordered, next number).
    let mut lists: Vec<(bool, i32)> = Vec::new();
    let mut centers: Vec<bool> = Vec::new();
    let mut paragraphs = 0;
    let mut chars = text.char_indices().peekable();
    let mut pending = String::new();
    let flush = |pending: &mut String, builder: &mut Builder, style: &HtmlStyle| {
        if !pending.is_empty() {
            builder.push_text(pending, style);
            pending.clear();
        }
    };
    while let Some((index, c)) = chars.next() {
        match c {
            '<' => {
                let Some(end) = text[index..].find('>') else {
                    pending.push(c);
                    continue;
                };
                let tag = &text[index + 1..index + end];
                while chars.peek().is_some_and(|(next, _)| *next <= index + end) {
                    chars.next();
                }
                if tag.starts_with("!--") {
                    continue;
                }
                flush(&mut pending, &mut builder, &style);
                let closing = tag.starts_with('/');
                let tag_body = tag.trim_start_matches('/').trim_end_matches('/');
                let (name, attributes) = match tag_body.find(char::is_whitespace) {
                    Some(split) => (&tag_body[..split], &tag_body[split..]),
                    None => (tag_body, ""),
                };
                let name = name.to_ascii_lowercase();
                match (name.as_str(), closing) {
                    ("br", _) => builder.break_line(true),
                    ("hr", _) => {
                        builder.break_line(false);
                        builder.break_line(true);
                    }
                    (
                        "p" | "div" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "center" | "pre",
                        false,
                    ) => {
                        builder.break_line(false);
                        if name == "p" && (paragraphs > 0 || !builder.lines.is_empty()) {
                            // `p { margin-top: 15 }` in Swing's default style
                            // sheet, scaled down to the label's font.
                            builder.gap = 8.0;
                        }
                        if name == "p" {
                            paragraphs += 1;
                        }
                        let center = name == "center"
                            || html_attribute(attributes, "align")
                                .is_some_and(|align| align.eq_ignore_ascii_case("center"));
                        centers.push(builder.center);
                        builder.center = builder.center || center;
                        styles.push((name.clone(), style.clone()));
                        if name.starts_with('h') {
                            style.bold = true;
                        }
                        if let Some(css) = html_attribute(attributes, "style") {
                            for declaration in css.split(';') {
                                let mut parts = declaration.splitn(2, ':');
                                let property =
                                    parts.next().unwrap_or("").trim().to_ascii_lowercase();
                                let value = parts.next().unwrap_or("").trim();
                                match property.as_str() {
                                    "color" => style.color = html_color(value),
                                    "font-weight" => {
                                        style.bold = value.eq_ignore_ascii_case("bold")
                                    }
                                    "font-style" => {
                                        style.italic = value.eq_ignore_ascii_case("italic")
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    (
                        "p" | "div" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "center" | "pre",
                        true,
                    ) => {
                        builder.break_line(false);
                        if let Some(center) = centers.pop() {
                            builder.center = center;
                        }
                        if let Some(position) = styles.iter().rposition(|(open, _)| *open == name) {
                            style = styles[position].1.clone();
                            styles.truncate(position);
                        }
                    }
                    ("ul" | "ol", false) => {
                        builder.break_line(false);
                        lists.push((name == "ol", 1));
                        builder.indent += 24.0;
                    }
                    ("ul" | "ol", true) => {
                        builder.break_line(false);
                        if lists.pop().is_some() {
                            builder.indent -= 24.0;
                        }
                    }
                    ("li", false) => {
                        builder.break_line(false);
                        let marker = match lists.last_mut() {
                            Some((true, number)) => {
                                *number += 1;
                                format!("{}. ", *number - 1)
                            }
                            _ => "\u{2022} ".to_owned(),
                        };
                        builder.runs.push(html_run(&marker, &style));
                    }
                    ("li", true) => builder.break_line(false),
                    ("b" | "strong" | "i" | "em" | "u" | "font" | "span" | "a", false) => {
                        styles.push((name.clone(), style.clone()));
                        match name.as_str() {
                            "b" | "strong" => style.bold = true,
                            "i" | "em" => style.italic = true,
                            "u" | "a" => style.underline = true,
                            _ => {}
                        }
                        if let Some(color) =
                            html_attribute(attributes, "color").and_then(|color| html_color(&color))
                        {
                            style.color = Some(color);
                        }
                        if let Some(css) = html_attribute(attributes, "style") {
                            for declaration in css.split(';') {
                                let mut parts = declaration.splitn(2, ':');
                                let property =
                                    parts.next().unwrap_or("").trim().to_ascii_lowercase();
                                let value = parts.next().unwrap_or("").trim();
                                match property.as_str() {
                                    "color" => style.color = html_color(value),
                                    "font-weight" => {
                                        style.bold = value.eq_ignore_ascii_case("bold")
                                    }
                                    "font-style" => {
                                        style.italic = value.eq_ignore_ascii_case("italic")
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    ("b" | "strong" | "i" | "em" | "u" | "font" | "span" | "a", true) => {
                        if let Some(position) = styles.iter().rposition(|(open, _)| *open == name) {
                            style = styles[position].1.clone();
                            styles.truncate(position);
                        }
                    }
                    _ => {}
                }
            }
            '&' => {
                // An entity: letters or `#` digits, optionally ended by ';'.
                let rest = &text[index + 1..];
                let length = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '#'))
                    .unwrap_or(rest.len());
                let name = &rest[..length];
                let decoded = match name {
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "amp" => Some('&'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    "nbsp" => Some('\u{a0}'),
                    _ => name.strip_prefix('#').and_then(|number| {
                        match number.strip_prefix(['x', 'X']) {
                            Some(hex) => u32::from_str_radix(hex, 16).ok(),
                            None => number.parse().ok(),
                        }
                        .and_then(char::from_u32)
                    }),
                };
                match decoded {
                    Some(decoded) => {
                        // A non-breaking space is not collapsed; draw it as a
                        // space.
                        pending.push(if decoded == '\u{a0}' {
                            '\u{2007}'
                        } else {
                            decoded
                        });
                        let mut skip = length;
                        if rest[length..].starts_with(';') {
                            skip += 1;
                        }
                        while chars.peek().is_some_and(|(next, _)| *next <= index + skip) {
                            chars.next();
                        }
                    }
                    None => pending.push('&'),
                }
            }
            c if c.is_whitespace() => {
                if !pending.ends_with(' ') {
                    pending.push(' ');
                }
            }
            c => pending.push(c),
        }
    }
    flush(&mut pending, &mut builder, &style);
    builder.break_line(false);
    if builder.lines.is_empty() {
        builder.break_line(true);
    }
    // Restore the non-breaking spaces drawn as figure spaces.
    for line in &builder.lines {
        for index in 0..line.runs.row_count() {
            if let Some(mut run) = line.runs.row_data(index)
                && run.text.contains('\u{2007}')
            {
                run.text = run.text.replace('\u{2007}', " ").into();
                line.runs.set_row_data(index, run);
            }
        }
    }
    builder.lines
}

/// Shows a `JFileChooser` as a modal dialog in a child process of this binary
/// (`imod etomo-filechooser`, [`etomo_filechooser`]) and answers the chooser
/// as its buttons do: the chosen file(s) are selected and approved; closing or
/// "Cancel" cancels.  The dialog is set up from the chooser's state as
/// `JFileChooser` uses it: the title (`getDialogTitle`, else the look and
/// feel's "Open"/"Save"), the approve button ("Open"/"Save" by dialog type),
/// the current directory, the selection mode, multi-selection, file hiding and
/// the "Files of Type" list (the look and feel's "All Files", then the
/// choosable filters, the current one selected).  While it shows, the child
/// asks for each directory's entries and this process answers with the ones
/// `JFileChooser` lists: those its file filter accepts (the translated
/// `FileFilter.accept`, or every file under "All Files"), hidden files left out
/// when file hiding is on, directories always traversable, files only when
/// file selection is enabled; directories first, then files, each sorted by
/// name (`BasicDirectoryModel`).
fn present_file_chooser(chooser: &FileChooser) {
    use std::io::{BufRead, Write};
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let directory = chooser
        .get_current_directory()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_default();
    let title = chooser.get_dialog_title().unwrap_or_else(|| {
        if chooser.get_dialog_type() == file_chooser::SAVE_DIALOG {
            "Save".to_owned()
        } else {
            file_chooser::DEFAULT_TITLE.to_owned()
        }
    });
    let approve_text = if chooser.get_dialog_type() == file_chooser::SAVE_DIALOG {
        "Save"
    } else {
        "Open"
    };
    // `getChoosableFileFilters()`: the accept-all filter first (eTomo never
    // turns it off), then the added ones; the current filter.
    let filters = chooser.get_choosable_file_filters();
    let current = chooser.get_file_filter();
    let current_index = current
        .as_ref()
        .and_then(|current| {
            filters
                .iter()
                .position(|filter| Rc::ptr_eq(filter, current))
        })
        .map_or(0, |index| index + 1);
    let mut command = std::process::Command::new(exe);
    command
        .arg("etomo-filechooser")
        .arg(&title)
        .arg(chooser.get_file_selection_mode().to_string())
        .arg(chooser.get_dialog_type().to_string())
        .arg(&directory)
        .arg(approve_text)
        .arg(if chooser.is_multi_selection_enabled() {
            "1"
        } else {
            "0"
        })
        .arg(current_index.to_string())
        .arg("All Files");
    for filter in &filters {
        command.arg(filter.get_description().unwrap_or_default());
    }
    let Ok(mut child) = command
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
    else {
        return;
    };
    let mut to_child = child.stdin.take();
    let Some(from_child) = child.stdout.take() else {
        let _ = child.kill();
        return;
    };
    let mut answer: Option<Vec<std::path::PathBuf>> = None;
    for line in std::io::BufReader::new(from_child).lines() {
        let Ok(line) = line else {
            break;
        };
        let mut fields = line.split('\t');
        match fields.next() {
            Some("LIST") => {
                let filter_index: usize = fields.next().and_then(|f| f.parse().ok()).unwrap_or(0);
                let dir = std::path::PathBuf::from(fields.next().unwrap_or(""));
                // `filter_index` 0 is "All Files" (`AcceptAllFileFilter`).
                let filter = filter_index
                    .checked_sub(1)
                    .and_then(|index| filters.get(index).cloned());
                let entries = file_chooser::list_entries(chooser, &dir, filter.as_ref());
                if let Some(to_child) = to_child.as_mut() {
                    let mut reply = String::new();
                    for (name, is_dir) in entries {
                        reply.push_str(&format!("{}\t{name}\n", if is_dir { "D" } else { "F" }));
                    }
                    reply.push_str("END\n");
                    let _ = to_child.write_all(reply.as_bytes());
                    let _ = to_child.flush();
                }
            }
            Some("APPROVE") => {
                answer = Some(fields.map(std::path::PathBuf::from).collect());
            }
            _ => {}
        }
    }
    drop(to_child);
    let _ = child.wait();
    match answer {
        Some(files) if !files.is_empty() => {
            if let Some(parent) = files[0].parent() {
                chooser.set_current_directory(Some(parent));
            }
            if chooser.is_multi_selection_enabled() {
                chooser.set_selected_files(&files);
            }
            chooser.set_selected_file(Some(&files[0]));
            chooser.approve_selection();
        }
        _ => chooser.cancel_selection(),
    }
}

/// Rust-only command `etomo-filechooser TITLE SELECTION_MODE DIALOG_TYPE DIR
/// APPROVE_TEXT MULTI CURRENT_FILTER FILTER...`: shows one file chooser.  It
/// asks the parent ([`present_file_chooser`]) for the entries of each
/// directory it shows ("LIST\t<filter>\t<dir>" on standard output, answered on
/// standard input with "D\t<name>" / "F\t<name>" lines and "END"), and prints
/// "APPROVE\t<path>[\t<path>...]" with the absolute paths chosen (nothing when
/// it was cancelled or closed).
pub fn etomo_filechooser(arguments: &[String]) -> i32 {
    use std::io::{BufRead, Write};
    let Ok(window) = FileChooserWindow::new() else {
        return 1;
    };
    let title = arguments.first().cloned().unwrap_or_default();
    let selection_mode: i32 = arguments.get(1).and_then(|t| t.parse().ok()).unwrap_or(0);
    let dialog_type: i32 = arguments.get(2).and_then(|t| t.parse().ok()).unwrap_or(0);
    let start = arguments
        .get(3)
        .map(std::path::PathBuf::from)
        .filter(|dir| dir.is_dir())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_default();
    let approve_text = arguments.get(4).cloned().unwrap_or_else(|| {
        if dialog_type == file_chooser::SAVE_DIALOG {
            "Save".to_owned()
        } else {
            "Open".to_owned()
        }
    });
    let multi = arguments.get(5).map(String::as_str) == Some("1");
    let current_filter: i32 = arguments.get(6).and_then(|t| t.parse().ok()).unwrap_or(0);
    let filters: Vec<slint::SharedString> = arguments.iter().skip(7).map(Into::into).collect();
    let directory = Rc::new(RefCell::new(start));
    let filter = Rc::new(std::cell::Cell::new(current_filter));
    // The listed entries (name, is directory) and which are selected.
    let entries = Rc::new(RefCell::new(Vec::<(String, bool)>::new()));
    let selected = Rc::new(RefCell::new(Vec::<bool>::new()));
    let stdin = Rc::new(RefCell::new(std::io::BufReader::new(std::io::stdin())));
    let show_selection = {
        let entries = entries.clone();
        let selected = selected.clone();
        let weak = window.as_weak();
        move || {
            let Some(window) = weak.upgrade() else {
                return;
            };
            window.set_selected_entries(slint::ModelRc::new(slint::VecModel::from(
                selected.borrow().clone(),
            )));
            // The "File Name" field shows the selected file (a multi-selection
            // chooser quotes each name), as `MetalFileChooserUI` does.
            let names: Vec<String> = entries
                .borrow()
                .iter()
                .zip(selected.borrow().iter())
                .filter(|(_, selected)| **selected)
                .map(|((name, _), _)| name.clone())
                .collect();
            let text = if names.len() > 1 {
                names
                    .iter()
                    .map(|name| format!("\"{name}\""))
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                names.first().cloned().unwrap_or_default()
            };
            window.set_file_name(text.into());
        }
    };
    let list = {
        let directory = directory.clone();
        let filter = filter.clone();
        let entries = entries.clone();
        let selected = selected.clone();
        let stdin = stdin.clone();
        let weak = window.as_weak();
        move || {
            let Some(window) = weak.upgrade() else {
                return;
            };
            let dir = directory.borrow().clone();
            let mut stdout = std::io::stdout();
            let _ = writeln!(stdout, "LIST\t{}\t{}", filter.get(), dir.to_string_lossy());
            let _ = stdout.flush();
            let mut listed = Vec::new();
            let mut line = String::new();
            loop {
                line.clear();
                match stdin.borrow_mut().read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                let line = line.trim_end_matches('\n');
                if line == "END" {
                    break;
                }
                if let Some(name) = line.strip_prefix("D\t") {
                    listed.push((name.to_owned(), true));
                } else if let Some(name) = line.strip_prefix("F\t") {
                    listed.push((name.to_owned(), false));
                }
            }
            window.set_directory(dir.to_string_lossy().into_owned().into());
            window.set_entries(slint::ModelRc::new(slint::VecModel::from(
                listed
                    .iter()
                    .map(|(name, is_dir)| {
                        slint::SharedString::from(if *is_dir {
                            format!("{name}/")
                        } else {
                            name.clone()
                        })
                    })
                    .collect::<Vec<_>>(),
            )));
            *selected.borrow_mut() = vec![false; listed.len()];
            *entries.borrow_mut() = listed;
            window.set_selected_entries(slint::ModelRc::new(slint::VecModel::from(
                selected.borrow().clone(),
            )));
        }
    };
    window.set_title_text(title.into());
    window.set_approve_text(approve_text.into());
    window.set_filters(slint::ModelRc::new(slint::VecModel::from(filters)));
    window.set_current_filter(current_filter);
    list();
    let chosen = Rc::new(RefCell::new(Vec::<std::path::PathBuf>::new()));
    {
        // A click selects the entry; Ctrl toggles it in a multi-selection
        // chooser.  A directory is only selectable when directories can be
        // chosen (`isDirectorySelectionEnabled`).
        let entries = entries.clone();
        let selected = selected.clone();
        let show_selection = show_selection.clone();
        window.on_entry_clicked(move |index, control| {
            let index = index.max(0) as usize;
            let Some((_, is_dir)) = entries.borrow().get(index).cloned() else {
                return;
            };
            if is_dir && selection_mode == file_chooser::FILES_ONLY {
                return;
            }
            {
                let mut selected = selected.borrow_mut();
                if multi && control {
                    selected[index] = !selected[index];
                } else {
                    selected.iter_mut().for_each(|s| *s = false);
                    selected[index] = true;
                }
            }
            show_selection();
        });
    }
    {
        let directory = directory.clone();
        let list = list.clone();
        let chosen = chosen.clone();
        let weak = window.as_weak();
        window.on_enter(move |entry| {
            let path = directory.borrow().join(entry.trim_end_matches('/'));
            if path.is_dir() {
                *directory.borrow_mut() = path;
                list();
            } else {
                // A double click on a file approves it.
                *chosen.borrow_mut() = vec![path];
                if let Some(window) = weak.upgrade() {
                    let _ = window.hide();
                }
            }
        });
    }
    {
        let directory = directory.clone();
        let list = list.clone();
        window.on_up(move || {
            let parent = directory
                .borrow()
                .parent()
                .map(|parent| parent.to_path_buf());
            if let Some(parent) = parent {
                *directory.borrow_mut() = parent;
                list();
            }
        });
    }
    {
        let filter = filter.clone();
        let list = list.clone();
        let weak = window.as_weak();
        window.on_filter_chosen(move |index| {
            filter.set(index);
            if let Some(window) = weak.upgrade() {
                window.set_current_filter(index);
            }
            list();
        });
    }
    {
        let directory = directory.clone();
        let chosen = chosen.clone();
        let list = list.clone();
        let weak = window.as_weak();
        window.on_approve(move |name| {
            let name = name.trim();
            let dir = directory.borrow().clone();
            let resolve = |name: &str| {
                let name = name.trim_end_matches('/');
                if std::path::Path::new(name).is_absolute() {
                    std::path::PathBuf::from(name)
                } else {
                    dir.join(name)
                }
            };
            // A multi-selection field holds quoted names.
            let names: Vec<String> = if multi && name.starts_with('"') {
                name.split('"')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(str::to_owned)
                    .collect()
            } else {
                vec![name.to_owned()]
            };
            if names.len() == 1 {
                let path = if names[0].is_empty() {
                    dir.clone()
                } else {
                    resolve(&names[0])
                };
                // A directory typed in a files-only chooser is entered, not
                // chosen.
                if !names[0].is_empty()
                    && path.is_dir()
                    && selection_mode == file_chooser::FILES_ONLY
                {
                    *directory.borrow_mut() = path;
                    list();
                    return;
                }
                *chosen.borrow_mut() = vec![path];
            } else {
                *chosen.borrow_mut() = names.iter().map(|name| resolve(name)).collect();
            }
            if let Some(window) = weak.upgrade() {
                let _ = window.hide();
            }
        });
    }
    {
        let weak = window.as_weak();
        window.on_cancel(move || {
            if let Some(window) = weak.upgrade() {
                let _ = window.hide();
            }
        });
    }
    if window.run().is_err() {
        return 1;
    }
    let chosen = chosen.borrow();
    if !chosen.is_empty() {
        let paths: Vec<String> = chosen
            .iter()
            .map(|path| utilities::java_io_file_get_absolute_path(&path.to_string_lossy()))
            .collect();
        println!("APPROVE\t{}", paths.join("\t"));
    }
    0
}
