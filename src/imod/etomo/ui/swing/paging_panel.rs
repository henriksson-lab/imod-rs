//! `IMOD/Etomo/src/etomo/ui/swing/PagingPanel.java`.
//!
//! Panel that holds paging buttons.  May also receive display Components and
//! define key actions on them.  Passes button and key action paging commands
//! on to Viewport.
//!
//! Key bindings (`InputMap`/`ActionMap`, `KeyStroke`) and focusability are not
//! modelled by `jdk.rs`; those statements are kept as comments, and the
//! actions are registered on the buttons as in the Java.

use std::rc::{Rc, Weak};

use super::single_line_button::SingleLineButton;
use super::viewport::Viewport;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};

/// Java package-private `final class PagingPanel`.
pub struct PagingPanel {
    /// Java `private final JPanel pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    btn_home: Rc<SingleLineButton>,
    btn_page_up: Rc<SingleLineButton>,
    btn_up: Rc<SingleLineButton>,
    btn_down: Rc<SingleLineButton>,
    btn_page_down: Rc<SingleLineButton>,
    btn_end: Rc<SingleLineButton>,
    /// Java `private final Viewport viewport`.  The viewport owns this panel,
    /// so the back reference is weak.
    viewport: Weak<Viewport>,
    /// Java `private final String uniqueKey`.
    unique_key: Option<String>,
}

impl PagingPanel {
    /// Java private `PagingPanel(Viewport, String)`.
    fn new(viewport: &Rc<Viewport>, unique_key: Option<&str>) -> PagingPanel {
        PagingPanel {
            pnl_root: JComponent::new_panel(),
            btn_home: SingleLineButton::new_void(),
            btn_page_up: SingleLineButton::new_void(),
            btn_up: SingleLineButton::new_void(),
            btn_down: SingleLineButton::new_void(),
            btn_page_down: SingleLineButton::new_void(),
            btn_end: SingleLineButton::new_void(),
            viewport: Rc::downgrade(viewport),
            unique_key: unique_key.map(str::to_owned),
        }
    }

    /// Java package-private static `getInstance(Viewport, String)`.
    ///
    /// Get PagingPanel instance with hotkey connections to 0 to three
    /// JComponents.  The JComponents can be the panel that the table has been
    /// placed on.  The uniqueKey uniquely identifies the table.  This is
    /// necessary when two tables appear on the same JPanel, so that separate
    /// actions are stored (no sure if this works yet).
    pub fn get_instance(viewport: &Rc<Viewport>, unique_key: Option<&str>) -> Rc<PagingPanel> {
        let instance = Rc::new(PagingPanel::new(viewport, unique_key));
        instance.create_panel();
        instance.add_listeners();
        instance.add_tooltips();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.setup_button(&self.btn_home, "home.png");
        self.setup_button(&self.btn_page_up, "pageUp.png");
        self.setup_button(&self.btn_up, "up.png");
        self.setup_button(&self.btn_down, "down.png");
        self.setup_button(&self.btn_page_down, "pageDown.png");
        self.setup_button(&self.btn_end, "end.png");
        // root
        // Swing layout: pnlRoot.setLayout(new BoxLayout(pnlRoot, BoxLayout.Y_AXIS)).
        self.pnl_root.add(&self.btn_home.get_component());
        self.pnl_root.add(&self.btn_page_up.get_component());
        self.pnl_root.add(&self.btn_up.get_component());
        // Swing layout: pnlRoot.add(Box.createVerticalGlue()).
        self.pnl_root.add(&self.btn_down.get_component());
        self.pnl_root.add(&self.btn_page_down.get_component());
        self.pnl_root.add(&self.btn_end.get_component());
    }

    /// Java private `setupButton(SingleLineButton, String)`.
    fn setup_button(&self, button: &SingleLineButton, icon_file: &str) {
        button.set_manual_name();
        // Swing layout: button.setBorder(BorderFactory.createBevelBorder(RAISED));
        // button.setIcon(new ImageIcon(ClassLoader.getSystemResource("images/" +
        // iconFile))); size = button.getPreferredSize(); a width below the height
        // is raised to the height; button.setSize(size).  Icons and sizes are not
        // modelled.
        let _ = icon_file;
    }

    /// Java private `addAction(InputMap[], ActionMap, int, String, AbstractAction)`.
    ///
    /// Binds `keyStroke` (no modifiers) to `uniqueKey + key` in each of the
    /// parent's three input maps, and `uniqueKey + key` to `action` in its action
    /// map.  Key bindings are not modelled, so only the key is formed.
    fn add_action(&self, key_stroke: i32, key: &str, action: &ActionListener) {
        // `inputMapsForParent != null` always holds at the one call site.
        // Java string concatenation renders a null uniqueKey as "null".
        let key = format!("{}{}", self.unique_key.as_deref().unwrap_or("null"), key);
        // Swing key binding: KeyStroke keystroke = KeyStroke.getKeyStroke(keyStroke, 0);
        // inputMapsForParent[i].put(keystroke, key) for each non-null map;
        // actionMap.put(key, action).
        let _ = (key_stroke, key, action);
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let Some(viewport) = self.viewport.upgrade() else {
            return;
        };
        // Java null test: the stand-in returns an empty array for null.
        let focusable_parents = viewport.get_focusable_parents();
        // int focusTypes = 3; InputMap[][] inputMaps; ActionMap[] actionMaps -
        // key bindings, not modelled.
        let home_action = PagingPanelHomeAction::new(&viewport);
        let page_up_action = PagingPanelPageUpAction::new(&viewport);
        let up_action = PagingPanelUpAction::new(&viewport);
        let down_action = PagingPanelDownAction::new(&viewport);
        let page_down_action = PagingPanelPageDownAction::new(&viewport);
        let end_action = PagingPanelEndAction::new(&viewport);
        // The AbstractActions as ActionListeners (Swing's Action is one).
        let home_listener: ActionListener = {
            let action = home_action.clone();
            Rc::new(move |event: &ActionEvent| action.action_performed(event))
        };
        let page_up_listener: ActionListener = {
            let action = page_up_action.clone();
            Rc::new(move |event: &ActionEvent| action.action_performed(event))
        };
        let up_listener: ActionListener = {
            let action = up_action.clone();
            Rc::new(move |event: &ActionEvent| action.action_performed(event))
        };
        let down_listener: ActionListener = {
            let action = down_action.clone();
            Rc::new(move |event: &ActionEvent| action.action_performed(event))
        };
        let page_down_listener: ActionListener = {
            let action = page_down_action.clone();
            Rc::new(move |event: &ActionEvent| action.action_performed(event))
        };
        let end_listener: ActionListener = {
            let action = end_action.clone();
            Rc::new(move |event: &ActionEvent| action.action_performed(event))
        };
        for focusable_parent in &focusable_parents {
            // Java skips null entries; the Rust array holds none.
            let _ = focusable_parent;
            // Swing focus: focusableParents[i].setFocusable(true).
            // Swing key binding: inputMaps[i][0..2] = getInputMap(WHEN_FOCUSED /
            // WHEN_IN_FOCUSED_WINDOW / WHEN_ANCESTOR_OF_FOCUSED_COMPONENT);
            // actionMaps[i] = getActionMap().
            self.add_action(VK_HOME, "HOME", &home_listener);
            self.add_action(VK_PAGE_UP, "PAGE_UP", &page_up_listener);
            self.add_action(VK_UP, "UP", &up_listener);
            self.add_action(VK_DOWN, "DOWN", &down_listener);
            self.add_action(VK_PAGE_DOWN, "PAGE_DOWN", &page_down_listener);
            self.add_action(VK_END, "END", &end_listener);
        }
        // add listeners
        self.btn_home.add_action_listener(home_listener);
        self.btn_page_up.add_action_listener(page_up_listener);
        self.btn_up.add_action_listener(up_listener);
        self.btn_down.add_action_listener(down_listener);
        self.btn_page_down.add_action_listener(page_down_listener);
        self.btn_end.add_action_listener(end_listener);
    }

    /// Java private `addTooltips()`.
    fn add_tooltips(&self) {
        let mut hotkeys = false;
        let focusable_parents = self
            .viewport
            .upgrade()
            .map(|viewport| viewport.get_focusable_parents());
        if focusable_parents.is_some_and(|parents| !parents.is_empty()) {
            hotkeys = true;
        }
        self.btn_home.set_tool_tip_text(Some("Top of table"));
        self.btn_page_up.set_tool_tip_text(Some(&format!(
            "Page up{}",
            if hotkeys { " [Page Up]" } else { "" }
        )));
        self.btn_up.set_tool_tip_text(Some(&format!(
            "Up one line{}",
            if hotkeys { " [Up_Arrow]" } else { "" }
        )));
        self.btn_down.set_tool_tip_text(Some(&format!(
            "Down one line{}",
            if hotkeys { " [Down_Arrow]" } else { "" }
        )));
        // Upstream typo fixed (PagingPanel.java:179): Java labels the page-down
        // button "Page up" as well; we write "Page down".
        self.btn_page_down.set_tool_tip_text(Some(&format!(
            "Page down{}",
            if hotkeys { " [Page Down]" } else { "" }
        )));
        self.btn_end.set_tool_tip_text(Some("Bottom of table"));
    }

    /// Java package-private `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java package-private `setUpEnabled(boolean)`.
    pub fn set_up_enabled(&self, enabled: bool) {
        self.btn_home.set_enabled(enabled);
        self.btn_page_up.set_enabled(enabled);
        self.btn_up.set_enabled(enabled);
    }

    /// Java package-private `setDownEnabled(boolean)`.
    pub fn set_down_enabled(&self, enabled: bool) {
        self.btn_end.set_enabled(enabled);
        self.btn_page_down.set_enabled(enabled);
        self.btn_down.set_enabled(enabled);
    }

    /// Java package-private `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.pnl_root.set_visible(visible);
    }
}

/// Java `KeyEvent.VK_HOME`.
const VK_HOME: i32 = 0x24;
/// Java `KeyEvent.VK_PAGE_UP`.
const VK_PAGE_UP: i32 = 0x21;
/// Java `KeyEvent.VK_UP`.
const VK_UP: i32 = 0x26;
/// Java `KeyEvent.VK_DOWN`.
const VK_DOWN: i32 = 0x28;
/// Java `KeyEvent.VK_PAGE_DOWN`.
const VK_PAGE_DOWN: i32 = 0x22;
/// Java `KeyEvent.VK_END`.
const VK_END: i32 = 0x23;

/// Java `private static final class PagingPanelHomeAction extends AbstractAction`.
struct PagingPanelHomeAction {
    viewport: Weak<Viewport>,
}

impl PagingPanelHomeAction {
    fn new(viewport: &Rc<Viewport>) -> Rc<Self> {
        Rc::new(PagingPanelHomeAction {
            viewport: Rc::downgrade(viewport),
        })
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, _event: &ActionEvent) {
        if let Some(viewport) = self.viewport.upgrade() {
            viewport.home_button_action();
        }
    }
}

/// Java `private static final class PagingPanelPageUpAction extends AbstractAction`.
struct PagingPanelPageUpAction {
    viewport: Weak<Viewport>,
}

impl PagingPanelPageUpAction {
    fn new(viewport: &Rc<Viewport>) -> Rc<Self> {
        Rc::new(PagingPanelPageUpAction {
            viewport: Rc::downgrade(viewport),
        })
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, _event: &ActionEvent) {
        if let Some(viewport) = self.viewport.upgrade() {
            viewport.page_up_button_action();
        }
    }
}

/// Java `private static final class PagingPanelUpAction extends AbstractAction`.
struct PagingPanelUpAction {
    viewport: Weak<Viewport>,
}

impl PagingPanelUpAction {
    fn new(viewport: &Rc<Viewport>) -> Rc<Self> {
        Rc::new(PagingPanelUpAction {
            viewport: Rc::downgrade(viewport),
        })
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, _event: &ActionEvent) {
        if let Some(viewport) = self.viewport.upgrade() {
            viewport.up_button_action();
        }
    }
}

/// Java `private static final class PagingPanelDownAction extends AbstractAction`.
struct PagingPanelDownAction {
    viewport: Weak<Viewport>,
}

impl PagingPanelDownAction {
    fn new(viewport: &Rc<Viewport>) -> Rc<Self> {
        Rc::new(PagingPanelDownAction {
            viewport: Rc::downgrade(viewport),
        })
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, _event: &ActionEvent) {
        if let Some(viewport) = self.viewport.upgrade() {
            viewport.down_button_action();
        }
    }
}

/// Java `private static final class PagingPanelPageDownAction extends AbstractAction`.
struct PagingPanelPageDownAction {
    viewport: Weak<Viewport>,
}

impl PagingPanelPageDownAction {
    fn new(viewport: &Rc<Viewport>) -> Rc<Self> {
        Rc::new(PagingPanelPageDownAction {
            viewport: Rc::downgrade(viewport),
        })
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, _event: &ActionEvent) {
        if let Some(viewport) = self.viewport.upgrade() {
            viewport.page_down_button_action();
        }
    }
}

/// Java `private static final class PagingPanelEndAction extends AbstractAction`.
struct PagingPanelEndAction {
    viewport: Weak<Viewport>,
}

impl PagingPanelEndAction {
    fn new(viewport: &Rc<Viewport>) -> Rc<Self> {
        Rc::new(PagingPanelEndAction {
            viewport: Rc::downgrade(viewport),
        })
    }

    /// Java `actionPerformed(ActionEvent)`.
    fn action_performed(&self, _event: &ActionEvent) {
        if let Some(viewport) = self.viewport.upgrade() {
            viewport.end_button_action();
        }
    }
}
