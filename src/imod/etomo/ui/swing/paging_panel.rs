//! `IMOD/Etomo/src/etomo/ui/swing/PagingPanel.java`.
//!
//! `JPanel`, `SingleLineButton`, `InputMap`, `ActionMap`, image loading, and
//! the Swing event loop are presentation boundaries.  The source-owned button
//! order, icons, action-map names, hotkeys, enablement, visibility, tooltips,
//! and the six forwarding actions are retained here.
#![allow(dead_code)]

use super::etomo_frame::ActionEvent;
use std::cell::RefCell;
use std::rc::Rc;

/// Direct `Viewport` boundary used by `PagingPanel`.
///
/// Java stores a final `Viewport` reference in both the panel and each nested
/// `AbstractAction`.  `Rc<RefCell<_>>` preserves that shared mutable ownership
/// in Rust without substituting paging logic into this presentation unit.
pub trait PagingViewport {
    /// Java `Viewport.getFocusableParents()`; component identities are native
    /// GUI identifiers at this boundary.
    fn get_focusable_parents(&self) -> Option<Vec<String>>;
    fn home_button_action(&mut self);
    fn page_up_button_action(&mut self);
    fn up_button_action(&mut self);
    fn down_button_action(&mut self);
    fn page_down_button_action(&mut self);
    fn end_button_action(&mut self);
}

/// Java's three `JComponent.getInputMap(int)` modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputMapType {
    WhenFocused,
    WhenInFocusedWindow,
    WhenAncestorOfFocusedComponent,
}

/// Java `KeyEvent` constants used in `addListeners()`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagingKeyStroke {
    Home,
    PageUp,
    Up,
    Down,
    PageDown,
    End,
}

/// The six private Java `AbstractAction` identities stored in an `ActionMap`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagingPanelAction {
    Home,
    PageUp,
    Up,
    Down,
    PageDown,
    End,
}

/// A native-input/action-map entry made by Java `PagingPanel.addAction`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagingKeyBinding {
    pub focusable_parent: String,
    pub input_map_type: InputMapType,
    pub key_stroke: PagingKeyStroke,
    pub action_map_key: String,
    pub action: PagingPanelAction,
}

/// Source-visible state of Java `SingleLineButton` used only by this unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagingPanelButton {
    pub manual_name: bool,
    pub bevel_border_raised: bool,
    pub icon_resource: Option<String>,
    pub preferred_size: (i32, i32),
    pub size: (i32, i32),
    pub enabled: bool,
    pub tool_tip_text: Option<String>,
    pub action_listener_present: bool,
}

impl Default for PagingPanelButton {
    fn default() -> Self {
        Self {
            manual_name: false,
            bevel_border_raised: false,
            icon_resource: None,
            // The actual preferred dimensions come from Swing's image/button
            // delegate.  They stay at that direct native-layout boundary.
            preferred_size: (0, 0),
            size: (0, 0),
            enabled: true,
            tool_tip_text: None,
            action_listener_present: false,
        }
    }
}

/// Java's root `JPanel` state and exact vertical child ordering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagingPanelRoot {
    pub vertical_box_layout: bool,
    pub visible: bool,
    pub children: Vec<PagingPanelLayoutItem>,
}

impl Default for PagingPanelRoot {
    fn default() -> Self {
        Self {
            vertical_box_layout: false,
            visible: true,
            children: Vec::new(),
        }
    }
}

/// Java `pnlRoot.add(...)` insertion identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagingPanelLayoutItem {
    Home,
    PageUp,
    Up,
    VerticalGlue,
    Down,
    PageDown,
    End,
}

/// Java final `PagingPanel` fields.
pub struct PagingPanel<V: PagingViewport> {
    pub pnl_root: PagingPanelRoot,
    pub btn_home: PagingPanelButton,
    pub btn_page_up: PagingPanelButton,
    pub btn_up: PagingPanelButton,
    pub btn_down: PagingPanelButton,
    pub btn_page_down: PagingPanelButton,
    pub btn_end: PagingPanelButton,
    pub viewport: Rc<RefCell<V>>,
    pub unique_key: String,
    /// The native Swing map writes made by `addAction`.
    pub key_bindings: Vec<PagingKeyBinding>,
    /// `JComponent.setFocusable(true)` writes made by `addListeners`.
    pub focusable_parents: Vec<String>,
}

impl<V: PagingViewport> PagingPanel<V> {
    /// Java private `PagingPanel(Viewport, String)`.
    fn new(viewport: Rc<RefCell<V>>, unique_key: impl Into<String>) -> Self {
        Self {
            pnl_root: PagingPanelRoot::default(),
            btn_home: PagingPanelButton::default(),
            btn_page_up: PagingPanelButton::default(),
            btn_up: PagingPanelButton::default(),
            btn_down: PagingPanelButton::default(),
            btn_page_down: PagingPanelButton::default(),
            btn_end: PagingPanelButton::default(),
            viewport,
            unique_key: unique_key.into(),
            key_bindings: Vec::new(),
            focusable_parents: Vec::new(),
        }
    }

    /// Java `getInstance(Viewport, String)`.
    pub fn get_instance(viewport: Rc<RefCell<V>>, unique_key: impl Into<String>) -> Self {
        let mut instance = Self::new(viewport, unique_key);
        instance.create_panel();
        instance.add_listeners();
        instance.add_tooltips();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&mut self) {
        Self::setup_button(&mut self.btn_home, "home.png");
        Self::setup_button(&mut self.btn_page_up, "pageUp.png");
        Self::setup_button(&mut self.btn_up, "up.png");
        Self::setup_button(&mut self.btn_down, "down.png");
        Self::setup_button(&mut self.btn_page_down, "pageDown.png");
        Self::setup_button(&mut self.btn_end, "end.png");
        self.pnl_root.vertical_box_layout = true;
        self.pnl_root.children = vec![
            PagingPanelLayoutItem::Home,
            PagingPanelLayoutItem::PageUp,
            PagingPanelLayoutItem::Up,
            PagingPanelLayoutItem::VerticalGlue,
            PagingPanelLayoutItem::Down,
            PagingPanelLayoutItem::PageDown,
            PagingPanelLayoutItem::End,
        ];
    }

    /// Java private `setupButton(SingleLineButton, String)`.
    fn setup_button(button: &mut PagingPanelButton, icon_file: &str) {
        button.manual_name = true;
        button.bevel_border_raised = true;
        button.icon_resource = Some(format!("images/{icon_file}"));
        let (mut width, height) = button.preferred_size;
        if width < height {
            width = height;
        }
        button.size = (width, height);
    }

    /// Java private `addAction(InputMap[], ActionMap, int, String,
    /// AbstractAction)`.  One native binding is retained for each of the three
    /// source `InputMap`s for the selected focusable parent.
    fn add_action(
        &mut self,
        focusable_parent: &str,
        key_stroke: PagingKeyStroke,
        key: &str,
        action: PagingPanelAction,
    ) {
        let action_map_key = format!("{}{}", self.unique_key, key);
        for input_map_type in [
            InputMapType::WhenFocused,
            InputMapType::WhenInFocusedWindow,
            InputMapType::WhenAncestorOfFocusedComponent,
        ] {
            self.key_bindings.push(PagingKeyBinding {
                focusable_parent: focusable_parent.to_string(),
                input_map_type,
                key_stroke,
                action_map_key: action_map_key.clone(),
                action,
            });
        }
    }

    /// Java private `addListeners()`.
    fn add_listeners(&mut self) {
        let Some(focusable_parents) = self.viewport.borrow().get_focusable_parents() else {
            return;
        };
        for focusable_parent in focusable_parents {
            self.focusable_parents.push(focusable_parent.clone());
            self.add_action(
                &focusable_parent,
                PagingKeyStroke::Home,
                "HOME",
                PagingPanelAction::Home,
            );
            self.add_action(
                &focusable_parent,
                PagingKeyStroke::PageUp,
                "PAGE_UP",
                PagingPanelAction::PageUp,
            );
            self.add_action(
                &focusable_parent,
                PagingKeyStroke::Up,
                "UP",
                PagingPanelAction::Up,
            );
            self.add_action(
                &focusable_parent,
                PagingKeyStroke::Down,
                "DOWN",
                PagingPanelAction::Down,
            );
            self.add_action(
                &focusable_parent,
                PagingKeyStroke::PageDown,
                "PAGE_DOWN",
                PagingPanelAction::PageDown,
            );
            self.add_action(
                &focusable_parent,
                PagingKeyStroke::End,
                "END",
                PagingPanelAction::End,
            );
        }
        self.btn_home.action_listener_present = true;
        self.btn_page_up.action_listener_present = true;
        self.btn_up.action_listener_present = true;
        self.btn_down.action_listener_present = true;
        self.btn_page_down.action_listener_present = true;
        self.btn_end.action_listener_present = true;
    }

    /// Java private `addTooltips()`.
    fn add_tooltips(&mut self) {
        let hotkeys = self
            .viewport
            .borrow()
            .get_focusable_parents()
            .is_some_and(|parents| !parents.is_empty());
        self.btn_home.tool_tip_text = Some("Top of table".into());
        self.btn_page_up.tool_tip_text = Some(format!(
            "Page up{}",
            if hotkeys { " [Page Up]" } else { "" }
        ));
        self.btn_up.tool_tip_text = Some(format!(
            "Up one line{}",
            if hotkeys { " [Up_Arrow]" } else { "" }
        ));
        self.btn_down.tool_tip_text = Some(format!(
            "Down one line{}",
            if hotkeys { " [Down_Arrow]" } else { "" }
        ));
        // Preserve the Java source's "Page up" wording for this button.
        self.btn_page_down.tool_tip_text = Some(format!(
            "Page up{}",
            if hotkeys { " [Page Down]" } else { "" }
        ));
        self.btn_end.tool_tip_text = Some("Bottom of table".into());
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> &PagingPanelRoot {
        &self.pnl_root
    }

    /// Java `setUpEnabled(boolean)`.
    pub fn set_up_enabled(&mut self, enabled: bool) {
        self.btn_home.enabled = enabled;
        self.btn_page_up.enabled = enabled;
        self.btn_up.enabled = enabled;
    }

    /// Java `setDownEnabled(boolean)`.
    pub fn set_down_enabled(&mut self, enabled: bool) {
        self.btn_end.enabled = enabled;
        self.btn_page_down.enabled = enabled;
        self.btn_down.enabled = enabled;
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.pnl_root.visible = visible;
    }

    /// Native dispatch at the direct Swing action/button boundary.
    pub fn action_performed(&self, action: PagingPanelAction, event: &ActionEvent) {
        match action {
            PagingPanelAction::Home => {
                PagingPanelHomeAction::new(self.viewport.clone()).action_performed(event)
            }
            PagingPanelAction::PageUp => {
                PagingPanelPageUpAction::new(self.viewport.clone()).action_performed(event)
            }
            PagingPanelAction::Up => {
                PagingPanelUpAction::new(self.viewport.clone()).action_performed(event)
            }
            PagingPanelAction::Down => {
                PagingPanelDownAction::new(self.viewport.clone()).action_performed(event)
            }
            PagingPanelAction::PageDown => {
                PagingPanelPageDownAction::new(self.viewport.clone()).action_performed(event)
            }
            PagingPanelAction::End => {
                PagingPanelEndAction::new(self.viewport.clone()).action_performed(event)
            }
        }
    }
}

/// Java private static `PagingPanelHomeAction`.
pub struct PagingPanelHomeAction<V: PagingViewport> {
    pub viewport: Rc<RefCell<V>>,
}
impl<V: PagingViewport> PagingPanelHomeAction<V> {
    fn new(viewport: Rc<RefCell<V>>) -> Self {
        Self { viewport }
    }
    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.viewport.borrow_mut().home_button_action();
    }
}

/// Java private static `PagingPanelPageUpAction`.
pub struct PagingPanelPageUpAction<V: PagingViewport> {
    pub viewport: Rc<RefCell<V>>,
}
impl<V: PagingViewport> PagingPanelPageUpAction<V> {
    fn new(viewport: Rc<RefCell<V>>) -> Self {
        Self { viewport }
    }
    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.viewport.borrow_mut().page_up_button_action();
    }
}

/// Java private static `PagingPanelUpAction`.
pub struct PagingPanelUpAction<V: PagingViewport> {
    pub viewport: Rc<RefCell<V>>,
}
impl<V: PagingViewport> PagingPanelUpAction<V> {
    fn new(viewport: Rc<RefCell<V>>) -> Self {
        Self { viewport }
    }
    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.viewport.borrow_mut().up_button_action();
    }
}

/// Java private static `PagingPanelDownAction`.
pub struct PagingPanelDownAction<V: PagingViewport> {
    pub viewport: Rc<RefCell<V>>,
}
impl<V: PagingViewport> PagingPanelDownAction<V> {
    fn new(viewport: Rc<RefCell<V>>) -> Self {
        Self { viewport }
    }
    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.viewport.borrow_mut().down_button_action();
    }
}

/// Java private static `PagingPanelPageDownAction`.
pub struct PagingPanelPageDownAction<V: PagingViewport> {
    pub viewport: Rc<RefCell<V>>,
}
impl<V: PagingViewport> PagingPanelPageDownAction<V> {
    fn new(viewport: Rc<RefCell<V>>) -> Self {
        Self { viewport }
    }
    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.viewport.borrow_mut().page_down_button_action();
    }
}

/// Java private static `PagingPanelEndAction`.
pub struct PagingPanelEndAction<V: PagingViewport> {
    pub viewport: Rc<RefCell<V>>,
}
impl<V: PagingViewport> PagingPanelEndAction<V> {
    fn new(viewport: Rc<RefCell<V>>) -> Self {
        Self { viewport }
    }
    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&self, _event: &ActionEvent) {
        self.viewport.borrow_mut().end_button_action();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct TestViewport {
        parents: Option<Vec<String>>,
        actions: Vec<PagingPanelAction>,
    }
    impl PagingViewport for TestViewport {
        fn get_focusable_parents(&self) -> Option<Vec<String>> {
            self.parents.clone()
        }
        fn home_button_action(&mut self) {
            self.actions.push(PagingPanelAction::Home);
        }
        fn page_up_button_action(&mut self) {
            self.actions.push(PagingPanelAction::PageUp);
        }
        fn up_button_action(&mut self) {
            self.actions.push(PagingPanelAction::Up);
        }
        fn down_button_action(&mut self) {
            self.actions.push(PagingPanelAction::Down);
        }
        fn page_down_button_action(&mut self) {
            self.actions.push(PagingPanelAction::PageDown);
        }
        fn end_button_action(&mut self) {
            self.actions.push(PagingPanelAction::End);
        }
    }

    #[test]
    fn construction_retains_source_order_icons_keys_and_tooltips() {
        let viewport = Rc::new(RefCell::new(TestViewport {
            parents: Some(vec!["table".into(), "root".into()]),
            actions: vec![],
        }));
        let panel = PagingPanel::get_instance(viewport, "volume-");
        assert_eq!(panel.pnl_root.children.len(), 7);
        assert_eq!(
            panel.btn_home.icon_resource.as_deref(),
            Some("images/home.png")
        );
        assert_eq!(panel.key_bindings.len(), 36);
        assert_eq!(panel.key_bindings[0].action_map_key, "volume-HOME");
        assert_eq!(
            panel.key_bindings[0].input_map_type,
            InputMapType::WhenFocused
        );
        assert_eq!(
            panel.btn_page_down.tool_tip_text.as_deref(),
            Some("Page up [Page Down]")
        );
        assert!(panel.btn_end.action_listener_present);
    }

    #[test]
    fn action_classes_forward_every_source_command_to_viewport() {
        let viewport = Rc::new(RefCell::new(TestViewport {
            parents: None,
            actions: vec![],
        }));
        let panel = PagingPanel::get_instance(viewport.clone(), "key");
        let event = ActionEvent::new("");
        for action in [
            PagingPanelAction::Home,
            PagingPanelAction::PageUp,
            PagingPanelAction::Up,
            PagingPanelAction::Down,
            PagingPanelAction::PageDown,
            PagingPanelAction::End,
        ] {
            panel.action_performed(action, &event);
        }
        assert_eq!(
            viewport.borrow().actions,
            vec![
                PagingPanelAction::Home,
                PagingPanelAction::PageUp,
                PagingPanelAction::Up,
                PagingPanelAction::Down,
                PagingPanelAction::PageDown,
                PagingPanelAction::End,
            ]
        );
    }

    #[test]
    fn enablement_and_visibility_follow_the_three_button_groups() {
        let viewport = Rc::new(RefCell::new(TestViewport::default()));
        let mut panel = PagingPanel::get_instance(viewport, "key");
        panel.set_up_enabled(false);
        panel.set_down_enabled(false);
        panel.set_visible(false);
        assert!(!panel.btn_home.enabled && !panel.btn_page_up.enabled && !panel.btn_up.enabled);
        assert!(!panel.btn_end.enabled && !panel.btn_page_down.enabled && !panel.btn_down.enabled);
        assert!(!panel.get_container().visible);
        assert_eq!(panel.btn_up.tool_tip_text.as_deref(), Some("Up one line"));
    }
}
