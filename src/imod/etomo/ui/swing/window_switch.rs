//! `IMOD/Etomo/src/etomo/ui/swing/WindowSwitch.java`.
//!
//! This unit intentionally owns only window-menu/tab state.  Java calls
//! `EtomoDirector.INSTANCE.setCurrentManager` and obtains close listeners from
//! `UIHarness`; their Rust equivalents are callback boundaries supplied by the
//! director once its manager list is fully translated.  No manager action is
//! invented here.

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::unique_hashed_array::UniqueHashedArray;
use crate::imod::etomo::util::unique_key::UniqueKey;

const MENU_ITEM_DIVIDER_CHAR: char = ':';
const MENU_ITEM_DIVIDER: &str = ": ";

/// The `BaseManager.getMainPanel()` boundary used by Java `WindowSwitch.add`.
pub trait WindowManager<P> {
    /// Java `BaseManager.getMainPanel()`.
    fn get_main_panel(&mut self) -> Option<P>;
}

/// The `MainPanel.saveDisplayState()` boundary used by Java `setTabs`.
pub trait WindowMainPanel {
    /// Java `MainPanel.saveDisplayState()`.
    fn save_display_state(&mut self);
}

/// Java `Menu` state as used by this source unit.
#[derive(Debug)]
pub struct Menu {
    name: String,
    items: Vec<Rc<RefCell<CheckBoxMenuItem>>>,
}

impl Menu {
    /// Java `new Menu("Window")`.
    fn new(name: String) -> Self {
        Self {
            name,
            items: Vec::new(),
        }
    }

    /// Java `add(JCheckBoxMenuItem)`.
    fn add(&mut self, item: Rc<RefCell<CheckBoxMenuItem>>) {
        self.items.push(item);
    }

    /// Java `remove(JCheckBoxMenuItem)`.
    fn remove(&mut self, item: &Rc<RefCell<CheckBoxMenuItem>>) {
        self.items
            .retain(|stored_item| !Rc::ptr_eq(stored_item, item));
    }

    /// Read-only Rust view of Java `Menu` for the Slint menu boundary.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Read-only Rust view of the Java `Menu` items.
    pub fn items(&self) -> &[Rc<RefCell<CheckBoxMenuItem>>] {
        &self.items
    }
}

/// Java `JCheckBoxMenuItem` fields used by `WindowSwitch`.
#[derive(Debug, Default)]
pub struct CheckBoxMenuItem {
    text: String,
    visible: bool,
    selected: bool,
}

impl CheckBoxMenuItem {
    /// Java `new CheckBoxMenuItem()`.
    fn new() -> Self {
        Self::default()
    }

    /// Java `setText`.
    fn set_text(&mut self, text: String) {
        self.text = text;
    }

    /// Java `getText`.
    fn get_text(&self) -> &str {
        &self.text
    }

    /// Java `setVisible`.
    fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Java `setSelected`.
    fn set_selected(&mut self, selected: bool) {
        self.selected = selected;
    }

    /// Slint read boundary for Java `getText`.
    pub fn text(&self) -> &str {
        self.get_text()
    }

    /// Slint read boundary for Java `isVisible`.
    pub fn visible(&self) -> bool {
        self.visible
    }

    /// Slint read boundary for Java `isSelected`.
    pub fn selected(&self) -> bool {
        self.selected
    }
}

/// Java `Ebutton` close instance and the listener arguments supplied by
/// `WindowSwitch.createCloseButton`.
#[derive(Clone, Debug)]
pub struct CloseButton {
    axis_id: AxisID,
    manager_key: UniqueKey,
}

impl CloseButton {
    /// Java `Ebutton.getCloseInstance()` plus the listener creation in
    /// `WindowSwitch.createCloseButton`.
    fn get_close_instance(axis_id: AxisID, manager_key: UniqueKey) -> Self {
        Self {
            axis_id,
            manager_key,
        }
    }

    /// Java `UIHarness.getCloseActionListener(axisID, managerKey)` arguments.
    pub fn action_arguments(&self) -> (AxisID, &UniqueKey) {
        (self.axis_id, &self.manager_key)
    }
}

/// Java `TabbedPane` state used by `WindowSwitch`.
#[derive(Debug, Default)]
pub struct TabbedPane {
    tabs: Vec<Tab>,
    selected_index: Option<usize>,
    change_listener_installed: bool,
}

/// One Java `TabbedPane.addTab` entry.
#[derive(Debug)]
pub struct Tab {
    title: String,
    contains_main_panel: bool,
    close_button: CloseButton,
}

impl TabbedPane {
    /// Java `new TabbedPane()`.
    fn new() -> Self {
        Self::default()
    }

    /// Java `getSelectedIndex()`.
    fn get_selected_index(&self) -> Option<usize> {
        self.selected_index
    }

    /// Java `removeChangeListener(tabChangeListener)`.
    fn remove_change_listener(&mut self) {
        self.change_listener_installed = false;
    }

    /// Java `removeAll()`.
    fn remove_all(&mut self) {
        self.tabs.clear();
        self.selected_index = None;
    }

    /// Java `addTab(String, Component)` plus `setTabComponentAt`.
    fn add_tab(&mut self, title: String, contains_main_panel: bool, close_button: CloseButton) {
        self.tabs.push(Tab {
            title,
            contains_main_panel,
            close_button,
        });
    }

    /// Java `setSelectedIndex(int)`.
    fn set_selected_index(&mut self, selected_index: usize) {
        self.selected_index = Some(selected_index);
    }

    /// Java `addChangeListener(tabChangeListener)`.
    fn add_change_listener(&mut self) {
        self.change_listener_installed = true;
    }

    /// Slint read boundary for Java `getTabCount`/`getTitleAt`.
    pub fn tabs(&self) -> &[Tab] {
        &self.tabs
    }

    /// Slint read boundary for Java `getSelectedIndex`.
    pub fn selected_index(&self) -> Option<usize> {
        self.get_selected_index()
    }
}

impl Tab {
    /// Java `setTitleAt` state.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Whether Java `setTabs` installed a `MainPanel` rather than its hidden
    /// `JLabel` placeholder.
    pub fn contains_main_panel(&self) -> bool {
        self.contains_main_panel
    }

    /// Java tab close component.
    pub fn close_button(&self) -> &CloseButton {
        &self.close_button
    }
}

/// Java `WindowSwitch.TabItems`.
#[derive(Debug)]
pub struct TabItems<P> {
    menu_item: Rc<RefCell<CheckBoxMenuItem>>,
    main_panel: P,
    close_button: CloseButton,
}

impl<P> TabItems<P> {
    /// Java `TabItems(JCheckBoxMenuItem, MainPanel, Ebutton)`.
    fn new(
        menu_item: Rc<RefCell<CheckBoxMenuItem>>,
        main_panel: P,
        close_button: CloseButton,
    ) -> Self {
        Self {
            menu_item,
            main_panel,
            close_button,
        }
    }

    /// Java `getMenuItem()`.
    fn get_menu_item(&self) -> &Rc<RefCell<CheckBoxMenuItem>> {
        &self.menu_item
    }

    /// Java `getMainPanel()`.
    fn get_main_panel(&self) -> &P {
        &self.main_panel
    }

    /// Mutable Rust form of Java `getMainPanel` required by
    /// `MainPanel.saveDisplayState()`.
    fn get_main_panel_mut(&mut self) -> &mut P {
        &mut self.main_panel
    }

    /// Java `getCloseButton()`.
    fn get_close_button(&self) -> &CloseButton {
        &self.close_button
    }
}

/// The component returned by Java `WindowSwitch.getPanel`.
#[derive(Debug)]
pub enum WindowPanel<'a, P> {
    /// Java returns `tabItems.getMainPanel()` for one window.
    MainPanel(&'a P),
    /// Java returns its one `TabbedPane` for two or more windows.
    TabbedPane(&'a TabbedPane),
}

/// Java `WindowSwitch`.
pub struct WindowSwitch<P: WindowMainPanel> {
    menu: Menu,
    tab_list: Option<UniqueHashedArray<TabItems<P>>>,
    tabbed_pane: Option<TabbedPane>,
    set_current_manager: Option<Box<dyn FnMut(UniqueKey)>>,
}

impl<P: WindowMainPanel> WindowSwitch<P> {
    /// Java `WindowSwitch()`.
    pub fn new() -> Self {
        Self {
            menu: Menu::new("Window".to_owned()),
            tab_list: None,
            tabbed_pane: None,
            set_current_manager: None,
        }
    }

    /// Installs the direct Rust endpoint for Java
    /// `EtomoDirector.INSTANCE.setCurrentManager`.  The endpoint is supplied
    /// by the translated director; it is not a replacement manager action.
    pub fn set_current_manager_listener(&mut self, listener: Box<dyn FnMut(UniqueKey)>) {
        self.set_current_manager = Some(listener);
    }

    /// Java `add(BaseManager, AxisID, UniqueKey)`.
    pub fn add<M: WindowManager<P>>(
        &mut self,
        manager: &mut M,
        axis_id: AxisID,
        manager_key: Option<UniqueKey>,
    ) {
        let Some(manager_key) = manager_key else {
            return;
        };
        if self.tab_list.is_none() {
            self.tab_list = Some(UniqueHashedArray::new());
            self.tabbed_pane = Some(TabbedPane::new());
        }
        let Some(main_panel) = manager.get_main_panel() else {
            return;
        };
        let menu_item = Rc::new(RefCell::new(CheckBoxMenuItem::new()));
        let index = self.tab_list.as_ref().map_or(0, UniqueHashedArray::size);
        menu_item.borrow_mut().set_text(format!(
            "{}{}{}",
            index + 1,
            MENU_ITEM_DIVIDER,
            manager_key.get_name()
        ));
        menu_item.borrow_mut().set_visible(true);
        self.menu.add(Rc::clone(&menu_item));
        let close_button = self.create_close_button(axis_id, manager_key.clone());
        let _ = self
            .tab_list
            .as_mut()
            .expect("WindowSwitch.add initializes tab_list")
            .add(
                manager_key,
                TabItems::new(menu_item, main_panel, close_button),
            );
    }

    /// Java `rename(UniqueKey, UniqueKey)`.
    pub fn rename(
        &mut self,
        old_manager_key: Option<&UniqueKey>,
        new_manager_key: Option<UniqueKey>,
    ) {
        let (Some(old_manager_key), Some(new_manager_key), Some(tab_list)) =
            (old_manager_key, new_manager_key, self.tab_list.as_mut())
        else {
            return;
        };
        let Some(index) = tab_list.get_index(old_manager_key) else {
            return;
        };
        let Some(tab_items) = tab_list.get(old_manager_key) else {
            return;
        };
        tab_items.get_menu_item().borrow_mut().set_text(format!(
            "{}{}{}",
            index + 1,
            MENU_ITEM_DIVIDER,
            new_manager_key.get_name()
        ));
        let _ = tab_list.rekey(old_manager_key, new_manager_key.clone());
        if tab_list.size() > 1
            && self
                .tabbed_pane
                .as_ref()
                .is_some_and(|tabbed_pane| tabbed_pane.tabs.len() > index)
        {
            self.tabbed_pane.as_mut().expect("checked above").tabs[index].title =
                new_manager_key.get_name().to_owned();
        }
        if let Some(listener) = &mut self.set_current_manager {
            listener(new_manager_key);
        }
    }

    /// Java `remove(UniqueKey)`.
    pub fn remove(&mut self, manager_key: Option<&UniqueKey>) {
        let (Some(manager_key), Some(tab_list)) = (manager_key, self.tab_list.as_mut()) else {
            return;
        };
        let Some(tab_items) = tab_list.get(manager_key) else {
            return;
        };
        let menu_item = Rc::clone(tab_items.get_menu_item());
        self.menu.remove(&menu_item);
        tab_list.remove(manager_key);
        self.renumber_menu();
    }

    /// Java private `renumberMenu()`.
    fn renumber_menu(&mut self) {
        let Some(tab_list) = self.tab_list.as_ref() else {
            return;
        };
        for index in 0..tab_list.size() {
            let Some(tab_items) = tab_list.get_at(index) else {
                continue;
            };
            let menu_item = tab_items.get_menu_item();
            let text = menu_item.borrow().get_text().to_owned();
            let Some(divider_index) = text.find(MENU_ITEM_DIVIDER_CHAR) else {
                continue;
            };
            menu_item
                .borrow_mut()
                .set_text(format!("{}{}", index + 1, &text[divider_index..]));
        }
    }

    /// Java `getMenu()`.
    pub fn get_menu(&self) -> &Menu {
        &self.menu
    }

    /// Java `getPanel(UniqueKey)`.
    pub fn get_panel(&mut self, manager_key: Option<&UniqueKey>) -> Option<WindowPanel<'_, P>> {
        let manager_key = manager_key?;
        let size = self.tab_list.as_ref()?.size();
        if size == 0 {
            return None;
        }
        if size == 1 {
            return self
                .tab_list
                .as_ref()?
                .get(manager_key)
                .map(|tab_items| WindowPanel::MainPanel(tab_items.get_main_panel()));
        }
        let selected_tab_index = self.tab_list.as_ref()?.get_index(manager_key)?;
        self.set_tabs(selected_tab_index, manager_key);
        self.tabbed_pane.as_ref().map(WindowPanel::TabbedPane)
    }

    /// Java `selectWindow(UniqueKey, boolean)`.
    pub fn select_window(&mut self, manager_key: Option<&UniqueKey>, _new_window: bool) {
        let Some(manager_key) = manager_key else {
            return;
        };
        let Some(index) = self
            .tab_list
            .as_ref()
            .and_then(|tab_list| tab_list.get_index(manager_key))
        else {
            return;
        };
        self.select_menu_item(index);
    }

    /// Java private `selectMenuItem(int)`.
    fn select_menu_item(&mut self, index: usize) {
        let Some(tab_list) = self.tab_list.as_ref() else {
            return;
        };
        for item_index in 0..tab_list.size() {
            let Some(tab_items) = tab_list.get_at(item_index) else {
                continue;
            };
            tab_items
                .get_menu_item()
                .borrow_mut()
                .set_selected(item_index == index);
        }
    }

    /// Java private `setTabs(int, UniqueKey)`.
    fn set_tabs(&mut self, selected_tab_index: usize, _manager_key: &UniqueKey) {
        let old_index = self
            .tabbed_pane
            .as_ref()
            .and_then(TabbedPane::get_selected_index);
        let (tab_count, tabs) = {
            let Some(tab_list) = self.tab_list.as_mut() else {
                return;
            };
            if let Some(old_index) = old_index
                && let Some(tab_items) = tab_list.get_at_mut(old_index)
            {
                tab_items.get_main_panel_mut().save_display_state();
            }
            let mut tabs = Vec::new();
            for index in 0..tab_list.size() {
                let Some(tab_items) = tab_list.get_at(index) else {
                    continue;
                };
                let text = tab_items.get_menu_item().borrow().get_text().to_owned();
                let Some(divider_index) = text.find(MENU_ITEM_DIVIDER) else {
                    continue;
                };
                tabs.push((
                    index,
                    text[divider_index + MENU_ITEM_DIVIDER.len()..].to_owned(),
                    tab_items.get_close_button().clone(),
                ));
            }
            (tab_list.size(), tabs)
        };
        if let Some(tabbed_pane) = self.tabbed_pane.as_mut() {
            tabbed_pane.remove_change_listener();
            tabbed_pane.remove_all();
            if tab_count < 2 {
                return;
            }
            for (index, tab_name, close_button) in &tabs {
                tabbed_pane.add_tab(
                    tab_name.clone(),
                    *index == selected_tab_index,
                    close_button.clone(),
                );
            }
            tabbed_pane.set_selected_index(selected_tab_index);
            tabbed_pane.add_change_listener();
        }
        for (index, tab_name, close_button) in &tabs {
            self.add_close_button_to_tab(close_button, tab_name, *index);
        }
    }

    /// Java private `createCloseButton(AxisID, UniqueKey)`.
    fn create_close_button(&mut self, axis_id: AxisID, manager_key: UniqueKey) -> CloseButton {
        if let Some(tab_items) = self
            .tab_list
            .as_ref()
            .and_then(|tab_list| tab_list.get(&manager_key))
        {
            return tab_items.get_close_button().clone();
        }
        CloseButton::get_close_instance(axis_id, manager_key)
    }

    /// Java private `addCloseButtonToTab(Ebutton, String, int)`.
    fn add_close_button_to_tab(
        &mut self,
        _close_button: &CloseButton,
        _name: &str,
        _tab_index: usize,
    ) {
        // The structural result is stored by `TabbedPane.add_tab`: a title,
        // `FixedDim.x25_y0` equivalent in the Slint renderer, and the same
        // close-button object.  Swing's GridBagConstraints themselves have no
        // retained model outside that component tree.
    }

    /// Java `menuAction(ActionEvent)`.
    pub fn menu_action(&mut self, action_command: &str) {
        let Some(divider_index) = action_command.find(MENU_ITEM_DIVIDER_CHAR) else {
            return;
        };
        let Ok(menu_number) = action_command[..divider_index].parse::<usize>() else {
            return;
        };
        let Some(new_index) = menu_number.checked_sub(1) else {
            return;
        };
        let Some(manager_key) = self
            .tab_list
            .as_ref()
            .and_then(|tab_list| tab_list.get_key(new_index))
            .cloned()
        else {
            return;
        };
        self.select_menu_item(new_index);
        if let Some(listener) = &mut self.set_current_manager {
            listener(manager_key);
        }
    }

    /// Rust callback endpoint for Java `MenuActionListener.actionPerformed`.
    ///
    /// The native menu frontend passes the selected item's action command;
    /// keeping this adapter preserves the Java listener boundary while the
    /// actual transition remains in `menu_action`.
    #[allow(non_snake_case)]
    pub fn actionPerformed(&mut self, action_command: &str) {
        self.menu_action(action_command);
    }

    /// Java `tabChanged(ChangeEvent)`.
    pub fn tab_changed(&mut self, new_index: Option<usize>) {
        let Some(new_index) = new_index else {
            return;
        };
        let Some(manager_key) = self
            .tab_list
            .as_ref()
            .and_then(|tab_list| tab_list.get_key(new_index))
            .cloned()
        else {
            return;
        };
        self.select_menu_item(new_index);
        if let Some(listener) = &mut self.set_current_manager {
            listener(manager_key);
        }
    }

    /// Rust callback endpoint for Java `TabChangeListener.stateChanged`.
    ///
    /// Slint supplies the currently selected tab index directly instead of a
    /// Swing `ChangeEvent`.
    #[allow(non_snake_case)]
    pub fn stateChanged(&mut self, new_index: Option<usize>) {
        self.tab_changed(new_index);
    }
}

impl<P: WindowMainPanel> Default for WindowSwitch<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::{WindowMainPanel, WindowManager, WindowPanel, WindowSwitch};
    use crate::imod::etomo::r#type::axis_id::AxisID;
    use crate::imod::etomo::util::unique_hashed_array::UniqueHashedArray;

    #[derive(Debug)]
    struct Panel(usize);

    impl WindowMainPanel for Panel {
        fn save_display_state(&mut self) {
            self.0 += 1;
        }
    }

    struct Manager(Option<Panel>);

    impl WindowManager<Panel> for Manager {
        fn get_main_panel(&mut self) -> Option<Panel> {
            self.0.take()
        }
    }

    #[test]
    fn tabs_follow_ordered_keys_and_keep_only_the_selected_panel() {
        let mut keys = UniqueHashedArray::<()>::new();
        let first = keys.add_with_name("one.edf".to_owned(), ());
        let second = keys.add_with_name("two.edf".to_owned(), ());
        let mut window_switch = WindowSwitch::new();
        window_switch.add(
            &mut Manager(Some(Panel(0))),
            AxisID::Only,
            Some(first.clone()),
        );
        window_switch.add(
            &mut Manager(Some(Panel(0))),
            AxisID::Only,
            Some(second.clone()),
        );
        let Some(WindowPanel::TabbedPane(tabs)) = window_switch.get_panel(Some(&second)) else {
            panic!("two windows must use TabbedPane");
        };
        assert_eq!(tabs.tabs().len(), 2);
        assert!(!tabs.tabs()[0].contains_main_panel());
        assert!(tabs.tabs()[1].contains_main_panel());
        assert_eq!(tabs.selected_index(), Some(1));
        assert_eq!(
            window_switch.get_menu().items()[1].borrow().text(),
            "2: two.edf"
        );
    }

    #[test]
    fn remove_renumbers_menu_items() {
        let mut keys = UniqueHashedArray::<()>::new();
        let first = keys.add_with_name("one.edf".to_owned(), ());
        let second = keys.add_with_name("two.edf".to_owned(), ());
        let mut window_switch = WindowSwitch::new();
        window_switch.add(
            &mut Manager(Some(Panel(0))),
            AxisID::Only,
            Some(first.clone()),
        );
        window_switch.add(
            &mut Manager(Some(Panel(0))),
            AxisID::Only,
            Some(second.clone()),
        );
        window_switch.remove(Some(&first));
        assert_eq!(
            window_switch.get_menu().items()[0].borrow().text(),
            "1: two.edf"
        );
    }

    #[test]
    fn source_rename_menu_and_tab_paths_select_the_director_key() {
        let mut keys = UniqueHashedArray::<()>::new();
        let first = keys.add_with_name("one.edf".to_owned(), ());
        let second = keys.add_with_name("two.edf".to_owned(), ());
        let renamed = keys.add_with_name("two-renamed.edf".to_owned(), ());
        let selected = Rc::new(RefCell::new(Vec::new()));
        let mut window_switch = WindowSwitch::new();
        let selected_listener = Rc::clone(&selected);
        window_switch.set_current_manager_listener(Box::new(move |key| {
            selected_listener
                .borrow_mut()
                .push(key.get_name().to_owned());
        }));
        window_switch.add(
            &mut Manager(Some(Panel(0))),
            AxisID::Only,
            Some(first.clone()),
        );
        window_switch.add(
            &mut Manager(Some(Panel(0))),
            AxisID::Second,
            Some(second.clone()),
        );

        window_switch.rename(Some(&second), Some(renamed.clone()));
        window_switch.menu_action("1: one.edf");
        window_switch.tab_changed(Some(1));

        assert_eq!(
            selected.borrow().as_slice(),
            ["two-renamed.edf", "one.edf", "two-renamed.edf"]
        );
        assert_eq!(
            window_switch.get_menu().items()[1].borrow().text(),
            "2: two-renamed.edf"
        );
        assert!(window_switch.get_menu().items()[1].borrow().selected());
    }
}
