//! `IMOD/Etomo/src/etomo/ui/swing/WindowSwitch.java`.
//!
//! Keeps the main frame's Window menu and its tabbed pane (one tab per open
//! manager, each with a close button) in step.  With one manager the main
//! frame shows its main panel directly; with several it shows the tabbed pane
//! with the selected manager's main panel on the selected tab.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::check_box_menu_item::CheckBoxMenuItem;
use super::ebutton::Ebutton;
use super::main_panel::MainPanelVirtual;
use super::menu::Menu;
use super::tabbed_pane::TabbedPane;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ChangeEvent, ChangeListener, JComponent,
};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::util::unique_hashed_array::UniqueHashedArray;
use crate::imod::etomo::util::unique_key::UniqueKey;

/// Java private static final `menuItemDividerChar`.
const MENU_ITEM_DIVIDER_CHAR: char = ':';
/// Java private static final `menuItemDivider = menuItemDividerChar + " "`.
const MENU_ITEM_DIVIDER: &str = ": ";

/// Java `public class WindowSwitch`.
pub struct WindowSwitch {
    /// Java private `Menu menu = new Menu("Window")`.
    menu: Rc<Menu>,
    /// Java private `UniqueHashedArray<TabItems> tabList = null`.
    tab_list: RefCell<Option<UniqueHashedArray<Rc<TabItems>>>>,
    /// Java private `TabbedPane tabbedPane = null`.
    tabbed_pane: RefCell<Option<Rc<TabbedPane>>>,
    /// Java private `MenuActionListener menuActionListener`.
    menu_action_listener: ActionListener,
    /// Java private `TabChangeListener tabChangeListener`.
    tab_change_listener: ChangeListener,
}

impl WindowSwitch {
    /// Java package-private `WindowSwitch()`.
    pub fn new() -> Rc<WindowSwitch> {
        Rc::new_cyclic(|self_ref: &Weak<WindowSwitch>| {
            // MenuActionListener
            let adaptee = self_ref.clone();
            let menu_action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.menu_action(event);
                }
            });
            // TabChangeListener
            let adaptee = self_ref.clone();
            let tab_change_listener: ChangeListener = Rc::new(move |event: &ChangeEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.tab_changed(event);
                }
            });
            WindowSwitch {
                menu: Menu::new("Window"),
                tab_list: RefCell::new(None),
                tabbed_pane: RefCell::new(None),
                menu_action_listener,
                tab_change_listener,
            }
        })
    }

    /// Java package-private `add(BaseManager, AxisID, UniqueKey)`.  Add a
    /// controller: add a menu item to the menu list, add the controller's
    /// mainPanel to the mainPanelList, add the menu item to the menu.
    pub fn add(
        &self,
        manager: Option<&'static dyn BaseManager>,
        axis_id: Option<AxisID>,
        manager_key: Option<&UniqueKey>,
    ) {
        let Some(manager_key) = manager_key else {
            return;
        };
        if self.tab_list.borrow().is_none() {
            *self.tab_list.borrow_mut() = Some(UniqueHashedArray::new());
            *self.tabbed_pane.borrow_mut() = Some(TabbedPane::new());
        }
        let menu_item = CheckBoxMenuItem::new_void();
        menu_item
            .get_component()
            .add_action_listener(self.menu_action_listener.clone());
        let index = self
            .tab_list
            .borrow()
            .as_ref()
            .map_or(0, |list| list.size());
        menu_item.set_text(Some(&format!(
            "{}{}{}",
            index + 1,
            MENU_ITEM_DIVIDER,
            manager_key.get_name()
        )));
        menu_item.get_component().set_visible(true);
        self.menu.add(&menu_item.get_component());
        // Upstream bug fixed (WindowSwitch.java:131): Java calls
        // manager.getMainPanel() on a null manager (NullPointerException); the
        // tab gets no main panel then.
        let main_panel = manager.and_then(|manager| manager.get_main_panel());
        let close_button = self.create_close_button(axis_id, manager_key);
        let tab_items = Rc::new(TabItems::new(menu_item, main_panel, close_button));
        if let Some(tab_list) = self.tab_list.borrow_mut().as_mut() {
            let _ = tab_list.add(manager_key.clone(), tab_items);
        }
    }

    /// Java package-private `rename(UniqueKey, UniqueKey)`.  Rename a window.
    /// Change the menu item, rekey the menuList and the mainPanelList.
    pub fn rename(&self, old_manager_key: Option<&UniqueKey>, new_manager_key: Option<&UniqueKey>) {
        let (Some(old_manager_key), Some(new_manager_key)) = (old_manager_key, new_manager_key)
        else {
            return;
        };
        if self.tab_list.borrow().is_none() {
            return;
        }
        let tab_items = self
            .tab_list
            .borrow()
            .as_ref()
            .and_then(|list| list.get(old_manager_key).cloned());
        let Some(tab_items) = tab_items else {
            return;
        };
        let menu_item = tab_items.get_menu_item();
        let index = self
            .tab_list
            .borrow()
            .as_ref()
            .and_then(|list| list.get_index(old_manager_key));
        let index = index.map_or(-1, |index| index as i64);
        menu_item.set_text(Some(&format!(
            "{}{}{}",
            index + 1,
            MENU_ITEM_DIVIDER,
            new_manager_key.get_name()
        )));
        let size = {
            let mut tab_list = self.tab_list.borrow_mut();
            let tab_list = tab_list.as_mut().unwrap();
            tab_list.rekey(old_manager_key, new_manager_key.clone());
            tab_list.size()
        };
        let tabbed_pane = self.tabbed_pane.borrow().clone();
        if size > 1
            && let Some(tabbed_pane) = tabbed_pane
            && index >= 0
            && tabbed_pane.get_component().get_tab_count() as i64 > index
        {
            tabbed_pane.set_title_at(index as usize, new_manager_key.get_name());
        }
        let _ = etomo_director::INSTANCE.set_current_manager_unique_key(Some(new_manager_key));
    }

    /// Java package-private `remove(UniqueKey)`.  Remove a window.  Removes the
    /// associated menuItem from the menu and from menuList, and the associated
    /// mainPanel from mainPanelList.
    pub fn remove(&self, manager_key: Option<&UniqueKey>) {
        let Some(manager_key) = manager_key else {
            return;
        };
        if self.tab_list.borrow().is_none() {
            return;
        }
        let tab_items = self
            .tab_list
            .borrow()
            .as_ref()
            .and_then(|list| list.get(manager_key).cloned());
        let Some(tab_items) = tab_items else {
            return;
        };
        let menu_item = tab_items.get_menu_item();
        let _index = self
            .tab_list
            .borrow()
            .as_ref()
            .and_then(|list| list.get_index(manager_key));
        self.menu.remove(&menu_item.get_component());
        if let Some(tab_list) = self.tab_list.borrow_mut().as_mut() {
            tab_list.remove(manager_key);
        }
        self.renumber_menu();
    }

    /// Java private `renumberMenu()`.  Renumbers the menu's displayed text.
    /// Used when a window is removed.
    fn renumber_menu(&self) {
        let size = match self.tab_list.borrow().as_ref() {
            None => return,
            Some(tab_list) => tab_list.size(),
        };
        for i in 0..size {
            let tab_items = self
                .tab_list
                .borrow()
                .as_ref()
                .and_then(|list| list.get_at(i).cloned());
            let Some(tab_items) = tab_items else {
                continue;
            };
            let menu_item = tab_items.get_menu_item();
            let text = menu_item.get_component().get_text();
            // text.substring(text.indexOf(menuItemDividerChar)): the whole text
            // when there is no divider is Java's substring(-1), which throws;
            // every item's text is built with the divider.
            let rest = text
                .find(MENU_ITEM_DIVIDER_CHAR)
                .map_or(text.as_str(), |position| &text[position..]);
            menu_item.set_text(Some(&format!("{}{}", i + 1, rest)));
        }
    }

    /// Java package-private `getMenu()`.  Returns the menu.
    pub fn get_menu(&self) -> Rc<Menu> {
        self.menu.clone()
    }

    /// Java package-private `getPanel(UniqueKey)`.  Returns the mainPanel
    /// associated with key, if there is only one window.  For multiple
    /// windows, returns a tabbed pane, with the mainPanel on the selected tab.
    pub fn get_panel(&self, manager_key: Option<&UniqueKey>) -> Option<Rc<JComponent>> {
        let size = self
            .tab_list
            .borrow()
            .as_ref()
            .map_or(0, |list| list.size());
        let Some(manager_key) = manager_key else {
            return None;
        };
        if self.tab_list.borrow().is_none() || size == 0 {
            return None;
        }
        if size == 1 {
            let tab_items = self
                .tab_list
                .borrow()
                .as_ref()
                .and_then(|list| list.get(manager_key).cloned());
            let tab_items = tab_items?;
            return tab_items
                .get_main_panel()
                .map(|panel| panel.main_panel().get_component());
        }
        let index = self
            .tab_list
            .borrow()
            .as_ref()
            .and_then(|list| list.get_index(manager_key))
            .map_or(-1, |index| index as i32);
        self.set_tabs(index, manager_key);
        self.tabbed_pane
            .borrow()
            .as_ref()
            .map(|tabbed_pane| tabbed_pane.get_component())
    }

    /// Java package-private `selectWindow(UniqueKey, boolean)`.  Allows the
    /// program to select a window.
    pub fn select_window(&self, manager_key: Option<&UniqueKey>, new_window: bool) {
        let _ = new_window;
        let Some(manager_key) = manager_key else {
            return;
        };
        let index = match self.tab_list.borrow().as_ref() {
            None => return,
            Some(tab_list) => tab_list
                .get_index(manager_key)
                .map_or(-1, |index| index as i32),
        };
        self.select_menu_item(index);
    }

    /// Java private `selectMenuItem(int)`.  Selects a menu item at index.
    /// Unselects all other menu items.  Index starts from zero.
    fn select_menu_item(&self, index: i32) {
        let size = match self.tab_list.borrow().as_ref() {
            None => return,
            Some(tab_list) => tab_list.size(),
        };
        for i in 0..size {
            let tab_items = self
                .tab_list
                .borrow()
                .as_ref()
                .and_then(|list| list.get_at(i).cloned());
            let Some(tab_items) = tab_items else {
                continue;
            };
            let menu_item = tab_items.get_menu_item().get_component();
            if i as i32 == index {
                menu_item.set_selected(true);
            } else {
                menu_item.set_selected(false);
            }
        }
    }

    /// Java private `setTabs(int, UniqueKey)`.  Sets up the tabbed pane:
    /// remove the change listener (it responds to changes caused by the
    /// program), remove everything on the pane, add the tabs placing the
    /// selected mainPanel on the associated tab, select the selected tab, and
    /// add the change listener back.
    fn set_tabs(&self, selected_tab_index: i32, manager_key: &UniqueKey) {
        let _ = manager_key;
        let size = match self.tab_list.borrow().as_ref() {
            None => return,
            Some(tab_list) => tab_list.size(),
        };
        let Some(tabbed_pane) = self.tabbed_pane.borrow().clone() else {
            return;
        };
        let pane = tabbed_pane.get_component();
        // The MainPanel can't always measure its display state accurately when
        // it is displayed on a tab.  Saving the display state allow MainPanel to
        // display correctly when it is brought up again.
        let old_index = pane.get_selected_tab();
        if old_index != -1 {
            let tab_items = self
                .tab_list
                .borrow()
                .as_ref()
                .and_then(|list| list.get_at(old_index as usize).cloned());
            if let Some(tab_items) = tab_items {
                let old_main_panel = tab_items.get_main_panel();
                if let Some(old_main_panel) = old_main_panel {
                    old_main_panel.main_panel().save_display_state();
                }
            }
        }
        pane.remove_change_listener(&self.tab_change_listener);
        pane.remove_all();
        if size < 2 {
            return;
        }
        for i in 0..size {
            let tab_items = self
                .tab_list
                .borrow()
                .as_ref()
                .and_then(|list| list.get_at(i).cloned());
            let Some(tab_items) = tab_items else {
                continue;
            };
            let text = tab_items.get_menu_item().get_component().get_text();
            let tab_name = text
                .find(MENU_ITEM_DIVIDER)
                .map_or(text.as_str(), |position| {
                    &text[position + MENU_ITEM_DIVIDER.len()..]
                })
                .to_owned();
            if i as i32 == selected_tab_index {
                match tab_items.get_main_panel() {
                    Some(main_panel) => tabbed_pane.add_tab_string_component(
                        &tab_name,
                        &main_panel.main_panel().get_component(),
                    ),
                    // A tab with a null component: jdk.rs tabs are children, so
                    // an empty panel stands for the missing main panel.
                    None => {
                        tabbed_pane.add_tab_string_component(&tab_name, &JComponent::new_panel())
                    }
                }
                self.add_close_button_to_tab(&tab_items.get_close_button(), &tab_name, i);
            } else {
                let place_holder = JComponent::new_label("");
                place_holder.set_visible(false);
                tabbed_pane.add_tab_string_component(&tab_name, &place_holder);
                self.add_close_button_to_tab(&tab_items.get_close_button(), &tab_name, i);
            }
        }
        pane.set_selected_tab(selected_tab_index);
        pane.add_change_listener(self.tab_change_listener.clone());
    }

    /// Java private `createCloseButton(AxisID, UniqueKey)`.  If it doesn't
    /// already exist, create a close button and add an action listener.
    fn create_close_button(&self, axis_id: Option<AxisID>, manager_key: &UniqueKey) -> Rc<Ebutton> {
        let tab_items = self
            .tab_list
            .borrow()
            .as_ref()
            .and_then(|list| list.get(manager_key).cloned());
        let mut btn_close = None;
        if let Some(tab_items) = tab_items {
            btn_close = Some(tab_items.get_close_button());
        }
        match btn_close {
            Some(btn_close) => btn_close,
            None => {
                let btn_close = Ebutton::get_close_instance();
                let close_action_listener = ui_harness::with(|harness| {
                    harness.get_close_action_listener(axis_id, manager_key.clone())
                });
                let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                    close_action_listener.action_performed(event)
                });
                btn_close.add_action_listener_action_listener(Some(listener));
                btn_close
            }
        }
    }

    /// Java private `addCloseButtonToTab(Ebutton, String, int)`.
    fn add_close_button_to_tab(&self, btn_close: &Rc<Ebutton>, name: &str, tab_index: usize) {
        let pnl_tab = JComponent::new_panel();
        // Swing layout: new JPanel(new GridBagLayout()); pnlTab.setOpaque(false).
        let lbl_title = JComponent::new_label(name);
        //
        // Swing layout: GridBagConstraints gridx 0, gridy 0, weightx 1.
        pnl_tab.add(&lbl_title);
        // Swing layout: gbc.gridx++; pnlTab.add(Box.createRigidArea(FixedDim.x25_y0)).
        // Swing layout: gbc.gridx++; gbc.weightx = 0.
        pnl_tab.add(&btn_close.get_component());
        //
        if let Some(tabbed_pane) = self.tabbed_pane.borrow().as_ref() {
            tabbed_pane
                .get_component()
                .set_tab_component_at(tab_index, &pnl_tab);
        }
    }

    /// Java package-private `menuAction(ActionEvent)`.  Open the specified
    /// window when the user chooses a window menu item.
    pub fn menu_action(&self, event: &ActionEvent) {
        let menu_choice = event.get_action_command().unwrap_or("");
        let number = menu_choice
            .find(MENU_ITEM_DIVIDER_CHAR)
            .and_then(|position| menu_choice[..position].parse::<i32>().ok());
        // Upstream bug fixed (WindowSwitch.java:397): Java's parseInt/substring
        // throw on a command without a number; the event is ignored instead.
        let Some(number) = number else {
            return;
        };
        let new_index = number - 1;
        self.select_menu_item(new_index);
        let key = self.tab_list.borrow().as_ref().and_then(|list| {
            usize::try_from(new_index)
                .ok()
                .and_then(|i| list.get_key(i).cloned())
        });
        let _ = etomo_director::INSTANCE.set_current_manager_unique_key(key.as_ref());
    }

    /// Java package-private `tabChanged(ChangeEvent)`.  Open the specified
    /// window when the user chooses a tab.
    pub fn tab_changed(&self, event: &ChangeEvent) {
        let _ = event;
        let new_index = self
            .tabbed_pane
            .borrow()
            .as_ref()
            .map_or(-1, |tabbed_pane| {
                tabbed_pane.get_component().get_selected_tab()
            });
        self.select_menu_item(new_index);
        // Upstream bug fixed (WindowSwitch.java:408): getKey(-1) throws in
        // Java when no tab is selected; a null key is passed instead, which
        // setCurrentManager ignores.
        let key = self.tab_list.borrow().as_ref().and_then(|list| {
            usize::try_from(new_index)
                .ok()
                .and_then(|i| list.get_key(i).cloned())
        });
        let _ = etomo_director::INSTANCE.set_current_manager_unique_key(key.as_ref());
    }
}

/// Java private static final `TabItems`.
struct TabItems {
    /// Java private final `JCheckBoxMenuItem menuItem`.
    menu_item: Rc<CheckBoxMenuItem>,
    /// Java private final `MainPanel mainPanel`.
    main_panel: Option<Rc<dyn MainPanelVirtual>>,
    /// Java private final `Ebutton closeButton`.
    close_button: Rc<Ebutton>,
}

impl TabItems {
    /// Java `TabItems(JCheckBoxMenuItem, MainPanel, Ebutton)`.
    fn new(
        menu_item: Rc<CheckBoxMenuItem>,
        main_panel: Option<Rc<dyn MainPanelVirtual>>,
        close_button: Rc<Ebutton>,
    ) -> TabItems {
        TabItems {
            menu_item,
            main_panel,
            close_button,
        }
    }

    /// Java private `getMenuItem()`.
    fn get_menu_item(&self) -> Rc<CheckBoxMenuItem> {
        self.menu_item.clone()
    }

    /// Java private `getMainPanel()`.
    fn get_main_panel(&self) -> Option<Rc<dyn MainPanelVirtual>> {
        self.main_panel.clone()
    }

    /// Java private `getCloseButton()`.
    fn get_close_button(&self) -> Rc<Ebutton> {
        self.close_button.clone()
    }
}
