//! Stand-in for the `javax.swing` / `java.awt` component classes eTomo builds
//! its dialogs from.
//!
//! **Not a translated unit.**  eTomo's dialogs are Swing component trees:
//! each widget class (`MultiLineButton`, `CheckBox`, `LabeledTextField`, ...)
//! wraps or extends a `JButton`, `JCheckBox`, `JTextField`, ... and the dialog
//! logic reads and writes those components' state (`getText`, `isSelected`,
//! `setEnabled`) and reacts to their listeners.  The translated widget and
//! dialog classes need that state and that event plumbing; they do not need
//! Swing's painting or layout, which the Slint files under `gui/` draw.
//!
//! So this module models exactly the component *state* and *events* the Java
//! relies on, as one node type ([`JComponent`]) shared by every kind:
//!
//! * name (`setName`, which eTomo's uitest naming drives: `bn.done`,
//!   `cb.use-fixed-stack`, ...), text, tooltip, enabled, visible, selected,
//!   editable, action command;
//! * the child list (`add`, `remove`, `removeAll`) so a driver can find a
//!   component by name under a root the way Java's `uitest` does;
//! * listeners: `ActionListener`, `ItemListener`, `ChangeListener` and
//!   `DocumentListener`, fired when Swing fires them (`doClick` fires item then
//!   action; `setSelected` fires item/change but not action; a text change
//!   fires the document listeners; `JSpinner.setValue` fires change;
//!   `JComboBox.setSelectedIndex` fires item then action);
//! * `ButtonGroup` exclusivity;
//! * the models of `JSpinner` (`SpinnerNumberModel`), `JComboBox` and
//!   `JTabbedPane`.
//!
//! Components are EDT objects (`Rc`, interior mutability), and every method
//! takes `&self`, so a listener may call back into the component that fired it
//! (Swing allows that) without a `RefCell` double borrow.  Listener closures
//! should capture a `Weak` to their owner to avoid `Rc` cycles.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

/// What a `JComponent` node stands for.  Only used to answer kind-specific
/// questions (a driver asking for "the button named X") and for debugging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentKind {
    Panel,
    Label,
    Button,
    ToggleButton,
    CheckBox,
    RadioButton,
    TextField,
    TextArea,
    Spinner,
    ComboBox,
    TabbedPane,
    ScrollPane,
    MenuItem,
    CheckBoxMenuItem,
    Menu,
    PopupMenu,
    /// `JPopupMenu.Separator` (`JSeparator`).
    Separator,
    ProgressBar,
    Other,
}

/// `java.awt.event.ActionEvent`: the source and its action command.
pub struct ActionEvent {
    pub source: Rc<JComponent>,
    pub action_command: Option<String>,
}

impl ActionEvent {
    /// Java `getActionCommand`.
    pub fn get_action_command(&self) -> Option<&str> {
        self.action_command.as_deref()
    }
    /// Java `getSource`.
    pub fn get_source(&self) -> &Rc<JComponent> {
        &self.source
    }
}

/// `java.awt.event.ItemEvent.SELECTED` / `DESELECTED`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemState {
    Selected,
    Deselected,
}

/// `java.awt.event.ItemEvent`.
pub struct ItemEvent {
    pub source: Rc<JComponent>,
    pub state_change: ItemState,
}

/// `javax.swing.event.ChangeEvent`.
pub struct ChangeEvent {
    pub source: Rc<JComponent>,
}

/// `javax.swing.event.DocumentEvent`, reduced to its source.
pub struct DocumentEvent {
    pub source: Rc<JComponent>,
}

pub type ActionListener = Rc<dyn Fn(&ActionEvent)>;
pub type ItemListener = Rc<dyn Fn(&ItemEvent)>;
pub type ChangeListener = Rc<dyn Fn(&ChangeEvent)>;
pub type DocumentListener = Rc<dyn Fn(&DocumentEvent)>;
/// `java.awt.event.FocusEvent`, reduced to its source and whether focus was
/// gained (`FOCUS_GAINED`) or lost (`FOCUS_LOST`).
pub struct FocusEvent {
    pub source: Rc<JComponent>,
    pub gained: bool,
}
/// `java.awt.event.FocusListener`: one closure stands for both
/// `focusGained` and `focusLost`; it reads `FocusEvent::gained`.
pub type FocusListener = Rc<dyn Fn(&FocusEvent)>;
/// `java.beans.PropertyChangeListener`, called with the property name and the
/// new value.  Only the boolean `enabled` property is fired by this stand-in.
pub type PropertyChangeListener = Rc<dyn Fn(&str, bool)>;

/// `javax.swing.SpinnerNumberModel` (numbers as `f64`; eTomo's spinners hold
/// integers or doubles and read them through `Number`).
#[derive(Clone, Debug, PartialEq)]
pub struct SpinnerNumberModel {
    pub value: f64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub step_size: f64,
    /// True when the model was built from `int`s, so `getValue` is an
    /// `Integer`.
    pub integer: bool,
}

impl SpinnerNumberModel {
    /// Java `SpinnerNumberModel(int value, int minimum, int maximum, int stepSize)`.
    pub fn new_int(value: i32, minimum: i32, maximum: i32, step_size: i32) -> Self {
        SpinnerNumberModel {
            value: value as f64,
            minimum: Some(minimum as f64),
            maximum: Some(maximum as f64),
            step_size: step_size as f64,
            integer: true,
        }
    }
    /// Java `SpinnerNumberModel(double value, double minimum, double maximum, double stepSize)`.
    pub fn new_double(value: f64, minimum: f64, maximum: f64, step_size: f64) -> Self {
        SpinnerNumberModel {
            value,
            minimum: Some(minimum),
            maximum: Some(maximum),
            step_size,
            integer: false,
        }
    }
}

/// `javax.swing.ButtonGroup`: at most one member selected.
#[derive(Default)]
pub struct ButtonGroup {
    buttons: RefCell<Vec<Weak<JComponent>>>,
}

impl ButtonGroup {
    /// Java `new ButtonGroup()`.
    pub fn new() -> Rc<ButtonGroup> {
        Rc::new(ButtonGroup::default())
    }
    /// Java `add(AbstractButton)`.
    pub fn add(self: &Rc<Self>, button: &Rc<JComponent>) {
        self.buttons.borrow_mut().push(Rc::downgrade(button));
        *button.group.borrow_mut() = Some(self.clone());
        // Swing: adding a selected button to a group that already has a
        // selection deselects the added one.
        if button.selected.get() {
            let others_selected = self.buttons.borrow().iter().any(|member| {
                member
                    .upgrade()
                    .is_some_and(|member| !Rc::ptr_eq(&member, button) && member.selected.get())
            });
            if others_selected {
                button.selected.set(false);
            }
        }
    }
    /// Java `remove(AbstractButton)`.
    pub fn remove(&self, button: &Rc<JComponent>) {
        self.buttons
            .borrow_mut()
            .retain(|member| member.upgrade().is_some_and(|m| !Rc::ptr_eq(&m, button)));
        *button.group.borrow_mut() = None;
    }
    /// Java `clearSelection`.
    pub fn clear_selection(&self) {
        let members: Vec<_> = self
            .buttons
            .borrow()
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for member in members {
            if member.selected.get() {
                member.selected.set(false);
                let model = member.model.borrow().clone();
                if let Some(model) = model {
                    model.set_selected(false);
                }
                member.fire_item_state_changed(ItemState::Deselected);
                member.fire_state_changed();
            }
        }
    }
    /// Java `getButtonCount`.
    pub fn get_button_count(&self) -> usize {
        self.buttons.borrow().len()
    }
    /// The members still alive, in insertion order (Java `getElements`).
    pub fn get_elements(&self) -> Vec<Rc<JComponent>> {
        self.buttons
            .borrow()
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }
    /// The selected member, if any (Java `getSelection`).
    pub fn get_selection(&self) -> Option<Rc<JComponent>> {
        self.get_elements().into_iter().find(|b| b.selected.get())
    }
}

/// `java.awt.Color`, as RGB.  Only the values eTomo compares or copies matter;
/// painting is not modelled.
pub type Color = (u8, u8, u8);

/// `java.awt.Dimension`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dimension {
    pub width: i32,
    pub height: i32,
}

/// `java.awt.Point`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

/// `javax.swing.border.TitledBorder`: only the title is modelled (it names
/// panels for the uitest).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TitledBorder {
    title: RefCell<Option<String>>,
}

impl TitledBorder {
    /// Java `new TitledBorder(String)` / `BorderFactory.createTitledBorder(String)`.
    pub fn new(title: Option<&str>) -> TitledBorder {
        TitledBorder {
            title: RefCell::new(title.map(str::to_owned)),
        }
    }
    /// Java `getTitle()`.
    pub fn get_title(&self) -> Option<String> {
        self.title.borrow().clone()
    }
    /// Java `setTitle(String)`.
    pub fn set_title(&self, title: Option<&str>) {
        *self.title.borrow_mut() = title.map(str::to_owned);
    }
}

/// `java.awt.FontMetrics`.  Text is never rendered by the stand-in, so a
/// fixed-pitch metric stands in: it only feeds layout decisions (preferred
/// widths, which part of a long path to show).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FontMetrics {
    /// Width of one character, in pixels.
    pub char_width: i32,
    /// Line height, in pixels.
    pub height: i32,
}

impl Default for FontMetrics {
    fn default() -> Self {
        FontMetrics {
            char_width: 7,
            height: 16,
        }
    }
}

impl FontMetrics {
    /// Java `stringWidth(String)`.
    pub fn string_width(&self, string: &str) -> i32 {
        string.chars().count() as i32 * self.char_width
    }
    /// Java `charsWidth(char[], int, int)`.
    pub fn chars_width(&self, _data: &[char], _off: usize, len: usize) -> i32 {
        len as i32 * self.char_width
    }
    /// Java `charWidth(char)`.
    pub fn char_width(&self, _c: char) -> i32 {
        self.char_width
    }
    /// Java `getHeight()`.
    pub fn get_height(&self) -> i32 {
        self.height
    }
}

/// `javax.swing.filechooser.FileFilter`.
pub trait FileFilter {
    /// Java `accept(File)`.
    fn accept(&self, file: &std::path::Path) -> bool;
    /// Java `getDescription()`.
    fn get_description(&self) -> Option<String>;
}

/// `javax.swing.JFileChooser`: the stand-in is eTomo's own `FileChooser`, whose
/// dialog boundary (`file_chooser::set_dialog_responder`) answers it.
pub type JFileChooser = crate::imod::etomo::ui::swing::file_chooser::FileChooser;

/// `java.awt.event.MouseEvent`, reduced to what eTomo reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MouseEvent {
    /// Java `getButton()`: 1 left, 2 middle, 3 right.
    pub button: i32,
    /// Java `isPopupTrigger()`.
    pub popup_trigger: bool,
    pub x: i32,
    pub y: i32,
}

/// `Toolkit.getDefaultToolkit().getScreenSize()`.  No display is modelled; the
/// size of the screen the reference Java run used (Xvfb 1600x1200) stands in.
pub fn get_screen_size() -> Dimension {
    Dimension {
        width: 1600,
        height: 1200,
    }
}

/// `SwingUtilities.isRightMouseButton(MouseEvent)`.
pub fn is_right_mouse_button(event: &MouseEvent) -> bool {
    event.button == 3
}

/// `java.awt.event.MouseListener`.  Mouse events are only delivered by a
/// driver (the context popups); every method defaults to Java's empty adapter.
pub trait MouseListener {
    fn mouse_clicked(&self, _event: &MouseEvent) {}
    fn mouse_pressed(&self, _event: &MouseEvent) {}
    fn mouse_released(&self, _event: &MouseEvent) {}
    fn mouse_entered(&self, _event: &MouseEvent) {}
    fn mouse_exited(&self, _event: &MouseEvent) {}
}

/// `javax.swing.ButtonModel` subclass hook: the selected state lives in the
/// `JComponent`; a custom model is told after the component has updated it
/// (what `ToggleButtonModel.setSelected` does before an override continues).
pub trait ButtonModel {
    /// The override body run after `super.setSelected(selected)`.
    fn set_selected(&self, selected: bool);
    fn as_any(&self) -> &dyn std::any::Any;
}

/// `javax.swing.Timer`: fires its listener on the EDT every `delay`
/// milliseconds while running.
pub struct Timer {
    delay_ms: u64,
    listener: Rc<dyn Fn()>,
    /// Generation of the running schedule; bumped by stop/restart so an old
    /// ticker thread stops posting.
    generation: Rc<Cell<u64>>,
    running: Cell<bool>,
    id: u64,
}

thread_local! {
    static TIMERS: RefCell<std::collections::HashMap<u64, (Rc<dyn Fn()>, Rc<Cell<u64>>)>> =
        RefCell::new(std::collections::HashMap::new());
    static NEXT_TIMER_ID: Cell<u64> = const { Cell::new(1) };
}

impl Timer {
    /// Java `new Timer(int delay, ActionListener listener)`.
    pub fn new(delay_ms: i32, listener: Rc<dyn Fn()>) -> Rc<Timer> {
        let id = NEXT_TIMER_ID.with(|next| {
            let id = next.get();
            next.set(id + 1);
            id
        });
        let generation = Rc::new(Cell::new(0));
        TIMERS.with(|timers| {
            timers
                .borrow_mut()
                .insert(id, (listener.clone(), generation.clone()))
        });
        Rc::new(Timer {
            delay_ms: delay_ms.max(0) as u64,
            listener,
            generation,
            running: Cell::new(false),
            id,
        })
    }
    /// Java `start()`.
    pub fn start(&self) {
        if self.running.get() {
            return;
        }
        self.running.set(true);
        let generation = self.generation.get() + 1;
        self.generation.set(generation);
        let id = self.id;
        let delay = self.delay_ms;
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(delay));
                let alive = crate::imod::etomo::util::event_queue::invoke_and_wait(move || {
                    TIMERS.with(|timers| {
                        let entry = timers.borrow().get(&id).cloned();
                        match entry {
                            Some((listener, current)) if current.get() == generation => {
                                listener();
                                true
                            }
                            _ => false,
                        }
                    })
                });
                if !alive {
                    break;
                }
            }
        });
    }
    /// Java `stop()`.
    pub fn stop(&self) {
        self.running.set(false);
        self.generation.set(self.generation.get() + 1);
    }
    /// Java `restart()`.
    pub fn restart(&self) {
        self.stop();
        self.start();
    }
    /// Java `isRunning()`.
    pub fn is_running(&self) -> bool {
        self.running.get()
    }
    /// The listener (Java `getActionListeners()[0]`).
    pub fn get_listener(&self) -> Rc<dyn Fn()> {
        self.listener.clone()
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let id = self.id;
        let _ = TIMERS.try_with(|timers| timers.borrow_mut().remove(&id));
    }
}

/// One Swing component.  See the module documentation.
pub struct JComponent {
    kind: ComponentKind,
    name: RefCell<Option<String>>,
    text: RefCell<String>,
    tool_tip_text: RefCell<Option<String>>,
    enabled: Cell<bool>,
    visible: Cell<bool>,
    selected: Cell<bool>,
    editable: Cell<bool>,
    action_command: RefCell<Option<String>>,
    /// A `TitledBorder`'s title, where the Java sets one.
    border_title: RefCell<Option<String>>,
    children: RefCell<Vec<Rc<JComponent>>>,
    parent: RefCell<Weak<JComponent>>,
    /// The button's group.  Strong, as in Swing, where the button model
    /// references its `ButtonGroup`: a group built in a local variable stays
    /// alive as long as its buttons do.
    group: RefCell<Option<Rc<ButtonGroup>>>,
    action_listeners: RefCell<Vec<ActionListener>>,
    item_listeners: RefCell<Vec<ItemListener>>,
    change_listeners: RefCell<Vec<ChangeListener>>,
    document_listeners: RefCell<Vec<DocumentListener>>,
    spinner_model: RefCell<Option<SpinnerNumberModel>>,
    items: RefCell<Vec<Option<String>>>,
    selected_index: Cell<i32>,
    /// `DefaultComboBoxModel.selectedObject` when it is not one of the items:
    /// text entered in an editable `JComboBox` (`setSelectedItem`).
    entered_item: RefCell<Option<String>>,
    /// `JTabbedPane` tab titles; the tab components are `children`.
    tab_titles: RefCell<Vec<String>>,
    tab_enabled: RefCell<Vec<bool>>,
    /// `JProgressBar` / `JSlider` state.
    minimum: Cell<i32>,
    maximum: Cell<i32>,
    int_value: Cell<i32>,
    indeterminate: Cell<bool>,
    string: RefCell<Option<String>>,
    /// Foreground colour as RGB, where the Java changes it (highlighting).
    foreground: Cell<Option<(u8, u8, u8)>>,
    /// A custom `ButtonModel` (`setModel`).
    model: RefCell<Option<Rc<dyn ButtonModel>>>,
    /// `JTabbedPane.setTabComponentAt` components.
    tab_title_components: RefCell<Vec<Option<Rc<JComponent>>>>,
    /// `addMouseListener`.
    mouse_listeners: RefCell<Vec<Rc<dyn MouseListener>>>,
    /// A shown popup menu's invoker (`JPopupMenu.show`).
    popup_invoker: RefCell<Option<Weak<JComponent>>>,
    /// `addPropertyChangeListener(String, PropertyChangeListener)`.
    property_change_listeners: RefCell<Vec<(String, PropertyChangeListener)>>,
    /// `addFocusListener(FocusListener)`.
    focus_listeners: RefCell<Vec<FocusListener>>,
    /// `GridBagLayout.setConstraints(component, constraints)`: the constraints the
    /// component was laid out with in its `GridBagLayout` parent.
    layout_constraints: Cell<Option<GridBagConstraints>>,
    /// `getInputMap(...).put(keyStroke, key)` + `getActionMap().put(key, action)`: the
    /// component's key bindings, by key stroke (`"alt UP"`).
    key_bindings: RefCell<Vec<(String, Rc<dyn Fn()>)>>,
    /// `setFocusable(boolean)`.
    focusable: Cell<bool>,
}

impl JComponent {
    fn with_kind(kind: ComponentKind, text: &str) -> Rc<JComponent> {
        Rc::new(JComponent {
            kind,
            name: RefCell::new(None),
            text: RefCell::new(text.to_owned()),
            tool_tip_text: RefCell::new(None),
            enabled: Cell::new(true),
            visible: Cell::new(true),
            selected: Cell::new(false),
            editable: Cell::new(true),
            action_command: RefCell::new(None),
            border_title: RefCell::new(None),
            children: RefCell::new(Vec::new()),
            parent: RefCell::new(Weak::new()),
            group: RefCell::new(None),
            action_listeners: RefCell::new(Vec::new()),
            item_listeners: RefCell::new(Vec::new()),
            change_listeners: RefCell::new(Vec::new()),
            document_listeners: RefCell::new(Vec::new()),
            spinner_model: RefCell::new(None),
            items: RefCell::new(Vec::new()),
            selected_index: Cell::new(-1),
            entered_item: RefCell::new(None),
            tab_titles: RefCell::new(Vec::new()),
            tab_enabled: RefCell::new(Vec::new()),
            minimum: Cell::new(0),
            maximum: Cell::new(100),
            int_value: Cell::new(0),
            indeterminate: Cell::new(false),
            string: RefCell::new(None),
            foreground: Cell::new(None),
            model: RefCell::new(None),
            tab_title_components: RefCell::new(Vec::new()),
            mouse_listeners: RefCell::new(Vec::new()),
            popup_invoker: RefCell::new(None),
            property_change_listeners: RefCell::new(Vec::new()),
            focus_listeners: RefCell::new(Vec::new()),
            layout_constraints: Cell::new(None),
            key_bindings: RefCell::new(Vec::new()),
            focusable: Cell::new(true),
        })
    }

    /// `new JPanel()` (also `Box`, `JScrollPane` content, containers).
    pub fn new_panel() -> Rc<JComponent> {
        Self::with_kind(ComponentKind::Panel, "")
    }
    /// `new JLabel(text)`.
    pub fn new_label(text: &str) -> Rc<JComponent> {
        Self::with_kind(ComponentKind::Label, text)
    }
    /// `new JButton(text)`.
    pub fn new_button(text: &str) -> Rc<JComponent> {
        Self::with_kind(ComponentKind::Button, text)
    }
    /// `new JToggleButton(text)`.
    pub fn new_toggle_button(text: &str) -> Rc<JComponent> {
        Self::with_kind(ComponentKind::ToggleButton, text)
    }
    /// `new JCheckBox(text)`.
    pub fn new_check_box(text: &str) -> Rc<JComponent> {
        Self::with_kind(ComponentKind::CheckBox, text)
    }
    /// `new JRadioButton(text)`.
    pub fn new_radio_button(text: &str) -> Rc<JComponent> {
        Self::with_kind(ComponentKind::RadioButton, text)
    }
    /// `new JTextField()`.
    pub fn new_text_field() -> Rc<JComponent> {
        Self::with_kind(ComponentKind::TextField, "")
    }
    /// `new JTextArea()`.
    pub fn new_text_area() -> Rc<JComponent> {
        Self::with_kind(ComponentKind::TextArea, "")
    }
    /// `new JSpinner(model)`.
    pub fn new_spinner(model: SpinnerNumberModel) -> Rc<JComponent> {
        let spinner = Self::with_kind(ComponentKind::Spinner, "");
        *spinner.spinner_model.borrow_mut() = Some(model);
        spinner
    }
    /// `new JComboBox()`.
    pub fn new_combo_box() -> Rc<JComponent> {
        let combo_box = Self::with_kind(ComponentKind::ComboBox, "");
        // `JComboBox.isEditable` is false until `setEditable(true)`.
        combo_box.editable.set(false);
        combo_box
    }
    /// `new JTabbedPane()`.
    pub fn new_tabbed_pane() -> Rc<JComponent> {
        Self::with_kind(ComponentKind::TabbedPane, "")
    }
    /// `new JScrollPane(view)`.
    pub fn new_scroll_pane(view: Option<&Rc<JComponent>>) -> Rc<JComponent> {
        let pane = Self::with_kind(ComponentKind::ScrollPane, "");
        if let Some(view) = view {
            pane.add(view);
        }
        pane
    }
    /// `new JMenuItem(text)`.
    pub fn new_menu_item(text: &str) -> Rc<JComponent> {
        Self::with_kind(ComponentKind::MenuItem, text)
    }
    /// `new JCheckBoxMenuItem(text)`.
    pub fn new_check_box_menu_item(text: &str) -> Rc<JComponent> {
        Self::with_kind(ComponentKind::CheckBoxMenuItem, text)
    }
    /// `new JMenu(text)`.
    pub fn new_menu(text: &str) -> Rc<JComponent> {
        Self::with_kind(ComponentKind::Menu, text)
    }
    /// `new JPopupMenu(label)`.  A popup menu is not visible until shown.
    pub fn new_popup_menu(label: &str) -> Rc<JComponent> {
        let popup_menu = Self::with_kind(ComponentKind::PopupMenu, label);
        popup_menu.visible.set(false);
        popup_menu
    }
    /// `new JPopupMenu.Separator()`.
    pub fn new_popup_separator() -> Rc<JComponent> {
        Self::with_kind(ComponentKind::Separator, "")
    }
    /// Java `JPopupMenu.show(Component invoker, int x, int y)`: the menu is
    /// visible over its invoker, at `(x, y)` in the invoker's coordinates; a
    /// driver finds its items by name.  Swing shows one popup menu at a time
    /// (`MenuSelectionManager`), so a popup already showing is hidden.
    pub fn show(self: &Rc<Self>, invoker: &Rc<JComponent>, x: i32, y: i32) {
        *self.popup_invoker.borrow_mut() = Some(Rc::downgrade(invoker));
        let previous =
            SHOWING_POPUP.with(|showing| showing.borrow_mut().replace((self.clone(), x, y)));
        if let Some((previous, _, _)) = previous
            && !Rc::ptr_eq(&previous, self)
        {
            previous.visible.set(false);
        }
        self.visible.set(true);
    }
    /// The component a popup menu was last shown over.
    pub fn get_invoker(&self) -> Option<Rc<JComponent>> {
        self.popup_invoker.borrow().as_ref().and_then(Weak::upgrade)
    }
    /// `new JProgressBar()`.
    pub fn new_progress_bar() -> Rc<JComponent> {
        Self::with_kind(ComponentKind::ProgressBar, "")
    }
    /// Any other component class the Java builds (icons, separators, glue).
    pub fn new_other() -> Rc<JComponent> {
        Self::with_kind(ComponentKind::Other, "")
    }

    pub fn kind(&self) -> ComponentKind {
        self.kind
    }

    // --- java.awt.Component ---

    /// Java `setName`.
    pub fn set_name(&self, name: Option<&str>) {
        *self.name.borrow_mut() = name.map(str::to_owned);
    }
    /// Java `getName`.
    pub fn get_name(&self) -> Option<String> {
        self.name.borrow().clone()
    }
    /// Java `setEnabled`.
    pub fn set_enabled(&self, enabled: bool) {
        let old = self.enabled.replace(enabled);
        // JComponent.setEnabled fires the bound "enabled" property on a change.
        if old != enabled {
            let listeners: Vec<_> = self.property_change_listeners.borrow().clone();
            for (name, listener) in listeners.iter().rev() {
                if name == "enabled" {
                    listener("enabled", enabled);
                }
            }
        }
    }
    /// Java `addPropertyChangeListener(String, PropertyChangeListener)`.
    pub fn add_property_change_listener(&self, name: &str, listener: PropertyChangeListener) {
        self.property_change_listeners
            .borrow_mut()
            .push((name.to_owned(), listener));
    }
    /// Java `removePropertyChangeListener(PropertyChangeListener)`, by identity.
    pub fn remove_property_change_listener(&self, listener: &PropertyChangeListener) {
        self.property_change_listeners
            .borrow_mut()
            .retain(|(_, l)| !Rc::ptr_eq(l, listener));
    }
    /// Java `isEnabled`.
    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }
    /// Java `setVisible`.
    pub fn set_visible(&self, visible: bool) {
        self.visible.set(visible);
        // A hidden popup menu leaves Swing's popup layer.
        if !visible && self.kind == ComponentKind::PopupMenu {
            SHOWING_POPUP.with(|showing| {
                let mut showing = showing.borrow_mut();
                if showing
                    .as_ref()
                    .is_some_and(|(popup, _, _)| std::ptr::eq(popup.as_ref(), self))
                {
                    *showing = None;
                }
            });
        }
    }
    /// Java `isVisible`.
    pub fn is_visible(&self) -> bool {
        self.visible.get()
    }
    /// Java `isShowing`, less the window: visible here and in every ancestor.
    /// A `JTabbedPane` shows only its selected tab's component
    /// (`BasicTabbedPaneUI` hides the others), so a tab component other than
    /// the selected one is not showing.
    pub fn is_showing(&self) -> bool {
        if !self.visible.get() {
            return false;
        }
        match self.parent.borrow().upgrade() {
            None => true,
            Some(parent) => {
                if parent.kind == ComponentKind::TabbedPane {
                    let selected = parent.selected_index.get();
                    let position = parent
                        .children
                        .borrow()
                        .iter()
                        .position(|c| std::ptr::eq(Rc::as_ptr(c), self));
                    if let Some(position) = position
                        && position as i32 != selected
                    {
                        return false;
                    }
                }
                parent.is_showing()
            }
        }
    }
    /// Java `setToolTipText`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        *self.tool_tip_text.borrow_mut() = text.map(str::to_owned);
    }
    /// Java `getToolTipText`.
    pub fn get_tool_tip_text(&self) -> Option<String> {
        self.tool_tip_text.borrow().clone()
    }
    /// Java `setForeground` (RGB), `None` for the look-and-feel default.
    pub fn set_foreground(&self, color: Option<(u8, u8, u8)>) {
        self.foreground.set(color);
    }
    /// Java `getForeground`.
    pub fn get_foreground(&self) -> Option<(u8, u8, u8)> {
        self.foreground.get()
    }
    /// Java `getFontMetrics(getFont())`.
    pub fn get_font_metrics(&self) -> FontMetrics {
        FontMetrics::default()
    }
    /// Java `AbstractButton.setModel(ButtonModel)`.
    pub fn set_model(&self, model: Option<Rc<dyn ButtonModel>>) {
        *self.model.borrow_mut() = model;
    }
    /// Java `AbstractButton.getModel()`: the custom model, if one was set.
    pub fn get_model(&self) -> Option<Rc<dyn ButtonModel>> {
        self.model.borrow().clone()
    }
    /// A `BorderFactory.createTitledBorder(title)` set on this component.
    pub fn set_border_title(&self, title: Option<&str>) {
        *self.border_title.borrow_mut() = title.map(str::to_owned);
    }
    pub fn get_border_title(&self) -> Option<String> {
        self.border_title.borrow().clone()
    }

    // --- java.awt.Container ---

    /// Java `add(Component)`.  A component has one parent: adding it again
    /// moves it, as in AWT.
    pub fn add(self: &Rc<Self>, child: &Rc<JComponent>) {
        if let Some(old) = child.parent.borrow().upgrade() {
            old.children.borrow_mut().retain(|c| !Rc::ptr_eq(c, child));
        }
        *child.parent.borrow_mut() = Rc::downgrade(self);
        self.children.borrow_mut().push(child.clone());
    }
    /// Java `remove(Component)`.
    pub fn remove(&self, child: &Rc<JComponent>) {
        self.children.borrow_mut().retain(|c| !Rc::ptr_eq(c, child));
        *child.parent.borrow_mut() = Weak::new();
    }
    /// Java `removeAll`.
    pub fn remove_all(&self) {
        let children = std::mem::take(&mut *self.children.borrow_mut());
        for child in children {
            *child.parent.borrow_mut() = Weak::new();
        }
        self.tab_titles.borrow_mut().clear();
        self.tab_enabled.borrow_mut().clear();
    }
    /// Java `getComponents`.
    pub fn get_components(&self) -> Vec<Rc<JComponent>> {
        self.children.borrow().clone()
    }
    /// Java `getComponentCount`.
    pub fn get_component_count(&self) -> usize {
        self.children.borrow().len()
    }
    /// Java `getParent`.
    pub fn get_parent(&self) -> Option<Rc<JComponent>> {
        self.parent.borrow().upgrade()
    }

    // --- AbstractButton / JLabel / JTextComponent text ---

    /// Java `setText`.  On a text component this fires the document listeners
    /// (Swing fires remove then insert; one event stands for both).
    pub fn set_text(self: &Rc<Self>, text: &str) {
        let changed = *self.text.borrow() != text;
        *self.text.borrow_mut() = text.to_owned();
        if changed
            && matches!(
                self.kind,
                ComponentKind::TextField | ComponentKind::TextArea
            )
        {
            self.fire_document_changed();
        }
    }
    /// Java `getText`.
    pub fn get_text(&self) -> String {
        self.text.borrow().clone()
    }
    /// Java `JTextComponent.setEditable`.
    pub fn set_editable(&self, editable: bool) {
        self.editable.set(editable);
    }
    /// Java `JTextComponent.isEditable`.
    pub fn is_editable(&self) -> bool {
        self.editable.get()
    }
    /// Java `JTextArea.append`.
    pub fn append(self: &Rc<Self>, text: &str) {
        self.text.borrow_mut().push_str(text);
        self.fire_document_changed();
    }

    // --- AbstractButton ---

    /// Java `setActionCommand`.
    pub fn set_action_command(&self, command: Option<&str>) {
        *self.action_command.borrow_mut() = command.map(str::to_owned);
    }
    /// Java `getActionCommand`: the text when no command was set.
    pub fn get_action_command(&self) -> Option<String> {
        match &*self.action_command.borrow() {
            Some(command) => Some(command.clone()),
            None => Some(self.text.borrow().clone()),
        }
    }
    /// Java `isSelected`.
    pub fn is_selected(&self) -> bool {
        self.selected.get()
    }
    /// Java `setSelected`: updates the group, fires item and change
    /// listeners on a change, never action listeners.
    pub fn set_selected(self: &Rc<Self>, selected: bool) {
        if self.selected.get() == selected {
            return;
        }
        if selected {
            let group = self.group.borrow().clone();
            if let Some(group) = group {
                for other in group.get_elements() {
                    if !Rc::ptr_eq(&other, self) && other.selected.get() {
                        other.selected.set(false);
                        let model = other.model.borrow().clone();
                        if let Some(model) = model {
                            model.set_selected(false);
                        }
                        other.fire_item_state_changed(ItemState::Deselected);
                        other.fire_state_changed();
                    }
                }
            }
        } else if self.group.borrow().is_some() {
            // `ButtonGroup.setSelected(model, false)` is ignored by Swing: a
            // grouped button cannot be deselected directly.
            if matches!(self.kind, ComponentKind::RadioButton) {
                return;
            }
        }
        self.selected.set(selected);
        let model = self.model.borrow().clone();
        if let Some(model) = model {
            model.set_selected(selected);
        }
        self.fire_item_state_changed(if selected {
            ItemState::Selected
        } else {
            ItemState::Deselected
        });
        self.fire_state_changed();
    }
    /// Java `doClick`: a user click.  Does nothing on a disabled button;
    /// toggles a check box / toggle button, selects a radio button, then fires
    /// the action listeners.
    pub fn do_click(self: &Rc<Self>) {
        if !self.enabled.get() {
            return;
        }
        match self.kind {
            ComponentKind::CheckBox
            | ComponentKind::ToggleButton
            | ComponentKind::CheckBoxMenuItem => {
                let selected = !self.selected.get();
                self.set_selected(selected);
            }
            ComponentKind::RadioButton => self.set_selected(true),
            _ => {}
        }
        self.fire_action_performed();
    }
    /// Java `addActionListener`.
    pub fn add_action_listener(&self, listener: ActionListener) {
        self.action_listeners.borrow_mut().push(listener);
    }
    /// Java `getActionListeners`.
    pub fn get_action_listeners(&self) -> Vec<ActionListener> {
        self.action_listeners.borrow().clone()
    }
    /// Java `removeActionListener`, by identity.
    pub fn remove_action_listener(&self, listener: &ActionListener) {
        self.action_listeners
            .borrow_mut()
            .retain(|l| !Rc::ptr_eq(l, listener));
    }
    /// Java `addItemListener`.
    pub fn add_item_listener(&self, listener: ItemListener) {
        self.item_listeners.borrow_mut().push(listener);
    }
    /// Java `addMouseListener`.  Mouse events reach a listener only through a
    /// driver (`fire_mouse_pressed`).
    pub fn add_mouse_listener(&self, listener: Rc<dyn MouseListener>) {
        self.mouse_listeners.borrow_mut().push(listener);
    }
    /// Whether a mouse listener is registered (Swing's `MOUSE_EVENT_MASK`
    /// test when it picks the event's target).
    pub fn has_mouse_listeners(&self) -> bool {
        !self.mouse_listeners.borrow().is_empty()
    }
    /// Delivers a mouse press to the mouse listeners (a driver's right click).
    pub fn fire_mouse_pressed(&self, event: &MouseEvent) {
        let listeners: Vec<_> = self.mouse_listeners.borrow().clone();
        for listener in listeners.iter().rev() {
            listener.mouse_pressed(event);
        }
    }
    /// Java `removeChangeListener`, by identity.
    pub fn remove_change_listener(&self, listener: &ChangeListener) {
        self.change_listeners
            .borrow_mut()
            .retain(|l| !Rc::ptr_eq(l, listener));
    }
    /// Java `addChangeListener`.
    pub fn add_change_listener(&self, listener: ChangeListener) {
        self.change_listeners.borrow_mut().push(listener);
    }
    /// Java `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&self, listener: FocusListener) {
        self.focus_listeners.borrow_mut().push(listener);
    }
    /// Java `removeFocusListener(FocusListener)`, by identity.
    pub fn remove_focus_listener(&self, listener: &FocusListener) {
        self.focus_listeners
            .borrow_mut()
            .retain(|l| !Rc::ptr_eq(l, listener));
    }
    /// Delivers a `FOCUS_GAINED` (`gained`) or `FOCUS_LOST` event to the focus
    /// listeners, in the order they were added (`AWTEventMulticaster`).  The
    /// window system calls this when the component gains or loses the keyboard
    /// focus.
    pub fn fire_focus_changed(self: &Rc<Self>, gained: bool) {
        let listeners: Vec<_> = self.focus_listeners.borrow().clone();
        let event = FocusEvent {
            source: self.clone(),
            gained,
        };
        for listener in listeners.iter() {
            listener(&event);
        }
    }
    /// `GridBagLayout.getConstraints(component)`, as set by
    /// [`GridBagLayout::set_constraints`]; `None` outside a `GridBagLayout`.
    pub fn get_layout_constraints(&self) -> Option<GridBagConstraints> {
        self.layout_constraints.get()
    }
    /// Java `setFocusable(boolean)`.
    pub fn set_focusable(&self, focusable: bool) {
        self.focusable.set(focusable);
    }
    /// Java `isFocusable()`.
    pub fn is_focusable(&self) -> bool {
        self.focusable.get()
    }
    /// `getInputMap(condition).put(KeyStroke, key)` together with
    /// `getActionMap().put(key, action)`: binds `key_stroke` (`"alt UP"`) to
    /// `action`.  A later binding of the same stroke replaces the earlier one, as
    /// the maps do.
    pub fn put_key_binding(&self, key_stroke: &str, action: Rc<dyn Fn()>) {
        let mut key_bindings = self.key_bindings.borrow_mut();
        key_bindings.retain(|(stroke, _)| stroke != key_stroke);
        key_bindings.push((key_stroke.to_owned(), action));
    }
    /// Delivers a key stroke to the component's key binding (Swing's
    /// `processKeyBinding`).  Returns true when a binding handled it.
    pub fn fire_key_binding(&self, key_stroke: &str) -> bool {
        let action = self
            .key_bindings
            .borrow()
            .iter()
            .find(|(stroke, _)| stroke == key_stroke)
            .map(|(_, action)| action.clone());
        match action {
            Some(action) => {
                action();
                true
            }
            None => false,
        }
    }
    /// Java `getDocument().addDocumentListener`.
    pub fn add_document_listener(&self, listener: DocumentListener) {
        self.document_listeners.borrow_mut().push(listener);
    }

    /// Fires the action listeners (Swing notifies the most recently added
    /// first).
    pub fn fire_action_performed(self: &Rc<Self>) {
        let listeners: Vec<_> = self.action_listeners.borrow().clone();
        let event = ActionEvent {
            source: self.clone(),
            action_command: self.get_action_command(),
        };
        for listener in listeners.iter().rev() {
            listener(&event);
        }
    }
    fn fire_item_state_changed(self: &Rc<Self>, state_change: ItemState) {
        let listeners: Vec<_> = self.item_listeners.borrow().clone();
        let event = ItemEvent {
            source: self.clone(),
            state_change,
        };
        for listener in listeners.iter().rev() {
            listener(&event);
        }
    }
    fn fire_state_changed(self: &Rc<Self>) {
        let listeners: Vec<_> = self.change_listeners.borrow().clone();
        let event = ChangeEvent {
            source: self.clone(),
        };
        for listener in listeners.iter().rev() {
            listener(&event);
        }
    }
    fn fire_document_changed(self: &Rc<Self>) {
        let listeners: Vec<_> = self.document_listeners.borrow().clone();
        let event = DocumentEvent {
            source: self.clone(),
        };
        for listener in listeners.iter().rev() {
            listener(&event);
        }
    }

    // --- JSpinner ---

    /// Java `JSpinner.getValue` as a number.
    pub fn get_spinner_value(&self) -> f64 {
        self.spinner_model
            .borrow()
            .as_ref()
            .map_or(0.0, |m| m.value)
    }
    /// The spinner's model.
    pub fn get_spinner_model(&self) -> Option<SpinnerNumberModel> {
        self.spinner_model.borrow().clone()
    }
    /// Java `JSpinner.setModel`.
    pub fn set_spinner_model(self: &Rc<Self>, model: SpinnerNumberModel) {
        *self.spinner_model.borrow_mut() = Some(model);
        self.fire_state_changed();
    }
    /// Java `JSpinner.setValue`: fires change listeners on a change.  An
    /// out-of-range value is refused (`IllegalArgumentException` in Swing's
    /// `SpinnerNumberModel.setValue` is not thrown for range; the editor
    /// refuses it), so it is ignored here.
    pub fn set_spinner_value(self: &Rc<Self>, value: f64) {
        let changed = {
            let mut model = self.spinner_model.borrow_mut();
            let Some(model) = model.as_mut() else {
                return;
            };
            if model.minimum.is_some_and(|min| value < min)
                || model.maximum.is_some_and(|max| value > max)
            {
                return;
            }
            let changed = model.value != value;
            model.value = value;
            changed
        };
        if changed {
            self.fire_state_changed();
        }
    }

    // --- JComboBox ---

    /// Java `JComboBox.addItem`.  The first item added becomes selected, as
    /// in Swing (firing the listeners).
    pub fn add_item(self: &Rc<Self>, item: &str) {
        self.add_item_nullable(Some(item));
    }
    /// Java `JComboBox.addItem(Object)` with a possibly null item (Swing
    /// renders a null item as empty).
    pub fn add_item_nullable(self: &Rc<Self>, item: Option<&str>) {
        self.items.borrow_mut().push(item.map(str::to_owned));
        if self.selected_index.get() < 0 && self.entered_item.borrow().is_none() {
            self.set_selected_index(0);
        }
    }
    /// Java `JComboBox.removeAllItems`.
    pub fn remove_all_items(&self) {
        self.items.borrow_mut().clear();
        self.selected_index.set(-1);
        *self.entered_item.borrow_mut() = None;
    }
    /// Java `JComboBox.getItemCount`.
    pub fn get_item_count(&self) -> usize {
        self.items.borrow().len()
    }
    /// Java `JComboBox.getItemAt` (a null item is `None`).
    pub fn get_item_at(&self, index: usize) -> Option<String> {
        self.items.borrow().get(index).cloned().flatten()
    }
    /// Java `JComboBox.getSelectedIndex`.
    pub fn get_selected_index(&self) -> i32 {
        self.selected_index.get()
    }
    /// Java `JComboBox.getSelectedItem`.
    pub fn get_selected_item(&self) -> Option<String> {
        let index = self.selected_index.get();
        if index < 0 {
            return self.entered_item.borrow().clone();
        }
        self.items.borrow().get(index as usize).cloned().flatten()
    }
    /// Java `JComboBox.setSelectedItem(Object)`: selects the first equal item;
    /// an item not in a non-editable combo box leaves the selection unchanged;
    /// in an editable one it becomes the selected object
    /// (`DefaultComboBoxModel.setSelectedItem`), with no index.
    pub fn set_selected_item(self: &Rc<Self>, item: Option<&str>) {
        let index = self
            .items
            .borrow()
            .iter()
            .position(|candidate| candidate.as_deref() == item);
        if let Some(index) = index {
            self.set_selected_index(index as i32);
        } else if self.editable.get() || item.is_none() {
            let old = self.get_selected_item();
            if old.as_deref() != item {
                self.selected_index.set(-1);
                *self.entered_item.borrow_mut() = item.map(str::to_owned);
                // `JComboBox.selectedItemChanged`
                if old.is_some() {
                    self.fire_item_state_changed(ItemState::Deselected);
                }
                if item.is_some() {
                    self.fire_item_state_changed(ItemState::Selected);
                }
            }
            self.fire_action_performed();
        }
    }
    /// Java `JComboBox.setSelectedIndex`: item listeners then action
    /// listeners on a change (Swing fires the action even when unchanged).
    pub fn set_selected_index(self: &Rc<Self>, index: i32) {
        let old = self.selected_index.get();
        self.selected_index.set(index);
        let had_entered = self.entered_item.borrow_mut().take().is_some();
        if old != index || had_entered {
            if old >= 0 || had_entered {
                self.fire_item_state_changed(ItemState::Deselected);
            }
            if index >= 0 {
                self.fire_item_state_changed(ItemState::Selected);
            }
        }
        self.fire_action_performed();
    }

    // --- JTabbedPane ---

    /// Java `JTabbedPane.addTab(title, component)`.
    pub fn add_tab(self: &Rc<Self>, title: &str, component: &Rc<JComponent>) {
        self.add(component);
        self.tab_titles.borrow_mut().push(title.to_owned());
        self.tab_enabled.borrow_mut().push(true);
        if self.selected_index.get() < 0 {
            self.selected_index.set(0);
            self.fire_state_changed();
        }
    }
    /// Java `JTabbedPane.setTabComponentAt(int, Component)`: the tab's title
    /// component (a check box or label drawn in the tab).
    pub fn set_tab_component_at(self: &Rc<Self>, index: usize, component: &Rc<JComponent>) {
        let mut titles = self.tab_title_components.borrow_mut();
        if titles.len() <= index {
            titles.resize(index + 1, None);
        }
        titles[index] = Some(component.clone());
        *component.parent.borrow_mut() = Rc::downgrade(self);
    }
    /// The components set with `setTabComponentAt`, searched like children.
    pub fn get_tab_title_components(&self) -> Vec<Rc<JComponent>> {
        self.tab_title_components
            .borrow()
            .iter()
            .flatten()
            .cloned()
            .collect()
    }
    /// Java `JTabbedPane.getTabCount`.
    pub fn get_tab_count(&self) -> usize {
        self.tab_titles.borrow().len()
    }
    /// Java `JTabbedPane.getTitleAt`.
    pub fn get_title_at(&self, index: usize) -> Option<String> {
        self.tab_titles.borrow().get(index).cloned()
    }
    /// Java `JTabbedPane.setTitleAt`.
    pub fn set_title_at(&self, index: usize, title: &str) {
        if let Some(slot) = self.tab_titles.borrow_mut().get_mut(index) {
            *slot = title.to_owned();
        }
    }
    /// Java `JTabbedPane.getComponentAt`.
    pub fn get_component_at(&self, index: usize) -> Option<Rc<JComponent>> {
        self.children.borrow().get(index).cloned()
    }
    /// Java `JTabbedPane.setEnabledAt`.
    pub fn set_enabled_at(&self, index: usize, enabled: bool) {
        if let Some(slot) = self.tab_enabled.borrow_mut().get_mut(index) {
            *slot = enabled;
        }
    }
    /// Java `JTabbedPane.isEnabledAt`.
    pub fn is_enabled_at(&self, index: usize) -> bool {
        self.tab_enabled
            .borrow()
            .get(index)
            .copied()
            .unwrap_or(false)
    }
    /// Java `JTabbedPane.setSelectedIndex`: change listeners on a change.
    pub fn set_selected_tab(self: &Rc<Self>, index: i32) {
        if self.selected_index.get() != index {
            self.selected_index.set(index);
            self.fire_state_changed();
        }
    }
    /// Java `JTabbedPane.getSelectedIndex`.
    pub fn get_selected_tab(&self) -> i32 {
        self.selected_index.get()
    }

    // --- JProgressBar ---

    pub fn set_minimum(&self, value: i32) {
        self.minimum.set(value);
    }
    pub fn get_minimum(&self) -> i32 {
        self.minimum.get()
    }
    pub fn set_maximum(&self, value: i32) {
        self.maximum.set(value);
    }
    pub fn get_maximum(&self) -> i32 {
        self.maximum.get()
    }
    pub fn set_value(&self, value: i32) {
        self.int_value.set(value);
    }
    pub fn get_value(&self) -> i32 {
        self.int_value.get()
    }
    pub fn set_indeterminate(&self, value: bool) {
        self.indeterminate.set(value);
    }
    pub fn is_indeterminate(&self) -> bool {
        self.indeterminate.get()
    }
    /// Java `JProgressBar.setString`.
    pub fn set_string(&self, value: Option<&str>) {
        *self.string.borrow_mut() = value.map(str::to_owned);
    }
    /// Java `JProgressBar.getString`: the string set, else the percentage
    /// complete as `NumberFormat.getPercentInstance()` formats it
    /// (`"0%"`, rounded half-even to a whole percent).
    pub fn get_string(&self) -> Option<String> {
        if let Some(string) = self.string.borrow().clone() {
            return Some(string);
        }
        let range = self.maximum.get() as f64 - self.minimum.get() as f64;
        let percent = if range <= 0.0 {
            0.0
        } else {
            (self.int_value.get() as f64 - self.minimum.get() as f64) / range
        };
        Some(format!("{}%", (percent * 100.0).round_ties_even() as i64))
    }
}

/// `java.awt.GridBagConstraints.REMAINDER`.
pub const GRID_BAG_REMAINDER: i32 = 0;
/// `java.awt.GridBagConstraints.RELATIVE`.
pub const GRID_BAG_RELATIVE: i32 = -1;
/// `java.awt.GridBagConstraints.CENTER`.
pub const GRID_BAG_CENTER: i32 = 10;
/// `java.awt.GridBagConstraints.NONE`.
pub const GRID_BAG_NONE: i32 = 0;
/// `java.awt.GridBagConstraints.BOTH`.
pub const GRID_BAG_BOTH: i32 = 1;

/// `java.awt.GridBagConstraints`: the fields eTomo's tables set.  The row
/// structure of a table (which cell ends a row, `gridwidth == REMAINDER`) is
/// read back from them by the Slint bridge, which draws the table from the
/// component tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridBagConstraints {
    pub gridwidth: i32,
    pub gridheight: i32,
    pub weightx: f64,
    pub weighty: f64,
    pub anchor: i32,
    pub fill: i32,
}

impl Default for GridBagConstraints {
    /// `new GridBagConstraints()`.
    fn default() -> GridBagConstraints {
        GridBagConstraints {
            gridwidth: 1,
            gridheight: 1,
            weightx: 0.0,
            weighty: 0.0,
            anchor: GRID_BAG_CENTER,
            fill: GRID_BAG_NONE,
        }
    }
}

/// `java.awt.GridBagLayout`.  Only `setConstraints` is modelled: it records a
/// copy of the constraints on the component.
#[derive(Default)]
pub struct GridBagLayout;

impl GridBagLayout {
    /// `new GridBagLayout()`.
    pub fn new() -> GridBagLayout {
        GridBagLayout
    }
    /// `setConstraints(Component, GridBagConstraints)`: a copy.
    pub fn set_constraints(&self, component: &Rc<JComponent>, constraints: &GridBagConstraints) {
        component.layout_constraints.set(Some(*constraints));
    }
}

/// `java.awt.event.WindowListener`.  Every method defaults to Java's empty
/// `WindowAdapter` body.
pub trait WindowListener {
    fn window_activated(&self) {}
    fn window_closed(&self) {}
    fn window_closing(&self) {}
    fn window_deactivated(&self) {}
    fn window_deiconified(&self) {}
    fn window_iconified(&self) {}
    fn window_opened(&self) {}
}

/// `WindowConstants.DO_NOTHING_ON_CLOSE`.
pub const DO_NOTHING_ON_CLOSE: i32 = 0;
/// `WindowConstants.HIDE_ON_CLOSE` (the `JDialog` default).
pub const HIDE_ON_CLOSE: i32 = 1;
/// `WindowConstants.DISPOSE_ON_CLOSE`.
pub const DISPOSE_ON_CLOSE: i32 = 2;

thread_local! {
    /// The showing `JPopupMenu` with the point it was shown at (Swing's popup
    /// layer holds one popup menu at a time).
    static SHOWING_POPUP: RefCell<Option<(Rc<JComponent>, i32, i32)>> = const { RefCell::new(None) };
}

/// The showing popup menu (`JPopupMenu.show`) and the point in its invoker it
/// was shown at.
pub fn showing_popup_menu() -> Option<(Rc<JComponent>, i32, i32)> {
    SHOWING_POPUP.with(|showing| showing.borrow().clone())
}

/// A mouse press at `component`: Swing's `LightweightDispatcher` gives the
/// event to the deepest component under the pointer that takes mouse events,
/// so a component that does not passes it to the nearest ancestor that does.
/// Besides the components with an application `MouseListener`, Swing's own
/// look and feel listens on every button, check box, radio button, menu item,
/// text component, spinner, combo box and tabbed pane, and `ToolTipManager`
/// on every component with a tool tip; such a component takes the press (and
/// a right press does nothing there) unless the application also listens.
/// Pressing anywhere hides a showing popup menu first (`BasicPopupMenuUI`'s
/// `MouseGrabber` cancels the menu on a press outside it).
pub fn dispatch_mouse_pressed(component: &Rc<JComponent>, event: &MouseEvent) {
    if let Some((popup, _, _)) = showing_popup_menu() {
        popup.set_visible(false);
    }
    let mut target = Some(component.clone());
    while let Some(candidate) = target {
        let takes_mouse_events = candidate.has_mouse_listeners()
            || candidate
                .get_tool_tip_text()
                .is_some_and(|tip| !tip.is_empty())
            || matches!(
                candidate.kind(),
                ComponentKind::Button
                    | ComponentKind::ToggleButton
                    | ComponentKind::CheckBox
                    | ComponentKind::RadioButton
                    | ComponentKind::TextField
                    | ComponentKind::TextArea
                    | ComponentKind::Spinner
                    | ComponentKind::ComboBox
                    | ComponentKind::TabbedPane
                    | ComponentKind::MenuItem
                    | ComponentKind::CheckBoxMenuItem
                    | ComponentKind::Menu
            );
        if takes_mouse_events {
            candidate.fire_mouse_pressed(event);
            return;
        }
        target = candidate.get_parent();
    }
}

thread_local! {
    /// The showing `JDialog`s, in the order they were shown (`Window.getWindows()`
    /// lists them; the driver and the Slint bridge look for components in them).
    static SHOWING_DIALOGS: RefCell<Vec<Rc<JDialog>>> = const { RefCell::new(Vec::new()) };
}

/// A value moved through `invoke_later` that is only touched on the event
/// dispatch thread, which both posts and runs it.
struct EdtOnly<T>(T);
// SAFETY: created on the EDT and consumed by a job that `invoke_later` runs on
// the EDT; no other thread touches it.
unsafe impl<T> Send for EdtOnly<T> {}

/// `javax.swing.JDialog`: its content pane (the root of its component tree),
/// title, modality, visibility, default close operation and window listeners.
/// Painting and layout (`pack`) are the Slint side's.
///
/// **Modality.**  `setVisible(true)` on a modal Swing dialog does not return
/// until the dialog is hidden: Swing runs a secondary event loop.  This
/// stand-in has no secondary loop (the Slint event loop cannot be nested), so
/// `set_visible(true)` returns at once, and code the Java runs *after* the
/// blocking `setVisible(true)` is handed to [`JDialog::after_modal_return`],
/// which runs it once the dialog is hidden - after the event that hid it has
/// finished, as the Java's return from the secondary loop happens.
pub struct JDialog {
    content_pane: Rc<JComponent>,
    title: RefCell<String>,
    modal: bool,
    visible: Cell<bool>,
    default_close_operation: Cell<i32>,
    window_listeners: RefCell<Vec<Rc<dyn WindowListener>>>,
    modal_return: RefCell<Vec<Box<dyn FnOnce()>>>,
    this: RefCell<Weak<JDialog>>,
}

impl JDialog {
    /// `new JDialog(Frame owner, String title, boolean modal)`.  A dialog is
    /// created invisible.
    pub fn new(title: &str, modal: bool) -> Rc<JDialog> {
        let content_pane = JComponent::new_panel();
        content_pane.set_visible(false);
        let dialog = Rc::new(JDialog {
            content_pane,
            title: RefCell::new(title.to_owned()),
            modal,
            visible: Cell::new(false),
            default_close_operation: Cell::new(HIDE_ON_CLOSE),
            window_listeners: RefCell::new(Vec::new()),
            modal_return: RefCell::new(Vec::new()),
            this: RefCell::new(Weak::new()),
        });
        *dialog.this.borrow_mut() = Rc::downgrade(&dialog);
        dialog
    }
    /// `getContentPane()`.
    pub fn get_content_pane(&self) -> Rc<JComponent> {
        self.content_pane.clone()
    }
    /// `getTitle()`.
    pub fn get_title(&self) -> String {
        self.title.borrow().clone()
    }
    /// `setTitle(String)`.
    pub fn set_title(&self, title: &str) {
        *self.title.borrow_mut() = title.to_owned();
    }
    /// `isModal()`.
    pub fn is_modal(&self) -> bool {
        self.modal
    }
    /// `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.visible.get()
    }
    /// `setDefaultCloseOperation(int)`.
    pub fn set_default_close_operation(&self, operation: i32) {
        self.default_close_operation.set(operation);
    }
    /// `addWindowListener(WindowListener)`.
    pub fn add_window_listener(&self, listener: Rc<dyn WindowListener>) {
        self.window_listeners.borrow_mut().push(listener);
    }
    /// `pack()`: layout only.
    pub fn pack(&self) {}
    /// `setVisible(boolean)`.  See the type comment on modality.
    pub fn set_visible(&self, visible: bool) {
        let was_visible = self.visible.replace(visible);
        self.content_pane.set_visible(visible);
        let Some(this) = self.this.borrow().upgrade() else {
            return;
        };
        if visible && !was_visible {
            SHOWING_DIALOGS.with(|dialogs| dialogs.borrow_mut().push(this));
            let listeners = self.window_listeners.borrow().clone();
            for listener in listeners {
                listener.window_opened();
            }
        } else if !visible && was_visible {
            SHOWING_DIALOGS.with(|dialogs| {
                dialogs
                    .borrow_mut()
                    .retain(|dialog| !Rc::ptr_eq(dialog, &this))
            });
            let jobs: Vec<Box<dyn FnOnce()>> = self.modal_return.borrow_mut().drain(..).collect();
            for job in jobs {
                let job = EdtOnly(job);
                crate::imod::etomo::util::event_queue::invoke_later(move || {
                    let job = job;
                    (job.0)()
                });
            }
        }
    }
    /// `dispose()`: hides the dialog and releases it.
    pub fn dispose(&self) {
        self.set_visible(false);
        let listeners = self.window_listeners.borrow().clone();
        for listener in listeners {
            listener.window_closed();
        }
    }
    /// The code the Java runs after a blocking `setVisible(true)` on this
    /// modal dialog; run once the dialog is hidden (at once when it is not
    /// showing or not modal).
    pub fn after_modal_return(&self, job: Box<dyn FnOnce()>) {
        if self.modal && self.visible.get() {
            self.modal_return.borrow_mut().push(job);
        } else {
            job();
        }
    }
    /// The window system's close request (the title bar's close button):
    /// `WINDOW_CLOSING` to the listeners, then the default close operation.
    pub fn process_window_closing(&self) {
        let listeners = self.window_listeners.borrow().clone();
        for listener in listeners {
            listener.window_closing();
        }
        match self.default_close_operation.get() {
            HIDE_ON_CLOSE => self.set_visible(false),
            DISPOSE_ON_CLOSE => self.dispose(),
            _ => {}
        }
    }
}

/// The showing `JDialog`s, oldest first.  EDT only.
pub fn showing_dialogs() -> Vec<Rc<JDialog>> {
    SHOWING_DIALOGS.with(|dialogs| dialogs.borrow().clone())
}

/// Finds the first component named `name` under `root` (depth first, the
/// root included), as eTomo's `uitest` finds a field by name.  With
/// `showing_only`, hidden subtrees are skipped.
pub fn find_component(
    root: &Rc<JComponent>,
    name: &str,
    showing_only: bool,
) -> Option<Rc<JComponent>> {
    if showing_only && !root.is_visible() {
        return None;
    }
    if root.name.borrow().as_deref() == Some(name) {
        return Some(root.clone());
    }
    for child in root
        .get_components()
        .into_iter()
        .chain(root.get_tab_title_components())
    {
        if let Some(found) = find_component(&child, name, showing_only) {
            return Some(found);
        }
    }
    None
}

/// Every named component under `root`, depth first (for dumps and the Slint
/// bridge's registry).
pub fn named_components(root: &Rc<JComponent>) -> Vec<(String, Rc<JComponent>)> {
    let mut out = Vec::new();
    fn walk(node: &Rc<JComponent>, out: &mut Vec<(String, Rc<JComponent>)>) {
        if let Some(name) = node.get_name() {
            out.push((name, node.clone()));
        }
        for child in node
            .get_components()
            .into_iter()
            .chain(node.get_tab_title_components())
        {
            walk(&child, out);
        }
    }
    walk(root, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radio_group_is_exclusive_and_click_fires_action() {
        let group = ButtonGroup::new();
        let a = JComponent::new_radio_button("A");
        let b = JComponent::new_radio_button("B");
        group.add(&a);
        group.add(&b);
        let fired = Rc::new(Cell::new(0));
        let f = fired.clone();
        b.add_action_listener(Rc::new(move |_| f.set(f.get() + 1)));
        a.set_selected(true);
        assert!(a.is_selected());
        b.do_click();
        assert!(!a.is_selected() && b.is_selected());
        assert_eq!(fired.get(), 1);
        b.set_selected(false);
        assert!(b.is_selected());
    }

    #[test]
    fn find_by_name_under_root() {
        let root = JComponent::new_panel();
        let panel = JComponent::new_panel();
        let button = JComponent::new_button("Done");
        button.set_name(Some("bn.done"));
        root.add(&panel);
        panel.add(&button);
        assert!(find_component(&root, "bn.done", true).is_some());
        panel.set_visible(false);
        assert!(find_component(&root, "bn.done", true).is_none());
        assert!(find_component(&root, "bn.done", false).is_some());
    }
}
