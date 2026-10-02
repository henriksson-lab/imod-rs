//! `IMOD/Etomo/src/etomo/ui/swing/ComboBox.java`: a self-naming combo box with
//! optional label, text entry, empty choice and placeholder policies.
//!
//! `final class ComboBox`.  The `JComboBox`, `JLabel` and `JPanel` are jdk
//! stand-in [`JComponent`]s.  Java combo-box items are `Object`s displayed by
//! their `toString()`; the jdk combo box holds that text, with a Java `null`
//! item (the empty choice) as `None`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionListener, FocusListener, JComponent};
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::debug_level::DebugLevel;
use crate::imod::etomo::r#type::ui_test_field_type;
use crate::imod::etomo::ui::swing::colors;
use crate::imod::etomo::util::utilities;

/// Java `ComboBox`.
pub struct ComboBox {
    combo_box: Rc<JComponent>,
    label: Option<Rc<JComponent>>,
    pnl_root: Option<Rc<JComponent>>,
    text_entry_policy: bool, // presidence:1
    empty_choice: bool,      // presidence:2

    checkpointed: Cell<bool>,
    checkpoint_index: Cell<i32>,
    debug: Cell<DebugLevel>,
    enabled_policy: Cell<bool>,
    enabled: Cell<bool>,
    editable: Cell<bool>,
    placeholder: Cell<bool>, // presidence:3
    empty_label: RefCell<Option<String>>,
    no_selection_label: RefCell<Option<String>>,
}

impl ComboBox {
    /// Java `ComboBox(String name, boolean labeled, boolean textEntryPolicy,
    /// boolean emptyChoice)`.
    fn new(
        name: Option<&str>,
        labeled: bool,
        text_entry_policy: bool,
        empty_choice: bool,
    ) -> ComboBox {
        let combo_box = JComponent::new_combo_box();
        let (label, pnl_root) = if labeled {
            (
                Some(JComponent::new_label(name.unwrap_or(""))),
                Some(JComponent::new_panel()),
            )
        } else {
            (None, None)
        };
        let instance = ComboBox {
            combo_box,
            label,
            pnl_root,
            text_entry_policy,
            empty_choice: if text_entry_policy {
                false
            } else {
                empty_choice
            },
            checkpointed: Cell::new(false),
            checkpoint_index: Cell::new(-1),
            debug: Cell::new(ARGUMENTS.lock().unwrap().get_debug_level()),
            enabled_policy: Cell::new(true),
            enabled: Cell::new(true),
            editable: Cell::new(true),
            placeholder: Cell::new(false),
            empty_label: RefCell::new(None),
            no_selection_label: RefCell::new(None),
        };
        // Java order: setName(name) before the label and panel are created (they
        // are built above only because the Rust fields are immutable).
        instance.set_name(name);
        if !text_entry_policy && empty_choice {
            instance.combo_box.add_item_nullable(None);
        }
        instance.update_display();
        instance
    }

    /// Java `getInstance(String)`.
    pub fn get_instance(name: Option<&str>) -> Rc<ComboBox> {
        let instance = Rc::new(ComboBox::new(name, true, false, false));
        instance.create_panel();
        instance
    }

    /// Java `getUnlabeledInstance(String)`.
    pub fn get_unlabeled_instance(name: Option<&str>) -> Rc<ComboBox> {
        let instance = Rc::new(ComboBox::new(name, false, false, false));
        instance.create_panel();
        instance
    }

    /// Java `getEditableInstance(String)`.
    pub fn get_editable_instance(name: Option<&str>) -> Rc<ComboBox> {
        let instance = Rc::new(ComboBox::new(name, true, true, false));
        instance.create_panel();
        instance
    }

    /// Java `getUnlabeledEmptyChoiceInstance(String)`.
    pub fn get_unlabeled_empty_choice_instance(name: Option<&str>) -> Rc<ComboBox> {
        let instance = Rc::new(ComboBox::new(name, false, false, true));
        instance.create_panel();
        instance
    }

    /// Java `getEmptyChoiceInstance(String)`.
    pub fn get_empty_choice_instance(name: Option<&str>) -> Rc<ComboBox> {
        let instance = Rc::new(ComboBox::new(name, true, false, true));
        instance.create_panel();
        instance
    }

    /// Java `createPanel()`.
    fn create_panel(&self) {
        if let Some(pnl_root) = &self.pnl_root {
            // Swing layout: BoxLayout X_AXIS, rigid areas FixedDim.x2_y0 before the
            // label, x3_y0 between label and combo box, x2_y0 after.
            if let Some(label) = &self.label {
                pnl_root.add(label);
            }
            pnl_root.add(&self.combo_box);
        }
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, listener: ActionListener) {
        self.combo_box.add_action_listener(listener);
    }

    /// Java `addFocusListener(FocusListener)`.
    pub fn add_focus_listener(&self, listener: FocusListener) {
        self.combo_box.add_focus_listener(listener);
    }

    /// Java `setPlaceholder(String emptyLabel, String noSelectionLabel)`.  Adds a
    /// placeholder item at the beginning of the combobox.  Can be called at any
    /// time.  This function has no effect if textEntryPolicy or emptyChoice are
    /// on.  Calling it more then once replaces the labels.  Calling it with two
    /// null labels turns off placeholder, or has no effect if placeholder is
    /// already off.
    pub fn set_placeholder(&self, empty_label: Option<&str>, no_selection_label: Option<&str>) {
        if self.text_entry_policy || self.empty_choice {
            // these take presidence
            return;
        }
        *self.empty_label.borrow_mut() = empty_label.map(str::to_owned);
        *self.no_selection_label.borrow_mut() = no_selection_label.map(str::to_owned);
        let no_label = empty_label.is_none() && no_selection_label.is_none();
        if !self.placeholder.get() && no_label {
            // nothing to do
            return;
        }
        // Check empty before changing placeholder boolean.
        let empty = self.is_empty();
        if empty {
            // Empty, but may contain an out-of-date placeholder
            self.combo_box.remove_all_items();
        }
        if self.placeholder.get() && no_label {
            // turn off placeholder
            self.placeholder.set(false);
        } else {
            self.placeholder.set(true);
        }
        if self.placeholder.get() && empty {
            self.combo_box.add_item_nullable(empty_label);
        }
        if !empty {
            // Temporarily store existing items so a placeholder can be added,
            // changed, or removed.
            let count = self.combo_box.get_item_count();
            let mut items: Vec<Option<String>> = Vec::with_capacity(count);
            for i in 0..count {
                items.push(self.combo_box.get_item_at(i));
            }
            self.combo_box.remove_all_items();
            if self.placeholder.get() {
                self.combo_box.add_item_nullable(no_selection_label);
            }
            for item in &items {
                self.combo_box.add_item_nullable(item.as_deref());
            }
        }
    }

    /// Java `addItem(Object)`.
    pub fn add_item(&self, input: Option<&str>) {
        if self.placeholder.get() && self.is_empty() {
            // Update placeholder
            self.combo_box.remove_all_items();
            let no_selection_label = self.no_selection_label.borrow().clone();
            self.combo_box
                .add_item_nullable(no_selection_label.as_deref());
        }
        self.combo_box.add_item_nullable(input);
        self.update_display();
    }

    /// Java `setMaximumWidth(int)`.
    pub fn set_maximum_width(&self, width: i32) {
        // Swing layout: comboBox.setMaximumSize(UIUtilities.calcNewComboBoxSize(
        // comboBox.getPreferredSize(), width, true)).
        let _ = width;
    }

    /// Java `setFieldHighlight()`.
    pub fn set_field_highlight(&self) {
        // Upstream bug fixed (ComboBox.java:196-200): `label.setForeground` throws a
        // NullPointerException on an unlabeled combo box; only an existing label
        // is coloured here.
        if let Some(label) = &self.label {
            label.set_foreground(Some(colors::FIELD_HIGHLIGHT));
        }
        self.combo_box.set_foreground(Some(colors::FIELD_HIGHLIGHT));
        // Swing painting: comboBox.setBorder(BorderFactory.createLineBorder(
        // Colors.FIELD_HIGHLIGHT)).
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.combo_box.get_action_command()
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        if let Some(pnl_root) = &self.pnl_root {
            return pnl_root.clone();
        }
        self.combo_box.clone()
    }

    /// Java `getSelectedIndex()`.  Returns the selected index.  If emptyChoice or
    /// placeholder is on, then the index is adjusted so that it starts from zero.
    /// If the placeholder was selected it returns -1.
    pub fn get_selected_index(&self) -> i32 {
        let index = self.combo_box.get_selected_index();
        if (self.empty_choice || self.placeholder.get()) && index > -1 {
            return index - 1;
        }
        index
    }

    /// Java `getSelectedItem()`.
    pub fn get_selected_item(&self) -> Option<String> {
        self.combo_box.get_selected_item()
    }

    /// Java `removeAllItems()`.
    pub fn remove_all_items(&self) {
        self.combo_box.remove_all_items();
        if self.empty_choice {
            self.combo_box.add_item_nullable(None);
        } else if self.placeholder.get() {
            let empty_label = self.empty_label.borrow().clone();
            self.combo_box.add_item_nullable(empty_label.as_deref());
        }
        self.update_display();
    }

    /// Java `updateDisplay()`.  Set combo box display based on the settings and
    /// what is in the pulldown list.
    /// Enabled:
    /// - should not be enabled if the enabled policy is false.
    /// - should not be enabled if it is empty - unless the text entry policy is
    ///   true.
    /// - When the text entry policy is false, makeing the field
    ///   editable/ineditable is done by enabling/disabling it.  So in this case
    ///   substitute (enabled && editable) for enabled.
    ///
    /// Editable:
    /// - Set to editable when editable, and the text entry policy is true.
    fn update_display(&self) {
        if !self.enabled_policy.get() || (self.is_empty() && !self.text_entry_policy) {
            self.combo_box.set_enabled(false);
        } else if !self.text_entry_policy {
            self.combo_box
                .set_enabled(self.enabled.get() && self.editable.get());
        } else {
            self.combo_box.set_enabled(self.enabled.get());
        }
        self.combo_box
            .set_editable(self.text_entry_policy && self.editable.get());
    }

    /// Java `isEmpty()`.  Takes the placeholder into account.
    pub fn is_empty(&self) -> bool {
        self.combo_box.get_item_count()
            <= if self.empty_choice || self.placeholder.get() {
                1
            } else {
                0
            }
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
        if let Some(label) = &self.label {
            label.set_enabled(enabled);
        }
        self.update_display();
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.editable.set(editable);
        self.update_display();
    }

    /// Java `setEnabledPolicy(boolean)`.  Sets the enabled policy.  The enabled
    /// policy default is true.  Makes sure that combobox is not in an illegal
    /// state.
    pub fn set_enabled_policy(&self, input: bool) {
        self.enabled_policy.set(input);
        self.update_display();
    }

    /// Java `getLabel()`.
    pub fn get_label(&self) -> Option<String> {
        // Upstream bug fixed (ComboBox.java:327-329): `label.getText()` throws a
        // NullPointerException on an unlabeled combo box; that answers null here.
        self.label.as_ref().map(|label| label.get_text())
    }

    /// Java `setDebug(DebugLevel)`.
    pub fn set_debug(&self, input: DebugLevel) {
        self.debug.set(input);
    }

    /// Java `setName(String)`.
    pub fn set_name(&self, text: Option<&str>) {
        let field_type = &ui_test_field_type::COMBO_BOX;
        let name = utilities::convert_label_to_name(text, field_type.is_unlimited_segments());
        // Java string concatenation of a null name gives "null".
        self.combo_box.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.combo_box.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `checkpoint()`.  Saves the current selected index as the checkpoint.
    pub fn checkpoint(&self) {
        self.checkpointed.set(true);
        self.checkpoint_index.set(self.get_selected_index());
    }

    /// Java `isDifferentFromCheckpoint(boolean alwaysCheck)`: check for difference
    /// even when the field is disabled or invisible.
    pub fn is_different_from_checkpoint(&self, always_check: bool) -> bool {
        if !always_check && (!self.is_enabled() || !self.is_visible()) {
            return false;
        }
        if !self.checkpointed.get() {
            return true;
        }
        self.checkpoint_index.get() != self.get_selected_index()
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.combo_box.is_visible()
    }

    /// Java `setSelectedIndex(int)`.  Selects an item.  If emptyChoice or
    /// placeholder is on, it adjusts for it, so that a zero index refers to first
    /// non-empty choice.
    pub fn set_selected_index(&self, mut index: i32) {
        if self.empty_choice || self.placeholder.get() {
            index += 1;
        }
        self.combo_box.set_selected_index(index);
    }

    /// Java `unselect()`.  Turns off selection unless emptyChoice or placeholder
    /// are set.  If so sets to either the empty choice or the placeholder.
    pub fn unselect(&self) {
        if self.empty_choice || self.placeholder.get() {
            self.combo_box.set_selected_index(0);
        } else {
            self.combo_box.set_selected_item(None);
        }
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, tooltip: Option<&str>) {
        if let Some(label) = &self.label {
            label.set_tool_tip_text(tooltip);
        }
        self.combo_box.set_tool_tip_text(tooltip);
    }

    /// Java `verifyComboBoxSource(JComboBox)`.
    pub fn verify_combo_box_source(&self, input: &Rc<JComponent>) -> bool {
        Rc::ptr_eq(&self.combo_box, input)
    }
}
