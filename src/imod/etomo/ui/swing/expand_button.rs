//! `IMOD/Etomo/src/etomo/ui/swing/ExpandButton.java`.
//!
//! A small single-line HTML button that toggles between an expanded and a
//! contracted state ("<"/">", "A"/"B", "-"/"+") and tells its `Expandable`s
//! (and an optional `GlobalExpandButton`) when it changes.
//!
//! Java `extends SingleLineButton`: the superclass is field `base` (deref);
//! see `multi_line_button.rs` for the object model.  The expandables are held
//! as `Weak<dyn Expandable>` (they own the button and usually pass themselves
//! while being constructed); the global expand button is held strongly and
//! holds this button weakly.
//!
//! Not modelled: the bevel border and the square size set in the
//! constructor, and `getPreferredWidth()` (the `TableComponent` member), which
//! are layout.  `add(JPanel, GridBagLayout, GridBagConstraints)` keeps the
//! panel and drops the layout arguments.

use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::rc::{Rc, Weak};

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::ui::expander::Expander;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::multi_line_button::{MultiLineButton, MultiLineButtonVirtual};
use super::single_line_button::SingleLineButton;
use super::swing_component::SwingComponent;

/// Java public final `ExpandButton`.
pub struct ExpandButton {
    /// Java superclass `SingleLineButton`.
    pub base: SingleLineButton,
    /// This object, for the Java calls that pass `this`.
    self_ref: RefCell<Weak<ExpandButton>>,
    /// Java final `type`.
    type_: &'static Type,
    /// Java final `expandable1`.
    expandable1: Option<Weak<dyn Expandable>>,
    /// Java final `expandable2`.
    expandable2: Option<Weak<dyn Expandable>>,
    /// Java final `globalExpandButton`.
    global_expand_button: Option<Rc<GlobalExpandButton>>,
    /// Java `expanded`.
    expanded: Cell<bool>,
    /// Java `jpanelContainer`.
    jpanel_container: RefCell<Option<Rc<JComponent>>>,
    /// Java `debug`.  Unused in the Java class.
    debug: Cell<bool>,
}

impl Deref for ExpandButton {
    type Target = SingleLineButton;
    fn deref(&self) -> &SingleLineButton {
        &self.base
    }
}

impl MultiLineButtonVirtual for ExpandButton {
    fn get_multi_line_button(&self) -> &MultiLineButton {
        &self.base.base
    }
    // Inherited from SingleLineButton.
    fn new_button(&self) -> Rc<JComponent> {
        SingleLineButton::new_button(&self.base)
    }
    fn setup_button(&self, set_minimum_size: bool) {
        SingleLineButton::setup_button(&self.base, set_minimum_size)
    }
    fn set_text_label(&self, text: Option<&str>) {
        SingleLineButton::set_text_label(&self.base, text)
    }
    // Overridden here.
    fn set_name(&self, associated_label: Option<&str>) {
        ExpandButton::set_name(self, associated_label)
    }
    fn create_button_state_key(&self, dialog_type: Option<DialogType>) -> Option<String> {
        ExpandButton::create_button_state_key(self, dialog_type)
    }
    fn get_button_state(&self) -> bool {
        ExpandButton::get_button_state(self)
    }
    fn set_button_state(&self, state: bool) {
        ExpandButton::set_button_state(self, state)
    }
}

impl ExpandButton {
    /// Java private static final `DEFAULT_TYPE`.
    const DEFAULT_TYPE: &'static Type = &Type::MORE;

    /// Java static `getInstance(Expandable, ExpandButton.Type)`.
    pub fn get_instance_expandable_type(
        expandable: Option<Weak<dyn Expandable>>,
        type_: Option<&'static Type>,
    ) -> Rc<ExpandButton> {
        let type_ = type_.unwrap_or(Self::DEFAULT_TYPE);
        ExpandButton::new_expandable_expandable_type_global_expand_button(
            expandable, None, type_, None,
        )
    }

    /// Java static `getInstance(Expandable, Expandable, ExpandButton.Type)`.
    pub fn get_instance_expandable_expandable_type(
        expandable1: Option<Weak<dyn Expandable>>,
        expandable2: Option<Weak<dyn Expandable>>,
        type_: Option<&'static Type>,
    ) -> Rc<ExpandButton> {
        let type_ = type_.unwrap_or(Self::DEFAULT_TYPE);
        ExpandButton::new_expandable_expandable_type_global_expand_button(
            expandable1,
            expandable2,
            type_,
            None,
        )
    }

    /// Java static `getGlobalInstance(Expandable, ExpandButton.Type, GlobalExpandButton)`.
    pub fn get_global_instance_expandable_type_global_expand_button(
        expandable1: Option<Weak<dyn Expandable>>,
        type_: Option<&'static Type>,
        global_expand_button: Option<Rc<GlobalExpandButton>>,
    ) -> Rc<ExpandButton> {
        let type_ = type_.unwrap_or(Self::DEFAULT_TYPE);
        ExpandButton::new_expandable_expandable_type_global_expand_button(
            expandable1,
            None,
            type_,
            global_expand_button,
        )
    }

    /// Java static
    /// `getGlobalInstance(Expandable, Expandable, ExpandButton.Type, GlobalExpandButton)`.
    pub fn get_global_instance_expandable_expandable_type_global_expand_button(
        expandable1: Option<Weak<dyn Expandable>>,
        expandable2: Option<Weak<dyn Expandable>>,
        type_: Option<&'static Type>,
        global_expand_button: Option<Rc<GlobalExpandButton>>,
    ) -> Rc<ExpandButton> {
        let type_ = type_.unwrap_or(Self::DEFAULT_TYPE);
        ExpandButton::new_expandable_expandable_type_global_expand_button(
            expandable1,
            expandable2,
            type_,
            global_expand_button,
        )
    }

    /// Java static `getExpandedInstance(Expandable, Expandable, ExpandButton.Type)`.
    pub fn get_expanded_instance(
        expandable1: Option<Weak<dyn Expandable>>,
        expandable2: Option<Weak<dyn Expandable>>,
        type_: Option<&'static Type>,
    ) -> Rc<ExpandButton> {
        let type_ = type_.unwrap_or(Self::DEFAULT_TYPE);
        ExpandButton::new_expandable_expandable_type_boolean_global_expand_button(
            expandable1,
            expandable2,
            type_,
            true,
            None,
        )
    }

    /// Java private `ExpandButton(Expandable, Expandable, Type, GlobalExpandButton)`.
    fn new_expandable_expandable_type_global_expand_button(
        expandable1: Option<Weak<dyn Expandable>>,
        expandable2: Option<Weak<dyn Expandable>>,
        type_: &'static Type,
        global_expand_button: Option<Rc<GlobalExpandButton>>,
    ) -> Rc<ExpandButton> {
        ExpandButton::new_expandable_expandable_type_boolean_global_expand_button(
            expandable1,
            expandable2,
            type_,
            false,
            global_expand_button,
        )
    }

    /// Java private
    /// `ExpandButton(Expandable, Expandable, Type, boolean, GlobalExpandButton)`.
    fn new_expandable_expandable_type_boolean_global_expand_button(
        expandable1: Option<Weak<dyn Expandable>>,
        expandable2: Option<Weak<dyn Expandable>>,
        type_: &'static Type,
        expanded: bool,
        global_expand_button: Option<Rc<GlobalExpandButton>>,
    ) -> Rc<ExpandButton> {
        // The final fields are in place before `super(...)` runs here; the
        // overridden methods `super` calls (`setName`) do not read them.
        let instance = Rc::new(ExpandButton {
            base: SingleLineButton::new_fields(None, false, None, true),
            self_ref: RefCell::new(Weak::new()),
            type_,
            expandable1,
            expandable2,
            global_expand_button,
            expanded: Cell::new(expanded),
            jpanel_container: RefCell::new(None),
            debug: Cell::new(false),
        });
        *instance.self_ref.borrow_mut() = Rc::downgrade(&instance);
        // Java `super(null, false, null, true)`.
        MultiLineButton::construct(&instance, false);
        instance.base.constructor_body(None, true);
        // Registor with the global expand button, if it exists.
        if let Some(global_expand_button) = &instance.global_expand_button {
            global_expand_button.register_expand_button(&instance);
        }
        instance.set_manual_name();
        instance.set_text(Some(type_.get_symbol(expanded)));
        instance.set_tool_tip_text(Some(type_.get_tool_tip(expanded)));
        // Java `addActionListener(new ExpandButtonActionListener(this))`.
        let adaptee = Rc::downgrade(&instance);
        instance.add_action_listener(Rc::new(move |_event| {
            if let Some(adaptee) = adaptee.upgrade() {
                // ExpandButtonActionListener.actionPerformed
                adaptee.button_action();
            }
        }));
        // Swing layout: setBorder(BorderFactory.createBevelBorder(BevelBorder.RAISED));
        // size = getPreferredSize(), widened to a square; setSize(size).
        instance
    }

    /// This object as an `Rc` (Java `this` passed to another object).
    fn this(&self) -> Rc<ExpandButton> {
        self.self_ref
            .borrow()
            .upgrade()
            .expect("ExpandButton used before construction or after drop")
    }

    /// Java public static `equals(AbstractButton, String)`.
    pub fn equals_abstract_button_string(
        button: Option<&Rc<JComponent>>,
        input: Option<&str>,
    ) -> bool {
        Type::equals(button, input)
    }

    /// Java `setName(String)` (overrides `MultiLineButton.setName`).
    pub fn set_name(&self, associated_label: Option<&str>) {
        let field_type = UITestFieldType::MINI_BUTTON;
        let name = utilities::convert_label_to_name(
            associated_label,
            field_type.is_unlimited_segments(),
        );
        // Java string concatenation writes a null name as "null".
        self.get_button().set_name(Some(&format!(
            "{}{}{}",
            field_type.to_string(),
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        // Java `EtomoDirector.INSTANCE.getArguments()` is the `ARGUMENTS` static.
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `isExpanded()`.
    pub fn is_expanded(&self) -> bool {
        self.expanded.get()
    }

    /// Java `createButtonStateKey(DialogType)` (overrides
    /// `MultiLineButton.createButtonStateKey`).
    pub fn create_button_state_key(&self, dialog_type: Option<DialogType>) -> Option<String> {
        // Upstream bug fixed (ExpandButton.java, createButtonStateKey): Java
        // dereferences `dialogType` unconditionally and throws
        // NullPointerException for null.  No key can be made without a dialog
        // type, so null gives null and the state key is left unchanged.
        let dialog_type = dialog_type?;
        let state_key = format!(
            "{}.{}.{}",
            dialog_type.get_storable_name(),
            self.get_name().as_deref().unwrap_or("null"),
            self.type_.get_expanded_state()
        );
        self.set_state_key(Some(state_key.clone()));
        Some(state_key)
    }

    /// Java `getButtonState()` (overrides `MultiLineButton.getButtonState`).
    pub fn get_button_state(&self) -> bool {
        self.is_expanded()
    }

    /// Java `setButtonState(boolean)` (overrides `MultiLineButton.setButtonState`).
    pub fn set_button_state(&self, state: bool) {
        self.set_original_process_result_display_state(state);
        self.set_expanded(state);
    }

    /// Java `getState()`.
    pub fn get_state(&self) -> &'static str {
        self.type_.get_state(self.expanded.get())
    }

    /// Java `setState(String)`.
    pub fn set_state(&self, state: Option<&str>) {
        let Some(state) = state else {
            return;
        };
        if state == self.type_.get_expanded_state() && !self.expanded.get() {
            self.set_expanded(true);
        } else if state == self.type_.get_contracted_state() && self.expanded.get() {
            self.set_expanded(false);
        }
    }

    /// Java `add(JPanel, GridBagLayout, GridBagConstraints)`.  The layout and
    /// constraints are not modelled.
    pub fn add(&self, panel: &Rc<JComponent>) {
        // Swing layout: weightx 0 for this component's constraints, restored after.
        panel.add(&self.get_button());
        *self.jpanel_container.borrow_mut() = Some(panel.clone());
    }

    /// Java `remove()`.
    pub fn remove(&self) {
        let jpanel_container = self.jpanel_container.borrow_mut().take();
        if let Some(jpanel_container) = jpanel_container {
            jpanel_container.remove(&self.get_button());
        }
    }

    /// Java public `equals(ExpandButton)`: `equals((Object) that)`, identity.
    pub fn equals_expand_button(&self, that: Option<&ExpandButton>) -> bool {
        that.is_some_and(|that| std::ptr::eq(self, that))
    }

    /// Java `setExpanded(boolean)`.
    pub fn set_expanded(&self, expanded: bool) {
        // prevent buttonAction from ignoring an unchanged button value
        if self.expanded.get() == expanded {
            self.expanded.set(!expanded);
        }
        self.button_action();
    }

    /// Java `update(boolean)`.
    pub fn update(&self, expanded: bool) {
        if self.expanded.get() == expanded {
            // Nothing to update
            return;
        }
        self.expanded.set(expanded);
        self.set_text(Some(self.type_.get_symbol(expanded)));
        self.set_tool_tip_text(Some(self.type_.get_tool_tip(expanded)));
    }

    /// Java private `buttonAction()`.
    fn button_action(&self) {
        self.expanded.set(!self.expanded.get());
        let expanded = self.expanded.get();
        self.set_text(Some(self.type_.get_symbol(expanded)));
        self.set_tool_tip_text(Some(self.type_.get_tool_tip(expanded)));
        let this = self.this();
        if let Some(expandable1) = self.expandable1.as_ref().and_then(Weak::upgrade) {
            expandable1.expand_expand_button(&this);
        }
        if let Some(expandable2) = self.expandable2.as_ref().and_then(Weak::upgrade) {
            expandable2.expand_expand_button(&this);
        }
        // Tell global expand button about this action.
        if let Some(global_expand_button) = &self.global_expand_button {
            global_expand_button.msg_expand_button_action(&this, self.expanded.get());
        }
    }
}

/// Java `implements Expander`.
impl Expander for ExpandButton {
    fn is_expanded(&self) -> bool {
        ExpandButton::is_expanded(self)
    }
}

/// Java `SwingComponent.getComponent()`, inherited from `MultiLineButton`.
impl SwingComponent for ExpandButton {
    fn get_component(&self) -> Rc<JComponent> {
        self.base.base.get_component()
    }
}

/// Java `UIComponent`, inherited from `MultiLineButton`.
impl UIComponent for ExpandButton {
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    fn get_component(&self) -> Rc<JComponent> {
        self.base.base.get_component()
    }
}

/// Java static final nested class `ExpandButton.Type`.  The instances are the
/// associated constants [`Type::MORE`], [`Type::ADVANCED`] and
/// [`Type::OPEN`], used as `&'static Type`; Java compares them by identity,
/// which the derived equality over all fields reproduces for these three.
#[derive(Debug, PartialEq, Eq)]
pub struct Type {
    /// Java final `expandedState`.  Backwards compatibility issue:
    /// expandedState is a key in the .edf file.
    expanded_state: &'static str,
    /// Java final `expandedSymbol`.
    expanded_symbol: &'static str,
    /// Java final `expandedToolTip`.
    expanded_tool_tip: &'static str,
    /// Java final `contractedState`.
    contracted_state: &'static str,
    /// Java final `contractedSymbol`.
    contracted_symbol: &'static str,
    /// Java final `contractedToolTip`.
    contracted_tool_tip: &'static str,
}

/// Java `Type.HTML_TAG`.
const HTML_TAG: &str = "<html>";
/// Java `Type.MORE_EXPANDED_TEXT`.
const MORE_EXPANDED_TEXT: &str = "&lt";
/// Java `Type.MORE_EXPANDED_SYMBOL`.
const MORE_EXPANDED_SYMBOL: &str = "<";
/// Java `Type.MORE_CONTRACTED_TEXT`.
const MORE_CONTRACTED_TEXT: &str = "&gt";
/// Java `Type.MORE_CONTRACTED_SYMBOL`.
const MORE_CONTRACTED_SYMBOL: &str = ">";
/// Java `Type.ADVANCED_EXPANDED_SYMBOL`.
const ADVANCED_EXPANDED_SYMBOL: &str = "B";
/// Java `Type.ADVANCED_CONTRACTED_SYMBOL`.
const ADVANCED_CONTRACTED_SYMBOL: &str = "A";
/// Java `Type.OPEN_EXPANDED_SYMBOL`.
const OPEN_EXPANDED_SYMBOL: &str = "-";
/// Java `Type.OPEN_CONTRACTED_SYMBOL`.
const OPEN_CONTRACTED_SYMBOL: &str = "+";

impl Type {
    /// Java `Type.MORE`.
    pub const MORE: Type = Type::new(
        "more",
        // HTML_TAG + MORE_EXPANDED_TEXT
        "<html>&lt",
        "Show less.",
        "less",
        // HTML_TAG + MORE_CONTRACTED_TEXT
        "<html>&gt",
        "Show more.",
    );
    /// Java `Type.ADVANCED`.
    pub const ADVANCED: Type = Type::new(
        "advanced",
        ADVANCED_EXPANDED_SYMBOL,
        "Show basic options.",
        "basic",
        ADVANCED_CONTRACTED_SYMBOL,
        "Show all options.",
    );
    /// Java `Type.OPEN`.
    pub const OPEN: Type = Type::new(
        "open",
        OPEN_EXPANDED_SYMBOL,
        "Close panel.",
        "closed",
        OPEN_CONTRACTED_SYMBOL,
        "Open panel.",
    );

    /// Java private `Type(String, String, String, String, String, String)`.
    const fn new(
        expanded_state: &'static str,
        expanded_symbol: &'static str,
        expanded_tool_tip: &'static str,
        contracted_state: &'static str,
        contracted_symbol: &'static str,
        contracted_tool_tip: &'static str,
    ) -> Type {
        Type {
            expanded_state,
            expanded_symbol,
            expanded_tool_tip,
            contracted_state,
            contracted_symbol,
            contracted_tool_tip,
        }
    }

    /// Java public static `equals(AbstractButton, String)`.
    pub fn equals(button: Option<&Rc<JComponent>>, input: Option<&str>) -> bool {
        let (Some(button), Some(input)) = (button, input) else {
            return false;
        };
        // String html stuff off the label.
        let Some(text) = utilities::strip_html_tags(Some(&button.get_text())) else {
            return false;
        };
        let mut symbol: Option<&str> = None;
        if text == MORE_EXPANDED_TEXT {
            symbol = Some(MORE_EXPANDED_SYMBOL);
        } else if text == MORE_CONTRACTED_TEXT {
            symbol = Some(MORE_CONTRACTED_SYMBOL);
        } else if text == ADVANCED_EXPANDED_SYMBOL
            || text == ADVANCED_CONTRACTED_SYMBOL
            || text == OPEN_EXPANDED_SYMBOL
            || text == OPEN_CONTRACTED_SYMBOL
        {
            symbol = Some(text.as_str());
        }
        let Some(symbol) = symbol else {
            return false;
        };
        symbol == input
    }

    /// Java private `getUnformattedText(boolean)`.  Used only by the
    /// (layout-only) `getPreferredWidth`.
    fn get_unformatted_text(&self, expanded: bool) -> &'static str {
        if *self == Type::MORE {
            if expanded {
                return MORE_EXPANDED_SYMBOL;
            }
            return MORE_CONTRACTED_SYMBOL;
        }
        if expanded {
            return self.expanded_symbol;
        }
        self.contracted_symbol
    }

    /// Java private `getState(boolean)`.
    fn get_state(&self, expanded: bool) -> &'static str {
        if expanded {
            return self.expanded_state;
        }
        self.contracted_state
    }

    /// Java private `getExpandedState()`.
    fn get_expanded_state(&self) -> &'static str {
        self.expanded_state
    }

    /// Java private `getContractedState()`.
    fn get_contracted_state(&self) -> &'static str {
        self.contracted_state
    }

    /// Java private `getSymbol(boolean)`.
    fn get_symbol(&self, expanded: bool) -> &'static str {
        if expanded {
            return self.expanded_symbol;
        }
        self.contracted_symbol
    }

    /// Java private `getToolTip(boolean)`.
    fn get_tool_tip(&self, expanded: bool) -> &'static str {
        if expanded {
            return self.expanded_tool_tip;
        }
        self.contracted_tool_tip
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn more_type_concatenates_the_html_tag() {
        assert_eq!(Type::MORE.expanded_symbol, format!("{HTML_TAG}{MORE_EXPANDED_TEXT}"));
        assert_eq!(Type::MORE.contracted_symbol, format!("{HTML_TAG}{MORE_CONTRACTED_TEXT}"));
        assert_eq!(Type::MORE.get_unformatted_text(true), "<");
        assert_eq!(Type::OPEN.get_state(false), "closed");
    }
}
