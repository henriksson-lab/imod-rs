//! `IMOD/Etomo/src/etomo/ui/swing/MultiLineButton.java`.
//!
//! A button (a `JButton`, or a `JToggleButton` for a toggle button) that can
//! show its label on two lines, and that is a `ProcessResultDisplay`: a
//! process button whose selected ("done") state follows the process it
//! starts and is saved in the screen state under its button state key.
//!
//! # Object model
//!
//! `MultiLineButton` is an EDT object built as an `Rc` (every method takes
//! `&self`).  Java subclasses (`SingleLineButton`, `Run3dmodButton`,
//! `ExpandButton`, `Run3dmodSingleLineButton`, `MenuButton`) embed it as field
//! `base` and deref to it.  The methods a subclass overrides and this class
//! calls (`newButton`, `setupButton`, `setName`, `setTextLabel`,
//! `createButtonStateKey`, `getButtonState`, `setButtonState`) are the trait
//! [`MultiLineButtonVirtual`], whose default bodies are the Java
//! `MultiLineButton` bodies; this struct keeps `this`, a `Weak` to the most
//! derived object, and dispatches through it.  The inherent methods with
//! those names on `MultiLineButton` are the virtual call sites.
//!
//! Java calls `newButton()` and `setupButton()` from the constructor, before
//! a subclass constructor body runs.  Rust cannot dispatch virtually before
//! the `Rc` exists, so construction is split: [`MultiLineButton::new_fields`]
//! is the field initialisation and the assignments before `newButton()`, the
//! subclass puts the result into its own struct (with its own final fields,
//! which the overridden methods never read during construction), wraps it in
//! an `Rc`, and then [`MultiLineButton::construct`] runs the rest of the Java
//! constructor with `this` set.  The subclass constructor body runs after it,
//! as in Java.
//!
//! Every `MultiLineButtonVirtual` implementor is a `ProcessResultDisplay`
//! through the blanket implementation at the end of this file (all of the
//! Java class's `ProcessResultDisplay` methods are `final`), so a subclass
//! must not implement `ProcessResultDisplay` itself.
//!
//! # Not modelled
//!
//! Layout, painting, fonts and focus are not modelled by `jdk.rs`, so these
//! Java members have no Rust counterpart: `setFocusable`, `getPreferredWidth()`
//! and `getPreferredWidth(String)`, `getWidth`, `getHeight`, `setIcon`,
//! `setBorder`, `setBorderPainted`, `getBorder`, `getPreferredSize`,
//! `setAlignmentX`, `setAlignmentY`, `isDisplayable`, `addMouseListener`,
//! `setSize` and the `background` field.  The label division (two `JLabel`s
//! inside the button) does depend on font metrics and on the button width
//! from `UIParameters`; it is translated, through those units.

use std::cell::{Cell, OnceCell, RefCell};
use std::rc::{Rc, Weak};
use std::sync::LazyLock;
use std::sync::atomic::AtomicI32;

use regex::Regex;

use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::{ActionListener, JComponent};
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::process_end_state::ProcessEndState;
use crate::imod::etomo::r#type::process_result::ProcessResult;
use crate::imod::etomo::r#type::process_result_display::{
    ProcessResultDisplay, ProcessResultDisplayHandle,
};
use crate::imod::etomo::r#type::process_result_display_state::ProcessResultDisplayState;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

use super::color_tool::ColorTool;
use super::swing_component::SwingComponent;
use super::tooltip_formatter::TooltipFormatter;
use super::ui_parameters::UIParameters;
use super::ui_utilities;
use crate::imod::etomo::jdk::FontMetrics;

/// Java private static final `maxPrint`.  Unused in the Java class.
const MAX_PRINT: i32 = 10;
/// Java private static `printed`.  Unused in the Java class.
static PRINTED: AtomicI32 = AtomicI32::new(0);

/// Java `text.split("\\s*\\Q\\n\\E\\s*")`: a literal backslash-n with any
/// surrounding (ASCII, as Java's `\s`) whitespace.
static NEWLINE_SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]*\\n[ \t\n\x0B\x0C\r]*").unwrap());
/// Java `labelText.split("\\s")`.
static WHITESPACE_SPLIT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[ \t\n\x0B\x0C\r]").unwrap());
/// Java `curWord.split(HYPHEN)`.
static HYPHEN_SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new("-").unwrap());

/// The methods Java subclasses override and `MultiLineButton` calls.  The
/// default bodies are the Java `MultiLineButton` bodies.  Inside them, a call
/// to one of these methods goes through the inherent method of the same name
/// on [`MultiLineButton`], which dispatches to the most derived object.
pub trait MultiLineButtonVirtual {
    /// The embedded `MultiLineButton` (the root of the Java `super` chain).
    fn get_multi_line_button(&self) -> &MultiLineButton;

    /// Java package-private `newButton()`.
    fn new_button(&self) -> Rc<JComponent> {
        let base = self.get_multi_line_button();
        if base.toggle_button {
            return JComponent::new_toggle_button("");
        }
        JComponent::new_button("")
    }

    /// Java package-private `setupButton(boolean)`.
    fn setup_button(&self, set_minimum_size: bool) {
        let base = self.get_multi_line_button();
        base.set_size(set_minimum_size);
        let unformatted_label = base.unformatted_label.borrow().clone();
        if unformatted_label.is_some() {
            base.set_text(unformatted_label.as_deref());
        }
    }

    /// Java package-private `createButtonStateKey(DialogType)`.
    fn create_button_state_key(&self, dialog_type: Option<DialogType>) -> Option<String> {
        let base = self.get_multi_line_button();
        if let Some(dialog_type) = dialog_type {
            // Java string concatenation writes a null name as "null".
            *base.state_key.borrow_mut() = Some(format!(
                "{}.{}.done",
                dialog_type.get_storable_name(),
                base.get_button().get_name().as_deref().unwrap_or("null")
            ));
        }
        base.state_key.borrow().clone()
    }

    /// Java `setButtonState(boolean)`.
    fn set_button_state(&self, state: bool) {
        let base = self.get_multi_line_button();
        base.set_original_process_result_display_state(state);
        base.set_selected(state);
    }

    /// Java package-private `getButtonState()`.
    fn get_button_state(&self) -> bool {
        self.get_multi_line_button().is_selected()
    }

    /// Java package-private `setName(String)`.
    fn set_name(&self, label: Option<&str>) {
        let base = self.get_multi_line_button();
        let field_type = UITestFieldType::BUTTON;
        let name =
            utilities::convert_label_to_name(label, field_type.is_unlimited_segments());
        // Java string concatenation writes a null name as "null".
        base.get_button().set_name(Some(&format!(
            "{}{}{}",
            field_type.to_string(),
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        // Java `EtomoDirector.INSTANCE.getArguments()` is the `ARGUMENTS` static.
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                base.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java package-private `setTextLabel(String)`.
    fn set_text_label(&self, text: Option<&str>) {
        let base = self.get_multi_line_button();
        let button = base.get_button();
        let mut text1: Option<String> = None;
        let mut text2: Option<String> = None;
        if text.is_some_and(|text| text.contains("\\n")) {
            let labels = utilities::java_lang_string_split(text.unwrap(), &NEWLINE_SPLIT);
            if labels.len() >= 2 {
                text1 = Some(labels[0].clone());
                text2 = Some(labels[1].clone());
            }
        } else if text.is_some() && base.width.get() != -1 {
            if base.font_metrics.get().is_none() {
                base.font_metrics
                    .set(ui_utilities::get_font_metrics_abstract_button(&button));
            }
            if base.font_metrics.get().is_some() {
                let mut label_divider = LabelDivider::new();
                label_divider.divide(base, text.unwrap());
                // Java keeps a commented-out earlier implementation of the
                // division here (stringWidth against width - 9, breaking at the
                // closest space or hyphen); it is not compiled.
                text1 = label_divider.get_line1();
                text2 = label_divider.get_line2();
            }
        }
        match text2 {
            None => {
                if base.label1.borrow().is_none() {
                    // Java `setText(null)`: the stand-in holds no null text.
                    button.set_text(text.unwrap_or(""));
                } else {
                    let label1 = base.label1.borrow_mut().take();
                    let label2 = base.label2.borrow_mut().take();
                    if let Some(label1) = label1 {
                        button.remove(&label1);
                    }
                    if let Some(label2) = label2 {
                        button.remove(&label2);
                    }
                    button.set_text(text.unwrap_or(""));
                }
            }
            Some(text2) => {
                button.set_text("");
                if !base.action_command_set.get() {
                    button.set_action_command(text);
                }
                let labels = (base.label1.borrow().clone(), base.label2.borrow().clone());
                if let (Some(label1), Some(label2)) = labels {
                    label1.set_text(text1.as_deref().unwrap_or(""));
                    label2.set_text(&text2);
                    button.add(&label2); // May have been removed
                } else {
                    let label1 = JComponent::new_label(text1.as_deref().unwrap_or(""));
                    // Swing layout: label1.setHorizontalAlignment(JLabel.CENTER).
                    let label2 = JComponent::new_label(&text2);
                    // Swing layout: label2.setHorizontalAlignment(JLabel.CENTER);
                    // button.setLayout(new BorderLayout()); the labels go NORTH and SOUTH.
                    *base.label1.borrow_mut() = Some(label1.clone());
                    *base.label2.borrow_mut() = Some(label2.clone());
                    button.add(&label1);
                    button.add(&label2);
                }
            }
        }
    }
}

/// Java `MultiLineButton`.
pub struct MultiLineButton {
    /// The most derived object, for the virtual calls.
    this: RefCell<Weak<dyn MultiLineButtonVirtual>>,

    /// Java final `button` (set by `newButton()` in the constructor).
    button: OnceCell<Rc<JComponent>>,
    /// Java final `toggleButton`.
    toggle_button: bool,
    /// Java final `processResultDisplayState` (set in the constructor).
    process_result_display_state: OnceCell<ProcessResultDisplayState>,
    // Java final `background` (`button.getBackground()`): painting, not modelled.
    /// Java final `html`.
    html: bool,

    /// Java `screenState`.
    screen_state: Cell<Option<&'static BaseScreenState>>,
    /// Java `dialogType`.
    dialog_type: Cell<Option<DialogType>>,
    /// Java `stateKey`.
    state_key: RefCell<Option<String>>,
    /// Java `manualName`.
    manual_name: Cell<bool>,
    /// Java `buttonForeground`.  Never assigned in the Java class; read by
    /// `dumpState`.
    button_foreground: Cell<Option<(u8, u8, u8)>>,
    /// Java `buttonHighlightForeground`.  Never assigned in the Java class;
    /// read by `dumpState`.
    button_highlight_foreground: Cell<Option<(u8, u8, u8)>>,
    /// Java `debug`.
    debug: Cell<bool>,
    /// Java `unformattedLabel`.
    unformatted_label: RefCell<Option<String>>,
    /// Java `fontMetrics`.
    font_metrics: Cell<Option<FontMetrics>>,
    /// Java `enabled`.
    enabled: Cell<bool>,
    /// Java `editable`.
    editable: Cell<bool>,
    /// Java `width`.
    width: Cell<i32>,
    /// Java `label1`.
    label1: RefCell<Option<Rc<JComponent>>>,
    /// Java `label2`.
    label2: RefCell<Option<Rc<JComponent>>>,
    /// Java `actionCommandSet`.
    action_command_set: Cell<bool>,
    /// Java `outputImageFileKey`.
    output_image_file_key: RefCell<Option<FileKey>>,
}

impl MultiLineButtonVirtual for MultiLineButton {
    fn get_multi_line_button(&self) -> &MultiLineButton {
        self
    }
}

impl MultiLineButton {
    /// The Java field initialisers and the constructor statements before
    /// `button = newButton()` of
    /// `MultiLineButton(String, boolean, DialogType, boolean, boolean, boolean, FileKey)`
    /// (`setMinimumSize` is used by [`MultiLineButton::construct`]).
    pub fn new_fields(
        label: Option<&str>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        html: bool,
        debug: bool,
        output_image_file_key: Option<FileKey>,
    ) -> MultiLineButton {
        let this: Weak<MultiLineButton> = Weak::new();
        MultiLineButton {
            this: RefCell::new(this as Weak<dyn MultiLineButtonVirtual>),
            button: OnceCell::new(),
            toggle_button,
            process_result_display_state: OnceCell::new(),
            html,
            screen_state: Cell::new(None),
            dialog_type: Cell::new(dialog_type),
            state_key: RefCell::new(None),
            manual_name: Cell::new(false),
            button_foreground: Cell::new(None),
            button_highlight_foreground: Cell::new(None),
            debug: Cell::new(debug),
            unformatted_label: RefCell::new(label.map(str::to_owned)),
            font_metrics: Cell::new(None),
            enabled: Cell::new(true),
            editable: Cell::new(true),
            width: Cell::new(-1),
            label1: RefCell::new(None),
            label2: RefCell::new(None),
            action_command_set: Cell::new(false),
            output_image_file_key: RefCell::new(output_image_file_key),
        }
    }

    /// The rest of the Java constructor, from `button = newButton()` on, run
    /// once the most derived object exists.  Sets `this`.
    pub fn construct<T: MultiLineButtonVirtual + 'static>(this: &Rc<T>, set_minimum_size: bool) {
        let base = this.get_multi_line_button();
        let weak: Weak<T> = Rc::downgrade(this);
        *base.this.borrow_mut() = weak.clone() as Weak<dyn MultiLineButtonVirtual>;
        let _ = base.button.set(base.new_button());
        base.setup_button(set_minimum_size);
        base.init();
        let display: Weak<dyn ProcessResultDisplay> = weak;
        let _ = base
            .process_result_display_state
            .set(ProcessResultDisplayState::new(display));
        // Java `String text = button.getText();` is an unused local.
        // Swing painting: background = button.getBackground().
    }

    /// Java package-private
    /// `MultiLineButton(String, boolean, DialogType, boolean, boolean, boolean, FileKey)`.
    pub fn new_string_boolean_dialog_type_boolean_boolean_boolean_file_key(
        label: Option<&str>,
        toggle_button: bool,
        dialog_type: Option<DialogType>,
        set_minimum_size: bool,
        html: bool,
        debug: bool,
        output_image_file_key: Option<FileKey>,
    ) -> Rc<MultiLineButton> {
        let instance = Rc::new(MultiLineButton::new_fields(
            label,
            toggle_button,
            dialog_type,
            html,
            debug,
            output_image_file_key,
        ));
        MultiLineButton::construct(&instance, set_minimum_size);
        instance
    }

    /// Java package-private `MultiLineButton()`.
    pub fn new_void() -> Rc<MultiLineButton> {
        Self::new_string_boolean_dialog_type_boolean_boolean_boolean_file_key(
            None, false, None, false, false, false, None,
        )
    }

    /// Java package-private `MultiLineButton(String)`.
    pub fn new_string(label: Option<&str>) -> Rc<MultiLineButton> {
        Self::new_string_boolean_dialog_type_boolean_boolean_boolean_file_key(
            label, false, None, false, false, false, None,
        )
    }

    /// Java package-private `MultiLineButton(String, FileKey)`.
    pub fn new_string_file_key(
        label: Option<&str>,
        output_image_file_key: Option<FileKey>,
    ) -> Rc<MultiLineButton> {
        Self::new_string_boolean_dialog_type_boolean_boolean_boolean_file_key(
            label,
            false,
            None,
            false,
            false,
            false,
            output_image_file_key,
        )
    }

    /// Java package-private `MultiLineButton(boolean, String)`.
    pub fn new_boolean_string(set_minimum_size: bool, label: Option<&str>) -> Rc<MultiLineButton> {
        Self::new_string_boolean_dialog_type_boolean_boolean_boolean_file_key(
            label,
            false,
            None,
            set_minimum_size,
            false,
            false,
            None,
        )
    }

    /// Java private `MultiLineButton(String, boolean)`.
    fn new_string_boolean(label: Option<&str>, toggle_button: bool) -> Rc<MultiLineButton> {
        Self::new_string_boolean_dialog_type_boolean_boolean_boolean_file_key(
            label,
            toggle_button,
            None,
            false,
            false,
            true,
            None,
        )
    }

    /// Java static `getDebugInstance(String)`.
    pub fn get_debug_instance(label: Option<&str>) -> Rc<MultiLineButton> {
        Self::new_string_boolean_dialog_type_boolean_boolean_boolean_file_key(
            label, false, None, false, false, true, None,
        )
    }

    /// Java static `getToggleButtonInstance()`.
    pub fn get_toggle_button_instance_void() -> Rc<MultiLineButton> {
        Self::new_string_boolean_dialog_type_boolean_boolean_boolean_file_key(
            None, true, None, false, false, false, None,
        )
    }

    /// Java public static `getToggleButtonInstance(String, DialogType)`.
    pub fn get_toggle_button_instance_string_dialog_type(
        label: Option<&str>,
        dialog_type: Option<DialogType>,
    ) -> Rc<MultiLineButton> {
        Self::new_string_boolean_dialog_type_boolean_boolean_boolean_file_key(
            label,
            true,
            dialog_type,
            false,
            false,
            false,
            None,
        )
    }

    /// Java static `getToggleButtonInstance(String)`.
    pub fn get_toggle_button_instance_string(label: Option<&str>) -> Rc<MultiLineButton> {
        Self::new_string_boolean(label, true)
    }

    /// The most derived object (Java `this` in a virtual call).
    fn this(&self) -> Rc<dyn MultiLineButtonVirtual> {
        self.this
            .borrow()
            .upgrade()
            .expect("MultiLineButton used before construct() or after drop")
    }

    /// Java field `processResultDisplayState`.
    fn process_result_display_state(&self) -> &ProcessResultDisplayState {
        self.process_result_display_state
            .get()
            .expect("MultiLineButton used before construct()")
    }

    // --- virtual call sites ---

    /// Java `newButton()` (virtual).
    fn new_button(&self) -> Rc<JComponent> {
        self.this().new_button()
    }

    /// Java `setupButton(boolean)` (virtual).
    fn setup_button(&self, set_minimum_size: bool) {
        self.this().setup_button(set_minimum_size)
    }

    /// Java `createButtonStateKey(DialogType)` (virtual).
    pub fn create_button_state_key(&self, dialog_type: Option<DialogType>) -> Option<String> {
        self.this().create_button_state_key(dialog_type)
    }

    /// Java `setButtonState(boolean)` (virtual).
    pub fn set_button_state(&self, state: bool) {
        self.this().set_button_state(state)
    }

    /// Java `getButtonState()` (virtual).
    pub fn get_button_state(&self) -> bool {
        self.this().get_button_state()
    }

    /// Java `setName(String)` (virtual).
    pub fn set_name(&self, label: Option<&str>) {
        self.this().set_name(label)
    }

    /// Java `setTextLabel(String)` (virtual).
    pub fn set_text_label(&self, text: Option<&str>) {
        self.this().set_text_label(text)
    }

    // --- the rest of the Java class ---

    /// Java `dumpState()`.
    pub fn dump_state(&self) {
        let color = |color: Option<(u8, u8, u8)>| match color {
            None => "null".to_owned(),
            // Java `Color.toString()`.
            Some((r, g, b)) => format!("java.awt.Color[r={r},g={g},b={b}]"),
        };
        eprint!(
            "[toggleButton:{},stateKey:{},\nmanualName:{},buttonForeground:{},\nbuttonHighlightForeground:{},debug:{},\nunformattedLabel:{}]",
            self.toggle_button,
            self.state_key.borrow().as_deref().unwrap_or("null"),
            self.manual_name.get(),
            color(self.button_foreground.get()),
            color(self.button_highlight_foreground.get()),
            self.debug.get(),
            self.unformatted_label.borrow().as_deref().unwrap_or("null")
        );
    }

    /// Java `setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.debug.set(input);
        self.process_result_display_state().set_debug(input);
    }

    /// Java `isDebug()`.
    pub fn is_debug(&self) -> bool {
        self.debug.get()
    }

    /// Java `getOutputImageFileKey()`.
    pub fn get_output_image_file_key(&self) -> Option<FileKey> {
        self.output_image_file_key.borrow().clone()
    }

    /// Java `setOutputImageFileKey(FileKey)`.
    pub fn set_output_image_file_key(&self, output_image_file_key: Option<FileKey>) {
        *self.output_image_file_key.borrow_mut() = output_image_file_key;
    }

    /// Java `isHtml()`.
    pub fn is_html(&self) -> bool {
        self.html
    }

    /// Java `doClick()`.
    pub fn do_click(&self) {
        self.get_button().do_click();
    }

    /// Java `setHighlight(boolean)`: an empty body.
    pub fn set_highlight(&self, _highlight: bool) {}

    /// Java `getButtonStateKey()`.
    pub fn get_button_state_key(&self) -> Option<String> {
        let dialog_type = self.dialog_type.get();
        if self.state_key.borrow().is_none() && dialog_type.is_some() {
            return self.create_button_state_key(dialog_type);
        }
        self.state_key.borrow().clone()
    }

    /// Java `setOriginalProcessResultDisplayState(boolean)`.
    pub fn set_original_process_result_display_state(&self, state: bool) {
        self.process_result_display_state().set_original_state(state);
    }

    /// Java `equalsID(int, String)`.
    pub fn equals_id(&self, display_id: i32, factory_id: Option<&str>) -> bool {
        self.process_result_display_state()
            .equals_id(display_id, factory_id)
    }

    /// Java `setID(int, String)`.
    pub fn set_id(&self, display_id: i32, factory_id: Option<String>) {
        self.process_result_display_state()
            .set_id(display_id, factory_id);
    }

    /// Java `setFactoryID(String)`.
    pub fn set_factory_id(&self, factory_id: Option<String>) {
        self.process_result_display_state()
            .set_factory_id(factory_id);
    }

    /// Java `getDisplayID()`.
    pub fn get_display_id(&self) -> i32 {
        self.process_result_display_state().get_display_id()
    }

    /// Java `getFactoryID()`.
    pub fn get_factory_id(&self) -> Option<String> {
        self.process_result_display_state().get_factory_id()
    }

    /// Java `getNext()`.
    pub fn get_next(&self) -> Option<ProcessResultDisplayHandle> {
        self.process_result_display_state().get_next()
    }

    /// Java `setNext(ProcessResultDisplay)`.
    pub fn set_next(&self, display: Option<ProcessResultDisplayHandle>) {
        self.process_result_display_state().set_next(display);
    }

    /// Java `setUseGlobalDependencyList(boolean)`.
    pub fn set_use_global_dependency_list(&self, use_: bool) {
        self.process_result_display_state()
            .set_use_global_dependency_list(use_);
    }

    /// Java `setManualName()`.
    pub fn set_manual_name(&self) {
        self.manual_name.set(true);
    }

    /// Java `getButton()`.
    pub fn get_button(&self) -> Rc<JComponent> {
        self.button
            .get()
            .expect("MultiLineButton used before construct()")
            .clone()
    }

    /// Java `getName()`.
    pub fn get_name(&self) -> Option<String> {
        self.get_button().get_name()
    }

    /// Java `setStateKey(String)`.
    pub fn set_state_key(&self, state_key: Option<String>) {
        *self.state_key.borrow_mut() = state_key;
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
        let button = self.get_button();
        button.set_enabled(enabled && self.editable.get());
        if !self.toggle_button || self.html {
            let color = ColorTool::get_button_instance(self.toggle_button)
                .get_foreground(self.is_enabled());
            button.set_foreground(Some(color));
        }
        let labels = (self.label1.borrow().clone(), self.label2.borrow().clone());
        if let (Some(label1), Some(label2)) = labels {
            label1.set_enabled(enabled);
            label2.set_enabled(enabled);
        }
    }

    /// Java `setEditable(boolean)`.
    pub fn set_editable(&self, editable: bool) {
        self.editable.set(editable);
        if self.enabled.get() {
            self.get_button().set_enabled(editable);
        }
    }

    /// Java `isEnabled()`.
    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }

    /// Java `isToggleButton()`.
    pub fn is_toggle_button(&self) -> bool {
        self.toggle_button
    }

    /// Java `setText(String)`.
    pub fn set_text(&self, text: Option<&str>) {
        if !self.manual_name.get() {
            self.set_name(text);
        }
        *self.unformatted_label.borrow_mut() = text.map(str::to_owned);
        self.set_text_label(text);
    }

    /// Java `toString()`.
    pub fn to_string(&self) -> Option<String> {
        self.get_text()
    }

    /// Java `getUnformattedLabel()`.
    pub fn get_unformatted_label(&self) -> Option<String> {
        self.unformatted_label.borrow().clone()
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, action_listener: ActionListener) {
        self.get_button().add_action_listener(action_listener);
    }

    /// Java `setActionCommand(String)`.  `Run3dmodButton` overrides it only to
    /// call this.
    pub fn set_action_command(&self, action_command: Option<&str>) {
        self.action_command_set.set(true);
        self.get_button().set_action_command(action_command);
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.get_button().get_action_command()
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.get_button()
    }

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&self, visible: bool) {
        self.get_button().set_visible(visible);
    }

    /// Java `setToolTipText(String)`.
    pub fn set_tool_tip_text(&self, text: Option<&str>) {
        self.get_button()
            .set_tool_tip_text(super::tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `setTooltip(MultiLineButton)`.
    pub fn set_tooltip(&self, multi_line_button: &MultiLineButton) {
        self.get_button()
            .set_tool_tip_text(multi_line_button.get_button().get_tool_tip_text().as_deref());
    }

    /// Java `getQuotedLabel()`.
    pub fn get_quoted_label(&self) -> Option<String> {
        utilities::quote_label(self.unformatted_label.borrow().as_deref())
    }

    /// Java `getText()`.
    pub fn get_text(&self) -> Option<String> {
        if self.label1.borrow().is_none() {
            return Some(self.get_button().get_text());
        }
        self.unformatted_label.borrow().clone()
    }

    /// Java `removeActionListener(ActionListener)`.
    pub fn remove_action_listener(&self, action_listener: &ActionListener) {
        self.get_button().remove_action_listener(action_listener);
    }

    /// Java `getFontMetrics()`.
    pub fn get_font_metrics(&self) -> Option<FontMetrics> {
        self.font_metrics.get()
    }

    /// Java private `setSize(boolean)`.  Only `width` is state; the sizes set
    /// on the button are layout.
    fn set_size(&self, set_minimum: bool) {
        let mut font_metrics: Option<FontMetrics> = None;
        if UIParameters::need_button_font_metrics() {
            font_metrics = ui_utilities::get_font_metrics_abstract_button(&self.get_button());
        }
        let size = UIParameters::get_instance_font_metrics(font_metrics).get_button_dimension();
        self.width.set(size.width);
        // Swing layout: button.setPreferredSize(size); button.setMaximumSize(size);
        // and button.setMinimumSize(size) when setMinimum.
        let _ = set_minimum;
    }

    /// Java `setProcessDone(boolean)`.
    pub fn set_process_done(&self, done: bool) {
        self.set_selected(done);
    }

    /// Java `setScreenState(BaseScreenState)`.
    pub fn set_screen_state(&self, screen_state: &'static BaseScreenState) {
        self.screen_state.set(Some(screen_state));
        self.get_button().set_selected(
            screen_state.get_button_state(self.get_button_state_key().as_deref()),
        );
    }

    /// Java `setSelected(boolean)`.
    pub fn set_selected(&self, selected: bool) {
        self.get_button().set_selected(selected);
        if let Some(screen_state) = self.screen_state.get() {
            screen_state.set_button_state(
                self.get_button_state_key().as_deref(),
                self.get_button_state(),
            );
        }
    }

    /// Java `getOriginalState()`.
    pub fn get_original_state(&self) -> bool {
        !self.is_selected()
    }

    /// Java `isSelected()`.
    pub fn is_selected(&self) -> bool {
        self.get_button().is_selected()
    }

    /// Java `isVisible()`.
    pub fn is_visible(&self) -> bool {
        self.get_button().is_visible()
    }

    /// Java `init()`.
    pub fn init(&self) {
        // Swing layout: button.setMargin(new Insets(2, 2, 2, 2)).
    }

    /// Java `msgProcessStarting()`.
    pub fn msg_process_starting(&self) {
        self.process_result_display_state().msg_process_starting();
    }

    /// Java `msg(ProcessResult)`.
    pub fn msg_process_result(&self, process_result: ProcessResult) {
        self.process_result_display_state()
            .msg_process_result(Some(process_result));
    }

    /// Java `msg(ProcessEndState)`.
    pub fn msg_process_end_state(&self, end_state: ProcessEndState) {
        self.process_result_display_state()
            .msg_process_end_state(Some(end_state));
    }

    /// Java `msgProcessSucceeded()`.
    pub fn msg_process_succeeded(&self) {
        self.process_result_display_state().msg_process_succeeded();
    }

    /// Java `msgProcessFailed()`.
    pub fn msg_process_failed(&self) {
        self.process_result_display_state().msg_process_failed();
    }

    /// Java `msgProcessFailedToStart()`.
    pub fn msg_process_failed_to_start(&self) {
        self.process_result_display_state()
            .msg_process_failed_to_start();
    }

    /// Java `msgSecondaryProcess()`.
    pub fn msg_secondary_process(&self) {
        self.process_result_display_state().msg_secondary_process();
    }

    /// Java `addDependentDisplay(ProcessResultDisplay)`.
    pub fn add_dependent_display(&self, dependent_display: Option<ProcessResultDisplayHandle>) {
        self.process_result_display_state()
            .add_dependent_display(dependent_display);
    }

    /// Java `setOriginalState(boolean)`.
    pub fn set_original_state(&self, original_state: bool) {
        self.process_result_display_state()
            .set_original_state(original_state);
    }

    /// Java `addFailureDisplay(ProcessResultDisplay)`.
    pub fn add_failure_display(&self, failure_display: Option<ProcessResultDisplayHandle>) {
        self.process_result_display_state()
            .add_failure_display(failure_display);
    }

    /// Java `addSuccessDisplay(ProcessResultDisplay)`.
    pub fn add_success_display(&self, success_display: Option<ProcessResultDisplayHandle>) {
        self.process_result_display_state()
            .add_success_display(success_display);
    }
}

/// Java `SwingComponent.getComponent()`.
impl SwingComponent for MultiLineButton {
    fn get_component(&self) -> Rc<JComponent> {
        self.get_button()
    }
}

/// Java `UIComponent`.
impl UIComponent for MultiLineButton {
    /// Java `getUIComponent()`: `this`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.get_button()
    }
}

/// Every Java `MultiLineButton` is a `ProcessResultDisplay`, and every one of
/// those methods is `final` there, so a subclass inherits them unchanged.
impl<T: MultiLineButtonVirtual + 'static> ProcessResultDisplay for T {
    fn as_any_rc(self: Rc<Self>) -> Rc<dyn std::any::Any> {
        self
    }
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        self.get_multi_line_button().get_output_image_file_key()
    }
    fn set_next(&self, display: Option<ProcessResultDisplayHandle>) {
        self.get_multi_line_button().set_next(display)
    }
    fn set_use_global_dependency_list(&self, input: bool) {
        self.get_multi_line_button()
            .set_use_global_dependency_list(input)
    }
    fn get_next(&self) -> Option<ProcessResultDisplayHandle> {
        self.get_multi_line_button().get_next()
    }
    fn set_debug(&self, input: bool) {
        self.get_multi_line_button().set_debug(input)
    }
    fn dump_state(&self) {
        self.get_multi_line_button().dump_state()
    }
    fn get_original_state(&self) -> bool {
        self.get_multi_line_button().get_original_state()
    }
    fn set_original_state(&self, original_state: bool) {
        self.get_multi_line_button().set_original_state(original_state)
    }
    fn set_process_done(&self, done: bool) {
        self.get_multi_line_button().set_process_done(done)
    }
    fn set_screen_state(&self, screen_state: &'static BaseScreenState) {
        self.get_multi_line_button().set_screen_state(screen_state)
    }
    fn msg_process_result(&self, display_state: ProcessResult) {
        self.get_multi_line_button().msg_process_result(display_state)
    }
    fn msg_process_end_state(&self, end_state: ProcessEndState) {
        self.get_multi_line_button().msg_process_end_state(end_state)
    }
    fn msg_process_starting(&self) {
        self.get_multi_line_button().msg_process_starting()
    }
    fn msg_process_succeeded(&self) {
        self.get_multi_line_button().msg_process_succeeded()
    }
    fn msg_process_failed(&self) {
        self.get_multi_line_button().msg_process_failed()
    }
    fn msg_process_failed_to_start(&self) {
        self.get_multi_line_button().msg_process_failed_to_start()
    }
    fn msg_secondary_process(&self) {
        self.get_multi_line_button().msg_secondary_process()
    }
    fn add_dependent_display(&self, dependent_display: ProcessResultDisplayHandle) {
        self.get_multi_line_button()
            .add_dependent_display(Some(dependent_display))
    }
    fn add_failure_display(&self, failure_display: ProcessResultDisplayHandle) {
        self.get_multi_line_button()
            .add_failure_display(Some(failure_display))
    }
    fn add_success_display(&self, success_display: ProcessResultDisplayHandle) {
        self.get_multi_line_button()
            .add_success_display(Some(success_display))
    }
    fn equals_id(&self, display_id: i32, factory_id: &str) -> bool {
        self.get_multi_line_button()
            .equals_id(display_id, Some(factory_id))
    }
    fn set_id(&self, display_id: i32, factory_id: String) {
        self.get_multi_line_button()
            .set_id(display_id, Some(factory_id))
    }
    fn get_display_id(&self) -> i32 {
        self.get_multi_line_button().get_display_id()
    }
    fn get_factory_id(&self) -> Option<String> {
        self.get_multi_line_button().get_factory_id()
    }
    fn set_factory_id(&self, factory_id: String) {
        self.get_multi_line_button().set_factory_id(Some(factory_id))
    }
    fn get_button_state_key(&self) -> Option<String> {
        self.get_multi_line_button().get_button_state_key()
    }
}

/// Java private inner class `LabelDivider`.  Its outer-instance accesses
/// (`fontMetrics`, `button`, `width`) go through the `outer` argument.
struct LabelDivider {
    /// Java `wordArray`.
    word_array: Option<Vec<String>>,
    /// Java `line1`.
    line1: Option<String>,
    /// Java `line2`.
    line2: Option<String>,
    /// Java `debug`.  Unused in the Java class.
    debug: bool,
}

impl LabelDivider {
    /// Java `PADDING`.
    const PADDING: i32 = 9;

    /// Java private `LabelDivider()`.
    fn new() -> LabelDivider {
        LabelDivider {
            word_array: None,
            line1: None,
            line2: None,
            debug: false,
        }
    }

    /// Java `getLine1()`.
    fn get_line1(&self) -> Option<String> {
        self.line1.clone()
    }

    /// Java `getLine2()`.
    fn get_line2(&self) -> Option<String> {
        self.line2.clone()
    }

    /// Java `divide(String)`.
    fn divide(&mut self, outer: &MultiLineButton, label_text: &str) {
        self.line1 = Some(label_text.to_owned());
        self.line2 = None;
        if outer.font_metrics.get().is_none() {
            outer
                .font_metrics
                .set(ui_utilities::get_font_metrics_abstract_button(&outer.get_button()));
        }
        let Some(font_metrics) = outer.font_metrics.get() else {
            return;
        };
        let text_width = font_metrics.string_width(label_text);
        // Java `button.toString();` is a statement with no effect.
        let label_space = outer.width.get() - Self::PADDING;
        if text_width <= 0 || text_width < label_space {
            return;
        }

        let word_array = utilities::java_lang_string_split(label_text, &WHITESPACE_SPLIT);
        self.word_array = Some(word_array.clone());
        if word_array.is_empty() {
            return;
        }
        let mut label_line1 = LabelLine::new(&word_array, true, label_space);
        let mut label_line2 = LabelLine::new(&word_array, false, label_space);
        let mut error = false;
        while label_line1.iterator_le(&label_line2) {
            let potential_width1 = label_line1.try_next_word(&font_metrics);
            let overflow1 = label_line1.is_overflow_on_next_word(&font_metrics);
            let potential_width2 = label_line2.try_next_word(&font_metrics);
            let overflow2 = label_line2.is_overflow_on_next_word(&font_metrics);
            if overflow1 || overflow2 {
                if !overflow1 {
                    label_line1.build();
                } else if !overflow2 {
                    label_line2.build();
                } else {
                    error = true;
                    if potential_width1 <= potential_width2 {
                        label_line1.build();
                    } else {
                        label_line2.build();
                    }
                }
                continue;
            }
            if potential_width1 <= potential_width2 {
                label_line1.build();
            } else {
                label_line2.build();
            }
        }
        if error {
            eprintln!(
                "Error: The label for this two-line button is too long.\\nLabel text:{}",
                label_text
            );
        }
        self.line1 = Some(label_line1.get_line());
        self.line2 = Some(label_line2.get_line());
    }
}

/// Java private inner class `LabelLine`.  Its outer-instance access
/// (`fontMetrics`) is passed in.
struct LabelLine {
    /// Java final `iterator`.
    iterator: WordArrayIterator,
    /// Java final `firstHalf`.
    first_half: bool,
    /// Java final `labelSpace`.
    label_space: i32,
    /// Java `label`.
    label: String,
}

impl LabelLine {
    /// Java private `LabelLine(String[], boolean, int)`.
    fn new(word_array: &[String], first_half: bool, label_space: i32) -> LabelLine {
        let mut iterator = WordArrayIterator::new(Some(word_array.to_vec()), first_half);
        let label = iterator.next_void();
        LabelLine {
            iterator,
            first_half,
            label_space,
            label,
        }
    }

    /// Java `toString()`.
    fn to_string(&self) -> String {
        self.label.clone()
    }

    /// Java `iteratorLe(LabelLine)`.
    fn iterator_le(&self, label_line: &LabelLine) -> bool {
        self.iterator.le(&label_line.iterator)
    }

    /// Java `tryNextWord()`.  (`peek` can advance the iterator into a
    /// hyphenated word, hence `&mut`.)
    fn try_next_word(&mut self, font_metrics: &FontMetrics) -> i32 {
        let next = self.iterator.peek();
        font_metrics.string_width(&format!("{}{}", self.label, next))
    }

    /// Java `isOverflowOnNextWord()`.
    fn is_overflow_on_next_word(&mut self, font_metrics: &FontMetrics) -> bool {
        self.try_next_word(font_metrics) > self.label_space
    }

    /// Java `build()`.
    fn build(&mut self) {
        if self.iterator.has_next() {
            if self.first_half {
                self.label = format!("{}{}", self.label, self.iterator.next_void());
            } else {
                self.label = format!("{}{}", self.iterator.next_void(), self.label);
            }
        }
    }

    /// Java `getLine()`.
    fn get_line(&self) -> String {
        self.label.clone()
    }
}

/// Java private static nested class `WordArrayIterator`.
struct WordArrayIterator {
    /// Java final `wordArray`.
    word_array: Option<Vec<String>>,
    /// Java final `forwards`.
    forwards: bool,
    /// Java `spaceIndex`.
    space_index: i32,
    /// Java `hyphenIndex`.
    hyphen_index: i32,
    /// Java `curHyphenatedArray`.
    cur_hyphenated_array: Option<Vec<String>>,
}

impl WordArrayIterator {
    /// Java `SPACE`.
    const SPACE: &'static str = " ";
    /// Java `HYPHEN`.
    const HYPHEN: &'static str = "-";

    /// Java private `WordArrayIterator(String[], boolean)`.
    fn new(word_array: Option<Vec<String>>, forwards: bool) -> WordArrayIterator {
        let mut space_index = -1;
        if let Some(word_array) = &word_array {
            if !word_array.is_empty() {
                if forwards {
                    space_index = 0;
                } else {
                    space_index = word_array.len() as i32 - 1;
                }
            }
        }
        WordArrayIterator {
            word_array,
            forwards,
            space_index,
            hyphen_index: -1,
            cur_hyphenated_array: None,
        }
    }

    /// Java `hasNext()`.
    fn has_next(&self) -> bool {
        let Some(word_array) = &self.word_array else {
            return false;
        };
        if self.space_index >= 0 && self.space_index < word_array.len() as i32 {
            return true;
        }
        if let Some(cur_hyphenated_array) = &self.cur_hyphenated_array {
            if self.hyphen_index >= 0 && self.hyphen_index < cur_hyphenated_array.len() as i32 {
                return true;
            }
        }
        false
    }

    /// Java `peek()`.
    fn peek(&mut self) -> String {
        self.next_boolean(false)
    }

    /// Java `next()`.
    fn next_void(&mut self) -> String {
        self.next_boolean(true)
    }

    /// Java `next(boolean)`.
    fn next_boolean(&mut self, increment: bool) -> String {
        if !self.has_next() {
            return String::new();
        }
        let element = self.next_hyphenated_element(increment);
        if let Some(element) = element {
            return element;
        }
        let word_array = self.word_array.as_ref().unwrap();
        // Upstream bug fixed (MultiLineButton.java, WordArrayIterator.next):
        // Java indexes `wordArray[spaceIndex]` here even when hasNext() was
        // true only because of the hyphen array (an empty hyphen element, as
        // in "a--b"), which throws ArrayIndexOutOfBoundsException.  Nothing
        // is left in the word array then, so return "" as hasNext() false does.
        if self.space_index < 0 || self.space_index >= word_array.len() as i32 {
            return String::new();
        }
        let cur_word = word_array[self.space_index as usize].clone();
        if self.cur_hyphenated_array.is_none()
            && !cur_word.is_empty()
            && !cur_word.starts_with(Self::HYPHEN)
            && cur_word.contains(Self::HYPHEN)
        {
            let cur_hyphenated_array = utilities::java_lang_string_split(&cur_word, &HYPHEN_SPLIT);
            if !cur_hyphenated_array.is_empty() {
                if self.forwards {
                    self.hyphen_index = 0;
                } else {
                    self.hyphen_index = cur_hyphenated_array.len() as i32 - 1;
                }
                self.cur_hyphenated_array = Some(cur_hyphenated_array);
                let element = self.next_hyphenated_element(increment);
                if let Some(element) = element {
                    return element;
                }
                return String::new();
            }
            self.hyphen_index = -1;
            self.cur_hyphenated_array = None;
        }
        let element = self.add_split_char(cur_word);
        if increment {
            self.space_index = self.increment_index(self.space_index);
        }
        element
    }

    /// Java `incrementIndex(int)`.
    fn increment_index(&self, index: i32) -> i32 {
        if self.forwards {
            return index + 1;
        }
        index - 1
    }

    /// Java `nextHyphenatedElement(boolean)`.
    fn next_hyphenated_element(&mut self, increment: bool) -> Option<String> {
        let cur_hyphenated_array = self.cur_hyphenated_array.as_ref()?;
        if self.hyphen_index < 0 || self.hyphen_index >= cur_hyphenated_array.len() as i32 {
            if increment {
                self.cur_hyphenated_array = None;
                self.hyphen_index = -1;
            }
            return None;
        }
        let element = cur_hyphenated_array[self.hyphen_index as usize].clone();
        if element.is_empty() {
            return None;
        }
        let element = self.add_split_char(element);
        if increment {
            self.hyphen_index = self.increment_index(self.hyphen_index);
            // Upstream bug fixed (MultiLineButton.java, WordArrayIterator):
            // Java never advances `spaceIndex` past a hyphenated word.  Once
            // its last part is consumed, the next call re-splits the same word
            // and returns its parts again, so a label with a hyphenated word
            // repeats it ("a-b a-"), and with a hyphenated word on each side
            // of the label both lines cycle forever and `divide` never ends.
            // The evident intent is to move on to the next word, so the word
            // is finished here, as `next` does for an unhyphenated word.
            let len = self.cur_hyphenated_array.as_ref().map_or(0, Vec::len) as i32;
            if self.hyphen_index < 0 || self.hyphen_index >= len {
                self.cur_hyphenated_array = None;
                self.hyphen_index = -1;
                self.space_index = self.increment_index(self.space_index);
            }
        }
        Some(element)
    }

    /// Java `addSplitChar(String)`.
    fn add_split_char(&self, element: String) -> String {
        if element.is_empty() {
            return element;
        }
        let word_array_len = self.word_array.as_ref().map_or(0, Vec::len) as i32;
        if self.forwards {
            match &self.cur_hyphenated_array {
                None => {
                    if self.space_index > 0 {
                        return format!("{}{}", Self::SPACE, element);
                    }
                }
                Some(cur_hyphenated_array) => {
                    if self.hyphen_index == 0 {
                        return format!("{}{}{}", Self::SPACE, element, Self::HYPHEN);
                    } else if self.hyphen_index < cur_hyphenated_array.len() as i32 - 1 {
                        return format!("{}{}", element, Self::HYPHEN);
                    }
                }
            }
        } else {
            match &self.cur_hyphenated_array {
                None => {
                    if self.space_index < word_array_len - 1 {
                        return format!("{}{}", element, Self::SPACE);
                    }
                }
                Some(cur_hyphenated_array) => {
                    if self.hyphen_index == cur_hyphenated_array.len() as i32 - 1 {
                        return format!("{}{}", element, Self::SPACE);
                    } else {
                        return format!("{}{}", element, Self::HYPHEN);
                    }
                }
            }
        }
        element
    }

    /// Java `le(WordArrayIterator)`.
    fn le(&self, iterator: &WordArrayIterator) -> bool {
        if self.space_index == -1
            || iterator.space_index == -1
            || self.space_index > iterator.space_index
        {
            return false;
        }
        if self.space_index < iterator.space_index {
            return true;
        }
        // Upstream bug fixed (MultiLineButton.java, WordArrayIterator.le):
        // Java returns `hyphenIndex <= iterator.hyphenIndex` when only this
        // iterator has entered the shared hyphenated word, and the other's
        // -1 ("not entered yet") makes that false, so the rest of the word
        // is dropped from both lines ("a-b c" divides into " a-" and "c").
        // Either iterator not having entered the word means parts of it are
        // still unconsumed.
        if self.hyphen_index == -1 || iterator.hyphen_index == -1 {
            return true;
        }
        self.hyphen_index <= iterator.hyphen_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyphenated_words_are_consumed_once_and_the_iterators_meet() {
        let words = utilities::java_lang_string_split("Pre-align Cross-correlate", &WHITESPACE_SPLIT);
        let mut forwards = WordArrayIterator::new(Some(words.clone()), true);
        let mut parts = Vec::new();
        while forwards.has_next() {
            parts.push(forwards.next_void());
        }
        assert_eq!(parts, vec![" Pre-", "align", " Cross-", "correlate"]);
        let mut backwards = WordArrayIterator::new(Some(words), false);
        let mut parts = Vec::new();
        while backwards.has_next() {
            parts.push(backwards.next_void());
        }
        assert_eq!(parts, vec!["correlate ", "Cross-", "align ", "Pre-"]);
    }
}
