//! `IMOD/Etomo/src/etomo/ui/swing/PanelHeader.java`.
//!
//! The native `JPanel`, `GridBagLayout`, `BoxLayout`, `HeaderCell`, and
//! `ExpandButton` widgets remain direct Swing presentation boundaries.  This
//! unit retains the Java header's construction choices, expander state, EDT
//! names, and screen-state key behaviour instead of replacing them with a
//! different panel abstraction.
#![allow(dead_code)]

use super::abstract_frame::ComponentState;
pub use super::expand_button::{ExpandButton, ExpandButtonType};
pub use super::expandable::Expandable;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::SEPARATOR_CHAR;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::util::utilities;
use std::collections::BTreeMap;

/// Java `ConstPanelHeaderSettings`.
pub trait ConstPanelHeaderSettings {
    fn is_advanced(&self) -> bool;
    fn is_more(&self) -> bool;
    fn is_open(&self) -> bool;
    fn is_advanced_null(&self) -> bool;
    fn is_open_null(&self) -> bool;
    fn is_more_null(&self) -> bool;
}

/// Java `ConstPanelHeaderState` value read by `PanelHeader.setState`.
pub trait ConstPanelHeaderState {
    fn get_open_close_state(&self) -> Option<&str>;
    fn get_advanced_basic_state(&self) -> Option<&str>;
    fn get_more_less_state(&self) -> Option<&str>;
}

/// Java `PanelHeaderState` fields used by this unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PanelHeaderState {
    pub open_close_state: Option<String>,
    pub advanced_basic_state: Option<String>,
    pub more_less_state: Option<String>,
}

impl PanelHeaderState {
    pub fn set_open_close_state(&mut self, state: String) {
        self.open_close_state = Some(state);
    }
    pub fn set_advanced_basic_state(&mut self, state: String) {
        self.advanced_basic_state = Some(state);
    }
    pub fn set_more_less_state(&mut self, state: String) {
        self.more_less_state = Some(state);
    }
}

impl ConstPanelHeaderState for PanelHeaderState {
    fn get_open_close_state(&self) -> Option<&str> {
        self.open_close_state.as_deref()
    }
    fn get_advanced_basic_state(&self) -> Option<&str> {
        self.advanced_basic_state.as_deref()
    }
    fn get_more_less_state(&self) -> Option<&str> {
        self.more_less_state.as_deref()
    }
}

/// Java `BaseScreenState` calls made by `PanelHeader`.
pub trait BaseScreenState {
    fn get_button_state(&mut self, key: &str, default_state: bool) -> bool;
    fn set_button_state(&mut self, key: &str, state: bool);
}

/// In-memory source-shaped screen-state implementation for native GUI tests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PanelHeaderScreenState {
    pub button_states: BTreeMap<String, bool>,
}

impl BaseScreenState for PanelHeaderScreenState {
    fn get_button_state(&mut self, key: &str, default_state: bool) -> bool {
        self.button_states
            .get(key)
            .copied()
            .unwrap_or(default_state)
    }
    fn set_button_state(&mut self, key: &str, state: bool) {
        self.button_states.insert(key.to_string(), state);
    }
}

/// Java `PanelHeader`, including its source-visible layout and expander state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PanelHeader {
    pub root_panel: ComponentState,
    pub cell_title: String,
    pub dialog_type: DialogType,
    pub title: String,
    pub btn_open_close: Option<ExpandButton>,
    pub btn_advanced_basic: Option<ExpandButton>,
    pub btn_more_less: Option<ExpandButton>,
    pub empty_when_basic: bool,
    pub global_advanced_button_present: bool,
    pub titled: bool,
    pub save_more_less_state: bool,
    pub separator: bool,
    pub north_rigid_area: bool,
    pub panel_name: Option<String>,
}

impl PanelHeader {
    /// Java `getInstance(String, Expandable, DialogType)`.
    pub fn get_instance(
        title: impl Into<String>,
        _expandable: &mut dyn Expandable,
        dialog_type: DialogType,
    ) -> Self {
        Self::new(
            title,
            false,
            false,
            dialog_type,
            true,
            false,
            true,
            false,
            true,
        )
    }

    /// Java `getTitleOnlyInstance(String, Expandable, DialogType)`.
    pub fn get_title_only_instance(
        title: impl Into<String>,
        _expandable: &mut dyn Expandable,
        dialog_type: DialogType,
    ) -> Self {
        Self::new(
            title,
            false,
            false,
            dialog_type,
            false,
            false,
            true,
            false,
            true,
        )
    }

    /// Java package-private `getUntitledInstance`.
    pub fn get_untitled_instance(
        title: impl Into<String>,
        _expandable: &mut dyn Expandable,
        dialog_type: DialogType,
    ) -> Self {
        Self::new(
            title,
            false,
            false,
            dialog_type,
            true,
            false,
            false,
            false,
            true,
        )
    }

    /// Java `getAdvancedBasicInstance(String, Expandable, DialogType)`.
    pub fn get_advanced_basic_instance(
        title: impl Into<String>,
        _expandable: &mut dyn Expandable,
        dialog_type: DialogType,
    ) -> Self {
        Self::new(
            title,
            true,
            false,
            dialog_type,
            true,
            false,
            true,
            false,
            true,
        )
    }

    /// Java package-private `getUntitledAdvancedBasicInstance`.
    pub fn get_untitled_advanced_basic_instance(
        title: impl Into<String>,
        _expandable: &mut dyn Expandable,
        dialog_type: DialogType,
    ) -> Self {
        Self::new(
            title,
            true,
            false,
            dialog_type,
            true,
            false,
            false,
            false,
            true,
        )
    }

    /// Java overloaded `getAdvancedBasicInstance` with `GlobalExpandButton`.
    pub fn get_advanced_basic_global_instance(
        title: impl Into<String>,
        _expandable: &mut dyn Expandable,
        dialog_type: DialogType,
    ) -> Self {
        Self::new(
            title,
            true,
            false,
            dialog_type,
            true,
            true,
            true,
            false,
            true,
        )
    }

    /// Java package-private `getAdvancedBasicOnlyInstance`.
    pub fn get_advanced_basic_only_instance(
        title: impl Into<String>,
        _expandable: &mut dyn Expandable,
        dialog_type: DialogType,
        empty_when_basic: bool,
    ) -> Self {
        Self::new(
            title,
            true,
            false,
            dialog_type,
            false,
            true,
            true,
            empty_when_basic,
            true,
        )
    }

    /// Java package-private `getAdvancedBasicOnlyNoSeparatorInstance`.
    pub fn get_advanced_basic_only_no_separator_instance(
        title: impl Into<String>,
        _expandable: &mut dyn Expandable,
        dialog_type: DialogType,
        empty_when_basic: bool,
    ) -> Self {
        Self::new(
            title,
            true,
            false,
            dialog_type,
            false,
            true,
            true,
            empty_when_basic,
            false,
        )
    }

    /// Java package-private `getMoreLessInstance`.
    pub fn get_more_less_instance(
        title: impl Into<String>,
        _expandable: &mut dyn Expandable,
        dialog_type: DialogType,
    ) -> Self {
        Self::new(
            title,
            false,
            true,
            dialog_type,
            true,
            false,
            true,
            false,
            true,
        )
    }

    /// Java private constructor.  Expandable and Swing widget ownership are
    /// explicit native-GUI boundaries; every Java construction flag is kept.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        title: impl Into<String>,
        advanced_basic: bool,
        more_less: bool,
        dialog_type: DialogType,
        open_close: bool,
        global_advanced_button_present: bool,
        titled: bool,
        empty_when_basic: bool,
        use_separator: bool,
    ) -> Self {
        let title = title.into();
        let mut btn_open_close =
            open_close.then(|| ExpandButton::new(ExpandButtonType::Open, true, false));
        let mut btn_advanced_basic = advanced_basic.then(|| {
            ExpandButton::new(
                ExpandButtonType::Advanced,
                false,
                global_advanced_button_present,
            )
        });
        let mut btn_more_less =
            more_less.then(|| ExpandButton::new(ExpandButtonType::More, true, false));
        if let Some(button) = &mut btn_open_close {
            button.set_name(&title);
        }
        if let Some(button) = &mut btn_advanced_basic {
            button.set_name(&title);
        }
        if let Some(button) = &mut btn_more_less {
            button.set_name(&title);
        }
        Self {
            root_panel: ComponentState::default(),
            cell_title: if titled { title.clone() } else { String::new() },
            dialog_type,
            title,
            btn_open_close,
            btn_advanced_basic,
            btn_more_less,
            empty_when_basic,
            global_advanced_button_present,
            titled,
            save_more_less_state: true,
            separator: use_separator && titled,
            north_rigid_area: !advanced_basic && !more_less,
            panel_name: None,
        }
    }

    /// Java `setText(String)`.
    pub fn set_text(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.cell_title = text.clone();
        self.set_name_from_text(&text);
    }

    /// Java overloaded `setText(String, String)`.
    pub fn set_text_with_additional(
        &mut self,
        text: impl Into<String>,
        additional_text: impl AsRef<str>,
    ) {
        let text = text.into();
        self.cell_title = format!("{} {}", text, additional_text.as_ref());
        self.set_name_from_text(&text);
    }

    /// Java private `setNameFromText(String)`.
    pub fn set_name_from_text(&mut self, text: &str) {
        self.set_panel_name(text);
        if let Some(button) = &mut self.btn_open_close {
            button.set_name(text);
        }
        if let Some(button) = &mut self.btn_advanced_basic {
            button.set_name(text);
        }
        if let Some(button) = &mut self.btn_more_less {
            button.set_name(text);
        }
    }

    /// Java `setPanelName(String)`.
    pub fn set_panel_name(&mut self, associated_label: &str) {
        self.panel_name = Some(format!(
            "pnl{SEPARATOR_CHAR}{}",
            utilities::convert_label_to_name(Some(associated_label), true).unwrap_or_default()
        ));
    }

    /// Java `getComponent()` / `getContainer()` at the JPanel boundary.
    pub fn get_container(&self) -> &ComponentState {
        &self.root_panel
    }
    /// Java `getComponent()`.
    pub fn get_component(&self) -> &ComponentState {
        &self.root_panel
    }
    /// Java `equalsOpenClose(ExpandButton)` identity comparison.
    pub fn equals_open_close(&self, button: &ExpandButton) -> bool {
        self.btn_open_close.as_ref().is_some_and(|owner_button| {
            owner_button.button_type == button.button_type && owner_button.name == button.name
        })
    }
    /// Java `equalsAdvancedBasic(ExpandButton)` identity comparison.
    pub fn equals_advanced_basic(&self, button: &ExpandButton) -> bool {
        self.btn_advanced_basic
            .as_ref()
            .is_some_and(|owner_button| {
                owner_button.button_type == button.button_type && owner_button.name == button.name
            })
    }
    /// Java `equalsMoreLess(ExpandButton)` identity comparison.
    pub fn equals_more_less(&self, button: &ExpandButton) -> bool {
        self.btn_more_less.as_ref().is_some_and(|owner_button| {
            owner_button.button_type == button.button_type && owner_button.name == button.name
        })
    }
    /// Java `setAdvanced(boolean)`.
    pub fn set_advanced(&mut self, advanced: bool) {
        if let Some(button) = &mut self.btn_advanced_basic {
            button.set_expanded(advanced);
        }
    }
    /// Java `setOpen(boolean)`.
    pub fn set_open(&mut self, open: bool) {
        if let Some(button) = &mut self.btn_open_close {
            button.set_expanded(open);
        }
    }
    /// Java `getTitle()`.
    pub fn get_title(&self) -> &str {
        &self.title
    }
    /// Java `getOpenCloseButton()`.
    pub fn get_open_close_button(&self) -> Option<&ExpandButton> {
        self.btn_open_close.as_ref()
    }
    /// Java `getMoreLessButton()`.
    pub fn get_more_less_button(&self) -> Option<&ExpandButton> {
        self.btn_more_less.as_ref()
    }
    /// Java `isAdvanced()`.
    pub fn is_advanced(&self) -> bool {
        self.btn_advanced_basic
            .as_ref()
            .is_some_and(ExpandButton::is_expanded)
    }
    /// Java `isLess()`.
    pub fn is_less(&self) -> bool {
        self.btn_more_less
            .as_ref()
            .is_some_and(|button| !button.is_expanded())
    }
    /// Java `isMore()`.
    pub fn is_more(&self) -> bool {
        self.btn_more_less
            .as_ref()
            .is_some_and(ExpandButton::is_expanded)
    }
    /// Java `isOpen()`.
    pub fn is_open(&self) -> bool {
        self.btn_open_close
            .as_ref()
            .is_some_and(ExpandButton::is_expanded)
    }
    /// Java `isAdvancedNull()`.
    pub fn is_advanced_null(&self) -> bool {
        self.btn_advanced_basic.is_none()
    }
    /// Java `isMoreNull()`.
    pub fn is_more_null(&self) -> bool {
        self.btn_more_less.is_none()
    }
    /// Java `isOpenNull()`.
    pub fn is_open_null(&self) -> bool {
        self.btn_open_close.is_none()
    }
    /// Java `expand(GlobalExpandButton)`, intentionally empty.
    pub fn expand_global(&mut self) {}
    /// Java `expand(ExpandButton)`: this requests a Swing `UIHarness.pack` if
    /// its source identity condition matches.  Packing remains a direct GUI boundary.
    pub fn expand(&mut self, button: &ExpandButton) -> bool {
        self.titled
            && (self.equals_open_close(button)
                || (self.equals_advanced_basic(button) && self.empty_when_basic))
    }

    /// Java `getState(PanelHeaderState)`.
    pub fn get_state(&self, state: Option<&mut PanelHeaderState>) {
        let Some(state) = state else { return };
        if let Some(button) = &self.btn_open_close {
            state.set_open_close_state(button.get_state());
        }
        if let Some(button) = &self.btn_advanced_basic {
            state.set_advanced_basic_state(button.get_state());
        }
        if self.save_more_less_state {
            if let Some(button) = &self.btn_more_less {
                state.set_more_less_state(button.get_state());
            }
        }
    }

    /// Java `set(ConstPanelHeaderSettings)`.
    pub fn set(&mut self, settings: Option<&dyn ConstPanelHeaderSettings>) {
        let Some(settings) = settings else { return };
        if let Some(button) = &mut self.btn_open_close {
            button.set_expanded(settings.is_open());
        }
        if let Some(button) = &mut self.btn_advanced_basic {
            button.set_expanded(settings.is_advanced());
        }
        if let Some(button) = &mut self.btn_more_less {
            button.set_expanded(settings.is_more());
        }
    }
    /// Java `updateOpenCloseButton(ConstPanelHeaderSettings)`.
    pub fn update_open_close_button(&mut self, settings: Option<&dyn ConstPanelHeaderSettings>) {
        if let (Some(settings), Some(button)) = (settings, &mut self.btn_open_close) {
            button.update(settings.is_open());
        }
    }
    /// Java `setState(ConstPanelHeaderState)`.
    pub fn set_state(&mut self, state: Option<&dyn ConstPanelHeaderState>) {
        let Some(state) = state else { return };
        if let Some(button) = &mut self.btn_open_close {
            button.set_state(state.get_open_close_state());
        }
        if let Some(button) = &mut self.btn_advanced_basic {
            button.set_state(state.get_advanced_basic_state());
        }
        if self.save_more_less_state {
            if let Some(button) = &mut self.btn_more_less {
                button.set_state(state.get_more_less_state());
            }
        }
    }
    /// Java `setSaveMoreLessState(boolean)`.
    pub fn set_save_more_less_state(&mut self, input: bool) {
        self.save_more_less_state = input;
    }
    /// Java overloaded `setButtonStates(BaseScreenState)`.
    pub fn set_button_states(&mut self, screen_state: Option<&mut dyn BaseScreenState>) {
        self.set_button_states_with_default(screen_state, true);
    }
    /// Java `createButtonStateKeys()`.
    pub fn create_button_state_keys(&mut self) {
        if let Some(button) = &mut self.btn_open_close {
            button.create_button_state_key(self.dialog_type);
        }
        if let Some(button) = &mut self.btn_advanced_basic {
            button.create_button_state_key(self.dialog_type);
        }
        if let Some(button) = &mut self.btn_more_less {
            button.create_button_state_key(self.dialog_type);
        }
    }
    /// Java overloaded `setButtonStates(BaseScreenState, boolean)`.
    pub fn set_button_states_with_default(
        &mut self,
        screen_state: Option<&mut dyn BaseScreenState>,
        default_is_open: bool,
    ) {
        let Some(screen_state) = screen_state else {
            return;
        };
        if let Some(button) = &mut self.btn_open_close {
            let key = button.create_button_state_key(self.dialog_type);
            button.set_expanded(screen_state.get_button_state(&key, default_is_open));
        }
        if let Some(button) = &mut self.btn_advanced_basic {
            let key = button.create_button_state_key(self.dialog_type);
            button.set_expanded(screen_state.get_button_state(&key, false));
        }
        if self.save_more_less_state {
            if let Some(button) = &mut self.btn_more_less {
                let key = button.create_button_state_key(self.dialog_type);
                button.set_expanded(screen_state.get_button_state(&key, false));
            }
        }
    }
    /// Java `getButtonStates(BaseScreenState)`.
    pub fn get_button_states(&mut self, screen_state: Option<&mut dyn BaseScreenState>) {
        let Some(screen_state) = screen_state else {
            return;
        };
        if let Some(button) = &self.btn_open_close {
            if let Some(key) = &button.state_key {
                screen_state.set_button_state(key, button.expanded);
            }
        }
        if let Some(button) = &self.btn_advanced_basic {
            if let Some(key) = &button.state_key {
                screen_state.set_button_state(key, button.expanded);
            }
        }
        if self.save_more_less_state {
            if let Some(button) = &self.btn_more_less {
                if let Some(key) = &button.state_key {
                    screen_state.set_button_state(key, button.expanded);
                }
            }
        }
    }
    /// Java `getAdvancedFieldDisplayer()`: the returned object has an explicit
    /// caller-owned header because Swing listeners cannot borrow it indefinitely.
    pub fn get_advanced_field_displayer(&mut self) -> AdvancedFieldDisplayer {
        AdvancedFieldDisplayer
    }
}

impl ConstPanelHeaderSettings for PanelHeader {
    fn is_advanced(&self) -> bool {
        self.is_advanced()
    }
    fn is_more(&self) -> bool {
        self.is_more()
    }
    fn is_open(&self) -> bool {
        self.is_open()
    }
    fn is_advanced_null(&self) -> bool {
        self.is_advanced_null()
    }
    fn is_open_null(&self) -> bool {
        self.is_open_null()
    }
    fn is_more_null(&self) -> bool {
        self.is_more_null()
    }
}

/// Java private `AdvancedFieldDisplayer`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdvancedFieldDisplayer;

impl AdvancedFieldDisplayer {
    /// Java `display()`.
    pub fn display(&self, panel_header: &mut PanelHeader) {
        if !panel_header.is_advanced() {
            panel_header.set_advanced(true);
        }
    }
    /// Java overloaded `display(UIComponent)`.
    pub fn display_ui_component(&self, panel_header: &mut PanelHeader) {
        self.display(panel_header);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct TestExpandable;
    impl Expandable for TestExpandable {
        fn expand_expand_button(&mut self, _: &ExpandButton) {}
        fn expand_global_button(
            &mut self,
            _: &crate::imod::etomo::ui::swing::process_dialog::GlobalExpandButton,
        ) {
        }
    }

    #[test]
    fn source_construction_defaults_and_uitest_names_are_retained() {
        let mut owner = TestExpandable;
        let header = PanelHeader::get_advanced_basic_instance(
            "Align: options",
            &mut owner,
            DialogType::Tools,
        );
        assert!(header.is_open());
        assert!(!header.is_advanced());
        assert_eq!(header.btn_open_close.as_ref().unwrap().name, "mb.align");
        assert!(header.separator);
    }

    #[test]
    fn state_round_trip_preserves_each_present_button() {
        let mut owner = TestExpandable;
        let mut header =
            PanelHeader::get_more_less_instance("Details", &mut owner, DialogType::Tools);
        header.set_open(false);
        let mut state = PanelHeaderState::default();
        header.get_state(Some(&mut state));
        assert_eq!(state.open_close_state.as_deref(), Some("closed"));
        assert_eq!(state.more_less_state.as_deref(), Some("more"));
        header.set_open(true);
        header.set_state(Some(&state));
        assert!(!header.is_open());
    }

    #[test]
    fn screen_state_uses_dialog_button_key_and_default_open() {
        let mut owner = TestExpandable;
        let mut header = PanelHeader::get_instance("Panel", &mut owner, DialogType::Tools);
        let mut screen = PanelHeaderScreenState::default();
        header.set_button_states(Some(&mut screen));
        assert!(header.is_open());
        header.get_button_states(Some(&mut screen));
        assert!(screen.button_states.values().any(|state| *state));
    }
}
