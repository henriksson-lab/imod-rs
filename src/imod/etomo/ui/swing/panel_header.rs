//! `IMOD/Etomo/src/etomo/ui/swing/PanelHeader.java`.
//!
//! Java `final class PanelHeader implements Expandable, ConstPanelHeaderSettings`:
//! the title bar of a collapsible panel, with open/close, advanced/basic and
//! more/less mini buttons.  GridBag/Box layout is Swing layout and is recorded as
//! comments; the component tree (root panel, north panel, buttons, title cell) is
//! kept, since the uitest driver finds the buttons by name under it.

use std::cell::{Cell as StdCell, RefCell};
use std::rc::{Rc, Weak};

use super::cell::CellVirtual;
use super::expand_button::{self, ExpandButton};
use super::expandable::Expandable;
use super::global_expand_button::GlobalExpandButton;
use super::header_cell::HeaderCell;
use super::ui_harness;
use crate::imod::etomo::etomo_director::ARGUMENTS;
use crate::imod::etomo::jdk::JComponent;
use crate::imod::etomo::storage::autodoc::autodoc_tokenizer::{DEFAULT_DELIMITER, SEPARATOR_CHAR};
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::const_panel_header_settings::ConstPanelHeaderSettings;
use crate::imod::etomo::r#type::const_panel_header_state::ConstPanelHeaderState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::panel_header_state::PanelHeaderState;
use crate::imod::etomo::r#type::ui_test_field_type::UITestFieldType;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::utilities;

/// Java `PanelHeader`.
pub struct PanelHeader {
    /// Java `rootPanel`.
    root_panel: Rc<JComponent>,
    /// Java `cellTitle`.
    cell_title: Rc<HeaderCell>,
    /// Java `dialogType`.
    dialog_type: Option<DialogType>,
    /// Java `title`.
    title: Option<String>,
    /// Java `btnOpenClose`.
    btn_open_close: Option<Rc<ExpandButton>>,
    /// Java `btnAdvancedBasic`.
    btn_advanced_basic: Option<Rc<ExpandButton>>,
    /// Java `btnMoreLess`.
    btn_more_less: Option<Rc<ExpandButton>>,
    /// Java `emptyWhenBasic`.
    empty_when_basic: bool,
    /// Java `globalAdvancedButton`.
    global_advanced_button: Option<Rc<GlobalExpandButton>>,
    /// Java `titled`.
    titled: bool,
    /// Java `saveMoreLessState`.
    save_more_less_state: StdCell<bool>,
    /// Java `advancedFieldDisplayer`.
    advanced_field_displayer: RefCell<Option<Rc<AdvancedFieldDisplayer>>>,
    /// `this`, for the inner class and for handing the header to its buttons as
    /// an `Expandable`.
    this: Weak<PanelHeader>,
}

impl PanelHeader {
    /// Java `getInstance(String, Expandable, DialogType)`.
    pub fn get_instance(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        dialog_type: Option<DialogType>,
    ) -> Rc<PanelHeader> {
        Self::new(
            title,
            expandable,
            false,
            false,
            dialog_type,
            true,
            None,
            true,
            false,
            true,
        )
    }

    /// Java `getTitleOnlyInstance(String, Expandable, DialogType)`.
    pub fn get_title_only_instance(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        dialog_type: Option<DialogType>,
    ) -> Rc<PanelHeader> {
        Self::new(
            title,
            expandable,
            false,
            false,
            dialog_type,
            false,
            None,
            true,
            false,
            true,
        )
    }

    /// Java `getUntitledInstance(String, Expandable, DialogType)`.
    pub fn get_untitled_instance(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        dialog_type: Option<DialogType>,
    ) -> Rc<PanelHeader> {
        Self::new(
            title,
            expandable,
            false,
            false,
            dialog_type,
            true,
            None,
            false,
            false,
            true,
        )
    }

    /// Java `getAdvancedBasicInstance(String, Expandable, DialogType)`.
    pub fn get_advanced_basic_instance_string_expandable_dialog_type(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        dialog_type: Option<DialogType>,
    ) -> Rc<PanelHeader> {
        Self::new(
            title,
            expandable,
            true,
            false,
            dialog_type,
            true,
            None,
            true,
            false,
            true,
        )
    }

    /// Java `getUntitledAdvancedBasicInstance(String, Expandable, DialogType)`.
    pub fn get_untitled_advanced_basic_instance(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        dialog_type: Option<DialogType>,
    ) -> Rc<PanelHeader> {
        Self::new(
            title,
            expandable,
            true,
            false,
            dialog_type,
            true,
            None,
            false,
            false,
            true,
        )
    }

    /// Java `getAdvancedBasicInstance(String, Expandable, DialogType,
    /// GlobalExpandButton)`.
    pub fn get_advanced_basic_instance_string_expandable_dialog_type_global_expand_button(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        dialog_type: Option<DialogType>,
        global_advanced_button: Option<Rc<GlobalExpandButton>>,
    ) -> Rc<PanelHeader> {
        Self::new(
            title,
            expandable,
            true,
            false,
            dialog_type,
            true,
            global_advanced_button,
            true,
            false,
            true,
        )
    }

    /// Java `getAdvancedBasicOnlyInstance(String, Expandable, DialogType,
    /// GlobalExpandButton, boolean)`.
    pub fn get_advanced_basic_only_instance(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        dialog_type: Option<DialogType>,
        global_advanced_button: Option<Rc<GlobalExpandButton>>,
        empty_when_basic: bool,
    ) -> Rc<PanelHeader> {
        Self::new(
            title,
            expandable,
            true,
            false,
            dialog_type,
            false,
            global_advanced_button,
            true,
            empty_when_basic,
            true,
        )
    }

    /// Java `getAdvancedBasicOnlyNoSeparatorInstance(String, Expandable,
    /// DialogType, GlobalExpandButton, boolean)`.
    pub fn get_advanced_basic_only_no_separator_instance(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        dialog_type: Option<DialogType>,
        global_advanced_button: Option<Rc<GlobalExpandButton>>,
        empty_when_basic: bool,
    ) -> Rc<PanelHeader> {
        Self::new(
            title,
            expandable,
            true,
            false,
            dialog_type,
            false,
            global_advanced_button,
            true,
            empty_when_basic,
            false,
        )
    }

    /// Java `getMoreLessInstance(String, Expandable, DialogType)`.
    pub fn get_more_less_instance(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        dialog_type: Option<DialogType>,
    ) -> Rc<PanelHeader> {
        Self::new(
            title,
            expandable,
            false,
            true,
            dialog_type,
            true,
            None,
            true,
            false,
            true,
        )
    }

    /// Java private `PanelHeader(String, Expandable, boolean, boolean, DialogType,
    /// boolean, GlobalExpandButton, boolean, boolean, boolean)`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        title: Option<&str>,
        expandable: Option<Weak<dyn Expandable>>,
        advanced_basic: bool,
        more_less: bool,
        dialog_type: Option<DialogType>,
        open_close: bool,
        global_advanced_button: Option<Rc<GlobalExpandButton>>,
        titled: bool,
        empty_when_basic: bool,
        use_separator: bool,
    ) -> Rc<PanelHeader> {
        Rc::new_cyclic(|this: &Weak<PanelHeader>| {
            let this_expandable: Weak<dyn Expandable> = this.clone();
            // panels
            let root_panel = JComponent::new_panel();
            // Swing layout: rootPanel BoxLayout Y_AXIS; northPanel GridBagLayout;
            // constraints fill BOTH, anchor CENTER, weights 0, grid 1x1.
            let north_panel = JComponent::new_panel();
            let btn_open_close = if open_close {
                // open/close button - default: open
                let button = ExpandButton::get_expanded_instance(
                    expandable.clone(),
                    Some(this_expandable.clone()),
                    Some(&expand_button::Type::OPEN),
                );
                button.set_name(title);
                north_panel.add(&button.get_component());
                Some(button)
            } else {
                None
            };
            // title
            // Swing layout: weightx/weighty 1.0.
            let cell_title = if titled {
                HeaderCell::new_string_boolean(title, false)
            } else {
                HeaderCell::new_string_boolean(Some(""), false)
            };
            cell_title.set_border_painted(false);
            CellVirtual::add(&*cell_title, &north_panel);
            // advanced/basic button - default: basic
            let btn_advanced_basic = if advanced_basic {
                // Swing layout: weights 0; gridwidth REMAINDER unless moreLess.
                let button = if !empty_when_basic {
                    ExpandButton::get_global_instance_expandable_type_global_expand_button(
                        expandable.clone(),
                        Some(&expand_button::Type::ADVANCED),
                        global_advanced_button.clone(),
                    )
                } else {
                    ExpandButton::get_global_instance_expandable_expandable_type_global_expand_button(
                        expandable.clone(),
                        Some(this_expandable.clone()),
                        Some(&expand_button::Type::ADVANCED),
                        global_advanced_button.clone(),
                    )
                };
                button.set_name(title);
                north_panel.add(&button.get_component());
                Some(button)
            } else {
                None
            };
            // more/less button - default: more
            let btn_more_less = if more_less {
                // Swing layout: weights 0.
                let button = ExpandButton::get_expanded_instance(
                    expandable.clone(),
                    None,
                    Some(&expand_button::Type::MORE),
                );
                button.set_name(title);
                north_panel.add(&button.get_component());
                Some(button)
            } else {
                None
            };
            if !advanced_basic && !more_less {
                // Swing layout: `northPanel.add(Box.createRigidArea(23, 0))`.
            }
            // rootPanel
            root_panel.add(&north_panel);
            if use_separator && titled {
                // Swing painting: a line border in `Colors.getBackgroundA()`.
            }
            PanelHeader {
                root_panel,
                cell_title,
                dialog_type,
                title: title.map(str::to_owned),
                btn_open_close,
                btn_advanced_basic,
                btn_more_less,
                empty_when_basic,
                global_advanced_button,
                titled,
                save_more_less_state: StdCell::new(true),
                advanced_field_displayer: RefCell::new(None),
                this: this.clone(),
            }
        })
    }

    /// Java `getAdvancedFieldDisplayer()`.
    pub fn get_advanced_field_displayer(&self) -> Rc<dyn FieldDisplayer> {
        if let Some(displayer) = self.advanced_field_displayer.borrow().as_ref() {
            return displayer.clone();
        }
        let displayer = Rc::new(AdvancedFieldDisplayer {
            header: self.this.clone(),
        });
        *self.advanced_field_displayer.borrow_mut() = Some(displayer.clone());
        displayer
    }

    /// Java `setText(String)`.
    pub fn set_text_string(&self, text: Option<&str>) {
        self.cell_title.set_text_string(text);
        self.set_name_from_text(text);
    }

    /// Java `setText(String, String)`.
    pub fn set_text_string_string(&self, text: Option<&str>, additional_text: Option<&str>) {
        self.cell_title.set_text_string(Some(&format!(
            "{} {}",
            text.unwrap_or("null"),
            additional_text.unwrap_or("null")
        )));
        self.set_name_from_text(text);
    }

    /// Java private `setNameFromText(String)`.
    fn set_name_from_text(&self, text: Option<&str>) {
        self.set_panel_name(text);
        if let Some(button) = &self.btn_open_close {
            button.set_name(text);
        }
        if let Some(button) = &self.btn_advanced_basic {
            button.set_name(text);
        }
        if let Some(button) = &self.btn_more_less {
            button.set_name(text);
        }
    }

    /// Java `setPanelName(String)`.
    pub fn set_panel_name(&self, associated_label: Option<&str>) {
        let field_type = UITestFieldType::PANEL;
        let name =
            utilities::convert_label_to_name(associated_label, field_type.is_unlimited_segments());
        self.root_panel.set_name(Some(&format!(
            "{}{}{}",
            field_type,
            SEPARATOR_CHAR,
            name.as_deref().unwrap_or("null")
        )));
        if ARGUMENTS.lock().unwrap().is_print_names() {
            println!(
                "{} {} ",
                self.root_panel.get_name().as_deref().unwrap_or("null"),
                DEFAULT_DELIMITER
            );
        }
    }

    /// Java `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.clone()
    }

    /// Java `equalsOpenClose(ExpandButton)`: identity.
    pub fn equals_open_close(&self, button: &Rc<ExpandButton>) -> bool {
        self.btn_open_close
            .as_ref()
            .is_some_and(|own| Rc::ptr_eq(own, button))
    }

    /// Java `equalsAdvancedBasic(ExpandButton)`: identity.
    pub fn equals_advanced_basic(&self, button: &Rc<ExpandButton>) -> bool {
        self.btn_advanced_basic
            .as_ref()
            .is_some_and(|own| Rc::ptr_eq(own, button))
    }

    /// Java `equalsMoreLess(ExpandButton)`: identity.
    pub fn equals_more_less(&self, button: &Rc<ExpandButton>) -> bool {
        self.btn_more_less
            .as_ref()
            .is_some_and(|own| Rc::ptr_eq(own, button))
    }

    /// Java `setAdvanced(boolean)`.
    pub fn set_advanced(&self, advanced: bool) {
        let Some(button) = &self.btn_advanced_basic else {
            return;
        };
        button.set_expanded(advanced);
    }

    /// Java `setOpen(boolean)`.
    pub fn set_open(&self, open: bool) {
        let Some(button) = &self.btn_open_close else {
            return;
        };
        button.set_expanded(open);
    }

    /// Java `getTitle()`.
    pub fn get_title(&self) -> Option<String> {
        self.title.clone()
    }

    /// Java `getOpenCloseButton()`.
    pub fn get_open_close_button(&self) -> Option<Rc<ExpandButton>> {
        self.btn_open_close.clone()
    }

    /// Java `getMoreLessButton()`.
    pub fn get_more_less_button(&self) -> Option<Rc<ExpandButton>> {
        self.btn_more_less.clone()
    }

    /// Java `isLess()`.
    pub fn is_less(&self) -> bool {
        let Some(button) = &self.btn_more_less else {
            return false;
        };
        !button.is_expanded()
    }

    /// Java `getState(PanelHeaderState)`.
    pub fn get_state(&self, state: Option<&PanelHeaderState>) {
        let Some(state) = state else {
            return;
        };
        if let Some(button) = &self.btn_open_close {
            state.set_open_close_state(Some(button.get_state()));
        }
        if let Some(button) = &self.btn_advanced_basic {
            state.set_advanced_basic_state(Some(button.get_state()));
        }
        if let Some(button) = &self.btn_more_less
            && self.save_more_less_state.get()
        {
            state.set_more_less_state(Some(button.get_state()));
        }
    }

    /// Java `set(ConstPanelHeaderSettings)`.
    pub fn set(&self, settings: Option<&dyn ConstPanelHeaderSettings>) {
        let Some(settings) = settings else {
            return;
        };
        if let Some(button) = &self.btn_open_close {
            button.set_expanded(settings.is_open());
        }
        if let Some(button) = &self.btn_advanced_basic {
            button.set_expanded(settings.is_advanced());
        }
        if let Some(button) = &self.btn_more_less {
            button.set_expanded(settings.is_more());
        }
    }

    /// Java `updateOpenCloseButton(ConstPanelHeaderSettings)`.
    pub fn update_open_close_button(&self, settings: Option<&dyn ConstPanelHeaderSettings>) {
        let Some(settings) = settings else {
            return;
        };
        let Some(button) = &self.btn_open_close else {
            return;
        };
        button.update(settings.is_open());
    }

    /// Java `setState(ConstPanelHeaderState)`.
    pub fn set_state(&self, state: Option<&dyn ConstPanelHeaderState>) {
        let Some(state) = state else {
            return;
        };
        if let Some(button) = &self.btn_open_close {
            button.set_state(state.get_open_close_state().as_deref());
        }
        if let Some(button) = &self.btn_advanced_basic {
            button.set_state(state.get_advanced_basic_state().as_deref());
        }
        if let Some(button) = &self.btn_more_less
            && self.save_more_less_state.get()
        {
            button.set_state(state.get_more_less_state().as_deref());
        }
    }

    /// Java `setSaveMoreLessState(boolean)`.
    pub fn set_save_more_less_state(&self, input: bool) {
        self.save_more_less_state.set(input);
    }

    /// Java `setButtonStates(BaseScreenState)`.
    pub fn set_button_states_base_screen_state(&self, screen_state: Option<&BaseScreenState>) {
        self.set_button_states_base_screen_state_boolean(screen_state, true);
    }

    /// Java `createButtonStateKeys()`.
    pub fn create_button_state_keys(&self) {
        if let Some(button) = &self.btn_open_close {
            button.create_button_state_key(self.dialog_type);
        }
        if let Some(button) = &self.btn_advanced_basic {
            button.create_button_state_key(self.dialog_type);
        }
        if let Some(button) = &self.btn_more_less {
            button.create_button_state_key(self.dialog_type);
        }
    }

    /// Java `setButtonStates(BaseScreenState, boolean)`.
    pub fn set_button_states_base_screen_state_boolean(
        &self,
        screen_state: Option<&BaseScreenState>,
        default_is_open: bool,
    ) {
        let Some(screen_state) = screen_state else {
            return;
        };
        if let Some(button) = &self.btn_open_close {
            button.set_button_state(screen_state.get_button_state_with_default(
                button.create_button_state_key(self.dialog_type).as_deref(),
                default_is_open,
            ));
        }
        if let Some(button) = &self.btn_advanced_basic {
            button.set_button_state(
                screen_state
                    .get_button_state(button.create_button_state_key(self.dialog_type).as_deref()),
            );
        }
        if let Some(button) = &self.btn_more_less
            && self.save_more_less_state.get()
        {
            button.set_button_state(
                screen_state
                    .get_button_state(button.create_button_state_key(self.dialog_type).as_deref()),
            );
        }
    }

    /// Java `getButtonStates(BaseScreenState)`.
    pub fn get_button_states(&self, screen_state: Option<&BaseScreenState>) {
        let Some(screen_state) = screen_state else {
            return;
        };
        if let Some(button) = &self.btn_open_close {
            screen_state.set_button_state(
                button.get_button_state_key().as_deref(),
                button.get_button_state(),
            );
        }
        if let Some(button) = &self.btn_advanced_basic {
            screen_state.set_button_state(
                button.get_button_state_key().as_deref(),
                button.get_button_state(),
            );
        }
        if let Some(button) = &self.btn_more_less
            && self.save_more_less_state.get()
        {
            screen_state.set_button_state(
                button.get_button_state_key().as_deref(),
                button.get_button_state(),
            );
        }
    }
}

impl Expandable for PanelHeader {
    /// Java `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, _button: &Rc<GlobalExpandButton>) {}

    /// Java `expand(ExpandButton)`.
    fn expand_expand_button(&self, button: &Rc<ExpandButton>) {
        if self.titled
            && (self.equals_open_close(button)
                || (self.equals_advanced_basic(button) && self.empty_when_basic))
        {
            ui_harness::with(|harness| harness.pack_base_manager(None));
        }
    }
}

impl ConstPanelHeaderSettings for PanelHeader {
    /// Java `isAdvanced()`.
    fn is_advanced(&self) -> bool {
        let Some(button) = &self.btn_advanced_basic else {
            return false;
        };
        button.is_expanded()
    }

    /// Java `isMore()`.
    fn is_more(&self) -> bool {
        let Some(button) = &self.btn_more_less else {
            return false;
        };
        button.is_expanded()
    }

    /// Java `isOpen()`.
    fn is_open(&self) -> bool {
        let Some(button) = &self.btn_open_close else {
            return false;
        };
        button.is_expanded()
    }

    /// Java `isAdvancedNull()`.
    fn is_advanced_null(&self) -> bool {
        self.btn_advanced_basic.is_none()
    }

    /// Java `isMoreNull()`.
    fn is_more_null(&self) -> bool {
        self.btn_more_less.is_none()
    }

    /// Java `isOpenNull()`.
    fn is_open_null(&self) -> bool {
        self.btn_open_close.is_none()
    }
}

/// Java private inner class `PanelHeader.AdvancedFieldDisplayer implements
/// FieldDisplayer`.
struct AdvancedFieldDisplayer {
    header: Weak<PanelHeader>,
}

impl FieldDisplayer for AdvancedFieldDisplayer {
    /// Java `display()`.
    fn display_void(&self) {
        let Some(header) = self.header.upgrade() else {
            return;
        };
        if let Some(button) = &header.btn_advanced_basic
            && !button.is_expanded()
        {
            button.do_click();
        }
    }

    /// Java `display(UIComponent)`.
    fn display_ui_component(&self, _ui_component: Option<&dyn UIComponent>) {
        self.display_void();
    }
}
