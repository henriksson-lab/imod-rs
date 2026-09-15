//! Translation of `IMOD/3dmod/formv_objed.cpp` together with
//! `IMOD/3dmod/formv_objed.h`.
//!
//! The upstream class owns the regular Qt widgets in the Object Edit dock.
//! Dynamic panel widgets are constructed by `ObjectEditField` in `mv_objed`;
//! their Qt construction, sizing, and event dispatch remain at the native GUI
//! boundary, while this unit retains every slot's source state and forwarding.
#![allow(dead_code)]

use crate::imod::three_dmod::imodv::ImodvApp;
use crate::imod::three_dmod::mv_objed::{
    self, ImodvObjed, OBJECT_EDIT_FIELD_DATA, imodv_objed_closing, imodv_objed_ctrl_key,
    imodv_objed_draw_data, imodv_objed_edit_data, imodv_objed_frame_picked,
    imodv_objed_make_on_offs, imodv_objed_name, imodv_objed_select, imodv_objed_style_data,
    meshing_busy,
};
use crate::imod::three_dmod::mv_window::KeyEvent;

/// `QColor` value used by `updateObject` and `updateColorBox`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjedColor {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

/// `QEvent` cases inspected by `imodvObjedForm::topChangeEvent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjedChangeEvent {
    FontChange,
    Other,
}

/// Native Qt docking/form and generated-`Ui` operations used directly by the
/// paired source unit.
pub trait ImodvObjedFormNativeBoundary {
    fn setup_ui(&mut self);
    fn set_delete_on_close(&mut self);
    fn set_always_show_tool_tips(&mut self);
    fn connect_objed_signals(&mut self);
    fn add_panel_list_item(&mut self, label: &str);
    fn make_object_edit_field_widget(&mut self, item: i32);
    fn object_edit_field_size_hint(&mut self, item: i32) -> (i32, i32);
    fn fix_object_edit_field_widget(&mut self, item: i32);
    fn process_events(&mut self);
    fn set_panel_frame_minimum_size(&mut self, width: i32, height: i32);
    fn panel_list_item_width(&self, item: i32) -> i32;
    fn panel_list_count(&self) -> i32;
    fn panel_list_font_height(&self) -> i32;
    fn device_pixel_ratio(&self) -> f64;
    fn set_panel_list_fixed_size(&mut self, width: i32, height: i32);
    fn set_stack_current_index(&mut self, item: i32);
    fn set_panel_list_current_row(&mut self, item: i32);
    fn set_sync_to_current_object_visible(&mut self, visible: bool);
    fn set_sync_to_current_object_checked(&mut self, checked: bool);
    fn set_object_spin_range_value(&mut self, minimum: i32, maximum: i32, value: i32);
    fn set_object_spin_enabled(&mut self, enabled: bool);
    fn set_object_slider_maximum(&mut self, maximum: i32);
    fn set_object_slider_value(&mut self, value: i32);
    fn set_object_slider_enabled(&mut self, enabled: bool);
    fn set_data_type_index(&mut self, item: i32);
    fn set_draw_style_index(&mut self, item: i32);
    fn set_name_text(&mut self, name: &str);
    fn set_color_box(&mut self, color: ObjedColor);
    fn set_one_all_index(&mut self, item: i32);
    fn widget_change_event(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn ignore_close_event(&mut self);
    fn accept_close_event(&mut self);
    fn hot_slider_enabled(&self) -> bool;
    fn hot_slider_key(&self, event: KeyEvent) -> bool;
    fn grab_keyboard(&mut self);
    fn release_keyboard(&mut self);
    fn close_key(&self, event: KeyEvent) -> bool;
    fn imodv_key_press(&mut self, event: KeyEvent);
    fn imodv_key_release(&mut self, event: KeyEvent);
    fn retranslate_ui(&mut self);
}

/// Original `imodvObjedForm`, including generated form state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImodvObjedForm {
    pub m_stack_index: i32,
    pub panel_labels: Vec<String>,
    pub panel_width: i32,
    pub panel_height: i32,
    pub panel_list_width: i32,
    pub panel_list_height: i32,
    pub object_spin_value: i32,
    pub object_spin_maximum: i32,
    pub object_spin_enabled: bool,
    pub object_slider_value: i32,
    pub object_slider_maximum: i32,
    pub object_slider_enabled: bool,
    pub name: String,
    pub draw_type: i32,
    pub draw_style: i32,
    pub one_all: i32,
    pub sync_to_current_object: bool,
    pub sync_visible: bool,
    pub color: ObjedColor,
    pub delete_on_close: bool,
    pub always_show_tooltips: bool,
}

impl ImodvObjedForm {
    /// `imodvObjedForm::imodvObjedForm`.
    pub fn new(app: &ImodvApp, native: &mut dyn ImodvObjedFormNativeBoundary) -> ImodvObjedForm {
        native.setup_ui();
        let mut form = ImodvObjedForm::default();
        form.init(app, native);
        form
    }

    /// `imodvObjedForm::~imodvObjedForm`.
    pub fn destroy(&mut self) {}

    /// `imodvObjedForm::languageChange`.
    pub fn language_change(&mut self, native: &mut dyn ImodvObjedFormNativeBoundary) {
        native.retranslate_ui();
    }

    /// `imodvObjedForm::init`.
    pub fn init(&mut self, app: &ImodvApp, native: &mut dyn ImodvObjedFormNativeBoundary) {
        native.set_delete_on_close();
        native.set_always_show_tool_tips();
        self.delete_on_close = true;
        self.always_show_tooltips = true;
        native.connect_objed_signals();

        let (mut width, mut height) = (0, 0);
        for (item, field) in OBJECT_EDIT_FIELD_DATA.iter().enumerate() {
            native.add_panel_list_item(field.label);
            native.make_object_edit_field_widget(item as i32);
            let (field_width, field_height) = native.object_edit_field_size_hint(item as i32);
            width = width.max(field_width);
            height = height.max(field_height);
            self.panel_labels.push(field.label.into());
        }
        if app.standalone != 0 {
            native.set_sync_to_current_object_visible(false);
            self.sync_visible = false;
        } else {
            self.sync_to_current_object = app.sync_objed_to_cur_obj != 0;
            self.sync_visible = true;
            native.set_sync_to_current_object_visible(true);
            native.set_sync_to_current_object_checked(self.sync_to_current_object);
        }
        self.set_font_dependent_sizes(width, height, native);
        native.set_stack_current_index(0);
        self.m_stack_index = 0;
        // The source calls this after connections and panel creation.
        let _ = imodv_objed_make_on_offs(app);
    }

    /// `imodvObjedForm::setFontDependentSizes`.
    pub fn set_font_dependent_sizes(
        &mut self,
        width: i32,
        height: i32,
        native: &mut dyn ImodvObjedFormNativeBoundary,
    ) {
        self.panel_width = width + 8;
        self.panel_height = height + 8;
        native.set_panel_frame_minimum_size(self.panel_width, self.panel_height);
        let mut max_width = 0;
        for item in 0..native.panel_list_count() {
            max_width = max_width.max(native.panel_list_item_width(item));
        }
        self.panel_list_width = max_width + 18;
        let ratio_extra = if native.device_pixel_ratio() > 1.25 {
            8
        } else {
            22
        };
        self.panel_list_height =
            (1.17 * native.panel_list_count() as f64 * native.panel_list_font_height() as f64
                + ratio_extra as f64)
                .round() as i32;
        native.set_panel_list_fixed_size(self.panel_list_width, self.panel_list_height);
    }

    /// `imodvObjedForm::topChangeEvent`.
    pub fn top_change_event(
        &mut self,
        event: ObjedChangeEvent,
        native: &mut dyn ImodvObjedFormNativeBoundary,
    ) {
        native.widget_change_event();
        native.check_and_set_mac_menu();
        if event != ObjedChangeEvent::FontChange {
            return;
        }
        let (mut width, mut height) = (0, 0);
        for item in 0..OBJECT_EDIT_FIELD_DATA.len() as i32 {
            native.fix_object_edit_field_widget(item);
            native.process_events();
            let (field_width, field_height) = native.object_edit_field_size_hint(item);
            width = width.max(field_width);
            height = height.max(field_height);
        }
        self.set_font_dependent_sizes(width, height, native);
    }

    /// `imodvObjedForm::objectSelected`.
    pub fn object_selected(&mut self, app: &mut ImodvApp, which: i32) {
        self.object_spin_value = which;
        imodv_objed_select(app, which);
    }

    /// `imodvObjedForm::editSelected`.
    pub fn edit_selected(&mut self, editor: &mut ImodvObjed, item: i32) {
        self.one_all = item;
        imodv_objed_edit_data(editor, item);
    }

    /// `imodvObjedForm::objSliderChanged`.
    pub fn obj_slider_changed(&mut self, app: &mut ImodvApp, value: i32) {
        self.object_slider_value = value;
        imodv_objed_select(app, value);
    }

    /// `imodvObjedForm::nameChanged`.
    pub fn name_changed(&mut self, app: &mut ImodvApp, name: &str) {
        self.name = name.into();
        imodv_objed_name(app, name);
    }

    /// `imodvObjedForm::syncToCurObjToggled`.
    pub fn sync_to_cur_obj_toggled(&mut self, app: &mut ImodvApp, state: bool) {
        self.sync_to_current_object = state;
        app.sync_objed_to_cur_obj = state as i32;
        if state {
            imodv_objed_select(app, self.object_spin_value);
        }
    }

    /// `imodvObjedForm::typeSelected`.
    pub fn type_selected(&mut self, app: &mut ImodvApp, item: i32) {
        self.draw_type = item;
        imodv_objed_draw_data(app, item, false);
    }

    /// `imodvObjedForm::styleSelected`.
    pub fn style_selected(&mut self, app: &mut ImodvApp, item: i32) {
        self.draw_style = item;
        imodv_objed_style_data(app, item, false);
    }

    /// `imodvObjedForm::frameSelected`.
    pub fn frame_selected(
        &mut self,
        editor: &mut ImodvObjed,
        item: i32,
        native: &mut dyn ImodvObjedFormNativeBoundary,
    ) {
        self.m_stack_index = item;
        native.set_stack_current_index(item);
        imodv_objed_frame_picked(editor, item);
    }

    /// `imodvObjedForm::updateObject`.
    pub fn update_object(
        &mut self,
        ob: i32,
        num_obj: i32,
        draw_type: i32,
        draw_style: i32,
        color: ObjedColor,
        name: &str,
        native: &mut dyn ImodvObjedFormNativeBoundary,
    ) {
        self.update_color_box(color, native);
        self.object_spin_value = ob;
        self.object_spin_maximum = num_obj;
        self.object_spin_enabled = num_obj > 1;
        native.set_object_spin_range_value(1, num_obj, ob);
        native.set_object_spin_enabled(self.object_spin_enabled);
        self.object_slider_value = ob;
        self.object_slider_maximum = num_obj;
        self.object_slider_enabled = num_obj > 1;
        native.set_object_slider_maximum(num_obj);
        native.set_object_slider_value(ob);
        native.set_object_slider_enabled(self.object_slider_enabled);
        self.draw_type = draw_type;
        self.draw_style = draw_style;
        native.set_data_type_index(draw_type);
        native.set_draw_style_index(draw_style);
        self.name = name.into();
        native.set_name_text(name);
    }

    /// `imodvObjedForm::updateColorBox`.
    pub fn update_color_box(
        &mut self,
        color: ObjedColor,
        native: &mut dyn ImodvObjedFormNativeBoundary,
    ) {
        self.color = color;
        native.set_color_box(color);
    }

    /// `imodvObjedForm::setCurrentFrame`.
    pub fn set_current_frame(
        &mut self,
        frame: i32,
        edit_data: i32,
        native: &mut dyn ImodvObjedFormNativeBoundary,
    ) {
        self.m_stack_index = frame;
        self.one_all = edit_data;
        native.set_stack_current_index(frame);
        native.set_panel_list_current_row(frame);
        native.set_one_all_index(edit_data);
    }

    /// `imodvObjedForm::topCloseEvent`.
    pub fn top_close_event(
        &mut self,
        editor: &mut ImodvObjed,
        native: &mut dyn ImodvObjedFormNativeBoundary,
    ) {
        if meshing_busy(editor) {
            native.ignore_close_event();
            return;
        }
        imodv_objed_closing(editor);
        native.accept_close_event();
    }

    /// `imodvObjedForm::keyPressEvent`.
    pub fn key_press_event(
        &mut self,
        editor: &mut ImodvObjed,
        event: KeyEvent,
        native: &mut dyn ImodvObjedFormNativeBoundary,
    ) {
        if native.hot_slider_enabled() && native.hot_slider_key(event) {
            imodv_objed_ctrl_key(editor, true);
            native.grab_keyboard();
        }
        if native.close_key(event) {
            mv_objed::imodv_objed_done(editor);
        } else {
            native.imodv_key_press(event);
        }
    }

    /// `imodvObjedForm::keyReleaseEvent`.
    pub fn key_release_event(
        &mut self,
        editor: &mut ImodvObjed,
        event: KeyEvent,
        native: &mut dyn ImodvObjedFormNativeBoundary,
    ) {
        if native.hot_slider_key(event) {
            imodv_objed_ctrl_key(editor, false);
            native.release_keyboard();
        }
        native.imodv_key_release(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::{Imod, Iobj};
    use crate::imod::three_dmod::mv_window::Key;

    #[derive(Default)]
    struct Native {
        stack: i32,
        color: ObjedColor,
        accepted: bool,
        ignored: bool,
        grabbed: bool,
        change_called: bool,
    }
    impl ImodvObjedFormNativeBoundary for Native {
        fn setup_ui(&mut self) {}
        fn set_delete_on_close(&mut self) {}
        fn set_always_show_tool_tips(&mut self) {}
        fn connect_objed_signals(&mut self) {}
        fn add_panel_list_item(&mut self, _: &str) {}
        fn make_object_edit_field_widget(&mut self, _: i32) {}
        fn object_edit_field_size_hint(&mut self, item: i32) -> (i32, i32) {
            (30 + item, 10 + item)
        }
        fn fix_object_edit_field_widget(&mut self, _: i32) {}
        fn process_events(&mut self) {}
        fn set_panel_frame_minimum_size(&mut self, _: i32, _: i32) {}
        fn panel_list_item_width(&self, item: i32) -> i32 {
            item + 10
        }
        fn panel_list_count(&self) -> i32 {
            OBJECT_EDIT_FIELD_DATA.len() as i32
        }
        fn panel_list_font_height(&self) -> i32 {
            12
        }
        fn device_pixel_ratio(&self) -> f64 {
            1.0
        }
        fn set_panel_list_fixed_size(&mut self, _: i32, _: i32) {}
        fn set_stack_current_index(&mut self, item: i32) {
            self.stack = item
        }
        fn set_panel_list_current_row(&mut self, _: i32) {}
        fn set_sync_to_current_object_visible(&mut self, _: bool) {}
        fn set_sync_to_current_object_checked(&mut self, _: bool) {}
        fn set_object_spin_range_value(&mut self, _: i32, _: i32, _: i32) {}
        fn set_object_spin_enabled(&mut self, _: bool) {}
        fn set_object_slider_maximum(&mut self, _: i32) {}
        fn set_object_slider_value(&mut self, _: i32) {}
        fn set_object_slider_enabled(&mut self, _: bool) {}
        fn set_data_type_index(&mut self, _: i32) {}
        fn set_draw_style_index(&mut self, _: i32) {}
        fn set_name_text(&mut self, _: &str) {}
        fn set_color_box(&mut self, color: ObjedColor) {
            self.color = color
        }
        fn set_one_all_index(&mut self, _: i32) {}
        fn widget_change_event(&mut self) {
            self.change_called = true;
        }
        fn check_and_set_mac_menu(&mut self) {}
        fn ignore_close_event(&mut self) {
            self.ignored = true
        }
        fn accept_close_event(&mut self) {
            self.accepted = true
        }
        fn hot_slider_enabled(&self) -> bool {
            true
        }
        fn hot_slider_key(&self, event: KeyEvent) -> bool {
            event.key == Key::Character('h')
        }
        fn grab_keyboard(&mut self) {
            self.grabbed = true
        }
        fn release_keyboard(&mut self) {
            self.grabbed = false
        }
        fn close_key(&self, _: KeyEvent) -> bool {
            false
        }
        fn imodv_key_press(&mut self, _: KeyEvent) {}
        fn imodv_key_release(&mut self, _: KeyEvent) {}
        fn retranslate_ui(&mut self) {}
    }

    #[test]
    fn source_slots_keep_widget_and_objed_state_in_sync() {
        let mut model = Box::new(Imod::default());
        model.obj.push(Iobj::default());
        let mut app = ImodvApp::default();
        app.imod = &mut *model;
        app.mod_.push(std::ptr::NonNull::from(&mut *model));
        app.num_mods = 1;
        let mut native = Native::default();
        let mut form = ImodvObjedForm::new(&app, &mut native);
        let mut editor = ImodvObjed::default();
        form.update_object(
            1,
            2,
            2,
            3,
            ObjedColor {
                red: 1,
                green: 2,
                blue: 3,
            },
            "obj",
            &mut native,
        );
        form.edit_selected(&mut editor, 1);
        form.frame_selected(&mut editor, 2, &mut native);
        assert_eq!(
            (
                form.object_spin_enabled,
                form.name.as_str(),
                native.color,
                editor.current_panel_frame
            ),
            (
                true,
                "obj",
                ObjedColor {
                    red: 1,
                    green: 2,
                    blue: 3
                },
                2
            )
        );
    }

    #[test]
    fn source_hot_key_and_meshing_close_paths_are_preserved() {
        let app = ImodvApp::default();
        let mut native = Native::default();
        let mut form = ImodvObjedForm::new(&app, &mut native);
        let mut editor = ImodvObjed {
            dialog_open: true,
            ..Default::default()
        };
        form.key_press_event(
            &mut editor,
            KeyEvent {
                key: Key::Character('h'),
                ..Default::default()
            },
            &mut native,
        );
        assert!(editor.ctrl_pressed && native.grabbed);
        editor.mesh_busy = true;
        form.top_close_event(&mut editor, &mut native);
        assert!(native.ignored && editor.dialog_open);
    }

    #[test]
    fn source_change_event_calls_base_widget_boundary() {
        let app = ImodvApp::default();
        let mut native = Native::default();
        let mut form = ImodvObjedForm::new(&app, &mut native);
        form.top_change_event(ObjedChangeEvent::Other, &mut native);
        assert!(native.change_called);
    }
}
