//! Translation of `IMOD/3dmod/formv_modeled.cpp` and `formv_modeled.h`.
//!
//! This is the native-window event boundary for the model edit form.  Widget
//! values are retained directly and every source slot forwards to `mv_modeled`.
#![allow(dead_code)]
use crate::imod::three_dmod::imodv::ImodvApp;
use crate::imod::three_dmod::mv_input::{self, InputEvent, MvInputNativeBoundary};
use crate::imod::three_dmod::mv_modeled::{self, ImodvModeled};
/// Original `imodvModeledForm`.
#[derive(Clone, Debug, Default)]
pub struct ImodvModeledForm {
    pub top_window_open: bool,
    pub current_model: i32,
    pub model_maximum: i32,
    pub filename: String,
    pub internal_name: String,
    pub pixel_size: String,
    pub move_group: i32,
    pub edit_group: i32,
    pub view_selection: i32,
    pub same_scale_enabled: bool,
    pub delete_on_close: bool,
    pub always_show_tooltips: bool,
    pub has_focus: bool,
    pub close_event_accepted: bool,
    pub base_change_event_called: bool,
    pub mac_menu_checked: bool,
}
/// Portable QKeyEvent subset consumed by source `keyPressEvent` / `keyReleaseEvent`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModeledKeyEvent {
    pub close: bool,
    pub key: i32,
}
/// Original `imodvModeledForm::imodvModeledForm`.
pub fn imodv_modeled_form_new(app: &ImodvApp) -> ImodvModeledForm {
    ImodvModeledForm {
        top_window_open: true,
        same_scale_enabled: app.standalone != 0,
        delete_on_close: true,
        always_show_tooltips: true,
        ..Default::default()
    }
}
impl ImodvModeledForm {
    /// Original destructor `~imodvModeledForm`.
    pub fn destroy(&mut self) {
        self.top_window_open = false;
    }
    /// Original `languageChange`; Qt's translated strings are owned by the native window boundary.
    pub fn language_change(&mut self) {}
    pub fn model_changed(&mut self, app: &mut ImodvApp, dialog: &mut ImodvModeled, which: i32) {
        mv_modeled::imodv_modeled_number(app, which, dialog);
    }
    pub fn edit_clicked(&mut self, app: &mut ImodvApp, which: i32) {
        mv_modeled::imodv_modeled_edit(app, which);
    }
    pub fn move_clicked(&mut self, app: &mut ImodvApp, which: i32) {
        mv_modeled::imodv_modeled_move(app, which);
    }
    pub fn view_selected(&mut self, app: &mut ImodvApp, which: i32) {
        mv_modeled::imodv_modeled_view(app, which);
    }
    pub fn name_changed(&mut self, app: &mut ImodvApp, dialog: &mut ImodvModeled, name: &str) {
        self.internal_name = name.into();
        mv_modeled::imodv_modeled_name(app, name, dialog);
    }
    pub fn new_pixel_size(&mut self, app: &mut ImodvApp, dialog: &mut ImodvModeled) {
        dialog.pixel_string = self.pixel_size.clone();
        mv_modeled::imodv_modeled_scale(app, dialog, true);
        self.pixel_size = dialog.pixel_string.clone();
    }
    pub fn same_scale_clicked(&mut self, app: &mut ImodvApp) {
        mv_modeled::imodv_modeled_same_scale(app);
    }
    pub fn get_pixel_string(&self) -> String {
        self.pixel_size.clone()
    }
    pub fn set_model(
        &mut self,
        current: i32,
        count: i32,
        file: String,
        internal: String,
        pixels: String,
    ) {
        self.current_model = current;
        self.model_maximum = count;
        self.filename = file;
        self.internal_name = internal;
        self.pixel_size = pixels;
    }
    pub fn set_move_edit(&mut self, move_: i32, edit: i32) {
        self.move_group = move_;
        self.edit_group = edit;
    }
    pub fn set_view_selection(&mut self, which: i32) {
        self.view_selection = which;
    }
    pub fn top_close_event(&mut self, app: &mut ImodvApp, dialog: &mut ImodvModeled) {
        mv_modeled::imodv_modeled_closing(app, dialog);
        self.close_event_accepted = true;
        self.top_window_open = false;
    }
    pub fn key_press_event(
        &mut self,
        app: &mut ImodvApp,
        dialog: &mut ImodvModeled,
        event: ModeledKeyEvent,
        input: &mut dyn MvInputNativeBoundary,
    ) {
        if event.close {
            mv_modeled::imodv_modeled_done(dialog);
            self.top_window_open = false;
        } else {
            mv_input::imodv_key_press(app, InputEvent::default(), input);
        }
    }
    pub fn key_release_event(
        &mut self,
        app: &mut ImodvApp,
        _event: ModeledKeyEvent,
        input: &mut dyn MvInputNativeBoundary,
    ) {
        mv_input::imodv_key_release(app, InputEvent::default(), input);
    }
    pub fn top_change_event(&mut self, font_changed: bool) {
        self.base_change_event_called = true;
        self.mac_menu_checked = true;
        let _ = font_changed;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::Imod;
    #[test]
    fn source_slots_forward_model_state() {
        let mut m = Box::new(Imod::default());
        let mut app = ImodvApp::default();
        app.imod = &mut *m;
        app.mod_.push(&mut *m);
        app.num_mods = 1;
        let mut d = ImodvModeled::default();
        let mut f = imodv_modeled_form_new(&app);
        f.name_changed(&mut app, &mut d, "model A");
        assert_eq!(m.name[0], b'm');
        f.pixel_size = "2.5 nm".into();
        f.new_pixel_size(&mut app, &mut d);
        assert_eq!(m.pixsize, 2.5);
    }

    #[test]
    fn source_close_key_and_change_paths_dispatch() {
        let mut app = ImodvApp::default();
        let mut dialog = ImodvModeled::default();
        let mut form = imodv_modeled_form_new(&app);
        struct NoInput;
        impl crate::imod::three_dmod::imod_input::InputNativeBoundary for NoInput {}
        impl MvInputNativeBoundary for NoInput {}
        let mut input = NoInput;
        form.key_press_event(
            &mut app,
            &mut dialog,
            ModeledKeyEvent::default(),
            &mut input,
        );
        form.key_release_event(&mut app, ModeledKeyEvent::default(), &mut input);
        form.top_close_event(&mut app, &mut dialog);
        assert!(form.close_event_accepted);
        form.top_change_event(true);
        assert!(form.base_change_event_called && form.mac_menu_checked);
    }
}
