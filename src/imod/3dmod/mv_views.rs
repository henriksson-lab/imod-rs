//! Translation of `IMOD/3dmod/mv_views.cpp` and `mv_views.h`.
#![allow(dead_code, unused_variables)]
use crate::imod::libimod::imodel::{Imod, Ipoint, Iview};
use crate::imod::libimod::iview::{
    VIEW_WORLD_LIGHT, imod_objview_complete, imod_view_default_scale, imod_view_model_new,
    imod_view_store, imod_view_use,
};
use crate::imod::three_dmod::formv_views::{ImodvViewsForm, ViewsOperations};
use crate::imod::three_dmod::imodv::{
    ImodvApp, imodv_draw, imodv_finish_chg_unit, imodv_new_model_angles, imodv_register_model_chg,
};
pub const VIEW_WORLD_WIREFRAME: u32 = 1 << 4;
pub const VIEW_WORLD_LOWRES: u32 = 1 << 5;
pub const VIEW_WORLD_LABELS: u32 = 1 << 6;
pub const VIEW_WORLD_INVERT_Z: u32 = 0x40;
/// Original static `imodv_viewed` / `auto_store`.
#[derive(Clone, Debug)]
pub struct ImodvViewed {
    pub dialog_open: bool,
    pub auto_store: i32,
}
impl Default for ImodvViewed {
    fn default() -> Self {
        Self {
            dialog_open: false,
            auto_store: 1,
        }
    }
}
/// Original static `imodvUpdateView`.
pub fn imodv_update_view(a: &mut ImodvApp) {
    let Some(model) = (unsafe { a.imod.as_ref() }) else {
        return;
    };
    let view = &model.view[0];
    a.wireframe = (view.world & VIEW_WORLD_WIREFRAME != 0) as i32;
    a.lowres = (view.world & VIEW_WORLD_LOWRES != 0) as i32;
    a.lighting = (view.world & VIEW_WORLD_LIGHT != 0) as i32;
    a.draw_labels = (view.world & VIEW_WORLD_LABELS != 0) as i32;
    a.invert_z = (view.world & VIEW_WORLD_INVERT_Z != 0) as i32;
}
pub fn imodv_update_model(
    a: &mut ImodvApp,
    set_view: bool,
    form: Option<&mut ImodvViewsForm>,
    state: &ImodvViewed,
) {
    imodv_update_view(a);
    if let Some(form) = form {
        form.remove_all_items();
        build_list(a, form);
        if let Some(m) = unsafe { a.imod.as_ref() } {
            form.select_item(m.cview - 1, true)
        }
        form.set_autostore(state.auto_store)
    }
}
/// Original static `manage_world_flags`.
pub fn manage_world_flags(a: &ImodvApp, view: &mut Iview) {
    for (flag, value) in [
        (VIEW_WORLD_INVERT_Z, a.invert_z),
        (VIEW_WORLD_LIGHT, a.lighting),
        (VIEW_WORLD_LABELS, a.draw_labels),
        (VIEW_WORLD_WIREFRAME, a.wireframe),
        (VIEW_WORLD_LOWRES, a.lowres),
    ] {
        if value != 0 {
            view.world |= flag
        } else {
            view.world &= !flag
        }
    }
}
pub fn imodv_auto_store_view(a: &mut ImodvApp, state: &ImodvViewed) {
    let Some(model) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    imod_objview_complete(model);
    if state.auto_store == 0 || model.cview == 0 {
        return;
    }
    let c = model.cview;
    let (wire, low, light, labels, inv) =
        (a.wireframe, a.lowres, a.lighting, a.draw_labels, a.invert_z);
    let view = &mut model.view[0];
    for (flag, value) in [
        (VIEW_WORLD_INVERT_Z, inv),
        (VIEW_WORLD_LIGHT, light),
        (VIEW_WORLD_LABELS, labels),
        (VIEW_WORLD_WIREFRAME, wire),
        (VIEW_WORLD_LOWRES, low),
    ] {
        if value != 0 {
            view.world |= flag
        } else {
            view.world &= !flag
        }
    }
    imod_view_store(model, c);
}
pub fn imodv_views_done(state: &mut ImodvViewed) {
    state.dialog_open = false
}
pub fn imodv_views_closing(state: &mut ImodvViewed) {
    state.dialog_open = false
}
pub fn imodv_view_edit_dialog(
    a: &mut ImodvApp,
    state_value: i32,
    state: &mut ImodvViewed,
    form: &mut ImodvViewsForm,
) {
    if state_value == 0 {
        state.dialog_open = false;
        form.top_window_open = false;
        return;
    }
    state.dialog_open = true;
    form.top_window_open = true;
    form.remove_all_items();
    build_list(a, form);
    if let Some(m) = unsafe { a.imod.as_ref() } {
        form.select_item(m.cview - 1, true)
    }
    form.set_autostore(state.auto_store)
}
pub fn imodv_views_save(_a: &mut ImodvApp) {}
pub fn imodv_views_goto(a: &mut ImodvApp, state: &ImodvViewed, item: i32, draw: bool, reg: bool) {
    imodv_views_set_view(a, state, item + 1, draw, false, reg)
}
pub fn imodv_views_set_view(
    a: &mut ImodvApp,
    state: &ImodvViewed,
    view: i32,
    draw: bool,
    external: bool,
    reg: bool,
) {
    let Some(model) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    if view <= 0 || view as usize >= model.view.len() {
        return;
    }
    if reg {
        imodv_finish_chg_unit()
    }
    if view != model.cview {
        imodv_auto_store_view(a, state)
    }
    model.cview = view;
    imod_view_use(model);
    imodv_update_view(a);
    if draw {
        unsafe { imodv_draw() }
    }
}
pub fn imodv_views_store(a: &mut ImodvApp, item: i32) {
    let Some(model) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    let view = item + 1;
    if view <= 0 || view as usize >= model.view.len() {
        return;
    }
    imodv_register_model_chg();
    imodv_finish_chg_unit();
    model.cview = view;
    let current = &mut model.view[0];
    if a.invert_z != 0 {
        current.world |= VIEW_WORLD_INVERT_Z
    } else {
        current.world &= !VIEW_WORLD_INVERT_Z
    };
    imod_view_store(model, view);
}
pub fn imodv_views_new(
    a: &mut ImodvApp,
    state: &ImodvViewed,
    label: &str,
    form: Option<&mut ImodvViewsForm>,
) {
    imodv_auto_store_view(a, state);
    let Some(model) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    imodv_register_model_chg();
    imodv_finish_chg_unit();
    imod_view_model_new(model);
    let index = model.view.len() - 1;
    model.cview = index as i32;
    model.view[index].label = [0; 32];
    for (d, s) in model.view[index]
        .label
        .iter_mut()
        .zip(label.bytes().take(31))
    {
        *d = s;
    }
    imod_view_store(model, index as i32);
    if let Some(form) = form {
        form.add_item(label);
        form.select_item(index as i32 - 1, true)
    }
}
pub fn imodv_views_delete(a: &mut ImodvApp, state: &ImodvViewed, item: i32, new_current: i32) {
    let Some(model) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    let index = item + 1;
    if index <= 0 || index as usize >= model.view.len() || model.view.len() < 2 {
        return;
    }
    imodv_finish_chg_unit();
    model.view.remove(index as usize);
    model.cview = (if new_current >= 0 { new_current } else { 0 }) + 1;
    imodv_views_goto(a, state, model.cview - 1, true, false)
}
pub fn imodv_views_label(a: &mut ImodvApp, label: &str, item: i32) {
    let Some(model) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    if let Some(view) = model.view.get_mut((item + 1).max(0) as usize) {
        view.label = [0; 32];
        for (d, s) in view.label.iter_mut().zip(label.bytes().take(31)) {
            *d = s;
        }
        imodv_register_model_chg();
        imodv_finish_chg_unit();
    }
}
pub fn imodv_views_autostore(state: &mut ImodvViewed, value: i32) {
    state.auto_store = value
}
pub fn imodv_views_initialize(model: &mut Imod, a: &mut ImodvApp) {
    if model.view.len() < 2 {
        imod_view_model_new(model);
        model.view[1].label[..6].copy_from_slice(b"view 1");
    }
    if model.cview == 0 {
        model.cview = 1
    }
    if model.view[model.cview as usize].rad == 1. {
        let max = Ipoint::default();
        let i = model.cview as usize;
        let base = model.clone();
        imod_view_default_scale(&base, &mut model.view[i], &max, 1.);
    }
    imod_view_use(model);
    a.imod = model;
    a.invert_z = (model.view[0].world & VIEW_WORLD_INVERT_Z != 0) as i32;
    a.lighting = (model.view[0].world & VIEW_WORLD_LIGHT != 0) as i32;
    a.draw_labels = (model.view[0].world & VIEW_WORLD_LABELS != 0) as i32;
    a.wireframe = (model.view[0].world & VIEW_WORLD_WIREFRAME != 0) as i32;
    a.lowres = (model.view[0].world & VIEW_WORLD_LOWRES != 0) as i32;
}
/// Original static `build_list`.
pub fn build_list(a: &ImodvApp, form: &mut ImodvViewsForm) {
    let Some(model) = (unsafe { a.imod.as_ref() }) else {
        return;
    };
    for view in model.view.iter().skip(1) {
        let label = view
            .label
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as char)
            .collect::<String>();
        form.add_item(&label)
    }
}
impl ViewsOperations for ImodvViewed {
    fn store(&mut self, _: i32) {}
    fn goto(&mut self, _: i32, _: bool) {}
    fn new_view(&mut self, _: &str) {}
    fn delete(&mut self, _: i32, _: i32) {}
    fn save(&mut self) {}
    fn autostore(&mut self, state: i32) {
        self.auto_store = state
    }
    fn label(&mut self, _: &str, _: i32) {}
    fn closing(&mut self) {
        self.dialog_open = false
    }
    fn done(&mut self) {
        self.dialog_open = false
    }
    fn key_press(&mut self, _: i32) {}
    fn key_release(&mut self, _: i32) {}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initialization_and_view_lifecycle_use_model_views() {
        let mut m = Box::new(Imod::default());
        let mut a = ImodvApp::default();
        let mut s = ImodvViewed::default();
        imodv_views_initialize(&mut m, &mut a);
        assert_eq!(m.cview, 1);
        imodv_views_new(&mut a, &s, "second", None);
        assert_eq!(m.view.len(), 3);
        imodv_views_label(&mut a, "renamed", 1);
        assert_eq!(&m.view[2].label[..7], b"renamed");
    }
}
