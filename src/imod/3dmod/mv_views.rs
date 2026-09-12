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

/// Save operations that cross from `mv_views.cpp` into the standalone or ImodView host.
pub trait ImodvViewsNativeBoundary {
    fn imodv_file_save(&mut self) {}
    fn input_save_model(&mut self) {}
    /// `imodvControlSetView`.
    fn control_set_view(&mut self, _a: &mut ImodvApp) {}
    /// `imodvObjedNewView`.
    fn objed_new_view(&mut self, _a: &mut ImodvApp) {}
    /// `imodvDepthCueSetWidgets`.
    fn depth_cue_set_widgets(&mut self, _a: &mut ImodvApp) {}
    fn menu_light(&mut self, _enabled: bool) {}
    fn menu_labels(&mut self, _enabled: bool) {}
    fn menu_wireframe(&mut self, _enabled: bool) {}
    fn menu_lowres(&mut self, _enabled: bool) {}
    fn menu_invert_z(&mut self, _enabled: bool) {}
    /// `imodvNewModelAngles` after a stored view has been made current.
    fn new_model_angles(&mut self) {}
    /// `imodvDrawImodImages(a->linkToSlicer)`.
    fn draw_imod_images(&mut self, _link_to_slicer: i32) {}
}
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
pub fn imodv_update_view(a: &mut ImodvApp, n: &mut dyn ImodvViewsNativeBoundary) {
    let Some(model) = (unsafe { a.imod.as_ref() }) else {
        return;
    };
    let view = &model.view[0];
    a.wireframe = (view.world & VIEW_WORLD_WIREFRAME != 0) as i32;
    a.lowres = (view.world & VIEW_WORLD_LOWRES != 0) as i32;
    a.lighting = (view.world & VIEW_WORLD_LIGHT != 0) as i32;
    a.draw_labels = (view.world & VIEW_WORLD_LABELS != 0) as i32;
    a.invert_z = (view.world & VIEW_WORLD_INVERT_Z != 0) as i32;
    n.control_set_view(a);
    n.objed_new_view(a);
    n.depth_cue_set_widgets(a);
    n.menu_light(a.lighting != 0);
    n.menu_labels(a.draw_labels != 0);
    n.menu_wireframe(a.wireframe != 0);
    n.menu_lowres(a.lowres != 0);
    n.menu_invert_z(a.invert_z != 0);
}
pub fn imodv_update_model(
    a: &mut ImodvApp,
    set_view: bool,
    form: Option<&mut ImodvViewsForm>,
    state: &ImodvViewed,
    n: &mut dyn ImodvViewsNativeBoundary,
) {
    imodv_update_view(a, n);
    if let Some(form) = form {
        form.remove_all_items();
        build_list(a, form);
        if set_view {
            let cview = unsafe { a.imod.as_ref() }.map_or(0, |m| m.cview);
            imodv_views_set_view(a, state, cview, false, false, false, n);
        }
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
pub fn imodv_views_save(a: &mut ImodvApp, n: &mut dyn ImodvViewsNativeBoundary) {
    if a.standalone != 0 {
        n.imodv_file_save();
    } else {
        n.input_save_model();
    }
}
pub fn imodv_views_goto(
    a: &mut ImodvApp,
    state: &ImodvViewed,
    item: i32,
    draw: bool,
    reg: bool,
    n: &mut dyn ImodvViewsNativeBoundary,
) {
    imodv_views_set_view(a, state, item + 1, draw, false, reg, n)
}
pub fn imodv_views_set_view(
    a: &mut ImodvApp,
    state: &ImodvViewed,
    view: i32,
    draw: bool,
    external: bool,
    reg: bool,
    n: &mut dyn ImodvViewsNativeBoundary,
) {
    let Some(model) = (unsafe { a.imod.as_ref() }) else {
        return;
    };
    if view <= 0 || view as usize >= model.view.len() {
        return;
    }
    let current_view = model.cview;
    if reg {
        imodv_finish_chg_unit()
    }
    if view != current_view {
        imodv_auto_store_view(a, state)
    }
    let Some(model) = (unsafe { a.imod.as_mut() }) else {
        return;
    };
    model.cview = view;
    imod_view_use(model);
    n.new_model_angles();
    n.draw_imod_images(a.link_to_slicer);
    imodv_update_view(a, n);
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
pub fn imodv_views_delete(
    a: &mut ImodvApp,
    state: &ImodvViewed,
    item: i32,
    new_current: i32,
    n: &mut dyn ImodvViewsNativeBoundary,
) {
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
    let view = model.cview;
    let _ = model;
    imodv_views_goto(a, state, view - 1, true, false, n)
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
    #[derive(Default)]
    struct Native {
        standalone_saves: usize,
        imodview_saves: usize,
    }
    impl ImodvViewsNativeBoundary for Native {
        fn imodv_file_save(&mut self) {
            self.standalone_saves += 1;
        }
        fn input_save_model(&mut self) {
            self.imodview_saves += 1;
        }
    }
    #[derive(Default)]
    struct ViewFeedback {
        events: Vec<String>,
    }
    impl ImodvViewsNativeBoundary for ViewFeedback {
        fn control_set_view(&mut self, _: &mut ImodvApp) {
            self.events.push("control".into());
        }
        fn objed_new_view(&mut self, _: &mut ImodvApp) {
            self.events.push("objed".into());
        }
        fn depth_cue_set_widgets(&mut self, _: &mut ImodvApp) {
            self.events.push("depthcue".into());
        }
        fn menu_light(&mut self, enabled: bool) {
            self.events.push(format!("light:{enabled}"));
        }
        fn menu_labels(&mut self, enabled: bool) {
            self.events.push(format!("labels:{enabled}"));
        }
        fn menu_wireframe(&mut self, enabled: bool) {
            self.events.push(format!("wire:{enabled}"));
        }
        fn menu_lowres(&mut self, enabled: bool) {
            self.events.push(format!("lowres:{enabled}"));
        }
        fn menu_invert_z(&mut self, enabled: bool) {
            self.events.push(format!("invert:{enabled}"));
        }
        fn new_model_angles(&mut self) {
            self.events.push("angles".into());
        }
        fn draw_imod_images(&mut self, link_to_slicer: i32) {
            self.events.push(format!("images:{link_to_slicer}"));
        }
    }
    fn feedback_events() -> Vec<String> {
        [
            "control",
            "objed",
            "depthcue",
            "light:true",
            "labels:true",
            "wire:true",
            "lowres:true",
            "invert:true",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
    #[test]
    fn update_view_synchronizes_all_source_feedback_in_source_order() {
        let mut m = Box::new(Imod::default());
        m.view[0].world = VIEW_WORLD_LIGHT
            | VIEW_WORLD_LABELS
            | VIEW_WORLD_WIREFRAME
            | VIEW_WORLD_LOWRES
            | VIEW_WORLD_INVERT_Z;
        let mut a = ImodvApp::default();
        a.imod = &mut *m;
        let mut native = ViewFeedback::default();

        imodv_update_view(&mut a, &mut native);

        assert_eq!(native.events, feedback_events());
        assert_eq!(
            (a.lighting, a.draw_labels, a.wireframe, a.lowres, a.invert_z),
            (1, 1, 1, 1, 1)
        );
    }
    #[test]
    fn set_view_refreshes_images_before_view_feedback() {
        let mut m = Box::new(Imod::default());
        imod_view_model_new(&mut m);
        m.view[1].world = VIEW_WORLD_LIGHT;
        let mut a = ImodvApp {
            imod: &mut *m,
            link_to_slicer: 1,
            ..Default::default()
        };
        let mut native = ViewFeedback::default();

        imodv_views_set_view(
            &mut a,
            &ImodvViewed::default(),
            1,
            false,
            false,
            false,
            &mut native,
        );

        assert_eq!(native.events[..2], ["angles", "images:1"]);
        assert_eq!(
            &native.events[2..],
            [
                "control",
                "objed",
                "depthcue",
                "light:true",
                "labels:false",
                "wire:false",
                "lowres:false",
                "invert:false",
            ]
        );
    }
    #[test]
    fn update_model_reapplies_current_view_only_with_an_open_views_form() {
        let mut m = Box::new(Imod::default());
        imod_view_model_new(&mut m);
        m.cview = 1;
        m.view[1].world = VIEW_WORLD_LOWRES;
        let mut a = ImodvApp {
            imod: &mut *m,
            link_to_slicer: 1,
            ..Default::default()
        };
        let mut form = ImodvViewsForm::default();
        let mut native = ViewFeedback::default();

        imodv_update_model(
            &mut a,
            true,
            Some(&mut form),
            &ImodvViewed::default(),
            &mut native,
        );

        assert_eq!(native.events.len(), 18);
        assert_eq!(native.events[8..10], ["angles", "images:1"]);
        assert_eq!(form.current_item, 0);
    }
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
    #[test]
    fn save_uses_the_source_host_branch() {
        let mut native = Native::default();
        let mut a = ImodvApp {
            standalone: 1,
            ..Default::default()
        };
        imodv_views_save(&mut a, &mut native);
        a.standalone = 0;
        imodv_views_save(&mut a, &mut native);
        assert_eq!((native.standalone_saves, native.imodview_saves), (1, 1));
    }
}
