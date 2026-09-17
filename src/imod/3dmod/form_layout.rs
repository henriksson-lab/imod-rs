//! Translation of `IMOD/3dmod/form_layout.cpp` and `form_layout.h`.
#[derive(Clone, Debug, Default)]
pub struct ImodPrefStruct {
    pub remember_geom: bool,
    pub iconify_image_win: bool,
    pub iconify_imod_dlg: bool,
    pub iconify_imodv_dlg: bool,
    pub stack_imod_dlgs: bool,
    pub raise_imod_dlg_stack: bool,
    pub keep_dlg_stack_on_top: bool,
    pub stack_imodv_dlgs: bool,
    pub raise_imodv_dlg_stack: bool,
    pub dlg_frame_adjustment: i32,
}
pub trait LayoutOperations {
    fn dlg_frame_adj_changed(&mut self);
}
#[derive(Clone, Debug, Default)]
pub struct LayoutForm {
    pub prefs: ImodPrefStruct,
    pub geom_check_box: bool,
    pub image_iconify_box: bool,
    pub imod_dlg_iconify_box: bool,
    pub imodv_dlg_iconify_box: bool,
    pub dock_imod_dlgs_box: bool,
    pub raise_imod_stack_box: bool,
    pub keep_stack_on_top_box: bool,
    pub dock_imodv_dlgs_box: bool,
    pub raise_imodv_stack_box: bool,
    pub frame_adj_spin_box: i32,
    pub frame_adj_listener_count: usize,
    pub ui_setup: bool,
    pub ui_retranslated: bool,
    pub initialized: bool,
}
impl LayoutForm {
    /// `LayoutForm()` source constructor.
    pub fn new(prefs: &ImodPrefStruct) -> LayoutForm {
        let mut f = LayoutForm {
            prefs: prefs.clone(),
            ..Default::default()
        };
        f.init();
        f
    }
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self) {
        self.ui_retranslated = true;
    }
    pub fn init(&mut self) {
        self.initialized = true;
        self.ui_setup = true;
        self.frame_adj_listener_count += 1;
        self.update()
    }
    pub fn border_adj_changed(&mut self, ops: &mut dyn LayoutOperations, value: i32) {
        self.prefs.dlg_frame_adjustment = value;
        ops.dlg_frame_adj_changed()
    }
    pub fn update(&mut self) {
        self.geom_check_box = self.prefs.remember_geom;
        self.image_iconify_box = self.prefs.iconify_image_win;
        self.imod_dlg_iconify_box = self.prefs.iconify_imod_dlg;
        self.imodv_dlg_iconify_box = self.prefs.iconify_imodv_dlg;
        self.dock_imod_dlgs_box = self.prefs.stack_imod_dlgs;
        self.raise_imod_stack_box = self.prefs.raise_imod_dlg_stack;
        self.keep_stack_on_top_box = self.prefs.keep_dlg_stack_on_top;
        self.dock_imodv_dlgs_box = self.prefs.stack_imodv_dlgs;
        self.raise_imodv_stack_box = self.prefs.raise_imodv_dlg_stack;
        self.frame_adj_spin_box = self.prefs.dlg_frame_adjustment;
    }
    pub fn unload(&self, prefs: &mut ImodPrefStruct) {
        prefs.remember_geom = self.geom_check_box;
        prefs.iconify_image_win = self.image_iconify_box;
        prefs.iconify_imod_dlg = self.imod_dlg_iconify_box;
        prefs.iconify_imodv_dlg = self.imodv_dlg_iconify_box;
        prefs.stack_imod_dlgs = self.dock_imod_dlgs_box;
        prefs.raise_imod_dlg_stack = self.raise_imod_stack_box;
        prefs.keep_dlg_stack_on_top = self.keep_stack_on_top_box;
        prefs.stack_imodv_dlgs = self.dock_imodv_dlgs_box;
        prefs.raise_imodv_dlg_stack = self.raise_imodv_stack_box;
        prefs.dlg_frame_adjustment = self.frame_adj_spin_box
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Ops(i32);
    impl LayoutOperations for Ops {
        fn dlg_frame_adj_changed(&mut self) {
            self.0 += 1
        }
    }
    #[test]
    fn source_pref_roundtrip() {
        let p = ImodPrefStruct {
            remember_geom: true,
            ..Default::default()
        };
        let mut f = LayoutForm::new(&p);
        let mut o = Ops::default();
        // `form_layout.cpp:52` connects `frameAdjSpinBox`'s `valueChanged(int)`
        // to `borderAdjChanged(int)`, so Qt has already stored the new value in
        // the spin box by the time the slot runs.  `unload` then reads the
        // *widget* (`form_layout.cpp:89`, `frameAdjSpinBox->value()`), not the
        // prefs field the slot wrote.  The test therefore has to set the box as
        // `setValue(3)` would before invoking the slot -- it previously called
        // the slot alone and expected `unload` to see 3, which no sequence in
        // the source produces.  Same shape as `form_appearance.rs`'s tests.
        f.frame_adj_spin_box = 3;
        f.border_adj_changed(&mut o, 3);
        let mut out = ImodPrefStruct::default();
        f.unload(&mut out);
        assert_eq!((out.dlg_frame_adjustment, o.0), (3, 1));
        // The slot's own write (`form_layout.cpp:58`) lands on the form's prefs.
        assert_eq!(f.prefs.dlg_frame_adjustment, 3);
    }
}
