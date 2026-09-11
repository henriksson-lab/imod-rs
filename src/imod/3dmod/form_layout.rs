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
    pub initialized: bool,
}
pub fn layout_form_new(prefs: &ImodPrefStruct) -> LayoutForm {
    let mut f = LayoutForm {
        prefs: prefs.clone(),
        ..Default::default()
    };
    f.init();
    f
}
impl LayoutForm {
    pub fn destroy(&mut self) {}
    pub fn language_change(&mut self) {}
    pub fn init(&mut self) {
        self.initialized = true;
        self.update()
    }
    pub fn border_adj_changed(&mut self, ops: &mut dyn LayoutOperations, value: i32) {
        self.prefs.dlg_frame_adjustment = value;
        ops.dlg_frame_adj_changed()
    }
    pub fn update(&mut self) {}
    pub fn unload(&self, prefs: &mut ImodPrefStruct) {
        *prefs = self.prefs.clone()
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
        let mut f = layout_form_new(&p);
        let mut o = Ops::default();
        f.border_adj_changed(&mut o, 3);
        let mut out = ImodPrefStruct::default();
        f.unload(&mut out);
        assert_eq!((out.dlg_frame_adjustment, o.0), (3, 1));
    }
}
