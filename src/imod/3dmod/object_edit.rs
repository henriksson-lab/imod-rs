//! Translation of `IMOD/3dmod/object_edit.cpp` and `object_edit.h`.
//! Qt presentation is held as state; each source slot mutates the selected
//! libimod object and invokes the existing model-view refresh boundary.
#![allow(dead_code, unused_variables)]
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OFF, IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_OUT, IMOD_OBJFLAG_SCAT, Imod, Iobj,
};
use crate::imod::libimod::iobj::*;
use crate::imod::three_dmod::imodv::{
    imodv_draw, imodv_finish_chg_unit, imodv_register_object_chg,
};

pub const MIN_LABEL_SIZE: i32 = 6;
pub const MAX_SYMBOLS: usize = 4;
pub const MAX_COLOR_SELECTORS: usize = 20;
pub const SYM_TABLE: [i32; MAX_SYMBOLS] = [
    IOBJ_SYM_NONE,
    IOBJ_SYM_CIRCLE,
    IOBJ_SYM_SQUARE,
    IOBJ_SYM_TRIANGLE,
];

/// Viewer keyboard forwarding used by `ImodObjColor`'s selector slots.
pub trait ObjectEditNativeBoundary {
    fn control_key(&mut self, release: bool, key: i32);
}
/// Form-equivalent source state for the static `Ioew_dialog` and selectors.
#[derive(Clone, Debug, Default)]
pub struct ObjectEdit {
    pub dialog_open: bool,
    pub current_object: i32,
    pub copy_color_name: i32,
    pub color_selectors: [Option<ImodObjColor>; MAX_COLOR_SELECTORS],
}
/// Original `ImodObjColor` class.
#[derive(Clone, Debug, Default)]
pub struct ImodObjColor {
    pub selector_open: bool,
    pub obj_num: i32,
    pub top_ind: i32,
    pub hot_sliding: bool,
}
/// Original static `getObjectOrClose`.
pub fn get_object_or_close<'a>(model: &'a mut Imod, edit: &mut ObjectEdit) -> Option<&'a mut Iobj> {
    let object = edit.current_object;
    if object < 0 || object as usize >= model.obj.len() {
        edit.dialog_open = false;
        None
    } else {
        model.obj.get_mut(object as usize)
    }
}
/// Original static `setObjectFlag`.
pub fn set_object_flag(
    model: &mut Imod,
    edit: &mut ObjectEdit,
    state: bool,
    symflag: u8,
    flag: u32,
) {
    if let Some(obj) = get_object_or_close(model, edit) {
        if symflag != 0 {
            if state {
                obj.symflags |= flag as u8
            } else {
                obj.symflags &= !(flag as u8)
            }
        } else if state {
            obj.flags |= flag
        } else {
            obj.flags &= !flag
        };
        imodv_register_object_chg(edit.current_object);
        imodv_finish_chg_unit();
    }
}
/// Original static `setExtraItem`.
pub fn set_extra_item(model: &mut Imod, edit: &mut ObjectEdit, index: usize, value: i32) {
    if let Some(obj) = get_object_or_close(model, edit) {
        obj.extra[index] = value as u32;
        imodv_register_object_chg(edit.current_object);
        imodv_finish_chg_unit();
    }
}
/// Original `ioew_draw`.
pub fn ioew_draw(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state == 0, 0, IMOD_OBJFLAG_OFF)
}
pub fn ioew_draw_labels(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state != 0, 0, IMOD_OBJFLAG_DRAW_LABEL)
}
pub fn ioew_label_size(model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
    set_extra_item(
        model,
        edit,
        IOBJ_EX_LABEL_SIZE,
        if value < MIN_LABEL_SIZE { 0 } else { value },
    )
}
pub fn ioew_fill(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state != 0, 1, IOBJ_SYMF_FILL as u32)
}
pub fn ioew_ends(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state != 0, 1, IOBJ_SYMF_ENDS as u32)
}
pub fn ioew_arrow(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state != 0, 1, IOBJ_SYMF_ARROW as u32)
}
pub fn ioew_linewidth(model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
    if let Some(o) = get_object_or_close(model, edit) {
        o.linewidth2 = value.clamp(0, 255) as u8;
        imodv_finish_chg_unit();
    }
}
pub fn ioew_scale_for_dpi(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state != 0, 0, IMOD_OBJFLAG_SCALE_WDTH)
}
/// Original `ioew_open`.
pub fn ioew_open(model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
    if let Some(o) = get_object_or_close(model, edit) {
        match value {
            0 => o.flags &= !(IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_SCAT),
            1 => {
                o.flags |= IMOD_OBJFLAG_OPEN;
                o.flags &= !IMOD_OBJFLAG_SCAT
            }
            2 => o.flags |= IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_SCAT,
            _ => {}
        }
        imodv_finish_chg_unit();
    }
}
pub fn ioew_surface(model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
    set_object_flag(model, edit, value != 0, 0, IMOD_OBJFLAG_OUT)
}
pub fn ioew_pointsize(model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
    if let Some(o) = get_object_or_close(model, edit) {
        o.pdrawsize = value;
        imodv_finish_chg_unit();
    }
}
pub fn ioew_nametext(model: &mut Imod, edit: &mut ObjectEdit, name: &str) {
    if let Some(o) = get_object_or_close(model, edit) {
        o.name = [0; 64];
        for (d, s) in o.name.iter_mut().zip(name.bytes().take(63)) {
            *d = s;
        }
        imodv_finish_chg_unit();
    }
}
pub fn ioew_symbol(model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
    if let Some(o) = get_object_or_close(model, edit) {
        if let Some(&symbol) = SYM_TABLE.get(value.max(0) as usize) {
            o.symbol = symbol as u8;
        }
        imodv_finish_chg_unit();
    }
}
pub fn ioew_symsize(model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
    if let Some(o) = get_object_or_close(model, edit) {
        o.symsize = value.clamp(0, 255) as u8;
        imodv_finish_chg_unit();
    }
}
pub fn ioew_time(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state != 0, 0, IMOD_OBJFLAG_TIME)
}
pub fn ioew_sphere_on_sec(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state != 0, 0, IMOD_OBJFLAG_PNT_ON_SEC)
}
pub fn ioew_planar(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state != 0, 0, IMOD_OBJFLAG_PLANAR)
}
pub fn ioew_point_limit(model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
    if let Some(o) = get_object_or_close(model, edit) {
        o.extra[IOBJ_EX_PNT_LIMIT] = value as u32;
    }
}
pub fn ioew_fill_trans(model: &mut Imod, edit: &mut ObjectEdit, value: i32) {
    set_extra_item(model, edit, IOBJ_EX_2D_TRANS, value)
}
pub fn ioew_outline(model: &mut Imod, edit: &mut ObjectEdit, state: i32) {
    set_object_flag(model, edit, state != 0, 0, IMOD_OBJFLAG_POLY_CONT)
}
/// Original `ioewCopyObj`.
pub fn ioew_copy_obj(model: &mut Imod, edit: &mut ObjectEdit, value: i32) -> Result<(), String> {
    let from = value - 1;
    if from < 0 || from as usize >= model.obj.len() {
        return Err("Object number to copy to is out of range".into());
    }
    if from == edit.current_object {
        return Err("Select an object different from the current one.".into());
    }
    let src = model.obj[from as usize].clone();
    let Some(dst) = get_object_or_close(model, edit) else {
        return Ok(());
    };
    dst.flags = src.flags;
    dst.symbol = src.symbol;
    dst.symsize = src.symsize;
    dst.symflags = src.symflags;
    dst.pdrawsize = src.pdrawsize;
    dst.linewidth2 = src.linewidth2;
    for i in [IOBJ_EX_PNT_LIMIT, IOBJ_EX_2D_TRANS, IOBJ_EX_LABEL_SIZE] {
        dst.extra[i] = src.extra[i]
    }
    if edit.copy_color_name & 1 != 0 {
        dst.red = src.red;
        dst.green = src.green;
        dst.blue = src.blue
    }
    if edit.copy_color_name & 2 != 0 {
        dst.name = src.name
    }
    imodv_finish_chg_unit();
    Ok(())
}
pub fn ioew_set_copy_color_name(edit: &mut ObjectEdit, value: i32) {
    edit.copy_color_name = value
}
pub fn ioew_get_copy_color_name(edit: &ObjectEdit) -> i32 {
    edit.copy_color_name
}
/// Original `imod_object_edit`.
pub fn imod_object_edit(model: &mut Imod, edit: &mut ObjectEdit) -> i32 {
    if get_object_or_close(model, edit).is_none() {
        -1
    } else {
        edit.dialog_open = true;
        0
    }
}
/// Original `imod_object_edit_draw`.
pub fn imod_object_edit_draw(model: &mut Imod, edit: &mut ObjectEdit) -> i32 {
    if !edit.dialog_open || get_object_or_close(model, edit).is_none() {
        -1
    } else {
        0
    }
}
pub fn ioew_closing(edit: &mut ObjectEdit) {
    edit.dialog_open = false
}
pub fn ioew_quit(edit: &mut ObjectEdit) {
    edit.dialog_open = false
}
/// Original `imod_object_color`.
pub fn imod_object_color(
    model: &Imod,
    edit: &mut ObjectEdit,
    obj_num: i32,
) -> Result<usize, String> {
    if obj_num < 0 || obj_num as usize >= model.obj.len() {
        return Err("Object number is out of range".into());
    }
    if let Some(i) = edit
        .color_selectors
        .iter()
        .position(|x| x.as_ref().is_some_and(|s| s.obj_num == obj_num))
    {
        return Ok(i);
    }
    let Some(i) = edit.color_selectors.iter().position(Option::is_none) else {
        return Err("Too many object color selectors open.".into());
    };
    edit.color_selectors[i] = Some(ImodObjColor {
        selector_open: true,
        obj_num,
        top_ind: i as i32,
        hot_sliding: false,
    });
    Ok(i)
}
impl ImodObjColor {
    pub fn new_color_slot(&mut self, model: &mut Imod, red: i32, green: i32, blue: i32) {
        if let Some(o) = model.obj.get_mut(self.obj_num.max(0) as usize) {
            o.red = red.clamp(0, 255) as f32 / 255.;
            o.green = green.clamp(0, 255) as f32 / 255.;
            o.blue = blue.clamp(0, 255) as f32 / 255.;
        }
    }
    pub fn done_slot(&mut self) {
        self.selector_open = false
    }
    pub fn closing_slot(&mut self) {
        self.selector_open = false
    }
    pub fn key_press_slot(&mut self, key: i32, n: &mut dyn ObjectEditNativeBoundary) {
        n.control_key(false, key);
    }
    pub fn key_release_slot(&mut self, key: i32, n: &mut dyn ObjectEditNativeBoundary) {
        n.control_key(true, key);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Native(Vec<(bool, i32)>);
    impl ObjectEditNativeBoundary for Native {
        fn control_key(&mut self, release: bool, key: i32) {
            self.0.push((release, key));
        }
    }
    #[test]
    fn object_slots_preserve_source_flags() {
        let mut m = Imod::default();
        m.obj.push(Iobj::default());
        let mut e = ObjectEdit {
            current_object: 0,
            ..Default::default()
        };
        ioew_open(&mut m, &mut e, 2);
        assert_eq!(
            m.obj[0].flags & (IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_SCAT),
            IMOD_OBJFLAG_OPEN | IMOD_OBJFLAG_SCAT
        );
        ioew_fill(&mut m, &mut e, 1);
        assert_ne!(m.obj[0].symflags & IOBJ_SYMF_FILL as u8, 0);
    }
    #[test]
    fn color_selector_forwards_key_press_and_release() {
        let mut selector = ImodObjColor::default();
        let mut native = Native::default();
        selector.key_press_slot(65, &mut native);
        selector.key_release_slot(65, &mut native);
        assert_eq!(native.0, vec![(false, 65), (true, 65)]);
    }
}
