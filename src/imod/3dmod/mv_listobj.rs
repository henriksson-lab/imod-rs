//! Translation of `IMOD/3dmod/mv_listobj.cpp` and `mv_listobj.h`.
//! The source's two checkbox arrays become exact boolean/object-label arrays;
//! actions retain group and on/off mutations over the active model.
#![allow(dead_code, unused_variables)]
use crate::imod::libimod::imodel::{IMOD_OBJFLAG_OFF, Imod, Iobj_group};
use crate::imod::libimod::objgroup::{obj_group_append, obj_group_lookup};
use crate::imod::three_dmod::imodv::{
    imodv_finish_chg_unit, imodv_register_model_chg, imodv_register_object_chg,
};
pub const OBJLIST_NUMBUTTONS: usize = 9;
pub const MAX_OOLIST_BUTTONS: usize = 5000;
pub const MAX_LIST_IN_COL: usize = 36;
pub const MAX_LIST_NAME: usize = 40;
pub const OBJGRP_NEW: i32 = 0;
pub const OBJGRP_DELETE: i32 = 1;
pub const OBJGRP_CLEAR: i32 = 2;
pub const OBJGRP_ADDALL: i32 = 3;
pub const OBJGRP_SWAP: i32 = 4;
pub const OBJGRP_TURNON: i32 = 5;
pub const OBJGRP_TURNOFF: i32 = 6;
pub const OBJGRP_OTHERSON: i32 = 7;
pub const OBJGRP_OTHERSOFF: i32 = 8;
pub const OBJLIST_BUTTON_LABELS: [&str; OBJLIST_NUMBUTTONS] = [
    "New", "Delete", "Clear", "Add All", "Swap", "ON", "OFF", "On", "Off",
];

/// Qt/docking and IMODV input operations that stay at the native boundary of
/// `mv_listobj.cpp`.
pub trait ImodvOlistNativeBoundary {
    fn set_focus(&mut self);
    fn info_input(&mut self);
    fn adjust_frame_size(&mut self);
    fn rounded_style(&mut self) -> bool;
    fn button_width(&mut self, rounded: bool, factor: f32, text: &str) -> i32;
    fn set_button_width(&mut self, button: usize, width: i32);
    fn widget_change_event(&mut self);
    fn check_and_set_mac_menu(&mut self);
    fn remove_dialog(&mut self);
    fn accept_close(&mut self);
    fn close_window(&mut self);
    fn key_press(&mut self);
    fn key_release(&mut self);
}
/// Original `ImodvOlist` widget and static button/group variables.
#[derive(Clone, Debug, Default)]
pub struct ImodvOlist {
    pub dialog_open: bool,
    pub object_checked: Vec<bool>,
    pub group_checked: Vec<bool>,
    pub object_labels: Vec<String>,
    pub current_group: i32,
    pub grouping: bool,
    pub num_per_col: usize,
    pub shift_pressed: bool,
    pub last_was_group: bool,
    pub last_but_toggled: i32,
    pub group_name: String,
}
/// Original `imodvObjectListDialog`.
pub fn imodv_object_list_dialog(model: &Imod, state: i32, list: &mut ImodvOlist) {
    if state == 0 {
        list.dialog_open = false;
        return;
    }
    list.dialog_open = true;
    let n = model.obj.len().min(MAX_OOLIST_BUTTONS);
    list.object_checked.resize(n, false);
    list.group_checked.resize(n, false);
    list.object_labels.resize(n, String::new());
    list.num_per_col = n.min(MAX_LIST_IN_COL).max(1);
    list.grouping = false;
    imodv_olist_update_on_offs(model, list);
}
pub fn imodv_olist_set_checked(model: &Imod, list: &mut ImodvOlist, ob: i32, state: bool) {
    if list.dialog_open
        && ob >= 0
        && (ob as usize) < list.object_checked.len()
        && (ob as usize) < model.obj.len()
    {
        list.object_checked[ob as usize] = state
    }
}
pub fn imodv_olist_set_color(model: &Imod, list: &mut ImodvOlist, ob: i32) {
    if list.dialog_open
        && ob >= 0
        && (ob as usize) < model.obj.len()
        && ob as usize >= list.object_labels.len()
    {
        list.object_labels.resize(ob as usize + 1, String::new())
    }
}
pub fn imodv_olist_update_groups(model: &Imod, list: &mut ImodvOlist) {
    if list.dialog_open {
        list.update_groups(model)
    }
}
/// Original `imodvOlistUpdateOnOffs`.
pub fn imodv_olist_update_on_offs(model: &Imod, list: &mut ImodvOlist) {
    if !list.dialog_open {
        return;
    }
    let n = model.obj.len().min(MAX_OOLIST_BUTTONS);
    list.object_checked.resize(n, false);
    list.group_checked.resize(n, false);
    list.object_labels.resize(n, String::new());
    for (ob, obj) in model.obj.iter().take(n).enumerate() {
        list.object_checked[ob] = obj.flags & IMOD_OBJFLAG_OFF == 0;
        let name: String = obj
            .name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8 as char)
            .collect();
        list.object_labels[ob] = format!(
            "{}: {}",
            ob + 1,
            name.chars().take(MAX_LIST_NAME - 1).collect::<String>()
        );
    }
    list.update_groups(model);
}
pub fn imodv_olist_obj_in_group(list: &ImodvOlist, ob: i32) -> bool {
    list.dialog_open
        && list.grouping
        && ob >= 0
        && list
            .group_checked
            .get(ob as usize)
            .copied()
            .unwrap_or(false)
}
pub fn imodv_olist_grouping(list: &ImodvOlist) -> bool {
    list.grouping
}
impl ImodvOlist {
    pub fn new() -> Self {
        Self::default()
    }
    /// Original `ImodvOlist::addOneObject`.
    pub fn add_one_object(&mut self, ob: i32) {
        if ob < 0 {
            return;
        }
        let i = ob as usize;
        if self.object_checked.len() <= i {
            self.object_checked.resize(i + 1, false);
            self.group_checked.resize(i + 1, false);
            self.object_labels.resize(i + 1, String::new())
        }
        self.object_labels[i] = format!("{}: ", ob + 1)
    }
    /// Original `ImodvOlist::toggleGroupSlot`.
    pub fn toggle_group_slot(&mut self, model: &mut Imod, ob: i32, state: bool) {
        let Some(group) = model.group_list.get_mut(self.current_group.max(0) as usize) else {
            return;
        };
        let (mut start, mut end) = (ob, ob);
        if self.shift_pressed
            && self.last_was_group
            && self.last_but_toggled >= 0
            && ob != self.last_but_toggled
            && self
                .group_checked
                .get(self.last_but_toggled as usize)
                .copied()
                == Some(state)
        {
            if ob < self.last_but_toggled {
                start = ob;
                end = self.last_but_toggled - 1
            } else {
                start = self.last_but_toggled + 1;
                end = ob
            }
        }
        for ind in start..=end {
            if ind < 0 {
                continue;
            }
            let present = obj_group_lookup(group, ind) >= 0;
            if state && !present {
                obj_group_append(group, ind);
            } else if !state && present {
                group.obj_list.retain(|&x| x != ind)
            }
            if let Some(check) = self.group_checked.get_mut(ind as usize) {
                *check = state
            }
        }
        self.last_was_group = true;
        self.last_but_toggled = ob;
    }
    /// Original `ImodvOlist::toggleListSlot`.
    pub fn toggle_list_slot(&mut self, model: &mut Imod, ob: i32, state: bool) {
        let (mut start, mut end) = (-1, -1);
        if self.shift_pressed
            && !self.last_was_group
            && self.last_but_toggled >= 0
            && ob != self.last_but_toggled
            && self
                .object_checked
                .get(self.last_but_toggled as usize)
                .copied()
                == Some(state)
        {
            start = ob.min(self.last_but_toggled) + 1;
            end = ob.max(self.last_but_toggled) - 1;
            for ind in start..=end {
                if let Some(obj) = model.obj.get_mut(ind.max(0) as usize) {
                    if state {
                        obj.flags &= !IMOD_OBJFLAG_OFF;
                    } else {
                        obj.flags |= IMOD_OBJFLAG_OFF;
                    }
                    if let Some(check) = self.object_checked.get_mut(ind.max(0) as usize) {
                        *check = state;
                    }
                    imodv_register_object_chg(ind);
                }
            }
        }
        if let Some(obj) = model.obj.get_mut(ob.max(0) as usize) {
            if state {
                obj.flags &= !IMOD_OBJFLAG_OFF;
            } else {
                obj.flags |= IMOD_OBJFLAG_OFF;
            }
            if let Some(check) = self.object_checked.get_mut(ob.max(0) as usize) {
                *check = state;
            }
            imodv_register_object_chg(ob);
        }
        self.last_was_group = false;
        self.last_but_toggled = ob;
        imodv_finish_chg_unit();
    }
    /// Original `ImodvOlist::actionButtonClicked`.
    pub fn action_button_clicked(&mut self, model: &mut Imod, which: i32) {
        self.last_but_toggled = -1;
        match which {
            OBJGRP_NEW => {
                imodv_register_model_chg();
                let mut group = Iobj_group::default();
                if let Some(old) = model.group_list.get(self.current_group.max(0) as usize) {
                    group.obj_list = old.obj_list.clone();
                }
                model.group_list.push(group);
                self.current_group = model.group_list.len() as i32 - 1;
            }
            OBJGRP_DELETE => {
                if self.current_group >= 0 && (self.current_group as usize) < model.group_list.len()
                {
                    imodv_register_model_chg();
                    model.group_list.remove(self.current_group as usize);
                    self.current_group = (self.current_group - 1)
                        .max(0)
                        .min(model.group_list.len() as i32 - 1);
                }
            }
            OBJGRP_CLEAR => {
                if let Some(g) = model.group_list.get_mut(self.current_group.max(0) as usize) {
                    imodv_register_model_chg();
                    g.obj_list.clear();
                }
            }
            OBJGRP_ADDALL => {
                if let Some(g) = model.group_list.get_mut(self.current_group.max(0) as usize) {
                    imodv_register_model_chg();
                    g.obj_list = (0..model.obj.len() as i32).collect();
                }
            }
            OBJGRP_SWAP => {
                if let Some(g) = model.group_list.get_mut(self.current_group.max(0) as usize) {
                    imodv_register_model_chg();
                    let old = g.obj_list.clone();
                    g.obj_list = (0..model.obj.len() as i32)
                        .filter(|x| !old.contains(x))
                        .collect();
                }
            }
            OBJGRP_TURNON | OBJGRP_TURNOFF | OBJGRP_OTHERSON | OBJGRP_OTHERSOFF => {
                let members = model
                    .group_list
                    .get(self.current_group.max(0) as usize)
                    .map(|g| g.obj_list.clone())
                    .unwrap_or_default();
                for (ob, obj) in model.obj.iter_mut().enumerate() {
                    let member = members.contains(&(ob as i32));
                    let on =
                        (member && which == OBJGRP_TURNON) || (!member && which == OBJGRP_OTHERSON);
                    let off = (member && which == OBJGRP_TURNOFF)
                        || (!member && which == OBJGRP_OTHERSOFF);
                    if on {
                        obj.flags &= !IMOD_OBJFLAG_OFF;
                        imodv_register_object_chg(ob as i32)
                    }
                    if off {
                        obj.flags |= IMOD_OBJFLAG_OFF;
                        imodv_register_object_chg(ob as i32)
                    }
                }
            }
            _ => {}
        }
        self.update_groups(model);
        imodv_finish_chg_unit();
    }
    /// Original `ImodvOlist::nameChanged`.
    pub fn name_changed(&mut self, model: &mut Imod, name: &str) {
        self.group_name = name.to_string();
        if let Some(g) = model.group_list.get_mut(self.current_group.max(0) as usize) {
            g.name = [0; 32];
            for (dst, src) in g.name.iter_mut().zip(name.bytes().take(31)) {
                *dst = src;
            }
        }
    }
    pub fn cur_group_changed(
        &mut self,
        model: &Imod,
        value: i32,
        native: &mut dyn ImodvOlistNativeBoundary,
    ) {
        native.set_focus();
        self.current_group = value - 1;
        self.update_groups(model);
        self.last_but_toggled = -1;
    }
    pub fn return_pressed(&mut self, native: &mut dyn ImodvOlistNativeBoundary) {
        native.set_focus();
    }
    /// Original `ImodvOlist::updateGroups`.
    pub fn update_groups(&mut self, model: &Imod) {
        if self.current_group < 0 || self.current_group as usize >= model.group_list.len() {
            self.grouping = false;
            self.group_checked.fill(false);
            return;
        }
        self.grouping = true;
        let g = &model.group_list[self.current_group as usize];
        self.group_name = g
            .name
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as char)
            .collect();
        for (ob, value) in self.group_checked.iter_mut().enumerate() {
            *value = g.obj_list.contains(&(ob as i32));
        }
    }
    pub fn adjust_frame_size(&mut self, native: &mut dyn ImodvOlistNativeBoundary) {
        native.info_input();
        native.adjust_frame_size();
    }
    pub fn set_font_dependent_widths(&mut self, native: &mut dyn ImodvOlistNativeBoundary) {
        let rounded = native.rounded_style();
        let min_width = native.button_width(rounded, 1.3, OBJLIST_BUTTON_LABELS[0]);
        for (button, label) in OBJLIST_BUTTON_LABELS.iter().enumerate() {
            let width = native.button_width(rounded, 1.3, label);
            native.set_button_width(button, min_width.max(width));
        }
    }
    pub fn top_change_event(
        &mut self,
        font_change: bool,
        native: &mut dyn ImodvOlistNativeBoundary,
    ) {
        native.widget_change_event();
        native.check_and_set_mac_menu();
        if font_change {
            self.set_font_dependent_widths(native);
            self.adjust_frame_size(native);
        }
    }
    pub fn top_close_event(&mut self, native: &mut dyn ImodvOlistNativeBoundary) {
        native.remove_dialog();
        self.dialog_open = false;
        self.object_checked.clear();
        self.group_checked.clear();
        self.grouping = false;
        native.accept_close();
    }
    pub fn key_press_event(
        &mut self,
        key: i32,
        close_key: bool,
        native: &mut dyn ImodvOlistNativeBoundary,
    ) {
        if key == 0x0100_0020 {
            self.shift_pressed = true
        }
        if close_key {
            native.close_window();
        } else {
            native.key_press();
        }
    }
    pub fn key_release_event(&mut self, key: i32, native: &mut dyn ImodvOlistNativeBoundary) {
        if key == 0x0100_0020 {
            self.shift_pressed = false
        }
        native.key_release();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::Iobj;
    #[derive(Default)]
    struct Native {
        focus: i32,
        widths: Vec<(usize, i32)>,
        resize: i32,
        change: i32,
        mac_menu: i32,
    }
    impl ImodvOlistNativeBoundary for Native {
        fn set_focus(&mut self) {
            self.focus += 1;
        }
        fn info_input(&mut self) {}
        fn adjust_frame_size(&mut self) {
            self.resize += 1;
        }
        fn rounded_style(&mut self) -> bool {
            true
        }
        fn button_width(&mut self, _: bool, _: f32, text: &str) -> i32 {
            text.len() as i32
        }
        fn set_button_width(&mut self, button: usize, width: i32) {
            self.widths.push((button, width));
        }
        fn widget_change_event(&mut self) {
            self.change += 1;
        }
        fn check_and_set_mac_menu(&mut self) {
            self.mac_menu += 1;
        }
        fn remove_dialog(&mut self) {}
        fn accept_close(&mut self) {}
        fn close_window(&mut self) {}
        fn key_press(&mut self) {}
        fn key_release(&mut self) {}
    }
    #[test]
    fn groups_and_action_mutate_source_model() {
        let mut m = Imod::default();
        m.obj = vec![Iobj::default(), Iobj::default()];
        let mut l = ImodvOlist::default();
        imodv_object_list_dialog(&m, 1, &mut l);
        l.action_button_clicked(&mut m, OBJGRP_NEW);
        l.action_button_clicked(&mut m, OBJGRP_ADDALL);
        assert_eq!(m.group_list[0].obj_list, vec![0, 1]);
        l.action_button_clicked(&mut m, OBJGRP_TURNOFF);
        assert!(m.obj.iter().all(|o| o.flags & IMOD_OBJFLAG_OFF != 0));
    }

    #[test]
    fn font_change_reapplies_source_button_widths_and_frame_size() {
        let mut list = ImodvOlist::default();
        let mut native = Native::default();
        list.top_change_event(true, &mut native);
        assert_eq!(native.widths.len(), OBJLIST_NUMBUTTONS);
        assert!(native.widths.iter().all(|&(_, width)| width >= 3));
        assert_eq!((native.change, native.mac_menu, native.resize), (1, 1, 1));
    }
}
