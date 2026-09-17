//! Translation of `IMOD/3dmod/undoredo.cpp`, `undoredo.h`, and `undoredoP.h`.
//!
//! The original owns an `ImodView *`; this Rust unit takes the corresponding
//! `Imod` explicitly.  Drawing, selection-list clearing, and Info-window
//! button updates are caller/viewer boundaries.  The undo pool and every
//! model mutation are retained here rather than delegated to a new undo API.
#![allow(dead_code)]

use crate::imod::libimod::imodel::{
    Icont, Iindex, Imod, Iobj, Ipoint, imod_move_object, imod_set_index,
};

pub const CONTOUR_DATA: i32 = 0;
pub const CONTOUR_PROPERTY: i32 = 1;
pub const CONTOUR_REMOVED: i32 = 2;
pub const CONTOUR_ADDED: i32 = 3;
pub const CONTOUR_MOVED: i32 = 4;
pub const OBJECT_CHANGED: i32 = 5;
pub const OBJECT_REMOVED: i32 = 6;
pub const OBJECT_ADDED: i32 = 7;
pub const OBJECT_MOVED: i32 = 8;
pub const MODEL_CHANGED: i32 = 9;
pub const POINTS_ADDED: i32 = 10;
pub const POINTS_REMOVED: i32 = 11;
pub const POINT_SHIFTED: i32 = 12;
pub const MODEL_SHIFTED: i32 = 13;
pub const VIEW_CHANGED: i32 = 14;
pub const NO_ERROR: i32 = 0;
pub const NONE_AVAILABLE: i32 = 1;
pub const STATE_MISMATCH: i32 = 2;
pub const MEMORY_ERROR: i32 = 3;
pub const NO_BACKUP_ITEM: i32 = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UndoState {
    pub index: Iindex,
    pub size: Iindex,
}
#[derive(Clone, Debug)]
pub struct UndoChange {
    pub type_: i32,
    pub object: i32,
    pub contour: i32,
    pub obj2_pt1: i32,
    pub cont2_pt2: i32,
    pub id: i32,
    pub bytes: usize,
}
#[derive(Clone, Debug)]
pub struct UndoUnit {
    pub before: UndoState,
    pub after: UndoState,
    pub changes: Vec<UndoChange>,
}
#[derive(Clone, Debug)]
pub enum BackupValue {
    Model(Imod),
    Object(Iobj),
    Contour(Icont),
    Points(Vec<Ipoint>),
    None,
}
#[derive(Clone, Debug)]
pub struct BackupItem {
    pub value: BackupValue,
    pub type_: i32,
    pub id: i32,
}

/// Original `UndoRedo`, with `mVi` made an explicit method argument.
#[derive(Debug)]
pub struct UndoRedo {
    pub unit_list: Vec<UndoUnit>,
    pub item_pool: Vec<BackupItem>,
    pub id: i32,
    pub reject_changes: bool,
    pub unit_open: bool,
    pub max_units: usize,
    pub max_bytes: usize,
    pub undo_index: isize,
    pub num_freed_in_pool: usize,
}

/// `UndoRedo()`: create an owned undo pool.  The native `ImodView *` is an
/// explicit model argument to mutation methods in this Rust translation.
pub fn undo_redo() -> UndoRedo {
    UndoRedo::new()
}

/// `~UndoRedo()`: discard recorded units and their owned backup payloads.
pub fn free_undo_redo(mut undo: UndoRedo) {
    undo.drop_units();
}

impl Default for UndoRedo {
    fn default() -> Self {
        Self::new()
    }
}
impl UndoRedo {
    /// `UndoRedo::UndoRedo`.
    pub fn new() -> Self {
        Self {
            unit_list: Vec::new(),
            item_pool: Vec::new(),
            id: 0,
            reject_changes: false,
            unit_open: false,
            max_units: 1000,
            max_bytes: 5_000_000,
            undo_index: -1,
            num_freed_in_pool: 0,
        }
    }
    /// `UndoRedo::~UndoRedo`.
    pub fn drop_units(&mut self) {
        self.remove_units(0, self.unit_list.len() as isize - 1);
        self.item_pool.clear();
    }
    pub fn point_change(
        &mut self,
        model: &mut Imod,
        type_: i32,
        mut object: i32,
        mut contour: i32,
        mut point: i32,
        mut point2: i32,
    ) {
        if object < -1 {
            object = model.cindex.object;
        }
        if contour < -1 {
            contour = model.cindex.contour;
        }
        if point < -1 {
            point = model.cindex.point;
        }
        if point2 < 0 {
            point2 = point;
        }
        let Some(cont) = self.contour(model, object, contour) else {
            self.memory_error();
            return;
        };
        if point < 0 || point2 < point || point2 as usize > cont.pts.len() {
            self.memory_error();
            return;
        }
        if (type_ == POINTS_ADDED || type_ == POINTS_REMOVED) && !cont.store.is_empty() {
            self.contour_change(model, CONTOUR_DATA, object, contour, -1, -1);
            return;
        }
        if self.get_open_unit(model).is_none() {
            return;
        }
        let mut change = self.init_change(type_, object, contour, point, point2);
        if type_ != POINTS_ADDED && self.copy_points(model, &mut change, false) != NO_ERROR {
            self.memory_error();
            return;
        }
        self.finish_change(change);
    }
    pub fn contour_change(
        &mut self,
        model: &mut Imod,
        type_: i32,
        mut object: i32,
        mut contour: i32,
        object2: i32,
        contour2: i32,
    ) {
        if object < -1 {
            object = model.cindex.object;
        }
        if contour < -1 {
            contour = model.cindex.contour;
        }
        if object < 0 || object as usize >= model.obj.len() {
            self.memory_error();
            return;
        }
        if (type_ == CONTOUR_REMOVED || type_ == CONTOUR_ADDED || type_ == CONTOUR_MOVED)
            && !model.obj[object as usize].store.is_empty()
        {
            self.object_change(model, OBJECT_CHANGED, object, -1);
        }
        if type_ == CONTOUR_MOVED
            && object2 >= 0
            && (object2 as usize) < model.obj.len()
            && !model.obj[object2 as usize].store.is_empty()
        {
            self.object_change(model, OBJECT_CHANGED, object2, -1);
        }
        if self.get_open_unit(model).is_none() {
            return;
        }
        let mut change = self.init_change(type_, object, contour, object2, contour2);
        if matches!(type_, CONTOUR_DATA | CONTOUR_PROPERTY | CONTOUR_REMOVED) {
            let Some(cont) = self.contour(model, object, contour).cloned() else {
                self.memory_error();
                return;
            };
            let saved = if type_ == CONTOUR_PROPERTY {
                Icont {
                    pts: Vec::new(),
                    sizes: Vec::new(),
                    label: None,
                    store: Vec::new(),
                    ..cont
                }
            } else {
                cont
            };
            change.bytes = self.contour_bytes(&saved);
            change.id = self.push_item(BackupValue::Contour(saved), 2);
        }
        self.finish_change(change);
    }
    pub fn all_contour_move(&mut self, model: &mut Imod, object: i32, object2: i32) {
        if self.get_open_unit(model).is_none() || object < 0 || object2 < 0 {
            return;
        }
        let source = model
            .obj
            .get(object as usize)
            .map(|o| o.cont.clone())
            .unwrap_or_default();
        let destination = model.obj.get(object2 as usize).map_or(0, |o| o.cont.len());
        for (co, cont) in source.into_iter().enumerate() {
            let mut change =
                self.init_change(CONTOUR_MOVED, object, 0, object2, (destination + co) as i32);
            change.bytes = self.contour_bytes(&cont);
            change.id = self.push_item(BackupValue::Contour(cont), 2);
            self.finish_change(change);
        }
    }
    pub fn object_change(&mut self, model: &mut Imod, type_: i32, mut object: i32, object2: i32) {
        if object < -1 {
            object = model.cindex.object;
        }
        if object < 0 || object as usize >= model.obj.len() {
            self.memory_error();
            return;
        }
        if type_ != OBJECT_CHANGED {
            self.model_change(model, MODEL_CHANGED, None);
        }
        if self.get_open_unit(model).is_none() {
            return;
        }
        let mut change = self.init_change(type_, object, -1, object2, -1);
        if type_ == OBJECT_CHANGED || type_ == OBJECT_REMOVED {
            let mut obj = model.obj[object as usize].clone();
            if type_ == OBJECT_CHANGED {
                obj.cont.clear();
                obj.mesh.clear();
            }
            change.bytes = self.object_bytes(&obj);
            change.id = self.push_item(BackupValue::Object(obj), 1);
        }
        self.finish_change(change);
    }
    pub fn model_change(&mut self, model: &mut Imod, type_: i32, point: Option<Ipoint>) {
        if type_ == VIEW_CHANGED {
            for ob in 0..model.obj.len() {
                self.object_change(model, OBJECT_CHANGED, ob as i32, -1);
            }
        }
        if self.get_open_unit(model).is_none() {
            return;
        }
        let mut change = self.init_change(type_, -1, -1, -1, -1);
        if type_ == MODEL_SHIFTED {
            let Some(point) = point else {
                self.memory_error();
                return;
            };
            change.bytes = core::mem::size_of::<Ipoint>();
            change.id = self.push_item(BackupValue::Points(vec![point]), 3);
        } else {
            let mut saved = model.clone();
            saved.obj.clear();
            change.bytes = self.model_bytes(&saved);
            change.id = self.push_item(BackupValue::Model(saved), 0);
        }
        self.finish_change(change);
    }
    pub fn init_change(
        &self,
        type_: i32,
        object: i32,
        contour: i32,
        object2: i32,
        contour2: i32,
    ) -> UndoChange {
        UndoChange {
            type_,
            object,
            contour,
            obj2_pt1: object2,
            cont2_pt2: contour2,
            id: -1,
            bytes: 0,
        }
    }
    pub fn finish_change(&mut self, change: UndoChange) {
        if let Some(unit) = self.unit_list.last_mut() {
            unit.changes.push(change);
        }
    }
    pub fn finish_unit(&mut self, model: &Imod) {
        if !self.unit_open {
            return;
        };
        let after = self.record_state(model);
        if let Some(unit) = self.unit_list.last_mut() {
            unit.after = after;
        }
        self.unit_open = false;
        self.reject_changes = false;
        self.id %= 2_000_000_000;
        self.trim_lists(1);
        if self.num_freed_in_pool > self.item_pool.len() / 8 {
            self.compact_pool();
        }
    }
    pub fn flush_unit(&mut self) {
        if !self.unit_list.is_empty() {
            let end = self.unit_list.len() as isize - 1;
            self.remove_units(end, end);
        }
        self.unit_open = false;
        self.reject_changes = false;
    }
    pub fn clear_units(&mut self) {
        self.remove_units(0, self.unit_list.len() as isize - 1);
        self.unit_open = false;
        self.reject_changes = false;
    }
    pub fn get_open_unit(&mut self, model: &Imod) -> Option<&mut UndoUnit> {
        if self.undo_index >= 0 {
            self.remove_units(self.undo_index, self.unit_list.len() as isize - 1);
            self.undo_index = -1;
        }
        if self.reject_changes {
            return None;
        }
        if !self.unit_open {
            self.unit_list.push(UndoUnit {
                before: self.record_state(model),
                after: UndoState::default(),
                changes: Vec::new(),
            });
            self.unit_open = true;
        }
        self.unit_list.last_mut()
    }
    pub fn memory_error(&mut self) {
        self.reject_changes = self.unit_open;
        self.remove_units(0, self.unit_list.len() as isize - 1);
        self.undo_index = -1;
    }
    pub fn undo(&mut self, model: &mut Imod) -> i32 {
        self.finish_unit(model);
        if self.undo_index == 0 || self.unit_list.is_empty() {
            return NONE_AVAILABLE;
        }
        if self.undo_index < 0 {
            self.undo_index = self.unit_list.len() as isize;
        }
        let pos = self.undo_index as usize - 1;
        if !self.state_matches(model, self.unit_list[pos].after) {
            return STATE_MISMATCH;
        }
        self.undo_index -= 1;
        for change_index in (0..self.unit_list[pos].changes.len()).rev() {
            let mut change = self.unit_list[pos].changes[change_index].clone();
            let err = self.apply_change(model, &mut change, false);
            if err != NO_ERROR {
                self.memory_error();
                return err;
            }
            self.unit_list[pos].changes[change_index] = change;
        }
        let index = self.unit_list[pos].before.index;
        self.finish_undo_redo(model, index);
        NO_ERROR
    }
    pub fn redo(&mut self, model: &mut Imod) -> i32 {
        if self.undo_index < 0 || self.undo_index as usize >= self.unit_list.len() {
            return NONE_AVAILABLE;
        }
        let pos = self.undo_index as usize;
        if !self.state_matches(model, self.unit_list[pos].before) {
            return STATE_MISMATCH;
        }
        for change_index in 0..self.unit_list[pos].changes.len() {
            let mut change = self.unit_list[pos].changes[change_index].clone();
            let err = self.apply_change(model, &mut change, true);
            if err != NO_ERROR {
                self.memory_error();
                return err;
            }
            self.unit_list[pos].changes[change_index] = change;
        }
        let index = self.unit_list[pos].after.index;
        self.undo_index += 1;
        self.finish_undo_redo(model, index);
        NO_ERROR
    }
    pub fn finish_undo_redo(&mut self, model: &mut Imod, index: Iindex) {
        imod_set_index(model, index.object, index.contour, index.point);
        self.id %= 2_000_000_000;
        if self.num_freed_in_pool > self.item_pool.len() / 8 {
            self.compact_pool();
        }
    }
    pub fn copy_points(&mut self, model: &mut Imod, change: &mut UndoChange, remove: bool) -> i32 {
        let Some(cont) = self.contour_mut(model, change.object, change.contour) else {
            return MEMORY_ERROR;
        };
        let start = change.obj2_pt1.max(0) as usize;
        let end = change.cont2_pt2 as usize;
        if end >= cont.pts.len() || start > end {
            return MEMORY_ERROR;
        }
        let points = cont.pts[start..=end].to_vec();
        change.bytes = points.len() * core::mem::size_of::<Ipoint>();
        if remove {
            cont.pts.drain(start..=end);
            if cont.sizes.len() > start {
                cont.sizes.drain(start..=end.min(cont.sizes.len() - 1));
            }
        }
        change.id = self.push_item(BackupValue::Points(points), 3);
        NO_ERROR
    }
    pub fn restore_points(&mut self, model: &mut Imod, change: &mut UndoChange) -> i32 {
        let Some(item) = self.find_pool_item(change.id).cloned() else {
            return NO_BACKUP_ITEM;
        };
        let BackupValue::Points(points) = item.value else {
            return NO_BACKUP_ITEM;
        };
        let Some(cont) = self.contour_mut(model, change.object, change.contour) else {
            return MEMORY_ERROR;
        };
        let at = change.obj2_pt1.clamp(0, cont.pts.len() as i32) as usize;
        cont.pts.splice(at..at, points);
        self.free_pool_item(change.id);
        change.id = -1;
        change.bytes = 0;
        NO_ERROR
    }
    pub fn shift_point(&mut self, model: &mut Imod, change: &UndoChange) -> i32 {
        let Some(pool_index) = self.item_pool.iter().position(|item| item.id == change.id) else {
            return NO_BACKUP_ITEM;
        };
        let BackupValue::Points(points) = &mut self.item_pool[pool_index].value else {
            return NO_BACKUP_ITEM;
        };
        let Some(point) = points.first_mut() else {
            return NO_BACKUP_ITEM;
        };
        let Some(cont) = model
            .obj
            .get_mut(change.object as usize)
            .and_then(|o| o.cont.get_mut(change.contour as usize))
        else {
            return MEMORY_ERROR;
        };
        let at = change.obj2_pt1 as usize;
        if at >= cont.pts.len() {
            return MEMORY_ERROR;
        };
        core::mem::swap(point, &mut cont.pts[at]);
        NO_ERROR
    }
    pub fn exchange_contours(
        &mut self,
        model: &mut Imod,
        change: &mut UndoChange,
        struct_only: bool,
    ) -> i32 {
        let Some(pool_index) = self.item_pool.iter().position(|item| item.id == change.id) else {
            return NO_BACKUP_ITEM;
        };
        let BackupValue::Contour(saved) = &mut self.item_pool[pool_index].value else {
            return NO_BACKUP_ITEM;
        };
        let Some(cont) = model
            .obj
            .get_mut(change.object as usize)
            .and_then(|o| o.cont.get_mut(change.contour as usize))
        else {
            return MEMORY_ERROR;
        };
        let mut replacement = saved.clone();
        core::mem::swap(saved, cont);
        if struct_only {
            replacement.pts = cont.pts.clone();
            replacement.sizes = cont.sizes.clone();
            replacement.label = cont.label.clone();
            replacement.store = cont.store.clone();
            *cont = replacement;
        }
        change.bytes = core::mem::size_of::<Icont>()
            + saved.pts.len() * core::mem::size_of::<Ipoint>()
            + saved.sizes.len() * core::mem::size_of::<f32>();
        NO_ERROR
    }
    pub fn restore_contour(&mut self, model: &mut Imod, change: &mut UndoChange) -> i32 {
        let Some(item) = self.find_pool_item(change.id).cloned() else {
            return NO_BACKUP_ITEM;
        };
        let BackupValue::Contour(cont) = item.value else {
            return NO_BACKUP_ITEM;
        };
        let Some(obj) = model.obj.get_mut(change.object as usize) else {
            return MEMORY_ERROR;
        };
        let at = change.contour.clamp(0, obj.cont.len() as i32) as usize;
        obj.cont.insert(at, cont);
        self.free_pool_item(change.id);
        change.id = -1;
        change.bytes = 0;
        NO_ERROR
    }
    pub fn remove_contour(&mut self, model: &mut Imod, change: &mut UndoChange) -> i32 {
        let Some(obj) = model.obj.get_mut(change.object as usize) else {
            return MEMORY_ERROR;
        };
        if change.contour < 0 || change.contour as usize >= obj.cont.len() {
            return MEMORY_ERROR;
        };
        let cont = obj.cont.remove(change.contour as usize);
        change.bytes = self.contour_bytes(&cont);
        change.id = self.push_item(BackupValue::Contour(cont), 2);
        NO_ERROR
    }
    pub fn move_contour(
        &mut self,
        model: &mut Imod,
        ob_from: i32,
        co_from: i32,
        ob_to: i32,
        co_to: i32,
    ) -> i32 {
        if ob_from < 0 || ob_to < 0 || co_from < 0 {
            return MEMORY_ERROR;
        }
        let cont = {
            let Some(obj) = model.obj.get_mut(ob_from as usize) else {
                return MEMORY_ERROR;
            };
            if co_from as usize >= obj.cont.len() {
                return MEMORY_ERROR;
            };
            obj.cont.remove(co_from as usize)
        };
        let Some(target) = model.obj.get_mut(ob_to as usize) else {
            return MEMORY_ERROR;
        };
        target
            .cont
            .insert(co_to.clamp(0, target.cont.len() as i32) as usize, cont);
        NO_ERROR
    }
    pub fn exchange_objects(&mut self, model: &mut Imod, change: &mut UndoChange) -> i32 {
        let Some(pool_index) = self.item_pool.iter().position(|item| item.id == change.id) else {
            return NO_BACKUP_ITEM;
        };
        let BackupValue::Object(saved) = &mut self.item_pool[pool_index].value else {
            return NO_BACKUP_ITEM;
        };
        let Some(obj) = model.obj.get_mut(change.object as usize) else {
            return MEMORY_ERROR;
        };
        core::mem::swap(saved, obj);
        change.bytes = core::mem::size_of::<Iobj>()
            + saved
                .cont
                .iter()
                .map(|cont| {
                    core::mem::size_of::<Icont>()
                        + cont.pts.len() * core::mem::size_of::<Ipoint>()
                        + cont.sizes.len() * core::mem::size_of::<f32>()
                })
                .sum::<usize>();
        NO_ERROR
    }
    pub fn restore_object(&mut self, model: &mut Imod, change: &mut UndoChange) -> i32 {
        let Some(item) = self.find_pool_item(change.id).cloned() else {
            return NO_BACKUP_ITEM;
        };
        let BackupValue::Object(obj) = item.value else {
            return NO_BACKUP_ITEM;
        };
        let at = change.object.clamp(0, model.obj.len() as i32) as usize;
        model.obj.insert(at, obj);
        self.free_pool_item(change.id);
        change.id = -1;
        change.bytes = 0;
        NO_ERROR
    }
    pub fn remove_object(&mut self, model: &mut Imod, change: &mut UndoChange) -> i32 {
        if change.object < 0 || change.object as usize >= model.obj.len() {
            return MEMORY_ERROR;
        };
        let obj = model.obj.remove(change.object as usize);
        change.bytes = self.object_bytes(&obj);
        change.id = self.push_item(BackupValue::Object(obj), 1);
        NO_ERROR
    }
    pub fn exchange_models(&mut self, model: &mut Imod, change: &mut UndoChange) -> i32 {
        let Some(pool_index) = self.item_pool.iter().position(|item| item.id == change.id) else {
            return NO_BACKUP_ITEM;
        };
        let BackupValue::Model(saved) = &mut self.item_pool[pool_index].value else {
            return NO_BACKUP_ITEM;
        };
        let mut kept_obj = core::mem::take(&mut model.obj);
        core::mem::swap(model, saved);
        model.obj = core::mem::take(&mut kept_obj);
        change.bytes = core::mem::size_of::<Imod>()
            + saved.view.iter().map(core::mem::size_of_val).sum::<usize>();
        NO_ERROR
    }
    pub fn shift_model(&mut self, model: &mut Imod, change: &UndoChange, dir: i32) -> i32 {
        let Some(item) = self.find_pool_item(change.id) else {
            return NO_BACKUP_ITEM;
        };
        let BackupValue::Points(points) = &item.value else {
            return NO_BACKUP_ITEM;
        };
        let Some(mut p) = points.first().copied() else {
            return NO_BACKUP_ITEM;
        };
        if dir < 0 {
            p.x = -p.x;
            p.y = -p.y;
            p.z = -p.z;
        }
        for obj in &mut model.obj {
            for cont in &mut obj.cont {
                for point in &mut cont.pts {
                    point.x += p.x;
                    point.y += p.y;
                    point.z += p.z;
                }
            }
        }
        NO_ERROR
    }
    pub fn update_buttons(&self) -> (bool, bool) {
        (
            self.undo_index != 0 && !self.unit_list.is_empty(),
            self.undo_index >= 0 && (self.undo_index as usize) < self.unit_list.len(),
        )
    }
    pub fn contour_bytes(&self, cont: &Icont) -> usize {
        core::mem::size_of::<Icont>()
            + cont.pts.len() * core::mem::size_of::<Ipoint>()
            + cont.sizes.len() * core::mem::size_of::<f32>()
            + cont.store.len() * core::mem::size_of_val(&cont.store[0..0])
    }
    pub fn label_bytes(&self, _label: &Option<crate::imod::libimod::ilabel::Ilabel>) -> usize {
        0
    }
    pub fn object_bytes(&self, obj: &Iobj) -> usize {
        core::mem::size_of::<Iobj>()
            + obj
                .cont
                .iter()
                .map(|c| self.contour_bytes(c))
                .sum::<usize>()
            + obj
                .mesh
                .iter()
                .map(|m| {
                    core::mem::size_of_val(m)
                        + m.vert.len() * core::mem::size_of::<Ipoint>()
                        + m.list.len() * core::mem::size_of::<i32>()
                })
                .sum::<usize>()
    }
    pub fn model_bytes(&self, model: &Imod) -> usize {
        core::mem::size_of::<Imod>() + model.view.iter().map(core::mem::size_of_val).sum::<usize>()
    }
    pub fn record_state(&self, model: &Imod) -> UndoState {
        let mut size = Iindex {
            object: model.obj.len() as i32,
            contour: -1,
            point: -1,
        };
        let index = model.cindex;
        if let Some(obj) = model.obj.get(index.object.max(0) as usize) {
            size.contour = obj.cont.len() as i32;
            if let Some(cont) = obj.cont.get(index.contour.max(0) as usize) {
                size.point = cont.pts.len() as i32;
            }
        }
        UndoState { index, size }
    }
    pub fn state_matches(&self, model: &Imod, state: UndoState) -> bool {
        let now = self.record_state(model);
        now.size.object == state.size.object
            && (state.index.object < 0 || now.size.contour == state.size.contour)
            && (state.index.contour < 0 || now.size.point == state.size.point)
    }
    pub fn free_pool_item(&mut self, id: i32) {
        if let Some(item) = self.find_pool_item_mut(id) {
            if item.id >= 0 {
                item.value = BackupValue::None;
                item.id = -1;
                self.num_freed_in_pool += 1;
            }
        }
    }
    pub fn trim_lists(&mut self, num_keep: usize) {
        let mut total = 0;
        let mut keep = 0;
        for i in (0..self.unit_list.len()).rev() {
            let unit = &self.unit_list[i];
            total += core::mem::size_of::<UndoUnit>()
                + unit
                    .changes
                    .iter()
                    .map(|c| core::mem::size_of::<UndoChange>() + c.bytes)
                    .sum::<usize>();
            keep += 1;
            if (keep > num_keep && total > self.max_bytes) || keep > self.max_units {
                self.remove_units(0, i as isize);
                break;
            }
        }
    }
    pub fn remove_units(&mut self, start: isize, end: isize) {
        if start < 0 || end < start {
            return;
        }
        let end = end.min(self.unit_list.len() as isize - 1);
        let removed: Vec<_> = self
            .unit_list
            .drain(start as usize..=end as usize)
            .collect();
        for unit in removed {
            for change in unit.changes {
                self.free_pool_item(change.id);
            }
        }
        self.compact_pool();
    }
    pub fn compact_pool(&mut self) {
        self.item_pool.retain(|item| item.id >= 0);
        self.num_freed_in_pool = 0;
    }
    pub fn find_pool_item(&self, id: i32) -> Option<&BackupItem> {
        self.item_pool.iter().find(|item| item.id == id)
    }
    fn find_pool_item_mut(&mut self, id: i32) -> Option<&mut BackupItem> {
        self.item_pool.iter_mut().find(|item| item.id == id)
    }
    fn push_item(&mut self, value: BackupValue, type_: i32) -> i32 {
        let id = self.id;
        self.id += 1;
        self.item_pool.push(BackupItem { value, type_, id });
        id
    }
    fn contour<'a>(&self, model: &'a Imod, object: i32, contour: i32) -> Option<&'a Icont> {
        model.obj.get(object as usize)?.cont.get(contour as usize)
    }
    fn contour_mut<'a>(
        &self,
        model: &'a mut Imod,
        object: i32,
        contour: i32,
    ) -> Option<&'a mut Icont> {
        model
            .obj
            .get_mut(object as usize)?
            .cont
            .get_mut(contour as usize)
    }
    fn apply_change(&mut self, model: &mut Imod, change: &mut UndoChange, redo: bool) -> i32 {
        match (change.type_, redo) {
            (POINTS_ADDED, false) | (POINTS_REMOVED, true) => self.copy_points(model, change, true),
            (POINTS_REMOVED, false) | (POINTS_ADDED, true) => self.restore_points(model, change),
            (POINT_SHIFTED, _) => self.shift_point(model, &change),
            (CONTOUR_DATA, _) => self.exchange_contours(model, change, false),
            (CONTOUR_PROPERTY, _) => self.exchange_contours(model, change, true),
            (CONTOUR_REMOVED, false) | (CONTOUR_ADDED, true) => self.restore_contour(model, change),
            (CONTOUR_ADDED, false) | (CONTOUR_REMOVED, true) => self.remove_contour(model, change),
            (CONTOUR_MOVED, false) => self.move_contour(
                model,
                change.obj2_pt1,
                change.cont2_pt2,
                change.object,
                change.contour,
            ),
            (CONTOUR_MOVED, true) => self.move_contour(
                model,
                change.object,
                change.contour,
                change.obj2_pt1,
                change.cont2_pt2,
            ),
            (OBJECT_CHANGED, _) => self.exchange_objects(model, change),
            (OBJECT_REMOVED, false) | (OBJECT_ADDED, true) => self.restore_object(model, change),
            (OBJECT_ADDED, false) | (OBJECT_REMOVED, true) => self.remove_object(model, change),
            (OBJECT_MOVED, _) => {
                let (from, to) = if redo {
                    (change.object, change.obj2_pt1)
                } else {
                    (change.obj2_pt1, change.object)
                };
                if imod_move_object(model, from, to) != 0 {
                    MEMORY_ERROR
                } else {
                    NO_ERROR
                }
            }
            (MODEL_CHANGED | VIEW_CHANGED, _) => self.exchange_models(model, change),
            (MODEL_SHIFTED, _) => self.shift_model(model, &change, if redo { 1 } else { -1 }),
            _ => NO_ERROR,
        }
    }
}

// Exported `undo*` calls from undoredo.cpp.  `ImodView::undo` and
// `ImodView::imod` become the two explicit Rust arguments.
pub fn undo_point_change(u: &mut UndoRedo, m: &mut Imod, t: i32, o: i32, c: i32, p: i32, p2: i32) {
    u.point_change(m, t, o, c, p, p2)
}
pub fn undo_contour_change(
    u: &mut UndoRedo,
    m: &mut Imod,
    t: i32,
    o: i32,
    c: i32,
    o2: i32,
    c2: i32,
) {
    u.contour_change(m, t, o, c, o2, c2)
}
pub fn undo_object_change(u: &mut UndoRedo, m: &mut Imod, t: i32, o: i32, o2: i32) {
    u.object_change(m, t, o, o2)
}
pub fn undo_model_change(u: &mut UndoRedo, m: &mut Imod, t: i32, p: Option<Ipoint>) {
    u.model_change(m, t, p)
}
pub fn undo_point_shift_cp(u: &mut UndoRedo, m: &mut Imod) {
    u.point_change(m, POINT_SHIFTED, -2, -2, -2, -1)
}
pub fn undo_point_shift(u: &mut UndoRedo, m: &mut Imod, p: i32) {
    u.point_change(m, POINT_SHIFTED, -2, -2, p, -1)
}
pub fn undo_point_addition_cc(u: &mut UndoRedo, m: &mut Imod, p: i32) {
    u.point_change(m, POINTS_ADDED, -2, -2, p, -1)
}
pub fn undo_point_addition_cc2(u: &mut UndoRedo, m: &mut Imod, p: i32, p2: i32) {
    u.point_change(m, POINTS_ADDED, -2, -2, p, p2)
}
pub fn undo_point_addition(u: &mut UndoRedo, m: &mut Imod, o: i32, c: i32, p: i32) {
    u.point_change(m, POINTS_ADDED, o, c, p, -1)
}
pub fn undo_point_removal_cp(u: &mut UndoRedo, m: &mut Imod) {
    u.point_change(m, POINTS_REMOVED, -2, -2, -2, -1)
}
pub fn undo_point_removal(u: &mut UndoRedo, m: &mut Imod, p: i32) {
    u.point_change(m, POINTS_REMOVED, -2, -2, p, -1)
}
pub fn undo_point_removal2(u: &mut UndoRedo, m: &mut Imod, p: i32, p2: i32) {
    u.point_change(m, POINTS_REMOVED, -2, -2, p, p2)
}
pub fn undo_contour_data_chg_cc(u: &mut UndoRedo, m: &mut Imod) {
    u.contour_change(m, CONTOUR_DATA, -2, -2, -1, -1)
}
pub fn undo_contour_data_chg(u: &mut UndoRedo, m: &mut Imod, o: i32, c: i32) {
    u.contour_change(m, CONTOUR_DATA, o, c, -1, -1)
}
pub fn undo_contour_prop_chg_cc(u: &mut UndoRedo, m: &mut Imod) {
    u.contour_change(m, CONTOUR_PROPERTY, -2, -2, -1, -1)
}
pub fn undo_contour_prop_chg(u: &mut UndoRedo, m: &mut Imod, o: i32, c: i32) {
    u.contour_change(m, CONTOUR_PROPERTY, o, c, -1, -1)
}
pub fn undo_contour_removal_cc(u: &mut UndoRedo, m: &mut Imod) {
    u.contour_change(m, CONTOUR_REMOVED, -2, -2, -1, -1)
}
pub fn undo_contour_removal_co(u: &mut UndoRedo, m: &mut Imod, c: i32) {
    u.contour_change(m, CONTOUR_REMOVED, -2, c, -1, -1)
}
pub fn undo_contour_removal(u: &mut UndoRedo, m: &mut Imod, o: i32, c: i32) {
    u.contour_change(m, CONTOUR_REMOVED, o, c, -1, -1)
}
pub fn undo_contour_addition_co(u: &mut UndoRedo, m: &mut Imod, c: i32) {
    u.contour_change(m, CONTOUR_ADDED, -2, c, -1, -1)
}
pub fn undo_contour_addition(u: &mut UndoRedo, m: &mut Imod, o: i32, c: i32) {
    u.contour_change(m, CONTOUR_ADDED, o, c, -1, -1)
}
pub fn undo_contour_move(u: &mut UndoRedo, m: &mut Imod, o: i32, c: i32, o2: i32, c2: i32) {
    u.contour_change(m, CONTOUR_MOVED, o, c, o2, c2)
}
pub fn undo_object_prop_chg_co(u: &mut UndoRedo, m: &mut Imod) {
    u.object_change(m, OBJECT_CHANGED, -2, -1)
}
pub fn undo_object_prop_chg(u: &mut UndoRedo, m: &mut Imod, o: i32) {
    u.object_change(m, OBJECT_CHANGED, o, -1)
}
pub fn undo_object_removal_co(u: &mut UndoRedo, m: &mut Imod) {
    u.object_change(m, OBJECT_REMOVED, -2, -1)
}
pub fn undo_object_removal(u: &mut UndoRedo, m: &mut Imod, o: i32) {
    u.object_change(m, OBJECT_REMOVED, o, -1)
}
pub fn undo_object_addition(u: &mut UndoRedo, m: &mut Imod, o: i32) {
    u.object_change(m, OBJECT_ADDED, o, -1)
}
pub fn undo_object_move(u: &mut UndoRedo, m: &mut Imod, o: i32, o2: i32) {
    u.object_change(m, OBJECT_MOVED, o, o2)
}
pub fn undo_model_shift(u: &mut UndoRedo, m: &mut Imod, p: Ipoint) {
    u.model_change(m, MODEL_SHIFTED, Some(p))
}
pub fn undo_finish_unit(u: &mut UndoRedo, m: &Imod) {
    u.finish_unit(m)
}
pub fn undo_flush_unit(u: &mut UndoRedo) {
    u.flush_unit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_facades_create_and_release_empty_pool() {
        let undo = undo_redo();
        assert_eq!(undo.max_units, 1000);
        assert_eq!(undo.max_bytes, 5_000_000);
        free_undo_redo(undo);
    }
    fn model() -> Imod {
        let mut m = Imod::default();
        m.obj.push(Iobj {
            cont: vec![Icont {
                pts: vec![Ipoint {
                    x: 1.,
                    y: 2.,
                    z: 3.,
                }],
                ..Default::default()
            }],
            ..Default::default()
        });
        m.cindex = Iindex {
            object: 0,
            contour: 0,
            point: 0,
        };
        m
    }
    #[test]
    fn point_shift_round_trips() {
        let mut m = model();
        let mut u = UndoRedo::new();
        u.point_change(&mut m, POINT_SHIFTED, -2, -2, -2, -1);
        m.obj[0].cont[0].pts[0].x = 8.;
        u.finish_unit(&m);
        assert_eq!(u.undo(&mut m), NO_ERROR);
        assert_eq!(m.obj[0].cont[0].pts[0].x, 1.);
        assert_eq!(u.redo(&mut m), NO_ERROR);
        assert_eq!(m.obj[0].cont[0].pts[0].x, 8.);
    }
    #[test]
    fn added_points_undo_redo() {
        let mut m = model();
        let mut u = UndoRedo::new();
        u.point_change(&mut m, POINTS_ADDED, 0, 0, 1, 1);
        m.obj[0].cont[0].pts.push(Ipoint::default());
        u.finish_unit(&m);
        assert_eq!(u.undo(&mut m), NO_ERROR);
        assert_eq!(m.obj[0].cont[0].pts.len(), 1);
        assert_eq!(u.redo(&mut m), NO_ERROR);
        assert_eq!(m.obj[0].cont[0].pts.len(), 2);
    }
}
