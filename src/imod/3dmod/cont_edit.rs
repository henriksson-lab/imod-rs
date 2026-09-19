//! Translation of `IMOD/3dmod/cont_edit.cpp` together with `cont_edit.h`.
//!
//! This unit deliberately owns the file-static dialog state in [`ContourEditState`].
//! Qt/DockingDialog, display, input, undo, and vertex-buffer actions are expressed
//! by [`ContourEditNativeBoundary`]; the model transformations are performed here.
#![allow(dead_code, unused_variables)]

use crate::imod::libimod::icont::{
    imod_contour_area, imod_contour_break, imod_contour_join, imod_contour_splice,
    imodel_contour_check_wild, imodel_unused_surface,
};
use crate::imod::libimod::ilabel::{
    imod_label_item_add, imod_label_item_get, imod_label_name, imod_label_name_get, imod_label_new,
};
use crate::imod::libimod::imodel::{
    ICONT_OPEN, Icont, Iindex, Imesh, Imod, Iobj, Ipoint, imod_get_cur_mesh_surf,
};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_TIME, imod_object_add_contour, imod_object_clean_surf, imod_object_remove_contour,
    iobj_open, iobj_scat,
};
use crate::imod::libimod::ipoint::{imod_point_delete, imod_point_get_size, imod_point_set_size};

pub const MAX_POINT_SIZE: i32 = 999;
pub const OPEN_TYPE_CONCAT: i32 = 0;
pub const OPEN_TYPE_SPLICE: i32 = 1;
pub const CLOSED_TYPE_CONCAT: i32 = 0;
pub const CLOSED_TYPE_NEAREST: i32 = 1;
pub const CLOSED_TYPE_AUTO: i32 = 2;
pub const CLOSED_TYPE_SETPOINT: i32 = 3;

/// The C `nullIndex` file static.
pub const NULL_INDEX: Iindex = Iindex {
    object: -1,
    contour: -1,
    point: -1,
};

/// `contour_edit_struct`, `contour_move_struct`, `contour_join_struct`, and
/// `contour_break_struct`, minus QObject pointers that belong at the boundary.
#[derive(Clone, Debug)]
pub struct ContourEditState {
    pub surf_dialog_open: bool,
    pub break_dialog_open: bool,
    pub join_dialog_open: bool,
    pub move_dialog_open: bool,
    pub wheel_for_size: i32,
    pub whole_surf: i32,
    pub move_to_surf: i32,
    pub surf_move_to: i32,
    pub replace: i32,
    pub expand: i32,
    pub convert_all_pt: i32,
    pub enabled: i32,
    pub keep_size: i32,
    pub move_up_down: i32,
    pub up_or_down: i32,
    pub join_i1: Iindex,
    pub join_i2: Iindex,
    pub open_type: i32,
    pub closed_type: i32,
    pub break_i1: Iindex,
    pub break_i2: Iindex,
    pub use_current: bool,
    pub move_first: bool,
}

impl Default for ContourEditState {
    fn default() -> Self {
        Self {
            surf_dialog_open: false,
            break_dialog_open: false,
            join_dialog_open: false,
            move_dialog_open: false,
            wheel_for_size: 0,
            whole_surf: 0,
            move_to_surf: 0,
            surf_move_to: 0,
            replace: 0,
            expand: 0,
            convert_all_pt: 0,
            enabled: 1,
            keep_size: 0,
            move_up_down: 0,
            up_or_down: 0,
            join_i1: NULL_INDEX,
            join_i2: NULL_INDEX,
            open_type: OPEN_TYPE_CONCAT,
            closed_type: CLOSED_TYPE_AUTO,
            break_i1: Iindex {
                object: 0,
                contour: 0,
                point: 0,
            },
            break_i2: Iindex {
                object: 0,
                contour: 0,
                point: 0,
            },
            use_current: false,
            move_first: true,
        }
    }
}

/// Concrete non-model calls in `cont_edit.cpp`.
pub trait ContourEditNativeBoundary {
    fn contour_data_changed(&mut self, object: i32, contour: i32);
    fn contour_property_changed(&mut self, object: i32, contour: i32);
    fn contour_addition(&mut self, object: i32, contour: i32);
    fn contour_removal(&mut self, object: i32, contour: i32);
    fn contour_move(
        &mut self,
        old_object: i32,
        old_contour: i32,
        new_object: i32,
        new_contour: i32,
    );
    fn object_property_changed(&mut self, object: i32);
    fn point_addition(&mut self, object: i32, contour: i32, point: i32);
    fn point_removal(&mut self, object: i32, contour: i32, point: i32);
    fn finish_undo_unit(&mut self);
    fn flush_undo_unit(&mut self);
    fn draw_model(&mut self);
    fn selection_contains(&self, object: i32, contour: i32) -> bool;
    fn clear_selection(&mut self);
    fn next_z(&mut self);
    fn prev_z(&mut self);
    fn warning(&mut self, text: &str);
}

/// Input calls used by `iceSurfGoto`, `iceContInSurf`, and `iceSurfNew`.
pub trait ContourEditInputBoundary: ContourEditNativeBoundary {
    fn goto_surface(&mut self, target: i32);
    fn adjacent_contour_in_surface(&mut self, direction: i32);
    fn new_surface(&mut self);
}

/// Values supplied to the translated `ContSurfPoint::set*` methods by
/// `imodContEditSurfShow`.
#[derive(Clone, Debug, PartialEq)]
pub struct ContourSurfacePointDisplay {
    pub surface: i32,
    pub surface_max: i32,
    /// Object-label item for the displayed surface, if any.
    pub surface_label: Option<String>,
    /// Contour-label name, if the current contour has one.
    pub contour_label: Option<String>,
    /// Contour-label item for the current point, if any.
    pub point_label: Option<String>,
    pub ghost_distance: i32,
    pub ghost_mode: i32,
    pub contour_open: bool,
    pub open_enabled: bool,
    pub time: i32,
    pub time_max: i32,
    pub point_size: f32,
    pub point_size_default: i32,
}

/// `indexGood`.
pub fn index_good(ind: Iindex) -> bool {
    ind.object >= 0 && ind.contour >= 0 && ind.point >= 0
}

/// `setlabel`, represented as the source's exact display text.
pub fn set_label(ind: Iindex) -> String {
    if index_good(ind) {
        format!(
            "Object {}, Contour {}, Point {}",
            ind.object + 1,
            ind.contour + 1,
            ind.point + 1
        )
    } else {
        "Obj None, Cont None, Pt None ".into()
    }
}

/// `imodContEditBreakOpen`.
pub fn imod_cont_edit_break_open(state: &mut ContourEditState, current: Iindex) {
    state.break_dialog_open = true;
    state.break_i1 = current;
    state.break_i2 = NULL_INDEX;
}

/// `set1Pressed()` source method.
/// `ContourBreak::set1Pressed`.
pub fn contour_break_set_1(state: &mut ContourEditState, current: Iindex) {
    state.break_i1 = current;
    if index_good(state.break_i2)
        && (state.break_i1.object != state.break_i2.object
            || state.break_i1.contour != state.break_i2.contour)
    {
        state.break_i2 = NULL_INDEX;
    }
}
/// `set2Pressed()` source method.
/// `ContourBreak::set2Pressed`.
pub fn contour_break_set_2(state: &mut ContourEditState, current: Iindex) {
    state.break_i2 = current;
    if index_good(state.break_i1)
        && (state.break_i1.object != state.break_i2.object
            || state.break_i1.contour != state.break_i2.contour)
    {
        state.break_i1 = NULL_INDEX;
    }
}
/// `ContourBreak::unsetPressed`.
pub fn contour_break_unset(state: &mut ContourEditState) {
    state.break_i2 = NULL_INDEX;
}
/// `ContourBreak::currentToggled`.
pub fn contour_break_current_toggled(state: &mut ContourEditState, value: bool) {
    state.use_current = value;
}

/// Values painted by `ContourBreak::setLabels`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContourBreakLabels {
    pub set_1: String,
    pub set_2: String,
    pub object_contour: String,
    pub set_1_enabled: bool,
}

/// `setLabels()` source method.
pub fn contour_break_set_labels(state: &ContourEditState) -> ContourBreakLabels {
    let first = if index_good(state.break_i1) && !state.use_current {
        format!("Point {}", state.break_i1.point + 1)
    } else {
        "Pt None ".into()
    };
    let second = if index_good(state.break_i2) {
        format!("Point {}", state.break_i2.point + 1)
    } else {
        "Pt None ".into()
    };
    let index = if index_good(state.break_i2) {
        state.break_i2
    } else if index_good(state.break_i1) && !state.use_current {
        state.break_i1
    } else {
        NULL_INDEX
    };
    let object_contour = if index.object < 0 || index.contour < 0 {
        "Obj None, Cont None".into()
    } else {
        format!("Object {}, Contour {}", index.object + 1, index.contour + 1)
    };
    ContourBreakLabels {
        set_1: first,
        set_2: second,
        object_contour,
        set_1_enabled: !state.use_current,
    }
}

/// `joinError()` source helper.  Rust ownership releases the index vector
/// automatically; callers retain the native diagnostic text.
pub fn contour_join_error(message: &str) -> String {
    format!("Contour Join Error:  {message}")
}

/// `ContourBreak::breakCont` and `imodContEditBreak`.
pub fn imod_cont_edit_break(
    state: &mut ContourEditState,
    imod: &mut Imod,
    current: Iindex,
    n: &mut dyn ContourEditNativeBoundary,
) -> Result<(), String> {
    let i1 = if state.use_current {
        current
    } else {
        state.break_i1
    };
    let i2 = state.break_i2;
    if !index_good(i1) {
        return Err("Contour Break Error:\n\tFirst break point not set.".into());
    }
    if index_good(i2) && (i1.object != i2.object || i1.contour != i2.contour) {
        return Err(
            "Contour Break Error:\n\tBoth break points must be on the same contour.".into(),
        );
    }
    let obj = imod
        .obj
        .get_mut(i1.object as usize)
        .ok_or("Contour Break Error:\n\tObject number no longer valid.")?;
    let contour_count = obj.cont.len() as i32;
    let cont = obj
        .cont
        .get_mut(i1.contour as usize)
        .ok_or("Contour Break Error:\n\tContour number is no longer valid.")?;
    let (p1, p2) = if index_good(i2) {
        (i1.point.min(i2.point), i1.point.max(i2.point))
    } else {
        (i1.point, -1)
    };
    if cont.pts.is_empty() {
        return Err("Contour Break Error:\n\tSelected contour has no points.".into());
    }
    if p1 < 0 || p1 as usize >= cont.pts.len() || (p2 >= 0 && p2 as usize >= cont.pts.len()) {
        return Err("Contour Break Error:\n\tInvalid break points set.".into());
    }
    n.contour_data_changed(i1.object, i1.contour);
    n.contour_addition(i1.object, contour_count);
    let mut new_cont =
        imod_contour_break(cont, p1, p2).ok_or("Memory or other error breaking contour.")?;
    imodel_contour_check_wild(Some(cont));
    imodel_contour_check_wild(Some(&mut new_cont));
    imod_object_add_contour(obj, new_cont);
    n.finish_undo_unit();
    n.draw_model();
    Ok(())
}

/// `imodContEditJoinOpen`.
pub fn imod_cont_edit_join_open(state: &mut ContourEditState, current: Iindex) {
    state.join_dialog_open = true;
    state.join_i1 = current;
    state.join_i2 = NULL_INDEX;
}
/// `ContourJoin::set1Pressed`.
pub fn contour_join_set_1(state: &mut ContourEditState, current: Iindex) {
    state.join_i1 = current;
    if state.join_i2.object >= 0 && state.join_i2.object != current.object {
        state.join_i2 = NULL_INDEX;
    }
}
/// `ContourJoin::set2Pressed`.
pub fn contour_join_set_2(state: &mut ContourEditState, current: Iindex) {
    state.join_i2 = current;
    if state.join_i1.object >= 0 && state.join_i1.object != current.object {
        state.join_i1 = NULL_INDEX;
    }
}
/// `openTypeSelected()` source method.
/// `ContourJoin::openTypeSelected`.
pub fn contour_join_open_type_selected(state: &mut ContourEditState, which: i32) {
    state.open_type = which;
}
/// `closedTypeSelected()` source method.
/// `ContourJoin::closedTypeSelected`.
pub fn contour_join_closed_type_selected(state: &mut ContourEditState, which: i32) {
    state.closed_type = which;
}
/// `surfToggled()` source method.
pub fn contour_move_surf_toggled(state: &mut ContourEditState, value: bool) {
    state.whole_surf = i32::from(value);
}
/// `toSurfToggled()` source method.
pub fn contour_move_to_surf_toggled(state: &mut ContourEditState, value: bool) {
    state.move_to_surf = i32::from(value);
}
/// `replaceToggled()` source method.
pub fn contour_move_replace_toggled(state: &mut ContourEditState, value: bool) {
    state.replace = i32::from(value);
}
/// `expandToggled()` source method.
pub fn contour_move_expand_toggled(state: &mut ContourEditState, value: bool) {
    state.expand = i32::from(value);
}
/// `convertAllToggled()` source method.
pub fn contour_move_convert_all_toggled(state: &mut ContourEditState, value: bool) {
    state.convert_all_pt = i32::from(value);
}
/// `keepSizeToggled()` source method.
pub fn contour_move_keep_size_toggled(state: &mut ContourEditState, value: bool) {
    state.keep_size = i32::from(value);
}

/// Widget sensitivity computed by `ContourMove::manageCheckBoxes`.  The Rust
/// GUI applies these values to its own controls instead of retaining Qt widgets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContourMoveCheckBoxes {
    pub replace_enabled: bool,
    pub expand_enabled: bool,
    pub convert_all_enabled: bool,
    pub move_all_enabled: bool,
    pub to_surface_enabled: bool,
    pub move_up_down_enabled: bool,
    pub keep_size_enabled: bool,
}

/// `manageCheckBoxes()` source method.
pub fn contour_move_manage_check_boxes(
    state: &ContourEditState,
    current_scattered: bool,
    destination_scattered: bool,
) -> ContourMoveCheckBoxes {
    let replaceable = destination_scattered
        && state.move_to_surf == 0
        && !current_scattered
        && state.move_up_down == 0;
    let replace_on = state.replace != 0 && replaceable;
    let expandable = !destination_scattered
        && state.move_to_surf == 0
        && current_scattered
        && state.move_up_down == 0;
    let expand_on = state.expand != 0 && expandable;
    let up_down_able = !replace_on && !expand_on && state.move_to_surf == 0;
    ContourMoveCheckBoxes {
        replace_enabled: replaceable,
        expand_enabled: expandable,
        convert_all_enabled: expand_on,
        move_all_enabled: !replace_on && !expand_on,
        to_surface_enabled: !replace_on && !expand_on && state.move_up_down == 0,
        move_up_down_enabled: up_down_able,
        keep_size_enabled: current_scattered
            && state.move_to_surf == 0
            && state.move_up_down == 0
            && !expand_on,
    }
}

/// `objSelected()` source method.  Returns whether the object-destination
/// controls need their checkbox sensitivity refreshed.
pub fn contour_move_obj_selected(
    state: &mut ContourEditState,
    obj_move_to: &mut i32,
    value: i32,
) -> bool {
    if state.move_to_surf != 0 {
        state.surf_move_to = value;
        false
    } else {
        *obj_move_to = value;
        true
    }
}

/// `moveUpDownToggled()` source method.
pub fn contour_move_up_down_toggled(state: &mut ContourEditState, value: bool) {
    state.move_up_down = i32::from(value);
}

/// `upDownSelected()` source method.
pub fn contour_move_up_down_selected(state: &mut ContourEditState, which: i32) {
    state.up_or_down = which;
}

/// `shiftContClicked()` source method.  The caller dispatches this request to
/// the active Rust Zap view when one exists.
pub fn contour_move_shift_cont_clicked(has_top_zap: bool) -> bool {
    has_top_zap
}

/// `buttonPressed()` source slot: whether an Apply action was selected.
pub fn contour_button_pressed(which: i32) -> bool {
    which == 0
}

/// `topChangeEvent()` source slot.  Native font and mac-menu handling is a
/// Rust GUI concern; this tells it whether a font-dependent remeasure is due.
pub fn contour_top_change_event(font_changed: bool) -> bool {
    font_changed
}

/// `setFontDependentWidths()` source method.  The destination button takes
/// the measured width of the source Set-1 button, as in Qt.
pub fn contour_set_font_dependent_widths(set_1_width: i32) -> i32 {
    set_1_width
}

/// `topCloseEvent()` source slot for the break dialog.
pub fn contour_break_top_close_event(state: &mut ContourEditState) {
    state.break_dialog_open = false;
}

/// `topCloseEvent()` source slot for the join dialog.
pub fn contour_join_top_close_event(state: &mut ContourEditState) {
    state.join_dialog_open = false;
    state.join_i1 = NULL_INDEX;
    state.join_i2 = NULL_INDEX;
}

/// `topCloseEvent()` source slot for the move dialog.
pub fn contour_move_top_close_event(state: &mut ContourEditState) {
    state.move_dialog_open = false;
}

/// `imodContEditJoin`, including pair-wise selection-list joining.  The
/// source selection-list ownership is supplied as `indices`.
pub fn imod_cont_edit_join(
    state: &mut ContourEditState,
    imod: &mut Imod,
    current: Iindex,
    indices: &[Iindex],
    n: &mut dyn ContourEditNativeBoundary,
) -> Result<(), String> {
    let mut selected = if indices.len() >= 2 {
        indices.to_vec()
    } else if index_good(state.join_i1) && index_good(state.join_i2) {
        vec![state.join_i1, state.join_i2]
    } else if index_good(state.join_i1) && state.join_i1.contour != current.contour {
        vec![state.join_i1, current]
    } else if index_good(state.join_i2) && state.join_i2.contour != current.contour {
        vec![current, state.join_i2]
    } else {
        return Err("Contour Join Error:  Two contours not selected.".into());
    };
    let object_no = selected[0].object;
    let obj = imod
        .obj
        .get_mut(object_no as usize)
        .ok_or("Contour Join Error:  Object number no longer valid.")?;
    let set_points = (iobj_open(obj.flags) != 0 && state.open_type == OPEN_TYPE_SPLICE)
        || (iobj_open(obj.flags) == 0 && state.closed_type == CLOSED_TYPE_SETPOINT);
    if set_points && selected.len() > 2 {
        return Err(
            "Contour Join Error:  Cannot join more than two contours by set points.".into(),
        );
    }
    for ind in &selected {
        if ind.object != object_no
            || ind.contour < 0
            || ind.contour as usize >= obj.cont.len()
            || (set_points
                && (ind.point < 0
                    || ind.point as usize >= obj.cont[ind.contour as usize].pts.len()))
        {
            return Err("Contour Join Error:  Invalid join index.".into());
        }
    }
    while selected.len() > 1 {
        let first = selected.remove(0);
        let second = selected.remove(0);
        if first.contour == second.contour {
            return Err("Contour Join Error:  Set points must be in different contours.".into());
        }
        let a = first.contour as usize;
        let b = second.contour as usize;
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let (left, right) = obj.cont.split_at_mut(hi);
        let (c1, c2) = if a < b {
            (&mut left[lo], &mut right[0])
        } else {
            (&mut right[0], &mut left[lo])
        };
        let (p1, p2) = if set_points {
            (first.point, second.point)
        } else {
            (-1, -1)
        };
        n.contour_data_changed(object_no, first.contour);
        n.contour_data_changed(object_no, second.contour);
        let joined = if iobj_open(obj.flags) != 0 || state.closed_type == CLOSED_TYPE_CONCAT {
            imod_contour_splice(
                Some(c1),
                Some(c2),
                if p1 < 0 { c1.pts.len() as i32 - 1 } else { p1 },
                if p2 < 0 { 0 } else { p2 },
            )
        } else {
            imod_contour_join(Some(c1), Some(c2), p1, p2, 0, 0)
        }
        .ok_or("Contour Join Error:  Failed to get memory for joined contour")?;
        obj.cont[a] = joined;
        let removed = b as i32;
        imod_object_remove_contour(obj, removed);
        for ind in &mut selected {
            if ind.contour > removed {
                ind.contour -= 1;
            }
        }
        selected.insert(
            0,
            Iindex {
                object: object_no,
                contour: if a > b { a as i32 - 1 } else { a as i32 },
                point: 0,
            },
        );
    }
    n.finish_undo_unit();
    n.draw_model();
    Ok(())
}

/// `imodContEditMoveDialog` first-open initialization.
pub fn imod_cont_edit_move_dialog(state: &mut ContourEditState, move_surf: i32) {
    state.move_dialog_open = true;
    if state.move_first {
        state.whole_surf = move_surf;
        state.move_first = false;
    } else {
        state.whole_surf = move_surf;
    }
}
/// `imodContEditMoveDialogUpdate`.
pub fn imod_cont_edit_move_dialog_update(
    state: &mut ContourEditState,
    imod: &Imod,
    current_object: i32,
    obj_move_to: &mut i32,
) -> (i32, i32, i32) {
    let (min, mut max, mut value) = if state.move_to_surf != 0 {
        let max = imod
            .obj
            .get(current_object as usize)
            .map_or(0, |o| o.surfsize + 1);
        state.surf_move_to = state.surf_move_to.min(max);
        (0, max, state.surf_move_to)
    } else {
        *obj_move_to = (*obj_move_to).min(imod.obj.len() as i32);
        (1, imod.obj.len() as i32, *obj_move_to)
    };
    if max <= min {
        value = min;
        max = min + 1;
        state.enabled = 0;
    } else {
        state.enabled = 1;
    };
    (min, max, value)
}

/// `imodContEditMove`, with all contour/surface/object transformations and
/// explicit undo/display calls.  `current` is `imod->cindex`.
pub fn imod_cont_edit_move(
    state: &mut ContourEditState,
    imod: &mut Imod,
    current: &mut Iindex,
    obj_move_to: &mut i32,
    xybin: f32,
    n: &mut dyn ContourEditNativeBoundary,
) -> Result<(), String> {
    if state.move_first {
        return Err("Error: Select Edit->Contour->Move to setup move.".into());
    }
    if state.enabled == 0 && state.move_up_down == 0 {
        return Err(
            "Error: Must have more than one object or surface to be able to move contours.".into(),
        );
    }
    let ob = current.object;
    let co = current.contour;
    let pt = current.point;
    if ob < 0
        || co < 0
        || ob as usize >= imod.obj.len()
        || co as usize >= imod.obj[ob as usize].cont.len()
    {
        return Ok(());
    }
    *obj_move_to = (*obj_move_to).clamp(1, imod.obj.len() as i32);
    let dest = *obj_move_to - 1;
    /* REPLACE CONTOUR BY POINT IS FIRST (`cont_edit.cpp:1067`). */
    if dest != ob
        && state.replace != 0
        && state.move_to_surf == 0
        && state.move_up_down == 0
        && iobj_scat(imod.obj[dest as usize].flags) != 0
        && iobj_scat(imod.obj[ob as usize].flags) == 0
    {
        let source_obj = &imod.obj[ob as usize];
        let source = &source_obj.cont[co as usize];
        if source.pts.len() < 3 {
            return Err("Error: Contour must have at least 3 points.".into());
        }
        let first_z = source.pts[0].z;
        if source.pts.iter().any(|point| point.z != first_z) {
            return Err("Error: Contour not all in one plane.".into());
        }
        let mut center = Ipoint::default();
        for point in &source.pts {
            center.x += point.x;
            center.y += point.y;
        }
        center.x /= source.pts.len() as f32;
        center.y /= source.pts.len() as f32;
        center.z = first_z;
        let size = (imod_contour_area(Some(source)) / std::f32::consts::PI).sqrt() * xybin;
        let (src, dst) = if ob < dest {
            let (left, right) = imod.obj.split_at_mut(dest as usize);
            (&mut left[ob as usize], &mut right[0])
        } else {
            let (left, right) = imod.obj.split_at_mut(ob as usize);
            (&mut right[0], &mut left[dest as usize])
        };
        if dst.cont.is_empty() {
            n.contour_addition(dest, 0);
            imod_object_add_contour(dst, Icont::default());
        }
        let point_index = dst.cont.last().unwrap().pts.len() as i32;
        n.point_addition(dest, dst.cont.len() as i32 - 1, point_index);
        dst.cont.last_mut().unwrap().pts.push(center);
        let last = dst.cont.len() - 1;
        imod_point_set_size(&mut dst.cont[last], point_index, size);
        n.contour_removal(ob, co);
        imod_object_remove_contour(src, co);
        current.contour = (co - 1).max(-1);
        current.point = -1;
        n.finish_undo_unit();
        n.clear_selection();
        n.draw_model();
        return Ok(());
    }
    /* EXPAND A SCATTERED POINT INTO CONTOURS (`cont_edit.cpp:1111`). */
    if dest != ob
        && state.expand != 0
        && state.move_to_surf == 0
        && state.move_up_down == 0
        && iobj_scat(imod.obj[dest as usize].flags) == 0
        && iobj_scat(imod.obj[ob as usize].flags) != 0
    {
        let source_obj = &imod.obj[ob as usize];
        let source = &source_obj.cont[co as usize];
        if pt < 0 && state.convert_all_pt == 0 {
            return Ok(());
        }
        let point_indices: Vec<usize> = if state.convert_all_pt != 0 {
            (0..source.pts.len()).collect()
        } else {
            vec![pt as usize]
        };
        let values: Vec<(usize, Ipoint, f32)> = point_indices
            .into_iter()
            .filter_map(|pi| {
                source.pts.get(pi).copied().map(|point| {
                    (
                        pi,
                        point,
                        imod_point_get_size(source_obj, source, pi as i32) / xybin,
                    )
                })
            })
            .collect();
        let zscale = (imod.zscale.max(1.) / xybin.max(f32::MIN_POSITIVE)).max(f32::MIN_POSITIVE);
        let resolution = imod.res.max(1) as f32;
        let (src, dst) = if ob < dest {
            let (left, right) = imod.obj.split_at_mut(dest as usize);
            (&mut left[ob as usize], &mut right[0])
        } else {
            let (left, right) = imod.obj.split_at_mut(ob as usize);
            (&mut right[0], &mut left[dest as usize])
        };
        let new_surface = imodel_unused_surface(Some(dst));
        let mut remove = Vec::new();
        for (pi, center, radius) in values {
            if radius < 1.45 {
                n.warning("Point radius must be at least 1.45");
                continue;
            }
            let dz_lim = (radius / zscale + 1.) as i32;
            for iz in (center.z.round() as i32 - dz_lim)..=(center.z.round() as i32 + dz_lim) {
                let dz = (iz as f32 - center.z) * zscale;
                if dz.abs() >= radius {
                    continue;
                }
                let circle_radius = (radius * radius - dz * dz).sqrt();
                if circle_radius < 1. {
                    continue;
                }
                let npts = ((2. * std::f32::consts::PI * circle_radius * xybin) / resolution)
                    .ceil()
                    .max(6.) as usize;
                let mut contour = Icont::default();
                contour.surf = new_surface;
                for p in 0..npts {
                    let angle = 2. * std::f32::consts::PI * p as f32 / npts as f32;
                    contour.pts.push(Ipoint {
                        x: center.x + circle_radius * angle.cos(),
                        y: center.y + circle_radius * angle.sin(),
                        z: iz as f32,
                    });
                }
                n.contour_addition(dest, dst.cont.len() as i32);
                imod_object_add_contour(dst, contour);
            }
            remove.push(pi);
        }
        for pi in remove.into_iter().rev() {
            n.point_removal(ob, co, pi as i32);
            imod_point_delete(&mut src.cont[co as usize], pi as i32);
        }
        if state.convert_all_pt != 0 && src.cont[co as usize].pts.is_empty() {
            n.contour_removal(ob, co);
            imod_object_remove_contour(src, co);
            current.contour = (co - 1).max(-1);
        }
        current.point = -1;
        n.finish_undo_unit();
        n.draw_model();
        return Ok(());
    }
    if state.move_up_down != 0 {
        let delta = if state.up_or_down != 0 { -1. } else { 1. };
        let surf = imod.obj[ob as usize].cont[co as usize].surf;
        for (oi, obj) in imod.obj.iter_mut().enumerate() {
            for (ci, cont) in obj.cont.iter_mut().enumerate() {
                if (oi == ob as usize
                    && (ci == co as usize || (state.whole_surf != 0 && cont.surf == surf)))
                    || (state.whole_surf == 0 && n.selection_contains(oi as i32, ci as i32))
                {
                    n.contour_data_changed(oi as i32, ci as i32);
                    for p in &mut cont.pts {
                        p.z += delta;
                    }
                }
            }
        }
        if state.up_or_down != 0 {
            n.prev_z();
        } else {
            n.next_z();
        };
        n.finish_undo_unit();
        return Ok(());
    }
    if state.move_to_surf != 0 {
        let obj = &mut imod.obj[ob as usize];
        let surf = obj.cont[co as usize].surf;
        for (ci, cont) in obj.cont.iter_mut().enumerate() {
            if (state.whole_surf != 0 && cont.surf == surf)
                || (state.whole_surf == 0
                    && (ci == co as usize || n.selection_contains(ob, ci as i32)))
            {
                n.contour_property_changed(ob, ci as i32);
                cont.surf = state.surf_move_to;
            }
        }
        obj.surfsize = obj.surfsize.max(state.surf_move_to);
        imod_object_clean_surf(obj);
        n.finish_undo_unit();
        n.draw_model();
        return Ok(());
    }
    if dest == ob {
        return Err("Error: Trying to move contour to object it is already in.".into());
    }
    let (src, dst) = if ob < dest {
        let (left, right) = imod.obj.split_at_mut(dest as usize);
        (&mut left[ob as usize], &mut right[0])
    } else {
        let (left, right) = imod.obj.split_at_mut(ob as usize);
        (&mut right[0], &mut left[dest as usize])
    };
    let source_surf = src.cont[co as usize].surf;
    let destination_surf = if state.whole_surf != 0 {
        imodel_unused_surface(Some(dst))
    } else {
        -1
    };
    let mut indices: Vec<usize> = src
        .cont
        .iter()
        .enumerate()
        .filter_map(|(ci, c)| {
            if (state.whole_surf != 0 && c.surf == source_surf)
                || (state.whole_surf == 0
                    && (ci == co as usize || n.selection_contains(ob, ci as i32)))
            {
                Some(ci)
            } else {
                None
            }
        })
        .collect();
    for ci in indices.drain(..).rev() {
        let mut cont = src.cont.remove(ci);
        if destination_surf >= 0 {
            cont.surf = destination_surf;
        }
        if state.keep_size != 0 && iobj_scat(src.flags) != 0 {
            for pi in 0..cont.pts.len() {
                let size = imod_point_get_size(src, &cont, pi as i32);
                imod_point_set_size(&mut cont, pi as i32, size);
            }
        }
        n.contour_move(ob, ci as i32, dest, dst.cont.len() as i32);
        imod_object_add_contour(dst, cont);
    }
    imod_object_clean_surf(src);
    imod_object_clean_surf(dst);
    current.contour = if src.cont.is_empty() {
        -1
    } else {
        (co - 1).max(0)
    };
    current.point = -1;
    n.finish_undo_unit();
    n.clear_selection();
    n.draw_model();
    Ok(())
}

/// `iceClosedOpen` for the current contour plus source selection semantics.
pub fn ice_closed_open(
    imod: &mut Imod,
    current: Iindex,
    selected: &[Iindex],
    state: i32,
    n: &mut dyn ContourEditNativeBoundary,
) {
    let valid: Vec<Iindex> = selected
        .iter()
        .copied()
        .filter(|i| {
            i.object >= 0
                && (i.object as usize) < imod.obj.len()
                && i.contour >= 0
                && (i.contour as usize) < imod.obj[i.object as usize].cont.len()
                && iobj_open(imod.obj[i.object as usize].flags) == 0
        })
        .collect();
    let change = if valid.len() < 2 {
        vec![current]
    } else {
        valid
    };
    for ind in change {
        if let Some(cont) = imod
            .obj
            .get_mut(ind.object as usize)
            .and_then(|o| o.cont.get_mut(ind.contour as usize))
        {
            n.contour_property_changed(ind.object, ind.contour);
            if state != 0 {
                cont.flags |= ICONT_OPEN
            } else {
                cont.flags &= !ICONT_OPEN
            };
        }
    }
    n.finish_undo_unit();
    n.draw_model();
}

/// `imodContEditSurf`.
pub fn imod_cont_edit_surf(state: &mut ContourEditState) {
    state.surf_dialog_open = true;
}

/// `imodContEditSurfShow`.  Form labels are maintained by `form_cont_edit.rs`;
/// this returns the numeric/value state pushed into that form.
pub fn imod_cont_edit_surf_show(
    imod: &Imod,
    current: Iindex,
    num_times: i32,
    ghost_distance: i32,
    ghost_mode: i32,
) -> Option<ContourSurfacePointDisplay> {
    let obj = imod.obj.get(current.object as usize)?;
    let cont = obj.cont.get(current.contour as usize);
    // `imodLabel*Get` exposes C label storage; forms receive its NUL-terminated
    // text, not the backing terminator retained by the model label item.
    let label_text = |bytes: Option<&[u8]>| {
        bytes.map(|text| {
            String::from_utf8_lossy(
                &text[..text
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(text.len())],
            )
            .into_owned()
        })
    };
    let (point_size, point_size_default) = match cont {
        Some(cont) if current.point >= 0 && (current.point as usize) < cont.pts.len() => (
            imod_point_get_size(obj, cont, current.point),
            if cont
                .sizes
                .get(current.point as usize)
                .copied()
                .unwrap_or(-1.)
                >= 0.
            {
                0
            } else {
                1
            },
        ),
        _ => (0., -1),
    };
    Some(ContourSurfacePointDisplay {
        // With no current contour, the native dialog displays the selected
        // mesh surface (`imodGetCurMeshSurf`), not an unconditional -1.
        surface: cont.map_or_else(|| imod_get_cur_mesh_surf(imod), |c| c.surf),
        surface_max: obj.surfsize,
        // `cont_edit.cpp:1738-1741, 1780-1795`: labels are associated with
        // the selected surface, contour name, and selected point respectively.
        surface_label: cont
            .and_then(|c| label_text(imod_label_item_get(obj.label.as_ref(), c.surf))),
        contour_label: cont.and_then(|c| label_text(imod_label_name_get(c.label.as_ref()))),
        point_label: cont.and_then(|c| {
            (current.point >= 0)
                .then(|| imod_label_item_get(c.label.as_ref(), current.point))
                .flatten()
                .and_then(|text| label_text(Some(text)))
        }),
        ghost_distance,
        ghost_mode,
        contour_open: cont.is_some_and(|c| c.flags & ICONT_OPEN != 0),
        open_enabled: iobj_open(obj.flags) == 0,
        time: if obj.flags & IMOD_OBJFLAG_TIME != 0 {
            cont.map_or(-1, |c| c.time)
        } else {
            -2
        },
        // `cont_edit.cpp:1761-1763`.
        time_max: imod.tmax.max(num_times),
        point_size,
        point_size_default,
    })
}

/// `iceSurfGoto`.
pub fn ice_surf_goto(target: i32, n: &mut dyn ContourEditInputBoundary) {
    n.goto_surface(target)
}
/// `iceContInSurf`.
pub fn ice_cont_in_surf(direction: i32, n: &mut dyn ContourEditInputBoundary) {
    n.adjacent_contour_in_surface(direction)
}
/// `iceSurfNew`.
pub fn ice_surf_new(n: &mut dyn ContourEditInputBoundary) {
    n.new_surface()
}

/// `iceTimeChanged`.
pub fn ice_time_changed(
    imod: &mut Imod,
    current: Iindex,
    value: i32,
    n: &mut dyn ContourEditNativeBoundary,
) {
    if let Some(obj) = imod.obj.get_mut(current.object as usize) {
        if obj.flags & IMOD_OBJFLAG_TIME != 0 {
            if let Some(cont) = obj.cont.get_mut(current.contour as usize) {
                n.contour_property_changed(current.object, current.contour);
                cont.time = value;
                n.finish_undo_unit();
            }
        }
    };
    n.draw_model();
}
/// `iceLabelChanged`.
pub fn ice_label_changed(
    imod: &mut Imod,
    current: Iindex,
    text: &str,
    cont_point: i32,
    n: &mut dyn ContourEditNativeBoundary,
) {
    let Some(obj) = imod.obj.get_mut(current.object as usize) else {
        return;
    };
    let bytes = text.as_bytes();
    if cont_point == 2 {
        n.object_property_changed(current.object);
        if obj.label.is_none() {
            obj.label = Some(imod_label_new())
        };
        imod_label_item_add(
            obj.label.as_mut().unwrap(),
            Some(bytes),
            obj.cont.get(current.contour as usize).map_or(0, |c| c.surf),
        );
    } else if let Some(cont) = obj.cont.get_mut(current.contour as usize) {
        n.contour_data_changed(current.object, current.contour);
        if cont.label.is_none() {
            cont.label = Some(imod_label_new())
        };
        if cont_point != 0 {
            imod_label_item_add(cont.label.as_mut().unwrap(), Some(bytes), current.point)
        } else {
            imod_label_name(cont.label.as_mut(), Some(bytes));
        }
    };
    n.finish_undo_unit();
}
/// `iceLabelFinished`.
pub fn ice_label_finished(cont_point: i32, n: &mut dyn ContourEditNativeBoundary) {
    if cont_point == 1 {
        n.draw_model()
    }
}
/// `iceLabelWithMeasure`.
pub fn ice_label_with_measure(
    imod: &mut Imod,
    current: Iindex,
    pixel_size: f32,
    xybin: f32,
    zbin: f32,
    area: bool,
    n: &mut dyn ContourEditNativeBoundary,
) {
    let Some(obj) = imod.obj.get(current.object as usize) else {
        return;
    };
    let Some(cont) = obj.cont.get(current.contour as usize) else {
        return;
    };
    if iobj_scat(obj.flags) != 0 || cont.pts.is_empty() {
        return;
    };
    let value = if area {
        imod_contour_area(Some(cont)) * pixel_size * pixel_size * xybin * xybin
    } else {
        let mut sum = 0.;
        for pair in cont.pts.windows(2) {
            let dx = (pair[1].x - pair[0].x) * xybin;
            let dy = (pair[1].y - pair[0].y) * xybin;
            let dz = (pair[1].z - pair[0].z) * zbin;
            sum += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        if iobj_open(obj.flags) == 0 && cont.flags & ICONT_OPEN == 0 && cont.pts.len() > 1 {
            let a = cont.pts[0];
            let b = cont.pts[cont.pts.len() - 1];
            sum += (((a.x - b.x) * xybin).powi(2)
                + ((a.y - b.y) * xybin).powi(2)
                + ((a.z - b.z) * zbin).powi(2))
            .sqrt();
        }
        sum * pixel_size
    };
    ice_label_changed(imod, current, &format!("{value:.4}"), 1, n);
    n.draw_model();
}
/// `icePointSize`.
pub fn ice_point_size(
    imod: &mut Imod,
    current: Iindex,
    size: f32,
    n: &mut dyn ContourEditNativeBoundary,
) {
    if let Some(cont) = imod
        .obj
        .get_mut(current.object as usize)
        .and_then(|o| o.cont.get_mut(current.contour as usize))
    {
        if current.point >= 0 {
            n.contour_data_changed(current.object, current.contour);
            imod_point_set_size(cont, current.point, size);
            n.finish_undo_unit();
            n.draw_model();
        }
    }
}
/// `iceSetWheelForSize`.
pub fn ice_set_wheel_for_size(edit: &mut ContourEditState, state: i32) {
    edit.wheel_for_size = state
}
/// `iceGetWheelForSize`.
pub fn ice_get_wheel_for_size(edit: &ContourEditState) -> i32 {
    edit.wheel_for_size
}
/// `iceGhostInterval`.
pub fn ice_ghost_interval(ghost_dist: &mut i32, value: i32, n: &mut dyn ContourEditNativeBoundary) {
    *ghost_dist = value;
    n.draw_model()
}
/// `iceGhostToggled`.
pub fn ice_ghost_toggled(
    ghost_mode: &mut i32,
    ghost_last: &mut i32,
    state: i32,
    flag: i32,
    n: &mut dyn ContourEditNativeBoundary,
) {
    if state == 0 {
        *ghost_mode &= !flag
    } else {
        *ghost_mode |= flag
    };
    if flag & 1 != 0 && *ghost_mode & 1 != 0 {
        *ghost_last = *ghost_mode
    };
    n.draw_model()
}
/// `iceClosing`.
pub fn ice_closing(state: &mut ContourEditState) {
    state.surf_dialog_open = false;
}

/// Source state representation of `ContourFrame`, `ContourMove`,
/// `ContourJoin`, and `ContourBreak`; widget painting/events stay with Qt.
#[derive(Clone, Debug, Default)]
pub struct ContourFrame {
    pub top_window_open: bool,
}

/// Routing decision from `ContourFrame` keyboard handlers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContourFrameKeyAction {
    Close,
    Forward { released: bool },
}

/// `ContourFrame::keyPressEvent`.  `is_close_key` is supplied by the Rust
/// windowing backend's platform-neutral close-key predicate.
pub fn contour_frame_key_press_event(is_close_key: bool) -> ContourFrameKeyAction {
    if is_close_key {
        ContourFrameKeyAction::Close
    } else {
        ContourFrameKeyAction::Forward { released: false }
    }
}

#[allow(non_snake_case)]
/// Native-name adapter for `ContourFrame::keyPressEvent`.
pub fn keyPressEvent(is_close_key: bool) -> ContourFrameKeyAction {
    contour_frame_key_press_event(is_close_key)
}

/// `ContourFrame::keyReleaseEvent`.
pub fn contour_frame_key_release_event() -> ContourFrameKeyAction {
    ContourFrameKeyAction::Forward { released: true }
}

#[allow(non_snake_case)]
/// Native-name adapter for `ContourFrame::keyReleaseEvent`.
pub fn keyReleaseEvent() -> ContourFrameKeyAction {
    contour_frame_key_release_event()
}
#[derive(Clone, Debug, Default)]
pub struct ContourMove {
    pub frame: ContourFrame,
}
#[derive(Clone, Debug, Default)]
pub struct ContourJoin {
    pub frame: ContourFrame,
}
#[derive(Clone, Debug, Default)]
pub struct ContourBreak {
    pub frame: ContourFrame,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct N {
        calls: Vec<&'static str>,
    }
    impl ContourEditNativeBoundary for N {
        fn contour_data_changed(&mut self, _: i32, _: i32) {
            self.calls.push("data")
        }
        fn contour_property_changed(&mut self, _: i32, _: i32) {}
        fn contour_addition(&mut self, _: i32, _: i32) {}
        fn contour_removal(&mut self, _: i32, _: i32) {}
        fn contour_move(&mut self, _: i32, _: i32, _: i32, _: i32) {}
        fn object_property_changed(&mut self, _: i32) {}
        fn point_addition(&mut self, _: i32, _: i32, _: i32) {}
        fn point_removal(&mut self, _: i32, _: i32, _: i32) {}
        fn finish_undo_unit(&mut self) {
            self.calls.push("finish")
        }
        fn flush_undo_unit(&mut self) {}
        fn draw_model(&mut self) {
            self.calls.push("draw")
        }
        fn selection_contains(&self, _: i32, _: i32) -> bool {
            false
        }
        fn clear_selection(&mut self) {}
        fn next_z(&mut self) {}
        fn prev_z(&mut self) {}
        fn warning(&mut self, _: &str) {}
    }
    fn model() -> Imod {
        let mut m = Imod::default();
        let mut o = Iobj::default();
        o.cont.push(Icont {
            pts: vec![
                Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 0.,
                    z: 0.,
                },
                Ipoint {
                    x: 1.,
                    y: 1.,
                    z: 0.,
                },
            ],
            ..Icont::default()
        });
        m.obj.push(o);
        m
    }
    #[test]
    fn break_contour_is_model_edit_and_draws() {
        let mut s = ContourEditState::default();
        let mut m = model();
        let mut n = N::default();
        imod_cont_edit_break_open(
            &mut s,
            Iindex {
                object: 0,
                contour: 0,
                point: 1,
            },
        );
        imod_cont_edit_break(&mut s, &mut m, Iindex::default(), &mut n).unwrap();
        assert_eq!(m.obj[0].cont.len(), 2);
        assert_eq!(n.calls.last(), Some(&"draw"));
    }
    #[test]
    fn labels_and_ghost_state_follow_source() {
        let mut m = model();
        let mut n = N::default();
        let i = Iindex {
            object: 0,
            contour: 0,
            point: 1,
        };
        ice_label_changed(&mut m, i, "p", 1, &mut n);
        assert!(m.obj[0].cont[0].label.is_some());
        let (mut mode, mut last) = (0, 0);
        ice_ghost_toggled(&mut mode, &mut last, 1, 1, &mut n);
        assert_eq!(last, 1);
    }
    #[test]
    fn surface_dialog_display_includes_all_label_targets() {
        let mut m = model();
        let mut n = N::default();
        let i = Iindex {
            object: 0,
            contour: 0,
            point: 1,
        };
        ice_label_changed(&mut m, i, "surface", 2, &mut n);
        ice_label_changed(&mut m, i, "contour", 0, &mut n);
        ice_label_changed(&mut m, i, "point", 1, &mut n);

        let display = imod_cont_edit_surf_show(&m, i, 1, 3, 0).unwrap();
        assert_eq!(display.surface_label.as_deref(), Some("surface"));
        assert_eq!(display.contour_label.as_deref(), Some("contour"));
        assert_eq!(display.point_label.as_deref(), Some("point"));
    }
    #[test]
    fn surface_dialog_uses_current_mesh_surface_without_contour() {
        let mut m = Imod::default();
        let mut obj = Iobj::default();
        obj.surfsize = 4;
        obj.mesh.push(Imesh::default());
        m.obj.push(obj);
        m.cindex = Iindex {
            object: 0,
            contour: -1,
            point: -1,
        };
        m.cur_mesh_surf = 3;

        let display = imod_cont_edit_surf_show(&m, m.cindex, 0, 0, 0).unwrap();
        assert_eq!(display.surface, 3);
        assert_eq!(display.surface_label, None);
    }
}
