//! Translation of `IMOD/3dmod/imod_edit.cpp` and `imod_edit.h`.
//!
//! The model-editing algorithms are source-shaped.  The original global App,
//! undo stack, and Qt refresh machinery are represented by explicit arguments
//! at their call sites rather than silently replaced by a new editor layer.
#![allow(dead_code, unused_variables)]

use std::collections::BTreeSet;

use crate::imod::libimod::icont::{imod_contour_break, imod_contour_copy};
use crate::imod::libimod::imat::{Imat, imod_mat_transform3d};
use crate::imod::libimod::imesh::imodel_mesh_add;
use crate::imod::libimod::imodel::{ICONT_OPEN, ICONT_WILD, Icont, Iindex, Imod, Iobj, Ipoint};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_TIME, imod_object_add_contour, imod_object_remove_contour, iobj_off, iobj_open,
};
use crate::imod::libimod::ipoint::{
    imod_point_add, imod_point_distance, imod_point_get_size, imod_point_inside_cont,
    imod_point_intersect, imod_point3d_scale_distance,
};
use crate::imod::libimod::istore::{istore_copy_cont_surf_items, istore_count_cont_surf_items};
use crate::imod::three_dmod::imod::imod_trace;
use crate::imod::three_dmod::imodview::ImodView;

/// `X_SLICE_BOX`, `Y_SLICE_BOX` and `Z_SLICE_BOX` from the anonymous enum at
/// `xxyz.h:51`, which begins at `NOT_IN_BOX = 0`.  These were previously
/// declared here as 0 and 1 with a `display.h` citation; the behaviour of
/// `imodContourIsPlanar` only distinguishes the three, but the values now
/// match the source so a caller that has them from `xxyz.h` agrees.
pub const X_SLICE_BOX: i32 = 1;
/// See [`X_SLICE_BOX`].
pub const Y_SLICE_BOX: i32 = 2;
/// See [`X_SLICE_BOX`].
pub const Z_SLICE_BOX: i32 = 3;

/// Explicit boundary for `vi->undo` in `imod_edit.cpp`.
pub trait ImodEditUndo {
    fn object_prop_chg(&mut self, object: i32);
    fn contour_move(
        &mut self,
        old_object: i32,
        old_contour: i32,
        new_object: i32,
        new_contour: i32,
    );
    fn all_contour_move(&mut self, old_object: i32, new_object: i32);
    fn contour_data_chg(&mut self, object: i32, contour: i32);
}

/// `imodContInSelectArea` (`imod_edit.cpp:31`).
pub fn imod_cont_in_select_area(obj: &Iobj, cont: &Icont, selmin: Ipoint, selmax: Ipoint) -> i32 {
    let mut pmin = Ipoint::default();
    let mut pmax = Ipoint::default();
    crate::imod::libimod::icont::imod_contour_get_bbox(Some(cont), &mut pmin, &mut pmax);
    if pmin.x >= selmin.x
        && pmax.x <= selmax.x
        && pmin.y >= selmin.y
        && pmax.y <= selmax.y
        && pmin.z >= selmin.z
        && pmax.z <= selmax.z
    {
        return 1;
    }
    if !(iobj_open(obj.flags) != 0 && cont.flags & ICONT_WILD != 0) {
        return 0;
    }
    let mut in_range = 0;
    for pnt in &cont.pts {
        if pnt.z >= selmin.z && pnt.z <= selmax.z {
            in_range = 1;
            if pnt.x < selmin.x || pnt.x > selmax.x || pnt.y < selmin.y || pnt.y > selmax.y {
                return 0;
            }
        }
    }
    in_range
}

/// `imodContInsideCont` (`imod_edit.cpp:63`).
pub fn imod_cont_inside_cont(obj: &Iobj, cont: &Icont, outer: &Icont, zmin: f32, zmax: f32) -> i32 {
    let wild_open = iobj_open(obj.flags) != 0 && cont.flags & ICONT_WILD != 0;
    let mut in_range = i32::from(!wild_open);
    for pnt in &cont.pts {
        let on_sec = pnt.z >= zmin && pnt.z <= zmax;
        let inside = imod_point_inside_cont(outer, pnt) != 0;
        if !inside || !on_sec {
            if !wild_open || on_sec {
                return 0;
            }
        } else if wild_open {
            in_range = 1;
        }
    }
    in_range
}

/// Static `imod_distance` (`imod_edit.cpp:246`).
fn imod_distance(x: f32, y: f32, pnt: &Ipoint) -> f32 {
    let distance = ((x - pnt.x) * (x - pnt.x) + (y - pnt.y) * (y - pnt.y)) as f64;
    distance.sqrt() as f32
}

/// `imod_obj_nearest` (`imod_edit.cpp:90`).
pub fn imod_obj_nearest(
    vi: &ImodView,
    obj: &Iobj,
    index: &mut Iindex,
    pnt: &Ipoint,
    selsize: f32,
    ctime: i32,
    mat: Option<&Imat>,
) -> f32 {
    let mut distance = -1.;
    let cz = (pnt.z + 0.5).floor() as i32;
    let twod = vi.dim & 4 == 0;
    let scale = Ipoint {
        x: 1.,
        y: 1.,
        z: (if !vi.imod.is_null() && unsafe { (*vi.imod).zscale } > 0. {
            unsafe { (*vi.imod).zscale }
        } else {
            1.
        } * vi.zbin as f32)
            / vi.xybin as f32,
    };
    let mut pntrot = Ipoint::default();
    if let Some(mat) = mat {
        imod_mat_transform3d(mat, pnt, &mut pntrot);
    }
    for (i, cont) in obj.cont.iter().enumerate() {
        if ctime != 0 && obj.flags & IMOD_OBJFLAG_TIME != 0 && cont.time != 0 && cont.time != ctime
        {
            continue;
        }
        if cont.pts.is_empty() {
            continue;
        }
        if let Some(mat) = mat {
            for (pindex, point) in cont.pts.iter().enumerate() {
                let mut ptsrot = Ipoint::default();
                imod_mat_transform3d(mat, point, &mut ptsrot);
                if (pntrot.x - ptsrot.x).abs() < selsize && (pntrot.y - ptsrot.y).abs() < selsize {
                    let mut delz = 0.5;
                    if obj.pdrawsize != 0 || !cont.sizes.is_empty() {
                        let rad = imod_point_get_size(obj, cont, pindex as i32) as f64;
                        if rad > 1. {
                            delz = (rad * rad - 1.).sqrt() as f32;
                        }
                    }
                    if (ptsrot.z - pntrot.z).abs() <= delz {
                        let temp_distance = imod_point3d_scale_distance(&pntrot, &ptsrot, &scale);
                        if distance < 0. || distance > temp_distance {
                            distance = temp_distance;
                            index.contour = i as i32;
                            index.point = pindex as i32;
                        }
                    }
                }
            }
        } else if (obj.pdrawsize != 0 || !cont.sizes.is_empty()) && !twod {
            for (pindex, point) in cont.pts.iter().enumerate() {
                if (pnt.x - point.x).abs() < selsize && (pnt.y - point.y).abs() < selsize {
                    let rad = imod_point_get_size(obj, cont, pindex as i32) / vi.xybin as f32;
                    let mut delz = 0.5;
                    if rad > 1. {
                        delz = ((rad * rad - 1.).sqrt()) / scale.z;
                    }
                    if (point.z - cz as f32).abs() <= delz {
                        let temp_distance = imod_point3d_scale_distance(point, pnt, &scale);
                        if distance < 0. || distance > temp_distance {
                            distance = temp_distance;
                            index.contour = i as i32;
                            index.point = pindex as i32;
                        }
                    }
                }
            }
        } else {
            if !twod && cont.flags & ICONT_WILD == 0 && cz != (cont.pts[0].z + 0.5).floor() as i32 {
                continue;
            }
            for (pindex, point) in cont.pts.iter().enumerate() {
                if (twod || cz == (point.z + 0.5).floor() as i32)
                    && (pnt.x - point.x).abs() < selsize
                    && (pnt.y - point.y).abs() < selsize
                {
                    let temp_distance = imod_distance(point.x, point.y, pnt);
                    if distance < 0. || distance > temp_distance {
                        distance = temp_distance;
                        index.contour = i as i32;
                        index.point = pindex as i32;
                    }
                }
            }
        }
    }
    distance
}

/// `imodAllObjNearest` (`imod_edit.cpp:205`).  `imod` is the source's `vi->imod`.
pub fn imod_all_obj_nearest(
    vi: &mut ImodView,
    imod: &mut Imod,
    index: &mut Iindex,
    pnt: &Ipoint,
    selsize: f32,
    ctime: i32,
    mat: Option<&Imat>,
    attach_to_on_obj: bool,
) -> f32 {
    let mut distance = -1.;
    imod.cindex.contour = -1;
    imod.cindex.point = -1;
    for i in 0..imod.obj.len() {
        if attach_to_on_obj && iobj_off(imod.obj[i].flags) != 0 {
            continue;
        }
        index.object = i as i32;
        let temp_distance = imod_obj_nearest(vi, &imod.obj[i], index, pnt, selsize, ctime, mat);
        if temp_distance < 0. {
            continue;
        }
        if distance < 0. || distance > temp_distance {
            distance = temp_distance;
            imod.cindex = *index;
            if let Some(point) = imod
                .obj
                .get(index.object as usize)
                .and_then(|o| o.cont.get(index.contour as usize))
                .and_then(|c| c.pts.get(index.point as usize))
            {
                vi.xmouse = point.x;
                vi.ymouse = point.y;
                if mat.is_some() {
                    vi.zmouse = point.z;
                }
            }
        }
    }
    distance
}

/// `imod_contour_move` (`imod_edit.cpp:261`).  `vi` replaces source global `App->cvi`.
pub fn imod_contour_move(vi: &mut ImodView, imod: &mut Imod, ob: i32, undo: &mut dyn ImodEditUndo) {
    let oldob = imod.cindex.object;
    let oldco = imod.cindex.contour;
    if oldob < 0
        || oldco < 0
        || oldob as usize >= imod.obj.len()
        || oldco as usize >= imod.obj[oldob as usize].cont.len()
        || ob == oldob
        || ob > imod.obj.len() as i32
        || ob < 0
    {
        return;
    }
    let ocont = imod.obj[oldob as usize].cont[oldco as usize].clone();
    if istore_count_cont_surf_items(&imod.obj[oldob as usize].store, oldco, 0) != 0 {
        undo.object_prop_chg(ob);
        let (old_obj, new_obj) = if oldob < ob {
            let (a, b) = imod.obj.split_at_mut(ob as usize);
            (&mut a[oldob as usize], &mut b[0])
        } else {
            let (a, b) = imod.obj.split_at_mut(oldob as usize);
            (&mut b[0], &mut a[ob as usize])
        };
        let new_contsize = new_obj.cont.len() as i32;
        let _ =
            istore_copy_cont_surf_items(&old_obj.store, &mut new_obj.store, oldco, new_contsize, 0);
    }
    let new_contsize = imod.obj[ob as usize].cont.len() as i32;
    undo.contour_move(oldob, 0, ob, new_contsize);
    let _ = imod_object_add_contour(&mut imod.obj[ob as usize], ocont);
    let _ = imod_object_remove_contour(&mut imod.obj[oldob as usize], oldco);
}

/// `imodMoveAllContours` (`imod_edit.cpp:301`).
pub fn imod_move_all_contours(
    vi: &mut ImodView,
    imod: &mut Imod,
    ob_new: i32,
    undo: &mut dyn ImodEditUndo,
) {
    let oldob = imod.cindex.object;
    if oldob < 0
        || oldob as usize >= imod.obj.len()
        || ob_new < 0
        || ob_new as usize >= imod.obj.len()
        || oldob == ob_new
    {
        return;
    }
    undo.object_prop_chg(oldob);
    undo.object_prop_chg(ob_new);
    if imod.obj[oldob as usize].cont.is_empty() && !imod.obj[oldob as usize].mesh.is_empty() {
        let meshes = imod.obj[oldob as usize].mesh.clone();
        for mesh in &meshes {
            let _ = imodel_mesh_add(Some(mesh), &mut imod.obj[ob_new as usize].mesh);
        }
        return;
    }
    let (old_obj, new_obj) = if oldob < ob_new {
        let (a, b) = imod.obj.split_at_mut(ob_new as usize);
        (&mut a[oldob as usize], &mut b[0])
    } else {
        let (a, b) = imod.obj.split_at_mut(oldob as usize);
        (&mut b[0], &mut a[ob_new as usize])
    };
    if !old_obj.store.is_empty() {
        for co in 0..=old_obj.surfsize {
            let _ = istore_copy_cont_surf_items(&old_obj.store, &mut new_obj.store, co, co, 1);
        }
        for co in 0..old_obj.cont.len() {
            let _ = istore_copy_cont_surf_items(
                &old_obj.store,
                &mut new_obj.store,
                co as i32,
                new_obj.cont.len() as i32 + co as i32,
                0,
            );
        }
        old_obj.store.clear();
    }
    undo.all_contour_move(oldob, ob_new);
    while !old_obj.cont.is_empty() {
        let cont = old_obj.cont.remove(0);
        let _ = imod_object_add_contour(new_obj, cont);
    }
}

/// `imodFillInContourZ` (`imod_edit.cpp:345`).
pub fn imod_fill_in_contour_z(
    vi: &mut ImodView,
    cont: &mut Icont,
    ob_num: i32,
    co_num: i32,
    cur_point: &mut i32,
    undo: &mut dyn ImodEditUndo,
) -> bool {
    let mut first = true;
    let mut ptb = 0;
    while ptb < cont.pts.len().saturating_sub(1) {
        let zcur = (cont.pts[ptb].z + 0.5).floor() as i32;
        let znext = (cont.pts[ptb + 1].z + 0.5).floor() as i32;
        let zfill = if zcur - znext > 1 {
            zcur - 1
        } else if znext - zcur > 1 {
            zcur + 1
        } else {
            zcur
        };
        if zcur != zfill {
            if first {
                undo.contour_data_chg(ob_num, co_num);
            }
            first = false;
            let cur = cont.pts[ptb];
            let next = cont.pts[ptb + 1];
            let new_pt = Ipoint {
                z: zfill as f32,
                x: cur.x + (next.x - cur.x) * (zfill as f32 - cur.z) / (next.z - cur.z),
                y: cur.y + (next.y - cur.y) * (zfill as f32 - cur.z) / (next.z - cur.z),
            };
            let _ = imod_point_add(cont, Some(new_pt), ptb as i32 + 1);
            if (ptb as i32) < *cur_point {
                *cur_point += 1;
            }
        }
        ptb += 1;
    }
    !first
}

/// Static `contourSublength` (`imod_edit.cpp:425`).
fn contour_sublength(cont: &Icont, p1: i32, mut p2: i32) -> f32 {
    if p2 < p1 {
        p2 += cont.pts.len() as i32;
    }
    let mut lsum = 0.;
    let mut pt = p1;
    for _ in p1..=p2 {
        let next = (pt + 1) % cont.pts.len() as i32;
        lsum += imod_point_distance(&cont.pts[pt as usize], &cont.pts[next as usize]) as f64;
        pt = next;
    }
    lsum as f32
}

/// `imodTrimContourLoops` (`imod_edit.cpp:385`).
pub fn imod_trim_contour_loops(cont: &mut Icont, open_obj: i32) {
    let open_cont = open_obj != 0 || cont.flags & ICONT_OPEN != 0;
    let mut found_loop = true;
    while found_loop {
        found_loop = false;
        let num_seg = cont.pts.len() as i32 - i32::from(!open_cont);
        let mut inner_lim = num_seg - i32::from(!open_cont);
        'outer: for sega in 0..num_seg - 2 {
            for segb in sega + 2..inner_lim {
                let nextb = (segb + 1) % cont.pts.len() as i32;
                if imod_point_intersect(
                    &cont.pts[sega as usize],
                    &cont.pts[(sega + 1) as usize],
                    &cont.pts[segb as usize],
                    &cont.pts[nextb as usize],
                ) != 0
                {
                    let swap = !open_cont
                        && contour_sublength(cont, sega + 1, segb)
                            > contour_sublength(cont, nextb, sega);
                    let Some(mut new_cont) = imod_contour_break(cont, sega + 1, segb) else {
                        return;
                    };
                    if swap {
                        let mut tmp_cont = Icont::default();
                        let _ = imod_contour_copy(cont, &mut tmp_cont);
                        let _ = imod_contour_copy(&new_cont, cont);
                        let _ = imod_contour_copy(&tmp_cont, &mut new_cont);
                    }
                    found_loop = true;
                    break 'outer;
                }
            }
            inner_lim = num_seg;
        }
    }
}

/// `imodSelectionListAdd` et al store `ViewInfo::selectionList` in this
/// source-local state until `imodview.cpp` exposes that C list field directly.
#[derive(Clone, Debug, Default)]
pub struct ImodEditSelection {
    pub selection_list: Vec<Iindex>,
}

/// Static `dumpSelectionList` (`imod_edit.cpp:439`).
fn dump_selection_list(selection: &ImodEditSelection) {
    if crate::imod::three_dmod::imod::imod_debug('S') {
        for index in &selection.selection_list {
            imod_trace(
                'S',
                &format!("{} {} {}", index.object, index.contour, index.point),
            );
        }
    }
}

/// `imodSelectionListAdd` (`imod_edit.cpp:451`).
pub fn imod_selection_list_add(selection: &mut ImodEditSelection, new_index: Iindex) {
    let multi_object = true;
    if !selection.selection_list.is_empty()
        && !multi_object
        && selection.selection_list[0].object != new_index.object
    {
        selection.selection_list.clear();
    }
    for index in &mut selection.selection_list {
        if index.object == new_index.object && index.contour == new_index.contour {
            index.point = new_index.point;
            imod_trace(
                'S',
                &format!(
                    "update {} {} {}",
                    new_index.object, new_index.contour, new_index.point
                ),
            );
            return;
        }
    }
    imod_trace(
        'S',
        &format!(
            "adding {} {} {}",
            new_index.object, new_index.contour, new_index.point
        ),
    );
    selection.selection_list.push(new_index);
    dump_selection_list(selection);
}

/// `imodSelectionListClear` (`imod_edit.cpp:486`).
pub fn imod_selection_list_clear(selection: &mut ImodEditSelection) -> i32 {
    let retval = selection.selection_list.len() as i32;
    selection.selection_list.clear();
    imod_trace('S', "List cleared");
    retval
}

/// `imodSelectionListQuery` (`imod_edit.cpp:495`).
pub fn imod_selection_list_query(selection: &ImodEditSelection, ob: i32, co: i32) -> i32 {
    for index in &selection.selection_list {
        if index.object == ob && (index.contour == co || co < 0) {
            imod_trace('S', &format!("Query returns {}", index.point));
            return index.point;
        }
    }
    -2
}

/// `imodNumSelectedObjects` (`imod_edit.cpp:514`).
pub fn imod_num_selected_objects(
    selection: &ImodEditSelection,
    imod: &Imod,
    min_ob: &mut i32,
    max_ob: &mut i32,
) -> i32 {
    let mut num = 0;
    for ob in 0..imod.obj.len() as i32 {
        if imod_selection_list_query(selection, ob, -1) > -2 || ob == imod.cindex.object {
            if num == 0 {
                *min_ob = ob;
            }
            num += 1;
            *max_ob = ob;
        }
    }
    num
}

/// `imodSelectionListRemove` (`imod_edit.cpp:533`).
pub fn imod_selection_list_remove(selection: &mut ImodEditSelection, ob: i32, co: i32) {
    if let Some(i) = selection
        .selection_list
        .iter()
        .position(|index| index.object == ob && index.contour == co)
    {
        selection.selection_list.remove(i);
        imod_trace(
            'S',
            &format!(
                "Removing item {i}, leaves {}",
                selection.selection_list.len()
            ),
        );
    }
}

/// `imodSelectionNewCurPoint` (`imod_edit.cpp:551`).
pub fn imod_selection_new_cur_point(
    selection: &mut ImodEditSelection,
    imod: &mut Imod,
    ind_save: Iindex,
    control_down: i32,
) {
    if control_down != 0 {
        let contourless = ind_save.object >= 0
            && imod
                .obj
                .get(ind_save.object as usize)
                .is_some_and(|obj| obj.cont.is_empty() && !obj.mesh.is_empty());
        if selection.selection_list.is_empty() && (ind_save.contour >= 0 || contourless) {
            imod_selection_list_add(selection, ind_save);
        }
        if imod_selection_list_query(selection, imod.cindex.object, imod.cindex.contour) < -1 {
            imod_selection_list_add(selection, imod.cindex);
        } else {
            imod_selection_list_remove(selection, imod.cindex.object, imod.cindex.contour);
            if let Some(indp) = selection.selection_list.last() {
                imod.cindex = *indp;
            }
        }
    } else {
        imod_selection_list_clear(selection);
    }
}

/// `imodContourIsPlanar` (`imod_edit.cpp:580`).
pub fn imod_contour_is_planar(cont: &Icont, plane: i32) -> bool {
    if cont.pts.len() < 2 {
        return false;
    }
    if plane == Z_SLICE_BOX {
        return cont.flags & ICONT_WILD == 0;
    }
    if plane == X_SLICE_BOX {
        let first = (cont.pts[0].x + 0.5).floor() as i32;
        cont.pts[1..]
            .iter()
            .all(|point| (point.x + 0.5).floor() as i32 == first)
    } else {
        let first = (cont.pts[0].y + 0.5).floor() as i32;
        cont.pts[1..]
            .iter()
            .all(|point| (point.y + 0.5).floor() as i32 == first)
    }
}

/// `imodCheckSurfForNewCont` (`imod_edit.cpp:603`).
pub fn imod_check_surf_for_new_cont(
    obj: &Iobj,
    cont: Option<&Icont>,
    time: i32,
    plane: i32,
) -> i32 {
    let mut checked_surfs = BTreeSet::new();
    let cur_surf = cont.map_or(-1, |cont| cont.surf);
    for candidate in cont.into_iter().chain(obj.cont.iter().rev()) {
        if candidate.time != time || checked_surfs.contains(&candidate.surf) {
            continue;
        }
        if imod_contour_is_planar(candidate, plane) {
            checked_surfs.insert(candidate.surf);
            if imod_surface_is_planar(obj, candidate.surf, time, plane) > 0 {
                imod_trace(
                    'P',
                    &format!(
                        "imodCheckSurfForNewCont found surface {} OK for plane {plane}",
                        candidate.surf
                    ),
                );
                return candidate.surf;
            }
        }
    }
    if plane == Z_SLICE_BOX && imod_surface_is_planar(obj, 0, time, plane) != 0 {
        return 0;
    }
    if plane != Z_SLICE_BOX
        && cur_surf > 0
        && imod_surface_is_planar(obj, cur_surf, time, plane) != 0
    {
        return cur_surf;
    }
    -1
}

/// `imodSurfaceIsPlanar` (`imod_edit.cpp:682`).
pub fn imod_surface_is_planar(obj: &Iobj, surface: i32, time: i32, plane: i32) -> i32 {
    let mut retval = -1;
    for cont in &obj.cont {
        if cont.pts.len() > 1 && cont.surf == surface && cont.time == time {
            if !imod_contour_is_planar(cont, plane) {
                retval = 0;
                break;
            }
            retval = 1;
        }
    }
    imod_trace(
        'P',
        &format!("surfaceIsPlanar surf {surface} time {time} plane {plane} return {retval}"),
    );
    retval
}

#[cfg(test)]
mod tests {
    use super::*;
    fn contour(points: &[(f32, f32, f32)]) -> Icont {
        Icont {
            pts: points.iter().map(|&(x, y, z)| Ipoint { x, y, z }).collect(),
            ..Icont::default()
        }
    }
    #[test]
    fn selection_and_planarity_follow_source() {
        let mut state = ImodEditSelection::default();
        imod_selection_list_add(
            &mut state,
            Iindex {
                object: 2,
                contour: 3,
                point: 4,
            },
        );
        imod_selection_list_add(
            &mut state,
            Iindex {
                object: 2,
                contour: 3,
                point: 7,
            },
        );
        assert_eq!(imod_selection_list_query(&state, 2, 3), 7);
        let c = contour(&[(2., 0., 0.), (2.1, 1., 1.)]);
        assert!(imod_contour_is_planar(&c, X_SLICE_BOX));
    }
    #[test]
    fn filling_z_inserts_each_missing_section() {
        let mut c = contour(&[(0., 0., 0.), (4., 8., 4.)]);
        let mut cur = 1;
        struct U;
        impl ImodEditUndo for U {
            fn object_prop_chg(&mut self, _: i32) {}
            fn contour_move(&mut self, _: i32, _: i32, _: i32, _: i32) {}
            fn all_contour_move(&mut self, _: i32, _: i32) {}
            fn contour_data_chg(&mut self, _: i32, _: i32) {}
        }
        let mut u = U;
        assert!(imod_fill_in_contour_z(
            &mut ImodView::default(),
            &mut c,
            0,
            0,
            &mut cur,
            &mut u
        ));
        assert_eq!(c.pts.len(), 5);
        assert_eq!(c.pts[2].z, 2.);
    }
}
