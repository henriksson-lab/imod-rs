//! Translation of `IMOD/libmesh/skinobj.c` -- skins an object with branching
//! and nesting analysis.

use std::cell::Cell;
use std::io::Write;

use crate::imod::libcfshr::b3dutil::{
    ImodFile, b3d_error, b3d_omp_thread_num, num_omp_threads, wall_time,
};
use crate::imod::libcfshr::convexbound::convex_bound;
use crate::imod::libcfshr::simplestat::ls_fit2;
use crate::imod::libimod::icont::{
    ICONT_CONNECT_BOTTOM, ICONT_CONNECT_INVERT, ICONT_CONNECT_TOP, IMOD_CONTOUR_CLOCKWISE,
    IMOD_CONTOUR_COUNTER_CLOCKWISE, Nesting, imod_cont_z_direction, imod_contour_area,
    imod_contour_center_of_mass, imod_contour_check_nesting, imod_contour_delete, imod_contour_dup,
    imod_contour_free_nests, imod_contour_free_z_tables, imod_contour_get_bbox, imod_contour_join,
    imod_contour_length, imod_contour_long_axis, imod_contour_make_z_tables, imod_contour_nearest,
    imod_contour_nest_levels, imod_contour_new, imod_contour_reduce, imod_contour_unique,
    imod_contours_new, imodel_contour_overlap, imodel_contour_scan, imodel_overlap_fractions,
    imodel_scans_overlap,
};
use crate::imod::libimod::imesh::{
    IMESH_CAP_ALL, IMESH_MK_CAP_DOME, IMESH_MK_CAP_TUBE, IMESH_MK_FAST, IMESH_MK_NORM,
    IMESH_MK_SKIP, IMESH_MK_STRAY, IMESH_MK_SURF, IMESH_MK_TIME, IMESH_MK_TUBE, imodel_mesh_add,
};
use crate::imod::libimod::imodel::{
    ICONT_OPEN, IMOD_OBJFLAG_OUT, IMOD_OBJFLAG_SCAT, Icont, Imesh, Iobj, Ipoint,
};
use crate::imod::libimod::iobj::iobj_close;
use crate::imod::libimod::ipoint::{
    imod_point_add, imod_point_append, imod_point_delete, imod_point_distance, imod_point_get_size,
    imod_point_inside_cont, imod_point_normalize,
};
use crate::imod::libimod::istore::{
    CHANGED_3DWIDTH, CHANGED_COLOR, CHANGED_FCOLOR, CHANGED_TRANS, DrawProps,
    istore_connect_number, istore_cont_surf_draw_props, istore_default_draw_props,
    istore_extract_changes, istore_first_change_index, istore_generate_items,
    istore_list_point_props, istore_next_change, istore_point_is_gap,
};
use crate::imod::libmesh::mkmesh::{
    Connector, chunk_mesh_add_index, imesh_contours_cost, imesh_set_skin_flags, join_tube_cont,
    make_cap_mesh, make_connectors, make_tube_cont,
};
use crate::imod::libmesh::remesh::imesh_remesh_normal;
use crate::imod::libmesh::skeletonize::skeletonize;

/// Original: `CONNECT_TOP` (`skinobj.c:22`).
const CONNECT_TOP: u32 = ICONT_CONNECT_TOP;
/// Original: `CONNECT_BOTTOM` (`skinobj.c:23`).
const CONNECT_BOTTOM: u32 = ICONT_CONNECT_BOTTOM;
/// Original: `CONNECT_BOTH` (`skinobj.c:24`).
const CONNECT_BOTH: u32 = ICONT_CONNECT_TOP & ICONT_CONNECT_BOTTOM;

/// Original: `connectTop` (`skinobj.c:26`).
fn connect_top(f: u32) -> u32 {
    f & CONNECT_TOP
}
/// Original: `connectBottom` (`skinobj.c:27`).
fn connect_bottom(f: u32) -> u32 {
    f & CONNECT_BOTTOM
}
/// Original: `connectBoth` (`skinobj.c:28`).
fn connect_both(f: u32) -> bool {
    connect_top(f) != 0 && connect_bottom(f) != 0
}

thread_local! {
    /// Original static `fastmesh` (`skinobj.c:99`).
    static FASTMESH: Cell<i32> = const { Cell::new(0) };
    /// Original static `skinFlags` (`skinobj.c:100`).
    static SKIN_FLAGS: Cell<u32> = const { Cell::new(0) };
    /// Original static `timeval1` (`skinobj.c:108`).
    static TIMEVAL1: Cell<f64> = const { Cell::new(0.) };
    /// Original static `timeval2` (`skinobj.c:108`).
    static TIMEVAL2: Cell<f64> = const { Cell::new(0.) };
    /// Original function-static `doReport` of `skin_report_time` (`skinobj.c:112`).
    static DO_REPORT: Cell<i32> = const { Cell::new(-1) };
    /// Original function-static `numWarn` of `imeshSkinObject` (`skinobj.c:216`).
    static NUM_WARN: Cell<i32> = const { Cell::new(0) };
}

/// Original static `imeshDefaultCallback` (`skinobj.c:102`).
pub fn imesh_default_callback(in_status: i32) -> i32 {
    let _dummy = in_status;
    0
}

/// Original: `skin_report_time` (`skinobj.c:110`).
pub fn skin_report_time(string: &str) {
    if DO_REPORT.with(|c| c.get()) < 0 {
        DO_REPORT.with(|c| c.set(i32::from(std::env::var_os("IMOD_MESH_TIMING").is_some())));
    }
    if DO_REPORT.with(|c| c.get()) == 0 {
        return;
    }
    TIMEVAL2.with(|c| c.set(wall_time()));
    let elapsed = (TIMEVAL2.with(|c| c.get()) - TIMEVAL1.with(|c| c.get())) as f32;
    let _ = writeln!(&mut ImodFile::Stdout, "{}: {:.3}", string, elapsed);
    TIMEVAL1.with(|c| c.set(TIMEVAL2.with(|c| c.get())));
}

/// Original static `getnextz` (`skinobj.c:126`).
pub fn getnextz(zlist: &[i32], zlsize: i32, cz: i32) -> i32 {
    for z in 0..(zlsize - 1) as usize {
        if cz == zlist[z] {
            return zlist[z + 1];
        }
    }
    /* DNM: this will work the same as when not connecting skips */
    if cz == zlist[(zlsize - 1) as usize] {
        return zlist[(zlsize - 1) as usize] + 1;
    }

    /* 11/15/08: nobody is testing for negative return, just return past the max */
    zlist[(zlsize - 1) as usize] + 1
}

/// Original static `dump_lists` (`skinobj.c:143`).
///
/// Dead in the source as well (every call site is commented out).
pub fn dump_lists(
    message: &str,
    blist: &[i32],
    nb: i32,
    tlist: &[i32],
    nt: i32,
    blook: Option<&[i32]>,
    tlook: Option<&[i32]>,
) {
    let mut out = ImodFile::Stdout;
    let _ = write!(&mut out, "{}:\nbottom: ", message);
    for i in 0..nb as usize {
        match blook {
            Some(blook) => {
                let _ = write!(&mut out, " {}", blook[blist[i] as usize] + 1);
            }
            None => {
                let _ = write!(&mut out, " {}", blist[i] + 1);
            }
        }
    }
    let _ = write!(&mut out, "\ntop: ");
    for i in 0..nt as usize {
        match tlook {
            Some(tlook) => {
                let _ = write!(&mut out, " {}", tlook[tlist[i] as usize] + 1);
            }
            None => {
                let _ = write!(&mut out, " {}", tlist[i] + 1);
            }
        }
    }
    let _ = writeln!(&mut out);
}

/// Original static `mesh_open_obj` (`skinobj.c:163`).
///
/// Connects open contours like a surface.
#[allow(clippy::too_many_arguments)]
pub fn mesh_open_obj(
    obj: &mut Iobj,
    scale: &Ipoint,
    incz: i32,
    flags: u32,
    skip_passes: i32,
    zmin: i32,
    zmax: i32,
    mut contz: Vec<i32>,
    mut zlist: Vec<i32>,
    zlsize: i32,
    mut numatz: Vec<i32>,
    mut contatz: Vec<Vec<i32>>,
    pmin: Vec<Ipoint>,
    pmax: Vec<Ipoint>,
) -> i32 {
    let mut inside = 0;
    let mut indmin: usize = 0;
    let mut ind_single: usize = 0;

    if obj.flags & IMOD_OBJFLAG_OUT != 0 {
        inside = 1;
    }

    FASTMESH.with(|c| c.set(1));
    imesh_set_skin_flags(
        SKIN_FLAGS.with(|c| c.get()) as i32,
        FASTMESH.with(|c| c.get()),
    );

    let co0 = (obj.cont.len() as i32 + zmax - zmin) / (zmax + 1 - zmin);
    let mut matsize = (co0 * co0) as usize;
    let mut sepmat: Vec<f32> = vec![0.; matsize];

    /* Do multiple passes for connections farther apart in Z */
    for zpass in 1..=skip_passes {
        for iz in 0..(zmax - zmin) as usize {
            let mut nextz = iz as i32 + zmin;
            for _i in 0..zpass {
                if flags & IMESH_MK_SKIP != 0 {
                    nextz = getnextz(&zlist, zlsize, nextz);
                } else {
                    nextz += incz;
                }
                if nextz > zmax {
                    break;
                }
            }
            if nextz > zmax {
                continue;
            }
            let nextiz = (nextz - zmin) as usize;
            let needmat = (numatz[iz] * numatz[nextiz]) as usize;
            if needmat > matsize {
                sepmat.resize(needmat, 0.);
                matsize = needmat;
            }

            for i in 0..numatz[iz] as usize {
                let co = contatz[iz][i] as usize;
                if connect_top(obj.cont[co].flags) != 0 {
                    continue;
                }

                for j in 0..numatz[nextiz] as usize {
                    let eco = contatz[nextiz][j] as usize;
                    if connect_bottom(obj.cont[eco].flags) != 0 {
                        continue;
                    }
                    sepmat[i + j * numatz[iz] as usize] = -1.;
                    if obj.cont[co].pts.is_empty() {
                        continue;
                    }
                    if obj.cont[eco].pts.is_empty() {
                        continue;
                    }
                    if (flags & IMESH_MK_SURF) != 0 && obj.cont[co].surf != obj.cont[eco].surf {
                        continue;
                    }
                    if (flags & IMESH_MK_TIME) != 0 && obj.cont[co].time != obj.cont[eco].time {
                        continue;
                    }

                    /* for a valid pair, compute a separation factor */
                    let mut xlap =
                        segment_separation(pmin[co].x, pmax[co].x, pmin[eco].x, pmax[eco].x);
                    let ylap = segment_separation(pmin[co].y, pmax[co].y, pmin[eco].y, pmax[eco].y);
                    if ylap > xlap {
                        xlap = ylap;
                    }
                    sepmat[i + j * numatz[iz] as usize] = xlap;
                }
            }

            loop {
                /* Loop: find pair with minimum separation and mesh them; keep track
                of a minimum for pairs including single points and pairs not */
                let mut minsep = 1.0e30f32;
                let mut single_sep = 1.0e30f32;
                for i in 0..needmat {
                    if sepmat[i] >= 0. {
                        let co = contatz[iz][i % numatz[iz] as usize] as usize;
                        let eco = contatz[nextiz][i / numatz[iz] as usize] as usize;
                        if obj.cont[co].pts.len() == 1 || obj.cont[eco].pts.len() == 1 {
                            if sepmat[i] < single_sep {
                                single_sep = sepmat[i];
                                ind_single = i;
                            }
                        } else if sepmat[i] >= 0. && sepmat[i] < minsep {
                            minsep = sepmat[i];
                            indmin = i;
                        }
                    }
                }

                /* If only a single was found, use it. */
                if minsep > 1.0e20 && single_sep < 1.0e20 {
                    indmin = ind_single;
                    minsep = single_sep;
                }

                /* Done if no more pairs available */
                if minsep > 1.0e20 {
                    break;
                }
                let i = indmin % numatz[iz] as usize;
                let j = indmin / numatz[iz] as usize;
                let co = contatz[iz][i] as usize;
                let eco = contatz[nextiz][j] as usize;
                obj.cont[co].flags |= ICONT_OPEN | CONNECT_TOP;
                obj.cont[eco].flags |= ICONT_OPEN | CONNECT_BOTTOM;
                let mut bc = obj.cont[co].clone();
                let mut tc = obj.cont[eco].clone();
                let nmesh = imesh_contours_cost(
                    obj, &mut bc, &mut tc, scale, inside, co as i32, eco as i32,
                );
                obj.cont[co].flags = bc.flags;
                obj.cont[eco].flags = tc.flags;
                let surf = obj.cont[co].surf;
                let time = obj.cont[co].time;
                add_mesh_to_object(obj, surf, time, nmesh, flags);

                /* mark all pairs involving these two conts as unavailable */
                for k in 0..numatz[nextiz] as usize {
                    sepmat[i + k * numatz[iz] as usize] = -1.;
                }
                for k in 0..numatz[iz] as usize {
                    sepmat[k + j * numatz[iz] as usize] = -1.;
                }
            }
        }
    }

    if flags & IMESH_MK_NORM != 0 {
        let mut size = obj.mesh.len() as i32;
        let mut meshes = std::mem::take(&mut obj.mesh);
        obj.mesh = imesh_remesh_normal(&mut meshes, &mut size, Some(scale), 0).unwrap_or_default();
    }

    imod_contour_free_z_tables(
        &mut numatz,
        &mut contatz,
        &mut contz,
        &mut zlist,
        zmin,
        zmax,
    );
    drop(sepmat);
    drop(pmin);
    drop(pmax);
    0
}

/// Original static `segment_separation` (`skinobj.c:286`).
pub fn segment_separation(l1: f32, u1: f32, l2: f32, u2: f32) -> f32 {
    let mut minlen = (u1 - l1) as i32;
    if ((u2 - l2) as i32) < minlen {
        minlen = (u2 - l2) as i32;
    }
    if (l1 >= l2 && u1 <= u2) || (l2 >= l1 && u2 <= u1) {
        return 0.;
    }
    if l1 > u2 {
        return minlen as f32 + l1 - u2;
    }
    if l2 > u1 {
        return minlen as f32 + l2 - u1;
    }

    if u1 > u2 {
        minlen = (u1 - u2) as i32;
        if ((l1 - l2) as i32) < minlen {
            minlen = (l1 - l2) as i32;
        }
    } else {
        minlen = (u2 - u1) as i32;
        if ((l2 - l1) as i32) < minlen {
            minlen = (l2 - l1) as i32;
        }
    }
    minlen as f32
}

/// Original static `mesh_open_tube_obj` (`skinobj.c:313`).
///
/// Meshes an open object as tubes.
pub fn mesh_open_tube_obj(obj: &mut Iobj, scale: &Ipoint, flags: u32, mesh_diameter: f64) -> i32 {
    let mut def_props = DrawProps::default();
    let mut cont_props = DrawProps::default();
    let mut pt_props = DrawProps::default();
    let mut last_props: DrawProps;
    let mut state_flags: i32 = 0;
    let mut change_flags: i32 = 0;
    let state_test = CHANGED_COLOR | CHANGED_FCOLOR | CHANGED_TRANS;
    let mut nrot = Ipoint::default();
    let mut dome_rot1 = Ipoint::default();
    let mut dome_rot2 = Ipoint::default();
    let mut dome_cap1 = Ipoint::default();
    let mut dome_cap2 = Ipoint::default();
    let mut dome_cont1: Option<Vec<Icont>> = None;
    let mut dome_cont2: Option<Vec<Icont>> = None;
    let mut num_dome1: i32 = 0;
    let mut num_dome2: i32 = 0;

    istore_default_draw_props(obj, &mut def_props);

    obj.mesh = Vec::new();

    for co in 0..obj.cont.len() {
        if obj.cont[co].pts.len() < 2 {
            continue;
        }
        let cont = obj.cont[co].clone();
        let mut clst = match imod_contours_new(cont.pts.len() as i32) {
            Some(c) => c,
            None => continue,
        };

        let mut last_state = istore_cont_surf_draw_props(
            &obj.store,
            &def_props,
            &mut cont_props,
            co as i32,
            cont.surf,
            &mut state_flags,
            &mut change_flags,
        );
        let skin_flags = SKIN_FLAGS.with(|c| c.get());
        if skin_flags & IMESH_MK_SURF == 0 {
            state_flags = last_state;
        }
        pt_props = cont_props;
        let mut cursor: usize = 0;
        let mut next_change = istore_first_change_index(&cont.store);

        /* DNM 8/25/03: Set tube diameter here instead of in makeTubeCont and make
        number of segments variable within limits */
        let mut tube_diameter = mesh_diameter as f32;

        for pt in 0..cont.pts.len() {
            if pt as i32 == next_change {
                next_change = istore_next_change(
                    &cont.store,
                    &mut cursor,
                    &cont_props,
                    &mut pt_props,
                    &mut state_flags,
                    &mut change_flags,
                );
            }
            if mesh_diameter <= 0. {
                if mesh_diameter == 0. {
                    tube_diameter = pt_props.linewidth as f32;
                } else if mesh_diameter < -1.0001 {
                    tube_diameter = pt_props.symsize as f32;
                } else {
                    tube_diameter = 2.0f32 * imod_point_get_size(obj, &cont, pt as i32);
                }
                tube_diameter = if 1.0f32 < tube_diameter {
                    tube_diameter
                } else {
                    1.0f32
                };
            }
            let mut slices = (tube_diameter / 2.) as i32;
            if slices < 12 {
                slices = 12;
            }
            if slices > 50 {
                slices = 50;
            }

            /* DNM: simplify and include scale in normal calculation */

            let mut ppt = pt as i32 - 1;
            if ppt < 0 {
                ppt = 0;
            }
            let mut npt = pt as i32 + 1;
            if npt == cont.pts.len() as i32 {
                npt = pt as i32;
            }

            nrot.x = scale.x * (cont.pts[npt as usize].x - cont.pts[ppt as usize].x);
            nrot.y = scale.y * (cont.pts[npt as usize].y - cont.pts[ppt as usize].y);
            nrot.z = scale.z * (cont.pts[npt as usize].z - cont.pts[ppt as usize].z);

            imod_point_normalize(&mut nrot);
            let loc = cont.pts[pt];
            make_tube_cont(&mut clst[pt], &loc, &nrot, scale, tube_diameter, slices);

            /* Get the contours for dome caps if at start or end */
            if pt == 0 && (flags & IMESH_MK_CAP_DOME) != 0 && cont_props.no_cap == 0 {
                dome_cont1 = make_dome_conts(
                    &clst[pt],
                    &cont.pts[pt],
                    &nrot,
                    scale,
                    tube_diameter,
                    -1.,
                    &mut num_dome1,
                    &mut dome_cap1,
                );
                dome_rot1 = nrot;
            }
            if pt == cont.pts.len() - 1
                && (flags & IMESH_MK_CAP_DOME) != 0
                && cont_props.no_cap == 0
            {
                dome_cont2 = make_dome_conts(
                    &clst[pt],
                    &cont.pts[pt],
                    &nrot,
                    scale,
                    tube_diameter,
                    1.,
                    &mut num_dome2,
                    &mut dome_cap2,
                );
                dome_rot2 = nrot;
            }
        }

        /* Restart the state for drawing */
        last_state = istore_cont_surf_draw_props(
            &obj.store,
            &def_props,
            &mut cont_props,
            co as i32,
            cont.surf,
            &mut state_flags,
            &mut change_flags,
        );
        if SKIN_FLAGS.with(|c| c.get()) & IMESH_MK_SURF == 0 {
            state_flags = last_state;
        }
        pt_props = cont_props;
        let mut cursor: usize = 0;
        let mut next_change = istore_first_change_index(&cont.store);
        pt_props.gap = 0;
        if next_change == 0 {
            next_change = istore_next_change(
                &cont.store,
                &mut cursor,
                &cont_props,
                &mut pt_props,
                &mut state_flags,
                &mut change_flags,
            );
        }

        /* DNM 8/25/02: cap if flag is set*/
        if (flags & IMESH_MK_CAP_DOME) != 0 && num_dome1 != 0 && cont_props.no_cap == 0 {
            /* At start, cap from point up to first regular contour for dome */
            let dc1 = dome_cont1.as_mut().unwrap();
            let nmesh = make_cap_mesh(
                &dc1[(num_dome1 - 1) as usize],
                &dome_cap1,
                1,
                &pt_props,
                state_flags,
                state_test,
            );
            add_mesh_to_object(obj, cont.surf, cont.time, nmesh, flags);
            for pt in (0..num_dome1).rev() {
                let dc1 = dome_cont1.as_ref().unwrap();
                let nextcont: Icont = if pt != 0 {
                    dc1[(pt - 1) as usize].clone()
                } else {
                    clst[0].clone()
                };
                let mut p1 = pt_props;
                let mut p2 = pt_props;
                let nmesh = join_tube_cont(
                    &dc1[pt as usize],
                    &nextcont,
                    &dome_rot1,
                    &mut p1,
                    state_flags,
                    &mut p2,
                    state_flags,
                );
                add_mesh_to_object(obj, cont.surf, cont.time, nmesh, flags);
            }
            dome_cont1 = None;
        } else if (flags & IMESH_MK_CAP_TUBE) != 0 && cont_props.no_cap == 0 {
            let nmesh = make_cap_mesh(
                &clst[0],
                &cont.pts[0],
                1,
                &pt_props,
                state_flags,
                state_test,
            );
            add_mesh_to_object(obj, cont.surf, cont.time, nmesh, flags);
        }

        for pt in 0..cont.pts.len() - 1 {
            let last_state_l = state_flags;
            last_props = pt_props;
            pt_props.gap = 0;
            if pt as i32 + 1 == next_change {
                next_change = istore_next_change(
                    &cont.store,
                    &mut cursor,
                    &cont_props,
                    &mut pt_props,
                    &mut state_flags,
                    &mut change_flags,
                );
            }
            if last_props.gap != 0 {
                continue;
            }

            nrot.x = scale.x * (cont.pts[pt + 1].x - cont.pts[pt].x);
            nrot.y = scale.y * (cont.pts[pt + 1].y - cont.pts[pt].y);
            nrot.z = scale.z * (cont.pts[pt + 1].z - cont.pts[pt].z);

            imod_point_normalize(&mut nrot);
            let mut p1 = last_props;
            let mut p2 = pt_props;
            let (a, b) = clst.split_at(pt + 1);
            let nmesh = join_tube_cont(
                &a[pt],
                &b[0],
                &nrot,
                &mut p1,
                last_state_l,
                &mut p2,
                state_flags,
            );
            add_mesh_to_object(obj, cont.surf, cont.time, nmesh, flags);
        }

        if (flags & IMESH_MK_CAP_DOME) != 0 && num_dome2 != 0 && cont_props.no_cap == 0 {
            /* At end, cap from last regular contour out to point for dome */
            for pt in 0..num_dome2 {
                let dc2 = dome_cont2.as_ref().unwrap();
                let nextcont: Icont = if pt != 0 {
                    dc2[(pt - 1) as usize].clone()
                } else {
                    clst[cont.pts.len() - 1].clone()
                };
                let mut p1 = pt_props;
                let mut p2 = pt_props;
                let nmesh = join_tube_cont(
                    &nextcont,
                    &dc2[pt as usize],
                    &dome_rot2,
                    &mut p1,
                    state_flags,
                    &mut p2,
                    state_flags,
                );
                add_mesh_to_object(obj, cont.surf, cont.time, nmesh, flags);
            }
            let dc2 = dome_cont2.as_ref().unwrap();
            let nmesh = make_cap_mesh(
                &dc2[(num_dome2 - 1) as usize],
                &dome_cap2,
                0,
                &pt_props,
                state_flags,
                state_test,
            );
            add_mesh_to_object(obj, cont.surf, cont.time, nmesh, flags);
            dome_cont2 = None;
        } else if (flags & IMESH_MK_CAP_TUBE) != 0 && cont_props.no_cap == 0 {
            let last = cont.pts.len() - 1;
            let nmesh = make_cap_mesh(
                &clst[last],
                &cont.pts[last],
                0,
                &pt_props,
                state_flags,
                state_test,
            );
            add_mesh_to_object(obj, cont.surf, cont.time, nmesh, flags);
        }

        clst.clear();
        let _ = (num_dome1, num_dome2);
        num_dome1 = 0;
        num_dome2 = 0;
    }

    let mut size = obj.mesh.len() as i32;
    let mut meshes = std::mem::take(&mut obj.mesh);
    obj.mesh = imesh_remesh_normal(&mut meshes, &mut size, Some(scale), 0).unwrap_or_default();
    0
}

/// Original static `makeDomeConts` (`skinobj.c:510`).
///
/// Makes contours for a hemispherical cap and returns the terminal point also.
#[allow(clippy::too_many_arguments)]
pub fn make_dome_conts(
    last_cont: &Icont,
    last_pt: &Ipoint,
    nrot: &Ipoint,
    scale: &Ipoint,
    diameter: f32,
    direction: f32,
    num_cont: &mut i32,
    cap_pt: &mut Ipoint,
) -> Option<Vec<Icont>> {
    let mut scl_pt = Ipoint::default();
    let num_pos = (last_cont.pts.len() as f64 / 4.).round() as i32;
    let mut conts = imod_contours_new(num_pos - 1);
    let dtor: f64 = 0.017453293;
    *num_cont = 0;
    let conts_v = conts.as_mut()?;
    scl_pt.x = last_pt.x * scale.x;
    scl_pt.y = last_pt.y * scale.y;
    scl_pt.z = last_pt.z * scale.z;

    for i in 1..=num_pos {
        let angle = (dtor * i as f64 * 90.) / num_pos as f64;
        let dist = 0.5 * diameter as f64 * direction as f64 * angle.sin();
        cap_pt.x = ((scl_pt.x as f64 + nrot.x as f64 * dist) / scale.x as f64) as f32;
        cap_pt.y = ((scl_pt.y as f64 + nrot.y as f64 * dist) / scale.y as f64) as f32;
        cap_pt.z = ((scl_pt.z as f64 + nrot.z as f64 * dist) / scale.z as f64) as f32;
        if i < num_pos {
            let radius = (0.5 * diameter as f64 * angle.cos()) as f32;
            let slices = 12.max(50.min(radius as i32));
            let loc = *cap_pt;
            make_tube_cont(
                &mut conts_v[(i - 1) as usize],
                &loc,
                nrot,
                scale,
                2.0f32 * radius,
                slices,
            );
        }
    }
    *num_cont = num_pos - 1;
    conts
}

/// Original static `addMeshToObject` (`skinobj.c:546`).
///
/// Common operations when adding a new mesh to an object's mesh.  The C takes
/// the contour and reads only its `surf` and `time`, which are passed directly
/// here because the contour usually lives inside `obj`.
pub fn add_mesh_to_object(
    obj: &mut Iobj,
    cont_surf: i32,
    cont_time: i32,
    nmesh: Option<Imesh>,
    flags: u32,
) {
    if let Some(mut nmesh) = nmesh {
        nmesh.surf = if flags & IMESH_MK_SURF != 0 {
            cont_surf as i16
        } else {
            0
        };
        nmesh.time = if flags & IMESH_MK_TIME != 0 {
            cont_time as i16
        } else {
            0
        };
        imodel_mesh_add(Some(&nmesh), &mut obj.mesh);
    }
}

/// Original: `MAX_THREADS` (`skinobj.c:556`).
const MAX_THREADS: usize = 16;

/// Original: `imeshSkinObject` (`skinobj.c:573`).
///
/// The main meshing routine for closed or open contours; calls appropriate
/// routines for open contours or tube contours.
#[allow(clippy::too_many_arguments)]
pub fn imesh_skin_object(
    obj: &mut Iobj,
    scale: &Ipoint,
    overlap: f64,
    do_cap: i32,
    cap_skip_zlist: Option<&[i32]>,
    cap_skip_num_z: i32,
    z_increment: i32,
    flags: u32,
    skip_passes: i32,
    tube_diameter: f64,
    in_call_back: Option<fn(i32) -> i32>,
) -> i32 {
    let mut z_min: i32 = 0;
    let mut z_max: i32 = 0;
    let mut z_list_size: i32 = 0;
    let mut max_cont_per_sec: i32 = 0;
    let mut contour_z: Vec<i32> = Vec::new();
    let mut z_list: Vec<i32> = Vec::new();
    let mut num_cont_at_z: Vec<i32> = Vec::new();
    let mut conts_at_z: Vec<Vec<i32>> = Vec::new();
    let mut num_nests: i32 = 0;
    let mut nests: Vec<Nesting> = Vec::new();
    let mut total_points: i32 = 0;
    let mut bot_cen_mass = Ipoint::default();
    let mut top_cen_mass = Ipoint::default();

    /* start the timer if doing reports */
    TIMEVAL1.with(|c| c.set(wall_time()));

    /* Check that the input object is skinable. */
    if obj.flags & IMOD_OBJFLAG_SCAT != 0 {
        return -1;
    }
    obj.mesh = Vec::new();
    if obj.cont.is_empty() {
        return 0;
    }

    if iobj_close(obj.flags) == 0 && (flags & IMESH_MK_TUBE) != 0 {
        return mesh_open_tube_obj(obj, scale, flags, tube_diameter);
    }

    if flags & IMESH_MK_FAST != 0 {
        FASTMESH.with(|c| c.set(0));
    } else {
        FASTMESH.with(|c| c.set(1));
    }

    SKIN_FLAGS.with(|c| c.set(flags));
    imesh_set_skin_flags(flags as i32, FASTMESH.with(|c| c.get()));
    if flags & crate::imod::libimod::imesh::IMESH_MK_NO_WARN != 0 {
        NUM_WARN.with(|c| c.set(-1));
    }

    if imod_contour_make_z_tables(
        obj,
        z_increment,
        CONNECT_BOTH | ICONT_CONNECT_INVERT,
        &mut contour_z,
        &mut z_list,
        &mut num_cont_at_z,
        &mut conts_at_z,
        &mut z_min,
        &mut z_max,
        &mut z_list_size,
        &mut max_cont_per_sec,
    ) != 0
    {
        return -1;
    }

    /* Allocate space for lists of min's max's, and found contours */

    let mut pt_min: Vec<Ipoint> = vec![Ipoint::default(); obj.cont.len()];
    let mut pt_max: Vec<Ipoint> = vec![Ipoint::default(); obj.cont.len()];

    for co_num in 0..obj.cont.len() {
        if !obj.cont[co_num].pts.is_empty() {
            imod_contour_get_bbox(
                Some(&obj.cont[co_num]),
                &mut pt_min[co_num],
                &mut pt_max[co_num],
            );
        }
    }

    /* OPEN OBJECT: GO OFF AND DO IT */
    if iobj_close(obj.flags) == 0 {
        return mesh_open_obj(
            obj,
            scale,
            z_increment,
            flags,
            skip_passes,
            z_min,
            z_max,
            contour_z,
            z_list,
            z_list_size,
            num_cont_at_z,
            conts_at_z,
            pt_min,
            pt_max,
        );
    }

    /* Tell callback function we are just getting started. */
    let callback: fn(i32) -> i32 = in_call_back.unwrap_or(imesh_default_callback);
    let status = callback(1);
    if status != 0 {
        return status;
    }

    /*
     * Prescan contours for faster rendering.  Also add up total points
     */
    let mut scan_conts: Vec<Icont> = Vec::with_capacity(obj.cont.len());
    for co_num in 0..obj.cont.len() {
        scan_conts.push(imodel_contour_scan(Some(&obj.cont[co_num])).unwrap_or_default());
        total_points += obj.cont[co_num].pts.len() as i32;
    }

    /* This gives 2 threads at 6800 points, 6 at 91200, 8 at 170000 */
    let mut num_threads = ((total_points as f64).sqrt() / 55.).round() as i32;
    let max_threads = 8.min(z_max + 1 - z_min);
    num_threads = 1.max(max_threads.min(num_threads));
    num_threads = num_omp_threads(num_threads);
    num_threads = num_threads.min(MAX_THREADS as i32);
    let num_threads = num_threads.max(1) as usize;
    let mut temp_meshes: Vec<Vec<Imesh>> = vec![Vec::new(); num_threads];
    let mut bot_list: Vec<Vec<i32>> = vec![vec![0; obj.cont.len()]; num_threads];
    let mut top_list: Vec<Vec<i32>> = vec![vec![0; obj.cont.len()]; num_threads];

    skin_report_time("Got scan contours");

    /* Get array for pointers to inside/outside information */
    let mut cont_nest_ind: Vec<i32> = vec![-1; obj.cont.len()];

    /*
     *  Mark all contours with no data as connected.
     *  Build lists of inside and outside contours.
     */

    for ind_z in 0..(z_max + 1 - z_min) as usize {
        for jnd_at_z in 0..(num_cont_at_z[ind_z] - 1).max(0) as usize {
            let co_num = conts_at_z[ind_z][jnd_at_z] as usize;
            if obj.cont[co_num].pts.is_empty() {
                obj.cont[co_num].flags |= CONNECT_BOTH;
            }
            if obj.cont[co_num].flags & CONNECT_BOTH != 0 {
                continue;
            }
            for ind_at_z in jnd_at_z + 1..num_cont_at_z[ind_z] as usize {
                let other_co_num = conts_at_z[ind_z][ind_at_z] as usize;
                if obj.cont[other_co_num].pts.is_empty() {
                    obj.cont[other_co_num].flags |= CONNECT_BOTH;
                }
                if obj.cont[other_co_num].flags & CONNECT_BOTH != 0 {
                    continue;
                }
                if (flags & IMESH_MK_TIME) != 0
                    && obj.cont[co_num].time != obj.cont[other_co_num].time
                {
                    continue;
                }

                let mut num_warn = NUM_WARN.with(|c| c.get());
                let err = imod_contour_check_nesting(
                    co_num as i32,
                    other_co_num as i32,
                    &mut scan_conts,
                    &pt_min,
                    &pt_max,
                    &mut nests,
                    &mut cont_nest_ind,
                    &mut num_nests,
                    &mut num_warn,
                );
                NUM_WARN.with(|c| c.set(num_warn));
                if err != 0 {
                    return -1;
                }
            }
        }
    }

    /* Analyze inside and outside contours to determine level */
    imod_contour_nest_levels(&mut nests, &cont_nest_ind, num_nests);

    /* Compose a scan contour excluding inner areas for each outer contour
    at an odd level */
    /* DNM 12/18/01: Need to include even levels also now */
    for nest_ind in 0..num_nests as usize {
        if !nests[nest_ind].inside.is_empty() {
            let co = nests[nest_ind].co as usize;
            let mut inscan = imod_contour_dup(&scan_conts[co]);
            if let Some(inscan_c) = inscan.as_mut() {
                for ind in 0..nests[nest_ind].inside.len() {
                    let other = nests[nest_ind].inside[ind] as usize;
                    let cs2 = scan_conts[other].clone();
                    subtract_scan_contours(inscan_c, &cs2);
                }
            }
            nests[nest_ind].inscan = inscan;
        }
    }

    skin_report_time("Marked contours");

    let status = callback(25); /* quarter of the way done. */
    if status != 0 {
        return status;
    }

    /* Do multiple passes for connections farther apart in Z */
    for z_pass in 1..=skip_passes {
        let mut error = 0;

        for ind_z in 0..(z_max + 1 - z_min) as usize {
            if error != 0 {
                continue;
            }
            let thread = b3d_omp_thread_num() as usize;
            let thread = thread.min(num_threads - 1);
            let mut jnd_at_z = 0usize;
            while jnd_at_z < num_cont_at_z[ind_z] as usize && error == 0 {
                let co_num = conts_at_z[ind_z][jnd_at_z] as usize;
                jnd_at_z += 1;

                if obj.cont[co_num].pts.is_empty() {
                    continue;
                }

                /* use type to tell if already connected. */
                if connect_top(obj.cont[co_num].flags) != 0 {
                    continue;
                }

                obj.cont[co_num].flags |= CONNECT_TOP;

                /* The current contour is now the bottom contour */
                let bot_co = co_num;

                /* define z-sections that connections will be made to; step farther
                in Z for each pass */
                let mut next_z = contour_z[co_num];
                for _list_ind in 0..z_pass {
                    if flags & IMESH_MK_SKIP != 0 {
                        next_z = getnextz(&z_list, z_list_size, next_z);
                    } else {
                        next_z += z_increment;
                    }
                    if next_z > z_list[(z_list_size - 1) as usize] {
                        break;
                    }
                }
                let prev_z = contour_z[co_num];
                let mut found_data = 1;

                let mut num_bot: usize = 1;
                let mut num_top: usize = 0;
                bot_list[thread][0] = co_num as i32;
                let mut found_inside = 0;
                if cont_nest_ind[co_num] >= 0 {
                    /* If inside/outside, find outermost */
                    found_inside = 1;
                    add_whole_nest(
                        cont_nest_ind[co_num],
                        obj,
                        &nests,
                        &cont_nest_ind,
                        CONNECT_TOP,
                        &mut bot_list[thread],
                        &mut num_bot,
                    );
                }

                /* find all contours that link to current contour. */
                while found_data != 0 {
                    found_data = 0;

                    for ind_at_z in 0..num_cont_at_z[(next_z - z_min) as usize] as usize {
                        let top_co_num = conts_at_z[(next_z - z_min) as usize][ind_at_z] as usize;

                        /* Skip contours that are already connected */
                        if obj.cont[top_co_num].pts.is_empty() {
                            continue;
                        }
                        if connect_bottom(obj.cont[top_co_num].flags) != 0 {
                            continue;
                        }
                        if (flags & IMESH_MK_SURF) != 0
                            && obj.cont[bot_co].surf != obj.cont[top_co_num].surf
                        {
                            continue;
                        }
                        if (flags & IMESH_MK_TIME) != 0
                            && obj.cont[bot_co].time != obj.cont[top_co_num].time
                        {
                            continue;
                        }

                        /* See if it overlaps any contour in bottom list */
                        let top_cnct_num = istore_connect_number(&obj.store, top_co_num as i32);
                        for list_ind in 0..num_bot {
                            let cont_ind = bot_list[thread][list_ind] as usize;

                            let bot_cnct_num = istore_connect_number(&obj.store, cont_ind as i32);
                            if top_cnct_num >= 0
                                && bot_cnct_num >= 0
                                && top_cnct_num != bot_cnct_num
                            {
                                continue;
                            }
                            let force_connect = i32::from(
                                top_cnct_num >= 0
                                    && bot_cnct_num >= 0
                                    && top_cnct_num == bot_cnct_num,
                            );

                            let overlaps = imodel_scans_overlap(
                                Some(&scan_conts[cont_ind]),
                                pt_min[cont_ind],
                                pt_max[cont_ind],
                                Some(&scan_conts[top_co_num]),
                                pt_min[top_co_num],
                                pt_max[top_co_num],
                            );
                            if overlaps != 0 || force_connect != 0 {
                                if force_connect == 0 && overlap != 0.0 {
                                    /* contours must overlap by given fraction*/
                                    let mut frac1 = 0.0f32;
                                    let mut frac2 = 0.0f32;
                                    overlap_fractions_at(
                                        &mut scan_conts,
                                        cont_ind,
                                        pt_min[cont_ind],
                                        pt_max[cont_ind],
                                        top_co_num,
                                        pt_min[top_co_num],
                                        pt_max[top_co_num],
                                        &mut frac1,
                                        &mut frac2,
                                    );
                                    if frac1 as f64 <= overlap && frac2 as f64 <= overlap {
                                        continue;
                                    }
                                }
                                found_data = 1;
                                obj.cont[top_co_num].flags |= CONNECT_BOTTOM;
                                top_list[thread][num_top] = top_co_num as i32;
                                num_top += 1;
                                if cont_nest_ind[top_co_num] >= 0 {
                                    found_inside = 1;
                                    add_whole_nest(
                                        cont_nest_ind[top_co_num],
                                        obj,
                                        &nests,
                                        &cont_nest_ind,
                                        CONNECT_BOTTOM,
                                        &mut top_list[thread],
                                        &mut num_top,
                                    );
                                }
                                break;
                            }
                        }
                    }

                    /* if any new top contours were found, need to scan for more
                    bottom contours */
                    if found_data == 0 {
                        break;
                    }
                    for ind_at_z in 0..num_cont_at_z[(prev_z - z_min) as usize] as usize {
                        let top_co_num = conts_at_z[(prev_z - z_min) as usize][ind_at_z] as usize;

                        if obj.cont[top_co_num].pts.is_empty() {
                            continue;
                        }
                        if connect_top(obj.cont[top_co_num].flags) != 0 {
                            continue;
                        }

                        if (flags & IMESH_MK_SURF) != 0
                            && obj.cont[bot_co].surf != obj.cont[top_co_num].surf
                        {
                            continue;
                        }
                        if (flags & IMESH_MK_TIME) != 0
                            && obj.cont[bot_co].time != obj.cont[top_co_num].time
                        {
                            continue;
                        }

                        /* See if it overlaps any contour in top list */
                        let bot_cnct_num = istore_connect_number(&obj.store, top_co_num as i32);
                        for list_ind in 0..num_top {
                            let cont_ind = top_list[thread][list_ind] as usize;
                            let top_cnct_num = istore_connect_number(&obj.store, cont_ind as i32);
                            if top_cnct_num >= 0
                                && bot_cnct_num >= 0
                                && top_cnct_num != bot_cnct_num
                            {
                                continue;
                            }
                            let force_connect = i32::from(
                                top_cnct_num >= 0
                                    && bot_cnct_num >= 0
                                    && top_cnct_num == bot_cnct_num,
                            );

                            let overlaps = imodel_scans_overlap(
                                Some(&scan_conts[cont_ind]),
                                pt_min[cont_ind],
                                pt_max[cont_ind],
                                Some(&scan_conts[top_co_num]),
                                pt_min[top_co_num],
                                pt_max[top_co_num],
                            );
                            if overlaps != 0 || force_connect != 0 {
                                if force_connect == 0 && overlap != 0.0 {
                                    let mut frac1 = 0.0f32;
                                    let mut frac2 = 0.0f32;
                                    overlap_fractions_at(
                                        &mut scan_conts,
                                        cont_ind,
                                        pt_min[cont_ind],
                                        pt_max[cont_ind],
                                        top_co_num,
                                        pt_min[top_co_num],
                                        pt_max[top_co_num],
                                        &mut frac1,
                                        &mut frac2,
                                    );
                                    if frac1 as f64 <= overlap && frac2 as f64 <= overlap {
                                        continue;
                                    }
                                }
                                found_data = 1;
                                obj.cont[top_co_num].flags |= CONNECT_TOP;
                                bot_list[thread][num_bot] = top_co_num as i32;
                                num_bot += 1;
                                if cont_nest_ind[top_co_num] >= 0 {
                                    found_inside = 1;
                                    add_whole_nest(
                                        cont_nest_ind[top_co_num],
                                        obj,
                                        &nests,
                                        &cont_nest_ind,
                                        CONNECT_TOP,
                                        &mut bot_list[thread],
                                        &mut num_bot,
                                    );
                                }
                                break;
                            }
                        }
                    }
                }

                if num_top != 0 && found_inside == 0 {
                    let bot_cont = join_all_contours(
                        obj,
                        &bot_list[thread].clone(),
                        num_bot as i32,
                        num_bot as i32,
                        1,
                        Some(&top_list[thread].clone()),
                        num_top as i32,
                        num_top as i32,
                        None,
                    );
                    let top_cont = join_all_contours(
                        obj,
                        &top_list[thread].clone(),
                        num_top as i32,
                        num_top as i32,
                        -1,
                        Some(&bot_list[thread].clone()),
                        num_bot as i32,
                        num_bot as i32,
                        None,
                    );
                    let surf = obj.cont[co_num].surf;
                    let time = obj.cont[co_num].time;
                    let bco = bot_list[thread][0];
                    let tco = top_list[thread][0];
                    if mesh_contours(
                        obj,
                        &mut temp_meshes[thread],
                        bot_cont,
                        top_cont,
                        surf,
                        time,
                        scale,
                        0,
                        bco,
                        tco,
                    ) != 0
                    {
                        error = 1;
                        break;
                    }
                } else if num_top != 0 {
                    /* There are inside contours to resolve */
                    let mut min_frac: Vec<f32> = vec![0.; num_top * num_bot];
                    let mut max_frac: Vec<f32> = vec![0.; num_top * num_bot];
                    let mut top_inside_list: Vec<i32> = vec![0; num_top];
                    let mut bot_inside_list: Vec<i32> = vec![0; num_bot];
                    let mut top_just_in_list: Vec<i32> = vec![0; num_top];
                    let mut bot_just_in_list: Vec<i32> = vec![0; num_bot];
                    let mut top_out_list: Vec<i32> = vec![0; num_top];
                    let mut bot_out_list: Vec<i32> = vec![0; num_bot];
                    let mut top_used: Vec<i32> = vec![0; num_top];
                    let mut bot_used: Vec<i32> = vec![0; num_bot];
                    let mut top_levels: Vec<i32> = vec![0; num_top];
                    let mut bot_levels: Vec<i32> = vec![0; num_bot];

                    /* Make tables of overlap fractions between top and bottom */
                    for ind in 0..num_top {
                        top_used[ind] = 0;
                        top_levels[ind] = 1;
                        let mut top_cont = scan_conts[top_list[thread][ind] as usize].clone();
                        let mut top_pt_min = pt_min[top_list[thread][ind] as usize];
                        let mut top_pt_max = pt_max[top_list[thread][ind] as usize];
                        if cont_nest_ind[top_list[thread][ind] as usize] >= 0 {
                            let nest =
                                &nests[cont_nest_ind[top_list[thread][ind] as usize] as usize];
                            top_levels[ind] = nest.level;
                            /* DNM 12/18/01: Need to include even levels also */
                            if !nest.inside.is_empty() {
                                if let Some(inscan) = nest.inscan.as_ref() {
                                    top_cont = inscan.clone();
                                    imod_contour_get_bbox(
                                        Some(&top_cont),
                                        &mut top_pt_min,
                                        &mut top_pt_max,
                                    );
                                }
                            }
                        }
                        for jnd in 0..num_bot {
                            bot_used[jnd] = 0;
                            bot_levels[jnd] = 1;
                            let mut bot_cont = scan_conts[bot_list[thread][jnd] as usize].clone();
                            let mut bot_pt_min = pt_min[bot_list[thread][jnd] as usize];
                            let mut bot_pt_max = pt_max[bot_list[thread][jnd] as usize];
                            if cont_nest_ind[bot_list[thread][jnd] as usize] >= 0 {
                                let bot_nest =
                                    &nests[cont_nest_ind[bot_list[thread][jnd] as usize] as usize];
                                bot_levels[jnd] = bot_nest.level;
                                if !bot_nest.inside.is_empty() {
                                    if let Some(inscan) = bot_nest.inscan.as_ref() {
                                        bot_cont = inscan.clone();
                                        imod_contour_get_bbox(
                                            Some(&bot_cont),
                                            &mut bot_pt_min,
                                            &mut bot_pt_max,
                                        );
                                    }
                                }
                            }
                            let mut frac1 = 0.0f32;
                            let mut frac2 = 0.0f32;
                            imodel_overlap_fractions(
                                &mut bot_cont,
                                bot_pt_min,
                                bot_pt_max,
                                &mut top_cont,
                                top_pt_min,
                                top_pt_max,
                                &mut frac1,
                                &mut frac2,
                            );
                            if frac1 > frac2 {
                                max_frac[ind * num_bot + jnd] = frac1;
                                min_frac[ind * num_bot + jnd] = frac2;
                            } else {
                                max_frac[ind * num_bot + jnd] = frac2;
                                min_frac[ind * num_bot + jnd] = frac1;
                            }
                        }
                    }

                    /* Loop, starting with the maximum overlap pair of odd levels */
                    let mut odd_even = 1;
                    loop {
                        let mut frac_max = -1.0f32;
                        let mut ind_max = 0usize;
                        let mut jnd_max = 0usize;
                        for ind in 0..num_top {
                            if top_used[ind] == 0 && top_levels[ind] % 2 == odd_even {
                                for jnd in 0..num_bot {
                                    if bot_used[jnd] == 0 && bot_levels[jnd] % 2 == odd_even {
                                        let frac1 = min_frac[ind * num_bot + jnd];
                                        if frac1 > frac_max {
                                            frac_max = frac1;
                                            ind_max = ind;
                                            jnd_max = jnd;
                                        }
                                    }
                                }
                            }
                        }

                        /* If there is no overlap, switch to doing even levels and
                        restart, or break out if evens are done also */
                        if frac_max as f64 <= overlap {
                            if odd_even != 0 {
                                odd_even = 0;
                                continue;
                            } else {
                                break;
                            }
                        }

                        /* Start new lists with best pair, take them off original lists */
                        let top_level = top_levels[ind_max];
                        let bot_level = bot_levels[jnd_max];
                        let mut num_top_out = 1usize;
                        let mut num_bot_out = 1usize;
                        top_out_list[0] = ind_max as i32;
                        bot_out_list[0] = jnd_max as i32;
                        top_used[ind_max] = 1;
                        bot_used[jnd_max] = 1;

                        loop {
                            let mut found = 0;
                            /* check every available top contour of same level
                            against the bottom list, add to top list if overlap */
                            for ind in 0..num_top {
                                if top_used[ind] == 0 && top_levels[ind] == top_level {
                                    for jnd in 0..num_bot_out {
                                        if max_frac[ind * num_bot + bot_out_list[jnd] as usize]
                                            as f64
                                            > overlap
                                        {
                                            top_out_list[num_top_out] = ind as i32;
                                            num_top_out += 1;
                                            top_used[ind] = 1;
                                            found = 1;
                                            break;
                                        }
                                    }
                                }
                            }
                            /* Do the same for bottom contours against top list */
                            for jnd in 0..num_bot {
                                if bot_used[jnd] == 0 && bot_levels[jnd] == bot_level {
                                    for ind in 0..num_top_out {
                                        if max_frac[top_out_list[ind] as usize * num_bot + jnd]
                                            as f64
                                            > overlap
                                        {
                                            bot_out_list[num_bot_out] = jnd as i32;
                                            num_bot_out += 1;
                                            bot_used[jnd] = 1;
                                            found = 1;
                                            break;
                                        }
                                    }
                                }
                            }
                            if found == 0 {
                                break;
                            }
                        }

                        /* Next, build lists of contours just inside this set */
                        let mut num_bot_just_in = 0usize;
                        let mut num_top_just_in = 0usize;
                        for ind in 0..num_top {
                            if top_used[ind] == 0 && top_levels[ind] == top_level + 1 {
                                let nest =
                                    &nests[cont_nest_ind[top_list[thread][ind] as usize] as usize];
                                for knd in 0..nest.outside.len() {
                                    let mut found = 0;
                                    for jnd in 0..num_top_out {
                                        if nest.outside[knd]
                                            == top_list[thread][top_out_list[jnd] as usize]
                                        {
                                            found = 1;
                                            top_just_in_list[num_top_just_in] = ind as i32;
                                            num_top_just_in += 1;
                                            break;
                                        }
                                    }
                                    if found != 0 {
                                        break;
                                    }
                                }
                            }
                        }
                        for ind in 0..num_bot {
                            if bot_used[ind] == 0 && bot_levels[ind] == bot_level + 1 {
                                let nest =
                                    &nests[cont_nest_ind[bot_list[thread][ind] as usize] as usize];
                                for knd in 0..nest.outside.len() {
                                    let mut found = 0;
                                    for jnd in 0..num_bot_out {
                                        if nest.outside[knd]
                                            == bot_list[thread][bot_out_list[jnd] as usize]
                                        {
                                            found = 1;
                                            bot_just_in_list[num_bot_just_in] = ind as i32;
                                            num_bot_just_in += 1;
                                            break;
                                        }
                                    }
                                    if found != 0 {
                                        break;
                                    }
                                }
                            }
                        }

                        /* Now iteratively find overlapping sets of inside conts */
                        for ind_just_in in 0..num_bot_just_in {
                            if bot_used[bot_just_in_list[ind_just_in] as usize] == 0 {
                                let mut num_top_inside = 0usize;
                                let mut num_bot_inside = 1usize;
                                bot_inside_list[0] = bot_just_in_list[ind_just_in];
                                bot_used[bot_just_in_list[ind_just_in] as usize] = 1;
                                loop {
                                    let mut found = 0;
                                    for jnd in 0..num_top_just_in {
                                        let co_num_just_in = top_just_in_list[jnd] as usize;
                                        if top_used[co_num_just_in] == 0 {
                                            for knd in 0..num_bot_inside {
                                                if max_frac[co_num_just_in * num_bot
                                                    + bot_inside_list[knd] as usize]
                                                    as f64
                                                    > overlap
                                                {
                                                    found = 1;
                                                    top_inside_list[num_top_inside] =
                                                        co_num_just_in as i32;
                                                    num_top_inside += 1;
                                                    top_used[co_num_just_in] = 1;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    for jnd in 0..num_bot_just_in {
                                        let co_num_just_in = bot_just_in_list[jnd] as usize;
                                        if bot_used[co_num_just_in] == 0 {
                                            for knd in 0..num_top_inside {
                                                if max_frac[top_inside_list[knd] as usize * num_bot
                                                    + co_num_just_in]
                                                    as f64
                                                    > overlap
                                                {
                                                    found = 1;
                                                    bot_inside_list[num_bot_inside] =
                                                        co_num_just_in as i32;
                                                    num_bot_inside += 1;
                                                    bot_used[co_num_just_in] = 1;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    if found == 0 {
                                        break;
                                    }
                                }

                                /* If got top contours, process and mesh the set */
                                if num_top_inside != 0 {
                                    let num_bot_same = num_bot_inside;
                                    let num_top_same = num_top_inside;
                                    /* look for contours just inside these and add them
                                    to lists if they overlap the interior area */

                                    loop {
                                        let mut found_more = 0;
                                        for ind in 0..num_top {
                                            if top_used[ind] != 0
                                                || top_levels[ind] != top_level + 2
                                            {
                                                continue;
                                            }

                                            /* see if it's inside */
                                            let nest = &nests[cont_nest_ind
                                                [top_list[thread][ind] as usize]
                                                as usize];
                                            let mut found = 0;
                                            for knd in 0..nest.outside.len() {
                                                for jnd in 0..num_top_inside {
                                                    if nest.outside[knd]
                                                        == top_list[thread]
                                                            [top_inside_list[jnd] as usize]
                                                    {
                                                        found = 1;
                                                    }
                                                }
                                            }
                                            if found == 0 {
                                                continue;
                                            }

                                            /* see if overlap outsides */
                                            found = 0;
                                            for knd in 0..num_bot_out {
                                                if max_frac
                                                    [ind * num_bot + bot_out_list[knd] as usize]
                                                    as f64
                                                    > overlap
                                                {
                                                    found = 1;
                                                }
                                            }

                                            /* see if overlap added ones */
                                            for knd in num_bot_same..num_bot_inside {
                                                if max_frac
                                                    [ind * num_bot + bot_inside_list[knd] as usize]
                                                    as f64
                                                    > overlap
                                                {
                                                    found = 1;
                                                }
                                            }
                                            if found != 0 {
                                                found_more = 1;
                                                top_inside_list[num_top_inside] = ind as i32;
                                                num_top_inside += 1;
                                                top_used[ind] = 1;
                                            }
                                        }

                                        for ind in 0..num_bot {
                                            if bot_used[ind] != 0
                                                || bot_levels[ind] != bot_level + 2
                                            {
                                                continue;
                                            }

                                            let nest = &nests[cont_nest_ind
                                                [bot_list[thread][ind] as usize]
                                                as usize];
                                            let mut found = 0;
                                            for knd in 0..nest.outside.len() {
                                                for jnd in 0..num_bot_inside {
                                                    if nest.outside[knd]
                                                        == bot_list[thread]
                                                            [bot_inside_list[jnd] as usize]
                                                    {
                                                        found = 1;
                                                    }
                                                }
                                            }
                                            if found == 0 {
                                                continue;
                                            }

                                            found = 0;
                                            for knd in 0..num_top_out {
                                                if max_frac
                                                    [top_out_list[knd] as usize * num_bot + ind]
                                                    as f64
                                                    > overlap
                                                {
                                                    found = 1;
                                                }
                                            }

                                            for knd in num_top_same..num_top_inside {
                                                if max_frac
                                                    [top_inside_list[knd] as usize * num_bot + ind]
                                                    as f64
                                                    > overlap
                                                {
                                                    found = 1;
                                                }
                                            }
                                            if found != 0 {
                                                found_more = 1;
                                                bot_inside_list[num_bot_inside] = ind as i32;
                                                num_bot_inside += 1;
                                                bot_used[ind] = 1;
                                            }
                                        }

                                        if found_more == 0 {
                                            break;
                                        }
                                    }

                                    for ind in 0..num_top_inside {
                                        top_inside_list[ind] =
                                            top_list[thread][top_inside_list[ind] as usize];
                                    }
                                    for ind in 0..num_bot_inside {
                                        bot_inside_list[ind] =
                                            bot_list[thread][bot_inside_list[ind] as usize];
                                    }

                                    let bot_cont = join_all_contours(
                                        obj,
                                        &bot_inside_list,
                                        num_bot_inside as i32,
                                        num_bot_same as i32,
                                        1,
                                        Some(&top_inside_list),
                                        num_top_inside as i32,
                                        num_top_same as i32,
                                        None,
                                    );
                                    let top_cont = join_all_contours(
                                        obj,
                                        &top_inside_list,
                                        num_top_inside as i32,
                                        num_top_same as i32,
                                        -1,
                                        Some(&bot_inside_list),
                                        num_bot_inside as i32,
                                        num_bot_same as i32,
                                        None,
                                    );
                                    let surf = obj.cont[co_num].surf;
                                    let time = obj.cont[co_num].time;
                                    if mesh_contours(
                                        obj,
                                        &mut temp_meshes[thread],
                                        bot_cont,
                                        top_cont,
                                        surf,
                                        time,
                                        scale,
                                        odd_even,
                                        bot_inside_list[0],
                                        top_inside_list[0],
                                    ) != 0
                                    {
                                        error = 1;
                                        break;
                                    }
                                } else {
                                    bot_used[bot_just_in_list[ind_just_in] as usize] = 0;
                                }
                            }
                        }
                        if error != 0 {
                            break;
                        }

                        /* There may now be orphan inside contours left. */
                        for ind in 0..num_top_out {
                            top_out_list[ind] = top_list[thread][top_out_list[ind] as usize];
                        }
                        for ind in 0..num_bot_out {
                            bot_out_list[ind] = bot_list[thread][bot_out_list[ind] as usize];
                        }

                        let bot_out_cont = join_all_contours(
                            obj,
                            &bot_out_list,
                            num_bot_out as i32,
                            num_bot_out as i32,
                            1,
                            Some(&top_out_list),
                            num_top_out as i32,
                            num_top_out as i32,
                            None,
                        );
                        let top_out_cont = join_all_contours(
                            obj,
                            &top_out_list,
                            num_top_out as i32,
                            num_top_out as i32,
                            -1,
                            Some(&bot_out_list),
                            num_bot_out as i32,
                            num_bot_out as i32,
                            None,
                        );
                        let (Some(bot_out_cont), Some(top_out_cont)) = (bot_out_cont, top_out_cont)
                        else {
                            error = 1;
                            break;
                        };

                        let surf = obj.cont[co_num].surf;
                        let time = obj.cont[co_num].time;
                        let bot_out_cont = connect_orphans(
                            obj,
                            &mut temp_meshes[thread],
                            bot_out_cont,
                            &top_list[thread].clone(),
                            &mut top_used,
                            &top_just_in_list,
                            num_top_just_in as i32,
                            &mut top_inside_list,
                            -1,
                            odd_even,
                            surf,
                            time,
                            scale,
                            &scan_conts,
                            &pt_min,
                            &pt_max,
                        );
                        let top_out_cont = connect_orphans(
                            obj,
                            &mut temp_meshes[thread],
                            top_out_cont,
                            &bot_list[thread].clone(),
                            &mut bot_used,
                            &bot_just_in_list,
                            num_bot_just_in as i32,
                            &mut bot_inside_list,
                            1,
                            odd_even,
                            surf,
                            time,
                            scale,
                            &scan_conts,
                            &pt_min,
                            &pt_max,
                        );
                        /* finally ready to mesh the outside contours */
                        if mesh_contours(
                            obj,
                            &mut temp_meshes[thread],
                            bot_out_cont,
                            top_out_cont,
                            surf,
                            time,
                            scale,
                            1 - odd_even,
                            bot_out_list[0],
                            top_out_list[0],
                        ) != 0
                        {
                            error = 1;
                            break;
                        }
                    }

                    /* Need to unmark connections for any that weren't used */
                    for ind in 0..num_bot {
                        if bot_used[ind] <= 0 {
                            obj.cont[bot_list[thread][ind] as usize].flags &= !CONNECT_TOP;
                        }
                    }
                    for ind in 0..num_top {
                        if top_used[ind] <= 0 {
                            obj.cont[top_list[thread][ind] as usize].flags &= !CONNECT_BOTTOM;
                        }
                    }

                    drop(min_frac);
                    drop(max_frac);
                } else {
                    /* We didn't find a contour to connect to - remove marks. */
                    for ind in 0..num_bot {
                        obj.cont[bot_list[thread][ind] as usize].flags &= !CONNECT_TOP;
                    }
                }
            }
        }
    } /* end of zpass loop */

    /* Add up number of meshes and transfer the meshes */
    for thread in 0..num_threads {
        let meshes = std::mem::take(&mut temp_meshes[thread]);
        for mesh in meshes {
            obj.mesh.push(mesh);
        }
    }

    /* force connection of stray contours to nearest contour */
    if flags & IMESH_MK_STRAY != 0 {
        for ind_z in 0..(z_max + 1 - z_min) as usize {
            /* Loop on Z and get next Z value for connection */
            let next_z = if flags & IMESH_MK_SKIP != 0 {
                getnextz(&z_list, z_list_size, ind_z as i32 + z_min)
            } else {
                ind_z as i32 + z_min + z_increment
            };
            if next_z > z_list[(z_list_size - 1) as usize] {
                continue;
            }

            /* Loop on pairs of contours between the two sections */
            loop {
                let mut have_top = false;
                let mut top_co_num = 0usize;
                let mut bot_co_num = 0usize;
                for jnd_at_z in 0..num_cont_at_z[ind_z] as usize {
                    let co_num = conts_at_z[ind_z][jnd_at_z] as usize;
                    if connect_top(obj.cont[co_num].flags) != 0 {
                        continue;
                    }

                    let bot_cnct_num = istore_connect_number(&obj.store, co_num as i32);
                    robust_center_of_mass(
                        &obj.cont[co_num],
                        Some(&scan_conts[co_num]),
                        &mut bot_cen_mass,
                    );
                    let mut min_dist = 1.0e30f32;

                    for ind_at_z in 0..num_cont_at_z[(next_z - z_min) as usize] as usize {
                        let other_co_num = conts_at_z[(next_z - z_min) as usize][ind_at_z] as usize;
                        if obj.cont[other_co_num].flags & CONNECT_BOTTOM != 0 {
                            continue;
                        }
                        if (flags & IMESH_MK_SURF) != 0
                            && obj.cont[co_num].surf != obj.cont[other_co_num].surf
                        {
                            continue;
                        }
                        if (flags & IMESH_MK_TIME) != 0
                            && obj.cont[co_num].time != obj.cont[other_co_num].time
                        {
                            continue;
                        }
                        if obj.cont[co_num].pts.len() == 1 && obj.cont[other_co_num].pts.len() == 1
                        {
                            continue;
                        }

                        /* Do not connect if they have non-matching connection numbers */
                        let top_cnct_num = istore_connect_number(&obj.store, other_co_num as i32);
                        if top_cnct_num >= 0 && bot_cnct_num >= 0 && top_cnct_num != bot_cnct_num {
                            continue;
                        }

                        robust_center_of_mass(&obj.cont[other_co_num], None, &mut top_cen_mass);
                        let dist = (top_cen_mass.x - bot_cen_mass.x)
                            * (top_cen_mass.x - bot_cen_mass.x)
                            + (top_cen_mass.y - bot_cen_mass.y) * (top_cen_mass.y - bot_cen_mass.y);

                        /* Keep track of closest pair of contours */
                        if dist < min_dist {
                            have_top = true;
                            min_dist = dist;
                            bot_co_num = co_num;
                            top_co_num = other_co_num;
                        }
                    }
                }

                if have_top {
                    let mut in_temp = 0;
                    if obj.flags & IMOD_OBJFLAG_OUT != 0 {
                        in_temp = 1 - in_temp;
                    }
                    let mut bc = obj.cont[bot_co_num].clone();
                    let mut tc = obj.cont[top_co_num].clone();
                    let new_mesh = imesh_contours_cost(
                        obj,
                        &mut bc,
                        &mut tc,
                        scale,
                        in_temp,
                        bot_co_num as i32,
                        top_co_num as i32,
                    );
                    obj.cont[bot_co_num].flags = bc.flags;
                    obj.cont[top_co_num].flags = tc.flags;
                    if let Some(new_mesh) = new_mesh {
                        imodel_mesh_add(Some(&new_mesh), &mut obj.mesh);
                    }
                    obj.cont[bot_co_num].flags |= CONNECT_TOP;
                    obj.cont[top_co_num].flags |= CONNECT_BOTTOM;
                } else {
                    break;
                }
            }
        }
    }

    skin_report_time("Connected contours");
    let _status = callback(50); /* half way done. */
    let status = callback(1);
    if status != 0 {
        return status;
    }

    if do_cap != 0 {
        let mut def_props = DrawProps::default();
        let mut cont_props = DrawProps::default();
        let mut cont_state: i32 = 0;
        let mut surf_state: i32 = 0;

        /* Loop twice, first time looking for outside contours with inside ones */
        for nest_loop in (0..=1).rev() {
            istore_default_draw_props(obj, &mut def_props);
            for co_num in 0..obj.cont.len() {
                let mut inside = 0;
                istore_cont_surf_draw_props(
                    &obj.store,
                    &def_props,
                    &mut cont_props,
                    co_num as i32,
                    obj.cont[co_num].surf,
                    &mut cont_state,
                    &mut surf_state,
                );
                if cont_props.no_cap != 0 {
                    continue;
                }

                /* if the contour has two or less points then the surface must
                already be capped. */
                if obj.cont[co_num].pts.len() > 2 && !connect_both(obj.cont[co_num].flags) {
                    /* But don't cap if the thickness is nil */
                    if imod_contour_area(Some(&obj.cont[co_num]))
                        / imod_contour_length(Some(&obj.cont[co_num]), 1)
                        < 0.01
                    {
                        continue;
                    }

                    /* Don't cap an open contour with phantom extensions */
                    if (obj.cont[co_num].flags & ICONT_OPEN) != 0
                        && istore_point_is_gap(&obj.cont[co_num].store, 0) != 0
                        && istore_point_is_gap(
                            &obj.cont[co_num].store,
                            obj.cont[co_num].pts.len() as i32 - 2,
                        ) != 0
                    {
                        continue;
                    }

                    if obj.flags & IMOD_OBJFLAG_OUT != 0 {
                        inside = 1 - inside;
                    }

                    /* Skip capping if the Z value is in list to exclude */
                    let direction = if connect_bottom(obj.cont[co_num].flags) != 0
                        || obj.cont[co_num].pts[0].z == z_max as f32
                    {
                        1
                    } else {
                        -1
                    };
                    let mut skip_z = 0;
                    if let Some(list) = cap_skip_zlist {
                        for ind in 0..cap_skip_num_z as usize {
                            if ((obj.cont[co_num].pts[0].z + direction as f32 + 0.5) as f64).floor()
                                == list[ind] as f64
                            {
                                skip_z = 1;
                                break;
                            }
                        }
                    }
                    if skip_z != 0 {
                        continue;
                    }

                    // Determine of inside contour
                    let mut num_inside = 0usize;
                    let mut scan_use: Icont = scan_conts[co_num].clone();
                    let mut mesh_cont: Option<Icont> = None;
                    let mut top_cont: Option<Icont> = None;
                    if cont_nest_ind[co_num] >= 0
                        && nests[cont_nest_ind[co_num] as usize].level % 2 == 0
                    {
                        inside = 1;
                    }

                    // On nesting loop, skip inside or non-nested ones and make list
                    // of ones that are just inside
                    if nest_loop != 0 {
                        if cont_nest_ind[co_num] < 0 || inside != 0 || do_cap != IMESH_CAP_ALL {
                            continue;
                        }
                        let nest_level = nests[cont_nest_ind[co_num] as usize].level;
                        let inside_list = nests[cont_nest_ind[co_num] as usize].inside.clone();
                        for &jnd in inside_list.iter() {
                            let jnd = jnd as usize;
                            if cont_nest_ind[jnd] >= 0
                                && nests[cont_nest_ind[jnd] as usize].level == nest_level + 1
                                && ((connect_top(obj.cont[jnd].flags) == 0
                                    && connect_top(obj.cont[co_num].flags) == 0)
                                    || (connect_bottom(obj.cont[jnd].flags) == 0
                                        && connect_bottom(obj.cont[co_num].flags) == 0))
                            {
                                bot_list[0][num_inside] = jnd as i32;
                                num_inside += 1;
                            }
                        }
                        if num_inside == 0 {
                            continue;
                        }

                        // Do the joins here with a size-dependent backoff
                        let mut min_pt = Ipoint::default();
                        let mut max_pt = Ipoint::default();
                        imod_contour_get_bbox(Some(&obj.cont[co_num]), &mut min_pt, &mut max_pt);
                        let mut backoff =
                            11 + ((max_pt.x - min_pt.x).max(max_pt.y - min_pt.y) / 50.) as i32;
                        backoff = backoff.max(15);

                        // Join all the inside contours first
                        let cont_clone = obj.cont[co_num].clone();
                        let bot_cont = join_all_contours(
                            obj,
                            &bot_list[0].clone(),
                            num_inside as i32,
                            num_inside as i32,
                            backoff,
                            None,
                            0,
                            0,
                            Some(&cont_clone),
                        );
                        let Some(bot_cont) = bot_cont else {
                            continue;
                        };
                        let inner_for_mesh = join_all_contours(
                            obj,
                            &bot_list[0].clone(),
                            num_inside as i32,
                            num_inside as i32,
                            2,
                            None,
                            0,
                            0,
                            Some(&cont_clone),
                        );
                        let Some(inner_for_mesh) = inner_for_mesh else {
                            continue;
                        };

                        top_list[0][0] = -1;
                        top_list[0][1] = co_num as i32;

                        // Again, join with backoff, and join again with the inner
                        let tc = join_all_contours(
                            obj,
                            &top_list[0].clone(),
                            2,
                            1,
                            backoff,
                            None,
                            0,
                            0,
                            Some(&bot_cont),
                        );
                        mesh_cont = join_all_contours(
                            obj,
                            &top_list[0].clone(),
                            2,
                            1,
                            2,
                            None,
                            0,
                            0,
                            Some(&inner_for_mesh),
                        );
                        let Some(tc) = tc else {
                            continue;
                        };

                        // Make scan contour of the one to be analyzed
                        let Some(su) = imodel_contour_scan(Some(&tc)) else {
                            continue;
                        };
                        scan_use = su;
                        top_cont = Some(tc);
                    }

                    if do_cap == IMESH_CAP_ALL {
                        // Connect bottom and/or top if needed
                        let mut cap_cont: Icont = if num_inside != 0 {
                            top_cont.clone().unwrap()
                        } else {
                            obj.cont[co_num].clone()
                        };
                        if connect_bottom(obj.cont[co_num].flags) == 0 {
                            let new_mesh = imesh_contour_cap(
                                obj,
                                &mut cap_cont,
                                Some(&scan_use),
                                co_num as i32,
                                -1,
                                inside,
                                scale,
                                mesh_cont.as_ref(),
                            );
                            let surf = obj.cont[co_num].surf;
                            let time = obj.cont[co_num].time;
                            add_mesh_to_object(obj, surf, time, new_mesh, flags);
                        }
                        if connect_top(obj.cont[co_num].flags) == 0 {
                            let new_mesh = imesh_contour_cap(
                                obj,
                                &mut cap_cont,
                                Some(&scan_use),
                                co_num as i32,
                                1,
                                inside,
                                scale,
                                mesh_cont.as_ref(),
                            );
                            let surf = obj.cont[co_num].surf;
                            let time = obj.cont[co_num].time;
                            add_mesh_to_object(obj, surf, time, new_mesh, flags);
                        }

                        // Clean up and mark from inside contour meshing
                        if num_inside != 0 {
                            obj.cont[co_num].flags |= CONNECT_TOP | CONNECT_BOTTOM;
                            for ind in 0..num_inside {
                                obj.cont[bot_list[0][ind] as usize].flags |=
                                    CONNECT_TOP | CONNECT_BOTTOM;
                            }
                        }
                        continue;
                    }

                    if obj.cont[co_num].pts[0].z == z_min as f32 {
                        let mut cap_cont = obj.cont[co_num].clone();
                        let new_mesh = imesh_contour_cap(
                            obj,
                            &mut cap_cont,
                            Some(&scan_use),
                            co_num as i32,
                            -1,
                            inside,
                            scale,
                            None,
                        );
                        let surf = obj.cont[co_num].surf;
                        let time = obj.cont[co_num].time;
                        add_mesh_to_object(obj, surf, time, new_mesh, flags);
                    }
                    if obj.cont[co_num].pts[0].z == z_max as f32 {
                        let mut cap_cont = obj.cont[co_num].clone();
                        let new_mesh = imesh_contour_cap(
                            obj,
                            &mut cap_cont,
                            Some(&scan_use),
                            co_num as i32,
                            1,
                            inside,
                            scale,
                            None,
                        );
                        let surf = obj.cont[co_num].surf;
                        let time = obj.cont[co_num].time;
                        add_mesh_to_object(obj, surf, time, new_mesh, flags);
                    }
                }
            }
        }
    }
    /*
     * Clean up scan conversions.
     */
    for co_num in 0..scan_conts.len() {
        imod_contour_delete(&mut scan_conts[co_num]);
    }
    drop(scan_conts);

    skin_report_time("Capped");
    for mesh_ind in 0..obj.mesh.len() {
        if flags & IMESH_MK_SURF == 0 {
            obj.mesh[mesh_ind].surf = 0;
        }
        if flags & IMESH_MK_TIME == 0 {
            obj.mesh[mesh_ind].time = 0;
        }
    }

    if flags & IMESH_MK_NORM != 0 {
        let mut size = obj.mesh.len() as i32;
        let mut meshes = std::mem::take(&mut obj.mesh);
        obj.mesh = imesh_remesh_normal(&mut meshes, &mut size, Some(scale), 0).unwrap_or_default();
    }

    imod_contour_free_nests(&mut nests, num_nests);
    imod_contour_free_z_tables(
        &mut num_cont_at_z,
        &mut conts_at_z,
        &mut contour_z,
        &mut z_list,
        z_min,
        z_max,
    );

    drop(cont_nest_ind);
    drop(top_list);
    drop(bot_list);
    drop(pt_min);
    drop(pt_max);
    0
}

/// `imodel_overlap_fractions` on two distinct elements of the same scan
/// contour array, split so both can be borrowed mutably, as the C's two
/// `Icont **` arguments into one array are.
#[allow(clippy::too_many_arguments)]
fn overlap_fractions_at(
    scan_conts: &mut [Icont],
    ind1: usize,
    pmin1: Ipoint,
    pmax1: Ipoint,
    ind2: usize,
    pmin2: Ipoint,
    pmax2: Ipoint,
    frac1: &mut f32,
    frac2: &mut f32,
) -> i32 {
    if ind1 == ind2 {
        let mut copy = scan_conts[ind1].clone();
        return imodel_overlap_fractions(
            &mut copy,
            pmin1,
            pmax1,
            &mut scan_conts[ind2],
            pmin2,
            pmax2,
            frac1,
            frac2,
        );
    }
    if ind1 < ind2 {
        let (left, right) = scan_conts.split_at_mut(ind2);
        imodel_overlap_fractions(
            &mut left[ind1],
            pmin1,
            pmax1,
            &mut right[0],
            pmin2,
            pmax2,
            frac1,
            frac2,
        )
    } else {
        let (left, right) = scan_conts.split_at_mut(ind1);
        imodel_overlap_fractions(
            &mut right[0],
            pmin1,
            pmax1,
            &mut left[ind2],
            pmin2,
            pmax2,
            frac1,
            frac2,
        )
    }
}

/// Original static `add_whole_nest` (`skinobj.c:1696`).
///
/// Makes a list of all contours in a nested set, starting with the nest at
/// index `nind`, finding the outermost nest, and adding each contour in that
/// nest to `tlist` if not in it already.
pub fn add_whole_nest(
    mut nind: i32,
    obj: &mut Iobj,
    nests: &[Nesting],
    nestind: &[i32],
    flag: u32,
    tlist: &mut [i32],
    ntop: &mut usize,
) {
    let mut nest = &nests[nind as usize];

    // Loop through the outside contours in this nest until find the one with a nest at
    // level 1 and switch to that nest
    for i in 0..nest.outside.len() {
        let oind = nestind[nest.outside[i] as usize];
        if nests[oind as usize].level == 1 {
            nind = oind;
            nest = &nests[nind as usize];
            break;
        }
    }

    /* Add primary contour of nest to list if not on it */
    if obj.cont[nest.co as usize].flags & flag == 0 {
        obj.cont[nest.co as usize].flags |= flag;
        tlist[*ntop] = nest.co;
        *ntop += 1;
    }

    /* Add all inside contours to list */
    for i in 0..nest.inside.len() {
        let jnd = nest.inside[i] as usize;
        if obj.cont[jnd].flags & flag == 0 {
            obj.cont[jnd].flags |= flag;
            tlist[*ntop] = nest.inside[i];
            *ntop += 1;
        }
    }
}

/// Original: `NSECT` (`skinobj.c:1735`).
const NSECT: usize = 38;
/// Original: `MAXTEST` (`skinobj.c:1736`).
const MAXTEST: usize = 100;
/// Original: `NSEARCH` (`skinobj.c:1737`).
const NSEARCH: usize = 13;

/// Original static `connect_orphans` (`skinobj.c:1748`).
///
/// Routine to connect orphan contours.
#[allow(clippy::too_many_arguments)]
pub fn connect_orphans(
    obj: &Iobj,
    obj_mesh: &mut Vec<Imesh>,
    cout_in: Icont,
    list: &[i32],
    used: &mut [i32],
    just_in_list: &[i32],
    num_just_in: i32,
    inlist: &mut [i32],
    joindir: i32,
    oddeven: i32,
    surf: i32,
    time: i32,
    scale: &Ipoint,
    scancont: &[Icont],
    pmin: &[Ipoint],
    pmax: &[Ipoint],
) -> Option<Icont> {
    let mut cout = cout_in;
    let mut outcand = [0i32; NSECT];
    let mut indorder = [0usize; NSECT * NSECT];
    let mut pathlen = [0.0f32; NSECT * NSECT];
    let mut forlen = [0.0f32; NSECT];
    let mut st1test = [0i32; MAXTEST];
    let mut st2test = [0i32; MAXTEST];
    let mut testratio = [0.0f32; MAXTEST];
    let dst1: [i32; NSEARCH] = [0, 1, 1, 1, 2, 2, 0, 2, -1, 0, -1, -1, 1];
    let dst2: [i32; NSEARCH] = [1, 0, 1, 2, 1, 2, 2, 0, 0, -1, -1, 1, -1];
    let mut inmatch = [0i32; NSECT];
    let nscan = 2;

    let mut found_data = 0;
    for inl in 0..num_just_in as usize {
        if used[just_in_list[inl] as usize] == 0 {
            found_data = 1;
        }
    }
    if found_data == 0 {
        return Some(cout);
    }

    let mut outscan = imodel_contour_scan(Some(&cout))?;
    let mut pminout = Ipoint::default();
    let mut pmaxout = Ipoint::default();
    imod_contour_get_bbox(Some(&cout), &mut pminout, &mut pmaxout);

    /* go through and mark ones that will get capped first */
    for inl in 0..num_just_in as usize {
        if used[just_in_list[inl] as usize] == 0 {
            let co = list[just_in_list[inl] as usize] as usize;
            let mut frac1 = 0.0f32;
            let mut frac2 = 0.0f32;
            let mut cs1 = scancont[co].clone();
            imodel_overlap_fractions(
                &mut cs1,
                pmin[co],
                pmax[co],
                &mut outscan,
                pminout,
                pmaxout,
                &mut frac1,
                &mut frac2,
            );
            if frac1 > 0.8 {
                used[just_in_list[inl] as usize] = -1;
            }
        }
    }
    imod_contour_delete(&mut outscan);

    for inl in 0..num_just_in as usize {
        if used[just_in_list[inl] as usize] != 0 {
            continue;
        }
        let full_area = imod_contour_area(Some(&cout));
        let totlen = imod_contour_length(Some(&cout), 1);

        /* start with whole inner contour, then iterate on a segment */
        let co = list[just_in_list[inl] as usize] as usize;
        let cont = &obj.cont[co];
        let mut instart: i32 = 0;
        let mut inend: i32 = 0;
        let mut ncand: usize = 0;
        let mut indmin: usize = 0;
        let mut icand1: usize = 0;
        let mut icand2: usize = 0;
        let mut min_ratio = 1.0e30f32;
        let mut scandir: i32 = 1;
        for _iscan in 0..nscan {
            let mut pt = instart;
            let mut ddist: f64 = 0.;
            let pts = &cont.pts;
            let mut next;
            loop {
                next = (pt + 1) % cont.pts.len() as i32;
                let ddx = (pts[next as usize].x - pts[pt as usize].x) as f64;
                let ddy = (pts[next as usize].y - pts[pt as usize].y) as f64;
                ddist += (ddx * ddx + ddy * ddy).sqrt();
                pt = next;
                if next == inend {
                    break;
                }
            }
            let seglen = ddist as f32;

            /* go to equally spaced points around the segment of the inner contour
            and make a list of nearest points in the outer contour */
            let mut ddist: f64 = 0.;
            let mut nextlen = seglen / (NSECT - 2) as f32;
            ncand = 1;
            let mut nforward = 0;
            let mut pt = instart;
            inmatch[0] = instart;
            outcand[0] = imod_contour_nearest(Some(&cout), &pts[pt as usize]);
            loop {
                let next = (pt + 1) % cont.pts.len() as i32;
                let ddx = (pts[next as usize].x - pts[pt as usize].x) as f64;
                let ddy = (pts[next as usize].y - pts[pt as usize].y) as f64;
                ddist += (ddx * ddx + ddy * ddy).sqrt();
                if ddist > nextlen as f64 || next == inend {
                    nextlen += seglen / (NSECT - 2) as f32;
                    inmatch[ncand] = next;
                    outcand[ncand] = imod_contour_nearest(Some(&cout), &pts[next as usize]);
                    ncand += 1;
                    if outcand[ncand - 1] > outcand[ncand - 2] {
                        nforward += 1;
                    }
                    if ncand == NSECT {
                        break;
                    }
                }
                pt = next;
                if next == inend {
                    break;
                }
            }
            scandir = if nforward as f32 > ncand as f32 / 2. {
                1
            } else {
                -1
            };

            /* order the candidates and eliminate duplicates */
            for i in 0..ncand - 1 {
                for j in i + 1..ncand {
                    if outcand[i] > outcand[j] {
                        outcand.swap(i, j);
                        inmatch.swap(i, j);
                    }
                }
            }
            let mut j = 1usize;
            for i in 1..ncand {
                if outcand[j - 1] != outcand[i] {
                    inmatch[j] = inmatch[i];
                    outcand[j] = outcand[i];
                    j += 1;
                }
            }
            ncand = j;
            if ncand < 2 {
                break;
            }

            /* get forward distances */
            let opts = &cout.pts;
            for i in 0..ncand - 1 {
                let mut ddist: f64 = 0.;
                for pt in outcand[i]..outcand[i + 1] {
                    ddist +=
                        imod_point_distance(&opts[pt as usize], &opts[(pt + 1) as usize]) as f64;
                }
                forlen[i] = ddist as f32;
            }

            /* compute minimum path length between each pair of candidates */
            let mut norder = 0usize;
            for i in 0..ncand - 1 {
                let mut ddist: f64 = 0.;
                for j in i + 1..ncand {
                    ddist += forlen[j - 1] as f64;
                    if ddist < totlen as f64 / 2. {
                        pathlen[i * ncand + j] = ddist as f32;
                        pathlen[j * ncand + i] = 1.;
                    } else {
                        pathlen[i * ncand + j] = (totlen as f64 - ddist) as f32;
                        pathlen[j * ncand + i] = -1.;
                    }
                    indorder[norder] = i * ncand + j;
                    norder += 1;
                }
            }

            /* order by decreasing separation */
            for i in 0..norder.saturating_sub(1) {
                for j in i..norder {
                    if pathlen[indorder[i]] < pathlen[indorder[j]] {
                        indorder.swap(i, j);
                    }
                }
            }

            /* run through the pairs of candidates finding one with a maximum inner
            area */
            let mut maxarea = -1.0f32;
            for i in 0..norder {
                let st1 = outcand[indorder[i] / ncand];
                let st2 = outcand[indorder[i] % ncand];
                let mut inner_area = 0.0f32;
                let ratio = evaluate_break(
                    &cout,
                    list,
                    used,
                    just_in_list,
                    num_just_in,
                    scancont,
                    pmin,
                    pmax,
                    st1,
                    st2,
                    maxarea,
                    pathlen[indorder[i]],
                    full_area,
                    joindir,
                    1,
                    &mut inner_area,
                );
                if ratio > 1.0e25 {
                    return None;
                }
                if inner_area > maxarea {
                    min_ratio = ratio;
                    maxarea = inner_area;
                    indmin = indorder[i];
                } else if inner_area == maxarea && ratio < min_ratio {
                    min_ratio = ratio;
                    indmin = indorder[i];
                }
            }

            /* set up for another scan between these points */
            icand1 = indmin / ncand;
            icand2 = indmin % ncand;
            if scandir as f32 * pathlen[icand2 * ncand + icand1] > 0. {
                instart = inmatch[icand2];
                inend = inmatch[icand1];
            } else {
                instart = inmatch[icand1];
                inend = inmatch[icand2];
            }
        }
        if ncand < 2 {
            continue;
        }

        /* Set up for search for somewhat local minimum */
        let minpath = pathlen[indmin];
        let mut st1min = outcand[icand1];
        let mut st2min = outcand[icand2];
        st1test[0] = st1min;
        st2test[0] = st2min;
        testratio[0] = min_ratio;
        let mut ntest = 1usize;
        let mut nexttest = 1usize;
        let outsize = cout.pts.len() as i32;
        let sweepdir = -pathlen[icand2 * ncand + icand1] as i32;
        loop {
            let mut found_data = 0;
            for i in 0..NSEARCH {
                let mut curpath = minpath;
                let st1 = (st1min + dst1[i] * sweepdir + outsize) % outsize;
                let mut dstdir = if dst1[i] > 0 { 1 } else { -1 };
                let mut pt = st1;
                for _j in 0..dstdir * dst1[i] {
                    let next = (pt + dstdir * sweepdir + outsize) % outsize;
                    curpath += dstdir as f32
                        * imod_point_distance(&cout.pts[pt as usize], &cout.pts[next as usize]);
                    pt = next;
                }
                let st2 = (st2min - dst2[i] * sweepdir + outsize) % outsize;
                dstdir = if dst2[i] > 0 { 1 } else { -1 };
                pt = st2min;
                for _j in 0..dstdir * dst2[i] {
                    let next = (pt + dstdir * sweepdir + outsize) % outsize;
                    curpath += dstdir as f32
                        * imod_point_distance(&cout.pts[pt as usize], &cout.pts[next as usize]);
                    pt = next;
                }

                /* search on the test list */
                let mut found = 0;
                let mut ratio = 0.0f32;
                for j in 0..ntest {
                    if st1test[j] == st1 && st2test[j] == st2 {
                        found = 1;
                        ratio = testratio[j];
                        break;
                    }
                }
                if found == 0 {
                    let mut inner_area = 0.0f32;
                    ratio = evaluate_break(
                        &cout,
                        list,
                        used,
                        just_in_list,
                        num_just_in,
                        scancont,
                        pmin,
                        pmax,
                        st1,
                        st2,
                        min_ratio,
                        curpath,
                        full_area,
                        joindir,
                        0,
                        &mut inner_area,
                    );
                    if ratio > 1.0e25 {
                        return None;
                    }
                    st1test[nexttest] = st1;
                    st2test[nexttest] = st2;
                    testratio[nexttest] = ratio;
                    nexttest += 1;
                    if ntest < nexttest {
                        ntest = nexttest;
                    }
                    if nexttest == MAXTEST {
                        nexttest = 0;
                    }
                }
                if ratio < min_ratio {
                    min_ratio = ratio;
                    st1min = st1;
                    st2min = st2;
                    found_data = 1;
                    break;
                }
            }
            if found_data == 0 {
                break;
            }
        }

        if min_ratio < 1.0e20 {
            let mut c1: Option<Icont> = None;
            let mut c2: Option<Icont> = None;
            if break_contour_inout(&cout, st1min, st2min, -joindir, &mut c1, &mut c2) != 0 {
                return None;
            }
            let c1 = c1?;
            let c2 = c2?;
            let area1 = imod_contour_area(Some(&c1));
            let area2 = imod_contour_area(Some(&c2));
            let (mut cinmin, coutmin) = if area2 < area1 { (c2, c1) } else { (c1, c2) };

            let inscan = imodel_contour_scan(Some(&cinmin))?;
            let mut inscan = inscan;
            let mut pminin = Ipoint::default();
            let mut pmaxin = Ipoint::default();
            imod_contour_get_bbox(Some(&cinmin), &mut pminin, &mut pmaxin);

            /* build list of ones that overlap new inner contour */
            let mut ninlist = 0usize;
            for i in 0..num_just_in as usize {
                if used[just_in_list[i] as usize] != 0 {
                    continue;
                }
                let co = list[just_in_list[i] as usize] as usize;
                if imodel_scans_overlap(
                    Some(&scancont[co]),
                    pmin[co],
                    pmax[co],
                    Some(&inscan),
                    pminin,
                    pmaxin,
                ) != 0
                {
                    inlist[ninlist] = co as i32;
                    ninlist += 1;
                    used[just_in_list[i] as usize] = 1;
                }
            }

            /* Join these contours, mesh to the created inside one */
            let mut c1 = join_all_contours(
                obj,
                inlist,
                ninlist as i32,
                ninlist as i32,
                joindir,
                None,
                0,
                0,
                Some(&cinmin),
            );
            if joindir < 0 {
                let tmp = c1;
                c1 = Some(cinmin);
                cinmin = tmp?;
            }

            let inlist0 = if ninlist > 0 { inlist[0] } else { 0 };
            if mesh_contours(
                obj,
                obj_mesh,
                c1,
                Some(cinmin),
                surf,
                time,
                scale,
                oddeven,
                inlist0,
                inlist0,
            ) != 0
            {
                return None;
            }
            imod_contour_delete(&mut inscan);
            cout = coutmin;
        }
    }
    Some(cout)
}

/// Original static `cross_cont` (`skinobj.c:2016`).
///
/// Tests whether contour `cout` is crossed by the line segment between
/// `x1s,y1s` and `x1e,y1e`.  It omits segments containing points `st1` and
/// `st2`.
pub fn cross_cont(cout: &Icont, x1s: f32, y1s: f32, x1e: f32, y1e: f32, st1: i32, st2: i32) -> i32 {
    let dx1 = x1e - x1s;
    let dy1 = y1e - y1s;
    for pt in 0..cout.pts.len() as i32 {
        let next = (pt + 1) % cout.pts.len() as i32;
        if pt == st1 || pt == st2 || next == st1 || next == st2 {
            continue;
        }
        let x2s = cout.pts[pt as usize].x;
        let x2e = cout.pts[next as usize].x;
        let y2s = cout.pts[pt as usize].y;
        let y2e = cout.pts[next as usize].y;
        let dx2 = x2s - x2e;
        let dy2 = y2s - y2e;
        let dxs = x2s - x1s;
        let dys = y2s - y1s;
        let den = dx1 * dy2 - dy1 * dx2;
        let tnum = dxs * dy2 - dys * dx2;
        let unum = dx1 * dys - dy1 * dxs;
        if den < 0. {
            if tnum <= 0. && unum <= 0. && tnum >= den && unum >= den {
                return 1;
            }
        } else if tnum >= 0. && unum >= 0. && tnum <= den && unum <= den {
            return 1;
        }
    }
    0
}

/// Original static `evaluate_break` (`skinobj.c:2055`).
///
/// Evaluates a potential break in the outer contour on one slice for
/// `connect_orphans`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_break(
    cout: &Icont,
    list: &[i32],
    used: &[i32],
    just_in_list: &[i32],
    num_just_in: i32,
    scancont: &[Icont],
    pmin: &[Ipoint],
    pmax: &[Ipoint],
    st1: i32,
    st2: i32,
    cur_min: f32,
    pathlen: f32,
    full_area: f32,
    joindir: i32,
    area_only: i32,
    inner_area: &mut f32,
) -> f32 {
    let areapow: f64 = 2.0;

    /* does the biggest possible area, implied by distance and pathlen,
    not enough to give a lower ratio? */
    let dist = imod_point_distance(&cout.pts[st1 as usize], &cout.pts[st2 as usize]);
    let maxarea = (dist + pathlen) * (dist + pathlen) / 12.5664f32;
    let mut ratio = 1.0e20f32;
    if maxarea > 0. {
        ratio = (dist as f64 / (maxarea as f64).powf(areapow)) as f32;
    }
    if (area_only == 0 && ratio > cur_min) || (area_only != 0 && maxarea < cur_min) {
        *inner_area = maxarea;
        return ratio;
    }

    /* second test: does connector cross contour? */
    let x1s = cout.pts[st1 as usize].x;
    let x1e = cout.pts[st2 as usize].x;
    let y1s = cout.pts[st1 as usize].y;
    let y1e = cout.pts[st2 as usize].y;

    *inner_area = 0.;
    if cross_cont(cout, x1s, y1s, x1e, y1e, st1, st2) != 0 {
        return 1.0e20f32;
    }

    /* Make two new contours, require that one be bigger than the original */
    let mut c1: Option<Icont> = None;
    let mut c2: Option<Icont> = None;
    if break_contour_inout(cout, st1, st2, -joindir, &mut c1, &mut c2) != 0 {
        return 1.0e30f32;
    }
    let c1 = c1.unwrap();
    let c2 = c2.unwrap();
    let area1 = imod_contour_area(Some(&c1));
    let area2 = imod_contour_area(Some(&c2));
    let (cin_test, inarea) = if area1 > full_area && area2 < area1 {
        (c2, area2)
    } else if area2 > full_area {
        (c1, area1)
    } else {
        return 1.0e20f32;
    };

    let mut ratio = 1.0e20f32;
    if inarea > 0. {
        ratio = (dist as f64 / (inarea as f64).powf(areapow)) as f32;
    }
    if (area_only == 0 && ratio > cur_min) || (area_only != 0 && inarea < cur_min) {
        *inner_area = inarea;
        return ratio;
    }

    /* scan convert the inner one for testing overlaps */
    let Some(mut inscan) = imodel_contour_scan(Some(&cin_test)) else {
        return 1.0e30f32;
    };
    let mut pminin = Ipoint::default();
    let mut pmaxin = Ipoint::default();
    imod_contour_get_bbox(Some(&cin_test), &mut pminin, &mut pmaxin);

    /* sum the area of orphans overlapping this contour */
    let mut areasum = 0.0f32;
    for i in 0..num_just_in as usize {
        if used[just_in_list[i] as usize] != 0 {
            continue;
        }
        let co = list[just_in_list[i] as usize] as usize;
        let mut frac1 = 0.0f32;
        let mut frac2 = 0.0f32;
        let mut cs1 = scancont[co].clone();
        imodel_overlap_fractions(
            &mut cs1,
            pmin[co],
            pmax[co],
            &mut inscan,
            pminin,
            pmaxin,
            &mut frac1,
            &mut frac2,
        );
        areasum += frac2 * inarea;
    }

    let mut ratio = 1.0e20f32;
    if areasum > 0. {
        ratio = (dist as f64 / (areasum as f64).powf(areapow)) as f32;
    }
    *inner_area = areasum;
    ratio
}

/// Original static `robustCenterOfMass` (`skinobj.c:2158`).
///
/// Gets the center of mass; if it fails, takes the midpoint of the bounding
/// box.
pub fn robust_center_of_mass(cont: &Icont, scan_cont: Option<&Icont>, cm: &mut Ipoint) -> i32 {
    let mut use_cont = match scan_cont {
        Some(sc) => sc.clone(),
        None => cont.clone(),
    };
    if imod_contour_center_of_mass(Some(&mut use_cont), cm) != 0 {
        let mut ll = Ipoint::default();
        let mut ur = Ipoint::default();
        imod_contour_get_bbox(Some(cont), &mut ll, &mut ur);
        cm.x = (ll.x + ur.x) / 2.0f32;
        cm.y = (ll.y + ur.y) / 2.0f32;
        cm.z = (ll.z + ur.z) / 2.0f32;
        return 1;
    }
    0
}

/// Original static `isContConvexIfSimple` (`skinobj.c:2175`).
///
/// Tests whether a polygon is convex, using a quick test valid only if it is
/// non self-intersecting.
pub fn is_cont_convex_if_simple(cont: &Icont) -> i32 {
    let mut all_pos = 0;
    let pts = &cont.pts;
    for ind in 0..cont.pts.len() {
        let next = (ind + 1) % cont.pts.len();
        let sec = (next + 1) % cont.pts.len();
        let cross_prod = (pts[next].x - pts[ind].x) * (pts[sec].y - pts[next].y)
            - (pts[next].y - pts[ind].y) * (pts[sec].x - pts[next].x);
        if ind == 0 {
            all_pos = i32::from(cross_prod > 0.);
        }
        if (cross_prod > 0. && all_pos == 0) || (cross_prod < 0. && all_pos != 0) {
            return 0;
        }
    }
    1
}

/// Original static `concavityAreaFraction` (`skinobj.c:2210`).
///
/// Finds by what fraction the area is smaller than its convex hull.
pub fn concavity_area_fraction(cont: &Icont, hull_diff: &mut f32) -> f32 {
    let psize = cont.pts.len();
    *hull_diff = 0.;
    if psize < 4 {
        return 0.;
    }

    let mut bx = vec![0.0f32; psize];
    let mut by = vec![0.0f32; psize];
    let mut sx = vec![0.0f32; psize];
    let mut sy = vec![0.0f32; psize];
    let mut all_area = 0.0f32;
    let mut hull_area = 0.0f32;
    let mut nvert: i32 = 0;
    let mut xcen = 0.0f32;
    let mut ycen = 0.0f32;
    let pts = &cont.pts;
    for ind in 0..psize {
        let next = (ind + 1) % psize;
        sx[ind] = pts[ind].x;
        sy[ind] = pts[ind].y;
        all_area += pts[ind].x * pts[next].y - pts[ind].y * pts[next].x;
    }
    all_area = 0.5f32 * all_area.abs();
    convex_bound(
        &sx, &sy, 0., 0., &mut bx, &mut by, &mut nvert, &mut xcen, &mut ycen,
    );
    for ind in 0..nvert as usize {
        let next = (ind + 1) % nvert as usize;
        hull_area += bx[ind] * by[next] - by[ind] * bx[next];
    }
    hull_area = 0.5f32 * hull_area.abs();
    if nvert <= 0 {
        return -1.;
    }
    *hull_diff = hull_area - all_area;
    if hull_area > 0. {
        (hull_area - all_area) / hull_area
    } else {
        0.0f32
    }
}

/// Original static `imeshContourCap` (`skinobj.c:2311`).
///
/// Creates a cap mesh on a contour.  `side` has two values: 1 caps on top, -1
/// caps on bottom.
#[allow(clippy::too_many_arguments)]
pub fn imesh_contour_cap(
    obj: &Iobj,
    cont: &mut Icont,
    scan_cont: Option<&Icont>,
    co_num: i32,
    side: i32,
    inside: i32,
    scale: &Ipoint,
    mesh_cont: Option<&Icont>,
) -> Option<Imesh> {
    let mut meshdir = 0;
    let mut do_skeleton = 1;
    let ratio_crit = 1.5f32;
    let frac_crit = 0.005f32;
    let diff_crit = 5.0f32;
    let mut cont_props = DrawProps::default();
    let mut obj_props = DrawProps::default();
    let mut cont_state: i32 = 0;
    let mut surf_state: i32 = 0;
    let state_test = CHANGED_COLOR | CHANGED_FCOLOR | CHANGED_3DWIDTH | CHANGED_TRANS;
    let mut cm = Ipoint::default();
    let mut ratio = 0.0f32;
    let mut length = 0.0f32;
    let mut hull_diff = 0.0f32;

    let failedcm = robust_center_of_mass(cont, scan_cont, &mut cm);
    let direction = imod_cont_z_direction(Some(cont));
    if side == 1 {
        cm.z += 0.5f32;
    } else {
        cm.z -= 0.5f32;
    }

    let mut mesh_cont_owned: Icont = match mesh_cont {
        Some(mc) => mc.clone(),
        None => cont.clone(),
    };

    // Do simple point cap if contour is small or something failed
    if failedcm != 0 || cont.pts.len() < 4 {
        do_skeleton = 0;
    }

    // Otherwise, do skeleton cap if it is either elongated enough or non-convex
    if do_skeleton != 0 {
        imod_contour_long_axis(Some(cont), 5., &mut ratio, &mut length);
        if ratio < ratio_crit {
            if is_cont_convex_if_simple(cont) > 0 {
                do_skeleton = 0;
            } else {
                let concave_frac = concavity_area_fraction(cont, &mut hull_diff);
                if concave_frac < frac_crit && hull_diff < diff_crit {
                    do_skeleton = 0;
                }
            }
        }
    }

    // Do skeleton: get contour, skeletonize it, and use it if that succeeded
    if do_skeleton != 0 {
        if let Some(mut skel_cont) = imod_contour_new() {
            if skeletonize(cont, scan_cont, &cm, &mut skel_cont) != 0 && skel_cont.pts.len() > 4 {
                smooth_reduce_skeleton(&mut skel_cont);
                skel_cont.surf = cont.surf;

                let m = if side == 1 {
                    imesh_contours_cost(
                        obj,
                        &mut mesh_cont_owned,
                        &mut skel_cont,
                        scale,
                        inside,
                        co_num,
                        co_num,
                    )
                } else {
                    imesh_contours_cost(
                        obj,
                        &mut skel_cont,
                        &mut mesh_cont_owned,
                        scale,
                        inside,
                        co_num,
                        co_num,
                    )
                };

                imod_contour_delete(&mut skel_cont);
                return m;
            }
            imod_contour_delete(&mut skel_cont);
        }
    }

    /* ccw on top , cw on bottom (unless inside!)*/
    if ((side < 0) && (direction == IMOD_CONTOUR_COUNTER_CLOCKWISE))
        || ((side > 0) && (direction == IMOD_CONTOUR_CLOCKWISE))
    {
        meshdir = 1;
    }
    if inside != 0 {
        meshdir = 1 - meshdir;
    }

    let pt = istore_cont_surf_draw_props(
        &obj.store,
        &obj_props,
        &mut cont_props,
        co_num,
        cont.surf,
        &mut cont_state,
        &mut surf_state,
    );
    if SKIN_FLAGS.with(|c| c.get()) & IMESH_MK_SURF == 0 {
        cont_state = pt;
    }

    make_cap_mesh(
        &mesh_cont_owned,
        &cm,
        meshdir,
        &cont_props,
        cont_state,
        state_test,
    )
}

/// Original static `smoothReduceSkeleton` (`skinobj.c:2404`).
///
/// Smooths out the jagged lines from following pixel boundaries and reduces
/// points in the skeleton contour.
pub fn smooth_reduce_skeleton(cont: &mut Icont) -> i32 {
    let max_fit: i32 = 5;
    let mut xx = [0.0f32; 11];
    let mut xxsq = [0.0f32; 11];
    let mut aa = 0.0f32;
    let mut bb = 0.0f32;
    let mut cc = 0.0f32;

    if cont.pts.len() < 5 {
        return 1;
    }

    // Allocate needed arrays
    let num_pts = cont.pts.len() as i32;
    let mut pair_ind: Vec<i32> = vec![-1; num_pts as usize];
    let mut xrot: Vec<f32> = vec![0.; (num_pts + 4) as usize];
    let mut yrot: Vec<f32> = vec![0.; (num_pts + 4) as usize];
    let Some(mut red_cont) = imod_contour_new() else {
        return 1;
    };
    red_cont.pts = vec![Ipoint::default(); (num_pts / 2 + 2) as usize];

    // Initialize paired indexes then search for pairs
    for ind in 0..num_pts as usize {
        pair_ind[ind] = -1;
    }
    let mut base: i32 = 0;
    while base < num_pts - 1 {
        if pair_ind[base as usize] >= 0 {
            base += 1;
            continue;
        }

        // Look for a paired point and record indexes if found
        let mut ind = base + 1;
        while ind < num_pts {
            if pair_ind[ind as usize] < 0
                && ((cont.pts[base as usize].x - cont.pts[ind as usize].x) as f64).abs() < 0.01
                && ((cont.pts[base as usize].y - cont.pts[ind as usize].y) as f64).abs() < 0.01
            {
                pair_ind[base as usize] = ind;
                pair_ind[ind as usize] = base;
                break;
            }
            ind += 1;
        }

        // None found, go on to next
        if ind == num_pts {
            base += 1;
            continue;
        }

        // Found: walk forward with base and backward with ind as long as they match
        while base + 1 < ind - 1
            && pair_ind[(base + 1) as usize] < 0
            && pair_ind[(ind - 1) as usize] < 0
            && (cont.pts[(base + 1) as usize].x - cont.pts[(ind - 1) as usize].x).abs() < 0.01f32
            && (cont.pts[(base + 1) as usize].y - cont.pts[(ind - 1) as usize].y).abs() < 0.01f32
        {
            base += 1;
            ind -= 1;
            pair_ind[base as usize] = ind;
            pair_ind[ind as usize] = base;
        }
        base += 1;
    }

    // Now look for segments bracketed by unpaired ones, or by boundaries between
    // different pairings
    let mut base: i32 = 0;
    while base < num_pts - 1 {
        if !(pair_ind[(base + 1) as usize] >= 0
            && (pair_ind[base as usize] == -1
                || pair_ind[base as usize] != pair_ind[(base + 1) as usize] + 1))
        {
            base += 1;
            continue;
        }
        let mut ind = base + 2;
        while ind < num_pts {
            if pair_ind[ind as usize] < 0
                || pair_ind[ind as usize] != pair_ind[(ind - 1) as usize] - 1
            {
                break;
            }
            ind += 1;
        }
        if ind == num_pts {
            ind -= 1;
        }

        // The segment start and end are fixed points
        let mut seg_start = base + 1;
        let mut seg_end = ind - 1;
        if pair_ind[base as usize] == -1
            && pair_ind[seg_start as usize] == (base + num_pts - 1) % num_pts
        {
            seg_start -= 1;
        }
        if pair_ind[ind as usize] == -1 && pair_ind[seg_end as usize] == (ind + 1) % num_pts {
            seg_end += 1;
        }

        if seg_end + 1 - seg_start < 3 {
            base += 1;
            continue;
        }

        let nin_seg = seg_end + 1 - seg_start;
        if nin_seg >= 4 {
            let mut fit_start: i32 = 0;
            while fit_start <= 0.max(nin_seg - max_fit) {
                // For each fit, get start and end points of fit, determine an angle
                // to rotate it to be level, rotate
                let fit_end = (nin_seg - 1).min(fit_start + max_fit - 1);
                let nin_fit = fit_end + 1 - fit_start;
                let seg_angle = (cont.pts[(seg_start + fit_end) as usize].y
                    - cont.pts[(seg_start + fit_start) as usize].y)
                    .atan2(
                        cont.pts[(seg_start + fit_end) as usize].x
                            - cont.pts[(seg_start + fit_start) as usize].x,
                    );
                let cos_ang = seg_angle.cos();
                let sin_ang = seg_angle.sin();

                let mut pt_ind = 0usize;
                for jnd in seg_start + fit_start..=seg_start + fit_end {
                    xrot[pt_ind] =
                        cos_ang * cont.pts[jnd as usize].x + sin_ang * cont.pts[jnd as usize].y;
                    yrot[pt_ind] =
                        -sin_ang * cont.pts[jnd as usize].x + cos_ang * cont.pts[jnd as usize].y;
                    pt_ind += 1;
                }

                // load xx and xxsq, and fit
                for jnd in 0..nin_fit as usize {
                    xx[jnd] = xrot[jnd] - xrot[0];
                    xxsq[jnd] = xx[jnd] * xx[jnd];
                }
                ls_fit2(&xx, &xxsq, &yrot, nin_fit, &mut aa, &mut bb, Some(&mut cc));

                // Replace the midpoint at least, or back to start or forward to end
                let mut fill_st = nin_fit / 2;
                let mut fill_end = nin_fit / 2;
                if fit_start == 0 {
                    fill_st = 1;
                }
                if fit_end == nin_seg - 1 {
                    fill_end = nin_fit - 2;
                }
                for jnd in fill_st..=fill_end {
                    let yfit = aa * xx[jnd as usize] + bb * xxsq[jnd as usize] + cc;
                    let pt_ind = seg_start + fit_start + jnd;
                    cont.pts[pt_ind as usize].x = cos_ang * xrot[jnd as usize] - sin_ang * yfit;
                    cont.pts[pt_ind as usize].y = sin_ang * xrot[jnd as usize] + cos_ang * yfit;
                    let paired = pair_ind[pt_ind as usize];
                    cont.pts[paired as usize] = cont.pts[pt_ind as usize];
                }
                fit_start += 1;
            }
        }

        // Load the contour for reduction
        let mut red_size = 0usize;
        for jnd in seg_start..=seg_end {
            red_cont.pts[red_size] = cont.pts[jnd as usize];
            red_size += 1;
        }
        red_cont.pts.truncate(red_size);
        imod_contour_reduce(Some(&mut red_cont), 0.25f32);

        // Replace the points, duplicating the last one as needed
        for jnd in seg_start + 1..seg_end {
            let src = red_cont.pts[(red_cont.pts.len() as i32 - 1).min(jnd - seg_start) as usize];
            cont.pts[jnd as usize] = src;
            let paired = pair_ind[jnd as usize];
            cont.pts[paired as usize] = src;
        }
        red_cont
            .pts
            .resize((num_pts / 2 + 2) as usize, Ipoint::default());

        // Mark as done by changing paired index to - 2
        for jnd in base + 1..ind {
            let paired = pair_ind[jnd as usize];
            pair_ind[paired as usize] = -2;
            pair_ind[jnd as usize] = -2;
        }

        // Advance base
        base = seg_end - 1;
        base += 1;
    }

    // Remove duplicated points
    imod_contour_unique(cont);

    drop(pair_ind);
    drop(xrot);
    drop(yrot);
    imod_contour_delete(&mut red_cont);
    0
}

/// Original static `interpolate_point` (`skinobj.c:2545`).
pub fn interpolate_point(pt1: Ipoint, pt2: Ipoint, frac: f32, pt3: &mut Ipoint) {
    pt3.x = frac * pt2.x + (1. - frac) * pt1.x;
    pt3.y = frac * pt2.y + (1. - frac) * pt1.y;
    pt3.z = frac * pt2.z + (1. - frac) * pt1.z;
}

/// Original static `backoff_overlap` (`skinobj.c:2556`).
///
/// Removes points from `c1` that are inside `c2`, adding two new points that
/// are just outside `c2` by 1/10 - 2/10 of the crossing segments.
pub fn backoff_overlap(c1: &mut Icont, c2: &Icont) {
    let mut pt3 = Ipoint::default();
    let mut ptbefore = Ipoint::default();
    let mut ptafter = Ipoint::default();

    /* find first point outside of c1 */
    let mut first_out = 0usize;
    while first_out < c1.pts.len() {
        if imod_point_inside_cont(c2, &c1.pts[first_out]) == 0 {
            break;
        }
        first_out += 1;
    }

    /* give up if never got out */
    if first_out == c1.pts.len() {
        return;
    }

    /* now find last point out from there, and first point back in*/
    let mut last_out = first_out;
    let mut first_in = last_out + 1;
    let mut i = 0usize;
    while i < c1.pts.len() {
        if first_in == c1.pts.len() {
            first_in = 0;
        }
        if imod_point_inside_cont(c2, &c1.pts[first_in]) != 0 {
            break;
        }
        last_out = first_in;
        first_in += 1;
        i += 1;
    }

    /* give up if never got in */
    if i == c1.pts.len() {
        return;
    }

    /* find first interpolated point that's outside and go 1 more tenth out*/
    let mut before = 0;
    for i in (2..=9).rev() {
        interpolate_point(
            c1.pts[last_out],
            c1.pts[first_in],
            0.1f32 * i as f32,
            &mut pt3,
        );
        if imod_point_inside_cont(c2, &pt3) == 0 {
            interpolate_point(
                c1.pts[last_out],
                c1.pts[first_in],
                0.1f32 * (i - 1) as f32,
                &mut ptbefore,
            );
            before = 1;
            break;
        }
    }

    /* Now find first point outside c2 again from here */
    let mut last_in = first_in;
    let mut first_out = last_in + 1;
    let mut i = 0usize;
    while i < c1.pts.len() {
        if first_out == c1.pts.len() {
            first_out = 0;
        }
        if imod_point_inside_cont(c2, &c1.pts[first_out]) == 0 {
            break;
        }
        last_in = first_out;
        first_out += 1;
        i += 1;
    }

    /* find first interpolated point that's outside and go 1 more tenth out*/
    let mut after = 0;
    for i in (2..=9).rev() {
        interpolate_point(
            c1.pts[first_out],
            c1.pts[last_in],
            0.1f32 * i as f32,
            &mut pt3,
        );
        if imod_point_inside_cont(c2, &pt3) == 0 {
            interpolate_point(
                c1.pts[first_out],
                c1.pts[last_in],
                0.1f32 * (i - 1) as f32,
                &mut ptafter,
            );
            after = 1;
            break;
        }
    }

    /* Add the two points after the last one out */
    if before != 0 {
        last_out += 1;
        imod_point_add(c1, Some(ptbefore), last_out as i32);
    }
    if after != 0 {
        last_out += 1;
        imod_point_add(c1, Some(ptafter), last_out as i32);
    }

    /* Just delete all points inside */
    let mut i: i32 = 0;
    while i < c1.pts.len() as i32 {
        if imod_point_inside_cont(c2, &c1.pts[i as usize]) != 0 {
            imod_point_delete(c1, i);
            i -= 1;
        }
        i += 1;
    }
}

/// Original static `eliminate_overlap` (`skinobj.c:2652`).
///
/// Eliminates overlap between `c1` and `c2` by backing off `c1` from being
/// inside `c2` then backing off `c2` from being inside `c1`, iterating twice.
pub fn eliminate_overlap(c1: &mut Icont, c2: &mut Icont) -> i32 {
    for i in 0..2 {
        if imodel_contour_overlap(c1, c2) == 0 {
            return i;
        }
        let c1cp = c1.clone();
        backoff_overlap(c1, c2);
        backoff_overlap(c2, &c1cp);
    }
    1
}

/// Original: `CHECK_DIVISIONS` (`skinobj.c:2668`).
const CHECK_DIVISIONS: i32 = 0;
/// Original: `CHECK_MIN_FRACTION` (`skinobj.c:2669`).
const CHECK_MIN_FRACTION: f64 = 0.45;
/// Original: `CHECK_MIN_DISTANCE` (`skinobj.c:2670`).
const CHECK_MIN_DISTANCE: f64 = 2.;
/// Original: `INSIDE_MAX_DISTANCE` (`skinobj.c:2671`).
const INSIDE_MAX_DISTANCE: f64 = 0.1;
/// Original: `INSIDE_MAX_FRACTION` (`skinobj.c:2672`).
const INSIDE_MAX_FRACTION: f64 = 0.1;

/// Original static `check_legal_joiner` (`skinobj.c:2686`).
///
/// Checks whether it is valid to join contours `c1` and `c2` with a connector
/// from `pt1` in `c1` to `pt2` in `c2`.
#[allow(clippy::too_many_arguments)]
pub fn check_legal_joiner(
    pt1: Ipoint,
    pt2: Ipoint,
    c1: &Icont,
    c2: &Icont,
    samelevel: i32,
    obj: &Iobj,
    olist: Option<&[i32]>,
    num_other: i32,
    num_other_same: i32,
    other: Option<&Icont>,
) -> i32 {
    let mut mid = Ipoint::default();
    let mut retval = 0;
    let ncheck = CHECK_DIVISIONS;
    let len = (((pt1.x - pt2.x) as f64 * (pt1.x - pt2.x) as f64)
        + ((pt1.y - pt2.y) as f64 * (pt1.y - pt2.y) as f64))
        .sqrt() as f32;

    /* check that a connector trimmed by 0.1 pixel or 0.1 of length does not
    cross either contour */
    if len > 1.0e-5 {
        let mut frac = (INSIDE_MAX_DISTANCE / len as f64) as f32;
        if frac as f64 > INSIDE_MAX_FRACTION {
            frac = INSIDE_MAX_FRACTION as f32;
        }
        let xs1 = (1.0f32 - frac) * pt1.x + frac * pt2.x;
        let ys1 = (1.0f32 - frac) * pt1.y + frac * pt2.y;

        let xs2 = (1.0f32 - frac) * pt2.x + frac * pt1.x;
        let ys2 = (1.0f32 - frac) * pt2.y + frac * pt1.y;
        if cross_cont(c1, xs1, ys1, xs2, ys2, -1, -1) != 0 {
            return 0;
        }
        if cross_cont(c2, xs1, ys1, xs2, ys2, -1, -1) != 0 {
            return 0;
        }
    }

    /* check one or more points in middle of connector to verify proper
    relation to contours on next slice */
    let mut minfrac = 0.45f32;
    if len > 1.0e-5 {
        minfrac = (CHECK_MIN_DISTANCE / len as f64) as f32;
    }
    if (minfrac as f64) < CHECK_MIN_FRACTION {
        minfrac = CHECK_MIN_FRACTION as f32;
    }
    if minfrac > 0.5 {
        minfrac = 0.5;
    }

    for j in 0..=CHECK_DIVISIONS {
        retval = 1;
        if samelevel == 0 {
            retval = 0;
        }

        let mut frac = 0.5f32;
        if ncheck > 0 {
            frac = minfrac + j as f32 * (1.0f32 - 2. * minfrac) / ncheck as f32;
        }
        mid.x = (1.0f32 - frac) * pt1.x + frac * pt2.x;
        mid.y = (1.0f32 - frac) * pt1.y + frac * pt2.y;
        if let Some(other) = other {
            if imod_point_inside_cont(other, &mid) != 0 {
                return retval;
            }
        }
        if let Some(olist) = olist {
            for i in 0..num_other_same as usize {
                if imod_point_inside_cont(&obj.cont[olist[i] as usize], &mid) != 0 {
                    return retval;
                }
            }
            retval = 1 - retval;
            for i in num_other_same as usize..num_other as usize {
                if imod_point_inside_cont(&obj.cont[olist[i] as usize], &mid) != 0 {
                    return retval;
                }
            }
        } else {
            retval = 1 - retval;
        }
    }
    retval
}

/// Original static `segment_mm` (`skinobj.c:2755`).
///
/// Evaluates the min and max coordinates for points in contour `cont` from
/// `ptst` to `ptnd`.
pub fn segment_mm(
    cont: &Icont,
    ptst: i32,
    ptnd: i32,
    xmin: &mut f32,
    xmax: &mut f32,
    ymin: &mut f32,
    ymax: &mut f32,
) {
    let mut pt = ptst - 1;
    let pts = &cont.pts;
    *xmin = pts[ptst as usize].x;
    *ymin = pts[ptst as usize].y;
    *xmax = *xmin;
    *ymax = *ymin;
    loop {
        pt = (pt + 1) % cont.pts.len() as i32;
        if *xmin > pts[pt as usize].x {
            *xmin = pts[pt as usize].x;
        }
        if *xmax < pts[pt as usize].x {
            *xmax = pts[pt as usize].x;
        }
        if *ymin > pts[pt as usize].y {
            *ymin = pts[pt as usize].y;
        }
        if *ymax < pts[pt as usize].y {
            *ymax = pts[pt as usize].y;
        }
        if pt == ptnd {
            break;
        }
    }
}

/// Original: `SEGSCAN_DIV` (`skinobj.c:2779`).
const SEGSCAN_DIV: usize = 30;

/// Original static `scan_points_to_segments` (`skinobj.c:2789`).
///
/// Finds the closest approach between a point in `c1` and a line segment in
/// `c2`.
#[allow(clippy::too_many_arguments)]
pub fn scan_points_to_segments(
    c1: &Icont,
    c2: &Icont,
    legalmin: &mut f64,
    dsqrmin: &mut f64,
    pt1: &mut i32,
    pt2: &mut i32,
    tbest: &mut f32,
    close: &mut Ipoint,
    docheck: i32,
    samelevel: i32,
    obj: &Iobj,
    olist: Option<&[i32]>,
    num_other: i32,
    num_other_same: i32,
    other: Option<&Icont>,
) {
    let mut xmin1 = [0.0f32; SEGSCAN_DIV];
    let mut xmax1 = [0.0f32; SEGSCAN_DIV];
    let mut ymin1 = [0.0f32; SEGSCAN_DIV];
    let mut ymax1 = [0.0f32; SEGSCAN_DIV];
    let mut xmin2 = [0.0f32; SEGSCAN_DIV];
    let mut xmax2 = [0.0f32; SEGSCAN_DIV];
    let mut ymin2 = [0.0f32; SEGSCAN_DIV];
    let mut ymax2 = [0.0f32; SEGSCAN_DIV];
    let mut bbsep = [0.0f32; SEGSCAN_DIV * SEGSCAN_DIV];
    let mut indorder = [0usize; SEGSCAN_DIV * SEGSCAN_DIV];
    let mut cltmp = Ipoint::default();

    /* divide each contour into segments and get the mins and maxes of each */
    let mut num_div1 = SEGSCAN_DIV as i32;
    if (c1.pts.len() as i32) < num_div1 {
        num_div1 = c1.pts.len() as i32;
    }
    let num_per_div1 = (c1.pts.len() as i32 + num_div1 - 1) / num_div1;
    num_div1 = (c1.pts.len() as i32 + num_per_div1 - 1) / num_per_div1;
    for j in 0..num_div1 as usize {
        let pt_start = num_per_div1 * j as i32;
        let mut pt_end = pt_start + num_per_div1 - 1;
        if pt_end >= c1.pts.len() as i32 {
            pt_end = c1.pts.len() as i32 - 1;
        }
        segment_mm(
            c1,
            pt_start,
            pt_end,
            &mut xmin1[j],
            &mut xmax1[j],
            &mut ymin1[j],
            &mut ymax1[j],
        );
    }

    let mut num_div2 = SEGSCAN_DIV as i32;
    if (c2.pts.len() as i32) < num_div2 {
        num_div2 = c2.pts.len() as i32;
    }
    let num_per_div2 = (c2.pts.len() as i32 + num_div2 - 1) / num_div2;
    num_div2 = (c2.pts.len() as i32 + num_per_div2 - 1) / num_per_div2;
    for j in 0..num_div2 as usize {
        let pt_start = num_per_div2 * j as i32;
        let mut pt_end = pt_start + num_per_div2;
        if pt_end >= c2.pts.len() as i32 {
            pt_end = 0;
        }
        segment_mm(
            c2,
            pt_start,
            pt_end,
            &mut xmin2[j],
            &mut xmax2[j],
            &mut ymin2[j],
            &mut ymax2[j],
        );
    }

    /* measure separations between divisions, remember to square them! */
    for i in 0..num_div1 as usize {
        for j in 0..num_div2 as usize {
            let ind = i * num_div2 as usize + j;
            let mut sep = 0.0f32;
            if sep < xmin1[i] - xmax2[j] {
                sep = xmin1[i] - xmax2[j];
            }
            if sep < xmin2[j] - xmax1[i] {
                sep = xmin2[j] - xmax1[i];
            }
            if sep < ymin1[i] - ymax2[j] {
                sep = ymin1[i] - ymax2[j];
            }
            if sep < ymin2[j] - ymax1[i] {
                sep = ymin2[j] - ymax1[i];
            }
            bbsep[ind] = sep * sep;
            indorder[ind] = ind;
        }
    }

    /* order by increasing separation */
    let total = (num_div1 * num_div2) as usize;
    for i in 0..total.saturating_sub(1) {
        for j in i + 1..total {
            if bbsep[indorder[i]] > bbsep[indorder[j]] {
                indorder.swap(i, j);
            }
        }
    }

    let pts1 = &c1.pts;
    let pts2 = &c2.pts;

    /* loop on the divisions in that order, skip ones above current minimum */
    for i in 0..total {
        if bbsep[indorder[i]] as f64 >= *legalmin {
            continue;
        }
        let ind = indorder[i] / num_div2 as usize;
        let mut pta = ind as i32 * num_per_div1;
        let mut pta_end = pta + num_per_div1;
        if pta_end > c1.pts.len() as i32 {
            pta_end = c1.pts.len() as i32;
        }
        let ind = indorder[i] % num_div2 as usize;
        let pt_start = ind as i32 * num_per_div2;
        let mut pt_end = pt_start + num_per_div2;
        if pt_end > c2.pts.len() as i32 {
            pt_end = c2.pts.len() as i32;
        }
        while pta < pta_end {
            let x0 = pts1[pta as usize].x;
            let y0 = pts1[pta as usize].y;
            let mut xs = pts2[pt_start as usize].x;
            let mut ys = pts2[pt_start as usize].y;
            for ptb in pt_start..pt_end {
                /* determine closest approach between point in c1 and segment in c2 */
                let next = (ptb + 1) % c2.pts.len() as i32;
                let xe = pts2[next as usize].x;
                let ye = pts2[next as usize].y;
                let mut dx = xe - xs;
                let mut dy = ye - ys;
                let mut t = 0.0f32;
                if dx != 0. || dy != 0. {
                    t = ((x0 - xs) * dx + (y0 - ys) * dy) / (dx * dx + dy * dy);
                    if t < 0. {
                        t = 0.;
                    }
                    if t > 1. {
                        t = 1.;
                    }
                }
                cltmp.x = xs + t * dx;
                cltmp.y = ys + t * dy;
                dx = cltmp.x - x0;
                dy = cltmp.y - y0;
                let dsqr = dx * dx + dy * dy;
                if (dsqr as f64) < *legalmin {
                    /* if less than the current min, check legality if required */
                    let mut legal = 1;
                    if docheck != 0 {
                        legal = check_legal_joiner(
                            pts1[pta as usize],
                            cltmp,
                            c1,
                            c2,
                            samelevel,
                            obj,
                            olist,
                            num_other,
                            num_other_same,
                            other,
                        );
                    }
                    /* if still legal, save new min */
                    if legal != 0 {
                        *legalmin = dsqr as f64;
                        *pt1 = pta;
                        *pt2 = ptb;
                        *tbest = t;
                        cltmp.z = pts2[ptb as usize].z;
                        *close = cltmp;
                    }
                }
                if *legalmin > 1.0e20 && (dsqr as f64) < *dsqrmin {
                    /* or, if no legal min found yet, keep track of nonlegal minimum */
                    *dsqrmin = dsqr as f64;
                    *pt1 = pta;
                    *pt2 = ptb;
                    *tbest = t;
                    cltmp.z = pts2[ptb as usize].z;
                    *close = cltmp;
                }
                xs = xe;
                ys = ye;
            }
            pta += 1;
        }
    }
}

/// Original static `find_closest_contour` (`skinobj.c:2980`).
///
/// Finds the closest contour and connecting point with a legal connector.
#[allow(clippy::too_many_arguments)]
pub fn find_closest_contour(
    tcont: &Icont,
    onecont: Option<&Icont>,
    obj: &Iobj,
    list: &[i32],
    ncont_in: i32,
    used: &[i32],
    pt1min: &mut i32,
    pt2min: &mut i32,
    firstbest: &mut i32,
    tbest: &mut f32,
    close: &mut Ipoint,
    olist: Option<&[i32]>,
    num_other: i32,
    num_other_same: i32,
    other: Option<&Icont>,
    num_same_level: i32,
) -> i32 {
    let mut testpt1 = Ipoint::default();
    let mut testpt2 = Ipoint::default();
    let mut closet = Ipoint::default();
    let mut jmin: i32 = 0;
    let mut samelevel: i32;
    let mut pt1mint: i32 = 0;
    let mut pt2mint: i32 = 0;

    /* first pass: find contour with absolute closest approach regardless
    of legality of connector */
    let mut legalmin = 1.0e30f64;
    let mut lastmin = 1.0e30f64;
    let mut distmin = 1.0e30f64;
    let ncont = if onecont.is_some() { 2 } else { ncont_in };
    for j in 1..ncont {
        let jcont: &Icont = match onecont {
            Some(c) => c,
            None => {
                if used[j as usize] != 0 {
                    continue;
                }
                &obj.cont[list[j as usize] as usize]
            }
        };

        /* First look for designated connectors */
        let mut num_con = 0;
        let conn: Option<Vec<Connector>> = make_connectors(tcont, jcont, &mut num_con, 1, 0);
        if let Some(conn) = conn {
            *firstbest = 1;
            *pt1min = conn[0].b1;
            *pt2min = conn[0].t1;
            *tbest = 0.;
            *close = jcont.pts[conn[0].t1 as usize];
            jmin = j;
            testpt1 = tcont.pts[*pt1min as usize];
            testpt2 = *close;
            break;
        }

        /* first measure points of tcont to segments of jcont; then vice versa */
        scan_points_to_segments(
            tcont,
            jcont,
            &mut legalmin,
            &mut distmin,
            &mut pt1mint,
            &mut pt2mint,
            tbest,
            &mut closet,
            0,
            0,
            obj,
            olist,
            num_other,
            num_other_same,
            other,
        );
        if legalmin < lastmin {
            lastmin = legalmin;
            *firstbest = 1;
            *pt1min = pt1mint;
            *pt2min = pt2mint;
            *close = closet;
            jmin = j;
            testpt1 = tcont.pts[*pt1min as usize];
            testpt2 = *close;
        }
        scan_points_to_segments(
            jcont,
            tcont,
            &mut legalmin,
            &mut distmin,
            &mut pt2mint,
            &mut pt1mint,
            tbest,
            &mut closet,
            0,
            0,
            obj,
            olist,
            num_other,
            num_other_same,
            other,
        );
        if legalmin < lastmin {
            lastmin = legalmin;
            *firstbest = 0;
            *pt1min = pt1mint;
            *pt2min = pt2mint;
            *close = closet;
            jmin = j;
            testpt2 = jcont.pts[*pt2min as usize];
            testpt1 = *close;
        }
    }

    samelevel = 1;
    let jcont: &Icont = match onecont {
        Some(c) => c,
        None => {
            if jmin >= num_same_level {
                samelevel = 0;
            }
            &obj.cont[list[jmin as usize] as usize]
        }
    };
    if check_legal_joiner(
        testpt1,
        testpt2,
        tcont,
        jcont,
        samelevel,
        obj,
        olist,
        num_other,
        num_other_same,
        other,
    ) == 0
    {
        /* if first pass didn't give legal connector, then find closest legal
        connector by brute force */
        distmin = 1.0e30;
        legalmin = 1.0e30;
        lastmin = 1.0e30;
        for j in 1..ncont {
            let mut samelevel = 1;
            let jcont: &Icont = match onecont {
                Some(c) => c,
                None => {
                    if used[j as usize] != 0 {
                        continue;
                    }
                    if j >= num_same_level {
                        samelevel = 0;
                    }
                    &obj.cont[list[j as usize] as usize]
                }
            };

            scan_points_to_segments(
                tcont,
                jcont,
                &mut legalmin,
                &mut distmin,
                &mut pt1mint,
                &mut pt2mint,
                tbest,
                &mut closet,
                1,
                samelevel,
                obj,
                olist,
                num_other,
                num_other_same,
                other,
            );
            if legalmin < lastmin || (legalmin > 1.0e20 && distmin < lastmin) {
                lastmin = legalmin;
                if legalmin > 1.0e20 {
                    lastmin = distmin;
                }
                *firstbest = 1;
                *pt1min = pt1mint;
                *pt2min = pt2mint;
                *close = closet;
                jmin = j;
            }
            scan_points_to_segments(
                jcont,
                tcont,
                &mut legalmin,
                &mut distmin,
                &mut pt2mint,
                &mut pt1mint,
                tbest,
                &mut closet,
                1,
                samelevel,
                obj,
                olist,
                num_other,
                num_other_same,
                other,
            );
            if legalmin < lastmin || (legalmin > 1.0e20 && distmin < lastmin) {
                lastmin = legalmin;
                if legalmin > 1.0e20 {
                    lastmin = distmin;
                }
                *firstbest = 0;
                *pt1min = pt1mint;
                *pt2min = pt2mint;
                *close = closet;
                jmin = j;
            }
        }
    }

    jmin
}

/// Original static `join_all_contours` (`skinobj.c:3117`).
///
/// Joins together all `ncont` contours in `list`, of which `num_same_level`
/// are at the same level.
#[allow(clippy::too_many_arguments)]
pub fn join_all_contours(
    obj: &Iobj,
    list: &[i32],
    ncont: i32,
    num_same_level: i32,
    fill: i32,
    olist: Option<&[i32]>,
    num_other: i32,
    num_other_same: i32,
    other_in: Option<&Icont>,
) -> Option<Icont> {
    let mut other = other_in;
    let mut cont_props = DrawProps::default();
    let obj_props = DrawProps::default();
    let mut pt_props = DrawProps::default();
    let mut cont_state: i32 = 0;
    let mut surf_state: i32 = 0;
    let gen_flags = CHANGED_COLOR | CHANGED_FCOLOR | CHANGED_3DWIDTH | CHANGED_TRANS;

    /* start with first contour if it is valid # */
    let mut tcont: Icont = if list[0] >= 0 {
        imod_contour_dup(&obj.cont[list[0] as usize])?
    } else {
        // Otherwise use the contour in other and cancel its other meaning
        let o = other?;
        let dup = imod_contour_dup(o)?;
        other = None;
        dup
    };

    if ncont < 2 {
        return Some(tcont);
    }

    /* If there are contour or surface properties, convert them to a property
    for the first point of the contour */
    let mut j = istore_cont_surf_draw_props(
        &obj.store,
        &obj_props,
        &mut cont_props,
        list[0],
        tcont.surf,
        &mut cont_state,
        &mut surf_state,
    );
    if SKIN_FLAGS.with(|c| c.get()) & IMESH_MK_SURF == 0 {
        cont_state = j;
    }
    if cont_state != 0 {
        j = istore_list_point_props(&tcont.store, &cont_props, &mut pt_props, 0);
        cont_state &= !j;
        istore_generate_items(&mut tcont.store, &cont_props, cont_state, 0, gen_flags);
    }

    let mut used: Vec<i32> = vec![0; ncont as usize];

    /* loop to connect one more contour each time */
    for _i in 1..ncont {
        let mut tptmin: i32 = 0;
        let mut jptmin: i32 = 0;
        let mut firstbest: i32 = 0;
        let mut tbest: f32 = 0.;
        let mut close = Ipoint::default();
        let jmin = find_closest_contour(
            &tcont,
            None,
            obj,
            list,
            ncont,
            &used,
            &mut tptmin,
            &mut jptmin,
            &mut firstbest,
            &mut tbest,
            &mut close,
            olist,
            num_other,
            num_other_same,
            other,
            num_same_level,
        );

        let mut jcont = imod_contour_dup(&obj.cont[list[jmin as usize] as usize])?;
        used[jmin as usize] = 1;

        /* Transfer contour or surface properties to the first point of contour */
        let mut j = istore_cont_surf_draw_props(
            &obj.store,
            &obj_props,
            &mut cont_props,
            list[jmin as usize],
            jcont.surf,
            &mut cont_state,
            &mut surf_state,
        );
        if SKIN_FLAGS.with(|c| c.get()) & IMESH_MK_SURF == 0 {
            cont_state = j;
        }
        if cont_state != 0 {
            j = istore_list_point_props(&jcont.store, &cont_props, &mut pt_props, 0);
            cont_state &= !j;
            istore_generate_items(&mut jcont.store, &cont_props, cont_state, 0, gen_flags);
        }

        /* eliminate overlap for contours on the basic level. */
        if jmin < num_same_level && eliminate_overlap(&mut tcont, &mut jcont) != 0 {
            find_closest_contour(
                &tcont,
                Some(&jcont.clone()),
                obj,
                list,
                ncont,
                &used,
                &mut tptmin,
                &mut jptmin,
                &mut firstbest,
                &mut tbest,
                &mut close,
                olist,
                num_other,
                num_other_same,
                other,
                num_same_level,
            );
        }

        /* Add the connector point if it is inside a segment */
        if tbest > 0. && tbest < 1. {
            if firstbest != 0 {
                jptmin += 1;
                imod_point_add(&mut jcont, Some(close), jptmin);
            } else {
                tptmin += 1;
                imod_point_add(&mut tcont, Some(close), tptmin);
            }
        }

        let mut counterdir = 0;
        if jmin >= num_same_level {
            counterdir = 1;
        }

        let mut tempcont = tcont;
        tcont = imod_contour_join(
            Some(&mut tempcont),
            Some(&mut jcont),
            tptmin,
            jptmin,
            fill,
            counterdir,
        )?;
        imod_contour_delete(&mut tempcont);
        imod_contour_delete(&mut jcont);
    }

    Some(tcont)
}

/// Original static `subtract_scan_contours` (`skinobj.c:3222`).
///
/// Modifies scan contour `cs1` to exclude the area covered by `cs2`.
pub fn subtract_scan_contours(cs1: &mut Icont, cs2: &Icont) {
    let mut jstrt: i32 = 0;
    let mut i: i32 = 0;
    while i < cs1.pts.len() as i32 - 1 {
        let mut j = jstrt;
        while j < cs2.pts.len() as i32 - 1 {
            if cs1.pts[i as usize].y == cs2.pts[j as usize].y {
                let s1 = cs1.pts[i as usize].x;
                let e1 = cs1.pts[(i + 1) as usize].x;
                let s2 = cs2.pts[j as usize].x;
                let e2 = cs2.pts[(j + 1) as usize].x;

                if s1 < e2 && s2 < e1 {
                    if s1 < s2 && e1 <= e2 {
                        cs1.pts[(i + 1) as usize].x = s2; /* truncate right end */
                    } else if s1 >= s2 && e1 > e2 {
                        cs1.pts[i as usize].x = e2; /* truncate left end */
                    } else if s1 >= s2 && e1 < e2 {
                        /* wipe out the line completely */
                        imod_point_delete(cs1, i);
                        imod_point_delete(cs1, i);
                        i -= 2;
                        break;
                    } else {
                        /* split the line in two */
                        let p1 = cs2.pts[j as usize];
                        let p2 = cs2.pts[(j + 1) as usize];
                        imod_point_add(cs1, Some(p1), i + 1);
                        imod_point_add(cs1, Some(p2), i + 2);
                    }
                }
            } else if cs1.pts[i as usize].y > cs2.pts[j as usize].y {
                jstrt = j;
            } else {
                break;
            }
            j += 2;
        }
        i += 2;
    }
}

/// Original static `mesh_contours` (`skinobj.c:3267`).
///
/// Meshes `bcont` to `tcont` and adds it to the `objMesh` array.
#[allow(clippy::too_many_arguments)]
pub fn mesh_contours(
    obj: &Iobj,
    obj_mesh: &mut Vec<Imesh>,
    bcont: Option<Icont>,
    tcont: Option<Icont>,
    surf: i32,
    time: i32,
    scale: &Ipoint,
    mut inside: i32,
    bco: i32,
    tco: i32,
) -> i32 {
    let (Some(mut bcont), Some(mut tcont)) = (bcont, tcont) else {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("Fatal Error: Not enough memory to get new contour.\n"),
        );
        return -1;
    };

    if bcont.pts.len() == 1 {
        let p = bcont.pts[0];
        imod_point_append(&mut bcont, p);
    }
    if tcont.pts.len() == 1 {
        let p = tcont.pts[0];
        imod_point_append(&mut tcont, p);
    }

    if obj.flags & IMOD_OBJFLAG_OUT != 0 {
        inside = 1 - inside;
    }

    let nmesh = imesh_contours_cost(obj, &mut bcont, &mut tcont, scale, inside, bco, tco);
    if let Some(mut nmesh) = nmesh {
        nmesh.surf = surf as i16;
        nmesh.time = time as i16;
        imodel_mesh_add(Some(&nmesh), obj_mesh);
    }
    imod_contour_delete(&mut tcont);
    imod_contour_delete(&mut bcont);
    0
}

/// Original static `break_contour_inout` (`skinobj.c:3303`).
///
/// Breaks a contour into two contours at `st1` and `st2`.
/// `fill = 1` fills in the line dividing the contours up, `fill = -1` fills it
/// down, `fill = 0` does not fill.
pub fn break_contour_inout(
    cin: &Icont,
    st1_in: i32,
    st2_in: i32,
    fill: i32,
    cout1: &mut Option<Icont>,
    cout2: &mut Option<Icont>,
) -> i32 {
    let mut point = Ipoint::default();
    let mut st1 = st1_in;
    let mut st2 = st2_in;
    let mut reversed = 0;

    if st2 < st1 {
        std::mem::swap(&mut st1, &mut st2);
        reversed = 1;
    }

    /* set up new contour and fill point. */
    let Some(mut c1) = imod_contour_new() else {
        return 1;
    };
    let Some(mut c2) = imod_contour_new() else {
        return 1;
    };

    /* transfer an open flag to the c1 contour */
    if cin.flags & ICONT_OPEN != 0 {
        c1.flags |= ICONT_OPEN;
    }

    point.x = (cin.pts[st1 as usize].x + cin.pts[st2 as usize].x) * 0.5f32;
    point.y = (cin.pts[st1 as usize].y + cin.pts[st2 as usize].y) * 0.5f32;
    point.z = (cin.pts[st1 as usize].z + cin.pts[st2 as usize].z) * 0.5f32;
    if fill == 1 {
        point.z += 0.75f32;
    }
    if fill == -1 {
        point.z -= 0.75f32;
    }

    let mut c1_psize = st1 + 1 + cin.pts.len() as i32 - st2;
    let mut c2_psize = st2 + 1 - st1;
    if fill != 0 {
        c1_psize += 1;
        c2_psize += 1;
    }
    c1.pts = vec![Ipoint::default(); c1_psize as usize];
    c2.pts = vec![Ipoint::default(); c2_psize as usize];

    /* add points to new contours */
    let mut pto = 0usize;
    for pt in 0..=st1 as usize {
        c1.pts[pto] = cin.pts[pt];
        pto += 1;
    }
    if fill != 0 {
        c1.pts[pto] = point;
        pto += 1;
    }
    for pt in st2 as usize..cin.pts.len() {
        c1.pts[pto] = cin.pts[pt];
        pto += 1;
    }
    let mut pto = 0usize;
    if fill != 0 {
        c2.pts[pto] = point;
        pto += 1;
    }
    for pt in st1 as usize..=st2 as usize {
        c2.pts[pto] = cin.pts[pt];
        pto += 1;
    }

    /* Transfer properties */
    if istore_extract_changes(&cin.store, &mut c1.store, 0, st1, 0, cin.pts.len() as i32) != 0 {
        return 1;
    }
    if fill != 0
        && istore_extract_changes(
            &cin.store,
            &mut c1.store,
            st1,
            st1,
            st1 + 1,
            cin.pts.len() as i32,
        ) != 0
    {
        return 1;
    }

    if istore_extract_changes(
        &cin.store,
        &mut c1.store,
        st2,
        cin.pts.len() as i32 - 1,
        st1 + if fill != 0 { 2 } else { 1 },
        cin.pts.len() as i32,
    ) != 0
    {
        return 1;
    }

    if fill != 0
        && istore_extract_changes(&cin.store, &mut c2.store, st1, st1, 0, cin.pts.len() as i32) != 0
    {
        return 1;
    }

    if istore_extract_changes(
        &cin.store,
        &mut c2.store,
        st1,
        st2,
        if fill != 0 { 1 } else { 0 },
        cin.pts.len() as i32,
    ) != 0
    {
        return 1;
    }

    if reversed != 0 {
        *cout1 = Some(c2);
        *cout2 = Some(c1);
    } else {
        *cout1 = Some(c1);
        *cout2 = Some(c2);
    }
    0
}
